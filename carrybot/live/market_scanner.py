"""전 코인 시장 스캐너 — 거래량 상위 40개 심볼의 조건 근접도 관측 (표시 전용).

지위: 관측 전용 — 자동 거래 아님, 어떤 트랙 상태·원장·판정 파일에도 쓰지 않는다.
거래는 동결된 18계정(E01~E18) 규칙만 수행하며, 이 산출물은 대시보드 표시 외의
입력(게이트·승급·주문)으로 절대 사용하지 않는다.

유니버스: 봇 관례(src/exchange/bybit_client.fetch_top_symbols 동형) — Bybit
USDT 선형 무기한, 24h 거래대금(quoteVolume) 내림차순 상위 40, $5M 미만 제외.

지표 정의는 carrybot/aggressive/scalp_farm.py 교정판과 동일 계산식이되,
표시용이므로 전부 **최신 확정봉 [i]** 값이다 (게이트의 [i-1] 규약과 시점만 다름):
- 채널(24/96): 확정봉 [i] 를 제외한 직전 N봉 고가 최대 (엔진 shift 채널 관례)
  → 거리% = (상단/종가[i] − 1)×100. 음수 = 이미 돌파.
- RSI14·RSI2: Wilder 평활, 첫 diff 시드 (pandas ewm(alpha=1/n, adjust=False)
  선행 NaN 스킵과 동치). dn==0 은 published_systems 원전 — 상승만 100, 무변동 50.
- BB %b(20, 2σ): 중심 SMA20 · ddof=0 모표준편차 (published_systems 원전).
- SMA200 대비: (종가[i]/SMA200 − 1)×100 (SMA 는 [i] 포함 200봉).
- 거래량 서지: vol[i] / mean(vol[i-20..i-1]) (직전 20봉 평균).
- 3중 게이트 (confluence_gate 구조, [i] 시점 평가): ① 추세 종가 vs SMA200
  (롱 >, 숏 <) ② RSI14 vs 50 (롱 >, 숏 <) ③ 거래량 vol[i] > 직전 20봉 평균
  (롱숏 공통). 미형성·NaN = 미충족 (fail-closed).

실행: python -m carrybot.live.market_scanner (1회형, 재실행 안전 — 산출물 원자 교체만,
다른 어떤 상태·원장 파일도 만들거나 고치지 않는다).
출력: logs/market_scan.json {generated_at_utc, coins: [...], skipped, ...}.
실패 심볼은 스킵·카운트, 전 심볼 실패 시 파일 미갱신 + 종료코드 1 (조용한 초록 금지).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt

logger = logging.getLogger(__name__)

H1 = 3_600_000                  # 1h (ms)
TOP_N = 40                      # 봇 유니버스 규칙 — 거래대금 상위 40
MIN_TURNOVER = 5_000_000.0      # 봇 유니버스 최소 24h 거래대금 규칙과 동일
CANDLE_LIMIT = 220              # 1h 캔들 조회 깊이 (SMA200 + 여유)
GATE_SMA_N = 200                # 게이트 ① 추세 SMA (scalp_farm 동결 상수와 동수)
GATE_RSI_N = 14                 # 게이트 ② Wilder RSI
GATE_VOL_N = 20                 # 게이트 ③ 거래량 평균 창
BB_N = 20                       # BB 밴드 길이 (중심 SMA20)
BB_K = 2.0                      # BB 밴드 폭 (2σ, ddof=0 모표준편차)
RSI2_N = 2                      # Connors RSI 길이
CONSEC_FAIL_MAX = 10            # 연속 실패 이 이상 = 거래소 광역 장애 추정 — 조기 중단
OUT = Path("logs/market_scan.json")


def _retry(fn, *a, **k):
    """공개 엔드포인트 재시도 (6회, 선형 백오프). 실패 시 None (러너 관례)."""
    for i in range(6):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001 — 네트워크 계열 전반 재시도
            if i == 5:
                logger.error("호출 실패: %s %s", type(exc).__name__, str(exc)[:160])
                return None
            time.sleep(1.5 * (i + 1))
    return None


def _f(x) -> float:
    """수치 변환 — 결측/기형은 NaN (fail-closed 비교용)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def top_usdt_perp(tickers: dict, limit: int = TOP_N,
                  min_volume: float = MIN_TURNOVER,
                  quote: str = "USDT") -> list[tuple[str, dict]]:
    """거래량 상위 USDT 선형 무기한 선별 — 봇 유니버스 규칙(fetch_top_symbols) 동형.

    Args:
        tickers: ccxt fetch_tickers 결과 (sym -> 티커 dict).
        limit: 반환 상한.
        min_volume: 최소 24h 거래대금 (quoteVolume) 필터.
        quote: 견적 통화.

    Returns:
        [(symbol, ticker)] — 24h 거래대금 내림차순 (표시 기본 정렬 = 이 순서 고정).
    """
    rows: list[tuple[str, dict, float]] = []
    for sym, t in tickers.items():
        # linear USDT 무기한만: 'BTC/USDT:USDT'
        if not sym.endswith(f":{quote}") or f"/{quote}:" not in sym:
            continue
        vol = _f((t or {}).get("quoteVolume"))
        if not math.isfinite(vol) or vol < min_volume:
            continue
        rows.append((sym, t, vol))
    rows.sort(key=lambda x: -x[2])
    return [(s, t) for s, t, _ in rows[:limit]]


def wilder_rsi(closes: list[float], n: int) -> float | None:
    """마지막 확정봉의 Wilder RSI — 첫 diff 시드 (scalp_farm _update_x2 동형).

    dn==0 정의는 published_systems 원전: 상승만 100, 무변동 50.

    Args:
        closes: 종가 시계열 (오름차순, 확정봉만).
        n: 평활 길이.

    Returns:
        RSI 값. diff 가 하나도 없으면 None.
    """
    u: float | None = None
    dn: float | None = None
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        ux, dx = max(diff, 0.0), max(-diff, 0.0)
        u = ux if u is None else u + (ux - u) / n
        dn = dx if dn is None else dn + (dx - dn) / n
    if u is None or dn is None:
        return None
    if dn > 0:
        return 100.0 - 100.0 / (1.0 + u / dn)
    return 100.0 if u > 0 else 50.0


def compute_metrics(bars: list[tuple]) -> dict | None:
    """확정봉 시퀀스에서 표시 지표를 계산한다 (전부 최신 확정봉 [i] 값).

    Args:
        bars: [(ts, open, high, low, close, vol), ...] 시간 오름차순, 확정봉만.

    Returns:
        지표 dict (미형성 항목은 None, 게이트는 fail-closed False).
        종가가 유효하지 않으면 None (심볼 스킵).
    """
    if not bars:
        return None
    highs = [_f(b[2]) for b in bars]
    closes = [_f(b[4]) for b in bars]
    vols = [_f(b[5]) if len(b) > 5 else float("nan") for b in bars]
    close = closes[-1]
    if not math.isfinite(close) or close <= 0:
        return None

    out: dict = {"bar_ts": int(bars[-1][0]), "close": round(close, 8)}

    def _chan_dist(n: int) -> float | None:
        # 채널 상단 = [i] 제외 직전 n봉 고가 최대 (엔진 shift 채널 관례)
        if len(highs) < n + 1:
            return None
        w = highs[-(n + 1):-1]
        # 창 내 NaN 은 max() 위치에 따라 조용히 무시될 수 있다 — 전량 유효 요구
        if not all(math.isfinite(x) for x in w):
            return None
        hi = max(w)
        if hi <= 0:
            return None
        return round((hi / close - 1.0) * 100.0, 4)

    out["dist24h_pct"] = _chan_dist(24)
    out["dist96h_pct"] = _chan_dist(96)

    rsi14 = wilder_rsi(closes, GATE_RSI_N)
    rsi2 = wilder_rsi(closes, RSI2_N)
    if rsi14 is not None and not math.isfinite(rsi14):
        rsi14 = None                        # NaN 종가 오염 — 차단 (fail-closed)
    if rsi2 is not None and not math.isfinite(rsi2):
        rsi2 = None
    out["rsi14"] = round(rsi14, 2) if rsi14 is not None else None
    out["rsi2"] = round(rsi2, 2) if rsi2 is not None else None

    # BB %b (20, 2σ) — 중심 SMA20 · ddof=0 모표준편차 (published_systems 원전)
    out["bb_pctb"] = None
    if len(closes) >= BB_N:
        w = closes[-BB_N:]
        sma = sum(w) / BB_N
        sd = math.sqrt(sum((x - sma) ** 2 for x in w) / BB_N)
        if math.isfinite(sd) and sd > 0:
            lower = sma - BB_K * sd
            out["bb_pctb"] = round((close - lower) / (2.0 * BB_K * sd), 4)

    # SMA200 대비 위치 (%)
    sma200: float | None = None
    out["sma200_pct"] = None
    if len(closes) >= GATE_SMA_N:
        m = sum(closes[-GATE_SMA_N:]) / GATE_SMA_N
        if math.isfinite(m) and m > 0:
            sma200 = m
            out["sma200_pct"] = round((close / m - 1.0) * 100.0, 4)

    # 거래량 서지 — vol[i] / mean(vol[i-20..i-1])
    v_last: float | None = None
    v_mean: float | None = None
    out["vol_surge"] = None
    if len(vols) >= GATE_VOL_N + 1:
        v_last = vols[-1]
        vm = sum(vols[-(GATE_VOL_N + 1):-1]) / GATE_VOL_N
        if math.isfinite(v_last) and math.isfinite(vm):
            v_mean = vm
            if vm > 0:
                out["vol_surge"] = round(v_last / vm, 4)

    # 3중 게이트 — 미형성·NaN = 미충족 (fail-closed)
    vol_ok = v_last is not None and v_mean is not None and v_last > v_mean
    trend_long = sma200 is not None and close > sma200
    trend_short = sma200 is not None and close < sma200
    rsi_long = rsi14 is not None and rsi14 > 50.0
    rsi_short = rsi14 is not None and rsi14 < 50.0
    out["gate_long"] = bool(trend_long and rsi_long and vol_ok)
    out["gate_short"] = bool(trend_short and rsi_short and vol_ok)
    out["gate_parts"] = {
        "trend_long": bool(trend_long), "trend_short": bool(trend_short),
        "rsi_long": bool(rsi_long), "rsi_short": bool(rsi_short),
        "vol": bool(vol_ok),
    }
    return out


def fetch_closed_1h(ex, sym: str, now_h: int) -> list[tuple] | None:
    """최근 1h 확정봉 시퀀스 수집 — 미확정 봉 제외·중복 제거·연속 꼬리만.

    검증 (fail-closed — 위반 시 None 으로 심볼 스킵):
    - 최신 확정봉(now_h − 1h)이 응답에 반드시 있어야 한다 — 오래된 응답을
      신선한 스캔으로 오인 금지 (generated_at 만 새 값이 되는 사고 차단).
    - 최신 봉에서 거꾸로 1h 간격으로 이어지는 연속 꼬리만 사용한다 — 갭 낀
      이력으로 24/96 '시간' 창이 24/96 '행' 창으로 변질되는 것 방지.

    Args:
        ex: ccxt bybit 인스턴스 (enableRateLimit 관례).
        sym: 통합 심볼 (예: 'BTC/USDT:USDT').
        now_h: 현재 시각의 1h 내림 정렬 ts (이 시각 이후 봉 = 미확정).
    """
    rs = _retry(ex.fetch_ohlcv, sym, "1h", limit=CANDLE_LIMIT)
    if rs is None:
        return None
    by_ts: dict[int, tuple] = {}
    for r in rs:
        try:
            ts = int(r[0])
            row = (ts, _f(r[1]), _f(r[2]), _f(r[3]), _f(r[4]),
                   _f(r[5]) if len(r) > 5 else float("nan"))
        except (TypeError, ValueError, IndexError):
            continue                        # 기형 행 — 행 단위 스킵
        if ts >= now_h:
            continue                        # 미확정 봉 제외 (확정봉 기준 명시)
        by_ts[ts] = row                     # 중복 ts 는 마지막 값
    last = now_h - H1
    if last not in by_ts:
        logger.warning("%s 최신 확정봉(%d) 결측 — 오래된 응답 스킵", sym, last)
        return None
    out: list[tuple] = []
    t = last
    while t in by_ts:                       # 연속 꼬리만 (러너 contiguous 관례)
        out.append(by_ts[t])
        t -= H1
    out.reverse()
    return out


def run_scan(ex, now_ms: int | None = None) -> dict:
    """유니버스 선정 → 심볼별 지표 계산 → 저장용 페이로드 조립.

    Args:
        ex: ccxt bybit 인스턴스.
        now_ms: 현재 시각 epoch ms (기본 실시간, 테스트 주입용).

    Returns:
        {generated_at_utc, universe, coins, skipped, skipped_symbols}.

    Raises:
        RuntimeError: 티커 조회 실패 (유니버스 선정 불가 — fail-closed).
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    now_h = now_ms - now_ms % H1
    tickers = _retry(ex.fetch_tickers)
    if tickers is None:
        raise RuntimeError("티커 조회 실패 — 유니버스 선정 불가")

    coins: list[dict] = []
    skipped: list[str] = []
    consec_fail = 0
    universe = top_usdt_perp(tickers)
    for k, (sym, t) in enumerate(universe):
        try:
            bars = fetch_closed_1h(ex, sym, now_h)
            m = compute_metrics(bars) if bars is not None else None
        except Exception as exc:  # noqa: BLE001 — 심볼 단위 격리 (전체 중단 금지)
            logger.warning("%s 계산 예외 — 스킵: %s %s", sym,
                           type(exc).__name__, str(exc)[:120])
            m = None
        if m is None:
            skipped.append(sym)
            consec_fail += 1
            logger.warning("%s 스킵 — 캔들 수집/계산 실패", sym)
            if consec_fail >= CONSEC_FAIL_MAX:
                # 거래소 광역 장애 추정 — 남은 심볼 재시도 낭비(심볼당 최대
                # 22.5s 백오프) 방지, 잔여는 전부 스킵으로 계상
                rest = [s for s, _ in universe[k + 1:]]
                skipped.extend(rest)
                logger.error("연속 실패 %d회 — 조기 중단 (잔여 %d심볼 스킵)",
                             consec_fail, len(rest))
                break
            continue
        consec_fail = 0
        last = _f((t or {}).get("last"))
        chg = _f((t or {}).get("percentage"))
        turn = _f((t or {}).get("quoteVolume"))
        row = {
            "symbol": sym,
            "coin": sym.split("/")[0],
            "price": round(last, 8) if math.isfinite(last) else m["close"],
            "chg24h_pct": round(chg, 4) if math.isfinite(chg) else None,
            "turnover24h": round(turn, 2) if math.isfinite(turn) else None,
        }
        row.update(m)
        coins.append(row)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": {"top_n": TOP_N, "min_turnover_usdt": MIN_TURNOVER,
                     "timeframe": "1h", "candle_limit": CANDLE_LIMIT},
        "coins": coins,                     # 24h 거래대금 내림차순 (고정 정렬)
        "skipped": len(skipped),
        "skipped_symbols": skipped,
    }


def save_atomic(payload: dict, path: Path | None = None) -> None:
    """임시파일→rename 원자적 저장 (러너 _atomic_write 관례).

    임시파일명에 PID 를 붙여 동시 실행 간 절단 사고를 막고,
    allow_nan=False 로 비표준 NaN 직렬화(파서별 소비 실패)를 차단한다.

    Args:
        payload: 저장할 페이로드.
        path: 산출물 경로 (기본 모듈 상수 OUT — 테스트 주입용).

    Raises:
        ValueError: 페이로드에 NaN/Inf 잔존 (fail-closed — 파일 미갱신).
    """
    path = path if path is not None else OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=False),
                       encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)         # 직렬화 실패 잔재 정리
        raise


def main() -> int:
    """1회형 스캔 — 성공 시 logs/market_scan.json 원자 교체, 실패 시 미갱신.

    Returns:
        종료코드 (0 정상 / 1 실패 — 전 심볼 실패 포함, 조용한 초록 금지).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.info("전 코인 스캐너 — 관측 전용 (자동 거래 아님, "
                "거래는 동결된 18계정 규칙만 수행)")
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    if _retry(ex.load_markets) is None:
        logger.error("마켓 로드 실패 — 파일 미갱신 (fail-closed)")
        return 1
    try:
        payload = run_scan(ex)
    except RuntimeError as exc:
        logger.error("%s — 파일 미갱신 (fail-closed)", exc)
        return 1
    if not payload["coins"]:
        logger.error("전 심볼 실패 (스킵 %d) — 파일 미갱신 (fail-closed)",
                     payload["skipped"])
        return 1
    try:
        save_atomic(payload)
    except ValueError as exc:
        logger.error("직렬화 실패(NaN/Inf 잔존?) — 파일 미갱신: %s", exc)
        return 1
    logger.info("스캔 완료 — %d코인 저장, 스킵 %d (%s)", len(payload["coins"]),
                payload["skipped"], OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
