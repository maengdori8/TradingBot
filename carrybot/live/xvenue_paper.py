from __future__ import annotations

"""Track C 페이퍼 러너 — 교차 거래소 펀딩 차익 (HL 숏 / Bybit 롱, 고정 방향).

사양: docs/XVENUE_ARBITRAGE_2026-08.md 사전 등록 + 2026-08-29 정정 공시.
페이퍼 전용 — 실주문 없음.

적재하는 것 (실제 구현과 1:1 대응. 구현하지 않은 것을 적지 않는다):
- `day_diff`  : 그 UTC 일자의 실현 펀딩 손익 (HL 숏 수취 − Bybit 롱 지불).
- `basis_diff`: 그 UTC 일자의 **일봉 종가 베이시스 MTM** —
                `n_b·(B_t − B_{t−1}) − n_h·(H_t − H_{t−1})`, 고정 수량.
- `notional_base`: 그 유니버스가 활성화될 때 고정된 명목 기준 (다리당).
                강제청산 파생본은 생존 수량을 그대로 두므로 기준을 유지한다.
- `cost`      : 그날 부과된 진입/재구성/청산 비용의 **절대액** (인덱스 단위).
- `equity`    : 총합 인덱스 = 진입/재구성 비용 차감 + Σ(펀딩+베이시스).
- `equity_funding`: 같은 비용·명목 기준의 **펀딩만** 진단 계열 (구 사양 비교용).

명시 규약 (사전등록의 빈칸을 메운 것 — 문서 addendum 확정 대상):
1. 유니버스 = 양쪽 상장 + **양쪽 30일 중앙 일거래대금 ≥ $5M**, 월 1회 재평가.
   as-of 는 "그 유니버스가 적용되는 첫 UTC 일자 00:00 직전 닫힌 30일"이다
   (룩어헤드 없음). Bybit 은 일봉 `turnover`(실측 USDT), HL 은 일봉
   `v × (h+l+c)/3` **추정** 명목 — 추정치임을 명시하고 v×low / v×high 경계도
   함께 적재해 $5M 경계 종목의 estimator 의존성을 감사 가능하게 둔다.
2. 다리 정렬 = 리밸런스 시점 **다리별 등달러 명목** (코인당 1/N). 수량은
   리밸런스 사이 고정 — 일별 등명목 재설정이 아니다. 따라서 일별 손익은
   그 명목 기준에 대한 **가산**이며 일별 복리가 아니다.
3. 비용 = 12bp × (Σ|Δw_B| + Σ|Δw_H|)/2. 직전 비중은 고정 수량을 전환일
   종가로 평가한 실측치이므로 생존 종목의 비중 드리프트와 총명목 드리프트가
   모두 과금된다. 최초 진입은 Σ|Δw|=1 → 12bp.
4. 결측 종목 동적 재가중 금지: 그날 전 종목이 완전할 때만 적재하고,
   불완전하면 그날을 보류한 뒤 이후 실행에서 **시간순 백필**한다. 결측을
   0 으로 대체하지 않는다 (손실을 숨기는 경로).
5. 상장폐지는 유일한 예외다 — 베뉴 목록에서 사라진 것이 확인되면 마지막
   권위 종가로 **강제 청산**(12bp×비중)하고 유니버스에서 뺀다.
6. ROE = 인덱스 수익률 ÷ 2 (담보 이중 소요). 판정 기준(전향 90일 ROE > 현금)은
   이 파일이 바꾸지 않는다.

측정 한계 (숨기지 않는다):
- 판정 binding 이 `equity`(펀딩+베이시스)인지 `equity_funding`(펀딩만)인지는
  **문서 addendum 사안**이다. 두 계열을 모두 적재하므로 사후 재계산은 불필요하다.
- 베이시스는 **거래 일봉 종가**다 (사전등록 문구 "양쪽 일봉 종가"). 각 베뉴의
  마크 프라이스 원장이 아니며, 펀딩 손익도 정산 시점 마크가 아니라 그날 종가를
  명목 대용으로 쓴다.
- 재구성 비용의 신규 비중은 새 명목 기준, 직전 비중은 직전 명목 기준으로 재므로
  둘의 비(base_new/base_old)만큼 오차가 있다 (월 <1% → 12bp 과금의 2차항).
- 청산·예비담보 회계 없음 (정정 공시 C1) → 이 인덱스는 상한이다.
- 실행 슬리피지·부분체결·브리지 정지 없음.
"""

import hashlib
import json
import logging
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STATE = Path("logs/trackc_state.json")
HIST = Path("logs/trackc_history.csv")
UNIVERSE_F = Path("logs/trackc_universe.json")

COST_RT = 0.0012                     # 왕복 12bp/코인 (사전등록)
MIN_MEDIAN_NTL = 5e6                 # 양쪽 30일 중앙 일거래대금 하한 ($5M)
MEDIAN_WINDOW = 30                   # 중앙값 창 (닫힌 UTC 일자 수)
UNIVERSE_RULE = "median30d_ntl_5m_v2"
DAY = 86_400_000

HL_SLEEP = 1.0                       # HL info 는 weight 20 (1200/min) → 60req/min
BY_SLEEP = 0.15
DEADLINE_SEC = 600.0                 # 워크플로 timeout 15분 대비 (실측 빌드 ~2분)
MAX_BACKFILL_DAYS = 4                # 실행 1회당 백필 상한

CSV_COLS = ["day", "row_type", "phase", "universe_id", "n_coins", "day_diff",
            "basis_diff", "cost", "notional_base", "equity", "equity_funding"]

# HL 의 k 접두(=1000배 계약)는 일반 규칙으로 변환하면 안 된다 — Bybit 표기가
# 제각각이다(1000PEPE / SHIB1000). 명시 매핑만 허용하고 존재 여부는 Bybit
# instruments-info 로 확인한다. 펀딩률은 계약 승수와 무관하고, 가격·수량은 각
# 베뉴 표기 그대로 짝지어 쓰므로 승수 차이는 회계에 영향이 없다.
K_SYMBOL_MAP: dict[str, str] = {
    "kPEPE": "1000PEPEUSDT",
    "kSHIB": "SHIB1000USDT",
    "kBONK": "1000BONKUSDT",
    "kLUNC": "1000LUNCUSDT",
    "kFLOKI": "1000FLOKIUSDT",
}
# 매핑하면 **다른 자산**과 짝지어지는 코인 — 조용히 떨어뜨리지 않고 명시 제외.
EXCLUDED_COINS: dict[str, str] = {
    "kNEIRO": "Bybit 에는 1000NEIROCTOUSDT(다른 토큰)만 있어 짝지을 수 없음",
}


class LedgerError(RuntimeError):
    """원장 불변식 위반 — 조용히 넘어가면 안 되는 오류 (비정상 종료)."""


# ── HTTP (전송 실패 = None = unknown, 정상 응답의 빈 결과와 구별) ──────────

def _post_hl(body: dict, retries: int = 4):
    """HL info 호출. 전송·파싱 실패는 None (unknown) 으로 돌려준다."""
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.debug("HL 재시도 %d: %s", i, e)
            time.sleep(1.5 * (i + 1))
    return None


def _get_bybit(url: str, retries: int = 4):
    """Bybit 공개 호출. 전송 실패·retCode≠0 은 None (unknown)."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                d = json.loads(r.read())
            if d.get("retCode") not in (0, None):
                logger.debug("Bybit retCode=%s", d.get("retCode"))
                time.sleep(1.5 * (i + 1))
                continue
            return d
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.debug("Bybit 재시도 %d: %s", i, e)
            time.sleep(1.5 * (i + 1))
    return None


def _now_utc() -> pd.Timestamp:
    """현재 UTC 시각 (테스트 주입점)."""
    return pd.Timestamp.now(tz="utc")


def _atomic_write(path: Path, text: str) -> None:
    """임시 파일 → os.replace 로 원자 저장 (중간 크래시에 잘린 파일 금지)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ── 심볼 매핑 ────────────────────────────────────────────────────────────

def bybit_symbol(coin: str, available: set[str]) -> str | None:
    """HL 코인명 → Bybit linear USDT 심볼. 짝지을 수 없으면 None."""
    if coin in EXCLUDED_COINS:
        return None
    if coin in K_SYMBOL_MAP:
        sym = K_SYMBOL_MAP[coin]
        return sym if sym in available else None
    if coin.startswith("k") and coin[1:].isupper():
        # 미등록 k 코인 — 추측 매핑은 다른 자산과 짝지을 위험이 있어 금지.
        logger.warning("k접두 미등록 코인 제외: %s (K_SYMBOL_MAP 에 명시 필요)", coin)
        return None
    sym = f"{coin}USDT"
    return sym if sym in available else None


# ── 원자료 수집 ──────────────────────────────────────────────────────────

def hl_perp_names() -> list[str] | None:
    """HL 상장 무기한 코인명 (delisted 제외). 실패 시 None."""
    meta = _post_hl({"type": "metaAndAssetCtxs"})
    if not meta or not isinstance(meta, list) or not meta[0].get("universe"):
        return None
    return [a["name"] for a in meta[0]["universe"] if not a.get("isDelisted")]


def bybit_linear_usdt() -> set[str] | None:
    """Bybit linear USDT 거래중 심볼 집합. 실패·페이지 미완주는 None."""
    out: set[str] = set()
    cursor = ""
    for _ in range(20):
        url = ("https://api.bybit.com/v5/market/instruments-info"
               f"?category=linear&limit=1000{'&cursor=' + cursor if cursor else ''}")
        d = _get_bybit(url)
        if d is None:
            return None
        res = d.get("result") or {}
        for it in res.get("list", []):
            if (it.get("quoteCoin") == "USDT" and it.get("status") == "Trading"
                    and "-" not in it.get("symbol", "-")):
                out.add(it["symbol"])
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            return out or None
        time.sleep(BY_SLEEP)
    logger.error("Bybit instruments 페이지 미완주 — 부분 목록을 쓰지 않는다")
    return None                       # 부분 목록으로 "미상장" 판정하면 안 된다


def bybit_daily(symbol: str, start_ms: int, end_ms: int) -> dict[int, dict] | None:
    """Bybit 일봉 {일자ms: {c, h, l, turnover}}. 전송 실패는 None (unknown)."""
    d = _get_bybit("https://api.bybit.com/v5/market/kline?category=linear"
                   f"&symbol={symbol}&interval=D&start={start_ms}&end={end_ms - 1}"
                   f"&limit=1000")
    if d is None:
        return None
    out: dict[int, dict] = {}
    for row in (d.get("result") or {}).get("list", []):
        t = int(row[0])
        if start_ms <= t < end_ms:
            out[t] = dict(c=float(row[4]), turnover=float(row[6]),
                          h=float(row[2]), l=float(row[3]))
    return out


def hl_daily(coin: str, start_ms: int, end_ms: int) -> dict[int, dict] | None:
    """HL 일봉 {일자ms: {c, h, l, v}}. 전송 실패는 None (unknown)."""
    d = _post_hl({"type": "candleSnapshot", "req": {
        "coin": coin, "interval": "1d", "startTime": start_ms, "endTime": end_ms - 1}})
    if d is None or not isinstance(d, list):
        return None
    out: dict[int, dict] = {}
    for row in d:
        t = int(row["t"])
        if start_ms <= t < end_ms:
            out[t] = dict(c=float(row["c"]), h=float(row["h"]),
                          l=float(row["l"]), v=float(row["v"]))
    return out


def hl_notional(candle: dict) -> tuple[float, float, float]:
    """HL 일봉 → 추정 명목 (하한 v×low, 점추정 v×(h+l+c)/3, 상한 v×high).

    HL 일봉은 기초자산 수량(v)만 주므로 실측 turnover 가 아니다 — 이 삼중값을
    유니버스 파일에 적재해 $5M 경계 종목의 estimator 의존성을 드러낸다.
    """
    v = candle["v"]
    typ = (candle["h"] + candle["l"] + candle["c"]) / 3.0
    return v * candle["l"], v * typ, v * candle["h"]


def bybit_funding(symbol: str, t0: int, t1: int) -> list[float] | None:
    """Bybit 실현 펀딩률 목록 ([t0, t1)). 전송 실패는 None (unknown)."""
    d = _get_bybit("https://api.bybit.com/v5/market/funding/history?category=linear"
                   f"&symbol={symbol}&startTime={t0}&endTime={t1 - 1}&limit=200")
    if d is None:
        return None
    return [float(x["fundingRate"]) for x in (d.get("result") or {}).get("list", [])
            if t0 <= int(x["fundingRateTimestamp"]) < t1]


def hl_funding(coin: str, t0: int, t1: int) -> list[float] | None:
    """HL 실현 펀딩률 목록 ([t0, t1)). 전송 실패는 None (unknown)."""
    d = _post_hl({"type": "fundingHistory", "coin": coin,
                  "startTime": t0, "endTime": t1 - 1})
    if d is None or not isinstance(d, list):
        return None
    return [float(x["fundingRate"]) for x in d if t0 <= int(x["time"]) < t1]


# ── 유니버스 구축 (사전등록 게이트: 양쪽 30일 중앙 일거래대금 ≥ $5M) ────────

def _median_or_none(vals: list[float]) -> float | None:
    """30개 완전일 때만 중앙값 — 부족하면 None (이력 부족 → 부적격)."""
    return statistics.median(vals) if len(vals) == MEDIAN_WINDOW else None


def _snapshot_id(snap: dict) -> str:
    """스냅샷 정체성 해시 — 게이트 증거·**수량 원장**·비용 입력까지 포함한다.

    코인 목록만으로 해시하면 거래소가 과거 종가를 정정했을 때 같은 id 에 다른
    수량 원장이 실려 이미 부과된 비용이 은폐된다. 전환비용의 멱등성이 id 에
    묶여 있으므로 prev_id·gross_traded 도 정체성의 일부다.
    """
    payload = json.dumps(dict(
        rule=snap["rule"], as_of=snap["as_of"], month=snap.get("month"),
        estimator=snap["estimator"], coins=snap["coins"],
        exited=snap.get("exited", []), prev_id=snap.get("prev_id"),
        gross=snap.get("gross_traded"), keep_base=bool(snap.get("keep_base")),
        exit_pnl=snap.get("exit_pnl"),
        medians={c: [m.get("by"), m.get("hl")] for c, m in snap["medians"].items()},
        positions={c: [p["b_ref"], p["h_ref"], p["n_b"], p["n_h"]]
                   for c, p in snap["positions"].items()},
    ), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _transition_cost(prev_snap: dict | None, coins: dict[str, str],
                     by_close: dict[str, float], hl_close: dict[str, float],
                     scale: float = 1.0) -> tuple[float, dict]:
    """(Σ|Δw_B| + Σ|Δw_H|)/2 와 그 내역 — 양다리 모두 계상.

    직전 비중은 고정 수량 원장을 **전환일 실측 종가**(상장폐지 종목은 마지막
    체결 종가)로 평가하며 정규화하지 않는다: 정규화하면 총명목 드리프트(전
    종목 동반 상승 등)를 놓친다. Bybit 다리만으로 재면 HL 쪽 발산이 비용에서
    사라지므로 두 다리를 평균한다. 마크가 하나라도 없으면 계산하지 않는다
    (진입가 대체는 비용을 조용히 틀리게 만든다).

    Args:
        scale: 직전 명목 기준 ÷ 전환 시점 자본. 기존 노출은 직전 notional_base
            단위, 신규 비중은 새 기준(= 현재 자본) 단위라서 환산해야 한다 —
            강제청산 후 현금 비중이나 누적 손익으로 둘이 벌어질 수 있다.
    """
    n = len(coins)
    w_new = {c: 1.0 / n for c in coins} if n else {}
    u_b: dict[str, float] = {}
    u_h: dict[str, float] = {}
    for c, p in ((prev_snap or {}).get("positions") or {}).items():
        if c not in by_close or c not in hl_close:
            raise LedgerError(f"전환 마크 없음: {c} — 비용 계산 불가")
        u_b[c] = p["n_b"] * by_close[c] * scale
        u_h[c] = p["n_h"] * hl_close[c] * scale
    keys = set(w_new) | set(u_b)
    gross_b = sum(abs(w_new.get(c, 0.0) - u_b.get(c, 0.0)) for c in keys)
    gross_h = sum(abs(w_new.get(c, 0.0) - u_h.get(c, 0.0)) for c in keys)
    gross = (gross_b + gross_h) / 2.0
    detail = dict(
        added=sorted(set(w_new) - set(u_b)), removed=sorted(set(u_b) - set(w_new)),
        survived=len(set(w_new) & set(u_b)), n_prev=len(u_b), n_new=n,
        gross_b=round(gross_b, 8), gross_h=round(gross_h, 8),
        scale=round(scale, 10), cost_bp=round(gross * COST_RT * 1e4, 4))
    return gross, detail


def last_closes(coin: str, sym: str, end_ms: int, lookback: int = 30
                ) -> tuple[float, float] | None:
    """end_ms 직전 **마지막 체결 종가** (Bybit, HL). 구할 수 없으면 None.

    상장폐지·거래정지 종목의 청산 마크용 — 전송 실패와 "그 창에 체결 없음"을
    구별하지 않고 둘 다 None 으로 돌려 호출자가 보류하게 한다.
    """
    b = bybit_daily(sym, end_ms - lookback * DAY, end_ms)
    time.sleep(BY_SLEEP)
    h = hl_daily(coin, end_ms - lookback * DAY, end_ms)
    time.sleep(HL_SLEEP)
    if not b or not h:
        return None
    return b[max(b)]["c"], h[max(h)]["c"]


def build_universe(as_of: pd.Timestamp, deadline: float,
                   prev_snap: dict | None = None, scale: float = 1.0
                   ) -> tuple[dict | None, list[str]]:
    """as_of 00:00 UTC 직전 닫힌 30일로 유니버스를 만든다 (룩어헤드 없음).

    Args:
        as_of: 이 유니버스가 적용되는 첫 UTC 일자 (00:00 기준).
        deadline: time.monotonic() 기준 마감 — 넘으면 중단(unknown 취급).
        prev_snap: 직전 유니버스 스냅샷 (전환비용 Σ|Δw| 계산용).
        scale: 직전 명목 기준 ÷ 전환 시점 자본 (`_transition_cost` 참조).

    Returns:
        (스냅샷, unknown 목록). 후보 중 하나라도 unknown 이면 (None, [...]) —
        전송 실패를 "부적격"으로 강등하지 않는다 (fail-closed).
    """
    end = int(as_of.timestamp() * 1000)
    start = end - MEDIAN_WINDOW * DAY
    names = hl_perp_names()
    if names is None:
        return None, ["<hl_meta>"]
    available = bybit_linear_usdt()
    if available is None:
        return None, ["<bybit_instruments>"]

    cands: list[tuple[str, str]] = []
    for coin in sorted(names):
        sym = bybit_symbol(coin, available)
        if sym:
            cands.append((coin, sym))

    unknown: list[str] = []
    medians: dict[str, dict] = {}
    by_ok: list[tuple[str, str]] = []
    by_close: dict[str, float] = {}
    hl_close: dict[str, float] = {}
    for coin, sym in cands:
        if time.monotonic() > deadline:
            return None, ["<deadline>"]
        rows = bybit_daily(sym, start, end)
        time.sleep(BY_SLEEP)
        if rows is None:
            unknown.append(coin)
            continue
        med = _median_or_none([r["turnover"] for r in rows.values()])
        medians[coin] = dict(sym=sym, by=med, by_days=len(rows))
        if end - DAY in rows:
            by_close[coin] = rows[end - DAY]["c"]
        # AND 게이트라 Bybit 미달이면 HL 조회 없이 확정 탈락 (근사 아님).
        if med is not None and med >= MIN_MEDIAN_NTL:
            by_ok.append((coin, sym))
    if unknown:
        return None, unknown

    coins: dict[str, str] = {}
    for coin, sym in by_ok:
        if time.monotonic() > deadline:
            return None, ["<deadline>"]
        rows = hl_daily(coin, start, end)
        time.sleep(HL_SLEEP)
        if rows is None:
            unknown.append(coin)
            continue
        bounds = [hl_notional(r) for r in rows.values()]
        lo = _median_or_none([b[0] for b in bounds])
        mid = _median_or_none([b[1] for b in bounds])
        hi = _median_or_none([b[2] for b in bounds])
        medians[coin].update(hl_low=lo, hl=mid, hl_high=hi, hl_days=len(rows))
        if end - DAY in rows:
            hl_close[coin] = rows[end - DAY]["c"]
        if mid is not None and mid >= MIN_MEDIAN_NTL:
            if coin in by_close and coin in hl_close:
                coins[coin] = sym
            else:
                unknown.append(coin)          # 게이트는 통과했는데 기준 마크 없음
    if unknown:
        return None, unknown
    if not coins:
        return None, ["<empty_universe>"]

    # 전환비용은 **직전 보유 종목 전부**의 전환 마크가 있어야 정확하다.
    # 상장폐지로 전환일 캔들이 없으면 마지막 체결 종가를 쓴다 (진입가 대체 금지).
    for coin in ((prev_snap or {}).get("positions") or {}):
        if coin in by_close and coin in hl_close:
            continue
        if time.monotonic() > deadline:
            return None, ["<deadline>"]
        sym = (prev_snap.get("coins") or {}).get(coin, f"{coin}USDT")
        marks = last_closes(coin, sym, end)
        if marks is None:
            unknown.append(coin)              # 전송 실패·마크 부재 → 보류
            continue
        by_close.setdefault(coin, marks[0])
        hl_close.setdefault(coin, marks[1])
    if unknown:
        return None, unknown

    w = 1.0 / len(coins)
    positions = {c: dict(w=w, b_ref=by_close[c], h_ref=hl_close[c],
                         n_b=w / by_close[c], n_h=w / hl_close[c])
                 for c in coins}
    snap = dict(
        rule=UNIVERSE_RULE, as_of=str(as_of.date()),
        month=f"{as_of.year}-{as_of.month:02d}",
        estimator="bybit_turnover / hl_v*(h+l+c)/3",
        coins=coins, positions=positions, medians=medians,
        candidates=len(cands), prev_id=(prev_snap or {}).get("id"), exited=[])
    gross, detail = _transition_cost(prev_snap, coins, by_close, hl_close, scale)
    snap["gross_traded"] = round(gross, 10)
    snap["transition"] = detail
    snap["id"] = _snapshot_id(snap)
    return snap, []


def delist_exit(snap: dict, gone: list[str], day: pd.Timestamp
                ) -> tuple[dict | None, list[str]]:
    """폐지 종목의 청산 마크·직전 종가·당일 펀딩으로 **최종 손익**을 확정한다.

    직전 회계 종가에서 청산가까지의 베이시스 변동을 적재하지 않으면 마지막
    구간 손익이 통째로 사라진다. 조회가 하나라도 실패하면 (None, [코인]).
    """
    t0 = int(day.timestamp() * 1000)
    t1 = t0 + DAY
    marks: dict[str, tuple[float, float]] = {}
    funding = basis = 0.0
    for c in gone:
        sym = snap["coins"][c]
        m = last_closes(c, sym, t1)
        if m is None:
            return None, [c]
        by = bybit_daily(sym, t0 - DAY, t0)
        time.sleep(BY_SLEEP)
        hl = hl_daily(c, t0 - DAY, t0)
        time.sleep(HL_SLEEP)
        bf = bybit_funding(sym, t0, t1)
        time.sleep(BY_SLEEP)
        hf = hl_funding(c, t0, t1)
        time.sleep(HL_SLEEP)
        if by is None or hl is None or bf is None or hf is None:
            return None, [c]
        p = snap["positions"][c]
        # 직전 회계 기준가 = 전일 종가. 유니버스 첫날에만 리밸런스 기준가(=그
        # 전일 종가)로 대체할 수 있다 — 그 이후에 대체하면 이미 적재한 베이시스
        # 손익을 다시 세게 된다.
        first = str(day.date()) == snap["as_of"]
        if not first and (t0 - DAY not in by or t0 - DAY not in hl):
            return None, [c]
        b_prev = by.get(t0 - DAY, {}).get("c", p["b_ref"])
        h_prev = hl.get(t0 - DAY, {}).get("c", p["h_ref"])
        marks[c] = m
        basis += p["n_b"] * (m[0] - b_prev) - p["n_h"] * (m[1] - h_prev)
        funding += p["n_h"] * m[1] * sum(hf) - p["n_b"] * m[0] * sum(bf)
    return dict(marks=marks, funding=funding, basis=basis), []


def exit_snapshot(snap: dict, gone: list[str],
                  exit_marks: dict[str, tuple[float, float]],
                  exit_pnl: dict | None = None) -> dict:
    """상장폐지 종목을 **마지막 체결 종가**로 강제 청산한 파생 스냅샷.

    남은 종목의 수량은 건드리지 않는다 (부분 청산이지 재구성이 아니다) →
    `keep_base=True`: 명목 기준을 재설정하지 않으므로 생존 포지션이 비용 없이
    커지지 않는다. 청산분은 다음 월 재구성까지 현금으로 남는다 (손익 0).
    전 종목이 폐지되면 빈 스냅샷(전액 현금)이 되며 이 또한 유효한 상태다.
    """
    survivors = {c: s for c, s in snap["coins"].items() if c not in gone}
    positions = {c: snap["positions"][c] for c in survivors}
    gross = 0.0
    for c in gone:
        p = snap["positions"][c]
        b, h = exit_marks[c]
        gross += (p["n_b"] * b + p["n_h"] * h) / 2      # 청산 시점 실제 명목
    out = dict(snap, coins=survivors, positions=positions, keep_base=True,
               exited=sorted(set(snap.get("exited", [])) | set(gone)),
               prev_id=snap["id"], derived_from=snap["id"],
               exit_marks={c: list(exit_marks[c]) for c in gone},
               exit_pnl=dict(funding=round((exit_pnl or {}).get("funding", 0.0), 12),
                             basis=round((exit_pnl or {}).get("basis", 0.0), 12)),
               gross_traded=round(gross, 10),
               transition=dict(added=[], removed=sorted(gone), survived=len(survivors),
                               n_prev=len(snap["coins"]), n_new=len(survivors),
                               reason="delisted_forced_exit",
                               cost_bp=round(gross * COST_RT * 1e4, 4)))
    out["id"] = _snapshot_id(out)
    return out


def find_delisted(snap: dict) -> tuple[list[str], bool]:
    """스냅샷 종목 중 베뉴 목록에서 사라진 것. (목록, 조회실패여부)."""
    names = hl_perp_names()
    available = bybit_linear_usdt()
    if names is None or available is None:
        return [], True
    live = set(names)
    return sorted(c for c, sym in snap["coins"].items()
                  if c not in live or sym not in available), False


# ── 일별 손익 (고정 수량, 결측 대체 금지) ────────────────────────────────

def day_pnl(snap: dict, day: pd.Timestamp, deadline: float
            ) -> tuple[dict | None, list[str]]:
    """그 UTC 일자의 (펀딩, 베이시스) 손익. 불완전하면 (None, unknown목록).

    명목 1(다리당)에 대한 **가산** 손익이다 — 호출자가 그 유니버스의
    notional_base 를 곱한다. 결측은 0 으로 대체하지 않고 그날을 보류한다.

    Args:
        snap: 그날 적용되는 유니버스 스냅샷 (고정 수량 원장 포함).
        day: 대상 UTC 일자 (00:00 정규화).
        deadline: time.monotonic() 기준 마감 — 넘으면 그날을 보류한다.

    Returns:
        ({funding, basis}, []) 또는 (None, unknown 코인 목록).
    """
    t0 = int(day.timestamp() * 1000)
    t1 = t0 + DAY
    funding = basis = 0.0
    for coin, sym in snap["coins"].items():
        if time.monotonic() > deadline:
            return None, ["<deadline>"]
        by = bybit_daily(sym, t0 - DAY, t1)
        time.sleep(BY_SLEEP)
        if by is None or t0 not in by or t0 - DAY not in by:
            return None, [coin]                       # 첫 결측에서 단락 (시간 절약)
        hl = hl_daily(coin, t0 - DAY, t1)
        time.sleep(HL_SLEEP)
        if hl is None or t0 not in hl or t0 - DAY not in hl:
            return None, [coin]
        bf = bybit_funding(sym, t0, t1)
        time.sleep(BY_SLEEP)
        if not bf:
            return None, [coin]
        hf = hl_funding(coin, t0, t1)
        time.sleep(HL_SLEEP)
        if not hf:
            return None, [coin]
        p = snap["positions"][coin]
        b_cur, b_prev = by[t0]["c"], by[t0 - DAY]["c"]
        h_cur, h_prev = hl[t0]["c"], hl[t0 - DAY]["c"]
        basis += p["n_b"] * (b_cur - b_prev) - p["n_h"] * (h_cur - h_prev)
        # HL 숏이 받고(+), Bybit 롱이 낸다(−). 명목은 그날 종가 × 고정 수량.
        funding += p["n_h"] * h_cur * sum(hf) - p["n_b"] * b_cur * sum(bf)
    return dict(funding=funding, basis=basis), []


# ── 원장 (CSV 가 회계 단일 진실원, state 는 파생·진단) ────────────────────

def load_rows() -> list[dict]:
    """이력 CSV 를 dict 목록으로 읽는다 (없으면 빈 목록)."""
    if not HIST.exists():
        return []
    df = pd.read_csv(HIST, dtype={"day": str, "universe_id": str})
    return df.fillna("").to_dict("records")


def write_rows(rows: list[dict]) -> None:
    """이력 CSV 전체를 원자 저장한다 (append 중 잘림 방지)."""
    _atomic_write(HIST, pd.DataFrame(rows, columns=CSV_COLS).to_csv(index=False))


def last_accounting(rows: list[dict]) -> dict | None:
    """마지막 회계 행 (물리 마지막 행이 아니라 row_type 으로 판정)."""
    for r in reversed(rows):
        if r.get("row_type") in ("anchor", "daily"):
            return r
    return None


def audit_rows(rows: list[dict]) -> None:
    """원장 불변식 검사 — 위반은 LedgerError (조용한 진행 금지)."""
    days = [r["day"] for r in rows if r.get("row_type") == "daily"]
    if len(days) != len(set(days)):
        raise LedgerError("일자 중복 적재")
    if days != sorted(days):
        raise LedgerError("일자 비단조 — 시간순 백필 위반")
    charged: dict[str, int] = {}
    for r in rows:
        if r.get("row_type") == "daily" and float(r.get("cost") or 0) > 0:
            charged[r["universe_id"]] = charged.get(r["universe_id"], 0) + 1
    dup = [u for u, n in charged.items() if n > 1]
    if dup:
        raise LedgerError(f"같은 유니버스에 재구성 비용 중복 부과: {dup}")


def migrate_legacy(state: dict, rows: list[dict]) -> tuple[list[dict], bool]:
    """정정 이전(결함 코드) 계열을 state 에 동결하고 이력을 비운다 (멱등).

    정정 공시 C4 의 네 결함 위에서 적재된 계열은 spec 계열과 비교 불가다.
    삭제하지 않고 state(`legacy`) 에 원문·SHA-256 으로 보존한다 — 워크플로가
    커밋하는 파일이라 CI 에서도 실제로 남는다. state 를 먼저 갱신하고 CSV 를
    비우므로, 중간 크래시에도 원문이 사라지지 않는다.

    Returns:
        (이후 사용할 행, 변경 여부).
    """
    if not rows or any(r.get("row_type") for r in rows):
        return rows, False                            # 없음 / 이미 신형식
    if len(rows) > 5:
        raise LedgerError(f"예상 밖 레거시 행 수 {len(rows)} — 수동 확인 필요")
    raw = HIST.read_text(encoding="utf-8")
    frozen = dict(
        reason="정정 공시 C4 (24h 유니버스·k접두 누락·재구성비용 0·베이시스 미기록)",
        phase="legacy_invalid", source=str(HIST),
        sha256=hashlib.sha256(raw.encode()).hexdigest(),
        rows=[{k: r.get(k) for k in ("day", "equity", "day_diff", "n_coins")}
              for r in rows])
    old = state.get("legacy")
    if old and old.get("sha256") != frozen["sha256"]:
        raise LedgerError("레거시 동결본과 현재 이력이 불일치 — 수동 확인 필요")
    state["legacy"] = frozen
    _atomic_write(STATE, json.dumps(state, ensure_ascii=False, indent=1))
    logger.warning("레거시 %d행을 state 로 동결하고 spec 계열을 새로 시작한다",
                   len(rows))
    return [], True


# ── 유니버스 파일 (직전 스냅샷을 같은 파일에 보관 — 쓰고-나서-읽기 금지) ──

def load_universe() -> dict:
    """유니버스 파일 로드. 구형식(코인 리스트)이면 빈 장부로 취급한다."""
    if not UNIVERSE_F.exists():
        return dict(rule=UNIVERSE_RULE, active=None, order=[], snapshots={})
    try:
        d = json.loads(UNIVERSE_F.read_text())
    except ValueError as e:
        raise LedgerError(f"유니버스 파일 파손: {e}") from e
    if isinstance(d, list) or d.get("rule") != UNIVERSE_RULE:
        logger.warning("구 규칙 유니버스 감지 → 사양(30일 중앙 $5M)으로 즉시 재구축")
        return dict(rule=UNIVERSE_RULE, active=None, order=[], snapshots={})
    return d


def save_universe(book: dict, snap: dict) -> None:
    """스냅샷을 추가하고 활성 포인터를 옮긴다 (최근 6개 보존)."""
    old = book["snapshots"].get(snap["id"])
    if old and _snapshot_id(old) != snap["id"]:
        raise LedgerError(f"같은 id 에 다른 스냅샷 덮어쓰기 시도: {snap['id'][:12]}")
    book["snapshots"][snap["id"]] = snap
    book["order"] = [i for i in book.get("order", []) if i != snap["id"]] + [snap["id"]]
    for stale_id in book["order"][:-6]:
        book["snapshots"].pop(stale_id, None)
    book["order"] = book["order"][-6:]
    book["active"] = snap["id"]
    book["rule"] = UNIVERSE_RULE
    _atomic_write(UNIVERSE_F, json.dumps(book, ensure_ascii=False, indent=1))


def expected_as_of(day: pd.Timestamp, spec_t0: pd.Timestamp) -> pd.Timestamp:
    """day 에 적용될 유니버스의 as-of (월초, 단 첫 부분월은 spec_t0)."""
    return min(max(day.normalize().replace(day=1), spec_t0), day.normalize())


def ensure_universe(book: dict, day: pd.Timestamp, spec_t0: pd.Timestamp,
                    deadline: float, scale: float = 1.0
                    ) -> tuple[dict | None, list[str]]:
    """day 에 적용될 유니버스를 확보한다 (월 1회 재평가 + 규칙 변경 시 즉시).

    `active` 포인터가 아니라 (월, as-of) 로 **정확히 그 날짜의** 스냅샷을 고른다
    — 백필 중 크래시로 미래 스냅샷이 active 로 남아 있어도 역방향 전환비용을
    만들지 않는다. 직전 스냅샷도 as-of 가 더 이른 것 중 최신으로 고른다.
    """
    want = expected_as_of(day, spec_t0)
    want_s, month = str(want.date()), f"{day.year}-{day.month:02d}"
    for sid in reversed(book.get("order", [])):
        s = book["snapshots"].get(sid) or {}
        if s.get("rule") == UNIVERSE_RULE and s.get("as_of") == want_s \
                and s.get("month") == month:
            return s, []
    # 직전 스냅샷은 as-of 가 더 이른 것 중 **최신**이다 (같은 as-of 면 나중에
    # 기록된 파생본). book order 가 뒤섞여도 날짜로 고른다.
    earlier = [(s["as_of"], i, s) for i, sid in enumerate(book.get("order", []))
               if (s := book["snapshots"].get(sid)) and s.get("as_of", "9999") < want_s]
    prev = max(earlier, default=(None, None, None))[2]
    snap, unknown = build_universe(want, deadline, prev_snap=prev, scale=scale)
    if snap is None:
        return None, unknown
    snap["month"] = month
    snap["id"] = _snapshot_id(snap)
    save_universe(book, snap)
    logger.info("유니버스 재구축 %s: %d종 (후보 %d) — 전환비용 %.2fbp %s",
                snap["as_of"], len(snap["coins"]), snap["candidates"],
                snap["transition"]["cost_bp"], snap["transition"])
    return snap, []


# ── 관측성 ───────────────────────────────────────────────────────────────

def _announce(title: str, msg: str) -> None:
    """GitHub Actions 주석·스텝 요약으로 막힘을 노출한다 (조용한 초록 방지).

    워크플로에 `if: always()` 를 붙일 수 없어(파일 소유권) 비정상 종료는 상태
    커밋 자체를 스킵시킨다 → 데이터 보류는 종료코드 대신 annotation + state 로
    드러내고, 원장 불변식 위반만 예외로 비정상 종료시킨다.
    """
    logger.error("%s: %s", title, msg)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(f"\n### ⚠ {title}\n\n{msg}\n")
        except OSError as e:                      # 요약 실패가 수집을 막으면 안 된다
            logger.debug("STEP_SUMMARY 기록 실패: %s", e)
    if os.environ.get("GITHUB_ACTIONS"):
        # Actions 주석 채널은 stdout 전용이라 logging 으로 대체 불가.
        sys.stdout.write(f"::error title={title}::{msg}\n")


# ── 메인 ─────────────────────────────────────────────────────────────────

def _resolve_t0(state: dict, rows: list[dict], yesterday: pd.Timestamp) -> str:
    """spec 계열 T0 를 확정한다 — 한 번 정하면 이동하지 않는다.

    우선순위: state → 이력 앵커(앵커일+1) → 환경변수 TRACKC_SPEC_T0 → 어제.
    state 와 앵커가 어긋나면 LedgerError (조용한 시계 이동 금지).
    """
    anchor = next((r["day"] for r in rows if r.get("row_type") == "anchor"), None)
    from_anchor = (str((pd.Timestamp(anchor, tz="utc") + pd.Timedelta(days=1)).date())
                   if anchor else None)
    t0 = state.get("spec_t0") or from_anchor or os.environ.get("TRACKC_SPEC_T0") \
        or str(yesterday.date())
    if from_anchor and t0 != from_anchor:
        raise LedgerError(f"spec_t0({t0}) 와 이력 앵커({from_anchor}) 불일치")
    if not state.get("spec_t0"):
        logger.warning("spec 계열 T0 고정: %s (정정 후 최초, 이후 이동 금지)", t0)
    return t0


def main() -> None:
    """spec_t0 부터 어제까지를 시간순으로 처리한다 (멱등·백필·fail-closed)."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    deadline = time.monotonic() + DEADLINE_SEC
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    rows, dirty = migrate_legacy(state, load_rows())
    audit_rows(rows)

    now = _now_utc()
    yesterday = now.normalize() - pd.Timedelta(days=1)
    state["spec_t0"] = _resolve_t0(state, rows, yesterday)
    spec_t0 = pd.Timestamp(state["spec_t0"], tz="utc")
    if not rows:
        # 기준점 행: 대시보드가 첫 행 equity 를 base 로 삼으므로, 진입비용과
        # 첫날 손익이 base 에 흡수돼 사라지지 않도록 1.0 앵커를 먼저 놓는다.
        rows = [dict(day=str((spec_t0 - pd.Timedelta(days=1)).date()),
                     row_type="anchor", phase="spec_v2", universe_id="", n_coins=0,
                     day_diff=0.0, basis_diff=0.0, cost=0.0, notional_base=0.0,
                     equity=1.0, equity_funding=1.0)]
        dirty = True

    book = load_universe()
    done = {r["day"] for r in rows if r.get("row_type") == "daily"}
    pending = [d for d in pd.date_range(spec_t0, yesterday, freq="D", tz="utc")
               if str(d.date()) not in done]
    blocked: dict | None = None
    for day in pending[:MAX_BACKFILL_DAYS]:
        if time.monotonic() > deadline:
            blocked = dict(reason_code="DEADLINE", day=str(day.date()))
            break
        prev = last_accounting(rows)
        if prev is None:
            raise LedgerError("회계 기준 행 없음 — 이력 파일 확인 필요")
        prev_base, prev_eq = float(prev["notional_base"]), float(prev["equity"])
        scale = prev_base / prev_eq if prev_base > 0 and prev_eq > 0 else 1.0
        snap, unknown = ensure_universe(book, day, spec_t0, deadline, scale)
        if snap is None:
            blocked = dict(reason_code="UNIVERSE_UNKNOWN", day=str(day.date()),
                           unknown_coins=unknown[:20])
            break
        pnl, unknown = day_pnl(snap, day, deadline)
        if pnl is None and unknown and not unknown[0].startswith("<"):
            gone, lookup_failed = find_delisted(snap)
            if gone and not lookup_failed:
                ex, missing = delist_exit(snap, gone, day)
                if ex is None:
                    blocked = dict(reason_code="DELIST_EXIT_PRICE_PENDING",
                                   day=str(day.date()), unknown_coins=missing)
                    break
                derived = exit_snapshot(snap, gone, ex["marks"], ex)
                logger.warning("상장폐지 강제청산 %s → %.2fbp, 청산손익 "
                               "펀딩 %+.5f%% 베이시스 %+.5f%% (잔여 %d종)", gone,
                               derived["transition"]["cost_bp"], ex["funding"] * 100,
                               ex["basis"] * 100, len(derived["coins"]))
                save_universe(book, derived)
                snap = derived
                pnl, unknown = day_pnl(snap, day, deadline)
        if pnl is None:
            blocked = dict(reason_code="DAY_INCOMPLETE", day=str(day.date()),
                           unknown_coins=unknown[:20])
            break

        activation = not any(r.get("universe_id") == snap["id"]
                             and r.get("row_type") == "daily" for r in rows)
        # 명목은 리밸런스 시점에 고정된다 → 일별 손익은 가산(복리 아님).
        # 강제청산(keep_base)은 생존 수량을 그대로 두므로 명목 기준도 유지한다 —
        # 재설정하면 생존 포지션이 비용 없이 커진다.
        keep = bool(snap.get("keep_base"))
        cost = ((prev_base if keep else prev_eq)
                * snap["gross_traded"] * COST_RT) if activation else 0.0
        base = (prev_base if keep or not activation else prev_eq - cost)
        eq_pre = prev_eq - cost
        eqf_pre = float(prev["equity_funding"]) - cost
        # 강제청산 종목의 마지막 구간 손익은 그 파생본의 첫 행에서 1회 반영한다.
        ex = snap.get("exit_pnl") if activation else None
        f_day = pnl["funding"] + (ex["funding"] if ex else 0.0)
        b_day = pnl["basis"] + (ex["basis"] if ex else 0.0)
        eq = eq_pre + base * (f_day + b_day)
        eqf = eqf_pre + base * f_day
        rows.append(dict(
            day=str(day.date()), row_type="daily", phase="spec_v2",
            universe_id=snap["id"], n_coins=len(snap["coins"]),
            day_diff=round(base * f_day, 12), basis_diff=round(base * b_day, 12),
            cost=round(cost, 12), notional_base=round(base, 12),
            equity=round(eq, 12), equity_funding=round(eqf, 12)))
        audit_rows(rows)
        write_rows(rows)                       # 날짜마다 즉시 확정 (고아 스냅샷 방지)
        dirty = False
        logger.info("%s: 펀딩 %+.5f%% 베이시스 %+.5f%% 비용 %.2fbp → 자본 %.6f "
                    "(코인 %d, ROE 환산 ÷2)", day.date(), pnl["funding"] * 100,
                    pnl["basis"] * 100, cost * 1e4, eq, len(snap["coins"]))

    if dirty:
        write_rows(rows)
    left = [str(d.date()) for d in pending
            if str(d.date()) not in {r["day"] for r in rows
                                     if r.get("row_type") == "daily"}]
    last = last_accounting(rows) or {}
    # 판정 입력이 무엇인지 T0 시점에 저장소에 못박는다 (사후 유리한 계열 선택 방지).
    state.update(binding="equity = funding + basis − costs (ROE = ÷2)",
                 rule=UNIVERSE_RULE, equity=float(last.get("equity", 1.0)),
                 equity_funding=float(last.get("equity_funding", 1.0)),
                 universe_id=last.get("universe_id") or book.get("active"),
                 last_day=last.get("day"), n_coins=int(last.get("n_coins", 0) or 0),
                 verdict_day=str((spec_t0 + pd.Timedelta(days=90)).date()))
    if blocked or left:
        blocked = blocked or dict(reason_code="BACKLOG", day=left[0])
        blocked.update(last_attempt_utc=now.isoformat(), pending_days=left[:10],
                       last_success_day=last.get("day"))
        blocked.setdefault("blocked_since",
                           (state.get("blocked") or {}).get("blocked_since",
                                                            now.isoformat()))
        blocked["retry_count"] = (state.get("blocked") or {}).get("retry_count", 0) + 1
        state["blocked"] = blocked
        _announce("Track C 수집 보류",
                  f"{blocked['reason_code']} @ {blocked['day']} — "
                  f"unknown={blocked.get('unknown_coins')} 미처리={left[:5]} "
                  f"(재시도 {blocked['retry_count']}회, 이후 백필)")
    elif state.pop("blocked", None):
        logger.info("보류 해소 — 백필 완료 (마지막 %s)", last.get("day"))
    _atomic_write(STATE, json.dumps(state, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
