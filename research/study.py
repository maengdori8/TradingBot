"""
전략 신호 재현 연구 — 과거 데이터 전체에 ICT 스캔을 재생해 신호·결과를 수집한다.

설계:
- Bybit 거래량 상위 N개 USDT 무기한 + BTC, 15m/1h/4h 캔들 다운로드(페이지네이션, 캐시)
- 킬존 게이트를 열어두고(always-true 패치) 24시간 전체에서 신호 생성
  → 실제 킬존 여부는 타임스탬프로 별도 태깅 → 킬존 가치 자체를 검증 가능
- 신호마다 (손절 ATR배수 × 목표 R:R) 그리드의 순R(수수료·슬리피지 반영) 결과 시뮬레이션
- 룩어헤드 차단: 신호 시점 이후 캔들로만 결과 판정, 동시 터치는 손절 우선(보수적)

출력: research/out/signals.csv
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "research" / "data"
OUT_DIR = ROOT / "research" / "out"

logger = logging.getLogger("study")

# 수수료/슬리피지 (paper_engine과 동일 가정: taker 0.055%×2 + slip 0.05%×2)
COST_PCT = 0.0021
# 결과 판정 최대 보유 (15m 캔들 수): 7일
MAX_HOLD = 672
# 손절폭(ATR 배수) × 목표 R:R 그리드
SL_MULTS = [1.0, 1.5, 2.0]
RR_LEVELS = [1.5, 2.0, 2.5, 3.0]


def _san(symbol: str) -> str:
    """심볼을 파일명으로 변환한다."""
    return symbol.replace("/", "_").replace(":", "-")


# ──────────────────────────────────────────────────────────────────────
# 데이터 다운로드 (페이지네이션 + 캐시)
# ──────────────────────────────────────────────────────────────────────

def fetch_history(ex, symbol: str, tf: str, months: int) -> pd.DataFrame:
    """과거 캔들을 페이지네이션으로 수집한다 (캐시 사용).

    Args:
        ex: ccxt 거래소 인스턴스
        symbol: 거래 심볼
        tf: 타임프레임
        months: 수집 개월 수

    Returns:
        OHLCV DataFrame (UTC index)
    """
    cache = DATA_DIR / f"{_san(symbol)}_{tf}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    tf_ms = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}[tf]
    since = ex.milliseconds() - months * 30 * 24 * 3600 * 1000
    rows: list = []
    while since < ex.milliseconds() - tf_ms:
        batch = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch or batch[-1][0] < since:   # 진전 없음
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        time.sleep(0.05)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    logger.info("%s %s: %d캔들 캐시", symbol, tf, len(df))
    return df


def top_symbols(ex, limit: int) -> list[str]:
    """거래량 상위 USDT 무기한 심볼 목록 (BTC/ETH 포함 보장)."""
    tickers = ex.fetch_tickers()
    scored = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDT") or "/USDT:" not in sym:
            continue
        vol = t.get("quoteVolume") or 0
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            vol = 0
        if vol > 0:
            scored.append((sym, vol))
    scored.sort(key=lambda x: x[1], reverse=True)
    syms = [s for s, _ in scored[:limit]]
    for must in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        if must not in syms:
            syms.insert(0, must)
    return syms


def download_all(symbols: list[str], months: int) -> None:
    """모든 심볼의 15m/1h/4h 데이터를 받는다 (순차, 캐시 스킵)."""
    import ccxt
    ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        for tf in ("15m", "1h", "4h"):
            try:
                fetch_history(ex, sym, tf, months)
            except Exception as e:
                logger.warning("%s %s 다운로드 실패: %s", sym, tf, e)
        if i % 5 == 0:
            logger.info("다운로드 진행 %d/%d", i, total)


# ──────────────────────────────────────────────────────────────────────
# 신호 재현 (워커: 심볼 1개)
# ──────────────────────────────────────────────────────────────────────

class _FakeDT:
    """signal_engine의 datetime.now()를 캔들 시각으로 바꾸는 셰임."""

    _now: datetime = datetime.now(timezone.utc)

    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        return cls._now

    @classmethod
    def fromisoformat(cls, s):  # noqa: ANN001
        return datetime.fromisoformat(s)


def _simulate(entry: float, direction: str, atr: float,
              highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
              start: int) -> dict:
    """신호 이후 가격경로에서 (손절배수×RR) 그리드의 순R 결과를 계산한다.

    동시 터치 캔들은 손절 우선(보수적). 미해결 시 타임아웃 청산.

    Returns:
        {"r_m{mult}_rr{rr}": 순R, ..., "resolve_idx": 기준조합 해소 인덱스}
    """
    end = min(start + MAX_HOLD, len(closes))
    h = highs[start:end]
    lo = lows[start:end]
    if len(h) == 0:
        return {}
    out: dict = {}
    resolve_idx = end
    for mult in SL_MULTS:
        risk = atr * mult
        if risk <= 0:
            continue
        cost_r = COST_PCT * entry / risk  # 수수료+슬리피지를 R로 환산
        if direction == "long":
            sl = entry - risk
            sl_hits = np.nonzero(lo <= sl)[0]
        else:
            sl = entry + risk
            sl_hits = np.nonzero(h >= sl)[0]
        sl_i = sl_hits[0] if len(sl_hits) else 10**9
        for rr in RR_LEVELS:
            if direction == "long":
                tp = entry + risk * rr
                tp_hits = np.nonzero(h >= tp)[0]
            else:
                tp = entry - risk * rr
                tp_hits = np.nonzero(lo <= tp)[0]
            tp_i = tp_hits[0] if len(tp_hits) else 10**9
            if sl_i <= tp_i and sl_i < 10**9:        # 손절 우선
                r = -1.0
                done = sl_i
            elif tp_i < sl_i:
                r = rr
                done = tp_i
            else:                                     # 타임아웃: 종가 청산
                last = closes[end - 1]
                r = (last - entry) / risk if direction == "long" else (entry - last) / risk
                done = end - start - 1
            out[f"r_m{mult:g}_rr{rr:g}"] = round(r - cost_r, 4)
            if mult == 1.5 and rr == 2.0:
                resolve_idx = start + done
    out["resolve_idx"] = int(resolve_idx)
    return out


def replay_symbol(symbol: str) -> list[dict]:
    """심볼 1개의 전체 역사에 스캔을 재생해 신호+결과 행들을 반환한다."""
    import src.strategy.signal_engine as SE
    from src.strategy.kill_zone import is_in_kill_zone as real_kz
    from src.strategy.kill_zone import get_active_session as real_session
    from src.strategy import load_strategy_params

    # 킬존 게이트 개방 + 시계 패치 (역사 재생)
    SE.datetime = _FakeDT
    SE.is_in_kill_zone = lambda dt: True
    SE.get_active_session = lambda dt: real_session(dt) or "none"

    try:
        df15 = pd.read_pickle(DATA_DIR / f"{_san(symbol)}_15m.pkl")
        df1h = pd.read_pickle(DATA_DIR / f"{_san(symbol)}_1h.pkl")
        df4h = pd.read_pickle(DATA_DIR / f"{_san(symbol)}_4h.pkl")
    except FileNotFoundError:
        return []
    if len(df15) < 800 or len(df4h) < 120:
        return []

    atr_mult = load_strategy_params()["atr"]["multiplier"]

    ts15 = df15.index
    highs = df15["high"].to_numpy()
    lows = df15["low"].to_numpy()
    closes = df15["close"].to_numpy()
    # 변동성 레짐: ATR%(14)의 전체기간 백분위 (연구용 공변량)
    atr_series = (df15["high"] - df15["low"]).rolling(14).mean().to_numpy()
    atr_pct = atr_series / closes
    valid = ~np.isnan(atr_pct)
    ranks = np.full(len(atr_pct), np.nan)
    if valid.sum() > 0:
        order = atr_pct[valid].argsort().argsort()
        ranks[valid] = order / max(1, valid.sum() - 1)

    # 각 15m 시각 이하의 마지막 1h/4h 캔들 위치 (룩어헤드 없음)
    idx1h = np.searchsorted(df1h.index.values, ts15.values, side="right") - 1
    idx4h = np.searchsorted(df4h.index.values, ts15.values, side="right") - 1

    # 4h 워밍업 100개 보장 지점부터 시작
    start_i = int(np.searchsorted(idx4h, 100))
    start_i = max(start_i, 100)

    rows: list[dict] = []
    busy_until = -1
    for i in range(start_i, len(df15)):
        if i <= busy_until:
            continue
        j1, j4 = idx1h[i], idx4h[i]
        if j1 < 100 or j4 < 100:
            continue
        ts = ts15[i].to_pydatetime()
        _FakeDT._now = ts
        w15 = df15.iloc[i - 99: i + 1]
        w1h = df1h.iloc[j1 - 99: j1 + 1]
        w4h = df4h.iloc[j4 - 99: j4 + 1]
        price = closes[i]
        try:
            res = SE.scan_symbol(
                w4h, w1h, w15, symbol, float(price),
                min_rr=2.0, min_score=0.0, require_volume=False,
            )
        except Exception:
            continue
        sig = res.signal
        if sig is None:
            continue

        risk_d = abs(sig.entry_price - sig.stop_loss)
        atr = risk_d / atr_mult if atr_mult > 0 else risk_d
        sim = _simulate(float(sig.entry_price), sig.direction, atr,
                        highs, lows, closes, i + 1)
        if not sim:
            continue
        busy_until = sim.pop("resolve_idx")

        kz = bool(real_kz(ts))
        ck = res.checks or {}
        rows.append({
            "symbol": symbol, "ts": ts15[i].isoformat(),
            "direction": sig.direction,
            "score_raw": round(res.score, 1),
            "kz_true": kz,
            "session": real_session(ts) or "none",
            "weekday": ts.weekday(),
            "volume_ok": bool(ck.get("volume", False)),
            "zone_both": "FVG+OB" in (res.reason or ""),
            "entry": float(sig.entry_price),
            "atr_pct_rank": round(float(ranks[i]), 3) if not np.isnan(ranks[i]) else None,
            **sim,
        })
    return rows


def run_replay(symbols: list[str], workers: int) -> pd.DataFrame:
    """전 심볼 병렬 재생."""
    all_rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(replay_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("%s 재생 실패: %s", sym, e)
                rows = []
            done += 1
            logger.info("재생 %d/%d — %s 신호 %d개 (누적 %d)",
                        done, len(symbols), sym, len(rows), len(all_rows))
    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "signals.csv", index=False)
    return df
