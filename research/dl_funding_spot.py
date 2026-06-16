"""델타중립 펀딩 캐리용 데이터 — 현물 1h OHLCV + 펀딩 히스토리(8h) 다운로드.

research/data/{SYM}_spot_1h.pkl, {SYM}_funding.pkl 캐시(idempotent). perp 1h는 기존 캐시 재사용.
사용법: python -m research.dl_funding_spot --start 2024-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402
import research.wfo as wfo  # noqa: E402

logger = logging.getLogger("dl_fund_spot")
DATA = ROOT / "research" / "data"


def _san(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "-")


def dl_spot(ex, perp_symbol: str, start: str) -> None:
    """현물 1h OHLCV 다운로드 (perp 심볼 → 현물 심볼 변환)."""
    spot_symbol = perp_symbol.split(":")[0]          # BTC/USDT:USDT → BTC/USDT
    cache = DATA / f"{_san(perp_symbol)}_spot_1h.pkl"
    if cache.exists():
        logger.info("  %s 현물 캐시 존재", spot_symbol); return
    import ccxt
    tf_ms = 3_600_000
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list = []
    while since < ex.milliseconds() - tf_ms:
        for attempt in range(6):
            try:
                batch = ex.fetch_ohlcv(spot_symbol, "1h", since=since, limit=1000)
                break
            except ccxt.RateLimitExceeded:
                time.sleep(0.5 * (2 ** attempt))
        else:
            raise RuntimeError(f"{spot_symbol} 레이트리밋 초과")
        if not batch or batch[-1][0] < since:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        time.sleep(0.15)
    if not rows:
        logger.warning("  %s 현물 데이터 없음", spot_symbol); return
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    logger.info("  %s 현물 %d캔들 캐시", spot_symbol, len(df))


def dl_funding(ex, perp_symbol: str, start: str) -> None:
    """펀딩 히스토리(8h) 다운로드."""
    cache = DATA / f"{_san(perp_symbol)}_funding.pkl"
    if cache.exists():
        logger.info("  %s 펀딩 캐시 존재", perp_symbol); return
    import ccxt
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list = []
    while since < ex.milliseconds():
        for attempt in range(6):
            try:
                batch = ex.fetch_funding_rate_history(perp_symbol, since=since, limit=200)
                break
            except ccxt.RateLimitExceeded:
                time.sleep(0.5 * (2 ** attempt))
        else:
            raise RuntimeError(f"{perp_symbol} 펀딩 레이트리밋 초과")
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["timestamp"]
        if last <= since:
            break
        since = last + 1
        time.sleep(0.15)
    if not rows:
        logger.warning("  %s 펀딩 없음", perp_symbol); return
    df = pd.DataFrame([{"ts": r["timestamp"], "funding": float(r["fundingRate"])}
                       for r in rows]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df.to_pickle(cache)
    logger.info("  %s 펀딩 %d건 캐시 (%s~%s)", perp_symbol, len(df), df.index.min(), df.index.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    import ccxt
    spot_ex = ccxt.bybit({"options": {"defaultType": "spot"}, "enableRateLimit": True})
    perp_ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    t0 = time.time()
    for i, sym in enumerate(wfo.UNIVERSE, 1):
        logger.info("[%d/%d] %s", i, len(wfo.UNIVERSE), sym)
        try:
            dl_spot(spot_ex, sym, args.start)
        except Exception as e:  # noqa: BLE001
            logger.warning("  현물 실패: %s", e)
        try:
            dl_funding(perp_ex, sym, args.start)
        except Exception as e:  # noqa: BLE001
            logger.warning("  펀딩 실패: %s", e)
    logger.info("완료 (%.1f분)", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
