from __future__ import annotations

"""Track D 라이브 러너 — 마지막 처리 이후의 '닫힌' 1h 봉을 전부 재생한다 (일 1회, 멱등)."""

import json
import logging
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from carrybot.aggressive.swing import (ATR_N, ENTRY_N, EXIT_N, Bar1h, SwingState,
                                       _mtm, step_bar)

logger = logging.getLogger(__name__)

STATE = Path("logs/trackd_state.json")
HIST = Path("logs/trackd_history.csv")
SYMS = ("BTC", "ETH", "SOL")


def _retry(fn, *a, **k):
    for i in range(6):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            if i == 5:
                return None
            time.sleep(1.5 * (i + 1))
    return None


def fetch_1h(ex, sym: str, n: int) -> pd.DataFrame | None:
    """닫힌 1h 봉 n개 (진행 중 봉 제외)."""
    rs = _retry(ex.fetch_ohlcv, f"{sym}/USDT:USDT", "1h", limit=min(n + 2, 1000))
    if not rs:
        return None
    df = pd.DataFrame(rs, columns=["ts", "open", "high", "low", "close", "vol"])
    now_h = int(pd.Timestamp.now(tz="utc").floor("h").timestamp() * 1000)
    return df[df.ts < now_h].tail(n)


def fetch_funding_day(ex, sym: str, day: pd.Timestamp) -> float:
    r = _retry(ex.publicGetV5MarketFundingHistory,
               {"category": "linear", "symbol": f"{sym}USDT", "limit": 30})
    if not r:
        return float("nan")
    tot, seen = 0.0, False
    for x in r.get("result", {}).get("list", []):
        t = pd.to_datetime(int(x["fundingRateTimestamp"]), unit="ms", utc=True)
        if t.normalize() == day:
            tot += float(x["fundingRate"])
            seen = True
    return tot if seen else float("nan")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ex = ccxt.bybit({"enableRateLimit": True})
    _retry(ex.load_markets)
    state = SwingState.from_dict(json.loads(STATE.read_text())) if STATE.exists() else SwingState()

    hist = {s: fetch_1h(ex, s, ENTRY_N + 30) for s in SYMS}
    if any(h is None or len(h) < ENTRY_N + 5 for h in hist.values()):
        logger.error("1h 데이터 부족 — fail-closed 스킵")
        return
    # 전일 펀딩 (일 첫 봉에 반영)
    fund = {}
    for s in SYMS:
        y = pd.Timestamp.now(tz="utc").normalize() - pd.Timedelta(days=1)
        fund[s] = fetch_funding_day(ex, s, y)

    common = sorted(set.intersection(*[set(h.ts) for h in hist.values()]))
    new_ts = [t for t in common if t > state.last_ts]
    if not new_ts:
        logger.info("새 봉 없음")
        return
    n_fills = 0
    for t in new_ts:
        bars = {}
        day_key = str(pd.Timestamp(t, unit="ms", tz="utc").date())
        is_day_first = pd.Timestamp(t, unit="ms", tz="utc").hour == 0
        for s in SYMS:
            h = hist[s]
            i = h.index[h.ts == t]
            if len(i) == 0:
                continue
            i = i[0]
            pos = h.index.get_loc(i)
            past = h.iloc[:pos]
            if len(past) < ENTRY_N:
                continue
            row = h.loc[i]
            bars[s] = Bar1h(ts=int(t), open=row.open, high=row.high, low=row.low,
                            close=row.close,
                            ehi=past.high.tail(ENTRY_N).max(), elo=past.low.tail(ENTRY_N).min(),
                            xhi=past.high.tail(EXIT_N).max(), xlo=past.low.tail(EXIT_N).min(),
                            funding=(fund.get(s, float("nan")) if is_day_first else 0.0))
        if not bars:
            continue
        fills = step_bar(state, bars, day_key)
        n_fills += len(fills)
        for f in fills:
            logger.info("  %s %s %s @ %.2f pnl %+.5f",
                        pd.Timestamp(t, unit="ms", tz="utc"), f["sym"], f["action"],
                        f["price"], f["pnl"])
    STATE.write_text(json.dumps(state.to_dict(), default=float))
    last_bars = bars
    mtm = _mtm(state, last_bars)
    row = pd.DataFrame([dict(day=str(pd.Timestamp.now(tz="utc").date()),
                             equity=round(mtm, 8), n_pos=len(state.positions),
                             bars=len(new_ts), fills=n_fills)])
    row.to_csv(HIST, mode="a", header=not HIST.exists(), index=False)
    logger.info("봉 %d개 처리, 체결 %d, 자본 %.5f, 포지션 %d",
                len(new_ts), n_fills, mtm, len(state.positions))


if __name__ == "__main__":
    main()
