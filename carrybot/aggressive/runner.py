from __future__ import annotations

"""Track B 라이브 러너 — 일 1회, 직전 '닫힌' UTC 일봉을 step()에 공급한다.

페이퍼 전용. 상태는 logs/trackb_state.json, 이력은 logs/trackb_history.csv.
백테스트와 동일한 step()을 사용하므로 로직 불일치가 구조적으로 불가능하다.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from carrybot.aggressive.turtle import Bar, TurtleConfig, TurtleState, step, _mtm

logger = logging.getLogger(__name__)

STATE = Path("logs/trackb_state.json")
HIST = Path("logs/trackb_history.csv")


def _retry(fn, *a, **k):
    """공개 엔드포인트 재시도."""
    for i in range(6):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            if i == 5:
                return None
            time.sleep(1.5 * (i + 1))
    return None


def fetch_closed_daily(ex, sym: str, n: int = 90) -> pd.DataFrame | None:
    """닫힌 일봉만 반환한다 (진행 중인 오늘 봉 제외)."""
    r = _retry(ex.publicGetV5MarketKline,
               {"category": "linear", "symbol": f"{sym}USDT", "interval": "D", "limit": n + 2})
    if not r:
        return None
    rows = sorted(((int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
                   for x in r.get("result", {}).get("list", [])), key=lambda z: z[0])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    today = pd.Timestamp.now(tz="utc").normalize()
    return df[df["ts"] < today].set_index("ts")


def fetch_funding_day(ex, sym: str, day: pd.Timestamp) -> float:
    """해당 UTC 일자에 정산된 펀딩 합."""
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


def build_bars(ex, cfg: TurtleConfig) -> tuple[dict[str, Bar], pd.Timestamp] | None:
    """직전 닫힌 일자의 Bar 딕셔너리를 만든다."""
    bars, day = {}, None
    for s in cfg.syms:
        d = fetch_closed_daily(ex, s)
        if d is None or len(d) < cfg.entry_n + 2:
            logger.warning("%s 일봉 부족 — 스킵", s)
            continue
        last = d.index[-1]
        day = last if day is None else max(day, last)
        hist = d.iloc[:-1]      # 당일 제외한 채널
        bars[s] = Bar(
            open=float(d["open"].iloc[-1]), high=float(d["high"].iloc[-1]),
            low=float(d["low"].iloc[-1]), close=float(d["close"].iloc[-1]),
            entry_hi=float(hist["high"].tail(cfg.entry_n).max()),
            entry_lo=float(hist["low"].tail(cfg.entry_n).min()),
            exit_hi=float(hist["high"].tail(cfg.exit_n).max()),
            exit_lo=float(hist["low"].tail(cfg.exit_n).min()),
            funding=fetch_funding_day(ex, s, last),
        )
    if not bars or day is None:
        return None
    return bars, day


def load_state() -> TurtleState:
    """상태를 읽는다 (없으면 초기 상태)."""
    if STATE.exists():
        return TurtleState.from_dict(json.loads(STATE.read_text()))
    return TurtleState()


def main() -> None:
    """직전 닫힌 일봉 1개를 처리하고 상태·이력을 저장한다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TurtleConfig()
    ex = ccxt.bybit({"enableRateLimit": True})
    out = build_bars(ex, cfg)
    if out is None:
        logger.error("일봉 수집 실패 — 오늘은 건너뜀 (fail-closed)")
        return
    bars, day = out

    # 멱등성: 같은 날을 두 번 처리하지 않는다
    if HIST.exists():
        h = pd.read_csv(HIST)
        if len(h) and h["day"].iloc[-1] == str(day.date()):
            logger.info("%s 이미 처리됨 — 종료", day.date())
            return

    state = load_state()
    if state.killed:
        logger.warning("월손실 킬 상태 — 수동 리셋 필요 (logs/trackb_state.json killed=false)")
        return
    state, fills = step(state, bars, cfg, month_key=f"{day.year}-{day.month:02d}")

    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state.to_dict(), indent=2, default=float))
    mtm = _mtm(state, bars)
    row = pd.DataFrame([dict(day=str(day.date()), equity=round(mtm, 8),
                             cash=round(state.equity, 8),
                             n_pos=len(state.positions),
                             fills="; ".join(f"{f['sym']}:{f['action']}@{f['price']:.2f}"
                                             for f in fills) or "-")])
    row.to_csv(HIST, mode="a", header=not HIST.exists(), index=False)
    logger.info("%s 처리 완료: 자본 %.5f, 포지션 %d, 체결 %d건",
                day.date(), mtm, len(state.positions), len(fills))
    for f in fills:
        logger.info("  %s %s @ %.2f pnl %.5f", f["sym"], f["action"], f["price"], f["pnl"])


if __name__ == "__main__":
    main()
