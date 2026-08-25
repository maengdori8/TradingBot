from __future__ import annotations

"""Track A 페이퍼 러너 — 사전등록 v2 캐리 알고리즘을 실존 코인으로 매일 검증한다.

매일 UTC 마감 후 1회:
  1. 등록 알고리즘(ADV상위2 유니버스, 초과캐리 허들 9.73%, 최소보유 30일)으로 판단
  2. 보유 포지션에 전일 펀딩·베이시스 변동을 정산 (3계정 회계)
  3. 상태를 logs/tracka_state.json, 이력을 logs/tracka_history.csv 에 기록

수익 검증의 기준 구현은 carrybot/research/ledger.py 이며, 이 러너는 그 규칙의
전향적 실행 기록이다. 월 1회 동결 데이터 재생과 대사(reconcile)한다.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from carrybot.live.signal import _adv_thin_leg, _daily_funding, _retry
from carrybot.research.ledger import LedgerConfig

logger = logging.getLogger(__name__)

STATE = Path("logs/tracka_state.json")
HIST = Path("logs/tracka_history.csv")
BASES = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")


@dataclass
class CarryPaperState:
    """직렬화 가능한 Track A 페이퍼 상태 (비중 기반)."""

    equity: float = 1.0
    positions: dict = field(default_factory=dict)   # sym -> {weight, opened, basis}
    last_day: str = ""

    def to_dict(self) -> dict:
        """JSON 직렬화."""
        return dict(equity=self.equity, positions=self.positions, last_day=self.last_day)

    @classmethod
    def from_dict(cls, d: dict) -> "CarryPaperState":
        """JSON 역직렬화."""
        return cls(equity=d["equity"], positions=dict(d.get("positions", {})),
                   last_day=d.get("last_day", ""))


def _closes(ex, base: str) -> tuple[float, float] | None:
    """직전 닫힌 일자의 (현물 종가, perp 종가)."""
    out = []
    for cat in ("spot", "linear"):
        r = _retry(ex.publicGetV5MarketKline,
                   {"category": cat, "symbol": f"{base}USDT", "interval": "D", "limit": 3})
        if not r:
            return None
        rows = sorted(((int(x[0]), float(x[4])) for x in r.get("result", {}).get("list", [])),
                      key=lambda z: z[0])
        today = int(pd.Timestamp.now(tz="utc").normalize().timestamp() * 1000)
        rows = [c for ts, c in rows if ts < today]
        if not rows:
            return None
        out.append(rows[-1])
    return out[0], out[1]          # (spot, perp)


def run_day(state: CarryPaperState, ex, cfg: LedgerConfig) -> tuple[CarryPaperState, list[str]]:
    """직전 닫힌 일자 하나를 처리한다."""
    events: list[str] = []
    day = (pd.Timestamp.now(tz="utc").normalize() - pd.Timedelta(days=1))

    # --- 시장 상태 수집 ---
    carry, adv, basis, fund_day = {}, {}, {}, {}
    for b in BASES:
        d = _daily_funding(ex, f"{b}USDT", cfg.lookback_days)
        if d is not None and len(d.dropna()) >= cfg.lookback_days * cfg.min_valid_ratio:
            vals = np.sort(d.dropna().to_numpy())
            k = int(len(vals) * 0.10)
            if k > 0 and len(vals) - 2 * k >= 3:
                vals = vals[k:len(vals) - k]
            carry[b] = float(np.mean(vals)) * 365
            fund_day[b] = float(d.iloc[-1]) if len(d) else np.nan
        adv[b] = _adv_thin_leg(ex, b)
        px = _closes(ex, b)
        if px:
            basis[b] = (px[1] - px[0]) / px[0]

    eligible = [(b, adv[b]) for b in BASES
                if adv.get(b, 0) >= cfg.min_adv_usd and b in carry and b in basis]
    universe = [b for b, _ in sorted(eligible, key=lambda z: -z[1])[:2]]

    # --- 1) 보유 포지션 정산: 펀딩 − Δ베이시스, 비중 기준 ---
    for s, p in list(state.positions.items()):
        f = fund_day.get(s, np.nan)
        b_now = basis.get(s, np.nan)
        if np.isnan(f) or np.isnan(b_now):
            state.equity -= p["weight"] * 2 * cfg.leg_cost      # fail-closed 청산 (2배 비용)
            events.append(f"{s}:force_exit(데이터결측)")
            state.positions.pop(s)
            continue
        pnl = p["weight"] * (f - (b_now - p["basis"]))
        state.equity += pnl
        p["basis"] = b_now

    # --- 2) 청산 판정 (등록 규칙: 초과 < exit_ann, 최소보유 우선) ---
    for s, p in list(state.positions.items()):
        excess = carry.get(s, np.nan) - cfg.cash_rate
        held_days = (day - pd.Timestamp(p["opened"], tz="utc")).days
        if np.isnan(excess):
            continue
        if excess < cfg.exit_ann or (held_days >= cfg.min_hold_days
                                     and excess < cfg.hurdle_ann and s not in universe):
            state.equity -= p["weight"] * cfg.leg_cost
            events.append(f"{s}:exit(초과 {excess*100:.2f}%)")
            state.positions.pop(s)

    # --- 3) 진입 판정 (ADV상위2 안에서 허들 초과) ---
    for s in universe:
        if s in state.positions or len(state.positions) >= cfg.max_positions:
            continue
        excess = carry.get(s, np.nan) - cfg.cash_rate
        if not np.isnan(excess) and excess >= cfg.hurdle_ann:
            w = cfg.target_spot_fraction / cfg.max_positions
            state.equity -= w * cfg.leg_cost
            state.positions[s] = dict(weight=w, opened=str(day.date()), basis=basis[s])
            events.append(f"{s}:enter(초과 {excess*100:.2f}%, 비중 {w:.2f})")

    # --- 4) 장외 현금수익 (거래소 노출 = 현물비중 x 1.7) ---
    on_ex = sum(p["weight"] for p in state.positions.values()) * (1 + cfg.exchange_collateral_ratio)
    state.equity += max(0.0, 1.0 - on_ex) * cfg.cash_rate / 365.0
    state.last_day = str(day.date())

    top = " ".join(f"{b}:{carry.get(b, float('nan'))*100:+.1f}%" for b in universe)
    events.append(f"universe[{top}]")
    return state, events


def main() -> None:
    """일 1회 실행 (멱등)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = LedgerConfig()
    day = str((pd.Timestamp.now(tz="utc").normalize() - pd.Timedelta(days=1)).date())
    if HIST.exists():
        h = pd.read_csv(HIST)
        if len(h) and str(h["day"].iloc[-1]) == day:
            logger.info("%s 이미 처리됨 — 종료", day)
            return
    state = (CarryPaperState.from_dict(json.loads(STATE.read_text()))
             if STATE.exists() else CarryPaperState())
    ex = ccxt.bybit({"enableRateLimit": True})
    state, events = run_day(state, ex, cfg)
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state.to_dict(), indent=2, default=float))
    row = pd.DataFrame([dict(day=day, equity=round(state.equity, 8),
                             n_pos=len(state.positions), events="; ".join(events))])
    row.to_csv(HIST, mode="a", header=not HIST.exists(), index=False)
    logger.info("%s 처리: 자본 %.6f, 포지션 %d — %s", day, state.equity,
                len(state.positions), "; ".join(events))


if __name__ == "__main__":
    main()
