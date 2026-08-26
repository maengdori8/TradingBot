from __future__ import annotations

"""Track D — 1h 스윙 돌파 슬리브 (BRK96, 페이퍼 전용).

경고: 격자 8셀 중 유일 생존(선택 할인 필요). Sharpe 0.57, MDD 49%,
드로다운 20% 초과 구간이 시간의 37%. 2021/22/25는 평/손실 — 레짐 의존.
승급 근거 사용 금지. 백테스트·라이브가 같은 step_bar()를 사용한다.

규칙(고정): 96h 채널 돌파 진입(스탑주문 체결 모델), 48h 반대채널 또는
6xATR(24) 트레일 청산, 리스크 2%/거래, 그로스 캡 10x, 일손실 -5% 정지,
비용 편도 8bp(taker+슬립), 펀딩 일 1회 반영.
"""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

COST_SIDE = 0.0008
ENTRY_N, EXIT_N, ATR_N, ATR_MULT = 96, 48, 24, 6.0
RISK, GROSS_CAP, DAILY_HALT = 0.02, 10.0, -0.05


@dataclass
class SwingPos:
    """한 심볼 포지션."""

    d: int
    u: float
    e: float
    stop: float


@dataclass
class SwingState:
    """직렬화 가능한 상태."""

    equity: float = 1.0
    positions: dict = field(default_factory=dict)
    atr: dict = field(default_factory=dict)
    day: str = ""
    day_eq: float = 1.0
    halted: bool = False
    last_ts: int = 0

    def to_dict(self) -> dict:
        return dict(equity=self.equity, atr=self.atr, day=self.day, day_eq=self.day_eq,
                    halted=self.halted, last_ts=self.last_ts,
                    positions={s: asdict(p) for s, p in self.positions.items()})

    @classmethod
    def from_dict(cls, d: dict) -> "SwingState":
        st = cls(equity=d["equity"], atr=dict(d.get("atr", {})), day=d.get("day", ""),
                 day_eq=d.get("day_eq", d["equity"]), halted=d.get("halted", False),
                 last_ts=d.get("last_ts", 0))
        st.positions = {s: SwingPos(**p) for s, p in d.get("positions", {}).items()}
        return st


@dataclass(frozen=True)
class Bar1h:
    """닫힌 1h 봉 + 채널 레벨(직전 봉까지) + 당일 펀딩(일 첫 봉에만)."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    ehi: float      # 직전 96h 최고
    elo: float
    xhi: float      # 직전 48h 최고
    xlo: float
    funding: float  # 이 봉이 새 UTC 일의 첫 봉이면 전일 펀딩, 아니면 0/nan


def step_bar(state: SwingState, bars: dict[str, Bar1h], day_key: str) -> list[dict]:
    """닫힌 1h 봉 1개(전 심볼)를 처리한다. 백테스트=라이브 공용 경로."""
    fills: list[dict] = []
    if day_key != state.day:
        state.day, state.halted = day_key, False
        state.day_eq = _mtm(state, bars)
        for s, p in state.positions.items():           # 펀딩 (일 1회)
            b = bars.get(s)
            if b is not None and b.funding == b.funding and not np.isnan(b.close):
                state.equity -= p.d * b.funding * p.u * b.close

    for s, b in bars.items():
        if any(np.isnan(x) for x in (b.open, b.high, b.low, b.close)):
            continue
        tr = max(b.high - b.low, abs(b.high - b.close), abs(b.low - b.close))
        a0 = state.atr.get(s)
        state.atr[s] = tr if a0 is None else a0 + (tr - a0) / ATR_N
        a = state.atr[s]
        p = state.positions.get(s)
        if p:
            exit_px = None
            if p.d > 0:
                lvl = max(p.stop, b.xlo)
                if b.low <= lvl:
                    exit_px = min(lvl, b.open)
            else:
                lvl = min(p.stop, b.xhi)
                if b.high >= lvl:
                    exit_px = max(lvl, b.open)
            if exit_px is not None:
                pnl = p.u * (exit_px - p.e) * p.d
                state.equity += pnl - p.u * exit_px * COST_SIDE
                fills.append(dict(sym=s, action="exit", price=exit_px, pnl=pnl))
                state.positions.pop(s)
                continue
            if p.d > 0:
                p.stop = max(p.stop, b.close - ATR_MULT * a)
            else:
                p.stop = min(p.stop, b.close + ATR_MULT * a)
            continue
        if state.halted or np.isnan(b.ehi) or np.isnan(b.elo) or a <= 0:
            continue
        gross = sum(pp.u * bars[ss].close for ss, pp in state.positions.items()
                    if ss in bars and not np.isnan(bars[ss].close))
        if gross >= GROSS_CAP * state.equity:
            continue
        d_ = 0
        if b.high > b.ehi:
            d_, fill = 1, max(b.open, b.ehi)
        elif b.low < b.elo:
            d_, fill = -1, min(b.open, b.elo)
        if not d_:
            continue
        stop = fill - d_ * ATR_MULT * a
        u = min(RISK * state.equity / abs(fill - stop),
                max(0.0, GROSS_CAP * state.equity - gross) / fill)
        if u <= 0:
            continue
        state.equity -= u * fill * COST_SIDE
        if (d_ > 0 and b.low <= stop) or (d_ < 0 and b.high >= stop):   # 같은봉 스탑(비관)
            pnl = u * (stop - fill) * d_
            state.equity += pnl - u * stop * COST_SIDE
            fills.append(dict(sym=s, action="same_bar_stop", price=stop, pnl=pnl))
            continue
        state.positions[s] = SwingPos(d_, u, fill, stop)
        fills.append(dict(sym=s, action="enter", price=fill, pnl=0.0, direction=d_))

    mtm = _mtm(state, bars)
    if not state.halted and state.day_eq > 0 and mtm / state.day_eq - 1 < DAILY_HALT \
            and state.positions:
        for s, p in list(state.positions.items()):
            b = bars.get(s)
            if b is None or np.isnan(b.close):
                continue
            pnl = p.u * (b.close - p.e) * p.d
            state.equity += pnl - p.u * b.close * COST_SIDE
            fills.append(dict(sym=s, action="daily_halt", price=b.close, pnl=pnl))
            state.positions.pop(s)
        state.halted = True
    state.last_ts = max(state.last_ts, max(b.ts for b in bars.values()))
    return fills


def _mtm(state: SwingState, bars: dict[str, Bar1h]) -> float:
    """시가평가."""
    v = state.equity
    for s, p in state.positions.items():
        b = bars.get(s)
        if b is not None and not np.isnan(b.close):
            v += p.u * (b.close - p.e) * p.d
    return v
