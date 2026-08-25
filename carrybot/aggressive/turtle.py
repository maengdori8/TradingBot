from __future__ import annotations

"""Track B — 터틀식 돌파 추세추종 (perp, 페이퍼 전용).

경고(사전 고지): 이 트랙에는 검증된 엣지가 없다. 2026-06 정직 WFO에서 방향성
전략군은 전부 기각됐고, 이 시스템의 과거 +7.9%/yr은 레짐 운(2023/2026 추세장)
이며 2025년 −19% 같은 해가 정상 범위다. 실거래 승급 근거로 사용 금지.

설계 원칙:
- 백테스트와 라이브가 같은 `step()` 함수를 쓴다 (불일치 구조적 차단).
- 결정은 '닫힌 일봉'만 사용. 진입은 스탑주문 모델(채널레벨 vs 시가 중 불리한 쪽),
  갭 스탑은 시가로 악화, 같은날 스탑아웃은 비관적 처리.
- 펀딩 반영: 롱은 양수 펀딩 지불, 숏은 수취.
- 하드캡: 거래당 리스크, 포트폴리오 히트, 그로스, 일손실 정지, 월손실 킬(래치).
"""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

ENTRY_COST, EXIT_COST = 0.0004, 0.0008


@dataclass(frozen=True)
class TurtleConfig:
    """Track B 설정 — 출판된 터틀 파라미터, 튜닝 금지."""

    entry_n: int = 20
    exit_n: int = 10
    atr_n: int = 20
    atr_mult: float = 2.0
    risk_pct: float = 0.02          # 거래당 자본 리스크
    heat_cap: float = 0.06          # 동시 총 리스크 상한
    gross_cap: float = 3.0          # 총명목/자본 상한
    daily_halt: float = -0.05       # 일손실 정지
    monthly_kill: float = -0.15     # 월손실 킬 (수동 리셋 래치)
    syms: tuple[str, ...] = ("BTC", "ETH", "SOL")


@dataclass
class TurtlePosition:
    """한 심볼의 포지션."""

    direction: int
    units: float
    entry: float
    stop: float
    risk_d: float


@dataclass
class TurtleState:
    """직렬화 가능한 전체 상태 (라이브 러너가 JSON으로 보존)."""

    equity: float = 1.0
    positions: dict = field(default_factory=dict)      # sym -> TurtlePosition dict
    atr: dict = field(default_factory=dict)            # sym -> float (EMA 상태)
    month_start_eq: float = 1.0
    month_key: str = ""
    killed: bool = False

    def to_dict(self) -> dict:
        """JSON 직렬화."""
        return dict(equity=self.equity, atr=self.atr, month_start_eq=self.month_start_eq,
                    month_key=self.month_key, killed=self.killed,
                    positions={s: asdict(p) if isinstance(p, TurtlePosition) else p
                               for s, p in self.positions.items()})

    @classmethod
    def from_dict(cls, d: dict) -> "TurtleState":
        """JSON 역직렬화."""
        st = cls(equity=d["equity"], atr=dict(d.get("atr", {})),
                 month_start_eq=d.get("month_start_eq", d["equity"]),
                 month_key=d.get("month_key", ""), killed=d.get("killed", False))
        st.positions = {s: TurtlePosition(**p) for s, p in d.get("positions", {}).items()}
        return st


@dataclass(frozen=True)
class Bar:
    """한 심볼의 닫힌 일봉 + 채널 레벨 (호출자가 닫힌 봉으로만 계산)."""

    open: float
    high: float
    low: float
    close: float
    entry_hi: float      # 직전 entry_n일 최고 (당일 제외)
    entry_lo: float
    exit_hi: float       # 직전 exit_n일 최고 (당일 제외)
    exit_lo: float
    funding: float       # 당일 정산 펀딩 합 (결측이면 nan)


def step(state: TurtleState, bars: dict[str, Bar], cfg: TurtleConfig,
         month_key: str) -> tuple[TurtleState, list[dict]]:
    """닫힌 일봉 하나를 처리한다. 백테스트와 라이브가 공유하는 유일한 경로.

    Args:
        state: 이전 상태 (변형됨).
        bars: 심볼별 닫힌 일봉.
        cfg: 설정.
        month_key: 'YYYY-MM' (월손실 킬 경계).

    Returns:
        (갱신된 상태, 체결 이벤트 목록).
    """
    fills: list[dict] = []
    if state.month_key != month_key:
        state.month_key, state.month_start_eq = month_key, state.equity

    # ATR 갱신 (EMA, 닫힌 봉)
    for s, b in bars.items():
        prev_c = getattr(b, "close", np.nan)
        tr = max(b.high - b.low, abs(b.high - b.close), abs(b.low - b.close))
        a0 = state.atr.get(s)
        state.atr[s] = tr if a0 is None else a0 + (tr - a0) / cfg.atr_n

    day_start = _mtm(state, bars)

    # --- 청산 (스탑/채널, 갭은 시가로 악화) + 펀딩 ---
    for s in list(state.positions):
        b = bars.get(s)
        p = state.positions[s]
        if b is None or any(np.isnan(x) for x in (b.open, b.high, b.low, b.close)):
            # 데이터 결측: fail-closed — 마지막 진입가 기준 청산 + 2배 비용
            state.equity -= p.units * p.entry * EXIT_COST * 2
            fills.append(dict(sym=s, action="force_exit", price=p.entry, pnl=0.0))
            state.positions.pop(s)
            continue
        if not np.isnan(b.funding):
            state.equity -= p.direction * b.funding * p.units * b.close
        exit_px = None
        if p.direction > 0:
            lvl = max(p.stop, b.exit_lo)
            if b.low <= lvl:
                exit_px = min(lvl, b.open)
        else:
            lvl = min(p.stop, b.exit_hi)
            if b.high >= lvl:
                exit_px = max(lvl, b.open)
        if exit_px is not None:
            pnl = p.units * (exit_px - p.entry) * p.direction
            state.equity += pnl - p.units * exit_px * EXIT_COST
            fills.append(dict(sym=s, action="exit", price=exit_px, pnl=pnl))
            state.positions.pop(s)

    # --- 킬/정지 판정 ---
    mtm = _mtm(state, bars)
    if state.killed:
        return state, fills
    if state.month_start_eq > 0 and mtm / state.month_start_eq - 1 < cfg.monthly_kill:
        _flatten(state, bars, fills, "monthly_kill")
        state.killed = True
        logger.warning("월손실 킬 발동 — 수동 리셋 전까지 정지")
        return state, fills
    if day_start > 0 and mtm / day_start - 1 < cfg.daily_halt:
        _flatten(state, bars, fills, "daily_halt")
        return state, fills

    # --- 진입 (스탑주문 모델) ---
    for s, b in bars.items():
        if s in state.positions or any(np.isnan(x) for x in (b.open, b.high, b.low)):
            continue
        if np.isnan(b.entry_hi) or np.isnan(b.entry_lo):
            continue
        a = state.atr.get(s)
        if not a or a <= 0:
            continue
        heat = sum(pp.risk_d * pp.units for pp in state.positions.values())
        if heat >= cfg.heat_cap * state.equity:
            continue
        gross = sum(pp.units * bars[ss].close for ss, pp in state.positions.items()
                    if ss in bars)
        if b.high > b.entry_hi:
            d_, fill = 1, max(b.open, b.entry_hi)
            stop = fill - cfg.atr_mult * a
        elif b.low < b.entry_lo:
            d_, fill = -1, min(b.open, b.entry_lo)
            stop = fill + cfg.atr_mult * a
        else:
            continue
        units = min(cfg.risk_pct * state.equity / (cfg.atr_mult * a),
                    max(0.0, cfg.gross_cap * state.equity - gross) / fill)
        if units <= 0:
            continue
        state.equity -= units * fill * ENTRY_COST
        # 같은날 스탑아웃 — 비관적 (진입 직후 역행 가정)
        if (d_ > 0 and b.low <= stop) or (d_ < 0 and b.high >= stop):
            pnl = units * (stop - fill) * d_
            state.equity += pnl - units * stop * EXIT_COST
            fills.append(dict(sym=s, action="same_day_stop", price=stop, pnl=pnl))
            continue
        state.positions[s] = TurtlePosition(d_, units, fill, stop, cfg.atr_mult * a)
        fills.append(dict(sym=s, action="enter", price=fill,
                          pnl=0.0, direction=d_))

    # 트레일링 스탑 갱신 (다음 봉에 적용)
    for s, p in state.positions.items():
        b = bars.get(s)
        if b is None or np.isnan(b.close):
            continue
        a = state.atr.get(s, 0.0)
        if p.direction > 0:
            p.stop = max(p.stop, b.close - cfg.atr_mult * a)
        else:
            p.stop = min(p.stop, b.close + cfg.atr_mult * a)
    return state, fills


def _mtm(state: TurtleState, bars: dict[str, Bar]) -> float:
    """시가평가 자본."""
    v = state.equity
    for s, p in state.positions.items():
        b = bars.get(s)
        if b is not None and not np.isnan(b.close):
            v += p.units * (b.close - p.entry) * p.direction
    return v


def _flatten(state: TurtleState, bars: dict[str, Bar], fills: list[dict], reason: str) -> None:
    """전량 청산."""
    for s in list(state.positions):
        p = state.positions.pop(s)
        b = bars.get(s)
        px = b.close if (b is not None and not np.isnan(b.close)) else p.entry
        pnl = p.units * (px - p.entry) * p.direction
        state.equity += pnl - p.units * px * EXIT_COST
        fills.append(dict(sym=s, action=reason, price=px, pnl=pnl))
