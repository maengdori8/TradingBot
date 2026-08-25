from __future__ import annotations

"""수량 원장 기반 델타중립 캐리 엔진.

Codex 7라운드 결함 지적을 전면 반영한 판본.

핵심 설계:
- 유니버스는 **전일 ADV 상위 N으로 먼저 확정**한다. 캐리 허들은 그 안에서만 적용한다.
- 현금은 거래소 담보(cash_exchange)와 장외(cash_offvenue)로 분리한다. 장외 자금만
  무위험 수익을 얻고, 거래소 담보만 증거금으로 인정된다.
- 리스크 조치는 상태를 가진다(latching). REDUCE 이후에는 모든 불변식이 연속으로
  통과할 때까지 증액이 금지된다. BLOCK_ADD·REBALANCE도 실제로 강제된다.
- 결측(펀딩·가격·차입)은 0이 아니라 **강제청산 사유**다.
- 평가 구간은 평평하게 시작한다(워밍업 구간에서는 신호만 계산하고 거래하지 않는다).
- 종료 시 잔여 포지션을 청산하고 비용을 부과한다.
- reverse(현물 차입 공매도)는 **부채 인지형 리스크 모델이 생기기 전까지 비활성**이다.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from carrybot.risk.invariants import Action, Limits, check_invariants, worst_action
from carrybot.risk.margin import AccountState, CarryLeg, StressScenario

logger = logging.getLogger(__name__)

FORWARD, REVERSE = 1, -1


class RiskState(str, Enum):
    """리스크 조치 래치 상태."""

    NORMAL = "normal"       # 정상 — 증액 가능
    BLOCKED = "blocked"     # 증액 금지, 보유 유지
    REDUCING = "reducing"   # 감축 중 — 회복 확인 전까지 증액 금지
    HALTED = "halted"       # 긴급 정지 — 수동 리셋 전까지 신규 금지


@dataclass
class Position:
    """한 심볼의 델타중립 포지션 (수량 단위)."""

    symbol: str
    direction: int
    units: float
    spot_entry: float
    perp_entry: float
    opened_at: pd.Timestamp
    borrow_accrued: float = 0.0

    def spot_value(self, spot: float) -> float:
        """현물 다리의 부호 있는 가치 (reverse면 부채)."""
        return round(self.direction * self.units * spot, 10)

    def perp_upl(self, mark: float) -> float:
        """perp 다리의 미실현손익."""
        return round(self.direction * self.units * (self.perp_entry - mark), 10)


@dataclass
class Book:
    """캐리 계좌 원장. 거래소 자금과 장외 자금을 분리한다."""

    positions: dict[str, Position] = field(default_factory=dict)
    cash_exchange: float = 0.0      # 거래소 스테이블 (증거금으로 인정)
    cash_offvenue: float = 1.0      # 장외 현금 (무위험 수익, 증거금 아님)
    risk_state: RiskState = RiskState.NORMAL
    clear_cycles: int = 0           # 연속 무위반 사이클 수

    @property
    def cash(self) -> float:
        """총 현금."""
        return round(self.cash_exchange + self.cash_offvenue, 10)

    def equity(self, spot: dict[str, float], mark: dict[str, float]) -> float:
        """경제적 자기자본."""
        e = self.cash
        for s, p in self.positions.items():
            if s in spot and s in mark:
                e += p.spot_value(spot[s]) + p.perp_upl(mark[s])
        return round(e, 10)

    def spot_notional(self, spot: dict[str, float]) -> float:
        """현물 명목 (절대값)."""
        return round(sum(abs(p.units) * spot[s] for s, p in self.positions.items() if s in spot), 10)

    def to_account_state(self, spot: dict[str, float], mark: dict[str, float],
                         mmr: dict[str, float] | None = None) -> AccountState:
        """리스크 모듈용 상태로 변환한다.

        거래소 담보만 stable_collateral로 인정한다. reverse는 부채 구조가 달라
        이 모델로 표현할 수 없으므로 엔진 차원에서 비활성이다.
        """
        legs = []
        for s, p in self.positions.items():
            if s not in spot or s not in mark:
                continue
            if p.direction != FORWARD:
                raise NotImplementedError("reverse 포지션은 부채 인지형 리스크 모델이 필요합니다")
            legs.append(CarryLeg(s, p.units, p.units, spot[s], mark[s], p.perp_entry,
                                 base_mmr=(mmr or {}).get(s, 0.0033)))
        # 차입(음수 off-venue)은 거래소 부채이므로 순담보에서 차감한다
        stable_net = self.cash_exchange + min(0.0, self.cash_offvenue)
        return AccountState(legs=legs, stable_collateral=stable_net)


@dataclass(frozen=True)
class LedgerConfig:
    """원장 엔진 설정 (사전등록 대상)."""

    universe_top_n: int = 2             # 전일 ADV 상위 N — 유니버스 (펀딩 미참조)
    max_positions: int = 2              # 동시 보유 상한 (≤ universe_top_n)
    lookback_days: int = 30
    min_hold_days: int = 30
    cost_multiple: float = 2.0
    cash_rate: float = 0.04
    target_spot_fraction: float = 0.50  # 운용 목표
    add_stop_fraction: float = 0.60     # 이 위로는 증액 금지
    max_spot_fraction: float = 0.65     # 하드 한도 (불변식)
    no_trade_band: float = 0.10         # 목표 대비 이탈 허용폭 (0.50x1.10=0.55 < 0.60)
    exchange_collateral_ratio: float = 0.70   # 거래소 스테이블 담보 / 현물명목
    notional_multiplier: float = 1.0          # 명목 배수 (>1이면 USDT 차입 레버리지)
    usdt_borrow_rate: float = 0.05            # USDT 차입 연율 (오늘 4.74% — PIT 아님, 민감도 필수)
    transfer_fee: float = 0.0005              # 거래소<->장외 이체 비용 (편도)
    collateral_band: float = 0.20             # 담보가 목표에서 이만큼 벗어나야 이체
    min_adv_usd: float = 2e7
    adv_participation: float = 0.01
    min_listing_age_days: int = 180
    delist_buffer_days: int = 30
    spot_fee: float = 0.0010
    perp_fee: float = 0.0006
    slippage: float = 0.0002
    allow_reverse: bool = False
    deleverage_step: float = 0.25
    recovery_cycles: int = 3            # REDUCE 해제에 필요한 연속 무위반 사이클
    record_stress: bool = False         # 일별 스트레스 헤드룸 기록 (연구용)
    min_valid_ratio: float = 0.90

    def __post_init__(self) -> None:
        """설정 정합성을 검증한다."""
        if self.target_spot_fraction * (1 + self.no_trade_band) > self.add_stop_fraction:
            raise ValueError("무거래 밴드 상단이 증액정지선을 넘습니다")
        if self.add_stop_fraction > self.max_spot_fraction:
            raise ValueError("증액정지선이 하드 한도를 넘습니다")
        if self.max_positions > self.universe_top_n:
            raise ValueError("보유 상한이 유니버스 크기를 넘습니다")
        # 거래소 계정 내부에서 현물/자기자본 = 1/(1+r) <= max_spot_fraction 이어야 한다
        implied = 1.0 / (1.0 + self.exchange_collateral_ratio)
        if implied > self.max_spot_fraction:
            raise ValueError(
                f"담보비율 {self.exchange_collateral_ratio}에서 거래소 현물비중 {implied:.3f}가 "
                f"한도 {self.max_spot_fraction}를 넘습니다")
        if self.notional_multiplier < 1.0:
            raise ValueError("명목 배수는 1 이상이어야 합니다")
        if self.allow_reverse:
            raise NotImplementedError("reverse는 부채 인지형 리스크 모델 완성 전까지 금지")

    @property
    def leg_cost(self) -> float:
        """양다리 편도 총비용."""
        return self.spot_fee + self.perp_fee + 2 * self.slippage

    @property
    def hurdle_ann(self) -> float:
        """현금 대비 초과캐리 진입 허들."""
        return self.cost_multiple * (2.0 * self.leg_cost) * 365.0 / self.min_hold_days

    @property
    def exit_ann(self) -> float:
        """청산 임계."""
        return -(2.0 * self.leg_cost) * 365.0 / self.min_hold_days


def _cost(units: float, spot: float, mark: float, cfg: LedgerConfig) -> float:
    """양다리 거래 비용."""
    return abs(units) * (spot * (cfg.spot_fee + cfg.slippage)
                         + mark * (cfg.perp_fee + cfg.slippage))


@dataclass
class LedgerResult:
    """원장 시뮬레이션 결과."""

    daily: pd.DataFrame
    config: LedgerConfig
    trades: pd.DataFrame
    episodes: pd.DataFrame

    @property
    def excess(self) -> pd.Series:
        """현금 대비 초과 일별수익."""
        return self.daily["ret"] - self.config.cash_rate / 365.0


def _episodes(daily: pd.DataFrame, cash_rate: float) -> pd.DataFrame:
    """평→평 활성화 에피소드를 추출한다 (부트스트랩 단위).

    진입일과 청산일의 비용·수익이 경계 행에 실리므로 앞뒤 1행씩 포함한다.
    수익은 현금 대비 '초과'로 계산한다 (원시 수익이 아니다).
    """
    on = (daily["n_pos"] > 0).to_numpy()
    n = len(on)
    rows, i = [], 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        lo, hi = max(0, i - 1), min(n, j + 1)          # 경계 포함
        seg = daily.iloc[lo:hi]
        exc = (seg["ret"] - cash_rate / 365.0)
        rows.append(dict(start=seg.index[0], end=seg.index[-1], days=int(j - i),
                         excess=float(exc.sum()),
                         equity_change=float((1 + seg["ret"]).prod() - 1)))
        i = j
    return pd.DataFrame(rows)


def simulate(panels: dict, uni: pd.DataFrame, cfg: LedgerConfig,
             borrow: pd.DataFrame | None = None, start=None, end=None,
             limits: Limits | None = None) -> LedgerResult:
    """수량 원장으로 캐리 포트폴리오를 시점보존 시뮬레이션한다.

    Args:
        panels: build_panels() 산출물.
        uni: 시점보존 유니버스 메타.
        cfg: 엔진 설정.
        borrow: 시점보존 차입금리 (현재는 reverse 비활성이라 미사용).
        start, end: 평가 구간. start 이전은 신호 워밍업 전용이며 거래하지 않는다.

    Returns:
        LedgerResult.
    """
    from carrybot.research.carry import CarryConfig, _eligibility, _signal

    spotP, perpP, fd, syms = panels["spot_close"], panels["perp_close"], panels["fd"], panels["syms"]
    adv_lag = panels["adv"].shift(1)
    idx = panels["idx"]

    def _utc(v, default):
        """tz 유무와 무관하게 UTC로 정규화한다."""
        if v is None:
            return default
        ts = pd.Timestamp(v)
        return ts.tz_localize("utc") if ts.tzinfo is None else ts.tz_convert("utc")

    eval_start = _utc(start, idx[0])
    eval_end = _utc(end, idx[-1])
    warm = eval_start - pd.Timedelta(days=cfg.lookback_days + 5)
    idx = idx[(idx >= warm) & (idx <= eval_end)]

    ccfg = CarryConfig(lookback_days=cfg.lookback_days, min_valid_ratio=cfg.min_valid_ratio,
                       min_listing_age_days=cfg.min_listing_age_days,
                       min_adv_usd=cfg.min_adv_usd, delist_buffer_days=cfg.delist_buffer_days)
    sig = _signal(fd.loc[idx], ccfg)
    elig = _eligibility(idx, syms, uni, adv_lag.loc[idx], ccfg)

    book = Book(cash_exchange=0.0, cash_offvenue=1.0)
    limits = limits or Limits(max_spot_to_equity=cfg.max_spot_fraction)
    rows, trades = [], []
    prev_eq = 1.0

    def close_position(t, s, spot, mark, reason):
        """포지션을 전량 청산하고 비용을 부과한다."""
        p = book.positions.pop(s)
        px_s, px_p = spot.get(s), mark.get(s)
        if px_s is None or px_p is None:      # 가격 결측: 최후 관측가로 청산하되 비용을 2배 부과
            px_s, px_p = p.spot_entry, p.perp_entry
            pen = 2.0
        else:
            pen = 1.0
        book.cash_exchange += p.spot_value(px_s) + p.perp_upl(px_p)
        c = _cost(p.units, px_s, px_p, cfg) * pen
        book.cash_exchange -= c
        trades.append(dict(ts=t, sym=s, action=reason, units=p.units, cost=c))

    for t in idx:
        trading = t >= eval_start
        spot = {s: v for s, v in spotP.loc[t].items() if not np.isnan(v)}
        mark = {s: v for s, v in perpP.loc[t].items() if not np.isnan(v)}
        f_t, e_t, s_t = fd.loc[t], elig.loc[t], sig.loc[t]

        # --- 1) 펀딩 정산 (전일 포지션). 결측은 0이 아니라 청산 사유로 표시 ---
        funding_pnl = 0.0
        stale = []
        for s, p in book.positions.items():
            if s not in spot or s not in mark or np.isnan(f_t.get(s, np.nan)):
                stale.append(s)
                continue
            funding_pnl += p.direction * f_t[s] * p.units * mark[s]
        book.cash_exchange += funding_pnl

        # --- 2) 결측·부적격 강제청산 (fail-closed) ---
        for s in list(book.positions):
            if s in stale or not e_t.get(s, False):
                close_position(t, s, spot, mark, "force_exit")

        # --- 3) 리스크 검사와 래치 ---
        act = Action.OK
        if book.positions:
            viol = check_invariants(book.to_account_state(spot, mark), limits)
            act = worst_action(viol)
            if act is Action.EMERGENCY:
                book.risk_state = RiskState.HALTED
                for s in list(book.positions):
                    close_position(t, s, spot, mark, "emergency_flat")
            elif act in (Action.REDUCE, Action.REBALANCE):
                book.risk_state = RiskState.REDUCING
                book.clear_cycles = 0
                for s, p in list(book.positions.items()):
                    cut = p.units * cfg.deleverage_step
                    book.cash_exchange += (p.direction * cut * spot[s]
                                           + p.direction * cut * (p.perp_entry - mark[s]))
                    c = _cost(cut, spot[s], mark[s], cfg)
                    book.cash_exchange -= c
                    trades.append(dict(ts=t, sym=s, action="deleverage", units=cut, cost=c))
                    p.units -= cut
                    if p.units <= 1e-12:
                        book.positions.pop(s)
            elif act is Action.BLOCK_ADD:
                if book.risk_state is not RiskState.HALTED:
                    book.risk_state = RiskState.BLOCKED
                book.clear_cycles = 0
            else:
                book.clear_cycles += 1
                if (book.risk_state in (RiskState.BLOCKED, RiskState.REDUCING)
                        and book.clear_cycles >= cfg.recovery_cycles):
                    book.risk_state = RiskState.NORMAL
        else:
            book.clear_cycles += 1
            if book.risk_state is RiskState.REDUCING or book.risk_state is RiskState.BLOCKED:
                if book.clear_cycles >= cfg.recovery_cycles:
                    book.risk_state = RiskState.NORMAL

        eq = book.equity(spot, mark)

        # --- 4) 유니버스: 전일 ADV 상위 N (펀딩 미참조) ---
        investable = [s for s in syms if e_t.get(s, False) and s in spot and s in mark
                      and s not in stale and not np.isnan(f_t.get(s, np.nan))]
        pool = adv_lag.loc[t][investable].dropna()
        universe = list(pool.nlargest(min(cfg.universe_top_n, len(pool))).index)

        # --- 5) 유니버스 안에서만 허들 적용 ---
        keep: dict[str, int] = {}
        if trading:
            for s, p in book.positions.items():     # 잠금·히스테리시스 우선
                # ADV 순위에서 밀려나도 '적격'하면 최소보유는 지킨다 (등록된 규칙)
                if not e_t.get(s, False) or s in stale or np.isnan(s_t[s]):
                    continue
                excess = s_t[s] - cfg.cash_rate
                locked = (t - p.opened_at).days < cfg.min_hold_days
                if excess >= cfg.exit_ann and (locked or excess >= cfg.hurdle_ann):
                    keep[s] = FORWARD
            if book.risk_state is RiskState.NORMAL:
                for s in universe:
                    if len(keep) >= cfg.max_positions or s in keep or np.isnan(s_t[s]):
                        continue
                    if s_t[s] - cfg.cash_rate >= cfg.hurdle_ann:
                        keep[s] = FORWARD
        else:
            keep = {s: FORWARD for s in book.positions}

        # --- 6) 체결 ---
        for s in list(book.positions):
            if s not in keep:
                close_position(t, s, spot, mark, "exit")
        if keep and trading:
            budget = max(0.0, eq) * cfg.target_spot_fraction * cfg.notional_multiplier
            per = budget / len(keep)
            for s in keep:
                cap = cfg.adv_participation * adv_lag.loc[t, s]
                tn = min(per, cap if np.isfinite(cap) else per)
                tu = tn / spot[s]
                cur = book.positions.get(s)
                du = tu - (cur.units if cur else 0.0)
                if cur is not None and abs(du) < cfg.no_trade_band * tu:
                    continue
                if du > 0 and book.risk_state is not RiskState.NORMAL:
                    continue                        # 증액 금지 상태
                if du > 0 and (book.spot_notional(spot) + du * spot[s]
                               > eq * cfg.add_stop_fraction * cfg.notional_multiplier):
                    continue                        # 증액 정지선
                if abs(du) * spot[s] < 1e-9:
                    continue
                c = _cost(du, spot[s], mark[s], cfg)
                book.cash_exchange -= c + du * spot[s]
                if cur is None:
                    book.positions[s] = Position(s, FORWARD, tu, spot[s], mark[s], t)
                else:
                    if du > 0:
                        w = cur.units / (cur.units + du)
                        cur.perp_entry = cur.perp_entry * w + mark[s] * (1 - w)
                    else:
                        book.cash_exchange += (-du) * (cur.perp_entry - mark[s])
                    cur.units = tu
                trades.append(dict(ts=t, sym=s, action="adjust", units=du, cost=c))

        # --- 7) 자금 배치: 거래소에는 현물+필수담보만, 나머지는 장외 ---
        sn = book.spot_notional(spot)
        need_ex = sn * cfg.exchange_collateral_ratio       # 필수 스테이블 담보
        total_cash = book.cash_exchange + book.cash_offvenue
        want = max(need_ex, 0.0)          # 부족분은 USDT 차입(음수 off-venue)으로 조달
        drift = abs(want - book.cash_exchange)
        if drift > max(cfg.collateral_band * max(need_ex, 1e-12), 1e-12):
            book.cash_offvenue -= (want - book.cash_exchange)
            book.cash_exchange = want
            fee = drift * cfg.transfer_fee                 # 이체는 공짜가 아니다
            book.cash_offvenue -= fee
            trades.append(dict(ts=t, sym="-", action="transfer", units=drift, cost=fee))
        if trading:
            # 장외 현금이 음수 = USDT 차입 상태 → 현금수익 대신 차입이자 지불
            rate = cfg.cash_rate if book.cash_offvenue >= 0 else cfg.usdt_borrow_rate
            book.cash_offvenue += book.cash_offvenue * rate / 365.0

        if t == idx[-1] and book.positions:              # 종료 청산 (일별 수익에 포함)
            for s in list(book.positions):
                close_position(t, s, spot, mark, "terminal")
            sn = book.spot_notional(spot)

        eq = book.equity(spot, mark)
        on_ex = (sn + book.cash_exchange) / eq if eq > 0 else 0.0

        headroom = float("inf")
        if cfg.record_stress and book.positions:
            st_now = book.to_account_state(spot, mark)
            def _mmr(x: float) -> float:
                sc = StressScenario("h", x, 0.15, 0.70, 2.0, 0.02, 0.01, 0.90)
                return st_now.maintenance_ratio(sc)
            if _mmr(0.0) > 0.50:
                headroom = 0.0
            else:
                lo_x, hi_x = 0.0, 20.0
                if _mmr(hi_x) <= 0.50:
                    headroom = float("inf")
                else:
                    for _ in range(50):
                        mid_x = (lo_x + hi_x) / 2
                        if _mmr(mid_x) > 0.50:
                            hi_x = mid_x
                        else:
                            lo_x = mid_x
                    headroom = lo_x
        rows.append(dict(ts=t, equity=eq, ret=(eq / prev_eq - 1.0) if prev_eq > 0 else 0.0,
                         spot_frac=sn / eq if eq > 0 else 0.0, on_exchange=on_ex,
                         n_pos=len(book.positions), funding=funding_pnl,
                         borrowed=max(0.0, -book.cash_offvenue) / eq if eq > 0 else 0.0,
                         stress_headroom=headroom,
                         risk=book.risk_state.value, action=act.value))
        prev_eq = eq

    daily = pd.DataFrame(rows).set_index("ts")
    daily = daily[daily.index >= eval_start]
    return LedgerResult(daily=daily, config=cfg, trades=pd.DataFrame(trades),
                        episodes=_episodes(daily, cfg.cash_rate))
