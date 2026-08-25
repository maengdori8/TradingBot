from __future__ import annotations

"""매 사이클 강제되는 하드 리스크 불변식.

'권고'가 아니라 '차단'이다. 위반 시 신규 진입을 막고 심각도에 따라 감축·긴급청산을
지시한다. 임의 완화는 금지한다.

Codex 검토 반영 수정:
- 히스테리시스가 죽어 있었다(최고 한도를 넘을 때만 평가되어 block_add가 발동 불가).
  → 단계별 임계값을 각각 독립 평가한다.
- max_spot_notional이 총명목·ADL 제약을 무시했다. → 모든 제약의 최소값을 취한다.
- perp 미실현손실 누적 한도를 추가한다. 델타중립이라도 랠리가 길면 숏 손실이
  1:1로 쌓이는 반면 현물 담보에는 헤어컷이 걸려 청산될 수 있다.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from carrybot.risk.margin import (
    DEFAULT_SCENARIOS,
    AccountState,
    adl_orphan_loss,
)

logger = logging.getLogger(__name__)


class Action(str, Enum):
    """불변식 위반 시 지시되는 조치. 뒤로 갈수록 강하다."""

    OK = "ok"
    BLOCK_ADD = "block_add"
    REBALANCE = "rebalance"
    REDUCE = "reduce"
    EMERGENCY = "emergency"


_SEVERITY = {a: i for i, a in enumerate(
    [Action.OK, Action.BLOCK_ADD, Action.REBALANCE, Action.REDUCE, Action.EMERGENCY])}


@dataclass(frozen=True)
class Limits:
    """하드 한도. 완화하려면 명시적 근거와 재검증이 필요하다."""

    max_unit_mismatch: float = 0.005
    max_spot_to_equity: float = 0.65
    min_stable_to_spot: float = 0.50
    max_gross_to_equity: float = 1.30
    max_stressed_mmr: float = 0.50
    max_orphan_loss: float = 0.30
    max_upl_to_equity: float = 0.25      # perp 미실현손실 ≤ 자기자본의 25% → 리밸런스
    mmr_block_add: float = 0.10
    mmr_rebalance: float = 0.15
    mmr_reduce: float = 0.20
    mmr_emergency: float = 0.30
    min_listing_age_days: int = 180
    min_days_to_delisting: int = 30


@dataclass(frozen=True)
class Violation:
    """단일 불변식 위반 기록."""

    rule: str
    observed: float
    limit: float
    action: Action

    def __str__(self) -> str:
        """사람이 읽는 위반 설명."""
        return f"{self.rule}: {self.observed:.4f} (한도 {self.limit:.4f}) → {self.action.value}"


def check_invariants(state: AccountState, limits: Limits | None = None) -> list[Violation]:
    """계좌 상태에 대해 모든 하드 불변식을 검사한다.

    Args:
        state: 검사 대상 계좌 상태.
        limits: 적용 한도. None이면 기본 한도.

    Returns:
        위반 목록. 비어 있으면 통과.
    """
    lim = limits or Limits()
    v: list[Violation] = []
    eq = state.equity

    if eq <= 0:
        return [Violation("equity_nonpositive", eq, 0.0, Action.EMERGENCY)]

    for leg in state.legs:
        if leg.unit_mismatch > lim.max_unit_mismatch:
            v.append(Violation(f"unit_mismatch[{leg.symbol}]", leg.unit_mismatch,
                               lim.max_unit_mismatch, Action.REBALANCE))

    if (ratio := state.spot_notional / eq) > lim.max_spot_to_equity:
        v.append(Violation("spot_to_equity", ratio, lim.max_spot_to_equity, Action.REDUCE))

    if state.spot_notional > 0:
        stable = state.stable_collateral / state.spot_notional
        if stable < lim.min_stable_to_spot:
            v.append(Violation("stable_to_spot", stable, lim.min_stable_to_spot, Action.BLOCK_ADD))

    if (gross := state.gross_notional / eq) > lim.max_gross_to_equity:
        v.append(Violation("gross_to_equity", gross, lim.max_gross_to_equity, Action.REDUCE))

    # 누적 미실현손실 — 랠리 중 청산의 주경로
    upl_loss = max(0.0, -state.perp_upl) / eq
    if upl_loss > lim.max_upl_to_equity:
        v.append(Violation("perp_upl_loss", upl_loss, lim.max_upl_to_equity, Action.REBALANCE))

    # 단계별 독립 평가 (구버전은 최고 한도 초과 시에만 평가되어 하위 단계가 죽어 있었다)
    mmr = state.maintenance_ratio()
    for thr, act in ((lim.mmr_emergency, Action.EMERGENCY), (lim.mmr_reduce, Action.REDUCE),
                     (lim.mmr_rebalance, Action.REBALANCE), (lim.mmr_block_add, Action.BLOCK_ADD)):
        if mmr >= thr:
            v.append(Violation("live_mmr", mmr, thr, act))
            break

    for sc in DEFAULT_SCENARIOS:
        smmr = state.maintenance_ratio(sc)
        if smmr > lim.max_stressed_mmr:
            act = Action.EMERGENCY if smmr == float("inf") else Action.REDUCE
            v.append(Violation(f"stressed_mmr[{sc.name}]", smmr, lim.max_stressed_mmr, act))

    if (ol := adl_orphan_loss(state)["loss_frac"]) > lim.max_orphan_loss:
        v.append(Violation("adl_orphan_loss", ol, lim.max_orphan_loss, Action.REDUCE))

    return v


def worst_action(violations: list[Violation]) -> Action:
    """위반 목록에서 가장 강한 조치를 고른다."""
    return max((x.action for x in violations), key=lambda a: _SEVERITY[a], default=Action.OK)


def max_spot_notional(equity: float, limits: Limits | None = None,
                      post_adl_drop: float = 0.35, unwind_cost: float = 0.02) -> float:
    """자기자본에서 허용되는 최대 현물 명목 — 모든 제약의 교집합.

    현물명목 S, 스테이블 = equity - S 로 두면 (진입 시점 UPL=0):
      1) S <= equity x max_spot_to_equity
      2) (equity - S)/S >= min_stable_to_spot        → S <= equity/(1 + min_stable_to_spot)
      3) 2S <= equity x max_gross_to_equity           → S <= equity x max_gross/2
      4) ADL 손실률 <= max_orphan_loss:
         loss = S x (drop + (1-drop) x unwind) / equity
    """
    lim = limits or Limits()
    if equity <= 0:
        return 0.0
    adl_unit = post_adl_drop + (1 - post_adl_drop) * unwind_cost
    caps = (
        equity * lim.max_spot_to_equity,
        equity / (1.0 + lim.min_stable_to_spot),
        equity * lim.max_gross_to_equity / 2.0,
        equity * lim.max_orphan_loss / adl_unit if adl_unit > 0 else float("inf"),
    )
    return round(min(caps), 8)
