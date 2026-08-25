from __future__ import annotations

"""자본구조 최적화 — 스트레스를 견디는 최대 현물 명목을 푼다.

임의의 '현물 65%' 같은 관행적 상한 대신, 실제 Bybit 증거금 티어와 스트레스
시나리오로부터 허용 명목을 역산한다. 이렇게 하면 (1) 근거가 명시되고
(2) 시나리오를 강화/완화할 때 자본구조가 자동으로 따라온다.

Bybit UTA 전제:
- 현물과 스테이블이 모두 계좌 담보이며, 각각 담보인정비율(헤어컷)이 적용된다.
- 무기한 숏의 미실현손익은 담보에 1:1로 반영된다(이익도 손실도).
- 따라서 랠리 구간에서 '현물 헤어컷 < 1'인 만큼 구조적 결손이 발생한다.
"""

import logging
from dataclasses import dataclass

from carrybot.risk.margin import AccountState, CarryLeg, StressScenario

logger = logging.getLogger(__name__)

# Bybit 실측 리스크 티어 (publicGetV5MarketRiskLimit, 2026-08 조회)
BYBIT_MMR_TIER1 = 0.0033      # BTCUSDT/ETHUSDT 최저 티어 유지증거금률
BYBIT_MMR_TIER2 = 0.0050      # 명목 30만 USDT 초과


@dataclass(frozen=True)
class CapitalPlan:
    """자본구조 해."""

    spot_fraction: float          # 현물 명목 / 자기자본
    stable_fraction: float        # 스테이블 담보 / 자기자본
    worst_stressed_mmr: float     # 최악 시나리오 유지증거금률
    binding_scenario: str         # 제약을 만든 시나리오

    @property
    def notional_multiple(self) -> float:
        """자기자본 대비 캐리 명목 배수 (= 수익 배수)."""
        return self.spot_fraction


def stressed_mmr_for(spot_fraction: float, scenario: StressScenario,
                     mmr: float = BYBIT_MMR_TIER1,
                     collateral_ratio: float = 0.90,
                     entry_price: float = 100.0) -> float:
    """주어진 현물비중에서의 스트레스 유지증거금률을 계산한다.

    Args:
        spot_fraction: 현물 명목 / 자기자본 (진입 시점 기준).
        scenario: 스트레스 시나리오.
        mmr: 기본 유지증거금률.
        collateral_ratio: 평시 현물 담보인정비율.
        entry_price: 정규화 진입가.

    Returns:
        스트레스 유지증거금률 (낮을수록 안전).
    """
    units = spot_fraction / entry_price          # 자기자본 1 기준
    stable = 1.0 - spot_fraction
    if units <= 0:
        return 0.0
    state = AccountState(
        legs=[CarryLeg("X", units, units, entry_price, entry_price, entry_price,
                       base_mmr=mmr, collateral_ratio=collateral_ratio)],
        stable_collateral=stable,
    )
    return state.maintenance_ratio(scenario)


def solve_max_spot(scenarios: tuple[StressScenario, ...], max_stressed_mmr: float = 0.50,
                   mmr: float = BYBIT_MMR_TIER1, collateral_ratio: float = 0.90,
                   lo: float = 0.0, hi: float = 1.0, tol: float = 1e-4) -> CapitalPlan:
    """모든 스트레스 시나리오를 만족하는 최대 현물비중을 이분탐색으로 찾는다.

    Args:
        scenarios: 반드시 통과해야 하는 시나리오들.
        max_stressed_mmr: 스트레스 유지증거금률 상한.
        mmr: 기본 유지증거금률.
        collateral_ratio: 평시 현물 담보인정비율.
        lo, hi: 탐색 구간.
        tol: 수렴 허용오차.

    Returns:
        CapitalPlan — 허용 현물비중과 제약을 만든 시나리오.
    """
    def ok(f: float) -> tuple[bool, float, str]:
        worst, name = 0.0, ""
        for sc in scenarios:
            r = stressed_mmr_for(f, sc, mmr, collateral_ratio)
            if r > worst:
                worst, name = r, sc.name
        return worst <= max_stressed_mmr, worst, name

    if not ok(lo)[0]:
        return CapitalPlan(0.0, 1.0, ok(lo)[1], ok(lo)[2])
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if ok(mid)[0]:
            lo = mid
        else:
            hi = mid
    _, worst, name = ok(lo)
    return CapitalPlan(round(lo, 6), round(1.0 - lo, 6), worst, name)
