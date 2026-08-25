from __future__ import annotations

"""델타중립 캐리 포지션의 증거금·청산 모델.

Bybit UTA(통합거래계정) 크로스마진 전제. 계좌 유지증거금률과 스트레스 시
최악 유지증거금률을 산출한다.

Codex 적대적 검토에서 확인된 결함을 수정한 판본:
- equity가 perp 미실현손익(UPL)을 무시하면 '랠리 중 자본 과대계상'이 발생한다.
  현물 이익만 세고 숏 손실을 빼지 않으므로, 정작 청산이 문제되는 상태에서
  안전해 보인다. → 경제적 자본과 담보조정 자본을 분리해 둘 다 계산한다.
- 펀딩 유출은 '유지증거금 가산'이 아니라 '자산 차감'이다.
- 긴급청산 비용은 현재 명목이 아니라 '스트레스 후 명목'에 부과해야 한다.
- 유지증거금률은 심볼·리스크티어별이므로 하드코딩하지 않는다.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressScenario:
    """스트레스 시나리오 정의. 모든 값은 비율(1.0 = 100%)."""

    name: str
    spot_shock: float
    mark_premium: float
    collateral_ratio_cap: float
    mmr_multiplier: float
    unwind_cost: float
    funding_debit: float
    stable_haircut: float

    def __post_init__(self) -> None:
        """시나리오 파라미터의 정합성을 검증한다."""
        if not 0 < self.collateral_ratio_cap <= 1:
            raise ValueError("담보인정비율 상한은 (0, 1] 범위여야 합니다")
        if self.mmr_multiplier < 1:
            raise ValueError("유지증거금 배수는 1 이상이어야 합니다")
        if not 0 < self.stable_haircut <= 1:
            raise ValueError("스테이블코인 평가는 (0, 1] 범위여야 합니다")


SHORT_SQUEEZE = StressScenario(
    name="short_squeeze", spot_shock=1.00, mark_premium=0.15,
    collateral_ratio_cap=0.70, mmr_multiplier=2.0, unwind_cost=0.02,
    funding_debit=0.01, stable_haircut=0.90,
)

CRASH = StressScenario(
    name="crash", spot_shock=-0.50, mark_premium=-0.05,
    collateral_ratio_cap=0.70, mmr_multiplier=2.0, unwind_cost=0.02,
    funding_debit=0.0, stable_haircut=0.90,
)

DEFAULT_SCENARIOS: tuple[StressScenario, ...] = (SHORT_SQUEEZE, CRASH)


@dataclass
class CarryLeg:
    """한 심볼의 델타중립 캐리 다리."""

    symbol: str
    units: float
    perp_units: float
    spot_price: float
    perp_mark: float
    perp_entry: float                  # perp 숏 진입가 (UPL 계산에 필수)
    base_mmr: float = 0.005            # 심볼·티어별 유지증거금률
    collateral_ratio: float = 0.90     # 현물 담보인정비율

    def __post_init__(self) -> None:
        """다리 파라미터의 부호·범위를 검증한다."""
        if self.units < 0 or self.perp_units < 0:
            raise ValueError("수량은 음수일 수 없습니다 (숏 크기는 양수로 표현)")
        if min(self.spot_price, self.perp_mark, self.perp_entry) <= 0:
            raise ValueError("가격은 양수여야 합니다")

    @property
    def spot_notional(self) -> float:
        """현물 명목 가치."""
        return round(self.units * self.spot_price, 8)

    @property
    def perp_notional(self) -> float:
        """perp 숏 명목 가치 (마크 기준)."""
        return round(self.perp_units * self.perp_mark, 8)

    @property
    def perp_upl(self) -> float:
        """숏 perp 미실현손익. 마크가 진입가보다 높으면 손실(음수)."""
        return round(self.perp_units * (self.perp_entry - self.perp_mark), 8)

    @property
    def unit_mismatch(self) -> float:
        """현물/perp 수량 불일치 비율."""
        target = max(self.units, self.perp_units)
        return 0.0 if target <= 0 else round(abs(self.units - self.perp_units) / target, 8)

    def shocked(self, s: StressScenario) -> tuple[float, float]:
        """스트레스 후 (현물가, perp 마크)를 반환한다."""
        spot = self.spot_price * (1 + s.spot_shock)
        return spot, spot * (1 + s.mark_premium)


@dataclass
class AccountState:
    """캐리 계좌의 전체 상태."""

    legs: list[CarryLeg] = field(default_factory=list)
    stable_collateral: float = 0.0

    @property
    def spot_notional(self) -> float:
        """전체 현물 명목."""
        return round(sum(l.spot_notional for l in self.legs), 8)

    @property
    def perp_notional(self) -> float:
        """전체 perp 숏 명목."""
        return round(sum(l.perp_notional for l in self.legs), 8)

    @property
    def gross_notional(self) -> float:
        """총 명목 (양다리 합산)."""
        return round(self.spot_notional + self.perp_notional, 8)

    @property
    def perp_upl(self) -> float:
        """전체 perp 미실현손익."""
        return round(sum(l.perp_upl for l in self.legs), 8)

    @property
    def equity(self) -> float:
        """경제적 자기자본 = 현물 시가 + 스테이블 + perp 미실현손익.

        델타중립이면 가격변동의 순효과는 베이시스 변화분만 남는다. UPL을
        빼먹으면 랠리 구간에서 자본이 과대계상되어 위험을 은폐한다.
        """
        return round(self.spot_notional + self.stable_collateral + self.perp_upl, 8)

    def maintenance_ratio(self, scenario: StressScenario | None = None) -> float:
        """계좌 유지증거금률 = 유지증거금 / 담보조정 자산. 낮을수록 안전."""
        if not self.legs:
            return 0.0

        if scenario is None:
            assets = (sum(l.spot_notional * l.collateral_ratio for l in self.legs)
                      + self.stable_collateral + self.perp_upl)
            mm = sum(l.perp_notional * l.base_mmr for l in self.legs)
            return round(mm / assets, 8) if assets > 0 else float("inf")

        s = scenario
        # 스테이블이 양수(담보)면 헤어컷, 음수(차입 부채)면 액면 전액 (부채는 깎이지 않는다)
        assets = (self.stable_collateral * s.stable_haircut
                  if self.stable_collateral > 0 else self.stable_collateral)
        mm = 0.0
        gross_after = 0.0
        for l in self.legs:
            spot_s, mark_s = l.shocked(s)
            h = min(l.collateral_ratio, s.collateral_ratio_cap)
            assets += l.units * spot_s * h
            assets += l.perp_units * (l.perp_entry - mark_s)          # 스트레스 후 UPL(누적 포함)
            assets -= l.perp_units * mark_s * s.funding_debit          # 펀딩 유출은 자산 차감
            mm += l.perp_units * mark_s * l.base_mmr * s.mmr_multiplier
            gross_after += l.units * spot_s + l.perp_units * mark_s
        assets -= gross_after * s.unwind_cost                          # 스트레스 후 명목 기준
        return float("inf") if assets <= 0 else round(mm / assets, 8)


def adl_orphan_loss(state: AccountState, post_adl_drop: float = 0.35,
                    unwind_cost: float = 0.02) -> dict[str, float]:
    """ADL로 숏 헤지가 소멸한 뒤의 손실을 사건 순서대로 계산한다.

    순서 (Codex 지적: 숏 청산이익을 이중으로 빼거나 빠뜨리면 안 된다):
      1. 사건 직전 경제적 자본 기록 (UPL 포함)
      2. ADL로 숏이 '현재 마크'에 강제 청산 → UPL이 실현이익으로 전환
      3. 숏 소멸 후 현물이 post_adl_drop 만큼 추가 하락
      4. 긴급 현물 청산 비용 부과

    Args:
        state: 사건 직전 계좌 상태.
        post_adl_drop: 헤지 소멸 후 현물 추가 하락률.
        unwind_cost: 긴급 현물 청산 비용률.

    Returns:
        pre_equity / post_equity / loss_frac (사건 직전 자본 대비 손실률).
    """
    pre = state.equity
    if pre <= 0:
        return dict(pre_equity=pre, post_equity=pre, loss_frac=float("inf"))
    realized = state.perp_upl              # ADL이 현재 마크에 청산 → UPL이 현금으로 실현
    spot_after = state.spot_notional * (1 - post_adl_drop)
    cost = spot_after * unwind_cost
    post = spot_after + state.stable_collateral + realized - cost
    return dict(pre_equity=round(pre, 8), post_equity=round(post, 8),
                realized_short_pnl=round(realized, 8),
                loss_frac=round((pre - post) / pre, 8))
