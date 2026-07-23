from __future__ import annotations

"""현물-무기한 델타중립 캐리 후보의 순수 신호 함수."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from .decision import DecisionContext

logger = logging.getLogger(__name__)


def _utc_timestamp(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 시각을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class CarrySnapshot:
    """한 시점의 동일 자산 현물·무기한 시장 정보."""

    symbol: str
    spot_price: float
    perpetual_price: float
    expected_funding_rate: float
    observed_at: datetime

    def __post_init__(self) -> None:
        """가격과 관측 시각을 검증한다."""
        values = (self.spot_price, self.perpetual_price, self.expected_funding_rate)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("캐리 입력값은 유한한 숫자여야 합니다")
        if self.spot_price <= 0 or self.perpetual_price <= 0:
            raise ValueError("현물·무기한 가격은 양수여야 합니다")
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        object.__setattr__(
            self,
            "observed_at",
            _utc_timestamp(self.observed_at, "observed_at"),
        )


@dataclass(frozen=True)
class CarryConfig:
    """캐리 진입 판단에 필요한 사전 등록 파라미터."""

    expected_funding_intervals: int
    basis_capture_ratio: float
    spot_fee_rate: float
    perpetual_fee_rate: float
    slippage_rate_per_fill: float
    min_cost_multiple: float
    max_abs_basis_rate: float
    max_snapshot_age: timedelta

    def __post_init__(self) -> None:
        """과도하거나 잘못된 연구 설정을 거부한다."""
        if self.expected_funding_intervals <= 0:
            raise ValueError("expected_funding_intervals는 양수여야 합니다")
        if not 0.0 <= self.basis_capture_ratio <= 1.0:
            raise ValueError("basis_capture_ratio는 0~1이어야 합니다")
        if min(
            self.spot_fee_rate,
            self.perpetual_fee_rate,
            self.slippage_rate_per_fill,
        ) < 0:
            raise ValueError("비용률은 음수일 수 없습니다")
        if (
            self.spot_fee_rate
            + self.perpetual_fee_rate
            + self.slippage_rate_per_fill
            <= 0
        ):
            raise ValueError("최소 하나의 체결 비용률은 양수여야 합니다")
        if self.min_cost_multiple < 1.5:
            raise ValueError("min_cost_multiple은 최소 1.5여야 합니다")
        if self.max_abs_basis_rate <= 0:
            raise ValueError("max_abs_basis_rate는 양수여야 합니다")
        if self.max_snapshot_age <= timedelta(0):
            raise ValueError("max_snapshot_age는 양수여야 합니다")


@dataclass(frozen=True)
class CarrySignal:
    """현물 롱·무기한 숏으로 구성된 델타중립 진입 후보."""

    symbol: str
    action: Literal["enter"]
    spot_direction: Literal["long"]
    perpetual_direction: Literal["short"]
    expected_gross_carry: float
    expected_total_cost: float
    expected_net_carry: float
    basis_rate: float
    reason: str
    context: DecisionContext


def generate_carry_signal(
    snapshot: CarrySnapshot,
    config: CarryConfig,
    context: DecisionContext,
) -> CarrySignal | None:
    """예상 순캐리가 비용 문턱을 넘을 때만 델타중립 후보를 만든다."""
    if snapshot.observed_at > context.data_cutoff:
        raise ValueError("data_cutoff 이후 관측값은 사용할 수 없습니다")
    if context.data_cutoff - snapshot.observed_at > config.max_snapshot_age:
        logger.warning(
            "[%s] 오래된 캐리 스냅샷으로 진입 차단 run_id=%s",
            snapshot.symbol,
            context.run_id,
        )
        return None

    basis_rate = snapshot.perpetual_price / snapshot.spot_price - 1.0
    if basis_rate <= 0 or abs(basis_rate) > config.max_abs_basis_rate:
        return None
    funding_income = (
        snapshot.expected_funding_rate * config.expected_funding_intervals
    )
    basis_income = basis_rate * config.basis_capture_ratio
    expected_gross = funding_income + basis_income
    expected_cost = 2.0 * (
        config.spot_fee_rate
        + config.perpetual_fee_rate
        + 2.0 * config.slippage_rate_per_fill
    )
    required_gross = expected_cost * config.min_cost_multiple
    expected_net = expected_gross - expected_cost
    if expected_gross <= required_gross or expected_net <= 0:
        return None

    return CarrySignal(
        symbol=snapshot.symbol,
        action="enter",
        spot_direction="long",
        perpetual_direction="short",
        expected_gross_carry=round(expected_gross, 8),
        expected_total_cost=round(expected_cost, 8),
        expected_net_carry=round(expected_net, 8),
        basis_rate=round(basis_rate, 8),
        reason=(
            f"현물 롱/무기한 숏: 예상 총캐리 {expected_gross:.6f}, "
            f"왕복비용 {expected_cost:.6f}의 "
            f"{expected_gross / expected_cost:.2f}배"
        ),
        context=context,
    )
