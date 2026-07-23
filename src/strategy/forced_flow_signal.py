from __future__ import annotations

"""OI·펀딩·청산·오더북을 결합한 저회전 강제흐름 후보."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from .decision import DecisionContext

logger = logging.getLogger(__name__)


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """숫자를 지정 구간으로 제한한다."""
    return min(upper, max(lower, value))


def _as_utc(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 datetime을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ForcedFlowSnapshot:
    """저회전 강제흐름 판단에 사용하는 시점 보존 특징."""

    symbol: str
    price_return: float
    open_interest_change: float
    funding_rate: float
    liquidation_imbalance: float
    orderbook_imbalance: float
    volume_zscore: float
    observed_at: datetime

    def __post_init__(self) -> None:
        """특징 범위와 관측 시각을 검증한다."""
        values = (
            self.price_return,
            self.open_interest_change,
            self.funding_rate,
            self.liquidation_imbalance,
            self.orderbook_imbalance,
            self.volume_zscore,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("강제흐름 특징은 유한한 숫자여야 합니다")
        if not -1.0 <= self.liquidation_imbalance <= 1.0:
            raise ValueError("liquidation_imbalance는 -1~1이어야 합니다")
        if not -1.0 <= self.orderbook_imbalance <= 1.0:
            raise ValueError("orderbook_imbalance는 -1~1이어야 합니다")
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        object.__setattr__(
            self,
            "observed_at",
            _as_utc(self.observed_at, "observed_at"),
        )


@dataclass(frozen=True)
class ForcedFlowConfig:
    """강제흐름 점수와 재조정 주기의 사전 등록 설정."""

    rebalance_hours: Literal[4, 8]
    signal_threshold: float
    price_scale: float
    oi_scale: float
    funding_scale: float
    min_volume_zscore: float
    weight_oi_price: float
    weight_liquidation: float
    weight_orderbook: float
    weight_funding_crowding: float
    max_snapshot_age: timedelta

    def __post_init__(self) -> None:
        """점수 설정과 4h/8h 저회전 제약을 검증한다."""
        if self.rebalance_hours not in (4, 8):
            raise ValueError("rebalance_hours는 4 또는 8이어야 합니다")
        if not 0 < self.signal_threshold <= 1:
            raise ValueError("signal_threshold는 0~1 사이여야 합니다")
        if min(self.price_scale, self.oi_scale, self.funding_scale) <= 0:
            raise ValueError("특징 scale은 양수여야 합니다")
        weights = (
            self.weight_oi_price,
            self.weight_liquidation,
            self.weight_orderbook,
            self.weight_funding_crowding,
        )
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("점수 가중치는 음수가 아니며 합이 양수여야 합니다")
        if self.max_snapshot_age <= timedelta(0):
            raise ValueError("max_snapshot_age는 양수여야 합니다")


@dataclass(frozen=True)
class ForcedFlowSignal:
    """구성 요소를 설명할 수 있는 방향성 강제흐름 후보."""

    symbol: str
    direction: Literal["long", "short"]
    score: float
    components: tuple[tuple[str, float], ...]
    rebalance_after: datetime
    reason: str
    context: DecisionContext


def generate_forced_flow_signal(
    snapshot: ForcedFlowSnapshot,
    config: ForcedFlowConfig,
    context: DecisionContext,
    *,
    last_rebalance_time: datetime | None = None,
) -> ForcedFlowSignal | None:
    """비가격 특징을 합성하고 4h/8h 쿨다운을 지켜 방향 신호를 만든다."""
    if snapshot.observed_at > context.data_cutoff:
        raise ValueError("data_cutoff 이후 관측값은 사용할 수 없습니다")
    if context.data_cutoff - snapshot.observed_at > config.max_snapshot_age:
        logger.warning(
            "[%s] 오래된 강제흐름 스냅샷으로 진입 차단 run_id=%s",
            snapshot.symbol,
            context.run_id,
        )
        return None
    if last_rebalance_time is not None:
        last_rebalance = _as_utc(last_rebalance_time, "last_rebalance_time")
        if last_rebalance > context.decision_time:
            raise ValueError("last_rebalance_time은 decision_time 이후일 수 없습니다")
        cooldown = timedelta(hours=config.rebalance_hours)
        if context.decision_time - last_rebalance < cooldown:
            return None
    if snapshot.volume_zscore < config.min_volume_zscore:
        return None

    price_direction = _clip(snapshot.price_return / config.price_scale)
    oi_expansion = _clip(
        max(0.0, snapshot.open_interest_change) / config.oi_scale,
        0.0,
        1.0,
    )
    components = (
        ("oi_price", price_direction * oi_expansion),
        ("liquidation", snapshot.liquidation_imbalance),
        ("orderbook", snapshot.orderbook_imbalance),
        ("funding_crowding", -_clip(snapshot.funding_rate / config.funding_scale)),
    )
    weights = (
        config.weight_oi_price,
        config.weight_liquidation,
        config.weight_orderbook,
        config.weight_funding_crowding,
    )
    weight_sum = sum(weights)
    score = sum(
        component_value * weight
        for (_, component_value), weight in zip(components, weights)
    ) / weight_sum
    score = _clip(score)
    if abs(score) < config.signal_threshold:
        return None

    direction: Literal["long", "short"] = "long" if score > 0 else "short"
    rounded_components = tuple(
        (name, round(value, 8)) for name, value in components
    )
    return ForcedFlowSignal(
        symbol=snapshot.symbol,
        direction=direction,
        score=round(score, 8),
        components=rounded_components,
        rebalance_after=context.decision_time
        + timedelta(hours=config.rebalance_hours),
        reason=(
            f"OI-가격·청산·오더북·펀딩 합성점수 {score:+.3f}, "
            f"{config.rebalance_hours}h 재조정"
        ),
        context=context,
    )
