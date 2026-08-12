from __future__ import annotations

"""근거 연구와 paper/demo가 공유하는 시점 안전 특징·주문 의도 API."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from .carry_signal import CarryConfig, CarrySnapshot, generate_carry_signal
from .decision import DecisionContext
from .forced_flow_signal import (
    ForcedFlowConfig,
    ForcedFlowSnapshot,
    generate_forced_flow_signal,
)


def _as_utc(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 시각을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


def _finite(value: float, field_name: str) -> float:
    """유한한 실수만 반환한다."""
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name}은(는) 유한한 숫자여야 합니다")
    return normalized


@dataclass(frozen=True)
class TimedValue:
    """시장 관측 시각과 실제 사용 가능 시각을 분리한 숫자 값."""

    observed_at: datetime
    available_at: datetime
    value: float

    def __post_init__(self) -> None:
        """시각 순서와 숫자를 검증한다."""
        observed = _as_utc(self.observed_at, "observed_at")
        available = _as_utc(self.available_at, "available_at")
        if available < observed:
            raise ValueError("available_at은 observed_at 이전일 수 없습니다")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "value", _finite(self.value, "value"))


@dataclass(frozen=True)
class LiquidationNotional:
    """공격적 체결 방향으로 정규화한 단일 청산 명목금액.

    ``sell``은 롱 청산의 시장 매도, ``buy``는 숏 청산의 시장 매수를 뜻한다.
    거래소 원시 포지션 방향을 이 필드에 그대로 전달하면 안 된다.
    """

    observed_at: datetime
    available_at: datetime
    side: Literal["buy", "sell"]
    notional: float

    def __post_init__(self) -> None:
        """청산 시각·방향·명목금액을 검증한다."""
        observed = _as_utc(self.observed_at, "observed_at")
        available = _as_utc(self.available_at, "available_at")
        if available < observed:
            raise ValueError("available_at은 observed_at 이전일 수 없습니다")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"지원하지 않는 청산 side입니다: {self.side}")
        notional = _finite(self.notional, "notional")
        if notional <= 0:
            raise ValueError("청산 명목금액은 양수여야 합니다")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "notional", notional)


@dataclass(frozen=True)
class BookLevel:
    """한 가격 수준의 가격과 수량."""

    price: float
    quantity: float

    def __post_init__(self) -> None:
        """양의 가격과 수량을 검증한다."""
        price = _finite(self.price, "price")
        quantity = _finite(self.quantity, "quantity")
        if price <= 0 or quantity <= 0:
            raise ValueError("호가 가격과 수량은 양수여야 합니다")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "quantity", quantity)


@dataclass(frozen=True)
class OrderBookEvidence:
    """결정 시점에 수신한 동일 상품 25레벨 오더북."""

    observed_at: datetime
    available_at: datetime
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    def __post_init__(self) -> None:
        """오더북 시각·깊이·정렬을 검증한다."""
        observed = _as_utc(self.observed_at, "observed_at")
        available = _as_utc(self.available_at, "available_at")
        if available < observed:
            raise ValueError("available_at은 observed_at 이전일 수 없습니다")
        if len(self.bids) < 25 or len(self.asks) < 25:
            raise ValueError("25레벨 bid/ask가 모두 필요합니다")
        selected_bids = self.bids[:25]
        selected_asks = self.asks[:25]
        if any(
            selected_bids[index].price < selected_bids[index + 1].price
            for index in range(24)
        ):
            raise ValueError("bid는 가격 내림차순이어야 합니다")
        if any(
            selected_asks[index].price > selected_asks[index + 1].price
            for index in range(24)
        ):
            raise ValueError("ask는 가격 오름차순이어야 합니다")
        if selected_bids[0].price >= selected_asks[0].price:
            raise ValueError("교차된 오더북은 사용할 수 없습니다")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True)
class FeedGap:
    """특징 스트림에서 신뢰할 수 없는 반개구간."""

    started_at: datetime
    ended_at: datetime | None
    component: Literal["price", "open_interest", "funding", "liquidation", "orderbook", "volume"]

    def __post_init__(self) -> None:
        """gap 시각과 구성요소를 검증한다."""
        start = _as_utc(self.started_at, "started_at")
        end = None if self.ended_at is None else _as_utc(self.ended_at, "ended_at")
        if end is not None and end < start:
            raise ValueError("ended_at은 started_at 이전일 수 없습니다")
        if self.component not in {
            "price",
            "open_interest",
            "funding",
            "liquidation",
            "orderbook",
            "volume",
        }:
            raise ValueError(f"지원하지 않는 feed gap component입니다: {self.component}")
        object.__setattr__(self, "started_at", start)
        object.__setattr__(self, "ended_at", end)


@dataclass(frozen=True)
class FeatureFreshness:
    """강제흐름 특징 구성요소별 최대 지연 허용치."""

    price: timedelta
    open_interest: timedelta
    funding: timedelta
    orderbook: timedelta
    volume: timedelta
    baseline_skew: timedelta

    def __post_init__(self) -> None:
        """모든 최신성 제한이 양수인지 검증한다."""
        if any(
            value <= timedelta(0)
            for value in (
                self.price,
                self.open_interest,
                self.funding,
                self.orderbook,
                self.volume,
                self.baseline_skew,
            )
        ):
            raise ValueError("특징 최신성 제한은 모두 양수여야 합니다")


@dataclass(frozen=True)
class ForcedFlowFeatureBundle:
    """강제흐름 특징을 시점 안전하게 만들기 위한 원시 입력 묶음."""

    symbol: str
    prices: tuple[TimedValue, ...]
    open_interest: tuple[TimedValue, ...]
    completed_volumes: tuple[TimedValue, ...]
    funding: tuple[TimedValue, ...]
    liquidations: tuple[LiquidationNotional, ...]
    orderbook: OrderBookEvidence
    feed_gaps: tuple[FeedGap, ...] = ()

    def __post_init__(self) -> None:
        """심볼과 필수 시계열 존재 여부를 검증한다."""
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        if not self.prices or not self.open_interest:
            raise ValueError("가격과 OI 시계열이 필요합니다")
        if not self.completed_volumes or not self.funding:
            raise ValueError("완결 거래량과 펀딩 시계열이 필요합니다")


@dataclass(frozen=True)
class StrategyIntentLeg:
    """실행 모듈에 독립적인 전략 주문 다리 의도."""

    symbol: str
    side: Literal["buy", "sell"]
    requested_quantity: float
    reference_price: float

    def __post_init__(self) -> None:
        """주문 다리 값을 검증한다."""
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"지원하지 않는 side입니다: {self.side}")
        quantity = _finite(self.requested_quantity, "requested_quantity")
        price = _finite(self.reference_price, "reference_price")
        if quantity <= 0 or price <= 0:
            raise ValueError("주문 수량과 기준 가격은 양수여야 합니다")
        object.__setattr__(self, "requested_quantity", quantity)
        object.__setattr__(self, "reference_price", price)


@dataclass(frozen=True)
class StrategyTradeIntent:
    """검증된 전략 버전이 생성한 단일 또는 다중 다리 주문 의도."""

    candidate_id: str
    family: Literal["delta_neutral_carry", "forced_flow"]
    direction: Literal["delta_neutral", "long", "short"]
    legs: tuple[StrategyIntentLeg, ...]
    reason: str
    context: DecisionContext

    def __post_init__(self) -> None:
        """후보 식별자와 전략군별 주문 형태를 검증한다."""
        if not self.candidate_id.strip():
            raise ValueError("candidate_id는 비어 있을 수 없습니다")
        if self.family == "delta_neutral_carry":
            if self.direction != "delta_neutral" or len(self.legs) != 2:
                raise ValueError("캐리 의도는 정확히 두 다리 델타중립이어야 합니다")
        elif self.family == "forced_flow":
            if self.direction not in {"long", "short"} or len(self.legs) != 1:
                raise ValueError("강제흐름 의도는 정확히 한 방향 한 다리여야 합니다")
        else:
            raise ValueError(f"지원하지 않는 전략군입니다: {self.family}")


def _known_values(values: tuple[TimedValue, ...], cutoff: datetime) -> list[TimedValue]:
    """cutoff까지 실제로 알 수 있었던 값만 관측 시각 순서로 반환한다."""
    known = [
        item
        for item in values
        if item.observed_at <= cutoff and item.available_at <= cutoff
    ]
    return sorted(known, key=lambda item: (item.observed_at, item.available_at))


def _latest(values: tuple[TimedValue, ...], cutoff: datetime) -> TimedValue | None:
    """cutoff 시점의 최신 알려진 값을 반환한다."""
    known = _known_values(values, cutoff)
    return known[-1] if known else None


def _baseline(
    values: tuple[TimedValue, ...],
    cutoff: datetime,
    target: datetime,
    tolerance: timedelta,
) -> TimedValue | None:
    """목표 시각 이하에서 가장 가까운 시점 안전 기준값을 반환한다."""
    eligible = [item for item in _known_values(values, cutoff) if item.observed_at <= target]
    if not eligible:
        return None
    selected = eligible[-1]
    if target - selected.observed_at > tolerance:
        return None
    return selected


def _has_overlapping_gap(
    gaps: tuple[FeedGap, ...],
    start: datetime,
    end: datetime,
) -> bool:
    """특징 구간과 겹치는 feed gap이 하나라도 있는지 반환한다."""
    for gap in gaps:
        gap_end = gap.ended_at or end
        if gap.started_at <= end and gap_end >= start:
            return True
    return False


def build_forced_flow_snapshot(
    bundle: ForcedFlowFeatureBundle,
    context: DecisionContext,
    *,
    horizon: timedelta,
    freshness: FeatureFreshness,
    volume_lookback: int = 30,
) -> ForcedFlowSnapshot | None:
    """4h/8h 원시 입력으로 시점 안전 강제흐름 특징을 만든다."""
    if horizon not in {timedelta(hours=4), timedelta(hours=8)}:
        raise ValueError("강제흐름 horizon은 4h 또는 8h여야 합니다")
    if volume_lookback < 20:
        raise ValueError("volume_lookback은 최소 20이어야 합니다")
    cutoff = context.data_cutoff
    window_start = cutoff - horizon
    if _has_overlapping_gap(bundle.feed_gaps, window_start, cutoff):
        return None

    current_price = _latest(bundle.prices, cutoff)
    current_oi = _latest(bundle.open_interest, cutoff)
    funding = _latest(bundle.funding, cutoff)
    if current_price is None or current_oi is None or funding is None:
        return None
    if cutoff - current_price.observed_at > freshness.price:
        return None
    if cutoff - current_oi.observed_at > freshness.open_interest:
        return None
    if cutoff - funding.observed_at > freshness.funding:
        return None
    if cutoff - bundle.orderbook.observed_at > freshness.orderbook:
        return None
    if bundle.orderbook.available_at > cutoff or bundle.orderbook.observed_at > cutoff:
        return None

    price_base = _baseline(
        bundle.prices,
        cutoff,
        current_price.observed_at - horizon,
        freshness.baseline_skew,
    )
    oi_base = _baseline(
        bundle.open_interest,
        cutoff,
        current_oi.observed_at - horizon,
        freshness.baseline_skew,
    )
    if price_base is None or oi_base is None:
        return None
    if min(current_price.value, price_base.value, current_oi.value, oi_base.value) <= 0:
        raise ValueError("가격과 OI는 양수여야 합니다")

    volumes = _known_values(bundle.completed_volumes, cutoff)
    if len(volumes) < volume_lookback:
        return None
    selected_volumes = volumes[-volume_lookback:]
    if cutoff - selected_volumes[-1].observed_at > freshness.volume:
        return None
    if any(item.value < 0 for item in selected_volumes):
        raise ValueError("완결 거래량은 음수일 수 없습니다")
    prior = [item.value for item in selected_volumes[:-1]]
    prior_mean = sum(prior) / len(prior)
    prior_variance = sum((value - prior_mean) ** 2 for value in prior) / len(prior)
    prior_std = math.sqrt(prior_variance)
    volume_zscore = (
        0.0
        if prior_std == 0
        else (selected_volumes[-1].value - prior_mean) / prior_std
    )

    known_liquidations = [
        event
        for event in bundle.liquidations
        if window_start < event.observed_at <= cutoff
        and event.available_at <= cutoff
    ]
    buy_notional = sum(
        event.notional for event in known_liquidations if event.side == "buy"
    )
    sell_notional = sum(
        event.notional for event in known_liquidations if event.side == "sell"
    )
    liquidation_total = buy_notional + sell_notional
    liquidation_imbalance = (
        0.0
        if liquidation_total == 0
        else (buy_notional - sell_notional) / liquidation_total
    )

    bid_notional = sum(level.price * level.quantity for level in bundle.orderbook.bids[:25])
    ask_notional = sum(level.price * level.quantity for level in bundle.orderbook.asks[:25])
    book_total = bid_notional + ask_notional
    if book_total <= 0:
        return None

    return ForcedFlowSnapshot(
        symbol=bundle.symbol,
        price_return=current_price.value / price_base.value - 1.0,
        open_interest_change=current_oi.value / oi_base.value - 1.0,
        funding_rate=funding.value,
        liquidation_imbalance=liquidation_imbalance,
        orderbook_imbalance=(bid_notional - ask_notional) / book_total,
        volume_zscore=volume_zscore,
        observed_at=cutoff,
    )


def decide_carry_intent(
    snapshot: CarrySnapshot,
    config: CarryConfig,
    context: DecisionContext,
    *,
    candidate_id: str,
    spot_symbol: str,
    perpetual_symbol: str,
    requested_quantity: float,
) -> StrategyTradeIntent | None:
    """캐리 신호를 현물 매수·무기한 매도의 원자 의도로 변환한다."""
    signal = generate_carry_signal(snapshot, config, context)
    if signal is None:
        return None
    return StrategyTradeIntent(
        candidate_id=candidate_id,
        family="delta_neutral_carry",
        direction="delta_neutral",
        legs=(
            StrategyIntentLeg(
                symbol=spot_symbol,
                side="buy",
                requested_quantity=requested_quantity,
                reference_price=snapshot.spot_price,
            ),
            StrategyIntentLeg(
                symbol=perpetual_symbol,
                side="sell",
                requested_quantity=requested_quantity,
                reference_price=snapshot.perpetual_price,
            ),
        ),
        reason=signal.reason,
        context=context,
    )


def decide_forced_flow_intent(
    bundle: ForcedFlowFeatureBundle,
    config: ForcedFlowConfig,
    context: DecisionContext,
    *,
    freshness: FeatureFreshness,
    candidate_id: str,
    perpetual_symbol: str,
    requested_quantity: float,
    last_rebalance_time: datetime | None = None,
    volume_lookback: int = 30,
) -> StrategyTradeIntent | None:
    """공용 특징 정의로 강제흐름 단일 방향 주문 의도를 만든다."""
    snapshot = build_forced_flow_snapshot(
        bundle,
        context,
        horizon=timedelta(hours=config.rebalance_hours),
        freshness=freshness,
        volume_lookback=volume_lookback,
    )
    if snapshot is None:
        return None
    signal = generate_forced_flow_signal(
        snapshot,
        config,
        context,
        last_rebalance_time=last_rebalance_time,
    )
    if signal is None:
        return None
    side: Literal["buy", "sell"] = "buy" if signal.direction == "long" else "sell"
    reference_price = (bundle.orderbook.asks[0].price if side == "buy" else bundle.orderbook.bids[0].price)
    return StrategyTradeIntent(
        candidate_id=candidate_id,
        family="forced_flow",
        direction=signal.direction,
        legs=(
            StrategyIntentLeg(
                symbol=perpetual_symbol,
                side=side,
                requested_quantity=requested_quantity,
                reference_price=reference_price,
            ),
        ),
        reason=signal.reason,
        context=context,
    )
