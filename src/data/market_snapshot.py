from __future__ import annotations

# 시점 보존형 시장 데이터 모델.

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from numbers import Real
from typing import Any
from typing import Literal


def ensure_utc(value: datetime) -> datetime:
    """datetime 값을 UTC timezone-aware 형태로 반환한다."""
    if value.tzinfo is None:
        raise ValueError("timestamp에는 timezone 정보가 필요합니다")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class DataProvenance:
    """시장 데이터의 출처와 상품 종류."""

    exchange: str
    market_type: str
    requested_symbol: str
    resolved_symbol: str
    endpoint: str


@dataclass(frozen=True)
class PointInTimeRecord:
    """거래소 발생 시각과 로컬 수신 시각을 함께 보존하는 데이터."""

    exchange_timestamp: datetime
    receive_timestamp: datetime
    provenance: DataProvenance

    def __post_init__(self) -> None:
        """두 타임스탬프를 UTC 기준으로 검증한다."""
        ensure_utc(self.exchange_timestamp)
        ensure_utc(self.receive_timestamp)

    def age_seconds(self, as_of: datetime | None = None) -> float:
        """지정 시점 기준 거래소 데이터 발생 후 경과 초를 반환한다."""
        reference = ensure_utc(as_of or datetime.now(timezone.utc))
        return max(
            (reference - ensure_utc(self.exchange_timestamp)).total_seconds(),
            0.0,
        )

    @property
    def transport_latency_seconds(self) -> float:
        """거래소 발생부터 로컬 수신까지 지연 시간을 반환한다."""
        return max(
            (
                ensure_utc(self.receive_timestamp)
                - ensure_utc(self.exchange_timestamp)
            ).total_seconds(),
            0.0,
        )


@dataclass(frozen=True)
class MarketSnapshot(PointInTimeRecord):
    """ticker와 주문장을 원자적으로 묶은 시장 스냅샷."""

    symbol: str
    last: float
    bid: float | None
    ask: float | None
    bids: tuple[tuple[float, float], ...] = ()
    asks: tuple[tuple[float, float], ...] = ()
    max_age_seconds: float = 5.0
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def is_fresh(self, as_of: datetime | None = None) -> bool:
        """스냅샷이 허용된 최신성 범위 안인지 반환한다."""
        return self.age_seconds(as_of) <= self.max_age_seconds

    def assert_usable(
        self,
        expected_exchange: str = "bybit",
        expected_market_type: str = "swap",
        as_of: datetime | None = None,
    ) -> None:
        """출처와 최신성이 주문 결정에 적합한지 검증한다."""
        reference = ensure_utc(as_of or datetime.now(timezone.utc))
        if ensure_utc(self.exchange_timestamp) > reference + timedelta(seconds=2):
            raise RuntimeError("시장 데이터 timestamp가 검증 시점보다 미래입니다")
        if self.provenance.exchange != expected_exchange:
            raise RuntimeError(
                f"시장 데이터 출처 불일치: {self.provenance.exchange}"
            )
        if self.provenance.market_type != expected_market_type:
            raise RuntimeError(
                f"시장 상품 종류 불일치: {self.provenance.market_type}"
            )
        if not self.is_fresh(reference):
            raise RuntimeError(
                f"오래된 시장 데이터: age={self.age_seconds(reference):.3f}s"
            )


@dataclass(frozen=True)
class DerivativesFeatureSnapshot(PointInTimeRecord):
    """OI·펀딩·주문장을 묶은 Bybit 선물 특징 스냅샷."""

    symbol: str
    open_interest: float
    current_funding_rate: float
    next_funding_timestamp: datetime
    open_interest_timestamp: datetime
    funding_timestamp: datetime
    order_book_timestamp: datetime
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    next_funding_rate: float | None = None
    max_age_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    open_interest_max_age_seconds: float = 360.0
    funding_max_age_seconds: float = 60.0
    order_book_max_age_seconds: float = 5.0
    max_component_skew_seconds: float = 360.0

    def _component_limits(self) -> dict[str, float]:
        """구성요소별 최신성 한도를 반환한다.

        ``max_age_seconds``는 이전 호출자의 호환을 위한 명시적 override다.
        새 호출자는 OI·펀딩·주문장 한도를 각각 사용해야 한다.
        """
        if self.max_age_seconds is not None:
            legacy_limit = _validated_non_negative_seconds(
                "max_age_seconds",
                self.max_age_seconds,
            )
            return {
                "open_interest": legacy_limit,
                "funding": legacy_limit,
                "order_book": legacy_limit,
            }
        return {
            "open_interest": _validated_non_negative_seconds(
                "open_interest_max_age_seconds",
                self.open_interest_max_age_seconds,
            ),
            "funding": _validated_non_negative_seconds(
                "funding_max_age_seconds",
                self.funding_max_age_seconds,
            ),
            "order_book": _validated_non_negative_seconds(
                "order_book_max_age_seconds",
                self.order_book_max_age_seconds,
            ),
        }

    def _skew_limit(self) -> float:
        """호환 override를 고려한 구성요소 간 시각 편차 한도를 반환한다."""
        if self.max_age_seconds is not None:
            return _validated_non_negative_seconds(
                "max_age_seconds",
                self.max_age_seconds,
            )
        return _validated_non_negative_seconds(
            "max_component_skew_seconds",
            self.max_component_skew_seconds,
        )

    def __post_init__(self) -> None:
        """비가격 특징의 값·호가·시점 불변식을 생성 즉시 검증한다."""
        super().__post_init__()
        if not self.symbol.strip():
            raise ValueError("선물 특징 symbol은 비어 있을 수 없습니다")
        if (
            self.provenance.requested_symbol != self.symbol
            or self.provenance.resolved_symbol != self.symbol
        ):
            raise ValueError("선물 특징 provenance 심볼이 데이터 심볼과 다릅니다")
        if (
            isinstance(self.open_interest, bool)
            or not isinstance(self.open_interest, Real)
            or not isfinite(float(self.open_interest))
            or self.open_interest < 0
        ):
            raise ValueError("open_interest는 0 이상의 유한한 값이어야 합니다")
        if (
            isinstance(self.current_funding_rate, bool)
            or not isinstance(self.current_funding_rate, Real)
            or not isfinite(float(self.current_funding_rate))
        ):
            raise ValueError("current_funding_rate는 유한한 값이어야 합니다")
        if self.next_funding_rate is not None and (
            isinstance(self.next_funding_rate, bool)
            or not isinstance(self.next_funding_rate, Real)
            or not isfinite(float(self.next_funding_rate))
        ):
            raise ValueError("next_funding_rate는 유한한 값이어야 합니다")
        component_limits = self._component_limits()
        skew_limit = self._skew_limit()
        for side_name, levels in (("bids", self.bids), ("asks", self.asks)):
            if not levels:
                raise ValueError(f"{side_name} 호가는 비어 있을 수 없습니다")
            for level in levels:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    raise ValueError(f"{side_name} 호가 형식이 잘못되었습니다")
                price, quantity = level[0], level[1]
                if (
                    isinstance(price, bool)
                    or isinstance(quantity, bool)
                    or not isinstance(price, Real)
                    or not isinstance(quantity, Real)
                    or not isfinite(float(price))
                    or not isfinite(float(quantity))
                    or price <= 0
                    or quantity <= 0
                ):
                    raise ValueError(
                        f"{side_name} 가격과 수량은 양의 유한값이어야 합니다"
                    )
        received = ensure_utc(self.receive_timestamp)
        component_times = {
            "open_interest": ensure_utc(self.open_interest_timestamp),
            "funding": ensure_utc(self.funding_timestamp),
            "order_book": ensure_utc(self.order_book_timestamp),
        }
        for component, component_time in component_times.items():
            if component_time > received:
                raise ValueError(f"{component} timestamp가 수신 시각보다 미래입니다")
            age = (received - component_time).total_seconds()
            if age > component_limits[component]:
                raise ValueError(
                    f"오래된 {component} 데이터: age={age:.3f}s, "
                    f"limit={component_limits[component]:.3f}s"
                )
        if (
            max(component_times.values()) - min(component_times.values())
        ).total_seconds() > skew_limit:
            raise ValueError("선물 특징 timestamp 편차가 허용 범위를 초과합니다")
        if ensure_utc(self.exchange_timestamp) != min(component_times.values()):
            raise ValueError("복합 선물 특징 시각은 가장 오래된 입력이어야 합니다")
        if ensure_utc(self.next_funding_timestamp) <= received:
            raise ValueError("다음 펀딩 시각이 이미 지났습니다")

    def assert_usable(self, as_of: datetime | None = None) -> None:
        """각 입력의 출처·최신성·시각 편차를 검증한다."""
        reference = ensure_utc(as_of or datetime.now(timezone.utc))
        if self.provenance.exchange != "bybit":
            raise RuntimeError(
                f"선물 특징 데이터 출처 불일치: {self.provenance.exchange}"
            )
        if self.provenance.market_type != "swap":
            raise RuntimeError(
                f"선물 특징 상품 종류 불일치: {self.provenance.market_type}"
            )
        if (
            self.provenance.requested_symbol != self.symbol
            or self.provenance.resolved_symbol != self.symbol
        ):
            raise RuntimeError("선물 특징 provenance 심볼이 데이터 심볼과 다릅니다")
        component_limits = self._component_limits()
        skew_limit = self._skew_limit()
        component_times = {
            "open_interest": ensure_utc(self.open_interest_timestamp),
            "funding": ensure_utc(self.funding_timestamp),
            "order_book": ensure_utc(self.order_book_timestamp),
        }
        for component, component_time in component_times.items():
            if component_time > reference:
                raise RuntimeError(f"{component} timestamp가 검증 시점보다 미래입니다")
            age = (reference - component_time).total_seconds()
            if age > component_limits[component]:
                raise RuntimeError(
                    f"오래된 {component} 데이터: age={age:.3f}s, "
                    f"limit={component_limits[component]:.3f}s"
                )
        skew = (
            max(component_times.values()) - min(component_times.values())
        ).total_seconds()
        if skew > skew_limit:
            raise RuntimeError("선물 특징 timestamp 편차가 허용 범위를 초과합니다")
        if ensure_utc(self.exchange_timestamp) != min(component_times.values()):
            raise RuntimeError("복합 선물 특징 시각은 가장 오래된 입력이어야 합니다")
        if ensure_utc(self.next_funding_timestamp) <= reference:
            raise RuntimeError("다음 펀딩 시각이 이미 지났습니다")


def _validated_non_negative_seconds(name: str, value: float) -> float:
    """0 이상의 유한한 초 단위 설정값을 검증한다."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name}는 0 이상의 유한한 값이어야 합니다")
    return float(value)


@dataclass(frozen=True)
class LiquidationRecord(PointInTimeRecord):
    """Bybit public liquidation 스트림에서 수신한 청산 이벤트."""

    event_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """청산 레코드의 시각과 필수 값을 검증한다."""
        super().__post_init__()
        if not self.event_id.strip():
            raise ValueError("liquidation event_id는 비어 있을 수 없습니다")
        if not self.symbol.strip():
            raise ValueError("liquidation symbol은 비어 있을 수 없습니다")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"지원하지 않는 liquidation side입니다: {self.side}")
        if not isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("liquidation quantity는 0보다 큰 유한값이어야 합니다")
        if not isfinite(self.price) or self.price <= 0:
            raise ValueError("liquidation price는 0보다 큰 유한값이어야 합니다")
        if ensure_utc(self.exchange_timestamp) > ensure_utc(self.receive_timestamp):
            raise ValueError("liquidation exchange_timestamp가 수신 시각보다 미래입니다")
        if self.provenance.exchange != "bybit":
            raise ValueError("liquidation 출처는 bybit여야 합니다")
        if self.provenance.market_type != "swap":
            raise ValueError("liquidation 상품 종류는 swap이어야 합니다")
