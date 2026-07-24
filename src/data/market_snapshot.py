from __future__ import annotations

# 시점 보존형 시장 데이터 모델.

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
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
    max_age_seconds: float = 5.0
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

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
        if not isfinite(self.max_age_seconds) or self.max_age_seconds < 0:
            raise ValueError("max_age_seconds는 0 이상의 유한한 값이어야 합니다")
        component_times = (
            ensure_utc(self.open_interest_timestamp),
            ensure_utc(self.funding_timestamp),
            ensure_utc(self.order_book_timestamp),
        )
        for component_time in component_times:
            if component_time > reference:
                raise RuntimeError("선물 특징 timestamp가 검증 시점보다 미래입니다")
            if (reference - component_time).total_seconds() > self.max_age_seconds:
                raise RuntimeError("오래된 선물 특징 데이터입니다")
        skew = (max(component_times) - min(component_times)).total_seconds()
        if skew > self.max_age_seconds:
            raise RuntimeError("선물 특징 timestamp 편차가 허용 범위를 초과합니다")
        if ensure_utc(self.exchange_timestamp) != min(component_times):
            raise RuntimeError("복합 선물 특징 시각은 가장 오래된 입력이어야 합니다")
        if ensure_utc(self.next_funding_timestamp) <= reference:
            raise RuntimeError("다음 펀딩 시각이 이미 지났습니다")


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
