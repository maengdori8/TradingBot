"""거래 실행 계층에서 공유하는 주문·체결 계약."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Literal


class TradingMode(str, Enum):
    """지원하는 거래 실행 모드."""

    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


class OrderState(str, Enum):
    """거래소 주문의 정규화된 상태."""

    CREATED = "created"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class OrderRequest:
    """거래소에 전달할 정규화된 주문 요청."""

    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit"] = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    reduce_only: bool = False
    time_in_force: Literal["GTC", "IOC", "FOK", "PostOnly"] = "GTC"
    strategy_version: str = "unknown"
    run_id: str = "unknown"

    def __post_init__(self) -> None:
        """주문 요청이 거래소에 전달 가능한지 검증한다."""
        if not self.client_order_id.strip():
            raise ValueError("client_order_id는 비어 있을 수 없습니다")
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        if not self.strategy_version.strip():
            raise ValueError("strategy_version은 비어 있을 수 없습니다")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"지원하지 않는 side입니다: {self.side}")
        if self.order_type not in {"market", "limit"}:
            raise ValueError(f"지원하지 않는 order_type입니다: {self.order_type}")
        if self.time_in_force not in {"GTC", "IOC", "FOK", "PostOnly"}:
            raise ValueError(
                f"지원하지 않는 time_in_force입니다: {self.time_in_force}"
            )
        if not isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("quantity는 0보다 커야 합니다")
        if self.order_type == "limit" and (
            self.price is None
            or not isfinite(self.price)
            or self.price <= 0
        ):
            raise ValueError("limit 주문에는 0보다 큰 price가 필요합니다")
        if self.stop_loss is not None and (
            not isfinite(self.stop_loss) or self.stop_loss <= 0
        ):
            raise ValueError("stop_loss는 0보다 커야 합니다")
        if self.take_profit is not None and (
            not isfinite(self.take_profit) or self.take_profit <= 0
        ):
            raise ValueError("take_profit은 0보다 커야 합니다")


@dataclass(frozen=True)
class Fill:
    """개별 체결 이벤트."""

    fill_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    fee: float = 0.0
    fee_currency: str | None = None
    liquidity: Literal["maker", "taker", "unknown"] = "unknown"
    exchange_timestamp: datetime | None = None
    receive_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ExecutionReport:
    """주문 상태와 누적 체결을 담는 실행 보고서."""

    order_id: str
    client_order_id: str
    symbol: str
    state: OrderState
    requested_quantity: float
    filled_quantity: float
    average_price: float | None
    fills: tuple[Fill, ...] = ()
    exchange_timestamp: datetime | None = None
    receive_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reject_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def remaining_quantity(self) -> float:
        """미체결 수량을 반환한다."""
        return round(max(self.requested_quantity - self.filled_quantity, 0.0), 8)

    def to_dict(self) -> dict[str, Any]:
        """하위 호환 호출부용 직렬화 딕셔너리를 반환한다."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "status": self.state.value,
            "requested_qty": self.requested_quantity,
            "filled_qty": self.filled_quantity,
            "remaining_qty": self.remaining_quantity,
            "average_price": self.average_price,
            "reject_reason": self.reject_reason,
            "exchange_timestamp": (
                self.exchange_timestamp.isoformat()
                if self.exchange_timestamp is not None
                else None
            ),
            "receive_timestamp": self.receive_timestamp.isoformat(),
        }


@dataclass(frozen=True)
class FeeRateSnapshot:
    """계정별 거래 수수료율 스냅샷."""

    symbol: str
    maker_rate: float
    taker_rate: float
    exchange_timestamp: datetime | None
    receive_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source: str = "bybit_account"
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
