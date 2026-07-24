from __future__ import annotations

# 공용 거래 계약을 사용하는 주문장 기반 페이퍼 체결 모델.

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from src.exchange.contracts import ExecutionReport, Fill, OrderRequest, OrderState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderBookSnapshot:
    """한 시점의 주문장 스냅샷."""

    symbol: str
    bids: Sequence[tuple[float, float]]
    asks: Sequence[tuple[float, float]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "bybit"

    def __post_init__(self) -> None:
        """가격·수량과 스냅샷 시간대를 검증한다."""
        if self.timestamp.tzinfo is None:
            raise ValueError("주문장 timestamp는 timezone-aware여야 합니다.")
        for price, qty in [*self.bids, *self.asks]:
            if price <= 0 or qty < 0:
                raise ValueError("호가 가격은 양수이고 수량은 음수가 아니어야 합니다.")

    @property
    def best_bid(self) -> float | None:
        """최우선 매수호가를 반환한다."""
        return max((float(level[0]) for level in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        """최우선 매도호가를 반환한다."""
        return min((float(level[0]) for level in self.asks), default=None)

    @property
    def mid_price(self) -> float | None:
        """양방향 최우선 호가의 중간값을 반환한다."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2


def report_fill_rate(report: ExecutionReport) -> float:
    """공용 실행 보고서의 체결률을 반환한다."""
    if report.requested_quantity <= 0:
        return 0.0
    return report.filled_quantity / report.requested_quantity


def report_maker_quantity(report: ExecutionReport) -> float:
    """공용 실행 보고서의 메이커 체결량을 반환한다."""
    return sum(fill.quantity for fill in report.fills if fill.liquidity == "maker")


def report_taker_quantity(report: ExecutionReport) -> float:
    """공용 실행 보고서의 테이커 체결량을 반환한다."""
    return sum(fill.quantity for fill in report.fills if fill.liquidity == "taker")


def report_total_fee(report: ExecutionReport) -> float:
    """공용 실행 보고서의 수수료 합계를 반환한다."""
    return sum(fill.fee for fill in report.fills)


def fill_slippage_cost(fill: Fill) -> float:
    """체결 raw 메타데이터의 슬리피지·불리한 선택 비용을 반환한다."""
    return float(fill.raw.get("slippage_cost", 0.0)) + float(
        fill.raw.get("adverse_selection_cost", 0.0)
    )


def report_total_slippage_cost(report: ExecutionReport) -> float:
    """공용 실행 보고서의 슬리피지·불리한 선택 비용 합계를 반환한다."""
    return sum(fill_slippage_cost(fill) for fill in report.fills)


class OrderBookExecutionModel:
    """주문장 깊이·대기열·불리한 선택을 반영하는 결정론적 체결 모델."""

    def __init__(
        self,
        queue_fill_ratio: float = 0.25,
        adverse_selection_bps: float = 0.0,
        max_slippage_bps: float | None = None,
    ) -> None:
        """체결 모델을 초기화한다.

        Args:
            queue_fill_ratio: 관측된 메이커 수량 중 내 주문 체결 비율.
            adverse_selection_bps: 체결 직후 불리한 가격 이동 비용(bp).
            max_slippage_bps: 시장가가 중간가에서 벗어날 수 있는 최대 폭.
        """
        if not 0 <= queue_fill_ratio <= 1:
            raise ValueError("queue_fill_ratio는 0과 1 사이여야 합니다.")
        if adverse_selection_bps < 0:
            raise ValueError("adverse_selection_bps는 음수일 수 없습니다.")
        if max_slippage_bps is not None and max_slippage_bps < 0:
            raise ValueError("max_slippage_bps는 음수일 수 없습니다.")
        self.queue_fill_ratio = queue_fill_ratio
        self.adverse_selection_bps = adverse_selection_bps
        self.max_slippage_bps = max_slippage_bps

    def execute(
        self,
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
        maker_available_qty: float = 0.0,
    ) -> ExecutionReport:
        """주문장 스냅샷에서 부분체결·미체결·취소를 계산한다."""
        if request.symbol != orderbook.symbol:
            return self._empty_report(
                request,
                OrderState.REJECTED,
                orderbook,
                "주문과 주문장 심볼 불일치",
            )
        if orderbook.mid_price is None:
            return self._empty_report(
                request,
                OrderState.REJECTED,
                orderbook,
                "양방향 주문장 없음",
            )
        marketable = self._is_marketable(request, orderbook)
        if request.time_in_force == "PostOnly" and marketable:
            return self._empty_report(
                request,
                OrderState.CANCELED,
                orderbook,
                "PostOnly 주문이 시장가로 체결될 수 있음",
            )
        if request.order_type == "limit" and not marketable:
            return self._execute_maker(request, orderbook, maker_available_qty)
        return self._execute_taker(request, orderbook)

    @staticmethod
    def _is_marketable(
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
    ) -> bool:
        """주문이 즉시 시장성 체결 가능한지 판정한다."""
        if request.order_type == "market":
            return True
        if request.side == "buy":
            return bool(
                orderbook.best_ask is not None
                and request.price is not None
                and request.price >= orderbook.best_ask
            )
        return bool(
            orderbook.best_bid is not None
            and request.price is not None
            and request.price <= orderbook.best_bid
        )

    def _execute_maker(
        self,
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
        maker_available_qty: float,
    ) -> ExecutionReport:
        """대기 중 지정가 주문의 메이커 부분체결을 계산한다."""
        executable = max(0.0, maker_available_qty) * self.queue_fill_ratio
        filled_quantity = round(min(request.quantity, executable), 8)
        if request.time_in_force == "FOK" and filled_quantity < request.quantity:
            return self._empty_report(
                request,
                OrderState.CANCELED,
                orderbook,
                "FOK 메이커 전량 체결 불가",
            )
        fills: tuple[Fill, ...] = ()
        if filled_quantity > 0 and request.price is not None:
            adverse_cost = round(
                request.price
                * filled_quantity
                * self.adverse_selection_bps
                / 10_000,
                8,
            )
            fills = (
                self._fill(
                    request,
                    orderbook,
                    request.price,
                    filled_quantity,
                    "maker",
                    0.0,
                    adverse_cost,
                ),
            )
        return self._build_report(request, orderbook, fills, resting=True)

    def _execute_taker(
        self,
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
    ) -> ExecutionReport:
        """주문장 깊이를 소진해 테이커 체결을 계산한다."""
        levels = (
            sorted(orderbook.asks, key=lambda level: level[0])
            if request.side == "buy"
            else sorted(orderbook.bids, key=lambda level: level[0], reverse=True)
        )
        remaining = request.quantity
        fills: list[Fill] = []
        mid = float(orderbook.mid_price or 0.0)
        for raw_price, raw_qty in levels:
            price = float(raw_price)
            available = max(0.0, float(raw_qty))
            if available <= 0 or not self._within_order_limits(request, price, mid):
                continue
            fill_quantity = round(min(remaining, available), 8)
            if fill_quantity <= 0:
                continue
            reference = request.price if request.price is not None else mid
            sign = 1.0 if request.side == "buy" else -1.0
            slippage_cost = round(
                max(0.0, sign * (price - reference)) * fill_quantity,
                8,
            )
            adverse_cost = round(
                price * fill_quantity * self.adverse_selection_bps / 10_000,
                8,
            )
            fills.append(
                self._fill(
                    request,
                    orderbook,
                    price,
                    fill_quantity,
                    "taker",
                    slippage_cost,
                    adverse_cost,
                )
            )
            remaining = round(remaining - fill_quantity, 8)
            if remaining <= 0:
                break
        if request.time_in_force == "FOK" and remaining > 0:
            return self._empty_report(
                request,
                OrderState.CANCELED,
                orderbook,
                "FOK 전량 체결 불가",
            )
        return self._build_report(request, orderbook, tuple(fills), resting=False)

    def _within_order_limits(
        self,
        request: OrderRequest,
        price: float,
        mid_price: float,
    ) -> bool:
        """지정가와 최대 슬리피지 한도 안의 호가인지 확인한다."""
        if request.price is not None:
            if request.side == "buy" and price > request.price:
                return False
            if request.side == "sell" and price < request.price:
                return False
        if self.max_slippage_bps is None or mid_price <= 0:
            return True
        slippage_bps = abs(price - mid_price) / mid_price * 10_000
        return slippage_bps <= self.max_slippage_bps

    @staticmethod
    def _fill(
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
        price: float,
        quantity: float,
        liquidity: str,
        slippage_cost: float,
        adverse_selection_cost: float,
    ) -> Fill:
        """공용 Fill 계약에 페이퍼 비용 메타데이터를 담는다."""
        return Fill(
            fill_id=str(uuid.uuid4()),
            order_id=f"paper-{request.client_order_id}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=quantity,
            price=price,
            liquidity=liquidity,
            exchange_timestamp=orderbook.timestamp,
            receive_timestamp=orderbook.timestamp,
            raw={
                "slippage_cost": slippage_cost,
                "adverse_selection_cost": adverse_selection_cost,
                "orderbook_source": orderbook.source,
            },
        )

    def _build_report(
        self,
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
        fills: tuple[Fill, ...],
        resting: bool,
    ) -> ExecutionReport:
        """체결 목록에서 공용 실행 보고서를 집계한다."""
        filled_quantity = round(sum(fill.quantity for fill in fills), 8)
        remaining = round(max(0.0, request.quantity - filled_quantity), 8)
        average = (
            round(
                sum(fill.price * fill.quantity for fill in fills)
                / filled_quantity,
                8,
            )
            if filled_quantity > 0
            else None
        )
        if remaining <= 0:
            state = OrderState.FILLED
        elif filled_quantity > 0:
            state = OrderState.PARTIALLY_FILLED
        elif resting and request.time_in_force in {"GTC", "PostOnly"}:
            state = OrderState.ACCEPTED
        else:
            state = OrderState.CANCELED
        return ExecutionReport(
            order_id=f"paper-{request.client_order_id}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            state=state,
            requested_quantity=request.quantity,
            filled_quantity=filled_quantity,
            average_price=average,
            fills=fills,
            exchange_timestamp=orderbook.timestamp,
            receive_timestamp=orderbook.timestamp,
        )

    @staticmethod
    def _empty_report(
        request: OrderRequest,
        state: OrderState,
        orderbook: OrderBookSnapshot,
        reason: str,
    ) -> ExecutionReport:
        """체결 없는 공용 실행 보고서를 만든다."""
        logger.info(
            "주문 미체결: id=%s reason=%s",
            request.client_order_id,
            reason,
        )
        return ExecutionReport(
            order_id=f"paper-{request.client_order_id}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            state=state,
            requested_quantity=request.quantity,
            filled_quantity=0.0,
            average_price=None,
            exchange_timestamp=orderbook.timestamp,
            receive_timestamp=orderbook.timestamp,
            reject_reason=reason,
        )
