"""
주문 실행 추상 인터페이스 및 페이퍼 트레이딩 구현.
페이퍼/실전 공통 주문 인터페이스를 정의하고,
PaperEngine과 연동하는 페이퍼 주문 실행기를 제공한다.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

logger = logging.getLogger(__name__)


class OrderExecutor(ABC):
    """주문 실행 추상 인터페이스.

    페이퍼 트레이딩과 실전 트레이딩 모두 이 인터페이스를 구현한다.
    """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        qty: float,
        order_type: Literal["market", "limit"],
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """주문을 실행한다.

        Args:
            symbol: 거래 심볼
            direction: 매매 방향 ('long' 또는 'short')
            qty: 수량
            order_type: 주문 유형 ('market' 또는 'limit')
            price: 지정가 (limit 주문 시 필수)
            stop_loss: 손절가
            take_profit: 익절가

        Returns:
            주문 결과 dict (order_id, status 등)
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """주문을 취소한다.

        Args:
            order_id: 취소할 주문 ID

        Returns:
            취소 성공 여부
        """
        ...

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """미체결 주문 목록을 조회한다.

        Args:
            symbol: 특정 심볼 (None이면 전체)

        Returns:
            미체결 주문 리스트
        """
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """현재 포지션을 조회한다.

        Args:
            symbol: 거래 심볼

        Returns:
            포지션 정보 dict 또는 None
        """
        ...


class PaperOrderExecutor(OrderExecutor):
    """페이퍼 트레이딩용 주문 실행기.

    PaperEngine의 로직을 위임받아 OrderExecutor 인터페이스를 충족한다.
    """

    def __init__(self, paper_engine: Any) -> None:
        """PaperOrderExecutor를 초기화한다.

        Args:
            paper_engine: PaperEngine 인스턴스
        """
        self._engine = paper_engine
        self._order_counter: int = 0
        self._pending_orders: dict[str, dict[str, Any]] = {}
        logger.info("PaperOrderExecutor 초기화 완료")

    def place_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        qty: float,
        order_type: Literal["market", "limit"],
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """페이퍼 주문을 실행한다.

        market 주문은 즉시 PaperEngine으로 포지션을 생성하고,
        limit 주문은 미체결 목록에 추가한다.

        Args:
            symbol: 거래 심볼
            direction: 매매 방향
            qty: 수량
            order_type: 주문 유형
            price: 지정가 (limit 주문 시 필수)
            stop_loss: 손절가
            take_profit: 익절가

        Returns:
            주문 결과 dict
        """
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        if order_type == "limit" and price is None:
            logger.error("limit 주문에 price가 필요합니다: %s", order_id)
            return {"order_id": order_id, "status": "rejected", "reason": "price required"}

        if order_type == "market":
            # market 주문: PaperEngine에 즉시 위임
            entry_price = price if price is not None else 0.0
            sl = stop_loss if stop_loss is not None else 0.0
            tp = take_profit if take_profit is not None else 0.0

            pos = self._engine.open_position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                qty=qty,
                stop_loss=sl,
                take_profit=tp,
            )

            if pos is None:
                logger.warning("[PAPER] 주문 실패 (잔고 부족): %s", order_id)
                return {"order_id": order_id, "status": "rejected", "reason": "insufficient_balance"}

            logger.info("[PAPER] 시장가 주문 체결: %s %s %s qty=%.4f", order_id, symbol, direction, qty)
            return {
                "order_id": order_id,
                "status": "filled",
                "symbol": symbol,
                "direction": direction,
                "qty": qty,
                "filled_price": pos.entry_price,
            }

        # limit 주문: 미체결 대기열에 추가
        order_info: dict[str, Any] = {
            "order_id": order_id,
            "status": "pending",
            "symbol": symbol,
            "direction": direction,
            "qty": qty,
            "order_type": order_type,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        self._pending_orders[order_id] = order_info
        logger.info(
            "[PAPER] 지정가 주문 등록: %s %s %s price=%.4f qty=%.4f",
            order_id, symbol, direction, price, qty,  # type: ignore[arg-type]
        )
        return order_info

    def cancel_order(self, order_id: str) -> bool:
        """미체결 주문을 취소한다.

        Args:
            order_id: 취소할 주문 ID

        Returns:
            취소 성공 여부
        """
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            logger.info("[PAPER] 주문 취소: %s", order_id)
            return True
        logger.warning("[PAPER] 취소 실패 — 주문 없음: %s", order_id)
        return False

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """미체결 주문 목록을 조회한다.

        Args:
            symbol: 특정 심볼 (None이면 전체)

        Returns:
            미체결 주문 리스트
        """
        orders = list(self._pending_orders.values())
        if symbol is not None:
            orders = [o for o in orders if o["symbol"] == symbol]
        return orders

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """PaperEngine에서 현재 포지션을 조회한다.

        Args:
            symbol: 거래 심볼

        Returns:
            포지션 정보 dict 또는 None
        """
        for pos in self._engine.positions:
            if pos.symbol == symbol:
                return {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "qty": pos.qty,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "margin": pos.margin,
                    "entry_time": pos.entry_time.isoformat(),
                }
        return None
