from __future__ import annotations

# 트레이딩 엔진 추상 인터페이스 — 페이퍼/실전 공통 계약.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from src.exchange.contracts import TradingMode


@dataclass
class Position:
    """페이퍼/실전 공통 포지션 데이터 클래스."""

    id: str
    symbol: str
    direction: Literal["long", "short"]
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    margin: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unrealized_pnl: float = 0.0
    # 진입 시점 ICT 맥락 (자동 학습용, 모두 옵셔널 = 하위호환)
    entry_score: float | None = None       # 컨플루언스 점수 0~100
    entry_session: str | None = None       # "london" | "newyork" | None
    entry_checks: dict | None = None        # ICT 단계별 통과여부
    entry_rr: float | None = None           # 진입 시 목표 R:R
    risk_amount: float | None = None        # |entry-stop|*qty (R-multiple 산출용)
    entry_fee: float = 0.0                  # 미청산 수량에 귀속된 진입 수수료
    entry_slippage_cost: float = 0.0         # 미청산 수량에 귀속된 진입 슬리피지
    entry_requested_price: float | None = None
    entry_liquidity: Literal["maker", "taker"] = "taker"
    requested_qty: float | None = None


class TradingEngine(ABC):
    """페이퍼/실전 공통 트레이딩 엔진 인터페이스."""

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        entry_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
    ) -> Position | None:
        """포지션 진입."""
        ...

    @abstractmethod
    def close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str,
        qty: float | None = None,
    ) -> float:
        """포지션 청산 (부분 청산 지원)."""
        ...

    @abstractmethod
    def check_stops(
        self, symbol: str, current_high: float, current_low: float
    ) -> None:
        """SL/TP 자동 체크."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """현재 보유 포지션 목록."""
        ...

    @abstractmethod
    def get_performance(self) -> dict:
        """성과 지표 조회."""
        ...

    @property
    @abstractmethod
    def balance(self) -> float:
        """현재 잔고."""
        ...
