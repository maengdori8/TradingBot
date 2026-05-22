"""
리스크 관리 통합 모듈
"""
from __future__ import annotations
import logging
import yaml
from pathlib import Path

from .position_sizer import calculate_position_size, calculate_stop_loss_atr, calculate_take_profit
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class RiskManager:
    """전체 리스크 관리 통합."""

    def __init__(self) -> None:
        cfg = load_config()
        cap = cfg["capital"]
        risk = cfg["risk"]

        self.total_capital = cap["total_capital"]
        self.trading_capital = self.total_capital * cap["trading_allocation"]
        self.risk_per_trade = cap["risk_per_trade"]
        self.leverage = cfg["exchange"]["leverage"]
        self.min_rr = risk["min_rr_ratio"]
        self.max_positions = risk["max_positions"]

        self.cb = CircuitBreaker(
            trading_capital=self.trading_capital,
            daily_loss_limit=risk["daily_loss_limit"],
            weekly_loss_limit=risk["weekly_loss_limit"],
            max_consecutive_losses=risk["max_consecutive_losses"],
        )
        logger.info("RiskManager 초기화: 트레이딩 자본=%.0f USDT", self.trading_capital)

    def check_trade_allowed(self, current_positions: int) -> tuple[bool, str]:
        """거래 허용 여부 종합 체크."""
        allowed, reason = self.cb.is_trading_allowed()
        if not allowed:
            return False, reason
        if current_positions >= self.max_positions:
            return False, f"최대 포지션 수 초과: {current_positions}/{self.max_positions}"
        return True, "OK"

    def calculate_trade_params(self, entry: float, stop_loss: float) -> dict:
        """진입/SL/TP/수량 계산."""
        qty = calculate_position_size(
            capital=self.trading_capital,
            risk_pct=self.risk_per_trade,
            entry_price=entry,
            stop_loss_price=stop_loss,
            leverage=self.leverage,
        )
        tp = calculate_take_profit(entry, stop_loss, self.min_rr)
        return {"qty": qty, "entry": entry, "stop_loss": stop_loss, "take_profit": tp}

    def record_result(self, pnl: float) -> None:
        """거래 결과 기록."""
        self.cb.record_trade(pnl)
