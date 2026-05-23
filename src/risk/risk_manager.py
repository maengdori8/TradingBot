from __future__ import annotations
import logging
from pathlib import Path
import yaml

from src.risk.position_sizer import calculate_position_size, calculate_take_profit
from src.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent


def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


class RiskManager:
    def __init__(self) -> None:
        cfg = load_config()
        cap  = cfg["capital"]
        risk = cfg["risk"]

        self.total_capital    = cap["total_capital"]
        self.trading_capital  = self.total_capital * cap["trading_allocation"]
        self.risk_per_trade   = cap["risk_per_trade"]
        self.leverage         = cfg["exchange"]["leverage"]
        self.min_rr           = risk["min_rr_ratio"]
        self.max_positions    = risk["max_positions"]

        self.cb = CircuitBreaker(
            trading_capital       = self.trading_capital,
            daily_loss_limit      = risk["daily_loss_limit"],
            weekly_loss_limit     = risk["weekly_loss_limit"],
            max_consecutive_losses= risk["max_consecutive_losses"],
        )
        logger.info("RiskManager: 트레이딩 자본=%.0f USDT", self.trading_capital)

    def check_trade_allowed(self, current_positions: int) -> tuple[bool, str]:
        allowed, reason = self.cb.is_trading_allowed()
        if not allowed:
            return False, reason
        if current_positions >= self.max_positions:
            return False, f"최대 포지션 초과 ({current_positions}/{self.max_positions})"
        return True, "OK"

    def calculate_trade_params(self, entry: float, stop_loss: float) -> dict:
        qty = calculate_position_size(
            self.trading_capital, self.risk_per_trade,
            entry, stop_loss, self.leverage,
        )
        tp = calculate_take_profit(entry, stop_loss, self.min_rr)
        return {"qty": qty, "entry": entry, "stop_loss": stop_loss, "take_profit": tp}

    def record_result(self, pnl: float) -> None:
        self.cb.record_trade(pnl)
