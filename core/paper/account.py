"""가상 계좌 — 잔고, 포지션, 주문 시뮬레이션."""

import time
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Position:
    symbol: str
    side: str           # "Buy" / "Sell"
    entry_price: float
    qty: float
    sl: float
    tp: float
    open_time: float
    reason: str = ""
    unrealized_pnl: float = 0.0

    @property
    def direction(self) -> int:
        return 1 if self.side == "Buy" else -1

    def update_pnl(self, current_price: float):
        self.unrealized_pnl = (current_price - self.entry_price) * self.direction * self.qty


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    open_time: float
    close_time: float
    reason: str
    exit_reason: str    # "tp" / "sl" / "manual"


class PaperAccount:

    def __init__(self, initial_balance: float = 10000.0, leverage: int = 10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.peak_balance = initial_balance

    def get_balance(self) -> float:
        return self.balance

    def get_equity(self) -> float:
        return self.balance + sum(p.unrealized_pnl for p in self.positions)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        result = []
        for p in self.positions:
            if symbol and p.symbol != symbol:
                continue
            result.append({"size": str(p.qty), "side": p.side, "symbol": p.symbol})
        return result

    def open_position(self, order: dict) -> bool:
        margin_required = (order["qty"] * order["entry"]) / self.leverage
        if margin_required > self.balance:
            logger.warning(
                f"[PAPER] 마진 부족: 필요 {margin_required:.2f} > 잔고 {self.balance:.2f}"
            )
            return False

        pos = Position(
            symbol=order["symbol"],
            side=order["side"],
            entry_price=order["entry"],
            qty=order["qty"],
            sl=order["sl"],
            tp=order["tp"],
            open_time=time.time(),
            reason=order.get("reason", ""),
        )
        self.positions.append(pos)
        logger.info(
            f"[PAPER] 포지션 오픈: {pos.side} {pos.qty} {pos.symbol} "
            f"@ {pos.entry_price} | SL: {pos.sl} TP: {pos.tp}"
        )
        return True

    def check_exits(self, symbol: str, current_price: float) -> List[ClosedTrade]:
        closed = []
        remaining = []

        for pos in self.positions:
            if pos.symbol != symbol:
                remaining.append(pos)
                continue

            pos.update_pnl(current_price)
            exit_reason = None
            exit_price = current_price

            if pos.side == "Buy":
                if current_price <= pos.sl:
                    exit_reason = "sl"
                    exit_price = pos.sl
                elif current_price >= pos.tp:
                    exit_reason = "tp"
                    exit_price = pos.tp
            else:
                if current_price >= pos.sl:
                    exit_reason = "sl"
                    exit_price = pos.sl
                elif current_price <= pos.tp:
                    exit_reason = "tp"
                    exit_price = pos.tp

            if exit_reason:
                pnl = (exit_price - pos.entry_price) * pos.direction * pos.qty
                trade = ClosedTrade(
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    qty=pos.qty,
                    pnl=pnl,
                    open_time=pos.open_time,
                    close_time=time.time(),
                    reason=pos.reason,
                    exit_reason=exit_reason,
                )
                self.closed_trades.append(trade)
                self.balance += pnl
                self.peak_balance = max(self.peak_balance, self.balance)
                closed.append(trade)

                emoji = "+" if pnl >= 0 else ""
                logger.info(
                    f"[PAPER] 포지션 청산 ({exit_reason.upper()}): "
                    f"{pos.side} {pos.qty} {pos.symbol} | "
                    f"진입 {pos.entry_price} -> 청산 {exit_price} | "
                    f"PnL: {emoji}{pnl:.2f} USDT"
                )
            else:
                remaining.append(pos)

        self.positions = remaining
        return closed

    def get_stats(self) -> dict:
        total = len(self.closed_trades)
        if total == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "balance": self.balance,
                "equity": self.get_equity(),
                "return_pct": 0.0,
                "open_positions": len(self.positions),
            }

        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        pnls = [t.pnl for t in self.closed_trades]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        running_balance = self.initial_balance
        peak = running_balance
        max_dd = 0.0
        for t in self.closed_trades:
            running_balance += t.pnl
            peak = max(peak, running_balance)
            dd = (peak - running_balance) / peak
            max_dd = max(max_dd, dd)

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / total * 100,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / total,
            "max_drawdown": max_dd * 100,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "avg_winner": gross_profit / len(wins) if wins else 0,
            "avg_loser": -gross_loss / len(losses) if losses else 0,
            "balance": self.balance,
            "equity": self.get_equity(),
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
            "open_positions": len(self.positions),
        }
