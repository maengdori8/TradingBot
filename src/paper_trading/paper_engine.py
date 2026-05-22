"""
페이퍼 트레이딩 엔진 — 가상 주문 실행 및 성과 추적
슬리피지 0.05%, 수수료 0.055% (Bybit Taker 기준)
잔고 = 담보금 + 미실현손익 기반으로 관리
"""
from __future__ import annotations
import sqlite3
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "paper_trades.db"

SLIPPAGE  = 0.0005    # 0.05%
TAKER_FEE = 0.00055   # 0.055%


def _init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL,
            qty REAL, pnl REAL, pnl_pct REAL,
            entry_time TEXT, exit_time TEXT, status TEXT
        )
    """)
    conn.commit()
    return conn


@dataclass
class PaperPosition:
    symbol: str
    direction: Literal["long", "short"]
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    margin: float = 0.0   # 담보금 (잔고에서 차감된 금액)
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaperEngine:
    """페이퍼 트레이딩 엔진."""

    def __init__(self, initial_balance: float = 1250.0) -> None:
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: list[PaperPosition] = []
        self.conn = _init_db()
        logger.info("페이퍼 엔진 초기화: 잔고=%.2f USDT", initial_balance)

    def _apply_slippage(self, price: float, direction: Literal["long", "short"], is_entry: bool) -> float:
        # Long 진입 / Short 청산: 불리한 방향 = 더 높은 가격
        if (direction == "long") == is_entry:
            return price * (1 + SLIPPAGE)
        return price * (1 - SLIPPAGE)

    def _fee(self, notional: float) -> float:
        return notional * TAKER_FEE

    def open_position(
        self, symbol: str, direction: Literal["long", "short"],
        entry_price: float, qty: float,
        stop_loss: float, take_profit: float,
    ) -> "PaperPosition | None":
        """포지션 진입 — 담보금 및 수수료 차감."""
        actual_entry = self._apply_slippage(entry_price, direction, is_entry=True)
        notional = actual_entry * qty
        entry_fee = self._fee(notional)
        total_cost = notional + entry_fee   # 담보금 + 수수료

        if total_cost > self.balance:
            logger.warning("잔고 부족: 필요=%.2f 보유=%.2f", total_cost, self.balance)
            return None

        self.balance -= total_cost          # 담보금 + 수수료 차감
        pos = PaperPosition(
            symbol=symbol, direction=direction,
            entry_price=actual_entry, qty=qty,
            stop_loss=stop_loss, take_profit=take_profit,
            margin=notional,
        )
        self.positions.append(pos)
        logger.info("[PAPER] %s %s 진입: price=%.4f qty=%.4f margin=%.2f",
                    symbol, direction.upper(), actual_entry, qty, notional)
        return pos

    def close_position(self, pos: PaperPosition, exit_price: float, reason: str = "") -> float:
        """포지션 청산 — 담보금 반환 + PnL."""
        actual_exit = self._apply_slippage(exit_price, pos.direction, is_entry=False)
        exit_fee = self._fee(actual_exit * pos.qty)

        if pos.direction == "long":
            gross_pnl = (actual_exit - pos.entry_price) * pos.qty
        else:
            gross_pnl = (pos.entry_price - actual_exit) * pos.qty

        pnl = gross_pnl - exit_fee
        pnl_pct = pnl / pos.margin if pos.margin > 0 else 0

        # 담보금 반환 + 순이익
        self.balance += pos.margin + pnl

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO trades(symbol,direction,entry_price,exit_price,qty,pnl,pnl_pct,entry_time,exit_time,status)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (pos.symbol, pos.direction, pos.entry_price, actual_exit,
              pos.qty, pnl, pnl_pct, pos.entry_time.isoformat(), now, reason))
        self.conn.commit()
        self.positions.remove(pos)
        logger.info("[PAPER] %s 청산(%s): PnL=%.4f (%.2f%%)", pos.symbol, reason, pnl, pnl_pct * 100)
        return pnl

    def check_stops(self, symbol: str, current_high: float, current_low: float) -> None:
        """SL/TP 자동 체크."""
        for pos in list(self.positions):
            if pos.symbol != symbol:
                continue
            if pos.direction == "long":
                if current_low <= pos.stop_loss:
                    self.close_position(pos, pos.stop_loss, "SL")
                elif current_high >= pos.take_profit:
                    self.close_position(pos, pos.take_profit, "TP")
            else:
                if current_high >= pos.stop_loss:
                    self.close_position(pos, pos.stop_loss, "SL")
                elif current_low <= pos.take_profit:
                    self.close_position(pos, pos.take_profit, "TP")

    def get_performance(self) -> dict:
        """성과 지표 계산."""
        rows = self.conn.execute(
            "SELECT pnl, pnl_pct FROM trades"
        ).fetchall()

        if not rows:
            return {"message": "거래 내역 없음"}

        pnls = [r[0] for r in rows]
        pnl_pcts = [r[1] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity = np.array([self.initial_balance] + list(np.cumsum(pnls) + self.initial_balance))
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
        mdd = float(drawdown.max())

        pnl_arr = np.array(pnl_pcts)
        sharpe = float(pnl_arr.mean() / pnl_arr.std() * math.sqrt(252)) if pnl_arr.std() > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss

        return {
            "total_trades":  len(pnls),
            "win_rate":      len(wins) / len(pnls),
            "avg_pnl":       sum(pnls) / len(pnls),
            "total_pnl":     sum(pnls),
            "mdd":           mdd,
            "sharpe":        sharpe,
            "profit_factor": profit_factor,
            "current_balance": self.balance,
            "return_pct":    (self.balance - self.initial_balance) / self.initial_balance,
        }
