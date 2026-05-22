"""리스크 관리 — 포지션 사이징, 일일 손실 한도, 최대 포지션 수 제한."""

import math
from typing import Optional
from loguru import logger


class RiskManager:

    def __init__(self, config: dict, client, symbol: str):
        self.max_risk = config.get("max_risk_per_trade", 0.01)
        self.max_daily_loss = config.get("max_daily_loss", 0.03)
        self.max_positions = config.get("max_open_positions", 3)
        self.rr_ratio = config.get("risk_reward_ratio", 2.0)
        self.use_trailing = config.get("trailing_stop", False)
        self.client = client
        self.symbol = symbol
        self.daily_pnl = 0.0
        self._instrument_info: Optional[dict] = None

    def load_instrument_info(self, info: dict):
        self._instrument_info = info

    def _round_qty(self, qty: float) -> float:
        if self._instrument_info is None:
            return round(qty, 3)
        step = self._instrument_info["qty_step"]
        min_qty = self._instrument_info["min_qty"]
        qty = math.floor(qty / step) * step
        qty = round(qty, 8)
        if qty < min_qty:
            return 0.0
        return qty

    def _round_price(self, price: float) -> float:
        if self._instrument_info is None:
            return round(price, 2)
        tick = self._instrument_info["tick_size"]
        return round(round(price / tick) * tick, 8)

    def can_open_trade(self, signal: dict) -> bool:
        positions = self.client.get_positions(self.symbol)
        open_count = sum(1 for p in positions if float(p.get("size", 0)) > 0)
        if open_count >= self.max_positions:
            logger.warning(f"최대 포지션 수 도달: {open_count}/{self.max_positions}")
            return False

        balance = self.client.get_balance()
        if balance <= 0:
            logger.warning("잔고 부족")
            return False

        if abs(self.daily_pnl) >= balance * self.max_daily_loss:
            logger.warning(
                f"일일 최대 손실 한도 도달: {self.daily_pnl:.2f} "
                f"(한도: {balance * self.max_daily_loss:.2f})"
            )
            return False

        return True

    def size_position(self, signal: dict) -> Optional[dict]:
        balance = self.client.get_balance()
        entry = signal["entry"]
        sl = signal["sl"]
        risk_amount = balance * self.max_risk
        distance = abs(entry - sl)

        if distance == 0:
            logger.warning("SL과 진입가가 동일 — 포지션 사이징 불가")
            return None

        qty = risk_amount / distance
        qty = self._round_qty(qty)

        if qty <= 0:
            logger.warning(f"수량이 최소 주문 단위 미만: risk={risk_amount:.2f}, dist={distance:.4f}")
            return None

        if signal["side"] == "Buy":
            tp = entry + distance * self.rr_ratio
        else:
            tp = entry - distance * self.rr_ratio

        order = {
            "symbol": self.symbol,
            "side": signal["side"],
            "entry": self._round_price(entry),
            "sl": self._round_price(sl),
            "tp": self._round_price(tp),
            "qty": qty,
            "risk_amount": round(risk_amount, 2),
            "reason": signal.get("reason", ""),
        }

        logger.info(
            f"포지션 사이징: {order['side']} {order['qty']} @ {order['entry']} | "
            f"SL: {order['sl']} TP: {order['tp']} | 리스크: ${order['risk_amount']}"
        )
        return order

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        logger.info(f"일일 PnL: {self.daily_pnl:+.2f} USDT")

    def sync_daily_pnl(self):
        records = self.client.get_closed_pnl(self.symbol, limit=50)
        total = sum(float(r.get("closedPnl", 0)) for r in records)
        self.daily_pnl = total
        logger.info(f"일일 PnL 동기화: {self.daily_pnl:+.2f} USDT")

    def reset_daily_pnl(self):
        self.daily_pnl = 0.0
        logger.info("일일 PnL 초기화")
