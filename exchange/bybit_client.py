"""Bybit Unified Trading API 래퍼."""

from pybit.unified_trading import HTTP
from loguru import logger


class BybitClient:

    def __init__(self, credentials: dict):
        self.testnet = credentials.get("testnet", True)
        self.http = HTTP(
            testnet=self.testnet,
            api_key=credentials.get("api_key", ""),
            api_secret=credentials.get("api_secret", ""),
        )

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.http.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            logger.info(f"레버리지 설정: {symbol} x{leverage}")
        except Exception as e:
            if "leverage not modified" in str(e).lower() or "110043" in str(e):
                logger.debug(f"레버리지 이미 {leverage}x로 설정됨")
            else:
                logger.warning(f"레버리지 설정 실패: {e}")

    def set_margin_mode(self, symbol: str, mode: str):
        trade_mode = 0 if mode == "cross" else 1
        try:
            self.http.switch_margin_mode(
                category="linear",
                symbol=symbol,
                tradeMode=trade_mode,
                buyLeverage="10",
                sellLeverage="10",
            )
            logger.info(f"마진 모드 설정: {symbol} → {mode}")
        except Exception as e:
            if "not modified" in str(e).lower() or "110026" in str(e):
                logger.debug(f"마진 모드 이미 {mode}")
            else:
                logger.warning(f"마진 모드 설정 실패: {e}")

    def get_balance(self) -> float:
        resp = self.http.get_wallet_balance(accountType="UNIFIED")
        coins = resp["result"]["list"][0]["coin"]
        for coin in coins:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0

    def get_positions(self, symbol: str = None) -> list:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = self.http.get_positions(**params)
        return resp["result"]["list"]

    def get_open_orders(self, symbol: str) -> list:
        resp = self.http.get_open_orders(
            category="linear", symbol=symbol
        )
        return resp["result"]["list"]

    def place_limit_order(self, order: dict) -> dict:
        resp = self.http.place_order(
            category="linear",
            symbol=order["symbol"],
            side=order["side"],
            orderType="Limit",
            qty=str(order["qty"]),
            price=str(order["entry"]),
            stopLoss=str(order["sl"]),
            takeProfit=str(order["tp"]),
            timeInForce="GTC",
        )
        logger.info(f"지정가 주문: {order['side']} {order['qty']} @ {order['entry']}")
        return resp

    def place_market_order(self, symbol: str, side: str, qty: float,
                           sl: float = None, tp: float = None) -> dict:
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
        }
        if sl is not None:
            params["stopLoss"] = str(sl)
        if tp is not None:
            params["takeProfit"] = str(tp)

        resp = self.http.place_order(**params)
        logger.info(f"시장가 주문: {side} {qty} {symbol}")
        return resp

    def cancel_all_orders(self, symbol: str):
        resp = self.http.cancel_all_orders(category="linear", symbol=symbol)
        logger.info(f"전체 주문 취소: {symbol}")
        return resp

    def close_position(self, symbol: str, side: str, qty: float) -> dict:
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_market_order(symbol, close_side, qty)

    def get_closed_pnl(self, symbol: str, limit: int = 50) -> list:
        resp = self.http.get_closed_pnl(
            category="linear", symbol=symbol, limit=limit
        )
        return resp["result"]["list"]
