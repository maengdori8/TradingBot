"""
Bybit API 클라이언트 — ccxt 기반
Testnet / 실거래 전환을 config로 제어, 자동 재시도 포함
"""
from __future__ import annotations
import logging
import time
from typing import Any
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BybitClient:
    """Bybit Futures API 래퍼."""

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True) -> None:
        self.testnet = testnet
        self.exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("Bybit Testnet 모드")
        else:
            logger.info("Bybit 실거래 모드")

    def _retry(self, func, *args, max_retries: int = 3, **kwargs) -> Any:
        """최대 3회 재시도, exponential backoff."""
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                wait = 2 ** attempt
                logger.warning("API 오류 (%d/%d): %s — %ds 후 재시도", attempt, max_retries, e, wait)
                if attempt == max_retries:
                    raise
                time.sleep(wait)
            except ccxt.BaseError as e:
                logger.error("ccxt 오류: %s", e)
                raise

    def fetch_balance(self) -> dict[str, Any]:
        """USDT 잔고 조회."""
        return self._retry(self.exchange.fetch_balance)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        """OHLCV 캔들 데이터 조회."""
        raw = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    def fetch_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """오픈 포지션 조회."""
        positions = self._retry(self.exchange.fetch_positions, [symbol] if symbol else None)
        return [p for p in positions if float(p.get("contracts", 0)) != 0]

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """현재 가격 조회."""
        return self._retry(self.exchange.fetch_ticker, symbol)

    def create_order(self, symbol: str, order_type: str, side: str, amount: float,
                     price: float | None = None, params: dict | None = None) -> dict[str, Any]:
        """주문 생성."""
        logger.info("주문: %s %s %s qty=%.4f", symbol, side, order_type, amount)
        return self._retry(self.exchange.create_order, symbol, order_type, side, amount, price, params or {})

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """주문 취소."""
        return self._retry(self.exchange.cancel_order, order_id, symbol)

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """레버리지 설정."""
        self._retry(self.exchange.set_leverage, leverage, symbol)
