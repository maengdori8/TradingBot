"""
Bybit 퍼블릭 API 클라이언트 — API 키 불필요
시세/캔들 조회만 사용 (페이퍼 트레이딩용)
"""
from __future__ import annotations
import logging
import time
from typing import Any
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BybitPublicClient:
    """Bybit 퍼블릭 엔드포인트 전용 — 인증 불필요."""

    def __init__(self) -> None:
        self.exchange = ccxt.bybit({
            "options": {"defaultType": "future"},
        })
        logger.info("Bybit 퍼블릭 클라이언트 초기화 (API 키 불필요)")

    def _retry(self, func, *args, max_retries: int = 3, **kwargs) -> Any:
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                wait = 2 ** attempt
                logger.warning("재시도 %d/%d: %s (%ds 대기)", attempt, max_retries, e, wait)
                if attempt == max_retries:
                    raise
                time.sleep(wait)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> pd.DataFrame:
        """실시간 캔들 데이터 조회 (인증 불필요)."""
        raw = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """현재 시세 조회."""
        return self._retry(self.exchange.fetch_ticker, symbol)

    def fetch_current_price(self, symbol: str) -> float:
        """현재가만 빠르게 조회."""
        ticker = self.fetch_ticker(symbol)
        return float(ticker["last"])
