"""
멀티 거래소 퍼블릭 클라이언트
Bybit → BinanceUS → Kraken → Coinbase fallback
선물 심볼 자동 변환
"""
from __future__ import annotations
import logging
import time
from typing import Any
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# (이름, 클래스, 옵션, 선물심볼 지원여부)
EXCHANGE_CONFIGS = [
    ("bybit",     ccxt.bybit,     {"options": {"defaultType": "future"}}, True),
    ("binanceus", ccxt.binanceus, {},                                      False),
    ("kraken",    ccxt.kraken,    {},                                      False),
    ("coinbase",  ccxt.coinbase,  {},                                      False),
]


def _spot_symbols(symbol: str) -> list[str]:
    """'BTC/USDT:USDT' → ['BTC/USDT', 'BTC/USD']"""
    base = symbol.split(":")[0]
    coin, quote = base.split("/")
    result = [base]
    if quote == "USDT":
        result.append(f"{coin}/USD")
    return result


class MarketDataClient:
    """퍼블릭 시세 전용 — 인증 불필요."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        for name, cls, cfg, _ in EXCHANGE_CONFIGS:
            try:
                self._clients[name] = cls(cfg)
            except Exception as e:
                logger.warning("거래소 초기화 실패 [%s]: %s", name, e)
        logger.info("거래소 초기화 완료: %s", list(self._clients))

    def _call(self, method: str, symbol: str, *args, **kwargs) -> tuple[Any, str]:
        """순서대로 시도, 첫 성공 반환."""
        for name, _, _, futures_ok in EXCHANGE_CONFIGS:
            client = self._clients.get(name)
            if client is None:
                continue

            candidates = [symbol] if futures_ok else _spot_symbols(symbol)

            for sym in candidates:
                try:
                    result = getattr(client, method)(sym, *args, **kwargs)
                    if name != "bybit":
                        logger.info("[fallback] %s → %s (%s)", symbol, name, sym)
                    return result, name
                except Exception as e:
                    logger.debug("[%s/%s] %s", name, sym, e)

        raise RuntimeError(f"모든 거래소 실패: {symbol}")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> pd.DataFrame:
        raw, src = self._call("fetch_ohlcv", symbol, timeframe, None, {"limit": limit})
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    def fetch_ticker(self, symbol: str) -> dict:
        ticker, _ = self._call("fetch_ticker", symbol)
        return ticker

    def fetch_current_price(self, symbol: str) -> float:
        return float(self.fetch_ticker(symbol)["last"])


# 하위 호환
BybitPublicClient = MarketDataClient
