"""
멀티 거래소 퍼블릭 클라이언트
Bybit → BinanceUS → Kraken → Coinbase fallback
선물 심볼 자동 변환
"""
from __future__ import annotations
import logging
from typing import Any
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

EXCHANGE_CONFIGS = [
    ("bybit",     ccxt.bybit,     {"options": {"defaultType": "swap"}}, True),
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
    """퍼블릭 시세 전용 클라이언트."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        for name, cls, cfg, _ in EXCHANGE_CONFIGS:
            try:
                self._clients[name] = cls(cfg)
            except Exception as e:
                logger.warning("[%s] 초기화 실패: %s", name, e)
        logger.info("거래소 준비: %s", list(self._clients))

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> pd.DataFrame:
        """캔들 데이터 조회 — fallback 포함."""
        for name, _, _, futures_ok in EXCHANGE_CONFIGS:
            client = self._clients.get(name)
            if not client:
                continue
            candidates = [symbol] if futures_ok else _spot_symbols(symbol)
            for sym in candidates:
                try:
                    # 인자를 명시적으로 전달 (params dict 방식 제거)
                    raw = client.fetch_ohlcv(sym, timeframe, limit=limit)
                    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp")
                    if name != "bybit":
                        logger.info("[fallback] %s → %s/%s", symbol, name, sym)
                    return df
                except Exception as e:
                    logger.debug("[%s/%s] 실패: %s", name, sym, str(e)[:80])

        raise RuntimeError(f"모든 거래소 실패: {symbol}")

    def fetch_ticker(self, symbol: str) -> dict:
        """현재가 조회 — fallback 포함."""
        for name, _, _, futures_ok in EXCHANGE_CONFIGS:
            client = self._clients.get(name)
            if not client:
                continue
            candidates = [symbol] if futures_ok else _spot_symbols(symbol)
            for sym in candidates:
                try:
                    ticker = client.fetch_ticker(sym)
                    if name != "bybit":
                        logger.info("[fallback] ticker %s → %s/%s", symbol, name, sym)
                    return ticker
                except Exception as e:
                    logger.debug("[%s/%s] ticker 실패: %s", name, sym, str(e)[:80])

        raise RuntimeError(f"ticker 조회 실패: {symbol}")

    def fetch_current_price(self, symbol: str) -> float:
        return float(self.fetch_ticker(symbol)["last"])


# 하위 호환
BybitPublicClient = MarketDataClient
