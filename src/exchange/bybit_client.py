"""
Bybit 퍼블릭 API 클라이언트 — API 키 불필요
시세/캔들 조회만 사용 (페이퍼 트레이딩용)
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import time
from typing import Any, Sequence

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_EXCHANGES = ("binanceus", "kraken", "coinbase")


@dataclass(frozen=True)
class _MarketSource:
    name: str
    exchange: Any
    is_primary: bool = False


class BybitPublicClient:
    """Bybit 우선 퍼블릭 시세 클라이언트.

    GitHub-hosted runner처럼 Bybit이 지역/IP 정책으로 차단되는 환경에서는
    페이퍼 트레이딩용 가격/캔들을 공개 spot 데이터 소스로 자동 fallback 한다.
    """

    def __init__(
        self,
        fallback_exchange_names: Sequence[str] = DEFAULT_FALLBACK_EXCHANGES,
    ) -> None:
        self.sources: list[_MarketSource] = []

        bybit = self._build_source("bybit", is_primary=True, default_type="future")
        if bybit is not None:
            self.sources.append(bybit)

        for name in fallback_exchange_names:
            source = self._build_source(name)
            if source is not None:
                self.sources.append(source)

        if not self.sources:
            raise RuntimeError("사용 가능한 시세 데이터 소스가 없습니다.")

        logger.info(
            "퍼블릭 시세 클라이언트 초기화: %s",
            " -> ".join(source.name for source in self.sources),
        )

    def _build_source(
        self,
        name: str,
        *,
        is_primary: bool = False,
        default_type: str | None = None,
    ) -> _MarketSource | None:
        exchange_cls = getattr(ccxt, name, None)
        if exchange_cls is None:
            logger.warning("ccxt 거래소 미지원: %s", name)
            return None

        config: dict[str, Any] = {"enableRateLimit": True}
        if default_type:
            config["options"] = {"defaultType": default_type}

        return _MarketSource(name=name, exchange=exchange_cls(config), is_primary=is_primary)

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

    def _symbol_candidates(self, source_name: str, symbol: str) -> tuple[str, ...]:
        """Return exchange-specific symbol candidates.

        Bybit futures symbols look like ``BTC/USDT:USDT``. Spot fallback exchanges
        generally use ``BTC/USDT`` or ``BTC/USD``.
        """
        spot_symbol = symbol.split(":", 1)[0]
        candidates: list[str] = []

        if source_name == "bybit":
            candidates.append(symbol)

        candidates.append(spot_symbol)

        if "/" in spot_symbol:
            base, quote = spot_symbol.split("/", 1)
            quote = quote.upper()
            alternate_quotes: tuple[str, ...] = ()
            if quote in {"USDT", "USDC"}:
                alternate_quotes = ("USD", "USDC", "USDT")
            elif quote == "USD":
                alternate_quotes = ("USDT", "USDC", "USD")

            for alternate_quote in alternate_quotes:
                if alternate_quote != quote:
                    candidates.append(f"{base}/{alternate_quote}")

        return tuple(dict.fromkeys(candidates))

    def _fetch_from_sources(self, method_name: str, symbol: str, *args, **kwargs) -> Any:
        failures: list[str] = []

        for source in self.sources:
            method = getattr(source.exchange, method_name)
            for data_symbol in self._symbol_candidates(source.name, symbol):
                try:
                    result = self._retry(method, data_symbol, *args, **kwargs)
                    if not source.is_primary or data_symbol != symbol:
                        logger.info(
                            "[%s] %s 데이터 소스 사용: %s (%s)",
                            symbol,
                            method_name,
                            source.name,
                            data_symbol,
                        )
                    return result
                except ccxt.BadSymbol as e:
                    failures.append(f"{source.name}:{data_symbol} {type(e).__name__}: {e}")
                    continue
                except ccxt.BaseError as e:
                    failures.append(f"{source.name}:{data_symbol} {type(e).__name__}: {e}")
                    logger.warning(
                        "[%s] %s 데이터 소스 실패: %s (%s) — %s",
                        symbol,
                        method_name,
                        source.name,
                        data_symbol,
                        e,
                    )
                    break

        detail = "; ".join(failures[-6:]) or "unknown error"
        raise RuntimeError(f"[{symbol}] 사용 가능한 시세 데이터 소스가 없습니다: {detail}")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> pd.DataFrame:
        """실시간 캔들 데이터 조회 (인증 불필요)."""
        raw = self._fetch_from_sources("fetch_ohlcv", symbol, timeframe, limit=limit)
        if not raw:
            raise RuntimeError(f"[{symbol}] 캔들 데이터가 비어 있습니다.")

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """현재 시세 조회."""
        return self._fetch_from_sources("fetch_ticker", symbol)

    def fetch_current_price(self, symbol: str) -> float:
        """현재가만 빠르게 조회."""
        ticker = self.fetch_ticker(symbol)
        return float(ticker["last"])
