"""
캔들 데이터 수집 및 캐싱 모듈.
MarketDataClient를 래핑하여 타임프레임별 TTL 캐시를 제공한다.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from src.exchange.bybit_client import MarketDataClient

logger = logging.getLogger(__name__)

# 타임프레임별 캐시 만료 시간 (초)
_CACHE_TTL: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

# 기본 TTL (매핑에 없는 타임프레임)
_DEFAULT_TTL: int = 900


def _get_ttl(timeframe: str) -> int:
    """타임프레임 문자열에 해당하는 TTL(초)을 반환한다.

    Args:
        timeframe: 캔들 주기 (예: '15m', '1h', '4h')

    Returns:
        캐시 유효 시간(초)
    """
    return _CACHE_TTL.get(timeframe, _DEFAULT_TTL)


class CandleFetcher:
    """캔들 데이터 수집 -- MarketDataClient 래퍼.

    타임프레임별 TTL 기반 인메모리 캐시를 관리하여
    불필요한 API 호출을 줄인다.
    """

    def __init__(self, client: MarketDataClient | None = None) -> None:
        """CandleFetcher를 초기화한다.

        client를 전달하지 않으면 첫 API 호출 시 MarketDataClient를
        자동 생성한다 (lazy DI).

        Args:
            client: 시세 데이터 클라이언트 (None이면 lazy 생성)
        """
        self._client_instance = client
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_ts: dict[str, float] = {}  # 캐시 저장 시각 (epoch)
        logger.info("CandleFetcher 초기화 완료 (client=%s)", "주입" if client else "lazy")

    @property
    def _client(self) -> MarketDataClient:
        """MarketDataClient를 반환한다. 없으면 lazy 생성한다.

        Returns:
            MarketDataClient 인스턴스
        """
        if self._client_instance is None:
            logger.info("MarketDataClient lazy 생성")
            self._client_instance = MarketDataClient()
        return self._client_instance

    def _cache_key(self, symbol: str, timeframe: str) -> str:
        """캐시 키를 생성한다.

        Args:
            symbol: 거래 심볼
            timeframe: 캔들 주기

        Returns:
            캐시 키 문자열
        """
        return f"{symbol}_{timeframe}"

    def _is_cache_valid(self, key: str, timeframe: str) -> bool:
        """캐시가 유효한지 확인한다.

        Args:
            key: 캐시 키
            timeframe: 캔들 주기 (TTL 결정용)

        Returns:
            캐시가 유효하면 True
        """
        if key not in self._cache or key not in self._cache_ts:
            return False
        elapsed = time.time() - self._cache_ts[key]
        ttl = _get_ttl(timeframe)
        return elapsed < ttl

    def get_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        """캔들 데이터를 반환한다. 캐시가 유효하면 캐시에서, 아니면 API를 호출한다.

        Args:
            symbol: 거래 심볼 (예: 'BTC/USDT:USDT')
            timeframe: 캔들 주기 (예: '15m', '1h', '4h')
            limit: 조회할 캔들 수 (기본 200)

        Returns:
            OHLCV DataFrame (columns=[open,high,low,close,volume], index=timestamp UTC)
        """
        key = self._cache_key(symbol, timeframe)

        if self._is_cache_valid(key, timeframe):
            logger.debug("캐시 히트: %s", key)
            return self._cache[key].copy()

        logger.debug("캐시 미스: %s — API 호출", key)
        df = self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        self._cache[key] = df
        self._cache_ts[key] = time.time()
        logger.info(
            "캔들 수집 완료: %s %s (%d개, TTL=%ds)",
            symbol, timeframe, len(df), _get_ttl(timeframe),
        )
        return df.copy()

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        limits: dict[str, int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """여러 타임프레임의 캔들 데이터를 일괄 조회한다.

        Args:
            symbol: 거래 심볼 (예: 'BTC/USDT:USDT')
            timeframes: 타임프레임 목록 (예: ["4h", "1h", "15m"])
            limits: 타임프레임별 캔들 수 (기본: {"4h": 100, "1h": 100, "15m": 200})

        Returns:
            타임프레임별 DataFrame dict

        Raises:
            RuntimeError: 하나라도 조회 실패 시
        """
        default_limits: dict[str, int] = {"4h": 100, "1h": 100, "15m": 200}
        effective_limits = {**default_limits, **(limits or {})}

        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            limit = effective_limits.get(tf, 200)
            try:
                df = self.get_candles(symbol, tf, limit=limit)
                result[tf] = df
                logger.debug("멀티 타임프레임 조회 성공: %s %s (%d건)", symbol, tf, len(df))
            except Exception as exc:
                logger.error(
                    "멀티 타임프레임 조회 실패: %s %s — %s: %s",
                    symbol, tf, type(exc).__name__, str(exc)[:200],
                )
                raise RuntimeError(
                    f"멀티 타임프레임 조회 실패: {symbol} {tf} — {exc}"
                ) from exc

        logger.info(
            "멀티 타임프레임 일괄 조회 완료: %s (%s)",
            symbol, ", ".join(timeframes),
        )
        return result

    def invalidate_cache(self, symbol: str | None = None) -> None:
        """캐시를 무효화한다.

        Args:
            symbol: 특정 심볼만 무효화 (None이면 전체 캐시 초기화)
        """
        if symbol is None:
            count = len(self._cache)
            self._cache.clear()
            self._cache_ts.clear()
            logger.info("전체 캐시 초기화 (%d건 삭제)", count)
        else:
            keys_to_remove = [
                k for k in self._cache if k.startswith(f"{symbol}_")
            ]
            for k in keys_to_remove:
                del self._cache[k]
                self._cache_ts.pop(k, None)
            logger.info(
                "'%s' 캐시 초기화 (%d건 삭제)", symbol, len(keys_to_remove)
            )
