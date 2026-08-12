from __future__ import annotations

"""candle_fetcher.py (CandleFetcher) 테스트 — 캐시 로직 중심"""

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.candle_fetcher import CandleFetcher, _get_ttl, _DEFAULT_TTL


# ─── _get_ttl 유틸 ───────────────────────────────────────────────────

class TestGetTtl:
    """타임프레임별 TTL 매핑 테스트."""

    def test_known_timeframes(self):
        """등록된 타임프레임은 매핑 값을 반환."""
        assert _get_ttl("1m") == 60
        assert _get_ttl("15m") == 900
        assert _get_ttl("1h") == 3600
        assert _get_ttl("4h") == 14400

    def test_unknown_timeframe_returns_default(self):
        """등록 안 된 타임프레임은 기본 TTL 반환."""
        assert _get_ttl("3m") == _DEFAULT_TTL
        assert _get_ttl("12h") == _DEFAULT_TTL


# ─── CandleFetcher ───────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """mock MarketDataClient."""
    client = MagicMock()
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        },
        index=pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-01 00:15"], utc=True
        ),
    )
    df.index.name = "timestamp"
    client.fetch_ohlcv.return_value = df
    return client


class TestCandleFetcher:
    """CandleFetcher 캐시 동작 테스트."""

    def test_cache_miss_calls_client(self, mock_client):
        """캐시 미스 시 client.fetch_ohlcv 호출."""
        fetcher = CandleFetcher(mock_client)

        df = fetcher.get_candles("BTC/USDT:USDT", "15m", limit=200)

        mock_client.fetch_ohlcv.assert_called_once_with(
            "BTC/USDT:USDT", "15m", limit=200
        )
        assert len(df) == 2

    def test_cache_hit_no_client_call(self, mock_client):
        """캐시 히트 시 client.fetch_ohlcv 미호출."""
        fetcher = CandleFetcher(mock_client)

        fetcher.get_candles("BTC/USDT:USDT", "15m")
        mock_client.fetch_ohlcv.reset_mock()

        df2 = fetcher.get_candles("BTC/USDT:USDT", "15m")

        mock_client.fetch_ohlcv.assert_not_called()
        assert len(df2) == 2

    def test_cache_ttl_expired_re_fetches(self, mock_client):
        """캐시 TTL 만료 시 다시 API 호출."""
        fetcher = CandleFetcher(mock_client)

        fetcher.get_candles("BTC/USDT:USDT", "15m")
        mock_client.fetch_ohlcv.reset_mock()

        # TTL을 과거로 설정하여 만료 시뮬레이션
        key = fetcher._cache_key("BTC/USDT:USDT", "15m")
        fetcher._cache_ts[key] = time.time() - 10000  # TTL(900초) 초과

        fetcher.get_candles("BTC/USDT:USDT", "15m")
        mock_client.fetch_ohlcv.assert_called_once()

    def test_invalidate_cache_all(self, mock_client):
        """invalidate_cache(None)은 전체 캐시 초기화."""
        fetcher = CandleFetcher(mock_client)

        fetcher.get_candles("BTC/USDT:USDT", "15m")
        fetcher.get_candles("ETH/USDT:USDT", "1h")
        assert len(fetcher._cache) == 2

        fetcher.invalidate_cache()
        assert len(fetcher._cache) == 0
        assert len(fetcher._cache_ts) == 0

    def test_invalidate_cache_specific_symbol(self, mock_client):
        """invalidate_cache(symbol)은 해당 심볼만 제거."""
        fetcher = CandleFetcher(mock_client)

        fetcher.get_candles("BTC/USDT:USDT", "15m")
        fetcher.get_candles("ETH/USDT:USDT", "1h")
        assert len(fetcher._cache) == 2

        fetcher.invalidate_cache("BTC/USDT:USDT")
        assert len(fetcher._cache) == 1
        assert "ETH/USDT:USDT_1h" in fetcher._cache

    def test_returns_copy_not_reference(self, mock_client):
        """반환된 DataFrame이 캐시의 복사본인지 확인."""
        fetcher = CandleFetcher(mock_client)

        df1 = fetcher.get_candles("BTC/USDT:USDT", "15m")
        df2 = fetcher.get_candles("BTC/USDT:USDT", "15m")

        # 같은 데이터이지만 다른 객체
        assert df1 is not df2
        pd.testing.assert_frame_equal(df1, df2)

        # 하나를 수정해도 다른 것에 영향 없음
        df1.iloc[0, 0] = -999.0
        df3 = fetcher.get_candles("BTC/USDT:USDT", "15m")
        assert df3.iloc[0, 0] != -999.0

    def test_different_timeframes_separate_cache(self, mock_client):
        """같은 심볼이라도 타임프레임이 다르면 별도 캐시."""
        fetcher = CandleFetcher(mock_client)

        fetcher.get_candles("BTC/USDT:USDT", "15m")
        fetcher.get_candles("BTC/USDT:USDT", "1h")

        assert len(fetcher._cache) == 2
        assert mock_client.fetch_ohlcv.call_count == 2
