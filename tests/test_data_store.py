"""data_store.py (DataStore) 테스트 — SQLite tmp_path 기반"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.data_store import DataStore


@pytest.fixture
def store(tmp_path):
    """임시 DB 파일을 사용하는 DataStore."""
    db_path = tmp_path / "test_market_data.db"
    ds = DataStore(db_path=db_path)
    yield ds
    ds.close()


@pytest.fixture
def sample_df():
    """테스트용 OHLCV DataFrame."""
    index = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=index,
    )


class TestDataStore:
    """DataStore CRUD 테스트."""

    def test_save_and_load_roundtrip(self, store, sample_df):
        """save_candles + load_candles 왕복 — 데이터 무결성."""
        store.save_candles("BTC/USDT:USDT", "15m", sample_df)
        loaded = store.load_candles("BTC/USDT:USDT", "15m")

        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
        # 값 비교
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True),
            sample_df.reset_index(drop=True),
            check_dtype=False,
        )

    def test_save_empty_dataframe_skips(self, store):
        """빈 DataFrame 저장 시 경고만 남기고 건너뜀."""
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        store.save_candles("BTC/USDT:USDT", "15m", empty_df)

        loaded = store.load_candles("BTC/USDT:USDT", "15m")
        assert loaded is None

    def test_load_nonexistent_returns_none(self, store):
        """저장된 데이터가 없으면 None 반환."""
        result = store.load_candles("DOGE/USDT:USDT", "1h")
        assert result is None

    def test_get_latest_timestamp(self, store, sample_df):
        """get_latest_timestamp가 가장 최근 시간을 정확히 반환."""
        store.save_candles("BTC/USDT:USDT", "15m", sample_df)
        latest = store.get_latest_timestamp("BTC/USDT:USDT", "15m")

        assert latest is not None
        expected = datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
        assert latest == expected

    def test_get_latest_timestamp_empty(self, store):
        """데이터가 없으면 get_latest_timestamp가 None 반환."""
        result = store.get_latest_timestamp("BTC/USDT:USDT", "15m")
        assert result is None

    def test_upsert_overwrites(self, store):
        """같은 timestamp 재저장 시 덮어쓰기 (UPSERT)."""
        index = pd.to_datetime(["2024-01-01 00:00"], utc=True)
        df1 = pd.DataFrame(
            {"open": [100.0], "high": [102.0], "low": [99.0],
             "close": [101.0], "volume": [1000.0]},
            index=index,
        )
        df2 = pd.DataFrame(
            {"open": [200.0], "high": [202.0], "low": [199.0],
             "close": [201.0], "volume": [2000.0]},
            index=index,
        )

        store.save_candles("BTC/USDT:USDT", "15m", df1)
        store.save_candles("BTC/USDT:USDT", "15m", df2)

        loaded = store.load_candles("BTC/USDT:USDT", "15m")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded.iloc[0]["open"] == 200.0
        assert loaded.iloc[0]["volume"] == 2000.0

    def test_different_symbols_isolated(self, store, sample_df):
        """다른 심볼의 데이터는 서로 독립."""
        store.save_candles("BTC/USDT:USDT", "15m", sample_df)
        store.save_candles("ETH/USDT:USDT", "15m", sample_df)

        btc = store.load_candles("BTC/USDT:USDT", "15m")
        eth = store.load_candles("ETH/USDT:USDT", "15m")

        assert btc is not None
        assert eth is not None
        assert len(btc) == 3
        assert len(eth) == 3

    def test_load_respects_limit(self, store):
        """load_candles limit 파라미터 동작."""
        index = pd.to_datetime(
            [f"2024-01-01 {h:02d}:00" for h in range(10)], utc=True
        )
        df = pd.DataFrame(
            {
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            },
            index=index,
        )
        df = df.astype(float)
        store.save_candles("BTC/USDT:USDT", "1h", df)

        loaded = store.load_candles("BTC/USDT:USDT", "1h", limit=3)
        assert loaded is not None
        assert len(loaded) == 3

    def test_loaded_data_sorted_by_timestamp(self, store, sample_df):
        """로드된 데이터가 timestamp 기준으로 정렬."""
        store.save_candles("BTC/USDT:USDT", "15m", sample_df)
        loaded = store.load_candles("BTC/USDT:USDT", "15m")

        assert loaded is not None
        timestamps = loaded.index.tolist()
        assert timestamps == sorted(timestamps)
