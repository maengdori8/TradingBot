from __future__ import annotations

"""
tests/conftest.py — Shared fixtures for the entire test suite.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv():
    """Standard OHLCV test data: 100 candles with an uptrend."""
    np.random.seed(42)
    n = 100
    # Uptrend base with sine wave for swings
    x = np.linspace(0, 8 * np.pi, n)
    base = np.linspace(100, 140, n)
    noise = np.random.randn(n) * 0.5
    closes = base + np.sin(x) * 5 + noise
    highs = closes + np.abs(np.random.randn(n)) * 1.5 + 0.5
    lows = closes - np.abs(np.random.randn(n)) * 1.5 - 0.5
    opens = closes + np.random.randn(n) * 0.3

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(500, 5000, size=n).astype(float),
    })
    return df


@pytest.fixture
def sample_ohlcv_downtrend():
    """Downtrend OHLCV test data: 100 candles."""
    np.random.seed(123)
    n = 100
    x = np.linspace(0, 8 * np.pi, n)
    base = np.linspace(140, 100, n)  # Descending
    noise = np.random.randn(n) * 0.5
    closes = base + np.sin(x) * 5 + noise
    highs = closes + np.abs(np.random.randn(n)) * 1.5 + 0.5
    lows = closes - np.abs(np.random.randn(n)) * 1.5 - 0.5
    opens = closes + np.random.randn(n) * 0.3

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(500, 5000, size=n).astype(float),
    })
    return df


@pytest.fixture
def make_ohlcv_df():
    """Factory fixture: pass a list of [o,h,l,c,v] rows to build a DataFrame."""
    def _make(data):
        return pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])
    return _make
