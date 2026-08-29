"""Order Block detection tests."""
import pandas as pd
import numpy as np

from src.strategy.order_block import detect_order_blocks, is_price_in_ob, OrderBlock


def _make_df(data):
    return pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])


# ── Bullish OB detection ─────────────────────────────────────────────


class TestBullishOB:
    """Bullish OB: bearish candle (close < open) followed by strong upward move."""

    def test_bullish_ob_detected(self):
        """Bearish candle at index 1, then 3 candles rising sharply."""
        df = _make_df([
            [100, 102, 99, 101, 1000],    # candle 0
            [105, 106, 100, 101, 1000],    # candle 1: bearish (open=105 > close=101)
            [101, 108, 100, 107, 1000],    # candle 2: strong up
            [107, 115, 106, 114, 1000],    # candle 3: strong up
            [114, 120, 113, 119, 1000],    # candle 4: strong up
        ])
        obs = detect_order_blocks(df, threshold=0.003, lookback=3)
        bullish = [ob for ob in obs if ob.type == "bullish"]
        assert len(bullish) >= 1
        # Bullish OB: top = open of the bearish candle, bottom = close
        ob = bullish[0]
        assert ob.top > ob.bottom

    def test_bullish_ob_strength_above_threshold(self):
        """Strength must meet threshold."""
        df = _make_df([
            [100, 102, 99, 101, 1000],
            [105, 106, 100, 101, 1000],    # bearish candle
            [101, 108, 100, 107, 1000],
            [107, 115, 106, 114, 1000],
            [114, 120, 113, 119, 1000],
        ])
        obs = detect_order_blocks(df, threshold=0.003, lookback=3)
        for ob in obs:
            assert ob.strength >= 0.003


class TestBearishOB:
    """Bearish OB: bullish candle (close > open) followed by strong downward move."""

    def test_bearish_ob_detected(self):
        df = _make_df([
            [120, 122, 118, 119, 1000],    # candle 0
            [115, 120, 114, 119, 1000],    # candle 1: bullish (close=119 > open=115)
            [119, 120, 110, 111, 1000],    # candle 2: strong drop
            [111, 112, 102, 103, 1000],    # candle 3: strong drop
            [103, 104, 95,  96,  1000],    # candle 4: strong drop
        ])
        obs = detect_order_blocks(df, threshold=0.003, lookback=3)
        bearish = [ob for ob in obs if ob.type == "bearish"]
        assert len(bearish) >= 1
        ob = bearish[0]
        assert ob.top > ob.bottom


# ── Threshold filtering ──────────────────────────────────────────────


class TestThresholdFiltering:
    def test_no_ob_when_move_too_small(self):
        """Flat/tiny move should not produce OBs with a large threshold."""
        df = _make_df([
            [100, 101, 99, 100, 500],
            [100, 101, 99,  99, 500],  # tiny bearish
            [99,  101, 99, 100, 500],
            [100, 101, 99, 100, 500],
            [100, 101, 99, 100, 500],
        ])
        obs = detect_order_blocks(df, threshold=0.05, lookback=3)
        assert len(obs) == 0

    def test_lower_threshold_finds_more(self):
        """Lower threshold should detect more (or equal) OBs."""
        np.random.seed(99)
        n = 30
        closes = np.cumsum(np.random.randn(n)) + 100
        highs = closes + 1
        lows = closes - 1
        opens = closes + np.random.randn(n) * 0.5
        df = pd.DataFrame({
            "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": 1000
        })
        obs_strict = detect_order_blocks(df, threshold=0.05, lookback=3)
        obs_loose = detect_order_blocks(df, threshold=0.001, lookback=3)
        assert len(obs_loose) >= len(obs_strict)


# ── is_price_in_ob ───────────────────────────────────────────────────


class TestIsPriceInOB:
    def _sample_obs(self):
        return [
            OrderBlock(type="bullish", top=105, bottom=101, candle_index=1, strength=0.01),
            OrderBlock(type="bearish", top=120, bottom=115, candle_index=3, strength=0.01),
        ]

    def test_price_inside_bullish_ob(self):
        result = is_price_in_ob(103, self._sample_obs())
        assert len(result) == 1
        assert result[0].type == "bullish"

    def test_price_inside_bearish_ob(self):
        result = is_price_in_ob(117, self._sample_obs())
        assert len(result) == 1
        assert result[0].type == "bearish"

    def test_price_outside_all_obs(self):
        result = is_price_in_ob(110, self._sample_obs())
        assert len(result) == 0

    def test_price_at_boundary(self):
        result = is_price_in_ob(105, self._sample_obs())
        assert len(result) == 1

    def test_empty_ob_list(self):
        result = is_price_in_ob(100, [])
        assert len(result) == 0


# ── Empty / edge data ────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        obs = detect_order_blocks(df)
        assert obs == []

    def test_single_row(self):
        df = _make_df([[100, 102, 98, 101, 500]])
        obs = detect_order_blocks(df)
        assert obs == []

    def test_two_rows(self):
        df = _make_df([
            [100, 102, 98, 101, 500],
            [101, 103, 99, 102, 500],
        ])
        obs = detect_order_blocks(df)
        assert obs == []

    def test_minimum_viable_length(self):
        """Need at least 1 + lookback rows. With lookback=3, need 5 rows."""
        df = _make_df([
            [100, 102, 99, 101, 500],
            [105, 106, 100, 101, 500],
            [101, 108, 100, 107, 500],
            [107, 115, 106, 114, 500],
            [114, 120, 113, 119, 500],
        ])
        obs = detect_order_blocks(df, threshold=0.003, lookback=3)
        # Should run without error
        assert isinstance(obs, list)
