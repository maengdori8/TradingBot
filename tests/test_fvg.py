import pandas as pd
from strategy.fvg import FVGDetector


def _make_df(ohlc_list):
    df = pd.DataFrame(ohlc_list, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="15min")
    df["volume"] = 100.0
    return df


class TestBullishFVG:

    def test_bullish_fvg_detected(self):
        df = _make_df([
            [100, 102, 99, 101],
            [101, 103, 100, 102],
            [104, 106, 103, 105],
        ])
        detector = FVGDetector({"min_gap_pct": 0.001, "lookback": 50})
        zones = detector.detect(df)
        bullish = [z for z in zones if z["type"] == "bullish_fvg"]
        assert len(bullish) == 1
        assert bullish[0]["bottom"] == 102  # c1.high
        assert bullish[0]["top"] == 103     # c3.low

    def test_small_gap_filtered(self):
        df = _make_df([
            [100, 102, 99, 101],
            [101, 103, 100, 102],
            [102.05, 104, 102.01, 103],
        ])
        detector = FVGDetector({"min_gap_pct": 0.01, "lookback": 50})
        zones = detector.detect(df)
        bullish = [z for z in zones if z["type"] == "bullish_fvg"]
        assert len(bullish) == 0


class TestBearishFVG:

    def test_bearish_fvg_detected(self):
        df = _make_df([
            [105, 106, 103, 104],
            [104, 105, 102, 103],
            [101, 102, 99,  100],
        ])
        detector = FVGDetector({"min_gap_pct": 0.001, "lookback": 50})
        zones = detector.detect(df)
        bearish = [z for z in zones if z["type"] == "bearish_fvg"]
        assert len(bearish) == 1
        assert bearish[0]["top"] == 103    # c1.low
        assert bearish[0]["bottom"] == 102  # c3.high


class TestNoFVG:

    def test_flat_market(self):
        df = _make_df([
            [100, 100.1, 99.9, 100],
            [100, 100.1, 99.9, 100],
            [100, 100.1, 99.9, 100],
            [100, 100.1, 99.9, 100],
            [100, 100.1, 99.9, 100],
        ])
        detector = FVGDetector({"min_gap_pct": 0.001, "lookback": 50})
        zones = detector.detect(df)
        assert len(zones) == 0
