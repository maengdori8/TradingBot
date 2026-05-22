import pandas as pd
from strategy.structure import StructureAnalyzer


def _make_uptrend_df():
    # HH/HL 상승 구조: 뚜렷한 스윙 고점/저점
    closes = [
        100, 101, 105, 103, 102,  # 첫 번째 고점 ~105, 저점 ~102
        103, 104, 108, 106, 105,  # 두 번째 고점 ~108 (HH), 저점 ~105 (HL)
        106, 107, 112, 110, 108,  # 세 번째 고점 ~112 (HH), 저점 ~108 (HL)
        109, 110, 115, 113, 112,  # 네 번째 고점 ~115
    ]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="1h"),
        "open": [c - 0.5 for c in closes],
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    })
    return df


def _make_downtrend_df():
    # LH/LL 하락 구조
    closes = [
        120, 119, 115, 117, 118,
        117, 116, 112, 114, 115,
        114, 113, 108, 110, 112,
        111, 110, 105, 107, 108,
    ]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="1h"),
        "open": [c + 0.5 for c in closes],
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    })
    return df


class TestStructure:

    def test_bullish_trend(self):
        df = _make_uptrend_df()
        analyzer = StructureAnalyzer({"lookback": 50, "swing_strength": 2})
        result = analyzer.analyze(df)
        assert result["trend"] in ("bullish", "neutral")
        assert isinstance(result["swings"], list)

    def test_bearish_trend(self):
        df = _make_downtrend_df()
        analyzer = StructureAnalyzer({"lookback": 50, "swing_strength": 2})
        result = analyzer.analyze(df)
        assert result["trend"] in ("bearish", "neutral")

    def test_swing_points_found(self):
        df = _make_uptrend_df()
        analyzer = StructureAnalyzer({"lookback": 50, "swing_strength": 2})
        result = analyzer.analyze(df)
        assert len(result["swings"]) > 0

    def test_flat_market_neutral(self):
        prices = [100.0] * 20
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1h"),
            "open": prices,
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
            "volume": [100] * 20,
        })
        analyzer = StructureAnalyzer({"lookback": 50, "swing_strength": 2})
        result = analyzer.analyze(df)
        assert result["trend"] == "neutral"
