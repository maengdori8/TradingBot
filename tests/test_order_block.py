import pandas as pd
from strategy.order_block import OrderBlockDetector


def _make_df(ohlc_list):
    df = pd.DataFrame(ohlc_list, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="15min")
    df["volume"] = 100.0
    return df


class TestBullishOB:

    def test_bullish_ob_detected(self):
        df = _make_df([
            [105, 106, 100, 101],  # 강한 음봉 (OB 후보)
            [101, 108, 100, 107],  # 양봉이 음봉의 고가를 돌파
        ])
        detector = OrderBlockDetector({"lookback": 20, "min_body_ratio": 0.5})
        zones = detector.detect(df)
        bullish = [z for z in zones if z["type"] == "bullish_ob"]
        assert len(bullish) == 1
        assert bullish[0]["top"] == 106
        assert bullish[0]["bottom"] == 100


class TestBearishOB:

    def test_bearish_ob_detected(self):
        df = _make_df([
            [100, 106, 99, 105],   # 강한 양봉 (OB 후보)
            [105, 106, 97, 98],    # 음봉이 양봉의 저가를 하회
        ])
        detector = OrderBlockDetector({"lookback": 20, "min_body_ratio": 0.5})
        zones = detector.detect(df)
        bearish = [z for z in zones if z["type"] == "bearish_ob"]
        assert len(bearish) == 1
        assert bearish[0]["top"] == 106
        assert bearish[0]["bottom"] == 99


class TestNoOB:

    def test_no_ob_on_weak_candles(self):
        df = _make_df([
            [100, 100.5, 99.5, 100.1],  # body_ratio 낮음
            [100.1, 100.6, 99.6, 100.2],
            [100.2, 100.7, 99.7, 100.3],
        ])
        detector = OrderBlockDetector({"lookback": 20, "min_body_ratio": 0.5})
        zones = detector.detect(df)
        assert len(zones) == 0
