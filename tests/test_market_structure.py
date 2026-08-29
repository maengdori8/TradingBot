"""시장 구조 (BOS/CHoCH) 테스트"""
import sys
sys.path.insert(0, "/home/claude/trading-bot")
import pandas as pd
import numpy as np
from src.strategy.market_structure import (
    detect_swing_points,
    detect_bos,
    detect_choch,
    detect_htf_trend,
)


def wave_df(n=60, trend_dir=1):
    """명확한 사인파 + 추세 데이터 (로컬 극값이 뚜렷함)"""
    x = np.linspace(0, 6 * np.pi, n)
    base = np.linspace(100, 100 + trend_dir * 30, n)
    closes = base + np.sin(x) * 8
    highs  = closes + 1.5
    lows   = closes - 1.5
    opens  = closes - 0.3 * trend_dir
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000})


def test_swing_points_detected():
    """사인파 데이터에서 스윙 포인트 탐지"""
    df = wave_df(60)
    result = detect_swing_points(df, lookback=4)
    assert "swing_high" in result.columns
    assert "swing_low" in result.columns
    assert result["swing_high"].sum() >= 2, "스윙 고점 최소 2개 필요"
    assert result["swing_low"].sum() >= 2, "스윙 저점 최소 2개 필요"


def test_bos_bullish_on_uptrend():
    """상승 추세 + 마지막 캔들 강한 돌파 → Bullish BOS"""
    df = wave_df(60, trend_dir=1)
    sw = detect_swing_points(df, lookback=4)
    sh_vals = sw[sw["swing_high"]]["high"]
    if len(sh_vals) > 0:
        breakout = sh_vals.max() + 10
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("close")] = breakout
        df.iloc[-1, df.columns.get_loc("high")]  = breakout + 1
        result = detect_bos(df, lookback=4)
        assert result == "bullish"


def test_bos_bearish_on_downtrend():
    """하락 추세 + 마지막 캔들 강한 하향 돌파 → Bearish BOS"""
    df = wave_df(60, trend_dir=-1)
    sw = detect_swing_points(df, lookback=4)
    sl_vals = sw[sw["swing_low"]]["low"]
    if len(sl_vals) > 0:
        breakdown = sl_vals.min() - 10
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("close")] = breakdown
        df.iloc[-1, df.columns.get_loc("low")]   = breakdown - 1
        result = detect_bos(df, lookback=4)
        assert result == "bearish"


def test_bos_none_on_range():
    """횡보장에서는 BOS 없음"""
    x = np.linspace(0, 6 * np.pi, 50)
    closes = 100 + np.sin(x) * 5
    df = pd.DataFrame({
        "open": closes, "high": closes + 1,
        "low": closes - 1, "close": closes, "volume": 1000
    })
    result = detect_bos(df, lookback=4)
    assert result is None


def test_choch_returns_valid_or_none():
    """CHoCH 반환값 유효성 확인"""
    df = wave_df(60)
    result = detect_choch(df, lookback=4)
    assert result in ("bullish", "bearish", None)


class TestDetectHtfTrend:
    """BTC 상위TF 추세 레짐 판정 (연구 정의: EMA50 + 기울기)."""

    def test_uptrend_is_bull(self):
        import numpy as np
        df = pd.DataFrame({"close": np.linspace(100, 200, 120)})
        assert detect_htf_trend(df) == "bull"

    def test_downtrend_is_bear(self):
        import numpy as np
        df = pd.DataFrame({"close": np.linspace(200, 100, 120)})
        assert detect_htf_trend(df) == "bear"

    def test_flat_is_flat(self):
        import numpy as np
        df = pd.DataFrame({"close": np.full(120, 100.0)})
        assert detect_htf_trend(df) == "flat"

    def test_insufficient_data_is_flat(self):
        import numpy as np
        df = pd.DataFrame({"close": np.linspace(100, 200, 30)})
        assert detect_htf_trend(df) == "flat"
