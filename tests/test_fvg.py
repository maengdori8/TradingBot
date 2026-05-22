"""FVG 탐지 테스트"""
import sys
sys.path.insert(0, "/home/claude/trading-bot")

import pandas as pd
import numpy as np
import pytest
from src.strategy.fvg_detector import detect_fvg, is_price_in_fvg, update_fvg_fills


def make_df(data):
    return pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"])


def test_bullish_fvg_detected():
    """Bullish FVG: candle[0].high < candle[2].low"""
    df = make_df([
        [100, 105, 98,  103, 1000],  # 캔들 0
        [103, 108, 101, 106, 1000],  # 캔들 1 (중간)
        [108, 115, 107, 113, 1000],  # 캔들 2: low(107) > candle[0].high(105) → Bullish FVG
    ])
    fvgs = detect_fvg(df, min_gap_pct=0.0)
    bullish = [f for f in fvgs if f.type == "bullish"]
    assert len(bullish) >= 1
    assert bullish[0].bottom == pytest.approx(105)
    assert bullish[0].top == pytest.approx(107)


def test_bearish_fvg_detected():
    """Bearish FVG: candle[0].low > candle[2].high"""
    df = make_df([
        [110, 112, 108, 109, 1000],  # 캔들 0: low=108
        [109, 110, 105, 107, 1000],  # 캔들 1
        [107, 107, 100, 101, 1000],  # 캔들 2: high=107 < candle[0].low=108 → Bearish FVG
    ])
    fvgs = detect_fvg(df, min_gap_pct=0.0)
    bearish = [f for f in fvgs if f.type == "bearish"]
    assert len(bearish) >= 1
    assert bearish[0].top == pytest.approx(108)
    assert bearish[0].bottom == pytest.approx(107)


def test_no_fvg_flat_market():
    """횡보장에서는 FVG 없음"""
    df = make_df([
        [100, 101, 99, 100, 500],
        [100, 101, 99, 100, 500],
        [100, 101, 99, 100, 500],
        [100, 101, 99, 100, 500],
    ])
    fvgs = detect_fvg(df, min_gap_pct=0.001)
    assert len(fvgs) == 0


def test_is_price_in_fvg():
    """가격이 FVG 내부에 있는지 확인"""
    df = make_df([
        [100, 105, 98,  103, 1000],
        [103, 108, 101, 106, 1000],
        [108, 115, 107, 113, 1000],
    ])
    fvgs = detect_fvg(df, min_gap_pct=0.0)
    bullish = [f for f in fvgs if f.type == "bullish"]
    assert len(is_price_in_fvg(106.0, bullish)) >= 1
    assert len(is_price_in_fvg(120.0, bullish)) == 0


def test_fvg_fill_removes_entry():
    """FVG가 채워지면 리스트에서 제거"""
    df = make_df([
        [100, 105, 98,  103, 1000],
        [103, 108, 101, 106, 1000],
        [108, 115, 107, 113, 1000],
    ])
    fvgs = detect_fvg(df, min_gap_pct=0.0)
    bullish = [f for f in fvgs if f.type == "bullish"]
    assert len(bullish) >= 1
    # 하락 캔들로 FVG 채우기
    updated = update_fvg_fills(bullish, current_high=113, current_low=104)
    assert all(f.filled or f not in updated for f in bullish)
