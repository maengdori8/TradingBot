"""포지션 사이징 테스트"""
import sys
sys.path.insert(0, "/home/claude/trading-bot")

import pandas as pd
import numpy as np
import pytest
from src.risk.position_sizer import calculate_position_size, calculate_stop_loss_atr, calculate_take_profit


def test_basic_position_size():
    """기본 포지션 수량 계산"""
    # 자본 1000, 리스크 1%, 진입 100, SL 98 (2% 하락)
    # risk_amount = 10, price_risk = 2, qty = 10/2 = 5
    qty = calculate_position_size(1000, 0.01, 100, 98, leverage=10)
    assert qty == pytest.approx(5.0, rel=0.01)


def test_position_size_with_leverage_cap():
    """레버리지 한도 초과 방지"""
    # 자본 100, 레버리지 1, 진입 100 → max_qty = 100/100 = 1
    qty = calculate_position_size(100, 0.5, 100, 50, leverage=1)
    assert qty <= 1.0


def test_position_size_btc_example():
    """BTC 실거래 예시"""
    # 트레이딩 자본 1250, 리스크 1%, BTC 50000, SL 49500
    qty = calculate_position_size(1250, 0.01, 50000, 49500, leverage=5)
    expected = (1250 * 0.01) / 500  # = 0.025
    assert qty == pytest.approx(expected, rel=0.01)


def test_invalid_prices_raise():
    """잘못된 가격 입력 시 예외"""
    with pytest.raises(ValueError):
        calculate_position_size(1000, 0.01, 0, 100)
    with pytest.raises(ValueError):
        calculate_position_size(1000, 0.01, 100, 100)  # 동일 가격


def test_take_profit_long():
    """Long TP: entry + risk * rr"""
    tp = calculate_take_profit(100, 98, rr_ratio=2.0)
    assert tp == pytest.approx(104.0)


def test_take_profit_short():
    """Short TP: entry - risk * rr"""
    tp = calculate_take_profit(100, 102, rr_ratio=2.0)
    assert tp == pytest.approx(96.0)


def test_take_profit_rr3():
    """R:R 1:3 목표가 계산"""
    tp = calculate_take_profit(50000, 49000, rr_ratio=3.0)
    assert tp == pytest.approx(53000.0)


def test_atr_stop_loss():
    """ATR 기반 손절가 계산"""
    np.random.seed(42)
    closes = pd.Series(np.cumsum(np.random.randn(50)) + 100)
    highs = closes + 1
    lows = closes - 1
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    sl = calculate_stop_loss_atr(df, "long", 100, atr_period=14, atr_multiplier=1.5)
    assert sl < 100  # Long이면 SL은 진입가 아래
    sl_short = calculate_stop_loss_atr(df, "short", 100, atr_period=14, atr_multiplier=1.5)
    assert sl_short > 100  # Short이면 SL은 진입가 위
