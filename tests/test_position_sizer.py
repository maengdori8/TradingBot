"""포지션 사이징 테스트"""
import sys
sys.path.insert(0, "/home/claude/trading-bot")

import pandas as pd
import numpy as np
import pytest
from src.risk.position_sizer import (
    calculate_position_size,
    calculate_stop_loss_atr,
    calculate_take_profit,
    calculate_auto_leverage,
)


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


class TestAutoLeverage:
    """손절 거리 기반 자동 레버리지 테스트."""

    def test_tight_stop_higher_leverage(self):
        """타이트한 손절(1%)은 넓은 손절(5%)보다 높은 레버리지."""
        lev_tight = calculate_auto_leverage(100, 99, max_leverage=50)   # 1%
        lev_wide = calculate_auto_leverage(100, 95, max_leverage=50)    # 5%
        assert lev_tight > lev_wide

    def test_formula(self):
        """레버리지 = 1 / (liq_buffer * sl_pct), 내림."""
        # SL 2%, buffer 2.0 → 1/(2*0.02)=25
        lev = calculate_auto_leverage(100, 98, max_leverage=50, liq_buffer=2.0)
        assert lev == 25

    def test_max_cap(self):
        """상한 초과 시 max_leverage로 제한."""
        # SL 0.5%, buffer 2.0 → raw=100 → max 10으로 제한
        lev = calculate_auto_leverage(100, 99.5, max_leverage=10, liq_buffer=2.0)
        assert lev == 10

    def test_min_floor(self):
        """매우 넓은 손절은 min_leverage로 제한."""
        # SL 50%, buffer 2.0 → raw=1.0
        lev = calculate_auto_leverage(100, 50, max_leverage=10, min_leverage=1)
        assert lev == 1

    def test_liq_always_beyond_sl(self):
        """청산 거리(1/lev)가 항상 손절 거리보다 멀다 (buffer>=1)."""
        entry, sl = 100.0, 97.0   # 3%
        lev = calculate_auto_leverage(entry, sl, max_leverage=50, liq_buffer=2.0)
        sl_pct = abs(entry - sl) / entry
        liq_pct = 1.0 / lev       # 대략적 청산 거리
        assert liq_pct > sl_pct

    def test_short_direction(self):
        """숏(SL이 진입가 위)도 동일하게 동작."""
        lev = calculate_auto_leverage(100, 102, max_leverage=50, liq_buffer=2.0)
        assert lev == 25   # 2% 손절

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError):
            calculate_auto_leverage(0, 100)


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
