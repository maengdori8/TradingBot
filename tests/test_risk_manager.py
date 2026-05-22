"""서킷브레이커 / 리스크 관리 테스트"""
import sys, os, tempfile
sys.path.insert(0, "/home/claude/trading-bot")

import pytest
from unittest.mock import patch
from src.risk.circuit_breaker import CircuitBreaker

# 테스트용 임시 DB
import src.risk.circuit_breaker as cb_module


@pytest.fixture
def tmp_cb(tmp_path):
    """임시 DB를 사용하는 CircuitBreaker"""
    db_path = tmp_path / "test_cb.db"
    with patch.object(cb_module, "DB_PATH", db_path):
        yield CircuitBreaker(
            trading_capital=1000,
            daily_loss_limit=0.03,
            weekly_loss_limit=0.08,
            max_consecutive_losses=3,
        )


def test_initial_state_allows_trading(tmp_cb):
    """초기 상태에서 거래 허용"""
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is True


def test_daily_loss_limit_blocks(tmp_cb):
    """일일 손실 한도 초과 시 차단"""
    tmp_cb.record_trade(-31)  # 1000 * 3% = 30 → 31 손실로 초과
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is False
    assert "일일" in reason


def test_consecutive_loss_blocks(tmp_cb):
    """3연패 시 차단"""
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is False
    assert "연속" in reason


def test_win_resets_consecutive(tmp_cb):
    """승리 시 연속 손실 초기화"""
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(10)   # 승리
    tmp_cb.record_trade(-5)  # 1연패 (3연패 아님)
    # 손실 합계 체크: -5-5+10-5 = -5 (일일 한도 미초과)
    allowed, reason = tmp_cb.is_trading_allowed()
    # 연속 손실은 1이므로 연속패 차단은 없어야 함
    assert "연속" not in reason or allowed is True


def test_daily_pnl_positive_allows(tmp_cb):
    """수익 상태에서는 거래 허용"""
    tmp_cb.record_trade(50)
    allowed, _ = tmp_cb.is_trading_allowed()
    assert allowed is True


def test_reset_consecutive_losses(tmp_cb):
    """수동 리셋 후 거래 허용"""
    import src.risk.circuit_breaker as mod
    from unittest.mock import patch
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    tmp_cb.reset_consecutive_losses()
    allowed, _ = tmp_cb.is_trading_allowed()
    # 일일 손실은 -15 (1000*3%=30 미만) → 허용
    assert allowed is True
