"""서킷브레이커 + RiskManager 통합 테스트"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import src.risk.circuit_breaker as cb_module
from src.risk.circuit_breaker import CircuitBreaker
from src.paper_trading import Position


# ── config mock ──────────────────────────────────────────────────────

MOCK_CONFIG = {
    "capital": {
        "total_capital": 5000,
        "trading_allocation": 0.25,
        "risk_per_trade": 0.01,
    },
    "exchange": {
        "leverage": 5,
        "symbols": ["BTC/USDT:USDT"],
    },
    "risk": {
        "min_rr_ratio": 2.0,
        "max_positions": 2,
        "daily_loss_limit": 0.03,
        "weekly_loss_limit": 0.08,
        "max_consecutive_losses": 3,
    },
}


def _make_position(symbol="BTC/USDT:USDT", direction="long", margin=100.0,
                   entry_price=50000, qty=0.01, sl=49000, tp=52000):
    return Position(
        id="test-pos",
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        qty=qty,
        stop_loss=sl,
        take_profit=tp,
        margin=margin,
        entry_time=datetime.now(timezone.utc),
    )


# ── CircuitBreaker 테스트 ────────────────────────────────────────────

@pytest.fixture
def tmp_cb(tmp_path):
    db_path = tmp_path / "test_cb.db"
    with patch.object(cb_module, "DB_PATH", db_path):
        yield CircuitBreaker(
            trading_capital=1000,
            daily_loss_limit=0.03,
            weekly_loss_limit=0.08,
            max_consecutive_losses=3,
        )


def test_initial_state_allows_trading(tmp_cb):
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is True


def test_daily_loss_limit_blocks(tmp_cb):
    tmp_cb.record_trade(-31)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is False
    assert "일일" in reason


def test_consecutive_loss_blocks(tmp_cb):
    for _ in range(3):
        tmp_cb.record_trade(-5)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is False
    assert "연속" in reason


def test_win_resets_consecutive(tmp_cb):
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(10)
    tmp_cb.record_trade(-5)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert "연속" not in reason or allowed is True


def test_daily_pnl_positive_allows(tmp_cb):
    tmp_cb.record_trade(50)
    allowed, _ = tmp_cb.is_trading_allowed()
    assert allowed is True


def test_reset_consecutive_losses(tmp_cb):
    for _ in range(3):
        tmp_cb.record_trade(-5)
    tmp_cb.reset_consecutive_losses()
    allowed, _ = tmp_cb.is_trading_allowed()
    assert allowed is True


# ── RiskManager 테스트 ───────────────────────────────────────────────

@pytest.fixture
def risk_manager(tmp_path):
    db = tmp_path / "rm_cb.db"
    with patch("src.risk.risk_manager.load_config", return_value=MOCK_CONFIG), \
         patch.object(cb_module, "DB_PATH", db):
        from src.risk.risk_manager import RiskManager
        yield RiskManager()


class TestRiskManagerInit:
    def test_capital_computed(self, risk_manager):
        assert risk_manager.total_capital == 5000
        assert risk_manager.trading_capital == 1250.0

    def test_leverage(self, risk_manager):
        assert risk_manager.leverage == 5

    def test_min_rr(self, risk_manager):
        assert risk_manager.min_rr == 2.0


class TestCheckTradeAllowed:
    def test_allowed_initially(self, risk_manager):
        allowed, reason = risk_manager.check_trade_allowed(current_positions=0)
        assert allowed is True

    def test_blocked_by_max_positions(self, risk_manager):
        allowed, reason = risk_manager.check_trade_allowed(current_positions=2)
        assert allowed is False
        assert "최대 포지션" in reason

    def test_duplicate_blocked(self, risk_manager):
        pos = _make_position()
        allowed, reason = risk_manager.check_trade_allowed(
            current_positions=1, positions=[pos],
            symbol="BTC/USDT:USDT", direction="long",
        )
        assert allowed is False
        assert "중복" in reason

    def test_different_direction_allowed(self, risk_manager):
        pos = _make_position(direction="long")
        allowed, _ = risk_manager.check_trade_allowed(
            current_positions=1, positions=[pos],
            symbol="BTC/USDT:USDT", direction="short",
        )
        assert allowed is True

    def test_circuit_breaker_integration(self, risk_manager):
        for _ in range(3):
            risk_manager.record_result(-100, "SL")
        allowed, reason = risk_manager.check_trade_allowed(current_positions=0)
        assert allowed is False


class TestCalculateTradeParams:
    def test_returns_valid_params(self, risk_manager):
        params = risk_manager.calculate_trade_params(entry=50000, stop_loss=49000)
        assert params["qty"] > 0
        assert params["take_profit"] > 50000
        assert params["entry"] == 50000
        assert params["stop_loss"] == 49000


class TestExposure:
    def test_empty(self, risk_manager):
        exp = risk_manager.calculate_total_exposure([])
        assert exp["total_margin"] == 0.0

    def test_long_exposure(self, risk_manager):
        pos = _make_position(margin=500, entry_price=50000, qty=0.01, sl=49000)
        exp = risk_manager.calculate_total_exposure([pos])
        assert exp["total_margin"] == 500.0
        assert exp["total_risk"] > 0
        assert "BTC/USDT:USDT" in exp["positions_by_symbol"]

    def test_short_exposure(self, risk_manager):
        pos = _make_position(direction="short", entry_price=3000, qty=0.1, sl=3100)
        exp = risk_manager.calculate_total_exposure([pos])
        assert exp["total_risk"] == pytest.approx(10.0, abs=0.01)


class TestRecordResult:
    def test_updates_cb(self, risk_manager):
        risk_manager.record_result(-50.0, "SL")
        assert risk_manager.cb.get_daily_pnl() < 0

    def test_callback_called(self, risk_manager):
        cb = MagicMock(__name__="test_callback")
        risk_manager.register_on_result(cb)
        risk_manager.record_result(100.0, "TP")
        cb.assert_called_once_with(100.0, "TP")

    def test_callback_error_safe(self, risk_manager):
        risk_manager.register_on_result(lambda p, r: 1 / 0)
        risk_manager.record_result(-10.0, "SL")
