"""Circuit Breaker tests — daily/weekly loss limits and consecutive-loss management."""
import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

import src.risk.circuit_breaker as cb_module
from src.risk.circuit_breaker import CircuitBreaker


@pytest.fixture
def cb(tmp_path):
    """CircuitBreaker backed by a temporary SQLite DB."""
    db_path = tmp_path / "test_cb.db"
    with patch.object(cb_module, "DB_PATH", db_path):
        yield CircuitBreaker(
            trading_capital=10_000,
            daily_loss_limit=0.03,     # 300 USDT
            weekly_loss_limit=0.08,    # 800 USDT
            max_consecutive_losses=3,
        )


# ── Initial state ────────────────────────────────────────────────────


class TestInitialState:
    def test_trading_allowed_initially(self, cb):
        allowed, reason = cb.is_trading_allowed()
        assert allowed is True
        assert reason == "거래 가능"

    def test_daily_pnl_zero_initially(self, cb):
        assert cb.get_daily_pnl() == 0.0


# ── Daily loss limit ─────────────────────────────────────────────────


class TestDailyLossLimit:
    def test_within_limit_allows_trading(self, cb):
        cb.record_trade(-100)  # 100 < 300 limit
        allowed, _ = cb.is_trading_allowed()
        assert allowed is True

    def test_exceeds_limit_blocks_trading(self, cb):
        cb.record_trade(-301)  # > 300 limit
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False
        assert "일일" in reason

    def test_multiple_small_losses_exceed_limit(self, cb):
        for _ in range(4):
            cb.record_trade(-80)  # total = -320 > 300
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False
        assert "일일" in reason or "연속" in reason

    def test_gains_offset_losses(self, cb):
        cb.record_trade(-200)
        cb.record_trade(150)  # net = -50, well under 300
        allowed, _ = cb.is_trading_allowed()
        assert allowed is True


# ── Weekly loss limit ────────────────────────────────────────────────


class TestWeeklyLossLimit:
    def test_exceeds_weekly_limit(self, cb):
        """A single large loss exceeding weekly limit (800)."""
        cb.record_trade(-801)
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False
        # Could be daily or weekly — both are exceeded
        assert "손실" in reason

    def test_accumulates_over_week(self, cb):
        """Multiple losses that stay under daily but exceed weekly."""
        # 3 losses of 280 each: daily never exceeds 300,
        # but we record them to simulate same-day accumulation.
        # In this simplified test (same datetime), daily = -840 > 300.
        # So daily will trigger first. This is expected behavior.
        for _ in range(3):
            cb.record_trade(-280)
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False


# ── Consecutive losses ───────────────────────────────────────────────


class TestConsecutiveLosses:
    def test_three_consecutive_losses_block(self, cb):
        """max_consecutive_losses=3 triggers after 3 losses."""
        cb.record_trade(-10)
        cb.record_trade(-10)
        cb.record_trade(-10)
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False
        assert "연속" in reason

    def test_two_losses_still_allowed(self, cb):
        cb.record_trade(-10)
        cb.record_trade(-10)
        allowed, _ = cb.is_trading_allowed()
        assert allowed is True

    def test_win_resets_consecutive_count(self, cb):
        cb.record_trade(-10)
        cb.record_trade(-10)
        cb.record_trade(50)   # resets streak
        cb.record_trade(-10)  # only 1 consecutive loss now
        allowed, reason = cb.is_trading_allowed()
        # net PnL = -10-10+50-10 = 20 (positive) => daily OK
        # consecutive = 1 => OK
        assert allowed is True

    def test_four_consecutive_losses_still_blocked(self, cb):
        for _ in range(4):
            cb.record_trade(-5)
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False
        assert "연속" in reason


# ── Rest period after consecutive losses ─────────────────────────────


class TestRestPeriod:
    def test_rest_period_set_on_block(self, cb):
        """When consecutive losses trigger, a rest_until is set."""
        for _ in range(3):
            cb.record_trade(-5)
        allowed, reason = cb.is_trading_allowed()
        assert allowed is False
        # First call sets rest_until
        # Second call checks rest_until is in the future
        allowed2, reason2 = cb.is_trading_allowed()
        assert allowed2 is False
        assert "휴식" in reason2


# ── record_trade state changes ───────────────────────────────────────


class TestRecordTrade:
    def test_record_trade_updates_daily_pnl(self, cb):
        cb.record_trade(-50)
        assert cb.get_daily_pnl() == pytest.approx(-50.0)

    def test_multiple_trades_accumulate(self, cb):
        cb.record_trade(-30)
        cb.record_trade(20)
        cb.record_trade(-10)
        assert cb.get_daily_pnl() == pytest.approx(-20.0)

    def test_positive_trade_increases_pnl(self, cb):
        cb.record_trade(100)
        assert cb.get_daily_pnl() == pytest.approx(100.0)


# ── Manual reset ─────────────────────────────────────────────────────


class TestManualReset:
    def test_reset_consecutive_losses_re_enables_trading(self, cb):
        for _ in range(3):
            cb.record_trade(-5)
        allowed, _ = cb.is_trading_allowed()
        assert allowed is False

        cb.reset_consecutive_losses()
        # daily pnl = -15, limit = 300 => OK
        allowed, _ = cb.is_trading_allowed()
        assert allowed is True

    def test_reset_clears_rest_until(self, cb):
        for _ in range(3):
            cb.record_trade(-5)
        cb.is_trading_allowed()  # sets rest_until
        cb.reset_consecutive_losses()
        allowed, reason = cb.is_trading_allowed()
        assert allowed is True
        assert "연속" not in reason
