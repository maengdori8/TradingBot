"""Kill Zone time filter tests."""
from datetime import datetime, timezone, timedelta

import pytest

from src.strategy.kill_zone import is_in_kill_zone, get_active_session


def _utc(hour: int, minute: int = 0, second: int = 0) -> datetime:
    """Helper: build a UTC datetime at a given hour."""
    return datetime(2025, 1, 15, hour, minute, second, tzinfo=timezone.utc)


# ── is_in_kill_zone ──────────────────────────────────────────────────


class TestIsInKillZone:
    """Verify London (02:00-05:00) and NY (07:00-10:00) detection."""

    def test_london_middle(self):
        assert is_in_kill_zone(_utc(3, 30)) is True

    def test_london_start_boundary(self):
        assert is_in_kill_zone(_utc(2, 0)) is True

    def test_london_end_boundary(self):
        assert is_in_kill_zone(_utc(5, 0)) is True

    def test_ny_middle(self):
        assert is_in_kill_zone(_utc(8, 30)) is True

    def test_ny_start_boundary(self):
        assert is_in_kill_zone(_utc(7, 0)) is True

    def test_ny_end_boundary(self):
        assert is_in_kill_zone(_utc(10, 0)) is True

    def test_outside_kill_zone_noon(self):
        assert is_in_kill_zone(_utc(12, 0)) is False

    def test_outside_kill_zone_midnight(self):
        assert is_in_kill_zone(_utc(0, 0)) is False

    def test_outside_between_sessions(self):
        """Between London end and NY start (05:01 ~ 06:59)."""
        assert is_in_kill_zone(_utc(6, 0)) is False

    def test_outside_after_ny(self):
        assert is_in_kill_zone(_utc(11, 0)) is False

    def test_just_before_london(self):
        assert is_in_kill_zone(_utc(1, 59)) is False

    def test_just_after_ny(self):
        """10:01 should be outside (seconds truncated)."""
        assert is_in_kill_zone(_utc(10, 1)) is False


# ── get_active_session ───────────────────────────────────────────────


class TestGetActiveSession:
    def test_london_session(self):
        assert get_active_session(_utc(3, 0)) == "london"

    def test_ny_session(self):
        assert get_active_session(_utc(9, 0)) == "newyork"

    def test_no_session_at_noon(self):
        assert get_active_session(_utc(12, 0)) is None

    def test_no_session_at_midnight(self):
        assert get_active_session(_utc(0, 0)) is None

    def test_london_start_returns_london(self):
        assert get_active_session(_utc(2, 0)) == "london"

    def test_london_end_returns_london(self):
        assert get_active_session(_utc(5, 0)) == "london"

    def test_ny_start_returns_newyork(self):
        assert get_active_session(_utc(7, 0)) == "newyork"

    def test_ny_end_returns_newyork(self):
        assert get_active_session(_utc(10, 0)) == "newyork"


# ── timezone-naive datetime handling ─────────────────────────────────


class TestTimezoneNaive:
    """timezone-naive datetimes are treated as UTC."""

    def test_naive_london(self):
        naive = datetime(2025, 1, 15, 3, 0, 0)  # no tzinfo
        assert is_in_kill_zone(naive) is True

    def test_naive_outside(self):
        naive = datetime(2025, 1, 15, 12, 0, 0)
        assert is_in_kill_zone(naive) is False

    def test_naive_session(self):
        naive = datetime(2025, 1, 15, 8, 0, 0)
        assert get_active_session(naive) == "newyork"


# ── timezone-aware non-UTC ───────────────────────────────────────────


class TestTimezoneAware:
    """Non-UTC timezones should be converted properly."""

    def test_utc_plus_5_in_london(self):
        """07:00 UTC+5 = 02:00 UTC => London start."""
        tz5 = timezone(timedelta(hours=5))
        dt = datetime(2025, 1, 15, 7, 0, 0, tzinfo=tz5)
        assert is_in_kill_zone(dt) is True
        assert get_active_session(dt) == "london"

    def test_utc_minus_5_in_ny(self):
        """04:00 UTC-5 = 09:00 UTC => NY."""
        tz_neg5 = timezone(timedelta(hours=-5))
        dt = datetime(2025, 1, 15, 4, 0, 0, tzinfo=tz_neg5)
        assert is_in_kill_zone(dt) is True
        assert get_active_session(dt) == "newyork"
