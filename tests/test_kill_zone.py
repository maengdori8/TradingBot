from datetime import datetime, timezone
from strategy.kill_zone import KillZoneFilter


KZ_CONFIG = {
    "london_open": "08:00",
    "london_close": "12:00",
    "ny_open": "13:00",
    "ny_close": "17:00",
    "asian_open": "00:00",
    "asian_close": "04:00",
}


class TestKillZone:

    def test_london_session_active(self):
        kz = KillZoneFilter(KZ_CONFIG)
        t = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        result = kz.is_active(t)
        assert result["london"] is True
        assert result["any"] is True

    def test_ny_session_active(self):
        kz = KillZoneFilter(KZ_CONFIG)
        t = datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc)
        result = kz.is_active(t)
        assert result["new_york"] is True
        assert result["any"] is True

    def test_asian_session_active(self):
        kz = KillZoneFilter(KZ_CONFIG)
        t = datetime(2024, 1, 15, 2, 0, tzinfo=timezone.utc)
        result = kz.is_active(t)
        assert result["asian"] is True
        assert result["any"] is True

    def test_no_session_active(self):
        kz = KillZoneFilter(KZ_CONFIG)
        t = datetime(2024, 1, 15, 6, 0, tzinfo=timezone.utc)
        result = kz.is_active(t)
        assert result["london"] is False
        assert result["new_york"] is False
        assert result["asian"] is False
        assert result["any"] is False

    def test_session_boundary(self):
        kz = KillZoneFilter(KZ_CONFIG)
        t = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        result = kz.is_active(t)
        assert result["london"] is True
