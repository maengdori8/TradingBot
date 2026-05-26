"""Optimal Trade Entry (OTE) zone tests."""
import pytest

from src.strategy.ote import calculate_ote_zone, is_price_in_ote, OTEZone


# ── Bullish OTE ──────────────────────────────────────────────────────


class TestBullishOTE:
    """Bullish OTE: retracement from swing_high down toward swing_low."""

    def test_basic_calculation(self):
        zone = calculate_ote_zone(swing_high=200, swing_low=100, direction="bullish")
        rng = 100  # 200 - 100
        expected_ote_low = 200 - rng * 0.786   # 121.4
        expected_ote_high = 200 - rng * 0.618  # 138.2
        assert zone.ote_low == pytest.approx(expected_ote_low)
        assert zone.ote_high == pytest.approx(expected_ote_high)

    def test_direction_is_bullish(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert zone.direction == "bullish"

    def test_ote_low_less_than_ote_high(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert zone.ote_low < zone.ote_high

    def test_ote_zone_within_swing_range(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert zone.ote_low >= 100
        assert zone.ote_high <= 200

    def test_preserves_swing_values(self):
        zone = calculate_ote_zone(500, 400, "bullish")
        assert zone.high == 500
        assert zone.low == 400


# ── Bearish OTE ──────────────────────────────────────────────────────


class TestBearishOTE:
    """Bearish OTE: retracement from swing_low up toward swing_high."""

    def test_basic_calculation(self):
        zone = calculate_ote_zone(swing_high=200, swing_low=100, direction="bearish")
        rng = 100
        expected_ote_low = 100 + rng * 0.618   # 161.8
        expected_ote_high = 100 + rng * 0.786  # 178.6
        assert zone.ote_low == pytest.approx(expected_ote_low)
        assert zone.ote_high == pytest.approx(expected_ote_high)

    def test_direction_is_bearish(self):
        zone = calculate_ote_zone(200, 100, "bearish")
        assert zone.direction == "bearish"

    def test_ote_low_less_than_ote_high(self):
        zone = calculate_ote_zone(200, 100, "bearish")
        assert zone.ote_low < zone.ote_high

    def test_ote_zone_within_swing_range(self):
        zone = calculate_ote_zone(200, 100, "bearish")
        assert zone.ote_low >= 100
        assert zone.ote_high <= 200


# ── is_price_in_ote ──────────────────────────────────────────────────


class TestIsPriceInOTE:
    def test_price_inside_bullish_ote(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        mid = (zone.ote_low + zone.ote_high) / 2
        assert is_price_in_ote(mid, zone) is True

    def test_price_at_ote_low_boundary(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert is_price_in_ote(zone.ote_low, zone) is True

    def test_price_at_ote_high_boundary(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert is_price_in_ote(zone.ote_high, zone) is True

    def test_price_below_ote(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert is_price_in_ote(zone.ote_low - 1, zone) is False

    def test_price_above_ote(self):
        zone = calculate_ote_zone(200, 100, "bullish")
        assert is_price_in_ote(zone.ote_high + 1, zone) is False

    def test_price_inside_bearish_ote(self):
        zone = calculate_ote_zone(200, 100, "bearish")
        mid = (zone.ote_low + zone.ote_high) / 2
        assert is_price_in_ote(mid, zone) is True

    def test_price_outside_bearish_ote(self):
        zone = calculate_ote_zone(200, 100, "bearish")
        assert is_price_in_ote(50, zone) is False


# ── Edge cases ───────────────────────────────────────────────────────


class TestOTEEdgeCases:
    def test_equal_swing_high_low(self):
        """swing_high == swing_low => range is 0 => ote_low == ote_high == swing."""
        zone = calculate_ote_zone(100, 100, "bullish")
        assert zone.ote_low == pytest.approx(100)
        assert zone.ote_high == pytest.approx(100)
        assert is_price_in_ote(100, zone) is True

    def test_very_small_range(self):
        zone = calculate_ote_zone(100.01, 100.00, "bullish")
        assert zone.ote_low < zone.ote_high

    def test_large_range(self):
        zone = calculate_ote_zone(50000, 40000, "bullish")
        rng = 10000
        assert zone.ote_low == pytest.approx(50000 - rng * 0.786)
        assert zone.ote_high == pytest.approx(50000 - rng * 0.618)
