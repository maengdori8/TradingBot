from unittest.mock import MagicMock
from risk.manager import RiskManager


def _make_risk_manager(balance=10000.0, open_positions=0):
    client = MagicMock()
    client.get_balance.return_value = balance
    client.get_positions.return_value = [
        {"size": "1.0"} for _ in range(open_positions)
    ]
    client.get_closed_pnl.return_value = []

    rm = RiskManager(
        config={
            "max_risk_per_trade": 0.01,
            "max_daily_loss": 0.03,
            "max_open_positions": 3,
            "risk_reward_ratio": 2.0,
        },
        client=client,
        symbol="BTCUSDT",
    )
    rm.load_instrument_info({
        "min_qty": 0.001,
        "max_qty": 100.0,
        "qty_step": 0.001,
        "tick_size": 0.10,
        "min_price": 0.10,
        "max_price": 999999.0,
    })
    return rm


class TestCanOpenTrade:

    def test_allows_when_under_limits(self):
        rm = _make_risk_manager(balance=10000, open_positions=0)
        assert rm.can_open_trade({"side": "Buy"}) is True

    def test_blocks_at_max_positions(self):
        rm = _make_risk_manager(balance=10000, open_positions=3)
        assert rm.can_open_trade({"side": "Buy"}) is False

    def test_blocks_at_daily_loss_limit(self):
        rm = _make_risk_manager(balance=10000, open_positions=0)
        rm.daily_pnl = -300.0
        assert rm.can_open_trade({"side": "Buy"}) is False

    def test_blocks_zero_balance(self):
        rm = _make_risk_manager(balance=0, open_positions=0)
        assert rm.can_open_trade({"side": "Buy"}) is False


class TestSizePosition:

    def test_buy_sizing(self):
        rm = _make_risk_manager(balance=10000)
        signal = {"side": "Buy", "entry": 50000.0, "sl": 49500.0, "reason": "test"}
        order = rm.size_position(signal)
        assert order is not None
        assert order["side"] == "Buy"
        assert order["qty"] == 0.2  # 100 / 500 = 0.2
        assert order["tp"] == 51000.0  # entry + 500 * 2
        assert order["sl"] == 49500.0

    def test_sell_sizing(self):
        rm = _make_risk_manager(balance=10000)
        signal = {"side": "Sell", "entry": 50000.0, "sl": 50500.0, "reason": "test"}
        order = rm.size_position(signal)
        assert order is not None
        assert order["side"] == "Sell"
        assert order["tp"] == 49000.0  # entry - 500 * 2

    def test_zero_distance_returns_none(self):
        rm = _make_risk_manager(balance=10000)
        signal = {"side": "Buy", "entry": 50000.0, "sl": 50000.0, "reason": "test"}
        order = rm.size_position(signal)
        assert order is None

    def test_qty_respects_step(self):
        rm = _make_risk_manager(balance=10000)
        signal = {"side": "Buy", "entry": 50000.0, "sl": 49000.0, "reason": "test"}
        order = rm.size_position(signal)
        qty_str = f"{order['qty']:.3f}"
        assert float(qty_str) == order["qty"]
