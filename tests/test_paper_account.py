from core.paper.account import PaperAccount


class TestPaperAccount:

    def test_initial_state(self):
        acc = PaperAccount(initial_balance=10000.0)
        assert acc.get_balance() == 10000.0
        assert acc.get_equity() == 10000.0
        assert len(acc.positions) == 0

    def test_open_position(self):
        acc = PaperAccount(initial_balance=10000.0, leverage=10)
        order = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "entry": 50000.0,
            "sl": 49500.0,
            "tp": 51000.0,
            "qty": 0.1,
            "reason": "test",
        }
        result = acc.open_position(order)
        assert result is True
        assert len(acc.positions) == 1
        assert acc.positions[0].entry_price == 50000.0

    def test_margin_check(self):
        acc = PaperAccount(initial_balance=100.0, leverage=10)
        order = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "entry": 50000.0,
            "sl": 49500.0,
            "tp": 51000.0,
            "qty": 1.0,  # 50000 / 10 = 5000 > 100
            "reason": "test",
        }
        result = acc.open_position(order)
        assert result is False
        assert len(acc.positions) == 0

    def test_tp_exit(self):
        acc = PaperAccount(initial_balance=10000.0)
        order = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "entry": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0,
            "qty": 0.1,
            "reason": "test",
        }
        acc.open_position(order)
        closed = acc.check_exits("BTCUSDT", 52500.0)  # TP 도달
        assert len(closed) == 1
        assert closed[0].exit_reason == "tp"
        assert closed[0].pnl == (52000.0 - 50000.0) * 0.1  # 200
        assert acc.balance == 10200.0
        assert len(acc.positions) == 0

    def test_sl_exit(self):
        acc = PaperAccount(initial_balance=10000.0)
        order = {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "entry": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0,
            "qty": 0.1,
            "reason": "test",
        }
        acc.open_position(order)
        closed = acc.check_exits("BTCUSDT", 48500.0)  # SL 도달
        assert len(closed) == 1
        assert closed[0].exit_reason == "sl"
        assert closed[0].pnl == (49000.0 - 50000.0) * 0.1  # -100
        assert acc.balance == 9900.0

    def test_sell_tp_exit(self):
        acc = PaperAccount(initial_balance=10000.0)
        order = {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "entry": 50000.0,
            "sl": 51000.0,
            "tp": 48000.0,
            "qty": 0.1,
            "reason": "test",
        }
        acc.open_position(order)
        closed = acc.check_exits("BTCUSDT", 47500.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "tp"
        assert closed[0].pnl == (50000.0 - 48000.0) * 0.1  # 200

    def test_stats_calculation(self):
        acc = PaperAccount(initial_balance=10000.0)

        for entry, tp_price in [(50000, 52000), (50000, 48000)]:
            order = {
                "symbol": "BTCUSDT", "side": "Buy",
                "entry": entry, "sl": 49000.0, "tp": tp_price,
                "qty": 0.1, "reason": "test",
            }
            acc.open_position(order)

        acc.check_exits("BTCUSDT", 52500.0)  # 첫 번째 TP
        acc.check_exits("BTCUSDT", 48500.0)  # 두 번째 SL (tp=48000이므로 TP 발동)

        stats = acc.get_stats()
        assert stats["total_trades"] == 2
        assert stats["total_trades"] > 0

    def test_unrealized_pnl_in_equity(self):
        acc = PaperAccount(initial_balance=10000.0)
        order = {
            "symbol": "BTCUSDT", "side": "Buy",
            "entry": 50000.0, "sl": 49000.0, "tp": 52000.0,
            "qty": 0.1, "reason": "test",
        }
        acc.open_position(order)
        acc.positions[0].update_pnl(51000.0)
        assert acc.get_equity() == 10100.0  # 10000 + (51000-50000)*0.1
