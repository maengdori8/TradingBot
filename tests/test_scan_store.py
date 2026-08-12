from __future__ import annotations

"""scan_store 테스트 — TradingView 변환 + 스캔 상태 저장/로드"""

from src.scan_store import to_tradingview, save_scan_state, load_scan_state


class TestToTradingView:
    def test_perpetual_symbol(self):
        assert to_tradingview("BTC/USDT:USDT") == "BYBIT:BTCUSDT.P"

    def test_eth_perpetual(self):
        assert to_tradingview("ETH/USDT:USDT") == "BYBIT:ETHUSDT.P"

    def test_spot_symbol(self):
        assert to_tradingview("BTC/USDT") == "BYBIT:BTCUSDT"


class TestSaveLoadRoundtrip:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "scan_state.json"
        watchlist = [
            {"symbol": "SOL/USDT:USDT", "score": 72.0, "direction": "long"},
            {"symbol": "XRP/USDT:USDT", "score": 55.0, "direction": "short"},
        ]
        save_scan_state(watchlist, scanned_count=40, qualified_count=1, path=path)

        loaded = load_scan_state(path)
        assert loaded["scanned_count"] == 40
        assert loaded["qualified_count"] == 1
        assert len(loaded["watchlist"]) == 2
        assert loaded["watchlist"][0]["symbol"] == "SOL/USDT:USDT"
        assert loaded["updated_at"] is not None

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = load_scan_state(tmp_path / "nope.json")
        assert loaded["watchlist"] == []
        assert loaded["scanned_count"] == 0
        assert loaded["updated_at"] is None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        loaded = load_scan_state(path)
        assert loaded["watchlist"] == []
