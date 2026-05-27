"""PaperOrderExecutor 단위 테스트"""
from __future__ import annotations

import pytest
from unittest.mock import patch
from pathlib import Path

import src.paper_trading.paper_engine as pe_module
from src.paper_trading.paper_engine import PaperEngine
from src.exchange.order_executor import PaperOrderExecutor


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "executor_test.db"
    with patch.object(pe_module, "DB_PATH", db):
        yield PaperEngine(initial_balance=10000.0)


@pytest.fixture
def executor(engine):
    return PaperOrderExecutor(engine)


# ─── 시장가 주문 ─────────────────────────────────────────────────────

class TestMarketOrder:
    def test_market_order_success(self, executor, engine):
        result = executor.place_order(
            symbol="BTC/USDT:USDT", direction="long", qty=0.001,
            order_type="market", price=50000, stop_loss=49000, take_profit=52000,
        )
        assert result["status"] == "filled"
        assert result["order_id"].startswith("PAPER-")
        assert result["qty"] == 0.001
        assert "filled_price" in result
        assert len(engine.positions) == 1

    def test_market_order_insufficient_balance(self, executor, engine):
        result = executor.place_order(
            symbol="BTC/USDT:USDT", direction="long", qty=100.0,
            order_type="market", price=50000, stop_loss=49000, take_profit=52000,
        )
        assert result["status"] == "rejected"
        assert "insufficient_balance" in result["reason"]
        assert len(engine.positions) == 0

    def test_market_order_increments_id(self, executor):
        r1 = executor.place_order("BTC/USDT:USDT", "long", 0.001, "market",
                                  price=50000, stop_loss=49000, take_profit=52000)
        r2 = executor.place_order("ETH/USDT:USDT", "short", 0.01, "market",
                                  price=3000, stop_loss=3100, take_profit=2800)
        assert r1["order_id"] == "PAPER-000001"
        assert r2["order_id"] == "PAPER-000002"


# ─── 지정가 주문 ─────────────────────────────────────────────────────

class TestLimitOrder:
    def test_limit_order_pending(self, executor):
        result = executor.place_order(
            symbol="BTC/USDT:USDT", direction="long", qty=0.01,
            order_type="limit", price=48000, stop_loss=47000, take_profit=50000,
        )
        assert result["status"] == "pending"
        assert result["price"] == 48000

    def test_limit_order_no_price_rejected(self, executor):
        result = executor.place_order(
            symbol="BTC/USDT:USDT", direction="long", qty=0.01,
            order_type="limit",
        )
        assert result["status"] == "rejected"
        assert "price" in result["reason"]


# ─── 주문 취소 ───────────────────────────────────────────────────────

class TestCancelOrder:
    def test_cancel_pending_order(self, executor):
        result = executor.place_order(
            "BTC/USDT:USDT", "long", 0.01, "limit",
            price=48000, stop_loss=47000, take_profit=50000,
        )
        assert executor.cancel_order(result["order_id"]) is True
        assert len(executor.get_open_orders()) == 0

    def test_cancel_nonexistent_order(self, executor):
        assert executor.cancel_order("PAPER-999999") is False


# ─── 미체결 주문 조회 ────────────────────────────────────────────────

class TestGetOpenOrders:
    def test_no_orders_initially(self, executor):
        assert executor.get_open_orders() == []

    def test_filter_by_symbol(self, executor):
        executor.place_order("BTC/USDT:USDT", "long", 0.01, "limit",
                             price=48000, stop_loss=47000, take_profit=50000)
        executor.place_order("ETH/USDT:USDT", "short", 0.1, "limit",
                             price=2800, stop_loss=2900, take_profit=2600)
        btc_orders = executor.get_open_orders("BTC/USDT:USDT")
        assert len(btc_orders) == 1
        assert btc_orders[0]["symbol"] == "BTC/USDT:USDT"

    def test_all_symbols(self, executor):
        executor.place_order("BTC/USDT:USDT", "long", 0.01, "limit",
                             price=48000, stop_loss=47000, take_profit=50000)
        executor.place_order("ETH/USDT:USDT", "short", 0.1, "limit",
                             price=2800, stop_loss=2900, take_profit=2600)
        assert len(executor.get_open_orders()) == 2


# ─── 포지션 조회 ─────────────────────────────────────────────────────

class TestGetPosition:
    def test_no_position(self, executor):
        assert executor.get_position("BTC/USDT:USDT") is None

    def test_position_after_market_order(self, executor):
        executor.place_order("BTC/USDT:USDT", "long", 0.001, "market",
                             price=50000, stop_loss=49000, take_profit=52000)
        pos = executor.get_position("BTC/USDT:USDT")
        assert pos is not None
        assert pos["symbol"] == "BTC/USDT:USDT"
        assert pos["direction"] == "long"
        assert pos["qty"] == 0.001
        assert "entry_time" in pos
