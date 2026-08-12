from __future__ import annotations

"""
Integration test — full bot flow from market data to position close.

All external dependencies (exchange, DB, time) are mocked.
The test exercises:
  market data fetch -> signal generation -> risk check -> position entry
  -> SL/TP check -> position close -> performance recording
"""
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest

import src.risk.circuit_breaker as cb_module
import src.paper_trading.paper_engine as pe_module
from src.paper_trading.paper_engine import PaperEngine
from src.risk.circuit_breaker import CircuitBreaker
from src.strategy.signal_engine import TradeSignal, generate_signal


# ── Helpers ──────────────────────────────────────────────────────────


def _trending_df(n=60, direction=1, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 6 * np.pi, n)
    base = np.linspace(100, 100 + direction * 30, n)
    closes = base + np.sin(x) * 5
    highs = closes + np.abs(np.random.randn(n)) * 1.5 + 1
    lows = closes - np.abs(np.random.randn(n)) * 1.5 - 1
    opens = closes + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": opens, "high": highs,
        "low": lows, "close": closes,
        "volume": np.full(n, 1000.0),
    })


# ── Integration: full flow ───────────────────────────────────────────


class TestBotFlowIntegration:
    """End-to-end flow using mocked signal + real PaperEngine + real CircuitBreaker."""

    @pytest.fixture
    def paper_engine(self, tmp_path):
        db = tmp_path / "paper.db"
        with patch.object(pe_module, "DB_PATH", db):
            yield PaperEngine(initial_balance=10_000.0)

    @pytest.fixture
    def circuit_breaker(self, tmp_path):
        db = tmp_path / "cb.db"
        with patch.object(cb_module, "DB_PATH", db):
            yield CircuitBreaker(
                trading_capital=10_000,
                daily_loss_limit=0.03,
                weekly_loss_limit=0.08,
                max_consecutive_losses=3,
            )

    def test_full_long_tp_flow(self, paper_engine, circuit_breaker):
        """Signal -> risk check -> entry -> TP hit -> close -> record."""
        # 1. Simulate a long signal
        signal = TradeSignal(
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            symbol="BTC/USDT",
            reason="4H bullish + 1H FVG + KZ + OTE",
            rr_ratio=2.0,
        )

        # 2. Risk check
        allowed, reason = circuit_breaker.is_trading_allowed()
        assert allowed is True

        # 3. Open position
        pos = paper_engine.open_position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            qty=0.01,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        assert pos is not None
        assert len(paper_engine.positions) == 1

        # 4. TP hit — close
        pnl = paper_engine.close_position(pos, exit_price=52000.0, reason="TP")
        assert pnl > 0
        assert len(paper_engine.positions) == 0

        # 5. Record in circuit breaker
        circuit_breaker.record_trade(pnl)
        allowed, _ = circuit_breaker.is_trading_allowed()
        assert allowed is True

        # 6. Performance stats
        perf = paper_engine.get_performance()
        assert perf["total_trades"] == 1
        assert perf["win_rate"] == 1.0

    def test_full_short_sl_flow(self, paper_engine, circuit_breaker):
        """Short signal -> SL hit -> loss recorded."""
        signal = TradeSignal(
            direction="short",
            entry_price=3000.0,
            stop_loss=3100.0,
            take_profit=2800.0,
            symbol="ETH/USDT",
            reason="4H bearish + 1H OB + KZ + OTE",
            rr_ratio=2.0,
        )

        pos = paper_engine.open_position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            qty=0.1,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        assert pos is not None

        # SL hit
        pnl = paper_engine.close_position(pos, exit_price=3100.0, reason="SL")
        assert pnl < 0

        circuit_breaker.record_trade(pnl)
        daily = circuit_breaker.get_daily_pnl()
        assert daily < 0

    def test_consecutive_losses_block_new_trades(self, paper_engine, circuit_breaker):
        """3 consecutive losses -> circuit breaker blocks 4th trade."""
        for i in range(3):
            pos = paper_engine.open_position(
                "BTC/USDT", "long", 50000, 0.001, 49000, 52000
            )
            pnl = paper_engine.close_position(pos, 49000, "SL")
            circuit_breaker.record_trade(pnl)

        allowed, reason = circuit_breaker.is_trading_allowed()
        assert allowed is False
        assert "연속" in reason

    def test_check_stops_auto_closes_sl(self, paper_engine):
        """PaperEngine.check_stops closes position when SL is hit."""
        assert paper_engine.open_position(
            "BTC/USDT", "long", 50000, 0.01, 49000, 52000
        ) is not None
        assert len(paper_engine.positions) == 1

        # Simulate candle where low touches SL
        paper_engine.check_stops("BTC/USDT", current_high=50500, current_low=48900)
        assert len(paper_engine.positions) == 0

    def test_check_stops_auto_closes_tp(self, paper_engine):
        """PaperEngine.check_stops closes position when TP is hit."""
        assert paper_engine.open_position(
            "BTC/USDT", "long", 50000, 0.01, 49000, 52000
        ) is not None
        paper_engine.check_stops("BTC/USDT", current_high=52100, current_low=50000)
        assert len(paper_engine.positions) == 0

    def test_multiple_trades_performance(self, paper_engine):
        """Multiple trades produce correct aggregate performance."""
        # 3 wins, 1 loss
        for _ in range(3):
            pos = paper_engine.open_position("BTC/USDT", "long", 50000, 0.01, 49000, 52000)
            paper_engine.close_position(pos, 52000, "TP")

        pos = paper_engine.open_position("BTC/USDT", "long", 50000, 0.01, 49000, 52000)
        paper_engine.close_position(pos, 49000, "SL")

        perf = paper_engine.get_performance()
        assert perf["total_trades"] == 4
        assert perf["win_rate"] == pytest.approx(0.75)
        assert perf["profit_factor"] > 0
        assert "mdd" in perf
        assert "sharpe" in perf


class TestSignalToExecution:
    """Test the signal -> execution pipeline with mocked signal engine."""

    @pytest.fixture
    def paper_engine(self, tmp_path):
        db = tmp_path / "paper.db"
        with patch.object(pe_module, "DB_PATH", db):
            yield PaperEngine(initial_balance=5_000.0)

    @pytest.fixture
    def circuit_breaker(self, tmp_path):
        db = tmp_path / "cb.db"
        with patch.object(cb_module, "DB_PATH", db):
            yield CircuitBreaker(
                trading_capital=5_000,
                daily_loss_limit=0.03,
                weekly_loss_limit=0.08,
                max_consecutive_losses=3,
            )

    def test_signal_generates_and_executes(self, paper_engine, circuit_breaker):
        """generate_signal -> execute trade -> close with profit."""
        with patch("src.strategy.signal_engine.is_in_kill_zone", return_value=True), \
             patch("src.strategy.signal_engine.detect_bos", return_value="bullish"), \
             patch("src.strategy.signal_engine.detect_choch", return_value=None), \
             patch("src.strategy.signal_engine.detect_fvg", return_value=[]), \
             patch("src.strategy.signal_engine.detect_order_blocks", return_value=[]), \
             patch("src.strategy.signal_engine.is_price_in_fvg", return_value=[MagicMock(type="bullish")]), \
             patch("src.strategy.signal_engine.is_price_in_ob", return_value=[]), \
             patch("src.strategy.signal_engine.is_price_in_ote", return_value=True):
            df = _trending_df(60, 1)
            price = float(df["close"].iloc[-1])
            signal = generate_signal(df, df, df, "BTC/USDT", price, min_rr=2.0)

            if signal is not None:
                allowed, _ = circuit_breaker.is_trading_allowed()
                assert allowed is True

                pos = paper_engine.open_position(
                    signal.symbol, signal.direction,
                    signal.entry_price, 0.01,
                    signal.stop_loss, signal.take_profit,
                )
                assert pos is not None

                # Close at take profit
                pnl = paper_engine.close_position(pos, signal.take_profit, "TP")
                circuit_breaker.record_trade(pnl)
                assert pnl > 0
