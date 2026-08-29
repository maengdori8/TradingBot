"""백테스터 단위 테스트 — 합성 데이터로 ICT 전략 백테스트 검증"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.paper_trading.backtester import Backtester, BacktestResult


# ── 헬퍼 ─────────────────────────────────────────────────────────────

def _make_ohlcv(n=200, base_price=100, direction=1, seed=42):
    """합성 OHLCV DataFrame 생성."""
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    closes = np.linspace(base_price, base_price + direction * 20, n) + np.random.randn(n) * 0.5
    highs = closes + np.abs(np.random.randn(n)) * 1.5 + 0.5
    lows = closes - np.abs(np.random.randn(n)) * 1.5 - 0.5
    opens = closes + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": opens, "high": highs,
        "low": lows, "close": closes,
        "volume": np.full(n, 1000.0),
    }, index=timestamps)


# ── BacktestResult 테스트 ────────────────────────────────────────────

class TestBacktestResult:
    def test_dataclass_fields(self):
        r = BacktestResult(
            total_trades=5, win_rate=0.6, total_pnl=100.0,
            max_drawdown=0.05, sharpe_ratio=1.5,
            profit_factor=2.0, avg_trade_pnl=20.0,
        )
        assert r.total_trades == 5
        assert r.win_rate == 0.6
        assert r.trades == []
        assert r.equity_curve == []

    def test_with_trades_and_curve(self):
        r = BacktestResult(
            total_trades=1, win_rate=1.0, total_pnl=50.0,
            max_drawdown=0.01, sharpe_ratio=2.0,
            profit_factor=999.0, avg_trade_pnl=50.0,
            trades=[{"entry_price": 100}],
            equity_curve=[1000, 1050],
        )
        assert len(r.trades) == 1
        assert len(r.equity_curve) == 2


# ── Backtester 초기화 ────────────────────────────────────────────────

class TestBacktesterInit:
    def test_default_params(self):
        bt = Backtester()
        assert bt.initial_balance == 1250.0
        assert bt.risk_per_trade == 0.005   # 2026-06 신체제 동기화
        assert bt.leverage == 5.0
        assert bt.min_rr == 2.5             # 2026-06 신체제 동기화
        assert bt.ignore_kill_zone is False

    def test_custom_params(self):
        bt = Backtester(
            initial_balance=5000, risk_per_trade=0.02,
            leverage=3, min_rr=1.5, ignore_kill_zone=True,
        )
        assert bt.initial_balance == 5000
        assert bt.ignore_kill_zone is True

    def test_get_results_before_run_raises(self):
        bt = Backtester()
        with pytest.raises(RuntimeError, match="run"):
            bt.get_results()


# ── 정적 메서드 테스트 ───────────────────────────────────────────────

class TestStaticMethods:
    def test_max_drawdown_basic(self):
        # 100 → 110 → 90 → 100: 최대 낙폭 = (110-90)/110 ≈ 18.18%
        mdd = Backtester._calculate_max_drawdown([100, 110, 90, 100])
        assert mdd == pytest.approx(20 / 110, abs=0.01)

    def test_max_drawdown_empty(self):
        assert Backtester._calculate_max_drawdown([]) == 0.0

    def test_max_drawdown_monotonic_up(self):
        assert Backtester._calculate_max_drawdown([100, 110, 120]) == 0.0

    def test_sharpe_positive(self):
        # 꾸준한 상승: Sharpe > 0
        curve = [100 + i * 0.1 for i in range(100)]
        sharpe = Backtester._calculate_sharpe(curve)
        assert sharpe > 0

    def test_sharpe_empty(self):
        assert Backtester._calculate_sharpe([]) == 0.0

    def test_sharpe_single_point(self):
        assert Backtester._calculate_sharpe([100]) == 0.0


# ── 백테스트 실행 (신호 없음) ────────────────────────────────────────

class TestBacktesterNoSignals:
    def test_no_signal_returns_zero_trades(self):
        """모든 신호가 None이면 거래 0건."""
        bt = Backtester(initial_balance=1000, ignore_kill_zone=True)

        df_15m = _make_ohlcv(200, direction=0)
        df_1h = _make_ohlcv(200, direction=0)
        df_4h = _make_ohlcv(200, direction=0)

        with patch("src.paper_trading.backtester.generate_signal", return_value=None):
            result = bt.run(df_4h, df_1h, df_15m, "TEST/USDT")

        assert result.total_trades == 0
        assert result.total_pnl == 0.0

    def test_insufficient_data(self):
        """데이터 부족 시 빈 결과."""
        bt = Backtester(initial_balance=1000)

        df_short = _make_ohlcv(50)  # lookback 100 미만
        result = bt.run(df_short, df_short, df_short, "TEST/USDT")

        assert result.total_trades == 0


# ── 백테스트 실행 (mock 신호) ────────────────────────────────────────

class TestBacktesterWithSignals:
    def test_single_trade_long(self):
        """Long 신호 1건 → 진입 후 SL/TP에서 청산."""
        from src.strategy.signal_engine import TradeSignal

        bt = Backtester(initial_balance=1000, ignore_kill_zone=True)

        df_15m = _make_ohlcv(200, base_price=100, direction=1)
        df_1h = _make_ohlcv(200, base_price=100, direction=1)
        df_4h = _make_ohlcv(200, base_price=100, direction=1)

        price = float(df_15m.iloc[100]["close"])
        signal = TradeSignal(
            direction="long",
            entry_price=price,
            stop_loss=price * 0.98,
            take_profit=price * 1.04,
            symbol="TEST/USDT",
            reason="test signal",
            rr_ratio=2.0,
        )

        call_count = 0

        def mock_signal(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 첫 번째 호출에서만 신호 반환
            if call_count == 1:
                return signal
            return None

        with patch("src.paper_trading.backtester.generate_signal", side_effect=mock_signal):
            result = bt.run(df_4h, df_1h, df_15m, "TEST/USDT")

        # 최소 1건 이상 거래 (진입 + SL/TP 또는 backtest_end 청산)
        assert result.total_trades >= 1
        assert len(result.equity_curve) > 0

    def test_get_results_after_run(self):
        """run() 후 get_results() 동일 결과 반환."""
        bt = Backtester(initial_balance=1000, ignore_kill_zone=True)
        df = _make_ohlcv(200)

        with patch("src.paper_trading.backtester.generate_signal", return_value=None):
            result1 = bt.run(df, df, df, "TEST/USDT")

        result2 = bt.get_results()
        assert result1.total_trades == result2.total_trades
        assert result1.total_pnl == result2.total_pnl
