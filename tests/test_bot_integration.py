"""
bot.py run() 통합 테스트 — 모든 외부 의존성을 mock하여
run() 함수의 전체 흐름을 검증한다.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock, ANY

import numpy as np
import pandas as pd
import pytest

import src.paper_trading.paper_engine as pe_module
import src.risk.circuit_breaker as cb_module

# ── 공통 mock 데이터 ────────────────────────────────────────────────

MOCK_CONFIG = {
    "exchange": {"symbols": ["BTC/USDT:USDT"], "leverage": 5},
    "capital": {
        "total_capital": 10000,
        "trading_allocation": 0.5,
        "risk_per_trade": 0.01,
    },
    "risk": {
        "min_rr_ratio": 2.0,
        "max_positions": 2,
        "daily_loss_limit": 0.03,
        "weekly_loss_limit": 0.08,
        "max_consecutive_losses": 3,
    },
}


def _mock_ohlcv(n: int = 100) -> pd.DataFrame:
    """일정한 값의 OHLCV 테스트 DataFrame을 생성한다."""
    return pd.DataFrame(
        {
            "open": np.full(n, 50000.0),
            "high": np.full(n, 50500.0),
            "low": np.full(n, 49500.0),
            "close": np.full(n, 50000.0),
            "volume": np.full(n, 1000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )


def _make_signal():
    """테스트용 TradeSignal 객체를 반환한다."""
    from src.strategy.signal_engine import TradeSignal

    return TradeSignal(
        direction="long",
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        symbol="BTC/USDT:USDT",
        reason="4H bullish(BOS) + 1H FVG + KZ(london) + OTE",
        rr_ratio=2.0,
    )


# ── Fixture: DB 경로를 tmp_path로 교체 + 공통 mock 적용 ──────────────


@pytest.fixture()
def _run_env(tmp_path):
    """run() 실행에 필요한 공통 mock 환경을 제공한다.

    Returns:
        dict: mock 객체들을 담은 딕셔너리
    """
    pe_db = tmp_path / "paper.db"
    cb_db = tmp_path / "cb.db"

    mock_client_cls = MagicMock()
    mock_client = mock_client_cls.return_value
    mock_client.fetch_current_price.return_value = 50000.0
    mock_client.fetch_ohlcv.return_value = _mock_ohlcv()

    mock_notifier_cls = MagicMock()
    mock_notifier = mock_notifier_cls.return_value

    with (
        patch.object(pe_module, "DB_PATH", pe_db),
        patch.object(cb_module, "DB_PATH", cb_db),
        patch("src.bot.load_config", return_value=MOCK_CONFIG),
        patch("src.risk.risk_manager.load_config", return_value=MOCK_CONFIG),
        patch("src.exchange.bybit_client.MarketDataClient", mock_client_cls),
        patch(
            "src.notification.discord_bot.DiscordNotifier", mock_notifier_cls
        ),
    ):
        yield {
            "client_cls": mock_client_cls,
            "client": mock_client,
            "notifier_cls": mock_notifier_cls,
            "notifier": mock_notifier,
        }


# ── 테스트 케이스 ───────────────────────────────────────────────────


class TestRunNoSignal:
    """시그널이 없을 때 정상 종료되는지 검증."""

    def test_run_no_signal(self, _run_env):
        """generate_signal이 None을 반환하면 포지션 진입 없이 종료한다."""
        with patch(
            "src.strategy.signal_engine.generate_signal", return_value=None
        ):
            from src.bot import run

            run()

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_not_called()
        notifier.notify_error.assert_not_called()


class TestRunWithSignalEntry:
    """시그널 발생 시 포지션 진입 흐름 검증."""

    def test_run_with_signal_entry(self, _run_env):
        """유효한 시그널이 있으면 포지션을 진입하고 Discord 알림을 보낸다."""
        signal = _make_signal()

        with patch(
            "src.strategy.signal_engine.generate_signal", return_value=signal
        ):
            from src.bot import run

            run()

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_called_once()
        call_kwargs = notifier.notify_entry.call_args
        assert call_kwargs[1]["symbol"] == "BTC/USDT:USDT"
        assert call_kwargs[1]["direction"] == "long"


class TestRunCheckStops:
    """기존 포지션의 SL/TP 체크 흐름 검증."""

    def test_run_check_stops(self, _run_env):
        """기존 포지션이 있을 때 check_stops가 호출되어 SL/TP를 체크한다."""
        # 첫 번째 run: 포지션 진입
        signal = _make_signal()
        with patch(
            "src.strategy.signal_engine.generate_signal", return_value=signal
        ):
            from src.bot import run

            run()

        # 진입 확인
        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_called_once()

        # 두 번째 run: SL 히트 시뮬레이션 (low를 SL 이하로)
        sl_ohlcv = _mock_ohlcv()
        sl_ohlcv.iloc[-1, sl_ohlcv.columns.get_loc("low")] = 49000.0
        _run_env["client"].fetch_ohlcv.return_value = sl_ohlcv

        with patch(
            "src.strategy.signal_engine.generate_signal", return_value=None
        ):
            run()

        # SL 히트로 청산 시 notify_exit 콜백이 실행됨
        # (PaperEngine._fire_on_trade -> bot._on_trade -> notifier.notify_exit)
        # 포지션이 SL에 걸렸으므로 exit 호출이 있어야 함
        assert notifier.notify_exit.called


class TestRunRiskBlocked:
    """서킷브레이커 차단 시 진입이 차단되는지 검증."""

    def test_run_risk_blocked(self, _run_env):
        """서킷브레이커가 차단하면 시그널이 있어도 진입하지 않는다."""
        signal = _make_signal()

        with patch(
            "src.strategy.signal_engine.generate_signal", return_value=signal
        ):
            from src.bot import run
            from src.risk.circuit_breaker import CircuitBreaker

            # 연속 손실 3회를 기록하여 서킷브레이커 발동
            with patch.object(
                CircuitBreaker,
                "is_trading_allowed",
                return_value=(False, "연속 3패 -- 1일 강제 휴식"),
            ):
                run()

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_not_called()


class TestRunExchangeError:
    """거래소 오류 발생 시 에러 핸들링 + Discord 알림 검증."""

    def test_run_exchange_error(self, _run_env):
        """fetch_current_price에서 예외 발생 시 notify_error가 호출된다."""
        _run_env["client"].fetch_current_price.side_effect = ConnectionError(
            "거래소 연결 실패"
        )

        with patch(
            "src.strategy.signal_engine.generate_signal", return_value=None
        ):
            from src.bot import run

            # run()은 예외를 내부에서 잡으므로 정상 종료되어야 함
            run()

        notifier = _run_env["notifier"]
        notifier.notify_error.assert_called_once()
        error_msg = notifier.notify_error.call_args[0][0]
        assert "거래소 연결 실패" in error_msg
