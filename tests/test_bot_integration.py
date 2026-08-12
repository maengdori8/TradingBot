from __future__ import annotations

"""
bot.py run() 통합 테스트 — 모든 외부 의존성을 mock하여
전체 코인 스캔 기반 run() 흐름을 검증한다.
"""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

import src.paper_trading.paper_engine as pe_module
import src.risk.circuit_breaker as cb_module
from src.data.market_snapshot import DataProvenance, MarketSnapshot

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
        "max_per_symbol": 1,
        "max_same_direction": 3,
        "max_exposure_pct": 0.80,
        "daily_loss_limit": 0.03,
        "weekly_loss_limit": 0.08,
        "max_consecutive_losses": 3,
    },
    "scan": {"mode": "static", "min_score": 75, "require_volume": True},
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


def _make_scan(qualified: bool, signal=None, score: float = 80.0):
    """테스트용 ScanResult 객체를 반환한다."""
    from src.strategy.signal_engine import ScanResult

    return ScanResult(
        symbol="BTC/USDT:USDT",
        direction="long",
        score=score,
        stage=4 if qualified else 2,
        qualified=qualified,
        price=50000.0,
        reason="test scan",
        signal=signal,
        checks={},
    )


# ── Fixture: DB 경로를 tmp_path로 교체 + 공통 mock 적용 ──────────────


@pytest.fixture()
def _run_env(tmp_path):
    """run() 실행에 필요한 공통 mock 환경을 제공한다."""
    pe_db = tmp_path / "paper.db"
    cb_db = tmp_path / "cb.db"

    mock_client_cls = MagicMock()
    mock_client = mock_client_cls.return_value
    mock_client.fetch_current_price.return_value = 50000.0
    mock_client.fetch_ohlcv.return_value = _mock_ohlcv()
    received = datetime.now(timezone.utc)
    mock_client.fetch_market_snapshot.return_value = MarketSnapshot(
        exchange_timestamp=received,
        receive_timestamp=received,
        provenance=DataProvenance(
            exchange="bybit",
            market_type="swap",
            requested_symbol="BTC/USDT:USDT",
            resolved_symbol="BTC/USDT:USDT",
            endpoint="fetch_order_book",
        ),
        symbol="BTC/USDT:USDT",
        last=50000.0,
        bid=49999.0,
        ask=50001.0,
        bids=((49999.0, 2.0),),
        asks=((50001.0, 2.0),),
    )

    mock_notifier_cls = MagicMock()
    mock_notifier = mock_notifier_cls.return_value

    with (
        patch.object(pe_module, "DB_PATH", pe_db),
        patch.object(cb_module, "DB_PATH", cb_db),
        patch("src.bot.load_config", return_value=MOCK_CONFIG),
        patch("src.risk.risk_manager.load_config", return_value=MOCK_CONFIG),
        patch("src.exchange.bybit_client.MarketDataClient", mock_client_cls),
        patch("src.notification.discord_bot.DiscordNotifier", mock_notifier_cls),
        patch("src.scan_store.save_scan_state"),
        patch("src.risk.learner.maybe_update"),   # 자동학습 격리 (실제 파일 미접근)
    ):
        yield {
            "client_cls": mock_client_cls,
            "client": mock_client,
            "notifier_cls": mock_notifier_cls,
            "notifier": mock_notifier,
        }


# ── 테스트 케이스 ───────────────────────────────────────────────────


class TestRunNoSignal:
    """확정 신호가 없을 때 진입하지 않는지 검증."""

    def test_run_no_signal(self, _run_env):
        """scan_symbol이 qualified=False면 포지션 진입 없이 종료한다."""
        with patch(
            "src.strategy.signal_engine.scan_symbol",
            return_value=_make_scan(qualified=False),
        ):
            from src.bot import run
            run()

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_not_called()


class TestRunWithSignalEntry:
    """확정 신호 발생 시 포지션 진입 흐름 검증."""

    def test_run_with_signal_entry(self, _run_env):
        """qualified 스캔 결과가 있으면 포지션을 진입하고 알림을 보낸다."""
        scan = _make_scan(qualified=True, signal=_make_signal())

        with patch(
            "src.strategy.signal_engine.scan_symbol", return_value=scan
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
        """보유 포지션이 있을 때 SL 히트 시 청산 알림이 발생한다."""
        # 1차 run: 진입
        scan = _make_scan(qualified=True, signal=_make_signal())
        with patch(
            "src.strategy.signal_engine.scan_symbol", return_value=scan
        ):
            from src.bot import run
            run()

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_called_once()

        # 2차 run: SL 히트 (low를 SL 이하로)
        sl_ohlcv = _mock_ohlcv()
        sl_ohlcv.iloc[-1, sl_ohlcv.columns.get_loc("low")] = 49000.0
        _run_env["client"].fetch_ohlcv.return_value = sl_ohlcv

        with patch(
            "src.strategy.signal_engine.scan_symbol",
            return_value=_make_scan(qualified=False),
        ):
            run()

        assert notifier.notify_exit.called


class TestRunRiskBlocked:
    """서킷브레이커 차단 시 진입이 차단되는지 검증."""

    def test_run_risk_blocked(self, _run_env):
        """서킷브레이커가 차단하면 확정 신호가 있어도 진입하지 않는다."""
        scan = _make_scan(qualified=True, signal=_make_signal())

        with patch(
            "src.strategy.signal_engine.scan_symbol", return_value=scan
        ):
            from src.bot import run
            from src.risk.circuit_breaker import CircuitBreaker

            with patch.object(
                CircuitBreaker,
                "is_trading_allowed",
                return_value=(False, "연속 3패 -- 1일 강제 휴식"),
            ):
                run()

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_not_called()


class TestRunExchangeError:
    """거래소 오류 발생 시 안전하게 처리되는지 검증."""

    def test_run_exchange_error(self, _run_env):
        """fetch 예외 발생 시 해당 심볼을 건너뛰고 run()이 정상 종료한다."""
        _run_env["client"].fetch_current_price.side_effect = ConnectionError(
            "거래소 연결 실패"
        )

        with patch(
            "src.strategy.signal_engine.scan_symbol",
            return_value=_make_scan(qualified=False),
        ):
            from src.bot import run
            run()  # 예외 없이 종료해야 함

        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_not_called()
