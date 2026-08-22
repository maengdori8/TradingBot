"""
bot.py run() 통합 테스트 — 모든 외부 의존성을 mock하여
전체 코인 스캔 기반 run() 흐름을 검증한다.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

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
    """일정한 값의 OHLCV 테스트 DataFrame을 생성한다.

    인덱스는 '현재 형성 중인 15m 봉'에서 끝나도록 잡는다 — 엔진의 SL/TP 판정이 진입 이후 봉만
    시간순으로 보기 때문에(과거 날짜 프레임이면 진입 이후 봉이 없어 판정이 일어나지 않는다).
    """
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    return pd.DataFrame(
        {
            "open": np.full(n, 50000.0),
            "high": np.full(n, 50500.0),
            "low": np.full(n, 49500.0),
            "close": np.full(n, 50000.0),
            "volume": np.full(n, 1000.0),
        },
        index=pd.date_range(end=end, periods=n, freq="15min", tz="UTC"),
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
    # 실행 판정용 Bybit 전용 경로 (기본: 사용 가능, 같은 프레임)
    mock_client.bybit_available.return_value = True
    mock_client.fetch_ohlcv_bybit.return_value = _mock_ohlcv()
    mock_client.fetch_ohlcv_history.return_value = _mock_ohlcv()
    mock_client.fetch_current_price_bybit.return_value = 50000.0

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

        # 시간 경과 시뮬레이션: 진입을 40분 전으로 되돌려 '진입 뒤 닫힌 봉'이 존재하게 한다
        # (엔진은 형성 중 봉을 판정하지 않으므로, 같은 분 안에서는 SL이 날 수 없다)
        import sqlite3 as _sq
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        _c = _sq.connect(pe_module.DB_PATH)
        _c.execute("UPDATE open_positions SET entry_time=?, last_checked_bar=NULL",
                   ((_dt.now(_tz.utc) - _td(minutes=40)).isoformat(),))
        _c.commit()
        _c.close()

        # 2차 run: SL 히트 (직전 닫힌 봉 low를 SL 이하로)
        sl_ohlcv = _mock_ohlcv()
        sl_ohlcv.iloc[-2, sl_ohlcv.columns.get_loc("low")] = 49000.0   # 직전 닫힌 봉에서 SL
        _run_env["client"].fetch_ohlcv.return_value = sl_ohlcv
        _run_env["client"].fetch_ohlcv_bybit.return_value = sl_ohlcv

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


# ── 메이커 모드 배선: 등록 → (닫힌 봉 터치) 체결 → SL 청산 ───────────────────────


class TestRunMakerMode:
    """execution.mode=maker 에서 run()의 주문 등록/체결/청산 배선을 검증한다."""

    def test_maker_place_fill_then_sl(self, _run_env, tmp_path):
        import sqlite3
        from datetime import datetime, timedelta, timezone

        cfg = dict(MOCK_CONFIG)
        cfg["execution"] = {"mode": "maker", "limit_offset_r": 0.10, "fill_expiry_bars": 8,
                            "maker_fee": 0.00035}
        db_path = pe_module.DB_PATH   # _run_env 가 tmp 경로로 패치해 둠

        # 1차 run: 확정 신호 → 시장가 진입이 아니라 지정가 '등록'만 (체결 알림 없음)
        scan = _make_scan(qualified=True, signal=_make_signal())
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
            from src.bot import run
            run()
        notifier = _run_env["notifier"]
        notifier.notify_entry.assert_not_called()
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, status, limit_price, signal_key FROM pending_orders").fetchall()
        assert len(rows) == 1 and rows[0][1] == "pending"
        assert rows[0][2] == pytest.approx(50000.0 - 0.10 * 1000.0)   # 49900
        assert rows[0][3] and rows[0][3].startswith("BTC/USDT:USDT|long|")
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0

        # 같은 결정봉에서 다시 run 해도 중복 등록 없음
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
            run()
        assert conn.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0] == 1

        # 주문 시각을 40분 전으로 돌려 '닫힌 대상봉'이 존재하게 만든다 (mock 프레임 저가 49500 ≤ 49900 → 체결)
        placed = datetime.now(timezone.utc) - timedelta(minutes=40)
        conn.execute("UPDATE pending_orders SET place_time=?", (placed.isoformat(),))
        conn.commit()

        # 2차 run: 체결 → 진입 알림 1회, 포지션 1개, 주문 filled
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            run()
        notifier.notify_entry.assert_called_once()
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 1
        st = conn.execute("SELECT status, position_id FROM pending_orders").fetchone()
        assert st[0] == "filled" and st[1] is not None
        assert conn.execute("SELECT is_maker FROM open_positions").fetchone()[0] == 1
        notifier.notify_exit.assert_not_called()

        # 3차 run: 직전 닫힌 봉 저가를 SL(49000) 이하로 → SL 청산 알림 (형성 중 봉은 판정하지 않음)
        # 2차 run이 직전 닫힌 봉까지 판정 완료로 표시했으므로, 시간 경과를 흉내내기 위해 표시를 되돌린다
        conn.execute("UPDATE open_positions SET last_checked_bar=NULL")
        conn.commit()
        sl_ohlcv = _mock_ohlcv()
        sl_ohlcv.iloc[-2, sl_ohlcv.columns.get_loc("low")] = 48990.0
        _run_env["client"].fetch_ohlcv.return_value = sl_ohlcv
        _run_env["client"].fetch_ohlcv_bybit.return_value = sl_ohlcv
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            run()
        assert notifier.notify_exit.called
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
        tr = conn.execute("SELECT status, is_maker, entry_fee FROM trades").fetchone()
        assert tr[0] == "SL" and tr[1] == 1 and tr[2] > 0
        conn.close()

    def test_taker_mode_cancels_leftover_pending(self, _run_env):
        """maker 로 등록된 미체결이 남은 상태에서 taker 모드로 기동하면 전부 취소된다."""
        import sqlite3
        cfg_m = dict(MOCK_CONFIG)
        cfg_m["execution"] = {"mode": "maker"}
        scan = _make_scan(qualified=True, signal=_make_signal())
        with patch("src.bot.load_config", return_value=cfg_m), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
            from src.bot import run
            run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        assert conn.execute("SELECT COUNT(*) FROM pending_orders WHERE status='pending'").fetchone()[0] == 1
        # taker 모드 (execution 섹션 없음 = 기본 taker)
        with patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            run()
        row = conn.execute("SELECT status, resolve_reason FROM pending_orders").fetchone()
        assert row[0] == "cancelled" and row[1] == "mode_change"
        conn.close()

    def test_invalid_mode_fails_loud(self, _run_env):
        """execution.mode 오타는 조용히 taker 가 되지 않고 ValueError 로 기동 실패한다."""
        cfg = dict(MOCK_CONFIG)
        cfg["execution"] = {"mode": "makr"}
        with patch("src.bot.load_config", return_value=cfg):
            from src.bot import run
            with pytest.raises(ValueError):
                run()


# ── 실행 데이터 무결성: Bybit 불가 → 관찰 전용, 서킷브레이커 순서 ─────────────


class TestExecutionIntegrity:
    def test_observe_only_when_bybit_unavailable(self, _run_env):
        """Bybit 접근 불가(403 등)면 신규 진입/주문을 하지 않고 알림만 보낸다 (폴백 캔들로 기록 오염 금지)."""
        import sqlite3
        _run_env["client"].bybit_available.return_value = False
        cfg = dict(MOCK_CONFIG)
        cfg["execution"] = {"mode": "maker"}
        scan = _make_scan(qualified=True, signal=_make_signal())
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
            from src.bot import run
            run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        assert conn.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
        conn.close()
        assert _run_env["notifier"].notify_error.called
        _run_env["notifier"].notify_entry.assert_not_called()
        # taker 모드도 동일
        with patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
            run()
        _run_env["notifier"].notify_entry.assert_not_called()

    def test_loss_before_fill_cancels_pending(self, _run_env):
        """A의 SL(이른 봉)이 일일 손실한도를 넘기면, 그 뒤 봉에서 체결될 B의 미체결은 취소된다 (심볼 순서 무관)."""
        import sqlite3
        from datetime import datetime, timedelta, timezone
        from src.paper_trading.paper_engine import PaperEngine, _bar_open

        cfg = dict(MOCK_CONFIG)
        cfg["exchange"] = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "leverage": 5}
        cfg["risk"] = dict(MOCK_CONFIG["risk"], daily_loss_limit=0.0001, max_positions=4)
        cfg["execution"] = {"mode": "maker"}
        now = datetime.now(timezone.utc)
        # 사전 상태: A 포지션(75분 전 진입, SL 49000), B 미체결(45분 전 주문, limit 49900)
        eng = PaperEngine(initial_balance=5000.0, db_path=pe_module.DB_PATH)
        eng.open_position("BTC/USDT:USDT", "long", 50000.0, 0.05, 49000.0, 52000.0,
                          entry_time=now - timedelta(minutes=75))
        eng.place_pending_limit("ETH/USDT:USDT", "long", 49900.0, 0.01, 49000.0, 52000.0,
                                place_time=now - timedelta(minutes=45))
        eng.conn.close()
        # 프레임: A의 SL은 60분 전 봉(진입봉 다음), B의 첫 대상봉은 30분 전 봉(그 뒤) — 둘 다 닫힘
        frame = _mock_ohlcv()
        sl_bar = _bar_open(now - timedelta(minutes=60), 15)
        frame.loc[frame.index == sl_bar, "low"] = 48990.0
        _run_env["client"].fetch_ohlcv.return_value = frame
        _run_env["client"].fetch_ohlcv_bybit.return_value = frame
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            from src.bot import run
            run()
        assert _run_env["notifier"].notify_exit.called          # A SL 청산
        conn = sqlite3.connect(pe_module.DB_PATH)
        st = conn.execute("SELECT status, resolve_reason FROM pending_orders").fetchone()
        assert st[0] == "cancelled" and st[1].startswith("risk_blocked")
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0   # B 미체결
        conn.close()
        _run_env["notifier"].notify_entry.assert_not_called()


class TestExecutionDataIntegrity:
    """실행 데이터(Bybit 전용 프레임) 결손 시 보수 동작."""

    def test_symbol_without_bybit_frame_gets_no_entry(self, _run_env):
        """Bybit 초기화는 됐지만 해당 심볼 15m 조회가 실패하면 폴백 신호로 주문/진입하지 않는다."""
        import sqlite3
        _run_env["client"].fetch_ohlcv_bybit.side_effect = RuntimeError("403")
        scan = _make_scan(qualified=True, signal=_make_signal())
        for mode in ("maker", "taker"):
            cfg = dict(MOCK_CONFIG)
            cfg["execution"] = {"mode": mode}
            with patch("src.bot.load_config", return_value=cfg), \
                 patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
                from src.bot import run
                run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        assert conn.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
        conn.close()
        _run_env["notifier"].notify_entry.assert_not_called()

    def test_not_deeper_frames_hold_everything(self, _run_env):
        """100봉 프레임에 빈 봉이 있고, 1000봉/히스토리 재조회가 같은(더 깊지 않은) 프레임을 돌려주면
        포지션 판정은 진행하지 않고(TP 부여 금지), 기존 포지션 판정 미완이므로 미체결 체결도 보류된다."""
        import sqlite3
        from datetime import datetime, timedelta, timezone
        from src.paper_trading.paper_engine import PaperEngine, _bar_open

        cfg = dict(MOCK_CONFIG)
        cfg["exchange"] = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "leverage": 5}
        cfg["risk"] = dict(MOCK_CONFIG["risk"], max_positions=4)
        cfg["execution"] = {"mode": "maker"}
        now = datetime.now(timezone.utc)
        eng = PaperEngine(initial_balance=5000.0, db_path=pe_module.DB_PATH)
        pos = eng.open_position("BTC/USDT:USDT", "long", 50000.0, 0.05, 49000.0, 50400.0,
                                entry_time=now - timedelta(minutes=75))
        eng.place_pending_limit("ETH/USDT:USDT", "long", 49900.0, 0.01, 49000.0, 52000.0,
                                place_time=now - timedelta(minutes=45))
        eng.conn.close()
        # 100봉 프레임: 60분 전 봉 누락(빈 봉), 30분 전 봉 고가는 TP(50400) 이상
        frame = _mock_ohlcv()
        hole = _bar_open(now - timedelta(minutes=60), 15)
        tp_bar = _bar_open(now - timedelta(minutes=30), 15)
        frame.loc[frame.index == tp_bar, "high"] = 50600.0
        holey = frame[frame.index != hole]
        _run_env["client"].fetch_ohlcv.return_value = holey
        _run_env["client"].fetch_ohlcv_bybit.side_effect = lambda *a, **k: holey     # 1000봉도 같은 프레임
        _run_env["client"].fetch_ohlcv_history.return_value = holey                 # 백필도 같은 프레임
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            from src.bot import run
            run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        # BTC 포지션: 빈 봉에서 멈춤 → TP 미부여, last_checked_bar < 빈 봉
        row = conn.execute("SELECT last_checked_bar FROM open_positions WHERE symbol='BTC/USDT:USDT'").fetchone()
        assert row is not None
        assert row[0] is None or datetime.fromisoformat(row[0]) < hole
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
        # ETH 미체결: 체결 가능한 봉이 있어도 보류(취소도 아님)
        st = conn.execute("SELECT status FROM pending_orders").fetchone()
        assert st[0] == "pending"
        conn.close()
        _run_env["notifier"].notify_entry.assert_not_called()
        _run_env["notifier"].notify_exit.assert_not_called()


class TestCycleGating:
    """판정 미완 → 신규 진입 보류 / 미체결 체결은 예상 체결봉 시간순."""

    @staticmethod
    def _eth_scan():
        from src.strategy.signal_engine import ScanResult, TradeSignal
        sig = TradeSignal(direction="long", entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0,
                          symbol="ETH/USDT:USDT", reason="test", rr_ratio=2.0)
        return ScanResult(symbol="ETH/USDT:USDT", direction="long", score=85.0, stage=4, qualified=True,
                          price=50000.0, reason="test", signal=sig, checks={})

    def test_incomplete_position_blocks_new_entries(self, _run_env):
        """BTC 포지션 판정이 미완(더 깊은 프레임 없음)이면 ETH의 확정 신호도 주문/진입하지 않는다 (양 모드)."""
        import sqlite3
        from datetime import datetime, timedelta, timezone
        from src.paper_trading.paper_engine import PaperEngine, _bar_open

        now = datetime.now(timezone.utc)
        eng = PaperEngine(initial_balance=5000.0, db_path=pe_module.DB_PATH)
        eng.open_position("BTC/USDT:USDT", "long", 50000.0, 0.05, 49000.0, 52000.0,
                          entry_time=now - timedelta(minutes=75))
        eng.conn.close()
        frame = _mock_ohlcv()
        hole = _bar_open(now - timedelta(minutes=60), 15)
        holey = frame[frame.index != hole]
        _run_env["client"].fetch_ohlcv.return_value = holey
        _run_env["client"].fetch_ohlcv_bybit.side_effect = lambda *a, **k: holey
        _run_env["client"].fetch_ohlcv_history.return_value = holey
        eth = self._eth_scan()
        for mode in ("maker", "taker"):
            cfg = dict(MOCK_CONFIG)
            cfg["exchange"] = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "leverage": 5}
            cfg["risk"] = dict(MOCK_CONFIG["risk"], max_positions=4)
            cfg["execution"] = {"mode": mode}

            def _scan(df_4h, df_1h, df_15m, symbol, price, **kw):
                return eth if symbol == "ETH/USDT:USDT" else _make_scan(qualified=False)

            with patch("src.bot.load_config", return_value=cfg), \
                 patch("src.risk.risk_manager.load_config", return_value=cfg), \
                 patch("src.strategy.signal_engine.scan_symbol", side_effect=_scan):
                from src.bot import run
                run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        assert conn.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 1   # BTC만
        conn.close()
        _run_env["notifier"].notify_entry.assert_not_called()

    def test_fills_processed_in_chronological_order_across_symbols(self, _run_env):
        """ETH가 먼저 체결·SL(한도 초과) → 그 뒤 봉에서 체결될 BTC는 취소. (알파벳 순이면 BTC가 먼저 체결됐을 것)"""
        import sqlite3
        from datetime import datetime, timedelta, timezone
        from src.paper_trading.paper_engine import PaperEngine, _bar_open

        cfg = dict(MOCK_CONFIG)
        cfg["exchange"] = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "leverage": 5}
        cfg["risk"] = dict(MOCK_CONFIG["risk"], daily_loss_limit=0.0001, max_positions=4)
        cfg["execution"] = {"mode": "maker"}
        now = datetime.now(timezone.utc)
        eng = PaperEngine(initial_balance=5000.0, db_path=pe_module.DB_PATH)
        # ETH: 60분 전 주문 → 첫 대상봉 floor(now-60)+15 = floor(now-45) 에서 체결(저가 49500 ≤ 49900)
        eng.place_pending_limit("ETH/USDT:USDT", "long", 49900.0, 0.05, 49000.0, 52000.0,
                                place_time=now - timedelta(minutes=60))
        # BTC: 30분 전 주문 → 첫 대상봉 floor(now-30)+15 = floor(now-15) (ETH의 SL 봉 뒤)
        eng.place_pending_limit("BTC/USDT:USDT", "long", 49900.0, 0.01, 49000.0, 52000.0,
                                place_time=now - timedelta(minutes=30))
        eng.conn.close()
        frame = _mock_ohlcv()
        sl_bar = _bar_open(now - timedelta(minutes=30), 15)          # ETH 체결봉 다음 봉에서 SL
        frame.loc[frame.index == sl_bar, "low"] = 48990.0
        _run_env["client"].fetch_ohlcv.return_value = frame
        _run_env["client"].fetch_ohlcv_bybit.return_value = frame
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            from src.bot import run
            run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        rows = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT symbol, status, resolve_reason FROM pending_orders")}
        assert rows["ETH/USDT:USDT"][0] == "filled"
        assert rows["BTC/USDT:USDT"][0] == "cancelled" and rows["BTC/USDT:USDT"][1].startswith("risk_blocked")
        trades = conn.execute("SELECT symbol, status FROM trades").fetchall()
        assert trades == [("ETH/USDT:USDT", "SL")]
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
        conn.close()
        _run_env["notifier"].notify_entry.assert_called_once()


class TestObservationCutoffAndExecData:
    """사이클 단위 관측 기준 시각 + 실행 데이터(Bybit 현재가/프레임) 검증."""

    def test_all_engine_evaluations_share_one_cycle_now(self, _run_env):
        """한 사이클의 사전판정·체결·SL/TP 판정은 모두 같은 now(프레임 수신 전 시각)를 쓴다."""
        from datetime import datetime, timedelta, timezone
        from src.paper_trading.paper_engine import PaperEngine
        now0 = datetime.now(timezone.utc)
        eng = PaperEngine(initial_balance=5000.0, db_path=pe_module.DB_PATH)
        eng.open_position("BTC/USDT:USDT", "long", 50000.0, 0.01, 49000.0, 52000.0,
                          entry_time=now0 - timedelta(minutes=75))
        eng.place_pending_limit("ETH/USDT:USDT", "long", 49900.0, 0.01, 49000.0, 52000.0,
                                place_time=now0 - timedelta(minutes=5))   # 아직 대상봉 없음 → 체결 X
        eng.conn.close()
        seen: list = []
        real_stops = PaperEngine.check_stops_history
        real_peek = PaperEngine.peek_pending_fills
        real_fills = PaperEngine.check_pending_fills

        def rec(fn):
            def _w(self_, symbol, candles, now=None, **kw):
                seen.append(now)
                return fn(self_, symbol, candles, now=now, **kw)
            return _w

        cfg = dict(MOCK_CONFIG)
        cfg["exchange"] = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "leverage": 5}
        cfg["risk"] = dict(MOCK_CONFIG["risk"], max_positions=4)
        cfg["execution"] = {"mode": "maker"}
        with patch.object(PaperEngine, "check_stops_history", rec(real_stops)), \
             patch.object(PaperEngine, "peek_pending_fills", rec(real_peek)), \
             patch.object(PaperEngine, "check_pending_fills", rec(real_fills)), \
             patch("src.bot.load_config", return_value=cfg), \
             patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            from src.bot import run
            run()
        assert len(seen) >= 3
        assert all(t is not None for t in seen)
        assert len({t for t in seen}) == 1                    # 전부 동일한 고정 시각
        assert now0 <= seen[0] <= datetime.now(timezone.utc)

    def test_empty_or_stale_exec_frame_or_no_bybit_price_blocks_entry(self, _run_env):
        """Bybit 프레임이 비었거나 지연/미래거나 Bybit 현재가가 없으면 진입/주문하지 않는다 (양 모드)."""
        import sqlite3
        import pandas as pd
        scan = _make_scan(qualified=True, signal=_make_signal())
        cases = []
        empty = _mock_ohlcv().iloc[0:0]
        cases.append(("empty", empty, 50000.0))
        stale = _mock_ohlcv()
        stale.index = stale.index - pd.Timedelta(hours=3)
        cases.append(("stale", stale, 50000.0))
        future = _mock_ohlcv()
        future.index = future.index + pd.Timedelta(hours=3)
        cases.append(("future", future, 50000.0))
        next_bar = _mock_ohlcv()
        next_bar.index = next_bar.index + pd.Timedelta(minutes=15)     # 바로 다음 봉(미래)까지 포함
        cases.append(("next-bar-future", next_bar, 50000.0))
        cases.append(("no-price", _mock_ohlcv(), RuntimeError("no ticker")))
        for name, frame, price in cases:
            _run_env["client"].fetch_ohlcv_bybit.side_effect = None
            _run_env["client"].fetch_ohlcv_bybit.return_value = frame
            if isinstance(price, Exception):
                _run_env["client"].fetch_current_price_bybit.side_effect = price
            else:
                _run_env["client"].fetch_current_price_bybit.side_effect = None
                _run_env["client"].fetch_current_price_bybit.return_value = price
            for mode in ("maker", "taker"):
                cfg = dict(MOCK_CONFIG)
                cfg["execution"] = {"mode": mode}
                with patch("src.bot.load_config", return_value=cfg), \
                     patch("src.strategy.signal_engine.scan_symbol", return_value=scan):
                    from src.bot import run
                    run()
            conn = sqlite3.connect(pe_module.DB_PATH)
            assert conn.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0] == 0, name
            assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0, name
            conn.close()
        _run_env["notifier"].notify_entry.assert_not_called()

    def test_hole_after_first_fill_holds_everything(self, _run_env):
        """ETH 체결봉 뒤에 빈 봉이 있으면(깊은 프레임도 같음) ETH·BTC 체결 모두 보류, 신규 진입도 보류."""
        import sqlite3
        from datetime import datetime, timedelta, timezone
        from src.paper_trading.paper_engine import PaperEngine, _bar_open
        now = datetime.now(timezone.utc)
        eng = PaperEngine(initial_balance=5000.0, db_path=pe_module.DB_PATH)
        eng.place_pending_limit("ETH/USDT:USDT", "long", 49900.0, 0.05, 49000.0, 52000.0,
                                place_time=now - timedelta(minutes=75))   # 체결봉 floor(now-60)
        eng.place_pending_limit("BTC/USDT:USDT", "long", 49900.0, 0.01, 49000.0, 52000.0,
                                place_time=now - timedelta(minutes=45))   # 첫 대상봉 floor(now-30)
        eng.conn.close()
        frame = _mock_ohlcv()
        hole = _bar_open(now - timedelta(minutes=45), 15)                 # ETH 체결봉 바로 다음 봉 누락
        holey = frame[frame.index != hole]
        _run_env["client"].fetch_ohlcv.return_value = holey
        _run_env["client"].fetch_ohlcv_bybit.side_effect = lambda *a, **k: holey
        _run_env["client"].fetch_ohlcv_history.return_value = holey
        cfg = dict(MOCK_CONFIG)
        cfg["exchange"] = {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "leverage": 5}
        cfg["risk"] = dict(MOCK_CONFIG["risk"], max_positions=4)
        cfg["execution"] = {"mode": "maker"}
        with patch("src.bot.load_config", return_value=cfg), \
             patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch("src.strategy.signal_engine.scan_symbol", return_value=_make_scan(qualified=False)):
            from src.bot import run
            run()
        conn = sqlite3.connect(pe_module.DB_PATH)
        sts = {r[0]: r[1] for r in conn.execute("SELECT symbol, status FROM pending_orders")}
        assert sts == {"ETH/USDT:USDT": "pending", "BTC/USDT:USDT": "pending"}
        assert conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
        conn.close()
        _run_env["notifier"].notify_entry.assert_not_called()
