"""페이퍼 트레이딩 엔진 테스트"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import src.paper_trading.paper_engine as pe_module
from src.paper_trading.paper_engine import PaperEngine, SLIPPAGE, TAKER_FEE
from src.paper_trading import Position


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test_paper.db"
    with patch.object(pe_module, "DB_PATH", db):
        yield PaperEngine(initial_balance=1000.0)


def test_long_profit(engine):
    """Long 포지션 수익 시나리오"""
    pos = engine.open_position("BTC/USDT", "long", entry_price=50000, qty=0.01,
                                stop_loss=49000, take_profit=52000)
    assert pos is not None
    pnl = engine.close_position(pos, exit_price=52000, reason="TP")
    assert pnl > 0


def test_long_loss(engine):
    """Long 포지션 손실 시나리오"""
    pos = engine.open_position("BTC/USDT", "long", entry_price=50000, qty=0.01,
                                stop_loss=49000, take_profit=52000)
    pnl = engine.close_position(pos, exit_price=49000, reason="SL")
    assert pnl < 0


def test_short_profit(engine):
    """Short 포지션 수익 시나리오"""
    pos = engine.open_position("ETH/USDT", "short", entry_price=3000, qty=0.1,
                                stop_loss=3100, take_profit=2800)
    pnl = engine.close_position(pos, exit_price=2800, reason="TP")
    assert pnl > 0


def test_short_loss(engine):
    """Short 포지션 손실 시나리오"""
    pos = engine.open_position("ETH/USDT", "short", entry_price=3000, qty=0.1,
                                stop_loss=3100, take_profit=2800)
    pnl = engine.close_position(pos, exit_price=3100, reason="SL")
    assert pnl < 0


def test_slippage_applied(engine):
    """슬리피지 적용 확인 — 실제 진입가가 요청가보다 불리해야 함"""
    pos = engine.open_position("BTC/USDT", "long", entry_price=50000, qty=0.01,
                                stop_loss=49000, take_profit=52000)
    assert pos.entry_price > 50000  # Long 진입 시 슬리피지로 높아짐


def test_fee_reduces_pnl(engine):
    """수수료가 PnL을 감소시키는지 확인"""
    entry = 50000
    exit_p = 50000  # 변화 없음 → 수수료만큼 손실
    pos = engine.open_position("BTC/USDT", "long", entry_price=entry, qty=0.01,
                                stop_loss=49000, take_profit=52000)
    pnl = engine.close_position(pos, exit_price=exit_p, reason="manual")
    assert pnl < 0  # 수수료 + 슬리피지로 손실


def test_balance_unchanged_after_roundtrip(engine):
    """무손익 트레이드 후 잔고가 거의 유지되는지 (수수료만 차감)"""
    initial = engine.balance
    pos = engine.open_position("BTC/USDT", "long", entry_price=50000, qty=0.001,
                                stop_loss=49000, take_profit=52000)
    engine.close_position(pos, exit_price=50000, reason="manual")
    # 수수료만큼 감소
    assert engine.balance < initial
    assert engine.balance > initial * 0.99  # 1% 이상 감소 없어야 함


def test_performance_metrics(engine):
    """성과 지표 계산 확인"""
    for _ in range(3):
        pos = engine.open_position("BTC/USDT", "long", 50000, 0.01, 49000, 52000)
        engine.close_position(pos, 52000, "TP")
    pos = engine.open_position("BTC/USDT", "long", 50000, 0.01, 49000, 52000)
    engine.close_position(pos, 49000, "SL")

    perf = engine.get_performance()
    assert perf["total_trades"] == 4
    assert 0 <= perf["win_rate"] <= 1
    assert "mdd" in perf
    assert "profit_factor" in perf


# ─── 부분 청산 ────────────────────────────────────────────────────────

def test_partial_close(tmp_path):
    """부분 청산 — qty 파라미터로 절반만 청산."""
    db = tmp_path / "partial_close.db"
    with patch.object(pe_module, "DB_PATH", db):
        engine = PaperEngine(initial_balance=50000.0)
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.02,
        stop_loss=49000, take_profit=52000,
    )
    assert pos is not None
    original_qty = pos.qty

    pnl = engine.close_position(pos, exit_price=51000, reason="partial", qty=0.01)

    # 포지션이 여전히 남아 있음
    assert pos in engine.positions
    assert pos.qty == pytest.approx(original_qty - 0.01, abs=1e-6)
    assert pos.margin > 0

    # 나머지 전량 청산
    pnl2 = engine.close_position(pos, exit_price=52000, reason="TP")
    assert pos not in engine.positions


def test_partial_close_exceeding_qty(engine):
    """부분 청산 수량이 보유 수량을 초과하면 전량 청산."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.01,
        stop_loss=49000, take_profit=52000,
    )
    pnl = engine.close_position(pos, exit_price=51000, reason="manual", qty=0.05)
    # 초과 수량이므로 전량 청산됨
    assert pos not in engine.positions


# ─── 트레일링 스톱 ────────────────────────────────────────────────────

def test_trailing_stop_long(engine):
    """Long 트레일링 스톱 — 가격 상승 시 SL이 올라감."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.01,
        stop_loss=49000, take_profit=55000,
    )
    original_sl = pos.stop_loss

    # 가격이 진입가 위로 올라간 상황에서 트레일링 스톱 갱신
    engine.update_trailing_stop("BTC/USDT", current_price=52000, trail_pct=0.02)

    assert pos.stop_loss > original_sl
    expected_new_sl = round(52000 * (1 - 0.02), 8)
    assert pos.stop_loss == expected_new_sl


def test_trailing_stop_long_no_update_below_entry(engine):
    """Long 트레일링 스톱 — 가격이 진입가 아래면 SL 변경 안 함."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.01,
        stop_loss=49000, take_profit=55000,
    )
    original_sl = pos.stop_loss

    engine.update_trailing_stop("BTC/USDT", current_price=49500, trail_pct=0.02)
    assert pos.stop_loss == original_sl


def test_trailing_stop_short(engine):
    """Short 트레일링 스톱 — 가격 하락 시 SL이 내려감."""
    pos = engine.open_position(
        "ETH/USDT", "short", entry_price=3000, qty=0.1,
        stop_loss=3100, take_profit=2800,
    )
    original_sl = pos.stop_loss

    engine.update_trailing_stop("ETH/USDT", current_price=2900, trail_pct=0.02)

    assert pos.stop_loss < original_sl
    expected_new_sl = round(2900 * (1 + 0.02), 8)
    assert pos.stop_loss == expected_new_sl


# ─── 미실현 손익 ─────────────────────────────────────────────────────

def test_update_unrealized_pnl_long(engine):
    """미실현 손익 — Long 포지션."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.01,
        stop_loss=49000, take_profit=52000,
    )
    assert pos.unrealized_pnl == 0.0

    engine.update_unrealized_pnl("BTC/USDT", current_price=51000)

    # (51000 - entry) * qty, entry는 슬리피지 적용 후
    expected = round((51000 - pos.entry_price) * pos.qty, 8)
    assert pos.unrealized_pnl == expected


def test_update_unrealized_pnl_short(engine):
    """미실현 손익 — Short 포지션."""
    pos = engine.open_position(
        "ETH/USDT", "short", entry_price=3000, qty=0.1,
        stop_loss=3100, take_profit=2800,
    )
    engine.update_unrealized_pnl("ETH/USDT", current_price=2900)

    expected = round((pos.entry_price - 2900) * pos.qty, 8)
    assert pos.unrealized_pnl == expected
    assert pos.unrealized_pnl > 0  # 유리한 방향


# ─── 포지션 복원 (DB 영속화) ──────────────────────────────────────────

def test_position_restore_from_db(tmp_path):
    """open -> DB 저장 -> 새 엔진 인스턴스 -> 포지션 존재 확인."""
    db = tmp_path / "restore_test.db"

    with patch.object(pe_module, "DB_PATH", db):
        engine1 = PaperEngine(initial_balance=10000.0)
        pos = engine1.open_position(
            "BTC/USDT", "long", entry_price=50000, qty=0.01,
            stop_loss=49000, take_profit=52000,
        )
        assert pos is not None
        pos_id = pos.id

    # 새 엔진 인스턴스 — DB에서 복원
    with patch.object(pe_module, "DB_PATH", db):
        engine2 = PaperEngine(initial_balance=10000.0)
        restored = engine2.get_positions()

        assert len(restored) == 1
        assert restored[0].id == pos_id
        assert restored[0].symbol == "BTC/USDT"
        assert restored[0].direction == "long"
        assert restored[0].entry_price == pos.entry_price


# ─── 잔고 부족 ───────────────────────────────────────────────────────

def test_insufficient_balance_returns_none(engine):
    """잔고 부족 시 open_position이 None 반환."""
    # 엔진 잔고가 1000인데, 매우 큰 포지션 시도
    result = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=1.0,
        stop_loss=49000, take_profit=52000,
    )
    assert result is None
    # 잔고가 변하지 않아야 함
    assert engine.balance == 1000.0


# ─── 콜백 등록 및 호출 ───────────────────────────────────────────────

def test_register_and_fire_callback(engine):
    """콜백 등록 후 청산 시 호출."""
    callback = MagicMock(__name__="test_callback")
    engine.register_on_trade(callback)

    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.001,
        stop_loss=49000, take_profit=52000,
    )
    pnl = engine.close_position(pos, exit_price=52000, reason="TP")

    callback.assert_called_once()
    args = callback.call_args[0]
    assert args[0] == pnl  # pnl
    assert args[1] == "TP"  # reason
    assert isinstance(args[2], Position)  # position


def test_callback_error_does_not_crash(engine):
    """콜백에서 에러가 발생해도 엔진은 정상 동작."""
    def bad_callback(pnl, reason, pos):
        raise RuntimeError("callback error!")

    engine.register_on_trade(bad_callback)

    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.001,
        stop_loss=49000, take_profit=52000,
    )
    # 에러가 전파되지 않아야 함
    pnl = engine.close_position(pos, exit_price=52000, reason="TP")
    assert pnl > 0


def test_multiple_callbacks_all_called(engine):
    """여러 콜백 등록 시 모두 호출."""
    cb1 = MagicMock(__name__="cb1")
    cb2 = MagicMock(__name__="cb2")
    engine.register_on_trade(cb1)
    engine.register_on_trade(cb2)

    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.001,
        stop_loss=49000, take_profit=52000,
    )
    engine.close_position(pos, exit_price=52000, reason="TP")

    cb1.assert_called_once()
    cb2.assert_called_once()


# ─── check_stops 테스트 ──────────────────────────────────────────────

def test_check_stops_long_sl(engine):
    """Long 포지션 SL 자동 트리거."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.001,
        stop_loss=49000, take_profit=52000,
    )
    assert pos is not None

    # 저가가 SL 이하로 내려감
    engine.check_stops("BTC/USDT", current_high=50500, current_low=48500)

    assert pos not in engine.positions


def test_check_stops_long_tp(engine):
    """Long 포지션 TP 자동 트리거."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.001,
        stop_loss=49000, take_profit=52000,
    )
    # 고가가 TP 이상으로 올라감
    engine.check_stops("BTC/USDT", current_high=53000, current_low=51000)

    assert pos not in engine.positions


def test_check_stops_ignores_other_symbol(engine):
    """다른 심볼의 포지션은 무시."""
    pos = engine.open_position(
        "BTC/USDT", "long", entry_price=50000, qty=0.001,
        stop_loss=49000, take_profit=52000,
    )
    engine.check_stops("ETH/USDT", current_high=53000, current_low=48000)

    assert pos in engine.positions  # 심볼이 달라서 그대로
