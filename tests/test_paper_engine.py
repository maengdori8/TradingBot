"""페이퍼 트레이딩 엔진 테스트"""
import sys
sys.path.insert(0, "/home/claude/trading-bot")

import pytest
from unittest.mock import patch
from pathlib import Path
import src.paper_trading.paper_engine as pe_module
from src.paper_trading.paper_engine import PaperEngine, SLIPPAGE, TAKER_FEE


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
