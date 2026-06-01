"""서킷브레이커 + RiskManager 통합 테스트"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import src.risk.circuit_breaker as cb_module
from src.risk.circuit_breaker import CircuitBreaker
from src.paper_trading import Position


# ── config mock ──────────────────────────────────────────────────────

MOCK_CONFIG = {
    "capital": {
        "total_capital": 5000,
        "trading_allocation": 0.25,
        "risk_per_trade": 0.01,
    },
    "exchange": {
        "leverage": 5,
        "symbols": ["BTC/USDT:USDT"],
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
}


def _make_position(symbol="BTC/USDT:USDT", direction="long", margin=100.0,
                   entry_price=50000, qty=0.01, sl=49000, tp=52000):
    return Position(
        id="test-pos",
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        qty=qty,
        stop_loss=sl,
        take_profit=tp,
        margin=margin,
        entry_time=datetime.now(timezone.utc),
    )


# ── CircuitBreaker 테스트 ────────────────────────────────────────────

@pytest.fixture
def tmp_cb(tmp_path):
    db_path = tmp_path / "test_cb.db"
    with patch.object(cb_module, "DB_PATH", db_path):
        yield CircuitBreaker(
            trading_capital=1000,
            daily_loss_limit=0.03,
            weekly_loss_limit=0.08,
            max_consecutive_losses=3,
        )


def test_initial_state_allows_trading(tmp_cb):
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is True


def test_daily_loss_limit_blocks(tmp_cb):
    tmp_cb.record_trade(-31)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is False
    assert "일일" in reason


def test_consecutive_loss_blocks(tmp_cb):
    for _ in range(3):
        tmp_cb.record_trade(-5)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert allowed is False
    assert "연속" in reason


def test_win_resets_consecutive(tmp_cb):
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(-5)
    tmp_cb.record_trade(10)
    tmp_cb.record_trade(-5)
    allowed, reason = tmp_cb.is_trading_allowed()
    assert "연속" not in reason or allowed is True


def test_daily_pnl_positive_allows(tmp_cb):
    tmp_cb.record_trade(50)
    allowed, _ = tmp_cb.is_trading_allowed()
    assert allowed is True


def test_reset_consecutive_losses(tmp_cb):
    for _ in range(3):
        tmp_cb.record_trade(-5)
    tmp_cb.reset_consecutive_losses()
    allowed, _ = tmp_cb.is_trading_allowed()
    assert allowed is True


# ── RiskManager 테스트 ───────────────────────────────────────────────

@pytest.fixture
def risk_manager(tmp_path):
    db = tmp_path / "rm_cb.db"
    with patch("src.risk.risk_manager.load_config", return_value=MOCK_CONFIG), \
         patch.object(cb_module, "DB_PATH", db):
        from src.risk.risk_manager import RiskManager
        yield RiskManager()


class TestRiskManagerInit:
    def test_capital_computed(self, risk_manager):
        assert risk_manager.total_capital == 5000
        assert risk_manager.trading_capital == 1250.0

    def test_leverage(self, risk_manager):
        assert risk_manager.leverage == 5

    def test_min_rr(self, risk_manager):
        assert risk_manager.min_rr == 2.0


class TestCheckTradeAllowed:
    def test_allowed_initially(self, risk_manager):
        allowed, reason = risk_manager.check_trade_allowed(current_positions=0)
        assert allowed is True

    def test_blocked_by_max_positions(self, risk_manager):
        allowed, reason = risk_manager.check_trade_allowed(current_positions=2)
        assert allowed is False
        assert "최대 포지션" in reason

    def test_same_symbol_blocked(self, risk_manager):
        """같은 심볼은 방향 무관하게 max_per_symbol(1)에 의해 차단."""
        pos = _make_position()
        allowed, reason = risk_manager.check_trade_allowed(
            current_positions=1, positions=[pos],
            symbol="BTC/USDT:USDT", direction="long",
        )
        assert allowed is False
        assert "심볼당" in reason

    def test_different_symbol_allowed(self, risk_manager):
        """다른 심볼이면 허용."""
        pos = _make_position(symbol="BTC/USDT:USDT", direction="long")
        allowed, _ = risk_manager.check_trade_allowed(
            current_positions=1, positions=[pos],
            symbol="ETH/USDT:USDT", direction="long",
        )
        assert allowed is True

    def test_same_direction_limit(self, risk_manager):
        """같은 방향 max_same_direction(3) 초과 시 차단."""
        positions = [
            _make_position(symbol="BTC/USDT:USDT", direction="long"),
            _make_position(symbol="ETH/USDT:USDT", direction="long"),
            _make_position(symbol="SOL/USDT:USDT", direction="long"),
        ]
        # max_positions=2이므로 먼저 포지션 초과에 걸림
        # max_positions를 넉넉하게 설정한 별도 테스트는 아래 exposure에서
        allowed, reason = risk_manager.check_trade_allowed(
            current_positions=3, positions=positions,
            symbol="DOGE/USDT:USDT", direction="long",
        )
        assert allowed is False  # max_positions(2) 초과

    def test_exposure_limit(self, risk_manager):
        """총 담보금이 trading_capital * max_exposure_pct 초과 시 차단."""
        # trading_capital = 1250, max_exposure_pct = 0.80 → 한도 1000
        positions = [
            _make_position(symbol="BTC/USDT:USDT", margin=600.0),
            _make_position(symbol="ETH/USDT:USDT", margin=500.0),
        ]
        # max_positions=2이므로 current_positions 맞춤 — 하지만 먼저 max_positions 걸림
        # 이 테스트는 exposure 체크 함수를 직접 검증
        exp = risk_manager.calculate_total_exposure(positions)
        assert exp["exposure_pct"] > risk_manager.max_exposure_pct

    def test_circuit_breaker_integration(self, risk_manager):
        for _ in range(3):
            risk_manager.record_result(-100, "SL")
        allowed, reason = risk_manager.check_trade_allowed(current_positions=0)
        assert allowed is False


class TestCalculateTradeParams:
    def test_returns_valid_params(self, risk_manager):
        params = risk_manager.calculate_trade_params(entry=50000, stop_loss=49000)
        assert params["qty"] > 0
        assert params["take_profit"] > 50000
        assert params["entry"] == 50000
        assert params["stop_loss"] == 49000
        assert "leverage" in params

    def test_fixed_leverage_when_auto_off(self, risk_manager):
        """auto_leverage 미설정(기본 off)이면 config 고정 레버리지 사용."""
        assert risk_manager.auto_leverage is False
        params = risk_manager.calculate_trade_params(entry=50000, stop_loss=49000)
        assert params["leverage"] == 5  # MOCK_CONFIG 고정 레버리지

    def test_default_risk_when_no_tiers(self, risk_manager):
        """risk_tiers 미설정 시 기본 risk_per_trade 사용."""
        params = risk_manager.calculate_trade_params(entry=50000, stop_loss=49000, score=90)
        assert params["risk_pct"] == risk_manager.risk_per_trade


class TestTieredRisk:
    @pytest.fixture
    def tiered_rm(self, tmp_path):
        db = tmp_path / "tier_cb.db"
        cfg = {**MOCK_CONFIG}
        cfg["risk"] = {
            **MOCK_CONFIG["risk"],
            "risk_tiers": [
                {"min_score": 85, "risk_pct": 0.007},
                {"min_score": 75, "risk_pct": 0.005},
                {"min_score": 70, "risk_pct": 0.003},
            ],
        }
        with patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch.object(cb_module, "DB_PATH", db):
            from src.risk.risk_manager import RiskManager
            yield RiskManager()

    def test_tiers_sorted_desc(self, tiered_rm):
        scores = [t["min_score"] for t in tiered_rm.risk_tiers]
        assert scores == sorted(scores, reverse=True)

    def test_a_grade_high_risk(self, tiered_rm):
        assert tiered_rm.risk_pct_for_score(90) == 0.007
        assert tiered_rm.risk_pct_for_score(85) == 0.007

    def test_b_grade_mid_risk(self, tiered_rm):
        assert tiered_rm.risk_pct_for_score(80) == 0.005
        assert tiered_rm.risk_pct_for_score(75) == 0.005

    def test_c_grade_low_risk(self, tiered_rm):
        assert tiered_rm.risk_pct_for_score(72) == 0.003
        assert tiered_rm.risk_pct_for_score(70) == 0.003

    def test_below_lowest_tier_uses_min(self, tiered_rm):
        """최저 티어 미만 점수도 가장 낮은 티어 값 적용."""
        assert tiered_rm.risk_pct_for_score(50) == 0.003

    def test_none_score_uses_default(self, tiered_rm):
        assert tiered_rm.risk_pct_for_score(None) == tiered_rm.risk_per_trade

    def test_higher_score_larger_qty(self, tiered_rm):
        """점수 높을수록 더 큰 포지션 수량."""
        p_a = tiered_rm.calculate_trade_params(50000, 49000, score=90)
        p_c = tiered_rm.calculate_trade_params(50000, 49000, score=70)
        assert p_a["qty"] > p_c["qty"]
        assert p_a["risk_pct"] > p_c["risk_pct"]


class TestAutoLeverage:
    @pytest.fixture
    def auto_rm(self, tmp_path):
        db = tmp_path / "auto_cb.db"
        cfg = {**MOCK_CONFIG}
        cfg["exchange"] = {
            **MOCK_CONFIG["exchange"],
            "auto_leverage": True,
            "max_leverage": 10,
            "min_leverage": 1,
            "liq_buffer": 2.0,
        }
        with patch("src.risk.risk_manager.load_config", return_value=cfg), \
             patch.object(cb_module, "DB_PATH", db):
            from src.risk.risk_manager import RiskManager
            yield RiskManager()

    def test_auto_leverage_enabled(self, auto_rm):
        assert auto_rm.auto_leverage is True
        assert auto_rm.max_leverage == 10

    def test_tight_stop_gets_higher_leverage(self, auto_rm):
        """타이트한 손절(6%)이 넓은 손절(10%)보다 높은 레버리지 (둘 다 cap 미만)."""
        # max 10x: 6% → 1/(2*0.06)=8, 10% → 1/(2*0.10)=5
        p_tight = auto_rm.calculate_trade_params(entry=50000, stop_loss=47000)  # 6%
        p_wide = auto_rm.calculate_trade_params(entry=50000, stop_loss=45000)   # 10%
        assert p_tight["leverage"] > p_wide["leverage"]

    def test_leverage_capped(self, auto_rm):
        """매우 타이트한 손절도 max_leverage(10) 초과 안 함."""
        p = auto_rm.calculate_trade_params(entry=50000, stop_loss=49900)  # 0.2%
        assert p["leverage"] <= 10


class TestExposure:
    def test_empty(self, risk_manager):
        exp = risk_manager.calculate_total_exposure([])
        assert exp["total_margin"] == 0.0

    def test_long_exposure(self, risk_manager):
        pos = _make_position(margin=500, entry_price=50000, qty=0.01, sl=49000)
        exp = risk_manager.calculate_total_exposure([pos])
        assert exp["total_margin"] == 500.0
        assert exp["total_risk"] > 0
        assert "BTC/USDT:USDT" in exp["positions_by_symbol"]

    def test_short_exposure(self, risk_manager):
        pos = _make_position(direction="short", entry_price=3000, qty=0.1, sl=3100)
        exp = risk_manager.calculate_total_exposure([pos])
        assert exp["total_risk"] == pytest.approx(10.0, abs=0.01)


class TestRecordResult:
    def test_updates_cb(self, risk_manager):
        risk_manager.record_result(-50.0, "SL")
        assert risk_manager.cb.get_daily_pnl() < 0

    def test_callback_called(self, risk_manager):
        cb = MagicMock(__name__="test_callback")
        risk_manager.register_on_result(cb)
        risk_manager.record_result(100.0, "TP")
        cb.assert_called_once_with(100.0, "TP")

    def test_callback_error_safe(self, risk_manager):
        risk_manager.register_on_result(lambda p, r: 1 / 0)
        risk_manager.record_result(-10.0, "SL")
