"""실전 전환 판별기 테스트 — 통계 게이트(Wilson) 포함"""
from __future__ import annotations

from unittest.mock import patch


from src.risk.promote_checker import PromoteChecker


PROMOTE_CFG = {
    "min_trades": 50, "min_win_rate": 0.38, "min_profit_factor": 1.5,
    "max_mdd": 0.10, "min_sharpe": 1.0, "min_return_pct": 0.0,
    "require_wilson_gate": True, "breakeven_winrate": 0.286,
}


def _checker(cfg=None):
    with patch.object(PromoteChecker, "_load_promote_config",
                      staticmethod(lambda: cfg or PROMOTE_CFG)):
        return PromoteChecker()


def _perf(n=100, wr=0.45, pf=1.8, mdd=0.05, sharpe=1.5, ret=0.10):
    return {
        "total_trades": n, "win_rate": wr, "profit_factor": pf,
        "mdd": mdd, "sharpe": sharpe, "return_pct": ret, "total_pnl": ret * 1250,
    }


class TestWilsonGate:
    def test_good_large_sample_passes(self):
        """n=200 승률 45% → Wilson 하한 ~38% > 28.6% → 통과."""
        r = _checker().check(_perf(n=200, wr=0.45))
        assert r.criteria["winrate_lb"].passed is True
        assert r.eligible is True

    def test_lucky_small_sample_blocked(self):
        """n=50 승률 38%(점추정 통과)지만 Wilson 하한 < 28.6% → 전환 차단.

        운으로 점추정 기준만 넘은 케이스를 통계 게이트가 막는지 검증.
        """
        r = _checker().check(_perf(n=50, wr=0.38))
        assert r.criteria["승률" != "" and "win_rate"].passed is True   # 점추정은 통과
        assert r.criteria["winrate_lb"].passed is False                 # 신뢰하한 미달
        assert r.eligible is False

    def test_gate_disabled(self):
        """require_wilson_gate=false면 기준 자체가 없음."""
        cfg = {**PROMOTE_CFG, "require_wilson_gate": False}
        r = _checker(cfg).check(_perf(n=50, wr=0.38))
        assert "winrate_lb" not in r.criteria

    def test_min_trades_50(self):
        r = _checker().check(_perf(n=49))
        assert r.criteria["min_trades"].passed is False
        assert r.eligible is False
