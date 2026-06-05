"""자동 파라미터 튜너(learner) 테스트 — 통계/의사결정/오버레이/병합/안전성"""
from __future__ import annotations

import sqlite3

import pytest

import src.config_loader as cl
import src.risk.learner as L
from src.paper_trading.paper_engine import _init_db


# ── 공통 fixtures ───────────────────────────────────────────────────

def _cfg():
    return {
        "scan": {"min_score": 70},
        "risk": {
            "min_rr_ratio": 2.0,
            "risk_tiers": [
                {"min_score": 85, "risk_pct": 0.007},
                {"min_score": 75, "risk_pct": 0.005},
                {"min_score": 70, "risk_pct": 0.003},
            ],
        },
        "learning": {
            "enabled": True, "dry_run": False, "min_trades_to_learn": 20,
            "every_n_trades": 5, "cooldown_trades": 10, "lookback_trades": 0,
            "deadband_R": 0.05, "bucket_min_n": 12, "session_min_n": 10,
            "global_min_n_for_min_score": 20,
        },
        "promote": {"max_mdd": 0.05},
    }


def _baseline():
    return {
        "scan.min_score": 70,
        "risk.min_rr_ratio": 2.0,
        "risk.risk_tiers": [
            {"min_score": 85, "risk_pct": 0.007},
            {"min_score": 75, "risk_pct": 0.005},
            {"min_score": 70, "risk_pct": 0.003},
        ],
    }


def _mk(score, r, direction="long", session="london"):
    return {
        "id": f"t{id(object())}", "direction": direction,
        "entry_score": score, "entry_session": session,
        "status": "TP" if r > 0 else "SL", "entry_rr": 2.0,
        "risk_amount": 10.0, "r_multiple": float(r), "pnl": r * 10,
        "exit_time": "2024-01-01T00:00:00",
    }


BOUNDS = [85, 75, 70]


# ── 통계 헬퍼 ────────────────────────────────────────────────────────

class TestStats:
    def test_laplace_bounds(self):
        assert L.laplace_winrate(0, 0) == 0.5
        assert 0 < L.laplace_winrate(0, 10) < 0.5
        assert 0.5 < L.laplace_winrate(10, 10) < 1.0

    def test_wilson_widens_with_small_n(self):
        lb1, ub1 = L.wilson_interval(5, 10)
        lb2, ub2 = L.wilson_interval(50, 100)
        assert (ub1 - lb1) > (ub2 - lb2)  # 표본 작을수록 구간 넓음

    def test_wilson_in_range(self):
        for w, n in [(0, 0), (0, 5), (5, 5), (3, 10)]:
            lb, ub = L.wilson_interval(w, n)
            assert 0.0 <= lb <= ub <= 1.0

    def test_expectancy_none_without_r(self):
        assert L.expectancy_r([{"r_multiple": None}]) is None
        assert L.expectancy_r([{"r_multiple": 1.0}, {"r_multiple": -0.5}]) == 0.25


# ── 의사결정: 위험 축소 ──────────────────────────────────────────────

class TestReduceRisk:
    def test_losing_bucket_reduces_risk_pct(self):
        trades = [_mk(72, -1) for _ in range(15)]   # C버킷(>=70) 전패
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec is not None
        assert dec["kind"] == "risk_pct"
        assert dec["tier_min_score"] == 70
        assert dec["new"] < dec["old"]

    def test_realistic_losing_bucket_reduces(self):
        """현실적 손실 분포(25% 승률, 기대값 -0.25R, n=12)도 축소 발동 (리뷰 #2)."""
        trades = [_mk(72, 2) for _ in range(3)] + [_mk(72, -1) for _ in range(9)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec is not None
        assert dec["kind"] == "risk_pct"
        assert dec["new"] < dec["old"]

    def test_noisy_winning_bucket_not_expanded(self):
        """기대값 양수지만 신뢰하한<0(노이즈)이면 확대 안 함 (안전=어렵게)."""
        # 큰 변동: 6승(+2) 6패(-1.9) → 평균 +0.05, 분산 큼 → lb<0
        trades = [_mk(90, 2) for _ in range(6)] + [_mk(90, -1.9) for _ in range(6)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        # 확대(risk_pct 상향) 발동 안 됨
        assert dec is None or not (dec["kind"] == "risk_pct" and dec["new"] > dec["old"])

    def test_min_score_raised_when_risk_at_floor(self):
        cfg = _cfg()
        cfg["risk"]["risk_tiers"][2]["risk_pct"] = 0.002  # C버킷 바닥
        trades = [_mk(72, -1) for _ in range(20)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, cfg, _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec is not None
        assert dec["kind"] == "min_score"
        assert dec["new"] == 72   # 70 + 2

    def test_reduction_priority_over_expansion(self):
        # C버킷 전패 + A버킷 전승 동시 → 축소가 먼저
        trades = [_mk(72, -1) for _ in range(15)] + [_mk(90, 2) for _ in range(15)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec["kind"] == "risk_pct"
        assert dec["tier_min_score"] == 70   # 축소 대상(약한 버킷)
        assert dec["new"] < dec["old"]


# ── 의사결정: 위험 확대 + 게이트 ─────────────────────────────────────

class TestExpandAndGates:
    def test_winning_bucket_expands_risk(self):
        # A버킷 전승(n>=12) + B버킷 소량 손실(n<12, baseline만 낮춤)
        trades = [_mk(90, 2) for _ in range(15)] + [_mk(80, -1) for _ in range(8)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec is not None
        assert dec["kind"] == "risk_pct"
        assert dec["tier_min_score"] == 85
        assert dec["new"] > dec["old"]

    def test_insufficient_sample_no_op(self):
        trades = [_mk(72, -1) for _ in range(8)]   # n<12
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec is None

    def test_cooldown_blocks(self):
        trades = [_mk(72, -1) for _ in range(15)]
        agg = L.aggregate(trades, BOUNDS)
        # 직전 변경이 5건 전 (쿨다운 10 미충족)
        state = {"last_change_at": {"risk_tiers[70].risk_pct": 15 - 5}, "kill_switch": False}
        dec = L.decide_adjustment(agg, _cfg(), _baseline(), state)
        assert dec is None

    def test_kill_switch_freezes_expansion(self):
        trades = [_mk(90, 2) for _ in range(15)] + [_mk(80, -1) for _ in range(8)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, _cfg(), _baseline(),
                                  {"last_change_at": {}, "kill_switch": True})
        assert dec is None   # 확대 동결

    def test_risk_pct_clamped_to_floor(self):
        cfg = _cfg()
        cfg["risk"]["risk_tiers"][2]["risk_pct"] = 0.0021
        trades = [_mk(72, -1) for _ in range(15)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, cfg, _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        assert dec["new"] == L.RISK_PCT_MIN   # 0.002로 클램프

    def test_monotonicity_blocks_bad_expansion(self):
        # C버킷이 A버킷보다 risk_pct 높아지는 확대는 거부
        cfg = _cfg()
        cfg["risk"]["risk_tiers"] = [
            {"min_score": 85, "risk_pct": 0.003},
            {"min_score": 75, "risk_pct": 0.003},
            {"min_score": 70, "risk_pct": 0.003},
        ]
        # 70버킷만 강하게 이김 → 확대하면 70>75 단조성 위반
        trades = [_mk(72, 2) for _ in range(15)] + [_mk(90, -0.5) for _ in range(8)]
        agg = L.aggregate(trades, BOUNDS)
        dec = L.decide_adjustment(agg, cfg, _baseline(),
                                  {"last_change_at": {}, "kill_switch": False})
        # 70 확대는 단조성 위반으로 거부됨 (다른 후보 없으면 None)
        if dec is not None:
            assert not (dec["kind"] == "risk_pct" and dec["tier_min_score"] == 70 and dec["new"] > dec["old"])


# ── 오버레이 검증 ────────────────────────────────────────────────────

class TestOverlayValidation:
    def test_validate_rejects_out_of_range(self):
        assert L._validate_overlay({"values": {"scan": {"min_score": 200}}}) is False
        assert L._validate_overlay({"values": {"risk": {"min_rr_ratio": 9.0}}}) is False
        assert L._validate_overlay({"values": {"risk": {"risk_tiers": [{"min_score": 70, "risk_pct": 0.5}]}}}) is False

    def test_validate_accepts_in_range(self):
        assert L._validate_overlay({"values": {"scan": {"min_score": 72}}}) is True

    def test_write_overlay_atomic(self, tmp_path, monkeypatch):
        ov = tmp_path / "learned_params.yaml"
        monkeypatch.setattr(L, "OVERLAY_PATH", ov)
        raw = {"scan": {"min_score": 70}, "risk": {"min_rr_ratio": 2.0,
               "risk_tiers": [{"min_score": 70, "risk_pct": 0.003}]}}
        adj = {"kind": "min_score", "param": "scan.min_score", "old": 70, "new": 72,
               "reason": "test", "segment": {}}
        ok = L.write_overlay(adj, raw, total_n=25)
        assert ok
        assert ov.exists()
        import yaml
        data = yaml.safe_load(ov.read_text())
        assert data["values"]["scan"]["min_score"] == 72
        assert data["baseline"]["scan.min_score"] == 70


# ── config_loader 병합/화이트리스트/드리프트 ─────────────────────────

class TestConfigLoaderMerge:
    def test_overlay_applied(self):
        cfg = {"scan": {"min_score": 70}, "risk": {"min_rr_ratio": 2.0, "risk_tiers": [{"min_score": 70, "risk_pct": 0.003}]}}
        overlay = {"baseline": {"scan.min_score": 70, "risk.min_rr_ratio": 2.0,
                   "risk.risk_tiers": [{"min_score": 70, "risk_pct": 0.003}]},
                   "values": {"scan": {"min_score": 74},
                              "risk": {"risk_tiers": [{"min_score": 70, "risk_pct": 0.0025}]}}}
        cl._apply_overlay(cfg, overlay)
        assert cfg["scan"]["min_score"] == 74
        assert cfg["risk"]["risk_tiers"][0]["risk_pct"] == 0.0025

    def test_baseline_drift_invalidates(self):
        cfg = {"scan": {"min_score": 72}}  # 사용자가 70→72로 손수정
        overlay = {"baseline": {"scan.min_score": 70},
                   "values": {"scan": {"min_score": 80}}}
        cl._apply_overlay(cfg, overlay)
        assert cfg["scan"]["min_score"] == 72   # 오버레이 무효화, 손수정값 유지

    def test_whitelist_blocks_arbitrary_keys(self):
        cfg = {"scan": {"min_score": 70}, "risk": {"min_rr_ratio": 2.0}}
        overlay = {"baseline": {}, "values": {"capital": {"total_capital": 999999},
                   "exchange": {"leverage": 100}}}
        cl._apply_overlay(cfg, overlay)
        assert "capital" not in cfg or cfg.get("capital", {}).get("total_capital") != 999999
        assert "exchange" not in cfg

    def test_out_of_range_min_score_rejected(self):
        """손상/손편집된 오버레이의 범위 밖 min_score는 적용 거부 (리뷰 #3)."""
        cfg = {"scan": {"min_score": 70}}
        overlay = {"baseline": {}, "values": {"scan": {"min_score": 5}}}
        cl._apply_overlay(cfg, overlay)
        assert cfg["scan"]["min_score"] == 70   # 거부, 원값 유지

    def test_out_of_range_risk_pct_rejected(self):
        """범위 밖 risk_pct(0.9 = 90%!)는 적용 거부 — 라이브 주입 차단."""
        cfg = {"risk": {"risk_tiers": [{"min_score": 70, "risk_pct": 0.003}]}}
        overlay = {"baseline": {}, "values": {"risk": {"risk_tiers": [{"min_score": 70, "risk_pct": 0.9}]}}}
        cl._apply_overlay(cfg, overlay)
        assert cfg["risk"]["risk_tiers"][0]["risk_pct"] == 0.003   # 거부

    def test_out_of_range_min_rr_rejected(self):
        cfg = {"risk": {"min_rr_ratio": 2.0}}
        overlay = {"baseline": {}, "values": {"risk": {"min_rr_ratio": 50.0}}}
        cl._apply_overlay(cfg, overlay)
        assert cfg["risk"]["min_rr_ratio"] == 2.0   # 거부

    def test_in_range_values_applied(self):
        """정상 범위 값은 적용됨."""
        cfg = {"scan": {"min_score": 70}, "risk": {"min_rr_ratio": 2.0,
               "risk_tiers": [{"min_score": 70, "risk_pct": 0.003}]}}
        overlay = {"baseline": {}, "values": {"scan": {"min_score": 73},
                   "risk": {"min_rr_ratio": 2.25, "risk_tiers": [{"min_score": 70, "risk_pct": 0.0025}]}}}
        cl._apply_overlay(cfg, overlay)
        assert cfg["scan"]["min_score"] == 73
        assert cfg["risk"]["min_rr_ratio"] == 2.25
        assert cfg["risk"]["risk_tiers"][0]["risk_pct"] == 0.0025


# ── maybe_update 통합 (룩어헤드/dry-run/적용) ────────────────────────

def _seed_db(path, trades):
    conn = _init_db(path)
    for i, t in enumerate(trades):
        conn.execute(
            """INSERT INTO trades (id, symbol, direction, entry_price, exit_price,
               qty, pnl, pnl_pct, margin, entry_time, exit_time, status,
               entry_score, entry_session, c_trend, c_zone, c_kill_zone, c_ote,
               c_volume, c_rr, entry_rr, risk_amount, r_multiple)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"id{i}", "BTC/USDT:USDT", t["direction"], 100, 100, 1.0, t["pnl"], 0.0,
             10.0, "2024-01-01T00:00:00", t["exit_time"], t["status"],
             t["entry_score"], t["entry_session"], 1, 1, 1, 1, 1, 1,
             t["entry_rr"], t["risk_amount"], t["r_multiple"]),
        )
    conn.commit()
    return conn


class TestMaybeUpdate:
    def _patch(self, monkeypatch, tmp_path, cfg=None):
        cfg = cfg or _cfg()
        monkeypatch.setattr(L, "load_config", lambda *a, **k: cfg)
        monkeypatch.setattr(L, "load_raw_config", lambda: _baseline_cfg())
        monkeypatch.setattr(L, "OVERLAY_PATH", tmp_path / "learned_params.yaml")
        monkeypatch.setattr(L, "REPORT_PATH", tmp_path / "report.json")
        monkeypatch.setattr(L, "HISTORY_PATH", tmp_path / "hist.jsonl")

    def test_insufficient_no_op(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        conn = _seed_db(tmp_path / "p.db", [_mk(72, -1) for _ in range(10)])
        res = L.maybe_update(conn=conn)
        assert res["status"] == "insufficient"

    def test_dry_run_no_write(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        conn = _seed_db(tmp_path / "p.db", [_mk(72, -1) for _ in range(20)])
        res = L.maybe_update(conn=conn, dry_run=True)
        assert res["decision"] is not None
        assert res["applied"] is False
        assert not (tmp_path / "learned_params.yaml").exists()

    def test_apply_writes_overlay(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        conn = _seed_db(tmp_path / "p.db", [_mk(72, -1) for _ in range(20)])
        res = L.maybe_update(conn=conn, dry_run=False)
        assert res["applied"] is True
        ov = tmp_path / "learned_params.yaml"
        assert ov.exists()
        import yaml
        data = yaml.safe_load(ov.read_text())
        # C버킷 risk_pct가 낮아짐
        tiers = {t["min_score"]: t["risk_pct"] for t in data["values"]["risk"]["risk_tiers"]}
        assert tiers[70] < 0.003

    def test_no_lookahead_open_positions_excluded(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        conn = _seed_db(tmp_path / "p.db", [_mk(72, -1) for _ in range(20)])
        # 미청산 포지션 추가 → 학습에 절대 포함 안 됨
        conn.execute(
            """INSERT INTO open_positions (id, symbol, direction, entry_price, qty,
               stop_loss, take_profit, margin, entry_time)
               VALUES ('open1','ETH/USDT:USDT','long',100,1,99,104,100,'2024-01-01T00:00:00')""")
        conn.commit()
        trades = L.fetch_closed_trades(conn)
        assert len(trades) == 20   # open_positions 미포함
        assert all(t["id"] != "open1" for t in trades)


def _baseline_cfg():
    return {
        "scan": {"min_score": 70},
        "risk": {"min_rr_ratio": 2.0, "risk_tiers": [
            {"min_score": 85, "risk_pct": 0.007},
            {"min_score": 75, "risk_pct": 0.005},
            {"min_score": 70, "risk_pct": 0.003}]},
    }
