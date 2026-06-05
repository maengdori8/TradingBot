"""대시보드 Flask 앱 스모크 테스트 — 라우트 응답 + 헬퍼 함수"""
from __future__ import annotations

import json

import pytest

import src.dashboard.app as dash


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """격리된 임시 DB를 사용하는 Flask 테스트 클라이언트."""
    db = tmp_path / "paper.db"
    cb_db = tmp_path / "cb.db"
    scan = tmp_path / "scan_state.json"
    monkeypatch.setattr(dash, "DB_PATH", db)
    monkeypatch.setattr(dash, "CB_DB_PATH", cb_db)
    # 스캔 상태 로더가 임시 경로를 보도록 패치
    monkeypatch.setattr(
        dash, "load_scan_state",
        lambda path=None: {
            "updated_at": None, "scanned_count": 0,
            "qualified_count": 0, "watchlist": [],
        },
    )
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        yield c


class TestRoutes:
    def test_index_empty_db(self, client):
        """거래 내역 없는 상태에서 메인 페이지 200."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Paper Trading" in resp.data

    def test_api_status_empty(self, client):
        """API가 빈 상태에서도 정상 JSON 반환."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "balance" in data
        assert "performance" in data
        assert data["performance"]["total_trades"] == 0
        assert "scan" in data

    def test_api_live_empty(self, client):
        """실시간 엔드포인트가 포지션 없을 때도 정상 (거래소 호출 안 함)."""
        resp = client.get("/api/live")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["position_count"] == 0
        assert data["open_positions"] == []
        assert data["equity"] == data["balance"]
        assert "timestamp" in data


class TestPerformanceCalc:
    def test_empty_trades(self):
        perf = dash._calc_performance([], 1250.0)
        assert perf["total_trades"] == 0
        assert perf["win_rate"] == 0.0

    def test_with_trades(self):
        trades = [
            {"pnl": 50.0, "pnl_pct": 0.04, "direction": "long"},
            {"pnl": -20.0, "pnl_pct": -0.016, "direction": "short"},
            {"pnl": 30.0, "pnl_pct": 0.024, "direction": "long"},
        ]
        perf = dash._calc_performance(trades, 1250.0)
        assert perf["total_trades"] == 3
        assert perf["win_rate"] == pytest.approx(2 / 3)
        assert perf["total_pnl"] == 60.0
        assert perf["long_count"] == 2
        assert perf["short_count"] == 1
        assert perf["best_trade"] == 50.0
        assert perf["worst_trade"] == -20.0

    def test_equity_curve(self):
        trades = [
            {"pnl": 50.0, "exit_time": "2024-01-01T10:00:00"},
            {"pnl": -20.0, "exit_time": "2024-01-02T10:00:00"},
        ]
        curve = dash._build_equity_curve(trades, 1000.0)
        assert curve["values"][0] == 1000.0
        assert curve["values"][-1] == 1030.0

    def test_equity_curve_empty(self):
        curve = dash._build_equity_curve([], 1000.0)
        assert curve == {"labels": [], "values": []}


class TestPromoteStatus:
    def test_promote_status_structure(self):
        perf = {
            "total_trades": 25, "win_rate": 0.6, "profit_factor": 1.8,
            "mdd": 0.03, "sharpe": 1.2, "total_pnl": 100.0,
        }
        result = dash._promote_status(perf)
        assert "criteria" in result
        assert result["total_count"] == 6
        assert 0 <= result["passed_count"] <= 6
