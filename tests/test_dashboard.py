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


# ------------------------------------------------------------------
# 투 트랙 검증 차트 데이터
# ------------------------------------------------------------------

class TestLoadTrackCurves:
    """_load_track_curves — Track A/B 이력 CSV 로더."""

    def test_파일이_없으면_빈_시리즈를_반환한다(self, tmp_path):
        from src.dashboard.app import _load_track_curves
        t = _load_track_curves(logs_dir=tmp_path)
        assert set(t.keys()) == {"a", "b", "c", "d"}
        for tr in t.values():
            assert tr["labels"] == [] and tr["pct"] == []
            assert tr["equity"] is None

    def test_이력을_누적수익률로_변환한다(self, tmp_path):
        from src.dashboard.app import _load_track_curves
        (tmp_path / "tracka_history.csv").write_text(
            "day,equity,n_pos,events\n"
            "2026-08-24,1.0,0,universe[]\n"
            "2026-08-25,1.01,1,BTC:enter\n", encoding="utf-8")
        t = _load_track_curves(logs_dir=tmp_path)
        assert t["a"]["labels"] == ["2026-08-24", "2026-08-25"]
        assert t["a"]["pct"][0] == 0.0
        assert abs(t["a"]["pct"][1] - 1.0) < 1e-6
        assert t["a"]["n_pos"] == 1

    def test_손상된_행은_전체를_무너뜨리지_않는다(self, tmp_path):
        from src.dashboard.app import _load_track_curves
        (tmp_path / "trackb_history.csv").write_text(
            "day,equity,cash,n_pos,fills\n"
            "2026-08-24,깨진값,0.99,1,x\n", encoding="utf-8")
        t = _load_track_curves(logs_dir=tmp_path)
        assert t["b"]["labels"] == []          # fail-closed, 빈 시리즈


class TestLoadTraderStudy:
    """_load_trader_study — 지속성 연구 데이터 로더."""

    def _cohort(self, tmp_path, wallets):
        import gzip, json
        with gzip.open(tmp_path / "trader_cohort.json.gz", "wt", encoding="utf-8") as f:
            json.dump(dict(locked_at="2026-08-25", n=len(wallets), wallets=wallets), f)

    def _daily(self, tmp_path, day, rows):
        import gzip, csv
        d = tmp_path / "trader_daily"; d.mkdir(exist_ok=True)
        with gzip.open(d / f"{day}.csv.gz", "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["address", "day_pnl"])
            w.writeheader()
            for r in rows: w.writerow(r)

    def test_데이터_없으면_빈_구조(self, tmp_path):
        from src.dashboard.app import _load_trader_study
        t = _load_trader_study(logs_dir=tmp_path)
        assert t["available"] is False and t["labels"] == []

    def test_잠금일_스냅샷은_전방수익에서_제외된다(self, tmp_path):
        from src.dashboard.app import _load_trader_study, _TRADER_CACHE
        _TRADER_CACHE.clear()
        wallets = [dict(address=f"0x{i}", t0_account=10000.0, t0_month_roi=i / 100)
                   for i in range(200)]
        self._cohort(tmp_path, wallets)
        self._daily(tmp_path, "2026-08-25",  # 잠금일 — 제외돼야 함
                    [dict(address="0x0", day_pnl=99999)])
        t = _load_trader_study(logs_dir=tmp_path)
        assert t["available"] and t["days"] == 0

    def test_십분위_곡선이_계산된다(self, tmp_path):
        from src.dashboard.app import _load_trader_study, _TRADER_CACHE
        _TRADER_CACHE.clear()
        wallets = [dict(address=f"0x{i}", t0_account=10000.0, t0_month_roi=i / 100)
                   for i in range(200)]
        self._cohort(tmp_path, wallets)
        # 상위 십분위(월ROI 높은 지갑)가 전방에서도 +100 USD, 하위는 -100 USD
        rows = [dict(address=f"0x{i}", day_pnl=(100 if i >= 180 else (-100 if i < 20 else 0)))
                for i in range(200)]
        self._daily(tmp_path, "2026-08-26", rows)
        t = _load_trader_study(logs_dir=tmp_path)
        assert t["days"] == 1
        assert t["top"][0] > 0 > t["bottom"][0]
        assert abs(t["spread"] - (t["top"][0] - t["bottom"][0])) < 1e-9


class TestBuildSummary:
    """_build_summary — 상단 요약 스트립."""

    def _mk(self, a_pos=0, b_pos=1):
        from src.dashboard.app import _build_summary
        tracks = {"a": dict(pct=[0.0, 0.011], n_pos=a_pos, note="universe[BTC]"),
                  "b": dict(pct=[0.0, -0.189], n_pos=b_pos, note="BTC:enter@79559")}
        study = dict(n=5790, days=1, verdicts=["2026-09-24"])
        return _build_summary(1250.0, [], [], tracks, study)

    def test_한_줄_요약이_상태를_반영한다(self):
        s = self._mk()
        assert "터틀 1포지션 보유" in s["headline"]
        assert "캐리 현금 대기" in s["headline"]
        assert "D-" in s["headline"]

    def test_카드는_4개다(self):
        s = self._mk()
        assert len(s["cards"]) == 4
        assert [c["key"] for c in s["cards"]] == ["ict", "carry", "turtle", "study"]

    def test_트랙_이력이_없어도_뜬다(self):
        from src.dashboard.app import _build_summary
        s = _build_summary(1250.0, [], [], {}, {})
        assert s["cards"][3]["value"] == "-"
        assert "터틀 대기" in s["headline"]
