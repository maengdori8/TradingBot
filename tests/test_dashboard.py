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
    # 실시간 가격 조회 차단 (네트워크 금지 — 폴백 경로로만 평가)
    monkeypatch.setattr(dash, "_live_price", lambda s: None)
    # 코인 차트: 캐시 격리 + 캔들 조회 차단 (네트워크 금지 — 테스트별 목 주입)
    monkeypatch.setattr(dash, "_chart_cache", {})
    monkeypatch.setattr(dash, "_chart_fetch_1h", lambda s: None)
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        yield c


class TestRoutes:
    def test_index_empty_db(self, client):
        """거래 내역 없는 상태에서 메인 페이지 200."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Paper Trading" in resp.data

    def test_index_h2_card_renders(self, client):
        """H2 연구 상태 카드가 메인 페이지에 표시된다."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "꾸준함 가설".encode() in resp.data
        assert "자본 권한 없음".encode() in resp.data

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
        import gzip
        import json
        with gzip.open(tmp_path / "trader_cohort.json.gz", "wt", encoding="utf-8") as f:
            json.dump(dict(locked_at="2026-08-25", n=len(wallets), wallets=wallets), f)

    def _daily(self, tmp_path, day, rows):
        import gzip
        import csv
        d = tmp_path / "trader_daily"
        d.mkdir(exist_ok=True)
        with gzip.open(d / f"{day}.csv.gz", "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["address", "day_pnl"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

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


class TestLoadH2Study:
    """_load_h2_study — H2 꾸준함 가설 연구 상태 카드 로더."""

    def _cohort(self, tmp_path, n=136, mde=0.2429):
        import gzip
        import json
        with gzip.open(tmp_path / "h2_cohort.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"header": {"counts": {"eligible_primary": n},
                                  "mde": {"n_primary": n, "ic": mde}},
                       "wallets": []}, f)

    def _snapshot(self, tmp_path, day="2026-08-27", rows=3):
        import gzip
        import json
        d = tmp_path / "h2_snapshots"
        d.mkdir(exist_ok=True)
        with gzip.open(d / f"{day}.jsonl.gz", "wt", encoding="utf-8") as f:
            for i in range(rows):
                f.write(json.dumps({"address": f"0x{i}"}) + "\n")

    def _fills(self, tmp_path):
        import json
        (tmp_path / "h2_fills_state.json").write_text(json.dumps({
            "high_turnover": [],
            "wallets": {
                "0x1": {"status": "ok"},
                "0x2": {"status": "fill-history-censored"},
                "0x3": {"status": "ok", "initial_window_truncated": True},
            }}), encoding="utf-8")

    def _gate(self, tmp_path):
        import json
        (tmp_path / "h2_trackb_gate.json").write_text(json.dumps({
            "stage2_eligible": False,
            "entries": [{"judgment_date": "2026-12-26", "horizon_days": 30,
                         "ic": 0.31, "p": 0.004, "passed": True,
                         "indeterminate": False}]}), encoding="utf-8")

    def test_파일이_모두_있으면_전_항목이_채워진다(self, tmp_path):
        from datetime import date
        from src.dashboard.app import _load_h2_study
        self._cohort(tmp_path)
        self._snapshot(tmp_path)
        self._fills(tmp_path)
        self._gate(tmp_path)
        h2 = _load_h2_study(logs_dir=tmp_path, today=date(2026, 8, 27))
        assert h2["cohort"]["n"] == 136
        assert h2["cohort"]["mde_ic"] == pytest.approx(0.2429)
        assert h2["snapshot"] == {"day": "2026-08-27", "rows": 3}
        assert h2["fills"] == {"tracked": 3, "censored": 1, "truncated": 1}
        assert h2["gate"]["n_entries"] == 1
        assert h2["gate"]["stage2_eligible"] is False
        assert "IC +0.310" in h2["gate"]["verdict"]
        assert "통과" in h2["gate"]["verdict"]
        # 카운트다운: 고정 일정표에서 미래 최근접 2개 (H1 T+30 → 트랙A T+30)
        assert h2["upcoming"][0] == {"day": "2026-09-24", "label": "H1 T+30", "dday": 28}
        assert h2["upcoming"][1]["day"] == "2026-09-26"
        assert h2["upcoming"][1]["dday"] == 30

    def test_파일이_전부_없어도_크래시_없이_대기_구조(self, tmp_path):
        from datetime import date
        from src.dashboard.app import _load_h2_study
        h2 = _load_h2_study(logs_dir=tmp_path, today=date(2026, 8, 27))
        assert h2["cohort"] is None
        assert h2["snapshot"] is None
        assert h2["fills"] is None
        assert h2["gate"] is None                     # 카드에서 "판정 전"
        assert len(h2["upcoming"]) == 2               # 일정은 파일과 무관하게 고정

    def test_일부만_있으면_있는_항목만_채운다(self, tmp_path):
        from datetime import date
        from src.dashboard.app import _load_h2_study
        self._cohort(tmp_path)
        self._snapshot(tmp_path, day="2026-08-26", rows=2)
        h2 = _load_h2_study(logs_dir=tmp_path, today=date(2026, 8, 27))
        assert h2["cohort"]["n"] == 136
        assert h2["snapshot"] == {"day": "2026-08-26", "rows": 2}
        assert h2["fills"] is None
        assert h2["gate"] is None

    def test_일정이_모두_지나면_카운트다운은_빈_목록(self, tmp_path):
        from datetime import date
        from src.dashboard.app import _load_h2_study
        h2 = _load_h2_study(logs_dir=tmp_path, today=date(2027, 3, 1))
        assert h2["upcoming"] == []


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
        assert "터틀" in s["headline"]
        assert "캐리 현금 대기" in s["headline"]
        assert "D-" in s["headline"]

    def test_카드는_6개다(self):
        s = self._mk()
        assert len(s["cards"]) == 6
        assert [c["key"] for c in s["cards"]] == ["ict", "carry", "turtle", "xvenue", "swing", "study"]

    def test_트랙_이력이_없어도_뜬다(self):
        from src.dashboard.app import _build_summary
        s = _build_summary(1250.0, [], [], {}, {})
        assert s["cards"][5]["value"] == "-"          # study 카드
        assert "캐리 현금 대기" in s["headline"]


class TestTracksLive:
    """_tracks_live — 초단위 트랙 시가평가."""

    def test_상태파일이_없으면_빈_딕셔너리(self, tmp_path):
        from src.dashboard.app import _tracks_live
        assert _tracks_live(logs_dir=tmp_path) == {}

    def test_포지션_미실현이_시가평가에_반영된다(self, tmp_path, monkeypatch):
        import json
        import src.dashboard.app as dash
        (tmp_path / "trackb_state.json").write_text(json.dumps(dict(
            equity=1.0, positions={"BTC": dict(direction=1, units=0.00001,
                                               entry=100000.0, stop=90000.0)})))
        monkeypatch.setattr(dash, "_live_price", lambda s: 110000.0)
        t = dash._tracks_live(logs_dir=tmp_path)
        assert t["b"]["pct"] == pytest.approx(10.0)          # 0.00001×10000 = +0.1 → +10%
        assert t["b"]["positions"][0]["direction"] == "long"

    def test_가격조회_실패시_진입가로_평가한다(self, tmp_path, monkeypatch):
        import json
        import src.dashboard.app as dash
        (tmp_path / "trackd_state.json").write_text(json.dumps(dict(
            equity=0.98, positions={"SOL": dict(d=1, u=0.001, e=100.0, stop=90.0)})))
        monkeypatch.setattr(dash, "_live_price", lambda s: None)
        t = dash._tracks_live(logs_dir=tmp_path)
        assert t["d"]["pct"] == pytest.approx(-2.0)


# ------------------------------------------------------------------
# Track C 표기 정직성 — 명목/ROE 병기 (÷2 표기만으로는 오독)
# ------------------------------------------------------------------

class TestTrackCNotation:
    """Track C 는 명목(양다리 차익) 기준이라 ROE(÷2)를 반드시 병기한다."""

    CAVEAT = "2024-12 인샘플 청산 발견 — 과거 수치는 상한(정정 공시 참조)"

    def _curves(self, tmp_path):
        (tmp_path / "trackc_history.csv").write_text(
            "day,equity,day_diff,n_coins\n"
            "2026-08-25,1.0,0.00010,37\n"
            "2026-08-26,1.02,0.00013,31\n", encoding="utf-8")
        return dash._load_track_curves(logs_dir=tmp_path)

    def test_이력_로더가_명목과_ROE를_함께_싣는다(self, tmp_path):
        c = self._curves(tmp_path)["c"]
        assert c["pct"] == pytest.approx([0.0, 2.0])          # 명목 (equity 기준)
        assert c["roe_pct"] == pytest.approx([0.0, 1.0])      # 담보 이중 → ÷2
        assert "명목" in c["basis"] and "ROE" in c["roe_label"]
        assert c["caveat"] == self.CAVEAT

    def test_라벨이_어떤_수치인지_말한다(self, tmp_path):
        c = self._curves(tmp_path)["c"]
        assert "명목" in c["label"] and "ROE" in c["label"]
        assert "÷2" in c["roe_label"]                # 변환식이 라벨에 남는다

    def test_다른_트랙은_자본_기준이며_ROE_시리즈가_없다(self, tmp_path):
        t = self._curves(tmp_path)
        for k in ("a", "b", "d"):
            assert t[k]["basis"] == "자본"
            assert t[k]["roe_pct"] is None
            assert t[k]["roe_label"] is None and t[k]["caveat"] is None

    def test_요약_카드가_명목과_ROE를_모두_담는다(self, tmp_path):
        tracks = self._curves(tmp_path)
        s = dash._build_summary(1250.0, [], [], tracks, {})
        card = next(c for c in s["cards"] if c["key"] == "xvenue")
        assert card["value"] == "+2.000%"             # 큰 값 = 명목
        assert card["unit"] == "명목"                  # 값 옆에 기준 표기
        assert card["roe"] == "+1.000%"               # 병기된 ROE
        assert "ROE" in card["roe_label"] and "÷2" in card["roe_label"]
        assert card["caveat"] == self.CAVEAT
        assert "ROE는 표시값 ÷2" not in card["sub"]     # 옛 오독 표기 제거

    def test_이력이_없어도_카드가_뜬다(self):
        s = dash._build_summary(1250.0, [], [], {}, {})
        card = next(c for c in s["cards"] if c["key"] == "xvenue")
        assert card["value"] == "+0.000%" and card["roe"] == "+0.000%"

    def test_실시간_트랙C에_ROE가_병기된다(self, tmp_path):
        (tmp_path / "trackc_state.json").write_text(
            json.dumps(dict(equity=1.02)), encoding="utf-8")
        t = dash._tracks_live(logs_dir=tmp_path)
        assert t["c"]["pct"] == pytest.approx(2.0)
        assert t["c"]["roe_pct"] == pytest.approx(1.0)
        assert "명목" in t["c"]["basis"]

    def test_실시간_다른_트랙에는_ROE키가_없다(self, tmp_path):
        (tmp_path / "tracka_state.json").write_text(
            json.dumps(dict(equity=1.01)), encoding="utf-8")
        t = dash._tracks_live(logs_dir=tmp_path)
        assert "roe_pct" not in t["a"]                # 자본 기준 — 변환 대상 아님

    def test_페이지에_두_수치와_주의문구가_보인다(self, client, monkeypatch, tmp_path):
        tracks = self._curves(tmp_path)
        monkeypatch.setattr(dash, "_load_track_curves", lambda *a, **k: tracks)
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"+2.000%" in resp.data                      # 명목
        assert b"+1.000%" in resp.data                      # ROE
        assert "명목".encode() in resp.data
        assert self.CAVEAT.encode() in resp.data
        assert "Track C ROE".encode() in resp.data          # 차트 점선 라벨
        assert "ROE는 표시값 ÷2".encode() not in resp.data   # 옛 표기 잔존 금지


# ------------------------------------------------------------------
# Track E — 단타 팜 (표시 전용 로더 + 구조적 방화벽)
# ------------------------------------------------------------------

class TestLoadTracke:
    """_load_tracke — Track E 표시 데이터 로더 (고정 순서·고정 라벨)."""

    FIXED_ORDER = [f"E{i:02d}" for i in range(1, 11)]

    def _hist_wide(self, tmp_path, rows):
        cols = ["ts"] + self.FIXED_ORDER
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r[c]) for c in cols))
        (tmp_path / "tracke_history.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    def test_파일이_없으면_T0_대기_구조(self, tmp_path):
        from src.dashboard.app import _load_tracke
        t = _load_tracke(logs_dir=tmp_path)
        assert t["available"] is False
        assert t["max_cell"] is None
        assert [c["id"] for c in t["cells"]] == self.FIXED_ORDER
        assert all(c["pct"] is None for c in t["cells"])
        assert t["farm"]["labels"] == []
        assert t["verdicts"] == ["2026-09-26", "2026-11-25", "2027-02-23"]

    def test_고정_라벨이_명세대로_붙는다(self, tmp_path):
        from src.dashboard.app import _load_tracke
        t = _load_tracke(logs_dir=tmp_path)
        by_id = {c["id"]: c for c in t["cells"]}
        assert by_id["E01"]["label"] == "역사적 탈락"
        assert "선택할인" in by_id["E05"]["label"]
        assert "Track D 중복" in by_id["E06"]["label"]
        assert by_id["E09"]["label"] == "미검증 가설 U1"
        assert by_id["E02"]["basket_label"] == "OOD·미검증 코인셋"

    def test_이력을_수익률과_팜_통계로_변환한다(self, tmp_path):
        from src.dashboard.app import _load_tracke
        base = {c: 10000.0 for c in self.FIXED_ORDER}
        row2 = dict(base)
        row2.update(E01=10100.0, E02=9900.0)          # +1% / -1%, 나머지 0%
        self._hist_wide(tmp_path, [dict(ts="2026-09-01T00:00:00Z", **base),
                                   dict(ts="2026-09-01T01:00:00Z", **row2)])
        t = _load_tracke(logs_dir=tmp_path)
        assert t["available"] is True
        by_id = {c["id"]: c for c in t["cells"]}
        assert by_id["E01"]["pct"] == pytest.approx(1.0)
        assert by_id["E02"]["pct"] == pytest.approx(-1.0)
        assert by_id["E02"]["mdd"] == pytest.approx(1.0)
        assert len(t["farm"]["labels"]) == 2
        assert t["farm"]["mean"][-1] == pytest.approx(0.0)      # 동일가중 (+1-1)/10
        assert t["farm"]["median"][-1] == pytest.approx(0.0)
        assert t["farm"]["q1"][-1] <= t["farm"]["median"][-1] <= t["farm"]["q3"][-1]

    def test_고정_순서는_성과와_무관하고_최대셀은_태그만_단다(self, tmp_path):
        from src.dashboard.app import _load_tracke
        base = {c: 10000.0 for c in self.FIXED_ORDER}
        row2 = dict(base)
        row2.update(E07=11000.0, E01=9000.0)          # E07 이 최고 성과
        self._hist_wide(tmp_path, [dict(ts="2026-09-01T00:00:00Z", **base),
                                   dict(ts="2026-09-01T01:00:00Z", **row2)])
        t = _load_tracke(logs_dir=tmp_path)
        assert [c["id"] for c in t["cells"]] == self.FIXED_ORDER   # 정렬 금지
        assert t["max_cell"] == "E07"
        flags = [c["is_max"] for c in t["cells"]]
        assert flags.count(True) == 1
        assert t["cells"][6]["is_max"] is True

    def test_상태_지표를_셀별로_읽는다(self, tmp_path):
        # 픽스처를 합성 키가 아니라 엔진 CellState.to_dict() 실스키마로 만들어
        # 스키마 표류(키 개명 등)를 자동 검출한다 (carrybot 은 import 만, 수정 금지).
        from carrybot.aggressive.scalp_farm import CellState, FarmPos
        from src.dashboard.app import _load_tracke
        cell = CellState(equity=10120.0, halts=2, cost=34.5, fund=-2.1,
                         turnover=51000.0)
        cell.positions = {
            "BTC": FarmPos(d=1, u=0.2, e=50000.0, stop=48000.0, kind="BRK"),
            "SOL": FarmPos(d=-1, u=50.0, e=150.0, stop=160.0, kind="MR"),
        }
        px = {"BTC": 51000.0}                  # SOL 가격 이력 없음 → 진입가 마크
        d = cell.to_dict(px)
        # 엔진 직렬화 계약: gross = sum(|u|×마지막 유효 종가)/equity (봉 종가 기준)
        exp_gross = (0.2 * 51000.0 + 50.0 * 150.0) / 10120.0
        assert d["gross"] == pytest.approx(exp_gross)
        (tmp_path / "tracke_state.json").write_text(json.dumps({
            "t0": "2026-08-28T00:00:00Z",
            "cells": {"E01": d},
        }), encoding="utf-8")
        t = _load_tracke(logs_dir=tmp_path)
        assert t["available"] is True and t["t0"] == "2026-08-28T00:00:00Z"
        c = t["cells"][0]
        assert c["pct"] == pytest.approx(1.2)
        assert c["cost"] == pytest.approx(34.5)
        assert c["funding"] == pytest.approx(-2.1)
        assert c["turnover"] == pytest.approx(51000.0)
        assert c["gross"] == pytest.approx(exp_gross)
        assert c["halts"] == 2

    def test_포지션_없는_셀의_gross_는_0으로_표시된다(self, tmp_path):
        # 엔진 계약: 포지션 없으면 gross=0.0 — 대시보드는 '-'가 아니라 0을 표시
        from carrybot.aggressive.scalp_farm import CellState
        from src.dashboard.app import _load_tracke
        d = CellState(equity=10000.0).to_dict()
        assert d["gross"] == 0.0
        (tmp_path / "tracke_state.json").write_text(json.dumps(
            {"cells": {"E01": d}}), encoding="utf-8")
        t = _load_tracke(logs_dir=tmp_path)
        assert t["cells"][0]["gross"] == 0.0          # None('-') 아님

    def test_손상된_파일은_크래시_없이_대기_구조(self, tmp_path):
        from src.dashboard.app import _load_tracke
        (tmp_path / "tracke_history.csv").write_text(
            "ts,E01\n2026-09-01,깨진값\n", encoding="utf-8")
        (tmp_path / "tracke_state.json").write_text("{{{{", encoding="utf-8")
        t = _load_tracke(logs_dir=tmp_path)
        assert t["available"] is False
        assert [c["id"] for c in t["cells"]] == self.FIXED_ORDER

    def test_long_형식_이력도_읽는다(self, tmp_path):
        from src.dashboard.app import _load_tracke
        (tmp_path / "tracke_history.csv").write_text(
            "ts,cell,equity\n"
            "2026-09-01T00:00:00Z,E03,10000\n"
            "2026-09-01T01:00:00Z,E03,10050\n", encoding="utf-8")
        t = _load_tracke(logs_dir=tmp_path)
        by_id = {c["id"]: c for c in t["cells"]}
        assert by_id["E03"]["pct"] == pytest.approx(0.5)

    def test_엔진_실스키마_이력과_상태를_읽는다(self, tmp_path):
        # scalp_farm_runner 실제 스키마: HIST = day,ts(epoch ms),equity(총합),
        # e01..e10(소문자, USD),n_pos,bars,fills / STATE = t0(epoch ms),
        # cells.E01 = CellState.to_dict() (엔진에서 직접 생성 — 표류 자동 검출)
        from carrybot.aggressive.scalp_farm import CellState
        from src.dashboard.app import _load_tracke
        cells_lower = [c.lower() for c in self.FIXED_ORDER]
        head = "day,ts,equity," + ",".join(cells_lower) + ",n_pos,bars,fills"
        r1 = "2026-09-08,1788829200000,100000," + ",".join(["10000"] * 10) + ",0,1,0"
        vals = ["10250" if c == "e05" else "10000" for c in cells_lower]
        r2 = "2026-09-08,1788832800000,100250," + ",".join(vals) + ",1,1,2"
        (tmp_path / "tracke_history.csv").write_text(
            "\n".join([head, r1, r2]) + "\n", encoding="utf-8")
        e05 = CellState(equity=10250.0, halts=1, cost=12.5, fund=3.3,
                        turnover=41000.0).to_dict()
        (tmp_path / "tracke_state.json").write_text(json.dumps({
            "t0": 1788825600000, "last_ts": 1788832800000,
            "basket_b": ["XRP", "DOGE", "1000PEPE"],
            "cells": {"E05": e05},
        }), encoding="utf-8")
        t = _load_tracke(logs_dir=tmp_path)
        assert t["available"] is True
        assert t["t0"] == "2026-09-08 00:00 UTC"          # epoch ms -> 사람용
        by_id = {c["id"]: c for c in t["cells"]}
        assert by_id["E05"]["pct"] == pytest.approx(2.5)
        assert by_id["E05"]["funding"] == pytest.approx(3.3)   # 엔진 키 "fund"
        assert by_id["E05"]["cost"] == pytest.approx(12.5)
        assert by_id["E05"]["halts"] == 1
        assert by_id["E05"]["gross"] == 0.0                    # 포지션 없음 → 0.0
        assert t["max_cell"] == "E05"
        # 라벨은 epoch ms -> 연도 포함 표시 형식 (표시 직전 포맷)
        assert t["farm"]["labels"][0] == "26-09-08 01:00"

    def test_연도_경계에서도_팜_라벨_순서가_보존된다(self, tmp_path):
        # 집계 키가 원시 epoch ts 라서 2026-12-31 → 2027-01-01 경계에서도
        # 시간순이 유지된다 (연도 없는 라벨 문자열 정렬이면 01-01 이 앞에 옴).
        from datetime import datetime, timezone
        from src.dashboard.app import _load_tracke

        def ms(*a):
            return int(datetime(*a, tzinfo=timezone.utc).timestamp() * 1000)

        base = {c: 10000.0 for c in self.FIXED_ORDER}
        row2 = dict(base)
        row2["E01"] = 10100.0
        row3 = dict(base)
        row3["E01"] = 10200.0
        self._hist_wide(tmp_path, [
            dict(ts=ms(2026, 12, 31, 23), **base),
            dict(ts=ms(2027, 1, 1, 0), **row2),
            dict(ts=ms(2027, 1, 1, 1), **row3),
        ])
        t = _load_tracke(logs_dir=tmp_path)
        assert t["farm"]["labels"] == [
            "26-12-31 23:00", "27-01-01 00:00", "27-01-01 01:00"]
        assert t["farm"]["mean"] == [
            0.0, pytest.approx(0.1), pytest.approx(0.2)]      # E01 만 +1%→+2%
        by_id = {c["id"]: c for c in t["cells"]}
        assert by_id["E01"]["pct"] == pytest.approx(2.0)

    def test_index에_트랙E_섹션이_렌더링된다(self, client, tmp_path, monkeypatch):
        orig = dash._load_tracke
        monkeypatch.setattr(dash, "_load_tracke",
                            lambda logs_dir=None: orig(logs_dir=tmp_path))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "PAPER ONLY".encode() in resp.data
        assert "승급".encode() in resp.data
        assert "T0 대기".encode() in resp.data           # 데이터 없음 → 대기

    def test_데이터가_있으면_사후최대_태그가_뜬다(self, client, tmp_path, monkeypatch):
        base = {c: 10000.0 for c in self.FIXED_ORDER}
        row2 = dict(base)
        row2["E05"] = 10200.0
        self._hist_wide(tmp_path, [dict(ts="2026-09-01T00:00:00Z", **base),
                                   dict(ts="2026-09-01T01:00:00Z", **row2)])
        orig = dash._load_tracke
        monkeypatch.setattr(dash, "_load_tracke",
                            lambda logs_dir=None: orig(logs_dir=tmp_path))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "사후 최대값 — 선택 금지".encode() in resp.data
        assert "고정 순서".encode() in resp.data
        assert "🏆".encode() not in resp.data            # 트로피 금지


class TestTrackeLive:
    """_tracke_live — Track E 단타 팜 초단위 시가평가 (표시 전용)."""

    def _state(self, tmp_path, cells, ind=None):
        (tmp_path / "tracke_state.json").write_text(json.dumps(
            {"t0": 1787813928812, "cells": cells, "ind": ind or {}}),
            encoding="utf-8")

    def test_상태파일이_없으면_None(self, tmp_path):
        assert dash._tracke_live(logs_dir=tmp_path) is None

    def test_손상된_상태파일도_None(self, tmp_path):
        (tmp_path / "tracke_state.json").write_text("{{{{", encoding="utf-8")
        assert dash._tracke_live(logs_dir=tmp_path) is None
        (tmp_path / "tracke_state.json").write_text(
            json.dumps({"cells": "깨진값"}), encoding="utf-8")
        assert dash._tracke_live(logs_dir=tmp_path) is None

    def test_롱숏_시가평가와_팜_합계(self, tmp_path, monkeypatch):
        # 엔진 계약: 셀 equity 는 실현 기준 현금 → 시가평가 = equity + Σu(px−e)d
        self._state(tmp_path, {
            "E01": dict(equity=10000.0,
                        positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
            "E02": dict(equity=10000.0,
                        positions={"SOL": dict(d=-1, u=100.0, e=100.0)}),
        })
        asked = []
        px = {"BTC/USDT:USDT": 81000.0, "SOL/USDT:USDT": 95.0}
        monkeypatch.setattr(dash, "_live_price",
                            lambda s: (asked.append(s), px[s])[1])
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t["cells"]["E01"] == pytest.approx(1.0)   # 롱 +0.1×1000
        assert t["cells"]["E02"] == pytest.approx(5.0)   # 숏 100×(95−100)×(−1)
        assert t["farm_equity"] == pytest.approx(20600.0)
        assert t["farm_pct"] == pytest.approx(3.0)       # 기준 = 셀 수 × $10,000
        assert t["n_pos"] == 2
        assert t["fallback"] == []
        # 심볼 매핑: 엔진과 동일한 Bybit linear 관례 (바스켓 B 포함)
        assert set(asked) == {"BTC/USDT:USDT", "SOL/USDT:USDT"}

    def test_조회_실패는_마지막_종가로_폴백하고_표기한다(self, tmp_path, monkeypatch):
        # HYPE 는 상태의 마지막 유효 종가(ind.pc), BTR 은 pc 도 없어 진입가(손익 0)
        self._state(tmp_path, {
            "E05": dict(equity=10000.0, positions={
                "HYPE": dict(d=1, u=10.0, e=80.0),
                "BTR": dict(d=1, u=5.0, e=2.0),
            })}, ind={"HYPE": dict(pc=90.0)})
        monkeypatch.setattr(dash, "_live_price", lambda s: None)
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t["cells"]["E05"] == pytest.approx(1.0)   # 10×(90−80)=+100, BTR 0
        assert t["farm_equity"] == pytest.approx(10100.0)
        assert t["n_pos"] == 2
        assert sorted(t["fallback"]) == ["BTR", "HYPE"]  # 폴백 여부 표기

    def test_포지션_없는_셀은_현금_자본으로_평가된다(self, tmp_path, monkeypatch):
        self._state(tmp_path, {"E03": dict(equity=10120.0, positions={})})
        monkeypatch.setattr(
            dash, "_live_price",
            lambda s: (_ for _ in ()).throw(AssertionError("가격 조회 금지")))
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t["cells"] == {"E03": pytest.approx(1.2)}
        assert t["n_pos"] == 0 and t["fallback"] == []

    def test_api_live_응답에_tracke_블록이_실린다(self, client, tmp_path, monkeypatch):
        # 표시 전용 블록 — 가격 조회는 fixture 에서 차단(None) → 폴백 경로
        self._state(tmp_path, {
            "E01": dict(equity=10000.0,
                        positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
        }, ind={"BTC": dict(pc=81000.0)})
        orig = dash._tracke_live
        monkeypatch.setattr(dash, "_tracke_live",
                            lambda logs_dir=None: orig(logs_dir=tmp_path))
        resp = client.get("/api/live")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["tracke"]["cells"]["E01"] == pytest.approx(1.0)
        assert data["tracke"]["farm_pct"] == pytest.approx(1.0)
        assert data["tracke"]["n_pos"] == 1
        assert data["tracke"]["fallback"] == ["BTC"]

    def test_api_live는_tracke가_None이어도_뜬다(self, client, tmp_path, monkeypatch):
        # 상태 파일 부재 → tracke: null, 나머지 페이로드는 정상 (크래시 금지)
        orig = dash._tracke_live
        monkeypatch.setattr(dash, "_tracke_live",
                            lambda logs_dir=None: orig(logs_dir=tmp_path))
        resp = client.get("/api/live")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["tracke"] is None
        assert "balance" in data and "tracks_live" in data


class TestTrackeVariant:
    """변형 셀 소구역 — 분리 표시 전용 (공식 판정 대상 아님).

    셀 목록은 엔진 상수(_tracke_variant_spec) 주도 — 테스트도 명세에서
    순서를 읽는다 (셀 추가 시 테스트 무수정).
    """

    VARIANT_ORDER = [c[0] for c in dash._tracke_variant_spec()[0]]
    MAIN_ORDER = [f"E{i:02d}" for i in range(1, 11)]

    def _state(self, tmp_path, main_cells=None, variant=None, variant2=None):
        """tracke_state.json 작성 — variant/variant2 는 병렬 블록 전체."""
        st = {"t0": 1787813928812, "cells": main_cells or {}, "ind": {}}
        if variant is not None:
            st["variant_cells"] = variant
        if variant2 is not None:
            st["variant2_cells"] = variant2
        (tmp_path / "tracke_state.json").write_text(
            json.dumps(st), encoding="utf-8")

    @staticmethod
    def _fake_engine(cells, cells2=()):
        """가짜 scalp_farm 모듈 — 실제 구조(VCELLS/VLABELS + V2CELLS/V2LABELS)."""
        from types import SimpleNamespace

        def _specs(cs):
            return tuple(SimpleNamespace(cell=c, strategy=s, basket=b)
                         for c, s, b in cs)

        def _labels(cs):
            return {c: f"{s} 변형 · 미검증 · 판정 권한 없음" for c, s, _ in cs}

        return SimpleNamespace(VCELLS=_specs(cells), VLABELS=_labels(cells),
                               V2CELLS=_specs(cells2), V2LABELS=_labels(cells2))

    def _main_hist(self, tmp_path):
        """본 이력 — E05 만 +2% (사후최대 태그 대상)."""
        cols = ["ts"] + self.MAIN_ORDER
        base = {c: 10000.0 for c in self.MAIN_ORDER}
        row2 = dict(base)
        row2["E05"] = 10200.0
        lines = [",".join(cols)]
        for ts, r in [("2026-09-01T00:00:00Z", base),
                      ("2026-09-01T01:00:00Z", row2)]:
            lines.append(",".join([ts] + [str(r[c]) for c in self.MAIN_ORDER]))
        (tmp_path / "tracke_history.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    def _patch_loaders(self, monkeypatch, tmp_path):
        orig_t, orig_v = dash._load_tracke, dash._load_tracke_variant
        monkeypatch.setattr(dash, "_load_tracke",
                            lambda logs_dir=None: orig_t(logs_dir=tmp_path))
        monkeypatch.setattr(dash, "_load_tracke_variant",
                            lambda logs_dir=None: orig_v(logs_dir=tmp_path))

    # ── 페이지 로드 로더 ──────────────────────────────────────────
    def test_파일이_없으면_대기_구조(self, tmp_path):
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert v["available"] is False
        assert [c["id"] for c in v["cells"]] == self.VARIANT_ORDER
        assert all(c["pct"] is None for c in v["cells"])

    def test_variant_키가_없거나_손상돼도_크래시_없이_대기(self, tmp_path):
        self._state(tmp_path, main_cells={"E01": dict(equity=10000.0)})
        assert dash._load_tracke_variant(logs_dir=tmp_path)["available"] is False
        self._state(tmp_path, variant="깨진값")
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert v["available"] is False
        assert all(c["pct"] is None for c in v["cells"])

    def test_상태와_이력에서_수익률과_고정_라벨을_읽는다(self, tmp_path):
        # 엔진 실스키마: 변형 이력은 소문자 e11/e12 (본 이력과 동일 관례)
        (tmp_path / "tracke_variant_history.csv").write_text(
            "day,ts,equity,e11,e12,n_pos,bars,fills\n"
            "2026-08-28,1788829200000,20000,10000,10000,0,1,0\n"
            "2026-08-28,1788832800000,20150,10200,9950,2,1,2\n",
            encoding="utf-8")
        self._state(tmp_path, variant={
            "t0_variant": 1787881216290,
            "cells": {
                "E11": dict(equity=10200.0,
                            positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
                "E12": dict(equity=9950.0,
                            positions={"XRP": dict(d=-1, u=1000.0, e=2.0)}),
            },
        })
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert v["available"] is True
        assert [c["id"] for c in v["cells"]] == self.VARIANT_ORDER
        by_id = {c["id"]: c for c in v["cells"]}
        e11, e12 = by_id["E11"], by_id["E12"]
        assert e11["pct"] == pytest.approx(2.0)
        assert e12["pct"] == pytest.approx(-0.5)
        assert e11["strategy"] == e12["strategy"] == "BRK24TP"
        assert e11["label"] == e12["label"] == "빠른 익절 변형 · 미검증 · 판정 권한 없음"
        assert e11["basket_label"] == "BTC·ETH·SOL"
        assert e12["basket_label"] == "XRP·HYPE·BTR"
        assert e11["positions"] == ["BTC 롱"]
        assert e12["positions"] == ["XRP 숏"]

    def test_변형_셀은_사후최대_태그_계산에_불포함(self, tmp_path):
        # 본 셀 최고 E05 +2%, 변형 E11 +50% — max_cell 은 본 표에서만 나온다
        self._main_hist(tmp_path)
        self._state(tmp_path, variant={
            "cells": {"E11": dict(equity=15000.0), "E12": dict(equity=14000.0)}})
        t = dash._load_tracke(logs_dir=tmp_path)
        assert t["max_cell"] == "E05"                        # 변형 +50% 무시
        assert [c["id"] for c in t["cells"]] == self.MAIN_ORDER
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert all("is_max" not in c for c in v["cells"])    # 태그 플래그 자체가 없음
        by_id = {c["id"]: c for c in v["cells"]}
        assert by_id["E11"]["pct"] == pytest.approx(50.0)

    # ── 실시간 (/api/live) ───────────────────────────────────────
    def test_변형_롱숏_시가평가는_variant_블록에만_실린다(self, tmp_path, monkeypatch):
        self._state(
            tmp_path,
            main_cells={"E01": dict(equity=10000.0, positions={})},
            variant={"cells": {
                "E11": dict(equity=10000.0,
                            positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
                "E12": dict(equity=10000.0,
                            positions={"XRP": dict(d=-1, u=1000.0, e=2.0)}),
            }})
        px = {"BTC/USDT:USDT": 81000.0, "XRP/USDT:USDT": 1.9}
        monkeypatch.setattr(dash, "_live_price", lambda s: px[s])
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t["variant"]["E11"] == pytest.approx(1.0)     # 롱 +0.1×1000
        assert t["variant"]["E12"] == pytest.approx(1.0)     # 숏 1000×(1.9−2.0)×(−1)
        assert t["variant"]["n_pos"] == 2
        # 본 팜 집계와 완전 분리 — 변형이 farm/cells/n_pos/fallback 에 안 섞인다
        assert t["farm_equity"] == pytest.approx(10000.0)
        assert set(t["cells"]) == {"E01"}
        assert t["n_pos"] == 0 and t["fallback"] == []

    def test_변형_폴백은_variant_ind_종가를_쓴다(self, tmp_path, monkeypatch):
        self._state(
            tmp_path,
            main_cells={"E01": dict(equity=10000.0, positions={})},
            variant={"cells": {
                "E11": dict(equity=10000.0,
                            positions={"HYPE": dict(d=1, u=10.0, e=80.0)})},
                "ind": {"HYPE": dict(pc=90.0)}})
        monkeypatch.setattr(dash, "_live_price", lambda s: None)
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t["variant"]["E11"] == pytest.approx(1.0)     # 10×(90−80)
        assert t["fallback"] == []                           # 본 블록 목록에 비혼입

    def test_변형_상태가_없으면_variant는_None이고_본_블록은_정상(self, tmp_path, monkeypatch):
        self._state(tmp_path,
                    main_cells={"E01": dict(equity=10100.0, positions={})})
        monkeypatch.setattr(dash, "_live_price", lambda s: None)
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t is not None and t["variant"] is None
        assert t["cells"]["E01"] == pytest.approx(1.0)

    def test_api_live에_variant가_실린다(self, client, tmp_path, monkeypatch):
        # fixture 가 가격 조회를 차단(None) → 변형 ind 종가 폴백 경로
        self._state(
            tmp_path,
            main_cells={"E01": dict(equity=10000.0, positions={})},
            variant={"cells": {
                "E11": dict(equity=10000.0,
                            positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
                "E12": dict(equity=10050.0, positions={})},
                "ind": {"BTC": dict(pc=81000.0)}})
        orig = dash._tracke_live
        monkeypatch.setattr(dash, "_tracke_live",
                            lambda logs_dir=None: orig(logs_dir=tmp_path))
        resp = client.get("/api/live")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["tracke"]["variant"]["E11"] == pytest.approx(1.0)
        assert data["tracke"]["variant"]["E12"] == pytest.approx(0.5)
        assert data["tracke"]["variant"]["n_pos"] == 1

    # ── 렌더링 ───────────────────────────────────────────────────
    def test_index에_변형_소구역이_렌더링되고_최대태그는_본_표에만(
            self, client, tmp_path, monkeypatch):
        self._main_hist(tmp_path)                            # 본 E05 +2% (max)
        self._state(tmp_path, variant={
            "cells": {"E11": dict(equity=15000.0),           # 변형 +50% > 본 최대
                      "E12": dict(equity=10000.0)}})
        self._patch_loaders(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "빠른 익절·게이트·출판 변형 — 공식 판정 대상 아님".encode() in resp.data
        assert b"trackeVarPct-E11" in resp.data
        assert b"trackeVarPct-E12" in resp.data
        assert b"BRK24TP" in resp.data
        # 셀 고정 라벨은 엔진 명세에서 온다 (데이터 주도)
        assert dash._tracke_variant_spec()[1]["E11"].encode() in resp.data
        # 변형이 더 커도 "사후 최대값 — 선택 금지" 태그는 본 표 최대 셀 1곳에만
        assert resp.data.count("사후 최대값 — 선택 금지".encode()) == 1

    def test_변형_부재시_소구역은_대기_표시(self, client, tmp_path, monkeypatch):
        self._main_hist(tmp_path)                            # 본 표는 정상 렌더
        self._patch_loaders(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "빠른 익절·게이트·출판 변형 — 공식 판정 대상 아님".encode() in resp.data
        assert "변형 상태·이력이 기록되면".encode() in resp.data   # "대기" 안내
        assert b"trackeVarPct-E11" not in resp.data          # 행 없음, 크래시 없음

    # ── 데이터 주도 일반화 (엔진 V*CELLS 그룹 → 행 생성, 하드코딩 금지) ──
    TWO = [("E11", "BRK24TP", "A"), ("E12", "BRK24TP", "B")]
    SIX = [("E13", "BRK24GATE", "A"), ("E14", "BRK24GATE", "B"),
           ("E15", "BBMR", "A"), ("E16", "BBMR", "B"),
           ("E17", "RSI2", "A"), ("E18", "RSI2", "B")]

    def test_일반화_로더는_엔진_두_그룹_명세를_그대로_따른다(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "carrybot.aggressive.scalp_farm",
                            self._fake_engine(self.TWO, self.SIX))
        self._state(
            tmp_path,
            variant={
                "t0_variant": 1787881216290,
                "cells": {"E11": dict(equity=10100.0),
                          "E12": dict(equity=10000.0)}},
            variant2={
                "t0_variant2": 1788881216290,     # 별도 t0 키 병렬 블록
                "cells": {c: dict(equity=10000.0 + i * 100.0)
                          for i, (c, _, _) in enumerate(self.SIX)}})
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert v["available"] is True
        assert [c["id"] for c in v["cells"]] == \
            [c for c, _, _ in self.TWO + self.SIX]           # E11..E18 고정 순서
        assert [c["strategy"] for c in v["cells"]] == \
            [s for _, s, _ in self.TWO + self.SIX]
        by_id = {c["id"]: c for c in v["cells"]}
        assert by_id["E11"]["pct"] == pytest.approx(1.0)
        assert by_id["E13"]["pct"] == pytest.approx(0.0)
        assert by_id["E18"]["pct"] == pytest.approx(5.0)
        assert by_id["E15"]["label"] == "BBMR 변형 · 미검증 · 판정 권한 없음"
        assert by_id["E13"]["basket_label"] == "BTC·ETH·SOL"  # 바스켓 표기 폴백

    def test_부분_존재_E11_가동_E13_대기(self, client, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "carrybot.aggressive.scalp_farm",
                            self._fake_engine([("E11", "BRK24TP", "A")],
                                              [("E13", "BRK24GATE", "A")]))
        self._state(tmp_path,
                    main_cells={"E01": dict(equity=10000.0)},   # 본 표 렌더용
                    variant={
                        "t0_variant": 1787881216290,
                        "cells": {"E11": dict(equity=10200.0)},
                    })                        # variant2 블록 자체 부재 (t0 대기)
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert v["available"] is True
        by_id = {c["id"]: c for c in v["cells"]}
        assert by_id["E11"]["pct"] == pytest.approx(2.0)
        assert by_id["E13"]["pct"] is None               # 미가동 → "대기" 표시 대상
        # 렌더링: E13 행이 뜨되 수익률 자리는 "대기", 크래시 없음
        self._patch_loaders(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"trackeVarPct-E13" in resp.data
        assert 'id="trackeVarPct-E13">대기<'.encode() in resp.data
        assert 'id="trackeVarPct-E11">+2.00%<'.encode() in resp.data

    def test_live_변형은_두_상태_블록을_모두_평가한다(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "carrybot.aggressive.scalp_farm",
                            self._fake_engine([("E11", "BRK24TP", "A")],
                                              [("E13", "BRK24GATE", "A"),
                                               ("E15", "BBMR", "A")]))
        self._state(
            tmp_path,
            main_cells={"E01": dict(equity=10000.0, positions={})},
            variant={"cells": {
                "E11": dict(equity=10000.0,
                            positions={"BTC": dict(d=1, u=0.1, e=80000.0)})}},
            variant2={"cells": {
                "E13": dict(equity=10100.0, positions={}),
                "E15": dict(equity=10000.0,
                            positions={"HYPE": dict(d=1, u=10.0, e=80.0)})},
                "ind": {"HYPE": dict(pc=90.0)}})     # 블록 자체 ind 폴백
        monkeypatch.setattr(dash, "_live_price",
                            lambda s: {"BTC/USDT:USDT": 81000.0}.get(s))
        t = dash._tracke_live(logs_dir=tmp_path)
        assert t["variant"]["E11"] == pytest.approx(1.0)   # 롱 +0.1×1000
        assert t["variant"]["E13"] == pytest.approx(1.0)   # 현금 자본
        assert t["variant"]["E15"] == pytest.approx(1.0)   # 변형2 자체 ind 종가 폴백
        assert t["variant"]["n_pos"] == 2
        assert set(t["cells"]) == {"E01"}                  # 본 집계 비혼입

    def test_엔진_import_실패시_E11_E12_폴백(self, tmp_path, monkeypatch):
        import sys
        # sys.modules 의 None 엔트리는 import 를 ImportError 로 중단시킨다
        monkeypatch.setitem(sys.modules, "carrybot.aggressive.scalp_farm", None)
        cells, labels, baskets = dash._tracke_variant_spec()
        assert [c[0] for c in cells] == ["E11", "E12"]
        assert labels["E11"] == "빠른 익절 변형 · 미검증 · 판정 권한 없음"
        self._state(tmp_path, variant={"cells": {"E11": dict(equity=10100.0)}})
        v = dash._load_tracke_variant(logs_dir=tmp_path)   # 크래시 금지
        assert v["available"] is True
        assert [c["id"] for c in v["cells"]] == ["E11", "E12"]
        by_id = {c["id"]: c for c in v["cells"]}
        assert by_id["E11"]["pct"] == pytest.approx(1.0)
        assert by_id["E12"]["pct"] is None


class TestTrackeCellsDetail:
    """cells_detail — 셀 행 펼침 상세 (표시 전용 · /api/live 탑재 · 강조 없음)."""

    def _state(self, tmp_path, cells, ind=None, variant=None):
        st = {"t0": 1787813928812, "cells": cells, "ind": ind or {}}
        if variant is not None:
            st["variant_cells"] = variant
        (tmp_path / "tracke_state.json").write_text(
            json.dumps(st), encoding="utf-8")

    def _patch_all(self, monkeypatch, tmp_path):
        """페이지 로드 로더 + 실시간 평가를 임시 디렉토리로 돌린다."""
        orig_t, orig_v, orig_l = (dash._load_tracke, dash._load_tracke_variant,
                                  dash._tracke_live)
        monkeypatch.setattr(dash, "_load_tracke",
                            lambda logs_dir=None: orig_t(logs_dir=tmp_path))
        monkeypatch.setattr(dash, "_load_tracke_variant",
                            lambda logs_dir=None: orig_v(logs_dir=tmp_path))
        monkeypatch.setattr(dash, "_tracke_live",
                            lambda logs_dir=None: orig_l(logs_dir=tmp_path))

    # ── 계산 ─────────────────────────────────────────────────────
    def test_롱숏_개별_미실현과_합계가_계산된다(self, tmp_path, monkeypatch):
        self._state(tmp_path, {
            "E01": dict(equity=10050.0, cost_cum=12.5, funding_cum=-1.25,
                        positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
            "E02": dict(equity=10000.0,
                        positions={"SOL": dict(d=-1, u=100.0, e=100.0)}),
        })
        px = {"BTC/USDT:USDT": 81000.0, "SOL/USDT:USDT": 95.0}
        monkeypatch.setattr(dash, "_live_price", lambda s: px[s])
        det = dash._tracke_live(logs_dir=tmp_path)["cells_detail"]
        e01 = det["E01"]
        assert e01["equity"] == pytest.approx(10050.0)
        # 실현손익 = equity − 셀 초기자본($10,000 동결 상수)
        assert e01["equity"] - dash.TRACKE_CELL_CAPITAL == pytest.approx(50.0)
        assert e01["unrealized"] == pytest.approx(100.0)   # 롱 0.1×(81000−80000)
        assert e01["cost"] == pytest.approx(12.5)
        assert e01["fund"] == pytest.approx(-1.25)
        p = e01["positions"][0]
        assert (p["sym"], p["dir"]) == ("BTC", "롱")
        assert p["entry"] == pytest.approx(80000.0)
        assert p["mark"] == pytest.approx(81000.0)
        assert p["qty"] == pytest.approx(0.1)
        assert p["upnl"] == pytest.approx(100.0)
        assert p["upnl_pct"] == pytest.approx(1.25)
        sp = det["E02"]["positions"][0]
        assert sp["dir"] == "숏"
        assert sp["upnl"] == pytest.approx(500.0)          # 100×(95−100)×(−1)
        assert sp["upnl_pct"] == pytest.approx(5.0)        # 방향 반영 +5%
        assert det["E02"]["unrealized"] == pytest.approx(500.0)

    def test_조회_실패시_폴백_마크가_상세에_실린다(self, tmp_path, monkeypatch):
        self._state(tmp_path, {
            "E05": dict(equity=10000.0, positions={
                "HYPE": dict(d=1, u=10.0, e=80.0),
                "BTR": dict(d=1, u=5.0, e=2.0)})},
            ind={"HYPE": dict(pc=90.0)})
        monkeypatch.setattr(dash, "_live_price", lambda s: None)
        det = dash._tracke_live(logs_dir=tmp_path)["cells_detail"]["E05"]
        by_sym = {p["sym"]: p for p in det["positions"]}
        assert by_sym["HYPE"]["mark"] == pytest.approx(90.0)   # ind.pc 폴백
        assert by_sym["BTR"]["mark"] == pytest.approx(2.0)     # 진입가 폴백
        assert by_sym["BTR"]["upnl"] == pytest.approx(0.0)     # 손익 0
        assert det["unrealized"] == pytest.approx(100.0)

    def test_포지션_없는_셀은_빈_상세(self, tmp_path, monkeypatch):
        self._state(tmp_path, {"E03": dict(equity=10120.0, positions={})})
        monkeypatch.setattr(dash, "_live_price", lambda s: None)
        det = dash._tracke_live(logs_dir=tmp_path)["cells_detail"]["E03"]
        assert det["positions"] == []
        assert det["unrealized"] == pytest.approx(0.0)
        assert det["cost"] is None and det["fund"] is None

    def test_변형_셀도_cells_detail에_포함되고_집계는_분리(self, tmp_path, monkeypatch):
        self._state(tmp_path,
                    cells={"E01": dict(equity=10000.0, positions={})},
                    variant={"cells": {"E11": dict(
                        equity=9900.0, cost_cum=3.0,
                        positions={"XRP": dict(d=-1, u=1000.0, e=2.0)})}})
        monkeypatch.setattr(dash, "_live_price",
                            lambda s: {"XRP/USDT:USDT": 2.1}.get(s))
        t = dash._tracke_live(logs_dir=tmp_path)
        det = t["cells_detail"]
        assert set(det) == {"E01", "E11"}                  # 본 + 변형 전부
        e11 = det["E11"]
        assert e11["equity"] == pytest.approx(9900.0)      # 실현 −100
        assert e11["unrealized"] == pytest.approx(-100.0)  # 숏 1000×(2.1−2.0)×(−1)
        assert e11["cost"] == pytest.approx(3.0)
        assert e11["positions"][0]["dir"] == "숏"
        assert e11["positions"][0]["upnl_pct"] == pytest.approx(-5.0)
        # 본 팜 집계에는 변형 비혼입 (기존 방화벽 유지)
        assert set(t["cells"]) == {"E01"}
        assert t["farm_equity"] == pytest.approx(10000.0)

    # ── /api/live 페이로드 ───────────────────────────────────────
    def test_api_live_페이로드에_cells_detail이_실린다(self, client, tmp_path,
                                                monkeypatch):
        # fixture 가 가격 조회를 차단(None) → ind 종가 폴백 마크
        self._state(tmp_path, {"E01": dict(
            equity=10000.0, positions={"BTC": dict(d=1, u=0.1, e=80000.0)})},
            ind={"BTC": dict(pc=81000.0)})
        orig = dash._tracke_live
        monkeypatch.setattr(dash, "_tracke_live",
                            lambda logs_dir=None: orig(logs_dir=tmp_path))
        resp = client.get("/api/live")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        d = data["tracke"]["cells_detail"]["E01"]
        assert d["unrealized"] == pytest.approx(100.0)
        assert d["positions"][0]["sym"] == "BTC"
        assert d["positions"][0]["dir"] == "롱"
        assert d["positions"][0]["mark"] == pytest.approx(81000.0)

    # ── 렌더링 (행 클릭 펼침 상세 — 페이지 로드 시 전부 접힘) ────
    def test_상세_행이_전부_접힘으로_렌더된다(self, client, tmp_path, monkeypatch):
        import re as _re
        self._state(tmp_path, {"E01": dict(equity=10000.0, positions={})})
        self._patch_all(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        for cid in dash.TRACKE_CELL_IDS:                   # 본 셀 10개 전부
            assert f'id="trackeDetail-{cid}"' in html
            assert f'id="trackeDetailBody-{cid}"' in html
        for m in _re.finditer(r'<tr class="tracke-detail-row"[^>]*', html):
            assert "display:none" in m.group(0)            # 로드 시 접힘

    def test_상세는_포지션_유무를_렌더한다(self, client, tmp_path, monkeypatch):
        # E01 보유 (ind 종가 폴백 마크) · E02 보유 없음 — 실현 = equity − 10000
        self._state(tmp_path, {
            "E01": dict(equity=10000.0,
                        positions={"BTC": dict(d=1, u=0.1, e=80000.0)}),
            "E02": dict(equity=10120.0, positions={}),
        }, ind={"BTC": dict(pc=81000.0)})
        self._patch_all(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        e01 = html.split('id="trackeDetailBody-E01"')[1].split(
            'id="trackeDetail-E02"')[0]
        assert "코인" in e01 and "방향" in e01             # 미니 표 헤더
        assert "BTC" in e01 and "롱" in e01
        assert "80000" in e01 and "81000" in e01           # 진입가·폴백 마크
        assert "+100.00" in e01                            # 개별 미실현 $
        e02 = html.split('id="trackeDetailBody-E02"')[1].split(
            'id="trackeDetail-E03"')[0]
        assert "보유 없음" in e02
        assert "+120.00" in e02                            # 실현 = 10120 − 10000

    def test_변형_셀_상세_행도_렌더된다(self, client, tmp_path, monkeypatch):
        self._state(tmp_path,
                    cells={"E01": dict(equity=10000.0)},
                    variant={"cells": {"E11": dict(
                        equity=10050.0,
                        positions={"XRP": dict(d=-1, u=1000.0, e=2.0)})},
                        "ind": {"XRP": dict(pc=1.9)}})
        self._patch_all(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert 'id="trackeDetail-E11"' in html
        e11 = html.split('id="trackeDetailBody-E11"')[1].split(
            'id="trackeDetail-E12"')[0]
        assert "XRP" in e11 and "숏" in e11
        assert "+100.00" in e11                            # 1000×(1.9−2.0)×(−1)
        assert "+50.00" in e11                             # 실현 = 10050 − 10000


class TestTrackeFirewall:
    """구조적 방화벽 — Track E 상태·이력이 승급/실거래 게이트에 못 들어간다."""

    def test_트랙E_파일참조는_대시보드_표시층에만_있다(self):
        from pathlib import Path
        src = Path(dash.__file__).resolve().parent.parent   # src/
        offenders = []
        for p in src.rglob("*.py"):
            if "dashboard" in p.parts:
                continue                                     # 표시층만 허용
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
            if "tracke_" in text:
                offenders.append(str(p))
        assert offenders == [], f"게이트 계층에서 Track E 파일 참조 발견: {offenders}"

    def test_게이트_함수_소스에_트랙E_참조가_없다(self):
        # api_live 는 표시 전용 tracke 블록(_tracke_live)을 실을 수 있으나,
        # 게이트 계산 함수(_promote_status/_tracks_live/_build_summary)는
        # 여전히 Track E 를 일절 참조하지 않는다.
        import inspect
        for fn in (dash._promote_status, dash._tracks_live,
                   dash._build_summary):
            code = inspect.getsource(fn)
            assert "tracke_" not in code.lower(), fn.__name__
            assert "TRACKE" not in code, fn.__name__
        from pathlib import Path
        checker = (Path(dash.__file__).resolve().parent.parent
                   / "risk" / "promote_checker.py")
        assert "tracke" not in checker.read_text(encoding="utf-8").lower()

    def test_tracks_live는_트랙E_상태를_읽지_않는다(self, tmp_path):
        (tmp_path / "tracke_state.json").write_text(json.dumps(dict(
            equity=1.5, positions={"BTC": dict(direction=1, units=1.0,
                                               entry=100.0)})))
        assert dash._tracks_live(logs_dir=tmp_path) == {}

    def test_track_curves는_트랙E_이력을_읽지_않는다(self, tmp_path):
        (tmp_path / "tracke_history.csv").write_text(
            "day,equity,n_pos,fills\n2026-09-01,1.5,1,x\n", encoding="utf-8")
        t = dash._load_track_curves(logs_dir=tmp_path)
        assert set(t.keys()) == {"a", "b", "c", "d"}          # "e" 없음
        assert all(tr["labels"] == [] for tr in t.values())

    def test_트랙E_실시간_평가는_상태_파일을_변경하지_않는다(self, tmp_path, monkeypatch):
        # /api/live 의 tracke 블록은 읽기 전용 표시 경로다 —
        # 판정·원장·상태 파일에 어떤 쓰기도 발생하지 않는다.
        raw = json.dumps({"cells": {"E01": dict(
            equity=10000.0,
            positions={"BTC": dict(d=1, u=0.1, e=80000.0)})}})
        (tmp_path / "tracke_state.json").write_text(raw, encoding="utf-8")
        monkeypatch.setattr(dash, "_live_price", lambda s: 81000.0)
        out = dash._tracke_live(logs_dir=tmp_path)
        assert out is not None
        assert (tmp_path / "tracke_state.json").read_text(
            encoding="utf-8") == raw                  # 바이트 단위 불변
        assert list(tmp_path.iterdir()) == [tmp_path / "tracke_state.json"]

    def test_요약_카드는_여전히_6개이고_트랙E_카드가_없다(self):
        s = dash._build_summary(1250.0, [], [], {}, {})
        assert len(s["cards"]) == 6
        assert all("E0" not in c["key"] and c["key"] != "tracke"
                   for c in s["cards"])


class TestLoadMarketScan:
    """전 코인 스캐너 로더 — 존재/부재/오래됨/손상 (표시 전용, 크래시 금지)."""

    def _write(self, tmp_path, generated_at="2026-08-28T03:00:00+00:00",
               coins=None, extra=None):
        d = {"generated_at_utc": generated_at,
             "coins": coins if coins is not None else [
                 {"symbol": "BTC/USDT:USDT", "coin": "BTC", "price": 100000.0,
                  "chg24h_pct": 1.2, "turnover24h": 5e9,
                  "dist24h_pct": 0.5, "dist96h_pct": -1.2, "rsi14": 61.0,
                  "rsi2": 95.0, "bb_pctb": 1.02, "sma200_pct": 3.1,
                  "vol_surge": 2.5, "gate_long": True, "gate_short": False},
                 {"symbol": "ETH/USDT:USDT", "coin": "ETH", "price": 4000.0,
                  "chg24h_pct": -0.3, "turnover24h": 3e9,
                  "dist24h_pct": 2.0, "dist96h_pct": 4.0, "rsi14": 42.0,
                  "rsi2": 8.0, "bb_pctb": 0.30, "sma200_pct": -1.0,
                  "vol_surge": 0.8, "gate_long": False, "gate_short": False},
             ],
             "skipped": 1}
        d.update(extra or {})
        (tmp_path / "market_scan.json").write_text(
            json.dumps(d), encoding="utf-8")

    def test_파일이_없으면_스캔_대기_구조(self, tmp_path):
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is False
        assert s["coins"] == []

    def test_파일_순서를_보존하고_메타를_채운다(self, tmp_path):
        from datetime import datetime, timezone
        self._write(tmp_path)
        now = datetime(2026, 8, 28, 3, 30, tzinfo=timezone.utc)   # 30분 뒤 — 신선
        s = dash._load_market_scan(logs_dir=tmp_path, now=now)
        assert s["available"] is True
        assert [c["coin"] for c in s["coins"]] == ["BTC", "ETH"]  # 재정렬 금지
        assert s["generated_at"] == "2026-08-28 03:00 UTC"
        assert s["stale"] is False
        assert s["age_label"] is None
        assert s["skipped"] == 1

    def test_오래되면_n시간_전_라벨(self, tmp_path):
        from datetime import datetime, timezone
        self._write(tmp_path)
        now = datetime(2026, 8, 28, 8, 10, tzinfo=timezone.utc)   # 5시간+ 경과
        s = dash._load_market_scan(logs_dir=tmp_path, now=now)
        assert s["stale"] is True
        assert s["age_label"] == "5시간 전"

    def test_손상_JSON은_대기로_흡수(self, tmp_path):
        (tmp_path / "market_scan.json").write_text("{broken", encoding="utf-8")
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is False

    def test_빈_코인_목록은_대기로_흡수(self, tmp_path):
        self._write(tmp_path, coins=[])
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is False

    def test_생성시각_기형이어도_크래시_없음(self, tmp_path):
        self._write(tmp_path, generated_at="언젠가")
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is True
        assert s["stale"] is False

    def test_coins가_리스트가_아니면_대기로_흡수(self, tmp_path):
        (tmp_path / "market_scan.json").write_text(
            json.dumps({"coins": 1, "generated_at_utc": "x"}), encoding="utf-8")
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is False

    def test_거대수치_Infinity_극단시각도_크래시_없음(self, tmp_path):
        """OverflowError 경로 3종 — float 승격·시각 변환·skipped 정수화."""
        (tmp_path / "market_scan.json").write_text(
            '{"generated_at_utc": "9999-12-31T23:59:59-23:59",'
            ' "skipped": Infinity,'
            ' "coins": [{"coin": "BTC", "price": 1e999,'
            f' "rsi14": {10 ** 400}, "gate_long": true}}]}}',
            encoding="utf-8")
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is True
        assert s["coins"][0]["price"] is None      # JSON Infinity — 무시
        assert s["coins"][0]["rsi14"] is None      # 거대 int — 무시
        assert s["coins"][0]["gate_long"] is True
        assert s["skipped"] == 0

    def test_불완전한_행은_전_키를_None으로_정규화한다(self, tmp_path):
        """필드 결손·기형 타입이 템플릿 UndefinedError 를 일으키지 않는다."""
        self._write(tmp_path, coins=[
            {"coin": "BTC"},                              # 숫자 필드 전부 결손
            {"symbol": "ETH/USDT:USDT", "price": "많이",   # 문자열 수치 — 무시
             "gate_long": "yes"},                          # 비불리언 — False
            {"price": 1.0},                                # 식별자 없음 — 행 폐기
            "행이 아님",                                    # dict 아님 — 폐기
        ])
        s = dash._load_market_scan(logs_dir=tmp_path)
        assert s["available"] is True
        assert [c["coin"] for c in s["coins"]] == ["BTC", "ETH"]
        btc = s["coins"][0]
        for k in ("price", "chg24h_pct", "dist24h_pct", "dist96h_pct",
                  "rsi14", "rsi2", "bb_pctb"):
            assert btc[k] is None
        assert btc["gate_long"] is False
        assert s["coins"][1]["price"] is None
        assert s["coins"][1]["gate_long"] is False

    def test_불완전한_행도_렌더된다(self, tmp_path, monkeypatch, client):
        """정규화된 결손 행(전부 None)이 템플릿을 통과한다 — 크래시 금지."""
        self._write(tmp_path, coins=[{"coin": "BTC"}])
        data = dash._load_market_scan(logs_dir=tmp_path)   # 실제 로더 산출물
        monkeypatch.setattr(dash, "_load_market_scan", lambda *a, **k: data)
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"BTC" in resp.data


class TestMarketScanRender:
    """전 코인 스캐너 카드 렌더 — 존재/부재, 주의 문구, /api/live 비대상."""

    _FIXTURE = {
        "available": True, "generated_at": "2026-08-28 03:00 UTC",
        "age_label": None, "stale": False, "skipped": 0,
        "coins": [
            {"symbol": "BTC/USDT:USDT", "coin": "BTC", "price": 100000.0,
             "chg24h_pct": 1.2, "turnover24h": 5e9, "dist24h_pct": 0.5,
             "dist96h_pct": -1.2, "rsi14": 61.0, "rsi2": 95.0,
             "bb_pctb": 1.02, "sma200_pct": 3.1, "vol_surge": 2.5,
             "gate_long": True, "gate_short": False},
        ],
    }

    def test_카드와_주의_문구가_렌더된다(self, client, monkeypatch):
        monkeypatch.setattr(dash, "_load_market_scan",
                            lambda *a, **k: dict(self._FIXTURE))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "전 코인 스캐너".encode() in resp.data
        assert "관측 전용".encode() in resp.data
        assert "자동 거래 아님. 거래는 동결된 18계정 규칙만 수행.".encode() in resp.data
        assert b"BTC" in resp.data
        assert "●".encode() in resp.data          # 롱 게이트 충족 표시
        assert "○".encode() in resp.data          # 숏 게이트 미충족 표시
        assert b"2026-08-28 03:00 UTC" in resp.data

    def test_부재시_스캔_대기(self, client, monkeypatch):
        monkeypatch.setattr(dash, "_load_market_scan",
                            lambda *a, **k: {"available": False,
                                             "generated_at": None,
                                             "age_label": None, "stale": False,
                                             "coins": [], "skipped": 0})
        resp = client.get("/")
        assert resp.status_code == 200
        assert "스캔 대기".encode() in resp.data

    def test_오래되면_회색_라벨이_보인다(self, client, monkeypatch):
        d = dict(self._FIXTURE)
        d.update(stale=True, age_label="5시간 전")
        monkeypatch.setattr(dash, "_load_market_scan", lambda *a, **k: d)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "5시간 전".encode() in resp.data

    def test_api_status에_포함되고_api_live에는_없다(self, client, monkeypatch):
        monkeypatch.setattr(dash, "_load_market_scan",
                            lambda *a, **k: dict(self._FIXTURE))
        st = json.loads(client.get("/api/status").data)
        assert st["market_scan"]["available"] is True
        live = json.loads(client.get("/api/live").data)
        assert "market_scan" not in live          # 시간당 데이터 — 초단위 비대상


class TestTrackeLevels:
    """셀 상세 레벨(손절·목표·추매) + 미니 레벨 바 — 표시 전용, 크래시 금지."""

    def _state(self, tmp_path, cells, ind=None):
        (tmp_path / "tracke_state.json").write_text(json.dumps(
            {"t0": 1787813928812, "cells": cells, "ind": ind or {}}),
            encoding="utf-8")

    def _patch_all(self, monkeypatch, tmp_path):
        orig_t, orig_v, orig_l = (dash._load_tracke, dash._load_tracke_variant,
                                  dash._tracke_live)
        monkeypatch.setattr(dash, "_load_tracke",
                            lambda logs_dir=None: orig_t(logs_dir=tmp_path))
        monkeypatch.setattr(dash, "_load_tracke_variant",
                            lambda logs_dir=None: orig_v(logs_dir=tmp_path))
        monkeypatch.setattr(dash, "_tracke_live",
                            lambda logs_dir=None: orig_l(logs_dir=tmp_path))

    def test_엔진_실스키마_stop_tgt이_거리와_함께_상세에_실린다(self, tmp_path,
                                                    monkeypatch):
        # 픽스처는 엔진 동결 계약(FarmPos.stop/tgt 실키)에서 직접 생성 —
        # 키 개명 등 스키마 표류를 자동 검출한다 (carrybot 은 import 만).
        from carrybot.aggressive.scalp_farm import CellState, FarmPos
        cell = CellState(equity=10000.0)
        cell.positions = {"BTC": FarmPos(d=1, u=0.1, e=80000.0, stop=76000.0,
                                         kind="BRKTP", tgt=84000.0)}
        self._state(tmp_path, {"E01": cell.to_dict()})
        monkeypatch.setattr(dash, "_live_price", lambda s: 80000.0)
        p = dash._tracke_live(
            logs_dir=tmp_path)["cells_detail"]["E01"]["positions"][0]
        assert p["stop"] == pytest.approx(76000.0)
        assert p["tgt"] == pytest.approx(84000.0)
        assert p["stop_pct"] == pytest.approx(-5.0)     # 현재가 대비 −5%
        assert p["tgt_pct"] == pytest.approx(5.0)       # 현재가 대비 +5%
        assert p["add"] is None and p["add_pct"] is None
        bar = p["bar"]
        # 마커 좌표: 손절 < 진입=현재가 < 목표 (4~96% 매핑)
        assert bar["stop"] == pytest.approx(4.0)
        assert bar["tgt"] == pytest.approx(96.0)
        assert bar["entry"] == bar["mark"] == pytest.approx(50.0)
        assert bar["add"] is None

    def test_0_센티널과_키_부재는_None으로_바도_없다(self, tmp_path, monkeypatch):
        # 엔진 계약: stop=0.0/tgt=0.0 은 "레벨 없음" 센티널 (스탑 없는 출판 시스템)
        from carrybot.aggressive.scalp_farm import CellState, FarmPos
        cell = CellState(equity=10000.0)
        cell.positions = {"CL": FarmPos(d=1, u=1.0, e=100.0, stop=0.0,
                                        kind="MRPUB", tgt=0.0)}
        self._state(tmp_path, {"E01": cell.to_dict()})
        monkeypatch.setattr(dash, "_live_price", lambda s: 110.0)
        p = dash._tracke_live(
            logs_dir=tmp_path)["cells_detail"]["E01"]["positions"][0]
        assert p["stop"] is None and p["tgt"] is None and p["add"] is None
        assert p["stop_pct"] is None and p["tgt_pct"] is None
        assert p["bar"] is None                          # 레벨 전무 — 바 없음
        # 키 자체 부재(구형/최소 상태 dict)도 동일하게 None (크래시 금지)
        self._state(tmp_path, {"E02": dict(
            equity=10000.0, positions={"BTC": dict(d=1, u=0.1, e=80000.0)})})
        p2 = dash._tracke_live(
            logs_dir=tmp_path)["cells_detail"]["E02"]["positions"][0]
        assert p2["stop"] is None and p2["tgt"] is None and p2["bar"] is None

    def test_v5_BBADD는_tgt키가_추매가로_재매핑된다(self, tmp_path, monkeypatch):
        # 엔진 실스키마(확정 #a): BBADD 는 FarmPos.tgt 에 추매 트리거가를 싣는다 —
        # 목표가가 아니라 추매가로 표시해야 한다 (감사 MAJOR 2 교정)
        self._state(tmp_path, {"E01": dict(
            equity=10000.0,
            positions={"BTC": dict(d=1, u=0.1, e=100.0, kind="BBADD",
                                   stop=0.0, tgt=94.0)})})
        monkeypatch.setattr(dash, "_live_price", lambda s: 100.0)
        p = dash._tracke_live(
            logs_dir=tmp_path)["cells_detail"]["E01"]["positions"][0]
        assert p["add"] == pytest.approx(94.0)
        assert p["tgt"] is None                          # 목표가로 오표시 금지
        assert p["stop"] is None                         # 0.0 = 센티널
        bar = p["bar"]
        assert bar["add"] is not None and bar["tgt"] is None

    def test_비BBADD는_tgt키가_목표가로_남는다(self, tmp_path, monkeypatch):
        self._state(tmp_path, {"E01": dict(
            equity=10000.0,
            positions={"BTC": dict(d=1, u=0.1, e=100.0, kind="BRK",
                                   stop=95.0, tgt=110.0)})})
        monkeypatch.setattr(dash, "_live_price", lambda s: 100.0)
        p = dash._tracke_live(
            logs_dir=tmp_path)["cells_detail"]["E01"]["positions"][0]
        assert p["tgt"] == pytest.approx(110.0)
        assert p["add"] is None

    def test_숏은_손절이_현재가_오른쪽이다(self, tmp_path, monkeypatch):
        self._state(tmp_path, {"E01": dict(
            equity=10000.0,
            positions={"SOL": dict(d=-1, u=10.0, e=100.0, stop=110.0,
                                   tgt=80.0)})})
        monkeypatch.setattr(dash, "_live_price", lambda s: 95.0)
        p = dash._tracke_live(
            logs_dir=tmp_path)["cells_detail"]["E01"]["positions"][0]
        bar = p["bar"]
        assert bar["tgt"] < bar["mark"] < bar["entry"] < bar["stop"]
        assert p["stop_pct"] == pytest.approx(15.79, abs=0.01)   # 위쪽 +%
        assert p["tgt_pct"] == pytest.approx(-15.79, abs=0.01)

    def test_상세에_레벨_열과_바가_렌더된다(self, client, tmp_path, monkeypatch):
        # fixture 가 가격 조회 차단(None) → ind 종가 폴백 마크로 렌더
        self._state(tmp_path, {"E01": dict(
            equity=10000.0,
            positions={"BTC": dict(d=1, u=0.1, e=80000.0, stop=76000.0,
                                   tgt=84000.0)})},
            ind={"BTC": dict(pc=80000.0)})
        self._patch_all(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        e01 = html.split('id="trackeDetailBody-E01"')[1].split(
            'id="trackeDetail-E02"')[0]
        assert "손절가" in e01 and "목표가" in e01 and "추매가" in e01
        assert "76000" in e01 and "84000" in e01
        assert "(-5.0%)" in e01 and "(+5.0%)" in e01     # 현재가 대비 거리
        assert 'class="lvlbar"' in e01                   # 미니 레벨 바
        assert "lvl-stop" in e01 and "lvl-tgt" in e01 and "lvl-dot" in e01
        assert "&mdash;" in e01                          # 추매가 부재 — "—"

    def test_레벨_없는_포지션도_렌더된다(self, client, tmp_path, monkeypatch):
        self._state(tmp_path, {"E01": dict(
            equity=10000.0,
            positions={"BTC": dict(d=1, u=0.1, e=80000.0)})},
            ind={"BTC": dict(pc=80000.0)})
        self._patch_all(monkeypatch, tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200                   # 크래시 금지
        e01 = resp.data.decode().split('id="trackeDetailBody-E01"')[1].split(
            'id="trackeDetail-E02"')[0]
        assert e01.count("&mdash;") >= 4                 # 손절·목표·추매·바 전부 "—"
        assert 'class="lvlbar"' not in e01


class TestScannerGauges:
    """전 코인 스캐너 — BB %b 게이지 + 돌파 근접 게이지 (표시 전용)."""

    _FIXTURE = TestMarketScanRender._FIXTURE

    def test_게이지가_렌더된다(self, client, monkeypatch):
        monkeypatch.setattr(dash, "_load_market_scan",
                            lambda *a, **k: dict(self._FIXTURE))
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert 'class="bbgauge"' in html                 # %b 게이지 (숫자 대신)
        assert "하단" in html and "중심" in html and "상단" in html
        assert "bb-dot up" in html                       # %b 1.02 — 상단 밖 초록 점
        assert 'class="distgauge brk"' in html           # dist96h −1.2% — 돌파 초록
        assert "%b 1.02" in html                         # 정확값은 title 로 보존
        assert "+0.5" in html.replace("+0.50", "+0.5")   # 돌파 거리 수치 병기

    def test_결손_필드는_대시로_렌더되고_크래시_없다(self, client, monkeypatch, tmp_path):
        (tmp_path / "market_scan.json").write_text(json.dumps(
            {"generated_at_utc": "2026-08-28T03:00:00+00:00",
             "coins": [{"coin": "BTC"}]}), encoding="utf-8")
        data = dash._load_market_scan(logs_dir=tmp_path)   # 실로더 산출물 (전부 None)
        monkeypatch.setattr(dash, "_load_market_scan", lambda *a, **k: data)
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "BTC" in html
        assert 'class="bbgauge"' not in html             # None → 게이지 없이 '-'
        assert 'class="distgauge"' not in html

    def test_코인_순서는_파일_순서_그대로다(self, client, monkeypatch):
        # 게이지 추가 후에도 재정렬 금지 (거래대금 순 고정 — 표시 규율)
        two = dict(self._FIXTURE)
        second = dict(two["coins"][0])
        second.update(coin="ETH", symbol="ETH/USDT:USDT", bb_pctb=-0.10,
                      dist24h_pct=12.0, dist96h_pct=None)
        two["coins"] = [two["coins"][0], second]
        monkeypatch.setattr(dash, "_load_market_scan", lambda *a, **k: two)
        resp = client.get("/")
        html = resp.data.decode()
        assert html.index(">BTC<") < html.index(">ETH<")
        assert "bb-dot dn" in html                       # %b < 0 — 하단 밖 빨강 점


class TestVariant5Recognition:
    """v5(E24·E25) — 기존 정규식(V*CELLS·variant*_cells)의 자동 인식 확인."""

    def _fake_v5(self):
        from types import SimpleNamespace
        specs = tuple(SimpleNamespace(cell=c, strategy=s, basket=b)
                      for c, s, b in [("E24", "PARETO5", "A"),
                                      ("E25", "PARETO5", "B")])
        labels = {"E24": "파레토 앙상블 · 미검증 · 판정 권한 없음",
                  "E25": "파레토 앙상블 · 미검증 · 판정 권한 없음"}
        return SimpleNamespace(V5CELLS=specs, V5LABELS=labels)

    def test_V5CELLS와_variant5_cells를_무수정_인식한다(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "carrybot.aggressive.scalp_farm",
                            self._fake_v5())
        cells, labels, _ = dash._tracke_variant_spec()
        assert [c[0] for c in cells] == ["E24", "E25"]   # V5CELLS 매치
        assert "파레토" in labels["E24"]                  # V5LABELS 매치
        st = {"cells": {}, "variant5_cells": {
            "t0_variant5": 1791234567890,
            "cells": {"E24": dict(equity=10100.0)}}}
        assert len(dash._tracke_variant_blocks(st)) == 1  # variant5_cells 매치
        (tmp_path / "tracke_state.json").write_text(
            json.dumps(st), encoding="utf-8")
        v = dash._load_tracke_variant(logs_dir=tmp_path)
        assert v["available"] is True
        by_id = {c["id"]: c for c in v["cells"]}
        assert by_id["E24"]["pct"] == pytest.approx(1.0)
        assert by_id["E25"]["pct"] is None               # 미가동 → "대기"


# ------------------------------------------------------------------
# 코인 차트 뷰 — /api/chart/<coin> (관측 도구 · 표시 전용)
# ------------------------------------------------------------------

def _mk_chart_bars(n=220, t0=1_756_700_400_000):
    """결정적 합성 1h 확정봉 [(ts, o, h, l, c, v), ...] — 테스트 전용."""
    import math as _m
    bars = []
    px = 100.0
    for i in range(n):
        drift = _m.sin(i * 0.7) * 1.5 + _m.sin(i * 0.13) * 3.0
        o = px
        c = 100.0 + drift + i * 0.05
        h = max(o, c) + 0.8 + 0.3 * abs(_m.sin(i * 1.3))
        low = min(o, c) - 0.8 - 0.3 * abs(_m.cos(i * 0.9))
        v = 50.0 + 20.0 * _m.sin(i * 0.31) + (30.0 if i == n - 1 else 0.0)
        bars.append((t0 + i * 3_600_000, o, h, low, c, v))
        px = c
    return bars


class TestChartSeries:
    """_chart_series — 엔진 동일식 (prev-close TR · Wilder RSI · ddof=0 · shift 1)."""

    def test_지표_산식이_스캐너_엔진과_동치다(self):
        """대시보드 시계열 마지막 값 == market_scanner.compute_metrics (동일 봉)."""
        from carrybot.live.market_scanner import compute_metrics
        bars = _mk_chart_bars(220)
        s = dash._chart_series(bars)
        m = compute_metrics(bars)
        close = s["close"][-1]
        assert close == pytest.approx(m["close"], abs=1e-6)
        assert s["rsi14"][-1] == pytest.approx(m["rsi14"], abs=0.01)
        assert ((s["ch24_up"][-1] / close - 1) * 100
                == pytest.approx(m["dist24h_pct"], abs=1e-3))
        assert ((s["ch96_up"][-1] / close - 1) * 100
                == pytest.approx(m["dist96h_pct"], abs=1e-3))
        pctb = (close - s["bb_low"][-1]) / (s["bb_up"][-1] - s["bb_low"][-1])
        assert pctb == pytest.approx(m["bb_pctb"], abs=1e-3)
        assert ((close / s["sma200"][-1] - 1) * 100
                == pytest.approx(m["sma200_pct"], abs=1e-3))
        assert (s["volume"][-1] / s["vol_avg20"][-1]
                == pytest.approx(m["vol_surge"], abs=1e-3))

    def test_ATR은_prev_close_TR_기준이다(self):
        """갭 봉에서 |h−pc| 가 TR 에 반영된다 (swing.py 버그 재발 금지)."""
        bars = [(0, 100.0, 101.0, 99.0, 100.0, 1.0),
                (3_600_000, 120.0, 121.0, 119.0, 120.0, 1.0)]
        s = dash._chart_series(bars)
        # TR1 = 2, TR2 = max(2, |121−100|, |119−100|) = 21 → 2 + (21−2)/24
        assert s["atr24"] == pytest.approx(2 + 19 / 24, abs=1e-6)

    def test_채널은_현재봉을_제외한다(self):
        """shift(1) — 마지막 봉 고가 스파이크가 ch24_up[-1] 에 미반영."""
        bars = _mk_chart_bars(40)
        t, o, h, low, c, v = bars[-1]
        spiked = bars[:-1] + [(t, o, h + 1000.0, low, c, v)]
        assert (dash._chart_series(spiked)["ch24_up"][-1]
                == dash._chart_series(bars)["ch24_up"][-1])

    def test_거래량_평균은_직전_20봉이다(self):
        """vol_avg20 은 현재 봉 거래량을 제외한다 (shift 1)."""
        bars = _mk_chart_bars(30)
        s = dash._chart_series(bars)
        vols = [b[5] for b in bars]
        assert s["vol_avg20"][-1] == pytest.approx(sum(vols[-21:-1]) / 20)

    def test_미형성_구간은_None(self):
        s = dash._chart_series(_mk_chart_bars(50))
        assert s["sma200"][-1] is None          # 200봉 미달
        assert s["bb_up"][0] is None            # BB 워밍업
        assert s["ch96_up"][-1] is None         # 96봉 창 미형성 (50 < 97)
        assert s["ch24_up"][-1] is not None


class TestCompositeScore:
    """_composite_score — 동결 산식 경계값 (각 요소 ±20 · 합 ±100)."""

    def test_전부_미형성이면_중립_0이고_라벨이_붙는다(self):
        r = dash._composite_score(None, None, None, None, None, None, None)
        assert r["score"] == 0
        assert all(v == 0.0 for v in r["parts"].values())
        assert "매매 신호가 아님" in r["note"]
        assert "SWEEP-2026-08-31" in r["note"]

    def test_모멘텀_경계(self):
        part = lambda rsi: dash._composite_score(
            None, None, None, rsi, None, None, None)["parts"]["momentum"]
        assert part(100.0) == 20.0
        assert part(0.0) == -20.0
        assert part(50.0) == 0.0
        assert part(75.0) == 10.0

    def test_밴드는_낮을수록_매수측_플러스(self):
        part = lambda pctb: dash._composite_score(
            None, None, None, None, pctb, None, None)["parts"]["band"]
        assert part(0.0) == 20.0                # 하단 = 매수측 +
        assert part(1.0) == -20.0               # 상단 = 매도측 −
        assert part(0.5) == 0.0
        assert part(-0.5) == 20.0               # 밴드 밖 — 클립
        assert part(1.5) == -20.0

    def test_거래량_로그_스케일(self):
        part = lambda sg: dash._composite_score(
            None, None, None, None, None, sg, None)["parts"]["volume"]
        assert part(1.0) == 0.0
        assert part(2.0) == pytest.approx(10.0)
        assert part(4.0) == pytest.approx(20.0)
        assert part(8.0) == 20.0                # 클립
        assert part(0.5) == pytest.approx(-10.0)
        assert part(0.0) == 0.0                 # 비정상 → 중립
        assert part(-1.0) == 0.0

    def test_돌파_역수_캡(self):
        part = lambda d: dash._composite_score(
            None, None, None, None, None, None, d)["parts"]["breakout"]
        assert part(-2.0) == 20.0               # 이미 돌파 — 캡
        assert part(0.0) == 20.0
        assert part(1.0) == pytest.approx(10.0)
        assert part(3.0) == pytest.approx(5.0)
        assert part(99.0) == pytest.approx(0.2)  # 원거리 → 0 수렴 (음수 없음)

    def test_추세_정배열_가중과_클립(self):
        part = lambda c, s20, s200: dash._composite_score(
            c, s20, s200, None, None, None, None)["parts"]["trend"]
        assert part(110.0, 105.0, 100.0) == 20.0    # base +15 + 정배열 +5
        assert part(120.0, 105.0, 100.0) == 20.0    # base 클립 ±10 → 여전히 20
        assert part(95.0, 90.0, 100.0) == pytest.approx(-12.5)  # −7.5 − 5
        assert part(90.0, 95.0, 100.0) == -20.0     # base −15 − 역배열 5
        assert part(100.0, None, None) == 0.0        # SMA200 미형성 → 중립

    def test_점수는_합산_클립_정수(self):
        # 5요소 전부 매수측 극단 → 정확히 +100
        r = dash._composite_score(110.0, 105.0, 100.0, 100.0, 0.0, 4.0, -1.0)
        assert r["score"] == 100
        assert isinstance(r["score"], int)
        # 4요소 매도측 극단 + 돌파 0 → −80 (돌파는 음수 없음 — 산식 명기)
        r2 = dash._composite_score(90.0, 95.0, 100.0, 0.0, 1.0, 0.25, 99.0)
        assert r2["score"] == -80


class TestChartPosLevels:
    """_chart_pos_levels — 전 그룹 스캔·병합·BBADD 재매핑 (읽기 전용)."""

    def _state(self, tmp_path, st):
        (tmp_path / "tracke_state.json").write_text(
            json.dumps(st), encoding="utf-8")

    def test_상태_부재_손상은_빈_목록(self, tmp_path):
        assert dash._chart_pos_levels("BTC", logs_dir=tmp_path) == []
        (tmp_path / "tracke_state.json").write_text("{깨짐", encoding="utf-8")
        assert dash._chart_pos_levels("BTC", logs_dir=tmp_path) == []

    def test_전_그룹_스캔과_동일_레벨_셀_병합(self, tmp_path):
        self._state(tmp_path, {
            "cells": {
                "E01": {"positions": {"BTC": dict(
                    d=1, u=0.1, e=100.0, stop=94.0, tgt=0.0, kind="BRK")}},
                "E03": {"positions": {"BTC": dict(
                    d=1, u=0.2, e=100.0, stop=94.0, tgt=0.0, kind="BRK")}},
            },
            "variant2_cells": {"cells": {
                "E13": {"positions": {"BTC": dict(
                    d=-1, u=1.0, e=101.0, stop=106.0, tgt=95.0,
                    kind="BRKTP")}}}},
        })
        out = dash._chart_pos_levels("BTC", logs_dir=tmp_path)
        assert len(out) == 2                       # 본 셀 병합 1 + 변형 1
        long_g = next(g for g in out if g["dir"] == "롱")
        assert long_g["cells"] == ["E01", "E03"]   # 고정 순서 병합
        assert long_g["entry"] == pytest.approx(100.0)
        assert long_g["stop"] == pytest.approx(94.0)
        assert long_g["tgt"] is None               # 0.0 센티널 → None
        assert long_g["add"] is None
        short_g = next(g for g in out if g["dir"] == "숏")
        assert short_g["cells"] == ["E13"]
        assert short_g["stop"] == pytest.approx(106.0)
        assert short_g["tgt"] == pytest.approx(95.0)

    def test_BBADD는_tgt를_추매가로_재매핑한다(self, tmp_path):
        """FarmPos 스키마 재사용 — BBADD 의 tgt 키 = 추매 트리거가 (엔진 #a)."""
        self._state(tmp_path, {
            "cells": {},
            "variant5_cells": {"cells": {
                "E24": {"positions": {"BTC": dict(
                    d=1, u=0.1, e=100.0, stop=0.0, tgt=97.5, kind="BBADD")}}}},
        })
        out = dash._chart_pos_levels("BTC", logs_dir=tmp_path)
        assert len(out) == 1
        g = out[0]
        assert g["add"] == pytest.approx(97.5)     # tgt → 추매가 재매핑
        assert g["tgt"] is None                    # 익절 목표 아님
        assert g["stop"] is None                   # BBADD 스탑 없음 (0.0 센티널)

    def test_다른_코인은_제외된다(self, tmp_path):
        self._state(tmp_path, {"cells": {"E01": {"positions": {
            "ETH": dict(d=1, u=1.0, e=2000.0, stop=1900.0, kind="BRK")}}}})
        assert dash._chart_pos_levels("BTC", logs_dir=tmp_path) == []
        assert len(dash._chart_pos_levels("eth", logs_dir=tmp_path)) == 1

    def test_상태_파일을_변경하지_않는다(self, tmp_path):
        raw = json.dumps({"cells": {"E01": {"positions": {
            "BTC": dict(d=1, u=0.1, e=100.0, stop=94.0, kind="BRK")}}}})
        (tmp_path / "tracke_state.json").write_text(raw, encoding="utf-8")
        dash._chart_pos_levels("BTC", logs_dir=tmp_path)
        assert (tmp_path / "tracke_state.json").read_text(
            encoding="utf-8") == raw               # 바이트 단위 불변 (읽기 전용)


class TestApiChart:
    """/api/chart/<coin> — 응답 구조 · 캐시 · 대기 흡수 (크래시 금지)."""

    def _mock_fetch(self, monkeypatch, bars):
        calls: list[str] = []

        def fake(symbol):
            calls.append(symbol)
            return bars
        monkeypatch.setattr(dash, "_chart_fetch_1h", fake)
        return calls

    def test_정상_응답_구조와_상시_라벨(self, client, monkeypatch):
        self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [])
        resp = client.get("/api/chart/BTC")
        assert resp.status_code == 200
        d = json.loads(resp.data)
        assert d["available"] is True
        assert d["coin"] == "BTC"
        assert d["symbol"] == "BTC/USDT:USDT"
        assert len(d["bars"]["close"]) == 200      # 응답 봉 수 상한
        assert len(d["series"]["rsi14"]) == 200
        for k in ("bb_up", "bb_low", "sma20", "sma200", "rsi14",
                  "ch24_up", "ch24_dn", "ch96_up", "ch96_dn", "vol_avg20"):
            assert k in d["series"]
        assert set(d["composite"]["parts"]) == {
            "trend", "momentum", "band", "volume", "breakout"}
        assert -100 <= d["composite"]["score"] <= 100
        assert all(-20 <= v <= 20 for v in d["composite"]["parts"].values())
        # 관측 도구 라벨 상시 동봉 (표시 규율)
        assert "우연 수준" in d["label"]
        assert "매매 신호가 아님" in d["label"]
        assert d["atr24"] is not None

    def test_팜_보유_없으면_전략_기준_레벨(self, client, monkeypatch):
        self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [])
        d = json.loads(client.get("/api/chart/BTC").data)
        lv = d["levels"]
        assert lv["mode"] == "strategy"
        assert lv["brk_entry"] is not None         # 24h 채널 상단 (돌파 진입)
        assert lv["mr_entry"] is not None          # BB 하단 (평균회귀 진입)
        assert lv["stop_mult"] == 6.0              # BRK_ATR_MULT 동수
        assert lv["stop_dist"] == pytest.approx(6.0 * d["atr24"], abs=1e-6)

    def test_팜_보유_있으면_포지션_레벨(self, client, monkeypatch):
        self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        fixture = [dict(cells=["E01", "E03"], kind="BRK", dir="롱",
                        entry=100.0, stop=94.0, tgt=None, add=None)]
        monkeypatch.setattr(dash, "_chart_pos_levels",
                            lambda *a, **k: list(fixture))
        d = json.loads(client.get("/api/chart/BTC").data)
        assert d["levels"]["mode"] == "positions"
        assert d["levels"]["positions"][0]["cells"] == ["E01", "E03"]

    def test_소문자_코인명도_정규화된다(self, client, monkeypatch):
        self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [])
        d = json.loads(client.get("/api/chart/btc").data)
        assert d["available"] is True
        assert d["coin"] == "BTC"

    def test_기형_코인명은_대기(self, client, monkeypatch):
        calls = self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        d = json.loads(client.get("/api/chart/B-C").data)
        assert d["available"] is False
        assert d["note"] == "차트 데이터 대기"
        assert "매매 신호가 아님" in d["label"]
        assert calls == []                         # 페치 자체를 안 한다

    def test_페치_실패는_대기로_흡수(self, client):
        # fixture 기본값: _chart_fetch_1h → None (캐시도 비어 있음)
        resp = client.get("/api/chart/BTC")
        assert resp.status_code == 200
        d = json.loads(resp.data)
        assert d["available"] is False
        assert d["note"] == "차트 데이터 대기"

    def test_캐시_5분_이내_재사용_만료_후_재페치(self, client, monkeypatch):
        import time as _time
        calls = self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [])
        client.get("/api/chart/BTC")
        client.get("/api/chart/BTC")
        assert len(calls) == 1                     # 5분 이내 — 캐시 재사용
        ts, payload = dash._chart_cache["BTC"]
        dash._chart_cache["BTC"] = (_time.time() - 400.0, payload)  # 만료 주입
        client.get("/api/chart/BTC")
        assert len(calls) == 2                     # 만료 — 재페치

    def test_만료_후_페치_실패시_스테일_잔존값_유지(self, client, monkeypatch):
        import time as _time
        self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [])
        d1 = json.loads(client.get("/api/chart/BTC").data)
        assert d1["available"] is True
        ts, payload = dash._chart_cache["BTC"]
        dash._chart_cache["BTC"] = (_time.time() - 400.0, payload)
        monkeypatch.setattr(dash, "_chart_fetch_1h", lambda s: None)  # 장애
        d2 = json.loads(client.get("/api/chart/BTC").data)
        assert d2["available"] is True             # 스테일 잔존값 (무크래시)
        assert d2["bars"] == d1["bars"]

    def test_레벨은_캐시와_무관하게_매요청_재계산된다(self, client, monkeypatch):
        self._mock_fetch(monkeypatch, _mk_chart_bars(220))
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [])
        d1 = json.loads(client.get("/api/chart/BTC").data)
        assert d1["levels"]["mode"] == "strategy"
        monkeypatch.setattr(dash, "_chart_pos_levels", lambda *a, **k: [
            dict(cells=["E01"], kind="BRK", dir="롱", entry=100.0,
                 stop=94.0, tgt=None, add=None)])
        d2 = json.loads(client.get("/api/chart/BTC").data)
        assert d2["levels"]["mode"] == "positions"  # 시장 캐시는 그대로, 레벨 갱신

    def test_확정봉만_수집한다(self, monkeypatch):
        """_chart_fetch_1h — 현재 1h 구간의 미확정 봉을 제외한다."""
        import pandas as pd
        import time as _time
        now_ms = int(_time.time() * 1000)
        cur_h = now_ms - now_ms % 3_600_000
        idx = pd.to_datetime([cur_h - 2 * 3_600_000, cur_h - 3_600_000, cur_h],
                             unit="ms", utc=True)
        df = pd.DataFrame(
            {"open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
             "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2],
             "volume": [10.0, 20.0, 30.0]}, index=idx)

        class _FakeClient:
            def fetch_ohlcv(self, symbol, timeframe, limit):
                assert timeframe == "1h"
                return df
        monkeypatch.setattr(dash, "_get_market_client", lambda: _FakeClient())
        bars = dash._chart_fetch_1h("BTC/USDT:USDT")
        assert len(bars) == 2                      # 미확정 봉(현재 시간대) 제외
        assert bars[-1][4] == pytest.approx(2.2)


class TestCoinChartRender:
    """차트 뷰 프런트 연결 — 스캐너 행·Track E 상세 코인명 (렌더 스모크)."""

    _SCAN = {
        "available": True, "generated_at": "2026-09-01 03:00 UTC",
        "age_label": None, "stale": False, "skipped": 0,
        "coins": [
            {"symbol": "BTC/USDT:USDT", "coin": "BTC", "price": 100000.0,
             "chg24h_pct": 1.2, "turnover24h": 5e9, "dist24h_pct": 0.5,
             "dist96h_pct": -1.2, "rsi14": 61.0, "rsi2": 95.0,
             "bb_pctb": 1.02, "sma200_pct": 3.1, "vol_surge": 2.5,
             "gate_long": True, "gate_short": False},
        ],
    }

    def test_스캐너_행이_코인_차트를_연다(self, client, monkeypatch):
        monkeypatch.setattr(dash, "_load_market_scan",
                            lambda *a, **k: dict(self._SCAN))
        html = client.get("/").data.decode()
        assert "openCoinChart('BTC')" in html       # 행 클릭 연결
        assert 'class="scan-row"' in html

    def test_모달과_상시_라벨이_렌더된다(self, client):
        html = client.get("/").data.decode()
        assert 'id="ccModal"' in html
        assert "우연 수준(SWEEP-2026-08-31)" in html  # 관측 도구 라벨 상시
        assert "매매 신호가 아님" in html
        assert "/api/chart/" in html                 # 자체 canvas 페치 경로

    def test_트랙E_상세_코인명이_차트를_연다(self, client, tmp_path, monkeypatch):
        (tmp_path / "tracke_state.json").write_text(json.dumps({
            "t0": 1787813928812,
            "cells": {"E01": dict(equity=10000.0, positions={
                "BTC": dict(d=1, u=0.1, e=80000.0, stop=76000.0,
                            kind="BRK")})},
            "ind": {"BTC": dict(pc=81000.0)},
        }), encoding="utf-8")
        for name in ("_load_tracke", "_load_tracke_variant", "_tracke_live"):
            orig = getattr(dash, name)
            monkeypatch.setattr(dash, name,
                                (lambda o: lambda logs_dir=None:
                                 o(logs_dir=tmp_path))(orig))
        html = client.get("/").data.decode()
        assert "openCoinChart('BTC')" in html        # 상세 코인명 → 같은 차트
        assert 'class="coin-link"' in html
