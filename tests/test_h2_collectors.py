from __future__ import annotations

"""H2 트랙 B 수집기 테스트 — 네트워크 없이 목/픽스처로 검증.

대상: carrybot/live/portfolio_snapshot.py, carrybot/live/fills_recorder.py
- fills 중복 제거 / 연속성 판정 / 절단(censored) 전환 / 멱등 스킵 / 스냅샷 스키마.
- userFillsByTime 소급 페이지네이션 (만석+겹침 실패 복구 / 윈도 소진 절단).
- 잘린 gzip 멤버 recover-and-rewrite 후 이어받기 (스냅샷·fills 원본/요약 공통).
- 최초 폴링 만석 initial_window_truncated 플래그 / T0 단일 초기화(--t0-init) 순서.
"""

import calendar
import gzip
import json
import time
import zlib

import pytest

import carrybot.live.fills_recorder as fills_recorder
from carrybot.live.fills_recorder import (
    FILLS_RESP_CAP,
    FILLS_WINDOW,
    STATUS_CENSORED,
    STATUS_OK,
    aggregate_fills,
    backfill_gap,
    daily_todo,
    dedup_new_fills,
    judge_continuity,
    load_state,
    pick_high_turnover,
    poll_wallets,
    process_response,
    read_fills_dedup,
    snapshot_positions,
)
from carrybot.live.portfolio_snapshot import (
    collect,
    load_done,
    load_done_labeled,
    parse_row,
)


# ── 합성 픽스처 ──────────────────────────────────────────────────────────

def make_portfolio(pnl_pts, acct_pts):
    """portfolio API 응답 형태([이름, 값] 쌍 리스트)를 흉내낸다."""
    return [["day", {"pnlHistory": [], "accountValueHistory": []}],
            ["perpAllTime", {"pnlHistory": pnl_pts, "accountValueHistory": acct_pts}]]


def make_cohort(tmp_path, wallets):
    """gzip 코호트 파일을 tmp_path 에 만든다."""
    p = tmp_path / "cohort.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(dict(locked_at="2026-08-25", n=len(wallets), wallets=wallets), f)
    return p


def fill(t, tid, px=10.0, sz=2.0, crossed=True):
    """userFills 체결 1건을 흉내낸다 (px/sz 는 실제 API 처럼 문자열)."""
    return dict(time=t, tid=tid, px=str(px), sz=str(sz), crossed=crossed, coin="BTC")


def read_jsonl_gz(path):
    """jsonl.gz 를 dict 리스트로 읽는다."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ── 스냅샷 수집기 ────────────────────────────────────────────────────────

class TestSnapshotSchema:
    """portfolio 응답 → 저장 행 스키마."""

    PNL = [[1000, "1.5"], [2000, "2.5"], [3000, "3.5"], [4000, "4.5"]]
    ACCT = [[1000, "100.0"], [2000, "110.0"], [3000, "120.0"], [4000, "130.5"]]

    def test_최종점과_꼬리_3점을_보존한다(self):
        row = parse_row("0xA", make_portfolio(self.PNL, self.ACCT), "daily", "2026-08-27T00:50:00Z")
        assert row["address"] == "0xA" and row["label"] == "daily"
        assert row["captured_at_utc"] == "2026-08-27T00:50:00Z"
        assert row["perp_alltime_pnl"] == pytest.approx(4.5)      # 문자열 → float
        assert row["account_value"] == pytest.approx(130.5)
        assert row["pnl_ts"] == 4000 and row["acct_ts"] == 4000
        assert row["pnl_tail"] == self.PNL[-3:] and row["acct_tail"] == self.ACCT[-3:]

    def test_perpAllTime_없으면_None(self):
        assert parse_row("0xA", [["day", {}]], "t0", "x") is None

    def test_곡선_결측이나_비정상_payload_는_None(self):
        assert parse_row("0xA", None, "t0", "x") is None
        assert parse_row("0xA", {"error": 1}, "t0", "x") is None
        assert parse_row("0xA", make_portfolio([], self.ACCT), "t0", "x") is None


class TestSnapshotIdempotency:
    """같은 날 재실행 시 이미 수집된 지갑은 건너뛴다 (이어받기)."""

    def _fetch(self, calls):
        def f(addr):
            calls.append(addr)
            return make_portfolio([[1, "1.0"]], [[1, "50.0"]])
        return f

    def test_같은날_재실행은_수집된_지갑을_건너뛴다(self, tmp_path):
        cohort = make_cohort(tmp_path, [dict(address=a) for a in ("0xA", "0xB", "0xC")])
        calls: list[str] = []
        r1 = collect("daily", cohort=cohort, out_dir=tmp_path / "snap",
                     fetch=self._fetch(calls), pause_s=0)
        assert r1["n_ok"] == 3 and len(calls) == 3
        r2 = collect("daily", cohort=cohort, out_dir=tmp_path / "snap",
                     fetch=self._fetch(calls), pause_s=0)
        assert r2["n_ok"] == 0 and len(calls) == 3, "재실행은 API 호출 없음"
        rows = read_jsonl_gz(r1["out"])
        assert [r["address"] for r in rows] == ["0xA", "0xB", "0xC"]

    def test_실패_지갑은_기록하지_않고_재시도한다(self, tmp_path):
        cohort = make_cohort(tmp_path, [dict(address=a) for a in ("0xA", "0xB")])
        bad = lambda addr: None if addr == "0xB" else make_portfolio([[1, "1.0"]], [[1, "2.0"]])  # noqa: E731
        r1 = collect("t0", cohort=cohort, out_dir=tmp_path / "snap", fetch=bad, pause_s=0)
        assert r1["failed"] == ["0xB"] and r1["n_ok"] == 1
        calls: list[str] = []
        r2 = collect("t0", cohort=cohort, out_dir=tmp_path / "snap",
                     fetch=self._fetch(calls), pause_s=0)
        assert calls == ["0xB"] and r2["n_ok"] == 1, "실패분만 재시도"
        assert load_done(tmp_path / "snap" / f"{r1['day']}.jsonl.gz") == {"0xA", "0xB"}

    def test_같은날_다른_라벨은_별도로_수집한다(self, tmp_path):
        # 판정일은 일별 크론과 같은 UTC 일자에 겹친다 — 라벨별 이어받기 필수
        cohort = make_cohort(tmp_path, [dict(address="0xA")])
        calls: list[str] = []
        r1 = collect("daily", cohort=cohort, out_dir=tmp_path / "snap",
                     fetch=self._fetch(calls), pause_s=0)
        r2 = collect("verdict", cohort=cohort, out_dir=tmp_path / "snap",
                     fetch=self._fetch(calls), pause_s=0)
        assert r1["n_ok"] == 1 and r2["n_ok"] == 1 and len(calls) == 2
        rows = read_jsonl_gz(r1["out"])
        assert [r["label"] for r in rows] == ["daily", "verdict"]


# ── fills: 중복 제거 ────────────────────────────────────────────────────

class TestFillsDedup:
    """tid 중복 제거 — 커서 이후 신규 체결만 남긴다."""

    def test_첫_폴링은_전부_신규다(self):
        fills = [fill(1, "t1"), fill(2, "t2")]
        assert dedup_new_fills(fills, None, set()) == fills

    def test_커서_이전_체결은_버린다(self):
        fills = [fill(50, "t0"), fill(100, "t1"), fill(150, "t2")]
        out = dedup_new_fills(fills, 100, {"t1"})
        assert [f["tid"] for f in out] == ["t2"]

    def test_경계_ts의_미저장_tid는_신규로_남긴다(self):
        # 직전 newest=100 에서 t1 만 저장됨 → 같은 ms 의 t1b 는 신규
        fills = [fill(100, "t1"), fill(100, "t1b")]
        out = dedup_new_fills(fills, 100, {"t1"})
        assert [f["tid"] for f in out] == ["t1b"]

    def test_응답_내부_tid_중복도_제거한다(self):
        out = dedup_new_fills([fill(1, "t1"), fill(1, "t1")], None, set())
        assert len(out) == 1


# ── fills: 연속성 판정 ──────────────────────────────────────────────────

class TestContinuity:
    """명세: 이번 응답이 직전 폴링의 newest ts 를 덮어야(겹침) 인정."""

    def test_직전_폴링이_없으면_first(self):
        assert judge_continuity(None, 10, 20, 5) == "first"

    def test_응답이_직전_newest를_덮으면_ok(self):
        assert judge_continuity(100, 100, 150, 5) == "ok"
        assert judge_continuity(100, 50, 100, 5) == "ok"

    def test_직전_폴링이_있는데_빈_응답은_empty로_표기한다(self):
        assert judge_continuity(100, None, None, 0) == "empty"

    def test_응답_전체가_커서_이전이면_stale이다(self):
        # 최신 체결 부재를 증명 못함 — ok 로 치면 당일 폴링 완료로 오인
        assert judge_continuity(100, 50, 90, 2) == "stale"

    def test_겹침실패_cap미달은_gap_incomplete(self):
        assert judge_continuity(100, 101, 200, 5, cap=10) == "gap-incomplete"

    def test_겹침실패_cap도달은_gap_censored(self):
        assert judge_continuity(100, 101, 200, 10, cap=10) == "gap-censored"

    def test_cap도달이라도_겹치면_ok(self):
        assert judge_continuity(100, 99, 200, 10, cap=10) == "ok"


# ── fills: 절단 전환 ────────────────────────────────────────────────────

class TestCensorTransition:
    """10k cap AND 겹침 실패 → 즉시 fill-history-censored (영구, 집계만)."""

    def test_cap도달_겹침실패는_즉시_절단되고_원본을_생략한다(self):
        wst = dict(newest_ts=100, boundary_tids=["t0"], status=STATUS_OK)
        fills = [fill(200 + i, f"n{i}", px=10, sz=1) for i in range(5)]
        wst, raw, summary = process_response("0xA", fills, wst, "T", "daily", cap=5)
        assert wst["status"] == STATUS_CENSORED and wst["censored_at"] == "T"
        assert raw is None, "절단 지갑은 원본 저장 생략"
        assert summary["continuity"] == "gap-censored"
        # 집계는 남는다: 체결수·명목·maker비중·ts범위
        assert summary["new_n"] == 5
        assert summary["new_notional"] == pytest.approx(50.0)
        assert summary["new_maker_frac"] == pytest.approx(0.0)
        assert (summary["new_ts_min"], summary["new_ts_max"]) == (200, 204)
        assert (summary["resp_oldest_ts"], summary["resp_newest_ts"]) == (200, 204)

    def test_절단은_영구다(self):
        wst = dict(newest_ts=204, boundary_tids=[], status=STATUS_CENSORED,
                   censored_at="T0")
        wst, raw, summary = process_response(
            "0xA", [fill(204, "a"), fill(300, "b")], wst, "T1", "daily", cap=5)
        assert summary["continuity"] == "ok", "겹침이 회복돼도"
        assert wst["status"] == STATUS_CENSORED and raw is None, "절단은 유지"
        assert summary["new_n"] == 2, "집계는 계속"

    def test_겹침실패_cap미달은_불완전_표시_원본은_계속(self):
        wst = dict(newest_ts=100, boundary_tids=[], status=STATUS_OK)
        wst, raw, summary = process_response(
            "0xA", [fill(200, "x")], wst, "T", "daily", cap=5)
        assert summary["continuity"] == "gap-incomplete"
        assert wst["incomplete"] is True and wst["status"] == STATUS_OK
        assert raw is not None and [f["tid"] for f in raw["fills"]] == ["x"]
        assert wst["newest_ts"] == 100, "커서 보존 — 이후 덮는 응답이 구멍 복구 가능"

    def test_불완전_이후_덮는_응답이_구멍을_복구한다(self):
        # 커서 100 → gap-incomplete 로 [200] 저장, 커서 유지
        wst = dict(newest_ts=100, boundary_tids=["t100"], status=STATUS_OK)
        wst, _, _ = process_response("0xA", [fill(200, "x")], wst, "T1", "daily", cap=5)
        # 다음 폴링이 [100..250] 을 덮음 → 구멍(150)이 원본에 담긴다
        fills = [fill(100, "t100"), fill(150, "hole"), fill(200, "x"), fill(250, "y")]
        wst, raw, summary = process_response("0xA", fills, wst, "T2", "daily", cap=99)
        assert summary["continuity"] == "ok"
        tids = [f["tid"] for f in raw["fills"]]
        assert "hole" in tids and "t100" not in tids
        assert wst["newest_ts"] == 250, "겹침 확인 후에만 커서 전진"
        assert wst["incomplete"] is False, "덮는 응답이 갭을 복구하면 플래그 해제"

    def test_갭_시작점에_못_미치는_복구는_플래그를_해제하지_않는다(self):
        # 커서 100 → 불완전 [200,300] (갭 = 100~200)
        wst = dict(newest_ts=100, boundary_tids=[], status=STATUS_OK)
        wst, _, _ = process_response("0xA", [fill(200, "a"), fill(300, "b")],
                                     wst, "T1", "daily", cap=99)
        assert wst["gap_until_ts"] == 200
        # 덮는 응답이 150 까지만 도달 — 150~200 미복구
        wst, _, s2 = process_response("0xA", [fill(50, "c"), fill(150, "d")],
                                      wst, "T2", "daily", cap=99)
        assert s2["continuity"] == "ok"
        assert wst["incomplete"] is True and wst["newest_ts"] == 150
        # 이어서 210 까지 덮으면 해제
        wst, _, _ = process_response("0xA", [fill(140, "e"), fill(210, "f")],
                                     wst, "T3", "daily", cap=99)
        assert wst["incomplete"] is False and "gap_until_ts" not in wst

    def test_절단시에는_커서를_전진해_일별_이중집계를_막는다(self):
        wst = dict(newest_ts=100, boundary_tids=[], status=STATUS_OK)
        fills = [fill(200 + i, f"n{i}") for i in range(5)]
        wst, _, _ = process_response("0xA", fills, wst, "T", "daily", cap=5)
        assert wst["status"] == STATUS_CENSORED and wst["newest_ts"] == 204

    def test_커서보다_오래된_응답은_stale_커서를_되돌리지_않는다(self):
        wst = dict(newest_ts=100, boundary_tids=["t100"], status=STATUS_OK)
        wst, raw, summary = process_response(
            "0xA", [fill(50, "old"), fill(90, "old2")], wst, "T", "daily", cap=99)
        assert summary["continuity"] == "stale"
        assert wst["newest_ts"] == 100 and raw is None

    def test_절단_지갑은_gap이라도_커서를_전진해_이중집계를_막는다(self):
        wst = dict(newest_ts=100, boundary_tids=[], status=STATUS_CENSORED,
                   censored_at="T0")
        wst, raw, summary = process_response(
            "0xA", [fill(200, "x")], wst, "T1", "daily", cap=5)
        assert summary["continuity"] == "gap-incomplete"
        assert wst["newest_ts"] == 200 and raw is None

    def test_요약은_사용한_커서를_기록한다(self):
        wst = dict(newest_ts=100, boundary_tids=[], status=STATUS_OK)
        _, _, summary = process_response("0xA", [fill(150, "a")], wst, "T", "daily", cap=99)
        assert summary["prev_newest_ts"] == 100, "크래시 재개 요약 중복 제거 키"

    def test_커서는_최신으로_전진하고_경계_tid를_기록한다(self):
        wst = dict(newest_ts=100, boundary_tids=["t0"], status=STATUS_OK)
        fills = [fill(100, "t0"), fill(150, "a"), fill(150, "b")]
        wst, raw, _ = process_response("0xA", fills, wst, "T", "daily", cap=99)
        assert wst["newest_ts"] == 150 and sorted(wst["boundary_tids"]) == ["a", "b"]
        assert [f["tid"] for f in raw["fills"]] == ["a", "b"], "t0 은 기저장 — 중복 제거"

    def test_커서_정지시_경계_tid는_합집합이다(self):
        wst = dict(newest_ts=100, boundary_tids=["t0"], status=STATUS_OK)
        wst, raw, _ = process_response("0xA", [fill(100, "t0"), fill(100, "t1")],
                                       wst, "T", "daily", cap=99)
        assert wst["newest_ts"] == 100 and sorted(wst["boundary_tids"]) == ["t0", "t1"]
        assert [f["tid"] for f in raw["fills"]] == ["t1"]


# ── fills: 집계·고회전율·멱등 ───────────────────────────────────────────

class TestAggregates:
    def test_명목과_maker비중을_계산한다(self):
        fills = [fill(1, "a", px=100, sz=2, crossed=False),   # maker, 명목 200
                 fill(2, "b", px=50, sz=1, crossed=True)]     # taker, 명목 50
        agg = aggregate_fills(fills)
        assert agg["n"] == 2
        assert agg["notional"] == pytest.approx(250.0)
        assert agg["maker_frac"] == pytest.approx(0.5)
        assert (agg["ts_min"], agg["ts_max"]) == (1, 2)

    def test_빈_리스트는_None_필드다(self):
        agg = aggregate_fills([])
        assert agg == dict(n=0, notional=0.0, maker_frac=None, ts_min=None, ts_max=None)


class TestHighTurnover:
    def test_회전율_상위_3분위만_뽑는다(self):
        wallets = [dict(address=f"0x{i}", t0_month_vlm=v, t0_account=100.0)
                   for i, v in enumerate([100, 600, 300, 500, 200, 400])]
        top = pick_high_turnover(wallets)      # k = 6//3 = 2 → 회전율 6.0, 5.0
        assert top == ["0x1", "0x3"]

    def test_손상_지갑은_건너뛴다(self):
        wallets = [dict(address="0xA", t0_month_vlm=100.0, t0_account=0.0),
                   dict(address="0xB", t0_month_vlm=100.0, t0_account=10.0)]
        assert pick_high_turnover(wallets) == ["0xB"]


class TestDailyIdempotency:
    """일별 폴링 멱등 — 같은 UTC 일자 재실행은 미폴링 지갑만."""

    def test_daily_todo는_오늘_폴링된_지갑을_뺀다(self):
        state = dict(wallets={"0xA": dict(last_daily="2026-08-27"),
                              "0xB": dict(last_daily="2026-08-26")})
        assert daily_todo(["0xA", "0xB", "0xC"], state, "2026-08-27") == ["0xB", "0xC"]

    def test_poll_wallets는_원본과_요약을_남기고_상태를_갱신한다(self, tmp_path):
        state_f = tmp_path / "state.json"
        responses = {"0xA": [fill(10, "a1"), fill(20, "a2")], "0xB": None}
        state = dict(high_turnover=[], wallets={})
        r = poll_wallets(["0xA", "0xB"], state, "daily",
                         fetch=lambda a: responses[a],
                         fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        assert r["n_raw"] == 1 and r["failed"] == ["0xB"]
        day = r["day"]
        raws = read_jsonl_gz(tmp_path / "fills" / day / "fills.jsonl.gz")
        sums = read_jsonl_gz(tmp_path / "fills" / day / "summary.jsonl.gz")
        assert len(raws) == 1 and raws[0]["address"] == "0xA"
        assert [f["tid"] for f in raws[0]["fills"]] == ["a1", "a2"]
        assert len(sums) == 1 and sums[0]["continuity"] == "first"
        assert (sums[0]["resp_oldest_ts"], sums[0]["resp_newest_ts"]) == (10, 20)
        st = load_state(state_f)
        assert st["wallets"]["0xA"]["newest_ts"] == 20
        assert st["wallets"]["0xA"]["last_daily"] == day
        assert "0xB" not in st["wallets"], "실패 지갑은 상태 미변경 (복구 규칙)"
        # 같은 날 재실행: 0xA 는 스킵 대상, 0xB 만 남는다
        assert daily_todo(["0xA", "0xB"], st, day) == ["0xB"]

    def test_형식_이상_응답은_실패_처리하고_상태를_바꾸지_않는다(self, tmp_path):
        state_f = tmp_path / "state.json"
        state = dict(high_turnover=[], wallets={})
        r = poll_wallets(["0xA"], state, "daily",
                         fetch=lambda a: ["문자열", 123],     # 비-dict 원소 — 스키마 이상
                         fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        assert r["failed"] == ["0xA"]
        assert load_state(state_f)["wallets"] == {}, "last_daily 미기록 → 재시도 대상"

    def test_필수키_없는_스키마_변경도_실패_처리한다(self, tmp_path):
        # time/tid 가 과반에서 사라짐 = 스키마 변경 → 상태 오염 없이 정지 효과
        state = dict(high_turnover=[], wallets={})
        r = poll_wallets(["0xA"], state, "daily",
                         fetch=lambda a: [dict(timestamp=1, fill_id="a")] * 3,
                         fills_dir=tmp_path / "f", state_f=tmp_path / "s.json", pause_s=0)
        assert r["failed"] == ["0xA"]

    def test_stale이나_empty_응답은_당일_폴링_완료로_치지_않는다(self, tmp_path):
        state_f = tmp_path / "state.json"
        state = dict(high_turnover=[],
                     wallets={"0xA": dict(newest_ts=100, boundary_tids=[],
                                          status="ok")})
        r = poll_wallets(["0xA"], state, "daily",
                         fetch=lambda a: [fill(90, "old")],   # stale
                         fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        st = load_state(state_f)
        assert "last_daily" not in st["wallets"]["0xA"], "당일 재시도 허용"
        assert daily_todo(["0xA"], st, r["day"]) == ["0xA"]

    def test_두번째_폴링은_신규분만_원본에_쌓는다(self, tmp_path):
        state_f = tmp_path / "state.json"
        state = dict(high_turnover=[], wallets={})
        kw = dict(fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        poll_wallets(["0xA"], state, "intraday",
                     fetch=lambda a: [fill(10, "a1")], **kw)
        r2 = poll_wallets(["0xA"], state, "intraday",
                          fetch=lambda a: [fill(10, "a1"), fill(30, "a2")], **kw)
        raws = read_jsonl_gz(tmp_path / "fills" / r2["day"] / "fills.jsonl.gz")
        assert [f["tid"] for row in raws for f in row["fills"]] == ["a1", "a2"], \
            "겹침 재수신분(a1)은 두 번째 원본에서 제외"


class TestPositionsSnapshot:
    """T0 clearinghouseState 포지션 스냅 — 1회 저장·이어받기."""

    def test_스냅은_저장되고_재실행은_건너뛴다(self, tmp_path):
        cohort = make_cohort(tmp_path, [dict(address="0xA"), dict(address="0xB")])
        out = tmp_path / "positions.jsonl.gz"
        calls: list[str] = []

        def fetch(addr):
            calls.append(addr)
            return dict(assetPositions=[], marginSummary=dict(accountValue="123.0"))

        r1 = snapshot_positions(cohort_path=cohort, out=out, fetch=fetch, pause_s=0)
        assert r1["n_ok"] == 2 and len(calls) == 2
        r2 = snapshot_positions(cohort_path=cohort, out=out, fetch=fetch, pause_s=0)
        assert r2["n_ok"] == 0 and len(calls) == 2, "재실행은 호출 없음"
        rows = read_jsonl_gz(out)
        assert {r["address"] for r in rows} == {"0xA", "0xB"}
        assert rows[0]["state"]["marginSummary"]["accountValue"] == "123.0"

    def test_에러_형태_dict는_완료로_오인하지_않는다(self, tmp_path):
        cohort = make_cohort(tmp_path, [dict(address="0xA")])
        out = tmp_path / "positions.jsonl.gz"
        r1 = snapshot_positions(cohort_path=cohort, out=out,
                                fetch=lambda a: dict(error="rate limited"), pause_s=0)
        assert r1["n_ok"] == 0 and r1["failed"] == ["0xA"]
        ok = dict(assetPositions=[], marginSummary=dict(accountValue="1.0"))
        r2 = snapshot_positions(cohort_path=cohort, out=out,
                                fetch=lambda a: ok, pause_s=0)
        assert r2["n_ok"] == 1, "재실행에서 재시도된다"


# ── fills: 응답 cap 상수 (실측 회귀) ────────────────────────────────────

class TestRespCapConstants:
    """실측: userFills/userFillsByTime 응답당 최대 2,000건, 소급 가용 윈도 1만 건.

    (구 FILLS_CAP=10_000 은 응답에서 도달 불가 — gap-censored 미발화 결함.)"""

    def test_응답_cap_상수는_2000이다(self):
        assert FILLS_RESP_CAP == 2000

    def test_가용_윈도_상수는_10000이다(self):
        assert FILLS_WINDOW == 10_000

    def test_judge_continuity_기본_cap은_응답_cap이다(self):
        # 파라미터 주입이 상수 결함을 가리지 않도록 기본값으로 검증
        assert judge_continuity(100, 101, 300, FILLS_RESP_CAP) == "gap-censored"
        assert judge_continuity(100, 101, 300, FILLS_RESP_CAP - 1) == "gap-incomplete"

    def test_process_response_기본_cap도_응답_cap이다(self):
        wst = dict(newest_ts=100, boundary_tids=[], status=STATUS_OK)
        fills = [fill(200 + i, f"n{i}") for i in range(FILLS_RESP_CAP)]
        _, raw, summary = process_response("0xA", fills, wst, "T", "daily")
        assert summary["continuity"] == "gap-censored" and raw is None


# ── fills: userFillsByTime 소급 페이지네이션 ────────────────────────────

class TestBackfillGap:
    """만석+겹침 실패 → userFillsByTime 순방향 페이지네이션으로 갭 소급 복구."""

    def test_페이지네이션_복구_성공(self):
        # 커서 100, 만석(3) 페이지 → 미만석 페이지로 자연 종료 = 갭 메움
        pages = {
            100: [fill(100, "b0"), fill(150, "b1"), fill(200, "b2")],
            200: [fill(200, "b2"), fill(300, "b3")],
        }
        calls: list[int] = []

        def fbt(addr, start):
            calls.append(start)
            return pages[start]

        outcome, back = backfill_gap("0xA", 100, 500, fbt, resp_cap=3, window=10)
        assert outcome == "recovered"
        assert [f["tid"] for f in back] == ["b0", "b1", "b2", "b3"], \
            "페이지 경계 재수신(b2)은 tid 로 제거"
        assert calls == [100, 200], "다음 startTime = 직전 페이지 마지막 fill time"

    def test_만석이라도_메인_응답_oldest에_닿으면_복구다(self):
        page = [fill(110, "a1"), fill(500, "a2")]        # 만석(2)이지만 target 도달
        outcome, back = backfill_gap("0xA", 100, 400, lambda a, s: page,
                                     resp_cap=2, window=10)
        assert outcome == "recovered" and len(back) == 2

    def test_윈도_소진시에만_censored_확정(self):
        pages = {
            100: [fill(110, "a1"), fill(120, "a2")],
            120: [fill(130, "a3"), fill(140, "a4")],
        }
        outcome, back = backfill_gap("0xA", 100, 999, lambda a, s: pages[s],
                                     resp_cap=2, window=4)
        assert outcome == "censored", "고유 누계가 윈도에 닿았는데 갭 미복구"
        assert len(back) == 4

    def test_페이지_실패는_failed로_상태_보존_대상이다(self):
        assert backfill_gap("0xA", 100, 500, lambda a, s: None,
                            resp_cap=2, window=4) == ("failed", [])


class TestPollWalletsBackfill:
    """poll_wallets 통합 — 기본 상수(2,000/10,000) 배선으로 소급 경로 검증."""

    def _state(self, tmp_path, boundary):
        return dict(high_turnover=[], wallets={
            "0xA": dict(newest_ts=1000, boundary_tids=boundary, status=STATUS_OK)})

    def _main_resp(self, t0=5000):
        return [fill(t0 + i, f"m{i}") for i in range(FILLS_RESP_CAP)]

    def test_만석_겹침실패는_소급으로_복구된다(self, tmp_path):
        state_f = tmp_path / "state.json"
        state = self._state(tmp_path, ["t1000"])
        main = self._main_resp()                          # oldest 5000 > 커서 1000
        page1 = [fill(1000, "t1000")] + [
            fill(1001 + i, f"b{i}") for i in range(FILLS_RESP_CAP - 1)]
        page2 = [fill(2999, f"b{FILLS_RESP_CAP - 2}")] + [
            fill(3000 + i, f"c{i}") for i in range(99)]   # 미만석 → 종료
        by_time = {1000: page1, 2999: page2}
        r = poll_wallets(["0xA"], state, "daily",
                         fetch=lambda a: main,
                         fetch_by_time=lambda a, s: by_time[s],
                         fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        assert r["n_backfilled"] == 1 and r["n_censored"] == 0 and r["failed"] == []
        wst = load_state(state_f)["wallets"]["0xA"]
        assert wst["status"] == STATUS_OK and not wst.get("incomplete")
        assert wst["newest_ts"] == 5000 + FILLS_RESP_CAP - 1, "커서는 원본 최신으로"
        raws = read_jsonl_gz(tmp_path / "fills" / r["day"] / "fills.jsonl.gz")
        tids = [f["tid"] for row in raws for f in row["fills"]]
        assert "t1000" not in tids, "경계 기저장 체결은 제외"
        assert "b0" in tids and "c0" in tids and "m0" in tids, "소급분+원본 병합"
        assert len(tids) == len(set(tids)) == 1999 + 99 + FILLS_RESP_CAP
        sums = read_jsonl_gz(tmp_path / "fills" / r["day"] / "summary.jsonl.gz")
        assert sums[0]["continuity"] == "ok", "복구 성공 = 갭 없음 증명"
        assert sums[0]["backfill_n"] == 2000 + 99

    def test_윈도_소진_지갑만_censored_확정(self, tmp_path):
        state_f = tmp_path / "state.json"
        state = self._state(tmp_path, [])
        main = self._main_resp(t0=50000)                  # oldest 50000 ≫ 소급 도달점

        def fbt(addr, start):                             # 항상 만석 — 갭이 안 메워짐
            return [fill(start + 1 + i, f"t{start + 1 + i}")
                    for i in range(FILLS_RESP_CAP)]

        r = poll_wallets(["0xA"], state, "daily", fetch=lambda a: main,
                         fetch_by_time=fbt,
                         fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        assert r["n_censored"] == 1 and r["n_backfilled"] == 0 and r["failed"] == []
        wst = load_state(state_f)["wallets"]["0xA"]
        assert wst["status"] == STATUS_CENSORED and wst["censored_at"]
        assert wst["newest_ts"] == 50000 + FILLS_RESP_CAP - 1, "절단 시 커서 전진"
        assert r["n_raw"] == 0, "절단 지갑은 원본 생략"
        sums = read_jsonl_gz(tmp_path / "fills" / r["day"] / "summary.jsonl.gz")
        assert sums[0]["continuity"] == "gap-censored"
        assert sums[0]["new_n"] == FILLS_RESP_CAP, "집계만 남는다"

    def test_소급_실패는_지갑_실패_처리_상태_미변경(self, tmp_path):
        state_f = tmp_path / "state.json"
        state = self._state(tmp_path, [])
        r = poll_wallets(["0xA"], state, "daily", fetch=lambda a: self._main_resp(),
                         fetch_by_time=lambda a, s: None,
                         fills_dir=tmp_path / "fills", state_f=state_f, pause_s=0)
        assert r["failed"] == ["0xA"]
        wst = load_state(state_f)["wallets"]["0xA"]
        assert wst == dict(newest_ts=1000, boundary_tids=[], status=STATUS_OK), \
            "상태 미변경 — 다음 폴링 재시도 (기술 실패는 절단 아님)"

    def test_절단_지갑은_소급을_시도하지_않는다(self, tmp_path):
        calls: list[int] = []
        state = dict(high_turnover=[], wallets={
            "0xA": dict(newest_ts=1000, boundary_tids=[],
                        status=STATUS_CENSORED, censored_at="T0")})
        poll_wallets(["0xA"], state, "daily", fetch=lambda a: self._main_resp(),
                     fetch_by_time=lambda a, s: calls.append(s),
                     fills_dir=tmp_path / "fills",
                     state_f=tmp_path / "state.json", pause_s=0)
        assert calls == [], "절단 확정 지갑은 기존 동작(집계만·커서 전진) 유지"
        assert state["wallets"]["0xA"]["status"] == STATUS_CENSORED


# ── 잘린 gzip 멤버 복구 (recover-and-rewrite) ───────────────────────────

class TestTruncatedGzipRecovery:
    """수집 중 kill → 잘린 gzip 멤버. 이어받기 로더가 유효 행만으로 재작성한다."""

    def _write_rows(self, path, rows):
        with gzip.open(path, "at", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _append_truncated(self, path, row):
        """트레일러+α 를 자른 gzip 멤버를 덧붙인다 (kill 시뮬레이션)."""
        member = gzip.compress((json.dumps(row) + "\n").encode())
        with open(path, "ab") as f:
            f.write(member[:len(member) - 12])

    def test_잘린_멤버는_유효행만_읽고_재작성한다(self, tmp_path):
        out = tmp_path / "snap.jsonl.gz"
        self._write_rows(out, [dict(address="0xA", label="daily"),
                               dict(address="0xB", label="daily")])
        self._append_truncated(out, dict(address="0xC", label="daily"))
        with pytest.raises((EOFError, OSError, zlib.error)):
            read_jsonl_gz(out)                    # 픽스처 전제: 표준 리더는 죽는다
        assert load_done_labeled(out, "daily") == {"0xA", "0xB"}, "크래시 없이 유효 행만"
        rows = read_jsonl_gz(out)                 # recover-and-rewrite 후 표준 리더 OK
        assert [r["address"] for r in rows] == ["0xA", "0xB"]

    def test_load_done도_잘린_멤버를_복구한다(self, tmp_path):
        out = tmp_path / "positions.jsonl.gz"
        self._write_rows(out, [dict(address="0xA")])
        self._append_truncated(out, dict(address="0xB"))
        assert load_done(out) == {"0xA"}
        assert read_jsonl_gz(out) == [dict(address="0xA")]

    def test_킬_후_같은날_이어받기가_크래시_없이_이어쓴다(self, tmp_path):
        # 실재현 시나리오: 장시간 수집 중 kill → 같은 날 재실행이 시작 즉시 크래시하던 결함
        cohort = make_cohort(tmp_path, [dict(address=a) for a in ("0xA", "0xB", "0xC")])
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir()
        day = time.strftime("%Y-%m-%d", time.gmtime())
        out = snap_dir / f"{day}.jsonl.gz"
        self._write_rows(out, [dict(address="0xA", label="daily")])   # 확정 멤버
        self._append_truncated(out, dict(address="0xB", label="daily"))  # 잘린 멤버
        calls: list[str] = []

        def fetch(addr):
            calls.append(addr)
            return make_portfolio([[1, "1.0"]], [[1, "50.0"]])

        r = collect("daily", cohort=cohort, out_dir=snap_dir, fetch=fetch, pause_s=0)
        assert calls == ["0xB", "0xC"], "잘린 행(0xB)은 재수집, 확정 행(0xA)은 스킵"
        assert r["n_ok"] == 2
        rows = read_jsonl_gz(out)   # 잘린 멤버 뒤 append 멤버 도달 가능 문제 해소 확인
        assert [row["address"] for row in rows] == ["0xA", "0xB", "0xC"]
        assert rows[1]["label"] == "daily" and "perp_alltime_pnl" in rows[1]


def append_truncated_member(path, row):
    """트레일러+α 를 자른 gzip 멤버를 덧붙인다 (수집 중 kill 시뮬레이션)."""
    member = gzip.compress((json.dumps(row) + "\n").encode())
    with open(path, "ab") as f:
        f.write(member[:len(member) - 12])


class TestFillsCrashDurability:
    """fills 원본/요약도 잘린 gzip 멤버를 복구 후 append 재개한다 (차단 2번).

    포인트: recover-and-rewrite 없이는 다음 실행이 깨진 멤버 뒤에 append 해
    표준 gzip 리더가 이후 데이터에 도달하지 못하던 결함의 end-to-end 재현."""

    def test_잘린_원본_요약_복구_후_append가_표준_리더에_도달한다(self, tmp_path):
        # 시나리오: 정상 폴링 → kill(두 파일 마지막 멤버 절단 + 상태 미저장) →
        # 재폴링(같은 커서 재사용 = 원본 중복 tid) → 표준 리더 전체 도달 확인
        state_f = tmp_path / "state.json"
        fills_dir = tmp_path / "fills"
        kw = dict(fills_dir=fills_dir, state_f=state_f, pause_s=0)
        r1 = poll_wallets(["0xA"], dict(high_turnover=[], wallets={}), "daily",
                          fetch=lambda a: [fill(10, "a1"), fill(20, "a2")], **kw)
        raw_f = fills_dir / r1["day"] / "fills.jsonl.gz"
        sum_f = fills_dir / r1["day"] / "summary.jsonl.gz"
        append_truncated_member(raw_f, dict(address="0xK", fills=[fill(5, "k1")]))
        append_truncated_member(sum_f, dict(address="0xK", continuity="first"))
        for p in (raw_f, sum_f):
            with pytest.raises((EOFError, OSError, zlib.error)):
                read_jsonl_gz(p)          # 픽스처 전제: 표준 리더는 죽는다
        # 크래시로 상태 저장 전 → 재폴링은 커서 없이(first) 같은 체결을 재기록
        r2 = poll_wallets(["0xA"], dict(high_turnover=[], wallets={}), "daily",
                          fetch=lambda a: [fill(10, "a1"), fill(20, "a2"),
                                           fill(30, "a3")], **kw)
        assert r2["n_raw"] == 1
        raws = read_jsonl_gz(raw_f)       # 복구 후 append 멤버까지 도달 가능
        sums = read_jsonl_gz(sum_f)
        tids = [f["tid"] for row in raws for f in row["fills"]]
        assert tids == ["a1", "a2", "a1", "a2", "a3"], "수집 단계 중복은 허용"
        assert [s["address"] for s in sums] == ["0xA", "0xA"], "잘린 요약행은 제거"
        # 분석 단계: 전역 tid keep-first 로 유일성 보장 (명세 §3.2)
        deduped = read_fills_dedup(fills_dir)
        assert [f["tid"] for f in deduped] == ["a1", "a2", "a3"]
        assert all(f["address"] == "0xA" for f in deduped)

    def test_정상_파일은_복구_경로가_건드리지_않는다(self, tmp_path):
        state_f = tmp_path / "state.json"
        fills_dir = tmp_path / "fills"
        kw = dict(fills_dir=fills_dir, state_f=state_f, pause_s=0)
        state = dict(high_turnover=[], wallets={})
        r1 = poll_wallets(["0xA"], state, "intraday",
                          fetch=lambda a: [fill(10, "a1")], **kw)
        poll_wallets(["0xA"], state, "intraday",
                     fetch=lambda a: [fill(10, "a1"), fill(30, "a2")], **kw)
        raws = read_jsonl_gz(fills_dir / r1["day"] / "fills.jsonl.gz")
        assert [f["tid"] for row in raws for f in row["fills"]] == ["a1", "a2"]


class TestReadFillsDedup:
    """read_fills_dedup — 일자 전체에 걸친 전역 tid keep-first (명세 §3.2)."""

    def _write_day(self, root, day, rows):
        d = root / day
        d.mkdir(parents=True)
        with gzip.open(d / "fills.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_일자를_넘는_재폴링_중복_tid를_keep_first로_제거한다(self, tmp_path):
        self._write_day(tmp_path, "2026-08-27",
                        [dict(address="0xA", polled_at_utc="T1", mode="daily",
                              fills=[fill(10, "a1"), fill(20, "a2")])])
        self._write_day(tmp_path, "2026-08-28",
                        [dict(address="0xA", polled_at_utc="T2", mode="daily",
                              fills=[fill(20, "a2"), fill(30, "a3"),
                                     dict(time=35, px="1", sz="1")])])
        out = read_fills_dedup(tmp_path)
        assert [f.get("tid") for f in out] == ["a1", "a2", "a3"], \
            "tid 없는 체결은 유일성 판정 불가 — 분석 목록에서 제외 (원본 보존)"
        a2 = next(f for f in out if f.get("tid") == "a2")
        assert a2["polled_at_utc"] == "T1", "keep-first — 먼저 기록된 행 유지"
        assert a2["address"] == "0xA" and a2["mode"] == "daily"

    def test_tid_표기_흔들림도_str_정규화로_중복_제거한다(self, tmp_path):
        # 같은 tid 가 int/str 로 흔들려도 keep-first 를 우회하지 못한다
        self._write_day(tmp_path, "2026-08-27",
                        [dict(address="0xA", polled_at_utc="T1", mode="daily",
                              fills=[fill(10, 77)])])
        self._write_day(tmp_path, "2026-08-28",
                        [dict(address="0xA", polled_at_utc="T2", mode="daily",
                              fills=[fill(10, "77")])])
        out = read_fills_dedup(tmp_path)
        assert len(out) == 1 and out[0]["polled_at_utc"] == "T1"

    def test_잘린_멤버가_있어도_분석_읽기는_죽지_않는다(self, tmp_path):
        self._write_day(tmp_path, "2026-08-27",
                        [dict(address="0xA", polled_at_utc="T1", mode="daily",
                              fills=[fill(10, "a1")])])
        raw_f = tmp_path / "2026-08-27" / "fills.jsonl.gz"
        append_truncated_member(raw_f, dict(address="0xB", fills=[fill(99, "z")]))
        assert [f["tid"] for f in read_fills_dedup(tmp_path)] == ["a1"]
        with pytest.raises((EOFError, OSError, zlib.error)):
            read_jsonl_gz(raw_f)          # 읽기 전용 — 파일은 재작성하지 않는다


# ── fills: 최초 폴링 절단 플래그 / T0 단일 초기화 ───────────────────────

class TestInitialWindowTruncated:
    """최초 폴링 응답 만석 → initial_window_truncated (명세 §2.3, 차단 3번)."""

    def test_최초_폴링_만석이면_플래그를_기록하고_요약에_노출한다(self):
        wst: dict = {}
        fills = [fill(100 + i, f"t{i}") for i in range(5)]
        wst, raw, summary = process_response("0xA", fills, wst, "T", "daily", cap=5)
        assert summary["continuity"] == "first", "first 처리 자체는 유지"
        assert wst["initial_window_truncated"] is True
        assert summary["initial_window_truncated"] is True, "집계 노출"
        assert raw is not None and wst["newest_ts"] == 104, "커서 기준선은 정상 수립"

    def test_최초_폴링_미만석이면_플래그가_없다(self):
        wst: dict = {}
        wst, _, summary = process_response("0xA", [fill(1, "a")], wst, "T", "daily", cap=5)
        assert "initial_window_truncated" not in wst
        assert "initial_window_truncated" not in summary

    def test_플래그는_이후_폴링에도_유지되고_노출된다(self):
        wst: dict = {}
        wst, _, _ = process_response(
            "0xA", [fill(100 + i, f"t{i}") for i in range(5)], wst, "T0", "daily", cap=5)
        wst, _, s2 = process_response(
            "0xA", [fill(104, "t4"), fill(200, "n")], wst, "T1", "daily", cap=5)
        assert s2["continuity"] == "ok"
        assert wst["initial_window_truncated"] is True
        assert s2["initial_window_truncated"] is True, "이후 요약에도 계속 노출"

    def test_기본_cap은_응답_cap이다(self):
        wst: dict = {}
        wst, _, summary = process_response(
            "0xA", [fill(i, f"t{i}") for i in range(FILLS_RESP_CAP)], wst, "T", "daily")
        assert wst["initial_window_truncated"] is True
        wst2: dict = {}
        wst2, _, _ = process_response(
            "0xA", [fill(i, f"t{i}") for i in range(FILLS_RESP_CAP - 1)],
            wst2, "T", "daily")
        assert "initial_window_truncated" not in wst2


class TestT0Init:
    """T0 단일 초기화 원샷 — 순서·fail-closed·기준선 검증 (명세 §2.3, 차단 3번)."""

    def _patch_ok(self, monkeypatch, calls, saved,
                  collect_r=None, pos_r=None,
                  state=None, wallets=None):
        """세 단계 + 상태 I/O 를 목으로 배선한다 (기본은 전부 완주)."""
        state = state if state is not None else dict(
            wallets={"0xA": dict(newest_ts=5), "0xB": dict(empty_first_at_ms=9)})
        wallets = wallets or [dict(address="0xA"), dict(address="0xB")]
        monkeypatch.setattr(
            fills_recorder, "collect",
            lambda label: (calls.append(f"portfolio:{label}"),
                           collect_r or dict(n_done=2, n_cohort=2))[1])
        monkeypatch.setattr(
            fills_recorder, "snapshot_positions",
            lambda: (calls.append("positions"),
                     pos_r or dict(n_done=2, n_cohort=2))[1])
        monkeypatch.setattr(
            fills_recorder, "run_daily_poll",
            lambda: (calls.append("fills-first-poll"), dict(n_failed=0))[1])
        monkeypatch.setattr(fills_recorder, "load_state", lambda: state)
        monkeypatch.setattr(fills_recorder, "load_cohort_wallets", lambda: wallets)
        monkeypatch.setattr(fills_recorder, "save_state",
                            lambda st: saved.append(st))

    def test_세_단계를_순서대로_수행하고_완주시_플래그를_기록한다(self, monkeypatch):
        calls: list[str] = []
        saved: list[dict] = []
        self._patch_ok(monkeypatch, calls, saved)
        fills_recorder.t0_init()
        assert calls == ["portfolio:t0", "positions", "fills-first-poll"], \
            "① portfolio t0 → ② 포지션 기준선 → ③ fills 첫 폴링"
        assert saved and saved[-1]["t0_initialized_at"], \
            "완주 시에만 크론 게이트 해제 플래그 기록"

    def test_1단계_미완주는_중단하고_다음_단계를_밟지_않는다(self, monkeypatch):
        calls: list[str] = []
        saved: list[dict] = []
        self._patch_ok(monkeypatch, calls, saved,
                       collect_r=dict(n_done=1, n_cohort=2))
        with pytest.raises(SystemExit):
            fills_recorder.t0_init()
        assert calls == ["portfolio:t0"], "fail-closed — 포지션·fills 진행 금지"
        assert saved == [], "플래그 미기록 → 크론 폴링도 계속 대기"

    def test_2단계_미완주도_중단한다(self, monkeypatch):
        calls: list[str] = []
        saved: list[dict] = []
        self._patch_ok(monkeypatch, calls, saved,
                       pos_r=dict(n_done=1, n_cohort=2))
        with pytest.raises(SystemExit):
            fills_recorder.t0_init()
        assert calls == ["portfolio:t0", "positions"]
        assert saved == []

    def test_fills_기준선_미수립_지갑이_있으면_플래그를_기록하지_않는다(self, monkeypatch):
        calls: list[str] = []
        saved: list[dict] = []
        # 0xB 는 커서도 빈 기준선도 없음 → 미완주
        self._patch_ok(monkeypatch, calls, saved,
                       state=dict(wallets={"0xA": dict(newest_ts=5), "0xB": {}}))
        with pytest.raises(SystemExit):
            fills_recorder.t0_init()
        assert calls == ["portfolio:t0", "positions", "fills-first-poll"]
        assert saved == [], "재디스패치로 이어받기 전까지 게이트 유지"

    def test_main의_t0_init_플래그는_원샷_절차를_호출한다(self, monkeypatch):
        called: list[bool] = []
        monkeypatch.setattr(fills_recorder, "t0_init", lambda: called.append(True))
        fills_recorder.main(["--t0-init"])
        assert called == [True]


class TestCronGate:
    """t0_initialized_at 전에는 일별/intraday 크론이 폴링하지 않는다 (§2.3 순서)."""

    def test_플래그_없으면_일별_폴링을_시작하지_않는다(self, monkeypatch):
        called: list[bool] = []
        monkeypatch.setattr(fills_recorder, "load_state", lambda: dict(wallets={}))
        monkeypatch.setattr(fills_recorder, "run_daily_poll",
                            lambda: called.append(True))
        fills_recorder.main([])
        assert called == [], "T0 단일 절차 전 커서 생성 차단"

    def test_플래그가_있으면_일별_폴링을_진행한다(self, monkeypatch):
        called: list[bool] = []
        monkeypatch.setattr(fills_recorder, "load_state",
                            lambda: dict(t0_initialized_at="T", wallets={}))
        monkeypatch.setattr(fills_recorder, "run_daily_poll",
                            lambda: called.append(True))
        fills_recorder.main([])
        assert called == [True]

    def test_intraday도_동일하게_대기한다(self, monkeypatch):
        polled: list[bool] = []
        monkeypatch.setattr(fills_recorder, "load_state", lambda: dict(wallets={}))
        monkeypatch.setattr(fills_recorder, "poll_wallets",
                            lambda *a, **k: polled.append(True))
        fills_recorder.main(["--intraday"])
        assert polled == []

    def test_플래그가_있으면_intraday는_고회전율만_폴링한다(self, monkeypatch):
        seen: list[tuple] = []
        monkeypatch.setattr(fills_recorder, "load_state",
                            lambda: dict(t0_initialized_at="T",
                                         high_turnover=["0xH"], wallets={}))
        monkeypatch.setattr(fills_recorder, "load_cohort_wallets",
                            lambda: [dict(address="0xH")])
        monkeypatch.setattr(fills_recorder, "poll_wallets",
                            lambda addrs, state, mode: seen.append((addrs, mode)))
        fills_recorder.main(["--intraday"])
        assert seen == [(["0xH"], "intraday")]


class TestEmptyFirstBaseline:
    """빈 최초 응답 기준선 — 이후 만석은 초기 절단이 아니라 post-T0 갭이다."""

    ISO = "2026-08-27T01:00:00Z"
    MS = calendar.timegm(time.strptime(ISO, "%Y-%m-%dT%H:%M:%SZ")) * 1000

    def test_빈_최초_응답은_기준선_시각만_기록한다(self):
        wst: dict = {}
        wst, raw, summary = process_response("0xA", [], wst, self.ISO, "daily", cap=5)
        assert summary["continuity"] == "first" and raw is None
        assert wst["empty_first_at_ms"] == self.MS
        assert "initial_window_truncated" not in wst, "이력 전무 = 완전 관측"
        assert "newest_ts" not in wst, "체결 커서는 실제 체결에서만"

    def test_반복_빈_폴링은_기준선을_앞으로_밀지_않는다(self):
        wst: dict = {}
        wst, _, _ = process_response("0xA", [], wst, self.ISO, "daily", cap=5)
        wst, _, _ = process_response("0xA", [], wst, "2026-08-28T01:00:00Z",
                                     "daily", cap=5)
        assert wst["empty_first_at_ms"] == self.MS, "최초(가장 이른) 기준선 보존"

    def test_빈_기준선_후_미만석_최초체결은_first_무플래그다(self, tmp_path):
        # 미만석 userFills = 지갑의 전체 이력 — 놓친 것 없음
        state = dict(high_turnover=[], wallets={
            "0xA": dict(empty_first_at_ms=1000, status=STATUS_OK)})
        bt_calls: list[int] = []
        r = poll_wallets(["0xA"], state, "daily",
                         fetch=lambda a: [fill(2000, "x")],
                         fetch_by_time=lambda a, s: bt_calls.append(s),
                         fills_dir=tmp_path / "fills",
                         state_f=tmp_path / "s.json", pause_s=0)
        assert bt_calls == [], "미만석은 소급 불필요"
        wst = load_state(tmp_path / "s.json")["wallets"]["0xA"]
        assert wst["newest_ts"] == 2000
        assert "initial_window_truncated" not in wst
        sums = read_jsonl_gz(tmp_path / "fills" / r["day"] / "summary.jsonl.gz")
        assert sums[0]["continuity"] == "first"

    def test_빈_기준선_후_만석은_기준선에서_소급_복구된다(self, tmp_path):
        state = dict(high_turnover=[], wallets={
            "0xA": dict(empty_first_at_ms=1000, status=STATUS_OK)})
        main = [fill(5000 + i, f"m{i}") for i in range(FILLS_RESP_CAP)]
        page1 = [fill(1001 + i, f"b{i}") for i in range(FILLS_RESP_CAP)]  # 만석
        page2 = [fill(3001 + i, f"c{i}") for i in range(99)]              # 미만석
        by_time = {1000: page1, 1001 + FILLS_RESP_CAP - 1: page2}
        r = poll_wallets(["0xA"], state, "daily",
                         fetch=lambda a: main,
                         fetch_by_time=lambda a, s: by_time[s],
                         fills_dir=tmp_path / "fills",
                         state_f=tmp_path / "s.json", pause_s=0)
        assert r["n_backfilled"] == 1 and r["n_censored"] == 0
        wst = load_state(tmp_path / "s.json")["wallets"]["0xA"]
        assert wst["status"] == STATUS_OK
        assert "initial_window_truncated" not in wst, \
            "post-T0 갭 복구 — 초기 절단 플래그가 아니다"
        sums = read_jsonl_gz(tmp_path / "fills" / r["day"] / "summary.jsonl.gz")
        assert sums[0]["continuity"] == "ok" and sums[0]["backfill_n"] == \
            FILLS_RESP_CAP + 99

    def test_빈_기준선_후_만석_윈도_소진은_절단이지_초기플래그가_아니다(self, tmp_path):
        state = dict(high_turnover=[], wallets={
            "0xA": dict(empty_first_at_ms=1000, status=STATUS_OK)})
        main = [fill(50000 + i, f"m{i}") for i in range(FILLS_RESP_CAP)]

        def fbt(addr, start):                    # 항상 만석 — 갭이 안 메워짐
            return [fill(start + 1 + i, f"t{start + 1 + i}")
                    for i in range(FILLS_RESP_CAP)]

        r = poll_wallets(["0xA"], state, "daily", fetch=lambda a: main,
                         fetch_by_time=fbt, fills_dir=tmp_path / "fills",
                         state_f=tmp_path / "s.json", pause_s=0)
        assert r["n_censored"] == 1
        wst = load_state(tmp_path / "s.json")["wallets"]["0xA"]
        assert wst["status"] == STATUS_CENSORED
        assert "initial_window_truncated" not in wst
        sums = read_jsonl_gz(tmp_path / "fills" / r["day"] / "summary.jsonl.gz")
        assert sums[0]["continuity"] == "gap-censored"
