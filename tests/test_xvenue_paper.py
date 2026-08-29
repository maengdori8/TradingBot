"""Track C 교차거래소 페이퍼 러너 테스트 — 네트워크 없이 합성 베뉴로 검증.

대상: carrybot/live/xvenue_paper.py
정정 공시(docs/XVENUE_ARBITRAGE_2026-08.md §C4)가 지적한 4개 결함의 회귀 방지:
1. 재구성 비용이 항상 0 (쓰고-나서-같은-파일-읽기) → Σ|Δw| 실계산·1회만 부과
2. 유니버스 게이트가 24h 거래대금 → 30일 중앙 일거래대금 $5M
3. 거짓 docstring (베이시스 미기록) → basis_diff 실적재
4. k접두 심볼 매핑 누락 (kPEPE→kPEPEUSDT) → 명시 매핑
그리고 결측 종목 동적 재가중 금지·시간순 백필·레거시 계열 동결(연속성).
"""

from __future__ import annotations

import json
import urllib.parse

import pandas as pd
import pytest

import carrybot.live.xvenue_paper as xv

DAY_MS = 86_400_000


def dms(day: str) -> int:
    """'YYYY-MM-DD' → UTC 자정 ms."""
    return int(pd.Timestamp(day, tz="utc").timestamp() * 1000)


# ── 합성 베뉴 ────────────────────────────────────────────────────────────

class FakeVenue:
    """HL info / Bybit v5 응답을 흉내내는 결정적 픽스처."""

    def __init__(self) -> None:
        self.coins: dict[str, dict] = {}
        self.fail_hl: set[str] = set()
        self.fail_by: set[str] = set()
        self.fail_funding: set[str] = set()      # 펀딩 조회만 전송 실패
        self.delisted: set[str] = set()          # 베뉴 목록에서 사라진 코인
        self.meta_fail = False

    def add(self, name: str, sym: str, *, by_ntl: float = 6e6,
            hl_ntl: float = 6e6, px: float = 100.0, by_rate: float = 0.0,
            hl_rate: float = 0.0, first_day: str | None = None,
            skip_funding: tuple = (),
            listed: bool = True, by_ntl_days: dict | None = None,
            by_px_days: dict | None = None, hl_px_days: dict | None = None,
            skip_days: tuple = ()) -> None:
        """코인 1종을 등록한다 (일별 예외는 *_days 로 덮어쓴다)."""
        self.coins[name] = dict(
            sym=sym, by_ntl=by_ntl, hl_ntl=hl_ntl, px=px, by_rate=by_rate,
            hl_rate=hl_rate, first=dms(first_day) if first_day else 0,
            listed=listed, by_ntl_days=by_ntl_days or {},
            by_px_days=by_px_days or {}, hl_px_days=hl_px_days or {},
            skip=set(skip_days), skip_funding=set(skip_funding))

    # -- 내부 --

    def _days(self, spec: dict, start: int, end: int) -> list[int]:
        return [t for t in range(start, end, DAY_MS)
                if t >= spec["first"] and t not in spec["skip"]]

    def _by_coin(self, sym: str) -> tuple[str, dict] | tuple[None, None]:
        for n, s in self.coins.items():
            if s["sym"] == sym:
                return n, s
        return None, None

    # -- HL --

    def post_hl(self, body: dict, retries: int = 4):
        if body["type"] == "metaAndAssetCtxs":
            if self.meta_fail:
                return None
            live = [n for n in self.coins if n not in self.delisted]
            return [{"universe": [{"name": n} for n in live]},
                    [{} for _ in live]]
        if body["type"] == "candleSnapshot":
            req = body["req"]
            coin = req["coin"]
            if coin in self.fail_hl:
                return None
            s = self.coins[coin]
            out = []
            for t in self._days(s, req["startTime"], req["endTime"] + 1):
                c = s["hl_px_days"].get(t, s["px"])
                out.append(dict(t=t, o=c, h=c * 1.01, l=c * 0.99, c=c,
                                v=s["hl_ntl"] / c))
            return out
        if body["type"] == "fundingHistory":
            coin = body["coin"]
            if coin in self.fail_hl or coin in self.fail_funding:
                return None
            s = self.coins[coin]
            out = []
            for t in self._days(s, body["startTime"], body["endTime"] + 1):
                if t in s["skip_funding"]:
                    continue
                out += [dict(time=t + i * 3_600_000,
                             fundingRate=str(s["hl_rate"] / 24))
                        for i in range(24)]
            return out
        raise AssertionError(f"미지원 HL 호출: {body['type']}")

    # -- Bybit --

    def get_bybit(self, url: str, retries: int = 4):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "instruments-info" in url:
            return dict(retCode=0, result=dict(nextPageCursor="", list=[
                dict(symbol=s["sym"], quoteCoin="USDT", status="Trading")
                for n, s in self.coins.items()
                if s["listed"] and n not in self.delisted]))
        sym = q["symbol"][0]
        if sym in self.fail_by:
            return None
        name, s = self._by_coin(sym)
        if s is None:
            return dict(retCode=0, result=dict(list=[]))
        start, end = int(q["start"][0] if "start" in q else q["startTime"][0]), None
        end = int(q["end"][0] if "end" in q else q["endTime"][0]) + 1
        if "kline" in url:
            rows = []
            for t in self._days(s, start, end):
                c = s["by_px_days"].get(t, s["px"])
                rows.append([str(t), str(c), str(c * 1.01), str(c * 0.99), str(c),
                             "1", str(s["by_ntl_days"].get(t, s["by_ntl"]))])
            return dict(retCode=0, result=dict(list=list(reversed(rows))))
        if "funding/history" in url:
            rows = []
            for t in self._days(s, start, end):
                if t in s["skip_funding"]:
                    continue
                rows += [dict(fundingRateTimestamp=str(t + i * 28_800_000),
                              fundingRate=str(s["by_rate"] / 3)) for i in range(3)]
            return dict(retCode=0, result=dict(list=rows))
        raise AssertionError(f"미지원 Bybit 호출: {url}")


@pytest.fixture
def venue(monkeypatch, tmp_path):
    """합성 베뉴 + 임시 로그 경로로 모듈을 격리한다."""
    v = FakeVenue()
    monkeypatch.setattr(xv, "_post_hl", v.post_hl)
    monkeypatch.setattr(xv, "_get_bybit", v.get_bybit)
    monkeypatch.setattr(xv, "HL_SLEEP", 0.0)
    monkeypatch.setattr(xv, "BY_SLEEP", 0.0)
    monkeypatch.setattr(xv, "STATE", tmp_path / "trackc_state.json")
    monkeypatch.setattr(xv, "HIST", tmp_path / "trackc_history.csv")
    monkeypatch.setattr(xv, "UNIVERSE_F", tmp_path / "trackc_universe.json")
    return v


def at(monkeypatch, when: str):
    """모듈 시계를 고정한다."""
    monkeypatch.setattr(xv, "_now_utc", lambda: pd.Timestamp(when, tz="utc"))


def rows_of() -> list[dict]:
    """기록된 이력 행."""
    return pd.read_csv(xv.HIST).to_dict("records")


def state_of() -> dict:
    """기록된 상태."""
    return json.loads(xv.STATE.read_text())


# ── 결함 4: k접두 심볼 매핑 ──────────────────────────────────────────────

class TestSymbolMapping:
    """HL 코인명 → Bybit 심볼. 추측 매핑 금지."""

    AV = {"BTCUSDT", "1000PEPEUSDT", "SHIB1000USDT", "1000BONKUSDT",
          "1000NEIROCTOUSDT"}

    def test_plain_and_k_prefix(self):
        assert xv.bybit_symbol("BTC", self.AV) == "BTCUSDT"
        # 결함 4 회귀: 예전 코드는 f"{coin}USDT" → 'kPEPEUSDT' (없는 심볼)
        assert xv.bybit_symbol("kPEPE", self.AV) == "1000PEPEUSDT"
        assert xv.bybit_symbol("kSHIB", self.AV) == "SHIB1000USDT"  # 접미형
        assert xv.bybit_symbol("kBONK", self.AV) == "1000BONKUSDT"

    def test_ambiguous_and_missing_are_excluded(self):
        # kNEIRO 는 Bybit 에 1000NEIROCTOUSDT(다른 토큰)뿐 → 짝짓지 않는다
        assert xv.bybit_symbol("kNEIRO", self.AV) is None
        assert xv.bybit_symbol("kFLOKI", self.AV) is None      # 미상장
        assert xv.bybit_symbol("NOPE", self.AV) is None

    def test_unregistered_k_coin_is_not_guessed(self):
        assert xv.bybit_symbol("kFOO", {"1000FOOUSDT"}) is None


# ── 결함 2: 유니버스 게이트 (30일 중앙 일거래대금) ────────────────────────

class TestUniverseGate:
    """사전등록 게이트 = 양쪽 30일 중앙 $5M. 24h 값이 아니다."""

    AS_OF = pd.Timestamp("2026-09-01", tz="utc")

    def _build(self, v):
        return xv.build_universe(self.AS_OF, deadline=1e18)

    def test_24h_spike_does_not_qualify(self, venue):
        """직전 1일만 폭증한 코인은 24h 게이트는 통과해도 중앙값에서 탈락."""
        spike = {dms("2026-08-31"): 900e6}
        venue.add("STEADY", "STEADYUSDT", by_ntl=6e6, hl_ntl=6e6)
        venue.add("SPIKE", "SPIKEUSDT", by_ntl=1e6, hl_ntl=9e6,
                  by_ntl_days=spike)
        snap, unknown = self._build(venue)
        assert unknown == []
        assert set(snap["coins"]) == {"STEADY"}
        assert snap["medians"]["SPIKE"]["by"] == 1e6          # 중앙값은 스파이크 무시

    def test_one_sided_liquidity_fails(self, venue):
        venue.add("BOTH", "BOTHUSDT", by_ntl=6e6, hl_ntl=6e6)
        venue.add("BYONLY", "BYONLYUSDT", by_ntl=9e6, hl_ntl=1e6)
        venue.add("HLONLY", "HLONLYUSDT", by_ntl=1e6, hl_ntl=9e6)
        snap, _ = self._build(venue)
        assert set(snap["coins"]) == {"BOTH"}
        # AND 게이트 단락평가: Bybit 미달이면 HL 은 조회하지 않는다
        assert "hl" not in snap["medians"]["HLONLY"]

    def test_short_history_is_ineligible(self, venue):
        venue.add("OK", "OKUSDT")
        venue.add("NEW", "NEWUSDT", first_day="2026-08-20")   # 30일 미만
        snap, _ = self._build(venue)
        assert set(snap["coins"]) == {"OK"}
        assert snap["medians"]["NEW"]["by"] is None
        assert snap["medians"]["NEW"]["by_days"] < xv.MEDIAN_WINDOW

    def test_transport_failure_is_unknown_not_ineligible(self, venue):
        venue.add("OK", "OKUSDT")
        venue.add("FLAKY", "FLAKYUSDT")
        venue.fail_by.add("FLAKYUSDT")
        snap, unknown = self._build(venue)
        assert snap is None and unknown == ["FLAKY"]          # fail-closed

    def test_hl_transport_failure_blocks(self, venue):
        venue.add("OK", "OKUSDT")
        venue.add("HLDOWN", "HLDOWNUSDT")
        venue.fail_hl.add("HLDOWN")
        snap, unknown = self._build(venue)
        assert snap is None and unknown == ["HLDOWN"]

    def test_lookahead_free_window(self, venue):
        """as_of 당일 거래대금은 게이트에 들어가지 않는다."""
        venue.add("X", "XUSDT", by_ntl=1e6, hl_ntl=9e6,
                  by_ntl_days={dms("2026-09-01"): 900e6})
        snap, _ = self._build(venue)
        assert snap is None or "X" not in (snap or {}).get("coins", {})

    def test_hl_notional_bounds_recorded(self, venue):
        venue.add("OK", "OKUSDT", by_ntl=6e6, hl_ntl=6e6)
        snap, _ = self._build(venue)
        m = snap["medians"]["OK"]
        assert m["hl_low"] < m["hl"] < m["hl_high"]           # estimator 불확실성 공시
        assert snap["estimator"].startswith("bybit_turnover")


# ── 결함 1: 재구성 비용 ──────────────────────────────────────────────────

def prev_book(coins: str, px: float = 100.0) -> dict:
    """등가중 고정 수량 원장 (양다리 동일가)."""
    w = 1.0 / len(coins)
    return dict(coins={c: f"{c}USDT" for c in coins},
                positions={c: dict(w=w, b_ref=px, h_ref=px,
                                   n_b=w / px, n_h=w / px) for c in coins})


class TestRebalanceCost:
    """(Σ|Δw_B|+Σ|Δw_H|)/2 — 생존 드리프트·총명목 드리프트 포함, 유니버스당 1회."""

    def test_first_build_charges_full_entry(self):
        gross, det = xv._transition_cost(None, {"A": "a", "B": "b"}, {}, {})
        assert gross == pytest.approx(1.0)                   # 진입 12bp
        assert det["cost_bp"] == pytest.approx(12.0)

    def test_shrinking_universe_charges_survivor_drift(self):
        """37→15 형태: 제거·추가뿐 아니라 생존 종목 재조정도 실제 거래다."""
        px = {c: 100.0 for c in "ABCDE"}
        gross, det = xv._transition_cost(prev_book("ABCD"), {"A": "a", "E": "e"},
                                         px, px)
        # 생존 A: |1/2−1/4|, 신규 E: 1/2, 제거 B/C/D: 3×1/4
        assert gross == pytest.approx(0.25 + 0.5 + 0.75)
        assert det["added"] == ["E"] and det["removed"] == ["B", "C", "D"]
        assert det["survived"] == 1

    def test_cost_is_never_silently_zero(self):
        """결함 1 회귀: prev 를 신규와 동일하게 읽어 비용이 0 이 되던 경로."""
        px = {c: 100.0 for c in "ABC"}
        gross, _ = xv._transition_cost(prev_book("AB"), {"A": "a", "B": "b"}, px, px)
        assert gross == pytest.approx(0.0)          # 진짜 무변화일 때만 0
        gross2, _ = xv._transition_cost(prev_book("AB"), {"A": "a", "C": "c"},
                                        px, px)
        assert gross2 > 0.9

    def test_hl_leg_divergence_is_charged(self):
        """Bybit 다리만으로 재면 HL 쪽 발산이 비용에서 사라진다."""
        flat = {c: 100.0 for c in "AB"}
        moved = {"A": 200.0, "B": 100.0}
        gross, det = xv._transition_cost(prev_book("AB"), {"A": "a", "B": "b"},
                                         flat, moved)
        assert det["gross_b"] == pytest.approx(0.0)
        assert det["gross_h"] == pytest.approx(0.5)          # A 다리 0.5→1.0
        assert gross == pytest.approx(0.25)

    def test_total_notional_drift_is_charged(self):
        """전 종목 동반 상승 → 상대비중은 같아도 총명목 축소 거래가 필요하다."""
        doubled = {c: 200.0 for c in "AB"}
        gross, _ = xv._transition_cost(prev_book("AB"), {"A": "a", "B": "b"},
                                       doubled, doubled)
        assert gross == pytest.approx(1.0)          # 정규화했다면 0 으로 사라졌다


# ── 결함 3 + 회계: 일별 적재 ─────────────────────────────────────────────

class TestDailyRun:
    """main() 의 적재·멱등·백필·연속성."""

    def _two_coins(self, venue, **kw):
        venue.add("AAA", "AAAUSDT", **kw)
        venue.add("BBB", "BBBUSDT", **kw)

    def test_first_run_anchors_and_charges_entry(self, venue, monkeypatch):
        self._two_coins(venue, hl_rate=0.0003, by_rate=0.0001)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        rows = rows_of()
        assert [r["row_type"] for r in rows] == ["anchor", "daily"]
        assert rows[0]["equity"] == 1.0 and rows[0]["day"] == "2026-08-31"
        assert rows[1]["day"] == "2026-09-01"
        # 진입 12bp 부과 + 펀딩 수취 (HL 숏 수취 0.03% − Bybit 롱 지불 0.01%)
        base = 1.0 - 0.0012
        assert rows[1]["cost"] == pytest.approx(0.0012)
        assert rows[1]["notional_base"] == pytest.approx(base)
        assert rows[1]["day_diff"] == pytest.approx(base * 0.0002, rel=1e-9)
        assert rows[1]["equity"] == pytest.approx(base + base * 0.0002, rel=1e-12)
        st = state_of()
        assert st["spec_t0"] == "2026-09-01"
        assert st["verdict_day"] == "2026-11-30"
        assert st["equity"] == pytest.approx(rows[1]["equity"])

    def test_basis_is_actually_recorded(self, venue, monkeypatch):
        """결함 3 회귀: docstring 이 약속한 베이시스가 실제로 적재된다."""
        venue.add("AAA", "AAAUSDT", px=100.0,
                  by_px_days={dms("2026-09-01"): 101.0})     # Bybit 만 +1%
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        row = rows_of()[-1]
        assert "basis_diff" in row
        assert row["basis_diff"] == pytest.approx((1 - 0.0012) * 0.01, rel=1e-9)
        assert row["equity"] != row["equity_funding"]        # 두 계열 분리 적재

    def test_fixed_quantity_pnl_is_additive_not_compounded(self, venue, monkeypatch):
        """월간 고정 수량이면 일별 손익은 가산이다 — 복리로 부풀리면 안 된다."""
        venue.add("AAA", "AAAUSDT", px=100.0, by_px_days={
            dms("2026-09-01"): 101.0, dms("2026-09-02"): 102.0})
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()                                            # T0 = 2026-09-01
        at(monkeypatch, "2026-09-03T00:35Z")
        xv.main()
        rows = [r for r in rows_of() if r["row_type"] == "daily"]
        base = 1.0 - 0.0012
        assert [r["basis_diff"] for r in rows] == [
            pytest.approx(base * 0.01, rel=1e-9)] * 2        # 고정 수량 → 동일 증분
        assert rows[-1]["equity"] == pytest.approx(base * 1.02, rel=1e-12)
        assert rows[-1]["equity"] != pytest.approx(base * 1.01 * 1.01, rel=1e-12)

    def test_idempotent_rerun(self, venue, monkeypatch):
        self._two_coins(venue, hl_rate=0.0003)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        first = rows_of()
        xv.main()
        assert rows_of() == first                            # 중복 적재·중복 비용 없음

    def test_incomplete_day_blocks_then_backfills(self, venue, monkeypatch):
        """결측 종목 동적 재가중 금지 — 보류 후 시간순 백필."""
        self._two_coins(venue, hl_rate=0.0002)
        venue.fail_funding.add("BBB")            # 유니버스는 서지만 그날이 불완전
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        assert not any(r["row_type"] == "daily" for r in rows_of())
        st = state_of()
        assert st["blocked"]["reason_code"] == "DAY_INCOMPLETE"
        assert st["blocked"]["unknown_coins"] == ["BBB"]
        assert st["blocked"]["retry_count"] == 1

        venue.fail_funding.clear()
        at(monkeypatch, "2026-09-03T00:35Z")
        xv.main()
        days = [r["day"] for r in rows_of() if r["row_type"] == "daily"]
        assert days == ["2026-09-01", "2026-09-02"]          # 빠진 날 소급 적재
        assert "blocked" not in state_of()

    def test_universe_unknown_blocks_without_writing_universe(self, venue,
                                                              monkeypatch):
        self._two_coins(venue)
        venue.fail_by.add("BBBUSDT")
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        assert state_of()["blocked"]["reason_code"] == "UNIVERSE_UNKNOWN"
        assert not xv.UNIVERSE_F.exists()                    # 열화 유니버스 미기록

    def test_monthly_rebalance_charges_once(self, venue, monkeypatch):
        """월 경계에서 1회만 전환비용, 같은 유니버스에서는 재부과 없음."""
        venue.add("AAA", "AAAUSDT")
        # BBB 는 9월 중순까지만 유동적 → 9월 유니버스엔 들고 10월엔 빠진다
        dry = {dms(f"2026-09-{d:02d}"): 9e6 for d in range(1, 15)}
        dry.update({dms(f"2026-08-{d:02d}"): 9e6 for d in range(25, 32)})
        venue.add("BBB", "BBBUSDT", by_ntl=1e6, by_ntl_days=dry)
        at(monkeypatch, "2026-09-30T00:35Z")
        xv.main()
        assert rows_of()[-1]["n_coins"] == 2
        at(monkeypatch, "2026-10-03T00:35Z")
        xv.main()
        rows = [r for r in rows_of() if r["row_type"] == "daily"]
        costs = {r["day"]: r["cost"] for r in rows}
        assert costs["2026-09-29"] == pytest.approx(0.0012)  # 최초 진입 (2종)
        assert costs["2026-09-30"] == 0.0
        # 10월 유니버스는 1종 — 생존 AAA 비중 0.5→1.0 + BBB 청산 0.5 = Σ|Δw| 1.0
        # 비용은 절대액이라 직전 자본(진입비용 차감분)에 비례한다
        assert costs["2026-10-01"] == pytest.approx((1 - 0.0012) * 0.0012, rel=1e-9)
        assert costs["2026-10-02"] == 0.0
        assert rows[-1]["n_coins"] == 1

    def test_legacy_series_is_frozen_not_mixed(self, venue, monkeypatch):
        """정정 이전 계열은 state 에 동결하고 spec 계열은 1.0 에서 새로 시작."""
        xv.HIST.parent.mkdir(parents=True, exist_ok=True)
        xv.HIST.write_text("day,equity,day_diff,n_coins\n"
                           "2026-08-25,0.99890525,0.00010538,37\n"
                           "2026-08-28,0.99903532,0.00013022,31\n")
        self._two_coins(venue, hl_rate=0.0002)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        st = state_of()
        assert st["legacy"]["phase"] == "legacy_invalid"
        assert [r["day"] for r in st["legacy"]["rows"]] == ["2026-08-25", "2026-08-28"]
        assert st["legacy"]["sha256"]
        rows = rows_of()
        assert rows[0]["equity"] == 1.0                      # 대시보드 base = 1.0
        assert all(r["phase"] == "spec_v2" for r in rows)
        assert "2026-08-25" not in [r["day"] for r in rows]

    def test_legacy_migration_is_idempotent(self, venue, monkeypatch):
        xv.HIST.parent.mkdir(parents=True, exist_ok=True)
        xv.HIST.write_text("day,equity,day_diff,n_coins\n"
                           "2026-08-25,0.99890525,0.00010538,37\n")
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        at(monkeypatch, "2026-09-02T01:35Z")
        xv.main()
        assert len(state_of()["legacy"]["rows"]) == 1

    def test_unexpected_legacy_shape_raises(self, venue, monkeypatch):
        xv.HIST.parent.mkdir(parents=True, exist_ok=True)
        xv.HIST.write_text("day,equity,day_diff,n_coins\n" + "".join(
            f"2026-08-{d:02d},1.0,0.0,37\n" for d in range(1, 12)))
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        with pytest.raises(xv.LedgerError):
            xv.main()

    def test_old_rule_universe_is_rebuilt(self, venue, monkeypatch):
        """구 규칙(24h) 유니버스 파일은 즉시 폐기하고 사양대로 재구축한다."""
        xv.UNIVERSE_F.parent.mkdir(parents=True, exist_ok=True)
        xv.UNIVERSE_F.write_text(json.dumps(["AAA", "BBB", "GONE"]))
        venue.add("AAA", "AAAUSDT")
        venue.add("GONE", "GONEUSDT", by_ntl=1e6)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        book = json.loads(xv.UNIVERSE_F.read_text())
        assert book["rule"] == xv.UNIVERSE_RULE
        snap = book["snapshots"][book["active"]]
        assert set(snap["coins"]) == {"AAA"}
        assert snap["prev_id"] is None                       # 구 장부는 이월 불가

    def test_missing_data_is_never_imputed_as_zero(self, venue, monkeypatch):
        """오래된 날이어도 결측을 0 으로 확정하지 않는다 (손실 은폐 경로)."""
        self._two_coins(venue, hl_rate=0.0002)
        venue.coins["BBB"]["skip"].add(dms("2026-09-01"))    # 상장중인데 캔들 부재
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        assert state_of()["blocked"]["reason_code"] == "DAY_INCOMPLETE"
        at(monkeypatch, "2026-09-10T00:35Z")                 # 9일 지나도 마찬가지
        xv.main()
        assert not any(r["row_type"] == "daily" for r in rows_of())
        assert state_of()["blocked"]["reason_code"] == "DAY_INCOMPLETE"
        assert state_of()["blocked"]["retry_count"] == 2

    def test_delisting_forces_exit_and_continues(self, venue, monkeypatch):
        """상장폐지만 예외 — 마지막 마크로 강제 청산하고 계속 간다."""
        self._two_coins(venue, hl_rate=0.0002)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        venue.delisted.add("BBB")
        venue.coins["BBB"]["skip"].add(dms("2026-09-02"))
        at(monkeypatch, "2026-09-03T00:35Z")
        xv.main()
        rows = [r for r in rows_of() if r["row_type"] == "daily"]
        assert [r["day"] for r in rows] == ["2026-09-01", "2026-09-02"]
        assert rows[-1]["n_coins"] == 1
        base = 1 - 0.0012
        # 청산 명목 = 절반 → 비용은 **기존 명목 기준**에 대한 절대액
        assert rows[-1]["cost"] == pytest.approx(base * 0.5 * 0.0012, rel=1e-9)
        # 생존 수량은 그대로이므로 명목 기준을 재설정하지 않는다 (무상 증액 금지)
        assert rows[-1]["notional_base"] == pytest.approx(base)
        assert "blocked" not in state_of()

    def test_delist_exit_uses_current_mark_not_entry_price(self, venue, monkeypatch):
        """청산 명목은 진입가가 아니라 마지막 체결가로 잰다."""
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        venue.coins["BBB"]["by_px_days"][dms("2026-09-02")] = 200.0   # 2배 후 폐지
        venue.coins["BBB"]["hl_px_days"][dms("2026-09-02")] = 200.0
        venue.delisted.add("BBB")
        venue.coins["BBB"]["skip"].add(dms("2026-09-03"))
        at(monkeypatch, "2026-09-04T00:35Z")
        xv.main()
        exit_row = [r for r in rows_of() if r["day"] == "2026-09-03"][0]
        base = 1 - 0.0012
        # 진입가 기준이면 0.5, 마지막 체결가(2배) 기준이면 1.0
        assert exit_row["cost"] == pytest.approx(base * 1.0 * 0.0012, rel=1e-6)

    def test_delist_books_final_leg_pnl(self, venue, monkeypatch):
        """청산 구간(직전 종가 → 청산가) 손익을 누락하면 최종 손익이 사라진다."""
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()                                        # T0 = 09-01, 기준가 100
        # 09-02: Bybit 만 +10% 후 폐지 (펀딩 미게시 → 그날 보류 → 강제청산 경로)
        venue.coins["BBB"]["by_px_days"][dms("2026-09-02")] = 110.0
        venue.coins["BBB"]["skip_funding"].add(dms("2026-09-02"))
        venue.delisted.add("BBB")
        at(monkeypatch, "2026-09-03T00:35Z")
        xv.main()
        row = [r for r in rows_of() if r["day"] == "2026-09-02"][0]
        base = 1 - 0.0012
        # BBB 청산 베이시스 = n_b×(110−100) = (0.5/100)×10 = 0.05
        assert row["basis_diff"] == pytest.approx(base * 0.05, rel=1e-6)
        assert row["n_coins"] == 1

    def test_full_delisting_becomes_cash_and_continues(self, venue, monkeypatch):
        """전 종목 폐지도 유효한 상태(전액 현금) — 영구 정지시키지 않는다."""
        self._two_coins(venue, hl_rate=0.0002)
        venue.add("CCC", "CCCUSDT", by_ntl=1e6)          # 유동성 미달 — 상장은 유지
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        venue.delisted.update({"AAA", "BBB"})
        for c in ("AAA", "BBB"):
            venue.coins[c]["skip"].add(dms("2026-09-02"))
        at(monkeypatch, "2026-09-03T00:35Z")
        xv.main()
        rows = [r for r in rows_of() if r["row_type"] == "daily"]
        assert rows[-1]["day"] == "2026-09-02" and rows[-1]["n_coins"] == 0
        assert rows[-1]["day_diff"] == 0.0 and rows[-1]["basis_diff"] == 0.0
        assert "blocked" not in state_of()

    def test_delist_without_exit_mark_blocks(self, venue, monkeypatch):
        """청산 가격을 못 구하면 회계를 완료하지 않는다."""
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        venue.delisted.add("BBB")
        venue.fail_by.add("BBBUSDT")                     # 마크 조회 자체가 실패
        at(monkeypatch, "2026-09-03T00:35Z")
        xv.main()
        assert state_of()["blocked"]["reason_code"] == "DELIST_EXIT_PRICE_PENDING"

    def test_spec_t0_recovers_from_anchor(self, venue, monkeypatch):
        """state 를 잃어도 T0 는 이력 앵커에서 복원되고 이동하지 않는다."""
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        xv.STATE.unlink()
        at(monkeypatch, "2026-09-05T00:35Z")
        xv.main()
        assert state_of()["spec_t0"] == "2026-09-01"
        assert state_of()["verdict_day"] == "2026-11-30"

    def test_t0_mismatch_raises(self, venue, monkeypatch):
        self._two_coins(venue)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()
        st = state_of()
        st["spec_t0"] = "2026-09-04"                         # 시계 임의 이동 시도
        xv.STATE.write_text(json.dumps(st))
        with pytest.raises(xv.LedgerError):
            xv.main()

    def test_backlog_keeps_blocked_marker(self, venue, monkeypatch):
        """백필 상한을 다 채웠는데 잔여가 있으면 초록으로 끝내지 않는다."""
        self._two_coins(venue)
        monkeypatch.setattr(xv, "MAX_BACKFILL_DAYS", 2)
        at(monkeypatch, "2026-09-02T00:35Z")
        xv.main()                                            # T0 = 2026-09-01
        at(monkeypatch, "2026-09-06T00:35Z")                 # 4일 밀림, 상한 2일
        xv.main()
        st = state_of()
        assert st["blocked"]["reason_code"] == "BACKLOG"
        assert st["blocked"]["pending_days"] == ["2026-09-04", "2026-09-05"]
        assert st["blocked"]["last_success_day"] == "2026-09-03"

    def test_universe_file_keeps_previous_snapshot(self, venue, monkeypatch):
        """쓰고-나서-같은-파일-읽기 금지: 직전 스냅샷이 파일 안에 남는다."""
        venue.add("AAA", "AAAUSDT")
        venue.add("BBB", "BBBUSDT", by_ntl=1e6)
        at(monkeypatch, "2026-09-30T00:35Z")
        xv.main()
        at(monkeypatch, "2026-10-02T00:35Z")
        xv.main()
        book = json.loads(xv.UNIVERSE_F.read_text())
        assert len(book["order"]) == 2
        assert book["snapshots"][book["active"]]["prev_id"] == book["order"][0]


# ── 원장 불변식 · 부분 응답 ──────────────────────────────────────────────

class TestLedgerInvariants:
    """조용히 넘어가면 안 되는 상태는 LedgerError 로 멈춘다."""

    def _row(self, day, uid="u1", cost=0.0):
        return dict(day=day, row_type="daily", universe_id=uid, cost=cost)

    def test_duplicate_day_raises(self):
        with pytest.raises(xv.LedgerError):
            xv.audit_rows([self._row("2026-09-01"), self._row("2026-09-01")])

    def test_out_of_order_raises(self):
        with pytest.raises(xv.LedgerError):
            xv.audit_rows([self._row("2026-09-02"), self._row("2026-09-01")])

    def test_double_charged_universe_raises(self):
        with pytest.raises(xv.LedgerError):
            xv.audit_rows([self._row("2026-09-01", cost=0.0012),
                           self._row("2026-09-02", cost=0.0012)])

    def test_clean_ledger_passes(self):
        xv.audit_rows([self._row("2026-09-01", cost=0.0012),
                       self._row("2026-09-02"),
                       self._row("2026-09-03", uid="u2", cost=0.0006)])


class TestPartialResponses:
    """부분 응답을 완전한 것으로 오인하면 멀쩡한 종목이 '미상장'이 된다."""

    def test_unfinished_pagination_is_unknown(self, monkeypatch):
        monkeypatch.setattr(xv, "BY_SLEEP", 0.0)
        monkeypatch.setattr(xv, "_get_bybit", lambda url, retries=4: dict(
            retCode=0, result=dict(nextPageCursor="more", list=[
                dict(symbol="AAAUSDT", quoteCoin="USDT", status="Trading")])))
        assert xv.bybit_linear_usdt() is None

    def test_complete_pagination_returns_set(self, monkeypatch):
        monkeypatch.setattr(xv, "BY_SLEEP", 0.0)
        monkeypatch.setattr(xv, "_get_bybit", lambda url, retries=4: dict(
            retCode=0, result=dict(nextPageCursor="", list=[
                dict(symbol="AAAUSDT", quoteCoin="USDT", status="Trading"),
                dict(symbol="OLDUSDT", quoteCoin="USDT", status="Delivering")])))
        assert xv.bybit_linear_usdt() == {"AAAUSDT"}


class TestExposureScaling:
    """기존 노출은 직전 명목 기준 단위, 신규 비중은 새 기준 단위 — 환산 필요."""

    def test_scale_converts_prior_exposure(self):
        px = {c: 100.0 for c in "AB"}
        # 강제청산으로 절반이 현금이 된 상태를 흉내: 남은 노출을 절반으로 환산
        gross_full, _ = xv._transition_cost(prev_book("AB"), {"A": "a", "B": "b"},
                                            px, px, scale=1.0)
        gross_half, det = xv._transition_cost(prev_book("AB"), {"A": "a", "B": "b"},
                                              px, px, scale=0.5)
        assert gross_full == pytest.approx(0.0)      # 변화 없음 → 거래 없음
        assert gross_half == pytest.approx(0.5)      # 현금 절반 재투입분이 잡힌다
        assert det["scale"] == pytest.approx(0.5)


class TestDelistPriorMark:
    """청산 기준가는 직전 회계 종가여야 한다 (진입가 대체는 이중계상)."""

    def _snap(self, as_of: str) -> dict:
        return dict(as_of=as_of, coins={"BBB": "BBBUSDT"},
                    positions={"BBB": dict(w=1.0, b_ref=100.0, h_ref=100.0,
                                           n_b=0.01, n_h=0.01)})

    def test_missing_prior_candle_blocks_after_first_day(self, venue, monkeypatch):
        venue.add("BBB", "BBBUSDT", px=100.0,
                  by_px_days={dms("2026-09-05"): 130.0},
                  skip_days=(dms("2026-09-04"),))          # 전일 캔들 부재
        ex, missing = xv.delist_exit(self._snap("2026-09-01"),
                                     ["BBB"], pd.Timestamp("2026-09-05", tz="utc"))
        assert ex is None and missing == ["BBB"]

    def test_first_day_may_use_rebalance_ref(self, venue, monkeypatch):
        venue.add("BBB", "BBBUSDT", px=100.0,
                  by_px_days={dms("2026-09-05"): 130.0},
                  skip_days=(dms("2026-09-04"),))
        ex, _ = xv.delist_exit(self._snap("2026-09-05"),
                               ["BBB"], pd.Timestamp("2026-09-05", tz="utc"))
        assert ex["basis"] == pytest.approx(0.01 * (130.0 - 100.0))
