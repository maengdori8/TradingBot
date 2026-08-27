"""lab/tracke_null.py 단위 테스트 — 합성 데이터로 판정 파이프라인 검증."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from lab.tracke_null import (
    CELLS,
    CELL_CAPITAL,
    VERDICT_EXCEED,
    VERDICT_NULL,
    bootstrap_max_dist,
    build_daily_matrix,
    center_gross,
    check_judgment_day,
    choose_verdict,
    load_ledger,
    mc_p_value,
    observed_max,
    parse_ts,
    resolve_window,
    stationary_block_indices,
    upper_quantile,
)


def _row(cell="E01", ts="2026-09-01T05:00:00Z", gross=0.0, cost=0.0,
         funding=0.0, symbol="BTC", strategy="BRK24", action="exit"):
    """유일키 열을 전부 갖춘 원장 행 dict."""
    return dict(cell=cell, symbol=symbol, strategy=strategy, bar_close=ts,
                action=action, gross_pnl=gross, cost=cost, funding=funding)


def _ledger_csv(tmp_path, rows):
    """행 목록을 원장 CSV로 저장하고 경로를 돌려준다."""
    p = tmp_path / "tracke_ledger.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


class TestLedgerAndMatrix:
    def test_일별_행렬은_고정_창_전_셀_그리드로_구성된다(self, tmp_path):
        p = _ledger_csv(tmp_path, [
            _row("E01", "2026-09-01T05:00:00Z", gross=100.0, cost=8.0, funding=2.0),
            _row("E01", "2026-09-01T09:00:00Z", gross=-40.0, cost=4.0,
                 symbol="ETH"),
            _row("E10", "2026-09-03T01:00:00Z", gross=50.0, cost=5.0,
                 funding=-1.0, symbol="XRP", strategy="RSI-DIV"),
        ])
        led = load_ledger(p)
        days, gross, drag = build_daily_matrix(
            led, date(2026, 9, 1), date(2026, 9, 3))
        assert days == ["2026-09-01", "2026-09-02", "2026-09-03"]
        assert gross.shape == (3, 10) and drag.shape == (3, 10)
        # E01: 같은 날 이벤트 합산 (100-40)/10000
        assert gross[0, 0] == pytest.approx(60.0 / CELL_CAPITAL)
        # drag = (펀딩 − 비용)/자본 = (2-12)/10000
        assert drag[0, 0] == pytest.approx(-10.0 / CELL_CAPITAL)
        # 이벤트 없는 날/셀은 0 (보간 없음)
        assert gross[1].sum() == 0.0 and drag[1].sum() == 0.0
        assert gross[2, 9] == pytest.approx(50.0 / CELL_CAPITAL)
        assert drag[2, 9] == pytest.approx(-6.0 / CELL_CAPITAL)

    def test_컷오프_이후_이벤트는_제외되고_빈_원장도_유효하다(self, tmp_path):
        p = _ledger_csv(tmp_path, [
            _row("E01", "2026-09-05T05:00:00Z", gross=999.0, cost=1.0),
        ])
        led = load_ledger(p)
        days, gross, drag = build_daily_matrix(
            led, date(2026, 9, 1), date(2026, 9, 4))   # 9/5 는 창 밖
        assert len(days) == 4
        assert gross.sum() == 0.0 and drag.sum() == 0.0   # 전부 0 = 유효 결과

    def test_필수_열이_없으면_fail_closed(self, tmp_path):
        p = _ledger_csv(tmp_path, [
            dict(cell="E01", bar_close="2026-09-01T00:00:00Z", pnl_total=1.0),
        ])
        with pytest.raises(ValueError, match="필수 열"):
            load_ledger(p)

    def test_ISO와_epoch_ms_혼합_열도_행별로_정확히_파싱된다(self, tmp_path):
        # 일괄 파싱이면 epoch 문자열이 1970년으로 오염되는 함정 — 행별 파싱 검증
        p = _ledger_csv(tmp_path, [
            _row("E01", "2026-09-08T03:00:00Z", gross=1.0),
            _row("E02", "1788825600000", gross=10.0, cost=1.0),  # 2026-09-08Z
        ])
        led = load_ledger(p)
        assert set(led["day"]) == {"2026-09-08"}
        days, gross, _ = build_daily_matrix(led, date(2026, 9, 8),
                                            date(2026, 9, 8))
        assert days == ["2026-09-08"]
        assert gross[0, 1] == pytest.approx(10.0 / CELL_CAPITAL)

    def test_시각_오염은_fail_closed(self, tmp_path):
        p = _ledger_csv(tmp_path, [_row(ts="말도안되는값")])
        with pytest.raises(ValueError, match="시각 파싱 실패"):
            load_ledger(p)
        with pytest.raises(ValueError, match="범위 이상"):
            parse_ts("1999-01-01T00:00:00Z")           # [2020,2100] 밖

    def test_비유한_수치는_fail_closed(self, tmp_path):
        p = _ledger_csv(tmp_path, [_row(gross="inf")])
        with pytest.raises(ValueError, match="비유한값"):
            load_ledger(p)
        p2 = _ledger_csv(tmp_path, [_row(cost="깨진값")])
        with pytest.raises(ValueError, match="비유한값"):
            load_ledger(p2)

    def test_알_수_없는_셀은_fail_closed(self, tmp_path):
        p = _ledger_csv(tmp_path, [_row(cell="E99")])
        with pytest.raises(ValueError, match="알 수 없는 셀"):
            load_ledger(p)

    def test_유일키_중복은_fail_closed(self, tmp_path):
        r = _row()
        p = _ledger_csv(tmp_path, [r, dict(r)])
        with pytest.raises(ValueError, match="중복"):
            load_ledger(p)

    def test_MTM_스냅샷_행은_fail_closed(self, tmp_path):
        p = _ledger_csv(tmp_path, [_row(action="mtm")])
        with pytest.raises(ValueError, match="MTM"):
            load_ledger(p)

    def test_공식_모드는_funding_열과_유일키를_요구한다(self, tmp_path):
        rows = [_row()]
        no_fund = [{k: v for k, v in r.items() if k != "funding"}
                   for r in rows]
        p = _ledger_csv(tmp_path, no_fund)
        with pytest.raises(ValueError, match="funding"):
            load_ledger(p, official=True)
        load_ledger(p, official=False)                 # 리허설은 경고로 완화
        no_key = [{k: v for k, v in r.items() if k != "symbol"} for r in rows]
        p2 = _ledger_csv(tmp_path, no_key)
        with pytest.raises(ValueError, match="유일키"):
            load_ledger(p2, official=True)

    def test_유일키_공란은_공식_모드에서_거부한다(self, tmp_path):
        p = _ledger_csv(tmp_path, [_row(symbol="")])
        with pytest.raises(ValueError, match="비어 있는"):
            load_ledger(p, official=True)
        load_ledger(p, official=False)                 # 리허설은 경고로 완화

    def test_엔진_실스키마를_읽고_funding_행을_재분류한다(self, tmp_path):
        # scalp_farm_runner 실제 스키마: cell,sym,strategy,bar_close(epoch ms),
        # action,price,qty,pnl(gross),cost,direction — funding 은 action 행
        base = dict(cell="E05", sym="BTC", strategy="BRK96", price=100.0,
                    qty=1.0, direction=1)
        p = _ledger_csv(tmp_path, [
            dict(base, bar_close=1788829200000, action="exit",
                 pnl=120.0, cost=8.0),                 # 2026-09-08T01:00Z
            dict(base, bar_close=1788854400000, action="funding",
                 pnl=-4.0, cost=0.0),                  # 펀딩 지불 4 USD
        ])
        led = load_ledger(p, official=True)            # 펀딩 행 존재 → 공식 통과
        days, gross, drag = build_daily_matrix(led, date(2026, 9, 8),
                                               date(2026, 9, 8))
        assert gross[0, 4] == pytest.approx(120.0 / CELL_CAPITAL)   # 펀딩 제외
        assert drag[0, 4] == pytest.approx((-4.0 - 8.0) / CELL_CAPITAL)


class TestResolveWindow:
    def _led(self, tmp_path, ts="2026-09-01T05:00:00Z"):
        return load_ledger(_ledger_csv(tmp_path, [_row(ts=ts)]))

    def test_상태파일_t0와_어제_컷오프를_쓴다(self, tmp_path):
        led = self._led(tmp_path)
        sp = tmp_path / "tracke_state.json"
        sp.write_text(json.dumps({"t0": "2026-08-28T11:00:00Z"}))
        t0, end = resolve_window(led, None, None, sp, official=True,
                                 today=date(2026, 9, 26))
        assert t0.isoformat() == "2026-08-28T11:00:00+00:00"
        assert end == date(2026, 9, 25)                # 어제 = 마지막 닫힌 날

    def test_공식_판정은_상태_T0가_없으면_인자가_있어도_거부한다(self, tmp_path):
        led = self._led(tmp_path)
        for t0_arg in (None, "2026-08-28T00:00:00Z"):   # --t0 로 대체 불가
            with pytest.raises(ValueError, match="사전등록 T0"):
                resolve_window(led, t0_arg, None, tmp_path / "없음.json",
                               official=True, today=date(2026, 9, 26))

    def test_공식_판정에서_t0_인자가_등록값과_다르면_거부한다(self, tmp_path):
        led = self._led(tmp_path)
        sp = tmp_path / "tracke_state.json"
        sp.write_text(json.dumps({"t0": "2026-08-28T11:00:00Z"}))
        with pytest.raises(ValueError, match="T0 변경 금지"):
            resolve_window(led, "2026-09-01T00:00:00Z", None, sp,
                           official=True, today=date(2026, 9, 26))
        # 같은 날짜라도 timestamp 가 다르면 거부 (같은 날 T0 이전 이벤트 차단)
        with pytest.raises(ValueError, match="T0 변경 금지"):
            resolve_window(led, "2026-08-28T00:00:00Z", None, sp,
                           official=True, today=date(2026, 9, 26))
        # 정확히 일치하면 허용
        t0, _ = resolve_window(led, "2026-08-28T11:00:00Z", None, sp,
                               official=True, today=date(2026, 9, 26))
        assert t0.isoformat() == "2026-08-28T11:00:00+00:00"

    def test_공식_판정_컷오프는_어제로_고정된다(self, tmp_path):
        led = self._led(tmp_path)
        sp = tmp_path / "tracke_state.json"
        sp.write_text(json.dumps({"t0": "2026-08-28T11:00:00Z"}))
        with pytest.raises(ValueError, match="컷오프는 어제"):
            resolve_window(led, None, "2026-09-20", sp, official=True,
                           today=date(2026, 9, 26))
        _, end = resolve_window(led, None, "2026-09-25", sp, official=True,
                                today=date(2026, 9, 26))   # 어제와 일치 → 허용
        assert end == date(2026, 9, 25)

    def test_T0_이전_이벤트는_원장_오염으로_거부한다(self, tmp_path):
        led = self._led(tmp_path, ts="2026-08-27T00:00:00Z")
        sp = tmp_path / "tracke_state.json"
        sp.write_text(json.dumps({"t0": "2026-08-28T11:00:00Z"}))
        with pytest.raises(ValueError, match="T0.*이전 이벤트"):
            resolve_window(led, None, None, sp, official=True,
                           today=date(2026, 9, 26))

    def test_리허설은_최초_이벤트로_T0를_추정한다(self, tmp_path, capsys):
        led = self._led(tmp_path)
        t0, end = resolve_window(led, None, "2026-09-10", tmp_path / "없음.json",
                                 official=False, today=date(2026, 9, 12))
        assert t0 == led["ts"].min()
        assert end == date(2026, 9, 10)
        assert "리허설 전용" in capsys.readouterr().out


class TestCentering:
    def test_중심화는_셀별_평균을_0으로_만든다(self):
        rng = np.random.default_rng(1)
        gross = rng.normal(0.001, 0.01, size=(60, 10)) + np.linspace(
            -0.002, 0.002, 10)
        centered = center_gross(gross)
        np.testing.assert_allclose(centered.mean(axis=0), np.zeros(10),
                                   atol=1e-15)
        # 원본 비파괴 + 편차 구조 보존
        assert gross.mean(axis=0).max() > 0
        np.testing.assert_allclose(centered - (gross - gross.mean(axis=0)),
                                   0.0, atol=1e-15)


class TestBlockResampling:
    def test_같은_seed는_같은_인덱스를_만든다(self):
        a = stationary_block_indices(90, 50, 5.0, np.random.default_rng(20260827))
        b = stationary_block_indices(90, 50, 5.0, np.random.default_rng(20260827))
        np.testing.assert_array_equal(a, b)
        c = stationary_block_indices(90, 50, 5.0, np.random.default_rng(1))
        assert not np.array_equal(a, c)

    def test_인덱스는_범위_안이고_블록은_연속이다(self):
        t_len = 30
        idx = stationary_block_indices(
            t_len, 200, 1e12, np.random.default_rng(7))   # 재시작 확률 ≈ 0
        assert idx.min() >= 0 and idx.max() < t_len
        # 재시작이 없으면 각 경로는 시작점부터 순환 연속이어야 한다
        expect = (idx[:, :1] + np.arange(t_len)) % t_len
        np.testing.assert_array_equal(idx, expect)

    def test_부트스트랩_분포는_결정론적이고_chunk에_불변이다(self):
        rng = np.random.default_rng(3)
        centered = center_gross(rng.normal(0, 0.01, size=(40, 10)))
        drag = np.full((40, 10), -0.0001)
        d1 = bootstrap_max_dist(centered, drag, n_paths=500, seed=20260827,
                                chunk=1000)
        d2 = bootstrap_max_dist(centered, drag, n_paths=500, seed=20260827,
                                chunk=7)
        d3 = bootstrap_max_dist(centered, drag, n_paths=500, seed=20260827,
                                chunk=500)
        np.testing.assert_array_equal(d1, d2)          # chunk 는 결과에 무관
        np.testing.assert_array_equal(d1, d3)


class TestMaxDistribution:
    def test_gross가_0이면_분포는_전_경로_비용합과_같다(self):
        # zero-edge 극단: gross 전부 0, 비용만 상수 → 어떤 재표집이든
        # 셀 누적 = −T×c, 최대도 −T×c (비용 재차감 검증)
        t_len, c = 20, 0.0002
        gross = np.zeros((t_len, 10))
        drag = np.full((t_len, 10), -c)
        dist = bootstrap_max_dist(center_gross(gross), drag, n_paths=300,
                                  seed=20260827)
        np.testing.assert_allclose(dist, -t_len * c, atol=1e-12)
        obs, cell = observed_max(gross, drag)
        assert obs == pytest.approx(-t_len * c)
        assert cell == CELLS[0]                     # 동률 → 고정 순서 첫 셀
        assert choose_verdict(obs, dist) == VERDICT_NULL

    def test_한_셀의_강한_평균_엣지는_상단을_초과한다(self):
        # E05 만 매일 +0.5% gross — 중심화된 null 에서는 사라져야 하고
        # 관측 최대는 분포 상단을 크게 초과해야 한다
        rng = np.random.default_rng(11)
        t_len = 60
        gross = rng.normal(0.0, 0.002, size=(t_len, 10))
        gross[:, 4] += 0.005
        drag = np.full((t_len, 10), -0.0001)
        dist = bootstrap_max_dist(center_gross(gross), drag, n_paths=2000,
                                  seed=20260827)
        obs, cell = observed_max(gross, drag)
        assert cell == "E05"
        assert obs > upper_quantile(dist)
        assert choose_verdict(obs, dist) == VERDICT_EXCEED
        assert mc_p_value(obs, dist) < 0.05

    def test_동기화_재표집은_교차상관을_보존한다(self):
        # 10셀이 완전 동일한 수익 → 어느 경로에서든 셀 간 누적이 같아야 함
        rng = np.random.default_rng(5)
        base = rng.normal(0, 0.01, size=(30, 1))
        centered = center_gross(np.repeat(base, 10, axis=1))
        drag = np.zeros((30, 10))
        combined = centered + drag
        idx = stationary_block_indices(30, 100, 5.0,
                                       np.random.default_rng(20260827))
        cum = combined[idx].sum(axis=1)             # [100, 10]
        np.testing.assert_allclose(cum - cum[:, :1], 0.0, atol=1e-12)


class TestQuantileAndP:
    def test_상단_분위는_higher_방식이다(self):
        dist = np.arange(100, dtype=float)           # 0..99
        # ceil(0.95 * 99) = 95 → 값 95.0 (선형보간이면 94.05)
        assert upper_quantile(dist, 0.95) == 95.0

    def test_mc_p는_보정_공식이다(self):
        dist = np.arange(10, dtype=float)            # 0..9
        # obs=8.5 → #{>=} = 1 (9) → (1+1)/11
        assert mc_p_value(8.5, dist) == pytest.approx(2 / 11)
        # 전부 관측 이상이면 (1+10)/11 = 1.0
        assert mc_p_value(-1.0, dist) == pytest.approx(1.0)


class TestVerdictWording:
    def test_문구는_명세_2문장만_존재한다(self):
        dist = np.linspace(-0.05, 0.05, 1001)
        below = choose_verdict(0.0, dist)
        above = choose_verdict(0.20, dist)
        assert below == "최대값이 zero-edge 공동 null과 구별되지 않음"
        assert above == ("공동 null 상단(95%) 초과 — 엣지 입증 아님, "
                         "별도 전방 확인 필요")
        assert {below, above} == {VERDICT_NULL, VERDICT_EXCEED}

    def test_상단과_정확히_같으면_null_문구다(self):
        dist = np.zeros(100)
        assert choose_verdict(0.0, dist) == VERDICT_NULL


class TestJudgmentGuard:
    def test_사전_지정_판정일만_허용한다(self):
        assert check_judgment_day(date(2026, 9, 26), force=False)
        assert check_judgment_day(date(2026, 11, 25), force=False)
        assert check_judgment_day(date(2027, 2, 23), force=False)
        assert not check_judgment_day(date(2026, 9, 25), force=False)
        assert not check_judgment_day(date(2026, 8, 27), force=False)

    def test_force는_판정일_가드를_우회한다(self):
        assert check_judgment_day(date(2026, 8, 27), force=True)

    def test_판정일_아니면_main이_거부한다(self, capsys):
        import lab.tracke_null as tn
        # 오늘이 판정일이 아님을 보장 (JUDGMENT_DATES 는 미래 고정 일자)
        rc = tn.main(["--ledger", "존재하지_않는_경로.csv"])
        out = capsys.readouterr().out
        assert rc == 2
        assert "거부" in out

    def test_force여도_원장이_없으면_거부한다(self, tmp_path, capsys):
        import lab.tracke_null as tn
        rc = tn.main(["--force", "--ledger", str(tmp_path / "없음.csv")])
        assert rc == 2
        assert "입력 검증 실패" in capsys.readouterr().out

    def test_관측일수가_최소_미만이면_거부한다(self, tmp_path, capsys):
        import lab.tracke_null as tn
        p = _ledger_csv(tmp_path, [_row(ts="2026-08-20T05:00:00Z")])
        rc = tn.main(["--force", "--ledger", str(p),
                      "--t0", "2026-08-20T00:00:00Z", "--end", "2026-08-22"])
        assert rc == 2
        assert "최소" in capsys.readouterr().out

    def test_리허설_전체_경로가_동작한다(self, tmp_path, capsys):
        import lab.tracke_null as tn
        rows = []
        rng = np.random.default_rng(0)
        for d in pd.date_range("2026-08-28", periods=15, freq="D", tz="UTC"):
            for i in range(1, 11):
                rows.append(_row(cell=f"E{i:02d}",
                                 ts=(d + pd.Timedelta(hours=5)).isoformat(),
                                 gross=float(rng.normal(0, 20)),
                                 cost=3.0, funding=0.1,
                                 symbol=f"S{i}"))
        p = _ledger_csv(tmp_path, rows)
        rc = tn.main(["--force", "--ledger", str(p),
                      "--t0", "2026-08-28T00:00:00Z", "--end", "2026-09-11"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "비공식 리허설" in out
        assert "판정: " in out
        assert (VERDICT_NULL in out) or (VERDICT_EXCEED in out)
