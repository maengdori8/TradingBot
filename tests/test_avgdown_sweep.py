"""AVGDOWN-2026-09-01 엔진·판정 단위 테스트 — 전부 합성 데이터 (실데이터·본 격자 미접촉).

인과성 테스트 원칙: 각 테스트는 규약 위반 시 실패하도록 기대값을 수기 산술로
고정한다 (엔진 로직 재구현 금지 — 분석적 기대값만).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ads = _load("ads_test", "lab/avgdown_sweep.py")
adv = _load("adv_test", "lab/avgdown_verdict.py")

COST = 0.0008
ZERO_FUND = pd.Series(dtype=float)


def _flat_df(n: int, spike_at: int, bars: dict[int, tuple[float, float, float, float]]
             ) -> pd.DataFrame:
    """평탄 100 시계열 + 지정 봉 오버라이드 (o, h, l, c)."""
    o = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    c = np.full(n, 100.0)
    bars = {spike_at: (100.0, 100.0, 90.0, 90.0), **bars}
    for i, (oo, hh, ll, cc) in bars.items():
        o[i], h[i], l[i], c[i] = oo, hh, ll, cc
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.ones(n)}, index=idx)


def test_enumeration_count_dedup_and_unique_ids():
    trials = ads.enumerate_trials()
    assert len(trials) == ads.N_TRIALS == 1248
    # 전수 곱 1,536 에서 kmax=0 의 간격 차원 붕괴 (4→1) → 13/16 비율
    assert 1536 * 13 // 16 == 1248
    assert len({t.tid() for t in trials}) == 1248
    for t in trials:
        assert (t.spacing is None) == (t.kmax == 0)


def test_rsi_copy_matches_frozen_sweep_engine():
    sw = _load("sweep_engine_frozen_t", "lab/sweep_engine.py")
    rng = np.random.default_rng(3)
    px = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
    feat = sw.Feat(px, px, px, px, np.ones_like(px))
    ref, got = feat.rsi(14), ads.rsi_wilder(px)
    both = np.isfinite(ref) & np.isfinite(got)
    assert both.sum() > 400
    assert float(np.max(np.abs(ref[both] - got[both]))) == 0.0


def test_warmup_blocks_orders():
    """워밍업(100봉) 내 신호는 주문을 만들지 않는다 — 위반 시 거래 발생."""
    df = _flat_df(130, 50, {51: (90.0, 100.0, 89.9, 100.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    res = ads.simulate_sleeve(df, ZERO_FUND, ta)
    assert int(res["n_trades"][0]) == 0
    assert float(res["final_eq"][0]) == 1.0


def test_entry_fill_next_bar_open_and_stop_gap_worse():
    """진입 = 신호 다음 봉 시가, 손절 = min(시가, 동결 레벨) — 수기 산술 대조.

    스파이크 봉 j: TR=10 → ATR24[j]=10/24 (이전 TR 전부 0). 진입 j+1 시가 90,
    손절 동결 = 90 − 6×(10/24) = 87.5.
    A: j+2 시가 89 > 87.5, 저가 80 → 체결 87.5 (레벨).
    B: j+2 시가 85 < 87.5 (갭) → 체결 85 (악화 — max 모델이면 이 테스트가 실패한다).
    """
    j = 110
    n = 160
    trial = ads.Trial("E1", 0, None, 0, "MID", 6.0, "1h")
    ta = ads.trial_arrays([trial])
    common = {j + 1: (90.0, 90.0, 89.9, 89.9)}
    df_a = _flat_df(n, j, {**common, j + 2: (89.0, 100.0, 80.0, 100.0)})
    df_b = _flat_df(n, j, {**common, j + 2: (85.0, 100.0, 80.0, 100.0)})
    ra = ads.simulate_sleeve(df_a, ZERO_FUND, ta)
    rb = ads.simulate_sleeve(df_b, ZERO_FUND, ta)
    assert int(ra["n_trades"][0]) == 1 and int(rb["n_trades"][0]) == 1
    u = (1.0 / 12.0) / 90.0                      # 트랜치 = eq/12, 체결가 90
    stop = 90.0 - 6.0 * (10.0 / 24.0)
    assert stop == 87.5

    def _expect(x: float) -> float:
        return 1.0 - 90.0 * u * COST + (u * x - 90.0 * u - u * x * COST)

    assert abs(float(ra["final_eq"][0]) - _expect(87.5)) < 1e-12
    assert abs(float(rb["final_eq"][0]) - _expect(85.0)) < 1e-12
    assert int(ra["time_viol"][0]) == 0 and int(rb["time_viol"][0]) == 0


def test_add_trigger_frozen_threshold():
    """추매 임계 = 체결가 − 간격×ATR[체결봉−1] **동결** — 경계 위·아래 대조.

    임계 = 90 − 1.0×(10/24) ≈ 89.5833. 확정봉 종가 89.7 (> 임계) 은 추매 없음,
    89.5 (≤ 임계) 는 다음 봉 시가 추매. 추매 발생 여부는 kmax=0 시행과의 자본
    분기(수수료·노출)로 판정한다.
    """
    j = 110
    n = 200
    trials = [ads.Trial("E1", 0, 1.0, 3, "MID", None, "1h"),
              ads.Trial("E1", 0, None, 0, "MID", None, "1h")]
    ta = ads.trial_arrays(trials)
    rec = {i: (100.0, 100.0, 100.0, 100.0) for i in range(j + 3, n)}  # 회복 → TP
    base = {j + 1: (90.0, 90.0, 89.4, 0.0)}      # c 는 아래에서 케이스별 설정
    for c_prev, expect_add in ((89.7, False), (89.5, True)):
        bars = {**base, **rec, j + 2: (89.6, 89.6, 89.4, 89.6)}
        bars[j + 1] = (90.0, 90.0, 89.4, c_prev)
        df = _flat_df(n, j, bars)
        res = ads.simulate_sleeve(df, ZERO_FUND, ta)
        same = abs(float(res["final_eq"][0]) - float(res["final_eq"][1])) < 1e-15
        assert same != expect_add, (c_prev, res["final_eq"])
        assert int(res["time_viol"].sum()) == 0


def test_lookahead_control_differs():
    """같은 봉 신호 평가(위반 대조군)는 결과가 달라야 한다."""
    rng = np.random.default_rng(11)
    n = 2000
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1] * (1 + rng.normal(0, 1e-3, n - 1))
    h = np.maximum(o, c) * 1.005
    l = np.minimum(o, c) * 0.995
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                       "volume": np.ones(n)}, index=idx)
    ta = ads.trial_arrays([ads.Trial("E1", 0, 1.0, 3, "MID", None, "1h")])
    a = ads.simulate_sleeve(df, ZERO_FUND, ta, causal=True)
    b = ads.simulate_sleeve(df, ZERO_FUND, ta, causal=False)
    assert abs(float(a["final_eq"][0]) - float(b["final_eq"][0])) > 1e-9


def test_trend_filter_blocks_entry_fail_closed():
    """SMA200 필터: 미형성(NaN)·이하 구간 진입 차단 — 하락 스파이크 진입은
    필터 켠 시행에서 0거래여야 한다 (100 평탄 후 90 스파이크 → c1 < SMA200)."""
    j = 250
    df = _flat_df(300, j, {j + 1: (90.0, 90.0, 89.9, 89.9)})
    ta = ads.trial_arrays([ads.Trial("E1", 1, None, 0, "MID", None, "1h"),
                           ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    res = ads.simulate_sleeve(df, ZERO_FUND, ta)
    assert int(res["n_trades"][0]) == 0            # 필터 on: 90 < SMA200(≈100)
    assert int(res["n_trades"][1]) == 1            # 필터 off: 진입 (eod 청산 포함)


def test_sleeve_returns_grid_and_pre_start_cash():
    """마스터 그리드: 슬리브 시작 전 = 현금(수익률 0), 이후 = 자본 비율."""
    grid = ads.master_days()
    assert len(grid) == ads.N_DAYS + 1 == 2057
    days = pd.date_range("2024-01-01", periods=3, freq="D", tz="utc")
    day_eq = np.array([[1.0, 1.1, 1.21]])
    ret = ads.sleeve_returns(day_eq, days)
    assert ret.shape == (1, ads.N_DAYS)
    pos = grid.get_loc(pd.Timestamp("2024-01-02", tz="utc")) - 1
    assert ret[0, : grid.get_loc(pd.Timestamp("2024-01-01", tz="utc")) - 1].max() == 0.0
    assert abs(ret[0, pos] - 0.1) < 1e-12
    assert abs(ret[0, pos + 1] - 0.1) < 1e-12
    assert ret[0, pos + 2] == 0.0                  # 스냅샷 종료 후 = 자본 유지


def test_verdict_selftest_passes():
    """판정 기계 자가검증 (고정 ω̂ 항등식·오지정 탐지·검정력·결정론)."""
    adv.selftest()


def test_verdict_fixed_scale_degenerate():
    ret = np.zeros((3, 100))
    ret[1] = np.random.default_rng(0).normal(0, 0.01, 100)
    om = adv.fixed_scale(ret)
    assert om[0] == 0.0 and om[2] == 0.0 and om[1] > 0
    st = adv.obs_stat(ret, om)
    assert st[0] == 0.0 and st[2] == 0.0
