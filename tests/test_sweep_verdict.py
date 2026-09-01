"""`lab/sweep_verdict.py` 검증 — §11.1 테스트 ⑦ 포함.

핵심은 **판정이 옳게 틀리는지**다: 진짜 엣지가 없으면 통과시키지 않고(위양성),
충분히 큰 엣지를 심으면 반드시 통과시킨다(위음성). 두 방향을 전부 심는다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lab import sweep_verdict as V


# ── 1차 지표 ──────────────────────────────────────────────────────────────
def test_daily_sharpe_matches_definition() -> None:
    """§5.1 정의(mean/std ddof=1)와 일치하고 퇴화는 0 이다."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, size=(5, 400))
    r[3] = 0.0                      # 거래 0 규칙
    r[4] = 0.005                    # std = 0
    sr = V.daily_sharpe(r)
    for i in range(3):
        assert sr[i] == pytest.approx(r[i].mean() / r[i].std(ddof=1))
    assert sr[3] == 0.0
    assert sr[4] == 0.0


# ── stationary bootstrap ──────────────────────────────────────────────────
def test_stationary_bootstrap_shape_and_range() -> None:
    """인덱스는 [0, T) 안에 있고 모양이 맞다."""
    idx = V.stationary_bootstrap_indices(200, 50, 5.0, np.random.default_rng(1))
    assert idx.shape == (50, 200)
    assert idx.min() >= 0 and idx.max() < 200


def test_stationary_bootstrap_block_length_is_geometric_mean_5() -> None:
    """새 블록 개시 비율 ≈ 1/5 (평균 블록 5일)."""
    idx = V.stationary_bootstrap_indices(500, 400, 5.0, np.random.default_rng(2))
    cont = idx[:, 1:] == ((idx[:, :-1] + 1) % 500)
    # 이어붙기가 아닌 비율 ≈ p = 0.2 (우연히 이어지는 경우 때문에 살짝 작다)
    assert 0.15 < 1.0 - cont.mean() < 0.21


def test_stationary_bootstrap_is_seed_deterministic() -> None:
    """같은 seed 는 같은 인덱스 수열 — §9.4 결정론 요구."""
    a = V.stationary_bootstrap_indices(300, 20, 5.0, np.random.default_rng(V.SEED))
    b = V.stationary_bootstrap_indices(300, 20, 5.0, np.random.default_rng(V.SEED))
    assert np.array_equal(a, b)


def test_bootstrap_is_synchronized_across_rules() -> None:
    """한 경로의 날짜 수열이 전 규칙에 동일 적용되어 상관이 보존된다 (§6.2-3).

    완전 상관(r2 = 2·r1)인 두 규칙은 어떤 경로에서도 Sharpe 가 정확히 같아야 한다.
    비동기화 부트스트랩이면 이 등식이 깨진다.
    """
    rng = np.random.default_rng(3)
    r1 = rng.normal(0.0, 0.01, 300)
    r = np.vstack([r1, 2.0 * r1])
    boot = V.bootstrap_null_sharpes(r, n_paths=200, seed=7, chunk=64)
    assert np.allclose(boot[:, 0], boot[:, 1], rtol=1e-9, atol=0.0)


# ── §11.1 테스트 ⑦ — 중복도 행렬 항등식 ──────────────────────────────────
def test_counts_matrix_row_sums_equal_T() -> None:
    """중복도 행렬의 행 합은 항상 T (표본 크기 보존)."""
    idx = V.stationary_bootstrap_indices(150, 30, 5.0, np.random.default_rng(4))
    c = V.counts_matrix(idx, 150)
    assert np.all(c.sum(axis=1) == 150)
    assert np.array_equal(c[0], np.bincount(idx[0], minlength=150).astype(float))


def test_duplicity_matrix_identity_matches_naive_resampling() -> None:
    """§11.1 ⑦ — 항등식이 순진한 재표집과 일치한다.

    실수 연산에서는 **정확한 항등식**이나 부동소수점 덧셈 순서가 달라 마지막 몇 ulp
    가 다를 수 있다(어떤 구현으로도 비트 일치는 보장 불가). 기준은 `atol 1e-12 +
    rtol 1e-10` 이며, 판정 통계량 규모(일 Sharpe ~1e-2)보다 10자리 작다.
    """
    rng = np.random.default_rng(5)
    r = rng.normal(0.0005, 0.015, size=(40, 500))
    r[7] = 0.0                                   # 퇴화 규칙도 섞는다
    rc = r - r.mean(axis=1, keepdims=True)
    idx = V.stationary_bootstrap_indices(500, 120, 5.0, np.random.default_rng(V.SEED))
    fast = V.bootstrap_sharpes(np.ascontiguousarray(rc.T),
                               np.ascontiguousarray((rc * rc).T),
                               V.counts_matrix(idx, 500), 500)
    slow = V.naive_bootstrap_sharpes(rc, idx)
    rep = V.identity_report(fast, slow)
    assert rep["within_tolerance"], rep
    assert rep["max_abs_err"] < 1e-12
    assert rep["max_rel_err_significant"] < 1e-10
    assert np.all(fast[:, 7] == 0.0) and np.all(slow[:, 7] == 0.0)


def test_identity_report_catches_a_real_discrepancy() -> None:
    """검증 자체가 무력하지 않음을 증명 — 한 항을 흔들면 불합격이 나온다."""
    a = np.full((4, 6), 0.05)
    b = a.copy()
    assert V.identity_report(a, b)["within_tolerance"] is True
    b[2, 3] += 1e-6
    rep = V.identity_report(a, b)
    assert rep["within_tolerance"] is False
    assert rep["max_abs_err"] == pytest.approx(1e-6)


def test_pure_relative_error_is_not_a_valid_criterion() -> None:
    """0 근방 항에서 상대오차가 발산함을 명시 — 단독 기준 금지의 근거."""
    a = np.array([[1e-13]])
    b = np.array([[3e-13]])
    rep = V.identity_report(a, b)
    assert rep["within_tolerance"] is True          # 절대오차 2e-13... atol 초과분은 rtol
    assert rep["max_rel_err_unconditional"] > 0.5   # 그런데 상대오차는 0.67
    assert rep["max_rel_err_significant"] == 0.0


def test_sparse_rule_constant_resample_is_degenerate_not_1e13() -> None:
    """거래일이 2일뿐인 규칙 — 그 날짜를 안 뽑은 경로는 상수 표본이므로 `SR := 0`.

    실측 회귀: 관측 데이터에 1,737일 중 거래일 2일짜리 규칙이 존재한다. 그런 경로에서
    `s2 − T·mean²` 는 전 유효숫자를 잃고, 순진한 `np.std` 는 1e-19 급 먼지를 낸다.
    분모에 그 먼지를 쓰면 `SR ≈ 4.5e13` 짜리 가짜 최대통계량이 생겨 귀무분포가
    오염되고 RC p 값이 1 쪽으로 부풀려진다. 두 경로 모두 0 을 내야 한다.
    """
    t = 400
    r = np.zeros((1, t))
    r[0, 100] = 0.004
    r[0, 250] = -0.0015
    rc = V.null_centered(r)
    idx = np.arange(t)[(np.arange(t) != 100) & (np.arange(t) != 250)][:t]
    idx = np.tile(np.resize(idx, t), (1, 1))            # 거래일을 배제한 경로
    fast = V.bootstrap_sharpes(np.ascontiguousarray(rc.T),
                               np.ascontiguousarray((rc * rc).T),
                               V.counts_matrix(idx, t), t)
    slow = V.naive_bootstrap_sharpes(rc, idx)
    assert fast[0, 0] == 0.0
    assert slow[0, 0] == 0.0


def test_var_zero_threshold_separates_real_and_dust_by_many_orders() -> None:
    """상대 문턱이 진짜 규칙과 부동소수점 먼지 사이에서 명확히 분리된다."""
    # 진짜 규칙 규모: 분산/2차적률 비율 ≈ 1
    x = np.random.default_rng(9).normal(0.0005, 0.02, (1, 1000))
    m2 = float((x * x).mean())
    var = float(x.var(ddof=1))
    assert var / m2 > 1e-3 > V.VAR_ZERO_REL * 1e6
    # 상수 계열: 정확히 0 으로 판정
    assert V._sharpe_from_var(np.array([0.005]), np.array([1e-38]),
                              np.array([2.5e-5]))[0] == 0.0


def test_bootstrap_sharpe_of_identity_path_equals_observed() -> None:
    """항등 경로(각 날짜 1회)의 부트스트랩 Sharpe = 중심화 표본의 Sharpe(=0)."""
    rng = np.random.default_rng(6)
    r = rng.normal(0.002, 0.01, size=(3, 200))
    rc = r - r.mean(axis=1, keepdims=True)
    idx = np.tile(np.arange(200), (1, 1))
    sr = V.bootstrap_sharpes(np.ascontiguousarray(rc.T),
                             np.ascontiguousarray((rc * rc).T),
                             V.counts_matrix(idx, 200), 200)
    assert np.allclose(sr, 0.0, atol=1e-12)


# ── White RC ──────────────────────────────────────────────────────────────
def test_rc_does_not_reject_pure_noise() -> None:
    """진짜 엣지가 없으면 RC 는 기각하지 않는다 (위양성 방어)."""
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 0.01, size=(200, 400))
    boot = V.bootstrap_null_sharpes(r, n_paths=500, seed=V.SEED, chunk=250)
    p, _ = V.reality_check_p(V.daily_sharpe(r), boot)
    assert p > 0.05


def test_rc_rejects_a_planted_edge() -> None:
    """충분히 큰 엣지를 심으면 RC 가 기각한다 (위음성 방어 — 검정력 존재 증명)."""
    rng = np.random.default_rng(12)
    r = rng.normal(0.0, 0.01, size=(200, 400))
    r[137] += 0.006                       # 일 Sharpe ≈ 0.6 → 연환산 ≈ 11
    boot = V.bootstrap_null_sharpes(r, n_paths=500, seed=V.SEED, chunk=250)
    sr = V.daily_sharpe(r)
    p, _ = V.reality_check_p(sr, boot)
    assert p < 0.05
    assert int(np.argmax(sr)) == 137


def test_rc_p_is_bounded_below_by_one_over_bplus1() -> None:
    """p 의 하한은 1/(B+1) — 경로 수가 곧 해상도."""
    rng = np.random.default_rng(13)
    r = rng.normal(0.0, 0.01, size=(20, 300))
    r[0] += 0.05
    boot = V.bootstrap_null_sharpes(r, n_paths=200, seed=1, chunk=100)
    p, _ = V.reality_check_p(V.daily_sharpe(r), boot)
    assert p == pytest.approx(1.0 / 201.0)


# ── StepM ─────────────────────────────────────────────────────────────────
def test_stepm_empty_when_nothing_beats_critical_value() -> None:
    """순수 잡음에서는 생존 집합이 비어 있다."""
    rng = np.random.default_rng(21)
    r = rng.normal(0.0, 0.01, size=(150, 400))
    boot = V.bootstrap_null_sharpes(r, n_paths=400, seed=V.SEED, chunk=200)
    rej, steps = V.stepm(V.daily_sharpe(r), boot)
    assert rej == []
    assert steps[0]["n_active"] == 150 and steps[0]["n_rejected_this_step"] == 0


def test_stepm_finds_multiple_planted_rules() -> None:
    """복수 엣지를 심으면 단계적으로 전부 기각한다 — 'RC 는 최댓값 하나만' 보완."""
    rng = np.random.default_rng(22)
    r = rng.normal(0.0, 0.01, size=(150, 400))
    for k in (10, 40, 90):
        r[k] += 0.005
    boot = V.bootstrap_null_sharpes(r, n_paths=400, seed=V.SEED, chunk=200)
    rej, steps = V.stepm(V.daily_sharpe(r), boot)
    assert set(rej) == {10, 40, 90}
    assert len(steps) >= 2                      # 최소 한 번은 단계 진행 후 종료


def test_stepm_critical_value_is_monotone_nonincreasing() -> None:
    """집합이 줄면 최대통계량 분포도 줄어 임계값이 커지지 않는다."""
    rng = np.random.default_rng(23)
    r = rng.normal(0.0, 0.01, size=(120, 400))
    for k in (5, 15, 25, 35):
        r[k] += 0.006
    boot = V.bootstrap_null_sharpes(r, n_paths=400, seed=V.SEED, chunk=200)
    _, steps = V.stepm(V.daily_sharpe(r), boot)
    cv = [s["critical_value_daily"] for s in steps]
    assert all(cv[i + 1] <= cv[i] + 1e-12 for i in range(len(cv) - 1))


def test_stepm_subset_of_rc_rejection() -> None:
    """RC 가 기각하지 못하면 StepM 도 아무것도 기각하지 못한다 (첫 단계가 동일 검정)."""
    rng = np.random.default_rng(24)
    r = rng.normal(0.0, 0.01, size=(100, 350))
    boot = V.bootstrap_null_sharpes(r, n_paths=400, seed=V.SEED, chunk=200)
    sr = V.daily_sharpe(r)
    p, _ = V.reality_check_p(sr, boot)
    rej, _ = V.stepm(sr, boot)
    assert (p >= 0.05) == (len(rej) == 0)


# ── DSR ───────────────────────────────────────────────────────────────────
def test_sr0_matches_prereg_constants() -> None:
    """§6.4 참고 상수 재현: Z⁻¹(1−1/N)=3.4362, Z⁻¹(1−1/(Ne))=3.6983, 곱수 3.5875."""
    z1 = float(stats.norm.ppf(1.0 - 1.0 / V.N_TRIALS))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (V.N_TRIALS * math.e)))
    assert z1 == pytest.approx(3.4362, abs=5e-4)
    assert z2 == pytest.approx(3.6983, abs=5e-4)
    mult = (1.0 - V.EULER_GAMMA) * z1 + V.EULER_GAMMA * z2
    assert mult == pytest.approx(3.5875, abs=5e-4)
    assert V.sr0_threshold(0.25) == pytest.approx(0.5 * mult)


def test_dsr_half_at_sr_equals_sr0() -> None:
    """`DSR > 0.5 ⟺ SR > SR0` (§6.5 편차 마진 공시의 근거)."""
    d, margin = V.deflated_sharpe(0.05, 0.05, 0.0, 3.0)
    assert d == pytest.approx(0.5)
    assert margin == pytest.approx(0.0)
    assert V.deflated_sharpe(0.06, 0.05, 0.0, 3.0)[0] > 0.5
    assert V.deflated_sharpe(0.04, 0.05, 0.0, 3.0)[0] < 0.5


def test_dsr_matches_closed_form() -> None:
    """Bailey–López de Prado 식을 손으로 계산한 값과 일치."""
    sr, sr0, g3, g4, t = 0.06, 0.045, -0.4, 7.0, 1737
    want = stats.norm.cdf((sr - sr0) * math.sqrt(t - 1)
                          / math.sqrt(1 - g3 * sr + ((g4 - 1) / 4) * sr * sr))
    assert V.deflated_sharpe(sr, sr0, g3, g4, t)[0] == pytest.approx(float(want))


def test_dsr_penalizes_fat_tails_and_negative_skew() -> None:
    """음의 왜도·두터운 꼬리는 DSR 을 낮춘다 (분모 확대)."""
    base = V.deflated_sharpe(0.06, 0.045, 0.0, 3.0)[0]
    assert V.deflated_sharpe(0.06, 0.045, -1.0, 3.0)[0] < base
    assert V.deflated_sharpe(0.06, 0.045, 0.0, 20.0)[0] < base


def test_moments_normal_reference() -> None:
    """정규 표본의 γ3≈0, γ4≈3 (비초과 첨도 정의 확인)."""
    x = np.random.default_rng(31).normal(0, 1, 200_000)
    g3, g4 = V.moments(x)
    assert abs(g3) < 0.05
    assert g4 == pytest.approx(3.0, abs=0.1)
    assert V.moments(np.zeros(50)) == (0.0, 3.0)


# ── SPA (진단) ────────────────────────────────────────────────────────────
def test_spa_p_is_at_most_rc_p() -> None:
    """SPA 는 열등 모형을 귀무에서 빼므로 RC 보다 작거나 같은 p 를 준다."""
    rng = np.random.default_rng(41)
    r = rng.normal(0.0, 0.01, size=(120, 400))
    r[:60] -= 0.004                        # 명백히 열등한 모형 다수
    boot = V.bootstrap_null_sharpes(r, n_paths=400, seed=V.SEED, chunk=200)
    sr = V.daily_sharpe(r)
    p_rc, _ = V.reality_check_p(sr, boot)
    out = V.spa_p(sr, boot, n_obs=400)
    assert out["p"] <= p_rc + 1e-12
    assert out["n_excluded"] > 0


# ── IS/OOS ────────────────────────────────────────────────────────────────
def test_is_split_index_matches_prereg_1137() -> None:
    """§7 분할: 2021-11-21 시작 · IS 종료 2024-12-31 → IS 1,137일."""
    snap = pd.date_range("2021-11-21", periods=V.N_DAYS + 1, freq="D", tz="UTC")
    assert V.is_split_index(snap) == 1137
    assert V.N_DAYS - V.is_split_index(snap) == 600


def test_ann_sharpe_block_annualizes_by_sqrt365() -> None:
    """부분 구간 Sharpe 는 √365 로 연환산된다."""
    rng = np.random.default_rng(51)
    b = rng.normal(0.001, 0.02, size=(3, 300))
    got = V.ann_sharpe_block(b)
    want = b.mean(axis=1) / b.std(axis=1, ddof=1) * math.sqrt(365.0)
    assert np.allclose(got, want)
    assert np.array_equal(V.ann_sharpe_block(np.zeros((3, 1))), np.zeros(3))


# ── 분포 보고 ─────────────────────────────────────────────────────────────
def test_distribution_counts_are_complete_partition() -> None:
    """히스토그램 구간이 전 규칙을 빠짐없이 한 번씩만 담는다."""
    sr = np.random.default_rng(61).normal(-0.3, 0.5, 3390)
    d = V.distribution(sr)
    assert sum(h["count"] for h in d["histogram"]) == 3390
    assert d["quantiles"]["p0"] == pytest.approx(sr.min())
    assert d["quantiles"]["p100"] == pytest.approx(sr.max())


# ── 통합: 순수 잡음 전체 파이프라인 ───────────────────────────────────────
def _fake_sweep(n_rules: int, n_days: int, seed: int, plant: float = 0.0) -> V.SweepData:
    """합성 SweepData 를 만든다 (엔진 미실행)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.012, size=(n_rules, n_days))
    if plant:
        r[0] += plant
    rid = np.array([f"R{i:04d}|1h" for i in range(n_rules)], dtype=object)
    snap = pd.date_range("2021-11-21", periods=n_days + 1, freq="D", tz="UTC")
    return V.SweepData(r, rid, snap, {"spec": "test"}, None)


def test_pipeline_noise_verdict_is_failure() -> None:
    """엣지 없는 합성 스윕은 '실패' 판정과 사전등록 문구를 낸다."""
    d = _fake_sweep(300, 500, 71)
    v = V.run_verdict(d, n_paths=400, seed=V.SEED, identity_paths=8)
    assert v["verdict"]["pass"] is False
    assert v["verdict"]["statement"].startswith("3,390 시행 최대 Sharpe")
    assert v["reality_check"]["p"] > 0.05
    assert v["stepm"]["n_rejected"] == 0
    assert v["bootstrap"]["identity_check"]["within_tolerance"] is True
    assert v["distribution_full"]["n"] == 300
    assert len(v["top20"]) == 20


def test_pipeline_planted_edge_verdict_is_pass() -> None:
    """큰 엣지를 심으면 RC·StepM·DSR 세 게이트를 전부 통과한다."""
    d = _fake_sweep(300, 500, 72, plant=0.008)
    v = V.run_verdict(d, n_paths=400, seed=V.SEED, identity_paths=0)
    assert v["verdict"]["rc_pass"] and v["verdict"]["stepm_pass"]
    assert v["verdict"]["dsr_pass"] and v["verdict"]["pass"] is True
    assert v["best"]["rule_id"] == "R0000|1h"
    assert v["verdict"]["passing_rule_ids"] == ["R0000|1h"]
    assert v["verdict"]["statement"].startswith("공동 null 상단 초과")


def test_pipeline_is_deterministic_under_same_seed() -> None:
    """같은 입력·seed 는 같은 p 를 준다 (§9.2-10 재실행 금지의 전제)."""
    d = _fake_sweep(120, 400, 73)
    a = V.run_verdict(d, n_paths=200, seed=V.SEED, identity_paths=0)
    b = V.run_verdict(d, n_paths=200, seed=V.SEED, identity_paths=0)
    assert a["reality_check"]["p"] == b["reality_check"]["p"]
    assert a["dsr"]["sr0_daily"] == b["dsr"]["sr0_daily"]
    assert a["spa_diagnostic"]["p"] == b["spa_diagnostic"]["p"]


def test_load_sweep_rejects_wrong_shape(tmp_path) -> None:
    """모양이 동결 상수와 다르면 fail-closed 로 중단한다."""
    p = tmp_path / "sweep_returns.npz"
    np.savez_compressed(p, daily_returns=np.zeros((5, 5)),
                        rule_ids=np.array(["a", "b", "c", "d", "e"], dtype=object),
                        snap_ts=np.array(["2021-11-21"] * 6, dtype=object),
                        meta="{}")
    with pytest.raises(SystemExit):
        V.load_sweep(p)
