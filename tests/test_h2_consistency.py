from __future__ import annotations

"""H2 트랙 A(빠른 H2) 스크린·평가기 단위 테스트 — 합성 곡선으로 사전등록 산식 고정.

검증 대상 (명세 §트랙 A / §테스트):
- 그리드 스냅 후방성 (미래 점 절대 사용 안 함)
- 흐름 필터 (|Δacct−Δpnl|/acct(주 시작) ≤ 한도)
- ES20 k 산식 (k = ceil(0.2 × n_유효주))
- 적격 판정 (유효주 ≥ 20 AND 형성 총 perp PnL > 0)
- Y 클리핑 ([−0.95, +5.0]) 및 기간 정규화 (Y × H/실제일수, 스팬 [H−5,H+5] 밖 제외)
- 순열 p 산식 (p = (1+#{perm≥obs})/(n_perm+1), seed 고정 결정론)
- 판정불가 조건 (결측률 > 10% 또는 결측~점수 유의 상관)
- 스냅샷 라벨 필터링 (같은 파일의 daily 재시도가 t0 기준선을 덮는 경로 차단)
"""

import argparse
import gzip
import json
import math
from collections import Counter

import numpy as np
import pytest

from lab.h2_consistency import (
    CLIP_HI,
    CLIP_LO,
    DAY_MS,
    MIN_VALID_WEEKS,
    N_PERM,
    PERM_SEED,
    SENS_MIN_VALID_WEEKS,
    WEEK_MS,
    build_grid,
    cmd_screen,
    compute_y,
    es20_of,
    es_k,
    estimate_phase,
    evaluate_forward,
    is_eligible,
    load_snapshot,
    missingness_verdict,
    rank_avg,
    screen_wallet,
    snap_backward,
    spearman_avg,
    stratified_perm_p,
    terciles,
    variant_stats,
    weekly_phases,
)

PHASE_H = 166.0  # 합성 곡선 위상 (실측 ≈166.97h와 유사)


def make_wallet(n_weeks: int = 30, phase_h: float = PHASE_H, acct0: float = 20000.0,
                pnl_steps=None, deposits=None, drop_weeks=(), extra_points=()):
    """주간 곡선 합성: 점 w는 t = (phase_h + 168·w)h.

    pnl_steps: 주별 손익 증분 리스트(길이 n_weeks, 기본 +200).
    deposits: {주 인덱스: 입금액} — 해당 점부터 acct에 가산 (pnl 불변 = 외부 흐름).
    drop_weeks: 제거할 주 인덱스(곡선 결손 시뮬레이션).
    extra_points: [(ts_ms, pnl, acct)] 추가 점 (정렬 병합).
    """
    if pnl_steps is None:
        pnl_steps = [200.0] * n_weeks
    ts, pnl, acct = [], [], []
    cum, dep = 0.0, 0.0
    deposits = deposits or {}
    for w in range(n_weeks + 1):
        if w > 0:
            cum += pnl_steps[w - 1]
        dep += deposits.get(w, 0.0)
        if w in drop_weeks:
            continue
        ts.append((phase_h * 3600 + w * 168 * 3600) * 1000.0)
        pnl.append(cum)
        acct.append(acct0 + cum + dep)
    for (t, p, a) in extra_points:
        ts.append(float(t)); pnl.append(float(p)); acct.append(float(a))
    order = np.argsort(ts, kind='mergesort')
    ts = np.asarray(ts, dtype=float)[order]
    pnl = np.asarray(pnl, dtype=float)[order]
    acct = np.asarray(acct, dtype=float)[order]
    return ts, pnl, ts.copy(), acct


def grid_for(ts: np.ndarray, phase_h: float = PHASE_H) -> np.ndarray:
    """합성 곡선 위상에 정렬된 그리드."""
    return build_grid((phase_h * 3600e3) % WEEK_MS, float(ts[0]), float(ts[-1]))


# ── 1. 그리드 스냅 후방성 ───────────────────────────────────────────────────
class TestSnapBackward:
    def test_future_point_never_used(self):
        """미래 점이 후방 점보다 가까워도 절대 선택하지 않는다."""
        ts = np.array([0.0, 2.9 * DAY_MS, 6.1 * DAY_MS])
        # t=5.8d: 후방 2.9d(거리 2.9d ≤ 3d) 선택 — 미래 6.1d(거리 0.3d)는 금지
        assert snap_backward(ts, 5.8 * DAY_MS) == 1
        # t=6.0d: 후방 2.9d는 3.1d 초과 → None (미래 6.1d가 0.1d 거리인데도)
        assert snap_backward(ts, 6.0 * DAY_MS) is None

    def test_exact_and_bounds(self):
        """t와 정확히 일치하는 점은 후방으로 인정, 허용오차 경계 준수."""
        ts = np.array([10.0 * DAY_MS])
        assert snap_backward(ts, 10.0 * DAY_MS) == 0
        assert snap_backward(ts, 13.0 * DAY_MS) == 0          # 정확히 3일 = 허용
        assert snap_backward(ts, 13.0 * DAY_MS + 1) is None   # 3일 초과
        assert snap_backward(ts, 9.9 * DAY_MS) is None        # 과거에 점 없음

    def test_screen_never_snaps_forward(self):
        """주 10 점을 그리드점(φ+36h 오프셋 포함) 직후(+1h) 점으로 대체 —
        양방향 스냅이면 유효 26주, 후방 전용이면 해당 그리드점 스냅 실패로 24주."""
        base_ts, _, _, _ = make_wallet(n_weeks=30)
        # 새 그리드점 = 주10 샘플위상 + GRID_OFFSET(36h); 그 직후 +1h에 점 배치
        g10_grid = (PHASE_H * 3600 + 10 * 168 * 3600) * 1000.0 + 36 * 3600e3
        ts, pnl, ats, avs = make_wallet(
            n_weeks=30, drop_weeks={10},
            extra_points=[(g10_grid + 3600e3, 2000.0, 22000.0)])
        res = screen_wallet(ts, pnl, ats, avs, grid_for(base_ts))
        assert res['ok']
        # 그리드점 w=10의 후방 최근접은 주 9 점(204h 전) → 72h 초과 → 스냅 None
        # → 주 9–10, 10–11 두 개가 무효
        assert len(res['weeks']) == 24

    def test_clean_wallet_full_window(self):
        """결손 없는 곡선: 26주 전부 유효, r_w = 증분/A_fix, 스팬 정확히 7일."""
        ts, pnl, ats, avs = make_wallet(n_weeks=30)
        res = screen_wallet(ts, pnl, ats, avs, grid_for(ts))
        assert res['ok']
        assert len(res['weeks']) == 26
        # 그리드 = φ+36h → 마지막 유효 그리드점은 주29 샘플+36h.
        # 윈도우 = 주 3..29 (트레일링 26주), A_fix = acct(주3) = 20000+600
        assert res['a_fix'] == pytest.approx(20600.0)
        assert res['total_pnl'] == pytest.approx(26 * 200.0)
        for wk in res['weeks']:
            assert wk['span_d'] == pytest.approx(7.0)
            assert wk['r_w'] == pytest.approx(200.0 / 20600.0)
            assert wk['flow_frac'] == pytest.approx(0.0)


# ── 2. 흐름 필터 ────────────────────────────────────────────────────────────
class TestFlowFilter:
    def test_flow_frac_and_variants(self):
        """입금 30%가 낀 주는 1차(≤20%)에서 제외, 감도(≤50%)에서는 포함."""
        # 주 10 점에 입금: 직전 주 시작 acct = 20000 + pnl(주9)=1800 + 0 = 21800
        dep = 0.30 * 21800.0
        ts, pnl, ats, avs = make_wallet(n_weeks=30, deposits={10: dep})
        res = screen_wallet(ts, pnl, ats, avs, grid_for(ts))
        assert res['ok']
        dirty = [wk for wk in res['weeks'] if wk['flow_frac'] > 1e-9]
        assert len(dirty) == 1
        assert dirty[0]['flow_frac'] == pytest.approx(0.30)
        prim = variant_stats(res['weeks'], res['total_pnl'], 0.20, MIN_VALID_WEEKS)
        sens = variant_stats(res['weeks'], res['total_pnl'], 0.50, MIN_VALID_WEEKS)
        assert prim['n_valid_weeks'] == 25
        assert sens['n_valid_weeks'] == 26

    def test_flow_denominator_is_week_start_acct(self):
        """흐름 비율 분모는 그 주 시작 acct (A_fix 아님)."""
        ts, pnl, ats, avs = make_wallet(n_weeks=30, deposits={20: 5000.0})
        res = screen_wallet(ts, pnl, ats, avs, grid_for(ts))
        wk = [w for w in res['weeks'] if w['flow_frac'] > 1e-9][0]
        acct_week_start = 20000.0 + 19 * 200.0        # 주 19 시작 acct
        assert wk['flow_frac'] == pytest.approx(5000.0 / acct_week_start)


# ── 3. ES20 k 산식 ─────────────────────────────────────────────────────────
class TestES20:
    @pytest.mark.parametrize('n,k', [(26, 6), (25, 5), (21, 5), (20, 4),
                                     (16, 4), (5, 1), (1, 1)])
    def test_k_formula(self, n, k):
        assert es_k(n) == k

    def test_es20_lowest_k_mean(self):
        r = [0.01] * 20 + [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0]
        # n=26 → k=6, 최저 6개 = [-.05..0], 평균 -0.025
        assert es20_of(r) == pytest.approx(-0.025)

    def test_es20_empty(self):
        assert es20_of([]) is None


# ── 4. 적격 판정 ────────────────────────────────────────────────────────────
class TestEligibility:
    def test_needs_min_valid_weeks_and_positive_pnl(self):
        assert is_eligible(20, 1.0) is True
        assert is_eligible(19, 1.0) is False                  # 유효주 미달
        assert is_eligible(26, 0.0) is False                  # 총 PnL > 0 아님
        assert is_eligible(26, -100.0) is False
        assert is_eligible(16, 1.0, SENS_MIN_VALID_WEEKS) is True   # 감도 변형
        assert is_eligible(15, 1.0, SENS_MIN_VALID_WEEKS) is False

    def test_negative_total_pnl_wallet_not_eligible(self):
        """26주 전부 유효해도 형성 총 perp PnL ≤ 0 이면 부적격."""
        ts, pnl, ats, avs = make_wallet(n_weeks=30, pnl_steps=[-10.0] * 30)
        res = screen_wallet(ts, pnl, ats, avs, grid_for(ts))
        assert res['ok'] and res['total_pnl'] < 0
        v = variant_stats(res['weeks'], res['total_pnl'], 0.20, MIN_VALID_WEEKS)
        assert v['n_valid_weeks'] == 26
        assert v['eligible'] is False


# ── 5. Y 클리핑 ─────────────────────────────────────────────────────────────
class TestYClipping:
    def test_clip_bounds(self):
        # 손실 -200% → -0.95 로 클립
        assert compute_y(0.0, -20000.0, 10000.0, 30.0, 30.0) == pytest.approx(CLIP_LO)
        # 이익 +600% → +5.0 로 클립
        assert compute_y(0.0, 60000.0, 10000.0, 30.0, 30.0) == pytest.approx(CLIP_HI)
        # 범위 내는 그대로
        assert compute_y(100.0, 1100.0, 10000.0, 30.0, 30.0) == pytest.approx(0.10)

    def test_unclipped_sensitivity(self):
        assert compute_y(0.0, 60000.0, 10000.0, 30.0, 30.0,
                         clip=False) == pytest.approx(6.0)

    def test_invalid_denominator(self):
        assert compute_y(0.0, 100.0, 0.0, 30.0, 30.0) is None
        assert compute_y(0.0, 100.0, -5.0, 30.0, 30.0) is None


# ── 6. 기간 정규화 ─────────────────────────────────────────────────────────
class TestPeriodNormalization:
    def test_normalization_factor(self):
        # 스팬 25일, H=30 → Y × 30/25
        assert compute_y(0.0, 1000.0, 10000.0, 25.0, 30.0) == pytest.approx(0.10 * 30 / 25)
        # 스팬 35일 → Y × 30/35
        assert compute_y(0.0, 1000.0, 10000.0, 35.0, 30.0) == pytest.approx(0.10 * 30 / 35)

    def test_span_window_exclusion(self):
        # [H−5, H+5] = [25, 35] 밖은 제외
        assert compute_y(0.0, 1000.0, 10000.0, 24.9, 30.0) is None
        assert compute_y(0.0, 1000.0, 10000.0, 35.1, 30.0) is None
        assert compute_y(0.0, 1000.0, 10000.0, 25.0, 30.0) is not None
        assert compute_y(0.0, 1000.0, 10000.0, 35.0, 30.0) is not None

    def test_clip_then_normalize_order(self):
        """명세 서술 순서 고정: 클리핑 후 정규화 (raw 6.0 → clip 5.0 → ×30/25 = 6.0)."""
        y = compute_y(0.0, 60000.0, 10000.0, 25.0, 30.0)
        assert y == pytest.approx(5.0 * 30 / 25)


# ── 7. 순열 p 산식 (작은 n 결정론) ─────────────────────────────────────────
class TestPermutationP:
    def test_singleton_strata_gives_p_exactly_one(self):
        """전 원소가 각자 한 층이면 순열이 항등 → 모든 perm IC == obs → p = 1.0 정확."""
        scores = np.array([1.0, 2.0, 3.0])
        ys = np.array([0.3, 0.1, 0.2])
        p, obs, cnt = stratified_perm_p(scores, ys, np.array([0, 1, 2]))
        assert cnt == N_PERM
        assert p == pytest.approx((1 + N_PERM) / (N_PERM + 1))
        assert p == 1.0

    def test_two_point_distribution(self):
        """단일 층 n=2, 완전 정렬: perm IC ∈ {+1, −1} 각 1/2 → p ≈ 0.5."""
        p, obs, cnt = stratified_perm_p(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                        np.zeros(2, dtype=int))
        assert obs == pytest.approx(1.0)
        assert 0.45 < p < 0.55

    def test_three_point_distribution_and_formula(self):
        """단일 층 n=3, 완전 정렬: P(perm IC ≥ 1) = 1/6 → p ≈ 0.167, 산식 격자 검증."""
        p, obs, cnt = stratified_perm_p(np.array([1.0, 2.0, 3.0]),
                                        np.array([10.0, 20.0, 30.0]),
                                        np.zeros(3, dtype=int))
        assert obs == pytest.approx(1.0)
        assert 0.12 < p < 0.22
        # p = (1 + cnt) / (N_PERM + 1) 정확 일치 + 최소값 바닥 보장
        assert p == pytest.approx((1 + cnt) / (N_PERM + 1))
        assert p >= 1.0 / (N_PERM + 1)

    def test_seed_determinism(self):
        """seed=20260827 고정 → 두 번 호출 결과가 비트 단위 동일."""
        scores = np.arange(30, dtype=float)
        rng = np.random.default_rng(3)
        ys = scores + rng.normal(0, 10, 30)
        strata = np.tile([0, 1, 2], 10)
        r1 = stratified_perm_p(scores, ys, strata, n_perm=500, seed=PERM_SEED)
        r2 = stratified_perm_p(scores, ys, strata, n_perm=500, seed=PERM_SEED)
        assert r1 == r2

    def test_stratified_vs_unstratified_differ(self):
        """층간 평균 차가 클 때 층화가 층간 혼합을 차단하는지 (구현 건전성)."""
        rng = np.random.default_rng(7)
        n = 60
        strata = np.repeat([0, 1, 2], n // 3)
        ys = strata * 10.0 + rng.normal(0, 1, n)      # Y가 층에 강하게 종속
        scores = strata * 5.0 + rng.normal(0, 1, n)    # 점수도 층에 종속 → 가짜 IC
        p_str, _, _ = stratified_perm_p(scores, ys, strata, n_perm=1000)
        p_un, _, _ = stratified_perm_p(scores, ys, np.zeros(n, dtype=int), n_perm=1000)
        # 무층화는 층 교락으로 유의, 층화는 층 내 무상관이므로 비유의
        assert p_un < 0.01
        assert p_str > 0.05


# ── 8. 판정불가 조건 ────────────────────────────────────────────────────────
class TestIndeterminate:
    def test_missing_rate_over_10pct(self):
        scores = np.arange(100, dtype=float)
        miss = np.zeros(100)
        miss[::9][:11] = 1.0                           # 11% — 고르게 분산
        out = missingness_verdict(scores, miss, n_perm=2000)
        assert out['missing_rate'] == pytest.approx(0.11)
        assert out['indeterminate'] is True

    def test_missing_correlated_with_score(self):
        """결측률 8%(≤10%)라도 하위 점수에 몰리면 유의 상관 → 판정불가."""
        scores = np.arange(100, dtype=float)
        miss = np.zeros(100)
        miss[:8] = 1.0                                 # 최하위 8개 결측
        out = missingness_verdict(scores, miss, n_perm=2000)
        assert out['missing_rate'] == pytest.approx(0.08)
        assert out['p'] < 0.05
        assert out['indeterminate'] is True

    def test_clean_case_not_indeterminate(self):
        """결측 5%가 점수 전 구간에 고르게 → 상관 없음 → 판정 가능."""
        scores = np.arange(100, dtype=float)
        miss = np.zeros(100)
        miss[[10, 30, 50, 70, 90]] = 1.0
        out = missingness_verdict(scores, miss, n_perm=2000)
        assert out['missing_rate'] == pytest.approx(0.05)
        assert out['p'] >= 0.05
        assert out['indeterminate'] is False

    def test_no_missing(self):
        out = missingness_verdict(np.arange(50, dtype=float), np.zeros(50))
        assert out == {'missing_rate': 0.0, 'rho': 0.0, 'p': 1.0,
                       'indeterminate': False}


# ── 보조: 순위·위상 유틸 ────────────────────────────────────────────────────
class TestRankUtils:
    def test_rank_avg_ties(self):
        assert rank_avg(np.array([1.0, 2.0, 2.0, 3.0])).tolist() == [1.0, 2.5, 2.5, 4.0]

    def test_spearman_perfect_and_ties(self):
        assert spearman_avg(np.array([1.0, 2.0, 3.0]),
                            np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
        assert spearman_avg(np.array([1.0, 2.0, 3.0]),
                            np.array([30.0, 20.0, 10.0])) == pytest.approx(-1.0)
        # 동점 average rank 검증 (scipy.stats.spearmanr 대조값 0.9747 근사 사례)
        x = np.array([1.0, 2.0, 2.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        rx, ry = rank_avg(x), rank_avg(y)
        expect = np.corrcoef(rx, ry)[0, 1]
        assert spearman_avg(x, y) == pytest.approx(expect)

    def test_terciles(self):
        t = terciles(np.arange(9, dtype=float))
        assert t.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_phase_estimate_with_wrap(self):
        """자정 랩 근방 위상도 원형 중앙값으로 복원."""
        h = 3600e3
        est = estimate_phase(np.array([167.5 * h, 167.9 * h, 0.3 * h]))
        assert est == pytest.approx(167.9 * h)

    def test_weekly_phases_filters_dense_points(self):
        """주간 간격 점만 위상 표본에 들어간다 (촘촘한 최근 구간 제외)."""
        base = 166.0 * 3600e3
        weekly = base + WEEK_MS * np.arange(5, dtype=float)
        dense = weekly[-1] + 3600e3 * np.array([1.0, 2.0, 3.0])
        ph = weekly_phases(np.concatenate([weekly, dense]))
        assert len(ph) == 5
        assert np.allclose(ph, base % WEEK_MS)


# ── 9. 방어 가드 회귀 (적대적 감사 경미 결함 3건) ──────────────────────────
class TestDefensiveGuards:
    @staticmethod
    def _write_gz(path, text: str) -> None:
        with gzip.open(path, 'wt') as f:
            f.write(text)

    def test_nan_snapshot_line_treated_as_missing(self, tmp_path):
        """json.loads가 허용하는 NaN/Infinity 리터럴 줄은 skip → 결측 흡수."""
        p = tmp_path / 'snap.jsonl.gz'
        cap = '2026-08-01T00:00:00+00:00'
        self._write_gz(p, '\n'.join([
            f'{{"address": "0xAAA", "label": "t0", "perp_alltime_pnl": 100.0, '
            f'"account_value": 20000.0, "captured_at_utc": "{cap}"}}',
            f'{{"address": "0xbbb", "label": "t0", "perp_alltime_pnl": NaN, '
            f'"account_value": 20000.0, "captured_at_utc": "{cap}"}}',
            f'{{"address": "0xccc", "label": "t0", "perp_alltime_pnl": 1.0, '
            f'"account_value": Infinity, "captured_at_utc": "{cap}"}}',
        ]) + '\n')
        snap = load_snapshot(str(p), 't0')
        assert set(snap) == {'0xaaa'}          # NaN·Inf 줄 skip + 키 소문자
        # 자연 흡수: 결측 지갑은 no_t0 사유로 집계 (크래시·NaN 전파 없음)
        fwd = evaluate_forward([{'address': '0xbbb'}], snap, snap, 0.0, 30.0)
        assert fwd['0xbbb']['y'] is None and fwd['0xbbb']['reason'] == 'no_t0'

    def test_evaluate_forward_bad_value_defense(self):
        """스냅샷 dict에 NaN이 직접 주입돼도 bad_value 결측 처리."""
        t0 = {'0xaaa': {'pnl': float('nan'), 'acct': 10000.0, 'cap_ms': 0.0}}
        th = {'0xaaa': {'pnl': 1000.0, 'acct': 11000.0, 'cap_ms': 30.0 * DAY_MS}}
        fwd = evaluate_forward([{'address': '0xAAA'}], t0, th, 30.0 * DAY_MS, 30.0)
        assert fwd['0xAAA']['y'] is None
        assert fwd['0xAAA']['reason'] == 'bad_value'

    def test_compute_y_nonfinite_inputs(self):
        """compute_y 비유한 입력은 전부 None (spearman_avg NaN 유입 차단)."""
        nan = float('nan')
        assert compute_y(nan, 1000.0, 10000.0, 30.0, 30.0) is None
        assert compute_y(0.0, nan, 10000.0, 30.0, 30.0) is None
        assert compute_y(0.0, 1000.0, nan, 30.0, 30.0) is None
        assert compute_y(0.0, 1000.0, 10000.0, nan, 30.0) is None
        assert compute_y(0.0, float('inf'), 10000.0, 30.0, 30.0) is None

    def test_uppercase_cohort_address_matches_snapshot(self):
        """코호트 주소가 대문자여도 소문자 스냅샷 키와 매칭 (대칭 정규화)."""
        t0 = {'0xabc': {'pnl': 0.0, 'acct': 10000.0, 'cap_ms': 0.0}}
        th = {'0xabc': {'pnl': 1000.0, 'acct': 11000.0, 'cap_ms': 30.0 * DAY_MS}}
        fwd = evaluate_forward([{'address': '0xABC'}], t0, th, 30.0 * DAY_MS, 30.0)
        rec = fwd['0xABC']                     # 출력 키는 코호트 원 주소 유지
        assert rec['reason'] is None
        assert rec['y'] == pytest.approx(0.10)
        assert rec['span_d'] == pytest.approx(30.0)

    def test_screen_null_perp_dup_line_and_case(self, tmp_path):
        """perpAllTime null 행·중복 줄이 크래시 없이 제외 집계되고,
        대문자 포트폴리오 주소가 소문자 코호트와 매칭된다."""
        ts, pnl, ats, avs = make_wallet(n_weeks=30)
        curve = {'pnl': [[t, v] for t, v in zip(ts.tolist(), pnl.tolist())],
                 'acct': [[t, v] for t, v in zip(ats.tolist(), avs.tolist())]}
        port = tmp_path / 'portfolio.jsonl.gz'
        self._write_gz(port, '\n'.join([
            json.dumps({'address': '0xAAA', 'perpAllTime': curve}),   # 대문자 행
            json.dumps({'address': '0xaaa', 'perpAllTime': curve}),   # 중복 줄
            json.dumps({'address': '0xbbb', 'perpAllTime': None}),    # null 행
        ]) + '\n')
        coh = tmp_path / 'cohort.json.gz'
        self._write_gz(coh, json.dumps({
            'locked_at': '2026-08-27T00:00:00+00:00',
            'wallets': [
                {'address': '0xaaa', 't0_account': 20000.0, 't0_month_vlm': 40000.0},
                {'address': '0xbbb', 't0_account': 20000.0, 't0_month_vlm': 40000.0},
            ]}))
        out = tmp_path / 'h2_cohort.json.gz'
        cmd_screen(argparse.Namespace(portfolio=str(port), cohort=str(coh),
                                      out=str(out)))                  # 크래시 없어야 함
        with gzip.open(out, 'rt') as f:
            frozen = json.load(f)
        excl = frozen['header']['counts']['exclusions']
        assert excl.get('dup_line') == 1                              # 중복 줄 집계
        assert excl.get('too_few_points') == 1                        # null 행 제외 집계
        assert 'no_curve' not in excl                                 # 둘 다 '본' 것으로 처리
        addrs = [w['address'] for w in frozen['wallets']]
        assert addrs == ['0xaaa']                                     # 대소문자 매칭 + 첫 줄 채택
        assert frozen['wallets'][0]['eligible'] is True


# ── 10. 스냅샷 라벨 필터링 회귀 (감사 차단 4번) ─────────────────────────────
class TestLabelFiltering:
    """같은 날짜 파일에서 daily 재시도가 t0 뒤에 append 되어도 T0 기준선 불변."""

    @staticmethod
    def _write_rows(path, rows) -> None:
        with gzip.open(path, 'wt') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')

    @staticmethod
    def _row(addr: str, label: str, pnl: float, acct: float, cap: str) -> dict:
        return {'address': addr, 'label': label, 'perp_alltime_pnl': pnl,
                'account_value': acct, 'captured_at_utc': cap}

    def test_t0_kept_when_daily_appended_after(self, tmp_path):
        """t0 행 뒤에 daily 재시도가 append 된 파일 — t0 라벨 요청 시 t0 행 유지."""
        p = tmp_path / 'snap.jsonl.gz'
        self._write_rows(p, [
            self._row('0xaaa', 't0', 100.0, 20000.0, '2026-08-27T01:00:00+00:00'),
            self._row('0xaaa', 'daily', 999.0, 30000.0, '2026-08-27T12:00:00+00:00'),
        ])
        snap = load_snapshot(str(p), 't0')
        assert set(snap) == {'0xaaa'}
        assert snap['0xaaa']['pnl'] == pytest.approx(100.0)     # daily 로 덮이지 않음
        assert snap['0xaaa']['acct'] == pytest.approx(20000.0)

    def test_verdict_label_excludes_daily_and_t0(self, tmp_path):
        """verdict 라벨 요청 시 daily/t0 행은 배제되고 verdict 행만 채택된다."""
        p = tmp_path / 'snap.jsonl.gz'
        self._write_rows(p, [
            self._row('0xaaa', 't0', 100.0, 20000.0, '2026-08-27T01:00:00+00:00'),
            self._row('0xaaa', 'daily', 150.0, 20050.0, '2026-09-26T00:50:00+00:00'),
            self._row('0xaaa', 'verdict', 200.0, 20100.0, '2026-09-26T02:00:00+00:00'),
            self._row('0xbbb', 'daily', 500.0, 15000.0, '2026-09-26T00:50:00+00:00'),
        ])
        cnt: Counter = Counter()
        snap = load_snapshot(str(p), 'verdict', cnt)
        assert set(snap) == {'0xaaa'}                           # daily 전용 지갑 배제
        assert snap['0xaaa']['pnl'] == pytest.approx(200.0)
        assert cnt['other_label'] == 3                          # t0 1 + daily 2

    def test_keep_last_within_label(self, tmp_path):
        """같은 라벨이 여러 줄이면 마지막 줄 우선 (재시도 갱신) — 라벨 내 keep-last."""
        p = tmp_path / 'snap.jsonl.gz'
        self._write_rows(p, [
            self._row('0xaaa', 't0', 100.0, 20000.0, '2026-08-27T01:00:00+00:00'),
            self._row('0xaaa', 't0', 110.0, 20010.0, '2026-08-27T03:00:00+00:00'),
        ])
        snap = load_snapshot(str(p), 't0')
        assert snap['0xaaa']['pnl'] == pytest.approx(110.0)

    def test_no_label_rows_rejected_and_counted(self, tmp_path):
        """label 필드 없는 행은 계약 위반 — 채택하지 않고 거부 카운트."""
        p = tmp_path / 'snap.jsonl.gz'
        no_label = {'address': '0xbbb', 'perp_alltime_pnl': 1.0,
                    'account_value': 20000.0,
                    'captured_at_utc': '2026-08-27T01:00:00+00:00'}
        self._write_rows(p, [
            self._row('0xaaa', 't0', 100.0, 20000.0, '2026-08-27T01:00:00+00:00'),
            no_label,
        ])
        cnt: Counter = Counter()
        snap = load_snapshot(str(p), 't0', cnt)
        assert set(snap) == {'0xaaa'}
        assert cnt['no_label'] == 1
        assert cnt['other_label'] == 0
