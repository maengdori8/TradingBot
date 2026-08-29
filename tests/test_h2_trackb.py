"""H2 트랙 B 랭커·게이트 단위 테스트 — 합성 스냅샷 시퀀스로 사전등록 산식 고정.

검증 대상 (명세 §3.2 / §5 테스트):
- 경계 스냅 후방성 (경계 직후 스냅 미사용 — 미래 스냅 금지)
- 후방 48h 한도 (경계 이전 48h 이내 스냅만 인정)
- 주 실제 스팬 [6, 8]일 유효 범위
- 흐름 필터 (1차 ≤20% / 감도 ≤50%, 분모 = 주시작 acct)
- 적격의 fills 제외 3사유 (fill-history-censored / 7일 초과 미해소 gap / 정상)
- 유효주 경계 (1차 10/13, 감도 8/13) 및 A(T0) < $10,000 제외
- 게이트 누적·멱등(판정일 중복 기록 거부)·stage2 판정 (충족/미충족)
- verdict --out-json → gate 파이프라인 (공용 판정 평가기 연동)
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import pytest

import lab.h2_consistency as h2c
from lab.h2_consistency import DAY_MS, WEEK_MS, cmd_verdict, iso_utc, parse_iso_ms
from lab.h2_trackb import (
    FILLS_EXCL_MAX_FRAC,
    GAP_UNRESOLVED_MAX_MS,
    JUDGMENT_DATES,
    MIN_VALID_WEEKS_B,
    N_WEEKS_B,
    SENS_MIN_VALID_WEEKS_B,
    STATUS_CENSORED,
    build_gate_entry,
    cmd_gate,
    cmd_rank,
    es_k,
    es20_of,
    fills_exclusion,
    form_wallet,
    load_snapshot_series,
    stage2_from_entries,
    validate_result_provenance,
    variant_stats,
)

T0_MS = float(parse_iso_ms('2026-08-27T01:00:00+00:00'))
H = 3600e3          # 1시간 (ms)
A0 = 20000.0
STEP = 200.0        # 기본 주간 pnl 증분 → r_w = 200/20000 = 0.01


# ── 합성 헬퍼 ───────────────────────────────────────────────────────────────
def weekly_series(a0: float = A0, pnl0: float = 1000.0, pnl_steps=None,
                  deposits=None, shift=None, drop=(), extra=()):
    """13주 경계 정각 스냅 시계열 합성 (점 k = T0 + k×168h + shift[k]).

    deposits: {k: 입금액} — 점 k부터 acct에 가산 (pnl 불변 = 외부 흐름).
    drop: 제거할 k (스냅 결손 시뮬레이션). extra: [(ts, pnl, acct)] 추가 점.
    """
    pnl_steps = pnl_steps or [STEP] * N_WEEKS_B
    shift = shift or {}
    deposits = deposits or {}
    ts, pnl, acct = [], [], []
    cum, dep = 0.0, 0.0
    for k in range(N_WEEKS_B + 1):
        if k > 0:
            cum += pnl_steps[k - 1]
        dep += deposits.get(k, 0.0)
        if k in drop:
            continue
        ts.append(T0_MS + k * WEEK_MS + shift.get(k, 0.0))
        pnl.append(pnl0 + cum)
        acct.append(a0 + cum + dep)
    for (t, p, a) in extra:
        ts.append(float(t))
        pnl.append(float(p))
        acct.append(float(a))
    order = np.argsort(ts, kind='mergesort')
    return (np.asarray(ts, dtype=float)[order],
            np.asarray(pnl, dtype=float)[order],
            np.asarray(acct, dtype=float)[order])


def form(a0: float = A0, **kw) -> dict:
    """weekly_series → form_wallet 축약 호출."""
    ts, pnl, acct = weekly_series(a0=a0, **kw)
    return form_wallet(T0_MS, a0, ts, pnl, acct)


def snap_row(addr: str, label: str, cap_ms: float, pnl: float, acct: float) -> dict:
    """스냅샷 파일 1행 (portfolio_snapshot 계약 필드)."""
    return {'address': addr, 'label': label, 'captured_at_utc': iso_utc(cap_ms),
            'perp_alltime_pnl': pnl, 'account_value': acct}


def wallet_rows(addr: str, a0: float = A0, step: float = STEP) -> list[dict]:
    """t0 1행 + 주 경계 정각 daily 13행 (13주 전부 유효한 깨끗한 지갑)."""
    rows = [snap_row(addr, 't0', T0_MS, 1000.0, a0)]
    for k in range(1, N_WEEKS_B + 1):
        rows.append(snap_row(addr, 'daily', T0_MS + k * WEEK_MS,
                             1000.0 + step * k, a0 + step * k))
    return rows


def run_rank(tmp_path, snap_rows: list[dict], fills_wallets: dict,
             addrs: list[str]) -> dict:
    """cmd_rank 엔드투엔드 실행 → 산출 json.gz 로드."""
    sdir = tmp_path / 'snaps'
    sdir.mkdir(exist_ok=True)
    with gzip.open(sdir / '2026-08-27.jsonl.gz', 'wt') as f:
        for r in snap_rows:
            f.write(json.dumps(r) + '\n')
    fills = tmp_path / 'h2_fills_state.json'
    fills.write_text(json.dumps({'high_turnover': [], 'wallets': fills_wallets}))
    cohort = tmp_path / 'cohort.json.gz'
    with gzip.open(cohort, 'wt') as f:
        json.dump({'locked_at': '2026-08-25T00:00:00+00:00',
                   'wallets': [{'address': a, 't0_account': 20000.0,
                                't0_month_vlm': 40000.0} for a in addrs]}, f)
    out = tmp_path / 'h2_trackb_cohort.json.gz'
    cmd_rank(argparse.Namespace(snapshots_dir=str(sdir), fills_state=str(fills),
                                cohort=str(cohort), out=str(out)))
    with gzip.open(out, 'rt') as f:
        return json.load(f)


def weeks_n(n: int) -> list[dict]:
    """흐름 0%·스팬 7일의 유효주 n개 (경계값 테스트용)."""
    return [{'w': i, 'r_w': 0.01, 'flow_frac': 0.0, 'span_d': 7.0}
            for i in range(n)]


# ── 0. ES 산식 재사용 (명세: import 재사용, 재정의 금지) ────────────────────
class TestESReuse:
    def test_same_objects_as_h2_consistency(self):
        assert es_k is h2c.es_k
        assert es20_of is h2c.es20_of
        assert variant_stats is h2c.variant_stats


# ── 1. 경계 스냅 후방성 ─────────────────────────────────────────────────────
class TestBoundarySnapBackward:
    def test_clean_wallet_13_weeks(self):
        """정각 스냅 13주: 전부 유효, r_w = 증분/A(T0), 스팬 정확히 7일."""
        res = form()
        assert res['ok']
        assert len(res['weeks']) == 13
        assert res['total_pnl'] == pytest.approx(13 * STEP)
        assert res['formation_end_ms'] == pytest.approx(T0_MS + 13 * WEEK_MS)
        for wk in res['weeks']:
            assert wk['span_d'] == pytest.approx(7.0)
            assert wk['r_w'] == pytest.approx(STEP / A0)
            assert wk['flow_frac'] == pytest.approx(0.0)

    def test_snap_just_after_boundary_never_used(self):
        """경계 직후(+1h) 스냅은 아무리 가까워도 미사용 (후방만).

        k=5 정각 스냅을 지우고 경계 직후 +1h에 점을 두면: 양방향 스냅이면
        13주 전부 유효했겠지만, 후방 전용에서는 경계 5의 후방 후보(k=4, 168h 전)가
        48h 초과 → 스냅 실패 → 주 4·5 두 개가 무효 (11주)."""
        res = form(drop={5},
                   extra=[(T0_MS + 5 * WEEK_MS + H, 2000.0, 21000.0)])
        assert res['ok']
        assert len(res['weeks']) == 11
        assert {wk['w'] for wk in res['weeks']} == set(range(13)) - {4, 5}
        for wk in res['weeks']:                 # +1h 점이 어느 주에도 안 섞임
            assert wk['span_d'] == pytest.approx(7.0)


# ── 2. 후방 48h 한도 ────────────────────────────────────────────────────────
class Test48hLimit:
    def test_within_48h_accepted(self):
        """형성 종료 경계 47h 전 스냅은 인정 — 총 PnL 관측 가능 (ok)."""
        res = form(shift={13: -47 * H})
        assert res['ok']
        assert res['total_pnl'] == pytest.approx(13 * STEP)
        # 단 주 12는 스팬 (168−47)/24 ≈ 5.04일 < 6 → 무효 → 12주
        assert len(res['weeks']) == 12

    def test_exactly_48h_accepted(self):
        res = form(shift={13: -48 * H})
        assert res['ok']

    def test_beyond_48h_rejected(self):
        """경계 48h+1ms 전 스냅뿐이면 형성 종료 스냅 없음 → 지갑 제외."""
        res = form(shift={13: -(48 * H + 1)})
        assert res['ok'] is False
        assert res['reason'] == 'no_formation_end_snap'

    def test_interior_boundary_beyond_48h_kills_two_weeks(self):
        """중간 경계(k=8)의 유일 후방 스냅이 49h 전이면 주 7·8 무효."""
        res = form(shift={8: -49 * H})
        assert res['ok']
        assert {wk['w'] for wk in res['weeks']} == set(range(13)) - {7, 8}


# ── 3. 주 스팬 [6, 8]일 ─────────────────────────────────────────────────────
class TestSpanWindow:
    def test_span_outside_6_8_invalid(self):
        """k=3 스냅 −30h: 주 2 스팬 5.75일(<6)·주 3 스팬 8.25일(>8) → 둘 다 무효."""
        res = form(shift={3: -30 * H})
        assert res['ok']
        assert {wk['w'] for wk in res['weeks']} == set(range(13)) - {2, 3}

    def test_span_inside_6_8_valid(self):
        """k=3 스냅 −20h: 주 2 스팬 ≈6.17일·주 3 스팬 ≈7.83일 → 둘 다 유효."""
        res = form(shift={3: -20 * H})
        assert len(res['weeks']) == 13
        spans = {wk['w']: wk['span_d'] for wk in res['weeks']}
        assert spans[2] == pytest.approx((168 - 20) / 24)
        assert spans[3] == pytest.approx((168 + 20) / 24)

    def test_span_exactly_6_and_8_valid(self):
        """경계값: 스팬 정확히 6일·8일은 [6, 8] 이내 → 유효."""
        res = form(shift={3: -24 * H})
        assert len(res['weeks']) == 13
        spans = {wk['w']: wk['span_d'] for wk in res['weeks']}
        assert spans[2] == pytest.approx(6.0)
        assert spans[3] == pytest.approx(8.0)


# ── 4. 흐름 필터 ────────────────────────────────────────────────────────────
class TestFlowFilter:
    def test_flow_frac_and_variants(self):
        """입금 30%가 낀 주는 1차(≤20%)에서 제외, 감도(≤50%)에서는 포함."""
        # 점 k=6 입금 → 주 5(k5→k6)만 오염. 주 5 시작 acct = 20000+1000 = 21000
        res = form(deposits={6: 0.30 * 21000.0})
        assert res['ok']
        dirty = [wk for wk in res['weeks'] if wk['flow_frac'] > 1e-9]
        assert len(dirty) == 1
        assert dirty[0]['w'] == 5
        assert dirty[0]['flow_frac'] == pytest.approx(0.30)
        prim = variant_stats(res['weeks'], res['total_pnl'], 0.20, MIN_VALID_WEEKS_B)
        sens = variant_stats(res['weeks'], res['total_pnl'], 0.50, MIN_VALID_WEEKS_B)
        assert prim['n_valid_weeks'] == 12
        assert sens['n_valid_weeks'] == 13

    def test_flow_denominator_is_week_start_acct(self):
        """흐름 비율 분모는 그 주 시작 acct (A(T0) 아님)."""
        res = form(deposits={10: 5000.0})
        wk = [w for w in res['weeks'] if w['flow_frac'] > 1e-9][0]
        week_start_acct = A0 + 9 * STEP          # 주 9 시작 acct
        assert wk['flow_frac'] == pytest.approx(5000.0 / week_start_acct)

    def test_flow_exactly_at_limits_included(self):
        """경계값: flow_frac 정확히 0.20/0.50 은 한도 이내(≤) → 유효."""
        wks = weeks_n(13)
        wks[0]['flow_frac'] = 0.20
        wks[1]['flow_frac'] = 0.50
        prim = variant_stats(wks, 1.0, 0.20, MIN_VALID_WEEKS_B)
        sens = variant_stats(wks, 1.0, 0.50, MIN_VALID_WEEKS_B)
        assert prim['n_valid_weeks'] == 12       # 0.20 포함, 0.50 제외
        assert sens['n_valid_weeks'] == 13       # 둘 다 포함


# ── 5. fills 제외 3사유 ─────────────────────────────────────────────────────
class TestFillsExclusion:
    FE = T0_MS + 13 * WEEK_MS                    # 형성 종료

    def test_censored(self):
        assert fills_exclusion({'status': STATUS_CENSORED}, self.FE) \
            == 'fill_history_censored'

    def test_unresolved_gap_over_7d(self):
        wst = {'status': 'ok', 'incomplete': True,
               'gap_until_ts': self.FE - 8 * DAY_MS}
        assert fills_exclusion(wst, self.FE) == 'gap_incomplete_over_7d'

    def test_recent_gap_not_excluded(self):
        """형성 종료 시점 6일 된 미해소 gap 은 7일 이내 → 제외 아님."""
        wst = {'status': 'ok', 'incomplete': True,
               'gap_until_ts': self.FE - 6 * DAY_MS}
        assert fills_exclusion(wst, self.FE) is None

    def test_exactly_7d_gap_not_excluded(self):
        """경계값: 정확히 7일은 '7일 초과'가 아니므로 제외 아님."""
        wst = {'status': 'ok', 'incomplete': True,
               'gap_until_ts': self.FE - GAP_UNRESOLVED_MAX_MS}
        assert fills_exclusion(wst, self.FE) is None

    def test_normal_and_resolved_ok(self):
        assert fills_exclusion({'status': 'ok'}, self.FE) is None
        assert fills_exclusion({'status': 'ok', 'incomplete': False}, self.FE) is None
        assert fills_exclusion(None, self.FE) is None      # 상태 없음(미폴링)
        assert fills_exclusion({}, self.FE) is None

    def test_incomplete_without_gap_ts_conservatively_excluded(self):
        """gap_until_ts 없는 incomplete 는 나이 미상 → 보수적 제외."""
        wst = {'status': 'ok', 'incomplete': True}
        assert fills_exclusion(wst, self.FE) == 'gap_incomplete_over_7d'

    def test_censored_as_of_formation_end(self):
        """절단은 형성 종료 시점 기준(as-of): 이전 절단만 제외, 이후 절단은 무관."""
        before = {'status': STATUS_CENSORED,
                  'censored_at': iso_utc(self.FE - DAY_MS)}
        after = {'status': STATUS_CENSORED,
                 'censored_at': iso_utc(self.FE + DAY_MS)}
        assert fills_exclusion(before, self.FE) == 'fill_history_censored'
        assert fills_exclusion(after, self.FE) is None
        # censored_at 없는 절단은 시점 미상 → 보수적 제외 (test_censored 와 동일)
        assert fills_exclusion({'status': STATUS_CENSORED}, self.FE) \
            == 'fill_history_censored'


# ── 6. 유효주 경계 (10/13 · 8/13) ──────────────────────────────────────────
class TestEligibilityBounds:
    def test_primary_10_of_13_boundary(self):
        assert variant_stats(weeks_n(10), 1.0, 0.20,
                             MIN_VALID_WEEKS_B)['eligible'] is True
        assert variant_stats(weeks_n(9), 1.0, 0.20,
                             MIN_VALID_WEEKS_B)['eligible'] is False

    def test_sensitivity_8_of_13_boundary(self):
        assert variant_stats(weeks_n(8), 1.0, 0.20,
                             SENS_MIN_VALID_WEEKS_B)['eligible'] is True
        assert variant_stats(weeks_n(7), 1.0, 0.20,
                             SENS_MIN_VALID_WEEKS_B)['eligible'] is False

    def test_positive_total_pnl_required(self):
        assert variant_stats(weeks_n(13), 0.0, 0.20,
                             MIN_VALID_WEEKS_B)['eligible'] is False
        assert variant_stats(weeks_n(13), -1.0, 0.20,
                             MIN_VALID_WEEKS_B)['eligible'] is False

    def test_three_dirty_weeks_leaves_10_valid(self):
        """입금 3회(각 주시작 acct 대비 20~50%) → 1차 10주 유효 = 적격 경계 도달."""
        res = form(deposits={3: 8000.0, 6: 9000.0, 9: 12000.0})
        prim = variant_stats(res['weeks'], res['total_pnl'], 0.20, MIN_VALID_WEEKS_B)
        sens = variant_stats(res['weeks'], res['total_pnl'], 0.50, MIN_VALID_WEEKS_B)
        assert prim['n_valid_weeks'] == 10
        assert prim['eligible'] is True
        assert sens['n_valid_weeks'] == 13
        assert prim['k'] == es_k(10) == 2
        assert prim['es20'] == pytest.approx(STEP / A0)    # 동일 r_w → 최저 2개 평균


# ── 7. rank 엔드투엔드 (A(T0) 필터·라벨·랭킹) ──────────────────────────────
class TestRankEndToEnd:
    def test_rank_a_t0_filter_and_verdict_label_ignored(self, tmp_path):
        """A(T0)<$10k 제외, =$10k 통과, no_t0 제외, verdict 행 경계 스냅 미사용."""
        rows = (wallet_rows('0xaaa', a0=20000.0)          # r_w 0.01
                + wallet_rows('0xbbb', a0=9999.0)         # A(T0) 미달
                + wallet_rows('0xddd', a0=10000.0))       # 경계값 통과, r_w 0.02
        # 0xccc: t0 없이 daily 만 → no_t0
        rows += wallet_rows('0xccc')[1:]
        # verdict 라벨 오염 행 — 경계 5 직전 1h, 사용되면 r_w 폭주
        rows.append(snap_row('0xaaa', 'verdict', T0_MS + 5 * WEEK_MS - H,
                             1e9, 1e9))
        frozen = run_rank(tmp_path, rows,
                          {'0xaaa': {'status': 'ok'},
                           '0xbbb': {'status': STATUS_CENSORED}},
                          ['0xaaa', '0xbbb', '0xccc', '0xddd'])
        w = {r['address']: r for r in frozen['wallets']}
        assert len(frozen['wallets']) == 4
        # A(T0) 필터 — 대표 사유는 base, fills 사유는 별도 필드에 병기
        assert w['0xbbb']['exclusion'] == 'a_t0_below_min'
        assert w['0xbbb']['eligible'] is False
        assert w['0xbbb']['fills_exclusion'] == 'fill_history_censored'
        assert w['0xccc']['exclusion'] == 'no_t0'
        # 적격 + verdict 라벨 미사용 (es20 오염 없음)
        assert w['0xaaa']['eligible'] is True
        assert w['0xaaa']['n_valid_weeks'] == 13
        assert w['0xaaa']['es20'] == pytest.approx(0.01)
        assert w['0xddd']['eligible'] is True
        assert w['0xddd']['es20'] == pytest.approx(0.02)
        # ES20 내림차순 랭킹
        assert w['0xddd']['rank'] == 1
        assert w['0xaaa']['rank'] == 2
        hdr = frozen['header']
        assert hdr['counts']['eligible_primary'] == 2
        assert hdr['counts']['exclusions']['no_t0'] == 1
        assert hdr['counts']['exclusions']['a_t0_below_min'] == 1
        assert hdr['counts']['snapshot_rows']['label_verdict'] == 1
        assert hdr['fills']['n_excluded'] == 0
        assert hdr['fills']['subgroup_indeterminate'] is False
        # 잔존(0xaaa·0xddd) 중 fills 상태 없는 지갑 = 0xddd (제외 아님, 진단 공개)
        assert hdr['fills']['n_no_fills_state_among_surviving'] == 1

    def test_rank_fills_three_reasons_and_subgroup_flag(self, tmp_path):
        """fills 3사유: censored / 7일 초과 gap / 정상 — 별도 집계와 30% 플래그."""
        fe = T0_MS + 13 * WEEK_MS
        rows = (wallet_rows('0xaaa') + wallet_rows('0xbbb')
                + wallet_rows('0xccc'))
        fills = {
            '0xaaa': {'status': STATUS_CENSORED, 'censored_at': 'x'},
            '0xbbb': {'status': 'ok', 'incomplete': True,
                      'gap_until_ts': fe - 8 * DAY_MS},
            '0xccc': {'status': 'ok'},
        }
        frozen = run_rank(tmp_path, rows, fills, ['0xaaa', '0xbbb', '0xccc'])
        w = {r['address']: r for r in frozen['wallets']}
        assert w['0xaaa']['eligible'] is False
        assert w['0xaaa']['exclusion'] == 'fill_history_censored'
        assert w['0xbbb']['eligible'] is False
        assert w['0xbbb']['exclusion'] == 'gap_incomplete_over_7d'
        assert w['0xccc']['eligible'] is True
        assert w['0xccc']['rank'] == 1
        # fills 사유 제외는 감도 변형 적격에도 반영
        assert w['0xaaa']['sens_flow50']['eligible'] is False
        assert w['0xccc']['sens_flow50']['eligible'] is True
        f = frozen['header']['fills']
        assert f['n_excluded'] == 2
        assert f['n_surviving_before_fills'] == 3
        assert f['excluded_frac'] == pytest.approx(2 / 3)
        assert f['excluded_frac'] > FILLS_EXCL_MAX_FRAC
        assert f['subgroup_indeterminate'] is True

    def test_fills_ratio_exactly_30pct_no_flag(self, tmp_path):
        """경계값: 제외 비율 정확히 30%는 '초과'가 아니므로 플래그 미발동."""
        addrs = [f'0x{i:02d}' for i in range(10)]
        rows = []
        for a in addrs:
            rows += wallet_rows(a)
        fills = {a: {'status': STATUS_CENSORED} for a in addrs[:3]}
        fills.update({a: {'status': 'ok'} for a in addrs[3:]})
        frozen = run_rank(tmp_path, rows, fills, addrs)
        f = frozen['header']['fills']
        assert f['n_excluded'] == 3
        assert f['n_surviving_before_fills'] == 10
        assert f['excluded_frac'] == pytest.approx(FILLS_EXCL_MAX_FRAC)
        assert f['subgroup_indeterminate'] is False


# ── 8. 스냅샷 시계열 로더 ──────────────────────────────────────────────────
class TestSnapshotSeriesLoader:
    def test_labels_keep_last_and_dedup(self, tmp_path):
        d = tmp_path / 'snaps'
        d.mkdir()
        cap_daily = T0_MS + WEEK_MS
        f1 = [snap_row('0xaaa', 't0', T0_MS, 100.0, 20000.0),
              snap_row('0xaaa', 'daily', cap_daily, 150.0, 20050.0),
              snap_row('0xaaa', 'verdict', cap_daily, 999.0, 30000.0),
              {'address': '0xaaa', 'perp_alltime_pnl': 1.0,       # label 없음
               'account_value': 1.0, 'captured_at_utc': iso_utc(T0_MS)}]
        f2 = [snap_row('0xaaa', 't0', T0_MS + 2 * H, 105.0, 20005.0),  # t0 재시도
              snap_row('0xaaa', 'daily', cap_daily, 160.0, 20060.0)]   # 동일 cap 갱신
        for name, rows in (('2026-08-27.jsonl.gz', f1), ('2026-08-28.jsonl.gz', f2)):
            with gzip.open(d / name, 'wt') as f:
                for r in rows:
                    f.write(json.dumps(r) + '\n')
        t0_map, series, counts = load_snapshot_series(
            sorted(str(p) for p in d.glob('*.jsonl.gz')))
        assert t0_map['0xaaa']['pnl'] == pytest.approx(105.0)   # t0 keep-last
        s = series['0xaaa']
        # verdict·무라벨 행 미포함: t0 2점 + daily 1점(동일 cap keep-last)
        assert len(s['ts']) == 3
        i = int(np.searchsorted(s['ts'], cap_daily))
        assert s['pnl'][i] == pytest.approx(160.0)              # 동일 cap keep-last
        assert counts['no_label'] == 1
        assert counts['label_verdict'] == 1
        assert counts['label_t0'] == 2
        assert counts['label_daily'] == 2
        assert counts['t0_dup_overwritten'] == 1    # T0 앵커 이동 진단 노출


# ── 9. 게이트 (누적·멱등·stage2) ───────────────────────────────────────────
COHORT_SHA = 'c' * 64


def verdict_doc(ic=0.10, p=0.01, spread=0.05, top=0.04, allm=0.01, n=120,
                indeterminate=False, horizon=30, judgment_utc=None,
                cohort_sha=COHORT_SHA, t0_label='verdict', th_label='verdict',
                kind='h2_verdict'):
    """verdict --out-json 형식의 판정 결과 문서 합성 (출처 필드 포함)."""
    if judgment_utc is None:
        judgment_utc = JUDGMENT_DATES.get(horizon, '2026-12-26') \
            + 'T02:00:00+00:00'
    return {'kind': kind, 'horizon_days': horizon,
            'judgment_utc': judgment_utc,
            'inputs': {'h2_cohort': {'path': 'logs/h2_trackb_cohort.json.gz',
                                     'sha256': cohort_sha},
                       't0': {'path': 't0.jsonl.gz', 'sha256': 'a' * 64,
                              'label': t0_label},
                       'th': {'path': 'th.jsonl.gz', 'sha256': 'b' * 64,
                              'label': th_label}},
            'indeterminate': indeterminate,
            'main': {'label': '1차 (층화)', 'n': n, 'ic': ic, 'p': p,
                     'spread': spread, 'top_median': top, 'all_median': allm}}


TRACKB_COHORT_SPEC = ('H2 트랙 B 13주 형성 랭커 '
                      '(docs/PREREGISTRATION_H2_2026-08-27.md §3.2)')


def write_trackb_cohort(path, wallets=(), spec=TRACKB_COHORT_SPEC) -> None:
    """게이트 결속 검증용 트랙 B 코호트 산출물 파일 합성."""
    with gzip.open(path, 'wt') as f:
        json.dump({'header': {'spec': spec}, 'wallets': list(wallets)}, f)


def run_gate(tmp_path, doc, date, horizon, gate_name='h2_trackb_gate.json',
             cohort_path=None):
    """cmd_gate 실행 → 게이트 파일 파싱 반환.

    tmp_path 에 동결 코호트 파일을 1회 생성하고 (gzip mtime 때문에 재작성 금지 —
    같은 테스트 내 SHA 안정성), 문서의 코호트 SHA 가 기본 센티널(COHORT_SHA)이면
    실제 파일 SHA 로 치환해 결속시킨다. 센티널이 아닌 값(테스트가 명시한
    불일치·결측)은 그대로 둔다.
    """
    if cohort_path is None:
        cohort_path = tmp_path / 'h2_trackb_cohort.json.gz'
        if not cohort_path.exists():
            write_trackb_cohort(cohort_path)
    real_sha = h2c.sha256_of(str(cohort_path))
    hc = (doc.get('inputs') or {}).get('h2_cohort')
    if hc and hc.get('sha256') == COHORT_SHA:
        hc['sha256'] = real_sha
    rp = tmp_path / f'result_{date}_h{horizon}.json'
    rp.write_text(json.dumps(doc))
    gp = tmp_path / gate_name
    cmd_gate(argparse.Namespace(result=str(rp), judgment_date=date,
                                horizon=horizon, cohort=str(cohort_path),
                                gate=str(gp)))
    return json.loads(gp.read_text())


class TestGate:
    def test_accumulate_and_stage2_pass(self, tmp_path):
        """T+30 충족만으로는 미진입, T+60 까지 충족 시에만 stage2_eligible."""
        g = run_gate(tmp_path, verdict_doc(horizon=30), '2026-12-26', 30)
        assert len(g['entries']) == 1
        assert g['entries'][0]['passed'] is True
        assert g['stage2_eligible'] is False               # T+60 미기록
        g = run_gate(tmp_path, verdict_doc(horizon=60), '2027-01-25', 60)
        assert len(g['entries']) == 2
        assert g['stage2_eligible'] is True

    def test_stage2_not_eligible_when_one_fails(self, tmp_path):
        g = run_gate(tmp_path, verdict_doc(horizon=30), '2026-12-26', 30)
        g = run_gate(tmp_path, verdict_doc(ic=0.02, horizon=60), '2027-01-25', 60)
        assert g['entries'][1]['passed'] is False
        assert g['stage2_eligible'] is False

    def test_indeterminate_blocks_pass(self, tmp_path):
        """수치가 전부 기준 이상이어도 판정불가면 충족 아님."""
        g = run_gate(tmp_path, verdict_doc(indeterminate=True), '2026-12-26', 30)
        assert g['entries'][0]['indeterminate'] is True
        assert g['entries'][0]['passed'] is False

    def test_duplicate_same_result_idempotent_noop(self, tmp_path):
        """같은 판정일·같은 결과 재실행 = 멱등 no-op (파일 불변, 정상 종료)."""
        run_gate(tmp_path, verdict_doc(), '2026-12-26', 30)
        gp = tmp_path / 'h2_trackb_gate.json'
        before = gp.read_text()
        g = run_gate(tmp_path, verdict_doc(), '2026-12-26', 30)
        assert gp.read_text() == before
        assert len(g['entries']) == 1

    def test_duplicate_conflicting_result_rejected(self, tmp_path):
        """같은 판정일에 다른 결과 = 충돌 — 비정상 종료로 거부, 파일 불변."""
        run_gate(tmp_path, verdict_doc(), '2026-12-26', 30)
        gp = tmp_path / 'h2_trackb_gate.json'
        before = gp.read_text()
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(ic=0.99), '2026-12-26', 30)
        assert gp.read_text() == before
        g = json.loads(gp.read_text())
        assert g['entries'][0]['ic'] == pytest.approx(0.10)   # 최초 기록 유지

    def test_horizon_mismatch_between_doc_and_arg_rejected(self, tmp_path):
        """결과 JSON horizon_days ≠ --horizon → 잘못된 파일 지정, 기록 거부."""
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(horizon=30), '2027-01-25', 60)
        assert not (tmp_path / 'h2_trackb_gate.json').exists()

    def test_official_date_of_other_horizon_rejected(self, tmp_path):
        """다른 호라이즌의 공식 판정일로 기록 시도 → 거부 (공식 기록 자리 보호)."""
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(horizon=30), '2027-01-25', 30)
        assert not (tmp_path / 'h2_trackb_gate.json').exists()

    def test_offdate_entry_recorded_but_not_counted_for_stage2(self, tmp_path):
        """비공식 날짜 재판정은 기술 기록으로 남지만 stage2 를 열 수 없다."""
        run_gate(tmp_path, verdict_doc(horizon=30), '2026-12-26', 30)
        g = run_gate(tmp_path, verdict_doc(horizon=60), '2027-02-01', 60)
        assert len(g['entries']) == 2
        assert g['entries'][1]['passed'] is True
        assert g['stage2_eligible'] is False                  # 공식 T+60 아님

    def test_nonfinite_or_out_of_range_metrics_rejected(self, tmp_path):
        """비유한(Infinity/NaN)·범위 밖 지표는 게이트가 거부한다."""
        for doc in (verdict_doc(ic=float('inf')),
                    verdict_doc(spread=float('nan')),
                    verdict_doc(p=-0.5),
                    verdict_doc(p=1.5),
                    verdict_doc(ic=1.5)):
            with pytest.raises(SystemExit):
                run_gate(tmp_path, doc, '2026-12-26', 30)
        assert not (tmp_path / 'h2_trackb_gate.json').exists()

    def test_bad_n_rejected(self, tmp_path):
        """main 존재 시 n<10(평가기 계약 위반)·비정수·Infinity 전부 거부."""
        for doc in (verdict_doc(n=1),                       # <10 인데 main 존재
                    verdict_doc(n=9),
                    verdict_doc(n=11.9),                    # 비정수
                    verdict_doc(n=float('inf'))):           # OverflowError 방어
            with pytest.raises(SystemExit):
                run_gate(tmp_path, doc, '2026-12-26', 30)
        assert not (tmp_path / 'h2_trackb_gate.json').exists()


class TestGateProvenance:
    """CLI 날짜 인자만으로 임의 JSON 이 게이트를 열 수 없다 (출처 검증)."""

    def test_wrong_kind_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(kind='other'), '2026-12-26', 30)

    def test_missing_horizon_days_rejected(self, tmp_path):
        doc = verdict_doc()
        del doc['horizon_days']
        with pytest.raises(SystemExit):
            run_gate(tmp_path, doc, '2026-12-26', 30)

    def test_non_verdict_baseline_labels_rejected(self, tmp_path):
        """트랙 B 기준선 계약: t0/th 라벨이 verdict 아니면 거부 (T0_main 오용 차단)."""
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(t0_label='t0'), '2026-12-26', 30)
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(th_label='daily'), '2026-12-26', 30)

    def test_missing_cohort_sha_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(cohort_sha=None), '2026-12-26', 30)

    def test_official_date_requires_matching_judgment_utc(self, tmp_path):
        """공식 판정일 기록인데 결과 judgment_utc 일자가 다르면 거부."""
        doc = verdict_doc(judgment_utc='2026-09-26T02:00:00+00:00')
        with pytest.raises(SystemExit):
            run_gate(tmp_path, doc, '2026-12-26', 30)
        assert not (tmp_path / 'h2_trackb_gate.json').exists()

    def test_validate_result_provenance_returns_sha(self):
        sha = validate_result_provenance(verdict_doc(), 30, '2026-12-26')
        assert sha == COHORT_SHA

    def test_cohort_file_sha_mismatch_rejected(self, tmp_path):
        """결과의 코호트 SHA ≠ 동결 산출물 실측 SHA → 다른 코호트 결과, 거부."""
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(cohort_sha='e' * 64), '2026-12-26', 30)
        assert not (tmp_path / 'h2_trackb_gate.json').exists()

    def test_cohort_file_missing_rejected(self, tmp_path):
        """동결 코호트 파일이 없으면 게이트 결속 불가 → 거부."""
        rp = tmp_path / 'r.json'
        rp.write_text(json.dumps(verdict_doc()))
        with pytest.raises(SystemExit):
            cmd_gate(argparse.Namespace(
                result=str(rp), judgment_date='2026-12-26', horizon=30,
                cohort=str(tmp_path / 'missing.json.gz'),
                gate=str(tmp_path / 'g.json')))

    def test_non_trackb_cohort_header_rejected(self, tmp_path):
        """헤더 spec 이 트랙 B 랭커 산출물이 아니면 (예: 트랙 A) 거부."""
        cohort_path = tmp_path / 'h2_trackb_cohort.json.gz'
        write_trackb_cohort(cohort_path, spec='H2 트랙 A 형성 스크린')
        with pytest.raises(SystemExit):
            run_gate(tmp_path, verdict_doc(), '2026-12-26', 30)

    def test_entry_records_required_fields(self, tmp_path):
        """게이트 기록: 판정일·3기준·p·표본·판정불가·MDE·출처 (명세 §3.2)."""
        doc = verdict_doc(n=136)
        g = run_gate(tmp_path, doc, '2026-12-26', 30)   # run_gate 가 SHA 결속
        e = g['entries'][0]
        assert e['judgment_date'] == '2026-12-26'
        assert e['horizon_days'] == 30
        assert e['n'] == 136
        assert set(e['criteria']) == {'ic_ge_min', 'p_lt_alpha',
                                      'spread_ge_min', 'top_gt_all'}
        assert e['indeterminate'] is False
        assert e['mde_ic'] == pytest.approx(h2c.mde_ic(136))
        assert e['result_sha256']
        # 결속 후 코호트 SHA = 실제 동결 산출물 파일 SHA (센티널 아님)
        assert e['cohort_sha256'] == doc['inputs']['h2_cohort']['sha256']
        assert len(e['cohort_sha256']) == 64
        assert e['judgment_utc'].startswith('2026-12-26')

    def test_criteria_thresholds(self):
        """임계 경계: IC=0.05 충족, p=0.025 미달, 스프레드=0.03 충족, 동중앙 미달."""
        e = build_gate_entry(verdict_doc(ic=0.05, p=0.025, spread=0.03,
                                         top=0.02, allm=0.02),
                             '2026-12-26', 30, 'r.json', 'sha')
        assert e['criteria'] == {'ic_ge_min': True, 'p_lt_alpha': False,
                                 'spread_ge_min': True, 'top_gt_all': False}
        assert e['passed'] is False

    def test_stage2_from_entries_combinations(self):
        e30 = {'horizon_days': 30, 'judgment_date': '2026-12-26',
               'cohort_sha256': COHORT_SHA, 'passed': True}
        e60 = {'horizon_days': 60, 'judgment_date': '2027-01-25',
               'cohort_sha256': COHORT_SHA, 'passed': True}
        e60f = {**e60, 'passed': False}
        e60_off = {**e60, 'judgment_date': '2027-02-01'}
        e60_sha = {**e60, 'cohort_sha256': 'd' * 64}
        e60_nosha = {**e60, 'cohort_sha256': None}
        e90 = {'horizon_days': 90, 'judgment_date': '2027-02-24',
               'cohort_sha256': COHORT_SHA, 'passed': True}
        assert stage2_from_entries([e30, e60]) is True
        assert stage2_from_entries([e30]) is False
        assert stage2_from_entries([e30, e60f]) is False
        assert stage2_from_entries([e30, e60_off]) is False  # 비공식 날짜 미산입
        assert stage2_from_entries([e30, e60_sha]) is False  # 이질 코호트 짝 불가
        assert stage2_from_entries([e30, e60_nosha]) is False
        assert stage2_from_entries([e30, e90]) is False      # T+90 은 대체 불가
        assert stage2_from_entries([]) is False

    def test_result_without_main_records_not_passed(self, tmp_path):
        """main 없는 결과(평가 가능 <10)는 지표 None·미충족으로 기록."""
        doc = verdict_doc()
        doc['main'] = None
        doc['insufficient_evaluable'] = 3
        g = run_gate(tmp_path, doc, '2026-12-26', 30)
        e = g['entries'][0]
        assert e['ic'] is None
        assert e['passed'] is False


# ── 10. verdict --out-json → gate 파이프라인 (공용 판정 평가기 연동) ────────
class TestVerdictGatePipeline:
    def test_verdict_out_json_feeds_gate(self, tmp_path):
        """진짜 트랙 B 파이프라인: 형성종료(2026-11-26) verdict 기준선 →
        T+30(2026-12-26) verdict 스냅 → verdict --out-json → gate 기록."""
        n = 12
        t0_cap = float(parse_iso_ms('2026-11-26T01:00:00+00:00'))  # 형성 종료
        th_cap = t0_cap + 30 * DAY_MS                              # 2026-12-26
        wallets = []
        t0_rows, th_rows = [], []
        for i in range(1, n + 1):
            addr = f'0x{i:02d}'
            wallets.append({'address': addr, 'eligible': True,
                            'es20': i * 0.01, 'rank': n + 1 - i,
                            'turnover': 2.0,
                            'sens_flow50': {'eligible': False, 'es20': None,
                                            'rank': None},
                            'sens_minw8': {'eligible': False, 'es20': None,
                                           'rank': None}})
            t0_rows.append(snap_row(addr, 'verdict', t0_cap, 0.0, 20000.0))
            # daily 오염 행 — 라벨 필터가 없으면 기준선이 이 값으로 덮임
            t0_rows.append(snap_row(addr, 'daily', t0_cap + H, 1e8, 1e8))
            th_rows.append(snap_row(addr, 'verdict', th_cap, i * 300.0,
                                    20000.0 + i * 300.0))
        cohort = tmp_path / 'trackb_cohort.json.gz'
        write_trackb_cohort(cohort, wallets=wallets)
        t0_f = tmp_path / 'snap_2026-11-26.jsonl.gz'
        th_f = tmp_path / 'snap_2026-12-26.jsonl.gz'
        for path, rows in ((t0_f, t0_rows), (th_f, th_rows)):
            with gzip.open(path, 'wt') as f:
                for r in rows:
                    f.write(json.dumps(r) + '\n')
        out_json = tmp_path / 'verdict_t30.json'
        cmd_verdict(argparse.Namespace(
            horizon=30, h2_cohort=str(cohort), t0=str(t0_f), th=str(th_f),
            judgment=None, t0_label='verdict', th_label='verdict',
            out_json=str(out_json)))
        doc = json.loads(out_json.read_text())
        assert doc['main']['n'] == n
        assert doc['main']['ic'] == pytest.approx(1.0)      # 라벨 오염 없음 증명
        assert doc['passed'] is True
        assert doc['judgment_utc'].startswith('2026-12-26')
        g = run_gate(tmp_path, doc, '2026-12-26', 30, cohort_path=cohort)
        assert g['entries'][0]['passed'] is True
        assert g['entries'][0]['cohort_sha256'] \
            == doc['inputs']['h2_cohort']['sha256']
        assert g['stage2_eligible'] is False                # T+60 미기록
