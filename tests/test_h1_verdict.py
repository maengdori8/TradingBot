from __future__ import annotations

"""H1 판정 평가기(lab/h1_verdict.py) 단위 테스트 — 합성 데이터로 동결 산식 고정.

검증 대상 (동결 전 고정 — 첫 판정일 2026-09-24 이전):
- 동결 의존성·입력 바이트 동일성 (h2_consistency·코호트·T0 파일 SHA256 핀)
- 기준 경계 (IC ≥ +0.05 / p < 0.05·0.025 / 스프레드 ≥ +3%p / D1중앙 > 전체중앙)
- 결측 규칙 (마지막 유효 ≤ 판정일, 스팬 [H−5,H+5] 밖 제외, 클리핑 후 기간 정규화)
- 2026-08-28 류 내부 결손일 내성 (2점 차분 — 결과 불변, 인벤토리 공개만)
- 판정일 이후 파일 절대 미개봉 + 판정일 도래 전 공식 실행 차단 (조기 판정 봉쇄)
- 판정불가 (결측률 > 10% / 결측~점수 유의 상관 — 경계: 정확히 10% 는 비발동)
- T0 잠금 십분위·경계 동점 공개, 3상태 판정, 공식/탐색 구분
- 감도·진단 실패 격리 (1차 판정 보존), RFC 준수 JSON (NaN 금지)
- 최종 게이트 결합 (정준화·자기일관성 재계산·fail > indeterminate > pass)
"""

import copy
import gzip
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import lab.h2_consistency as h2
from lab import h1_verdict as h1v

T0 = '2026-08-25'
J30 = '2026-09-24'
HEADER = 'address,account,day_pnl,month_pnl,month_roi,month_vlm,alltime_pnl\n'


# ── 합성 데이터 헬퍼 ─────────────────────────────────────────────────────────
def addr_of(i: int) -> str:
    return f'0x{i:040x}'


def mk_wallet(i: int, roi: float, acct: float = 20000.0) -> dict:
    return {'address': addr_of(i), 't0_account': acct, 't0_month_roi': roi,
            't0_month_pnl': roi * acct, 't0_month_vlm': 1e6, 'is_vault': False}


def write_cohort(path, wallets: list[dict]) -> None:
    with gzip.open(path, 'wt') as f:
        json.dump({'locked_at': T0, 'n': len(wallets),
                   'filters': {'min_account': 10000.0, 'min_month_vlm': 1e6},
                   'wallets': wallets}, f)


def write_daily(daily_dir, date_str: str, rows: list[tuple]) -> None:
    """rows: (addr, account, alltime_pnl) — 나머지 컬럼은 0."""
    with gzip.open(Path(daily_dir) / f'{date_str}.csv.gz', 'wt') as f:
        f.write(HEADER)
        for addr, acct, pnl in rows:
            f.write(f'{addr},{acct},0,0,0,0,{pnl}\n')


def build_scenario(root, n: int = 40, reverse: bool = False,
                   missing: set | None = None,
                   rois: list | None = None) -> tuple[str, str]:
    """표준 시나리오: T0 잠금 점수 내림차순(i↑ = 점수↓), 판정일(J30) 전방 Y 가
    점수와 완전 단조(또는 reverse) — 코호트·일별 디렉토리 경로 반환."""
    missing = missing or set()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cohort = root / 'cohort.json.gz'
    daily = root / 'daily'
    daily.mkdir(exist_ok=True)
    if rois is None:
        rois = [1.0 - 0.04 * i for i in range(n)]
    write_cohort(cohort, [mk_wallet(i, rois[i]) for i in range(n)])
    write_daily(daily, T0, [(addr_of(i), 20000.0, 1000.0 * i) for i in range(n)])
    ys = [(0.5 - 0.01 * i) for i in range(n)]
    if reverse:
        ys = ys[::-1]
    write_daily(daily, J30, [(addr_of(i), 20000.0, 1000.0 * i + ys[i] * 20000.0)
                             for i in range(n) if i not in missing])
    return str(cohort), str(daily)


def run_patched(mp, cohort: str, daily: str, horizon: int,
                judgment: str | None = None, today: date | None = None) -> dict:
    """합성 입력에 맞춰 동결 핀(코호트 크기·SHA256)을 패치하고 판정 실행.

    today 기본값 = 판정일 (조기 실행 차단을 통과하는 최소 시각)."""
    with gzip.open(cohort, 'rt') as f:
        n = json.load(f)['n']
    mp.setattr(h1v, 'EXPECTED_COHORT_N', n)
    mp.setattr(h1v, 'COHORT_SHA256_PINNED', h2.sha256_of(cohort))
    mp.setattr(h1v, 'T0_DAILY_SHA256_PINNED',
               h2.sha256_of(str(Path(daily) / f'{T0}.csv.gz')))
    if today is None:
        today = (date.fromisoformat(judgment[:10]) if judgment
                 else date.fromisoformat(T0) + timedelta(days=horizon))
    return h1v.run_verdict(cohort, daily, horizon, judgment, today=today)


@pytest.fixture(scope='module')
def pass_run(tmp_path_factory):
    """40지갑 완전 단조 지속성 시나리오의 공식 T+30 판정 (여러 테스트 공유)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp('h1pass')
    cohort, daily = build_scenario(root)
    res = run_patched(mp, cohort, daily, 30, None)
    mp.undo()
    return {'res': res, 'cohort': cohort, 'daily': daily}


def pin_from(monkeypatch, res: dict) -> None:
    """combine 검증용: 판정 결과가 기록한 합성 입력 SHA 를 핀으로 패치."""
    monkeypatch.setattr(h1v, 'COHORT_SHA256_PINNED',
                        res['inputs']['cohort']['sha256'])
    monkeypatch.setattr(h1v, 'T0_DAILY_SHA256_PINNED',
                        res['inputs']['t0_daily']['sha256'])


# ── 동결 의존성·입력 핀 ─────────────────────────────────────────────────────
def test_frozen_h2_module_byte_identity():
    """h2_consistency 는 사전등록 핀과 바이트 동일해야 한다 (수정 금지 증명)."""
    assert h1v.verify_frozen_dep() == h1v.H2_SHA256_PINNED
    assert h1v.H2_SHA256_PINNED == (
        'e953af8fdd21286a3507e3f9855e5007271321c173891391060911141f198b64')


def test_h2_hash_mismatch_aborts(monkeypatch):
    monkeypatch.setattr(h1v, 'H2_SHA256_PINNED', '0' * 64)
    with pytest.raises(h1v.FrozenSpecError):
        h1v.verify_frozen_dep()


def test_input_pin_mismatch_aborts(tmp_path, monkeypatch):
    """코호트·T0 파일이 핀과 다르면 파싱 전에 중단 (변조·오배선 차단)."""
    cohort, daily = build_scenario(tmp_path)
    with gzip.open(cohort, 'rt') as f:
        n = json.load(f)['n']
    monkeypatch.setattr(h1v, 'EXPECTED_COHORT_N', n)
    monkeypatch.setattr(h1v, 'COHORT_SHA256_PINNED', '1' * 64)
    with pytest.raises(h1v.FrozenSpecError, match='코호트 파일'):
        h1v.run_verdict(cohort, daily, 30, None, today=date(2026, 9, 24))
    monkeypatch.setattr(h1v, 'COHORT_SHA256_PINNED', h2.sha256_of(cohort))
    monkeypatch.setattr(h1v, 'T0_DAILY_SHA256_PINNED', '1' * 64)
    with pytest.raises(h1v.FrozenSpecError, match='T0 일별 파일'):
        h1v.run_verdict(cohort, daily, 30, None, today=date(2026, 9, 24))


def test_premature_official_run_blocked(tmp_path, monkeypatch):
    """판정일 도래 전 '공식' 실행은 계산 자체를 차단 (조기 판정 = 위반 경로)."""
    cohort, daily = build_scenario(tmp_path)
    with pytest.raises(h1v.FrozenSpecError, match='조기 공식 실행'):
        run_patched(monkeypatch, cohort, daily, 30, None,
                    today=date(2026, 9, 23))
    # 판정일 당일부터 허용
    res = run_patched(monkeypatch, cohort, daily, 30, None,
                      today=date(2026, 9, 24))
    assert res['analysis_status'] == 'official'
    assert res['evaluated_on'] == '2026-09-24'
    # 탐색 실행은 시계 제약 없음 (운영 점검용 — 문서 허용)
    res = run_patched(monkeypatch, cohort, daily, 30, '2026-09-20',
                      today=date(2026, 8, 29))
    assert res['analysis_status'] == 'exploratory' and res['status'] is None


# ── 순수 함수: 점수·판정 경계 ────────────────────────────────────────────────
def test_clip_score_bounds():
    assert h1v.clip_score(-2.0) == h2.CLIP_LO == -0.95
    assert h1v.clip_score(7.3) == h2.CLIP_HI == 5.0
    assert h1v.clip_score(0.1) == 0.1


def test_judge_boundaries():
    base = dict(ic=0.2, p=0.001, spread=0.5, top_median=0.1, all_median=0.0)
    assert h1v.judge(**base)['all_pass_doc_alpha']
    # IC 경계: ≥ +0.05 (같으면 충족)
    assert h1v.judge(**{**base, 'ic': 0.05})['ic_ge_005']
    assert not h1v.judge(**{**base, 'ic': 0.0499})['ic_ge_005']
    # p 경계: < (같으면 미달) — 원문 .05 / 거버넌스 .025 병기
    j = h1v.judge(**{**base, 'p': 0.05})
    assert not j['p_lt_005'] and not j['p_lt_0025']
    j = h1v.judge(**{**base, 'p': 0.0499})
    assert j['p_lt_005'] and not j['p_lt_0025']
    assert j['all_pass_doc_alpha'] and not j['all_pass_strict_alpha']
    assert h1v.judge(**{**base, 'p': 0.0249})['p_lt_0025']
    # 스프레드 경계: ≥ +3%p
    assert h1v.judge(**{**base, 'spread': 0.03})['spread_ge_3pp']
    assert not h1v.judge(**{**base, 'spread': 0.0299})['spread_ge_3pp']
    # 상위중앙 > 전체중앙 (같으면 미달)
    assert not h1v.judge(**{**base, 'top_median': 0.0})['top_gt_all']
    # NaN 은 fail-closed
    for k in ('ic', 'spread', 'top_median'):
        assert not h1v.judge(**{**base, k: math.nan})['all_pass_doc_alpha']


def test_overall_status_three_state():
    assert h1v.overall_status(True, True) == 'indeterminate'
    assert h1v.overall_status(True, False) == 'indeterminate'
    assert h1v.overall_status(False, True) == 'pass'
    assert h1v.overall_status(False, False) == 'fail'


# ── 코호트 계약 검증 (fail-closed) ──────────────────────────────────────────
def test_cohort_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(h1v, 'EXPECTED_COHORT_N', 2)
    ok = [mk_wallet(0, 0.5), mk_wallet(1, -0.2)]
    p = tmp_path / 'c.json.gz'
    write_cohort(p, ok)
    assert len(h1v.load_cohort(str(p))) == 2
    # locked_at 불일치
    with gzip.open(p, 'wt') as f:
        json.dump({'locked_at': '2026-08-26', 'n': 2, 'wallets': ok}, f)
    with pytest.raises(h1v.FrozenSpecError):
        h1v.load_cohort(str(p))
    # 크기 불일치 (조용한 축소 금지)
    write_cohort(p, ok[:1])
    with pytest.raises(h1v.FrozenSpecError):
        h1v.load_cohort(str(p))
    # 주소 중복 (대소문자 정규화 후)
    dup = [mk_wallet(0, 0.5), {**mk_wallet(0, 0.1), 'address': addr_of(0).upper()}]
    write_cohort(p, dup)
    with pytest.raises(h1v.FrozenSpecError):
        h1v.load_cohort(str(p))
    # 비유한 점수
    bad = [mk_wallet(0, 0.5), {**mk_wallet(1, 0.1), 't0_month_roi': math.inf}]
    write_cohort(p, bad)
    with pytest.raises(h1v.FrozenSpecError):
        h1v.load_cohort(str(p))
    # 감도 전용 필드는 무효여도 1차를 죽이지 않음 — 0.0 정규화
    soft = [mk_wallet(0, 0.5), {**mk_wallet(1, 0.1), 't0_month_pnl': None,
                                't0_month_vlm': math.nan}]
    write_cohort(p, soft)
    loaded = h1v.load_cohort(str(p))
    assert loaded[1]['t0_month_pnl'] == 0.0 and loaded[1]['t0_month_vlm'] == 0.0


# ── 일별 행 파싱 규칙 ────────────────────────────────────────────────────────
def test_read_daily_rows_rules(tmp_path):
    from collections import Counter
    p = tmp_path / 'd.csv.gz'
    with gzip.open(p, 'wt') as f:
        f.write(HEADER)
        f.write(f'{addr_of(1)},100,0,0,0,0,10\n')
        f.write(f'{addr_of(1).upper()},200,0,0,0,0,20\n')     # 케이스 정규화 + keep-last
        f.write(f'{addr_of(2)},300,0,0,0,0,nan\n')            # 비유한 pnl 거부
        f.write(f'{addr_of(3)},300,0,0,0,0,notafloat\n')      # 파싱 불가 거부
        f.write(f'{addr_of(4)},nan,0,0,0,0,40\n')             # acct 무효 → None, 행 유지
    cnt: Counter = Counter()
    rows = h1v._read_daily_rows(str(p), cnt)
    assert rows[addr_of(1)] == {'pnl': 20.0, 'acct': 200.0}
    assert cnt['dup_row'] == 1 and cnt['nonfinite_pnl'] == 1 and cnt['bad_row'] == 1
    assert addr_of(2) not in rows and addr_of(3) not in rows
    assert rows[addr_of(4)]['acct'] is None and rows[addr_of(4)]['pnl'] == 40.0


def test_post_judgment_files_never_opened(tmp_path):
    """판정일 이후 파일은 이름 단계에서 걸러 열지도 않는다 (룩어헤드 물리 차단)."""
    daily = tmp_path
    write_daily(daily, T0, [(addr_of(0), 100.0, 0.0)])
    write_daily(daily, '2026-09-21', [(addr_of(0), 100.0, 5.0)])
    write_daily(daily, J30, [(addr_of(0), 100.0, 7.0)])
    # 판정일 다음날 파일: gzip 도 아닌 쓰레기 — 열면 예외가 났을 것
    (daily / '2026-09-25.csv.gz').write_bytes(b'NOT-GZIP-GARBAGE')
    (daily / 'notadate.csv.gz').write_bytes(b'ignored-by-name')
    stats: list = []
    ep, ep_mid = h1v.load_endpoints(h1v.list_daily_files(str(daily)),
                                    date(2026, 9, 24), None, stats)
    assert ep[addr_of(0)]['date'] == date(2026, 9, 24)
    assert ep[addr_of(0)]['pnl'] == 7.0
    assert [s['date'] for s in stats] == ['2026-09-21', '2026-09-24']
    assert ep_mid == {}


# ── 지갑별 결측·폴백·정규화 규칙 ────────────────────────────────────────────
def test_evaluate_wallet_fallback_and_normalization():
    j, h = date(2026, 9, 24), 30
    t0_row = {'pnl': 0.0, 'acct': 20000.0}
    # 판정일 당일 행: 스팬 30 = H → 정규화 없음, stale=False
    r = h1v.evaluate_wallet(t0_row, {'date': j, 'pnl': 3000.0, 'acct': 23000.0}, j, h)
    assert r['y'] == pytest.approx(0.15) and r['span_d'] == 30 and r['stale'] is False
    assert r['flow'] == pytest.approx((3000.0 - 3000.0) / 20000.0)
    # 폴백: 마지막 유효 T0+27 → clip 후 × 30/27, stale=True
    r = h1v.evaluate_wallet(t0_row, {'date': date(2026, 9, 21), 'pnl': 2700.0,
                                     'acct': None}, j, h)
    assert r['y'] == pytest.approx((2700.0 / 20000.0) * 30 / 27)
    assert r['span_d'] == 27 and r['stale'] is True and r['flow'] is None
    # 클리핑 후 정규화 순서: raw 6.0 → clip 5.0 → × 30/25 = 6.0 (unclipped 7.2)
    r = h1v.evaluate_wallet(t0_row, {'date': date(2026, 9, 19), 'pnl': 120000.0,
                                     'acct': None}, j, h)
    assert r['y'] == pytest.approx(5.0 * 30 / 25)
    assert r['y_unclipped'] == pytest.approx(6.0 * 30 / 25)
    assert r['clipped'] is True
    # 스팬 게이트: T0+20 → 제외 (LOCF 값은 게이트·정규화 없이 남는다)
    r = h1v.evaluate_wallet(t0_row, {'date': date(2026, 9, 14), 'pnl': 4000.0,
                                     'acct': None}, j, h)
    assert r['y'] is None and r['reason'] == 'span_out'
    assert r['y_locf'] == pytest.approx(0.2)
    # 결측 사유들
    assert h1v.evaluate_wallet(None, None, j, h)['reason'] == 'no_t0'
    assert h1v.evaluate_wallet(t0_row, None, j, h)['reason'] == 'no_th'
    assert h1v.evaluate_wallet({'pnl': 0.0, 'acct': None}, None, j, h)['reason'] == 'bad_a0'
    assert h1v.evaluate_wallet({'pnl': 0.0, 'acct': -1.0}, None, j, h)['reason'] == 'bad_a0'


def test_rolling_y_rules():
    """롤링 감도: 기준선 T0 기점 스팬 [25,30], 분모 = 기준선 account, 게이트 [25,35]."""
    end = {'date': date(2026, 10, 24), 'pnl': 600.0, 'acct': None}
    mid = {'date': date(2026, 9, 22), 'pnl': 100.0, 'acct': 5000.0}   # T0+28
    # span_roll = 32 → 정규화 ×30/32, 분모는 mid account 5000 (T0 acct 아님)
    assert h1v.rolling_y(mid, end) == pytest.approx((500.0 / 5000.0) * 30 / 32)
    # 기준선 스팬 경계: T0+24 → 불가, T0+25·T0+30 → 가능
    assert h1v.rolling_y({**mid, 'date': date(2026, 9, 18)}, end) is None
    assert h1v.rolling_y({**mid, 'date': date(2026, 9, 19)}, end) is not None
    assert h1v.rolling_y({**mid, 'date': date(2026, 9, 24)}, end) is not None
    # 롤링 스팬 게이트: 09-19 기점 end 10-25 → 36일 → 불가
    assert h1v.rolling_y({**mid, 'date': date(2026, 9, 19)},
                         {**end, 'date': date(2026, 10, 25)}) is None
    # 분모 무효 → 롤링만 결측
    assert h1v.rolling_y({**mid, 'acct': None}, end) is None
    assert h1v.rolling_y({**mid, 'acct': 0.0}, end) is None
    assert h1v.rolling_y(None, end) is None and h1v.rolling_y(mid, None) is None


# ── T0 잠금 십분위 ───────────────────────────────────────────────────────────
def test_locked_deciles_and_boundary_ties():
    n = 20
    addrs = [addr_of(i) for i in range(n)]
    scores = np.asarray([1.0] + [0.9, 0.9] + [0.8 - 0.01 * i for i in range(n - 3)])
    labels = h1v.locked_decile_labels(scores, addrs)
    assert [int((labels == d).sum()) for d in range(10)] == [2] * 10
    # 멤버십은 점수·주소만의 함수 — 결측과 무관 (같은 입력 → 같은 라벨)
    assert np.array_equal(labels, h1v.locked_decile_labels(scores, addrs))
    # D1/D2 경계 동점 0.9 가 2지갑 — 주소 타이브레이크로 갈렸음을 공개
    ties = h1v.boundary_tie_counts(scores, addrs)
    assert ties['d1_d2']['tied_wallets'] == 2
    assert ties['d9_d10']['tied_wallets'] == 0
    # locked_block: 결측은 중앙값에서만 빠지고 멤버십 불변
    ys = [float(s) for s in scores]
    ys[0] = None                                  # D1 한 명 결측
    blk = h1v.locked_block(labels, scores, ys)
    assert blk['n'] == n - 1
    assert blk['decile_n_obs'][0] == 1 and sum(blk['decile_n_obs']) == n - 1
    assert blk['top_median'] == pytest.approx(0.9)


def test_omnibus_missingness_warning():
    labels = np.repeat(np.arange(10), 10)         # n=100, 십분위당 10
    miss = np.zeros(100)
    miss[:10] = 1.0                               # 전부 D1 에 집중 → 경고
    assert h1v.omnibus_missingness_p(labels, miss)['p'] < 0.05
    spread = np.zeros(100)
    spread[::10] = 1.0                            # 십분위당 정확히 1개 → 균등
    assert h1v.omnibus_missingness_p(labels, spread)['p'] > 0.5


# ── 본 판정: 통과/미달/판정불가/공식성 ──────────────────────────────────────
def test_verdict_pass_official(pass_run):
    res = pass_run['res']
    assert res['kind'] == 'h1_verdict'
    assert res['analysis_status'] == 'official' and res['gate_eligible'] is True
    assert res['indeterminate'] is False
    assert res['status'] == 'pass' and res['passed'] is True
    assert res['status_strict'] == 'pass' and res['passed_strict'] is True
    crit = res['criteria']
    assert crit['ic_ge_005'] and crit['p_lt_005'] and crit['p_lt_0025']
    assert crit['spread_ge_3pp'] and crit['top_gt_all']
    assert res['primary']['ic'] == pytest.approx(1.0)
    assert res['primary']['p_unstratified'] == pytest.approx(1 / (h2.N_PERM + 1))
    # 잠금 십분위: 40지갑 → 십분위당 4, 전부 관측
    assert res['primary']['decile_n_obs'] == [4] * 10
    # 2026-08-28 영구 결손 공개 (합성 디렉토리에도 당연히 없음 — 항상 목록에 등장)
    assert '2026-08-28' in res['inputs']['missing_dates_in_window']
    assert res['inputs']['known_missing_days'] == ['2026-08-28']
    assert res['code']['h2_sha256'] == h1v.H2_SHA256_PINNED
    assert res['mde']['alpha05'] < res['mde']['alpha025']
    assert res['decision_basis'].startswith('H1 문서 원문')
    assert 'sensitivity_error' not in res


def test_verdict_fail_anti_persistence(tmp_path, monkeypatch):
    cohort, daily = build_scenario(tmp_path, reverse=True)
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    assert res['status'] == 'fail' and res['passed'] is False
    assert res['indeterminate'] is False
    assert res['primary']['ic'] == pytest.approx(-1.0)
    assert res['primary']['spread'] < 0


def test_primary_ic_single_source_with_ties_and_missing(tmp_path, monkeypatch):
    """동점·결측 혼재 시 1차 IC == 동결 spearman_avg 수계산 (순열 관측치와 동일원)."""
    n = 30
    rois = [6.0, 6.0, 5.5] + [1.0 - 0.05 * i for i in range(n - 3)]  # 클립 동점 유발
    cohort, daily = build_scenario(tmp_path, n=n, rois=rois, missing={7, 21})
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    scores = np.asarray([h1v.clip_score(r) for i, r in enumerate(rois)
                         if i not in {7, 21}])
    ys = np.asarray([0.5 - 0.01 * i for i in range(n) if i not in {7, 21}])
    assert res['primary']['ic'] == pytest.approx(
        h2.spearman_avg(scores, ys), abs=1e-12)
    assert res['primary']['n'] == n - 2


def test_interior_gap_immaterial(tmp_path, monkeypatch):
    """내부 결손일(08-28 류)은 2점 차분 1차 결과에 영향 없음 — 인벤토리만 다름."""
    ca, da = build_scenario(tmp_path / 'a')
    cb, db = build_scenario(tmp_path / 'b')
    # b 에만 내부 날짜 파일 추가 (판정일 행이 덮으므로 값은 무엇이든 무관해야 함)
    write_daily(db, '2026-09-15', [(addr_of(i), 12345.0, 999999.0)
                                   for i in range(40)])
    ra = run_patched(monkeypatch, ca, da, 30, None)
    rb = run_patched(monkeypatch, cb, db, 30, None)
    assert ra['primary'] == rb['primary']
    assert ra['criteria'] == rb['criteria'] and ra['status'] == rb['status']
    assert '2026-09-15' in ra['inputs']['missing_dates_in_window']
    assert '2026-09-15' not in rb['inputs']['missing_dates_in_window']


def test_indeterminate_missing_rate(tmp_path, monkeypatch):
    """결측률 > 10% → 판정불가 (기각 아님) — passed 는 null."""
    missing = {3, 9, 16, 22, 29, 35}              # 15%, 점수 전역에 분산
    cohort, daily = build_scenario(tmp_path, missing=missing)
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    assert res['missingness']['missing_rate'] == pytest.approx(0.15)
    assert res['missingness']['reasons'] == {'no_th': 6}
    assert res['indeterminate'] is True
    assert res['status'] == 'indeterminate' and res['passed'] is None
    assert res['passed_strict'] is None


def test_indeterminate_corr_and_rate_boundary(tmp_path, monkeypatch):
    """결측률 정확히 10% 는 '>10%' 미발동 — 상위 집중 결측은 상관 규칙이 잡는다."""
    cohort, daily = build_scenario(tmp_path, n=100, missing=set(range(10)))
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    mv = res['missingness']
    assert mv['missing_rate'] == pytest.approx(0.10)
    assert not mv['missing_rate'] > h2.MISS_RATE_MAX     # 경계: 비발동
    assert mv['p'] < h2.MISS_CORR_P                      # 상관 규칙 발동
    assert res['indeterminate'] is True and res['status'] == 'indeterminate'
    assert res['missingness']['decile_missing'][0] == 10


def test_ushaped_missingness_omnibus_warns_only(tmp_path, monkeypatch):
    """U자 탈락(상·하위 집중): 동결 Spearman 게이트는 못 잡고(단조 전용)
    옴니버스 경고가 잡는다 — 판정불가 규칙 자체는 동결 그대로 (권한 없음)."""
    missing = set(range(5)) | set(range(95, 100))        # D1·D10 각 5 = 10%
    cohort, daily = build_scenario(tmp_path, n=100, missing=missing)
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    mv = res['missingness']
    assert mv['missing_rate'] == pytest.approx(0.10)     # 비율 규칙 비발동
    assert mv['p'] > h2.MISS_CORR_P                      # 단조 상관 미검출
    assert res['indeterminate'] is False                 # 동결 규칙상 판정 유효
    assert mv['omnibus']['p'] < 0.05                     # 경고는 발동
    assert mv['decile_missing'][0] == 5 and mv['decile_missing'][9] == 5


def test_insufficient_evaluable(tmp_path, monkeypatch):
    """평가 가능 < 10 → 판정불가 (fail 아님), primary/criteria null."""
    cohort, daily = build_scenario(tmp_path, n=12, missing=set(range(5, 12)))
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    assert res['insufficient_evaluable'] == 5
    assert res['status'] == 'indeterminate' and res['passed'] is None
    assert res['primary'] is None and res['criteria'] is None


def test_exploratory_and_checkpoint(tmp_path, monkeypatch, pass_run):
    # 공식일 아님 → exploratory, 게이트 상태 미발행 (09-24 행으로 폴백, 스팬 30)
    res = run_patched(monkeypatch, pass_run['cohort'], pass_run['daily'],
                      30, '2026-09-26')
    assert res['analysis_status'] == 'exploratory'
    assert res['gate_eligible'] is False
    assert res['status'] is None and res['passed'] is None
    assert res['criteria'] is not None                   # 기술 보고는 유지
    # H=90 공식일 → descriptive_checkpoint (세 번째 기회 아님)
    cohort, daily = build_scenario(tmp_path / 'h90')
    write_daily(Path(daily), '2026-11-23',
                [(addr_of(i), 20000.0, 1000.0 * i + (0.5 - 0.01 * i) * 20000.0)
                 for i in range(40)])
    res = run_patched(monkeypatch, cohort, daily, 90, None)
    assert res['analysis_status'] == 'descriptive_checkpoint'
    assert res['gate_eligible'] is False
    assert res['status'] is None and res['passed'] is None


def test_sensitivity_failure_isolated(tmp_path, monkeypatch):
    """감도·진단 실패는 1차 판정을 죽이지 않는다 (보고 전용 격리)."""
    cohort, daily = build_scenario(tmp_path)
    def boom(*a, **k):
        raise RuntimeError('sensitivity crashed')
    monkeypatch.setattr(h1v, '_analysis_block', boom)
    res = run_patched(monkeypatch, cohort, daily, 30, None)
    assert 'sensitivity_error' in res
    assert res['status'] == 'pass' and res['passed'] is True
    assert res['criteria']['ic_ge_005'] is True


def test_h60_rolling_and_locf(tmp_path, monkeypatch):
    """H=60: T0 기점 누적 60일이 1차, 롤링 2개월차는 감도. LOCF 는 게이트 없는 전수."""
    n = 20
    tmp_path.mkdir(exist_ok=True)
    cohort = tmp_path / 'c.json.gz'
    daily = tmp_path / 'daily'
    daily.mkdir()
    write_cohort(cohort, [mk_wallet(i, 1.0 - 0.05 * i) for i in range(n)])
    write_daily(daily, T0, [(addr_of(i), 20000.0, 0.0) for i in range(n)])
    # 중간 기준선: T0+28 (롤링 기준선 게이트 [25,30] 충족)
    write_daily(daily, '2026-09-22',
                [(addr_of(i), 20000.0, (0.3 - 0.01 * i) * 20000.0) for i in range(n)])
    # 판정일 T0+60 — 지갑 19 는 누락 (마지막 유효 = T0+28 → 스팬 28 → 1차 제외)
    write_daily(daily, '2026-10-24',
                [(addr_of(i), 20000.0, (0.6 - 0.02 * i) * 20000.0)
                 for i in range(n - 1)])
    res = run_patched(monkeypatch, str(cohort), str(daily), 60, None)
    assert res['analysis_status'] == 'official' and res['horizon_days'] == 60
    assert res['judgment_date'] == '2026-10-24'
    assert res['primary']['n'] == n - 1                  # 지갑 19 span_out
    assert res['missingness']['reasons'] == {'span_out': 1}
    roll = res['sensitivity']['rolling_30_60']
    assert roll['n'] == n - 1                            # 롤링도 지갑 19 불가
    assert roll['ic'] == pytest.approx(1.0)              # 2개월차도 완전 단조 구성
    locf = res['sensitivity']['locf']
    assert locf['n'] == n                                # LOCF 는 게이트 없음 → 전수
    # 신선/스테일 분리: 전원 판정일 당일 → stale 블록 비어 있음
    assert res['sensitivity']['fresh_same_day']['n'] == n - 1
    assert res['sensitivity']['stale_fallback']['n'] == 0


# ── 최종 게이트 결합 ─────────────────────────────────────────────────────────
def _fake_t60(res30: dict) -> dict:
    r = copy.deepcopy(res30)
    r['horizon_days'] = 60
    r['judgment_date'] = '2026-10-24'
    r['evaluated_on'] = '2026-10-24'
    return r


def _combine(tmp_path, r30: dict, r60: dict) -> dict:
    p30, p60 = tmp_path / 't30.json', tmp_path / 't60.json'
    p30.write_text(json.dumps(h1v._sanitize(r30)), encoding='utf-8')
    p60.write_text(json.dumps(h1v._sanitize(r60)), encoding='utf-8')
    return h1v.combine_gate(str(p30), str(p60))


def _mark_fail(r: dict) -> dict:
    """기준·상태·passed 를 자기일관되게 fail 로 조작 (결합기 재계산 통과용)."""
    r = copy.deepcopy(r)
    r['criteria']['all_pass_doc_alpha'] = False
    r['criteria']['all_pass_strict_alpha'] = False
    r['criteria']['ic_ge_005'] = False
    r['status'] = r['status_strict'] = 'fail'
    r['passed'] = r['passed_strict'] = False
    return r


def _mark_indeterminate(r: dict) -> dict:
    r = copy.deepcopy(r)
    r['indeterminate'] = True
    r['status'] = r['status_strict'] = 'indeterminate'
    r['passed'] = r['passed_strict'] = None
    return r


def test_combine_gate_precedence(tmp_path, monkeypatch, pass_run):
    r30 = pass_run['res']
    pin_from(monkeypatch, r30)
    # pass + pass → pass (입력 순서 뒤집어도 정준화 — t30/t60 라벨 유지)
    g = _combine(tmp_path, r30, _fake_t60(r30))
    assert g['overall'] == 'pass' and g['overall_strict'] == 'pass'
    assert g['problems'] == []
    g_rev = h1v.combine_gate(str(tmp_path / 't60.json'), str(tmp_path / 't30.json'))
    assert g_rev['overall'] == 'pass'
    assert g_rev['statuses']['t30'] == 'pass' and g_rev['statuses']['t60'] == 'pass'
    # pass + indeterminate → indeterminate
    g = _combine(tmp_path, r30, _mark_indeterminate(_fake_t60(r30)))
    assert g['overall'] == 'indeterminate'
    # fail 은 indeterminate 를 지배 (AND 게이트 — 확정 실패 우선)
    g = _combine(tmp_path, _mark_fail(r30), _mark_indeterminate(_fake_t60(r30)))
    assert g['overall'] == 'fail'


def test_combine_gate_integrity(tmp_path, monkeypatch, pass_run):
    r30 = pass_run['res']
    pin_from(monkeypatch, r30)
    t60 = _fake_t60(r30)
    # 코호트 SHA ≠ 핀 → invalid
    bad = copy.deepcopy(t60)
    bad['inputs']['cohort']['sha256'] = '1' * 64
    assert _combine(tmp_path, r30, bad)['overall'] == 'invalid'
    # 서로 일치하는 가짜 코드 해시 → invalid (상호 비교가 아니라 현재 파일과 대조)
    f30, f60 = copy.deepcopy(r30), copy.deepcopy(t60)
    f30['code']['h1_sha256'] = f60['code']['h1_sha256'] = '2' * 64
    assert _combine(tmp_path, f30, f60)['overall'] == 'invalid'
    # horizon 중복 (30+30) → invalid
    assert _combine(tmp_path, r30, copy.deepcopy(r30))['overall'] == 'invalid'
    # 탐색 실행 → invalid
    ex = _fake_t60(r30)
    ex['analysis_status'] = 'exploratory'
    assert _combine(tmp_path, r30, ex)['overall'] == 'invalid'
    # 판정일 전 평가 (evaluated_on < 판정일) → invalid
    early = _fake_t60(r30)
    early['evaluated_on'] = '2026-10-23'
    assert _combine(tmp_path, r30, early)['overall'] == 'invalid'
    # 상태 무효값 → invalid
    weird = _fake_t60(r30)
    weird['status'] = 'maybe'
    assert _combine(tmp_path, r30, weird)['overall'] == 'invalid'
    # 상태·기준 불일치 (기준은 pass 인데 status=fail 조작) → invalid
    lie = _fake_t60(r30)
    lie['status'] = lie['status_strict'] = 'fail'
    lie['passed'] = lie['passed_strict'] = False
    assert _combine(tmp_path, r30, lie)['overall'] == 'invalid'
    # 형식 파괴 (horizon 이 리스트) → invalid, 예외 아님
    broken = _fake_t60(r30)
    broken['horizon_days'] = [60]
    assert _combine(tmp_path, r30, broken)['overall'] == 'invalid'


# ── CLI + JSON 표준 준수 ────────────────────────────────────────────────────
def _strict_loads(text: str):
    def _no_const(s):
        raise ValueError(f'비표준 JSON 상수: {s}')
    return json.loads(text, parse_constant=_no_const)


def test_cli_smoke_and_strict_json(tmp_path, monkeypatch, pass_run):
    with gzip.open(pass_run['cohort'], 'rt') as f:
        n = json.load(f)['n']
    monkeypatch.setattr(h1v, 'EXPECTED_COHORT_N', n)
    pin_from(monkeypatch, pass_run['res'])
    out = tmp_path / 'v30.json'
    # 탐색 판정일 (실제 시계와 무관) — 스테일 폴백 → fresh 블록 비어 NaN 후보 발생
    h1v.main(['--horizon', '30', '--judgment', '2026-09-26',
              '--cohort', pass_run['cohort'], '--daily-dir', pass_run['daily'],
              '--out-json', str(out)])
    text = out.read_text(encoding='utf-8')
    res = _strict_loads(text)                            # NaN/Infinity 리터럴 거부
    assert 'NaN' not in text and 'Infinity' not in text
    assert res['kind'] == 'h1_verdict' and res['analysis_status'] == 'exploratory'
    assert res['sensitivity']['fresh_same_day']['n'] == 0        # 빈 블록 → null 화
    assert res['sensitivity']['fresh_same_day']['top_median'] is None
    # combine CLI (공식 판정 JSON 두 개)
    p30, p60 = tmp_path / 't30.json', tmp_path / 't60.json'
    p30.write_text(json.dumps(h1v._sanitize(pass_run['res'])), encoding='utf-8')
    p60.write_text(json.dumps(h1v._sanitize(_fake_t60(pass_run['res']))),
                   encoding='utf-8')
    gate = tmp_path / 'gate.json'
    h1v.main(['--combine', str(p30), str(p60), '--out-json', str(gate)])
    assert _strict_loads(gate.read_text(encoding='utf-8'))['overall'] == 'pass'
