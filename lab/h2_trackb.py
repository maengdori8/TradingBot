"""H2 트랙 B(본검정) — 13주 형성 랭커(rank) + 게이트 상태(gate). 사전등록 동결 코드.

명세: docs/PREREGISTRATION_H2_2026-08-27.md §3.2 (형성·판정 — 랭커 코드
lab/h2_trackb.py 지금 작성·동결, 형성 데이터 조회 전). 모든 수치·식은 명세 그대로.
지위: 본검정 — 통과 시에도 2단계는 카피-팔로우 '페이퍼' 설계 진입일 뿐이며
실거래 권한이 아니다 (자본 관련 권한 최초 가능일 2027-01-25).

사전 고정 — rank (13주 형성 랭킹):
- T0_main = 2026-08-27. 형성 = T0부터 연속 7×24h 비중첩 블록 × 13 = 91일
  (ISO주 금지 — T0 앵커 블록), 형성 종료 2026-11-26.
- 경계 t_k = t0 라벨 스냅샷의 captured_at + k×(7×24h), k=0..13 (지갑별 앵커).
- 스냅 = t_k '이전'(후방만) 마지막 t0/daily 라벨 스냅샷(captured_at 기준),
  후방 48h 이내만 (일별 수집 + GitHub cron 지연 허용). 미래 스냅 사용 금지.
  verdict 라벨 행은 경계 스냅에 사용하지 않는다 (전방 기준선 전용).
- r_w = (P_perp(t_{k+1}스냅) − P_perp(t_k스냅)) / A(T0_main).
  A(T0_main) = t0 라벨 스냅샷 account_value, ≥ $10,000 필수 (미달 제외).
- 주 유효: 양끝 스냅 존재 AND 실제 스팬 [6, 8]일 AND 주시작 acct > 0.
- 흐름 필터: flow_frac = |Δacct − Δpnl| / acct(주시작). 1차 ≤ 20%, 감도 ≤ 50%
  — 트랙 A와 동일.
- 적격: 흐름-유효 주 ≥ 10/13 (감도 ≥ 8/13) AND 형성 총 perp PnL > 0
  (pnl(t_13 스냅) − pnl(t_0 스냅); t_13 스냅 결측이면 총 PnL 관측 불가 → 제외)
  AND fills 상태(logs/h2_fills_state.json)가 `fill-history-censored` 아님
  AND 형성 종료 시점에 7일 초과 미해소 gap-incomplete 아님.
- fills 사유 제외 지갑 수는 별도 집계·공개 — 그 비율이 '잔존'(fills 외 1차 기준
  전부 통과 지갑 수) 대비 30% 초과면 fills 기반 서브그룹 분석 판정불가 플래그.
- 랭킹 = ES20 (k = ceil(0.2 × 유효주수)) 내림차순 — 산식·랭킹은
  lab/h2_consistency 의 es_k/es20_of/variant_stats/_assign_ranks 를 import 재사용.
  빠른 H2(26주)와 "동일한 하방 일관성 구성개념의 다른 lookback" — 동일 estimator 아님.
- 산출물 logs/h2_trackb_cohort.json.gz — 헤더에 파라미터·입력 경위(SHA256)·생성
  시각, 지갑별 es20/rank/n_valid_weeks/a_t0/제외사유. 공용 판정 평가기
  (h2_consistency verdict --h2-cohort 이 파일)와 스키마 호환 (eligible/es20/rank/
  turnover/sens_flow50/sens_minw8).

사전 고정 — gate (게이트 상태):
- 각 판정 결과(h2_consistency verdict --out-json 산출 JSON)를 판정일·호라이즌
  인자와 함께 logs/h2_trackb_gate.json 에 누적 기록
  (판정일·3기준·p·표본·판정불가 여부·MDE 의무 보고).
- 이미 기록된 판정일: 동일 결과 SHA → 멱등 no-op (파일 불변), 상이한 결과 →
  충돌 거부. 다른 호라이즌의 공식 판정일로의 기록도 거부 (게이트 오염 방지).
- 출처 검증 (validate_result_provenance): kind=='h2_verdict', horizon_days ==
  --horizon, judgment_utc 존재 (공식 판정일 기록이면 일자 일치 필수),
  inputs.t0/th 라벨 == 'verdict' (트랙 B 기준선 계약), 코호트 SHA 존재 —
  위반 시 거부. 추가로 결과의 코호트 SHA 를 --cohort 파일(동결 트랙 B 랭킹
  산출물, 헤더 spec 확인)의 실측 SHA 와 대조 결속한다.
  CLI 날짜 인자만으로 임의 JSON 이 게이트를 열 수 없다.
- 2단계 진입: T+30(2026-12-26) AND T+60(2027-01-25) 판정이 모두 3기준
  (IC ≥ +0.05, 상위−하위 십분위 스프레드 ≥ +3%p, 상위 십분위 중앙 > 전체 중앙)
  + 단측 p < 0.025 를 전부 충족 AND 판정불가 아님 → stage2_eligible=true.
  공식 판정일(JUDGMENT_DATES)로 기록되고 두 판정의 코호트 SHA 가 동일한 행만
  stage2 에 산입 — 비공식 날짜 재판정·이질 코호트 짝으로 게이트를 열 수 없다.
  임계값은 h2_consistency 의 동결 상수를 import 재사용 (이중 정의 금지).

트랙 B 운영 절차 (형성 종료 후):
1. 형성 종료일(2026-11-26)에 logs/h2_fills_state.json 사본 동결 → rank 의
   --fills-state 입력 (as-of 평가 재현 — fills_exclusion docstring 참조).
2. rank 실행 → logs/h2_trackb_cohort.json.gz.
3. 판정일 verdict (공용 평가기 — 트랙 B 전방 기준선은 형성 종료일 verdict 라벨
   스냅샷이다, T0_main 아님 — 명세 §3.2 '전방 기준선'):
   python lab/h2_consistency.py verdict \
       --h2-cohort logs/h2_trackb_cohort.json.gz \
       --t0 logs/h2_snapshots/2026-11-26.jsonl.gz --t0-label verdict \
       --th logs/h2_snapshots/<판정일>.jsonl.gz --th-label verdict \
       --horizon 30 --judgment <판정시각> --out-json logs/h2_trackb_verdict_t30.json
4. gate 기록: python lab/h2_trackb.py gate --result ... --judgment-date ... --horizon ...

실행 (cwd = 저장소 루트):
  python lab/h2_trackb.py rank [--snapshots-dir logs/h2_snapshots]
      [--fills-state logs/h2_fills_state.json] [--cohort logs/trader_cohort.json.gz]
      [--out logs/h2_trackb_cohort.json.gz]
  python lab/h2_trackb.py gate --result VERDICT.json --judgment-date 2026-12-26
      --horizon 30 [--cohort logs/h2_trackb_cohort.json.gz]
      [--gate logs/h2_trackb_gate.json]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:                    # python lab/h2_trackb.py 직접 실행 지원
    sys.path.insert(0, _ROOT)

from lab.h2_consistency import (  # noqa: E402  (경로 셋업 후 import)
    DAY_MS,
    ES_FRAC,
    IC_PASS_MIN,
    MAX_FLOW_FRAC,
    MIN_ACCT,
    P_ONE_SIDED,
    SENS_FLOW_FRAC,
    SPREAD_PASS_MIN,
    WEEK_MS,
    _assign_ranks,
    es20_of,
    es_k,
    iso_utc,
    mde_ic,
    parse_iso_ms,
    sha256_of,
    snap_backward,
    variant_stats,
)

# 명세 §3.2: ES20·k 산식은 빠른 H2와 동일 구현을 import 재사용 (재정의 금지).
# variant_stats 가 내부에서 es_k/es20_of 를 호출한다 — 참조를 모듈에 고정해 둔다.
_REUSED_ES = (es_k, es20_of)

logger = logging.getLogger(__name__)

# ── 사전 고정 파라미터 (명세 §3.2 — 변경 금지) ──────────────────────────────
N_WEEKS_B = 13                        # 형성 = 7×24h 블록 × 13 = 91일
SNAP_BACK_B_MS = 48 * 3600 * 1000     # 경계 후방 스냅 허용 48h (일별 수집 + cron 지연)
SPAN_B_MIN_D, SPAN_B_MAX_D = 6.0, 8.0  # 주 실제 스팬 유효 범위
MIN_VALID_WEEKS_B = 10                # 적격: 흐름-유효 주 ≥ 10/13
SENS_MIN_VALID_WEEKS_B = 8            # 감도: ≥ 8/13
GAP_UNRESOLVED_MAX_MS = 7 * DAY_MS    # 형성종료 시점 미해소 gap 허용 한도 7일
FILLS_EXCL_MAX_FRAC = 0.30            # fills 제외 비율 > 30% → 서브그룹 판정불가
STATUS_CENSORED = 'fill-history-censored'   # carrybot/live/fills_recorder.py 계약 문자열
SNAP_LABELS = ('t0', 'daily')         # 경계 스냅에 쓰는 라벨 (verdict 제외)
# 게이트 (명세 §3.2 판정 일정 — 날짜 불일치는 경고만, 기록은 인자 우선)
JUDGMENT_DATES = {30: '2026-12-26', 60: '2027-01-25', 90: '2027-02-24'}
STAGE2_HORIZONS = (30, 60)            # 2단계 진입: T+30 AND T+60 모두 충족
STAGE2_EARLIEST = '2027-01-25'        # 페이퍼 설계 진입 최초 가능일 (실거래 아님)


# ── 스냅샷 로드 (rank) ──────────────────────────────────────────────────────
def load_snapshot_series(paths: list[str]) -> tuple[dict[str, dict], dict[str, dict], Counter]:
    """스냅샷 파일들에서 t0 맵과 지갑별 t0/daily 스냅 시계열을 로드한다.

    파일은 이름 정렬(YYYY-MM-DD 일자) + 줄 순서로 읽는다 (시간순 근사) —
    t0 라벨은 주소별 keep-last (재시도 갱신 가정, h2_consistency 계약과 동일).
    verdict 라벨 행은 경계 스냅에 사용하지 않으므로 집계만 하고 버린다.
    label 필드 없는 행·파싱 불가 행·비유한 값 행은 거부 카운트.

    반환: (t0맵 addr→{'pnl','acct','cap_ms'},
           시계열 addr→{'ts','pnl','acct'} — cap_ms 오름차순 np.ndarray,
           동일 cap_ms 는 keep-last,
           집계 Counter — label_t0/label_daily/label_<기타>/no_label/bad_json/
           bad_row/nonfinite).
    """
    counts: Counter = Counter()
    t0_map: dict[str, dict] = {}
    acc: dict[str, list[tuple[int, float, float]]] = {}
    for path in sorted(paths):
        with gzip.open(path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    counts['bad_json'] += 1
                    continue
                if not isinstance(r, dict):
                    counts['bad_row'] += 1
                    continue
                label = r.get('label')
                if label is None:
                    counts['no_label'] += 1
                    continue
                if label not in SNAP_LABELS:
                    counts[f'label_{label}'] += 1   # verdict 등 — 경계 스냅 미사용
                    continue
                try:
                    addr = str(r['address']).lower()
                    pnl = float(r['perp_alltime_pnl'])
                    acct = float(r['account_value'])
                    cap_ms = parse_iso_ms(r['captured_at_utc'])
                except (KeyError, TypeError, ValueError):
                    counts['bad_row'] += 1
                    continue
                if not (math.isfinite(pnl) and math.isfinite(acct)):
                    counts['nonfinite'] += 1
                    continue
                counts[f'label_{label}'] += 1
                if label == 't0':
                    if addr in t0_map:
                        counts['t0_dup_overwritten'] += 1   # T0 앵커 이동 진단
                    t0_map[addr] = {'pnl': pnl, 'acct': acct, 'cap_ms': cap_ms}
                acc.setdefault(addr, []).append((cap_ms, pnl, acct))
    series: dict[str, dict] = {}
    for addr, rows in acc.items():
        dedup: dict[int, tuple[int, float, float]] = {}
        for row in rows:                     # 동일 cap_ms keep-last (읽은 순서)
            dedup[row[0]] = row
        rows2 = sorted(dedup.values(), key=lambda x: x[0])
        series[addr] = {
            'ts': np.asarray([x[0] for x in rows2], dtype=float),
            'pnl': np.asarray([x[1] for x in rows2], dtype=float),
            'acct': np.asarray([x[2] for x in rows2], dtype=float),
        }
    return t0_map, series, counts


# ── 형성 (지갑 단위) ────────────────────────────────────────────────────────
def form_wallet(t0_cap_ms: float, a_t0: float, ts: np.ndarray,
                pnl: np.ndarray, acct: np.ndarray) -> dict:
    """지갑 1개의 13주 형성 블록 계산 (명세 §3.2).

    경계 t_k = t0 captured_at + k×168h (k=0..13), 스냅 = t_k 이전(후방만) 마지막
    스냅샷 — 후방 48h 이내만 (snap_backward 재사용, 미래 스냅 금지).
    총 PnL = pnl(t_13 스냅) − pnl(t_0 스냅) — t_13 스냅 결측이면 관측 불가로
    ok=False (no_formation_end_snap).
    주 유효: 양끝 스냅 존재 AND 스팬 [6,8]일 AND 주시작 acct > 0. 흐름 한도는
    여기서 걸지 않고 flow_frac 만 기록한다 (1차·감도 변형을 variant_stats 로
    일괄 적용 — 트랙 A screen_wallet 과 동일 구조).

    반환: ok=True 시 total_pnl / weeks(주별 r_w·flow_frac·span_d) /
    formation_end_ms, ok=False 시 reason.
    """
    bounds = t0_cap_ms + WEEK_MS * np.arange(N_WEEKS_B + 1, dtype=float)
    snaps = [snap_backward(ts, float(b), tol_ms=SNAP_BACK_B_MS) for b in bounds]
    if snaps[0] is None:
        return {'ok': False, 'reason': 'no_t0_snap'}
    if snaps[N_WEEKS_B] is None:
        return {'ok': False, 'reason': 'no_formation_end_snap'}
    total_pnl = float(pnl[snaps[N_WEEKS_B]] - pnl[snaps[0]])
    weeks: list[dict] = []
    for w in range(N_WEEKS_B):
        ia, ib = snaps[w], snaps[w + 1]
        if ia is None or ib is None or ib <= ia:
            continue
        span_d = float((ts[ib] - ts[ia]) / DAY_MS)
        if span_d < SPAN_B_MIN_D or span_d > SPAN_B_MAX_D:
            continue
        a0w = float(acct[ia])
        if a0w <= 0:
            continue
        dpnl = float(pnl[ib] - pnl[ia])
        da = float(acct[ib] - acct[ia])
        weeks.append({'w': w, 'r_w': dpnl / a_t0,
                      'flow_frac': abs(da - dpnl) / a0w,
                      'span_d': span_d})
    return {'ok': True, 'total_pnl': total_pnl, 'weeks': weeks,
            'formation_end_ms': float(bounds[N_WEEKS_B])}


def fills_exclusion(wst: dict | None, formation_end_ms: float) -> str | None:
    """fills 상태(logs/h2_fills_state.json 의 wallets[addr]) 기반 제외 사유.

    명세 §3.2 적격: `fill-history-censored` 아님 AND 형성 종료 시점에 7일 초과
    미해소 gap-incomplete 아님 — 두 조건 모두 '형성 종료 시점' 기준(as-of).
    - status == fill-history-censored → 'fill_history_censored'.
      단 censored_at 이 형성 종료 '이후'면 형성 기간의 절단이 아니므로 제외 아님
      (as-of 평가 — 상태 파일은 가변이므로 형성 종료일 사본 사용 권장, 하단 참조).
    - incomplete(미해소 gap) 이고 갭 종점(gap_until_ts, 결손 구간의 끝)이 형성
      종료보다 7일 초과 과거 → 'gap_incomplete_over_7d'. 7일 유예의 취지는 복구
      가능성이다: 결손 구간이 오래됐는데 아직 미해소면 cap 윈도(최근 1만 건)
      슬라이드로 영구 소실 가능성이 높고, 최근(≤7일) 갭은 후속 폴링이 덮어 복구할
      여지가 있다 — 따라서 척도는 갭 '탐지 시각'이 아니라 결손 구간 종점이다.
      gap_until_ts 없는 incomplete 는 나이 미상 → 보수적으로 동일 제외.
    - 상태 없음(미폴링)·정상·최근(≤7일) 갭 → None (제외 아님 — 수집기 자체의
      기술 실패는 지갑 제외가 아니라 §3.1 판정불가 경로; 잔존 중 상태 결측 수는
      헤더에 별도 공개).

    as-of 주의: 형성 종료 후에도 일별 폴링이 계속되며 recorder 는 갭 복구 시
    incomplete/gap_until_ts 를 지운다. rank 는 --fills-state 로 받은 파일을 그대로
    쓰므로, 형성 종료일(2026-11-26)에 logs/h2_fills_state.json 사본을 동결해 두고
    그 사본을 입력해야 '형성 종료 시점' 평가가 재현된다 (censored 방향은
    censored_at 으로 as-of 를 코드에서 보정한다).
    """
    if not wst:
        return None
    if wst.get('status') == STATUS_CENSORED:
        cens_at = wst.get('censored_at')
        try:
            cens_ms = parse_iso_ms(str(cens_at)) if cens_at else None
        except ValueError:
            cens_ms = None
        if cens_ms is None or cens_ms <= formation_end_ms:
            return 'fill_history_censored'
        # 형성 종료 이후 절단 — 형성 기간 데이터는 절단 전 확보됨 → 제외 아님
    if wst.get('incomplete'):
        g = wst.get('gap_until_ts')
        if g is None or formation_end_ms - float(g) > GAP_UNRESOLVED_MAX_MS:
            return 'gap_incomplete_over_7d'
    return None


def _empty_stats() -> dict:
    """형성 계산 불가 지갑의 variant_stats 자리 채움 (스키마 일관성)."""
    return {'n_valid_weeks': 0, 'k': None, 'es20': None,
            'pos_week_frac': None, 'median_rw': None, 'iqr_rw': None,
            'eligible': False}


# ── rank 서브커맨드 ─────────────────────────────────────────────────────────
def cmd_rank(args: argparse.Namespace) -> None:
    """13주 형성 랭킹 실행 → logs/h2_trackb_cohort.json.gz 산출."""
    paths = sorted(glob.glob(str(Path(args.snapshots_dir) / '*.jsonl.gz')))
    if not paths:
        sys.exit(f'스냅샷 없음: {args.snapshots_dir}/*.jsonl.gz')
    fills_path = Path(args.fills_state)
    if not fills_path.exists():
        sys.exit(f'fills 상태 파일 없음: {fills_path} — 적격식의 fills 조건 평가 불가')
    fills_state = json.loads(fills_path.read_text(encoding='utf-8'))
    fills_map = {str(k).lower(): v
                 for k, v in fills_state.get('wallets', {}).items()}
    with gzip.open(args.cohort, 'rt') as f:
        coh = json.load(f)
    print(f"코호트 {len(coh['wallets'])}지갑 (locked_at={coh.get('locked_at')}) "
          f"— 신규 스크린 없음")

    t0_map, series, row_counts = load_snapshot_series(paths)
    print(f"스냅샷 {len(paths)}파일 — 행 집계 {dict(row_counts)}")

    excl: Counter = Counter()
    wallets_out: list[dict] = []
    n_with_t0 = 0
    n_base_pass = 0          # fills 외 1차 기준(유효주·총PnL·A(T0)) 통과 = '잔존'
    n_fills_excluded = 0     # 잔존 중 fills 사유로만 제외된 수 (별도 공개)
    n_no_state = 0           # 잔존 중 fills 상태 자체가 없는 수 (§3.1 기술실패 진단)
    for cw in coh['wallets']:
        addr = str(cw['address']).lower()
        t0a_coh = float(cw.get('t0_account') or 0.0)
        t0v_coh = float(cw.get('t0_month_vlm') or 0.0)
        rec: dict = {
            'address': addr, 'a_t0': None, 't0_captured_at_utc': None,
            'formation_end_utc': None, 'total_pnl': None,
            'eligible': False, 'exclusion': None,
            'n_valid_weeks': 0, 'k': None, 'es20': None, 'rank': None,
            'pos_week_frac': None, 'median_rw': None, 'iqr_rw': None,
            't0_month_vlm': t0v_coh,
            'turnover': (t0v_coh / t0a_coh) if t0a_coh > 0 else None,
            'fills_exclusion': None,
            'sens_flow50': {**_empty_stats(), 'rank': None},
            'sens_minw8': {**_empty_stats(), 'rank': None},
        }
        wallets_out.append(rec)
        t0 = t0_map.get(addr)
        if t0 is None:
            rec['exclusion'] = 'no_t0'
            excl['no_t0'] += 1
            continue
        n_with_t0 += 1
        rec['a_t0'] = t0['acct']
        rec['t0_captured_at_utc'] = iso_utc(t0['cap_ms'])
        # fills 사유는 t0 앵커만 있으면 평가 가능 (형성 종료 = t0 + 13×168h) —
        # 다른 사유로 먼저 제외되는 지갑에도 별도 필드로 기록한다 (투명성).
        fe_ms = float(t0['cap_ms']) + N_WEEKS_B * WEEK_MS
        f_excl = fills_exclusion(fills_map.get(addr), fe_ms)
        fills_ok = f_excl is None
        rec['fills_exclusion'] = f_excl
        if t0['acct'] < MIN_ACCT:
            rec['exclusion'] = 'a_t0_below_min'
            excl['a_t0_below_min'] += 1
            continue
        ser = series.get(addr)
        if ser is None:                       # t0 행이 시계열에 들어가므로 방어용
            rec['exclusion'] = 'no_series'
            excl['no_series'] += 1
            continue
        res = form_wallet(float(t0['cap_ms']), float(t0['acct']),
                          ser['ts'], ser['pnl'], ser['acct'])
        if not res['ok']:
            rec['exclusion'] = res['reason']
            excl[res['reason']] += 1
            continue
        rec['formation_end_utc'] = iso_utc(res['formation_end_ms'])
        rec['total_pnl'] = res['total_pnl']
        primary = variant_stats(res['weeks'], res['total_pnl'],
                                MAX_FLOW_FRAC, MIN_VALID_WEEKS_B)
        flow50 = variant_stats(res['weeks'], res['total_pnl'],
                               SENS_FLOW_FRAC, MIN_VALID_WEEKS_B)
        minw8 = variant_stats(res['weeks'], res['total_pnl'],
                              MAX_FLOW_FRAC, SENS_MIN_VALID_WEEKS_B)
        rec.update({k: primary[k] for k in ('n_valid_weeks', 'k', 'es20',
                                            'pos_week_frac', 'median_rw',
                                            'iqr_rw')})
        rec['sens_flow50'] = {**flow50, 'rank': None,
                              'eligible': bool(flow50['eligible'] and fills_ok)}
        rec['sens_minw8'] = {**minw8, 'rank': None,
                             'eligible': bool(minw8['eligible'] and fills_ok)}
        if primary['eligible']:
            n_base_pass += 1
            if fills_map.get(addr) is None:
                n_no_state += 1          # 잔존 중 fills 상태 결측 (정보 공개용)
            if fills_ok:
                rec['eligible'] = True
            else:
                n_fills_excluded += 1
                rec['exclusion'] = f_excl
                excl[f_excl] += 1
        else:
            reason = ('insufficient_valid_weeks'
                      if primary['n_valid_weeks'] < MIN_VALID_WEEKS_B
                      else 'nonpositive_total_pnl')
            rec['exclusion'] = reason
            excl[reason] += 1

    # 랭킹 (변형별 독립, ES20 내림차순 — 동점은 주소 오름차순)
    n_primary = _assign_ranks(
        wallets_out,
        lambda w: w['es20'] if w['eligible'] else None,
        lambda w, rk: w.__setitem__('rank', rk))
    n_flow50 = _assign_ranks(
        wallets_out,
        lambda w: w['sens_flow50']['es20'] if w['sens_flow50']['eligible'] else None,
        lambda w, rk: w['sens_flow50'].__setitem__('rank', rk))
    n_minw8 = _assign_ranks(
        wallets_out,
        lambda w: w['sens_minw8']['es20'] if w['sens_minw8']['eligible'] else None,
        lambda w, rk: w['sens_minw8'].__setitem__('rank', rk))

    fills_frac = (n_fills_excluded / n_base_pass) if n_base_pass else None
    fills_indet = bool(fills_frac is not None and fills_frac > FILLS_EXCL_MAX_FRAC)

    header = {
        'spec': 'H2 트랙 B 13주 형성 랭커 (docs/PREREGISTRATION_H2_2026-08-27.md §3.2)',
        'generated_at_utc': datetime.now(tz=timezone.utc).isoformat(),
        'inputs': {
            'snapshots': [{'path': p, 'sha256': sha256_of(p)} for p in paths],
            'fills_state': {'path': str(fills_path),
                            'sha256': sha256_of(str(fills_path))},
            'cohort': {'path': args.cohort, 'sha256': sha256_of(args.cohort)},
        },
        'params': {
            'week_hours': 168, 'n_weeks': N_WEEKS_B,
            'snap_backward_hours': SNAP_BACK_B_MS / 3.6e6,
            'span_min_days': SPAN_B_MIN_D, 'span_max_days': SPAN_B_MAX_D,
            'max_flow_frac': MAX_FLOW_FRAC, 'min_acct_usd': MIN_ACCT,
            'min_valid_weeks': MIN_VALID_WEEKS_B, 'es_frac': ES_FRAC,
            'gap_unresolved_max_days': GAP_UNRESOLVED_MAX_MS / DAY_MS,
            'fills_excl_max_frac': FILLS_EXCL_MAX_FRAC,
            'boundary_rule': 't_k = t0 라벨 captured_at + k×168h (k=0..13, 지갑별 앵커), '
                             '스냅 = t_k 이전 마지막 t0/daily 라벨 스냅 '
                             '(후방 48h 이내만, 미래 스냅 금지)',
            'eligibility': '흐름-유효 주 ≥ 10/13 AND 형성 총 perp PnL > 0 AND '
                           'fill-history-censored 아님 AND 형성종료 시점 '
                           '7일 초과 미해소 gap-incomplete 아님',
            'es20': 'k = ceil(0.2 × n_유효주), 최저 k개 r_w 산술평균, 내림차순 랭킹 '
                    '(h2_consistency es_k/es20_of 재사용 — 26주판과 동일 구성개념의 '
                    '다른 lookback, 동일 estimator 아님)',
        },
        'sensitivity_params': {
            'flow50': {'max_flow_frac': SENS_FLOW_FRAC,
                       'min_valid_weeks': MIN_VALID_WEEKS_B},
            'minw8': {'max_flow_frac': MAX_FLOW_FRAC,
                      'min_valid_weeks': SENS_MIN_VALID_WEEKS_B},
        },
        'counts': {
            'cohort': len(coh['wallets']), 'with_t0': n_with_t0,
            'eligible_primary': n_primary, 'eligible_flow50': n_flow50,
            'eligible_minw8': n_minw8, 'exclusions': dict(excl),
            'snapshot_rows': dict(row_counts),
        },
        'fills': {
            'n_excluded': n_fills_excluded,
            'n_surviving_before_fills': n_base_pass,
            'excluded_frac': fills_frac,
            'subgroup_indeterminate': fills_indet,
            'n_no_fills_state_among_surviving': n_no_state,
            'note': "분모 '잔존' = fills 외 1차 기준(유효주 ≥10/13·총 PnL>0·"
                    "A(T0)≥$10k) 전부 통과한 지갑 수. n_no_fills_state 는 제외가 "
                    "아니라 수집기 기술실패 진단용 (§3.1 판정불가 판단 근거)",
        },
        'mde': {'n_primary': n_primary, 'ic': mde_ic(n_primary),
                'alpha_one_sided': 0.025, 'power': 0.80,
                'formula': '(z_.975 + z_.80) / sqrt(n-3)'},
    }
    with gzip.open(args.out, 'wt') as f:
        json.dump({'header': header, 'wallets': wallets_out}, f, ensure_ascii=False)

    es_arr = np.asarray([w['es20'] for w in wallets_out if w['eligible']],
                        dtype=float)
    print(f"\n1차 적격 {n_primary}지갑 (감도: 흐름≤50% {n_flow50}, 유효주≥8 {n_minw8})")
    print(f"제외 사유: {dict(excl)}")
    print(f"fills 사유 제외 {n_fills_excluded}/{n_base_pass} (잔존 대비 "
          f"{fills_frac * 100:.1f}%)" if fills_frac is not None
          else "fills 사유 제외: 잔존 0 — 비율 계산 불가")
    if fills_indet:
        print(f"→ fills 서브그룹 판정불가 플래그 (제외 비율 > "
              f"{FILLS_EXCL_MAX_FRAC * 100:.0f}%)")
    if len(es_arr):
        q = np.quantile(es_arr, [0.1, 0.25, 0.5, 0.75, 0.9]) * 100
        print(f"ES20 분포(1차, %): p10 {q[0]:+.2f} p25 {q[1]:+.2f} 중앙 {q[2]:+.2f} "
              f"p75 {q[3]:+.2f} p90 {q[4]:+.2f}")
    m = mde_ic(n_primary)
    print(f"MDE: 단측 α=.025, power 80%에서 검출가능 IC ≈ {m:.3f} (n={n_primary})"
          if m else "MDE: 표본 부족")
    print(f"\n산출물: {args.out} (입력 SHA256 헤더 기록)")


# ── gate 서브커맨드 ─────────────────────────────────────────────────────────
def parse_verdict_result(doc: dict) -> dict:
    """verdict --out-json 산출물에서 게이트 기록 지표를 추출·검증한다.

    main 블록이 None(평가 가능 <10)이면 지표 전부 None — 기준 전부 미충족 처리.
    main 존재 시 n/ic/p/spread/top_median/all_median 필수 (결측이면 ValueError).
    indeterminate 는 문서 최상위(판정불가 규칙 — 결측률·결측~점수 상관) 우선.
    """
    ind = bool(doc.get('indeterminate', False))
    main = doc.get('main')
    if main is None:
        return {'n': None, 'ic': None, 'p': None, 'spread': None,
                'top_median': None, 'all_median': None, 'indeterminate': ind}
    if not isinstance(main, dict):
        raise ValueError('판정 결과 main 블록 형식 오류 (dict 아님)')
    n_raw = main.get('n')
    if isinstance(n_raw, bool) or not (
            isinstance(n_raw, int)
            or (isinstance(n_raw, float) and math.isfinite(n_raw)
                and float(n_raw).is_integer())):
        raise ValueError(f'판정 결과 n 정수 아님: {n_raw!r}')
    if int(n_raw) < 10:
        # 평가기 계약: 평가 가능 <10 이면 main=None — main 이 있으면서 n<10 은 손상
        raise ValueError(f'판정 결과 n={int(n_raw)} < 10 (평가기 계약 위반)')
    try:
        out: dict = {'n': int(n_raw), 'ic': float(main['ic']),
                     'p': float(main['p']), 'spread': float(main['spread']),
                     'top_median': float(main['top_median']),
                     'all_median': float(main['all_median'])}
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f'판정 결과 필수 필드 결측/형식 오류: {e}') from e
    # 값 범위 검증 — json 의 비표준 Infinity/NaN 이나 손상 결과로 게이트가
    # 잘못 열리는 것을 차단 (통과 판정은 유한한 지표에서만 의미를 가짐).
    for key in ('ic', 'p', 'spread', 'top_median', 'all_median'):
        if not math.isfinite(out[key]):
            raise ValueError(f'판정 결과 {key} 비유한값: {out[key]}')
    if not 0.0 <= out['p'] <= 1.0:
        raise ValueError(f"판정 결과 p 범위 밖 [0,1]: {out['p']}")
    if not -1.0 <= out['ic'] <= 1.0:
        raise ValueError(f"판정 결과 IC 범위 밖 [-1,1]: {out['ic']}")
    out['indeterminate'] = ind
    return out


def validate_result_provenance(doc: dict, horizon: int, judgment_date: str) -> str:
    """게이트 입력 결과 JSON 의 출처(provenance)를 검증하고 코호트 SHA256 반환.

    공식 판정일 CLI 인자만으로 임의 JSON 이 stage2 를 여는 것을 차단한다:
    - kind == 'h2_verdict' (verdict --out-json 산출물).
    - horizon_days 존재 AND == --horizon.
    - judgment_utc 존재; --judgment-date 가 해당 호라이즌의 공식 판정일이면
      judgment_utc 의 일자와 일치 필수 (비공식 날짜는 cmd_gate 가 경고만).
    - inputs.t0.label == inputs.th.label == 'verdict' — 트랙 B 기준선 계약
      (전방 기점 = 형성종료일 verdict 스냅, T_H = 판정일 verdict 스냅, 명세 §3.2).
    - inputs.h2_cohort.sha256 존재 (T+30/T+60 동일 코호트 검증은
      stage2_from_entries 에서 수행).
    위반 시 ValueError.
    """
    if doc.get('kind') != 'h2_verdict':
        raise ValueError(f"결과 JSON kind={doc.get('kind')!r} ≠ 'h2_verdict'")
    doc_h = doc.get('horizon_days')
    if doc_h is None or float(doc_h) != float(horizon):
        raise ValueError(f'결과 JSON horizon_days={doc_h!r} ≠ --horizon={horizon}')
    jd = doc.get('judgment_utc')
    if not isinstance(jd, str) or len(jd) < 10:
        raise ValueError(f'결과 JSON judgment_utc 결측/형식 오류: {jd!r}')
    if JUDGMENT_DATES.get(int(horizon)) == judgment_date and jd[:10] != judgment_date:
        raise ValueError(f'공식 판정일 {judgment_date} 기록인데 결과 판정시각 '
                         f'{jd} 의 일자가 다름 — 잘못된 판정 결과')
    inputs = doc.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('결과 JSON inputs 블록 결측')
    for key in ('t0', 'th'):
        label = (inputs.get(key) or {}).get('label')
        if label != 'verdict':
            raise ValueError(f"결과 JSON inputs.{key}.label={label!r} ≠ 'verdict' "
                             f'— 트랙 B 기준선(형성종료/판정일 verdict 스냅) 아님')
    sha = (inputs.get('h2_cohort') or {}).get('sha256')
    if not sha:
        raise ValueError('결과 JSON inputs.h2_cohort.sha256 결측')
    return str(sha)


def gate_criteria(m: dict) -> dict:
    """3기준+p 평가 — 임계값은 h2_consistency 동결 상수 재사용.

    지표가 None(평가 불능)이면 전 기준 미충족.
    """
    if m['ic'] is None:
        return {'ic_ge_min': False, 'p_lt_alpha': False,
                'spread_ge_min': False, 'top_gt_all': False}
    return {'ic_ge_min': bool(m['ic'] >= IC_PASS_MIN),
            'p_lt_alpha': bool(m['p'] < P_ONE_SIDED),
            'spread_ge_min': bool(m['spread'] >= SPREAD_PASS_MIN),
            'top_gt_all': bool(m['top_median'] > m['all_median'])}


def build_gate_entry(doc: dict, judgment_date: str, horizon: int,
                     result_path: str, result_sha256: str) -> dict:
    """판정 결과 1건의 게이트 기록 행을 만든다 (판정일·3기준·p·표본·판정불가·MDE).

    출처 필드(judgment_utc·코호트 SHA)도 함께 기록한다 — 검증은
    validate_result_provenance (cmd_gate 경유) 가 선행한다.
    """
    m = parse_verdict_result(doc)
    crit = gate_criteria(m)
    passed = bool(all(crit.values()) and not m['indeterminate'])
    return {'judgment_date': judgment_date, 'horizon_days': int(horizon),
            'recorded_at_utc': datetime.now(tz=timezone.utc).isoformat(),
            'result_path': result_path, 'result_sha256': result_sha256,
            'judgment_utc': doc.get('judgment_utc'),
            'cohort_sha256': ((doc.get('inputs') or {}).get('h2_cohort')
                              or {}).get('sha256'),
            'n': m['n'], 'ic': m['ic'], 'p': m['p'], 'spread': m['spread'],
            'top_median': m['top_median'], 'all_median': m['all_median'],
            'mde_ic': mde_ic(m['n']) if m['n'] else None,
            'indeterminate': m['indeterminate'],
            'criteria': crit, 'passed': passed}


def stage2_from_entries(entries: list[dict]) -> bool:
    """2단계(카피-팔로우 페이퍼 설계) 진입 판정.

    T+30 AND T+60 판정이 모두 passed(3기준+p 전부 충족 AND 판정불가 아님)여야
    True. 각 호라이즌은 명세의 공식 판정일(JUDGMENT_DATES)로 기록된 행만
    인정한다 — 비공식 날짜의 재판정 행(기술 기록)이 게이트를 열 수 없다.
    추가로 두 판정이 '같은 트랙 B 코호트'(cohort_sha256 동일·비결측)에서
    나왔어야 한다 — 서로 다른 랭킹 산출물의 결과를 짝지어 열 수 없다.
    그 외 조합(한쪽만 충족, 미기록, 판정불가)은 전부 False.
    """
    by_h: dict[int, dict] = {}
    for e in entries:
        if e.get('passed') and \
                e.get('judgment_date') == JUDGMENT_DATES.get(e.get('horizon_days')):
            by_h[e['horizon_days']] = e
    if not all(h in by_h for h in STAGE2_HORIZONS):
        return False
    shas = {by_h[h].get('cohort_sha256') for h in STAGE2_HORIZONS}
    return len(shas) == 1 and None not in shas


def load_gate(path: Path) -> dict:
    """게이트 상태 파일 로드 (없으면 빈 골격)."""
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'spec': 'H2 트랙 B 게이트 상태 '
                    '(docs/PREREGISTRATION_H2_2026-08-27.md §3.2)',
            'stage2_rule': 'T+30 AND T+60 모두 3기준+p 전부 충족 (판정불가 제외)',
            'stage2_earliest_utc': STAGE2_EARLIEST,
            'stage2_note': '충족해도 카피-팔로우 페이퍼 설계 진입일 뿐 — '
                           '실거래 권한 아님',
            'stage2_eligible': False, 'entries': []}


def save_gate(gate: dict, path: Path) -> None:
    """게이트 상태를 임시파일 경유로 원자적 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    tmp.replace(path)


def cmd_gate(args: argparse.Namespace) -> None:
    """판정 결과 1건을 게이트에 누적 기록하고 stage2_eligible 을 갱신한다.

    가드 (게이트 오염 방지):
    - 출처 검증 (validate_result_provenance): kind/horizon_days/judgment_utc/
      inputs 라벨('verdict'×2)/코호트 SHA — 위반 시 거부. 공식 판정일 기록이면
      judgment_utc 일자 일치도 필수.
    - 동결 코호트 결속: 결과의 코호트 SHA == --cohort 파일(기본
      logs/h2_trackb_cohort.json.gz) 실측 SHA AND 그 헤더가 트랙 B 랭커
      산출물임을 확인 — 아니면 거부.
    - --judgment-date 가 다른 호라이즌의 공식 판정일이면 거부 (미래 공식 기록
      자리 보호). 해당 호라이즌의 공식 판정일과 다른 날짜는 기록은 허용(경고)
      하되 stage2 판정에는 산입되지 않는다 (stage2_from_entries — 공식 날짜만).
    - 같은 판정일 재실행: 결과 SHA 동일 → 멱등 no-op (파일 불변, 정상 종료),
      결과 SHA 상이 → 충돌로 거부 (비정상 종료 — 자동화가 감지 가능).
    """
    try:
        datetime.strptime(args.judgment_date, '%Y-%m-%d')
    except ValueError:
        sys.exit(f'--judgment-date 형식 오류 (YYYY-MM-DD): {args.judgment_date}')
    try:
        with open(args.result, encoding='utf-8') as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f'판정 결과 JSON 로드 실패: {args.result} ({e})')
    other_official = {d: h for h, d in JUDGMENT_DATES.items()
                      if h != int(args.horizon)}
    if args.judgment_date in other_official:
        sys.exit(f'{args.judgment_date} 는 H={other_official[args.judgment_date]} '
                 f'의 공식 판정일 — --horizon {args.horizon} 로 기록 거부')
    try:
        validate_result_provenance(doc, args.horizon, args.judgment_date)
    except ValueError as e:
        sys.exit(str(e))
    # 동결 코호트 결속: 결과 JSON 의 코호트 SHA 가 실제 트랙 B 랭킹 산출물과
    # 일치해야 한다 — verdict 라벨만 갖춘 임의(트랙 A·재생성) 코호트 결과 차단.
    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        sys.exit(f'트랙 B 코호트 산출물 없음: {cohort_path} — '
                 f'게이트는 동결 코호트에 결속되어야 함 (--cohort)')
    cohort_sha = sha256_of(str(cohort_path))
    doc_sha = ((doc.get('inputs') or {}).get('h2_cohort') or {}).get('sha256')
    if doc_sha != cohort_sha:
        sys.exit(f'결과 JSON 코호트 SHA {str(doc_sha)[:16]}… ≠ {cohort_path} '
                 f'실제 SHA {cohort_sha[:16]}… — 다른 코호트의 판정 결과, 기록 거부')
    try:
        with gzip.open(cohort_path, 'rt') as f:
            cohort_spec = (json.load(f).get('header') or {}).get('spec', '')
    except (OSError, EOFError, json.JSONDecodeError) as e:
        sys.exit(f'코호트 산출물 읽기 실패: {cohort_path} ({e})')
    if '트랙 B' not in str(cohort_spec):
        sys.exit(f'코호트 산출물 헤더가 트랙 B 랭커 산출물이 아님: '
                 f'spec={cohort_spec!r}')
    result_sha = sha256_of(args.result)
    try:
        entry = build_gate_entry(doc, args.judgment_date, args.horizon,
                                 args.result, result_sha)
    except ValueError as e:
        sys.exit(str(e))

    expected = JUDGMENT_DATES.get(int(args.horizon))
    if expected and expected != args.judgment_date:
        print(f"경고: H={args.horizon} 의 명세 판정일은 {expected} — "
              f"{args.judgment_date} 는 기술 기록으로만 남고 stage2 에 산입 안 됨")
        jd = doc.get('judgment_utc')
        if isinstance(jd, str) and jd[:10] != args.judgment_date:
            print(f"경고: 결과 JSON 판정시각 {jd} 의 일자 ≠ --judgment-date "
                  f"{args.judgment_date}")

    gate_path = Path(args.gate)
    gate = load_gate(gate_path)
    dup = [e for e in gate['entries'] if e['judgment_date'] == args.judgment_date]
    if dup:
        if dup[0].get('result_sha256') == result_sha:
            print(f"이미 기록된 판정일 {args.judgment_date} (동일 결과 SHA) — "
                  f"멱등 no-op (파일 불변)")
            print(f"stage2_eligible = {gate.get('stage2_eligible', False)}")
            return
        sys.exit(f"판정일 {args.judgment_date} 에 다른 결과가 이미 기록됨 "
                 f"(기존 SHA {dup[0].get('result_sha256', '')[:16]}… ≠ "
                 f"신규 {result_sha[:16]}…) — 충돌, 기록 거부")
    gate['entries'].append(entry)
    gate['stage2_eligible'] = stage2_from_entries(gate['entries'])
    gate['updated_at_utc'] = datetime.now(tz=timezone.utc).isoformat()
    save_gate(gate, gate_path)

    ok = lambda b: '충족' if b else '미달'  # noqa: E731
    c = entry['criteria']
    print(f"기록: {args.judgment_date} (H={args.horizon}일) → {gate_path}")
    print(f"  n={entry['n']}  IC={entry['ic']}  p={entry['p']}  "
          f"MDE≈{entry['mde_ic']:.3f}" if entry['mde_ic'] is not None
          else f"  n={entry['n']} — 지표 없음 (평가 불능)")
    print(f"  (a)IC≥+{IC_PASS_MIN}: {ok(c['ic_ge_min'])}  "
          f"(b)단측 p<{P_ONE_SIDED}: {ok(c['p_lt_alpha'])}  "
          f"(c)스프레드≥+{SPREAD_PASS_MIN * 100:.0f}%p: {ok(c['spread_ge_min'])}  "
          f"(d)상위중앙>전체중앙: {ok(c['top_gt_all'])}  "
          f"판정불가: {'예' if entry['indeterminate'] else '아니오'}")
    print(f"  이번 판정: {'충족' if entry['passed'] else '미충족'}")
    print(f"stage2_eligible = {gate['stage2_eligible']} "
          f"(조건: T+{STAGE2_HORIZONS[0]} AND T+{STAGE2_HORIZONS[1]} 모두 충족; "
          f"충족 시에도 페이퍼 설계 진입일 뿐 실거래 권한 아님)")


def main(argv: list[str] | None = None) -> None:
    """CLI 엔트리 — rank / gate 서브커맨드."""
    ap = argparse.ArgumentParser(
        prog='h2_trackb',
        description='H2 트랙 B: 13주 형성 랭커(rank) / 게이트 상태(gate) — 사전등록 동결')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('rank', help='13주 형성 랭킹 → logs/h2_trackb_cohort.json.gz')
    r.add_argument('--snapshots-dir', default='logs/h2_snapshots',
                   help='스냅샷 디렉토리 (*.jsonl.gz)')
    r.add_argument('--fills-state', default='logs/h2_fills_state.json')
    r.add_argument('--cohort', default='logs/trader_cohort.json.gz')
    r.add_argument('--out', default='logs/h2_trackb_cohort.json.gz')
    r.set_defaults(func=cmd_rank)
    g = sub.add_parser('gate', help='판정 결과 게이트 기록 → logs/h2_trackb_gate.json')
    g.add_argument('--result', required=True,
                   help='h2_consistency verdict --out-json 산출 JSON 경로')
    g.add_argument('--judgment-date', required=True, help='판정일 YYYY-MM-DD')
    g.add_argument('--horizon', type=int, required=True, choices=(30, 60, 90),
                   help='H (일): 30/60/90')
    g.add_argument('--cohort', default='logs/h2_trackb_cohort.json.gz',
                   help='동결 트랙 B 코호트 산출물 (결과 SHA 결속 검증용)')
    g.add_argument('--gate', default='logs/h2_trackb_gate.json')
    g.set_defaults(func=cmd_gate)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main(sys.argv[1:])
