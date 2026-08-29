"""H1 (기존 트레이더 지속성 연구) 판정 평가기 — 사전등록 동결 코드 (2026-08-29 작성).

원전: docs/TRADER_PERSISTENCE_STUDY.md 의 "판정 기준 (사전 고정)" — 2026-08-25 잠금
코호트 5,790지갑. 첫 판정일 T+30 = 2026-09-24 이전에 이 파일이 존재해야 하며
(판정 시점 작성 = 사전등록 위반), 커밋·태그로 동결된다.

동결 시점 정직 명기 (Codex 합의 문구):
  "첫 코호트 수준 판정 이전에 동결. 전방 일별 파일 4일치(08-25~27, 08-29)는 작성
  시점에 이미 존재하나, 집계·IC·십분위·기준값 등 코호트 수준 전방 결과는 일절
  조회하지 않았다. 스키마 검증 목적으로 지갑 2개의 원시 행을 열람했다."

판정 기준 (H1 문서 원문, 사전 고정 — 세 기준 모두 T+30 AND T+60 통과 필요):
  (1) Spearman 순위 IC (T0 월ROI → 전방 ROI) ≥ +0.05, 순열검정 p < 0.05
  (2) 상위 − 하위 십분위 전방 수익 스프레드 ≥ +3%p
  (3) 상위 십분위 전방 수익 중앙값 > 코호트 중앙값
  α 거버넌스: H1 원문은 p<0.05. 2026-08-27 거버넌스 수정의 "H1/H2 각 단측 p<.025"는
  오케스트레이터 지시에 따라 H1에 소급 적용하지 않되 **둘 다 보고**한다
  (status = 원문 α=.05 기준, status_strict = α=.025 기준, decision_basis 명기).

해석 확정 지점 (결과 조회 전 고정 — Codex 2라운드 합의):
  - T0 순위 변수 = logs/trader_cohort.json.gz 의 t0_month_roi,
    [−95%, +500%] = [−0.95, +5.0] 클리핑 (h2_consistency CLIP_LO/HI 재사용).
    강건성: t0_month_pnl 순위 IC 병기 (문서 "PnL 순위를 강건성 비교").
  - 전방 결과변수 Y: H2 사전등록 §2.4 준용 — 단 데이터는 logs/trader_daily/*.csv.gz
    (컬럼 address,account,day_pnl,month_pnl,month_roi,month_vlm,alltime_pnl).
    Y = clip((alltime_pnl(T_H) − alltime_pnl(T0)) / account(T0), −0.95, +5.0),
    클리핑 **후** 기간 정규화 Y × (H / 실제스팬일) — h2_consistency.compute_y 재사용.
    T0·T_H 모두 같은 일별 리더보드 alltime_pnl 계열의 차분이므로 계열 내 일관
    (H2 §2.3의 alltime_pnl↔perpAllTime 괴리 경고는 계열 간 문제 — 여기 해당 없음).
    A(T0)는 고정분모이므로 Y는 엄밀한 ROI가 아니라 "T0 자본 고정 PnL 수익 프록시".
  - T0 행 = trader_daily/2026-08-25.csv.gz (코호트 잠금 스냅과 값 동일 실측 확인).
  - T_H 행 = 지갑별 **마지막 유효 행** (T0 < 날짜 ≤ 판정일, alltime_pnl 유한).
    스팬 = (행 날짜 − T0)일; [H−5, H+5] 밖이면 제외(결측). **명시적 H1 고유 규칙**:
    H2 는 판정시각 이전 24h 스냅만 인정하나, H1 일별 벌크 파일에서는 오케스트레이터
    지시대로 ≤5일 후방 폴백 + 기간 정규화를 채택한다 (탈락 직전 지갑을 결측으로
    버리면 생존편향이 커지는 방향 — 신선/스테일 분리 IC·LOCF 감도로 왜곡 감시).
  - 판정일 해상도 = 달력 날짜 (일별 파일에 행 단위 타임스탬프 없음).
    공식 판정일 = T0 + H (H=30 → 2026-09-24, H=60 → 10-24, H=90 → 11-23).
    H∈{30,60} 공식일 실행만 게이트 자격(gate_eligible). H=90 공식일 =
    기술통계 확인점(descriptive_checkpoint — 세 번째 기회 아님, 상태 null).
    그 외 --judgment/--horizon = exploratory (상태 null, 게이트 발행 금지).
  - T+60 판정의 Y = T0 기점 누적 60일 (H2 방식: T0 고정, T_H 이동 — --horizon 60).
    "T+30→T+60 롤링 2개월차" 해석은 기술 보고 감도로 병기 (기준선 = T0+30 이전
    마지막 유효 행, T0 기점 스팬 [25,30]일, 분모 = 그 행의 account).
  - 십분위(1차) = **T0 잠금 십분위**: 전 코호트 5,790을 클리핑 ROI 내림차순
    (동점 주소 오름차순)으로 10등분한 고정 멤버십. 결측 지갑이 십분위 경계를
    움직이지 못한다. 관측 Y 의 십분위 중앙값으로 (2)(3) 판정.
    h2 _analysis_block 의 평가가능-부분집합 재절단은 감도로만 병기.
  - 판정불가(기각 아님) = h2_consistency.missingness_verdict 동결 규칙 그대로:
    결측률 > 10% 또는 결측~점수 Spearman 순열 양측 p < 0.05. 평가가능 < 10 도
    판정불가. 보조 경고: 십분위×결측 옴니버스 순열검정 (단조 아닌 U자 탈락 감시).
  - 순열: n_perm=10,000, seed=20260827, 단측 p=(1+#{perm≥obs})/(n_perm+1)
    — h2_consistency.stratified_perm_p 재사용 (무층화 = 전 지갑 단일 층).
    층화(H2 §2.5 방식: log A(T0) 3분위 × log1p(회전율) 3분위)는 감도 병기.
  - **2026-08-28 일별 파일 영구 결손** (수집 실패, 복구 불가): Y 는 T0 행과 마지막
    유효 행의 2점 차분이므로 내부 결손일은 1차 산식에 영향 없음(inventory defect,
    outcome defect 아님). 판정일이 결손일이면 지갑별 폴백+정규화가 흡수. 결손일
    목록·복구 여부를 산출물에 항상 공개한다.
  - 동결 의존성 검증: 실행 시 lab/h2_consistency.py SHA256 이 사전등록 핀
    (docs/PREREGISTRATION_H2_2026-08-27.md §6.1)과 다르면 즉시 중단 (fail-closed).

실행 (cwd = 저장소 루트):
  python lab/h1_verdict.py --horizon 30 [--judgment ISO] [--out-json V30.json]
  python lab/h1_verdict.py --horizon 60 [--out-json V60.json]
  python lab/h1_verdict.py --combine V30.json V60.json [--out-json GATE.json]
    → H1 최종 게이트 (T+30 AND T+60, fail > indeterminate > pass 우선순위).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:                    # python lab/h1_verdict.py 직접 실행 지원
    sys.path.insert(0, _ROOT)

import lab.h2_consistency as h2  # noqa: E402
from lab.h2_consistency import (  # noqa: E402  (경로 셋업 후 import — 동결 유틸 재사용)
    CLIP_HI,
    CLIP_LO,
    IC_PASS_MIN,
    N_PERM,
    PERM_SEED,
    SPAN_TOL_D,
    SPREAD_PASS_MIN,
    _analysis_block,
    compute_y,
    mde_ic,
    missingness_verdict,
    ordered_indices,
    sha256_of,
    spearman_avg,
    stratified_perm_p,
    terciles,
)

logger = logging.getLogger(__name__)

# ── 사전 고정 파라미터 (변경 금지) ──────────────────────────────────────────
T0_DATE = date(2026, 8, 25)                  # 코호트 잠금일 = T0 (문서 원문)
EXPECTED_COHORT_N = 5790                     # 잠금 코호트 크기 (검증 실패 시 중단)
GATE_HORIZONS = (30, 60)                     # 게이트 자격 판정일 (T+90 은 기술통계)
OFFICIAL_HORIZONS = (30, 60, 90)             # 공식 판정일 = T0 + H
ALPHA_DOC = 0.05                             # H1 문서 원문 단측 α
ALPHA_GOV = 0.025                            # 거버넌스 강화 α (소급 미적용, 병기)
Z_ALPHA_05 = 1.644854                        # 단측 α=.05 (MDE 병기용)
MID_SPAN_MIN_D, MID_SPAN_MAX_D = 25, 30      # 롤링 감도: T+30 기준선 스팬 [25,30]일
ROLL_H = 30                                  # 롤링 감도 목표 스팬 (게이트 [25,35])
KNOWN_MISSING_DAYS = ('2026-08-28',)         # 영구 결손 확정 일별 파일 (수집 실패)
DAILY_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})\.csv\.gz$')
DECISION_BASIS = ('H1 문서 원문 단측 α=0.05 (거버넌스 α=0.025 는 소급 미적용·병기 — '
                  '오케스트레이터 지시, 2026-08-29)')
# 동결 의존성 핀 — docs/PREREGISTRATION_H2_2026-08-27.md §6.1 의 lab/h2_consistency.py
H2_SHA256_PINNED = 'e953af8fdd21286a3507e3f9855e5007271321c173891391060911141f198b64'
# 동결 입력 핀 (2026-08-29 shasum -a 256 실측 — 코호트는 H2 사전등록 §6.2 와 일치)
COHORT_SHA256_PINNED = '349a7ce19ed67f5a4e65365294c15cf5f3fdd327767123a2adcb98a9fee0d033'
T0_DAILY_SHA256_PINNED = '3bf00ab167205b02110f28d8befce45ff7fe6d0aa97a9ca0f1bb191cb01c57f2'


class FrozenSpecError(RuntimeError):
    """동결 명세 위반 (의존성 해시 불일치·코호트 계약 위반) — fail-closed 중단."""


def verify_frozen_dep() -> str:
    """lab/h2_consistency.py 실물 SHA256 을 사전등록 핀과 대조. 불일치면 중단."""
    actual = sha256_of(h2.__file__)
    if actual != H2_SHA256_PINNED:
        raise FrozenSpecError(
            f'동결 의존성 위반: lab/h2_consistency.py sha256={actual} ≠ '
            f'사전등록 핀 {H2_SHA256_PINNED} — 평가 중단 (fail-closed)')
    return actual


def clip_score(v: float) -> float:
    """T0 순위 변수 클리핑 [−0.95, +5.0] (문서: [−95%, +500%])."""
    return min(max(v, CLIP_LO), CLIP_HI)


def parse_judgment_date(s: str) -> date:
    """--judgment ISO 문자열 → UTC 달력 날짜 (naive 는 UTC 간주)."""
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.date()


# ── 입력 로드 ────────────────────────────────────────────────────────────────
def load_cohort(path: str) -> list[dict]:
    """잠금 코호트 로드 + 계약 검증 (fail-closed — 조용한 축소 금지).

    검증: locked_at == 2026-08-25, n 필드 == 지갑 수 == EXPECTED_COHORT_N,
    소문자 주소 유일, t0_month_roi·t0_account 유한. 위반 시 FrozenSpecError.
    반환 지갑 dict 에 address 소문자 정규화, score(클리핑 ROI) 추가.
    """
    with gzip.open(path, 'rt') as f:
        coh = json.load(f)
    if str(coh.get('locked_at')) != T0_DATE.isoformat():
        raise FrozenSpecError(f"코호트 locked_at={coh.get('locked_at')} ≠ {T0_DATE}")
    wallets = coh['wallets']
    if not (len(wallets) == int(coh.get('n', -1)) == EXPECTED_COHORT_N):
        raise FrozenSpecError(
            f"코호트 크기 위반: wallets={len(wallets)} n필드={coh.get('n')} "
            f"기대={EXPECTED_COHORT_N}")
    out: list[dict] = []
    seen: set[str] = set()
    for w in wallets:
        addr = str(w['address']).lower()
        roi = float(w['t0_month_roi'])
        acct = float(w['t0_account'])
        if addr in seen:
            raise FrozenSpecError(f'코호트 주소 중복: {addr}')
        if not (math.isfinite(roi) and math.isfinite(acct)):
            raise FrozenSpecError(f'코호트 비유한 값: {addr}')
        seen.add(addr)
        # 감도 전용 필드 정규화 (1차와 무관 — 무효값이 감도 계산을 죽이지 않게)
        pnl = w.get('t0_month_pnl')
        vlm = w.get('t0_month_vlm')
        pnl = float(pnl) if isinstance(pnl, (int, float)) and math.isfinite(pnl) else 0.0
        vlm = float(vlm) if isinstance(vlm, (int, float)) and math.isfinite(vlm) else 0.0
        out.append({**w, 'address': addr, 'score': clip_score(roi),
                    't0_month_pnl': pnl, 't0_month_vlm': vlm})
    return out


def _sanitize(obj):
    """JSON 표준 준수: 비유한 float(NaN/Inf) → None 재귀 변환 (RFC 8259)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _read_daily_rows(path: str, counts: Counter) -> dict[str, dict]:
    """일별 CSV(gz) 1개 → addr(소문자) → {'pnl','acct'} (파일 내 중복은 keep-last).

    alltime_pnl 비유한/파싱불가 행은 거부·집계 (마지막 '유효' 행 규칙 — 무효 행은
    이전 유효 관측으로 폴백). account 는 유한하지 않으면 None (1차 Y 는 T_H
    account 를 쓰지 않으므로 유지 — 흐름 진단만 불가 처리).
    """
    out: dict[str, dict] = {}
    with gzip.open(path, 'rt') as f:
        rd = csv.DictReader(f)
        need = {'address', 'account', 'alltime_pnl'}
        if rd.fieldnames is None or not need.issubset(rd.fieldnames):
            counts['bad_header_file'] += 1
            return {}
        for r in rd:
            try:
                addr = str(r['address']).lower()
                pnl = float(r['alltime_pnl'])
            except (KeyError, TypeError, ValueError):
                counts['bad_row'] += 1
                continue
            if not math.isfinite(pnl):
                counts['nonfinite_pnl'] += 1
                continue
            try:
                acct: float | None = float(r['account'])
                if not math.isfinite(acct):
                    acct = None
            except (TypeError, ValueError):
                acct = None
            if addr in out:
                counts['dup_row'] += 1        # 재시도 append 가정 — keep-last
            out[addr] = {'pnl': pnl, 'acct': acct}
    return out


def list_daily_files(daily_dir: str) -> list[tuple[date, str]]:
    """trader_daily 디렉토리의 YYYY-MM-DD.csv.gz 를 날짜 오름차순 (파일명 기준)."""
    out: list[tuple[date, str]] = []
    for p in sorted(Path(daily_dir).iterdir()):
        m = DAILY_RE.match(p.name)
        if m:
            out.append((date.fromisoformat(m.group(1)), str(p)))
    return out


def load_endpoints(daily_files: list[tuple[date, str]], judgment: date,
                   mid_date: date | None, file_stats: list[dict]
                   ) -> tuple[dict[str, dict], dict[str, dict]]:
    """T0 < 날짜 ≤ 판정일 파일들에서 지갑별 마지막 유효 행 선택 (룩어헤드 불가 —
    판정일 이후 파일은 이름 단계에서 걸러 **열지도 않는다**).

    반환: (ep, ep_mid). ep[addr] = {'date','pnl','acct'} — 뒤 날짜가 앞을 덮음.
    ep_mid 는 mid_date(롤링 감도 기준선, T0+30) 이하 마지막 유효 행 (mid_date=None
    이면 빈 dict). file_stats 에 파일별 행 품질 집계를 append.
    """
    ep: dict[str, dict] = {}
    ep_mid: dict[str, dict] = {}
    for d, path in daily_files:
        if d <= T0_DATE or d > judgment:
            continue
        cnt: Counter = Counter()
        rows = _read_daily_rows(path, cnt)
        file_stats.append({'date': d.isoformat(), 'path': path,
                           'sha256': sha256_of(path), 'valid_rows': len(rows),
                           'rejects': dict(cnt)})
        for addr, r in rows.items():
            rec = {'date': d, 'pnl': r['pnl'], 'acct': r['acct']}
            ep[addr] = rec
            if mid_date is not None and d <= mid_date:
                ep_mid[addr] = rec
    return ep, ep_mid


# ── 지갑별 전방 결과 ─────────────────────────────────────────────────────────
def evaluate_wallet(t0_row: dict | None, ep_row: dict | None,
                    judgment: date, horizon: int) -> dict:
    """지갑 1개 전방 결과 — Y(클립)·Y(unclipped)·LOCF·흐름 진단·결측 사유.

    1차 Y 유효성은 T0 행(alltime_pnl·account 유한, account>0)과 마지막 유효 행의
    alltime_pnl 에만 의존한다. T_H account 무효는 흐름 진단만 불가(Codex 합의 —
    보고용 진단이 1차 표본을 바꾸는 경로 차단).
    """
    rec: dict = {'y': None, 'y_unclipped': None, 'y_locf': None, 'reason': None,
                 'span_d': None, 'ep_date': None, 'a0': None, 'flow': None,
                 'clipped': False, 'stale': None}
    if t0_row is None:
        rec['reason'] = 'no_t0'
        return rec
    a0 = t0_row['acct']
    if a0 is None or not math.isfinite(a0) or a0 <= 0:
        rec['reason'] = 'bad_a0'
        return rec
    rec['a0'] = float(a0)
    if ep_row is None:
        rec['reason'] = 'no_th'
        return rec
    p0, ph = t0_row['pnl'], ep_row['pnl']
    raw = (ph - p0) / a0
    rec['y_locf'] = clip_score(raw)          # LOCF 감도: 스팬 게이트·정규화 없음
    span_d = (ep_row['date'] - T0_DATE).days
    rec['span_d'] = span_d
    rec['ep_date'] = ep_row['date'].isoformat()
    y = compute_y(p0, ph, a0, float(span_d), float(horizon), clip=True)
    if y is None:
        rec['reason'] = 'span_out'
        return rec
    rec['y'] = y
    rec['y_unclipped'] = compute_y(p0, ph, a0, float(span_d), float(horizon),
                                   clip=False)
    rec['clipped'] = bool(raw < CLIP_LO or raw > CLIP_HI)
    rec['stale'] = bool(ep_row['date'] < judgment)
    ah = ep_row['acct']
    if ah is not None and math.isfinite(ah):
        rec['flow'] = ((ah - a0) - (ph - p0)) / a0
    return rec


def rolling_y(mid_row: dict | None, end_row: dict | None) -> float | None:
    """롤링 2개월차 감도(D3 대안 해석)의 Y — 게이트 권한 없음.

    기준선 = T0+30 이하 마지막 유효 행 (T0 기점 스팬 [25,30]일 필수),
    분모 = 그 기준선 행의 account (유한·양수 필수 — 실패는 롤링 감도만 결측),
    Y = clip((P(끝) − P(기준선)) / A(기준선)) → 목표 30일 정규화, 게이트 [25,35].
    """
    if mid_row is None or end_row is None:
        return None
    a_mid = mid_row['acct']
    if a_mid is None or not math.isfinite(a_mid) or a_mid <= 0:
        return None
    if not (MID_SPAN_MIN_D <= (mid_row['date'] - T0_DATE).days <= MID_SPAN_MAX_D):
        return None
    span_r = (end_row['date'] - mid_row['date']).days
    return compute_y(mid_row['pnl'], end_row['pnl'], a_mid, float(span_r),
                     float(ROLL_H), clip=True)


# ── T0 잠금 십분위 ───────────────────────────────────────────────────────────
def locked_decile_labels(scores: np.ndarray, addrs: list[str]) -> np.ndarray:
    """전 코호트 T0 잠금 십분위 라벨(0=상위 … 9=하위) — 결측과 무관한 고정 멤버십.

    정렬 = 클리핑 ROI 내림차순, 동점 주소 오름차순 (ordered_indices 재사용).
    분할 = np.array_split 10등분 (h2 _analysis_block 과 동일 크기 규칙).
    """
    order = ordered_indices(scores, addrs)
    labels = np.empty(len(addrs), dtype=int)
    for d, grp in enumerate(np.array_split(np.asarray(order), 10)):
        labels[grp] = d
    return labels


def boundary_tie_counts(scores: np.ndarray, addrs: list[str]) -> dict:
    """D1/D2·D9/D10 경계의 동점 지갑 수 (주소 타이브레이크가 가른 경계 투명화)."""
    order = ordered_indices(scores, addrs)
    groups = np.array_split(np.asarray(order), 10)
    out = {}
    for name, g_hi, g_lo in (('d1_d2', 0, 1), ('d9_d10', 8, 9)):
        v_last = float(scores[groups[g_hi][-1]])
        v_first = float(scores[groups[g_lo][0]])
        tied = int((scores == v_last).sum()) if v_last == v_first else 0
        out[name] = {'boundary_score': v_last, 'tied_wallets': tied}
    return out


def locked_block(labels: np.ndarray, scores: np.ndarray, ys: list[float | None]
                 ) -> dict:
    """잠금 십분위 기준 통계 블록 — IC·십분위 중앙·D1−D10 스프레드·전체 중앙.

    ys[i] = None 은 결측 (십분위 멤버십은 불변, 중앙값은 관측치만).
    빈 십분위 중앙값은 NaN → 기준 비교에서 False (판정불가 규칙이 병행 감시).
    """
    obs = np.asarray([y is not None for y in ys], dtype=bool)
    yv = np.asarray([y if y is not None else math.nan for y in ys], dtype=float)
    med = []
    n_obs = []
    for d in range(10):
        m = (labels == d) & obs
        n_obs.append(int(m.sum()))
        med.append(float(np.median(yv[m])) if m.any() else math.nan)
    ic = spearman_avg(scores[obs], yv[obs]) if int(obs.sum()) >= 2 else math.nan
    return {'n': int(obs.sum()), 'ic': float(ic), 'decile_medians': med,
            'decile_n_obs': n_obs, 'spread': med[0] - med[9], 'top_median': med[0],
            'all_median': float(np.median(yv[obs])) if obs.any() else math.nan}


def omnibus_missingness_p(labels: np.ndarray, miss: np.ndarray,
                          n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """십분위×결측 옴니버스 순열검정 (보고용 경고 — 판정불가 규칙 아님).

    통계량 = Σ_d n_d (rate_d − rate)². 자유 순열 n_perm 회, seed 고정,
    p = (1 + #{perm ≥ 관측}) / (n_perm + 1). 단조가 아닌(U자) 탈락 패턴 감시 —
    동결 missingness_verdict 의 Spearman 은 단조 연관만 검출한다 (Codex 지적).
    """
    miss = np.asarray(miss, dtype=float)
    n = len(miss)
    if n == 0 or miss.min() == miss.max():
        return {'stat': 0.0, 'p': 1.0}
    overall = miss.mean()
    idx = [np.where(labels == d)[0] for d in range(10)]
    sizes = np.asarray([len(i) for i in idx], dtype=float)

    def stat_of(m: np.ndarray) -> float:
        rates = np.asarray([m[i].mean() if len(i) else 0.0 for i in idx])
        return float((sizes * (rates - overall) ** 2).sum())

    obs = stat_of(miss)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if stat_of(miss[rng.permutation(n)]) >= obs:
            cnt += 1
    return {'stat': obs, 'p': (1 + cnt) / (n_perm + 1)}


# ── 판정 (순수 함수 — 경계 테스트 대상) ─────────────────────────────────────
def judge(ic: float, p: float, spread: float, top_median: float,
          all_median: float) -> dict:
    """세 기준 판정 (양 α 병기). NaN 은 모든 비교에서 False → 미달 (fail-closed)."""
    c_ic = bool(ic >= IC_PASS_MIN)
    c_p05 = bool(p < ALPHA_DOC)
    c_p025 = bool(p < ALPHA_GOV)
    c_spread = bool(spread >= SPREAD_PASS_MIN)
    c_top = bool(top_median > all_median)
    return {'ic_ge_005': c_ic, 'p_lt_005': c_p05, 'p_lt_0025': c_p025,
            'spread_ge_3pp': c_spread, 'top_gt_all': c_top,
            'all_pass_doc_alpha': bool(c_ic and c_p05 and c_spread and c_top),
            'all_pass_strict_alpha': bool(c_ic and c_p025 and c_spread and c_top)}


def overall_status(indeterminate: bool, all_pass: bool) -> str:
    """3상태 판정: indeterminate(기각 아님) / pass / fail."""
    if indeterminate:
        return 'indeterminate'
    return 'pass' if all_pass else 'fail'


# ── 본 판정 실행 ────────────────────────────────────────────────────────────
def run_verdict(cohort_path: str, daily_dir: str, horizon: int,
                judgment_arg: str | None, today: date | None = None) -> dict:
    """H1 판정 1회 실행 (한 판정일·한 horizon) — 결과 dict 반환 (stdout 보고 병행).

    H1 최종 게이트는 T+30 AND T+60 두 공식 실행의 결합 (--combine) 이다.
    today 는 시계 주입 (테스트 전용 — CLI 는 항상 실제 UTC 오늘). 공식/기술통계
    실행은 today ≥ 판정일이어야 한다: 판정일 도래 전 '공식' 판정 계산 자체를
    차단한다 (조기 판정 = 사전등록 위반 경로 — Codex 3라운드 블로커).
    """
    h2_sha = verify_frozen_dep()
    self_sha = sha256_of(__file__)
    if today is None:
        today = datetime.now(tz=timezone.utc).date()
    judgment = parse_judgment_date(judgment_arg) if judgment_arg \
        else T0_DATE + timedelta(days=horizon)
    if judgment <= T0_DATE:
        raise FrozenSpecError(f'판정일 {judgment} ≤ T0 {T0_DATE}')
    official = (horizon in OFFICIAL_HORIZONS
                and judgment == T0_DATE + timedelta(days=horizon))
    if not official:
        analysis_status = 'exploratory'
    elif horizon in GATE_HORIZONS:
        analysis_status = 'official'
    else:
        analysis_status = 'descriptive_checkpoint'
    gate_eligible = analysis_status == 'official'
    if analysis_status != 'exploratory' and today < judgment:
        raise FrozenSpecError(
            f'조기 공식 실행 차단: 오늘 {today} < 판정일 {judgment} — '
            f'판정일 전 점검은 --judgment 로 탐색(exploratory) 실행만 가능')

    # 동결 입력 핀 검증 (fail-closed — 변조·오배선 차단, 파싱 전에 수행)
    cohort_sha = sha256_of(cohort_path)
    if cohort_sha != COHORT_SHA256_PINNED:
        raise FrozenSpecError(
            f'코호트 파일 sha256={cohort_sha} ≠ 핀 {COHORT_SHA256_PINNED}')
    t0_path = str(Path(daily_dir) / f'{T0_DATE.isoformat()}.csv.gz')
    t0_sha = sha256_of(t0_path)
    if t0_sha != T0_DAILY_SHA256_PINNED:
        raise FrozenSpecError(
            f'T0 일별 파일 sha256={t0_sha} ≠ 핀 {T0_DAILY_SHA256_PINNED}')
    wallets = load_cohort(cohort_path)

    # T0 행 (trader_daily 의 T0 파일 — 잠금 스냅과 값 동일 실측 확인)
    t0_cnt: Counter = Counter()
    t0_rows = _read_daily_rows(t0_path, t0_cnt)

    daily_files = list_daily_files(daily_dir)
    mid_date = T0_DATE + timedelta(days=ROLL_H) if horizon == 60 else None
    file_stats: list[dict] = []
    ep, ep_mid = load_endpoints(daily_files, judgment, mid_date, file_stats)

    present = {d.isoformat() for d, _ in daily_files}
    window = [(T0_DATE + timedelta(days=k)).isoformat()
              for k in range(1, (judgment - T0_DATE).days + 1)]
    missing_dates = [d for d in window if d not in present]
    recovered = [d for d in KNOWN_MISSING_DAYS if d in present]

    print(f"H1 판정 평가 — T0={T0_DATE} 판정일={judgment} H={horizon}일 "
          f"[{analysis_status}]")
    print(f"코호트 {len(wallets)}지갑 (vault {sum(1 for w in wallets if w.get('is_vault'))}) "
          f"| T0 파일 유효 {len(t0_rows)}행 (거부 {dict(t0_cnt)})")
    print(f"윈도 내 일별 파일 {len(file_stats)}개, 결손일 {len(missing_dates)}개: "
          f"{missing_dates}")
    print(f"알려진 영구 결손 {list(KNOWN_MISSING_DAYS)} — 2점 차분 1차 산식에 영향 "
          f"없음(inventory defect)" + (f" | 주의: 복구 감지 {recovered}" if recovered else ""))

    # 지갑별 전방 결과 (코호트 순서 고정)
    addrs = [w['address'] for w in wallets]
    scores = np.asarray([w['score'] for w in wallets], dtype=float)
    recs = {a: evaluate_wallet(t0_rows.get(a), ep.get(a), judgment, horizon)
            for a in addrs}
    labels = locked_decile_labels(scores, addrs)
    ties = boundary_tie_counts(scores, addrs)

    # ── 결측·판정불가 (동결 규칙 = h2 missingness_verdict 그대로) ──
    miss = np.asarray([1.0 if recs[a]['y'] is None else 0.0 for a in addrs])
    reasons = Counter(recs[a]['reason'] for a in addrs if recs[a]['y'] is None)
    mv = missingness_verdict(scores, miss)
    omni = omnibus_missingness_p(labels, miss)
    dec_missing = [int(miss[labels == d].sum()) for d in range(10)]
    dec_rate = [float(miss[labels == d].mean()) for d in range(10)]
    n_missing = int(miss.sum())
    print(f"\n결측 {n_missing}/{len(addrs)} ({mv['missing_rate'] * 100:.1f}%) "
          f"사유 {dict(reasons)}")
    print("잠금 십분위별 결측(상위→하위): " + ' '.join(str(x) for x in dec_missing))
    print(f"결측~점수 Spearman rho={mv['rho']:+.3f} 순열 양측 p={mv['p']:.4f} | "
          f"옴니버스(비단조 경고용) stat={omni['stat']:.3f} p={omni['p']:.4f}")

    ev_mask = miss == 0.0
    n_ev = int(ev_mask.sum())
    indeterminate = bool(mv['indeterminate'] or n_ev < 10)
    if indeterminate:
        why = []
        if mv['missing_rate'] > h2.MISS_RATE_MAX:
            why.append(f"결측률 {mv['missing_rate'] * 100:.1f}% > 10%")
        if mv['p'] < h2.MISS_CORR_P:
            why.append(f"결측~점수 유의 상관 (p={mv['p']:.4f} < 0.05)")
        if n_ev < 10:
            why.append(f"평가 가능 {n_ev} < 10")
        print(f"→ 판정불가 (기각 아님): {'; '.join(why)}")

    result: dict = {
        'kind': 'h1_verdict',
        'spec': 'H1 판정 평가기 v1 (2026-08-29 동결) — docs/TRADER_PERSISTENCE_STUDY.md',
        'generated_at_utc': datetime.now(tz=timezone.utc).isoformat(),
        't0_date': T0_DATE.isoformat(),
        'judgment_date': judgment.isoformat(),
        'evaluated_on': today.isoformat(),
        'horizon_days': horizon,
        'analysis_status': analysis_status,
        'gate_eligible': gate_eligible,
        'calendar_day_resolution': True,
        'decision_basis': DECISION_BASIS,
        'params': {'clip': [CLIP_LO, CLIP_HI], 'span_gate_days':
                   [horizon - SPAN_TOL_D, horizon + SPAN_TOL_D],
                   'n_perm': N_PERM, 'seed': PERM_SEED,
                   'perm_rule': 'p=(1+#{perm>=obs})/(n_perm+1), 단측',
                   'mc_se_at_p05': round(math.sqrt(0.05 * 0.95 / N_PERM), 5),
                   'ic_min': IC_PASS_MIN, 'spread_min': SPREAD_PASS_MIN,
                   'alpha_doc': ALPHA_DOC, 'alpha_governance': ALPHA_GOV,
                   'endpoint_rule': '지갑별 마지막 유효 행 ≤ 판정일 (H1 고유 — '
                                    'H2 24h 규칙과 다름을 명시), 스팬 게이트+정규화'},
        'code': {'h1_sha256': self_sha, 'h2_sha256': h2_sha},
        'inputs': {'cohort': {'path': cohort_path, 'sha256': cohort_sha},
                   't0_daily': {'path': t0_path, 'sha256': t0_sha,
                                'valid_rows': len(t0_rows), 'rejects': dict(t0_cnt)},
                   'daily_files': file_stats,
                   'missing_dates_in_window': missing_dates,
                   'known_missing_days': list(KNOWN_MISSING_DAYS),
                   'known_missing_recovered': recovered},
        'n_cohort': len(wallets),
        'boundary_ties': ties,
        'missingness': {**mv, 'n_missing': n_missing, 'reasons': dict(reasons),
                        'decile_missing': dec_missing, 'decile_rate': dec_rate,
                        'omnibus': omni},
        'indeterminate': indeterminate,
        'note': 'H1 최종 게이트 = T+30 AND T+60 모두 통과 (--combine 모드로 결합)',
    }

    if n_ev < 10:
        print(f"\n평가 가능 {n_ev}지갑 (<10) — 검정 불가")
        result.update(primary=None, criteria=None, status='indeterminate' if
                      gate_eligible else None, status_strict='indeterminate' if
                      gate_eligible else None, passed=None, passed_strict=None,
                      insufficient_evaluable=n_ev)
        return result

    # ── 1차 검정: 무층화 순열 IC + 잠금 십분위 기준 ──
    ys_all = [recs[a]['y'] for a in addrs]
    prim = locked_block(labels, scores, ys_all)
    sc_ev = scores[ev_mask]
    ys_ev = np.asarray([y for y in ys_all if y is not None], dtype=float)
    p_unstrat, ic_obs, perm_ge = stratified_perm_p(
        sc_ev, ys_ev, np.zeros(n_ev, dtype=int))
    if abs(prim['ic'] - ic_obs) > 1e-12:      # 단일 진실원 강제 (같은 배열·같은 동결 함수)
        raise FrozenSpecError(
            f"IC 불일치: locked_block {prim['ic']} ≠ 순열 관측 {ic_obs}")
    prim.update(ic=float(ic_obs), p_unstratified=float(p_unstrat),
                perm_ge=int(perm_ge))
    print(f"\n── 1차 검정 (n={n_ev}, 무층화 순열 {N_PERM}회, seed={PERM_SEED}) ──")
    print(f"IC = {ic_obs:+.4f}  단측 p = {p_unstrat:.4f} (#{{perm≥obs}}={perm_ge}, "
          f"MC SE@p=.05 ≈ {result['params']['mc_se_at_p05']})")
    print("잠금 십분위 Y 중앙(상위→하위, %): "
          + ' '.join('---' if math.isnan(m) else f"{m * 100:+.2f}"
                     for m in prim['decile_medians']))
    print(f"D1−D10 스프레드 {prim['spread'] * 100:+.2f}%p | D1 중앙 "
          f"{prim['top_median'] * 100:+.2f}% vs 관측 전체 중앙 "
          f"{prim['all_median'] * 100:+.2f}%")

    crit = judge(prim['ic'], p_unstrat, prim['spread'], prim['top_median'],
                 prim['all_median'])
    ok = lambda b: '충족' if b else '미달'  # noqa: E731
    print(f"\n기준: (1)IC≥+0.05: {ok(crit['ic_ge_005'])}  순열 p<0.05(원문): "
          f"{ok(crit['p_lt_005'])} / p<0.025(거버넌스 병기): {ok(crit['p_lt_0025'])}  "
          f"(2)스프레드≥+3%p: {ok(crit['spread_ge_3pp'])}  "
          f"(3)D1중앙>전체중앙: {ok(crit['top_gt_all'])}")
    status = overall_status(indeterminate, crit['all_pass_doc_alpha'])
    status_strict = overall_status(indeterminate, crit['all_pass_strict_alpha'])
    if gate_eligible:
        print(f"  이 판정일 상태: {status} (원문 α) / {status_strict} (강화 α) — "
              f"최종 게이트는 T+30 AND T+60")
    else:
        print(f"  [{analysis_status}] 게이트 자격 없음 — 상태 미발행 "
              f"(기준값은 위 기술 보고)")

    mde05 = mde_ic(n_ev, z_alpha=Z_ALPHA_05)
    mde025 = mde_ic(n_ev)
    print(f"MDE(power 80%): α=.05 IC≈{mde05:.3f} / α=.025 IC≈{mde025:.3f} (n={n_ev})"
          if mde05 else "MDE: 표본 부족")

    result.update(
        primary=prim, criteria=crit,
        status=status if gate_eligible else None,
        status_strict=status_strict if gate_eligible else None,
        passed=(None if (not gate_eligible or indeterminate)
                else crit['all_pass_doc_alpha']),
        passed_strict=(None if (not gate_eligible or indeterminate)
                       else crit['all_pass_strict_alpha']),
        mde={'alpha05': mde05, 'alpha025': mde025})

    # ── 감도·진단 (통과 권한 없음 — 기술 보고만) ──
    # 실패해도 1차 판정을 죽이지 않는다 (보고 전용이 1차 산출을 막는 경로 차단 —
    # Codex 3라운드 합의). 부분 결과는 유지하고 sensitivity_error 에 사유 기록.
    sens: dict = {}
    result['sensitivity'] = sens
    try:
        print("\n── 감도분석 (기술 보고만 — 게이트 권한 없음) ──")
        addrs_ev = [a for a, m in zip(addrs, ev_mask) if m]

        # (s1) 층화 순열 + 평가가능 재절단 십분위 (h2 방식 그대로 — 감도로 강등)
        a0s = np.asarray([recs[a]['a0'] for a in addrs_ev], dtype=float)
        coh_map = {w['address']: w for w in wallets}
        turn = np.asarray([
            (coh_map[a]['t0_month_vlm'] / coh_map[a]['t0_account'])
            if float(coh_map[a].get('t0_account') or 0) > 0 else 0.0
            for a in addrs_ev], dtype=float)
        strata = 3 * terciles(np.log(a0s)) + terciles(np.log1p(turn))
        recut = _analysis_block('층화+재절단(감도)', sc_ev, ys_ev, strata, addrs_ev)
        sens['stratified_recut'] = recut
        print(f"층화 순열(H2 §2.5 방식): IC = {recut['ic']:+.4f}  단측 p = "
              f"{recut['p']:.4f} | 재절단 스프레드 {recut['spread'] * 100:+.2f}%p")

        # (s2) unclipped Y · PnL 순위 (문서 "PnL 순위를 강건성 비교")
        yu = np.asarray([recs[a]['y_unclipped'] for a in addrs_ev], dtype=float)
        sens['unclipped_ic'] = float(spearman_avg(sc_ev, yu))
        pnl_scores = np.asarray([float(coh_map[a]['t0_month_pnl'])
                                 for a in addrs_ev], dtype=float)
        sens['pnl_rank_ic'] = float(spearman_avg(pnl_scores, ys_ev))
        print(f"unclipped Y IC = {sens['unclipped_ic']:+.4f} | "
              f"T0 월PnL 순위 IC = {sens['pnl_rank_ic']:+.4f}")

        # (s3) 신선(판정일 당일) vs 스테일(폴백) — 폴백 규칙 왜곡 감시 (Codex 합의)
        for name, want_stale in (('fresh_same_day', False), ('stale_fallback', True)):
            ys_sub = [recs[a]['y'] if (recs[a]['y'] is not None
                                       and recs[a]['stale'] == want_stale) else None
                      for a in addrs]
            blk = locked_block(labels, scores, ys_sub)
            sens[name] = blk
            print(f"{name}: n={blk['n']} IC={blk['ic']:+.4f} "
                  f"스프레드 {blk['spread'] * 100:+.2f}%p "
                  f"D1 {blk['top_median'] * 100:+.2f}% "
                  f"전체 {blk['all_median'] * 100:+.2f}%")

        # (s4) LOCF (스팬 게이트·정규화 없음 — 소멸 지갑 생존편향 감시)
        ys_locf = [recs[a]['y_locf'] for a in addrs]
        blk = locked_block(labels, scores, ys_locf)
        sens['locf'] = blk
        print(f"LOCF(게이트·정규화 없음): n={blk['n']} IC={blk['ic']:+.4f} "
              f"스프레드 {blk['spread'] * 100:+.2f}%p")
        # 정규화 유발 방향성 격리: (Y_1차 − Y_LOCF) ~ 점수 (Codex 합의)
        diff = [recs[a]['y'] - recs[a]['y_locf'] if recs[a]['y'] is not None
                else None for a in addrs]
        dobs = np.asarray([d is not None for d in diff], dtype=bool)
        dv = np.asarray([d if d is not None else math.nan for d in diff],
                        dtype=float)
        sens['y_minus_locf'] = {
            'decile_medians': [float(np.median(dv[(labels == d) & dobs]))
                               if ((labels == d) & dobs).any() else math.nan
                               for d in range(10)],
            'spearman_vs_score': float(spearman_avg(scores[dobs], dv[dobs]))
            if int(dobs.sum()) >= 2 else math.nan}
        print(f"Y−Y_LOCF ~ 점수 Spearman = "
              f"{sens['y_minus_locf']['spearman_vs_score']:+.4f}")

        # (s5) 롤링 2개월차 [T+30, T+60] (H=60 전용 — D3 대안 해석의 감도 고정)
        if horizon == 60 and mid_date is not None:
            ys_roll = [rolling_y(ep_mid.get(a), ep.get(a)) for a in addrs]
            blk = locked_block(labels, scores, ys_roll)
            sens['rolling_30_60'] = blk
            print(f"롤링 2개월차(T+30→T+60, 감도): n={blk['n']} IC={blk['ic']:+.4f} "
                  f"스프레드 {blk['spread'] * 100:+.2f}%p")

        # ── 진단 (보고 전용) ──
        flow_med = []
        for d in range(10):
            fv = [recs[a]['flow'] for a, l0 in zip(addrs, labels)
                  if l0 == d and recs[a]['flow'] is not None
                  and recs[a]['y'] is not None]
            flow_med.append(float(np.median(fv)) if fv else math.nan)
        n_flow_na = sum(1 for a in addrs_ev if recs[a]['flow'] is None)
        clip_dec = [sum(1 for a, l0 in zip(addrs, labels)
                        if l0 == d and recs[a]['clipped']) for d in range(10)]
        stale_days = Counter((judgment - date.fromisoformat(recs[a]['ep_date'])).days
                             for a in addrs_ev)
        stale_by_dec = [sum(1 for a, l0 in zip(addrs, labels)
                            if l0 == d and recs[a]['y'] is not None
                            and recs[a]['stale']) for d in range(10)]
        result['diagnostics'] = {
            'flow_median_by_decile': flow_med,
            'flow_unavailable_evaluable': n_flow_na,
            'clip_count_by_decile': clip_dec,
            'endpoint_age_days_hist': {str(k): v
                                       for k, v in sorted(stale_days.items())},
            'stale_count_by_decile': stale_by_dec}
        print(f"흐름 진단 중앙(십분위, %A0): "
              + ' '.join('---' if math.isnan(m) else f"{m * 100:+.1f}"
                         for m in flow_med)
              + f" (진단 불가 {n_flow_na})")
        print(f"엔드포인트 나이 분포(일:지갑수): {dict(sorted(stale_days.items()))}")
    except Exception as e:  # noqa: BLE001 — 보고 전용 실패 격리 (1차 판정 보존)
        logger.warning('감도·진단 계산 실패 (1차 판정은 유효): %r', e)
        result['sensitivity_error'] = repr(e)
        print(f"[경고] 감도·진단 계산 실패 — 1차 판정은 유효: {e!r}")
    return result


# ── 최종 게이트 결합 (T+30 AND T+60) ────────────────────────────────────────
def _validate_gate_record(h: int, r: dict, self_sha: str,
                          problems: list[str]) -> None:
    """공식 판정 JSON 1건의 무결성·자기일관성 검증 (실패는 problems 에 축적)."""
    valid_status = ('pass', 'fail', 'indeterminate')
    if r.get('kind') != 'h1_verdict':
        problems.append(f'H{h}: kind={r.get("kind")}')
    if r.get('analysis_status') != 'official' or r.get('gate_eligible') is not True:
        problems.append(f'H{h}: 공식 판정 아님 ({r.get("analysis_status")})')
    if r.get('t0_date') != T0_DATE.isoformat():
        problems.append(f'H{h}: t0_date={r.get("t0_date")}')
    want = (T0_DATE + timedelta(days=h)).isoformat()
    if r.get('judgment_date') != want:
        problems.append(f'H{h}: 판정일 {r.get("judgment_date")} ≠ {want}')
    ev = r.get('evaluated_on')
    if not isinstance(ev, str) or ev < want:      # ISO 날짜는 사전순 = 시간순
        problems.append(f'H{h}: 판정일 전 평가 (evaluated_on={ev})')
    code = r.get('code') or {}
    if code.get('h1_sha256') != self_sha:
        problems.append(f'H{h}: h1 코드 해시 ≠ 현재 동결 파일')
    if code.get('h2_sha256') != H2_SHA256_PINNED:
        problems.append(f'H{h}: h2 코드 해시 ≠ 사전등록 핀')
    inp = r.get('inputs') or {}
    if (inp.get('cohort') or {}).get('sha256') != COHORT_SHA256_PINNED:
        problems.append(f'H{h}: 코호트 sha ≠ 핀')
    if (inp.get('t0_daily') or {}).get('sha256') != T0_DAILY_SHA256_PINNED:
        problems.append(f'H{h}: T0 파일 sha ≠ 핀')
    params = r.get('params') or {}
    if params.get('seed') != PERM_SEED or params.get('n_perm') != N_PERM:
        problems.append(f'H{h}: seed/n_perm 불일치')
    if (params.get('alpha_doc') != ALPHA_DOC
            or params.get('alpha_governance') != ALPHA_GOV):
        problems.append(f'H{h}: α 불일치')
    st, st_s = r.get('status'), r.get('status_strict')
    if st not in valid_status or st_s not in valid_status:
        problems.append(f'H{h}: 상태 무효 ({st}/{st_s})')
        return
    # 상태 ↔ 기준 ↔ passed 자기일관성 재계산 (기록만 믿지 않는다)
    ind = bool(r.get('indeterminate'))
    crit = r.get('criteria')
    if crit is None:
        if st != 'indeterminate' or st_s != 'indeterminate':
            problems.append(f'H{h}: criteria 없음인데 status={st}/{st_s}')
        return
    want_doc = overall_status(ind, bool(crit.get('all_pass_doc_alpha')))
    want_strict = overall_status(ind, bool(crit.get('all_pass_strict_alpha')))
    if st != want_doc or st_s != want_strict:
        problems.append(f'H{h}: 상태·기준 불일치 ({st}≠{want_doc} 또는 '
                        f'{st_s}≠{want_strict})')
    want_passed = None if ind else bool(crit.get('all_pass_doc_alpha'))
    if r.get('passed') != want_passed:
        problems.append(f'H{h}: passed={r.get("passed")} ≠ 재계산 {want_passed}')
    want_ps = None if ind else bool(crit.get('all_pass_strict_alpha'))
    if r.get('passed_strict') != want_ps:
        problems.append(f'H{h}: passed_strict={r.get("passed_strict")} ≠ '
                        f'재계산 {want_ps}')


def combine_gate(path30: str, path60: str) -> dict:
    """두 공식 판정 JSON 을 결합해 H1 최종 게이트 산출 (동결 결합 규칙).

    우선순위 (Codex 합의): 어느 한쪽 fail → fail (AND 게이트에서 확정 실패가
    판정불가를 지배) → 어느 한쪽 indeterminate/무효 → indeterminate → 둘 다 pass.
    검증 (하나라도 실패 시 overall='invalid'): kind·공식 상태·게이트 자격·
    horizon {30,60} 각 1개·판정일/T0 정확 일치·evaluated_on ≥ 판정일·
    코호트/T0 파일 sha == 핀·h1 코드 해시 == 현재 파일·h2 == 사전등록 핀·
    seed/n_perm/α == 동결 상수·status ↔ criteria ↔ passed 재계산 일치.
    입력 순서는 무관 (horizon 으로 정준화).
    """
    verify_frozen_dep()
    self_sha = sha256_of(__file__)
    with open(path30, encoding='utf-8') as f:
        ra = json.load(f)
    with open(path60, encoding='utf-8') as f:
        rb = json.load(f)
    problems: list[str] = []
    by_h: dict[int, dict] = {}
    for src, r in (('입력1', ra), ('입력2', rb)):
        try:
            h = r.get('horizon_days')
        except AttributeError:
            problems.append(f'{src}: 레코드가 객체 아님')
            continue
        if not isinstance(h, int) or isinstance(h, bool):   # "30"·30.9·true 거부
            problems.append(f'{src}: horizon_days 무효 ({h!r})')
            continue
        if h in by_h:
            problems.append(f'horizon {h} 중복')
            continue
        by_h[h] = r
    if set(by_h) != {30, 60}:
        problems.append(f'horizon 집합 {sorted(by_h)} ≠ [30, 60]')
    for h in sorted(by_h):
        try:
            _validate_gate_record(h, by_h[h], self_sha, problems)
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            problems.append(f'H{h}: 검증 예외 {e!r}')

    def _combined(field: str) -> str:
        sts = [by_h[h].get(field) for h in (30, 60) if h in by_h]
        if any(s == 'fail' for s in sts):
            return 'fail'
        if len(sts) < 2 or any(s != 'pass' for s in sts):
            return 'indeterminate'
        return 'pass'

    overall = 'invalid' if problems else _combined('status')
    overall_strict = 'invalid' if problems else _combined('status_strict')
    g30, g60 = by_h.get(30) or {}, by_h.get(60) or {}
    out = {'kind': 'h1_gate', 'generated_at_utc':
           datetime.now(tz=timezone.utc).isoformat(),
           'decision_basis': DECISION_BASIS,
           'rule': 'T+30 AND T+60 모두 pass — fail > indeterminate > pass',
           'inputs': {'t30': {'path': path30, 'sha256': sha256_of(path30)},
                      't60': {'path': path60, 'sha256': sha256_of(path60)}},
           'statuses': {'t30': g30.get('status'), 't60': g60.get('status'),
                        't30_strict': g30.get('status_strict'),
                        't60_strict': g60.get('status_strict')},
           'problems': problems,
           'overall': overall, 'overall_strict': overall_strict}
    print(f"H1 최종 게이트: {overall} (원문 α=.05) / {overall_strict} (강화 α=.025)")
    if problems:
        print("검증 실패: " + '; '.join(problems))
    if overall == 'fail':
        print("→ 문서 원문: '상위 = 로또' 확정, H1 랭킹 규칙 폐기 (재탕 금지 — "
              "거버넌스 2026-08-27: H1 규칙 한정, 별도 사전등록 선택변수엔 비소급)")
    elif overall == 'pass':
        print("→ 2단계(카피-팔로우 페이퍼 설계) 진행 조건 충족 — 실거래 권한 아님")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """CLI 엔트리 — 판정 1회(기본) 또는 --combine 최종 게이트 결합."""
    ap = argparse.ArgumentParser(
        prog='h1_verdict',
        description='H1 트레이더 지속성 판정 평가기 — 사전등록 동결 (2026-08-29)')
    ap.add_argument('--horizon', type=int, default=30,
                    help='H (일). 공식: 30/60(게이트)·90(기술통계). 기본 30')
    ap.add_argument('--judgment', default=None,
                    help='판정일 ISO-8601 (기본: T0 + H일 = 공식 판정일)')
    ap.add_argument('--cohort', default='logs/trader_cohort.json.gz')
    ap.add_argument('--daily-dir', default='logs/trader_daily')
    ap.add_argument('--combine', nargs=2, metavar=('T30_JSON', 'T60_JSON'),
                    default=None, help='두 공식 판정 JSON → 최종 게이트')
    ap.add_argument('--out-json', default=None, help='결과 JSON 저장 경로')
    args = ap.parse_args(argv)
    if args.combine:
        result = combine_gate(args.combine[0], args.combine[1])
    else:
        result = run_verdict(args.cohort, args.daily_dir, args.horizon,
                             args.judgment)
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            # RFC 8259 준수: 비유한 float 는 null 로 (NaN 리터럴 금지)
            json.dump(_sanitize(result), f, ensure_ascii=False, indent=2,
                      allow_nan=False)
        print(f"\n결과 JSON 저장: {args.out_json}")


if __name__ == '__main__':
    main(sys.argv[1:])
