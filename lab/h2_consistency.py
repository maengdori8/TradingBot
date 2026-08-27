"""H2 트랙 A(빠른 H2) — 형성 스크린(screen)과 판정 평가기(verdict). 사전등록 동결 코드.

명세: H2 구현 명세 v1 (Codex 2라운드 합의 확정본, 2026-08-27) →
docs/PREREGISTRATION_H2_2026-08-27.md. 모든 수치·식은 명세 그대로 (변경 금지).
지위: 보조검정 — 결과와 무관하게 자본 권한 영구 없음. 통과 기준은 보고용.

사전 고정 — screen (형성 랭킹):
- 코호트 logs/trader_cohort.json.gz 5,790지갑 그대로 (신규 스크린 금지, 파일럿 스킵 없음).
- 위상정렬 주간 그리드: perpAllTime pnl 곡선의 주간 샘플 점(인접 간격 168h±6h)의
  위상(ts mod 168h)을 1h 빈 최빈 → 최빈 빈 중심 ±3h 원형 중앙값으로 φ 추정,
  그리드 t_k ≡ φ + 36h (mod 168h) — 오프셋 근거는 GRID_OFFSET_MS 주석.
- 스냅 = 그리드시각 '이전'(후방만) 최근접 곡선점, 후방 3일(72h) 이내만. 미래 점 사용 금지.
- 트레일링 26주: 곡선 수집시각(마지막 곡선점) 이전 마지막 유효(후방 스냅 존재)
  그리드점에서 역산한 27개 그리드점.
- 주간 r_w = (pnl(g_{w+1}) − pnl(g_w)) / A_fix. A_fix = 윈도우 시작 그리드점 acct 스냅
  (고정분모, ≥ $10,000 필수 — 미달·결측은 제외, 바닥값 대체 금지).
- 주 유효: 양끝 스냅 존재 AND 실제 스팬 5–9일 AND 흐름 |Δacct−Δpnl|/acct(주 시작) ≤ 20%.
- 적격: 흐름-유효 주 ≥ 20/26 AND 형성 총 perp PnL > 0 (윈도우 끝−시작 스냅 차분).
- 1차 지표 ES20: k = ceil(0.2 × n_유효주), 최저 k개 r_w 산술평균. 내림차순 랭킹.
- 감도 변형(1차 랭킹과 구분, 별도 필드): 흐름≤50%(유효주≥20), 유효주≥16(흐름≤20%).
- 동결 산출물 logs/h2_cohort.json.gz — 헤더에 입력 SHA256·생성시각·파라미터 전부.

사전 고정 — verdict (판정 평가):
- Y = clip((P_perp(T_H)−P_perp(T0)) / A(T0), −0.95, +5.0).
  결측 규칙: ① 판정일 직접 재시도(수집기 측) → ② 판정시각 이전 24h 이내 스냅만 인정
  (평가기: 판정시각 − captured_at(T_H) ∈ [0, 24h], 판정시각 기본값 = T_H 파일
  captured_at 최댓값, --judgment로 명시 가능) → ③ 실제 스팬 ≠ H이면 기간 정규화
  Y × (H / 실제일수) — 명세 서술 순서 그대로 '클리핑 후 정규화';
  스팬 [H−5, H+5]일 밖(H=30이면 [25,35])이면 제외.
- Spearman IC(ES20 → Y), 동점 average rank.
- 층화 순열: log(A(T0)) 3분위 × log(1+T0회전율) 3분위 = 9층, 층 내 Y 라벨 순열
  10,000회, seed=20260827, p = (1 + #{perm IC ≥ 관측 IC}) / 10,001. 단측.
- 통과 기준(보고용): IC ≥ +0.05 AND 단측 p < 0.025 AND 상위−하위 십분위 Y 중앙
  스프레드 ≥ +3%p AND 상위 십분위 Y 중앙 > 전체 중앙.
- 판정불가(기각 아님): 결측률 > 10% 또는 결측 지표가 ES20과 유의 상관
  (Spearman, 순열 양측 p < 0.05). 제외 지갑수는 ES20 십분위별 공개.
- 감도(통과 권한 없음, 기술 보고만): 무층화, unclipped, 고회전율 서브그룹
  (회전율 상위 3분위 — "고회전율" 명칭만), 흐름≤50%·유효주≥16 랭킹 IC.

스냅샷 파일 계약 (carrybot/live/portfolio_snapshot.py 출력, T0·판정일 공통):
  jsonl.gz, 지갑당 1줄 JSON:
    {"address": str(0x…), "label": "t0"|"daily"|"verdict",
     "perp_alltime_pnl": float, "account_value": float,
     "captured_at_utc": ISO-8601 UTC str}
  평가기는 지정 라벨 행만 사용한다 (--t0-label 기본 "t0", --th-label 기본
  "verdict") — 같은 날짜 파일에 daily 재시도가 t0 뒤에 append 되어도 T0
  기준선이 daily 로 덮이지 않는다 (명세 §2.3 라벨 구분 조회).
  같은 주소·같은 라벨이 여러 줄이면 마지막 줄 우선(재시도 갱신 가정).
  label 필드 없는 행은 계약 위반으로 거부(카운트).

persistence_v3 재사용 검토: v3의 grid(2024-01-01 앵커, 위상 미정렬)와 snap(양방향
최근접)은 감사판 델타(위상정렬·후방만)로 의미 자체가 바뀌어 import 불가 —
구조(스냅 후 실제점 차분, 보간 제로)만 준용하고 함수는 본 파일에 동결한다.

실행 (cwd = 저장소 루트):
  python lab/h2_consistency.py screen [--portfolio P] [--cohort C] [--out O]
  python lab/h2_consistency.py verdict --t0 T0.jsonl.gz --th TH.jsonl.gz --horizon 30
      [--h2-cohort logs/h2_cohort.json.gz] [--judgment ISO8601]
      [--t0-label t0] [--th-label verdict] [--out-json VERDICT.json]
  --out-json 산출 JSON 은 트랙 B 게이트(lab/h2_trackb.py gate)의 입력이다.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# ── 사전 고정 파라미터 (명세 §트랙 A — 변경 금지) ──────────────────────────
WEEK_MS = 168 * 3600 * 1000          # 주간 그리드 간격
DAY_MS = 86400 * 1000
SNAP_BACK_MS = 3 * DAY_MS            # 후방 스냅 허용 3일(72h)
GRID_OFFSET_MS = 36 * 3600 * 1000    # 그리드 = φ + 36h (후방 72h 창의 중앙)
# 근거(2026-08-27 실측, 동결 전 고정): 플랫폼 주간 샘플 위상은 φ 추정치 주변
# ±1.3h(p01~p99) 지터. 오프셋 0이면 φ 직후(+0~+1h)에 떨어지는 ~50% 샘플의
# 후방 스냅이 한 주 전 점으로 미끄러져 유효주가 붕괴한다(실측 중앙 6주).
# +12h~+48h 어느 값이든 포착률 99.8%로 동일 — 튜닝 여지 없이 창 중앙 36h 채택.
N_WEEKS = 26                         # 트레일링 26주
SPAN_MIN_D, SPAN_MAX_D = 5.0, 9.0    # 주 실제 스팬 유효 범위
MAX_FLOW_FRAC = 0.20                 # 1차 흐름 한도
MIN_ACCT = 10000.0                   # A_fix 최소 (고정분모)
MIN_VALID_WEEKS = 20                 # 적격: 유효주 ≥ 20/26
ES_FRAC = 0.2                        # ES20: k = ceil(0.2 × n_유효주)
PHASE_GAP_TOL_H = 6.0                # 주간 샘플 판정: 인접 간격 168h±6h
# 감도 변형 (1차 랭킹과 구분)
SENS_FLOW_FRAC = 0.50
SENS_MIN_VALID_WEEKS = 16
# verdict (명세 §전방 결과변수·§검정)
CLIP_LO, CLIP_HI = -0.95, 5.0
SPAN_TOL_D = 5.0                     # 판정 스팬 허용 [H−5, H+5] (H=30 → [25,35])
SNAP_24H_MS = 24 * 3600 * 1000       # 판정시각 이전 24h 이내 스냅만
N_PERM = 10000
PERM_SEED = 20260827
P_ONE_SIDED = 0.025
IC_PASS_MIN = 0.05
SPREAD_PASS_MIN = 0.03               # +3%p
MISS_RATE_MAX = 0.10
MISS_CORR_P = 0.05
Z_ALPHA_025 = 1.959964               # 단측 α=.025
Z_POWER_80 = 0.841621                # power 80%


# ── 공통 유틸 ────────────────────────────────────────────────────────────────
def sha256_of(path: str) -> str:
    """파일 SHA256 (동결 헤더 기록용)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_iso_ms(s: str) -> int:
    """ISO-8601 UTC 문자열 → epoch 밀리초. naive는 UTC로 간주."""
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def iso_utc(ms: float) -> str:
    """epoch 밀리초 → ISO-8601 UTC 문자열."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def mde_ic(n: int, z_alpha: float = Z_ALPHA_025, z_power: float = Z_POWER_80) -> float | None:
    """Fisher z 근사 MDE: 검출가능 최소 IC ≈ (z_{1−α} + z_{power}) / √(n−3).

    단측 α=.025, power 80% → n=274에서 ≈ 0.17 (명세 명기치와 일치).
    """
    if n <= 3:
        return None
    return (z_alpha + z_power) / math.sqrt(n - 3)


# ── 위상정렬 그리드 (감사판 델타 1: 위상 미정렬 → mod 168h 최빈 정렬) ────────
def weekly_phases(ts: np.ndarray) -> np.ndarray:
    """인접 간격이 168h±6h인 '주간 샘플' 점들의 위상(ts mod 168h, ms)을 반환."""
    if len(ts) < 2:
        return np.empty(0, dtype=float)
    gaps = np.diff(ts)
    lo = (168.0 - PHASE_GAP_TOL_H) * 3600e3
    hi = (168.0 + PHASE_GAP_TOL_H) * 3600e3
    okg = (gaps >= lo) & (gaps <= hi)
    keep = np.zeros(len(ts), dtype=bool)
    keep[:-1] |= okg
    keep[1:] |= okg
    return np.asarray(ts, dtype=float)[keep] % WEEK_MS


def estimate_phase(phases: np.ndarray) -> float:
    """위상 φ(ms) 추정 — 1h 빈 최빈(명세: mod 168h 최빈) → 최빈 빈 중심 ±3h
    원형 중앙값으로 정밀화(빈 경계 분할·자정 랩 방어)."""
    phases = np.asarray(phases, dtype=float)
    if len(phases) == 0:
        raise ValueError('위상 추정 불가: 주간 샘플 점 없음')
    hours = np.floor(phases / 3600e3).astype(int) % 168
    counts = np.bincount(hours, minlength=168)
    mode_h = int(np.argmax(counts))
    center = (mode_h + 0.5) * 3600e3
    delta = ((phases - center + WEEK_MS / 2) % WEEK_MS) - WEEK_MS / 2
    near = delta[np.abs(delta) <= 3 * 3600e3]
    if len(near) == 0:
        return float(center % WEEK_MS)
    return float((center + np.median(near)) % WEEK_MS)


def build_grid(phase_ms: float, t_min: float, t_max: float) -> np.ndarray:
    """t_k ≡ φ + 36h (mod 168h) 인 전역 주간 그리드 — [t_min−1주, t_max+1주] 커버.

    그리드를 위상 φ 자체가 아니라 φ + GRID_OFFSET(36h)에 두어, φ 주변
    ±1.3h로 지터하는 주간 샘플 전부가 후방 72h 스냅 창 안에 들어오게 한다
    (상수 정의부의 실측 근거 주석 참조). 후방 스냅만 쓰는 원칙은 불변.
    """
    anchor = phase_ms + GRID_OFFSET_MS
    k0 = int(math.floor((t_min - anchor) / WEEK_MS)) - 1
    k1 = int(math.ceil((t_max - anchor) / WEEK_MS)) + 1
    return anchor + WEEK_MS * np.arange(k0, k1 + 1, dtype=float)


# ── 후방 스냅 (감사판 델타 2: 양방향 최근접 → t 이하 후방만) ────────────────
def snap_backward(ts: np.ndarray, t: float, tol_ms: float = SNAP_BACK_MS) -> int | None:
    """t '이전'(ts[i] ≤ t) 마지막 실제점 인덱스. 후방 tol_ms 이내 없으면 None.

    미래 점(ts > t)은 아무리 가까워도 절대 사용하지 않는다 (스냅 룩어헤드 제거).
    """
    i = int(np.searchsorted(ts, t, side='right')) - 1
    if i < 0:
        return None
    if t - ts[i] > tol_ms:
        return None
    return i


def acct_value_at(acct_ts: np.ndarray, acct_vs: np.ndarray, t: float) -> float | None:
    """타임스탬프 t와 '정확히 일치'하는 acct 곡선 값. 없으면 None.

    실측상 지갑 내 pnl/acct 타임스탬프 배열은 완전 동일 — 불일치는 결측 처리
    (보수적: 대체값·보간 금지).
    """
    j = int(np.searchsorted(acct_ts, t, side='right')) - 1
    if j < 0 or acct_ts[j] != t:
        return None
    return float(acct_vs[j])


# ── 형성 스크린 (지갑 단위) ─────────────────────────────────────────────────
def es_k(n_valid: int) -> int:
    """ES20의 k = ceil(0.2 × n_유효주)."""
    return int(math.ceil(ES_FRAC * n_valid))


def es20_of(r_ws: list[float] | np.ndarray) -> float | None:
    """ES20 = 최저 k개 주간 수익률의 산술평균 (k = ceil(0.2 × n)). n=0이면 None."""
    arr = np.sort(np.asarray(r_ws, dtype=float))
    if len(arr) == 0:
        return None
    return float(arr[:es_k(len(arr))].mean())


def is_eligible(n_valid_weeks: int, total_pnl: float,
                min_weeks: int = MIN_VALID_WEEKS) -> bool:
    """적격: 흐름-유효 주 ≥ min_weeks AND 형성 총 perp PnL > 0."""
    return n_valid_weeks >= min_weeks and total_pnl > 0


def screen_wallet(pnl_ts: np.ndarray, pnl_vs: np.ndarray,
                  acct_ts: np.ndarray, acct_vs: np.ndarray,
                  grid: np.ndarray) -> dict:
    """지갑 1개 트레일링 26주 형성 스크린 (명세 §트랙 A 형성).

    반환: ok=True 시 a_fix / total_pnl / weeks(주별 r_w·flow_frac·span_d) /
    g_start / g_end, ok=False 시 reason. 주별 흐름 한도는 여기서 걸지 않고
    flow_frac만 기록한다 (1차·감도 변형을 variant_stats에서 일괄 적용).
    """
    if len(pnl_ts) < 2:
        return {'ok': False, 'reason': 'too_few_points'}
    last_ts = float(pnl_ts[-1])
    # 수집시각(마지막 곡선점) 이전 마지막 '유효'(후방 스냅 존재) 그리드점에서 역산
    gi_last = int(np.searchsorted(grid, last_ts, side='right')) - 1
    while gi_last >= 0 and snap_backward(pnl_ts, grid[gi_last]) is None:
        gi_last -= 1
    if gi_last < 0:
        return {'ok': False, 'reason': 'no_valid_end_grid'}
    gi_start = gi_last - N_WEEKS
    if gi_start < 0:
        return {'ok': False, 'reason': 'grid_too_short'}
    snaps = [snap_backward(pnl_ts, grid[gi_start + w]) for w in range(N_WEEKS + 1)]
    i0, i_end = snaps[0], snaps[N_WEEKS]
    if i0 is None:
        return {'ok': False, 'reason': 'no_window_start_snap'}
    a_fix = acct_value_at(acct_ts, acct_vs, float(pnl_ts[i0]))
    if a_fix is None:
        return {'ok': False, 'reason': 'no_afix_acct'}
    if a_fix < MIN_ACCT:
        return {'ok': False, 'reason': 'afix_below_min'}
    total_pnl = float(pnl_vs[i_end] - pnl_vs[i0])
    weeks: list[dict] = []
    for w in range(N_WEEKS):
        ia, ib = snaps[w], snaps[w + 1]
        if ia is None or ib is None or ib <= ia:
            continue
        span_d = float((pnl_ts[ib] - pnl_ts[ia]) / DAY_MS)
        if span_d < SPAN_MIN_D or span_d > SPAN_MAX_D:
            continue
        a0w = acct_value_at(acct_ts, acct_vs, float(pnl_ts[ia]))
        a1w = acct_value_at(acct_ts, acct_vs, float(pnl_ts[ib]))
        if a0w is None or a1w is None or a0w <= 0:
            continue
        dpnl = float(pnl_vs[ib] - pnl_vs[ia])
        weeks.append({'w': w, 'r_w': dpnl / a_fix,
                      'flow_frac': abs((a1w - a0w) - dpnl) / a0w,
                      'span_d': span_d})
    return {'ok': True, 'a_fix': float(a_fix), 'total_pnl': total_pnl,
            'weeks': weeks, 'g_start': float(grid[gi_start]),
            'g_end': float(grid[gi_last])}


def variant_stats(weeks: list[dict], total_pnl: float,
                  max_flow: float, min_weeks: int) -> dict:
    """흐름 한도·최소 유효주 한 변형에 대한 유효주 집계·ES20·적격·진단지표."""
    rws = np.asarray([wk['r_w'] for wk in weeks if wk['flow_frac'] <= max_flow],
                     dtype=float)
    n = int(len(rws))
    out: dict = {'n_valid_weeks': n, 'k': None, 'es20': None,
                 'pos_week_frac': None, 'median_rw': None, 'iqr_rw': None,
                 'eligible': is_eligible(n, total_pnl, min_weeks)}
    if n > 0:
        out['k'] = es_k(n)
        out['es20'] = es20_of(rws)
        out['pos_week_frac'] = float((rws > 0).mean())
        out['median_rw'] = float(np.median(rws))
        out['iqr_rw'] = float(np.quantile(rws, 0.75) - np.quantile(rws, 0.25))
    return out


# ── 순위 통계 (verdict) ─────────────────────────────────────────────────────
def rank_avg(x: np.ndarray) -> np.ndarray:
    """1-기반 평균 순위 (동점 average rank)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='mergesort')
    sx = x[order]
    new_grp = np.r_[True, sx[1:] != sx[:-1]]
    grp = np.cumsum(new_grp) - 1
    counts = np.bincount(grp)
    ends = np.cumsum(counts)
    starts = ends - counts
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = avg[grp]
    return ranks


def spearman_avg(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 순위상관 — 동점 average rank (명세 §검정)."""
    rx, ry = rank_avg(x), rank_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def terciles(v: np.ndarray) -> np.ndarray:
    """3분위 라벨(0/1/2) — 경계 1/3·2/3 분위수, 경계값은 상위 그룹."""
    v = np.asarray(v, dtype=float)
    q1, q2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(v, [q1, q2])


def stratified_perm_p(scores: np.ndarray, ys: np.ndarray, strata: np.ndarray,
                      n_perm: int = N_PERM, seed: int = PERM_SEED
                      ) -> tuple[float, float, int]:
    """층화 순열검정 (단측) — 층 내에서만 Y 라벨 순열.

    p = (1 + #{perm IC ≥ 관측 IC}) / (n_perm + 1). seed=20260827 고정.
    무층화 감도는 strata를 단일 값으로 넘겨 동일 함수로 계산한다.
    반환: (p, 관측 IC, 초과 횟수).
    """
    scores = np.asarray(scores, dtype=float)
    ys = np.asarray(ys, dtype=float)
    strata = np.asarray(strata)
    obs = spearman_avg(scores, ys)
    idx_by_s = [np.where(strata == s)[0] for s in sorted(set(strata.tolist()))]
    rng = np.random.default_rng(seed)
    yp = ys.copy()
    cnt = 0
    for _ in range(n_perm):
        for idx in idx_by_s:
            yp[idx] = ys[idx][rng.permutation(len(idx))]
        if spearman_avg(scores, yp) >= obs:
            cnt += 1
    return (1 + cnt) / (n_perm + 1), obs, cnt


def missingness_verdict(scores: np.ndarray, miss: np.ndarray,
                        n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """판정불가 검사 (명세 §전방 결과변수).

    결측률 > 10% 또는 결측 지표~점수 Spearman 유의(순열 양측 p < 0.05)면
    indeterminate=True (기각 아님). p = (1 + #{|perm| ≥ |관측|}) / (n_perm + 1).
    """
    scores = np.asarray(scores, dtype=float)
    miss = np.asarray(miss, dtype=float)
    n = len(miss)
    rate = float(miss.mean()) if n else 0.0
    if n == 0 or float(miss.min()) == float(miss.max()):
        rho, p = 0.0, 1.0
    else:
        rho = spearman_avg(scores, miss)
        rng = np.random.default_rng(seed)
        cnt = 0
        for _ in range(n_perm):
            r = spearman_avg(scores, miss[rng.permutation(n)])
            if abs(r) >= abs(rho):
                cnt += 1
        p = (1 + cnt) / (n_perm + 1)
    return {'missing_rate': rate, 'rho': float(rho), 'p': float(p),
            'indeterminate': bool(rate > MISS_RATE_MAX or p < MISS_CORR_P)}


# ── 전방 결과변수 Y (verdict) ───────────────────────────────────────────────
def compute_y(p0: float, ph: float, a0: float, span_d: float, horizon: float,
              clip: bool = True) -> float | None:
    """전방 결과변수 Y — 명세 서술 순서 그대로: 비율 → 클리핑 → 기간 정규화.

    Y = clip((P_perp(T_H) − P_perp(T0)) / A(T0), −0.95, +5.0);
    실제 스팬 ≠ H이면 Y × (H / 실제일수). 스팬 [H−5, H+5]일 밖이면 제외(None).
    clip=False는 unclipped 감도분석용 (클리핑만 생략, 정규화·제외는 동일).
    비유한(NaN/Inf) 입력은 결측(None) — NaN이 spearman_avg에 닿지 않게 차단.
    """
    if not (math.isfinite(p0) and math.isfinite(ph) and math.isfinite(span_d)):
        return None
    if not math.isfinite(a0) or a0 <= 0:
        return None
    if span_d < horizon - SPAN_TOL_D or span_d > horizon + SPAN_TOL_D:
        return None
    y = (ph - p0) / a0
    if clip:
        y = min(max(y, CLIP_LO), CLIP_HI)
    if span_d != horizon:
        y = y * (horizon / span_d)
    return float(y)


def load_snapshot(path: str, expected_label: str,
                  counts: Counter | None = None) -> dict[str, dict]:
    """스냅샷 jsonl.gz 로드 (계약: docstring 상단) — 지정 라벨 행만, 라벨 내 keep-last.

    같은 날짜 파일에 t0/daily/verdict 라벨이 공존할 수 있다 (판정일은 일별
    크론과 같은 날, 재시도는 뒤에 append). 라벨을 무시하면 t0 뒤에 append 된
    daily 재시도가 T0 기준선을 덮으므로 expected_label 행만 사용한다.
    label 필드 없는 행은 계약 위반으로 거부하고 counts['no_label'] 로,
    다른 라벨 행은 counts['other_label'] 로 집계한다 (counts=None 이면 로깅만).

    json.loads가 허용하는 비표준 NaN/Infinity 리터럴은 줄 단위로 건너뛰어
    자연히 결측(no_t0/no_th)으로 흡수한다 — spearman_avg는 NaN 유입 시
    무정의이므로 입구에서 차단.
    """
    cnt: Counter = counts if counts is not None else Counter()
    out: dict[str, dict] = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                cnt['bad_json'] += 1
                continue
            if not isinstance(r, dict):
                cnt['bad_row'] += 1
                continue
            label = r.get('label')
            if label is None:
                cnt['no_label'] += 1
                continue
            if str(label) != expected_label:
                cnt['other_label'] += 1
                continue
            try:
                addr = str(r['address']).lower()
                pnl = float(r['perp_alltime_pnl'])
                acct = float(r['account_value'])
                cap_ms = parse_iso_ms(r['captured_at_utc'])
            except (KeyError, TypeError, ValueError):
                cnt['bad_row'] += 1
                continue
            if not (math.isfinite(pnl) and math.isfinite(acct)):
                cnt['nonfinite'] += 1
                continue
            out[addr] = {'pnl': pnl, 'acct': acct, 'cap_ms': cap_ms}
    if counts is None and (cnt['no_label'] or cnt['bad_json'] or cnt['bad_row']):
        logger.warning('load_snapshot(%s, label=%s): 거부 행 %s',
                       path, expected_label, dict(cnt))
    return out


def evaluate_forward(wallets: list[dict], t0: dict[str, dict], th: dict[str, dict],
                     judgment_ms: float, horizon: float) -> dict[str, dict]:
    """지갑별 전방 결과 계산 — Y(클립)·Y(unclipped)·A(T0)·스팬·결측 사유.

    스냅샷 조회는 소문자 정규화 주소(load_snapshot 키와 대칭), 출력 키는
    코호트 원 주소 유지. 비유한 pnl은 bad_value로 결측 처리.
    """
    out: dict[str, dict] = {}
    for w in wallets:
        addr = w['address']
        low = str(addr).lower()
        rec: dict = {'y': None, 'y_unclipped': None, 'a0': None,
                     'span_d': None, 'reason': None}
        s0, sh = t0.get(low), th.get(low)
        if s0 is None:
            rec['reason'] = 'no_t0'
        elif sh is None:
            rec['reason'] = 'no_th'
        elif not (math.isfinite(s0['pnl']) and math.isfinite(sh['pnl'])):
            rec['reason'] = 'bad_value'
        elif not math.isfinite(s0['acct']) or s0['acct'] <= 0:
            rec['reason'] = 'bad_a0'
        elif not (0 <= judgment_ms - sh['cap_ms'] <= SNAP_24H_MS):
            rec['reason'] = 'outside_24h'
        else:
            span_d = (sh['cap_ms'] - s0['cap_ms']) / DAY_MS
            rec['a0'] = s0['acct']
            rec['span_d'] = span_d
            y = compute_y(s0['pnl'], sh['pnl'], s0['acct'], span_d, horizon, clip=True)
            if y is None:
                rec['reason'] = 'span_out'
            else:
                rec['y'] = y
                rec['y_unclipped'] = compute_y(s0['pnl'], sh['pnl'], s0['acct'],
                                               span_d, horizon, clip=False)
        out[addr] = rec
    return out


def ordered_indices(scores: np.ndarray, addrs: list[str]) -> list[int]:
    """ES20 내림차순(동점은 주소 오름차순) 결정론적 정렬 인덱스."""
    return sorted(range(len(addrs)), key=lambda i: (-scores[i], addrs[i]))


# ── screen 서브커맨드 ───────────────────────────────────────────────────────
def _parse_curves(r: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """포트폴리오 행에서 perpAllTime pnl/acct 곡선 4배열 (값은 문자열 → float)."""
    cur = r.get('perpAllTime') or {}          # "perpAllTime": null 방어
    pnl_c, acct_c = cur.get('pnl', []), cur.get('acct', [])
    if len(pnl_c) < 2 or len(acct_c) < 2:
        return None
    pts = np.asarray([float(q[0]) for q in pnl_c], dtype=float)
    pvs = np.asarray([float(q[1]) for q in pnl_c], dtype=float)
    ats = np.asarray([float(q[0]) for q in acct_c], dtype=float)
    avs = np.asarray([float(q[1]) for q in acct_c], dtype=float)
    return pts, pvs, ats, avs


def _assign_ranks(wallets: list[dict], es_getter, rank_setter) -> int:
    """변형별 적격 지갑에 ES20 내림차순 1-기반 랭크 부여. 반환: 적격 수."""
    elig = [w for w in wallets if es_getter(w) is not None]
    elig.sort(key=lambda w: (-es_getter(w), w['address']))
    for rk, w in enumerate(elig, start=1):
        rank_setter(w, rk)
    return len(elig)


def cmd_screen(args: argparse.Namespace) -> None:
    """형성 스크린 실행 → logs/h2_cohort.json.gz 동결 산출."""
    sha_port = sha256_of(args.portfolio)
    sha_coh = sha256_of(args.cohort)
    with gzip.open(args.cohort, 'rt') as f:
        coh = json.load(f)
    coh_map = {str(w['address']).lower(): w for w in coh['wallets']}
    print(f"코호트 {len(coh_map)}지갑 (locked_at={coh.get('locked_at')}) — 신규 스크린 없음")

    # 1패스: 위상 추정 + 전역 ts 범위
    phase_parts: list[np.ndarray] = []
    tmin, tmax = math.inf, -math.inf
    with gzip.open(args.portfolio, 'rt') as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pnl_c = (r.get('perpAllTime') or {}).get('pnl', [])
            if len(pnl_c) < 2:
                continue
            ts = np.asarray([float(q[0]) for q in pnl_c], dtype=float)
            tmin = min(tmin, float(ts[0]))
            tmax = max(tmax, float(ts[-1]))
            ph = weekly_phases(ts)
            if len(ph):
                phase_parts.append(ph)
    phase_ms = estimate_phase(np.concatenate(phase_parts))
    grid = build_grid(phase_ms, tmin, tmax)
    print(f"위상 φ = {phase_ms / 3.6e6:.2f}h (mod 168h, 주간 샘플 {sum(len(p) for p in phase_parts)}점) "
          f"— 그리드 {len(grid)}점, t ≡ φ + 36h (mod 168h)")

    # 2패스: 지갑별 스크린
    excl: Counter = Counter()
    seen: set[str] = set()
    wallets: list[dict] = []
    with gzip.open(args.portfolio, 'rt') as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                excl['bad_json'] += 1
                continue
            addr = r.get('address')
            addr = str(addr).lower() if addr is not None else None
            cw = coh_map.get(addr)
            if cw is None:
                excl['not_in_cohort'] += 1
                continue
            if addr in seen:
                excl['dup_line'] += 1         # 같은 주소 중복 줄 → 첫 줄만 사용
                continue
            seen.add(addr)
            curves = _parse_curves(r)
            if curves is None:
                excl['too_few_points'] += 1
                continue
            res = screen_wallet(*curves, grid)
            if not res['ok']:
                excl[res['reason']] += 1
                continue
            primary = variant_stats(res['weeks'], res['total_pnl'],
                                    MAX_FLOW_FRAC, MIN_VALID_WEEKS)
            flow50 = variant_stats(res['weeks'], res['total_pnl'],
                                   SENS_FLOW_FRAC, MIN_VALID_WEEKS)
            minw16 = variant_stats(res['weeks'], res['total_pnl'],
                                   MAX_FLOW_FRAC, SENS_MIN_VALID_WEEKS)
            if not (primary['eligible'] or flow50['eligible'] or minw16['eligible']):
                excl['not_eligible_any'] += 1
                continue
            t0_acct = float(cw.get('t0_account') or 0.0)
            t0_vlm = float(cw.get('t0_month_vlm') or 0.0)
            wallets.append({
                'address': addr,
                'a_fix': res['a_fix'],
                'total_pnl': res['total_pnl'],
                'eligible': primary['eligible'],
                'n_valid_weeks': primary['n_valid_weeks'],
                'k': primary['k'],
                'es20': primary['es20'],
                'rank': None,
                'pos_week_frac': primary['pos_week_frac'],
                'median_rw': primary['median_rw'],
                'iqr_rw': primary['iqr_rw'],
                't0_month_vlm': t0_vlm,
                'turnover': (t0_vlm / t0_acct) if t0_acct > 0 else None,
                'g_start_utc': iso_utc(res['g_start']),
                'g_end_utc': iso_utc(res['g_end']),
                'sens_flow50': {**flow50, 'rank': None},
                'sens_minw16': {**minw16, 'rank': None},
            })
    for a in coh_map:
        if a not in seen:
            excl['no_curve'] += 1

    # 랭킹 (변형별로 독립, 1차 랭킹과 구분)
    n_primary = _assign_ranks(
        wallets,
        lambda w: w['es20'] if w['eligible'] else None,
        lambda w, rk: w.__setitem__('rank', rk))
    n_flow50 = _assign_ranks(
        wallets,
        lambda w: w['sens_flow50']['es20'] if w['sens_flow50']['eligible'] else None,
        lambda w, rk: w['sens_flow50'].__setitem__('rank', rk))
    n_minw16 = _assign_ranks(
        wallets,
        lambda w: w['sens_minw16']['es20'] if w['sens_minw16']['eligible'] else None,
        lambda w, rk: w['sens_minw16'].__setitem__('rank', rk))

    header = {
        'spec': 'H2 트랙 A 형성 스크린 (H2 구현 명세 v1, Codex 2라운드 합의, 2026-08-27)',
        'generated_at_utc': datetime.now(tz=timezone.utc).isoformat(),
        'inputs': {
            'portfolio': {'path': args.portfolio, 'sha256': sha_port},
            'cohort': {'path': args.cohort, 'sha256': sha_coh},
        },
        'params': {
            'week_hours': 168, 'snap_backward_days': 3, 'n_weeks': N_WEEKS,
            'span_min_days': SPAN_MIN_D, 'span_max_days': SPAN_MAX_D,
            'max_flow_frac': MAX_FLOW_FRAC, 'min_acct_usd': MIN_ACCT,
            'min_valid_weeks': MIN_VALID_WEEKS, 'es_frac': ES_FRAC,
            'phase_ms': phase_ms, 'phase_hours': phase_ms / 3.6e6,
            'phase_gap_tol_hours': PHASE_GAP_TOL_H,
            'grid_offset_hours': GRID_OFFSET_MS / 3.6e6,
            'grid_rule': 't ≡ phase + 36h (mod 168h), 후방 3일 스냅만 '
                         '(오프셋: 주간 샘플 위상 지터 ±1.3h 포착용, 코드 주석 참조)',
            'eligibility': '흐름-유효 주 ≥ 20/26 AND 형성 총 perp PnL > 0',
            'es20': 'k = ceil(0.2 × n_유효주), 최저 k개 r_w 산술평균, 내림차순 랭킹',
        },
        'sensitivity_params': {
            'flow50': {'max_flow_frac': SENS_FLOW_FRAC, 'min_valid_weeks': MIN_VALID_WEEKS},
            'minw16': {'max_flow_frac': MAX_FLOW_FRAC, 'min_valid_weeks': SENS_MIN_VALID_WEEKS},
        },
        'counts': {
            'cohort': len(coh_map), 'screened_any_eligible': len(wallets),
            'eligible_primary': n_primary, 'eligible_flow50': n_flow50,
            'eligible_minw16': n_minw16, 'exclusions': dict(excl),
        },
        'mde': {'n_primary': n_primary, 'ic': mde_ic(n_primary),
                'alpha_one_sided': 0.025, 'power': 0.80,
                'formula': '(z_.975 + z_.80) / sqrt(n-3)'},
    }
    with gzip.open(args.out, 'wt') as f:
        json.dump({'header': header, 'wallets': wallets}, f, ensure_ascii=False)

    es_arr = np.asarray([w['es20'] for w in wallets if w['eligible']], dtype=float)
    print(f"\n1차 적격 {n_primary}지갑 (감도: 흐름≤50% {n_flow50}, 유효주≥16 {n_minw16})")
    print(f"제외 사유: {dict(excl)}")
    if len(es_arr):
        q = np.quantile(es_arr, [0.1, 0.25, 0.5, 0.75, 0.9]) * 100
        print(f"ES20 분포(1차, %): p10 {q[0]:+.2f} p25 {q[1]:+.2f} 중앙 {q[2]:+.2f} "
              f"p75 {q[3]:+.2f} p90 {q[4]:+.2f}")
    m = mde_ic(n_primary)
    print(f"MDE: 단측 α=.025, power 80%에서 검출가능 IC ≈ {m:.3f} (n={n_primary})"
          if m else "MDE: 표본 부족")
    print(f"\n동결 산출물: {args.out} (입력 SHA256 헤더 기록)")


# ── verdict 서브커맨드 ──────────────────────────────────────────────────────
def _write_result_json(path: str | None, result: dict) -> None:
    """--out-json 지정 시 판정 결과 JSON 저장 — 트랙 B 게이트(h2_trackb gate) 입력.

    미지정(None)이면 아무것도 하지 않는다 (stdout 보고는 기존과 동일).
    """
    if not path:
        return
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n판정 결과 JSON 저장: {path}")


def _analysis_block(label: str, scores: np.ndarray, ys: np.ndarray,
                    strata: np.ndarray, addrs: list[str]) -> dict:
    """한 (점수, Y) 집합의 IC·층화 순열 p·십분위 통계를 계산."""
    p, ic, cnt = stratified_perm_p(scores, ys, strata)
    order = ordered_indices(scores, addrs)
    groups = np.array_split(np.asarray(order), 10)
    dec_med = [float(np.median(ys[g])) if len(g) else math.nan for g in groups]
    top_med = dec_med[0]
    bot_med = dec_med[-1]
    return {'label': label, 'n': len(ys), 'ic': ic, 'p': p, 'perm_ge': cnt,
            'decile_medians': dec_med, 'spread': top_med - bot_med,
            'top_median': top_med, 'all_median': float(np.median(ys))}


def cmd_verdict(args: argparse.Namespace) -> None:
    """판정 평가 실행 — 사전 동결된 검정 절차 (데이터 조회 전 작성).

    스냅샷은 라벨 필터로 로드한다 (--t0-label 기본 t0, --th-label 기본 verdict) —
    같은 날짜 파일의 daily 재시도 행이 T0/T_H 기준선을 덮는 경로 차단 (명세 §2.3).
    --out-json 지정 시 판정 결과를 JSON 으로도 저장한다 (트랙 B 게이트 입력).
    """
    horizon = float(args.horizon)
    with gzip.open(args.h2_cohort, 'rt') as f:
        cohort = json.load(f)
    all_wallets = cohort['wallets']
    ranked = sorted((w for w in all_wallets if w['eligible'] and w['es20'] is not None),
                    key=lambda w: w['rank'])
    t0_cnt: Counter = Counter()
    th_cnt: Counter = Counter()
    t0 = load_snapshot(args.t0, args.t0_label, t0_cnt)
    th = load_snapshot(args.th, args.th_label, th_cnt)
    if args.judgment:
        judgment_ms = float(parse_iso_ms(args.judgment))
    else:
        if not th:
            print(f"T_H 스냅샷 비어 있음 (라벨 '{args.th_label}') — 평가 불가")
            return
        judgment_ms = float(max(s['cap_ms'] for s in th.values()))
    sha_cohort = sha256_of(args.h2_cohort)
    sha_t0 = sha256_of(args.t0)
    sha_th = sha256_of(args.th)
    print(f"입력: h2_cohort={args.h2_cohort} sha256={sha_cohort[:16]}…")
    print(f"      t0={args.t0} sha256={sha_t0[:16]}… "
          f"(라벨 '{args.t0_label}' {len(t0)}지갑, 비채택 {dict(t0_cnt)})")
    print(f"      th={args.th} sha256={sha_th[:16]}… "
          f"(라벨 '{args.th_label}' {len(th)}지갑, 비채택 {dict(th_cnt)})")
    print(f"H={horizon:.0f}일, 판정시각={iso_utc(judgment_ms)} "
          f"({'명시' if args.judgment else 'T_H captured_at 최댓값'}), "
          f"스팬 허용 [{horizon - SPAN_TOL_D:.0f}, {horizon + SPAN_TOL_D:.0f}]일, "
          f"클리핑 [{CLIP_LO}, {CLIP_HI}]")

    fwd = evaluate_forward(all_wallets, t0, th, judgment_ms, horizon)

    result: dict = {
        'kind': 'h2_verdict',
        'generated_at_utc': datetime.now(tz=timezone.utc).isoformat(),
        'horizon_days': horizon,
        'judgment_utc': iso_utc(judgment_ms),
        'inputs': {
            'h2_cohort': {'path': args.h2_cohort, 'sha256': sha_cohort},
            't0': {'path': args.t0, 'sha256': sha_t0, 'label': args.t0_label},
            'th': {'path': args.th, 'sha256': sha_th, 'label': args.th_label},
        },
    }

    # ── 1차 랭킹: 결측 분석 (제외 = 결측으로 집계) ──
    n_rank = len(ranked)
    miss = np.asarray([1.0 if fwd[w['address']]['y'] is None else 0.0 for w in ranked])
    scores_all = np.asarray([w['es20'] for w in ranked], dtype=float)
    addrs_all = [w['address'] for w in ranked]
    reasons = Counter(fwd[w['address']]['reason'] for w in ranked
                      if fwd[w['address']]['y'] is None)
    mv = missingness_verdict(scores_all, miss)
    print(f"\n1차 적격 {n_rank}지갑 — 결측/제외 {int(miss.sum())}개 "
          f"({mv['missing_rate'] * 100:.1f}%), 사유 {dict(reasons)}")
    order_all = ordered_indices(scores_all, addrs_all)
    dec_groups_all = np.array_split(np.asarray(order_all), 10)
    dec_miss = [int(miss[g].sum()) for g in dec_groups_all]
    print("ES20 십분위별 결측(상위→하위): " + ' '.join(f"{d}" for d in dec_miss))
    print(f"결측~점수 Spearman rho={mv['rho']:+.3f} 순열 양측 p={mv['p']:.4f}")
    result['n_ranked'] = n_rank
    result['missingness'] = {**mv, 'n_missing': int(miss.sum()),
                             'reasons': dict(reasons), 'decile_missing': dec_miss}
    indeterminate = mv['indeterminate']
    result['indeterminate'] = bool(indeterminate)
    if indeterminate:
        why = []
        if mv['missing_rate'] > MISS_RATE_MAX:
            why.append(f"결측률 {mv['missing_rate'] * 100:.1f}% > 10%")
        if mv['p'] < MISS_CORR_P:
            why.append(f"결측~점수 유의 상관 (p={mv['p']:.4f} < 0.05)")
        print(f"→ 판정불가 (기각 아님): {'; '.join(why)}")

    ev = [w for w in ranked if fwd[w['address']]['y'] is not None]
    if len(ev) < 10:
        print(f"\n평가 가능 {len(ev)}지갑 (<10) — 십분위 분석 불가, 종료")
        result.update(main=None, criteria=None, passed=False,
                      insufficient_evaluable=len(ev))
        _write_result_json(args.out_json, result)
        return
    scores = np.asarray([w['es20'] for w in ev], dtype=float)
    addrs = [w['address'] for w in ev]
    ys = np.asarray([fwd[w['address']]['y'] for w in ev], dtype=float)
    yu = np.asarray([fwd[w['address']]['y_unclipped'] for w in ev], dtype=float)
    a0s = np.asarray([fwd[w['address']]['a0'] for w in ev], dtype=float)
    turn = np.asarray([w['turnover'] if w['turnover'] is not None else 0.0
                       for w in ev], dtype=float)
    strata = 3 * terciles(np.log(a0s)) + terciles(np.log1p(turn))

    main = _analysis_block('1차 (층화)', scores, ys, strata, addrs)
    print(f"\n── 1차 검정 (n={main['n']}, 층화 순열 {N_PERM}회, seed={PERM_SEED}) ──")
    print(f"IC = {main['ic']:+.4f}  단측 p = {main['p']:.4f} "
          f"(#{{perm≥obs}}={main['perm_ge']})")
    print("십분위 Y 중앙(상위→하위, %): "
          + ' '.join(f"{m * 100:+.2f}" for m in main['decile_medians']))
    print(f"상위−하위 십분위 스프레드 {main['spread'] * 100:+.2f}%p | "
          f"상위 십분위 중앙 {main['top_median'] * 100:+.2f}% vs "
          f"전체 중앙 {main['all_median'] * 100:+.2f}%")
    m = mde_ic(main['n'])
    print(f"MDE(단측 α=.025, power 80%): IC ≈ {m:.3f} (n={main['n']})" if m else "MDE: n/a")

    c1 = main['ic'] >= IC_PASS_MIN
    c2 = main['p'] < P_ONE_SIDED
    c3 = main['spread'] >= SPREAD_PASS_MIN
    c4 = main['top_median'] > main['all_median']
    ok = lambda b: '충족' if b else '미달'  # noqa: E731
    print(f"\n기준(보고용 — 자본 권한 없음): (a)IC≥+0.05: {ok(c1)}  "
          f"(b)단측 p<0.025: {ok(c2)}  (c)스프레드≥+3%p: {ok(c3)}  "
          f"(d)상위중앙>전체중앙: {ok(c4)}")
    if indeterminate:
        print("  종합: 판정불가 (기각 아님 — 결측 규칙)")
    else:
        print(f"  종합: {'통과' if all([c1, c2, c3, c4]) else '미달'} "
              f"[트랙 A는 보조검정 — 자본 권한 영구 없음]")
    result['main'] = main
    result['mde_ic'] = m
    result['criteria'] = {'ic_ge_min': bool(c1), 'p_lt_alpha': bool(c2),
                          'spread_ge_min': bool(c3), 'top_gt_all': bool(c4)}
    result['passed'] = bool(not indeterminate and c1 and c2 and c3 and c4)
    _write_result_json(args.out_json, result)   # 감도분석 전에 저장 (게이트 입력 확보)

    # ── 감도 (통과 권한 없음, 기술 보고만) ──
    print("\n── 감도분석 (기술 보고만) ──")
    p_u, ic_u, _ = stratified_perm_p(scores, ys, np.zeros(len(ys), dtype=int))
    print(f"무층화: IC = {ic_u:+.4f}  단측 p = {p_u:.4f}")
    unc = _analysis_block('unclipped (층화)', scores, yu, strata, addrs)
    print(f"unclipped: IC = {unc['ic']:+.4f}  단측 p = {unc['p']:.4f}  "
          f"스프레드 {unc['spread'] * 100:+.2f}%p")
    hi = terciles(turn) == 2
    if int(hi.sum()) >= 10:
        sub = _analysis_block('고회전율', scores[hi], ys[hi],
                              strata[hi], [a for a, h in zip(addrs, hi) if h])
        print(f"고회전율 서브그룹(회전율 상위 3분위, n={sub['n']}): "
              f"IC = {sub['ic']:+.4f}  단측 p = {sub['p']:.4f}  "
              f"Y 중앙 {float(np.median(ys[hi])) * 100:+.2f}%")
    else:
        print(f"고회전율 서브그룹: n={int(hi.sum())} (<10) — 생략")
    for key, name in (('sens_flow50', '흐름≤50% 랭킹'), ('sens_minw16', '유효주≥16 랭킹'),
                      ('sens_minw8', '유효주≥8 랭킹')):
        have = [w for w in all_wallets if isinstance(w.get(key), dict)]
        if not have:
            continue    # 이 감도 변형이 없는 코호트 (트랙 A: minw16, 트랙 B: minw8)
        sv = [w for w in have
              if w[key]['eligible'] and w[key]['es20'] is not None
              and fwd[w['address']]['y'] is not None]
        if len(sv) < 10:
            print(f"{name}: 평가 가능 {len(sv)} (<10) — 생략")
            continue
        s_sc = np.asarray([w[key]['es20'] for w in sv], dtype=float)
        s_ys = np.asarray([fwd[w['address']]['y'] for w in sv], dtype=float)
        print(f"{name} (n={len(sv)}): IC = {spearman_avg(s_sc, s_ys):+.4f}")


def main(argv: list[str] | None = None) -> None:
    """CLI 엔트리 — screen / verdict 서브커맨드."""
    ap = argparse.ArgumentParser(
        prog='h2_consistency',
        description='H2 트랙 A: 형성 스크린(screen) / 판정 평가(verdict) — 사전등록 동결')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('screen', help='형성 스크린 → logs/h2_cohort.json.gz')
    s.add_argument('--portfolio', default='logs/trader_portfolio.jsonl.gz')
    s.add_argument('--cohort', default='logs/trader_cohort.json.gz')
    s.add_argument('--out', default='logs/h2_cohort.json.gz')
    s.set_defaults(func=cmd_screen)
    v = sub.add_parser('verdict', help='판정일 평가 (T0·T_H 스냅샷 필요)')
    v.add_argument('--t0', required=True, help='T0 스냅샷 jsonl.gz')
    v.add_argument('--th', required=True, help='판정일 스냅샷 jsonl.gz')
    v.add_argument('--horizon', type=int, default=30, help='H (일), 기본 30')
    v.add_argument('--h2-cohort', default='logs/h2_cohort.json.gz')
    v.add_argument('--judgment', default=None,
                   help='판정시각 ISO-8601 UTC (기본: T_H captured_at 최댓값)')
    v.add_argument('--t0-label', default='t0',
                   help='T0 스냅샷에서 사용할 라벨 행 (기본 t0)')
    v.add_argument('--th-label', default='verdict',
                   help='판정일 스냅샷에서 사용할 라벨 행 (기본 verdict)')
    v.add_argument('--out-json', default=None,
                   help='판정 결과 JSON 저장 경로 (트랙 B 게이트 입력, 선택)')
    v.set_defaults(func=cmd_verdict)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main(sys.argv[1:])
