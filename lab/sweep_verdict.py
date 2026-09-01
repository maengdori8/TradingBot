"""SWEEP-2026-08-31 판정 스크립트 — White RC · Romano–Wolf StepM · DSR · SPA.

명세: `docs/PREREGISTRATION_SWEEP_2026-08-31.md` §6 (다중검정 보정) · §7 (IS/OOS) ·
§5.1 (1차 지표). 입력은 `lab/sweep_engine.py` 가 기록한 일수익률 행렬이며, 본 모듈은
백테스트를 다시 돌리지 않는다 — **판정만** 한다.

파이프라인
----------
1. `logs/sweep_returns.npz` → `R` (3,390 × 1,737) 일수익률.
2. 관측 통계량 = 규칙별 일 단위 Sharpe (`mean/std(ddof=1)`), 퇴화 시 `0` (§5.1).
3. 귀무 부과: 각 규칙 열을 **평균 0으로 중심화** (§6.2-2).
4. **동기화 stationary bootstrap** (Politis–Romano, 평균 블록 5일, 10,000 경로,
   seed 20260831). 한 경로의 날짜 인덱스 수열을 3,390 규칙 전부에 동일 적용.
5. `C @ Rᵀ` · `C @ (R²)ᵀ` 두 번의 행렬곱으로 전 경로·전 규칙 Sharpe 를 얻는다
   (§6.2 계산 항등식). 경로 청크 1,000.
6. RC p (§6.2-5) → StepM 생존 집합 (§6.3) → 생존 규칙별 DSR (§6.4) → 통과 판정 (§6.5).
7. 진단(권한 없음): SPA p (§6.7) · IS 전용 RC (§7) · IS/OOS 순위 상관.

계산 항등식의 적용 범위 (§6.2)
-----------------------------
Sharpe 는 표본 다중집합의 대칭함수이므로 중복도 행렬 단축이 **수학적으로 정확**하다.
MDD·자본곡선 등 **순서 의존 통계에는 절대 사용하지 않는다** — 본 모듈은 순서 의존
통계를 부트스트랩하지 않으며, 보조 지표는 엔진 요약 CSV 의 관측값만 인용한다.

부동소수점 공시 (정직 — §11.1 테스트 ⑦ 문언과의 편차)
----------------------------------------------------
§11.1 ⑦ 은 "중복도 행렬 항등식이 순진한 재표집과 **비트 수준** 일치" 를 요구한다.
항등식은 실수 연산에서 **정확**하지만, 부동소수점에서는 덧셈 순서가 다르므로
(BLAS 블록 합 vs 재표집 계열의 순차 합) 마지막 몇 ulp 가 다를 수 있다 — 이는
구현 결함이 아니라 부동소수점 결합법칙 위반이며, 어떤 알고리즘으로도 비트 일치를
보장할 수 없다. 따라서 테스트는 (a) 상대오차 상한 `1e-10` 을 강제하고
(b) 실측 최대 상대오차·정확 일치 비율을 함께 보고한다. 결정 임계(1e-4 수준의 Sharpe
차이)보다 6자리 이상 작으므로 판정에 영향이 없다.

실행 환경 공시 (§9.4)
--------------------
§9.4 는 판정을 Linux x86-64 / Python 3.11 에서 수행하라고 규정한다. 본 실행 환경은
JSON `env` 에 그대로 기록되며, 규정 환경과 다르면 `env.matches_prereg = false` 로
표시된다. 백테스트에는 무작위성이 없고 부트스트랩은 numpy PCG64 (플랫폼 독립
비트 재현) 를 쓰므로 인덱스 수열은 동일하나, BLAS 구현 차이로 마지막 ulp 가
달라질 수 있다 — 이 사실을 은폐하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── 동결 상수 (§11.3 — 변경 금지) ──────────────────────────────────────────
SEED: int = 20260831
N_PATHS: int = 10_000
MEAN_BLOCK_DAYS: float = 5.0
RC_ALPHA: float = 0.05
FWER: float = 0.05
DSR_MIN: float = 0.95
N_TRIALS: int = 3390
N_DAYS: int = 1737
ANN: float = math.sqrt(365.0)
EULER_GAMMA: float = 0.5772156649
IS_END: pd.Timestamp = pd.Timestamp("2024-12-31T23:59:00Z")

# §9.4 규정 실행 환경
PREREG_ENV: dict[str, str] = {"platform": "linux", "machine": "x86_64", "python": "3.11"}

CHUNK: int = 1_000
REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 입력 ──────────────────────────────────────────────────────────────────
@dataclass
class SweepData:
    """엔진 산출물 1건."""

    returns: np.ndarray          # (N, T) 일수익률
    rule_ids: np.ndarray         # (N,) 문자열
    snap_ts: pd.DatetimeIndex    # (T+1,) 자정 스냅샷 시각
    meta: dict[str, Any]
    summary: pd.DataFrame | None


def load_sweep(npz_path: Path, csv_path: Path | None = None) -> SweepData:
    """엔진이 기록한 일수익률 행렬과 요약을 읽는다.

    Args:
        npz_path: `sweep_returns.npz` 경로.
        csv_path: `sweep_summary.csv` 경로 (없으면 계열/방향 분해 생략).

    Returns:
        `SweepData`.

    Raises:
        SystemExit: 행렬 모양이 §11.3 동결 상수와 다를 때 (fail-closed).
    """
    z = np.load(npz_path, allow_pickle=True)
    ret = np.ascontiguousarray(z["daily_returns"], dtype=np.float64)
    rid = np.asarray(z["rule_ids"], dtype=object)
    snap = pd.DatetimeIndex([pd.Timestamp(s) for s in z["snap_ts"]])
    meta = json.loads(str(z["meta"]))
    if ret.shape != (N_TRIALS, N_DAYS):
        raise SystemExit(f"일수익률 행렬 {ret.shape} != ({N_TRIALS}, {N_DAYS})")
    if len(rid) != N_TRIALS or len(set(rid.tolist())) != N_TRIALS:
        raise SystemExit("규칙 ID 개수/유일성 위반")
    if not np.isfinite(ret).all():
        raise SystemExit("일수익률에 NaN/Inf 존재 — fail-closed")
    summ = pd.read_csv(csv_path) if csv_path is not None and csv_path.exists() else None
    return SweepData(ret, rid, snap, meta, summ)


# ── 1차 지표 (§5.1) ───────────────────────────────────────────────────────
def degenerate_mask(ret: np.ndarray) -> np.ndarray:
    """§5.1 퇴화 규칙 마스크 — 일수익률이 **상수**인 규칙 (거래 0 포함).

    "std(r) = 0" 은 수학적으로 "모든 값이 같다" 와 동치다. 부동소수점 `std()` 는
    상수 계열에서도 1e-18 급 잔차를 낼 수 있고 그것을 나누면 Sharpe 가 1e15 로
    폭발한다 — 그래서 std 값이 아니라 **상수성(ptp == 0)** 으로 판정한다.
    실제 일수익률의 std 는 1e-2 급이라 이 판정이 진짜 규칙을 걸러낼 여지는 없다
    (사후 필터가 아니라 §5.1 정의의 정확한 구현이다).

    Args:
        ret: (N, T) 일수익률.

    Returns:
        (N,) bool. True 면 `SR := 0`.
    """
    return np.ptp(ret, axis=1) == 0.0


# 분산이 "0과 구별되지 않는다"고 볼 상대 문턱 (§5.1 퇴화 처리의 수치 구현).
# 2차 적률 대비 이 값 이하이면 표본을 상수로 간주한다. 기계 엡실론(2.2e-16)의
# 약 4,600배 — 어떤 누산 오차보다 크고, 실제 일수익률 분산(비율 ~1)보다 12자리 작다.
VAR_ZERO_REL: float = 1e-12


def _sharpe_from_var(mean: np.ndarray, var: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """평균·분산·2차적률에서 Sharpe 를 만든다 (§5.1 퇴화 처리 포함).

    `var <= VAR_ZERO_REL · m2` 이면 표본을 상수로 보고 `SR := 0`.

    Args:
        mean: 표본 평균. var: 표본 분산(ddof=1). m2: 2차 적률 `Σx²/T`.

    Returns:
        Sharpe (퇴화 시 0).
    """
    ok = (var > 0.0) & (var > VAR_ZERO_REL * m2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = np.where(ok, mean / np.sqrt(np.where(ok, var, 1.0)), 0.0)
    return np.where(np.isfinite(sr), sr, 0.0)


def daily_sharpe(ret: np.ndarray) -> np.ndarray:
    """규칙별 **비연환산** Sharpe. 퇴화(거래 0 / std 0)는 0 (§5.1 동결).

    분산은 **2-pass**(평균을 뺀 뒤 제곱합)로 계산해 상쇄 오차를 피한다.

    Args:
        ret: (N, T) 일수익률.

    Returns:
        (N,) 일 단위 Sharpe.
    """
    n = ret.shape[1]
    mean = ret.mean(axis=1)
    dev = ret - mean[:, None]
    var = (dev * dev).sum(axis=1) / (n - 1)
    m2 = (ret * ret).sum(axis=1) / n
    return _sharpe_from_var(mean, var, m2)


# ── stationary bootstrap (§6.2-3) ─────────────────────────────────────────
def stationary_bootstrap_indices(
    n_obs: int, n_paths: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis–Romano stationary bootstrap 의 날짜 인덱스 행렬을 만든다.

    블록 길이는 기하분포(평균 `mean_block`)를 따른다. 각 시점에서 확률
    `p = 1/mean_block` 로 새 블록을 열고(균등 무작위 시작), 아니면 직전 인덱스 +1
    (순환)을 잇는다. 반환 행 하나가 **한 경로의 날짜 수열**이며, 이 수열이
    3,390 규칙 **전부에 동일 적용**되어 규칙 간 교차상관이 보존된다 (§6.2-3).

    Args:
        n_obs: 관측 일수 T. n_paths: 경로 수. mean_block: 평균 블록 길이(일).
        rng: 난수 생성기 (seed 고정 호출부 책임).

    Returns:
        (n_paths, n_obs) int64 인덱스 행렬. 각 원소 ∈ [0, n_obs).
    """
    p = 1.0 / mean_block
    starts = rng.integers(0, n_obs, size=(n_paths, n_obs), dtype=np.int64)
    newblk = rng.random((n_paths, n_obs)) < p
    idx = np.empty((n_paths, n_obs), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    prev = idx[:, 0]
    for t in range(1, n_obs):
        cont = prev + 1
        np.mod(cont, n_obs, out=cont)
        prev = np.where(newblk[:, t], starts[:, t], cont)
        idx[:, t] = prev
    return idx


def counts_matrix(idx: np.ndarray, n_obs: int) -> np.ndarray:
    """인덱스 행렬 → 경로 × 날짜 **중복도 행렬** `C` (§6.2 계산 항등식).

    Args:
        idx: (B, T) 인덱스 행렬. n_obs: T.

    Returns:
        (B, T) float64. `C[b, j]` = 경로 b 가 날짜 j 를 뽑은 횟수.
    """
    b = idx.shape[0]
    flat = (idx + (np.arange(b, dtype=np.int64) * n_obs)[:, None]).ravel()
    return np.bincount(flat, minlength=b * n_obs).reshape(b, n_obs).astype(np.float64)


def bootstrap_sharpes(
    rc_t: np.ndarray, rc2_t: np.ndarray, counts: np.ndarray, n_obs: int
) -> np.ndarray:
    """중복도 행렬 항등식으로 (경로 × 규칙) Sharpe 를 계산한다.

    `sum  = C @ Rᵀ`, `sum2 = C @ (R²)ᵀ` → `mean = sum/T`,
    `var = (sum2 − T·mean²)/(T−1)`, `SR = mean/√var` (퇴화 시 0).
    Sharpe 는 표본 **다중집합**의 대칭함수이므로 재표집 계열을 실체화한 결과와
    (실수 연산에서) 정확히 같다.

    **상쇄 오차 주의 (실측으로 발견)**: 거래일이 극소수인 규칙(관측 데이터에 2일짜리
    규칙이 존재)은 그 날짜를 하나도 뽑지 않은 경로에서 재표집 표본이 **상수**가 된다.
    이때 `s2 − T·mean²` 는 유효숫자를 전부 잃고 부호조차 불안정하다. 참값은 분산 0
    이므로 §5.1 퇴화 규칙(`SR := 0`)을 `VAR_ZERO_REL` 상대 문턱으로 강제한다.
    이 처리가 없으면 부동소수점 먼지가 `SR ≈ 1e13` 짜리 가짜 최대통계량을 만들어
    귀무분포를 오염시키고 RC p 값을 1 쪽으로 부풀린다.

    Args:
        rc_t: (T, N) 중심화 수익률의 전치. rc2_t: (T, N) 그 제곱의 전치.
        counts: (B, T) 중복도 행렬. n_obs: T.

    Returns:
        (B, N) float64 일 단위 Sharpe.
    """
    s1 = counts @ rc_t
    s2 = counts @ rc2_t
    mean = s1 / n_obs
    var = (s2 - n_obs * mean * mean) / (n_obs - 1)
    return _sharpe_from_var(mean, var, s2 / n_obs)


def naive_bootstrap_sharpes(rc: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """재표집 계열을 **실체화해서** Sharpe 를 계산한다 (§11.1 테스트 ⑦ 대조군).

    분산은 2-pass 로 계산한다 — 행렬곱 경로(`s2 − T·mean²`)와 **수치적으로 독립**한
    계산이어야 항등식 검증이 의미를 갖는다. 퇴화 판정만 동일한 `_sharpe_from_var`
    를 공유한다(§5.1 규약이지 계산 경로가 아니다).

    느리다 — 검증 전용이며 판정 경로에서는 쓰지 않는다.

    Args:
        rc: (N, T) 중심화 수익률. idx: (B, T) 인덱스 행렬.

    Returns:
        (B, N) 일 단위 Sharpe.
    """
    n = idx.shape[1]
    out = np.empty((idx.shape[0], rc.shape[0]), dtype=np.float64)
    for b in range(idx.shape[0]):
        x = rc[:, idx[b]]
        m = x.mean(axis=1)
        dev = x - m[:, None]
        var = (dev * dev).sum(axis=1) / (n - 1)
        out[b] = _sharpe_from_var(m, var, (x * x).sum(axis=1) / n)
    return out


def null_centered(ret: np.ndarray) -> np.ndarray:
    """§6.2-2 귀무 부과 — 각 규칙 열을 평균 0으로 중심화한다.

    상수 계열의 중심화는 **항등적으로 0** 이다. 부동소수점 잔차(1e-18)를 남기면
    재표집 분산이 그 잔차의 제곱이 되어 퇴화 규칙이 가짜 최대통계량을 만든다.

    Args:
        ret: (N, T) 일수익률.

    Returns:
        (N, T) 중심화 수익률 (퇴화 행은 정확히 0).
    """
    rc = ret - ret.mean(axis=1, keepdims=True)
    rc[degenerate_mask(ret)] = 0.0
    return rc


# §11.1 ⑦ 허용 오차. 판정 통계량(일 Sharpe)의 관심 규모는 1e-2 이므로
# 절대 1e-12 는 결정 임계보다 10자리 작다. 상대 항은 큰 값에만 의미가 있다.
IDENT_ATOL: float = 1e-12
IDENT_RTOL: float = 1e-10


def identity_report(fast: np.ndarray, slow: np.ndarray) -> dict[str, Any]:
    """§11.1 ⑦ 항등식 검증 결과를 정직하게 보고한다.

    "비트 수준 일치" 는 부동소수점 결합법칙 위반 때문에 어떤 구현으로도 달성할 수
    없다 (BLAS 블록 합 vs 재표집 계열 순차 합). 따라서 (a) `atol + rtol·|slow|`
    합격 여부, (b) 실측 최대 절대·상대 오차, (c) 정확 비트 일치 비율을 전부 남긴다.
    **순수 상대오차는 통계량이 0 근방일 때 발산하므로 단독 기준이 될 수 없다** —
    실측에서 최대 절대오차 1e-13 인데도 상대오차가 1.9 까지 오르는 항이 존재한다.

    Args:
        fast: 행렬곱 경로 결과. slow: 재표집 실체화 경로 결과.

    Returns:
        보고 딕셔너리 (`within_tolerance` 가 합격 여부).
    """
    diff = np.abs(fast - slow)
    aslow = np.abs(slow)
    ok = bool(np.all(diff <= IDENT_ATOL + IDENT_RTOL * aslow))
    sig = aslow > 1e-6                     # 통계적으로 의미 있는 크기의 항만
    rel_sig = float((diff[sig] / aslow[sig]).max()) if sig.any() else 0.0
    return {
        "within_tolerance": ok, "atol": IDENT_ATOL, "rtol": IDENT_RTOL,
        "max_abs_err": float(diff.max()),
        "max_rel_err_significant": rel_sig,
        "max_rel_err_unconditional": float((diff / np.maximum(aslow, 1e-300)).max()),
        "exact_bit_match_frac": float(np.mean(fast == slow)),
        "note": ("실수 연산에서 정확한 항등식. 잔차는 부동소수점 덧셈 순서 차이이며 "
                 "비트 일치는 원리적으로 달성 불가 — §11.1 ⑦ 문언과의 편차."),
    }


def bootstrap_null_sharpes(
    ret: np.ndarray, n_paths: int = N_PATHS, seed: int = SEED,
    mean_block: float = MEAN_BLOCK_DAYS, chunk: int = CHUNK,
    progress: bool = False,
) -> np.ndarray:
    """귀무(열별 평균 0 중심화) 하 부트스트랩 Sharpe 행렬 (B, N) 을 만든다.

    Args:
        ret: (N, T) 일수익률 (중심화 전 원본). n_paths: 경로 수. seed: 난수 seed.
        mean_block: 평균 블록 길이. chunk: 경로 청크 크기. progress: 진행 로그.

    Returns:
        (n_paths, N) float64.
    """
    n_rules, n_obs = ret.shape
    rc = null_centered(ret)
    rc_t = np.ascontiguousarray(rc.T)
    rc2_t = np.ascontiguousarray((rc * rc).T)
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, n_rules), dtype=np.float64)
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        idx = stationary_bootstrap_indices(n_obs, m, mean_block, rng)
        out[done:done + m] = bootstrap_sharpes(rc_t, rc2_t, counts_matrix(idx, n_obs), n_obs)
        done += m
        if progress:
            logger.info("  부트스트랩 %d/%d 경로", done, n_paths)
    return out


# ── White RC (§6.2) ───────────────────────────────────────────────────────
def reality_check_p(sr_obs: np.ndarray, sr_boot: np.ndarray) -> tuple[float, np.ndarray]:
    """White's Reality Check p 값 (§6.2-5).

    Args:
        sr_obs: (N,) 관측 일 Sharpe. sr_boot: (B, N) 귀무 부트스트랩 Sharpe.

    Returns:
        (p, 경로별 최대통계량 (B,)).
    """
    boot_max = sr_boot.max(axis=1)
    obs_max = float(sr_obs.max())
    p = (1.0 + float(np.sum(boot_max >= obs_max))) / (sr_boot.shape[0] + 1.0)
    return p, boot_max


# ── Romano–Wolf StepM (§6.3) ──────────────────────────────────────────────
def stepm(
    sr_obs: np.ndarray, sr_boot: np.ndarray, alpha: float = FWER, max_steps: int = 100
) -> tuple[list[int], list[dict[str, Any]]]:
    """Romano–Wolf StepM — 생존(기각) **집합**을 정의한다 (§6.3).

    활성 집합의 부트스트랩 최대통계량 `(1−α)` 분위를 임계값으로 삼아 관측 통계량이
    이를 초과하는 규칙을 기각·제거하고, 더 이상 기각이 없을 때까지 반복한다.
    "상위 몇 개" 를 사후에 고르는 것을 금지하기 위한 절차다.

    Args:
        sr_obs: (N,) 관측 통계량. sr_boot: (B, N) 귀무 분포. alpha: FWER.
        max_steps: 안전 상한.

    Returns:
        (기각된 규칙 인덱스 목록, 단계별 기록).
    """
    n = sr_obs.shape[0]
    active = np.ones(n, dtype=bool)
    rejected: list[int] = []
    steps: list[dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        if not active.any():
            break
        blk = sr_boot[:, active]
        maxima = blk.max(axis=1)
        crit = float(np.quantile(maxima, 1.0 - alpha, method="linear"))
        hit = np.flatnonzero(active & (sr_obs > crit))
        steps.append({
            "step": step, "n_active": int(active.sum()), "critical_value_daily": crit,
            "critical_value_ann": crit * ANN,
            "max_obs_active_daily": float(sr_obs[active].max()),
            "max_obs_active_ann": float(sr_obs[active].max()) * ANN,
            "n_rejected_this_step": int(hit.size),
        })
        if hit.size == 0:
            break
        rejected.extend(int(i) for i in hit)
        active[hit] = False
    return rejected, steps


# ── DSR (§6.4) ────────────────────────────────────────────────────────────
def sr0_threshold(var_sr: float, n_trials: int = N_TRIALS) -> float:
    """Bailey–López de Prado 의 기대 최대 Sharpe `SR0` (일 단위).

    Args:
        var_sr: 시행 간 일 Sharpe 의 표본분산(ddof=1). n_trials: 시행 수 N.

    Returns:
        `SR0`.
    """
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(max(var_sr, 0.0)) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(
    sr: float, sr0: float, skew: float, kurt: float, n_obs: int = N_DAYS
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (§6.4).

    `DSR = Φ[ (SR − SR0)·√(T−1) / √(1 − γ3·SR + ((γ4−1)/4)·SR²) ]`.
    모든 Sharpe 는 **비연환산**. `γ4` 는 **비초과** 첨도.

    Args:
        sr: 관측 일 Sharpe. sr0: 기대 최대 Sharpe. skew: γ3. kurt: γ4 (비초과).
        n_obs: T.

    Returns:
        (DSR, 편차 마진 `SR − SR0`).
    """
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0.0:
        return float("nan"), sr - sr0
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z)), sr - sr0


def moments(x: np.ndarray) -> tuple[float, float]:
    """표본 왜도 γ3 와 **비초과** 첨도 γ4 (적률 정의).

    Args:
        x: 1-D 수익률.

    Returns:
        (γ3, γ4). 분산 0 이면 (0, 3).
    """
    d = x - x.mean()
    m2 = float((d * d).mean())
    if m2 <= 0.0:
        return 0.0, 3.0
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    return m3 / m2 ** 1.5, m4 / (m2 * m2)


# ── SPA (§6.7 — 진단 전용, 승격 권한 없음) ────────────────────────────────
def spa_p(sr_obs: np.ndarray, sr_boot: np.ndarray, n_obs: int = N_DAYS) -> dict[str, float]:
    """Hansen(2005) SPA_c p 값 — **진단 전용** (§6.7).

    RC 는 나쁜 모형까지 귀무 최대에 포함해 보수적이다. SPA 는 통계량이
    `−√(2 log log T)` 보다 낮은(= 명백히 열등한) 규칙을 귀무 재중심에서 제외한다.
    본 구현은 studentized 척도(= Sharpe)에서 동일 조작을 수행한다:
    `Z*_{b,k} = √T · (SR*_{b,k} + SR_k · 1{√T·SR_k < −√(2 log log T)})`.

    **어떤 규칙도 승격시킬 수 없다.** "RC 실패 · SPA 통과" 는 실패로 기록한다.

    Args:
        sr_obs: (N,) 관측 일 Sharpe. sr_boot: (B, N) 귀무 부트스트랩. n_obs: T.

    Returns:
        `{"p": …, "threshold_daily_sr": …, "n_excluded": …}`.
    """
    rt = math.sqrt(n_obs)
    thr = -math.sqrt(2.0 * math.log(math.log(n_obs))) / rt
    bad = sr_obs < thr
    adj = np.where(bad, sr_obs, 0.0)
    stat = max(0.0, float(sr_obs.max()) * rt)
    boot = np.maximum(0.0, (sr_boot + adj).max(axis=1) * rt)
    p = (1.0 + float(np.sum(boot >= stat))) / (sr_boot.shape[0] + 1.0)
    return {"p": p, "threshold_daily_sr": thr, "threshold_ann_sr": thr * ANN,
            "n_excluded": int(bad.sum())}


# ── IS/OOS (§7) ───────────────────────────────────────────────────────────
def is_split_index(snap_ts: pd.DatetimeIndex, is_end: pd.Timestamp = IS_END) -> int:
    """IS 구간 길이(일수). 수익률 j 는 `snap_ts[j]` 로 시작하는 하루를 덮는다.

    Args:
        snap_ts: (T+1,) 자정 스냅샷. is_end: IS 종료 시각.

    Returns:
        IS 일수.
    """
    return int(np.searchsorted(snap_ts[:-1], is_end, side="right"))


def ann_sharpe_block(block: np.ndarray) -> np.ndarray:
    """부분 구간의 연환산 Sharpe (퇴화 0).

    Args:
        block: (N, t) 일수익률.

    Returns:
        (N,) 연환산 Sharpe.
    """
    if block.shape[1] < 2:
        return np.zeros(block.shape[0])
    return daily_sharpe(block) * ANN


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """상수 입력에서 경고 대신 NaN 을 주는 Spearman 순위상관."""
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


# ── 분포 보고 (§5.2) ──────────────────────────────────────────────────────
QUANTILES: tuple[float, ...] = (0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0,
                                75.0, 90.0, 95.0, 99.0, 99.9, 100.0)
_BIG: float = 1.0e9   # ±무한대 대용 (np.histogram 이 무한 경계에서 불안정)
HIST_EDGES: tuple[float, ...] = (-_BIG, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
                                 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, _BIG)


def distribution(sr_ann: np.ndarray) -> dict[str, Any]:
    """연환산 Sharpe 분포 요약 (분위수 + 고정 구간 히스토그램).

    Args:
        sr_ann: (N,) 연환산 Sharpe.

    Returns:
        분위수·히스토그램·기술통계 딕셔너리.
    """
    qs = {f"p{q:g}": float(np.percentile(sr_ann, q)) for q in QUANTILES}
    edges = np.array(HIST_EDGES)
    counts = np.histogram(sr_ann, bins=edges)[0]
    hist = [{"lo": (-np.inf if edges[i] == -_BIG else float(edges[i])),
             "hi": (np.inf if edges[i + 1] == _BIG else float(edges[i + 1])),
             "count": int(counts[i]),
             "pct": round(100.0 * counts[i] / sr_ann.size, 3)}
            for i in range(len(counts))]
    return {
        "n": int(sr_ann.size), "mean": float(sr_ann.mean()),
        "std": float(sr_ann.std(ddof=1)), "quantiles": qs, "histogram": hist,
        "n_positive": int((sr_ann > 0).sum()), "n_zero": int((sr_ann == 0).sum()),
        "n_gt_1": int((sr_ann > 1.0).sum()), "n_gt_1p3": int((sr_ann > 1.3).sum()),
        "n_gt_1p9": int((sr_ann > 1.9).sum()),
    }


def power_diagnostic(
    ret: np.ndarray, sr_d: np.ndarray, boot: np.ndarray, n_trades: np.ndarray | None
) -> dict[str, Any]:
    """검정력 진단 — **판정 권한 없음** (§9.2-12 하위표본 구제 금지).

    거래가 극소수인 규칙은 귀무 하에서 병리적으로 큰 Sharpe 를 만든다. 중심화된
    계열이 거의 상수(+mean 만큼의 균일 '이익')이고 표본 표준편차가 미미하기 때문에,
    희소 거래일을 한 번만 뽑은 경로에서 일 Sharpe 가 1.6(연환산 31) 까지 오른다.
    RC 는 3,390 규칙의 **최댓값**을 보므로 이런 규칙 몇 개가 임계값 전체를 지배한다.

    이 함수는 그 지배력을 정량화한다. **여기서 계산한 어떤 부분집합 p 값도 규칙을
    승격시킬 수 없다** — 부분집합 선택은 그 자체가 사후선택이다. 목적은 오직
    "공식 불통과가 병리 때문인가, 아니면 부분집합을 봐도 불통과인가" 를 밝혀
    §9.3 이 요구하는 정직한 실패 서술을 가능하게 하는 것이다.

    Args:
        ret: (N, T) 일수익률. sr_d: (N,) 관측 일 Sharpe.
        boot: (B, N) 귀무 부트스트랩 Sharpe. n_trades: (N,) 거래 수 (없으면 None).

    Returns:
        부분집합별 임계값·p 값과 귀무 최대 소유권 분해.
    """
    nz = (ret != 0).sum(axis=1)
    subsets: list[tuple[str, np.ndarray]] = [
        ("all_official", np.ones(sr_d.size, dtype=bool)),
        ("nonzero_days_ge_30", nz >= 30),
        ("nonzero_days_ge_100", nz >= 100),
        ("nonzero_days_ge_250", nz >= 250),
    ]
    if n_trades is not None:
        subsets += [("trades_ge_100", n_trades >= 100), ("trades_ge_500", n_trades >= 500)]
    rows = []
    for name, m in subsets:
        if not m.any():
            continue
        bm = boot[:, m].max(axis=1)
        om = float(sr_d[m].max())
        rows.append({
            "subset": name, "n_rules": int(m.sum()),
            "obs_max_ann": om * ANN,
            "null_median_ann": float(np.median(bm)) * ANN,
            "null_crit95_ann": float(np.quantile(bm, 0.95)) * ANN,
            "rc_p": (1.0 + float((bm >= om).sum())) / (boot.shape[0] + 1.0),
        })
    arg = boot.argmax(axis=1)
    own = []
    for lo, hi in ((0, 5), (5, 30), (30, 100), (100, 500), (500, 10 ** 9)):
        c = int(((nz[arg] >= lo) & (nz[arg] < hi)).sum())
        own.append({"nonzero_days_lo": lo, "nonzero_days_hi": hi, "paths": c,
                    "pct": round(100.0 * c / boot.shape[0], 2)})
    return {"authority": "none — 진단 전용 (§9.2-12)", "subsets": rows,
            "null_max_ownership_by_active_days": own,
            "n_rules_lt_30_active_days": int((nz < 30).sum())}


def dsr_sensitivity(sr: float, skew: float, kurt: float,
                    n_obs: int = N_DAYS) -> list[dict[str, Any]]:
    """DSR 의 `SR0` 민감도 — **진단 전용**.

    §6.6 은 시행 간 Sharpe 표준편차를 연환산 0.19~0.38 로 가정해 `SR0 ≈ 0.69~1.37`
    을 예상했다. 실제 표준편차가 그보다 크면 `SR0` 가 비례해 커진다. 최선의 가정
    에서조차 DSR 이 0.95 에 못 미치는지 확인한다 (불통과의 강건성).

    Args:
        sr: 관측 일 Sharpe. skew: γ3. kurt: γ4. n_obs: T.

    Returns:
        가정 표준편차별 `SR0`·DSR 목록.
    """
    out = []
    for std_ann in (0.19, 0.38, 0.75, 1.0, 1.6657):
        s0 = sr0_threshold((std_ann / ANN) ** 2)
        dsr, margin = deflated_sharpe(sr, s0, skew, kurt, n_obs=n_obs)
        out.append({"assumed_std_sr_ann": std_ann, "sr0_ann": s0 * ANN,
                    "dsr": dsr, "margin_daily": margin})
    return out


def _env() -> dict[str, Any]:
    """실행 환경 기록 (§9.4). 규정 환경과의 일치 여부를 함께 남긴다."""
    import platform
    machine = platform.machine()
    pyver = ".".join(sys.version.split()[0].split(".")[:2])
    ok = (sys.platform.startswith(PREREG_ENV["platform"])
          and machine == PREREG_ENV["machine"] and pyver == PREREG_ENV["python"])
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
            "platform": sys.platform, "machine": machine,
            "prereg_env": PREREG_ENV, "matches_prereg": bool(ok)}


# ── 판정 ──────────────────────────────────────────────────────────────────
def run_verdict(
    data: SweepData, n_paths: int = N_PATHS, seed: int = SEED,
    identity_paths: int = 64,
) -> dict[str, Any]:
    """§6 전 절차를 1회 수행하고 판정 딕셔너리를 반환한다.

    Args:
        data: 엔진 산출물. n_paths: 부트스트랩 경로 수. seed: seed.
        identity_paths: 계산 항등식 검증(§11.1 ⑦)에 쓸 경로 수 (0 이면 생략).

    Returns:
        판정·진단 전체를 담은 직렬화 가능 딕셔너리.
    """
    ret = data.returns
    rid = data.rule_ids
    n_obs = ret.shape[1]
    sr_d = daily_sharpe(ret)
    sr_a = sr_d * ANN

    # 엔진 요약 CSV 와의 교차검증 — 두 독립 경로가 같은 1차 지표를 내야 한다
    xcheck: dict[str, Any] = {"checked": False}
    if data.summary is not None:
        pos0 = {str(r): i for i, r in enumerate(rid)}
        eng = np.array([float(x) for x in data.summary["sharpe_ann"]])
        ours = np.array([sr_a[pos0[str(r)]] for r in data.summary["rule_id"]])
        xcheck = {"checked": True, "max_abs_diff": float(np.max(np.abs(eng - ours))),
                  "n_mismatch_1e9": int(np.sum(np.abs(eng - ours) > 1e-9))}
        if xcheck["max_abs_diff"] > 1e-9:
            raise SystemExit(f"엔진 요약과 1차 지표 불일치: {xcheck['max_abs_diff']:g}")
    if np.max(np.abs(sr_a)) > 100.0:
        raise SystemExit("비현실적 Sharpe 탐지 — 퇴화 처리 결함 의심, fail-closed")

    logger.info("부트스트랩 (전 기간 T=%d, 경로 %d) 시작", n_obs, n_paths)
    boot = bootstrap_null_sharpes(ret, n_paths=n_paths, seed=seed, progress=True)

    # §11.1 ⑦ 계산 항등식 검증 (판정 전에 수행 — 실패 시 결과 신뢰 불가)
    ident: dict[str, Any] = {"checked": False}
    if identity_paths > 0:
        rng = np.random.default_rng(seed)
        idx = stationary_bootstrap_indices(n_obs, identity_paths, MEAN_BLOCK_DAYS, rng)
        rc = null_centered(ret)
        fast = bootstrap_sharpes(np.ascontiguousarray(rc.T),
                                 np.ascontiguousarray((rc * rc).T),
                                 counts_matrix(idx, n_obs), n_obs)
        slow = naive_bootstrap_sharpes(rc, idx)
        ident = identity_report(fast, slow)
        ident.update({"checked": True, "paths": identity_paths,
                      "var_zero_rel": VAR_ZERO_REL})
        if not ident["within_tolerance"]:
            raise SystemExit(f"계산 항등식 검증 실패: {ident}")

    rc_p, boot_max = reality_check_p(sr_d, boot)
    rejected, steps = stepm(sr_d, boot, alpha=FWER)
    spa = spa_p(sr_d, boot, n_obs=n_obs)
    ntr = None
    if data.summary is not None:
        pos1 = {str(r): i for i, r in enumerate(rid)}
        ntr = np.zeros(len(rid), dtype=np.int64)
        for r, t in zip(data.summary["rule_id"], data.summary["n_trades"], strict=True):
            ntr[pos1[str(r)]] = int(t)
    power = power_diagnostic(ret, sr_d, boot, ntr)

    var_sr = float(np.var(sr_d, ddof=1))
    sr0 = sr0_threshold(var_sr)

    def _dsr_row(i: int) -> dict[str, Any]:
        g3, g4 = moments(ret[i])
        d, margin = deflated_sharpe(float(sr_d[i]), sr0, g3, g4, n_obs=n_obs)
        return {"rule_id": str(rid[i]), "index": int(i),
                "sharpe_ann": float(sr_a[i]), "sharpe_daily": float(sr_d[i]),
                "skew": g3, "kurtosis": g4, "dsr": d, "margin_sr_minus_sr0": margin}

    order = np.argsort(-sr_d, kind="stable")
    top_idx = order[:20].tolist()
    bot_idx = order[-10:].tolist()
    best = int(order[0])

    # IS/OOS (§7)
    split = is_split_index(data.snap_ts)
    is_sr = ann_sharpe_block(ret[:, :split])
    oos_sr = ann_sharpe_block(ret[:, split:])
    is_order = np.argsort(-is_sr, kind="stable")
    oos_order = np.argsort(-oos_sr, kind="stable")
    oos_rank = np.empty(oos_order.size, dtype=np.int64)
    oos_rank[oos_order] = np.arange(1, oos_order.size + 1)
    rho_all = _spearman(is_sr, oos_sr)
    top_tbl = []
    for r, i in enumerate(is_order[:20], start=1):
        top_tbl.append({"is_rank": r, "rule_id": str(rid[i]), "index": int(i),
                        "is_sharpe_ann": float(is_sr[i]), "oos_sharpe_ann": float(oos_sr[i]),
                        "full_sharpe_ann": float(sr_a[i]),
                        "oos_rank": int(oos_rank[i])})
    k100 = is_order[:100]
    isoos = {
        "is_days": int(split), "oos_days": int(n_obs - split),
        "spearman_all": rho_all,
        "spearman_top100_is": _spearman(is_sr[k100], oos_sr[k100]),
        "top20_is": top_tbl,
        "is_top20_mean_is_sharpe": float(is_sr[is_order[:20]].mean()),
        "is_top20_mean_oos_sharpe": float(oos_sr[is_order[:20]].mean()),
        "is_top20_n_oos_positive": int((oos_sr[is_order[:20]] > 0).sum()),
        "all_mean_is_sharpe": float(is_sr.mean()),
        "all_mean_oos_sharpe": float(oos_sr.mean()),
        "is_distribution": distribution(is_sr),
        "oos_distribution": distribution(oos_sr),
    }

    # IS 전용 RC (진단 전용, 권한 없음 — §7)
    del boot          # (B, N) 271MB 해제 — 이후 절차는 참조하지 않는다
    logger.info("IS 전용 RC (진단) 시작 T=%d", split)
    boot_is = bootstrap_null_sharpes(ret[:, :split], n_paths=n_paths, seed=seed)
    sr_is_d = daily_sharpe(ret[:, :split])
    is_p, _ = reality_check_p(sr_is_d, boot_is)
    del boot_is

    # 계열/타임프레임/방향 분해 (보조 — §5.2)
    breakdown: dict[str, Any] = {}
    if data.summary is not None:
        s = data.summary.copy()
        pos = {str(r): i for i, r in enumerate(rid)}
        s["_sr"] = [float(sr_a[pos[str(r)]]) for r in s["rule_id"]]
        for key in ("family", "tf", "direction"):
            g = s.groupby(key)["_sr"]
            breakdown[key] = {str(k): {"n": int(v.size), "mean": float(v.mean()),
                                       "median": float(v.median()), "max": float(v.max()),
                                       "min": float(v.min())}
                              for k, v in g}
        breakdown["n_trades"] = {
            "median": float(s["n_trades"].median()), "min": int(s["n_trades"].min()),
            "max": int(s["n_trades"].max()), "n_zero_trade_rules": int((s["n_trades"] == 0).sum()),
        }

    survivors = [_dsr_row(i) for i in sorted(rejected, key=lambda i: -sr_d[i])]
    passed = [r for r in survivors if r["dsr"] >= DSR_MIN]
    verdict_pass = bool(rc_p < RC_ALPHA and len(rejected) > 0 and len(passed) > 0)

    return {
        "spec": "SWEEP-2026-08-31",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_meta": data.meta,
        "env": _env(),
        "bootstrap": {"n_paths": n_paths, "seed": seed, "mean_block_days": MEAN_BLOCK_DAYS,
                      "rng": "numpy.random.default_rng (PCG64)",
                      "statistic": "daily Sharpe (studentized)",
                      "identity_check": ident},
        "engine_crosscheck": xcheck,
        "distribution_full": distribution(sr_a),
        "best": {**_dsr_row(best), "rank": 1},
        "top20": [{**_dsr_row(i), "rank": r} for r, i in enumerate(top_idx, start=1)],
        "bottom10": [_dsr_row(i) for i in bot_idx[::-1]],
        "reality_check": {
            "p": rc_p, "alpha": RC_ALPHA,
            "obs_max_daily": float(sr_d.max()), "obs_max_ann": float(sr_a.max()),
            "null_max_quantiles_ann": {f"p{q:g}": float(np.percentile(boot_max, q) * ANN)
                                       for q in (50, 90, 95, 99, 99.9, 100)},
            "critical_ann_at_95": float(np.quantile(boot_max, 0.95) * ANN),
            "reject": bool(rc_p < RC_ALPHA),
        },
        "stepm": {"fwer": FWER, "n_rejected": len(rejected), "steps": steps,
                  "rejected_rule_ids": [str(rid[i]) for i in rejected]},
        "dsr": {"n_trials": N_TRIALS, "var_sr_daily": var_sr,
                "std_sr_daily": math.sqrt(var_sr), "std_sr_ann": math.sqrt(var_sr) * ANN,
                "sr0_daily": sr0, "sr0_ann": sr0 * ANN, "dsr_min": DSR_MIN,
                "z1": float(stats.norm.ppf(1.0 - 1.0 / N_TRIALS)),
                "z2": float(stats.norm.ppf(1.0 - 1.0 / (N_TRIALS * math.e))),
                "survivors": survivors},
        "power_diagnostic": power,
        "dsr_sensitivity": dsr_sensitivity(float(sr_d[best]), *moments(ret[best]),
                                           n_obs=n_obs),
        "spa_diagnostic": spa,
        "is_rc_diagnostic": {"p": is_p, "is_days": int(split),
                             "obs_max_ann": float(sr_is_d.max() * ANN)},
        "is_oos": isoos,
        "breakdown": breakdown,
        "verdict": {
            "pass": verdict_pass,
            "criteria": "RC p < 0.05 AND StepM 기각 AND DSR >= 0.95",
            "rc_pass": bool(rc_p < RC_ALPHA),
            "stepm_pass": bool(len(rejected) > 0),
            "dsr_pass": bool(len(passed) > 0),
            "n_passing_rules": len(passed),
            "passing_rule_ids": [r["rule_id"] for r in passed],
            "statement": (
                "공동 null 상단 초과 — 엣지 입증 아님, 전방 확인 필요"
                if verdict_pass else
                "3,390 시행 최대 Sharpe 가 zero-edge 공동 null 과 구별되지 않음"
            ),
        },
    }


def _fmt_report(v: dict[str, Any]) -> str:
    """판정 딕셔너리를 사람이 읽는 보고서 문자열로 만든다."""
    out: list[str] = []
    a = out.append
    d = v["distribution_full"]
    a("=" * 78)
    a("SWEEP-2026-08-31 판정 보고 (연환산 Sharpe, 비용·펀딩 차감 후)")
    a("=" * 78)
    a(f"\n[1] 전 규칙 성과 분포  N = {d['n']}  (평균 {d['mean']:+.4f} · 표준편차 {d['std']:.4f})")
    a("  분위수:")
    for k, val in d["quantiles"].items():
        a(f"    {k:>7} : {val:+.4f}")
    a("  히스토그램:")
    for h in d["histogram"]:
        lo = "-inf" if h["lo"] == float("-inf") else f"{h['lo']:+.2f}"
        hi = "+inf" if h["hi"] == float("inf") else f"{h['hi']:+.2f}"
        bar = "#" * max(0, int(h["pct"] / 2))
        a(f"    [{lo:>5}, {hi:>5}) {h['count']:>5}  {h['pct']:>6.2f}%  {bar}")
    a(f"  SR>0 {d['n_positive']} · SR==0 {d['n_zero']} · SR>1.0 {d['n_gt_1']} · "
      f"SR>1.3 {d['n_gt_1p3']} · SR>1.9 {d['n_gt_1p9']}")
    b = v["best"]
    a(f"\n[2] 최고 규칙: {b['rule_id']}")
    a(f"    연환산 Sharpe {b['sharpe_ann']:+.4f} (일 {b['sharpe_daily']:+.6f}) · "
      f"왜도 {b['skew']:+.3f} · 첨도 {b['kurtosis']:.2f}")
    a(f"    DSR = {b['dsr']:.6f}  (마진 SR−SR0 = {b['margin_sr_minus_sr0']:+.6f} 일 단위)")
    r = v["reality_check"]
    a(f"    RC p = {r['p']:.4f}  (귀무 최대 95% 임계 연환산 {r['critical_ann_at_95']:+.4f})")
    a("\n[3] IS 상위 20 → OOS")
    a("    rank  rule_id                                     IS_SR     OOS_SR   OOS_rank")
    for row in v["is_oos"]["top20_is"]:
        a(f"    {row['is_rank']:>4}  {row['rule_id'][:42]:<42} {row['is_sharpe_ann']:>+7.3f}  "
          f"{row['oos_sharpe_ann']:>+7.3f}   {row['oos_rank']:>6}")
    io = v["is_oos"]
    a(f"    IS 상위20 평균: IS {io['is_top20_mean_is_sharpe']:+.3f} → "
      f"OOS {io['is_top20_mean_oos_sharpe']:+.3f} "
      f"(OOS 양수 {io['is_top20_n_oos_positive']}/20)")
    a(f"    순위 상관 Spearman(IS, OOS): 전체 {io['spearman_all']:+.4f} · "
      f"IS상위100 내 {io['spearman_top100_is']:+.4f}")
    a("\n[4] 게이트 상세")
    st = v["stepm"]["steps"][0]
    a(f"    RC        : p = {r['p']:.4f} (기준 <0.05) · 귀무 최대 중앙값 "
      f"{r['null_max_quantiles_ann']['p50']:+.3f} · 95% 임계 {r['critical_ann_at_95']:+.3f} 연환산")
    a(f"    StepM     : 1단계 임계 {st['critical_value_ann']:+.3f} vs 관측 최대 "
      f"{st['max_obs_active_ann']:+.3f} → 기각 {st['n_rejected_this_step']}개")
    dd = v["dsr"]
    a(f"    DSR       : SR0 = {dd['sr0_ann']:+.3f} 연환산 (시행간 Sharpe 표준편차 "
      f"{dd['std_sr_ann']:.3f}) · 최고 규칙 DSR {b['dsr']:.4f} (기준 >=0.95)")
    sp = v["spa_diagnostic"]
    a(f"    SPA(진단) : p = {sp['p']:.4f} · 열등 제외 {sp['n_excluded']}개 "
      f"(승격 권한 없음 — RC 실패는 실패)")
    ir = v["is_rc_diagnostic"]
    a(f"    IS RC(진단): p = {ir['p']:.4f}")
    pw = v["power_diagnostic"]
    a(f"\n[5] 검정력 진단 (권한 없음 — {pw['authority']})")
    a(f"    거래일 30일 미만 규칙 {pw['n_rules_lt_30_active_days']}개가 귀무 최대를 지배한다.")
    a("    부분집합                   n     관측최대   귀무중앙   95%임계     RC p")
    for row in pw["subsets"]:
        a(f"    {row['subset']:<22} {row['n_rules']:>5} {row['obs_max_ann']:>+9.3f} "
          f"{row['null_median_ann']:>+10.3f} {row['null_crit95_ann']:>+10.3f} {row['rc_p']:>8.4f}")
    a("    귀무 최대를 차지한 규칙의 거래일수:")
    for o in pw["null_max_ownership_by_active_days"]:
        hi = "inf" if o["nonzero_days_hi"] >= 10 ** 9 else str(o["nonzero_days_hi"])
        a(f"      [{o['nonzero_days_lo']:>4}, {hi:>5}) : {o['paths']:>6} 경로 ({o['pct']:>5.2f}%)")
    a("\n    DSR 민감도 (최고 규칙) — SR0 가정별:")
    for row in v["dsr_sensitivity"]:
        a(f"      시행간 SR 표준편차 {row['assumed_std_sr_ann']:>6.3f} → "
          f"SR0 {row['sr0_ann']:>+6.3f} 연환산 · DSR {row['dsr']:.4f}")
    vd = v["verdict"]
    a(f"\n[6] 판정: {'통과' if vd['pass'] else '실패'} — {vd['statement']}")
    a(f"    RC p<0.05 {vd['rc_pass']} · StepM 기각 {vd['stepm_pass']} "
      f"({v['stepm']['n_rejected']}개) · DSR>=0.95 {vd['dsr_pass']}")
    a(f"    전방 페이퍼 후보: {vd['passing_rule_ids'] or '없음 (§8-7 편입 없음)'}")
    if not v["env"]["matches_prereg"]:
        a(f"\n    [공시] 실행 환경 {v['env']['platform']}/{v['env']['machine']} 은 "
          f"§9.4 규정({PREREG_ENV['platform']}/{PREREG_ENV['machine']})과 다르다.")
    a("=" * 78)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 판정 1회 수행 후 JSON·보고서를 기록한다."""
    ap = argparse.ArgumentParser(description="SWEEP-2026-08-31 판정 (RC · StepM · DSR · SPA)")
    ap.add_argument("--indir", default="logs", help="엔진 산출물 디렉터리")
    ap.add_argument("--outdir", default="logs", help="판정 산출물 디렉터리")
    ap.add_argument("--paths", type=int, default=N_PATHS, help="부트스트랩 경로 수 (동결 10000)")
    ap.add_argument("--identity-paths", type=int, default=64,
                    help="§11.1 ⑦ 항등식 검증 경로 수 (0=생략)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ind, outd = Path(args.indir), Path(args.outdir)
    data = load_sweep(ind / "sweep_returns.npz", ind / "sweep_summary.csv")
    logger.info("입력: %s (%s) · 엔진 %s", ind / "sweep_returns.npz", data.returns.shape,
                data.meta.get("engine_sha256", "?")[:16])
    v = run_verdict(data, n_paths=args.paths, identity_paths=args.identity_paths)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "sweep_verdict.json").write_text(json.dumps(v, ensure_ascii=False, indent=2))
    rep = _fmt_report(v)
    (outd / "sweep_verdict_report.txt").write_text(rep + "\n")
    print(rep)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
