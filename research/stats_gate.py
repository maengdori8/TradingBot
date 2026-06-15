"""통계 게이트 — 다중검정/데이터스누핑 방어의 단일 진실원천.

자율 연구 루프(research/ledger/GOAL_LOOP_PROMPT.md §5-6)가 어떤 가설을 PROMOTE 하기
전에 반드시 통과해야 하는 통계 검정을 한 곳에 모은다. 사이클마다 재구현/암산하지 않고
이 모듈의 `gate()` 한 함수만 호출한다.

핵심 방어:
1. **클러스터 실효표본**: 신호가 1만 개라도 같은 날/주에 몰리면 독립 표본이 아니다.
   거래일/주 블록으로 묶어 n_eff를 추정하고, 표준오차를 그에 맞춰 키운다(클러스터-로버스트).
2. **다중검정 보정**: N개의 가설을 시도하면 그중 하나가 우연히 좋아 보일 확률이 커진다.
   - 본페로니: 보정 p < α/N 이어야 유의.
   - Deflated Sharpe Ratio(Bailey & López de Prado): N회 시도 시 '순수 노이즈가 만들
     기대 최대 Sharpe(SR0)'를 넘는지를 비정규성(왜도·첨도)까지 보정해 확률로 판정.
3. **블록 부트스트랩 CI**: 거래일 블록을 복원추출해 평균 R의 신뢰구간을 만든다. 하한이
   0 이하이면 유의하지 않음.

문턱은 사이클 독립적으로 코드 상수에 고정한다(루프 도중 변경 = 거짓 승리, GOAL_LOOP §5-6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# 고정 문턱 (사이클 독립 — 절대 런타임에 바꾸지 말 것)
# ──────────────────────────────────────────────────────────────────────
BASE_BAR_R = 0.05      # 실효표본 ~700 클러스터 SE 채택금지선 (docs/RESEARCH_2026-06.md 36행)
K_LOG = 0.02           # 시도수 N에 따른 문턱 상승 계수
MIN_N_EFF = 50         # 실효표본 하한 (미만이면 자동 REJECT)
DSR_PROB_MIN = 0.95    # Deflated Sharpe 확률 하한 (참 Sharpe > SR0 확신도)
ALPHA = 0.05           # 유의수준 (본페로니 전 기준)
N_BOOT = 2000          # 블록 부트스트랩 반복수
EULER_MASCHERONI = 0.5772156649015329


# ──────────────────────────────────────────────────────────────────────
# 정규분포 (scipy 의존 제거 — 순수 구현)
# ──────────────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """표준정규 CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """표준정규 역CDF Φ⁻¹(p) — Acklam 유리근사 (|오차| < 1.15e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ──────────────────────────────────────────────────────────────────────
# 실효표본 (클러스터)
# ──────────────────────────────────────────────────────────────────────

def _day_keys(ts: np.ndarray) -> np.ndarray:
    """타임스탬프 배열 → UTC 날짜(거래일) 키."""
    t = pd.to_datetime(ts, utc=True)
    return t.normalize().view("int64")


def effective_sample(ts: np.ndarray) -> dict:
    """거래일/주 클러스터 기반 실효표본 수."""
    if len(ts) == 0:
        return {"n_trades": 0, "n_days": 0, "n_weeks": 0}
    t = pd.to_datetime(ts, utc=True)
    n_days = int(t.normalize().nunique())
    iso = t.isocalendar()
    n_weeks = int(pd.Series(list(zip(iso["year"], iso["week"]))).nunique())
    return {"n_trades": int(len(ts)), "n_days": n_days, "n_weeks": n_weeks}


# ──────────────────────────────────────────────────────────────────────
# Sharpe / Deflated Sharpe
# ──────────────────────────────────────────────────────────────────────

def per_obs_sharpe(r: np.ndarray) -> float:
    """관측당(거래당) Sharpe = mean/std (무위험 0 가정)."""
    if len(r) < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r)) / sd if sd > 0 else 0.0


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """N회 독립 시도 시 순수 노이즈가 만드는 기대 최대 Sharpe(SR0).

    Bailey & López de Prod (2014)의 순서통계 근사:
        SR0 = sr_std · [ (1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e)) ]
    """
    n = max(int(n_trials), 1)
    if n == 1 or sr_std <= 0:
        return 0.0
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
    return sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float, n_eff: int) -> float:
    """PSR: 참 Sharpe가 sr_benchmark를 초과할 확률 (왜도·첨도 보정).

    PSR(SR*) = Φ[ (SR̂ - SR*)·√(n_eff-1) / √(1 - g₃·SR̂ + (g₄-1)/4·SR̂²) ]
    g₃=왜도, g₄=첨도(정규=3). n_eff는 클러스터 실효표본(독립성 보정).
    """
    if n_eff < 2 or len(r) < 2:
        return 0.0
    sr = per_obs_sharpe(r)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurtosis()) + 3.0  # pandas kurtosis는 초과첨도 → +3
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        denom = 1e-9
    z = (sr - sr_benchmark) * math.sqrt(max(n_eff - 1, 1)) / math.sqrt(denom)
    return norm_cdf(z)


def deflated_sharpe(r: np.ndarray, n_trials: int, n_eff: int,
                    sr_std: float | None = None) -> dict:
    """Deflated Sharpe Ratio: SR0(다중검정) 대비 PSR.

    sr_std 미지정 시 Sharpe 추정량의 표준오차로 보수 추정:
        se(SR) ≈ √((1 + ½·SR²)/n_eff)  → 시도 간 Sharpe 분산의 대용.
    """
    sr = per_obs_sharpe(r)
    if sr_std is None:
        sr_std = math.sqrt((1.0 + 0.5 * sr * sr) / max(n_eff, 2))
    sr0 = expected_max_sharpe(n_trials, sr_std)
    dsr_prob = probabilistic_sharpe(r, sr0, n_eff)
    return {
        "sharpe_per_trade": round(sr, 4),
        "sr_std_used": round(sr_std, 4),
        "sr0_noise_ceiling": round(sr0, 4),
        "sharpe_excess_over_sr0": round(sr - sr0, 4),
        "dsr_prob": round(dsr_prob, 4),
    }


# ──────────────────────────────────────────────────────────────────────
# 블록 부트스트랩 (거래일 블록 복원추출)
# ──────────────────────────────────────────────────────────────────────

def block_bootstrap_ci(r: np.ndarray, ts: np.ndarray, alpha: float = ALPHA,
                       n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """거래일 블록을 복원추출해 평균 R의 (1-α) 신뢰구간을 만든다."""
    if len(r) < 2:
        return {"mean_r": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_days": 0}
    keys = _day_keys(np.asarray(ts))
    uniq = np.unique(keys)
    by_day = {k: r[keys == k] for k in uniq}
    day_list = list(uniq)
    rng = np.random.default_rng(seed)
    n_days = len(day_list)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_days, size=n_days)
        vals = np.concatenate([by_day[day_list[i]] for i in pick])
        means[b] = vals.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean_r": round(float(r.mean()), 4), "ci_low": round(lo, 4),
            "ci_high": round(hi, 4), "n_days": int(n_days)}


# ──────────────────────────────────────────────────────────────────────
# 본페로니 보정 t-검정 (클러스터-로버스트 SE)
# ──────────────────────────────────────────────────────────────────────

def cluster_t_pvalue(r: np.ndarray, n_eff: int) -> dict:
    """평균 R>0 단측검정 p값을 클러스터 실효표본으로 보수 산출."""
    if n_eff < 2 or len(r) < 2:
        return {"t_stat": 0.0, "p_one_sided": 1.0}
    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    se = sd / math.sqrt(n_eff)        # 거래수가 아니라 실효표본으로 SE 확대
    t = mean / se if se > 0 else 0.0
    p = 1.0 - norm_cdf(t)             # 단측 (mean>0)
    return {"t_stat": round(t, 4), "p_one_sided": round(p, 6)}


# ──────────────────────────────────────────────────────────────────────
# 채택 문턱
# ──────────────────────────────────────────────────────────────────────

def r_min(n_tested: int) -> float:
    """누적 시도수 N에 따른 mean_r 채택 문턱 (시도 많을수록 상승)."""
    n = max(int(n_tested), 1)
    return max(BASE_BAR_R, BASE_BAR_R + K_LOG * math.log(n))


# ──────────────────────────────────────────────────────────────────────
# 종합 게이트
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    mean_r: float = 0.0
    r_min_required: float = 0.0
    n_tested: int = 0
    eff: dict = field(default_factory=dict)
    sharpe: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    ttest: dict = field(default_factory=dict)
    bonferroni_threshold: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def gate(r, ts, n_tested: int, sr_std: float | None = None,
         alpha: float = ALPHA, n_boot: int = N_BOOT) -> GateResult:
    """가설 PROMOTE 가부를 판정한다. 모든 하위 검정을 통과해야 passed=True.

    Args:
        r: 가설의 OOS 거래당 순R 배열
        ts: 각 거래의 타임스탬프 (클러스터링용)
        n_tested: 누적 시도수 N (GOAL_LOOP §5-6 정의)
        sr_std: 시도 간 Sharpe 표준편차(있으면). None이면 보수 추정.

    Returns:
        GateResult — passed 와 각 검정 수치/사유.
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    ts = np.asarray(ts)[: len(r)] if len(ts) >= len(r) else np.asarray(ts)
    reasons: list[str] = []

    if len(r) < 2:
        return GateResult(passed=False, reasons=["표본 부족 (<2)"], n_tested=n_tested)

    eff = effective_sample(ts)
    n_eff = eff["n_days"]
    mean_r = float(r.mean())
    req = r_min(n_tested)
    sh = deflated_sharpe(r, n_tested, n_eff, sr_std)
    bs = block_bootstrap_ci(r, ts, alpha, n_boot)
    tt = cluster_t_pvalue(r, n_eff)
    bonf = alpha / max(int(n_tested), 1)

    # ── 통과 조건 (전부 충족) ──
    if n_eff < MIN_N_EFF:
        reasons.append(f"실효표본 n_eff={n_eff} < {MIN_N_EFF}")
    if mean_r < req:
        reasons.append(f"mean_r={mean_r:+.4f} < 문턱 r_min(N={n_tested})={req:.4f}")
    if sh["sharpe_excess_over_sr0"] <= 0:
        reasons.append(f"Sharpe {sh['sharpe_per_trade']} ≤ 노이즈천장 SR0 {sh['sr0_noise_ceiling']}")
    if sh["dsr_prob"] < DSR_PROB_MIN:
        reasons.append(f"DSR 확률 {sh['dsr_prob']} < {DSR_PROB_MIN}")
    if bs["ci_low"] <= 0:
        reasons.append(f"부트스트랩 CI 하한 {bs['ci_low']} ≤ 0")
    if tt["p_one_sided"] >= bonf:
        reasons.append(f"보정 p={tt['p_one_sided']} ≥ α/N={bonf:.6f}")

    return GateResult(
        passed=len(reasons) == 0,
        reasons=reasons or ["모든 검정 통과"],
        mean_r=round(mean_r, 4), r_min_required=round(req, 4),
        n_tested=int(n_tested), eff=eff, sharpe=sh, bootstrap=bs,
        ttest=tt, bonferroni_threshold=round(bonf, 6),
    )


if __name__ == "__main__":
    # 빠른 수동 점검 (정식 검증은 research/test_stats_gate.py)
    rng = np.random.default_rng(1)
    days = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    ts = np.repeat(days.values, 3)
    noise = rng.normal(0.0, 1.0, size=len(ts))
    print("노이즈:", gate(noise, ts, n_tested=50).passed, "(기대 False)")
    signal = rng.normal(0.25, 1.0, size=len(ts))
    print("강신호:", gate(signal, ts, n_tested=5).passed, "(기대 True)")
    print("강신호+N폭증:", gate(signal, ts, n_tested=100000).passed, "(문턱 상승)")
