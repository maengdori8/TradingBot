"""CONF-TUNE-2026-09-04 판정 — White RC(고정 ω̂) · Romano–Wolf StepM · DSR.

`lab/confluence_tune.py` 가 기록한 일수익률 행렬만 읽는다. 백테스트를 다시 돌리지
않으며 판정만 한다. 부트스트랩 기계는 동결 `lab/sweep_verdict.py` 에서 **읽기 전용
임포트**하고, 통계량은 `lab/avgdown_verdict.py` 가 확정한 White 고정 스케일 형태를
그대로 쓴다.

통계량 (동결)
------------
`T_k = √T · mean(r_k) / ω̂_k`, **ω̂_k 는 원표본 일수익률의 std(ddof=1) 로 1회 추정한
고정 스케일**. 부트스트랩 경로에서는 분자(평균)만 재계산하고 ω̂ 는 절대 재추정하지
않는다 — White(2000)·Hansen(2005) 원형. 경로별 재추정은 희소거래 시행이 귀무분포를
지배해 검정력을 0 으로 만든다 (SWEEP-2026-08-31 §13 사고).

사전 선언 (결과 조회 전 고정)
---------------------------
- **1차 창 = 전체 구간** (2021-01-06 … 2026-08-23, T=2056). RC 판정 권한은 여기에만
  있다. IS/OOS 분해는 **진단 전용, 권한 없음** — 전체 구간 탈락 시행을 OOS 성적으로
  되살리거나, 통과 시행을 OOS 약세로 버리지 않는다 (AVGDOWN §7 계승).
- N = 시행 총계, T = 2056, SEED/N_PATHS/MEAN_BLOCK_DAYS/CHUNK 는 아래 상수.
- 합격 = `RC p < 0.05` AND `StepM 기각 >= 1` AND `DSR >= 0.95` 인 시행 존재.

가설별 대조 (본 명세의 고유 판정 — RC 와 별개로 항상 보고한다)
-----------------------------------------------------------
동일 구조에서 확신도 규칙만 바꾼 **짝지은 비교**를 한다. 짝은 구조 ID 로 정확히
1:1 대응하므로 시장 경로가 공통이며, 차이만 본다.

- `real 개선`  = ramp/x*Ap 계열 − flat
- `위약 개선`  = plcR3 − flat   (크기 분포는 real 과 동일)
- `역전 개선`  = invR − flat

컨플루언스가 정보라면 `real 개선 > 위약 개선` 이고 `역전 개선 < 0` 이어야 한다.
`real 개선 ≈ 위약 개선` 이면 관측된 상승은 **컨플루언스가 아니라 레버리지**다.
짝 차이의 유의성은 같은 정상 부트스트랩 날짜 경로로 재표본한 짝차이 평균의
양측 분포로 판정한다 (시행별 독립 검정이 아니라 계열 평균 하나).

실행:
  python3 -m lab.confluence_verdict --selftest   # 합성 데이터 자가검증
  python3 -m lab.confluence_verdict --run        # 판정 1회 (스윕 산출물 필요)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── 동결 판정 기계 로드 (읽기 전용 — 수정 없음) ───────────────────────────
_SV_SPEC = importlib.util.spec_from_file_location(
    "sweep_verdict_frozen", str(ROOT / "lab" / "sweep_verdict.py"))
sv = importlib.util.module_from_spec(_SV_SPEC)
sys.modules["sweep_verdict_frozen"] = sv
_SV_SPEC.loader.exec_module(sv)

# ── 동결 상수 (변경 금지 — 재현성은 (seed, n_paths, chunk, T) 의 함수다) ───
SPEC = "CONF-TUNE-2026-09-04"
SEED: int = 20260904
N_PATHS: int = 1000
CHUNK: int = 500
MEAN_BLOCK_DAYS: float = 5.0
RC_ALPHA: float = 0.05
FWER: float = 0.05
DSR_MIN: float = 0.95
N_DAYS: int = 2056
ANN: float = math.sqrt(365.0)
VAR_ZERO_REL: float = sv.VAR_ZERO_REL

NPZ = ROOT / "logs/conf_tune_returns.npz"
SUMMARY = ROOT / "logs/conf_tune_summary.csv"
OUT_JSON = ROOT / "logs/conf_tune_verdict.json"
OUT_TXT = ROOT / "logs/conf_tune_verdict_report.txt"


# ── 통계량 (avgdown_verdict 형태 계승) ────────────────────────────────────
def fixed_scale(ret: np.ndarray) -> np.ndarray:
    """ω̂_k — 원표본 1회 추정 고정 스케일 (퇴화 시행은 0).

    퇴화 시행은 N 에 그대로 산입한다 (사후 제거 금지).

    Args:
        ret: (N, T) 일수익률.

    Returns:
        (N,) 표준편차. 퇴화(분산 0·상수열)면 0.
    """
    n = ret.shape[1]
    mean = ret.mean(axis=1)
    dev = ret - mean[:, None]
    var = (dev * dev).sum(axis=1) / (n - 1)
    m2 = (ret * ret).sum(axis=1) / n
    ok = (var > 0.0) & (var > VAR_ZERO_REL * m2) & (np.ptp(ret, axis=1) > 0)
    return np.where(ok, np.sqrt(np.where(ok, var, 1.0)), 0.0)


def obs_stat(ret: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """관측 통계량 `T_k = √T · mean(r_k) / ω̂_k` (퇴화 = 0)."""
    t = ret.shape[1]
    s = math.sqrt(t) * ret.mean(axis=1) / np.where(omega > 0, omega, 1.0)
    return np.where(omega > 0, s, 0.0)


def bootstrap_null_stats(ret: np.ndarray, omega: np.ndarray, seed: int,
                         n_paths: int = N_PATHS, chunk: int = CHUNK,
                         return_idx: bool = False):
    """귀무 분포 (n_paths, N) — 열 중심화 후 중복도 행렬 항등식으로 평균만 재계산.

    ω̂ 는 인자로 받은 고정값만 쓴다 (경로별 재추정 금지).

    Args:
        ret: (N, T) 일수익률. omega: (N,) 고정 스케일. seed: RNG 시드.
        n_paths: 부트스트랩 경로 수. chunk: 경로 청크 (RNG 스트림에 영향 — 동결).
        return_idx: True 면 마지막 청크가 아니라 전체 경로 인덱스도 반환.

    Returns:
        (n_paths, N) 귀무 통계량. `return_idx` 면 `(stats, idx_list)`.
    """
    t = ret.shape[1]
    rc = ret - ret.mean(axis=1, keepdims=True)
    rc[omega <= 0] = 0.0
    rc_t = np.ascontiguousarray(rc.T)
    inv = np.where(omega > 0, math.sqrt(t) / np.where(omega > 0, omega, 1.0), 0.0)
    out = np.empty((n_paths, ret.shape[0]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    idxs: list[np.ndarray] = []
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        idx = sv.stationary_bootstrap_indices(t, m, MEAN_BLOCK_DAYS, rng)
        if return_idx:
            idxs.append(idx)
        cmat = sv.counts_matrix(idx, t)
        out[done:done + m] = (cmat @ rc_t / t) * inv[None, :]
        done += m
    return (out, idxs) if return_idx else out


def stat_block(ret: np.ndarray, tids: np.ndarray, n_trials: int,
               seed: int) -> dict[str, Any]:
    """한 행렬에 대한 RC · StepM · DSR 전체 판정 블록."""
    t = ret.shape[1]
    omega = fixed_scale(ret)
    stat = obs_stat(ret, omega)
    guard = np.abs(stat) > 10.0 * math.sqrt(t)
    if guard.any():
        raise SystemExit(f"비현실적 통계량 {int(guard.sum())}건 — 퇴화 처리 결함 의심")
    boot = bootstrap_null_stats(ret, omega, seed=seed)
    rc_p, boot_max = sv.reality_check_p(stat, boot)
    rejected, steps = sv.stepm(stat, boot, alpha=FWER)

    sr_d = stat / math.sqrt(t)
    var_sr = float(np.var(sr_d, ddof=1))
    sr0 = sv.sr0_threshold(var_sr, n_trials=n_trials)
    survivors = []
    for i in rejected:
        g3, g4 = sv.moments(ret[i])
        dsr, margin = sv.deflated_sharpe(float(sr_d[i]), sr0, g3, g4, n_obs=t)
        survivors.append({"trial_id": str(tids[i]), "stat": float(stat[i]),
                          "sharpe_ann": float(sr_d[i] * ANN), "dsr": float(dsr),
                          "margin": float(margin)})
    passing = [r for r in survivors if (r["dsr"] == r["dsr"]) and r["dsr"] >= DSR_MIN]
    best = int(np.argmax(stat))
    return {
        "n_trials": int(ret.shape[0]), "n_days": int(t),
        "rc_p": float(rc_p), "rc_reject": bool(rc_p < RC_ALPHA),
        "critical_stat_at_95": float(np.quantile(boot_max, 0.95)),
        "obs_max_stat": float(stat[best]),
        "obs_max_sharpe_ann": float(sr_d[best] * ANN),
        "best_trial_id": str(tids[best]),
        "n_degenerate": int((omega <= 0).sum()),
        "stepm_n_rejected": len(rejected),
        "stepm_steps": len(steps),
        "sr0_daily": float(sr0), "var_sr_daily": var_sr,
        "n_passing_dsr": len(passing),
        "survivors": survivors[:50], "passing": passing[:50],
    }


# ── 짝지은 확신도 대조 ────────────────────────────────────────────────────
def paired_contrast(ret: np.ndarray, tids: np.ndarray, base_conv: str,
                    test_conv: str, seed: int) -> dict[str, Any]:
    """`test_conv` 와 `base_conv` 의 짝지은 일수익률 차이를 검정한다.

    구조 ID(확신도 토큰만 제거한 나머지)로 1:1 대응시킨 뒤, 짝차이 계열의
    **구조 평균** 하나를 만들고 같은 정상 부트스트랩으로 양측 p 를 구한다.
    시행별 다중검정이 아니라 계열 하나의 검정이므로 RC 와 독립적으로 읽는다.

    Args:
        ret: (N, T). tids: (N,) 시행 ID. base_conv/test_conv: 확신도 키.
        seed: RNG 시드.

    Returns:
        짝 수·평균 차이·연환산 차이·부트스트랩 양측 p.
    """
    def key(tid: str) -> str:
        return "|".join(p for p in tid.split("|") if not p.startswith("cv="))

    idx_base = {key(str(x)): i for i, x in enumerate(tids) if f"|cv={base_conv}|" in str(x)}
    idx_test = {key(str(x)): i for i, x in enumerate(tids) if f"|cv={test_conv}|" in str(x)}
    common = sorted(set(idx_base) & set(idx_test))
    if not common:
        return {"n_pairs": 0}
    b = ret[[idx_base[k] for k in common]]
    a = ret[[idx_test[k] for k in common]]
    d = (a - b).mean(axis=0)                       # (T,) 구조 평균 짝차이
    t = len(d)
    mu = float(d.mean())
    sd = float(d.std(ddof=1))
    rng = np.random.default_rng(seed)
    idx = sv.stationary_bootstrap_indices(t, N_PATHS, MEAN_BLOCK_DAYS, rng)
    cmat = sv.counts_matrix(idx, t)
    dc = d - mu                                     # 귀무: 평균 0
    boot_mu = cmat @ dc / t
    p_two = float((1.0 + np.sum(np.abs(boot_mu) >= abs(mu))) / (N_PATHS + 1.0))
    eq_a = np.cumprod(1.0 + a, axis=1)[:, -1] - 1.0
    eq_b = np.cumprod(1.0 + b, axis=1)[:, -1] - 1.0
    return {
        "n_pairs": len(common),
        "mean_daily_diff_bp": mu * 1e4,
        "ann_diff_pct": ((1.0 + mu) ** 365 - 1.0) * 100.0,
        "t_like": mu / sd * math.sqrt(t) if sd > 0 else 0.0,
        "p_two_sided": p_two,
        "median_cum_base": float(np.median(eq_b)),
        "median_cum_test": float(np.median(eq_a)),
        "frac_pairs_test_better": float((eq_a > eq_b).mean()),
    }


# ── 실행 ──────────────────────────────────────────────────────────────────
def load() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """스윕 산출물을 읽고 fail-closed 검사한다."""
    z = np.load(NPZ, allow_pickle=True)
    ret = np.ascontiguousarray(z["daily_returns"], dtype=np.float64)
    tids = np.asarray(z["trial_ids"], dtype=object)
    if ret.shape[1] != N_DAYS:
        raise SystemExit(f"일수 {ret.shape[1]} != {N_DAYS} — fail-closed")
    if ret.shape[0] != len(tids):
        raise SystemExit("행렬 행수와 시행 ID 수 불일치 — fail-closed")
    if len(set(tids.tolist())) != len(tids):
        raise SystemExit("시행 ID 유일성 위반 — fail-closed")
    if not np.isfinite(ret).all():
        raise SystemExit("일수익률 NaN/Inf — fail-closed")
    return ret, tids, pd.read_csv(SUMMARY)


def run_verdict() -> int:
    """1차 창 RC 판정 + 확신도 짝대조 + 진단용 IS/OOS 분해."""
    ret, tids, summ = load()
    n = ret.shape[0]
    logger.info("%s — 행렬 %s, 시행 %d", SPEC, ret.shape, n)

    main = stat_block(ret, tids, n_trials=n, seed=SEED)

    convs = sorted({str(x).split("|cv=")[1].split("|")[0] for x in tids})
    contrasts = {c: paired_contrast(ret, tids, "flat", c, SEED)
                 for c in convs if c != "flat"}

    snap = pd.to_datetime(np.load(NPZ, allow_pickle=True)["snap_ts"], utc=True)
    if len(snap) != N_DAYS + 1:
        raise SystemExit(f"snap_ts 길이 {len(snap)} != {N_DAYS + 1} — fail-closed")
    # 수익률 j 는 snap_ts[j] 로 시작하는 하루를 덮는다 (sweep_verdict.is_split_index 규약)
    split = int(np.searchsorted(snap[:-1].values,
                                np.datetime64("2024-12-31T23:59:00"), side="right"))
    diag = {}
    for name, blk in (("IS", ret[:, :split]), ("OOS", ret[:, split:])):
        om = fixed_scale(blk)
        st = obs_stat(blk, om)
        bt = bootstrap_null_stats(blk, om, seed=SEED)
        p, _ = sv.reality_check_p(st, bt)
        diag[name] = {"n_days": int(blk.shape[1]), "rc_p": float(p),
                      "obs_max_sharpe_ann": float(st.max() / math.sqrt(blk.shape[1]) * ANN),
                      "best_trial_id": str(tids[int(np.argmax(st))]),
                      "authority": "none — 진단 전용"}

    verdict_pass = bool(main["rc_reject"] and main["stepm_n_rejected"] > 0
                        and main["n_passing_dsr"] > 0)
    out = {
        "spec": SPEC, "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {"seed": SEED, "n_paths": N_PATHS, "chunk": CHUNK,
                      "mean_block_days": MEAN_BLOCK_DAYS, "T": N_DAYS},
        "criteria": "RC p < 0.05 AND StepM 기각 >= 1 AND DSR >= 0.95",
        "primary_window": "full sample (권한 있음)",
        "main": main, "conviction_contrasts": contrasts,
        "is_oos_diagnostic": diag, "verdict_pass": verdict_pass,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    OUT_TXT.write_text(report(out, summ))
    print(report(out, summ))
    return 0


def report(v: dict, summ: pd.DataFrame) -> str:
    """사람이 읽는 판정 보고서."""
    m = v["main"]
    L = ["=" * 78,
         f"{v['spec']} 판정 (1차 = 전체 구간 3심볼 균등 합산 일수익률, 비용·펀딩 차감 후)",
         "=" * 78,
         f"시행 {m['n_trials']} · 일수 {m['n_days']} · 퇴화 {m['n_degenerate']}",
         f"최고 시행: {m['best_trial_id']}",
         f"  통계량 {m['obs_max_stat']:+.3f} · 연환산 SR {m['obs_max_sharpe_ann']:+.3f}",
         f"  RC p = {m['rc_p']:.4f} (임계 95% = {m['critical_stat_at_95']:+.3f})",
         f"  StepM 기각 {m['stepm_n_rejected']}개 · DSR>=0.95 통과 {m['n_passing_dsr']}개",
         f"판정: {'통과' if v['verdict_pass'] else '실패'} — "
         f"{v['criteria']}",
         "-" * 78,
         "확신도 짝대조 (flat 대비, 구조 1:1 짝지음)",
         f"{'규칙':<8}{'짝수':>6}{'일평균차(bp)':>14}{'연환산차(%)':>13}"
         f"{'양측p':>9}{'중앙누적(기준→시험)':>24}{'개선비율':>9}"]
    for k, c in v["conviction_contrasts"].items():
        if not c.get("n_pairs"):
            continue
        L.append(f"{k:<8}{c['n_pairs']:>6}{c['mean_daily_diff_bp']:>14.4f}"
                 f"{c['ann_diff_pct']:>13.3f}{c['p_two_sided']:>9.4f}"
                 f"{c['median_cum_base']:>11.3f} → {c['median_cum_test']:<10.3f}"
                 f"{c['frac_pairs_test_better']:>9.3f}")
    L += ["-" * 78, "IS/OOS 분해 (진단 전용 · 판정 권한 없음)"]
    for k, d in v["is_oos_diagnostic"].items():
        L.append(f"  {k}: 일수 {d['n_days']} · RC p {d['rc_p']:.4f} · "
                 f"최대 연환산 SR {d['obs_max_sharpe_ann']:+.3f}")
    L.append("=" * 78)
    return "\n".join(L)


def selftest() -> None:
    """합성 데이터 자가검증.

    1. 순수 잡음 행렬에서 RC p 가 크게 나온다 (위양성 방어).
    2. 경로별 스케일 재추정 구성과 결과가 다르다 (오지정 탐지).
    3. 짝대조가 크기 차이만 있는 인공 짝에서 0 이 아닌 차이를 잡아낸다.

    Raises:
        AssertionError: 불변식 위반 시.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(1)
    n, t = 300, N_DAYS
    ret = rng.normal(0.0, 0.01, size=(n, t))
    om = fixed_scale(ret)
    st = obs_stat(ret, om)
    boot = bootstrap_null_stats(ret, om, seed=SEED, n_paths=200)
    p, _ = sv.reality_check_p(st, boot)
    assert p > 0.01, f"(1) 잡음에서 RC p={p:.4f} — 위양성 의심"
    logger.info("(1) OK — 순수 잡음 RC p=%.4f", p)

    restud = np.empty_like(boot)
    rc = ret - ret.mean(axis=1, keepdims=True)
    rng2 = np.random.default_rng(SEED)
    idx = sv.stationary_bootstrap_indices(t, 200, MEAN_BLOCK_DAYS, rng2)
    for b in range(200):
        x = rc[:, idx[b]]
        sd = x.std(axis=1, ddof=1)
        restud[b] = math.sqrt(t) * x.mean(axis=1) / np.where(sd > 0, sd, 1.0)
    assert np.abs(boot - restud).max() > 1e-6, "(2) 고정 ω̂ 구성이 경로별 재추정과 같다"
    logger.info("(2) OK — 고정 ω̂ 와 경로별 재추정이 다르다 (최대차 %.3f)",
                np.abs(boot - restud).max())

    tids = np.array([f"CT|s={i}|cv=flat|15m" for i in range(50)]
                    + [f"CT|s={i}|cv=x2Ap|15m" for i in range(50)], dtype=object)
    base = rng.normal(0.0002, 0.01, size=(50, t))
    ret2 = np.vstack([base, base * 2.0])
    c = paired_contrast(ret2, tids, "flat", "x2Ap", SEED)
    assert c["n_pairs"] == 50, f"(3) 짝 수 {c['n_pairs']} != 50"
    assert c["mean_daily_diff_bp"] > 0, "(3) 2배 확대가 양의 차이를 못 낸다"
    logger.info("(3) OK — 짝대조 %d짝, 일평균차 %+.4fbp, p=%.4f",
                c["n_pairs"], c["mean_daily_diff_bp"], c["p_two_sided"])
    logger.info("자가검증 전부 통과")


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    ap = argparse.ArgumentParser(description=f"{SPEC} 판정")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("--selftest 또는 --run 중 하나가 필요하다")
    return run_verdict()


if __name__ == "__main__":
    raise SystemExit(main())
