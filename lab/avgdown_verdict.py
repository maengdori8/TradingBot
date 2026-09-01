"""AVGDOWN-2026-09-01 판정 — White RC(고정 ω̂) · Romano–Wolf StepM · DSR.

명세: `docs/PREREGISTRATION_AVGDOWN_2026-09-01.md` §6. 입력은 `lab/avgdown_sweep.py`
가 기록한 일수익률 행렬이며, 본 모듈은 백테스트를 다시 돌리지 않는다 — 판정만 한다.

통계량 (동결 — §6.2)
--------------------
관측 `T_k = √T · mean(r_k) / ω̂_k`, **ω̂_k = 원표본 일수익률의 std(ddof=1) 로 1회
추정한 고정 스케일**. 부트스트랩 경로에서도 분자(평균)만 재계산하고 ω̂ 는 절대
재추정하지 않는다 — White(2000)·Hansen(2005) 원형. SWEEP-2026-08-31 §13 의 오지정
사고(경로별 스케일 재추정 → 희소거래 시행이 귀무분포를 지배, 임계 연환산 +31 로
폭발)의 재발을 구조적으로 차단한다. selftest 가 "경로별 재추정 구성과 결과가
다르다"를 강제해 오지정을 탐지한다.

부트스트랩 기계는 동결 `lab/sweep_verdict.py` 에서 **읽기 전용 임포트**한다
(`stationary_bootstrap_indices` · `counts_matrix` · `reality_check_p` · `stepm` ·
`sr0_threshold` · `deflated_sharpe` · `moments`). 평균은 표본 다중집합의 대칭함수
이므로 중복도 행렬 항등식이 정확하다 (§6.2 — sweep_verdict 규약 계승).

실행:
  .venv/bin/python lab/avgdown_verdict.py --selftest   # 합성 데이터 자가검증만
  .venv/bin/python lab/avgdown_verdict.py --run        # 판정 1회 (스윕 산출물 필요)
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

# ── 동결 상수 (§10.3 — 변경 금지) ─────────────────────────────────────────
SEED: int = 20260901
N_PATHS: int = 1000
MEAN_BLOCK_DAYS: float = 5.0
RC_ALPHA: float = 0.05
FWER: float = 0.05
DSR_MIN: float = 0.95
N_TRIALS: int = 1248
N_DAYS: int = 2056
ANN: float = math.sqrt(365.0)
CHUNK: int = 500
SYMS = ("BTC", "ETH", "SOL")
VAR_ZERO_REL: float = sv.VAR_ZERO_REL                # 퇴화 문턱 (동결 규약 계승)


def fixed_scale(ret: np.ndarray) -> np.ndarray:
    """원표본에서 1회 추정하는 고정 스케일 ω̂ (시행별, §6.2).

    `ω̂_k = std(r_k, ddof=1)`. 퇴화(상수 계열, 거래 0 포함)는 ω̂ := 0 으로 표시하고
    통계량을 0 으로 둔다 — N 에는 그대로 산입한다 (사후 제거 금지).

    Args:
        ret: (N, T) 일수익률.

    Returns:
        (N,) ω̂ — 퇴화 시행은 0.
    """
    n = ret.shape[1]
    mean = ret.mean(axis=1)
    dev = ret - mean[:, None]
    var = (dev * dev).sum(axis=1) / (n - 1)
    m2 = (ret * ret).sum(axis=1) / n
    ok = (var > 0.0) & (var > VAR_ZERO_REL * m2) & (np.ptp(ret, axis=1) > 0)
    return np.where(ok, np.sqrt(np.where(ok, var, 1.0)), 0.0)


def obs_stat(ret: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """관측 통계량 `T_k = √T · mean_k / ω̂_k` (퇴화 ω̂=0 → 0)."""
    t = ret.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = math.sqrt(t) * ret.mean(axis=1) / np.where(omega > 0, omega, 1.0)
    return np.where(omega > 0, s, 0.0)


def bootstrap_null_stats(
    ret: np.ndarray, omega: np.ndarray, n_paths: int = N_PATHS, seed: int = SEED,
    mean_block: float = MEAN_BLOCK_DAYS, chunk: int = CHUNK,
) -> np.ndarray:
    """귀무(열 평균 0 중심화) 하 부트스트랩 통계량 행렬 (B, N).

    **ω̂ 는 인자로 받은 고정값만 쓴다 — 경로별 재추정 금지 (§6.2 동결).**
    평균은 대칭함수이므로 중복도 행렬 `C @ rcᵀ / T` 가 재표집 실체화와 정확히 같다.

    Args:
        ret: (N, T) 일수익률 (중심화 전). omega: (N,) 고정 스케일.
        n_paths: 경로 수. seed: 난수 seed. mean_block: 평균 블록(일). chunk: 청크.

    Returns:
        (n_paths, N) float64.
    """
    n_rules, t = ret.shape
    rc = ret - ret.mean(axis=1, keepdims=True)
    rc[omega <= 0] = 0.0                             # 퇴화 시행 — 정확히 0
    rc_t = np.ascontiguousarray(rc.T)
    inv = np.where(omega > 0, math.sqrt(t) / np.where(omega > 0, omega, 1.0), 0.0)
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, n_rules))
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        idx = sv.stationary_bootstrap_indices(t, m, mean_block, rng)
        cmat = sv.counts_matrix(idx, t)
        out[done:done + m] = (cmat @ rc_t / t) * inv[None, :]
        done += m
    return out


def _stat_block(ret: np.ndarray, label: str, n_paths: int, seed: int
                ) -> dict[str, Any]:
    """행렬 1개에 대한 RC·StepM·DSR 일괄 판정 블록."""
    t = ret.shape[1]
    omega = fixed_scale(ret)
    stat = obs_stat(ret, omega)
    sr_d = stat / math.sqrt(t)                       # 일 단위 Sharpe (= mean/ω̂)
    boot = bootstrap_null_stats(ret, omega, n_paths=n_paths, seed=seed)
    rc_p, boot_max = sv.reality_check_p(stat, boot)
    rejected, steps = sv.stepm(stat, boot, alpha=FWER)
    var_sr = float(np.var(sr_d, ddof=1))
    sr0 = sv.sr0_threshold(var_sr, n_trials=N_TRIALS)
    order = np.argsort(-stat, kind="stable")
    best = int(order[0])

    def _row(i: int) -> dict[str, Any]:
        g3, g4 = sv.moments(ret[i])
        dsr, margin = sv.deflated_sharpe(float(sr_d[i]), sr0, g3, g4, n_obs=t)
        return {"index": int(i), "stat": float(stat[i]),
                "mean_daily_ret": float(ret[i].mean()),
                "ann_mean_ret": float(ret[i].mean()) * 365.0,
                "sharpe_ann": float(sr_d[i]) * ANN,
                "skew": g3, "kurtosis": g4, "dsr": dsr,
                "margin_sr_minus_sr0": margin}

    survivors = [_row(i) for i in sorted(rejected, key=lambda i: -stat[i])]
    passed = [r for r in survivors if (r["dsr"] == r["dsr"]) and r["dsr"] >= DSR_MIN]
    return {
        "label": label, "n_trials": int(ret.shape[0]), "t_days": int(t),
        "n_degenerate": int((omega <= 0).sum()),
        "reality_check": {
            "p": rc_p, "alpha": RC_ALPHA, "obs_max_stat": float(stat.max()),
            "obs_max_sharpe_ann": float(sr_d.max()) * ANN,
            "null_max_quantiles": {f"p{q:g}": float(np.percentile(boot_max, q))
                                   for q in (50, 90, 95, 99, 100)},
            "critical_stat_at_95": float(np.quantile(boot_max, 0.95)),
            "reject": bool(rc_p < RC_ALPHA),
        },
        "stepm": {"fwer": FWER, "n_rejected": len(rejected), "steps": steps,
                  "rejected_indices": [int(i) for i in rejected]},
        "dsr": {"n_trials": N_TRIALS, "var_sr_daily": var_sr,
                "std_sr_ann": math.sqrt(var_sr) * ANN,
                "sr0_daily": sr0, "sr0_ann": sr0 * ANN, "dsr_min": DSR_MIN,
                "survivors": survivors},
        "best": _row(best),
        "top10": [_row(int(i)) for i in order[:10]],
        "n_passing": len(passed),
        "passing_indices": [r["index"] for r in passed],
    }


def run_verdict(npz_path: Path, n_paths: int = N_PATHS, seed: int = SEED
                ) -> dict[str, Any]:
    """판정 1회 — 1차(3심볼 균등 합산) + 심볼별 참고 블록.

    Args:
        npz_path: `avgdown_returns.npz` 경로. n_paths: 경로 수. seed: seed.

    Returns:
        직렬화 가능한 판정 딕셔너리.

    Raises:
        SystemExit: 행렬 모양·유일성·유한성 위반 (fail-closed).
    """
    z = np.load(npz_path, allow_pickle=True)
    ret = np.ascontiguousarray(z["daily_returns"], dtype=np.float64)
    tids = np.asarray(z["trial_ids"], dtype=object)
    meta = json.loads(str(z["meta"]))
    if ret.shape != (N_TRIALS, N_DAYS):
        raise SystemExit(f"행렬 {ret.shape} != ({N_TRIALS}, {N_DAYS}) — fail-closed")
    if len(set(tids.tolist())) != N_TRIALS:
        raise SystemExit("시행 ID 유일성 위반 — fail-closed")
    if not np.isfinite(ret).all():
        raise SystemExit("일수익률 NaN/Inf — fail-closed")

    main_blk = _stat_block(ret, "combined_equal_weight", n_paths, seed)
    # 시행 ID 주석 부여
    for row in ([main_blk["best"]] + main_blk["top10"] + main_blk["dsr"]["survivors"]):
        row["trial_id"] = str(tids[row["index"]])
    main_blk["stepm"]["rejected_trial_ids"] = \
        [str(tids[i]) for i in main_blk["stepm"]["rejected_indices"]]

    per_sym = {}
    for s in SYMS:
        if f"ret_{s}" in z:
            blk = _stat_block(np.ascontiguousarray(z[f"ret_{s}"], dtype=np.float64),
                              f"per_symbol_{s}", n_paths, seed)
            blk["authority"] = "none — 참고 전용 (§5.3, 하위표본 구제 금지)"
            for row in [blk["best"]] + blk["top10"]:
                row["trial_id"] = str(tids[row["index"]])
            per_sym[s] = blk

    rc = main_blk["reality_check"]
    verdict_pass = bool(rc["reject"] and main_blk["stepm"]["n_rejected"] > 0
                        and main_blk["n_passing"] > 0)
    return {
        "spec": "AVGDOWN-2026-09-01",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_meta": meta,
        "bootstrap": {"n_paths": n_paths, "seed": seed,
                      "mean_block_days": MEAN_BLOCK_DAYS,
                      "statistic": "sqrt(T)·mean/ω̂ — ω̂ 원표본 1회 추정 고정 "
                                   "(경로별 재추정 금지, SWEEP §13 오지정 재발 방지)",
                      "rng": "numpy.random.default_rng (PCG64)"},
        "primary": main_blk,
        "per_symbol_reference": per_sym,
        "verdict": {
            "pass": verdict_pass,
            "criteria": "RC p < 0.05 AND StepM 기각 >= 1 AND DSR >= 0.95",
            "rc_pass": bool(rc["reject"]),
            "stepm_pass": bool(main_blk["stepm"]["n_rejected"] > 0),
            "dsr_pass": bool(main_blk["n_passing"] > 0),
            "n_passing_trials": main_blk["n_passing"],
            "passing_trial_ids": [str(tids[i]) for i in main_blk["passing_indices"]],
            "statement": (
                "공동 null 상단 초과 — 엣지 입증 아님, 전방 확인 필요"
                if verdict_pass else
                "1,248 시행 최대 통계량이 zero-edge 공동 null 과 구별되지 않음"
            ),
        },
    }


# ── 자가검증 (합성 데이터 전용 — 본 산출물 미접촉) ─────────────────────────
def selftest() -> None:
    """판정 기계 자가검증 — 위반 시 AssertionError.

    1. 고정 ω̂ 항등식: 중복도 행렬 경로 == 재표집 실체화 경로 (동일 고정 ω̂).
    2. 오지정 탐지: 경로별 ω̂ 재추정 구성과 결과가 **다르다** (§13 사고 재발 방지).
    3. 검정력 배선: 강한 양(+) 신호 주입 시행이 RC·StepM 에서 기각된다.
    4. 결정론: 같은 seed → 같은 부트스트랩 통계.
    """
    print("--- selftest (avgdown_verdict — 합성 데이터) ---")
    rng = np.random.default_rng(42)
    n, t = 40, 800
    ret = rng.normal(0.0, 0.01, (n, t))
    ret[5] = 0.0                                     # 퇴화 시행
    omega = fixed_scale(ret)
    assert omega[5] == 0.0 and (omega[:5] > 0).all()
    boot = bootstrap_null_stats(ret, omega, n_paths=64, seed=SEED)
    # 1) 항등식 — 같은 인덱스로 재표집 실체화 (동일 고정 ω̂)
    rng2 = np.random.default_rng(SEED)
    idx = sv.stationary_bootstrap_indices(t, 64, MEAN_BLOCK_DAYS, rng2)
    rc = ret - ret.mean(axis=1, keepdims=True)
    rc[omega <= 0] = 0.0
    naive = np.empty((64, n))
    for b in range(64):
        x = rc[:, idx[b]]
        naive[b] = np.where(omega > 0,
                            math.sqrt(t) * x.mean(axis=1)
                            / np.where(omega > 0, omega, 1.0), 0.0)
    err = float(np.max(np.abs(boot - naive)))
    assert err < 1e-10, err
    print(f"  [OK] 고정 ω̂ 중복도 항등식 — 최대 오차 {err:.2e}")
    # 2) 오지정 대조 — 경로별 재추정(금지 구성)과 상이해야 한다
    restud = np.empty((64, n))
    for b in range(64):
        x = rc[:, idx[b]]
        sd = x.std(axis=1, ddof=1)
        restud[b] = np.where(sd > 0, math.sqrt(t) * x.mean(axis=1)
                             / np.where(sd > 0, sd, 1.0), 0.0)
    assert float(np.max(np.abs(boot - restud))) > 1e-6
    print("  [OK] 경로별 ω̂ 재추정 구성과 결과 상이 — 오지정(§13 사고) 탐지 배선")
    # 3) 검정력 — 강한 신호 주입
    sig = ret.copy()
    sig[7] = sig[7] + 0.02                           # 일 +2% 균일 엣지
    om2 = fixed_scale(sig)
    st = obs_stat(sig, om2)
    bt = bootstrap_null_stats(sig, om2, n_paths=200, seed=SEED)
    p, _ = sv.reality_check_p(st, bt)
    rej, _steps = sv.stepm(st, bt, alpha=FWER)
    assert p < 0.05 and 7 in rej, (p, rej)
    print(f"  [OK] 주입 신호 기각 — RC p={p:.4f}, StepM 기각 {rej}")
    # 4) 결정론
    boot2 = bootstrap_null_stats(ret, omega, n_paths=64, seed=SEED)
    assert np.array_equal(boot, boot2)
    print("  [OK] 같은 seed → 비트 동일 부트스트랩")
    print("--- selftest 전부 통과 ---")


def _fmt(v: dict[str, Any]) -> str:
    """판정 딕셔너리 → 사람이 읽는 보고서."""
    out: list[str] = []
    a = out.append
    p = v["primary"]
    rc = p["reality_check"]
    a("=" * 78)
    a("AVGDOWN-2026-09-01 판정 (1차 = 3심볼 균등 합산 일수익률, 비용·펀딩 차감 후)")
    a("=" * 78)
    b = p["best"]
    a(f"최고 시행: {b.get('trial_id', '?')}")
    a(f"  일평균 순수익률 {b['mean_daily_ret']*1e4:+.3f}bp (연환산 {b['ann_mean_ret']*100:+.2f}%) · "
      f"연환산 SR {b['sharpe_ann']:+.3f} · DSR {b['dsr'] if b['dsr'] != b['dsr'] else round(b['dsr'], 4)}")
    a(f"  RC p = {rc['p']:.4f} (임계 통계량 95% = {rc['critical_stat_at_95']:+.3f}, "
      f"관측 최대 {rc['obs_max_stat']:+.3f})")
    a(f"  StepM 기각 {p['stepm']['n_rejected']}개 · DSR>=0.95 통과 {p['n_passing']}개 · "
      f"퇴화 {p['n_degenerate']}개")
    vd = v["verdict"]
    a(f"판정: {'통과' if vd['pass'] else '실패'} — {vd['statement']}")
    for s, blk in v["per_symbol_reference"].items():
        a(f"  [참고 {s}] RC p={blk['reality_check']['p']:.4f} · "
          f"StepM {blk['stepm']['n_rejected']} · 최대 SR "
          f"{blk['reality_check']['obs_max_sharpe_ann']:+.3f} (권한 없음)")
    a("=" * 78)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    ap = argparse.ArgumentParser(description="AVGDOWN-2026-09-01 판정")
    ap.add_argument("--selftest", action="store_true", help="합성 자가검증만")
    ap.add_argument("--run", action="store_true", help="판정 1회 수행")
    ap.add_argument("--indir", default="logs")
    ap.add_argument("--outdir", default="logs")
    ap.add_argument("--paths", type=int, default=N_PATHS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        selftest()
        return 0
    if not args.run:
        print("아무 것도 하지 않음 — --selftest 또는 --run 지정")
        return 1
    selftest()                                       # 판정 전 자가검증 강제
    v = run_verdict(ROOT / args.indir / "avgdown_returns.npz", n_paths=args.paths)
    outd = ROOT / args.outdir
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "avgdown_verdict.json").write_text(
        json.dumps(v, ensure_ascii=False, indent=2))
    rep = _fmt(v)
    (outd / "avgdown_verdict_report.txt").write_text(rep + "\n")
    print(rep)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
