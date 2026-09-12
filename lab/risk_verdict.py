"""RISK-2026-09-04 판정 — shrink 초과수익 · 짝지은 낙폭 대조 · 보조 RC.

명세: `docs/PREREGISTRATION_RISK_2026-09-04.md` §5·§6. 입력은 `lab/risk_tune.py`
가 기록한 산출물이며, 본 모듈은 백테스트를 다시 돌리지 않는다 — 판정만 한다.

1차 판정 (§5.1)
--------------
각 구조가 `base·shr75·shr50·shr33·shr25` 로 그리는 `(MDD, 누적수익)` **축소 곡선**
위에서, 리스크 규칙 `R` 의 낙폭 `MDD_R` 에 해당하는 수익을 단조 선형보간으로 구하고

    shrink 초과수익 = cum_R − cum_shrink(MDD_R)

를 계산한다. 외삽은 금지한다 (곡선 범위 밖 짝은 제외하고 제외 수를 보고).

합격 조건 (§6 — 셋 다 통과해야 한다)
  (a) 짝지은 대조에서 MDD 가 base 보다 유의하게 작다 (양측 p < 0.05)
  (b) shrink 초과수익 중앙값 > 0 이고 단측 p < 0.05
  (c) 변동성 타깃 계열은 위약 `plcvt` 의 초과수익보다 높다

보조 (§5.3, 권한 없음)
--------------------
전체 7,488 시행 White RC(고정 ω̂) · StepM · DSR. 리스크 규칙의 목적은 수익
최대화가 아니므로 판정 권한이 없다.

실행:
  python3 -m lab.risk_verdict --selftest
  python3 -m lab.risk_verdict --run
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab import confluence_verdict as cv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPEC = "RISK-2026-09-04"
SEED, N_PATHS, CHUNK = 20260904, 1000, 500
MEAN_BLOCK_DAYS, RC_ALPHA = 5.0, 0.05
N_DAYS = 2056

SHRINK_KEYS = ("base", "shr75", "shr50", "shr33", "shr25")
VT_FAMILY = ("vt", "vtT", "full")
PLACEBO_KEY = "plcvt"

SUMMARY = ROOT / "logs/risk_tune_summary.csv"
NPZ = ROOT / "logs/risk_tune_returns.npz"
OUT_JSON = ROOT / "logs/risk_tune_verdict.json"
OUT_TXT = ROOT / "logs/risk_tune_verdict_report.txt"


def struct_key(tid: str) -> str:
    """리스크 토큰을 제거한 구조 ID (짝지음 키)."""
    return "|".join(p for p in tid.split("|") if not p.startswith("rk="))


def shrink_excess(df: pd.DataFrame) -> pd.DataFrame:
    """구조별 축소 곡선 대비 초과수익 (§5.1).

    축소 곡선은 `SHRINK_KEYS` 5점의 `(-mdd, cum)` 이다. `-mdd` 오름차순으로
    정렬해 단조 구간만 사용하고, 대상 낙폭이 곡선 범위 밖이면 그 짝을 제외한다.

    Args:
        df: `risk_tune_summary.csv` 를 읽은 DataFrame.

    Returns:
        컬럼 `struct`·`tf`·`risk`·`mdd`·`cum`·`cum_shrink`·`excess`·`ok` 의
        DataFrame (제외된 짝은 `ok=False`, `excess=NaN`).
    """
    df = df.copy()
    df["struct"] = df.trial_id.map(struct_key)
    rows = []
    for struct, g in df.groupby("struct", sort=False):
        cur = g.set_index("risk")
        if not all(k in cur.index for k in SHRINK_KEYS):
            continue
        pts = np.array([[-cur.loc[k, "mdd"], cur.loc[k, "cum"]] for k in SHRINK_KEYS],
                       dtype=float)
        order = np.argsort(pts[:, 0])
        x, y = pts[order, 0], pts[order, 1]
        keep = np.concatenate([[True], np.diff(x) > 0])   # 중복 낙폭 제거
        x, y = x[keep], y[keep]
        monotone = len(x) >= 2
        for risk, r in cur.iterrows():
            xr = -float(r["mdd"])
            ok = bool(monotone and x[0] <= xr <= x[-1])
            cs = float(np.interp(xr, x, y)) if ok else float("nan")
            rows.append({"struct": struct, "tf": r["tf"], "risk": risk,
                         "mdd": float(r["mdd"]), "cum": float(r["cum"]),
                         "cum_shrink": cs, "excess": float(r["cum"]) - cs if ok else float("nan"),
                         "ok": ok})
    return pd.DataFrame(rows)


def paired_boot(diff: np.ndarray, seed: int, n_boot: int = 20000
                ) -> tuple[float, float, float]:
    """짝차이 벡터의 중앙값과 부트스트랩 p (단측·양측).

    Args:
        diff: 짝차이 (n,). seed: RNG 시드. n_boot: 재표본 수.

    Returns:
        `(중앙값, 단측 p(중앙값<=0), 양측 p)`.
    """
    d = diff[np.isfinite(diff)]
    if len(d) < 5:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(d))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    null = meds - med                       # 귀무: 중앙값 0
    p_one = float((1.0 + np.sum(null >= abs(med) if med < 0 else null >= med))
                  / (n_boot + 1.0)) if med >= 0 else 1.0
    p_two = float((1.0 + np.sum(np.abs(null) >= abs(med))) / (n_boot + 1.0))
    return med, p_one, p_two


def run_verdict() -> int:
    """1차 판정 + 보조 RC."""
    df = pd.read_csv(SUMMARY)
    if int(df.time_viol.sum()) != 0:
        raise SystemExit("시간 인과성 위반 — 결과 폐기")
    logger.info("%s — 시행 %d", SPEC, len(df))

    ex = shrink_excess(df)
    n_struct = ex.struct.nunique()
    base = df[df.risk == "base"].set_index(df[df.risk == "base"].trial_id.map(struct_key))

    results: dict[str, Any] = {}
    for risk in df.risk.unique():
        sub = ex[ex.risk == risk].set_index("struct")
        cur = df[df.risk == risk].copy()
        cur["struct"] = cur.trial_id.map(struct_key)
        cur = cur.set_index("struct")
        common = base.index.intersection(cur.index)
        dmdd = (cur.loc[common, "mdd"] - base.loc[common, "mdd"]).to_numpy(float)
        dcum = (cur.loc[common, "cum"] - base.loc[common, "cum"]).to_numpy(float)
        exc = sub.loc[sub.index.intersection(common), "excess"].to_numpy(float)
        med_mdd, _, p_mdd = paired_boot(dmdd, SEED)
        med_exc, p_exc_one, _ = paired_boot(exc, SEED + 1)
        results[risk] = {
            "n_pairs": int(len(common)),
            "n_excess_usable": int(np.isfinite(exc).sum()),
            "n_excess_dropped": int((~np.isfinite(exc)).sum()),
            "median_mdd": float(cur.loc[common, "mdd"].median()),
            "median_cum": float(cur.loc[common, "cum"].median()),
            "median_calmar": float(cur.loc[common, "calmar"].median()),
            "median_ulcer": float(cur.loc[common, "ulcer"].median()),
            "median_exposure": float(cur.loc[common, "exposure"].median()),
            "median_win_rate": float(cur.loc[common, "win_rate"].median()),
            "median_trades": float(cur.loc[common, "n_trades"].median()),
            "worst_day": float(cur.loc[common, "worst_day"].min()),
            "worst_trade": float(cur.loc[common, "worst_trade"].min()),
            "ruin_frac": float(cur.loc[common, "ruin"].mean()),
            "d_mdd_median": med_mdd, "d_mdd_p_two": p_mdd,
            "d_cum_median": float(np.median(dcum)),
            "shrink_excess_median": med_exc, "shrink_excess_p_one": p_exc_one,
            "frac_excess_positive": float(np.mean(exc[np.isfinite(exc)] > 0))
            if np.isfinite(exc).any() else float("nan"),
        }

    # 축소 곡선 선형성 (예측 1)
    lin = {}
    for k in SHRINK_KEYS[1:]:
        cur = df[df.risk == k].copy()
        cur["struct"] = cur.trial_id.map(struct_key)
        cur = cur.set_index("struct")
        common = base.index.intersection(cur.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = (cur.loc[common, "mdd"] / base.loc[common, "mdd"]).to_numpy(float)
            rc = (cur.loc[common, "cum"] / base.loc[common, "cum"]).to_numpy(float)
        good = np.isfinite(rm) & np.isfinite(rc) & (base.loc[common, "mdd"].to_numpy() < -1e-6)
        lin[k] = {"mdd_ratio_median": float(np.median(rm[good])),
                  "cum_ratio_median": float(np.median(rc[good])),
                  "n": int(good.sum())}

    plc = results.get(PLACEBO_KEY, {}).get("shrink_excess_median", float("nan"))
    verdict = {}
    for risk, r in results.items():
        if risk in SHRINK_KEYS:
            continue
        a = bool(r["d_mdd_median"] > 0 and r["d_mdd_p_two"] < RC_ALPHA)  # mdd 는 음수 → 차이>0 이 감소
        b = bool(r["shrink_excess_median"] > 0 and r["shrink_excess_p_one"] < RC_ALPHA)
        c = True if risk not in VT_FAMILY else bool(
            r["shrink_excess_median"] > plc)
        verdict[risk] = {"a_mdd_reduced": a, "b_beats_shrink": b,
                         "c_beats_placebo": c, "pass": bool(a and b and c)}
    any_pass = any(v["pass"] for v in verdict.values())

    # 보조 RC (권한 없음)
    aux = None
    if NPZ.exists():
        z = np.load(NPZ, allow_pickle=True)
        ret = np.ascontiguousarray(z["daily_returns"], dtype=np.float64)
        tids = np.asarray(z["trial_ids"], dtype=object)
        blk = cv.stat_block(ret, tids, n_trials=ret.shape[0], seed=SEED)
        aux = {k: blk[k] for k in ("rc_p", "rc_reject", "critical_stat_at_95",
                                   "obs_max_stat", "obs_max_sharpe_ann",
                                   "best_trial_id", "stepm_n_rejected",
                                   "n_passing_dsr")}
        aux["authority"] = "none — 수익 검정이며 본 명세의 1차 판정이 아니다"

    out = {"spec": SPEC, "generated_at": datetime.now(timezone.utc).isoformat(),
           "n_trials": int(len(df)), "n_structs": int(n_struct),
           "criteria": "(a) MDD 유의 감소 AND (b) shrink 초과수익 > 0 유의 "
                       "AND (c) 위약 초과 (vt 계열)",
           "shrink_linearity": lin, "per_rule": results, "verdict": verdict,
           "any_pass": any_pass, "aux_reality_check": aux}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    OUT_TXT.write_text(report(out))
    print(report(out))
    return 0


def report(v: dict) -> str:
    """사람이 읽는 판정 보고서."""
    L = ["=" * 100,
         f"{v['spec']} 판정 — 리스크 통제가 '그냥 작게 하는 것'보다 나은가",
         "=" * 100,
         f"시행 {v['n_trials']} · 구조 {v['n_structs']} · 기준 {v['criteria']}",
         "",
         "── 축소 곡선 선형성 (예측 1: 기울기 0.9~1.1) " + "─" * 45]
    for k, d in v["shrink_linearity"].items():
        ratio = d["cum_ratio_median"] / d["mdd_ratio_median"] if d["mdd_ratio_median"] else float("nan")
        L.append(f"   {k:<7} MDD비 {d['mdd_ratio_median']:.3f} · 수익비 "
                 f"{d['cum_ratio_median']:.3f} · 수익비/MDD비 {ratio:.3f}  (n={d['n']})")
    L += ["", "── 규칙별 (구조 짝지음, 중앙값) " + "─" * 60,
          f"{'규칙':<9}{'MDD':>9}{'누적':>9}{'Calmar':>8}{'Ulcer':>8}"
          f"{'노출':>7}{'승률':>7}{'거래':>7}{'ΔMDD':>9}{'p':>7}"
          f"{'shrink초과':>11}{'p':>7}{'양수%':>7}"]
    for risk, r in v["per_rule"].items():
        L.append(f"{risk:<9}{r['median_mdd']:>9.3f}{r['median_cum']:>9.3f}"
                 f"{r['median_calmar']:>8.3f}{r['median_ulcer']:>8.3f}"
                 f"{r['median_exposure']:>7.3f}{r['median_win_rate']:>7.3f}"
                 f"{r['median_trades']:>7.0f}{r['d_mdd_median']:>9.4f}"
                 f"{r['d_mdd_p_two']:>7.4f}{r['shrink_excess_median']:>11.4f}"
                 f"{r['shrink_excess_p_one']:>7.4f}"
                 f"{r['frac_excess_positive']:>7.3f}")
    L += ["", "── 합격 판정 (a) MDD 감소 · (b) 축소 초과 · (c) 위약 초과 " + "─" * 30]
    def mark(b):
        return "O" if b else "X"
    for risk, d in v["verdict"].items():
        L.append(f"   {risk:<9} (a) {mark(d['a_mdd_reduced'])}  "
                 f"(b) {mark(d['b_beats_shrink'])}  (c) {mark(d['c_beats_placebo'])}"
                 f"   → {'통과' if d['pass'] else '실패'}")
    L += ["", f"판정: {'통과 규칙 있음' if v['any_pass'] else '실패 — 통과 규칙 0개'}"]
    if v.get("aux_reality_check"):
        a = v["aux_reality_check"]
        L += ["", "── 보조 다중검정 (권한 없음) " + "─" * 62,
              f"   RC p = {a['rc_p']:.4f} · 임계 95% {a['critical_stat_at_95']:+.3f} · "
              f"관측 최대 {a['obs_max_stat']:+.3f}",
              f"   최고 시행 {a['best_trial_id']} (연환산 SR {a['obs_max_sharpe_ann']:+.3f})",
              f"   StepM 기각 {a['stepm_n_rejected']} · DSR 통과 {a['n_passing_dsr']}"]
    L.append("=" * 100)
    return "\n".join(L)


def selftest() -> None:
    """합성 데이터 자가검증.

    1. 축소 곡선 보간이 곡선 위의 점에 대해 초과수익 0 을 준다.
    2. 곡선 범위 밖 점은 제외된다.
    3. 곡선 위쪽 점은 양의 초과수익을 받는다.

    Raises:
        AssertionError: 불변식 위반 시.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = []
    for s in range(3):
        for k, sz in zip(SHRINK_KEYS, (1.0, 0.75, 0.5, 0.33, 0.25)):
            rows.append({"trial_id": f"RK|s={s}|rk={k}|15m", "tf": "15m", "risk": k,
                         "mdd": -0.40 * sz, "cum": 0.80 * sz})
        rows.append({"trial_id": f"RK|s={s}|rk=onc|15m", "tf": "15m", "risk": "onc",
                     "mdd": -0.20, "cum": 0.40})            # 곡선 위 (정확히)
        rows.append({"trial_id": f"RK|s={s}|rk=above|15m", "tf": "15m", "risk": "above",
                     "mdd": -0.20, "cum": 0.55})            # 곡선 위쪽
        rows.append({"trial_id": f"RK|s={s}|rk=out|15m", "tf": "15m", "risk": "out",
                     "mdd": -0.80, "cum": 1.60})            # 범위 밖
    ex = shrink_excess(pd.DataFrame(rows))
    onc = ex[ex.risk == "onc"]
    assert np.allclose(onc.excess, 0.0, atol=1e-12), f"(1) 곡선 위 점 초과 {onc.excess.values}"
    out = ex[ex.risk == "out"]
    assert (~out.ok).all(), "(2) 범위 밖 점이 제외되지 않았다"
    ab = ex[ex.risk == "above"]
    assert (ab.excess > 0.14).all(), f"(3) 곡선 위쪽 점 초과 {ab.excess.values}"
    logger.info("(1)(2)(3) OK — 보간 0 · 외삽 제외 · 위쪽 양수 (%.4f)", ab.excess.iloc[0])

    med, p1, p2 = paired_boot(np.full(200, 0.05), SEED)
    assert abs(med - 0.05) < 1e-12 and p1 < 0.01, f"(4) 상수 양수 짝차이 p={p1}"
    med0, p1_0, _ = paired_boot(np.random.default_rng(0).normal(0, 1, 500), SEED)
    assert p1_0 > 0.05, f"(4) 잡음 짝차이가 유의하게 나왔다 p={p1_0}"
    logger.info("(4) OK — 부트스트랩 짝검정 (상수 p=%.4f · 잡음 p=%.4f)", p1, p1_0)
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
