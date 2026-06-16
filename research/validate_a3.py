"""[H-002 / A3] 비용 민감도 곡선 — '어느 비용에서 죽는가' + cost=0 게이트.

핵심 질문(A1 후속): 메이커가 비용 부호를 뒤집었지만, **비용을 0으로 해도(그로스)** 신호가
다중검정 보정 후 노이즈 이상인가? 0이어도 노이즈 이하라면 tier-A(비용각도) 전체가 소진된다.

해석적 비용복원(재생 불필요): mk0_(메이커0.0007)와 tkr_(테이커0.0021)는 동일 진입·동일 부분집합,
비용만 다르다. 따라서 행별로
    e/risk = (mk0_net - tkr_net) / (0.0021 - 0.0007)
    gross  = mk0_net + 0.0007·e/risk = mk0_net + 0.5·(mk0_net - tkr_net)
    net(c) = gross - c·e/risk
로 임의 비용 c의 순R을 복원한다. cost=0(그로스)에서 stats_gate를 돌려 tier-A를 판정한다.

출력: research/out/a3_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.stats_gate as sg  # noqa: E402
from research.validate_a1 import _enrich, _best_round, _eval, ROUNDS  # noqa: E402

logger = logging.getLogger("validate_a3")
OUT_DIR = ROOT / "research" / "out"

MAKER, TAKER = 0.0007, 0.0021
DCOST = TAKER - MAKER  # 0.0014


def _add_cost_columns(df: pd.DataFrame, cost: float) -> tuple[pd.DataFrame, list[str]]:
    """mk0_/tkr_ 쌍에서 임의 비용 c의 순R 컬럼(cst_m*_rr*)을 해석적으로 만든다."""
    df = df.copy()
    cols = []
    mk_cols = [c for c in df.columns if c.startswith("mk0_m") and "_rr" in c]
    for mc in mk_cols:
        suffix = mc[len("mk0"):]          # _m{m}_rr{rr}
        tc = "tkr" + suffix
        if tc not in df.columns:
            continue
        e_over_risk = (df[mc] - df[tc]) / DCOST
        gross = df[mc] + MAKER * e_over_risk
        new = "cst" + suffix
        df[new] = (gross - cost * e_over_risk).round(4)
        cols.append(new)
    return df, cols


def main() -> None:
    """A3 비용 민감도 + cost=0 게이트."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-until", required=True)
    parser.add_argument("--n-prior", type=int, default=405,
                        help="A1까지 누적 N (기본 405)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    path = OUT_DIR / "a1_signals.csv"
    if not path.exists():
        logger.error("a1_signals.csv 없음 — exec_a1 먼저"); sys.exit(1)
    df_all = _enrich(pd.read_csv(path))
    holdout_start = pd.Timestamp(args.holdout_until, tz="UTC")

    # ── 비용 곡선: 각 비용에서 (전수, 사후선택 없는) 대표 출구 m2_rr2.5 OOS평균 + 최선그로스 ──
    cost_grid = [0.0, 0.0003, 0.0005, MAKER, 0.0010, 0.0013, 0.0017, TAKER, 0.0028]
    curve = []
    for c in cost_grid:
        dfc, _ = _add_cost_columns(df_all, c)
        rep = dfc["cst_m2_rr2.5"].dropna()
        # 전수 대표 출구 평균 + 그 비용에서의 전체 최선 단일출구(전수기준)
        best_col, best_mean = None, -1e9
        for col in [x for x in dfc.columns if x.startswith("cst_m")]:
            mm = dfc[col].dropna().mean()
            if mm > best_mean:
                best_mean, best_col = mm, col
        curve.append({"cost": c, "rep_m2_rr2.5_mean": round(float(rep.mean()), 4),
                      "best_single_exit": best_col, "best_single_mean": round(float(best_mean), 4)})

    # 손익분기 비용(대표출구 전수평균 = 0 되는 c) 선형보간
    xs = [p["cost"] for p in curve]; ys = [p["rep_m2_rr2.5_mean"] for p in curve]
    breakeven = None
    for i in range(len(xs) - 1):
        if ys[i] >= 0 >= ys[i + 1]:
            breakeven = round(xs[i] + (xs[i+1]-xs[i]) * ys[i] / (ys[i]-ys[i+1]), 5)
            break

    # ── cost=0(그로스) 동결홀드아웃 WFO + stats_gate (사후선택 정직 반영) ──
    df0, _ = _add_cost_columns(df_all, 0.0)
    # 체결분만(메이커 일관) — A1과 동일 모집단
    pop = df0[df0["filled"] == 1].reset_index(drop=True) if "filled" in df0 else df0
    search = pop[pop["ts"] < holdout_start].reset_index(drop=True)
    hold = pop[pop["ts"] >= holdout_start].reset_index(drop=True)
    br = _best_round(search, "cst")
    report = {
        "cost_curve": curve, "breakeven_cost_roundtrip": breakeven,
        "interpretation": "대표출구 손익분기 비용. 메이커 0.0007 < 손익분기면 메이커서 그로스양수.",
        "gross_cost0": {},
    }
    if br["best"] is None:
        report["gross_cost0"] = {"verdict": "표본부족", "final_verdict": "REJECT"}
        report["final_verdict"] = "REJECT"
    else:
        b = br["best"]
        ho = _eval(hold, b["robust_param"])
        n_tested = args.n_prior + 1 + b["grid_size"] + 3 + 1
        gate = sg.gate(b["oos_r"], b["oos_ts"], n_tested=n_tested)
        report["gross_cost0"] = {
            "robust_param": b["robust_param"], "oos_mean_r": b["oos_mean_r"],
            "oos_n": b["oos_n"], "windows": b["windows"],
            "holdout": {k: round(v, 4) for k, v in ho.items()},
            "n_tested": n_tested, "stats_gate_passed": gate.passed,
            "stats_gate_reasons": gate.reasons,
            "sharpe": gate.sharpe, "bootstrap": gate.bootstrap,
        }
        report["all_rounds_gross"] = br["all_rounds"]
        # tier-A 판정: cost=0(그로스)에서도 게이트 실패면 tier-A 소진
        report["final_verdict"] = "PROMOTE" if gate.passed else "REJECT"
        report["tier_a_exhausted"] = not gate.passed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "a3_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("손익분기 비용(왕복): %s", breakeven)
    logger.info("cost=0 그로스 판정: %s", report["final_verdict"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:1800])


if __name__ == "__main__":
    main()
