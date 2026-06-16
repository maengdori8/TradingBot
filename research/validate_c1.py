"""[H-007 / C1] 검증 — 동결홀드아웃 WFO + stats_gate (축소 그리드).

C1은 ICT 노브(min_score/zone/btc)를 안 쓰므로 §2대로 그리드를 출구기하(SL×RR=25)로 축소한다
(평탄 노브 곱하기 = 사후선택 부풀림 회피, N 절감). 메이커 PRIMARY vs 테이커 vs 플라시보 비교.
출력: research/out/c1_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.stats_gate as sg  # noqa: E402

logger = logging.getLogger("validate_c1")
OUT_DIR = ROOT / "research" / "out"

ROUNDS = [
    {"train_days": 180, "test_days": 30, "min_train_trades": 40},
    {"train_days": 120, "test_days": 30, "min_train_trades": 25},
    {"train_days": 240, "test_days": 45, "min_train_trades": 60},
]


def _exit_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(f"{prefix}_m") and "_rr" in c]


def _walk_forward(df: pd.DataFrame, prefix: str, rc: dict) -> dict:
    """축소그리드(출구만) 롤링 WFO. OOS (ts,r) 수집."""
    cols = _exit_cols(df, prefix)
    t0, t_end = df["ts"].min(), df["ts"].max()
    oos_ts, oos_r, chosen = [], [], []
    cursor = t0 + timedelta(days=rc["train_days"])
    while cursor < t_end:
        tr = df[(df["ts"] >= cursor - timedelta(days=rc["train_days"])) & (df["ts"] < cursor)]
        te = df[(df["ts"] >= cursor) & (df["ts"] < cursor + timedelta(days=rc["test_days"]))]
        cursor += timedelta(days=rc["test_days"])
        if len(te) == 0:
            continue
        best, best_r = None, -1e9
        for col in cols:
            r = tr[col].dropna()
            if len(r) >= rc["min_train_trades"] and r.mean() > best_r:
                best, best_r = col, r.mean()
        if best is None:
            continue
        sub = te[["ts", best]].dropna()
        oos_ts.extend(sub["ts"].tolist())
        oos_r.extend(sub[best].tolist())
        chosen.append(best)
    rs = np.array(oos_r, dtype=float)
    robust = max(set(chosen), key=chosen.count) if chosen else None
    return {"oos_ts": oos_ts, "oos_r": rs, "windows": len(chosen), "robust_col": robust,
            "grid_size": len(cols), "oos_mean_r": round(float(rs.mean()), 4) if len(rs) else 0.0,
            "oos_n": int(len(rs))}


def _best_round(df: pd.DataFrame, prefix: str) -> dict:
    res = [{**_walk_forward(df, prefix, rc), "config": rc} for rc in ROUNDS]
    valid = [r for r in res if r["oos_n"] >= 50]
    best = max(valid, key=lambda r: r["oos_mean_r"]) if valid else None
    return {"best": best, "all_rounds": [
        {"train_days": r["config"]["train_days"], "oos_mean_r": r["oos_mean_r"],
         "oos_n": r["oos_n"], "windows": r["windows"]} for r in res]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-until", required=True)
    parser.add_argument("--n-prior", type=int, default=811)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    path = OUT_DIR / "c1_signals.csv"
    if not path.exists():
        logger.error("c1_signals.csv 없음"); sys.exit(1)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    hs = pd.Timestamp(args.holdout_until, tz="UTC")
    search = df[df["ts"] < hs].reset_index(drop=True)
    hold = df[df["ts"] >= hs].reset_index(drop=True)

    report = {"n_signals": len(df), "holdout_frozen_until": str(hs),
              "search_n": len(search), "holdout_n": len(hold),
              "params": {"k_bars": 4, "z_thresh": 2.0}}

    mk = _best_round(search, "r")
    report["maker_rounds"] = mk["all_rounds"]
    report["taker_oos"] = _best_round(search, "tkr")["all_rounds"]

    if mk["best"] is None:
        report["final_verdict"] = "REJECT"
        report["reason"] = "OOS<50 표본부족"
    else:
        b = mk["best"]
        hr = hold[b["robust_col"]].dropna()
        ho = {"n": int(len(hr)), "mean_r": round(float(hr.mean()), 4) if len(hr) else 0.0,
              "winrate": round(float((hr > 0).mean()), 4) if len(hr) else 0.0}
        n_tested = args.n_prior + 1 + b["grid_size"] + 3 + 1
        gate = sg.gate(b["oos_r"], b["oos_ts"], n_tested=n_tested)
        plc = df["placebo_r_ref"].dropna()
        report["maker"] = {
            "robust_col": b["robust_col"], "oos_mean_r": b["oos_mean_r"], "oos_n": b["oos_n"],
            "windows": b["windows"], "holdout": ho, "n_tested": n_tested,
            "stats_gate_passed": gate.passed, "stats_gate_reasons": gate.reasons,
            "sharpe": gate.sharpe, "bootstrap": gate.bootstrap,
            "placebo_mean_r": round(float(plc.mean()), 4) if len(plc) else None,
        }
        ho_ok = ho["mean_r"] > 0 and ho["n"] >= 20
        plc_ok = (len(plc) == 0) or (b["oos_mean_r"] > float(plc.mean()))
        report["maker"]["holdout_ok"] = ho_ok
        report["maker"]["placebo_separation_ok"] = plc_ok
        report["final_verdict"] = "PROMOTE" if (gate.passed and ho_ok and plc_ok) else "REJECT"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "c1_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("최종 판정: %s", report["final_verdict"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:1600])


if __name__ == "__main__":
    main()
