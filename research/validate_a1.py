"""[H-001 / A1] 검증 하니스 — 동결 홀드아웃 WFO + stats_gate 판정.

GOAL_LOOP §2의 함정을 우회한다:
- wfo.py 내장 _verdict 사용 안 함(노이즈도 PASS). stats_gate.gate() 가 유일 판정자.
- wfo.py의 이동창 홀드아웃 사용 안 함. STATE.json의 HOLDOUT_FROZEN_UNTIL 고정 경계로 분리.
- 평가한 (필터×출구) 조합수를 세어 N에 반영(다중검정).

비교군: 메이커 PRIMARY(r_m*) vs 테이커 베이스라인(tkr_m*) vs 플라시보(방향셔플).
출력: research/out/a1_validation.json
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

import research.wfo as wfo  # noqa: E402
import research.stats_gate as sg  # noqa: E402

logger = logging.getLogger("validate_a1")
OUT_DIR = ROOT / "research" / "out"

ROUNDS = [
    {"train_days": 180, "test_days": 30, "min_train_trades": 40},
    {"train_days": 120, "test_days": 30, "min_train_trades": 25},
    {"train_days": 240, "test_days": 45, "min_train_trades": 60},
]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """bool 정규화 + btc_aligned 추가 (wfo 규약)."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("zone_both", "kz_true"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    df["btc_aligned"] = wfo._btc_alignment(df)
    return df.sort_values("ts").reset_index(drop=True)


def _exit_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """주어진 접두사의 출구 그리드 컬럼."""
    return [c for c in df.columns if c.startswith(f"{prefix}_m") and "_rr" in c]


def _grid(df: pd.DataFrame, prefix: str) -> list[dict]:
    """진입필터 × 출구기하 조합 (해당 접두사)."""
    import itertools
    cols = _exit_cols(df, prefix)
    out = []
    for ms, zb, btc, ec in itertools.product(
        [65, 70, 75, 80], [False, True], ["any", "no_counter"], cols):
        out.append({"min_score": ms, "zone_both": zb, "btc": btc, "exit_col": ec})
    return out


def _eval(df: pd.DataFrame, p: dict) -> dict:
    """파라미터 성과 (해당 출구의 순R)."""
    r = df.loc[wfo._mask(df, p), p["exit_col"]].dropna()
    n = len(r)
    if n == 0:
        return {"n": 0, "mean_r": 0.0, "winrate": 0.0}
    return {"n": n, "mean_r": float(r.mean()), "winrate": float((r > 0).mean())}


def _walk_forward(df: pd.DataFrame, prefix: str, rc: dict) -> dict:
    """롤링 train→test. OOS 거래의 (ts, r)를 수집 (stats_gate용)."""
    grid = _grid(df, prefix)
    t0, t_end = df["ts"].min(), df["ts"].max()
    oos_ts: list = []
    oos_r: list = []
    chosen: list[str] = []
    cursor = t0 + timedelta(days=rc["train_days"])
    while cursor < t_end:
        tr = df[(df["ts"] >= cursor - timedelta(days=rc["train_days"])) & (df["ts"] < cursor)]
        te = df[(df["ts"] >= cursor) & (df["ts"] < cursor + timedelta(days=rc["test_days"]))]
        cursor += timedelta(days=rc["test_days"])
        if len(te) == 0:
            continue
        best, best_r = None, -1e9
        for p in grid:
            s = _eval(tr, p)
            if s["n"] >= rc["min_train_trades"] and s["mean_r"] > best_r:
                best, best_r = p, s["mean_r"]
        if best is None:
            continue
        m = wfo._mask(te, best)
        sub = te.loc[m, ["ts", best["exit_col"]]].dropna()
        oos_ts.extend(sub["ts"].tolist())
        oos_r.extend(sub[best["exit_col"]].tolist())
        chosen.append(f"{best['min_score']}|{int(best['zone_both'])}|{best['btc']}|{best['exit_col']}")
    rs = np.array(oos_r, dtype=float)
    robust = max(set(chosen), key=chosen.count) if chosen else None
    rp = None
    if robust:
        a, b, c, d = robust.split("|")
        rp = {"min_score": int(a), "zone_both": bool(int(b)), "btc": c, "exit_col": d}
    return {"oos_ts": oos_ts, "oos_r": rs, "windows": len(chosen),
            "robust_param": rp, "grid_size": len(grid),
            "oos_mean_r": round(float(rs.mean()), 4) if len(rs) else 0.0,
            "oos_n": int(len(rs))}


def _best_round(df: pd.DataFrame, prefix: str) -> dict:
    """3라운드 중 OOS mean_r 최고 채택 (wfo.main 미러 — N에 +3)."""
    results = [{**_walk_forward(df, prefix, rc), "config": rc} for rc in ROUNDS]
    valid = [r for r in results if r["oos_n"] >= 50]
    best = max(valid, key=lambda r: r["oos_mean_r"]) if valid else None
    return {"best": best, "all_rounds": [
        {"config": r["config"], "oos_mean_r": r["oos_mean_r"], "oos_n": r["oos_n"],
         "windows": r["windows"]} for r in results]}


def main() -> None:
    """A1 검증 실행."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-until", required=True,
                        help="동결 홀드아웃 시작 경계 ISO8601 (이후=홀드아웃)")
    parser.add_argument("--n-prior", type=int, default=0,
                        help="이전 누적 시도수 (이번 가설 전까지)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    path = OUT_DIR / "a1_signals.csv"
    if not path.exists():
        logger.error("a1_signals.csv 없음 — exec_a1 먼저 실행")
        sys.exit(1)
    raw = pd.read_csv(path)
    df_all = _enrich(raw)
    n_total = len(df_all)
    n_filled = int(df_all["filled"].sum()) if "filled" in df_all else 0
    fill_rate = n_filled / n_total if n_total else 0.0

    holdout_start = pd.Timestamp(args.holdout_until, tz="UTC")

    def split(df):
        return (df[df["ts"] < holdout_start].reset_index(drop=True),
                df[df["ts"] >= holdout_start].reset_index(drop=True))

    report = {
        "holdout_frozen_until": str(holdout_start),
        "n_signals_total": n_total, "n_filled": n_filled,
        "maker_fill_rate": round(fill_rate, 4),
        "offset_atr": 0.25, "fill_window_bars": 8,
        "maker_cost": float(raw.get("offset_atr", pd.Series([0.25])).iloc[0]) if False else 0.0007,
    }

    # ── 메이커 PRIMARY (체결분만, r_m* 컬럼) ──
    maker = df_all[df_all["filled"] == 1].reset_index(drop=True) if "filled" in df_all else df_all
    m_search, m_hold = split(maker)
    m_best = _best_round(m_search, "r")
    report["maker"] = {"search_n": len(m_search), "holdout_n": len(m_hold),
                       "rounds": m_best["all_rounds"]}

    if m_best["best"] is None:
        report["maker"]["verdict"] = "표본부족 — OOS<50, REJECT(미달)"
        report["final_verdict"] = "REJECT"
    else:
        b = m_best["best"]
        # 홀드아웃 평가 (동결 경계, 1회)
        ho = _eval(m_hold, b["robust_param"])
        # 다중검정 N: 이전누적 + 티켓1 + 그리드조합 + 라운드3 + 홀드아웃1
        n_tested = args.n_prior + 1 + b["grid_size"] + 3 + 1
        gate = sg.gate(b["oos_r"], b["oos_ts"], n_tested=n_tested)
        # 플라시보 음성통제: 체결분 기준출구 플라시보 평균
        plc = df_all.loc[df_all["filled"] == 1, "placebo_r_ref"].dropna() \
            if "placebo_r_ref" in df_all else pd.Series(dtype=float)
        report["maker"].update({
            "robust_param": b["robust_param"],
            "oos_mean_r": b["oos_mean_r"], "oos_n": b["oos_n"], "windows": b["windows"],
            "holdout": {k: round(v, 4) for k, v in ho.items()},
            "n_tested": n_tested,
            "stats_gate": gate.to_dict(),
            "placebo_mean_r": round(float(plc.mean()), 4) if len(plc) else None,
            "placebo_n": int(len(plc)),
        })
        # 최종 판정: stats_gate 통과 AND 홀드아웃 양수 AND 플라시보 대비 우월
        ho_ok = ho.get("mean_r", -1) > 0 and ho.get("n", 0) >= 20
        plc_ok = (len(plc) == 0) or (b["oos_mean_r"] > float(plc.mean()))
        passed = gate.passed and ho_ok and plc_ok
        report["maker"]["holdout_ok"] = ho_ok
        report["maker"]["placebo_separation_ok"] = plc_ok
        report["final_verdict"] = "PROMOTE" if passed else "REJECT"

    # ── 테이커 베이스라인 (전 신호, tkr_m*) — 비교용 ──
    t_search, _ = split(df_all)
    t_best = _best_round(t_search, "tkr")
    report["taker_baseline"] = {
        "rounds": t_best["all_rounds"],
        "oos_mean_r": t_best["best"]["oos_mean_r"] if t_best["best"] else None,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "a1_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("최종 판정: %s", report["final_verdict"])
    if report.get("maker", {}).get("stats_gate"):
        logger.info("게이트 사유: %s", report["maker"]["stats_gate"]["reasons"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
