"""[H-008 / D1] 메타라벨링 — ML로 얇은 ICT 신호를 구제할 수 있는가.

가설: 베이스 ICT 신호(거래당 ~0R)에 2차 분류기를 얹어 '이길 신호'만 거래하면 OOS 엣지가
살아나는가? GOAL_LOOP §6: D1은 과적합 위험 최고 — 라벨 룩어헤드·CV 누수 극도 주의.

엄격 규율:
- 분류기는 **train 구간에서만** 학습. 워크포워드(train→미지의 test)로 OOS 누적. 홀드아웃 1회.
- 피처는 결정시점 가용만(score_raw, kz_true, zone_both, btc_aligned, direction, weekday, hour, symbol).
- 라벨 y = 베이스 거래 승패(mk0_m2_rr2.5 > 0). 사전등록 모델·선택규칙 고정.
- stats_gate가 유일 판정. N에 모델 선택 자유도 가산.
출력: research/out/d1_validation.json
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

logger = logging.getLogger("validate_d1")
OUT_DIR = ROOT / "research" / "out"

BASE_EXIT = "mk0_m2_rr2.5"      # 베이스 거래 순R (메이커, 시장진입) — 사전등록
MODEL_TRIALS = 6                # 모델·임계 선택 자유도 (N 가산용)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """결정시점 가용 피처만 인코딩 (룩어헤드 없음)."""
    X = pd.DataFrame(index=df.index)
    X["score_raw"] = df["score_raw"].astype(float)
    X["kz_true"] = df["kz_true"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    X["zone_both"] = df["zone_both"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    X["is_long"] = (df["direction"] == "long").astype(int)
    X["weekday"] = df["ts"].dt.weekday
    X["hour"] = df["ts"].dt.hour
    ba = df["btc_aligned"].astype(str)
    X["ba_aligned"] = (ba == "aligned").astype(int)
    X["ba_counter"] = (ba == "counter").astype(int)
    # 심볼 원핫 (저카디널리티)
    for s in sorted(df["symbol"].unique()):
        X[f"sym_{s.split('/')[0]}"] = (df["symbol"] == s).astype(int)
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-until", required=True)
    parser.add_argument("--n-prior", type=int, default=841)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    from sklearn.ensemble import GradientBoostingClassifier

    path = OUT_DIR / "a1_signals.csv"
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["btc_aligned"] = wfo._btc_alignment(df)
    df = df.dropna(subset=[BASE_EXIT]).sort_values("ts").reset_index(drop=True)
    df["y"] = (df[BASE_EXIT] > 0).astype(int)

    hs = pd.Timestamp(args.holdout_until, tz="UTC")
    search = df[df["ts"] < hs].reset_index(drop=True)
    hold = df[df["ts"] >= hs].reset_index(drop=True)

    feat_cols = None
    # ── 워크포워드: train 240d → test 45d, 분류기 train에서만 학습 ──
    train_days, test_days, min_train = 240, 45, 300
    t0, t_end = search["ts"].min(), search["ts"].max()
    cursor = t0 + timedelta(days=train_days)
    oos_ts, oos_r_sel, oos_r_all = [], [], []
    thr_list = []
    while cursor < t_end:
        tr = search[(search["ts"] >= cursor - timedelta(days=train_days)) & (search["ts"] < cursor)]
        te = search[(search["ts"] >= cursor) & (search["ts"] < cursor + timedelta(days=test_days))]
        cursor += timedelta(days=test_days)
        if len(tr) < min_train or len(te) == 0:
            continue
        Xtr, Xte = _features(tr), _features(te)
        feat_cols = list(Xtr.columns)
        Xte = Xte.reindex(columns=feat_cols, fill_value=0)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, random_state=0)
        clf.fit(Xtr, tr["y"])
        # 임계 = train 예측확률 중앙값(상위 절반 거래) — 사전등록 규칙
        ptr = clf.predict_proba(Xtr)[:, 1]
        thr = float(np.median(ptr))
        thr_list.append(thr)
        pte = clf.predict_proba(Xte)[:, 1]
        sel = pte >= thr
        rsel = te.loc[sel, BASE_EXIT].to_numpy()
        oos_ts.extend(te.loc[sel, "ts"].tolist())
        oos_r_sel.extend(rsel.tolist())
        oos_r_all.extend(te[BASE_EXIT].tolist())

    rs = np.array(oos_r_sel, dtype=float)
    n_tested = args.n_prior + 1 + MODEL_TRIALS + 1 + 1
    report = {
        "base_exit": BASE_EXIT, "n_search": len(search), "n_holdout": len(hold),
        "oos_selected_n": int(len(rs)),
        "oos_selected_mean_r": round(float(rs.mean()), 4) if len(rs) else 0.0,
        "oos_all_mean_r": round(float(np.mean(oos_r_all)), 4) if oos_r_all else 0.0,
        "selection_rate": round(len(rs) / max(len(oos_r_all), 1), 3),
        "n_tested": n_tested,
    }
    if len(rs) >= 50:
        gate = sg.gate(rs, oos_ts, n_tested=n_tested)
        # 홀드아웃: 전체 search로 학습 → 홀드아웃 예측·선택
        Xall = _features(search).reindex(columns=feat_cols, fill_value=0)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, random_state=0)
        clf.fit(Xall, search["y"])
        thr = float(np.median(clf.predict_proba(Xall)[:, 1]))
        Xho = _features(hold).reindex(columns=feat_cols, fill_value=0)
        pho = clf.predict_proba(Xho)[:, 1]
        hsel = hold.loc[pho >= thr, BASE_EXIT].dropna()
        report["holdout"] = {"n": int(len(hsel)),
                             "mean_r": round(float(hsel.mean()), 4) if len(hsel) else 0.0,
                             "winrate": round(float((hsel > 0).mean()), 4) if len(hsel) else 0.0}
        report["stats_gate_passed"] = gate.passed
        report["stats_gate_reasons"] = gate.reasons
        report["sharpe"] = gate.sharpe
        report["bootstrap"] = gate.bootstrap
        ho_ok = report["holdout"]["mean_r"] > 0 and report["holdout"]["n"] >= 20
        improves = report["oos_selected_mean_r"] > report["oos_all_mean_r"]
        report["holdout_ok"] = ho_ok
        report["improves_over_unfiltered"] = improves
        report["final_verdict"] = "PROMOTE" if (gate.passed and ho_ok and improves) else "REJECT"
    else:
        report["final_verdict"] = "REJECT"
        report["reason"] = "선택 OOS<50"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "d1_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("최종 판정: %s", report["final_verdict"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
