from __future__ import annotations

# 대체 전략 배터리 — 여러 전략×파라미터를 14심볼 전체에 재생 후 WFO로 OOS 평가.
# 각 config는 룩어헤드 없이 재생하지만 legacy_non_evidence 탐색 결과다.
# 사용: python3 research/alt_driver.py --cost 0.0007 --workers 6

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "research" / "out"
logger = logging.getLogger("alt_driver")

# 전략별 표준 파라미터 (소수 — 과최적화 방지)
CONFIGS = [
    ("donchian", 96), ("donchian", 192), ("donchian", 384),
    ("tsmom", 100), ("tsmom", 200),
    ("meanrev", 48), ("meanrev", 96),
    ("bbreak", 96), ("bbreak", 192),
]

HOLDOUT_DAYS = 60


def evaluate_df(df: pd.DataFrame) -> dict:
    """alt 신호 df를 WFO로 평가 (wfo.py 재사용)."""
    import research.wfo as wfo
    if len(df) < 200:
        return {"signals": int(len(df)), "verdict": "표본부족"}
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("zone_both",):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    df["btc_aligned"] = wfo._btc_alignment(df)
    df = df.sort_values("ts").reset_index(drop=True)

    # 전체표본 최선 출구
    cols = wfo.exit_columns(df)
    best_full = max(cols, key=lambda c: df[c].dropna().mean())
    rfull = df[best_full].dropna()

    # 홀드아웃 분리 + WFO (탐색 구간만)
    holdout_start = df["ts"].max() - timedelta(days=HOLDOUT_DAYS)
    search = df[df["ts"] < holdout_start].reset_index(drop=True)
    hold = df[df["ts"] >= holdout_start].reset_index(drop=True)
    wf = wfo.walk_forward(search, train_days=180, test_days=30, min_train_trades=30)
    holdout_eval = None
    if wf.get("robust_param"):
        holdout_eval = wfo.evaluate(hold, wf["robust_param"])

    return {
        "signals": int(len(df)),
        "period": [str(df["ts"].min()), str(df["ts"].max())],
        "best_full_exit": best_full,
        "best_full_meanR": round(float(rfull.mean()), 4),
        "best_full_WR": round(float((rfull > 0).mean()), 4),
        "oos": wf.get("oos"),
        "robust_param": wf.get("robust_param"),
        "holdout": holdout_eval,
    }


def main() -> None:
    """전 config 순차 재생 + 평가 → alt_battery.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", default="0.0007")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    os.environ["STUDY_COST_PCT"] = args.cost
    import research.altsignals as alt

    results = []
    for strat, param in CONFIGS:
        os.environ["ALT_STRAT"] = strat
        os.environ["ALT_PARAM"] = str(param)
        logger.info("=== %s(%d) 재생 (cost=%s) ===", strat, param, args.cost)
        df = alt.run(alt.UNIVERSE, args.workers)
        ev = evaluate_df(df)
        ev["strategy"] = f"{strat}_{param}"
        results.append(ev)
        oos = ev.get("oos") or {}
        ho = ev.get("holdout") or {}
        logger.info("%s → 신호%s 전체최선 %s %+.4fR | OOS %s %+.4fR | 홀드 n%s %+.4fR",
                    ev["strategy"], ev.get("signals"), ev.get("best_full_exit"),
                    ev.get("best_full_meanR", 0), oos.get("n"), oos.get("mean_r", 0),
                    ho.get("n"), ho.get("mean_r", 0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "alt_battery.json", "w", encoding="utf-8") as f:
        json.dump({"cost": args.cost, "results": results}, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장: %s", OUT_DIR / "alt_battery.json")

    # 정직 요약: OOS와 홀드아웃 둘 다 양수인 것만 후보
    cand = [r for r in results
            if (r.get("oos") or {}).get("mean_r", -1) > 0
            and (r.get("holdout") or {}).get("mean_r", -1) > 0
            and (r.get("holdout") or {}).get("n", 0) >= 20]
    logger.info("=== 엣지 후보(OOS>0 & 홀드아웃>0): %s ===",
                [r["strategy"] for r in cand] or "없음")


if __name__ == "__main__":
    main()
