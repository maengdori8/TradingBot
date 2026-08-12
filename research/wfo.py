from __future__ import annotations

# legacy ICT WFO 탐색. 자동 승급 근거는 evidence_runner의 고정 후보 WFO만 허용한다.

import argparse
import itertools
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402

logger = logging.getLogger("wfo")
EVIDENCE_STATUS = "legacy_non_evidence"
EVIDENCE_NOTE = "ICT 파라미터 탐색 출력이며 자동 승급 증거로 사용할 수 없습니다."
OUT_DIR = ROOT / "research" / "out"

# 2024-01 이전부터 존재한 고유동성 심볼 (역사 충분)
UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "BNB/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT", "DOT/USDT:USDT", "TRX/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT",
]

HOLDOUT_DAYS = 60        # 어떤 루프도 안 보는 최종 검증 구간


# ──────────────────────────────────────────────────────────────────────
# 데이터 준비 (룩어헤드 차단 재생 — study 재사용)
# ──────────────────────────────────────────────────────────────────────

def prepare_signals(start: str, workers: int) -> pd.DataFrame:
    """start~현재 전 구간 신호 테이블 생성 (캐시 있으면 재사용)."""
    cache = OUT_DIR / "wfo_signals.csv"
    if cache.exists():
        logger.info("기존 wfo_signals.csv 재사용 (재생성하려면 삭제)")
        return _load_and_enrich(cache)

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    months = int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(start, tz="UTC")).days / 30) + 1
    logger.info("데이터 다운로드: %d심볼 × ~%d개월 (start=%s)", len(UNIVERSE), months, start)

    import ccxt
    ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    for i, sym in enumerate(UNIVERSE, 1):
        for tf in ("15m", "1h", "4h"):
            try:
                study.fetch_history(ex, sym, tf, months)
            except Exception as e:
                logger.warning("%s %s 다운로드 실패: %s", sym, tf, e)
        logger.info("다운로드 %d/%d", i, len(UNIVERSE))

    logger.info("신호 재생 (룩어헤드 차단, %d workers)", workers)
    df = study.run_replay(UNIVERSE, workers)   # research/out/signals.csv 생성
    # WFO 전용 사본 저장
    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return _load_and_enrich(cache)


def _load_and_enrich(path: Path) -> pd.DataFrame:
    """CSV 로드 + bool 정규화 + btc_aligned 컬럼 추가."""
    df = pd.read_csv(path, parse_dates=["ts"])
    for col in ("zone_both", "volume_ok", "kz_true"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    df["btc_aligned"] = _btc_alignment(df)
    return df.sort_values("ts").reset_index(drop=True)


def _btc_alignment(df: pd.DataFrame) -> pd.Series:
    """각 신호 시점의 BTC 4H EMA50 추세와 방향 정렬 (aligned/counter/flat)."""
    try:
        btc = pd.read_pickle(study.DATA_DIR / f"{study._san('BTC/USDT:USDT')}_4h.pkl")
    except FileNotFoundError:
        return pd.Series(["flat"] * len(df), index=df.index)
    ema = btc["close"].ewm(span=50, adjust=False).mean()
    slope = ema.diff(10)
    trend = pd.Series("flat", index=btc.index)
    trend[(btc["close"] > ema) & (slope > 0)] = "bull"
    trend[(btc["close"] < ema) & (slope < 0)] = "bear"
    pos = (trend.index.searchsorted(df["ts"]) - 1).clip(0, len(trend) - 1)
    bt = trend.iloc[pos].to_numpy()
    out = np.where(
        ((df["direction"].to_numpy() == "long") & (bt == "bull")) |
        ((df["direction"].to_numpy() == "short") & (bt == "bear")), "aligned",
        np.where(bt == "flat", "flat", "counter"),
    )
    return pd.Series(out, index=df.index)


# ──────────────────────────────────────────────────────────────────────
# 파라미터 공간 + 평가
# ──────────────────────────────────────────────────────────────────────

def exit_columns(df: pd.DataFrame) -> list[str]:
    """존재하는 출구 그리드 컬럼 (r_m{mult}_rr{rr})."""
    return [c for c in df.columns if c.startswith("r_m") and "_rr" in c]


def param_grid(df: pd.DataFrame) -> list[dict]:
    """탐색할 파라미터 조합 (진입 필터 × 출구 기하). 과최적화 방지 위해 핵심 knob만."""
    grid = []
    for min_score, zone_both, btc, exit_col in itertools.product(
        [65, 70, 75, 80],
        [False, True],
        ["any", "no_counter"],
        exit_columns(df),
    ):
        grid.append({"min_score": min_score, "zone_both": zone_both,
                     "btc": btc, "exit_col": exit_col})
    return grid


def _mask(df: pd.DataFrame, p: dict) -> pd.Series:
    """파라미터 진입 필터 마스크."""
    m = df["score_raw"] >= p["min_score"]
    if p["zone_both"]:
        m &= df["zone_both"]
    if p["btc"] == "no_counter":
        m &= df["btc_aligned"] != "counter"
    return m


def evaluate(df: pd.DataFrame, p: dict) -> dict:
    """파라미터의 성과 (해당 출구 컬럼의 순R 분포)."""
    r = df.loc[_mask(df, p), p["exit_col"]].dropna()
    n = len(r)
    if n == 0:
        return {"n": 0, "winrate": 0.0, "mean_r": 0.0, "total_r": 0.0}
    return {
        "n": n,
        "winrate": round(float((r > 0).mean()), 4),
        "mean_r": round(float(r.mean()), 4),
        "total_r": round(float(r.sum()), 2),
    }


# ──────────────────────────────────────────────────────────────────────
# Walk-Forward
# ──────────────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, train_days: int, test_days: int,
                 min_train_trades: int) -> dict:
    """롤링 train→test. train에서만 파라미터 선택, test(미지의 미래)에 적용."""
    grid = param_grid(df)
    t0 = df["ts"].min()
    t_end = df["ts"].max()
    oos_rows: list[pd.Series] = []
    chosen: list[str] = []
    cursor = t0 + timedelta(days=train_days)
    while cursor < t_end:
        tr = df[(df["ts"] >= cursor - timedelta(days=train_days)) & (df["ts"] < cursor)]
        te = df[(df["ts"] >= cursor) & (df["ts"] < cursor + timedelta(days=test_days))]
        cursor += timedelta(days=test_days)
        if len(te) == 0:
            continue
        # 훈련 구간에서 기대값 최대 파라미터 (최소 표본 충족)
        best, best_r = None, -1e9
        for p in grid:
            s = evaluate(tr, p)
            if s["n"] >= min_train_trades and s["mean_r"] > best_r:
                best, best_r = p, s["mean_r"]
        if best is None:
            continue
        # 검증(OOS): 선택된 파라미터를 미지의 test 구간에 적용
        m = _mask(te, best)
        r = te.loc[m, best["exit_col"]].dropna()
        for val in r:
            oos_rows.append(val)
        chosen.append(_pkey(best))

    rs = np.array(oos_rows, dtype=float)
    return {
        "oos": _stats(rs),
        "windows": len(chosen),
        "param_freq": _top_params(chosen),
        "robust_param": _decode(max(set(chosen), key=chosen.count)) if chosen else None,
    }


def _pkey(p: dict) -> str:
    return f"{p['min_score']}|{int(p['zone_both'])}|{p['btc']}|{p['exit_col']}"


def _decode(k: str) -> dict:
    a, b, c, d = k.split("|")
    return {"min_score": int(a), "zone_both": bool(int(b)), "btc": c, "exit_col": d}


def _top_params(keys: list[str]) -> dict:
    from collections import Counter
    return {k: v for k, v in Counter(keys).most_common(5)}


def _stats(rs: np.ndarray) -> dict:
    """순R 배열 통계 + 고정분수 자본곡선 MDD."""
    if len(rs) == 0:
        return {"n": 0}
    eq, peak, mdd = 1.0, 1.0, 0.0
    f = 0.01
    for r in rs:
        eq *= (1 + f * r)
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak if peak > 0 else 0)
    return {
        "n": int(len(rs)),
        "winrate": round(float((rs > 0).mean()), 4),
        "mean_r": round(float(rs.mean()), 4),
        "total_r": round(float(rs.sum()), 2),
        "equity_mult_1pct": round(float(eq), 3),
        "mdd_1pct": round(float(mdd), 4),
    }


# ──────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """WFO 실행: 여러 라운드로 train/표본 조건을 바꿔가며 OOS 최적 탐색 + 최종 홀드아웃."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    df = prepare_signals(args.start, args.workers)
    logger.info("총 신호 %d개 (%s ~ %s)", len(df), df["ts"].min(), df["ts"].max())

    # 최종 홀드아웃 분리 (어떤 탐색도 보지 않음)
    holdout_start = df["ts"].max() - timedelta(days=HOLDOUT_DAYS)
    search_df = df[df["ts"] < holdout_start].reset_index(drop=True)
    holdout_df = df[df["ts"] >= holdout_start].reset_index(drop=True)
    logger.info("탐색 %d / 홀드아웃 %d 신호", len(search_df), len(holdout_df))

    # 여러 라운드 (train 길이 × 최소표본) — '루프'를 정직하게: OOS로 비교만, 홀드아웃은 안 봄
    rounds = [
        {"train_days": 180, "test_days": 30, "min_train_trades": 40},
        {"train_days": 120, "test_days": 30, "min_train_trades": 25},
        {"train_days": 240, "test_days": 45, "min_train_trades": 60},
    ]
    results = []
    for i, rc in enumerate(rounds, 1):
        wf = walk_forward(search_df, **rc)
        wf["config"] = rc
        results.append(wf)
        logger.info("라운드 %d %s → OOS %s", i, rc, wf["oos"])

    # OOS 기대값 최고 라운드 채택
    valid = [r for r in results if r["oos"].get("n", 0) >= 50]
    best = max(valid, key=lambda r: r["oos"]["mean_r"]) if valid else None

    holdout_eval = None
    if best and best["robust_param"]:
        holdout_eval = evaluate(holdout_df, best["robust_param"])

    report = {
        "evidence_status": EVIDENCE_STATUS,
        "evidence_note": EVIDENCE_NOTE,
        "period": [str(df["ts"].min()), str(df["ts"].max())],
        "total_signals": int(len(df)),
        "rounds": results,
        "best_round_oos": best["oos"] if best else None,
        "robust_param": best["robust_param"] if best else None,
        "final_holdout": holdout_eval,
        "verdict": _verdict(best, holdout_eval),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "wfo_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장: %s", OUT_DIR / "wfo_report.json")
    logger.info("판정: %s", report["verdict"])


def _verdict(best: dict | None, holdout: dict | None) -> str:
    """정직한 판정: OOS와 홀드아웃 모두 양수여야 '엣지 있음'."""
    if not best or not holdout:
        return "표본 부족 — 판정 불가"
    oos_ok = best["oos"]["mean_r"] > 0
    ho_ok = holdout.get("mean_r", -1) > 0 and holdout.get("n", 0) >= 20
    if oos_ok and ho_ok:
        return (f"엣지 있음(잠정): OOS {best['oos']['mean_r']:+.3f}R / "
                f"홀드아웃 {holdout['mean_r']:+.3f}R, 승률 {holdout['winrate']:.1%}")
    if oos_ok and not ho_ok:
        return (f"OOS 양수({best['oos']['mean_r']:+.3f}R)이나 홀드아웃 미확인 "
                f"({holdout.get('mean_r')}R, n={holdout.get('n')}) — 견고성 의심")
    return f"엣지 없음: OOS {best['oos']['mean_r']:+.3f}R — 과최적화 없이 수익 구조 미발견"


if __name__ == "__main__":
    main()
