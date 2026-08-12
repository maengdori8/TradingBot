from __future__ import annotations

# 신호 연구 집계 — 레짐 태깅(공포지수/BTC추세/변동성/이벤트일) 후 세그먼트별 성과 통계.
# 출력: research/out/report.json + REPORT.md

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "research" / "data"
OUT_DIR = ROOT / "research" / "out"

logger = logging.getLogger("aggregate")

BASE = "r_m1.5_rr2"          # 현재 봇 설정과 동일한 기준 조합 (ATR×1.5, R:R 2.0)


# ──────────────────────────────────────────────────────────────────────
# 레짐 데이터
# ──────────────────────────────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    """Fear&Greed 전체 역사 (date_str -> int)."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20)
        data = r.json()["data"]
        out = {}
        for d in data:
            day = pd.to_datetime(int(d["timestamp"]), unit="s", utc=True).strftime("%Y-%m-%d")
            out[day] = int(d["value"])
        logger.info("Fear&Greed %d일 수신", len(out))
        return out
    except Exception as e:
        logger.warning("F&G 수신 실패: %s", e)
        return {}


def btc_trend_series() -> pd.Series:
    """BTC 4h EMA50 기반 추세 시리즈 (bull/bear/flat)."""
    df = pd.read_pickle(DATA_DIR / "BTC_USDT-USDT_4h.pkl")
    ema = df["close"].ewm(span=50, adjust=False).mean()
    slope = ema.diff(10)
    cond_bull = (df["close"] > ema) & (slope > 0)
    cond_bear = (df["close"] < ema) & (slope < 0)
    trend = pd.Series("flat", index=df.index)
    trend[cond_bull] = "bull"
    trend[cond_bear] = "bear"
    return trend


def btc_daily_moves() -> pd.DataFrame:
    """BTC 일별 등락/레인지 (이벤트일 프록시: 상위 5% 레인지)."""
    df = pd.read_pickle(DATA_DIR / "BTC_USDT-USDT_4h.pkl")
    daily = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["open"] * 100
    daily["chg_pct"] = (daily["close"] - daily["open"]) / daily["open"] * 100
    thr = daily["range_pct"].quantile(0.95)
    daily["event_day"] = daily["range_pct"] >= thr
    return daily


def tag_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """신호 DataFrame에 레짐 컬럼들을 붙인다."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")

    fng = fetch_fear_greed()
    df["fng"] = df["date"].map(fng)

    trend = btc_trend_series()
    pos = trend.index.searchsorted(df["ts"]) - 1
    pos = pos.clip(0, len(trend) - 1)
    df["btc_trend"] = trend.iloc[pos].to_numpy()
    df["btc_aligned"] = np.select(
        [
            (df["direction"] == "long") & (df["btc_trend"] == "bull"),
            (df["direction"] == "short") & (df["btc_trend"] == "bear"),
            df["btc_trend"] == "flat",
        ],
        ["aligned", "aligned", "flat"],
        default="counter",
    )

    daily = btc_daily_moves()
    df["event_day"] = df["date"].map(
        {d.strftime("%Y-%m-%d"): bool(v) for d, v in daily["event_day"].items()}
    ).fillna(False)

    # 킬존 보정 점수: 킬존 밖 신호는 KZ 가점 15를 제거해 공정 비교
    df["score_eff"] = np.where(df["kz_true"], df["score_raw"], df["score_raw"] - 15)
    return df


# ──────────────────────────────────────────────────────────────────────
# 통계
# ──────────────────────────────────────────────────────────────────────

def _stats(sub: pd.DataFrame, col: str = BASE) -> dict:
    """세그먼트 성과 (n, 승률, 평균순R, 총R)."""
    r = sub[col].dropna()
    if len(r) == 0:
        return {"n": 0}
    wins = (r > 0).sum()
    return {
        "n": int(len(r)),
        "winrate": round(float(wins / len(r)), 3),
        "avg_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 1),
    }


def _by(df: pd.DataFrame, key: str, col: str = BASE) -> dict:
    """key 그룹별 성과."""
    return {str(k): _stats(g, col) for k, g in df.groupby(key, dropna=False)}


def build_report(df: pd.DataFrame) -> dict:
    """전체 리포트 dict 생성."""
    kz = df[df["kz_true"]]                       # 봇 현실 (킬존 내 신호만)
    rep: dict = {
        "meta": {
            "total_signals_24h": int(len(df)),
            "kz_signals": int(len(kz)),
            "symbols": int(df["symbol"].nunique()),
            "period": [str(df["ts"].min()), str(df["ts"].max())],
            "baseline": BASE,
            "note": "순R = 수수료/슬리피지 0.21% 반영. 동시터치=손절 우선(보수적).",
        },
        # 1) 진입 문턱 스윕 (킬존 내, 현재 봇과 동일 조건)
        "min_score_sweep_kz": {
            str(t): _stats(kz[kz["score_raw"] >= t]) for t in (60, 65, 70, 72, 75, 78, 80, 85)
        },
        # 2) 킬존 가치 검증 (보정점수 70+ 동일 조건에서 KZ vs 비KZ)
        "kz_vs_nonkz_score70": {
            "in_kz": _stats(df[(df["kz_true"]) & (df["score_eff"] >= 70)]),
            "out_kz": _stats(df[(~df["kz_true"]) & (df["score_eff"] >= 70)]),
        },
        "by_session": _by(df[df["score_eff"] >= 70], "session"),
        # 3) 방향/레짐 (킬존 내 score70+)
        "by_direction": _by(kz[kz["score_raw"] >= 70], "direction"),
        "by_btc_aligned": _by(kz[kz["score_raw"] >= 70], "btc_aligned"),
        "by_fng_bucket": {},
        "by_vol_bucket": {},
        "by_event_day": _by(kz[kz["score_raw"] >= 70], "event_day"),
        "by_volume_ok": _by(kz[kz["score_raw"] >= 70], "volume_ok"),
        "by_zone_both": _by(kz[kz["score_raw"] >= 70], "zone_both"),
        "by_weekday": _by(kz[kz["score_raw"] >= 70], "weekday"),
        # 4) 손절폭×RR 그리드 (킬존 내 score70+)
        "exit_grid_kz_score70": {},
        # 5) 워크포워드 (전반 vs 후반)
        "walk_forward": {},
    }

    s70 = kz[kz["score_raw"] >= 70]
    fb = pd.cut(s70["fng"], [0, 20, 40, 60, 100],
                labels=["extreme_fear", "fear", "neutral", "greed+"])
    rep["by_fng_bucket"] = {str(k): _stats(g) for k, g in s70.groupby(fb, observed=True)}
    vb = pd.cut(s70["atr_pct_rank"], [0, 0.33, 0.66, 1.0],
                labels=["low_vol", "mid_vol", "high_vol"])
    rep["by_vol_bucket"] = {str(k): _stats(g) for k, g in s70.groupby(vb, observed=True)}

    for c in [c for c in df.columns if c.startswith("r_m")]:
        rep["exit_grid_kz_score70"][c] = _stats(s70, c)

    mid = df["ts"].quantile(0.5)
    first, second = kz[kz["ts"] <= mid], kz[kz["ts"] > mid]
    rep["walk_forward"] = {
        "first_half_score70": _stats(first[first["score_raw"] >= 70]),
        "second_half_score70": _stats(second[second["score_raw"] >= 70]),
        "first_half_score78": _stats(first[first["score_raw"] >= 78]),
        "second_half_score78": _stats(second[second["score_raw"] >= 78]),
    }

    # 심볼 티어 (BTC/ETH/SOL 메이저 vs 나머지)
    majors = {"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"}
    rep["by_symbol_tier"] = {
        "majors": _stats(s70[s70["symbol"].isin(majors)]),
        "alts": _stats(s70[~s70["symbol"].isin(majors)]),
    }
    # 최악/최선 심볼 (n>=10)
    sym_stats = {s: _stats(g) for s, g in s70.groupby("symbol") if len(g) >= 10}
    ranked = sorted(sym_stats.items(), key=lambda kv: kv[1].get("avg_r", 0))
    rep["worst_symbols"] = dict(ranked[:5])
    rep["best_symbols"] = dict(ranked[-5:])

    # 이벤트일(급변동) 상위 목록 → 뉴스 조회용
    daily = btc_daily_moves()
    top_moves = daily.nlargest(10, "range_pct")
    rep["top_move_days"] = [
        {"date": d.strftime("%Y-%m-%d"), "range_pct": round(r["range_pct"], 1),
         "chg_pct": round(r["chg_pct"], 1)}
        for d, r in top_moves.iterrows()
    ]
    return rep


def main() -> dict:
    """집계 실행 + 저장."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_csv(OUT_DIR / "signals.csv")
    df = tag_regimes(df)
    df.to_csv(OUT_DIR / "signals_tagged.csv", index=False)
    rep = build_report(df)
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    logger.info("리포트 저장: %s", OUT_DIR / "report.json")
    return rep


if __name__ == "__main__":
    main()
