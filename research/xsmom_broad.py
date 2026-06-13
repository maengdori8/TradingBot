"""
광범위 유니버스 횡단면 모멘텀 — 중·소형주는 덜 효율적이라 모멘텀 엣지가 강할 수 있다.

14 메이저 + 44 중형 크립토 = 58종목. 상장일이 달라 NaN 인지 랭킹(데이터 유효 종목만).
현실적 비용(중형주 스프레드 반영) + 정직 WFO(config도 train선택) + 연도별.

룩어헤드 차단: 랭킹은 t까지 종가만. 진입 close[t], 청산 close[t+H]. NaN인 종목(미상장/
데이터없음)은 그 시점 랭킹에서 제외 = 미래 상장정보 안 씀.

사용: python3 research/xsmom_broad.py --cost 0.0010
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
import research.study as study  # noqa: E402
from research.altsignals import UNIVERSE as MAJORS  # noqa: E402

logger = logging.getLogger("xsmom_broad")
OUT_DIR = ROOT / "research" / "out"
HOLDOUT_DAYS = 60

MIDS = ['SUI', 'APT', 'ARB', 'OP', 'INJ', 'SEI', 'WLD', 'ORDI', 'WIF', '1000PEPE', 'BCH',
        'XLM', 'TON', 'AAVE', 'CRV', 'RENDER', 'HBAR', 'XMR', 'ZEC', 'ENA', 'ONDO', 'TAO',
        'WAVES', 'ENJ', 'ORCA', 'STG', 'VIRTUAL', 'HYPE', 'TRUMP', 'FARTCOIN', 'HMSTR',
        'WLFI', 'PUMPFUN', 'APE', 'GALA', 'FIL', 'ETC', 'UNI', 'SAND', 'MANA', 'AXS',
        'GRT', 'ALGO', 'FLOW']


def load_panel() -> pd.DataFrame:
    """전 종목 일봉 종가 패널 (NaN 허용 — 상장 전은 NaN)."""
    series = {}
    syms = list(MAJORS) + [f"{s}/USDT:USDT" for s in MIDS]
    for sym in syms:
        cache = study.DATA_DIR / f"{study._san(sym)}_4h.pkl"
        if not cache.exists():
            continue
        df = pd.read_pickle(cache)
        series[sym] = df["close"].resample("1D").last()
    panel = pd.DataFrame(series).sort_index()
    # 전 종목 합쳐 최소 한 종목이라도 있는 구간
    return panel.dropna(how="all")


def ret_series(panel: pd.DataFrame, lookback: int, hold: int, m: int, cost: float,
               long_short: bool, min_valid: int) -> pd.Series:
    """NaN 인지 랭킹 횡단면 모멘텀. 유효종목 ≥ min_valid일 때만 거래."""
    closes = panel.to_numpy()
    dates = panel.index
    n, k = closes.shape
    out = []
    t = lookback
    while t + hold < n:
        p0, pl, ph = closes[t], closes[t - lookback], closes[t + hold]
        past = p0 / pl - 1.0
        valid = np.isfinite(past) & np.isfinite(ph) & np.isfinite(p0)
        idx = np.where(valid)[0]
        if len(idx) >= max(min_valid, 2 * m):
            ranked = idx[np.argsort(past[idx])]
            longs = ranked[-m:] if long_short else ranked[-m:]
            shorts = ranked[:m]
            fwd = ph / p0 - 1.0
            if long_short:
                ret = fwd[longs].mean() - fwd[shorts].mean() - 2.0 * 2 * cost
            else:
                ret = fwd[longs].mean() - 1.0 * 2 * cost
            out.append((dates[t + hold], ret))
        t += hold
    return pd.Series(dict(out)) if out else pd.Series(dtype=float)


def stats(rs: pd.Series, hold: int) -> dict:
    if len(rs) == 0:
        return {"rebalances": 0}
    a = rs.to_numpy()
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    per_year = 365.0 / hold
    sh = float(a.mean() / a.std() * np.sqrt(per_year)) if a.std() > 0 else 0.0
    ho = rs[rs.index >= rs.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)].to_numpy()
    return {"rebalances": int(len(a)), "mean_ret": round(float(a.mean()), 5),
            "winrate": round(float((a > 0).mean()), 4), "sharpe_ann": round(sh, 3),
            "equity_mult": round(eq, 3), "mdd": round(mdd, 4),
            "holdout_equity": round(float(np.prod(1 + ho)), 3) if len(ho) else None,
            "holdout_mean": round(float(ho.mean()), 5) if len(ho) else None}


def walk_forward(panel, cost, grid, min_valid) -> dict:
    t0, tend = panel.index.min(), panel.index.max()
    cursor = t0 + pd.Timedelta(days=365)
    oos, chosen = [], []
    while cursor < tend:
        tr = panel[(panel.index >= cursor - pd.Timedelta(days=365)) & (panel.index < cursor)]
        best, best_sh = None, -1e9
        for (lb, h, m, ls) in grid:
            s = stats(ret_series(tr, lb, h, m, cost, ls, min_valid), h)
            if s.get("rebalances", 0) >= 8 and s["sharpe_ann"] > best_sh:
                best, best_sh = (lb, h, m, ls), s["sharpe_ann"]
        if best:
            rs = ret_series(panel, best[0], best[1], best[2], cost, best[3], min_valid)
            seg = rs[(rs.index >= cursor) & (rs.index < cursor + pd.Timedelta(days=90))]
            oos.extend(seg.tolist()); chosen.append(best)
        cursor += pd.Timedelta(days=90)
    a = np.array(oos, dtype=float)
    if len(a) == 0:
        return {"oos_rebalances": 0}
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    from collections import Counter
    return {"oos_rebalances": int(len(a)), "oos_mean": round(float(a.mean()), 5),
            "oos_winrate": round(float((a > 0).mean()), 4), "oos_equity": round(eq, 3),
            "oos_mdd": round(mdd, 4), "chosen": {str(k): v for k, v in Counter(map(str, chosen)).most_common(5)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0010)
    parser.add_argument("--min-valid", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    panel = load_panel()
    cov = panel.notna().sum(axis=1)
    logger.info("패널: %d종목 × %d일 (%s ~ %s), 유효종목수 중앙값 %d",
                panel.shape[1], panel.shape[0], panel.index.min().date(),
                panel.index.max().date(), int(cov.median()))

    grid = [(lb, h, m, ls) for lb in (7, 14, 30, 60) for h in (7, 14, 30)
            for m in (3, 5, 8) for ls in (True, False)]
    results = []
    for cfg in grid:
        s = stats(ret_series(panel, cfg[0], cfg[1], cfg[2], args.cost, cfg[3], args.min_valid), cfg[1])
        if s.get("rebalances", 0) >= 10:
            s["config"] = f"L{cfg[0]}_H{cfg[1]}_M{cfg[2]}_{'LS' if cfg[3] else 'LO'}"
            results.append(s)
    results.sort(key=lambda x: -x["sharpe_ann"])
    logger.info("=== 인샘플 상위 8 (Sharpe, cost=%.4f) ===", args.cost)
    for r in results[:8]:
        logger.info("%-16s Sharpe%6.2f 자본x%7.2f MDD%5.1f%% 홀드x%s WR%.2f", r["config"],
                    r["sharpe_ann"], r["equity_mult"], r["mdd"] * 100, r["holdout_equity"], r["winrate"])

    for c in (args.cost, 0.0021):
        wf = walk_forward(panel, c, grid, args.min_valid)
        logger.info("=== 정직 WFO cost=%.4f → OOS %s리밸 평균%.5f 승률%.3f 자본x%.2f MDD%.1f%% ===",
                    c, wf.get("oos_rebalances"), wf.get("oos_mean", 0), wf.get("oos_winrate", 0),
                    wf.get("oos_equity", 0), wf.get("oos_mdd", 0) * 100)
        logger.info("   선택: %s", wf.get("chosen"))

    rep = (14, 14, 5, True)
    rs = ret_series(panel, rep[0], rep[1], rep[2], args.cost, rep[3], args.min_valid)
    logger.info("=== 연도별 %s ===", rep)
    for yr, g in rs.groupby(rs.index.year):
        logger.info("   %s: 리밸%d 자본x%.3f", yr, len(g), float(np.prod(1 + g.to_numpy())))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "xsmom_broad.json", "w", encoding="utf-8") as f:
        json.dump({"cost": args.cost, "n_symbols": int(panel.shape[1]), "results": results[:15],
                   "wfo_cost": walk_forward(panel, args.cost, grid, args.min_valid),
                   "wfo_taker": walk_forward(panel, 0.0021, grid, args.min_valid)},
                  f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", OUT_DIR / "xsmom_broad.json")


if __name__ == "__main__":
    main()
