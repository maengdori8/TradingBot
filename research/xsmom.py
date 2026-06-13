"""
횡단면 모멘텀 (cross-sectional momentum) — 롱숏 중립 포트폴리오, 룩어헤드 없이 검증.

가설: 과거 L일 수익률 상위 종목은 계속 오르고 하위는 계속 내린다(상대강도). crypto에서
문서화된 엣지. 베타 제거를 위해 상위 M 롱 / 하위 M 숏.

룩어헤드 차단: 리밸런스 시점 t의 랭킹은 [t-L, t] 종가(전부 확정)만 사용. 진입 close[t],
청산 close[t+H]. 비용은 회전율×비용%로 진입·청산 양쪽 반영.

정직: 여러 (L,H,M)을 전구간 + 최근 홀드아웃 60일로 분해 보고. OOS/홀드 양수만 후보.

사용: python3 research/xsmom.py --cost 0.0007
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
from research.altsignals import UNIVERSE  # noqa: E402

logger = logging.getLogger("xsmom")
OUT_DIR = ROOT / "research" / "out"
HOLDOUT_DAYS = 60


def load_daily_panel() -> pd.DataFrame:
    """전 심볼 일봉 종가 패널 (공통 날짜 정렬)."""
    series = {}
    for sym in UNIVERSE:
        try:
            df = pd.read_pickle(study.DATA_DIR / f"{study._san(sym)}_4h.pkl")
        except FileNotFoundError:
            continue
        daily = df["close"].resample("1D").last()
        series[sym] = daily
    panel = pd.DataFrame(series).dropna(how="all")
    return panel.ffill().dropna()  # 공통 구간만


def backtest(panel: pd.DataFrame, lookback: int, hold: int, m: int, cost: float,
             long_short: bool = True) -> dict:
    """L일 모멘텀 랭킹 → 상/하위 M 롱숏, H일 보유. 비용 반영 수익 시계열."""
    closes = panel.to_numpy()
    dates = panel.index
    n_days, n_sym = closes.shape
    if n_sym < 2 * m + 1 or n_days < lookback + hold + 5:
        return {"trades": 0}

    rebal_rets: list[tuple[pd.Timestamp, float]] = []
    t = lookback
    while t + hold < n_days:
        past = closes[t] / closes[t - lookback] - 1.0     # [t-L, t] 수익률 (확정)
        order = np.argsort(past)
        longs = order[-m:]
        shorts = order[:m]
        fwd = closes[t + hold] / closes[t] - 1.0          # 보유기간 실현수익
        if long_short:
            ret = fwd[longs].mean() - fwd[shorts].mean()
            turnover = 2.0                                  # 롱+숏 양다리
        else:
            ret = fwd[longs].mean()
            turnover = 1.0
        ret -= turnover * 2 * cost                          # 진입+청산 비용
        rebal_rets.append((dates[t + hold], ret))
        t += hold

    if not rebal_rets:
        return {"trades": 0}
    rdf = pd.DataFrame(rebal_rets, columns=["date", "ret"]).set_index("date")
    return _stats(rdf, lookback, hold, m, long_short)


def _stats(rdf: pd.DataFrame, lookback: int, hold: int, m: int, ls: bool) -> dict:
    """수익 시계열 → 전구간/홀드아웃 통계."""
    def eq_mdd(rets: np.ndarray) -> tuple[float, float]:
        eq, peak, mdd = 1.0, 1.0, 0.0
        for r in rets:
            eq *= (1 + r)
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak if peak > 0 else 0)
        return eq, mdd

    all_r = rdf["ret"].to_numpy()
    hold_start = rdf.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    ho = rdf[rdf.index >= hold_start]["ret"].to_numpy()
    full_eq, full_mdd = eq_mdd(all_r)
    ho_eq, ho_mdd = eq_mdd(ho) if len(ho) else (1.0, 0.0)
    per_year = 365.0 / hold
    sharpe = float(all_r.mean() / all_r.std() * np.sqrt(per_year)) if all_r.std() > 0 else 0.0
    return {
        "config": f"L{lookback}_H{hold}_M{m}_{'LS' if ls else 'LO'}",
        "rebalances": int(len(all_r)),
        "mean_ret_per_rebal": round(float(all_r.mean()), 5),
        "winrate": round(float((all_r > 0).mean()), 4),
        "sharpe_ann": round(sharpe, 3),
        "equity_mult_full": round(full_eq, 3),
        "mdd_full": round(full_mdd, 4),
        "holdout_n": int(len(ho)),
        "holdout_mean_ret": round(float(ho.mean()), 5) if len(ho) else None,
        "holdout_equity": round(ho_eq, 3),
    }


def ret_series(panel: pd.DataFrame, lookback: int, hold: int, m: int, cost: float,
               long_short: bool) -> pd.Series:
    """리밸런스별 순수익 시계열 (검증용 — 통계 대신 원시 수익)."""
    closes = panel.to_numpy()
    dates = panel.index
    n_days, n_sym = closes.shape
    out = []
    t = lookback
    while t + hold < n_days:
        past = closes[t] / closes[t - lookback] - 1.0
        order = np.argsort(past)
        fwd = closes[t + hold] / closes[t] - 1.0
        if long_short:
            ret = fwd[order[-m:]].mean() - fwd[order[:m]].mean() - 2.0 * 2 * cost
        else:
            ret = fwd[order[-m:]].mean() - 1.0 * 2 * cost
        out.append((dates[t + hold], ret))
        t += hold
    return pd.Series(dict(out)) if out else pd.Series(dtype=float)


def walk_forward_xs(panel: pd.DataFrame, cost: float, grid: list[tuple]) -> dict:
    """config도 train에서만 선택 → 다음 test 구간에 적용 (정직한 OOS).

    매 365일 train에서 Sharpe 최고 config 선택 → 직후 90일 test 수익 누적.
    H가 달라 리밸런스 시점이 config마다 다르므로, test는 '날짜로' 잘라 해당 config의
    test구간 수익만 취한다.
    """
    t0, tend = panel.index.min(), panel.index.max()
    cursor = t0 + pd.Timedelta(days=365)
    oos = []
    chosen = []
    while cursor < tend:
        tr = panel[(panel.index >= cursor - pd.Timedelta(days=365)) & (panel.index < cursor)]
        te_end = cursor + pd.Timedelta(days=90)
        best, best_sh = None, -1e9
        for (lb, h, m, ls) in grid:
            s = backtest(tr, lb, h, m, cost, ls)
            if s.get("rebalances", 0) >= 8 and s["sharpe_ann"] > best_sh:
                best, best_sh = (lb, h, m, ls), s["sharpe_ann"]
        if best:
            rs = ret_series(panel, best[0], best[1], best[2], cost, best[3])
            seg = rs[(rs.index >= cursor) & (rs.index < te_end)]
            oos.extend(seg.tolist())
            chosen.append(best)
        cursor = te_end
    arr = np.array(oos, dtype=float)
    if len(arr) == 0:
        return {"oos_rebalances": 0}
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in arr:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    from collections import Counter
    return {
        "oos_rebalances": int(len(arr)),
        "oos_mean_ret": round(float(arr.mean()), 5),
        "oos_winrate": round(float((arr > 0).mean()), 4),
        "oos_equity_mult": round(float(eq), 3),
        "oos_mdd": round(float(mdd), 4),
        "chosen_freq": {str(k): v for k, v in Counter(map(str, chosen)).most_common(5)},
    }


def yearly(panel: pd.DataFrame, cfg: tuple, cost: float) -> dict:
    """연도별 수익 분해 (견고성 — 단일 레짐 의존 여부)."""
    rs = ret_series(panel, cfg[0], cfg[1], cfg[2], cost, cfg[3])
    out = {}
    for yr, grp in rs.groupby(rs.index.year):
        eq = float(np.prod(1 + grp.to_numpy()))
        out[str(yr)] = {"rebal": int(len(grp)), "equity_mult": round(eq, 3),
                        "mean_ret": round(float(grp.mean()), 5)}
    return out


def main() -> None:
    """배터리 + 정직한 walk-forward(config선택 포함) + 연도별 + 비용민감도."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0007)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    panel = load_daily_panel()
    logger.info("패널: %d심볼 × %d일 (%s ~ %s)", panel.shape[1], panel.shape[0],
                panel.index.min().date(), panel.index.max().date())

    grid = [(lb, h, m, ls) for lb in (7, 14, 30, 90) for h in (7, 14, 30)
            for m in (3, 4) for ls in (True, False)]
    results = []
    for cfg in grid:
        r = backtest(panel, cfg[0], cfg[1], cfg[2], args.cost, cfg[3])
        if r.get("rebalances", 0) >= 10:
            results.append(r)
    results.sort(key=lambda x: -x["sharpe_ann"])

    logger.info("=== 인샘플 상위 6 (Sharpe, cost=%.4f) ===", args.cost)
    for r in results[:6]:
        logger.info("%-16s Sharpe%6.2f 자본x%6.2f MDD%5.1f%%", r["config"],
                    r["sharpe_ann"], r["equity_mult_full"], r["mdd_full"] * 100)

    # 정직한 walk-forward (config도 OOS에서 선택 안 함)
    for c in (args.cost, 0.0021):
        wf = walk_forward_xs(panel, c, grid)
        logger.info("=== WFO(config선택 포함) cost=%.4f → OOS리밸%s 평균%.5f 승률%.3f 자본x%.2f MDD%.1f%% ===",
                    c, wf.get("oos_rebalances"), wf.get("oos_mean_ret", 0),
                    wf.get("oos_winrate", 0), wf.get("oos_equity_mult", 0), wf.get("oos_mdd", 0) * 100)
        logger.info("   선택빈도: %s", wf.get("chosen_freq"))

    # 대표 config 연도별 (견고성)
    rep = (14, 14, 3, True)
    logger.info("=== 연도별 %s (maker) ===", rep)
    for yr, s in yearly(panel, rep, args.cost).items():
        logger.info("   %s: 리밸%d 자본x%.3f 평균%.5f", yr, s["rebal"], s["equity_mult"], s["mean_ret"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "xsmom_battery.json", "w", encoding="utf-8") as f:
        json.dump({"cost": args.cost, "results": results,
                   "wfo_maker": walk_forward_xs(panel, args.cost, grid),
                   "wfo_taker": walk_forward_xs(panel, 0.0021, grid),
                   "yearly_rep": yearly(panel, rep, args.cost)}, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", OUT_DIR / "xsmom_battery.json")


if __name__ == "__main__":
    main()
