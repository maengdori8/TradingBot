"""
펀딩비 캐리/컨트라리안 — 가격이 아닌 신호. 룩어헤드 없이 검증.

가설: 펀딩이 극단적으로 높다 = 롱이 과밀/과열 → 숏(가격 반전 + 펀딩 수취). 낮다/음수 = 숏 과밀
→ 롱. 횡단면 롱숏으로 베타 중립. 숏은 양(+)펀딩을 수취, 롱은 펀딩 지불 → 캐리 순풍.

수익 = 가격수익(롱 − 숏) + 펀딩수익(숏 수취 − 롱 지불) − 거래비용.

룩어헤드 차단: 랭킹은 t까지 '이미 정산된' 펀딩만. [t, t+H] 가격변화·펀딩은 실현 결과.

사용:
  python3 research/funding.py --download     # 펀딩 히스토리 받기 (1회)
  python3 research/funding.py --cost 0.0007
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import research.study as study  # noqa: E402
from research.altsignals import UNIVERSE  # noqa: E402

logger = logging.getLogger("funding")
OUT_DIR = ROOT / "research" / "out"
FUND_DIR = ROOT / "research" / "data"
HOLDOUT_DAYS = 60


def download_funding(start: str = "2024-01-01") -> None:
    """전 심볼 펀딩 히스토리 페이지네이션 다운로드 → pkl 캐시."""
    import ccxt
    ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    for sym in UNIVERSE:
        cache = FUND_DIR / f"{study._san(sym)}_funding.pkl"
        if cache.exists():
            logger.info("%s 펀딩 캐시 존재", sym)
            continue
        rows, since = [], start_ms
        while True:
            for attempt in range(6):
                try:
                    batch = ex.fetch_funding_rate_history(sym, since=since, limit=200)
                    break
                except ccxt.RateLimitExceeded:
                    time.sleep(0.5 * (2 ** attempt))
            else:
                break
            if not batch:
                break
            rows.extend(batch)
            nxt = batch[-1]["timestamp"] + 1
            if nxt <= since or batch[-1]["timestamp"] >= ex.milliseconds() - 8 * 3600 * 1000:
                break
            since = nxt
            time.sleep(0.15)
        if not rows:
            logger.warning("%s 펀딩 없음", sym)
            continue
        df = pd.DataFrame([(r["timestamp"], float(r["fundingRate"])) for r in rows],
                          columns=["ts", "rate"]).drop_duplicates("ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df.to_pickle(cache)
        logger.info("%s 펀딩 %d건 캐시", sym, len(df))


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """펀딩 패널 + 동일 8h 그리드 가격 패널 (가격은 15m 종가 ffill, 룩어헤드 없음)."""
    fund = {}
    for sym in UNIVERSE:
        cache = FUND_DIR / f"{study._san(sym)}_funding.pkl"
        if cache.exists():
            fund[sym] = pd.read_pickle(cache)["rate"]
    fpanel = pd.DataFrame(fund).dropna(how="all").sort_index()

    # 가격: 각 8h 펀딩 시각에 '그 이하 마지막 15m 종가' (룩어헤드 없음)
    price = {}
    for sym in UNIVERSE:
        try:
            df15 = pd.read_pickle(study.DATA_DIR / f"{study._san(sym)}_15m.pkl")
        except FileNotFoundError:
            continue
        idx = df15.index.searchsorted(fpanel.index, side="right") - 1
        valid = idx >= 0
        ser = pd.Series(np.nan, index=fpanel.index)
        ser.iloc[valid] = df15["close"].to_numpy()[idx[valid]]
        price[sym] = ser
    ppanel = pd.DataFrame(price)

    # 공통 심볼/구간
    common = [s for s in UNIVERSE if s in fpanel.columns and s in ppanel.columns]
    fpanel, ppanel = fpanel[common], ppanel[common]
    mask = ppanel.notna().all(axis=1) & fpanel.notna().all(axis=1)
    return fpanel[mask], ppanel[mask]


def ret_series(fpanel: pd.DataFrame, ppanel: pd.DataFrame, *, lookback: int, hold: int,
               m: int, cost: float, contrarian: bool = True) -> pd.Series:
    """리밸런스별 순수익(가격+펀딩−비용) 시계열. lookback/hold 단위=8h 인터벌."""
    frate = fpanel.to_numpy()
    px = ppanel.to_numpy()
    dates = fpanel.index
    n, k = px.shape
    if k < 2 * m + 1 or n < lookback + hold + 2:
        return pd.Series(dtype=float)
    out = []
    t = lookback
    while t + hold < n:
        sig = frate[t - lookback:t].mean(axis=0)        # 최근 평균 펀딩 (t까지 정산분)
        order = np.argsort(sig)
        if contrarian:                                   # 고펀딩=과열 → 숏 / 저펀딩 → 롱
            longs, shorts = order[:m], order[-m:]
        else:                                            # 모멘텀 방향(추세 추종)
            longs, shorts = order[-m:], order[:m]
        price_ret = (px[t + hold][longs] / px[t][longs] - 1).mean() \
            - (px[t + hold][shorts] / px[t][shorts] - 1).mean()
        # 펀딩 수익: 보유기간(t+1..t+hold) 숏 수취(+) − 롱 지불(+롱펀딩만큼 손실)
        fwin = frate[t + 1:t + hold + 1]
        fund_pnl = fwin[:, shorts].sum() / m - fwin[:, longs].sum() / m
        ret = price_ret + fund_pnl - 2.0 * 2 * cost      # 롱숏 양다리 진입+청산
        out.append((dates[t + hold], ret))
        t += hold
    return pd.Series(dict(out)) if out else pd.Series(dtype=float)


def stats(rs: pd.Series, hold: int) -> dict:
    """수익 시계열 통계 (Sharpe·자본배율·MDD·홀드아웃)."""
    if len(rs) == 0:
        return {"rebalances": 0}
    a = rs.to_numpy()
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    per_year = 365.0 / (hold * 8 / 24)
    sh = float(a.mean() / a.std() * np.sqrt(per_year)) if a.std() > 0 else 0.0
    ho = rs[rs.index >= rs.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)].to_numpy()
    ho_eq = float(np.prod(1 + ho)) if len(ho) else 1.0
    return {"rebalances": int(len(a)), "mean_ret": round(float(a.mean()), 5),
            "winrate": round(float((a > 0).mean()), 4), "sharpe_ann": round(sh, 3),
            "equity_mult_full": round(eq, 3), "mdd_full": round(mdd, 4),
            "holdout_n": int(len(ho)), "holdout_equity": round(ho_eq, 3),
            "holdout_mean": round(float(ho.mean()), 5) if len(ho) else None}


def walk_forward(fp: pd.DataFrame, pp: pd.DataFrame, cost: float, grid: list) -> dict:
    """config도 train(365d)에서만 선택 → 미지 test(90d)에 적용 → 누적 OOS."""
    t0, tend = fp.index.min(), fp.index.max()
    cursor = t0 + pd.Timedelta(days=365)
    oos, chosen = [], []
    while cursor < tend:
        m_tr = (fp.index >= cursor - pd.Timedelta(days=365)) & (fp.index < cursor)
        ftr, ptr = fp[m_tr], pp[m_tr]
        best, best_sh = None, -1e9
        for (lb, h, mm, con) in grid:
            s = stats(ret_series(ftr, ptr, lookback=lb, hold=h, m=mm, cost=cost, contrarian=con), h)
            if s.get("rebalances", 0) >= 8 and s["sharpe_ann"] > best_sh:
                best, best_sh = (lb, h, mm, con), s["sharpe_ann"]
        if best:
            rs = ret_series(fp, pp, lookback=best[0], hold=best[1], m=best[2],
                            cost=cost, contrarian=best[3])
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


def yearly(fp, pp, cfg, cost) -> dict:
    """연도별 분해 (단일 레짐 의존 여부)."""
    rs = ret_series(fp, pp, lookback=cfg[0], hold=cfg[1], m=cfg[2], cost=cost, contrarian=cfg[3])
    return {str(yr): {"rebal": int(len(g)), "equity_mult": round(float(np.prod(1 + g.to_numpy())), 3)}
            for yr, g in rs.groupby(rs.index.year)}


def main() -> None:
    """펀딩 캐리 배터리 + 정직 WFO + 연도별 + 비용민감도."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cost", type=float, default=0.0007)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if args.download:
        download_funding()
        return

    fp, pp = load_panels()
    logger.info("패널: %d심볼 × %d개 8h봉 (%s ~ %s)", fp.shape[1], fp.shape[0],
                fp.index.min(), fp.index.max())

    # lookback/hold 단위 = 8h 인터벌 (3=1일, 9=3일, 21=7일)
    grid = [(lb, h, m, con) for lb in (3, 9, 21, 63) for h in (3, 9, 21)
            for m in (3, 4) for con in (True, False)]
    results = []
    for cfg in grid:
        s = stats(ret_series(fp, pp, lookback=cfg[0], hold=cfg[1], m=cfg[2], cost=args.cost,
                             contrarian=cfg[3]), cfg[1])
        if s.get("rebalances", 0) >= 10:
            s["config"] = f"L{cfg[0]}_H{cfg[1]}_M{cfg[2]}_{'CON' if cfg[3] else 'MOM'}"
            results.append(s)
    results.sort(key=lambda x: -x["sharpe_ann"])
    logger.info("=== 인샘플 상위 6 (Sharpe, cost=%.4f) ===", args.cost)
    for r in results[:6]:
        logger.info("%-18s Sharpe%6.2f 자본x%6.2f MDD%5.1f%% 홀드x%s", r["config"],
                    r["sharpe_ann"], r["equity_mult_full"], r["mdd_full"] * 100, r["holdout_equity"])

    for c in (args.cost, 0.0021):
        wf = walk_forward(fp, pp, c, grid)
        logger.info("=== 정직 WFO cost=%.4f → OOS리밸%s 평균%.5f 승률%.3f 자본x%.2f MDD%.1f%% ===",
                    c, wf.get("oos_rebalances"), wf.get("oos_mean", 0), wf.get("oos_winrate", 0),
                    wf.get("oos_equity", 0), wf.get("oos_mdd", 0) * 100)
        logger.info("   선택: %s", wf.get("chosen"))

    rep = (9, 9, 3, True)
    logger.info("=== 연도별 %s ===", rep)
    for yr, s in yearly(fp, pp, rep, args.cost).items():
        logger.info("   %s: 리밸%d 자본x%.3f", yr, s["rebal"], s["equity_mult"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "funding_battery.json", "w", encoding="utf-8") as f:
        json.dump({"cost": args.cost, "results": results,
                   "wfo_maker": walk_forward(fp, pp, args.cost, grid),
                   "wfo_taker": walk_forward(fp, pp, 0.0021, grid),
                   "yearly_rep": yearly(fp, pp, rep, args.cost)}, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", OUT_DIR / "funding_battery.json")


if __name__ == "__main__":
    main()
