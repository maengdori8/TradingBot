from __future__ import annotations

# legacy_non_evidence 저회전 펀딩 하베스터 탐색.
# 자동 승급 근거는 research.evidence_runner만 생성한다.

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import research.funding as fv1  # noqa: E402  (load_panels 재사용)

logger = logging.getLogger("funding_v2")
EVIDENCE_STATUS = "legacy_non_evidence"
EVIDENCE_NOTE = "탐색용 동적 그리드 출력이며 자동 승급 증거로 사용할 수 없습니다."
OUT_DIR = ROOT / "research" / "out"
HOLDOUT_DAYS = 60


def simulate(fp: pd.DataFrame, pp: pd.DataFrame, *, lookback: int, rebal: int, m: int,
             cost: float, wmode: str = "equal", price_neutral: bool = False) -> pd.Series:
    """연속 북 시뮬 → 인터벌별 순손익 시계열(자본 대비 비율). 인덱스=시각."""
    frate = fp.to_numpy()
    px = pp.to_numpy()
    dates = fp.index
    n, k = px.shape
    if k < 2 * m + 1 or n < lookback + rebal + 2:
        return pd.Series(dtype=float)

    pos = np.zeros(k)
    rets = []
    idx = []
    for t in range(lookback, n - 1):
        # 리밸런스: 목표 가중치 재계산 (회전율만큼 비용)
        cost_t = 0.0
        if (t - lookback) % rebal == 0:
            sig = frate[t - lookback:t].mean(axis=0)       # t까지 정산 펀딩
            order = np.argsort(sig)
            longs, shorts = order[:m], order[-m:]           # 저펀딩 롱 / 고펀딩 숏
            target = np.zeros(k)
            if wmode == "fund":                              # 펀딩 크기 비례 가중
                ls = np.abs(sig[longs]); ss = np.abs(sig[shorts])
                target[longs] = -(ls / ls.sum()) if ls.sum() > 0 else -1.0 / m
                target[shorts] = (ss / ss.sum()) if ss.sum() > 0 else 1.0 / m
                # 부호 보정: 롱=+, 숏=−
                target[longs] = (ls / ls.sum()) if ls.sum() > 0 else 1.0 / m
                target[shorts] = -((ss / ss.sum()) if ss.sum() > 0 else 1.0 / m)
            else:                                            # 등가중
                target[longs] = 1.0 / m
                target[shorts] = -1.0 / m
            cost_t = np.abs(target - pos).sum() * cost
            pos = target

        # 보유 손익 (다음 인터벌 t→t+1)
        ret_px = px[t + 1] / px[t] - 1.0
        price_pnl = float((pos * ret_px).sum())
        fund_pnl = float(-(pos * frate[t + 1]).sum())        # 롱 지불 / 숏 수취
        total = fund_pnl - cost_t + (0.0 if price_neutral else price_pnl)
        rets.append(total)
        idx.append(dates[t + 1])
    return pd.Series(rets, index=pd.DatetimeIndex(idx))


def stats(rs: pd.Series) -> dict:
    """수익 시계열 → Sharpe(연율)·자본배율·MDD·홀드아웃."""
    if len(rs) == 0:
        return {"intervals": 0}
    a = rs.to_numpy()
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    per_year = 365.0 * 3      # 8h 인터벌 → 연 1095
    sh = float(a.mean() / a.std() * np.sqrt(per_year)) if a.std() > 0 else 0.0
    ho = rs[rs.index >= rs.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)].to_numpy()
    ho_eq = float(np.prod(1 + ho)) if len(ho) else 1.0
    return {"intervals": int(len(a)), "ann_ret": round(float(a.mean() * per_year), 4),
            "sharpe_ann": round(sh, 3), "equity_mult": round(eq, 3), "mdd": round(mdd, 4),
            "holdout_equity": round(ho_eq, 3),
            "holdout_ann": round(float(ho.mean() * per_year), 4) if len(ho) else None}


def walk_forward(fp, pp, cost, grid) -> dict:
    """config도 train(365d)에서만 선택 → 미지 test(90d) 누적 OOS."""
    t0, tend = fp.index.min(), fp.index.max()
    cursor = t0 + pd.Timedelta(days=365)
    oos, chosen = [], []
    while cursor < tend:
        m_tr = (fp.index >= cursor - pd.Timedelta(days=365)) & (fp.index < cursor)
        ftr, ptr = fp[m_tr], pp[m_tr]
        best, best_sh = None, -1e9
        for (lb, r, mm, wm) in grid:
            s = stats(simulate(ftr, ptr, lookback=lb, rebal=r, m=mm, cost=cost, wmode=wm))
            if s.get("intervals", 0) >= 40 and s["sharpe_ann"] > best_sh:
                best, best_sh = (lb, r, mm, wm), s["sharpe_ann"]
        if best:
            rs = simulate(fp, pp, lookback=best[0], rebal=best[1], m=best[2], cost=cost, wmode=best[3])
            seg = rs[(rs.index >= cursor) & (rs.index < cursor + pd.Timedelta(days=90))]
            oos.extend(seg.tolist()); chosen.append(best)
        cursor += pd.Timedelta(days=90)
    a = np.array(oos, dtype=float)
    if len(a) == 0:
        return {"oos_intervals": 0}
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    from collections import Counter
    per_year = 365.0 * 3
    return {"oos_intervals": int(len(a)), "oos_ann": round(float(a.mean() * per_year), 4),
            "oos_sharpe": round(float(a.mean() / a.std() * np.sqrt(per_year)), 3) if a.std() > 0 else 0,
            "oos_equity": round(eq, 3), "oos_mdd": round(mdd, 4),
            "chosen": {str(k): v for k, v in Counter(map(str, chosen)).most_common(5)}}


def yearly(fp, pp, cfg, cost) -> dict:
    rs = simulate(fp, pp, lookback=cfg[0], rebal=cfg[1], m=cfg[2], cost=cost, wmode=cfg[3])
    return {str(yr): {"n": int(len(g)), "equity_mult": round(float(np.prod(1 + g.to_numpy())), 3)}
            for yr, g in rs.groupby(rs.index.year)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    fp, pp = fv1.load_panels()
    logger.info("패널: %d심볼 × %d 8h봉 (%s ~ %s)", fp.shape[1], fp.shape[0],
                fp.index.min(), fp.index.max())

    # L(랭킹 lookback, 8h단위) × R(리밸 간격) × M(다리수) × wmode
    grid = [(lb, r, m, wm) for lb in (3, 9, 21, 63) for r in (3, 9, 21, 63)
            for m in (2, 3, 4) for wm in ("equal", "fund") if r >= 3]
    results = []
    for cfg in grid:
        s = stats(simulate(fp, pp, lookback=cfg[0], rebal=cfg[1], m=cfg[2], cost=args.cost, wmode=cfg[3]))
        if s.get("intervals", 0) >= 40:
            s["config"] = f"L{cfg[0]}_R{cfg[1]}_M{cfg[2]}_{cfg[3]}"
            results.append(s)
    results.sort(key=lambda x: -x["sharpe_ann"])
    logger.info("=== 인샘플 상위 8 (Sharpe, cost=%.4f) ===", args.cost)
    for r in results[:8]:
        logger.info("%-20s Sharpe%6.2f 연%6.1f%% 자본x%6.2f MDD%5.1f%% 홀드x%s", r["config"],
                    r["sharpe_ann"], r["ann_ret"] * 100, r["equity_mult"], r["mdd"] * 100, r["holdout_equity"])

    # 순수 캐리(가격중립) 참고 — 구조적 엣지 크기
    pn = stats(simulate(fp, pp, lookback=9, rebal=21, m=3, cost=args.cost, wmode="equal", price_neutral=True))
    logger.info("순수캐리(price_neutral) L9_R21_M3: 연%.1f%% Sharpe%.2f 자본x%.2f",
                pn["ann_ret"] * 100, pn["sharpe_ann"], pn["equity_mult"])

    for c in (args.cost, 0.0021):
        wf = walk_forward(fp, pp, c, grid)
        logger.info("=== 정직 WFO cost=%.4f → OOS %s인터벌 연%.1f%% Sharpe%.2f 자본x%.2f MDD%.1f%% ===",
                    c, wf.get("oos_intervals"), wf.get("oos_ann", 0) * 100, wf.get("oos_sharpe", 0),
                    wf.get("oos_equity", 0), wf.get("oos_mdd", 0) * 100)
        logger.info("   선택: %s", wf.get("chosen"))

    rep = (9, 21, 3, "equal")
    logger.info("=== 연도별 %s ===", rep)
    for yr, s in yearly(fp, pp, rep, args.cost).items():
        logger.info("   %s: %d인터벌 자본x%.3f", yr, s["n"], s["equity_mult"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "funding_v2.json", "w", encoding="utf-8") as f:
        json.dump({"evidence_status": EVIDENCE_STATUS,
                   "evidence_note": EVIDENCE_NOTE,
                   "cost": args.cost, "results": results[:15],
                   "wfo_maker": walk_forward(fp, pp, args.cost, grid),
                   "wfo_taker": walk_forward(fp, pp, 0.0021, grid),
                   "yearly_rep": yearly(fp, pp, rep, args.cost)}, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", OUT_DIR / "funding_v2.json")


if __name__ == "__main__":
    main()
