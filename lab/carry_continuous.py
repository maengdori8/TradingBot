"""연속보유 캐리 — 강제 월별 청산 없이 실제 캐리북 운용 방식 재현."""
from __future__ import annotations
import numpy as np, pandas as pd
from lab.carry_falsifier import load, series, px_at, FEE

def continuous(sym, perp, spot, fund, exec_mode='maker', slip_bp=0.0,
               t_start=None, t_end=None):
    """진입 1회 → 만기까지 보유. 8h 펀딩 캐시플로 시계열 반환(명목 대비)."""
    P, S = series(perp, sym), series(spot, sym)
    f = fund[sym].dropna()
    lo = max(P.index[0], S.index[0], f.index[0])
    t0 = max(pd.Timestamp(t_start, tz='utc'), lo) if t_start is not None else lo
    t1 = min(pd.Timestamp(t_end, tz='utc'), P.index[-1], S.index[-1]) if t_end is not None else min(P.index[-1], S.index[-1])
    p0, s0 = px_at(P, t0), px_at(S, t0)
    p1, s1 = px_at(P, t1), px_at(S, t1)
    if any(np.isnan(x) for x in (p0, s0, p1, s1)): return None
    fe, slip = FEE[exec_mode], slip_bp / 1e4
    entry_fee = s0 * (fe['spot'] + slip) + p0 * (fe['perp'] + slip)
    exit_fee = s1 * (fe['spot'] + slip) + p1 * (fe['perp'] + slip)

    fw = f[(f.index > t0) & (f.index <= t1)]
    ts, cf = [], []
    for t, r in fw.items():
        px = px_at(P, t)
        if np.isnan(px): continue
        ts.append(t); cf.append(r * px / s0)          # 명목 대비 8h 캐시플로
    fs = pd.Series(cf, index=pd.DatetimeIndex(ts))
    price_pnl = ((s1 - s0) + (p0 - p1)) / s0
    return dict(sym=sym, t0=t0, t1=t1, funding=fs,
                fund_total=float(fs.sum()), price_pnl=price_pnl,
                entry_fee=entry_fee / s0, exit_fee=exit_fee / s0,
                total=float(fs.sum()) + price_pnl - (entry_fee + exit_fee) / s0)

if __name__ == '__main__':
    perp, spot, fund = load()
    print("=" * 78)
    print("연속보유 캐시앤캐리 — 진입/청산 각 1회 (실제 캐리북 방식)")
    print("=" * 78)
    for mode in ('taker', 'maker'):
        res = [continuous(s, perp, spot, fund, exec_mode=mode) for s in ('BTC', 'ETH')]
        res = [r for r in res if r]
        for r in res:
            yrs = (r['t1'] - r['t0']).days / 365.25
            print(f"\n {r['sym']} {mode}: {r['t0'].date()}~{r['t1'].date()} ({yrs:.2f}y)")
            print(f"   펀딩누적 {r['fund_total']*100:+.2f}%  가격/베이시스 {r['price_pnl']*100:+.3f}%  "
                  f"비용 {(r['entry_fee']+r['exit_fee'])*100:.3f}%")
            print(f"   순합계 {r['total']*100:+.2f}%  →  명목기준 연율 {(((1+r['total'])**(1/yrs))-1)*100:+.2f}%")
            f8 = r['funding']
            ann = f8.resample('YE').sum()
            print("   연도별 펀딩: " + "  ".join(f"{i.year}:{v*100:+.2f}%" for i, v in ann.items()))
            neg = f8.resample('ME').sum()
            print(f"   월별 펀딩 음수비율 {(neg<0).mean()*100:.1f}%  최악월 {neg.min()*100:+.2f}%  최고월 {neg.max()*100:+.2f}%")
