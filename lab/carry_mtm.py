"""정직한 일별 시가평가 캐리 — 펀딩 + 베이시스 변동 모두 반영.

일별 P&L(명목 대비) = funding_t + (basis_{t-1} - basis_t)
  basis = (P - S)/S.  숏 perp이므로 베이시스 축소가 이익.
이전 버전은 펀딩만 반영해 변동성을 과소평가(Sharpe 11.93)했다.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from lab.carry_falsifier import load, series

def daily_mtm(sym, perp, spot, fund):
    P, S = series(perp, sym), series(spot, sym)
    idx = P.index.intersection(S.index)
    P, S = P.loc[idx], S.loc[idx]
    basis = ((P - S) / S).resample('D').last()
    f = fund[sym].dropna().resample('D').sum()
    df = pd.concat({'basis': basis, 'f': f}, axis=1).dropna(subset=['basis'])
    df['f'] = df['f'].fillna(0.0)
    df['carry'] = df['f'] - df['basis'].diff()       # 숏 perp: 베이시스 감소 = 이익
    return df.dropna()

def portfolio(syms, perp, spot, fund):
    parts = {s: daily_mtm(s, perp, spot, fund) for s in syms}
    car = pd.DataFrame({s: p['carry'] for s, p in parts.items()}).dropna(how='all')
    fnd = pd.DataFrame({s: p['f'] for s, p in parts.items()}).reindex(car.index)
    bas = pd.DataFrame({s: p['basis'] for s, p in parts.items()}).reindex(car.index)
    return car.mean(axis=1), fnd.mean(axis=1), bas.mean(axis=1)

def stats(r, label, cash_yield=0.0, deployed=1.0):
    r = r.dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    cagr = float(eq.iloc[-1]) ** (1 / yrs) - 1
    mdd = float((1 - eq / eq.cummax()).max())
    vol = r.std() * np.sqrt(365)
    sharpe = (r.mean() * 365) / vol if vol > 0 else np.nan
    tot = cagr + cash_yield * (1 - deployed)
    print(f"\n── {label}")
    print(f"   CAGR {cagr*100:+.2f}%  변동성 {vol*100:.2f}%  Sharpe {sharpe:.2f}  MDD {mdd*100:.2f}%  ({yrs:.1f}y)")
    if deployed < 1.0:
        print(f"   투입비율 {deployed*100:.1f}% → 현금수익 {cash_yield*100:.1f}% 가산 시 총 {tot*100:+.2f}%")
    print(f"   일별: 최악 {r.min()*100:+.3f}%  최고 {r.max()*100:+.3f}%  양수일 {(r>0).mean()*100:.1f}%")
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    print("   연도별: " + "  ".join(f"{y}:{v*100:+.2f}%" for y, v in yr.items()))
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd)

if __name__ == '__main__':
    perp, spot, fund = load()
    syms = ['BTC', 'ETH']
    car, fnd, bas = portfolio(syms, perp, spot, fund)
    print("=" * 78)
    print("정직한 일별 시가평가 캐리 (펀딩 + 베이시스 변동, BTC/ETH 등가중)")
    print("=" * 78)
    stats(car, "무게이트 연속보유 (비용 제외)")
    print(f"\n   [분해] 펀딩 연율 {fnd.mean()*365*100:+.2f}%  |  "
          f"베이시스 일변동 표준편차 {bas.diff().std()*100:.4f}%  "
          f"→ 연율 변동성 기여 {bas.diff().std()*np.sqrt(365)*100:.2f}%")
    print(f"   [베이시스] 평균 {bas.mean()*100:+.4f}%  최대 {bas.max()*100:+.3f}%  최소 {bas.min()*100:+.3f}%")

    # 게이트 재평가 — 정직한 변동성 하에서
    print("\n" + "=" * 78); print("레짐 게이트 재평가 (정직한 MTM + 현금수익 4% 가산)"); print("=" * 78)
    trail = fnd.rolling(30).mean().shift(1) * 365
    for h in (0.0, 0.03, 0.05, 0.08):
        pos = (trail > h).astype(float)
        # 최소보유 14일 히스테리시스
        pos = pos.where(pos.diff().fillna(0) == 0).ffill().fillna(pos)
        cost = pos.diff().abs().fillna(0) * (0.0022 / 2)
        r = pos * car - cost
        stats(r, f"허들 {h*100:.0f}%/yr", cash_yield=0.04, deployed=float(pos.mean()))
