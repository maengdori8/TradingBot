"""공격 트랙(Track B) — 고전 터틀 추세추종, perp 일봉, 페이퍼 전용.

정직성 원칙: 파라미터는 출판된 터틀 시스템 그대로(S1: 20일 진입/10일 청산,
S2: 55/20, 스탑 2xATR(20), 유닛리스크 고정). 튜닝 없음. 좋아 보일 때까지
반복하는 것은 p-해킹이므로 1회 실행 결과를 그대로 보고한다.

전제: 방향성 엣지는 검증된 바 없다. 목적은 '보이는 P&L'의 분포를 정직하게
보여주는 것이다.
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')

ENTRY_COST, EXIT_COST = 0.0004, 0.0008

def load_daily(syms=('BTC','ETH','SOL')):
    p = pd.read_parquet('lab/frozen/perp_1d.parquet')
    out = {}
    for s in syms:
        d = p.xs(s, level='sym')[['open','high','low','close']]
        out[s] = d[~d.index.duplicated()].sort_index()
    return out

def atr(d, n=20):
    tr = pd.concat([d.high-d.low, (d.high-d.close.shift()).abs(),
                    (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def run(entry_n=20, exit_n=10, risk_pct=0.02, gross_cap=3.0, daily_stop=-0.05,
        heat_cap=0.04, syms=('BTC','ETH','SOL')):
    data = load_daily(syms)
    fund = pd.read_parquet('lab/frozen/funding.parquet')[list(syms)].resample('D').sum(min_count=1)
    idx = None
    for s in syms:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()
    D = {s: data[s].reindex(idx) for s in syms}
    A = {s: atr(D[s]) for s in syms}
    EH = {s: D[s].high.rolling(entry_n).max().shift(1) for s in syms}
    EL = {s: D[s].low.rolling(entry_n).min().shift(1) for s in syms}
    XH = {s: D[s].high.rolling(exit_n).max().shift(1) for s in syms}
    XL = {s: D[s].low.rolling(exit_n).min().shift(1) for s in syms}

    eq, pos, rows, trades = 1.0, {}, [], []
    mtm_prev = 1.0
    for i, t in enumerate(idx):
        if i < entry_n + 1: continue
        day_start = eq + sum(p['units']*(D[s].close.iloc[i-1]-p['entry'])*p['dir']
                             for s, p in pos.items() if not np.isnan(D[s].close.iloc[i-1]))
        for s in syms:
            px, a = D[s].close.iloc[i], A[s].iloc[i]
            hi_, lo_ = D[s].high.iloc[i], D[s].low.iloc[i]
            if np.isnan(px) or np.isnan(a): continue
            op = D[s].open.iloc[i] if 'open' in D[s] else px
            f_d = fund.loc[t, s] if (t in fund.index and s in fund.columns) else np.nan
            p = pos.get(s)
            if p:
                # 펀딩: 롱은 양수 펀딩 지불, 숏은 수취 (결측일은 0 아닌 보수적 스킵)
                if not np.isnan(f_d):
                    eq -= p['dir'] * f_d * p['units'] * px
                exit_px = None
                if p['dir'] > 0:
                    lvl = max(p['stop'], XL[s].iloc[i])
                    if lo_ <= lvl: exit_px = min(lvl, op)      # 갭하락 시 시가 체결
                else:
                    lvl = min(p['stop'], XH[s].iloc[i])
                    if hi_ >= lvl: exit_px = max(lvl, op)      # 갭상승 시 시가 체결
                if exit_px is not None:
                    pnl = p['units']*(exit_px-p['entry'])*p['dir']
                    eq += pnl - p['units']*exit_px*EXIT_COST
                    trades.append(dict(ts=t, sym=s, dir=p['dir'],
                                       r=pnl/(p['units']*p['risk_d'])))
                    pos.pop(s)
                    continue
            if s in pos: continue
            gross = sum(pp['units']*D[ss].close.iloc[i] for ss, pp in pos.items()
                        if not np.isnan(D[ss].close.iloc[i]))
            if gross >= gross_cap*eq: continue
            heat = sum(pp['risk_frac'] for pp in pos.values())
            if heat >= heat_cap: continue
            if hi_ > EH[s].iloc[i]:
                d_ = 1; fill = max(op, EH[s].iloc[i]); stop = fill - 2*a
            elif lo_ < EL[s].iloc[i]:
                d_ = -1; fill = min(op, EL[s].iloc[i]); stop = fill + 2*a
            else: continue
            units = min(risk_pct*max(mtm_prev, 1e-9)/(2*a), max(0.0, (gross_cap*eq-gross))/fill)
            if units <= 0: continue
            eq -= units*fill*ENTRY_COST
            # 같은날 스탑아웃 (비관적: 진입 후 역행했다고 가정)
            if (d_ > 0 and lo_ <= stop) or (d_ < 0 and hi_ >= stop):
                pnl = units*(stop-fill)*d_
                eq += pnl - units*stop*EXIT_COST
                trades.append(dict(ts=t, sym=s, dir=d_, r=pnl/(units*2*a)))
                continue
            pos[s] = dict(dir=d_, units=units, stop=stop, entry=fill, risk_d=2*a,
                          risk_frac=risk_pct)
        mtm = eq + sum(p['units']*(D[s].close.iloc[i]-p['entry'])*p['dir']
                       for s, p in pos.items() if not np.isnan(D[s].close.iloc[i]))
        eq_for_size = mtm            # 사이징은 시가평가 자본 기준 (Codex: 폐쇄자본은 과대사이징)
        if (mtm/day_start - 1) < daily_stop and pos:      # 일손실 한도: 전량청산
            for s, p in list(pos.items()):
                px = D[s].close.iloc[i]
                if np.isnan(px): continue
                eq += p['units']*(px-p['entry'])*p['dir'] - p['units']*px*EXIT_COST
                pos.pop(s)
            mtm = eq
        gross = sum(p['units']*D[s].close.iloc[i] for s, p in pos.items()
                    if not np.isnan(D[s].close.iloc[i]))
        rows.append(dict(ts=t, equity=mtm, lev=gross/mtm if mtm > 0 else 0, n=len(pos)))
        mtm_prev = mtm
    # 종료 청산 (잔여 포지션 시장가 + 비용)
    if pos:
        i = len(idx) - 1
        for s_, p in list(pos.items()):
            px = D[s_].close.iloc[i]
            if np.isnan(px): continue
            eq += p['units']*(px-p['entry'])*p['dir'] - p['units']*px*EXIT_COST
        rows[-1]['equity'] = eq
    return pd.DataFrame(rows).set_index('ts'), pd.DataFrame(trades)

def report(name, df, tr, cap=10_000_000):
    d = df.equity
    r = d.pct_change().dropna()
    yrs = (d.index[-1]-d.index[0]).days/365.25
    cagr = (d.iloc[-1]/d.iloc[0])**(1/yrs)-1
    mdd = (1-d/d.cummax()).max()
    won = r*cap
    print(f"\n{'='*96}\n{name}  ({d.index[0].date()}~{d.index[-1].date()}, {yrs:.1f}y, 거래 {len(tr)}회)\n{'='*96}")
    print(f"  CAGR {cagr*100:+.1f}%   MDD {mdd*100:.1f}%   평균레버 {df.lev.mean():.2f}x  최대레버 {df.lev.max():.1f}x   양수일 {(r>0).mean()*100:.0f}%")
    print(f"  1,000만원 일별 P&L: 중앙 {won.median():+,.0f}  상위10% {won.quantile(.9):+,.0f}  "
          f"하위10% {won.quantile(.1):+,.0f}  최고 {won.max():+,.0f}  최악 {won.min():+,.0f}")
    yr = d.groupby(d.index.year).apply(lambda x: x.iloc[-1]/x.iloc[0]-1)
    print("  연도별: " + "  ".join(f"{y}:{v*100:+.1f}%" for y, v in yr.items()))
    if len(tr):
        wr = (tr.r>0).mean()
        print(f"  승률 {wr*100:.0f}%  평균 R {tr.r.mean():+.2f}  최고 R {tr.r.max():+.1f}  기대값/거래 {tr.r.mean()*2:+.2f}% (자본 대비)")

if __name__ == '__main__':
    for name, en, xn in [("터틀 S1 (20일 진입/10일 청산, 2xATR 스탑, 리스크 2%)", 20, 10),
                          ("터틀 S2 (55일 진입/20일 청산)", 55, 20)]:
        df, tr = run(entry_n=en, exit_n=xn)
        report(name, df, tr)
    df, tr = run(entry_n=55, exit_n=20, risk_pct=0.05)
    report("터틀 S2 + 리스크 5%/거래 (공격판)", df, tr)
