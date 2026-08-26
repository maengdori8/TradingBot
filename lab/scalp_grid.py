"""단타 슬리브 검증 — 1h 봉, 레버리지, 정직 비용. 격자 전체 공개(선택 없음).

사전 고정 (실행 전):
- 시스템 2종 (출판 표준형, 튜닝 금지):
  BRK: 돌파 — N시간 채널 상/하단 돌파 진입, N/2 반대채널 or 6xATR(24) 스탑 청산
  MR : 평균회귀 — 24h SMA 대비 z>2 페이드, z=0 회귀 or 4xATR(24) 스탑, 최대보유 24h
- 파라미터: BRK N ∈ {24, 48, 96}, MR 고정 1종
- 리스크/거래: 2% / 5%  → 레버리지는 결과값 (스탑거리 역산), 총 그로스 캡 10x
- 일손실 -5% 정지(당일), 비용: taker 6bp+슬립 2bp 편도 = 왕복 16bp, 펀딩 반영(롱 지불)
- 데이터: BTC/ETH (2021-01~), SOL (2021-10~) perp 1h + 일별 펀딩
- 판정: 순 CAGR>0 이고 MDD<50%인 셀만 생존. 전멸이면 전멸 보고.
"""
from __future__ import annotations
import numpy as np, pandas as pd

COST_IN, COST_OUT = 0.0008, 0.0008     # 편도 8bp(taker+슬립)

def load():
    p = pd.read_parquet('lab/frozen/perp_1h.parquet')
    d = {s: p.xs(s, level='sym')[['open','high','low','close']] for s in ('BTC','ETH')}
    d['SOL'] = pd.read_parquet('lab/data/sol_1h.parquet')[['open','high','low','close']]
    f = pd.read_parquet('lab/frozen/funding.parquet')[['BTC','ETH','SOL']].resample('D').sum(min_count=1)
    return d, f

def atr(df, n=24):
    tr = pd.concat([df.high-df.low, (df.high-df.close.shift()).abs(),
                    (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def run(data, fund, system, N, risk, gross_cap=10.0, daily_halt=-0.05):
    syms = list(data)
    idx = None
    for s in syms:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()
    D = {s: data[s].reindex(idx) for s in syms}
    A = {s: atr(D[s]) for s in syms}
    if system == 'BRK':
        HI = {s: D[s].high.rolling(N).max().shift(1) for s in syms}
        LO = {s: D[s].low.rolling(N).min().shift(1) for s in syms}
        XH = {s: D[s].high.rolling(N//2).max().shift(1) for s in syms}
        XL = {s: D[s].low.rolling(N//2).min().shift(1) for s in syms}
    else:
        SMA = {s: D[s].close.rolling(24).mean().shift(1) for s in syms}
        SD  = {s: D[s].close.rolling(24).std().shift(1) for s in syms}

    eq, pos, rows, trades = 1.0, {}, [], 0
    day, day_eq, halted = None, 1.0, False
    fd = {s: fund[s].reindex(idx.normalize()).to_numpy() for s in syms}
    for i, t in enumerate(idx):
        if i < 100: continue
        if t.date() != day:
            day, halted = t.date(), False
            day_eq = eq + sum(p['u']*(D[s].close.iloc[i-1]-p['e'])*p['d'] for s,p in pos.items()
                              if not np.isnan(D[s].close.iloc[i-1]))
            # 펀딩 (일 1회, 보유 방향 기준)
            for s, p in pos.items():
                f = fund[s].get(pd.Timestamp(day, tz='utc'), np.nan)
                px = D[s].close.iloc[i-1]
                if not np.isnan(f) and not np.isnan(px):
                    eq -= p['d'] * f * p['u'] * px
        for s in syms:
            o,h,l,c = D[s].open.iloc[i], D[s].high.iloc[i], D[s].low.iloc[i], D[s].close.iloc[i]
            a = A[s].iloc[i]
            if np.isnan(c) or np.isnan(a) or a <= 0: continue
            p = pos.get(s)
            if p:
                p['hold'] += 1
                exit_px = None
                if system == 'BRK':
                    if p['d'] > 0:
                        lvl = max(p['stop'], XL[s].iloc[i])
                        if l <= lvl: exit_px = min(lvl, o)
                    else:
                        lvl = min(p['stop'], XH[s].iloc[i])
                        if h >= lvl: exit_px = max(lvl, o)
                else:
                    z_now = (c - SMA[s].iloc[i]) / SD[s].iloc[i] if SD[s].iloc[i] > 0 else 0
                    if p['d'] > 0 and (l <= p['stop']):   exit_px = min(p['stop'], o)
                    elif p['d'] < 0 and (h >= p['stop']): exit_px = max(p['stop'], o)
                    elif (p['d'] > 0 and z_now >= 0) or (p['d'] < 0 and z_now <= 0) or p['hold'] >= 24:
                        exit_px = c
                if exit_px is not None:
                    eq += p['u']*(exit_px-p['e'])*p['d'] - p['u']*exit_px*COST_OUT
                    trades += 1; pos.pop(s); continue
            if halted or s in pos: continue
            gross = sum(pp['u']*D[ss].close.iloc[i] for ss,pp in pos.items() if not np.isnan(D[ss].close.iloc[i]))
            if gross >= gross_cap*eq: continue
            d_ = 0
            if system == 'BRK':
                if h > HI[s].iloc[i]: d_, fill = 1, max(o, HI[s].iloc[i])
                elif l < LO[s].iloc[i]: d_, fill = -1, min(o, LO[s].iloc[i])
                if d_: stop = fill - d_*6*a
            else:
                sm, sd = SMA[s].iloc[i], SD[s].iloc[i]
                if np.isnan(sm) or sd <= 0: continue
                z = (c - sm) / sd
                if z > 2: d_, fill = -1, c
                elif z < -2: d_, fill = 1, c
                if d_: stop = fill - d_*4*a
            if not d_: continue
            u = min(risk*eq/abs(fill-stop), max(0.0,(gross_cap*eq-gross))/fill)
            if u <= 0: continue
            eq -= u*fill*COST_IN
            if (d_ > 0 and l <= stop and system=='BRK') or (d_ < 0 and h >= stop and system=='BRK'):
                eq += u*(stop-fill)*d_ - u*stop*COST_OUT; trades += 1; continue
            pos[s] = dict(d=d_, u=u, e=fill, stop=stop, hold=0)
        mtm = eq + sum(p['u']*(D[s].close.iloc[i]-p['e'])*p['d'] for s,p in pos.items()
                       if not np.isnan(D[s].close.iloc[i]))
        if not halted and day_eq > 0 and mtm/day_eq - 1 < daily_halt and pos:
            for s,p in list(pos.items()):
                px = D[s].close.iloc[i]
                if np.isnan(px): continue
                eq += p['u']*(px-p['e'])*p['d'] - p['u']*px*COST_OUT; trades += 1; pos.pop(s)
            halted, mtm = True, eq
        rows.append((t, mtm, sum(p['u']*D[s].close.iloc[i] for s,p in pos.items()
                                 if not np.isnan(D[s].close.iloc[i]))/mtm if mtm>0 else 0))
    dfr = pd.DataFrame(rows, columns=['ts','equity','lev']).set_index('ts')
    return dfr, trades

if __name__ == '__main__':
    data, fund = load()
    print(f"{'시스템':>10s} {'리스크':>6s} {'거래':>6s} {'CAGR':>8s} {'MDD':>7s} {'평균레버':>8s} {'최대레버':>8s} {'판정':>6s}")
    for system, N in [('BRK',24), ('BRK',48), ('BRK',96), ('MR',0)]:
        for risk in (0.02, 0.05):
            dfr, tr = run(data, fund, system, N, risk)
            d = dfr.equity.resample('D').last().dropna()
            yrs = (d.index[-1]-d.index[0]).days/365.25
            cagr = (d.iloc[-1]/d.iloc[0])**(1/yrs)-1
            mdd = (1-d/d.cummax()).max()
            ok = cagr > 0 and mdd < 0.5
            nm = f"{system}{N if N else ''}"
            print(f"{nm:>10s} {risk*100:5.0f}% {tr:6d} {cagr*100:+7.1f}% {mdd*100:6.1f}% "
                  f"{dfr.lev.mean():7.2f}x {dfr.lev.max():7.1f}x {'생존' if ok else '탈락':>6s}", flush=True)
