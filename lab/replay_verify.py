"""검증: 동결 데이터를 step()으로 재생 — 라이브 코드 경로의 백테스트."""
from __future__ import annotations
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from carrybot.aggressive.turtle import Bar, TurtleConfig, TurtleState, step, _mtm

cfg = TurtleConfig()
perp = pd.read_parquet('lab/frozen/perp_1d.parquet')
fund = pd.read_parquet('lab/frozen/funding.parquet')[list(cfg.syms)].resample('D').sum(min_count=1)
D = {s: perp.xs(s, level='sym')[['open','high','low','close']].sort_index() for s in cfg.syms}
idx = None
for s in cfg.syms:
    idx = D[s].index if idx is None else idx.union(D[s].index)
idx = idx.sort_values()

state = TurtleState()
rows = []
for i, t in enumerate(idx):
    if i < cfg.entry_n + 2: continue
    bars = {}
    for s in cfg.syms:
        d = D[s].reindex(idx)
        if np.isnan(d.close.iloc[i]): continue
        hist = d.iloc[max(0, i - cfg.entry_n):i]
        if len(hist) < cfg.entry_n: continue
        bars[s] = Bar(open=d.open.iloc[i], high=d.high.iloc[i], low=d.low.iloc[i],
                      close=d.close.iloc[i],
                      entry_hi=hist.high.tail(cfg.entry_n).max(),
                      entry_lo=hist.low.tail(cfg.entry_n).min(),
                      exit_hi=hist.high.tail(cfg.exit_n).max(),
                      exit_lo=hist.low.tail(cfg.exit_n).min(),
                      funding=fund.loc[t, s] if t in fund.index else np.nan)
    if not bars: continue
    state, fills = step(state, bars, cfg, month_key=f"{t.year}-{t.month:02d}")
    if state.killed:
        state.killed = False          # 백테스트에선 다음달 자동 리셋으로 간주
    rows.append(dict(ts=t, equity=_mtm(state, bars)))

d = pd.DataFrame(rows).set_index('ts').equity
r = d.pct_change().dropna()
yrs = (d.index[-1] - d.index[0]).days / 365.25
CAP = 10_000_000
won = r * CAP
print(f"step() 재생 백테스트 ({d.index[0].date()}~{d.index[-1].date()}, {yrs:.1f}y)")
print(f"  CAGR {((d.iloc[-1]/d.iloc[0])**(1/yrs)-1)*100:+.1f}%   MDD {(1-d/d.cummax()).max()*100:.1f}%   양수일 {(r>0).mean()*100:.0f}%")
print(f"  1,000만원 일별: 상위10% {won.quantile(.9):+,.0f}원  하위10% {won.quantile(.1):+,.0f}원  "
      f"최고 {won.max():+,.0f}원  최악 {won.min():+,.0f}원")
yr = d.groupby(d.index.year).apply(lambda x: x.iloc[-1]/x.iloc[0]-1)
print("  연도별: " + "  ".join(f"{y}:{v*100:+.1f}%" for y, v in yr.items()))
