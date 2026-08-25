"""동일자산 perp 간 펀딩 차이 측정 — 양다리 1bp 델타중립 구성의 실현가능성."""
from __future__ import annotations
import ccxt, time
import pandas as pd, numpy as np

def retry(fn,*a,**k):
    for i in range(6):
        try: return fn(*a,**k)
        except Exception:
            if i==5: return None
            time.sleep(1.5*(i+1))

ex=ccxt.bybit({'enableRateLimit':True})
FLOOR=int(pd.Timestamp('2021-01-01',tz='utc').timestamp()*1000)

def funding_back(market_id: str, category: str='linear') -> pd.Series | None:
    """endTime 역방향 페이징으로 펀딩 전량 수집."""
    rows, end = [], int(time.time()*1000)
    while end > FLOOR:
        r = retry(ex.publicGetV5MarketFundingHistory,
                  {'category':category,'symbol':market_id,'endTime':end,'limit':200})
        if not r: break
        lst = r.get('result',{}).get('list',[])
        if not lst: break
        for x in lst: rows.append((int(x['fundingRateTimestamp']), float(x['fundingRate'])))
        oldest = min(int(x['fundingRateTimestamp']) for x in lst)
        if oldest >= end: break
        end = oldest - 1
        if len(lst) < 200: break
    if not rows: return None
    return (pd.DataFrame(rows,columns=['ts','f']).drop_duplicates('ts')
            .assign(ts=lambda d: pd.to_datetime(d.ts,unit='ms',utc=True))
            .set_index('ts').sort_index()['f'])

PAIRS = [('BTC','BTCUSDT','BTCPERP'), ('ETH','ETHUSDT','ETHPERP'), ('SOL','SOLUSDT','SOLPERP')]
INV   = [('BTC','BTCUSDT','BTCUSD'), ('ETH','ETHUSDT','ETHUSD')]
out={}
print(f"{'pair':>18s} {'n':>5s} {'from':>11s} {'USDT ann':>10s} {'ALT ann':>10s} {'차이 ann':>10s} {'|차이|평균/8h':>13s} {'차이>0':>7s}")
for label, a_id, b_id, cat in ([(f'{b}: USDT-USDC', x, y, 'linear') for b,x,y in PAIRS] +
                               [(f'{b}: USDT-INV',  x, y, 'inverse') for b,x,y in INV]):
    fa = funding_back(a_id,'linear'); fb = funding_back(b_id, cat)
    if fa is None or fb is None: print(f"{label:>18s} 데이터 없음"); continue
    j = pd.concat({'a':fa,'b':fb},axis=1).dropna()
    if len(j) < 100: print(f"{label:>18s} 겹치는 구간 부족 n={len(j)}"); continue
    d = j['a'] - j['b']
    out[label]=j
    print(f"{label:>18s} {len(j):5d} {str(j.index[0].date()):>11s} "
          f"{j['a'].mean()*3*365*100:+9.2f}% {j['b'].mean()*3*365*100:+9.2f}% "
          f"{d.mean()*3*365*100:+9.2f}% {d.abs().mean()*100:12.5f}% {(d>0).mean()*100:6.1f}%")
    out[label]=d
pd.DataFrame(out).to_parquet('lab/data/perp_pair_funding.parquet')
print("\nsaved lab/data/perp_pair_funding.parquet")
