"""동일자산 perp 쌍의 가격·펀딩 수집 (USDT vs USDC vs 인버스)."""
from __future__ import annotations
import ccxt, time
import pandas as pd

def retry(fn,*a,**k):
    for i in range(8):
        try: return fn(*a,**k)
        except Exception:
            if i==7: return None
            time.sleep(1.5*(i+1))

ex=ccxt.bybit({'enableRateLimit':True})
FLOOR=int(pd.Timestamp('2021-01-01',tz='utc').timestamp()*1000)
STEP={'1h':3600e3,'8h':8*3600e3,'1d':86400e3}
IV={'1h':'60','8h':'480','1d':'D'}

def kline(mid, cat, tf='1h'):
    """v5 kline 역방향 수집 (심볼이 ccxt에 없어도 동작)."""
    rows, end = [], int(time.time()*1000)
    while end > FLOOR:
        r = retry(ex.publicGetV5MarketKline,
                  {'category':cat,'symbol':mid,'interval':IV[tf],'end':end,'limit':1000})
        if not r: break
        lst=r.get('result',{}).get('list',[])
        if not lst: break
        for x in lst: rows.append((int(x[0]), float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])))
        oldest=min(int(x[0]) for x in lst)
        if oldest>=end: break
        end=oldest-1
        if len(lst)<1000: break
    if not rows: return None
    df=(pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']).drop_duplicates('ts')
        .assign(ts=lambda d: pd.to_datetime(d.ts,unit='ms',utc=True)).set_index('ts').sort_index())
    return df

def funding_back(mid, cat):
    rows,end=[],int(time.time()*1000)
    while end>FLOOR:
        r=retry(ex.publicGetV5MarketFundingHistory,{'category':cat,'symbol':mid,'endTime':end,'limit':200})
        if not r: break
        lst=r.get('result',{}).get('list',[])
        if not lst: break
        for x in lst: rows.append((int(x['fundingRateTimestamp']),float(x['fundingRate'])))
        o=min(int(x['fundingRateTimestamp']) for x in lst)
        if o>=end: break
        end=o-1
        if len(lst)<200: break
    if not rows: return None
    return (pd.DataFrame(rows,columns=['ts','f']).drop_duplicates('ts')
            .assign(ts=lambda d: pd.to_datetime(d.ts,unit='ms',utc=True)).set_index('ts').sort_index()['f'])

SPECS=[('BTC_USDT','BTCUSDT','linear'),('BTC_USDC','BTCPERP','linear'),
       ('ETH_USDT','ETHUSDT','linear'),('ETH_USDC','ETHPERP','linear'),
       ('SOL_USDT','SOLUSDT','linear'),('SOL_USDC','SOLPERP','linear'),
       ('XRP_USDT','XRPUSDT','linear'),('XRP_USDC','XRPPERP','linear'),
       ('DOGE_USDT','DOGEUSDT','linear'),('DOGE_USDC','DOGEPERP','linear')]
px, fu = {}, {}
for name, mid, cat in SPECS:
    k=kline(mid,cat,'1h'); f=funding_back(mid,cat)
    if k is not None: px[name]=k
    if f is not None: fu[name]=f
    print(f"{name:11s} kline={0 if k is None else len(k):6d} "
          f"({k.index[0].date() if k is not None else '-'}) funding={0 if f is None else len(f):5d}", flush=True)
pd.concat(px,names=['sym']).to_parquet('lab/data/pairperp_1h.parquet')
pd.DataFrame(fu).to_parquet('lab/data/pairperp_funding.parquet')
print(f"DONE px={len(px)} fu={len(fu)}")
