"""캐리 검증용 데이터 수집 — perp/spot OHLCV + 펀딩. 시점보존(닫힌 봉만)."""
from __future__ import annotations
import ccxt, time, sys
import pandas as pd

def mk(dt): return ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': dt}})
SWAP, SPOT = mk('swap'), mk('spot')

def retry(fn, *a, **k):
    for i in range(6):
        try: return fn(*a, **k)
        except Exception:
            if i == 5: raise
            time.sleep(2 * (i + 1))

for e in (SWAP, SPOT): retry(e.load_markets)

def ohlcv(ex, sym, tf, start_ms):
    out, since, now = [], start_ms, time.time() * 1000
    step = {'1h': 3600e3, '4h': 4*3600e3, '1d': 86400e3}[tf]
    while since < now:
        rs = retry(ex.fetch_ohlcv, sym, tf, since=since, limit=1000)
        if not rs: break
        out += rs
        nxt = rs[-1][0] + step
        if nxt <= since: break
        since = nxt
        if len(rs) < 1000: break
    if not out: return None
    df = pd.DataFrame(out, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.drop_duplicates('ts').set_index('ts').sort_index().astype(float)

def funding(sym, start_ms):
    out, since, now = [], start_ms, time.time() * 1000
    while True:
        rs = retry(SWAP.fetch_funding_rate_history, sym, since=since, limit=200)
        if not rs: break
        out += rs
        last = rs[-1]['timestamp']
        if last <= since or len(rs) < 200 or last > now - 8*3600e3: break
        since = last + 1
    if not out: return None
    return (pd.DataFrame([{'ts': pd.to_datetime(r['timestamp'], unit='ms', utc=True),
                           'f': float(r['fundingRate'])} for r in out])
            .drop_duplicates('ts').set_index('ts').sort_index()['f'])

BASES = ['BTC','ETH','SOL','XRP','DOGE','BNB','ADA','AVAX','LINK','LTC','DOT','TRX','MATIC','UNI','ATOM','NEAR','APT','ARB','OP','FIL']
START = int(pd.Timestamp('2021-01-01', tz='utc').timestamp() * 1000)

perp, spot, fund = {}, {}, {}
for b in BASES:
    ps, ss = f'{b}/USDT:USDT', f'{b}/USDT'
    try:
        if ps in SWAP.markets:
            d = ohlcv(SWAP, ps, '1h', START)
            if d is not None: perp[b] = d
            f = funding(ps, START)
            if f is not None: fund[b] = f
        if ss in SPOT.markets:
            d = ohlcv(SPOT, ss, '1h', START)
            if d is not None: spot[b] = d
        pn = len(perp.get(b, [])); sn = len(spot.get(b, [])); fn = len(fund.get(b, []))
        p0 = perp[b].index[0].date() if b in perp else '-'
        print(f"{b:6s} perp={pn:6d}({p0}) spot={sn:6d} fund={fn:5d}", flush=True)
    except Exception as ex_:
        print(f"{b:6s} ERR {str(ex_)[:80]}", flush=True)

pd.concat({k: v for k, v in perp.items()}, names=['sym']).to_parquet('lab/data/perp_1h.parquet')
pd.concat({k: v for k, v in spot.items()}, names=['sym']).to_parquet('lab/data/spot_1h.parquet')
pd.DataFrame(fund).to_parquet('lab/data/funding_8h.parquet')
print(f"\nsaved: perp={len(perp)} spot={len(spot)} funding={len(fund)} symbols")
