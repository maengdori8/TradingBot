"""광범위 캐리 데이터 수집 — 291 carry-eligible 종목의 펀딩 + 일봉."""
from __future__ import annotations
import ccxt, time, json, os, sys
import pandas as pd

def retry(fn, *a, **k):
    for i in range(6):
        try: return fn(*a, **k)
        except Exception as e:
            if i == 5: return None
            time.sleep(1.0 * (i + 1))

SWAP = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
SPOT = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
retry(SWAP.load_markets); retry(SPOT.load_markets)
bases = json.load(open('lab/data/carry_eligible.json'))
START = int(pd.Timestamp('2021-01-01', tz='utc').timestamp() * 1000)

def ohlcv_d(ex, sym):
    out, since, now = [], START, time.time()*1000
    while since < now:
        rs = retry(ex.fetch_ohlcv, sym, '1d', since=int(since), limit=1000)
        if not rs: break
        out += rs
        nxt = rs[-1][0] + 86400e3
        if nxt <= since or len(rs) < 1000: break
        since = nxt
    if not out: return None
    df = pd.DataFrame(out, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.drop_duplicates('ts').set_index('ts').sort_index().astype(float)

def funding(sym):
    out, since, now = [], START, time.time()*1000
    while True:
        rs = retry(SWAP.fetch_funding_rate_history, sym, since=int(since), limit=200)
        if not rs: break
        out += rs
        last = rs[-1]['timestamp']
        if last <= since or len(rs) < 200 or last > now - 8*3600e3: break
        since = last + 1
    if not out: return None
    return (pd.DataFrame([{'ts': pd.to_datetime(r['timestamp'], unit='ms', utc=True), 'f': float(r['fundingRate'])}
                          for r in out]).drop_duplicates('ts').set_index('ts').sort_index()['f'])

F, PD_, SD_ = {}, {}, {}
for i, b in enumerate(bases):
    ps, ss = f'{b}/USDT:USDT', f'{b}/USDT'
    if ps not in SWAP.markets or ss not in SPOT.markets: continue
    f = funding(ps)
    if f is not None and len(f) >= 90: F[b] = f
    p = ohlcv_d(SWAP, ps);  s = ohlcv_d(SPOT, ss)
    if p is not None: PD_[b] = p
    if s is not None: SD_[b] = s
    if (i+1) % 20 == 0:
        print(f"[{i+1}/{len(bases)}] fund={len(F)} perp={len(PD_)} spot={len(SD_)}", flush=True)
        pd.DataFrame(F).to_parquet('lab/data/broad_funding_8h.parquet')

pd.DataFrame(F).to_parquet('lab/data/broad_funding_8h.parquet')
pd.concat(PD_, names=['sym']).to_parquet('lab/data/broad_perp_1d.parquet')
pd.concat(SD_, names=['sym']).to_parquet('lab/data/broad_spot_1d.parquet')
print(f"DONE fund={len(F)} perp={len(PD_)} spot={len(SD_)}")
