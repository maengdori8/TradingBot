"""캐리 검증 데이터 수집 — BTC/ETH 우선(Codex 합의: 최소 데이터셋)."""
from __future__ import annotations
import ccxt, time, sys
import pandas as pd

def mk(dt): return ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': dt}})
SWAP, SPOT = mk('swap'), mk('spot')

def retry(fn, *a, **k):
    for i in range(8):
        try: return fn(*a, **k)
        except Exception as e:
            if i == 7: raise
            time.sleep(1.5 * (i + 1))

for e in (SWAP, SPOT): retry(e.load_markets)
STEP = {'5m': 300e3, '1h': 3600e3, '4h': 4*3600e3, '1d': 86400e3}

def ohlcv(ex, sym, tf, start_ms):
    out, since, now = [], start_ms, time.time() * 1000
    while since < now:
        rs = retry(ex.fetch_ohlcv, sym, tf, since=int(since), limit=1000)
        if not rs: break
        out += rs
        nxt = rs[-1][0] + STEP[tf]
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
        rs = retry(SWAP.fetch_funding_rate_history, sym, since=int(since), limit=200)
        if not rs: break
        out += rs
        last = rs[-1]['timestamp']
        if last <= since or len(rs) < 200 or last > now - 8*3600e3: break
        since = last + 1
    if not out: return None
    return (pd.DataFrame([{'ts': pd.to_datetime(r['timestamp'], unit='ms', utc=True),
                           'f': float(r['fundingRate'])} for r in out])
            .drop_duplicates('ts').set_index('ts').sort_index()['f'])

BASES = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTC','ETH']
TF = sys.argv[2] if len(sys.argv) > 2 else '1h'
START = int(pd.Timestamp('2021-01-01', tz='utc').timestamp() * 1000)

perp, spot, fund = {}, {}, {}
for b in BASES:
    ps, ss = f'{b}/USDT:USDT', f'{b}/USDT'
    if ps in SWAP.markets:
        d = ohlcv(SWAP, ps, TF, START)
        if d is not None: perp[b] = d
        f = funding(ps, START)
        if f is not None: fund[b] = f
    if ss in SPOT.markets:
        d = ohlcv(SPOT, ss, TF, START)
        if d is not None: spot[b] = d
    print(f"{b:6s} perp={len(perp.get(b,[])):6d} spot={len(spot.get(b,[])):6d} fund={len(fund.get(b,[])):5d}"
          f" perp_from={perp[b].index[0].date() if b in perp else '-'}"
          f" spot_from={spot[b].index[0].date() if b in spot else '-'}", flush=True)

sfx = TF
if perp: pd.concat(perp, names=['sym']).to_parquet(f'lab/data/perp_{sfx}.parquet')
if spot: pd.concat(spot, names=['sym']).to_parquet(f'lab/data/spot_{sfx}.parquet')
if fund: pd.DataFrame(fund).to_parquet('lab/data/funding_8h.parquet')
print(f"DONE perp={len(perp)} spot={len(spot)} fund={len(fund)}")
