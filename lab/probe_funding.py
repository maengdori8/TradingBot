"""펀딩/베이시스 캐리 경제성 1차 실증 — 순수 데이터 조사(전략 아님)."""
from __future__ import annotations
import ccxt, time, sys
import pandas as pd

ex = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def retry(fn, *a, **k):
    for i in range(6):
        try:
            return fn(*a, **k)
        except Exception as e:
            if i == 5: raise
            time.sleep(1.5 * (i + 1))
    return None

retry(ex.load_markets)

SYMS = ['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','XRP/USDT:USDT','DOGE/USDT:USDT',
        'BNB/USDT:USDT','ADA/USDT:USDT','AVAX/USDT:USDT','LINK/USDT:USDT','LTC/USDT:USDT']

def funding_hist(sym, start_ms):
    out, since, now = [], start_ms, time.time() * 1000
    while True:
        rs = retry(ex.fetch_funding_rate_history, sym, since=since, limit=200)
        if not rs: break
        out += rs
        last = rs[-1]['timestamp']
        if last <= since or len(rs) < 200 or last > now - 8 * 3600 * 1000: break
        since = last + 1
    if not out: return None
    df = (pd.DataFrame([{'ts': pd.to_datetime(r['timestamp'], unit='ms', utc=True),
                         'f': float(r['fundingRate'])} for r in out])
          .drop_duplicates('ts').set_index('ts').sort_index())
    return df['f']

start = int(pd.Timestamp('2021-01-01', tz='utc').timestamp() * 1000)
res = {}
print(f"{'symbol':22s} {'n':>6s} {'from':>11s} {'to':>11s} {'mean/8h':>10s} {'ann':>9s} {'pos%':>6s} {'med/8h':>10s}")
for s in SYMS:
    f = funding_hist(s, start)
    if f is None or len(f) < 100:
        print(f"{s:22s} no data"); continue
    res[s] = f
    print(f"{s:22s} {len(f):6d} {str(f.index[0].date()):>11s} {str(f.index[-1].date()):>11s} "
          f"{f.mean()*100:+9.5f}% {f.mean()*3*365*100:+8.2f}% {(f>0).mean()*100:5.1f}% {f.median()*100:+9.5f}%")

pd.DataFrame(res).to_parquet('lab/data/funding_probe.parquet')
print("\nsaved lab/data/funding_probe.parquet", pd.DataFrame(res).shape)
