"""시점보존 유니버스 + 광범위 캐리 데이터 수집 (생존편향 제거용 메타 포함)."""
from __future__ import annotations
import ccxt, time, json
import pandas as pd

def retry(fn, *a, **k):
    for i in range(8):
        try: return fn(*a, **k)
        except Exception:
            if i == 7: raise
            time.sleep(1.5 * (i + 1))

ex = ccxt.bybit({'enableRateLimit': True})
# --- 1) 시점보존 유니버스 메타: Trading + Closed 전부 ---
rows = []
for st in ['Trading', 'Closed', 'PreLaunch']:
    r = retry(ex.publicGetV5MarketInstrumentsInfo, {'category': 'linear', 'status': st, 'limit': 1000})
    for x in r['result']['list']:
        rows.append({'symbol': x['symbol'], 'status': st, 'contractType': x.get('contractType'),
                     'baseCoin': x.get('baseCoin'), 'quoteCoin': x.get('quoteCoin'),
                     'launchTime': x.get('launchTime'), 'deliveryTime': x.get('deliveryTime'),
                     'fundingInterval': x.get('fundingInterval')})
uni = pd.DataFrame(rows)
for c in ('launchTime', 'deliveryTime'):
    uni[c] = pd.to_numeric(uni[c], errors='coerce')
    uni[c] = pd.to_datetime(uni[c].where(uni[c] > 0), unit='ms', utc=True)
uni.to_parquet('lab/data/universe_linear.parquet')
perp_all = uni[(uni.contractType == 'LinearPerpetual') & (uni.quoteCoin == 'USDT')]
print(f"linear instruments: {len(uni)} | USDT perps ever: {len(perp_all)} "
      f"(trading {(perp_all.status=='Trading').sum()}, closed {(perp_all.status=='Closed').sum()})")
print("delisting years:", perp_all[perp_all.status=='Closed']['deliveryTime'].dt.year.value_counts().sort_index().to_dict())

# --- 2) 현물 교집합 (캐리는 양다리 필요) ---
sp = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
retry(sp.load_markets)
spot_bases = {v['base'] for v in sp.markets.values() if v.get('spot') and v.get('quote') == 'USDT' and v.get('active')}
trading_bases = set(perp_all[perp_all.status == 'Trading']['baseCoin'])
both = sorted(spot_bases & trading_bases)
print(f"spot USDT bases: {len(spot_bases)} | trading perp bases: {len(trading_bases)} | BOTH (carry-eligible): {len(both)}")
json.dump(both, open('lab/data/carry_eligible.json', 'w'))
print("sample:", both[:30])
