"""펀딩 이력 수집 — endTime 역방향 페이징(상장일 이전 since면 빈배열 반환하는 버그 회피)."""
from __future__ import annotations
import ccxt, time, json
import pandas as pd

def retry(fn, *a, **k):
    for i in range(6):
        try: return fn(*a, **k)
        except Exception:
            if i == 5: return None
            time.sleep(1.0 * (i + 1))

ex = ccxt.bybit({'enableRateLimit': True})
FLOOR = int(pd.Timestamp('2021-01-01', tz='utc').timestamp() * 1000)

def funding_back(market_id: str) -> pd.Series | None:
    """현재부터 endTime을 뒤로 밀며 전량 수집."""
    rows, end = [], int(time.time() * 1000)
    while end > FLOOR:
        r = retry(ex.publicGetV5MarketFundingHistory,
                  {'category': 'linear', 'symbol': market_id, 'endTime': end, 'limit': 200})
        if not r: break
        lst = r.get('result', {}).get('list', [])
        if not lst: break
        for x in lst:
            rows.append((int(x['fundingRateTimestamp']), float(x['fundingRate'])))
        oldest = min(int(x['fundingRateTimestamp']) for x in lst)
        if oldest >= end: break
        end = oldest - 1
        if len(lst) < 200: break
    if not rows: return None
    s = (pd.DataFrame(rows, columns=['ts', 'f']).drop_duplicates('ts')
         .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
         .set_index('ts').sort_index()['f'])
    return s

bases = json.load(open('lab/data/carry_eligible.json'))
F = {}
for i, b in enumerate(bases):
    s = funding_back(f'{b}USDT')
    if s is not None and len(s) >= 90: F[b] = s
    if (i + 1) % 25 == 0:
        print(f"[{i+1}/{len(bases)}] ok={len(F)}", flush=True)
        pd.DataFrame(F).to_parquet('lab/data/broad_funding_8h.parquet')
pd.DataFrame(F).to_parquet('lab/data/broad_funding_8h.parquet')
print(f"DONE {len(F)}/{len(bases)}")
