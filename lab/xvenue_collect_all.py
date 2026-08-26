"""사전 규칙 유니버스(유동성 $5M+) 전체 펀딩 이력 수집 — 코인별 즉시 저장."""
from __future__ import annotations
import json, time, urllib.request
import pandas as pd

def hl_post(body, retries=4):
    req = urllib.request.Request('https://api.hyperliquid.xyz/info',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
        except Exception: time.sleep(1.5*(i+1))
    return None

def bybit_get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,
                    headers={'User-Agent':'Mozilla/5.0'}), timeout=30) as r:
                return json.loads(r.read())
        except Exception: time.sleep(1.5*(i+1))
    return None

scan = pd.read_csv('lab/data/xvenue_scan.csv')
ALL = list(scan[scan.min_vol >= 5].coin)
try: hl_all = {c: s.dropna() for c, s in pd.read_parquet('lab/data/xv_hl_deep.parquet').items()}
except Exception: hl_all = {}
try: by_all = {c: s.dropna() for c, s in pd.read_parquet('lab/data/xv_by_deep.parquet').items()}
except Exception: by_all = {}

for c in ALL:
    if c in hl_all and c in by_all:
        continue
    if c not in hl_all:
        rows, start = [], 0
        while True:
            d = hl_post({"type": "fundingHistory", "coin": c, "startTime": start})
            if not d: break
            rows += [(int(x['time']), float(x['fundingRate'])) for x in d]
            if len(d) < 500: break
            start = int(d[-1]['time']) + 1
            time.sleep(0.12)
        if rows:
            hl_all[c] = (pd.DataFrame(rows, columns=['ts','f']).drop_duplicates('ts')
                         .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
                         .set_index('ts').sort_index()['f'])
    if c not in by_all:
        bn = ('1000' + c[1:]) if c.startswith('k') else c
        rows, end = [], int(time.time()*1000)
        while True:
            d = bybit_get(f'https://api.bybit.com/v5/market/funding/history?category=linear'
                          f'&symbol={bn}USDT&endTime={end}&limit=200')
            if not d: break
            lst = d['result']['list']
            if not lst: break
            rows += [(int(x['fundingRateTimestamp']), float(x['fundingRate'])) for x in lst]
            o = min(int(x['fundingRateTimestamp']) for x in lst)
            if o >= end or len(lst) < 200: break
            end = o - 1
            time.sleep(0.1)
        if rows:
            by_all[c] = (pd.DataFrame(rows, columns=['ts','f']).drop_duplicates('ts')
                         .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
                         .set_index('ts').sort_index()['f'])
    # 코인별 체크포인트
    pd.DataFrame(hl_all).to_parquet('lab/data/xv_hl_deep.parquet')
    pd.DataFrame(by_all).to_parquet('lab/data/xv_by_deep.parquet')
    print(f"  {c:9s} HL {len(hl_all.get(c, [])):6d}  By {len(by_all.get(c, [])):5d}  [저장]", flush=True)
print("DONE")
