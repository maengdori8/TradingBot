"""교차 거래소 펀딩 차익 — Bybit vs Hyperliquid 역사 검증.

구조: 같은 코인 perp을 두 거래소에서 반대 방향 보유 → 순수 펀딩 차이 수취.
양다리 perp: 왕복 비용 = 4bp(maker) ~ 16bp(taker+슬립). 현물·차입 불필요.
데이터: Bybit 8h 펀딩(동결본) + HL 1h 펀딩(수집) → 일별 합산으로 정규화.
"""
from __future__ import annotations
import gzip, json, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
OUT = Path('lab/data/hl_funding.parquet')

def post(body, retries=5):
    req = urllib.request.Request('https://api.hyperliquid.xyz/info',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None

def collect():
    out = {}
    for c in COINS:
        rows, start = [], 0
        while True:
            d = post({"type": "fundingHistory", "coin": c, "startTime": start})
            if not d: break
            rows += [(int(x['time']), float(x['fundingRate'])) for x in d]
            if len(d) < 500: break
            start = int(d[-1]['time']) + 1
            time.sleep(0.15)
        s = (pd.DataFrame(rows, columns=['ts', 'f']).drop_duplicates('ts')
             .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
             .set_index('ts').sort_index()['f'])
        out[c] = s
        print(f"  HL {c}: {len(s)}건 {s.index[0].date()}~{s.index[-1].date()} "
              f"연율 {s.mean()*24*365*100:+.2f}%", flush=True)
    pd.DataFrame(out).to_parquet(OUT)
    return out

if __name__ == '__main__':
    print("HL 펀딩 수집:")
    collect()
