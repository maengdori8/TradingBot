"""교차 거래소 펀딩 차이 전수 스캔 — 현재 30일, 양쪽 상장 전 코인."""
from __future__ import annotations
import json, time, urllib.request
import numpy as np, pandas as pd

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

# 1) HL 유니버스 + 일거래대금
meta = hl_post({"type": "metaAndAssetCtxs"})
hl_uni = {}
for asset, ctx in zip(meta[0]['universe'], meta[1]):
    if asset.get('isDelisted'): continue
    hl_uni[asset['name']] = float(ctx.get('dayNtlVlm', 0))
print(f"HL perp: {len(hl_uni)}개")

# 2) Bybit 심볼 교집합
r = bybit_get('https://api.bybit.com/v5/market/tickers?category=linear')
by_syms = {}
for t in r['result']['list']:
    s = t['symbol']
    if s.endswith('USDT') and '-' not in s:
        by_syms[s[:-4]] = float(t.get('turnover24h', 0))
common = sorted(set(hl_uni) & set(by_syms))
# HL의 kPEPE 등 접두 표기 처리
for h in list(hl_uni):
    if h.startswith('k') and ('1000' + h[1:]) in by_syms:
        common.append(h)
print(f"교집합: {len(common)}개")

# 3) 각 코인 30일 펀딩 (양쪽)
t30 = int(time.time()*1000) - 30*86400*1000
rows = []
for i, c in enumerate(sorted(set(common))):
    by_name = ('1000' + c[1:]) if c.startswith('k') and ('1000'+c[1:]) in by_syms else c
    # HL: 최근 30일 (2페이지)
    fs, start = [], t30
    for _ in range(2):
        d = hl_post({"type": "fundingHistory", "coin": c, "startTime": start})
        if not d: break
        fs += [float(x['fundingRate']) for x in d]
        if len(d) < 500: break
        start = int(d[-1]['time']) + 1
    if len(fs) < 200: continue
    hl_ann = np.mean(fs) * 24 * 365 * 100
    # Bybit: 최근 200건(8h→66일) 중 30일치
    d = bybit_get(f'https://api.bybit.com/v5/market/funding/history?category=linear&symbol={by_name}USDT&limit=200')
    if not d: continue
    lst = [(int(x['fundingRateTimestamp']), float(x['fundingRate'])) for x in d['result']['list']]
    lst = [(t, f) for t, f in lst if t >= t30]
    if len(lst) < 30: continue
    n_per_day = len(lst) / 30
    by_ann = np.mean([f for _, f in lst]) * n_per_day * 365 * 100
    rows.append(dict(coin=c, by=by_ann, hl=hl_ann, diff=by_ann-hl_ann,
                     hl_vol=hl_uni[c]/1e6, by_vol=by_syms[by_name]/1e6))
    if (i+1) % 30 == 0: print(f"  [{i+1}] …", flush=True)
    time.sleep(0.1)

df = pd.DataFrame(rows)
df['abs_diff'] = df['diff'].abs()
df['min_vol'] = df[['hl_vol','by_vol']].min(axis=1)
liq = df[df.min_vol >= 5]                      # 양쪽 일 $5M+
print(f"\n스캔 {len(df)}개, 유동성(양쪽 $5M+) {len(liq)}개")
print(f"\n|현재 30일 차이| 상위 20 (유동성 충족만) — 연율 %:")
print(f"{'코인':>10s} {'Bybit':>8s} {'HL':>8s} {'차이':>8s} {'HL거래대금':>10s} {'By거래대금':>10s}")
for _, r_ in liq.nlargest(20, 'abs_diff').iterrows():
    print(f"{r_.coin:>10s} {r_.by:+7.2f}% {r_.hl:+7.2f}% {r_['diff']:+7.2f}% "
          f"${r_.hl_vol:8.0f}M ${r_.by_vol:8.0f}M")
print(f"\n유동성군 |차이| 중앙값 {liq.abs_diff.median():.2f}%  "
      f"maker허들(4.2%) 초과 {int((liq.abs_diff>4.2).sum())}개  "
      f"10% 초과 {int((liq.abs_diff>10).sum())}개")
df.to_csv('lab/data/xvenue_scan.csv', index=False)
