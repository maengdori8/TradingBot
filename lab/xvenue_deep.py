"""교차 거래소 차익 심층 검증 — 스캔 상위 코인의 전체 이력 수집 + 정직 시뮬."""
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

scan = pd.read_csv('lab/data/xvenue_scan.csv')
liq = scan[scan.min_vol >= 5].nlargest(15, 'abs_diff')
COINS = list(liq.coin)
print("대상:", COINS)

hl_all, by_all = {}, {}
for c in COINS:
    # HL 전체 이력 (시작부터 앞으로)
    rows, start = [], 0
    while True:
        d = hl_post({"type": "fundingHistory", "coin": c, "startTime": start})
        if not d: break
        rows += [(int(x['time']), float(x['fundingRate'])) for x in d]
        if len(d) < 500: break
        start = int(d[-1]['time']) + 1
        time.sleep(0.12)
    if rows:
        s = (pd.DataFrame(rows, columns=['ts','f']).drop_duplicates('ts')
             .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
             .set_index('ts').sort_index()['f'])
        hl_all[c] = s
    # Bybit 역방향
    by_name = ('1000' + c[1:]) if c.startswith('k') else c
    rows, end = [], int(time.time()*1000)
    while True:
        d = bybit_get(f'https://api.bybit.com/v5/market/funding/history?category=linear'
                      f'&symbol={by_name}USDT&endTime={end}&limit=200')
        if not d: break
        lst = d['result']['list']
        if not lst: break
        rows += [(int(x['fundingRateTimestamp']), float(x['fundingRate'])) for x in lst]
        oldest = min(int(x['fundingRateTimestamp']) for x in lst)
        if oldest >= end or len(lst) < 200: break
        end = oldest - 1
        time.sleep(0.1)
    if rows:
        s = (pd.DataFrame(rows, columns=['ts','f']).drop_duplicates('ts')
             .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
             .set_index('ts').sort_index()['f'])
        by_all[c] = s
    hn = len(hl_all.get(c, [])); bn = len(by_all.get(c, []))
    print(f"  {c:9s} HL {hn:6d}건  Bybit {bn:5d}건", flush=True)

pd.DataFrame(hl_all).to_parquet('lab/data/xv_hl_deep.parquet')
pd.DataFrame(by_all).to_parquet('lab/data/xv_by_deep.parquet')

# ── 정직 시뮬 (메이저와 동일 규칙) ──
print("\n" + "=" * 96)
print("정직 시뮬 — 트레일링 7일, shift 1일, 최소보유 7일, 혼합 8bp (허들 8.3%/yr)")
print("=" * 96)
rt = 8/1e4; hurdle = 2*rt*365/7
print(f"{'코인':>9s} {'이력':>14s} {'|차이|연율':>9s} {'그로스':>8s} {'순연율':>8s} {'투입':>6s} {'양수월':>6s} {'최근90일순':>10s}")
port = {}
for c in COINS:
    if c not in hl_all or c not in by_all: continue
    h = hl_all[c].resample('D').sum(min_count=1)
    b = by_all[c].resample('D').sum(min_count=1)
    j = pd.concat({'b': b, 'h': h}, axis=1).dropna()
    if len(j) < 120: 
        print(f"{c:>9s} 공통이력 부족 ({len(j)}일)"); continue
    d = j['b'] - j['h']
    trail = d.rolling(7).mean().shift(1) * 365
    cur, held, vals = 0, 99, []
    for t, tr in trail.items():
        held += 1
        if not np.isnan(tr):
            if cur == 0:
                if tr > hurdle: cur, held = 1, 0
                elif tr < -hurdle: cur, held = -1, 0
            elif held >= 7:
                if (cur == 1 and tr < 0) or (cur == -1 and tr > 0): cur = 0
        vals.append(cur)
    pos = pd.Series(vals, index=d.index, dtype=float)
    net = pos * d - pos.diff().abs().fillna(0) * rt
    port[c] = net
    yrs = len(d)/365.25
    mo = net.resample('ME').sum(); mo = mo[mo != 0]
    r90 = net.tail(90).sum()*365/90*100
    print(f"{c:>9s} {str(j.index[0].date()):>14s} {d.abs().mean()*365*100:8.2f}% "
          f"{(pos*d).sum()/yrs*100:+7.2f}% {net.sum()/yrs*100:+7.2f}% "
          f"{(pos!=0).mean()*100:5.0f}% {(mo>0).mean()*100 if len(mo) else 0:5.0f}% {r90:+9.2f}%")

P = pd.concat(port, axis=1)
eq = P.mean(axis=1).dropna()
yr = eq.groupby(eq.index.year).sum()*100
print(f"\n포트폴리오(등가중) 연도별: " + "  ".join(f"{y}:{v:+.2f}%" for y, v in yr.items()))
print(f"최근 90일 연환산: {eq.tail(90).sum()*365/90*100:+.2f}%  일변동성 {eq.std()*100:.4f}%  최악일 {eq.min()*100:+.3f}%")
