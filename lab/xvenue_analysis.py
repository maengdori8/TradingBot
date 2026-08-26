"""교차 거래소 펀딩 차이 분석 — 크기·지속성·정직한 거래 시뮬."""
from __future__ import annotations
import numpy as np, pandas as pd

hl = pd.read_parquet('lab/data/hl_funding.parquet')
by = pd.read_parquet('lab/frozen/funding.parquet')

COINS = [c for c in hl.columns if c in by.columns]
hl_d = hl.resample('D').sum(min_count=1)
by_d = by[COINS].resample('D').sum(min_count=1)

print("=" * 100)
print("1) 동일 기간 펀딩 비교 (일별 합산, 연율 %)")
print("=" * 100)
print(f"{'코인':>6s} {'기간':>24s} {'일수':>6s} {'Bybit':>8s} {'HL':>8s} {'차이(B-H)':>10s} {'|차이|':>8s} {'부호지속':>8s}")
diffs = {}
for c in COINS:
    j = pd.concat({'b': by_d[c], 'h': hl_d[c]}, axis=1).dropna()
    if len(j) < 200: continue
    d = j['b'] - j['h']
    diffs[c] = d
    sgn = np.sign(d.rolling(7).sum())
    persist = float((sgn == sgn.shift(1)).mean())
    print(f"{c:>6s} {str(j.index[0].date())+'~'+str(j.index[-1].date()):>24s} {len(j):6d} "
          f"{j['b'].mean()*365*100:+7.2f}% {j['h'].mean()*365*100:+7.2f}% "
          f"{d.mean()*365*100:+9.2f}% {d.abs().mean()*365*100:7.2f}% {persist*100:7.1f}%")

print()
print("=" * 100)
print("2) 정직한 거래 시뮬 — 트레일링 7일 차이로 방향 결정 (shift 1일, 룩어헤드 없음)")
print("   포지션: 펀딩 높은 쪽 숏 + 낮은 쪽 롱. 최소보유 7일. 진입허들 = 2×왕복비용×365/7")
print("=" * 100)
for rt_bp, label in [(4, 'maker 4bp'), (8, '혼합 8bp'), (16, 'taker+슬립 16bp')]:
    rt = rt_bp / 1e4
    hurdle = 2 * rt * 365 / 7
    print(f"\n── 왕복 {label} (허들 {hurdle*100:.1f}%/yr) ──")
    print(f"{'코인':>6s} {'그로스':>8s} {'전환/년':>7s} {'비용/년':>8s} {'순연율':>8s} {'투입비율':>8s} {'양수월%':>7s}")
    for c, d in diffs.items():
        trail = d.rolling(7).mean().shift(1) * 365
        pos = pd.Series(0.0, index=d.index)   # +1: Bybit숏/HL롱 (B-H 수취)
        cur, held = 0, 99
        vals = []
        for t, tr in trail.items():
            held += 1
            if not np.isnan(tr):
                if cur == 0:
                    if tr > hurdle: cur, held = 1, 0
                    elif tr < -hurdle: cur, held = -1, 0
                elif held >= 7:
                    if cur == 1 and tr < 0: cur = 0
                    elif cur == -1 and tr > 0: cur = 0
            vals.append(cur)
        pos = pd.Series(vals, index=d.index, dtype=float)
        pnl = pos * d
        switches = pos.diff().abs().fillna(0)
        yrs = len(d) / 365.25
        gross = pnl.sum() / yrs * 100
        cost = switches.sum() * rt / yrs * 100
        mo = pnl.resample('ME').sum()
        mo = mo[pos.resample('ME').apply(lambda x: (x != 0).any())]
        print(f"{c:>6s} {gross:+7.2f}% {switches.sum()/2/yrs:6.1f} {cost:7.2f}% "
              f"{gross-cost:+7.2f}% {(pos!=0).mean()*100:7.1f}% "
              f"{(mo>0).mean()*100 if len(mo) else 0:6.0f}%")

print()
print("=" * 100)
print("3) 연도별 순수익 (혼합 8bp, 전 코인 등가중)")
print("=" * 100)
rt = 8 / 1e4; hurdle = 2 * rt * 365 / 7
port = []
for c, d in diffs.items():
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
    # 재실행 (위 루프 버그 방지용 명시 재구성)
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
    port.append(net)
P = pd.concat(port, axis=1).mean(axis=1).dropna()
yr = P.groupby(P.index.year).sum() * 100
print("  " + "  ".join(f"{y}:{v:+.2f}%" for y, v in yr.items()))
print(f"  전체 연율 {P.sum()/ (len(P)/365.25) * 100:+.2f}%  일변동성 {P.std()*100:.4f}%  "
      f"최악일 {P.min()*100:+.3f}%")
