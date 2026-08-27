"""불단왕 5강 주장 검증: "4h RSI 다이버전스만 추적해도 BTC/ETH 승률 높다"

기계화 규칙 (사전 고정, 튜닝 없음):
- 4h 봉 (1h 동결 데이터 리샘플), RSI(14)
- 피벗: 좌우 2봉 극값 (확정은 2봉 뒤 → 진입은 확정봉 종가, 룩어헤드 차단)
- 상승 다이버전스: 가격 저점 하락 + RSI 저점 상승 (최근 두 피벗, 50봉 이내) → 롱
- 하락 다이버전스: 가격 고점 상승 + RSI 고점 하락 → 숏
- 스탑: 피벗 극값 ∓ 0.5×ATR(14) / 목표: 2R / 타임아웃 42봉(7일) 종가 청산
- 비용: 왕복 16bp (taker+슬립)
"""
from __future__ import annotations
import numpy as np, pandas as pd

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    tr = pd.concat([df.high-df.low, (df.high-df.close.shift()).abs(),
                    (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

RT_COST = 0.0016
K = 2          # 피벗 좌우 봉
LOOKBACK = 50
TIMEOUT = 42

def run(sym):
    p = pd.read_parquet('lab/frozen/perp_1h.parquet').xs(sym, level='sym')
    df = pd.DataFrame({'open': p.open.resample('4h').first(), 'high': p.high.resample('4h').max(),
                       'low': p.low.resample('4h').min(), 'close': p.close.resample('4h').last()}).dropna()
    r = rsi(df.close); a = atr(df)
    lows, highs = [], []          # (idx, price, rsi)
    trades = []
    pos = None
    for i in range(K, len(df) - 1):
        # 피벗 확정 (i-K 위치가 극값인지, i 시점에 확정됨)
        j = i - K
        win = df.low.iloc[j-K:j+K+1]
        if len(win) == 2*K+1 and df.low.iloc[j] == win.min():
            lows.append((j, df.low.iloc[j], r.iloc[j]))
        win = df.high.iloc[j-K:j+K+1]
        if len(win) == 2*K+1 and df.high.iloc[j] == win.max():
            highs.append((j, df.high.iloc[j], r.iloc[j]))

        px = df.close.iloc[i]
        # 포지션 관리
        if pos is not None:
            d_, e, stop, tgt, ei = pos
            hi, lo = df.high.iloc[i], df.low.iloc[i]
            done = None
            if d_ > 0:
                if lo <= stop: done = (stop - e)/e
                elif hi >= tgt: done = (tgt - e)/e
            else:
                if hi >= stop: done = (e - stop)/e
                elif lo <= tgt: done = (e - tgt)/e
            if done is None and i - ei >= TIMEOUT:
                done = d_*(px - e)/e
            if done is not None:
                trades.append(dict(ts=df.index[i], d=d_, ret=done - RT_COST))
                pos = None
            continue
        # 신규 신호 (확정된 피벗 2개 비교)
        if len(lows) >= 2:
            (j1, p1, r1), (j2, p2, r2) = lows[-2], lows[-1]
            if j2 - j1 <= LOOKBACK and j2 == i - K and p2 < p1 and r2 > r1 and not np.isnan(a.iloc[i]):
                stop = p2 - 0.5*a.iloc[i]; e = px
                pos = (1, e, stop, e + 2*(e - stop), i); continue
        if len(highs) >= 2:
            (j1, p1, r1), (j2, p2, r2) = highs[-2], highs[-1]
            if j2 - j1 <= LOOKBACK and j2 == i - K and p2 > p1 and r2 < r1 and not np.isnan(a.iloc[i]):
                stop = p2 + 0.5*a.iloc[i]; e = px
                pos = (-1, e, stop, e - 2*(stop - e), i)
    return pd.DataFrame(trades)

for sym in ('BTC', 'ETH'):
    t = run(sym)
    if not len(t): print(sym, "신호 없음"); continue
    wr = (t.ret > 0).mean()
    yrs = (t.ts.iloc[-1] - t.ts.iloc[0]).days / 365.25
    print(f"\n{sym}: 거래 {len(t)}건 ({len(t)/yrs:.0f}건/년, {t.ts.iloc[0].date()}~)")
    print(f"  승률 {wr*100:.1f}%  평균 {t.ret.mean()*100:+.3f}%/건  누적 {t.ret.sum()*100:+.1f}%")
    print(f"  롱 {int((t.d>0).sum())}건 승률 {(t[t.d>0].ret>0).mean()*100:.0f}%  "
          f"숏 {int((t.d<0).sum())}건 승률 {(t[t.d<0].ret>0).mean()*100:.0f}%")
    yr = t.groupby(t.ts.dt.year).ret.sum()*100
    print("  연도별 누적: " + "  ".join(f"{y}:{v:+.1f}%" for y, v in yr.items()))
