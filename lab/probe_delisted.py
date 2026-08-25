"""폐지 종목의 실제 거래대금 표본조사 — 생존편향이 유니버스를 바꿨는지 판정.

전량 다운로드는 비현실적(178종목 x 1000일)이므로 수명 구간에 균등한 표본일만 받아
중앙 일거래대금을 추정한다. 임계 초과 종목만 이후 전량 복원 대상이 된다.
"""
from __future__ import annotations
import gzip, io, sys
import urllib.request as U
import pandas as pd, numpy as np

BASE = "https://public.bybit.com/trading"

def day_notional(sym: str, day: pd.Timestamp) -> float | None:
    """해당 일의 체결 명목 합계(USDT)를 아카이브에서 계산한다."""
    url = f"{BASE}/{sym}/{sym}{day.date()}.csv.gz"
    try:
        with U.urlopen(url, timeout=40) as r:
            raw = r.read()
    except Exception:
        return None
    try:
        df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), usecols=['size', 'price'])
    except Exception:
        return None
    return float((df['size'] * df['price']).sum())

def sample_adv(sym: str, lo: pd.Timestamp, hi: pd.Timestamp, k: int = 5) -> tuple[float, int]:
    """수명 구간에서 k개 표본일의 중앙 거래대금과 성공 표본수."""
    days = pd.date_range(lo + pd.Timedelta(days=30), hi - pd.Timedelta(days=5), periods=k)
    vals = [v for d in days if (v := day_notional(sym, d)) is not None]
    return (float(np.median(vals)) if vals else 0.0), len(vals)

if __name__ == '__main__':
    uni = pd.read_parquet('lab/frozen/universe.parquet')
    c = uni[(uni.contractType == 'LinearPerpetual') & (uni.quoteCoin == 'USDT')
            & (uni.status == 'Closed')].copy()
    c['life'] = (c.deliveryTime - c.launchTime).dt.days
    c = c[c.life >= 365].sort_values('life', ascending=False)
    print(f"수명 1년 이상 폐지 perp {len(c)}개 표본조사 (각 5일)\n")
    print(f"{'symbol':>16s} {'launch':>11s} {'delist':>11s} {'중앙일거래대금':>15s} {'표본':>5s} {'$20M초과':>9s}")
    rows = []
    for _, r in c.iterrows():
        adv, n = sample_adv(r['symbol'], r.launchTime, r.deliveryTime)
        rows.append(dict(symbol=r['symbol'], base=r.baseCoin, launch=r.launchTime,
                         delist=r.deliveryTime, adv=adv, n=n))
        flag = 'YES' if adv >= 2e7 else ''
        print(f"{r['symbol']:>16s} {str(r.launchTime.date()):>11s} {str(r.deliveryTime.date()):>11s} "
              f"${adv/1e6:13.2f}M {n:5d} {flag:>9s}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet('lab/data/delisted_adv.parquet')
    big = df[df.adv >= 2e7]
    print(f"\n$20M 초과: {len(big)}/{len(df)}  →  {sorted(big.base)}")
