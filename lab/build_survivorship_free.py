"""생존편향 교정 패널 — 폐지 종목을 포함한 캐리 순위 유니버스를 만든다.

폐지 4종목(MATIC/FTM/TON/UNFI)은 kline이 0행이라 가격을 얻을 수 없다.
그러나 (1) 펀딩은 API로 전량 조회되고 (2) 델타중립 캐리의 장기 손익에서
베이시스 기여는 실측상 0에 가깝다(-0.004%/월). 따라서 이 종목들은
'펀딩만' 반영하고 베이시스를 0으로 두는 근사로 포함한다. 근사의 방향은
보수적이지 않지만(수익 과소·과대 불명), 생존편향 제거 효과가 훨씬 크다.
ADV는 아카이브 표본 추정치를 상수로 사용한다.
"""
from __future__ import annotations
import ccxt, time
import pandas as pd, numpy as np

DELISTED = {   # symbol: (base, 표본 ADV USD)
    'MATICUSDT': ('MATIC', 48.7e6), 'FTMUSDT': ('FTM', 49.3e6),
    'TONUSDT': ('TON', 44.7e6), 'UNFIUSDT': ('UNFI', 59.7e6),
}

def retry(fn, *a, **k):
    for i in range(6):
        try: return fn(*a, **k)
        except Exception:
            if i == 5: return None
            time.sleep(1.5 * (i + 1))

def funding_back(ex, mid, floor_ms):
    rows, end = [], int(time.time() * 1000)
    while end > floor_ms:
        r = retry(ex.publicGetV5MarketFundingHistory,
                  {'category': 'linear', 'symbol': mid, 'endTime': end, 'limit': 200})
        if not r: break
        lst = r.get('result', {}).get('list', [])
        if not lst: break
        for x in lst: rows.append((int(x['fundingRateTimestamp']), float(x['fundingRate'])))
        o = min(int(x['fundingRateTimestamp']) for x in lst)
        if o >= end: break
        end = o - 1
        if len(lst) < 200: break
    if not rows: return None
    return (pd.DataFrame(rows, columns=['ts', 'f']).drop_duplicates('ts')
            .assign(ts=lambda d: pd.to_datetime(d.ts, unit='ms', utc=True))
            .set_index('ts').sort_index()['f'])

if __name__ == '__main__':
    ex = ccxt.bybit({'enableRateLimit': True})
    floor = int(pd.Timestamp('2021-01-01', tz='utc').timestamp() * 1000)
    uni = pd.read_parquet('lab/frozen/universe.parquet')
    meta = uni[(uni.contractType == 'LinearPerpetual') & (uni.quoteCoin == 'USDT')]

    fu, adv, launch, delist = {}, {}, {}, {}
    for mid, (base, a) in DELISTED.items():
        s = funding_back(ex, mid, floor)
        if s is None: print(f"{mid}: 실패"); continue
        fu[base] = s; adv[base] = a
        m = meta[meta.symbol == mid].iloc[0]
        launch[base], delist[base] = m.launchTime, m.deliveryTime
        print(f"{base:6s} 펀딩 {len(s):5d}건 {s.index[0].date()}~{s.index[-1].date()} "
              f"상장 {launch[base].date()} 폐지 {delist[base].date()} ADV ${a/1e6:.1f}M")

    live_f = pd.read_parquet('lab/frozen/funding.parquet')
    all_f = pd.concat([live_f, pd.DataFrame(fu)], axis=1).sort_index()
    all_f = all_f.loc[:, ~all_f.columns.duplicated()]
    all_f.to_parquet('lab/frozen/funding_survfree.parquet')

    # 폐지 종목의 시점보존 메타 (생존 종목 메타에 추가)
    rows = [dict(symbol=f'{b}USDT', status='Closed', contractType='LinearPerpetual',
                 baseCoin=b, quoteCoin='USDT', launchTime=launch[b],
                 deliveryTime=delist[b], fundingInterval=480) for b in fu]
    uni2 = pd.concat([uni, pd.DataFrame(rows)], ignore_index=True)
    uni2.to_parquet('lab/frozen/universe_survfree.parquet')
    pd.Series(adv).to_frame('adv').to_parquet('lab/frozen/delisted_adv_const.parquet')
    print(f"\n저장: funding_survfree({all_f.shape}), universe_survfree({len(uni2)}), delisted_adv_const")
