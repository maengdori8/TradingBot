"""사전적 유니버스 규칙 — 관측된 펀딩을 절대 참조하지 않는다.

규칙 (선택편향 차단):
  1. 현물·perp 양쪽 상장
  2. 상장 후 min_age_days 경과 (시점보존)
  3. 거래대금(ADV) 상위 N — 유동성은 반응변수가 아님
LINK/LTC를 '펀딩이 높아서' 고르는 것은 금지. 오직 위 규칙만 사용한다.
"""
from __future__ import annotations
import numpy as np, pandas as pd

F = 'lab/frozen'

def load_frozen():
    fund = pd.read_parquet(f'{F}/funding.parquet')
    perp = pd.read_parquet(f'{F}/perp_1d.parquet')
    spot = pd.read_parquet(f'{F}/spot_1d.parquet')
    uni = pd.read_parquet(f'{F}/universe.parquet')
    return fund, perp, spot, uni

def adv_panel(perp: pd.DataFrame) -> pd.DataFrame:
    """일별 거래대금(종가×거래량)의 30일 중앙값 패널."""
    q = {}
    for s in perp.index.get_level_values('sym').unique():
        d = perp.xs(s, level='sym')
        q[s] = (d['close'] * d['volume']).rolling(30, min_periods=20).median()
    return pd.DataFrame(q)

def build(min_age_days=1095, top_n=12, min_adv_usd=2e7):
    fund, perp, spot, uni = load_frozen()
    u = uni[(uni.contractType == 'LinearPerpetual') & (uni.quoteCoin == 'USDT')]
    launch = u.groupby('baseCoin')['launchTime'].min()
    delist = u[u.status == 'Closed'].groupby('baseCoin')['deliveryTime'].max()

    adv = adv_panel(perp)
    syms = [s for s in adv.columns if s in fund.columns]
    adv = adv[syms]
    ages = pd.DataFrame({s: (adv.index - launch.get(s, pd.NaT)).days for s in syms}, index=adv.index)

    elig = (ages >= min_age_days) & (adv >= min_adv_usd)
    for s in syms:
        if s in delist.index and pd.notna(delist[s]):
            elig.loc[elig.index >= delist[s] - pd.Timedelta(days=30), s] = False

    # 매일 ADV 상위 top_n만 선택 (시점보존)
    sel = pd.DataFrame(False, index=adv.index, columns=syms)
    for t in adv.index:
        cand = adv.loc[t][elig.loc[t]].dropna()
        if len(cand) == 0: continue
        sel.loc[t, list(cand.nlargest(min(top_n, len(cand))).index)] = True
    return sel, adv, elig

if __name__ == '__main__':
    fund, perp, spot, uni = load_frozen()
    sel, adv, elig = build()
    print(f"적격(연수+유동성) 일평균 종목수: {elig.sum(axis=1).mean():.1f}")
    print(f"선택(ADV상위12) 일평균: {sel.sum(axis=1).mean():.1f}  최근: {int(sel.iloc[-1].sum())}")
    print(f"\n최근 선택 종목: {sorted(sel.columns[sel.iloc[-1]])}")
    freq = sel.sum().sort_values(ascending=False)
    freq = freq[freq > 0]
    print(f"\n선택 빈도 상위 20 (총 {len(freq)}종목이 한번이라도 선택됨):")
    print((freq.head(20) / len(sel) * 100).round(1).to_string())
    # 이 유니버스의 펀딩 프로파일 — 규칙 확정 '후에' 관측
    d = fund.resample('D').sum(min_count=1)
    ann = {s: d[s].reindex(sel.index)[sel[s]].mean() * 365 * 100
           for s in freq.index if s in d.columns}
    ann = pd.Series(ann).dropna().sort_values(ascending=False)
    print(f"\n[규칙 확정 후 관측] 선택기간 중 펀딩 연율: median {ann.median():+.2f}%  "
          f"mean {ann.mean():+.2f}%  min {ann.min():+.2f}%  max {ann.max():+.2f}%")
    print(ann.round(1).to_string())
