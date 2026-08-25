"""레짐 게이팅 캐리 — 예상 캐리가 허들 넘을 때만 자본 투입.

핵심 질문: 2022/2026 같은 저캐리 레짐을 회피하면 수익이 유의하게 개선되는가?
게이트는 '경제적 허들'(예상펀딩 > 비용+요구프리미엄)이며 성과최적화 탐색이 아님.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from lab.carry_falsifier import load, series, px_at

def daily_funding_panel(fund, perp, syms):
    """일별 펀딩 수익률 패널(명목 대비). 시점 t의 값 = t에 정산된 펀딩."""
    out = {}
    for s in syms:
        f = fund[s].dropna()
        out[s] = f.resample('D').sum()
    return pd.DataFrame(out).fillna(0.0)

def gated_carry(fund, syms, lookback_d=30, hurdle_ann=0.0, cost_rt=0.0022,
                min_hold_d=14, start=None, end=None):
    """트레일링 펀딩이 허들 초과 시 보유, 미만 시 현금.

    - 결정은 t일 종료 시점의 '이미 정산된' 펀딩만 사용 (룩어헤드 없음)
    - 진입/청산 시 왕복비용 cost_rt 부과
    - min_hold_d: 최소보유일 (채터링 억제)
    """
    d = daily_funding_panel(fund, None, syms)
    if start is not None: d = d[d.index >= pd.Timestamp(start, tz='utc')]
    if end is not None: d = d[d.index <= pd.Timestamp(end, tz='utc')]
    port = d.mean(axis=1)                      # 등가중 캐리 바스켓 일별 수익
    trail = port.rolling(lookback_d).mean().shift(1)   # shift(1): 결정시점 정보만
    hurdle_d = hurdle_ann / 365.0
    pos, held, rets, states = 0, 0, [], []
    for t, r in port.items():
        sig = trail.get(t, np.nan)
        cost = 0.0
        if not np.isnan(sig):
            if pos == 0 and sig > hurdle_d:
                pos, held, cost = 1, 0, cost_rt / 2      # 진입비용(편도)
            elif pos == 1:
                held += 1
                if sig <= hurdle_d and held >= min_hold_d:
                    pos, cost = 0, cost_rt / 2            # 청산비용(편도)
        rets.append(pos * r - cost)
        states.append(pos)
    out = pd.Series(rets, index=port.index)
    return out, pd.Series(states, index=port.index), port

def summarize(r, label):
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    tot = float((1 + r).prod())
    cagr = tot ** (1 / yrs) - 1
    eq = (1 + r).cumprod()
    mdd = float((1 - eq / eq.cummax()).max())
    ann_vol = r.std() * np.sqrt(365)
    sharpe = (r.mean() * 365) / ann_vol if ann_vol > 0 else np.nan
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    print(f"\n── {label}")
    print(f"   CAGR(명목) {cagr*100:+.2f}%  MDD {mdd*100:.2f}%  Sharpe {sharpe:.2f}  ({yrs:.1f}y)")
    print("   연도별: " + "  ".join(f"{y}:{v*100:+.2f}%" for y, v in yr.items()))
    return dict(label=label, cagr=cagr, mdd=mdd, sharpe=sharpe)

if __name__ == '__main__':
    perp, spot, fund = load()
    syms = ['BTC', 'ETH']
    print("=" * 78); print("레짐 게이팅 캐리 — 저캐리 구간 회피 효과 (BTC/ETH 등가중)"); print("=" * 78)
    base, _, port = gated_carry(fund, syms, hurdle_ann=-1e9, cost_rt=0.0022, start='2021-07-05')
    summarize(base, "게이트 없음 (항상 보유)")
    for h in (0.02, 0.04, 0.06, 0.08, 0.10):
        r, st, _ = gated_carry(fund, syms, hurdle_ann=h, cost_rt=0.0022, start='2021-07-05')
        s = summarize(r, f"게이트 허들 {h*100:.0f}%/yr")
        print(f"   투입비율 {st.mean()*100:.1f}%  전환횟수 {int(st.diff().abs().sum())}")
