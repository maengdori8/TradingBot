"""현물-perp 캐시앤캐리 최속 반증 테스트.

Codex 합의 설계:
- BTC/ETH 등가중, 롱 현물 + 숏 등델타 perp
- 매월 1일 UTC 진입, 30일 후 청산 (강제 왕복 = 의도적으로 비관적)
- 펀딩 예측·종목선택 없음 (선택편향 0)
- taker 실행 우선(큐모델 논쟁 제거), maker는 낙관적 상한으로 별도 산출
- r = Σfunding + basis_entry - basis_exit - fees - slippage
- 총 투입자본 기준 수익률 병기
"""
from __future__ import annotations
import numpy as np, pandas as pd

FEE = {  # per side, fraction of notional
    'taker': {'spot': 0.0010, 'perp': 0.0006},
    'maker': {'spot': 0.0010, 'perp': 0.0001},   # Bybit 현물은 maker도 0.1%
}

def load():
    perp = pd.read_parquet('lab/data/perp_1h.parquet')
    spot = pd.read_parquet('lab/data/spot_1h.parquet')
    fund = pd.read_parquet('lab/data/funding_8h.parquet')
    return perp, spot, fund

def series(df, sym, col='close'):
    s = df.xs(sym, level='sym')[col]
    return s[~s.index.duplicated()].sort_index()

def px_at(s: pd.Series, ts: pd.Timestamp, tol_h: int = 3):
    """ts 이하의 가장 가까운 관측 (미래 참조 금지)."""
    i = s.index.searchsorted(ts, side='right') - 1
    if i < 0: return np.nan
    if (ts - s.index[i]) > pd.Timedelta(hours=tol_h): return np.nan
    return float(s.iloc[i])

def run_block(sym, t0, t1, perp, spot, fund, exec_mode, slip_bp):
    """한 보유블록의 명목 대비 수익률과 진단값."""
    P, S = series(perp, sym), series(spot, sym)
    f = fund[sym].dropna()
    p0, p1, s0, s1 = px_at(P, t0), px_at(P, t1), px_at(S, t0), px_at(S, t1)
    if any(np.isnan(x) for x in (p0, p1, s0, s1)): return None

    # 보유기간 중 정산된 펀딩 (진입 후 ~ 청산 시점까지)
    fw = f[(f.index > t0) & (f.index <= t1)]
    if len(fw) == 0: return None
    # 숏 perp: 펀딩>0 이면 수취. 각 정산 시점의 perp 가격 명목 기준
    fund_pnl = float(sum(r * px_at(P, ts) for ts, r in fw.items() if not np.isnan(px_at(P, ts))))

    fe = FEE[exec_mode]
    slip = slip_bp / 1e4
    fees = (s0 * (fe['spot'] + slip) + p0 * (fe['perp'] + slip)
            + s1 * (fe['spot'] + slip) + p1 * (fe['perp'] + slip))

    price_pnl = (s1 - s0) + (p0 - p1)          # = basis0 - basis1
    pnl = price_pnl + fund_pnl - fees
    return dict(sym=sym, t0=t0, t1=t1, ret=pnl / s0,
                fund_r=fund_pnl / s0, basis_r=price_pnl / s0, fee_r=fees / s0,
                n_fund=len(fw), basis0=(p0 - s0) / s0, basis1=(p1 - s1) / s1,
                s0=s0, s1=s1, p0=p0, p1=p1)

def backtest(exec_mode='taker', slip_bp=2.0, hold_days=30, syms=('BTC','ETH'), end=None):
    perp, spot, fund = load()
    start = max(series(spot, s).index[0] for s in syms).ceil('D') + pd.Timedelta(days=1)
    start = pd.Timestamp(year=start.year, month=start.month, day=1, tz='utc') + pd.DateOffset(months=1)
    end = pd.Timestamp(end, tz='utc') if end else min(series(perp, s).index[-1] for s in syms)
    rows, t0 = [], start
    while t0 + pd.Timedelta(days=hold_days) <= end:
        t1 = t0 + pd.Timedelta(days=hold_days)
        blk = [run_block(s, t0, t1, perp, spot, fund, exec_mode, slip_bp) for s in syms]
        blk = [b for b in blk if b]
        if blk:
            rows.append(dict(t0=t0, t1=t1, ret=float(np.mean([b['ret'] for b in blk])),
                             fund_r=float(np.mean([b['fund_r'] for b in blk])),
                             basis_r=float(np.mean([b['basis_r'] for b in blk])),
                             fee_r=float(np.mean([b['fee_r'] for b in blk])),
                             n=len(blk), detail=blk))
        t0 = t0 + pd.DateOffset(months=1)
    return pd.DataFrame(rows)

def block_bootstrap_lb(r, n_boot=20000, block=3, alpha=0.05, seed=7):
    """정상 블록 부트스트랩 — 기하평균 월수익의 단측 95% 하한."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, float); n = len(r)
    if n < 4: return np.nan
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx, out = [], 0
        while out < n:
            st = rng.integers(0, n); L = rng.geometric(1 / block)
            idx.extend((st + np.arange(L)) % n); out += L
        samp = r[np.array(idx[:n])]
        means[b] = np.expm1(np.mean(np.log1p(samp)))
    return float(np.quantile(means, alpha))

def report(df, label, margin_frac=1.0):
    r = df['ret'].to_numpy()
    n = len(r)
    geo_m = np.expm1(np.mean(np.log1p(r)))
    cagr_notional = (1 + geo_m) ** (365 / 30) - 1
    # 총 투입자본 = 현물 1 + perp 증거금 margin_frac  → 수익률 희석
    r_eq = r / (1 + margin_frac)
    geo_eq = np.expm1(np.mean(np.log1p(r_eq)))
    cagr_eq = (1 + geo_eq) ** (365 / 30) - 1
    equity = np.cumprod(1 + r_eq)
    mdd = float((1 - equity / np.maximum.accumulate(equity)).max())
    lb = block_bootstrap_lb(r)
    lb_cagr = (1 + lb) ** (365 / 30) - 1 if not np.isnan(lb) else np.nan
    print(f"\n── {label} (n={n} blocks, {df['t0'].iloc[0].date()}~{df['t1'].iloc[-1].date()})")
    print(f"   월평균(기하) {geo_m*100:+.3f}%  |  명목기준 CAGR {cagr_notional*100:+.2f}%")
    print(f"   투입자본기준(margin={margin_frac:.2f}) CAGR {cagr_eq*100:+.2f}%  MDD {mdd*100:.2f}%")
    print(f"   양수블록 {(r>0).mean()*100:.1f}%  최악블록 {r.min()*100:+.2f}%  최고 {r.max()*100:+.2f}%")
    print(f"   분해: 펀딩 {df['fund_r'].mean()*100:+.3f}%  베이시스 {df['basis_r'].mean()*100:+.3f}%  "
          f"비용 {df['fee_r'].mean()*100:-.3f}%")
    print(f"   부트스트랩 월수익 95% 단측하한 {lb*100:+.4f}%  (연율 {lb_cagr*100:+.2f}%)")
    print(f"   >>> 판정: {'통과 (하한>0)' if lb > 0 else '실패 (하한<=0)'}")
    return dict(label=label, n=n, geo_m=geo_m, cagr_notional=cagr_notional,
                cagr_eq=cagr_eq, mdd=mdd, lb=lb, win=(r > 0).mean())

if __name__ == '__main__':
    print("=" * 78)
    print("현물-perp 캐시앤캐리 반증 테스트 — 월별 강제 왕복, 종목선택 없음")
    print("=" * 78)
    out = []
    for mode, slip in [('taker', 2.0), ('taker', 0.0), ('maker', 0.0)]:
        df = backtest(exec_mode=mode, slip_bp=slip)
        lbl = f"{mode} slip={slip}bp"
        out.append(report(df, lbl))
        df.drop(columns=['detail']).to_csv(f'lab/data/carry_{mode}_{int(slip)}.csv', index=False)
    print("\n" + "=" * 78)
