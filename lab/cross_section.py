"""횡단면 캐리 선택 — Codex 요구: raw top-N이 아니라 '증분 가치'를 측정.

측정 대상: R_topN - R_equalweight_eligible  (등가중 대비 초과분)
검증: 시점보존 유니버스, 상장연수 필터, 정직 WFO(파라미터도 train 선택), 시간클러스터 부트스트랩.
펀딩 주기가 심볼마다 다르므로(8h/4h/1h) 항상 '일별 합산'으로 정규화한다.
"""
from __future__ import annotations
import numpy as np, pandas as pd

def load_panel(min_days: int = 400):
    f = pd.read_parquet('lab/data/broad_funding_8h.parquet')
    d = f.resample('D').sum(min_count=1)          # 주기 무관 일별 캐리 수익
    alive = f.notna().resample('D').max().fillna(False)
    d = d.where(alive)                             # 미상장 구간은 NaN 유지
    keep = d.notna().sum()
    d = d[keep[keep >= min_days].index]
    return d

def uni_meta():
    u = pd.read_parquet('lab/data/universe_linear.parquet')
    u = u[(u.contractType == 'LinearPerpetual') & (u.quoteCoin == 'USDT')]
    return u.set_index('baseCoin')[['status', 'launchTime', 'deliveryTime']]

def eligible_mask(d: pd.DataFrame, meta: pd.DataFrame, min_age_days=365) -> pd.DataFrame:
    """시점보존 적격성: 상장 후 min_age_days 경과 + 데이터 존재."""
    m = pd.DataFrame(False, index=d.index, columns=d.columns)
    for c in d.columns:
        if c not in meta.index: continue
        lt = meta.loc[c, 'launchTime']
        lt = lt.iloc[0] if hasattr(lt, 'iloc') else lt
        if pd.isna(lt): continue
        ok = d.index >= (lt + pd.Timedelta(days=min_age_days))
        m[c] = ok & d[c].notna()
    return m

def topn_vs_equal(d, elig, lookback=30, N=10, rebal=30, cost_rt=0.0022):
    """top-N 선택 포트폴리오와 적격 등가중 벤치마크의 일별 수익."""
    trail = d.rolling(lookback, min_periods=lookback // 2).mean().shift(1)
    dates = d.index
    top_r, eq_r, held = [], [], None
    for i, t in enumerate(dates):
        av = elig.loc[t]
        av = av[av].index
        if len(av) < N + 5:
            top_r.append(np.nan); eq_r.append(np.nan); continue
        if held is None or i % rebal == 0:
            sc = trail.loc[t, av].dropna()
            new = list(sc.nlargest(min(N, len(sc))).index) if len(sc) >= N else None
            if new is not None:
                turn = 0.0 if held is None else len(set(new) ^ set(held)) / (2 * N)
                held = new
                top_r.append(float(d.loc[t, held].mean()) - turn * cost_rt)
                eq_r.append(float(d.loc[t, av].mean()))
                continue
        if held is None:
            top_r.append(np.nan); eq_r.append(np.nan); continue
        cur = [s for s in held if s in av and not np.isnan(d.loc[t, s])]
        top_r.append(float(d.loc[t, cur].mean()) if cur else np.nan)
        eq_r.append(float(d.loc[t, av].mean()))
    return (pd.Series(top_r, index=dates).dropna(), pd.Series(eq_r, index=dates).dropna())

def ann(r):
    return float(r.mean() * 365)

if __name__ == '__main__':
    d = load_panel()
    meta = uni_meta()
    elig = eligible_mask(d, meta)
    print(f"패널: {d.shape[1]}종목 × {len(d)}일  ({d.index[0].date()}~{d.index[-1].date()})")
    print(f"적격 종목수 (일별 평균): {elig.sum(axis=1).mean():.1f}  최근: {int(elig.iloc[-1].sum())}")
    print("\n" + "=" * 78)
    print("top-N 캐리 선택의 '증분 가치' (등가중 적격 대비)")
    print("=" * 78)
    for N in (5, 10, 20):
        top, eqw = topn_vs_equal(d, elig, N=N)
        idx = top.index.intersection(eqw.index)
        top, eqw = top.loc[idx], eqw.loc[idx]
        exc = top - eqw
        print(f"\n N={N:2d}  top-N 연율 {ann(top)*100:+7.2f}%  |  등가중 {ann(eqw)*100:+7.2f}%  "
              f"|  증분 {ann(exc)*100:+6.2f}%")
        yr = exc.groupby(exc.index.year).apply(lambda x: x.mean() * 365 * 100)
        print("      증분 연도별: " + "  ".join(f"{y}:{v:+.2f}%" for y, v in yr.items()))
