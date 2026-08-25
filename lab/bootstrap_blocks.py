"""Codex 지적 대응 — 블록길이 3/6/12/18개월 부트스트랩, 최악 하한 채택."""
from __future__ import annotations
import numpy as np, pandas as pd
from lab.carry_falsifier import backtest, block_bootstrap_lb

print("=" * 78)
print("블록 부트스트랩 민감도 — 펀딩 레짐 대비 블록길이 확대 (Codex 지적)")
print("=" * 78)
for mode, slip in [('taker', 2.0), ('maker', 0.0)]:
    df = backtest(exec_mode=mode, slip_bp=slip)
    r = df['ret'].to_numpy()
    print(f"\n── {mode} slip={slip}bp  (n={len(r)} 월블록)")
    worst = None
    for blk in (3, 6, 12, 18):
        lb = block_bootstrap_lb(r, n_boot=20000, block=blk)
        ann = (1 + lb) ** (365 / 30) - 1
        worst = lb if worst is None else min(worst, lb)
        print(f"   block={blk:2d}개월  월수익 95% 단측하한 {lb*100:+.4f}%  (연율 {ann*100:+.2f}%)")
    print(f"   >>> 최악하한 {worst*100:+.4f}%/월 → 판정: {'통과' if worst > 0 else '실패'}")

# 연도별 분해 — 레짐 의존성 노출
df = backtest(exec_mode='maker', slip_bp=0.0)
df['yr'] = df['t0'].dt.year
print("\n── 연도별 (maker, 월별강제왕복)")
g = df.groupby('yr').agg(n=('ret','size'), mean=('ret','mean'), fund=('fund_r','mean'),
                         fee=('fee_r','mean'), win=('ret', lambda x: (x>0).mean()))
for yr, row in g.iterrows():
    print(f"   {yr}  n={int(row['n']):2d}  월평균 {row['mean']*100:+.3f}%  "
          f"펀딩 {row['fund']*100:+.3f}%  비용 {row['fee']*100:.3f}%  승률 {row['win']*100:.0f}%")
