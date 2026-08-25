"""레버리지 사다리 — 명목 배수별 수익과 청산 충격을 함께 계산한다.

탐색적 연구다. 사전등록(현물 50%, 무차입)은 그대로 두고, "레버리지를 걸면
수익이 얼마가 되고 어떤 충격에 죽는가"를 정직하게 계량한다.
- 진입/청산 규칙·에피소드는 레버리지와 무관 (허들은 명목당 비용이라 불변)
- 명목 초과분은 USDT 차입 (오늘 4.74%/yr — 과거 PIT 없음 → 3/5/8% 민감도)
- 리스크 불변식은 이 연구에서 의도적으로 비활성 (대신 청산 충격을 직접 보고)
"""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

from carrybot.research.ledger import simulate, LedgerConfig
from carrybot.research.report import load
from carrybot.risk.invariants import Limits
from carrybot.risk.margin import AccountState, CarryLeg, StressScenario

INF = float('inf')
NOBIND = Limits(max_unit_mismatch=INF, max_spot_to_equity=INF, min_stable_to_spot=-INF,
                max_gross_to_equity=INF, max_stressed_mmr=INF, max_orphan_loss=INF,
                mmr_block_add=INF, mmr_rebalance=INF, mmr_reduce=INF, mmr_emergency=INF,
                max_upl_to_equity=INF)


def kill_shock(f: float, premium: float, haircut_cap: float, mmr_mult: float,
               base_mmr: float = 0.0033, coll_ratio: float = 0.95) -> float:
    """현물비중 f(자기자본 대비)에서 즉시 청산을 유발하는 최소 상승 충격(%)을 푼다.

    스테이블 = 1 - 1.0*f(현물 매수) 잔여이며, f > 1이면 차입 부채(음수).
    거래소 담보 = 잔여 전액이 거래소에 있다고 가정 (레버리지 시 보수적 아님에 유의).
    """
    st = AccountState(
        legs=[CarryLeg('X', f, f, 1.0, 1.0, 1.0, base_mmr=base_mmr, collateral_ratio=coll_ratio)],
        stable_collateral=1.0 - f)
    lo, hi = 0.0, 20.0
    def dead(x: float) -> bool:
        sc = StressScenario('k', x, premium, haircut_cap, mmr_mult, 0.02, 0.01, 0.90)
        return st.maintenance_ratio(sc) >= 1.0
    if not dead(hi):
        return float('inf')
    for _ in range(60):
        mid = (lo + hi) / 2
        if dead(mid): hi = mid
        else: lo = mid
    return hi


def run_ladder(borrow_rate: float = 0.05, cash_rate: float = 0.04):
    P, uni = load(False)          # 완전 관측 시장 (생존편향 합성 제외)
    rows = []
    for m in (1.0, 1.3, 1.6, 1.7, 1.8, 2.0, 3.0, 4.0):
        f = 0.50 * m
        cfg = LedgerConfig(notional_multiplier=m, usdt_borrow_rate=borrow_rate,
                           cash_rate=cash_rate, record_stress=True)
        r = simulate(P, uni, cfg, limits=NOBIND)
        d = r.daily
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        cagr = float((1 + d.ret).prod()) ** (1 / yrs) - 1
        eqc = (1 + d.ret).cumprod()
        mdd = float((1 - eqc / eqc.cummax()).max())
        ep = r.episodes
        ep_ann = [(1 + e) ** (365 / max(dd, 1)) - 1 for e, dd in zip(ep.equity_change, ep.days)] if len(ep) else []
        rows.append(dict(
            m=m, f=f, cagr=cagr, excess=cagr - cash_rate, mdd=mdd,
            episodes=len(ep), ep_ann_med=float(np.median(ep_ann)) if ep_ann else np.nan,
            ep_ann_max=float(np.max(ep_ann)) if ep_ann else np.nan,
            worst_ep=float(ep.equity_change.min()) if len(ep) else np.nan,
            borrowed_max=float(d.borrowed.max()),
            cost=float(r.trades.cost.sum()),
            kill_real=kill_shock(f, premium=0.05, haircut_cap=0.90, mmr_mult=1.0),
            kill_stress=kill_shock(f, premium=0.15, haircut_cap=0.70, mmr_mult=2.0),
            headroom_min=float(d.loc[d.n_pos > 0, 'stress_headroom'].min()) if (d.n_pos > 0).any() else np.nan,
        ))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for br in (0.05,):
        df = run_ladder(borrow_rate=br)
        print(f"\n{'='*126}")
        print(f"레버리지 사다리 (차입 {br*100:.0f}%/yr, 현금 4%, 동결 데이터 2021-07~2026-08, 완전 관측 시장)")
        print('='*126)
        print(f"{'명목/자본':>9s} {'CAGR':>8s} {'초과':>9s} {'MDD':>7s} {'에피':>4s} "
              f"{'에피중앙(연환산)':>15s} {'에피최대':>9s} {'최악에피':>9s} {'최대차입':>9s} "
              f"{'청산충격(실측)':>13s} {'청산충격(스트레스)':>16s} {'최소헤드룸':>10s}")
        for _, r in df.iterrows():
            kr = f"+{r.kill_real*100:.0f}%" if np.isfinite(r.kill_real) else "생존"
            ks = f"+{r.kill_stress*100:.0f}%" if np.isfinite(r.kill_stress) else "생존"
            hm = f"+{r.headroom_min*100:.0f}%" if np.isfinite(r.headroom_min) else "무한"
            passed = "통과" if (np.isfinite(r.headroom_min) and r.headroom_min >= 1.0) or not np.isfinite(r.headroom_min) else "실패"
            print(f"{r.f:8.2f}x {r.cagr*100:+7.2f}% {r.excess*100:+8.2f}%p {r.mdd*100:6.3f}% "
                  f"{int(r.episodes):4d} {r.ep_ann_med*100:+14.1f}% {r.ep_ann_max*100:+8.1f}% "
                  f"{r.worst_ep*100:+8.2f}% {r.borrowed_max*100:8.1f}% {kr:>13s} {ks:>16s} {hm:>7s}({passed})")
        df.to_csv('lab/data/leverage_ladder.csv', index=False)

    # 차입금리 민감도 (통과 경계 근처 rung)
    print(f"\n차입금리 민감도:")
    for br in (0.03, 0.05, 0.08, 0.12):
        df = run_ladder(borrow_rate=br)
        for f_sel in (0.85, 1.50):
            sel = df[np.isclose(df.f, f_sel)]
            if len(sel):
                r = sel.iloc[0]
                print(f"  S/E {f_sel:.2f}x 차입 {br*100:4.0f}% → 초과 {r.excess*100:+6.2f}%p")
