from __future__ import annotations

"""캐리 전략 평가 — 3계정 회계와 '현금 대비 초과수익' 기준 유의성 검정.

회계 (Codex 지적 반영):
  자기자본 E = 거래소 현물(S_ex) + 거래소 스테이블 담보(C_ex) + 장외 현금(Cash_off)
  C_ex = stable_ratio x S_ex  (숏을 지키는 담보이므로 무위험 현금수익 불가)
  거래소 노출 = S_ex + C_ex   (파산 시 전액 손실 가능)
  장외 현금만 cash_rate를 번다.
판정 기준은 '현금 대비 초과 CAGR의 부트스트랩 단측 하한 > 0'이다.
"""

import logging

import numpy as np
import pandas as pd

from carrybot.research.carry import CarryResult

logger = logging.getLogger(__name__)


def portfolio_returns(res: CarryResult, spot_budget: float = 0.65,
                      stable_ratio: float = 0.50, cash_rate: float = 0.04
                      ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """일별 포트폴리오 수익·현금수익·거래소 노출을 3계정 회계로 계산한다.

    Args:
        res: 백테스트 결과.
        spot_budget: 자기자본 대비 최대 현물예산.
        stable_ratio: 현물명목 대비 필수 스테이블 담보 비율.
        cash_rate: 장외 현금 연수익률.

    Returns:
        (포트폴리오 일별수익, 현금 일별수익, 거래소 노출 비율).
    """
    d = res.daily
    on_exchange = (d["deployed"] * spot_budget * (1.0 + stable_ratio)).clip(upper=1.0)
    off_venue = (1.0 - on_exchange).clip(lower=0.0)
    strat = d["net"] * spot_budget                      # 캐리 손익 (자기자본 대비)
    cash = off_venue * cash_rate / 365.0
    return strat + cash, pd.Series(cash_rate / 365.0, index=d.index), on_exchange


def _cagr(r: pd.Series) -> float:
    """복리 연환산 수익률."""
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return float((1 + r).prod()) ** (1 / yrs) - 1 if yrs > 0 else np.nan


def bootstrap_cagr_lb(r: np.ndarray, n_boot: int = 5000, block: int = 180,
                      alpha: float = 0.05, seed: int = 11) -> float:
    """정상 블록 부트스트랩 — 복리 CAGR의 단측 하한 (평균×365가 아님)."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, float)
    n = len(r)
    block = max(5, min(block, n // 4))
    log1p = np.log1p(r)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx, got = [], 0
        while got < n:
            st = rng.integers(0, n)
            L = rng.geometric(1 / block)
            idx.extend((st + np.arange(L)) % n)
            got += L
        out[b] = np.exp(log1p[np.array(idx[:n])].mean() * 365) - 1
    return float(np.quantile(out, alpha))


def report(res: CarryResult, spot_budget: float = 0.65, stable_ratio: float = 0.50,
           cash_rate: float = 0.04, label: str = "") -> dict:
    """전략 성과를 3계정 회계로 출력하고 초과수익 하한을 판정한다."""
    port, cash, on_ex = portfolio_returns(res, spot_budget, stable_ratio, cash_rate)
    excess = port - cash
    d = res.daily
    strat_only = d["net"] * spot_budget

    yrs = (port.index[-1] - port.index[0]).days / 365.25
    eq = (1 + port).cumprod()
    mdd = float((1 - eq / eq.cummax()).max())
    vol = float(port.std() * np.sqrt(365))
    lb_excess = bootstrap_cagr_lb(excess.to_numpy())
    lb_strat = bootstrap_cagr_lb(strat_only.to_numpy())

    print(f"\n{'='*78}\n{label or '캐리 전략'}  (현물예산 {spot_budget:.0%}, 담보비율 {stable_ratio:.0%}, 장외현금 {cash_rate:.1%})\n{'='*78}")
    print(f" 기간 {port.index[0].date()} ~ {port.index[-1].date()} ({yrs:.2f}y)")
    print(f" 포트폴리오 CAGR {_cagr(port)*100:+6.2f}%   현금 전액 대안 {cash_rate*100:+.2f}%   "
          f"초과 {(_cagr(port)-cash_rate)*100:+.2f}%p")
    print(f" 변동성 {vol*100:5.2f}%   MDD {mdd*100:5.2f}%   "
          f"Sharpe(초과) {excess.mean()*365/vol if vol > 0 else np.nan:6.2f}")
    print(f" 거래소 노출: 평균 {on_ex.mean()*100:5.1f}%  최대 {on_ex.max()*100:5.1f}%  "
          f"(자기자본 대비, 현물+담보)")
    print(f" 캐리 자체 기여 CAGR {_cagr(strat_only)*100:+.2f}%   "
          f"평균 보유종목 {d['n_pos'].mean():.2f}   연간회전 {d['turnover'].sum()/yrs*100:.0f}%")
    print(f"\n [판정] 초과수익 CAGR 95% 단측하한 {lb_excess*100:+.3f}%p  "
          f"→ {'통과' if lb_excess > 0 else '실패'}")
    print(f"        캐리 기여 CAGR 95% 단측하한 {lb_strat*100:+.3f}%")
    yr = strat_only.groupby(strat_only.index.year).sum() * 100
    ex = on_ex.groupby(on_ex.index.year).mean() * 100
    print(" 연도별 캐리기여(노출): " + "  ".join(f"{y}:{v:+.2f}%({ex[y]:.0f}%)" for y, v in yr.items()))
    return dict(cagr=_cagr(port), excess=_cagr(port) - cash_rate, mdd=mdd, vol=vol,
                lb_excess=lb_excess, lb_strat=lb_strat, exposure=float(on_ex.mean()),
                strat_cagr=_cagr(strat_only))
