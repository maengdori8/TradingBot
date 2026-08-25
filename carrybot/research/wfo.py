from __future__ import annotations

"""워크포워드 진단 — 설정 선택이 가치를 더하는지 검사한다.

결론(2026-08 실행): 설정 선택은 가치를 파괴했다. 따라서 운용 설계는
'단일 설정 사전등록'이며, 이 모듈은 그 판단의 근거를 재생성하는 진단 도구다.
"""

import logging

import numpy as np
import pandas as pd

from carrybot.research.ledger import LedgerConfig, simulate

logger = logging.getLogger(__name__)


def default_grid(cash_rate: float = 0.04) -> list[LedgerConfig]:
    """사전 등록된 후보 설정 격자."""
    return [LedgerConfig(universe_top_n=n, max_positions=n, min_hold_days=mh,
                         cash_rate=cash_rate)
            for n in (1, 2, 4) for mh in (30, 60, 90)]


def _excess_sum(P, uni, cfg, a, b) -> float:
    """구간 초과수익 합."""
    r = simulate(P, uni, cfg, start=a, end=b)
    return float(r.excess.sum()) if len(r.daily) else 0.0


def run_wfo(P: dict, uni: pd.DataFrame, grid: list[LedgerConfig],
            min_train_months: int = 24, oos_months: int = 6,
            inner_months: int = 6, min_positive_frac: float = 2 / 3
            ) -> tuple[pd.Series, pd.DataFrame]:
    """설정을 train에서 고르고 미지의 OOS에서만 평가한다.

    Args:
        P: build_panels 산출물.
        uni: 유니버스 메타.
        grid: 후보 설정.
        min_train_months: 최소 train 길이.
        oos_months: OOS 길이.
        inner_months: train 내부 평가 구간 길이.
        min_positive_frac: 내부 구간 중 양수여야 하는 최소 비율.

    Returns:
        (OOS 초과수익 시계열, 폴드 로그).
    """
    idx = P["idx"]
    start = (pd.Timestamp(idx[0]).normalize().replace(day=1) + pd.DateOffset(months=1)).tz_convert("utc")
    end = idx[-1]
    parts, log = [], []
    train_end = start + pd.DateOffset(months=min_train_months)

    while train_end + pd.DateOffset(months=oos_months) <= end:
        oos_end = train_end + pd.DateOffset(months=oos_months)
        best, best_cfg = None, None
        for cfg in grid:
            inner, cur = [], start
            while cur + pd.DateOffset(months=inner_months) <= train_end:
                nxt = cur + pd.DateOffset(months=inner_months)
                inner.append(_excess_sum(P, uni, cfg, cur, nxt - pd.Timedelta(days=1)))
                cur = nxt
            if len(inner) < 2:
                continue
            med = float(np.median(inner))
            if med <= 0 or np.mean([x > 0 for x in inner]) < min_positive_frac:
                continue
            if best is None or med > best:
                best, best_cfg = med, cfg

        if best_cfg is None:
            days = pd.date_range(train_end, oos_end - pd.Timedelta(days=1), freq="D", tz="utc")
            parts.append(pd.Series(0.0, index=days))
            log.append(dict(train_end=train_end, oos_end=oos_end, cfg="CASH",
                            train_med=np.nan, oos_excess=0.0))
        else:
            r = simulate(P, uni, best_cfg, start=train_end, end=oos_end - pd.Timedelta(days=1))
            e = r.excess
            parts.append(e)
            log.append(dict(train_end=train_end, oos_end=oos_end,
                            cfg=f"N{best_cfg.universe_top_n}/H{best_cfg.min_hold_days}",
                            train_med=best, oos_excess=float(e.sum())))
        train_end = oos_end

    oos = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return oos, pd.DataFrame(log)


def main() -> None:
    """워크포워드 진단을 실행한다."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from carrybot.research.evaluate import bootstrap_cagr_lb
    from carrybot.research.report import load

    P, uni = load(survivorship_free=True)
    oos, log = run_wfo(P, uni, default_grid())
    print(f"\n{'train종료':>12s} {'OOS종료':>12s} {'선택':>10s} {'train중앙':>10s} {'OOS초과':>9s}")
    for _, r in log.iterrows():
        tm = f"{r.train_med*100:9.3f}%" if pd.notna(r.train_med) else "        -"
        print(f"{str(r.train_end.date()):>12s} {str(r.oos_end.date()):>12s} {r.cfg:>10s} "
              f"{tm} {r.oos_excess*100:+8.3f}%")
    if len(oos):
        yrs = (oos.index[-1] - oos.index[0]).days / 365.25
        cagr = float((1 + oos).prod()) ** (1 / yrs) - 1
        lb = bootstrap_cagr_lb(oos.to_numpy(), n_boot=5000)
        print(f"\nOOS 초과 CAGR {cagr*100:+.3f}%p   95% 하한 {lb*100:+.3f}%p   "
              f"→ {'통과' if lb > 0 else '실패'}   양수폴드 {(log.oos_excess>0).sum()}/{len(log)}")


if __name__ == "__main__":
    main()
