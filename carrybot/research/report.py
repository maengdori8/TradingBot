from __future__ import annotations

"""보고 표 전량을 동결 입력에서 1회 명령으로 재생성한다 (재현성 요구 대응).

사용: PYTHONPATH=. python -m carrybot.research.report
"""

import logging

import numpy as np
import pandas as pd

from carrybot.research.carry import build_panels
from carrybot.research.evaluate import bootstrap_cagr_lb
from carrybot.research.ledger import LedgerConfig, simulate

logger = logging.getLogger(__name__)
F = "lab/frozen"


def load(survivorship_free: bool = False) -> tuple[dict, pd.DataFrame]:
    """동결 데이터를 읽어 패널과 유니버스를 만든다.

    Args:
        survivorship_free: True면 폐지 종목을 포함한다. 폐지 종목은 가격 이력이
            없으므로 상수 가격(베이시스 0)으로 넣어 손익이 펀딩만 반영되게 한다.

    Returns:
        (패널, 유니버스 메타).
    """
    perp = pd.read_parquet(f"{F}/perp_1d.parquet")
    spot = pd.read_parquet(f"{F}/spot_1d.parquet")
    if not survivorship_free:
        return build_panels(pd.read_parquet(f"{F}/funding.parquet"), perp, spot), \
            pd.read_parquet(f"{F}/universe.parquet")

    fund = pd.read_parquet(f"{F}/funding_survfree.parquet")
    uni = pd.read_parquet(f"{F}/universe_survfree.parquet")
    adv_const = pd.read_parquet(f"{F}/delisted_adv_const.parquet")["adv"]
    P = build_panels(fund, perp, spot)

    idx = P["idx"]
    for b, a in adv_const.items():
        if b not in fund.columns:
            continue
        life = fund[b].dropna()
        alive = (idx >= life.index[0]) & (idx <= life.index[-1])
        px = pd.Series(np.where(alive, 1.0, np.nan), index=idx)
        P["spot_close"][b] = px
        P["perp_close"][b] = px                    # 베이시스 0
        P["basis"][b] = pd.Series(np.where(alive, 0.0, np.nan), index=idx)
        P["adv"][b] = pd.Series(np.where(alive, a, np.nan), index=idx)
        P["fd"][b] = fund[b].resample("D").sum(min_count=1).reindex(idx)
    P["syms"] = sorted(set(P["spot_close"].columns) & set(P["fd"].columns) & set(P["adv"].columns))
    for k in ("spot_close", "perp_close", "basis", "adv", "fd"):
        P[k] = P[k][P["syms"]]
    return P, uni


def summarize(r, cash_rate: float) -> dict:
    """단일 실행 결과를 지표로 요약한다."""
    d = r.daily
    if len(d) < 30:
        return {}
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    cagr = float(d["equity"].iloc[-1]) ** (1 / yrs) - 1
    eq = d["equity"]
    exc = r.excess
    return dict(cagr=cagr, excess=cagr - cash_rate,
                mdd=float((1 - eq / eq.cummax()).max()),
                exposure=float(d["on_exchange"].mean()),
                max_exposure=float(d["on_exchange"].max()),
                episodes=len(r.episodes), trades=len(r.trades),
                cost=float(r.trades["cost"].sum()) if len(r.trades) else 0.0,
                reduce_days=int((d["action"] == "reduce").sum()),
                lb=bootstrap_cagr_lb(exc.to_numpy(), n_boot=3000))


def episode_bootstrap_lb(r, n_boot: int = 5000, alpha: float = 0.05, seed: int = 11) -> float:
    """에피소드 단위 부트스트랩 — 활성화 구간이 진짜 독립 관측 단위다."""
    ep = r.episodes
    if len(ep) < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = ep["equity_change"].to_numpy()
    out = np.array([np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)])
    return float(np.quantile(out, alpha))


def run_grid(cash_rates=(0.0, 0.02, 0.04), tops=(2, 4), survfree=(False, True)) -> pd.DataFrame:
    """유니버스 크기·현금금리·생존편향 교정 조합을 전량 실행한다."""
    rows = []
    for sf in survfree:
        P, uni = load(sf)
        for n in tops:
            for cr in cash_rates:
                cfg = LedgerConfig(universe_top_n=n, max_positions=n, cash_rate=cr,
                                   target_spot_fraction=0.50, add_stop_fraction=0.60,
                                   no_trade_band=0.10, exchange_collateral_ratio=0.70)
                r = simulate(P, uni, cfg)
                s = summarize(r, cr)
                if not s:
                    continue
                s.update(survfree=sf, top_n=n, cash_rate=cr,
                         ep_lb=episode_bootstrap_lb(r))
                rows.append(s)
    return pd.DataFrame(rows)


def main() -> None:
    """전체 보고 표를 출력한다."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = run_grid()
    print("\n" + "=" * 118)
    print("델타중립 캐리 — 전체 결과표 (동결 데이터, ADV 상위N 유니버스, 목표 50%/담보 70%)")
    print("=" * 118)
    print(f"{'생존교정':>8s} {'N':>3s} {'현금':>6s} {'CAGR':>8s} {'초과':>9s} {'MDD':>7s} "
          f"{'노출평균':>9s} {'노출최대':>9s} {'에피':>5s} {'거래':>5s} {'비용':>7s} "
          f"{'일별하한':>9s} {'에피하한':>9s}")
    for _, r in df.iterrows():
        print(f"{'O' if r.survfree else 'X':>8s} {int(r.top_n):3d} {r.cash_rate*100:5.0f}% "
              f"{r.cagr*100:+7.2f}% {r.excess*100:+8.2f}%p {r.mdd*100:6.3f}% "
              f"{r.exposure*100:8.1f}% {r.max_exposure*100:8.1f}% {int(r.episodes):5d} "
              f"{int(r.trades):5d} {r.cost*100:6.2f}% {r.lb*100:+8.3f}%p {r.ep_lb*100:+8.3f}%")
    df.to_csv("lab/data/report_grid.csv", index=False)
    print("\n저장: lab/data/report_grid.csv")


if __name__ == "__main__":
    main()
