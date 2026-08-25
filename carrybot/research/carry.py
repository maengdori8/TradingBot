from __future__ import annotations

"""델타중립 펀딩 캐리 포트폴리오 백테스트 엔진.

설계 원칙 (Codex 적대적 검토 5라운드 반영):
- 시점보존: 유니버스·신호·유동성 모두 t-1까지의 정보만 사용하고 t에 실행한다.
- 유니버스는 ADV(거래대금) 순위로 '먼저' 확정한다. 그 안에서만 캐리 허들을 적용한다.
  (이전 판본은 펀딩 순위로 뽑아놓고 ADV 상위라고 잘못 표기했다.)
- 펀딩 주기 무관: 심볼·시기별 8h/4h/1h 혼재 → 항상 일별 합산으로 정규화.
- 결측은 0이 아니다. 보유 중 데이터가 끊기면 fail-closed로 청산한다.
- 수량 원장(unit ledger)을 유지해 '비중 고정이 매일 공짜 리사이징'이 되는 것을 막는다.
"""

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CarryConfig:
    """캐리 전략 설정. 사전 등록되며 사후 조정 시 재검증이 필요하다."""

    universe_top_n: int = 2             # 유니버스 크기
    select_by: str = "adv"              # "adv" = ADV 상위N / "carry" = 적격 전체 중 캐리 상위N
    cash_rate: float = 0.04             # 기회비용 (현금 대안 연수익률)
    min_hold_days: int = 30             # 최소 보유일 (과회전 차단)
    cost_multiple: float = 2.0          # 기대캐리 ≥ 왕복비용 × 이 배수
    lookback_days: int = 30             # 펀딩 추정 룩백
    min_valid_ratio: float = 0.90       # 룩백 내 유효 관측 최소 비율
    min_listing_age_days: int = 180     # 상장 후 최소 경과일
    min_adv_usd: float = 2e7            # 30일 중앙 거래대금 하한
    max_weight: float = 0.5             # 종목당 현물예산 비중 상한
    adv_participation: float = 0.01     # 포지션 ≤ ADV의 1%
    spot_fee: float = 0.0010            # 현물 편도 수수료
    perp_fee: float = 0.0006            # perp 편도 수수료 (taker)
    slippage: float = 0.0002            # 편도 슬리피지
    delist_buffer_days: int = 30        # 상장폐지 회피 기간
    trim_quantile: float = 0.10         # 신호 절사평균 비율
    _clip_from: object = None           # 내부용: 워밍업 구간 절단 시점

    @property
    def leg_cost(self) -> float:
        """양다리 편도 총비용 (현물+perp+슬리피지)."""
        return self.spot_fee + self.perp_fee + 2 * self.slippage

    @property
    def entry_hurdle_ann(self) -> float:
        """진입 허들 — '현금 대비 초과캐리' 기준. 절대 펀딩이 아니다.

        최소보유기간 동안 벌 '초과' 캐리가 왕복비용의 cost_multiple 배를 넘어야 한다:
            (기대캐리 - 현금) x (min_hold/365) >= cost_multiple x (2 x leg_cost)
        반환값은 현금 대비 초과분 기준 임계이며, 절대 기준으로는 cash_rate를 더한다.
        """
        return self.cost_multiple * (2.0 * self.leg_cost) * 365.0 / self.min_hold_days

    @property
    def exit_threshold_ann(self) -> float:
        """청산 임계 — 현금 대비 초과캐리가 이만큼 밑돌면 청산.

        나가 있을 기간에 벌 현금수익이 왕복비용을 넘어야 청산이 정당하다.
        보수적으로 최소보유기간과 같은 기간을 가정한다.
        """
        return -(2.0 * self.leg_cost) * 365.0 / self.min_hold_days


@dataclass
class CarryResult:
    """백테스트 산출물."""

    daily: pd.DataFrame
    weights: pd.DataFrame
    config: CarryConfig
    diagnostics: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        """현물예산 대비 일별 순수익률."""
        return self.daily["net"]


def _daily_funding(fund: pd.DataFrame) -> pd.DataFrame:
    """펀딩 정산을 일별 합산으로 정규화한다 (주기 혼재 대응). 결측은 NaN 유지."""
    alive = fund.notna().resample("D").max().astype(bool)
    return fund.resample("D").sum(min_count=1).where(alive)


def _panels(perp: pd.DataFrame, spot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일별 베이시스 패널과 병목 ADV 패널을 만든다."""
    basis, adv = {}, {}
    syms = set(perp.index.get_level_values("sym")) & set(spot.index.get_level_values("sym"))
    for s in sorted(syms):
        p, q = perp.xs(s, level="sym"), spot.xs(s, level="sym")
        idx = p.index.intersection(q.index)
        if len(idx) == 0:
            continue
        pc, qc = p["close"].loc[idx], q["close"].loc[idx]
        basis[s] = ((pc - qc) / qc).resample("D").last()
        pv = (pc * p["volume"].loc[idx]).rolling(30, min_periods=20).median()
        sv = (qc * q["volume"].loc[idx]).rolling(30, min_periods=20).median()
        adv[s] = pd.concat([pv, sv], axis=1).min(axis=1)     # 양다리 중 얇은 쪽
    return pd.DataFrame(basis), pd.DataFrame(adv)


def _eligibility(index: pd.DatetimeIndex, syms: list[str], uni: pd.DataFrame,
                 adv_lag: pd.DataFrame, cfg: CarryConfig) -> pd.DataFrame:
    """시점보존 적격성 (상장연수·유동성·상장폐지 회피). 전일 ADV만 사용한다."""
    u = uni[(uni.contractType == "LinearPerpetual") & (uni.quoteCoin == "USDT")]
    launch = u.groupby("baseCoin")["launchTime"].min()
    delist = u[u.status == "Closed"].groupby("baseCoin")["deliveryTime"].max()

    elig = pd.DataFrame(False, index=index, columns=syms)
    for s in syms:
        lt = launch.get(s, pd.NaT)
        if pd.isna(lt):
            continue
        ok = (index >= (lt + pd.Timedelta(days=cfg.min_listing_age_days)))
        ok &= (adv_lag[s].reindex(index) >= cfg.min_adv_usd).fillna(False).to_numpy()
        dt = delist.get(s, pd.NaT)
        if pd.notna(dt):
            ok &= index < (dt - pd.Timedelta(days=cfg.delist_buffer_days))
        elig[s] = ok
    return elig


def _signal(fd: pd.DataFrame, cfg: CarryConfig) -> pd.DataFrame:
    """연율 기대 캐리 = 룩백 일별펀딩의 절사평균 × 365.

    유효 관측이 min_valid_ratio 미만이면 NaN (결측을 0으로 취급하지 않는다).
    shift(1)로 결정 시점 이후 정보를 차단한다.
    """
    need = max(3, int(cfg.lookback_days * cfg.min_valid_ratio))

    def trimmed(x: np.ndarray) -> float:
        v = x[~np.isnan(x)]
        if len(v) < need:
            return np.nan
        k = int(len(v) * cfg.trim_quantile)
        vs = np.sort(v)
        if k > 0 and len(vs) - 2 * k >= 3:
            vs = vs[k:len(vs) - k]
        return float(np.mean(vs))

    return (fd.rolling(cfg.lookback_days, min_periods=need)
            .apply(trimmed, raw=True).shift(1) * 365)


def build_panels(fund: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame) -> dict:
    """가격·펀딩 패널을 1회 계산해 재사용한다 (WFO에서 반복 계산 방지)."""
    basis, adv = _panels(perp, spot)
    fd = _daily_funding(fund)
    syms = sorted(set(basis.columns) & set(fd.columns) & set(adv.columns))
    idx = basis.index.intersection(fd.index).intersection(adv.index).sort_values()
    sc, pc = {}, {}
    for s in syms:
        sc[s] = spot.xs(s, level="sym")["close"].resample("D").last()
        pc[s] = perp.xs(s, level="sym")["close"].resample("D").last()
    return dict(basis=basis.loc[idx, syms], adv=adv.loc[idx, syms],
                fd=fd.loc[idx, syms], syms=syms, idx=idx,
                spot_close=pd.DataFrame(sc).reindex(idx)[syms],
                perp_close=pd.DataFrame(pc).reindex(idx)[syms])


def backtest(fund: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame,
             uni: pd.DataFrame, cfg: CarryConfig | None = None,
             equity: float = 1.0, panels: dict | None = None,
             start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> CarryResult:
    """델타중립 캐리 포트폴리오를 시점보존 방식으로 백테스트한다.

    선택 순서 (중요):
      1. 전일 ADV 상위 universe_top_n → 사전적 유니버스 확정 (펀딩 미참조)
      2. 그 유니버스 안에서만 캐리 허들·최소보유·위험청산 적용

    Args:
        fund: 펀딩 정산 패널 (index=시각, columns=심볼).
        perp: perp 일봉 (MultiIndex sym/ts).
        spot: 현물 일봉 (MultiIndex sym/ts).
        uni: 시점보존 유니버스 메타.
        cfg: 전략 설정.
        equity: 현물예산 (1.0 = 정규화).

    Returns:
        CarryResult — 일별 수익·비중·진단.
    """
    cfg = cfg or CarryConfig()
    P = panels if panels is not None else build_panels(fund, perp, spot)
    basis, adv, fd, syms = P["basis"], P["adv"], P["fd"], P["syms"]
    if not syms:
        raise ValueError("공통 심볼이 없습니다")
    idx = P["idx"]
    if start is not None or end is not None:
        def _utc(v, default):
            """tz 유무에 관계없이 UTC Timestamp로 정규화한다."""
            if v is None:
                return default
            ts = pd.Timestamp(v)
            return ts.tz_localize("utc") if ts.tzinfo is None else ts.tz_convert("utc")

        lo, hi = _utc(start, idx[0]), _utc(end, idx[-1])
        # 신호 워밍업을 위해 룩백만큼 앞을 포함해 계산하고 이후 잘라낸다
        warm = lo - pd.Timedelta(days=cfg.lookback_days + 5)
        idx = idx[(idx >= warm) & (idx <= hi)]
        basis, adv, fd = basis.loc[idx], adv.loc[idx], fd.loc[idx]
        cfg = replace(cfg, _clip_from=lo)

    adv_lag = adv.shift(1)                       # 결정에는 전일 ADV만 사용
    sig = _signal(fd, cfg)
    elig = _eligibility(idx, syms, uni, adv_lag, cfg)
    dbasis = basis.diff()

    w_prev = pd.Series(0.0, index=syms)
    held_since: dict[str, pd.Timestamp] = {}
    rows, wrec = [], []

    for t in idx:
        # --- 1) 전일 비중의 당일 손익. 데이터 결측은 0이 아니라 fail-closed 대상 ---
        f_t, db_t = fd.loc[t], dbasis.loc[t]
        stale = [s for s in syms if w_prev[s] > 0 and (np.isnan(f_t[s]) or np.isnan(db_t[s]))]
        gross = float(sum(w_prev[s] * (f_t[s] - db_t[s])
                          for s in syms if w_prev[s] > 0 and s not in stale))

        # --- 2) 유니버스 확정 ---
        e_t = elig.loc[t]
        pool = adv_lag.loc[t][e_t].dropna()
        s_t = sig.loc[t]
        if cfg.select_by == "adv":
            universe = list(pool.nlargest(min(cfg.universe_top_n, len(pool))).index)
        else:                                    # "carry": 적격 전체 중 초과캐리 상위 N
            sc = (s_t[pool.index] - cfg.cash_rate).dropna()
            universe = list(sc.nlargest(min(cfg.universe_top_n, len(sc))).index)

        # --- 3) 현금 대비 초과캐리로 진입/청산 판정 ---
        # 보유 중인 종목은 유니버스에서 밀려나도 최소보유 잠금이 유지되도록 후보에 포함한다.
        held_now = [s for s in syms if w_prev[s] > 0]
        consider = list(dict.fromkeys(held_now + universe))
        cand = pd.Series(0.0, index=syms)
        keep: list[str] = []
        for s in consider:
            if not e_t[s]:                       # 적격성 상실은 잠금보다 우선
                continue
            if np.isnan(s_t[s]):
                continue
            if len(keep) >= cfg.universe_top_n:
                break
            excess = s_t[s] - cfg.cash_rate      # 현금 대비 초과캐리
            held = w_prev[s] > 0
            if held:
                if excess < cfg.exit_threshold_ann:      # 위험청산은 잠금보다 우선
                    continue
                locked = (t - held_since.get(s, t)).days < cfg.min_hold_days
                if locked or excess >= cfg.entry_hurdle_ann:
                    keep.append(s)
            elif excess >= cfg.entry_hurdle_ann:
                keep.append(s)

        keep = [s for s in keep if s not in stale]        # 데이터 끊긴 종목은 청산
        if keep:
            w = min(cfg.max_weight, 1.0 / len(keep))
            for s in keep:
                a = adv_lag.loc[t, s]
                cap = cfg.adv_participation * a / equity if (equity > 0 and np.isfinite(a)) else w
                cand[s] = min(w, cap)                     # 상한 초과분은 현금 (재정규화 금지)

        for s in syms:
            if cand[s] > 0 and w_prev[s] <= 0:
                held_since[s] = t
            elif cand[s] <= 0:
                held_since.pop(s, None)

        turn = float((cand - w_prev).abs().sum())
        cost = turn * cfg.leg_cost
        rows.append(dict(ts=t, gross=gross, cost=cost, net=gross - cost, turnover=turn,
                         n_pos=int((cand > 0).sum()), deployed=float(cand.sum()),
                         n_stale=len(stale), n_universe=len(universe)))
        wrec.append(cand.copy())
        w_prev = cand

    daily = pd.DataFrame(rows).set_index("ts")
    weights = pd.DataFrame(wrec, index=idx)
    if cfg._clip_from is not None:
        m = daily.index >= cfg._clip_from
        daily, weights = daily.loc[m], weights.loc[m]
        if len(daily) == 0:
            return CarryResult(daily=daily, weights=weights, config=cfg)
    if (last := float(weights.iloc[-1].sum())) > 0:       # 최종 청산비용
        i = daily.index[-1]
        daily.loc[i, "cost"] += last * cfg.leg_cost
        daily.loc[i, "net"] = daily.loc[i, "gross"] - daily.loc[i, "cost"]
    return CarryResult(daily=daily, weights=weights, config=cfg)
