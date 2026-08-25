from __future__ import annotations

"""캐리 백테스트 엔진 테스트 — 룩어헤드 차단을 최우선으로 검증한다.

이 저장소의 과거 연구는 '결정 시점에 아직 닫히지 않은 봉'을 사용해 전량
오염된 전력이 있다. 동일 실패를 구조적으로 막기 위해 합성 데이터로
미래 정보 유입을 직접 검사한다.
"""

import numpy as np
import pandas as pd
import pytest

from carrybot.research.carry import CarryConfig, backtest


def make_data(n_days=400, funding_by_day=None, syms=("AAA", "BBB"),
              price=100.0, basis=0.0, volume=1e7, launch_offset_days=400):
    """합성 시장 데이터를 만든다 (기본은 무펀딩·무베이시스)."""
    idx = pd.date_range("2022-01-01", periods=n_days, freq="D", tz="utc")
    fund = pd.DataFrame(0.0, index=idx, columns=list(syms))
    if funding_by_day:
        for s, series in funding_by_day.items():
            fund[s] = series
    perp, spot = {}, {}
    for s in syms:
        p = pd.DataFrame({"open": price, "high": price, "low": price,
                          "close": price * (1 + basis), "volume": volume / price}, index=idx)
        q = pd.DataFrame({"open": price, "high": price, "low": price,
                          "close": price, "volume": volume / price}, index=idx)
        perp[s], spot[s] = p, q
    uni = pd.DataFrame([{"symbol": f"{s}USDT", "status": "Trading",
                         "contractType": "LinearPerpetual", "baseCoin": s, "quoteCoin": "USDT",
                         "launchTime": idx[0] - pd.Timedelta(days=launch_offset_days),
                         "deliveryTime": pd.NaT, "fundingInterval": 480} for s in syms])
    return (fund, pd.concat(perp, names=["sym"]), pd.concat(spot, names=["sym"]), uni)


def cfg(**kw):
    """테스트 기본 설정."""
    base = dict(min_listing_age_days=180, min_adv_usd=1e6, universe_top_n=2,
                max_weight=0.5, adv_participation=1e9)
    base.update(kw)
    return CarryConfig(**base)


class TestNoLookahead:
    def test_미래_펀딩_급등에_사전_반응하지_않는다(self):
        """t0에 펀딩이 급등하면 t0 이전에는 절대 포지션이 없어야 한다."""
        spike_at = 250
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[spike_at:] = 0.01                      # 이후 구간만 고펀딩
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": f})
        res = backtest(fund, perp, spot, uni, cfg())
        w = res.weights["AAA"]
        assert w.iloc[:spike_at].sum() == 0.0, "급등 이전에 포지션이 생기면 룩어헤드다"
        assert w.iloc[spike_at:].sum() > 0.0, "급등 이후에는 진입해야 한다"

    def test_진입은_룩백_이후에만_발생한다(self):
        """신호는 룩백 창을 채워야 하므로 즉시 진입할 수 없다."""
        spike_at = 250
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx); f.iloc[spike_at:] = 0.01
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": f})
        c = cfg(lookback_days=30)
        res = backtest(fund, perp, spot, uni, c)
        first = res.weights["AAA"].to_numpy().nonzero()[0]
        assert len(first) > 0
        assert first[0] > spike_at, "급등 당일 이전 진입 불가"

    def test_당일_펀딩은_전일_비중에만_귀속된다(self):
        """신규 진입일의 펀딩을 당일 비중으로 받으면 룩어헤드다."""
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.01, index=idx)
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": f})
        res = backtest(fund, perp, spot, uni, cfg())
        first = res.weights["AAA"].to_numpy().nonzero()[0][0]
        assert res.daily["gross"].iloc[first] == pytest.approx(0.0), "진입 당일 그로스는 0이어야 한다"


class TestCosts:
    def test_회전이_없으면_비용도_없다(self):
        fund, perp, spot, uni = make_data()
        res = backtest(fund, perp, spot, uni, cfg())
        assert res.daily["cost"].sum() == pytest.approx(0.0)

    def test_진입허들은_비용에서_유도된다(self):
        c = cfg(min_hold_days=30, cost_multiple=2.0)
        expected = 2.0 * (2 * c.leg_cost) * 365 / 30
        assert c.entry_hurdle_ann == pytest.approx(expected)

    def test_최소보유기간이_길수록_허들이_낮아진다(self):
        assert cfg(min_hold_days=60).entry_hurdle_ann < cfg(min_hold_days=30).entry_hurdle_ann

    def test_최종_청산비용이_부과된다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": pd.Series(0.01, index=idx)})
        res = backtest(fund, perp, spot, uni, cfg())
        assert res.weights.iloc[-1].sum() > 0
        assert res.daily["cost"].iloc[-1] > 0, "보유 중 종료 시 청산비용을 반영해야 한다"


class TestEligibility:
    def test_상장연수_미달_종목은_보유하지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": pd.Series(0.01, index=idx)},
                                          launch_offset_days=0)
        res = backtest(fund, perp, spot, uni, cfg(min_listing_age_days=1095))
        assert res.weights.to_numpy().sum() == 0.0

    def test_유동성_미달_종목은_보유하지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": pd.Series(0.01, index=idx)},
                                          volume=1e3)
        res = backtest(fund, perp, spot, uni, cfg(min_adv_usd=1e7))
        assert res.weights.to_numpy().sum() == 0.0

    def test_유동성_상한이_비중을_제한한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": pd.Series(0.01, index=idx)},
                                          volume=1e7)
        res = backtest(fund, perp, spot, uni,
                       cfg(adv_participation=0.01), equity=1e8)
        assert 0 < res.weights["AAA"].max() < 0.5, "ADV 상한 초과분은 현금으로 남겨야 한다"


class TestHolding:
    def test_최소보유기간_동안_청산하지_않는다(self):
        """진입 직후 펀딩이 사라져도 최소보유기간은 지켜야 한다."""
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[100:160] = 0.01            # 60일만 고펀딩
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": f})
        res = backtest(fund, perp, spot, uni, cfg(min_hold_days=30))
        w = (res.weights["AAA"] > 0).to_numpy()
        on = w.nonzero()[0]
        assert len(on) >= 30, "최소보유기간만큼은 유지되어야 한다"

    def test_음수_펀딩_전환시_위험청산한다(self):
        idx = pd.date_range("2022-01-01", periods=500, freq="D", tz="utc")
        f = pd.Series(0.01, index=idx)
        f.iloc[300:] = -0.01
        fund, perp, spot, uni = make_data(n_days=500, funding_by_day={"AAA": f})
        res = backtest(fund, perp, spot, uni, cfg(min_hold_days=30))
        assert res.weights["AAA"].iloc[-1] == 0.0, "펀딩이 음수면 청산해야 한다"


class TestBasisAccounting:
    def test_베이시스_축소는_숏에게_이익이다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        fund, perp, spot, uni = make_data(funding_by_day={"AAA": pd.Series(0.01, index=idx)})
        # perp 가격을 서서히 낮춰 베이시스 축소를 만든다
        perp = perp.copy()
        aaa = perp.xs("AAA", level="sym").copy()
        aaa["close"] = np.linspace(101.0, 100.0, len(aaa))
        perp.loc[("AAA", slice(None)), "close"] = aaa["close"].to_numpy()
        res = backtest(fund, perp, spot, uni, cfg())
        held = res.daily[res.weights["AAA"] > 0]
        assert held["gross"].sum() > 0
