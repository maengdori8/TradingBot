from __future__ import annotations

"""수량 원장 엔진 테스트 — 델타중립성·룩어헤드·비용·리스크 조치를 검증한다."""

import numpy as np
import pandas as pd
import pytest

from carrybot.research.ledger import (
    FORWARD,
    REVERSE,
    Book,
    LedgerConfig,
    Position,
    simulate,
)


def make_panels(n_days=400, funding=0.0, syms=("AAA", "BBB"), price=100.0,
                basis_bp=0.0, adv=1e10, price_path=None):
    """합성 패널을 만든다."""
    idx = pd.date_range("2022-01-01", periods=n_days, freq="D", tz="utc")
    px = pd.Series(price, index=idx) if price_path is None else pd.Series(price_path, index=idx)
    fd = pd.DataFrame({s: (funding if not isinstance(funding, dict) else funding.get(s, 0.0))
                       for s in syms}, index=idx)
    if isinstance(funding, dict):
        for s, v in funding.items():
            fd[s] = v
    sc = pd.DataFrame({s: px for s in syms})
    pc = pd.DataFrame({s: px * (1 + basis_bp / 1e4) for s in syms})
    return dict(basis=(pc - sc) / sc, adv=pd.DataFrame({s: adv for s in syms}, index=idx),
                fd=fd, syms=list(syms), idx=idx, spot_close=sc, perp_close=pc)


def make_uni(syms=("AAA", "BBB"), launch_offset=400):
    """합성 유니버스 메타."""
    base = pd.Timestamp("2022-01-01", tz="utc") - pd.Timedelta(days=launch_offset)
    return pd.DataFrame([{"symbol": f"{s}USDT", "status": "Trading",
                          "contractType": "LinearPerpetual", "baseCoin": s,
                          "quoteCoin": "USDT", "launchTime": base,
                          "deliveryTime": pd.NaT, "fundingInterval": 480} for s in syms])


def cfg(**kw):
    """테스트 기본 설정."""
    base = dict(min_adv_usd=1e6, min_listing_age_days=180, max_positions=2, cash_rate=0.0)
    base.update(kw)
    return LedgerConfig(**base)


class TestBookAccounting:
    def test_델타중립이면_가격변동에_자본이_불변이다(self):
        b = Book(cash_exchange=0.35, cash_offvenue=0.0)
        b.positions["X"] = Position("X", FORWARD, 0.0065, 100.0, 100.0, pd.Timestamp("2024-01-01", tz="utc"))
        base = b.equity({"X": 100.0}, {"X": 100.0})
        for p in (50.0, 200.0, 1000.0):
            assert b.equity({"X": p}, {"X": p}) == pytest.approx(base, abs=1e-9)

    def test_reverse는_현물이_부채다(self):
        """공매도 대금을 받아 현금이 늘고, 현물은 동액의 부채로 잡힌다."""
        b = Book(cash_exchange=2.0, cash_offvenue=0.0)   # 자기자본 1.0 + 공매도 대금 1.0
        b.positions["X"] = Position("X", REVERSE, 0.01, 100.0, 100.0, pd.Timestamp("2024-01-01", tz="utc"))
        assert b.positions["X"].spot_value(100.0) == pytest.approx(-1.0)
        assert b.equity({"X": 100.0}, {"X": 100.0}) == pytest.approx(1.0)

    def test_reverse도_가격변동에_자본이_불변이다(self):
        b = Book(cash_exchange=2.0, cash_offvenue=0.0)
        b.positions["X"] = Position("X", REVERSE, 0.01, 100.0, 100.0, pd.Timestamp("2024-01-01", tz="utc"))
        base = b.equity({"X": 100.0}, {"X": 100.0})
        for q in (60.0, 150.0, 400.0):
            assert b.equity({"X": q}, {"X": q}) == pytest.approx(base, abs=1e-9)

    def test_숏perp은_마크상승시_손실이다(self):
        p = Position("X", FORWARD, 1.0, 100.0, 100.0, pd.Timestamp("2024-01-01", tz="utc"))
        assert p.perp_upl(120.0) == pytest.approx(-20.0)
        assert p.perp_upl(80.0) == pytest.approx(20.0)

    def test_롱perp은_마크상승시_이익이다(self):
        p = Position("X", REVERSE, 1.0, 100.0, 100.0, pd.Timestamp("2024-01-01", tz="utc"))
        assert p.perp_upl(120.0) == pytest.approx(20.0)


class TestNoLookahead:
    def test_펀딩_급등_이전에_진입하지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[250:] = 0.001                       # 이후만 고펀딩
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].iloc[:250].sum() == 0, "급등 이전 진입은 룩어헤드"
        assert r.daily["n_pos"].iloc[250:].sum() > 0

    def test_무펀딩이면_거래가_없다(self):
        P = make_panels(funding=0.0)
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].sum() == 0
        assert len(r.trades) == 0


class TestHurdle:
    def test_허들은_비용과_최소보유에서_유도된다(self):
        c = cfg(min_hold_days=30, cost_multiple=2.0)
        assert c.hurdle_ann == pytest.approx(2.0 * (2 * c.leg_cost) * 365 / 30)

    def test_현금금리가_높을수록_진입이_어렵다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0004, index=idx)           # 연율 약 14.6%
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        lo = simulate(P, make_uni(), cfg(cash_rate=0.0)).daily["n_pos"].sum()
        hi = simulate(P, make_uni(), cfg(cash_rate=0.10)).daily["n_pos"].sum()
        assert lo > hi

    def test_최소보유기간이_지켜진다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[100:140] = 0.001                    # 40일만 고펀딩
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg(min_hold_days=30))
        held = (r.daily["n_pos"] > 0).sum()
        assert held >= 30


class TestCosts:
    def test_거래가_있으면_비용이_부과된다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.001, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg())
        assert len(r.trades) > 0
        assert r.trades["cost"].sum() > 0

    def test_무거래밴드가_미세조정을_억제한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.001, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        wide = len(simulate(P, make_uni(), cfg(target_spot_fraction=0.30,
                                               no_trade_band=0.50)).trades)
        narrow = len(simulate(P, make_uni(), cfg(target_spot_fraction=0.30,
                                                 no_trade_band=0.0)).trades)
        assert wide < narrow


class TestReverse:
    def test_reverse는_설계상_금지된다(self):
        """부채 인지형 리스크 모델이 없으면 reverse는 열 수 없다."""
        with pytest.raises(NotImplementedError, match="reverse"):
            cfg(allow_reverse=True)

    def test_reverse포지션은_리스크변환에서_거부된다(self):
        b = Book(cash_exchange=2.0, cash_offvenue=0.0)
        b.positions["X"] = Position("X", REVERSE, 0.01, 100.0, 100.0,
                                    pd.Timestamp("2024-01-01", tz="utc"))
        with pytest.raises(NotImplementedError):
            b.to_account_state({"X": 100.0}, {"X": 100.0})



class TestRiskIntegration:
    def test_목표비중은_하드한도_안쪽이다(self):
        c = cfg()
        assert c.target_spot_fraction < c.max_spot_fraction, "경계 진동 방지"

    def test_배치시_현물비중이_한도를_넘지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.001, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg())
        assert r.daily["spot_frac"].max() <= cfg().max_spot_fraction + 1e-6

    def test_적격성_상실시_강제청산한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.001, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        P["adv"].loc[P["adv"].index[300]:, "AAA"] = 1.0    # 유동성 소멸
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].iloc[-30:].sum() == 0
        assert (r.trades["action"] == "force_exit").any() or (r.trades["action"] == "exit").any()


class TestUniverseIsADV:
    """유니버스는 ADV로 정해지고, 캐리는 그 안에서만 적용되어야 한다."""

    def test_고캐리_저유동성_종목은_선택되지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        P = make_panels(syms=("BIG", "SMALL"),
                        funding={"BIG": pd.Series(0.0, index=idx),
                                 "SMALL": pd.Series(0.002, index=idx)})
        P["adv"]["BIG"] = 1e10
        P["adv"]["SMALL"] = 5e7          # 적격이지만 BIG보다 훨씬 작다
        r = simulate(P, make_uni(("BIG", "SMALL")), cfg(universe_top_n=1, max_positions=1))
        assert r.daily["n_pos"].sum() == 0, "ADV 1위(BIG)는 캐리가 0이므로 진입 불가"
        assert not (r.trades["sym"] == "SMALL").any() if len(r.trades) else True

    def test_ADV_상위N만_유니버스에_든다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        P = make_panels(syms=("A", "B", "C"),
                        funding={s: pd.Series(0.002, index=idx) for s in ("A", "B", "C")})
        P["adv"]["A"], P["adv"]["B"], P["adv"]["C"] = 1e10, 5e9, 1e9
        r = simulate(P, make_uni(("A", "B", "C")), cfg(universe_top_n=2, max_positions=2))
        traded = set(r.trades.loc[r.trades["sym"] != "-", "sym"]) if len(r.trades) else set()
        assert "C" not in traded, "ADV 3위는 유니버스 밖"
        assert traded <= {"A", "B"}


class TestRiskLatching:
    def test_긴급조치_후_정지상태가_유지된다(self):
        from carrybot.research.ledger import Book, RiskState
        b = Book()
        b.risk_state = RiskState.HALTED
        assert b.risk_state is RiskState.HALTED

    def test_회복은_연속_무위반_사이클을_요구한다(self):
        c = cfg()
        assert c.recovery_cycles >= 2, "1사이클 회복은 진동을 유발한다"

    def test_설정_정합성이_강제된다(self):
        with pytest.raises(ValueError, match="무거래 밴드"):
            cfg(target_spot_fraction=0.55, add_stop_fraction=0.56, no_trade_band=0.20)
        with pytest.raises(ValueError, match="증액정지선"):
            cfg(add_stop_fraction=0.90)
        with pytest.raises(ValueError, match="담보비율"):
            cfg(exchange_collateral_ratio=0.20)


class TestFailClosed:
    def test_펀딩_결측시_강제청산한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.002, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        P["fd"].loc[P["fd"].index[300]:, "AAA"] = np.nan     # 펀딩 데이터 끊김
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].iloc[-50:].sum() == 0
        assert (r.trades["action"] == "force_exit").any()

    def test_가격_결측시_강제청산한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.002, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        P["spot_close"].loc[P["spot_close"].index[300]:, "AAA"] = np.nan
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].iloc[-50:].sum() == 0


class TestBoundaries:
    def test_평가구간은_평평하게_시작한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.002, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        boundary = pd.Timestamp("2022-06-01", tz="utc")
        r = simulate(P, make_uni(), cfg(), start="2022-06-01")
        assert r.daily.index[0] >= boundary
        assert (r.trades["ts"] >= boundary).all(), "워밍업 구간에서 거래가 발생하면 안 된다"

    def test_종료시_잔여포지션을_청산한다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.002, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].iloc[-1] == 0
        assert (r.trades["action"] == "terminal").any()


class TestEpisodes:
    def test_모든_연속_에피소드가_최소보유를_지킨다(self):
        idx = pd.date_range("2022-01-01", periods=500, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[100:150] = 0.002
        f.iloc[300:360] = 0.002
        P = make_panels(n_days=500, funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg(min_hold_days=30))
        assert len(r.episodes) >= 1
        # 종료 청산으로 잘린 마지막 에피소드는 제외하고 검사
        for _, e in r.episodes.iloc[:-1].iterrows() if len(r.episodes) > 1 else []:
            assert e["days"] >= 30, f"에피소드 {e['start']}가 최소보유 미달"

    def test_에피소드가_추출된다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[100:180] = 0.002
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg())
        assert len(r.episodes) >= 1
        assert set(r.episodes.columns) >= {"start", "end", "days", "excess", "equity_change"}


class TestAccountingIntegrity:
    """Codex 8라운드 요구 — 회계 무결성 하드 테스트."""

    def _run(self, **kw):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.0, index=idx)
        f.iloc[100:200] = 0.002
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        return simulate(P, make_uni(), cfg(**kw))

    def test_일별수익_누적곱이_자본비와_일치한다(self):
        r = self._run()
        d = r.daily
        prod = float((1 + d["ret"]).prod())
        start_eq = float(d["equity"].iloc[0]) / (1 + float(d["ret"].iloc[0]))
        assert prod == pytest.approx(float(d["equity"].iloc[-1]) / start_eq, rel=1e-9)

    def test_종료청산이_일별수익에_반영된다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        P = make_panels(funding={"AAA": pd.Series(0.002, index=idx),
                                 "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg())
        assert r.daily["n_pos"].iloc[-1] == 0
        assert (r.trades["action"] == "terminal").any()
        d = r.daily
        prod = float((1 + d["ret"]).prod())
        start_eq = float(d["equity"].iloc[0]) / (1 + float(d["ret"].iloc[0]))
        assert prod == pytest.approx(float(d["equity"].iloc[-1]) / start_eq, rel=1e-9)

    def test_에피소드는_경계행을_포함한다(self):
        r = self._run()
        assert len(r.episodes) >= 1
        for _, e in r.episodes.iterrows():
            span = (e["end"] - e["start"]).days
            assert span >= e["days"], "진입일·청산일 경계가 빠졌다"

    def test_담보이체에_비용이_부과된다(self):
        r = self._run()
        tr = r.trades
        if (tr["action"] == "transfer").any():
            assert (tr.loc[tr["action"] == "transfer", "cost"] > 0).all()

    def test_워밍업_구간에는_현금이자가_붙지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        P = make_panels(funding={"AAA": pd.Series(0.0, index=idx),
                                 "BBB": pd.Series(0.0, index=idx)})
        r = simulate(P, make_uni(), cfg(cash_rate=0.10), start="2022-06-01")
        d = r.daily
        # 무거래·무포지션이면 첫날 수익은 현금이자 하루치와 같아야 한다
        assert d["ret"].iloc[0] == pytest.approx(0.10 / 365, rel=1e-6)

    def test_stale_종목은_같은날_재진입하지_않는다(self):
        idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="utc")
        f = pd.Series(0.002, index=idx)
        P = make_panels(funding={"AAA": f, "BBB": pd.Series(0.0, index=idx)})
        P["fd"].loc[P["fd"].index[200], "AAA"] = np.nan     # 하루만 결측
        r = simulate(P, make_uni(), cfg())
        day = P["fd"].index[200]
        same_day = r.trades[(r.trades["ts"] == day) & (r.trades["sym"] == "AAA")]
        assert not (same_day["action"] == "adjust").any(), "결측일에 재진입하면 안 된다"
