from __future__ import annotations

"""Track B 터틀 엔진 테스트 — 체결 모델·래치·상태 직렬화."""

import numpy as np
import pytest

from carrybot.aggressive.turtle import (
    Bar,
    TurtleConfig,
    TurtlePosition,
    TurtleState,
    _mtm,
    step,
)


def bar(open_=100.0, high=100.0, low=100.0, close=100.0,
        ehi=110.0, elo=90.0, xhi=105.0, xlo=95.0, funding=0.0):
    return Bar(open_, high, low, close, ehi, elo, xhi, xlo, funding)


def cfg(**kw):
    base = dict(syms=("AAA",), risk_pct=0.02)
    base.update(kw)
    return TurtleConfig(**base)


def warmed_state(atr=2.0):
    st = TurtleState()
    st.atr["AAA"] = atr
    return st


class TestEntry:
    def test_돌파가_없으면_진입하지_않는다(self):
        st, fills = step(warmed_state(), {"AAA": bar()}, cfg(), "2024-01")
        assert not st.positions and not fills

    def test_상향돌파는_채널레벨과_시가_중_불리한_쪽에_체결된다(self):
        # 시가 108 < 채널 110, 고가 115 → 체결가는 110 (채널 스탑주문)
        st, fills = step(warmed_state(), {"AAA": bar(open_=108, high=115, low=108, close=114)},
                         cfg(), "2024-01")
        assert st.positions["AAA"].entry == pytest.approx(110.0)

    def test_갭상승_시가가_채널보다_높으면_시가에_체결된다(self):
        st, fills = step(warmed_state(), {"AAA": bar(open_=113, high=115, low=112, close=114)},
                         cfg(), "2024-01")
        assert st.positions["AAA"].entry == pytest.approx(113.0)

    def test_하향돌파는_숏이다(self):
        st, fills = step(warmed_state(), {"AAA": bar(open_=92, high=93, low=85, close=86)},
                         cfg(), "2024-01")
        assert st.positions["AAA"].direction == -1

    def test_같은날_스탑아웃은_비관적으로_손실_처리한다(self):
        # 진입 110, 스탑 110-2x2=106, 저가 104 → 같은날 스탑
        st, fills = step(warmed_state(), {"AAA": bar(open_=108, high=115, low=104, close=114)},
                         cfg(), "2024-01")
        assert "AAA" not in st.positions
        assert fills[0]["action"] == "same_day_stop"
        assert st.equity < 1.0

    def test_리스크는_설정값을_넘지_않는다(self):
        st, _ = step(warmed_state(), {"AAA": bar(open_=108, high=115, low=108, close=114)},
                     cfg(risk_pct=0.02), "2024-01")
        p = st.positions["AAA"]
        assert p.units * p.risk_d <= 0.02 * 1.0 + 1e-9


class TestExit:
    def held(self, direction=1, entry=100.0, stop=94.0):
        st = warmed_state()
        st.positions["AAA"] = TurtlePosition(direction, 0.005, entry, stop, 4.0)
        return st

    def test_갭하락_스탑은_시가로_악화된다(self):
        st = self.held(stop=94.0)
        st2, fills = step(st, {"AAA": bar(open_=90, high=91, low=88, close=89,
                                          ehi=200, elo=1, xlo=95)}, cfg(), "2024-01")
        assert fills[0]["action"] == "exit"
        assert fills[0]["price"] == pytest.approx(90.0)   # 스탑 94가 아닌 시가 90

    def test_롱은_양수펀딩을_지불한다(self):
        st = self.held()
        eq0 = st.equity
        st2, _ = step(st, {"AAA": bar(funding=0.001, ehi=200, elo=1)}, cfg(), "2024-01")
        assert st2.equity < eq0

    def test_숏은_양수펀딩을_수취한다(self):
        st = self.held(direction=-1, stop=106.0)
        eq0 = st.equity
        st2, _ = step(st, {"AAA": bar(funding=0.001, ehi=200, elo=1, xhi=101.0)}, cfg(), "2024-01")
        assert st2.equity > eq0, "숏은 양수 펀딩을 수취해야 한다"
        assert "AAA" in st2.positions

    def test_데이터_결측은_강제청산한다(self):
        st = self.held()
        st2, fills = step(st, {"AAA": bar(open_=np.nan)}, cfg(), "2024-01")
        assert "AAA" not in st2.positions
        assert fills[0]["action"] == "force_exit"


class TestRiskLatches:
    def test_월손실_킬은_래치된다(self):
        st = warmed_state()
        st.equity = 0.80
        st.month_key, st.month_start_eq = "2024-01", 1.0
        st2, fills = step(st, {"AAA": bar()}, cfg(), "2024-01")
        assert st2.killed
        st3, fills3 = step(st2, {"AAA": bar(open_=108, high=115, close=114)}, cfg(), "2024-01")
        assert not st3.positions, "킬 상태에서 신규 진입 금지"

    def test_새_달에는_월시작자본이_갱신된다(self):
        st = warmed_state()
        st.equity = 0.90
        st.month_key, st.month_start_eq = "2024-01", 1.0
        st2, _ = step(st, {"AAA": bar()}, cfg(), "2024-02")
        assert st2.month_start_eq == pytest.approx(0.90)
        assert not st2.killed


class TestStateSerde:
    def test_상태_왕복_직렬화(self):
        st = warmed_state()
        st.positions["AAA"] = TurtlePosition(1, 0.005, 100.0, 94.0, 4.0)
        st.equity = 1.2345
        st2 = TurtleState.from_dict(st.to_dict())
        assert st2.equity == st.equity
        assert st2.positions["AAA"].stop == 94.0
        assert st2.atr["AAA"] == 2.0

    def test_mtm은_미실현손익을_포함한다(self):
        st = warmed_state()
        st.positions["AAA"] = TurtlePosition(1, 0.01, 100.0, 94.0, 4.0)
        assert _mtm(st, {"AAA": bar(close=110.0)}) == pytest.approx(1.0 + 0.1)


class TestCarryPaperSerde:
    def test_상태_왕복_직렬화(self):
        from carrybot.live.carry_paper import CarryPaperState
        st = CarryPaperState(equity=1.234)
        st.positions["BTC"] = dict(weight=0.25, opened="2026-08-24", basis=0.001)
        st2 = CarryPaperState.from_dict(st.to_dict())
        assert st2.equity == st.equity
        assert st2.positions["BTC"]["weight"] == 0.25
