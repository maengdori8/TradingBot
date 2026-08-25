from __future__ import annotations

"""캐리 리스크 모듈 단위 테스트."""

import pytest

from carrybot.risk.invariants import (
    Action,
    Limits,
    check_invariants,
    max_spot_notional,
    worst_action,
)
from carrybot.risk.margin import (
    CRASH,
    SHORT_SQUEEZE,
    AccountState,
    CarryLeg,
    StressScenario,
    adl_orphan_loss,
)


def leg(units=0.65, perp_units=None, spot=100.0, mark=None, entry=None, **kw):
    """테스트용 캐리 다리를 만든다."""
    perp_units = units if perp_units is None else perp_units
    mark = spot if mark is None else mark
    entry = mark if entry is None else entry
    return CarryLeg("BTC", units, perp_units, spot, mark, entry, **kw)


def base_account(stable=35.0, **kw):
    """권고 운용점의 계좌 상태를 만든다."""
    return AccountState([leg(**kw)], stable)


class TestCarryLeg:
    def test_수량이_음수면_거부한다(self):
        with pytest.raises(ValueError, match="음수"):
            CarryLeg("BTC", -1.0, 1.0, 100.0, 100.0, 100.0)

    def test_가격이_0이하면_거부한다(self):
        with pytest.raises(ValueError, match="양수"):
            CarryLeg("BTC", 1.0, 1.0, 0.0, 100.0, 100.0)

    def test_숏_UPL은_마크상승시_손실이다(self):
        assert leg(spot=200.0, mark=200.0, entry=100.0).perp_upl == pytest.approx(-65.0)

    def test_숏_UPL은_마크하락시_이익이다(self):
        assert leg(spot=50.0, mark=50.0, entry=100.0).perp_upl == pytest.approx(32.5)

    def test_수량_불일치를_비율로_계산한다(self):
        assert leg(units=1.0, perp_units=0.99).unit_mismatch == pytest.approx(0.01)

    def test_수량이_같으면_불일치는_0이다(self):
        assert leg().unit_mismatch == 0.0


class TestStressScenario:
    def test_담보인정비율이_범위를_벗어나면_거부한다(self):
        with pytest.raises(ValueError, match="담보인정비율"):
            StressScenario("x", 1.0, 0.1, 1.5, 2.0, 0.02, 0.01, 0.9)

    def test_유지증거금_배수가_1미만이면_거부한다(self):
        with pytest.raises(ValueError, match="배수"):
            StressScenario("x", 1.0, 0.1, 0.7, 0.5, 0.02, 0.01, 0.9)


class TestAccountState:
    def test_자기자본은_UPL을_포함한다(self):
        """랠리 후 자본이 과대계상되지 않아야 한다 (구버전의 치명적 결함)."""
        st = AccountState([leg(spot=200.0, mark=200.0, entry=100.0)], 35.0)
        assert st.equity == pytest.approx(100.0)   # 165가 아니다
        assert st.perp_upl == pytest.approx(-65.0)

    def test_포지션이_없으면_유지증거금률은_0이다(self):
        assert AccountState([], 100.0).maintenance_ratio() == 0.0

    def test_권고_운용점은_스트레스를_견딘다(self):
        st = base_account()
        assert st.maintenance_ratio() < 0.01
        assert st.maintenance_ratio(SHORT_SQUEEZE) < 0.50
        assert st.maintenance_ratio(CRASH) < 0.50

    def test_랠리_방치시_스퀴즈_스트레스에서_파산한다(self):
        """진입 후 2배 오른 상태를 방치하면 추가 스퀴즈에 청산된다."""
        st = AccountState([leg(spot=200.0, mark=200.0, entry=100.0)], 35.0)
        assert st.maintenance_ratio(SHORT_SQUEEZE) == float("inf")

    def test_펀딩유출은_자산을_줄인다(self):
        """펀딩 유출을 유지증거금에 더하면 안 된다(자산 차감이 옳다)."""
        base = SHORT_SQUEEZE
        no_debit = StressScenario(base.name, base.spot_shock, base.mark_premium,
                                  base.collateral_ratio_cap, base.mmr_multiplier,
                                  base.unwind_cost, 0.0, base.stable_haircut)
        st = base_account()
        assert st.maintenance_ratio(base) > st.maintenance_ratio(no_debit)


class TestAdlOrphanLoss:
    def test_사건_직전_자본을_기준으로_손실을_계산한다(self):
        r = adl_orphan_loss(base_account())
        assert r["pre_equity"] == pytest.approx(100.0)
        assert 0.20 < r["loss_frac"] < 0.30

    def test_숏이_이익중이면_손실이_완화된다(self):
        """크래시 중 ADL은 숏 이익을 실현시키므로 순손실이 줄어야 한다."""
        crashed = AccountState([leg(spot=70.0, mark=70.0, entry=100.0)], 35.0)
        flat = base_account()
        assert crashed.perp_upl > 0
        assert adl_orphan_loss(crashed)["loss_frac"] < adl_orphan_loss(flat)["loss_frac"]

    def test_자본이_0이하면_무한대를_반환한다(self):
        st = AccountState([leg(units=1.0, spot=100.0, mark=300.0, entry=100.0)], 0.0)
        assert st.equity <= 0
        assert adl_orphan_loss(st)["loss_frac"] == float("inf")


class TestInvariants:
    def test_권고_운용점은_모든_불변식을_통과한다(self):
        assert check_invariants(base_account()) == []

    def test_과다_레버리지는_감축을_지시한다(self):
        st = AccountState([leg(units=0.90)], 10.0)
        assert worst_action(check_invariants(st)) == Action.REDUCE

    def test_수량_불일치는_리밸런스를_지시한다(self):
        st = AccountState([leg(units=0.65, perp_units=0.60)], 35.0)
        rules = {v.rule for v in check_invariants(st)}
        assert any(r.startswith("unit_mismatch") for r in rules)

    def test_랠리_방치는_긴급조치를_지시한다(self):
        st = AccountState([leg(spot=200.0, mark=200.0, entry=100.0)], 35.0)
        assert worst_action(check_invariants(st)) == Action.EMERGENCY

    def test_누적_미실현손실_한도가_리밸런스를_발동한다(self):
        st = AccountState([leg(spot=160.0, mark=160.0, entry=100.0)], 60.0)
        assert any(v.rule == "perp_upl_loss" for v in check_invariants(st))

    def test_히스테리시스_하위단계가_살아있다(self):
        """구버전은 최고 한도 초과 시에만 평가되어 block_add가 발동 불가였다."""
        lim = Limits(mmr_block_add=0.001, mmr_rebalance=0.15,
                     mmr_reduce=0.20, mmr_emergency=0.30)
        acts = [v.action for v in check_invariants(base_account(), lim) if v.rule == "live_mmr"]
        assert acts == [Action.BLOCK_ADD]

    def test_자본이_0이하면_즉시_긴급이다(self):
        st = AccountState([leg()], -50.0)
        assert worst_action(check_invariants(st)) == Action.EMERGENCY

    def test_worst_action은_가장_강한_조치를_고른다(self):
        assert worst_action([]) == Action.OK


class TestMaxSpotNotional:
    def test_모든_제약의_최소값을_취한다(self):
        assert max_spot_notional(100.0) == pytest.approx(65.0)

    def test_자본에_비례한다(self):
        assert max_spot_notional(1000.0) == pytest.approx(10 * max_spot_notional(100.0))

    def test_자본이_0이하면_0이다(self):
        assert max_spot_notional(0.0) == 0.0
        assert max_spot_notional(-5.0) == 0.0

    def test_산출된_한도는_실제로_불변식을_통과한다(self):
        eq = 100.0
        s = max_spot_notional(eq)
        st = AccountState([leg(units=s / 100.0)], eq - s)
        assert check_invariants(st) == []

    def test_ADL제약이_빡세지면_한도가_줄어든다(self):
        tight = Limits(max_orphan_loss=0.10)
        assert max_spot_notional(100.0, tight) < max_spot_notional(100.0)
