"""Track E 단타 팜 테스트 — 인과성·리스크·멱등·원자성 (전부 합성 데이터, 네트워크 0)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from carrybot.aggressive.scalp_farm import (
    ATR1H_N,
    BRK_ATR_MULT,
    COST_SIDE,
    H1,
    MR_ATR_MULT,
    BarE,
    CELLS,
    FarmPos,
    FarmState,
    _confirm_4h,
    _new_ind,
    farm_equities,
    mark_delisted,
    new_farm,
    step,
)
from carrybot.aggressive.scalp_farm_runner import (
    DATA_STALL_H,
    ERR_MARK,
    apply_stall_policy,
    check_gaps,
    contiguous_prefix,
    LEDGER,
    LEDGER_COLS,
    LEDGER_KEY,
    _abort,
    _atomic_write,
    _clear_abort,
    _save_all,
    append_csv_atomic,
    missing_hours,
    pick_basket_b,
    stalled_syms,
)

BASE = 1_767_225_600_000        # 2026-01-01 00:00 UTC (1h·4h·일 경계 정렬)


def bar(k: int, o: float, h: float, lo: float, c: float) -> BarE:
    """BASE + k시간 봉."""
    return BarE(BASE + k * H1, o, h, lo, c)


def farm(t0: int = 1) -> FarmState:
    """테스트 팜 (t0=1이면 전 봉 라이브)."""
    return new_farm(["XRP", "DOGE", "ADA"], t0)


def warm(state: FarmState, sym: str, atr: float = 2.0, hi: float = 100.0,
         lo: float = 90.0, n: int = 24, pc: float = 100.0,
         closes: list | None = None) -> dict:
    """심볼 지표 워밍업 주입."""
    ind = _new_ind()
    ind["atr1"] = atr
    ind["hl"] = [[hi, lo] for _ in range(n)]
    ind["cl"] = list(closes) if closes else []
    ind["pc"] = pc
    state.ind[sym] = ind
    return ind


class TestCausality:
    def test_BRK_스탑과_수량은_이전봉_ATR을_쓴다(self):
        # ATR[i-1]=2.0, 이번 봉 TR=15 → 같은 봉 ATR을 쓰면 스탑이 88이 아니게 된다
        st = farm()
        warm(st, "BTC", atr=2.0)
        fills = step(st, {"BTC": bar(0, 100, 115, 100, 114)})
        p = st.cells["E01"].positions["BTC"]
        assert p.stop == pytest.approx(100 - BRK_ATR_MULT * 2.0)     # = 88
        assert p.u == pytest.approx(0.02 * 10_000 / (BRK_ATR_MULT * 2.0))
        assert any(f["action"] == "enter" and f["cell"] == "E01" for f in fills)

    def test_TR은_전봉_종가_기준이다(self):
        # 봉내 범위 0인 갭 봉: prev close를 쓰지 않으면 TR=0으로 퇴화한다
        st = farm()
        step(st, {"BTC": bar(0, 100, 100, 100, 100)})
        step(st, {"BTC": bar(1, 120, 120, 120, 120)})
        assert st.ind["BTC"]["atr1"] == pytest.approx(20.0 / ATR1H_N)

    def test_MR_체결은_신호봉_종가가_아니라_다음_봉_시가다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0, closes=[99.0, 101.0] * 12)
        step(st, {"BTC": bar(0, 100, 103, 100, 103)})        # z≈+2.9 → 숏 신호
        cell = st.cells["E07"]
        assert "BTC" not in cell.positions, "신호봉 종가 체결 금지"
        assert cell.pending["BTC"]["d"] == -1
        step(st, {"BTC": bar(1, 105, 105, 104, 104.5)})
        p = cell.positions["BTC"]
        assert p.e == pytest.approx(105.0), "다음 봉 시가 체결"
        # 스탑 ATR도 체결봉 직전(신호봉까지 갱신된) 값: 1 + (3-1)/24
        a = 1.0 + (3.0 - 1.0) / ATR1H_N
        assert p.stop == pytest.approx(105.0 + MR_ATR_MULT * a)

    def test_4h봉은_1h봉_4개_전부_있어야_확정된다(self):
        st = farm()
        for k in range(4):
            step(st, {"BTC": bar(k, 100, 101, 99, 100)})
        assert st.ind["BTC"]["h4"]["n4"] == 1
        nan = float("nan")
        step(st, {"BTC": BarE(BASE + 4 * H1, nan, nan, nan, nan)})   # 결측
        for k in (5, 6, 7):
            step(st, {"BTC": bar(k, 100, 101, 99, 100)})
        assert st.ind["BTC"]["h4"]["n4"] == 1, "3/4 창은 폐기 (fail-closed)"

    def test_4h_창은_UTC_00_04_정렬이다(self):
        st = farm()
        for k in (2, 3):                                     # 창 중간 합류 → 미확정
            step(st, {"BTC": bar(k, 100, 101, 99, 100)})
        assert st.ind["BTC"]["h4"]["n4"] == 0
        for k in (4, 5, 6, 7):
            step(st, {"BTC": bar(k, 100, 101, 99, 100)})
        assert st.ind["BTC"]["h4"]["n4"] == 1

    def test_T0_이전에는_어떤_주문도_없다(self):
        st = farm(t0=BASE + 100 * H1)
        warm(st, "BTC")
        fills = step(st, {"BTC": bar(0, 100, 115, 100, 114)})   # 명백한 돌파
        assert fills == []
        for c in st.cells.values():
            assert not c.positions and not c.pending

    def test_T0_이후_첫_봉부터만_진입한다(self):
        st = farm(t0=BASE + 2 * H1)
        warm(st, "BTC")
        step(st, {"BTC": bar(1, 100, 115, 100, 114)})           # T0 이전 → 무시
        assert not st.cells["E01"].positions
        step(st, {"BTC": bar(2, 116, 120, 116, 118)})           # T0 이후 → 진입
        assert st.cells["E01"].positions["BTC"].e == pytest.approx(116.0)

    def test_같은_봉_재실행은_무변화다(self):
        st = farm()
        warm(st, "BTC")
        step(st, {"BTC": bar(0, 100, 115, 100, 114)})
        snap = json.dumps(st.to_dict(), sort_keys=True, default=float)
        fills = step(st, {"BTC": bar(0, 100, 115, 100, 114)})   # 재실행
        assert fills == []
        assert json.dumps(st.to_dict(), sort_keys=True, default=float) == snap


class TestRisk:
    def test_수량은_스탑거리_역산이다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 115, 100, 114)})
        p = st.cells["E01"].positions["BTC"]
        assert p.u * (p.e - p.stop) == pytest.approx(0.02 * 10_000)

    def test_그로스_캡_10x가_수량을_자른다(self):
        st = farm()
        warm(st, "BTC", atr=0.01)                # 스탑 초근접 → 리스크 수량 폭발
        step(st, {"BTC": bar(0, 100, 101, 100, 100.5)})
        p = st.cells["E01"].positions["BTC"]
        assert p.u == pytest.approx(10.0 * 10_000 / 100.0)      # = 1000 (그로스 캡)

    def test_heat_6퍼센트_초과_진입은_차단된다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        cell = st.cells["E01"]
        for s in ("ETH", "SOL"):                 # 기존 heat 5% 주입
            cell.positions[s] = FarmPos(d=1, u=25.0, e=100.0, stop=90.0,
                                        kind="BRK", risk_d=10.0)
        fills = step(st, {"BTC": bar(0, 100, 115, 100, 114)})
        assert "BTC" not in cell.positions
        assert not any(f["cell"] == "E01" and f["action"] == "enter" for f in fills)

    def test_일손실_5퍼센트는_진입만_막고_청산하지_않는다(self):
        st = farm()
        warm(st, "BTC", atr=2.0, hi=1000.0, lo=1.0)      # 채널 청산·진입 배제
        cell = st.cells["E01"]
        cell.positions["BTC"] = FarmPos(d=1, u=60.0, e=100.0, stop=1.0,
                                        kind="BRK", risk_d=3.0)
        step(st, {"BTC": bar(0, 100, 100, 100, 100)})
        assert not cell.halted
        step(st, {"BTC": bar(1, 95, 95, 90, 90)})        # 일 MTM -6%
        assert cell.halted and cell.halts == 1
        assert "BTC" in cell.positions, "트리거이지 손실 상한 아님 — 청산 금지"
        warm(st, "ETH", atr=2.0)
        step(st, {"BTC": bar(2, 90, 90, 89, 89),
                  "ETH": bar(2, 100, 115, 100, 114)})    # 당일 신규 진입 정지
        assert "ETH" not in cell.positions
        step(st, {"BTC": bar(24, 89, 89, 88, 88),        # 다음 UTC 일 — 해제
                  "ETH": bar(24, 116, 130, 116, 120)})
        assert not cell.halted
        assert "ETH" in cell.positions

    def test_비용은_편도_8bp_왕복_16bp다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 115, 100, 114)})
        u = st.cells["E01"].positions["BTC"].u
        step(st, {"BTC": bar(1, 89, 89.5, 88.5, 89)})    # 갭 하락 → 시가 청산
        cell = st.cells["E01"]
        assert not cell.positions
        assert cell.cost == pytest.approx(u * 100.0 * COST_SIDE + u * 89.0 * COST_SIDE)


class TestExitModels:
    def test_BRK_갭_청산은_시가로_악화된다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        cell = st.cells["E01"]
        cell.positions["BTC"] = FarmPos(d=1, u=10.0, e=100.0, stop=88.0,
                                        kind="BRK", risk_d=12.0)
        fills = step(st, {"BTC": bar(0, 85, 86, 84, 85)})    # 시가가 채널(90) 아래
        assert fills[0]["action"] == "exit"
        assert fills[0]["price"] == pytest.approx(85.0), "레벨 90이 아닌 시가 85"

    def test_MR_평균회귀_청산은_다음_봉_시가다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0, closes=[99.0, 101.0] * 12)
        cell = st.cells["E07"]
        cell.positions["BTC"] = FarmPos(d=1, u=10.0, e=98.0, stop=50.0,
                                        kind="MR", hold=1, risk_d=4.0)
        step(st, {"BTC": bar(0, 100, 100.5, 100, 100.2)})    # z≥0 → 청산 신호
        assert cell.positions["BTC"].pending_exit == "signal"
        fills = step(st, {"BTC": bar(1, 101, 101.5, 100.5, 101)})
        assert fills[0]["action"] == "exit_signal"
        assert fills[0]["price"] == pytest.approx(101.0)

    def test_MR은_24봉_타임아웃된다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0, closes=[99.0, 101.0] * 12)
        cell = st.cells["E07"]
        cell.positions["BTC"] = FarmPos(d=1, u=10.0, e=98.0, stop=50.0,
                                        kind="MR", hold=23, risk_d=4.0)
        step(st, {"BTC": bar(0, 99.5, 99.6, 99.4, 99.5)})    # z<0 유지, hold→24
        assert cell.positions["BTC"].pending_exit == "timeout"
        fills = step(st, {"BTC": bar(1, 99.4, 99.5, 99.3, 99.4)})
        assert fills[0]["action"] == "exit_timeout"

    def test_RSIDIV_스탑과_목표_동시충족이면_스탑_우선이다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.positions["BTC"] = FarmPos(d=1, u=10.0, e=100.0, stop=95.0,
                                        kind="RSIDIV", tgt=110.0, risk_d=5.0)
        fills = step(st, {"BTC": bar(0, 100, 111, 94, 100)})
        assert fills[0]["action"] == "stop"
        assert fills[0]["price"] == pytest.approx(95.0)

    def test_RSIDIV_대기주문은_다음_1h봉_시가에_체결된다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.pending["BTC"] = {"kind": "RSIDIV", "d": 1, "stop": 95.0,
                               "n4": 5, "ets": BASE}
        step(st, {"BTC": bar(0, 100, 105, 99, 104)})
        p = cell.positions["BTC"]
        assert p.e == pytest.approx(100.0)
        assert p.tgt == pytest.approx(110.0)                 # 2R
        assert p.u == pytest.approx(0.02 * 10_000 / 5.0)
        assert p.n4_entry == 5

    def test_숏_2R_목표는_진입_아래다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.pending["BTC"] = {"kind": "RSIDIV", "d": -1, "stop": 105.0,
                               "n4": 5, "ets": BASE}
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        p = cell.positions["BTC"]
        assert p.tgt == pytest.approx(90.0)      # 3*100 - 2*105

    def test_RSIDIV_스탑은_갭에도_레벨_그대로_체결된다(self):
        # 명세 미결 #15 확정: 원본 유지 — 갭 악화는 BRK/MR에만
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.positions["BTC"] = FarmPos(d=1, u=10.0, e=100.0, stop=95.0,
                                        kind="RSIDIV", tgt=110.0, risk_d=5.0)
        fills = step(st, {"BTC": bar(0, 93, 94, 92, 93)})    # 시가가 스탑 아래
        assert fills[0]["action"] == "stop"
        assert fills[0]["price"] == pytest.approx(95.0), "레벨 체결 (갭 악화 없음)"

    def test_대기주문은_연속성이_깨지면_취소된다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.pending["BTC"] = {"kind": "RSIDIV", "d": 1, "stop": 95.0,
                               "n4": 5, "ets": BASE + H1}    # 다음 봉이 아님
        step(st, {"BTC": bar(0, 100, 105, 99, 104)})
        assert not cell.positions and not cell.pending


class TestPhaseOrder:
    """시가 체결은 봉내 사건보다 먼저다 — Codex 리뷰 반영 인과 규약."""

    def test_대기진입_사이징은_직전_종가_마크를_쓴다(self):
        # 그로스 잔여가 이번 봉 '종가'로 계산되면 미래 정보 사용
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        warm(st, "ETH", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.positions["ETH"] = FarmPos(d=1, u=990.0, e=100.0, stop=1.0,
                                        kind="RSIDIV", tgt=1e9, risk_d=0.01)
        cell.pending["BTC"] = {"kind": "RSIDIV", "d": 1, "stop": 95.0,
                               "n4": 0, "ets": BASE}
        step(st, {"BTC": bar(0, 100, 101, 99, 100.5),
                  "ETH": bar(0, 100, 102, 100, 101.9)})      # ETH 종가 급등
        # 직전 종가 마크(100): 그로스 99000 → 잔여 1000/100 = 10
        # 이번 봉 종가 마크(101.9)였다면 잔여 0 → 차단됐을 것
        assert cell.positions["BTC"].u == pytest.approx(10.0)

    def test_같은_봉_스탑된_시가_체결도_다른_시가_체결을_소급_허용하지_않는다(self):
        # 시가 체결 2건은 동시에 확정 — 첫 체결의 봉내 스탑이 둘째의 용량이 될 수 없다
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        warm(st, "ETH", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.pending["BTC"] = {"kind": "RSIDIV", "d": 1, "stop": 99.9,
                               "n4": 0, "ets": BASE}         # 그로스 캡까지 체결
        cell.pending["ETH"] = {"kind": "RSIDIV", "d": 1, "stop": 95.0,
                               "n4": 0, "ets": BASE}
        fills = step(st, {"BTC": bar(0, 100, 100.5, 99.5, 100),   # 봉내 스탑
                          "ETH": bar(0, 100, 101, 99, 100.4)})
        assert any(f["sym"] == "BTC" and f["action"] == "same_bar_stop"
                   for f in fills)
        assert not any(f["sym"] == "ETH" and f["action"] == "enter" for f in fills)

    def test_갭_시가_체결은_실제_체결가로_마크된다(self):
        # 직전 종가 100, 시가 200 갭 체결 — 이후 체결 사이징이 100으로 마크하면
        # 그로스 캡을 실질 위반한다 (Codex 재현: 14.9x)
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        warm(st, "ETH", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.pending["BTC"] = {"kind": "RSIDIV", "d": 1, "stop": 199.8,
                               "n4": 0, "ets": BASE}
        cell.pending["ETH"] = {"kind": "RSIDIV", "d": 1, "stop": 95.0,
                               "n4": 0, "ets": BASE}
        step(st, {"BTC": bar(0, 200, 200.3, 199.9, 200.2),   # 목표(200.4) 미도달
                  "ETH": bar(0, 100, 101, 99, 100.4)})
        assert cell.positions["BTC"].u == pytest.approx(500.0)   # 100k/200
        assert "ETH" not in cell.positions, "잔여 그로스 0 — 차단돼야 한다"

    def test_봉내_스탑이_같은_봉_시가_체결을_소급_허용하지_않는다(self):
        st = farm()
        warm(st, "BTC", atr=1.0, n=0)
        warm(st, "SOL", atr=1.0, n=0)
        cell = st.cells["E09"]
        cell.positions["BTC"] = FarmPos(d=1, u=1000.0, e=100.0, stop=95.0,
                                        kind="RSIDIV", tgt=1e9, risk_d=0.001)
        cell.pending["SOL"] = {"kind": "RSIDIV", "d": 1, "stop": 95.0,
                               "n4": 0, "ets": BASE}
        fills = step(st, {"BTC": bar(0, 100, 100.5, 94, 94.5),   # 봉내 스탑
                          "SOL": bar(0, 100, 105, 99, 104)})
        assert any(f["sym"] == "BTC" and f["action"] == "stop" for f in fills)
        # 시가 시점 그로스는 BTC 보유분(100k)으로 캡 — 봉내 스탑이 소급으로
        # 용량을 풀어줄 수 없다
        assert "SOL" not in cell.positions


class TestFunding:
    def _held(self, d: int) -> FarmState:
        st = farm()
        warm(st, "BTC", atr=2.0, hi=1000.0, lo=1.0)
        st.cells["E01"].positions["BTC"] = FarmPos(
            d=d, u=10.0, e=100.0, stop=1.0 if d > 0 else 1000.0,
            kind="BRK", risk_d=99.0)
        return st

    def test_롱은_양수_펀딩을_정산봉에서_지불한다(self):
        st = self._held(1)
        step(st, {"BTC": bar(0, 100, 100, 100, 100)}, {"BTC": 0.0001})
        cell = st.cells["E01"]
        assert cell.equity == pytest.approx(10_000 - 0.0001 * 10 * 100)
        assert cell.fund == pytest.approx(0.1)

    def test_숏은_양수_펀딩을_수취한다(self):
        st = self._held(-1)
        step(st, {"BTC": bar(0, 100, 100, 100, 100)}, {"BTC": 0.0001})
        assert st.cells["E01"].equity == pytest.approx(10_000 + 0.1)

    def test_펀딩이_없는_봉은_변화가_없다(self):
        st = self._held(1)
        step(st, {"BTC": bar(0, 100, 100, 100, 100)})
        assert st.cells["E01"].equity == pytest.approx(10_000)

    def test_펀딩은_원장_이벤트로_남는다(self):
        # tracke_null 판정 계약: action=='funding' 행의 pnl 양수 = 수취
        st = self._held(1)
        fills = step(st, {"BTC": bar(0, 100, 100, 100, 100)}, {"BTC": 0.0001})
        ev = [f for f in fills if f["action"] == "funding"]
        assert len(ev) == 1
        assert ev[0]["pnl"] == pytest.approx(-0.1), "롱 지불 = 음수"
        assert ev[0]["cost"] == 0.0
        st2 = self._held(-1)
        fills2 = step(st2, {"BTC": bar(0, 100, 100, 100, 100)}, {"BTC": 0.0001})
        ev2 = [f for f in fills2 if f["action"] == "funding"]
        assert ev2[0]["pnl"] == pytest.approx(+0.1), "숏 수취 = 양수"

    def test_펀딩_손실도_같은_봉에서_일손실을_트리거한다(self):
        st = farm()
        warm(st, "BTC", atr=2.0, hi=1000.0, lo=1.0)
        st.cells["E01"].positions["BTC"] = FarmPos(d=1, u=600.0, e=100.0,
                                                   stop=1.0, kind="BRK",
                                                   risk_d=3.0)
        step(st, {"BTC": bar(0, 100, 100, 100, 100)}, {"BTC": 0.01})   # -6%
        cell = st.cells["E01"]
        assert cell.halted and cell.halts == 1


class TestDivergence:
    def _h4(self) -> dict:
        return {"w": None, "o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "n": 0,
                "pc4": 100.0, "atr4": 2.0, "up": 1.0, "dn": 1.0,
                "b5": [[101.0, 102.0, 30.0], [100.5, 101.5, 28.0],
                       [99.0, 103.0, 25.0], [100.0, 101.0, 27.0]],
                "n4": 6, "plo": [[0, 100.0, 20.0]], "phi": []}

    def test_가격_저점하락_RSI_저점상승이면_롱_신호다(self):
        h4 = self._h4()
        r = _confirm_4h(h4, 100.4, 101.0, 100.2, 100.8)
        assert r is not None and r[0] == 1
        a4 = 2.0 + (1.0 - 2.0) / 14        # TR=1 (prev close 100 갭 포함)
        assert r[1] == pytest.approx(99.0 - 0.5 * a4)
        assert r[2] == 6

    def test_RSI가_같이_내려가면_다이버전스가_아니다(self):
        h4 = self._h4()
        h4["plo"] = [[0, 100.0, 30.0]]     # 직전 피벗 RSI가 더 높음 → r2>r1 실패
        assert _confirm_4h(h4, 100.4, 101.0, 100.2, 100.8) is None

    def test_피벗은_좌우_2봉_극값일_때만_확정된다(self):
        h4 = self._h4()
        h4["b5"] = [[98.0, 102.0, 30.0], [100.5, 101.5, 28.0],
                    [99.0, 103.0, 25.0], [100.0, 101.0, 27.0]]   # 좌측이 더 낮음
        _confirm_4h(h4, 100.4, 101.0, 100.2, 100.8)
        assert len(h4["plo"]) == 1, "새 피벗 저점 미확정"

    def test_50봉_룩백을_넘긴_피벗쌍은_무효다(self):
        h4 = self._h4()
        h4["plo"] = [[-47, 100.0, 20.0]]   # j2(4) - j1(-47) = 51 > 50
        assert _confirm_4h(h4, 100.4, 101.0, 100.2, 100.8) is None


class TestDelist:
    def test_폐지는_마지막_유효가로_청산하고_슬롯은_영구_공석이다(self):
        st = farm()
        warm(st, "BTC", atr=2.0, pc=100.0)
        st.cells["E01"].positions["BTC"] = FarmPos(d=1, u=10.0, e=90.0, stop=1.0,
                                                   kind="BRK", risk_d=89.0)
        fills = mark_delisted(st, "BTC")
        assert fills[0]["action"] == "force_exit"
        assert fills[0]["price"] == pytest.approx(100.0)
        assert "BTC" in st.delisted
        fills2 = step(st, {"BTC": bar(0, 100, 115, 100, 114)})   # 재진입 시도
        assert fills2 == []
        assert not st.cells["E01"].positions


class TestSerde:
    def test_상태_왕복_직렬화(self):
        st = farm(t0=BASE)
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 115, 100, 114)})
        st.cells["E07"].pending["BTC"] = {"kind": "MR", "d": -1, "ets": BASE + H1}
        st2 = FarmState.from_dict(json.loads(
            json.dumps(st.to_dict(), default=float)))
        assert json.dumps(st2.to_dict(), sort_keys=True, default=float) == \
            json.dumps(st.to_dict(), sort_keys=True, default=float)
        assert st2.cells["E01"].positions["BTC"].stop == pytest.approx(88.0)

    def test_직렬화_gross는_봉_종가_명목_레버리지다(self):
        # 계약 (감사 #6): cells[*].gross = sum(|u| x 마지막 유효 종가)/equity —
        # 대시보드 로더가 이 키를 읽는다. 포지션 없으면 0.0.
        st = farm()
        warm(st, "BTC", atr=2.0, pc=110.0)
        cell = st.cells["E01"]
        assert st.to_dict()["cells"]["E01"]["gross"] == 0.0
        cell.positions["BTC"] = FarmPos(d=-1, u=10.0, e=100.0, stop=120.0,
                                        kind="BRK", risk_d=20.0)
        g = st.to_dict()["cells"]["E01"]["gross"]
        assert g == pytest.approx(10.0 * 110.0 / cell.equity)   # 숏도 |u|·종가
        # 역직렬화는 gross 를 무시하고 상태를 복원한다 (읽기 전용 파생값)
        st2 = FarmState.from_dict(json.loads(
            json.dumps(st.to_dict(), default=float)))
        assert st2.to_dict()["cells"]["E01"]["gross"] == pytest.approx(g)

    def test_셀별_시가평가는_마지막_종가_기준이다(self):
        st = farm()
        warm(st, "BTC", atr=2.0, pc=110.0)
        st.cells["E01"].positions["BTC"] = FarmPos(d=1, u=10.0, e=100.0, stop=1.0,
                                                   kind="BRK", risk_d=99.0)
        eqs = farm_equities(st)
        assert eqs["E01"] == pytest.approx(10_000 + 10.0 * 10.0)
        assert eqs["E02"] == pytest.approx(10_000)
        assert len(eqs) == len(CELLS) == 10


class TestRunnerHelpers:
    def test_바스켓B는_거래대금_차상위_3종이다(self):
        tk = [{"symbol": "BTCUSDT", "turnover24h": "9e9"},
              {"symbol": "ETHUSDT", "turnover24h": "5e9"},
              {"symbol": "SOLUSDT", "turnover24h": "3e9"},
              {"symbol": "XRPUSDT", "turnover24h": "900000000"},
              {"symbol": "DOGEUSDT", "turnover24h": "800000000"},
              {"symbol": "1000PEPEUSDT", "turnover24h": "650000000"},
              {"symbol": "LINKUSDT", "turnover24h": "600000000"},
              {"symbol": "BTC-27JUN25", "turnover24h": "700000000"},
              {"symbol": "LOWUSDT", "turnover24h": "1000000"}]
        assert pick_basket_b(tk) == ["XRP", "DOGE", "1000PEPE"]

    def test_봉_갭은_감지된다(self):
        grid = [BASE, BASE + H1, BASE + 2 * H1, BASE + 3 * H1]
        have = {BASE, BASE + H1, BASE + 3 * H1}
        assert missing_hours(have, grid, BASE) == [BASE + 2 * H1]

    def test_데이터_단절은_경계_포함_48시간이다(self):
        end = BASE + 100 * H1
        latest = {"BTC": end, "XRP": end - DATA_STALL_H * H1,        # 정확히 48h
                  "DOGE": end - (DATA_STALL_H - 1) * H1}             # 47h — 유예
        assert stalled_syms(latest, end) == ["XRP"]

    def test_단절_심볼은_잔여봉_재생_후_다음_실행에서_폐지된다(self):
        # Codex 재현: 잔여 캔들에 내부 갭이 있어도 2회차에 반드시 폐지 (영구 차단 금지)
        ohlc = (100.0, 101.0, 99.0, 100.0)
        fresh = BASE + 400 * H1
        # 1회차: XRP 캔들 {100h, 101h, 103h} — 연속 선두 2봉만 재생 대상으로 남는다
        st = farm()
        st.last_ts = BASE + 99 * H1
        warm(st, "XRP", pc=100.0)
        syms = ["BTC", "XRP"]
        data = {"BTC": {t: ohlc for t in range(BASE + 100 * H1, fresh + H1, H1)},
                "XRP": {BASE + 100 * H1: ohlc, BASE + 101 * H1: ohlc,
                        BASE + 103 * H1: ohlc}}
        latest = {s: max(d) for s, d in data.items()}
        fills = apply_stall_policy(st, syms, data, latest, BASE + 100 * H1)
        assert fills == [] and "XRP" in syms, "잔여봉 있으면 아직 폐지 금지"
        assert set(data["XRP"]) == {BASE + 100 * H1, BASE + 101 * H1}
        assert latest["XRP"] == BASE + 101 * H1, "재생은 연속 구간 끝까지 캡"
        # 2회차: since=102h, XRP 응답은 갭 너머 103h뿐 → 연속 구간 없음 → 폐지
        st.last_ts = BASE + 101 * H1
        st.cells["E02"].positions["XRP"] = FarmPos(d=1, u=10.0, e=90.0, stop=1.0,
                                                   kind="BRK", risk_d=3.0)
        syms = ["BTC", "XRP"]
        data = {"BTC": {t: ohlc for t in range(BASE + 102 * H1, fresh + H1, H1)},
                "XRP": {BASE + 103 * H1: ohlc}}
        latest = {s: max(d) for s, d in data.items()}
        fills = apply_stall_policy(st, syms, data, latest, BASE + 102 * H1)
        assert "XRP" in st.delisted and "XRP" not in syms
        assert fills[0]["action"] == "force_exit"
        assert fills[0]["price"] == pytest.approx(100.0), "마지막 처리 종가 청산"

    def test_연속_선두_구간만_남긴다(self):
        d = {BASE: 1, BASE + H1: 2, BASE + 3 * H1: 4}        # 2h 지점 갭
        assert contiguous_prefix(d, BASE) == {BASE: 1, BASE + H1: 2}
        assert contiguous_prefix(d, BASE + 2 * H1) == {}

    def test_구_스키마_원장에도_funding_열이_이월된다(self, tmp_path):
        led = tmp_path / "ledger.csv"
        old_cols = [c for c in LEDGER_COLS if c != "funding"]
        pd.DataFrame([dict(cell="E01", sym="BTC", strategy="BRK24",
                           bar_close=BASE + H1, action="enter", price=100.0,
                           qty=1.0, pnl=0.0, cost=0.08,
                           direction=1)])[old_cols].to_csv(led, index=False)
        rows = pd.DataFrame([dict(cell="E01", sym="BTC", strategy="BRK24",
                                  bar_close=BASE + 2 * H1, action="exit",
                                  price=101.0, qty=1.0, pnl=1.0, cost=0.08,
                                  direction=1, funding=0.0)])[LEDGER_COLS]
        append_csv_atomic(led, rows, ["cell", "sym", "strategy", "bar_close",
                                      "action"])
        out = pd.read_csv(led)
        assert not out["funding"].isna().any(), "구 행 funding NaN 오염 금지"

    def test_늦게_상장된_심볼의_선행_결측은_갭이_아니다(self):
        grid = [BASE, BASE + H1, BASE + 2 * H1, BASE + 3 * H1]
        have = {BASE + 2 * H1, BASE + 3 * H1}
        assert missing_hours(have, grid, BASE + 2 * H1) == []

    def test_원장_append는_유일키로_멱등이다(self, tmp_path):
        led = tmp_path / "ledger.csv"
        rows = pd.DataFrame([dict(cell="E01", sym="BTC", strategy="BRK24",
                                  bar_close=BASE + H1, action="enter",
                                  price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                                  direction=1, funding=0.0)])[LEDGER_COLS]
        assert append_csv_atomic(led, rows,
                                 ["cell", "sym", "strategy", "bar_close",
                                  "action"]) == 1
        assert append_csv_atomic(led, rows,
                                 ["cell", "sym", "strategy", "bar_close",
                                  "action"]) == 0, "재실행 시 중복 없음"
        assert len(pd.read_csv(led)) == 1

    def test_원자적_저장은_임시파일을_남기지_않는다(self, tmp_path):
        p = tmp_path / "state.json"
        _atomic_write(p, "{\"a\": 1}")
        assert json.loads(p.read_text()) == {"a": 1}
        assert list(tmp_path.glob("*.tmp")) == []

    def test_첫_저장은_체결이_없어도_세_파일을_전부_만든다(self, tmp_path, monkeypatch):
        # 원장이 없으면 워크플로 git add pathspec 실패로 T0 동결이 유실된다
        monkeypatch.chdir(tmp_path)
        _save_all(farm(), [], BASE, 0)
        led = pd.read_csv(tmp_path / LEDGER)
        assert list(led.columns) == LEDGER_COLS and len(led) == 0
        assert (tmp_path / "logs/tracke_history.csv").exists()
        assert (tmp_path / "logs/tracke_state.json").exists()

    def test_봉_ts_불일치는_오류다(self):
        st = farm()
        with pytest.raises(ValueError):
            step(st, {"BTC": bar(0, 100, 101, 99, 100),
                      "ETH": bar(1, 100, 101, 99, 100)})


class TestGapAging:
    """감사 #2 — 영구 봉 갭이 팜을 영구 동결시키지 않는다 (노화 갭 결측 재생)."""

    OHLC = (100.0, 101.0, 99.0, 100.0)

    def _grid(self, n: int) -> list:
        return [BASE + k * H1 for k in range(n)]

    def test_신선_갭은_전체_중단_사유를_돌려준다(self):
        grid = self._grid(10)
        data = {"BTC": {t: self.OHLC for t in grid if t != grid[7]}}
        reason = check_gaps(data, grid, grid[-1], continuing=True)
        assert reason is not None and "BTC" in reason

    def test_노화_갭은_결측_재생으로_진행한다(self):
        grid = self._grid(60)
        hole = {grid[5], grid[6]}          # 최신 결측 5h vs 최신 봉 59h — 노화
        data = {"BTC": {t: self.OHLC for t in grid if t not in hole}}
        assert check_gaps(data, grid, grid[-1], continuing=True) is None

    def test_노화_경계는_48시간_이상이다(self):
        grid = self._grid(DATA_STALL_H + 2)          # 지연 정확히 48h — 노화
        data = {"BTC": {t: self.OHLC for t in grid if t != grid[1]}}
        assert check_gaps(data, grid, grid[-1], continuing=True) is None
        grid2 = self._grid(DATA_STALL_H + 1)         # 지연 47h — 신선
        data2 = {"BTC": {t: self.OHLC for t in grid2 if t != grid2[1]}}
        assert check_gaps(data2, grid2, grid2[-1], continuing=True) is not None

    def test_계속_실행의_선두_결손도_갭이며_노화되면_진행한다(self):
        grid = self._grid(60)
        data = {"BTC": {t: self.OHLC for t in grid[5:]}}     # 선두 5봉 결손
        assert check_gaps(data, grid, grid[-1], continuing=True) is None

    def test_계속_실행의_신선한_선두_결손은_중단한다(self):
        grid = self._grid(10)
        data = {"BTC": {t: self.OHLC for t in grid[3:]}}
        assert check_gaps(data, grid, grid[-1], continuing=True) is not None

    def test_워밍업의_늦은_상장_선두_결손은_갭이_아니다(self):
        grid = self._grid(10)
        data = {"XRP": {t: self.OHLC for t in grid[7:]}}     # 신선하지만 워밍업
        assert check_gaps(data, grid, grid[-1], continuing=False) is None


class TestObservability:
    """감사 #3 — 중단 사유 마커 (Actions 초록 위장 방지, 종료코드 0 유지)."""

    def test_중단_마커는_사유와_스텝요약을_남기고_성공시_지워진다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        summ = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summ))
        _abort("BTC 봉 갭 3개 (예: 123)")
        assert "BTC 봉 갭 3개" in (tmp_path / ERR_MARK).read_text()
        assert "BTC 봉 갭 3개" in summ.read_text()
        _clear_abort()
        assert not (tmp_path / ERR_MARK).exists()

    def test_스텝요약_변수가_없어도_마커는_남고_삭제는_멱등이다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        _clear_abort()                                   # 파일 없음 — 예외 없음
        _abort("펀딩 조회 실패")
        assert "펀딩 조회 실패" in (tmp_path / ERR_MARK).read_text()


class TestHistoryKeep:
    """감사 #4 — 이력은 같은 ts 재저장 시 최신 행, 원장은 keep='first' 멱등."""

    def test_이력은_같은_ts_재저장시_최신_행이_남는다(self, tmp_path, monkeypatch):
        # 폐지 강제청산 뒤 같은 ts 보존 저장이 keep='first'에 버려지면
        # 청산 반영 자본이 이력에서 영구 유실된다
        monkeypatch.chdir(tmp_path)
        st = farm()
        _save_all(st, [], BASE, 0)
        st.cells["E01"].equity = 9_000.0                 # 강제청산 손실 반영 가정
        _save_all(st, [], BASE, 0)
        hist = pd.read_csv(tmp_path / "logs/tracke_history.csv")
        assert len(hist) == 1
        assert hist["e01"].iloc[0] == pytest.approx(9_000.0)

    def test_원장은_재실행이_기존_이벤트를_덮지_못한다(self, tmp_path):
        led = tmp_path / "ledger.csv"
        r1 = pd.DataFrame([dict(cell="E01", sym="BTC", strategy="BRK24",
                                bar_close=BASE + H1, action="enter", price=100.0,
                                qty=1.0, pnl=0.0, cost=0.08, direction=1,
                                funding=0.0)])[LEDGER_COLS]
        r2 = r1.copy()
        r2["price"] = 999.0                              # 같은 유일키, 다른 값
        append_csv_atomic(led, r1, LEDGER_KEY)
        append_csv_atomic(led, r2, LEDGER_KEY)
        out = pd.read_csv(led)
        assert len(out) == 1
        assert out["price"].iloc[0] == pytest.approx(100.0), "멱등 계약 keep='first'"


class TestNullContract:
    """감사 #1 end-to-end — 실제 엔진 재생 원장이 공식 판정 로더를 통과한다."""

    def test_엔진_합성_재생_원장은_official_로더를_통과한다(
            self, tmp_path, monkeypatch):
        # tracke_null.load_ledger(official=True)는 펀딩 기록·유일키 완전성을
        # 요구한다 — 통과 못 하면 첫 판정일(T+30)에 반드시 거부된다.
        from lab.tracke_null import load_ledger
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        fills = list(step(st, {"BTC": bar(0, 100, 115, 100, 114)}))   # BRK 진입
        u = st.cells["E01"].positions["BTC"].u
        fills += step(st, {"BTC": bar(1, 114, 114.5, 113.5, 114)},
                      {"BTC": 0.0001})                                # 펀딩 정산
        fills += step(st, {"BTC": bar(2, 89, 89.5, 88.5, 89)})        # 채널 청산
        assert {f["action"] for f in fills} == {"enter", "funding", "exit"}
        _save_all(st, fills, BASE + 2 * H1, 3)
        led = load_ledger(tmp_path / LEDGER, official=True)           # 예외 없어야 함
        assert len(led) == len(fills)
        fund = led[led["funding"] != 0.0]
        assert len(fund) == 1
        assert fund["funding"].iloc[0] == pytest.approx(-0.0001 * u * 114.0), \
            "롱 지불 = 음수 (양수 = 수취 계약)"
        assert fund["gross"].iloc[0] == 0.0, "펀딩 행 gross 재분류"
        assert led["cost"].sum() == pytest.approx(st.cells["E01"].cost)


class TestFirewall:
    def test_승급_게이트_입력에_트랙E_경로가_없다(self):
        """구조적 방화벽 — 승급 판정 코드가 Track E 상태·이력을 읽지 않는다."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        gate = root / "src"
        hits = []
        for f in gate.rglob("*.py"):
            if "dashboard" in f.parts:        # 표시 전용 — 게이트 아님
                continue
            text = f.read_text(errors="ignore")
            if "tracke_state" in text or "tracke_history" in text or \
                    "tracke_ledger" in text:
                hits.append(str(f))
        assert hits == [], f"승급 경로에서 Track E 파일 참조 금지: {hits}"
