"""Track E 단타 팜 테스트 — 인과성·리스크·멱등·원자성 (전부 합성 데이터, 네트워크 0)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from carrybot.aggressive.scalp_farm import (
    ATR1H_N,
    BB_N,
    BRK_ATR_MULT,
    COST_SIDE,
    GATE_SMA_N,
    GATE_VOL_N,
    H1,
    HEAT_CAP,
    MR_ATR_MULT,
    RSI2_EXIT_N,
    RSI2_TREND_N,
    V2CELLS,
    V2LABELS,
    V2_HEAT_FRAC,
    V2_NOTIONAL_FRAC,
    V3CELLS,
    V3LABELS,
    V3_COST_MODEL,
    V3_COST_SIDE_ALT,
    V3_MAX_POS,
    V3_RISK,
    V3_UNIVERSE_N,
    V4CELLS,
    V4LABELS,
    V4_TP_R,
    V5CELLS,
    V5LABELS,
    V5_ADD_ATR,
    V5_TRANCHES,
    V5_TRANCHE_FRAC,
    VAR_MAX_HOLD,
    VAR_TP_R,
    VBASKET_LABELS,
    VCELLS,
    VLABELS,
    BarE,
    CELLS,
    CellSpec,
    FarmPos,
    FarmState,
    _confirm_4h,
    _new_ind,
    _new_x2,
    cell_syms,
    farm_equities,
    mark_delisted,
    new_farm,
    new_variant,
    new_variant2,
    new_variant3,
    new_variant4,
    new_variant5,
    step,
    step_variant,
    step_variant2,
    step_variant3,
    step_variant4,
    step_variant5,
    variant2_delist,
    variant2_equities,
    variant2_from_dict,
    variant2_to_dict,
    variant3_delist,
    variant3_equities,
    variant3_from_dict,
    variant3_to_dict,
    variant4_delist,
    variant4_equities,
    variant4_from_dict,
    variant4_to_dict,
    variant5_delist,
    variant5_equities,
    variant5_from_dict,
    variant5_to_dict,
    variant_from_dict,
    variant_to_dict,
    warmup_full,
    warmup_x2,
)
from carrybot.aggressive.scalp_farm_runner import (
    DATA_STALL_H,
    ERR_MARK,
    VHIST,
    VHIST_COLS,
    VLEDGER,
    apply_stall_policy,
    check_gaps,
    contiguous_prefix,
    LEDGER,
    LEDGER_COLS,
    LEDGER_KEY,
    _abort,
    _atomic_write,
    _clear_abort,
    _finalize_variant,
    _finalize_variant2,
    _finalize_variant4,
    _finalize_variant5,
    _migrate_vhist_schema,
    _run_variant,
    _run_variant2,
    _run_variant3,
    _run_variant4,
    _run_variant5,
    _safe_variant,
    _save_all,
    _save_variant,
    _save_variant2,
    _save_variant3,
    _save_variant4,
    _save_variant5,
    _variant3_inputs,
    append_csv_atomic,
    fetch_funding_range,
    missing_hours,
    pick_basket_b,
    pick_universe40,
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


def ledger_lines_raw(path, cells: tuple) -> bytes:
    """원장 CSV 의 헤더 + 지정 셀 물리 행을 **원문 바이트 그대로** 뽑는다.

    회귀 계약이 "바이트 동일"이므로 pandas 왕복(read_csv→to_csv)으로 비교하면
    안 된다 — 기본 파서가 correctly-rounded 가 아니라 비교 양쪽에 파서 오차를
    덧씌우고(플랫폼 의존), 열 dtype 추론이 다른 행 때문에 바뀌면 값과 무관하게
    표기가 갈린다. 줄바꿈·행 순서까지 원문으로 비교한다.
    """
    lines = path.read_bytes().splitlines(keepends=True)
    # 첫 열이 셀 ID 라는 가정을 조용히 두지 않는다 (Codex 검토)
    assert lines[0].decode().strip().split(",") == LEDGER_COLS, "원장 스키마 가정"
    want = tuple(c.encode() for c in cells)
    return lines[0] + b"".join(ln for ln in lines[1:]
                               if ln.split(b",", 1)[0] in want)


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


class TestLedgerByteImmutability:
    """무결성 교정 2026-08-29 — append 재작성이 기존 행 바이트를 못 바꾼다.

    예전 append_csv_atomic 은 파일 전체를 `pd.read_csv` 기본 파서로 되읽어
    재작성했다. 그 파서(float_precision='high')는 correctly-rounded 가 아니라
    왕복마다 값이 ±1 ULP 흔들렸고, 이동 방향이 플랫폼마다 달라 "같은 파일에
    몇 번 append 되었나"(= 변형 그룹 활성 수)가 산출 바이트를 바꾸는 그룹 간
    결합을 만들었다. 아래 값들은 그 드리프트를 실제로 유발했던 값이다.
    """

    DRIFTY = dict(price=100.0, qty=16.666666666666668, pnl=-183.33333333333334,
                  cost=1.3333333333333335, direction=1,
                  funding=-0.17333333333333334)

    def _row(self, ts: int, **kw) -> pd.DataFrame:
        d = dict(cell="E01", sym="BTC", strategy="BRK24", bar_close=ts,
                 action="enter", **self.DRIFTY)
        d.update(kw)
        return pd.DataFrame([d])[LEDGER_COLS]

    def test_새_행_추가는_기존_바이트에_새_줄만_덧붙인다(self, tmp_path):
        led = tmp_path / "ledger.csv"
        append_csv_atomic(led, self._row(BASE + H1), LEDGER_KEY)
        before = led.read_bytes()
        assert b"-183.33333333333334" in before, "쓰인 값 자체가 계산값 그대로"
        assert append_csv_atomic(led, self._row(BASE + 2 * H1), LEDGER_KEY) == 1
        after = led.read_bytes()
        assert after.startswith(before), "기존 바이트 앞부분 그대로 보존"
        assert after[len(before):].count(b"\n") == 1, "덧붙은 건 새 줄 하나뿐"

    def test_빈_append도_파일_바이트를_바꾸지_않는다(self, tmp_path):
        led = tmp_path / "ledger.csv"
        append_csv_atomic(led, self._row(BASE + H1), LEDGER_KEY)
        before = led.read_bytes()
        assert append_csv_atomic(led, pd.DataFrame([], columns=LEDGER_COLS),
                                 LEDGER_KEY) == 0
        assert led.read_bytes() == before, "체결 0인 시간의 재작성도 무변화"

    def test_같은_키_재실행은_바이트_단위로_멱등이다(self, tmp_path):
        led = tmp_path / "ledger.csv"
        append_csv_atomic(led, self._row(BASE + H1), LEDGER_KEY)
        before = led.read_bytes()
        assert append_csv_atomic(led, self._row(BASE + H1, price=999.0),
                                 LEDGER_KEY) == 0
        assert led.read_bytes() == before, "keep='first' — 값까지 불변"

    def test_신규_행이_기존_정수열을_실수로_승격시키지_못한다(self, tmp_path):
        # 기존 price 토큰이 "100" 인 파일에 89.5 가 들어오면 열 dtype 추론이
        # int64→float64 로 올라가 기존 행이 "100.0" 으로 다시 쓰이는 사고
        led = tmp_path / "ledger.csv"
        append_csv_atomic(led, self._row(BASE + H1, price=100), LEDGER_KEY)
        assert b",100," in led.read_bytes()
        append_csv_atomic(led, self._row(BASE + 2 * H1, price=89.5), LEDGER_KEY)
        lines = led.read_bytes().splitlines()
        assert lines[1].split(b",")[5] == b"100", "기존 정수 토큰 보존"
        assert lines[2].split(b",")[5] == b"89.5"

    def test_헤더만_있는_파일도_바이트가_보존된다(self, tmp_path):
        # 체결 0으로 시작한 T0 실행이 만든 헤더 전용 파일 (감사 — git add 함정)
        led = tmp_path / "ledger.csv"
        append_csv_atomic(led, pd.DataFrame([], columns=LEDGER_COLS), LEDGER_KEY)
        head = led.read_bytes()
        assert head.decode().strip().split(",") == LEDGER_COLS
        assert append_csv_atomic(led, pd.DataFrame([], columns=LEDGER_COLS),
                                 LEDGER_KEY) == 0
        assert led.read_bytes() == head, "헤더 전용 파일 재작성 무변화"
        assert append_csv_atomic(led, self._row(BASE + H1), LEDGER_KEY) == 1
        assert led.read_bytes().startswith(head), "헤더 바이트 그대로 prefix"

    def test_이력_keep_last_교체는_무관한_행을_건드리지_않는다(self, tmp_path):
        # 이력은 keep='last' upsert — 갱신 대상 외 행의 원문·순서가 불변이어야
        hist = tmp_path / "hist.csv"
        cols = ["ts", "equity", "e01"]

        def row(ts, eq):
            return pd.DataFrame([dict(ts=ts, equity=eq, e01=eq)])[cols]
        append_csv_atomic(hist, row(BASE, 9999.99999999), ["ts"], keep="last")
        append_csv_atomic(hist, row(BASE + H1, 1.3333333333333335), ["ts"],
                          keep="last")
        before = hist.read_bytes().splitlines()
        assert append_csv_atomic(hist, row(BASE + H1, 7.0), ["ts"],
                                 keep="last") == 0, "교체는 행 증가 0"
        after = hist.read_bytes().splitlines()
        assert len(after) == 3
        assert after[0] == before[0] and after[1] == before[1], "무관한 행 불변"
        assert after[2].decode().split(",")[1] == "7.0", "대상 행만 교체"

    def test_이력_스키마_정렬은_기존_토큰을_보존한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs").mkdir()
        old_cols = ["day", "ts", "equity", "e11", "e12", "n_pos", "bars", "fills"]
        (tmp_path / VHIST).write_text(
            ",".join(old_cols) + "\n"
            "2026-01-01,1767225600000,20000.0,10000.0,"
            "9999.99999999,0,3,2\n")
        _migrate_vhist_schema()
        lines = (tmp_path / VHIST).read_bytes().splitlines()
        assert lines[0].decode().split(",") == VHIST_COLS
        cells = dict(zip(VHIST_COLS, lines[1].decode().split(",")))
        assert cells["e12"] == "9999.99999999", "구 스키마 토큰 원문 보존"
        assert cells["equity"] == "20000.0" and cells["fills"] == "2"
        assert all(cells[c] == "0.0" for c in
                   ("e13", "e14", "e15", "e16", "e17", "e18",
                    "e19", "e20", "e21")), "결여 열은 0.0 이월"


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


def vpos(**kw) -> FarmPos:
    """변형 BRKTP 보유 포지션 기본값 (롱 e=100, stop=88, tgt=112, hold=1)."""
    base = dict(d=1, u=10.0, e=100.0, stop=88.0, kind="BRKTP", tgt=112.0,
                hold=1, risk_d=12.0)
    base.update(kw)
    return FarmPos(**base)


class TestVariantRules:
    """E11·E12 (BRK24TP) — 사전 고정 규칙: 1R 익절·12봉 타임아웃·BRK24 병존."""

    def test_진입_스탑_사이징은_BRK24와_정확히_같다(self):
        # 같은 워밍업·같은 봉에서 E01(BRK24)과 E11(BRK24TP)은 완전 동일 체결
        st = farm()
        warm(st, "BTC", atr=2.0)
        v = new_variant(st, t0=1)
        b = bar(0, 100, 105, 100, 104)               # 돌파하되 1R(112) 미도달
        fills = step(st, {"BTC": b})
        vfills = step_variant(v, {"BTC": b})
        p = st.cells["E01"].positions["BTC"]
        q = v.cells["E11"].positions["BTC"]
        assert (q.e, q.stop, q.u, q.risk_d) == (p.e, p.stop, p.u, p.risk_d)
        e1 = next(f for f in fills if f["cell"] == "E01" and f["action"] == "enter")
        e11 = next(f for f in vfills if f["cell"] == "E11" and f["action"] == "enter")
        assert (e11["price"], e11["qty"], e11["cost"]) == \
            (e1["price"], e1["qty"], e1["cost"])
        assert e11["strategy"] == "BRK24TP"
        assert q.kind == "BRKTP"
        assert q.tgt == 2.0 * q.e - q.stop           # 1R = fill + (fill - stop)

    def test_1R_목표는_봉내_레벨_체결이다(self):
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=100.0, lo=1.0)    # 역채널(저가 1) 배제
        v.cells["E11"].positions["BTC"] = vpos()
        fills = step_variant(v, {"BTC": bar(0, 105, 113, 104, 106)})
        assert fills[0]["action"] == "target"
        assert fills[0]["price"] == pytest.approx(112.0)
        assert "BTC" not in v.cells["E11"].positions

    def test_갭이_유리해도_목표는_레벨_체결이다(self):
        # RSI-DIV #15 관례 동일 — 시가 120 > 목표 112 여도 112 체결
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=100.0, lo=1.0)
        v.cells["E11"].positions["BTC"] = vpos()
        fills = step_variant(v, {"BTC": bar(0, 120, 125, 118, 121)})
        assert fills[0]["action"] == "target"
        assert fills[0]["price"] == pytest.approx(112.0), "레벨 체결 (유리한 갭 무시)"

    def test_스탑과_목표_동시_도달이면_BRK_청산이_우선이다(self):
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=100.0, lo=1.0)
        v.cells["E11"].positions["BTC"] = vpos()
        fills = step_variant(v, {"BTC": bar(0, 100, 113, 80, 90)})   # 88·112 모두 터치
        assert fills[0]["action"] == "exit"
        assert fills[0]["price"] == pytest.approx(88.0), "스탑 우선 (비관)"
        assert not any(f["action"] == "target" for f in fills)

    def test_역채널_추적_청산은_변형에도_유지된다(self):
        # 기존 BRK24 청산 규칙 병존 — n/2=12봉 역채널 + 갭 악화
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=100.0, lo=95.0)   # 최근 저가 95 = 역채널
        v.cells["E11"].positions["BTC"] = vpos()
        fills = step_variant(v, {"BTC": bar(0, 96, 97, 94, 96)})
        assert fills[0]["action"] == "exit"
        assert fills[0]["price"] == pytest.approx(95.0), "역채널 레벨 (스탑 88 아님)"

    def test_12봉_타임아웃은_도달_봉_종가_청산이다(self):
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=1.0)   # 채널 진입·청산 배제
        v.cells["E11"].positions["BTC"] = vpos(stop=1.0, tgt=1e9, hold=10,
                                               risk_d=99.0)
        step_variant(v, {"BTC": bar(0, 100, 100.5, 99.5, 100)})      # hold 11
        assert "BTC" in v.cells["E11"].positions, "11봉째는 생존"
        fills = step_variant(v, {"BTC": bar(1, 100, 100.5, 99.5, 99.7)})  # hold 12
        assert fills[0]["action"] == "timeout"
        assert fills[0]["price"] == pytest.approx(99.7), "그 봉 종가 청산"

    def test_진입부터_12봉째_봉에서_타임아웃된다(self):
        # 실제 진입 경로 검증 — B0 진입(hold=1), B1~B10 생존, B11(12봉째) 종가 청산
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0)                      # 채널 100/90
        step_variant(v, {"BTC": bar(0, 100, 105, 100, 104)})     # 돌파 진입
        assert v.cells["E11"].positions["BTC"].hold == 1, "진입봉 = 1"
        for k in range(1, VAR_MAX_HOLD - 1):                     # B1..B10
            step_variant(v, {"BTC": bar(k, 104, 105, 103, 104)})
        p = v.cells["E11"].positions["BTC"]
        assert p.hold == VAR_MAX_HOLD - 1, "B10 까지 생존 (hold=11)"
        fills = step_variant(v, {"BTC": bar(VAR_MAX_HOLD - 1,
                                            104, 105, 103, 104.3)})
        assert fills[0]["action"] == "timeout"
        assert fills[0]["price"] == pytest.approx(104.3), "B11(12봉째) 종가"
        assert "BTC" not in v.cells["E11"].positions

    def test_결측_봉은_타임아웃_카운트를_멈춘다(self):
        # 엔진 공통 fail-closed — 결측 봉은 관리(카운트·청산) 전부 정지
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=1.0)
        warm(v, "ETH", atr=2.0, hi=1000.0, lo=1.0)
        v.cells["E11"].positions["BTC"] = vpos(stop=1.0, tgt=1e9, hold=11,
                                               risk_d=99.0)
        step_variant(v, {"ETH": bar(0, 100, 100.5, 99.5, 100)})      # BTC 결측
        assert v.cells["E11"].positions["BTC"].hold == 11, "카운트 정지"
        fills = step_variant(v, {"BTC": bar(1, 100, 100.5, 99.5, 99.9),
                                 "ETH": bar(1, 100, 100.5, 99.5, 100)})
        assert any(f["action"] == "timeout" for f in fills)

    def test_진입봉_1R_도달은_같은_봉_익절된다(self):
        # RSI-DIV #17 관례 — 체결봉부터 목표 검사 (레벨 체결)
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0)
        fills = step_variant(v, {"BTC": bar(0, 100, 113, 100, 110)})
        acts = [f["action"] for f in fills if f["cell"] == "E11"]
        assert acts == ["enter", "same_bar_target"]
        tgt_fill = fills[-1]
        assert tgt_fill["price"] == pytest.approx(112.0)
        assert "BTC" not in v.cells["E11"].positions

    def test_진입봉_동시_도달은_스탑이_우선이다(self):
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0)
        fills = step_variant(v, {"BTC": bar(0, 100, 113, 87, 100)})  # 88·112 모두
        acts = [f["action"] for f in fills if f["cell"] == "E11"]
        assert acts == ["enter", "same_bar_stop"]
        assert not any(f["action"] == "same_bar_target" for f in fills)

    def test_숏_1R_목표는_진입_아래_대칭이다(self):
        v = new_variant(farm(), t0=1)
        warm(v, "BTC", atr=2.0, hi=100.0, lo=90.0)
        step_variant(v, {"BTC": bar(0, 89, 89.5, 80, 82)})   # 하향 돌파 숏
        p = v.cells["E11"].positions["BTC"]
        assert p.d == -1
        assert p.e == pytest.approx(89.0)                    # min(시가, 채널 90)
        assert p.stop == pytest.approx(101.0)                # 89 + 6*2
        assert p.tgt == pytest.approx(77.0), "2*89 - 101 (1R 아래)"

    def test_변형은_t0_variant_이전_워밍업_무주문이다(self):
        v = new_variant(farm(), t0=BASE + 100 * H1)
        warm(v, "BTC", atr=2.0)
        fills = step_variant(v, {"BTC": bar(0, 100, 115, 100, 114)})  # 명백한 돌파
        assert fills == []
        for c in v.cells.values():
            assert not c.positions and not c.pending

    def test_변형_동결_상수와_라벨(self):
        assert VAR_TP_R == 1.0 and VAR_MAX_HOLD == 12
        assert [s.cell for s in VCELLS] == ["E11", "E12"]
        assert all(s.strategy == "BRK24TP" and s.n == 24 for s in VCELLS)
        assert [s.basket for s in VCELLS] == ["A", "B"]
        assert VLABELS["E11"] == VLABELS["E12"] == \
            "빠른 익절 변형 · 미검증 · 판정 권한 없음"


class TestVariantState:
    """변형 서브상태 — variant_cells 키, t0_variant write-once, 지표 스냅숏 분리."""

    def test_new_variant는_동결_바스켓_재사용과_지표_깊은_분리다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        v = new_variant(st, t0=123)
        assert v.t0 == 123
        assert v.basket_b == st.basket_b == ["XRP", "DOGE", "ADA"]
        assert v.last_ts == st.last_ts, "t0_variant 이전 봉 재생 구조적 차단"
        assert v.ind == st.ind, "시장 전용 지표 스냅숏 상속 (값 동일)"
        assert set(v.cells) == {"E11", "E12"}
        assert all(c.equity == 10_000.0 for c in v.cells.values())
        v.ind["BTC"]["atr1"] = 999.0                 # 변형 쪽 변이가
        v.ind["BTC"]["hl"].append([1.0, 1.0])
        assert st.ind["BTC"]["atr1"] != 999.0, "본 팜 지표에 새면 안 된다 (깊은 복사)"
        assert len(st.ind["BTC"]["hl"]) != len(v.ind["BTC"]["hl"])

    def test_변형_직렬화는_t0_variant_명명으로_왕복된다(self):
        v = new_variant(farm(), t0=BASE)
        warm(v, "BTC", atr=2.0)
        step_variant(v, {"BTC": bar(0, 100, 105, 100, 104)})  # 포지션 포함 상태
        d = variant_to_dict(v)
        assert "t0_variant" in d and "t0" not in d and "variant_cells" not in d
        v2 = variant_from_dict(json.loads(json.dumps(d, default=float)))
        assert json.dumps(variant_to_dict(v2), sort_keys=True, default=float) == \
            json.dumps(d, sort_keys=True, default=float)
        assert v2.cells["E11"].positions["BTC"].kind == "BRKTP"

    def test_본_상태는_variant_cells를_불투명하게_왕복_보존한다(self):
        st = farm()
        assert st.variant_cells is None, "미초기화 기본값 None (초기화 대상)"
        v = new_variant(st, t0=BASE)
        st.variant_cells = variant_to_dict(v)
        rt = FarmState.from_dict(json.loads(json.dumps(st.to_dict(), default=float)))
        assert rt.variant_cells == st.variant_cells
        assert json.dumps(rt.to_dict(), sort_keys=True, default=float) == \
            json.dumps(st.to_dict(), sort_keys=True, default=float)

    def test_손상된_변형_상태는_fail_closed다(self):
        # 부재(None)만 초기화 대상 — 손상은 예외 (조용한 재초기화 = t0 이동 금지)
        with pytest.raises(ValueError):
            variant_from_dict(None)
        with pytest.raises(ValueError):
            variant_from_dict({})
        with pytest.raises(ValueError):
            variant_from_dict({"t0_variant": 0, "cells": {}})
        with pytest.raises(ValueError):
            variant_from_dict({"t0_variant": 5,
                               "cells": {"E11": {"equity": 10_000.0}}})


class TestVariantRunner:
    """원장 분리·격리 실패·따라잡기·t0 불변·종말 폐지 정리."""

    OHLC = (100.0, 101.0, 99.0, 100.0)

    def _replayed(self, with_funding: bool = True):
        """본 E01 + 변형 E11 이 같은 합성 시퀀스를 재생한 (st, v, fills, vfills)."""
        st = farm()
        warm(st, "BTC", atr=2.0)
        v = new_variant(st, t0=1)
        seq = [(bar(0, 100, 105, 100, 104), 0.0),            # 돌파 진입
               (bar(1, 104, 104.5, 103.5, 104), 0.0001 if with_funding else 0.0),
               (bar(2, 89, 89.5, 88.5, 89), 0.0)]            # 갭 하락 청산
        fills: list = []
        vfills: list = []
        for b, f in seq:
            fm = {"BTC": f} if f else None
            fills += step(st, {"BTC": b}, fm)
            vfills += step_variant(v, {"BTC": b}, fm)
        return st, v, fills, vfills

    def test_원장은_분리되고_공식_로더는_본_원장만_통과시킨다(
            self, tmp_path, monkeypatch):
        from lab.tracke_null import load_ledger
        monkeypatch.chdir(tmp_path)
        st, v, fills, vfills = self._replayed()
        _save_all(st, fills, BASE + 2 * H1, 3)
        st.variant_cells = variant_to_dict(v)
        _save_variant(st, v, vfills, BASE + 2 * H1, 3)
        led = pd.read_csv(tmp_path / LEDGER)
        vled = pd.read_csv(tmp_path / VLEDGER)
        assert set(led["cell"]) <= {s.cell for s in CELLS}, "본 원장에 E11/E12 금지"
        assert set(vled["cell"]) == {"E11"}
        assert set(vled["strategy"]) == {"BRK24TP"}
        assert {"enter", "funding", "exit"} <= set(vled["action"])
        assert load_ledger(tmp_path / LEDGER, official=True) is not None
        # 분리하지 않았다면: 합쳐진 원장은 공식 로더가 미지 셀로 거부한다
        merged = tmp_path / "merged.csv"
        pd.concat([led, vled], ignore_index=True).to_csv(merged, index=False)
        with pytest.raises(ValueError, match="알 수 없는 셀"):
            load_ledger(merged, official=True)
        # 상태는 같은 파일의 variant_cells 키
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant_cells"]["t0_variant"] == 1
        assert set(raw["variant_cells"]["cells"]) == {"E11", "E12"}
        vh = pd.read_csv(tmp_path / VHIST)
        assert list(vh.columns) == ["day", "ts", "equity", "e11", "e12",
                                    "e13", "e14", "e15", "e16", "e17", "e18",
                                    "e19", "e20", "e21", "e22", "e23",
                                    "e24", "e25", "n_pos", "bars", "fills"]
        # 변형2·3·4·5 미초기화 — e13~e25 열은 부재 표기 0.0, equity 는 E11+E12 만
        assert (vh[["e13", "e14", "e15", "e16", "e17", "e18", "e19", "e20",
                    "e21", "e22", "e23", "e24", "e25"]] == 0.0).all().all()
        assert vh["equity"].iloc[-1] == pytest.approx(
            vh["e11"].iloc[-1] + vh["e12"].iloc[-1])

    def test_save_all은_변형_행_오염을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        row = dict(cell="E11", sym="BTC", strategy="BRK24TP", bar_close=BASE + H1,
                   action="enter", price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                   direction=1, funding=0.0)
        with pytest.raises(ValueError, match="비공식 셀"):
            _save_all(st, [row], BASE, 1)
        assert not (tmp_path / LEDGER).exists(), "오염 기록 대신 예외 (기록 0)"

    def test_save_variant는_본_셀_행과_비변형_전략을_거부한다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = new_variant(st, t0=1)
        good = dict(cell="E11", sym="BTC", strategy="BRK24TP", bar_close=BASE + H1,
                    action="enter", price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                    direction=1, funding=0.0)
        with pytest.raises(ValueError, match="비변형 행"):
            _save_variant(st, v, [dict(good, cell="E01")], BASE, 1)
        with pytest.raises(ValueError, match="비변형 행"):
            _save_variant(st, v, [dict(good, strategy="BRK24")], BASE, 1)
        assert not (tmp_path / VLEDGER).exists()

    def test_회귀_본_원장_이력_상태는_변형_유무와_바이트_동일하다(
            self, tmp_path, monkeypatch):
        # 요건 #1 — 같은 합성 시퀀스에서 변형이 죽은 경로일 때의 replay 동일성.
        # 변형 활성 재생(실체결 발생) 후에도 본 셀 산출물이 바이트 단위로 같다.
        def scenario(root, with_variant: bool):
            monkeypatch.chdir(root)
            st = farm()
            warm(st, "BTC", atr=2.0)
            warm(st, "ETH", atr=1.0, n=0, closes=[99.0, 101.0] * 12)
            v = new_variant(st, t0=1) if with_variant else None
            seq = [({"BTC": bar(0, 100, 105, 100, 104),
                     "ETH": bar(0, 100, 103, 100, 103)}, None),
                   ({"BTC": bar(1, 104, 104.5, 103.5, 104),
                     "ETH": bar(1, 105, 105.5, 104, 104.5)},
                    {"BTC": 0.0001, "ETH": 0.0001}),
                   ({"BTC": bar(2, 89, 89.5, 88.5, 89),
                     "ETH": bar(2, 104, 104.8, 103.8, 104.2)}, None)]
            fills: list = []
            vfills: list = []
            for bars, fm in seq:
                fills += step(st, bars, fm)
                if v is not None:
                    vfills += step_variant(v, bars, fm)
            # 변형 활동 '이후' 본 셀 봉 1개 추가 — 잠복 지표 별칭 오염 검출
            fills += step(st, {"BTC": bar(3, 89, 90, 88.8, 89.5),
                               "ETH": bar(3, 104, 104.5, 103, 103.5)})
            _save_all(st, fills, BASE + 3 * H1, 4)
            if v is not None:
                st.variant_cells = variant_to_dict(v)
                _save_variant(st, v, vfills, BASE + 2 * H1, 3)
            # 영속 상태(변형 키 포함)를 재적재한 뒤의 본 재생도 동일해야 한다
            st2 = FarmState.from_dict(
                json.loads((root / "logs/tracke_state.json").read_text()))
            fills2 = step(st2, {"BTC": bar(4, 89.5, 90.5, 89, 90),
                                "ETH": bar(4, 103.5, 104, 102.5, 103)})
            _save_all(st2, fills2, BASE + 4 * H1, 1)
            raw = json.loads((root / "logs/tracke_state.json").read_text())
            raw.pop("variant_cells")
            return ((root / LEDGER).read_bytes(), (root / "logs/tracke_history.csv"
                                                   ).read_bytes(),
                    json.dumps(raw, sort_keys=True), len(vfills))
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        a_dir.mkdir(), b_dir.mkdir()
        led_a, hist_a, st_a, _ = scenario(a_dir, with_variant=False)
        led_b, hist_b, st_b, n_v = scenario(b_dir, with_variant=True)
        assert led_a == led_b, "본 원장 바이트 동일"
        assert hist_a == hist_b, "본 이력 바이트 동일"
        assert st_a == st_b, "본 상태(variant_cells 제외) 동일"
        assert n_v > 0, "변형이 실제 체결을 냈다 (공허한 비교 아님)"
        assert not (a_dir / VLEDGER).exists()
        vled = pd.read_csv(b_dir / VLEDGER)
        assert len(vled) and set(vled["cell"]) == {"E11"}

    def test_변형_실패는_본_커밋을_막지_않고_재실행이_따라잡는다(
            self, tmp_path, monkeypatch):
        # 요건 #3 — 장애 주입: 변형 상태 저장 직전 실패 → 본 파일 불변·마커 없음,
        # 다음 실행 재생은 변형 원장 유일키 멱등으로 중복 없이 따라잡는다.
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        v = new_variant(st, t0=1)
        v.last_ts = BASE - H1                        # 본 재생 구간과 정렬
        st.variant_cells = variant_to_dict(v)
        data = {"BTC": {BASE: (100.0, 105.0, 100.0, 104.0),
                        BASE + H1: (104.0, 104.5, 103.5, 104.0),
                        BASE + 2 * H1: (89.0, 89.5, 88.5, 89.0)}}
        fills: list = []
        for t in sorted(data["BTC"]):
            fills += step(st, {"BTC": BarE(t, *data["BTC"][t])})
        _save_all(st, fills, BASE + 2 * H1, 3)
        main_led = (tmp_path / LEDGER).read_bytes()
        main_state = (tmp_path / "logs/tracke_state.json").read_bytes()
        orig_write = runner._atomic_write

        def boom(path, text):
            raise OSError("디스크 장애 주입")
        monkeypatch.setattr(runner, "_atomic_write", boom)
        _safe_variant(_run_variant, None, st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)           # 예외가 전파되면 실패
        assert (tmp_path / LEDGER).read_bytes() == main_led, "본 원장 불변"
        assert (tmp_path / "logs/tracke_state.json").read_bytes() == main_state
        assert not (tmp_path / ERR_MARK).exists(), "본 팜 중단 마커 오염 금지"
        n1 = len(pd.read_csv(tmp_path / VLEDGER))
        assert n1 > 0, "변형 원장은 상태 저장 전에 append 됨"
        # 재실행 — 디스크 상태(뒤처진 variant_cells)에서 다시 시작
        monkeypatch.setattr(runner, "_atomic_write", orig_write)
        st2 = FarmState.from_dict(
            json.loads((tmp_path / "logs/tracke_state.json").read_text()))
        _safe_variant(_run_variant, None, st2, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)
        assert len(pd.read_csv(tmp_path / VLEDGER)) == n1, "유일키 멱등 — 중복 0"
        assert st2.variant_cells["last_ts"] == BASE + 2 * H1, "따라잡기 완료"
        assert st2.variant_cells["t0_variant"] == 1, "t0_variant 불변 (write-once)"

    def test_뒤처진_변형은_따라잡기_수집으로_같은_격자를_소화한다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE + 2 * H1
        v = new_variant(st, t0=1)
        v.last_ts = BASE - H1                        # 3봉 뒤처짐
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=1.0)   # 무거래 중립
        st.variant_cells = variant_to_dict(v)
        data = {"BTC": {BASE + 2 * H1: self.OHLC}}   # 본 실행분은 마지막 봉만
        # 수집 실패는 fail-closed (본 커밋과 무관하게 예외)
        monkeypatch.setattr(runner, "fetch_1h_paged", lambda *a: None)
        with pytest.raises(RuntimeError, match="따라잡기"):
            _run_variant(object(), st, data, {"BTC": {}}, BASE + 2 * H1,
                         BASE + 2 * H1)
        # 성공 수집 — [vsince, since) 를 채우면 변형이 본 격자 끝까지 정렬된다
        monkeypatch.setattr(
            runner, "fetch_1h_paged",
            lambda ex, coin, since, end: {t: self.OHLC
                                          for t in range(since, end, H1)})
        _run_variant(object(), st, data, {"BTC": {}}, BASE + 2 * H1,
                     BASE + 2 * H1)
        assert st.variant_cells["last_ts"] == BASE + 2 * H1

    def test_t0_variant는_최초_초기화에_동결되고_파일이_선생성된다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        _run_variant(None, st, {}, {}, BASE, BASE)   # 첫 호출 = 초기화만
        t0v = st.variant_cells["t0_variant"]
        assert t0v > 0
        assert st.variant_cells["last_ts"] == BASE, "본 팜과 정렬 (워밍업 무주문)"
        led = pd.read_csv(tmp_path / VLEDGER)
        assert list(led.columns) == LEDGER_COLS and len(led) == 0
        assert (tmp_path / VHIST).exists(), "git add pathspec 함정 방지 선생성"
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant_cells"]["t0_variant"] == t0v
        # 이후 재생 실행이 t0 를 절대 옮기지 않는다
        st.last_ts = BASE + H1
        _run_variant(None, st, {"BTC": {BASE + H1: self.OHLC}}, {"BTC": {}},
                     BASE + H1, BASE + H1)
        assert st.variant_cells["t0_variant"] == t0v
        assert st.variant_cells["last_ts"] == BASE + H1

    def test_종말_폐지_정리는_변형_포지션을_방치하지_않는다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE + H1
        v = new_variant(st, t0=1)
        warm(v, "XRP", pc=100.0)
        v.cells["E12"].positions["XRP"] = vpos(e=90.0, stop=1.0, tgt=180.0,
                                               risk_d=89.0)
        st.variant_cells = variant_to_dict(v)
        st.delisted.append("XRP")                    # 본 팜은 이미 폐지 완료
        _finalize_variant(st)
        vled = pd.read_csv(tmp_path / VLEDGER)
        row = vled.iloc[0]
        assert (row["cell"], row["action"]) == ("E12", "force_exit")
        assert row["price"] == pytest.approx(100.0), "변형 상태의 마지막 처리 종가"
        assert st.variant_cells["delisted"] == ["XRP"]
        assert not st.variant_cells["cells"]["E12"]["positions"]
        _finalize_variant(st)                        # 멱등 — 추가 기록 없음
        assert len(pd.read_csv(tmp_path / VLEDGER)) == len(vled)

    def test_미초기화_변형은_종말_정리를_만들지_않는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.delisted.append("XRP")
        _finalize_variant(st)                        # None → 아무것도 안 함
        assert st.variant_cells is None
        assert not (tmp_path / VLEDGER).exists()

    def test_포지션_없는_폐지_미러도_공석_표시를_저장한다(
            self, tmp_path, monkeypatch):
        # 변형에 해당 심볼 포지션이 없어도 영구 공석(delisted)은 상태에 남아야
        # 이후 재진입이 구조적으로 차단된다 (Codex 리뷰 반영)
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE + H1
        v = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v)
        st.delisted.append("XRP")
        _finalize_variant(st)
        assert st.variant_cells["delisted"] == ["XRP"], "공석 표시 저장"
        assert len(pd.read_csv(tmp_path / VLEDGER)) == 0, "체결은 없음"

    def test_save_variant는_t0_variant_변경을_거부한다(self, tmp_path, monkeypatch):
        # write-once 를 저장 경계에서도 강제 — 어떤 파일도 쓰기 전에 죽는다
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v)
        v.t0 = 2                                     # 오염된 t0
        with pytest.raises(ValueError, match="write-once"):
            _save_variant(st, v, [], BASE, 0)
        assert not (tmp_path / VLEDGER).exists()
        assert st.variant_cells["t0_variant"] == 1, "기존 기록 보존"

    def test_변형_t0의_비정상_타입은_fail_closed다(self):
        with pytest.raises(ValueError):
            variant_from_dict({"t0_variant": -5,
                               "cells": {"E11": {}, "E12": {}}})
        with pytest.raises(ValueError):
            variant_from_dict({"t0_variant": "1",
                               "cells": {"E11": {}, "E12": {}}})
        with pytest.raises(ValueError):
            variant_from_dict({"t0_variant": True,
                               "cells": {"E11": {}, "E12": {}}})

    def test_펀딩_범위_수집은_후진_페이지네이션으로_구간을_덮는다(self):
        calls = []

        class FakeEx:
            def publicGetV5MarketFundingHistory(self, params):
                calls.append(dict(params))
                end = int(params.get("endTime", 10**15))
                rows = [{"fundingRateTimestamp": str(t),
                         "fundingRate": "0.0001"}
                        for t in (30_000, 20_000, 10_000) if t <= end][:2]
                return {"retCode": "0", "result": {"list": rows}}
        ev = fetch_funding_range(FakeEx(), "BTC", need_from=10_000)
        assert set(ev) == {30_000, 20_000, 10_000}
        assert len(calls) == 2, "endTime 을 물려가며 후진"
        assert calls[1]["endTime"] == str(20_000 - 1)

        class DeadEx:
            def publicGetV5MarketFundingHistory(self, params):
                return {"retCode": "1"}
        assert fetch_funding_range(DeadEx(), "BTC", 0) is None, "fail-closed"


def warm2(v: FarmState, sym: str, atr: float = 2.0, hi: float = 100.0,
          lo: float = 90.0, n: int = 24, c2: list | None = None,
          v2: list | None = None, u14=None, d14=None, u2=None, d2=None) -> dict:
    """변형2 심볼 워밍업 주입 — BRK 지표(warm) + 확장 지표(x2)."""
    ind = warm(v, sym, atr=atr, hi=hi, lo=lo, n=n,
               pc=(c2[-1] if c2 else 100.0))
    x2 = _new_x2()
    x2["c2"] = list(c2) if c2 else []
    x2["v2"] = list(v2) if v2 else []
    x2["u14"], x2["d14"], x2["u2"], x2["d2"] = u14, d14, u2, d2
    ind["x2"] = x2
    return ind


# 게이트 3조건 전부 통과하는 확장 지표 (롱 기준): close 100 > SMA200 99.005,
# RSI14 = 100-100/(1+1/0.5) ≈ 66.7 > 50, vol 20 > mean(직전 20봉) 10
GATE_PASS = dict(c2=[99.0] * 199 + [100.0], v2=[10.0] * 20 + [20.0],
                 u14=1.0, d14=0.5)


class TestV2Gate:
    """E13·E14 (BRK24GATE) — 진입·스탑·사이징·청산 BRK24 동일 + 3중 게이트 AND."""

    def _v(self, **over) -> FarmState:
        v = new_variant2(farm(), t0=1)
        kw = dict(GATE_PASS)
        kw.update(over)
        warm2(v, "BTC", atr=2.0, **kw)
        return v

    def test_게이트_전부_통과면_진입은_BRK24와_완전_동일하다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        v = self._v()
        b = bar(0, 100, 105, 100, 104)
        fills = step(st, {"BTC": b})
        vfills = step_variant2(v, {"BTC": b})
        p = st.cells["E01"].positions["BTC"]
        q = v.cells["E13"].positions["BTC"]
        assert (q.e, q.stop, q.u, q.risk_d) == (p.e, p.stop, p.u, p.risk_d)
        assert q.kind == "BRK" and q.tgt == 0.0, "목표 없음 — 순수 BRK24 경로"
        e13 = next(f for f in vfills if f["cell"] == "E13" and f["action"] == "enter")
        e01 = next(f for f in fills if f["cell"] == "E01" and f["action"] == "enter")
        assert (e13["price"], e13["qty"], e13["cost"]) == \
            (e01["price"], e01["qty"], e01["cost"])
        assert e13["strategy"] == "BRK24GATE"

    def test_게이트1_추세_역방향이면_차단된다(self):
        v = self._v(c2=[101.0] * 199 + [100.0])          # close[i-1] < SMA200[i-1]
        fills = step_variant2(v, {"BTC": bar(0, 100, 105, 100, 104)})
        assert "BTC" not in v.cells["E13"].positions
        assert not any(f["cell"] == "E13" for f in fills)

    def test_게이트2_RSI_50이하면_차단된다(self):
        v = self._v(u14=0.5, d14=1.0)                    # RSI ≈ 33.3
        step_variant2(v, {"BTC": bar(0, 100, 105, 100, 104)})
        assert "BTC" not in v.cells["E13"].positions

    def test_게이트3_거래량_평균이하면_차단된다(self):
        v = self._v(v2=[10.0] * 21)                      # 10 > 10 은 False (엄격 부등호)
        step_variant2(v, {"BTC": bar(0, 100, 105, 100, 104)})
        assert "BTC" not in v.cells["E13"].positions

    def test_게이트_NaN_미형성은_전부_차단이다(self):
        cases = (dict(c2=[99.0] * 198 + [100.0]),        # SMA200 미형성 (199 < 200)
                 dict(v2=[10.0] * 20),                   # 거래량 창 미형성 (20 < 21)
                 dict(u14=None, d14=None),               # RSI 미형성
                 dict(v2=[10.0] * 19 + [float("nan"), 20.0]),  # NaN 거래량 창
                 dict(u14=0.0, d14=0.0))                 # 0/0 = NaN RSI
        for over in cases:
            v = self._v(**over)
            step_variant2(v, {"BTC": bar(0, 100, 105, 100, 104)})
            assert "BTC" not in v.cells["E13"].positions, f"차단 실패: {over}"

    def test_숏_게이트는_대칭이다(self):
        v = self._v(c2=[101.0] * 199 + [100.0], u14=0.5, d14=1.0)
        step_variant2(v, {"BTC": bar(0, 89, 89.5, 80, 82)})   # 하향 돌파 (lo=90)
        p = v.cells["E13"].positions["BTC"]
        assert p.d == -1
        assert p.e == pytest.approx(89.0)                # min(시가, 채널 90)
        assert p.stop == pytest.approx(101.0)            # 89 + 6*2 — BRK24 동일

    def test_RSI_dn0_상승만_100은_롱_통과다(self):
        # pandas ru/0 = inf → RSI 100 동치 (confluence_gate_test rsi_wilder)
        v = self._v(u14=1.0, d14=0.0)
        step_variant2(v, {"BTC": bar(0, 100, 105, 100, 104)})
        assert "BTC" in v.cells["E13"].positions

    def test_이중_돌파봉은_롱_해석_후_게이트라_역방향_폴백이_없다(self):
        # 숏 게이트는 통과 가능한 상태지만 롱 우선 해석 → 롱 게이트 차단 → 무행동
        v = self._v(c2=[101.0] * 199 + [100.0], u14=0.5, d14=1.0)
        fills = step_variant2(v, {"BTC": bar(0, 95, 105, 80, 85)})  # 상하 동시 돌파
        assert "BTC" not in v.cells["E13"].positions
        assert not any(f["cell"] == "E13" for f in fills)

    def test_게이트는_E01_BRK24에_영향이_없다(self):
        # 같은 봉에서 게이트 차단이어도 본 BRK24(E01)는 정상 진입 (경로 분리)
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 105, 100, 104)})
        assert "BTC" in st.cells["E01"].positions


class TestV2BB:
    """E15·E16 (BBMR) — BB(20,2σ,ddof=0) 종가 이탈 롱, SMA20 청산, 스탑 없음."""

    def _v(self, c2: list) -> FarmState:
        v = new_variant2(farm(), t0=1)
        warm2(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, c2=c2, v2=[10.0] * 21)
        return v

    def test_확정봉_종가가_하단밴드_아래면_다음_봉_시가_롱이다(self):
        v = self._v([100.0] * 19)
        step_variant2(v, {"BTC": bar(0, 100, 100, 90, 90)})   # 90 < 하단 ≈ 95.14
        cell = v.cells["E15"]
        assert "BTC" not in cell.positions, "신호봉 종가 체결 금지"
        assert cell.pending["BTC"] == {"kind": "BBMR", "d": 1, "ets": BASE + H1}
        step_variant2(v, {"BTC": bar(1, 91, 92, 90.5, 91)})
        p = cell.positions["BTC"]
        assert p.e == pytest.approx(91.0), "다음 봉 시가 체결 (U1 실행 규약)"
        assert p.d == 1 and p.kind == "BBMR"
        assert p.stop == 0.0 and p.tgt == 0.0, "스탑·목표 없음 (출판 충실)"

    def test_밴드는_봉내_터치가_아니라_종가_기준이다(self):
        v = self._v([100.0] * 19)
        step_variant2(v, {"BTC": bar(0, 100, 100, 90, 100)})  # 저가만 관통, 종가 복귀
        assert "BTC" not in v.cells["E15"].pending

    def test_상단_이탈에도_숏은_없다_롱온리(self):
        v = self._v([100.0] * 19)
        step_variant2(v, {"BTC": bar(0, 100, 130, 100, 125)})
        cell = v.cells["E15"]
        assert not cell.pending and not cell.positions

    def test_청산은_확정봉_종가_SMA20_이상_다음_봉_시가다(self):
        v = self._v([100.0] * 19)
        cell = v.cells["E15"]
        cell.positions["BTC"] = FarmPos(d=1, u=30.0, e=95.0, stop=0.0,
                                        kind="BBMR", risk_d=95.0 * V2_HEAT_FRAC)
        step_variant2(v, {"BTC": bar(0, 99, 100.5, 99, 100)})     # 100 >= SMA20(100)
        assert cell.positions["BTC"].pending_exit == "signal"
        fills = step_variant2(v, {"BTC": bar(1, 100.4, 101, 100, 100.8)})
        assert fills[0]["action"] == "exit_signal"
        assert fills[0]["price"] == pytest.approx(100.4), "다음 봉 시가"
        assert "BTC" not in cell.positions

    def test_체결봉_종가가_이미_SMA20_이상이면_같은_봉에_청산신호가_선다(self):
        # lab run_bollinger 1:1 — 체결봉 종가도 확정봉 (청산 체결은 다음 봉 시가)
        v = self._v([100.0] * 19)
        cell = v.cells["E15"]
        cell.pending["BTC"] = {"kind": "BBMR", "d": 1, "ets": BASE}
        step_variant2(v, {"BTC": bar(0, 95, 101, 94, 100.5)})
        p = cell.positions["BTC"]
        assert p.e == pytest.approx(95.0)
        assert p.pending_exit == "signal", "체결봉 종가 >= SMA20 — 같은 봉 신호"
        fills = step_variant2(v, {"BTC": bar(1, 100.2, 100.6, 99.9, 100.1)})
        assert fills[0]["action"] == "exit_signal"
        assert fills[0]["price"] == pytest.approx(100.2)

    def test_스탑이_없어_폭락에도_봉내_청산이_없다(self):
        v = self._v([100.0] * 19)
        cell = v.cells["E15"]
        cell.positions["BTC"] = FarmPos(d=1, u=30.0, e=100.0, stop=0.0,
                                        kind="BBMR", risk_d=5.0)
        fills = step_variant2(v, {"BTC": bar(0, 60, 61, 50, 55)})
        assert fills == []
        assert "BTC" in cell.positions, "스탑 없음 — 신호 청산만 존재"
        assert not cell.positions["BTC"].pending_exit


class TestV2RSI2:
    """E17·E18 (Connors RSI2) — 5/95 원전 임계, SMA200 레짐, SMA5 청산, 숏 대칭."""

    def _v(self, c2: list, u2=None, d2=None) -> FarmState:
        v = new_variant2(farm(), t0=1)
        warm2(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, c2=c2, v2=[10.0] * 21,
              u2=u2, d2=d2)
        return v

    def test_RSI2_5_정확히면_미진입_5미만이면_롱이다(self):
        # diff=0 → u=u2/2, dn=d2/2 → RSI = 100-100/(1+u2/d2). u2/d2=1/19 → 정확히 5
        up = [0.9] * 198 + [1.0]                     # close 1.0 > SMA200 ≈ 0.901
        v = self._v(up, u2=1.0, d2=19.0)
        step_variant2(v, {"BTC": bar(0, 1.0, 1.0, 1.0, 1.0)})
        assert "BTC" not in v.cells["E17"].pending, "경계값 5는 미진입 (엄격 <)"
        v = self._v(up, u2=1.0, d2=21.0)             # RSI ≈ 4.55 < 5
        step_variant2(v, {"BTC": bar(0, 1.0, 1.0, 1.0, 1.0)})
        assert v.cells["E17"].pending["BTC"] == \
            {"kind": "RSI2", "d": 1, "ets": BASE + H1}

    def test_RSI2_95_정확히면_미진입_95초과면_숏이다(self):
        dn = [1.1] * 198 + [1.0]                     # close 1.0 < SMA200 ≈ 1.099
        v = self._v(dn, u2=19.0, d2=1.0)             # RSI 정확히 95
        step_variant2(v, {"BTC": bar(0, 1.0, 1.0, 1.0, 1.0)})
        assert "BTC" not in v.cells["E17"].pending, "경계값 95는 미진입 (엄격 >)"
        v = self._v(dn, u2=21.0, d2=1.0)             # RSI ≈ 95.45 > 95
        step_variant2(v, {"BTC": bar(0, 1.0, 1.0, 1.0, 1.0)})
        assert v.cells["E17"].pending["BTC"] == \
            {"kind": "RSI2", "d": -1, "ets": BASE + H1}

    def test_SMA200_레짐_필터가_방향을_막는다(self):
        # RSI2 < 5 여도 close < SMA200 이면 롱 금지 (숏 조건도 아님 → 무신호)
        v = self._v([1.1] * 198 + [1.0], u2=1.0, d2=21.0)
        step_variant2(v, {"BTC": bar(0, 1.0, 1.0, 1.0, 1.0)})
        assert "BTC" not in v.cells["E17"].pending

    def test_롱_청산은_종가_SMA5_초과_다음_봉_시가다(self):
        v = self._v([100.0] * 4)                     # SMA200 미형성 — 신호 경로 무관
        cell = v.cells["E17"]
        cell.positions["BTC"] = FarmPos(d=1, u=33.0, e=100.0, stop=0.0,
                                        kind="RSI2", risk_d=5.0)
        step_variant2(v, {"BTC": bar(0, 100.5, 101.2, 100.4, 101.0)})  # 101 > 100.2
        assert cell.positions["BTC"].pending_exit == "signal"
        fills = step_variant2(v, {"BTC": bar(1, 101.3, 101.5, 101, 101.2)})
        assert fills[0]["action"] == "exit_signal"
        assert fills[0]["price"] == pytest.approx(101.3)

    def test_숏_청산은_종가_SMA5_미만이다(self):
        v = self._v([100.0] * 4)
        cell = v.cells["E17"]
        cell.positions["BTC"] = FarmPos(d=-1, u=33.0, e=100.0, stop=0.0,
                                        kind="RSI2", risk_d=5.0)
        step_variant2(v, {"BTC": bar(0, 99.5, 99.6, 98.8, 99.0)})   # 99 < 99.8
        assert cell.positions["BTC"].pending_exit == "signal"

    def test_종가가_SMA5와_같으면_보유_유지다(self):
        v = self._v([100.0] * 4)
        cell = v.cells["E17"]
        cell.positions["BTC"] = FarmPos(d=1, u=33.0, e=100.0, stop=0.0,
                                        kind="RSI2", risk_d=5.0)
        step_variant2(v, {"BTC": bar(0, 100, 100.3, 99.8, 100.0)})  # == SMA5
        assert not cell.positions["BTC"].pending_exit

    def test_숏도_스탑이_없어_same_bar_stop_오검이_없다(self):
        # stop=0.0 센티널이 숏 same_bar_stop(b.high >= 0)으로 오검되면 즉시 청산된다
        v = self._v([90.0] * 4)
        cell = v.cells["E17"]
        cell.pending["BTC"] = {"kind": "RSI2", "d": -1, "ets": BASE}
        fills = step_variant2(v, {"BTC": bar(0, 100, 150, 95, 96)})
        p = cell.positions["BTC"]
        assert p.d == -1 and p.stop == 0.0
        assert not any(f["action"].startswith("same_bar") for f in fills)


class TestV2SizingHeat:
    """스탑 없는 셀 사이징·리스크 — 명목 equity/3, heat 기여 = 명목 × 5%."""

    def _pending(self, v: FarmState, sym: str) -> None:
        warm2(v, sym, atr=2.0, hi=1000.0, lo=0.001, c2=[200.0] * 19,
              v2=[10.0] * 21)
        v.cells["E15"].pending[sym] = {"kind": "BBMR", "d": 1, "ets": BASE}

    def test_명목_사이징은_equity_3분의1이다(self):
        v = new_variant2(farm(), t0=1)
        self._pending(v, "BTC")
        step_variant2(v, {"BTC": bar(0, 100, 101, 99, 100)})
        p = v.cells["E15"].positions["BTC"]
        assert p.u == pytest.approx(V2_NOTIONAL_FRAC * 10_000 / 100.0)  # 33.33
        assert p.u * p.e == pytest.approx(10_000 / 3.0), "명목 = equity × 1/3"
        assert v.cells["E15"].cost == pytest.approx(p.u * 100.0 * COST_SIDE)

    def test_스탑없는_포지션의_heat기여는_명목x일손실한도다(self):
        # 동결 정의: risk_d = fill × V2_HEAT_FRAC → 기여 = u·risk_d = 명목 × 5%
        v = new_variant2(farm(), t0=1)
        self._pending(v, "BTC")
        step_variant2(v, {"BTC": bar(0, 100, 101, 99, 100)})
        p = v.cells["E15"].positions["BTC"]
        assert p.risk_d == pytest.approx(100.0 * V2_HEAT_FRAC)          # = 5.0
        assert p.u * p.risk_d == pytest.approx(10_000 / 3.0 * 0.05)     # ≈ 166.67

    def test_3슬롯_만재는_heat캡_6퍼센트_이내다(self):
        # 슬롯당 기여 ≈ 1.667% × 3 = 5% ≤ 6% — 설계 슬롯은 구조적으로 허용
        v = new_variant2(farm(), t0=1)
        bars = {}
        for s in ("BTC", "ETH", "SOL"):
            self._pending(v, s)
            bars[s] = bar(0, 100, 101, 99, 100)
        step_variant2(v, bars)
        cell = v.cells["E15"]
        assert set(cell.positions) == {"BTC", "ETH", "SOL"}
        heat = sum(p.u * p.risk_d for p in cell.positions.values())
        assert heat <= 0.06 * cell.equity * (1 + 1e-9)

    def test_드로다운_후_잔존_heat가_캡을_넘기면_신규진입_차단이다(self):
        v = new_variant2(farm(), t0=1)
        self._pending(v, "BTC")
        cell = v.cells["E15"]
        cell.equity = 5_200.0                        # 캡 = 312.0
        for s in ("ETH", "SOL"):                     # 잔존 heat 333.33 (과거 고자본 진입)
            cell.positions[s] = FarmPos(d=1, u=33.3333, e=100.0, stop=0.0,
                                        kind="BBMR", risk_d=5.0)
        fills = step_variant2(v, {"BTC": bar(0, 100, 101, 99, 100)})
        assert "BTC" not in cell.positions, "333.33 + 86.67 > 312 — heat 캡 차단"
        assert not any(f["action"] == "enter" for f in fills)

    def test_gross는_스탑없는_명목도_통상_산입한다(self):
        v = new_variant2(farm(), t0=1)
        warm2(v, "BTC", atr=2.0, c2=[110.0] * 19, v2=[10.0] * 21)
        cell = v.cells["E15"]
        cell.positions["BTC"] = FarmPos(d=1, u=30.0, e=100.0, stop=0.0,
                                        kind="BBMR", risk_d=5.0)
        g = v.to_dict()["cells"]["E15"]["gross"]
        assert g == pytest.approx(30.0 * 110.0 / cell.equity), "|u|×마크/equity"


class TestV2State:
    """변형2 서브상태 — variant2_cells 키, t0_variant2 write-once, 그룹 분리."""

    def test_new_variant2는_x2_초기화와_깊은_분리다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        v = new_variant2(st, t0=123)
        assert v.t0 == 123
        assert v.last_ts == st.last_ts, "t0_variant2 이전 봉 재생 구조적 차단"
        assert set(v.cells) == {"E13", "E14", "E15", "E16", "E17", "E18"}
        assert all(c.equity == 10_000.0 for c in v.cells.values()), "셀당 신규 $10,000"
        assert v.ind["BTC"]["x2"] == _new_x2(), "확장 지표는 빈 상태로 시작"
        assert "x2" not in st.ind["BTC"], "본 팜 ind 에 x2 생성 금지"
        v.ind["BTC"]["atr1"] = 999.0
        assert st.ind["BTC"]["atr1"] != 999.0, "깊은 복사 — 본 팜 무영향"

    def test_E11_직렬화는_x2와_variant2_variant3_키가_없이_불변이다(self):
        # E11/E12 재생 결과 바이트 동일성 — 서브상태 dict 에 새 키가 새면 안 된다
        st = farm()
        warm(st, "BTC", atr=2.0)
        v1 = new_variant(st, t0=1)
        d = variant_to_dict(v1)
        assert "variant2_cells" not in d and "t0_variant2" not in d
        assert "variant3_cells" not in d and "t0_variant3" not in d
        assert all("x2" not in i for i in d["ind"].values())

    def test_변형2_직렬화는_t0_variant2_명명으로_왕복된다(self):
        v = new_variant2(farm(), t0=BASE)
        warm2(v, "BTC", atr=2.0, c2=GATE_PASS["c2"], v2=GATE_PASS["v2"],
              u14=1.0, d14=0.5)
        step_variant2(v, {"BTC": bar(0, 100, 105, 100, 104)})   # E13 포지션 포함
        d = variant2_to_dict(v)
        assert "t0_variant2" in d and "t0" not in d
        assert "variant_cells" not in d and "variant2_cells" not in d
        v2 = variant2_from_dict(json.loads(json.dumps(d, default=float)))
        assert json.dumps(variant2_to_dict(v2), sort_keys=True, default=float) == \
            json.dumps(d, sort_keys=True, default=float)
        assert v2.cells["E13"].positions["BTC"].kind == "BRK"
        assert v2.ind["BTC"]["x2"]["c2"][-1] == 104.0, "x2 이력 왕복 보존"

    def test_본_상태는_세_변형_키를_불투명하게_왕복_보존한다(self):
        st = farm()
        assert st.variant2_cells is None, "미초기화 기본값 None"
        assert st.variant3_cells is None, "미초기화 기본값 None"
        st.variant_cells = variant_to_dict(new_variant(st, t0=1))
        st.variant2_cells = variant2_to_dict(new_variant2(st, t0=2))
        st.variant3_cells = variant3_to_dict(new_variant3(st, ["AAA", "BBB"], t0=3))
        rt = FarmState.from_dict(json.loads(json.dumps(st.to_dict(), default=float)))
        assert rt.variant_cells == st.variant_cells
        assert rt.variant2_cells == st.variant2_cells
        assert rt.variant3_cells == st.variant3_cells

    def test_손상된_변형2_상태는_fail_closed다(self):
        with pytest.raises(ValueError):
            variant2_from_dict(None)
        with pytest.raises(ValueError):
            variant2_from_dict({})
        with pytest.raises(ValueError):
            variant2_from_dict({"t0_variant2": 0, "cells": {}})
        with pytest.raises(ValueError):                  # E11 구성은 변형2가 아니다
            variant2_from_dict({"t0_variant": 5,
                                "cells": {"E11": {}, "E12": {}}})
        with pytest.raises(ValueError):                  # 셀 부분 결손
            variant2_from_dict({"t0_variant2": 5,
                                "cells": {"E13": {"equity": 10_000.0}}})

    def test_변형2_시가평가와_폐지는_그룹_전용이다(self):
        v = new_variant2(farm(), t0=1)
        warm2(v, "BTC", atr=2.0, c2=[110.0] * 19, v2=[10.0] * 21)
        v.cells["E15"].positions["BTC"] = FarmPos(d=1, u=10.0, e=100.0, stop=0.0,
                                                  kind="BBMR", risk_d=5.0)
        eqs = variant2_equities(v)
        assert set(eqs) == {"E13", "E14", "E15", "E16", "E17", "E18"}
        assert eqs["E15"] == pytest.approx(10_000 + 10.0 * 10.0), "마지막 종가 마크"
        fills = variant2_delist(v, "BTC")
        assert fills[0]["cell"] == "E15" and fills[0]["action"] == "force_exit"
        assert "BTC" in v.delisted
        assert not v.cells["E15"].positions

    def test_변형2는_t0_이전_워밍업_무주문이다(self):
        st = farm()
        v = new_variant2(st, t0=BASE + 100 * H1)
        warm2(v, "BTC", atr=2.0, c2=GATE_PASS["c2"], v2=GATE_PASS["v2"],
              u14=1.0, d14=0.5)
        fills = step_variant2(v, {"BTC": bar(0, 100, 115, 100, 114)})
        assert fills == []
        for c in v.cells.values():
            assert not c.positions and not c.pending

    def test_변형2_동결_상수와_라벨(self):
        assert [s.cell for s in V2CELLS] == ["E13", "E14", "E15", "E16", "E17", "E18"]
        assert [s.strategy for s in V2CELLS] == \
            ["BRK24GATE", "BRK24GATE", "BBMR", "BBMR", "RSI2", "RSI2"]
        assert [s.basket for s in V2CELLS] == ["A", "B", "A", "B", "A", "B"]
        assert V2_NOTIONAL_FRAC == pytest.approx(1.0 / 3.0)
        assert V2_HEAT_FRAC == pytest.approx(0.05)
        assert GATE_SMA_N == RSI2_TREND_N == 200 and GATE_VOL_N == 20
        assert BB_N == 20 and RSI2_EXIT_N == 5
        assert V2LABELS["E13"] == V2LABELS["E14"] == \
            "컨플루언스 게이트 변형 · 미검증 · 판정 권한 없음"
        assert V2LABELS["E15"] == V2LABELS["E16"] == \
            "볼린저 평균회귀 (출판) · 미검증 · 판정 권한 없음"
        assert V2LABELS["E17"] == V2LABELS["E18"] == \
            "Connors RSI2 (출판) · 미검증 · 판정 권한 없음"

    def test_warmup_x2는_NaN_종가를_건너뛰고_이력을_캡한다(self):
        v = new_variant2(farm(), t0=1)
        rows = [(100.0 + k * 0.1, 10.0) for k in range(300)]
        rows.insert(150, (float("nan"), 5.0))            # 결측 봉 — 건너뜀
        warmup_x2(v, "BTC", rows)
        x2 = v.ind["BTC"]["x2"]
        assert len(x2["c2"]) == 200 and len(x2["v2"]) == GATE_VOL_N + 1
        assert x2["c2"][-1] == pytest.approx(100.0 + 299 * 0.1)
        assert x2["u14"] is not None and x2["u2"] is not None


class TestV2Runner:
    """E13~E18 러너 — 워밍업 후 t0 동결, 통합 원장/이력, 방화벽, 격리, 회귀."""

    OHLC5 = (100.0, 101.0, 99.0, 100.0, 10.0)

    def test_첫_호출은_워밍업_수집_후_t0를_동결하고_파일을_선생성한다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        calls = []

        def fake_fetch(ex, coin, since, now_h):
            calls.append(coin)
            return {t: self.OHLC5 for t in range(since, now_h, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        _run_variant2(None, st, {}, {}, BASE, BASE)
        assert set(calls) == {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"}, \
            "바스켓 A+B 전 심볼 워밍업 수집"
        t0v = st.variant2_cells["t0_variant2"]
        assert t0v > 0
        assert st.variant2_cells["last_ts"] == BASE, "본 팜과 정렬 (워밍업 무주문)"
        assert set(st.variant2_cells["cells"]) == \
            {"E13", "E14", "E15", "E16", "E17", "E18"}
        x2 = st.variant2_cells["ind"]["BTC"]["x2"]
        assert len(x2["c2"]) == 200, "워밍업이 확장 지표를 채운다"
        led = pd.read_csv(tmp_path / VLEDGER)
        assert list(led.columns) == LEDGER_COLS and len(led) == 0
        vh = pd.read_csv(tmp_path / VHIST)
        assert list(vh.columns) == VHIST_COLS
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant2_cells"]["t0_variant2"] == t0v
        # 이후 재생 실행이 t0 를 절대 옮기지 않는다
        st.last_ts = BASE + H1
        _run_variant2(None, st, {"BTC": {BASE + H1: self.OHLC5}}, {"BTC": {}},
                      BASE + H1, BASE + H1)
        assert st.variant2_cells["t0_variant2"] == t0v
        assert st.variant2_cells["last_ts"] == BASE + H1

    def test_워밍업_수집_실패는_t0를_동결하지_않는다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        monkeypatch.setattr(runner, "fetch_1h_paged", lambda *a: None)
        with pytest.raises(RuntimeError, match="워밍업"):
            _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is None, "동결 지연 — 다음 실행 재시도"
        assert not (tmp_path / VLEDGER).exists()

    def test_빈_또는_last_ts_미달_워밍업은_t0를_동결하지_않는다(
            self, tmp_path, monkeypatch):
        # Codex 재현: 빈/부분 응답으로 식은 지표 채 t0 가 영구 동결되는 사고 방지
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        monkeypatch.setattr(runner, "fetch_1h_paged", lambda *a: {})
        with pytest.raises(RuntimeError, match="불완전"):
            _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is None
        monkeypatch.setattr(                            # 마지막 봉(last_ts) 결손
            runner, "fetch_1h_paged",
            lambda ex, coin, since, now_h: {t: self.OHLC5
                                            for t in range(since, now_h - H1, H1)})
        with pytest.raises(RuntimeError, match="불완전"):
            _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is None

    def test_바스켓A_워밍업_깊이_부족은_t0를_동결하지_않는다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        monkeypatch.setattr(                            # 짧은 응답 (100봉만)
            runner, "fetch_1h_paged",
            lambda ex, coin, since, now_h: {t: self.OHLC5
                                            for t in range(now_h - 100 * H1,
                                                           now_h, H1)})
        with pytest.raises(RuntimeError, match="깊이 부족"):
            _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is None

    def test_바스켓B_늦은_상장은_짧은_연속_이력으로_동결된다(
            self, tmp_path, monkeypatch):
        # 바스켓 B(XRP 등)는 늦은 상장 가능 — 연속 이력 전체가 last_ts 로 끝나면 허용
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE

        def fake_fetch(ex, coin, since, now_h):
            if coin == "XRP":                           # 진짜 늦은 상장 — 50봉 연속
                return {t: self.OHLC5
                        for t in range(now_h - 50 * H1, now_h, H1)}
            return {t: self.OHLC5 for t in range(since, now_h, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is not None
        assert len(st.variant2_cells["ind"]["XRP"]["x2"]["c2"]) == 50
        assert len(st.variant2_cells["ind"]["BTC"]["x2"]["c2"]) == 200

    def test_바스켓B_내부_갭은_늦은_상장이_아니라_재시도다(
            self, tmp_path, monkeypatch):
        # Codex 검토 반영: 꼬리 밖 관측 봉 존재 = 내부 갭 — 일시 수집 결손일 수
        # 있으므로 갭 앞 관측을 버리고 동결하는 대신 fail-closed 재시도
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE

        def fake_fetch(ex, coin, since, now_h):
            if coin == "XRP":                           # 꼬리 50봉 + 갭 너머 고아 봉
                d = {t: self.OHLC5
                     for t in range(now_h - 50 * H1, now_h, H1)}
                d[now_h - 60 * H1] = self.OHLC5
                return d
            return {t: self.OHLC5 for t in range(since, now_h, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        with pytest.raises(RuntimeError, match="내부 갭"):
            _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is None, "동결 지연 — 다음 실행 재시도"

    def test_E11_상태쓰기_실패는_변형2_저장으로_새지_않는다(
            self, tmp_path, monkeypatch):
        # Codex 재현 반영: 그룹1 상태 쓰기 실패 시 메모리 변이를 롤백해,
        # 뒤따르는 그룹2 저장이 미커밋 E11 변이를 대신 영속화하지 못한다
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        _atomic_write(runner.STATE,
                      json.dumps(st.to_dict(), indent=1, default=float))
        before = dict(st.variant_cells)
        v1.last_ts = BASE + 5 * H1                       # 미커밋 변이 시뮬레이션
        orig_write = runner._atomic_write

        def boom(path, text):
            raise OSError("디스크 장애 주입")
        monkeypatch.setattr(runner, "_atomic_write", boom)
        with pytest.raises(OSError):
            _save_variant(st, v1, [], BASE + 5 * H1, 0)
        assert st.variant_cells == before, "실패한 저장의 메모리 변이 롤백"
        monkeypatch.setattr(runner, "_atomic_write", orig_write)
        _save_variant2(st, v2, [], BASE, 0)              # 그룹2 정상 저장
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant_cells"]["last_ts"] == before["last_ts"], \
            "그룹2 저장이 그룹1의 실패 변이를 영속화하지 않는다"

    def test_구_스키마_이력은_초기화시_열_순서까지_이월된다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        old_cols = ["day", "ts", "equity", "e11", "e12", "n_pos", "bars", "fills"]
        (tmp_path / "logs").mkdir()
        pd.DataFrame([{"day": "2026-01-01", "ts": BASE, "equity": 20_500.0,
                       "e11": 10_250.0, "e12": 10_250.0, "n_pos": 1,
                       "bars": 3, "fills": 2}])[old_cols].to_csv(
            tmp_path / VHIST, index=False)
        st = farm()
        st.last_ts = BASE
        monkeypatch.setattr(
            runner, "fetch_1h_paged",
            lambda ex, coin, since, now_h: {t: self.OHLC5
                                            for t in range(since, now_h, H1)})
        _run_variant2(None, st, {}, {}, BASE, BASE)
        vh = pd.read_csv(tmp_path / VHIST)
        assert list(vh.columns) == VHIST_COLS, "선언 스키마 열 순서로 정렬"
        assert vh["e11"].iloc[0] == pytest.approx(10_250.0), "기존 행 값 불변"
        assert (vh[["e13", "e14", "e15", "e16", "e17", "e18"]].iloc[0]
                == 0.0).all(), "구 행 신설 열 = 0.0 (부재 표기)"

    def test_본팜_미가동이면_초기화하지_않는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()                                      # last_ts = 0
        _run_variant2(None, st, {}, {}, BASE, BASE)
        assert st.variant2_cells is None

    def test_통합_이력_행은_두_그룹_수치를_모두_담는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        _save_variant(st, v1, [], BASE, 0)               # 그룹1 저장 (같은 ts)
        _save_variant2(st, v2, [], BASE, 0)              # 그룹2 저장 — keep-last
        vh = pd.read_csv(tmp_path / VHIST)
        assert len(vh) == 1, "같은 ts keep-last — 마지막(그룹2) 행"
        row = vh.iloc[0]
        for c in ("e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18"):
            assert row[c] == pytest.approx(10_000.0)
        assert row["equity"] == pytest.approx(80_000.0), "전 변형 셀 합"

    def test_변형2_방화벽은_비변형2_행을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = new_variant2(st, t0=1)
        st.variant2_cells = variant2_to_dict(v)
        good = dict(cell="E15", sym="BTC", strategy="BBMR", bar_close=BASE + H1,
                    action="enter", price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                    direction=1, funding=0.0)
        for bad in (dict(good, cell="E01"), dict(good, cell="E11"),
                    dict(good, strategy="BRK24TP"),
                    dict(good, cell="E13")):     # 유효 셀×유효 전략 교차 오염도 거부
            with pytest.raises(ValueError, match="비변형2"):
                _save_variant2(st, v, [bad], BASE, 1)
        assert not (tmp_path / VLEDGER).exists()

    def test_본_원장은_변형2_행을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        row = dict(cell="E13", sym="BTC", strategy="BRK24GATE", bar_close=BASE + H1,
                   action="enter", price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                   direction=1, funding=0.0)
        with pytest.raises(ValueError, match="비공식 셀"):
            _save_all(farm(), [row], BASE, 1)
        assert not (tmp_path / LEDGER).exists()

    def test_save_variant2는_t0_변경을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = new_variant2(st, t0=1)
        st.variant2_cells = variant2_to_dict(v)
        v.t0 = 2
        with pytest.raises(ValueError, match="write-once"):
            _save_variant2(st, v, [], BASE, 0)
        assert not (tmp_path / VLEDGER).exists()
        assert st.variant2_cells["t0_variant2"] == 1, "기존 기록 보존"

    def test_변형2_실패는_본_커밋과_E11을_막지_않고_재실행이_따라잡는다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        v1 = new_variant(st, t0=1)
        v1.last_ts = BASE - H1
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        v2.last_ts = BASE - H1
        warm2(v2, "BTC", atr=2.0, c2=GATE_PASS["c2"], v2=GATE_PASS["v2"],
              u14=1.0, d14=0.5)
        st.variant2_cells = variant2_to_dict(v2)
        data = {"BTC": {BASE: (100.0, 105.0, 100.0, 104.0, 20.0),
                        BASE + H1: (104.0, 104.5, 103.5, 104.0, 10.0),
                        BASE + 2 * H1: (89.0, 89.5, 88.5, 89.0, 10.0)}}
        fills: list = []
        for t in sorted(data["BTC"]):
            fills += step(st, {"BTC": BarE(t, *data["BTC"][t])})
        _save_all(st, fills, BASE + 2 * H1, 3)
        _safe_variant(_run_variant, None, st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)               # 그룹1 정상 커밋
        main_led = (tmp_path / LEDGER).read_bytes()
        main_state = (tmp_path / "logs/tracke_state.json").read_bytes()
        vled_1 = pd.read_csv(tmp_path / VLEDGER)
        orig_write = runner._atomic_write

        def boom(path, text):
            raise OSError("디스크 장애 주입")
        monkeypatch.setattr(runner, "_atomic_write", boom)
        _safe_variant(_run_variant2, None, st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)               # 예외가 전파되면 실패
        assert (tmp_path / LEDGER).read_bytes() == main_led, "본 원장 불변"
        assert (tmp_path / "logs/tracke_state.json").read_bytes() == main_state
        assert not (tmp_path / ERR_MARK).exists(), "본 팜 중단 마커 오염 금지"
        n1 = len(pd.read_csv(tmp_path / VLEDGER))
        assert n1 > len(vled_1), "변형2 원장은 상태 저장 전에 append 됨"
        # 재실행 — 디스크 상태(뒤처진 variant2_cells)에서 유일키 멱등 따라잡기
        monkeypatch.setattr(runner, "_atomic_write", orig_write)
        st2 = FarmState.from_dict(
            json.loads((tmp_path / "logs/tracke_state.json").read_text()))
        _safe_variant(_run_variant2, None, st2, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)
        assert len(pd.read_csv(tmp_path / VLEDGER)) == n1, "유일키 멱등 — 중복 0"
        assert st2.variant2_cells["last_ts"] == BASE + 2 * H1, "따라잡기 완료"
        assert st2.variant2_cells["t0_variant2"] == 2, "t0 불변 (write-once)"
        assert st2.variant_cells["last_ts"] == BASE + 2 * H1, "그룹1 무영향"

    def test_종말_폐지_정리는_변형2_포지션을_방치하지_않는다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE + H1
        v = new_variant2(st, t0=1)
        warm2(v, "XRP", atr=2.0, c2=[100.0] * 19, v2=[10.0] * 21)
        v.cells["E16"].positions["XRP"] = FarmPos(d=1, u=30.0, e=90.0, stop=0.0,
                                                  kind="BBMR", risk_d=4.5)
        st.variant2_cells = variant2_to_dict(v)
        st.delisted.append("XRP")
        _finalize_variant2(st)
        vled = pd.read_csv(tmp_path / VLEDGER)
        row = vled.iloc[0]
        assert (row["cell"], row["action"]) == ("E16", "force_exit")
        assert row["price"] == pytest.approx(100.0), "변형2 상태의 마지막 처리 종가"
        assert st.variant2_cells["delisted"] == ["XRP"]
        assert not st.variant2_cells["cells"]["E16"]["positions"]
        _finalize_variant2(st)                           # 멱등 — 추가 기록 없음
        assert len(pd.read_csv(tmp_path / VLEDGER)) == len(vled)

    def test_회귀_v2_활성이_본셀과_E11_산출을_바꾸지_않는다(
            self, tmp_path, monkeypatch):
        # 요건 #1 확장 — v2 활성 재생(실체결 발생) 후에도 본 셀 산출물과
        # E11/E12 서브상태·원장 행이 바이트 단위로 같다.
        def scenario(root, with_v2: bool):
            monkeypatch.chdir(root)
            st = farm()
            warm(st, "BTC", atr=2.0)
            v1 = new_variant(st, t0=1)
            v2 = new_variant2(st, t0=2) if with_v2 else None
            if v2 is not None:
                x2 = v2.ind["BTC"]["x2"]
                x2["c2"] = list(GATE_PASS["c2"])
                x2["v2"] = list(GATE_PASS["v2"])
                x2["u14"], x2["d14"] = 1.0, 0.5
                st.variant2_cells = variant2_to_dict(v2)  # 러너 초기화 순서 미러
            seq = [({"BTC": bar(0, 100, 105, 100, 104)}, None),
                   ({"BTC": bar(1, 104, 104.5, 103.5, 104)}, {"BTC": 0.0001}),
                   ({"BTC": bar(2, 89, 89.5, 88.5, 89)}, None)]
            fills: list = []
            f1: list = []
            f2: list = []
            for bars, fm in seq:
                fills += step(st, bars, fm)
                f1 += step_variant(v1, bars, fm)
                if v2 is not None:
                    f2 += step_variant2(v2, bars, fm)
            _save_all(st, fills, BASE + 2 * H1, 3)
            st.variant_cells = variant_to_dict(v1)
            _save_variant(st, v1, f1, BASE + 2 * H1, 3)
            if v2 is not None:
                _save_variant2(st, v2, f2, BASE + 2 * H1, 3)
            # 영속 상태 재적재 뒤의 본 재생도 동일해야 한다
            st2 = FarmState.from_dict(
                json.loads((root / "logs/tracke_state.json").read_text()))
            fills2 = step(st2, {"BTC": bar(3, 89, 90, 88.8, 89.5)})
            _save_all(st2, fills2, BASE + 3 * H1, 1)
            raw = json.loads((root / "logs/tracke_state.json").read_text())
            v1rows = ledger_lines_raw(root / VLEDGER, ("E11", "E12"))
            vcells = json.dumps(raw.pop("variant_cells"), sort_keys=True)
            raw.pop("variant2_cells")
            return ((root / LEDGER).read_bytes(),
                    (root / "logs/tracke_history.csv").read_bytes(),
                    json.dumps(raw, sort_keys=True), vcells,
                    v1rows, len(f2))
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        a_dir.mkdir(), b_dir.mkdir()
        ra = scenario(a_dir, with_v2=False)
        rb = scenario(b_dir, with_v2=True)
        assert ra[0] == rb[0], "본 원장 바이트 동일"
        assert ra[1] == rb[1], "본 이력 바이트 동일"
        assert ra[2] == rb[2], "본 상태(변형 키 제외) 동일"
        assert ra[3] == rb[3], "E11/E12 서브상태(variant_cells) 동일"
        assert ra[4] == rb[4], "변형 원장 내 E11/E12 행 원문 바이트 동일"
        assert rb[5] > 0, "변형2가 실제 체결을 냈다 (공허한 비교 아님)"


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

    def test_승급_게이트_입력에_변형_경로도_없다(self):
        """변형(E11·E12/E13~E18/E19~E21) 원장·상태도 승급/게이트 입력 금지.

        'variant2_cells'·'variant3_cells' 는 'variant_cells' 의 부분 문자열이
        아니므로 별도 검사.
        """
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        hits = []
        for f in (root / "src").rglob("*.py"):
            if "dashboard" in f.parts:
                continue
            text = f.read_text(errors="ignore")
            if "tracke_variant" in text or "variant_cells" in text \
                    or "variant2_cells" in text or "variant3_cells" in text:
                hits.append(str(f))
        assert hits == [], f"승급 경로에서 변형 파일 참조 금지: {hits}"


V3_UNI = ["BTC", "ETH", "SOL"]        # 테스트 기본 유니버스 (메이저만)
V3_LABEL = "전 유니버스 · 미검증 · 백테스트 기준선 없음(전방 전용) · 판정 권한 없음"


def v3farm(universe: list | None = None, t0: int = 1) -> FarmState:
    """변형3 테스트 서브팜 (t0=1이면 전 봉 라이브)."""
    return new_variant3(farm(), universe or list(V3_UNI), t0)


class TestV3Rules:
    """E19~E21 — 사전 고정: 리스크 최대 1%·6포지션·알파벳 선착·2단계 비용."""

    def test_E19_진입은_BRK24_경로에_리스크_1퍼센트다(self):
        # 같은 워밍업·같은 봉에서 E01(2%)과 E19(1%)는 가격·스탑 동일, 수량 절반
        st = farm()
        warm(st, "BTC", atr=2.0)
        v = v3farm()
        warm2(v, "BTC", atr=2.0, **GATE_PASS)
        b = bar(0, 100, 105, 100, 104)
        step(st, {"BTC": b})
        vfills = step_variant3(v, {"BTC": b})
        p = st.cells["E01"].positions["BTC"]
        q = v.cells["E19"].positions["BTC"]
        assert (q.e, q.stop, q.risk_d) == (p.e, p.stop, p.risk_d)
        assert q.u == pytest.approx(p.u / 2.0), "1% vs 2% — 수량 절반"
        assert q.u * (q.e - q.stop) == pytest.approx(V3_RISK * 10_000)
        e19 = next(f for f in vfills
                   if f["cell"] == "E19" and f["action"] == "enter")
        assert e19["strategy"] == "BRK24"
        # BTC 는 메이저 — 편도 8bp 유지 (2단계 비용의 메이저 단)
        assert e19["cost"] == pytest.approx(q.u * 100.0 * COST_SIDE)
        # E20 도 게이트 통과 시 완전 동일 진입 (BRK24 경로 재사용)
        r = v.cells["E20"].positions["BTC"]
        assert (r.e, r.stop, r.u) == (q.e, q.stop, q.u)

    def test_비메이저_비용은_편도_11bp_왕복_22bp다(self):
        v = v3farm(["AAA", "ETH", "SOL"])
        warm2(v, "AAA", atr=2.0, **GATE_PASS)
        fills = step_variant3(v, {"AAA": bar(0, 100, 105, 100, 104)})
        u = v.cells["E19"].positions["AAA"].u
        e = next(f for f in fills if f["cell"] == "E19" and f["action"] == "enter")
        assert e["cost"] == pytest.approx(u * 100.0 * V3_COST_SIDE_ALT)
        fills2 = step_variant3(v, {"AAA": bar(1, 89, 89.5, 88.5, 89)})
        x = next(f for f in fills2 if f["cell"] == "E19" and f["action"] == "exit")
        assert x["price"] == pytest.approx(89.0), "갭 악화 시가 청산 (BRK 동일)"
        assert x["cost"] == pytest.approx(u * 89.0 * V3_COST_SIDE_ALT)

    def test_동시_신호_6개_초과는_알파벳_선착_최대_6이다(self):
        uni = ["HHH", "GGG", "FFF", "EEE", "DDD", "CCC", "BBB", "AAA"]
        v = v3farm(uni)
        assert v.basket_b == sorted(uni), "유니버스는 알파벳순 동결"
        bars = {}
        for s in v.basket_b:
            warm2(v, s, atr=2.0, **GATE_PASS)
            bars[s] = bar(0, 100, 105, 100, 104)
        fills = step_variant3(v, bars)
        cell = v.cells["E19"]
        assert list(cell.positions) == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        assert len(cell.positions) == V3_MAX_POS == 6
        assert not any(f["cell"] == "E19" and f["action"] == "enter"
                       and f["sym"] in ("GGG", "HHH") for f in fills)
        # 각 진입 시점 검사 기준(진입 시 equity)의 상계: 리스크 <= 1% × 초기자본
        # (비용 차감 후 equity 대비로는 소폭 초과 가능 — 그 잔존 heat 가 신규
        # 진입을 막는 기존 관례 그대로)
        heat = sum(p.u * p.risk_d for p in cell.positions.values())
        assert heat <= HEAT_CAP * 10_000 * (1 + 1e-9), "heat 6% (진입 시점 기준)"
        # 6번째는 heat 잔여 클램프로 체결 — 리스크 '최대' 1% (풀 1%보다 소폭 작음)
        assert cell.positions["FFF"].u < cell.positions["AAA"].u

    def test_heat_클램프가_없으면_차단될_6번째가_체결된다(self):
        # 비용 차감 후 equity 수축 산술 (Codex 검토) — 클램프 사전 교정의 회귀
        uni = [f"S{i}A" for i in range(6)]
        v = v3farm(uni)
        bars = {}
        for s in v.basket_b:
            warm2(v, s, atr=2.0, **GATE_PASS)
            bars[s] = bar(0, 100, 105, 100, 104)
        step_variant3(v, bars)
        cell = v.cells["E19"]
        assert len(cell.positions) == 6, "클램프 없이는 5개에서 heat 차단됐다"
        last = cell.positions[v.basket_b[-1]]
        assert last.u * last.risk_d <= V3_RISK * cell.equity * (1 + 1e-6), \
            "마지막 슬롯 리스크 <= 1% (클램프)"

    def test_7번째_봉_이후에도_6포지션_캡이_유지된다(self):
        uni = [f"C{i}Z" for i in range(7)]
        v = v3farm(uni)
        bars = {}
        for s in v.basket_b[:-1]:                        # 6개 먼저 진입
            warm2(v, s, atr=2.0, **GATE_PASS)
            bars[s] = bar(0, 100, 105, 100, 104)
        warm2(v, v.basket_b[-1], atr=2.0, **GATE_PASS)
        bars[v.basket_b[-1]] = bar(0, 100, 100.5, 99.5, 100)   # 무돌파
        step_variant3(v, bars)
        assert len(v.cells["E19"].positions) == 6
        fills = step_variant3(v, {v.basket_b[-1]: bar(1, 100, 115, 100, 114)})
        assert not any(f["cell"] == "E19" and f["action"] == "enter"
                       for f in fills), "6 보유 중 신규 진입 금지"

    def test_E20_게이트_차단은_E19와_분화된다(self):
        v = v3farm()
        warm2(v, "BTC", atr=2.0, c2=[101.0] * 199 + [100.0],
              v2=[10.0] * 20 + [20.0], u14=1.0, d14=0.5)   # 추세 게이트 실패
        step_variant3(v, {"BTC": bar(0, 100, 105, 100, 104)})
        assert "BTC" in v.cells["E19"].positions, "E19 는 게이트 없음"
        assert "BTC" not in v.cells["E20"].positions, "E20 3중 게이트 차단"

    def test_E21_MR은_다음봉_시가_1퍼센트_사이징이다(self):
        v = v3farm()
        warm(v, "BTC", atr=1.0, n=0, closes=[99.0, 101.0] * 12)
        step_variant3(v, {"BTC": bar(0, 100, 103, 100, 103)})    # z≈+2.9 → 숏
        cell = v.cells["E21"]
        assert "BTC" not in cell.positions, "신호봉 종가 체결 금지"
        assert cell.pending["BTC"]["d"] == -1
        step_variant3(v, {"BTC": bar(1, 105, 105, 104, 104.5)})
        p = cell.positions["BTC"]
        a = 1.0 + (3.0 - 1.0) / ATR1H_N
        assert p.e == pytest.approx(105.0), "다음 봉 시가 체결 (본 MR 동일)"
        assert p.stop == pytest.approx(105.0 + MR_ATR_MULT * a)
        assert p.u * (p.stop - p.e) == pytest.approx(V3_RISK * 10_000), "1% 역산"

    def test_v3는_t0_이전_워밍업_무주문이다(self):
        v = v3farm(t0=BASE + 100 * H1)
        warm2(v, "BTC", atr=2.0, **GATE_PASS)
        fills = step_variant3(v, {"BTC": bar(0, 100, 115, 100, 114)})
        assert fills == []
        for c in v.cells.values():
            assert not c.positions and not c.pending

    def test_변형3_동결_상수와_라벨(self):
        assert [s.cell for s in V3CELLS] == ["E19", "E20", "E21"]
        assert [s.strategy for s in V3CELLS] == ["BRK24", "BRK24GATE", "MR"]
        assert all(s.basket == "U" for s in V3CELLS)
        assert all(s.risk == V3_RISK == 0.01 for s in V3CELLS)
        assert all(s.max_pos == V3_MAX_POS == 6 for s in V3CELLS)
        assert all(s.cost_model == V3_COST_MODEL == "major8_alt11"
                   for s in V3CELLS)
        assert all(s.heat_clamp for s in V3CELLS)
        assert V3_UNIVERSE_N == 40
        assert V3_COST_SIDE_ALT == pytest.approx(0.0011)
        assert all(V3LABELS[c] == V3_LABEL for c in ("E19", "E20", "E21"))
        assert "U" in VBASKET_LABELS
        # 기존 셀(E01~E18)은 전부 기본 파라미터 — 바이트 동일성의 전제
        for s in CELLS + VCELLS + V2CELLS:
            assert (s.risk, s.max_pos, s.cost_model, s.heat_clamp) == \
                (0.02, 3, "", False)

    def test_미지_바스켓_코드는_fail_closed다(self):
        st = farm()
        with pytest.raises(ValueError, match="바스켓"):
            cell_syms(CellSpec("EXX", "BRK24", "X", 24), st)
        assert cell_syms(CellSpec("EYY", "BRK24", "U", 24), st) == \
            ("XRP", "DOGE", "ADA"), "U 는 상태 basket_b 슬롯"

    def test_미지_cost_model은_fail_closed다(self):
        # 오타가 레거시 8bp 로 조용히 폴백하면 비용이 낙관된다 (Codex 검토)
        with pytest.raises(ValueError, match="cost_model"):
            CellSpec("EXX", "BRK24", "U", 24, cost_model="major8_alt12")


class TestV3State:
    """변형3 서브상태 — variant3_cells 키, t0·유니버스 write-once, 독립 초기화."""

    def test_new_variant3는_알파벳_동결과_독립_초기화다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        st.delisted.append("LUNA")
        v = new_variant3(st, ["ZZZ", "BTC", "AAA"], t0=123)
        assert v.t0 == 123
        assert v.basket_b == ["AAA", "BTC", "ZZZ"], "알파벳순 동결 (선착 순서)"
        assert v.last_ts == st.last_ts, "t0_variant3 이전 봉 재생 구조적 차단"
        assert set(v.cells) == {"E19", "E20", "E21"}
        assert all(c.equity == 10_000.0 for c in v.cells.values()), "셀당 신규 $10,000"
        assert v.ind == {}, "본 팜 지표 미상속 — 40심볼 fresh 워밍업 (대칭)"
        assert v.delisted == [], "본 폐지 목록 미상속 — 자체 판정"

    def test_변형3_직렬화는_t0_variant3_명명으로_왕복된다(self):
        v = v3farm(t0=BASE)
        warm2(v, "BTC", atr=2.0, **GATE_PASS)
        step_variant3(v, {"BTC": bar(0, 100, 105, 100, 104)})
        d = variant3_to_dict(v)
        assert "t0_variant3" in d and "t0" not in d
        assert all(k not in d for k in
                   ("variant_cells", "variant2_cells", "variant3_cells"))
        v2 = variant3_from_dict(json.loads(json.dumps(d, default=float)))
        assert json.dumps(variant3_to_dict(v2), sort_keys=True, default=float) \
            == json.dumps(d, sort_keys=True, default=float)
        assert v2.cells["E19"].positions["BTC"].kind == "BRK"
        assert v2.basket_b == ["BTC", "ETH", "SOL"], "유니버스 왕복 보존"

    def test_손상된_변형3_상태는_fail_closed다(self):
        with pytest.raises(ValueError):
            variant3_from_dict(None)
        with pytest.raises(ValueError):
            variant3_from_dict({})
        with pytest.raises(ValueError):
            variant3_from_dict({"t0_variant3": 0, "cells": {}})
        with pytest.raises(ValueError):                  # E13 구성은 변형3이 아니다
            variant3_from_dict({"t0_variant2": 5,
                                "cells": {c: {} for c in
                                          ("E13", "E14", "E15", "E16",
                                           "E17", "E18")}})
        with pytest.raises(ValueError):                  # 셀 부분 결손
            variant3_from_dict({"t0_variant3": 5,
                                "cells": {"E19": {"equity": 10_000.0}}})

    def test_변형3_시가평가와_폐지는_그룹_전용이다(self):
        v = v3farm(["AAA", "BTC"])
        warm2(v, "AAA", atr=2.0, c2=[110.0] * 19, v2=[10.0] * 21)
        v.cells["E19"].positions["AAA"] = FarmPos(d=1, u=10.0, e=100.0,
                                                  stop=88.0, kind="BRK",
                                                  risk_d=12.0)
        eqs = variant3_equities(v)
        assert set(eqs) == {"E19", "E20", "E21"}
        assert eqs["E19"] == pytest.approx(10_000 + 10.0 * 10.0), "마지막 종가 마크"
        fills = variant3_delist(v, "AAA")
        assert fills[0]["cell"] == "E19" and fills[0]["action"] == "force_exit"
        assert fills[0]["price"] == pytest.approx(110.0)
        assert fills[0]["cost"] == pytest.approx(10.0 * 110.0 * V3_COST_SIDE_ALT), \
            "비메이저 폐지 비용도 편도 11bp"
        assert "AAA" in v.delisted
        assert not v.cells["E19"].positions

    def test_warmup_full은_기저와_확장_지표를_함께_채운다(self):
        v = v3farm(["AAA"])
        rows = [(100.0, 101.0, 99.0, 100.0 + k * 0.01, 10.0) for k in range(300)]
        rows.insert(150, (float("nan"), 101.0, 99.0, 100.0, 5.0))   # 결측 스킵
        warmup_full(v, "AAA", rows)
        ind = v.ind["AAA"]
        assert ind["atr1"] is not None and ind["atr1"] > 0, "기저 ATR 형성"
        assert len(ind["hl"]) == 96 and len(ind["cl"]) == 24
        assert ind["pc"] == pytest.approx(100.0 + 299 * 0.01)
        x2 = ind["x2"]
        assert len(x2["c2"]) == 200 and len(x2["v2"]) == GATE_VOL_N + 1
        assert x2["u14"] is not None and x2["u2"] is not None


class TestV3Runner:
    """E19~E21 러너 — 유니버스 동결·40심볼 워밍업·자체 단절 정책·방화벽·격리."""

    OHLC5 = (100.0, 101.0, 99.0, 100.0, 10.0)

    @staticmethod
    def _fake_ex(n_alts: int = 45):
        """티커 48종(메이저 3 + 알트 n) + 활성 마켓을 가진 가짜 ccxt."""
        coins = ["BTC", "ETH", "SOL"] + [f"A{i:02d}" for i in range(n_alts)]
        tickers = [{"symbol": f"{c}USDT", "turnover24h": str(9e9 - i * 1e6)}
                   for i, c in enumerate(coins)]

        class Ex:
            markets = {f"{c}/USDT:USDT": {"active": True} for c in coins}

            def publicGetV5MarketTickers(self, params):
                return {"retCode": "0", "result": {"list": tickers}}
        return Ex()

    def test_첫_호출은_유니버스_동결과_40심볼_워밍업이다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        calls = []

        def fake_fetch(ex, coin, since, now_h):
            calls.append(coin)
            return {t: self.OHLC5 for t in range(since, now_h, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        _run_variant3(self._fake_ex(), st, {}, {}, BASE, BASE)
        vc = st.variant3_cells
        t0v = vc["t0_variant3"]
        assert t0v > 0
        uni = vc["basket_b"]
        assert len(uni) == V3_UNIVERSE_N and uni == sorted(uni), "상위 40 알파벳순"
        assert {"BTC", "ETH", "SOL"} <= set(uni), "메이저 포함 (중복 허용)"
        assert calls == uni, "유니버스 40심볼 전부 fresh 워밍업 수집"
        assert vc["last_ts"] == BASE, "본 팜과 정렬 (워밍업 무주문)"
        assert set(vc["cells"]) == {"E19", "E20", "E21"}
        assert vc["ind"]["BTC"]["atr1"] > 0, "기저 지표 fresh 워밍업"
        assert len(vc["ind"]["BTC"]["x2"]["c2"]) == 200, "확장 지표 워밍업"
        vh = pd.read_csv(tmp_path / VHIST)
        assert list(vh.columns) == VHIST_COLS, "e19~e21 스키마 이월"
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant3_cells"]["t0_variant3"] == t0v
        # 이후 재생 실행이 t0·유니버스를 절대 옮기지 않는다
        monkeypatch.setattr(runner, "fetch_funding_range", lambda *a: {})
        st.last_ts = BASE + H1
        _run_variant3(object(), st, {s: {BASE + H1: self.OHLC5} for s in uni},
                      {}, BASE + H1, BASE + H1)
        assert st.variant3_cells["t0_variant3"] == t0v
        assert st.variant3_cells["basket_b"] == uni
        assert st.variant3_cells["last_ts"] == BASE + H1

    def test_워밍업_실패나_후보부족은_t0를_동결하지_않는다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE

        class DeadEx:
            markets = {}

            def publicGetV5MarketTickers(self, params):
                return {"retCode": "1"}
        with pytest.raises(RuntimeError, match="티커"):
            _run_variant3(DeadEx(), st, {}, {}, BASE, BASE)
        assert st.variant3_cells is None
        with pytest.raises(RuntimeError, match="부족"):     # 후보 10 < 40
            _run_variant3(self._fake_ex(n_alts=7), st, {}, {}, BASE, BASE)
        assert st.variant3_cells is None
        monkeypatch.setattr(runner, "fetch_1h_paged", lambda *a: None)
        with pytest.raises(RuntimeError, match="불완전"):   # 캔들 수집 실패
            _run_variant3(self._fake_ex(), st, {}, {}, BASE, BASE)
        assert st.variant3_cells is None
        monkeypatch.setattr(                               # 메이저 깊이 부족
            runner, "fetch_1h_paged",
            lambda ex, coin, since, now_h: {t: self.OHLC5
                                            for t in range(now_h - 100 * H1,
                                                           now_h, H1)})
        with pytest.raises(RuntimeError, match="깊이 부족"):
            _run_variant3(self._fake_ex(), st, {}, {}, BASE, BASE)
        assert st.variant3_cells is None
        assert not (tmp_path / VLEDGER).exists(), "동결 지연 — 파일 미생성"

    def test_비메이저_늦은_상장은_짧은_연속_이력으로_동결된다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE

        def fake_fetch(ex, coin, since, now_h):
            if coin == "A00":                             # 늦은 상장 — 50봉 연속
                return {t: self.OHLC5
                        for t in range(now_h - 50 * H1, now_h, H1)}
            return {t: self.OHLC5 for t in range(since, now_h, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        _run_variant3(self._fake_ex(), st, {}, {}, BASE, BASE)
        assert st.variant3_cells is not None
        assert len(st.variant3_cells["ind"]["A00"]["x2"]["c2"]) == 50
        assert len(st.variant3_cells["ind"]["BTC"]["x2"]["c2"]) == 200

    def test_save_variant3는_t0와_유니버스_변경을_거부한다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = v3farm(["BTC", "ETH"])
        st.variant3_cells = variant3_to_dict(v)
        v.t0 = 2
        with pytest.raises(ValueError, match="write-once"):
            _save_variant3(st, v, [], BASE, 0)
        v.t0 = 1
        v.basket_b = ["BTC", "XXX"]
        with pytest.raises(ValueError, match="유니버스 변경"):
            _save_variant3(st, v, [], BASE, 0)
        assert not (tmp_path / VLEDGER).exists()
        assert st.variant3_cells["t0_variant3"] == 1, "기존 기록 보존"
        assert st.variant3_cells["basket_b"] == ["BTC", "ETH"]

    def test_save_variant3는_영구_공석_축소를_거부한다(self, tmp_path, monkeypatch):
        # delisted 는 단조 증가 — 축소(부활)는 저장 경계에서 죽는다 (Codex 검토)
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = v3farm(["BTC", "ETH"])
        v.delisted.append("ETH")
        st.variant3_cells = variant3_to_dict(v)
        v.delisted = []                                  # 부활 시도
        with pytest.raises(ValueError, match="공석 축소"):
            _save_variant3(st, v, [], BASE, 0)
        assert not (tmp_path / VLEDGER).exists()
        assert st.variant3_cells["delisted"] == ["ETH"], "기존 기록 보존"

    def test_변형3_방화벽은_비변형3_행을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = v3farm()
        st.variant3_cells = variant3_to_dict(v)
        good = dict(cell="E19", sym="BTC", strategy="BRK24", bar_close=BASE + H1,
                    action="enter", price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                    direction=1, funding=0.0)
        for bad in (dict(good, cell="E01"), dict(good, cell="E11"),
                    dict(good, cell="E13", strategy="BRK24GATE"),
                    dict(good, strategy="MR"),       # 유효 셀×유효 전략 교차 오염
                    dict(good, cell="E21")):         # E21 은 MR 이어야 한다
            with pytest.raises(ValueError, match="비변형3"):
                _save_variant3(st, v, [bad], BASE, 1)
        assert not (tmp_path / VLEDGER).exists()

    def test_본_원장은_변형3_행을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        row = dict(cell="E19", sym="BTC", strategy="BRK24", bar_close=BASE + H1,
                   action="enter", price=100.0, qty=1.0, pnl=0.0, cost=0.08,
                   direction=1, funding=0.0)
        with pytest.raises(ValueError, match="비공식 셀"):
            _save_all(farm(), [row], BASE, 1)
        assert not (tmp_path / LEDGER).exists()

    def test_변형3_단절은_자체_정책으로_폐지된다(self, tmp_path, monkeypatch):
        # 본 폐지 미러가 아니라 자체 48h 단절 판정 — 잔여봉 없으면 즉시 폐지
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        v = v3farm(["AAA", "BBB"])
        v.last_ts = BASE + 99 * H1
        warm2(v, "AAA", atr=2.0, c2=[100.0] * 19, v2=[10.0] * 21)
        v.cells["E19"].positions["AAA"] = FarmPos(d=1, u=10.0, e=90.0, stop=1.0,
                                                  kind="BRK", risk_d=3.0)
        fresh = BASE + 400 * H1

        def fake_fetch(ex, coin, since, end):
            if coin == "AAA":
                return {}                                 # 300시간 무봉 — 단절
            return {t: self.OHLC5 for t in range(since, end, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        monkeypatch.setattr(runner, "fetch_funding_range", lambda *a: {})
        _, v3_end, vdata, _, dfills = _variant3_inputs(
            object(), v, {}, {}, BASE + 100 * H1, fresh)
        assert "AAA" in v.delisted, "자체 단절 폐지"
        assert dfills[0]["cell"] == "E19"
        assert dfills[0]["action"] == "force_exit"
        assert dfills[0]["price"] == pytest.approx(100.0), "마지막 처리 종가"
        assert "AAA" not in vdata and v3_end == fresh
        assert set(vdata) == {"BBB"}

    def test_변형3_신선_갭과_수집실패는_단계_실패다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        v = v3farm(["BBB"])
        v.last_ts = BASE - H1
        monkeypatch.setattr(runner, "fetch_1h_paged", lambda *a: None)
        with pytest.raises(RuntimeError, match="수집 실패"):
            _variant3_inputs(object(), v, {}, {}, BASE, BASE + 2 * H1)
        monkeypatch.setattr(                               # 가운데 봉 결손 (신선)
            runner, "fetch_1h_paged",
            lambda ex, coin, since, end: {BASE: self.OHLC5,
                                          BASE + 2 * H1: self.OHLC5})
        with pytest.raises(RuntimeError, match="갭"):
            _variant3_inputs(object(), v, {}, {}, BASE, BASE + 2 * H1)
        monkeypatch.setattr(runner, "fetch_1h_paged", lambda *a: {})
        with pytest.raises(RuntimeError, match="새 봉 없음"):
            _variant3_inputs(object(), v, {}, {}, BASE, BASE + 2 * H1)

    def test_바스켓_중복_심볼은_본_수집분을_재사용한다(self, tmp_path, monkeypatch):
        # 공유 캐시 — 본 실행이 수집한 심볼은 재페치 없이 유니버스 전용만 페치
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        v = v3farm(["AAA", "BTC"])
        v.last_ts = BASE - H1
        fetched = []

        def fake_fetch(ex, coin, since, end):
            fetched.append(coin)
            return {t: self.OHLC5 for t in range(since, end, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        monkeypatch.setattr(runner, "fetch_funding_range", lambda *a: {})
        data = {"BTC": {BASE: self.OHLC5}}                # 본 실행 수집분
        _, _, vdata, _, _ = _variant3_inputs(
            object(), v, data, {"BTC": {}}, BASE, BASE)
        assert fetched == ["AAA"], "BTC 는 공유 캐시 — 재페치 없음"
        assert set(vdata) == {"AAA", "BTC"}
        assert data["BTC"] == {BASE: self.OHLC5}, "본 수집분 원본 불변"

    def test_신선하게_뒤처진_심볼은_v3_재생_끝을_캡한다(self, tmp_path, monkeypatch):
        # 본 러너 replay_end = min(전 심볼 최신 봉) 규칙 미러 — 48h 미만 지연
        # 심볼이 있으면 v3 전체가 그 봉까지만 전진하고 다음 실행이 따라잡는다
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        v = v3farm(["AAA", "BBB"])
        v.last_ts = BASE - H1

        def fake_fetch(ex, coin, since, end):
            last = BASE + H1 if coin == "AAA" else BASE + 2 * H1   # AAA 1봉 지연
            return {t: self.OHLC5 for t in range(since, last + H1, H1)}
        monkeypatch.setattr(runner, "fetch_1h_paged", fake_fetch)
        monkeypatch.setattr(runner, "fetch_funding_range", lambda *a: {})
        vsince, v3_end, vdata, _, dfills = _variant3_inputs(
            object(), v, {}, {}, BASE, BASE + 2 * H1)
        assert (vsince, v3_end) == (BASE, BASE + H1), "지연 심볼 최신 봉으로 캡"
        assert dfills == [], "신선 지연은 폐지 아님"
        assert set(vdata) == {"AAA", "BBB"}

    def test_전_유니버스_공석이면_재생_없이_조기_반환한다(
            self, tmp_path, monkeypatch):
        # 잔여 심볼 0 — v3_end < vsince 로 재생·펀딩·페치 전부 생략 (fail-closed)
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        v = v3farm(["AAA", "BBB"])
        v.last_ts = BASE + 10 * H1
        v.delisted = ["AAA", "BBB"]                       # 전부 영구 공석
        called = []
        monkeypatch.setattr(runner, "fetch_1h_paged",
                            lambda *a: called.append(a) or {})
        monkeypatch.setattr(runner, "fetch_funding_range",
                            lambda *a: called.append(a) or {})
        vsince, v3_end, vdata, vfund, dfills = _variant3_inputs(
            object(), v, {}, {}, BASE + 11 * H1, BASE + 11 * H1)
        assert v3_end < vsince, "재생할 심볼 없음"
        assert vdata == {} and vfund == {} and dfills == []
        assert called == [], "페치·펀딩 수집 전부 생략"

    def test_변형3_실패는_본_E11_E18_커밋을_막지_않고_재실행이_따라잡는다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        v1 = new_variant(st, t0=1)
        v1.last_ts = BASE - H1
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        v2.last_ts = BASE - H1
        warm2(v2, "BTC", atr=2.0, **GATE_PASS)
        st.variant2_cells = variant2_to_dict(v2)
        v3 = new_variant3(st, ["BTC"], t0=3)
        v3.last_ts = BASE - H1
        warm2(v3, "BTC", atr=2.0, **GATE_PASS)
        st.variant3_cells = variant3_to_dict(v3)
        data = {"BTC": {BASE: (100.0, 105.0, 100.0, 104.0, 20.0),
                        BASE + H1: (104.0, 104.5, 103.5, 104.0, 10.0),
                        BASE + 2 * H1: (89.0, 89.5, 88.5, 89.0, 10.0)}}
        fills: list = []
        for t in sorted(data["BTC"]):
            fills += step(st, {"BTC": BarE(t, *data["BTC"][t])})
        _save_all(st, fills, BASE + 2 * H1, 3)
        _safe_variant(_run_variant, None, st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)                # 그룹1 정상 커밋
        _safe_variant(_run_variant2, None, st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)                # 그룹2 정상 커밋
        main_led = (tmp_path / LEDGER).read_bytes()
        main_state = (tmp_path / "logs/tracke_state.json").read_bytes()
        n_before = len(pd.read_csv(tmp_path / VLEDGER))
        orig_write = runner._atomic_write

        def boom(path, text):
            raise OSError("디스크 장애 주입")
        monkeypatch.setattr(runner, "_atomic_write", boom)
        _safe_variant(_run_variant3, object(), st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)                # 예외가 전파되면 실패
        assert (tmp_path / LEDGER).read_bytes() == main_led, "본 원장 불변"
        assert (tmp_path / "logs/tracke_state.json").read_bytes() == main_state
        assert not (tmp_path / ERR_MARK).exists(), "본 팜 중단 마커 오염 금지"
        n1 = len(pd.read_csv(tmp_path / VLEDGER))
        assert n1 > n_before, "변형3 원장은 상태 저장 전에 append 됨"
        # 재실행 — 디스크 상태(뒤처진 variant3_cells)에서 유일키 멱등 따라잡기
        monkeypatch.setattr(runner, "_atomic_write", orig_write)
        st2 = FarmState.from_dict(
            json.loads((tmp_path / "logs/tracke_state.json").read_text()))
        _safe_variant(_run_variant3, object(), st2, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)
        assert len(pd.read_csv(tmp_path / VLEDGER)) == n1, "유일키 멱등 — 중복 0"
        assert st2.variant3_cells["last_ts"] == BASE + 2 * H1, "따라잡기 완료"
        assert st2.variant3_cells["t0_variant3"] == 3, "t0 불변 (write-once)"
        assert st2.variant_cells["last_ts"] == BASE + 2 * H1, "그룹1 무영향"
        assert st2.variant2_cells["last_ts"] == BASE + 2 * H1, "그룹2 무영향"

    def test_회귀_v3_활성이_본셀과_E11_E18_산출을_바꾸지_않는다(
            self, tmp_path, monkeypatch):
        # 요건 — v3 활성 재생(실체결 발생) 후에도 본 셀 산출물(원장·이력·상태
        # projection)과 E11~E18 서브상태·변형 원장 행이 바이트 단위로 같다.
        def scenario(root, with_v3: bool):
            monkeypatch.chdir(root)
            st = farm()
            warm(st, "BTC", atr=2.0)
            v1 = new_variant(st, t0=1)
            v2 = new_variant2(st, t0=2)
            x2 = v2.ind["BTC"]["x2"]
            x2["c2"] = list(GATE_PASS["c2"])
            x2["v2"] = list(GATE_PASS["v2"])
            x2["u14"], x2["d14"] = 1.0, 0.5
            st.variant2_cells = variant2_to_dict(v2)
            v3 = None
            if with_v3:
                v3 = new_variant3(st, ["BTC", "ETH", "SOL"], t0=3)
                warm2(v3, "BTC", atr=2.0, **GATE_PASS)
                st.variant3_cells = variant3_to_dict(v3)
            seq = [({"BTC": bar(0, 100, 105, 100, 104)}, None),
                   ({"BTC": bar(1, 104, 104.5, 103.5, 104)}, {"BTC": 0.0001}),
                   ({"BTC": bar(2, 89, 89.5, 88.5, 89)}, None)]
            fills: list = []
            f1: list = []
            f2: list = []
            f3: list = []
            for bars, fm in seq:
                fills += step(st, bars, fm)
                f1 += step_variant(v1, bars, fm)
                f2 += step_variant2(v2, bars, fm)
                if v3 is not None:
                    f3 += step_variant3(v3, bars, fm)
            # 비교가 공허하지 않음 — 본·E11·E13 경로 전부 실체결이 있다
            assert any(f["cell"] == "E01" for f in fills)
            assert any(f["cell"] == "E11" for f in f1)
            assert any(f["cell"] == "E13" for f in f2)
            _save_all(st, fills, BASE + 2 * H1, 3)
            st.variant_cells = variant_to_dict(v1)
            _save_variant(st, v1, f1, BASE + 2 * H1, 3)
            _save_variant2(st, v2, f2, BASE + 2 * H1, 3)
            if v3 is not None:
                _save_variant3(st, v3, f3, BASE + 2 * H1, 3)
            # 영속 상태 재적재 뒤의 본 재생도 동일해야 한다
            st2 = FarmState.from_dict(
                json.loads((root / "logs/tracke_state.json").read_text()))
            fills2 = step(st2, {"BTC": bar(3, 89, 90, 88.8, 89.5)})
            _save_all(st2, fills2, BASE + 3 * H1, 1)
            raw = json.loads((root / "logs/tracke_state.json").read_text())
            prior = ledger_lines_raw(root / VLEDGER, (
                "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18"))
            vcells = json.dumps(raw.pop("variant_cells"), sort_keys=True)
            v2cells = json.dumps(raw.pop("variant2_cells"), sort_keys=True)
            raw.pop("variant3_cells")
            return ((root / LEDGER).read_bytes(),
                    (root / "logs/tracke_history.csv").read_bytes(),
                    json.dumps(raw, sort_keys=True), vcells, v2cells,
                    prior, len(f3))
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        a_dir.mkdir(), b_dir.mkdir()
        ra = scenario(a_dir, with_v3=False)
        rb = scenario(b_dir, with_v3=True)
        assert ra[0] == rb[0], "본 원장 바이트 동일"
        assert ra[1] == rb[1], "본 이력 바이트 동일"
        assert ra[2] == rb[2], "본 상태(변형3 키 제외 projection) 동일"
        assert ra[3] == rb[3], "E11/E12 서브상태 동일"
        assert ra[4] == rb[4], "E13~E18 서브상태 동일"
        assert ra[5] == rb[5], "변형 원장 내 E11~E18 행 원문 바이트 동일"
        assert rb[6] > 0, "변형3이 실제 체결을 냈다 (공허한 비교 아님)"

    def test_통합_이력_행은_세_그룹_수치를_담는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        v3 = v3farm()
        st.variant3_cells = variant3_to_dict(v3)
        _save_variant(st, v1, [], BASE, 0)
        _save_variant2(st, v2, [], BASE, 0)
        _save_variant3(st, v3, [], BASE, 0)               # keep-last — 마지막 행
        vh = pd.read_csv(tmp_path / VHIST)
        assert len(vh) == 1
        row = vh.iloc[0]
        for c in ("e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18",
                  "e19", "e20", "e21"):
            assert row[c] == pytest.approx(10_000.0)
        assert row["equity"] == pytest.approx(110_000.0), "전 변형 셀(11개) 합"

    def test_변형3_유니버스_피커는_결정론적_상위40이다(self):
        tk = [{"symbol": "BTCUSDT", "turnover24h": "9e9"},
              {"symbol": "AAAUSDT", "turnover24h": "100000000"},
              {"symbol": "BBBUSDT", "turnover24h": "100000000"},   # 동률
              {"symbol": "BTC-27JUN25", "turnover24h": "8e9"},     # 만기물 제외
              {"symbol": "NANUSDT", "turnover24h": "nan"},         # 비유한 제외
              {"symbol": "LOWUSDT", "turnover24h": "1000000"},     # $5M 미만
              {"symbol": "DEADUSDT", "turnover24h": "500000000"},  # 비활성 마켓
              {"symbol": "EXPUSDT", "turnover24h": "500000000"},   # 만기 메타
              {"symbol": "AAAUSDT", "turnover24h": "700000000"}]   # 중복 행
        mk = {f"{c}/USDT:USDT": {"active": True}
              for c in ("BTC", "AAA", "BBB", "NAN", "LOW")}
        mk["DEAD/USDT:USDT"] = {"active": False}
        mk["EXP/USDT:USDT"] = {"active": True, "expiry": 1_800_000_000_000}
        assert pick_universe40(tk, mk) == ["AAA", "BBB", "BTC"], "알파벳순 반환"
        # 중복 행은 최대 유효 거래대금으로 집계 — 입력 순서 무관 (Codex 검토)
        assert pick_universe40(list(reversed(tk)), mk) == ["AAA", "BBB", "BTC"]
        # 컷 경계 동률은 (-거래대금, 심볼) tie-break — C00..C39 가 남는다
        tk2 = [{"symbol": f"C{i:02d}USDT", "turnover24h": "6000000"}
               for i in range(41)]
        out = pick_universe40(tk2)
        assert len(out) == V3_UNIVERSE_N
        assert out == [f"C{i:02d}" for i in range(40)], "동률 tie-break 결정론"
        # 중복이 컷 경계를 바꾸는 경우도 순서 무관 — 최대값 집계라 X 가 든다
        tk3 = tk2 + [{"symbol": "XXXUSDT", "turnover24h": "5500000"},
                     {"symbol": "XXXUSDT", "turnover24h": "7000000"}]
        assert "XXX" in pick_universe40(tk3)
        assert "XXX" in pick_universe40(list(reversed(tk3)))


V4_LABEL = "손익비 1.5:1 익절 변형 · 미검증 · 판정 권한 없음"


def v4farm(t0: int = 1) -> FarmState:
    """변형4 테스트 서브팜 (t0=1이면 전 봉 라이브)."""
    return new_variant4(farm(), t0)


def v4pos(**kw) -> FarmPos:
    """변형4 BRKR15 보유 포지션 기본값 (롱 e=100, stop=88, tgt=118 = 1.5R)."""
    base = dict(d=1, u=10.0, e=100.0, stop=88.0, kind="BRKR15", tgt=118.0,
                hold=1, risk_d=12.0)
    base.update(kw)
    return FarmPos(**base)


class TestV4Rules:
    """E22·E23 (BRK24R15) — 사전 고정: 손익비 1.5:1 익절·최대보유 제한 없음."""

    def test_진입_스탑_사이징은_BRK24와_정확히_같다(self):
        # 같은 워밍업·같은 봉에서 E01(BRK24)과 E22(BRK24R15)는 완전 동일 체결
        st = farm()
        warm(st, "BTC", atr=2.0)
        v = new_variant4(st, t0=1)
        b = bar(0, 100, 105, 100, 104)               # 돌파하되 1.5R(118) 미도달
        fills = step(st, {"BTC": b})
        vfills = step_variant4(v, {"BTC": b})
        p = st.cells["E01"].positions["BTC"]
        q = v.cells["E22"].positions["BTC"]
        assert (q.e, q.stop, q.u, q.risk_d) == (p.e, p.stop, p.u, p.risk_d)
        e1 = next(f for f in fills if f["cell"] == "E01" and f["action"] == "enter")
        e22 = next(f for f in vfills
                   if f["cell"] == "E22" and f["action"] == "enter")
        assert (e22["price"], e22["qty"], e22["cost"]) == \
            (e1["price"], e1["qty"], e1["cost"])
        assert e22["strategy"] == "BRK24R15"
        assert q.kind == "BRKR15"
        assert q.tgt == 2.5 * q.e - 1.5 * q.stop, "1.5R = fill + 1.5×(fill−stop)"
        assert q.tgt == pytest.approx(118.0)

    def test_1_5R_목표는_봉내_레벨_체결이다(self):
        v = v4farm()
        warm(v, "BTC", atr=2.0, hi=100.0, lo=1.0)     # 역채널(저가 1) 배제
        v.cells["E22"].positions["BTC"] = v4pos()
        fills = step_variant4(v, {"BTC": bar(0, 105, 119, 104, 106)})
        assert fills[0]["action"] == "target"
        assert fills[0]["price"] == pytest.approx(118.0)
        assert "BTC" not in v.cells["E22"].positions

    def test_1R에서는_청산되지_않는다(self):
        # 손익비 분화의 핵심 — 1R(112) 도달만으로는 무청산 (E11 과 구분)
        st = farm()
        warm(st, "BTC", atr=2.0)
        v1 = new_variant(st, t0=1)
        v4 = new_variant4(st, t0=1)
        b0 = {"BTC": bar(0, 100, 105, 100, 104)}
        step_variant(v1, b0)
        step_variant4(v4, b0)
        assert v1.cells["E11"].positions["BTC"].tgt == pytest.approx(112.0)
        assert v4.cells["E22"].positions["BTC"].tgt == pytest.approx(118.0)
        b1 = {"BTC": bar(1, 104, 113, 103, 110)}     # 1R 도달, 1.5R 미도달
        f1 = step_variant(v1, b1)
        f4 = step_variant4(v4, b1)
        assert [f["action"] for f in f1 if f["cell"] == "E11"] == ["target"]
        assert f1[0]["price"] == pytest.approx(112.0)
        assert f4 == [], "1.5R 미도달 — 변형4 는 보유 유지"
        assert "BTC" in v4.cells["E22"].positions

    def test_갭이_유리해도_목표는_레벨_체결이다(self):
        # RSI-DIV #15 관례 동일 — 시가 125 > 목표 118 여도 118 체결
        v = v4farm()
        warm(v, "BTC", atr=2.0, hi=100.0, lo=1.0)
        v.cells["E22"].positions["BTC"] = v4pos()
        fills = step_variant4(v, {"BTC": bar(0, 125, 130, 123, 126)})
        assert fills[0]["action"] == "target"
        assert fills[0]["price"] == pytest.approx(118.0), "레벨 체결 (유리한 갭 무시)"

    def test_스탑과_목표_동시_도달이면_BRK_청산이_우선이다(self):
        v = v4farm()
        warm(v, "BTC", atr=2.0, hi=100.0, lo=1.0)
        v.cells["E22"].positions["BTC"] = v4pos()
        fills = step_variant4(v, {"BTC": bar(0, 100, 119, 80, 90)})  # 88·118 모두
        assert fills[0]["action"] == "exit"
        assert fills[0]["price"] == pytest.approx(88.0), "스탑 우선 (비관)"
        assert not any(f["action"] == "target" for f in fills)

    def test_역채널_추적_청산은_변형4에도_유지된다(self):
        # 기존 BRK24 청산 규칙 병존 — n/2=12봉 역채널 + 갭 악화
        v = v4farm()
        warm(v, "BTC", atr=2.0, hi=100.0, lo=95.0)   # 최근 저가 95 = 역채널
        v.cells["E22"].positions["BTC"] = v4pos()
        fills = step_variant4(v, {"BTC": bar(0, 96, 97, 94, 96)})
        assert fills[0]["action"] == "exit"
        assert fills[0]["price"] == pytest.approx(95.0), "역채널 레벨 (스탑 88 아님)"

    def test_최대보유_제한이_없다(self):
        # E11/E12 의 12봉 캡과의 명시적 분화 — 같은 봉 시퀀스에서 E11 은
        # 12봉째 타임아웃, E22 는 40봉을 지나도 보유 (손익비 효과만 분리)
        st = farm()
        v1 = new_variant(st, t0=1)
        v4 = new_variant4(st, t0=1)
        warm(v1, "BTC", atr=2.0, hi=1000.0, lo=1.0)
        warm(v4, "BTC", atr=2.0, hi=1000.0, lo=1.0)
        v1.cells["E11"].positions["BTC"] = vpos(stop=1.0, tgt=1e9, risk_d=99.0)
        v4.cells["E22"].positions["BTC"] = v4pos(stop=1.0, tgt=1e9, risk_d=99.0)
        f1: list = []
        f4: list = []
        for k in range(40):
            b = {"BTC": bar(k, 100.0, 100.5, 99.5 + 0.01 * k, 100.0)}
            f1 += step_variant(v1, b)
            f4 += step_variant4(v4, b)
        assert [f["action"] for f in f1] == ["timeout"], "E11 은 12봉 캡"
        assert f4 == [], "E22 는 최대보유 제한 없음 (40봉 무청산)"
        p = v4.cells["E22"].positions["BTC"]
        assert p.hold == 1, "hold 는 변형4가 읽지도 늘리지도 않는 죽은 필드"

    def test_진입봉_1_5R_도달은_같은_봉_익절된다(self):
        # RSI-DIV #17 관례 — 체결봉부터 목표 검사 (레벨 체결)
        v = v4farm()
        warm(v, "BTC", atr=2.0)
        fills = step_variant4(v, {"BTC": bar(0, 100, 119, 100, 110)})
        acts = [f["action"] for f in fills if f["cell"] == "E22"]
        assert acts == ["enter", "same_bar_target"]
        assert fills[-1]["price"] == pytest.approx(118.0)
        assert "BTC" not in v.cells["E22"].positions

    def test_진입봉_동시_도달은_스탑이_우선이다(self):
        v = v4farm()
        warm(v, "BTC", atr=2.0)
        fills = step_variant4(v, {"BTC": bar(0, 100, 119, 87, 100)})  # 88·118
        acts = [f["action"] for f in fills if f["cell"] == "E22"]
        assert acts == ["enter", "same_bar_stop"]
        assert not any(f["action"] == "same_bar_target" for f in fills)

    def test_숏_1_5R_목표는_진입_아래_대칭이다(self):
        v = v4farm()
        warm(v, "BTC", atr=2.0, hi=100.0, lo=90.0)
        step_variant4(v, {"BTC": bar(0, 89, 89.5, 80, 82)})   # 하향 돌파 숏
        p = v.cells["E22"].positions["BTC"]
        assert p.d == -1
        assert p.e == pytest.approx(89.0)                     # min(시가, 채널 90)
        assert p.stop == pytest.approx(101.0)                 # 89 + 6*2
        assert p.tgt == pytest.approx(71.0), "2.5*89 − 1.5*101 (1.5R 아래)"
        assert p.tgt == 2.5 * p.e - 1.5 * p.stop

    def test_결측_봉은_변형4_관리를_정지시킨다(self):
        # 엔진 공통 fail-closed — 결측 봉은 목표·스탑 검사 전부 정지
        v = v4farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=1.0)
        warm(v, "ETH", atr=2.0, hi=1000.0, lo=1.0)
        v.cells["E22"].positions["BTC"] = v4pos()
        fills = step_variant4(v, {"ETH": bar(0, 130, 130, 130, 130)})  # BTC 결측
        assert fills == [] and "BTC" in v.cells["E22"].positions
        fills = step_variant4(v, {"BTC": bar(1, 119, 120, 118, 119),
                                  "ETH": bar(1, 130, 130, 130, 130)})
        assert fills[0]["action"] == "target"

    def test_변형4는_t0_variant4_이전_워밍업_무주문이다(self):
        v = new_variant4(farm(), t0=BASE + 100 * H1)
        warm(v, "BTC", atr=2.0)
        fills = step_variant4(v, {"BTC": bar(0, 100, 115, 100, 114)})  # 명백 돌파
        assert fills == []
        for c in v.cells.values():
            assert not c.positions and not c.pending

    def test_변형4_동결_상수와_라벨(self):
        assert V4_TP_R == 1.5
        assert [s.cell for s in V4CELLS] == ["E22", "E23"]
        assert all(s.strategy == "BRK24R15" and s.n == 24 for s in V4CELLS)
        assert [s.basket for s in V4CELLS] == ["A", "B"]
        assert V4LABELS["E22"] == V4LABELS["E23"] == V4_LABEL
        # 리스크 2%·MAX_POS 3·균일 16bp — 본 셀 공통 파라미터 그대로
        for s in V4CELLS:
            assert (s.risk, s.max_pos, s.cost_model, s.heat_clamp) == \
                (0.02, 3, "", False)


class TestV4State:
    """변형4 서브상태 — variant4_cells 키, t0_variant4 write-once, 그룹 분리."""

    def test_new_variant4는_동결_바스켓_재사용과_지표_깊은_분리다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        v = new_variant4(st, t0=123)
        assert v.t0 == 123
        assert v.basket_b == st.basket_b == ["XRP", "DOGE", "ADA"]
        assert v.last_ts == st.last_ts, "t0_variant4 이전 봉 재생 구조적 차단"
        assert v.ind == st.ind, "시장 전용 지표 스냅숏 상속 (값 동일)"
        assert set(v.cells) == {"E22", "E23"}
        assert all(c.equity == 10_000.0 for c in v.cells.values())
        v.ind["BTC"]["atr1"] = 999.0                 # 변형 쪽 변이가
        v.ind["BTC"]["hl"].append([1.0, 1.0])
        assert st.ind["BTC"]["atr1"] != 999.0, "본 팜 지표에 새면 안 된다 (깊은 복사)"
        assert len(st.ind["BTC"]["hl"]) != len(v.ind["BTC"]["hl"])

    def test_변형4_직렬화는_t0_variant4_명명으로_왕복된다(self):
        v = new_variant4(farm(), t0=BASE)
        warm(v, "BTC", atr=2.0)
        step_variant4(v, {"BTC": bar(0, 100, 105, 100, 104)})   # 포지션 포함
        d = variant4_to_dict(v)
        assert "t0_variant4" in d and "t0" not in d
        assert all(k not in d for k in ("variant_cells", "variant2_cells",
                                        "variant3_cells", "variant4_cells"))
        v2 = variant4_from_dict(json.loads(json.dumps(d, default=float)))
        assert json.dumps(variant4_to_dict(v2), sort_keys=True, default=float) \
            == json.dumps(d, sort_keys=True, default=float)
        assert v2.cells["E22"].positions["BTC"].kind == "BRKR15"
        assert v2.cells["E22"].positions["BTC"].tgt == pytest.approx(118.0)

    def test_기존_그룹_직렬화에_variant4_키가_새지_않는다(self):
        # 바이트 동일성의 전제 — v1·v2·v3 서브상태 JSON 에 새 키가 생기면
        # 기존 그룹 상태 바이트가 변한다 (_vgroup_to_dict 의 pop 회귀)
        st = farm()
        for d in (variant_to_dict(new_variant(st, t0=1)),
                  variant2_to_dict(new_variant2(st, t0=2)),
                  variant3_to_dict(new_variant3(st, ["BTC"], t0=3)),
                  variant4_to_dict(new_variant4(st, t0=4))):
            assert "variant4_cells" not in d
            assert "t0" not in d

    def test_본_상태는_variant4_cells를_불투명하게_왕복_보존한다(self):
        st = farm()
        assert st.variant4_cells is None, "미초기화 기본값 None (초기화 대상)"
        st.variant4_cells = variant4_to_dict(new_variant4(st, t0=BASE))
        rt = FarmState.from_dict(json.loads(json.dumps(st.to_dict(), default=float)))
        assert rt.variant4_cells == st.variant4_cells
        assert json.dumps(rt.to_dict(), sort_keys=True, default=float) == \
            json.dumps(st.to_dict(), sort_keys=True, default=float)

    def test_손상된_변형4_상태는_fail_closed다(self):
        with pytest.raises(ValueError):
            variant4_from_dict(None)
        with pytest.raises(ValueError):
            variant4_from_dict({})
        with pytest.raises(ValueError):
            variant4_from_dict({"t0_variant4": 0, "cells": {}})
        with pytest.raises(ValueError):                  # E11 구성은 변형4가 아니다
            variant4_from_dict({"t0_variant4": 5,
                                "cells": {"E11": {"equity": 10_000.0},
                                          "E12": {"equity": 10_000.0}}})
        with pytest.raises(ValueError):                  # 셀 부분 결손
            variant4_from_dict({"t0_variant4": 5,
                                "cells": {"E22": {"equity": 10_000.0}}})

    def test_변형4_시가평가와_폐지는_그룹_전용이다(self):
        v = v4farm()
        warm(v, "XRP", atr=2.0, pc=110.0)
        v.cells["E23"].positions["XRP"] = v4pos()
        eqs = variant4_equities(v)
        assert set(eqs) == {"E22", "E23"}
        assert eqs["E23"] == pytest.approx(10_000 + 10.0 * 10.0), "마지막 종가 마크"
        assert eqs["E22"] == pytest.approx(10_000.0)
        fills = variant4_delist(v, "XRP")
        assert fills[0]["cell"] == "E23" and fills[0]["action"] == "force_exit"
        assert fills[0]["price"] == pytest.approx(110.0)
        assert fills[0]["cost"] == pytest.approx(10.0 * 110.0 * COST_SIDE), \
            "변형4 비용은 균일 편도 8bp (2단계 모델 미적용)"
        assert "XRP" in v.delisted
        assert not v.cells["E23"].positions


class TestV4Runner:
    """E22·E23 러너 — t0 동결·방화벽·다섯째 단계 격리·바이트 동일성 회귀."""

    OHLC5 = (100.0, 101.0, 99.0, 100.0, 10.0)

    def test_첫_호출은_t0_동결이고_추가_수집이_없다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        called = []
        monkeypatch.setattr(runner, "fetch_1h_paged",
                            lambda *a: called.append(a) or {})
        _run_variant4(object(), st, {}, {}, BASE, BASE)
        vc = st.variant4_cells
        assert called == [], "본 팜 지표 스냅숏 상속 — 별도 워밍업 수집 없음"
        assert vc["t0_variant4"] > 0
        assert vc["last_ts"] == BASE == st.last_ts, "본 팜과 정렬 (워밍업 무주문)"
        assert set(vc["cells"]) == {"E22", "E23"}
        assert vc["basket_b"] == st.basket_b
        assert vc["ind"]["BTC"]["atr1"] > 0, "본 팜 지표 상속"
        assert all(not c["positions"] for c in vc["cells"].values()), \
            "동결 실행은 재생하지 않는다"
        vh = pd.read_csv(tmp_path / VHIST)
        assert list(vh.columns) == VHIST_COLS, "e22·e23 스키마 이월"
        assert (tmp_path / VLEDGER).exists(), "헤더 선생성 (git add 함정 방지)"
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant4_cells"]["t0_variant4"] == vc["t0_variant4"]

    def test_본_팜_미가동이면_t0를_동결하지_않는다(self, tmp_path, monkeypatch):
        # 워밍업 fail-closed — 본 팜이 봉을 하나도 처리하지 않았으면 동결 지연
        monkeypatch.chdir(tmp_path)
        st = farm()
        assert st.last_ts == 0
        _run_variant4(object(), st, {}, {}, BASE, BASE)
        assert st.variant4_cells is None, "빈 지표로 t0 동결 금지"
        assert not (tmp_path / VLEDGER).exists()
        assert not (tmp_path / "logs/tracke_state.json").exists()

    def test_동결_실패는_t0를_영속화하지_않는다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        monkeypatch.setattr(runner, "_atomic_write",
                            lambda *a: (_ for _ in ()).throw(OSError("장애")))
        with pytest.raises(OSError):
            _run_variant4(object(), st, {}, {}, BASE, BASE)
        assert st.variant4_cells is None, "미기록 t0 롤백"

    def test_save_variant4는_t0_변경을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = v4farm()
        st.variant4_cells = variant4_to_dict(v)
        v.t0 = 2
        with pytest.raises(ValueError, match="write-once"):
            _save_variant4(st, v, [], BASE, 0)
        assert st.variant4_cells["t0_variant4"] == 1, "기존 기록 보존"
        assert not (tmp_path / VLEDGER).exists()

    def test_방화벽은_그룹_경계를_양방향으로_막는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v4 = v4farm()
        st.variant4_cells = variant4_to_dict(v4)
        row22 = dict(cell="E22", sym="BTC", strategy="BRK24R15",
                     bar_close=BASE + H1, action="enter", price=100.0, qty=1.0,
                     pnl=0.0, cost=0.08, direction=1, funding=0.0)
        with pytest.raises(ValueError, match="비공식 셀"):   # 본 원장 기록 금지
            _save_all(st, [row22], BASE, 1)
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        with pytest.raises(ValueError, match="비변형 행"):
            _save_variant(st, v1, [row22], BASE, 1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        with pytest.raises(ValueError, match="비변형2"):
            _save_variant2(st, v2, [row22], BASE, 1)
        v3 = new_variant3(st, ["BTC"], t0=3)
        st.variant3_cells = variant3_to_dict(v3)
        with pytest.raises(ValueError, match="비변형3"):
            _save_variant3(st, v3, [row22], BASE, 1)
        for bad in (dict(row22, cell="E01", strategy="BRK24"),
                    dict(row22, cell="E11", strategy="BRK24TP"),
                    dict(row22, cell="E19", strategy="BRK24"),
                    dict(row22, strategy="BRK24"),       # 유효 셀 × 타 전략
                    dict(row22, cell="E13")):            # 타 셀 × 유효 전략
            with pytest.raises(ValueError, match="비변형4"):
                _save_variant4(st, v4, [bad], BASE, 1)
        assert not (tmp_path / LEDGER).exists()
        assert not (tmp_path / VLEDGER).exists(), "오염 기록 대신 예외 (기록 0)"

    def test_종말_폐지_정리는_변형4_포지션을_방치하지_않는다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE + H1
        v = new_variant4(st, t0=1)
        warm(v, "XRP", atr=2.0, pc=100.0)
        v.cells["E23"].positions["XRP"] = v4pos()
        st.variant4_cells = variant4_to_dict(v)
        st.delisted.append("XRP")
        _finalize_variant4(st)
        vled = pd.read_csv(tmp_path / VLEDGER)
        row = vled.iloc[0]
        assert (row["cell"], row["action"]) == ("E23", "force_exit")
        assert row["price"] == pytest.approx(100.0), "변형4 상태의 마지막 처리 종가"
        assert st.variant4_cells["delisted"] == ["XRP"]
        assert not st.variant4_cells["cells"]["E23"]["positions"]
        _finalize_variant4(st)                           # 멱등 — 추가 기록 없음
        assert len(pd.read_csv(tmp_path / VLEDGER)) == len(vled)

    def test_변형4_실패는_본_v1_v2_v3_커밋을_막지_않고_재실행이_따라잡는다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        v1 = new_variant(st, t0=1)
        v1.last_ts = BASE - H1
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        v2.last_ts = BASE - H1
        warm2(v2, "BTC", atr=2.0, **GATE_PASS)
        st.variant2_cells = variant2_to_dict(v2)
        v3 = new_variant3(st, ["BTC"], t0=3)
        v3.last_ts = BASE - H1
        warm2(v3, "BTC", atr=2.0, **GATE_PASS)
        st.variant3_cells = variant3_to_dict(v3)
        v4 = new_variant4(st, t0=4)
        v4.last_ts = BASE - H1
        st.variant4_cells = variant4_to_dict(v4)
        data = {"BTC": {BASE: (100.0, 105.0, 100.0, 104.0, 20.0),
                        BASE + H1: (104.0, 104.5, 103.5, 104.0, 10.0),
                        BASE + 2 * H1: (89.0, 89.5, 88.5, 89.0, 10.0)}}
        fills: list = []
        for t in sorted(data["BTC"]):
            fills += step(st, {"BTC": BarE(t, *data["BTC"][t])})
        _save_all(st, fills, BASE + 2 * H1, 3)
        for fn in (_run_variant, _run_variant2, _run_variant3):
            _safe_variant(fn, None, st, data, {"BTC": {}}, BASE, BASE + 2 * H1)
        main_led = (tmp_path / LEDGER).read_bytes()
        main_state = (tmp_path / "logs/tracke_state.json").read_bytes()
        prior = ledger_lines_raw(tmp_path / VLEDGER,
                                 ("E11", "E12", "E13", "E14", "E15", "E16",
                                  "E17", "E18", "E19", "E20", "E21"))
        n_before = len(pd.read_csv(tmp_path / VLEDGER))
        orig_write = runner._atomic_write

        def boom(path, text):
            raise OSError("디스크 장애 주입")
        monkeypatch.setattr(runner, "_atomic_write", boom)
        _safe_variant(_run_variant4, None, st, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)              # 예외가 전파되면 실패
        assert (tmp_path / LEDGER).read_bytes() == main_led, "본 원장 불변"
        assert (tmp_path / "logs/tracke_state.json").read_bytes() == main_state
        assert not (tmp_path / ERR_MARK).exists(), "본 팜 중단 마커 오염 금지"
        n1 = len(pd.read_csv(tmp_path / VLEDGER))
        assert n1 > n_before, "변형4 원장은 상태 저장 전에 append 됨"
        assert ledger_lines_raw(tmp_path / VLEDGER,
                                ("E11", "E12", "E13", "E14", "E15", "E16",
                                 "E17", "E18", "E19", "E20", "E21")) == prior, \
            "선행 그룹 원장 행 바이트 불변"
        # 재실행 — 디스크 상태(뒤처진 variant4_cells)에서 유일키 멱등 따라잡기
        monkeypatch.setattr(runner, "_atomic_write", orig_write)
        st2 = FarmState.from_dict(
            json.loads((tmp_path / "logs/tracke_state.json").read_text()))
        _safe_variant(_run_variant4, None, st2, data, {"BTC": {}},
                      BASE, BASE + 2 * H1)
        assert len(pd.read_csv(tmp_path / VLEDGER)) == n1, "유일키 멱등 — 중복 0"
        assert st2.variant4_cells["last_ts"] == BASE + 2 * H1, "따라잡기 완료"
        assert st2.variant4_cells["t0_variant4"] == 4, "t0 불변 (write-once)"
        assert st2.variant_cells["last_ts"] == BASE + 2 * H1, "그룹1 무영향"
        assert st2.variant2_cells["last_ts"] == BASE + 2 * H1, "그룹2 무영향"
        assert st2.variant3_cells["last_ts"] == BASE + 2 * H1, "그룹3 무영향"

    def test_회귀_v4_활성이_본셀과_E11_E21_산출을_바꾸지_않는다(
            self, tmp_path, monkeypatch):
        # 요건 — v4 활성 재생(실체결 발생) 후에도 본 셀 산출물(원장·이력·상태
        # projection)과 E11~E21 서브상태·변형 원장 행이 바이트 단위로 같다.
        def scenario(root, with_v4: bool):
            monkeypatch.chdir(root)
            st = farm()
            warm(st, "BTC", atr=2.0)
            v1 = new_variant(st, t0=1)
            v2 = new_variant2(st, t0=2)
            x2 = v2.ind["BTC"]["x2"]
            x2["c2"] = list(GATE_PASS["c2"])
            x2["v2"] = list(GATE_PASS["v2"])
            x2["u14"], x2["d14"] = 1.0, 0.5
            st.variant2_cells = variant2_to_dict(v2)
            v3 = new_variant3(st, ["BTC", "ETH", "SOL"], t0=3)
            warm2(v3, "BTC", atr=2.0, **GATE_PASS)
            st.variant3_cells = variant3_to_dict(v3)
            v4 = None
            if with_v4:
                v4 = new_variant4(st, t0=4)
                st.variant4_cells = variant4_to_dict(v4)
            seq = [({"BTC": bar(0, 100, 105, 100, 104)}, None),
                   ({"BTC": bar(1, 104, 104.5, 103.5, 104)}, {"BTC": 0.0001}),
                   ({"BTC": bar(2, 89, 89.5, 88.5, 89)}, None)]
            fills: list = []
            f1: list = []
            f2: list = []
            f3: list = []
            f4: list = []
            for bars, fm in seq:
                fills += step(st, bars, fm)
                f1 += step_variant(v1, bars, fm)
                f2 += step_variant2(v2, bars, fm)
                f3 += step_variant3(v3, bars, fm)
                if v4 is not None:
                    f4 += step_variant4(v4, bars, fm)
            # 비교가 공허하지 않음 — 본·E11·E13·E19 경로 전부 실체결이 있다
            assert any(f["cell"] == "E01" for f in fills)
            assert any(f["cell"] == "E11" for f in f1)
            assert any(f["cell"] == "E13" for f in f2)
            assert any(f["cell"] == "E19" for f in f3)
            _save_all(st, fills, BASE + 2 * H1, 3)
            st.variant_cells = variant_to_dict(v1)
            _save_variant(st, v1, f1, BASE + 2 * H1, 3)
            _save_variant2(st, v2, f2, BASE + 2 * H1, 3)
            _save_variant3(st, v3, f3, BASE + 2 * H1, 3)
            if v4 is not None:
                _save_variant4(st, v4, f4, BASE + 2 * H1, 3)
            # 영속 상태 재적재 뒤의 본 재생도 동일해야 한다
            st2 = FarmState.from_dict(
                json.loads((root / "logs/tracke_state.json").read_text()))
            fills2 = step(st2, {"BTC": bar(3, 89, 90, 88.8, 89.5)})
            _save_all(st2, fills2, BASE + 3 * H1, 1)
            raw = json.loads((root / "logs/tracke_state.json").read_text())
            prior = ledger_lines_raw(root / VLEDGER, (
                "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18",
                "E19", "E20", "E21"))
            vcells = json.dumps(raw.pop("variant_cells"), sort_keys=True)
            v2cells = json.dumps(raw.pop("variant2_cells"), sort_keys=True)
            v3cells = json.dumps(raw.pop("variant3_cells"), sort_keys=True)
            raw.pop("variant4_cells")
            return ((root / LEDGER).read_bytes(),
                    (root / "logs/tracke_history.csv").read_bytes(),
                    json.dumps(raw, sort_keys=True), vcells, v2cells, v3cells,
                    prior, len(f4))
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        a_dir.mkdir(), b_dir.mkdir()
        ra = scenario(a_dir, with_v4=False)
        rb = scenario(b_dir, with_v4=True)
        assert ra[0] == rb[0], "본 원장 바이트 동일"
        assert ra[1] == rb[1], "본 이력 바이트 동일"
        assert ra[2] == rb[2], "본 상태(변형4 키 제외 projection) 동일"
        assert ra[3] == rb[3], "E11/E12 서브상태 동일"
        assert ra[4] == rb[4], "E13~E18 서브상태 동일"
        assert ra[5] == rb[5], "E19~E21 서브상태 동일"
        assert ra[6] == rb[6], "변형 원장 내 E11~E21 행 원문 바이트 동일"
        assert rb[7] > 0, "변형4가 실제 체결을 냈다 (공허한 비교 아님)"

    def test_통합_이력_행은_네_그룹_수치를_담는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        v3 = v3farm()
        st.variant3_cells = variant3_to_dict(v3)
        v4 = v4farm()
        st.variant4_cells = variant4_to_dict(v4)
        _save_variant(st, v1, [], BASE, 0)
        _save_variant2(st, v2, [], BASE, 0)
        _save_variant3(st, v3, [], BASE, 0)
        _save_variant4(st, v4, [], BASE, 0)              # keep-last — 마지막 행
        vh = pd.read_csv(tmp_path / VHIST)
        assert len(vh) == 1
        row = vh.iloc[0]
        for s in VCELLS + V2CELLS + V3CELLS + V4CELLS:
            assert row[s.cell.lower()] == pytest.approx(10_000.0)
        assert row["equity"] == pytest.approx(130_000.0), "전 변형 셀(13개) 합"


V5_LABEL = "볼린저 추매 변형 · 승률 제조 구조 시연 · 미검증 · 판정 권한 없음"


def v5farm(t0: int = 1) -> FarmState:
    """변형5 테스트 서브팜 (t0=1이면 전 봉 라이브)."""
    return new_variant5(farm(), t0)


def v5pos(**kw) -> FarmPos:
    """변형5 BBADD 보유 포지션 기본값 (롱 평균단가 100, 트랜치 1, 트리거 95)."""
    base = dict(d=1, u=10.0, e=100.0, stop=0.0, kind="BBADD", tgt=95.0,
                hold=1, risk_d=100.0 * V2_HEAT_FRAC)
    base.update(kw)
    return FarmPos(**base)


class TestV5Rules:
    """E24·E25 (BBADD) — 사전 고정: 볼린저 하단 진입·ATR 추매·4트랜치 캡·SMA20 전량."""

    def test_하단밴드_종가_이탈은_다음_봉_시가_트랜치1이다(self):
        # E15(BBMR) 테스트와 같은 수치 — 90 < 하단 ≈ 95.14. 신호봉 종가 체결
        # 금지·다음 봉 시가 체결 (뮤테이션: 신호봉 종가 체결로 바꾸면 실패)
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        step_variant5(v, {"BTC": bar(0, 100, 100, 90, 90)})
        cell = v.cells["E24"]
        assert "BTC" not in cell.positions, "신호봉 종가 체결 금지"
        assert cell.pending["BTC"] == {"kind": "BBADD", "d": 1, "ets": BASE + H1}
        fills = step_variant5(v, {"BTC": bar(1, 91, 92, 90.5, 91)})
        p = cell.positions["BTC"]
        assert p.e == pytest.approx(91.0), "다음 봉 시가 체결"
        assert p.d == 1 and p.kind == "BBADD" and p.stop == 0.0
        assert p.hold == 1, "트랜치 수 = 1 (진입)"
        # 트랜치 명목 = 체결 시점 equity × 1/12 (미결 #b)
        assert p.u == pytest.approx(V5_TRANCHE_FRAC * 10_000 / 91.0)
        assert p.u * p.e == pytest.approx(10_000 / 12.0)
        assert p.risk_d == pytest.approx(91.0 * V2_HEAT_FRAC), "heat 단가 = 명목 5%"
        # 추매 트리거가는 체결 시 동결 (미결 #a): 체결가 − 1.0 × ATR[체결봉−1]
        # (신호봉 TR=10 반영: a = 2 + (10−2)/24). 배수는 리터럴 1.0 로 검증
        # (뮤테이션: V5_ADD_ATR 계수를 바꾸면 실패)
        a = 2.0 + (10.0 - 2.0) / ATR1H_N
        assert p.tgt == pytest.approx(91.0 - 1.0 * a)
        e = next(f for f in fills if f["cell"] == "E24")
        assert (e["action"], e["strategy"]) == ("enter", "BBADD")
        assert e["price"] == pytest.approx(91.0)
        assert v.cells["E24"].cost == pytest.approx(p.u * 91.0 * COST_SIDE)

    def test_밴드는_봉내_터치가_아니라_종가_기준이다(self):
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        step_variant5(v, {"BTC": bar(0, 100, 100, 90, 100)})  # 저가만 관통
        assert "BTC" not in v.cells["E24"].pending

    def test_상단_이탈에도_숏은_없다_롱온리(self):
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        step_variant5(v, {"BTC": bar(0, 100, 130, 100, 125)})
        cell = v.cells["E24"]
        assert not cell.pending and not cell.positions

    def test_추매는_종가가_동결_트리거_이하일_때_다음_봉_시가다(self):
        # 트리거 95: 봉내 관통 무시(종가 기준), 경계 <= 포함 (미결 #d),
        # 체결은 다음 봉 시가 (뮤테이션: 신호봉 종가 체결로 바꾸면 실패)
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(tgt=95.0)
        step_variant5(v, {"BTC": bar(0, 96, 97, 94, 96)})     # 저가만 관통
        assert "BTC" not in cell.pending, "종가 기준 — 봉내 터치는 추매 아님"
        step_variant5(v, {"BTC": bar(1, 95.5, 96, 94.5, 95.0)})  # 경계 ==
        assert cell.pending["BTC"] == \
            {"kind": "BBADD_ADD", "d": 1, "ets": BASE + 2 * H1}
        fills = step_variant5(v, {"BTC": bar(2, 94.0, 95, 93, 94.5)})
        p = cell.positions["BTC"]
        add = next(f for f in fills if f["cell"] == "E24")
        assert add["action"] == "add"
        assert add["price"] == pytest.approx(94.0), "다음 봉 시가 (신호 종가 95 아님)"
        u_add = V5_TRANCHE_FRAC * 10_000 / 94.0
        assert add["qty"] == pytest.approx(u_add), "트랜치 명목 = 체결 시점 eq/12"
        assert p.u == pytest.approx(10.0 + u_add)
        assert p.e == pytest.approx((100.0 * 10.0 + 94.0 * u_add) / (10.0 + u_add)), \
            "평균 단가 갱신"
        assert p.hold == 2, "트랜치 수 증가"
        assert p.risk_d == pytest.approx(p.e * V2_HEAT_FRAC), "heat 단가 = 평균 단가 5%"
        # 다음 트리거 = 체결가 − 1.0 × ATR[체결봉−1] (TR 6.0 → 1.5 반영 — #a 동결)
        a = 2.0
        for tr in (6.0, 1.5):
            a += (tr - a) / ATR1H_N
        assert p.tgt == pytest.approx(94.0 - 1.0 * a), "리터럴 1.0 배수 (동결)"

    def test_트랜치_4개_소진_후엔_어떤_조건에도_추매가_없다(self):
        # 하드캡 (뮤테이션: 캡 제거·완화 시 실패) — 신호 단계 차단
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(hold=V5_TRANCHES, tgt=95.0)
        step_variant5(v, {"BTC": bar(0, 94, 95, 92, 93)})     # 93 <= 95 여도
        assert "BTC" not in cell.pending, "4트랜치 소진 — 추매 신호 없음"

    def test_체결_단계도_트랜치_캡을_이중_강제한다(self):
        # 신호 단계를 우회해 대기주문을 강제 주입해도 체결이 거부된다
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(hold=V5_TRANCHES, tgt=95.0)
        cell.pending["BTC"] = {"kind": "BBADD_ADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"BTC": bar(0, 94, 95, 92, 93)})
        assert not any(f["action"] == "add" for f in fills)
        assert cell.positions["BTC"].u == 10.0, "수량 불변 (체결 단계 거부)"

    def test_골든_경로_진입_추매3_캡_전량청산(self):
        # 라이브 시퀀스: enter → add×3 → 캡 차단(트리거 충족에도) → SMA20 전량.
        # 뮤테이션 민감: 캡(5번째 트랜치), 전량 청산(부분 청산), 시가 체결.
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        seq = [bar(0, 100, 100, 90, 90), bar(1, 91, 92, 88, 88.5),
               bar(2, 88, 89, 85, 85.5), bar(3, 85, 86, 82, 82.5),
               bar(4, 82, 83, 79, 79.0), bar(5, 79, 80, 77, 77.0),
               bar(6, 90, 101, 90, 100.5), bar(7, 100, 100.5, 99, 100.2)]
        fills: list = []
        for b in seq:
            fills += step_variant5(v, {"BTC": b})
        acts = [(f["action"], f["price"]) for f in fills if f["cell"] == "E24"]
        assert [a for a, _ in acts] == ["enter", "add", "add", "add",
                                        "exit_signal"]
        assert [p_ for _, p_ in acts[:4]] == [91.0, 88.0, 85.0, 82.0], \
            "전 트랜치 다음 봉 시가 체결"
        # bar4 종가 79.0 <= 직전 트리거(≈79.47)·bar5 종가 77.0 에도 5번째 없음
        assert sum(1 for a, _ in acts if a == "add") == V5_TRANCHES - 1, \
            "추매 최대 3회 (4트랜치 하드캡)"
        ex = next(f for f in fills if f["action"] == "exit_signal")
        qtys = [f["qty"] for f in fills
                if f["cell"] == "E24" and f["action"] in ("enter", "add")]
        assert ex["qty"] == pytest.approx(sum(qtys)), "전량 청산 (부분 아님)"
        assert ex["price"] == pytest.approx(100.0), "익절도 다음 봉 시가"
        assert "BTC" not in v.cells["E24"].positions

    def test_익절은_확정봉_종가_SMA20_이상_다음_봉_시가_전량이다(self):
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(u=25.0, e=97.0, hold=3, tgt=1.0)
        step_variant5(v, {"BTC": bar(0, 99, 101, 99, 100.5)})  # >= SMA20(≈100.03)
        assert cell.positions["BTC"].pending_exit == "signal"
        fills = step_variant5(v, {"BTC": bar(1, 100.2, 100.6, 99.9, 100.1)})
        assert fills[0]["action"] == "exit_signal"
        assert fills[0]["price"] == pytest.approx(100.2), "다음 봉 시가"
        assert fills[0]["qty"] == pytest.approx(25.0), "전량"
        assert "BTC" not in cell.positions

    def test_같은_봉_익절_추매_동시면_익절_우선이다(self):
        # 미결 #c — 트리거(200)가 종가 위여도 익절 신호가 서면 추매 미대기
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(tgt=200.0)
        step_variant5(v, {"BTC": bar(0, 99, 101, 99, 100.5)})
        assert cell.positions["BTC"].pending_exit == "signal"
        assert "BTC" not in cell.pending, "익절 우선 — 추매 대기 없음"

    def test_스탑이_없어_폭락에도_봉내_청산이_없다(self):
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(tgt=1.0)
        fills = step_variant5(v, {"BTC": bar(0, 60, 61, 50, 55)})
        assert fills == []
        assert "BTC" in cell.positions, "스탑 없음 — 신호 청산·추매만 존재"
        assert not cell.positions["BTC"].pending_exit

    def test_체결봉_종가도_추매·익절_신호_평가에_포함된다(self):
        # 미결 #e — 트랜치 1 체결봉 종가가 새 트리거 이하면 같은 봉 추매 대기
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        cell = v.cells["E24"]
        cell.pending["BTC"] = {"kind": "BBADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"BTC": bar(0, 100, 100, 95, 95)})
        p = cell.positions["BTC"]
        assert p.e == pytest.approx(100.0) and p.hold == 1
        assert p.tgt == pytest.approx(98.0), "체결가 100 − 1.0 × ATR 2.0"
        assert cell.pending["BTC"] == \
            {"kind": "BBADD_ADD", "d": 1, "ets": BASE + H1}, \
            "체결봉 종가 95 <= 98 — 같은 봉 추매 신호 (체결은 다음 봉 시가)"
        assert not any(f["action"].startswith("same_bar") for f in fills), \
            "stop=0.0 센티널·추매 트리거 tgt 오검 없음"
        # 체결봉 종가가 이미 SMA20 이상이면 같은 봉 익절 신호 (BBMR (ii) 관례)
        v2_ = v5farm()
        warm(v2_, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        cell2 = v2_.cells["E24"]
        cell2.pending["BTC"] = {"kind": "BBADD", "d": 1, "ets": BASE}
        step_variant5(v2_, {"BTC": bar(0, 95, 101, 94, 100.5)})
        assert cell2.positions["BTC"].pending_exit == "signal"

    def test_ATR_미형성이면_트랜치1도_추매도_무체결이다(self):
        # 트리거가 산정 불가 — fail-closed 무행동 (#a)
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        v.ind["BTC"]["atr1"] = None
        cell = v.cells["E24"]
        cell.pending["BTC"] = {"kind": "BBADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"BTC": bar(0, 95, 96, 94, 95)})
        assert "BTC" not in cell.positions
        assert not any(f["action"] == "enter" for f in fills)

    def test_일손실_정지는_트랜치1과_추매_체결을_모두_막는다(self):
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        warm(v, "ETH", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.day = "2026-01-01"                      # BASE 일자 — 리셋 방지
        cell.halted = True
        cell.pending["BTC"] = {"kind": "BBADD", "d": 1, "ets": BASE}
        cell.positions["ETH"] = v5pos(tgt=95.0)
        cell.pending["ETH"] = {"kind": "BBADD_ADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"BTC": bar(0, 95, 96, 94, 95),
                                  "ETH": bar(0, 94, 95, 93, 94)})
        assert "BTC" not in cell.positions, "진입 정지 (체결 시점 halted)"
        assert cell.positions["ETH"].u == 10.0, "추매도 정지"
        assert not any(f["action"] in ("enter", "add") for f in fills)

    def test_결측_봉은_변형5_신호와_대기주문을_정지시킨다(self):
        # 엔진 공통 fail-closed — 결측 봉은 대기주문 취소·관리 정지
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        warm(v, "ETH", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.positions["BTC"] = v5pos(tgt=95.0)
        cell.pending["BTC"] = {"kind": "BBADD_ADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"ETH": bar(0, 130, 130, 130, 130)})  # BTC 결측
        assert fills == [] and "BTC" not in cell.pending
        assert cell.positions["BTC"].u == 10.0, "무행동 (fail-closed)"

    def test_진입_신호는_E15_BBMR와_동일하고_명목은_4분의1이다(self):
        # 같은 종가 이력·같은 봉 — 신호·체결가 동일, 수량만 예산 4등분
        vb = new_variant2(farm(), t0=1)
        warm2(vb, "BTC", atr=2.0, hi=1000.0, lo=0.001, c2=[100.0] * 19,
              v2=[10.0] * 21)
        va = v5farm()
        warm(va, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        for b in (bar(0, 100, 100, 90, 90), bar(1, 91, 92, 90.5, 91)):
            step_variant2(vb, {"BTC": b})
            step_variant5(va, {"BTC": b})
        p15 = vb.cells["E15"].positions["BTC"]
        p24 = va.cells["E24"].positions["BTC"]
        assert p24.e == p15.e == pytest.approx(91.0), "같은 신호·같은 시가 체결"
        assert p24.u == pytest.approx(p15.u / V5_TRANCHES), "명목 1/3 의 4등분"

    def test_트랜치_heat_기여는_명목x일손실한도이고_만재는_캡_이내다(self):
        # 2심볼 4트랜치 만재 상당 + 3번째 심볼 4트랜치 — 12트랜치 ≈ 5% <= 6%
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        for s in ("ETH", "SOL"):                     # 만재 명목 eq/3, hold 4
            cell.positions[s] = v5pos(u=33.3333, e=100.0, hold=V5_TRANCHES,
                                      tgt=1.0)
        cell.positions["BTC"] = v5pos(u=25.0, e=100.0, hold=3, tgt=95.0)
        cell.pending["BTC"] = {"kind": "BBADD_ADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"BTC": bar(0, 94, 95, 93, 94)})
        add = next(f for f in fills if f["action"] == "add")
        assert add["qty"] == pytest.approx(V5_TRANCHE_FRAC * 10_000 / 94.0), \
            "만재 상당 heat(≈5%)에서도 마지막 트랜치 허용 (설계 이내)"
        heat = sum(p.u * p.risk_d for p in cell.positions.values())
        assert heat <= HEAT_CAP * cell.equity * (1 + 1e-9)

    def test_잔존_heat가_캡을_넘기면_추매_차단이다(self):
        v = v5farm()
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 24)
        cell = v.cells["E24"]
        cell.equity = 5_200.0                        # 캡 = 312.0
        for s in ("ETH", "SOL"):                     # 잔존 heat 333.33 (과거 고자본)
            cell.positions[s] = v5pos(u=33.3333, e=100.0, hold=V5_TRANCHES,
                                      tgt=1.0)
        cell.positions["BTC"] = v5pos(u=10.0, e=100.0, hold=1, tgt=95.0)
        cell.pending["BTC"] = {"kind": "BBADD_ADD", "d": 1, "ets": BASE}
        fills = step_variant5(v, {"BTC": bar(0, 94, 95, 93, 94)})
        assert not any(f["action"] == "add" for f in fills), "heat 캡 차단"
        assert cell.positions["BTC"].u == 10.0

    def test_변형5는_t0_variant5_이전_워밍업_무주문이다(self):
        v = new_variant5(farm(), t0=BASE + 100 * H1)
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        fills = step_variant5(v, {"BTC": bar(0, 100, 100, 90, 90)})  # 명백 이탈
        assert fills == []
        for c in v.cells.values():
            assert not c.positions and not c.pending

    def test_변형5_동결_상수와_라벨(self):
        assert V5_TRANCHES == 4
        assert V5_ADD_ATR == 1.0
        assert V5_TRANCHE_FRAC == pytest.approx(V2_NOTIONAL_FRAC / 4)
        assert [s.cell for s in V5CELLS] == ["E24", "E25"]
        assert all(s.strategy == "BBADD" and s.n == 0 for s in V5CELLS)
        assert [s.basket for s in V5CELLS] == ["A", "B"]
        assert V5LABELS["E24"] == V5LABELS["E25"] == V5_LABEL
        # MAX_POS 3·균일 16bp — 본 셀 공통 파라미터 그대로 (스탑 부재라 risk 미사용)
        for s in V5CELLS:
            assert (s.risk, s.max_pos, s.cost_model, s.heat_clamp) == \
                (0.02, 3, "", False)


class TestV5State:
    """변형5 서브상태 — variant5_cells 키, t0_variant5 write-once, 그룹 분리."""

    def test_new_variant5는_동결_바스켓_재사용과_지표_깊은_분리다(self):
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        v = new_variant5(st, t0=123)
        assert v.t0 == 123
        assert v.basket_b == st.basket_b == ["XRP", "DOGE", "ADA"]
        assert v.last_ts == st.last_ts, "t0_variant5 이전 봉 재생 구조적 차단"
        assert v.ind == st.ind, "시장 전용 지표 스냅숏 상속 (값 동일)"
        assert set(v.cells) == {"E24", "E25"}
        assert all(c.equity == 10_000.0 for c in v.cells.values())
        v.ind["BTC"]["atr1"] = 999.0                 # 변형 쪽 변이가
        v.ind["BTC"]["cl"].append(1.0)
        assert st.ind["BTC"]["atr1"] != 999.0, "본 팜 지표에 새면 안 된다 (깊은 복사)"
        assert len(st.ind["BTC"]["cl"]) != len(v.ind["BTC"]["cl"])

    def test_변형5_직렬화는_t0_variant5_명명으로_왕복된다(self):
        v = new_variant5(farm(), t0=BASE)
        warm(v, "BTC", atr=2.0, hi=1000.0, lo=0.001, closes=[100.0] * 19)
        step_variant5(v, {"BTC": bar(0, 100, 100, 90, 90)})
        step_variant5(v, {"BTC": bar(1, 91, 92, 90.5, 91)})    # 포지션 포함
        d = variant5_to_dict(v)
        assert "t0_variant5" in d and "t0" not in d
        assert all(k not in d for k in ("variant_cells", "variant2_cells",
                                        "variant3_cells", "variant4_cells",
                                        "variant5_cells"))
        v2 = variant5_from_dict(json.loads(json.dumps(d, default=float)))
        assert json.dumps(variant5_to_dict(v2), sort_keys=True, default=float) \
            == json.dumps(d, sort_keys=True, default=float)
        p = v2.cells["E24"].positions["BTC"]
        assert p.kind == "BBADD" and p.hold == 1
        assert p.tgt == pytest.approx(91.0 - (2.0 + 8.0 / ATR1H_N)), \
            "동결 트리거가 왕복 보존 (상태에 보존 요건)"

    def test_기존_그룹_직렬화에_variant5_키가_새지_않는다(self):
        # 바이트 동일성의 전제 — v1~v4 서브상태 JSON 에 새 키가 생기면
        # 기존 그룹 상태 바이트가 변한다 (_vgroup_to_dict 의 pop 회귀)
        st = farm()
        for d in (variant_to_dict(new_variant(st, t0=1)),
                  variant2_to_dict(new_variant2(st, t0=2)),
                  variant3_to_dict(new_variant3(st, ["BTC"], t0=3)),
                  variant4_to_dict(new_variant4(st, t0=4)),
                  variant5_to_dict(new_variant5(st, t0=5))):
            assert "variant5_cells" not in d
            assert "t0" not in d

    def test_본_상태는_variant5_cells를_불투명하게_왕복_보존한다(self):
        st = farm()
        assert st.variant5_cells is None, "미초기화 기본값 None (초기화 대상)"
        st.variant5_cells = variant5_to_dict(new_variant5(st, t0=BASE))
        rt = FarmState.from_dict(json.loads(json.dumps(st.to_dict(), default=float)))
        assert rt.variant5_cells == st.variant5_cells
        assert json.dumps(rt.to_dict(), sort_keys=True, default=float) == \
            json.dumps(st.to_dict(), sort_keys=True, default=float)

    def test_손상된_변형5_상태는_fail_closed다(self):
        with pytest.raises(ValueError):
            variant5_from_dict(None)
        with pytest.raises(ValueError):
            variant5_from_dict({})
        with pytest.raises(ValueError):
            variant5_from_dict({"t0_variant5": 0, "cells": {}})
        with pytest.raises(ValueError):                  # E22 구성은 변형5가 아니다
            variant5_from_dict({"t0_variant5": 5,
                                "cells": {"E22": {"equity": 10_000.0},
                                          "E23": {"equity": 10_000.0}}})
        with pytest.raises(ValueError):                  # 셀 부분 결손
            variant5_from_dict({"t0_variant5": 5,
                                "cells": {"E24": {"equity": 10_000.0}}})

    def test_변형5_시가평가와_폐지는_그룹_전용이다(self):
        v = v5farm()
        warm(v, "XRP", atr=2.0, pc=110.0)
        v.cells["E25"].positions["XRP"] = v5pos()
        eqs = variant5_equities(v)
        assert set(eqs) == {"E24", "E25"}
        assert eqs["E25"] == pytest.approx(10_000 + 10.0 * 10.0), "마지막 종가 마크"
        assert eqs["E24"] == pytest.approx(10_000.0)
        fills = variant5_delist(v, "XRP")
        assert fills[0]["cell"] == "E25" and fills[0]["action"] == "force_exit"
        assert fills[0]["price"] == pytest.approx(110.0)
        assert fills[0]["cost"] == pytest.approx(10.0 * 110.0 * COST_SIDE), \
            "변형5 비용은 균일 편도 8bp (2단계 모델 미적용)"
        assert "XRP" in v.delisted
        assert not v.cells["E25"].positions


class TestV5Runner:
    """E24·E25 러너 — t0 동결·방화벽·여섯째 단계 격리·바이트 동일성 회귀."""

    def test_첫_호출은_t0_동결이고_추가_수집이_없다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0)
        step(st, {"BTC": bar(0, 100, 101, 99, 100)})
        called = []
        monkeypatch.setattr(runner, "fetch_1h_paged",
                            lambda *a: called.append(a) or {})
        _run_variant5(object(), st, {}, {}, BASE, BASE)
        vc = st.variant5_cells
        assert called == [], "본 팜 지표 스냅숏 상속 — 별도 워밍업 수집 없음"
        assert vc["t0_variant5"] > 0
        assert vc["last_ts"] == BASE == st.last_ts, "본 팜과 정렬 (워밍업 무주문)"
        assert set(vc["cells"]) == {"E24", "E25"}
        assert vc["basket_b"] == st.basket_b
        assert vc["ind"]["BTC"]["atr1"] > 0, "본 팜 지표 상속"
        assert all(not c["positions"] for c in vc["cells"].values()), \
            "동결 실행은 재생하지 않는다"
        vh = pd.read_csv(tmp_path / VHIST)
        assert list(vh.columns) == VHIST_COLS, "e24·e25 스키마 이월"
        assert (tmp_path / VLEDGER).exists(), "헤더 선생성 (git add 함정 방지)"
        raw = json.loads((tmp_path / "logs/tracke_state.json").read_text())
        assert raw["variant5_cells"]["t0_variant5"] == vc["t0_variant5"]

    def test_본_팜_미가동이면_t0를_동결하지_않는다(self, tmp_path, monkeypatch):
        # 워밍업 fail-closed — 본 팜이 봉을 하나도 처리하지 않았으면 동결 지연
        monkeypatch.chdir(tmp_path)
        st = farm()
        assert st.last_ts == 0
        _run_variant5(object(), st, {}, {}, BASE, BASE)
        assert st.variant5_cells is None, "빈 지표로 t0 동결 금지"
        assert not (tmp_path / VLEDGER).exists()
        assert not (tmp_path / "logs/tracke_state.json").exists()

    def test_동결_실패는_t0를_영속화하지_않는다(self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE
        monkeypatch.setattr(runner, "_atomic_write",
                            lambda *a: (_ for _ in ()).throw(OSError("장애")))
        with pytest.raises(OSError):
            _run_variant5(object(), st, {}, {}, BASE, BASE)
        assert st.variant5_cells is None, "미기록 t0 롤백"

    def test_save_variant5는_t0_변경을_거부한다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v = v5farm()
        st.variant5_cells = variant5_to_dict(v)
        v.t0 = 2
        with pytest.raises(ValueError, match="write-once"):
            _save_variant5(st, v, [], BASE, 0)
        assert st.variant5_cells["t0_variant5"] == 1, "기존 기록 보존"
        assert not (tmp_path / VLEDGER).exists()

    def test_방화벽은_그룹_경계를_양방향으로_막는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v5 = v5farm()
        st.variant5_cells = variant5_to_dict(v5)
        row24 = dict(cell="E24", sym="BTC", strategy="BBADD",
                     bar_close=BASE + H1, action="enter", price=100.0, qty=1.0,
                     pnl=0.0, cost=0.08, direction=1, funding=0.0)
        with pytest.raises(ValueError, match="비공식 셀"):   # 본 원장 기록 금지
            _save_all(st, [row24], BASE, 1)
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        with pytest.raises(ValueError, match="비변형 행"):
            _save_variant(st, v1, [row24], BASE, 1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        with pytest.raises(ValueError, match="비변형2"):
            _save_variant2(st, v2, [row24], BASE, 1)
        v3 = new_variant3(st, ["BTC"], t0=3)
        st.variant3_cells = variant3_to_dict(v3)
        with pytest.raises(ValueError, match="비변형3"):
            _save_variant3(st, v3, [row24], BASE, 1)
        v4 = new_variant4(st, t0=4)
        st.variant4_cells = variant4_to_dict(v4)
        with pytest.raises(ValueError, match="비변형4"):
            _save_variant4(st, v4, [row24], BASE, 1)
        for bad in (dict(row24, cell="E01", strategy="BRK24"),
                    dict(row24, cell="E11", strategy="BRK24TP"),
                    dict(row24, cell="E22", strategy="BRK24R15"),
                    dict(row24, strategy="BBMR"),        # 유효 셀 × 타 전략
                    dict(row24, cell="E15")):            # 타 셀 × 유효 전략
            with pytest.raises(ValueError, match="비변형5"):
                _save_variant5(st, v5, [bad], BASE, 1)
        assert not (tmp_path / LEDGER).exists()
        assert not (tmp_path / VLEDGER).exists(), "오염 기록 대신 예외 (기록 0)"

    def test_종말_폐지_정리는_변형5_포지션을_방치하지_않는다(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        st.last_ts = BASE + H1
        v = new_variant5(st, t0=1)
        warm(v, "XRP", atr=2.0, pc=100.0)
        v.cells["E25"].positions["XRP"] = v5pos()
        st.variant5_cells = variant5_to_dict(v)
        st.delisted.append("XRP")
        _finalize_variant5(st)
        vled = pd.read_csv(tmp_path / VLEDGER)
        row = vled.iloc[0]
        assert (row["cell"], row["action"]) == ("E25", "force_exit")
        assert row["price"] == pytest.approx(100.0), "변형5 상태의 마지막 처리 종가"
        assert st.variant5_cells["delisted"] == ["XRP"]
        assert not st.variant5_cells["cells"]["E25"]["positions"]
        _finalize_variant5(st)                           # 멱등 — 추가 기록 없음
        assert len(pd.read_csv(tmp_path / VLEDGER)) == len(vled)

    def test_변형5_실패는_본_v1_v4_커밋을_막지_않고_재실행이_따라잡는다(
            self, tmp_path, monkeypatch):
        import carrybot.aggressive.scalp_farm_runner as runner
        monkeypatch.chdir(tmp_path)
        st = farm()
        warm(st, "BTC", atr=2.0, closes=[100.0] * 24)
        v1 = new_variant(st, t0=1)
        v1.last_ts = BASE - H1
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        v2.last_ts = BASE - H1
        warm2(v2, "BTC", atr=2.0, **GATE_PASS)
        st.variant2_cells = variant2_to_dict(v2)
        v3 = new_variant3(st, ["BTC"], t0=3)
        v3.last_ts = BASE - H1
        warm2(v3, "BTC", atr=2.0, **GATE_PASS)
        st.variant3_cells = variant3_to_dict(v3)
        v4 = new_variant4(st, t0=4)
        v4.last_ts = BASE - H1
        st.variant4_cells = variant4_to_dict(v4)
        v5 = new_variant5(st, t0=5)
        v5.last_ts = BASE - H1
        st.variant5_cells = variant5_to_dict(v5)
        data = {"BTC": {BASE: (100.0, 105.0, 100.0, 104.0, 20.0),
                        BASE + H1: (104.0, 104.5, 103.5, 104.0, 10.0),
                        BASE + 2 * H1: (89.0, 89.5, 88.5, 89.0, 10.0),
                        BASE + 3 * H1: (89.0, 90.0, 88.8, 89.5, 10.0)}}
        fills: list = []
        for t in sorted(data["BTC"]):
            fills += step(st, {"BTC": BarE(t, *data["BTC"][t])})
        _save_all(st, fills, BASE + 3 * H1, 4)
        for fn in (_run_variant, _run_variant2, _run_variant3, _run_variant4):
            _safe_variant(fn, None, st, data, {"BTC": {}}, BASE, BASE + 3 * H1)
        main_led = (tmp_path / LEDGER).read_bytes()
        main_state = (tmp_path / "logs/tracke_state.json").read_bytes()
        prior = ledger_lines_raw(tmp_path / VLEDGER,
                                 ("E11", "E12", "E13", "E14", "E15", "E16",
                                  "E17", "E18", "E19", "E20", "E21", "E22",
                                  "E23"))
        n_before = len(pd.read_csv(tmp_path / VLEDGER))
        orig_write = runner._atomic_write

        def boom(path, text):
            raise OSError("디스크 장애 주입")
        monkeypatch.setattr(runner, "_atomic_write", boom)
        _safe_variant(_run_variant5, None, st, data, {"BTC": {}},
                      BASE, BASE + 3 * H1)              # 예외가 전파되면 실패
        assert (tmp_path / LEDGER).read_bytes() == main_led, "본 원장 불변"
        assert (tmp_path / "logs/tracke_state.json").read_bytes() == main_state
        assert not (tmp_path / ERR_MARK).exists(), "본 팜 중단 마커 오염 금지"
        n1 = len(pd.read_csv(tmp_path / VLEDGER))
        assert n1 > n_before, "변형5 원장은 상태 저장 전에 append 됨 (E24 진입)"
        assert ledger_lines_raw(tmp_path / VLEDGER,
                                ("E11", "E12", "E13", "E14", "E15", "E16",
                                 "E17", "E18", "E19", "E20", "E21", "E22",
                                 "E23")) == prior, \
            "선행 그룹 원장 행 바이트 불변"
        # 재실행 — 디스크 상태(뒤처진 variant5_cells)에서 유일키 멱등 따라잡기
        monkeypatch.setattr(runner, "_atomic_write", orig_write)
        st2 = FarmState.from_dict(
            json.loads((tmp_path / "logs/tracke_state.json").read_text()))
        _safe_variant(_run_variant5, None, st2, data, {"BTC": {}},
                      BASE, BASE + 3 * H1)
        assert len(pd.read_csv(tmp_path / VLEDGER)) == n1, "유일키 멱등 — 중복 0"
        assert st2.variant5_cells["last_ts"] == BASE + 3 * H1, "따라잡기 완료"
        assert st2.variant5_cells["t0_variant5"] == 5, "t0 불변 (write-once)"
        assert st2.variant_cells["last_ts"] == BASE + 3 * H1, "그룹1 무영향"
        assert st2.variant2_cells["last_ts"] == BASE + 3 * H1, "그룹2 무영향"
        assert st2.variant3_cells["last_ts"] == BASE + 3 * H1, "그룹3 무영향"
        assert st2.variant4_cells["last_ts"] == BASE + 3 * H1, "그룹4 무영향"

    def test_회귀_v5_활성이_본셀과_E11_E23_산출을_바꾸지_않는다(
            self, tmp_path, monkeypatch):
        # 요건 — v5 활성 재생(실체결 발생) 후에도 본 셀 산출물(원장·이력·상태
        # projection)과 E11~E23 서브상태·변형 원장 행이 바이트 단위로 같다.
        def scenario(root, with_v5: bool):
            monkeypatch.chdir(root)
            st = farm()
            warm(st, "BTC", atr=2.0, closes=[100.0] * 24)
            v1 = new_variant(st, t0=1)
            v2 = new_variant2(st, t0=2)
            x2 = v2.ind["BTC"]["x2"]
            x2["c2"] = list(GATE_PASS["c2"])
            x2["v2"] = list(GATE_PASS["v2"])
            x2["u14"], x2["d14"] = 1.0, 0.5
            st.variant2_cells = variant2_to_dict(v2)
            v3 = new_variant3(st, ["BTC", "ETH", "SOL"], t0=3)
            warm2(v3, "BTC", atr=2.0, **GATE_PASS)
            st.variant3_cells = variant3_to_dict(v3)
            v4 = new_variant4(st, t0=4)
            st.variant4_cells = variant4_to_dict(v4)
            v5 = None
            if with_v5:
                v5 = new_variant5(st, t0=5)
                st.variant5_cells = variant5_to_dict(v5)
            seq = [({"BTC": bar(0, 100, 105, 100, 104)}, None),
                   ({"BTC": bar(1, 104, 104.5, 103.5, 104)}, {"BTC": 0.0001}),
                   ({"BTC": bar(2, 89, 89.5, 88.5, 89)}, None),
                   ({"BTC": bar(3, 89, 90, 88.8, 89.5)}, None)]
            fills: list = []
            f1: list = []
            f2: list = []
            f3: list = []
            f4: list = []
            f5: list = []
            for bars, fm in seq:
                fills += step(st, bars, fm)
                f1 += step_variant(v1, bars, fm)
                f2 += step_variant2(v2, bars, fm)
                f3 += step_variant3(v3, bars, fm)
                f4 += step_variant4(v4, bars, fm)
                if v5 is not None:
                    f5 += step_variant5(v5, bars, fm)
            # 비교가 공허하지 않음 — 본·E11·E13·E19·E22 경로 전부 실체결이 있다
            assert any(f["cell"] == "E01" for f in fills)
            assert any(f["cell"] == "E11" for f in f1)
            assert any(f["cell"] == "E13" for f in f2)
            assert any(f["cell"] == "E19" for f in f3)
            assert any(f["cell"] == "E22" for f in f4)
            _save_all(st, fills, BASE + 3 * H1, 4)
            st.variant_cells = variant_to_dict(v1)
            _save_variant(st, v1, f1, BASE + 3 * H1, 4)
            _save_variant2(st, v2, f2, BASE + 3 * H1, 4)
            _save_variant3(st, v3, f3, BASE + 3 * H1, 4)
            _save_variant4(st, v4, f4, BASE + 3 * H1, 4)
            if v5 is not None:
                _save_variant5(st, v5, f5, BASE + 3 * H1, 4)
            # 영속 상태 재적재 뒤의 본 재생도 동일해야 한다
            st2 = FarmState.from_dict(
                json.loads((root / "logs/tracke_state.json").read_text()))
            fills2 = step(st2, {"BTC": bar(4, 89.5, 90.5, 89.2, 90)})
            _save_all(st2, fills2, BASE + 4 * H1, 1)
            raw = json.loads((root / "logs/tracke_state.json").read_text())
            prior = ledger_lines_raw(root / VLEDGER, (
                "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18",
                "E19", "E20", "E21", "E22", "E23"))
            vcells = json.dumps(raw.pop("variant_cells"), sort_keys=True)
            v2cells = json.dumps(raw.pop("variant2_cells"), sort_keys=True)
            v3cells = json.dumps(raw.pop("variant3_cells"), sort_keys=True)
            v4cells = json.dumps(raw.pop("variant4_cells"), sort_keys=True)
            raw.pop("variant5_cells")
            return ((root / LEDGER).read_bytes(),
                    (root / "logs/tracke_history.csv").read_bytes(),
                    json.dumps(raw, sort_keys=True), vcells, v2cells, v3cells,
                    v4cells, prior, len(f5))
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        a_dir.mkdir(), b_dir.mkdir()
        ra = scenario(a_dir, with_v5=False)
        rb = scenario(b_dir, with_v5=True)
        assert ra[0] == rb[0], "본 원장 바이트 동일"
        assert ra[1] == rb[1], "본 이력 바이트 동일"
        assert ra[2] == rb[2], "본 상태(변형5 키 제외 projection) 동일"
        assert ra[3] == rb[3], "E11/E12 서브상태 동일"
        assert ra[4] == rb[4], "E13~E18 서브상태 동일"
        assert ra[5] == rb[5], "E19~E21 서브상태 동일"
        assert ra[6] == rb[6], "E22/E23 서브상태 동일"
        assert ra[7] == rb[7], "변형 원장 내 E11~E23 행 원문 바이트 동일"
        assert rb[8] > 0, "변형5가 실제 체결을 냈다 (공허한 비교 아님)"

    def test_통합_이력_행은_다섯_그룹_수치를_담는다(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = farm()
        v1 = new_variant(st, t0=1)
        st.variant_cells = variant_to_dict(v1)
        v2 = new_variant2(st, t0=2)
        st.variant2_cells = variant2_to_dict(v2)
        v3 = v3farm()
        st.variant3_cells = variant3_to_dict(v3)
        v4 = v4farm()
        st.variant4_cells = variant4_to_dict(v4)
        v5 = v5farm()
        st.variant5_cells = variant5_to_dict(v5)
        _save_variant(st, v1, [], BASE, 0)
        _save_variant2(st, v2, [], BASE, 0)
        _save_variant3(st, v3, [], BASE, 0)
        _save_variant4(st, v4, [], BASE, 0)
        _save_variant5(st, v5, [], BASE, 0)              # keep-last — 마지막 행
        vh = pd.read_csv(tmp_path / VHIST)
        assert len(vh) == 1
        row = vh.iloc[0]
        for s in VCELLS + V2CELLS + V3CELLS + V4CELLS + V5CELLS:
            assert row[s.cell.lower()] == pytest.approx(10_000.0)
        assert row["equity"] == pytest.approx(150_000.0), "전 변형 셀(15개) 합"
