"""AVGDOWN-FEES-2026-09-01 엔진·판정 단위 테스트 — 전부 합성 데이터.

실데이터·본 격자 미접촉. 인과성 테스트 원칙: 규약 위반 시 실패하도록 기대값을
수기 산술로 고정한다 (엔진 로직 재구현 금지 — 분석적 기대값만).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


af = _load("af_test", "lab/avgdown_fees.py")
ads = af.asw                                         # 동결 엔진 (읽기 전용)

ZERO_FUND = pd.Series(dtype=float)
MAKER = 0.0002


def _flat_df(n, bars):
    return af._flat_df(n, bars)


def _sig_bar(j: int) -> dict:
    """봉 j 를 E1 신호봉(종가 90 < BB 하단)으로 만드는 오버라이드."""
    return {j: (100.0, 100.0, 90.0, 90.0)}


def test_scenario_constants_frozen():
    """시나리오 상수 동결 — 값이 바뀌면 사전등록 위반."""
    assert tuple(af.SCENARIOS) == ("a", "b", "c")
    assert af.SCENARIOS["a"] == {**af.SCENARIOS["a"],
                                 "cost_side": 0.00055, "fill_model": "market"}
    assert af.SCENARIOS["b"]["cost_side"] == 0.0002
    assert af.SCENARIOS["b"]["fill_model"] == "market"
    assert af.SCENARIOS["c"]["cost_side"] == 0.0002
    assert af.SCENARIOS["c"]["fill_model"] == "limit"
    assert af.COST_SIDE_FROZEN == 0.0008
    assert af.SEED == 20260901 and af.N_TRIALS == 1248 and af.N_DAYS == 2056


def test_market_8bp_bit_identical_to_frozen_sleeve():
    """편도 8bp·시장가 파라미터화 == 동결 simulate_sleeve — 합성 비트 동일.

    수수료 파라미터화가 기존 경로의 산술을 단 1비트도 바꾸지 않았음을 증명한다
    (실데이터 대조는 셀프테스트 (i) 이 전체 격자로 수행).
    """
    df = af._synthetic_walk(7)
    trials = [t for t in ads.enumerate_trials() if t.tf == "1h"][::150]
    assert len(trials) >= 4
    ta = ads.trial_arrays(trials)
    ref = ads.simulate_sleeve(df, ZERO_FUND, ta)
    got = af.simulate_sleeve_fees(df, ZERO_FUND, ta, 0.0008, 0.0008,
                                  fill_model="market")
    assert np.array_equal(ref["day_eq"], got["day_eq"])
    assert np.array_equal(ref["final_eq"], got["final_eq"])
    assert np.array_equal(ref["n_trades"], got["n_trades"])
    assert np.array_equal(ref["n_wins"], got["n_wins"])


def test_warmup_blocks_orders_limit():
    """워밍업(100봉) 내 신호는 지정가 모델에서도 주문을 만들지 않는다."""
    df = _flat_df(130, {**_sig_bar(50), 51: (95.0, 96.0, 80.0, 100.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    res = af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                                  fill_model="limit")
    assert int(res["n_fills"][0]) == 0
    assert float(res["final_eq"][0]) == 1.0


def test_limit_entry_pierce_fills_at_limit():
    """관통(low < 지정가) → 체결가 = 지정가 (시가가 위에 있을 때)."""
    log: list = []
    df = _flat_df(140, {**_sig_bar(130), 131: (95.0, 96.0, 89.0, 95.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                            fill_model="limit", fill_log=log)
    assert [e for e in log if e[0] == "fill"] == [("fill", 131, 90.0)]


def test_limit_entry_equal_price_no_fill_and_no_carryover():
    """동가(low == 지정가) 미체결 + 신호 소멸(이월 금지).

    봉 131 low=90 == 지정가 → 미체결. 봉 132 low=89 는 '옛 지정가' 아래지만
    새 확정봉(종가 100)은 신호가 아니므로 체결이 없어야 한다 — 주문 이월이
    구현돼 있으면 이 테스트가 실패한다.
    """
    df = _flat_df(140, {**_sig_bar(130),
                        131: (95.0, 96.0, 90.0, 100.0),
                        132: (95.0, 96.0, 89.0, 100.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    res = af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                                  fill_model="limit")
    assert int(res["n_fills"][0]) == 0
    assert float(res["final_eq"][0]) == 1.0


def test_limit_entry_gap_fills_at_open():
    """갭 하락 개장(시가 < 지정가) → 체결가 = 시가 (유리한 쪽)."""
    log: list = []
    df = _flat_df(140, {**_sig_bar(130), 131: (88.0, 89.0, 87.0, 95.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                            fill_model="limit", fill_log=log)
    assert [e for e in log if e[0] == "fill"] == [("fill", 131, 88.0)]


def test_limit_tp_strict_and_reevaluate():
    """익절 대칭: high == 지정가 미체결(엄격 >), 다음 확정봉 재평가 후 체결.

    수기 산술: 진입 90 체결, 익절 지정가 100 — 봉 133·134 는 high==100 미체결,
    봉 135 high 101 > 100 → 체결가 = max(시가 99.5, 100) = 100.
    """
    log: list = []
    df = _flat_df(140, {**_sig_bar(130),
                        131: (95.0, 96.0, 89.0, 95.0),
                        132: (100.0, 100.0, 96.0, 100.0),
                        133: (99.0, 100.0, 98.0, 100.0),
                        134: (99.5, 100.0, 99.0, 100.0),
                        135: (99.5, 101.0, 99.0, 100.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", None, "1h")])
    res = af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                                  fill_model="limit", fill_log=log)
    assert [e for e in log if e[0] == "exit"] == [("exit", 135, 100.0)]
    u = (1.0 / 12.0) / 90.0
    expect = (1.0 - u * 90.0 * MAKER) \
        + (u * 100.0 - u * 90.0 - u * 100.0 * MAKER)
    assert abs(float(res["final_eq"][0]) - expect) < 1e-15


def test_limit_keeps_frozen_stop_model_gap_worse():
    """시나리오 (c)에서도 손절은 원 스윕 봉내 스탑 모델 — min(시가, 레벨) 악화.

    신호봉 TR=10 → ATR24=10/24, 진입 90 (관통 체결), 손절 동결 = 87.5.
    갭: 다음 봉 시가 85 < 87.5 → 체결 85 (유리 쪽 max 모델이면 실패한다).
    """
    j = 110
    df = _flat_df(160, {**_sig_bar(j),
                        j + 1: (90.0, 90.0, 89.9, 89.9),
                        j + 2: (85.0, 100.0, 80.0, 100.0)})
    ta = ads.trial_arrays([ads.Trial("E1", 0, None, 0, "MID", 6.0, "1h")])
    res = af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                                  fill_model="limit")
    assert int(res["n_trades"][0]) == 1
    u = (1.0 / 12.0) / 90.0                          # 지정가 90 == low 89.9 관통
    expect = 1.0 - 90.0 * u * MAKER + (u * 85.0 - 90.0 * u - u * 85.0 * MAKER)
    assert abs(float(res["final_eq"][0]) - expect) < 1e-12
    assert int(res["time_viol"][0]) == 0


def test_lookahead_control_differs_both_models():
    """같은 봉 신호 평가(위반 대조군)는 market·limit 모두 결과가 달라야 한다."""
    df = af._synthetic_walk(11, n=2500)
    ta = ads.trial_arrays([ads.Trial("E1", 0, 1.0, 3, "MID", None, "1h")])
    for fm in ("market", "limit"):
        a = af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                                    fill_model=fm, causal=True)
        b = af.simulate_sleeve_fees(df, ZERO_FUND, ta, MAKER, MAKER,
                                    fill_model=fm, causal=False)
        assert abs(float(a["final_eq"][0]) - float(b["final_eq"][0])) > 1e-9, fm


def test_fees_verdict_wrapper_selftest():
    """판정 래퍼 자가검증 (동결 기계 + 배선·경고 각인·결정론·fail-closed)."""
    afv = _load("afv_test", "lab/avgdown_fees_verdict.py")
    afv.selftest()
