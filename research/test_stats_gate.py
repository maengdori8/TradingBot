"""stats_gate 단위테스트 — 알려진 입력으로 게이트 동작을 검증한다.

GOAL_LOOP §5-6: stats_gate.py는 '없으면 첫 사이클 생성 + 알려진 입력으로 단위테스트 검증'
이 의무다. 이 파일이 그 검증이다. 실행: python -m research.test_stats_gate (또는 pytest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import stats_gate as sg


def _ts(n_days: int, per_day: int) -> np.ndarray:
    """n_days 거래일 × per_day 거래의 타임스탬프 배열."""
    days = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return np.repeat(days.values, per_day)


def test_norm_ppf_roundtrip() -> None:
    """역CDF가 CDF의 역함수인지 (근사 오차 내)."""
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.99):
        x = sg.norm_ppf(p)
        assert abs(sg.norm_cdf(x) - p) < 1e-6, (p, x)


def test_noise_fails() -> None:
    """순수 노이즈(평균 0)는 통과하면 안 된다."""
    rng = np.random.default_rng(1)
    ts = _ts(300, 3)
    r = rng.normal(0.0, 1.0, size=len(ts))
    res = sg.gate(r, ts, n_tested=50)
    assert not res.passed, res.reasons


def test_strong_signal_passes() -> None:
    """충분한 독립 거래일의 강한 양의 신호는 통과해야 한다."""
    rng = np.random.default_rng(2)
    ts = _ts(300, 3)
    r = rng.normal(0.30, 1.0, size=len(ts))
    res = sg.gate(r, ts, n_tested=5)
    assert res.passed, res.reasons


def test_multiple_testing_deflation() -> None:
    """같은 신호라도 시도수 N이 폭증하면 문턱이 올라간다 (r_min 단조 증가)."""
    assert sg.r_min(100000) > sg.r_min(5)
    rng = np.random.default_rng(3)
    ts = _ts(300, 3)
    # 약한 양의 신호: 적은 N에선 통과 가능, 큰 N에선 문턱·SR0에 막혀야 함
    r = rng.normal(0.08, 1.0, size=len(ts))
    lo_n = sg.gate(r, ts, n_tested=3)
    hi_n = sg.gate(r, ts, n_tested=500000)
    assert not hi_n.passed, hi_n.reasons
    # 시도수가 클수록 더 통과하기 어렵다 (사유 수 비감소)
    assert len(hi_n.reasons) >= len(lo_n.reasons) or not lo_n.passed


def test_clustered_signal_fails_on_neff() -> None:
    """거래가 소수 거래일에 몰리면(클러스터) 실효표본 부족으로 막혀야 한다."""
    rng = np.random.default_rng(4)
    ts = _ts(5, 200)         # 단 5거래일에 1000거래 — n_eff=5
    r = rng.normal(0.5, 1.0, size=len(ts))
    res = sg.gate(r, ts, n_tested=5)
    assert not res.passed
    assert any("실효표본" in x for x in res.reasons), res.reasons


def test_effective_sample_counts() -> None:
    """실효표본이 거래수가 아니라 거래일/주를 센다."""
    ts = _ts(10, 50)         # 10일 × 50 = 500거래
    eff = sg.effective_sample(ts)
    assert eff["n_trades"] == 500
    assert eff["n_days"] == 10
    assert eff["n_weeks"] <= 3


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} 통과")


if __name__ == "__main__":
    _run_all()
