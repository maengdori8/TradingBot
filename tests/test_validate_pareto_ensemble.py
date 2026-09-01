from __future__ import annotations

"""Pareto v5 앙상블 검증기의 위험·빈도·게이트 계약 테스트."""

from typing import Any

import pandas as pd
import pytest

from lab import validate_pareto_ensemble as ensemble
from lab.validate_live_candidate import CandidateTrade


def _trade(
    net_r: float,
    entry_time: str,
    *,
    exit_time: str | None = None,
    symbol: str = "BTC",
    risk_committed_r: float = 1.0,
    gross_r: float | None = None,
    execution_cost_r: float = 0.0,
    funding_cost_r: float = 0.0,
) -> CandidateTrade:
    """순수 함수 검증용 합성 거래를 만든다."""

    entry = pd.Timestamp(entry_time)
    exit_timestamp = (
        entry + pd.Timedelta(hours=1)
        if exit_time is None
        else pd.Timestamp(exit_time)
    )
    return CandidateTrade(
        symbol=symbol,
        entry_time=entry.isoformat(),
        exit_time=exit_timestamp.isoformat(),
        direction="long",
        entry=100.0,
        average_entry=100.0,
        stop=90.0,
        target=None,
        exit=101.0,
        exit_reason="channel_exit",
        holding_hours=int((exit_timestamp - entry) / pd.Timedelta(hours=1)),
        additions=0,
        risk_committed_r=risk_committed_r,
        gross_r=net_r if gross_r is None else gross_r,
        execution_cost_r=execution_cost_r,
        funding_cost_r=funding_cost_r,
        net_r=net_r,
    )


def _bootstrap(
    *,
    p05: float,
    probability_positive: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """네 필수 블록에 같은 결정론적 bootstrap 진단을 채운다."""

    return {
        f"{days}d": {
            "status": "ok",
            "net_r_p05": p05,
            "probability_positive": probability_positive,
        }
        for days in ensemble.BOOTSTRAP_BLOCK_DAYS
    }


def _paired() -> dict[str, dict[str, Any]]:
    """세 paired 후보우위 지표가 모두 통과하는 네 블록 입력을 만든다."""

    metric = {"p05": 0.01, "probability_positive": 0.95}
    return {
        f"{days}d": {
            "status": "ok",
            "candidate_minus_reference_net_r": dict(metric),
            "candidate_minus_reference_risk_normalized_expectancy_r": dict(metric),
            "reference_minus_candidate_max_drawdown_r": dict(metric),
        }
        for days in ensemble.BOOTSTRAP_BLOCK_DAYS
    }


def _gate_inputs() -> dict[str, Any]:
    """모든 수치 조건이 정확한 포함 경계에서 통과하는 입력을 만든다."""

    return {
        "strict_candidate": {
            "profit_factor": ensemble.REQUIRED_STRICT_PROFIT_FACTOR,
            "risk_normalized_expectancy_r": 1e-12,
        },
        "strict_dimensions": {
            "symbols": {
                symbol: {"risk_normalized_expectancy_r": 0.01}
                for symbol in ensemble.SYMBOLS
            },
            "years": {
                str(year): {
                    "complete_coverage_year": True,
                    "risk_normalized_expectancy_r": 0.01,
                }
                for year in range(2021, 2025)
            },
        },
        "stress_candidate": {
            "profit_factor": ensemble.REQUIRED_STRESS_PROFIT_FACTOR,
            "risk_normalized_expectancy_r": 1e-12,
        },
        "strict_bootstrap": _bootstrap(p05=1e-12),
        "stress_bootstrap": _bootstrap(p05=0.0, probability_positive=0.0),
        "candidate_frequency": {"median_per_month": 11.0, "p10_per_month": 2.0},
        "reference_frequency": {"median_per_month": 10.0, "p10_per_month": 2.0},
        "candidate_risk_frequency": {"median_per_month": 11.0, "p10_per_month": 2.0},
        "reference_risk_frequency": {"median_per_month": 10.0, "p10_per_month": 2.0},
        "candidate_clusters": {"clusters": 11},
        "reference_clusters": {"clusters": 10},
        "candidate_heat": {
            "max_weighted_concurrent_heat_r": (
                ensemble.MAX_WEIGHTED_CONCURRENT_HEAT_R
            ),
            "risk_scaled_max_heat_percent": (
                ensemble.MAX_WEIGHTED_CONCURRENT_HEAT_R
                * ensemble.BASE_RISK_PERCENT
            ),
        },
        "pareto": {"dominates": True},
        "paired_bootstrap": _paired(),
        "top_five_retained": {
            "profit_factor": ensemble.REQUIRED_STRICT_PROFIT_FACTOR,
            "risk_normalized_expectancy_r": 1e-12,
        },
    }


def test_weighted_trade_scaling_and_risk_normalized_expectancy() -> None:
    """위험 가중은 모든 R 금액을 선형 축소하고 기대값은 투입위험으로 나눠야 한다."""

    winning = _trade(
        1.8,
        "2024-01-01T00:00:00Z",
        gross_r=2.0,
        execution_cost_r=0.2,
    )
    losing = _trade(
        -1.0,
        "2024-01-02T00:00:00Z",
        gross_r=-0.8,
        execution_cost_r=0.2,
    )

    scaled_winning = ensemble.scale_trade(winning, 0.50)
    scaled_losing = ensemble.scale_trade(losing, 0.25)
    summary = ensemble.portfolio_summary([scaled_winning, scaled_losing])

    assert scaled_winning.risk_committed_r == pytest.approx(0.50)
    assert scaled_winning.gross_r == pytest.approx(1.0)
    assert scaled_winning.execution_cost_r == pytest.approx(0.10)
    assert scaled_winning.net_r == pytest.approx(0.90)
    assert scaled_losing.risk_committed_r == pytest.approx(0.25)
    assert scaled_losing.net_r == pytest.approx(-0.25)
    assert scaled_winning.entry == winning.entry
    assert winning.net_r == pytest.approx(1.8)

    assert summary["risk_committed_r"] == pytest.approx(0.75)
    assert summary["net_r"] == pytest.approx(0.65)
    assert summary["risk_normalized_expectancy_r"] == pytest.approx(0.866667)
    assert summary["profit_factor"] == pytest.approx(3.6)

    with pytest.raises(ValueError, match="0 초과"):
        ensemble.scale_trade(winning, 0.0)


def test_same_symbol_and_entry_time_is_deduplicated_from_frequency() -> None:
    """여러 엔진의 동일 심볼·진입시각은 월 빈도를 한 번만 늘려야 한다."""

    trades = [
        _trade(0.8, "2024-01-10T03:00:00Z", symbol="BTC"),
        _trade(0.1, "2024-01-10T03:00:00Z", symbol="BTC"),
        _trade(0.2, "2024-01-10T03:00:00Z", symbol="ETH"),
    ]

    summary = ensemble.portfolio_summary(trades)
    frequency = ensemble.unique_complete_month_frequency(
        trades,
        coverage_start=pd.Timestamp("2024-01-01T00:00:00Z"),
        coverage_end=pd.Timestamp("2024-01-31T23:00:00Z"),
    )

    assert summary["component_trades"] == 3
    assert summary["unique_entries"] == 2
    assert frequency["unique_entries"] == 2
    assert frequency["counts"] == {"2024-01": 2}
    assert frequency["median_per_month"] == pytest.approx(2.0)
    assert frequency["zero_months"] == 0


def test_risk_equivalent_frequency_does_not_count_tiny_probe_as_full_trade() -> None:
    """5% probe 신호는 raw 한 건이어도 위험등가 빈도에서는 0.05건이어야 한다."""

    trades = [
        _trade(
            0.01,
            "2024-01-10T03:00:00Z",
            symbol="BTC",
            risk_committed_r=0.05,
        ),
        _trade(
            0.02,
            "2024-01-10T03:00:00Z",
            symbol="BTC",
            risk_committed_r=0.05,
        ),
        _trade(
            0.20,
            "2024-01-20T03:00:00Z",
            symbol="ETH",
            risk_committed_r=1.0,
        ),
    ]

    frequency = ensemble.risk_equivalent_complete_month_frequency(
        trades,
        coverage_start=pd.Timestamp("2024-01-01T00:00:00Z"),
        coverage_end=pd.Timestamp("2024-01-31T23:00:00Z"),
    )

    assert frequency["unique_entries"] == 2
    assert frequency["risk_equivalent_entries"] == pytest.approx(1.1)
    assert frequency["median_per_month"] == pytest.approx(1.1)
    assert frequency["events_at_least_0_10r"] == 2
    assert frequency["events_at_least_0_25r"] == 1


def test_concurrent_heat_uses_entry_before_exit_at_same_timestamp() -> None:
    """동시 진입·청산은 보수적으로 진입을 먼저 더해 순간 heat 상한을 잡아야 한다."""

    trades = [
        _trade(
            0.1,
            "2024-01-01T00:00:00Z",
            exit_time="2024-01-01T01:00:00Z",
            risk_committed_r=1.0,
        ),
        _trade(
            0.1,
            "2024-01-01T01:00:00Z",
            exit_time="2024-01-01T02:00:00Z",
            risk_committed_r=0.5,
        ),
    ]

    heat = ensemble.max_weighted_concurrent_heat(trades, risk_percent=0.25)

    assert heat["max_weighted_concurrent_heat_r"] == pytest.approx(1.5)
    assert heat["max_concurrent_engine_components"] == 2
    assert heat["peak_time"] == "2024-01-01T01:00:00+00:00"
    assert heat["risk_scaled_max_heat_percent"] == pytest.approx(0.375)
    assert "진입을 먼저" in heat["tie_policy"]


def test_paired_bootstrap_uses_shared_blocks_for_candidate_advantage() -> None:
    """같은 달력 블록에서 항상 우세한 후보의 paired 하한은 양수여야 한다."""

    candidate = [
        _trade(
            1.0,
            f"2024-01-{day:02d}T00:00:00Z",
            exit_time=f"2024-01-{day:02d}T01:00:00Z",
        )
        for day in range(1, 21)
    ]
    reference = [
        _trade(
            0.1,
            f"2024-01-{day:02d}T00:00:00Z",
            exit_time=f"2024-01-{day:02d}T01:00:00Z",
        )
        for day in range(1, 21)
    ]

    result = ensemble.paired_calendar_bootstrap(
        candidate,
        reference,
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-30T00:00:00Z"),
        block_days=3,
        samples=200,
        seed=7,
    )

    assert result["status"] == "ok"
    assert result["candidate_minus_reference_net_r"]["p05"] > 0.0
    assert result["candidate_minus_reference_net_r"]["probability_positive"] == 1.0
    assert result["candidate_minus_reference_risk_normalized_expectancy_r"]["p05"] > 0.0


def test_strict_and_severe_gate_boundaries_have_intended_roles() -> None:
    """포함 경계는 통과하고 strict p05=0만 차단하며 severe p05=0은 경고여야 한다."""

    passing = ensemble.build_gate(**_gate_inputs())

    assert passing["statistical_conditions_pass"] is True
    assert all(
        condition["pass"] for condition in passing["conditions"].values()
    )
    assert passing["conditions"]["strict_profit_factor"]["pass"] is True
    assert passing["conditions"]["stress_20bp_profit_factor"]["pass"] is True
    assert passing["conditions"]["weighted_concurrent_heat"]["pass"] is True
    assert all(
        row["warning"]
        for row in passing["stress_bootstrap_diagnostics_not_a_gate"].values()
    )
    assert all(
        row["absolute_gate"] is False
        for row in passing["stress_bootstrap_diagnostics_not_a_gate"].values()
    )

    failing_inputs = _gate_inputs()
    failing_inputs["strict_bootstrap"]["14d"]["net_r_p05"] = 0.0
    failing = ensemble.build_gate(**failing_inputs)

    strict_rows = failing["conditions"]["strict_bootstrap_all_blocks"]
    assert strict_rows["rows"]["14d"]["pass"] is False
    assert strict_rows["pass"] is False
    assert failing["statistical_conditions_pass"] is False


def test_discovery_gate_never_returns_pass_even_if_all_numbers_pass() -> None:
    """모든 통계 조건을 통과해도 discovery 실행기는 실거래 PASS를 발급하지 않는다."""

    gate = ensemble.build_gate(**_gate_inputs())

    assert gate["statistical_conditions_pass"] is True
    assert gate["status"] == "FAIL"
    assert gate["promotion_allowed"] is False
    assert gate["promotion_capability"] == "DISABLED_IN_DISCOVERY_RUNNER"
    assert gate["prospective_gate_not_evaluated"]["promotion_allowed"] is False
    assert gate["blocking_reasons"]
