from __future__ import annotations

"""Pareto v4 검증기의 순수 함수 계약 회귀 테스트."""

from dataclasses import asdict

import pandas as pd
import pytest

from lab import validate_pareto_candidate as pareto
from lab.pareto_trial_ledger import DISCOVERY_CLASSIFICATION
from lab.validate_live_candidate import CandidateTrade


def _trade(
    net_r: float,
    entry_time: str,
    *,
    symbol: str = "BTC",
    gross_r: float | None = None,
    execution_cost_r: float = 0.0,
    funding_cost_r: float = 0.0,
) -> CandidateTrade:
    """순수 함수 테스트에 필요한 최소 합성 거래를 만든다."""

    entry = pd.Timestamp(entry_time)
    return CandidateTrade(
        symbol=symbol,
        entry_time=entry.isoformat(),
        exit_time=(entry + pd.Timedelta(hours=1)).isoformat(),
        direction="long",
        entry=100.0,
        average_entry=100.0,
        stop=90.0,
        target=None,
        exit=101.0,
        exit_reason="channel_exit",
        holding_hours=1,
        additions=0,
        risk_committed_r=1.0,
        gross_r=net_r if gross_r is None else gross_r,
        execution_cost_r=execution_cost_r,
        funding_cost_r=funding_cost_r,
        net_r=net_r,
    )


def test_fixed_candidate_and_matched_reference_contracts() -> None:
    """v4 후보는 탐색 여지 없이 고정되고 기준은 변동성 필터만 제거해야 한다."""

    candidate = pareto.fixed_candidate_params()
    reference = pareto.fixed_reference_params(candidate)

    assert candidate.entry_channel == 24
    assert candidate.exit_channel == 12
    assert candidate.atr_length == 24
    assert candidate.stop_atr == pytest.approx(7.0)
    assert candidate.volatility_filter_days == 365
    assert candidate.volatility_filter_quantile == pytest.approx(0.60)
    assert candidate.volatility_filter_min_samples == 30
    assert candidate.volatility_filter_require_full_window is True
    assert candidate.add_fractions == ()
    assert candidate.tranche_weights == (100.0,)
    assert candidate.cost_bps_side == pytest.approx(12.0)
    assert candidate.allow_short is False
    assert candidate.discovery_only is True

    candidate_values = asdict(candidate)
    reference_values = asdict(reference)
    differing = {
        key
        for key in candidate_values
        if candidate_values[key] != reference_values[key]
    }
    assert differing == {
        "volatility_filter_days",
        "volatility_filter_require_full_window",
    }
    assert reference.volatility_filter_days == 0
    assert reference.volatility_filter_require_full_window is False


def test_stress_scales_execution_doubles_debits_and_removes_credits() -> None:
    """스트레스는 실행비용을 비례 확대하고 펀딩에 비대칭 보수 처리를 해야 한다."""

    debit = _trade(
        0.85,
        "2024-01-01T00:00:00Z",
        gross_r=1.0,
        execution_cost_r=0.12,
        funding_cost_r=0.03,
    )
    credit = _trade(
        -0.21,
        "2024-01-02T00:00:00Z",
        gross_r=-0.2,
        execution_cost_r=0.06,
        funding_cost_r=-0.05,
    )

    stressed = pareto.apply_execution_funding_stress(
        [debit, credit],
        original_cost_bps_side=12.0,
        stressed_cost_bps_side=20.0,
    )

    assert stressed[0].execution_cost_r == pytest.approx(0.20)
    assert stressed[0].funding_cost_r == pytest.approx(0.06)
    assert stressed[0].net_r == pytest.approx(0.74)
    assert stressed[1].execution_cost_r == pytest.approx(0.10)
    assert stressed[1].funding_cost_r == pytest.approx(0.0)
    assert stressed[1].net_r == pytest.approx(-0.30)
    assert debit.execution_cost_r == pytest.approx(0.12)
    assert credit.funding_cost_r == pytest.approx(-0.05)


def test_matched_coverage_filters_on_signal_time_not_entry_time() -> None:
    """확정봉 진입은 체결시각보다 한 시간 전 신호 마스크와 경계를 사용해야 한다."""

    index = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    ready = pd.Series([False, True, False, True, True, True], index=index)
    trades = [
        _trade(0.1, index[2].isoformat()),
        _trade(0.2, index[3].isoformat()),
        _trade(0.3, index[4].isoformat()),
        _trade(0.4, index[5].isoformat()),
    ]

    matched = pareto.filter_matched_coverage(
        trades,
        ready,
        signal_start=index[1],
        signal_end=index[3],
        close_confirmation=True,
    )

    assert [trade.net_r for trade in matched] == [0.1, 0.3]

    intrabar = pareto.filter_matched_coverage(
        trades,
        ready,
        signal_start=index[1],
        signal_end=index[3],
        close_confirmation=False,
    )
    assert [trade.net_r for trade in intrabar] == [0.2]


def test_realized_bootstrap_is_deterministic_and_keeps_zero_trade_days() -> None:
    """달력 bootstrap은 무거래일을 보존하고 같은 seed에서 MDD까지 재현해야 한다."""

    trades = [
        _trade(1.0, "2024-01-01T00:00:00Z"),
        _trade(-0.75, "2024-01-07T00:00:00Z"),
    ]
    arguments = {
        "calendar_start": pd.Timestamp("2024-01-01T00:00:00Z"),
        "calendar_end": pd.Timestamp("2024-01-12T23:00:00Z"),
        "block_days": 3,
        "samples": 256,
        "seed": 1729,
    }

    first = pareto.realized_calendar_bootstrap(trades, **arguments)
    second = pareto.realized_calendar_bootstrap(trades, **arguments)

    assert first == second
    assert first["status"] == "ok"
    assert first["calendar_days"] == 12
    assert first["method"] == "realized_daily_pnl_circular_moving_block"
    assert first["max_drawdown_r_p95"] >= 0.0
    assert 0.0 <= first["probability_positive"] <= 1.0


def test_complete_month_counts_include_months_with_zero_entries() -> None:
    """완전 coverage 월의 월별 빈도에는 거래가 없던 달도 0으로 포함해야 한다."""

    frequency = pareto.complete_month_counts(
        [
            pd.Timestamp("2024-02-10T00:00:00Z"),
            pd.Timestamp("2024-04-01T00:00:00Z"),
            pd.Timestamp("2024-04-30T23:00:00Z"),
        ],
        coverage_start=pd.Timestamp("2024-01-15T00:00:00Z"),
        coverage_end=pd.Timestamp("2024-05-10T00:00:00Z"),
    )

    assert frequency["complete_months"] == 3
    assert frequency["counts"] == {
        "2024-02": 1,
        "2024-03": 0,
        "2024-04": 2,
    }
    assert frequency["median_per_month"] == pytest.approx(1.0)
    assert frequency["mean_per_month"] == pytest.approx(1.0)


def test_six_hour_clusters_use_inclusive_single_linkage_gaps() -> None:
    """정확히 6시간인 인접 진입은 사슬처럼 묶이고 6시간 초과에서만 분리돼야 한다."""

    trades = [
        _trade(1.0, "2024-01-01T00:00:00Z", symbol="BTC"),
        _trade(-0.4, "2024-01-01T06:00:00Z", symbol="ETH"),
        _trade(0.2, "2024-01-01T12:00:00Z", symbol="SOL"),
        _trade(-0.5, "2024-01-01T18:01:00Z", symbol="XRP"),
        _trade(0.1, "2024-01-02T00:01:00Z", symbol="DOGE"),
    ]

    clusters = pareto.six_hour_entry_clusters(trades)

    assert clusters["method"] == "single_linkage_entry_gap_at_most_6h"
    assert clusters["independence_claim"] is False
    assert clusters["trades"] == 5
    assert clusters["clusters"] == 2
    assert clusters["median_trades_per_cluster"] == pytest.approx(2.5)
    assert clusters["max_trades_per_cluster"] == 3
    assert clusters["cluster_win_rate"] == pytest.approx(0.5)
    assert clusters["cluster_expectancy_r"] == pytest.approx(0.2)
    assert clusters["cluster_net_r"] == pytest.approx(0.4)
    assert clusters["cluster_profit_factor"] == pytest.approx(2.0)
    assert clusters["cluster_max_drawdown_r"] == pytest.approx(0.4)


def test_ledger_metrics_can_dominate_but_remain_discovery_only() -> None:
    """세 축을 모두 개선해도 discovery 비교는 실거래 승격을 허용하지 않아야 한다."""

    candidate = pareto.metrics_for_ledger(
        {
            "profit_factor": 1.5,
            "expectancy_r": 0.2,
            "net_r": 10.0,
            "max_drawdown_r": 2.0,
        },
        {
            "14d": {"status": "ok", "max_drawdown_r_p95": 3.0},
            "28d": {"status": "ok", "max_drawdown_r_p95": 4.0},
        },
        {"median_per_month": 5.0},
    )
    reference = pareto.metrics_for_ledger(
        {
            "profit_factor": 1.2,
            "expectancy_r": 0.1,
            "net_r": 5.0,
            "max_drawdown_r": 3.0,
        },
        {"14d": {"status": "ok", "max_drawdown_r_p95": 5.0}},
        {"median_per_month": 4.0},
    )

    comparison = pareto.compare_pareto(candidate, reference)

    assert candidate.bootstrap_mdd_p95_r == pytest.approx(4.0)
    assert comparison.dominates is True
    assert comparison.improves_all_axes is True
    assert comparison.promotion_allowed is False
    assert comparison.classification == DISCOVERY_CLASSIFICATION
    assert comparison.promotion_capability == "DISABLED_IN_DISCOVERY_LEDGER"
