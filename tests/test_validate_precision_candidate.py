from __future__ import annotations

"""고정 Ω 기반 precision V6 검증기의 인과·분모·체결 계약 테스트."""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lab import validate_precision_candidate as precision
from lab.validate_live_candidate import CandidateTrade


def _trade(
    net_r: float,
    entry_time: str,
    *,
    symbol: str = "BTC",
    exit_time: str | None = None,
    entry: float = 100.0,
) -> CandidateTrade:
    """순수 함수 테스트용 long Ω 거래를 만든다."""

    entered = pd.Timestamp(entry_time)
    exited = entered + pd.Timedelta(hours=1) if exit_time is None else pd.Timestamp(exit_time)
    return CandidateTrade(
        symbol=symbol,
        entry_time=entered.isoformat(),
        exit_time=exited.isoformat(),
        direction="long",
        entry=entry,
        average_entry=entry,
        stop=entry - 14.0,
        target=None,
        exit=entry + net_r,
        exit_reason="channel_exit",
        holding_hours=int((exited - entered) / pd.Timedelta(hours=1)),
        additions=0,
        risk_committed_r=1.0,
        gross_r=net_r,
        execution_cost_r=0.0,
        funding_cost_r=0.0,
        net_r=net_r,
    )


def _decision(
    trade: CandidateTrade,
    selected: bool,
    *,
    score: int | None = None,
) -> precision.MetaDecision:
    """한 Ω 거래와 정확히 대응하는 합성 메타 결정을 만든다."""

    entry_time = pd.Timestamp(trade.entry_time)
    passed = 4 if selected else 0
    return precision.MetaDecision(
        symbol=trade.symbol,
        signal_time=(entry_time - pd.Timedelta(hours=1)).isoformat(),
        entry_time=entry_time.isoformat(),
        selected=selected,
        score=passed if score is None else score,
        funding_72h_sum=0.0,
        return_24h=0.03,
        body_atr=1.0,
        btc_return_168h=0.1,
        funding_pass=selected,
        return_pass=selected,
        body_pass=selected,
        btc_regime_pass=selected,
    )


def _featured_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """고정 Ω 단일거래 재생에 필요한 최소 확정 특징 프레임을 만든다."""

    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "atr_entry": 2.0,
            "exit_low": 90.0,
            "long_gate": True,
            "entry_high": 101.0,
        },
        index=index,
    )


def test_fixed_contracts_fail_closed_on_any_parameter_drift() -> None:
    """Ω와 실행 변형은 허용된 고정 계약에서 한 필드도 벗어나면 안 된다."""

    omega = precision.fixed_omega_params()
    execution = precision.fixed_execution_variant_params(omega)

    assert omega.entry_channel == 24
    assert omega.exit_channel == 12
    assert omega.stop_atr == pytest.approx(7.0)
    assert omega.volatility_filter_days == 365
    assert omega.volatility_filter_quantile == pytest.approx(0.60)
    assert omega.volatility_filter_require_full_window is True
    assert omega.add_fractions == ()
    assert omega.allow_short is False
    assert execution.max_holding_hours == 24
    assert execution.target_r == pytest.approx(1.0)
    assert execution.add_fractions == (0.20, 0.40, 0.60, 0.80)
    assert execution.tranche_weights == (80.0, 5.0, 5.0, 5.0, 5.0)

    with pytest.raises(ValueError, match="V5 core"):
        precision.validate_omega_contract(replace(omega, stop_atr=6.9))
    with pytest.raises(ValueError, match="실행 변형"):
        precision.validate_execution_contract(
            replace(execution, target_r=0.9),
            omega,
        )
    with pytest.raises(ValueError, match="72시간"):
        precision.PrecisionGateSpec(funding_lookback_hours=71)


def test_settled_funding_sum_is_causal_and_rejects_timestamp_mismatch() -> None:
    """미래 정산률은 과거 합계에 섞이지 않고 비정시 이벤트는 실패해야 한다."""

    index = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    early_only = pd.Series([0.0001], index=pd.DatetimeIndex([index[10]]))
    with_future = pd.Series(
        [0.0001, 0.0050],
        index=pd.DatetimeIndex([index[10], index[80]]),
    )

    first = precision.causal_settled_funding_sum(index, early_only)
    second = precision.causal_settled_funding_sum(index, with_future)

    assert first.loc[index[79]] == pytest.approx(second.loc[index[79]])
    assert second.loc[index[80]] == pytest.approx(0.0051)
    assert np.isnan(second.loc[index[70]])

    off_hour = pd.Series(
        [0.0001],
        index=pd.DatetimeIndex([index[10] + pd.Timedelta(minutes=30)]),
    )
    with pytest.raises(ValueError, match="정시 UTC"):
        precision.causal_settled_funding_sum(index, off_hour)

    missing_price_hour = index.delete(50)
    with pytest.raises(ValueError, match="누락"):
        precision.causal_settled_funding_sum(missing_price_hour, early_only)


def test_gate_features_do_not_change_when_only_future_prices_change() -> None:
    """신호봉 이후 가격을 바꿔도 그 신호의 네 A+ 특징은 변하지 않아야 한다."""

    index = pd.date_range("2023-01-01", periods=420, freq="1h", tz="UTC")
    close = pd.Series(100.0 + np.arange(len(index)) * 0.12, index=index)
    frame = pd.DataFrame(
        {
            "open": close - 1.0,
            "high": close + 0.5,
            "low": close - 1.5,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    btc = frame.copy()
    funding = pd.Series(
        0.00001,
        index=index[::8],
        dtype=float,
    )
    signal_time = index[300]
    changed = frame.copy()
    changed.loc[index[301]:, ["open", "high", "low", "close"]] *= 10.0

    original = precision.build_precision_gate_components(
        frame,
        "ETH",
        btc,
        funding,
        precision.fixed_omega_params(),
        precision.PrecisionGateSpec(),
    )
    future_changed = precision.build_precision_gate_components(
        changed,
        "ETH",
        btc,
        funding,
        precision.fixed_omega_params(),
        precision.PrecisionGateSpec(),
    )

    columns = [
        "funding_72h_sum",
        "return_24h",
        "body_atr",
        "btc_return_168h",
        "score",
        "selected",
    ]
    pd.testing.assert_series_equal(
        original.loc[signal_time, columns],
        future_changed.loc[signal_time, columns],
    )


def test_meta_selection_is_exact_subset_of_fixed_omega_and_fails_on_missing_row() -> None:
    """메타 규칙은 Ω 밖 거래를 만들 수 없고 특징 누락도 기권으로 숨기면 안 된다."""

    first = _trade(1.0, "2024-01-02T01:00:00Z")
    second = _trade(-1.0, "2024-01-03T01:00:00Z")
    decisions = [_decision(first, True), _decision(second, False)]

    selected = precision.select_fixed_omega_trades([first, second], decisions)

    assert selected == [first]
    with pytest.raises(ValueError, match="정확히 일치"):
        precision.select_fixed_omega_trades([first], decisions)

    component_index = pd.DatetimeIndex(
        [pd.Timestamp(first.entry_time) - pd.Timedelta(hours=1)]
    )
    components = pd.DataFrame(
        {
            "funding_72h_sum": [np.nan],
            "return_24h": [0.03],
            "body_atr": [1.0],
            "btc_return_168h": [0.1],
            "funding_pass": [False],
            "return_pass": [True],
            "body_pass": [True],
            "btc_regime_pass": [True],
            "score": [3],
            "selected": [False],
        },
        index=component_index,
    )
    with pytest.raises(ValueError, match="준비되지"):
        precision.build_meta_decisions([first], components)


def test_precision_brier_and_coverage_keep_full_omega_denominator() -> None:
    """선택이 적어져도 precision·Brier·coverage 분모는 네 Ω 기회로 고정돼야 한다."""

    trades = [
        _trade(1.0, "2024-01-01T01:00:00Z"),
        _trade(-1.0, "2024-01-02T01:00:00Z"),
        _trade(0.5, "2024-01-03T01:00:00Z"),
        _trade(-0.5, "2024-01-04T01:00:00Z"),
    ]
    decisions = [
        _decision(trades[0], True),
        _decision(trades[1], True),
        _decision(trades[2], False),
        _decision(trades[3], False),
    ]

    metrics = precision.fixed_omega_classification_metrics(trades, decisions)

    assert metrics["omega_opportunities"] == 4
    assert metrics["selected"] == 2
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["diagnostic_binary_action_brier_not_probability"] == pytest.approx(0.5)
    assert "binary_action_accuracy" not in metrics
    assert metrics["abstained"] == 2


def test_execution_precision_keeps_omega_denominator_and_capacity_as_abstain() -> None:
    """실제 정확도는 체결 승률이며 용량 거절은 Ω를 줄이지 않는 기권이어야 한다."""

    omega = [
        _trade(1.0, "2024-01-01T01:00:00Z"),
        _trade(-1.0, "2024-01-02T01:00:00Z"),
        _trade(1.0, "2024-01-03T01:00:00Z"),
        _trade(-1.0, "2024-01-04T01:00:00Z"),
    ]
    decisions = [
        _decision(omega[0], True),
        _decision(omega[1], True),
        _decision(omega[2], True),
        _decision(omega[3], False),
    ]
    rejects = [
        {
            "symbol": omega[2].symbol,
            "entry_time": omega[2].entry_time,
            "signal_time": "2024-01-03T00:00:00+00:00",
            "active_until": "2024-01-03T03:00:00+00:00",
            "reason": "capacity_reject_existing_position",
        }
    ]

    metrics = precision.fixed_execution_classification_metrics(
        omega,
        decisions,
        omega[:2],
        rejects,
    )

    assert metrics["omega_opportunities"] == 4
    assert metrics["meta_selected"] == 3
    assert metrics["capacity_rejects_as_abstain"] == 1
    assert metrics["filled"] == 2
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["execution_coverage"] == pytest.approx(0.5)
    assert metrics["total_abstained_including_capacity"] == 2
    with pytest.raises(ValueError, match="정확히 일치"):
        precision.fixed_execution_classification_metrics(
            omega,
            decisions[:-1],
            omega[:2],
            rejects,
        )


def test_precision_calendar_bootstrap_is_joint_deterministic_and_fail_closed() -> None:
    """wins·accepted는 같은 블록으로 뽑고 0 accepted 경로는 통계통과가 아니다."""

    trades = [
        _trade(
            1.0 if day % 3 else -1.0,
            (pd.Timestamp("2024-01-01T01:00:00Z") + pd.Timedelta(days=day)).isoformat(),
        )
        for day in range(40)
    ]
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2024-02-09T23:00:00Z")

    first = precision.precision_calendar_block_bootstrap(
        trades,
        start,
        end,
        14,
        samples=256,
        seed=123,
    )
    second = precision.precision_calendar_block_bootstrap(
        trades,
        start,
        end,
        14,
        samples=256,
        seed=123,
    )
    empty = precision.precision_calendar_block_bootstrap(
        [],
        start,
        end,
        14,
        samples=32,
        seed=123,
    )

    assert first == second
    assert first["status"] == "ok"
    assert first["zero_accepted_samples"] == 0
    assert 0.0 <= first["precision_p05"] <= first["precision_p95"] <= 1.0
    assert empty["status"] == "insufficient_no_accepted_events"
    assert empty["precision_p05"] is None


def test_same_bar_stop_precedes_channel_and_target() -> None:
    """진입봉에 stop·channel·target이 모두 닿으면 최후손절을 우선한다."""

    index = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    featured = _featured_frame(index)
    featured.loc[index[2], ["open", "high", "low", "close"]] = [100.0, 115.0, 85.0, 105.0]
    omega = _trade(0.1, index[2].isoformat(), entry=100.0)
    params = precision.fixed_execution_variant_params(precision.fixed_omega_params())

    replayed = precision.simulate_fixed_omega_opportunity(
        featured[["open", "high", "low", "close", "volume"]],
        featured,
        omega,
        params,
        pd.Series([0.0], index=pd.DatetimeIndex([index[0]])),
    )

    assert replayed.exit_reason == "same_bar_stop"
    assert replayed.exit == pytest.approx(86.0)
    assert replayed.additions == 0


def test_recalculated_target_after_addition_activates_on_next_bar() -> None:
    """추매봉의 가까워진 새 TP는 같은 봉 고가가 닿아도 다음 봉부터만 유효하다."""

    index = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    featured = _featured_frame(index)
    featured.loc[index[2], ["open", "high", "low", "close"]] = [100.0, 101.0, 98.0, 100.0]
    featured.loc[index[3], ["open", "high", "low", "close"]] = [96.0, 99.0, 95.0, 98.0]
    featured.loc[index[4], ["open", "high", "low", "close"]] = [97.0, 115.0, 96.0, 110.0]
    featured.loc[index[5], ["open", "high", "low", "close"]] = [113.7, 114.0, 113.0, 113.8]
    omega = _trade(0.1, index[2].isoformat(), entry=100.0)
    params = precision.fixed_execution_variant_params(precision.fixed_omega_params())

    replayed = precision.simulate_fixed_omega_opportunity(
        featured[["open", "high", "low", "close", "volume"]],
        featured,
        omega,
        params,
        pd.Series([0.0], index=pd.DatetimeIndex([index[0]])),
    )

    assert replayed.additions == 1
    assert replayed.exit_reason == "target_gap"
    assert pd.Timestamp(replayed.exit_time) == index[5]
    assert replayed.target is not None
    assert 113.5 < replayed.target < 113.7


def test_execution_replay_records_capacity_reject_without_new_opportunity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """선택 Ω 도착 때 포지션이 바쁘면 새 분모를 만들지 않고 capacity reject한다."""

    index = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    frames = {
        symbol: _featured_frame(index)[["open", "high", "low", "close", "volume"]]
        for symbol in precision.SYMBOLS
    }
    funding = pd.DataFrame(0.0, index=index, columns=precision.SYMBOLS)
    first = _trade(
        0.1,
        index[2].isoformat(),
        exit_time=index[3].isoformat(),
    )
    second = _trade(
        0.1,
        index[4].isoformat(),
        exit_time=index[5].isoformat(),
    )

    monkeypatch.setattr(precision, "add_features", lambda frame, params: frame)

    def fake_simulator(
        frame: pd.DataFrame,
        featured: pd.DataFrame,
        omega_trade: CandidateTrade,
        params: Any,
        rates: pd.Series,
    ) -> CandidateTrade:
        del frame, featured, params, rates
        return replace(omega_trade, exit_time=index[6].isoformat())

    monkeypatch.setattr(precision, "simulate_fixed_omega_opportunity", fake_simulator)
    omega_params = precision.fixed_omega_params()
    execution_params = precision.fixed_execution_variant_params(omega_params)

    replay = precision.replay_execution_variant(
        frames,
        funding,
        [first, second],
        execution_params,
        omega_params,
    )

    assert len(replay.trades) == 1
    assert len(replay.capacity_rejects) == 1
    assert replay.capacity_rejects[0]["entry_time"] == index[4].isoformat()
    assert replay.capacity_rejects[0]["reason"] == "capacity_reject_existing_position"


def test_top_macro_episode_removal_and_discovery_gate_are_conservative() -> None:
    """최고 수익 주간 전체를 제거하고 성과와 무관하게 승격은 실패해야 한다."""

    trades = [
        _trade(2.0, "2024-01-02T01:00:00Z", symbol="BTC"),
        _trade(1.0, "2024-01-03T01:00:00Z", symbol="ETH"),
        _trade(0.5, "2024-01-10T01:00:00Z", symbol="SOL"),
    ]

    removal = precision.macro_episode_removal(trades)
    gate = precision.discovery_hard_fail_gate()

    assert removal["removed"]["trades"] == 2
    assert removal["removed"]["net_r"] == pytest.approx(3.0)
    assert removal["retained_summary"]["net_r"] == pytest.approx(0.5)
    assert gate["status"] == "FAIL"
    assert gate["promotion_allowed"] is False
    assert gate["performance_cannot_override_hard_fail"] is True


def test_statistical_conditions_report_each_failure_without_overriding_discovery() -> None:
    """통계 조건은 항목별로 실패하며 discovery 하드 FAIL과 독립이어야 한다."""

    meta_classifications = {
        severity: {
            "precision": 0.51,
            "coverage": 0.10,
            "omega_opportunities": 100,
            "selected": 10,
        }
        for severity in ("strict", "severe")
    }
    execution_classifications = {
        severity: {
            "precision": 0.51,
            "execution_coverage": 0.08,
            "filled": 8,
        }
        for severity in ("strict", "severe")
    }
    empty_symbols = {
        symbol: {"component_trades": 0, "component_win_rate": None}
        for symbol in precision.SYMBOLS
    }
    severity_suite = {
        "summary": {
            "profit_factor": 1.0,
            "risk_normalized_expectancy_r": 0.0,
            "risk_committed_r": 8.0,
        },
        "dimensions": {"symbols": empty_symbols},
    }
    suite = {
        "strict_12bp_actual_funding": severity_suite,
        "severe_20bp_funding_debit_x2_credit_zero": severity_suite,
    }
    bootstrap_result = {
        f"{days}d": {
            "status": "ok",
            "precision_p05": 0.49,
            "probability_precision_gt_0_50": 0.50,
            "zero_accepted_samples": 0,
        }
        for days in precision.BOOTSTRAP_BLOCK_DAYS
    }
    bootstraps = {"strict": bootstrap_result, "severe": bootstrap_result}
    execution_trades = [
        replace(
            _trade(
                0.1,
                (
                    pd.Timestamp("2024-03-01T01:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
            ),
            additions=1,
        )
        for index in range(18)
    ]

    statistical = precision.build_statistical_conditions(
        meta_classifications,
        execution_classifications,
        suite,
        suite,
        bootstraps,
        bootstraps,
        execution_trades,
    )
    discovery = precision.discovery_hard_fail_gate()

    assert statistical["statistical_conditions_pass"] is False
    assert "meta_shadow_strict_precision" in statistical["failed_conditions"]
    assert "actual_execution_coverage" in statistical["failed_conditions"]
    assert "actual_risk_coverage" in statistical["failed_conditions"]
    for stage in range(1, 5):
        assert f"actual_execution_addition_stage_{stage}_count" in statistical[
            "failed_conditions"
        ]
    assert "actual_execution_severe_precision_bootstrap_84d" in statistical[
        "failed_conditions"
    ]
    stage_conditions = {
        condition["name"]: condition["observed"]
        for condition in statistical["conditions"]
        if "addition_stage" in condition["name"]
    }
    assert list(stage_conditions.values()) == [18, 0, 0, 0]
    assert all(
        set(condition) == {"name", "observed", "required", "pass"}
        for condition in statistical["conditions"]
    )
    assert discovery["status"] == "FAIL"
    assert discovery["promotion_allowed"] is False
