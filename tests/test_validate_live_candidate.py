from __future__ import annotations

"""실전 후보 검증기의 체결·위험·통계 계약 회귀 테스트."""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from lab import validate_live_candidate as candidate


def _featured_frame(periods: int = 90) -> pd.DataFrame:
    """지표 계산을 우회할 수 있는 결정론적 1시간 합성 봉을 만든다."""

    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000.0,
            "atr_entry": 10.0,
            "entry_high": 200.0,
            "entry_low": 0.0,
            "exit_high": 120.0,
            "exit_low": 80.0,
            "long_gate": True,
            "short_gate": False,
        },
        index=index,
    )
    frame.loc[index[3], ["open", "high", "low", "close", "entry_high"]] = [
        100.0,
        102.0,
        99.0,
        101.0,
        100.0,
    ]
    return frame


def _params(**changes: object) -> candidate.CandidateParams:
    """짧은 합성 데이터에서 바로 거래가 생기는 고정 파라미터를 반환한다."""

    defaults: dict[str, object] = {
        "sma_length": 1,
        "stop_atr": 1.0,
        "cost_bps_side": 0.0,
        "review_holding_hours": 24,
        "max_holding_hours": 72,
        "target_r": 0.0,
    }
    defaults.update(changes)
    return replace(candidate.CandidateParams(), **defaults)


def _complete_funding(frame: pd.DataFrame) -> pd.Series:
    """합성 구간의 모든 8시간 정산시각에 0인 펀딩률을 만든다."""

    return pd.Series(
        0.0,
        index=pd.date_range(frame.index[0], frame.index[-1], freq="8h", tz="UTC"),
        dtype=float,
    )


@pytest.fixture
def direct_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """합성 프레임의 사전 계산 열을 검증기가 그대로 사용하게 한다."""

    def identity_features(
        frame: pd.DataFrame,
        params: candidate.CandidateParams,
    ) -> pd.DataFrame:
        """테스트가 지정한 특징 열을 변경하지 않고 반환한다."""

        del params
        return frame.copy()

    monkeypatch.setattr(candidate, "add_features", identity_features)


def _simulate(
    frame: pd.DataFrame,
    params: candidate.CandidateParams,
    *,
    allow_additions: bool = True,
    funding: pd.Series | None = None,
) -> list[candidate.CandidateTrade]:
    """합성 프레임을 기본 롱 심볼로 재생한다."""

    return candidate.simulate_symbol(
        frame,
        "BTC",
        params,
        _complete_funding(frame) if funding is None else funding,
        allow_additions=allow_additions,
    )


def test_pending_add_uses_actual_open_risk_before_same_bar_stop(
    direct_features: None,
) -> None:
    """예약 추매는 다음 시가에서 5% 위험으로 체결된 뒤 동일 봉 손절을 맞아야 한다."""

    del direct_features
    frame = _featured_frame(12)
    index = frame.index
    frame.loc[index[5], ["open", "high", "low", "close"]] = [98.5, 99.5, 97.5, 99.0]
    frame.loc[index[6], ["open", "high", "low", "close"]] = [99.0, 99.5, 89.0, 90.0]

    trades = _simulate(frame, _params())

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "stop"
    assert trade.additions == 1
    assert trade.exit == pytest.approx(90.0)
    assert trade.gross_r == pytest.approx(-0.85)


def test_pending_add_does_not_fill_when_open_is_beyond_active_channel(
    direct_features: None,
) -> None:
    """추매 예정 시가가 활성 채널 밖이면 추매 없이 갭 가격으로 먼저 청산해야 한다."""

    del direct_features
    frame = _featured_frame(12)
    index = frame.index
    frame.loc[index[5], ["open", "high", "low", "close"]] = [98.5, 99.5, 97.5, 99.0]
    frame.loc[index[6], ["open", "high", "low", "close", "exit_low"]] = [
        94.0,
        95.0,
        93.0,
        94.0,
        95.0,
    ]

    trades = _simulate(frame, _params())

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "channel_exit"
    assert trade.exit == pytest.approx(94.0)
    assert trade.additions == 0


def test_entry_bar_checks_stop_then_active_channel(direct_features: None) -> None:
    """다음 시가 진입 봉부터 손절 우선으로 활성 채널 청산을 검사해야 한다."""

    del direct_features
    frame = _featured_frame(12)
    index = frame.index
    frame.loc[index[4], ["open", "high", "low", "close", "exit_low"]] = [
        100.0,
        101.0,
        96.0,
        100.0,
        97.0,
    ]
    frame.loc[index[5], ["low", "exit_low"]] = [96.0, 97.0]

    trades = _simulate(frame, _params())

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "channel_exit"
    assert trade.exit == pytest.approx(97.0)
    assert trade.holding_hours == 0
    assert trade.exit_time == trade.entry_time


def test_no_add_reference_uses_full_one_risk_unit(direct_features: None) -> None:
    """추매 없는 기준선은 최초 진입에 위험예산 100%를 배정해야 한다."""

    del direct_features
    frame = _featured_frame(12)
    frame.loc[frame.index[4], ["open", "high", "low", "close"]] = [
        100.0,
        101.0,
        89.0,
        90.0,
    ]

    trades = _simulate(frame, _params(), allow_additions=False)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "same_bar_stop"
    assert trade.additions == 0
    assert trade.gross_r == pytest.approx(-1.0)


def _trade(net_r: float, exit_time: str) -> candidate.CandidateTrade:
    """요약 순서 테스트용 최소 거래 레코드를 만든다."""

    exit_timestamp = pd.Timestamp(exit_time)
    return candidate.CandidateTrade(
        symbol="BTC",
        entry_time=(exit_timestamp - pd.Timedelta(hours=1)).isoformat(),
        exit_time=exit_timestamp.isoformat(),
        direction="long",
        entry=100.0,
        average_entry=100.0,
        stop=90.0,
        target=float("nan"),
        exit=100.0,
        exit_reason="channel_exit",
        holding_hours=1,
        additions=0,
        risk_committed_r=1.0,
        gross_r=net_r,
        execution_cost_r=0.0,
        funding_cost_r=0.0,
        net_r=net_r,
    )


def test_summarize_orders_by_exit_time_and_uses_configured_risk() -> None:
    """요약 MDD는 청산 시각순이며 설정된 거래당 위험비율로 환산돼야 한다."""

    trades = [
        _trade(1.0, "2024-01-01T01:00:00Z"),
        _trade(1.0, "2024-01-01T03:00:00Z"),
        _trade(-1.0, "2024-01-01T02:00:00Z"),
        _trade(-1.0, "2024-01-01T04:00:00Z"),
    ]
    params = _params(risk_percent=0.5)

    summary = candidate.summarize(trades, params)

    assert summary["max_drawdown_r"] == pytest.approx(1.0)
    assert summary["risk_scaled_max_drawdown_percent"] == pytest.approx(0.5)


def test_review_occurs_once_at_24h_and_final_exit_is_72h(
    direct_features: None,
) -> None:
    """24시간 심사는 한 번만 하고 통과한 거래는 72시간 라벨로 종료해야 한다."""

    del direct_features
    frame = _featured_frame(85)
    entry_index = 4
    frame.loc[frame.index[entry_index + 24], "long_gate"] = True
    frame.loc[frame.index[entry_index + 25] :, "long_gate"] = False

    trades = _simulate(frame, _params())

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "time_exit_72h"
    assert trade.holding_hours == 72
    assert trade.exit_time == frame.index[entry_index + 72].isoformat()


def test_failed_review_exits_at_open_before_same_candle_stop(
    direct_features: None,
) -> None:
    """24시간 재심사 실패는 아직 모르는 그 봉의 저가 손절보다 먼저 시가 청산한다."""

    del direct_features
    frame = _featured_frame(36)
    entry_index = 4
    review_index = entry_index + 24
    frame.loc[frame.index[review_index], "long_gate"] = False
    frame.loc[
        frame.index[review_index],
        ["open", "high", "low", "close"],
    ] = [100.0, 101.0, 89.0, 90.0]

    trades = _simulate(frame, _params(), allow_additions=False)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "time_review_exit"
    assert trade.exit == pytest.approx(100.0)
    assert trade.exit_time == frame.index[review_index].isoformat()


def test_max_holding_exit_precedes_same_candle_intrabar_stop(
    direct_features: None,
) -> None:
    """최대 보유시간 청산도 해당 시가 뒤의 봉내 손절보다 먼저 실행해야 한다."""

    del direct_features
    frame = _featured_frame(40)
    entry_index = 4
    exit_index = entry_index + 30
    frame.loc[frame.index[exit_index], ["open", "high", "low", "close"]] = [
        100.0,
        101.0,
        89.0,
        90.0,
    ]

    trades = _simulate(
        frame,
        _params(review_holding_hours=20, max_holding_hours=30),
        allow_additions=False,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "time_exit_72h"
    assert trade.exit == pytest.approx(100.0)
    assert trade.exit_time == frame.index[exit_index].isoformat()


def test_failed_breakout_exits_at_next_open_after_confirmed_close(
    direct_features: None,
) -> None:
    """초기 돌파선 재이탈은 확정 종가 뒤 다음 시가에서만 청산해야 한다."""

    del direct_features
    frame = _featured_frame(12)
    index = frame.index
    frame.loc[index[4], ["open", "high", "low", "close"]] = [100.0, 101.0, 98.0, 99.0]
    frame.loc[index[5], ["open", "high", "low", "close"]] = [98.0, 99.0, 97.0, 98.0]

    trades = _simulate(
        frame,
        _params(failed_breakout_exit_hours=1),
        allow_additions=False,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "failed_breakout_exit"
    assert trade.exit == pytest.approx(98.0)
    assert trade.exit_time == index[5].isoformat()
    assert trade.holding_hours == 1


def test_protective_gap_precedes_pending_failed_breakout_exit(
    direct_features: None,
) -> None:
    """실패 돌파 청산 대기 중에도 다음 시가가 손절 밖이면 갭 손절을 우선한다."""

    del direct_features
    frame = _featured_frame(12)
    index = frame.index
    frame.loc[index[4], ["open", "high", "low", "close"]] = [100.0, 101.0, 98.0, 99.0]
    frame.loc[index[5], ["open", "high", "low", "close"]] = [85.0, 86.0, 84.0, 85.0]

    trades = _simulate(
        frame,
        _params(failed_breakout_exit_hours=1),
        allow_additions=False,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "stop_gap"
    assert trade.exit == pytest.approx(85.0)


def test_missing_funding_settlement_fails_closed() -> None:
    """가격 구간 안의 8시간 정산 하나라도 빠지면 검증을 중단해야 한다."""

    frame = _featured_frame(17)
    funding = pd.Series(
        [0.0001, 0.0001],
        index=pd.DatetimeIndex([frame.index[0], frame.index[16]]),
        dtype=float,
    )

    with pytest.raises(ValueError, match="funding|settlement|펀딩|정산"):
        candidate.validate_funding_coverage(frame, funding, "BTC")


def test_all_four_additions_fill_sequentially_without_exceeding_one_risk(
    direct_features: None,
) -> None:
    """네 단계 reclaim은 순서대로 체결되고 총 약정 위험이 1R을 넘지 않아야 한다."""

    del direct_features
    frame = _featured_frame(18)
    index = frame.index
    candles = {
        5: (98.5, 99.5, 97.5, 99.0),
        6: (99.0, 99.5, 98.5, 99.2),
        7: (96.5, 97.5, 95.5, 97.0),
        8: (97.0, 97.5, 96.5, 97.2),
        9: (94.5, 95.5, 93.5, 95.0),
        10: (95.0, 95.5, 94.5, 95.2),
        11: (92.5, 93.5, 91.5, 93.0),
        12: (93.0, 94.0, 92.5, 93.5),
    }
    for position, values in candles.items():
        frame.loc[index[position], ["open", "high", "low", "close"]] = values
    frame.loc[index[13], ["open", "high", "low", "close", "exit_low"]] = [
        96.0,
        97.0,
        93.0,
        94.0,
        94.0,
    ]

    trades = _simulate(frame, _params())

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "channel_exit"
    assert trade.additions == 4
    assert trade.risk_committed_r == pytest.approx(1.0)
    assert trade.risk_committed_r <= 1.0


def test_open_trade_at_end_of_data_is_excluded(direct_features: None) -> None:
    """정상 청산 없이 데이터가 끝난 미결제 포지션은 성과 표본에서 제외해야 한다."""

    del direct_features
    frame = _featured_frame(12)

    trades = _simulate(frame, _params())

    assert trades == []


def test_no_target_trade_serializes_as_strict_json_null(direct_features: None) -> None:
    """목표가 없는 거래는 비표준 NaN이 아니라 JSON null로 직렬화돼야 한다."""

    del direct_features
    frame = _featured_frame(12)
    frame.loc[frame.index[4], ["open", "high", "low", "close"]] = [
        100.0,
        101.0,
        89.0,
        90.0,
    ]

    trades = _simulate(frame, _params())

    assert len(trades) == 1
    assert trades[0].target is None
    payload = json.dumps(asdict(trades[0]), allow_nan=False)
    assert json.loads(payload)["target"] is None


def test_calendar_bootstrap_is_circular_deterministic_and_reports_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """달력 부트스트랩은 모든 날짜를 시작점으로 쓰며 같은 seed 결과를 보고해야 한다."""

    trades = [
        _trade(
            0.25 if day % 3 else -0.5,
            (pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)).isoformat(),
        )
        for day in range(90)
    ]

    first = candidate.calendar_block_bootstrap(trades, block_days=7, samples=250)
    second = candidate.calendar_block_bootstrap(trades, block_days=7, samples=250)

    assert first == second
    assert first["method"] == "circular_moving_block"
    assert first["calendar_days"] == 90

    populations: list[np.ndarray] = []

    class RecordingGenerator:
        """choice에 전달된 원형 블록 시작점 모집단을 기록한다."""

        def choice(
            self,
            values: np.ndarray,
            size: int,
            replace: bool,
        ) -> np.ndarray:
            """모집단을 기록하고 결정론적으로 필요한 개수만 반환한다."""

            assert replace is True
            population = np.asarray(values, dtype=int)
            populations.append(population.copy())
            return np.resize(population, size)

    monkeypatch.setattr(
        candidate.np.random,
        "default_rng",
        lambda seed: RecordingGenerator(),
    )
    candidate.calendar_block_bootstrap(trades, block_days=7, samples=1)

    assert len(populations) == 1
    np.testing.assert_array_equal(populations[0], np.arange(first["calendar_days"]))


def test_volatility_entry_gate_is_prefix_invariant_and_uses_prior_candidates() -> None:
    """변동성 분위수는 현재·미래 후보를 빼고 과거 적격 돌파만 사용해야 한다."""

    index = pd.date_range("2024-01-01", periods=80, freq="1h", tz="UTC")
    close_values = np.concatenate(
        [np.asarray([100.0, 99.0, 98.0]), 100.0 + np.arange(len(index) - 3) * 2.0]
    )
    close = pd.Series(close_values, index=index)
    spread = pd.Series(0.5 + np.arange(len(index)) % 5 * 0.1, index=index)
    frame = pd.DataFrame(
        {
            "open": close - 1.0,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1_000.0 + np.arange(len(index)) * 10.0,
        },
        index=index,
    )
    params = _params(
        entry_channel=2,
        exit_channel=2,
        atr_length=2,
        sma_length=3,
        rsi_length=2,
        volume_length=3,
        volatility_filter_days=30,
        volatility_filter_quantile=0.60,
        volatility_filter_min_samples=5,
    )

    full = candidate.add_features(frame, params)
    prefix = candidate.add_features(frame.iloc[:60], params)

    pd.testing.assert_series_equal(
        full.loc[prefix.index, "entry_regime_gate"],
        prefix["entry_regime_gate"],
    )
    eligible = full["long_gate"] & (full["close"] > full["entry_high"])
    first_allowed = full.index[full["entry_regime_gate"]][0]
    assert int(eligible.loc[eligible.index < first_allowed].sum()) >= 5


def test_causal_quantile_keeps_exact_cutoff_and_excludes_current_sample() -> None:
    """365일 경계 표본은 포함하고 현재 후보는 다음 시점부터만 반영해야 한다."""

    current = pd.Timestamp("2025-01-01", tz="UTC")
    index = pd.DatetimeIndex(
        [
            current - pd.Timedelta(days=365),
            current - pd.Timedelta(days=1),
            current,
            current + pd.Timedelta(hours=1),
        ]
    )
    values = pd.Series([1.0, 3.0, 100.0, np.nan], index=index)

    threshold = candidate.causal_rolling_quantile(
        values,
        days=365,
        min_samples=2,
        quantile=0.60,
    )

    assert threshold.loc[current] == pytest.approx(2.2)
    assert threshold.loc[current + pd.Timedelta(hours=1)] == pytest.approx(61.2)


def test_volatility_filter_rejects_unconfirmed_intrabar_entry_mode() -> None:
    """현재 종가를 쓰는 변동성 게이트는 같은 봉 intrabar 진입과 결합할 수 없다."""

    frame = _featured_frame(220)[["open", "high", "low", "close", "volume"]]
    params = _params(
        volatility_filter_days=30,
        entry_close_confirmation=False,
    )

    with pytest.raises(ValueError, match="확정 종가"):
        candidate.add_features(frame, params)


def test_negative_failed_breakout_window_is_rejected(direct_features: None) -> None:
    """실패 돌파 감시시간의 음수 설정은 조용히 비활성화되지 않아야 한다."""

    del direct_features
    frame = _featured_frame(12)

    with pytest.raises(ValueError, match="0 이상"):
        _simulate(frame, _params(failed_breakout_exit_hours=-1))
