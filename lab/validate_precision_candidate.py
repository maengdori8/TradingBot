from __future__ import annotations

"""고정 V5 기회집합에서 A+ 정밀도 후보와 실행 변형을 검증한다.

정확도 분모를 사후 필터나 조기 청산으로 줄이지 않기 위해 기회집합 Ω는 V5 core
q60/full365/24-12/7ATR long-only 재생으로 먼저 고정한다. A+ 메타 규칙은 Ω의
신호시각만 선택하거나 기권하며, 선택되지 않은 시점에서 새 진입 기회를 만들지 않는다.

24시간 최대보유·1R 목표·4단계 추매 실행 변형도 같은 선택 Ω만 입력으로 받는다.
따라서 조기 청산이 원시 돌파를 새로 열어도 Ω 밖 신호는 거래하지 않으며, 실행 엔진이
바쁜 경우에는 선택 신호를 capacity reject로 별도 기록한다. 모든 결과는 이미 관측한
자료에서 찾은 discovery 결과이므로 성과와 무관하게 실거래 승격은 하드 실패다.
"""

import argparse
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.pareto_trial_ledger import (
    DISCOVERY_CLASSIFICATION,
    PROMOTION_CAPABILITY,
    ParetoMetrics,
    ParetoTrialLedger,
    sha256_files,
)
from lab.validate_live_candidate import (
    ROOT,
    CandidateParams,
    CandidateTrade,
    add_features,
    trade_pnl,
)
from lab.validate_pareto_candidate import (
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    SYMBOLS,
    apply_execution_funding_stress,
    bootstrap_suite,
    conservative_bootstrap_mdd,
    data_hash_manifest,
    prepare_inputs,
    replay_matched,
)
from lab.validate_pareto_ensemble import (
    FOLLOW_UP_EMBARGO_HOURS,
    dimension_summary,
    max_weighted_concurrent_heat,
    portfolio_summary,
    risk_equivalent_complete_month_frequency,
    six_hour_unique_entry_clusters,
    top_winner_analysis,
    unique_complete_month_frequency,
    unique_entry_groups,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "logs" / "validation" / "precision_candidate_v6"
LEDGER_PATH = ROOT / "logs" / "validation" / "precision_candidate_trials.jsonl"

STRICT_COST_BPS_SIDE = 12.0
SEVERE_COST_BPS_SIDE = 20.0
FUNDING_LOOKBACK_HOURS = 72
FUNDING_SUM_MAX = 0.0004
RETURN_24H_MIN = 0.02
BODY_ATR_MIN = 0.75
BTC_RETURN_168H_MIN = 0.0
MACRO_EPISODE_FREQUENCY = "W-SUN"
BASE_RISK_PERCENT = 0.25
REQUIRED_PRECISION = 0.50
REQUIRED_PRECISION_BOOTSTRAP_PROBABILITY = 0.95


@dataclass(frozen=True)
class PrecisionGateSpec:
    """A+ 메타 선택에 쓰는 고정 인과 특징 계약이다."""

    funding_lookback_hours: int = FUNDING_LOOKBACK_HOURS
    funding_sum_max: float = FUNDING_SUM_MAX
    return_24h_min: float = RETURN_24H_MIN
    body_atr_min: float = BODY_ATR_MIN
    btc_return_168h_min: float = BTC_RETURN_168H_MIN

    def __post_init__(self) -> None:
        """잘못된 기간·임계값을 fail-closed로 차단한다."""

        if self.funding_lookback_hours != FUNDING_LOOKBACK_HOURS:
            raise ValueError("A+ funding lookback은 정확히 72시간이어야 합니다.")
        finite = (
            self.funding_sum_max,
            self.return_24h_min,
            self.body_atr_min,
            self.btc_return_168h_min,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("A+ 게이트 임계값은 모두 유한해야 합니다.")
        if self.body_atr_min <= 0.0:
            raise ValueError("A+ body/ATR 임계값은 0보다 커야 합니다.")


@dataclass(frozen=True)
class MetaDecision:
    """고정 Ω 한 건의 A+ 선택·기권과 네 인과 조건을 기록한다."""

    symbol: str
    signal_time: str
    entry_time: str
    selected: bool
    score: int
    funding_72h_sum: float
    return_24h: float
    body_atr: float
    btc_return_168h: float
    funding_pass: bool
    return_pass: bool
    body_pass: bool
    btc_regime_pass: bool


@dataclass(frozen=True)
class ExecutionReplay:
    """선택 Ω 실행 변형의 체결 거래와 용량 거절 원장을 담는다."""

    trades: tuple[CandidateTrade, ...]
    capacity_rejects: tuple[dict[str, str], ...]


def fixed_omega_params() -> CandidateParams:
    """정확도 분모 Ω를 만드는 V5 core 파라미터를 반환한다."""

    return CandidateParams(
        entry_channel=24,
        exit_channel=12,
        atr_length=24,
        stop_atr=7.0,
        volatility_filter_days=365,
        volatility_filter_quantile=0.60,
        volatility_filter_min_samples=30,
        volatility_filter_require_full_window=True,
        add_fractions=(),
        tranche_weights=(100.0,),
        cost_bps_side=STRICT_COST_BPS_SIDE,
        allow_short=False,
        discovery_only=True,
    )


def fixed_execution_variant_params(omega: CandidateParams) -> CandidateParams:
    """24시간·1R 목표·80/5/5/5/5 추매 실행 변형을 반환한다."""

    validate_omega_contract(omega)
    return replace(
        omega,
        target_r=1.0,
        review_holding_hours=24,
        max_holding_hours=24,
        add_fractions=(0.20, 0.40, 0.60, 0.80),
        tranche_weights=(80.0, 5.0, 5.0, 5.0, 5.0),
    )


def validate_omega_contract(params: CandidateParams) -> None:
    """Ω 파라미터가 V5 core 계약에서 벗어나면 즉시 실패시킨다."""

    expected = fixed_omega_params()
    if asdict(params) != asdict(expected):
        raise ValueError("고정 Ω 파라미터가 V5 core 계약과 일치하지 않습니다.")


def validate_execution_contract(
    params: CandidateParams,
    omega: CandidateParams,
) -> None:
    """실행 변형이 허용된 여섯 필드 외 Ω 계약을 바꾸지 못하게 한다."""

    expected = fixed_execution_variant_params(omega)
    if asdict(params) != asdict(expected):
        raise ValueError("V6 실행 변형 파라미터 계약이 일치하지 않습니다.")


def _validate_utc_hourly_index(index: pd.DatetimeIndex, label: str) -> None:
    """특징 재생 입력의 UTC·정시·중복·간격 무결성을 검증한다."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{label} 인덱스는 DatetimeIndex여야 합니다.")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError(f"{label} 인덱스는 UTC여야 합니다.")
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{label} 인덱스가 비었거나 중복·비정렬입니다.")
    if ((index.minute != 0) | (index.second != 0) | (index.microsecond != 0)).any():
        raise ValueError(f"{label} 인덱스는 정시 1시간 경계여야 합니다.")
    if len(index) > 1 and not (
        index.to_series().diff().dropna() == pd.Timedelta(hours=1)
    ).all():
        raise ValueError(f"{label} 인덱스에 1시간 누락 또는 불규칙 간격이 있습니다.")


def causal_settled_funding_sum(
    frame_index: pd.DatetimeIndex,
    funding: pd.Series,
    lookback_hours: int = FUNDING_LOOKBACK_HOURS,
) -> pd.Series:
    """시각 t까지 실제 정산된 funding만 72시간 창으로 합산한다."""

    _validate_utc_hourly_index(frame_index, "가격")
    if lookback_hours != FUNDING_LOOKBACK_HOURS:
        raise ValueError("A+ funding 합산창은 정확히 72시간이어야 합니다.")
    rates = funding.copy()
    if not isinstance(rates.index, pd.DatetimeIndex):
        raise TypeError("funding 인덱스는 DatetimeIndex여야 합니다.")
    if rates.index.tz is None or str(rates.index.tz) != "UTC":
        raise ValueError("funding 인덱스는 UTC여야 합니다.")
    if rates.empty or rates.index.has_duplicates or not rates.index.is_monotonic_increasing:
        raise ValueError("funding이 비었거나 중복·비정렬입니다.")
    if rates.isna().any() or not np.isfinite(rates.to_numpy(dtype=float)).all():
        raise ValueError("funding에 NaN 또는 무한대가 있습니다.")
    if (
        (rates.index.minute != 0)
        | (rates.index.second != 0)
        | (rates.index.microsecond != 0)
    ).any():
        raise ValueError("funding 정산시각은 정시 UTC여야 합니다.")
    in_price_range = rates.loc[
        (rates.index >= frame_index[0]) & (rates.index <= frame_index[-1])
    ]
    missing_event_times = in_price_range.index.difference(frame_index)
    if not missing_event_times.empty:
        raise ValueError("가격축에 없는 funding 정산시각이 있습니다.")
    hourly_events = rates.reindex(frame_index, fill_value=0.0).astype(float)
    return hourly_events.rolling(
        lookback_hours,
        min_periods=lookback_hours,
    ).sum()


def build_precision_gate_components(
    frame: pd.DataFrame,
    symbol: str,
    btc_frame: pd.DataFrame,
    funding: pd.Series,
    params: CandidateParams,
    spec: PrecisionGateSpec,
) -> pd.DataFrame:
    """각 1시간 신호봉에서 알 수 있던 A+ 네 특징과 통과값을 만든다."""

    validate_omega_contract(params)
    if symbol not in SYMBOLS:
        raise ValueError(f"허용되지 않은 심볼입니다: {symbol}")
    _validate_utc_hourly_index(frame.index, f"{symbol} 가격")
    _validate_utc_hourly_index(btc_frame.index, "BTC 가격")
    featured = add_features(frame, params)
    funding_sum = causal_settled_funding_sum(
        frame.index,
        funding,
        spec.funding_lookback_hours,
    )
    return_24h = frame["close"] / frame["close"].shift(24) - 1.0
    body_atr = (frame["close"] - frame["open"]) / featured["atr_entry"]
    btc_return = btc_frame["close"] / btc_frame["close"].shift(168) - 1.0
    btc_return = btc_return.reindex(frame.index)
    components = pd.DataFrame(
        {
            "funding_72h_sum": funding_sum,
            "return_24h": return_24h,
            "body_atr": body_atr,
            "btc_return_168h": btc_return,
        },
        index=frame.index,
    )
    components["funding_pass"] = components["funding_72h_sum"] <= spec.funding_sum_max
    components["return_pass"] = components["return_24h"] >= spec.return_24h_min
    components["body_pass"] = components["body_atr"] >= spec.body_atr_min
    components["btc_regime_pass"] = (
        components["btc_return_168h"] >= spec.btc_return_168h_min
    )
    pass_columns = [
        "funding_pass",
        "return_pass",
        "body_pass",
        "btc_regime_pass",
    ]
    components["score"] = components[pass_columns].fillna(False).sum(axis=1).astype(int)
    components["selected"] = components[pass_columns].fillna(False).all(axis=1)
    return components


def build_meta_decisions(
    omega_trades: Sequence[CandidateTrade],
    components: pd.DataFrame,
) -> list[MetaDecision]:
    """고정 Ω 거래마다 직전 신호봉의 A+ 선택·기권을 기록한다."""

    decisions: list[MetaDecision] = []
    seen: set[tuple[str, pd.Timestamp]] = set()
    numeric_columns = [
        "funding_72h_sum",
        "return_24h",
        "body_atr",
        "btc_return_168h",
    ]
    for trade in sorted(
        omega_trades,
        key=lambda item: (item.entry_time, item.symbol, item.exit_time),
    ):
        if trade.direction != "long":
            raise ValueError("고정 Ω에는 long 거래만 허용됩니다.")
        entry_time = pd.Timestamp(trade.entry_time)
        signal_time = entry_time - pd.Timedelta(hours=1)
        key = (trade.symbol, entry_time)
        if key in seen:
            raise ValueError("고정 Ω에 동일 심볼·진입시각이 중복됐습니다.")
        seen.add(key)
        if signal_time not in components.index:
            raise ValueError("Ω 신호시각의 A+ 특징 행이 없습니다.")
        row = components.loc[signal_time]
        numeric = row[numeric_columns].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Ω 신호시각의 A+ 인과 특징이 준비되지 않았습니다.")
        decisions.append(
            MetaDecision(
                symbol=trade.symbol,
                signal_time=signal_time.isoformat(),
                entry_time=entry_time.isoformat(),
                selected=bool(row["selected"]),
                score=int(row["score"]),
                funding_72h_sum=float(row["funding_72h_sum"]),
                return_24h=float(row["return_24h"]),
                body_atr=float(row["body_atr"]),
                btc_return_168h=float(row["btc_return_168h"]),
                funding_pass=bool(row["funding_pass"]),
                return_pass=bool(row["return_pass"]),
                body_pass=bool(row["body_pass"]),
                btc_regime_pass=bool(row["btc_regime_pass"]),
            )
        )
    return decisions


def _trade_key(trade: CandidateTrade) -> tuple[str, str]:
    """Ω 거래와 메타 결정을 연결하는 심볼·진입시각 키를 반환한다."""

    return trade.symbol, pd.Timestamp(trade.entry_time).isoformat()


def select_fixed_omega_trades(
    omega_trades: Sequence[CandidateTrade],
    decisions: Sequence[MetaDecision],
) -> list[CandidateTrade]:
    """새 기회를 만들지 않고 Ω 중 selected 거래만 그대로 반환한다."""

    decision_map = {
        (decision.symbol, pd.Timestamp(decision.entry_time).isoformat()): decision
        for decision in decisions
    }
    if len(decision_map) != len(decisions):
        raise ValueError("메타 결정 키가 중복됐습니다.")
    omega_keys = {_trade_key(trade) for trade in omega_trades}
    if omega_keys != set(decision_map):
        raise ValueError("메타 결정과 고정 Ω 기회집합이 정확히 일치하지 않습니다.")
    return [trade for trade in omega_trades if decision_map[_trade_key(trade)].selected]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | int | None]:
    """이항 비율의 Wilson 구간을 반환한다."""

    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson 성공수와 전체수 범위가 잘못됐습니다.")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("Wilson z는 유한한 양수여야 합니다.")
    if total == 0:
        return {"successes": successes, "total": total, "estimate": None, "low": None, "high": None}
    estimate = successes / total
    denominator = 1.0 + z * z / total
    midpoint = (estimate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "estimate": round(estimate, 6),
        "low": round(midpoint - half_width, 6),
        "high": round(midpoint + half_width, 6),
    }


def fixed_omega_classification_metrics(
    omega_trades: Sequence[CandidateTrade],
    decisions: Sequence[MetaDecision],
) -> dict[str, Any]:
    """Ω 고정 분모에서 shadow precision·coverage와 행동 진단을 계산한다."""

    decision_map = {
        (decision.symbol, pd.Timestamp(decision.entry_time).isoformat()): decision
        for decision in decisions
    }
    if len(decision_map) != len(decisions) or len(decisions) != len(omega_trades):
        raise ValueError("정확도 분모 Ω와 메타 결정 수가 일치하지 않습니다.")
    outcomes: list[float] = []
    predictions: list[float] = []
    selected_outcomes: list[float] = []
    for trade in omega_trades:
        decision = decision_map.get(_trade_key(trade))
        if decision is None:
            raise ValueError("Ω 거래에 대응하는 메타 결정이 없습니다.")
        outcome = float(trade.net_r > 0.0)
        prediction = float(decision.selected)
        outcomes.append(outcome)
        predictions.append(prediction)
        if decision.selected:
            selected_outcomes.append(outcome)
    outcome_array = np.asarray(outcomes, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    selected_array = np.asarray(selected_outcomes, dtype=float)
    selected_wins = int(selected_array.sum())
    selected_count = len(selected_array)
    omega_wins = int(outcome_array.sum())
    return {
        "denominator": "fixed_v5_core_shadow_omega",
        "omega_opportunities": len(omega_trades),
        "omega_wins": omega_wins,
        "omega_base_rate": (
            round(float(outcome_array.mean()), 6) if len(outcome_array) else None
        ),
        "selected": selected_count,
        "selected_wins": selected_wins,
        "precision": round(float(selected_array.mean()), 6) if selected_count else None,
        "precision_wilson_95": wilson_interval(selected_wins, selected_count),
        "coverage": (
            round(selected_count / len(omega_trades), 6) if omega_trades else None
        ),
        "abstained": len(omega_trades) - selected_count,
        "false_positives": selected_count - selected_wins,
        "diagnostic_binary_action_brier_not_probability": (
            round(float(np.mean((prediction_array - outcome_array) ** 2)), 6)
            if len(outcome_array)
            else None
        ),
        "brier_note": (
            "selected=1, abstain=0인 이진 행동 진단이며 보정된 확률 예측이 아님; "
            "정확도 또는 hard gate에 사용하지 않음"
        ),
    }


def fixed_execution_classification_metrics(
    omega_trades: Sequence[CandidateTrade],
    decisions: Sequence[MetaDecision],
    filled_trades: Sequence[CandidateTrade],
    capacity_rejects: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """고정 Ω 분모에서 실제 체결 precision과 용량 기권을 계산한다."""

    omega_keys = [_trade_key(trade) for trade in omega_trades]
    decision_keys = [
        (decision.symbol, pd.Timestamp(decision.entry_time).isoformat())
        for decision in decisions
    ]
    selected_keys = [
        key for key, decision in zip(decision_keys, decisions, strict=True) if decision.selected
    ]
    filled_keys = [_trade_key(trade) for trade in filled_trades]
    reject_keys = [
        (
            str(record.get("symbol", "")),
            pd.Timestamp(str(record.get("entry_time", ""))).isoformat(),
        )
        for record in capacity_rejects
    ]
    named_keys = {
        "Ω": omega_keys,
        "메타 결정": decision_keys,
        "선택": selected_keys,
        "체결": filled_keys,
        "용량 거절": reject_keys,
    }
    for label, keys in named_keys.items():
        if len(keys) != len(set(keys)):
            raise ValueError(f"{label} 심볼·진입시각 키가 중복됐습니다.")
    omega_set = set(omega_keys)
    decision_set = set(decision_keys)
    selected_set = set(selected_keys)
    filled_set = set(filled_keys)
    reject_set = set(reject_keys)
    if decision_set != omega_set:
        raise ValueError("실제 실행 메타 결정과 고정 Ω가 정확히 일치하지 않습니다.")
    if not selected_set.issubset(omega_set):
        raise ValueError("실제 실행 선택이 고정 Ω 밖 기회를 포함합니다.")
    if not filled_set.isdisjoint(reject_set):
        raise ValueError("동일 선택 기회가 체결과 용량 거절에 동시에 기록됐습니다.")
    if filled_set | reject_set != selected_set:
        raise ValueError("선택 Ω가 체결 또는 용량 거절로 정확히 보존되지 않았습니다.")

    wins = sum(float(trade.net_r) > 0.0 for trade in filled_trades)
    filled = len(filled_trades)
    omega_count = len(omega_trades)
    selected = len(selected_keys)
    return {
        "denominator": "fixed_v5_core_shadow_omega",
        "omega_opportunities": omega_count,
        "meta_selected": selected,
        "meta_abstained": omega_count - selected,
        "capacity_rejects_as_abstain": len(reject_keys),
        "filled": filled,
        "filled_wins": wins,
        "precision": round(wins / filled, 6) if filled else None,
        "precision_wilson_95": wilson_interval(wins, filled),
        "execution_coverage": round(filled / omega_count, 6) if omega_count else None,
        "selected_fill_rate": round(filled / selected, 6) if selected else None,
        "total_abstained_including_capacity": omega_count - filled,
        "false_positives": filled - wins,
    }


def precision_calendar_block_bootstrap(
    trades: Sequence[CandidateTrade],
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
    block_days: int,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """일별 wins·accepted를 같은 달력 블록으로 뽑아 precision을 재표집한다."""

    start = pd.Timestamp(calendar_start)
    end = pd.Timestamp(calendar_end)
    if start.tz is None or end.tz is None:
        raise ValueError("precision bootstrap 달력 경계는 timezone-aware여야 합니다.")
    start = start.tz_convert("UTC").normalize()
    end = end.tz_convert("UTC").normalize()
    if end < start:
        raise ValueError("precision bootstrap 달력 종료가 시작보다 빠릅니다.")
    if block_days <= 0 or samples <= 0:
        raise ValueError("precision bootstrap block과 sample 수는 양수여야 합니다.")

    calendar = pd.date_range(start, end, freq="1D", tz="UTC")
    accepted = np.zeros(len(calendar), dtype=float)
    wins = np.zeros(len(calendar), dtype=float)
    seen: set[tuple[str, str]] = set()
    for trade in trades:
        key = _trade_key(trade)
        if key in seen:
            raise ValueError("precision bootstrap 입력 거래 키가 중복됐습니다.")
        seen.add(key)
        entered = pd.Timestamp(trade.entry_time)
        if entered.tz is None:
            raise ValueError("precision bootstrap 진입시각은 timezone-aware여야 합니다.")
        day = entered.tz_convert("UTC").normalize()
        if day < start or day > end:
            raise ValueError("precision bootstrap 거래가 고정 달력 밖에 있습니다.")
        position = int((day - start) / pd.Timedelta(days=1))
        accepted[position] += 1.0
        wins[position] += float(trade.net_r > 0.0)

    if not trades:
        return {
            "status": "insufficient_no_accepted_events",
            "method": "joint_daily_wins_accepted_circular_moving_block",
            "samples": samples,
            "calendar_days": len(calendar),
            "block_days": block_days,
            "zero_accepted_samples": samples,
            "precision_p05": None,
            "precision_p50": None,
            "precision_p95": None,
            "probability_precision_gt_0_50": 0.0,
        }

    generator = np.random.default_rng(seed + block_days)
    offsets = np.arange(block_days)
    blocks_needed = math.ceil(len(calendar) / block_days)
    precisions = np.zeros(samples, dtype=float)
    zero_accepted = 0
    batch_size = 256
    for first in range(0, samples, batch_size):
        batch = min(batch_size, samples - first)
        starts = generator.integers(0, len(calendar), size=(batch, blocks_needed))
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(calendar)
        paths = indices.reshape(batch, -1)[:, : len(calendar)]
        accepted_totals = accepted[paths].sum(axis=1)
        win_totals = wins[paths].sum(axis=1)
        valid = accepted_totals > 0.0
        zero_accepted += int((~valid).sum())
        precisions[first : first + batch] = np.divide(
            win_totals,
            accepted_totals,
            out=np.zeros(batch, dtype=float),
            where=valid,
        )

    status = "ok" if zero_accepted == 0 else "insufficient_zero_accepted_paths"
    return {
        "status": status,
        "method": "joint_daily_wins_accepted_circular_moving_block",
        "samples": samples,
        "calendar_days": len(calendar),
        "block_days": block_days,
        "zero_accepted_samples": zero_accepted,
        "precision_p05": round(float(np.quantile(precisions, 0.05)), 6),
        "precision_p50": round(float(np.quantile(precisions, 0.50)), 6),
        "precision_p95": round(float(np.quantile(precisions, 0.95)), 6),
        "probability_precision_gt_0_50": round(
            float(np.mean(precisions > REQUIRED_PRECISION)),
            6,
        ),
    }


def precision_bootstrap_suite(
    trades: Sequence[CandidateTrade],
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
) -> dict[str, Any]:
    """14·28·56·84일 fixed-calendar precision bootstrap을 반환한다."""

    return {
        f"{block_days}d": precision_calendar_block_bootstrap(
            trades,
            calendar_start,
            calendar_end,
            block_days,
        )
        for block_days in BOOTSTRAP_BLOCK_DAYS
    }


def gate_component_pass_counts(decisions: Sequence[MetaDecision]) -> dict[str, Any]:
    """네 A+ 조건의 단독 및 모든 교집합 통과 수를 감사용으로 반환한다."""

    fields = (
        "funding_pass",
        "return_pass",
        "body_pass",
        "btc_regime_pass",
    )
    intersections: dict[str, int] = {}
    for size in range(2, len(fields) + 1):
        for chosen in combinations(fields, size):
            intersections["&".join(chosen)] = sum(
                all(bool(getattr(decision, field)) for field in chosen)
                for decision in decisions
            )
    return {
        "omega_opportunities": len(decisions),
        "single_condition_pass_counts": {
            field: sum(bool(getattr(decision, field)) for decision in decisions)
            for field in fields
        },
        "intersection_pass_counts": intersections,
        "all_four_intersection": intersections["&".join(fields)],
    }


def _updated_target(
    fills: Sequence[tuple[float, float, pd.Timestamp]],
    stop: float,
    target_r: float,
) -> float | None:
    """현재 분할체결의 평균가와 총 stop 위험에 맞춘 목표가를 반환한다."""

    if target_r <= 0.0:
        return None
    quantity = sum(fill_quantity for _, fill_quantity, _ in fills)
    average_entry = sum(
        price * fill_quantity for price, fill_quantity, _ in fills
    ) / quantity
    risk_per_quantity = average_entry - stop
    if risk_per_quantity <= 0.0:
        raise ValueError("long 분할체결 평균가가 최후손절보다 높아야 합니다.")
    return average_entry + target_r * risk_per_quantity


def simulate_fixed_omega_opportunity(
    frame: pd.DataFrame,
    featured: pd.DataFrame,
    omega_trade: CandidateTrade,
    params: CandidateParams,
    funding: pd.Series,
) -> CandidateTrade:
    """Ω의 고정 신호·다음 시가만 사용해 한 실행 변형을 정확히 재생한다."""

    if omega_trade.direction != "long":
        raise ValueError("V6 실행 변형은 long Ω만 지원합니다.")
    entry_time = pd.Timestamp(omega_trade.entry_time)
    signal_time = entry_time - pd.Timedelta(hours=1)
    if signal_time not in featured.index or entry_time not in featured.index:
        raise ValueError("Ω 신호 또는 진입시각이 실행 프레임에 없습니다.")
    signal_index = int(featured.index.get_loc(signal_time))
    entry_index = int(featured.index.get_loc(entry_time))
    if entry_index != signal_index + 1:
        raise ValueError("Ω 체결은 신호 다음 1시간 시가여야 합니다.")
    entry = float(featured.iloc[entry_index]["open"])
    if not math.isclose(entry, float(omega_trade.entry), rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("Ω 기록 진입가와 원시 다음 시가가 일치하지 않습니다.")
    atr = float(featured.iloc[signal_index]["atr_entry"])
    if not math.isfinite(atr) or atr <= 0.0:
        raise ValueError("Ω 신호의 직전 확정 ATR이 유효하지 않습니다.")
    weights = np.asarray(params.tranche_weights, dtype=float)
    fractions = params.add_fractions
    if len(weights) != len(fractions) + 1 or (weights < 0.0).any() or weights.sum() <= 0.0:
        raise ValueError("실행 변형의 분할 위험 배분이 잘못됐습니다.")
    weights = weights / weights.sum()
    stop_distance = atr * params.stop_atr
    stop = entry - stop_distance
    if stop <= 0.0:
        raise ValueError("계산된 long 최후손절이 0 이하여서 실행할 수 없습니다.")
    add_levels = [entry - stop_distance * fraction for fraction in fractions]
    fills: list[tuple[float, float, pd.Timestamp]] = [
        (entry, float(weights[0]) / stop_distance, entry_time)
    ]
    average_entry = entry
    active_target = _updated_target(fills, stop, params.target_r)
    pending_target: float | None = None
    pending_target_activation_index: int | None = None
    add_done = [False] * len(add_levels)
    pending_add: int | None = None
    failed_breakout_exit_pending = False
    reviewed = False
    active_exit = stop
    exit_index = entry_index
    exit_price = entry
    exit_reason = "end_of_data"
    signal = featured.iloc[signal_index]

    for cursor in range(entry_index, len(featured)):
        if (
            pending_target_activation_index is not None
            and cursor >= pending_target_activation_index
        ):
            active_target = pending_target
            pending_target = None
            pending_target_activation_index = None
        candle = featured.iloc[cursor]
        candle_time = featured.index[cursor]
        candle_open = float(candle["open"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])
        candle_close = float(candle["close"])
        elapsed_hours = (candle_time - entry_time).total_seconds() / 3600.0

        raw_channel = float(candle["exit_low"])
        if math.isfinite(raw_channel):
            active_exit = max(active_exit, stop, raw_channel)

        if cursor > entry_index:
            if candle_open <= active_exit:
                exit_price = candle_open
                exit_index = cursor
                exit_reason = "stop_gap" if active_exit == stop else "channel_exit"
                break
            if active_target is not None and candle_open >= active_target:
                exit_price = active_target
                exit_index = cursor
                exit_reason = "target_gap"
                break
            if failed_breakout_exit_pending:
                exit_price = candle_open
                exit_index = cursor
                exit_reason = "failed_breakout_exit"
                break
            if elapsed_hours >= params.max_holding_hours:
                exit_price = candle_open
                exit_index = cursor
                exit_reason = "time_exit_max_hold"
                break
            if not reviewed and elapsed_hours >= params.review_holding_hours:
                reviewed = True
                if not bool(candle["long_gate"]):
                    exit_price = candle_open
                    exit_index = cursor
                    exit_reason = "time_review_exit"
                    break
            if pending_add is not None:
                adverse_open = candle_open < average_entry
                inside_final_stop = candle_open > stop
                if adverse_open and inside_final_stop:
                    actual_distance = candle_open - stop
                    addition_quantity = float(weights[pending_add + 1]) / actual_distance
                    fills.append((candle_open, addition_quantity, candle_time))
                    add_done[pending_add] = True
                    total_quantity = sum(quantity for _, quantity, _ in fills)
                    average_entry = sum(
                        price * quantity for price, quantity, _ in fills
                    ) / total_quantity
                    pending_target = _updated_target(fills, stop, params.target_r)
                    pending_target_activation_index = cursor + 1
                    active_target = None
                pending_add = None

        if candle_low <= stop:
            exit_price = stop
            exit_index = cursor
            exit_reason = "same_bar_stop" if cursor == entry_index else "stop"
            break
        if candle_low <= active_exit:
            exit_price = entry if cursor == entry_index and active_exit >= entry else active_exit
            exit_index = cursor
            exit_reason = "channel_exit"
            break
        if active_target is not None and candle_high >= active_target:
            exit_price = active_target
            exit_index = cursor
            exit_reason = "target"
            break

        bars_since_entry = cursor - entry_index + 1
        breakout_level = float(signal["entry_high"])
        if (
            params.failed_breakout_exit_hours > 0
            and bars_since_entry <= params.failed_breakout_exit_hours
            and candle_close < breakout_level
        ):
            failed_breakout_exit_pending = True

        if pending_add is None and not failed_breakout_exit_pending:
            next_add = next(
                (position for position, done in enumerate(add_done) if not done),
                None,
            )
            if next_add is not None:
                level = add_levels[next_add]
                reclaimed = (
                    candle_low <= level
                    and candle_close > level
                    and candle_close > candle_open
                )
                if bool(candle["long_gate"]) and reclaimed:
                    pending_add = next_add

        exit_index = cursor
        exit_price = candle_close

    if exit_reason == "end_of_data":
        raise ValueError("embargo 뒤에도 실행 변형 거래가 종료되지 않았습니다.")
    exit_time = featured.index[exit_index]
    holding_hours = max(0, int((exit_time - entry_time).total_seconds() // 3600))
    average_entry, gross_r, execution_cost, funding_cost, net_r = trade_pnl(
        fills,
        1,
        exit_price,
        exit_time,
        funding,
        featured["open"],
        params,
    )
    committed = float(
        weights[0]
        + sum(weights[position + 1] for position, done in enumerate(add_done) if done)
    )
    return CandidateTrade(
        symbol=omega_trade.symbol,
        entry_time=entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        direction="long",
        entry=round(entry, 8),
        average_entry=round(average_entry, 8),
        stop=round(stop, 8),
        target=round(active_target, 8) if active_target is not None else None,
        exit=round(exit_price, 8),
        exit_reason=exit_reason,
        holding_hours=holding_hours,
        additions=sum(add_done),
        risk_committed_r=round(committed, 8),
        gross_r=round(gross_r, 8),
        execution_cost_r=round(execution_cost, 8),
        funding_cost_r=round(funding_cost, 8),
        net_r=round(net_r, 8),
    )


def replay_execution_variant(
    frames: Mapping[str, pd.DataFrame],
    funding_frame: pd.DataFrame,
    selected_omega: Sequence[CandidateTrade],
    params: CandidateParams,
    omega_params: CandidateParams,
) -> ExecutionReplay:
    """선택 Ω만 심볼별 시간순 재생하고 바쁜 엔진의 거절을 기록한다."""

    validate_execution_contract(params, omega_params)
    trades: list[CandidateTrade] = []
    rejects: list[dict[str, str]] = []
    for symbol in SYMBOLS:
        if symbol not in frames or symbol not in funding_frame:
            raise ValueError(f"실행 재생 입력에서 {symbol}이 누락됐습니다.")
        featured = add_features(frames[symbol], params)
        active_until: pd.Timestamp | None = None
        opportunities = sorted(
            [trade for trade in selected_omega if trade.symbol == symbol],
            key=lambda trade: (trade.entry_time, trade.exit_time),
        )
        for omega_trade in opportunities:
            entry_time = pd.Timestamp(omega_trade.entry_time)
            if active_until is not None and entry_time <= active_until:
                rejects.append(
                    {
                        "symbol": symbol,
                        "signal_time": (entry_time - pd.Timedelta(hours=1)).isoformat(),
                        "entry_time": entry_time.isoformat(),
                        "active_until": active_until.isoformat(),
                        "reason": "capacity_reject_existing_position",
                    }
                )
                continue
            replayed = simulate_fixed_omega_opportunity(
                frames[symbol],
                featured,
                omega_trade,
                params,
                funding_frame[symbol].dropna(),
            )
            trades.append(replayed)
            active_until = pd.Timestamp(replayed.exit_time)
    return ExecutionReplay(
        trades=tuple(sorted(trades, key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time))),
        capacity_rejects=tuple(rejects),
    )


def macro_episode_removal(trades: Sequence[CandidateTrade]) -> dict[str, Any]:
    """가장 수익이 큰 고정 UTC 달력 주간의 모든 진입을 제거한다."""

    trade_list = list(trades)
    grouped: dict[str, list[CandidateTrade]] = {}
    for trade in trade_list:
        entry_time = pd.Timestamp(trade.entry_time)
        naive = entry_time.tz_convert("UTC").tz_localize(None)
        period = naive.to_period(MACRO_EPISODE_FREQUENCY)
        key = f"{period.start_time.date().isoformat()}/{period.end_time.date().isoformat()}"
        grouped.setdefault(key, []).append(trade)
    if not grouped:
        return {
            "method": "fixed_utc_calendar_week_monday_to_sunday",
            "episodes": 0,
            "removed": None,
            "retained_summary": portfolio_summary([]),
        }
    ranked = sorted(
        grouped,
        key=lambda key: (
            -sum(trade.net_r for trade in grouped[key]),
            key,
        ),
    )
    removed_key = ranked[0]
    removed_trades = grouped[removed_key]
    retained = [
        trade
        for key, episode_trades in grouped.items()
        if key != removed_key
        for trade in episode_trades
    ]
    return {
        "method": "fixed_utc_calendar_week_monday_to_sunday",
        "independence_claim": False,
        "episodes": len(grouped),
        "removed": {
            "episode": removed_key,
            "trades": len(removed_trades),
            "symbols": sorted({trade.symbol for trade in removed_trades}),
            "net_r": round(float(sum(trade.net_r for trade in removed_trades)), 6),
        },
        "retained_summary": portfolio_summary(retained),
    }


def _trade_suite(
    strict_trades: Sequence[CandidateTrade],
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
    calendar_end: pd.Timestamp,
) -> dict[str, Any]:
    """한 실행 경로의 strict·severe·차원·빈도·bootstrap을 묶는다."""

    strict = list(strict_trades)
    severe = apply_execution_funding_stress(
        strict,
        original_cost_bps_side=STRICT_COST_BPS_SIDE,
        stressed_cost_bps_side=SEVERE_COST_BPS_SIDE,
    )
    strict_summary = portfolio_summary(strict)
    severe_summary = portfolio_summary(severe)
    strict_bootstrap = bootstrap_suite(strict, coverage_start, calendar_end)
    severe_bootstrap = bootstrap_suite(severe, coverage_start, calendar_end)
    return {
        "strict_12bp_actual_funding": {
            "summary": strict_summary,
            "dimensions": dimension_summary(strict, coverage_start, coverage_end),
            "bootstrap": strict_bootstrap,
            "top_winners": top_winner_analysis(strict),
            "top_macro_episode_removal": macro_episode_removal(strict),
        },
        "severe_20bp_funding_debit_x2_credit_zero": {
            "summary": severe_summary,
            "dimensions": dimension_summary(severe, coverage_start, coverage_end),
            "bootstrap": severe_bootstrap,
            "top_winners": top_winner_analysis(severe),
            "top_macro_episode_removal": macro_episode_removal(severe),
        },
        "frequency_unique": unique_complete_month_frequency(
            strict,
            coverage_start,
            coverage_end,
        ),
        "frequency_risk_equivalent": risk_equivalent_complete_month_frequency(
            strict,
            coverage_start,
            coverage_end,
        ),
        "clusters_6h": six_hour_unique_entry_clusters(strict),
        "concurrent_heat": max_weighted_concurrent_heat(
            strict,
            risk_percent=BASE_RISK_PERCENT,
        ),
    }


def _trial_metrics(suite: Mapping[str, Any]) -> ParetoMetrics:
    """검증 suite를 append-only discovery 원장의 최소 지표로 변환한다."""

    strict = suite["strict_12bp_actual_funding"]
    summary = strict["summary"]
    frequency = suite["frequency_risk_equivalent"]
    required = (
        summary.get("profit_factor"),
        summary.get("risk_normalized_expectancy_r"),
        frequency.get("median_per_month"),
    )
    if any(value is None for value in required):
        raise ValueError("원장에 기록할 precision trial 지표가 불완전합니다.")
    return ParetoMetrics(
        profit_factor=float(summary["profit_factor"]),
        expectancy_r=float(summary["risk_normalized_expectancy_r"]),
        net_r=float(summary["net_r"]),
        max_drawdown_r=float(summary["realized_max_drawdown_r"]),
        bootstrap_mdd_p95_r=conservative_bootstrap_mdd(strict["bootstrap"]),
        trades_per_month=float(frequency["median_per_month"]),
    )


def precision_code_hash() -> str:
    """현재 러너와 체결·coverage·원장 의존성의 결합 SHA256을 반환한다."""

    return sha256_files(
        [
            Path(__file__),
            ROOT / "lab" / "validate_live_candidate.py",
            ROOT / "lab" / "validate_pareto_candidate.py",
            ROOT / "lab" / "validate_pareto_ensemble.py",
            ROOT / "lab" / "pareto_trial_ledger.py",
        ],
        root=ROOT,
    )


def discovery_hard_fail_gate() -> dict[str, Any]:
    """성과로 덮어쓸 수 없는 discovery-only 하드 실패를 반환한다."""

    return {
        "status": "FAIL",
        "promotion_allowed": False,
        "hard_fail_reason": (
            "A+ 임계값은 이미 관측한 자료의 다중 탐색에서 선택됐고 "
            "사전등록 미래표본이 없음"
        ),
        "performance_cannot_override_hard_fail": True,
    }


def build_statistical_conditions(
    meta_classifications: Mapping[str, Mapping[str, Any]],
    execution_classifications: Mapping[str, Mapping[str, Any]],
    meta_suite: Mapping[str, Any],
    execution_suite: Mapping[str, Any],
    meta_precision_bootstrap: Mapping[str, Mapping[str, Mapping[str, Any]]],
    execution_precision_bootstrap: Mapping[str, Mapping[str, Mapping[str, Any]]],
    execution_trades: Sequence[CandidateTrade],
) -> dict[str, Any]:
    """사전 고정한 precision·coverage·PF·bootstrap 조건을 모두 판정한다."""

    conditions: list[dict[str, Any]] = []

    def add_condition(
        name: str,
        observed: Any,
        required: Any,
        passed: bool,
    ) -> None:
        """한 조건의 관측값·요구값·판정을 직렬화 가능한 형태로 추가한다."""

        conditions.append(
            {
                "name": name,
                "observed": observed,
                "required": required,
                "pass": bool(passed),
            }
        )

    severity_contract = {
        "strict": {
            "suite_key": "strict_12bp_actual_funding",
            "precision": 0.60,
            "profit_factor": 1.50,
            "expectancy": 0.12,
            "bootstrap_p05": 0.52,
        },
        "severe": {
            "suite_key": "severe_20bp_funding_debit_x2_credit_zero",
            "precision": 0.55,
            "profit_factor": 1.20,
            "expectancy": 0.04,
            "bootstrap_p05": 0.50,
        },
    }
    for route, classifications in (
        ("meta_shadow", meta_classifications),
        ("actual_execution", execution_classifications),
    ):
        for severity, contract in severity_contract.items():
            observed = classifications[severity].get("precision")
            required = contract["precision"]
            add_condition(
                f"{route}_{severity}_precision",
                observed,
                {"min": required},
                observed is not None and float(observed) >= required,
            )

    raw_coverage = meta_classifications["strict"].get("coverage")
    execution_coverage = execution_classifications["strict"].get("execution_coverage")
    omega_count = int(meta_classifications["strict"].get("omega_opportunities", 0))
    committed = execution_suite["strict_12bp_actual_funding"]["summary"].get(
        "risk_committed_r"
    )
    risk_coverage = (
        float(committed) / omega_count
        if committed is not None and omega_count > 0
        else None
    )
    add_condition(
        "raw_meta_coverage",
        raw_coverage,
        {"min": 0.20, "denominator": "fixed_omega"},
        raw_coverage is not None and float(raw_coverage) >= 0.20,
    )
    add_condition(
        "actual_execution_coverage",
        execution_coverage,
        {"min": 0.15, "denominator": "fixed_omega"},
        execution_coverage is not None and float(execution_coverage) >= 0.15,
    )
    add_condition(
        "actual_risk_coverage",
        round(risk_coverage, 6) if risk_coverage is not None else None,
        {"min": 0.15, "definition": "filled risk_committed_r / fixed_omega"},
        risk_coverage is not None and risk_coverage >= 0.15,
    )
    selected = int(meta_classifications["strict"].get("selected", 0))
    filled = int(execution_classifications["strict"].get("filled", 0))
    add_condition("meta_selected_count", selected, {"min": 300}, selected >= 300)
    add_condition("actual_filled_count", filled, {"min": 300}, filled >= 300)
    for stage in range(1, 5):
        reached = sum(int(trade.additions) >= stage for trade in execution_trades)
        add_condition(
            f"actual_execution_addition_stage_{stage}_count",
            reached,
            {"min": 30, "definition": f"count(additions >= {stage})"},
            reached >= 30,
        )

    for route, suite in (("meta_shadow", meta_suite), ("actual_execution", execution_suite)):
        for severity, contract in severity_contract.items():
            summary = suite[contract["suite_key"]]["summary"]
            profit_factor = summary.get("profit_factor")
            expectancy = summary.get("risk_normalized_expectancy_r")
            add_condition(
                f"{route}_{severity}_profit_factor",
                profit_factor,
                {"min": contract["profit_factor"]},
                profit_factor is not None
                and float(profit_factor) >= contract["profit_factor"],
            )
            add_condition(
                f"{route}_{severity}_risk_normalized_expectancy_r",
                expectancy,
                {"min": contract["expectancy"]},
                expectancy is not None and float(expectancy) >= contract["expectancy"],
            )

    for route, bootstraps in (
        ("meta_shadow", meta_precision_bootstrap),
        ("actual_execution", execution_precision_bootstrap),
    ):
        for severity, contract in severity_contract.items():
            for block_days in BOOTSTRAP_BLOCK_DAYS:
                key = f"{block_days}d"
                result = bootstraps[severity][key]
                p05 = result.get("precision_p05")
                probability = result.get("probability_precision_gt_0_50")
                required = {
                    "status": "ok",
                    "precision_p05_min": contract["bootstrap_p05"],
                    "probability_precision_gt_0_50_min": (
                        REQUIRED_PRECISION_BOOTSTRAP_PROBABILITY
                    ),
                }
                passed = (
                    result.get("status") == "ok"
                    and p05 is not None
                    and float(p05) >= contract["bootstrap_p05"]
                    and probability is not None
                    and float(probability)
                    >= REQUIRED_PRECISION_BOOTSTRAP_PROBABILITY
                )
                add_condition(
                    f"{route}_{severity}_precision_bootstrap_{key}",
                    {
                        "status": result.get("status"),
                        "precision_p05": p05,
                        "probability_precision_gt_0_50": probability,
                        "zero_accepted_samples": result.get("zero_accepted_samples"),
                    },
                    required,
                    passed,
                )

    for route, suite in (("meta_shadow", meta_suite), ("actual_execution", execution_suite)):
        for severity, contract in severity_contract.items():
            symbols = suite[contract["suite_key"]]["dimensions"]["symbols"]
            observed_symbols = {
                symbol: {
                    "trades": summary.get("component_trades", 0),
                    "precision": summary.get("component_win_rate"),
                }
                for symbol, summary in symbols.items()
            }
            all_symbols_pass = set(observed_symbols) == set(SYMBOLS) and all(
                int(values["trades"]) >= 30
                and values["precision"] is not None
                and float(values["precision"]) >= REQUIRED_PRECISION
                for values in observed_symbols.values()
            )
            add_condition(
                f"{route}_{severity}_all_5_symbols",
                observed_symbols,
                {"symbols": list(SYMBOLS), "min_trades_each": 30, "precision_min": 0.50},
                all_symbols_pass,
            )

    failed = [condition["name"] for condition in conditions if not condition["pass"]]
    return {
        "statistical_conditions_pass": not failed,
        "failed_conditions": failed,
        "conditions": conditions,
    }


def _append_trials(
    ledger_path: Path,
    omega_params: CandidateParams,
    gate_spec: PrecisionGateSpec,
    execution_params: CandidateParams,
    hashes: Mapping[str, str],
    code_hash: str,
    meta_suite: Mapping[str, Any],
    execution_suite: Mapping[str, Any],
    coverage: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    """A+ 메타와 실행 변형을 해시체인 JSONL에 멱등 append한다."""

    ledger = ParetoTrialLedger(ledger_path)
    common_metadata = {
        "status": "FAIL_DISCOVERY_ONLY",
        "promotion_allowed": False,
        "omega_denominator": "fixed_v5_core_shadow_replay",
        "common_signal_start": coverage["common_signal_start"],
        "common_signal_end": coverage["common_signal_end"],
        "follow_up_embargo_hours": FOLLOW_UP_EMBARGO_HOURS,
    }
    meta = ledger.append_success(
        trial_name="precision_a_plus_meta_on_fixed_v5_core_omega",
        params={
            "omega": asdict(omega_params),
            "gate": asdict(gate_spec),
            "selection_only_on_omega": True,
        },
        data_hashes=hashes,
        code_hash=code_hash,
        metrics=_trial_metrics(meta_suite),
        metadata={
            **common_metadata,
            "role": "primary_fixed_denominator_precision",
            "precision": classification["precision"],
            "coverage": classification["coverage"],
            "diagnostic_binary_action_brier_not_probability": classification[
                "diagnostic_binary_action_brier_not_probability"
            ],
        },
    )
    execution = ledger.append_success(
        trial_name="precision_a_plus_fixed_omega_execution_24h_target1_add4",
        params={
            "omega": asdict(omega_params),
            "gate": asdict(gate_spec),
            "execution": asdict(execution_params),
            "new_raw_opportunities_forbidden": True,
        },
        data_hashes=hashes,
        code_hash=code_hash,
        metrics=_trial_metrics(execution_suite),
        metadata={
            **common_metadata,
            "role": "secondary_execution_variant",
        },
    )
    return {
        "path": str(ledger_path),
        "meta": {
            "trial_id": meta.trial_id,
            "appended": meta.appended,
            "record_hash": meta.record["record_hash"],
        },
        "execution": {
            "trial_id": execution.trial_id,
            "appended": execution.appended,
            "record_hash": execution.record["record_hash"],
        },
    }


def _coverage_manifest(
    frames: Mapping[str, pd.DataFrame],
    ready_masks: Mapping[str, pd.Series],
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> dict[str, Any]:
    """고정 Ω 공통 coverage와 심볼별 데이터 경계를 반환한다."""

    symbols: dict[str, Any] = {}
    for symbol in SYMBOLS:
        ready_index = ready_masks[symbol].index[ready_masks[symbol]]
        symbols[symbol] = {
            "rows": len(frames[symbol]),
            "data_first": frames[symbol].index[0].isoformat(),
            "data_last": frames[symbol].index[-1].isoformat(),
            "ready_first": ready_index[0].isoformat(),
            "ready_last": ready_index[-1].isoformat(),
        }
    return {
        "policy": "V5 core q60 full365 readiness 5-symbol intersection then 73h embargo",
        "common_signal_start": signal_start.isoformat(),
        "common_signal_end": signal_end.isoformat(),
        "entry_coverage_start": (signal_start + pd.Timedelta(hours=1)).isoformat(),
        "entry_coverage_end": (signal_end + pd.Timedelta(hours=1)).isoformat(),
        "follow_up_embargo_hours": FOLLOW_UP_EMBARGO_HOURS,
        "symbols": symbols,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """완전한 JSON만 보이도록 임시파일을 같은 디렉터리에서 원자 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_validation(
    output_dir: Path = OUTPUT_DIR,
    ledger_path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    """고정 Ω A+ 메타 선택과 분리된 V6 실행 변형을 검증·기록한다."""

    omega_params = fixed_omega_params()
    validate_omega_contract(omega_params)
    execution_params = fixed_execution_variant_params(omega_params)
    validate_execution_contract(execution_params, omega_params)
    gate_spec = PrecisionGateSpec()
    (
        frames,
        funding_frame,
        ready_masks,
        common_signal_start,
        common_signal_end,
    ) = prepare_inputs(omega_params)
    common_signal_end -= pd.Timedelta(hours=FOLLOW_UP_EMBARGO_HOURS)
    if common_signal_end <= common_signal_start:
        raise ValueError("73시간 embargo 뒤 고정 Ω coverage가 비었습니다.")

    omega_by_symbol = replay_matched(
        frames,
        funding_frame,
        ready_masks,
        omega_params,
        common_signal_start,
        common_signal_end,
    )
    omega_trades = sorted(
        [trade for symbol in SYMBOLS for trade in omega_by_symbol[symbol]],
        key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time),
    )
    if not omega_trades:
        raise ValueError("고정 V5 core Ω 기회가 없습니다.")

    decisions: list[MetaDecision] = []
    for symbol in SYMBOLS:
        components = build_precision_gate_components(
            frames[symbol],
            symbol,
            frames["BTC"],
            funding_frame[symbol].dropna(),
            omega_params,
            gate_spec,
        )
        decisions.extend(build_meta_decisions(omega_by_symbol[symbol], components))
    selected = select_fixed_omega_trades(omega_trades, decisions)

    entry_coverage_start = common_signal_start + pd.Timedelta(hours=1)
    entry_coverage_end = common_signal_end + pd.Timedelta(hours=1)
    calendar_end = min(frame.index[-1] for frame in frames.values())
    omega_suite = _trade_suite(
        omega_trades,
        entry_coverage_start,
        entry_coverage_end,
        calendar_end,
    )
    meta_suite = _trade_suite(
        selected,
        entry_coverage_start,
        entry_coverage_end,
        calendar_end,
    )
    strict_classification = fixed_omega_classification_metrics(
        omega_trades,
        decisions,
    )
    severe_omega = apply_execution_funding_stress(
        omega_trades,
        STRICT_COST_BPS_SIDE,
        SEVERE_COST_BPS_SIDE,
    )
    severe_classification = fixed_omega_classification_metrics(
        severe_omega,
        decisions,
    )
    severe_selected = apply_execution_funding_stress(
        selected,
        STRICT_COST_BPS_SIDE,
        SEVERE_COST_BPS_SIDE,
    )
    meta_precision_bootstrap = {
        "strict": precision_bootstrap_suite(
            selected,
            entry_coverage_start,
            entry_coverage_end,
        ),
        "severe": precision_bootstrap_suite(
            severe_selected,
            entry_coverage_start,
            entry_coverage_end,
        ),
    }

    execution = replay_execution_variant(
        frames,
        funding_frame,
        selected,
        execution_params,
        omega_params,
    )
    execution_suite = _trade_suite(
        execution.trades,
        entry_coverage_start,
        entry_coverage_end,
        calendar_end,
    )
    severe_execution_trades = apply_execution_funding_stress(
        execution.trades,
        STRICT_COST_BPS_SIDE,
        SEVERE_COST_BPS_SIDE,
    )
    execution_strict_classification = fixed_execution_classification_metrics(
        omega_trades,
        decisions,
        execution.trades,
        execution.capacity_rejects,
    )
    execution_severe_classification = fixed_execution_classification_metrics(
        omega_trades,
        decisions,
        severe_execution_trades,
        execution.capacity_rejects,
    )
    execution_precision_bootstrap = {
        "strict": precision_bootstrap_suite(
            execution.trades,
            entry_coverage_start,
            entry_coverage_end,
        ),
        "severe": precision_bootstrap_suite(
            severe_execution_trades,
            entry_coverage_start,
            entry_coverage_end,
        ),
    }
    statistical_gate = build_statistical_conditions(
        {"strict": strict_classification, "severe": severe_classification},
        {
            "strict": execution_strict_classification,
            "severe": execution_severe_classification,
        },
        meta_suite,
        execution_suite,
        meta_precision_bootstrap,
        execution_precision_bootstrap,
        execution.trades,
    )
    coverage = _coverage_manifest(
        frames,
        ready_masks,
        common_signal_start,
        common_signal_end,
    )
    hashes = data_hash_manifest()
    code_hash = precision_code_hash()
    ledger = _append_trials(
        ledger_path,
        omega_params,
        gate_spec,
        execution_params,
        hashes,
        code_hash,
        meta_suite,
        execution_suite,
        coverage,
        strict_classification,
    )

    decision_rows = [asdict(decision) for decision in decisions]
    selected_keys = {
        (decision.symbol, decision.entry_time)
        for decision in decisions
        if decision.selected
    }
    result: dict[str, Any] = {
        "classification": DISCOVERY_CLASSIFICATION,
        "promotion_capability": PROMOTION_CAPABILITY,
        "gate": {**discovery_hard_fail_gate(), **statistical_gate},
        "contracts": {
            "fixed_omega": asdict(omega_params),
            "precision_gate": asdict(gate_spec),
            "execution_variant": asdict(execution_params),
            "opportunity_policy": {
                "denominator": "fixed_v5_core_shadow_replay",
                "meta_selects_only_existing_omega": True,
                "new_raw_signals_after_early_exit_forbidden": True,
                "capacity_rejects_are_abstentions_not_new_denominator": True,
            },
        },
        "coverage": coverage,
        "data_hashes": hashes,
        "code_hash": code_hash,
        "fixed_omega": {
            "opportunities": len(omega_trades),
            "unique_events": len(unique_entry_groups(omega_trades)),
            "suite": omega_suite,
        },
        "a_plus_meta_selection": {
            "selected_keys": len(selected_keys),
            "gate_component_pass_counts": gate_component_pass_counts(decisions),
            "strict_fixed_omega_classification": strict_classification,
            "severe_fixed_omega_classification": severe_classification,
            "precision_calendar_block_bootstrap": meta_precision_bootstrap,
            "suite": meta_suite,
            "decision_score_distribution": {
                str(score): sum(decision.score == score for decision in decisions)
                for score in range(5)
            },
            "decisions": decision_rows,
        },
        "execution_variant_separate": {
            "ledger": {
                "omega_opportunities": len(omega_trades),
                "meta_abstains": len(omega_trades) - len(selected),
                "meta_selected": len(selected),
                "capacity_rejects": len(execution.capacity_rejects),
                "filled": len(execution.trades),
                "capacity_reject_records": list(execution.capacity_rejects),
            },
            "strict_fixed_omega_filled_classification": (
                execution_strict_classification
            ),
            "severe_fixed_omega_filled_classification": (
                execution_severe_classification
            ),
            "precision_calendar_block_bootstrap": execution_precision_bootstrap,
            "suite": execution_suite,
            "addition_distribution": {
                str(count): sum(trade.additions == count for trade in execution.trades)
                for count in range(5)
            },
            "exit_reason_distribution": {
                reason: sum(trade.exit_reason == reason for trade in execution.trades)
                for reason in sorted({trade.exit_reason for trade in execution.trades})
            },
        },
        "ledger": ledger,
        "limitations": [
            "동일 자료에서 100개가 넘는 후보를 본 뒤 고른 discovery 규칙이라 선택편향이 큼",
            "diagnostic_binary_action_brier_not_probability는 보정 확률이나 정확도 지표가 아님",
            "MTM·마진·강제청산·호가잔량·부분체결은 모델링하지 않음",
            "TradingView Pine에서 실제 거래소 funding을 동일 의미로 직접 요청할 수 있는지는 별도 확인 필요",
            "과거 성과는 미래 수익이나 50% 초과 정밀도를 보장하지 않음",
        ],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"precision_candidate_v6_{timestamp}.json"
    latest_path = output_dir / "latest_results.json"
    _atomic_write_json(run_path, result)
    _atomic_write_json(latest_path, result)
    logger.info("precision V6 검증 JSON 저장: %s", run_path)
    return result


def main() -> None:
    """명령행 인자를 읽어 precision V6 discovery 검증을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run_validation(arguments.output_dir, arguments.ledger)
    logger.info(
        "완료: status=%s omega=%d selected=%d execution=%d",
        result["gate"]["status"],
        result["fixed_omega"]["opportunities"],
        result["a_plus_meta_selection"]["selected_keys"],
        result["execution_variant_separate"]["ledger"]["filled"],
    )


if __name__ == "__main__":
    main()
