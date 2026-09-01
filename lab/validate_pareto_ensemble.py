from __future__ import annotations

"""고정 위험예산 Pareto 앙상블을 동일 coverage에서 엄격 검증한다.

이 러너는 결과를 본 뒤 파라미터를 고르는 탐색기가 아니다. 공통 coverage는
stop 7ATR, 과거 365일 선형보간 q60, 최소 30표본, full-window 규칙으로 고정한다.
후보는 서로 독립적으로 재생한 core 90%, scout 5%, fast 5% 엔진의 R 손익을
위험예산만큼 선형 축소해 합친다. 기준은 동일 coverage-ready 신호봉에서 재생한
24/12 no-filter 100% 엔진이다. 추매는 모든 엔진에서 금지한다.

성과가 좋아도 이미 관측한 자료의 discovery 결과이므로 코드상 실거래 PASS는
발급하지 않는다. MTM, 마진과 강제청산을 모델링하지 않은 한계도 명시적으로 남긴다.
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lab.pareto_trial_ledger import (
    ParetoMetrics,
    ParetoTrialLedger,
    compare_pareto,
    sha256_files,
)
from lab.validate_live_candidate import (
    ROOT,
    CandidateParams,
    CandidateTrade,
    simulate_symbol,
)
from lab.validate_pareto_candidate import (
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CLUSTER_HOURS,
    LEDGER_PATH,
    SYMBOLS,
    apply_execution_funding_stress,
    bootstrap_suite,
    complete_month_counts,
    conservative_bootstrap_mdd,
    coverage_manifest as v4_coverage_manifest,
    data_hash_manifest,
    filter_matched_coverage,
    numeric_summary_delta,
    prepare_inputs,
    replay_matched,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "logs" / "validation" / "pareto_ensemble_v5"
STRICT_COST_BPS_SIDE = 12.0
STRESS_COST_BPS_SIDE = 20.0
BASE_RISK_PERCENT = 0.25

REQUIRED_STRICT_PROFIT_FACTOR = 1.20
REQUIRED_STRESS_PROFIT_FACTOR = 1.05
REQUIRED_BOOTSTRAP_PROBABILITY_POSITIVE = 0.95
REQUIRED_CLUSTER_FREQUENCY_MULTIPLE = 1.10
REQUIRED_MONTHLY_MEDIAN_MULTIPLE = 1.10
REQUIRED_POSITIVE_SYMBOLS = len(SYMBOLS)
REQUIRED_POSITIVE_COMPLETE_YEARS = 4
MAX_WEIGHTED_CONCURRENT_HEAT_R = 5.0
PROSPECTIVE_MIN_UNIQUE_ENTRIES = 250
PROSPECTIVE_MIN_MONTHS = 12
FOLLOW_UP_EMBARGO_HOURS = 73


@dataclass(frozen=True)
class EngineSpec:
    """독립 재생 엔진의 이름·위험예산·체결 파라미터 계약이다."""

    name: str
    risk_weight: float
    params: CandidateParams


def coverage_candidate_params() -> CandidateParams:
    """공통 readiness를 결정하는 고정 q60 coverage 파라미터를 반환한다."""

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


def ensemble_engine_specs(coverage: CandidateParams) -> tuple[EngineSpec, ...]:
    """90/5/5 위험예산을 쓰는 core·scout·fast 엔진 계약을 반환한다."""

    specs = (
        EngineSpec("core_24_12_filtered", 0.90, coverage),
        EngineSpec(
            "scout_24_12_no_filter_matched_ready",
            0.05,
            replace(
                coverage,
                volatility_filter_days=0,
                volatility_filter_require_full_window=False,
            ),
        ),
        EngineSpec(
            "fast_12_6_filtered",
            0.05,
            replace(coverage, entry_channel=12, exit_channel=6),
        ),
    )
    total_weight = sum(spec.risk_weight for spec in specs)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("앙상블 위험예산 합계는 정확히 1이어야 합니다.")
    if any(spec.risk_weight <= 0.0 for spec in specs):
        raise ValueError("각 앙상블 엔진 위험예산은 0보다 커야 합니다.")
    return specs


def reference_params(coverage: CandidateParams) -> CandidateParams:
    """동일 stop·비용의 24/12 no-filter 100% 기준 파라미터를 반환한다."""

    return replace(
        coverage,
        entry_channel=24,
        exit_channel=12,
        volatility_filter_days=0,
        volatility_filter_require_full_window=False,
    )


def scale_trade(trade: CandidateTrade, risk_weight: float) -> CandidateTrade:
    """거래의 위험·손익·비용·펀딩 R만 위험예산에 맞춰 선형 축소한다."""

    if not 0.0 < risk_weight <= 1.0:
        raise ValueError("거래 위험 가중치는 0 초과 1 이하여야 합니다.")
    return replace(
        trade,
        risk_committed_r=float(trade.risk_committed_r) * risk_weight,
        gross_r=float(trade.gross_r) * risk_weight,
        execution_cost_r=float(trade.execution_cost_r) * risk_weight,
        funding_cost_r=float(trade.funding_cost_r) * risk_weight,
        net_r=float(trade.net_r) * risk_weight,
    )


def scale_trades(
    trades: Sequence[CandidateTrade],
    risk_weight: float,
) -> list[CandidateTrade]:
    """한 독립 엔진의 모든 거래를 같은 위험예산으로 선형 축소한다."""

    return [scale_trade(trade, risk_weight) for trade in trades]


def replay_scaled_engine(
    frames: Mapping[str, pd.DataFrame],
    funding_frame: pd.DataFrame,
    ready_masks: Mapping[str, pd.Series],
    spec: EngineSpec,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> dict[str, list[CandidateTrade]]:
    """한 엔진을 독립 재생하고 공통 readiness를 적용한 뒤 위험 가중한다."""

    replayed: dict[str, list[CandidateTrade]] = {}
    for symbol in SYMBOLS:
        logger.info(
            "앙상블 재생: engine=%s symbol=%s rows=%d weight=%.2f",
            spec.name,
            symbol,
            len(frames[symbol]),
            spec.risk_weight,
        )
        raw_trades = simulate_symbol(
            frames[symbol],
            symbol,
            spec.params,
            funding_frame[symbol].dropna(),
            allow_additions=False,
        )
        matched = filter_matched_coverage(
            raw_trades,
            ready_masks[symbol],
            signal_start,
            signal_end,
            spec.params.entry_close_confirmation,
        )
        replayed[symbol] = scale_trades(matched, spec.risk_weight)
    return replayed


def replay_ensemble(
    frames: Mapping[str, pd.DataFrame],
    funding_frame: pd.DataFrame,
    ready_masks: Mapping[str, pd.Series],
    specs: Sequence[EngineSpec],
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> dict[str, dict[str, list[CandidateTrade]]]:
    """세 엔진을 서로의 포지션 상태와 무관하게 독립 재생한다."""

    return {
        spec.name: replay_scaled_engine(
            frames,
            funding_frame,
            ready_masks,
            spec,
            signal_start,
            signal_end,
        )
        for spec in specs
    }


def flatten_symbol_trades(
    trades_by_symbol: Mapping[str, Sequence[CandidateTrade]],
) -> list[CandidateTrade]:
    """심볼별 거래를 청산시각 기준의 단일 목록으로 합친다."""

    trades = [
        trade
        for symbol in SYMBOLS
        for trade in trades_by_symbol.get(symbol, ())
    ]
    return sorted(
        trades,
        key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time),
    )


def flatten_ensemble_trades(
    trades_by_engine: Mapping[str, Mapping[str, Sequence[CandidateTrade]]],
) -> list[CandidateTrade]:
    """엔진·심볼별 가중 거래를 하나의 실현손익 목록으로 합친다."""

    trades = [
        trade
        for engine in sorted(trades_by_engine)
        for symbol in SYMBOLS
        for trade in trades_by_engine[engine].get(symbol, ())
    ]
    return sorted(
        trades,
        key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time),
    )


def unique_entry_key(trade: CandidateTrade) -> tuple[str, pd.Timestamp]:
    """엔진 중복을 제거하기 위한 심볼·UTC 진입시각 키를 반환한다."""

    return trade.symbol, pd.Timestamp(trade.entry_time)


def unique_entry_groups(
    trades: Sequence[CandidateTrade],
) -> dict[tuple[str, pd.Timestamp], list[CandidateTrade]]:
    """같은 심볼·진입시각의 엔진 구성 거래를 한 이벤트로 묶는다."""

    grouped: dict[tuple[str, pd.Timestamp], list[CandidateTrade]] = defaultdict(list)
    for trade in trades:
        grouped[unique_entry_key(trade)].append(trade)
    return dict(grouped)


def realized_max_drawdown_r(trades: Sequence[CandidateTrade]) -> float:
    """같은 청산시각 손익을 먼저 합산한 포트폴리오 실현 MDD를 계산한다."""

    if not trades:
        return 0.0
    realized: dict[pd.Timestamp, float] = defaultdict(float)
    for trade in trades:
        realized[pd.Timestamp(trade.exit_time)] += float(trade.net_r)
    values = np.asarray([realized[time] for time in sorted(realized)], dtype=float)
    equity = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = np.maximum.accumulate(equity) - equity
    return float(drawdown.max())


def portfolio_summary(trades: Sequence[CandidateTrade]) -> dict[str, Any]:
    """가중 거래의 PF·순R·실현 MDD와 위험정규화 기대값을 요약한다."""

    trade_list = list(trades)
    if not trade_list:
        return {
            "component_trades": 0,
            "unique_entries": 0,
            "risk_committed_r": 0.0,
            "risk_normalized_expectancy_r": None,
            "profit_factor": None,
            "net_r": 0.0,
            "realized_max_drawdown_r": 0.0,
        }
    values = np.asarray([trade.net_r for trade in trade_list], dtype=float)
    committed = float(sum(trade.risk_committed_r for trade in trade_list))
    if not math.isfinite(committed) or committed <= 0.0:
        raise ValueError("합산 투입 위험은 유한한 양수여야 합니다.")
    gains = float(values[values > 0.0].sum())
    losses = float(-values[values < 0.0].sum())
    gross = float(sum(trade.gross_r for trade in trade_list))
    execution = float(sum(trade.execution_cost_r for trade in trade_list))
    funding_debit = float(sum(max(trade.funding_cost_r, 0.0) for trade in trade_list))
    funding_credit = float(sum(max(-trade.funding_cost_r, 0.0) for trade in trade_list))
    net = float(values.sum())
    return {
        "component_trades": len(trade_list),
        "unique_entries": len(unique_entry_groups(trade_list)),
        "winning_components": int(np.sum(values > 0.0)),
        "component_win_rate": round(float(np.mean(values > 0.0)), 6),
        "risk_committed_r": round(committed, 6),
        "risk_normalized_expectancy_r": round(net / committed, 6),
        "profit_factor": round(gains / losses, 6) if losses > 0.0 else None,
        "gross_r": round(gross, 6),
        "execution_cost_r": round(execution, 6),
        "funding_debit_r": round(funding_debit, 6),
        "funding_credit_r": round(funding_credit, 6),
        "net_funding_cost_r": round(funding_debit - funding_credit, 6),
        "net_r": round(net, 6),
        "realized_max_drawdown_r": round(realized_max_drawdown_r(trade_list), 6),
        "median_holding_hours": round(
            float(np.median([trade.holding_hours for trade in trade_list])),
            2,
        ),
    }


def complete_coverage_year(
    year: int,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> bool:
    """주어진 UTC 연도가 평가 coverage에 완전히 포함되는지 판정한다."""

    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    next_year = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    return (
        coverage_start <= year_start
        and coverage_end >= next_year - pd.Timedelta(hours=1)
    )


def dimension_summary(
    trades: Sequence[CandidateTrade],
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> dict[str, Any]:
    """가중 거래를 전체·심볼·진입연도 기준으로 위험정규화 요약한다."""

    trade_list = list(trades)
    symbols = {
        symbol: portfolio_summary(
            [trade for trade in trade_list if trade.symbol == symbol]
        )
        for symbol in SYMBOLS
    }
    years: dict[str, Any] = {}
    for year in range(coverage_start.year, coverage_end.year + 1):
        yearly = [
            trade
            for trade in trade_list
            if pd.Timestamp(trade.entry_time).year == year
        ]
        years[str(year)] = {
            "complete_coverage_year": complete_coverage_year(
                year,
                coverage_start,
                coverage_end,
            ),
            **portfolio_summary(yearly),
        }
    return {
        "aggregate": portfolio_summary(trade_list),
        "symbols": symbols,
        "years": years,
    }


def unique_complete_month_frequency(
    trades: Sequence[CandidateTrade],
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> dict[str, Any]:
    """0건인 완전월을 포함해 고유 심볼·진입시각 월 빈도를 계산한다."""

    groups = unique_entry_groups(trades)
    timestamps = [entry_time for _, entry_time in groups]
    base = complete_month_counts(timestamps, coverage_start, coverage_end)
    counts = np.asarray(list(base["counts"].values()), dtype=float)
    return {
        **base,
        "definition": (
            "공통 coverage 완전 UTC 달의 unique (symbol, entry_time) 수; "
            "0건인 달 포함"
        ),
        "unique_entries": len(groups),
        "zero_months": int(np.sum(counts == 0.0)) if len(counts) else 0,
        "p10_per_month": (
            round(float(np.quantile(counts, 0.10)), 6) if len(counts) else None
        ),
    }


def risk_equivalent_complete_month_frequency(
    trades: Sequence[CandidateTrade],
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> dict[str, Any]:
    """고유 진입별 투입위험을 합쳐 1R 진입 등가 월 빈도를 계산한다."""

    groups = unique_entry_groups(trades)
    event_risk = {
        key: float(sum(trade.risk_committed_r for trade in components))
        for key, components in groups.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in event_risk.values()):
        raise ValueError("고유 진입 이벤트의 투입위험은 유한한 양수여야 합니다.")

    counts: dict[str, float] = {}
    first_month = pd.Timestamp(
        year=coverage_start.year,
        month=coverage_start.month,
        day=1,
        tz="UTC",
    )
    last_month = pd.Timestamp(
        year=coverage_end.year,
        month=coverage_end.month,
        day=1,
        tz="UTC",
    )
    for month_start in pd.date_range(first_month, last_month, freq="MS"):
        next_month = month_start + pd.DateOffset(months=1)
        if not (
            coverage_start <= month_start
            and coverage_end >= next_month - pd.Timedelta(hours=1)
        ):
            continue
        counts[month_start.strftime("%Y-%m")] = float(
            sum(
                risk
                for (_, entry_time), risk in event_risk.items()
                if month_start <= entry_time < next_month
            )
        )

    monthly = np.asarray(list(counts.values()), dtype=float)
    exact_distribution: dict[str, int] = defaultdict(int)
    for value in event_risk.values():
        exact_distribution[f"{value:.8f}".rstrip("0").rstrip(".")] += 1
    return {
        "definition": (
            "완전 UTC 달의 unique (symbol, entry_time)별 합산 투입위험; "
            "1.0R 진입 한 건을 1 위험등가 진입으로 계산하고 0건 달 포함"
        ),
        "complete_months": len(counts),
        "risk_equivalent_entries": round(float(sum(event_risk.values())), 6),
        "unique_entries": len(event_risk),
        "mean_event_risk_r": (
            round(float(np.mean(list(event_risk.values()))), 6)
            if event_risk
            else None
        ),
        "median_event_risk_r": (
            round(float(np.median(list(event_risk.values()))), 6)
            if event_risk
            else None
        ),
        "events_at_least_0_10r": sum(
            value + 1e-12 >= 0.10 for value in event_risk.values()
        ),
        "events_at_least_0_25r": sum(
            value + 1e-12 >= 0.25 for value in event_risk.values()
        ),
        "event_risk_distribution_r": dict(sorted(exact_distribution.items())),
        "median_per_month": (
            round(float(np.median(monthly)), 6) if len(monthly) else None
        ),
        "mean_per_month": (
            round(float(monthly.mean()), 6) if len(monthly) else None
        ),
        "p10_per_month": (
            round(float(np.quantile(monthly, 0.10)), 6) if len(monthly) else None
        ),
        "counts": {key: round(value, 6) for key, value in counts.items()},
    }


def six_hour_unique_entry_clusters(
    trades: Sequence[CandidateTrade],
) -> dict[str, Any]:
    """고유 진입 이벤트를 6시간 single-linkage 군집으로 합산한다."""

    grouped = unique_entry_groups(trades)
    ordered = sorted(grouped, key=lambda key: (key[1], key[0]))
    if not ordered:
        return {"unique_entries": 0, "clusters": 0}
    clusters: list[list[tuple[str, pd.Timestamp]]] = []
    current: list[tuple[str, pd.Timestamp]] = []
    previous_time: pd.Timestamp | None = None
    window = pd.Timedelta(hours=CLUSTER_HOURS)
    for key in ordered:
        entry_time = key[1]
        if previous_time is None or entry_time - previous_time <= window:
            current.append(key)
        else:
            clusters.append(current)
            current = [key]
        previous_time = entry_time
    clusters.append(current)

    cluster_net = np.asarray(
        [
            sum(trade.net_r for key in cluster for trade in grouped[key])
            for cluster in clusters
        ],
        dtype=float,
    )
    cluster_risk = np.asarray(
        [
            sum(trade.risk_committed_r for key in cluster for trade in grouped[key])
            for cluster in clusters
        ],
        dtype=float,
    )
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=float)
    gains = float(cluster_net[cluster_net > 0.0].sum())
    losses = float(-cluster_net[cluster_net < 0.0].sum())
    equity = np.concatenate([[0.0], np.cumsum(cluster_net)])
    drawdown = np.maximum.accumulate(equity) - equity
    total_risk = float(cluster_risk.sum())
    return {
        "unique_entries": len(ordered),
        "clusters": len(clusters),
        "method": "single_linkage_between_consecutive_unique_entries_gap_at_most_6h",
        "independence_claim": False,
        "median_unique_entries_per_cluster": round(float(np.median(sizes)), 6),
        "max_unique_entries_per_cluster": int(sizes.max()),
        "cluster_win_rate": round(float(np.mean(cluster_net > 0.0)), 6),
        "cluster_profit_factor": round(gains / losses, 6) if losses > 0.0 else None,
        "cluster_net_r": round(float(cluster_net.sum()), 6),
        "cluster_risk_committed_r": round(total_risk, 6),
        "cluster_risk_normalized_expectancy_r": (
            round(float(cluster_net.sum()) / total_risk, 6)
            if total_risk > 0.0
            else None
        ),
        "cluster_realized_max_drawdown_r": round(float(drawdown.max()), 6),
        "note": "근접 진입 중복을 줄이는 유효표본 proxy이며 통계적 독립을 증명하지 않음",
    }


def max_weighted_concurrent_heat(
    trades: Sequence[CandidateTrade],
    risk_percent: float = BASE_RISK_PERCENT,
) -> dict[str, Any]:
    """가중 투입위험의 보수적 최대 동시 heat를 진입 우선 sweep으로 계산한다."""

    if risk_percent <= 0.0:
        raise ValueError("R당 위험 비율은 0보다 커야 합니다.")
    entries: dict[pd.Timestamp, list[CandidateTrade]] = defaultdict(list)
    exits: dict[pd.Timestamp, list[CandidateTrade]] = defaultdict(list)
    for trade in trades:
        entries[pd.Timestamp(trade.entry_time)].append(trade)
        exits[pd.Timestamp(trade.exit_time)].append(trade)
    times = sorted(set(entries).union(exits))
    active_heat = 0.0
    active_components = 0
    maximum_heat = 0.0
    maximum_components = 0
    maximum_time: pd.Timestamp | None = None
    for timestamp in times:
        entering = entries.get(timestamp, [])
        active_heat += sum(trade.risk_committed_r for trade in entering)
        active_components += len(entering)
        if active_heat > maximum_heat:
            maximum_heat = active_heat
            maximum_components = active_components
            maximum_time = timestamp
        exiting = exits.get(timestamp, [])
        active_heat -= sum(trade.risk_committed_r for trade in exiting)
        active_components -= len(exiting)
        if abs(active_heat) < 1e-10:
            active_heat = 0.0
        if active_heat < -1e-8 or active_components < 0:
            raise ValueError("동시 heat sweep의 활성 포지션 상태가 음수가 됐습니다.")
    return {
        "max_weighted_concurrent_heat_r": round(maximum_heat, 6),
        "max_concurrent_engine_components": maximum_components,
        "peak_time": maximum_time.isoformat() if maximum_time is not None else None,
        "risk_percent_per_r": risk_percent,
        "risk_scaled_max_heat_percent": round(maximum_heat * risk_percent, 6),
        "tie_policy": "같은 시각에는 신규 진입을 먼저 더하고 청산을 나중에 빼는 보수적 상한",
        "mtm_margin_liquidation_modeled": False,
    }


def _daily_trade_values(
    trades: Sequence[CandidateTrade],
    calendar: pd.DatetimeIndex,
    timestamp_field: str,
    value_field: str,
) -> np.ndarray:
    """거래 필드를 UTC 일별 배열로 합쳐 paired bootstrap 입력을 만든다."""

    values = pd.Series(0.0, index=calendar, dtype=float)
    for trade in trades:
        timestamp = pd.Timestamp(getattr(trade, timestamp_field)).floor("D")
        if timestamp in values.index:
            values.loc[timestamp] += float(getattr(trade, value_field))
    return values.to_numpy(dtype=float)


def paired_calendar_bootstrap(
    candidate_trades: Sequence[CandidateTrade],
    reference_trades: Sequence[CandidateTrade],
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
    block_days: int,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """같은 달력 블록으로 후보-기준 순R·위험효율·MDD 차이를 재표본한다."""

    if block_days < 1 or samples < 1:
        raise ValueError("paired bootstrap 블록과 표본 수는 1 이상이어야 합니다.")
    calendar = pd.date_range(
        calendar_start.floor("D"),
        calendar_end.floor("D"),
        freq="1D",
        tz="UTC",
    )
    if len(calendar) < block_days * 3:
        return {
            "status": "insufficient",
            "calendar_days": len(calendar),
            "block_days": block_days,
        }

    candidate_net = _daily_trade_values(
        candidate_trades,
        calendar,
        "exit_time",
        "net_r",
    )
    reference_net = _daily_trade_values(
        reference_trades,
        calendar,
        "exit_time",
        "net_r",
    )
    candidate_risk = _daily_trade_values(
        candidate_trades,
        calendar,
        "entry_time",
        "risk_committed_r",
    )
    reference_risk = _daily_trade_values(
        reference_trades,
        calendar,
        "entry_time",
        "risk_committed_r",
    )

    generator = np.random.default_rng(seed + block_days)
    blocks_needed = math.ceil(len(calendar) / block_days)
    offsets = np.arange(block_days, dtype=int)
    net_delta = np.empty(samples, dtype=float)
    expectancy_delta = np.empty(samples, dtype=float)
    mdd_improvement = np.empty(samples, dtype=float)
    batch_size = 128
    for first in range(0, samples, batch_size):
        batch = min(batch_size, samples - first)
        starts = generator.integers(
            0,
            len(calendar),
            size=(batch, blocks_needed),
        )
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(calendar)
        indices = indices.reshape(batch, -1)[:, : len(calendar)]
        candidate_paths = candidate_net[indices]
        reference_paths = reference_net[indices]
        candidate_totals = candidate_paths.sum(axis=1)
        reference_totals = reference_paths.sum(axis=1)
        candidate_risk_totals = candidate_risk[indices].sum(axis=1)
        reference_risk_totals = reference_risk[indices].sum(axis=1)
        if (candidate_risk_totals <= 0.0).any() or (reference_risk_totals <= 0.0).any():
            raise ValueError("paired bootstrap 표본에 0 이하 투입위험이 있습니다.")

        net_delta[first : first + batch] = candidate_totals - reference_totals
        expectancy_delta[first : first + batch] = (
            candidate_totals / candidate_risk_totals
            - reference_totals / reference_risk_totals
        )
        candidate_equity = np.cumsum(candidate_paths, axis=1)
        reference_equity = np.cumsum(reference_paths, axis=1)
        candidate_origin = np.concatenate(
            [np.zeros((batch, 1)), candidate_equity],
            axis=1,
        )
        reference_origin = np.concatenate(
            [np.zeros((batch, 1)), reference_equity],
            axis=1,
        )
        candidate_mdd = (
            np.maximum.accumulate(candidate_origin, axis=1)[:, 1:]
            - candidate_equity
        ).max(axis=1)
        reference_mdd = (
            np.maximum.accumulate(reference_origin, axis=1)[:, 1:]
            - reference_equity
        ).max(axis=1)
        mdd_improvement[first : first + batch] = reference_mdd - candidate_mdd

    def metric(values: np.ndarray) -> dict[str, float]:
        """양수가 후보 우위인 paired 분포를 요약한다."""

        return {
            "p05": round(float(np.quantile(values, 0.05)), 6),
            "p50": round(float(np.quantile(values, 0.50)), 6),
            "probability_positive": round(float(np.mean(values > 0.0)), 6),
        }

    return {
        "status": "ok",
        "method": "paired_realized_daily_pnl_circular_moving_block",
        "samples": samples,
        "calendar_days": len(calendar),
        "block_days": block_days,
        "candidate_minus_reference_net_r": metric(net_delta),
        "candidate_minus_reference_risk_normalized_expectancy_r": metric(
            expectancy_delta
        ),
        "reference_minus_candidate_max_drawdown_r": metric(mdd_improvement),
        "mtm_modeled": False,
    }


def paired_bootstrap_suite(
    candidate_trades: Sequence[CandidateTrade],
    reference_trades: Sequence[CandidateTrade],
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
) -> dict[str, Any]:
    """14·28·56·84일 공통 블록의 후보-기준 paired 검정을 반환한다."""

    return {
        f"{block_days}d": paired_calendar_bootstrap(
            candidate_trades,
            reference_trades,
            calendar_start,
            calendar_end,
            block_days,
        )
        for block_days in BOOTSTRAP_BLOCK_DAYS
    }


def entry_event_record(
    key: tuple[str, pd.Timestamp],
    components: Sequence[CandidateTrade],
) -> dict[str, Any]:
    """고유 진입 이벤트 하나의 가중 성과를 JSON용 사전으로 만든다."""

    return {
        "symbol": key[0],
        "entry_time": key[1].isoformat(),
        "component_trades": len(components),
        "last_exit_time": max(
            pd.Timestamp(trade.exit_time) for trade in components
        ).isoformat(),
        "risk_committed_r": round(
            float(sum(trade.risk_committed_r for trade in components)),
            6,
        ),
        "net_r": round(float(sum(trade.net_r for trade in components)), 6),
    }


def top_winner_analysis(trades: Sequence[CandidateTrade]) -> dict[str, Any]:
    """고유 진입 이벤트 상위 수익과 top 1·3·5 제거 민감도를 반환한다."""

    grouped = unique_entry_groups(trades)
    ranked = sorted(
        grouped,
        key=lambda key: (
            -sum(trade.net_r for trade in grouped[key]),
            key[1],
            key[0],
        ),
    )
    top_events = [
        entry_event_record(key, grouped[key]) for key in ranked[:10]
    ]
    removal: dict[str, Any] = {}
    for count in (1, 3, 5):
        removed_keys = set(ranked[: min(count, len(ranked))])
        retained = [
            trade
            for key, components in grouped.items()
            if key not in removed_keys
            for trade in components
        ]
        removal[f"top_{count}"] = {
            "removed": [
                entry_event_record(key, grouped[key])
                for key in ranked[: min(count, len(ranked))]
            ],
            "retained_summary": portfolio_summary(retained),
        }
    return {
        "ranking_unit": "unique_(symbol_entry_time)_event",
        "top_10": top_events,
        "removal_sensitivity": removal,
    }


def metrics_for_ledger(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Mapping[str, Any]],
    frequency: Mapping[str, Any],
) -> ParetoMetrics:
    """위험정규화 기대값을 쓰는 엄격 요약을 Pareto 원장 지표로 변환한다."""

    required = (
        "profit_factor",
        "risk_normalized_expectancy_r",
        "net_r",
        "realized_max_drawdown_r",
    )
    missing = [name for name in required if summary.get(name) is None]
    if missing or frequency.get("median_per_month") is None:
        raise ValueError(f"Pareto 앙상블 원장 지표가 불완전합니다: {missing}")
    return ParetoMetrics(
        profit_factor=float(summary["profit_factor"]),
        expectancy_r=float(summary["risk_normalized_expectancy_r"]),
        net_r=float(summary["net_r"]),
        max_drawdown_r=float(summary["realized_max_drawdown_r"]),
        bootstrap_mdd_p95_r=conservative_bootstrap_mdd(bootstrap),
        trades_per_month=float(frequency["median_per_month"]),
    )


def runner_code_hash() -> str:
    """앙상블 러너와 재사용 v4·체결·원장 코드의 결합 SHA256을 반환한다."""

    return sha256_files(
        [
            Path(__file__),
            ROOT / "lab" / "validate_pareto_candidate.py",
            ROOT / "lab" / "validate_live_candidate.py",
            ROOT / "lab" / "pareto_trial_ledger.py",
        ],
        root=ROOT,
    )


def ensemble_contract(
    coverage: CandidateParams,
    specs: Sequence[EngineSpec],
) -> dict[str, Any]:
    """원장과 결과 파일에 공통으로 쓰는 고정 앙상블 계약을 만든다."""

    return {
        "coverage_candidate": {
            **asdict(coverage),
            "quantile_interpolation": "linear",
        },
        "engines": [
            {
                "name": spec.name,
                "risk_weight": spec.risk_weight,
                "params": asdict(spec.params),
            }
            for spec in specs
        ],
        "risk_scaling": (
            "CandidateTrade gross_r/execution_cost_r/funding_cost_r/net_r/"
            "risk_committed_r를 엔진 위험예산으로 선형 scale"
        ),
        "allow_additions": False,
    }


def append_completed_trials(
    ledger_path: Path,
    contract: Mapping[str, Any],
    reference: CandidateParams,
    hashes: Mapping[str, str],
    code_hash: str,
    candidate_metrics: ParetoMetrics,
    reference_metrics: ParetoMetrics,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """후보와 기준을 모두 COMPLETED discovery trial로 append-only 기록한다."""

    ledger = ParetoTrialLedger(ledger_path)
    metadata = {
        "status": "FAIL_DISCOVERY_ONLY",
        "promotion_allowed": False,
        "coverage_policy": coverage["policy"],
        "common_signal_start": coverage["common_signal_start"],
        "common_signal_end": coverage["common_signal_end"],
        "pareto_expectancy_definition": "net_r / sum(risk_committed_r)",
        "monthly_frequency_definition": (
            "완전 UTC 달의 unique (symbol, entry_time)별 합산 투입위험을 "
            "1R 진입 등가로 환산한 0건 포함 중앙값"
        ),
        "bootstrap_definition": "14/28/56/84d realized-calendar circular moving blocks",
    }
    reference_result = ledger.append_success(
        trial_name="pareto_ensemble_stop7_24_12_no_filter_full_reference",
        params={
            **asdict(reference),
            "risk_weight": 1.0,
            "coverage_ready_source": "q365_linear_q60_min30_full_window_candidate",
            "allow_additions": False,
        },
        data_hashes=hashes,
        code_hash=code_hash,
        metrics=reference_metrics,
        metadata={**metadata, "role": "matched_reference"},
    )
    candidate_result = ledger.append_success(
        trial_name="pareto_ensemble_core90_scout5_fast5",
        params=dict(contract),
        data_hashes=hashes,
        code_hash=code_hash,
        metrics=candidate_metrics,
        metadata={**metadata, "role": "candidate"},
    )
    return {
        "path": str(ledger_path),
        "candidate": {
            "trial_id": candidate_result.trial_id,
            "appended": candidate_result.appended,
            "outcome": candidate_result.record["outcome"],
        },
        "matched_reference": {
            "trial_id": reference_result.trial_id,
            "appended": reference_result.appended,
            "outcome": reference_result.record["outcome"],
        },
    }


def write_engine_trade_ledger(
    path: Path,
    trades_by_engine: Mapping[str, Mapping[str, Sequence[CandidateTrade]]],
    engine_weights: Mapping[str, float],
) -> None:
    """엔진 식별자와 가중치를 포함한 completed 거래 JSONL을 저장한다."""

    records: list[dict[str, Any]] = []
    for engine in sorted(trades_by_engine):
        for symbol in SYMBOLS:
            for trade in trades_by_engine[engine].get(symbol, ()):
                records.append(
                    {
                        "engine": engine,
                        "engine_risk_weight": engine_weights[engine],
                        **asdict(trade),
                    }
                )
    records.sort(
        key=lambda record: (
            record["exit_time"],
            record["symbol"],
            record["entry_time"],
            record["engine"],
        )
    )
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def bootstrap_condition_rows(
    bootstrap: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """strict 부트스트랩의 p05와 양수확률 필수 조건을 블록별 평가한다."""

    rows: dict[str, Any] = {}
    for block_days in BOOTSTRAP_BLOCK_DAYS:
        name = f"{block_days}d"
        result = bootstrap[name]
        p05 = result.get("net_r_p05")
        probability = result.get("probability_positive")
        passed = (
            result.get("status") == "ok"
            and p05 is not None
            and float(p05) > 0.0
            and probability is not None
            and float(probability) >= REQUIRED_BOOTSTRAP_PROBABILITY_POSITIVE
        )
        rows[name] = {
            "required_net_r_p05_strictly_greater_than": 0.0,
            "required_probability_positive_at_least": (
                REQUIRED_BOOTSTRAP_PROBABILITY_POSITIVE
            ),
            "observed_net_r_p05": p05,
            "observed_probability_positive": probability,
            "pass": passed,
        }
    return rows


def stress_bootstrap_diagnostics(
    bootstrap: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """20bp severe 부트스트랩을 비차단 진단과 경고로 변환한다."""

    rows: dict[str, Any] = {}
    for block_days in BOOTSTRAP_BLOCK_DAYS:
        name = f"{block_days}d"
        result = bootstrap[name]
        p05 = result.get("net_r_p05")
        rows[name] = {
            "observed_net_r_p05": p05,
            "warning": p05 is None or float(p05) <= 0.0,
            "absolute_gate": False,
        }
    return rows


def paired_condition_rows(
    paired: Mapping[str, Mapping[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    """paired 후보우위 분포의 p05와 양수확률을 블록별 게이트로 만든다."""

    rows: dict[str, Any] = {}
    for block_days in BOOTSTRAP_BLOCK_DAYS:
        name = f"{block_days}d"
        result = paired[name]
        metric = result.get(metric_name, {})
        p05 = metric.get("p05")
        probability = metric.get("probability_positive")
        rows[name] = {
            "observed_p05": p05,
            "observed_probability_positive": probability,
            "required_p05_strictly_greater_than": 0.0,
            "required_probability_positive_at_least": (
                REQUIRED_BOOTSTRAP_PROBABILITY_POSITIVE
            ),
            "pass": (
                result.get("status") == "ok"
                and p05 is not None
                and float(p05) > 0.0
                and probability is not None
                and float(probability) >= REQUIRED_BOOTSTRAP_PROBABILITY_POSITIVE
            ),
        }
    return rows


def build_gate(
    strict_candidate: Mapping[str, Any],
    strict_dimensions: Mapping[str, Any],
    stress_candidate: Mapping[str, Any],
    strict_bootstrap: Mapping[str, Mapping[str, Any]],
    stress_bootstrap: Mapping[str, Mapping[str, Any]],
    candidate_frequency: Mapping[str, Any],
    reference_frequency: Mapping[str, Any],
    candidate_risk_frequency: Mapping[str, Any],
    reference_risk_frequency: Mapping[str, Any],
    candidate_clusters: Mapping[str, Any],
    reference_clusters: Mapping[str, Any],
    candidate_heat: Mapping[str, Any],
    pareto: Mapping[str, Any],
    paired_bootstrap: Mapping[str, Mapping[str, Any]],
    top_five_retained: Mapping[str, Any],
) -> dict[str, Any]:
    """수치 게이트를 평가하되 discovery 실거래 승격은 항상 차단한다."""

    strict_pf = strict_candidate.get("profit_factor")
    strict_expectancy = strict_candidate.get("risk_normalized_expectancy_r")
    stress_pf = stress_candidate.get("profit_factor")
    stress_expectancy = stress_candidate.get("risk_normalized_expectancy_r")
    positive_symbols = sum(
        1
        for summary in strict_dimensions["symbols"].values()
        if summary.get("risk_normalized_expectancy_r") is not None
        and float(summary["risk_normalized_expectancy_r"]) > 0.0
    )
    complete_years = [
        summary
        for summary in strict_dimensions["years"].values()
        if summary["complete_coverage_year"]
    ]
    positive_complete_years = sum(
        1
        for summary in complete_years
        if summary.get("risk_normalized_expectancy_r") is not None
        and float(summary["risk_normalized_expectancy_r"]) > 0.0
    )
    bootstrap_rows = bootstrap_condition_rows(strict_bootstrap)
    strict_bootstrap_pass = all(row["pass"] for row in bootstrap_rows.values())

    candidate_cluster_count = int(candidate_clusters.get("clusters", 0))
    reference_cluster_count = int(reference_clusters.get("clusters", 0))
    cluster_required_raw = (
        reference_cluster_count * REQUIRED_CLUSTER_FREQUENCY_MULTIPLE
    )
    cluster_required = int(math.ceil(cluster_required_raw - 1e-12))
    cluster_pass = candidate_cluster_count >= cluster_required

    candidate_median = candidate_frequency.get("median_per_month")
    reference_median = reference_frequency.get("median_per_month")
    candidate_p10 = candidate_frequency.get("p10_per_month")
    reference_p10 = reference_frequency.get("p10_per_month")
    monthly_median_required: float | None = (
        float(reference_median) * REQUIRED_MONTHLY_MEDIAN_MULTIPLE
        if reference_median is not None
        else None
    )
    monthly_median_pass = (
        candidate_median is not None
        and monthly_median_required is not None
        and float(candidate_median) + 1e-12 >= monthly_median_required
    )
    monthly_p10_pass = (
        candidate_p10 is not None
        and reference_p10 is not None
        and float(candidate_p10) >= float(reference_p10)
    )

    candidate_risk_median = candidate_risk_frequency.get("median_per_month")
    reference_risk_median = reference_risk_frequency.get("median_per_month")
    candidate_risk_p10 = candidate_risk_frequency.get("p10_per_month")
    reference_risk_p10 = reference_risk_frequency.get("p10_per_month")
    risk_median_required: float | None = (
        float(reference_risk_median) * REQUIRED_MONTHLY_MEDIAN_MULTIPLE
        if reference_risk_median is not None
        else None
    )
    risk_median_pass = (
        candidate_risk_median is not None
        and risk_median_required is not None
        and float(candidate_risk_median) + 1e-12 >= risk_median_required
    )
    risk_p10_pass = (
        candidate_risk_p10 is not None
        and reference_risk_p10 is not None
        and float(candidate_risk_p10) >= float(reference_risk_p10)
    )
    paired_net_rows = paired_condition_rows(
        paired_bootstrap,
        "candidate_minus_reference_net_r",
    )
    paired_expectancy_rows = paired_condition_rows(
        paired_bootstrap,
        "candidate_minus_reference_risk_normalized_expectancy_r",
    )
    paired_mdd_rows = paired_condition_rows(
        paired_bootstrap,
        "reference_minus_candidate_max_drawdown_r",
    )
    top_five_pf = top_five_retained.get("profit_factor")
    top_five_expectancy = top_five_retained.get(
        "risk_normalized_expectancy_r"
    )

    heat = float(candidate_heat["max_weighted_concurrent_heat_r"])
    conditions = {
        "strict_profit_factor": {
            "required_at_least": REQUIRED_STRICT_PROFIT_FACTOR,
            "observed": strict_pf,
            "pass": strict_pf is not None
            and float(strict_pf) >= REQUIRED_STRICT_PROFIT_FACTOR,
        },
        "strict_risk_normalized_expectancy": {
            "required_strictly_greater_than": 0.0,
            "definition": "net_r / sum(risk_committed_r)",
            "observed": strict_expectancy,
            "pass": strict_expectancy is not None and float(strict_expectancy) > 0.0,
        },
        "strict_positive_symbols": {
            "required": REQUIRED_POSITIVE_SYMBOLS,
            "observed": positive_symbols,
            "pass": positive_symbols >= REQUIRED_POSITIVE_SYMBOLS,
        },
        "strict_positive_complete_years": {
            "required": REQUIRED_POSITIVE_COMPLETE_YEARS,
            "available_complete_years": len(complete_years),
            "observed": positive_complete_years,
            "pass": positive_complete_years >= REQUIRED_POSITIVE_COMPLETE_YEARS,
        },
        "strict_bootstrap_all_blocks": {
            "blocks_days": list(BOOTSTRAP_BLOCK_DAYS),
            "rows": bootstrap_rows,
            "pass": strict_bootstrap_pass,
        },
        "unique_clusters_6h_frequency": {
            "required_candidate_multiple_of_reference": (
                REQUIRED_CLUSTER_FREQUENCY_MULTIPLE
            ),
            "reference_clusters": reference_cluster_count,
            "required_candidate_clusters": cluster_required,
            "unrounded_multiple_target": round(cluster_required_raw, 6),
            "observed_candidate_clusters": candidate_cluster_count,
            "pass": cluster_pass,
        },
        "unique_monthly_median_frequency": {
            "required_candidate_multiple_of_reference": (
                REQUIRED_MONTHLY_MEDIAN_MULTIPLE
            ),
            "reference_median": reference_median,
            "required_candidate_median": monthly_median_required,
            "observed_candidate_median": candidate_median,
            "pass": monthly_median_pass,
        },
        "unique_monthly_p10_no_worse": {
            "required_candidate_at_least_reference": reference_p10,
            "observed_candidate": candidate_p10,
            "pass": monthly_p10_pass,
        },
        "risk_equivalent_monthly_median_frequency": {
            "required_candidate_multiple_of_reference": (
                REQUIRED_MONTHLY_MEDIAN_MULTIPLE
            ),
            "reference_median": reference_risk_median,
            "required_candidate_median": risk_median_required,
            "observed_candidate_median": candidate_risk_median,
            "pass": risk_median_pass,
        },
        "risk_equivalent_monthly_p10_no_worse": {
            "required_candidate_at_least_reference": reference_risk_p10,
            "observed_candidate": candidate_risk_p10,
            "pass": risk_p10_pass,
        },
        "paired_net_r_improvement_all_blocks": {
            "rows": paired_net_rows,
            "pass": all(row["pass"] for row in paired_net_rows.values()),
        },
        "paired_risk_normalized_expectancy_improvement_all_blocks": {
            "rows": paired_expectancy_rows,
            "pass": all(
                row["pass"] for row in paired_expectancy_rows.values()
            ),
        },
        "paired_realized_mdd_improvement_all_blocks": {
            "rows": paired_mdd_rows,
            "pass": all(row["pass"] for row in paired_mdd_rows.values()),
            "warning": "realized exit-time MDD only; MTM is not modeled",
        },
        "top_five_winner_removal_profit_factor": {
            "required_at_least": REQUIRED_STRICT_PROFIT_FACTOR,
            "observed": top_five_pf,
            "pass": (
                top_five_pf is not None
                and float(top_five_pf) >= REQUIRED_STRICT_PROFIT_FACTOR
            ),
        },
        "top_five_winner_removal_expectancy": {
            "required_strictly_greater_than": 0.0,
            "observed": top_five_expectancy,
            "pass": (
                top_five_expectancy is not None
                and float(top_five_expectancy) > 0.0
            ),
        },
        "weighted_concurrent_heat": {
            "required_at_most_r": MAX_WEIGHTED_CONCURRENT_HEAT_R,
            "required_at_most_percent_at_0_25pct_per_r": (
                MAX_WEIGHTED_CONCURRENT_HEAT_R * BASE_RISK_PERCENT
            ),
            "observed_r": heat,
            "observed_percent": candidate_heat["risk_scaled_max_heat_percent"],
            "pass": heat <= MAX_WEIGHTED_CONCURRENT_HEAT_R,
        },
        "stress_20bp_profit_factor": {
            "required_at_least": REQUIRED_STRESS_PROFIT_FACTOR,
            "observed": stress_pf,
            "pass": stress_pf is not None
            and float(stress_pf) >= REQUIRED_STRESS_PROFIT_FACTOR,
        },
        "stress_20bp_risk_normalized_expectancy": {
            "required_strictly_greater_than": 0.0,
            "observed": stress_expectancy,
            "pass": stress_expectancy is not None and float(stress_expectancy) > 0.0,
        },
        "pareto_dominates_matched_reference": {
            "required": True,
            "observed": bool(pareto.get("dominates", False)),
            "pass": bool(pareto.get("dominates", False)),
        },
    }
    statistical_conditions_pass = all(
        bool(condition["pass"]) for condition in conditions.values()
    )
    return {
        "status": "FAIL",
        "promotion_allowed": False,
        "promotion_capability": "DISABLED_IN_DISCOVERY_RUNNER",
        "statistical_conditions_pass": statistical_conditions_pass,
        "conditions": conditions,
        "stress_bootstrap_diagnostics_not_a_gate": stress_bootstrap_diagnostics(
            stress_bootstrap
        ),
        "prospective_gate_not_evaluated": {
            "status": "NOT_EVALUATED_ON_HISTORICAL_DISCOVERY_DATA",
            "required_risk_equivalent_entries_at_least": (
                PROSPECTIVE_MIN_UNIQUE_ENTRIES
            ),
            "required_forward_months_at_least": PROSPECTIVE_MIN_MONTHS,
            "promotion_allowed": False,
        },
        "blocking_reasons": [
            "이미 관측한 discovery 자료이므로 수치 조건과 무관하게 PASS를 발급하지 않음",
            "시간순 mark-to-market 계좌 경로가 없음",
            "교차·격리 마진과 유지증거금이 없음",
            "강제청산 가격·수수료·보험기금 상호작용이 없음",
        ],
    }


def ensemble_coverage_manifest(
    frames: Mapping[str, pd.DataFrame],
    ready_masks: Mapping[str, pd.Series],
    common_signal_start: pd.Timestamp,
    common_signal_end: pd.Timestamp,
) -> dict[str, Any]:
    """v4 coverage manifest에 앙상블 공통 readiness 계약을 명시한다."""

    manifest = v4_coverage_manifest(
        frames,
        ready_masks,
        common_signal_start,
        common_signal_end,
    )
    manifest["policy"] = (
        "stop7 24/12 q365 linear q60 min30 full-window candidate의 "
        "entry_regime_ready 신호봉 마스크와 5심볼 공통 구간을 core/scout/fast/"
        "no-filter reference 모두에 동일 적용; 마지막 신호 뒤 73시간 완전 follow-up "
        "embargo 확보"
    )
    manifest["quantile_interpolation"] = "linear"
    manifest["follow_up_embargo_hours"] = FOLLOW_UP_EMBARGO_HOURS
    return manifest


def run_validation(output_dir: Path, ledger_path: Path) -> dict[str, Any]:
    """고정 Pareto 앙상블과 matched reference를 검증하고 원장을 저장한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_params = coverage_candidate_params()
    specs = ensemble_engine_specs(coverage_params)
    reference = reference_params(coverage_params)
    (
        frames,
        funding_frame,
        ready_masks,
        common_signal_start,
        common_signal_end,
    ) = prepare_inputs(coverage_params)
    common_signal_end = common_signal_end - pd.Timedelta(
        hours=FOLLOW_UP_EMBARGO_HOURS
    )
    if common_signal_end <= common_signal_start:
        raise ValueError("최대 보유기간 embargo 뒤 공통 검증구간이 비었습니다.")

    candidate_by_engine = replay_ensemble(
        frames,
        funding_frame,
        ready_masks,
        specs,
        common_signal_start,
        common_signal_end,
    )
    reference_by_symbol = replay_matched(
        frames,
        funding_frame,
        ready_masks,
        reference,
        common_signal_start,
        common_signal_end,
    )
    candidate_trades = flatten_ensemble_trades(candidate_by_engine)
    reference_trades = flatten_symbol_trades(reference_by_symbol)

    entry_coverage_start = common_signal_start + pd.Timedelta(hours=1)
    entry_coverage_end = common_signal_end + pd.Timedelta(hours=1)
    calendar_end = min(frame.index[-1] for frame in frames.values())

    candidate_strict = dimension_summary(
        candidate_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    reference_strict = dimension_summary(
        reference_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    engine_strict = {
        engine: dimension_summary(
            flatten_symbol_trades(by_symbol),
            entry_coverage_start,
            entry_coverage_end,
        )
        for engine, by_symbol in candidate_by_engine.items()
    }

    stressed_candidate_trades = apply_execution_funding_stress(
        candidate_trades,
        STRICT_COST_BPS_SIDE,
        STRESS_COST_BPS_SIDE,
    )
    stressed_reference_trades = apply_execution_funding_stress(
        reference_trades,
        STRICT_COST_BPS_SIDE,
        STRESS_COST_BPS_SIDE,
    )
    stressed_candidate_by_engine = {
        engine: {
            symbol: apply_execution_funding_stress(
                trades,
                STRICT_COST_BPS_SIDE,
                STRESS_COST_BPS_SIDE,
            )
            for symbol, trades in by_symbol.items()
        }
        for engine, by_symbol in candidate_by_engine.items()
    }
    candidate_stress = dimension_summary(
        stressed_candidate_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    reference_stress = dimension_summary(
        stressed_reference_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    engine_stress = {
        engine: dimension_summary(
            flatten_symbol_trades(by_symbol),
            entry_coverage_start,
            entry_coverage_end,
        )
        for engine, by_symbol in stressed_candidate_by_engine.items()
    }

    candidate_frequency = unique_complete_month_frequency(
        candidate_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    reference_frequency = unique_complete_month_frequency(
        reference_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    candidate_risk_frequency = risk_equivalent_complete_month_frequency(
        candidate_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    reference_risk_frequency = risk_equivalent_complete_month_frequency(
        reference_trades,
        entry_coverage_start,
        entry_coverage_end,
    )
    candidate_clusters = six_hour_unique_entry_clusters(candidate_trades)
    reference_clusters = six_hour_unique_entry_clusters(reference_trades)
    candidate_heat = max_weighted_concurrent_heat(candidate_trades)
    reference_heat = max_weighted_concurrent_heat(reference_trades)

    candidate_bootstrap = bootstrap_suite(
        candidate_trades,
        entry_coverage_start,
        calendar_end,
    )
    reference_bootstrap = bootstrap_suite(
        reference_trades,
        entry_coverage_start,
        calendar_end,
    )
    stressed_candidate_bootstrap = bootstrap_suite(
        stressed_candidate_trades,
        entry_coverage_start,
        calendar_end,
    )
    stressed_reference_bootstrap = bootstrap_suite(
        stressed_reference_trades,
        entry_coverage_start,
        calendar_end,
    )
    paired_bootstrap = paired_bootstrap_suite(
        candidate_trades,
        reference_trades,
        entry_coverage_start,
        calendar_end,
    )
    candidate_top_winners = top_winner_analysis(candidate_trades)
    reference_top_winners = top_winner_analysis(reference_trades)

    candidate_metrics = metrics_for_ledger(
        candidate_strict["aggregate"],
        candidate_bootstrap,
        candidate_risk_frequency,
    )
    reference_metrics = metrics_for_ledger(
        reference_strict["aggregate"],
        reference_bootstrap,
        reference_risk_frequency,
    )
    pareto = compare_pareto(candidate_metrics, reference_metrics)
    pareto_dict = pareto.to_dict()
    coverage = ensemble_coverage_manifest(
        frames,
        ready_masks,
        common_signal_start,
        common_signal_end,
    )
    hashes = data_hash_manifest()
    code_hash = runner_code_hash()
    contract = ensemble_contract(coverage_params, specs)
    ledger_result = append_completed_trials(
        ledger_path,
        contract,
        reference,
        hashes,
        code_hash,
        candidate_metrics,
        reference_metrics,
        coverage,
    )

    gate = build_gate(
        candidate_strict["aggregate"],
        candidate_strict,
        candidate_stress["aggregate"],
        candidate_bootstrap,
        stressed_candidate_bootstrap,
        candidate_frequency,
        reference_frequency,
        candidate_risk_frequency,
        reference_risk_frequency,
        candidate_clusters,
        reference_clusters,
        candidate_heat,
        pareto_dict,
        paired_bootstrap,
        candidate_top_winners["removal_sensitivity"]["top_5"][
            "retained_summary"
        ],
    )
    results: dict[str, Any] = {
        "classification": "DISCOVERY_ONLY_NOT_PREREGISTERED",
        "gate": gate,
        "contracts": {
            "candidate": contract,
            "matched_reference": {
                **asdict(reference),
                "risk_weight": 1.0,
                "coverage_ready_source": (
                    "q365_linear_q60_min30_full_window_candidate"
                ),
                "allow_additions": False,
            },
            "strict_execution": "편도 12bp + Bybit 실제 정산시각 펀딩",
            "stress_execution": "편도 20bp + 펀딩 차변 2배 + 펀딩 대변 0",
            "pareto_expectancy": "net_r / sum(risk_committed_r)",
        },
        "coverage": coverage,
        "data_hashes": hashes,
        "code_hash": code_hash,
        "strict_12bp_actual_funding": {
            "candidate": candidate_strict,
            "candidate_engines": engine_strict,
            "matched_reference": reference_strict,
        },
        "stress_20bp_funding_debit_x2_credit_zero": {
            "candidate": candidate_stress,
            "candidate_engines": engine_stress,
            "matched_reference": reference_stress,
        },
        "frequency_unique_symbol_entry_time_zero_months": {
            "candidate": candidate_frequency,
            "matched_reference": reference_frequency,
        },
        "frequency_risk_equivalent_entries_zero_months": {
            "candidate": candidate_risk_frequency,
            "matched_reference": reference_risk_frequency,
            "warning": (
                "raw unique signal frequency can be inflated by tiny probe risk; "
                "Pareto and capital-deployment frequency use this risk-equivalent series"
            ),
        },
        "entry_clusters_6h_unique_single_linkage": {
            "candidate": candidate_clusters,
            "matched_reference": reference_clusters,
        },
        "weighted_concurrent_heat": {
            "candidate": candidate_heat,
            "matched_reference": reference_heat,
        },
        "realized_calendar_circular_bootstrap": {
            "strict_12bp_actual_funding": {
                "candidate": candidate_bootstrap,
                "matched_reference": reference_bootstrap,
            },
            "stress_20bp_funding_debit_x2_credit_zero": {
                "candidate": stressed_candidate_bootstrap,
                "matched_reference": stressed_reference_bootstrap,
                "gate_role": "diagnostic_warning_only_except_aggregate_pf_and_expectancy",
            },
            "paired_candidate_vs_reference": paired_bootstrap,
        },
        "top_winner_analysis": {
            "candidate": candidate_top_winners,
            "matched_reference": reference_top_winners,
        },
        "matched_reference_delta": {
            "sign_convention": (
                "candidate_minus_reference; realized drawdown 양수는 후보가 더 위험"
            ),
            "strict_summary": numeric_summary_delta(
                candidate_strict["aggregate"],
                reference_strict["aggregate"],
            ),
            "stress_summary": numeric_summary_delta(
                candidate_stress["aggregate"],
                reference_stress["aggregate"],
            ),
            "monthly_median_unique_entries": round(
                float(candidate_frequency["median_per_month"])
                - float(reference_frequency["median_per_month"]),
                6,
            ),
            "monthly_p10_unique_entries": round(
                float(candidate_frequency["p10_per_month"])
                - float(reference_frequency["p10_per_month"]),
                6,
            ),
            "monthly_median_risk_equivalent_entries": round(
                float(candidate_risk_frequency["median_per_month"])
                - float(reference_risk_frequency["median_per_month"]),
                6,
            ),
            "monthly_p10_risk_equivalent_entries": round(
                float(candidate_risk_frequency["p10_per_month"])
                - float(reference_risk_frequency["p10_per_month"]),
                6,
            ),
            "unique_clusters_6h": (
                int(candidate_clusters["clusters"])
                - int(reference_clusters["clusters"])
            ),
            "bootstrap_mdd_p95_r": round(
                candidate_metrics.bootstrap_mdd_p95_r
                - reference_metrics.bootstrap_mdd_p95_r,
                6,
            ),
            "pareto": pareto_dict,
            "pareto_metrics": {
                "candidate": candidate_metrics.to_dict(),
                "matched_reference": reference_metrics.to_dict(),
                "expectancy_field_definition": "net_r / sum(risk_committed_r)",
                "trades_per_month_field_definition": (
                    "monthly median risk-equivalent entries; raw alert count is diagnostic only"
                ),
            },
        },
        "ledger": ledger_result,
        "limitations": {
            "mtm_modeled": False,
            "margin_modeled": False,
            "liquidation_modeled": False,
            "explicit": [
                "보유 중 mark-to-market 미실현 계좌 경로와 intratrade drawdown 없음",
                "교차·격리 마진, 유지증거금, 레버리지와 증거금 부족 없음",
                "강제청산 가격과 청산 수수료 없음",
                "1시간 OHLCV 봉내 경로·부분체결·슬리피지 꼬리 없음",
                "거래소 장애·지연·주문 거절·호가 잔량 없음",
            ],
        },
    }

    result_path = output_dir / "latest_results.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    engine_weights = {spec.name: spec.risk_weight for spec in specs}
    write_engine_trade_ledger(
        output_dir / "candidate_trades.jsonl",
        candidate_by_engine,
        engine_weights,
    )
    write_engine_trade_ledger(
        output_dir / "reference_trades.jsonl",
        {"matched_reference": reference_by_symbol},
        {"matched_reference": 1.0},
    )
    logger.info("Pareto 앙상블 discovery 결과 저장: %s", result_path)
    return results


def main() -> None:
    """CLI 인수를 읽어 고정 Pareto 앙상블 discovery 검증을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = run_validation(args.output_dir, args.ledger)
    logger.info("gate=%s", result["gate"])
    logger.info("pareto=%s", result["matched_reference_delta"]["pareto"])


if __name__ == "__main__":
    main()
