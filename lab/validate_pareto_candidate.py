from __future__ import annotations

"""고정 Pareto 후보와 동일 coverage 기준을 보수적으로 비교한다.

이 러너는 long-only BRK24의 추매를 제거하고 ATR 최후손절을 7배로 고정한 뒤,
과거 365일 진입 후보 변동성의 60분위 이하에서만 진입하는 단일 후보를 평가한다.
후보와 기준 모두 편도 12bp와 Bybit 실제 펀딩을 적용하며, 기준에는 후보의 변동성
분위수를 계산할 수 있는 coverage-ready 시점만 동일하게 적용한다. 결과는 이미 관측한
자료의 discovery 결과이므로 코드상 실거래 PASS나 승격을 발급하지 않는다.
"""

import argparse
import json
import logging
import math
from dataclasses import asdict, replace
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
    CORE_PATH,
    PAIR_FUNDING_PATH,
    PAIR_PATH,
    ROOT,
    SOL_PATH,
    CandidateParams,
    CandidateTrade,
    add_features,
    load_funding,
    load_market_data,
    sha256_file,
    simulate_symbol,
    summarize,
    validate_funding_coverage,
    validate_market_frame,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "logs" / "validation" / "pareto_candidate_v4"
LEDGER_PATH = ROOT / "logs" / "validation" / "pareto_trials.jsonl"
SYMBOLS = ("BTC", "ETH", "SOL", "XRP", "DOGE")
BOOTSTRAP_BLOCK_DAYS = (14, 28, 56, 84)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_831
CLUSTER_HOURS = 6


def fixed_candidate_params() -> CandidateParams:
    """탐색 없이 고정한 변동성 q60 후보 파라미터를 반환한다."""

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
        cost_bps_side=12.0,
        allow_short=False,
        discovery_only=True,
    )


def fixed_reference_params(candidate: CandidateParams) -> CandidateParams:
    """손절·비용은 같고 변동성 진입 필터만 제거한 기준 파라미터를 반환한다."""

    return replace(
        candidate,
        volatility_filter_days=0,
        volatility_filter_require_full_window=False,
    )


def prepare_inputs(
    candidate: CandidateParams,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.Series],
    pd.Timestamp,
    pd.Timestamp,
]:
    """5개 심볼을 검증하고 후보 coverage-ready 마스크와 공통 구간을 만든다."""

    raw_frames = load_market_data(include_external=True)
    funding_frame = load_funding()
    frames: dict[str, pd.DataFrame] = {}
    ready_masks: dict[str, pd.Series] = {}
    first_ready: list[pd.Timestamp] = []
    last_ready: list[pd.Timestamp] = []

    for symbol in SYMBOLS:
        raw_frame = raw_frames[symbol]
        validate_market_frame(raw_frame, symbol)
        frame = validate_funding_coverage(
            raw_frame,
            funding_frame[symbol],
            symbol,
        )
        features = add_features(frame, candidate)
        ready = features["entry_regime_ready"].fillna(False).astype(bool)
        ready_index = ready.index[ready]
        if ready_index.empty:
            raise ValueError(f"{symbol}: 변동성 필터 coverage-ready 표본이 없습니다.")
        frames[symbol] = frame
        ready_masks[symbol] = ready
        first_ready.append(pd.Timestamp(ready_index[0]))
        last_ready.append(pd.Timestamp(ready_index[-1]))

    common_signal_start = max(first_ready)
    common_signal_end = min(last_ready)
    if common_signal_start >= common_signal_end:
        raise ValueError("5개 심볼의 공통 coverage-ready 구간이 없습니다.")
    return (
        frames,
        funding_frame,
        ready_masks,
        common_signal_start,
        common_signal_end,
    )


def filter_matched_coverage(
    trades: Sequence[CandidateTrade],
    ready: pd.Series,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
    close_confirmation: bool,
) -> list[CandidateTrade]:
    """진입 신호봉이 후보의 공통 coverage-ready 마스크에 속한 거래만 남긴다."""

    lag = pd.Timedelta(hours=1) if close_confirmation else pd.Timedelta(0)
    matched: list[CandidateTrade] = []
    for trade in trades:
        signal_time = pd.Timestamp(trade.entry_time) - lag
        if signal_time < signal_start or signal_time > signal_end:
            continue
        if signal_time not in ready.index or not bool(ready.loc[signal_time]):
            continue
        matched.append(trade)
    return matched


def replay_matched(
    frames: Mapping[str, pd.DataFrame],
    funding_frame: pd.DataFrame,
    ready_masks: Mapping[str, pd.Series],
    params: CandidateParams,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> dict[str, list[CandidateTrade]]:
    """기존 체결 엔진을 재생하고 동일 coverage-ready 구간으로 결과를 제한한다."""

    replayed: dict[str, list[CandidateTrade]] = {}
    for symbol in SYMBOLS:
        logger.info(
            "Pareto 재생: %s rows=%d stop=%.1f vol_days=%d",
            symbol,
            len(frames[symbol]),
            params.stop_atr,
            params.volatility_filter_days,
        )
        raw_trades = simulate_symbol(
            frames[symbol],
            symbol,
            params,
            funding_frame[symbol].dropna(),
            allow_additions=False,
        )
        replayed[symbol] = filter_matched_coverage(
            raw_trades,
            ready_masks[symbol],
            signal_start,
            signal_end,
            params.entry_close_confirmation,
        )
    return replayed


def flatten_trades(
    trades_by_symbol: Mapping[str, Sequence[CandidateTrade]],
) -> list[CandidateTrade]:
    """심볼별 거래를 청산시각 기준의 단일 목록으로 합친다."""

    trades = [trade for symbol in SYMBOLS for trade in trades_by_symbol[symbol]]
    return sorted(trades, key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time))


def cost_breakdown(trades: Sequence[CandidateTrade]) -> dict[str, float]:
    """실행비용과 실제 펀딩 차변·대변을 분리해 합산한다."""

    execution = sum(trade.execution_cost_r for trade in trades)
    funding_debit = sum(max(trade.funding_cost_r, 0.0) for trade in trades)
    funding_credit = sum(max(-trade.funding_cost_r, 0.0) for trade in trades)
    return {
        "execution_cost_r": round(float(execution), 6),
        "funding_debit_r": round(float(funding_debit), 6),
        "funding_credit_r": round(float(funding_credit), 6),
        "net_funding_cost_r": round(float(funding_debit - funding_credit), 6),
    }


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """두 UTC 시각이 걸친 모든 달의 월초 시각을 반환한다."""

    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    final = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    starts: list[pd.Timestamp] = []
    while current <= final:
        starts.append(current)
        current = current + pd.DateOffset(months=1)
    return starts


def is_complete_month(
    month_start: pd.Timestamp,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> bool:
    """월 전체가 평가 coverage 안에 들어오는지 판정한다."""

    next_month = month_start + pd.DateOffset(months=1)
    return coverage_start <= month_start and coverage_end >= next_month - pd.Timedelta(hours=1)


def dimension_summary(
    trades: Sequence[CandidateTrade],
    params: CandidateParams,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> dict[str, Any]:
    """엄격 거래를 aggregate·symbol·진입연도·진입월 단위로 요약한다."""

    trade_list = list(trades)
    by_symbol = {
        symbol: summarize([trade for trade in trade_list if trade.symbol == symbol], params)
        for symbol in SYMBOLS
    }
    by_year = {
        str(year): summarize(
            [
                trade
                for trade in trade_list
                if pd.Timestamp(trade.entry_time).year == year
            ],
            params,
        )
        for year in range(coverage_start.year, coverage_end.year + 1)
    }
    by_month: dict[str, Any] = {}
    for month_start in month_starts(coverage_start, coverage_end):
        next_month = month_start + pd.DateOffset(months=1)
        monthly = [
            trade
            for trade in trade_list
            if month_start <= pd.Timestamp(trade.entry_time) < next_month
        ]
        by_month[month_start.strftime("%Y-%m")] = {
            "complete_coverage_month": is_complete_month(
                month_start,
                coverage_start,
                coverage_end,
            ),
            **summarize(monthly, params),
        }
    return {
        "aggregate": summarize(trade_list, params),
        "costs": cost_breakdown(trade_list),
        "symbols": by_symbol,
        "years": by_year,
        "months": by_month,
    }


def complete_month_counts(
    timestamps: Sequence[pd.Timestamp],
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> dict[str, Any]:
    """0건인 달을 포함한 완전 coverage 월의 빈도와 중앙값을 계산한다."""

    normalized = [pd.Timestamp(timestamp) for timestamp in timestamps]
    counts: dict[str, int] = {}
    for month_start in month_starts(coverage_start, coverage_end):
        if not is_complete_month(month_start, coverage_start, coverage_end):
            continue
        next_month = month_start + pd.DateOffset(months=1)
        counts[month_start.strftime("%Y-%m")] = sum(
            month_start <= timestamp < next_month for timestamp in normalized
        )
    values = np.asarray(list(counts.values()), dtype=float)
    return {
        "definition": "공통 coverage에 완전히 포함된 UTC 달; 0건인 달 포함",
        "complete_months": len(counts),
        "median_per_month": round(float(np.median(values)), 6) if len(values) else None,
        "mean_per_month": round(float(values.mean()), 6) if len(values) else None,
        "counts": counts,
    }


def apply_execution_funding_stress(
    trades: Sequence[CandidateTrade],
    original_cost_bps_side: float,
    stressed_cost_bps_side: float = 20.0,
) -> list[CandidateTrade]:
    """실행비용 20bp와 펀딩 차변 2배·대변 0의 비대칭 스트레스를 적용한다."""

    if original_cost_bps_side <= 0.0 or stressed_cost_bps_side <= 0.0:
        raise ValueError("실행비용 bp는 0보다 커야 합니다.")
    stressed: list[CandidateTrade] = []
    scale = stressed_cost_bps_side / original_cost_bps_side
    for trade in trades:
        execution_cost = trade.execution_cost_r * scale
        funding_cost = max(trade.funding_cost_r, 0.0) * 2.0
        net_r = trade.gross_r - execution_cost - funding_cost
        stressed.append(
            replace(
                trade,
                execution_cost_r=round(float(execution_cost), 8),
                funding_cost_r=round(float(funding_cost), 8),
                net_r=round(float(net_r), 8),
            )
        )
    return stressed


def realized_calendar_bootstrap(
    trades: Sequence[CandidateTrade],
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
    block_days: int,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """실현 청산손익의 UTC 일별 합계를 순환 달력 블록으로 재표본한다."""

    if block_days < 1 or samples < 1:
        raise ValueError("bootstrap 블록과 표본 수는 1 이상이어야 합니다.")
    full_index = pd.date_range(
        calendar_start.floor("D"),
        calendar_end.floor("D"),
        freq="1D",
        tz="UTC",
    )
    if full_index.empty:
        return {"status": "insufficient", "calendar_days": 0}
    realized = pd.Series(0.0, index=full_index, dtype=float)
    for trade in trades:
        exit_day = pd.Timestamp(trade.exit_time).floor("D")
        if exit_day in realized.index:
            realized.loc[exit_day] += trade.net_r
    values = realized.to_numpy(dtype=float)
    if len(values) < block_days * 3:
        return {
            "status": "insufficient",
            "calendar_days": len(values),
            "block_days": block_days,
        }

    generator = np.random.default_rng(seed + block_days)
    blocks_needed = math.ceil(len(values) / block_days)
    offsets = np.arange(block_days, dtype=int)
    totals = np.empty(samples, dtype=float)
    max_drawdowns = np.empty(samples, dtype=float)
    batch_size = 128
    for first in range(0, samples, batch_size):
        batch = min(batch_size, samples - first)
        starts = generator.integers(
            0,
            len(values),
            size=(batch, blocks_needed),
        )
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(values)
        paths = values[indices].reshape(batch, -1)[:, : len(values)]
        totals[first : first + batch] = paths.sum(axis=1)
        equity = np.cumsum(paths, axis=1)
        equity_with_origin = np.concatenate(
            [np.zeros((batch, 1), dtype=float), equity],
            axis=1,
        )
        peaks = np.maximum.accumulate(equity_with_origin, axis=1)
        drawdowns = peaks[:, 1:] - equity
        max_drawdowns[first : first + batch] = drawdowns.max(axis=1)

    return {
        "status": "ok",
        "method": "realized_daily_pnl_circular_moving_block",
        "samples": samples,
        "calendar_days": len(values),
        "block_days": block_days,
        "net_r_p05": round(float(np.quantile(totals, 0.05)), 6),
        "net_r_p50": round(float(np.quantile(totals, 0.50)), 6),
        "probability_positive": round(float(np.mean(totals > 0.0)), 6),
        "max_drawdown_r_p95": round(float(np.quantile(max_drawdowns, 0.95)), 6),
    }


def bootstrap_suite(
    trades: Sequence[CandidateTrade],
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
) -> dict[str, Any]:
    """14·28·56·84일 달력 블록 bootstrap 결과를 한 번에 반환한다."""

    return {
        f"{block_days}d": realized_calendar_bootstrap(
            trades,
            calendar_start,
            calendar_end,
            block_days,
        )
        for block_days in BOOTSTRAP_BLOCK_DAYS
    }


def six_hour_entry_clusters(trades: Sequence[CandidateTrade]) -> dict[str, Any]:
    """서로 6시간을 초과해 떨어진 진입군을 독립표본 대용 cluster로 요약한다."""

    ordered = sorted(trades, key=lambda trade: (trade.entry_time, trade.symbol))
    if not ordered:
        return {"trades": 0, "clusters": 0}
    clusters: list[list[CandidateTrade]] = []
    current: list[CandidateTrade] = []
    previous_time: pd.Timestamp | None = None
    window = pd.Timedelta(hours=CLUSTER_HOURS)
    for trade in ordered:
        entry_time = pd.Timestamp(trade.entry_time)
        if previous_time is None or entry_time - previous_time <= window:
            current.append(trade)
        else:
            clusters.append(current)
            current = [trade]
        previous_time = entry_time
    clusters.append(current)

    values = np.asarray(
        [sum(trade.net_r for trade in cluster) for cluster in clusters],
        dtype=float,
    )
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=float)
    gains = values[values > 0.0].sum()
    losses = -values[values < 0.0].sum()
    equity = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": len(ordered),
        "clusters": len(clusters),
        "method": "single_linkage_entry_gap_at_most_6h",
        "independence_claim": False,
        "note": "동시·근접 진입 중복을 줄인 유효표본 proxy이며 통계적 독립을 증명하지 않음",
        "median_trades_per_cluster": round(float(np.median(sizes)), 6),
        "max_trades_per_cluster": int(sizes.max()),
        "cluster_win_rate": round(float(np.mean(values > 0.0)), 6),
        "cluster_expectancy_r": round(float(values.mean()), 6),
        "cluster_net_r": round(float(values.sum()), 6),
        "cluster_profit_factor": round(float(gains / losses), 6) if losses > 0.0 else None,
        "cluster_max_drawdown_r": round(float(drawdown.max()), 6),
    }


def top_winner_removal(
    trades: Sequence[CandidateTrade],
    params: CandidateParams,
) -> dict[str, Any]:
    """가장 수익이 큰 1·3·5개 거래를 제거한 민감도 결과를 반환한다."""

    trade_list = list(trades)
    ranked = sorted(
        range(len(trade_list)),
        key=lambda position: (
            -trade_list[position].net_r,
            trade_list[position].exit_time,
            trade_list[position].symbol,
            trade_list[position].entry_time,
        ),
    )
    results: dict[str, Any] = {}
    for count in (1, 3, 5):
        removed_positions = set(ranked[: min(count, len(ranked))])
        removed = [trade_list[position] for position in sorted(removed_positions)]
        retained = [
            trade
            for position, trade in enumerate(trade_list)
            if position not in removed_positions
        ]
        results[f"top_{count}"] = {
            "removed": [
                {
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "net_r": trade.net_r,
                }
                for trade in sorted(removed, key=lambda item: item.net_r, reverse=True)
            ],
            "retained_summary": summarize(retained, params),
        }
    return results


def numeric_summary_delta(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, float]:
    """두 요약의 공통 유한 수치에 대해 후보-기준 차이를 계산한다."""

    delta: dict[str, float] = {}
    for key in sorted(set(candidate).intersection(reference)):
        candidate_value = candidate[key]
        reference_value = reference[key]
        if isinstance(candidate_value, bool) or isinstance(reference_value, bool):
            continue
        if not isinstance(candidate_value, (int, float)) or not isinstance(
            reference_value,
            (int, float),
        ):
            continue
        if not math.isfinite(float(candidate_value)) or not math.isfinite(
            float(reference_value)
        ):
            continue
        delta[key] = round(float(candidate_value) - float(reference_value), 6)
    return delta


def conservative_bootstrap_mdd(bootstrap: Mapping[str, Mapping[str, Any]]) -> float:
    """여러 달력 블록 중 가장 큰 MDD p95를 원장용 보수값으로 고른다."""

    values = [
        float(result["max_drawdown_r_p95"])
        for result in bootstrap.values()
        if result.get("status") == "ok" and result.get("max_drawdown_r_p95") is not None
    ]
    if not values:
        raise ValueError("원장에 기록할 bootstrap MDD p95가 없습니다.")
    return max(values)


def metrics_for_ledger(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Mapping[str, Any]],
    frequency: Mapping[str, Any],
) -> ParetoMetrics:
    """엄격 요약·bootstrap·월 빈도를 검증된 원장 지표로 변환한다."""

    required = ("profit_factor", "expectancy_r", "net_r", "max_drawdown_r")
    missing = [name for name in required if summary.get(name) is None]
    if missing or frequency.get("median_per_month") is None:
        raise ValueError(f"원장 지표가 불완전합니다: {missing}")
    return ParetoMetrics(
        profit_factor=float(summary["profit_factor"]),
        expectancy_r=float(summary["expectancy_r"]),
        net_r=float(summary["net_r"]),
        max_drawdown_r=float(summary["max_drawdown_r"]),
        bootstrap_mdd_p95_r=conservative_bootstrap_mdd(bootstrap),
        trades_per_month=float(frequency["median_per_month"]),
    )


def data_hash_manifest() -> dict[str, str]:
    """실제 재생에 사용한 가격·펀딩 파일의 SHA256을 반환한다."""

    return {
        "core_btc_eth_1h": sha256_file(CORE_PATH),
        "sol_1h": sha256_file(SOL_PATH),
        "xrp_doge_1h": sha256_file(PAIR_PATH),
        "five_symbol_actual_funding": sha256_file(PAIR_FUNDING_PATH),
    }


def code_hash() -> str:
    """러너와 두 재사용 모듈을 묶은 결정론적 코드 해시를 반환한다."""

    return sha256_files(
        [
            Path(__file__),
            ROOT / "lab" / "validate_live_candidate.py",
            ROOT / "lab" / "pareto_trial_ledger.py",
        ],
        root=ROOT,
    )


def coverage_manifest(
    frames: Mapping[str, pd.DataFrame],
    ready_masks: Mapping[str, pd.Series],
    common_signal_start: pd.Timestamp,
    common_signal_end: pd.Timestamp,
) -> dict[str, Any]:
    """심볼별 readiness와 공통 평가기간을 JSON용 사전으로 만든다."""

    symbols: dict[str, Any] = {}
    for symbol in SYMBOLS:
        ready_index = ready_masks[symbol].index[ready_masks[symbol]]
        symbols[symbol] = {
            "rows": len(frames[symbol]),
            "data_first": frames[symbol].index[0].isoformat(),
            "data_last": frames[symbol].index[-1].isoformat(),
            "ready_signal_first": ready_index[0].isoformat(),
            "ready_signal_last": ready_index[-1].isoformat(),
            "ready_signal_bars": int(ready_masks[symbol].sum()),
        }
    return {
        "policy": (
            "후보 entry_regime_ready 신호봉 마스크를 후보와 no-filter 기준에 동일 적용; "
            "5개 심볼 ready 구간의 교집합만 집계"
        ),
        "common_signal_start": common_signal_start.isoformat(),
        "common_signal_end": common_signal_end.isoformat(),
        "common_entry_start": (
            common_signal_start + pd.Timedelta(hours=1)
        ).isoformat(),
        "common_entry_end": (
            common_signal_end + pd.Timedelta(hours=1)
        ).isoformat(),
        "symbols": symbols,
    }


def append_completed_trials(
    ledger_path: Path,
    candidate: CandidateParams,
    reference: CandidateParams,
    hashes: Mapping[str, str],
    runner_code_hash: str,
    candidate_metrics: ParetoMetrics,
    reference_metrics: ParetoMetrics,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """후보와 기준 completed trial을 append-only discovery 원장에 기록한다."""

    ledger = ParetoTrialLedger(ledger_path)
    common_metadata = {
        "status": "FAIL_DISCOVERY_ONLY",
        "promotion_allowed": False,
        "coverage_policy": coverage["policy"],
        "common_signal_start": coverage["common_signal_start"],
        "common_signal_end": coverage["common_signal_end"],
        "bootstrap_mdd_definition": (
            "14/28/56/84d realized-calendar circular bootstrap MDD p95 중 최댓값"
        ),
        "monthly_frequency_definition": "공통 coverage의 완전한 UTC 달, 0건 포함 중앙값",
    }
    reference_result = ledger.append_success(
        trial_name="pareto_stop7_no_volatility_filter_matched_reference",
        params={
            **asdict(reference),
            "coverage_ready_source": "candidate_entry_regime_ready",
        },
        data_hashes=hashes,
        code_hash=runner_code_hash,
        metrics=reference_metrics,
        metadata={**common_metadata, "role": "matched_reference"},
    )
    candidate_result = ledger.append_success(
        trial_name="pareto_stop7_volatility_365d_q60_candidate",
        params=asdict(candidate),
        data_hashes=hashes,
        code_hash=runner_code_hash,
        metrics=candidate_metrics,
        metadata={**common_metadata, "role": "candidate"},
    )
    return {
        "path": str(ledger_path),
        "reference": {
            "trial_id": reference_result.trial_id,
            "appended": reference_result.appended,
            "outcome": reference_result.record["outcome"],
        },
        "candidate": {
            "trial_id": candidate_result.trial_id,
            "appended": candidate_result.appended,
            "outcome": candidate_result.record["outcome"],
        },
    }


def run_validation(output_dir: Path, ledger_path: Path) -> dict[str, Any]:
    """고정 후보와 matched reference의 엄격 discovery 검증을 실행하고 저장한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = fixed_candidate_params()
    reference = fixed_reference_params(candidate)
    (
        frames,
        funding_frame,
        ready_masks,
        common_signal_start,
        common_signal_end,
    ) = prepare_inputs(candidate)
    candidate_by_symbol = replay_matched(
        frames,
        funding_frame,
        ready_masks,
        candidate,
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
    candidate_trades = flatten_trades(candidate_by_symbol)
    reference_trades = flatten_trades(reference_by_symbol)

    entry_coverage_start = common_signal_start + pd.Timedelta(hours=1)
    entry_coverage_end = common_signal_end + pd.Timedelta(hours=1)
    calendar_end = min(frame.index[-1] for frame in frames.values())
    candidate_frequency = complete_month_counts(
        [pd.Timestamp(trade.entry_time) for trade in candidate_trades],
        entry_coverage_start,
        entry_coverage_end,
    )
    reference_frequency = complete_month_counts(
        [pd.Timestamp(trade.entry_time) for trade in reference_trades],
        entry_coverage_start,
        entry_coverage_end,
    )
    candidate_strict = dimension_summary(
        candidate_trades,
        candidate,
        entry_coverage_start,
        entry_coverage_end,
    )
    reference_strict = dimension_summary(
        reference_trades,
        reference,
        entry_coverage_start,
        entry_coverage_end,
    )

    stress_candidate_params = replace(candidate, cost_bps_side=20.0)
    stress_reference_params = replace(reference, cost_bps_side=20.0)
    stressed_candidate_trades = apply_execution_funding_stress(
        candidate_trades,
        candidate.cost_bps_side,
    )
    stressed_reference_trades = apply_execution_funding_stress(
        reference_trades,
        reference.cost_bps_side,
    )
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

    candidate_metrics = metrics_for_ledger(
        candidate_strict["aggregate"],
        candidate_bootstrap,
        candidate_frequency,
    )
    reference_metrics = metrics_for_ledger(
        reference_strict["aggregate"],
        reference_bootstrap,
        reference_frequency,
    )
    pareto = compare_pareto(candidate_metrics, reference_metrics)
    hashes = data_hash_manifest()
    runner_code_hash = code_hash()
    coverage = coverage_manifest(
        frames,
        ready_masks,
        common_signal_start,
        common_signal_end,
    )
    ledger_result = append_completed_trials(
        ledger_path,
        candidate,
        reference,
        hashes,
        runner_code_hash,
        candidate_metrics,
        reference_metrics,
        coverage,
    )

    results: dict[str, Any] = {
        "classification": "DISCOVERY_ONLY_NOT_PREREGISTERED",
        "gate": {
            "status": "FAIL",
            "promotion_allowed": False,
            "promotion_capability": "DISABLED_IN_DISCOVERY_RUNNER",
            "reason": (
                "이미 관측한 자료의 단일 후보 비교이므로 성과 수치나 Pareto 우위와 무관하게 "
                "실거래 승격을 발급하지 않음"
            ),
        },
        "contracts": {
            "candidate": asdict(candidate),
            "matched_reference": {
                **asdict(reference),
                "coverage_ready_source": "candidate_entry_regime_ready",
            },
            "strict_execution": "편도 12bp + Bybit 실제 정산시각 펀딩",
            "stress_execution": "편도 20bp + 펀딩 차변 2배 + 펀딩 대변 0",
            "additions": "disabled",
        },
        "coverage": coverage,
        "data_hashes": hashes,
        "code_hash": runner_code_hash,
        "strict_12bp_actual_funding": {
            "candidate": candidate_strict,
            "matched_reference": reference_strict,
        },
        "frequency": {
            "candidate_trades_per_complete_month": candidate_frequency,
            "reference_trades_per_complete_month": reference_frequency,
        },
        "stress_20bp_funding_debit_x2_credit_zero": {
            "candidate": dimension_summary(
                stressed_candidate_trades,
                stress_candidate_params,
                entry_coverage_start,
                entry_coverage_end,
            ),
            "matched_reference": dimension_summary(
                stressed_reference_trades,
                stress_reference_params,
                entry_coverage_start,
                entry_coverage_end,
            ),
        },
        "realized_calendar_circular_bootstrap": {
            "strict_12bp_actual_funding": {
                "candidate": candidate_bootstrap,
                "matched_reference": reference_bootstrap,
            },
            "stress_20bp_funding_debit_x2_credit_zero": {
                "candidate": stressed_candidate_bootstrap,
                "matched_reference": stressed_reference_bootstrap,
            },
        },
        "entry_clusters_6h": {
            "candidate": six_hour_entry_clusters(candidate_trades),
            "matched_reference": six_hour_entry_clusters(reference_trades),
        },
        "top_winner_removal": {
            "candidate": top_winner_removal(candidate_trades, candidate),
            "matched_reference": top_winner_removal(reference_trades, reference),
        },
        "matched_reference_delta": {
            "sign_convention": "candidate_minus_reference; drawdown 양수는 후보가 더 위험",
            "strict_summary": numeric_summary_delta(
                candidate_strict["aggregate"],
                reference_strict["aggregate"],
            ),
            "stress_summary": numeric_summary_delta(
                summarize(stressed_candidate_trades, stress_candidate_params),
                summarize(stressed_reference_trades, stress_reference_params),
            ),
            "median_trades_per_complete_month": round(
                candidate_metrics.trades_per_month - reference_metrics.trades_per_month,
                6,
            ),
            "bootstrap_mdd_p95_r": round(
                candidate_metrics.bootstrap_mdd_p95_r
                - reference_metrics.bootstrap_mdd_p95_r,
                6,
            ),
            "pareto": pareto.to_dict(),
        },
        "ledger": ledger_result,
        "limitations": {
            "mtm_modeled": False,
            "margin_modeled": False,
            "liquidation_modeled": False,
            "explicit": [
                "보유 중 mark-to-market 계좌 경로와 미실현 낙폭을 모델링하지 않음",
                "교차·격리 마진, 유지증거금, 레버리지와 증거금 부족을 모델링하지 않음",
                "강제청산 가격과 청산 수수료를 모델링하지 않음",
                "1시간 OHLCV라 봉내 실제 가격 경로·부분체결·슬리피지 꼬리를 복원하지 못함",
                "거래소 장애·API 지연·주문 거절·호가 잔량을 모델링하지 않음",
            ],
        },
    }
    result_path = output_dir / "latest_results.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Pareto discovery 결과 저장: %s", result_path)
    return results


def main() -> None:
    """CLI 인수를 읽어 고정 Pareto discovery 검증을 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_validation(args.output_dir, args.ledger)
    logger.info("gate=%s", result["gate"])
    logger.info("pareto=%s", result["matched_reference_delta"]["pareto"])


if __name__ == "__main__":
    main()
