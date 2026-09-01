from __future__ import annotations

"""실행형 ICT·기술적 합의 지표의 분할진입 독립 검증기."""

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from validate_ict_bb_mtf import (
    IndicatorParams,
    build_feature_frame,
    compute_metrics,
    pine_atr,
    pine_ema,
    pine_rma,
    sha256_file,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "logs" / "validation" / "ict_bb_mtf"
OUTPUT_DIR = ROOT / "logs" / "validation" / "execution_overlay"
PINE_PATH = ROOT / "tradingview" / "ict_bb_mtf_confluence.pine"


@dataclass(frozen=True)
class ExecutionParams:
    """Pine 실행형 기본값과 동일한 검증 파라미터."""

    score_threshold: int = 68
    score_lead: int = 12
    min_ict_score: int = 3
    stop_atr: float = 1.2
    max_stop_atr: float = 2.5
    minimum_stop_percent: float = 0.25
    rr_target: float = 1.8
    add_fractions: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)
    tranche_weights: tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 20.0)
    max_holding_hours: float = 24.0
    cooldown_bars: int = 10
    mtf_bias_threshold: float = 0.75
    adx_threshold: float = 20.0
    profile_lookback: int = 240
    relative_volume_length: int = 20
    cmf_length: int = 20
    volatility_rank_length: int = 100


@dataclass(frozen=True)
class ExecutionTrade:
    """최초 진입과 조건부 추매를 합친 한 거래의 결과."""

    symbol: str
    timeframe: str
    signal_time: str
    exit_time: str
    direction: str
    entry: float
    average_entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str
    holding_bars: int
    additions: int
    score: int
    opposing_score: int
    gross_r: float
    net_r: float


def _true_range(frame: pd.DataFrame) -> pd.Series:
    """OHLC에서 Wilder True Range를 계산한다."""

    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def _supertrend(
    frame: pd.DataFrame,
    length: int = 10,
    factor: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Pine ``ta.supertrend`` 방향과 밴드를 순방향으로 계산한다."""

    atr = pine_atr(frame, length)
    midpoint = (frame["high"] + frame["low"]) / 2.0
    basic_upper = midpoint + factor * atr
    basic_lower = midpoint - factor * atr
    close = frame["close"].to_numpy(dtype=float)
    upper_source = basic_upper.to_numpy(dtype=float)
    lower_source = basic_lower.to_numpy(dtype=float)
    upper = np.full(len(frame), np.nan)
    lower = np.full(len(frame), np.nan)
    value = np.full(len(frame), np.nan)
    direction = np.full(len(frame), np.nan)

    for index in range(len(frame)):
        if not math.isfinite(upper_source[index]) or not math.isfinite(lower_source[index]):
            continue
        if index == 0 or not math.isfinite(upper[index - 1]):
            upper[index] = upper_source[index]
            lower[index] = lower_source[index]
            value[index] = upper[index]
            direction[index] = 1.0
            continue
        upper[index] = (
            upper_source[index]
            if upper_source[index] < upper[index - 1] or close[index - 1] > upper[index - 1]
            else upper[index - 1]
        )
        lower[index] = (
            lower_source[index]
            if lower_source[index] > lower[index - 1] or close[index - 1] < lower[index - 1]
            else lower[index - 1]
        )
        if value[index - 1] == upper[index - 1]:
            value[index] = upper[index] if close[index] <= upper[index] else lower[index]
        else:
            value[index] = lower[index] if close[index] >= lower[index] else upper[index]
        direction[index] = -1.0 if value[index] == lower[index] else 1.0

    return (
        pd.Series(value, index=frame.index, dtype=float),
        pd.Series(direction, index=frame.index, dtype=float),
    )


def _dmi(
    frame: pd.DataFrame,
    length: int = 14,
    smoothing: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder DMI와 ADX를 계산한다."""

    upward = frame["high"].diff()
    downward = -frame["low"].diff()
    plus_dm = pd.Series(
        np.where((upward > downward) & (upward > 0.0), upward, 0.0),
        index=frame.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((downward > upward) & (downward > 0.0), downward, 0.0),
        index=frame.index,
        dtype=float,
    )
    true_range_rma = pine_rma(_true_range(frame), length)
    plus_di = 100.0 * pine_rma(plus_dm, length) / true_range_rma
    minus_di = 100.0 * pine_rma(minus_dm, length) / true_range_rma
    denominator = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    dx = dx.mask(denominator == 0.0, 0.0)
    return plus_di, minus_di, pine_rma(dx, smoothing)


def _session_vwap(frame: pd.DataFrame) -> pd.Series:
    """UTC 일 경계로 초기화되는 24시간 시장 세션 VWAP을 계산한다."""

    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    session = frame.index.floor("D")
    numerator = (typical * frame["volume"]).groupby(session).cumsum()
    denominator = frame["volume"].groupby(session).cumsum()
    return numerator / denominator.replace(0.0, np.nan)


def add_execution_features(
    frame: pd.DataFrame,
    chart_minutes: int,
    params: ExecutionParams,
) -> pd.DataFrame:
    """실행 합의 점수에 필요한 추세·강도·거래량 특징을 추가한다."""

    output = frame.copy()
    close = output["close"]
    ema_fast = pine_ema(close, 20)
    ema_trend = pine_ema(close, 50)
    session_vwap = _session_vwap(output)
    _, supertrend_direction = _supertrend(output)
    plus_di, minus_di, adx = _dmi(output)
    macd_line = pine_ema(close, 12) - pine_ema(close, 26)
    macd_signal = pine_ema(macd_line, 9)
    macd_histogram = macd_line - macd_signal
    macd_bull = (macd_line > macd_signal) & (macd_histogram > 0.0)
    macd_bear = (macd_line < macd_signal) & (macd_histogram < 0.0)

    average_volume = output["volume"].rolling(
        params.relative_volume_length,
        min_periods=params.relative_volume_length,
    ).mean()
    relative_volume = output["volume"] / average_volume
    price_range = output["high"] - output["low"]
    money_flow_multiplier = (
        ((close - output["low"]) - (output["high"] - close))
        / price_range.replace(0.0, np.nan)
    ).fillna(0.0)
    average_money_flow = (money_flow_multiplier * output["volume"]).rolling(
        params.cmf_length,
        min_periods=params.cmf_length,
    ).mean()
    average_cmf_volume = output["volume"].rolling(
        params.cmf_length,
        min_periods=params.cmf_length,
    ).mean()
    cmf = average_money_flow / average_cmf_volume.replace(0.0, np.nan)
    volume_delta = np.where(
        close > output["open"],
        output["volume"],
        np.where(close < output["open"], -output["volume"], 0.0),
    )
    cvd = pd.Series(volume_delta, index=output.index, dtype=float).cumsum()
    cvd_average = pine_ema(cvd, 20)

    typical = (output["high"] + output["low"] + close) / 3.0
    profile_numerator = (typical * output["volume"]).rolling(
        params.profile_lookback,
        min_periods=params.profile_lookback,
    ).mean()
    profile_volume = output["volume"].rolling(
        params.profile_lookback,
        min_periods=params.profile_lookback,
    ).mean()
    rolling_volume_center = profile_numerator / profile_volume.replace(0.0, np.nan)
    atr_percent = output["atr"] / close * 100.0
    volatility_rank = (
        atr_percent.rolling(
            params.volatility_rank_length,
            min_periods=params.volatility_rank_length,
        ).rank(pct=True)
        * 100.0
    )

    if chart_minutes == 1:
        long_aligned = (output["bias5"] >= 1) & (output["bias15"] >= 1)
        short_aligned = (output["bias5"] <= -1) & (output["bias15"] <= -1)
    elif chart_minutes == 5:
        long_aligned = output["bias15"] >= 1
        short_aligned = output["bias15"] <= -1
    else:
        local_bias = compute_metrics(output[["open", "high", "low", "close", "volume"]], IndicatorParams())["bias"]
        long_aligned = local_bias >= 2
        short_aligned = local_bias <= -2

    long_trend = (
        (ema_fast > ema_trend).astype(int) * 5
        + (close > session_vwap).astype(int) * 5
        + (supertrend_direction < 0.0).astype(int) * 5
        + macd_bull.astype(int) * 5
        + (long_aligned & (output["weighted_bias"] >= params.mtf_bias_threshold)).astype(int) * 5
    )
    short_trend = (
        (ema_fast < ema_trend).astype(int) * 5
        + (close < session_vwap).astype(int) * 5
        + (supertrend_direction > 0.0).astype(int) * 5
        + macd_bear.astype(int) * 5
        + (short_aligned & (output["weighted_bias"] <= -params.mtf_bias_threshold)).astype(int) * 5
    )
    long_momentum = (
        (output["rsi"] > 50.0).astype(int) * 5
        + (output["stoch_k"] > output["stoch_d"]).astype(int) * 5
        + (output["stc"] > 50.0).astype(int) * 5
        + ((adx >= params.adx_threshold) & (plus_di > minus_di)).astype(int) * 5
    )
    short_momentum = (
        (output["rsi"] < 50.0).astype(int) * 5
        + (output["stoch_k"] < output["stoch_d"]).astype(int) * 5
        + (output["stc"] < 50.0).astype(int) * 5
        + ((adx >= params.adx_threshold) & (minus_di > plus_di)).astype(int) * 5
    )
    long_volume = (
        (cmf > 0.0).astype(int) * 5
        + (cvd > cvd_average).astype(int) * 5
        + (relative_volume >= 1.0).astype(int) * 5
    )
    short_volume = (
        (cmf < 0.0).astype(int) * 5
        + (cvd < cvd_average).astype(int) * 5
        + (relative_volume >= 1.0).astype(int) * 5
    )
    long_ict = np.floor(np.minimum(output["long_score"], 12) / 12.0 * 25.0 + 0.5).astype(int)
    short_ict = np.floor(np.minimum(output["short_score"], 12) / 12.0 * 25.0 + 0.5).astype(int)
    volatility_usable = (volatility_rank >= 15.0) & (volatility_rank <= 90.0)
    long_location = (
        output["recent_bb_long"].astype(int) * 5
        + (close >= rolling_volume_center).astype(int) * 5
        + volatility_usable.astype(int) * 5
    )
    short_location = (
        output["recent_bb_short"].astype(int) * 5
        + (close <= rolling_volume_center).astype(int) * 5
        + volatility_usable.astype(int) * 5
    )

    output["long_execution_score"] = long_trend + long_momentum + long_volume + long_ict + long_location
    output["short_execution_score"] = short_trend + short_momentum + short_volume + short_ict + short_location
    output["long_aligned"] = long_aligned
    output["short_aligned"] = short_aligned
    return output


def apply_execution_signals(
    frame: pd.DataFrame,
    params: ExecutionParams,
    threshold: int,
) -> pd.DataFrame:
    """합의 점수·점수차·ICT 최소치·쿨다운으로 진입 신호를 만든다."""

    output = frame.copy()
    raw_long = (
        output["long_aligned"]
        & (output["long_score"] >= params.min_ict_score)
        & (output["long_execution_score"] >= threshold)
        & (
            output["long_execution_score"] - output["short_execution_score"]
            >= params.score_lead
        )
    ).fillna(False)
    raw_short = (
        output["short_aligned"]
        & (output["short_score"] >= params.min_ict_score)
        & (output["short_execution_score"] >= threshold)
        & (
            output["short_execution_score"] - output["long_execution_score"]
            >= params.score_lead
        )
    ).fillna(False)
    long_signal = np.zeros(len(output), dtype=bool)
    short_signal = np.zeros(len(output), dtype=bool)
    last_long = -10**9
    last_short = -10**9
    raw_long_values = raw_long.to_numpy(dtype=bool)
    raw_short_values = raw_short.to_numpy(dtype=bool)
    for index in range(len(output)):
        previous_long = raw_long_values[index - 1] if index else False
        previous_short = raw_short_values[index - 1] if index else False
        if raw_long_values[index] and not previous_long and index - last_long > params.cooldown_bars:
            long_signal[index] = True
            last_long = index
        if raw_short_values[index] and not previous_short and index - last_short > params.cooldown_bars:
            short_signal[index] = True
            last_short = index
    output["long_signal"] = long_signal
    output["short_signal"] = short_signal
    return output


def _trade_pnl_r(
    fills: list[tuple[float, float]],
    direction: int,
    exit_price: float,
    cost_bps_side: float,
) -> tuple[float, float, float]:
    """위험예산을 1R로 두고 분할 체결의 평균가·총손익을 계산한다."""

    quantity = sum(item[1] for item in fills)
    average_entry = sum(price * qty for price, qty in fills) / quantity
    gross = sum(direction * (exit_price - price) * qty for price, qty in fills)
    cost_rate = cost_bps_side / 10_000.0
    costs = cost_rate * (
        sum(price * qty for price, qty in fills) + exit_price * quantity
    )
    return average_entry, gross, gross - costs


def simulate_execution(
    frame: pd.DataFrame,
    symbol: str,
    chart_minutes: int,
    params: ExecutionParams,
    threshold: int,
    cost_bps_side: float,
) -> list[ExecutionTrade]:
    """손절우선·갭악화·조건부 추매·시간손절로 실행 거래를 재생한다."""

    trades: list[ExecutionTrade] = []
    index = 0
    total_weight = sum(params.tranche_weights)
    tranche_risks = [weight / total_weight for weight in params.tranche_weights]
    maximum_holding_bars = max(
        1,
        math.ceil(params.max_holding_hours * 60.0 / chart_minutes),
    )
    while index < len(frame) - 1:
        signal = frame.iloc[index]
        direction = 1 if bool(signal["long_signal"]) else -1 if bool(signal["short_signal"]) else 0
        if direction == 0:
            index += 1
            continue

        entry = float(signal["close"])
        atr = float(signal["atr"])
        structure = float(
            signal["last_swing_low"] if direction > 0 else signal["last_swing_high"]
        )
        structure_distance = (
            entry - structure
            if direction > 0 and math.isfinite(structure) and structure < entry
            else structure - entry
            if direction < 0 and math.isfinite(structure) and structure > entry
            else atr * params.stop_atr
        )
        minimum_price_risk = entry * params.minimum_stop_percent / 100.0
        maximum_allowed_risk = max(atr * params.max_stop_atr, minimum_price_risk)
        risk_distance = max(
            atr * params.stop_atr,
            minimum_price_risk,
            min(structure_distance, maximum_allowed_risk),
        )
        if not math.isfinite(risk_distance) or risk_distance <= 0.0:
            index += 1
            continue
        stop = entry - direction * risk_distance
        add_levels = [
            entry - direction * risk_distance * fraction
            for fraction in params.add_fractions
        ]
        base_qty = tranche_risks[0] / risk_distance
        add_quantities = [
            tranche_risks[index + 1] / abs(level - stop)
            if tranche_risks[index + 1] > 0.0
            else 0.0
            for index, level in enumerate(add_levels)
        ]
        fills: list[tuple[float, float]] = [(entry, base_qty)]
        average_entry = entry
        target = entry + direction * risk_distance * params.rr_target
        add_done = [False] * 4
        exit_index = len(frame) - 1
        exit_price = float(frame.iloc[-1]["close"])
        exit_reason = "end"

        for cursor in range(
            index + 1,
            min(len(frame), index + maximum_holding_bars + 1),
        ):
            candle = frame.iloc[cursor]
            candle_open = float(candle["open"])
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            candle_close = float(candle["close"])
            stop_touched = candle_low <= stop if direction > 0 else candle_high >= stop
            target_touched = candle_high >= target if direction > 0 else candle_low <= target
            if stop_touched:
                exit_price = min(stop, candle_open) if direction > 0 else max(stop, candle_open)
                exit_index = cursor
                exit_reason = "stop" if not target_touched else "both_stop_first"
                break
            if target_touched:
                exit_price = target
                exit_index = cursor
                exit_reason = "target"
                break

            direction_score = float(
                candle["long_execution_score"]
                if direction > 0
                else candle["short_execution_score"]
            )
            opposing_score = float(
                candle["short_execution_score"]
                if direction > 0
                else candle["long_execution_score"]
            )
            direction_valid = (
                direction_score >= threshold - 15
                and direction_score + 5 >= opposing_score
            )
            next_add_index = next(
                (
                    add_index
                    for add_index in range(4)
                    if not add_done[add_index]
                    and add_quantities[add_index] > 0.0
                ),
                None,
            )
            if next_add_index is not None:
                next_level = add_levels[next_add_index]
                add_reclaim = (
                    candle_low <= next_level
                    and candle_close > next_level
                    and candle_close > candle_open
                    if direction > 0
                    else candle_high >= next_level
                    and candle_close < next_level
                    and candle_close < candle_open
                )
                if direction_valid and add_reclaim:
                    next_quantity = add_quantities[next_add_index]
                    fills.append((next_level, next_quantity))
                    add_done[next_add_index] = True
                    average_entry = sum(
                        price * qty for price, qty in fills
                    ) / sum(qty for _, qty in fills)
                    target = (
                        average_entry
                        + direction
                        * abs(average_entry - stop)
                        * params.rr_target
                    )

            if cursor == index + maximum_holding_bars:
                exit_price = candle_close
                exit_index = cursor
                exit_reason = "time_exit_24h"
                break

        average_entry, gross_r, net_r = _trade_pnl_r(
            fills,
            direction,
            exit_price,
            cost_bps_side,
        )
        trades.append(
            ExecutionTrade(
                symbol=symbol,
                timeframe=f"{chart_minutes}m",
                signal_time=frame.index[index].isoformat(),
                exit_time=frame.index[exit_index].isoformat(),
                direction="long" if direction > 0 else "short",
                entry=round(entry, 8),
                average_entry=round(average_entry, 8),
                stop=round(stop, 8),
                target=round(target, 8),
                exit=round(exit_price, 8),
                exit_reason=exit_reason,
                holding_bars=exit_index - index,
                additions=sum(add_done),
                score=int(signal["long_execution_score"] if direction > 0 else signal["short_execution_score"]),
                opposing_score=int(signal["short_execution_score"] if direction > 0 else signal["long_execution_score"]),
                gross_r=round(gross_r, 8),
                net_r=round(net_r, 8),
            )
        )
        index = exit_index + 1
    return trades


def summarize(trades: list[ExecutionTrade]) -> dict[str, Any]:
    """R 기대값·PF·낙폭·추매·종료 사유를 요약한다."""

    if not trades:
        return {"trades": 0, "expectancy_r": None, "profit_factor": None}
    values = np.array([trade.net_r for trade in trades], dtype=float)
    gains = values[values > 0.0].sum()
    losses = -values[values < 0.0].sum()
    equity = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": len(trades),
        "win_rate": round(float(np.mean(values > 0.0)), 6),
        "expectancy_r": round(float(values.mean()), 6),
        "net_r": round(float(values.sum()), 6),
        "profit_factor": round(float(gains / losses), 6) if losses > 0.0 else None,
        "max_drawdown_r": round(float(drawdown.max()), 6),
        "target_rate": round(float(np.mean([trade.exit_reason == "target" for trade in trades])), 6),
        "time_exit_rate": round(float(np.mean([trade.exit_reason == "time_exit_24h" for trade in trades])), 6),
        "add1_rate": round(float(np.mean([trade.additions >= 1 for trade in trades])), 6),
        "add2_rate": round(float(np.mean([trade.additions >= 2 for trade in trades])), 6),
        "add3_rate": round(float(np.mean([trade.additions >= 3 for trade in trades])), 6),
        "add4_rate": round(float(np.mean([trade.additions >= 4 for trade in trades])), 6),
        "median_holding_bars": round(float(np.median([trade.holding_bars for trade in trades])), 2),
    }


def split_summary(trades: list[ExecutionTrade]) -> dict[str, Any]:
    """시간순 50%·25%·25% 분할 성과를 반환한다."""

    if not trades:
        return {}
    ordered = sorted(trades, key=lambda trade: trade.signal_time)
    first = len(ordered) // 2
    second = first + len(ordered) // 4
    return {
        "in_sample_50": summarize(ordered[:first]),
        "validation_25": summarize(ordered[first:second]),
        "holdout_25": summarize(ordered[second:]),
    }


def bootstrap(trades: list[ExecutionTrade], samples: int = 10_000) -> dict[str, Any]:
    """거래 R을 재표집해 양수 종결과 기대값 불확실성을 계산한다."""

    values = np.array([trade.net_r for trade in trades], dtype=float)
    if len(values) < 20:
        return {"status": "insufficient_trades", "trades": len(values)}
    rng = np.random.default_rng(20260831)
    means = np.empty(samples, dtype=float)
    finals = np.empty(samples, dtype=float)
    processed = 0
    while processed < samples:
        count = min(250, samples - processed)
        sampled = rng.choice(values, size=(count, len(values)), replace=True)
        means[processed : processed + count] = sampled.mean(axis=1)
        finals[processed : processed + count] = sampled.sum(axis=1)
        processed += count
    return {
        "samples": samples,
        "probability_positive_final": round(float(np.mean(finals > 0.0)), 6),
        "expectancy_r_p05": round(float(np.quantile(means, 0.05)), 6),
        "expectancy_r_p50": round(float(np.quantile(means, 0.50)), 6),
        "expectancy_r_p95": round(float(np.quantile(means, 0.95)), 6),
    }


def latest_data_files() -> dict[str, Path]:
    """기존 180일 원본 캐시에서 심볼별 최신 파일을 선택한다."""

    result: dict[str, Path] = {}
    for symbol in ("btc", "eth", "sol"):
        candidates = sorted(DATA_DIR.glob(f"{symbol}_1m_*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"검증 원본 캐시가 없습니다: {symbol}")
        result[symbol.upper()] = candidates[-1]
    return result


def run_validation(output_dir: Path) -> dict[str, Any]:
    """3심볼·3타임프레임·비용·점수 민감도 검증을 실행한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    indicator_params = IndicatorParams(min_score=3)
    execution_params = ExecutionParams()
    thresholds = (60, 64, 68, 72, 76)
    costs = (4.0, 5.5, 8.0)
    results: dict[str, Any] = {
        "pine_path": str(PINE_PATH.relative_to(ROOT)),
        "pine_sha256": sha256_file(PINE_PATH),
        "indicator_params": asdict(indicator_params),
        "execution_params": asdict(execution_params),
        "cost_bps_per_side": costs,
        "thresholds": thresholds,
        "data": {},
        "cells": {},
    }
    default_trades: list[ExecutionTrade] = []

    for symbol, path in latest_data_files().items():
        source = pd.read_parquet(path)
        source.index = pd.to_datetime(source.index, utc=True)
        results["data"][symbol] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(source),
            "first": source.index[0].isoformat(),
            "last": source.index[-1].isoformat(),
        }
        for chart_minutes in (1, 5, 15):
            logger.info("특징 계산: %s %dm", symbol, chart_minutes)
            base = build_feature_frame(source, chart_minutes, indicator_params)
            features = add_execution_features(base, chart_minutes, execution_params)
            for threshold in thresholds:
                signaled = apply_execution_signals(features, execution_params, threshold)
                for cost in costs:
                    trades = simulate_execution(
                        signaled,
                        symbol,
                        chart_minutes,
                        execution_params,
                        threshold,
                        cost,
                    )
                    key = f"{symbol}_{chart_minutes}m_score{threshold}_cost{cost:g}"
                    results["cells"][key] = {
                        "summary": summarize(trades),
                        "splits": split_summary(trades),
                    }
                    if threshold == execution_params.score_threshold and cost == 5.5:
                        default_trades.extend(trades)

    default_summary = summarize(default_trades)
    results["default_aggregate"] = default_summary
    results["default_bootstrap"] = bootstrap(default_trades)
    default_cells = [
        value["summary"]
        for key, value in results["cells"].items()
        if "_score68_cost5.5" in key
    ]
    positive_default_cells = sum(
        1
        for value in default_cells
        if value.get("expectancy_r") is not None and value["expectancy_r"] > 0.0
    )
    holdout_cells = [
        value["splits"].get("holdout_25", {})
        for key, value in results["cells"].items()
        if "_score68_cost5.5" in key
    ]
    positive_holdouts = sum(
        1
        for value in holdout_cells
        if value.get("expectancy_r") is not None and value["expectancy_r"] > 0.0
    )
    bootstrap_p05 = results["default_bootstrap"].get("expectancy_r_p05")
    passed = (
        positive_default_cells == 9
        and positive_holdouts >= 7
        and bootstrap_p05 is not None
        and bootstrap_p05 > 0.0
        and default_summary.get("profit_factor", 0.0) >= 1.1
    )
    results["gate"] = {
        "status": "PASS" if passed else "FAIL",
        "positive_default_cells": positive_default_cells,
        "total_default_cells": 9,
        "positive_holdout_cells": positive_holdouts,
        "required_positive_holdouts": 7,
        "bootstrap_expectancy_p05": bootstrap_p05,
        "aggregate_profit_factor": default_summary.get("profit_factor"),
    }

    json_path = output_dir / "latest_results.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("결과 저장: %s", json_path)
    return results


def main() -> None:
    """CLI 진입점."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_validation(args.output_dir)
    logger.info("검증 게이트: %s", result["gate"])


if __name__ == "__main__":
    main()
