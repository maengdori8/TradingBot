from __future__ import annotations

"""실전 후보 V3의 연구 규칙을 장기 1시간봉에서 보수적으로 검증한다.

이 검증기는 결과를 좋게 만들기 위한 파라미터 탐색기가 아니다. 기존 연구에서 이미
사용한 출판형 BRK24 규칙에 다음 실행 계약을 고정한다.

- 이전 24시간 Donchian 종가 돌파 + 이전 확정봉 SMA200·RSI14·거래량 게이트
- 이전 확정봉 ATR24의 6배 고정 최후 손절, 반대 12시간 채널 추적 청산
- 24시간 경과 시 추세 게이트를 한 번 재심사, 통과 거래만 최대 72시간 보유
- 최초 80% + 조건부 추매 네 번 각 5%의 고정 총 위험예산
- 추매는 최후 손절거리의 20%/40%/60%/80%에서 재확인한 다음 봉 시가 체결
- 편도 8/10/12bp 비용 스트레스, Bybit 실제 정산시각 펀딩률
- 갭은 불리하게, 동일 봉 청산 충돌은 손절·반대채널 우선

추매 재확인 봉의 과거 가격으로 소급 체결하지 않는다. 다음 봉 시가에서 수량을 다시
계산하고, 그 봉의 손절도 반영한다. 이 결과는 이미 관측한 자료의 discovery 결과이며
사전등록된 미래 표본이 없으므로 어떤 수치가 나오더라도 실거래 PASS를 부여하지 않는다.
"""

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "logs" / "validation" / "live_candidate_v3"
CORE_PATH = ROOT / "lab" / "frozen" / "perp_1h.parquet"
SOL_PATH = ROOT / "lab" / "data" / "sol_1h.parquet"
PAIR_PATH = ROOT / "lab" / "data" / "pairperp_1h.parquet"
FUNDING_PATH = ROOT / "lab" / "frozen" / "funding.parquet"
PAIR_FUNDING_PATH = ROOT / "lab" / "data" / "pairperp_funding.parquet"
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CandidateParams:
    """결과 확인 전에 고정한 실전 후보 파라미터."""

    entry_channel: int = 24
    exit_channel: int = 12
    atr_length: int = 24
    stop_atr: float = 6.0
    sma_length: int = 200
    macro_sma_days: int = 200
    require_macro_filter: bool = False
    rsi_length: int = 14
    volume_length: int = 20
    volatility_filter_days: int = 0
    volatility_filter_quantile: float = 0.60
    volatility_filter_min_samples: int = 30
    volatility_filter_require_full_window: bool = False
    target_r: float = 0.0
    review_holding_hours: int = 24
    max_holding_hours: int = 72
    failed_breakout_exit_hours: int = 0
    add_fractions: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)
    tranche_weights: tuple[float, ...] = (80.0, 5.0, 5.0, 5.0, 5.0)
    cost_bps_side: float = 8.0
    risk_percent: float = 0.25
    entry_close_confirmation: bool = True
    allow_short: bool = False
    discovery_only: bool = True


@dataclass(frozen=True)
class CandidateTrade:
    """한 방향성 거래의 체결·청산·비용 결과."""

    symbol: str
    entry_time: str
    exit_time: str
    direction: str
    entry: float
    average_entry: float
    stop: float
    target: float | None
    exit: float
    exit_reason: str
    holding_hours: int
    additions: int
    risk_committed_r: float
    gross_r: float
    execution_cost_r: float
    funding_cost_r: float
    net_r: float


def sha256_file(path: Path) -> str:
    """파일 SHA256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_market_data(include_external: bool = True) -> dict[str, pd.DataFrame]:
    """동결 코어와 사전 미확인 외부 심볼 1시간봉을 읽는다."""

    panel = pd.read_parquet(CORE_PATH)
    frames = {
        symbol: panel.xs(symbol, level="sym")[OHLCV].copy()
        for symbol in ("BTC", "ETH")
    }
    frames["SOL"] = pd.read_parquet(SOL_PATH)[OHLCV].copy()
    if include_external:
        external = pd.read_parquet(PAIR_PATH)
        for symbol in ("XRP", "DOGE"):
            frames[symbol] = external.xs(f"{symbol}_USDT", level="sym")[OHLCV].copy()
    for symbol, frame in frames.items():
        frame.index = pd.to_datetime(frame.index, utc=True)
        frames[symbol] = frame.sort_index().astype(float)
    return frames


def load_funding() -> pd.DataFrame:
    """같은 Bybit USDT 무기한 심볼의 실제 정산시각별 펀딩률을 읽는다."""

    funding = pd.read_parquet(PAIR_FUNDING_PATH).rename(
        columns={f"{symbol}_USDT": symbol for symbol in ("BTC", "ETH", "SOL", "XRP", "DOGE")}
    )
    funding.index = pd.to_datetime(funding.index, utc=True)
    funding = funding[["BTC", "ETH", "SOL", "XRP", "DOGE"]].sort_index()
    if funding.index.has_duplicates or not funding.index.is_monotonic_increasing:
        raise ValueError("펀딩 인덱스가 중복되거나 정렬되지 않았습니다.")
    return funding


def validate_market_frame(frame: pd.DataFrame, symbol: str) -> None:
    """부분봉·비정상 OHLCV·중복을 fail-closed로 차단한다."""

    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: 비어 있거나 중복/비정렬 데이터입니다.")
    if len(frame) > 1 and not (frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{symbol}: 누락되거나 불규칙한 1시간봉이 있습니다.")
    if frame.index.max() + pd.Timedelta(hours=1) > pd.Timestamp.now(tz="UTC").floor("h"):
        raise ValueError(f"{symbol}: 아직 닫히지 않은 마지막 1시간봉이 있습니다.")
    values = frame[OHLCV].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{symbol}: NaN/inf OHLCV가 있습니다.")
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError(f"{symbol}: 0 이하 가격이 있습니다.")
    if (frame["volume"] < 0.0).any():
        raise ValueError(f"{symbol}: 음수 거래량이 있습니다.")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: high 무결성 오류입니다.")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"{symbol}: low 무결성 오류입니다.")


def validate_funding_coverage(frame: pd.DataFrame, funding: pd.Series, symbol: str) -> pd.DataFrame:
    """펀딩 열·유효 범위를 검사하고 가격을 마지막 정산시각으로 보수적으로 자른다."""

    rates = funding.dropna().astype(float)
    if rates.empty or not np.isfinite(rates.to_numpy()).all():
        raise ValueError(f"{symbol}: 펀딩 열이 없거나 비정상입니다.")
    if len(rates) > 1 and rates.index.to_series().diff().dropna().max() > pd.Timedelta(hours=8):
        raise ValueError(f"{symbol}: 8시간을 넘는 펀딩 정산 공백이 있습니다.")
    if len(rates) >= 20:
        gaps = rates.index.to_series().diff().dt.total_seconds().div(3600.0)
        prior_typical = gaps.shift(1).rolling(16, min_periods=4).median()
        next_typical = gaps.shift(-1)[::-1].rolling(16, min_periods=4).median()[::-1]
        local_interval = pd.concat([prior_typical, next_typical], axis=1).max(axis=1)
        missing_inside_regime = gaps > local_interval * 1.5
        if missing_inside_regime.fillna(False).any():
            raise ValueError(f"{symbol}: 현지 정산주기 대비 누락된 펀딩 이벤트가 있습니다.")
    trimmed = frame.loc[
        (frame.index >= rates.index.min()) & (frame.index <= rates.index.max())
    ].copy()
    if trimmed.empty:
        raise ValueError(f"{symbol}: 펀딩 공통 구간에 가격이 없습니다.")
    return trimmed


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder RMA를 계산한다."""

    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def causal_rolling_quantile(
    series: pd.Series,
    days: int,
    min_samples: int,
    quantile: float,
) -> pd.Series:
    """현재값을 제외한 ``[t-days, t)`` 표본의 선형보간 분위수를 계산한다."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("인과 분위수에는 DatetimeIndex가 필요합니다.")
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError("인과 분위수 인덱스는 중복 없이 정렬돼야 합니다.")
    return series.rolling(
        f"{days}D",
        min_periods=min_samples,
        closed="left",
    ).quantile(quantile)


def add_features(frame: pd.DataFrame, params: CandidateParams) -> pd.DataFrame:
    """모든 진입·청산 특징을 과거 확정값만으로 만든다."""

    output = frame.copy()
    previous_close = output["close"].shift(1)
    true_range = pd.concat(
        [
            output["high"] - output["low"],
            (output["high"] - previous_close).abs(),
            (output["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = wilder_rma(true_range, params.atr_length)
    change = output["close"].diff()
    gain = wilder_rma(change.clip(lower=0.0), params.rsi_length)
    loss = wilder_rma((-change).clip(lower=0.0), params.rsi_length)
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
    rsi = rsi.mask((loss == 0.0) & (gain == 0.0), 50.0)
    sma = output["close"].rolling(params.sma_length).mean()
    prior_volume_average = output["volume"].rolling(params.volume_length).mean().shift(2)
    daily_close = output["close"].resample("1D").last()
    daily_sma = daily_close.rolling(params.macro_sma_days).mean()
    confirmed_day = output.index.floor("D") - pd.Timedelta(days=1)
    macro_close = daily_close.reindex(confirmed_day).to_numpy(dtype=float)
    macro_sma = daily_sma.reindex(confirmed_day).to_numpy(dtype=float)
    macro_long = pd.Series(macro_close > macro_sma, index=output.index)
    macro_short = pd.Series(macro_close < macro_sma, index=output.index)
    macro_long_gate = macro_long if params.require_macro_filter else pd.Series(True, index=output.index)
    macro_short_gate = macro_short if params.require_macro_filter else pd.Series(True, index=output.index)
    macro_ready = (
        pd.Series(np.isfinite(macro_close) & np.isfinite(macro_sma), index=output.index)
        if params.require_macro_filter
        else pd.Series(True, index=output.index)
    )

    output["atr_entry"] = atr.shift(1)
    output["entry_high"] = output["high"].rolling(params.entry_channel).max().shift(1)
    output["entry_low"] = output["low"].rolling(params.entry_channel).min().shift(1)
    output["exit_high"] = output["high"].rolling(params.exit_channel).max().shift(1)
    output["exit_low"] = output["low"].rolling(params.exit_channel).min().shift(1)
    output["long_gate"] = (
        (output["close"].shift(1) > sma.shift(1))
        & (rsi.shift(1) > 50.0)
        & (output["volume"].shift(1) > prior_volume_average)
        & macro_long_gate
    ).fillna(False)
    output["short_gate"] = (
        (output["close"].shift(1) < sma.shift(1))
        & (rsi.shift(1) < 50.0)
        & (output["volume"].shift(1) > prior_volume_average)
        & macro_short_gate
    ).fillna(False)
    if params.volatility_filter_days < 0:
        raise ValueError("변동성 필터 기간은 0 이상이어야 합니다.")
    if params.volatility_filter_days > 0 and not params.entry_close_confirmation:
        raise ValueError("변동성 필터는 확정 종가 진입 신호에서만 사용할 수 있습니다.")
    if not 0.0 < params.volatility_filter_quantile <= 1.0:
        raise ValueError("변동성 필터 분위수는 0 초과 1 이하여야 합니다.")
    if params.volatility_filter_min_samples < 1:
        raise ValueError("변동성 필터 최소 표본은 1 이상이어야 합니다.")
    if params.volatility_filter_days == 0:
        output["entry_regime_gate"] = True
        output["entry_regime_ready"] = True
    else:
        relative_atr = output["atr_entry"] / output["close"]
        eligible_breakout = (
            (output["long_gate"] & (output["close"] > output["entry_high"]))
            | (
                params.allow_short
                & output["short_gate"]
                & (output["close"] < output["entry_low"])
            )
        )
        eligible_relative_atr = relative_atr.where(eligible_breakout)
        historical_threshold = causal_rolling_quantile(
            eligible_relative_atr,
            days=params.volatility_filter_days,
            min_samples=params.volatility_filter_min_samples,
            quantile=params.volatility_filter_quantile,
        )
        feature_ready = (
            output["atr_entry"].notna()
            & output["entry_high"].notna()
            & output["entry_low"].notna()
            & sma.shift(1).notna()
            & rsi.shift(1).notna()
            & prior_volume_average.notna()
            & macro_ready
        )
        first_ready = feature_ready[feature_ready].index.min()
        if params.volatility_filter_require_full_window and pd.notna(first_ready):
            coverage_ready = pd.Series(
                output.index >= first_ready + pd.Timedelta(days=params.volatility_filter_days),
                index=output.index,
            )
        else:
            coverage_ready = pd.Series(True, index=output.index)
        output["entry_regime_gate"] = (
            (relative_atr <= historical_threshold) & coverage_ready
        ).fillna(False)
        output["entry_regime_ready"] = historical_threshold.notna() & coverage_ready
    return output


def trade_pnl(
    fills: list[tuple[float, float, pd.Timestamp]],
    direction: int,
    exit_price: float,
    exit_time: pd.Timestamp,
    funding: pd.Series,
    settlement_marks: pd.Series,
    params: CandidateParams,
) -> tuple[float, float, float, float, float]:
    """분할 체결의 평균가와 비용 후 R 손익을 계산한다."""

    quantity = sum(quantity for _, quantity, _ in fills)
    average_entry = sum(price * quantity for price, quantity, _ in fills) / quantity
    gross_r = sum(
        direction * (exit_price - price) * quantity
        for price, quantity, _ in fills
    )
    cost_rate = params.cost_bps_side / 10_000.0
    execution_cost = cost_rate * (
        sum(price * quantity for price, quantity, _ in fills)
        + exit_price * quantity
    )
    funding_cost = 0.0
    for fill_price, fill_quantity, fill_time in fills:
        rates = funding.loc[(funding.index > fill_time) & (funding.index <= exit_time)].dropna()
        if rates.empty:
            continue
        marks = settlement_marks.reindex(rates.index)
        if marks.isna().any():
            raise ValueError("펀딩 정산시각의 가격이 누락되었습니다.")
        funding_cost += direction * float((rates * marks).sum()) * fill_quantity
    return average_entry, gross_r, execution_cost, funding_cost, gross_r - execution_cost - funding_cost


def simulate_symbol(
    frame: pd.DataFrame,
    symbol: str,
    params: CandidateParams,
    funding: pd.Series,
    allow_additions: bool = True,
) -> list[CandidateTrade]:
    """한 심볼을 다음 시가 주문·고정 위험·불리한 봉내 충돌 순서로 재생한다."""

    featured = add_features(frame, params)
    if params.failed_breakout_exit_hours < 0:
        raise ValueError("실패 돌파 청산 시간은 0 이상이어야 합니다.")
    weights = np.asarray(params.tranche_weights if allow_additions else (100.0,), dtype=float)
    fractions = params.add_fractions if allow_additions else ()
    if len(weights) != len(fractions) + 1 or (weights < 0.0).any() or weights.sum() <= 0.0:
        raise ValueError("분할 위험 배분과 추매 단계가 일치하지 않습니다.")
    weights = weights / weights.sum()
    trades: list[CandidateTrade] = []
    signal_index = params.sma_length + 2
    size = len(featured)

    while signal_index < size - 1:
        signal = featured.iloc[signal_index]
        entry_regime_gate = bool(signal.get("entry_regime_gate", True))
        long_break = entry_regime_gate and bool(signal["long_gate"]) and (
            float(signal["close"]) > float(signal["entry_high"])
            if params.entry_close_confirmation
            else float(signal["high"]) > float(signal["entry_high"])
        )
        short_break = entry_regime_gate and params.allow_short and bool(signal["short_gate"]) and (
            float(signal["close"]) < float(signal["entry_low"])
            if params.entry_close_confirmation
            else float(signal["low"]) < float(signal["entry_low"])
        )
        if long_break == short_break:
            signal_index += 1
            continue

        direction = 1 if long_break else -1
        entry_index = signal_index + 1 if params.entry_close_confirmation else signal_index
        entry_row = featured.iloc[entry_index]
        entry = (
            float(entry_row["open"])
            if params.entry_close_confirmation
            else max(float(signal["open"]), float(signal["entry_high"]))
            if direction > 0
            else min(float(signal["open"]), float(signal["entry_low"]))
        )
        atr = float(signal["atr_entry"])
        if not math.isfinite(atr) or atr <= 0.0:
            signal_index += 1
            continue
        stop_distance = atr * params.stop_atr
        stop = entry - direction * stop_distance
        add_levels = [
            entry - direction * stop_distance * fraction
            for fraction in fractions
        ]
        fills: list[tuple[float, float, pd.Timestamp]] = [
            (entry, weights[0] / stop_distance, featured.index[entry_index])
        ]
        average_entry = entry
        target: float | None = None
        add_done = [False] * len(add_levels)
        pending_add: int | None = None
        failed_breakout_exit_pending = False
        reviewed = False
        active_exit = stop
        exit_index = entry_index
        exit_price = entry
        exit_reason = "end_of_data"
        entry_time = featured.index[entry_index]

        for cursor in range(entry_index, size):
            candle = featured.iloc[cursor]
            candle_time = featured.index[cursor]
            candle_open = float(candle["open"])
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            candle_close = float(candle["close"])
            elapsed_hours = (candle_time - entry_time).total_seconds() / 3600.0

            raw_channel = float(candle["exit_low"] if direction > 0 else candle["exit_high"])
            if math.isfinite(raw_channel):
                active_exit = (
                    max(active_exit, stop, raw_channel)
                    if direction > 0
                    else min(active_exit, stop, raw_channel)
                )

            if cursor > entry_index:
                open_exit = candle_open <= active_exit if direction > 0 else candle_open >= active_exit
                if open_exit:
                    exit_price = candle_open
                    exit_index = cursor
                    exit_reason = "stop_gap" if active_exit == stop else "channel_exit"
                    break

                if failed_breakout_exit_pending:
                    exit_price = candle_open
                    exit_index = cursor
                    exit_reason = "failed_breakout_exit"
                    break

                if elapsed_hours >= params.max_holding_hours:
                    exit_price = candle_open
                    exit_index = cursor
                    exit_reason = "time_exit_72h"
                    break

                if not reviewed and elapsed_hours >= params.review_holding_hours:
                    reviewed = True
                    gate_valid = bool(
                        candle["long_gate"] if direction > 0 else candle["short_gate"]
                    )
                    if not gate_valid:
                        exit_price = candle_open
                        exit_index = cursor
                        exit_reason = "time_review_exit"
                        break

                if pending_add is not None:
                    adverse_open = candle_open < average_entry if direction > 0 else candle_open > average_entry
                    inside_final_stop = candle_open > stop if direction > 0 else candle_open < stop
                    if adverse_open and inside_final_stop:
                        actual_distance = abs(candle_open - stop)
                        addition_quantity = weights[pending_add + 1] / actual_distance
                        fills.append((candle_open, addition_quantity, candle_time))
                        add_done[pending_add] = True
                        total_quantity = sum(quantity for _, quantity, _ in fills)
                        average_entry = sum(
                            price * quantity for price, quantity, _ in fills
                        ) / total_quantity
                    pending_add = None

            final_stop_touched = candle_low <= stop if direction > 0 else candle_high >= stop
            if final_stop_touched:
                exit_price = stop
                exit_index = cursor
                exit_reason = "same_bar_stop" if cursor == entry_index else "stop"
                break

            channel_touched = candle_low <= active_exit if direction > 0 else candle_high >= active_exit
            if channel_touched:
                if cursor == entry_index and (
                    (direction > 0 and active_exit >= entry)
                    or (direction < 0 and active_exit <= entry)
                ):
                    exit_price = entry
                else:
                    exit_price = active_exit
                exit_index = cursor
                exit_reason = "channel_exit"
                break

            bars_since_entry = cursor - entry_index + 1
            breakout_level = float(signal["entry_high"] if direction > 0 else signal["entry_low"])
            failed_breakout = (
                candle_close < breakout_level
                if direction > 0
                else candle_close > breakout_level
            )
            if (
                params.failed_breakout_exit_hours > 0
                and bars_since_entry <= params.failed_breakout_exit_hours
                and failed_breakout
            ):
                failed_breakout_exit_pending = True

            if allow_additions and pending_add is None and not failed_breakout_exit_pending:
                next_add = next(
                    (position for position, done in enumerate(add_done) if not done),
                    None,
                )
                if next_add is not None:
                    level = add_levels[next_add]
                    gate_valid = bool(candle["long_gate"] if direction > 0 else candle["short_gate"])
                    reclaimed = (
                        candle_low <= level and candle_close > level and candle_close > candle_open
                        if direction > 0
                        else candle_high >= level and candle_close < level and candle_close < candle_open
                    )
                    if gate_valid and reclaimed:
                        pending_add = next_add

            exit_index = cursor
            exit_price = candle_close

        if exit_reason == "end_of_data":
            break
        exit_time = featured.index[exit_index]
        holding_hours = max(0, int((exit_time - entry_time).total_seconds() // 3600))
        average_entry, gross_r, execution_cost, funding_cost, net_r = trade_pnl(
            fills,
            direction,
            exit_price,
            exit_time,
            funding,
            featured["open"],
            params,
        )
        trades.append(
            CandidateTrade(
                symbol=symbol,
                entry_time=entry_time.isoformat(),
                exit_time=exit_time.isoformat(),
                direction="long" if direction > 0 else "short",
                entry=round(entry, 8),
                average_entry=round(average_entry, 8),
                stop=round(stop, 8),
                target=round(target, 8) if target is not None else None,
                exit=round(exit_price, 8),
                exit_reason=exit_reason,
                holding_hours=holding_hours,
                additions=sum(add_done),
                risk_committed_r=round(float(weights[0] + sum(
                    weights[position + 1] for position, done in enumerate(add_done) if done
                )), 8),
                gross_r=round(gross_r, 8),
                execution_cost_r=round(execution_cost, 8),
                funding_cost_r=round(funding_cost, 8),
                net_r=round(net_r, 8),
            )
        )
        signal_index = exit_index + 1
    return trades


def summarize(
    trades: list[CandidateTrade],
    params: CandidateParams | None = None,
) -> dict[str, Any]:
    """거래 목록을 청산시각순으로 정렬해 비용 후 실현 성과를 요약한다."""

    if not trades:
        return {"trades": 0, "expectancy_r": None, "profit_factor": None}
    ordered = sorted(trades, key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time))
    values = np.asarray([trade.net_r for trade in ordered], dtype=float)
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
        "risk_scaled_max_drawdown_percent": round(
            float(drawdown.max()) * (params.risk_percent if params is not None else 0.25),
            4,
        ),
        "median_holding_hours": round(float(np.median([trade.holding_hours for trade in trades])), 2),
        "target_rate": round(float(np.mean([trade.exit_reason == "target" for trade in trades])), 6),
        "review_exit_rate": round(float(np.mean([trade.exit_reason == "time_review_exit" for trade in trades])), 6),
        "time_exit_rate": round(float(np.mean([trade.exit_reason == "time_exit_72h" for trade in trades])), 6),
        "add1_rate": round(float(np.mean([trade.additions >= 1 for trade in trades])), 6),
        "add2_rate": round(float(np.mean([trade.additions >= 2 for trade in trades])), 6),
        "add3_rate": round(float(np.mean([trade.additions >= 3 for trade in trades])), 6),
        "add4_rate": round(float(np.mean([trade.additions >= 4 for trade in trades])), 6),
    }


def summarize_risk_efficiency(trades: list[CandidateTrade]) -> dict[str, Any]:
    """각 거래 손익을 실제 투입된 최후손절 위험으로 나눈 효율을 요약한다."""

    if not trades:
        return {"trades": 0, "expectancy_per_committed_r": None}
    ordered = sorted(trades, key=lambda trade: (trade.exit_time, trade.symbol, trade.entry_time))
    committed = np.asarray([trade.risk_committed_r for trade in ordered], dtype=float)
    if (committed <= 0.0).any():
        raise ValueError("0 이하 투입 위험 거래가 있습니다.")
    values = np.asarray([trade.net_r for trade in ordered], dtype=float) / committed
    gains = values[values > 0.0].sum()
    losses = -values[values < 0.0].sum()
    equity = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": len(trades),
        "average_committed_r": round(float(committed.mean()), 6),
        "expectancy_per_committed_r": round(float(values.mean()), 6),
        "net_per_committed_r": round(float(values.sum()), 6),
        "profit_factor_per_committed_r": round(float(gains / losses), 6)
        if losses > 0.0
        else None,
        "realized_max_drawdown_per_committed_r": round(float(drawdown.max()), 6),
    }


def block_bootstrap(trades: list[CandidateTrade], block: int = 20, samples: int = 10_000) -> dict[str, Any]:
    """끝 관측치도 같은 확률로 뽑는 순환 거래 블록 부트스트랩을 수행한다."""

    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    values = np.asarray([trade.net_r for trade in ordered], dtype=float)
    if len(values) < block * 3:
        return {"status": "insufficient", "trades": len(values)}
    rng = np.random.default_rng(20260831)
    blocks_needed = math.ceil(len(values) / block)
    starts = np.arange(len(values))
    offsets = np.arange(block)
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        path = np.concatenate(
            [values[(start + offsets) % len(values)] for start in selected]
        )[: len(values)]
        means[sample] = path.mean()
    return {
        "samples": samples,
        "block_trades": block,
        "method": "circular_moving_block",
        "expectancy_p05": round(float(np.quantile(means, 0.05)), 6),
        "expectancy_p50": round(float(np.quantile(means, 0.50)), 6),
        "expectancy_p95": round(float(np.quantile(means, 0.95)), 6),
        "probability_positive": round(float(np.mean(means > 0.0)), 6),
    }


def calendar_block_bootstrap(
    trades: list[CandidateTrade],
    block_days: int = 28,
    samples: int = 10_000,
) -> dict[str, Any]:
    """동일 날짜의 전 종목 손익을 묶은 뒤 달력 블록으로 공동 충격을 재표본한다."""

    if not trades:
        return {"status": "insufficient", "days": 0}
    exits = pd.Series(
        [trade.net_r for trade in trades],
        index=pd.DatetimeIndex([pd.Timestamp(trade.exit_time) for trade in trades]),
        dtype=float,
    )
    daily = exits.groupby(exits.index.floor("D")).sum().sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="1D", tz="UTC")
    values = daily.reindex(full_index, fill_value=0.0).to_numpy(dtype=float)
    if len(values) < block_days * 3:
        return {"status": "insufficient", "days": len(values)}
    rng = np.random.default_rng(20260831)
    blocks_needed = math.ceil(len(values) / block_days)
    starts = np.arange(len(values))
    offsets = np.arange(block_days)
    totals = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        path = np.concatenate(
            [values[(start + offsets) % len(values)] for start in selected]
        )[: len(values)]
        totals[sample] = path.sum()
    return {
        "samples": samples,
        "calendar_days": len(values),
        "block_days": block_days,
        "method": "circular_moving_block",
        "net_r_p05": round(float(np.quantile(totals, 0.05)), 6),
        "net_r_p50": round(float(np.quantile(totals, 0.50)), 6),
        "net_r_p95": round(float(np.quantile(totals, 0.95)), 6),
        "probability_positive": round(float(np.mean(totals > 0.0)), 6),
    }


def portfolio_heat(trades: list[CandidateTrade], params: CandidateParams) -> dict[str, Any]:
    """동시 설정 수와 보수적 최대 위험 heat를 계산한다."""

    events: list[tuple[pd.Timestamp, int]] = []
    for trade in trades:
        events.append((pd.Timestamp(trade.entry_time), 1))
        events.append((pd.Timestamp(trade.exit_time), -1))
    concurrent = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        concurrent += delta
        maximum = max(maximum, concurrent)
    return {
        "max_concurrent_positions": maximum,
        "conservative_max_heat_r": float(maximum),
        "risk_scaled_max_heat_percent": round(maximum * params.risk_percent, 4),
        "note": "각 활성 설정이 최종적으로 1R까지 채울 수 있다고 가정한 상한; MTM·마진은 미모델링",
    }


def complete_calendar_years(frames: dict[str, pd.DataFrame]) -> list[int]:
    """모든 검증 심볼이 1월 1일부터 12월 31일까지 존재하는 공통 연도를 구한다."""

    start = max(frame.index.min() for frame in frames.values())
    end = min(frame.index.max() for frame in frames.values())
    years: list[int] = []
    for year in range(start.year, end.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        next_year = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        if start <= year_start and end >= next_year - pd.Timedelta(hours=1):
            years.append(year)
    return years


def yearly_summary(
    trades: list[CandidateTrade],
    params: CandidateParams | None = None,
) -> dict[str, Any]:
    """진입연도별 성과를 반환한다."""

    result: dict[str, Any] = {}
    for year in sorted({pd.Timestamp(trade.entry_time).year for trade in trades}):
        result[str(year)] = summarize(
            [trade for trade in trades if pd.Timestamp(trade.entry_time).year == year],
            params,
        )
    return result


def run_validation(output_dir: Path) -> dict[str, Any]:
    """교정 후보·동일 위험 무추매·비용 스트레스를 함께 평가한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    params = CandidateParams()
    raw_frames = load_market_data(include_external=True)
    funding_frame = load_funding()
    frames: dict[str, pd.DataFrame] = {}
    for symbol, raw_frame in raw_frames.items():
        validate_market_frame(raw_frame, symbol)
        frames[symbol] = validate_funding_coverage(raw_frame, funding_frame[symbol], symbol)

    def replay(test_params: CandidateParams, additions: bool) -> dict[str, list[CandidateTrade]]:
        """모든 심볼을 같은 계약으로 재생한다."""

        replayed: dict[str, list[CandidateTrade]] = {}
        for replay_symbol, replay_frame in frames.items():
            logger.info(
                "검증 시작: %s %d봉 cost=%.1fbp adds=%s",
                replay_symbol,
                len(replay_frame),
                test_params.cost_bps_side,
                additions,
            )
            replayed[replay_symbol] = simulate_symbol(
                replay_frame,
                replay_symbol,
                test_params,
                funding_frame[replay_symbol].dropna(),
                allow_additions=additions,
            )
        return replayed

    candidate_by_symbol = replay(params, additions=True)
    reference_by_symbol = replay(params, additions=False)
    stress_by_cost: dict[str, dict[str, list[CandidateTrade]]] = {
        "8": candidate_by_symbol,
        "10": replay(replace(params, cost_bps_side=10.0), additions=True),
        "12": replay(replace(params, cost_bps_side=12.0), additions=True),
    }
    strict_params = replace(params, cost_bps_side=12.0)
    strict_reference_by_symbol = replay(strict_params, additions=False)
    candidate_trades = [
        trade for symbol_trades in candidate_by_symbol.values() for trade in symbol_trades
    ]
    reference_trades = [
        trade for symbol_trades in reference_by_symbol.values() for trade in symbol_trades
    ]
    results: dict[str, Any] = {
        "classification": "DISCOVERY_ONLY_NOT_PREREGISTERED",
        "params": asdict(params),
        "data": {
            "funding": {
                "source": str(PAIR_FUNDING_PATH.relative_to(ROOT)),
                "sha256": sha256_file(PAIR_FUNDING_PATH),
            }
        },
        "symbols": {},
    }
    for symbol, frame in frames.items():
        source_path = (
            SOL_PATH
            if symbol == "SOL"
            else PAIR_PATH
            if symbol in {"XRP", "DOGE"}
            else CORE_PATH
        )
        candidate = candidate_by_symbol[symbol]
        reference = reference_by_symbol[symbol]
        results["data"][symbol] = {
            "rows": len(frame),
            "first": frame.index[0].isoformat(),
            "last": frame.index[-1].isoformat(),
            "source": str(source_path.relative_to(ROOT)),
            "sha256": sha256_file(source_path),
        }
        results["symbols"][symbol] = {
            "candidate": summarize(candidate, params),
            "reference_no_add_full_risk": summarize(reference, params),
            "yearly": yearly_summary(candidate, params),
        }

    aggregate = summarize(candidate_trades, params)
    reference_aggregate = summarize(reference_trades, params)
    candidate_risk_efficiency = summarize_risk_efficiency(candidate_trades)
    reference_risk_efficiency = summarize_risk_efficiency(reference_trades)
    years = yearly_summary(candidate_trades, params)
    complete_years = complete_calendar_years(frames)
    stress_results: dict[str, Any] = {}
    for cost, by_symbol in stress_by_cost.items():
        cost_params = replace(params, cost_bps_side=float(cost))
        cost_trades = [trade for trades in by_symbol.values() for trade in trades]
        stress_results[cost] = {
            "aggregate": summarize(cost_trades, cost_params),
            "risk_efficiency": summarize_risk_efficiency(cost_trades),
            "symbols": {
                symbol: summarize(trades, cost_params) for symbol, trades in by_symbol.items()
            },
            "yearly": yearly_summary(cost_trades, cost_params),
            "trade_block_bootstrap_20": block_bootstrap(cost_trades, block=20),
            "trade_block_bootstrap_50": block_bootstrap(cost_trades, block=50),
            "calendar_block_bootstrap_28d": calendar_block_bootstrap(cost_trades),
        }

    strict_stress = stress_results["12"]
    strict_aggregate = strict_stress["aggregate"]
    strict_years = strict_stress["yearly"]
    positive_complete_years = sum(
        1
        for year in complete_years
        if strict_years.get(str(year), {}).get("net_r", 0.0) > 0.0
    )
    strict_reference_trades = [
        trade for trades in strict_reference_by_symbol.values() for trade in trades
    ]
    strict_reference_aggregate = summarize(strict_reference_trades, strict_params)
    strict_reference_risk_efficiency = summarize_risk_efficiency(strict_reference_trades)
    strict_positive_symbols = sum(
        1
        for summary in strict_stress["symbols"].values()
        if summary.get("expectancy_r", -math.inf) > 0.0
    )
    calendar_p05 = strict_stress["calendar_block_bootstrap_28d"].get(
        "net_r_p05", -math.inf
    )
    additions_improve = (
        strict_stress["risk_efficiency"].get("expectancy_per_committed_r", -math.inf)
        >= strict_reference_risk_efficiency.get("expectancy_per_committed_r", math.inf)
    )
    add_stage_counts = {
        str(stage): sum(trade.additions >= stage for trade in candidate_trades)
        for stage in range(1, 5)
    }
    additions_validated = all(count >= 30 for count in add_stage_counts.values())
    statistical_conditions = (
        strict_aggregate.get("profit_factor", 0.0) >= 1.20
        and strict_aggregate.get("expectancy_r", -math.inf) > 0.0
        and strict_positive_symbols == len(frames)
        and positive_complete_years >= 4
        and calendar_p05 > 0.0
        and additions_improve
        and additions_validated
    )
    passed = False

    results["aggregate"] = aggregate
    results["reference_no_add_full_risk"] = reference_aggregate
    results["risk_efficiency"] = {
        "candidate": candidate_risk_efficiency,
        "reference_no_add_full_risk": reference_risk_efficiency,
    }
    results["strict_12bp_reference_no_add_full_risk"] = strict_reference_aggregate
    results["strict_12bp_reference_risk_efficiency"] = strict_reference_risk_efficiency
    results["yearly"] = years
    results["complete_calendar_years"] = complete_years
    results["portfolio_heat"] = portfolio_heat(candidate_trades, params)
    results["stress"] = stress_results
    results["gate"] = {
        "status": "PASS" if passed else "FAIL",
        "promotion_capability": "DISABLED_IN_DISCOVERY_RUNNER",
        "statistical_conditions_pass": statistical_conditions,
        "discovery_only": params.discovery_only,
        "required_cost_bps_side": 12.0,
        "required_profit_factor": 1.20,
        "stress_profit_factor": strict_aggregate.get("profit_factor"),
        "stress_expectancy_r": strict_aggregate.get("expectancy_r"),
        "positive_symbols": strict_positive_symbols,
        "required_positive_symbols": len(frames),
        "positive_complete_years": positive_complete_years,
        "required_positive_complete_years": 4,
        "calendar_bootstrap_net_r_p05": calendar_p05,
        "strict_12bp_additions_improve_risk_efficiency": additions_improve,
        "add_stage_minimum_samples": 30,
        "add_stage_sample_counts": add_stage_counts,
        "additions_validated": additions_validated,
        "blocking_reasons": [
            "이 discovery 검증기는 코드상 PASS를 발급할 수 없음; 별도 사전등록 전방 러너 필요",
            "MTM·마진·부분체결·거래소 장애는 이 OHLCV 재생에 포함되지 않음",
        ],
    }
    result_path = output_dir / "latest_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    trade_path = output_dir / "candidate_trades.jsonl"
    trade_path.write_text(
        "".join(json.dumps(asdict(trade), ensure_ascii=False) + "\n" for trade in candidate_trades),
        encoding="utf-8",
    )
    logger.info("결과 저장: %s", result_path)
    return results


def main() -> None:
    """CLI 진입점."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_validation(args.output_dir)
    logger.info("aggregate=%s", result["aggregate"])
    logger.info("gate=%s", result["gate"])


if __name__ == "__main__":
    main()
