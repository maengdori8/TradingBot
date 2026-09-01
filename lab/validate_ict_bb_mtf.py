from __future__ import annotations

"""TradingView ICT·BB MTF 지표의 독립 재현 및 보수적 검증기."""

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "logs" / "validation" / "ict_bb_mtf"
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class IndicatorParams:
    """Pine 기본 입력값을 고정한 검증 파라미터."""

    min_score: int = 7
    setup_window: int = 5
    cooldown_bars: int = 10
    mtf_bias_threshold: float = 0.75
    bb_length: int = 20
    bb_mult: float = 2.0
    rsi_length: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    stoch_length: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3
    stc_fast: int = 23
    stc_slow: int = 50
    stc_cycle: int = 10
    stc_smooth: int = 3
    swing_length: int = 3
    ict_lookback: int = 12
    atr_length: int = 14
    displacement_atr: float = 0.8
    stop_atr: float = 1.2
    rr_target: float = 1.8


@dataclass(frozen=True)
class Trade:
    """보수적 체결 모형으로 완결된 한 거래."""

    signal_time: str
    entry_time: str
    exit_time: str
    direction: str
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str
    holding_bars: int
    gross_return: float
    net_return: float
    r_multiple: float
    score: int
    weighted_bias: float
    volatility: float


def _retry_fetch(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    limit: int = 1000,
    retries: int = 7,
) -> list[list[float]]:
    """네트워크 오류를 지수 백오프로 재시도해 1분봉 한 페이지를 받는다."""

    for attempt in range(retries):
        try:
            return exchange.fetch_ohlcv(
                symbol,
                timeframe="1m",
                since=since_ms,
                limit=limit,
            )
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            if attempt + 1 >= retries:
                raise
            delay = min(8.0, 0.5 * (2**attempt))
            logger.warning(
                "데이터 재시도 %d/%d: %s (%.1f초)",
                attempt + 1,
                retries,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("도달 불가능한 재시도 상태")


def fetch_one_minute(
    exchange: ccxt.Exchange,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Bybit 선형 무기한 1분봉을 시작 포함·종료 미포함으로 전부 수집한다."""

    step_ms = 60_000
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[float]] = []
    page = 0
    while cursor < end_ms:
        batch = _retry_fetch(exchange, symbol, cursor)
        if not batch:
            break
        rows.extend(batch)
        last_ms = int(batch[-1][0])
        next_cursor = last_ms + step_ms
        if next_cursor <= cursor:
            raise RuntimeError(f"페이지 진행 정지: {symbol} cursor={cursor}")
        cursor = next_cursor
        page += 1
        if page % 50 == 0:
            logger.info("%s 수집 중: %d 페이지, %d봉", symbol, page, len(rows))
        if len(batch) < 1000 and last_ms >= end_ms - step_ms:
            break

    frame = pd.DataFrame(
        rows,
        columns=["timestamp", *OHLCV_COLUMNS],
    )
    if frame.empty:
        raise RuntimeError(f"수집된 캔들이 없음: {symbol}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[(frame.index >= start) & (frame.index < end), OHLCV_COLUMNS]
    frame = frame.astype(float)
    expected = pd.date_range(start, end - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
    missing = expected.difference(frame.index)
    if len(missing) > 0:
        raise RuntimeError(
            f"1분봉 결측: {symbol} {len(missing)}개, 첫 결측={missing[0].isoformat()}"
        )
    return frame


def sha256_file(path: Path) -> str:
    """파일의 SHA256 해시를 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_fetch_data(
    symbols: list[str],
    days: int,
    output_dir: Path,
    refresh: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """검증 구간을 고정하고 심볼별 캐시를 재사용하거나 새로 수집한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    start = end - pd.Timedelta(days=days)
    tag = f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"
    exchange = ccxt.bybit(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    exchange.load_markets()
    frames: dict[str, pd.DataFrame] = {}
    manifest: dict[str, Any] = {
        "exchange": "Bybit",
        "market_type": "linear perpetual",
        "timeframe": "1m",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "symbols": {},
    }
    for symbol in symbols:
        safe = symbol.split("/")[0].lower()
        path = output_dir / f"{safe}_1m_{tag}.parquet"
        if path.exists() and not refresh:
            frame = pd.read_parquet(path)
            frame.index = pd.to_datetime(frame.index, utc=True)
            logger.info("캐시 사용: %s (%d봉)", path.name, len(frame))
        else:
            logger.info("Bybit 수집 시작: %s, %s ~ %s", symbol, start, end)
            frame = fetch_one_minute(exchange, symbol, start, end)
            frame.to_parquet(path)
            logger.info("수집 완료: %s (%d봉)", path.name, len(frame))
        frames[symbol] = frame
        manifest["symbols"][symbol] = {
            "rows": len(frame),
            "first": frame.index[0].isoformat(),
            "last": frame.index[-1].isoformat(),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    return frames, manifest


def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """첫 유효값 시드의 Pine ``ta.ema`` 재귀식을 계산한다."""

    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    alpha = 2.0 / (length + 1.0)
    previous = math.nan
    for index, value in enumerate(values):
        if math.isnan(value):
            continue
        previous = value if math.isnan(previous) else alpha * value + (1.0 - alpha) * previous
        result[index] = previous
    return pd.Series(result, index=series.index, dtype=float)


def pine_rma(series: pd.Series, length: int) -> pd.Series:
    """SMA 시드 후 Wilder 재귀를 적용하는 Pine ``ta.rma``를 계산한다."""

    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    seed_values: list[float] = []
    previous = math.nan
    alpha = 1.0 / length
    for index, value in enumerate(values):
        if math.isnan(value):
            continue
        if math.isnan(previous):
            seed_values.append(value)
            if len(seed_values) < length:
                continue
            previous = float(np.mean(seed_values[-length:]))
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        result[index] = previous
    return pd.Series(result, index=series.index, dtype=float)


def pine_rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder 상승·하락 RMA로 Pine RSI를 계산한다."""

    change = close.diff()
    upward = pine_rma(change.clip(lower=0.0), length)
    downward = pine_rma((-change).clip(lower=0.0), length)
    ratio = upward / downward
    result = 100.0 - 100.0 / (1.0 + ratio)
    result = result.mask((downward == 0.0) & (upward > 0.0), 100.0)
    result = result.mask((downward == 0.0) & (upward == 0.0), 50.0)
    return result


def pine_atr(frame: pd.DataFrame, length: int) -> pd.Series:
    """첫 봉의 고저폭을 허용하는 Pine ATR을 계산한다."""

    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return pine_rma(true_range, length)


def crossover(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """Pine ``ta.crossover``와 같은 현재 초과·직전 이하 조건을 반환한다."""

    right_series = pd.Series(float(right), index=left.index) if np.isscalar(right) else right
    result = (left > right_series) & (left.shift(1) <= right_series.shift(1))
    return result.fillna(False)


def crossunder(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """Pine ``ta.crossunder``와 같은 현재 미만·직전 이상 조건을 반환한다."""

    right_series = pd.Series(float(right), index=left.index) if np.isscalar(right) else right
    result = (left < right_series) & (left.shift(1) >= right_series.shift(1))
    return result.fillna(False)


def recent(condition: pd.Series, bars: int) -> pd.Series:
    """현재 봉을 포함해 ``bars``봉 전까지 참이 있었는지 계산한다."""

    return condition.astype(int).rolling(bars + 1, min_periods=1).max().astype(bool)


def stc(source: pd.Series, params: IndicatorParams) -> pd.Series:
    """Pine 지표와 동일한 이중 스토캐스틱 STC 근사값을 계산한다."""

    macd = pine_ema(source, params.stc_fast) - pine_ema(source, params.stc_slow)
    macd_low = macd.rolling(params.stc_cycle, min_periods=params.stc_cycle).min()
    macd_high = macd.rolling(params.stc_cycle, min_periods=params.stc_cycle).max()
    macd_range = macd_high - macd_low
    cycle_k = 100.0 * (macd - macd_low) / macd_range
    cycle_k = cycle_k.mask(macd_range == 0.0, 50.0)
    cycle_d = pine_ema(cycle_k, params.stc_smooth)
    cycle_low = cycle_d.rolling(params.stc_cycle, min_periods=params.stc_cycle).min()
    cycle_high = cycle_d.rolling(params.stc_cycle, min_periods=params.stc_cycle).max()
    cycle_range = cycle_high - cycle_low
    cycle_k2 = 100.0 * (cycle_d - cycle_low) / cycle_range
    cycle_k2 = cycle_k2.mask(cycle_range == 0.0, 50.0)
    return pine_ema(cycle_k2, params.stc_smooth).clip(0.0, 100.0)


def resample_ohlcv(one_minute: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """1분봉을 UTC 경계의 완전한 N분 표준 OHLCV 봉으로 집계한다."""

    if minutes == 1:
        return one_minute.copy()
    rule = f"{minutes}min"
    grouped = one_minute.resample(rule, label="left", closed="left")
    frame = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    return frame.loc[counts == minutes, OHLCV_COLUMNS].dropna()


def compute_metrics(frame: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
    """MTF 대시보드에 쓰는 방향 점수·RSI·STC를 계산한다."""

    close = frame["close"]
    basis = close.rolling(params.bb_length, min_periods=params.bb_length).mean()
    rsi_value = pine_rsi(close, params.rsi_length)
    lowest = frame["low"].rolling(params.stoch_length, min_periods=params.stoch_length).min()
    highest = frame["high"].rolling(params.stoch_length, min_periods=params.stoch_length).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest)
    k_value = raw_k.rolling(params.stoch_smooth_k, min_periods=params.stoch_smooth_k).mean()
    d_value = k_value.rolling(params.stoch_smooth_d, min_periods=params.stoch_smooth_d).mean()
    stc_value = stc(close, params)
    trend_ema = pine_ema(close, 50)
    bull = (
        (close > basis).astype(int)
        + (basis > basis.shift(1)).astype(int)
        + (close > trend_ema).astype(int)
        + (rsi_value > 52.0).astype(int)
        + (k_value > d_value).astype(int)
        + (stc_value > 50.0).astype(int)
    )
    bear = (
        (close < basis).astype(int)
        + (basis < basis.shift(1)).astype(int)
        + (close < trend_ema).astype(int)
        + (rsi_value < 48.0).astype(int)
        + (k_value < d_value).astype(int)
        + (stc_value < 50.0).astype(int)
    )
    return pd.DataFrame(
        {"bias": bull - bear, "rsi": rsi_value, "stc": stc_value},
        index=frame.index,
    )


def confirmed_pivots(
    values: pd.Series,
    left: int,
    right: int,
    high: bool,
) -> pd.Series:
    """후행 ``right``봉이 닫힌 시점에만 확정되는 스윙 피벗을 반환한다."""

    source = values.to_numpy(dtype=float)
    result = np.full(len(source), np.nan)
    for confirmation in range(left + right, len(source)):
        pivot_index = confirmation - right
        candidate = source[pivot_index]
        left_values = source[pivot_index - left : pivot_index]
        right_values = source[pivot_index + 1 : pivot_index + right + 1]
        if high:
            valid = bool(np.all(candidate > left_values) and np.all(candidate >= right_values))
        else:
            valid = bool(np.all(candidate < left_values) and np.all(candidate <= right_values))
        if valid:
            result[confirmation] = candidate
    return pd.Series(result, index=values.index, dtype=float)


def _last_and_previous(event_values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """이벤트 값의 최신값과 직전값을 Pine ``ta.valuewhen``처럼 확장한다."""

    values = event_values.to_numpy(dtype=float)
    latest = np.full(len(values), np.nan)
    previous = np.full(len(values), np.nan)
    last_value = math.nan
    previous_value = math.nan
    for index, value in enumerate(values):
        if not math.isnan(value):
            previous_value = last_value
            last_value = value
        latest[index] = last_value
        previous[index] = previous_value
    return (
        pd.Series(latest, index=event_values.index, dtype=float),
        pd.Series(previous, index=event_values.index, dtype=float),
    )


def _pivot_rsi_values(
    pivot: pd.Series,
    rsi_value: pd.Series,
    right: int,
) -> tuple[pd.Series, pd.Series]:
    """피벗 실제 시점의 RSI를 확정 시점에 기록하고 최근 두 값을 확장한다."""

    event = pd.Series(np.nan, index=pivot.index, dtype=float)
    positions = np.flatnonzero(pivot.notna().to_numpy())
    rsi_array = rsi_value.to_numpy(dtype=float)
    for confirmation in positions:
        event.iloc[confirmation] = rsi_array[confirmation - right]
    return _last_and_previous(event)


def compute_local_features(frame: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
    """Pine 로컬 타임프레임의 ICT·BB·모멘텀 상태를 순방향으로 재현한다."""

    output = frame.copy()
    close = frame["close"]
    basis = close.rolling(params.bb_length, min_periods=params.bb_length).mean()
    deviation = close.rolling(params.bb_length, min_periods=params.bb_length).std(ddof=0)
    upper = basis + params.bb_mult * deviation
    lower = basis - params.bb_mult * deviation
    atr_value = pine_atr(frame, params.atr_length)
    rsi_value = pine_rsi(close, params.rsi_length)
    lowest = frame["low"].rolling(params.stoch_length, min_periods=params.stoch_length).min()
    highest = frame["high"].rolling(params.stoch_length, min_periods=params.stoch_length).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest)
    stoch_k = raw_k.rolling(params.stoch_smooth_k, min_periods=params.stoch_smooth_k).mean()
    stoch_d = stoch_k.rolling(params.stoch_smooth_d, min_periods=params.stoch_smooth_d).mean()
    stc_value = stc(close, params)

    pivot_high = confirmed_pivots(
        frame["high"], params.swing_length, params.swing_length, high=True
    )
    pivot_low = confirmed_pivots(
        frame["low"], params.swing_length, params.swing_length, high=False
    )
    last_high, previous_high = _last_and_previous(pivot_high)
    last_low, previous_low = _last_and_previous(pivot_low)
    last_high_rsi, previous_high_rsi = _pivot_rsi_values(
        pivot_high, rsi_value, params.swing_length
    )
    last_low_rsi, previous_low_rsi = _pivot_rsi_values(
        pivot_low, rsi_value, params.swing_length
    )

    bullish_structure = (last_high > previous_high) & (last_low > previous_low)
    bearish_structure = (last_high < previous_high) & (last_low < previous_low)
    sweep_low = last_low.notna() & (frame["low"] < last_low) & (close > last_low)
    sweep_high = last_high.notna() & (frame["high"] > last_high) & (close < last_high)
    recent_sweep_low = recent(sweep_low, params.setup_window)
    recent_sweep_high = recent(sweep_high, params.setup_window)
    bos_up = crossover(close, last_high)
    bos_down = crossunder(close, last_low)
    choch_up = bos_up & (bearish_structure.shift(1).fillna(False) | recent_sweep_low)
    choch_down = bos_down & (bullish_structure.shift(1).fillna(False) | recent_sweep_high)
    recent_choch_up = recent(choch_up, params.ict_lookback)
    recent_choch_down = recent(choch_down, params.ict_lookback)
    bullish_divergence = (
        pivot_low.notna()
        & (last_low < previous_low)
        & (last_low_rsi > previous_low_rsi)
    )
    bearish_divergence = (
        pivot_high.notna()
        & (last_high > previous_high)
        & (last_high_rsi < previous_high_rsi)
    )

    candle_body = (close - frame["open"]).abs()
    bullish_displacement = (
        (close > frame["open"])
        & (candle_body >= atr_value * params.displacement_atr)
        & (close > frame["high"].shift(1))
    )
    bearish_displacement = (
        (close < frame["open"])
        & (candle_body >= atr_value * params.displacement_atr)
        & (close < frame["low"].shift(1))
    )
    new_bull_fvg = (frame["low"] > frame["high"].shift(2)) & bullish_displacement.shift(1).fillna(False)
    new_bear_fvg = (frame["high"] < frame["low"].shift(2)) & bearish_displacement.shift(1).fillna(False)
    new_bull_ob = bullish_displacement & (close.shift(1) < frame["open"].shift(1))
    new_bear_ob = bearish_displacement & (close.shift(1) > frame["open"].shift(1))

    size = len(frame)
    bull_fvg_top = np.full(size, np.nan)
    bull_fvg_bottom = np.full(size, np.nan)
    bear_fvg_top = np.full(size, np.nan)
    bear_fvg_bottom = np.full(size, np.nan)
    bull_ob_top = np.full(size, np.nan)
    bull_ob_bottom = np.full(size, np.nan)
    bear_ob_top = np.full(size, np.nan)
    bear_ob_bottom = np.full(size, np.nan)
    bull_fvg_tap = np.zeros(size, dtype=bool)
    bear_fvg_tap = np.zeros(size, dtype=bool)
    bull_ob_tap = np.zeros(size, dtype=bool)
    bear_ob_tap = np.zeros(size, dtype=bool)
    current = [math.nan] * 8
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    opens = frame["open"].to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    bull_fvg_events = new_bull_fvg.to_numpy(dtype=bool)
    bear_fvg_events = new_bear_fvg.to_numpy(dtype=bool)
    bull_ob_events = new_bull_ob.to_numpy(dtype=bool)
    bear_ob_events = new_bear_ob.to_numpy(dtype=bool)
    for index in range(size):
        if bull_fvg_events[index]:
            current[0], current[1] = lows[index], highs[index - 2]
        if bear_fvg_events[index]:
            current[2], current[3] = lows[index - 2], highs[index]
        if bull_ob_events[index]:
            current[4], current[5] = opens[index - 1], lows[index - 1]
        if bear_ob_events[index]:
            current[6], current[7] = highs[index - 1], opens[index - 1]

        bull_fvg_tap[index] = (
            not bull_fvg_events[index]
            and not math.isnan(current[0])
            and lows[index] <= current[0]
            and highs[index] >= current[1]
            and closes[index] >= current[1]
        )
        bear_fvg_tap[index] = (
            not bear_fvg_events[index]
            and not math.isnan(current[3])
            and highs[index] >= current[3]
            and lows[index] <= current[2]
            and closes[index] <= current[2]
        )
        bull_ob_tap[index] = (
            not bull_ob_events[index]
            and not math.isnan(current[4])
            and lows[index] <= current[4]
            and highs[index] >= current[5]
            and closes[index] >= current[5]
        )
        bear_ob_tap[index] = (
            not bear_ob_events[index]
            and not math.isnan(current[7])
            and highs[index] >= current[7]
            and lows[index] <= current[6]
            and closes[index] <= current[6]
        )

        if not math.isnan(current[1]) and closes[index] < current[1]:
            current[0], current[1] = math.nan, math.nan
        if not math.isnan(current[2]) and closes[index] > current[2]:
            current[2], current[3] = math.nan, math.nan
        if not math.isnan(current[5]) and closes[index] < current[5]:
            current[4], current[5] = math.nan, math.nan
        if not math.isnan(current[6]) and closes[index] > current[6]:
            current[6], current[7] = math.nan, math.nan

        bull_fvg_top[index], bull_fvg_bottom[index] = current[0], current[1]
        bear_fvg_top[index], bear_fvg_bottom[index] = current[2], current[3]
        bull_ob_top[index], bull_ob_bottom[index] = current[4], current[5]
        bear_ob_top[index], bear_ob_bottom[index] = current[6], current[7]

    index = frame.index
    bull_fvg_tap_s = pd.Series(bull_fvg_tap, index=index)
    bear_fvg_tap_s = pd.Series(bear_fvg_tap, index=index)
    bull_ob_tap_s = pd.Series(bull_ob_tap, index=index)
    bear_ob_tap_s = pd.Series(bear_ob_tap, index=index)
    recent_bull_fvg = recent(new_bull_fvg | bull_fvg_tap_s, params.ict_lookback)
    recent_bear_fvg = recent(new_bear_fvg | bear_fvg_tap_s, params.ict_lookback)
    recent_bull_ob = recent(new_bull_ob | bull_ob_tap_s, params.ict_lookback)
    recent_bear_ob = recent(new_bear_ob | bear_ob_tap_s, params.ict_lookback)

    swing_range = last_high - last_low
    long_ote_low = last_high - swing_range * 0.79
    long_ote_high = last_high - swing_range * 0.62
    short_ote_low = last_low + swing_range * 0.62
    short_ote_high = last_low + swing_range * 0.79
    long_ote = (
        (swing_range > 0.0)
        & bullish_structure
        & (close >= long_ote_low)
        & (close <= long_ote_high)
    )
    short_ote = (
        (swing_range > 0.0)
        & bearish_structure
        & (close >= short_ote_low)
        & (close <= short_ote_high)
    )

    bb_long = (frame["low"] < lower) & (close > lower) & (close > frame["open"])
    bb_short = (frame["high"] > upper) & (close < upper) & (close < frame["open"])
    recent_bb_long = recent(bb_long, params.setup_window)
    recent_bb_short = recent(bb_short, params.setup_window)
    rsi_turn_long = crossover(rsi_value, params.rsi_oversold) | (
        (rsi_value < 45.0) & (rsi_value > rsi_value.shift(1))
    )
    rsi_turn_short = crossunder(rsi_value, params.rsi_overbought) | (
        (rsi_value > 55.0) & (rsi_value < rsi_value.shift(1))
    )
    stoch_turn_long = crossover(stoch_k, stoch_d) & (stoch_k < 30.0)
    stoch_turn_short = crossunder(stoch_k, stoch_d) & (stoch_k > 70.0)
    stc_turn_long = crossover(stc_value, 25.0)
    stc_turn_short = crossunder(stc_value, 75.0)
    momentum_long = recent(
        rsi_turn_long | stoch_turn_long | stc_turn_long,
        params.setup_window,
    )
    momentum_short = recent(
        rsi_turn_short | stoch_turn_short | stc_turn_short,
        params.setup_window,
    )
    location_long = recent_bb_long | recent_sweep_low | long_ote | bull_fvg_tap_s | bull_ob_tap_s
    location_short = recent_bb_short | recent_sweep_high | short_ote | bear_fvg_tap_s | bear_ob_tap_s

    long_score = (
        recent_bb_long.astype(int) * 2
        + recent_sweep_low.astype(int) * 2
        + recent_choch_up.astype(int) * 2
        + recent_bull_fvg.astype(int)
        + recent_bull_ob.astype(int)
        + long_ote.astype(int)
        + recent(bullish_divergence, params.ict_lookback).astype(int)
        + momentum_long.astype(int) * 2
    )
    short_score = (
        recent_bb_short.astype(int) * 2
        + recent_sweep_high.astype(int) * 2
        + recent_choch_down.astype(int) * 2
        + recent_bear_fvg.astype(int)
        + recent_bear_ob.astype(int)
        + short_ote.astype(int)
        + recent(bearish_divergence, params.ict_lookback).astype(int)
        + momentum_short.astype(int) * 2
    )

    output["basis"] = basis
    output["upper"] = upper
    output["lower"] = lower
    output["atr"] = atr_value
    output["rsi"] = rsi_value
    output["stoch_k"] = stoch_k
    output["stoch_d"] = stoch_d
    output["stc"] = stc_value
    output["last_swing_high"] = last_high
    output["last_swing_low"] = last_low
    output["bullish_structure"] = bullish_structure
    output["bearish_structure"] = bearish_structure
    output["long_location"] = location_long
    output["short_location"] = location_short
    output["long_momentum"] = momentum_long
    output["short_momentum"] = momentum_short
    output["recent_bb_long"] = recent_bb_long
    output["recent_bb_short"] = recent_bb_short
    output["long_score"] = long_score
    output["short_score"] = short_score
    output["volatility"] = atr_value / close
    return output


def _lookup(series: pd.Series, timestamps: pd.DatetimeIndex) -> pd.Series:
    """요청 타임프레임의 정확한 봉 시작 시각에서 값을 조회한다."""

    values = series.reindex(timestamps).to_numpy(dtype=float)
    return pd.Series(values, index=timestamps, dtype=float)


def build_feature_frame(
    one_minute: pd.DataFrame,
    chart_minutes: int,
    params: IndicatorParams,
) -> pd.DataFrame:
    """1·5·15분 차트별 Pine MTF 문맥과 로컬 특징을 결합한다."""

    source_frames = {
        1: resample_ohlcv(one_minute, 1),
        5: resample_ohlcv(one_minute, 5),
        15: resample_ohlcv(one_minute, 15),
    }
    metrics = {minutes: compute_metrics(value, params) for minutes, value in source_frames.items()}
    chart = compute_local_features(source_frames[chart_minutes], params)
    timestamps = chart.index

    if chart_minutes == 1:
        one_times = timestamps
        five_times = timestamps.floor("5min") - pd.Timedelta(minutes=5)
        fifteen_times = timestamps.floor("15min") - pd.Timedelta(minutes=15)
        bias1 = _lookup(metrics[1]["bias"], one_times)
        rsi1 = _lookup(metrics[1]["rsi"], one_times)
        stc1 = _lookup(metrics[1]["stc"], one_times)
        bias5 = _lookup(metrics[5]["bias"], five_times)
        rsi5 = _lookup(metrics[5]["rsi"], five_times)
        stc5 = _lookup(metrics[5]["stc"], five_times)
        bias15 = _lookup(metrics[15]["bias"], fifteen_times)
        rsi15 = _lookup(metrics[15]["rsi"], fifteen_times)
        stc15 = _lookup(metrics[15]["stc"], fifteen_times)
    elif chart_minutes == 5:
        one_times = timestamps + pd.Timedelta(minutes=4)
        fifteen_times = timestamps.floor("15min") - pd.Timedelta(minutes=15)
        bias1 = _lookup(metrics[1]["bias"], one_times)
        rsi1 = _lookup(metrics[1]["rsi"], one_times)
        stc1 = _lookup(metrics[1]["stc"], one_times)
        bias5 = metrics[5]["bias"].reindex(timestamps)
        rsi5 = metrics[5]["rsi"].reindex(timestamps)
        stc5 = metrics[5]["stc"].reindex(timestamps)
        bias15 = _lookup(metrics[15]["bias"], fifteen_times)
        rsi15 = _lookup(metrics[15]["rsi"], fifteen_times)
        stc15 = _lookup(metrics[15]["stc"], fifteen_times)
    elif chart_minutes == 15:
        one_times = timestamps + pd.Timedelta(minutes=14)
        five_times = timestamps + pd.Timedelta(minutes=10)
        bias1 = _lookup(metrics[1]["bias"], one_times)
        rsi1 = _lookup(metrics[1]["rsi"], one_times)
        stc1 = _lookup(metrics[1]["stc"], one_times)
        bias5 = _lookup(metrics[5]["bias"], five_times)
        rsi5 = _lookup(metrics[5]["rsi"], five_times)
        stc5 = _lookup(metrics[5]["stc"], five_times)
        bias15 = metrics[15]["bias"].reindex(timestamps)
        rsi15 = metrics[15]["rsi"].reindex(timestamps)
        stc15 = metrics[15]["stc"].reindex(timestamps)
    else:
        raise ValueError(f"지원하지 않는 차트 분: {chart_minutes}")

    chart["bias1"] = bias1.to_numpy()
    chart["bias5"] = bias5.to_numpy()
    chart["bias15"] = bias15.to_numpy()
    chart["rsi1"] = rsi1.to_numpy()
    chart["rsi5"] = rsi5.to_numpy()
    chart["rsi15"] = rsi15.to_numpy()
    chart["stc1"] = stc1.to_numpy()
    chart["stc5"] = stc5.to_numpy()
    chart["stc15"] = stc15.to_numpy()
    chart["weighted_bias"] = chart["bias1"] * 0.20 + chart["bias5"] * 0.35 + chart["bias15"] * 0.45
    return chart


def apply_signal_rules(
    feature_frame: pd.DataFrame,
    chart_minutes: int,
    params: IndicatorParams,
) -> pd.DataFrame:
    """점수·MTF·쿨다운을 적용해 확정봉 신호와 표시 리스크선을 만든다."""

    frame = feature_frame.copy()
    if chart_minutes == 1:
        long_aligned = (frame["bias5"] >= 1) & (frame["bias15"] >= 1)
        short_aligned = (frame["bias5"] <= -1) & (frame["bias15"] <= -1)
    elif chart_minutes == 5:
        long_aligned = frame["bias15"] >= 1
        short_aligned = frame["bias15"] <= -1
    else:
        local_bias = compute_metrics(frame[OHLCV_COLUMNS], params)["bias"]
        long_aligned = local_bias >= 2
        short_aligned = local_bias <= -2

    raw_long = (
        frame["long_location"]
        & frame["long_momentum"]
        & long_aligned
        & (frame["weighted_bias"] >= params.mtf_bias_threshold)
        & (frame["long_score"] >= params.min_score)
        & (frame["long_score"] > frame["short_score"])
    ).fillna(False)
    raw_short = (
        frame["short_location"]
        & frame["short_momentum"]
        & short_aligned
        & (frame["weighted_bias"] <= -params.mtf_bias_threshold)
        & (frame["short_score"] >= params.min_score)
        & (frame["short_score"] > frame["long_score"])
    ).fillna(False)

    long_signal = np.zeros(len(frame), dtype=bool)
    short_signal = np.zeros(len(frame), dtype=bool)
    last_long = -10**9
    last_short = -10**9
    raw_long_values = raw_long.to_numpy(dtype=bool)
    raw_short_values = raw_short.to_numpy(dtype=bool)
    for index in range(len(frame)):
        previous_long = raw_long_values[index - 1] if index > 0 else False
        previous_short = raw_short_values[index - 1] if index > 0 else False
        if (
            raw_long_values[index]
            and not previous_long
            and index - last_long > params.cooldown_bars
        ):
            long_signal[index] = True
            last_long = index
        if (
            raw_short_values[index]
            and not previous_short
            and index - last_short > params.cooldown_bars
        ):
            short_signal[index] = True
            last_short = index

    minimum_long_stop = frame["close"] - frame["atr"] * params.stop_atr
    minimum_short_stop = frame["close"] + frame["atr"] * params.stop_atr
    long_stop = pd.concat([frame["last_swing_low"], minimum_long_stop], axis=1).min(axis=1)
    short_stop = pd.concat([frame["last_swing_high"], minimum_short_stop], axis=1).max(axis=1)
    long_stop = long_stop.fillna(minimum_long_stop)
    short_stop = short_stop.fillna(minimum_short_stop)
    long_target = frame["close"] + (frame["close"] - long_stop) * params.rr_target
    short_target = frame["close"] - (short_stop - frame["close"]) * params.rr_target

    frame["raw_long"] = raw_long
    frame["raw_short"] = raw_short
    frame["long_signal"] = long_signal
    frame["short_signal"] = short_signal
    frame["long_stop"] = long_stop
    frame["short_stop"] = short_stop
    frame["long_target"] = long_target
    frame["short_target"] = short_target
    return frame


def simulate_trades(
    signal_frame: pd.DataFrame,
    cost_bps_side: float,
) -> list[Trade]:
    """다음 봉 시가 진입·동시 터치 손절 우선·갭 손절 악화로 거래를 재생한다."""

    trades: list[Trade] = []
    index = 0
    size = len(signal_frame)
    cost = cost_bps_side / 10_000.0
    while index < size - 1:
        row = signal_frame.iloc[index]
        direction = 1 if bool(row["long_signal"]) else -1 if bool(row["short_signal"]) else 0
        if direction == 0:
            index += 1
            continue
        entry_index = index + 1
        entry = float(signal_frame.iloc[entry_index]["open"])
        stop = float(row["long_stop"] if direction > 0 else row["short_stop"])
        target = float(row["long_target"] if direction > 0 else row["short_target"])
        valid = (
            math.isfinite(entry)
            and math.isfinite(stop)
            and math.isfinite(target)
            and ((stop < entry < target) if direction > 0 else (target < entry < stop))
        )
        if not valid:
            index += 1
            continue
        exit_price = float(signal_frame.iloc[-1]["close"])
        exit_index = size - 1
        exit_reason = "end"
        for cursor in range(entry_index, size):
            candle = signal_frame.iloc[cursor]
            candle_open = float(candle["open"])
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            if direction > 0:
                stop_hit = candle_low <= stop
                target_hit = candle_high >= target
                if stop_hit:
                    exit_price = min(stop, candle_open)
                    exit_index = cursor
                    exit_reason = "stop" if not target_hit else "both_stop_first"
                    break
                if target_hit:
                    exit_price = target
                    exit_index = cursor
                    exit_reason = "target"
                    break
            else:
                stop_hit = candle_high >= stop
                target_hit = candle_low <= target
                if stop_hit:
                    exit_price = max(stop, candle_open)
                    exit_index = cursor
                    exit_reason = "stop" if not target_hit else "both_stop_first"
                    break
                if target_hit:
                    exit_price = target
                    exit_index = cursor
                    exit_reason = "target"
                    break
        gross_return = direction * (exit_price - entry) / entry
        net_return = gross_return - cost * (1.0 + exit_price / entry)
        risk_fraction = abs(entry - stop) / entry
        r_multiple = net_return / risk_fraction if risk_fraction > 0.0 else math.nan
        score = int(row["long_score"] if direction > 0 else row["short_score"])
        trades.append(
            Trade(
                signal_time=signal_frame.index[index].isoformat(),
                entry_time=signal_frame.index[entry_index].isoformat(),
                exit_time=signal_frame.index[exit_index].isoformat(),
                direction="long" if direction > 0 else "short",
                entry=entry,
                stop=stop,
                target=target,
                exit=exit_price,
                exit_reason=exit_reason,
                holding_bars=exit_index - entry_index + 1,
                gross_return=gross_return,
                net_return=net_return,
                r_multiple=r_multiple,
                score=score,
                weighted_bias=float(row["weighted_bias"]),
                volatility=float(row["volatility"]),
            )
        )
        index = exit_index + 1
    return trades


def max_consecutive_losses(values: Iterable[float]) -> int:
    """연속 음수 거래의 최대 길이를 계산한다."""

    longest = 0
    current = 0
    for value in values:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def metrics(trades: list[Trade]) -> dict[str, Any]:
    """R 단위 기대값·PF·낙폭·연속손실을 포함한 성과를 요약한다."""

    if not trades:
        return {
            "trades": 0,
            "win_rate": None,
            "net_r": 0.0,
            "expectancy_r": None,
            "profit_factor": None,
            "max_drawdown_r": 0.0,
            "max_consecutive_losses": 0,
            "median_holding_bars": None,
            "expectancy_net_bps": None,
            "target_hit_rate": None,
            "both_touched_rate": None,
        }
    values = np.array([trade.r_multiple for trade in trades], dtype=float)
    values = values[np.isfinite(values)]
    gains = values[values > 0.0].sum()
    losses = -values[values < 0.0].sum()
    equity = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": int(len(values)),
        "win_rate": round(float(np.mean(values > 0.0)), 6),
        "net_r": round(float(values.sum()), 6),
        "expectancy_r": round(float(values.mean()), 6),
        "profit_factor": round(float(gains / losses), 6) if losses > 0.0 else None,
        "max_drawdown_r": round(float(drawdown.max()), 6),
        "max_consecutive_losses": max_consecutive_losses(values),
        "median_holding_bars": round(float(np.median([trade.holding_bars for trade in trades])), 2),
        "expectancy_net_bps": round(float(np.mean([trade.net_return for trade in trades]) * 10_000.0), 6),
        "target_hit_rate": round(float(np.mean([trade.exit_reason == "target" for trade in trades])), 6),
        "both_touched_rate": round(
            float(np.mean([trade.exit_reason == "both_stop_first" for trade in trades])),
            6,
        ),
    }


def split_metrics(
    trades: list[Trade],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    """시간 순서 50%·25%·25%의 인샘플·검증·최종 홀드아웃을 평가한다."""

    first_cut = start + (end - start) * 0.50
    second_cut = start + (end - start) * 0.75
    groups: dict[str, list[Trade]] = {"in_sample_50": [], "validation_25": [], "holdout_25": []}
    for trade in trades:
        timestamp = pd.Timestamp(trade.signal_time)
        if timestamp < first_cut:
            groups["in_sample_50"].append(trade)
        elif timestamp < second_cut:
            groups["validation_25"].append(trade)
        else:
            groups["holdout_25"].append(trade)
    return {name: metrics(values) for name, values in groups.items()}


def rolling_block_metrics(
    trades: list[Trade],
    start: pd.Timestamp,
    end: pd.Timestamp,
    blocks: int = 6,
) -> list[dict[str, Any]]:
    """전 기간을 동일 길이 순차 블록으로 나눠 레짐 지속성을 확인한다."""

    width = (end - start) / blocks
    result: list[dict[str, Any]] = []
    for block in range(blocks):
        left = start + width * block
        right = end if block + 1 == blocks else start + width * (block + 1)
        selected = [
            trade
            for trade in trades
            if left <= pd.Timestamp(trade.signal_time) < right
        ]
        result.append(
            {
                "block": block + 1,
                "start": left.isoformat(),
                "end": right.isoformat(),
                **metrics(selected),
            }
        )
    return result


def bootstrap_metrics(
    trades: list[Trade],
    seed: int = 20260831,
    samples: int = 20_000,
) -> dict[str, Any]:
    """거래 R을 재표집해 기대값과 낙폭 불확실성을 추정한다."""

    values = np.array([trade.r_multiple for trade in trades], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return {"samples": samples, "status": "insufficient_trades"}
    rng = np.random.default_rng(seed)
    positive_final = 0
    mean_samples = np.empty(samples, dtype=float)
    drawdown_samples = np.empty(samples, dtype=float)
    chunk = min(500, samples)
    processed = 0
    while processed < samples:
        count = min(chunk, samples - processed)
        sampled = rng.choice(values, size=(count, len(values)), replace=True)
        paths = np.cumsum(sampled, axis=1)
        paths = np.concatenate([np.zeros((count, 1)), paths], axis=1)
        drawdowns = np.maximum.accumulate(paths, axis=1) - paths
        mean_samples[processed : processed + count] = sampled.mean(axis=1)
        drawdown_samples[processed : processed + count] = drawdowns.max(axis=1)
        positive_final += int(np.sum(paths[:, -1] > 0.0))
        processed += count
    return {
        "samples": samples,
        "probability_positive_final": round(positive_final / samples, 6),
        "expectancy_r_p05": round(float(np.quantile(mean_samples, 0.05)), 6),
        "expectancy_r_p50": round(float(np.quantile(mean_samples, 0.50)), 6),
        "expectancy_r_p95": round(float(np.quantile(mean_samples, 0.95)), 6),
        "max_drawdown_r_p50": round(float(np.quantile(drawdown_samples, 0.50)), 6),
        "max_drawdown_r_p95": round(float(np.quantile(drawdown_samples, 0.95)), 6),
    }


def moving_block_bootstrap_metrics(
    trades: list[Trade],
    seed: int = 20260831,
    samples: int = 20_000,
    block_lengths: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    """연속 거래 묶음을 보존한 순환 이동 블록 부트스트랩을 수행한다."""

    values = np.array([trade.r_multiple for trade in trades], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(block_lengths) * 2:
        return {"samples": samples, "status": "insufficient_trades"}
    results: dict[str, Any] = {"samples": samples, "blocks": {}}
    for block_length in block_lengths:
        rng = np.random.default_rng(seed + block_length)
        positive_final = 0
        expectancy_samples = np.empty(samples, dtype=float)
        drawdown_samples = np.empty(samples, dtype=float)
        blocks_needed = math.ceil(len(values) / block_length)
        offsets = np.arange(block_length, dtype=int)
        processed = 0
        chunk_size = min(250, samples)
        while processed < samples:
            count = min(chunk_size, samples - processed)
            starts = rng.integers(0, len(values), size=(count, blocks_needed, 1))
            indices = (starts + offsets.reshape(1, 1, -1)) % len(values)
            indices = indices.reshape(count, -1)[:, : len(values)]
            sampled = values[indices]
            paths = np.cumsum(sampled, axis=1)
            paths = np.concatenate([np.zeros((count, 1)), paths], axis=1)
            drawdowns = np.maximum.accumulate(paths, axis=1) - paths
            expectancy_samples[processed : processed + count] = sampled.mean(axis=1)
            drawdown_samples[processed : processed + count] = drawdowns.max(axis=1)
            positive_final += int(np.sum(paths[:, -1] > 0.0))
            processed += count
        results["blocks"][str(block_length)] = {
            "probability_positive_final": round(positive_final / samples, 6),
            "expectancy_r_p05": round(float(np.quantile(expectancy_samples, 0.05)), 6),
            "expectancy_r_p50": round(float(np.quantile(expectancy_samples, 0.50)), 6),
            "max_drawdown_r_p95": round(float(np.quantile(drawdown_samples, 0.95)), 6),
        }
    return results


def subgroup_metrics(trades: list[Trade]) -> dict[str, Any]:
    """방향과 진입 시점 변동성 삼분위별 취약 구간을 요약한다."""

    result = {
        "long": metrics([trade for trade in trades if trade.direction == "long"]),
        "short": metrics([trade for trade in trades if trade.direction == "short"]),
    }
    if trades:
        volatilities = np.array([trade.volatility for trade in trades], dtype=float)
        finite = volatilities[np.isfinite(volatilities)]
        if len(finite) > 2:
            low_cut, high_cut = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
            result["vol_low"] = metrics([trade for trade in trades if trade.volatility <= low_cut])
            result["vol_mid"] = metrics(
                [trade for trade in trades if low_cut < trade.volatility <= high_cut]
            )
            result["vol_high"] = metrics([trade for trade in trades if trade.volatility > high_cut])
    return result


def signal_risk_diagnostics(signal_frame: pd.DataFrame) -> dict[str, Any]:
    """구조 손절이 ATR 대비 비정상적으로 먼지와 신호 밀도를 진단한다."""

    rows: list[tuple[float, float]] = []
    for _, row in signal_frame.loc[signal_frame["long_signal"] | signal_frame["short_signal"]].iterrows():
        if bool(row["long_signal"]):
            distance = float(row["close"] - row["long_stop"])
        else:
            distance = float(row["short_stop"] - row["close"])
        if float(row["atr"]) > 0.0:
            rows.append((distance / float(row["atr"]), distance / float(row["close"])))
    if not rows:
        return {"signals": 0}
    atr_distances = np.array([value[0] for value in rows])
    price_distances = np.array([value[1] for value in rows])
    return {
        "signals": len(rows),
        "stop_atr_p50": round(float(np.quantile(atr_distances, 0.50)), 4),
        "stop_atr_p95": round(float(np.quantile(atr_distances, 0.95)), 4),
        "stop_atr_max": round(float(np.max(atr_distances)), 4),
        "stop_over_3atr_rate": round(float(np.mean(atr_distances > 3.0)), 6),
        "stop_price_pct_p95": round(float(np.quantile(price_distances, 0.95)), 6),
    }


def forward_prediction_diagnostics(
    signal_frame: pd.DataFrame,
    chart_minutes: int,
    cost_bps_side: float = 5.5,
) -> dict[str, Any]:
    """모든 신호의 고정 전방 구간 방향 적중률과 비용 후 평균을 측정한다."""

    directions = np.where(
        signal_frame["long_signal"].to_numpy(dtype=bool),
        1.0,
        np.where(signal_frame["short_signal"].to_numpy(dtype=bool), -1.0, 0.0),
    )
    opens = signal_frame["open"].to_numpy(dtype=float)
    closes = signal_frame["close"].to_numpy(dtype=float)
    signal_positions = np.flatnonzero(directions != 0.0)
    result: dict[str, Any] = {}
    round_trip_cost_bps = cost_bps_side * 2.0
    for horizon in (1, 3, 6, 12):
        returns: list[float] = []
        for position in signal_positions:
            entry_position = position + 1
            exit_position = position + horizon
            if entry_position >= len(opens) or exit_position >= len(closes):
                continue
            entry = opens[entry_position]
            exit_price = closes[exit_position]
            directional_return = directions[position] * (exit_price - entry) / entry
            returns.append(float(directional_return * 10_000.0))
        values = np.array(returns, dtype=float)
        label = f"{horizon}_bars_{horizon * chart_minutes}m"
        result[label] = {
            "signals": len(values),
            "gross_direction_hit_rate": round(float(np.mean(values > 0.0)), 6) if len(values) else None,
            "gross_mean_bps": round(float(values.mean()), 6) if len(values) else None,
            "gross_median_bps": round(float(np.median(values)), 6) if len(values) else None,
            "official_taker_net_mean_bps": round(float(values.mean() - round_trip_cost_bps), 6)
            if len(values)
            else None,
            "official_taker_net_positive_rate": round(
                float(np.mean(values > round_trip_cost_bps)),
                6,
            )
            if len(values)
            else None,
        }
    return result


def prefix_stability_audit(
    one_minute: pd.DataFrame,
    chart_minutes: int,
    params: IndicatorParams,
) -> dict[str, Any]:
    """여러 과거 절단점에서 전체 계산과 접두 계산의 신호가 같은지 검사한다."""

    full = apply_signal_rules(build_feature_frame(one_minute, chart_minutes, params), chart_minutes, params)
    chart_index = full.index
    fractions = (0.55, 0.70, 0.85)
    mismatches = 0
    compared = 0
    details: list[dict[str, Any]] = []
    for fraction in fractions:
        position = int(len(chart_index) * fraction)
        cut = chart_index[position]
        source_end = cut + pd.Timedelta(minutes=chart_minutes)
        prefix_data = one_minute.loc[one_minute.index < source_end]
        prefix = apply_signal_rules(
            build_feature_frame(prefix_data, chart_minutes, params),
            chart_minutes,
            params,
        )
        common = prefix.index.intersection(full.index)
        full_values = full.loc[common, ["long_signal", "short_signal"]].to_numpy(dtype=bool)
        prefix_values = prefix.loc[common, ["long_signal", "short_signal"]].to_numpy(dtype=bool)
        mismatch = int(np.sum(full_values != prefix_values))
        mismatches += mismatch
        compared += int(full_values.size)
        details.append({"cut": cut.isoformat(), "compared_cells": int(full_values.size), "mismatches": mismatch})
    return {
        "chart_minutes": chart_minutes,
        "compared_cells": compared,
        "mismatches": mismatches,
        "passed": mismatches == 0,
        "cuts": details,
    }


def sensitivity_grid(
    feature_frame: pd.DataFrame,
    chart_minutes: int,
    params: IndicatorParams,
    cost_bps_side: float,
) -> list[dict[str, Any]]:
    """점수·MTF 임계값의 사전 고정 격자 전체를 평가한다."""

    rows: list[dict[str, Any]] = []
    for min_score in (6, 7, 8, 9):
        for threshold in (0.50, 0.75, 1.00, 1.25):
            candidate = replace(
                params,
                min_score=min_score,
                mtf_bias_threshold=threshold,
            )
            signaled = apply_signal_rules(feature_frame, chart_minutes, candidate)
            result = metrics(simulate_trades(signaled, cost_bps_side))
            rows.append(
                {
                    "min_score": min_score,
                    "mtf_bias_threshold": threshold,
                    **result,
                }
            )
    return rows


def summarize_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """민감도 격자에서 양의 기대값 비율과 중앙·최악 PF를 계산한다."""

    eligible = [row for row in rows if row["trades"] >= 30 and row["expectancy_r"] is not None]
    if not eligible:
        return {"eligible_cells": 0}
    pfs = [float(row["profit_factor"]) for row in eligible if row["profit_factor"] is not None]
    return {
        "eligible_cells": len(eligible),
        "positive_expectancy_rate": round(
            float(np.mean([float(row["expectancy_r"]) > 0.0 for row in eligible])),
            6,
        ),
        "profit_factor_median": round(float(np.median(pfs)), 6) if pfs else None,
        "profit_factor_min": round(float(np.min(pfs)), 6) if pfs else None,
        "trades_median": round(float(np.median([int(row["trades"]) for row in eligible])), 2),
    }


def run_validation(
    data: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    params: IndicatorParams,
    output_dir: Path,
) -> dict[str, Any]:
    """전 심볼·전 타임프레임의 기준선과 강건성 검증을 실행한다."""

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "params": asdict(params),
        "execution_model": {
            "entry": "next_bar_open",
            "same_bar_stop_and_target": "stop_first",
            "stop_gap": "worse_open",
            "cost_bps_side": [0.0, 5.5, 8.0, 12.0],
            "funding": "excluded; short holding-period strategy",
        },
        "cells": {},
        "prefix_stability": [],
    }
    for symbol, one_minute in data.items():
        symbol_results: dict[str, Any] = {}
        for chart_minutes in (1, 5, 15):
            logger.info("특징 계산: %s %d분", symbol, chart_minutes)
            feature_frame = build_feature_frame(one_minute, chart_minutes, params)
            signaled = apply_signal_rules(feature_frame, chart_minutes, params)
            base_trades = simulate_trades(signaled, 8.0)
            costs: dict[str, Any] = {}
            for cost in (0.0, 5.5, 8.0, 12.0):
                cost_trades = simulate_trades(signaled, cost)
                costs[f"{cost:g}bp"] = metrics(cost_trades)
            start = signaled.index[0]
            end = signaled.index[-1] + pd.Timedelta(minutes=chart_minutes)
            grid = sensitivity_grid(feature_frame, chart_minutes, params, 8.0)
            official_trades = simulate_trades(signaled, 5.5)
            symbol_results[f"{chart_minutes}m"] = {
                "period": {"start": start.isoformat(), "end": end.isoformat()},
                "baseline_8bp": metrics(base_trades),
                "cost_stress": costs,
                "chronological_split": split_metrics(base_trades, start, end),
                "rolling_six_blocks": rolling_block_metrics(base_trades, start, end),
                "subgroups": subgroup_metrics(base_trades),
                "bootstrap_20000": bootstrap_metrics(base_trades),
                "moving_block_bootstrap_20000": moving_block_bootstrap_metrics(base_trades),
                "official_taker_5_5bp_detail": {
                    "metrics": metrics(official_trades),
                    "chronological_split": split_metrics(official_trades, start, end),
                    "bootstrap_20000": bootstrap_metrics(official_trades),
                    "moving_block_bootstrap_20000": moving_block_bootstrap_metrics(
                        official_trades
                    ),
                },
                "risk_diagnostics": signal_risk_diagnostics(signaled),
                "forward_prediction": forward_prediction_diagnostics(
                    signaled,
                    chart_minutes,
                ),
                "sensitivity_summary": summarize_grid(grid),
                "sensitivity_grid": grid,
                "trades": [asdict(trade) for trade in base_trades],
            }
            logger.info(
                "%s %d분 완료: %s",
                symbol,
                chart_minutes,
                symbol_results[f"{chart_minutes}m"]["baseline_8bp"],
            )
        results["cells"][symbol] = symbol_results

    primary_symbol = next(iter(data))
    for chart_minutes in (1, 5, 15):
        logger.info("접두 안정성 감사: %s %d분", primary_symbol, chart_minutes)
        results["prefix_stability"].append(
            prefix_stability_audit(data[primary_symbol], chart_minutes, params)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("검증 결과 저장: %s", output_path)
    return results


def parse_args() -> argparse.Namespace:
    """명령행 인자를 해석한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180, help="검증 일수")
    parser.add_argument(
        "--symbols",
        default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT",
        help="쉼표 구분 ccxt 무기한 심볼",
    )
    parser.add_argument("--refresh", action="store_true", help="동일 구간 캐시 무시")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """데이터 수집부터 독립 검증 결과 저장까지 실행한다."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.days < 30:
        raise ValueError("최소 30일 이상을 검증해야 합니다.")
    symbols = [value.strip() for value in args.symbols.split(",") if value.strip()]
    data, manifest = load_or_fetch_data(symbols, args.days, args.output_dir, args.refresh)
    run_validation(data, manifest, IndicatorParams(), args.output_dir)


if __name__ == "__main__":
    main()
