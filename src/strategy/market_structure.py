from __future__ import annotations

"""
시장 구조 분석 — BOS / CHoCH 탐지
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_swing_points(df: pd.DataFrame, lookback: int | None = None) -> pd.DataFrame:
    """스윙 고점/저점을 탐지한다.

    Args:
        df: OHLCV DataFrame
        lookback: 스윙 포인트 탐지 범위. None 이면 yaml 값 사용.

    Returns:
        swing_high, swing_low 컬럼이 추가된 DataFrame
    """
    if lookback is None:
        from . import load_strategy_params
        lookback = load_strategy_params()["market_structure"]["swing_lookback"]

    result = df.copy()
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        left_h  = highs[i - lookback: i]
        right_h = highs[i + 1: i + lookback + 1]
        left_l  = lows[i - lookback: i]
        right_l = lows[i + 1: i + lookback + 1]

        if highs[i] > left_h.max() and highs[i] > right_h.max():
            swing_high[i] = True
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            swing_low[i] = True

    result["swing_high"] = swing_high
    result["swing_low"] = swing_low

    sh_count = int(swing_high.sum())
    sl_count = int(swing_low.sum())
    logger.debug("스윙 포인트 탐지 완료: 고점 %d개, 저점 %d개 (lookback=%d)", sh_count, sl_count, lookback)
    return result


def detect_bos(df: pd.DataFrame, lookback: int | None = None) -> Literal["bullish", "bearish"] | None:
    """Break of Structure 를 탐지한다.

    Args:
        df: OHLCV DataFrame
        lookback: 스윙 포인트 탐지 범위. None 이면 yaml 값 사용.

    Returns:
        'bullish' | 'bearish' | None
    """
    if lookback is None:
        from . import load_strategy_params
        lookback = load_strategy_params()["market_structure"]["swing_lookback"]

    df_sw = detect_swing_points(df, lookback)
    sh_idx = df_sw.index[df_sw["swing_high"]]
    sl_idx = df_sw.index[df_sw["swing_low"]]

    if len(sh_idx) < 2 or len(sl_idx) < 2:
        logger.debug("BOS 판단 불가: 스윙 포인트 부족 (고점 %d개, 저점 %d개)", len(sh_idx), len(sl_idx))
        return None

    last_close = df["close"].iloc[-1]
    prev_swing_high = df_sw.loc[sh_idx[-1], "high"]
    prev_swing_low  = df_sw.loc[sl_idx[-1], "low"]

    if last_close > prev_swing_high:
        logger.debug("BOS 감지: bullish (종가 %.4f > 직전 스윙 고점 %.4f)", last_close, prev_swing_high)
        return "bullish"
    if last_close < prev_swing_low:
        logger.debug("BOS 감지: bearish (종가 %.4f < 직전 스윙 저점 %.4f)", last_close, prev_swing_low)
        return "bearish"

    logger.debug("BOS 미감지: 종가 %.4f 이 스윙 범위 [%.4f, %.4f] 내부", last_close, prev_swing_low, prev_swing_high)
    return None


def detect_choch(df: pd.DataFrame, lookback: int | None = None) -> Literal["bullish", "bearish"] | None:
    """Change of Character 를 탐지한다.

    Args:
        df: OHLCV DataFrame
        lookback: 스윙 포인트 탐지 범위. None 이면 yaml 값 사용.

    Returns:
        'bullish' | 'bearish' | None
    """
    if lookback is None:
        from . import load_strategy_params
        lookback = load_strategy_params()["market_structure"]["swing_lookback"]

    df_sw = detect_swing_points(df, lookback)
    sh_idx = df_sw.index[df_sw["swing_high"]]
    sl_idx = df_sw.index[df_sw["swing_low"]]

    if len(sh_idx) < 3 or len(sl_idx) < 3:
        logger.debug("CHoCH 판단 불가: 스윙 포인트 부족 (고점 %d개, 저점 %d개)", len(sh_idx), len(sl_idx))
        return None

    highs_vals = df_sw.loc[sh_idx, "high"].values
    lows_vals  = df_sw.loc[sl_idx, "low"].values

    is_uptrend   = highs_vals[-1] > highs_vals[-2]
    is_downtrend = lows_vals[-1]  < lows_vals[-2]

    last_close = df["close"].iloc[-1]

    if is_uptrend and last_close < lows_vals[-2]:
        logger.debug(
            "CHoCH 감지: bearish (상승 추세 중 종가 %.4f < 직전 스윙 저점 %.4f)",
            last_close, lows_vals[-2],
        )
        return "bearish"
    if is_downtrend and last_close > highs_vals[-2]:
        logger.debug(
            "CHoCH 감지: bullish (하락 추세 중 종가 %.4f > 직전 스윙 고점 %.4f)",
            last_close, highs_vals[-2],
        )
        return "bullish"

    logger.debug("CHoCH 미감지: 추세 전환 조건 미충족")
    return None


def detect_htf_trend(
    df: pd.DataFrame,
    ema_period: int = 50,
    slope_lookback: int = 10,
) -> str:
    """상위 타임프레임(예: BTC 4H) 추세 레짐을 판정한다.

    신호연구(2025-12~2026-06, 36심볼)에서 사용한 정의와 동일:
    종가가 EMA 위 + EMA 기울기 상승 → "bull",
    종가가 EMA 아래 + EMA 기울기 하락 → "bear", 그 외 → "flat".

    Args:
        df: OHLCV DataFrame (close 컬럼 필요)
        ema_period: EMA 기간 (연구 사용값 50)
        slope_lookback: 기울기 판정 봉 수 (연구 사용값 10)

    Returns:
        "bull" | "bear" | "flat"
    """
    if len(df) < ema_period + slope_lookback:
        return "flat"
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    close = float(df["close"].iloc[-1])
    ema_now = float(ema.iloc[-1])
    slope = float(ema.iloc[-1] - ema.iloc[-1 - slope_lookback])
    if close > ema_now and slope > 0:
        return "bull"
    if close < ema_now and slope < 0:
        return "bear"
    return "flat"
