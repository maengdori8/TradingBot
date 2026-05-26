from __future__ import annotations

"""
Fair Value Gap (FVG) 탐지 — 3봉 패턴
"""

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FVG:
    """Fair Value Gap 데이터 클래스."""

    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    candle_index: int
    filled: bool = False


def detect_fvg(df: pd.DataFrame, min_gap_pct: float | None = None) -> list[FVG]:
    """3봉 패턴으로 FVG 를 탐지한다.

    Bullish FVG: candle[i-2].high < candle[i].low  (갭 상승)
    Bearish FVG: candle[i-2].low  > candle[i].high (갭 하락)

    Args:
        df: OHLCV DataFrame
        min_gap_pct: 최소 갭 크기 비율. None 이면 strategy_params.yaml 값 사용.

    Returns:
        미체워진 FVG 리스트
    """
    if min_gap_pct is None:
        from . import load_strategy_params
        min_gap_pct = load_strategy_params()["fvg"]["min_gap_pct"]

    fvg_list: list[FVG] = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(2, len(df)):
        mid_price = closes[i - 1]

        # Bullish FVG
        gap_top = lows[i]
        gap_bottom = highs[i - 2]
        if gap_top > gap_bottom and (gap_top - gap_bottom) / mid_price >= min_gap_pct:
            fvg_list.append(FVG(type="bullish", top=gap_top, bottom=gap_bottom, candle_index=i))

        # Bearish FVG
        gap_top2 = lows[i - 2]
        gap_bottom2 = highs[i]
        if gap_top2 > gap_bottom2 and (gap_top2 - gap_bottom2) / mid_price >= min_gap_pct:
            fvg_list.append(FVG(type="bearish", top=gap_top2, bottom=gap_bottom2, candle_index=i))

    logger.debug("FVG 탐지 완료: %d개 발견 (min_gap_pct=%.4f)", len(fvg_list), min_gap_pct)
    return fvg_list


def is_price_in_fvg(price: float, fvg_list: list[FVG]) -> list[FVG]:
    """현재 가격이 속한 미체워진 FVG 를 반환한다.

    Args:
        price: 현재 가격
        fvg_list: FVG 리스트

    Returns:
        가격이 속한 FVG 리스트 (비어있으면 [])
    """
    return [f for f in fvg_list if not f.filled and f.bottom <= price <= f.top]


def update_fvg_fills(fvg_list: list[FVG], current_high: float, current_low: float) -> list[FVG]:
    """현재 캔들로 체워진 FVG 를 표시하고 미체워진 것만 반환한다.

    Args:
        fvg_list: 기존 FVG 리스트
        current_high: 현재 캔들 고점
        current_low: 현재 캔들 저점

    Returns:
        업데이트된 FVG 리스트 (체워진 항목 제거)
    """
    active: list[FVG] = []
    for f in fvg_list:
        if f.type == "bullish" and current_low <= f.top:
            f.filled = True
        elif f.type == "bearish" and current_high >= f.bottom:
            f.filled = True
        if not f.filled:
            active.append(f)
    return active
