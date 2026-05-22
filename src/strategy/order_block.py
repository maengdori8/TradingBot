"""
Order Block (OB) 탐지 — 강한 이동 직전 마지막 반대 캔들
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd


@dataclass
class OrderBlock:
    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    candle_index: int
    strength: float  # 이후 이동 폭


def detect_order_blocks(df: pd.DataFrame, threshold: float = 0.003, lookback: int = 3) -> list[OrderBlock]:
    """
    OB 탐지: 강한 이동 직전 마지막 반대색 캔들.

    Bullish OB: 강한 상승 직전 마지막 하락 캔들
    Bearish OB: 강한 하락 직전 마지막 상승 캔들

    Args:
        df: OHLCV DataFrame
        threshold: 강한 이동으로 판단할 최소 변화율
        lookback: 이동 확인 범위

    Returns:
        OB 리스트
    """
    ob_list: list[OrderBlock] = []
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(1, len(df) - lookback):
        # 이후 이동 폭 계산
        future_high = max(highs[i + 1: i + 1 + lookback])
        future_low = min(lows[i + 1: i + 1 + lookback])
        body = abs(closes[i] - opens[i])
        mid = (closes[i] + opens[i]) / 2

        # Bullish OB: 현재 캔들이 하락봉이고 이후 강한 상승
        if closes[i] < opens[i]:  # 하락봉
            strength = (future_high - highs[i]) / highs[i]
            if strength >= threshold:
                ob_list.append(OrderBlock(
                    type="bullish",
                    top=opens[i],    # 하락봉의 시가 (위)
                    bottom=closes[i],  # 하락봉의 종가 (아래)
                    candle_index=i,
                    strength=strength,
                ))

        # Bearish OB: 현재 캔들이 상승봉이고 이후 강한 하락
        elif closes[i] > opens[i]:  # 상승봉
            strength = (lows[i] - future_low) / lows[i]
            if strength >= threshold:
                ob_list.append(OrderBlock(
                    type="bearish",
                    top=closes[i],   # 상승봉의 종가 (위)
                    bottom=opens[i],  # 상승봉의 시가 (아래)
                    candle_index=i,
                    strength=strength,
                ))

    return ob_list


def is_price_in_ob(price: float, ob_list: list[OrderBlock]) -> list[OrderBlock]:
    """
    현재 가격이 속한 OB 반환.

    Args:
        price: 현재 가격
        ob_list: OB 리스트

    Returns:
        가격이 속한 OB 리스트
    """
    return [ob for ob in ob_list if ob.bottom <= price <= ob.top]
