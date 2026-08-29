"""
Optimal Trade Entry (OTE) — 피보나치 되돌림 존
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OTEZone:
    """OTE 존 데이터 클래스."""

    high: float
    low: float
    ote_low: float   # fib_low 되돌림
    ote_high: float  # fib_high 되돌림
    direction: str   # 'bullish' | 'bearish'


def calculate_ote_zone(
    swing_high: float,
    swing_low: float,
    direction: str = "bullish",
    fib_low: float | None = None,
    fib_high: float | None = None,
) -> OTEZone:
    """OTE 존을 계산한다.

    Bullish OTE: 저점 -> 고점 이동 후 fib_low~fib_high 되돌림 매수 구간
    Bearish OTE: 고점 -> 저점 이동 후 fib_low~fib_high 되돌림 매도 구간

    Args:
        swing_high: 스윙 고점
        swing_low: 스윙 저점
        direction: 'bullish' or 'bearish'
        fib_low: 피보나치 하단 비율. None 이면 yaml 값 사용.
        fib_high: 피보나치 상단 비율. None 이면 yaml 값 사용.

    Returns:
        OTEZone 객체
    """
    if fib_low is None or fib_high is None:
        from . import load_strategy_params
        params = load_strategy_params()["ote"]
        if fib_low is None:
            fib_low = params["fib_low"]
        if fib_high is None:
            fib_high = params["fib_high"]

    rng = swing_high - swing_low

    if direction == "bullish":
        ote_low = swing_high - rng * fib_high   # 깊은 되돌림
        ote_high = swing_high - rng * fib_low   # 얕은 되돌림
    else:  # bearish
        ote_low = swing_low + rng * fib_low
        ote_high = swing_low + rng * fib_high

    logger.debug(
        "OTE 존 계산: direction=%s, range=%.4f, zone=[%.4f, %.4f] (fib=%.3f~%.3f)",
        direction, rng, ote_low, ote_high, fib_low, fib_high,
    )

    return OTEZone(
        high=swing_high,
        low=swing_low,
        ote_low=ote_low,
        ote_high=ote_high,
        direction=direction,
    )


def is_price_in_ote(price: float, zone: OTEZone) -> bool:
    """현재 가격이 OTE 존 내에 있는지 확인한다.

    Args:
        price: 현재 가격
        zone: OTEZone 객체

    Returns:
        OTE 존 내부이면 True
    """
    return zone.ote_low <= price <= zone.ote_high
