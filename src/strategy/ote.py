"""
Optimal Trade Entry (OTE) — 피보나치 0.618 ~ 0.786 되돌림 존
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OTEZone:
    high: float
    low: float
    ote_low: float   # 0.618 되돌림
    ote_high: float  # 0.786 되돌림
    direction: str   # 'bullish' | 'bearish'


def calculate_ote_zone(swing_high: float, swing_low: float, direction: str = "bullish") -> OTEZone:
    """
    OTE 존 계산.

    Bullish OTE: 저점 → 고점 이동 후 0.618~0.786 되돌림 매수 구간
    Bearish OTE: 고점 → 저점 이동 후 0.618~0.786 되돌림 매도 구간

    Args:
        swing_high: 스윙 고점
        swing_low: 스윙 저점
        direction: 'bullish' or 'bearish'

    Returns:
        OTEZone 객체
    """
    rng = swing_high - swing_low

    if direction == "bullish":
        ote_low = swing_high - rng * 0.786   # 깊은 되돌림
        ote_high = swing_high - rng * 0.618  # 얕은 되돌림
    else:  # bearish
        ote_low = swing_low + rng * 0.618
        ote_high = swing_low + rng * 0.786

    return OTEZone(
        high=swing_high,
        low=swing_low,
        ote_low=ote_low,
        ote_high=ote_high,
        direction=direction,
    )


def is_price_in_ote(price: float, zone: OTEZone) -> bool:
    """
    현재 가격이 OTE 존 내에 있는지 확인.

    Args:
        price: 현재 가격
        zone: OTEZone 객체

    Returns:
        True if in OTE zone
    """
    return zone.ote_low <= price <= zone.ote_high
