"""
Kill Zone 시간 필터 — London / New York 세션
"""
from __future__ import annotations
from datetime import datetime, timezone, time
from typing import Literal


LONDON_START = time(2, 0)
LONDON_END   = time(5, 0)
NY_START     = time(7, 0)
NY_END       = time(10, 0)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_in_kill_zone(dt: datetime) -> bool:
    """
    현재 시간이 Kill Zone(London 또는 NY)인지 확인.

    Args:
        dt: 확인할 datetime (timezone-aware 권장)

    Returns:
        True if in kill zone
    """
    utc_dt = _to_utc(dt)
    t = utc_dt.time().replace(second=0, microsecond=0)
    in_london = LONDON_START <= t <= LONDON_END
    in_ny = NY_START <= t <= NY_END
    return in_london or in_ny


def get_active_session(dt: datetime) -> Literal["london", "newyork"] | None:
    """
    현재 활성 세션 반환.

    Args:
        dt: 확인할 datetime

    Returns:
        'london' | 'newyork' | None
    """
    utc_dt = _to_utc(dt)
    t = utc_dt.time().replace(second=0, microsecond=0)
    if LONDON_START <= t <= LONDON_END:
        return "london"
    if NY_START <= t <= NY_END:
        return "newyork"
    return None
