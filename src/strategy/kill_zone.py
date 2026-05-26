from __future__ import annotations

"""
Kill Zone 시간 필터 — London / New York 세션
"""

import logging
from datetime import datetime, time, timezone
from typing import Literal

logger = logging.getLogger(__name__)


def _load_kill_zone_times() -> dict[str, time]:
    """strategy_params.yaml 에서 Kill Zone 시간을 로드한다.

    Returns:
        london_start, london_end, ny_start, ny_end 를 담은 딕셔너리
    """
    from . import load_strategy_params
    kz = load_strategy_params()["kill_zone"]

    def _parse(s: str) -> time:
        parts = s.split(":")
        return time(int(parts[0]), int(parts[1]))

    return {
        "london_start": _parse(kz["london_start_utc"]),
        "london_end": _parse(kz["london_end_utc"]),
        "ny_start": _parse(kz["newyork_start_utc"]),
        "ny_end": _parse(kz["newyork_end_utc"]),
    }


def _to_utc(dt: datetime) -> datetime:
    """datetime 을 UTC 로 변환한다.

    Args:
        dt: 변환할 datetime

    Returns:
        UTC 로 변환된 datetime
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_in_kill_zone(dt: datetime) -> bool:
    """현재 시간이 Kill Zone(London 또는 NY) 인지 확인한다.

    Args:
        dt: 확인할 datetime (timezone-aware 권장)

    Returns:
        Kill Zone 내부이면 True
    """
    kz = _load_kill_zone_times()
    utc_dt = _to_utc(dt)
    t = utc_dt.time().replace(second=0, microsecond=0)

    in_london = kz["london_start"] <= t <= kz["london_end"]
    in_ny = kz["ny_start"] <= t <= kz["ny_end"]

    result = in_london or in_ny
    logger.debug(
        "Kill Zone 확인: %s (london=%s, ny=%s, 시각=%s)",
        "내부" if result else "외부",
        in_london,
        in_ny,
        t.strftime("%H:%M"),
    )
    return result


def get_active_session(dt: datetime) -> Literal["london", "newyork"] | None:
    """현재 활성 세션을 반환한다.

    Args:
        dt: 확인할 datetime

    Returns:
        'london' | 'newyork' | None
    """
    kz = _load_kill_zone_times()
    utc_dt = _to_utc(dt)
    t = utc_dt.time().replace(second=0, microsecond=0)

    if kz["london_start"] <= t <= kz["london_end"]:
        return "london"
    if kz["ny_start"] <= t <= kz["ny_end"]:
        return "newyork"
    return None
