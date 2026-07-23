from __future__ import annotations

"""전략 결정 시각과 완전 종료 봉을 관리하는 공용 유틸리티."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

_TIMEFRAME_DELTAS: dict[str, pd.Timedelta] = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def _as_utc(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 datetime을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware datetime이어야 합니다")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class DecisionContext:
    """한 번의 전략 판단에 사용한 시각 경계를 고정한다.

    ``data_cutoff`` 이후에 알려진 데이터는 판단에 사용할 수 없고,
    ``bar_close_time``은 이번 15분 판단 봉의 종료 시각을 뜻한다.
    """

    decision_time: datetime
    data_cutoff: datetime
    bar_close_time: datetime
    strategy_version: str
    run_id: str

    def __post_init__(self) -> None:
        """UTC와 시간 순서, 15분 결정 경계를 검증한다."""
        decision_time = _as_utc(self.decision_time, "decision_time")
        data_cutoff = _as_utc(self.data_cutoff, "data_cutoff")
        bar_close_time = _as_utc(self.bar_close_time, "bar_close_time")
        if not self.strategy_version.strip():
            raise ValueError("strategy_version은 비어 있을 수 없습니다")
        if not self.run_id.strip():
            raise ValueError("run_id는 비어 있을 수 없습니다")
        if bar_close_time > data_cutoff:
            raise ValueError("bar_close_time은 data_cutoff 이후일 수 없습니다")
        if data_cutoff > decision_time:
            raise ValueError("data_cutoff는 decision_time 이후일 수 없습니다")
        if (
            bar_close_time.second
            or bar_close_time.microsecond
            or bar_close_time.minute % 15
        ):
            raise ValueError("bar_close_time은 UTC 15분 경계여야 합니다")
        object.__setattr__(self, "decision_time", decision_time)
        object.__setattr__(self, "data_cutoff", data_cutoff)
        object.__setattr__(self, "bar_close_time", bar_close_time)

    @classmethod
    def for_closed_bar(
        cls,
        bar_close_time: datetime,
        strategy_version: str,
        run_id: str,
        *,
        decision_time: datetime | None = None,
        data_cutoff: datetime | None = None,
    ) -> DecisionContext:
        """완전히 종료된 15분 봉을 기준으로 컨텍스트를 만든다."""
        close_time = _as_utc(bar_close_time, "bar_close_time")
        return cls(
            decision_time=decision_time or close_time,
            data_cutoff=data_cutoff or close_time,
            bar_close_time=close_time,
            strategy_version=strategy_version,
            run_id=run_id,
        )

    @classmethod
    def legacy_now(cls, strategy_version: str = "ict-benchmark-legacy") -> DecisionContext:
        """컨텍스트를 생략한 기존 호출을 위한 현재 시각 컨텍스트를 만든다."""
        now = datetime.now(timezone.utc)
        close = now.replace(
            minute=(now.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        return cls(
            decision_time=now,
            data_cutoff=now,
            bar_close_time=close,
            strategy_version=strategy_version,
            run_id=f"legacy-{now.isoformat()}",
        )


@dataclass(frozen=True)
class DecisionFrames:
    """하나의 결정 시각에 맞춰 잘린 멀티 타임프레임 봉."""

    df_4h: pd.DataFrame
    df_1h: pd.DataFrame
    df_15m: pd.DataFrame


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    """지원하는 타임프레임의 길이를 반환한다."""
    try:
        return _TIMEFRAME_DELTAS[timeframe]
    except KeyError as exc:
        raise ValueError(f"지원하지 않는 타임프레임입니다: {timeframe}") from exc


def validate_ohlcv_index(frame: pd.DataFrame, name: str = "OHLCV") -> None:
    """OHLCV 인덱스가 정렬된 UTC DatetimeIndex인지 검증한다."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} 인덱스는 DatetimeIndex여야 합니다")
    if frame.index.tz is None:
        raise ValueError(f"{name} 인덱스는 UTC timezone-aware여야 합니다")
    if str(frame.index.tz) != "UTC":
        raise ValueError(f"{name} 인덱스 timezone은 UTC여야 합니다")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} 인덱스는 오름차순이어야 합니다")
    if not frame.index.is_unique:
        raise ValueError(f"{name} 인덱스에 중복 시각이 있습니다")


def slice_closed_bars(
    frame: pd.DataFrame,
    timeframe: str,
    context: DecisionContext,
    *,
    lookback: int | None = None,
) -> pd.DataFrame:
    """컨텍스트 시점에 완전히 닫힌 봉만 반환한다.

    인덱스는 봉 시작 시각이다. 따라서 ``시작 + 봉 길이``가 데이터 컷오프와
    결정 봉 종료 시각을 모두 넘지 않는 행만 포함한다.
    """
    validate_ohlcv_index(frame, f"OHLCV[{timeframe}]")
    if lookback is not None and lookback <= 0:
        raise ValueError("lookback은 양수여야 합니다")
    cutoff = min(context.data_cutoff, context.bar_close_time)
    latest_open = pd.Timestamp(cutoff) - timeframe_delta(timeframe)
    end = frame.index.searchsorted(latest_open, side="right")
    start = 0 if lookback is None else max(0, end - lookback)
    sliced = frame.iloc[start:end].copy()
    if not sliced.empty:
        latest_close = sliced.index[-1] + timeframe_delta(timeframe)
        if latest_close > pd.Timestamp(cutoff):
            raise AssertionError("종료되지 않은 봉이 슬라이스에 포함됐습니다")
    return sliced


def slice_decision_frames(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    context: DecisionContext,
    *,
    lookbacks: Mapping[str, int] | None = None,
) -> DecisionFrames:
    """15분 결정 시점에 닫힌 4h/1h/15m 봉을 같은 규칙으로 자른다."""
    windows = dict(lookbacks or {})
    unknown = set(windows) - set(_TIMEFRAME_DELTAS)
    if unknown:
        raise ValueError(f"지원하지 않는 lookback 키입니다: {sorted(unknown)}")
    frames = DecisionFrames(
        df_4h=slice_closed_bars(
            df_4h,
            "4h",
            context,
            lookback=windows.get("4h"),
        ),
        df_1h=slice_closed_bars(
            df_1h,
            "1h",
            context,
            lookback=windows.get("1h"),
        ),
        df_15m=slice_closed_bars(
            df_15m,
            "15m",
            context,
            lookback=windows.get("15m"),
        ),
    )
    logger.debug(
        "결정 봉 슬라이스 완료 run_id=%s 4h=%d 1h=%d 15m=%d",
        context.run_id,
        len(frames.df_4h),
        len(frames.df_1h),
        len(frames.df_15m),
    )
    return frames


def iter_decision_contexts(
    bar_open_times: pd.DatetimeIndex,
    *,
    strategy_version: str,
    run_id: str,
) -> list[DecisionContext]:
    """15분 봉 시작 인덱스를 과거 재생용 결정 컨텍스트 목록으로 변환한다."""
    empty = pd.DataFrame(index=bar_open_times)
    validate_ohlcv_index(empty, "replay_index")
    delta = timedelta(minutes=15)
    return [
        DecisionContext.for_closed_bar(
            timestamp.to_pydatetime() + delta,
            strategy_version,
            f"{run_id}:{timestamp.isoformat()}",
        )
        for timestamp in bar_open_times
    ]
