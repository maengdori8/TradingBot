from __future__ import annotations

"""신규 후보 평가용 expanding walk-forward 경계를 생성한다."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd


def _as_utc(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 datetime을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class WalkForwardSplit:
    """purge와 사후 embargo를 포함한 단일 expanding 분할."""

    fold: int
    train_start: datetime
    train_end: datetime
    purge_start: datetime
    purge_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_start: datetime
    embargo_end: datetime

    def __post_init__(self) -> None:
        """모든 경계가 UTC이고 반개구간 순서를 지키는지 검증한다."""
        fields = (
            "train_start",
            "train_end",
            "purge_start",
            "purge_end",
            "test_start",
            "test_end",
            "embargo_start",
            "embargo_end",
        )
        for field_name in fields:
            object.__setattr__(
                self,
                field_name,
                _as_utc(getattr(self, field_name), field_name),
            )
        if self.fold < 0:
            raise ValueError("fold는 음수일 수 없습니다")
        if self.train_start >= self.train_end:
            raise ValueError("훈련 구간은 비어 있을 수 없습니다")
        if self.train_end != self.purge_start:
            raise ValueError("train_end와 purge_start가 이어져야 합니다")
        if self.purge_end != self.test_start:
            raise ValueError("purge_end와 test_start가 이어져야 합니다")
        if self.purge_start > self.purge_end:
            raise ValueError("purge 경계 순서가 잘못됐습니다")
        if self.test_start >= self.test_end:
            raise ValueError("테스트 구간은 비어 있을 수 없습니다")
        if self.test_end != self.embargo_start:
            raise ValueError("test_end와 embargo_start가 이어져야 합니다")
        if self.embargo_start > self.embargo_end:
            raise ValueError("embargo 경계 순서가 잘못됐습니다")


def validate_split_sequence(splits: tuple[WalkForwardSplit, ...]) -> None:
    """분할 시퀀스의 expanding·purge·embargo 불변식을 검증한다."""
    for index, split in enumerate(splits):
        if split.fold != index:
            raise ValueError("fold 번호는 0부터 연속이어야 합니다")
        if index == 0:
            continue
        previous = splits[index - 1]
        if split.train_start != previous.train_start:
            raise ValueError("expanding 훈련 시작점은 고정돼야 합니다")
        if split.train_end <= previous.train_end:
            raise ValueError("expanding 훈련 종료점은 증가해야 합니다")
        if split.test_start < previous.embargo_end:
            raise ValueError("다음 테스트가 이전 embargo와 겹칩니다")
        if split.test_start <= previous.test_start:
            raise ValueError("테스트 시작점은 증가해야 합니다")


def generate_expanding_splits(
    data_start: datetime,
    data_end: datetime,
    *,
    minimum_train: timedelta,
    test_window: timedelta,
    purge: timedelta,
    embargo: timedelta,
    include_partial_test: bool = False,
) -> tuple[WalkForwardSplit, ...]:
    """UTC 구간에서 purge·embargo가 적용된 expanding 분할을 생성한다.

    각 구간은 ``[start, end)``이다. 첫 테스트 전에 ``minimum_train`` 길이의
    실제 훈련 구간과 purge 구간을 확보하고, 다음 테스트는 이전 embargo 종료 뒤
    시작한다. 이후 훈련은 최초 시작점부터 현재 purge 시작점까지 확장된다.
    """
    start = _as_utc(data_start, "data_start")
    end = _as_utc(data_end, "data_end")
    if end <= start:
        raise ValueError("data_end는 data_start 이후여야 합니다")
    if minimum_train <= timedelta(0):
        raise ValueError("minimum_train은 양수여야 합니다")
    if test_window <= timedelta(0):
        raise ValueError("test_window는 양수여야 합니다")
    if purge < timedelta(0) or embargo < timedelta(0):
        raise ValueError("purge와 embargo는 음수일 수 없습니다")

    test_start = start + minimum_train + purge
    splits: list[WalkForwardSplit] = []
    fold = 0
    while test_start < end:
        full_test_end = test_start + test_window
        if full_test_end > end and not include_partial_test:
            break
        test_end = min(full_test_end, end)
        train_end = test_start - purge
        embargo_end = min(test_end + embargo, end)
        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_start=start,
                train_end=train_end,
                purge_start=train_end,
                purge_end=test_start,
                test_start=test_start,
                test_end=test_end,
                embargo_start=test_end,
                embargo_end=embargo_end,
            )
        )
        fold += 1
        test_start = test_end + embargo

    result = tuple(splits)
    validate_split_sequence(result)
    return result


def split_frame(
    frame: pd.DataFrame,
    split: WalkForwardSplit,
    *,
    timestamp_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """분할 경계로 훈련·테스트 DataFrame을 만들고 purge/embargo를 제외한다."""
    if timestamp_column is None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("timestamp_column 생략 시 DatetimeIndex가 필요합니다")
        timestamps = frame.index
    else:
        if timestamp_column not in frame.columns:
            raise ValueError(f"timestamp 컬럼이 없습니다: {timestamp_column}")
        timestamps = pd.DatetimeIndex(frame[timestamp_column])
    if timestamps.tz is None or str(timestamps.tz) != "UTC":
        raise ValueError("DataFrame 시각은 UTC timezone-aware여야 합니다")
    if timestamps.hasnans:
        raise ValueError("DataFrame 시각에 NaT가 포함될 수 없습니다")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("DataFrame 시각은 오름차순이어야 합니다")
    train_mask = (timestamps >= split.train_start) & (timestamps < split.train_end)
    test_mask = (timestamps >= split.test_start) & (timestamps < split.test_end)
    return frame.loc[train_mask].copy(), frame.loc[test_mask].copy()
