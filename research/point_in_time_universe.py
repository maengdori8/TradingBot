from __future__ import annotations

"""미래 정보 없이 일별 유동성으로 연구 유니버스를 선택한다."""

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Literal

import pandas as pd


def _as_utc(value: datetime | str, field_name: str) -> datetime:
    """시간대가 있는 datetime을 UTC로 정규화한다."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name}이(가) 유효한 ISO 시각이 아닙니다") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


def _as_trade_date(value: date | datetime | str) -> date:
    """날짜 입력을 UTC 거래일 date로 정규화한다."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.date()
        return _as_utc(value, "trade_date").date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"유효하지 않은 trade_date입니다: {value}") from exc


def _as_bool(value: object, field_name: str) -> bool:
    """DataFrame의 명시적 bool 표현을 안전하게 변환한다."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{field_name}은(는) bool 값이어야 합니다")


@dataclass(frozen=True)
class DailyLiquidityRecord:
    """특정 거래일 거래대금과 당시 알려진 상품 메타데이터."""

    symbol: str
    trade_date: date
    available_at: datetime
    quote_volume_usd: float
    listed_at: datetime
    market_type: Literal["swap", "spot"] = "swap"
    has_matching_spot: bool = False

    def __post_init__(self) -> None:
        """레코드 값과 시각을 검증하고 UTC로 정규화한다."""
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        if self.market_type not in {"swap", "spot"}:
            raise ValueError(f"지원하지 않는 market_type입니다: {self.market_type}")
        if not math.isfinite(self.quote_volume_usd) or self.quote_volume_usd < 0:
            raise ValueError("quote_volume_usd는 0 이상의 유한한 숫자여야 합니다")
        object.__setattr__(self, "trade_date", _as_trade_date(self.trade_date))
        object.__setattr__(
            self,
            "available_at",
            _as_utc(self.available_at, "available_at"),
        )
        object.__setattr__(
            self,
            "listed_at",
            _as_utc(self.listed_at, "listed_at"),
        )
        day_close = datetime.combine(
            self.trade_date + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        if self.available_at < day_close:
            raise ValueError("일별 거래대금은 해당 UTC 거래일 종료 후에만 사용 가능합니다")
        if self.listed_at > self.available_at:
            raise ValueError("listed_at은 available_at 이후일 수 없습니다")


@dataclass(frozen=True)
class UniversePolicy:
    """시점별 유니버스 선택 기준."""

    lookback_days: int = 30
    top_n: int = 10
    min_median_daily_volume_usd: float = 10_000_000.0
    min_listing_days: int = 180
    min_observation_days: int = 30

    def __post_init__(self) -> None:
        """선택 기준의 범위를 검증한다."""
        if self.lookback_days <= 0:
            raise ValueError("lookback_days는 양수여야 합니다")
        if self.top_n <= 0:
            raise ValueError("top_n은 양수여야 합니다")
        if self.min_median_daily_volume_usd < 0:
            raise ValueError("최소 거래대금은 음수일 수 없습니다")
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days는 음수일 수 없습니다")
        if not 1 <= self.min_observation_days <= self.lookback_days:
            raise ValueError("min_observation_days는 1~lookback_days여야 합니다")


@dataclass(frozen=True)
class UniverseMember:
    """선택된 심볼의 시점별 유동성 근거."""

    symbol: str
    rank: int
    median_daily_volume_usd: float
    observation_days: int
    listed_at: datetime
    has_matching_spot: bool


@dataclass(frozen=True)
class UniverseSelection:
    """한 판단 시각에 고정된 유니버스 결과."""

    as_of: datetime
    window_start: date
    window_end_exclusive: date
    carry_only: bool
    members: tuple[UniverseMember, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        """순위 순서의 심볼 튜플을 반환한다."""
        return tuple(member.symbol for member in self.members)


def records_from_frame(frame: pd.DataFrame) -> tuple[DailyLiquidityRecord, ...]:
    """시점 보존 DataFrame을 검증된 일별 유동성 레코드로 변환한다.

    필수 컬럼은 ``symbol``, ``trade_date``, ``available_at``,
    ``quote_volume_usd``, ``listed_at``이다. ``market_type``과
    ``has_matching_spot``은 생략 시 각각 swap, False로 처리한다.
    """
    required = {
        "symbol",
        "trade_date",
        "available_at",
        "quote_volume_usd",
        "listed_at",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"유동성 DataFrame 필수 컬럼 누락: {sorted(missing)}")
    records: list[DailyLiquidityRecord] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            DailyLiquidityRecord(
                symbol=str(row["symbol"]),
                trade_date=_as_trade_date(row["trade_date"]),
                available_at=row["available_at"],
                quote_volume_usd=float(row["quote_volume_usd"]),
                listed_at=row["listed_at"],
                market_type=row.get("market_type", "swap"),
                has_matching_spot=_as_bool(
                    row.get("has_matching_spot", False),
                    "has_matching_spot",
                ),
            )
        )
    return tuple(records)


def _materialize_records(
    data: pd.DataFrame | Iterable[DailyLiquidityRecord],
) -> tuple[DailyLiquidityRecord, ...]:
    """DataFrame 또는 레코드 이터러블을 동일한 튜플로 만든다."""
    if isinstance(data, pd.DataFrame):
        return records_from_frame(data)
    records = tuple(data)
    if not all(isinstance(record, DailyLiquidityRecord) for record in records):
        raise TypeError("records는 DailyLiquidityRecord 이터러블이어야 합니다")
    return records


def select_point_in_time_universe(
    data: pd.DataFrame | Iterable[DailyLiquidityRecord],
    as_of: datetime,
    *,
    carry_only: bool = False,
    policy: UniversePolicy | None = None,
) -> UniverseSelection:
    """30일 중간 일거래대금으로 시점 안전한 상위 심볼을 선택한다.

    현재 UTC 거래일은 완료되지 않은 것으로 간주해 제외한다. 동일 심볼·거래일의
    수정 레코드는 ``as_of`` 전에 사용 가능해진 것 중 가장 늦은 버전만 사용한다.
    """
    selection_time = _as_utc(as_of, "as_of")
    effective_policy = policy or UniversePolicy()
    window_end = selection_time.date()
    window_start = window_end - timedelta(days=effective_policy.lookback_days)
    records = _materialize_records(data)

    eligible_records = [
        record
        for record in records
        if record.market_type == "swap"
        and record.available_at <= selection_time
        and window_start <= record.trade_date < window_end
    ]
    revisions: dict[tuple[str, date], DailyLiquidityRecord] = {}
    for record in eligible_records:
        key = (record.symbol, record.trade_date)
        previous = revisions.get(key)
        if previous is None or record.available_at > previous.available_at:
            revisions[key] = record

    grouped: dict[str, list[DailyLiquidityRecord]] = {}
    for record in revisions.values():
        grouped.setdefault(record.symbol, []).append(record)

    ranked: list[tuple[str, float, int, datetime, bool]] = []
    min_listing_age = timedelta(days=effective_policy.min_listing_days)
    for symbol, symbol_records in grouped.items():
        latest_metadata = max(symbol_records, key=lambda item: item.available_at)
        if latest_metadata.listed_at > selection_time:
            continue
        if selection_time - latest_metadata.listed_at < min_listing_age:
            continue
        if carry_only and not latest_metadata.has_matching_spot:
            continue
        observation_days = len(symbol_records)
        if observation_days < effective_policy.min_observation_days:
            continue
        median_volume = float(
            pd.Series(
                [record.quote_volume_usd for record in symbol_records],
                dtype="float64",
            ).median()
        )
        if median_volume < effective_policy.min_median_daily_volume_usd:
            continue
        ranked.append(
            (
                symbol,
                median_volume,
                observation_days,
                latest_metadata.listed_at,
                latest_metadata.has_matching_spot,
            )
        )

    ranked.sort(key=lambda item: (-item[1], item[0]))
    members = tuple(
        UniverseMember(
            symbol=symbol,
            rank=rank,
            median_daily_volume_usd=round(median_volume, 8),
            observation_days=observation_days,
            listed_at=listed_at,
            has_matching_spot=has_matching_spot,
        )
        for rank, (
            symbol,
            median_volume,
            observation_days,
            listed_at,
            has_matching_spot,
        ) in enumerate(ranked[: effective_policy.top_n], 1)
    )
    return UniverseSelection(
        as_of=selection_time,
        window_start=window_start,
        window_end_exclusive=window_end,
        carry_only=carry_only,
        members=members,
    )
