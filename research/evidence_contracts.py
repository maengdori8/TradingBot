from __future__ import annotations

"""근거 연구 실행과 게이트 입력의 불변·정규 해시 계약."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from research.candidates import CandidateFamily

EVIDENCE_SCHEMA_VERSION = "research-evidence-v1"
LEGACY_RESEARCH_MODULES: Mapping[str, str] = {
    "research.funding": "legacy_non_evidence",
    "research.funding_v2": "legacy_non_evidence",
    "research.wfo": "legacy_non_evidence",
}


def _utc(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 datetime을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


def _canonical_value(value: object) -> object:
    """지원 타입을 결정론적 JSON 값으로 정규화한다."""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, datetime):
        normalized = _utc(value, "datetime")
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("정규 해시의 객체 키는 문자열이어야 합니다")
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("정규 해시에는 NaN 또는 무한대를 넣을 수 없습니다")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"정규 해시가 지원하지 않는 타입입니다: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """UTF-8·키 정렬·공백 제거 규칙의 정규 JSON을 반환한다."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    """정규 JSON의 SHA-256 해시를 반환한다."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchRunManifest:
    """가설·데이터·코드·비용·시점 경계를 고정한 연구 실행 매니페스트."""

    run_id: str
    hypothesis_hash: str
    data_hash: str
    code_hash: str
    fee_snapshot_hash: str
    cost_snapshot: Mapping[str, object]
    data_cutoff: datetime
    created_at: datetime
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """식별자·해시·UTC 시각과 비용 스냅샷을 검증한다."""
        if not self.run_id.strip():
            raise ValueError("run_id는 비어 있을 수 없습니다")
        for field_name in (
            "hypothesis_hash",
            "data_hash",
            "code_hash",
            "fee_snapshot_hash",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field_name}은 소문자 SHA-256이어야 합니다")
        cutoff = _utc(self.data_cutoff, "data_cutoff")
        created = _utc(self.created_at, "created_at")
        if created < cutoff:
            raise ValueError("created_at은 data_cutoff 이전일 수 없습니다")
        if not self.cost_snapshot:
            raise ValueError("cost_snapshot은 비어 있을 수 없습니다")
        canonical_json(self.cost_snapshot)
        object.__setattr__(self, "data_cutoff", cutoff)
        object.__setattr__(self, "created_at", created)

    def manifest(self) -> dict[str, object]:
        """자체 해시에서 제외되는 파생값 없는 매니페스트를 반환한다."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "hypothesis_hash": self.hypothesis_hash,
            "data_hash": self.data_hash,
            "code_hash": self.code_hash,
            "fee_snapshot_hash": self.fee_snapshot_hash,
            "cost_snapshot": dict(self.cost_snapshot),
            "data_cutoff": self.data_cutoff,
            "created_at": self.created_at,
        }

    @property
    def manifest_hash(self) -> str:
        """정규 연구 매니페스트 해시를 반환한다."""
        return canonical_hash(self.manifest())


@dataclass(frozen=True)
class ReplayTradeRecord:
    """게이트 계산의 원천이 되는 단일 시도 또는 완결 거래."""

    candidate_id: str
    family: CandidateFamily
    fold: int
    position_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    status: Literal[
        "closed",
        "entry_unfilled",
        "entry_legging_failure",
        "exit_forced",
        "constraint_rejected",
        "signal_rejected",
    ]
    gross_pnl: float
    funding_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    capital_at_entry: float

    def __post_init__(self) -> None:
        """시각·금액·순손익 등식을 검증한다."""
        entry = _utc(self.entry_time, "entry_time")
        exit_time = _utc(self.exit_time, "exit_time")
        if exit_time < entry:
            raise ValueError("exit_time은 entry_time 이전일 수 없습니다")
        if self.fold < 0:
            raise ValueError("fold는 음수일 수 없습니다")
        if self.status not in {
            "closed",
            "entry_unfilled",
            "entry_legging_failure",
            "exit_forced",
            "constraint_rejected",
            "signal_rejected",
        }:
            raise ValueError(f"지원하지 않는 replay status입니다: {self.status}")
        if not all((self.candidate_id.strip(), self.position_id.strip(), self.symbol.strip())):
            raise ValueError("거래 식별 문자열은 비어 있을 수 없습니다")
        monetary = (
            self.gross_pnl,
            self.funding_pnl,
            self.fees,
            self.slippage,
            self.net_pnl,
            self.capital_at_entry,
        )
        if not all(math.isfinite(float(value)) for value in monetary):
            raise ValueError("거래 금액은 유한해야 합니다")
        if min(self.fees, self.slippage) < 0 or self.capital_at_entry <= 0:
            raise ValueError("비용은 음수가 아니고 진입 자본은 양수여야 합니다")
        expected_net = self.gross_pnl + self.funding_pnl - self.fees - self.slippage
        if not math.isclose(expected_net, self.net_pnl, abs_tol=1e-7):
            raise ValueError("net_pnl 등식이 일치하지 않습니다")
        object.__setattr__(self, "entry_time", entry)
        object.__setattr__(self, "exit_time", exit_time)

    @property
    def net_return(self) -> float:
        """진입 시 자본 대비 순수익률을 반환한다."""
        return self.net_pnl / self.capital_at_entry


@dataclass(frozen=True)
class DailyEvidenceRecord:
    """UTC 일별 게이트 입력 자산과 순수익률."""

    candidate_id: str
    trade_date: date
    equity: float
    net_return: float

    def __post_init__(self) -> None:
        """일별 자산과 수익률을 검증한다."""
        if not self.candidate_id.strip():
            raise ValueError("candidate_id는 비어 있을 수 없습니다")
        if not math.isfinite(self.equity) or self.equity <= 0:
            raise ValueError("equity는 양의 유한한 숫자여야 합니다")
        if not math.isfinite(self.net_return) or self.net_return <= -1:
            raise ValueError("net_return은 -1보다 큰 유한한 숫자여야 합니다")


@dataclass(frozen=True)
class CandidateReplayResult:
    """후보 한 개의 정렬된 거래·일별·비용 스트레스 증거."""

    candidate_id: str
    family: CandidateFamily
    run_manifest_hash: str
    trades: tuple[ReplayTradeRecord, ...]
    daily: tuple[DailyEvidenceRecord, ...]
    stress_daily_returns: Mapping[str, tuple[float, ...]]
    eligible_evidence: bool
    ineligibility_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """후보 일관성·정렬·스트레스 행렬 길이를 검증한다."""
        if not self.candidate_id.strip():
            raise ValueError("candidate_id는 비어 있을 수 없습니다")
        if len(self.run_manifest_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.run_manifest_hash
        ):
            raise ValueError("run_manifest_hash는 SHA-256이어야 합니다")
        if any(
            trade.candidate_id != self.candidate_id or trade.family != self.family
            for trade in self.trades
        ):
            raise ValueError("거래 후보 식별자가 결과와 일치하지 않습니다")
        if tuple(sorted(self.trades, key=lambda item: (item.entry_time, item.position_id))) != self.trades:
            raise ValueError("trades는 entry_time·position_id 순으로 정렬돼야 합니다")
        if any(record.candidate_id != self.candidate_id for record in self.daily):
            raise ValueError("일별 후보 식별자가 결과와 일치하지 않습니다")
        if tuple(sorted(self.daily, key=lambda item: item.trade_date)) != self.daily:
            raise ValueError("daily는 날짜순으로 정렬돼야 합니다")
        expected_length = len(self.daily)
        required = {"1.0x", "1.5x", "2.0x"}
        if set(self.stress_daily_returns) != required:
            raise ValueError("비용 스트레스는 정확히 1.0x, 1.5x, 2.0x여야 합니다")
        if any(len(values) != expected_length for values in self.stress_daily_returns.values()):
            raise ValueError("스트레스 수익률 길이는 일별 행 길이와 같아야 합니다")
        if self.eligible_evidence and self.ineligibility_reasons:
            raise ValueError("적격 증거에는 부적격 사유가 있을 수 없습니다")
        if not self.eligible_evidence and not self.ineligibility_reasons:
            raise ValueError("부적격 증거에는 하나 이상의 사유가 필요합니다")
        canonical_json(self.stress_daily_returns)

    @property
    def evidence_hash(self) -> str:
        """후보 결과 전체의 정규 SHA-256을 반환한다."""
        return canonical_hash(self)

    def trade_frame(self) -> pd.DataFrame:
        """게이트용 거래 DataFrame을 고정 컬럼 순서로 반환한다."""
        columns = [
            "candidate_id",
            "family",
            "fold",
            "position_id",
            "symbol",
            "entry_time",
            "exit_time",
            "status",
            "gross_pnl",
            "funding_pnl",
            "fees",
            "slippage",
            "net_pnl",
            "capital_at_entry",
            "net_return",
        ]
        rows = [
            {
                **asdict(trade),
                "net_return": trade.net_return,
            }
            for trade in self.trades
        ]
        return pd.DataFrame(rows, columns=columns)

    def daily_frame(self) -> pd.DataFrame:
        """게이트용 UTC 일별 DataFrame을 고정 컬럼 순서로 반환한다."""
        return pd.DataFrame(
            [asdict(record) for record in self.daily],
            columns=["candidate_id", "trade_date", "equity", "net_return"],
        )


def daily_evidence_from_trades(
    candidate_id: str,
    trades: Sequence[ReplayTradeRecord],
    *,
    initial_capital: float,
    cost_multiple: float = 1.0,
) -> tuple[DailyEvidenceRecord, ...]:
    """거래 원천에서 빈 날짜를 포함한 UTC 일별 자산 수익률을 계산한다."""
    if initial_capital <= 0 or not math.isfinite(initial_capital):
        raise ValueError("initial_capital은 양의 유한한 숫자여야 합니다")
    if cost_multiple <= 0 or not math.isfinite(cost_multiple):
        raise ValueError("cost_multiple은 양의 유한한 숫자여야 합니다")
    if not trades:
        return ()
    first_day = min(trade.entry_time.date() for trade in trades)
    last_day = max(trade.exit_time.date() for trade in trades)
    pnl_by_day: dict[date, float] = {}
    for trade in trades:
        stressed_pnl = (
            trade.gross_pnl
            + trade.funding_pnl
            - cost_multiple * (trade.fees + trade.slippage)
        )
        pnl_by_day[trade.exit_time.date()] = pnl_by_day.get(trade.exit_time.date(), 0.0) + stressed_pnl
    records: list[DailyEvidenceRecord] = []
    equity = float(initial_capital)
    current = first_day
    while current <= last_day:
        previous = equity
        equity += pnl_by_day.get(current, 0.0)
        if equity <= 0:
            raise ValueError("재생 자산이 0 이하가 되어 게이트 증거를 만들 수 없습니다")
        records.append(
            DailyEvidenceRecord(
                candidate_id=candidate_id,
                trade_date=current,
                equity=round(equity, 8),
                net_return=round(equity / previous - 1.0, 12),
            )
        )
        current = date.fromordinal(current.toordinal() + 1)
    return tuple(records)


def candidate_return_matrix(
    results: Sequence[CandidateReplayResult],
) -> pd.DataFrame:
    """모든 후보의 공통 UTC 일자 수익률 행렬을 결측 없이 반환한다."""
    if len(results) < 2:
        raise ValueError("PBO/SPA 후보 행렬에는 최소 두 후보가 필요합니다")
    ids = [result.candidate_id for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id가 중복됐습니다")
    series = {
        result.candidate_id: pd.Series(
            [record.net_return for record in result.daily],
            index=pd.Index([record.trade_date for record in result.daily], name="trade_date"),
            dtype="float64",
        )
        for result in results
    }
    matrix = pd.DataFrame(series).sort_index().fillna(0.0)
    return matrix.reindex(sorted(matrix.columns), axis=1)
