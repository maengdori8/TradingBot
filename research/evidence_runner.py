from __future__ import annotations

"""정확히 16개 사전 등록 후보의 단일 OOS 근거 연구 실행기."""

import argparse
import csv
import hashlib
import io
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from research.candidate_replay import (
    CarryReplayOpportunity,
    ForcedFlowReplayOpportunity,
    FundingSettlement,
    ReplayCosts,
    replay_carry_candidate,
    replay_forced_flow_candidate,
)
from research.candidates import all_predefined_candidates
from research.evidence_contracts import (
    BenchmarkReturnRecord,
    CandidateReplayResult,
    ResearchRunManifest,
    candidate_return_matrix,
    canonical_hash,
    canonical_json,
    daily_evidence_from_trades,
    unavailable_candidate_result,
)
from research.execution_constraints import (
    InstrumentRules,
    ReplayExecutionPolicy,
)
from research.hypothesis_ledger import HypothesisLedger
from research.point_in_time_universe import DailyLiquidityRecord
from research.walk_forward_splits import WalkForwardSplit, generate_expanding_splits
from src.data.data_manifest import DataManifest
from src.strategy.evidence_decision import (
    BookLevel,
    FeatureFreshness,
    FeedGap,
    ForcedFlowFeatureBundle,
    LiquidationNotional,
    OrderBookEvidence,
    TimedValue,
)

logger = logging.getLogger(__name__)

_MINIMUM_TRAIN = timedelta(days=365)
_TEST_WINDOW = timedelta(days=90)
_PURGE = timedelta(hours=48)
_EMBARGO = timedelta(hours=48)

_MANIFEST_DATASETS_BY_ROLE: dict[str, frozenset[str]] = {
    "carry_spot_kline": frozenset({"kline"}),
    "carry_perpetual_kline": frozenset({"kline"}),
    "carry_funding": frozenset({"funding_settlement"}),
    "carry_open_interest": frozenset({"open_interest"}),
    "carry_metadata": frozenset({"instrument_metadata"}),
    "forced_flow_kline_volume": frozenset({"kline"}),
    "forced_flow_orderbook": frozenset({"derivatives_flow"}),
    "forced_flow_open_interest": frozenset({"derivatives_flow", "open_interest"}),
    "forced_flow_funding": frozenset({"derivatives_flow", "funding_settlement"}),
    "forced_flow_liquidation_heartbeat": frozenset({"liquidation"}),
}
_CARRY_MANIFEST_ROLES = frozenset(
    role for role in _MANIFEST_DATASETS_BY_ROLE if role.startswith("carry_")
)
_FORCED_FLOW_MANIFEST_ROLES = frozenset(
    role for role in _MANIFEST_DATASETS_BY_ROLE if role.startswith("forced_flow_")
)


@dataclass(frozen=True)
class VerifiedManifestBinding:
    """외부 고정 hash로 재검증된 DataManifest와 연구 역할의 결합."""

    role: str
    expected_sha256: str
    manifest: DataManifest

    def __post_init__(self) -> None:
        """역할·데이터셋·외부 hash·품질을 검증한다."""
        if self.role not in _MANIFEST_DATASETS_BY_ROLE:
            raise ValueError(f"지원하지 않는 data manifest role입니다: {self.role}")
        if self.manifest.dataset not in _MANIFEST_DATASETS_BY_ROLE[self.role]:
            raise ValueError(
                f"manifest dataset이 role과 맞지 않습니다: "
                f"role={self.role}, dataset={self.manifest.dataset}"
            )
        if self.expected_sha256 != self.manifest.evidence_hash:
            raise ValueError("외부 고정 hash와 manifest evidence hash가 다릅니다")
        self.manifest.assert_evidence_eligible()

    def hash_payload(self) -> dict[str, object]:
        """경로와 무관하게 data_hash에 포함할 감사 binding을 반환한다."""
        return {
            "role": self.role,
            "expected_sha256": self.expected_sha256,
            "evidence_hash": self.manifest.evidence_hash,
            "dataset": self.manifest.dataset,
            "symbol": self.manifest.symbol,
            "start": self.manifest.start,
            "end": self.manifest.end,
            "code_commit": self.manifest.code_commit,
            "raw_payload_root_sha256": self.manifest.raw_payload_root_sha256,
            "required_bindings": [
                binding.to_dict() for binding in self.manifest.required_bindings
            ],
        }


def load_verified_manifest_bindings(
    specifications: list[tuple[str, Path, str]],
) -> tuple[VerifiedManifestBinding, ...]:
    """정규 파일의 strict DataManifest를 외부 고정 hash로 재검증한다."""
    if not specifications:
        raise ValueError("최소 하나의 --data-manifest ROLE PATH SHA256가 필요합니다")
    bindings: list[VerifiedManifestBinding] = []
    identities: set[tuple[str, str, str]] = set()
    for role, path, expected_sha256 in specifications:
        if path.is_symlink():
            raise ValueError(f"data manifest symlink는 허용하지 않습니다: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"data manifest가 정규 파일이 아닙니다: {resolved}")
        try:
            serialized = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"data manifest는 UTF-8이어야 합니다: {resolved}") from exc
        manifest = DataManifest.from_json(serialized, expected_sha256)
        binding = VerifiedManifestBinding(
            role=role,
            expected_sha256=expected_sha256,
            manifest=manifest,
        )
        identity = (role, manifest.symbol, manifest.evidence_hash)
        if identity in identities:
            raise ValueError(f"data manifest binding이 중복됐습니다: {identity}")
        identities.add(identity)
        bindings.append(binding)
    return tuple(
        sorted(
            bindings,
            key=lambda item: (
                item.role,
                item.manifest.symbol,
                item.manifest.evidence_hash,
            ),
        )
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """중복 키가 있는 JSON 객체를 변조 가능 입력으로 거부한다."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 객체 키가 중복됐습니다: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    """JSON의 NaN·Infinity 확장 상수를 거부한다."""
    raise ValueError(f"비표준 JSON 숫자는 허용하지 않습니다: {value}")


def _boolean(value: object, field_name: str) -> bool:
    """JSON bool 또는 0/1만 명시적 bool로 변환한다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise TypeError(f"{field_name}은 bool이어야 합니다")


def _datetime(value: object, field_name: str) -> datetime:
    """ISO 문자열을 UTC timezone-aware datetime으로 변환한다."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name}은 ISO 문자열이어야 합니다")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name}이 유효한 ISO 시각이 아닙니다") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name}은 timezone-aware여야 합니다")
    return parsed.astimezone(timezone.utc)


def _date(value: object, field_name: str) -> date:
    """YYYY-MM-DD 문자열만 UTC 거래일 date로 변환한다."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name}은 YYYY-MM-DD 문자열이어야 합니다")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name}이 유효한 YYYY-MM-DD가 아닙니다") from exc
    return parsed


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    """JSON 객체만 문자열 키 매핑으로 반환한다."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name}은 문자열 키 JSON 객체여야 합니다")
    return value


def _sequence(value: object, field_name: str) -> list[object]:
    """JSON 배열만 목록으로 반환한다."""
    if not isinstance(value, list):
        raise TypeError(f"{field_name}은 JSON 배열이어야 합니다")
    return value


def _timed_value(value: object) -> TimedValue:
    """JSON 객체를 TimedValue로 변환한다."""
    row = _mapping(value, "timed_value")
    return TimedValue(
        observed_at=_datetime(row["observed_at"], "observed_at"),
        available_at=_datetime(row["available_at"], "available_at"),
        value=float(row["value"]),
    )


def _book_level(value: object) -> BookLevel:
    """JSON 가격 수준을 BookLevel로 변환한다."""
    row = _mapping(value, "book_level")
    return BookLevel(price=float(row["price"]), quantity=float(row["quantity"]))


def _feature_bundle(value: object) -> ForcedFlowFeatureBundle:
    """JSON 원시 특징 묶음을 공용 전략 입력으로 변환한다."""
    row = _mapping(value, "feature_bundle")
    book = _mapping(row["orderbook"], "orderbook")
    liquidations = []
    for item in _sequence(row.get("liquidations", []), "liquidations"):
        event = _mapping(item, "liquidation")
        liquidations.append(
            LiquidationNotional(
                observed_at=_datetime(event["observed_at"], "observed_at"),
                available_at=_datetime(event["available_at"], "available_at"),
                side=str(event["side"]),  # type: ignore[arg-type]
                notional=float(event["notional"]),
            )
        )
    gaps = []
    for item in _sequence(row.get("feed_gaps", []), "feed_gaps"):
        gap = _mapping(item, "feed_gap")
        ended = gap.get("ended_at")
        gaps.append(
            FeedGap(
                started_at=_datetime(gap["started_at"], "started_at"),
                ended_at=None if ended is None else _datetime(ended, "ended_at"),
                component=str(gap["component"]),  # type: ignore[arg-type]
            )
        )
    return ForcedFlowFeatureBundle(
        symbol=str(row["symbol"]),
        prices=tuple(_timed_value(item) for item in _sequence(row["prices"], "prices")),
        open_interest=tuple(
            _timed_value(item)
            for item in _sequence(row["open_interest"], "open_interest")
        ),
        completed_volumes=tuple(
            _timed_value(item)
            for item in _sequence(row["completed_volumes"], "completed_volumes")
        ),
        funding=tuple(
            _timed_value(item) for item in _sequence(row["funding"], "funding")
        ),
        liquidations=tuple(liquidations),
        orderbook=OrderBookEvidence(
            observed_at=_datetime(book["observed_at"], "observed_at"),
            available_at=_datetime(book["available_at"], "available_at"),
            bids=tuple(
                _book_level(item) for item in _sequence(book["bids"], "bids")
            ),
            asks=tuple(
                _book_level(item) for item in _sequence(book["asks"], "asks")
            ),
        ),
        feed_gaps=tuple(gaps),
    )


@dataclass(frozen=True)
class EvidenceDataset:
    """단일 실행에 필요한 감사 가능한 시장·비용·제약 데이터."""

    data_start: datetime
    data_end: datetime
    data_cutoff: datetime
    data_hash: str
    code_hash: str
    created_at: datetime
    feed_completeness: float
    max_unresolved_gap_seconds: float
    costs: ReplayCosts
    freshness: FeatureFreshness
    initial_capital: float
    execution_policy: ReplayExecutionPolicy
    rules_by_symbol: Mapping[str, InstrumentRules]
    liquidity: tuple[DailyLiquidityRecord, ...]
    benchmark_returns: tuple[BenchmarkReturnRecord, ...]
    manifest_bindings: tuple[VerifiedManifestBinding, ...]
    carry: tuple[CarryReplayOpportunity, ...]
    forced_flow: tuple[ForcedFlowReplayOpportunity, ...]

    def __post_init__(self) -> None:
        """연구 구간·해시·데이터 품질·후보 원천을 검증한다."""
        for field_name in ("data_start", "data_end", "data_cutoff", "created_at"):
            value = getattr(self, field_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name}은 timezone-aware여야 합니다")
            object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        if not (self.data_start < self.data_end <= self.data_cutoff <= self.created_at):
            raise ValueError("data_start < data_end <= data_cutoff <= created_at이어야 합니다")
        for name in ("data_hash", "code_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name}은 소문자 SHA-256이어야 합니다")
        if not math.isfinite(self.feed_completeness) or not 0 <= self.feed_completeness <= 1:
            raise ValueError("feed_completeness는 0~1이어야 합니다")
        if not math.isfinite(self.max_unresolved_gap_seconds) or self.max_unresolved_gap_seconds < 0:
            raise ValueError("max_unresolved_gap_seconds는 음수일 수 없습니다")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital은 양수여야 합니다")
        if not self.benchmark_returns:
            raise ValueError("benchmark_returns는 필수이며 비어 있을 수 없습니다")
        dates = [record.trade_date for record in self.benchmark_returns]
        if len(dates) != len(set(dates)):
            raise ValueError("benchmark_returns에 중복 UTC 거래일이 있습니다")
        if dates != sorted(dates):
            raise ValueError("benchmark_returns는 UTC 거래일 오름차순이어야 합니다")
        if any(record.available_at > self.data_cutoff for record in self.benchmark_returns):
            raise ValueError("data_cutoff 이후 공개된 benchmark return은 사용할 수 없습니다")
        if not self.manifest_bindings:
            raise ValueError("외부 고정 DataManifest binding은 필수입니다")

    @property
    def data_evidence_eligible(self) -> bool:
        """데이터 completeness와 gap 기준 충족 여부를 반환한다."""
        return self.feed_completeness >= 0.99 and self.max_unresolved_gap_seconds <= 900


def load_evidence_dataset(
    path: Path | str,
    *,
    manifest_bindings: tuple[VerifiedManifestBinding, ...],
) -> EvidenceDataset:
    """정규 JSON 입력을 검증해 EvidenceDataset으로 로드한다."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file, object_pairs_hook=_reject_duplicate_keys)
    root = _mapping(payload, "root")
    data = _mapping(root["data"], "data")
    supplied_data_hash = str(root["data_hash"])
    data_manifest = {
        "data_start": root["data_start"],
        "data_end": root["data_end"],
        "data_cutoff": root["data_cutoff"],
        "feed_completeness": root["feed_completeness"],
        "max_unresolved_gap_seconds": root["max_unresolved_gap_seconds"],
        "manifest_bindings": [
            binding.hash_payload() for binding in manifest_bindings
        ],
        "data": data,
    }
    calculated_data_hash = canonical_hash(data_manifest)
    if supplied_data_hash != calculated_data_hash:
        raise ValueError("data_hash가 품질 메타데이터 포함 정규 DataManifest와 일치하지 않습니다")

    costs_row = _mapping(root["costs"], "costs")
    freshness_row = _mapping(root["freshness_seconds"], "freshness_seconds")
    policy_row = _mapping(root["execution_policy"], "execution_policy")
    rules: dict[str, InstrumentRules] = {}
    for item in _sequence(data["instrument_rules"], "instrument_rules"):
        row = _mapping(item, "instrument_rule")
        rule = InstrumentRules(
            symbol=str(row["symbol"]),
            minimum_quantity=float(row["minimum_quantity"]),
            quantity_step=float(row["quantity_step"]),
            tick_size=float(row["tick_size"]),
            minimum_notional=float(row.get("minimum_notional", 0.0)),
            maximum_quantity=(
                None
                if row.get("maximum_quantity") is None
                else float(row["maximum_quantity"])
            ),
        )
        if rule.symbol in rules:
            raise ValueError(f"instrument rule symbol 중복: {rule.symbol}")
        rules[rule.symbol] = rule

    liquidity = []
    for item in _sequence(data["liquidity"], "liquidity"):
        row = _mapping(item, "liquidity_record")
        liquidity.append(
            DailyLiquidityRecord(
                symbol=str(row["symbol"]),
                trade_date=str(row["trade_date"]),
                available_at=_datetime(row["available_at"], "available_at"),
                quote_volume_usd=float(row["quote_volume_usd"]),
                listed_at=_datetime(row["listed_at"], "listed_at"),
                market_type=str(row.get("market_type", "swap")),  # type: ignore[arg-type]
                has_matching_spot=_boolean(
                    row.get("has_matching_spot", False),
                    "has_matching_spot",
                ),
            )
        )

    benchmark_returns = []
    for item in _sequence(data["benchmark_returns"], "benchmark_returns"):
        row = _mapping(item, "benchmark_return")
        benchmark_returns.append(
            BenchmarkReturnRecord(
                trade_date=_date(row["date"], "benchmark date"),
                benchmark_return=float(row["return"]),
                available_at=_datetime(row["available_at"], "benchmark available_at"),
            )
        )

    carry = []
    for item in _sequence(data.get("carry", []), "carry"):
        row = _mapping(item, "carry_opportunity")
        settlements = []
        for raw_settlement in _sequence(row.get("funding_settlements", []), "funding_settlements"):
            settlement = _mapping(raw_settlement, "funding_settlement")
            settlements.append(
                FundingSettlement(
                    timestamp=_datetime(settlement["timestamp"], "timestamp"),
                    rate=float(settlement["rate"]),
                    perpetual_mark_price=float(settlement["perpetual_mark_price"]),
                )
            )
        carry.append(
            CarryReplayOpportunity(
                asset_symbol=str(row["asset_symbol"]),
                spot_symbol=str(row["spot_symbol"]),
                perpetual_symbol=str(row["perpetual_symbol"]),
                entry_time=_datetime(row["entry_time"], "entry_time"),
                exit_time=_datetime(row["exit_time"], "exit_time"),
                spot_entry_price=float(row["spot_entry_price"]),
                perpetual_entry_price=float(row["perpetual_entry_price"]),
                spot_exit_price=float(row["spot_exit_price"]),
                perpetual_exit_price=float(row["perpetual_exit_price"]),
                expected_funding_rate=float(row["expected_funding_rate"]),
                observed_at=_datetime(row["observed_at"], "observed_at"),
                funding_settlements=tuple(settlements),
                requested_quantity=float(row["requested_quantity"]),
                spot_entry_fill_ratio=float(row.get("spot_entry_fill_ratio", 1.0)),
                perpetual_entry_fill_ratio=float(row.get("perpetual_entry_fill_ratio", 1.0)),
                spot_exit_fill_ratio=float(row.get("spot_exit_fill_ratio", 1.0)),
                perpetual_exit_fill_ratio=float(row.get("perpetual_exit_fill_ratio", 1.0)),
                spot_entry_slippage_rate=float(row.get("spot_entry_slippage_rate", 0.0)),
                perpetual_entry_slippage_rate=float(row.get("perpetual_entry_slippage_rate", 0.0)),
                spot_exit_slippage_rate=float(row.get("spot_exit_slippage_rate", 0.0)),
                perpetual_exit_slippage_rate=float(row.get("perpetual_exit_slippage_rate", 0.0)),
            )
        )

    forced_flow = []
    for item in _sequence(data.get("forced_flow", []), "forced_flow"):
        row = _mapping(item, "forced_flow_opportunity")
        forced_flow.append(
            ForcedFlowReplayOpportunity(
                perpetual_symbol=str(row["perpetual_symbol"]),
                decision_time=_datetime(row["decision_time"], "decision_time"),
                exit_time=_datetime(row["exit_time"], "exit_time"),
                feature_bundle=_feature_bundle(row["feature_bundle"]),
                exit_price=float(row["exit_price"]),
                requested_quantity=float(row["requested_quantity"]),
                entry_fill_ratio=float(row.get("entry_fill_ratio", 1.0)),
                exit_fill_ratio=float(row.get("exit_fill_ratio", 1.0)),
                entry_slippage_rate=float(row.get("entry_slippage_rate", 0.0)),
                exit_slippage_rate=float(row.get("exit_slippage_rate", 0.0)),
            )
        )

    return EvidenceDataset(
        data_start=_datetime(root["data_start"], "data_start"),
        data_end=_datetime(root["data_end"], "data_end"),
        data_cutoff=_datetime(root["data_cutoff"], "data_cutoff"),
        data_hash=supplied_data_hash,
        code_hash=str(root["code_hash"]),
        created_at=_datetime(root["created_at"], "created_at"),
        feed_completeness=float(root["feed_completeness"]),
        max_unresolved_gap_seconds=float(root["max_unresolved_gap_seconds"]),
        costs=ReplayCosts(
            spot_fee_rate=float(costs_row["spot_fee_rate"]),
            perpetual_fee_rate=float(costs_row["perpetual_fee_rate"]),
            assumed_slippage_rate_per_fill=float(
                costs_row["assumed_slippage_rate_per_fill"]
            ),
            fee_source=str(costs_row["fee_source"]),
        ),
        freshness=FeatureFreshness(
            price=timedelta(seconds=float(freshness_row["price"])),
            open_interest=timedelta(seconds=float(freshness_row["open_interest"])),
            funding=timedelta(seconds=float(freshness_row["funding"])),
            orderbook=timedelta(seconds=float(freshness_row["orderbook"])),
            volume=timedelta(seconds=float(freshness_row["volume"])),
            baseline_skew=timedelta(seconds=float(freshness_row["baseline_skew"])),
        ),
        initial_capital=float(root["initial_capital"]),
        execution_policy=ReplayExecutionPolicy(
            maximum_position_slots=int(policy_row["maximum_position_slots"]),
            maximum_leverage=float(policy_row["maximum_leverage"]),
            capital_utilization=float(policy_row.get("capital_utilization", 1.0)),
        ),
        rules_by_symbol=rules,
        liquidity=tuple(liquidity),
        benchmark_returns=tuple(benchmark_returns),
        manifest_bindings=manifest_bindings,
        carry=tuple(carry),
        forced_flow=tuple(forced_flow),
    )


def run_evidence_pipeline(
    dataset: EvidenceDataset,
    *,
    ledger: HypothesisLedger,
    run_id: str,
    created_by: str,
) -> tuple[CandidateReplayResult, ...]:
    """정확히 8+8 후보를 등록하고 같은 WFO·유니버스로 평가한다."""
    candidates = all_predefined_candidates()
    splits = generate_expanding_splits(
        dataset.data_start,
        dataset.data_end,
        minimum_train=_MINIMUM_TRAIN,
        test_window=_TEST_WINDOW,
        purge=_PURGE,
        embargo=_EMBARGO,
        include_partial_test=False,
    )
    if not splits:
        raise ValueError("365일 훈련과 완결 90일 OOS fold를 만들 데이터가 부족합니다")
    _validate_oos_benchmark(dataset.benchmark_returns, splits)
    fee_snapshot = asdict(dataset.costs)
    fee_snapshot_hash = canonical_hash(fee_snapshot)
    family_manifest_reasons = _family_manifest_reasons(dataset)
    results: list[CandidateReplayResult] = []
    for experiment in candidates:
        hypothesis = experiment.to_hypothesis(created_by)
        hypothesis_hash = ledger.register(hypothesis)
        manifest = ResearchRunManifest(
            run_id=f"{run_id}:{experiment.config_id}",
            hypothesis_hash=hypothesis_hash,
            data_hash=dataset.data_hash,
            code_hash=dataset.code_hash,
            fee_snapshot_hash=fee_snapshot_hash,
            cost_snapshot=fee_snapshot,
            data_cutoff=dataset.data_cutoff,
            created_at=dataset.created_at,
        )
        strategy_sha256 = canonical_hash(
            {
                "candidate_manifest_hash": hypothesis_hash,
                "code_hash": dataset.code_hash,
            }
        )
        manifest_reason = family_manifest_reasons.get(experiment.family)
        if manifest_reason is not None:
            result = unavailable_candidate_result(
                candidate_id=experiment.config_id,
                family=experiment.family,
                run_manifest_hash=manifest.manifest_hash,
                run_manifest=manifest.manifest(),
                hypothesis_hash=hypothesis_hash,
                hypothesis_manifest=hypothesis.manifest(),
                code_hash=dataset.code_hash,
                strategy_sha256=strategy_sha256,
                strategy_version=experiment.config_id,
                reason=manifest_reason,
            )
            results.append(result)
            ledger.record_run_result(
                hypothesis_hash,
                manifest.manifest_hash,
                "insufficient_data",
                {
                    "candidate_id": experiment.config_id,
                    "evidence_hash": result.evidence_hash,
                    "trade_count": 0,
                    "reasons": list(result.ineligibility_reasons),
                },
            )
            continue
        try:
            if experiment.family == "delta_neutral_carry":
                trades = replay_carry_candidate(
                    experiment,
                    dataset.carry,
                    costs=dataset.costs,
                    rules_by_symbol=dataset.rules_by_symbol,
                    liquidity=dataset.liquidity,
                    splits=splits,
                    execution_policy=dataset.execution_policy,
                    initial_capital=dataset.initial_capital,
                )
            else:
                trades = replay_forced_flow_candidate(
                    experiment,
                    dataset.forced_flow,
                    costs=dataset.costs,
                    freshness=dataset.freshness,
                    rules_by_symbol=dataset.rules_by_symbol,
                    liquidity=dataset.liquidity,
                    splits=splits,
                    execution_policy=dataset.execution_policy,
                    initial_capital=dataset.initial_capital,
                )
        except Exception as exc:
            ledger.record_run_result(
                hypothesis_hash,
                manifest.manifest_hash,
                "failed",
                {
                    "candidate_id": experiment.config_id,
                    "exception_type": type(exc).__name__,
                },
                note=str(exc)[:500],
            )
            raise
        try:
            daily_by_stress = {
                multiple: daily_evidence_from_trades(
                    experiment.config_id,
                    trades,
                    initial_capital=dataset.initial_capital,
                    cost_multiple=float(multiple.removesuffix("x")),
                )
                for multiple in ("1.0x", "1.5x", "2.0x")
            }
            reasons = []
            if not dataset.costs.promotion_eligible:
                reasons.append("account_fee_rate_snapshot_missing")
            if not dataset.data_evidence_eligible:
                reasons.append("data_manifest_quality_gate_failed")
            if not any(
                trade.status in {"closed", "exit_forced"} for trade in trades
            ):
                reasons.append("no_completed_oos_trade")
            result = CandidateReplayResult(
                candidate_id=experiment.config_id,
                family=experiment.family,
                run_manifest_hash=manifest.manifest_hash,
                run_manifest=manifest.manifest(),
                hypothesis_hash=hypothesis_hash,
                hypothesis_manifest=hypothesis.manifest(),
                code_hash=dataset.code_hash,
                strategy_sha256=strategy_sha256,
                strategy_version=experiment.config_id,
                trades=trades,
                daily=daily_by_stress["1.0x"],
                stress_daily_returns={
                    key: tuple(record.net_return for record in records)
                    for key, records in daily_by_stress.items()
                },
                eligible_evidence=not reasons,
                ineligibility_reasons=tuple(reasons),
            )
        except Exception as exc:
            ledger.record_run_result(
                hypothesis_hash,
                manifest.manifest_hash,
                "failed",
                {
                    "candidate_id": experiment.config_id,
                    "exception_type": type(exc).__name__,
                },
                note=str(exc)[:500],
            )
            raise
        results.append(result)
        ledger.record_run_result(
            hypothesis_hash,
            manifest.manifest_hash,
            "succeeded",
            {
                "candidate_id": experiment.config_id,
                "evidence_hash": result.evidence_hash,
                "trade_count": len(result.trades),
                "reasons": list(result.ineligibility_reasons),
            },
        )
    if len(results) != 16:
        raise AssertionError("근거 실행 결과는 정확히 16개여야 합니다")
    return tuple(results)


def _manifest_covers(
    binding: VerifiedManifestBinding,
    symbol: str,
    start: datetime,
    end: datetime,
) -> bool:
    """manifest가 동일 심볼의 전체 연구 범위를 포함하는지 반환한다."""
    return (
        binding.manifest.symbol == symbol
        and binding.manifest.start <= start
        and binding.manifest.end >= end
    )


def _family_manifest_reasons(
    dataset: EvidenceDataset,
) -> dict[str, str]:
    """후보군별 필수 역할·심볼·기간이 부족한 경우 사유를 반환한다."""
    by_role: dict[str, list[VerifiedManifestBinding]] = {}
    for binding in dataset.manifest_bindings:
        by_role.setdefault(binding.role, []).append(binding)

    reasons: dict[str, str] = {}
    carry_symbols_by_role = {
        "carry_spot_kline": {item.spot_symbol for item in dataset.carry},
        "carry_perpetual_kline": {item.perpetual_symbol for item in dataset.carry},
        "carry_funding": {item.perpetual_symbol for item in dataset.carry},
        "carry_open_interest": {item.perpetual_symbol for item in dataset.carry},
        "carry_metadata": {item.perpetual_symbol for item in dataset.carry},
    }
    carry_complete = bool(dataset.carry)
    for role in _CARRY_MANIFEST_ROLES:
        role_bindings = by_role.get(role, [])
        required_symbols = carry_symbols_by_role[role]
        if not role_bindings or any(
            not any(
                _manifest_covers(
                    binding,
                    symbol,
                    dataset.data_start,
                    dataset.data_end,
                )
                for binding in role_bindings
            )
            for symbol in required_symbols
        ):
            carry_complete = False
    if not carry_complete:
        reasons["delta_neutral_carry"] = "insufficient_data:carry_manifest_roles"

    forced_symbols = {
        item.perpetual_symbol for item in dataset.forced_flow
    }
    forced_complete = bool(dataset.forced_flow)
    minimum_flow_duration = timedelta(days=365)
    for role in _FORCED_FLOW_MANIFEST_ROLES:
        role_bindings = by_role.get(role, [])
        if not role_bindings or any(
            not any(
                _manifest_covers(
                    binding,
                    symbol,
                    dataset.data_start,
                    dataset.data_end,
                )
                and binding.manifest.end - binding.manifest.start
                >= minimum_flow_duration
                for binding in role_bindings
            )
            for symbol in forced_symbols
        ):
            forced_complete = False
    if not forced_complete:
        reasons["forced_flow"] = "insufficient_data:forced_flow_12_month_manifests"
    return reasons


def _oos_dates(splits: tuple[WalkForwardSplit, ...]) -> tuple[date, ...]:
    """완결 OOS test 반개구간과 겹치는 UTC 거래일을 정렬해 반환한다."""
    dates: set[date] = set()
    for split in splits:
        test_start = split.test_start
        test_end = split.test_end
        current = test_start.date()
        final = (test_end - timedelta(microseconds=1)).date()
        while current <= final:
            dates.add(current)
            current = date.fromordinal(current.toordinal() + 1)
    return tuple(sorted(dates))


def _validate_oos_benchmark(
    benchmark_returns: tuple[BenchmarkReturnRecord, ...],
    splits: tuple[WalkForwardSplit, ...],
) -> tuple[BenchmarkReturnRecord, ...]:
    """벤치마크 날짜가 모든 OOS 날짜와 정확히 일치하는지 검증한다."""
    expected = _oos_dates(splits)
    by_date = {record.trade_date: record for record in benchmark_returns}
    missing = set(expected) - set(by_date)
    if missing:
        raise ValueError(
            "benchmark가 모든 OOS 날짜를 포함해야 합니다: "
            f"missing={sorted(missing)}"
        )
    return tuple(by_date[trade_date] for trade_date in expected)


def write_evidence_outputs(
    results: tuple[CandidateReplayResult, ...],
    output_dir: Path | str,
    *,
    benchmark_returns: tuple[BenchmarkReturnRecord, ...],
) -> None:
    """정렬된 거래·일별·후보 행렬과 해시 요약을 원자 파일 집합으로 쓴다."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trade_frames = [result.trade_frame() for result in results]
    daily_frames = [result.daily_frame() for result in results]
    populated_trade_frames = [frame for frame in trade_frames if not frame.empty]
    populated_daily_frames = [frame for frame in daily_frames if not frame.empty]
    trades = (
        pd.concat(populated_trade_frames, ignore_index=True)
        if populated_trade_frames
        else trade_frames[0]
    )
    daily = (
        pd.concat(populated_daily_frames, ignore_index=True)
        if populated_daily_frames
        else daily_frames[0]
    )
    benchmark_dates = tuple(record.trade_date for record in benchmark_returns)
    matrix = candidate_return_matrix(results, required_dates=benchmark_dates)
    benchmark = pd.DataFrame(
        [asdict(record) for record in benchmark_returns],
        columns=["trade_date", "benchmark_return", "available_at"],
    )
    if tuple(matrix.index) != benchmark_dates:
        raise AssertionError("benchmark와 candidate matrix 날짜가 일치하지 않습니다")
    _write_atomic(destination / "trades.csv", trades.to_csv(index=False))
    _write_atomic(destination / "daily.csv", daily.to_csv(index=False))
    _write_atomic(
        destination / "candidate_matrix.csv",
        matrix.to_csv(index=True),
    )
    _write_atomic(destination / "benchmark.csv", benchmark.to_csv(index=False))
    summary = {
        "candidate_count": len(results),
        "evidence_admissible_count": sum(
            result.eligible_evidence for result in results
        ),
        "eligible_strategy_count": 0,
        "promotion_evaluation_status": "pending_risk_gate",
        "results": [
            {
                "candidate_id": result.candidate_id,
                "family": result.family,
                "run_manifest_hash": result.run_manifest_hash,
                "hypothesis_hash": result.hypothesis_hash,
                "strategy_sha256": result.strategy_sha256,
                "strategy_version": result.strategy_version,
                "evidence_hash": result.evidence_hash,
                "trade_count": len(result.trades),
                "daily_count": len(result.daily),
                "eligible_evidence": result.eligible_evidence,
                "ineligibility_reasons": list(result.ineligibility_reasons),
            }
            for result in results
        ],
    }
    _write_atomic(
        destination / "candidate_results.json",
        canonical_json(results) + "\n",
    )
    _write_atomic(
        destination / "evidence_summary.json",
        canonical_json(summary) + "\n",
    )


def _write_atomic(path: Path, content: str) -> None:
    """완성된 파생 산출물을 임시 파일에서 목적 경로로 원자 교체한다."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class LoadedEvidenceOutputs:
    """외부 hash와 authoritative 결과에서 재검증한 Risk gate 입력."""

    results: tuple[CandidateReplayResult, ...]
    trades: pd.DataFrame
    daily: pd.DataFrame
    candidate_matrix: pd.DataFrame
    benchmark_returns: tuple[BenchmarkReturnRecord, ...]


def _read_pinned_text(path: Path, expected_sha256: str) -> str:
    """정규 파일 bytes가 외부 고정 SHA-256과 같을 때만 UTF-8로 읽는다."""
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected output SHA-256이 올바르지 않습니다")
    if path.is_symlink():
        raise ValueError(f"evidence output symlink는 허용하지 않습니다: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"evidence output이 정규 파일이 아닙니다: {resolved}")
    payload = resolved.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"evidence output 외부 hash가 다릅니다: {path.name}")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"evidence output은 UTF-8이어야 합니다: {path.name}") from exc


def _parse_trade_record(value: object) -> object:
    """strict JSON 거래 객체를 ReplayTradeRecord로 복원한다."""
    from research.evidence_contracts import ReplayTradeRecord

    row = _mapping(value, "replay trade")
    expected = {
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
    }
    if set(row) != expected:
        raise ValueError("ReplayTradeRecord schema가 계약과 다릅니다")
    return ReplayTradeRecord(
        candidate_id=str(row["candidate_id"]),
        family=str(row["family"]),  # type: ignore[arg-type]
        fold=int(row["fold"]),
        position_id=str(row["position_id"]),
        symbol=str(row["symbol"]),
        entry_time=_datetime(row["entry_time"], "entry_time"),
        exit_time=_datetime(row["exit_time"], "exit_time"),
        status=str(row["status"]),  # type: ignore[arg-type]
        gross_pnl=float(row["gross_pnl"]),
        funding_pnl=float(row["funding_pnl"]),
        fees=float(row["fees"]),
        slippage=float(row["slippage"]),
        net_pnl=float(row["net_pnl"]),
        capital_at_entry=float(row["capital_at_entry"]),
    )


def _parse_daily_record(value: object) -> object:
    """strict JSON 일별 객체를 DailyEvidenceRecord로 복원한다."""
    from research.evidence_contracts import DailyEvidenceRecord

    row = _mapping(value, "daily evidence")
    if set(row) != {"candidate_id", "trade_date", "equity", "net_return"}:
        raise ValueError("DailyEvidenceRecord schema가 계약과 다릅니다")
    return DailyEvidenceRecord(
        candidate_id=str(row["candidate_id"]),
        trade_date=_date(row["trade_date"], "trade_date"),
        equity=float(row["equity"]),
        net_return=float(row["net_return"]),
    )


def _parse_candidate_result(value: object) -> CandidateReplayResult:
    """strict JSON 후보 객체를 hash·계보 검증 계약으로 복원한다."""
    row = _mapping(value, "candidate result")
    expected = {
        "candidate_id",
        "family",
        "run_manifest_hash",
        "run_manifest",
        "hypothesis_hash",
        "hypothesis_manifest",
        "code_hash",
        "strategy_sha256",
        "strategy_version",
        "trades",
        "daily",
        "stress_daily_returns",
        "eligible_evidence",
        "ineligibility_reasons",
    }
    if set(row) != expected:
        raise ValueError("CandidateReplayResult schema가 계약과 다릅니다")
    stress_row = _mapping(row["stress_daily_returns"], "stress_daily_returns")
    stress = {
        key: tuple(float(item) for item in _sequence(value, f"stress {key}"))
        for key, value in stress_row.items()
    }
    return CandidateReplayResult(
        candidate_id=str(row["candidate_id"]),
        family=str(row["family"]),  # type: ignore[arg-type]
        run_manifest_hash=str(row["run_manifest_hash"]),
        run_manifest=_mapping(row["run_manifest"], "run_manifest"),
        hypothesis_hash=str(row["hypothesis_hash"]),
        hypothesis_manifest=_mapping(
            row["hypothesis_manifest"],
            "hypothesis_manifest",
        ),
        code_hash=str(row["code_hash"]),
        strategy_sha256=str(row["strategy_sha256"]),
        strategy_version=str(row["strategy_version"]),
        trades=tuple(
            _parse_trade_record(item)  # type: ignore[arg-type]
            for item in _sequence(row["trades"], "trades")
        ),
        daily=tuple(
            _parse_daily_record(item)  # type: ignore[arg-type]
            for item in _sequence(row["daily"], "daily")
        ),
        stress_daily_returns=stress,
        eligible_evidence=_boolean(row["eligible_evidence"], "eligible_evidence"),
        ineligibility_reasons=tuple(
            str(item)
            for item in _sequence(
                row["ineligibility_reasons"],
                "ineligibility_reasons",
            )
        ),
    )


def _parse_benchmark_csv(serialized: str) -> tuple[BenchmarkReturnRecord, ...]:
    """고정 benchmark CSV를 중복·schema 없이 복원한다."""
    reader = csv.DictReader(io.StringIO(serialized))
    if reader.fieldnames != ["trade_date", "benchmark_return", "available_at"]:
        raise ValueError("benchmark.csv schema가 계약과 다릅니다")
    records = tuple(
        BenchmarkReturnRecord(
            trade_date=_date(row["trade_date"], "benchmark trade_date"),
            benchmark_return=float(row["benchmark_return"]),
            available_at=_datetime(row["available_at"], "benchmark available_at"),
        )
        for row in reader
    )
    dates = tuple(record.trade_date for record in records)
    if not records or dates != tuple(sorted(set(dates))):
        raise ValueError("benchmark.csv는 비어 있지 않은 고유 날짜 오름차순이어야 합니다")
    return records


def load_evidence_outputs(
    output_dir: Path | str,
    *,
    expected_results_sha256: str,
    expected_matrix_sha256: str,
    expected_benchmark_sha256: str,
) -> LoadedEvidenceOutputs:
    """authoritative 결과와 파생 matrix·benchmark를 외부 hash로 재검증한다."""
    source = Path(output_dir)
    results_json = _read_pinned_text(
        source / "candidate_results.json",
        expected_results_sha256,
    )
    payload = json.loads(
        results_json,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    raw_results = _sequence(payload, "candidate_results")
    results = tuple(_parse_candidate_result(item) for item in raw_results)
    expected_ids = tuple(
        candidate.config_id for candidate in all_predefined_candidates()
    )
    if tuple(result.candidate_id for result in results) != expected_ids:
        raise ValueError("authoritative 결과는 정확히 사전 등록 8+8 순서여야 합니다")

    benchmark_csv = _read_pinned_text(
        source / "benchmark.csv",
        expected_benchmark_sha256,
    )
    benchmark = _parse_benchmark_csv(benchmark_csv)
    dates = tuple(record.trade_date for record in benchmark)
    matrix = candidate_return_matrix(results, required_dates=dates)
    matrix_csv = _read_pinned_text(
        source / "candidate_matrix.csv",
        expected_matrix_sha256,
    )
    if matrix_csv != matrix.to_csv(index=True):
        raise ValueError("candidate_matrix.csv가 authoritative 결과와 일치하지 않습니다")

    trade_frames = [result.trade_frame() for result in results if result.trades]
    daily_frames = [result.daily_frame() for result in results if result.daily]
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else results[0].trade_frame()
    )
    daily = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else results[0].daily_frame()
    )
    return LoadedEvidenceOutputs(
        results=results,
        trades=trades,
        daily=daily,
        candidate_matrix=matrix,
        benchmark_returns=benchmark,
    )


def main() -> None:
    """근거 연구 JSON을 실행하거나 고정 후보 목록을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="감사 가능한 research-evidence-v1 JSON")
    parser.add_argument("--output", type=Path, default=Path("research/out/evidence"))
    parser.add_argument("--ledger", type=Path, default=Path("research/out/hypotheses.jsonl"))
    parser.add_argument("--run-id", default="evidence-run")
    parser.add_argument("--created-by", default="strategy-agent")
    parser.add_argument(
        "--data-manifest",
        action="append",
        nargs=3,
        metavar=("ROLE", "PATH", "SHA256"),
        help="외부 고정 DataManifest 역할·정규파일·evidence hash (반복 가능)",
    )
    parser.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.list_candidates:
        for candidate in all_predefined_candidates():
            logger.info("%s %s", candidate.family, candidate.config_id)
        return
    if args.input is None:
        parser.error("--input 또는 --list-candidates가 필요합니다")
    raw_manifest_specs = args.data_manifest or []
    manifest_bindings = load_verified_manifest_bindings(
        [
            (str(role), Path(path), str(expected_sha256))
            for role, path, expected_sha256 in raw_manifest_specs
        ]
    )
    dataset = load_evidence_dataset(
        args.input,
        manifest_bindings=manifest_bindings,
    )
    results = run_evidence_pipeline(
        dataset,
        ledger=HypothesisLedger(args.ledger),
        run_id=args.run_id,
        created_by=args.created_by,
    )
    splits = generate_expanding_splits(
        dataset.data_start,
        dataset.data_end,
        minimum_train=_MINIMUM_TRAIN,
        test_window=_TEST_WINDOW,
        purge=_PURGE,
        embargo=_EMBARGO,
        include_partial_test=False,
    )
    benchmark_returns = _validate_oos_benchmark(dataset.benchmark_returns, splits)
    write_evidence_outputs(
        results,
        args.output,
        benchmark_returns=benchmark_returns,
    )
    evidence_admissible_count = sum(result.eligible_evidence for result in results)
    logger.info(
        "근거 연구 완료 candidates=%d (carry=8 forced_flow=8) evidence_admissible=%d eligible_strategy=0 pending_risk_gate output=%s",
        len(results),
        evidence_admissible_count,
        args.output,
    )


if __name__ == "__main__":
    main()
