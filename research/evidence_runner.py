from __future__ import annotations

"""정확히 16개 사전 등록 후보의 단일 OOS 근거 연구 실행기."""

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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
    CandidateReplayResult,
    ResearchRunManifest,
    candidate_return_matrix,
    canonical_hash,
    canonical_json,
    daily_evidence_from_trades,
)
from research.execution_constraints import (
    InstrumentRules,
    ReplayExecutionPolicy,
)
from research.hypothesis_ledger import HypothesisLedger
from research.point_in_time_universe import DailyLiquidityRecord
from research.walk_forward_splits import generate_expanding_splits
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


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """중복 키가 있는 JSON 객체를 변조 가능 입력으로 거부한다."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 객체 키가 중복됐습니다: {key}")
        result[key] = value
    return result


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

    @property
    def data_evidence_eligible(self) -> bool:
        """데이터 completeness와 gap 기준 충족 여부를 반환한다."""
        return self.feed_completeness >= 0.99 and self.max_unresolved_gap_seconds <= 900


def load_evidence_dataset(path: Path | str) -> EvidenceDataset:
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
    fee_snapshot = asdict(dataset.costs)
    fee_snapshot_hash = canonical_hash(fee_snapshot)
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
        if not any(trade.status in {"closed", "exit_forced"} for trade in trades):
            reasons.append("no_completed_oos_trade")
        results.append(
            CandidateReplayResult(
                candidate_id=experiment.config_id,
                family=experiment.family,
                run_manifest_hash=manifest.manifest_hash,
                trades=trades,
                daily=daily_by_stress["1.0x"],
                stress_daily_returns={
                    key: tuple(record.net_return for record in records)
                    for key, records in daily_by_stress.items()
                },
                eligible_evidence=not reasons,
                ineligibility_reasons=tuple(reasons),
            )
        )
    if len(results) != 16:
        raise AssertionError("근거 실행 결과는 정확히 16개여야 합니다")
    return tuple(results)


def write_evidence_outputs(
    results: tuple[CandidateReplayResult, ...],
    output_dir: Path | str,
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
    matrix = candidate_return_matrix(results)
    _write_atomic(destination / "trades.csv", trades.to_csv(index=False))
    _write_atomic(destination / "daily.csv", daily.to_csv(index=False))
    _write_atomic(
        destination / "candidate_matrix.csv",
        matrix.to_csv(index=True),
    )
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


def main() -> None:
    """근거 연구 JSON을 실행하거나 고정 후보 목록을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="감사 가능한 research-evidence-v1 JSON")
    parser.add_argument("--output", type=Path, default=Path("research/out/evidence"))
    parser.add_argument("--ledger", type=Path, default=Path("research/out/hypotheses.jsonl"))
    parser.add_argument("--run-id", default="evidence-run")
    parser.add_argument("--created-by", default="strategy-agent")
    parser.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.list_candidates:
        for candidate in all_predefined_candidates():
            logger.info("%s %s", candidate.family, candidate.config_id)
        return
    if args.input is None:
        parser.error("--input 또는 --list-candidates가 필요합니다")
    dataset = load_evidence_dataset(args.input)
    results = run_evidence_pipeline(
        dataset,
        ledger=HypothesisLedger(args.ledger),
        run_id=args.run_id,
        created_by=args.created_by,
    )
    write_evidence_outputs(results, args.output)
    evidence_admissible_count = sum(result.eligible_evidence for result in results)
    logger.info(
        "근거 연구 완료 candidates=%d (carry=8 forced_flow=8) evidence_admissible=%d eligible_strategy=0 pending_risk_gate output=%s",
        len(results),
        evidence_admissible_count,
        args.output,
    )


if __name__ == "__main__":
    main()
