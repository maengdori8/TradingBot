from __future__ import annotations

"""고정 연구 산출물을 원시 통계 게이트와 Demo 활성화에 자동 연결한다."""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from research.evidence_contracts import CandidateReplayResult, ReplayTradeRecord
from research.evidence_runner import LoadedEvidenceOutputs, load_evidence_outputs
from src.risk.promotion_artifact import (
    StrategyActivation,
    build_offline_promotion_artifact,
)
from src.risk.validation_gate import (
    DatedCandidateReturns,
    DatedTradeReturn,
    build_offline_evidence_from_records,
)

logger = logging.getLogger(__name__)

_ECONOMIC_TRADE_STATUSES = frozenset(
    {"closed", "exit_forced", "entry_legging_failure"}
)


def _canonical_json(value: object) -> str:
    """NaN을 거부하는 정렬·공백 제거 JSON을 반환한다."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    """외부에 고정할 파일 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stressed_trade_return(trade: ReplayTradeRecord, cost_multiple: float) -> float:
    """거래 원천 손익에서 비용 배수별 자본 대비 순수익을 계산한다."""
    pnl = (
        trade.gross_pnl
        + trade.funding_pnl
        - cost_multiple * (trade.fees + trade.slippage)
    )
    return pnl / trade.capital_at_entry


def _dated_trades(result: CandidateReplayResult) -> tuple[DatedTradeReturn, ...]:
    """실제 경제적 노출이 있었던 거래만 통계 원시 레코드로 변환한다."""
    records = []
    for trade in result.trades:
        if trade.status not in _ECONOMIC_TRADE_STATUSES:
            continue
        records.append(
            DatedTradeReturn(
                trade_id=trade.position_id,
                closed_at=trade.exit_time,
                symbol=trade.symbol,
                net_return=trade.net_return,
                stressed_return=_stressed_trade_return(trade, 1.5),
                double_cost_return=_stressed_trade_return(trade, 2.0),
            )
        )
    return tuple(records)


def _dated_candidate_returns(
    outputs: LoadedEvidenceOutputs,
    selected_candidate_id: str,
) -> tuple[DatedCandidateReturns, ...]:
    """전체 사전 후보 행렬과 벤치마크를 같은 UTC 날짜 레코드로 묶는다."""
    matrix = outputs.candidate_matrix
    benchmark_by_date = {
        record.trade_date: record for record in outputs.benchmark_returns
    }
    records: list[DatedCandidateReturns] = []
    for trade_date, row in matrix.iterrows():
        if trade_date not in benchmark_by_date:
            raise ValueError("후보 행렬 날짜에 대응하는 benchmark가 없습니다")
        candidate_returns = {
            str(candidate_id): float(value)
            for candidate_id, value in row.items()
        }
        records.append(
            DatedCandidateReturns(
                observed_at=datetime.combine(
                    trade_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
                strategy_return=candidate_returns[selected_candidate_id],
                benchmark_return=benchmark_by_date[trade_date].benchmark_return,
                candidate_returns=candidate_returns,
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class CandidateGateOutcome:
    """후보별 자동 통계 판정과 파일 계보 요약."""

    candidate_id: str
    status: str
    passed: bool
    artifact_sha256: str | None
    activation_sha256: str | None
    failed_criteria: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        """canonical 요약용 JSON 객체를 반환한다."""
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "passed": self.passed,
            "artifact_sha256": self.artifact_sha256,
            "activation_sha256": self.activation_sha256,
            "failed_criteria": list(self.failed_criteria),
            "reason": self.reason,
        }


def _write_atomic(path: Path, payload: str) -> None:
    """완성된 문자열을 같은 디렉터리 임시 파일에서 원자 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _lineage_hash(result: CandidateReplayResult, field: str) -> str:
    """후보 run manifest의 SHA-256 계보 값을 검증해 반환한다."""
    value = result.run_manifest.get(field)
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"run manifest {field} 계보가 올바르지 않습니다")
    return value


def evaluate_offline_outputs(
    outputs: LoadedEvidenceOutputs,
    *,
    output_directory: Path | str,
    generated_at: datetime | None = None,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> tuple[CandidateGateOutcome, ...]:
    """모든 적격 후보를 원시 레코드로 재계산하고 통과분만 Demo에 활성화한다."""
    destination = Path(output_directory)
    report_time = generated_at or datetime.now(timezone.utc)
    if report_time.tzinfo is None:
        raise ValueError("generated_at은 timezone-aware여야 합니다")
    outcomes: list[CandidateGateOutcome] = []
    for result in outputs.results:
        if not result.eligible_evidence:
            outcomes.append(
                CandidateGateOutcome(
                    candidate_id=result.candidate_id,
                    status="insufficient_data",
                    passed=False,
                    artifact_sha256=None,
                    activation_sha256=None,
                    failed_criteria=(),
                    reason=",".join(result.ineligibility_reasons),
                )
            )
            continue
        trades = _dated_trades(result)
        try:
            report = build_offline_evidence_from_records(
                strategy_id=result.candidate_id,
                strategy_version=result.strategy_version,
                selected_candidate_id=result.candidate_id,
                trades=trades,
                daily_records=_dated_candidate_returns(
                    outputs,
                    result.candidate_id,
                ),
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                generated_at=report_time,
            )
            artifact = build_offline_promotion_artifact(
                report,
                strategy_sha256=result.strategy_sha256,
                code_sha256=result.code_hash,
                data_sha256=_lineage_hash(result, "data_hash"),
                hypothesis_sha256=result.hypothesis_hash,
                generated_at=report_time,
            )
        except (TypeError, ValueError, RuntimeError, PermissionError) as exc:
            outcomes.append(
                CandidateGateOutcome(
                    candidate_id=result.candidate_id,
                    status="statistical_evidence_rejected",
                    passed=False,
                    artifact_sha256=None,
                    activation_sha256=None,
                    failed_criteria=(),
                    reason=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            )
            continue
        artifact_path = destination / f"{result.candidate_id}.offline.json"
        _write_atomic(artifact_path, artifact.to_json())
        activation: StrategyActivation | None = None
        if artifact.passed:
            activation = StrategyActivation.from_promotion_artifact(artifact)
            _write_atomic(
                destination / f"{result.candidate_id}.demo-activation.json",
                activation.to_json(),
            )
        failed = tuple(
            key
            for key, criterion in artifact.criteria.items()
            if not bool(criterion["passed"])
        )
        outcomes.append(
            CandidateGateOutcome(
                candidate_id=result.candidate_id,
                status="demo_eligible" if artifact.passed else "offline_rejected",
                passed=artifact.passed,
                artifact_sha256=artifact.sha256,
                activation_sha256=(
                    activation.sha256 if activation is not None else None
                ),
                failed_criteria=failed,
                reason="" if artifact.passed else "offline gate criteria failed",
            )
        )
    return tuple(outcomes)


def run_offline_gate(
    evidence_directory: Path | str,
    *,
    expected_results_sha256: str,
    expected_matrix_sha256: str,
    expected_benchmark_sha256: str,
    output_directory: Path | str,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> Mapping[str, object]:
    """외부 고정 연구 파일을 검증하고 자동 승급 요약을 원자 저장한다."""
    source = Path(evidence_directory)
    outputs = load_evidence_outputs(
        source,
        expected_results_sha256=expected_results_sha256,
        expected_matrix_sha256=expected_matrix_sha256,
        expected_benchmark_sha256=expected_benchmark_sha256,
    )
    outcomes = evaluate_offline_outputs(
        outputs,
        output_directory=output_directory,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary: dict[str, object] = {
        "schema_version": "offline-promotion-summary/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_input": {
            "candidate_results_sha256": expected_results_sha256,
            "candidate_matrix_sha256": expected_matrix_sha256,
            "benchmark_sha256": expected_benchmark_sha256,
        },
        "candidate_count": len(outcomes),
        "eligible_strategy_count": sum(outcome.passed for outcome in outcomes),
        "outcomes": [outcome.to_dict() for outcome in outcomes],
    }
    output_path = Path(output_directory) / "promotion_summary.json"
    _write_atomic(output_path, _canonical_json(summary))
    summary["summary_file_sha256"] = _sha256_file(output_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    """오프라인 자동 게이트 CLI parser를 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--benchmark-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """고정 연구 결과를 판정하고 eligible count를 로깅한다."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    summary = run_offline_gate(
        args.evidence_dir,
        expected_results_sha256=args.results_sha256,
        expected_matrix_sha256=args.matrix_sha256,
        expected_benchmark_sha256=args.benchmark_sha256,
        output_directory=args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    logger.info(
        "오프라인 자동 게이트 완료: candidates=%s eligible_strategy_count=%s",
        summary["candidate_count"],
        summary["eligible_strategy_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
