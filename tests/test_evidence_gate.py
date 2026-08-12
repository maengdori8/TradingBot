from __future__ import annotations

"""고정 연구 산출물에서 offline 아티팩트와 Demo 활성화로 가는 자동 게이트 테스트."""

import json
import hashlib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from research.candidates import all_predefined_candidates
from research.evidence_contracts import (
    BenchmarkReturnRecord,
    ReplayTradeRecord,
    candidate_return_matrix,
)
from research.evidence_runner import LoadedEvidenceOutputs, run_evidence_pipeline
from research.hypothesis_ledger import HypothesisLedger
from src.evidence_gate import (
    CandidateGateOutcome,
    _dated_candidate_returns,
    _dated_trades,
    evaluate_offline_outputs,
    main,
    run_offline_gate,
)
from src.risk.promotion_artifact import PROMOTION_ARTIFACT_SCHEMA, PromotionArtifact
from tests.test_research_evidence import _empty_dataset


def _empty_outputs(tmp_path: Path) -> LoadedEvidenceOutputs:
    """정확히 8+8 부적격 결과와 고정 benchmark·matrix를 반환한다."""
    dataset = _empty_dataset()
    results = run_evidence_pipeline(
        dataset,
        ledger=HypothesisLedger(tmp_path / "ledger.jsonl"),
        run_id="gate",
        created_by="qa",
    )
    dates = tuple(item.trade_date for item in dataset.benchmark_returns)
    return LoadedEvidenceOutputs(
        results=results,
        trades=pd.DataFrame(),
        daily=pd.DataFrame(),
        candidate_matrix=candidate_return_matrix(results, required_dates=dates),
        benchmark_returns=dataset.benchmark_returns,
    )


def _artifact(result: object, passed: bool) -> PromotionArtifact:
    """후보 run 계보와 일치하는 외부 hash 검증 offline 아티팩트를 만든다."""
    candidate = result  # type checker용 지역 별칭
    artifact = PromotionArtifact(
        schema_version=PROMOTION_ARTIFACT_SCHEMA,
        stage="offline",
        strategy_id=candidate.candidate_id,
        strategy_version=candidate.strategy_version,
        passed=passed,
        criteria={
            "expectancy_ci": {
                "name": "95% lower bound",
                "passed": passed,
                "value": 0.001 if passed else -0.001,
                "threshold": ">0",
            }
        },
        strategy_sha256=candidate.strategy_sha256,
        code_sha256=candidate.code_hash,
        data_sha256=str(candidate.run_manifest["data_hash"]),
        hypothesis_sha256=candidate.hypothesis_hash,
        evidence_sha256="e" * 64,
        generated_at=datetime.now(timezone.utc),
    )
    return PromotionArtifact.from_json(
        artifact.to_json(), expected_sha256=artifact.sha256
    )


class TestEvidenceConversion:
    """거래 비용 스트레스와 전체 후보 일별 행 변환 검증."""

    def test_only_economic_trades_are_converted_with_cost_stress(self) -> None:
        """경제적 노출 거래만 남기고 1.5배·2배 비용을 원천 손익에서 재계산한다."""
        candidate = all_predefined_candidates()[0]
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        closed = ReplayTradeRecord(
            candidate_id=candidate.config_id,
            family=candidate.family,
            fold=0,
            position_id="closed",
            symbol="BTC/USDT:USDT",
            entry_time=now,
            exit_time=now + timedelta(hours=8),
            status="closed",
            gross_pnl=10.0,
            funding_pnl=2.0,
            fees=2.0,
            slippage=1.0,
            net_pnl=9.0,
            capital_at_entry=1000.0,
        )
        rejected = replace(
            closed,
            position_id="rejected",
            status="constraint_rejected",
        )
        converted = _dated_trades(SimpleNamespace(trades=(closed, rejected)))
        assert len(converted) == 1
        assert converted[0].net_return == pytest.approx(0.009)
        assert converted[0].stressed_return == pytest.approx(0.0075)
        assert converted[0].double_cost_return == pytest.approx(0.006)

    def test_candidate_rows_bind_selected_column_and_same_day_benchmark(self) -> None:
        """후보 행렬 날짜를 같은 UTC 날짜 benchmark와 결합한다."""
        first, second = (item.config_id for item in all_predefined_candidates()[:2])
        days = (date(2025, 1, 1), date(2025, 1, 2))
        matrix = pd.DataFrame(
            {first: [0.01, -0.01], second: [-0.02, 0.02]},
            index=pd.Index(days, name="trade_date"),
        )
        benchmarks = tuple(
            BenchmarkReturnRecord(
                trade_date=day,
                benchmark_return=0.001,
                available_at=datetime.combine(
                    day + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
            )
            for day in days
        )
        outputs = SimpleNamespace(
            candidate_matrix=matrix,
            benchmark_returns=benchmarks,
        )
        records = _dated_candidate_returns(outputs, first)
        assert records[0].strategy_return == 0.01
        assert dict(records[1].candidate_returns) == {
            first: -0.01,
            second: 0.02,
        }
        with pytest.raises(ValueError, match="benchmark"):
            _dated_candidate_returns(
                replace_loaded(outputs, benchmark_returns=benchmarks[:1]),
                first,
            )


def replace_loaded(value: object, **overrides: object) -> SimpleNamespace:
    """SimpleNamespace 기반 게이트 입력 일부를 교체한다."""
    fields = dict(vars(value))
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestOfflineAutomation:
    """부족·통계 탈락·Demo 적격 자동 분기와 파일 영속화 검증."""

    def test_ineligible_results_remain_insufficient_without_artifacts(
        self,
        tmp_path: Path,
    ) -> None:
        """부적격 8+8 결과를 통계에 넣거나 주문 경로에 연결하지 않는다."""
        outcomes = evaluate_offline_outputs(
            _empty_outputs(tmp_path),
            output_directory=tmp_path / "gate",
            bootstrap_samples=10,
        )
        assert len(outcomes) == 16
        assert all(outcome.status == "insufficient_data" for outcome in outcomes)
        assert all(not outcome.passed for outcome in outcomes)
        assert not list((tmp_path / "gate").glob("*.json"))

    def test_nominally_eligible_but_empty_result_is_statistically_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """eligible 표식만 바꿔도 원시 거래가 없으면 통계 증거 단계에서 거부한다."""
        outputs = _empty_outputs(tmp_path)
        eligible = replace(
            outputs.results[0],
            eligible_evidence=True,
            ineligibility_reasons=(),
        )
        narrowed = replace_loaded(outputs, results=(eligible,))
        outcome = evaluate_offline_outputs(
            narrowed,
            output_directory=tmp_path / "rejected",
            bootstrap_samples=10,
        )[0]
        assert outcome.status == "statistical_evidence_rejected"
        assert outcome.artifact_sha256 is None
        assert not (tmp_path / "rejected").exists()

    def test_failed_artifact_is_written_without_demo_activation(self, tmp_path: Path) -> None:
        """통계 산출 후 오프라인 기준 탈락은 감사 파일만 남기고 활성화하지 않는다."""
        outputs = _empty_outputs(tmp_path)
        candidate = replace(
            outputs.results[0], eligible_evidence=True, ineligibility_reasons=()
        )
        narrowed = replace_loaded(outputs, results=(candidate,))
        artifact = _artifact(candidate, passed=False)
        with (
            patch("src.evidence_gate.build_offline_evidence_from_records", return_value=object()),
            patch("src.evidence_gate.build_offline_promotion_artifact", return_value=artifact),
        ):
            outcome = evaluate_offline_outputs(
                narrowed, output_directory=tmp_path / "failed"
            )[0]
        assert outcome.status == "offline_rejected"
        assert outcome.failed_criteria == ("expectancy_ci",)
        assert (tmp_path / "failed" / f"{candidate.candidate_id}.offline.json").exists()
        assert not (tmp_path / "failed" / f"{candidate.candidate_id}.demo-activation.json").exists()

    def test_passed_artifact_alone_creates_demo_activation(self, tmp_path: Path) -> None:
        """모든 오프라인 기준을 통과한 검증 아티팩트만 Demo 활성화를 만든다."""
        outputs = _empty_outputs(tmp_path)
        candidate = replace(
            outputs.results[0], eligible_evidence=True, ineligibility_reasons=()
        )
        narrowed = replace_loaded(outputs, results=(candidate,))
        artifact = _artifact(candidate, passed=True)
        with (
            patch("src.evidence_gate.build_offline_evidence_from_records", return_value=object()),
            patch("src.evidence_gate.build_offline_promotion_artifact", return_value=artifact),
        ):
            outcome = evaluate_offline_outputs(
                narrowed, output_directory=tmp_path / "passed"
            )[0]
        assert outcome.status == "demo_eligible" and outcome.passed
        activation_path = (
            tmp_path / "passed" / f"{candidate.candidate_id}.demo-activation.json"
        )
        serialized = activation_path.read_text(encoding="utf-8")
        activation = json.loads(serialized)
        assert activation["allowed_modes"] == ["demo"]
        assert outcome.activation_sha256 == hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()


class TestPinnedGateRun:
    """외부 파일 hash 전달·요약 eligible count·CLI 연결 검증."""

    def test_run_uses_pinned_hashes_and_writes_summary(self, tmp_path: Path) -> None:
        """고정 세 파일 hash를 loader에 전달하고 결과 수를 canonical 요약으로 저장한다."""
        outputs = SimpleNamespace()
        outcome = CandidateGateOutcome(
            candidate_id="carry-a",
            status="offline_rejected",
            passed=False,
            artifact_sha256="a" * 64,
            activation_sha256=None,
            failed_criteria=("expectancy_ci",),
            reason="offline gate criteria failed",
        )
        destination = tmp_path / "summary"
        with (
            patch("src.evidence_gate.load_evidence_outputs", return_value=outputs) as loader,
            patch("src.evidence_gate.evaluate_offline_outputs", return_value=(outcome,)),
        ):
            summary = run_offline_gate(
                tmp_path / "evidence",
                expected_results_sha256="1" * 64,
                expected_matrix_sha256="2" * 64,
                expected_benchmark_sha256="3" * 64,
                output_directory=destination,
                bootstrap_samples=10,
                seed=7,
            )
        loader.assert_called_once_with(
            tmp_path / "evidence",
            expected_results_sha256="1" * 64,
            expected_matrix_sha256="2" * 64,
            expected_benchmark_sha256="3" * 64,
        )
        assert summary["eligible_strategy_count"] == 0
        assert len(str(summary["summary_file_sha256"])) == 64
        on_disk = json.loads(
            (destination / "promotion_summary.json").read_text(encoding="utf-8")
        )
        assert "summary_file_sha256" not in on_disk

    def test_cli_forwards_all_pinned_arguments(self, tmp_path: Path) -> None:
        """CLI가 세 input hash와 output을 자동 게이트에 그대로 전달한다."""
        summary = {"candidate_count": 16, "eligible_strategy_count": 0}
        with patch("src.evidence_gate.run_offline_gate", return_value=summary) as runner:
            assert main(
                [
                    "--evidence-dir",
                    str(tmp_path / "in"),
                    "--results-sha256",
                    "1" * 64,
                    "--matrix-sha256",
                    "2" * 64,
                    "--benchmark-sha256",
                    "3" * 64,
                    "--output",
                    str(tmp_path / "out"),
                    "--bootstrap-samples",
                    "10",
                    "--seed",
                    "7",
                ]
            ) == 0
        assert runner.call_args.kwargs["expected_results_sha256"] == "1" * 64
        assert runner.call_args.kwargs["bootstrap_samples"] == 10
