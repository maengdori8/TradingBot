from __future__ import annotations

"""승급 아티팩트·통계 게이트·실행 계보 장애 주입 테스트."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.exchange.contracts import TradingMode
from src.promotion import (
    VerifiedPromotion,
    load_verified_promotion,
    write_next_activation,
)
from src.risk.promotion_artifact import (
    PROMOTION_ARTIFACT_SCHEMA,
    STRATEGY_ACTIVATION_SCHEMA,
    PromotionArtifact,
    StrategyActivation,
)
from src.risk.validation_gate import (
    DatedCandidateReturns,
    DatedTradeReturn,
    DemoPromotionGate,
    DemoValidationEvidence,
    OfflinePromotionGate,
    OfflineValidationEvidence,
    build_offline_evidence_from_records,
    cscv_probability_of_backtest_overfitting,
    deflated_sharpe_probability,
    spa_block_bootstrap_pvalue,
    two_way_clustered_expectancy_ci,
)
from src.strategy.decision import DecisionContext
from src.strategy.evidence_decision import StrategyIntentLeg, StrategyTradeIntent

HASHES = {
    "strategy_sha256": "1" * 64,
    "code_sha256": "2" * 64,
    "data_sha256": "3" * 64,
    "hypothesis_sha256": "4" * 64,
    "evidence_sha256": "5" * 64,
}


def _artifact(
    *,
    stage: str = "offline",
    passed: bool = True,
    upstream: str = "",
) -> PromotionArtifact:
    """canonical 승급 아티팩트를 생성한다."""
    if stage == "demo" and not upstream:
        upstream = "a" * 64
    return PromotionArtifact(
        schema_version=PROMOTION_ARTIFACT_SCHEMA,
        stage=stage,  # type: ignore[arg-type]
        strategy_id="carry-01",
        strategy_version="carry-01-v1",
        passed=passed,
        criteria={
            "gate": {
                "name": "all criteria",
                "passed": passed,
                "value": passed,
                "threshold": True,
            }
        },
        generated_at=datetime.now(timezone.utc),
        upstream_artifact_sha256=upstream,
        **HASHES,
    )


def _verified_artifact(**kwargs: object) -> PromotionArtifact:
    """고정 hash로 재파싱한 검증 아티팩트를 반환한다."""
    artifact = _artifact(**kwargs)
    return PromotionArtifact.from_json(artifact.to_json(), expected_sha256=artifact.sha256)


def _activation(artifact: PromotionArtifact) -> StrategyActivation:
    """검증 아티팩트에서 다음 단계 활성화를 반환한다."""
    return StrategyActivation.from_promotion_artifact(artifact)


def _write_contracts(tmp_path: Path, artifact: PromotionArtifact) -> tuple[Path, Path, StrategyActivation]:
    """아티팩트와 활성화 canonical JSON을 저장한다."""
    activation = _activation(artifact)
    artifact_path = tmp_path / "artifact.json"
    activation_path = tmp_path / "activation.json"
    artifact_path.write_text(artifact.to_json(), encoding="utf-8")
    activation_path.write_text(activation.to_json(), encoding="utf-8")
    return artifact_path, activation_path, activation


def _load_kwargs(
    artifact: PromotionArtifact,
    activation: StrategyActivation,
    artifact_path: Path,
    activation_path: Path,
    mode: str = "demo",
) -> dict[str, object]:
    """load_verified_promotion 공통 인자를 반환한다."""
    return {
        "mode": mode,
        "artifact_path": artifact_path,
        "activation_path": activation_path,
        "strategy_version": artifact.strategy_version,
        "code_sha256": artifact.code_sha256,
        "data_sha256": artifact.data_sha256,
        "hypothesis_sha256": artifact.hypothesis_sha256,
        "strategy_sha256": artifact.strategy_sha256,
        "artifact_sha256": artifact.sha256,
        "activation_sha256": activation.sha256,
    }


class TestPromotionContracts:
    """엄격 JSON 스키마·hash·offline→demo→live 전이 검증."""

    def test_artifact_and_activation_round_trip_are_verified(self) -> None:
        """외부 hash와 승급 계보가 모두 일치할 때만 verified가 된다."""
        artifact = _verified_artifact()
        activation = _activation(artifact)
        parsed = StrategyActivation.from_json(
            activation.to_json(),
            expected_sha256=activation.sha256,
            promotion_artifact=artifact,
        )
        assert artifact.verified
        parsed.assert_mode_allowed(
            TradingMode.DEMO,
            strategy_version=artifact.strategy_version,
            code_sha256=artifact.code_sha256,
            data_sha256=artifact.data_sha256,
            hypothesis_sha256=artifact.hypothesis_sha256,
            strategy_sha256=artifact.strategy_sha256,
            promotion_artifact_sha256=artifact.sha256,
        )

    def test_offline_allows_demo_only_and_demo_allows_live_only(self) -> None:
        """단계를 건너뛰거나 반대 모드를 허용하지 않는다."""
        offline = _verified_artifact()
        assert _activation(offline).allowed_modes == ("demo",)
        demo = _verified_artifact(stage="demo", upstream=offline.sha256)
        assert _activation(demo).allowed_modes == ("live",)
        with pytest.raises(ValueError, match="demo만"):
            StrategyActivation(
                schema_version=STRATEGY_ACTIVATION_SCHEMA,
                promotion_stage="offline",
                strategy_id="carry-01",
                strategy_version="v1",
                allowed_modes=("live",),
                promotion_artifact_sha256="a" * 64,
                generated_at=datetime.now(timezone.utc),
                **HASHES,
            )

    def test_failed_or_unverified_artifact_cannot_activate(self) -> None:
        """미통과 또는 외부 hash 미검증 아티팩트를 활성화하지 못한다."""
        with pytest.raises(PermissionError):
            StrategyActivation.from_promotion_artifact(_artifact())
        with pytest.raises(PermissionError):
            StrategyActivation.from_promotion_artifact(_verified_artifact(passed=False))

    def test_duplicate_keys_extra_fields_and_hash_tamper_are_rejected(self) -> None:
        """중복 JSON 키·추가 필드·고정 hash 불일치를 거부한다."""
        artifact = _artifact()
        duplicate = artifact.to_json()[:-1] + ',"stage":"offline"}'
        with pytest.raises(ValueError, match="중복"):
            PromotionArtifact.from_json(duplicate)
        extra = artifact.to_dict()
        extra["unexpected"] = True
        with pytest.raises(ValueError, match="필드 집합"):
            PromotionArtifact.from_dict(extra)
        with pytest.raises(ValueError, match="고정 해시"):
            PromotionArtifact.from_json(artifact.to_json(), expected_sha256="f" * 64)

    def test_activation_lineage_tamper_is_rejected(self) -> None:
        """활성화의 코드·데이터·아티팩트 계보 변조를 거부한다."""
        artifact = _verified_artifact()
        activation = _activation(artifact)
        payload = activation.to_dict()
        payload["code_sha256"] = "f" * 64
        tampered = StrategyActivation.from_dict(payload)
        with pytest.raises(ValueError, match="계보 불일치"):
            StrategyActivation.from_json(
                tampered.to_json(),
                expected_sha256=tampered.sha256,
                promotion_artifact=artifact,
            )


class TestPromotionOrchestrator:
    """canonical 파일·symlink·실행 의도 인증 검증."""

    def test_canonical_files_load_and_executor_is_created_without_order(self, tmp_path: Path) -> None:
        """canonical 계약이 일치하면 demo 실행기 생성까지만 허용한다."""
        artifact = _verified_artifact()
        artifact_path, activation_path, activation = _write_contracts(tmp_path, artifact)
        verified = load_verified_promotion(
            **_load_kwargs(artifact, activation, artifact_path, activation_path)
        )
        assert verified.mode is TradingMode.DEMO
        with patch("src.promotion.BybitOrderExecutor") as executor:
            instance = verified.create_executor(db_path=tmp_path / "events.db")
        assert instance is executor.return_value
        executor.assert_called_once_with(mode=TradingMode.DEMO, db_path=tmp_path / "events.db")

    @pytest.mark.parametrize("suffix", ["\n", " "])
    def test_noncanonical_whitespace_is_rejected(self, tmp_path: Path, suffix: str) -> None:
        """JSON 의미가 같아도 newline·공백이 추가된 파일은 거부한다."""
        artifact = _verified_artifact()
        artifact_path, activation_path, activation = _write_contracts(tmp_path, artifact)
        artifact_path.write_text(artifact.to_json() + suffix, encoding="utf-8")
        with pytest.raises(ValueError, match="canonical"):
            load_verified_promotion(
                **_load_kwargs(artifact, activation, artifact_path, activation_path)
            )

    def test_symlink_contract_and_runtime_hash_mismatch_are_rejected(self, tmp_path: Path) -> None:
        """심볼릭 계약과 런타임 code hash 불일치를 fail-closed 처리한다."""
        artifact = _verified_artifact()
        artifact_path, activation_path, activation = _write_contracts(tmp_path, artifact)
        symlink = tmp_path / "artifact-link.json"
        symlink.symlink_to(artifact_path)
        kwargs = _load_kwargs(artifact, activation, symlink, activation_path)
        with pytest.raises(ValueError, match="정규 파일"):
            load_verified_promotion(**kwargs)
        kwargs = _load_kwargs(artifact, activation, artifact_path, activation_path)
        kwargs["code_sha256"] = "f" * 64
        with pytest.raises(PermissionError, match="code_sha256"):
            load_verified_promotion(**kwargs)

    def test_write_activation_is_atomic_and_rejects_failed_artifact(self, tmp_path: Path) -> None:
        """통과 계약만 원자적 활성화 파일로 전환한다."""
        artifact = _verified_artifact()
        source = tmp_path / "source.json"
        output = tmp_path / "nested" / "activation.json"
        source.write_text(artifact.to_json(), encoding="utf-8")
        activation = write_next_activation(
            artifact_path=source,
            output_path=output,
            artifact_sha256=artifact.sha256,
        )
        assert output.read_text(encoding="utf-8") == activation.to_json()
        failed = _verified_artifact(passed=False)
        source.write_text(failed.to_json(), encoding="utf-8")
        with pytest.raises(PermissionError, match="미통과"):
            write_next_activation(
                artifact_path=source,
                output_path=output,
                artifact_sha256=failed.sha256,
            )

    def test_authorize_intent_rejects_candidate_and_version_mismatch(self, tmp_path: Path) -> None:
        """승인 후에도 주문 의도의 후보 ID와 전략 버전을 다시 검증한다."""
        artifact = _verified_artifact()
        artifact_path, activation_path, activation = _write_contracts(tmp_path, artifact)
        verified = VerifiedPromotion(
            TradingMode.DEMO, artifact, activation, artifact_path, activation_path
        )
        now = datetime.now(timezone.utc)
        closed_bar = now.replace(
            minute=(now.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        context = DecisionContext.for_closed_bar(
            closed_bar,
            strategy_version="wrong",
            run_id="run",
            decision_time=closed_bar,
        )
        intent = StrategyTradeIntent(
            candidate_id="wrong",
            family="forced_flow",
            direction="long",
            legs=(StrategyIntentLeg(SYMBOL, "buy", 1.0, 100.0),),
            reason="test",
            context=context,
        )
        with pytest.raises(PermissionError, match="candidate_id.*strategy_version"):
            verified.authorize_intent(intent)


SYMBOL = "BTC/USDT:USDT"


def _offline_evidence(**overrides: object) -> OfflineValidationEvidence:
    """모든 오프라인 기준을 통과하는 증거를 반환한다."""
    now = datetime.now(timezone.utc)
    values: dict[str, object] = {
        "strategy_id": "carry-01",
        "strategy_version": "v1",
        "effective_bets": 220,
        "started_at": now - timedelta(days=400),
        "ended_at": now,
        "regimes": frozenset({"bull", "bear", "sideways", "high_volatility"}),
        "base_net_expectancy": 0.002,
        "stressed_net_expectancy": 0.001,
        "expectancy_ci_lower": 0.0001,
        "daily_sharpe": 1.2,
        "profit_factor": 1.3,
        "max_drawdown": 0.08,
        "deflated_sharpe_probability": 0.96,
        "pbo": 0.05,
        "spa_pvalue": 0.01,
        "max_symbol_contribution_share": 0.20,
        "max_quarter_contribution_share": 0.20,
        "double_cost_return": -0.05,
        "hypothesis_configs": 8,
    }
    values.update(overrides)
    return OfflineValidationEvidence(**values)  # type: ignore[arg-type]


def _demo_evidence(**overrides: object) -> DemoValidationEvidence:
    """모든 demo 기준을 통과하는 증거를 반환한다."""
    values: dict[str, object] = {
        "strategy_id": "carry-01",
        "strategy_version": "v1",
        "calendar_days": 90,
        "effective_bets": 100,
        "expectancy_ci_lower": 0.001,
        "daily_sharpe": 1.1,
        "profit_factor": 1.2,
        "max_drawdown": 0.07,
        "fill_error_median_bps": 4.0,
        "fill_error_p95_bps": 20.0,
        "fill_rate_error": 0.05,
        "reconciliation_rate": 1.0,
        "orphan_positions": 0,
        "duplicate_orders": 0,
        "parameters_unchanged": True,
    }
    values.update(overrides)
    return DemoValidationEvidence(**values)  # type: ignore[arg-type]


class TestStatisticalGates:
    """원시 수익률에서 산출되는 승급 통계 검증."""

    def test_offline_and_demo_require_every_criterion(self) -> None:
        """하나의 기준이라도 미통과하면 전체 승급을 거부한다."""
        assert OfflinePromotionGate().evaluate(_offline_evidence()).passed
        failed = OfflinePromotionGate().evaluate(
            _offline_evidence(expectancy_ci_lower=0.0)
        )
        assert not failed.passed and "expectancy_ci" in failed.failed_criteria
        assert DemoPromotionGate().evaluate(_demo_evidence()).passed
        assert not DemoPromotionGate().evaluate(
            _demo_evidence(duplicate_orders=1)
        ).passed

    def test_two_way_cluster_bootstrap_is_seeded_and_requires_both_dimensions(self) -> None:
        """일·심볼 독립 재표집이 고정 seed에서 재현된다."""
        args = ([0.01, 0.02, -0.01, 0.03], ["d1", "d1", "d2", "d2"], ["BTC", "ETH", "BTC", "ETH"])
        first = two_way_clustered_expectancy_ci(*args, bootstrap_samples=100, seed=7)
        second = two_way_clustered_expectancy_ci(*args, bootstrap_samples=100, seed=7)
        assert first == second and first[0] <= first[1]
        with pytest.raises(ValueError, match="2개 이상"):
            two_way_clustered_expectancy_ci(
                [0.1, 0.2], ["d1", "d1"], ["BTC", "ETH"], bootstrap_samples=10
            )

    def test_dsr_pbo_and_spa_reject_degenerate_candidate_evidence(self) -> None:
        """DSR은 다중시도를 반영하고 PBO/SPA는 중복 후보를 거부한다."""
        probability = deflated_sharpe_probability(0.5, 100, 8, 0.2)
        assert 0.0 < probability < 1.0
        duplicate = [[0.01, 0.01], [-0.01, -0.01]] * 4
        with pytest.raises(ValueError, match="중복 후보"):
            cscv_probability_of_backtest_overfitting(duplicate)
        with pytest.raises(ValueError, match="중복 후보"):
            spa_block_bootstrap_pvalue(duplicate, [0.0] * 8, bootstrap_samples=10)

    def test_raw_records_reject_single_or_duplicate_candidate_matrix(self) -> None:
        """전체 후보 행렬이 없거나 수익 열이 중복되면 증거 생성을 거부한다."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        trade = DatedTradeReturn("t1", start + timedelta(days=1), "BTC", 0.01, 0.008, 0.006)
        with pytest.raises(ValueError, match="2개 이상"):
            DatedCandidateReturns(start, 0.01, 0.0, {"only": 0.01})
        duplicate_daily = [
            DatedCandidateReturns(
                start + timedelta(days=index),
                0.01 if index % 2 else -0.005,
                0.0,
                {"a": 0.01 if index % 2 else -0.005, "b": 0.01 if index % 2 else -0.005},
            )
            for index in range(8)
        ]
        with pytest.raises(ValueError, match="중복 수익 열"):
            build_offline_evidence_from_records(
                strategy_id="carry",
                strategy_version="v1",
                selected_candidate_id="a",
                trades=[trade],
                daily_records=duplicate_daily,
                bootstrap_samples=10,
            )
