from __future__ import annotations

"""원시 거래·일별 수익에서 생성되는 오프라인·데모 증거 리포트 테스트."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from src.risk.validation_gate import (
    DatedCandidateReturns,
    DatedTradeReturn,
    DemoApprovalReport,
    DemoPromotionGate,
    DemoValidationEvidence,
    OfflineValidationEvidence,
    build_demo_approval_report,
    build_offline_evidence_from_records,
    build_offline_evidence_report,
    clustered_expectancy_ci,
    cscv_probability_of_backtest_overfitting,
    deflated_sharpe_probability,
    max_positive_contribution_share,
    spa_block_bootstrap_pvalue,
    two_way_clustered_expectancy_ci,
)

HASHES = {
    "raw_event_sha256": "1" * 64,
    "data_sha256": "2" * 64,
    "code_sha256": "3" * 64,
    "hypothesis_sha256": "4" * 64,
    "offline_artifact_sha256": "5" * 64,
}


def _demo(**overrides: object) -> DemoValidationEvidence:
    """모든 미래 데모 기준을 통과하는 증거를 반환한다."""
    values: dict[str, object] = {
        "strategy_id": "carry-a",
        "strategy_version": "carry-a-v1",
        "calendar_days": 90,
        "effective_bets": 100,
        "expectancy_ci_lower": 0.001,
        "daily_sharpe": 1.1,
        "profit_factor": 1.3,
        "max_drawdown": 0.05,
        "fill_error_median_bps": 4.0,
        "fill_error_p95_bps": 20.0,
        "fill_rate_error": 0.08,
        "reconciliation_rate": 1.0,
        "orphan_positions": 0,
        "duplicate_orders": 0,
        "parameters_unchanged": True,
    }
    values.update(overrides)
    return DemoValidationEvidence(**values)  # type: ignore[arg-type]


def _raw_daily(start: datetime, days: int = 60) -> list[DatedCandidateReturns]:
    """고정 후보 집합과 여러 시장 레짐을 포함한 일별 원시 레코드를 만든다."""
    records: list[DatedCandidateReturns] = []
    for index in range(days):
        selected = 0.004 if index % 3 else -0.002
        alternative = -0.003 if index % 4 else 0.006
        if index < 20:
            benchmark = 0.01
        elif index < 40:
            benchmark = -0.01
        else:
            benchmark = 0.025 if index % 2 else -0.025
        records.append(
            DatedCandidateReturns(
                observed_at=start + timedelta(days=index),
                strategy_return=selected,
                benchmark_return=benchmark,
                candidate_returns={"carry-a": selected, "carry-b": alternative},
            )
        )
    return records


def _raw_trades(start: datetime) -> list[DatedTradeReturn]:
    """두 일자·두 심볼 클러스터의 비용 시나리오별 거래를 만든다."""
    return [
        DatedTradeReturn("t1", start + timedelta(days=1), "BTC", 0.01, 0.008, 0.006),
        DatedTradeReturn("t2", start + timedelta(days=1), "ETH", -0.003, -0.005, -0.007),
        DatedTradeReturn("t3", start + timedelta(days=2), "BTC", 0.012, 0.009, 0.007),
        DatedTradeReturn("t4", start + timedelta(days=2), "ETH", 0.006, 0.004, 0.002),
    ]


class TestRawEvidenceContracts:
    """원시 레코드 시각·후보 행렬·통계 입력 계약 검증."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"trade_id": "", "symbol": "BTC"},
            {"trade_id": "t", "symbol": ""},
            {"trade_id": "t", "symbol": "BTC", "net_return": float("nan")},
            {"trade_id": "t", "symbol": "BTC", "net_return": -1.0},
        ],
    )
    def test_trade_record_rejects_invalid_identity_and_returns(
        self,
        kwargs: dict[str, object],
    ) -> None:
        """빈 식별자·비유한 수익·전액 손실을 증거로 허용하지 않는다."""
        values: dict[str, object] = {
            "trade_id": "t",
            "closed_at": datetime.now(timezone.utc),
            "symbol": "BTC",
            "net_return": 0.01,
            "stressed_return": 0.005,
            "double_cost_return": 0.0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError):
            DatedTradeReturn(**values)  # type: ignore[arg-type]

    def test_candidate_record_is_immutable_and_rejects_bad_rows(self) -> None:
        """후보 행렬 row를 정렬·불변화하고 단일·비유한 후보를 거부한다."""
        now = datetime.now(timezone.utc)
        record = DatedCandidateReturns(now, 0.01, 0.0, {"b": -0.01, "a": 0.01})
        assert tuple(record.candidate_returns) == ("a", "b")
        assert isinstance(record.candidate_returns, MappingProxyType)
        with pytest.raises(TypeError):
            record.candidate_returns["a"] = 0.0  # type: ignore[index]
        with pytest.raises(ValueError, match="2개 이상"):
            DatedCandidateReturns(now, 0.01, 0.0, {"a": 0.01})
        with pytest.raises(ValueError, match="유한"):
            DatedCandidateReturns(now, float("inf"), 0.0, {"a": 0.0, "b": 0.1})

    def test_statistical_primitives_reject_malformed_inputs(self) -> None:
        """클러스터·DSR·PBO·SPA의 불충분하거나 비유한 입력을 거부한다."""
        with pytest.raises(ValueError, match="같은 길이"):
            clustered_expectancy_ci([0.1], [], bootstrap_samples=10)
        with pytest.raises(ValueError, match="confidence"):
            clustered_expectancy_ci([0.1], ["d"], confidence=1.0)
        with pytest.raises(ValueError, match="유한"):
            two_way_clustered_expectancy_ci(
                [0.1, float("nan")], ["d1", "d2"], ["BTC", "ETH"]
            )
        with pytest.raises(ValueError, match="관측 수"):
            deflated_sharpe_probability(1.0, 1, 2, 0.1)
        with pytest.raises(ValueError, match="8행"):
            cscv_probability_of_backtest_overfitting([[0.1, 0.2]])
        with pytest.raises(ValueError, match="길이"):
            spa_block_bootstrap_pvalue([[0.1, 0.2]] * 8, [0.0] * 7)
        assert max_positive_contribution_share({"BTC": -1.0}) == 1.0
        assert max_positive_contribution_share({"BTC": 3.0, "ETH": 1.0}) == 0.75


class TestOfflineEvidenceReports:
    """호환 리포트와 승급 가능한 원시 레코드 리포트 검증."""

    def test_legacy_manual_report_is_explicitly_non_promotable(self) -> None:
        """수동 레짐·라벨 리포트는 통계를 계산해도 legacy로 명시한다."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        daily = [0.03, -0.004, 0.008, -0.003, 0.007, -0.002, 0.009, -0.001]
        candidates = [
            [
                value,
                (-0.5 * value if index % 2 else 0.2 * value),
                (-0.004 if index % 3 == 0 else 0.003),
            ]
            for index, value in enumerate(daily)
        ]
        report = build_offline_evidence_report(
            strategy_id="legacy",
            strategy_version="v1",
            started_at=start,
            ended_at=start + timedelta(days=365),
            regimes={"bull", "bear"},
            net_returns=[0.01, -0.002, 0.008, 0.004],
            stressed_returns=[0.008, -0.004, 0.006, 0.002],
            double_cost_returns=[0.006, -0.006, 0.004, 0.0],
            trade_clusters=["d1", "d1", "d2", "d2"],
            symbols=["BTC", "ETH", "BTC", "ETH"],
            quarters=["Q1", "Q1", "Q2", "Q2"],
            daily_returns=daily,
            candidate_return_matrix=candidates,
            benchmark_returns=[0.0] * len(daily),
            bootstrap_samples=20,
            seed=7,
        )
        assert report.methodology == "legacy-manual-labels/non-promotable-v1"
        assert report.evidence.effective_bets == 2
        assert len(report.sha256) == 64
        assert report.to_dict()["raw_input_sha256"] == report.raw_input_sha256

    @pytest.mark.parametrize(
        "override",
        [
            {"net_returns": []},
            {"stressed_returns": [0.1]},
            {"daily_returns": [0.1]},
            {"benchmark_returns": [0.0]},
            {"net_returns": [float("nan"), 0.1]},
        ],
    )
    def test_legacy_report_rejects_incomplete_or_nonfinite_arrays(
        self,
        override: dict[str, object],
    ) -> None:
        """거래·일별·후보 배열의 길이와 유한성을 검사한다."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values: dict[str, object] = {
            "strategy_id": "legacy",
            "strategy_version": "v1",
            "started_at": start,
            "ended_at": start + timedelta(days=10),
            "regimes": {"sideways"},
            "net_returns": [0.1, -0.02],
            "stressed_returns": [0.08, -0.04],
            "double_cost_returns": [0.06, -0.06],
            "trade_clusters": ["d1", "d2"],
            "symbols": ["BTC", "ETH"],
            "quarters": ["Q1", "Q1"],
            "daily_returns": [0.01, -0.01] * 4,
            "candidate_return_matrix": [[0.01, -0.01], [-0.01, 0.02]] * 4,
            "benchmark_returns": [0.0] * 8,
            "bootstrap_samples": 10,
        }
        values.update(override)
        with pytest.raises(ValueError):
            build_offline_evidence_report(**values)  # type: ignore[arg-type]

    def test_dated_records_build_reproducible_promotable_report(self) -> None:
        """날짜·심볼·전체 후보 행렬로 모든 승급 통계를 자동 산출한다."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = build_offline_evidence_from_records(
            strategy_id="carry-a",
            strategy_version="v1",
            selected_candidate_id="carry-a",
            trades=_raw_trades(start),
            daily_records=_raw_daily(start),
            bootstrap_samples=20,
            seed=11,
            generated_at=start + timedelta(days=61),
        )
        assert report.methodology.startswith("two-way-day-symbol-bootstrap")
        assert report.evidence.effective_bets == 4
        assert report.evidence.hypothesis_configs == 2
        assert {"bull", "bear", "sideways"} <= report.evidence.regimes
        assert len(report.raw_input_sha256) == 64

    def test_dated_records_reject_duplicate_ids_dates_and_outside_trades(self) -> None:
        """중복 거래·중복 UTC 일자·기간 밖 거래를 각각 거부한다."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        daily = _raw_daily(start, days=8)
        trades = _raw_trades(start)
        common = {
            "strategy_id": "carry-a",
            "strategy_version": "v1",
            "selected_candidate_id": "carry-a",
            "bootstrap_samples": 10,
        }
        with pytest.raises(ValueError, match="중복 trade_id"):
            build_offline_evidence_from_records(
                **common, trades=[trades[0], trades[0]], daily_records=daily
            )
        with pytest.raises(ValueError, match="UTC 하루"):
            build_offline_evidence_from_records(
                **common, trades=trades, daily_records=[daily[0], daily[0], *daily[1:]]
            )
        outside = DatedTradeReturn(
            "outside", start - timedelta(days=1), "BTC", 0.01, 0.008, 0.006
        )
        with pytest.raises(ValueError, match=r"\[시작, 종료\)"):
            build_offline_evidence_from_records(
                **common, trades=[outside, *trades], daily_records=daily
            )

    def test_dated_records_reject_candidate_set_and_selected_return_mismatch(self) -> None:
        """날짜별 후보 집합 변경과 선택 열 수익 불일치를 거부한다."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        daily = _raw_daily(start, days=8)
        changed = DatedCandidateReturns(
            daily[-1].observed_at,
            daily[-1].strategy_return,
            daily[-1].benchmark_return,
            {"carry-a": daily[-1].strategy_return, "carry-c": 0.0},
        )
        with pytest.raises(ValueError, match="후보 ID 집합"):
            build_offline_evidence_from_records(
                strategy_id="carry-a",
                strategy_version="v1",
                selected_candidate_id="carry-a",
                trades=_raw_trades(start),
                daily_records=[*daily[:-1], changed],
                bootstrap_samples=10,
            )
        mismatch = DatedCandidateReturns(
            daily[-1].observed_at,
            0.123,
            daily[-1].benchmark_return,
            daily[-1].candidate_returns,
        )
        with pytest.raises(ValueError, match="선택 열"):
            build_offline_evidence_from_records(
                strategy_id="carry-a",
                strategy_version="v1",
                selected_candidate_id="carry-a",
                trades=_raw_trades(start),
                daily_records=[*daily[:-1], mismatch],
                bootstrap_samples=10,
            )


class TestDemoApprovalReports:
    """데모 통과 판정과 실전 계보 리포트의 불변성 검증."""

    def test_passed_report_requires_full_lineage_and_is_canonical(self) -> None:
        """통과 데모는 다섯 계보 hash를 모두 결합해 canonical 승인 리포트를 만든다."""
        generated = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="계보 해시"):
            DemoPromotionGate().build_approval_report(
                _demo(), generated_at=generated
            )
        report = build_demo_approval_report(
            _demo(), generated_at=generated, **HASHES
        )
        assert report.passed
        assert report.generated_at == generated
        assert report.stage == "demo"
        assert report.to_dict()["methodology"] == report.methodology
        assert len(report.sha256) == 64
        with pytest.raises(FrozenInstanceError):
            report.stage = "offline"  # type: ignore[misc]

    def test_failed_report_needs_no_lineage_but_invalid_hash_is_rejected(self) -> None:
        """미통과 리포트는 진단용 생성이 가능하고 제공된 잘못된 hash는 거부한다."""
        failed = build_demo_approval_report(_demo(duplicate_orders=1))
        assert not failed.passed
        with pytest.raises(ValueError, match="raw_event_sha256"):
            build_demo_approval_report(
                _demo(duplicate_orders=1), raw_event_sha256="BAD"
            )
        with pytest.raises(ValueError, match="timezone"):
            build_demo_approval_report(
                _demo(duplicate_orders=1), generated_at=datetime(2025, 1, 1)
            )

    def test_direct_report_construction_is_forbidden(self) -> None:
        """외부 코드가 게이트 판정 토큰 없이 승인 리포트를 조작하지 못한다."""
        decision = DemoPromotionGate().evaluate(_demo())
        with pytest.raises(ValueError, match="DemoPromotionGate"):
            DemoApprovalReport(
                decision,
                "a" * 64,
                datetime.now(timezone.utc),
                object(),
            )

    def test_evidence_models_reject_invalid_ranges(self) -> None:
        """오프라인 확률·시각과 데모 대사율·오류 건수 범위를 검사한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="timezone"):
            OfflineValidationEvidence(
                strategy_id="x",
                strategy_version="v1",
                effective_bets=1,
                started_at=datetime(2024, 1, 1),
                ended_at=now,
                regimes=frozenset(),
                base_net_expectancy=0,
                stressed_net_expectancy=0,
                expectancy_ci_lower=0,
                daily_sharpe=0,
                profit_factor=0,
                max_drawdown=0,
                deflated_sharpe_probability=0,
                pbo=0,
                spa_pvalue=0,
                max_symbol_contribution_share=0,
                max_quarter_contribution_share=0,
                double_cost_return=0,
            )
        with pytest.raises(ValueError, match="reconciliation_rate"):
            _demo(reconciliation_rate=1.1)
        with pytest.raises(ValueError, match="오류 건수"):
            _demo(orphan_positions=-1)
