from __future__ import annotations

"""사전 등록 8+8 연구 계약·캐리 재생·출력 테스트."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.candidate_replay import (
    CarryReplayOpportunity,
    FundingSettlement,
    ReplayCosts,
    replay_carry_candidate,
)
from research.candidates import all_predefined_candidates, predefined_candidates
from research.evidence_contracts import (
    BenchmarkReturnRecord,
    CandidateReplayResult,
    DailyEvidenceRecord,
    ReplayTradeRecord,
    ResearchRunManifest,
    candidate_return_matrix,
    canonical_hash,
    canonical_json,
    daily_evidence_from_trades,
)
from research.evidence_runner import EvidenceDataset, run_evidence_pipeline, write_evidence_outputs
from research.execution_constraints import InstrumentRules, ReplayExecutionPolicy
from research.hypothesis_ledger import HypothesisLedger
from research.point_in_time_universe import DailyLiquidityRecord
from research.walk_forward_splits import WalkForwardSplit
from src.strategy.evidence_decision import FeatureFreshness

PERPETUAL = "BTC/USDT:USDT"
SPOT = "BTC/USDT"


def _entry_time() -> datetime:
    """UTC 15분 경계 재생 진입 시각을 반환한다."""
    return datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)


def _costs(source: str = "bybit_account_fee_rate_api") -> ReplayCosts:
    """스냅샷 출처가 고정된 수수료를 반환한다."""
    return ReplayCosts(0.0001, 0.0001, 0.0001, source)


def _rules() -> dict[str, InstrumentRules]:
    """현물·무기한 최소수량·tick 규칙을 반환한다."""
    return {
        SPOT: InstrumentRules(SPOT, 0.001, 0.001, 0.1, 5.0),
        PERPETUAL: InstrumentRules(PERPETUAL, 0.001, 0.001, 0.1, 5.0),
    }


def _result(
    candidate_index: int,
    trades: tuple[ReplayTradeRecord, ...],
    daily: tuple[DailyEvidenceRecord, ...],
    eligible: bool,
) -> CandidateReplayResult:
    """고정 가설·run·strategy 계보가 일치하는 후보 결과를 생성한다."""
    candidate = all_predefined_candidates()[candidate_index]
    hypothesis = candidate.to_hypothesis("qa")
    hypothesis_manifest = hypothesis.manifest()
    hypothesis_hash = canonical_hash(hypothesis_manifest)
    cutoff = datetime(2025, 12, 31, tzinfo=timezone.utc)
    run = ResearchRunManifest(
        run_id=f"qa:{candidate.config_id}",
        hypothesis_hash=hypothesis_hash,
        data_hash="d" * 64,
        code_hash="c" * 64,
        fee_snapshot_hash="f" * 64,
        cost_snapshot={"source": "api"},
        data_cutoff=cutoff,
        created_at=cutoff,
    )
    strategy_hash = canonical_hash(
        {"candidate_manifest_hash": hypothesis_hash, "code_hash": "c" * 64}
    )
    return CandidateReplayResult(
        candidate_id=candidate.config_id,
        family=candidate.family,
        run_manifest_hash=run.manifest_hash,
        run_manifest=run.manifest(),
        hypothesis_hash=hypothesis_hash,
        hypothesis_manifest=hypothesis_manifest,
        code_hash="c" * 64,
        strategy_sha256=strategy_hash,
        strategy_version=candidate.config_id,
        trades=trades,
        daily=daily,
        stress_daily_returns={
            "1.0x": tuple(row.net_return for row in daily),
            "1.5x": (0.0,) * len(daily),
            "2.0x": (0.0,) * len(daily),
        },
        eligible_evidence=eligible,
        ineligibility_reasons=() if eligible else ("no_trade",),
    )


def _liquidity(entry: datetime) -> tuple[DailyLiquidityRecord, ...]:
    """진입 시점 이전 30일 유동성만 반환한다."""
    listed = entry - timedelta(days=400)
    records = []
    for offset in range(30, 0, -1):
        trade_day = entry.date() - timedelta(days=offset)
        available = datetime.combine(
            trade_day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )
        records.append(
            DailyLiquidityRecord(
                symbol=PERPETUAL,
                trade_date=trade_day,
                available_at=available,
                quote_volume_usd=50_000_000.0,
                listed_at=listed,
                has_matching_spot=True,
            )
        )
    return tuple(records)


def _split(entry: datetime) -> tuple[WalkForwardSplit, ...]:
    """진입을 OOS에 포함하는 purge·embargo 분할을 반환한다."""
    return (
        WalkForwardSplit(
            fold=0,
            train_start=entry - timedelta(days=400),
            train_end=entry - timedelta(days=2),
            purge_start=entry - timedelta(days=2),
            purge_end=entry - timedelta(days=1),
            test_start=entry - timedelta(days=1),
            test_end=entry + timedelta(days=1),
            embargo_start=entry + timedelta(days=1),
            embargo_end=entry + timedelta(days=2),
        ),
    )


def _opportunity(**overrides: object) -> CarryReplayOpportunity:
    """보유 구간의 실제 펀딩 정산을 포함한 캐리 기회를 반환한다."""
    entry = _entry_time()
    exit_time = entry + timedelta(hours=10)
    values: dict[str, object] = {
        "asset_symbol": "BTC",
        "spot_symbol": SPOT,
        "perpetual_symbol": PERPETUAL,
        "entry_time": entry,
        "exit_time": exit_time,
        "spot_entry_price": 100.0,
        "perpetual_entry_price": 101.0,
        "spot_exit_price": 100.5,
        "perpetual_exit_price": 100.8,
        "expected_funding_rate": 0.001,
        "observed_at": entry - timedelta(seconds=1),
        "funding_settlements": (
            FundingSettlement(entry + timedelta(hours=8), 0.001, 100.9),
        ),
        "requested_quantity": 1.0,
        "spot_entry_slippage_rate": 0.0001,
        "perpetual_entry_slippage_rate": 0.0001,
        "spot_exit_slippage_rate": 0.0001,
        "perpetual_exit_slippage_rate": 0.0001,
    }
    values.update(overrides)
    return CarryReplayOpportunity(**values)  # type: ignore[arg-type]


class TestCanonicalEvidenceContracts:
    """정규 JSON·manifest·CandidateReplayResult 불변 계약 검증."""

    def test_canonical_hash_is_key_order_independent_and_rejects_nan(self) -> None:
        """키 순서와 -0.0을 정규화하고 NaN을 거부한다."""
        left = {"b": [1, -0.0], "a": "한글"}
        right = {"a": "한글", "b": [1, 0.0]}
        assert canonical_json(left) == canonical_json(right)
        assert canonical_hash(left) == canonical_hash(right)
        with pytest.raises(ValueError, match="NaN"):
            canonical_json({"bad": float("nan")})

    def test_manifest_binds_hypothesis_data_code_fee_and_cutoff(self) -> None:
        """가설·데이터·코드·비용·cutoff를 하나의 재현 hash로 고정한다."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=1)
        manifest = ResearchRunManifest(
            run_id="run",
            hypothesis_hash="1" * 64,
            data_hash="2" * 64,
            code_hash="3" * 64,
            fee_snapshot_hash="4" * 64,
            cost_snapshot={"maker": 0.0002, "source": "api"},
            data_cutoff=cutoff,
            created_at=cutoff + timedelta(seconds=1),
        )
        assert len(manifest.manifest_hash) == 64
        assert canonical_hash(manifest.manifest()) == manifest.manifest_hash
        with pytest.raises(ValueError, match="data_cutoff 이전"):
            replace(manifest, created_at=cutoff - timedelta(seconds=1))

    def test_replay_result_builds_daily_and_candidate_matrix(self) -> None:
        """거래 원천으로 빈 날을 포함한 일별 자산과 후보 행렬을 만든다."""
        entry = _entry_time()
        first_candidate = all_predefined_candidates()[0]
        second_candidate = all_predefined_candidates()[8]
        trade = ReplayTradeRecord(
            candidate_id=first_candidate.config_id,
            family="delta_neutral_carry",
            fold=0,
            position_id="p1",
            symbol=PERPETUAL,
            entry_time=entry,
            exit_time=entry + timedelta(days=2),
            status="closed",
            gross_pnl=10.0,
            funding_pnl=1.0,
            fees=2.0,
            slippage=1.0,
            net_pnl=8.0,
            capital_at_entry=1000.0,
        )
        daily = daily_evidence_from_trades(
            first_candidate.config_id, (trade,), initial_capital=1000.0
        )
        assert len(daily) == 3 and daily[-1].equity == 1008.0
        first = _result(0, (trade,), daily, True)
        second_daily = tuple(
            DailyEvidenceRecord(
                second_candidate.config_id,
                row.trade_date,
                row.equity,
                -row.net_return,
            )
            for row in daily
        )
        second = _result(8, (), second_daily, False)
        matrix = candidate_return_matrix((second, first))
        assert list(matrix.columns) == sorted(
            [second_candidate.config_id, first_candidate.config_id]
        ) and matrix.shape == (3, 2)
        assert not first.trade_frame().empty and not first.daily_frame().empty
        with pytest.raises(ValueError, match="최소 두"):
            candidate_return_matrix((first,))
        with pytest.raises(ValueError, match="중복"):
            candidate_return_matrix((first, first))


class TestCarryReplay:
    """두 다리 원자성·펀딩 경계·수수료 출처 재생 검증."""

    def test_actual_funding_basis_fees_and_slippage_are_realized(self) -> None:
        """실제 정산 시각·basis 변화·양쪽 비용으로 순손익을 계산한다."""
        entry = _entry_time()
        candidate = predefined_candidates("delta_neutral_carry")[0]
        records = replay_carry_candidate(
            candidate,
            (_opportunity(),),
            costs=_costs(),
            rules_by_symbol=_rules(),
            liquidity=_liquidity(entry),
            splits=_split(entry),
            execution_policy=ReplayExecutionPolicy(2, 2.0),
            initial_capital=10_000.0,
        )
        assert len(records) == 1
        record = records[0]
        assert record.status == "closed"
        assert record.gross_pnl == pytest.approx(0.7)
        assert record.funding_pnl == pytest.approx(0.1009)
        assert record.fees > 0 and record.slippage > 0
        assert record.net_pnl == pytest.approx(
            record.gross_pnl + record.funding_pnl - record.fees - record.slippage
        )

    def test_partial_single_leg_fill_is_unwound_as_legging_failure(self) -> None:
        """한 다리만 부분 체결되면 진입을 보유하지 않고 비용 손실로 끝낸다."""
        entry = _entry_time()
        record = replay_carry_candidate(
            predefined_candidates("delta_neutral_carry")[0],
            (_opportunity(perpetual_entry_fill_ratio=0.5),),
            costs=_costs(),
            rules_by_symbol=_rules(),
            liquidity=_liquidity(entry),
            splits=_split(entry),
            execution_policy=ReplayExecutionPolicy(2, 2.0),
            initial_capital=10_000.0,
        )[0]
        assert record.status == "entry_legging_failure"
        assert record.net_pnl < 0 and record.exit_time == record.entry_time

    def test_funding_must_be_strictly_inside_holding_boundary(self) -> None:
        """진입 시각 정산과 청산 후 정산을 수익에 포함하지 않는다."""
        entry = _entry_time()
        with pytest.raises(ValueError, match=r"\(entry, exit\]"):
            _opportunity(funding_settlements=(FundingSettlement(entry, 0.001, 100.0),))
        with pytest.raises(ValueError, match=r"\(entry, exit\]"):
            _opportunity(
                funding_settlements=(FundingSettlement(entry + timedelta(hours=11), 0.001, 100.0),)
            )

    def test_only_account_fee_api_snapshot_is_promotion_eligible(self) -> None:
        """보수적 기본 요율은 연구만 가능하고 승급 증거는 되지 않는다."""
        assert _costs().promotion_eligible
        assert not _costs("conservative_default").promotion_eligible


def _empty_dataset() -> EvidenceDataset:
    """WFO는 가능하지만 진입 기회가 없는 정직한 첫 실행 데이터를 반환한다."""
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)
    start = end - timedelta(days=600)
    test_starts = (start + timedelta(days=367), start + timedelta(days=459))
    benchmark = tuple(
        BenchmarkReturnRecord(
            trade_date=(test_start + timedelta(days=index)).date(),
            benchmark_return=0.0,
            available_at=(test_start + timedelta(days=index + 1)),
        )
        for test_start in test_starts
        for index in range(90)
    )
    return EvidenceDataset(
        data_start=start,
        data_end=end,
        data_cutoff=end,
        data_hash="d" * 64,
        code_hash="c" * 64,
        created_at=end,
        feed_completeness=1.0,
        max_unresolved_gap_seconds=0.0,
        costs=_costs(),
        freshness=FeatureFreshness(
            price=timedelta(minutes=5),
            open_interest=timedelta(minutes=6),
            funding=timedelta(minutes=1),
            orderbook=timedelta(seconds=5),
            volume=timedelta(minutes=15),
            baseline_skew=timedelta(minutes=5),
        ),
        initial_capital=10_000.0,
        execution_policy=ReplayExecutionPolicy(2, 2.0),
        rules_by_symbol={},
        liquidity=(),
        benchmark_returns=benchmark,
        manifest_bindings=(SimpleNamespace(role="qa_missing"),),  # type: ignore[arg-type]
        carry=(),
        forced_flow=(),
    )


class TestFixedCandidatePipeline:
    """정확히 8+8 사전 후보와 정직한 승급 0개 출력 검증."""

    def test_candidate_registry_is_exactly_eight_per_family(self) -> None:
        """캐리 8개·강제흐름 8개의 고유 설정만 평가한다."""
        candidates = all_predefined_candidates()
        assert len(candidates) == 16
        assert sum(x.family == "delta_neutral_carry" for x in candidates) == 8
        assert sum(x.family == "forced_flow" for x in candidates) == 8
        assert len({x.config_id for x in candidates}) == 16

    def test_empty_first_run_outputs_sixteen_failures_and_zero_eligible(self, tmp_path: Path) -> None:
        """거래 없는 첫 실행을 매개변수로 보정하지 않고 전부 실패로 남긴다."""
        ledger = HypothesisLedger(tmp_path / "ledger.jsonl")
        results = run_evidence_pipeline(
            _empty_dataset(), ledger=ledger, run_id="first", created_by="qa"
        )
        assert len(results) == 16
        assert all(not result.eligible_evidence for result in results)
        assert all(
            any(reason.startswith("insufficient_data:") for reason in result.ineligibility_reasons)
            for result in results
        )
        output = tmp_path / "out"
        write_evidence_outputs(
            results,
            output,
            benchmark_returns=_empty_dataset().benchmark_returns,
        )
        summary = json.loads((output / "evidence_summary.json").read_text(encoding="utf-8"))
        assert summary["candidate_count"] == 16
        assert summary["eligible_strategy_count"] == 0
        assert summary["evidence_admissible_count"] == 0
        assert (output / "candidate_matrix.csv").exists()
