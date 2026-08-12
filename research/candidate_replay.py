from __future__ import annotations

"""사전 등록 캐리·강제흐름 후보의 비용 포함 시점 안전 재생."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

from research.candidates import (
    CandidateExperiment,
    build_carry_config,
    build_forced_flow_config,
)
from research.evidence_contracts import ReplayTradeRecord
from research.execution_constraints import (
    InstrumentRules,
    ReplayExecutionPolicy,
    ReplayOrderLeg,
    ReplayPortfolioState,
    ReplayPosition,
    ReplayTradeIntent,
    apply_execution_constraints,
)
from research.point_in_time_universe import (
    DailyLiquidityRecord,
    select_point_in_time_universe,
)
from research.walk_forward_splits import WalkForwardSplit
from src.strategy.carry_signal import CarrySnapshot
from src.strategy.decision import DecisionContext
from src.strategy.evidence_decision import (
    FeatureFreshness,
    ForcedFlowFeatureBundle,
    StrategyTradeIntent,
    decide_carry_intent,
    decide_forced_flow_intent,
)


def _utc(value: datetime, field_name: str) -> datetime:
    """시간대가 있는 시각을 UTC로 정규화한다."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}은(는) timezone-aware여야 합니다")
    return value.astimezone(timezone.utc)


def _rate(value: float, field_name: str, *, upper: float | None = None) -> float:
    """0 이상의 유한한 비율을 검증한다."""
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name}은(는) 0 이상의 유한한 비율이어야 합니다")
    if upper is not None and normalized > upper:
        raise ValueError(f"{field_name}은(는) {upper} 이하여야 합니다")
    return normalized


@dataclass(frozen=True)
class ReplayCosts:
    """한 연구 실행에서 고정한 계정 수수료와 보수적 슬리피지."""

    spot_fee_rate: float
    perpetual_fee_rate: float
    assumed_slippage_rate_per_fill: float
    fee_source: str

    def __post_init__(self) -> None:
        """비용률과 수수료 출처를 검증한다."""
        object.__setattr__(self, "spot_fee_rate", _rate(self.spot_fee_rate, "spot_fee_rate"))
        object.__setattr__(
            self,
            "perpetual_fee_rate",
            _rate(self.perpetual_fee_rate, "perpetual_fee_rate"),
        )
        object.__setattr__(
            self,
            "assumed_slippage_rate_per_fill",
            _rate(self.assumed_slippage_rate_per_fill, "assumed_slippage_rate_per_fill"),
        )
        if not self.fee_source.strip():
            raise ValueError("fee_source는 비어 있을 수 없습니다")

    @property
    def promotion_eligible(self) -> bool:
        """실계정 Fee Rate API 스냅샷인지 반환한다."""
        return self.fee_source == "bybit_account_fee_rate_api"


@dataclass(frozen=True)
class FundingSettlement:
    """보유 중 실제 정산된 펀딩 시각·요율·무기한 mark 가격."""

    timestamp: datetime
    rate: float
    perpetual_mark_price: float

    def __post_init__(self) -> None:
        """정산 시각과 숫자를 검증한다."""
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "timestamp"))
        if not math.isfinite(self.rate):
            raise ValueError("funding rate는 유한해야 합니다")
        if not math.isfinite(self.perpetual_mark_price) or self.perpetual_mark_price <= 0:
            raise ValueError("funding mark 가격은 양의 유한한 숫자여야 합니다")


@dataclass(frozen=True)
class CarryReplayOpportunity:
    """캐리 진입 판단부터 청산까지의 시점 분리 원천 데이터."""

    asset_symbol: str
    spot_symbol: str
    perpetual_symbol: str
    entry_time: datetime
    exit_time: datetime
    spot_entry_price: float
    perpetual_entry_price: float
    spot_exit_price: float
    perpetual_exit_price: float
    expected_funding_rate: float
    observed_at: datetime
    funding_settlements: tuple[FundingSettlement, ...]
    requested_quantity: float
    spot_entry_fill_ratio: float = 1.0
    perpetual_entry_fill_ratio: float = 1.0
    spot_exit_fill_ratio: float = 1.0
    perpetual_exit_fill_ratio: float = 1.0
    spot_entry_slippage_rate: float = 0.0
    perpetual_entry_slippage_rate: float = 0.0
    spot_exit_slippage_rate: float = 0.0
    perpetual_exit_slippage_rate: float = 0.0

    def __post_init__(self) -> None:
        """시각·가격·체결률·슬리피지를 검증한다."""
        entry = _utc(self.entry_time, "entry_time")
        exit_time = _utc(self.exit_time, "exit_time")
        observed = _utc(self.observed_at, "observed_at")
        if not all((self.asset_symbol.strip(), self.spot_symbol.strip(), self.perpetual_symbol.strip())):
            raise ValueError("캐리 심볼은 비어 있을 수 없습니다")
        if exit_time <= entry:
            raise ValueError("캐리 exit_time은 entry_time 이후여야 합니다")
        if observed > entry:
            raise ValueError("캐리 관측값은 진입 판단 이후일 수 없습니다")
        prices = (
            self.spot_entry_price,
            self.perpetual_entry_price,
            self.spot_exit_price,
            self.perpetual_exit_price,
            self.requested_quantity,
        )
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise ValueError("캐리 가격과 요청 수량은 양수여야 합니다")
        if not math.isfinite(self.expected_funding_rate):
            raise ValueError("expected_funding_rate는 유한해야 합니다")
        for field_name in (
            "spot_entry_fill_ratio",
            "perpetual_entry_fill_ratio",
            "spot_exit_fill_ratio",
            "perpetual_exit_fill_ratio",
        ):
            object.__setattr__(self, field_name, _rate(getattr(self, field_name), field_name, upper=1.0))
        for field_name in (
            "spot_entry_slippage_rate",
            "perpetual_entry_slippage_rate",
            "spot_exit_slippage_rate",
            "perpetual_exit_slippage_rate",
        ):
            object.__setattr__(self, field_name, _rate(getattr(self, field_name), field_name))
        settlements = tuple(sorted(self.funding_settlements, key=lambda item: item.timestamp))
        if len({item.timestamp for item in settlements}) != len(settlements):
            raise ValueError("동일 시각 funding settlement가 중복됐습니다")
        if any(not (entry < item.timestamp <= exit_time) for item in settlements):
            raise ValueError("funding settlement는 실제 보유 구간 (entry, exit] 안이어야 합니다")
        object.__setattr__(self, "entry_time", entry)
        object.__setattr__(self, "exit_time", exit_time)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "funding_settlements", settlements)


@dataclass(frozen=True)
class ForcedFlowReplayOpportunity:
    """강제흐름 결정 입력과 이후 완결 청산 가격을 분리한 재생 건."""

    perpetual_symbol: str
    decision_time: datetime
    exit_time: datetime
    feature_bundle: ForcedFlowFeatureBundle
    exit_price: float
    requested_quantity: float
    entry_fill_ratio: float = 1.0
    exit_fill_ratio: float = 1.0
    entry_slippage_rate: float = 0.0
    exit_slippage_rate: float = 0.0

    def __post_init__(self) -> None:
        """결정·청산 시각과 체결 가정을 검증한다."""
        decision = _utc(self.decision_time, "decision_time")
        exit_time = _utc(self.exit_time, "exit_time")
        if not self.perpetual_symbol.strip():
            raise ValueError("perpetual_symbol은 비어 있을 수 없습니다")
        if exit_time <= decision:
            raise ValueError("강제흐름 exit_time은 decision_time 이후여야 합니다")
        if not math.isfinite(self.exit_price) or self.exit_price <= 0:
            raise ValueError("exit_price는 양의 유한한 숫자여야 합니다")
        if not math.isfinite(self.requested_quantity) or self.requested_quantity <= 0:
            raise ValueError("requested_quantity는 양의 유한한 숫자여야 합니다")
        object.__setattr__(self, "entry_fill_ratio", _rate(self.entry_fill_ratio, "entry_fill_ratio", upper=1.0))
        object.__setattr__(self, "exit_fill_ratio", _rate(self.exit_fill_ratio, "exit_fill_ratio", upper=1.0))
        object.__setattr__(self, "entry_slippage_rate", _rate(self.entry_slippage_rate, "entry_slippage_rate"))
        object.__setattr__(self, "exit_slippage_rate", _rate(self.exit_slippage_rate, "exit_slippage_rate"))
        object.__setattr__(self, "decision_time", decision)
        object.__setattr__(self, "exit_time", exit_time)


def _fold_for_interval(
    entry_time: datetime,
    exit_time: datetime,
    splits: tuple[WalkForwardSplit, ...],
) -> int | None:
    """진입은 OOS, 청산은 해당 embargo 안인 fold 번호를 반환한다."""
    for split in splits:
        if (
            split.test_start <= entry_time < split.test_end
            and entry_time < exit_time <= split.embargo_end
        ):
            return split.fold
    return None


def _allowed_by_universe(
    symbol: str,
    timestamp: datetime,
    liquidity: Iterable[DailyLiquidityRecord],
    *,
    carry_only: bool,
) -> bool:
    """시점별 유동성 유니버스에 심볼이 포함됐는지 반환한다."""
    selection = select_point_in_time_universe(
        liquidity,
        timestamp,
        carry_only=carry_only,
    )
    return symbol in selection.symbols


def _constraint_intent(intent: StrategyTradeIntent) -> ReplayTradeIntent:
    """공용 전략 의도를 연구 제약 입력으로 변환한다."""
    return ReplayTradeIntent(
        position_id=f"{intent.candidate_id}:{intent.context.run_id}",
        legs=tuple(
            ReplayOrderLeg(
                symbol=leg.symbol,
                side=leg.side,
                requested_quantity=leg.requested_quantity,
                reference_price=leg.reference_price,
            )
            for leg in intent.legs
        ),
    )


def _state_at(
    timestamp: datetime,
    active: list[tuple[datetime, ReplayPosition]],
    capital: float,
) -> ReplayPortfolioState:
    """종료된 포지션을 제거하고 현재 제약 상태를 반환한다."""
    active[:] = [item for item in active if item[0] > timestamp]
    return ReplayPortfolioState(
        capital=capital,
        positions=tuple(item[1] for item in active),
    )


def _realize_until(
    timestamp: datetime,
    pending: list[tuple[datetime, float]],
    capital: float,
) -> float:
    """현재 시각까지 실제 종료된 손익만 자본에 반영한다."""
    realized = sum(pnl for exit_time, pnl in pending if exit_time <= timestamp)
    pending[:] = [item for item in pending if item[0] > timestamp]
    updated = capital + realized
    if updated <= 0:
        raise ValueError("재생 자본이 0 이하가 됐습니다")
    return updated


def _rejected_record(
    experiment: CandidateExperiment,
    fold: int,
    position_id: str,
    symbol: str,
    entry_time: datetime,
    exit_time: datetime,
    capital: float,
    status: str,
) -> ReplayTradeRecord:
    """손익 0의 거절 기록을 만든다."""
    return ReplayTradeRecord(
        candidate_id=experiment.config_id,
        family=experiment.family,
        fold=fold,
        position_id=position_id,
        symbol=symbol,
        entry_time=entry_time,
        exit_time=exit_time,
        status=status,  # type: ignore[arg-type]
        gross_pnl=0.0,
        funding_pnl=0.0,
        fees=0.0,
        slippage=0.0,
        net_pnl=0.0,
        capital_at_entry=capital,
    )


def replay_carry_candidate(
    experiment: CandidateExperiment,
    opportunities: Iterable[CarryReplayOpportunity],
    *,
    costs: ReplayCosts,
    rules_by_symbol: Mapping[str, InstrumentRules],
    liquidity: Iterable[DailyLiquidityRecord],
    splits: tuple[WalkForwardSplit, ...],
    execution_policy: ReplayExecutionPolicy,
    initial_capital: float,
) -> tuple[ReplayTradeRecord, ...]:
    """실제 funding 시각·basis·양쪽 비용을 포함해 캐리 후보를 재생한다."""
    if experiment.family != "delta_neutral_carry":
        raise ValueError("캐리 후보만 replay_carry_candidate로 평가할 수 있습니다")
    config = build_carry_config(
        experiment,
        spot_fee_rate=costs.spot_fee_rate,
        perpetual_fee_rate=costs.perpetual_fee_rate,
        slippage_rate_per_fill=costs.assumed_slippage_rate_per_fill,
    )
    liquidity_records = tuple(liquidity)
    active: list[tuple[datetime, ReplayPosition]] = []
    pending: list[tuple[datetime, float]] = []
    records: list[ReplayTradeRecord] = []
    capital = initial_capital
    for opportunity in sorted(opportunities, key=lambda item: (item.entry_time, item.asset_symbol)):
        capital = _realize_until(opportunity.entry_time, pending, capital)
        fold = _fold_for_interval(
            opportunity.entry_time,
            opportunity.exit_time,
            splits,
        )
        if fold is None:
            continue
        if not _allowed_by_universe(
            opportunity.perpetual_symbol,
            opportunity.entry_time,
            liquidity_records,
            carry_only=True,
        ):
            continue
        context = DecisionContext.for_closed_bar(
            opportunity.entry_time,
            strategy_version=experiment.config_id,
            run_id=f"fold-{fold}:{opportunity.asset_symbol}:{opportunity.entry_time.isoformat()}",
            decision_time=opportunity.entry_time,
            data_cutoff=opportunity.entry_time,
        )
        snapshot = CarrySnapshot(
            symbol=opportunity.asset_symbol,
            spot_price=opportunity.spot_entry_price,
            perpetual_price=opportunity.perpetual_entry_price,
            expected_funding_rate=opportunity.expected_funding_rate,
            observed_at=opportunity.observed_at,
        )
        intent = decide_carry_intent(
            snapshot,
            config,
            context,
            candidate_id=experiment.config_id,
            spot_symbol=opportunity.spot_symbol,
            perpetual_symbol=opportunity.perpetual_symbol,
            requested_quantity=opportunity.requested_quantity,
        )
        position_id = f"{experiment.config_id}:{context.run_id}"
        if intent is None:
            records.append(
                _rejected_record(
                    experiment,
                    fold,
                    position_id,
                    opportunity.perpetual_symbol,
                    opportunity.entry_time,
                    opportunity.entry_time,
                    capital,
                    "signal_rejected",
                )
            )
            continue
        constraint = apply_execution_constraints(
            _constraint_intent(intent),
            rules_by_symbol,
            _state_at(opportunity.entry_time, active, capital),
            execution_policy,
        )
        if not constraint.accepted:
            records.append(
                _rejected_record(
                    experiment,
                    fold,
                    position_id,
                    opportunity.perpetual_symbol,
                    opportunity.entry_time,
                    opportunity.entry_time,
                    capital,
                    "constraint_rejected",
                )
            )
            continue
        spot_leg, perpetual_leg = constraint.legs
        entry_ratios = (
            opportunity.spot_entry_fill_ratio,
            opportunity.perpetual_entry_fill_ratio,
        )
        if entry_ratios == (0.0, 0.0):
            records.append(
                _rejected_record(
                    experiment,
                    fold,
                    position_id,
                    opportunity.perpetual_symbol,
                    opportunity.entry_time,
                    opportunity.entry_time,
                    capital,
                    "entry_unfilled",
                )
            )
            continue
        if entry_ratios != (1.0, 1.0) or not math.isclose(
            spot_leg.quantity,
            perpetual_leg.quantity,
            abs_tol=1e-8,
        ):
            spot_filled = spot_leg.quantity * entry_ratios[0]
            perpetual_filled = perpetual_leg.quantity * entry_ratios[1]
            spot_notional = spot_filled * opportunity.spot_entry_price
            perpetual_notional = perpetual_filled * opportunity.perpetual_entry_price
            fees = 2.0 * (
                spot_notional * costs.spot_fee_rate
                + perpetual_notional * costs.perpetual_fee_rate
            )
            slippage = 2.0 * (
                spot_notional * opportunity.spot_entry_slippage_rate
                + perpetual_notional * opportunity.perpetual_entry_slippage_rate
            )
            net = -(fees + slippage)
            record = ReplayTradeRecord(
                candidate_id=experiment.config_id,
                family=experiment.family,
                fold=fold,
                position_id=position_id,
                symbol=opportunity.perpetual_symbol,
                entry_time=opportunity.entry_time,
                exit_time=opportunity.entry_time,
                status="entry_legging_failure",
                gross_pnl=0.0,
                funding_pnl=0.0,
                fees=round(fees, 8),
                slippage=round(slippage, 8),
                net_pnl=round(net, 8),
                capital_at_entry=capital,
            )
            records.append(record)
            capital += record.net_pnl
            continue

        quantity = min(spot_leg.quantity, perpetual_leg.quantity)
        gross_pnl = quantity * (
            opportunity.spot_exit_price
            - opportunity.spot_entry_price
            + opportunity.perpetual_entry_price
            - opportunity.perpetual_exit_price
        )
        funding_pnl = quantity * sum(
            settlement.perpetual_mark_price * settlement.rate
            for settlement in opportunity.funding_settlements
        )
        entry_exit_notionals = (
            quantity * opportunity.spot_entry_price,
            quantity * opportunity.perpetual_entry_price,
            quantity * opportunity.spot_exit_price,
            quantity * opportunity.perpetual_exit_price,
        )
        fees = costs.spot_fee_rate * (entry_exit_notionals[0] + entry_exit_notionals[2])
        fees += costs.perpetual_fee_rate * (entry_exit_notionals[1] + entry_exit_notionals[3])
        exit_ratios = (
            opportunity.spot_exit_fill_ratio,
            opportunity.perpetual_exit_fill_ratio,
        )
        spot_exit_multiplier = 1.0 + (1.0 - exit_ratios[0])
        perp_exit_multiplier = 1.0 + (1.0 - exit_ratios[1])
        slippage = (
            entry_exit_notionals[0] * opportunity.spot_entry_slippage_rate
            + entry_exit_notionals[1] * opportunity.perpetual_entry_slippage_rate
            + entry_exit_notionals[2]
            * opportunity.spot_exit_slippage_rate
            * spot_exit_multiplier
            + entry_exit_notionals[3]
            * opportunity.perpetual_exit_slippage_rate
            * perp_exit_multiplier
        )
        status = "closed" if exit_ratios == (1.0, 1.0) else "exit_forced"
        net_pnl = gross_pnl + funding_pnl - fees - slippage
        record = ReplayTradeRecord(
            candidate_id=experiment.config_id,
            family=experiment.family,
            fold=fold,
            position_id=position_id,
            symbol=opportunity.perpetual_symbol,
            entry_time=opportunity.entry_time,
            exit_time=opportunity.exit_time,
            status=status,
            gross_pnl=round(gross_pnl, 8),
            funding_pnl=round(funding_pnl, 8),
            fees=round(fees, 8),
            slippage=round(slippage, 8),
            net_pnl=round(net_pnl, 8),
            capital_at_entry=capital,
        )
        records.append(record)
        active.append(
            (
                opportunity.exit_time,
                ReplayPosition(
                    position_id=position_id,
                    gross_notional=constraint.gross_notional,
                    symbols=(opportunity.spot_symbol, opportunity.perpetual_symbol),
                ),
            )
        )
        pending.append((opportunity.exit_time, record.net_pnl))
    return tuple(sorted(records, key=lambda item: (item.entry_time, item.position_id)))


def replay_forced_flow_candidate(
    experiment: CandidateExperiment,
    opportunities: Iterable[ForcedFlowReplayOpportunity],
    *,
    costs: ReplayCosts,
    freshness: FeatureFreshness,
    rules_by_symbol: Mapping[str, InstrumentRules],
    liquidity: Iterable[DailyLiquidityRecord],
    splits: tuple[WalkForwardSplit, ...],
    execution_policy: ReplayExecutionPolicy,
    initial_capital: float,
) -> tuple[ReplayTradeRecord, ...]:
    """공용 as-of 특징·제약으로 강제흐름 후보를 OOS 재생한다."""
    if experiment.family != "forced_flow":
        raise ValueError("강제흐름 후보만 replay_forced_flow_candidate로 평가할 수 있습니다")
    config = build_forced_flow_config(experiment)
    liquidity_records = tuple(liquidity)
    active: list[tuple[datetime, ReplayPosition]] = []
    pending: list[tuple[datetime, float]] = []
    records: list[ReplayTradeRecord] = []
    capital = initial_capital
    last_rebalance_by_symbol: dict[str, datetime] = {}
    for opportunity in sorted(opportunities, key=lambda item: (item.decision_time, item.perpetual_symbol)):
        capital = _realize_until(opportunity.decision_time, pending, capital)
        fold = _fold_for_interval(
            opportunity.decision_time,
            opportunity.exit_time,
            splits,
        )
        if fold is None:
            continue
        if not _allowed_by_universe(
            opportunity.perpetual_symbol,
            opportunity.decision_time,
            liquidity_records,
            carry_only=False,
        ):
            continue
        context = DecisionContext.for_closed_bar(
            opportunity.decision_time,
            strategy_version=experiment.config_id,
            run_id=f"fold-{fold}:{opportunity.perpetual_symbol}:{opportunity.decision_time.isoformat()}",
            decision_time=opportunity.decision_time,
            data_cutoff=opportunity.decision_time,
        )
        intent = decide_forced_flow_intent(
            opportunity.feature_bundle,
            config,
            context,
            freshness=freshness,
            candidate_id=experiment.config_id,
            perpetual_symbol=opportunity.perpetual_symbol,
            requested_quantity=opportunity.requested_quantity,
            last_rebalance_time=last_rebalance_by_symbol.get(opportunity.perpetual_symbol),
        )
        position_id = f"{experiment.config_id}:{context.run_id}"
        if intent is None:
            records.append(
                _rejected_record(
                    experiment,
                    fold,
                    position_id,
                    opportunity.perpetual_symbol,
                    opportunity.decision_time,
                    opportunity.decision_time,
                    capital,
                    "signal_rejected",
                )
            )
            continue
        constraint = apply_execution_constraints(
            _constraint_intent(intent),
            rules_by_symbol,
            _state_at(opportunity.decision_time, active, capital),
            execution_policy,
        )
        if not constraint.accepted:
            records.append(
                _rejected_record(
                    experiment,
                    fold,
                    position_id,
                    opportunity.perpetual_symbol,
                    opportunity.decision_time,
                    opportunity.decision_time,
                    capital,
                    "constraint_rejected",
                )
            )
            continue
        leg = constraint.legs[0]
        if opportunity.entry_fill_ratio == 0:
            records.append(
                _rejected_record(
                    experiment,
                    fold,
                    position_id,
                    opportunity.perpetual_symbol,
                    opportunity.decision_time,
                    opportunity.decision_time,
                    capital,
                    "entry_unfilled",
                )
            )
            continue
        filled_quantity = leg.quantity * opportunity.entry_fill_ratio
        direction_sign = 1.0 if intent.direction == "long" else -1.0
        gross_pnl = direction_sign * filled_quantity * (opportunity.exit_price - leg.price)
        entry_notional = filled_quantity * leg.price
        exit_notional = filled_quantity * opportunity.exit_price
        fees = (entry_notional + exit_notional) * costs.perpetual_fee_rate
        exit_multiplier = 1.0 + (1.0 - opportunity.exit_fill_ratio)
        slippage = (
            entry_notional * opportunity.entry_slippage_rate
            + exit_notional * opportunity.exit_slippage_rate * exit_multiplier
        )
        net_pnl = gross_pnl - fees - slippage
        status = "closed" if opportunity.exit_fill_ratio == 1.0 else "exit_forced"
        record = ReplayTradeRecord(
            candidate_id=experiment.config_id,
            family=experiment.family,
            fold=fold,
            position_id=position_id,
            symbol=opportunity.perpetual_symbol,
            entry_time=opportunity.decision_time,
            exit_time=opportunity.exit_time,
            status=status,
            gross_pnl=round(gross_pnl, 8),
            funding_pnl=0.0,
            fees=round(fees, 8),
            slippage=round(slippage, 8),
            net_pnl=round(net_pnl, 8),
            capital_at_entry=capital,
        )
        records.append(record)
        active.append(
            (
                opportunity.exit_time,
                ReplayPosition(
                    position_id=position_id,
                    gross_notional=constraint.gross_notional,
                    symbols=(opportunity.perpetual_symbol,),
                ),
            )
        )
        last_rebalance_by_symbol[opportunity.perpetual_symbol] = opportunity.decision_time
        pending.append((opportunity.exit_time, record.net_pnl))
    return tuple(sorted(records, key=lambda item: (item.entry_time, item.position_id)))
