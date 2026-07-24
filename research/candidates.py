from __future__ import annotations

"""두 신규 전략군의 사전 정의 설정과 공통 연구 실행 인터페이스."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Literal, Mapping

from research.hypothesis_ledger import (
    MAX_CONFIGS_PER_FAMILY,
    HypothesisLedger,
    HypothesisSpec,
)
from src.strategy.carry_signal import CarryConfig
from src.strategy.forced_flow_signal import ForcedFlowConfig

logger = logging.getLogger(__name__)

CandidateFamily = Literal["delta_neutral_carry", "forced_flow"]
CandidateEvaluator = Callable[["CandidateExperiment"], Mapping[str, object]]

# Bybit OI는 최소 5분 완결 버킷이다. 360초는 완결 300초 + 수집 여유 60초이며,
# 4h/8h 저회전 재조정 주기에 비해 충분히 짧다. 신호 함수는 이를 넘으면 fail-closed한다.
_FORCED_FLOW_MAX_SNAPSHOT_AGE_SECONDS = 360

_UNIVERSE_POLICY: dict[str, object] = {
    "selection": "point_in_time_30d_median_quote_volume",
    "top_n": 10,
    "min_daily_quote_volume_usd": 10_000_000,
    "min_listing_days": 180,
    "venue": "bybit",
}
_COST_POLICY: dict[str, object] = {
    "fee_source": "account_fee_rate_snapshot",
    "slippage_source": "orderbook_or_demo_calibration",
    "stress_multiples": [1.0, 1.5, 2.0],
}


@dataclass(frozen=True)
class CandidateExperiment:
    """실행 전에 고정된 단일 후보 설정."""

    config_id: str
    family: CandidateFamily
    thesis: str
    features: tuple[str, ...]
    parameters: Mapping[str, object]
    primary_metric: str = "oos_daily_net_sharpe"

    def to_hypothesis(self, created_by: str) -> HypothesisSpec:
        """후보 설정을 append-only 원장 매니페스트로 변환한다."""
        universe = dict(_UNIVERSE_POLICY)
        if self.family == "delta_neutral_carry":
            universe["requires_matching_spot"] = True
        return HypothesisSpec(
            hypothesis_id=self.config_id,
            family=self.family,
            thesis=self.thesis,
            features=self.features,
            universe=universe,
            parameters=dict(self.parameters),
            costs=dict(_COST_POLICY),
            primary_metric=self.primary_metric,
            created_by=created_by,
        )


def predefined_candidates(family: CandidateFamily) -> tuple[CandidateExperiment, ...]:
    """전략군별 최대 20개의 사전 정의 후보를 반환한다."""
    candidates: list[CandidateExperiment] = []
    if family == "delta_neutral_carry":
        for intervals in (3, 6):
            for basis_capture in (0.25, 0.50):
                for cost_multiple in (1.5, 2.0):
                    config_id = (
                        f"carry-i{intervals}-b{basis_capture:.2f}"
                        f"-c{cost_multiple:.1f}"
                    )
                    candidates.append(
                        CandidateExperiment(
                            config_id=config_id,
                            family=family,
                            thesis=(
                                "양의 펀딩과 콘탱고 수렴 수익이 현물 롱·무기한 "
                                "숏의 전체 비용을 충분히 초과한다"
                            ),
                            features=(
                                "spot_price",
                                "perpetual_price",
                                "expected_funding_rate",
                            ),
                            parameters={
                                "expected_funding_intervals": intervals,
                                "basis_capture_ratio": basis_capture,
                                "min_cost_multiple": cost_multiple,
                                "max_abs_basis_rate": 0.03,
                                "max_snapshot_age_seconds": 60,
                            },
                        )
                    )
    elif family == "forced_flow":
        for rebalance_hours in (4, 8):
            for threshold in (0.35, 0.50):
                for crowding_weight in (0.15, 0.25):
                    config_id = (
                        f"flow-r{rebalance_hours}-t{threshold:.2f}"
                        f"-f{crowding_weight:.2f}"
                        f"-oi{_FORCED_FLOW_MAX_SNAPSHOT_AGE_SECONDS}"
                    )
                    candidates.append(
                        CandidateExperiment(
                            config_id=config_id,
                            family=family,
                            thesis=(
                                "OI 확장과 청산·호가 압력이 만든 저회전 방향 흐름은 "
                                "펀딩 과밀 역풍을 통제한 뒤에도 지속된다"
                            ),
                            features=(
                                "price_return",
                                "open_interest_change",
                                "funding_rate",
                                "liquidation_imbalance",
                                "orderbook_imbalance",
                                "volume_zscore",
                            ),
                            parameters={
                                "rebalance_hours": rebalance_hours,
                                "signal_threshold": threshold,
                                "price_scale": 0.02,
                                "oi_scale": 0.05,
                                "funding_scale": 0.001,
                                "min_volume_zscore": 0.0,
                                "weight_oi_price": 0.35,
                                "weight_liquidation": 0.25,
                                "weight_orderbook": 0.25,
                                "weight_funding_crowding": crowding_weight,
                                "max_snapshot_age_seconds": (
                                    _FORCED_FLOW_MAX_SNAPSHOT_AGE_SECONDS
                                ),
                            },
                        )
                    )
    else:
        raise ValueError(f"지원하지 않는 후보 전략군입니다: {family}")
    if len(candidates) > MAX_CONFIGS_PER_FAMILY:
        raise AssertionError("사전 정의 후보가 전략군별 20개 제한을 넘었습니다")
    return tuple(candidates)


def build_carry_config(
    experiment: CandidateExperiment,
    *,
    spot_fee_rate: float,
    perpetual_fee_rate: float,
    slippage_rate_per_fill: float,
) -> CarryConfig:
    """원장 후보와 시점별 실제 비용 스냅샷으로 캐리 설정을 만든다."""
    if experiment.family != "delta_neutral_carry":
        raise ValueError("델타중립 캐리 후보가 아닙니다")
    params = experiment.parameters
    return CarryConfig(
        expected_funding_intervals=int(params["expected_funding_intervals"]),
        basis_capture_ratio=float(params["basis_capture_ratio"]),
        spot_fee_rate=spot_fee_rate,
        perpetual_fee_rate=perpetual_fee_rate,
        slippage_rate_per_fill=slippage_rate_per_fill,
        min_cost_multiple=float(params["min_cost_multiple"]),
        max_abs_basis_rate=float(params["max_abs_basis_rate"]),
        max_snapshot_age=timedelta(
            seconds=float(params["max_snapshot_age_seconds"])
        ),
    )


def build_forced_flow_config(experiment: CandidateExperiment) -> ForcedFlowConfig:
    """원장 후보에서 순수 강제흐름 신호 설정을 만든다."""
    if experiment.family != "forced_flow":
        raise ValueError("강제흐름 후보가 아닙니다")
    params = experiment.parameters
    rebalance_hours = int(params["rebalance_hours"])
    if rebalance_hours not in (4, 8):
        raise ValueError("rebalance_hours는 4 또는 8이어야 합니다")
    return ForcedFlowConfig(
        rebalance_hours=rebalance_hours,
        signal_threshold=float(params["signal_threshold"]),
        price_scale=float(params["price_scale"]),
        oi_scale=float(params["oi_scale"]),
        funding_scale=float(params["funding_scale"]),
        min_volume_zscore=float(params["min_volume_zscore"]),
        weight_oi_price=float(params["weight_oi_price"]),
        weight_liquidation=float(params["weight_liquidation"]),
        weight_orderbook=float(params["weight_orderbook"]),
        weight_funding_crowding=float(params["weight_funding_crowding"]),
        max_snapshot_age=timedelta(
            seconds=float(params["max_snapshot_age_seconds"])
        ),
    )


def register_predefined_candidates(
    ledger: HypothesisLedger,
    *,
    created_by: str,
) -> dict[str, str]:
    """두 전략군의 모든 사전 후보를 실행 전에 원장에 등록한다."""
    hashes: dict[str, str] = {}
    for family in ("delta_neutral_carry", "forced_flow"):
        for experiment in predefined_candidates(family):
            hashes[experiment.config_id] = ledger.register(
                experiment.to_hypothesis(created_by)
            )
    return hashes


def evaluate_registered_candidate(
    experiment: CandidateExperiment,
    ledger: HypothesisLedger,
    evaluator: CandidateEvaluator,
    *,
    created_by: str,
) -> Mapping[str, object]:
    """원장 등록을 선행한 뒤 평가하고 성공·실패 결과를 보존한다."""
    manifest_hash = ledger.register(experiment.to_hypothesis(created_by))
    try:
        metrics = dict(evaluator(experiment))
    except Exception as exc:
        ledger.record_result(
            manifest_hash,
            "failed",
            {},
            note=f"{type(exc).__name__}: {exc}",
        )
        logger.exception("후보 평가 실패 config_id=%s", experiment.config_id)
        raise
    ledger.record_result(manifest_hash, "succeeded", metrics)
    return metrics
