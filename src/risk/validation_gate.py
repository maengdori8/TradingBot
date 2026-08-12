from __future__ import annotations

# 오프라인·미래 데모 통계 승급 게이트.

import hashlib
import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
REQUIRED_REGIMES = frozenset({"bull", "bear", "sideways", "high_volatility"})

_REGIME_ALIASES = {
    "상승": "bull",
    "하락": "bear",
    "횡보": "sideways",
    "고변동": "high_volatility",
    "high-volatility": "high_volatility",
    "high_vol": "high_volatility",
}


@dataclass(frozen=True)
class GateCriterion:
    """통계 승급 게이트의 단일 판정."""

    name: str
    passed: bool
    value: float | int | bool | str
    threshold: float | int | bool | str


@dataclass(frozen=True)
class GateDecision:
    """단계별 승급 판정 결과."""

    passed: bool
    stage: str
    strategy_id: str
    strategy_version: str
    criteria: Mapping[str, GateCriterion]
    summary: str
    informational_only: bool = False

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        """미통과 기준 키를 반환한다."""
        return tuple(
            key for key, criterion in self.criteria.items() if not criterion.passed
        )


@dataclass(frozen=True)
class OfflineValidationEvidence:
    """오프라인 OOS 승급 판정에 필요한 사전 계산 증거."""

    strategy_id: str
    strategy_version: str
    effective_bets: int
    started_at: datetime
    ended_at: datetime
    regimes: frozenset[str]
    base_net_expectancy: float
    stressed_net_expectancy: float
    expectancy_ci_lower: float
    daily_sharpe: float
    profit_factor: float
    max_drawdown: float
    deflated_sharpe_probability: float
    pbo: float
    spa_pvalue: float
    max_symbol_contribution_share: float
    max_quarter_contribution_share: float
    double_cost_return: float
    strategy_logic_intact: bool = True
    hypothesis_configs: int = 1
    cost_stress_multiplier: float = 1.5
    double_cost_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """시간·표본·확률 값의 기본 계약을 검증한다."""
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("검증 시작·종료 시각은 timezone-aware여야 합니다.")
        if self.ended_at <= self.started_at:
            raise ValueError("검증 종료 시각은 시작 시각 이후여야 합니다.")
        if self.effective_bets < 0 or self.hypothesis_configs < 0:
            raise ValueError("표본 수와 설정 수는 음수일 수 없습니다.")
        for name in (
            "deflated_sharpe_probability",
            "pbo",
            "spa_pvalue",
            "max_symbol_contribution_share",
            "max_quarter_contribution_share",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name}은 0과 1 사이여야 합니다.")

    @property
    def calendar_days(self) -> int:
        """검증 구간 달력 일수를 반환한다."""
        return (self.ended_at - self.started_at).days

    @property
    def normalized_regimes(self) -> frozenset[str]:
        """한국어·별칭 레짐을 표준 키로 정규화한다."""
        return frozenset(_REGIME_ALIASES.get(item, item) for item in self.regimes)


@dataclass(frozen=True)
class DemoValidationEvidence:
    """미래 데이터 데모 승급 판정에 필요한 증거."""

    strategy_id: str
    strategy_version: str
    calendar_days: int
    effective_bets: int
    expectancy_ci_lower: float
    daily_sharpe: float
    profit_factor: float
    max_drawdown: float
    fill_error_median_bps: float
    fill_error_p95_bps: float
    fill_rate_error: float
    reconciliation_rate: float
    orphan_positions: int
    duplicate_orders: int
    parameters_unchanged: bool

    def __post_init__(self) -> None:
        """데모 증거의 표본과 비율 범위를 검증한다."""
        if self.calendar_days < 0 or self.effective_bets < 0:
            raise ValueError("기간과 표본 수는 음수일 수 없습니다.")
        if not 0 <= self.reconciliation_rate <= 1:
            raise ValueError("reconciliation_rate는 0과 1 사이여야 합니다.")
        if self.orphan_positions < 0 or self.duplicate_orders < 0:
            raise ValueError("오류 건수는 음수일 수 없습니다.")


@dataclass(frozen=True)
class DatedTradeReturn:
    """거래 단위 수익과 비용 스트레스를 담는 원시 검증 레코드."""

    trade_id: str
    closed_at: datetime
    symbol: str
    net_return: float
    stressed_return: float
    double_cost_return: float

    def __post_init__(self) -> None:
        """원시 거래 레코드의 시각·식별자·수익률을 검증한다."""
        if not self.trade_id.strip() or not self.symbol.strip():
            raise ValueError("trade_id와 symbol은 비어 있을 수 없습니다.")
        if self.closed_at.tzinfo is None:
            raise ValueError("closed_at은 timezone-aware여야 합니다.")
        values = (self.net_return, self.stressed_return, self.double_cost_return)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("거래 수익률은 유한한 값이어야 합니다.")
        if any(float(value) <= -1 for value in values):
            raise ValueError("거래 수익률은 -100% 이하일 수 없습니다.")


@dataclass(frozen=True)
class DatedCandidateReturns:
    """UTC 일별 선택 전략·벤치마크·전체 후보 수익률 레코드."""

    observed_at: datetime
    strategy_return: float
    benchmark_return: float
    candidate_returns: Mapping[str, float]

    def __post_init__(self) -> None:
        """일별 레코드와 후보 행렬 한 행을 검증하고 불변화한다."""
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at은 timezone-aware여야 합니다.")
        if not np.isfinite(float(self.strategy_return)) or not np.isfinite(
            float(self.benchmark_return)
        ):
            raise ValueError("일별 수익률은 유한한 값이어야 합니다.")
        if self.strategy_return <= -1 or self.benchmark_return <= -1:
            raise ValueError("일별 수익률은 -100% 이하일 수 없습니다.")
        normalized: dict[str, float] = {}
        for candidate_id, value in self.candidate_returns.items():
            key = str(candidate_id).strip()
            numeric = float(value)
            if not key or key in normalized or not np.isfinite(numeric):
                raise ValueError("후보 ID와 수익률이 올바르지 않습니다.")
            if numeric <= -1:
                raise ValueError("후보 일별 수익률은 -100% 이하일 수 없습니다.")
            normalized[key] = numeric
        if len(normalized) < 2:
            raise ValueError("후보 행렬에는 서로 다른 설정이 2개 이상 필요합니다.")
        object.__setattr__(
            self,
            "candidate_returns",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


def _load_validation_config() -> dict:
    """config.yaml의 validation 섹션을 로드한다."""
    try:
        with open(ROOT / "config" / "config.yaml", encoding="utf-8") as file:
            return (yaml.safe_load(file) or {}).get("validation", {})
    except FileNotFoundError:
        logger.warning("validation 설정 파일이 없어 보수적 기본값을 사용합니다.")
        return {}


class OfflinePromotionGate:
    """OOS 후보를 미래 데모 단계로 보낼 수 있는지 판정한다."""

    def __init__(
        self,
        config: Mapping[str, float | int] | None = None,
        frozen_strategy_ids: Iterable[str] = ("ict-benchmark-v1",),
        max_configs_per_family: int | None = None,
    ) -> None:
        """오프라인 게이트 기준을 초기화한다."""
        validation = _load_validation_config()
        self.config = dict(config or validation.get("offline", {}))
        self.frozen_strategy_ids = frozenset(frozen_strategy_ids)
        self.max_configs_per_family = int(
            max_configs_per_family
            or validation.get("hypothesis_max_configs_per_family", 20)
        )

    def evaluate(self, evidence: OfflineValidationEvidence) -> GateDecision:
        """오프라인 증거가 모든 승급 기준을 동시에 만족하는지 판정한다."""
        cfg = self.config
        min_bets = int(cfg.get("min_effective_bets", 200))
        min_days = int(round(float(cfg.get("min_months", 12)) * 365 / 12))
        max_contribution = float(cfg.get("max_contribution_share", 0.25))
        max_double_loss = float(cfg.get("max_double_cost_loss", 0.10))
        missing_regimes = REQUIRED_REGIMES - evidence.normalized_regimes
        criteria = {
            "strategy_not_frozen": GateCriterion(
                "동결되지 않은 전략",
                evidence.strategy_id not in self.frozen_strategy_ids,
                evidence.strategy_id,
                "not frozen",
            ),
            "effective_bets": GateCriterion(
                "유효 OOS 베팅 수",
                evidence.effective_bets >= min_bets,
                evidence.effective_bets,
                min_bets,
            ),
            "calendar_days": GateCriterion(
                "OOS 검증 기간",
                evidence.calendar_days >= min_days,
                evidence.calendar_days,
                min_days,
            ),
            "regimes": GateCriterion(
                "필수 시장 레짐",
                not missing_regimes,
                ",".join(sorted(evidence.normalized_regimes)),
                ",".join(sorted(REQUIRED_REGIMES)),
            ),
            "base_expectancy": GateCriterion(
                "기본 비용 순기대값",
                evidence.base_net_expectancy > float(cfg.get("min_net_expectancy", 0)),
                evidence.base_net_expectancy,
                f">{float(cfg.get('min_net_expectancy', 0))}",
            ),
            "stress_expectancy": GateCriterion(
                "비용 스트레스 순기대값",
                evidence.stressed_net_expectancy > 0,
                evidence.stressed_net_expectancy,
                ">0",
            ),
            "stress_multiplier": GateCriterion(
                "비용 스트레스 배수",
                evidence.cost_stress_multiplier
                >= float(cfg.get("cost_stress_mult", 1.5)),
                evidence.cost_stress_multiplier,
                float(cfg.get("cost_stress_mult", 1.5)),
            ),
            "expectancy_ci": GateCriterion(
                "클러스터 부트스트랩 기대값 95% 하한",
                evidence.expectancy_ci_lower
                > float(cfg.get("min_expectancy_ci_lower", 0)),
                evidence.expectancy_ci_lower,
                f">{float(cfg.get('min_expectancy_ci_lower', 0))}",
            ),
            "daily_sharpe": GateCriterion(
                "UTC 일별 Sharpe",
                evidence.daily_sharpe >= float(cfg.get("min_daily_sharpe", 1)),
                evidence.daily_sharpe,
                float(cfg.get("min_daily_sharpe", 1)),
            ),
            "profit_factor": GateCriterion(
                "Profit Factor",
                evidence.profit_factor >= float(cfg.get("min_profit_factor", 1.2)),
                evidence.profit_factor,
                float(cfg.get("min_profit_factor", 1.2)),
            ),
            "max_drawdown": GateCriterion(
                "최대 낙폭",
                evidence.max_drawdown <= float(cfg.get("max_mdd", 0.10)),
                evidence.max_drawdown,
                float(cfg.get("max_mdd", 0.10)),
            ),
            "deflated_sharpe": GateCriterion(
                "Deflated Sharpe 양의 확률",
                evidence.deflated_sharpe_probability
                >= float(cfg.get("min_deflated_sharpe_probability", 0.95)),
                evidence.deflated_sharpe_probability,
                float(cfg.get("min_deflated_sharpe_probability", 0.95)),
            ),
            "pbo": GateCriterion(
                "PBO",
                evidence.pbo < float(cfg.get("max_pbo", 0.10)),
                evidence.pbo,
                f"<{float(cfg.get('max_pbo', 0.10))}",
            ),
            "spa": GateCriterion(
                "SPA p-value",
                evidence.spa_pvalue < float(cfg.get("max_spa_pvalue", 0.05)),
                evidence.spa_pvalue,
                f"<{float(cfg.get('max_spa_pvalue', 0.05))}",
            ),
            "symbol_concentration": GateCriterion(
                "단일 심볼 이익 기여",
                evidence.max_symbol_contribution_share <= max_contribution,
                evidence.max_symbol_contribution_share,
                max_contribution,
            ),
            "quarter_concentration": GateCriterion(
                "단일 분기 이익 기여",
                evidence.max_quarter_contribution_share <= max_contribution,
                evidence.max_quarter_contribution_share,
                max_contribution,
            ),
            "double_cost": GateCriterion(
                "2배 비용 스트레스",
                evidence.double_cost_multiplier >= 2
                and evidence.double_cost_return >= -max_double_loss,
                evidence.double_cost_return,
                -max_double_loss,
            ),
            "strategy_logic": GateCriterion(
                "전략 논리 유지",
                evidence.strategy_logic_intact,
                evidence.strategy_logic_intact,
                True,
            ),
            "hypothesis_budget": GateCriterion(
                "전략군 사전 설정 수",
                evidence.hypothesis_configs <= self.max_configs_per_family,
                evidence.hypothesis_configs,
                self.max_configs_per_family,
            ),
        }
        return _decision("offline", evidence, criteria)


class DemoPromotionGate:
    """미래 데이터 데모 결과를 실전 승인 후보로 보낼지 판정한다."""

    def __init__(self, config: Mapping[str, float | int | bool] | None = None) -> None:
        """데모 게이트 기준을 초기화한다."""
        validation = _load_validation_config()
        self.config = dict(config or validation.get("demo", {}))

    def evaluate(self, evidence: DemoValidationEvidence) -> GateDecision:
        """데모 증거가 모든 고정 파라미터·체결·대사 기준을 만족하는지 판정한다."""
        cfg = self.config
        full_reconciliation = bool(cfg.get("require_full_reconciliation", True))
        criteria = {
            "calendar_days": GateCriterion(
                "미래 데모 기간",
                evidence.calendar_days >= int(cfg.get("min_calendar_days", 90)),
                evidence.calendar_days,
                int(cfg.get("min_calendar_days", 90)),
            ),
            "effective_bets": GateCriterion(
                "유효 독립 베팅",
                evidence.effective_bets >= int(cfg.get("min_effective_bets", 100)),
                evidence.effective_bets,
                int(cfg.get("min_effective_bets", 100)),
            ),
            "expectancy_ci": GateCriterion(
                "순기대값 95% 하한",
                evidence.expectancy_ci_lower
                > float(cfg.get("min_expectancy_ci_lower", 0)),
                evidence.expectancy_ci_lower,
                f">{float(cfg.get('min_expectancy_ci_lower', 0))}",
            ),
            "daily_sharpe": GateCriterion(
                "UTC 일별 Sharpe",
                evidence.daily_sharpe >= float(cfg.get("min_daily_sharpe", 1)),
                evidence.daily_sharpe,
                float(cfg.get("min_daily_sharpe", 1)),
            ),
            "profit_factor": GateCriterion(
                "Profit Factor",
                evidence.profit_factor >= float(cfg.get("min_profit_factor", 1.2)),
                evidence.profit_factor,
                float(cfg.get("min_profit_factor", 1.2)),
            ),
            "max_drawdown": GateCriterion(
                "최대 낙폭",
                evidence.max_drawdown <= float(cfg.get("max_mdd", 0.075)),
                evidence.max_drawdown,
                float(cfg.get("max_mdd", 0.075)),
            ),
            "fill_error_median": GateCriterion(
                "체결가 중앙 오차(bp)",
                evidence.fill_error_median_bps
                <= float(cfg.get("max_fill_error_median_bps", 5)),
                evidence.fill_error_median_bps,
                float(cfg.get("max_fill_error_median_bps", 5)),
            ),
            "fill_error_p95": GateCriterion(
                "체결가 P95 오차(bp)",
                evidence.fill_error_p95_bps
                <= float(cfg.get("max_fill_error_p95_bps", 25)),
                evidence.fill_error_p95_bps,
                float(cfg.get("max_fill_error_p95_bps", 25)),
            ),
            "fill_rate_error": GateCriterion(
                "체결률 예측 오차",
                evidence.fill_rate_error
                <= float(cfg.get("max_fill_rate_error_pct", 0.10)),
                evidence.fill_rate_error,
                float(cfg.get("max_fill_rate_error_pct", 0.10)),
            ),
            "reconciliation": GateCriterion(
                "주문·포지션·잔고 대사",
                (not full_reconciliation) or evidence.reconciliation_rate == 1.0,
                evidence.reconciliation_rate,
                1.0,
            ),
            "orphan_positions": GateCriterion(
                "고아 포지션",
                evidence.orphan_positions == 0,
                evidence.orphan_positions,
                0,
            ),
            "duplicate_orders": GateCriterion(
                "중복 주문",
                evidence.duplicate_orders == 0,
                evidence.duplicate_orders,
                0,
            ),
            "parameters_frozen": GateCriterion(
                "데모 중 파라미터 동결",
                evidence.parameters_unchanged,
                evidence.parameters_unchanged,
                True,
            ),
        }
        return _decision("demo", evidence, criteria)

    def build_approval_report(
        self,
        evidence: DemoValidationEvidence,
        generated_at: datetime | None = None,
    ) -> DemoApprovalReport:
        """데모 증거를 재판정해 실전 실행기가 검증할 승인 리포트를 만든다."""
        return build_demo_approval_report(
            evidence,
            gate=self,
            generated_at=generated_at,
        )


def _decision(
    stage: str,
    evidence: OfflineValidationEvidence | DemoValidationEvidence,
    criteria: Mapping[str, GateCriterion],
) -> GateDecision:
    """기준 맵을 불변 종합 판정으로 변환한다."""
    passed = all(criterion.passed for criterion in criteria.values())
    failures = [
        criterion.name for criterion in criteria.values() if not criterion.passed
    ]
    summary = (
        f"{stage} 승급 게이트 통과"
        if passed
        else f"{stage} 승급 게이트 미통과: {', '.join(failures)}"
    )
    logger.info("%s strategy=%s", summary, evidence.strategy_version)
    return GateDecision(
        passed=passed,
        stage=stage,
        strategy_id=evidence.strategy_id,
        strategy_version=evidence.strategy_version,
        criteria=dict(criteria),
        summary=summary,
    )


def clustered_expectancy_ci(
    returns: Sequence[float],
    clusters: Sequence[str],
    confidence: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    """일·심볼 등 클러스터 블록 재표집으로 평균 수익 신뢰구간을 계산한다."""
    if len(returns) != len(clusters) or not returns:
        raise ValueError("returns와 clusters는 같은 길이의 비어 있지 않은 배열이어야 합니다.")
    if not 0 < confidence < 1 or bootstrap_samples <= 0:
        raise ValueError("confidence와 bootstrap_samples 값이 올바르지 않습니다.")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(returns, clusters):
        grouped.setdefault(str(cluster), []).append(float(value))
    blocks = list(grouped.values())
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        sample = [value for block_index in selected for value in blocks[block_index]]
        means[index] = float(np.mean(sample))
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def two_way_clustered_expectancy_ci(
    returns: Sequence[float],
    dates: Sequence[str],
    symbols: Sequence[str],
    confidence: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    """일자와 심볼을 독립 재표집하는 2방향 pigeonhole 신뢰구간을 계산한다."""
    if not returns or len(returns) != len(dates) or len(returns) != len(symbols):
        raise ValueError("수익률·일자·심볼은 같은 길이의 비어 있지 않은 배열이어야 합니다.")
    if not 0 < confidence < 1 or bootstrap_samples <= 0:
        raise ValueError("confidence와 bootstrap_samples 값이 올바르지 않습니다.")
    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("수익률에 유한하지 않은 값이 있습니다.")
    unique_dates = sorted({str(value) for value in dates})
    unique_symbols = sorted({str(value) for value in symbols})
    if len(unique_dates) < 2 or len(unique_symbols) < 2:
        raise ValueError("2방향 클러스터에는 각각 2개 이상의 일자와 심볼이 필요합니다.")
    date_lookup = {value: index for index, value in enumerate(unique_dates)}
    symbol_lookup = {value: index for index, value in enumerate(unique_symbols)}
    date_index = np.asarray([date_lookup[str(value)] for value in dates], dtype=int)
    symbol_index = np.asarray(
        [symbol_lookup[str(value)] for value in symbols], dtype=int
    )
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for sample_index in range(bootstrap_samples):
        # 각 차원을 독립 재표집하고 교차 가중치로 관측치 종속성을 보존한다.
        for _ in range(100):
            date_counts = np.bincount(
                rng.integers(0, len(unique_dates), size=len(unique_dates)),
                minlength=len(unique_dates),
            )
            symbol_counts = np.bincount(
                rng.integers(0, len(unique_symbols), size=len(unique_symbols)),
                minlength=len(unique_symbols),
            )
            weights = date_counts[date_index] * symbol_counts[symbol_index]
            weight_sum = int(weights.sum())
            if weight_sum > 0:
                means[sample_index] = float(np.average(values, weights=weights))
                break
        else:
            raise RuntimeError("2방향 클러스터 재표집에서 유효 표본을 만들지 못했습니다.")
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def deflated_sharpe_probability(
    observed_sharpe: float,
    observations: int,
    trials: int,
    sharpe_std: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """다중 시도와 비정규성을 반영한 양의 Deflated Sharpe 확률을 근사한다."""
    inputs = (observed_sharpe, sharpe_std, skewness, kurtosis)
    if not all(np.isfinite(float(value)) for value in inputs):
        raise ValueError("DSR 입력은 모두 유한해야 합니다.")
    if observations < 2 or trials < 2 or sharpe_std <= 0 or kurtosis < 1:
        raise ValueError("관측 수·시도 수·Sharpe 표준편차가 올바르지 않습니다.")
    normal = NormalDist()
    euler_gamma = 0.5772156649
    expected_max = sharpe_std * (
        (1 - euler_gamma) * normal.inv_cdf(1 - 1 / trials)
        + euler_gamma * normal.inv_cdf(1 - 1 / (trials * np.e))
    )
    denominator = (
        1
        - skewness * observed_sharpe
        + ((kurtosis - 1) / 4) * observed_sharpe**2
    )
    if denominator <= 1e-12:
        raise ValueError("DSR 분산 보정치가 0 이하입니다.")
    statistic = (
        (observed_sharpe - expected_max)
        * np.sqrt(observations - 1)
        / np.sqrt(denominator)
    )
    return float(normal.cdf(float(statistic)))


def max_positive_contribution_share(
    contributions: Mapping[str, float],
) -> float:
    """양의 총이익 중 가장 큰 단일 기여 비중을 계산한다."""
    positives = [float(value) for value in contributions.values() if value > 0]
    if not positives:
        return 1.0
    return max(positives) / sum(positives)


def cscv_probability_of_backtest_overfitting(
    candidate_return_matrix: Sequence[Sequence[float]],
    partitions: int = 10,
) -> float:
    """CSCV 분할에서 IS 최우수 후보가 OOS 하위 절반인 비율(PBO)을 계산한다."""
    matrix = np.asarray(candidate_return_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 8 or matrix.shape[1] < 2:
        raise ValueError("candidate_return_matrix는 8행·2열 이상이어야 합니다.")
    if partitions < 4:
        raise ValueError("partitions는 4 이상이어야 합니다.")
    if not np.isfinite(matrix).all():
        raise ValueError("candidate_return_matrix에 유한하지 않은 값이 있습니다.")
    if np.unique(matrix, axis=1).shape[1] < 2:
        raise ValueError("중복 후보 열은 단일 후보 증거로 간주합니다.")
    block_count = min(partitions, matrix.shape[0])
    if block_count % 2:
        block_count -= 1
    if block_count < 2:
        raise ValueError("CSCV에는 최소 2개 파티션이 필요합니다.")
    blocks = [block for block in np.array_split(np.arange(matrix.shape[0]), block_count)]
    overfit = 0
    split_count = 0
    for selected in combinations(range(block_count), block_count // 2):
        selected_set = set(selected)
        train_index = np.concatenate([blocks[index] for index in selected])
        test_index = np.concatenate(
            [blocks[index] for index in range(block_count) if index not in selected_set]
        )
        train_scores = _column_sharpes(matrix[train_index])
        test_scores = _column_sharpes(matrix[test_index])
        best_index = int(np.argmax(train_scores))
        relative_rank = (
            float(np.sum(test_scores <= test_scores[best_index])) / matrix.shape[1]
        )
        overfit += int(relative_rank <= 0.5)
        split_count += 1
    return overfit / split_count if split_count else 1.0


def spa_block_bootstrap_pvalue(
    candidate_return_matrix: Sequence[Sequence[float]],
    benchmark_returns: Sequence[float],
    bootstrap_samples: int = 2_000,
    block_length: int | None = None,
    seed: int = 0,
) -> float:
    """벤치마크 대비 최우수 후보의 SPA block-bootstrap p-value를 계산한다."""
    candidates = np.asarray(candidate_return_matrix, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if (
        candidates.ndim != 2
        or benchmark.ndim != 1
        or candidates.shape[0] != benchmark.shape[0]
        or candidates.shape[0] < 8
        or candidates.shape[1] < 2
    ):
        raise ValueError("후보 행렬과 벤치마크 수익률의 길이·차원이 올바르지 않습니다.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples는 양수여야 합니다.")
    if not np.isfinite(candidates).all() or not np.isfinite(benchmark).all():
        raise ValueError("SPA 입력에 유한하지 않은 값이 있습니다.")
    if np.unique(candidates, axis=1).shape[1] < 2:
        raise ValueError("중복 후보 열은 단일 후보 증거로 간주합니다.")
    differential = candidates - benchmark[:, None]
    means = differential.mean(axis=0)
    scales = differential.std(axis=0, ddof=1)
    safe_scales = np.where(scales > 1e-12, scales, np.inf)
    observed = float(
        np.max(np.sqrt(candidates.shape[0]) * means / safe_scales)
    )
    if observed <= 0:
        return 1.0

    length = block_length or max(2, int(round(np.sqrt(candidates.shape[0]))))
    if length < 2 or length > candidates.shape[0] // 2:
        raise ValueError("block_length는 2 이상, 관측 수의 절반 이하여야 합니다.")
    # SPA 귀무가설 아래 양의 평균만 제거해 열별 데이터 스누핑을 보정한다.
    centered = differential - np.maximum(means, 0.0)
    rng = np.random.default_rng(seed)
    bootstrap_stats = np.empty(bootstrap_samples)
    for sample_index in range(bootstrap_samples):
        indices = _circular_block_indices(
            candidates.shape[0],
            length,
            rng,
        )
        sample_means = centered[indices].mean(axis=0)
        bootstrap_stats[sample_index] = float(
            np.max(
                np.sqrt(candidates.shape[0])
                * sample_means
                / safe_scales
            )
        )
    return float(
        (1 + np.sum(bootstrap_stats >= observed)) / (bootstrap_samples + 1)
    )


def _column_sharpes(matrix: np.ndarray) -> np.ndarray:
    """후보 수익률 행렬의 열별 비연율화 Sharpe를 계산한다."""
    means = matrix.mean(axis=0)
    deviations = matrix.std(axis=0, ddof=1)
    return np.divide(
        means,
        deviations,
        out=np.zeros_like(means),
        where=deviations > 1e-12,
    )


def _circular_block_indices(
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """원형 블록 부트스트랩 인덱스를 만든다."""
    blocks_needed = int(np.ceil(observations / block_length))
    starts = rng.integers(0, observations, size=blocks_needed)
    indices = np.concatenate(
        [
            (start + np.arange(block_length)) % observations
            for start in starts
        ]
    )
    return indices[:observations]


@dataclass(frozen=True)
class OfflineEvidenceReport:
    """원시 수익률에서 계산된 오프라인 증거와 감사용 해시."""

    evidence: OfflineValidationEvidence
    methodology: str
    generated_at: datetime
    raw_input_sha256: str

    def to_dict(self) -> dict:
        """서명 가능한 정규화 딕셔너리로 변환한다."""
        evidence = asdict(self.evidence)
        evidence["started_at"] = self.evidence.started_at.isoformat()
        evidence["ended_at"] = self.evidence.ended_at.isoformat()
        evidence["regimes"] = sorted(self.evidence.regimes)
        return {
            "evidence": evidence,
            "methodology": self.methodology,
            "generated_at": self.generated_at.isoformat(),
            "raw_input_sha256": self.raw_input_sha256,
        }

    def to_json(self) -> str:
        """키 정렬된 서명용 JSON을 반환한다."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        """서명용 JSON의 SHA-256을 반환한다."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


_DEMO_REPORT_TOKEN = object()


@dataclass(frozen=True, init=False)
class DemoApprovalReport:
    """DemoPromotionGate 판정에서만 생성되는 불변 실전 승인 리포트."""

    strategy_id: str
    strategy_version: str
    stage: str
    passed: bool
    criteria: Mapping[str, Mapping[str, object]]
    summary: str
    generated_at: datetime
    evidence_sha256: str
    raw_event_sha256: str
    data_sha256: str
    code_sha256: str
    hypothesis_sha256: str
    offline_artifact_sha256: str
    methodology: str

    def __init__(
        self,
        decision: GateDecision,
        evidence_sha256: str,
        generated_at: datetime,
        token: object,
        raw_event_sha256: str = "",
        data_sha256: str = "",
        code_sha256: str = "",
        hypothesis_sha256: str = "",
        offline_artifact_sha256: str = "",
    ) -> None:
        """내부 게이트 판정으로만 승인 리포트를 초기화한다."""
        if token is not _DEMO_REPORT_TOKEN:
            raise ValueError("DemoApprovalReport는 DemoPromotionGate를 통해 생성해야 합니다.")
        if decision.stage != "demo":
            raise ValueError("demo 단계 판정만 승인 리포트로 만들 수 있습니다.")
        criteria = {
            key: MappingProxyType(
                {
                    "name": criterion.name,
                    "passed": criterion.passed,
                    "value": criterion.value,
                    "threshold": criterion.threshold,
                }
            )
            for key, criterion in decision.criteria.items()
        }
        object.__setattr__(self, "strategy_id", decision.strategy_id)
        object.__setattr__(self, "strategy_version", decision.strategy_version)
        object.__setattr__(self, "stage", decision.stage)
        object.__setattr__(self, "passed", decision.passed)
        object.__setattr__(self, "criteria", MappingProxyType(criteria))
        object.__setattr__(self, "summary", decision.summary)
        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "evidence_sha256", evidence_sha256)
        object.__setattr__(self, "raw_event_sha256", raw_event_sha256)
        object.__setattr__(self, "data_sha256", data_sha256)
        object.__setattr__(self, "code_sha256", code_sha256)
        object.__setattr__(self, "hypothesis_sha256", hypothesis_sha256)
        object.__setattr__(self, "offline_artifact_sha256", offline_artifact_sha256)
        object.__setattr__(
            self,
            "methodology",
            "future-demo-gate/frozen-parameters+execution-reconciliation/v1",
        )

    def to_dict(self) -> dict:
        """실전 실행기 승인 스키마의 정규화 딕셔너리를 반환한다."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "stage": self.stage,
            "passed": self.passed,
            "criteria": {
                key: dict(value) for key, value in self.criteria.items()
            },
            "summary": self.summary,
            "generated_at": self.generated_at.isoformat(),
            "evidence_sha256": self.evidence_sha256,
            "raw_event_sha256": self.raw_event_sha256,
            "data_sha256": self.data_sha256,
            "code_sha256": self.code_sha256,
            "hypothesis_sha256": self.hypothesis_sha256,
            "offline_artifact_sha256": self.offline_artifact_sha256,
            "methodology": self.methodology,
        }

    def to_json(self) -> str:
        """키 정렬·공백 제거된 승인 JSON을 반환한다."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        """실전 실행기에 설정할 승인 JSON SHA-256을 반환한다."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def build_demo_approval_report(
    evidence: DemoValidationEvidence,
    gate: DemoPromotionGate | None = None,
    generated_at: datetime | None = None,
    *,
    raw_event_sha256: str = "",
    data_sha256: str = "",
    code_sha256: str = "",
    hypothesis_sha256: str = "",
    offline_artifact_sha256: str = "",
) -> DemoApprovalReport:
    """원시 데모 증거를 재판정하고 이벤트·데이터·코드 계보를 결합한다."""
    decision = (gate or DemoPromotionGate()).evaluate(evidence)
    lineage = {
        "raw_event_sha256": raw_event_sha256,
        "data_sha256": data_sha256,
        "code_sha256": code_sha256,
        "hypothesis_sha256": hypothesis_sha256,
        "offline_artifact_sha256": offline_artifact_sha256,
    }
    for name, value in lineage.items():
        if value and not _is_sha256(value):
            raise ValueError(f"{name}은 소문자 SHA-256 헥스여야 합니다.")
    if decision.passed and any(not value for value in lineage.values()):
        raise ValueError("통과 데모 승인에는 이벤트·데이터·코드·가설·offline 계보 해시가 모두 필요합니다.")
    evidence_json = json.dumps(
        asdict(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_hash = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
    report_time = generated_at or datetime.now(timezone.utc)
    if report_time.tzinfo is None:
        raise ValueError("generated_at은 timezone-aware여야 합니다.")
    return DemoApprovalReport(
        decision,
        evidence_hash,
        report_time,
        _DEMO_REPORT_TOKEN,
        raw_event_sha256,
        data_sha256,
        code_sha256,
        hypothesis_sha256,
        offline_artifact_sha256,
    )


def _is_sha256(value: str) -> bool:
    """문자열이 소문자 SHA-256 헥스인지 반환한다."""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def build_offline_evidence_report(
    *,
    strategy_id: str,
    strategy_version: str,
    started_at: datetime,
    ended_at: datetime,
    regimes: Iterable[str],
    net_returns: Sequence[float],
    stressed_returns: Sequence[float],
    double_cost_returns: Sequence[float],
    trade_clusters: Sequence[str],
    symbols: Sequence[str],
    quarters: Sequence[str],
    daily_returns: Sequence[float],
    candidate_return_matrix: Sequence[Sequence[float]],
    benchmark_returns: Sequence[float],
    strategy_logic_intact: bool = True,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> OfflineEvidenceReport:
    """호환용 수동 라벨 리포트를 만든다. 이 리포트는 승급 증거가 아니다."""
    net = np.asarray(net_returns, dtype=float)
    stressed = np.asarray(stressed_returns, dtype=float)
    doubled = np.asarray(double_cost_returns, dtype=float)
    daily = np.asarray(daily_returns, dtype=float)
    candidate_matrix = np.asarray(candidate_return_matrix, dtype=float)
    count = len(net)
    if count == 0 or not (
        len(stressed)
        == len(doubled)
        == len(trade_clusters)
        == len(symbols)
        == len(quarters)
        == count
    ):
        raise ValueError("거래 수익률·클러스터·기여 라벨 길이가 일치해야 합니다.")
    if daily.size < 2 or candidate_matrix.ndim != 2:
        raise ValueError("일별 수익률과 후보 행렬 표본이 부족합니다.")
    if candidate_matrix.shape[0] != len(benchmark_returns):
        raise ValueError("후보 행렬과 벤치마크 수익률 길이가 다릅니다.")
    arrays = [net, stressed, doubled, daily, candidate_matrix]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("검증 원시 입력에 유한하지 않은 값이 있습니다.")

    ci_lower, _ = clustered_expectancy_ci(
        net.tolist(),
        trade_clusters,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    daily_sharpe = _annualized_daily_sharpe(daily)
    # DSR 검정통계는 표본 수를 별도로 반영하므로 비연율화 Sharpe를 사용한다.
    candidate_daily_sharpes = np.apply_along_axis(
        _unannualized_sharpe,
        0,
        candidate_matrix,
    )
    dsr_probability = deflated_sharpe_probability(
        observed_sharpe=_unannualized_sharpe(daily),
        observations=len(daily),
        trials=candidate_matrix.shape[1],
        sharpe_std=float(candidate_daily_sharpes.std(ddof=1))
        if candidate_matrix.shape[1] > 1
        else 0.0,
        skewness=_sample_skewness(daily),
        kurtosis=_sample_kurtosis(daily),
    )
    pbo = cscv_probability_of_backtest_overfitting(candidate_matrix)
    spa = spa_block_bootstrap_pvalue(
        candidate_matrix,
        benchmark_returns,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    symbol_contributions = _aggregate_contributions(net, symbols)
    quarter_contributions = _aggregate_contributions(net, quarters)
    evidence = OfflineValidationEvidence(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        effective_bets=len(set(str(cluster) for cluster in trade_clusters)),
        started_at=started_at,
        ended_at=ended_at,
        regimes=frozenset(regimes),
        base_net_expectancy=float(net.mean()),
        stressed_net_expectancy=float(stressed.mean()),
        expectancy_ci_lower=ci_lower,
        daily_sharpe=daily_sharpe,
        profit_factor=_profit_factor(net),
        max_drawdown=_returns_max_drawdown(daily),
        deflated_sharpe_probability=dsr_probability,
        pbo=pbo,
        spa_pvalue=spa,
        max_symbol_contribution_share=max_positive_contribution_share(
            symbol_contributions
        ),
        max_quarter_contribution_share=max_positive_contribution_share(
            quarter_contributions
        ),
        double_cost_return=float(np.prod(1 + doubled) - 1),
        strategy_logic_intact=strategy_logic_intact,
        hypothesis_configs=candidate_matrix.shape[1],
    )
    raw_payload = {
        "net_returns": net.tolist(),
        "stressed_returns": stressed.tolist(),
        "double_cost_returns": doubled.tolist(),
        "trade_clusters": list(trade_clusters),
        "symbols": list(symbols),
        "quarters": list(quarters),
        "daily_returns": daily.tolist(),
        "candidate_return_matrix": candidate_matrix.tolist(),
        "benchmark_returns": list(map(float, benchmark_returns)),
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
    }
    raw_json = json.dumps(
        raw_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return OfflineEvidenceReport(
        evidence=evidence,
        methodology="legacy-manual-labels/non-promotable-v1",
        generated_at=datetime.now(tz=started_at.tzinfo),
        raw_input_sha256=hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
    )


def build_offline_evidence_from_records(
    *,
    strategy_id: str,
    strategy_version: str,
    selected_candidate_id: str,
    trades: Sequence[DatedTradeReturn],
    daily_records: Sequence[DatedCandidateReturns],
    bootstrap_samples: int = 2_000,
    seed: int = 0,
    generated_at: datetime | None = None,
) -> OfflineEvidenceReport:
    """날짜가 있는 원시 거래·일별 후보 행렬에서만 승급 가능 증거를 산출한다."""
    if not strategy_id.strip() or not strategy_version.strip():
        raise ValueError("전략 ID와 버전은 비어 있을 수 없습니다.")
    candidate_id = selected_candidate_id.strip()
    if not candidate_id or not trades or len(daily_records) < 8:
        raise ValueError("선택 후보, 거래, 8일 이상의 일별 레코드가 필요합니다.")
    ordered_trades = sorted(
        trades,
        key=lambda item: (item.closed_at.astimezone(timezone.utc), item.trade_id),
    )
    trade_ids = [item.trade_id for item in ordered_trades]
    if len(set(trade_ids)) != len(trade_ids):
        raise ValueError("중복 trade_id는 승급 증거에 사용할 수 없습니다.")
    ordered_daily = sorted(
        daily_records,
        key=lambda item: item.observed_at.astimezone(timezone.utc),
    )
    utc_dates = [
        item.observed_at.astimezone(timezone.utc).date().isoformat()
        for item in ordered_daily
    ]
    if len(set(utc_dates)) != len(utc_dates):
        raise ValueError("UTC 하루에 일별 후보 레코드는 하나만 허용됩니다.")
    candidate_ids = tuple(ordered_daily[0].candidate_returns)
    if candidate_id not in candidate_ids or len(candidate_ids) < 2:
        raise ValueError("선택 후보가 2개 이상의 전체 후보 집합에 없습니다.")
    for record in ordered_daily:
        if tuple(record.candidate_returns) != candidate_ids:
            raise ValueError("모든 일자의 후보 ID 집합과 순서가 같아야 합니다.")
        if not np.isclose(
            record.strategy_return,
            record.candidate_returns[candidate_id],
            rtol=0,
            atol=1e-15,
        ):
            raise ValueError("선택 전략 수익률이 후보 행렬의 선택 열과 다릅니다.")

    candidate_matrix = np.asarray(
        [
            [record.candidate_returns[key] for key in candidate_ids]
            for record in ordered_daily
        ],
        dtype=float,
    )
    if np.unique(candidate_matrix, axis=1).shape[1] < 2:
        raise ValueError("중복 수익 열은 단일 후보 증거로 간주합니다.")
    daily = np.asarray([item.strategy_return for item in ordered_daily], dtype=float)
    benchmark = np.asarray(
        [item.benchmark_return for item in ordered_daily], dtype=float
    )
    started_at = ordered_daily[0].observed_at.astimezone(timezone.utc)
    ended_at = ordered_daily[-1].observed_at.astimezone(timezone.utc)
    if ended_at <= started_at:
        raise ValueError("검증 기간은 0보다 길어야 합니다.")
    for trade in ordered_trades:
        closed_at = trade.closed_at.astimezone(timezone.utc)
        if closed_at < started_at or closed_at > ended_at:
            raise ValueError("거래 시각은 일별 증거 기간 안에 있어야 합니다.")

    net = np.asarray([item.net_return for item in ordered_trades], dtype=float)
    stressed = np.asarray(
        [item.stressed_return for item in ordered_trades], dtype=float
    )
    doubled = np.asarray(
        [item.double_cost_return for item in ordered_trades], dtype=float
    )
    trade_dates = [
        item.closed_at.astimezone(timezone.utc).date().isoformat()
        for item in ordered_trades
    ]
    symbols = [item.symbol for item in ordered_trades]
    quarters = [
        _quarter_label(item.closed_at.astimezone(timezone.utc))
        for item in ordered_trades
    ]
    ci_lower, _ = two_way_clustered_expectancy_ci(
        net.tolist(),
        trade_dates,
        symbols,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    candidate_sharpes = np.apply_along_axis(
        _unannualized_sharpe,
        0,
        candidate_matrix,
    )
    sharpe_std = float(candidate_sharpes.std(ddof=1))
    dsr_probability = deflated_sharpe_probability(
        observed_sharpe=_unannualized_sharpe(daily),
        observations=len(daily),
        trials=candidate_matrix.shape[1],
        sharpe_std=sharpe_std,
        skewness=_sample_skewness(daily),
        kurtosis=_sample_kurtosis(daily),
    )
    symbol_contributions = _aggregate_contributions(net, symbols)
    quarter_contributions = _aggregate_contributions(net, quarters)
    evidence = OfflineValidationEvidence(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        effective_bets=len(set(zip(trade_dates, symbols))),
        started_at=started_at,
        ended_at=ended_at,
        regimes=_derive_regimes(benchmark),
        base_net_expectancy=float(net.mean()),
        stressed_net_expectancy=float(stressed.mean()),
        expectancy_ci_lower=ci_lower,
        daily_sharpe=_annualized_daily_sharpe(daily),
        profit_factor=_profit_factor(net),
        max_drawdown=_returns_max_drawdown(daily),
        deflated_sharpe_probability=dsr_probability,
        pbo=cscv_probability_of_backtest_overfitting(candidate_matrix),
        spa_pvalue=spa_block_bootstrap_pvalue(
            candidate_matrix,
            benchmark,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        max_symbol_contribution_share=max_positive_contribution_share(
            symbol_contributions
        ),
        max_quarter_contribution_share=max_positive_contribution_share(
            quarter_contributions
        ),
        double_cost_return=float(np.prod(1 + doubled) - 1),
        strategy_logic_intact=True,
        hypothesis_configs=candidate_matrix.shape[1],
    )
    raw_payload = {
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "selected_candidate_id": candidate_id,
        "trades": [
            {
                "trade_id": item.trade_id,
                "closed_at": item.closed_at.astimezone(timezone.utc).isoformat(),
                "symbol": item.symbol,
                "net_return": item.net_return,
                "stressed_return": item.stressed_return,
                "double_cost_return": item.double_cost_return,
            }
            for item in ordered_trades
        ],
        "daily_records": [
            {
                "observed_at": item.observed_at.astimezone(timezone.utc).isoformat(),
                "strategy_return": item.strategy_return,
                "benchmark_return": item.benchmark_return,
                "candidate_returns": dict(item.candidate_returns),
            }
            for item in ordered_daily
        ],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    raw_json = json.dumps(
        raw_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    report_time = generated_at or datetime.now(timezone.utc)
    if report_time.tzinfo is None:
        raise ValueError("generated_at은 timezone-aware여야 합니다.")
    return OfflineEvidenceReport(
        evidence=evidence,
        methodology=(
            "two-way-day-symbol-bootstrap+derived-regimes+dSR+"
            "CSCV-PBO+block-bootstrap-SPA/v2"
        ),
        generated_at=report_time.astimezone(timezone.utc),
        raw_input_sha256=hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
    )


def _quarter_label(value: datetime) -> str:
    """UTC 시각을 분기 기여 라벨로 변환한다."""
    utc_value = value.astimezone(timezone.utc)
    return f"{utc_value.year}-Q{(utc_value.month - 1) // 3 + 1}"


def _derive_regimes(benchmark_returns: np.ndarray) -> frozenset[str]:
    """벤치마크 일별 수익률의 과거 창에서 시장 레짐을 결정적으로 유도한다."""
    regimes: set[str] = set()
    realized_volatility: list[float] = []
    for index in range(benchmark_returns.size):
        trend_window = benchmark_returns[max(0, index - 29) : index + 1]
        trend = float(np.prod(1 + trend_window) - 1)
        if trend > 0.05:
            regimes.add("bull")
        elif trend < -0.05:
            regimes.add("bear")
        else:
            regimes.add("sideways")
        if index >= 19:
            volatility = float(
                benchmark_returns[index - 19 : index + 1].std(ddof=1)
                * np.sqrt(365)
            )
            if len(realized_volatility) >= 20:
                history = realized_volatility[-252:]
                if volatility > float(np.quantile(history, 0.75)):
                    regimes.add("high_volatility")
            realized_volatility.append(volatility)
    return frozenset(regimes)


def _annualized_daily_sharpe(values: np.ndarray) -> float:
    """일별 수익률의 365일 연율화 Sharpe를 계산한다."""
    if values.size < 2:
        return 0.0
    deviation = float(values.std(ddof=1))
    if deviation <= 1e-12:
        return 0.0
    return float(values.mean() / deviation * np.sqrt(365))


def _unannualized_sharpe(values: np.ndarray) -> float:
    """DSR 표본 통계용 비연율화 일별 Sharpe를 계산한다."""
    if values.size < 2:
        return 0.0
    deviation = float(values.std(ddof=1))
    if deviation <= 1e-12:
        return 0.0
    return float(values.mean() / deviation)


def _sample_skewness(values: np.ndarray) -> float:
    """일별 수익률 표본 왜도를 계산한다."""
    centered = values - values.mean()
    deviation = values.std(ddof=1)
    if deviation <= 1e-12:
        return 0.0
    return float(np.mean((centered / deviation) ** 3))


def _sample_kurtosis(values: np.ndarray) -> float:
    """일별 수익률 표본 첨도를 계산한다."""
    centered = values - values.mean()
    deviation = values.std(ddof=1)
    if deviation <= 1e-12:
        return 3.0
    return float(np.mean((centered / deviation) ** 4))


def _aggregate_contributions(
    returns: np.ndarray,
    labels: Sequence[str],
) -> dict[str, float]:
    """라벨별 수익 기여를 합산한다."""
    result: dict[str, float] = {}
    for value, label in zip(returns, labels):
        key = str(label)
        result[key] = result.get(key, 0.0) + float(value)
    return result


def _profit_factor(returns: np.ndarray) -> float:
    """수익률 배열의 Profit Factor를 계산한다."""
    profit = float(returns[returns > 0].sum())
    loss = abs(float(returns[returns < 0].sum()))
    return profit / loss if loss > 0 else (float("inf") if profit > 0 else 0.0)


def _returns_max_drawdown(daily_returns: np.ndarray) -> float:
    """일별 수익률에서 복리 순자산 최대 낙폭을 계산한다."""
    equity = np.concatenate(([1.0], np.cumprod(1 + daily_returns)))
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(drawdown.max())
