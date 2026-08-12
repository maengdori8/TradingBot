from __future__ import annotations

"""캐리·강제흐름 공용 시점 안전 결정 API 테스트."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.strategy.carry_signal import CarryConfig, CarrySnapshot
from src.strategy.decision import DecisionContext
from src.strategy.evidence_decision import (
    BookLevel,
    FeatureFreshness,
    FeedGap,
    ForcedFlowFeatureBundle,
    LiquidationNotional,
    OrderBookEvidence,
    StrategyIntentLeg,
    StrategyTradeIntent,
    TimedValue,
    build_forced_flow_snapshot,
    decide_carry_intent,
    decide_forced_flow_intent,
)
from src.strategy.forced_flow_signal import ForcedFlowConfig

SYMBOL = "BTC/USDT:USDT"


def _cutoff() -> datetime:
    """UTC 15분 경계 결정 시각을 반환한다."""
    now = datetime.now(timezone.utc)
    return now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)


def _context(cutoff: datetime, version: str = "candidate-v1") -> DecisionContext:
    """완전 종료 봉 결정 컨텍스트를 반환한다."""
    return DecisionContext.for_closed_bar(
        cutoff,
        version,
        "run-1",
        decision_time=cutoff,
        data_cutoff=cutoff,
    )


def _book(cutoff: datetime, age_seconds: int = 1) -> OrderBookEvidence:
    """25레벨 미교차 주문장을 반환한다."""
    observed = cutoff - timedelta(seconds=age_seconds)
    return OrderBookEvidence(
        observed_at=observed,
        available_at=observed,
        bids=tuple(BookLevel(99.0 - index * 0.1, 3.0) for index in range(25)),
        asks=tuple(BookLevel(101.0 + index * 0.1, 1.0) for index in range(25)),
    )


def _bundle(cutoff: datetime, **overrides: object) -> ForcedFlowFeatureBundle:
    """강한 롱 특징이 나오는 4시간 원시 입력을 반환한다."""
    volumes = tuple(
        TimedValue(
            cutoff - timedelta(minutes=29 - index, seconds=10),
            cutoff - timedelta(minutes=29 - index, seconds=10),
            float(100 + index),
        )
        for index in range(29)
    ) + (
        TimedValue(cutoff - timedelta(seconds=10), cutoff - timedelta(seconds=10), 1000.0),
    )
    values: dict[str, object] = {
        "symbol": SYMBOL,
        "prices": (
            TimedValue(cutoff - timedelta(hours=4, seconds=10), cutoff - timedelta(hours=4, seconds=10), 100.0),
            TimedValue(cutoff - timedelta(seconds=10), cutoff - timedelta(seconds=10), 105.0),
        ),
        "open_interest": (
            TimedValue(cutoff - timedelta(hours=4, seconds=10), cutoff - timedelta(hours=4, seconds=10), 1000.0),
            TimedValue(cutoff - timedelta(seconds=10), cutoff - timedelta(seconds=10), 1200.0),
        ),
        "completed_volumes": volumes,
        "funding": (
            TimedValue(cutoff - timedelta(seconds=10), cutoff - timedelta(seconds=10), -0.001),
        ),
        "liquidations": (
            LiquidationNotional(
                cutoff - timedelta(hours=1),
                cutoff - timedelta(hours=1),
                "buy",
                10_000.0,
            ),
        ),
        "orderbook": _book(cutoff),
    }
    values.update(overrides)
    return ForcedFlowFeatureBundle(**values)  # type: ignore[arg-type]


def _freshness() -> FeatureFreshness:
    """강제흐름 특징별 최신성 정책을 반환한다."""
    return FeatureFreshness(
        price=timedelta(minutes=1),
        open_interest=timedelta(minutes=6),
        funding=timedelta(minutes=1),
        orderbook=timedelta(seconds=5),
        volume=timedelta(minutes=15),
        baseline_skew=timedelta(minutes=1),
    )


def _forced_config() -> ForcedFlowConfig:
    """테스트 특징에서 롱 신호를 내는 사전 등록 설정을 반환한다."""
    return ForcedFlowConfig(
        rebalance_hours=4,
        signal_threshold=0.1,
        price_scale=0.05,
        oi_scale=0.2,
        funding_scale=0.001,
        min_volume_zscore=0.0,
        weight_oi_price=1.0,
        weight_liquidation=1.0,
        weight_orderbook=1.0,
        weight_funding_crowding=1.0,
        max_snapshot_age=timedelta(minutes=1),
    )


def _carry_config() -> CarryConfig:
    """1.5배 비용 문턱을 사용하는 캐리 설정을 반환한다."""
    return CarryConfig(
        expected_funding_intervals=3,
        basis_capture_ratio=0.5,
        spot_fee_rate=0.0002,
        perpetual_fee_rate=0.0002,
        slippage_rate_per_fill=0.0001,
        min_cost_multiple=1.5,
        max_abs_basis_rate=0.05,
        max_snapshot_age=timedelta(minutes=1),
    )


class TestEvidenceValueContracts:
    """원시 시각·호가·gap 데이터 계약 검증."""

    def test_timed_value_rejects_naive_future_availability_and_nan(self) -> None:
        """timezone 누락·관측 전 가용·NaN을 거부한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="timezone-aware"):
            TimedValue(datetime.now(), datetime.now(), 1.0)
        with pytest.raises(ValueError, match="이전"):
            TimedValue(now, now - timedelta(seconds=1), 1.0)
        with pytest.raises(ValueError, match="유한"):
            TimedValue(now, now, float("nan"))

    def test_orderbook_requires_25_sorted_uncrossed_levels(self) -> None:
        """양쪽 25레벨·정렬·미교차 계약을 강제한다."""
        cutoff = _cutoff()
        with pytest.raises(ValueError, match="25레벨"):
            OrderBookEvidence(cutoff, cutoff, (BookLevel(99, 1),), (BookLevel(101, 1),))
        book = _book(cutoff)
        with pytest.raises(ValueError, match="bid"):
            replace(book, bids=tuple(reversed(book.bids)))
        crossed_bids = (BookLevel(102, 1),) + book.bids[1:]
        with pytest.raises(ValueError, match="교차"):
            replace(book, bids=crossed_bids)

    def test_gap_and_liquidation_validation(self) -> None:
        """gap 종료 순서와 청산 방향·명목을 검증한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="이전"):
            FeedGap(now, now - timedelta(seconds=1), "orderbook")
        with pytest.raises(ValueError, match="side"):
            LiquidationNotional(now, now, "long", 1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="양수"):
            LiquidationNotional(now, now, "buy", 0.0)

    def test_intent_family_shape_is_immutable(self) -> None:
        """캐리는 2다리, 강제흐름은 1다리만 허용한다."""
        cutoff = _cutoff()
        leg = StrategyIntentLeg(SYMBOL, "buy", 1.0, 100.0)
        with pytest.raises(ValueError, match="두 다리"):
            StrategyTradeIntent("carry", "delta_neutral_carry", "delta_neutral", (leg,), "bad", _context(cutoff))
        with pytest.raises(ValueError, match="한 다리"):
            StrategyTradeIntent("flow", "forced_flow", "long", (leg, leg), "bad", _context(cutoff))


class TestForcedFlowAsOf:
    """결정 시점 이전 강제흐름 특징 구성 검증."""

    def test_builds_snapshot_and_long_single_leg_intent(self) -> None:
        """4h 가격·OI, 청산, 25레벨, 완결 거래량으로 1다리 의도를 만든다."""
        cutoff = _cutoff()
        bundle = _bundle(cutoff)
        snapshot = build_forced_flow_snapshot(
            bundle,
            _context(cutoff),
            horizon=timedelta(hours=4),
            freshness=_freshness(),
        )
        assert snapshot is not None
        assert snapshot.price_return == pytest.approx(0.05)
        assert snapshot.open_interest_change == pytest.approx(0.2)
        assert snapshot.liquidation_imbalance == 1.0
        assert snapshot.orderbook_imbalance > 0
        intent = decide_forced_flow_intent(
            bundle,
            _forced_config(),
            _context(cutoff),
            freshness=_freshness(),
            candidate_id="flow-01",
            perpetual_symbol=SYMBOL,
            requested_quantity=0.1,
        )
        assert intent is not None
        assert intent.direction == "long"
        assert len(intent.legs) == 1 and intent.legs[0].side == "buy"
        assert intent.legs[0].reference_price == bundle.orderbook.asks[0].price

    def test_future_values_are_excluded_without_changing_snapshot(self) -> None:
        """cutoff 후 관측·수신 값을 추가해도 결정 특징은 변하지 않는다."""
        cutoff = _cutoff()
        bundle = _bundle(cutoff)
        base = build_forced_flow_snapshot(bundle, _context(cutoff), horizon=timedelta(hours=4), freshness=_freshness())
        future = TimedValue(cutoff + timedelta(seconds=1), cutoff + timedelta(seconds=1), 9999.0)
        changed = replace(bundle, prices=bundle.prices + (future,), open_interest=bundle.open_interest + (future,))
        actual = build_forced_flow_snapshot(changed, _context(cutoff), horizon=timedelta(hours=4), freshness=_freshness())
        assert actual == base

    def test_gap_stale_component_and_missing_baseline_return_no_signal(self) -> None:
        """구간 gap·stale 주문장·기준점 누락은 신호 없음으로 처리한다."""
        cutoff = _cutoff()
        gap = FeedGap(cutoff - timedelta(hours=2), cutoff - timedelta(hours=1), "liquidation")
        assert build_forced_flow_snapshot(
            _bundle(cutoff, feed_gaps=(gap,)), _context(cutoff), horizon=timedelta(hours=4), freshness=_freshness()
        ) is None
        assert build_forced_flow_snapshot(
            _bundle(cutoff, orderbook=_book(cutoff, 6)), _context(cutoff), horizon=timedelta(hours=4), freshness=_freshness()
        ) is None
        current_only = (TimedValue(cutoff - timedelta(seconds=10), cutoff - timedelta(seconds=10), 105.0),)
        assert build_forced_flow_snapshot(
            _bundle(cutoff, prices=current_only), _context(cutoff), horizon=timedelta(hours=4), freshness=_freshness()
        ) is None

    def test_invalid_horizon_volume_window_and_negative_volume_are_rejected(self) -> None:
        """사전 범위 밖 horizon·lookback·음수 거래량을 거부한다."""
        cutoff = _cutoff()
        with pytest.raises(ValueError, match="4h 또는 8h"):
            build_forced_flow_snapshot(_bundle(cutoff), _context(cutoff), horizon=timedelta(hours=1), freshness=_freshness())
        with pytest.raises(ValueError, match="최소 20"):
            build_forced_flow_snapshot(_bundle(cutoff), _context(cutoff), horizon=timedelta(hours=4), freshness=_freshness(), volume_lookback=19)
        volumes = list(_bundle(cutoff).completed_volumes)
        volumes[-2] = replace(volumes[-2], value=-1.0)
        with pytest.raises(ValueError, match="음수"):
            build_forced_flow_snapshot(
                _bundle(cutoff, completed_volumes=tuple(volumes)),
                _context(cutoff),
                horizon=timedelta(hours=4),
                freshness=_freshness(),
            )


class TestCarryAsOf:
    """캐리 시점·비용 문턱·2다리 결정 검증."""

    def test_profitable_carry_creates_spot_buy_and_perpetual_sell(self) -> None:
        """비용 1.5배를 넘는 순캐리는 정확히 두 다리 의도로 변환된다."""
        cutoff = _cutoff()
        snapshot = CarrySnapshot("BTC", 100.0, 101.0, 0.001, cutoff - timedelta(seconds=1))
        intent = decide_carry_intent(
            snapshot,
            _carry_config(),
            _context(cutoff),
            candidate_id="carry-01",
            spot_symbol="BTC/USDT",
            perpetual_symbol=SYMBOL,
            requested_quantity=2.0,
        )
        assert intent is not None and intent.direction == "delta_neutral"
        assert [(leg.symbol, leg.side) for leg in intent.legs] == [
            ("BTC/USDT", "buy"),
            (SYMBOL, "sell"),
        ]

    def test_future_stale_and_insufficient_carry_do_not_enter(self) -> None:
        """미래 입력은 예외, stale·비용 이하 캐리는 신호 없음으로 처리한다."""
        cutoff = _cutoff()
        args = {
            "config": _carry_config(),
            "context": _context(cutoff),
            "candidate_id": "carry-01",
            "spot_symbol": "BTC/USDT",
            "perpetual_symbol": SYMBOL,
            "requested_quantity": 1.0,
        }
        with pytest.raises(ValueError, match="data_cutoff"):
            decide_carry_intent(CarrySnapshot("BTC", 100, 101, 0.001, cutoff + timedelta(seconds=1)), **args)
        assert decide_carry_intent(CarrySnapshot("BTC", 100, 101, 0.001, cutoff - timedelta(minutes=2)), **args) is None
        assert decide_carry_intent(CarrySnapshot("BTC", 100, 100.01, 0.0, cutoff), **args) is None
