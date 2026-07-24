from __future__ import annotations

"""신규 후보 재생에 실제 주문·자본 제약을 적용하는 순수 유틸리티."""

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Literal, Mapping


def _positive_decimal(value: float, field_name: str) -> Decimal:
    """양의 유한 실수를 정확한 Decimal로 변환한다."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name}은(는) 양의 유한한 숫자여야 합니다")
    return Decimal(str(value))


def _round_down(value: Decimal, step: Decimal) -> Decimal:
    """값을 step 배수로 내림한다."""
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def _round_price(price: Decimal, tick: Decimal, side: Literal["buy", "sell"]) -> Decimal:
    """재생에 보수적인 방향으로 가격을 tick 배수화한다."""
    rounding = ROUND_CEILING if side == "buy" else ROUND_FLOOR
    return (price / tick).to_integral_value(rounding=rounding) * tick


@dataclass(frozen=True)
class InstrumentRules:
    """재생 시 적용할 거래소 심볼별 주문 규칙."""

    symbol: str
    minimum_quantity: float
    quantity_step: float
    tick_size: float
    minimum_notional: float = 0.0
    maximum_quantity: float | None = None

    def __post_init__(self) -> None:
        """수량·가격 단위와 주문 한도를 검증한다."""
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        _positive_decimal(self.minimum_quantity, "minimum_quantity")
        _positive_decimal(self.quantity_step, "quantity_step")
        _positive_decimal(self.tick_size, "tick_size")
        if not math.isfinite(self.minimum_notional) or self.minimum_notional < 0:
            raise ValueError("minimum_notional은 0 이상의 유한한 숫자여야 합니다")
        if self.maximum_quantity is not None:
            _positive_decimal(self.maximum_quantity, "maximum_quantity")
            if self.maximum_quantity < self.minimum_quantity:
                raise ValueError("maximum_quantity가 minimum_quantity보다 작습니다")


@dataclass(frozen=True)
class ReplayOrderLeg:
    """하나의 전략 포지션을 구성하는 주문 다리."""

    symbol: str
    side: Literal["buy", "sell"]
    requested_quantity: float
    reference_price: float

    def __post_init__(self) -> None:
        """주문 방향·수량·가격을 검증한다."""
        if not self.symbol.strip():
            raise ValueError("symbol은 비어 있을 수 없습니다")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"지원하지 않는 side입니다: {self.side}")
        _positive_decimal(self.requested_quantity, "requested_quantity")
        _positive_decimal(self.reference_price, "reference_price")


@dataclass(frozen=True)
class ReplayTradeIntent:
    """한 슬롯에서 원자적으로 수락하거나 거절할 단일·다중 다리 주문."""

    position_id: str
    legs: tuple[ReplayOrderLeg, ...]

    def __post_init__(self) -> None:
        """포지션 식별자와 주문 다리를 검증한다."""
        if not self.position_id.strip():
            raise ValueError("position_id는 비어 있을 수 없습니다")
        if not self.legs:
            raise ValueError("최소 한 개 주문 다리가 필요합니다")
        symbols = [leg.symbol for leg in self.legs]
        if len(symbols) != len(set(symbols)):
            raise ValueError("한 포지션에서 동일 심볼 다리를 중복할 수 없습니다")


@dataclass(frozen=True)
class ReplayPosition:
    """현재 재생 포트폴리오에서 슬롯과 자본을 점유하는 포지션."""

    position_id: str
    gross_notional: float
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        """포지션 상태를 검증한다."""
        if not self.position_id.strip():
            raise ValueError("position_id는 비어 있을 수 없습니다")
        _positive_decimal(self.gross_notional, "gross_notional")
        if not self.symbols:
            raise ValueError("포지션에는 최소 한 개 심볼이 필요합니다")


@dataclass(frozen=True)
class ReplayPortfolioState:
    """제약 적용 시점의 자본과 열린 포지션 스냅샷."""

    capital: float
    positions: tuple[ReplayPosition, ...] = ()

    def __post_init__(self) -> None:
        """자본과 포지션 식별자 중복을 검증한다."""
        _positive_decimal(self.capital, "capital")
        position_ids = [position.position_id for position in self.positions]
        if len(position_ids) != len(set(position_ids)):
            raise ValueError("position_id가 중복됐습니다")

    @property
    def used_gross_notional(self) -> float:
        """열린 포지션의 합산 명목노출을 반환한다."""
        return round(sum(position.gross_notional for position in self.positions), 8)


@dataclass(frozen=True)
class ReplayExecutionPolicy:
    """슬롯과 자본 사용 한도를 정의한다."""

    maximum_position_slots: int
    maximum_leverage: float
    capital_utilization: float = 1.0

    def __post_init__(self) -> None:
        """슬롯·레버리지·자본 사용률을 검증한다."""
        if self.maximum_position_slots <= 0:
            raise ValueError("maximum_position_slots는 양수여야 합니다")
        _positive_decimal(self.maximum_leverage, "maximum_leverage")
        if not 0 < self.capital_utilization <= 1:
            raise ValueError("capital_utilization은 0~1 사이여야 합니다")


@dataclass(frozen=True)
class ConstrainedOrderLeg:
    """거래소 단위와 자본 한도를 적용한 주문 다리."""

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    notional: float


@dataclass(frozen=True)
class ReplayConstraintResult:
    """원자적 주문 제약 결과."""

    accepted: bool
    reason: str
    position_id: str
    legs: tuple[ConstrainedOrderLeg, ...] = ()
    gross_notional: float = 0.0
    remaining_notional_capacity: float = 0.0


def _constrain_legs(
    legs: tuple[ReplayOrderLeg, ...],
    rules_by_symbol: Mapping[str, InstrumentRules],
    scale: Decimal,
) -> tuple[ConstrainedOrderLeg, ...] | None:
    """모든 주문 다리를 동일 비율로 줄이고 거래소 단위에 맞춘다."""
    constrained: list[ConstrainedOrderLeg] = []
    for leg in legs:
        try:
            rules = rules_by_symbol[leg.symbol]
        except KeyError as exc:
            raise ValueError(f"심볼 주문 규칙이 없습니다: {leg.symbol}") from exc
        if rules.symbol != leg.symbol:
            raise ValueError(f"주문 규칙 symbol 불일치: {leg.symbol}")
        quantity_step = _positive_decimal(rules.quantity_step, "quantity_step")
        requested_quantity = _positive_decimal(
            leg.requested_quantity,
            "requested_quantity",
        )
        quantity = _round_down(requested_quantity * scale, quantity_step)
        if rules.maximum_quantity is not None:
            quantity = min(
                quantity,
                _round_down(
                    _positive_decimal(rules.maximum_quantity, "maximum_quantity"),
                    quantity_step,
                ),
            )
        minimum_quantity = _positive_decimal(
            rules.minimum_quantity,
            "minimum_quantity",
        )
        if quantity < minimum_quantity:
            return None
        price = _round_price(
            _positive_decimal(leg.reference_price, "reference_price"),
            _positive_decimal(rules.tick_size, "tick_size"),
            leg.side,
        )
        notional = quantity * price
        if notional < Decimal(str(rules.minimum_notional)):
            return None
        constrained.append(
            ConstrainedOrderLeg(
                symbol=leg.symbol,
                side=leg.side,
                quantity=round(float(quantity), 8),
                price=round(float(price), 8),
                notional=round(float(notional), 8),
            )
        )
    return tuple(constrained)


def apply_execution_constraints(
    intent: ReplayTradeIntent,
    rules_by_symbol: Mapping[str, InstrumentRules],
    state: ReplayPortfolioState,
    policy: ReplayExecutionPolicy,
) -> ReplayConstraintResult:
    """슬롯·거래소 단위·자본 한도를 적용해 주문을 원자적으로 수락한다."""
    gross_limit = (
        Decimal(str(state.capital))
        * Decimal(str(policy.maximum_leverage))
        * Decimal(str(policy.capital_utilization))
    )
    used = Decimal(str(state.used_gross_notional))
    available = max(Decimal("0"), gross_limit - used)
    base_result = {
        "position_id": intent.position_id,
        "remaining_notional_capacity": round(float(available), 8),
    }
    if any(position.position_id == intent.position_id for position in state.positions):
        return ReplayConstraintResult(
            accepted=False,
            reason="position_id 중복",
            **base_result,
        )
    if len(state.positions) >= policy.maximum_position_slots:
        return ReplayConstraintResult(
            accepted=False,
            reason="포지션 슬롯 한도",
            **base_result,
        )
    if available <= 0:
        return ReplayConstraintResult(
            accepted=False,
            reason="자본 명목노출 한도",
            **base_result,
        )

    requested_gross = Decimal("0")
    for leg in intent.legs:
        try:
            rules = rules_by_symbol[leg.symbol]
        except KeyError as exc:
            raise ValueError(f"심볼 주문 규칙이 없습니다: {leg.symbol}") from exc
        price = _round_price(
            _positive_decimal(leg.reference_price, "reference_price"),
            _positive_decimal(rules.tick_size, "tick_size"),
            leg.side,
        )
        requested_gross += (
            _positive_decimal(leg.requested_quantity, "requested_quantity")
            * price
        )
    scale = min(Decimal("1"), available / requested_gross)
    constrained = _constrain_legs(intent.legs, rules_by_symbol, scale)
    if constrained is None:
        reason = (
            "최소 수량 또는 최소 주문금액 미달"
            if scale == Decimal("1")
            else "자본 축소 후 최소 수량 또는 최소 주문금액 미달"
        )
        return ReplayConstraintResult(
            accepted=False,
            reason=reason,
            **base_result,
        )
    gross_notional = sum(Decimal(str(leg.notional)) for leg in constrained)
    if gross_notional > available:
        raise AssertionError("제약 적용 주문이 가용 명목노출을 초과했습니다")
    return ReplayConstraintResult(
        accepted=True,
        reason="accepted",
        position_id=intent.position_id,
        legs=constrained,
        gross_notional=round(float(gross_notional), 8),
        remaining_notional_capacity=round(float(available - gross_notional), 8),
    )


def add_replay_position(
    state: ReplayPortfolioState,
    result: ReplayConstraintResult,
) -> ReplayPortfolioState:
    """수락된 제약 결과를 새 불변 포트폴리오 상태로 반영한다."""
    if not result.accepted or not result.legs or result.gross_notional <= 0:
        raise ValueError("수락된 주문 결과만 포지션에 반영할 수 있습니다")
    position = ReplayPosition(
        position_id=result.position_id,
        gross_notional=result.gross_notional,
        symbols=tuple(leg.symbol for leg in result.legs),
    )
    return ReplayPortfolioState(
        capital=state.capital,
        positions=state.positions + (position,),
    )


def close_replay_position(
    state: ReplayPortfolioState,
    position_id: str,
) -> ReplayPortfolioState:
    """지정 포지션을 제거해 슬롯과 명목노출을 해제한다."""
    remaining = tuple(
        position
        for position in state.positions
        if position.position_id != position_id
    )
    if len(remaining) == len(state.positions):
        raise ValueError(f"열린 position_id가 없습니다: {position_id}")
    return ReplayPortfolioState(capital=state.capital, positions=remaining)
