from __future__ import annotations

# 소액 실전 파일럿용 fail-closed 안전 가드와 포트폴리오 한도.

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Sequence

from src.exchange.contracts import TradingMode
from src.paper_trading import Position
from src.risk.validation_gate import GateDecision

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "live_guard.db"


@dataclass(frozen=True)
class LivePilotLimits:
    """소액 실전 파일럿의 변경 불가 기본 한도."""

    daily_loss_limit: float = 0.005
    weekly_loss_limit: float = 0.015
    max_drawdown: float = 0.03
    risk_per_trade: float = 0.001
    max_total_stop_risk: float = 0.005
    max_directional_notional_mult: float = 1.0
    max_hedged_gross_notional_mult: float = 2.0
    max_net_delta_mult: float = 0.10
    max_leverage: float = 2.0
    max_consecutive_order_errors: int = 3
    stale_after_seconds: float = 120.0


@dataclass(frozen=True)
class SafetySnapshot:
    """실전 안전 판정에 필요한 현재 외부 상태."""

    equity: float
    observed_at: datetime
    feed_timestamp: datetime | None
    reconciliation_ok: bool | None
    orphan_positions: int | None
    duplicate_orders: int | None
    server_protection_ok: bool | None

    def __post_init__(self) -> None:
        """순자산과 시간 계약을 검증한다."""
        if self.equity < 0:
            raise ValueError("equity는 음수일 수 없습니다.")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at은 timezone-aware여야 합니다.")


@dataclass(frozen=True)
class SafetyDecision:
    """신규 주문 차단과 비상 청산 여부."""

    allowed: bool
    must_flatten: bool
    reasons: tuple[str, ...]
    evaluated_at: datetime


@dataclass(frozen=True)
class TradeRiskProposal:
    """신규 주문의 포트폴리오 위험 입력."""

    symbol: str
    direction: Literal["long", "short"]
    notional: float
    stop_risk: float
    leverage: float
    margin_mode: str = "isolated"
    delta_notional: float | None = None

    def __post_init__(self) -> None:
        """위험 입력이 음수가 아닌지 검증한다."""
        if self.notional <= 0 or self.stop_risk < 0 or self.leverage <= 0:
            raise ValueError("명목가·손절 위험·레버리지 값이 올바르지 않습니다.")


@dataclass(frozen=True)
class PortfolioRiskDecision:
    """포트폴리오 한도 판정."""

    allowed: bool
    reasons: tuple[str, ...]
    total_stop_risk: float
    gross_notional: float
    net_notional: float


@dataclass(frozen=True)
class ScaleDecision:
    """실전 파일럿 증액 판정."""

    allowed: bool
    approved_step: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LiveActivationEvidence:
    """실전 실행기 생성 전 수동 승인 증거."""

    mode: TradingMode
    live_enabled: bool
    manual_approval_token: str
    expected_report_hash: str
    actual_report_hash: str
    strategy_version: str
    report_strategy_version: str
    demo_decision: GateDecision


class LiveActivationGate:
    """데모 통과와 수동 승인 없이는 실전 실행을 허용하지 않는다."""

    def evaluate(self, evidence: LiveActivationEvidence) -> SafetyDecision:
        """실전 실행기 활성화 증거를 fail-closed로 판정한다."""
        reasons: list[str] = []
        try:
            mode = TradingMode(evidence.mode)
        except ValueError:
            mode = None
        if mode is not TradingMode.LIVE:
            reasons.append("runtime mode가 live가 아님")
        if not evidence.live_enabled:
            reasons.append("live_enabled가 false")
        if not evidence.manual_approval_token.strip():
            reasons.append("수동 승인 토큰 없음")
        if (
            not evidence.expected_report_hash.strip()
            or evidence.expected_report_hash != evidence.actual_report_hash
        ):
            reasons.append("검증 리포트 해시 불일치")
        if evidence.strategy_version != evidence.report_strategy_version:
            reasons.append("전략 버전 불일치")
        if evidence.demo_decision.stage != "demo" or not evidence.demo_decision.passed:
            reasons.append("미래 데모 게이트 미통과")
        return SafetyDecision(
            allowed=not reasons,
            must_flatten=False,
            reasons=tuple(reasons),
            evaluated_at=datetime.now(timezone.utc),
        )


class LivePilotGuard:
    """손실·대사·시세·주문 오류를 영속 추적하는 fail-closed 킬스위치."""

    def __init__(
        self,
        initial_equity: float,
        limits: LivePilotLimits | None = None,
        db_path: Path | None = None,
    ) -> None:
        """실전 파일럿 가드를 초기화한다."""
        if initial_equity <= 0:
            raise ValueError("initial_equity는 0보다 커야 합니다.")
        self.initial_equity = initial_equity
        self.limits = limits or LivePilotLimits()
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        with self._connect() as conn:
            if self._state(conn, "peak_equity") is None:
                self._set_state(conn, "peak_equity", str(initial_equity))
            if self._state(conn, "consecutive_order_errors") is None:
                self._set_state(conn, "consecutive_order_errors", "0")

    def _connect(self) -> sqlite3.Connection:
        """가드 SQLite 연결을 반환한다."""
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        """가드 상태와 순자산 시계열 테이블을 만든다."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_guard_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_equity (
                    recorded_at TEXT PRIMARY KEY,
                    equity REAL NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def _state(conn: sqlite3.Connection, key: str) -> str | None:
        """영속 상태 값을 조회한다."""
        row = conn.execute(
            "SELECT value FROM live_guard_state WHERE key=?",
            (key,),
        ).fetchone()
        return str(row[0]) if row else None

    @staticmethod
    def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
        """영속 상태 값을 저장한다."""
        conn.execute(
            """INSERT OR REPLACE INTO live_guard_state(key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def record_order_result(self, success: bool) -> int:
        """주문 성공 여부를 기록하고 연속 오류 횟수를 반환한다."""
        with self._connect() as conn:
            current = int(self._state(conn, "consecutive_order_errors") or "0")
            current = 0 if success else current + 1
            self._set_state(conn, "consecutive_order_errors", str(current))
        return current

    def evaluate(self, snapshot: SafetySnapshot) -> SafetyDecision:
        """현재 실전 상태를 검사하고 하나라도 불명확하면 신규 주문을 차단한다."""
        now = snapshot.observed_at.astimezone(timezone.utc)
        reasons: list[str] = []
        with self._connect() as conn:
            peak = max(
                float(self._state(conn, "peak_equity") or self.initial_equity),
                snapshot.equity,
            )
            consecutive_errors = int(
                self._state(conn, "consecutive_order_errors") or "0"
            )
            already_tripped = self._state(conn, "tripped") == "1"
            trip_reasons = json.loads(
                self._state(conn, "trip_reasons") or "[]"
            )

            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = (now - timedelta(days=now.weekday())).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            daily_start_equity = self._period_start_equity(
                conn,
                day_start,
                self.initial_equity,
            )
            weekly_start_equity = self._period_start_equity(
                conn,
                week_start,
                self.initial_equity,
            )
            conn.execute(
                "INSERT OR REPLACE INTO live_equity(recorded_at, equity) VALUES (?, ?)",
                (now.isoformat(), snapshot.equity),
            )
            self._set_state(conn, "peak_equity", str(peak))

        if already_tripped:
            reasons.extend(
                str(reason) for reason in trip_reasons
            )
        if snapshot.equity <= daily_start_equity * (
            1 - self.limits.daily_loss_limit
        ):
            reasons.append("일일 손실 0.5% 한도 도달")
        if snapshot.equity <= weekly_start_equity * (
            1 - self.limits.weekly_loss_limit
        ):
            reasons.append("주간 손실 1.5% 한도 도달")
        if peak > 0 and (peak - snapshot.equity) / peak >= self.limits.max_drawdown:
            reasons.append("고점 대비 3% 낙폭 도달")
        if snapshot.feed_timestamp is None:
            reasons.append("시세 타임스탬프 없음")
        else:
            feed_time = snapshot.feed_timestamp
            if feed_time.tzinfo is None:
                reasons.append("시세 타임스탬프 timezone 없음")
            else:
                feed_age = (
                    now - feed_time.astimezone(timezone.utc)
                ).total_seconds()
                if feed_age < -5:
                    reasons.append("시세 타임스탬프가 미래임")
                elif feed_age > self.limits.stale_after_seconds:
                    reasons.append("시세 데이터 stale")
        if snapshot.reconciliation_ok is not True:
            reasons.append("주문·포지션·잔고 대사 불일치 또는 미확인")
        if snapshot.orphan_positions is None or snapshot.orphan_positions > 0:
            reasons.append("고아 포지션 존재 또는 미확인")
        if snapshot.duplicate_orders is None or snapshot.duplicate_orders > 0:
            reasons.append("중복 주문 존재 또는 미확인")
        if snapshot.server_protection_ok is not True:
            reasons.append("거래소 서버 보호주문 없음 또는 미확인")
        if consecutive_errors >= self.limits.max_consecutive_order_errors:
            reasons.append("연속 주문 오류 한도 도달")

        decision = SafetyDecision(
            allowed=not reasons,
            must_flatten=bool(reasons),
            reasons=tuple(reasons),
            evaluated_at=now,
        )
        if reasons:
            with self._connect() as conn:
                self._set_state(conn, "tripped", "1")
                self._set_state(
                    conn,
                    "trip_reasons",
                    json.dumps(list(dict.fromkeys(reasons)), ensure_ascii=False),
                )
            logger.critical("실전 킬스위치 발동: %s", ", ".join(reasons))
        return decision

    @staticmethod
    def _period_start_equity(
        conn: sqlite3.Connection,
        start: datetime,
        fallback: float,
    ) -> float:
        """기간 시작 이후 첫 순자산을 반환한다."""
        row = conn.execute(
            """SELECT equity FROM live_equity
               WHERE recorded_at>=? ORDER BY recorded_at LIMIT 1""",
            (start.isoformat(),),
        ).fetchone()
        return float(row[0]) if row else fallback

    def reset_trip(self, manual_approval: bool = False) -> bool:
        """수동 승인 때만 영속 킬스위치를 해제한다."""
        if not manual_approval:
            logger.warning("수동 승인 없는 실전 킬스위치 해제 거부")
            return False
        with self._connect() as conn:
            self._set_state(conn, "tripped", "0")
            self._set_state(conn, "trip_reasons", "[]")
            self._set_state(conn, "consecutive_order_errors", "0")
        return True


class PortfolioRiskGuard:
    """실전 파일럿의 거래당·총손절·명목·델타 한도를 검사한다."""

    def __init__(
        self,
        pilot_equity: float,
        limits: LivePilotLimits | None = None,
    ) -> None:
        """포트폴리오 위험 가드를 초기화한다."""
        if pilot_equity <= 0:
            raise ValueError("pilot_equity는 0보다 커야 합니다.")
        self.pilot_equity = pilot_equity
        self.limits = limits or LivePilotLimits()

    def evaluate(
        self,
        proposal: TradeRiskProposal,
        positions: Sequence[Position],
        hedged_strategy: bool = False,
    ) -> PortfolioRiskDecision:
        """신규 주문 포함 포트폴리오가 모든 파일럿 한도 안인지 판정한다."""
        signed_existing = [
            position.entry_price
            * position.qty
            * (1 if position.direction == "long" else -1)
            for position in positions
        ]
        proposed_delta = (
            proposal.delta_notional
            if proposal.delta_notional is not None
            else proposal.notional * (1 if proposal.direction == "long" else -1)
        )
        gross_notional = sum(abs(value) for value in signed_existing) + proposal.notional
        net_notional = sum(signed_existing) + proposed_delta
        existing_stop_risk = sum(
            abs(position.entry_price - position.stop_loss) * position.qty
            for position in positions
        )
        total_stop_risk = existing_stop_risk + proposal.stop_risk
        reasons: list[str] = []
        if proposal.stop_risk > self.pilot_equity * self.limits.risk_per_trade:
            reasons.append("거래당 위험 0.1% 초과")
        if total_stop_risk > self.pilot_equity * self.limits.max_total_stop_risk:
            reasons.append("동시 총 손절 위험 0.5% 초과")
        if proposal.leverage > self.limits.max_leverage:
            reasons.append("최대 레버리지 2배 초과")
        if proposal.margin_mode.lower() != "isolated":
            reasons.append("격리마진이 아님")
        if hedged_strategy:
            if gross_notional > (
                self.pilot_equity * self.limits.max_hedged_gross_notional_mult
            ):
                reasons.append("델타중립 합산 명목노출 2배 초과")
            if abs(net_notional) > self.pilot_equity * self.limits.max_net_delta_mult:
                reasons.append("델타중립 순델타 10% 초과")
        elif gross_notional > (
            self.pilot_equity * self.limits.max_directional_notional_mult
        ):
            reasons.append("방향성 명목노출 1배 초과")
        return PortfolioRiskDecision(
            allowed=not reasons,
            reasons=tuple(reasons),
            total_stop_risk=round(total_stop_risk, 8),
            gross_notional=round(gross_notional, 8),
            net_notional=round(net_notional, 8),
        )


def calculate_pilot_capital_krw(
    investable_assets_krw: float,
    max_capital_krw: float = 1_000_000,
    max_investable_fraction: float = 0.05,
) -> float:
    """100만원과 투자 가능 자산 5% 중 작은 파일럿 자본을 계산한다."""
    if investable_assets_krw < 0 or max_capital_krw <= 0:
        raise ValueError("자산과 파일럿 상한 값이 올바르지 않습니다.")
    if not 0 < max_investable_fraction <= 1:
        raise ValueError("투자 가능 비율은 0과 1 사이여야 합니다.")
    return round(
        min(max_capital_krw, investable_assets_krw * max_investable_fraction),
        8,
    )


def evaluate_scale_up(
    live_calendar_days: int,
    days_since_last_scale: int,
    requested_step: float,
    live_performance_decision: GateDecision,
    minimum_live_days: int = 90,
    review_interval_days: int = 30,
    max_scale_step: float = 0.25,
) -> ScaleDecision:
    """90일 실전 증거와 30일 검토 주기에서 최대 25% 증액을 판정한다."""
    reasons: list[str] = []
    if live_calendar_days < minimum_live_days:
        reasons.append("최소 실전 검증 90일 미달")
    if days_since_last_scale < review_interval_days:
        reasons.append("직전 증액 후 30일 미경과")
    if requested_step <= 0 or requested_step > max_scale_step:
        reasons.append("증액 폭이 0 초과 25% 이하여야 함")
    if (
        live_performance_decision.stage != "demo"
        or not live_performance_decision.passed
    ):
        reasons.append("실전 성과가 데모 게이트 기준 미달")
    return ScaleDecision(
        allowed=not reasons,
        approved_step=round(requested_step, 8) if not reasons else 0.0,
        reasons=tuple(reasons),
    )
