from __future__ import annotations

# 주문장 체결·동적 비용·순자산 성과를 영속 추적하는 페이퍼 엔진.

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping

import numpy as np

from src.exchange.contracts import FeeRateSnapshot
from src.paper_trading import Position, TradingEngine
from src.paper_trading.execution_model import (
    ExecutionReport,
    Fill,
    OrderBookExecutionModel,
    OrderBookSnapshot,
    OrderRequest,
    OrderState,
    report_maker_quantity,
    report_taker_quantity,
    report_total_fee,
    report_total_slippage_cost,
)

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "paper_trades.db"

SLIPPAGE = 0.0005  # 0.05%
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.00055  # 0.055%
# 펀딩비 보수 가정 (Bybit 00/08/16 UTC, 통상 0.01%/8h — 방향 무관 비용으로 처리)
FUNDING_RATE = 0.0001
FUNDING_HOURS = (0, 8, 16)


@dataclass(frozen=True)
class FeeSchedule:
    """계정·심볼별 수수료 스냅샷."""

    maker_rate: float = MAKER_FEE
    taker_rate: float = TAKER_FEE
    source: str = "bybit_default"
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """음수가 아닌 수수료율인지 검증한다."""
        if self.maker_rate < 0 or self.taker_rate < 0:
            raise ValueError("수수료율은 음수일 수 없습니다.")


FeeProvider = Callable[
    [str],
    "FeeSchedule | FeeRateSnapshot | Mapping[str, float | str]",
]


def _funding_events(entry_time: datetime, exit_time: datetime) -> int:
    """보유 구간 ``(entry, exit]``의 00/08/16 UTC 정산 경계 수를 계산한다."""
    if exit_time <= entry_time:
        return 0
    if entry_time.tzinfo is None or exit_time.tzinfo is None:
        raise ValueError("펀딩 계산 시각은 timezone-aware여야 합니다.")
    period_seconds = int(timedelta(hours=8).total_seconds())
    entry_epoch = entry_time.astimezone(timezone.utc).timestamp()
    exit_epoch = exit_time.astimezone(timezone.utc).timestamp()
    return int(exit_epoch // period_seconds - entry_epoch // period_seconds)


def _init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """SQLite DB 초기화 및 테이블 생성.

    Args:
        db_path: DB 파일 경로. None이면 기본 경로(DB_PATH) 사용.

    Returns:
        SQLite 연결 객체
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL,
            qty REAL, pnl REAL, pnl_pct REAL,
            margin REAL,
            entry_time TEXT, exit_time TEXT, status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS engine_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id TEXT PRIMARY KEY,
            symbol TEXT, direction TEXT,
            entry_price REAL, qty REAL,
            stop_loss REAL, take_profit REAL,
            margin REAL, entry_time TEXT
        )
    """)
    # 자동 학습용 진입조건 컬럼 멱등 추가 (기존 DB 하위호환)
    _migrate_columns(conn, "trades", {
        "entry_score": "REAL", "entry_session": "TEXT",
        "c_trend": "INT", "c_zone": "INT", "c_kill_zone": "INT",
        "c_ote": "INT", "c_volume": "INT", "c_rr": "INT",
        "entry_rr": "REAL", "risk_amount": "REAL", "r_multiple": "REAL",
        "funding_cost": "REAL",
        "gross_pnl": "REAL", "net_pnl": "REAL",
        "entry_fee": "REAL", "exit_fee": "REAL", "total_fee": "REAL",
        "funding_pnl": "REAL", "slippage_cost": "REAL",
        "maker_qty": "REAL", "taker_qty": "REAL",
        "requested_qty": "REAL", "filled_qty": "REAL",
        "funding_source": "TEXT", "funding_rate_assumption": "REAL",
    })
    _migrate_columns(conn, "open_positions", {
        "entry_score": "REAL", "entry_session": "TEXT",
        "entry_checks_json": "TEXT", "entry_rr": "REAL", "risk_amount": "REAL",
        "entry_fee": "REAL", "entry_slippage_cost": "REAL",
        "entry_requested_price": "REAL", "entry_liquidity": "TEXT",
        "requested_qty": "REAL",
    })
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fee_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            maker_rate REAL NOT NULL,
            taker_rate REAL NOT NULL,
            source TEXT NOT NULL,
            captured_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            funding_time TEXT NOT NULL,
            rate REAL NOT NULL,
            mark_price REAL,
            source TEXT NOT NULL,
            UNIQUE(symbol, funding_time, source)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS execution_reports (
            order_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            state TEXT NOT NULL,
            requested_qty REAL NOT NULL,
            filled_qty REAL NOT NULL,
            average_price REAL,
            maker_qty REAL NOT NULL,
            taker_qty REAL NOT NULL,
            fee REAL NOT NULL,
            slippage_cost REAL NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            fill_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            qty REAL NOT NULL,
            liquidity TEXT NOT NULL,
            fee_rate REAL NOT NULL,
            fee REAL NOT NULL,
            slippage_cost REAL NOT NULL,
            adverse_selection_cost REAL NOT NULL,
            filled_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            cash REAL NOT NULL,
            margin REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            equity REAL NOT NULL,
            UNIQUE(recorded_at)
        )
    """)
    conn.commit()
    return conn


def _migrate_columns(
    conn: sqlite3.Connection, table: str, cols: dict[str, str]
) -> None:
    """테이블에 없는 컬럼만 ALTER TABLE로 추가한다 (멱등).

    Args:
        conn: SQLite 연결
        table: 대상 테이블명
        cols: {컬럼명: 타입} 매핑
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ctype in cols.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")


def _generate_position_id() -> str:
    """UUID 기반 포지션 고유 ID 생성."""
    return str(uuid.uuid4())[:8]


class PaperEngine(TradingEngine):
    """페이퍼 트레이딩 엔진 — TradingEngine 인터페이스 구현."""

    def __init__(
        self,
        initial_balance: float = 1250.0,
        db_path: Path | None = None,
        fee_provider: FeeProvider | None = None,
        execution_model: OrderBookExecutionModel | None = None,
    ) -> None:
        """
        페이퍼 엔진 초기화.

        Args:
            initial_balance: 초기 잔고 (USDT)
            db_path: DB 파일 경로. None이면 기본 경로 사용.
            fee_provider: 심볼별 메이커·테이커 수수료 공급 함수.
            execution_model: 주문장 기반 체결 모델.
        """
        self.initial_balance: float = initial_balance
        self._positions: list[Position] = []
        self.conn: sqlite3.Connection = _init_db(db_path)
        self._fee_provider = fee_provider
        self._fee_overrides: dict[str, FeeSchedule] = {}
        self.execution_model = execution_model or OrderBookExecutionModel()
        self._on_trade_callbacks: list[Callable[[float, str, Position], None]] = []

        # DB에서 잔고 복원 시도 → 없으면 initial_balance 사용
        restored = self._restore_balance()
        self._balance: float = restored if restored is not None else initial_balance

        # 봇 재시작 시 기존 포지션 복원
        self._restore_positions()
        logger.info(
            "페이퍼 엔진 초기화: 잔고=%.2f USDT (복원=%s)",
            self._balance,
            restored is not None,
        )

    # ------------------------------------------------------------------
    # 콜백 등록 (Discord 알림 연동 등)
    # ------------------------------------------------------------------

    def register_on_trade(
        self, callback: Callable[[float, str, Position], None]
    ) -> None:
        """
        거래 완료 시 호출될 콜백 등록.

        Args:
            callback: (pnl, reason, position) 인자를 받는 콜백 함수
        """
        self._on_trade_callbacks.append(callback)
        logger.info("거래 콜백 등록: %s", callback.__name__)

    def _fire_on_trade(
        self, pnl: float, reason: str, position: Position
    ) -> None:
        """등록된 모든 거래 콜백 실행."""
        for cb in self._on_trade_callbacks:
            try:
                cb(pnl, reason, position)
            except Exception as e:
                logger.error("거래 콜백 실행 오류: %s", e)

    # ------------------------------------------------------------------
    # 잔고 영속화 (SQLite)
    # ------------------------------------------------------------------

    def _restore_balance(self) -> float | None:
        """DB에서 마지막 잔고를 복원한다.

        Returns:
            복원된 잔고 또는 DB에 없으면 None
        """
        row = self.conn.execute(
            "SELECT value FROM engine_state WHERE key = 'balance'"
        ).fetchone()
        if row is not None:
            balance = float(row[0])
            logger.info("DB에서 잔고 복원: %.2f USDT", balance)
            return balance
        return None

    def _save_balance(self) -> None:
        """현재 잔고를 DB에 저장한다."""
        self.conn.execute(
            "INSERT OR REPLACE INTO engine_state(key, value) VALUES('balance', ?)",
            (str(self._balance),),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # 포지션 영속화 (SQLite)
    # ------------------------------------------------------------------

    def _save_position(self, pos: Position) -> None:
        """열린 포지션을 DB에 저장 (진입조건 포함)."""
        checks_json = json.dumps(pos.entry_checks) if pos.entry_checks else None
        self.conn.execute(
            """INSERT OR REPLACE INTO open_positions
               (id, symbol, direction, entry_price, qty, stop_loss, take_profit, margin, entry_time,
                entry_score, entry_session, entry_checks_json, entry_rr, risk_amount,
                entry_fee, entry_slippage_cost, entry_requested_price, entry_liquidity,
                requested_qty)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos.id,
                pos.symbol,
                pos.direction,
                pos.entry_price,
                pos.qty,
                pos.stop_loss,
                pos.take_profit,
                pos.margin,
                pos.entry_time.isoformat(),
                pos.entry_score,
                pos.entry_session,
                checks_json,
                pos.entry_rr,
                pos.risk_amount,
                pos.entry_fee,
                pos.entry_slippage_cost,
                pos.entry_requested_price,
                pos.entry_liquidity,
                pos.requested_qty,
            ),
        )
        self.conn.commit()

    def _remove_position_from_db(self, position_id: str) -> None:
        """청산된 포지션을 open_positions에서 제거."""
        self.conn.execute(
            "DELETE FROM open_positions WHERE id = ?", (position_id,)
        )
        self.conn.commit()

    def _restore_positions(self) -> None:
        """봇 재시작 시 열린 포지션 복원 (진입조건 포함)."""
        rows = self.conn.execute(
            "SELECT id, symbol, direction, entry_price, qty, stop_loss, take_profit, margin, entry_time, "
            "entry_score, entry_session, entry_checks_json, entry_rr, risk_amount, "
            "entry_fee, entry_slippage_cost, entry_requested_price, entry_liquidity, requested_qty "
            "FROM open_positions"
        ).fetchall()
        for row in rows:
            checks = json.loads(row[11]) if row[11] else None
            pos = Position(
                id=row[0],
                symbol=row[1],
                direction=row[2],
                entry_price=row[3],
                qty=row[4],
                stop_loss=row[5],
                take_profit=row[6],
                margin=row[7],
                entry_time=datetime.fromisoformat(row[8]),
                entry_score=row[9],
                entry_session=row[10],
                entry_checks=checks,
                entry_rr=row[12],
                risk_amount=row[13],
                entry_fee=row[14] or 0.0,
                entry_slippage_cost=row[15] or 0.0,
                entry_requested_price=row[16],
                entry_liquidity=row[17] or "taker",
                requested_qty=row[18],
            )
            self._positions.append(pos)
        if rows:
            logger.info("포지션 복원: %d개", len(rows))

    # ------------------------------------------------------------------
    # 슬리피지 / 수수료
    # ------------------------------------------------------------------

    def _apply_slippage(
        self, price: float, direction: Literal["long", "short"], is_entry: bool
    ) -> float:
        """
        슬리피지 적용 — 항상 불리한 방향으로.

        Args:
            price: 원래 가격
            direction: 포지션 방향
            is_entry: 진입 여부

        Returns:
            슬리피지 적용된 가격
        """
        # Long 진입 / Short 청산: 불리한 방향 = 더 높은 가격
        if (direction == "long") == is_entry:
            return round(price * (1 + SLIPPAGE), 8)
        return round(price * (1 - SLIPPAGE), 8)

    def set_fee_rates(
        self,
        symbol: str,
        maker_rate: float,
        taker_rate: float,
        source: str = "account_api",
        captured_at: datetime | None = None,
    ) -> FeeSchedule:
        """동적 계정 수수료율을 저장하고 이후 체결에 적용한다.

        Args:
            symbol: 거래 심볼.
            maker_rate: 메이커 수수료율.
            taker_rate: 테이커 수수료율.
            source: 수수료 출처.
            captured_at: 조회 시각.

        Returns:
            저장된 수수료 스냅샷.
        """
        schedule = FeeSchedule(
            maker_rate=float(maker_rate),
            taker_rate=float(taker_rate),
            source=source,
            captured_at=captured_at or datetime.now(timezone.utc),
        )
        self._fee_overrides[symbol] = schedule
        self._persist_fee_schedule(symbol, schedule)
        return schedule

    def _get_fee_schedule(self, symbol: str) -> FeeSchedule:
        """심볼별 최신 수수료율을 가져오고 스냅샷으로 저장한다."""
        if symbol in self._fee_overrides:
            return self._fee_overrides[symbol]
        schedule: FeeSchedule
        if self._fee_provider is None:
            schedule = FeeSchedule()
        else:
            raw = self._fee_provider(symbol)
            if isinstance(raw, FeeSchedule):
                schedule = raw
            elif isinstance(raw, FeeRateSnapshot):
                schedule = FeeSchedule(
                    maker_rate=raw.maker_rate,
                    taker_rate=raw.taker_rate,
                    source=raw.source,
                    captured_at=raw.exchange_timestamp or raw.receive_timestamp,
                )
            else:
                schedule = FeeSchedule(
                    maker_rate=float(raw.get("maker_rate", MAKER_FEE)),
                    taker_rate=float(raw.get("taker_rate", TAKER_FEE)),
                    source=str(raw.get("source", "account_api")),
                )
        self._fee_overrides[symbol] = schedule
        self._persist_fee_schedule(symbol, schedule)
        return schedule

    def _persist_fee_schedule(self, symbol: str, schedule: FeeSchedule) -> None:
        """수수료 스냅샷을 DB에 영구 저장한다."""
        self.conn.execute(
            """INSERT INTO fee_snapshots
               (symbol, maker_rate, taker_rate, source, captured_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                symbol,
                schedule.maker_rate,
                schedule.taker_rate,
                schedule.source,
                schedule.captured_at.isoformat(),
            ),
        )
        self.conn.commit()

    def _fee(
        self,
        notional: float,
        symbol: str = "",
        liquidity: Literal["maker", "taker"] = "taker",
    ) -> float:
        """
        수수료 계산.

        Args:
            notional: 명목 금액
            symbol: 거래 심볼.
            liquidity: 유동성 역할.

        Returns:
            수수료 금액
        """
        schedule = self._get_fee_schedule(symbol)
        rate = schedule.maker_rate if liquidity == "maker" else schedule.taker_rate
        return round(notional * rate, 8)

    def submit_order(
        self,
        request: OrderRequest,
        orderbook: OrderBookSnapshot,
        maker_available_qty: float = 0.0,
    ) -> ExecutionReport:
        """주문장 체결 모델로 주문을 실행하고 상세 결과를 저장한다.

        Args:
            request: 공통 주문 요청.
            orderbook: 체결에 사용할 동일 시점 주문장.
            maker_available_qty: 비시장성 지정가에서 관측된 거래 가능 수량.

        Returns:
            실행 보고서.
        """
        report = self.execution_model.execute(
            request,
            orderbook,
            maker_available_qty=maker_available_qty,
        )
        schedule = self._get_fee_schedule(request.symbol)
        enriched_fills: list[Fill] = []
        for fill in report.fills:
            rate = (
                schedule.maker_rate
                if fill.liquidity == "maker"
                else schedule.taker_rate
            )
            enriched_fills.append(
                Fill(
                    fill_id=fill.fill_id,
                    order_id=fill.order_id,
                    client_order_id=fill.client_order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    price=fill.price,
                    quantity=fill.quantity,
                    liquidity=fill.liquidity,
                    fee=round(fill.price * fill.quantity * rate, 8),
                    fee_currency="USDT",
                    exchange_timestamp=fill.exchange_timestamp,
                    receive_timestamp=fill.receive_timestamp,
                    raw={**fill.raw, "fee_rate": rate},
                )
            )
        enriched = ExecutionReport(
            order_id=report.order_id,
            client_order_id=report.client_order_id,
            symbol=report.symbol,
            state=report.state,
            requested_quantity=report.requested_quantity,
            filled_quantity=report.filled_quantity,
            average_price=report.average_price,
            fills=tuple(enriched_fills),
            exchange_timestamp=report.exchange_timestamp,
            receive_timestamp=report.receive_timestamp,
            reject_reason=report.reject_reason,
            raw=report.raw,
        )
        self._persist_execution_report(request, enriched)
        return enriched

    def cancel_order(self, order_id: str, reason: str = "사용자 취소") -> bool:
        """대기 또는 부분체결 주문을 취소 상태로 바꾼다.

        Args:
            order_id: 취소할 주문 ID.
            reason: 취소 사유.

        Returns:
            취소 대상이 존재해 갱신됐으면 True.
        """
        cursor = self.conn.execute(
            """UPDATE execution_reports SET state=?, reason=?
               WHERE order_id=? AND state IN (?, ?)""",
            (
                OrderState.CANCELED.value,
                reason,
                order_id,
                OrderState.ACCEPTED.value,
                OrderState.PARTIALLY_FILLED.value,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def _persist_execution_report(
        self,
        request: OrderRequest,
        report: ExecutionReport,
    ) -> None:
        """실행 보고서와 개별 체결을 DB에 저장한다."""
        self.conn.execute(
            """INSERT OR REPLACE INTO execution_reports
               (order_id, symbol, side, state, requested_qty, filled_qty,
                average_price, maker_qty, taker_qty, fee, slippage_cost,
                reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.order_id,
                request.symbol,
                request.side,
                report.state.value,
                report.requested_quantity,
                report.filled_quantity,
                report.average_price,
                report_maker_quantity(report),
                report_taker_quantity(report),
                report_total_fee(report),
                report_total_slippage_cost(report),
                report.reject_reason or "",
                report.receive_timestamp.isoformat(),
            ),
        )
        for fill in report.fills:
            self.conn.execute(
                """INSERT OR IGNORE INTO fills
                   (fill_id, order_id, symbol, side, price, qty, liquidity,
                    fee_rate, fee, slippage_cost, adverse_selection_cost,
                    filled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.symbol,
                    fill.side,
                    fill.price,
                    fill.quantity,
                    fill.liquidity,
                    float(fill.raw.get("fee_rate", 0.0)),
                    fill.fee,
                    float(fill.raw.get("slippage_cost", 0.0)),
                    float(fill.raw.get("adverse_selection_cost", 0.0)),
                    fill.receive_timestamp.isoformat(),
                ),
            )
        self.conn.commit()

    def record_funding_rate(
        self,
        symbol: str,
        rate: float,
        funding_time: datetime,
        mark_price: float | None = None,
        source: str = "bybit",
    ) -> None:
        """실제 펀딩 정산율을 방향 정보와 함께 저장한다.

        양의 펀딩율은 롱이 지급하고 숏이 수취하며, 음의 펀딩율은 반대다.

        Args:
            symbol: 거래 심볼.
            rate: 부호 있는 펀딩율.
            funding_time: UTC 정산 시각.
            mark_price: 정산 시점 마크 가격.
            source: 데이터 출처.
        """
        if funding_time.tzinfo is None:
            raise ValueError("funding_time은 timezone-aware여야 합니다.")
        self.conn.execute(
            """INSERT OR REPLACE INTO funding_rates
               (symbol, funding_time, rate, mark_price, source)
               VALUES (?, ?, ?, ?, ?)""",
            (
                symbol,
                funding_time.astimezone(timezone.utc).isoformat(),
                float(rate),
                mark_price,
                source,
            ),
        )
        self.conn.commit()

    def _calculate_funding_pnl(
        self,
        position: Position,
        qty: float,
        exit_time: datetime,
    ) -> tuple[float, str, float | None]:
        """보유 구간 펀딩 손익과 데이터 출처·fallback 가정을 반환한다."""
        rows = self.conn.execute(
            """SELECT rate, mark_price, source FROM funding_rates
               WHERE symbol=? AND funding_time>? AND funding_time<=?
               ORDER BY funding_time""",
            (
                position.symbol,
                position.entry_time.isoformat(),
                exit_time.isoformat(),
            ),
        ).fetchall()
        direction_sign = 1.0 if position.direction == "long" else -1.0
        if rows:
            funding_pnl = sum(
                -direction_sign
                * float(rate)
                * float(mark_price or position.entry_price)
                * qty
                for rate, mark_price, _ in rows
            )
            sources = ",".join(sorted({str(row[2]) for row in rows}))
            return round(funding_pnl, 8), sources, None

        # 과거 데이터에 실제 펀딩이 없을 때만 보수적 기본값을 사용한다.
        event_count = _funding_events(position.entry_time, exit_time)
        return (
            round(
                -direction_sign
                * event_count
                * FUNDING_RATE
                * position.entry_price
                * qty,
                8,
            ),
            "assumed_bybit_utc_8h",
            FUNDING_RATE,
        )

    # ------------------------------------------------------------------
    # TradingEngine 구현
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """현재 잔고 (USDT)."""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """잔고 설정 (하위 호환용)."""
        self._balance = value

    @property
    def equity(self) -> float:
        """현금·묶인 증거금·미실현손익을 합산한 순자산을 반환한다."""
        margin = sum(position.margin for position in self._positions)
        unrealized = sum(position.unrealized_pnl for position in self._positions)
        return round(self._balance + margin + unrealized, 8)

    def record_equity(self, recorded_at: datetime | None = None) -> float:
        """현재 순자산 구성요소를 시계열 DB에 기록한다.

        Args:
            recorded_at: 기록 시각. 미지정 시 현재 UTC.

        Returns:
            기록한 순자산.
        """
        timestamp = recorded_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        margin = round(sum(position.margin for position in self._positions), 8)
        unrealized = round(
            sum(position.unrealized_pnl for position in self._positions),
            8,
        )
        equity = round(self._balance + margin + unrealized, 8)
        while True:
            try:
                self.conn.execute(
                    """INSERT INTO equity_snapshots
                       (recorded_at, cash, margin, unrealized_pnl, equity)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        timestamp.isoformat(),
                        self._balance,
                        margin,
                        unrealized,
                        equity,
                    ),
                )
                break
            except sqlite3.IntegrityError:
                timestamp += timedelta(microseconds=1)
        self.conn.commit()
        return equity

    @property
    def positions(self) -> list[Position]:
        """현재 보유 포지션 목록 (하위 호환용)."""
        return self._positions

    def get_positions(self) -> list[Position]:
        """현재 보유 포지션 목록."""
        return list(self._positions)

    def open_position(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        entry_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        leverage: float = 1.0,
        score: float | None = None,
        checks: dict | None = None,
        entry_rr: float | None = None,
        entry_session: str | None = None,
        entry_time: datetime | None = None,
        orderbook: OrderBookSnapshot | None = None,
        order_type: Literal["market", "limit"] = "market",
        limit_price: float | None = None,
        time_in_force: Literal["GTC", "IOC", "FOK"] = "IOC",
        post_only: bool = False,
        maker_available_qty: float = 0.0,
    ) -> Position | None:
        """
        포지션 진입 — 담보금 및 수수료 차감.

        레버리지가 적용되면 실제 묶이는 증거금은 명목가/레버리지이며,
        수수료는 명목가 기준으로 부과된다 (실거래와 동일).

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT:USDT")
            direction: 포지션 방향 ("long" 또는 "short")
            entry_price: 진입 가격
            qty: 수량 (코인 단위)
            stop_loss: 손절 가격
            take_profit: 목표가
            leverage: 레버리지 배수 (기본 1.0)
            score: 진입 컨플루언스 점수 (학습용, 옵셔널)
            checks: 진입 ICT 조건 dict (학습용, 옵셔널)
            entry_rr: 진입 목표 R:R (학습용, 옵셔널)
            entry_session: 진입 세션 london/newyork/None (학습용, 옵셔널)
            entry_time: 진입 시각 강제 지정 (백테스트의 시뮬레이션 시각 —
                미지정 시 현재 시각. 펀딩비 계산이 이 시각 기준)
            orderbook: 주문장 기반 체결에 사용할 스냅샷. 없으면 기존 보수적
                슬리피지 모델을 사용한다.
            order_type: 시장가 또는 지정가.
            limit_price: 지정가 주문 가격.
            time_in_force: 주문 유효 방식.
            post_only: 메이커 전용 여부.
            maker_available_qty: 메이커 주문의 관측 체결 가능 수량.

        Returns:
            생성된 Position 또는 잔고 부족 시 None
        """
        opened_at = entry_time or datetime.now(timezone.utc)
        self.record_equity(opened_at)
        lev = leverage if leverage and leverage > 0 else 1.0
        side: Literal["buy", "sell"] = "buy" if direction == "long" else "sell"
        request = OrderRequest(
            client_order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=order_type,
            price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_in_force="PostOnly" if post_only else time_in_force,
        )
        if orderbook is not None:
            visible_prices = (
                [float(price) for price, _ in orderbook.asks]
                if side == "buy"
                else [float(price) for price, _ in orderbook.bids]
            )
            conservative_price = (
                (max(visible_prices) if side == "buy" else min(visible_prices))
                if visible_prices
                else entry_price
            )
            conservative_notional = conservative_price * qty
            schedule = self._get_fee_schedule(symbol)
            conservative_cost = (
                conservative_notional / lev
                + conservative_notional * schedule.taker_rate
                + conservative_notional
                * self.execution_model.adverse_selection_bps
                / 10_000
            )
            if conservative_cost > self._balance:
                logger.warning(
                    "주문 전 잔고 검사 실패: 필요<=%.2f 보유=%.2f",
                    conservative_cost,
                    self._balance,
                )
                return None
            report = self.submit_order(
                request,
                orderbook,
                maker_available_qty=maker_available_qty,
            )
            if report.filled_quantity <= 0 or report.average_price is None:
                logger.info(
                    "포지션 미생성: 주문 미체결 id=%s state=%s",
                    report.order_id,
                    report.state.value,
                )
                return None
            actual_entry = report.average_price
            actual_qty = report.filled_quantity
            entry_fee = report_total_fee(report)
            entry_adverse_cost = sum(
                float(fill.raw.get("adverse_selection_cost", 0.0))
                for fill in report.fills
            )
            entry_slippage = round(
                abs(actual_entry - entry_price) * actual_qty
                + entry_adverse_cost,
                8,
            )
            liquidity: Literal["maker", "taker"] = (
                "maker"
                if report_maker_quantity(report) >= report_taker_quantity(report)
                else "taker"
            )
        else:
            actual_entry = self._apply_slippage(
                entry_price,
                direction,
                is_entry=True,
            )
            actual_qty = qty
            liquidity = "taker"
            entry_fee = self._fee(
                round(actual_entry * actual_qty, 8),
                symbol,
                liquidity,
            )
            entry_adverse_cost = 0.0
            entry_slippage = round(abs(actual_entry - entry_price) * actual_qty, 8)
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=f"paper-{request.client_order_id}",
                client_order_id=request.client_order_id,
                symbol=symbol,
                side=side,
                price=actual_entry,
                quantity=actual_qty,
                liquidity=liquidity,
                fee=entry_fee,
                fee_currency="USDT",
                exchange_timestamp=opened_at,
                receive_timestamp=opened_at,
                raw={
                    "fee_rate": self._get_fee_schedule(symbol).taker_rate,
                    "slippage_cost": entry_slippage,
                    "adverse_selection_cost": 0.0,
                },
            )
            report = ExecutionReport(
                order_id=f"paper-{request.client_order_id}",
                client_order_id=request.client_order_id,
                symbol=symbol,
                state=OrderState.FILLED,
                requested_quantity=qty,
                filled_quantity=actual_qty,
                average_price=actual_entry,
                fills=(fill,),
                exchange_timestamp=opened_at,
                receive_timestamp=opened_at,
            )
            self._persist_execution_report(request, report)

        notional = round(actual_entry * actual_qty, 8)
        margin = round(notional / lev, 8)        # 실제 묶이는 증거금
        total_cost = round(
            margin + entry_fee + entry_adverse_cost,
            8,
        )

        if total_cost > self._balance:
            logger.warning(
                "잔고 부족: 필요=%.2f 보유=%.2f", total_cost, self._balance
            )
            return None

        self._balance = round(self._balance - total_cost, 8)  # 증거금 + 수수료 차감
        self._save_balance()
        # R-multiple 산출용 리스크 금액 (슬리피지 적용 진입가 기준)
        risk_amount = round(abs(actual_entry - stop_loss) * actual_qty, 8)
        pos = Position(
            id=_generate_position_id(),
            symbol=symbol,
            direction=direction,
            entry_price=actual_entry,
            qty=actual_qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin=margin,
            entry_time=opened_at,
            entry_score=score,
            entry_session=entry_session,
            entry_checks=checks,
            entry_rr=entry_rr,
            risk_amount=risk_amount,
            entry_fee=entry_fee,
            entry_slippage_cost=entry_slippage,
            entry_requested_price=entry_price,
            entry_liquidity=liquidity,
            requested_qty=qty,
        )
        self._positions.append(pos)
        self._save_position(pos)
        self.record_equity(opened_at)
        logger.info(
            "[PAPER] %s %s 진입: id=%s price=%.4f qty=%.4f margin=%.2f lev=%gx notional=%.2f",
            symbol,
            direction.upper(),
            pos.id,
            actual_entry,
            qty,
            margin,
            lev,
            notional,
        )
        return pos

    def close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str = "",
        qty: float | None = None,
        exit_time: datetime | None = None,
        orderbook: OrderBookSnapshot | None = None,
        order_type: Literal["market", "limit"] = "market",
        limit_price: float | None = None,
        time_in_force: Literal["GTC", "IOC", "FOK"] = "IOC",
        post_only: bool = False,
        maker_available_qty: float = 0.0,
    ) -> float:
        """
        포지션 청산 — 담보금 반환 + PnL. 부분 청산 지원.

        Args:
            position: 청산할 포지션
            exit_price: 청산 가격
            reason: 청산 사유 (예: "SL", "TP", "manual")
            qty: 부분 청산 수량 (None이면 전량 청산)
            exit_time: 청산 시각 강제 지정 (백테스트의 시뮬레이션 시각 —
                미지정 시 현재 시각. 펀딩비/기록이 이 시각 기준)
            orderbook: 주문장 기반 체결에 사용할 스냅샷.
            order_type: 시장가 또는 지정가.
            limit_price: 지정가 주문 가격.
            time_in_force: 주문 유효 방식.
            post_only: 메이커 전용 여부.
            maker_available_qty: 메이커 주문의 관측 체결 가능 수량.

        Returns:
            실현 손익 (USDT)
        """
        requested_close_qty = qty if qty is not None else position.qty
        if requested_close_qty > position.qty:
            logger.warning(
                "청산 수량(%.6f)이 보유 수량(%.6f)을 초과합니다. 전량 청산합니다.",
                requested_close_qty,
                position.qty,
            )
            requested_close_qty = position.qty

        closed_at = exit_time or datetime.now(timezone.utc)
        side: Literal["buy", "sell"] = (
            "sell" if position.direction == "long" else "buy"
        )
        request = OrderRequest(
            client_order_id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=side,
            quantity=requested_close_qty,
            order_type=order_type,
            price=limit_price,
            reduce_only=True,
            time_in_force="PostOnly" if post_only else time_in_force,
        )
        if orderbook is not None:
            report = self.submit_order(
                request,
                orderbook,
                maker_available_qty=maker_available_qty,
            )
            if report.filled_quantity <= 0 or report.average_price is None:
                logger.warning(
                    "청산 미체결: position=%s order=%s state=%s",
                    position.id,
                    report.order_id,
                    report.state.value,
                )
                return 0.0
            close_qty = min(report.filled_quantity, position.qty)
            actual_exit = report.average_price
            exit_fee = report_total_fee(report)
            exit_slippage = round(
                abs(actual_exit - exit_price) * close_qty
                + sum(
                    float(fill.raw.get("adverse_selection_cost", 0.0))
                    for fill in report.fills
                ),
                8,
            )
            maker_qty = report_maker_quantity(report)
            taker_qty = report_taker_quantity(report)
            exit_adverse = sum(
                float(fill.raw.get("adverse_selection_cost", 0.0))
                for fill in report.fills
            )
        else:
            close_qty = requested_close_qty
            actual_exit = self._apply_slippage(
                exit_price,
                position.direction,
                is_entry=False,
            )
            exit_fee = self._fee(
                round(actual_exit * close_qty, 8),
                position.symbol,
                "taker",
            )
            exit_slippage = round(abs(actual_exit - exit_price) * close_qty, 8)
            maker_qty = 0.0
            taker_qty = close_qty
            exit_adverse = 0.0
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=f"paper-{request.client_order_id}",
                client_order_id=request.client_order_id,
                symbol=position.symbol,
                side=side,
                price=actual_exit,
                quantity=close_qty,
                liquidity="taker",
                fee=exit_fee,
                fee_currency="USDT",
                exchange_timestamp=closed_at,
                receive_timestamp=closed_at,
                raw={
                    "fee_rate": self._get_fee_schedule(
                        position.symbol
                    ).taker_rate,
                    "slippage_cost": exit_slippage,
                    "adverse_selection_cost": 0.0,
                },
            )
            report = ExecutionReport(
                order_id=f"paper-{request.client_order_id}",
                client_order_id=request.client_order_id,
                symbol=position.symbol,
                state=OrderState.FILLED,
                requested_quantity=requested_close_qty,
                filled_quantity=close_qty,
                average_price=actual_exit,
                fills=(fill,),
                exchange_timestamp=closed_at,
                receive_timestamp=closed_at,
            )
            self._persist_execution_report(request, report)

        is_partial = close_qty < position.qty
        close_ratio = close_qty / position.qty

        if position.direction == "long":
            gross_pnl = round((actual_exit - position.entry_price) * close_qty, 8)
        else:
            gross_pnl = round((position.entry_price - actual_exit) * close_qty, 8)

        funding_pnl, funding_source, funding_assumption = self._calculate_funding_pnl(
            position,
            close_qty,
            closed_at,
        )
        funding_cost = round(-funding_pnl, 8)

        entry_fee = round(position.entry_fee * close_ratio, 8)
        entry_slippage = round(position.entry_slippage_cost * close_ratio, 8)
        requested_entry = position.entry_requested_price or position.entry_price
        embedded_entry_slippage = round(
            abs(position.entry_price - requested_entry) * close_qty,
            8,
        )
        entry_adverse = max(0.0, entry_slippage - embedded_entry_slippage)
        adverse_cost = round(entry_adverse + exit_adverse, 8)
        net_pnl = round(
            gross_pnl - entry_fee - exit_fee + funding_pnl - adverse_cost,
            8,
        )
        cashflow_pnl = round(
            gross_pnl - exit_fee + funding_pnl - exit_adverse,
            8,
        )
        released_margin = round(position.margin * close_ratio, 8)
        pnl_pct = net_pnl / released_margin if released_margin > 0 else 0.0

        # 진입 수수료는 진입 때 이미 차감했으므로 현금에는 다시 차감하지 않는다.
        self._balance = round(
            self._balance + released_margin + cashflow_pnl,
            8,
        )
        self._save_balance()

        # R-multiple = 순손익 / 리스크금액 (이번 청산분 비례). risk_amount 없으면 None.
        if position.risk_amount and position.risk_amount > 0:
            risk_portion = round(position.risk_amount * close_ratio, 8)
            r_multiple = round(net_pnl / risk_portion, 6)
        else:
            risk_portion = None
            r_multiple = None

        ck = position.entry_checks or {}

        def _b(key: str) -> int | None:
            """진입 조건 bool 값을 SQLite 정수로 변환한다."""
            return int(bool(ck[key])) if key in ck else None

        now = closed_at.isoformat()
        trade_id = f"{position.id}-{now}" if is_partial else position.id
        self.conn.execute(
            """INSERT INTO trades
               (id, symbol, direction, entry_price, exit_price,
                qty, pnl, pnl_pct, margin, entry_time, exit_time, status,
                entry_score, entry_session, c_trend, c_zone, c_kill_zone,
                c_ote, c_volume, c_rr, entry_rr, risk_amount, r_multiple,
                funding_cost, gross_pnl, net_pnl, entry_fee, exit_fee,
                total_fee, funding_pnl, slippage_cost, maker_qty, taker_qty,
                requested_qty, filled_qty, funding_source,
                funding_rate_assumption)
               VALUES (?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?,
                       ?, ?)""",
            (
                trade_id,
                position.symbol,
                position.direction,
                position.entry_price,
                actual_exit,
                close_qty,
                net_pnl,
                pnl_pct,
                released_margin,
                position.entry_time.isoformat(),
                now,
                reason,
                position.entry_score,
                position.entry_session,
                _b("trend"), _b("zone"), _b("kill_zone"),
                _b("ote"), _b("volume"), _b("rr"),
                position.entry_rr,
                risk_portion,
                r_multiple,
                funding_cost,
                gross_pnl,
                net_pnl,
                entry_fee,
                exit_fee,
                round(entry_fee + exit_fee, 8),
                funding_pnl,
                round(entry_slippage + exit_slippage, 8),
                (close_qty if position.entry_liquidity == "maker" else 0.0)
                + maker_qty,
                (close_qty if position.entry_liquidity == "taker" else 0.0)
                + taker_qty,
                requested_close_qty,
                close_qty,
                funding_source,
                funding_assumption,
            ),
        )
        self.conn.commit()

        if is_partial:
            # 부분 청산: 잔여 수량/마진/리스크금액 비례 감소
            position.qty = round(position.qty - close_qty, 8)
            position.margin = round(position.margin - released_margin, 8)
            if position.risk_amount is not None and risk_portion is not None:
                position.risk_amount = round(position.risk_amount - risk_portion, 8)
            position.entry_fee = round(position.entry_fee - entry_fee, 8)
            position.entry_slippage_cost = round(
                position.entry_slippage_cost - entry_slippage,
                8,
            )
            self._save_position(position)
            logger.info(
                "[PAPER] %s 부분청산(%s): qty=%.6f PnL=%.4f (%.2f%%) 잔여=%.6f",
                position.symbol,
                reason,
                close_qty,
                net_pnl,
                pnl_pct * 100,
                position.qty,
            )
        else:
            # 전량 청산: 포지션 제거
            if position in self._positions:
                self._positions.remove(position)
            self._remove_position_from_db(position.id)
            logger.info(
                "[PAPER] %s 청산(%s): id=%s PnL=%.4f (%.2f%%)",
                position.symbol,
                reason,
                position.id,
                net_pnl,
                pnl_pct * 100,
            )

        self.record_equity(closed_at)
        self._fire_on_trade(net_pnl, reason, position)
        return net_pnl

    def check_stops(
        self,
        symbol: str,
        current_high: float,
        current_low: float,
        current_time: datetime | None = None,
        orderbook: OrderBookSnapshot | None = None,
    ) -> None:
        """
        SL/TP 자동 체크 — 해당 심볼의 모든 포지션 검사.

        Args:
            symbol: 거래 심볼
            current_high: 현재 캔들 고가
            current_low: 현재 캔들 저가
            current_time: 캔들 시각 (백테스트용 — 펀딩/기록 시각으로 전달)
            orderbook: SL/TP 시장가 청산에 사용할 주문장. 없으면 기존
                보수적 슬리피지 경로를 사용한다.
        """
        for pos in list(self._positions):
            if pos.symbol != symbol:
                continue
            if pos.direction == "long":
                if current_low <= pos.stop_loss:
                    self.close_position(
                        pos,
                        pos.stop_loss,
                        "SL",
                        exit_time=current_time,
                        orderbook=orderbook,
                        order_type="market",
                        time_in_force="IOC",
                    )
                elif current_high >= pos.take_profit:
                    self.close_position(
                        pos,
                        pos.take_profit,
                        "TP",
                        exit_time=current_time,
                        orderbook=orderbook,
                        order_type="market",
                        time_in_force="IOC",
                    )
            else:
                if current_high >= pos.stop_loss:
                    self.close_position(
                        pos,
                        pos.stop_loss,
                        "SL",
                        exit_time=current_time,
                        orderbook=orderbook,
                        order_type="market",
                        time_in_force="IOC",
                    )
                elif current_low <= pos.take_profit:
                    self.close_position(
                        pos,
                        pos.take_profit,
                        "TP",
                        exit_time=current_time,
                        orderbook=orderbook,
                        order_type="market",
                        time_in_force="IOC",
                    )

    # ------------------------------------------------------------------
    # 미실현 손익 / 트레일링 스톱
    # ------------------------------------------------------------------

    def update_unrealized_pnl(
        self,
        symbol: str,
        current_price: float,
        current_time: datetime | None = None,
    ) -> None:
        """
        미실현 손익 갱신 — 해당 심볼의 모든 포지션.

        Args:
            symbol: 거래 심볼
            current_price: 현재 가격
            current_time: 순자산 스냅샷 시각.
        """
        for pos in self._positions:
            if pos.symbol != symbol:
                continue
            if pos.direction == "long":
                pos.unrealized_pnl = round(
                    (current_price - pos.entry_price) * pos.qty, 8
                )
            else:
                pos.unrealized_pnl = round(
                    (pos.entry_price - current_price) * pos.qty, 8
                )
        logger.debug(
            "[PAPER] %s 미실현 PnL 갱신 (price=%.4f)", symbol, current_price
        )
        self.record_equity(current_time)

    def update_trailing_stop(
        self, symbol: str, current_price: float, trail_pct: float
    ) -> None:
        """
        트레일링 스톱 갱신 — 수익 방향으로 SL을 끌어올림.

        Args:
            symbol: 거래 심볼
            current_price: 현재 가격
            trail_pct: 트레일링 비율 (예: 0.01 = 1%)
        """
        for pos in self._positions:
            if pos.symbol != symbol:
                continue

            if pos.direction == "long":
                new_sl = round(current_price * (1 - trail_pct), 8)
                if new_sl > pos.stop_loss and current_price > pos.entry_price:
                    old_sl = pos.stop_loss
                    pos.stop_loss = new_sl
                    self._save_position(pos)
                    logger.info(
                        "[PAPER] %s 트레일링 SL 갱신: %.4f -> %.4f (price=%.4f)",
                        pos.symbol,
                        old_sl,
                        new_sl,
                        current_price,
                    )
            else:
                new_sl = round(current_price * (1 + trail_pct), 8)
                if new_sl < pos.stop_loss and current_price < pos.entry_price:
                    old_sl = pos.stop_loss
                    pos.stop_loss = new_sl
                    self._save_position(pos)
                    logger.info(
                        "[PAPER] %s 트레일링 SL 갱신: %.4f -> %.4f (price=%.4f)",
                        pos.symbol,
                        old_sl,
                        new_sl,
                        current_price,
                    )

    def last_sl_exit(self, symbol: str) -> datetime | None:
        """해당 심볼의 마지막 손절(SL) 청산 시각 (재진입 쿨다운용).

        Args:
            symbol: 거래 심볼

        Returns:
            마지막 SL 청산 시각 또는 없으면 None
        """
        row = self.conn.execute(
            "SELECT MAX(exit_time) FROM trades WHERE symbol = ? AND status = 'SL'",
            (symbol,),
        ).fetchone()
        if row and row[0]:
            try:
                return datetime.fromisoformat(row[0])
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------
    # 성과 지표
    # ------------------------------------------------------------------

    def get_performance(self, since: str | None = None) -> dict:
        """
        성과 지표 계산.

        Args:
            since: epoch 시작(ISO) — 지정 시 이 시각 이후 청산거래만 집계.
                전략 체제 변경 시 구체제 거래가 신체제 판정(실전 전환 등)을
                오염시키지 않도록 learning.epoch_start를 전달한다.

        Returns:
            성과 통계 딕셔너리
        """
        where = "WHERE exit_time >= ?" if since else ""
        params: tuple[str, ...] = (since,) if since else ()
        rows = self.conn.execute(
            f"""SELECT symbol, exit_time,
                       COALESCE(net_pnl, pnl),
                       COALESCE(gross_pnl, pnl),
                       COALESCE(entry_fee, 0),
                       COALESCE(exit_fee, 0),
                       COALESCE(funding_pnl, -COALESCE(funding_cost, 0)),
                       COALESCE(slippage_cost, 0),
                       COALESCE(maker_qty, 0),
                       COALESCE(taker_qty, qty),
                       COALESCE(requested_qty, qty),
                       COALESCE(filled_qty, qty)
                FROM trades {where}
                ORDER BY exit_time""",
            params,
        ).fetchall()

        if not rows:
            return {"message": "거래 내역 없음"}

        pnls = [float(row[2]) for row in rows]
        gross_pnls = [float(row[3]) for row in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity_rows = self.conn.execute(
            f"""SELECT recorded_at, equity FROM equity_snapshots
                {"WHERE recorded_at >= ?" if since else ""}
                ORDER BY recorded_at""",
            params,
        ).fetchall()
        equity_values = (
            [float(row[1]) for row in equity_rows]
            if equity_rows
            else [self.initial_balance]
            + list(np.cumsum(pnls) + self.initial_balance)
        )
        equity = np.array(equity_values, dtype=float)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
        mdd = float(drawdown.max())

        daily_returns = self._daily_equity_returns(equity_rows)
        sharpe = self._daily_sharpe(daily_returns)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss
        total_fees = sum(float(row[4]) + float(row[5]) for row in rows)
        total_funding_pnl = sum(float(row[6]) for row in rows)
        total_slippage = sum(float(row[7]) for row in rows)
        execution_row = self.conn.execute(
            f"""SELECT COALESCE(SUM(maker_qty), 0),
                       COALESCE(SUM(taker_qty), 0),
                       COALESCE(SUM(requested_qty), 0),
                       COALESCE(SUM(filled_qty), 0)
                FROM execution_reports
                {"WHERE created_at >= ?" if since else ""}""",
            params,
        ).fetchone()
        maker_qty = float(execution_row[0])
        taker_qty = float(execution_row[1])
        requested_qty = float(execution_row[2])
        filled_qty = float(execution_row[3])
        liquidity_qty = maker_qty + taker_qty
        symbol_contribution = self._contribution(rows, key_index=0)
        period_contribution = self._quarterly_contribution(rows)
        current_equity = self.equity

        return {
            "total_trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_pnl": round(sum(pnls) / len(pnls), 8),
            "net_expectancy": round(sum(pnls) / len(pnls), 8),
            "total_pnl": round(sum(pnls), 8),
            "net_pnl": round(sum(pnls), 8),
            "gross_pnl": round(sum(gross_pnls), 8),
            "fees": round(total_fees, 8),
            "funding_pnl": round(total_funding_pnl, 8),
            "funding_cost": round(-total_funding_pnl, 8),
            "slippage_cost": round(total_slippage, 8),
            "maker_ratio": maker_qty / liquidity_qty if liquidity_qty > 0 else 0.0,
            "taker_ratio": taker_qty / liquidity_qty if liquidity_qty > 0 else 0.0,
            "fill_rate": filled_qty / requested_qty if requested_qty > 0 else 0.0,
            "mdd": mdd,
            "sharpe": sharpe,
            "daily_sharpe": sharpe,
            "daily_returns": daily_returns,
            "profit_factor": profit_factor,
            "current_balance": self._balance,
            "current_equity": current_equity,
            "return_pct": (current_equity - self.initial_balance)
            / self.initial_balance,
            "symbol_contribution": symbol_contribution,
            "period_contribution": period_contribution,
        }

    @staticmethod
    def _daily_equity_returns(
        equity_rows: list[tuple[str, float]],
    ) -> list[float]:
        """UTC 일별 종가 순자산에서 일별 수익률을 계산한다."""
        daily_close: dict[str, float] = {}
        for timestamp, equity in equity_rows:
            try:
                dt = datetime.fromisoformat(timestamp)
            except (TypeError, ValueError):
                logger.warning("잘못된 순자산 스냅샷 시각 무시: %s", timestamp)
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            day = dt.astimezone(timezone.utc).date().isoformat()
            daily_close[day] = float(equity)
        values = [daily_close[day] for day in sorted(daily_close)]
        returns: list[float] = []
        for previous, current in zip(values, values[1:]):
            if previous > 0:
                returns.append((current - previous) / previous)
        return returns

    @staticmethod
    def _daily_sharpe(daily_returns: list[float]) -> float:
        """UTC 일별 수익률을 365일 기준으로 연율화한 Sharpe를 계산한다."""
        if len(daily_returns) < 2:
            return 0.0
        values = np.asarray(daily_returns, dtype=float)
        volatility = float(values.std(ddof=1))
        if volatility <= 0:
            return 0.0
        return float(values.mean() / volatility * math.sqrt(365))

    @staticmethod
    def _contribution(
        rows: list[tuple],
        key_index: int,
    ) -> dict[str, float]:
        """거래 행을 지정 키별 순손익으로 합산한다."""
        result: dict[str, float] = {}
        for row in rows:
            key = str(row[key_index])
            result[key] = round(result.get(key, 0.0) + float(row[2]), 8)
        return result

    @staticmethod
    def _quarterly_contribution(rows: list[tuple]) -> dict[str, float]:
        """청산 UTC 시각 기준 분기별 순손익을 합산한다."""
        result: dict[str, float] = {}
        for row in rows:
            try:
                dt = datetime.fromisoformat(str(row[1]))
            except ValueError:
                continue
            quarter = (dt.month - 1) // 3 + 1
            key = f"{dt.year}-Q{quarter}"
            result[key] = round(result.get(key, 0.0) + float(row[2]), 8)
        return result
