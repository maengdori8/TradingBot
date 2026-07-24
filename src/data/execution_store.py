"""주문·체결 이벤트와 계정 스냅샷을 영구 보존하는 SQLite 저장소."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.exchange.contracts import ExecutionReport, FeeRateSnapshot, Fill, OrderState

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "logs" / "execution_events.db"


def _utc_iso(value: datetime | None) -> str | None:
    """datetime을 UTC ISO 문자열로 변환한다."""
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("timestamp에는 timezone 정보가 필요합니다")
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    """ISO 문자열을 timezone-aware datetime으로 변환한다."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_default(value: Any) -> str:
    """JSON 기본 인코더가 처리하지 못한 값을 문자열로 변환한다."""
    if isinstance(value, datetime):
        encoded = _utc_iso(value)
        return encoded or ""
    if isinstance(value, (OrderState,)):
        return value.value
    return str(value)


class ExecutionEventStore:
    """재시작과 거래소 재대사에 사용하는 실행 이벤트 저장소."""

    def __init__(self, db_path: Path | None = None) -> None:
        """저장소 연결을 열고 필요한 테이블을 생성한다."""
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("ExecutionEventStore 초기화: %s", self._db_path)

    def _create_tables(self) -> None:
        """실행 보고서, 체결, 원시 이벤트, 수수료 테이블을 생성한다."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS execution_reports (
                client_order_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                state TEXT NOT NULL,
                requested_quantity REAL NOT NULL,
                filled_quantity REAL NOT NULL,
                average_price REAL,
                exchange_timestamp TEXT,
                receive_timestamp TEXT NOT NULL,
                reject_reason TEXT,
                raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_execution_reports_order
            ON execution_reports (order_id);

            CREATE TABLE IF NOT EXISTS order_claims (
                client_order_id TEXT PRIMARY KEY,
                order_link_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                client_order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL,
                fee_currency TEXT,
                liquidity TEXT NOT NULL,
                exchange_timestamp TEXT,
                receive_timestamp TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fills_client_order
            ON fills (client_order_id, exchange_timestamp);

            CREATE TABLE IF NOT EXISTS execution_events (
                event_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                channel TEXT NOT NULL,
                order_id TEXT,
                client_order_id TEXT,
                exchange_timestamp TEXT,
                receive_timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_execution_events_reconcile
            ON execution_events (channel, exchange_timestamp, receive_timestamp);

            CREATE TABLE IF NOT EXISTS fee_rate_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                maker_rate REAL NOT NULL,
                taker_rate REAL NOT NULL,
                exchange_timestamp TEXT,
                receive_timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fee_rate_latest
            ON fee_rate_snapshots (symbol, receive_timestamp);
            """
        )
        self._conn.commit()

    def claim_order(
        self,
        client_order_id: str,
        order_link_id: str,
        symbol: str,
    ) -> bool:
        """주문 ID를 원자적으로 선점해 동시 중복 제출을 차단한다."""
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO order_claims (
                client_order_id, order_link_id, symbol, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                client_order_id,
                order_link_id,
                symbol,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def resolve_client_order_id(self, order_link_id: str) -> str | None:
        """Bybit orderLinkId에 대응하는 원본 클라이언트 주문 ID를 조회한다."""
        row = self._conn.execute(
            """
            SELECT client_order_id FROM order_claims
            WHERE order_link_id = ?
            """,
            (order_link_id,),
        ).fetchone()
        return str(row["client_order_id"]) if row is not None else None

    def save_report(self, report: ExecutionReport) -> None:
        """최신 주문 보고서와 포함된 체결을 원자적으로 저장한다."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO execution_reports (
                    client_order_id, order_id, symbol, state,
                    requested_quantity, filled_quantity, average_price,
                    exchange_timestamp, receive_timestamp, reject_reason, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    order_id=excluded.order_id,
                    symbol=excluded.symbol,
                    state=excluded.state,
                    requested_quantity=excluded.requested_quantity,
                    filled_quantity=excluded.filled_quantity,
                    average_price=excluded.average_price,
                    exchange_timestamp=excluded.exchange_timestamp,
                    receive_timestamp=excluded.receive_timestamp,
                    reject_reason=excluded.reject_reason,
                    raw_json=excluded.raw_json
                """,
                (
                    report.client_order_id,
                    report.order_id,
                    report.symbol,
                    report.state.value,
                    report.requested_quantity,
                    report.filled_quantity,
                    report.average_price,
                    _utc_iso(report.exchange_timestamp),
                    _utc_iso(report.receive_timestamp),
                    report.reject_reason,
                    json.dumps(report.raw, default=_json_default, sort_keys=True),
                ),
            )
            for fill in report.fills:
                self._save_fill(fill)

    def _save_fill(self, fill: Fill) -> None:
        """현재 트랜잭션에 개별 체결을 중복 없이 저장한다."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fills (
                fill_id, order_id, client_order_id, symbol, side,
                quantity, price, fee, fee_currency, liquidity,
                exchange_timestamp, receive_timestamp, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.client_order_id,
                fill.symbol,
                fill.side,
                fill.quantity,
                fill.price,
                fill.fee,
                fill.fee_currency,
                fill.liquidity,
                _utc_iso(fill.exchange_timestamp),
                _utc_iso(fill.receive_timestamp),
                json.dumps(fill.raw, default=_json_default, sort_keys=True),
            ),
        )

    def get_report(self, client_order_id: str) -> ExecutionReport | None:
        """클라이언트 주문 ID로 최신 실행 보고서를 조회한다."""
        row = self._conn.execute(
            """
            SELECT * FROM execution_reports
            WHERE client_order_id = ?
            """,
            (client_order_id,),
        ).fetchone()
        if row is None:
            return None
        fill_rows = self._conn.execute(
            """
            SELECT * FROM fills
            WHERE client_order_id = ?
            ORDER BY exchange_timestamp, receive_timestamp
            """,
            (client_order_id,),
        ).fetchall()
        fills = tuple(self._row_to_fill(item) for item in fill_rows)
        return ExecutionReport(
            order_id=str(row["order_id"]),
            client_order_id=str(row["client_order_id"]),
            symbol=str(row["symbol"]),
            state=OrderState(str(row["state"])),
            requested_quantity=float(row["requested_quantity"]),
            filled_quantity=float(row["filled_quantity"]),
            average_price=(
                float(row["average_price"])
                if row["average_price"] is not None
                else None
            ),
            fills=fills,
            exchange_timestamp=_parse_time(row["exchange_timestamp"]),
            receive_timestamp=_parse_time(row["receive_timestamp"])
            or datetime.now(timezone.utc),
            reject_reason=row["reject_reason"],
            raw=json.loads(str(row["raw_json"])),
        )

    def _row_to_fill(self, row: sqlite3.Row) -> Fill:
        """SQLite 행을 체결 객체로 변환한다."""
        side = str(row["side"])
        if side not in {"buy", "sell"}:
            raise ValueError(f"지원하지 않는 체결 방향: {side}")
        liquidity = str(row["liquidity"])
        if liquidity not in {"maker", "taker", "unknown"}:
            liquidity = "unknown"
        return Fill(
            fill_id=str(row["fill_id"]),
            order_id=str(row["order_id"]),
            client_order_id=str(row["client_order_id"]),
            symbol=str(row["symbol"]),
            side=side,  # type: ignore[arg-type]
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            fee=float(row["fee"]),
            fee_currency=row["fee_currency"],
            liquidity=liquidity,  # type: ignore[arg-type]
            exchange_timestamp=_parse_time(row["exchange_timestamp"]),
            receive_timestamp=_parse_time(row["receive_timestamp"])
            or datetime.now(timezone.utc),
            raw=json.loads(str(row["raw_json"])),
        )

    def append_event(
        self,
        event_id: str,
        source: str,
        channel: str,
        payload: dict[str, Any],
        exchange_timestamp: datetime | None,
        receive_timestamp: datetime | None = None,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> bool:
        """REST/WebSocket 원시 이벤트를 중복 없이 저장한다."""
        received = receive_timestamp or datetime.now(timezone.utc)
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO execution_events (
                event_id, source, channel, order_id, client_order_id,
                exchange_timestamp, receive_timestamp, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                source,
                channel,
                order_id,
                client_order_id,
                _utc_iso(exchange_timestamp),
                _utc_iso(received),
                json.dumps(payload, default=_json_default, sort_keys=True),
            ),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def load_events(
        self,
        channel: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """재대사용 원시 이벤트를 수신 순서대로 조회한다."""
        query = "SELECT * FROM execution_events WHERE 1 = 1"
        params: list[Any] = []
        if channel is not None:
            query += " AND channel = ?"
            params.append(channel)
        if since is not None:
            query += " AND receive_timestamp >= ?"
            params.append(_utc_iso(since))
        query += " ORDER BY receive_timestamp, event_id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "source": row["source"],
                "channel": row["channel"],
                "order_id": row["order_id"],
                "client_order_id": row["client_order_id"],
                "exchange_timestamp": _parse_time(row["exchange_timestamp"]),
                "receive_timestamp": _parse_time(row["receive_timestamp"]),
                "payload": json.loads(str(row["payload_json"])),
            }
            for row in rows
        ]

    def save_fee_rate(self, snapshot: FeeRateSnapshot) -> None:
        """계정별 수수료율 스냅샷을 저장한다."""
        values = asdict(snapshot)
        self._conn.execute(
            """
            INSERT INTO fee_rate_snapshots (
                symbol, maker_rate, taker_rate, exchange_timestamp,
                receive_timestamp, source, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.symbol,
                snapshot.maker_rate,
                snapshot.taker_rate,
                _utc_iso(snapshot.exchange_timestamp),
                _utc_iso(snapshot.receive_timestamp),
                snapshot.source,
                json.dumps(values["raw"], default=_json_default, sort_keys=True),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        """SQLite 연결을 닫는다."""
        self._conn.close()
        logger.info("ExecutionEventStore 연결 종료")
