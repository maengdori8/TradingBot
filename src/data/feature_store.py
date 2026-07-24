from __future__ import annotations

# 비가격 시장 특징과 public liquidation 이벤트의 시점 보존 저장소.

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.market_snapshot import (
    DataProvenance,
    DerivativesFeatureSnapshot,
    LiquidationRecord,
    ensure_utc,
)

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "logs" / "market_features.db"


def _json_default(value: Any) -> str:
    """JSON 기본 인코더가 처리하지 못하는 값을 문자열로 변환한다."""
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()
    return str(value)


def _parse_timestamp(value: str) -> datetime:
    """저장된 ISO timestamp를 UTC datetime으로 변환한다."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("저장된 timestamp에 timezone 정보가 없습니다")
    return parsed.astimezone(timezone.utc)


def _normalize_bybit_symbol(symbol: str) -> str:
    """Bybit 원시 심볼을 ccxt USDT 무기한 선물 심볼로 변환한다."""
    normalized = symbol.strip().upper()
    if normalized.endswith("USDT") and "/" not in normalized:
        base = normalized[:-4]
        if base:
            return f"{base}/USDT:USDT"
    if normalized.endswith("/USDT:USDT"):
        return normalized
    raise ValueError(f"지원하지 않는 Bybit liquidation 심볼: {symbol}")


class MarketFeatureStore:
    """일반 시장 특징과 public liquidation 이벤트 SQLite 저장소."""

    def __init__(self, db_path: Path | None = None) -> None:
        """저장소 연결을 열고 시점 보존 테이블을 생성한다."""
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("MarketFeatureStore 초기화: %s", self._db_path)

    def _create_tables(self) -> None:
        """일반 특징과 liquidation 이벤트 테이블을 생성한다."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feature_events (
                event_id TEXT PRIMARY KEY,
                feature_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                source_exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                resolved_symbol TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                exchange_timestamp TEXT NOT NULL,
                receive_timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feature_events_asof
            ON feature_events (
                feature_type, symbol, receive_timestamp, exchange_timestamp
            );

            CREATE TABLE IF NOT EXISTS liquidation_events (
                event_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                source_exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                exchange_timestamp TEXT NOT NULL,
                receive_timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_liquidation_events_asof
            ON liquidation_events (
                symbol, receive_timestamp, exchange_timestamp
            );
            """
        )
        self._conn.commit()

    def append_feature(
        self,
        event_id: str,
        feature_type: str,
        symbol: str,
        payload: dict[str, Any],
        provenance: DataProvenance,
        exchange_timestamp: datetime,
        receive_timestamp: datetime,
    ) -> bool:
        """일반 시장 특징 이벤트를 중복 없이 시점 보존한다."""
        if not event_id.strip() or not feature_type.strip() or not symbol.strip():
            raise ValueError("feature 식별자는 비어 있을 수 없습니다")
        exchanged = ensure_utc(exchange_timestamp)
        received = ensure_utc(receive_timestamp)
        if exchanged > received:
            raise ValueError("feature exchange_timestamp가 수신 시각보다 미래입니다")
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO feature_events (
                event_id, feature_type, symbol, source_exchange, market_type,
                resolved_symbol, endpoint, exchange_timestamp,
                receive_timestamp, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                feature_type,
                symbol,
                provenance.exchange,
                provenance.market_type,
                provenance.resolved_symbol,
                provenance.endpoint,
                exchanged.isoformat(),
                received.isoformat(),
                json.dumps(payload, default=_json_default, sort_keys=True),
            ),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def save_derivatives_snapshot(
        self,
        snapshot: DerivativesFeatureSnapshot,
    ) -> bool:
        """OI·펀딩·주문장 복합 특징 스냅샷을 중복 없이 저장한다."""
        snapshot.assert_usable(snapshot.receive_timestamp)
        identity = (
            f"{snapshot.symbol}|{snapshot.open_interest_timestamp.isoformat()}|"
            f"{snapshot.funding_timestamp.isoformat()}|"
            f"{snapshot.order_book_timestamp.isoformat()}"
        )
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        payload = {
            "open_interest": snapshot.open_interest,
            "current_funding_rate": snapshot.current_funding_rate,
            "next_funding_rate": snapshot.next_funding_rate,
            "next_funding_timestamp": snapshot.next_funding_timestamp,
            "open_interest_timestamp": snapshot.open_interest_timestamp,
            "funding_timestamp": snapshot.funding_timestamp,
            "order_book_timestamp": snapshot.order_book_timestamp,
            "bids": snapshot.bids,
            "asks": snapshot.asks,
            "raw": snapshot.raw,
        }
        return self.append_feature(
            event_id=event_id,
            feature_type="derivatives_flow",
            symbol=snapshot.symbol,
            payload=payload,
            provenance=snapshot.provenance,
            exchange_timestamp=snapshot.exchange_timestamp,
            receive_timestamp=snapshot.receive_timestamp,
        )

    def save_liquidation(self, record: LiquidationRecord) -> bool:
        """정규화된 public liquidation 레코드를 중복 없이 저장한다."""
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO liquidation_events (
                event_id, symbol, side, quantity, price,
                source_exchange, market_type, endpoint,
                exchange_timestamp, receive_timestamp, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.event_id,
                record.symbol,
                record.side,
                record.quantity,
                record.price,
                record.provenance.exchange,
                record.provenance.market_type,
                record.provenance.endpoint,
                ensure_utc(record.exchange_timestamp).isoformat(),
                ensure_utc(record.receive_timestamp).isoformat(),
                json.dumps(record.raw, default=_json_default, sort_keys=True),
            ),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def ingest_bybit_liquidations(
        self,
        payload: dict[str, Any],
        receive_timestamp: datetime | None = None,
    ) -> int:
        """Bybit public all-liquidation 메시지를 검증하고 중복 없이 저장한다."""
        received = ensure_utc(receive_timestamp or datetime.now(timezone.utc))
        raw_records = payload.get("data")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("Bybit liquidation payload에 data 배열이 필요합니다")
        parsed_records: list[LiquidationRecord] = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ValueError("Bybit liquidation data 항목은 object여야 합니다")
            raw_timestamp = raw.get("T")
            if isinstance(raw_timestamp, bool) or not isinstance(
                raw_timestamp,
                (int, float),
            ):
                raise ValueError("Bybit liquidation T timestamp가 필요합니다")
            exchanged = datetime.fromtimestamp(
                float(raw_timestamp) / 1000.0,
                timezone.utc,
            )
            symbol = _normalize_bybit_symbol(str(raw.get("s") or ""))
            raw_side = str(raw.get("S") or "").lower()
            if raw_side not in {"buy", "sell"}:
                raise ValueError(f"지원하지 않는 liquidation side: {raw_side}")
            try:
                quantity = float(raw["v"])
                price = float(raw["p"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Bybit liquidation volume/price가 필요합니다"
                ) from exc
            identity = (
                f"bybit|{symbol}|{int(float(raw_timestamp))}|"
                f"{raw_side}|{quantity:.8f}|{price:.8f}"
            )
            event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            parsed_records.append(
                LiquidationRecord(
                    exchange_timestamp=exchanged,
                    receive_timestamp=received,
                    provenance=DataProvenance(
                        exchange="bybit",
                        market_type="swap",
                        requested_symbol=symbol,
                        resolved_symbol=symbol,
                        endpoint="public_ws_all_liquidation",
                    ),
                    event_id=event_id,
                    symbol=symbol,
                    side=raw_side,  # type: ignore[arg-type]
                    quantity=round(quantity, 8),
                    price=round(price, 8),
                    raw=dict(raw),
                )
            )
        inserted = 0
        with self._conn:
            for record in parsed_records:
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO liquidation_events (
                        event_id, symbol, side, quantity, price,
                        source_exchange, market_type, endpoint,
                        exchange_timestamp, receive_timestamp, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.event_id,
                        record.symbol,
                        record.side,
                        record.quantity,
                        record.price,
                        record.provenance.exchange,
                        record.provenance.market_type,
                        record.provenance.endpoint,
                        record.exchange_timestamp.isoformat(),
                        record.receive_timestamp.isoformat(),
                        json.dumps(
                            record.raw,
                            default=_json_default,
                            sort_keys=True,
                        ),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def load_liquidations(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> list[LiquidationRecord]:
        """저장된 liquidation 이벤트를 거래소 발생 순서로 조회한다."""
        filters = ["1 = 1"]
        params: list[Any] = []
        if symbol is not None:
            filters.append("symbol = ?")
            params.append(symbol)
        if since is not None:
            filters.append("receive_timestamp >= ?")
            params.append(ensure_utc(since).isoformat())
        rows = self._conn.execute(
            f"""
            SELECT * FROM liquidation_events
            WHERE {" AND ".join(filters)}
            ORDER BY exchange_timestamp, event_id
            """,
            params,
        ).fetchall()
        return [
            LiquidationRecord(
                exchange_timestamp=_parse_timestamp(row["exchange_timestamp"]),
                receive_timestamp=_parse_timestamp(row["receive_timestamp"]),
                provenance=DataProvenance(
                    exchange=str(row["source_exchange"]),
                    market_type=str(row["market_type"]),
                    requested_symbol=str(row["symbol"]),
                    resolved_symbol=str(row["symbol"]),
                    endpoint=str(row["endpoint"]),
                ),
                event_id=str(row["event_id"]),
                symbol=str(row["symbol"]),
                side=str(row["side"]),  # type: ignore[arg-type]
                quantity=float(row["quantity"]),
                price=float(row["price"]),
                raw=json.loads(str(row["payload_json"])),
            )
            for row in rows
        ]

    def close(self) -> None:
        """SQLite 연결을 닫는다."""
        self._conn.close()
        logger.info("MarketFeatureStore 연결 종료")
