from __future__ import annotations

# 비가격 시장 특징과 public liquidation 이벤트의 시점 보존 저장소.

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Sequence

from src.data.market_snapshot import (
    DataProvenance,
    DerivativesFeatureSnapshot,
    LiquidationRecord,
    ensure_utc,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.exchange.bybit_history import HistoricalMarketRecord

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "logs" / "market_features.db"


@dataclass(frozen=True)
class FeatureEventWrite:
    """원자적 batch 저장에 사용하는 일반 특징 이벤트."""

    event_id: str
    feature_type: str
    symbol: str
    payload: dict[str, Any]
    provenance: DataProvenance
    exchange_timestamp: datetime
    receive_timestamp: datetime


@dataclass(frozen=True)
class FeedHeartbeat:
    """수집 feed의 연결 상태와 관측 gap 기록."""

    feed: str
    symbol: str
    status: str
    exchange_timestamp: datetime
    receive_timestamp: datetime
    gap_seconds: float
    provenance: DataProvenance
    detail: dict[str, Any]


@dataclass(frozen=True)
class DataQualitySummary:
    """manifest 생성 전 데이터 완전성과 gap을 요약한 값."""

    dataset: str
    symbol: str
    timestamp_axis: str
    start: datetime
    end: datetime
    event_count: int
    expected_count: int
    completeness: float
    largest_gap_seconds: float
    unresolved_gap_count: int

    @property
    def evidence_eligible(self) -> bool:
        """99% 완전성과 15분 이하 gap 기준 충족 여부를 반환한다."""
        return self.completeness >= 0.99 and self.largest_gap_seconds <= 900.0


def _json_default(value: Any) -> str:
    """JSON 기본 인코더가 처리하지 못하는 값을 문자열로 변환한다."""
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()
    return str(value)


def _canonical_payload(payload: dict[str, Any]) -> tuple[str, str]:
    """payload의 canonical JSON과 SHA-256을 반환한다."""
    encoded = json.dumps(
        payload,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
        self._lock = RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
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
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL DEFAULT ''
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
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_liquidation_events_asof
            ON liquidation_events (
                symbol, receive_timestamp, exchange_timestamp
            );

            CREATE TABLE IF NOT EXISTS feed_heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                exchange_timestamp TEXT NOT NULL,
                receive_timestamp TEXT NOT NULL,
                gap_seconds REAL NOT NULL,
                source_exchange TEXT NOT NULL DEFAULT 'bybit',
                market_type TEXT NOT NULL DEFAULT 'swap',
                endpoint TEXT NOT NULL DEFAULT 'collector_heartbeat',
                detail_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feed_heartbeats_asof
            ON feed_heartbeats (feed, symbol, receive_timestamp);

            CREATE TABLE IF NOT EXISTS collector_checkpoints (
                checkpoint_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column(
            "feature_events", "payload_sha256", "TEXT NOT NULL DEFAULT ''"
        )
        self._ensure_column(
            "liquidation_events",
            "payload_sha256",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column(
            "feed_heartbeats",
            "source_exchange",
            "TEXT NOT NULL DEFAULT 'bybit'",
        )
        self._ensure_column(
            "feed_heartbeats",
            "market_type",
            "TEXT NOT NULL DEFAULT 'swap'",
        )
        self._ensure_column(
            "feed_heartbeats",
            "endpoint",
            "TEXT NOT NULL DEFAULT 'collector_heartbeat'",
        )
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        """기존 SQLite 파일에 append-only 감사 컬럼을 안전하게 추가한다."""
        columns = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

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
        return (
            self.append_feature_batch(
                [
                    FeatureEventWrite(
                        event_id=event_id,
                        feature_type=feature_type,
                        symbol=symbol,
                        payload=payload,
                        provenance=provenance,
                        exchange_timestamp=exchange_timestamp,
                        receive_timestamp=receive_timestamp,
                    )
                ]
            )
            == 1
        )

    def append_feature_batch(
        self,
        events: Sequence[FeatureEventWrite],
    ) -> int:
        """모든 이벤트를 먼저 검증한 뒤 한 transaction으로 저장한다."""
        if not events:
            return 0
        prepared: list[tuple[Any, ...]] = []
        for event in events:
            if (
                not event.event_id.strip()
                or not event.feature_type.strip()
                or not event.symbol.strip()
            ):
                raise ValueError("feature 식별자는 비어 있을 수 없습니다")
            exchanged = ensure_utc(event.exchange_timestamp)
            received = ensure_utc(event.receive_timestamp)
            if exchanged > received:
                raise ValueError(
                    "feature exchange_timestamp가 수신 시각보다 미래입니다"
                )
            payload_json, payload_sha256 = _canonical_payload(event.payload)
            prepared.append(
                (
                    event.event_id,
                    event.feature_type,
                    event.symbol,
                    event.provenance.exchange,
                    event.provenance.market_type,
                    event.provenance.resolved_symbol,
                    event.provenance.endpoint,
                    exchanged.isoformat(),
                    received.isoformat(),
                    payload_json,
                    payload_sha256,
                )
            )
        inserted = 0
        with self._lock, self._conn:
            for values in prepared:
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO feature_events (
                        event_id, feature_type, symbol, source_exchange,
                        market_type, resolved_symbol, endpoint,
                        exchange_timestamp, receive_timestamp, payload_json,
                        payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                inserted += cursor.rowcount
        return inserted

    def save_derivatives_snapshot(
        self,
        snapshot: DerivativesFeatureSnapshot,
    ) -> bool:
        """OI·펀딩·주문장 복합 특징 스냅샷을 중복 없이 저장한다."""
        snapshot.assert_usable(snapshot.receive_timestamp)
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
            "open_interest_max_age_seconds": (snapshot.open_interest_max_age_seconds),
            "funding_max_age_seconds": snapshot.funding_max_age_seconds,
            "order_book_max_age_seconds": snapshot.order_book_max_age_seconds,
            "max_component_skew_seconds": snapshot.max_component_skew_seconds,
            "max_age_seconds": snapshot.max_age_seconds,
            "raw": snapshot.raw,
        }
        _, payload_sha256 = _canonical_payload(payload)
        identity = (
            f"{snapshot.symbol}|{snapshot.open_interest_timestamp.isoformat()}|"
            f"{snapshot.funding_timestamp.isoformat()}|"
            f"{snapshot.order_book_timestamp.isoformat()}|{payload_sha256}"
        )
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.append_feature(
            event_id=event_id,
            feature_type="derivatives_flow",
            symbol=snapshot.symbol,
            payload=payload,
            provenance=snapshot.provenance,
            exchange_timestamp=snapshot.exchange_timestamp,
            receive_timestamp=snapshot.receive_timestamp,
        )

    def save_historical_records(
        self,
        records: Sequence[HistoricalMarketRecord],
    ) -> int:
        """공식 Bybit 백필 레코드를 한 transaction으로 중복 없이 저장한다."""
        events: list[FeatureEventWrite] = []
        for record in records:
            identity = (
                f"{record.record_type}|{record.symbol}|"
                f"{record.exchange_timestamp.isoformat()}|"
                f"{record.raw_payload_sha256}"
            )
            events.append(
                FeatureEventWrite(
                    event_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    feature_type=record.record_type,
                    symbol=record.symbol,
                    payload=record.payload,
                    provenance=record.provenance,
                    exchange_timestamp=record.exchange_timestamp,
                    receive_timestamp=record.receive_timestamp,
                )
            )
        return self.append_feature_batch(events)

    def save_liquidation(self, record: LiquidationRecord) -> bool:
        """정규화된 public liquidation 레코드를 중복 없이 저장한다."""
        payload_json, payload_sha256 = _canonical_payload(record.raw)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO liquidation_events (
                    event_id, symbol, side, quantity, price,
                    source_exchange, market_type, endpoint,
                    exchange_timestamp, receive_timestamp, payload_json,
                    payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    payload_json,
                    payload_sha256,
                ),
            )
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
                raise ValueError("Bybit liquidation volume/price가 필요합니다") from exc
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
        with self._lock, self._conn:
            for record in parsed_records:
                payload_json, payload_sha256 = _canonical_payload(record.raw)
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO liquidation_events (
                        event_id, symbol, side, quantity, price,
                        source_exchange, market_type, endpoint,
                        exchange_timestamp, receive_timestamp, payload_json,
                        payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        payload_json,
                        payload_sha256,
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
        with self._lock:
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

    def load_derivatives_snapshot(
        self,
        symbol: str,
        as_of: datetime,
    ) -> DerivativesFeatureSnapshot | None:
        """결정 시점까지 수신된 가장 최근 파생 특징을 fail-closed로 조회한다."""
        cutoff = ensure_utc(as_of)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM feature_events
                WHERE feature_type = 'derivatives_flow'
                  AND symbol = ?
                  AND receive_timestamp <= ?
                  AND exchange_timestamp <= ?
                ORDER BY receive_timestamp DESC, exchange_timestamp DESC
                LIMIT 1
                """,
                (symbol, cutoff.isoformat(), cutoff.isoformat()),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            raise RuntimeError("저장된 파생 특징 raw payload가 손상되었습니다")
        snapshot = DerivativesFeatureSnapshot(
            exchange_timestamp=_parse_timestamp(row["exchange_timestamp"]),
            receive_timestamp=_parse_timestamp(row["receive_timestamp"]),
            provenance=DataProvenance(
                exchange=str(row["source_exchange"]),
                market_type=str(row["market_type"]),
                requested_symbol=str(row["symbol"]),
                resolved_symbol=str(row["resolved_symbol"]),
                endpoint=str(row["endpoint"]),
            ),
            symbol=str(row["symbol"]),
            open_interest=float(payload["open_interest"]),
            current_funding_rate=float(payload["current_funding_rate"]),
            next_funding_rate=(
                float(payload["next_funding_rate"])
                if payload.get("next_funding_rate") is not None
                else None
            ),
            next_funding_timestamp=_parse_timestamp(
                str(payload["next_funding_timestamp"])
            ),
            open_interest_timestamp=_parse_timestamp(
                str(payload["open_interest_timestamp"])
            ),
            funding_timestamp=_parse_timestamp(str(payload["funding_timestamp"])),
            order_book_timestamp=_parse_timestamp(str(payload["order_book_timestamp"])),
            bids=tuple((float(item[0]), float(item[1])) for item in payload["bids"]),
            asks=tuple((float(item[0]), float(item[1])) for item in payload["asks"]),
            open_interest_max_age_seconds=float(
                payload.get("open_interest_max_age_seconds", 360.0)
            ),
            funding_max_age_seconds=float(payload.get("funding_max_age_seconds", 60.0)),
            order_book_max_age_seconds=float(
                payload.get("order_book_max_age_seconds", 5.0)
            ),
            max_component_skew_seconds=float(
                payload.get("max_component_skew_seconds", 360.0)
            ),
            max_age_seconds=(
                float(payload["max_age_seconds"])
                if payload.get("max_age_seconds") is not None
                else None
            ),
            raw=raw,
        )
        snapshot.assert_usable(cutoff)
        return snapshot

    def record_heartbeat(self, heartbeat: FeedHeartbeat) -> int:
        """연결 heartbeat와 직전 관측 대비 gap을 append-only로 저장한다."""
        if not heartbeat.feed.strip() or not heartbeat.symbol.strip():
            raise ValueError("heartbeat feed와 symbol은 비어 있을 수 없습니다")
        exchanged = ensure_utc(heartbeat.exchange_timestamp)
        received = ensure_utc(heartbeat.receive_timestamp)
        if exchanged > received:
            raise ValueError("heartbeat exchange_timestamp가 수신보다 미래입니다")
        if heartbeat.gap_seconds < 0:
            raise ValueError("heartbeat gap_seconds는 음수일 수 없습니다")
        if heartbeat.provenance.exchange != "bybit":
            raise ValueError("heartbeat 출처는 bybit여야 합니다")
        if heartbeat.provenance.market_type != "swap":
            raise ValueError("heartbeat 상품 종류는 swap이어야 합니다")
        if (
            heartbeat.provenance.requested_symbol != heartbeat.symbol
            or heartbeat.provenance.resolved_symbol != heartbeat.symbol
        ):
            raise ValueError("heartbeat provenance 심볼이 다릅니다")
        detail_json, payload_sha256 = _canonical_payload(heartbeat.detail)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO feed_heartbeats (
                    feed, symbol, status, exchange_timestamp,
                    receive_timestamp, gap_seconds, source_exchange,
                    market_type, endpoint, detail_json,
                    payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    heartbeat.feed,
                    heartbeat.symbol,
                    heartbeat.status,
                    exchanged.isoformat(),
                    received.isoformat(),
                    float(heartbeat.gap_seconds),
                    heartbeat.provenance.exchange,
                    heartbeat.provenance.market_type,
                    heartbeat.provenance.endpoint,
                    detail_json,
                    payload_sha256,
                ),
            )
        return int(cursor.lastrowid)

    def latest_heartbeat(
        self,
        feed: str,
        symbol: str,
    ) -> FeedHeartbeat | None:
        """지정 feed와 심볼의 마지막 heartbeat를 반환한다."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM feed_heartbeats
                WHERE feed = ? AND symbol = ?
                ORDER BY receive_timestamp DESC, id DESC
                LIMIT 1
                """,
                (feed, symbol),
            ).fetchone()
        if row is None:
            return None
        return FeedHeartbeat(
            feed=str(row["feed"]),
            symbol=str(row["symbol"]),
            status=str(row["status"]),
            exchange_timestamp=_parse_timestamp(row["exchange_timestamp"]),
            receive_timestamp=_parse_timestamp(row["receive_timestamp"]),
            gap_seconds=float(row["gap_seconds"]),
            provenance=DataProvenance(
                exchange=str(row["source_exchange"]),
                market_type=str(row["market_type"]),
                requested_symbol=str(row["symbol"]),
                resolved_symbol=str(row["symbol"]),
                endpoint=str(row["endpoint"]),
            ),
            detail=json.loads(str(row["detail_json"])),
        )

    def set_checkpoint(self, key: str, value: dict[str, Any]) -> None:
        """수집 재시작에 필요한 checkpoint를 원자적으로 갱신한다."""
        if not key.strip():
            raise ValueError("checkpoint key는 비어 있을 수 없습니다")
        payload_json, payload_sha256 = _canonical_payload(value)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO collector_checkpoints (
                    checkpoint_key, value_json, payload_sha256, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(checkpoint_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    payload_sha256 = excluded.payload_sha256,
                    updated_at = excluded.updated_at
                """,
                (key, payload_json, payload_sha256, updated_at),
            )

    def get_checkpoint(self, key: str) -> dict[str, Any] | None:
        """저장된 checkpoint 값을 반환한다."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT value_json FROM collector_checkpoints
                WHERE checkpoint_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row["value_json"]))
        if not isinstance(value, dict):
            raise RuntimeError("checkpoint payload가 object가 아닙니다")
        return value

    def summarize_quality(
        self,
        dataset: str,
        symbol: str,
        start: datetime,
        end: datetime,
        expected_interval_seconds: float,
        maximum_allowed_gap_seconds: float = 900.0,
        timestamp_axis: str = "receive",
    ) -> DataQualitySummary:
        """지정 구간의 수신 완전성과 미해결 gap을 계산한다."""
        started = ensure_utc(start)
        ended = ensure_utc(end)
        if ended <= started:
            raise ValueError("품질 요약 end는 start보다 뒤여야 합니다")
        if expected_interval_seconds <= 0 or maximum_allowed_gap_seconds <= 0:
            raise ValueError("품질 요약 interval과 gap 한도는 양수여야 합니다")
        if timestamp_axis not in {"receive", "exchange"}:
            raise ValueError("timestamp_axis는 receive 또는 exchange여야 합니다")
        if dataset.startswith("heartbeat:"):
            if timestamp_axis != "receive":
                raise ValueError("heartbeat 품질은 receive 시각만 사용할 수 있습니다")
            feed = dataset.split(":", 1)[1]
            query = (
                "SELECT receive_timestamp FROM feed_heartbeats "
                "WHERE feed = ? AND symbol = ? "
                "AND receive_timestamp BETWEEN ? AND ? "
                "ORDER BY receive_timestamp"
            )
            params: tuple[Any, ...] = (
                feed,
                symbol,
                started.isoformat(),
                ended.isoformat(),
            )
        elif dataset == "liquidation":
            column = f"{timestamp_axis}_timestamp"
            query = (
                f"SELECT DISTINCT {column} AS observed_timestamp "
                "FROM liquidation_events "
                f"WHERE symbol = ? AND {column} BETWEEN ? AND ? "
                f"ORDER BY {column}"
            )
            params = (symbol, started.isoformat(), ended.isoformat())
        else:
            column = f"{timestamp_axis}_timestamp"
            query = (
                f"SELECT DISTINCT {column} AS observed_timestamp "
                "FROM feature_events "
                "WHERE feature_type = ? AND symbol = ? "
                f"AND {column} BETWEEN ? AND ? "
                f"ORDER BY {column}"
            )
            params = (
                dataset,
                symbol,
                started.isoformat(),
                ended.isoformat(),
            )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        observed = [
            _parse_timestamp(
                row["receive_timestamp"]
                if dataset.startswith("heartbeat:")
                else row["observed_timestamp"]
            )
            for row in rows
        ]
        expected_count = max(
            int((ended - started).total_seconds() // expected_interval_seconds),
            1,
        )
        completeness = min(len(observed) / expected_count, 1.0)
        boundaries = [started, *observed, ended]
        gaps = [
            (right - left).total_seconds()
            for left, right in zip(boundaries, boundaries[1:])
        ]
        largest_gap = max(gaps, default=(ended - started).total_seconds())
        unresolved = sum(gap > maximum_allowed_gap_seconds for gap in gaps)
        return DataQualitySummary(
            dataset=dataset,
            symbol=symbol,
            timestamp_axis=timestamp_axis,
            start=started,
            end=ended,
            event_count=len(observed),
            expected_count=expected_count,
            completeness=completeness,
            largest_gap_seconds=largest_gap,
            unresolved_gap_count=unresolved,
        )

    def payload_hashes(
        self,
        dataset: str,
        symbol: str,
        start: datetime,
        end: datetime,
        timestamp_axis: str = "receive",
    ) -> tuple[str, ...]:
        """manifest에 사용할 원시 payload hash들을 시점 순서로 반환한다."""
        started = ensure_utc(start).isoformat()
        ended = ensure_utc(end).isoformat()
        if timestamp_axis not in {"receive", "exchange"}:
            raise ValueError("timestamp_axis는 receive 또는 exchange여야 합니다")
        if dataset.startswith("heartbeat:"):
            if timestamp_axis != "receive":
                raise ValueError("heartbeat hash는 receive 시각만 사용할 수 있습니다")
            feed = dataset.split(":", 1)[1]
            query = (
                "SELECT payload_sha256 FROM feed_heartbeats "
                "WHERE feed = ? AND symbol = ? "
                "AND receive_timestamp BETWEEN ? AND ? "
                "ORDER BY receive_timestamp, id"
            )
            params: tuple[Any, ...] = (feed, symbol, started, ended)
        elif dataset == "liquidation":
            column = f"{timestamp_axis}_timestamp"
            query = (
                "SELECT payload_sha256 FROM liquidation_events "
                f"WHERE symbol = ? AND {column} BETWEEN ? AND ? "
                f"ORDER BY {column}, event_id"
            )
            params = (symbol, started, ended)
        else:
            column = f"{timestamp_axis}_timestamp"
            query = (
                "SELECT payload_sha256 FROM feature_events "
                "WHERE feature_type = ? AND symbol = ? "
                f"AND {column} BETWEEN ? AND ? "
                f"ORDER BY {column}, event_id"
            )
            params = (dataset, symbol, started, ended)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        hashes = tuple(str(row["payload_sha256"]) for row in rows)
        if any(len(value) != 64 for value in hashes):
            raise RuntimeError("원시 payload SHA-256이 누락되거나 손상되었습니다")
        return hashes

    def close(self) -> None:
        """SQLite 연결을 닫는다."""
        with self._lock:
            self._conn.close()
        logger.info("MarketFeatureStore 연결 종료")
