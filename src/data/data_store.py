"""
OHLCV 데이터 SQLite 저장소.
시세 데이터를 로컬에 영속적으로 보관하여 API 호출을 절약한다.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.market_snapshot import DataProvenance, MarketSnapshot

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "logs" / "market_data.db"


class DataStore:
    """OHLCV 데이터 SQLite 저장소.

    심볼 + 타임프레임 조합별로 캔들 데이터를 저장/조회한다.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """DataStore를 초기화하고 테이블을 생성한다.

        Args:
            db_path: 데이터베이스 파일 경로 (기본: logs/market_data.db)
        """
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._create_table()
        logger.info("DataStore 초기화: %s", self._db_path)

    def _create_table(self) -> None:
        """OHLCV 테이블을 생성한다 (존재하지 않으면)."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol    TEXT    NOT NULL,
                timeframe TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                open      REAL    NOT NULL,
                high      REAL    NOT NULL,
                low       REAL    NOT NULL,
                close     REAL    NOT NULL,
                volume    REAL    NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
            ON ohlcv (symbol, timeframe, timestamp)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_observations (
                symbol             TEXT NOT NULL,
                timeframe          TEXT NOT NULL,
                timestamp          TEXT NOT NULL,
                open               REAL NOT NULL,
                high               REAL NOT NULL,
                low                REAL NOT NULL,
                close              REAL NOT NULL,
                volume             REAL NOT NULL,
                source_exchange    TEXT NOT NULL,
                market_type        TEXT NOT NULL,
                resolved_symbol    TEXT NOT NULL,
                exchange_timestamp TEXT NOT NULL,
                receive_timestamp  TEXT NOT NULL,
                PRIMARY KEY (
                    symbol, timeframe, timestamp, receive_timestamp
                )
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_observations_asof
            ON ohlcv_observations (
                symbol, timeframe, receive_timestamp, timestamp
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol             TEXT NOT NULL,
                source_exchange    TEXT NOT NULL,
                market_type        TEXT NOT NULL,
                resolved_symbol    TEXT NOT NULL,
                endpoint           TEXT NOT NULL,
                last               REAL NOT NULL,
                bid                REAL,
                ask                REAL,
                exchange_timestamp TEXT NOT NULL,
                receive_timestamp  TEXT NOT NULL,
                max_age_seconds    REAL NOT NULL,
                payload_json       TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_snapshots_asof
            ON market_snapshots (symbol, receive_timestamp)
        """)
        self._conn.commit()

    def save_candles(
        self, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> None:
        """캔들 데이터를 저장한다 (UPSERT).

        Args:
            symbol: 거래 심볼
            timeframe: 캔들 주기
            df: OHLCV DataFrame (index=timestamp UTC,
                columns=[open, high, low, close, volume])
        """
        if df.empty:
            logger.warning("빈 DataFrame — 저장 건너뜀: %s %s", symbol, timeframe)
            return

        received = datetime.now(timezone.utc)
        provenance = df.attrs.get("provenance", {})
        source_exchange = str(provenance.get("exchange") or "unknown")
        market_type = str(provenance.get("market_type") or "unknown")
        resolved_symbol = str(provenance.get("resolved_symbol") or symbol)
        raw_received = provenance.get("receive_timestamp")
        if raw_received:
            parsed_received = pd.to_datetime(raw_received, utc=True)
            received = parsed_received.to_pydatetime()
        rows: list[tuple[Any, ...]] = []
        observation_rows: list[tuple[Any, ...]] = []
        for ts, row in df.iterrows():
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append((
                symbol, timeframe, ts_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))
            observation_rows.append(
                (
                    symbol,
                    timeframe,
                    ts_str,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    source_exchange,
                    market_type,
                    resolved_symbol,
                    ts_str,
                    received.isoformat(),
                )
            )

        self._conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv
                (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO ohlcv_observations (
                symbol, timeframe, timestamp, open, high, low, close, volume,
                source_exchange, market_type, resolved_symbol,
                exchange_timestamp, receive_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            observation_rows,
        )
        self._conn.commit()
        logger.info(
            "캔들 저장 완료: %s %s (%d건)", symbol, timeframe, len(rows)
        )

    def load_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame | None:
        """저장된 캔들 데이터를 조회한다.

        Args:
            symbol: 거래 심볼
            timeframe: 캔들 주기
            limit: 최대 조회 건수

        Returns:
            OHLCV DataFrame (최신 limit개) 또는 데이터 없으면 None
        """
        cursor = self._conn.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, timeframe, limit),
        )
        rows = cursor.fetchall()

        if not rows:
            logger.debug("저장된 데이터 없음: %s %s", symbol, timeframe)
            return None

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        logger.debug(
            "캔들 로드 완료: %s %s (%d건)", symbol, timeframe, len(df)
        )
        return df

    def load_candles_as_of(
        self,
        symbol: str,
        timeframe: str,
        as_of: datetime,
        limit: int = 200,
        required_exchange: str | None = "bybit",
        required_market_type: str | None = "swap",
    ) -> pd.DataFrame | None:
        """해당 시점까지 수신된 최신 버전의 캔들만 조회한다.

        Args:
            symbol: 거래 심볼.
            timeframe: 캔들 주기.
            as_of: 데이터 수신 마감 시각.
            limit: 최대 조회 캔들 수.
            required_exchange: 허용할 원천 거래소. None이면 검사하지 않는다.
            required_market_type: 허용할 상품 종류. None이면 검사하지 않는다.

        Returns:
            point-in-time OHLCV DataFrame 또는 데이터가 없으면 None.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of에는 timezone 정보가 필요합니다")
        cutoff = as_of.astimezone(timezone.utc).isoformat()
        filters = [
            "symbol = ?",
            "timeframe = ?",
            "receive_timestamp <= ?",
            "exchange_timestamp <= ?",
        ]
        params: list[Any] = [symbol, timeframe, cutoff, cutoff]
        if required_exchange is not None:
            filters.append("source_exchange = ?")
            params.append(required_exchange)
        if required_market_type is not None:
            filters.append("market_type = ?")
            params.append(required_market_type)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol, timeframe, timestamp
                           ORDER BY receive_timestamp DESC
                       ) AS version_rank
                FROM ohlcv_observations
                WHERE {" AND ".join(filters)}
            )
            SELECT timestamp, open, high, low, close, volume,
                   source_exchange, market_type, resolved_symbol,
                   receive_timestamp
            FROM ranked
            WHERE version_rank = 1
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        if not rows:
            return None
        frame = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source_exchange",
                "market_type",
                "resolved_symbol",
                "receive_timestamp",
            ],
        ).sort_values("timestamp")
        metadata = frame.iloc[-1]
        result = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        result = result.set_index("timestamp")
        result.attrs["provenance"] = {
            "exchange": metadata["source_exchange"],
            "market_type": metadata["market_type"],
            "requested_symbol": symbol,
            "resolved_symbol": metadata["resolved_symbol"],
            "receive_timestamp": metadata["receive_timestamp"],
            "as_of": cutoff,
        }
        return result

    def save_market_snapshot(self, snapshot: MarketSnapshot) -> None:
        """출처와 두 타임스탬프를 포함한 시장 스냅샷을 저장한다."""
        self._conn.execute(
            """
            INSERT INTO market_snapshots (
                symbol, source_exchange, market_type, resolved_symbol,
                endpoint, last, bid, ask, exchange_timestamp,
                receive_timestamp, max_age_seconds, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.symbol,
                snapshot.provenance.exchange,
                snapshot.provenance.market_type,
                snapshot.provenance.resolved_symbol,
                snapshot.provenance.endpoint,
                snapshot.last,
                snapshot.bid,
                snapshot.ask,
                snapshot.exchange_timestamp.isoformat(),
                snapshot.receive_timestamp.isoformat(),
                snapshot.max_age_seconds,
                json.dumps(snapshot.raw, default=str, sort_keys=True),
            ),
        )
        self._conn.commit()

    def load_latest_market_snapshot(
        self,
        symbol: str,
        as_of: datetime | None = None,
    ) -> MarketSnapshot | None:
        """지정 시점까지 수신된 가장 최근 시장 스냅샷을 조회한다.

        Args:
            symbol: 거래 심볼.
            as_of: 수신 시각 상한. None이면 현재 시각을 사용한다.

        Returns:
            최신 시장 스냅샷 또는 저장된 데이터가 없으면 None.
        """
        cutoff_time = as_of or datetime.now(timezone.utc)
        if cutoff_time.tzinfo is None:
            raise ValueError("as_of에는 timezone 정보가 필요합니다")
        row = self._conn.execute(
            """
            SELECT symbol, source_exchange, market_type, resolved_symbol,
                   endpoint, last, bid, ask, exchange_timestamp,
                   receive_timestamp, max_age_seconds, payload_json
            FROM market_snapshots
            WHERE symbol = ? AND receive_timestamp <= ?
            ORDER BY receive_timestamp DESC
            LIMIT 1
            """,
            (symbol, cutoff_time.astimezone(timezone.utc).isoformat()),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[11]))
        order_book = payload.get("order_book", {})
        return MarketSnapshot(
            exchange_timestamp=pd.to_datetime(row[8], utc=True).to_pydatetime(),
            receive_timestamp=pd.to_datetime(row[9], utc=True).to_pydatetime(),
            provenance=DataProvenance(
                exchange=str(row[1]),
                market_type=str(row[2]),
                requested_symbol=str(row[0]),
                resolved_symbol=str(row[3]),
                endpoint=str(row[4]),
            ),
            symbol=str(row[0]),
            last=float(row[5]),
            bid=float(row[6]) if row[6] is not None else None,
            ask=float(row[7]) if row[7] is not None else None,
            bids=tuple(
                (float(level[0]), float(level[1]))
                for level in order_book.get("bids", [])
            ),
            asks=tuple(
                (float(level[0]), float(level[1]))
                for level in order_book.get("asks", [])
            ),
            max_age_seconds=float(row[10]),
            raw=payload,
        )

    def get_latest_timestamp(
        self, symbol: str, timeframe: str
    ) -> datetime | None:
        """저장된 가장 최근 캔들의 타임스탬프를 반환한다.

        Args:
            symbol: 거래 심볼
            timeframe: 캔들 주기

        Returns:
            최신 타임스탬프 (UTC) 또는 데이터 없으면 None
        """
        cursor = self._conn.execute(
            """
            SELECT MAX(timestamp) FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
            """,
            (symbol, timeframe),
        )
        row = cursor.fetchone()

        if row is None or row[0] is None:
            return None

        ts = pd.to_datetime(row[0], utc=True)
        return ts.to_pydatetime()

    def cleanup_old_data(self, days: int = 30) -> int:
        """지정 일수보다 오래된 캔들 데이터를 삭제한다.

        Args:
            days: 보관 기간 (기본: 30일)

        Returns:
            삭제된 행 수
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        cursor = self._conn.execute(
            "DELETE FROM ohlcv WHERE timestamp < ?",
            (cutoff_str,),
        )
        deleted = cursor.rowcount
        self._conn.commit()
        logger.info(
            "오래된 데이터 정리 완료: %d건 삭제 (기준: %d일, cutoff=%s)",
            deleted, days, cutoff_str,
        )
        return deleted

    def close(self) -> None:
        """데이터베이스 연결을 닫는다."""
        self._conn.close()
        logger.info("DataStore 연결 종료")
