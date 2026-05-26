"""
OHLCV 데이터 SQLite 저장소.
시세 데이터를 로컬에 영속적으로 보관하여 API 호출을 절약한다.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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

        rows: list[tuple[Any, ...]] = []
        for ts, row in df.iterrows():
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append((
                symbol, timeframe, ts_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))

        self._conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv
                (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
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

    def close(self) -> None:
        """데이터베이스 연결을 닫는다."""
        self._conn.close()
        logger.info("DataStore 연결 종료")
