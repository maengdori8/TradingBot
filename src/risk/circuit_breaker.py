"""
서킷브레이커 — 일일/주간 손실 한도 및 연속 손실 관리
SQLite로 상태 영속화 (봇 재시작 후에도 유지)
"""
from __future__ import annotations
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "circuit_breaker.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cb_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pnl REAL,
            pnl_pct REAL,
            recorded_at TEXT
        )
    """)
    conn.commit()
    return conn


def _set(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT OR REPLACE INTO cb_state(key,value,updated_at) VALUES(?,?,?)", (key, value, now))
    conn.commit()


def _get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM cb_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


class CircuitBreaker:
    """서킷브레이커 — 손실 한도 초과 시 거래 차단."""

    def __init__(
        self,
        trading_capital: float,
        daily_loss_limit: float = 0.03,
        weekly_loss_limit: float = 0.08,
        max_consecutive_losses: int = 3,
    ) -> None:
        """
        Args:
            trading_capital: 트레이딩 자본 (USDT)
            daily_loss_limit: 일일 손실 한도 비율
            weekly_loss_limit: 주간 손실 한도 비율
            max_consecutive_losses: 최대 연속 손실 횟수
        """
        self.capital = trading_capital
        self.daily_limit = daily_loss_limit
        self.weekly_limit = weekly_loss_limit
        self.max_consec = max_consecutive_losses

    # ------------------------------------------------------------------

    def record_trade(self, pnl: float) -> None:
        """거래 결과 기록."""
        pnl_pct = pnl / self.capital
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO trade_results(pnl, pnl_pct, recorded_at) VALUES(?,?,?)",
                (pnl, pnl_pct, datetime.now(timezone.utc).isoformat()),
            )
            # 연속 손실 추적
            consec = int(_get(conn, "consecutive_losses", "0"))
            if pnl < 0:
                consec += 1
            else:
                consec = 0
            _set(conn, "consecutive_losses", str(consec))
            conn.commit()
        logger.info("거래 기록: PnL=%.2f (%.2f%%) 연속손실=%d", pnl, pnl_pct * 100, consec)

    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        거래 허용 여부 확인.

        Returns:
            (allowed: bool, reason: str)
        """
        with _get_conn() as conn:
            now = datetime.now(timezone.utc)

            # 연속 손실 휴식 체크
            consec = int(_get(conn, "consecutive_losses", "0"))
            if consec >= self.max_consec:
                rest_until_str = _get(conn, "rest_until", "")
                if rest_until_str:
                    rest_until = datetime.fromisoformat(rest_until_str)
                    if now < rest_until:
                        return False, f"연속 {consec}패 — {rest_until.strftime('%Y-%m-%d %H:%M')} UTC까지 휴식"
                else:
                    rest_until = now + timedelta(days=1)
                    _set(conn, "rest_until", rest_until.isoformat())
                    return False, f"연속 {consec}패 — 1일 강제 휴식"

            # 일일 손실 체크
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            rows = conn.execute(
                "SELECT SUM(pnl) FROM trade_results WHERE recorded_at >= ?",
                (today_start.isoformat(),),
            ).fetchone()
            daily_pnl = rows[0] or 0.0
            if daily_pnl < -(self.capital * self.daily_limit):
                return False, f"일일 손실 한도 초과: {daily_pnl:.2f} USDT"

            # 주간 손실 체크
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            rows = conn.execute(
                "SELECT SUM(pnl) FROM trade_results WHERE recorded_at >= ?",
                (week_start.isoformat(),),
            ).fetchone()
            weekly_pnl = rows[0] or 0.0
            if weekly_pnl < -(self.capital * self.weekly_limit):
                return False, f"주간 손실 한도 초과: {weekly_pnl:.2f} USDT"

        return True, "거래 가능"

    def get_daily_pnl(self) -> float:
        """오늘 누적 PnL 조회."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT SUM(pnl) FROM trade_results WHERE recorded_at >= ?",
                (today_start.isoformat(),),
            ).fetchone()
        return row[0] or 0.0

    def reset_consecutive_losses(self) -> None:
        """연속 손실 카운터 초기화 (수동 리셋용)."""
        with _get_conn() as conn:
            _set(conn, "consecutive_losses", "0")
            _set(conn, "rest_until", "")
        logger.info("연속 손실 카운터 초기화")
