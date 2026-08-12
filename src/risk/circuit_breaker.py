from __future__ import annotations

# 일일·주간·연속 손실을 SQLite에 영속화하는 서킷브레이커.
# 휴식 후 자동 복귀하고 월요일 00:00 UTC에 주간 상태를 리셋한다.

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "circuit_breaker.db"


def _get_conn() -> sqlite3.Connection:
    """SQLite 연결 생성 및 테이블 초기화."""
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
    """상태 키-값 저장."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO cb_state(key,value,updated_at) VALUES(?,?,?)",
        (key, value, now),
    )
    conn.commit()


def _get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    """상태 키-값 조회."""
    row = conn.execute(
        "SELECT value FROM cb_state WHERE key=?", (key,)
    ).fetchone()
    return row[0] if row else default


class CircuitBreaker:
    """서킷브레이커 — 손실 한도 초과 시 거래 차단."""

    def __init__(
        self,
        trading_capital: float,
        daily_loss_limit: float = 0.05,
        weekly_loss_limit: float = 0.12,
        max_consecutive_losses: int = 7,
    ) -> None:
        """
        서킷브레이커 초기화.

        Args:
            trading_capital: 트레이딩 자본 (USDT)
            daily_loss_limit: 일일 손실 한도 비율 (기본 3%)
            weekly_loss_limit: 주간 손실 한도 비율 (기본 8%)
            max_consecutive_losses: 최대 연속 손실 횟수 (기본 3)
        """
        self.capital: float = trading_capital
        self.daily_limit: float = daily_loss_limit
        self.weekly_limit: float = weekly_loss_limit
        self.max_consec: int = max_consecutive_losses

    # ------------------------------------------------------------------
    # 주간 리셋
    # ------------------------------------------------------------------

    def _check_weekly_reset(self, conn: sqlite3.Connection) -> None:
        """
        주간 리셋 검사 — 월요일 00:00 UTC에 연속 손실 카운터 및 상태 초기화.

        Args:
            conn: SQLite 연결
        """
        now = datetime.now(timezone.utc)
        last_reset_str = _get(conn, "last_weekly_reset", "")

        if last_reset_str:
            last_reset = datetime.fromisoformat(last_reset_str)
        else:
            # 최초 실행: 직전 월요일로 설정
            last_reset = now - timedelta(days=now.weekday())
            last_reset = last_reset.replace(hour=0, minute=0, second=0, microsecond=0)
            _set(conn, "last_weekly_reset", last_reset.isoformat())

        # 현재 주의 월요일 00:00 UTC 계산
        current_monday = now - timedelta(days=now.weekday())
        current_monday = current_monday.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        if current_monday > last_reset:
            # 새 주가 시작됨 — 리셋
            _set(conn, "consecutive_losses", "0")
            _set(conn, "rest_until", "")
            _set(conn, "last_weekly_reset", current_monday.isoformat())
            logger.info("주간 리셋 완료: 연속 손실 카운터 초기화")

    # ------------------------------------------------------------------
    # 거래 기록
    # ------------------------------------------------------------------

    def record_trade(self, pnl: float) -> None:
        """
        거래 결과 기록.

        Args:
            pnl: 실현 손익 (USDT)
        """
        pnl_pct = pnl / self.capital if self.capital > 0 else 0.0
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
        logger.info(
            "거래 기록: PnL=%.2f (%.2f%%) 연속손실=%d", pnl, pnl_pct * 100, consec
        )

    # ------------------------------------------------------------------
    # 거래 허용 여부
    # ------------------------------------------------------------------

    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        거래 허용 여부 확인.

        Returns:
            (allowed: bool, reason: str) 튜플
        """
        with _get_conn() as conn:
            now = datetime.now(timezone.utc)

            # 주간 리셋 체크 (월요일 00:00 UTC)
            self._check_weekly_reset(conn)

            # 연속 손실 휴식 체크 (자동 복귀 포함)
            consec = int(_get(conn, "consecutive_losses", "0"))
            if consec >= self.max_consec:
                rest_until_str = _get(conn, "rest_until", "")
                if rest_until_str:
                    rest_until = datetime.fromisoformat(rest_until_str)
                    if now < rest_until:
                        return False, (
                            f"연속 {consec}패 — "
                            f"{rest_until.strftime('%Y-%m-%d %H:%M')} UTC까지 휴식"
                        )
                    # 휴식 기간 만료 — 자동 복귀
                    _set(conn, "consecutive_losses", "0")
                    _set(conn, "rest_until", "")
                    logger.info(
                        "휴식 기간 만료 — 연속 손실 카운터 리셋, 거래 재개"
                    )
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

    # ------------------------------------------------------------------
    # 조회 / 리셋
    # ------------------------------------------------------------------

    def get_daily_pnl(self) -> float:
        """
        오늘 누적 PnL 조회.

        Returns:
            오늘 누적 손익 (USDT)
        """
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT SUM(pnl) FROM trade_results WHERE recorded_at >= ?",
                (today_start.isoformat(),),
            ).fetchone()
        return row[0] or 0.0

    def get_consecutive_losses(self) -> int:
        """현재 연속 손실 횟수 조회 (연패 단계 리스크 축소용).

        Returns:
            연속 손실 횟수 (승리 시 0으로 리셋됨)
        """
        with _get_conn() as conn:
            return int(_get(conn, "consecutive_losses", "0"))

    def get_weekly_pnl(self) -> float:
        """
        이번 주 누적 PnL 조회.

        Returns:
            이번 주 누적 손익 (USDT)
        """
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT SUM(pnl) FROM trade_results WHERE recorded_at >= ?",
                (week_start.isoformat(),),
            ).fetchone()
        return row[0] or 0.0

    def reset_consecutive_losses(self) -> None:
        """연속 손실 카운터 초기화 (수동 리셋용)."""
        with _get_conn() as conn:
            _set(conn, "consecutive_losses", "0")
            _set(conn, "rest_until", "")
        logger.info("연속 손실 카운터 초기화")
