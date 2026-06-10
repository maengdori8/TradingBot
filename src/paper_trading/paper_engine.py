"""
페이퍼 트레이딩 엔진 — 가상 주문 실행 및 성과 추적
슬리피지 0.05%, 수수료 0.055% (Bybit Taker 기준)
잔고 = 담보금 + 미실현손익 기반으로 관리
TradingEngine 추상 인터페이스 구현
잔고 영속화: engine_state 테이블에 저장/복원
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from src.paper_trading import Position, TradingEngine

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "paper_trades.db"

SLIPPAGE = 0.0005  # 0.05%
TAKER_FEE = 0.00055  # 0.055%
# 펀딩비 보수 가정 (Bybit 00/08/16 UTC, 통상 0.01%/8h — 방향 무관 비용으로 처리)
FUNDING_RATE = 0.0001
FUNDING_HOURS = (0, 8, 16)


def _funding_events(entry_time: datetime, exit_time: datetime) -> int:
    """보유 구간에 통과한 펀딩 정산 시각(00/08/16 UTC) 횟수."""
    if exit_time <= entry_time:
        return 0
    count = 0
    # 진입 직후 첫 펀딩 시각부터 청산 전까지 8시간 간격 스캔
    t = entry_time.replace(minute=0, second=0, microsecond=0)
    while t <= exit_time:
        if t.hour in FUNDING_HOURS and t > entry_time:
            count += 1
        t += timedelta(hours=1)
    return count


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
    })
    _migrate_columns(conn, "open_positions", {
        "entry_score": "REAL", "entry_session": "TEXT",
        "entry_checks_json": "TEXT", "entry_rr": "REAL", "risk_amount": "REAL",
    })
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
    ) -> None:
        """
        페이퍼 엔진 초기화.

        Args:
            initial_balance: 초기 잔고 (USDT)
            db_path: DB 파일 경로. None이면 기본 경로 사용.
        """
        self.initial_balance: float = initial_balance
        self._positions: list[Position] = []
        self.conn: sqlite3.Connection = _init_db(db_path)
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
                entry_score, entry_session, entry_checks_json, entry_rr, risk_amount)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            "entry_score, entry_session, entry_checks_json, entry_rr, risk_amount "
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

    def _fee(self, notional: float) -> float:
        """
        수수료 계산.

        Args:
            notional: 명목 금액

        Returns:
            수수료 금액
        """
        return round(notional * TAKER_FEE, 8)

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

        Returns:
            생성된 Position 또는 잔고 부족 시 None
        """
        actual_entry = self._apply_slippage(entry_price, direction, is_entry=True)
        notional = round(actual_entry * qty, 8)
        lev = leverage if leverage and leverage > 0 else 1.0
        margin = round(notional / lev, 8)        # 실제 묶이는 증거금
        entry_fee = self._fee(notional)          # 수수료는 명목가 기준
        total_cost = round(margin + entry_fee, 8)  # 증거금 + 수수료

        if total_cost > self._balance:
            logger.warning(
                "잔고 부족: 필요=%.2f 보유=%.2f", total_cost, self._balance
            )
            return None

        self._balance = round(self._balance - total_cost, 8)  # 증거금 + 수수료 차감
        self._save_balance()
        # R-multiple 산출용 리스크 금액 (슬리피지 적용 진입가 기준)
        risk_amount = round(abs(actual_entry - stop_loss) * qty, 8)
        pos = Position(
            id=_generate_position_id(),
            symbol=symbol,
            direction=direction,
            entry_price=actual_entry,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin=margin,
            entry_score=score,
            entry_session=entry_session,
            entry_checks=checks,
            entry_rr=entry_rr,
            risk_amount=risk_amount,
        )
        self._positions.append(pos)
        self._save_position(pos)
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
    ) -> float:
        """
        포지션 청산 — 담보금 반환 + PnL. 부분 청산 지원.

        Args:
            position: 청산할 포지션
            exit_price: 청산 가격
            reason: 청산 사유 (예: "SL", "TP", "manual")
            qty: 부분 청산 수량 (None이면 전량 청산)

        Returns:
            실현 손익 (USDT)
        """
        close_qty = qty if qty is not None else position.qty
        if close_qty > position.qty:
            logger.warning(
                "청산 수량(%.6f)이 보유 수량(%.6f)을 초과합니다. 전량 청산합니다.",
                close_qty,
                position.qty,
            )
            close_qty = position.qty

        is_partial = close_qty < position.qty
        close_ratio = close_qty / position.qty

        actual_exit = self._apply_slippage(
            exit_price, position.direction, is_entry=False
        )
        exit_fee = self._fee(round(actual_exit * close_qty, 8))

        if position.direction == "long":
            gross_pnl = round((actual_exit - position.entry_price) * close_qty, 8)
        else:
            gross_pnl = round((position.entry_price - actual_exit) * close_qty, 8)

        # 펀딩비: 보유 중 통과한 정산 시각(00/08/16 UTC) × 0.01% × 명목가 (보수 가정, 비용 처리)
        funding_n = _funding_events(
            position.entry_time, datetime.now(timezone.utc)
        )
        funding_cost = round(
            funding_n * FUNDING_RATE * position.entry_price * close_qty, 8
        )

        pnl = round(gross_pnl - exit_fee - funding_cost, 8)
        released_margin = round(position.margin * close_ratio, 8)
        pnl_pct = pnl / released_margin if released_margin > 0 else 0.0

        # 담보금 반환 + 순이익
        self._balance = round(self._balance + released_margin + pnl, 8)
        self._save_balance()

        # R-multiple = 순손익 / 리스크금액 (이번 청산분 비례). risk_amount 없으면 None.
        if position.risk_amount and position.risk_amount > 0:
            risk_portion = round(position.risk_amount * close_ratio, 8)
            r_multiple = round(pnl / risk_portion, 6)
        else:
            risk_portion = None
            r_multiple = None

        ck = position.entry_checks or {}
        def _b(key: str) -> int | None:
            return int(bool(ck[key])) if key in ck else None

        now = datetime.now(timezone.utc).isoformat()
        trade_id = f"{position.id}-{now}" if is_partial else position.id
        self.conn.execute(
            """INSERT INTO trades
               (id, symbol, direction, entry_price, exit_price,
                qty, pnl, pnl_pct, margin, entry_time, exit_time, status,
                entry_score, entry_session, c_trend, c_zone, c_kill_zone,
                c_ote, c_volume, c_rr, entry_rr, risk_amount, r_multiple,
                funding_cost)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                position.symbol,
                position.direction,
                position.entry_price,
                actual_exit,
                close_qty,
                pnl,
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
            ),
        )
        self.conn.commit()

        if is_partial:
            # 부분 청산: 잔여 수량/마진/리스크금액 비례 감소
            position.qty = round(position.qty - close_qty, 8)
            position.margin = round(position.margin - released_margin, 8)
            if position.risk_amount is not None and risk_portion is not None:
                position.risk_amount = round(position.risk_amount - risk_portion, 8)
            self._save_position(position)
            logger.info(
                "[PAPER] %s 부분청산(%s): qty=%.6f PnL=%.4f (%.2f%%) 잔여=%.6f",
                position.symbol,
                reason,
                close_qty,
                pnl,
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
                pnl,
                pnl_pct * 100,
            )

        self._fire_on_trade(pnl, reason, position)
        return pnl

    def check_stops(
        self, symbol: str, current_high: float, current_low: float
    ) -> None:
        """
        SL/TP 자동 체크 — 해당 심볼의 모든 포지션 검사.

        Args:
            symbol: 거래 심볼
            current_high: 현재 캔들 고가
            current_low: 현재 캔들 저가
        """
        for pos in list(self._positions):
            if pos.symbol != symbol:
                continue
            if pos.direction == "long":
                if current_low <= pos.stop_loss:
                    self.close_position(pos, pos.stop_loss, "SL")
                elif current_high >= pos.take_profit:
                    self.close_position(pos, pos.take_profit, "TP")
            else:
                if current_high >= pos.stop_loss:
                    self.close_position(pos, pos.stop_loss, "SL")
                elif current_low <= pos.take_profit:
                    self.close_position(pos, pos.take_profit, "TP")

    # ------------------------------------------------------------------
    # 미실현 손익 / 트레일링 스톱
    # ------------------------------------------------------------------

    def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """
        미실현 손익 갱신 — 해당 심볼의 모든 포지션.

        Args:
            symbol: 거래 심볼
            current_price: 현재 가격
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

    # ------------------------------------------------------------------
    # 성과 지표
    # ------------------------------------------------------------------

    def get_performance(self) -> dict:
        """
        성과 지표 계산.

        Returns:
            성과 통계 딕셔너리
        """
        rows = self.conn.execute(
            "SELECT pnl, pnl_pct FROM trades"
        ).fetchall()

        if not rows:
            return {"message": "거래 내역 없음"}

        pnls = [r[0] for r in rows]
        pnl_pcts = [r[1] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity = np.array(
            [self.initial_balance]
            + list(np.cumsum(pnls) + self.initial_balance)
        )
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
        mdd = float(drawdown.max())

        pnl_arr = np.array(pnl_pcts)
        sharpe = (
            float(pnl_arr.mean() / pnl_arr.std() * math.sqrt(252))
            if pnl_arr.std() > 0
            else 0.0
        )

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss

        return {
            "total_trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_pnl": round(sum(pnls) / len(pnls), 8),
            "total_pnl": round(sum(pnls), 8),
            "mdd": mdd,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "current_balance": self._balance,
            "return_pct": (self._balance - self.initial_balance)
            / self.initial_balance,
        }
