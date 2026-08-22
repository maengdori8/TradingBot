"""
페이퍼 트레이딩 엔진 — 가상 주문 실행 및 성과 추적
슬리피지 0.05%, 수수료 0.055% (Bybit Taker 기준)
잔고 = 담보금 + 미실현손익 기반으로 관리
TradingEngine 추상 인터페이스 구현
잔고 영속화: engine_state 테이블에 저장/복원

정직한 기록 규약 (2026-08 패치 — 연구 exec_a1/study 재생과 동일 의미):
- 지정가 체결·SL/TP 판정은 **완전히 닫힌 15m 봉**만 사용한다(형성 중 봉은 쓰지 않는다 — 관측 기준
  시각 이후에 받은 부분 봉을 해석하지 않기 위함; 닫힌 뒤 다음 사이클에 판정). 캔들 인덱스 이상 시
  fail-closed(체결·만료 없음).
- **관측하지 못한 봉은 '없었던 봉'이 아니다.** 캔들 프레임이 판정 시작점(주문 다음 봉 / 마지막 판정봉
  다음 봉)까지 거슬러 닿지 않으면 판정을 보류하고 `last_history_gaps`에 심볼을 올린다(봇이 더 긴
  프레임으로 재조회). 프레임 끝이 현재에 못 미치면(지연 프레임) 있는 데까지만 판정한다. 프레임
  '안'에 비어 있는 봉도 기본적으로는 미관측으로 보고 보류한다(`allow_holes=False`); 봇이 더 깊은
  프레임(1000봉/히스토리 백필)으로 재확인한 뒤에도 비어 있으면 그때만 거래소 무데이터(거래 없음)로
  인정한다(`allow_holes=True`). 만료는 체결 창의 모든 봉을 관측했을 때만 선언한다. NaN 고저는 관측
  실패로 본다(보류). 같은 시각 중복 행은 고가=max/저가=min으로 보수 병합한다.
- 실행 판정에 쓰는 캔들은 거래 대상 거래소(Bybit 무기한)의 것이어야 한다 — 다른 거래소 현물 폴백
  캔들로 체결/SL/TP를 판정하지 않는다(봇이 Bybit 전용 조회를 쓰고, 불가 시 관찰 전용 모드).
- 진입(체결)봉에서는 SL만 인정하고 TP는 인정하지 않는다(봉 내부 순서 불명 → 보수). 이후 봉은 SL 우선.
- 기록 PnL / R-multiple은 진입 수수료를 포함한 순손익이다(잔고 반영과 분리 계산, 이중 차감 없음).
- 미체결 지정가는 signal_key(심볼|방향|결정봉)로 중복 등록을 막고, 체결/만료/취소 이력을 남긴다.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np
import pandas as pd

from src.paper_trading import Position, TradingEngine

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "logs" / "paper_trades.db"

SLIPPAGE = 0.0005  # 0.05%
TAKER_FEE = 0.00055  # 0.055%
MAKER_FEE = 0.00035  # 0.035% (지정가 메이커 — config execution.maker_fee로 재정의 가능)
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
    # 지정가 주문 원장 (메이커 실행 — 사이클 간 영속). status: pending|filled|expired|cancelled
    # 체결/만료/취소된 행은 삭제하지 않고 이력으로 남긴다(체결률·재주문 감사용).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_orders (
            id TEXT PRIMARY KEY,
            symbol TEXT, direction TEXT,
            limit_price REAL, qty REAL,
            stop_loss REAL, take_profit REAL,
            leverage REAL, place_time TEXT, expiry_time TEXT,
            entry_score REAL, entry_session TEXT,
            entry_checks_json TEXT, entry_rr REAL,
            status TEXT, signal_key TEXT, decision_bar TEXT,
            bar_minutes INT, expiry_bars INT,
            position_id TEXT, resolved_time TEXT, resolve_reason TEXT
        )
    """)
    _migrate_columns(conn, "pending_orders", {
        "status": "TEXT", "signal_key": "TEXT", "decision_bar": "TEXT",
        "bar_minutes": "INT", "expiry_bars": "INT",
        "position_id": "TEXT", "resolved_time": "TEXT", "resolve_reason": "TEXT",
    })
    # 같은 결정봉의 같은 신호는 한 번만 주문 (NULL은 SQLite 유니크 제약에서 서로 다름 = 구버전 행 허용)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_signal_key ON pending_orders(signal_key)"
    )
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
        "entry_fee": "REAL", "is_maker": "INT", "last_checked_bar": "TEXT",
    })
    _migrate_columns(conn, "trades", {"entry_fee": "REAL", "is_maker": "INT"})
    _migrate_legacy_fees(conn)
    conn.commit()
    return conn


SCHEMA_VERSION = 2


def _migrate_legacy_fees(conn: sqlite3.Connection) -> None:
    """1회성 마이그레이션(v2): 진입 수수료를 기록 PnL에 포함하는 정의로 레거시 행을 맞춘다.

    구버전은 진입 수수료를 잔고에서만 차감하고 trades.pnl / r_multiple 에는 빠뜨렸다(시장가만 존재).
    - open_positions.entry_fee NULL → entry_price×qty×TAKER_FEE, is_maker=0
    - trades.entry_fee NULL → 수수료 산출 후 pnl/pnl_pct/r_multiple 재계산
    완료 후 engine_state.schema_version=2 기록(멱등).
    """
    row = conn.execute("SELECT value FROM engine_state WHERE key='schema_version'").fetchone()
    if row is not None and int(row[0]) >= SCHEMA_VERSION:
        return
    n_pos = conn.execute(
        "UPDATE open_positions SET entry_fee = ROUND(entry_price*qty*?, 8), is_maker = 0 "
        "WHERE entry_fee IS NULL", (TAKER_FEE,)
    ).rowcount
    rows = conn.execute(
        "SELECT id, entry_price, qty, pnl, margin, risk_amount FROM trades WHERE entry_fee IS NULL"
    ).fetchall()
    for tid, ep, qty, pnl, margin, risk in rows:
        fee = round(float(ep or 0.0) * float(qty or 0.0) * TAKER_FEE, 8)
        new_pnl = round(float(pnl or 0.0) - fee, 8)
        pnl_pct = new_pnl / margin if margin else 0.0
        r_mult = round(new_pnl / risk, 6) if risk else None
        conn.execute(
            "UPDATE trades SET entry_fee=?, is_maker=0, pnl=?, pnl_pct=?, r_multiple=? WHERE id=?",
            (fee, new_pnl, pnl_pct, r_mult, tid),
        )
    conn.execute(
        "INSERT OR REPLACE INTO engine_state(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    if n_pos or rows:
        logger.info("스키마 v2 마이그레이션: 진입수수료 보정 — 포지션 %d건, 거래 %d건", n_pos, len(rows))


def _bar_open(ts: datetime, bar_minutes: int) -> datetime:
    """시각이 속한 봉의 시작시각(그리드 내림). 예: 12:07 → 12:00 (15m)."""
    return ts.replace(second=0, microsecond=0) - timedelta(minutes=ts.minute % bar_minutes)


def _ensure_utc_index(candles: pd.DataFrame) -> pd.DataFrame:
    """캔들 인덱스를 UTC-aware DatetimeIndex(오름차순)로 정규화한다.

    Raises:
        ValueError: DatetimeIndex가 아니거나 정규화 불가 — 호출측은 fail-closed 처리한다.
    """
    if not isinstance(candles, pd.DataFrame) or not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("캔들 인덱스가 DatetimeIndex가 아님")
    candles = candles.copy()
    if candles.index.tz is None:
        candles.index = candles.index.tz_localize("UTC")
    else:
        candles.index = candles.index.tz_convert("UTC")
    if not candles.index.is_monotonic_increasing:
        candles = candles.sort_index()
    if candles.index.has_duplicates:
        # 같은 시각 중복 행 — 어느 쪽이 진짜인지 알 수 없으므로 보수 병합: 고가=max, 저가=min
        # (SL 터치 저가를 버리고 TP 고가만 남기는 일이 없게). 그 외 컬럼은 open=first/close=last/volume=sum.
        agg: dict[str, str] = {}
        for col in candles.columns:
            agg[col] = {"high": "max", "low": "min", "open": "first",
                        "close": "last", "volume": "sum"}.get(col, "last")
        candles = candles.groupby(level=0).agg(agg)
    return candles


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
        maker_fee: float = MAKER_FEE,
    ) -> None:
        """
        페이퍼 엔진 초기화.

        Args:
            initial_balance: 초기 잔고 (USDT)
            db_path: DB 파일 경로. None이면 기본 경로 사용.
            maker_fee: 메이커(지정가) 수수료율 (config execution.maker_fee).
        """
        self.initial_balance: float = initial_balance
        self._maker_fee: float = maker_fee
        self._positions: list[Position] = []
        self._txn_depth: int = 0            # >0 이면 헬퍼의 commit 보류 (원자적 묶음용)
        self.conn: sqlite3.Connection = _init_db(db_path)
        self._on_trade_callbacks: list[Callable[[float, str, Position], None]] = []

        # DB에서 잔고 복원 시도 → 없으면 initial_balance 사용 (기준 행을 즉시 기록해 롤백 기준점 확보)
        restored = self._restore_balance()
        self._balance: float = restored if restored is not None else initial_balance
        if restored is None:
            self._save_balance()
        # 판정 보류된 심볼(프레임이 판정 시작점까지 못 미침 / 빈 봉) — 봇이 더 깊은 프레임으로 재조회한다
        self.last_history_gaps: set[str] = set()
        # 이번 호출에서 판정을 끝내지 못한 심볼(어떤 이유든: 갭·빈 봉·지연 프레임·NaN·빈/이상 프레임).
        # 봇은 기존 포지션 판정이 전부 완료된 사이클에만 미체결 체결 판정을 진행한다(시간순 보장).
        self.last_eval_incomplete: set[str] = set()

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
    # 트랜잭션 헬퍼 — 여러 DB 변경을 하나의 커밋으로 묶는다
    # ------------------------------------------------------------------

    def _commit(self) -> None:
        """트랜잭션 묶음 밖일 때만 실제 commit."""
        if self._txn_depth == 0:
            self.conn.commit()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """블록 내 모든 DB 변경을 단일 트랜잭션(BEGIN IMMEDIATE)으로 묶는다.

        본문 예외든 최종 commit 예외든 → 롤백하고 메모리 잔고/포지션을 진입 시점 스냅샷으로 되돌린 뒤
        재전파한다. 중첩 호출은 가장 바깥 블록만 실제 BEGIN/COMMIT 한다.
        """
        outer = self._txn_depth == 0
        if outer:
            if self.conn.in_transaction:
                # 이전 갱신의 commit 실패 흔적(미커밋 DML). 조용히 확정하지 않고 버린다 — 부분 상태를
                # 스냅샷 밖에서 커밋하면 이번 롤백으로 되돌릴 수 없게 되기 때문.
                logger.error("[PAPER] 커밋되지 않은 잔여 트랜잭션 발견 — 롤백 후 진행")
                self.conn.rollback()
            snap_balance = self._balance
            snap_positions = [dataclasses.replace(p) for p in self._positions]
            self.conn.execute("BEGIN IMMEDIATE")
        self._txn_depth += 1
        try:
            yield
            if outer:
                self.conn.commit()
        except BaseException:
            if outer:
                try:
                    self.conn.rollback()
                except sqlite3.Error as e:  # pragma: no cover - 롤백 자체 실패는 로그만
                    logger.error("트랜잭션 롤백 실패: %s", e)
                self._balance = snap_balance
                self._positions = snap_positions
            raise
        finally:
            self._txn_depth -= 1

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
        self._commit()

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
                entry_fee, is_maker, last_checked_bar)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                int(bool(pos.is_maker)),
                pos.last_checked_bar.isoformat() if pos.last_checked_bar else None,
            ),
        )
        self._commit()

    def _remove_position_from_db(self, position_id: str) -> None:
        """청산된 포지션을 open_positions에서 제거."""
        self.conn.execute(
            "DELETE FROM open_positions WHERE id = ?", (position_id,)
        )
        self._commit()

    def _restore_positions(self) -> None:
        """봇 재시작 시 열린 포지션 복원 (진입조건 포함)."""
        rows = self.conn.execute(
            "SELECT id, symbol, direction, entry_price, qty, stop_loss, take_profit, margin, entry_time, "
            "entry_score, entry_session, entry_checks_json, entry_rr, risk_amount, "
            "entry_fee, is_maker, last_checked_bar "
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
                entry_fee=float(row[14]) if row[14] is not None else 0.0,
                is_maker=bool(row[15]) if row[15] is not None else False,
                last_checked_bar=datetime.fromisoformat(row[16]) if row[16] else None,
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

    def _fee(self, notional: float, maker: bool = False) -> float:
        """
        수수료 계산.

        Args:
            notional: 명목 금액
            maker: 메이커(지정가) 여부 — True면 메이커 수수료 적용

        Returns:
            수수료 금액
        """
        rate = self._maker_fee if maker else TAKER_FEE
        return round(notional * rate, 8)

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
        entry_time: datetime | None = None,
        is_maker: bool = False,
        position_id: str | None = None,
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
            is_maker: 지정가(메이커) 체결 — 슬리피지 0 + 메이커 수수료
            position_id: 포지션 id 강제 지정 (지정가 체결 시 주문 id 재사용 → 추적/멱등)

        Returns:
            생성된 Position 또는 잔고 부족 시 None
        """
        # 메이커(지정가): 슬리피지 없음 — 내가 건 지정가에 정확히 체결. 메이커 수수료.
        # 테이커(시장가): 불리한 슬리피지 + 테이커 수수료 (기존).
        if is_maker:
            actual_entry = round(entry_price, 8)
        else:
            actual_entry = self._apply_slippage(entry_price, direction, is_entry=True)
        notional = round(actual_entry * qty, 8)
        lev = leverage if leverage and leverage > 0 else 1.0
        margin = round(notional / lev, 8)        # 실제 묶이는 증거금
        entry_fee = self._fee(notional, maker=is_maker)  # 수수료는 명목가 기준
        total_cost = round(margin + entry_fee, 8)  # 증거금 + 수수료

        if total_cost > self._balance:
            logger.warning(
                "잔고 부족: 필요=%.2f 보유=%.2f", total_cost, self._balance
            )
            return None

        with self._transaction():   # 잔고 차감 + 포지션 저장을 원자적으로
            self._balance = round(self._balance - total_cost, 8)  # 증거금 + 수수료 차감
            self._save_balance()
            # R-multiple 산출용 리스크 금액 (슬리피지 적용 진입가 기준)
            risk_amount = round(abs(actual_entry - stop_loss) * qty, 8)
            pos = Position(
                id=position_id or _generate_position_id(),
                symbol=symbol,
                direction=direction,
                entry_price=actual_entry,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                margin=margin,
                entry_time=entry_time or datetime.now(timezone.utc),
                entry_score=score,
                entry_session=entry_session,
                entry_checks=checks,
                entry_rr=entry_rr,
                risk_amount=risk_amount,
                entry_fee=entry_fee,
                is_maker=is_maker,
            )
            self._positions.append(pos)
            self._save_position(pos)
        logger.info(
            "[PAPER] %s %s 진입: id=%s price=%.4f qty=%.4f margin=%.2f lev=%gx notional=%.2f%s",
            symbol,
            direction.upper(),
            pos.id,
            actual_entry,
            qty,
            margin,
            lev,
            notional,
            " [maker]" if is_maker else "",
        )
        return pos

    # ------------------------------------------------------------------
    # 지정가 주문 (메이커 실행) — 사이클 간 영속, 닫힌 봉 기준 체결 판정
    # ------------------------------------------------------------------

    def _pending_reservation(self, od: dict) -> float:
        """미체결 주문 1건이 체결 시 묶을 증거금+수수료 (잔고 예약분)."""
        notional = float(od["limit_price"]) * float(od["qty"])
        lev = float(od["leverage"] or 1.0)
        return round(notional / lev + self._fee(notional, maker=True), 8)

    def reserved_margin(self) -> float:
        """모든 미체결 지정가가 체결될 때 필요한 증거금+수수료 합."""
        return round(sum(self._pending_reservation(od) for od in self.get_pending_orders()), 8)

    def available_balance(self) -> float:
        """잔고 − 미체결 예약분 (신규 주문/진입 가능 금액)."""
        return round(self._balance - self.reserved_margin(), 8)

    def required_history_start(self, symbol: str, bar_minutes: int = 15) -> datetime | None:
        """심볼의 판정을 완료하려면 캔들이 최소 어느 시각부터 있어야 하는지 (백필 조회 시작점).

        보유 포지션: last_checked_bar 다음 봉(없으면 진입봉). 미체결 주문: 창 첫 봉. 그중 가장 이른 시각.
        """
        starts: list[datetime] = []
        for pos in self._positions:
            if pos.symbol != symbol:
                continue
            if pos.last_checked_bar is not None:
                starts.append(pos.last_checked_bar + timedelta(minutes=bar_minutes))
            else:
                starts.append(_bar_open(pos.entry_time, bar_minutes))
        for od in self.get_pending_orders(symbol):
            bm = int(od.get("bar_minutes") or bar_minutes)
            starts.append(_bar_open(datetime.fromisoformat(od["place_time"]), bm) + timedelta(minutes=bm))
        return min(starts) if starts else None

    def pending_as_positions(self) -> list[Position]:
        """미체결 지정가를 리스크 한도 계산용 가상 포지션으로 변환한다.

        방향 쏠림/총 노출/명목노출/심볼당 개수 한도에 '예약분'으로 포함시키기 위한 것으로,
        잔고·성과에는 영향이 없다. id=주문 id.
        """
        out: list[Position] = []
        for od in self.get_pending_orders():
            limit = float(od["limit_price"])
            qty = float(od["qty"])
            lev = float(od["leverage"] or 1.0)
            out.append(Position(
                id=od["id"], symbol=od["symbol"], direction=od["direction"],
                entry_price=limit, qty=qty, stop_loss=float(od["stop_loss"]),
                take_profit=float(od["take_profit"]), margin=round(limit * qty / lev, 8),
                entry_time=datetime.fromisoformat(od["place_time"]),
                entry_score=od["entry_score"], entry_session=od["entry_session"],
                entry_rr=od["entry_rr"], is_maker=True,
            ))
        return out

    def place_pending_limit(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        limit_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        leverage: float = 1.0,
        expiry_bars: int = 8,
        bar_minutes: int = 15,
        score: float | None = None,
        checks: dict | None = None,
        entry_rr: float | None = None,
        entry_session: str | None = None,
        place_time: datetime | None = None,
        signal_key: str | None = None,
        decision_bar: datetime | None = None,
    ) -> str | None:
        """지정가 주문을 등록한다 (체결은 check_pending_fills에서 닫힌 봉 기준으로 판정).

        체결 창 = 주문 시각이 속한 봉 **다음** 봉부터 expiry_bars개 봉(연구 exec_a1의 FILL_WINDOW와 동일).

        Args:
            signal_key: 신호 고유키("심볼|방향|결정봉"). 같은 키가 이미 등록(어떤 상태든)돼 있으면
                중복으로 보고 등록하지 않는다(같은 결정봉에서 두 번 주문 금지). None이면 검사 생략.
            decision_bar: 신호가 계산된 결정봉 시작시각(기록용).

        Returns:
            주문 id, 또는 중복/잔고예약 부족/심볼 중복으로 등록하지 않았으면 None
        """
        pt = place_time or datetime.now(timezone.utc)
        if signal_key:
            dup = self.conn.execute(
                "SELECT id, status FROM pending_orders WHERE signal_key=?", (signal_key,)
            ).fetchone()
            if dup is not None:
                logger.info("[PAPER] 지정가 중복 등록 거부 (signal_key=%s, 기존=%s/%s)",
                            signal_key, dup[0], dup[1])
                return None
        if self.get_pending_orders(symbol):
            logger.info("[PAPER] 지정가 등록 거부: %s 이미 미체결 주문 있음", symbol)
            return None
        probe = {"limit_price": limit_price, "qty": qty, "leverage": leverage}
        need = self._pending_reservation(probe)
        if need > self.available_balance():
            logger.warning("[PAPER] 지정가 등록 거부: 예약가능 잔고 부족 (필요=%.2f 가용=%.2f)",
                           need, self.available_balance())
            return None
        expiry = _bar_open(pt, bar_minutes) + timedelta(minutes=bar_minutes * (expiry_bars + 1))
        oid = _generate_position_id()
        self.conn.execute(
            "INSERT INTO pending_orders (id, symbol, direction, limit_price, qty, "
            "stop_loss, take_profit, leverage, place_time, expiry_time, entry_score, "
            "entry_session, entry_checks_json, entry_rr, status, signal_key, decision_bar, "
            "bar_minutes, expiry_bars) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, symbol, direction, round(limit_price, 8), qty, stop_loss, take_profit,
             leverage, pt.isoformat(), expiry.isoformat(), score, entry_session,
             json.dumps(checks) if checks else None, entry_rr, "pending", signal_key,
             decision_bar.isoformat() if decision_bar else None, bar_minutes, expiry_bars),
        )
        self._commit()
        logger.info("[PAPER] 지정가 대기: %s %s limit=%.4f qty=%.4f 창=%d봉(~%s) key=%s",
                    symbol, direction, limit_price, qty, expiry_bars, expiry.isoformat(), signal_key)
        return oid

    def get_pending_orders(self, symbol: str | None = None) -> list[dict]:
        """미체결(status=pending) 지정가 주문 목록 (dict 리스트). 구버전 행(status NULL)도 pending."""
        sql = "SELECT * FROM pending_orders WHERE COALESCE(status,'pending')='pending'"
        args: tuple = ()
        if symbol:
            sql += " AND symbol=?"
            args = (symbol,)
        cur = self.conn.execute(sql + " ORDER BY place_time", args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_order_history(self, symbol: str | None = None) -> list[dict]:
        """주문 원장 전체(모든 상태) — 체결률/재주문 감사용."""
        sql = "SELECT * FROM pending_orders"
        args: tuple = ()
        if symbol:
            sql += " WHERE symbol=?"
            args = (symbol,)
        cur = self.conn.execute(sql + " ORDER BY place_time", args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _resolve_pending(
        self, order_id: str, status: str, reason: str,
        position_id: str | None = None, when: datetime | None = None,
    ) -> None:
        """미체결 주문을 종결 상태(filled/expired/cancelled)로 전이한다 (행은 이력으로 보존)."""
        self.conn.execute(
            "UPDATE pending_orders SET status=?, resolve_reason=?, resolved_time=?, position_id=? "
            "WHERE id=?",
            (status, reason, (when or datetime.now(timezone.utc)).isoformat(), position_id, order_id),
        )
        self._commit()

    def cancel_pending(self, order_id: str, reason: str = "manual") -> bool:
        """미체결 지정가 취소. 이미 종결된 주문이면 False."""
        row = self.conn.execute(
            "SELECT id FROM pending_orders WHERE id=? AND COALESCE(status,'pending')='pending'",
            (order_id,),
        ).fetchone()
        if row is None:
            return False
        self._resolve_pending(order_id, "cancelled", reason)
        return True

    def cancel_all_pending(self, reason: str) -> int:
        """모든 미체결 지정가를 취소한다 (예: 실행모드 maker→taker 전환 시). 취소 건수 반환."""
        n = 0
        for od in self.get_pending_orders():
            self._resolve_pending(od["id"], "cancelled", reason)
            n += 1
        if n:
            logger.info("[PAPER] 미체결 지정가 %d건 일괄 취소 (%s)", n, reason)
        return n

    def _prepare_frame(self, symbol: str, candles: pd.DataFrame, what: str) -> pd.DataFrame | None:
        """판정용 프레임 정규화. 이상/빈 프레임이면 미완(+재조회) 등록 후 None (fail-closed)."""
        try:
            c = _ensure_utc_index(candles)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("[PAPER] %s 캔들 인덱스 이상 — %s 판정 보류(fail-closed): %s", symbol, what, e)
            self.last_eval_incomplete.add(symbol)
            return None
        if len(c) == 0:
            logger.warning("[PAPER] %s 캔들 프레임 비어 있음 — %s 판정 보류(재조회 필요)", symbol, what)
            self.last_eval_incomplete.add(symbol)
            self.last_history_gaps.add(symbol)
            return None
        return c

    def _scan_fill_window(
        self, symbol: str, od: dict, c: pd.DataFrame, now_ts: pd.Timestamp, allow_holes: bool,
    ) -> dict:
        """미체결 주문 1건의 체결 창을 관측 규약대로 스캔한다 (상태 변경 없음).

        Returns:
            dict(fill_ts, unobserved, leading_gap, strict_hole, holes, last_open, window_closed_at, frame_last)
        """
        place_t = datetime.fromisoformat(od["place_time"])
        bar_min = int(od.get("bar_minutes") or 15)
        n_bars = int(od.get("expiry_bars") or 8)
        limit = float(od["limit_price"])
        direction = od["direction"]
        bar_td = pd.Timedelta(minutes=bar_min)
        first_open = pd.Timestamp(_bar_open(place_t, bar_min)) + bar_td
        last_open = first_open + bar_td * (n_bars - 1)
        res: dict = {
            "fill_ts": None, "unobserved": False, "leading_gap": False, "strict_hole": False,
            "holes": 0, "first_open": first_open, "last_open": last_open,
            "window_closed_at": last_open + bar_td, "frame_last": c.index[-1], "n_bars": n_bars,
        }
        frame_first, frame_last = c.index[0], c.index[-1]
        if frame_first > first_open:
            res["leading_gap"] = True
            return res
        for i in range(n_bars):
            ts = first_open + bar_td * i
            if ts + bar_td > now_ts:
                break               # 아직 닫히지 않은 봉부터는 판정 불가
            if ts > frame_last:
                res["unobserved"] = True   # 지연 프레임: 닫혔어야 할 봉이 프레임에 아직 없음
                break
            if ts not in c.index:
                if not allow_holes:
                    res["unobserved"] = True
                    res["strict_hole"] = True
                    res["hole_ts"] = ts
                    break
                res["holes"] += 1   # 깊은 프레임에서도 없음 = 거래소 무데이터(거래 없음) → 건너뜀
                continue
            row = c.loc[ts]
            hi, lo = float(row["high"]), float(row["low"])
            if not (math.isfinite(hi) and math.isfinite(lo)):
                res["unobserved"] = True
                res["nan_ts"] = ts
                break
            hit = (direction == "long" and lo <= limit) or (direction == "short" and hi >= limit)
            if hit:
                res["fill_ts"] = ts
                break
        return res

    def _register_scan_flags(self, symbol: str, od: dict, r: dict) -> None:
        """스캔 결과의 보류 사유를 레지스트리에 반영하고 로그를 남긴다."""
        if r["leading_gap"]:
            self.last_history_gaps.add(symbol)
            self.last_eval_incomplete.add(symbol)
            logger.warning("[PAPER] %s 지정가 %s: 프레임 시작이 창 시작(%s)보다 늦음 — 판정 보류(재조회 필요)",
                           symbol, od["id"], r["first_open"].isoformat())
            return
        if r["strict_hole"]:
            self.last_history_gaps.add(symbol)
            logger.warning("[PAPER] %s 지정가 %s: 봉 %s 프레임에 없음 — 보류(재조회 필요)",
                           symbol, od["id"], r["hole_ts"].isoformat())
        if r.get("nan_ts") is not None:
            logger.error("[PAPER] %s 봉 %s 고저 NaN — 체결 판정 보류", symbol, r["nan_ts"].isoformat())
        if r["holes"]:
            logger.warning("[PAPER] %s 지정가 %s: 창 안 공백 봉 %d개 (거래소 무데이터로 간주)",
                           symbol, od["id"], r["holes"])
        if r["unobserved"]:
            self.last_eval_incomplete.add(symbol)

    def _post_fill_observed(
        self, symbol: str, c: pd.DataFrame, fill_ts: pd.Timestamp, now_ts: pd.Timestamp,
        bar_minutes: int, allow_holes: bool,
    ) -> bool:
        """체결봉부터 마지막 닫힌 봉까지가 관측됐는지(체결 후 SL/TP 판정이 끝까지 가능한지) 확인한다.

        미관측(지연/엄격 빈 봉/NaN)이면 레지스트리에 등록하고 False. 봇은 이를 근거로 체결 전에
        '이 체결 뒤 손실이 한도를 넘겼을지 알 수 없는' 상황을 미리 보류한다.
        """
        bar_td = pd.Timedelta(minutes=bar_minutes)
        last_closed_bar = pd.Timestamp(_bar_open(now_ts.to_pydatetime(), bar_minutes)) - bar_td
        frame_last = c.index[-1]
        ts = fill_ts
        while ts <= last_closed_bar:
            if ts > frame_last:
                self.last_eval_incomplete.add(symbol)
                return False
            if ts not in c.index:
                if not allow_holes:
                    self.last_eval_incomplete.add(symbol)
                    self.last_history_gaps.add(symbol)
                    logger.warning("[PAPER] %s 체결 후 구간 봉 %s 프레임에 없음 — 보류(재조회 필요)",
                                   symbol, ts.isoformat())
                    return False
                ts += bar_td
                continue
            row = c.loc[ts]
            if not (math.isfinite(float(row["high"])) and math.isfinite(float(row["low"]))):
                self.last_eval_incomplete.add(symbol)
                return False
            ts += bar_td
        return True

    def peek_pending_fills(
        self, symbol: str, candles: pd.DataFrame, now: datetime | None = None,
        allow_holes: bool = False,
    ) -> list[tuple[str, datetime | None]]:
        """체결 판정을 **실행하지 않고** 각 미체결 주문의 예상 체결봉 시작시각만 돌려준다 (시간순 정렬용).

        보류 사유(갭/빈 봉/NaN/지연/빈 프레임)는 check_pending_fills 와 동일하게 레지스트리에 등록하고,
        체결이 예상되면 체결봉 이후 마지막 닫힌 봉까지의 관측 여부도 미리 확인해 미관측이면 등록한다.

        Returns:
            [(주문 id, 체결봉 시작시각 또는 None)]
        """
        now = now or datetime.now(timezone.utc)
        orders = self.get_pending_orders(symbol)
        if not orders:
            return []
        c = self._prepare_frame(symbol, candles, "체결")
        if c is None:
            return [(od["id"], None) for od in orders]
        now_ts = pd.Timestamp(now)
        out: list[tuple[str, datetime | None]] = []
        for od in orders:
            r = self._scan_fill_window(symbol, od, c, now_ts, allow_holes)
            self._register_scan_flags(symbol, od, r)
            if r["fill_ts"] is not None:
                self._post_fill_observed(symbol, c, r["fill_ts"], now_ts,
                                         int(od.get("bar_minutes") or 15), allow_holes)
            out.append((od["id"], r["fill_ts"].to_pydatetime() if r["fill_ts"] is not None else None))
        return out

    def check_pending_fills(
        self, symbol: str, candles: pd.DataFrame, now: datetime | None = None,
        allow_holes: bool = False,
    ) -> list[Position]:
        """심볼의 미체결 지정가를 **닫힌 봉** 기준으로 체결/만료 판정한다.

        - 체결 창: 주문 시각이 속한 봉의 다음 봉부터 expiry_bars개 봉. 봉은 `시작+bar_minutes <= now`
          일 때만(완전히 닫힘) 판정에 쓴다. 주문 이전 봉·형성 중 봉은 절대 쓰지 않는다.
        - 롱: 봉 저가 ≤ 지정가 / 숏: 봉 고가 ≥ 지정가 → 지정가 정확체결(슬리피지 0, 메이커 수수료).
          체결봉 시작시각을 진입시각으로 기록한다(펀딩 계산에 보수적). 체결봉 내 SL/TP는
          check_stops_history 가 '진입봉 규칙(SL만 인정)'으로 처리한다.
        - 관측 규약: 프레임 첫 봉이 창 첫 봉보다 늦으면(프레임이 창 시작을 못 덮음) 판정 보류 +
          `last_history_gaps`에 심볼 등록. 창 안 기대 봉이 프레임 범위 안에서 비어 있으면
          allow_holes=False(기본)일 땐 미관측으로 보류+등록하고, allow_holes=True(봇이 깊은 프레임으로
          재확인한 경우)일 땐 거래소 무데이터(거래 없음)로 건너뛴다. 고저가 NaN이면 관측 실패로 보고
          보류한다. 만료는 창의 마지막 봉까지 닫혔고(now ≥ 창 종료) 프레임이 창 끝까지 닿았으며 관측
          실패가 없을 때만 선언한다. 어떤 보류든 `last_eval_incomplete`에 심볼을 등록한다.
        - 캔들 인덱스가 비정상이면 fail-closed: 체결도 만료도 하지 않고 에러 로그만 남긴다.
        - 체결 = 포지션 생성 + 주문 종결을 단일 트랜잭션으로 묶는다(포지션 id = 주문 id). 체결 목록엔
          커밋이 끝난 뒤에만 추가한다.

        Args:
            symbol: 심볼
            candles: DatetimeIndex(UTC) + high/low 컬럼의 최근 15m 프레임
            now: 현재 시각 (닫힌 봉/만료 판정 기준, 기본 UTC now)
            allow_holes: 프레임 범위 안의 빈 봉을 거래소 무데이터로 인정할지 (깊은 프레임 재확인 시에만 True)

        Returns:
            이번 호출에서 체결된 Position 리스트
        """
        now = now or datetime.now(timezone.utc)
        orders = self.get_pending_orders(symbol)
        if not orders:
            return []
        c = self._prepare_frame(symbol, candles, "체결/만료")
        if c is None:
            return []
        now_ts = pd.Timestamp(now)
        filled: list[Position] = []
        for od in orders:
            r = self._scan_fill_window(symbol, od, c, now_ts, allow_holes)
            self._register_scan_flags(symbol, od, r)
            if r["leading_gap"]:
                continue
            limit = float(od["limit_price"])
            direction = od["direction"]
            if r["fill_ts"] is not None:
                checks = json.loads(od["entry_checks_json"]) if od["entry_checks_json"] else None
                pos: Position | None = None
                with self._transaction():
                    pos = self.open_position(
                        symbol=symbol, direction=direction, entry_price=limit, qty=float(od["qty"]),
                        stop_loss=float(od["stop_loss"]), take_profit=float(od["take_profit"]),
                        leverage=float(od["leverage"] or 1.0), score=od["entry_score"], checks=checks,
                        entry_rr=od["entry_rr"], entry_session=od["entry_session"],
                        entry_time=r["fill_ts"].to_pydatetime(), is_maker=True, position_id=od["id"],
                    )
                    if pos is not None:
                        self._resolve_pending(od["id"], "filled", "touched", position_id=pos.id, when=now)
                    else:
                        self._resolve_pending(od["id"], "cancelled", "insufficient_balance", when=now)
                        logger.warning("[PAPER] %s 지정가 터치했으나 잔고 부족 → 취소 (id=%s)", symbol, od["id"])
                if pos is not None:
                    filled.append(pos)      # 커밋 완료 후에만 노출
            elif now_ts >= r["window_closed_at"]:
                if r["unobserved"] or r["frame_last"] < r["last_open"]:
                    logger.warning("[PAPER] %s 지정가 %s: 창 종료했으나 미관측 봉 있음 — 만료 보류",
                                   symbol, od["id"])
                    self.last_eval_incomplete.add(symbol)
                    continue
                self._resolve_pending(od["id"], "expired", "window_elapsed", when=now)
                logger.info("[PAPER] 지정가 만료: %s %s limit=%.4f (창 %d봉 미터치)",
                            symbol, direction, limit, r["n_bars"])
        return filled

    def close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str = "",
        qty: float | None = None,
        exit_time: datetime | None = None,
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
        # 백테스트는 exit_time(시뮬레이션 시각)을 넘겨야 펀딩이 올바르게 계산됨
        closed_at = exit_time or datetime.now(timezone.utc)
        funding_n = _funding_events(position.entry_time, closed_at)
        funding_cost = round(
            funding_n * FUNDING_RATE * position.entry_price * close_qty, 8
        )

        # 잔고 반영분: 진입 수수료는 진입 시 이미 차감됐으므로 여기선 제외(이중 차감 방지)
        cash_pnl = round(gross_pnl - exit_fee - funding_cost, 8)
        # 기록 PnL: 진입 수수료(청산 비율만큼)까지 포함한 진짜 순손익 — 승률/PF/R/서킷브레이커 기준
        entry_fee_portion = round((position.entry_fee or 0.0) * close_ratio, 8)
        pnl = round(cash_pnl - entry_fee_portion, 8)
        released_margin = round(position.margin * close_ratio, 8)
        pnl_pct = pnl / released_margin if released_margin > 0 else 0.0

        # 담보금 반환 + 잔고 반영 손익
        with self._transaction():   # 잔고 반영 + 거래 기록 + 포지션 제거/갱신을 원자적으로 (콜백은 커밋 후)
            self._balance = round(self._balance + released_margin + cash_pnl, 8)
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

            now = closed_at.isoformat()
            trade_id = f"{position.id}-{now}" if is_partial else position.id
            self.conn.execute(
                """INSERT INTO trades
                   (id, symbol, direction, entry_price, exit_price,
                    qty, pnl, pnl_pct, margin, entry_time, exit_time, status,
                    entry_score, entry_session, c_trend, c_zone, c_kill_zone,
                    c_ote, c_volume, c_rr, entry_rr, risk_amount, r_multiple,
                    funding_cost, entry_fee, is_maker)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    entry_fee_portion,
                    int(bool(position.is_maker)),
                ),
            )
            self._commit()

            if is_partial:
                # 부분 청산: 잔여 수량/마진/리스크금액/진입수수료 비례 감소
                position.qty = round(position.qty - close_qty, 8)
                position.margin = round(position.margin - released_margin, 8)
                if position.risk_amount is not None and risk_portion is not None:
                    position.risk_amount = round(position.risk_amount - risk_portion, 8)
                position.entry_fee = round((position.entry_fee or 0.0) - entry_fee_portion, 8)
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
        self,
        symbol: str,
        current_high: float,
        current_low: float,
        current_time: datetime | None = None,
    ) -> None:
        """
        SL/TP 자동 체크 — 해당 심볼의 모든 포지션 검사.

        Args:
            symbol: 거래 심볼
            current_high: 현재 캔들 고가
            current_low: 현재 캔들 저가
            current_time: 캔들 시각 (백테스트용 — 펀딩/기록 시각으로 전달)
        """
        for pos in list(self._positions):
            if pos.symbol != symbol:
                continue
            if pos.direction == "long":
                if current_low <= pos.stop_loss:
                    self.close_position(pos, pos.stop_loss, "SL", exit_time=current_time)
                elif current_high >= pos.take_profit:
                    self.close_position(pos, pos.take_profit, "TP", exit_time=current_time)
            else:
                if current_high >= pos.stop_loss:
                    self.close_position(pos, pos.stop_loss, "SL", exit_time=current_time)
                elif current_low <= pos.take_profit:
                    self.close_position(pos, pos.take_profit, "TP", exit_time=current_time)

    def _check_stop_bar(
        self, pos: Position, high: float, low: float, exit_time: datetime, allow_tp: bool,
    ) -> bool:
        """봉 1개에 대해 포지션 SL/TP 판정 (SL 우선). allow_tp=False면 TP 불인정(진입봉 규칙).

        Returns:
            청산했으면 True
        """
        if pos.direction == "long":
            if low <= pos.stop_loss:
                self.close_position(pos, pos.stop_loss, "SL", exit_time=exit_time)
                return True
            if allow_tp and high >= pos.take_profit:
                self.close_position(pos, pos.take_profit, "TP", exit_time=exit_time)
                return True
        else:
            if high >= pos.stop_loss:
                self.close_position(pos, pos.stop_loss, "SL", exit_time=exit_time)
                return True
            if allow_tp and low <= pos.take_profit:
                self.close_position(pos, pos.take_profit, "TP", exit_time=exit_time)
                return True
        return False

    def check_stops_history(
        self, symbol: str, candles: pd.DataFrame,
        now: datetime | None = None, bar_minutes: int = 15,
        allow_holes: bool = False,
    ) -> None:
        """심볼 보유 포지션의 SL/TP를 **진입 이후 모든 봉을 시간순으로** 판정한다 (사이클 간 누락 방지).

        규칙(연구 study/exec_a1 재생과 동일 의미, 일부는 더 보수적):
        - 진입봉(진입시각이 속한 봉): SL만 인정, TP 불인정 — 봉 내부에서 진입 전/후 순서를 알 수 없음.
          (시장가 진입도 동일: 진입 전 저점이 SL을 건드렸을 가능성은 있으나, 연구 재생은 진입봉을
          SL·TP 모두 판정하므로 'SL만'은 연구보다 보수적이다. 진입봉을 통째로 건너뛰면 진입 후 실제
          터치를 놓쳐 낙관 편향이 생기므로 택하지 않는다.)
        - 이후 봉: SL 우선, 그다음 TP. 닫힌 봉은 판정 후 last_checked_bar로 표시해 재평가하지 않는다.
        - 형성 중 봉은 판정하지 않는다(관측 기준 시각 `now` 기준으로 닫힌 봉만). 닫힌 뒤 다음 사이클에 판정.
        - 관측 규약: 판정 시작봉(마지막 판정봉 다음 / 진입봉)이 프레임 첫 봉보다 앞이면 판정 보류 +
          `last_history_gaps` 등록(봇이 재조회). 프레임 범위 안의 공백 봉은 allow_holes=False(기본)면
          미관측으로 보류+등록하고, allow_holes=True(깊은 프레임 재확인)면 거래소 무데이터로 건너뛰되
          닫힌 봉이면 판정 완료로 표시한다. 프레임 끝 이후(지연 프레임)는 거기서 멈춘다. NaN 고저는
          관측 실패로 보고 그 봉부터 보류한다.
        - 청산 시각 = 봉 마감 (진입시각보다 이르지 않게). 캔들 인덱스 이상 시 fail-closed.

        Args:
            symbol: 심볼
            candles: DatetimeIndex(UTC) + high/low 컬럼의 최근 15m 프레임
            now: 현재 시각 (기본 UTC now)
            bar_minutes: 봉 길이(분)
            allow_holes: 프레임 범위 안의 빈 봉을 거래소 무데이터로 인정할지 (깊은 프레임 재확인 시에만 True)
        """
        targets = [p for p in self._positions if p.symbol == symbol]
        if not targets:
            return
        now = now or datetime.now(timezone.utc)
        try:
            c = _ensure_utc_index(candles)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("[PAPER] %s 캔들 인덱스 이상 — SL/TP 판정 보류(fail-closed): %s", symbol, e)
            self.last_eval_incomplete.add(symbol)
            return
        if len(c) == 0:
            logger.warning("[PAPER] %s 캔들 프레임 비어 있음 — SL/TP 판정 보류(재조회 필요)", symbol)
            self.last_eval_incomplete.add(symbol)
            self.last_history_gaps.add(symbol)
            return
        now_ts = pd.Timestamp(now)
        bar_td = pd.Timedelta(minutes=bar_minutes)
        frame_first, frame_last = c.index[0], c.index[-1]
        cur_bar = pd.Timestamp(_bar_open(now, bar_minutes))     # 형성 중 봉 시작
        last_closed_bar = cur_bar - bar_td                      # 가장 최근 '닫힌' 봉 시작
        for pos in targets:
            entry_bar = pd.Timestamp(_bar_open(pos.entry_time, bar_minutes))
            start = (pd.Timestamp(pos.last_checked_bar) + bar_td) if pos.last_checked_bar is not None \
                else entry_bar
            if start > cur_bar:
                continue                # 판정할 새 봉 없음
            if frame_first > start:
                self.last_history_gaps.add(symbol)
                self.last_eval_incomplete.add(symbol)
                logger.warning("[PAPER] %s 포지션 %s: 프레임 시작(%s) > 판정 시작봉(%s) — 보류(재조회 필요)",
                               symbol, pos.id, frame_first.isoformat(), start.isoformat())
                continue
            last_closed: datetime | None = None
            exited = False
            reached = start > last_closed_bar       # 닫힌 봉이 더 없으면 이미 최신
            ts = start
            while ts <= last_closed_bar:            # 닫힌 봉만 (형성 중 봉은 다음 사이클)
                if ts > frame_last:
                    break               # 지연 프레임: 이후 봉은 아직 못 봄
                is_closed = (ts + bar_td) <= now_ts
                if ts not in c.index:
                    if not allow_holes:
                        # 일시적 응답 누락일 수 있으므로 미관측으로 보류 — 표시를 진행하지 않는다 (깊은 프레임으로 재확인)
                        self.last_history_gaps.add(symbol)
                        logger.warning("[PAPER] %s 포지션 %s: 봉 %s 프레임에 없음 — 보류(재조회 필요)",
                                       symbol, pos.id, ts.isoformat())
                        break
                    # 깊은 프레임에서도 없음 = 거래소 무데이터(거래 없음) → 판정할 가격 없음, 닫힌 봉이면 완료 표시
                    logger.warning("[PAPER] %s 포지션 %s: 봉 %s 프레임에 없음(무데이터로 간주)",
                                   symbol, pos.id, ts.isoformat())
                    if is_closed:
                        last_closed = ts.to_pydatetime()
                        if ts == last_closed_bar:
                            reached = True
                    ts += bar_td
                    continue
                row = c.loc[ts]
                hi, lo = float(row["high"]), float(row["low"])
                if not (math.isfinite(hi) and math.isfinite(lo)):
                    logger.error("[PAPER] %s 포지션 %s: 봉 %s 고저 NaN — 이 봉부터 판정 보류",
                                 symbol, pos.id, ts.isoformat())
                    break
                exit_time = (ts + bar_td).to_pydatetime()
                if exit_time < pos.entry_time:
                    exit_time = pos.entry_time
                exited = self._check_stop_bar(pos, hi, lo, exit_time, allow_tp=(ts != entry_bar))
                if exited:
                    break
                last_closed = ts.to_pydatetime()
                if ts == last_closed_bar:
                    reached = True
                ts += bar_td
            if not exited and last_closed is not None and last_closed != pos.last_checked_bar:
                pos.last_checked_bar = last_closed
                self._save_position(pos)
            if not exited and not reached:
                # 마지막 닫힌 봉까지 판정을 끝내지 못함(지연 프레임/NaN/빈 봉) — 이번 사이클 '미완'
                self.last_eval_incomplete.add(symbol)

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
        if since:
            rows = self.conn.execute(
                "SELECT pnl, pnl_pct FROM trades WHERE exit_time >= ?", (since,)
            ).fetchall()
        else:
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
