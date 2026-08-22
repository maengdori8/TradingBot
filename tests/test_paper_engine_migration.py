"""paper_engine DB 마이그레이션 멱등성 + 진입조건 영속화 하위호환 테스트"""
from __future__ import annotations

import sqlite3

import pytest

import src.paper_trading.paper_engine as pe_module
from src.paper_trading.paper_engine import PaperEngine, _init_db, _migrate_columns


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "paper.db"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pe_module, "DB_PATH", db)
        yield PaperEngine(initial_balance=1000.0, db_path=db)


def _columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


class TestMigration:
    def test_new_columns_present(self, engine):
        cols = _columns(engine.conn, "trades")
        for c in ["entry_score", "entry_session", "c_trend", "c_zone",
                  "c_kill_zone", "c_ote", "c_volume", "c_rr",
                  "entry_rr", "risk_amount", "r_multiple"]:
            assert c in cols

    def test_open_positions_columns(self, engine):
        cols = _columns(engine.conn, "open_positions")
        for c in ["entry_score", "entry_session", "entry_checks_json",
                  "entry_rr", "risk_amount"]:
            assert c in cols

    def test_migration_idempotent(self, tmp_path):
        """_init_db / _migrate_columns 반복 실행해도 오류 없음."""
        db = tmp_path / "idem.db"
        conn = _init_db(db)
        # 두 번 더 실행
        _migrate_columns(conn, "trades", {"entry_score": "REAL", "r_multiple": "REAL"})
        _migrate_columns(conn, "trades", {"entry_score": "REAL"})
        conn.close()
        # 재오픈도 정상
        conn2 = _init_db(db)
        assert "entry_score" in _columns(conn2, "trades")
        conn2.close()

    def test_legacy_db_migrated(self, tmp_path):
        """구버전 스키마(신규 컬럼 없음) DB도 컬럼 추가됨, 기존 행 보존."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""CREATE TABLE trades (
            id TEXT PRIMARY KEY, symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, qty REAL, pnl REAL, pnl_pct REAL,
            margin REAL, entry_time TEXT, exit_time TEXT, status TEXT)""")
        conn.execute("INSERT INTO trades (id, symbol, pnl, status) VALUES ('x','BTC',5.0,'TP')")
        conn.commit()
        conn.close()
        # _init_db가 마이그레이션
        conn2 = _init_db(db)
        cols = _columns(conn2, "trades")
        assert "r_multiple" in cols
        row = conn2.execute("SELECT pnl, r_multiple FROM trades WHERE id='x'").fetchone()
        assert row[0] == 5.0
        assert row[1] is None  # 신규 컬럼은 NULL
        conn2.close()


class TestBackwardCompat:
    def test_open_position_without_context(self, engine):
        """진입조건 인자 없이 호출해도 동작 (하위호환)."""
        pos = engine.open_position("BTC/USDT", "long", 50000, 0.01, 49000, 52000)
        assert pos is not None
        assert pos.entry_score is None
        assert pos.entry_checks is None
        assert pos.risk_amount is not None  # 자동 계산됨

    def test_entry_context_roundtrip(self, engine):
        """진입조건이 저장되고 trades에 기록됨."""
        checks = {"trend": True, "zone": True, "kill_zone": True,
                  "ote": True, "volume": True, "rr": True}
        pos = engine.open_position(
            "BTC/USDT", "long", 50000, 0.01, 49000, 52000,
            score=82.0, checks=checks, entry_rr=2.0, entry_session="london",
        )
        assert pos.entry_score == 82.0
        assert pos.entry_session == "london"
        engine.close_position(pos, 52000, "TP")
        row = engine.conn.execute(
            "SELECT entry_score, entry_session, c_trend, c_rr, r_multiple "
            "FROM trades WHERE id=?", (pos.id,)
        ).fetchone()
        assert row[0] == 82.0
        assert row[1] == "london"
        assert row[2] == 1            # c_trend
        assert row[3] == 1            # c_rr
        assert row[4] is not None     # r_multiple 계산됨

    def test_r_multiple_sign(self, engine):
        """이익 거래는 r_multiple>0, 손실은 <0."""
        pos = engine.open_position(
            "BTC/USDT", "long", 50000, 0.01, 49000, 52000,
            score=80, checks={"trend": True}, entry_rr=2.0,
        )
        engine.close_position(pos, 52000, "TP")
        r = engine.conn.execute(
            "SELECT r_multiple FROM trades WHERE id=?", (pos.id,)
        ).fetchone()[0]
        assert r > 0

    def test_restored_position_close_safe(self, engine, tmp_path):
        """복원된 포지션 청산 시 진입조건 유지."""
        checks = {"trend": True, "zone": False}
        pos = engine.open_position(
            "ETH/USDT", "short", 3000, 0.1, 3100, 2800,
            score=71, checks=checks, entry_rr=2.0, entry_session="newyork",
        )
        pid = pos.id
        engine.conn.close()
        # 새 엔진으로 복원
        eng2 = PaperEngine(initial_balance=1000.0, db_path=tmp_path / "paper.db")
        restored = next(p for p in eng2.get_positions() if p.id == pid)
        assert restored.entry_score == 71
        assert restored.entry_session == "newyork"
        assert restored.entry_checks == checks
        assert restored.risk_amount is not None



class TestTrueLegacySchemas:
    """실제 과거 스키마로 만든 DB를 현재 엔진이 깨지지 않고 열고 보정하는지 (CI 2026-08-22 장애 재현)."""

    def test_oldest_schema_int_id_no_margin(self, tmp_path):
        """2026-05-22 최초 스키마: trades.id INTEGER AUTOINCREMENT, margin 없음, 다른 테이블 없음."""
        import sqlite3
        from datetime import datetime, timezone, timedelta
        from src.paper_trading.paper_engine import PaperEngine, TAKER_FEE
        db = tmp_path / "legacy.db"
        c = sqlite3.connect(db)
        c.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, qty REAL, pnl REAL, pnl_pct REAL,
            entry_time TEXT, exit_time TEXT, status TEXT)""")
        t0 = datetime(2026, 5, 22, tzinfo=timezone.utc)
        c.execute("INSERT INTO trades (symbol, direction, entry_price, exit_price, qty, pnl, pnl_pct, "
                  "entry_time, exit_time, status) VALUES ('BTC/USDT:USDT','long',100.0,105.0,2.0,9.8,0.049,?,?,'TP')",
                  (t0.isoformat(), (t0 + timedelta(hours=1)).isoformat()))
        c.commit()
        c.close()

        eng = PaperEngine(initial_balance=1000.0, db_path=db)      # 예전엔 여기서 'no such column: margin'
        info = {r[1]: r[2] for r in eng.conn.execute("PRAGMA table_info(trades)")}
        assert info["id"].upper() == "TEXT" and "margin" in info and "entry_fee" in info
        row = eng.conn.execute("SELECT id, entry_fee, pnl, is_maker FROM trades").fetchone()
        assert row[0] == "1"                                        # 정수 id → 텍스트로 보존
        fee = round(100.0 * 2.0 * TAKER_FEE, 8)
        assert row[1] == pytest.approx(fee, abs=1e-9) and row[2] == pytest.approx(9.8 - fee, abs=1e-9)
        assert row[3] == 0
        # 새 거래 기록(TEXT id)이 실패하지 않는다
        pos = eng.open_position("ETH/USDT:USDT", "long", 100.0, 1.0, 98.0, 105.0)
        eng.close_position(pos, 101.0, "manual")
        assert eng.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 2
        # 멱등: 다시 열어도 그대로
        eng.conn.close()
        eng2 = PaperEngine(initial_balance=1000.0, db_path=db)
        assert eng2.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 2
        assert not eng2.conn.execute("SELECT name FROM sqlite_master WHERE name='trades_legacy_intid'").fetchone()

    def test_b15a304_schema_text_id_with_margin(self, tmp_path):
        """2026-05-27 스키마: TEXT id + margin, 학습 컬럼/entry_fee 없음, open_positions 기본 컬럼만."""
        import sqlite3
        from datetime import datetime, timezone
        from src.paper_trading.paper_engine import PaperEngine, TAKER_FEE
        db = tmp_path / "legacy2.db"
        c = sqlite3.connect(db)
        c.execute("""CREATE TABLE trades (id TEXT PRIMARY KEY, symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, qty REAL, pnl REAL, pnl_pct REAL, margin REAL,
            entry_time TEXT, exit_time TEXT, status TEXT)""")
        c.execute("CREATE TABLE engine_state (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("""CREATE TABLE open_positions (id TEXT PRIMARY KEY, symbol TEXT, direction TEXT,
            entry_price REAL, qty REAL, stop_loss REAL, take_profit REAL, margin REAL, entry_time TEXT)""")
        c.execute("INSERT INTO engine_state VALUES ('balance', '900.0')")
        c.execute("INSERT INTO open_positions VALUES ('p1','BTC/USDT:USDT','long',100.0,1.0,98.0,105.0,100.0,?)",
                  (datetime(2026, 5, 27, tzinfo=timezone.utc).isoformat(),))
        c.commit()
        c.close()
        eng = PaperEngine(initial_balance=1000.0, db_path=db)
        assert eng.balance == 900.0
        p1 = [p for p in eng.get_positions() if p.id == "p1"][0]
        assert p1.entry_fee == pytest.approx(100.0 * 1.0 * TAKER_FEE, abs=1e-9) and p1.is_maker is False
        pnl = eng.close_position(p1, 105.0, "TP")
        row = eng.conn.execute("SELECT pnl, entry_fee FROM trades WHERE id='p1'").fetchone()
        assert row[0] == pytest.approx(pnl, abs=1e-9) and row[1] > 0
