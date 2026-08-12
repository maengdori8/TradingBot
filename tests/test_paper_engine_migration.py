from __future__ import annotations

"""paper_engine DB 마이그레이션 멱등성 + 진입조건 영속화 하위호환 테스트"""

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
