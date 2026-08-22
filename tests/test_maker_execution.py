"""메이커 지정가 실행 + 정직한 SL/TP 판정 단위테스트.

검증 범위:
- 지정가 정확체결(슬리피지 0) + 메이커 수수료, 테이커 대비 왕복 비용 감소
- 체결 판정은 **닫힌 봉만**, 주문 이후 N봉 창(연구 exec_a1 FILL_WINDOW와 동일 의미), 주문 이전 봉 제외,
  형성 중 봉 제외, 인덱스 이상 시 fail-closed, 창의 마지막 봉이 닫힌 뒤 만료
- 진입(체결)봉: SL만 인정 / TP 불인정, 이후 봉 SL 우선, 사이클 사이 봉의 SL 누락 없음, last_checked_bar 영속
- 프로세스 재시작(엔진 재생성) 간 미체결/포지션 영속, 체결=원자적(포지션 id=주문 id), 재시작 후 중복 없음
- signal_key 중복 등록 거부, 심볼당 1주문, 모드 전환 일괄 취소, 예약 잔고/가상 포지션
- 기록 PnL에 진입 수수료 포함: 잔고 변화 == 기록 PnL 합(왕복)

실행: pytest tests/test_maker_execution.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from src.paper_trading.paper_engine import PaperEngine, TAKER_FEE

SYM = "BTC/USDT:USDT"
T0 = datetime(2026, 6, 16, 0, 0, tzinfo=timezone.utc)


def _engine(path, balance: float = 10000.0) -> PaperEngine:
    return PaperEngine(initial_balance=balance, db_path=path, maker_fee=0.00035)


def _m(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _candles(rows: list[tuple[datetime, float, float]], tz: str | None = "UTC") -> pd.DataFrame:
    """[(봉 시작시각, high, low)] → DatetimeIndex 프레임 (tz=None이면 naive 인덱스)."""
    idx = pd.DatetimeIndex([r[0].replace(tzinfo=None) if tz is None else r[0] for r in rows])
    if tz is not None:
        idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return pd.DataFrame({"high": [r[1] for r in rows], "low": [r[2] for r in rows]}, index=idx)


def _place_long(eng: PaperEngine, **kw) -> str | None:
    base = dict(symbol=SYM, direction="long", limit_price=99.0, qty=1.0,
                stop_loss=97.0, take_profit=103.0, expiry_bars=8, place_time=_m(3))
    base.update(kw)
    return eng.place_pending_limit(**base)


def _closed_trades(eng: PaperEngine) -> list[tuple]:
    return eng.conn.execute(
        "SELECT status, pnl, exit_time, entry_fee FROM trades ORDER BY exit_time"
    ).fetchall()


# ────────────────────────────────────────────────────────────────────────
# 1. 수수료/슬리피지 기본
# ────────────────────────────────────────────────────────────────────────

def test_maker_no_slippage_and_fee(tmp_path) -> None:
    """is_maker=True: 진입가=지정가 정확(슬리피지 0), 메이커 수수료, entry_fee 기록."""
    eng = _engine(tmp_path / "t.db")
    b0 = eng.balance
    pos = eng.open_position(SYM, "long", entry_price=100.0, qty=1.0,
                            stop_loss=98.0, take_profit=105.0, is_maker=True)
    assert pos is not None
    assert pos.entry_price == 100.0
    assert pos.is_maker is True
    assert pos.entry_fee == pytest.approx(100.0 * 0.00035, abs=1e-9)
    assert b0 - eng.balance == pytest.approx(100.0 + 100.0 * 0.00035, abs=1e-6)


def test_taker_has_slippage(tmp_path) -> None:
    """is_maker=False: 불리한 슬리피지 + 테이커 수수료 (기존 동작 유지)."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position("ETH/USDT:USDT", "long", entry_price=100.0, qty=1.0,
                            stop_loss=98.0, take_profit=105.0, is_maker=False)
    assert pos.entry_price == pytest.approx(100.0 * 1.0005, abs=1e-6)
    assert pos.is_maker is False
    assert pos.entry_fee == pytest.approx(pos.entry_price * TAKER_FEE, abs=1e-9)


def test_roundtrip_balance_change_equals_recorded_pnl(tmp_path) -> None:
    """기록 PnL에 진입 수수료 포함 → 왕복 후 잔고 변화 == trades.pnl (메이커·테이커 모두)."""
    for is_maker in (True, False):
        eng = _engine(tmp_path / f"rt_{int(is_maker)}.db")
        b0 = eng.balance
        pos = eng.open_position(SYM, "long", 100.0, 1.0, 98.0, 105.0,
                                is_maker=is_maker, entry_time=_m(0))
        pnl = eng.close_position(pos, 105.0, "TP", exit_time=_m(30))
        row = eng.conn.execute("SELECT pnl, entry_fee FROM trades").fetchone()
        assert row[0] == pytest.approx(pnl, abs=1e-9)
        assert row[1] == pytest.approx(pos.entry_fee, abs=1e-9)
        assert eng.balance - b0 == pytest.approx(pnl, abs=1e-6), f"maker={is_maker}"


def test_maker_cheaper_than_taker_roundtrip(tmp_path) -> None:
    """동일 진입/청산 왕복에서 메이커 기록 PnL > 테이커 기록 PnL (출혈 감소)."""
    e_m = _engine(tmp_path / "m.db")
    e_t = _engine(tmp_path / "t.db")
    pm = e_m.open_position(SYM, "long", 100.0, 1.0, 98.0, 105.0, is_maker=True, entry_time=_m(0))
    pt = e_t.open_position(SYM, "long", 100.0, 1.0, 98.0, 105.0, is_maker=False, entry_time=_m(0))
    pnl_m = e_m.close_position(pm, 101.0, "manual", exit_time=_m(30))
    pnl_t = e_t.close_position(pt, 101.0, "manual", exit_time=_m(30))
    assert pnl_m > pnl_t
    assert TAKER_FEE > 0.00035


def test_partial_close_splits_entry_fee(tmp_path) -> None:
    """부분 청산 시 진입 수수료도 비례 배분되고 합계가 보존된다."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position(SYM, "long", 100.0, 2.0, 98.0, 105.0, is_maker=True, entry_time=_m(0))
    fee_total = pos.entry_fee
    eng.close_position(pos, 105.0, "TP1", qty=1.0, exit_time=_m(30))
    assert pos.entry_fee == pytest.approx(fee_total / 2, abs=1e-9)
    eng.close_position(pos, 105.0, "TP2", exit_time=_m(45))
    fees = [r[0] for r in eng.conn.execute("SELECT entry_fee FROM trades").fetchall()]
    assert sum(fees) == pytest.approx(fee_total, abs=1e-9)


# ────────────────────────────────────────────────────────────────────────
# 2. 체결 판정 — 닫힌 봉 / 창 / fail-closed
# ────────────────────────────────────────────────────────────────────────

def test_fill_only_when_bar_closed(tmp_path) -> None:
    """주문 00:03 → 첫 대상봉 00:15. 00:20(형성 중)엔 미체결, 00:30(닫힘) 이후 체결."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    candles = _candles([(_m(15), 100.5, 98.5)])
    assert eng.check_pending_fills(SYM, candles, now=_m(20)) == []
    assert len(eng.get_pending_orders(SYM)) == 1
    filled = eng.check_pending_fills(SYM, candles, now=_m(30))
    assert len(filled) == 1
    assert filled[0].entry_price == 99.0
    assert filled[0].entry_time == _m(15)          # 체결봉 시작시각
    assert eng.get_pending_orders(SYM) == []


def test_short_fills_on_high_touch(tmp_path) -> None:
    """숏 지정가는 닫힌 봉 고가가 닿아야 체결."""
    eng = _engine(tmp_path / "t.db")
    eng.place_pending_limit("ETH/USDT:USDT", "short", limit_price=101.0, qty=1.0,
                            stop_loss=103.0, take_profit=97.0, expiry_bars=8, place_time=_m(3))
    candles = _candles([(_m(15), 101.5, 100.0)])
    assert eng.check_pending_fills("ETH/USDT:USDT", candles, now=_m(20)) == []
    filled = eng.check_pending_fills("ETH/USDT:USDT", candles, now=_m(30))
    assert len(filled) == 1 and filled[0].entry_price == 101.0


def test_bars_before_order_are_ignored(tmp_path) -> None:
    """주문 이전 봉이 지정가에 닿았어도 체결 아님 (룩백 체결 금지)."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)                                   # 00:03
    candles = _candles([(_m(-15), 100.0, 95.0),        # 주문 전 봉 — 저가 95 ≤ 99 이지만 무시
                        (_m(0), 100.0, 98.0),          # 주문이 속한 봉 — 무시 (부분 정보)
                        (_m(15), 101.0, 99.5)])        # 첫 대상봉 — 미터치
    assert eng.check_pending_fills(SYM, candles, now=_m(35)) == []
    assert len(eng.get_pending_orders(SYM)) == 1


def test_bar_after_window_does_not_fill(tmp_path) -> None:
    """expiry_bars=2 → 대상봉 00:15, 00:30 뿐. 00:45 봉 터치는 체결 아님, 창 종료(00:45) 후 만료."""
    eng = _engine(tmp_path / "t.db")
    b0 = eng.balance
    _place_long(eng, expiry_bars=2)
    candles = _candles([(_m(15), 101.0, 99.5), (_m(30), 101.0, 99.5), (_m(45), 101.0, 90.0)])
    # 00:40: 00:15 봉만 닫힘(미터치), 00:30 봉은 아직 형성 중 → pending
    assert eng.check_pending_fills(SYM, candles, now=_m(40)) == []
    assert len(eng.get_pending_orders(SYM)) == 1
    # 00:50: 창 종료 → 만료. 00:45 봉의 터치는 무시. 잔고 불변(비용 0)
    assert eng.check_pending_fills(SYM, candles, now=_m(50)) == []
    assert eng.get_pending_orders(SYM) == []
    assert eng.balance == b0
    hist = eng.get_order_history(SYM)
    assert hist[0]["status"] == "expired"


def test_delayed_run_evaluates_whole_window(tmp_path) -> None:
    """실행이 지연돼 창 전체가 한 번에 닫혀 있어도, 창 안 마지막 봉(8번째)의 터치는 정상 체결."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)                                   # 00:03 → 대상봉 00:15 … 02:00 (8개)
    rows = [(_m(15 + 15 * i), 101.0, 99.5) for i in range(7)] + [(_m(120), 101.0, 98.9)]
    filled = eng.check_pending_fills(SYM, _candles(rows), now=_m(138))
    assert len(filled) == 1 and filled[0].entry_time == _m(120)


def test_naive_index_is_treated_as_utc(tmp_path) -> None:
    """tz 없는 인덱스는 UTC로 간주해 정상 판정한다."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    candles = _candles([(_m(15), 100.5, 98.5)], tz=None)
    assert len(eng.check_pending_fills(SYM, candles, now=_m(30))) == 1


def test_bad_index_fails_closed(tmp_path) -> None:
    """DatetimeIndex가 아니면 체결도 만료도 하지 않는다 (fail-closed) — 창이 지났어도."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng, expiry_bars=2)
    bad = pd.DataFrame({"high": [100.0], "low": [90.0]})       # RangeIndex
    assert eng.check_pending_fills(SYM, bad, now=_m(600)) == []
    assert len(eng.get_pending_orders(SYM)) == 1
    # 창 전체(00:15, 00:30)를 관측한 정상 프레임이 오면 그때 판정 (여기선 만료)
    full = _candles([(_m(15), 101.0, 99.5), (_m(30), 101.0, 99.5)])
    assert eng.check_pending_fills(SYM, full, now=_m(600)) == []
    assert eng.get_pending_orders(SYM) == []


# ────────────────────────────────────────────────────────────────────────
# 2b. 관측 규약 — 못 본 봉은 없었던 봉이 아니다
# ────────────────────────────────────────────────────────────────────────

def _window(first_min: int, n: int, hi: float = 101.0, lo: float = 99.5) -> list[tuple]:
    return [(_m(first_min + 15 * i), hi, lo) for i in range(n)]


def test_leading_gap_holds_and_flags_symbol(tmp_path) -> None:
    """프레임이 창 첫 봉(00:15)보다 늦게 시작하면 체결도 만료도 안 하고 재조회 표시만 한다."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng, expiry_bars=2)
    late = _candles([(_m(30), 101.0, 90.0)])           # 00:15 누락, 00:30은 터치
    assert eng.check_pending_fills(SYM, late, now=_m(200)) == []
    assert len(eng.get_pending_orders(SYM)) == 1
    assert SYM in eng.last_history_gaps
    # 창 전체를 덮는 프레임이 오면 정상 판정 — 00:15가 먼저 터치했으면 체결봉은 00:15
    eng.last_history_gaps.discard(SYM)
    full = _candles([(_m(15), 101.0, 98.0), (_m(30), 101.0, 90.0)])
    filled = eng.check_pending_fills(SYM, full, now=_m(200))
    assert len(filled) == 1 and filled[0].entry_time == _m(15)
    assert SYM not in eng.last_history_gaps


def test_stale_frame_holds_expiry(tmp_path) -> None:
    """창이 끝났어도 프레임 끝이 창 마지막 봉(00:30)에 못 미치면(지연 프레임) 만료하지 않는다."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng, expiry_bars=2)
    stale = _candles([(_m(15), 101.0, 99.5)])          # 00:30 아직 없음
    assert eng.check_pending_fills(SYM, stale, now=_m(50)) == []
    assert len(eng.get_pending_orders(SYM)) == 1
    full = _candles(_window(15, 2))
    assert eng.check_pending_fills(SYM, full, now=_m(50)) == []
    assert eng.get_pending_orders(SYM) == []           # 이제 만료


def test_internal_hole_strict_then_deep_frame(tmp_path) -> None:
    """프레임 안 빈 봉(00:30): 기본(allow_holes=False)은 미관측 → 보류+재조회 표시, 체결/만료 없음.
    깊은 프레임 재확인(allow_holes=True)에서도 비어 있으면 무데이터로 건너뛰고 창 관측 완료로 만료 가능."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng, expiry_bars=3)                     # 00:15, 00:30, 00:45
    holey = _candles([(_m(15), 101.0, 99.5), (_m(45), 101.0, 98.0)])   # 00:45 터치
    assert eng.check_pending_fills(SYM, holey, now=_m(70)) == []        # 보류 (00:30 못 봄)
    assert len(eng.get_pending_orders(SYM)) == 1 and SYM in eng.last_history_gaps
    # 깊은 프레임에서 00:30이 나타나고 먼저 터치했으면 체결봉은 00:30
    eng.last_history_gaps.discard(SYM)
    full = _candles([(_m(15), 101.0, 99.5), (_m(30), 101.0, 98.5), (_m(45), 101.0, 98.0)])
    filled = eng.check_pending_fills(SYM, full, now=_m(70), allow_holes=True)
    assert len(filled) == 1 and filled[0].entry_time == _m(30)
    # 깊은 프레임에서도 없으면 무데이터 → 00:45 체결 / 미터치면 만료
    eng2 = _engine(tmp_path / "t2.db")
    _place_long(eng2, expiry_bars=3)
    filled2 = eng2.check_pending_fills(SYM, holey, now=_m(70), allow_holes=True)
    assert len(filled2) == 1 and filled2[0].entry_time == _m(45)
    eng3 = _engine(tmp_path / "t3.db")
    _place_long(eng3, expiry_bars=3)
    nohit = _candles([(_m(15), 101.0, 99.5), (_m(45), 101.0, 99.5)])
    assert eng3.check_pending_fills(SYM, nohit, now=_m(70), allow_holes=True) == []
    assert eng3.get_order_history(SYM)[0]["status"] == "expired"


def test_nan_bar_holds(tmp_path) -> None:
    """고저가 NaN인 봉은 관측 실패 → 체결·만료 보류."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng, expiry_bars=1)
    nan = _candles([(_m(15), float("nan"), float("nan"))])
    assert eng.check_pending_fills(SYM, nan, now=_m(200)) == []
    assert len(eng.get_pending_orders(SYM)) == 1


def test_stops_leading_gap_holds_and_flags(tmp_path) -> None:
    """포지션 판정 시작봉(진입봉 00:00)이 프레임 밖이면 판정 보류 + 재조회 표시, 진행 표시 없음."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    late = _candles([(_m(30), 100.0, 96.0)])           # 00:00, 00:15 누락; 00:30은 SL 터치
    eng.check_stops_history(SYM, late, now=_m(50))
    assert len(eng.get_positions()) == 1 and pos.last_checked_bar is None
    assert SYM in eng.last_history_gaps
    full = _candles([(_m(0), 100.5, 99.5), (_m(15), 100.5, 99.5), (_m(30), 100.0, 96.0)])
    eng.check_stops_history(SYM, full, now=_m(50))
    assert _closed_trades(eng)[0][0] == "SL"


def test_stops_internal_hole_strict_then_deep_frame(tmp_path) -> None:
    """프레임 안 빈 봉(00:15): 기본은 보류+표시(진행 없음), 깊은 프레임(allow_holes=True)에서만 무데이터로 진행."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    holey = _candles([(_m(0), 100.5, 99.5), (_m(30), 104.0, 99.5)])   # 00:15 없음, 00:30은 TP 고가
    eng.check_stops_history(SYM, holey, now=_m(50))
    assert pos.last_checked_bar == _m(0) and SYM in eng.last_history_gaps   # 00:00만 완료, 00:15에서 멈춤
    assert len(eng.get_positions()) == 1                                     # TP 부여 안 함
    # 깊은 프레임에 00:15가 있고 SL 터치였다면 SL (TP 아님)
    eng.last_history_gaps.discard(SYM)
    full = _candles([(_m(0), 100.5, 99.5), (_m(15), 100.0, 96.5), (_m(30), 104.0, 99.5)])
    eng.check_stops_history(SYM, full, now=_m(50), allow_holes=True)
    assert _closed_trades(eng)[0][0] == "SL"
    # 깊은 프레임에서도 없으면 무데이터로 건너뛰고 진행
    eng2 = _engine(tmp_path / "t2.db")
    pos2 = eng2.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    eng2.check_stops_history(SYM, holey, now=_m(50), allow_holes=True)
    t2 = _closed_trades(eng2)
    assert len(t2) == 1 and t2[0][0] == "TP" and datetime.fromisoformat(t2[0][2]) == _m(45)
    assert eng2.get_positions() == [] and pos2 not in eng2.get_positions()


def test_duplicate_timestamps_consolidated_conservatively(tmp_path) -> None:
    """같은 시각 중복 행은 고가=max/저가=min으로 병합 → SL 터치 저가가 버려지지 않는다 (SL 우선)."""
    eng = _engine(tmp_path / "t.db")
    eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    dup = _candles([(_m(0), 100.5, 99.5), (_m(15), 101.0, 96.0), (_m(15), 104.0, 99.0)])
    eng.check_stops_history(SYM, dup, now=_m(40))
    assert _closed_trades(eng)[0][0] == "SL"


def test_required_history_start(tmp_path) -> None:
    """백필 시작점 = 포지션(last_checked_bar 다음 / 진입봉)과 미체결(창 첫 봉) 중 가장 이른 시각."""
    eng = _engine(tmp_path / "t.db")
    assert eng.required_history_start(SYM) is None
    pos = eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(63))
    assert eng.required_history_start(SYM) == _m(60)
    pos.last_checked_bar = _m(90)
    assert eng.required_history_start(SYM) == _m(105)
    eng.place_pending_limit("ETH/USDT:USDT", "long", 99.0, 1.0, 97.0, 103.0, place_time=_m(3))
    assert eng.required_history_start("ETH/USDT:USDT") == _m(15)


def test_stops_stale_frame_stops_at_frame_end(tmp_path) -> None:
    """프레임 끝(00:15) 이후 봉은 아직 못 본 것 — 거기까지만 표시한다."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    eng.check_stops_history(SYM, _candles([(_m(0), 100.5, 99.5), (_m(15), 100.5, 99.5)]), now=_m(100))
    assert pos.last_checked_bar == _m(15)


# ────────────────────────────────────────────────────────────────────────
# 2c. 트랜잭션 — 본문 예외 / commit 실패 모두 롤백 + 메모리 복원
# ────────────────────────────────────────────────────────────────────────

def test_fill_rollback_on_body_exception(tmp_path, monkeypatch) -> None:
    """체결 도중(주문 종결 단계) 예외 → 포지션/잔고/주문 모두 원상복구, 이후 정상 체결 가능."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    b0 = eng.balance
    real_resolve = eng._resolve_pending

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(eng, "_resolve_pending", boom)
    candles = _candles([(_m(15), 100.0, 98.5)])
    with pytest.raises(RuntimeError):
        eng.check_pending_fills(SYM, candles, now=_m(30))
    assert eng.get_positions() == [] and eng.balance == b0
    assert eng.conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
    assert len(eng.get_pending_orders(SYM)) == 1 and not eng.conn.in_transaction
    monkeypatch.setattr(eng, "_resolve_pending", real_resolve)
    assert len(eng.check_pending_fills(SYM, candles, now=_m(30))) == 1


class _CommitFailsOnce:
    """sqlite3.Connection 프록시 — 첫 commit()만 실패시킨다."""

    def __init__(self, conn):
        self._c = conn
        self.failed = False

    def commit(self):
        if not self.failed:
            self.failed = True
            raise sqlite3.OperationalError("database is locked")
        return self._c.commit()

    def __getattr__(self, name):
        return getattr(self._c, name)


def test_fill_rollback_on_commit_failure(tmp_path) -> None:
    """최종 commit 실패도 롤백·메모리 복원되고 예외가 전파된다 (조용한 반영 금지)."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    b0 = eng.balance
    eng.conn = _CommitFailsOnce(eng.conn)
    candles = _candles([(_m(15), 100.0, 98.5)])
    with pytest.raises(sqlite3.OperationalError):
        eng.check_pending_fills(SYM, candles, now=_m(30))
    assert eng.get_positions() == [] and eng.balance == b0
    assert eng.conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0] == 0
    assert len(eng.get_pending_orders(SYM)) == 1
    assert not eng.conn.in_transaction
    # 이후 재시도는 정상 (commit 프록시는 두 번째부터 통과)
    assert len(eng.check_pending_fills(SYM, candles, now=_m(30))) == 1


def test_bootstrap_balance_row_exists(tmp_path) -> None:
    """새 DB라도 엔진 생성 직후 engine_state.balance 기준 행이 있다 (롤백 기준점)."""
    eng = _engine(tmp_path / "t.db", balance=1234.5)
    row = eng.conn.execute("SELECT value FROM engine_state WHERE key='balance'").fetchone()
    assert row is not None and float(row[0]) == 1234.5


# ────────────────────────────────────────────────────────────────────────
# 2d. 레거시 수수료 마이그레이션 (v2)
# ────────────────────────────────────────────────────────────────────────

def test_legacy_fee_migration(tmp_path) -> None:
    """entry_fee NULL 레거시 행 → 테이커 수수료 보정(pnl/pnl_pct/r_multiple 재계산), 멱등."""
    db = tmp_path / "t.db"
    e1 = _engine(db)
    # 레거시 스타일 행 삽입 (entry_fee NULL) + 버전 행 제거 → 다음 오픈 때 마이그레이션 유도
    e1.conn.execute(
        "INSERT INTO trades (id, symbol, direction, entry_price, exit_price, qty, pnl, pnl_pct, margin, "
        "entry_time, exit_time, status, risk_amount, r_multiple) VALUES "
        "('L1', ?, 'long', 100.0, 105.0, 2.0, 9.8, 0.049, 200.0, ?, ?, 'TP', 4.0, 2.45)",
        (SYM, _m(0).isoformat(), _m(30).isoformat()),
    )
    e1.conn.execute(
        "INSERT INTO open_positions (id, symbol, direction, entry_price, qty, stop_loss, take_profit, "
        "margin, entry_time) VALUES ('P1', ?, 'long', 100.0, 1.0, 98.0, 105.0, 100.0, ?)",
        (SYM, _m(0).isoformat()),
    )
    e1.conn.execute("DELETE FROM engine_state WHERE key='schema_version'")
    e1.conn.commit()
    e1.conn.close()

    e2 = _engine(db)
    fee_t = round(100.0 * 2.0 * TAKER_FEE, 8)
    row = e2.conn.execute("SELECT entry_fee, pnl, pnl_pct, r_multiple, is_maker FROM trades WHERE id='L1'").fetchone()
    assert row[0] == pytest.approx(fee_t, abs=1e-9)
    assert row[1] == pytest.approx(9.8 - fee_t, abs=1e-9)
    assert row[2] == pytest.approx((9.8 - fee_t) / 200.0, abs=1e-9)
    assert row[3] == pytest.approx((9.8 - fee_t) / 4.0, abs=1e-6)
    assert row[4] == 0
    pos = [p for p in e2.get_positions() if p.id == "P1"][0]
    assert pos.entry_fee == pytest.approx(100.0 * 1.0 * TAKER_FEE, abs=1e-9) and pos.is_maker is False
    ver = e2.conn.execute("SELECT value FROM engine_state WHERE key='schema_version'").fetchone()[0]
    assert int(ver) == 2
    e2.conn.close()
    # 멱등: 다시 열어도 값 불변
    e3 = _engine(db)
    row2 = e3.conn.execute("SELECT pnl FROM trades WHERE id='L1'").fetchone()
    assert row2[0] == pytest.approx(9.8 - fee_t, abs=1e-9)


# ────────────────────────────────────────────────────────────────────────
# 3. 체결봉/진입봉 규칙 + 시간순 SL/TP
# ────────────────────────────────────────────────────────────────────────

def test_fill_bar_tp_not_awarded(tmp_path) -> None:
    """체결봉 고가가 TP를 넘어도 TP 인정 안 함(순서 불명) — 포지션 유지."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    candles = _candles([(_m(15), 104.0, 98.5)])        # 저가 98.5 ≤ 99 체결, 고가 104 ≥ 103
    assert len(eng.check_pending_fills(SYM, candles, now=_m(30))) == 1
    eng.check_stops_history(SYM, candles, now=_m(30))
    assert len(eng.get_positions()) == 1
    assert _closed_trades(eng) == []


def test_fill_bar_sl_recognized(tmp_path) -> None:
    """체결봉 저가가 SL 이하면 SL 청산 (보수)."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    candles = _candles([(_m(15), 100.0, 96.9)])        # 체결 + 저가 96.9 ≤ SL 97
    assert len(eng.check_pending_fills(SYM, candles, now=_m(30))) == 1
    eng.check_stops_history(SYM, candles, now=_m(30))
    assert eng.get_positions() == []
    trades = _closed_trades(eng)
    assert len(trades) == 1 and trades[0][0] == "SL"


def test_tp_awarded_on_later_bar(tmp_path) -> None:
    """체결봉 다음 봉에서 TP 터치 → TP 청산, 청산시각 = 그 봉 마감."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    candles = _candles([(_m(15), 100.0, 98.5), (_m(30), 103.5, 99.0)])
    eng.check_pending_fills(SYM, candles, now=_m(45))
    eng.check_stops_history(SYM, candles, now=_m(45))
    trades = _closed_trades(eng)
    assert len(trades) == 1 and trades[0][0] == "TP"
    assert datetime.fromisoformat(trades[0][2]) == _m(45)


def test_sl_first_within_bar(tmp_path) -> None:
    """같은 봉에 SL·TP 모두 터치 → SL 우선 (진입봉 아님)."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=True, entry_time=_m(3))
    candles = _candles([(_m(0), 100.0, 99.0), (_m(15), 104.0, 96.0)])
    eng.check_stops_history(SYM, candles, now=_m(40))
    assert _closed_trades(eng)[0][0] == "SL"
    assert pos not in eng.get_positions()


def test_intermediate_bar_sl_not_missed(tmp_path) -> None:
    """실행이 건너뛰어도 중간 봉의 SL 터치를 놓치지 않는다 (마지막 봉만 보던 구버전과 다름)."""
    eng = _engine(tmp_path / "t.db")
    eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    candles = _candles([(_m(0), 100.5, 99.5),          # 진입봉 — 미터치
                        (_m(15), 100.0, 96.5),         # SL 터치 (봇이 이 봉에서 실행 안 됐다고 가정)
                        (_m(30), 100.0, 99.0),         # 마지막 봉 — 미터치
                        (_m(45), 100.0, 99.0)])
    eng.check_stops_history(SYM, candles, now=_m(50))
    trades = _closed_trades(eng)
    assert len(trades) == 1 and trades[0][0] == "SL"
    assert datetime.fromisoformat(trades[0][2]) == _m(30)   # 00:15 봉 마감


def test_entry_bar_tp_not_awarded_for_taker(tmp_path) -> None:
    """시장가 진입(00:07)한 봉의 고가가 TP 이상이어도 TP 불인정 (진입 전 고점일 수 있음)."""
    eng = _engine(tmp_path / "t.db")
    eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(7))
    candles = _candles([(_m(0), 106.0, 99.5)])
    eng.check_stops_history(SYM, candles, now=_m(10))
    assert len(eng.get_positions()) == 1
    # 다음 봉에서 TP → 인정
    candles2 = _candles([(_m(0), 106.0, 99.5), (_m(15), 103.2, 100.0)])
    eng.check_stops_history(SYM, candles2, now=_m(31))
    assert _closed_trades(eng)[0][0] == "TP"


def test_forming_bar_not_evaluated_until_closed(tmp_path) -> None:
    """형성 중 봉은 판정하지 않는다 — 닫힌 뒤(다음 사이클) 판정. 청산시각 = 봉 마감."""
    eng = _engine(tmp_path / "t.db")
    pos = eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    # 00:28: 진입봉(00:00) 닫힘·미터치, 00:15 봉 형성 중이고 저가가 SL 이하여도 아직 판정 안 함
    c = _candles([(_m(0), 100.5, 99.5), (_m(15), 100.2, 96.8)])
    eng.check_stops_history(SYM, c, now=_m(28))
    assert pos.last_checked_bar == _m(0) and len(eng.get_positions()) == 1
    # 00:31: 00:15 봉 닫힘 → SL, 청산시각 00:30
    eng.check_stops_history(SYM, c, now=_m(31))
    t = _closed_trades(eng)
    assert t[0][0] == "SL" and datetime.fromisoformat(t[0][2]) == _m(30)


def test_last_checked_bar_persists_across_restart(tmp_path) -> None:
    """last_checked_bar가 DB에 저장돼 재시작 후에도 이어진다."""
    db = tmp_path / "t.db"
    e1 = _engine(db)
    e1.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    e1.check_stops_history(SYM, _candles([(_m(0), 100.5, 99.5), (_m(15), 100.5, 99.5)]), now=_m(35))
    e1.conn.close()
    e2 = _engine(db)
    assert len(e2.get_positions()) == 1
    assert e2.get_positions()[0].last_checked_bar == _m(15)


# ────────────────────────────────────────────────────────────────────────
# 4. 프로세스 재시작 영속 / 원자성 / 중복
# ────────────────────────────────────────────────────────────────────────

def test_pending_persists_and_fills_across_engines(tmp_path) -> None:
    """엔진 A가 등록 → 엔진 B가 체결 → 엔진 C가 포지션 복원(미체결 0, 포지션 id = 주문 id)."""
    db = tmp_path / "t.db"
    a = _engine(db)
    oid = _place_long(a)
    assert oid is not None
    a.conn.close()

    b = _engine(db)
    assert len(b.get_pending_orders(SYM)) == 1
    filled = b.check_pending_fills(SYM, _candles([(_m(15), 100.0, 98.5)]), now=_m(30))
    assert len(filled) == 1 and filled[0].id == oid
    b.conn.close()

    c = _engine(db)
    assert c.get_pending_orders(SYM) == []
    assert [p.id for p in c.get_positions()] == [oid]
    hist = c.get_order_history(SYM)[0]
    assert hist["status"] == "filled" and hist["position_id"] == oid
    # 재시작 후 같은 캔들로 다시 판정해도 중복 체결 없음 (멱등)
    assert c.check_pending_fills(SYM, _candles([(_m(15), 100.0, 98.5)]), now=_m(30)) == []
    assert len(c.get_positions()) == 1


def test_signal_key_dedupe_and_one_order_per_symbol(tmp_path) -> None:
    """같은 signal_key 재등록 거부, 같은 심볼 미체결 존재 시 거부, 만료 후 새 키는 허용."""
    eng = _engine(tmp_path / "t.db")
    k1 = f"{SYM}|long|{_m(0).isoformat()}"
    assert _place_long(eng, signal_key=k1) is not None
    assert _place_long(eng, signal_key=k1) is None                 # 같은 결정봉 중복
    assert _place_long(eng, signal_key=k1 + "x") is None           # 다른 키지만 심볼 중복
    # 만료 (창 8봉 전부 관측, 미터치)
    eng.check_pending_fills(SYM, _candles(_window(15, 8)), now=_m(200))
    assert eng.get_pending_orders(SYM) == []
    assert _place_long(eng, signal_key=k1) is None                 # 만료된 키 재사용 불가
    k2 = f"{SYM}|long|{_m(195).isoformat()}"
    assert _place_long(eng, signal_key=k2, place_time=_m(198)) is not None   # 새 결정봉 → 허용


def test_cancel_all_pending_on_mode_change(tmp_path) -> None:
    """taker 모드 기동 시 미체결 전부 취소 → 이력에 cancelled/mode_change."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng)
    eng.place_pending_limit("ETH/USDT:USDT", "short", 101.0, 1.0, 103.0, 97.0, place_time=_m(3))
    assert eng.cancel_all_pending("mode_change") == 2
    assert eng.get_pending_orders() == []
    assert {h["status"] for h in eng.get_order_history()} == {"cancelled"}
    assert {h["resolve_reason"] for h in eng.get_order_history()} == {"mode_change"}


def test_reservation_limits_new_orders(tmp_path) -> None:
    """미체결 예약분(증거금+수수료)이 가용 잔고에서 빠지고, 초과 주문은 거부된다."""
    eng = _engine(tmp_path / "t.db", balance=150.0)
    assert _place_long(eng, limit_price=100.0, qty=1.0, leverage=1.0) is not None   # 100 + 0.035
    assert eng.reserved_margin() == pytest.approx(100.035, abs=1e-6)
    assert eng.available_balance() == pytest.approx(150.0 - 100.035, abs=1e-6)
    # 다른 심볼 60 USDT 필요 → 가용 49.965 → 거부
    assert eng.place_pending_limit("ETH/USDT:USDT", "long", 60.0, 1.0, 58.0, 64.0,
                                   leverage=1.0, place_time=_m(3)) is None
    assert len(eng.get_pending_orders()) == 1


def test_pending_as_positions_for_risk_limits(tmp_path) -> None:
    """미체결이 리스크 한도 계산용 가상 포지션으로 나온다 (방향/심볼/마진/명목)."""
    eng = _engine(tmp_path / "t.db")
    oid = _place_long(eng, limit_price=100.0, qty=2.0, leverage=4.0)
    vp = eng.pending_as_positions()
    assert len(vp) == 1
    v = vp[0]
    assert v.id == oid and v.symbol == SYM and v.direction == "long"
    assert v.entry_price == 100.0 and v.qty == 2.0
    assert v.margin == pytest.approx(200.0 / 4.0, abs=1e-9)
    assert v.is_maker is True


def test_insufficient_balance_at_fill_cancels_not_drops(tmp_path) -> None:
    """체결 시점 잔고 부족이면 주문을 'cancelled/insufficient_balance'로 남긴다(조용한 삭제 금지)."""
    eng = _engine(tmp_path / "t.db", balance=150.0)
    oid = _place_long(eng, limit_price=100.0, qty=1.0, leverage=1.0)
    # 예약 후 잔고를 강제로 줄여 체결 불가 상태를 만든다
    eng.balance = 50.0
    assert eng.check_pending_fills(SYM, _candles([(_m(15), 101.0, 99.0)]), now=_m(30)) == []
    h = eng.get_order_history(SYM)[0]
    assert h["id"] == oid and h["status"] == "cancelled" and h["resolve_reason"] == "insufficient_balance"
    assert eng.get_positions() == []


# ────────────────────────────────────────────────────────────────────────
# 5. 리스크 한도 — 후보 주문 자체의 노출 포함
# ────────────────────────────────────────────────────────────────────────

def test_candidate_exposure_counted_in_limits(tmp_path) -> None:
    """후보 포지션의 증거금/명목이 총노출·명목노출 한도 판정에 더해진다 (사후 초과 방지)."""
    import src.risk.circuit_breaker as cb_module
    from src.paper_trading import Position
    cfg = {
        "exchange": {"symbols": [SYM], "leverage": 5},
        "capital": {"total_capital": 1000, "trading_allocation": 1.0, "risk_per_trade": 0.01},
        "risk": {"min_rr_ratio": 2.0, "max_positions": 4, "max_per_symbol": 1, "max_same_direction": 3,
                 "max_exposure_pct": 0.80, "max_notional_mult": 3.0, "daily_loss_limit": 0.05,
                 "weekly_loss_limit": 0.12, "max_consecutive_losses": 7},
        "scan": {"mode": "static", "min_score": 70},
    }
    with patch("src.risk.risk_manager.load_config", return_value=cfg), \
         patch.object(cb_module, "DB_PATH", tmp_path / "cb.db"):
        from src.risk.risk_manager import RiskManager
        rm = RiskManager()
        big_margin = Position(id="c", symbol=SYM, direction="long", entry_price=100.0, qty=9.0,
                              stop_loss=98.0, take_profit=105.0, margin=900.0)      # 900/1000 ≥ 80%
        ok, why = rm.check_trade_allowed(0, [], SYM, "long", candidate=big_margin)
        assert not ok and "총 노출" in why
        big_notional = Position(id="c", symbol=SYM, direction="long", entry_price=100.0, qty=35.0,
                                stop_loss=98.0, take_profit=105.0, margin=350.0)    # 명목 3500 ≥ 3×1000
        ok, why = rm.check_trade_allowed(0, [], SYM, "long", candidate=big_notional)
        assert not ok and "명목" in why
        fine = Position(id="c", symbol=SYM, direction="long", entry_price=100.0, qty=5.0,
                        stop_loss=98.0, take_profit=105.0, margin=100.0)
        assert rm.check_trade_allowed(0, [], SYM, "long", candidate=fine)[0]


# ────────────────────────────────────────────────────────────────────────
# 6. 판정 완료/미완 등록 — 봇의 '체결 전 기존 포지션 판정 완료' 게이트 근거
# ────────────────────────────────────────────────────────────────────────

def test_eval_incomplete_registry_for_stops(tmp_path) -> None:
    """마지막 닫힌 봉까지 판정하면 완료, 지연 프레임/NaN/빈 프레임/선행 갭이면 미완으로 등록."""
    eng = _engine(tmp_path / "t.db")
    eng.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    # 완료: 00:00, 00:15 닫힘 + 00:30 형성 중, now=00:40
    full = _candles([(_m(0), 100.5, 99.5), (_m(15), 100.5, 99.5), (_m(30), 100.5, 99.5)])
    eng.check_stops_history(SYM, full, now=_m(40))
    assert SYM not in eng.last_eval_incomplete
    # 지연 프레임: 00:15 봉이 닫혔는데(now=00:40) 프레임은 00:00까지만 → 미완
    eng.last_eval_incomplete.clear()
    eng2 = _engine(tmp_path / "t2.db")
    eng2.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    eng2.check_stops_history(SYM, _candles([(_m(0), 100.5, 99.5)]), now=_m(40))
    assert SYM in eng2.last_eval_incomplete and SYM not in eng2.last_history_gaps
    # NaN → 미완
    eng3 = _engine(tmp_path / "t3.db")
    eng3.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    eng3.check_stops_history(SYM, _candles([(_m(0), 100.5, 99.5), (_m(15), float("nan"), float("nan"))]), now=_m(40))
    assert SYM in eng3.last_eval_incomplete
    # 빈 프레임 → 미완 + 재조회
    eng4 = _engine(tmp_path / "t4.db")
    eng4.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    empty = _candles([])
    eng4.check_stops_history(SYM, empty, now=_m(40))
    assert SYM in eng4.last_eval_incomplete and SYM in eng4.last_history_gaps
    # 선행 갭 → 미완 + 재조회
    eng5 = _engine(tmp_path / "t5.db")
    eng5.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    eng5.check_stops_history(SYM, _candles([(_m(30), 100.5, 99.5)]), now=_m(40))
    assert SYM in eng5.last_eval_incomplete and SYM in eng5.last_history_gaps
    # 청산됐으면 완료
    eng6 = _engine(tmp_path / "t6.db")
    eng6.open_position(SYM, "long", 100.0, 1.0, 97.0, 103.0, is_maker=False, entry_time=_m(3))
    eng6.check_stops_history(SYM, _candles([(_m(0), 100.5, 96.0)]), now=_m(40))
    assert SYM not in eng6.last_eval_incomplete and eng6.get_positions() == []


def test_eval_incomplete_registry_for_fills(tmp_path) -> None:
    """체결 판정: 선행 갭/엄격 빈 봉/NaN/지연/빈 프레임 → 미완 등록, 정상 프레임 → 등록 없음."""
    eng = _engine(tmp_path / "t.db")
    _place_long(eng, expiry_bars=2)
    eng.check_pending_fills(SYM, _candles(_window(15, 2)), now=_m(50))      # 창 전부 관측 → 만료, 완료
    assert SYM not in eng.last_eval_incomplete
    eng2 = _engine(tmp_path / "t2.db")
    _place_long(eng2, expiry_bars=2)
    eng2.check_pending_fills(SYM, _candles([(_m(15), 101.0, 99.5)]), now=_m(50))   # 00:30 아직 없음 → 미완
    assert SYM in eng2.last_eval_incomplete
    eng3 = _engine(tmp_path / "t3.db")
    _place_long(eng3, expiry_bars=2)
    eng3.check_pending_fills(SYM, _candles([]), now=_m(50))
    assert SYM in eng3.last_eval_incomplete and SYM in eng3.last_history_gaps
