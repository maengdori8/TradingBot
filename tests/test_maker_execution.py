"""메이커 지정가 실행 단위테스트 — 출혈 감소 패치 검증.

검증: 지정가 정확체결(슬리피지 0) + 메이커 수수료, 체결 트리거(캔들 터치), 만료 취소,
테이커 대비 비용 감소. 실행: pytest tests/test_maker_execution.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.paper_trading.paper_engine import PaperEngine, TAKER_FEE


def _engine(tmp_path) -> PaperEngine:
    return PaperEngine(initial_balance=10000.0, db_path=tmp_path / "t.db", maker_fee=0.00035)


def _candles(rows: list[tuple[datetime, float, float]]) -> pd.DataFrame:
    """[(ts, high, low)] → DatetimeIndex(UTC) 프레임."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    return pd.DataFrame({"high": [r[1] for r in rows], "low": [r[2] for r in rows]}, index=idx)


def test_maker_no_slippage_and_fee(tmp_path) -> None:
    """is_maker=True: 진입가=지정가 정확(슬리피지 0), 메이커 수수료."""
    eng = _engine(tmp_path)
    b0 = eng.balance
    pos = eng.open_position("BTC/USDT:USDT", "long", entry_price=100.0, qty=1.0,
                            stop_loss=98.0, take_profit=105.0, is_maker=True)
    assert pos is not None
    assert pos.entry_price == 100.0          # 슬리피지 없음 (테이커면 100.05)
    # 수수료 = notional×maker_fee = 100×0.00035 = 0.035; 증거금 100 차감
    assert b0 - eng.balance == pytest.approx(100.0 + 100.0 * 0.00035, abs=1e-6)


def test_taker_has_slippage(tmp_path) -> None:
    """is_maker=False: 불리한 슬리피지 + 테이커 수수료 (기존 동작)."""
    eng = _engine(tmp_path)
    pos = eng.open_position("ETH/USDT:USDT", "long", entry_price=100.0, qty=1.0,
                            stop_loss=98.0, take_profit=105.0, is_maker=False)
    assert pos.entry_price > 100.0           # 롱 진입 슬리피지 = 더 높은 가격
    assert pos.entry_price == pytest.approx(100.0 * 1.0005, abs=1e-6)


def test_pending_fills_when_touched(tmp_path) -> None:
    """등록 후 캔들 저가가 롱 지정가에 닿으면 체결, 진입가=지정가."""
    eng = _engine(tmp_path)
    t0 = datetime(2026, 6, 16, 0, 0, tzinfo=timezone.utc)
    eng.place_pending_limit("BTC/USDT:USDT", "long", limit_price=99.0, qty=1.0,
                            stop_loss=97.0, take_profit=103.0, expiry_bars=8, place_time=t0)
    assert len(eng.get_pending_orders("BTC/USDT:USDT")) == 1
    # 등록 이후 캔들: 저가 98.5 ≤ 99.0 → 체결
    candles = _candles([(t0 + timedelta(minutes=15), 100.5, 98.5)])
    filled = eng.check_pending_fills("BTC/USDT:USDT", candles, now=t0 + timedelta(minutes=20))
    assert len(filled) == 1
    assert filled[0].entry_price == 99.0     # 지정가 정확체결
    assert len(eng.get_pending_orders("BTC/USDT:USDT")) == 0  # 대기열서 제거


def test_pending_not_filled_when_not_touched(tmp_path) -> None:
    """가격이 지정가에 안 닿으면 미체결 — 만료 전엔 대기 유지."""
    eng = _engine(tmp_path)
    t0 = datetime(2026, 6, 16, 0, 0, tzinfo=timezone.utc)
    eng.place_pending_limit("BTC/USDT:USDT", "long", limit_price=99.0, qty=1.0,
                            stop_loss=97.0, take_profit=103.0, expiry_bars=8, place_time=t0)
    candles = _candles([(t0 + timedelta(minutes=15), 101.0, 99.5)])  # 저가 99.5 > 99.0
    filled = eng.check_pending_fills("BTC/USDT:USDT", candles, now=t0 + timedelta(minutes=20))
    assert len(filled) == 0
    assert len(eng.get_pending_orders("BTC/USDT:USDT")) == 1   # 만료 전 — 유지


def test_pending_expires(tmp_path) -> None:
    """만료시각 초과 + 미터치 → 취소(거래없음), 잔고 불변."""
    eng = _engine(tmp_path)
    b0 = eng.balance
    t0 = datetime(2026, 6, 16, 0, 0, tzinfo=timezone.utc)
    eng.place_pending_limit("BTC/USDT:USDT", "long", limit_price=99.0, qty=1.0,
                            stop_loss=97.0, take_profit=103.0, expiry_bars=2, place_time=t0)
    candles = _candles([(t0 + timedelta(minutes=15), 101.0, 99.5)])
    now = t0 + timedelta(hours=1)            # expiry_bars=2 → 30분 만료 초과
    filled = eng.check_pending_fills("BTC/USDT:USDT", candles, now=now)
    assert len(filled) == 0
    assert len(eng.get_pending_orders("BTC/USDT:USDT")) == 0   # 만료 취소
    assert eng.balance == b0                  # 거래 없으니 잔고 불변 (출혈 0)


def test_short_fills_on_high_touch(tmp_path) -> None:
    """숏 지정가는 캔들 고가가 닿아야 체결."""
    eng = _engine(tmp_path)
    t0 = datetime(2026, 6, 16, 0, 0, tzinfo=timezone.utc)
    eng.place_pending_limit("ETH/USDT:USDT", "short", limit_price=101.0, qty=1.0,
                            stop_loss=103.0, take_profit=97.0, expiry_bars=8, place_time=t0)
    candles = _candles([(t0 + timedelta(minutes=15), 101.5, 100.0)])  # 고가 101.5 ≥ 101.0
    filled = eng.check_pending_fills("ETH/USDT:USDT", candles, now=t0 + timedelta(minutes=20))
    assert len(filled) == 1
    assert filled[0].entry_price == 101.0


def test_maker_cheaper_than_taker(tmp_path) -> None:
    """동일 거래에서 메이커 총비용 < 테이커 (출혈 감소 확인)."""
    e_m = _engine(tmp_path / "m")
    e_t = _engine(tmp_path / "t")
    bm0, bt0 = e_m.balance, e_t.balance
    e_m.open_position("BTC/USDT:USDT", "long", 100.0, 1.0, 98.0, 105.0, is_maker=True)
    e_t.open_position("BTC/USDT:USDT", "long", 100.0, 1.0, 98.0, 105.0, is_maker=False)
    cost_m = bm0 - e_m.balance
    cost_t = bt0 - e_t.balance
    assert cost_m < cost_t                    # 메이커가 덜 든다 (슬리피지+수수료 모두↓)
    assert TAKER_FEE > 0.00035                # 전제 확인


def _run_all(tmp_factory) -> None:
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for i, fn in enumerate(fns):
        fn(tmp_factory(i))
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} 통과")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    base = Path(tempfile.mkdtemp())
    _run_all(lambda i: (base / f"c{i}").resolve().__class__(base / f"c{i}") if False
             else _mkdir(base, i))


def _mkdir(base, i):
    from pathlib import Path
    p = Path(base) / f"case{i}"
    p.mkdir(parents=True, exist_ok=True)
    return p
