from __future__ import annotations

"""scan_symbol 테스트 — 컨플루언스 점수 + 관심종목/진입 분류"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.strategy.signal_engine import scan_symbol, ScanResult, TradeSignal


def _ohlcv(n: int = 100, vol: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, vol),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )


class TestScanNoTrend:
    def test_no_trend_returns_zero_score(self):
        df = _ohlcv()
        with (
            patch("src.strategy.signal_engine.detect_bos", return_value=None),
            patch("src.strategy.signal_engine.detect_choch", return_value=None),
        ):
            res = scan_symbol(df, df, df, "BTC/USDT:USDT", 100.0)
        assert isinstance(res, ScanResult)
        assert res.score == 0.0
        assert res.stage == 0
        assert res.direction == "none"
        assert res.qualified is False


class TestScanTrendOnly:
    def test_trend_but_no_zone_is_watchlist(self):
        df = _ohlcv()
        with (
            patch("src.strategy.signal_engine.detect_bos", return_value="bullish"),
            patch("src.strategy.signal_engine.detect_choch", return_value=None),
            patch("src.strategy.signal_engine.detect_fvg", return_value=[]),
            patch("src.strategy.signal_engine.detect_order_blocks", return_value=[]),
            patch("src.strategy.signal_engine.is_price_in_fvg", return_value=[]),
            patch("src.strategy.signal_engine.is_price_in_ob", return_value=[]),
        ):
            res = scan_symbol(df, df, df, "BTC/USDT:USDT", 100.0)
        assert res.direction == "long"
        assert res.stage == 1
        assert res.score > 0          # BOS 점수만큼
        assert res.qualified is False


class TestScanQualified:
    def test_full_confluence_qualifies(self):
        df = _ohlcv(vol=2000.0)
        fvg = [SimpleNamespace(type="bullish")]
        ote_zone = SimpleNamespace(ote_low=99.0, ote_high=101.0)
        signal = TradeSignal(
            direction="long", entry_price=100.0, stop_loss=98.0,
            take_profit=104.0, symbol="BTC/USDT:USDT",
            reason="confluence", rr_ratio=2.0,
        )
        with (
            patch("src.strategy.signal_engine.detect_bos", return_value="bullish"),
            patch("src.strategy.signal_engine.detect_choch", return_value=None),
            patch("src.strategy.signal_engine.detect_fvg", return_value=fvg),
            patch("src.strategy.signal_engine.detect_order_blocks", return_value=[]),
            patch("src.strategy.signal_engine.is_price_in_fvg", return_value=fvg),
            patch("src.strategy.signal_engine.is_price_in_ob", return_value=[]),
            patch("src.strategy.signal_engine.is_in_kill_zone", return_value=True),
            patch("src.strategy.signal_engine.get_active_session", return_value="london"),
            patch("src.strategy.signal_engine.calculate_ote_zone", return_value=ote_zone),
            patch("src.strategy.signal_engine.is_price_in_ote", return_value=True),
            patch("src.strategy.signal_engine.generate_signal", return_value=signal),
        ):
            res = scan_symbol(
                df, df, df, "BTC/USDT:USDT", 100.0,
                min_rr=1.5, min_score=70.0,
            )
        assert res.qualified is True
        assert res.signal is not None
        assert res.stage == 4
        assert res.score >= 70.0

    def test_high_min_score_blocks_entry(self):
        """min_score를 매우 높이면 컨플루언스가 부족해 진입 차단."""
        df = _ohlcv(vol=2000.0)
        fvg = [SimpleNamespace(type="bullish")]
        ote_zone = SimpleNamespace(ote_low=99.0, ote_high=101.0)
        signal = TradeSignal(
            direction="long", entry_price=100.0, stop_loss=98.0,
            take_profit=104.0, symbol="BTC/USDT:USDT",
            reason="x", rr_ratio=2.0,
        )
        with (
            patch("src.strategy.signal_engine.detect_bos", return_value=None),
            patch("src.strategy.signal_engine.detect_choch", return_value="bullish"),
            patch("src.strategy.signal_engine.detect_fvg", return_value=fvg),
            patch("src.strategy.signal_engine.detect_order_blocks", return_value=[]),
            patch("src.strategy.signal_engine.is_price_in_fvg", return_value=fvg),
            patch("src.strategy.signal_engine.is_price_in_ob", return_value=[]),
            patch("src.strategy.signal_engine.is_in_kill_zone", return_value=True),
            patch("src.strategy.signal_engine.get_active_session", return_value="ny"),
            patch("src.strategy.signal_engine.calculate_ote_zone", return_value=ote_zone),
            patch("src.strategy.signal_engine.is_price_in_ote", return_value=True),
            patch("src.strategy.signal_engine.generate_signal", return_value=signal),
        ):
            res = scan_symbol(
                df, df, df, "ETH/USDT:USDT", 100.0,
                min_rr=1.5, min_score=99.0,  # 도달 불가능한 점수
            )
        assert res.qualified is False
        assert res.signal is None        # 미확정이면 signal 미노출


class TestScanResultToDict:
    def test_to_dict_serializable(self):
        res = ScanResult(
            symbol="BTC/USDT:USDT", direction="long", score=72.5,
            stage=3, qualified=False, price=100.0, reason="test",
        )
        d = res.to_dict()
        assert d["symbol"] == "BTC/USDT:USDT"
        assert d["score"] == 72.5
        assert d["signal"] is None
        assert d["checks"] == {}


class TestKzGateRemoved:
    def test_kz_false_can_qualify(self):
        """킬존 밖이어도 추세/존/OTE 충족 + 점수 충분이면 qualified (24h 진입)."""
        df = _ohlcv(vol=2000.0)
        fvg = [SimpleNamespace(type="bullish")]
        ob = [SimpleNamespace(type="bullish")]
        ote_zone = SimpleNamespace(ote_low=99.0, ote_high=101.0)
        signal = TradeSignal(
            direction="long", entry_price=100.0, stop_loss=98.0,
            take_profit=105.0, symbol="BTC/USDT:USDT",
            reason="x", rr_ratio=2.5,
        )
        with (
            patch("src.strategy.signal_engine.detect_bos", return_value="bullish"),
            patch("src.strategy.signal_engine.detect_choch", return_value=None),
            patch("src.strategy.signal_engine.detect_fvg", return_value=fvg),
            patch("src.strategy.signal_engine.detect_order_blocks", return_value=ob),
            patch("src.strategy.signal_engine.is_price_in_fvg", return_value=fvg),
            patch("src.strategy.signal_engine.is_price_in_ob", return_value=ob),
            patch("src.strategy.signal_engine.is_in_kill_zone", return_value=False),
            patch("src.strategy.signal_engine.get_active_session", return_value=None),
            patch("src.strategy.signal_engine.calculate_ote_zone", return_value=ote_zone),
            patch("src.strategy.signal_engine.is_price_in_ote", return_value=True),
            patch("src.strategy.signal_engine.generate_signal", return_value=signal),
        ):
            res = scan_symbol(
                df, df, df, "BTC/USDT:USDT", 100.0,
                min_rr=2.5, min_score=70.0,
            )
        assert res.checks["kill_zone"] is False   # 태깅은 유지
        assert res.qualified is True              # 게이트는 아님
