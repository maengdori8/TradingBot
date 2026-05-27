"""bybit_client.py (MarketDataClient) 테스트 — 네트워크 호출 전부 mock"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, call
import ccxt
import pandas as pd

from src.exchange.bybit_client import (
    MarketDataClient,
    _spot_symbols,
    _retry_call,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)


# ─── _spot_symbols 변환 ──────────────────────────────────────────────

class TestSpotSymbols:
    """_spot_symbols 유틸 함수 테스트."""

    def test_futures_to_spot(self):
        """선물 심볼 -> 현물 심볼 후보 리스트 변환."""
        result = _spot_symbols("BTC/USDT:USDT")
        assert result == ["BTC/USDT", "BTC/USD"]

    def test_usd_quote_no_extra(self):
        """quote가 USD이면 추가 후보 없음."""
        result = _spot_symbols("BTC/USD:USD")
        assert result == ["BTC/USD"]

    def test_other_quote(self):
        """quote가 USDT가 아닌 경우 단일 후보."""
        result = _spot_symbols("ETH/EUR:EUR")
        assert result == ["ETH/EUR"]


# ─── _retry_call ─────────────────────────────────────────────────────

class TestRetryCall:
    """_retry_call exponential backoff 테스트."""

    def test_success_first_try(self):
        """첫 시도에 성공하면 그대로 반환."""
        func = MagicMock(return_value=42)
        result = _retry_call(func, "a", "b", key="val")
        assert result == 42
        func.assert_called_once_with("a", "b", key="val")

    @patch("src.exchange.bybit_client.time.sleep")
    def test_retry_on_network_error(self, mock_sleep):
        """NetworkError 발생 시 재시도 후 성공."""
        func = MagicMock(side_effect=[ccxt.NetworkError("fail"), "ok"])
        result = _retry_call(func)
        assert result == "ok"
        assert func.call_count == 2
        mock_sleep.assert_called_once_with(RETRY_BASE_DELAY * 1)

    @patch("src.exchange.bybit_client.time.sleep")
    def test_retry_on_exchange_not_available(self, mock_sleep):
        """ExchangeNotAvailable 에러도 재시도 대상."""
        func = MagicMock(
            side_effect=[ccxt.ExchangeNotAvailable("down"), "ok"]
        )
        result = _retry_call(func)
        assert result == "ok"

    @patch("src.exchange.bybit_client.time.sleep")
    def test_retry_on_request_timeout(self, mock_sleep):
        """RequestTimeout 에러도 재시도 대상."""
        func = MagicMock(
            side_effect=[ccxt.RequestTimeout("timeout"), "ok"]
        )
        result = _retry_call(func)
        assert result == "ok"

    @patch("src.exchange.bybit_client.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep):
        """최대 재시도 횟수 초과 시 마지막 예외 전파."""
        exc = ccxt.NetworkError("persistent failure")
        func = MagicMock(side_effect=[exc] * MAX_RETRIES)
        with pytest.raises(ccxt.NetworkError, match="persistent failure"):
            _retry_call(func)
        assert func.call_count == MAX_RETRIES

    @patch("src.exchange.bybit_client.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """재시도 간 지연이 exponential backoff 패턴을 따르는지 확인."""
        func = MagicMock(
            side_effect=[
                ccxt.NetworkError("1"),
                ccxt.NetworkError("2"),
                "ok",
            ]
        )
        _retry_call(func)
        assert mock_sleep.call_args_list == [
            call(RETRY_BASE_DELAY * 1),  # 2^0
            call(RETRY_BASE_DELAY * 2),  # 2^1
        ]

    def test_non_network_error_raises_immediately(self):
        """네트워크 이외 오류는 즉시 전파."""
        func = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            _retry_call(func)
        func.assert_called_once()


# ─── MarketDataClient ────────────────────────────────────────────────

def _make_mock_client(fetch_ohlcv_rv=None, fetch_ticker_rv=None, fail_init=False):
    """mock ccxt 거래소 인스턴스 생성 헬퍼."""
    mock = MagicMock()
    if fail_init:
        mock.load_markets.side_effect = ccxt.NetworkError("init fail")
    else:
        mock.load_markets.return_value = {}
        mock.markets = {}
    if fetch_ohlcv_rv is not None:
        mock.fetch_ohlcv.return_value = fetch_ohlcv_rv
    if fetch_ticker_rv is not None:
        mock.fetch_ticker.return_value = fetch_ticker_rv
    return mock


class TestMarketDataClient:
    """MarketDataClient 테스트 — ccxt 클래스 전부 mock."""

    @patch("src.exchange.bybit_client.time.sleep")
    def test_init_success(self, mock_sleep):
        """초기화 성공 — lazy 모드에서 첫 호출 시 거래소 등록."""
        mock_bybit = _make_mock_client()
        mock_bybit.fetch_ticker.return_value = {"last": 50000.0}
        mock_cls = MagicMock(return_value=mock_bybit)

        configs = [("bybit", mock_cls, {"options": {"defaultType": "swap"}}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            # lazy 모드이므로 생성 직후에는 _clients가 비어 있음
            assert len(client._clients) == 0
            # 첫 API 호출 시 초기화됨
            client.fetch_current_price("BTC/USDT:USDT")
            assert "bybit" in client._clients

    @patch("src.exchange.bybit_client.time.sleep")
    def test_init_exchange_fail_graceful(self, mock_sleep):
        """거래소 초기화 실패 시 에러 없이 건너뜀 (lazy 모드)."""
        mock_bybit = _make_mock_client(fail_init=True)
        mock_cls = MagicMock(return_value=mock_bybit)

        configs = [("bybit", mock_cls, {"options": {"defaultType": "swap"}}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            # lazy 모드에서 첫 호출 시 초기화 실패 → _clients 비어 있음
            with pytest.raises(RuntimeError):
                client.fetch_current_price("BTC/USDT:USDT")
            assert len(client._clients) == 0

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ohlcv_success_bybit(self, mock_sleep):
        """Bybit에서 OHLCV 정상 조회."""
        raw = [
            [1700000000000, 50000, 51000, 49500, 50500, 100],
            [1700000060000, 50500, 51500, 50000, 51000, 120],
        ]
        mock_bybit = _make_mock_client(fetch_ohlcv_rv=raw)
        mock_cls = MagicMock(return_value=mock_bybit)

        configs = [("bybit", mock_cls, {"options": {"defaultType": "swap"}}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            df = client.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=200)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        mock_bybit.fetch_ohlcv.assert_called_once_with(
            "BTC/USDT:USDT", "15m", limit=200
        )

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ohlcv_fallback_bybit_fail_kraken_ok(self, mock_sleep):
        """Bybit 실패 -> Kraken fallback 성공."""
        raw = [[1700000000000, 50000, 51000, 49500, 50500, 100]]

        mock_bybit = _make_mock_client(fail_init=True)
        mock_bybit_cls = MagicMock(return_value=mock_bybit)

        mock_kraken = _make_mock_client(fetch_ohlcv_rv=raw)
        mock_kraken_cls = MagicMock(return_value=mock_kraken)

        configs = [
            ("bybit", mock_bybit_cls, {"options": {"defaultType": "swap"}}, True),
            ("kraken", mock_kraken_cls, {}, False),
        ]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            df = client.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        # Kraken에서는 spot 심볼로 변환되어 호출
        mock_kraken.fetch_ohlcv.assert_called()

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ohlcv_all_exchanges_fail(self, mock_sleep):
        """모든 거래소 실패 시 RuntimeError."""
        mock_bybit = _make_mock_client(fail_init=True)
        mock_kraken = _make_mock_client(fail_init=True)

        configs = [
            ("bybit", MagicMock(return_value=mock_bybit), {}, True),
            ("kraken", MagicMock(return_value=mock_kraken), {}, False),
        ]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            with pytest.raises(RuntimeError, match="모든 거래소 OHLCV 실패"):
                client.fetch_ohlcv("BTC/USDT:USDT")

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ticker_success(self, mock_sleep):
        """ticker 정상 조회."""
        ticker_data = {
            "symbol": "BTC/USDT:USDT",
            "last": 50500.0,
            "bid": 50499.0,
            "ask": 50501.0,
        }
        mock_bybit = _make_mock_client(fetch_ticker_rv=ticker_data)
        mock_cls = MagicMock(return_value=mock_bybit)

        configs = [("bybit", mock_cls, {"options": {"defaultType": "swap"}}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            ticker = client.fetch_ticker("BTC/USDT:USDT")

        assert ticker["last"] == 50500.0
        assert ticker["symbol"] == "BTC/USDT:USDT"

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ticker_all_fail(self, mock_sleep):
        """모든 거래소 ticker 실패 시 RuntimeError."""
        mock_bybit = _make_mock_client(fail_init=True)

        configs = [("bybit", MagicMock(return_value=mock_bybit), {}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            with pytest.raises(RuntimeError, match="ticker 조회 실패"):
                client.fetch_ticker("BTC/USDT:USDT")

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_current_price_returns_last(self, mock_sleep):
        """fetch_current_price는 ticker의 'last' 값을 float로 반환."""
        mock_bybit = _make_mock_client(fetch_ticker_rv={"last": 42000.5})
        mock_cls = MagicMock(return_value=mock_bybit)

        configs = [("bybit", mock_cls, {"options": {"defaultType": "swap"}}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            price = client.fetch_current_price("BTC/USDT:USDT")

        assert price == 42000.5
        assert isinstance(price, float)

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ohlcv_spot_symbol_fallback_order(self, mock_sleep):
        """비선물 거래소에서는 spot 심볼 후보 순서대로 시도."""
        raw = [[1700000000000, 50000, 51000, 49500, 50500, 100]]

        # Bybit 초기화 실패
        mock_bybit = _make_mock_client(fail_init=True)
        mock_bybit_cls = MagicMock(return_value=mock_bybit)

        # Kraken: 첫 번째 spot 심볼 실패, 두 번째 성공
        mock_kraken = _make_mock_client()
        mock_kraken.fetch_ohlcv.side_effect = [
            ccxt.BadSymbol("BTC/USDT not found"),
            raw,
        ]
        mock_kraken_cls = MagicMock(return_value=mock_kraken)

        configs = [
            ("bybit", mock_bybit_cls, {"options": {"defaultType": "swap"}}, True),
            ("kraken", mock_kraken_cls, {}, False),
        ]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            df = client.fetch_ohlcv("BTC/USDT:USDT", "15m")

        assert len(df) == 1
        # BTC/USDT 시도 후 BTC/USD로 시도
        assert mock_kraken.fetch_ohlcv.call_count == 2

    @patch("src.exchange.bybit_client.time.sleep")
    def test_init_generic_exception_handled(self, mock_sleep):
        """거래소 생성자에서 일반 Exception 발생 시 lazy 초기화 실패 처리."""
        mock_cls = MagicMock(side_effect=RuntimeError("constructor fail"))

        configs = [("bybit", mock_cls, {}, True)]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            # lazy 모드이므로 생성 시에는 _clients 비어 있음
            assert len(client._clients) == 0
            # 첫 호출 시 초기화 시도 → 실패 → RuntimeError
            with pytest.raises(RuntimeError):
                client.fetch_current_price("BTC/USDT:USDT")

    @patch("src.exchange.bybit_client.time.sleep")
    def test_fetch_ohlcv_bybit_api_fail_kraken_api_ok(self, mock_sleep):
        """Bybit API 호출 실패 -> Kraken API 성공 (초기화는 둘 다 성공)."""
        raw = [[1700000000000, 50000, 51000, 49500, 50500, 100]]

        mock_bybit = _make_mock_client()
        mock_bybit.fetch_ohlcv.side_effect = ccxt.ExchangeError("bybit API down")
        mock_bybit_cls = MagicMock(return_value=mock_bybit)

        mock_kraken = _make_mock_client(fetch_ohlcv_rv=raw)
        mock_kraken_cls = MagicMock(return_value=mock_kraken)

        configs = [
            ("bybit", mock_bybit_cls, {"options": {"defaultType": "swap"}}, True),
            ("kraken", mock_kraken_cls, {}, False),
        ]
        with patch("src.exchange.bybit_client.EXCHANGE_CONFIGS", configs):
            client = MarketDataClient()
            df = client.fetch_ohlcv("BTC/USDT:USDT", "15m")

        assert len(df) == 1
