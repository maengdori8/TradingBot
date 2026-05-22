"""시장 데이터 fallback 테스트."""
from __future__ import annotations

import importlib
import sys
import types


class FakeExchange:
    def __init__(self, fake_ccxt, *, ticker=None, ohlcv=None, blocked=False, bad_symbols=()):
        self.fake_ccxt = fake_ccxt
        self.ticker = ticker or {}
        self.ohlcv = ohlcv or []
        self.blocked = blocked
        self.bad_symbols = set(bad_symbols)
        self.ticker_calls: list[str] = []
        self.ohlcv_calls: list[tuple[str, str, int]] = []

    def _maybe_raise(self, symbol: str) -> None:
        if self.blocked:
            raise self.fake_ccxt.BaseError("403 Forbidden country block")
        if symbol in self.bad_symbols:
            raise self.fake_ccxt.BadSymbol(f"bad symbol: {symbol}")

    def fetch_ticker(self, symbol: str):
        self.ticker_calls.append(symbol)
        self._maybe_raise(symbol)
        return self.ticker[symbol]

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200):
        self.ohlcv_calls.append((symbol, timeframe, limit))
        self._maybe_raise(symbol)
        return self.ohlcv


def import_client_with_fake_ccxt(monkeypatch, exchanges: dict[str, FakeExchange]):
    fake_ccxt = types.SimpleNamespace()

    class BaseError(Exception):
        pass

    class NetworkError(BaseError):
        pass

    class RequestTimeout(NetworkError):
        pass

    class BadSymbol(BaseError):
        pass

    fake_ccxt.BaseError = BaseError
    fake_ccxt.NetworkError = NetworkError
    fake_ccxt.RequestTimeout = RequestTimeout
    fake_ccxt.BadSymbol = BadSymbol

    for exchange in exchanges.values():
        exchange.fake_ccxt = fake_ccxt

    for name, exchange in exchanges.items():
        setattr(fake_ccxt, name, lambda config, exchange=exchange: exchange)

    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    sys.modules.pop("src.exchange.bybit_client", None)
    return importlib.import_module("src.exchange.bybit_client")


def test_fetch_ohlcv_falls_back_when_bybit_is_blocked(monkeypatch):
    bybit = FakeExchange(None, blocked=True)
    binanceus = FakeExchange(
        None,
        ohlcv=[
            [1_700_000_000_000, 100.0, 110.0, 90.0, 105.0, 12.0],
            [1_700_000_900_000, 105.0, 115.0, 95.0, 108.0, 15.0],
        ],
    )
    module = import_client_with_fake_ccxt(
        monkeypatch,
        {"bybit": bybit, "binanceus": binanceus},
    )

    client = module.BybitPublicClient(fallback_exchange_names=("binanceus",))
    df = client.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=2)

    assert bybit.ohlcv_calls == [("BTC/USDT:USDT", "15m", 2)]
    assert binanceus.ohlcv_calls == [("BTC/USDT", "15m", 2)]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert float(df.iloc[-1]["close"]) == 108.0


def test_fetch_current_price_tries_usd_symbol_for_spot_fallback(monkeypatch):
    bybit = FakeExchange(None, blocked=True)
    coinbase = FakeExchange(
        None,
        ticker={"BTC/USD": {"last": 50_000.0}},
        bad_symbols={"BTC/USDT"},
    )
    module = import_client_with_fake_ccxt(
        monkeypatch,
        {"bybit": bybit, "coinbase": coinbase},
    )

    client = module.BybitPublicClient(fallback_exchange_names=("coinbase",))
    price = client.fetch_current_price("BTC/USDT:USDT")

    assert price == 50_000.0
    assert bybit.ticker_calls == ["BTC/USDT:USDT"]
    assert coinbase.ticker_calls == ["BTC/USDT", "BTC/USD"]
