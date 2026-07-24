from __future__ import annotations

# 출처 검증과 재시도를 지원하는 퍼블릭 시세 클라이언트.

import logging
import time
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any

import ccxt
import pandas as pd

from src.data.market_snapshot import (
    DataProvenance,
    DerivativesFeatureSnapshot,
    MarketSnapshot,
)

if TYPE_CHECKING:
    from src.data.feature_store import MarketFeatureStore

logger = logging.getLogger(__name__)

# (이름, ccxt 클래스, 설정, 선물심볼 지원 여부)
EXCHANGE_CONFIGS: list[tuple[str, type, dict[str, Any], bool]] = [
    ("bybit", ccxt.bybit, {"options": {"defaultType": "swap"}}, True),
    ("kraken", ccxt.kraken, {}, False),
    ("coinbase", ccxt.coinbase, {}, False),
]

# 재시도 설정
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 1.0  # 초 단위


def _spot_symbols(symbol: str) -> list[str]:
    """선물 심볼을 현물 심볼 후보로 변환한다.

    Args:
        symbol: 선물 심볼 (예: 'BTC/USDT:USDT')

    Returns:
        현물 심볼 후보 리스트 (예: ['BTC/USDT', 'BTC/USD'])
    """
    base = symbol.split(":")[0]
    coin, quote = base.split("/")
    result = [base]
    if quote == "USDT":
        result.append(f"{coin}/USD")
    return result


def _retry_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    """API 호출을 exponential backoff 방식으로 재시도한다.

    Args:
        func: 호출할 함수
        *args: 위치 인자
        **kwargs: 키워드 인자

    Returns:
        함수 호출 결과

    Raises:
        Exception: 최대 재시도 횟수 초과 시 마지막 예외를 전파한다.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "재시도 %d/%d (%.1fs 후): %s — %s",
                    attempt, MAX_RETRIES, delay,
                    type(exc).__name__, str(exc)[:120],
                )
                time.sleep(delay)
            else:
                logger.error(
                    "최대 재시도 초과 (%d회): %s — %s",
                    MAX_RETRIES, type(exc).__name__, str(exc)[:200],
                )
        except Exception:
            # 네트워크 이외 오류는 즉시 전파
            raise
    raise last_exc  # type: ignore[misc]


def _validated_max_age(max_age_seconds: float) -> float:
    """최신성 한도를 0 이상의 유한한 초 단위 값으로 검증한다."""
    if (
        isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, Real)
        or not isfinite(float(max_age_seconds))
        or max_age_seconds < 0
    ):
        raise ValueError("max_age_seconds는 0 이상의 유한한 숫자여야 합니다")
    return float(max_age_seconds)


def _fresh_component_time(
    component: str,
    raw_timestamp: Any,
    received: datetime,
    max_age_seconds: float,
) -> datetime:
    """복합 스냅샷 구성요소의 거래소 timestamp와 최신성을 검증한다."""
    if (
        isinstance(raw_timestamp, bool)
        or not isinstance(raw_timestamp, Real)
        or not isfinite(float(raw_timestamp))
    ):
        raise RuntimeError(
            f"{component}에 유효한 numeric timestamp가 없습니다"
        )
    component_time = datetime.fromtimestamp(
        float(raw_timestamp) / 1000.0,
        timezone.utc,
    )
    if component_time > received:
        raise RuntimeError(
            f"{component} timestamp가 검증 시점보다 미래입니다"
        )
    component_age = (received - component_time).total_seconds()
    if component_age > max_age_seconds:
        raise RuntimeError(
            f"오래된 {component} 데이터: age={component_age:.3f}s"
        )
    return component_time


def _normalized_order_book_levels(
    order_book: dict[str, Any],
    side: str,
) -> tuple[tuple[float, float], ...]:
    """주문장 한쪽 호가를 양수 가격·수량 튜플로 정규화한다."""
    raw_levels = order_book.get(side)
    if not isinstance(raw_levels, list) or not raw_levels:
        raise RuntimeError(f"orderbook {side} 호가가 비어 있습니다")
    levels: list[tuple[float, float]] = []
    for raw_level in raw_levels:
        if not isinstance(raw_level, (list, tuple)) or len(raw_level) < 2:
            raise RuntimeError(f"orderbook {side} 호가 형식이 잘못되었습니다")
        price, quantity = raw_level[0], raw_level[1]
        if (
            isinstance(price, bool)
            or isinstance(quantity, bool)
            or not isinstance(price, Real)
            or not isinstance(quantity, Real)
            or not isfinite(float(price))
            or not isfinite(float(quantity))
            or float(price) <= 0
            or float(quantity) <= 0
        ):
            raise RuntimeError(f"orderbook {side} 호가 값이 유효하지 않습니다")
        levels.append((float(price), float(quantity)))
    return tuple(levels)


class MarketDataClient:
    """퍼블릭 시세 전용 클라이언트.

    현물은 Bybit -> Kraken -> Coinbase 순으로 fallback할 수 있다.
    Bybit 무기한 선물 심볼은 기본적으로 Bybit swap 데이터만 허용해
    다른 거래소 현물이 신호 입력에 섞이지 않도록 fail-closed 한다.

    거래소 클라이언트는 lazy 초기화된다. 생성자에서 load_markets()를
    호출하지 않고, 첫 번째 API 요청 시 필요한 거래소만 초기화한다.
    """

    def __init__(self, strict_derivatives: bool = True) -> None:
        """시세 클라이언트를 생성한다.

        Args:
            strict_derivatives: 파생상품 심볼을 요청 거래소·상품으로만
                제한할지 여부. 운영 신호에서는 True를 유지해야 한다.
        """
        self._exchange_configs = EXCHANGE_CONFIGS
        self._clients: dict[str, Any] = {}
        self._initialized: set[str] = set()
        self._strict_derivatives = strict_derivatives
        self._public_streams: list[Any] = []
        logger.info(
            "MarketDataClient 생성 (lazy 모드, 거래소 %d개, strict=%s)",
            len(self._exchange_configs),
            strict_derivatives,
        )

    def _eligible_configs(
        self,
        symbol: str,
    ) -> list[tuple[str, type, dict[str, Any], bool]]:
        """심볼의 상품 종류에 맞는 거래소 설정만 반환한다."""
        is_derivative = ":" in symbol
        if is_derivative and self._strict_derivatives:
            configs = [
                item
                for item in self._exchange_configs
                if item[0] == "bybit" and item[3]
            ]
            if not configs:
                raise RuntimeError(
                    f"Bybit swap 데이터 소스가 없습니다: {symbol}"
                )
            return configs
        return list(self._exchange_configs)

    def _ensure_client(self, name: str, cls: type, cfg: dict[str, Any]) -> Any | None:
        """거래소 클라이언트를 lazy 초기화한다.

        이미 초기화 시도한 거래소는 재시도하지 않으며,
        성공 시 클라이언트를, 실패 시 None을 반환한다.

        Args:
            name: 거래소 이름 (예: 'bybit')
            cls: ccxt 거래소 클래스
            cfg: 거래소 설정 dict

        Returns:
            초기화된 ccxt 클라이언트 또는 None
        """
        if name in self._initialized:
            return self._clients.get(name)
        self._initialized.add(name)
        try:
            client = cls(cfg)
            logger.info("[%s] 마켓 로딩 시작 (lazy)...", name)
            _retry_call(client.load_markets)
            self._clients[name] = client
            logger.info("[%s] 초기화 완료 (마켓 %d개)", name, len(client.markets))
            return client
        except ccxt.BaseError as exc:
            logger.warning(
                "[%s] 초기화 실패 (ccxt): %s — %s",
                name, type(exc).__name__, str(exc)[:200],
            )
            return None
        except Exception as exc:
            logger.warning(
                "[%s] 초기화 실패: %s — %s",
                name, type(exc).__name__, str(exc)[:200],
            )
            return None

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "15m", limit: int = 200
    ) -> pd.DataFrame:
        """캔들 데이터를 조회한다 (fallback + 재시도 포함).

        Args:
            symbol: 거래 심볼 (예: 'BTC/USDT:USDT')
            timeframe: 캔들 주기 (예: '15m', '1h', '4h')
            limit: 조회할 캔들 수

        Returns:
            OHLCV DataFrame (index=timestamp UTC)

        Raises:
            RuntimeError: 모든 거래소에서 조회 실패 시
        """
        errors: list[str] = []

        for name, cls, cfg, futures_ok in self._eligible_configs(symbol):
            client = self._ensure_client(name, cls, cfg)
            if not client:
                continue
            candidates = [symbol] if futures_ok else _spot_symbols(symbol)
            for sym in candidates:
                try:
                    raw = _retry_call(
                        client.fetch_ohlcv, sym, timeframe, limit=limit
                    )
                    df = pd.DataFrame(
                        raw,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="ms", utc=True
                    )
                    df = df.set_index("timestamp")
                    df.attrs["provenance"] = {
                        "exchange": name,
                        "market_type": "swap" if futures_ok and ":" in sym else "spot",
                        "requested_symbol": symbol,
                        "resolved_symbol": sym,
                        "endpoint": "fetch_ohlcv",
                        "receive_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if name != "bybit":
                        logger.info(
                            "[fallback] %s -> %s/%s (%d candles)",
                            symbol, name, sym, len(df),
                        )
                    return df
                except Exception as exc:
                    err_msg = f"[{name}/{sym}] {type(exc).__name__}: {str(exc)[:120]}"
                    errors.append(err_msg)
                    logger.debug("OHLCV 실패: %s", err_msg)

        error_detail = "; ".join(errors) if errors else "클라이언트 없음"
        raise RuntimeError(f"모든 거래소 OHLCV 실패: {symbol} — {error_detail}")

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """현재 시세(ticker)를 조회한다 (fallback + 재시도 포함).

        Args:
            symbol: 거래 심볼

        Returns:
            ticker dict (ccxt 표준)

        Raises:
            RuntimeError: 모든 거래소에서 조회 실패 시
        """
        errors: list[str] = []

        for name, cls, cfg, futures_ok in self._eligible_configs(symbol):
            client = self._ensure_client(name, cls, cfg)
            if not client:
                continue
            candidates = [symbol] if futures_ok else _spot_symbols(symbol)
            for sym in candidates:
                try:
                    ticker = _retry_call(client.fetch_ticker, sym)
                    ticker = dict(ticker)
                    ticker["_provenance"] = {
                        "exchange": name,
                        "market_type": "swap" if futures_ok and ":" in sym else "spot",
                        "requested_symbol": symbol,
                        "resolved_symbol": sym,
                        "endpoint": "fetch_ticker",
                        "receive_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if name != "bybit":
                        logger.info(
                            "[fallback] ticker %s -> %s/%s", symbol, name, sym
                        )
                    return ticker
                except Exception as exc:
                    err_msg = f"[{name}/{sym}] {type(exc).__name__}: {str(exc)[:120]}"
                    errors.append(err_msg)
                    logger.debug("ticker 실패: %s", err_msg)

        error_detail = "; ".join(errors) if errors else "클라이언트 없음"
        raise RuntimeError(f"ticker 조회 실패: {symbol} — {error_detail}")

    def health_check(self) -> dict[str, bool | str]:
        """각 거래소의 연결 상태를 확인한다.

        이미 초기화된 거래소만 실제 API 호출로 상태를 점검하며,
        아직 초기화되지 않은 거래소는 'not_initialized'로 반환한다.
        lazy loading 패턴과 호환 — 이 메서드가 거래소를 초기화하지 않는다.

        Returns:
            거래소별 연결 상태 dict
            (예: {"bybit": True, "kraken": False, "coinbase": "not_initialized"})
        """
        status: dict[str, bool | str] = {}

        for name, _cls, _cfg, futures_ok in self._exchange_configs:
            if name not in self._initialized:
                status[name] = "not_initialized"
                continue

            client = self._clients.get(name)
            if client is None:
                # 초기화 시도했으나 실패한 거래소
                status[name] = False
                continue

            try:
                test_symbol = "BTC/USDT:USDT" if futures_ok else "BTC/USDT"
                _retry_call(client.fetch_ticker, test_symbol)
                status[name] = True
                logger.debug("[health] %s: OK", name)
            except Exception as exc:
                status[name] = False
                logger.warning(
                    "[health] %s: FAIL — %s: %s",
                    name, type(exc).__name__, str(exc)[:120],
                )

        return status

    def fetch_current_price(self, symbol: str) -> float:
        """현재가를 조회한다.

        Args:
            symbol: 거래 심볼

        Returns:
            현재 가격 (float)
        """
        return float(self.fetch_ticker(symbol)["last"])

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 25,
    ) -> dict[str, Any]:
        """주문장을 조회하고 데이터 출처를 함께 반환한다.

        Args:
            symbol: 거래 심볼.
            limit: 매수·매도 호가별 최대 깊이.

        Returns:
            ccxt 주문장 딕셔너리와 ``_provenance`` 메타데이터.

        Raises:
            RuntimeError: 허용된 거래소에서 주문장을 가져오지 못한 경우.
        """
        errors: list[str] = []
        for name, cls, cfg, futures_ok in self._eligible_configs(symbol):
            client = self._ensure_client(name, cls, cfg)
            if client is None:
                continue
            candidates = [symbol] if futures_ok else _spot_symbols(symbol)
            for resolved_symbol in candidates:
                try:
                    order_book = dict(
                        _retry_call(
                            client.fetch_order_book,
                            resolved_symbol,
                            limit,
                        )
                    )
                    order_book["_provenance"] = {
                        "exchange": name,
                        "market_type": (
                            "swap"
                            if futures_ok and ":" in resolved_symbol
                            else "spot"
                        ),
                        "requested_symbol": symbol,
                        "resolved_symbol": resolved_symbol,
                        "endpoint": "fetch_order_book",
                        "receive_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    return order_book
                except Exception as exc:
                    message = (
                        f"[{name}/{resolved_symbol}] "
                        f"{type(exc).__name__}: {str(exc)[:120]}"
                    )
                    errors.append(message)
                    logger.debug("orderbook 실패: %s", message)
        detail = "; ".join(errors) if errors else "클라이언트 없음"
        raise RuntimeError(f"orderbook 조회 실패: {symbol} — {detail}")

    def fetch_market_snapshot(
        self,
        symbol: str,
        order_book_limit: int = 25,
        max_age_seconds: float = 5.0,
    ) -> MarketSnapshot:
        """ticker와 주문장을 출처·최신성 정보가 있는 스냅샷으로 반환한다.

        Args:
            symbol: 거래 심볼.
            order_book_limit: 주문장 조회 깊이.
            max_age_seconds: 주문 결정에 허용할 최대 데이터 나이.

        Returns:
            출처와 두 종류 타임스탬프를 포함한 시장 스냅샷.

        Raises:
            RuntimeError: ticker와 주문장의 출처가 다르거나 오래된 경우.
        """
        ticker = self.fetch_ticker(symbol)
        order_book = self.fetch_order_book(symbol, order_book_limit)
        ticker_source = ticker.get("_provenance", {})
        book_source = order_book.get("_provenance", {})
        source_keys = ("exchange", "market_type", "resolved_symbol")
        if any(ticker_source.get(key) != book_source.get(key) for key in source_keys):
            raise RuntimeError("ticker와 orderbook 데이터 출처가 일치하지 않습니다")

        received = datetime.now(timezone.utc)
        max_age = _validated_max_age(max_age_seconds)
        component_times: dict[str, datetime] = {}
        for component, raw_timestamp in (
            ("ticker", ticker.get("timestamp")),
            ("orderbook", order_book.get("timestamp")),
        ):
            component_times[component] = _fresh_component_time(
                component,
                raw_timestamp,
                received,
                max_age,
            )
        component_skew = abs(
            (
                component_times["ticker"]
                - component_times["orderbook"]
            ).total_seconds()
        )
        if component_skew > max_age:
            raise RuntimeError(
                "ticker와 orderbook timestamp 편차가 허용 범위를 초과합니다: "
                f"skew={component_skew:.3f}s"
            )
        exchange_time = min(component_times.values())
        last = ticker.get("last")
        if last is None:
            raise RuntimeError("ticker에 last 가격이 없습니다")
        provenance = DataProvenance(
            exchange=str(ticker_source["exchange"]),
            market_type=str(ticker_source["market_type"]),
            requested_symbol=symbol,
            resolved_symbol=str(ticker_source["resolved_symbol"]),
            endpoint="fetch_ticker+fetch_order_book",
        )
        snapshot = MarketSnapshot(
            exchange_timestamp=exchange_time,
            receive_timestamp=received,
            provenance=provenance,
            symbol=symbol,
            last=float(last),
            bid=float(ticker["bid"]) if ticker.get("bid") is not None else None,
            ask=float(ticker["ask"]) if ticker.get("ask") is not None else None,
            bids=tuple(
                (float(level[0]), float(level[1]))
                for level in order_book.get("bids", [])
            ),
            asks=tuple(
                (float(level[0]), float(level[1]))
                for level in order_book.get("asks", [])
            ),
            max_age_seconds=max_age,
            raw={"ticker": ticker, "order_book": order_book},
        )
        expected_market = "swap" if ":" in symbol else provenance.market_type
        snapshot.assert_usable(
            expected_exchange="bybit" if ":" in symbol else provenance.exchange,
            expected_market_type=expected_market,
        )
        return snapshot

    def fetch_derivatives_feature_snapshot(
        self,
        symbol: str,
        order_book_limit: int = 25,
        max_age_seconds: float = 5.0,
    ) -> DerivativesFeatureSnapshot:
        """동일 Bybit swap의 OI·펀딩·주문장을 시점 보존해 조회한다.

        Args:
            symbol: ccxt 형식 Bybit USDT 무기한 선물 심볼.
            order_book_limit: 주문장 조회 깊이.
            max_age_seconds: 각 입력 및 입력 간 시각 편차 허용 한도.

        Returns:
            세 비가격 입력과 provenance를 포함한 복합 스냅샷.

        Raises:
            RuntimeError: Bybit 데이터가 누락·오래됨·불일치 상태인 경우.
        """
        if ":" not in symbol:
            raise ValueError("선물 특징 조회에는 swap 심볼이 필요합니다")
        if (
            isinstance(order_book_limit, bool)
            or not isinstance(order_book_limit, int)
            or order_book_limit <= 0
        ):
            raise ValueError("order_book_limit는 0보다 큰 정수여야 합니다")
        max_age = _validated_max_age(max_age_seconds)
        client = self._ensure_client(
            "bybit",
            ccxt.bybit,
            {"options": {"defaultType": "swap"}},
        )
        if client is None:
            raise RuntimeError("Bybit swap 클라이언트를 초기화할 수 없습니다")
        try:
            open_interest = dict(_retry_call(client.fetch_open_interest, symbol))
            funding = dict(_retry_call(client.fetch_funding_rate, symbol))
            order_book = dict(
                _retry_call(
                    client.fetch_order_book,
                    symbol,
                    order_book_limit,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Bybit 선물 특징 조회 실패: {symbol} — "
                f"{type(exc).__name__}: {str(exc)[:160]}"
            ) from exc

        for component, payload in (
            ("open_interest", open_interest),
            ("funding", funding),
            ("orderbook", order_book),
        ):
            if payload.get("symbol") != symbol:
                raise RuntimeError(
                    f"{component} 응답 심볼이 요청과 일치하지 않습니다"
                )
        received = datetime.now(timezone.utc)
        component_times = {
            "open_interest": _fresh_component_time(
                "open_interest",
                open_interest.get("timestamp"),
                received,
                max_age,
            ),
            "funding": _fresh_component_time(
                "funding",
                funding.get("timestamp"),
                received,
                max_age,
            ),
            "orderbook": _fresh_component_time(
                "orderbook",
                order_book.get("timestamp"),
                received,
                max_age,
            ),
        }
        component_skew = (
            max(component_times.values()) - min(component_times.values())
        ).total_seconds()
        if component_skew > max_age:
            raise RuntimeError(
                "OI·펀딩·orderbook timestamp 편차가 허용 범위를 초과합니다: "
                f"skew={component_skew:.3f}s"
            )

        raw_open_interest = open_interest.get("openInterestAmount")
        raw_funding_rate = funding.get("fundingRate")
        for label, value in (
            ("openInterestAmount", raw_open_interest),
            ("fundingRate", raw_funding_rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
            ):
                raise RuntimeError(f"Bybit {label} 값이 누락되었거나 유효하지 않습니다")
        if float(raw_open_interest) < 0:
            raise RuntimeError("Bybit openInterestAmount는 음수일 수 없습니다")

        next_funding_raw = funding.get("nextFundingTimestamp")
        if next_funding_raw is None:
            next_funding_raw = funding.get("fundingTimestamp")
        if (
            isinstance(next_funding_raw, bool)
            or not isinstance(next_funding_raw, Real)
            or not isfinite(float(next_funding_raw))
        ):
            raise RuntimeError("Bybit 다음 펀딩 timestamp가 없습니다")
        next_funding_time = datetime.fromtimestamp(
            float(next_funding_raw) / 1000.0,
            timezone.utc,
        )
        if next_funding_time <= received:
            raise RuntimeError("Bybit 다음 펀딩 timestamp가 이미 지났습니다")

        next_funding_rate: float | None = None
        raw_next_funding_rate = funding.get("nextFundingRate")
        if raw_next_funding_rate is not None:
            if (
                isinstance(raw_next_funding_rate, bool)
                or not isinstance(raw_next_funding_rate, Real)
                or not isfinite(float(raw_next_funding_rate))
            ):
                raise RuntimeError("Bybit nextFundingRate 값이 유효하지 않습니다")
            next_funding_rate = float(raw_next_funding_rate)

        snapshot = DerivativesFeatureSnapshot(
            exchange_timestamp=min(component_times.values()),
            receive_timestamp=received,
            provenance=DataProvenance(
                exchange="bybit",
                market_type="swap",
                requested_symbol=symbol,
                resolved_symbol=symbol,
                endpoint=(
                    "fetch_open_interest+fetch_funding_rate+fetch_order_book"
                ),
            ),
            symbol=symbol,
            open_interest=float(raw_open_interest),
            current_funding_rate=float(raw_funding_rate),
            next_funding_rate=next_funding_rate,
            next_funding_timestamp=next_funding_time,
            open_interest_timestamp=component_times["open_interest"],
            funding_timestamp=component_times["funding"],
            order_book_timestamp=component_times["orderbook"],
            bids=_normalized_order_book_levels(order_book, "bids"),
            asks=_normalized_order_book_levels(order_book, "asks"),
            max_age_seconds=max_age,
            raw={
                "open_interest": open_interest,
                "funding": funding,
                "order_book": order_book,
            },
        )
        snapshot.assert_usable(received)
        return snapshot

    def start_public_liquidation_stream(
        self,
        symbols: list[str],
        store: MarketFeatureStore,
    ) -> Any:
        """Bybit public all-liquidation 스트림을 시점 보존 저장소에 연결한다.

        Args:
            symbols: ccxt 형식 Bybit swap 심볼 목록.
            store: liquidation 레코드를 저장할 feature 저장소.

        Returns:
            실행 중인 pybit public WebSocket 객체.
        """
        if not symbols or any(":" not in symbol for symbol in symbols):
            raise ValueError("public liquidation 스트림에는 swap 심볼이 필요합니다")
        try:
            from pybit.unified_trading import WebSocket
        except ImportError as exc:
            raise RuntimeError(
                "public liquidation stream에는 pybit 설치가 필요합니다"
            ) from exc
        bybit_symbols = [
            symbol.split(":")[0].replace("/", "")
            for symbol in symbols
        ]
        stream = WebSocket(testnet=False, channel_type="linear")
        stream.all_liquidation_stream(
            symbol=bybit_symbols,
            callback=store.ingest_bybit_liquidations,
        )
        self._public_streams.append(stream)
        return stream

    def fetch_top_symbols(
        self,
        limit: int = 30,
        quote: str = "USDT",
        min_volume_usdt: float = 5_000_000.0,
    ) -> list[str]:
        """Bybit USDT 무기한 선물 중 거래량 상위 심볼을 조회한다.

        24시간 명목 거래대금(quoteVolume) 기준으로 정렬하여 유동성이 높은
        심볼만 반환한다. 슬리피지가 큰 저유동성 코인을 자동 배제한다.

        Args:
            limit: 반환할 최대 심볼 수
            quote: 견적 통화 (기본 USDT)
            min_volume_usdt: 최소 24h 거래대금 필터

        Returns:
            'BTC/USDT:USDT' 형식 심볼 리스트 (거래량 내림차순).
            조회 실패 시 빈 리스트.
        """
        client = self._ensure_client(
            "bybit", ccxt.bybit, {"options": {"defaultType": "swap"}}
        )
        if client is None:
            logger.warning("Bybit 클라이언트 없음 — top symbols 조회 불가")
            return []

        try:
            tickers = _retry_call(client.fetch_tickers)
        except Exception as exc:
            logger.warning(
                "top symbols 조회 실패: %s — %s",
                type(exc).__name__, str(exc)[:160],
            )
            return []

        scored: list[tuple[str, float]] = []
        for sym, t in tickers.items():
            # linear USDT 무기한만: 'BTC/USDT:USDT'
            if not sym.endswith(f":{quote}"):
                continue
            if f"/{quote}:" not in sym:
                continue
            vol = t.get("quoteVolume") or 0.0
            try:
                vol = float(vol)
            except (TypeError, ValueError):
                vol = 0.0
            if vol < min_volume_usdt:
                continue
            scored.append((sym, vol))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in scored[:limit]]
        logger.info(
            "거래량 상위 심볼 %d개 선정 (전체 %d개 중 필터 통과 %d개)",
            len(result), len(tickers), len(scored),
        )
        return result


# 하위 호환
BybitPublicClient = MarketDataClient
