"""
멀티 거래소 퍼블릭 클라이언트
Bybit(swap) -> Kraken -> Coinbase fallback
선물 심볼 자동 변환 + 재시도 로직(exponential backoff)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import ccxt
import pandas as pd

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
        except Exception as exc:
            # 네트워크 이외 오류는 즉시 전파
            raise
    raise last_exc  # type: ignore[misc]


class MarketDataClient:
    """퍼블릭 시세 전용 클라이언트.

    Bybit(swap) -> Kraken -> Coinbase 순으로 fallback하며,
    각 호출에 exponential backoff 재시도 로직을 적용한다.

    거래소 클라이언트는 lazy 초기화된다. 생성자에서 load_markets()를
    호출하지 않고, 첫 번째 API 요청 시 필요한 거래소만 초기화한다.
    """

    def __init__(self) -> None:
        self._exchange_configs = EXCHANGE_CONFIGS
        self._clients: dict[str, Any] = {}
        self._initialized: set[str] = set()
        logger.info(
            "MarketDataClient 생성 (lazy 모드, 거래소 %d개 대기)",
            len(self._exchange_configs),
        )

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

        for name, cls, cfg, futures_ok in self._exchange_configs:
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
                    df.attrs["source"] = name          # 출처 태그 (실행 판정은 bybit 출처만 허용)
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

    # ------------------------------------------------------------------
    # 실행 판정용 — 거래 대상 거래소(Bybit) 전용, 폴백 없음
    # ------------------------------------------------------------------

    def _bybit_config(self) -> tuple[str, type, dict[str, Any]] | None:
        for name, cls, cfg, _futures_ok in self._exchange_configs:
            if name == "bybit":
                return name, cls, cfg
        return None

    def bybit_available(self) -> bool:
        """Bybit 퍼블릭 API 사용 가능 여부 (지역 차단 403 등이면 False).

        실행 판정(지정가 체결 / SL·TP)은 거래 대상 거래소의 가격만 써야 하므로, 봇은 이 값이 False면
        신규 주문·진입·체결·청산 판정을 모두 보류하는 '관찰 전용' 모드로 돈다.
        """
        cfg = self._bybit_config()
        if cfg is None:
            return False
        return self._ensure_client(*cfg) is not None

    def fetch_ohlcv_bybit(
        self, symbol: str, timeframe: str = "15m", limit: int = 200,
        since: datetime | None = None,
    ) -> pd.DataFrame:
        """Bybit 전용 캔들 조회 (다른 거래소 폴백 없음). 실행 판정용.

        Args:
            symbol: 거래 심볼 (예: 'BTC/USDT:USDT')
            timeframe: 캔들 주기
            limit: 조회 봉 수 (Bybit 최대 1000)
            since: 이 시각부터(포함) 오름차순 조회. None이면 최근 limit개

        Returns:
            OHLCV DataFrame (index=timestamp UTC, attrs['source']='bybit')

        Raises:
            RuntimeError: Bybit 클라이언트 초기화 불가 또는 조회 실패
        """
        cfg = self._bybit_config()
        client = self._ensure_client(*cfg) if cfg else None
        if client is None:
            raise RuntimeError("Bybit 사용 불가 — 실행 판정용 캔들 조회 실패")
        kwargs: dict[str, Any] = {"limit": limit}
        if since is not None:
            kwargs["since"] = int(since.timestamp() * 1000)
        try:
            raw = _retry_call(client.fetch_ohlcv, symbol, timeframe, **kwargs)
        except Exception as e:  # noqa: BLE001 — 호출측이 보류 처리
            raise RuntimeError(f"Bybit 캔들 조회 실패 {symbol} {timeframe}: {e}") from e
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.attrs["source"] = "bybit"
        return df

    def fetch_current_price_bybit(self, symbol: str) -> float:
        """Bybit 전용 현재가 (폴백 없음). 실행 판정/진입가 기준.

        Raises:
            RuntimeError: Bybit 사용 불가 또는 가격 없음
        """
        cfg = self._bybit_config()
        client = self._ensure_client(*cfg) if cfg else None
        if client is None:
            raise RuntimeError("Bybit 사용 불가 — 현재가 조회 실패")
        try:
            ticker = _retry_call(client.fetch_ticker, symbol)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Bybit 현재가 조회 실패 {symbol}: {e}") from e
        price = ticker.get("last") or ticker.get("close")
        if not price:
            raise RuntimeError(f"Bybit 현재가 없음 {symbol}")
        return float(price)

    def fetch_ohlcv_history(
        self, symbol: str, timeframe: str, since: datetime, max_pages: int = 40,
    ) -> pd.DataFrame:
        """Bybit 전용, since부터 현재까지 1000봉씩 페이지네이션해 이어붙인다 (판정 백필용).

        Args:
            symbol: 거래 심볼
            timeframe: 캔들 주기
            since: 시작 시각(UTC-aware)
            max_pages: 최대 페이지 수 (15m × 1000 × 40 ≈ 416일 상한)

        Returns:
            OHLCV DataFrame (중복 제거·오름차순, attrs['source']='bybit')
        """
        cfg = self._bybit_config()
        client = self._ensure_client(*cfg) if cfg else None
        if client is None:
            raise RuntimeError("Bybit 사용 불가 — 히스토리 백필 조회 실패")
        tf_ms = int(client.parse_timeframe(timeframe)) * 1000
        tf_td = timedelta(milliseconds=tf_ms)
        frames: list[pd.DataFrame] = []
        cursor = since
        now = datetime.now(timezone.utc)
        horizon = now - tf_td                 # 이 시각 이상의 봉(=마지막 닫힌 봉 또는 형성 중 봉)을 받으면 따라잡은 것
        prev_last: datetime | None = None
        reached = False
        for _ in range(max_pages):
            df = self.fetch_ohlcv_bybit(symbol, timeframe, limit=1000, since=cursor)
            if df.empty:
                break
            first = df.index[0].to_pydatetime()
            last = df.index[-1].to_pydatetime()
            if first < cursor - tf_td or first > cursor + tf_td:
                # 요청 구간 근처에서 시작하지 않은 페이지(너무 이르거나 — since 무시 — 훨씬 늦음 — 최근 봉만 반환).
                # 어느 쪽이든 "since부터 이어지는 관측"이 아니므로 신뢰하지 않고 중단한다(호출측 fail-closed).
                logger.warning("[bybit] %s 히스토리 페이지 시작(%s)이 요청(%s)과 불일치 — 중단", symbol, first, cursor)
                break
            if prev_last is not None and last <= prev_last:
                logger.warning("[bybit] %s 히스토리 페이지 진행 없음(last=%s) — 중단", symbol, last)
                break
            frames.append(df)
            prev_last = last
            if last >= horizon:
                reached = True
                break
            cursor = last + tf_td
        if not frames:
            out = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            out.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        else:
            out = pd.concat(frames)
            out = out[~out.index.duplicated(keep="last")].sort_index()
        out.attrs["source"] = "bybit"
        out.attrs["reached_horizon"] = reached
        if not reached:
            logger.warning("[bybit] %s %s 히스토리 백필이 현재까지 닿지 않음 (%d봉, since=%s) — 호출측이 보류 처리",
                           symbol, timeframe, len(out), since.isoformat())
        else:
            logger.info("[bybit] %s %s 히스토리 백필: %d봉 (since=%s)", symbol, timeframe, len(out), since.isoformat())
        return out

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

        for name, cls, cfg, futures_ok in self._exchange_configs:
            client = self._ensure_client(name, cls, cfg)
            if not client:
                continue
            candidates = [symbol] if futures_ok else _spot_symbols(symbol)
            for sym in candidates:
                try:
                    ticker = _retry_call(client.fetch_ticker, sym)
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
