from __future__ import annotations

"""공식 Bybit public REST만 사용하는 시점 보존 백필 primitive."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import ccxt

from src.data.market_snapshot import DataProvenance, ensure_utc

logger = logging.getLogger(__name__)

_TIMEFRAME_INTERVALS: dict[str, tuple[str, int]] = {
    "5m": ("5", 5 * 60 * 1000),
    "15m": ("15", 15 * 60 * 1000),
    "1h": ("60", 60 * 60 * 1000),
    "4h": ("240", 4 * 60 * 60 * 1000),
    "1d": ("D", 24 * 60 * 60 * 1000),
}


def _canonical_hash(payload: dict[str, Any]) -> str:
    """원시 payload의 canonical SHA-256을 계산한다."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _milliseconds(value: datetime) -> int:
    """UTC datetime을 epoch milliseconds로 변환한다."""
    return int(ensure_utc(value).timestamp() * 1000)


def _swap_market_id(symbol: str) -> str:
    """ccxt USDT 무기한 선물 심볼을 Bybit market id로 변환한다."""
    normalized = symbol.strip().upper()
    if not normalized.endswith("/USDT:USDT"):
        raise ValueError("Bybit 백필은 USDT 무기한 선물 심볼만 허용합니다")
    return normalized.split(":", 1)[0].replace("/", "")


@dataclass(frozen=True)
class HistoricalMarketRecord:
    """백필 API에서 받은 한 시점의 수정 불가능한 시장 레코드."""

    record_type: str
    symbol: str
    exchange_timestamp: datetime
    receive_timestamp: datetime
    provenance: DataProvenance
    payload: dict[str, Any]
    raw_payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """레코드의 시각·출처·payload 불변식을 검증한다."""
        exchanged = ensure_utc(self.exchange_timestamp)
        received = ensure_utc(self.receive_timestamp)
        if exchanged > received:
            raise ValueError("백필 exchange_timestamp가 수신 시각보다 미래입니다")
        if self.provenance.exchange != "bybit":
            raise ValueError("백필 출처는 bybit여야 합니다")
        if self.provenance.market_type not in {"swap", "spot"}:
            raise ValueError("백필 상품 종류가 잘못되었습니다")
        if not self.record_type.strip() or not self.symbol.strip():
            raise ValueError("백필 record_type과 symbol은 비어 있을 수 없습니다")
        object.__setattr__(self, "raw_payload_sha256", _canonical_hash(self.payload))


class BybitPublicBackfill:
    """Bybit 공식 v5 public API의 재시작 가능한 백필 호출 집합."""

    def __init__(self, client: Any | None = None) -> None:
        """주입된 ccxt Bybit 또는 public 전용 기본 클라이언트를 사용한다."""
        self._client = client or ccxt.bybit(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )

    def _call(
        self,
        method_name: str,
        path: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """ccxt implicit method를 우선 사용하고 같은 Bybit path로만 보완한다."""
        method = getattr(self._client, method_name, None)
        try:
            response = (
                method(params)
                if callable(method)
                else self._client.request(path, "public", "GET", params)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Bybit public 백필 실패: {path} — "
                f"{type(exc).__name__}: {str(exc)[:160]}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"Bybit {path} 응답이 object가 아닙니다")
        ret_code = response.get("retCode")
        if ret_code not in {None, 0, "0"}:
            raise RuntimeError(
                f"Bybit {path} 오류: retCode={ret_code}, "
                f"retMsg={str(response.get('retMsg') or '')[:120]}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Bybit {path} 응답에 result가 없습니다")
        return response

    @staticmethod
    def _validate_window(
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, datetime]:
        """백필 구간을 UTC로 정규화하고 미래 cutoff를 거부한다."""
        started = ensure_utc(start)
        ended = ensure_utc(end)
        if ended <= started:
            raise ValueError("백필 end는 start보다 뒤여야 합니다")
        now = datetime.now(timezone.utc)
        if ended > now:
            raise ValueError("백필 end는 현재보다 미래일 수 없습니다")
        return started, ended

    def fetch_closed_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[HistoricalMarketRecord]:
        """완전히 닫힌 캔들의 OHLCV와 turnover를 페이지네이션한다."""
        started, ended = self._validate_window(start, end)
        if timeframe not in _TIMEFRAME_INTERVALS:
            raise ValueError(f"지원하지 않는 Bybit timeframe입니다: {timeframe}")
        interval, interval_ms = _TIMEFRAME_INTERVALS[timeframe]
        market_id = _swap_market_id(symbol)
        started_ms = _milliseconds(started)
        cutoff_ms = _milliseconds(ended)
        cursor_end = cutoff_ms
        records: dict[int, HistoricalMarketRecord] = {}
        while cursor_end >= started_ms:
            response = self._call(
                "public_get_v5_market_kline",
                "v5/market/kline",
                {
                    "category": "linear",
                    "symbol": market_id,
                    "interval": interval,
                    "start": started_ms,
                    "end": cursor_end,
                    "limit": 1000,
                },
            )
            received = datetime.now(timezone.utc)
            raw_rows = response["result"].get("list")
            if not isinstance(raw_rows, list) or not raw_rows:
                break
            page_timestamps: list[int] = []
            for raw in raw_rows:
                if not isinstance(raw, list) or len(raw) < 7:
                    raise RuntimeError("Bybit kline row 형식이 잘못되었습니다")
                timestamp_ms = int(raw[0])
                page_timestamps.append(timestamp_ms)
                if timestamp_ms < started_ms or timestamp_ms + interval_ms > cutoff_ms:
                    continue
                payload = {
                    "timeframe": timeframe,
                    "open": float(raw[1]),
                    "high": float(raw[2]),
                    "low": float(raw[3]),
                    "close": float(raw[4]),
                    "volume": float(raw[5]),
                    "turnover": float(raw[6]),
                    "raw": raw,
                }
                records[timestamp_ms] = self._record(
                    "kline",
                    symbol,
                    timestamp_ms,
                    received,
                    "public_v5_market_kline",
                    payload,
                )
            oldest = min(page_timestamps)
            if oldest <= started_ms or oldest >= cursor_end:
                break
            cursor_end = oldest - 1
        return [records[key] for key in sorted(records)]

    def fetch_funding_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[HistoricalMarketRecord]:
        """실제 정산 시각 기준 펀딩 이력을 페이지네이션한다."""
        return self._fetch_timestamped_history(
            symbol=symbol,
            start=start,
            end=end,
            method_name="public_get_v5_market_funding_history",
            path="v5/market/funding/history",
            interval=None,
            limit=200,
            timestamp_key="fundingRateTimestamp",
            record_type="funding_settlement",
            normalize=lambda raw: {
                "funding_rate": float(raw["fundingRate"]),
                "raw": raw,
            },
        )

    def fetch_open_interest_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "5min",
    ) -> list[HistoricalMarketRecord]:
        """공식 OI bucket을 출처 혼합이나 보간 없이 페이지네이션한다."""
        allowed = {"5min", "15min", "30min", "1h", "4h", "1d"}
        if interval not in allowed:
            raise ValueError(f"지원하지 않는 OI interval입니다: {interval}")
        return self._fetch_timestamped_history(
            symbol=symbol,
            start=start,
            end=end,
            method_name="public_get_v5_market_open_interest",
            path="v5/market/open-interest",
            interval=interval,
            limit=200,
            timestamp_key="timestamp",
            record_type="open_interest",
            normalize=lambda raw: {
                "interval": interval,
                "open_interest": float(raw["openInterest"]),
                "raw": raw,
            },
        )

    def _fetch_timestamped_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        method_name: str,
        path: str,
        interval: str | None,
        limit: int,
        timestamp_key: str,
        record_type: str,
        normalize: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[HistoricalMarketRecord]:
        """timestamp 기반 Bybit 이력을 뒤쪽부터 누락 없이 조회한다."""
        started, ended = self._validate_window(start, end)
        market_id = _swap_market_id(symbol)
        started_ms = _milliseconds(started)
        cursor_end = _milliseconds(ended)
        records: dict[int, HistoricalMarketRecord] = {}
        while cursor_end >= started_ms:
            params: dict[str, Any] = {
                "category": "linear",
                "symbol": market_id,
                "startTime": started_ms,
                "endTime": cursor_end,
                "limit": limit,
            }
            if interval is not None:
                params["intervalTime"] = interval
            response = self._call(method_name, path, params)
            received = datetime.now(timezone.utc)
            raw_rows = response["result"].get("list")
            if not isinstance(raw_rows, list) or not raw_rows:
                break
            page_timestamps: list[int] = []
            for item in raw_rows:
                if not isinstance(item, dict) or timestamp_key not in item:
                    raise RuntimeError(f"Bybit {record_type} row 형식이 잘못되었습니다")
                timestamp_ms = int(item[timestamp_key])
                page_timestamps.append(timestamp_ms)
                if timestamp_ms < started_ms or timestamp_ms > cursor_end:
                    continue
                records[timestamp_ms] = self._record(
                    record_type,
                    symbol,
                    timestamp_ms,
                    received,
                    f"public_{path.replace('/', '_')}",
                    normalize(dict(item)),
                )
            oldest = min(page_timestamps)
            if oldest <= started_ms or oldest >= cursor_end:
                break
            cursor_end = oldest - 1
        return [records[key] for key in sorted(records)]

    def fetch_instruments_metadata(self) -> list[HistoricalMarketRecord]:
        """USDT 무기한 규칙과 대응 Bybit 현물 존재 여부를 현재 시점에 저장한다."""
        spot_ids = self._fetch_instrument_ids("spot")
        swap_rows = self._fetch_instrument_rows("linear")
        received = datetime.now(timezone.utc)
        records: list[HistoricalMarketRecord] = []
        for raw in swap_rows:
            market_id = str(raw.get("symbol") or "")
            if not market_id.endswith("USDT"):
                continue
            contract_type = str(raw.get("contractType") or "")
            if contract_type and contract_type != "LinearPerpetual":
                continue
            base = market_id[:-4]
            symbol = f"{base}/USDT:USDT"
            lot = raw.get("lotSizeFilter") or {}
            price = raw.get("priceFilter") or {}
            launch_ms = int(raw.get("launchTime") or 0)
            payload = {
                "listing_timestamp": (
                    datetime.fromtimestamp(launch_ms / 1000.0, timezone.utc).isoformat()
                    if launch_ms > 0
                    else None
                ),
                "has_matching_spot": market_id in spot_ids,
                "tick_size": float(price.get("tickSize") or 0.0),
                "qty_step": float(lot.get("qtyStep") or 0.0),
                "min_order_qty": float(lot.get("minOrderQty") or 0.0),
                "min_notional": float(lot.get("minNotionalValue") or 0.0),
                "funding_interval_minutes": int(raw.get("fundingInterval") or 0),
                "raw": raw,
            }
            records.append(
                HistoricalMarketRecord(
                    record_type="instrument_metadata",
                    symbol=symbol,
                    exchange_timestamp=received,
                    receive_timestamp=received,
                    provenance=DataProvenance(
                        exchange="bybit",
                        market_type="swap",
                        requested_symbol=symbol,
                        resolved_symbol=symbol,
                        endpoint="public_v5_market_instruments_info",
                    ),
                    payload=payload,
                )
            )
        return sorted(records, key=lambda item: item.symbol)

    def _fetch_instrument_ids(self, category: str) -> set[str]:
        """상품 category의 market id 집합을 반환한다."""
        return {
            str(row.get("symbol"))
            for row in self._fetch_instrument_rows(category)
            if row.get("symbol")
        }

    def _fetch_instrument_rows(self, category: str) -> list[dict[str, Any]]:
        """instruments-info cursor를 끝까지 순회한다."""
        cursor = ""
        rows: list[dict[str, Any]] = []
        seen_cursors: set[str] = set()
        while True:
            params: dict[str, Any] = {"category": category, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            response = self._call(
                "public_get_v5_market_instruments_info",
                "v5/market/instruments-info",
                params,
            )
            raw_rows = response["result"].get("list")
            if not isinstance(raw_rows, list):
                raise RuntimeError("Bybit instruments list 형식이 잘못되었습니다")
            for row in raw_rows:
                if not isinstance(row, dict):
                    raise RuntimeError("Bybit instrument row 형식이 잘못되었습니다")
                rows.append(dict(row))
            next_cursor = str(response["result"].get("nextPageCursor") or "")
            if not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise RuntimeError("Bybit instruments cursor가 진행되지 않습니다")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return rows

    @staticmethod
    def _record(
        record_type: str,
        symbol: str,
        timestamp_ms: int,
        received: datetime,
        endpoint: str,
        payload: dict[str, Any],
    ) -> HistoricalMarketRecord:
        """동일한 Bybit swap provenance를 가진 레코드를 생성한다."""
        exchanged = datetime.fromtimestamp(timestamp_ms / 1000.0, timezone.utc)
        return HistoricalMarketRecord(
            record_type=record_type,
            symbol=symbol,
            exchange_timestamp=exchanged,
            receive_timestamp=received,
            provenance=DataProvenance(
                exchange="bybit",
                market_type="swap",
                requested_symbol=symbol,
                resolved_symbol=symbol,
                endpoint=endpoint,
            ),
            payload=payload,
        )
