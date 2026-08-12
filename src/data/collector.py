from __future__ import annotations

"""Bybit 동일 venue 증거 데이터를 중단 후 재개 가능한 형태로 수집한다."""

import argparse
import logging
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from src.data.feature_store import FeedHeartbeat, MarketFeatureStore
from src.data.market_snapshot import DataProvenance, ensure_utc
from src.exchange.bybit_client import MarketDataClient
from src.exchange.bybit_history import BybitPublicBackfill, HistoricalMarketRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectorPolicy:
    """공개 market collector의 주기와 fail-closed 최신성 정책."""

    snapshot_interval_seconds: float = 5.0
    heartbeat_interval_seconds: float = 30.0
    backfill_interval_seconds: float = 300.0
    metadata_interval_seconds: float = 86400.0
    open_interest_max_age_seconds: float = 360.0
    funding_max_age_seconds: float = 60.0
    order_book_max_age_seconds: float = 5.0
    max_component_skew_seconds: float = 360.0
    clock_settle_tolerance_seconds: float = 5.0

    def __post_init__(self) -> None:
        """수집 주기와 최신성 한도가 모두 양수인지 검증한다."""
        values = (
            self.snapshot_interval_seconds,
            self.heartbeat_interval_seconds,
            self.backfill_interval_seconds,
            self.metadata_interval_seconds,
            self.open_interest_max_age_seconds,
            self.funding_max_age_seconds,
            self.order_book_max_age_seconds,
            self.max_component_skew_seconds,
        )
        if any(value <= 0 for value in values):
            raise ValueError("collector 주기와 최신성 한도는 모두 양수여야 합니다")
        if self.clock_settle_tolerance_seconds < 0:
            raise ValueError("collector clock settle 한도는 음수일 수 없습니다")


class BybitEvidenceCollector:
    """REST snapshot·공식 백필·청산 WebSocket을 24시간 수집한다."""

    def __init__(
        self,
        symbols: Sequence[str],
        store: MarketFeatureStore,
        market_client: MarketDataClient | None = None,
        history_client: BybitPublicBackfill | None = None,
        policy: CollectorPolicy | None = None,
    ) -> None:
        """수집 의존성과 동일 venue swap 심볼을 고정한다."""
        normalized = tuple(dict.fromkeys(symbol.strip() for symbol in symbols))
        if not normalized or any(
            not symbol.endswith("/USDT:USDT") for symbol in normalized
        ):
            raise ValueError("collector에는 Bybit USDT swap 심볼이 필요합니다")
        self._symbols = normalized
        self._store = store
        self._market = market_client or MarketDataClient(strict_derivatives=True)
        self._history = history_client or BybitPublicBackfill()
        self._policy = policy or CollectorPolicy()
        self._stop = threading.Event()
        self._liquidation_stream: Any | None = None
        self._last_heartbeat_monotonic = 0.0
        self._last_backfill_monotonic = 0.0
        self._last_metadata_monotonic = 0.0

    def stop(self) -> None:
        """다음 대기 지점에서 수집 loop를 종료하도록 요청한다."""
        self._stop.set()

    def _start_liquidation_stream(self) -> None:
        """청산 스트림을 시작하고 연결 실패를 heartbeat로 남긴다."""
        try:
            self._liquidation_stream = self._market.start_public_liquidation_stream(
                list(self._symbols),
                self._store,
            )
            self._record_stream_heartbeat("connected")
        except Exception as exc:
            self._liquidation_stream = None
            self._record_stream_heartbeat(
                "disconnected",
                {"error": f"{type(exc).__name__}: {str(exc)[:160]}"},
            )
            logger.error("Bybit 청산 stream 시작 실패: %s", exc)

    def _stream_connected(self) -> bool:
        """pybit 구현 차이를 고려해 현재 연결 여부를 보수적으로 판정한다."""
        stream = self._liquidation_stream
        if stream is None:
            return False
        checker = getattr(stream, "is_connected", None)
        try:
            if callable(checker):
                return bool(checker())
            if checker is not None:
                return bool(checker)
            socket = getattr(getattr(stream, "ws", None), "sock", None)
            return bool(socket is not None and getattr(socket, "connected", False))
        except Exception:
            return False

    def _close_liquidation_stream(self) -> None:
        """현재 WebSocket을 가능한 경우 정상 종료한다."""
        if self._liquidation_stream is None:
            return
        closer = getattr(self._liquidation_stream, "exit", None)
        try:
            if callable(closer):
                closer()
        except Exception as exc:
            logger.warning("Bybit 청산 stream 종료 실패: %s", exc)
        finally:
            self._liquidation_stream = None

    def _record_stream_heartbeat(
        self,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """심볼별 연결 heartbeat와 이전 heartbeat 이후 gap을 기록한다."""
        received = datetime.now(timezone.utc)
        for symbol in self._symbols:
            previous = self._store.latest_heartbeat(
                "public_ws_all_liquidation_connection",
                symbol,
            )
            gap_seconds = (
                (received - previous.receive_timestamp).total_seconds()
                if previous is not None
                else 0.0
            )
            effective_status = status
            if status == "connected" and gap_seconds > max(
                self._policy.heartbeat_interval_seconds * 2, 900.0
            ):
                effective_status = "gap"
            self._store.record_heartbeat(
                FeedHeartbeat(
                    feed="public_ws_all_liquidation_connection",
                    symbol=symbol,
                    status=effective_status,
                    exchange_timestamp=received,
                    receive_timestamp=received,
                    gap_seconds=max(gap_seconds, 0.0),
                    provenance=DataProvenance(
                        exchange="bybit",
                        market_type="swap",
                        requested_symbol=symbol,
                        resolved_symbol=symbol,
                        endpoint="public_ws_all_liquidation_connection",
                    ),
                    detail=detail or {},
                )
            )

    def collect_snapshots_once(self) -> dict[str, bool]:
        """모든 심볼의 OI·펀딩·25레벨 주문장을 한 번 수집한다."""
        results: dict[str, bool] = {}
        for symbol in self._symbols:
            try:
                snapshot = self._market.fetch_derivatives_feature_snapshot(
                    symbol,
                    order_book_limit=25,
                    open_interest_max_age_seconds=(
                        self._policy.open_interest_max_age_seconds
                    ),
                    funding_max_age_seconds=self._policy.funding_max_age_seconds,
                    order_book_max_age_seconds=(
                        self._policy.order_book_max_age_seconds
                    ),
                    max_component_skew_seconds=(
                        self._policy.max_component_skew_seconds
                    ),
                    clock_settle_tolerance_seconds=(
                        self._policy.clock_settle_tolerance_seconds
                    ),
                )
                inserted = self._store.save_derivatives_snapshot(snapshot)
                self._store.set_checkpoint(
                    f"collector:derivatives_flow:{symbol}",
                    {
                        "receive_timestamp": snapshot.receive_timestamp,
                        "exchange_timestamp": snapshot.exchange_timestamp,
                        "inserted": inserted,
                    },
                )
                results[symbol] = True
            except Exception as exc:
                results[symbol] = False
                logger.error(
                    "파생 특징 snapshot 수집 실패: %s — %s: %s",
                    symbol,
                    type(exc).__name__,
                    str(exc)[:160],
                )
        return results

    def backfill_once(self, now: datetime | None = None) -> int:
        """checkpoint 이후 닫힌 캔들·펀딩·5분 OI를 공식 REST로 보충한다."""
        cutoff = ensure_utc(now or datetime.now(timezone.utc))
        inserted = 0
        for symbol in self._symbols:
            for timeframe in ("15m", "1h", "4h", "1d"):
                key = f"backfill:kline:{timeframe}:{symbol}"
                start = self._checkpoint_start(key, cutoff - timedelta(days=2))
                records = self._history.fetch_closed_klines(
                    symbol,
                    timeframe,
                    start,
                    cutoff,
                )
                inserted += self._save_backfill_page(key, records, cutoff)
            funding_key = f"backfill:funding:{symbol}"
            funding_start = self._checkpoint_start(
                funding_key,
                cutoff - timedelta(days=7),
            )
            inserted += self._save_backfill_page(
                funding_key,
                self._history.fetch_funding_history(symbol, funding_start, cutoff),
                cutoff,
            )
            oi_key = f"backfill:open_interest:{symbol}"
            oi_start = self._checkpoint_start(oi_key, cutoff - timedelta(days=2))
            inserted += self._save_backfill_page(
                oi_key,
                self._history.fetch_open_interest_history(
                    symbol,
                    oi_start,
                    cutoff,
                    interval="5min",
                ),
                cutoff,
            )
        return inserted

    def _checkpoint_start(self, key: str, default: datetime) -> datetime:
        """마지막 성공 exchange timestamp 또는 지정 기본 시작점을 반환한다."""
        checkpoint = self._store.get_checkpoint(key)
        if checkpoint is None or not checkpoint.get("exchange_timestamp"):
            return ensure_utc(default)
        parsed = datetime.fromisoformat(str(checkpoint["exchange_timestamp"]))
        return ensure_utc(parsed)

    def _save_backfill_page(
        self,
        checkpoint_key: str,
        records: Sequence[HistoricalMarketRecord],
        cutoff: datetime,
    ) -> int:
        """한 종류의 백필 batch를 원자 저장한 뒤 checkpoint를 전진시킨다."""
        inserted = self._store.save_historical_records(records)
        last_exchange = (
            max(record.exchange_timestamp for record in records) if records else cutoff
        )
        self._store.set_checkpoint(
            checkpoint_key,
            {
                "exchange_timestamp": ensure_utc(last_exchange),
                "receive_timestamp": datetime.now(timezone.utc),
                "record_count": len(records),
                "inserted": inserted,
            },
        )
        return inserted

    def collect_metadata_once(self) -> int:
        """상품 상장·현물 대응·주문 규칙 snapshot을 저장한다."""
        records = self._history.fetch_instruments_metadata()
        inserted = self._store.save_historical_records(records)
        self._store.set_checkpoint(
            "collector:instrument_metadata",
            {
                "receive_timestamp": datetime.now(timezone.utc),
                "record_count": len(records),
                "inserted": inserted,
            },
        )
        return inserted

    def run_once(self) -> dict[str, bool]:
        """테스트·스케줄러용 단일 snapshot 수집 cycle을 실행한다."""
        return self.collect_snapshots_once()

    def run_forever(self) -> None:
        """종료 요청까지 연결 복구·snapshot·백필을 계속 실행한다."""
        self._start_liquidation_stream()
        try:
            while not self._stop.is_set():
                cycle_started = monotonic()
                self.collect_snapshots_once()
                now_mono = monotonic()
                if (
                    now_mono - self._last_heartbeat_monotonic
                    >= self._policy.heartbeat_interval_seconds
                ):
                    if not self._stream_connected():
                        self._record_stream_heartbeat("disconnected")
                        self._close_liquidation_stream()
                        self._start_liquidation_stream()
                    else:
                        self._record_stream_heartbeat("connected")
                    self._last_heartbeat_monotonic = now_mono
                if (
                    now_mono - self._last_backfill_monotonic
                    >= self._policy.backfill_interval_seconds
                ):
                    try:
                        self.backfill_once()
                    except Exception as exc:
                        logger.error("Bybit 공식 REST 백필 실패: %s", exc)
                    self._last_backfill_monotonic = now_mono
                if (
                    now_mono - self._last_metadata_monotonic
                    >= self._policy.metadata_interval_seconds
                ):
                    try:
                        self.collect_metadata_once()
                    except Exception as exc:
                        logger.error("Bybit 상품 metadata 수집 실패: %s", exc)
                    self._last_metadata_monotonic = now_mono
                elapsed = monotonic() - cycle_started
                wait_seconds = max(
                    self._policy.snapshot_interval_seconds - elapsed,
                    0.0,
                )
                self._stop.wait(wait_seconds)
        finally:
            self._close_liquidation_stream()


def _parser() -> argparse.ArgumentParser:
    """collector CLI parser를 생성한다."""
    parser = argparse.ArgumentParser(description="Bybit 연구 증거 데이터 수집기")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="ccxt swap 심볼 목록 (예: BTC/USDT:USDT)",
    )
    parser.add_argument("--db", type=Path, default=None, help="SQLite DB 경로")
    parser.add_argument(
        "--once",
        action="store_true",
        help="한 번만 snapshot을 수집하고 종료",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """명령행에서 1회 또는 24시간 collector를 실행한다."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store = MarketFeatureStore(args.db)
    collector = BybitEvidenceCollector(args.symbols, store)
    if args.once:
        results = collector.run_once()
        store.close()
        return 0 if all(results.values()) else 1

    def _request_stop(_signum: int, _frame: Any) -> None:
        """SIGINT/SIGTERM을 안전한 loop 종료 요청으로 바꾼다."""
        collector.stop()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        collector.run_forever()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
