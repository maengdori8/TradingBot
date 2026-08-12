from __future__ import annotations

"""캐리 연구용 Bybit spot·swap 12개월 데이터를 재시작 가능하게 백필한다."""

import argparse
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from src.data.feature_store import MarketFeatureStore
from src.data.market_snapshot import ensure_utc
from src.exchange.bybit_history import BybitPublicBackfill, HistoricalMarketRecord

logger = logging.getLogger(__name__)


def _parse_cli_timestamp(value: str) -> datetime:
    """YYYY-MM-DD 또는 timezone-aware ISO timestamp를 UTC로 변환한다."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("유효한 ISO 날짜/시각이 아닙니다") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class BybitResearchBackfill:
    """고정 구간을 작은 chunk로 저장하고 checkpoint 뒤에서 재개한다."""

    def __init__(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        store: MarketFeatureStore,
        history: BybitPublicBackfill | None = None,
        chunk_days: int = 7,
    ) -> None:
        """백필 구간·심볼·chunk 크기를 검증한다."""
        normalized = tuple(dict.fromkeys(symbol.strip() for symbol in symbols))
        if not normalized or any(
            not symbol.endswith("/USDT:USDT") for symbol in normalized
        ):
            raise ValueError("백필에는 Bybit USDT swap 심볼이 필요합니다")
        self._start = ensure_utc(start)
        self._end = ensure_utc(end)
        if self._end <= self._start:
            raise ValueError("백필 end는 start보다 뒤여야 합니다")
        if self._end > datetime.now(timezone.utc):
            raise ValueError("백필 end는 현재보다 미래일 수 없습니다")
        if chunk_days <= 0:
            raise ValueError("chunk_days는 양수여야 합니다")
        self._symbols = normalized
        self._store = store
        self._history = history or BybitPublicBackfill()
        self._chunk = timedelta(days=chunk_days)

    def _resume_at(self, key: str) -> datetime:
        """요청 구간이 같은 checkpoint의 다음 시작점만 허용한다."""
        checkpoint = self._store.get_checkpoint(key)
        if checkpoint is None:
            return self._start
        expected_start = self._start.isoformat()
        expected_end = self._end.isoformat()
        if (
            checkpoint.get("requested_start") != expected_start
            or checkpoint.get("requested_end") != expected_end
        ):
            raise RuntimeError(f"기존 checkpoint의 요청 구간이 다릅니다: {key}")
        raw_next = checkpoint.get("next_start")
        if not isinstance(raw_next, str):
            raise RuntimeError(f"기존 checkpoint next_start가 없습니다: {key}")
        return ensure_utc(datetime.fromisoformat(raw_next))

    def _run_chunks(
        self,
        key: str,
        fetch: Callable[[datetime, datetime], list[HistoricalMarketRecord]],
    ) -> int:
        """한 데이터 종류를 chunk마다 원자 저장하고 checkpoint를 전진시킨다."""
        cursor = self._resume_at(key)
        inserted = 0
        while cursor < self._end:
            chunk_end = min(cursor + self._chunk, self._end)
            records = fetch(cursor, chunk_end)
            inserted += self._store.save_historical_records(records)
            self._store.set_checkpoint(
                key,
                {
                    "requested_start": self._start,
                    "requested_end": self._end,
                    "next_start": chunk_end,
                    "last_chunk_records": len(records),
                    "completed": chunk_end >= self._end,
                },
            )
            cursor = chunk_end
        return inserted

    def run(self) -> int:
        """metadata와 matching spot·swap 가격, 펀딩, 5분 OI를 모두 백필한다."""
        symbol_hash = hashlib.sha256(
            "\n".join(self._symbols).encode("utf-8")
        ).hexdigest()
        metadata_key = f"research_backfill:metadata:{symbol_hash}"
        metadata_checkpoint = self._store.get_checkpoint(metadata_key)
        inserted = 0
        if metadata_checkpoint is None:
            metadata = self._history.fetch_instruments_metadata()
            inserted += self._store.save_historical_records(metadata)
            matching_spot = {
                record.symbol
                for record in metadata
                if record.symbol in self._symbols
                and record.payload.get("has_matching_spot") is True
            }
            self._store.set_checkpoint(
                metadata_key,
                {
                    "requested_start": self._start,
                    "requested_end": self._end,
                    "matching_spot_swaps": sorted(matching_spot),
                    "completed": True,
                },
            )
        else:
            if (
                metadata_checkpoint.get("requested_start") != self._start.isoformat()
                or metadata_checkpoint.get("requested_end") != self._end.isoformat()
            ):
                raise RuntimeError("기존 metadata checkpoint의 요청 구간이 다릅니다")
            raw_matching = metadata_checkpoint.get("matching_spot_swaps")
            if not isinstance(raw_matching, list) or any(
                not isinstance(item, str) for item in raw_matching
            ):
                raise RuntimeError(
                    "metadata checkpoint의 현물 대응 목록이 손상되었습니다"
                )
            matching_spot = set(raw_matching)
        for symbol in self._symbols:
            for timeframe in ("15m", "1h", "4h", "1d"):
                inserted += self._run_chunks(
                    f"research_backfill:kline:swap:{timeframe}:{symbol}",
                    lambda start, end, current=timeframe: (
                        self._history.fetch_closed_klines(
                            symbol,
                            current,
                            start,
                            end,
                        )
                    ),
                )
                if symbol in matching_spot:
                    spot_symbol = symbol.split(":", 1)[0]
                    inserted += self._run_chunks(
                        f"research_backfill:kline:spot:{timeframe}:{spot_symbol}",
                        lambda start, end, current=timeframe: (
                            self._history.fetch_closed_spot_klines(
                                spot_symbol,
                                current,
                                start,
                                end,
                            )
                        ),
                    )
            inserted += self._run_chunks(
                f"research_backfill:funding:{symbol}",
                lambda start, end: self._history.fetch_funding_history(
                    symbol,
                    start,
                    end,
                ),
            )
            inserted += self._run_chunks(
                f"research_backfill:open_interest:5min:{symbol}",
                lambda start, end: self._history.fetch_open_interest_history(
                    symbol,
                    start,
                    end,
                    interval="5min",
                ),
            )
        return inserted


def _parser() -> argparse.ArgumentParser:
    """12개월 연구 백필 CLI parser를 생성한다."""
    parser = argparse.ArgumentParser(description="Bybit 캐리 연구 데이터 백필")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=_parse_cli_timestamp, required=True)
    parser.add_argument("--end", type=_parse_cli_timestamp, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chunk-days", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """명령행 요청 구간을 백필하고 저장 건수를 로깅한다."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store = MarketFeatureStore(args.db)
    try:
        runner = BybitResearchBackfill(
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            store=store,
            chunk_days=args.chunk_days,
        )
        inserted = runner.run()
        logger.info("Bybit 연구 백필 완료: inserted=%d", inserted)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
