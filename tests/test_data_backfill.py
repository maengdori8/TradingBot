from __future__ import annotations

"""12개월 Bybit spot·swap 연구 백필의 재시작·원자성 테스트."""

import argparse
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.backfill import BybitResearchBackfill, _parse_cli_timestamp, main
from src.data.feature_store import MarketFeatureStore
from src.data.market_snapshot import DataProvenance
from src.exchange.bybit_history import HistoricalMarketRecord

SYMBOL = "BTC/USDT:USDT"
SPOT_SYMBOL = "BTC/USDT"


def _record(
    record_type: str,
    symbol: str,
    when: datetime,
    payload: dict[str, object],
) -> HistoricalMarketRecord:
    """동일 Bybit 상품의 감사 가능한 과거 레코드를 만든다."""
    market_type = "spot" if ":" not in symbol else "swap"
    return HistoricalMarketRecord(
        record_type=record_type,
        symbol=symbol,
        exchange_timestamp=when,
        receive_timestamp=datetime.now(timezone.utc),
        provenance=DataProvenance(
            exchange="bybit",
            market_type=market_type,
            requested_symbol=symbol,
            resolved_symbol=symbol,
            endpoint=f"test_{record_type}",
        ),
        payload=payload,
    )


def _history(now: datetime, has_matching_spot: bool = True) -> MagicMock:
    """chunk 경계를 payload에 남기는 Bybit history 대역을 반환한다."""
    history = MagicMock()
    history.fetch_instruments_metadata.return_value = [
        _record(
            "instruments_metadata",
            SYMBOL,
            now,
            {"has_matching_spot": has_matching_spot},
        )
    ]

    def swap(
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[HistoricalMarketRecord]:
        """swap 완결 봉 하나를 반환한다."""
        return [_record("kline", symbol, start, {"timeframe": timeframe, "end": end})]

    def spot(
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[HistoricalMarketRecord]:
        """spot 완결 봉 하나를 반환한다."""
        return [_record("kline", symbol, start, {"timeframe": timeframe, "end": end})]

    def funding(
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[HistoricalMarketRecord]:
        """실제 정산 시각 펀딩 하나를 반환한다."""
        return [_record("funding_settlement", symbol, start, {"end": end})]

    def oi(
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[HistoricalMarketRecord]:
        """5분 OI bucket 하나를 반환한다."""
        return [_record("open_interest", symbol, start, {"end": end, "interval": interval})]

    history.fetch_closed_klines.side_effect = swap
    history.fetch_closed_spot_klines.side_effect = spot
    history.fetch_funding_history.side_effect = funding
    history.fetch_open_interest_history.side_effect = oi
    return history


class TestBybitResearchBackfill:
    """연구 백필 범위·chunk·현물 대응·checkpoint 계약 검증."""

    @pytest.mark.parametrize(
        ("symbols", "days", "message"),
        [
            ([], 1, "swap"),
            (["BTC/USDT"], 1, "swap"),
            ([SYMBOL], 0, "양수"),
        ],
    )
    def test_constructor_rejects_invalid_scope(
        self,
        tmp_path: Path,
        symbols: list[str],
        days: int,
        message: str,
    ) -> None:
        """빈·현물 심볼과 양수가 아닌 chunk를 거부한다."""
        end = datetime.now(timezone.utc) - timedelta(days=1)
        store = MarketFeatureStore(tmp_path / "invalid.db")
        with pytest.raises(ValueError, match=message):
            BybitResearchBackfill(
                symbols,
                end - timedelta(days=2),
                end,
                store,
                chunk_days=days,
            )
        store.close()

    def test_constructor_rejects_reverse_and_future_window(self, tmp_path: Path) -> None:
        """역전 구간과 미래 cutoff를 fail-closed 처리한다."""
        now = datetime.now(timezone.utc)
        store = MarketFeatureStore(tmp_path / "window.db")
        with pytest.raises(ValueError, match="뒤여야"):
            BybitResearchBackfill([SYMBOL], now, now, store)
        with pytest.raises(ValueError, match="미래"):
            BybitResearchBackfill(
                [SYMBOL],
                now - timedelta(days=1),
                now + timedelta(days=1),
                store,
            )
        store.close()

    def test_matching_spot_swap_chunks_resume_without_refetch(self, tmp_path: Path) -> None:
        """대응 현물과 swap을 같은 구간으로 백필하고 완료 checkpoint 뒤 재호출하지 않는다."""
        now = datetime.now(timezone.utc) - timedelta(hours=1)
        start = now - timedelta(days=3)
        history = _history(now)
        store = MarketFeatureStore(tmp_path / "resume.db")
        runner = BybitResearchBackfill(
            [SYMBOL, SYMBOL], start, now, store, history, chunk_days=2
        )

        assert runner.run() == 21
        assert history.fetch_closed_klines.call_count == 8
        assert history.fetch_closed_spot_klines.call_count == 8
        assert history.fetch_funding_history.call_count == 2
        assert history.fetch_open_interest_history.call_count == 2
        assert runner.run() == 0
        assert history.fetch_closed_klines.call_count == 8
        assert history.fetch_closed_spot_klines.call_count == 8
        assert store.get_checkpoint(
            f"research_backfill:kline:spot:15m:{SPOT_SYMBOL}"
        )["completed"] is True
        store.close()

    def test_missing_matching_spot_never_fetches_other_product(self, tmp_path: Path) -> None:
        """metadata가 대응 현물을 확인하지 못하면 spot 호출을 만들지 않는다."""
        end = datetime.now(timezone.utc) - timedelta(hours=1)
        history = _history(end, has_matching_spot=False)
        store = MarketFeatureStore(tmp_path / "no-spot.db")
        runner = BybitResearchBackfill(
            [SYMBOL], end - timedelta(days=1), end, store, history
        )

        assert runner.run() == 7
        history.fetch_closed_spot_klines.assert_not_called()
        store.close()

    def test_checkpoint_window_mismatch_and_corrupt_metadata_fail_closed(
        self,
        tmp_path: Path,
    ) -> None:
        """다른 구간 checkpoint와 손상된 현물 목록을 재사용하지 않는다."""
        end = datetime.now(timezone.utc) - timedelta(hours=1)
        start = end - timedelta(days=1)
        store = MarketFeatureStore(tmp_path / "checkpoint.db")
        runner = BybitResearchBackfill([SYMBOL], start, end, store, _history(end))
        store.set_checkpoint(
            f"research_backfill:funding:{SYMBOL}",
            {
                "requested_start": (start - timedelta(days=1)).isoformat(),
                "requested_end": end.isoformat(),
                "next_start": start.isoformat(),
            },
        )
        with pytest.raises(RuntimeError, match="요청 구간"):
            runner._resume_at(f"research_backfill:funding:{SYMBOL}")

        history = _history(end)
        fresh = BybitResearchBackfill([SYMBOL], start, end, store, history)
        symbol_hash = hashlib.sha256(SYMBOL.encode("utf-8")).hexdigest()
        store.set_checkpoint(
            f"research_backfill:metadata:{symbol_hash}",
            {
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
                "matching_spot_swaps": [1],
            },
        )
        with pytest.raises(RuntimeError, match="손상"):
            fresh.run()
        store.close()

    def test_cli_timestamp_and_main_close_store(self, tmp_path: Path) -> None:
        """CLI 시각을 UTC로 바꾸고 성공·실패 모두 저장소를 닫는다."""
        assert _parse_cli_timestamp("2024-01-01").tzinfo == timezone.utc
        with pytest.raises(argparse.ArgumentTypeError, match="ISO"):
            _parse_cli_timestamp("not-a-date")
        fake_store = MagicMock()
        fake_runner = MagicMock()
        fake_runner.run.return_value = 3
        with (
            patch("src.data.backfill.MarketFeatureStore", return_value=fake_store),
            patch("src.data.backfill.BybitResearchBackfill", return_value=fake_runner),
        ):
            assert main(
                [
                    "--symbols",
                    SYMBOL,
                    "--start",
                    "2024-01-01",
                    "--end",
                    "2024-01-02",
                    "--db",
                    str(tmp_path / "cli.db"),
                ]
            ) == 0
        fake_store.close.assert_called_once()
