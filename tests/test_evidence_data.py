from __future__ import annotations

"""감사 가능한 시장 데이터·manifest·Bybit 백필 계층 테스트."""

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.data_manifest import (
    DataManifest,
    DataQualityBinding,
    build_data_manifest,
    file_sha256,
)
from src.data.feature_store import (
    DataQualitySummary,
    FeatureEventWrite,
    FeedHeartbeat,
    MarketFeatureStore,
)
from src.data.market_snapshot import (
    DataProvenance,
    DerivativesFeatureSnapshot,
    LiquidationRecord,
)
from src.exchange.bybit_history import BybitPublicBackfill, HistoricalMarketRecord

SYMBOL = "BTC/USDT:USDT"


def _provenance(endpoint: str = "test") -> DataProvenance:
    """Bybit swap 테스트 provenance를 반환한다."""
    return DataProvenance(
        exchange="bybit",
        market_type="swap",
        requested_symbol=SYMBOL,
        resolved_symbol=SYMBOL,
        endpoint=endpoint,
    )


def _snapshot(now: datetime, **overrides: object) -> DerivativesFeatureSnapshot:
    """구성요소별 TTL 경계 안의 파생 특징을 생성한다."""
    values: dict[str, object] = {
        "exchange_timestamp": now - timedelta(seconds=350),
        "receive_timestamp": now,
        "provenance": _provenance("snapshot"),
        "symbol": SYMBOL,
        "open_interest": 1_000_000.0,
        "current_funding_rate": 0.0001,
        "next_funding_rate": 0.0002,
        "next_funding_timestamp": now + timedelta(hours=8),
        "open_interest_timestamp": now - timedelta(seconds=350),
        "funding_timestamp": now - timedelta(seconds=50),
        "order_book_timestamp": now - timedelta(seconds=4),
        "bids": ((100.0, 2.0), (99.0, 3.0)),
        "asks": ((101.0, 2.0), (102.0, 3.0)),
        "raw": {"source": "unit"},
    }
    values.update(overrides)
    return DerivativesFeatureSnapshot(**values)  # type: ignore[arg-type]


def _event(event_id: str, when: datetime, payload: dict[str, object]) -> FeatureEventWrite:
    """일반 특징 이벤트를 반환한다."""
    return FeatureEventWrite(
        event_id=event_id,
        feature_type="open_interest",
        symbol=SYMBOL,
        payload=payload,
        provenance=_provenance("oi"),
        exchange_timestamp=when,
        receive_timestamp=when,
    )


def _quality(start: datetime, end: datetime, **overrides: object) -> DataQualitySummary:
    """DataManifest 테스트용 품질 요약을 반환한다."""
    values: dict[str, object] = {
        "dataset": "open_interest",
        "symbol": SYMBOL,
        "timestamp_axis": "receive",
        "start": start,
        "end": end,
        "event_count": 100,
        "expected_count": 100,
        "completeness": 1.0,
        "largest_gap_seconds": 300.0,
        "unresolved_gap_count": 0,
    }
    values.update(overrides)
    return DataQualitySummary(**values)  # type: ignore[arg-type]


class TestComponentFreshness:
    """구성요소별 TTL·미래·시각 편차 fail-closed 검증."""

    def test_default_component_limits_accept_independent_ages(self) -> None:
        """OI 360초·펀딩 60초·주문장 5초 내 스냅샷을 허용한다."""
        now = datetime.now(timezone.utc)
        snapshot = _snapshot(now)
        snapshot.assert_usable(now)
        assert snapshot.open_interest_max_age_seconds == 360.0
        assert snapshot.funding_max_age_seconds == 60.0
        assert snapshot.order_book_max_age_seconds == 5.0

    @pytest.mark.parametrize(
        ("field", "seconds", "message"),
        [
            ("open_interest_timestamp", 361, "open_interest"),
            ("funding_timestamp", 61, "funding"),
            ("order_book_timestamp", 6, "order_book"),
        ],
    )
    def test_each_component_expires_at_its_own_limit(
        self,
        field: str,
        seconds: int,
        message: str,
    ) -> None:
        """각 특징이 자신의 TTL을 넘으면 생성 시점부터 거부한다."""
        now = datetime.now(timezone.utc)
        overrides = {field: now - timedelta(seconds=seconds)}
        overrides["exchange_timestamp"] = min(
            overrides[field], now - timedelta(seconds=350)  # type: ignore[type-var]
        )
        with pytest.raises(ValueError, match=message):
            _snapshot(now, **overrides)

    def test_future_component_and_excessive_skew_are_rejected(self) -> None:
        """미래 timestamp와 허용치를 넘는 구성요소 편차를 거부한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="미래"):
            _snapshot(now, order_book_timestamp=now + timedelta(seconds=1))
        with pytest.raises(ValueError, match="편차"):
            _snapshot(
                now,
                exchange_timestamp=now - timedelta(seconds=350),
                max_component_skew_seconds=100.0,
            )

    def test_symbol_and_provenance_mismatch_is_rejected(self) -> None:
        """복합 특징 심볼과 provenance 심볼이 다르면 거부한다."""
        now = datetime.now(timezone.utc)
        wrong = replace(_provenance(), resolved_symbol="ETH/USDT:USDT")
        with pytest.raises(ValueError, match="심볼"):
            _snapshot(now, provenance=wrong)

    def test_snapshot_becomes_stale_at_later_decision_time(self) -> None:
        """저장 시 유효했더라도 뒤의 결정 시점에서는 다시 TTL을 검사한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(RuntimeError, match="order_book"):
            _snapshot(now).assert_usable(now + timedelta(seconds=2))


class TestFeatureStore:
    """WAL·중복 제거·원자 batch·as-of·재시작 검증."""

    def test_atomic_batch_dedup_hash_and_restart(self, tmp_path: Path) -> None:
        """batch를 원자 저장하고 hash·checkpoint를 재시작 후에도 복원한다."""
        database = tmp_path / "features.db"
        when = datetime.now(timezone.utc) - timedelta(minutes=10)
        store = MarketFeatureStore(database)
        assert store.append_feature_batch(
            [_event("one", when, {"b": 2, "a": 1}), _event("two", when, {"v": 2})]
        ) == 2
        assert store.append_feature_batch([_event("one", when, {"a": 1, "b": 2})]) == 0
        hashes = store.payload_hashes(
            "open_interest", SYMBOL, when - timedelta(seconds=1), when + timedelta(seconds=1)
        )
        expected = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
        assert hashes[0] == expected
        store.set_checkpoint("oi:btc", {"cursor": "abc", "at": when})
        assert store.get_checkpoint("oi:btc") == {"at": when.isoformat(), "cursor": "abc"}
        store.close()

        reopened = MarketFeatureStore(database)
        assert reopened.get_checkpoint("oi:btc") == {"at": when.isoformat(), "cursor": "abc"}
        assert len(reopened.payload_hashes(
            "open_interest", SYMBOL, when - timedelta(seconds=1), when + timedelta(seconds=1)
        )) == 2
        assert reopened._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        reopened.close()

    def test_invalid_batch_writes_nothing(self, tmp_path: Path) -> None:
        """batch 하나가 무효하면 검증된 앞 이벤트도 저장하지 않는다."""
        when = datetime.now(timezone.utc)
        store = MarketFeatureStore(tmp_path / "atomic.db")
        invalid = replace(_event("bad", when, {}), event_id="")
        with pytest.raises(ValueError, match="식별자"):
            store.append_feature_batch([_event("good", when, {"v": 1}), invalid])
        assert store.payload_hashes(
            "open_interest", SYMBOL, when - timedelta(seconds=1), when + timedelta(seconds=1)
        ) == ()
        store.close()

    def test_derivatives_asof_never_reads_future_receive(self, tmp_path: Path) -> None:
        """결정 cutoff 후에 수신한 특징을 as-of 조회에서 제외한다."""
        now = datetime.now(timezone.utc)
        store = MarketFeatureStore(tmp_path / "asof.db")
        older = _snapshot(now)
        future_received = _snapshot(
            now + timedelta(seconds=2),
            exchange_timestamp=now - timedelta(seconds=348),
            open_interest_timestamp=now - timedelta(seconds=348),
            funding_timestamp=now - timedelta(seconds=48),
            order_book_timestamp=now - timedelta(seconds=2),
            raw={"source": "future receive"},
        )
        assert store.save_derivatives_snapshot(older)
        assert store.save_derivatives_snapshot(future_received)
        loaded = store.load_derivatives_snapshot(SYMBOL, now)
        assert loaded is not None
        assert loaded.raw == {"source": "unit"}
        store.close()

    def test_heartbeat_gap_liquidation_and_corrupt_hash(self, tmp_path: Path) -> None:
        """heartbeat gap·청산 중복·손상 hash를 감사 데이터로 처리한다."""
        now = datetime.now(timezone.utc)
        store = MarketFeatureStore(tmp_path / "events.db")
        heartbeat = FeedHeartbeat(
            feed="allLiquidation",
            symbol=SYMBOL,
            status="gap",
            exchange_timestamp=now,
            receive_timestamp=now,
            gap_seconds=901.0,
            provenance=_provenance("heartbeat"),
            detail={"reason": "disconnect"},
        )
        assert store.record_heartbeat(heartbeat) > 0
        assert store.latest_heartbeat("allLiquidation", SYMBOL) == heartbeat
        payload = {
            "data": [
                {"T": int(now.timestamp() * 1000), "s": "BTCUSDT", "S": "Sell", "v": "2", "p": "100"}
            ]
        }
        assert store.ingest_bybit_liquidations(payload, now) == 1
        assert store.ingest_bybit_liquidations(payload, now) == 0
        records = store.load_liquidations(SYMBOL, now - timedelta(seconds=1))
        assert len(records) == 1 and records[0].side == "sell"
        store._conn.execute("UPDATE liquidation_events SET payload_sha256 = 'bad'")
        store._conn.commit()
        with pytest.raises(RuntimeError, match="SHA-256"):
            store.payload_hashes("liquidation", SYMBOL, now - timedelta(seconds=1), now + timedelta(seconds=1))
        store.close()


class TestDataManifest:
    """99% 완전성·15분 gap·원시 hash manifest 검증."""

    def test_build_eligible_manifest_with_source_hash(self, tmp_path: Path) -> None:
        """완전한 연속 데이터와 원시 파일을 해시로 고정한다."""
        start = datetime.now(timezone.utc) - timedelta(minutes=10)
        end = start + timedelta(minutes=10)
        store = MarketFeatureStore(tmp_path / "manifest.db")
        for index in range(1, 11):
            when = start + timedelta(minutes=index)
            store.append_feature(
                str(index), "open_interest", SYMBOL, {"v": index}, _provenance(), when, when
            )
        raw_file = tmp_path / "raw.jsonl"
        raw_file.write_text("evidence\n", encoding="utf-8")
        manifest = build_data_manifest(
            store,
            "open_interest",
            SYMBOL,
            start,
            end,
            60.0,
            "abc1234",
            source_files=(raw_file,),
        )
        assert manifest.evidence_eligible
        assert manifest.raw_payload_count == 10
        assert manifest.source_file_sha256[str(raw_file.resolve())] == file_sha256(raw_file)
        assert len(manifest.evidence_hash) == 64
        assert manifest.to_dict()["evidence_hash"] == manifest.evidence_hash
        restored = DataManifest.from_json(manifest.to_json(), manifest.evidence_hash)
        assert restored == manifest
        store.close()

    def test_manifest_rejects_duplicate_json_tamper_and_bad_external_hash(
        self,
    ) -> None:
        """중복 JSON key·내용 변조·외부 고정 hash 불일치를 거부한다."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        manifest = DataManifest(
            dataset="open_interest",
            symbol=SYMBOL,
            start=start,
            end=end,
            generated_at=end,
            code_commit="abc1234",
            raw_payload_root_sha256="a" * 64,
            raw_payload_count=1,
            source_file_sha256={},
            quality=_quality(start, end),
        )
        serialized = manifest.to_json()
        duplicate = serialized.replace("{", '{"dataset":"duplicate",', 1)
        with pytest.raises(ValueError, match="중복 key"):
            DataManifest.from_json(duplicate, manifest.evidence_hash)
        tampered = serialized.replace('"abc1234"', '"abc1235"')
        with pytest.raises(ValueError, match="내용과 일치"):
            DataManifest.from_json(tampered, manifest.evidence_hash)
        with pytest.raises(ValueError, match="외부 고정"):
            DataManifest.from_json(serialized, "b" * 64)

    def test_manifest_rejects_invalid_source_hash(self) -> None:
        """감사 원본 파일에 SHA-256이 아닌 digest를 허용하지 않는다."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        with pytest.raises(ValueError, match="source file hash"):
            DataManifest(
                dataset="open_interest",
                symbol=SYMBOL,
                start=start,
                end=end,
                generated_at=end,
                code_commit="abc1234",
                raw_payload_root_sha256="a" * 64,
                raw_payload_count=1,
                source_file_sha256={"raw.jsonl": "bad"},
                quality=_quality(start, end),
            )

    def test_zero_liquidations_require_complete_connection_heartbeat(self) -> None:
        """청산 0건도 완전한 공식 연결 heartbeat가 있어야 증거가 된다."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        liquidation_quality = _quality(
            start,
            end,
            dataset="liquidation",
            event_count=0,
        )
        without_binding = DataManifest(
            dataset="liquidation",
            symbol=SYMBOL,
            start=start,
            end=end,
            generated_at=end,
            code_commit="abc1234",
            raw_payload_root_sha256="a" * 64,
            raw_payload_count=0,
            source_file_sha256={},
            quality=liquidation_quality,
        )
        assert not without_binding.evidence_eligible

        heartbeat_dataset = "heartbeat:public_ws_all_liquidation_connection"
        heartbeat_quality = _quality(start, end, dataset=heartbeat_dataset)
        binding = DataQualityBinding(
            dataset=heartbeat_dataset,
            raw_payload_root_sha256="b" * 64,
            raw_payload_count=100,
            quality=heartbeat_quality,
        )
        with_binding = replace(without_binding, required_bindings=(binding,))
        assert with_binding.evidence_eligible

    @pytest.mark.parametrize(
        "quality",
        [
            {"completeness": 0.989},
            {"largest_gap_seconds": 901.0},
            {"unresolved_gap_count": 1},
        ],
    )
    def test_manifest_rejects_incomplete_or_gapped_evidence(
        self,
        quality: dict[str, object],
    ) -> None:
        """99% 미만·15분 초과·미해결 gap을 승급 증거에서 제외한다."""
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc)
        manifest = DataManifest(
            dataset="open_interest",
            symbol=SYMBOL,
            start=start,
            end=end,
            generated_at=end,
            code_commit="abc1234",
            raw_payload_root_sha256="a" * 64,
            raw_payload_count=1,
            source_file_sha256={},
            quality=_quality(start, end, **quality),
        )
        assert not manifest.evidence_eligible
        with pytest.raises(RuntimeError, match="manifest"):
            manifest.assert_evidence_eligible()


class TestBybitBackfill:
    """Bybit 공식 API 백필의 완결 봉·페이지·fail-closed 검증."""

    def test_closed_kline_filters_forming_bar_and_paginates(self) -> None:
        """형성 중 봉을 제외하고 이전 cursor 페이지를 순회한다."""
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = now - timedelta(hours=2)
        end = now - timedelta(minutes=5)
        timestamps = [
            int((end - timedelta(minutes=10)).timestamp() * 1000),
            int((end - timedelta(minutes=20)).timestamp() * 1000),
            int((start + timedelta(minutes=15)).timestamp() * 1000),
            int(start.timestamp() * 1000),
        ]
        client = MagicMock()
        client.public_get_v5_market_kline.side_effect = [
            {"retCode": 0, "result": {"list": [[str(timestamps[0]), "1", "2", "1", "2", "3", "4"], [str(timestamps[1]), "1", "2", "1", "2", "3", "4"]]}},
            {"retCode": 0, "result": {"list": [[str(timestamps[2]), "1", "2", "1", "2", "3", "4"], [str(timestamps[3]), "1", "2", "1", "2", "3", "4"]]}},
        ]
        records = BybitPublicBackfill(client).fetch_closed_klines(
            SYMBOL, "15m", start, end
        )
        assert client.public_get_v5_market_kline.call_count == 2
        assert [record.exchange_timestamp for record in records] == [
            start,
            start + timedelta(minutes=15),
            end - timedelta(minutes=20),
        ]
        assert all(record.payload["turnover"] == 4.0 for record in records)

    def test_timestamp_history_paginates_and_has_raw_hash(self) -> None:
        """펀딩 정산 이력을 timestamp cursor로 중복 없이 저장한다."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=2)
        middle = now - timedelta(days=1)
        client = MagicMock()
        client.public_get_v5_market_funding_history.side_effect = [
            {"retCode": 0, "result": {"list": [{"fundingRateTimestamp": str(int(middle.timestamp() * 1000)), "fundingRate": "0.001"}]}},
            {"retCode": 0, "result": {"list": [{"fundingRateTimestamp": str(int(start.timestamp() * 1000)), "fundingRate": "0.002"}]}},
        ]
        records = BybitPublicBackfill(client).fetch_funding_history(SYMBOL, start, now)
        assert len(records) == 2
        assert records[0].payload["funding_rate"] == 0.002
        assert len(records[0].raw_payload_sha256) == 64

    def test_backfill_never_falls_back_after_bybit_error(self) -> None:
        """Bybit API 예외를 다른 원천으로 대체하지 않고 RuntimeError로 바꿔 전파한다."""
        now = datetime.now(timezone.utc)
        client = MagicMock()
        client.public_get_v5_market_open_interest.side_effect = OSError("offline")
        with pytest.raises(RuntimeError, match="Bybit public 백필 실패"):
            BybitPublicBackfill(client).fetch_open_interest_history(
                SYMBOL, now - timedelta(hours=1), now
            )

    def test_historical_record_rejects_future_and_wrong_source(self) -> None:
        """백필 레코드의 미래 시각과 출처 혼합을 거부한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="미래"):
            HistoricalMarketRecord(
                "oi", SYMBOL, now + timedelta(seconds=1), now, _provenance(), {}
            )
        with pytest.raises(ValueError, match="bybit"):
            HistoricalMarketRecord(
                "oi",
                SYMBOL,
                now,
                now,
                replace(_provenance(), exchange="kraken"),
                {},
            )

    def test_normalized_liquidation_model_rejects_wrong_source(self) -> None:
        """청산 레코드에도 Bybit swap 출처 계약을 강제한다."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="bybit"):
            LiquidationRecord(
                exchange_timestamp=now,
                receive_timestamp=now,
                provenance=replace(_provenance(), exchange="kraken"),
                event_id="event",
                symbol=SYMBOL,
                side="buy",
                quantity=1.0,
                price=100.0,
            )
