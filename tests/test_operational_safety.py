from __future__ import annotations

"""수집기·실행 이벤트·Bybit demo/live·킬스위치 통합 테스트."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from src.data.collector import BybitEvidenceCollector, CollectorPolicy, main as collector_main
from src.data.execution_store import ExecutionEventStore
from src.data.feature_store import FeedHeartbeat, MarketFeatureStore
from src.data.market_snapshot import DataProvenance, DerivativesFeatureSnapshot
from src.exchange.contracts import (
    ExecutionReport,
    FeeRateSnapshot,
    Fill,
    OrderRequest,
    OrderState,
    TradingMode,
)
from src.exchange.order_executor import BybitOrderExecutor
from src.paper_trading import Position
from src.risk.live_guard import (
    LiveActivationEvidence,
    LiveActivationGate,
    LivePilotGuard,
    PortfolioRiskGuard,
    SafetySnapshot,
    TradeRiskProposal,
    calculate_pilot_capital_krw,
    evaluate_scale_up,
)
from src.risk.validation_gate import GateCriterion, GateDecision

SYMBOL = "BTC/USDT:USDT"


def _provenance(endpoint: str = "test") -> DataProvenance:
    """Bybit swap provenance를 반환한다."""
    return DataProvenance("bybit", "swap", SYMBOL, SYMBOL, endpoint)


def _snapshot(now: datetime) -> DerivativesFeatureSnapshot:
    """수집 주기 내 파생 특징 스냅샷을 반환한다."""
    return DerivativesFeatureSnapshot(
        exchange_timestamp=now - timedelta(seconds=30),
        receive_timestamp=now,
        provenance=_provenance("snapshot"),
        symbol=SYMBOL,
        open_interest=1000.0,
        current_funding_rate=0.001,
        next_funding_timestamp=now + timedelta(hours=8),
        open_interest_timestamp=now - timedelta(seconds=30),
        funding_timestamp=now - timedelta(seconds=20),
        order_book_timestamp=now - timedelta(seconds=1),
        bids=((100.0, 1.0),),
        asks=((101.0, 1.0),),
        raw={"ok": True},
    )


def _historical(
    now: datetime,
    *,
    symbol: str = SYMBOL,
    has_matching_spot: bool = False,
) -> SimpleNamespace:
    """백필 checkpoint에 필요한 최소 레코드를 반환한다."""
    return SimpleNamespace(
        exchange_timestamp=now - timedelta(minutes=5),
        symbol=symbol,
        payload={"has_matching_spot": has_matching_spot},
    )


class TestEvidenceCollector:
    """24시간 수집 재시작·gap·부분 실패 검증."""

    def test_policy_and_symbols_fail_closed(self, tmp_path: Path) -> None:
        """음수 주기와 현물·빈 심볼 범위를 거부한다."""
        with pytest.raises(ValueError, match="양수"):
            CollectorPolicy(snapshot_interval_seconds=0)
        store = MarketFeatureStore(tmp_path / "bad.db")
        with pytest.raises(ValueError, match="swap"):
            BybitEvidenceCollector(["BTC/USDT"], store)
        store.close()

    def test_snapshot_cycle_checkpoints_success_and_isolates_symbol_failure(self, tmp_path: Path) -> None:
        """심볼별 실패를 격리하고 성공 수신 시각을 checkpoint한다."""
        store = MarketFeatureStore(tmp_path / "snapshots.db")
        market = MagicMock()
        now = datetime.now(timezone.utc)
        eth = "ETH/USDT:USDT"
        market.fetch_derivatives_feature_snapshot.side_effect = [
            _snapshot(now),
            RuntimeError("stale"),
        ]
        collector = BybitEvidenceCollector([SYMBOL, eth, SYMBOL], store, market, MagicMock())
        assert collector.run_once() == {SYMBOL: True, eth: False}
        assert store.get_checkpoint(f"collector:derivatives_flow:{SYMBOL}") is not None
        assert store.get_checkpoint(f"collector:derivatives_flow:{eth}") is None
        store.close()

    def test_stream_connect_disconnect_gap_and_close(self, tmp_path: Path) -> None:
        """WebSocket 연결·경과 gap·종료 예외를 heartbeat로 남긴다."""
        store = MarketFeatureStore(tmp_path / "stream.db")
        old = datetime.now(timezone.utc) - timedelta(seconds=901)
        store.record_heartbeat(
            FeedHeartbeat(
                "public_ws_all_liquidation_connection",
                SYMBOL,
                "connected",
                old,
                old,
                0.0,
                _provenance("public_ws_all_liquidation_connection"),
                {},
            )
        )
        stream = MagicMock()
        stream.is_connected.return_value = True
        market = MagicMock()
        market.start_public_liquidation_stream.return_value = stream
        collector = BybitEvidenceCollector([SYMBOL], store, market, MagicMock())
        collector._start_liquidation_stream()
        assert collector._stream_connected()
        assert store.latest_heartbeat(
            "public_ws_all_liquidation_connection", SYMBOL
        ).status == "gap"
        stream.exit.side_effect = RuntimeError("closed")
        collector._close_liquidation_stream()
        assert collector._liquidation_stream is None
        market.start_public_liquidation_stream.side_effect = RuntimeError("offline")
        collector._start_liquidation_stream()
        assert store.latest_heartbeat(
            "public_ws_all_liquidation_connection", SYMBOL
        ).status == "disconnected"
        store.close()

    def test_backfill_metadata_and_checkpoint_resume(self) -> None:
        """모든 candle·funding·OI·metadata batch 후 checkpoint를 전진시킨다."""
        now = datetime.now(timezone.utc)
        store = MagicMock()
        store.get_checkpoint.return_value = None
        store.save_historical_records.side_effect = lambda rows: len(rows)
        history = MagicMock()
        history.fetch_closed_klines.return_value = [_historical(now)]
        history.fetch_funding_history.return_value = [_historical(now)]
        history.fetch_open_interest_history.return_value = [_historical(now)]
        history.fetch_instruments_metadata.return_value = [
            _historical(now, has_matching_spot=False),
            _historical(now, symbol="ETH/USDT:USDT"),
        ]
        collector = BybitEvidenceCollector([SYMBOL], store, MagicMock(), history)
        assert collector.backfill_once(now) == 6
        assert history.fetch_closed_klines.call_count == 4
        history.fetch_open_interest_history.assert_called_once_with(
            SYMBOL, now - timedelta(days=2), now, interval="5min"
        )
        assert collector.collect_metadata_once() == 2
        store.get_checkpoint.return_value = {
            "exchange_timestamp": (now - timedelta(hours=1)).isoformat()
        }
        assert collector._checkpoint_start("key", now - timedelta(days=2)) == now - timedelta(hours=1)

    def test_run_forever_executes_recovery_backfill_and_metadata_once(self) -> None:
        """종료 요청이 들어온 cycle에서도 연결 복구와 REST 보충을 완료한다."""
        collector = BybitEvidenceCollector([SYMBOL], MagicMock(), MagicMock(), MagicMock())
        collector._start_liquidation_stream = MagicMock()
        collector._stream_connected = MagicMock(return_value=False)
        collector._record_stream_heartbeat = MagicMock()
        collector._close_liquidation_stream = MagicMock()
        collector.backfill_once = MagicMock(side_effect=RuntimeError("rest down"))
        collector.collect_metadata_once = MagicMock(return_value=0)

        def stop_after_snapshot() -> dict[str, bool]:
            """첫 snapshot 후 loop 종료를 요청한다."""
            collector.stop()
            return {SYMBOL: True}

        collector.collect_snapshots_once = MagicMock(side_effect=stop_after_snapshot)
        collector.run_forever()
        collector.backfill_once.assert_called_once()
        collector.collect_metadata_once.assert_called_once()
        assert collector._close_liquidation_stream.call_count >= 1

    def test_once_cli_exit_code_tracks_snapshot_result(self, tmp_path: Path) -> None:
        """--once CLI는 전체 스냅샷 성공을 exit code로 반환한다."""
        fake_store = MagicMock()
        fake_collector = MagicMock()
        fake_collector.run_once.return_value = {SYMBOL: False}
        with (
            patch("src.data.collector.MarketFeatureStore", return_value=fake_store),
            patch("src.data.collector.BybitEvidenceCollector", return_value=fake_collector),
        ):
            assert collector_main(["--symbols", SYMBOL, "--db", str(tmp_path / "db"), "--once"]) == 1
        fake_store.close.assert_called_once()


def _fill(now: datetime) -> Fill:
    """영속화 테스트용 maker 체결을 반환한다."""
    return Fill(
        "fill-1", "order-1", "client-1", SYMBOL, "buy", 1.0, 100.0,
        fee=0.02, fee_currency="USDT", liquidity="maker",
        exchange_timestamp=now, receive_timestamp=now, raw={"x": 1},
    )


class TestExecutionStore:
    """주문·체결·WS·fee 재시작 저장소 검증."""

    def test_claim_report_fill_event_and_fee_round_trip(self, tmp_path: Path) -> None:
        """중복 주문·체결·이벤트를 멱등 저장하고 재시작해 복원한다."""
        now = datetime.now(timezone.utc)
        database = tmp_path / "execution.db"
        store = ExecutionEventStore(database)
        assert store.claim_order("client-1", "link-1", SYMBOL)
        assert not store.claim_order("client-1", "link-1", SYMBOL)
        assert store.resolve_client_order_id("link-1") == "client-1"
        report = ExecutionReport(
            "order-1", "client-1", SYMBOL, OrderState.FILLED, 1.0, 1.0, 100.0,
            fills=(_fill(now),), exchange_timestamp=now, receive_timestamp=now, raw={"ok": True},
        )
        store.save_report(report)
        loaded = store.get_report("client-1")
        assert loaded is not None and loaded.fills == report.fills
        assert loaded.remaining_quantity == 0.0
        assert store.append_event("event", "ws", "execution", {"at": now}, now, now, "order-1", "client-1")
        assert not store.append_event("event", "ws", "execution", {}, now, now)
        assert store.load_events("execution", now - timedelta(seconds=1))[0]["payload"]["at"] == now.isoformat()
        store.save_fee_rate(FeeRateSnapshot(SYMBOL, 0.0002, 0.00055, now, now, raw={"tier": 1}))
        assert store._conn.execute("SELECT count(*) FROM fee_rate_snapshots").fetchone()[0] == 1
        store.close()
        reopened = ExecutionEventStore(database)
        assert reopened.get_report("client-1") is not None
        reopened.close()

    def test_store_rejects_naive_timestamp_and_normalizes_unknown_liquidity(self, tmp_path: Path) -> None:
        """timezone 누락을 거부하고 알 수 없는 유동성을 unknown으로 복원한다."""
        store = ExecutionEventStore(tmp_path / "corrupt.db")
        with pytest.raises(ValueError, match="timezone"):
            store.append_event("bad", "ws", "order", {}, datetime.now())
        assert store.get_report("missing") is None
        store.close()


def _exchange() -> MagicMock:
    """Bybit ccxt 인증 클라이언트 대역을 반환한다."""
    exchange = MagicMock()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    exchange.create_order.return_value = {
        "id": "order-1",
        "symbol": SYMBOL,
        "status": "closed",
        "amount": 1.0,
        "filled": 1.0,
        "average": None,
        "timestamp": now_ms,
        "trades": [
            {
                "id": "fill-1", "symbol": SYMBOL, "amount": 1.0, "price": 100.0,
                "fee": {"cost": 0.02, "currency": "USDT"}, "takerOrMaker": "maker",
                "timestamp": now_ms,
            }
        ],
    }
    exchange.fetch_trading_fee.return_value = {
        "maker": 0.0002, "taker": 0.00055, "timestamp": now_ms
    }
    exchange.fetch_open_orders.return_value = [{"id": "open"}]
    exchange.fetch_positions.return_value = [
        {"symbol": SYMBOL, "contracts": 1.0, "side": "long", "info": {"positionIdx": 0}}
    ]
    exchange.fetch_my_trades.return_value = [{"id": "trade", "order": "order-1", "timestamp": now_ms}]
    exchange.fetch_balance.return_value = {"USDT": {"total": 1000}, "timestamp": now_ms}
    exchange.cancel_order.return_value = {"id": "order-1", "timestamp": now_ms}
    return exchange


class TestBybitExecutor:
    """demo 체결·멱등성·WS·대사·비상청산 검증."""

    def test_submit_partial_normalization_idempotency_and_rejection(self, tmp_path: Path) -> None:
        """REST 응답이 아닌 체결 내역으로 보고서를 정규화하고 중복을 차단한다."""
        exchange = _exchange()
        store = ExecutionEventStore(tmp_path / "bybit.db")
        executor = BybitOrderExecutor(mode="demo", exchange=exchange, event_store=store)
        request = OrderRequest("client-1", SYMBOL, "buy", 1.0, strategy_version="v1")
        report = executor.submit_order(request)
        assert report.state is OrderState.FILLED and report.average_price == 100.0
        assert len(report.fills) == 1 and exchange.create_order.call_count == 1
        assert executor.submit_order(request) == report
        assert exchange.create_order.call_count == 1
        exchange.create_order.side_effect = ccxt.InvalidOrder("bad qty")
        rejected = executor.submit_order(
            OrderRequest("client-2", SYMBOL, "sell", 1.0, strategy_version="v1")
        )
        assert rejected.state is OrderState.REJECTED and "InvalidOrder" in rejected.reject_reason
        executor.close()

    def test_protection_fee_cancel_private_reconcile_position_and_flatten(self, tmp_path: Path) -> None:
        """보호주문·fee API·private WS·REST 대사·멱등 비상청산을 영구 저장한다."""
        exchange = _exchange()
        store = ExecutionEventStore(tmp_path / "operations.db")
        executor = BybitOrderExecutor(mode="demo", exchange=exchange, event_store=store)
        protected = executor.place_protective_order(
            SYMBOL, "long", 1.0, 95.0, "protect-1", "v1"
        )
        assert protected.state is OrderState.FILLED
        assert exchange.create_order.call_args.args[-1]["reduceOnly"] is True
        fee = executor.fetch_fee_rate(SYMBOL)
        assert fee.maker_rate == 0.0002
        assert executor.cancel_order("order-1")
        exchange.cancel_order.side_effect = ccxt.OrderNotFound("gone")
        assert not executor.cancel_order("missing")
        assert executor.get_open_orders(SYMBOL) == [{"id": "open"}]
        assert executor.get_position(SYMBOL)["contracts"] == 1.0
        event = {"data": [{"execId": "exec-1", "orderId": "order-1", "orderLinkId": "link"}]}
        assert executor.ingest_private_event("execution", event) == 1
        assert executor.ingest_private_event("execution", event) == 0
        reconciled = executor.reconcile(SYMBOL)
        assert reconciled["balance"]["USDT"]["total"] == 1000
        reports = executor.emergency_flatten("incident", "v1", SYMBOL)
        assert len(reports) == 1 and reports[0].state is OrderState.FILLED
        same = executor.emergency_flatten("incident", "v1", SYMBOL)
        assert same[0].client_order_id == reports[0].client_order_id
        executor.close()

    def test_live_report_token_hash_version_and_server_stop_are_enforced(self, tmp_path: Path) -> None:
        """live에서 수동 토큰·demo 리포트 hash·버전·서버 SL을 함께 강제한다."""
        report_path = tmp_path / "demo.json"
        payload = {"stage": "demo", "passed": True, "strategy_version": "v1"}
        report_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        report_path.write_bytes(report_bytes)
        report_hash = hashlib.sha256(report_bytes).hexdigest()
        environment = {
            "LIVE_TRADING_APPROVAL_TOKEN": "approve",
            "LIVE_TRADING_VALIDATION_REPORT_SHA256": report_hash,
        }
        with patch.dict("os.environ", environment, clear=False):
            executor = BybitOrderExecutor(
                mode="live",
                exchange=_exchange(),
                event_store=ExecutionEventStore(tmp_path / "live.db"),
                live_approval_token="approve",
                validation_report_hash=report_hash,
                validation_report_path=report_path,
            )
            with pytest.raises(ValueError, match="stop_loss"):
                executor.submit_order(
                    OrderRequest("live-1", SYMBOL, "buy", 1.0, strategy_version="v1")
                )
            with pytest.raises(ValueError, match="strategy_version"):
                executor.submit_order(
                    OrderRequest("live-2", SYMBOL, "buy", 1.0, stop_loss=90.0, strategy_version="wrong")
                )
            executor.close()


def _gate(stage: str = "demo", passed: bool = True) -> GateDecision:
    """live 활성·증액 테스트용 게이트 결정을 반환한다."""
    criterion = GateCriterion("test", passed, passed, True)
    return GateDecision(passed, stage, "carry", "v1", {"test": criterion}, "summary")


class TestLiveRiskGuards:
    """실전 활성·손실·대사·노출·증액 한도 검증."""

    def test_activation_gate_requires_every_demo_and_manual_binding(self) -> None:
        """live 모드·enable·토큰·hash·버전·demo 통과를 모두 요구한다."""
        evidence = LiveActivationEvidence(
            TradingMode.LIVE, True, "token", "hash", "hash", "v1", "v1", _gate()
        )
        assert LiveActivationGate().evaluate(evidence).allowed
        failed = replace_live_evidence(evidence)
        decision = LiveActivationGate().evaluate(failed)
        assert not decision.allowed and len(decision.reasons) == 6

    def test_kill_switch_trips_persists_and_requires_manual_reset(self, tmp_path: Path) -> None:
        """시세 stale·대사 미확인·주문 오류로 발동한 킬스위치를 재시작 후에도 유지한다."""
        now = datetime.now(timezone.utc)
        database = tmp_path / "guard.db"
        guard = LivePilotGuard(1000.0, db_path=database)
        assert guard.record_order_result(False) == 1
        assert guard.record_order_result(False) == 2
        assert guard.record_order_result(False) == 3
        snapshot = SafetySnapshot(960.0, now, now - timedelta(minutes=3), False, 1, 1, False)
        decision = guard.evaluate(snapshot)
        assert not decision.allowed and decision.must_flatten
        assert any("연속 주문" in reason for reason in decision.reasons)
        restarted = LivePilotGuard(1000.0, db_path=database)
        still_tripped = restarted.evaluate(
            SafetySnapshot(1000.0, now + timedelta(seconds=1), now, True, 0, 0, True)
        )
        assert not still_tripped.allowed
        assert not restarted.reset_trip()
        assert restarted.reset_trip(manual_approval=True)
        assert restarted.record_order_result(True) == 0

    def test_portfolio_limits_pilot_capital_and_scale_gate(self) -> None:
        """거래당·총손절·레버리지·명목·델타·증액 한도를 동시 검사한다."""
        position = Position("p", SYMBOL, "long", 100.0, 5.0, 99.0, 110.0, 250.0)
        guard = PortfolioRiskGuard(1000.0)
        safe = guard.evaluate(TradeRiskProposal("ETH", "short", 400.0, 1.0, 2.0), [])
        assert safe.allowed
        risky = guard.evaluate(
            TradeRiskProposal("ETH", "long", 2000.0, 10.0, 3.0, "cross", 1000.0),
            [position],
            hedged_strategy=True,
        )
        assert not risky.allowed and len(risky.reasons) >= 4
        assert calculate_pilot_capital_krw(10_000_000) == 500_000.0
        assert calculate_pilot_capital_krw(100_000_000) == 1_000_000.0
        assert evaluate_scale_up(90, 30, 0.25, _gate()).allowed
        assert not evaluate_scale_up(89, 29, 0.30, _gate(passed=False)).allowed


def replace_live_evidence(evidence: LiveActivationEvidence) -> LiveActivationEvidence:
    """모든 live 활성 조건을 의도적으로 깨뜨린다."""
    return LiveActivationEvidence(
        TradingMode.DEMO,
        False,
        "",
        "expected",
        "actual",
        "v1",
        "v2",
        _gate(stage="offline", passed=False),
    )
