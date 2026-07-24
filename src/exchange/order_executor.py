from __future__ import annotations

# 주문 실행 추상 인터페이스 및 페이퍼 트레이딩 구현.
# 페이퍼/실전 공통 주문 인터페이스를 정의하고,
# PaperEngine과 연동하는 페이퍼 주문 실행기를 제공한다.

import logging
import hashlib
import hmac
import json
import os
import stat
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import ccxt

from src.data.execution_store import ExecutionEventStore
from src.exchange.contracts import (
    ExecutionReport,
    FeeRateSnapshot,
    Fill,
    OrderRequest,
    OrderState,
    TradingMode,
)

logger = logging.getLogger(__name__)

LIVE_APPROVAL_TOKEN_ENV = "LIVE_TRADING_APPROVAL_TOKEN"
LEGACY_LIVE_APPROVAL_TOKEN_ENV = "BYBIT_LIVE_APPROVAL_TOKEN"
LIVE_REPORT_HASH_ENV = "LIVE_TRADING_VALIDATION_REPORT_SHA256"
LEGACY_LIVE_REPORT_HASH_ENV = "BYBIT_VALIDATION_REPORT_HASH"
_MAX_VALIDATION_REPORT_BYTES = 10 * 1024 * 1024


class OrderExecutor(ABC):
    """주문 실행 추상 인터페이스.

    페이퍼 트레이딩과 실전 트레이딩 모두 이 인터페이스를 구현한다.
    """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        qty: float,
        order_type: Literal["market", "limit"],
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy_version: str = "unknown",
    ) -> dict[str, Any]:
        """주문을 실행한다.

        Args:
            symbol: 거래 심볼
            direction: 매매 방향 ('long' 또는 'short')
            qty: 수량
            order_type: 주문 유형 ('market' 또는 'limit')
            price: 지정가 (limit 주문 시 필수)
            stop_loss: 손절가
            take_profit: 익절가
            strategy_version: 주문을 생성한 전략 버전

        Returns:
            주문 결과 dict (order_id, status 등)
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """주문을 취소한다.

        Args:
            order_id: 취소할 주문 ID

        Returns:
            취소 성공 여부
        """
        ...

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """미체결 주문 목록을 조회한다.

        Args:
            symbol: 특정 심볼 (None이면 전체)

        Returns:
            미체결 주문 리스트
        """
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """현재 포지션을 조회한다.

        Args:
            symbol: 거래 심볼

        Returns:
            포지션 정보 dict 또는 None
        """
        ...


class PaperOrderExecutor(OrderExecutor):
    """페이퍼 트레이딩용 주문 실행기.

    PaperEngine의 로직을 위임받아 OrderExecutor 인터페이스를 충족한다.
    """

    def __init__(self, paper_engine: Any) -> None:
        """PaperOrderExecutor를 초기화한다.

        Args:
            paper_engine: PaperEngine 인스턴스
        """
        self._engine = paper_engine
        self._order_counter: int = 0
        self._pending_orders: dict[str, dict[str, Any]] = {}
        logger.info("PaperOrderExecutor 초기화 완료")

    def place_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        qty: float,
        order_type: Literal["market", "limit"],
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy_version: str = "unknown",
    ) -> dict[str, Any]:
        """페이퍼 주문을 실행한다.

        market 주문은 즉시 PaperEngine으로 포지션을 생성하고,
        limit 주문은 미체결 목록에 추가한다.

        Args:
            symbol: 거래 심볼
            direction: 매매 방향
            qty: 수량
            order_type: 주문 유형
            price: 지정가 (limit 주문 시 필수)
            stop_loss: 손절가
            take_profit: 익절가
            strategy_version: 주문을 생성한 전략 버전

        Returns:
            주문 결과 dict
        """
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        if order_type == "limit" and price is None:
            logger.error("limit 주문에 price가 필요합니다: %s", order_id)
            return {"order_id": order_id, "status": "rejected", "reason": "price required"}

        if order_type == "market":
            # market 주문: PaperEngine에 즉시 위임
            entry_price = price if price is not None else 0.0
            sl = stop_loss if stop_loss is not None else 0.0
            tp = take_profit if take_profit is not None else 0.0

            pos = self._engine.open_position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                qty=qty,
                stop_loss=sl,
                take_profit=tp,
            )

            if pos is None:
                logger.warning("[PAPER] 주문 실패 (잔고 부족): %s", order_id)
                return {"order_id": order_id, "status": "rejected", "reason": "insufficient_balance"}

            logger.info("[PAPER] 시장가 주문 체결: %s %s %s qty=%.4f", order_id, symbol, direction, qty)
            return {
                "order_id": order_id,
                "status": "filled",
                "symbol": symbol,
                "direction": direction,
                "qty": qty,
                "filled_price": pos.entry_price,
                "strategy_version": strategy_version,
            }

        # limit 주문: 미체결 대기열에 추가
        order_info: dict[str, Any] = {
            "order_id": order_id,
            "status": "pending",
            "symbol": symbol,
            "direction": direction,
            "qty": qty,
            "order_type": order_type,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "strategy_version": strategy_version,
        }
        self._pending_orders[order_id] = order_info
        logger.info(
            "[PAPER] 지정가 주문 등록: %s %s %s price=%.4f qty=%.4f",
            order_id, symbol, direction, price, qty,  # type: ignore[arg-type]
        )
        return order_info

    def cancel_order(self, order_id: str) -> bool:
        """미체결 주문을 취소한다.

        Args:
            order_id: 취소할 주문 ID

        Returns:
            취소 성공 여부
        """
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            logger.info("[PAPER] 주문 취소: %s", order_id)
            return True
        logger.warning("[PAPER] 취소 실패 — 주문 없음: %s", order_id)
        return False

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """미체결 주문 목록을 조회한다.

        Args:
            symbol: 특정 심볼 (None이면 전체)

        Returns:
            미체결 주문 리스트
        """
        orders = list(self._pending_orders.values())
        if symbol is not None:
            orders = [o for o in orders if o["symbol"] == symbol]
        return orders

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """PaperEngine에서 현재 포지션을 조회한다.

        Args:
            symbol: 거래 심볼

        Returns:
            포지션 정보 dict 또는 None
        """
        for pos in self._engine.positions:
            if pos.symbol == symbol:
                return {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "qty": pos.qty,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "margin": pos.margin,
                    "entry_time": pos.entry_time.isoformat(),
                }
        return None


def _timestamp_from_milliseconds(value: Any) -> datetime | None:
    """밀리초 epoch 값을 UTC datetime으로 변환한다."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_state(
    status: Any,
    filled_quantity: float,
    requested_quantity: float,
) -> OrderState:
    """ccxt 주문 상태를 내부 상태로 변환한다."""
    normalized = str(status or "").lower()
    if requested_quantity > 0 and filled_quantity >= requested_quantity:
        return OrderState.FILLED
    if filled_quantity > 0:
        return OrderState.PARTIALLY_FILLED
    mapping = {
        "open": OrderState.ACCEPTED,
        "new": OrderState.ACCEPTED,
        "pending": OrderState.ACCEPTED,
        "closed": OrderState.FILLED,
        "filled": OrderState.FILLED,
        "canceled": OrderState.CANCELED,
        "cancelled": OrderState.CANCELED,
        "rejected": OrderState.REJECTED,
        "expired": OrderState.EXPIRED,
    }
    return mapping.get(normalized, OrderState.ACCEPTED)


class BybitOrderExecutor(OrderExecutor):
    """Bybit demo/live 주문 실행기.

    demo 모드는 Bybit Demo Trading endpoint를 사용한다. live 모드는
    프로세스 시작 시 수동 승인 토큰과 고정된 검증 보고서 해시를 모두
    검증하며, 두 값이 없거나 일치하지 않으면 클라이언트도 만들지 않는다.
    """

    def __init__(
        self,
        mode: TradingMode | str = TradingMode.DEMO,
        api_key: str | None = None,
        api_secret: str | None = None,
        event_store: ExecutionEventStore | None = None,
        db_path: Path | None = None,
        live_approval_token: str | None = None,
        validation_report_hash: str | None = None,
        validation_report_path: Path | str | None = None,
        exchange: Any | None = None,
    ) -> None:
        """실행기를 안전한 모드로 초기화한다.

        Args:
            mode: demo 또는 live 실행 모드.
            api_key: Bybit API 키. 없으면 환경변수에서 읽는다.
            api_secret: Bybit API 시크릿. 없으면 환경변수에서 읽는다.
            event_store: 주문·체결 이벤트 저장소.
            db_path: 저장소를 생성할 SQLite 경로.
            live_approval_token: 운영 시작 시 직접 입력한 승인 토큰.
            validation_report_hash: 승급을 통과한 보고서의 SHA-256 해시.
            validation_report_path: 승인 리포트 JSON 파일 경로. live에서 필수다.
            exchange: 테스트 또는 의존성 주입용 ccxt 호환 클라이언트.

        Raises:
            ValueError: paper 모드를 요청하거나 live 승인이 불완전한 경우.
            RuntimeError: API 인증 정보가 없는 경우.
        """
        self.mode = TradingMode(mode)
        self._live_approval_token = live_approval_token
        self._validation_report_hash = validation_report_hash
        self._validation_report_path = (
            Path(validation_report_path).expanduser()
            if validation_report_path is not None
            else None
        )
        if self.mode is TradingMode.PAPER:
            raise ValueError("paper 모드는 PaperOrderExecutor를 사용해야 합니다")
        if self.mode is TradingMode.LIVE:
            self._validate_live_approval(
                live_approval_token,
                validation_report_hash,
                self._validation_report_path,
            )

        if self.mode is TradingMode.DEMO:
            self._api_key = (
                api_key
                or os.getenv("BYBIT_DEMO_API_KEY")
                or os.getenv("BYBIT_API_KEY")
            )
            self._api_secret = (
                api_secret
                or os.getenv("BYBIT_DEMO_API_SECRET")
                or os.getenv("BYBIT_API_SECRET")
            )
        else:
            self._api_key = api_key or os.getenv("BYBIT_API_KEY")
            self._api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        if exchange is None and (not self._api_key or not self._api_secret):
            raise RuntimeError("Bybit API 키와 시크릿이 필요합니다")
        self._store = event_store or ExecutionEventStore(db_path)
        self._exchange = exchange or self._create_exchange()
        self._order_symbols: dict[str, str] = {}
        self._link_to_client: dict[str, str] = {}
        self._private_stream: Any | None = None
        logger.info("BybitOrderExecutor 초기화 완료: mode=%s", self.mode.value)

    @staticmethod
    def _read_approval_env(
        primary_name: str,
        legacy_name: str,
        label: str,
    ) -> str:
        """표준·레거시 환경변수를 충돌 없이 읽는다."""
        primary = os.getenv(primary_name)
        legacy = os.getenv(legacy_name)
        populated = [value for value in (primary, legacy) if value]
        if not populated:
            raise ValueError(
                f"{label} 환경변수가 필요합니다: {primary_name}"
            )
        if primary and legacy and not hmac.compare_digest(primary, legacy):
            raise ValueError(
                f"{label} 환경변수가 서로 충돌합니다: "
                f"{primary_name}, {legacy_name}"
            )
        return populated[0]

    @staticmethod
    def _load_validation_report(
        report_path: Path | None,
    ) -> tuple[bytes, dict[str, Any]]:
        """승인 보고서 정규 파일을 한 번 열어 원본 바이트와 JSON을 읽는다."""
        if report_path is None:
            raise ValueError("live에는 validation_report_path가 필요합니다")
        try:
            path_status = os.lstat(report_path)
            if stat.S_ISLNK(path_status.st_mode):
                raise ValueError(
                    "validation_report_path는 심볼릭 링크일 수 없습니다"
                )
            if not stat.S_ISREG(path_status.st_mode):
                raise ValueError(
                    "validation_report_path는 정규 파일이어야 합니다"
                )
            if path_status.st_size > _MAX_VALIDATION_REPORT_BYTES:
                raise ValueError("validation report 파일이 허용 크기를 초과합니다")
            with report_path.open("rb") as report_file:
                file_status = os.fstat(report_file.fileno())
                if not stat.S_ISREG(file_status.st_mode):
                    raise ValueError(
                        "validation_report_path는 정규 파일이어야 합니다"
                    )
                if (
                    path_status.st_dev,
                    path_status.st_ino,
                ) != (
                    file_status.st_dev,
                    file_status.st_ino,
                ):
                    raise ValueError(
                        "validation report 파일이 검증 중 변경되었습니다"
                    )
                if file_status.st_size > _MAX_VALIDATION_REPORT_BYTES:
                    raise ValueError("validation report 파일이 허용 크기를 초과합니다")
                report_bytes = report_file.read(_MAX_VALIDATION_REPORT_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                f"validation report 파일을 읽을 수 없습니다: {report_path}"
            ) from exc
        if len(report_bytes) > _MAX_VALIDATION_REPORT_BYTES:
            raise ValueError("validation report 파일이 허용 크기를 초과합니다")
        try:
            decoded = report_bytes.decode("utf-8")
            parsed = json.loads(
                decoded,
                object_pairs_hook=BybitOrderExecutor._unique_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("validation report는 유효한 UTF-8 JSON이어야 합니다") from exc
        if not isinstance(parsed, dict):
            raise ValueError("validation report 최상위 값은 JSON object여야 합니다")
        return report_bytes, parsed

    @staticmethod
    def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """중복 키가 없는 JSON object를 생성한다."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"validation report에 중복 키가 있습니다: {key}")
            result[key] = value
        return result

    @classmethod
    def _validate_live_approval(
        cls,
        provided_token: str | None,
        report_hash: str | None,
        report_path: Path | None,
        request_strategy_version: str | None = None,
    ) -> str:
        """live 승인 토큰·보고서 파일·전략 버전을 fail-closed 검증한다."""
        expected_token = cls._read_approval_env(
            LIVE_APPROVAL_TOKEN_ENV,
            LEGACY_LIVE_APPROVAL_TOKEN_ENV,
            "live 승인 토큰",
        )
        expected_hash = cls._read_approval_env(
            LIVE_REPORT_HASH_ENV,
            LEGACY_LIVE_REPORT_HASH_ENV,
            "live 검증 리포트 SHA-256",
        )
        if not provided_token or not report_hash:
            raise ValueError(
                "live는 수동 승인 토큰과 검증 보고서 해시 설정이 필요합니다"
            )
        normalized_hash = str(report_hash).strip().lower()
        normalized_expected_hash = str(expected_hash).strip().lower()
        if len(normalized_hash) != 64 or any(
            char not in "0123456789abcdef" for char in normalized_hash
        ):
            raise ValueError("validation_report_hash는 SHA-256 형식이어야 합니다")
        if len(normalized_expected_hash) != 64 or any(
            char not in "0123456789abcdef"
            for char in normalized_expected_hash
        ):
            raise ValueError("승인 환경변수의 리포트 해시가 SHA-256 형식이 아닙니다")
        if not hmac.compare_digest(str(provided_token), str(expected_token)):
            raise ValueError("live 수동 승인 토큰이 일치하지 않습니다")
        if not hmac.compare_digest(normalized_hash, normalized_expected_hash):
            raise ValueError("검증 보고서 해시가 승인된 해시와 일치하지 않습니다")
        report_bytes, report = cls._load_validation_report(report_path)
        actual_hash = hashlib.sha256(report_bytes).hexdigest()
        if not hmac.compare_digest(actual_hash, normalized_hash):
            raise ValueError("검증 보고서 파일의 SHA-256이 승인 해시와 다릅니다")
        if report.get("stage") != "demo":
            raise ValueError(
                "validation report stage는 정확히 'demo'여야 합니다"
            )
        if report.get("passed") is not True:
            raise ValueError(
                "validation report의 demo 게이트가 통과되지 않았습니다"
            )
        report_strategy_version = report.get("strategy_version")
        if (
            not isinstance(report_strategy_version, str)
            or not report_strategy_version.strip()
        ):
            raise ValueError(
                "validation report에 유효한 strategy_version이 필요합니다"
            )
        if request_strategy_version is not None and not hmac.compare_digest(
            report_strategy_version,
            request_strategy_version,
        ):
            raise ValueError(
                "주문 전략 버전이 승인 리포트의 strategy_version과 다릅니다"
            )
        return report_strategy_version

    def _create_exchange(self) -> Any:
        """mode에 맞는 Bybit ccxt 인증 클라이언트를 생성한다."""
        client = ccxt.bybit(
            {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        if self.mode is TradingMode.DEMO:
            client.enable_demo_trading(True)
        return client

    @staticmethod
    def build_order_link_id(client_order_id: str) -> str:
        """재시도에도 동일한 Bybit orderLinkId를 생성한다."""
        digest = hashlib.sha256(client_order_id.encode("utf-8")).hexdigest()
        return f"tb-{digest[:32]}"

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        """정규화 주문을 idempotent하게 제출하고 결과를 영속화한다."""
        if self.mode is TradingMode.LIVE:
            self._validate_live_approval(
                self._live_approval_token,
                self._validation_report_hash,
                self._validation_report_path,
                request.strategy_version,
            )
        existing = self._store.get_report(request.client_order_id)
        if existing is not None:
            logger.info("중복 주문 제출 차단: %s", request.client_order_id)
            return existing
        if (
            self.mode is TradingMode.LIVE
            and not request.reduce_only
            and request.stop_loss is None
        ):
            raise ValueError("live 신규 진입에는 서버측 stop_loss가 필요합니다")

        order_link_id = self.build_order_link_id(request.client_order_id)
        if not self._store.claim_order(
            request.client_order_id,
            order_link_id,
            request.symbol,
        ):
            concurrent_report = self._store.get_report(request.client_order_id)
            if concurrent_report is not None:
                return concurrent_report
            raise RuntimeError(
                f"동일 주문이 이미 제출 중입니다: {request.client_order_id}"
            )
        self._link_to_client[order_link_id] = request.client_order_id
        params: dict[str, Any] = {
            "orderLinkId": order_link_id,
            "reduceOnly": request.reduce_only,
            "timeInForce": request.time_in_force,
        }
        if request.stop_loss is not None:
            params.update(
                {
                    "stopLoss": request.stop_loss,
                    "slOrderType": "Market",
                    "tpslMode": "Full",
                }
            )
        if request.take_profit is not None:
            params.update(
                {
                    "takeProfit": request.take_profit,
                    "tpOrderType": "Market",
                    "tpslMode": "Full",
                }
            )

        received = datetime.now(timezone.utc)
        try:
            raw = self._exchange.create_order(
                request.symbol,
                request.order_type,
                request.side,
                request.quantity,
                request.price,
                params,
            )
            report = self._report_from_order(request, raw, received)
        except ccxt.BaseError as exc:
            report = ExecutionReport(
                order_id=order_link_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                state=OrderState.REJECTED,
                requested_quantity=request.quantity,
                filled_quantity=0.0,
                average_price=None,
                receive_timestamp=received,
                reject_reason=f"{type(exc).__name__}: {str(exc)[:300]}",
                raw={},
            )
            self._store.save_report(report)
            logger.error(
                "Bybit 주문 거절: %s — %s",
                request.client_order_id,
                report.reject_reason,
            )
            return report

        self._order_symbols[report.order_id] = request.symbol
        self._store.save_report(report)
        self._persist_raw_order_event(report, raw)
        return report

    def _report_from_order(
        self,
        request: OrderRequest,
        raw: dict[str, Any],
        received: datetime,
    ) -> ExecutionReport:
        """ccxt 주문 응답을 실행 보고서로 정규화한다."""
        order_id = str(raw.get("id") or self.build_order_link_id(request.client_order_id))
        client_order_id = request.client_order_id
        requested_quantity = float(raw.get("amount") or request.quantity)
        filled_quantity = float(raw.get("filled") or 0.0)
        average = raw.get("average")
        fills = self._fills_from_order(
            raw,
            order_id,
            client_order_id,
            request.symbol,
            request.side,
            received,
        )
        if filled_quantity == 0.0 and fills:
            filled_quantity = round(sum(fill.quantity for fill in fills), 8)
        if average is None and fills and filled_quantity > 0:
            average = (
                sum(fill.price * fill.quantity for fill in fills)
                / filled_quantity
            )
        state = _normalized_state(
            raw.get("status"),
            filled_quantity,
            requested_quantity,
        )
        return ExecutionReport(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=str(raw.get("symbol") or request.symbol),
            state=state,
            requested_quantity=requested_quantity,
            filled_quantity=filled_quantity,
            average_price=float(average) if average is not None else None,
            fills=fills,
            exchange_timestamp=_timestamp_from_milliseconds(raw.get("timestamp")),
            receive_timestamp=received,
            raw=raw,
        )

    def _fills_from_order(
        self,
        raw: dict[str, Any],
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: Literal["buy", "sell"],
        received: datetime,
    ) -> tuple[Fill, ...]:
        """ccxt 주문 응답에 포함된 trades를 체결 계약으로 변환한다."""
        fills: list[Fill] = []
        for index, trade in enumerate(raw.get("trades") or []):
            fee_info = trade.get("fee") or {}
            taker_or_maker = str(trade.get("takerOrMaker") or "unknown")
            if taker_or_maker not in {"maker", "taker"}:
                taker_or_maker = "unknown"
            fill_id = str(
                trade.get("id")
                or hashlib.sha256(
                    f"{order_id}:{index}:{trade}".encode("utf-8")
                ).hexdigest()
            )
            fills.append(
                Fill(
                    fill_id=fill_id,
                    order_id=order_id,
                    client_order_id=client_order_id,
                    symbol=str(trade.get("symbol") or symbol),
                    side=side,
                    quantity=float(trade.get("amount") or 0.0),
                    price=float(trade.get("price") or 0.0),
                    fee=float(fee_info.get("cost") or 0.0),
                    fee_currency=fee_info.get("currency"),
                    liquidity=taker_or_maker,  # type: ignore[arg-type]
                    exchange_timestamp=_timestamp_from_milliseconds(
                        trade.get("timestamp")
                    ),
                    receive_timestamp=received,
                    raw=trade,
                )
            )
        return tuple(fills)

    def _persist_raw_order_event(
        self,
        report: ExecutionReport,
        raw: dict[str, Any],
    ) -> None:
        """REST 주문 응답을 재대사용 원시 이벤트로 저장한다."""
        event_basis = (
            f"rest:{report.order_id}:{report.state.value}:"
            f"{report.filled_quantity}:{report.receive_timestamp.isoformat()}"
        )
        event_id = hashlib.sha256(event_basis.encode("utf-8")).hexdigest()
        self._store.append_event(
            event_id=event_id,
            source=f"bybit_{self.mode.value}",
            channel="order",
            payload=raw,
            exchange_timestamp=report.exchange_timestamp,
            receive_timestamp=report.receive_timestamp,
            order_id=report.order_id,
            client_order_id=report.client_order_id,
        )

    def place_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        qty: float,
        order_type: Literal["market", "limit"],
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy_version: str = "unknown",
    ) -> dict[str, Any]:
        """기존 OrderExecutor 호출을 정규화 계약으로 변환해 주문한다.

        Args:
            symbol: 거래 심볼.
            direction: long 또는 short 방향.
            qty: 주문 수량.
            order_type: market 또는 limit 주문 종류.
            price: 지정가 주문 가격.
            stop_loss: 서버측 손절 가격.
            take_profit: 서버측 익절 가격.
            strategy_version: 승인 리포트와 대조할 전략 버전.

        Returns:
            하위 호환 실행 결과 딕셔너리.
        """
        unique_basis = (
            f"{symbol}:{direction}:{qty}:{order_type}:{price}:"
            f"{stop_loss}:{take_profit}:{datetime.now(timezone.utc).isoformat()}"
        )
        client_order_id = hashlib.sha256(
            unique_basis.encode("utf-8")
        ).hexdigest()
        request = OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            side="buy" if direction == "long" else "sell",
            quantity=qty,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_version=strategy_version,
        )
        return self.submit_order(request).to_dict()

    def place_protective_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        quantity: float,
        trigger_price: float,
        client_order_id: str,
        strategy_version: str = "unknown",
    ) -> ExecutionReport:
        """거래소 서버에 reduce-only 비상 청산 주문을 등록한다."""
        side: Literal["buy", "sell"] = "sell" if direction == "long" else "buy"
        request = OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="market",
            reduce_only=True,
            strategy_version=strategy_version,
        )
        if self.mode is TradingMode.LIVE:
            self._validate_live_approval(
                self._live_approval_token,
                self._validation_report_hash,
                self._validation_report_path,
                request.strategy_version,
            )
        existing = self._store.get_report(client_order_id)
        if existing is not None:
            return existing
        order_link_id = self.build_order_link_id(client_order_id)
        if not self._store.claim_order(client_order_id, order_link_id, symbol):
            concurrent_report = self._store.get_report(client_order_id)
            if concurrent_report is not None:
                return concurrent_report
            raise RuntimeError(
                f"동일 보호주문이 이미 제출 중입니다: {client_order_id}"
            )
        self._link_to_client[order_link_id] = client_order_id
        params: dict[str, Any] = {
            "orderLinkId": order_link_id,
            "reduceOnly": True,
            "closeOnTrigger": True,
            "triggerPrice": trigger_price,
            "triggerDirection": 2 if direction == "long" else 1,
        }
        received = datetime.now(timezone.utc)
        raw = self._exchange.create_order(
            symbol,
            "market",
            side,
            quantity,
            None,
            params,
        )
        report = self._report_from_order(request, raw, received)
        self._store.save_report(report)
        self._persist_raw_order_event(report, raw)
        return report

    def fetch_fee_rate(self, symbol: str) -> FeeRateSnapshot:
        """Bybit 계정의 실제 maker/taker 수수료율을 조회하고 저장한다."""
        received = datetime.now(timezone.utc)
        raw = dict(self._exchange.fetch_trading_fee(symbol))
        maker = raw.get("maker")
        taker = raw.get("taker")
        if maker is None or taker is None:
            raise RuntimeError(f"계정 수수료율 응답이 불완전합니다: {symbol}")
        snapshot = FeeRateSnapshot(
            symbol=symbol,
            maker_rate=float(maker),
            taker_rate=float(taker),
            exchange_timestamp=_timestamp_from_milliseconds(raw.get("timestamp")),
            receive_timestamp=received,
            raw=raw,
        )
        self._store.save_fee_rate(snapshot)
        return snapshot

    def cancel_order(self, order_id: str) -> bool:
        """Bybit 미체결 주문을 취소하고 취소 이벤트를 저장한다."""
        symbol = self._order_symbols.get(order_id)
        try:
            raw = self._exchange.cancel_order(order_id, symbol)
        except ccxt.OrderNotFound:
            logger.warning("취소할 Bybit 주문이 없습니다: %s", order_id)
            return False
        received = datetime.now(timezone.utc)
        event_id = hashlib.sha256(
            f"cancel:{order_id}:{received.isoformat()}".encode("utf-8")
        ).hexdigest()
        self._store.append_event(
            event_id=event_id,
            source=f"bybit_{self.mode.value}",
            channel="order",
            payload=dict(raw),
            exchange_timestamp=_timestamp_from_milliseconds(raw.get("timestamp")),
            receive_timestamp=received,
            order_id=order_id,
            client_order_id=raw.get("clientOrderId"),
        )
        return True

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Bybit의 현재 미체결 주문을 조회한다."""
        orders = self._exchange.fetch_open_orders(symbol)
        return [dict(order) for order in orders]

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Bybit에서 심볼의 열린 포지션을 조회한다."""
        positions = self._exchange.fetch_positions([symbol])
        for position in positions:
            contracts = float(position.get("contracts") or 0.0)
            if position.get("symbol") == symbol and contracts > 0:
                return dict(position)
        return None

    def ingest_private_event(
        self,
        channel: Literal["order", "execution"],
        payload: dict[str, Any],
    ) -> int:
        """private WebSocket 주문/체결 메시지를 로컬 DB에 영구 저장한다."""
        received = datetime.now(timezone.utc)
        records = payload.get("data")
        if not isinstance(records, list):
            records = [payload]
        inserted = 0
        for index, item in enumerate(records):
            exchange_time = _timestamp_from_milliseconds(
                item.get("execTime")
                or item.get("updatedTime")
                or payload.get("creationTime")
            )
            order_id = item.get("orderId")
            order_link_id = item.get("orderLinkId")
            client_order_id = self._link_to_client.get(
                str(order_link_id),
                self._store.resolve_client_order_id(str(order_link_id))
                or (str(order_link_id) if order_link_id is not None else ""),
            )
            basis = (
                item.get("execId")
                or f"{channel}:{order_id}:{order_link_id}:"
                f"{item.get('updatedTime')}:{index}"
            )
            event_id = hashlib.sha256(str(basis).encode("utf-8")).hexdigest()
            if self._store.append_event(
                event_id=event_id,
                source=f"bybit_{self.mode.value}_websocket",
                channel=channel,
                payload=dict(item),
                exchange_timestamp=exchange_time,
                receive_timestamp=received,
                order_id=str(order_id) if order_id is not None else None,
                client_order_id=(
                    client_order_id
                    if client_order_id
                    else None
                ),
            ):
                inserted += 1
        return inserted

    def start_private_stream(self) -> Any:
        """Bybit private order/execution WebSocket 구독을 시작한다."""
        try:
            from pybit.unified_trading import WebSocket
        except ImportError as exc:
            raise RuntimeError("private stream에는 pybit 설치가 필요합니다") from exc
        kwargs: dict[str, Any] = {
            "testnet": False,
            "channel_type": "private",
            "api_key": self._api_key,
            "api_secret": self._api_secret,
        }
        if self.mode is TradingMode.DEMO:
            kwargs["demo"] = True
        stream = WebSocket(**kwargs)
        stream.order_stream(
            callback=lambda message: self.ingest_private_event("order", message)
        )
        stream.execution_stream(
            callback=lambda message: self.ingest_private_event(
                "execution",
                message,
            )
        )
        self._private_stream = stream
        return stream

    def reconcile(self, symbol: str | None = None) -> dict[str, Any]:
        """거래소 주문·포지션·체결을 조회해 재대사용 이벤트로 저장한다."""
        received = datetime.now(timezone.utc)
        open_orders = [dict(item) for item in self._exchange.fetch_open_orders(symbol)]
        positions = [
            dict(item)
            for item in self._exchange.fetch_positions(
                [symbol] if symbol is not None else None
            )
        ]
        trades = [
            dict(item)
            for item in self._exchange.fetch_my_trades(symbol)
        ]
        collections = {
            "reconcile_order": open_orders,
            "reconcile_position": positions,
            "reconcile_execution": trades,
        }
        for channel, items in collections.items():
            for index, item in enumerate(items):
                basis = (
                    f"{channel}:{item.get('id')}:{item.get('order')}:"
                    f"{item.get('timestamp')}:{index}"
                )
                self._store.append_event(
                    event_id=hashlib.sha256(basis.encode("utf-8")).hexdigest(),
                    source=f"bybit_{self.mode.value}_rest",
                    channel=channel,
                    payload=item,
                    exchange_timestamp=_timestamp_from_milliseconds(
                        item.get("timestamp")
                    ),
                    receive_timestamp=received,
                    order_id=str(item.get("order") or item.get("id") or ""),
                    client_order_id=item.get("clientOrderId"),
                )
        return {
            "orders": open_orders,
            "positions": positions,
            "trades": trades,
            "reconciled_at": received,
        }

    def close(self) -> None:
        """이벤트 저장소 연결을 닫는다."""
        self._store.close()
