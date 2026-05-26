"""
리스크 관리자 — 포지션 사이징, 서킷브레이커, 중복 차단, 리스크 노출 추적
config.yaml에서 모든 파라미터를 읽어오며, 거래 결과 콜백 패턴 지원
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal

import yaml

from src.paper_trading import Position
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizer import calculate_position_size, calculate_take_profit

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent


def load_config() -> dict:
    """config/config.yaml 파일에서 설정을 로드한다."""
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


class RiskManager:
    """리스크 관리자 — 거래 허용 여부, 파라미터 계산, 리스크 추적."""

    def __init__(self) -> None:
        """config.yaml 기반 초기화."""
        cfg = load_config()
        cap = cfg["capital"]
        risk = cfg["risk"]

        self.total_capital: float = cap["total_capital"]
        self.trading_capital: float = self.total_capital * cap["trading_allocation"]
        self.risk_per_trade: float = cap["risk_per_trade"]
        self.leverage: float = cfg["exchange"]["leverage"]
        self.min_rr: float = risk["min_rr_ratio"]
        self.max_positions: int = risk["max_positions"]

        self.cb: CircuitBreaker = CircuitBreaker(
            trading_capital=self.trading_capital,
            daily_loss_limit=risk["daily_loss_limit"],
            weekly_loss_limit=risk["weekly_loss_limit"],
            max_consecutive_losses=risk["max_consecutive_losses"],
        )

        # 거래 결과 콜백 목록 (Discord 알림 등)
        self._on_result_callbacks: list[Callable[[float, str], None]] = []

        logger.info("RiskManager: 트레이딩 자본=%.0f USDT", self.trading_capital)

    # ------------------------------------------------------------------
    # 콜백 등록 (Discord 알림 연동 준비)
    # ------------------------------------------------------------------

    def register_on_result(self, callback: Callable[[float, str], None]) -> None:
        """
        거래 결과 기록 시 호출될 콜백 등록.

        Args:
            callback: (pnl, reason) 인자를 받는 콜백 함수
        """
        self._on_result_callbacks.append(callback)
        logger.info("거래 결과 콜백 등록: %s", callback.__name__)

    # ------------------------------------------------------------------
    # 거래 허용 여부
    # ------------------------------------------------------------------

    def check_trade_allowed(
        self,
        current_positions: int,
        positions: list[Position] | None = None,
        symbol: str | None = None,
        direction: str | None = None,
    ) -> tuple[bool, str]:
        """
        거래 허용 여부 확인.

        Args:
            current_positions: 현재 보유 포지션 수
            positions: 현재 보유 포지션 목록 (중복 체크용)
            symbol: 진입하려는 심볼
            direction: 진입하려는 방향

        Returns:
            (allowed, reason) 튜플
        """
        # 서킷브레이커 체크
        allowed, reason = self.cb.is_trading_allowed()
        if not allowed:
            return False, reason

        # 최대 포지션 수 체크
        if current_positions >= self.max_positions:
            return False, f"최대 포지션 초과 ({current_positions}/{self.max_positions})"

        # 같은 심볼 같은 방향 중복 포지션 차단
        if positions and symbol and direction:
            for pos in positions:
                if pos.symbol == symbol and pos.direction == direction:
                    return False, f"중복 포지션 차단: {symbol} {direction} 이미 보유 중"

        return True, "OK"

    # ------------------------------------------------------------------
    # 트레이드 파라미터 계산
    # ------------------------------------------------------------------

    def calculate_trade_params(self, entry: float, stop_loss: float) -> dict:
        """
        리스크 기반 트레이드 파라미터 계산.

        Args:
            entry: 진입 가격
            stop_loss: 손절 가격

        Returns:
            qty, entry, stop_loss, take_profit 포함 딕셔너리
        """
        qty = calculate_position_size(
            self.trading_capital,
            self.risk_per_trade,
            entry,
            stop_loss,
            self.leverage,
        )
        tp = calculate_take_profit(entry, stop_loss, self.min_rr)
        return {
            "qty": qty,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": tp,
        }

    # ------------------------------------------------------------------
    # 리스크 노출 추적
    # ------------------------------------------------------------------

    def calculate_total_exposure(self, positions: list[Position]) -> dict:
        """
        현재 총 리스크 노출 계산.

        Args:
            positions: 현재 보유 포지션 목록

        Returns:
            total_margin: 총 담보금
            total_risk: 총 리스크 금액 (SL 기준)
            exposure_pct: 트레이딩 자본 대비 노출 비율
            positions_by_symbol: 심볼별 노출 현황
        """
        total_margin = 0.0
        total_risk = 0.0
        by_symbol: dict[str, dict] = {}

        for pos in positions:
            total_margin += pos.margin
            # 손절 시 최대 손실
            if pos.direction == "long":
                max_loss = round((pos.entry_price - pos.stop_loss) * pos.qty, 8)
            else:
                max_loss = round((pos.stop_loss - pos.entry_price) * pos.qty, 8)
            total_risk += max_loss

            if pos.symbol not in by_symbol:
                by_symbol[pos.symbol] = {"margin": 0.0, "risk": 0.0, "count": 0}
            by_symbol[pos.symbol]["margin"] += pos.margin
            by_symbol[pos.symbol]["risk"] += max_loss
            by_symbol[pos.symbol]["count"] += 1

        exposure_pct = (
            total_margin / self.trading_capital if self.trading_capital > 0 else 0.0
        )

        return {
            "total_margin": round(total_margin, 8),
            "total_risk": round(total_risk, 8),
            "exposure_pct": round(exposure_pct, 4),
            "positions_by_symbol": by_symbol,
        }

    # ------------------------------------------------------------------
    # 거래 결과 기록
    # ------------------------------------------------------------------

    def record_result(self, pnl: float, reason: str = "") -> None:
        """
        거래 결과 기록 및 콜백 실행.

        Args:
            pnl: 실현 손익 (USDT)
            reason: 청산 사유
        """
        self.cb.record_trade(pnl)

        # 등록된 콜백 실행
        for cb in self._on_result_callbacks:
            try:
                cb(pnl, reason)
            except Exception as e:
                logger.error("거래 결과 콜백 실행 오류: %s", e)
