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
from src.risk.position_sizer import (
    calculate_auto_leverage,
    calculate_position_size,
    calculate_take_profit,
)

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

        exch = cfg["exchange"]
        self.total_capital: float = cap["total_capital"]
        self.trading_capital: float = self.total_capital * cap["trading_allocation"]
        self.risk_per_trade: float = cap["risk_per_trade"]
        self.leverage: float = exch["leverage"]
        self.min_rr: float = risk["min_rr_ratio"]
        self.max_positions: int = risk["max_positions"]
        self.max_per_symbol: int = risk.get("max_per_symbol", 1)
        self.max_same_direction: int = risk.get("max_same_direction", 3)
        self.max_exposure_pct: float = risk.get("max_exposure_pct", 0.80)

        # 자동 레버리지 설정 (포지션별 손절 거리 기반)
        self.auto_leverage: bool = exch.get("auto_leverage", False)
        self.max_leverage: float = exch.get("max_leverage", self.leverage)
        self.min_leverage: float = exch.get("min_leverage", 1.0)
        self.liq_buffer: float = exch.get("liq_buffer", 2.0)

        # 컨플루언스 점수별 차등 리스크 (내림차순 정렬)
        self.risk_tiers: list[dict] = sorted(
            risk.get("risk_tiers", []),
            key=lambda t: t["min_score"],
            reverse=True,
        )

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

        # 심볼당 포지션 수 제한 (분산 강제)
        if positions and symbol:
            symbol_count = sum(1 for p in positions if p.symbol == symbol)
            if symbol_count >= self.max_per_symbol:
                return False, f"심볼당 최대 포지션 초과: {symbol} ({symbol_count}/{self.max_per_symbol})"

        # 같은 방향 쏠림 방지
        if positions and direction:
            same_dir = sum(1 for p in positions if p.direction == direction)
            if same_dir >= self.max_same_direction:
                return False, f"동일 방향 최대 초과: {direction} ({same_dir}/{self.max_same_direction})"

        # 총 담보금 노출 한도 체크
        if positions:
            total_margin = sum(p.margin for p in positions)
            if total_margin / self.trading_capital >= self.max_exposure_pct:
                return False, (
                    f"총 노출 한도 초과: "
                    f"{total_margin:.0f}/{self.trading_capital * self.max_exposure_pct:.0f} USDT "
                    f"({total_margin / self.trading_capital * 100:.0f}%)"
                )

        return True, "OK"

    # ------------------------------------------------------------------
    # 트레이드 파라미터 계산
    # ------------------------------------------------------------------

    def risk_pct_for_score(self, score: float | None) -> float:
        """컨플루언스 점수에 해당하는 리스크 비율을 반환한다.

        risk_tiers가 설정돼 있으면 점수가 속한 티어의 risk_pct를 사용하고,
        없거나 점수 미제공 시 기본 risk_per_trade를 반환한다.

        Args:
            score: 컨플루언스 점수 (0~100), None이면 기본값

        Returns:
            적용할 리스크 비율 (예: 0.005)
        """
        if score is None or not self.risk_tiers:
            return self.risk_per_trade
        for tier in self.risk_tiers:   # 내림차순 정렬됨
            if score >= tier["min_score"]:
                return float(tier["risk_pct"])
        # 어떤 티어에도 못 미치면 가장 낮은 티어 값
        return float(self.risk_tiers[-1]["risk_pct"])

    def calculate_trade_params(
        self, entry: float, stop_loss: float, score: float | None = None
    ) -> dict:
        """
        리스크 기반 트레이드 파라미터 계산.

        auto_leverage가 켜져 있으면 손절 거리에 따라 포지션별 레버리지를
        자동 산출한다 (타이트한 손절 → 높은 레버리지, 청산가는 항상 손절가
        바깥). 꺼져 있으면 config의 고정 레버리지를 사용한다.

        score가 주어지면 risk_tiers에 따라 리스크 비율을 차등 적용한다
        (강한 셋업은 크게, 약한 셋업은 작게).

        Args:
            entry: 진입 가격
            stop_loss: 손절 가격
            score: 컨플루언스 점수 (차등 리스크용, None이면 기본 리스크)

        Returns:
            qty, entry, stop_loss, take_profit, leverage, risk_pct 포함 딕셔너리
        """
        risk_pct = self.risk_pct_for_score(score)

        if self.auto_leverage:
            leverage = float(calculate_auto_leverage(
                entry, stop_loss,
                max_leverage=self.max_leverage,
                min_leverage=self.min_leverage,
                liq_buffer=self.liq_buffer,
            ))
        else:
            leverage = self.leverage

        qty = calculate_position_size(
            self.trading_capital,
            risk_pct,
            entry,
            stop_loss,
            leverage,
        )
        tp = calculate_take_profit(entry, stop_loss, self.min_rr)
        logger.info(
            "트레이드 파라미터: entry=%.4f SL=%.4f qty=%.6f leverage=%gx "
            "risk=%.2f%% (score=%s, auto_lev=%s)",
            entry, stop_loss, qty, leverage, risk_pct * 100, score, self.auto_leverage,
        )
        return {
            "qty": qty,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": tp,
            "leverage": leverage,
            "risk_pct": risk_pct,
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
