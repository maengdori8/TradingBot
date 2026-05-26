from __future__ import annotations

"""
멀티 타임프레임 신호 통합 엔진
4H -> 1H -> 15m 조건 모두 충족 시 신호 발생
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from . import load_strategy_params
from .market_structure import detect_bos, detect_choch
from .fvg_detector import detect_fvg, is_price_in_fvg
from .order_block import detect_order_blocks, is_price_in_ob
from .kill_zone import is_in_kill_zone, get_active_session
from .ote import calculate_ote_zone, is_price_in_ote

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """트레이드 신호 데이터 클래스."""

    direction: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    reason: str
    rr_ratio: float


def generate_signal(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    current_price: float,
    min_rr: float = 2.0,
) -> TradeSignal | None:
    """멀티 타임프레임 신호를 생성한다.

    조건:
    1. 4H: BOS 또는 CHoCH 로 추세 방향 확인
    2. 1H: OB 또는 FVG 존 탐지
    3. 15m: Kill Zone 내 + OTE 레벨 확인

    Args:
        df_4h: 4시간봉 데이터
        df_1h: 1시간봉 데이터
        df_15m: 15분봉 데이터
        symbol: 거래 심볼
        current_price: 현재 가격
        min_rr: 최소 R:R 비율

    Returns:
        TradeSignal 또는 None
    """
    params = load_strategy_params()
    atr_period: int = params["atr"]["period"]
    atr_multiplier: float = params["atr"]["multiplier"]

    logger.debug("[%s] 신호 생성 시작 (price=%.4f, min_rr=%.1f)", symbol, current_price, min_rr)

    # ── 1단계: 4H 추세 방향 ──────────────────────────────────────────
    bos_4h = detect_bos(df_4h)
    choch_4h = detect_choch(df_4h)
    trend = bos_4h or choch_4h

    if trend is None:
        logger.debug(
            "[%s] 1단계 실패: 4H 구조 불명확 (BOS=%s, CHoCH=%s) — 신호 없음",
            symbol, bos_4h, choch_4h,
        )
        return None

    direction: Literal["long", "short"] = "long" if trend == "bullish" else "short"
    structure_type = "BOS" if bos_4h else "CHoCH"
    logger.debug(
        "[%s] 1단계 통과: 4H 추세=%s (근거=%s, BOS=%s, CHoCH=%s)",
        symbol, trend, structure_type, bos_4h, choch_4h,
    )

    # ── 2단계: 1H OB / FVG 존 ────────────────────────────────────────
    fvg_1h = detect_fvg(df_1h)
    ob_1h = detect_order_blocks(df_1h)

    logger.debug(
        "[%s] 2단계: 1H FVG %d개, OB %d개 탐지됨",
        symbol, len(fvg_1h), len(ob_1h),
    )

    in_fvg = is_price_in_fvg(current_price, fvg_1h)
    in_ob = is_price_in_ob(current_price, ob_1h)

    # 방향에 맞는 존만 필터
    if direction == "long":
        in_fvg = [f for f in in_fvg if f.type == "bullish"]
        in_ob = [o for o in in_ob if o.type == "bullish"]
    else:
        in_fvg = [f for f in in_fvg if f.type == "bearish"]
        in_ob = [o for o in in_ob if o.type == "bearish"]

    if not in_fvg and not in_ob:
        logger.debug(
            "[%s] 2단계 실패: 1H %s 방향 OB/FVG 존 미진입 "
            "(전체 FVG=%d, 전체 OB=%d, 방향일치 FVG=%d, 방향일치 OB=%d) — 신호 없음",
            symbol, direction, len(fvg_1h), len(ob_1h), len(in_fvg), len(in_ob),
        )
        return None

    zone_source = "FVG" if in_fvg else "OB"
    logger.debug(
        "[%s] 2단계 통과: 1H %s 존 진입 확인 (FVG %d개, OB %d개)",
        symbol, zone_source, len(in_fvg), len(in_ob),
    )

    # ── 3단계: 15m Kill Zone + OTE ───────────────────────────────────
    now = datetime.now(timezone.utc)
    if not is_in_kill_zone(now):
        active = get_active_session(now)
        logger.debug(
            "[%s] 3단계 실패: Kill Zone 외부 (현재 UTC=%s, 활성 세션=%s) — 신호 없음",
            symbol, now.strftime("%H:%M"), active,
        )
        return None

    active_session = get_active_session(now)
    logger.debug(
        "[%s] 3단계: Kill Zone 내부 확인 (세션=%s, UTC=%s)",
        symbol, active_session, now.strftime("%H:%M"),
    )

    # OTE: 최근 스윙 기준
    recent_high = df_15m["high"].rolling(20).max().iloc[-1]
    recent_low = df_15m["low"].rolling(20).min().iloc[-1]
    ote_zone = calculate_ote_zone(recent_high, recent_low, direction)

    if not is_price_in_ote(current_price, ote_zone):
        logger.debug(
            "[%s] 3단계 실패: OTE 존 미진입 "
            "(price=%.4f, OTE=[%.4f, %.4f], direction=%s) — 신호 없음",
            symbol, current_price, ote_zone.ote_low, ote_zone.ote_high, direction,
        )
        return None

    logger.debug(
        "[%s] 3단계 통과: OTE 존 진입 확인 (price=%.4f, OTE=[%.4f, %.4f])",
        symbol, current_price, ote_zone.ote_low, ote_zone.ote_high,
    )

    # ── 신호 생성 ────────────────────────────────────────────────────
    atr = (df_15m["high"] - df_15m["low"]).rolling(atr_period).mean().iloc[-1]
    logger.debug("[%s] ATR 계산: %.6f (period=%d)", symbol, atr, atr_period)

    if direction == "long":
        stop_loss = current_price - atr * atr_multiplier
        take_profit = current_price + atr * atr_multiplier * min_rr
    else:
        stop_loss = current_price + atr * atr_multiplier
        take_profit = current_price - atr * atr_multiplier * min_rr

    risk = abs(current_price - stop_loss)
    reward = abs(current_price - take_profit)
    rr = reward / risk if risk > 0 else 0

    if rr < min_rr:
        logger.debug(
            "[%s] R:R 부족: %.2f < %.2f (risk=%.4f, reward=%.4f) — 신호 없음",
            symbol, rr, min_rr, risk, reward,
        )
        return None

    signal = TradeSignal(
        direction=direction,
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        symbol=symbol,
        reason=f"4H {trend}({structure_type}) + 1H {zone_source} + KZ({active_session}) + OTE",
        rr_ratio=rr,
    )
    logger.info(
        "[%s] 신호 발생: %s entry=%.4f SL=%.4f TP=%.4f R:R=%.2f (ATR=%.6f, multiplier=%.1f)",
        symbol, direction.upper(), current_price, stop_loss, take_profit, rr,
        atr, atr_multiplier,
    )
    return signal
