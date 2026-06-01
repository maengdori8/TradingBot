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


@dataclass
class ScanResult:
    """심볼 스캔 결과 — 진입 신호 + 관심종목(watchlist) 근접도 점수.

    score(0~100)는 ICT 컨플루언스 충족 정도를 나타낸다.
    qualified=True 이면 즉시 진입 가능한 확정 신호이며 signal 필드가 채워진다.
    qualified=False 라도 score가 높으면 '진입 임박' 관심종목으로 분류된다.
    """

    symbol: str
    direction: Literal["long", "short", "none"]
    score: float            # 0~100 컨플루언스 점수
    stage: int              # 통과한 단계 수 (0~4)
    qualified: bool         # 즉시 진입 가능 여부
    price: float
    reason: str
    signal: TradeSignal | None = None
    checks: dict | None = None   # 단계별 통과 여부 상세

    def to_dict(self) -> dict:
        """대시보드/JSON 직렬화용 딕셔너리 변환."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "score": round(self.score, 1),
            "stage": self.stage,
            "qualified": self.qualified,
            "price": self.price,
            "reason": self.reason,
            "checks": self.checks or {},
            "signal": {
                "entry_price": self.signal.entry_price,
                "stop_loss": self.signal.stop_loss,
                "take_profit": self.signal.take_profit,
                "rr_ratio": self.signal.rr_ratio,
            } if self.signal else None,
        }


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
    rr = round(reward / risk, 6) if risk > 0 else 0

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


# 컨플루언스 점수 가중치 (합계 100)
_W_TREND_BOS = 30.0      # 4H BOS (강한 추세)
_W_TREND_CHOCH = 22.0    # 4H CHoCH (전환)
_W_ZONE_BOTH = 30.0      # 1H FVG + OB 동시
_W_ZONE_ONE = 20.0       # 1H FVG 또는 OB 하나
_W_KZ = 15.0             # Kill Zone 내부
_W_OTE_MAX = 25.0        # OTE (깊이에 따라 가중)


def scan_symbol(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    current_price: float,
    min_rr: float = 1.5,
    min_score: float = 70.0,
    require_volume: bool = True,
) -> ScanResult:
    """심볼을 스캔하여 진입 신호 + 관심종목 근접도 점수를 산출한다.

    모든 코인을 스캔할 때 사용한다. 각 ICT 단계의 충족도를 0~100 점수로 환산하며,
    모든 단계 통과 + R:R + min_score + 거래량 확인을 만족할 때만 qualified=True
    (즉시 진입). 그 외에는 score 기반 관심종목으로 분류된다.

    Args:
        df_4h: 4시간봉 데이터
        df_1h: 1시간봉 데이터
        df_15m: 15분봉 데이터
        symbol: 거래 심볼
        current_price: 현재 가격
        min_rr: 최소 R:R 비율
        min_score: 진입 확정 최소 컨플루언스 점수 (엄격도)
        require_volume: 거래량 확인 게이트 사용 여부

    Returns:
        ScanResult
    """
    checks: dict = {
        "trend": False, "zone": False, "kill_zone": False,
        "ote": False, "volume": False, "rr": False,
    }
    score = 0.0
    stage = 0
    direction: Literal["long", "short", "none"] = "none"

    # ── 1단계: 4H 추세 ───────────────────────────────────────────────
    bos_4h = detect_bos(df_4h)
    choch_4h = detect_choch(df_4h)
    trend = bos_4h or choch_4h
    if trend is None:
        return ScanResult(
            symbol=symbol, direction="none", score=0.0, stage=0,
            qualified=False, price=current_price,
            reason="4H 구조 불명확", checks=checks,
        )
    direction = "long" if trend == "bullish" else "short"
    structure_type = "BOS" if bos_4h else "CHoCH"
    score += _W_TREND_BOS if bos_4h else _W_TREND_CHOCH
    checks["trend"] = True
    stage = 1

    # ── 2단계: 1H OB / FVG 존 ────────────────────────────────────────
    fvg_1h = detect_fvg(df_1h)
    ob_1h = detect_order_blocks(df_1h)
    in_fvg = is_price_in_fvg(current_price, fvg_1h)
    in_ob = is_price_in_ob(current_price, ob_1h)
    if direction == "long":
        in_fvg = [f for f in in_fvg if f.type == "bullish"]
        in_ob = [o for o in in_ob if o.type == "bullish"]
    else:
        in_fvg = [f for f in in_fvg if f.type == "bearish"]
        in_ob = [o for o in in_ob if o.type == "bearish"]

    if in_fvg and in_ob:
        score += _W_ZONE_BOTH
        zone_source = "FVG+OB"
        checks["zone"] = True
        stage = 2
    elif in_fvg or in_ob:
        score += _W_ZONE_ONE
        zone_source = "FVG" if in_fvg else "OB"
        checks["zone"] = True
        stage = 2
    else:
        zone_source = "-"
        # 존 미진입이면 관심종목 후보 (추세만 잡힘)
        return ScanResult(
            symbol=symbol, direction=direction, score=score, stage=1,
            qualified=False, price=current_price,
            reason=f"4H {trend}({structure_type}) · 1H 존 대기",
            checks=checks,
        )

    # ── 3단계: 15m Kill Zone ─────────────────────────────────────────
    now = datetime.now(timezone.utc)
    kz_active = is_in_kill_zone(now)
    active_session = get_active_session(now)
    if kz_active:
        score += _W_KZ
        checks["kill_zone"] = True
        stage = 3

    # ── 4단계: OTE 존 (깊이 가중) ────────────────────────────────────
    recent_high = df_15m["high"].rolling(20).max().iloc[-1]
    recent_low = df_15m["low"].rolling(20).min().iloc[-1]
    ote_zone = calculate_ote_zone(recent_high, recent_low, direction)
    in_ote = is_price_in_ote(current_price, ote_zone)
    if in_ote:
        # OTE 중앙(0.705 부근)에 가까울수록 높은 점수
        lo, hi = ote_zone.ote_low, ote_zone.ote_high
        if hi > lo:
            mid = (lo + hi) / 2.0
            centered = 1.0 - min(1.0, abs(current_price - mid) / ((hi - lo) / 2.0))
        else:
            centered = 1.0
        score += _W_OTE_MAX * (0.6 + 0.4 * centered)
        checks["ote"] = True
        if stage == 3:
            stage = 4

    # ── 거래량 확인 (죽은 코인 배제) ──────────────────────────────────
    vol = df_15m["volume"]
    vol_avg = vol.rolling(20).mean().iloc[-1]
    vol_now = vol.iloc[-1]
    volume_ok = bool(vol_now >= vol_avg * 0.8) if vol_avg > 0 else True
    checks["volume"] = volume_ok

    # ── R:R 확인용 신호 생성 시도 ────────────────────────────────────
    signal: TradeSignal | None = None
    rr_ok = False
    if checks["trend"] and checks["zone"] and checks["kill_zone"] and checks["ote"]:
        signal = generate_signal(
            df_4h, df_1h, df_15m, symbol, current_price, min_rr=min_rr
        )
        if signal:
            rr_ok = signal.rr_ratio >= min_rr
            checks["rr"] = rr_ok
            signal.reason = (
                f"{signal.reason} · score {score:.0f}"
            )

    # ── 확정 여부 판정 (엄격) ────────────────────────────────────────
    qualified = bool(
        signal is not None
        and rr_ok
        and score >= min_score
        and (volume_ok or not require_volume)
    )

    reason = (
        f"4H {trend}({structure_type}) · 1H {zone_source}"
        f"{' · KZ' + ('✓' if kz_active else '✗')}"
        f"{' · OTE✓' if in_ote else ' · OTE대기'}"
        f"{' · 거래량✓' if volume_ok else ' · 거래량부족'}"
    )

    return ScanResult(
        symbol=symbol, direction=direction, score=score, stage=stage,
        qualified=qualified, price=current_price, reason=reason,
        signal=signal if qualified else None, checks=checks,
    )
