from __future__ import annotations

"""
멀티 타임프레임 신호 통합 엔진
4H -> 1H -> 15m 조건 모두 충족 시 신호 발생
"""

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from . import load_strategy_params
from .decision import DecisionContext, DecisionFrames, slice_decision_frames
from .market_structure import detect_bos, detect_choch
from .fvg_detector import detect_fvg, is_price_in_fvg
from .order_block import detect_order_blocks, is_price_in_ob
from .kill_zone import is_in_kill_zone, get_active_session
from .ote import calculate_ote_zone, is_price_in_ote

logger = logging.getLogger(__name__)


def _decision_inputs(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    context: DecisionContext | None,
) -> tuple[DecisionFrames, DecisionContext]:
    """결정 컨텍스트와 시간 안전한 입력 봉을 준비한다.

    기존 RangeIndex 기반 호출은 하위 호환을 위해 받은 프레임을 그대로 사용한다.
    명시적 컨텍스트가 전달되면 UTC 인덱스와 완전 종료 봉 규칙을 강제한다.
    """
    effective_context = context or DecisionContext.legacy_now()
    if context is None:
        return DecisionFrames(df_4h=df_4h, df_1h=df_1h, df_15m=df_15m), effective_context
    return (
        slice_decision_frames(df_4h, df_1h, df_15m, effective_context),
        effective_context,
    )


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
    min_rr: float = 2.5,
    context: DecisionContext | None = None,
) -> TradeSignal | None:
    """멀티 타임프레임 신호를 생성한다.

    조건:
    1. 4H: BOS 또는 CHoCH 로 추세 방향 확인
    2. 1H: OB 또는 FVG 존 탐지
    3. 15m: OTE 레벨 확인 (킬존은 게이트 아님 — 세션 태깅/가점만, 24h 진입)

    Args:
        df_4h: 4시간봉 데이터
        df_1h: 1시간봉 데이터
        df_15m: 15분봉 데이터
        symbol: 거래 심볼
        current_price: 현재 가격
        min_rr: 최소 R:R 비율
        context: 판단 시각과 데이터 컷오프. 생략 시 기존 현재 시각 동작을 유지한다.

    Returns:
        TradeSignal 또는 None
    """
    frames, effective_context = _decision_inputs(df_4h, df_1h, df_15m, context)
    df_4h, df_1h, df_15m = frames.df_4h, frames.df_1h, frames.df_15m
    if df_4h.empty or df_1h.empty or df_15m.empty:
        logger.warning(
            "[%s] 종료 봉 부족으로 신호 판단 중단 run_id=%s",
            symbol,
            effective_context.run_id,
        )
        return None

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

    # ── 3단계: 15m 세션 태깅 + OTE ───────────────────────────────────
    # 킬존은 더 이상 하드 게이트가 아님 (24h 진입 허용).
    # 근거: 6개월 36심볼 신호연구에서 어떤 시간 게이트도 양(+)의 가치를 보이지 않음
    # (정정된 ICT 창 기준 in +0.079R vs out +0.068R). 세션은 태깅/가점으로만 사용.
    now = effective_context.decision_time
    kz_active = is_in_kill_zone(now)
    active_session = get_active_session(now)
    logger.debug(
        "[%s] 3단계: 세션=%s KZ=%s (UTC=%s)",
        symbol, active_session, kz_active, now.strftime("%H:%M"),
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
        reason=f"4H {trend}({structure_type}) + 1H {zone_source} + "
               f"{'KZ(' + str(active_session) + ')' if kz_active else 'KZ밖'} + OTE",
        rr_ratio=rr,
    )
    logger.info(
        "[%s] 신호 발생: %s entry=%.4f SL=%.4f TP=%.4f R:R=%.2f (ATR=%.6f, multiplier=%.1f)",
        symbol, direction.upper(), current_price, stop_loss, take_profit, rr,
        atr, atr_multiplier,
    )
    return signal


# 컨플루언스 점수 가중치 — 만점 100 (BOS 30 + 존동시 35 + KZ 5 + OTE 30)
# 2026-06 재배분: _W_KZ 15→5 축소분(10)을 데이터가 지지하는 항목으로 이전 —
# FVG+OB 동시존은 신호연구에서 +0.093R(단일존 -0.035R)로 실측 우위 → +5,
# OTE 깊이 → +5. KZ 밖 만점 95라 risk_tiers A급(85) 도달이 24h 어디서나 가능.
_W_TREND_BOS = 30.0      # 4H BOS (강한 추세)
_W_TREND_CHOCH = 22.0    # 4H CHoCH (전환)
_W_ZONE_BOTH = 35.0      # 1H FVG + OB 동시 (연구 실측 우위 반영)
_W_ZONE_ONE = 20.0       # 1H FVG 또는 OB 하나
# 시간 게이트의 양(+)가치 미확인 + 종전 창 정의 오류로 15→5 축소.
# 정정된 창(런던 07-10/뉴욕 12-15 UTC) 재측정 후 0/5/15 최종 결정 예정.
_W_KZ = 5.0              # Kill Zone 내부 (가점만, 게이트 아님)
_W_OTE_MAX = 30.0        # OTE (깊이에 따라 가중)


def scan_symbol(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    current_price: float,
    min_rr: float = 2.5,
    min_score: float = 70.0,
    require_volume: bool = False,
    context: DecisionContext | None = None,
) -> ScanResult:
    """심볼을 스캔하여 진입 신호 + 관심종목 근접도 점수를 산출한다.

    모든 코인을 스캔할 때 사용한다. 각 ICT 단계의 충족도를 0~100 점수로 환산하며,
    추세+존+OTE 통과 + R:R + min_score(+옵션 거래량)를 만족할 때 qualified=True
    (즉시 진입). 킬존은 게이트가 아니라 가점/태깅(24h 진입, 2026-06 개정).
    그 외에는 score 기반 관심종목으로 분류된다.

    Args:
        df_4h: 4시간봉 데이터
        df_1h: 1시간봉 데이터
        df_15m: 15분봉 데이터
        symbol: 거래 심볼
        current_price: 현재 가격
        min_rr: 최소 R:R 비율
        min_score: 진입 확정 최소 컨플루언스 점수 (엄격도)
        require_volume: 거래량 확인 게이트 사용 여부
        context: 판단 시각과 데이터 컷오프. 전달하면 종료 봉만 사용한다.

    Returns:
        ScanResult
    """
    frames, effective_context = _decision_inputs(df_4h, df_1h, df_15m, context)
    df_4h, df_1h, df_15m = frames.df_4h, frames.df_1h, frames.df_15m

    checks: dict = {
        "trend": False, "zone": False, "kill_zone": False,
        "ote": False, "volume": False, "rr": False,
    }
    score = 0.0
    stage = 0
    direction: Literal["long", "short", "none"] = "none"
    if df_4h.empty or df_1h.empty or df_15m.empty:
        return ScanResult(
            symbol=symbol,
            direction="none",
            score=0.0,
            stage=0,
            qualified=False,
            price=current_price,
            reason="완전히 종료된 멀티 타임프레임 봉 부족",
            checks=checks,
        )

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

    # ── 킬존 태깅/가점 (게이트 아님 — stage와 무관) ───────────────────
    now = effective_context.decision_time
    kz_active = is_in_kill_zone(now)
    if kz_active:
        score += _W_KZ
        checks["kill_zone"] = True

    # ── 3단계: OTE 존 (깊이 가중) ────────────────────────────────────
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
        stage = 3

    # ── 거래량 확인 (죽은 코인 배제) ──────────────────────────────────
    vol = df_15m["volume"]
    vol_avg = vol.rolling(20).mean().iloc[-1]
    vol_now = vol.iloc[-1]
    volume_ok = bool(vol_now >= vol_avg * 0.8) if vol_avg > 0 else True
    checks["volume"] = volume_ok

    # ── R:R 확인용 신호 생성 시도 ────────────────────────────────────
    signal: TradeSignal | None = None
    rr_ok = False
    # 킬존은 게이트에서 제외 (24h 진입) — checks["kill_zone"]은 태깅/학습용으로만 기록
    if checks["trend"] and checks["zone"] and checks["ote"]:
        signal = generate_signal(
            df_4h,
            df_1h,
            df_15m,
            symbol,
            current_price,
            min_rr=min_rr,
            context=context,
        )
        if signal:
            rr_ok = signal.rr_ratio >= min_rr
            checks["rr"] = rr_ok
            if rr_ok:
                stage = 4          # 4단계 = 신호+R:R 확보 (KZ와 무관)
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
