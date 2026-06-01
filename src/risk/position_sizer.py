"""
포지션 사이징 — 리스크 % 기반 수량 계산 및 담보금 산출
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float = 1.0,
) -> float:
    """
    리스크 기반 포지션 수량 계산.

    공식: qty = (capital * risk_pct) / (|entry - stop_loss| / entry) / entry

    Args:
        capital: 트레이딩 자본 (USDT)
        risk_pct: 리스크 비율 (예: 0.01 = 1%)
        entry_price: 진입 가격
        stop_loss_price: 손절 가격
        leverage: 레버리지 (포지션 최대 수량 제한용)

    Returns:
        계약 수량 (코인 단위)
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("가격은 양수여야 합니다.")
    if risk_pct <= 0 or risk_pct > 1:
        raise ValueError("risk_pct는 0~1 사이여야 합니다.")

    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        raise ValueError("진입가와 손절가가 동일합니다.")

    qty = risk_amount / price_risk

    # 레버리지 한도 초과 방지
    max_qty = (capital * leverage) / entry_price
    qty = min(qty, max_qty)

    return round(qty, 6)


def calculate_auto_leverage(
    entry_price: float,
    stop_loss_price: float,
    max_leverage: float = 10.0,
    min_leverage: float = 1.0,
    liq_buffer: float = 2.0,
) -> int:
    """손절 거리 기반 자동 레버리지 산출.

    청산가가 항상 손절가보다 멀리 있도록 보장한다. 청산은 손실이 증거금에
    근접할 때(≈ 1/leverage 거리) 발생하므로, 청산 거리가 손절 거리의
    liq_buffer 배가 되도록 레버리지를 정한다.

        leverage = 1 / (liq_buffer * sl_distance_pct)

    예) 손절 2% + liq_buffer 2.0 → 청산 4% 거리 목표 → 레버리지 25x
        (단, max_leverage로 상한 제한)

    손절이 타이트할수록 레버리지가 높아(증거금 절약), 넓을수록 낮아진다.

    Args:
        entry_price: 진입 가격
        stop_loss_price: 손절 가격
        max_leverage: 레버리지 상한
        min_leverage: 레버리지 하한
        liq_buffer: 청산 거리 / 손절 거리 배수 (클수록 보수적)

    Returns:
        정수 레버리지 (min~max 범위)
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("가격은 양수여야 합니다.")
    if liq_buffer <= 0:
        raise ValueError("liq_buffer는 양수여야 합니다.")

    sl_distance_pct = abs(entry_price - stop_loss_price) / entry_price
    if sl_distance_pct == 0:
        return int(max_leverage)

    raw = 1.0 / (liq_buffer * sl_distance_pct)
    leverage = max(min_leverage, min(max_leverage, raw))
    result = int(leverage)  # 정수 내림 (보수적)
    return max(int(min_leverage), result)


def calculate_stop_loss_atr(
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
) -> float:
    """
    ATR 기반 손절 가격 계산.

    Args:
        df: OHLCV DataFrame
        direction: 'long' or 'short'
        entry_price: 진입 가격
        atr_period: ATR 기간
        atr_multiplier: ATR 배수

    Returns:
        손절 가격
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(atr_period).mean().iloc[-1]

    if direction == "long":
        return entry_price - atr * atr_multiplier
    else:
        return entry_price + atr * atr_multiplier


def calculate_take_profit(
    entry: float,
    stop_loss: float,
    rr_ratio: float = 2.0,
) -> float:
    """
    R:R 기반 목표가 계산.

    Args:
        entry: 진입 가격
        stop_loss: 손절 가격
        rr_ratio: 리스크:리워드 비율 (기본 1:2)

    Returns:
        목표가
    """
    risk = abs(entry - stop_loss)
    if entry > stop_loss:  # Long
        return entry + risk * rr_ratio
    else:  # Short
        return entry - risk * rr_ratio


def calculate_margin(
    qty: float,
    entry_price: float,
    leverage: float,
) -> float:
    """
    레버리지 적용 시 실제 필요 담보금 계산.

    Args:
        qty: 수량 (코인 단위)
        entry_price: 진입 가격
        leverage: 레버리지 배수

    Returns:
        필요 담보금 (USDT)

    Raises:
        ValueError: 레버리지가 0 이하인 경우
    """
    if leverage <= 0:
        raise ValueError("레버리지는 양수여야 합니다.")
    return round((qty * entry_price) / leverage, 8)
