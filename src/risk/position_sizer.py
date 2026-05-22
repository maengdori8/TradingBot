"""
포지션 사이징 — 리스크 % 기반 수량 계산
"""
from __future__ import annotations
import pandas as pd
import numpy as np


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
