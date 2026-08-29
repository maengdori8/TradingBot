"""
ICT 전략 패키지 — 공개 API 및 파라미터 로더.
"""

from __future__ import annotations

import yaml
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_PARAMS: dict | None = None


def load_strategy_params() -> dict:
    """strategy_params.yaml 에서 전략 파라미터를 로드한다.

    최초 호출 시 파일을 읽고, 이후에는 캐시된 값을 반환한다.

    Returns:
        전략 파라미터 딕셔너리
    """
    global _PARAMS
    if _PARAMS is None:
        with open(_ROOT / "config" / "strategy_params.yaml", encoding="utf-8") as f:
            _PARAMS = yaml.safe_load(f)
    return _PARAMS


# ── 공개 API ─────────────────────────────────────────────────────────
# 순환 import 회피를 위해 load_strategy_params 정의 이후에 재수출한다 (E402 의도적).
from .signal_engine import generate_signal, TradeSignal  # noqa: E402
from .market_structure import detect_bos, detect_choch  # noqa: E402
from .fvg_detector import detect_fvg, FVG  # noqa: E402
from .order_block import detect_order_blocks, OrderBlock  # noqa: E402
from .kill_zone import is_in_kill_zone, get_active_session  # noqa: E402
from .ote import calculate_ote_zone, is_price_in_ote, OTEZone  # noqa: E402

__all__ = [
    "load_strategy_params",
    "generate_signal",
    "TradeSignal",
    "detect_bos",
    "detect_choch",
    "detect_fvg",
    "FVG",
    "detect_order_blocks",
    "OrderBlock",
    "is_in_kill_zone",
    "get_active_session",
    "calculate_ote_zone",
    "is_price_in_ote",
    "OTEZone",
]
