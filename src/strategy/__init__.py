from __future__ import annotations

"""
ICT 전략 패키지 — 공개 API 및 파라미터 로더.
"""

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
from .decision import (
    DecisionContext,
    DecisionFrames,
    iter_decision_contexts,
    slice_closed_bars,
    slice_decision_frames,
)
from .carry_signal import (
    CarryConfig,
    CarrySignal,
    CarrySnapshot,
    generate_carry_signal,
)
from .forced_flow_signal import (
    ForcedFlowConfig,
    ForcedFlowSignal,
    ForcedFlowSnapshot,
    generate_forced_flow_signal,
)
from .signal_engine import generate_signal, TradeSignal
from .market_structure import detect_bos, detect_choch
from .fvg_detector import detect_fvg, FVG
from .order_block import detect_order_blocks, OrderBlock
from .kill_zone import is_in_kill_zone, get_active_session
from .ote import calculate_ote_zone, is_price_in_ote, OTEZone

__all__ = [
    "load_strategy_params",
    "DecisionContext",
    "DecisionFrames",
    "iter_decision_contexts",
    "slice_closed_bars",
    "slice_decision_frames",
    "CarryConfig",
    "CarrySignal",
    "CarrySnapshot",
    "generate_carry_signal",
    "ForcedFlowConfig",
    "ForcedFlowSignal",
    "ForcedFlowSnapshot",
    "generate_forced_flow_signal",
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
