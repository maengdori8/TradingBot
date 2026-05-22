"""시그널 통합 — ICT 조건들을 결합하여 최종 진입 시그널 생성."""

from typing import List, Optional
import pandas as pd
from loguru import logger


class SignalAggregator:

    def evaluate(
        self,
        structure: dict,
        fvg_zones: List[dict],
        ob_zones: List[dict],
        in_kill_zone: dict,
        df: pd.DataFrame,
    ) -> Optional[dict]:
        if not in_kill_zone.get("any", False):
            return None

        trend = structure.get("trend", "neutral")
        if trend == "neutral":
            return None

        last_close = df.iloc[-1]["close"]

        if trend == "bullish":
            entry_zone = self._find_confluence("bullish", fvg_zones, ob_zones, last_close)
            if entry_zone:
                return {
                    "side": "Buy",
                    "entry": entry_zone["level"],
                    "sl": entry_zone["sl"],
                    "reason": entry_zone["reason"],
                }

        elif trend == "bearish":
            entry_zone = self._find_confluence("bearish", fvg_zones, ob_zones, last_close)
            if entry_zone:
                return {
                    "side": "Sell",
                    "entry": entry_zone["level"],
                    "sl": entry_zone["sl"],
                    "reason": entry_zone["reason"],
                }

        return None

    def _find_confluence(
        self, bias: str, fvg_zones: list, ob_zones: list, price: float
    ) -> Optional[dict]:
        fvg_type = "bullish_fvg" if bias == "bullish" else "bearish_fvg"
        ob_type = "bullish_ob" if bias == "bullish" else "bearish_ob"

        relevant_fvgs = [z for z in fvg_zones if z["type"] == fvg_type]
        relevant_obs = [z for z in ob_zones if z["type"] == ob_type]

        for fvg in reversed(relevant_fvgs):
            for ob in reversed(relevant_obs):
                overlap = (
                    max(fvg["bottom"], ob["bottom"]) <= min(fvg["top"], ob["top"])
                )
                if not overlap:
                    continue

                if bias == "bullish" and fvg["bottom"] <= price <= fvg["top"]:
                    return {
                        "level": fvg["midpoint"],
                        "sl": ob["bottom"],
                        "reason": "FVG+OB confluence (bullish)",
                    }
                elif bias == "bearish" and fvg["bottom"] <= price <= fvg["top"]:
                    return {
                        "level": fvg["midpoint"],
                        "sl": ob["top"],
                        "reason": "FVG+OB confluence (bearish)",
                    }

        for fvg in reversed(relevant_fvgs):
            if bias == "bullish" and fvg["bottom"] <= price <= fvg["top"]:
                return {
                    "level": fvg["midpoint"],
                    "sl": fvg["bottom"],
                    "reason": "FVG retest (bullish)",
                }
            elif bias == "bearish" and fvg["bottom"] <= price <= fvg["top"]:
                return {
                    "level": fvg["midpoint"],
                    "sl": fvg["top"],
                    "reason": "FVG retest (bearish)",
                }

        return None
