"""시장 구조 분석 — BOS (Break of Structure) / CHoCH (Change of Character) 탐지."""

from typing import List
import pandas as pd


class StructureAnalyzer:

    def __init__(self, config: dict):
        self.lookback = config.get("lookback", 50)
        self.swing_strength = config.get("swing_strength", 3)

    def _find_swing_points(self, df: pd.DataFrame) -> List[dict]:
        swings = []
        n = self.swing_strength

        for i in range(n, len(df) - n):
            high_is_peak = all(
                df.iloc[i]["high"] >= df.iloc[i + j]["high"]
                for j in range(-n, n + 1) if j != 0
            )
            if high_is_peak:
                swings.append({
                    "type": "swing_high",
                    "price": df.iloc[i]["high"],
                    "index": i,
                    "timestamp": df.iloc[i]["timestamp"],
                })

            low_is_trough = all(
                df.iloc[i]["low"] <= df.iloc[i + j]["low"]
                for j in range(-n, n + 1) if j != 0
            )
            if low_is_trough:
                swings.append({
                    "type": "swing_low",
                    "price": df.iloc[i]["low"],
                    "index": i,
                    "timestamp": df.iloc[i]["timestamp"],
                })

        swings.sort(key=lambda s: s["index"])
        return swings

    def analyze(self, df: pd.DataFrame) -> dict:
        data = df.tail(self.lookback).reset_index(drop=True)
        swings = self._find_swing_points(data)

        result = {
            "trend": "neutral",
            "swings": swings,
            "bos": [],
            "choch": [],
        }

        highs = [s for s in swings if s["type"] == "swing_high"]
        lows = [s for s in swings if s["type"] == "swing_low"]

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1]["price"] > highs[-2]["price"]
            hl = lows[-1]["price"] > lows[-2]["price"]
            lh = highs[-1]["price"] < highs[-2]["price"]
            ll = lows[-1]["price"] < lows[-2]["price"]

            if hh and hl:
                result["trend"] = "bullish"
            elif lh and ll:
                result["trend"] = "bearish"

            last_price = data.iloc[-1]["close"]

            if result["trend"] == "bullish" and last_price < lows[-1]["price"]:
                result["choch"].append({
                    "direction": "bearish_choch",
                    "level": lows[-1]["price"],
                })
            elif result["trend"] == "bearish" and last_price > highs[-1]["price"]:
                result["choch"].append({
                    "direction": "bullish_choch",
                    "level": highs[-1]["price"],
                })

            if hh:
                result["bos"].append({
                    "direction": "bullish_bos",
                    "level": highs[-2]["price"],
                })
            if ll:
                result["bos"].append({
                    "direction": "bearish_bos",
                    "level": lows[-2]["price"],
                })

        return result
