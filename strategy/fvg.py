"""Fair Value Gap (FVG) 탐지 — 3캔들 패턴에서 가격 비효율 영역 식별."""

from typing import List
import pandas as pd


class FVGDetector:

    def __init__(self, config: dict):
        self.min_gap_pct = config.get("min_gap_pct", 0.001)
        self.lookback = config.get("lookback", 50)

    def detect(self, df: pd.DataFrame) -> List[dict]:
        zones = []
        data = df.tail(self.lookback).reset_index(drop=True)

        for i in range(2, len(data)):
            c1 = data.iloc[i - 2]
            c2 = data.iloc[i - 1]
            c3 = data.iloc[i]

            bull_gap = c3["low"] - c1["high"]
            if bull_gap > c2["close"] * self.min_gap_pct:
                zones.append({
                    "type": "bullish_fvg",
                    "top": c3["low"],
                    "bottom": c1["high"],
                    "midpoint": (c3["low"] + c1["high"]) / 2,
                    "timestamp": c2["timestamp"],
                })

            bear_gap = c1["low"] - c3["high"]
            if bear_gap > c2["close"] * self.min_gap_pct:
                zones.append({
                    "type": "bearish_fvg",
                    "top": c1["low"],
                    "bottom": c3["high"],
                    "midpoint": (c1["low"] + c3["high"]) / 2,
                    "timestamp": c2["timestamp"],
                })

        return zones
