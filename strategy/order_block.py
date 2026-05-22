"""Order Block 탐지 — 기관 주문 흔적이 남은 캔들 영역 식별."""

from typing import List
import pandas as pd


class OrderBlockDetector:

    def __init__(self, config: dict):
        self.lookback = config.get("lookback", 20)
        self.min_body_ratio = config.get("min_body_ratio", 0.5)

    def detect(self, df: pd.DataFrame) -> List[dict]:
        zones = []
        data = df.tail(self.lookback + 1).reset_index(drop=True)

        for i in range(1, len(data)):
            prev = data.iloc[i - 1]
            curr = data.iloc[i]

            prev_range = prev["high"] - prev["low"]
            if prev_range == 0:
                continue
            prev_body = abs(prev["close"] - prev["open"])
            body_ratio = prev_body / prev_range

            if body_ratio < self.min_body_ratio:
                continue

            prev_bearish = prev["close"] < prev["open"]
            curr_bullish = curr["close"] > curr["open"]
            if prev_bearish and curr_bullish and curr["close"] > prev["high"]:
                zones.append({
                    "type": "bullish_ob",
                    "top": prev["high"],
                    "bottom": prev["low"],
                    "timestamp": prev["timestamp"],
                })

            prev_bullish = prev["close"] > prev["open"]
            curr_bearish = curr["close"] < curr["open"]
            if prev_bullish and curr_bearish and curr["close"] < prev["low"]:
                zones.append({
                    "type": "bearish_ob",
                    "top": prev["high"],
                    "bottom": prev["low"],
                    "timestamp": prev["timestamp"],
                })

        return zones
