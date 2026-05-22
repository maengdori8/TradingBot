"""캔들 데이터 수집 — REST 폴링 + WebSocket 실시간 피드."""

import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import websockets
from loguru import logger
from pybit.unified_trading import HTTP


class DataFeed:

    def __init__(self, client: HTTP, symbol: str):
        self.client = client
        self.symbol = symbol
        self._ws_candles: Dict[str, pd.DataFrame] = {}

    def fetch_klines(self, interval: str, limit: int = 200) -> pd.DataFrame:
        resp = self.client.get_kline(
            category="linear",
            symbol=self.symbol,
            interval=interval,
            limit=limit,
        )
        rows = resp["result"]["list"]
        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover",
        ])
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float,
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_ticker(self) -> dict:
        resp = self.client.get_tickers(category="linear", symbol=self.symbol)
        item = resp["result"]["list"][0]
        return {
            "last_price": float(item["lastPrice"]),
            "mark_price": float(item["markPrice"]),
            "index_price": float(item["indexPrice"]),
            "volume_24h": float(item["volume24h"]),
            "bid": float(item["bid1Price"]),
            "ask": float(item["ask1Price"]),
        }

    def get_instrument_info(self) -> dict:
        resp = self.client.get_instruments_info(
            category="linear", symbol=self.symbol
        )
        item = resp["result"]["list"][0]
        lot = item["lotSizeFilter"]
        price = item["priceFilter"]
        return {
            "min_qty": float(lot["minOrderQty"]),
            "max_qty": float(lot["maxOrderQty"]),
            "qty_step": float(lot["qtyStep"]),
            "tick_size": float(price["tickSize"]),
            "min_price": float(price["minPrice"]),
            "max_price": float(price["maxPrice"]),
        }


class WebSocketFeed:

    MAINNET_WS = "wss://stream.bybit.com/v5/public/linear"
    TESTNET_WS = "wss://stream-testnet.bybit.com/v5/public/linear"

    def __init__(self, symbol: str, intervals: List[str], testnet: bool = True):
        self.symbol = symbol
        self.intervals = intervals
        self.url = self.TESTNET_WS if testnet else self.MAINNET_WS
        self._callbacks: list = []
        self._running = False

    def on_candle(self, callback):
        self._callbacks.append(callback)

    async def _handle_message(self, msg: str):
        data = json.loads(msg)
        if "topic" not in data:
            return

        topic = data["topic"]
        if not topic.startswith("kline."):
            return

        for candle in data.get("data", []):
            parsed = {
                "symbol": self.symbol,
                "interval": candle["interval"],
                "timestamp": datetime.fromtimestamp(
                    int(candle["start"]) / 1000, tz=timezone.utc
                ),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
                "confirmed": candle["confirm"],
            }
            for cb in self._callbacks:
                try:
                    cb(parsed)
                except Exception as e:
                    logger.error(f"WebSocket 콜백 에러: {e}")

    async def connect(self):
        self._running = True
        topics = [f"kline.{iv}.{self.symbol}" for iv in self.intervals]
        subscribe_msg = {"op": "subscribe", "args": topics}

        while self._running:
            try:
                async with websockets.connect(self.url, ping_interval=20) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"WebSocket 연결됨: {topics}")

                    async for msg in ws:
                        if not self._running:
                            break
                        await self._handle_message(msg)

            except websockets.ConnectionClosed:
                logger.warning("WebSocket 연결 종료, 3초 후 재연결")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"WebSocket 에러: {e}, 5초 후 재연결")
                await asyncio.sleep(5)

    def stop(self):
        self._running = False
