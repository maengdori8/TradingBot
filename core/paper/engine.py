"""페이퍼 트레이딩 엔진 — 실시간 시세 + ICT 전략 + 가상 체결."""

import asyncio
import signal
import math
from typing import Optional

from loguru import logger

from config.loader import load_config, get_api_credentials
from core.data_feed import DataFeed, WebSocketFeed
from core.paper.account import PaperAccount
from core.paper.trade_logger import TradeLogger
from core.paper.dashboard import Dashboard
from exchange.bybit_client import BybitClient
from strategy.fvg import FVGDetector
from strategy.order_block import OrderBlockDetector
from strategy.structure import StructureAnalyzer
from strategy.kill_zone import KillZoneFilter
from strategy.signal import SignalAggregator
from utils.healthcheck import HealthCheck
from utils.notifier import DiscordNotifier


class PaperEngine:

    def __init__(self):
        self.config = load_config()
        creds = get_api_credentials()

        self.client = BybitClient(creds)
        self.symbol = self.config["trading"]["symbol"]
        self.timeframe = self.config["trading"]["timeframe"]
        self.htf_timeframe = self.config["trading"]["htf_timeframe"]

        self.feed = DataFeed(self.client.http, self.symbol)

        paper_cfg = self.config.get("paper", {})
        self.account = PaperAccount(
            initial_balance=paper_cfg.get("initial_balance", 10000.0),
            leverage=self.config["trading"]["leverage"],
        )

        self.trade_logger = TradeLogger()
        self.dashboard = Dashboard()

        ict_cfg = self.config["ict"]
        self.fvg = FVGDetector(ict_cfg["fvg"])
        self.ob = OrderBlockDetector(ict_cfg["order_block"])
        self.structure = StructureAnalyzer(ict_cfg["structure"])
        self.kz = KillZoneFilter(ict_cfg["kill_zone"])
        self.signal_agg = SignalAggregator()

        risk_cfg = self.config["risk"]
        self.max_risk = risk_cfg.get("max_risk_per_trade", 0.01)
        self.max_daily_loss = risk_cfg.get("max_daily_loss", 0.03)
        self.max_positions = risk_cfg.get("max_open_positions", 3)
        self.rr_ratio = risk_cfg.get("risk_reward_ratio", 2.0)

        self._instrument_info: Optional[dict] = None
        self._last_price = 0.0
        self._tick_count = 0
        self._ws_feed: Optional[WebSocketFeed] = None
        self.health = HealthCheck()
        self.notifier = DiscordNotifier()

    def _load_instrument(self):
        self._instrument_info = self.feed.get_instrument_info()
        logger.info(
            f"[PAPER] 심볼 정보: {self.symbol} | "
            f"최소수량: {self._instrument_info['min_qty']} | "
            f"틱사이즈: {self._instrument_info['tick_size']}"
        )

    def _round_qty(self, qty: float) -> float:
        step = self._instrument_info["qty_step"]
        min_qty = self._instrument_info["min_qty"]
        qty = math.floor(qty / step) * step
        qty = round(qty, 8)
        return qty if qty >= min_qty else 0.0

    def _round_price(self, price: float) -> float:
        tick = self._instrument_info["tick_size"]
        return round(round(price / tick) * tick, 8)

    def _can_open(self) -> bool:
        open_count = len(self.account.positions)
        if open_count >= self.max_positions:
            return False

        balance = self.account.get_balance()
        if balance <= 0:
            return False

        stats = self.account.get_stats()
        daily_pnl = stats["total_pnl"]
        if abs(daily_pnl) >= self.account.initial_balance * self.max_daily_loss:
            logger.warning("[PAPER] 일일 최대 손실 한도 도달")
            return False

        return True

    def _size_position(self, signal_data: dict) -> Optional[dict]:
        balance = self.account.get_balance()
        entry = signal_data["entry"]
        sl = signal_data["sl"]
        distance = abs(entry - sl)

        if distance == 0:
            return None

        risk_amount = balance * self.max_risk
        qty = risk_amount / distance
        qty = self._round_qty(qty)

        if qty <= 0:
            return None

        if signal_data["side"] == "Buy":
            tp = entry + distance * self.rr_ratio
        else:
            tp = entry - distance * self.rr_ratio

        return {
            "symbol": self.symbol,
            "side": signal_data["side"],
            "entry": self._round_price(entry),
            "sl": self._round_price(sl),
            "tp": self._round_price(tp),
            "qty": qty,
            "reason": signal_data.get("reason", ""),
        }

    def tick(self):
        try:
            df = self.feed.fetch_klines(self.timeframe)
            df_htf = self.feed.fetch_klines(self.htf_timeframe)

            if df.empty or df_htf.empty:
                return

            self._last_price = float(df.iloc[-1]["close"])
            self.dashboard.update_price(self._last_price)

            closed = self.account.check_exits(self.symbol, self._last_price)
            for trade in closed:
                self.trade_logger.log_trade(trade)

            for pos in self.account.positions:
                pos.update_pnl(self._last_price)

            structure = self.structure.analyze(df_htf)
            fvg_zones = self.fvg.detect(df)
            ob_zones = self.ob.detect(df)
            kz_status = self.kz.is_active()

            self._tick_count += 1
            logger.debug(
                f"[PAPER] 틱 #{self._tick_count} | 가격: {self._last_price:,.2f} | "
                f"추세: {structure['trend']} | FVG: {len(fvg_zones)} | OB: {len(ob_zones)} | "
                f"킬존: {'ON' if kz_status.get('any') else 'OFF'}"
            )

            signal = self.signal_agg.evaluate(
                structure=structure,
                fvg_zones=fvg_zones,
                ob_zones=ob_zones,
                in_kill_zone=kz_status,
                df=df,
            )

            self.health.record_tick()

            if signal and self._can_open():
                order = self._size_position(signal)
                if order:
                    self.account.open_position(order)

            self.dashboard.render(
                stats=self.account.get_stats(),
                positions=self.account.positions,
                symbol=self.symbol,
            )

        except Exception as e:
            logger.error(f"[PAPER] 틱 에러: {e}")

    def _on_ws_candle(self, candle: dict):
        price = candle["close"]
        self._last_price = price
        self.dashboard.update_price(price)

        closed = self.account.check_exits(self.symbol, price)
        for trade in closed:
            self.trade_logger.log_trade(trade)
            self.notifier.send_sync(self.notifier.format_trade(trade))

        for pos in self.account.positions:
            pos.update_pnl(price)

        if candle["confirmed"] and candle["interval"] == self.timeframe:
            logger.debug(f"[PAPER] 확정 캔들: {candle['interval']}m close={price:,.2f}")
            self.tick()

    def _shutdown(self):
        self.health.stop()
        stats = self.account.get_stats()
        report = self.trade_logger.write_report(stats)
        print(report)
        self.notifier.send_sync(self.notifier.format_daily_report(stats))

    def run(self):
        logger.info("[PAPER] === ICT Paper Trading 시작 ===")
        logger.info(
            f"[PAPER] {self.symbol} | TF: {self.timeframe}m | HTF: {self.htf_timeframe}m | "
            f"초기 잔고: {self.account.initial_balance:,.2f} USDT"
        )

        self._load_instrument()
        self.health.start()
        self.tick()

        creds = get_api_credentials()
        self._ws_feed = WebSocketFeed(
            symbol=self.symbol,
            intervals=[self.timeframe, self.htf_timeframe],
            testnet=creds.get("testnet", True),
        )
        self._ws_feed.on_candle(self._on_ws_candle)

        loop = asyncio.new_event_loop()

        def handle_stop(signum, frame):
            logger.info("[PAPER] 종료 시그널 수신...")
            self._ws_feed.stop()
            loop.call_soon_threadsafe(loop.stop)

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        try:
            loop.run_until_complete(self._ws_feed.connect())
        except KeyboardInterrupt:
            handle_stop(None, None)
        finally:
            loop.close()
            self._shutdown()
            logger.info("[PAPER] 봇 종료 완료")
