"""메인 트레이딩 엔진 — 데이터 수집 → 전략 분석 → 주문 실행 루프."""

import asyncio
import signal
import threading
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from config.loader import load_config, get_api_credentials
from core.data_feed import DataFeed, WebSocketFeed
from exchange.bybit_client import BybitClient
from strategy.fvg import FVGDetector
from strategy.order_block import OrderBlockDetector
from strategy.structure import StructureAnalyzer
from strategy.kill_zone import KillZoneFilter
from strategy.signal import SignalAggregator
from risk.manager import RiskManager


class TradingEngine:

    def __init__(self):
        self.config = load_config()
        creds = get_api_credentials()
        self.client = BybitClient(creds)
        self.symbol = self.config["trading"]["symbol"]
        self.timeframe = self.config["trading"]["timeframe"]
        self.htf_timeframe = self.config["trading"]["htf_timeframe"]

        self.feed = DataFeed(self.client.http, self.symbol)
        self.ws_feed = WebSocketFeed(
            symbol=self.symbol,
            intervals=[self.timeframe, self.htf_timeframe],
            testnet=creds.get("testnet", True),
        )

        ict_cfg = self.config["ict"]
        self.fvg = FVGDetector(ict_cfg["fvg"])
        self.ob = OrderBlockDetector(ict_cfg["order_block"])
        self.structure = StructureAnalyzer(ict_cfg["structure"])
        self.kz = KillZoneFilter(ict_cfg["kill_zone"])
        self.signal_agg = SignalAggregator()
        self.risk = RiskManager(self.config["risk"], self.client, self.symbol)

        self._scheduler: Optional[BackgroundScheduler] = None
        self._stop_event = threading.Event()

    def _initialize(self):
        trading = self.config["trading"]
        self.client.set_leverage(self.symbol, trading["leverage"])
        self.client.set_margin_mode(self.symbol, trading["mode"])

        info = self.feed.get_instrument_info()
        self.risk.load_instrument_info(info)
        logger.info(
            f"초기화 완료: {self.symbol} | "
            f"레버리지 {trading['leverage']}x | "
            f"모드 {trading['mode']} | "
            f"최소수량 {info['min_qty']} | 틱사이즈 {info['tick_size']}"
        )

    def tick(self):
        try:
            df = self.feed.fetch_klines(self.timeframe)
            df_htf = self.feed.fetch_klines(self.htf_timeframe)

            structure = self.structure.analyze(df_htf)
            fvg_zones = self.fvg.detect(df)
            ob_zones = self.ob.detect(df)
            kz_status = self.kz.is_active()

            logger.debug(
                f"분석 완료 | 추세: {structure['trend']} | "
                f"FVG: {len(fvg_zones)}개 | OB: {len(ob_zones)}개 | "
                f"킬존: {kz_status}"
            )

            signal = self.signal_agg.evaluate(
                structure=structure,
                fvg_zones=fvg_zones,
                ob_zones=ob_zones,
                in_kill_zone=kz_status,
                df=df,
            )

            if signal is None:
                logger.debug("시그널 없음")
                return

            logger.info(f"시그널 감지: {signal['side']} @ {signal['entry']} | {signal['reason']}")

            if not self.risk.can_open_trade(signal):
                return

            order = self.risk.size_position(signal)
            if order is None:
                return

            self.client.place_limit_order(order)

        except Exception as e:
            logger.error(f"틱 처리 에러: {e}")

    def _on_ws_candle(self, candle: dict):
        if candle["confirmed"] and candle["interval"] == self.timeframe:
            logger.debug(f"확정 캔들 수신: {candle['interval']}m | close={candle['close']}")
            self.tick()

    def run(self):
        logger.info(f"=== ICT 트레이딩 봇 시작 ===")
        logger.info(f"심볼: {self.symbol} | TF: {self.timeframe}m | HTF: {self.htf_timeframe}m")

        self._initialize()

        self.tick()

        self._scheduler = BackgroundScheduler()
        interval_min = int(self.timeframe)
        self._scheduler.add_job(self.tick, "interval", minutes=interval_min, id="tick")
        self._scheduler.add_job(self.risk.sync_daily_pnl, "cron", hour=0, minute=0, id="pnl_reset")
        self._scheduler.start()

        self.ws_feed.on_candle(self._on_ws_candle)

        loop = asyncio.new_event_loop()

        def handle_stop(signum, frame):
            logger.info("종료 시그널 수신, 봇 종료 중...")
            self.ws_feed.stop()
            self._scheduler.shutdown(wait=False)
            loop.call_soon_threadsafe(loop.stop)

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        try:
            loop.run_until_complete(self.ws_feed.connect())
        except KeyboardInterrupt:
            handle_stop(None, None)
        finally:
            loop.close()
            logger.info("봇 종료 완료")
