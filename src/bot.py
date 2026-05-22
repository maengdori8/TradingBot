"""
메인 봇 — 페이퍼 트레이딩 전용
실시간 Bybit 시세 → ICT 신호 분석 → 가상 주문 실행 → Discord 알림
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "bot.log"),
    ],
)
logger = logging.getLogger("bot")


def load_config() -> dict:
    path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def run() -> None:
    from src.exchange.bybit_client import BybitPublicClient
    from src.strategy.signal_engine import generate_signal
    from src.risk.risk_manager import RiskManager
    from src.paper_trading.paper_engine import PaperEngine
    from src.notification.discord_bot import DiscordNotifier

    cfg     = load_config()
    client  = BybitPublicClient()
    notifier= DiscordNotifier(os.getenv("DISCORD_WEBHOOK_URL", ""))
    risk    = RiskManager()
    paper   = PaperEngine(initial_balance=risk.trading_capital)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("════════ 봇 실행 [페이퍼] %s ════════", now_str)

    for symbol in cfg["exchange"]["symbols"]:
        logger.info("── %s 분석 시작 ──", symbol)
        try:
            # ── 실시간 데이터 수집 ──────────────────────
            price  = client.fetch_current_price(symbol)
            df_4h  = client.fetch_ohlcv(symbol, "4h",  limit=100)
            df_1h  = client.fetch_ohlcv(symbol, "1h",  limit=100)
            df_15m = client.fetch_ohlcv(symbol, "15m", limit=100)

            logger.info("[%s] 현재가: %.2f", symbol, price)

            # ── 기존 포지션 SL/TP 체크 ──────────────────
            last = df_15m.iloc[-1]
            paper.check_stops(symbol, float(last["high"]), float(last["low"]))

            # ── 신규 신호 탐지 ───────────────────────────
            allowed, reason = risk.check_trade_allowed(len(paper.positions))
            if not allowed:
                logger.info("[%s] 거래 차단: %s", symbol, reason)
                continue

            signal = generate_signal(df_4h, df_1h, df_15m, symbol, price, risk.min_rr)

            if signal:
                params = risk.calculate_trade_params(signal.entry_price, signal.stop_loss)
                pos = paper.open_position(
                    symbol    = symbol,
                    direction = signal.direction,
                    entry_price = signal.entry_price,
                    qty       = params["qty"],
                    stop_loss = signal.stop_loss,
                    take_profit = signal.take_profit,
                )
                if pos:
                    notifier.notify_entry(
                        symbol    = symbol,
                        direction = signal.direction,
                        entry     = pos.entry_price,
                        stop_loss = pos.stop_loss,
                        take_profit = pos.take_profit,
                        qty       = pos.qty,
                        reason    = signal.reason,
                    )
            else:
                logger.info("[%s] 조건 미충족 — 신호 없음", symbol)

        except Exception as e:
            logger.error("[%s] 오류: %s", symbol, e, exc_info=True)
            notifier.notify_error(f"[{symbol}] {e}")

    # ── 성과 요약 로깅 ───────────────────────────────────
    perf = paper.get_performance()
    if "total_trades" in perf:
        logger.info(
            "성과 요약 | 거래:%d 승률:%.1f%% PnL:%.2f 잔고:%.2f",
            perf["total_trades"], perf["win_rate"]*100,
            perf["total_pnl"], perf["current_balance"],
        )

    logger.info("════════ 실행 완료 ════════")


if __name__ == "__main__":
    run()
