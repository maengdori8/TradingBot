"""
ICT Paper Trading Bot — 메인 실행 파일
GitHub Actions에서 15분마다 실행됨
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# PYTHONPATH 보정 (GitHub Actions 환경 대응)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")


def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def run() -> None:
    from src.exchange.bybit_client import MarketDataClient
    from src.strategy.signal_engine import generate_signal
    from src.risk.risk_manager import RiskManager
    from src.paper_trading.paper_engine import PaperEngine
    from src.notification.discord_bot import DiscordNotifier

    cfg      = load_config()
    client   = MarketDataClient()
    notifier = DiscordNotifier(os.getenv("DISCORD_WEBHOOK_URL", ""))
    risk     = RiskManager()
    paper    = PaperEngine(initial_balance=risk.trading_capital)

    # 청산 이벤트 → 리스크 기록 + Discord 알림 연동
    def _on_trade(pnl: float, reason: str, pos: object) -> None:
        risk.record_result(pnl, reason)
        notifier.notify_exit(
            symbol=pos.symbol, direction=pos.direction,
            exit_price=pos.entry_price, pnl=pnl, reason=reason,
        )
    paper.register_on_trade(_on_trade)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("══════ 봇 실행 [페이퍼] %s ══════", now_str)

    for symbol in cfg["exchange"]["symbols"]:
        logger.info("── %s 분석 ──", symbol)
        try:
            price  = client.fetch_current_price(symbol)
            df_4h  = client.fetch_ohlcv(symbol, "4h",  limit=100)
            df_1h  = client.fetch_ohlcv(symbol, "1h",  limit=100)
            df_15m = client.fetch_ohlcv(symbol, "15m", limit=100)
            logger.info("[%s] 현재가: %.2f", symbol, price)

            # 미실현 손익 갱신
            paper.update_unrealized_pnl(symbol, price)

            # 기존 포지션 SL/TP 체크
            last = df_15m.iloc[-1]
            paper.check_stops(symbol, float(last["high"]), float(last["low"]))

            # 신규 진입 조건 확인 (중복 포지션 차단 포함)
            allowed, reason = risk.check_trade_allowed(
                current_positions=len(paper.positions),
                positions=paper.get_positions(),
                symbol=symbol,
                direction=None,  # 방향은 신호 발생 전이므로 미정
            )
            if not allowed:
                logger.info("[%s] 거래 차단: %s", symbol, reason)
                continue

            signal = generate_signal(df_4h, df_1h, df_15m, symbol, price, risk.min_rr)

            if signal:
                params = risk.calculate_trade_params(signal.entry_price, signal.stop_loss)
                pos = paper.open_position(
                    symbol=symbol, direction=signal.direction,
                    entry_price=signal.entry_price, qty=params["qty"],
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                )
                if pos:
                    notifier.notify_entry(
                        symbol=symbol, direction=signal.direction,
                        entry=pos.entry_price, stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit, qty=pos.qty,
                        reason=signal.reason,
                    )
            else:
                logger.info("[%s] 신호 없음", symbol)

        except Exception as e:
            logger.error("[%s] 오류: %s", symbol, e, exc_info=True)
            notifier.notify_error(f"[{symbol}] {e}")

    # 성과 요약
    perf = paper.get_performance()
    if "total_trades" in perf:
        logger.info(
            "성과 | 거래:%d 승률:%.1f%% PnL:%.2f USDT 잔고:%.2f USDT",
            perf["total_trades"], perf["win_rate"] * 100,
            perf["total_pnl"], perf["current_balance"],
        )
        # 매 6시간(00:00, 06:00, 12:00, 18:00 UTC)에 일일 리포트 발송
        now = datetime.now(timezone.utc)
        if now.hour % 6 == 0 and now.minute < 15:
            notifier.notify_daily_report(perf)
            logger.info("일일 리포트 Discord 발송 완료")

    logger.info("══════ 완료 ══════")


if __name__ == "__main__":
    run()
