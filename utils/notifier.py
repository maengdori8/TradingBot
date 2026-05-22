"""디스코드 웹훅 알림 — 거래 체결, 에러, 일일 리포트 전송."""

import os
from datetime import datetime, timezone

import aiohttp
from loguru import logger


class DiscordNotifier:

    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.enabled = bool(self.webhook_url)
        if self.enabled:
            logger.info("[NOTIFY] 디스코드 웹훅 알림 활성화됨")
        else:
            logger.debug("[NOTIFY] DISCORD_WEBHOOK_URL 미설정 — 알림 비활성화")

    async def send(self, embed: dict):
        if not self.enabled:
            return
        payload = {"embeds": [embed]}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status not in (200, 204):
                        logger.warning(f"[NOTIFY] 디스코드 전송 실패: {resp.status}")
        except Exception as e:
            logger.warning(f"[NOTIFY] 디스코드 에러: {e}")

    def send_sync(self, embed: dict):
        if not self.enabled:
            return
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send(embed))
            else:
                loop.run_until_complete(self.send(embed))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.send(embed))
            loop.close()

    def format_trade(self, trade) -> dict:
        is_win = trade.pnl >= 0
        color = 0x2ECC71 if is_win else 0xE74C3C
        duration = (trade.close_time - trade.open_time) / 60

        return {
            "title": f"{'✅' if is_win else '❌'} {trade.side} {trade.symbol} — {trade.exit_reason.upper()}",
            "color": color,
            "fields": [
                {"name": "진입가", "value": f"`{trade.entry_price:,.2f}`", "inline": True},
                {"name": "청산가", "value": f"`{trade.exit_price:,.2f}`", "inline": True},
                {"name": "수량", "value": f"`{trade.qty}`", "inline": True},
                {"name": "PnL", "value": f"**{trade.pnl:+,.2f} USDT**", "inline": True},
                {"name": "보유 시간", "value": f"{duration:.0f}분", "inline": True},
                {"name": "사유", "value": trade.reason, "inline": True},
            ],
            "timestamp": datetime.fromtimestamp(trade.close_time, tz=timezone.utc).isoformat(),
        }

    def format_daily_report(self, stats: dict) -> dict:
        pnl = stats["total_pnl"]
        color = 0x2ECC71 if pnl >= 0 else 0xE74C3C

        return {
            "title": "📊 ICT Bot 리포트",
            "color": color,
            "fields": [
                {"name": "총 거래", "value": f"{stats['total_trades']}", "inline": True},
                {"name": "승률", "value": f"{stats['win_rate']:.1f}%", "inline": True},
                {"name": "Profit Factor", "value": f"{stats['profit_factor']:.2f}", "inline": True},
                {"name": "총 PnL", "value": f"**{pnl:+,.2f} USDT**", "inline": True},
                {"name": "수익률", "value": f"{stats['return_pct']:+.2f}%", "inline": True},
                {"name": "MDD", "value": f"{stats['max_drawdown']:.1f}%", "inline": True},
                {"name": "잔고", "value": f"{stats['balance']:,.2f} USDT", "inline": True},
                {"name": "최고 수익", "value": f"{stats['best_trade']:+,.2f}", "inline": True},
                {"name": "최대 손실", "value": f"{stats['worst_trade']:+,.2f}", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def format_signal(self, signal: dict, symbol: str) -> dict:
        color = 0x3498DB if signal["side"] == "Buy" else 0xF39C12
        return {
            "title": f"🔔 시그널 감지 — {signal['side']} {symbol}",
            "color": color,
            "fields": [
                {"name": "진입가", "value": f"`{signal['entry']:,.2f}`", "inline": True},
                {"name": "SL", "value": f"`{signal['sl']:,.2f}`", "inline": True},
                {"name": "사유", "value": signal.get("reason", ""), "inline": False},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def format_error(self, error_msg: str) -> dict:
        return {
            "title": "⚠️ 봇 에러",
            "color": 0xE67E22,
            "description": f"```{error_msg[:1000]}```",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
