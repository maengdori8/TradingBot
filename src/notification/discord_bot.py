"""
Discord 알림 모듈 — Webhook 기반
진입/청산/에러/일일 리포트 알림
"""
from __future__ import annotations
import logging
import requests
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Discord Webhook 알림."""

    # 색상 코드
    COLOR = {
        "long":   0x1D9E75,  # 초록
        "short":  0xE24B4A,  # 빨강
        "info":   0x378ADD,  # 파랑
        "warn":   0xEF9F27,  # 노랑
        "error":  0xA32D2D,  # 어두운 빨강
        "report": 0x7F77DD,  # 보라
    }

    def __init__(self, webhook_url: str) -> None:
        """
        Args:
            webhook_url: Discord 채널 Webhook URL
        """
        self.webhook_url = webhook_url

    # ------------------------------------------------------------------

    def _send(self, payload: dict) -> bool:
        """Webhook 전송."""
        try:
            res = requests.post(self.webhook_url, json=payload, timeout=10)
            res.raise_for_status()
            return True
        except Exception as e:
            logger.error("Discord 알림 실패: %s", e)
            return False

    def _embed(
        self,
        title: str,
        description: str,
        color: int,
        fields: list[dict] | None = None,
    ) -> dict:
        """Embed 페이로드 생성."""
        embed: dict = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "ICT Trading Bot"},
        }
        if fields:
            embed["fields"] = fields
        return {"embeds": [embed]}

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def notify_entry(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        entry: float,
        stop_loss: float,
        take_profit: float,
        qty: float,
        reason: str = "",
    ) -> None:
        """진입 알림."""
        emoji = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        rr = abs(take_profit - entry) / abs(entry - stop_loss) if entry != stop_loss else 0
        payload = self._embed(
            title=f"{emoji} 진입 — {symbol}",
            description=f"**이유**: {reason}" if reason else "",
            color=self.COLOR[direction],
            fields=[
                {"name": "진입가",  "value": f"`{entry:,.4f}`",     "inline": True},
                {"name": "손절가",  "value": f"`{stop_loss:,.4f}`", "inline": True},
                {"name": "목표가",  "value": f"`{take_profit:,.4f}`","inline": True},
                {"name": "수량",    "value": f"`{qty:.4f}`",         "inline": True},
                {"name": "R:R",    "value": f"`1 : {rr:.2f}`",      "inline": True},
            ],
        )
        self._send(payload)
        logger.info("Discord 진입 알림 전송: %s %s", symbol, direction)

    def notify_exit(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        exit_price: float,
        pnl: float,
        reason: str = "",
    ) -> None:
        """청산 알림."""
        is_profit = pnl >= 0
        emoji = "✅ TP" if reason == "TP" else ("❌ SL" if reason == "SL" else "🔄 청산")
        color = self.COLOR["long"] if is_profit else self.COLOR["error"]
        payload = self._embed(
            title=f"{emoji} — {symbol}",
            description="",
            color=color,
            fields=[
                {"name": "청산가",  "value": f"`{exit_price:,.4f}`",      "inline": True},
                {"name": "PnL",    "value": f"`{'+'if is_profit else ''}{pnl:.2f} USDT`", "inline": True},
                {"name": "사유",    "value": f"`{reason}`",                "inline": True},
            ],
        )
        self._send(payload)

    def notify_error(self, message: str) -> None:
        """에러 알림."""
        payload = self._embed(
            title="⚠️ 봇 오류",
            description=f"```{message}```",
            color=self.COLOR["error"],
        )
        self._send(payload)

    def notify_circuit_breaker(self, reason: str) -> None:
        """서킷브레이커 발동 알림."""
        payload = self._embed(
            title="🛑 서킷브레이커 발동",
            description=f"**사유**: {reason}\n거래가 일시 중단됩니다.",
            color=self.COLOR["warn"],
        )
        self._send(payload)

    def notify_promote(self, result: object) -> None:
        """실전 전환 기준 충족 알림.

        Args:
            result: PromoteResult 객체 (eligible, score, summary 속성)
        """
        fields = [
            {"name": "점수", "value": f"`{result.score:.0f}/100`", "inline": True},
        ]
        # 각 기준 결과 표시
        if hasattr(result, "criteria"):
            for name, cr in result.criteria.items():
                emoji = "✅" if cr.passed else "❌"
                fields.append({
                    "name": f"{emoji} {cr.name}",
                    "value": f"`{cr.value:.4f}` / 기준 `{cr.threshold:.4f}`",
                    "inline": True,
                })
        payload = self._embed(
            title="🏆 실전 전환 기준 충족!",
            description=result.summary,
            color=0xFFD700,  # 골드
            fields=fields,
        )
        self._send(payload)
        logger.info("Discord 실전 전환 알림 전송")

    def notify_daily_report(self, stats: dict) -> None:
        """일일 성과 리포트."""
        pnl = stats.get("total_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        payload = self._embed(
            title="📊 일일 트레이딩 리포트",
            description=f"**{datetime.now(timezone.utc).strftime('%Y-%m-%d')} UTC 기준**",
            color=self.COLOR["report"],
            fields=[
                {"name": "총 거래",    "value": f"`{stats.get('total_trades', 0)}건`",           "inline": True},
                {"name": "승률",       "value": f"`{stats.get('win_rate', 0)*100:.1f}%`",          "inline": True},
                {"name": "일일 PnL",   "value": f"`{sign}{pnl:.2f} USDT`",                        "inline": True},
                {"name": "Profit Factor","value": f"`{stats.get('profit_factor', 0):.2f}`",        "inline": True},
                {"name": "MDD",        "value": f"`{stats.get('mdd', 0)*100:.2f}%`",               "inline": True},
                {"name": "잔고",       "value": f"`{stats.get('current_balance', 0):,.2f} USDT`",  "inline": True},
            ],
        )
        self._send(payload)
