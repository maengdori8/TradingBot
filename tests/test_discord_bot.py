from __future__ import annotations

"""discord_bot.py (DiscordNotifier) 테스트 — requests.post mock"""

from unittest.mock import patch, MagicMock

import pytest

from src.notification.discord_bot import DiscordNotifier


FAKE_WEBHOOK = "https://discord.com/api/webhooks/123/fake"


@pytest.fixture
def notifier():
    """DiscordNotifier 인스턴스."""
    return DiscordNotifier(webhook_url=FAKE_WEBHOOK)


class TestDiscordNotifier:
    """DiscordNotifier 알림 메서드 테스트."""

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_entry_long(self, mock_post, notifier):
        """Long 진입 알림 — 올바른 payload 전송."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_entry(
            symbol="BTC/USDT:USDT",
            direction="long",
            entry=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            qty=0.01,
            reason="FVG + OB confluence",
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        embeds = payload["embeds"]
        assert len(embeds) == 1
        assert "LONG" in embeds[0]["title"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["long"]

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_entry_short(self, mock_post, notifier):
        """Short 진입 알림 — SHORT 이모지 및 색상."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_entry(
            symbol="ETH/USDT:USDT",
            direction="short",
            entry=3000.0,
            stop_loss=3100.0,
            take_profit=2800.0,
            qty=0.1,
        )

        payload = mock_post.call_args[1]["json"]
        embeds = payload["embeds"]
        assert "SHORT" in embeds[0]["title"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["short"]

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_exit_profit(self, mock_post, notifier):
        """수익 청산 알림 — 초록색 사용."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_exit(
            symbol="BTC/USDT:USDT",
            direction="long",
            exit_price=52000.0,
            pnl=20.0,
            reason="TP",
        )

        payload = mock_post.call_args[1]["json"]
        embeds = payload["embeds"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["long"]
        # TP 이모지 확인
        assert "TP" in embeds[0]["title"]

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_exit_loss(self, mock_post, notifier):
        """손실 청산 알림 — 빨간색 사용."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_exit(
            symbol="BTC/USDT:USDT",
            direction="long",
            exit_price=49000.0,
            pnl=-10.0,
            reason="SL",
        )

        payload = mock_post.call_args[1]["json"]
        embeds = payload["embeds"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["error"]
        assert "SL" in embeds[0]["title"]

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_error(self, mock_post, notifier):
        """에러 알림 전송."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_error("Connection timeout after 30s")

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embeds = payload["embeds"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["error"]
        assert "Connection timeout" in embeds[0]["description"]

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_circuit_breaker(self, mock_post, notifier):
        """서킷브레이커 발동 알림."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_circuit_breaker("연속 3회 손절")

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embeds = payload["embeds"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["warn"]
        assert "연속 3회 손절" in embeds[0]["description"]

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_daily_report(self, mock_post, notifier):
        """일일 리포트 알림 — 모든 필드 포함."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        stats = {
            "total_trades": 10,
            "win_rate": 0.6,
            "total_pnl": 55.25,
            "profit_factor": 1.8,
            "mdd": 0.05,
            "current_balance": 1305.25,
        }

        notifier.notify_daily_report(stats)

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embeds = payload["embeds"]
        assert embeds[0]["color"] == DiscordNotifier.COLOR["report"]
        # 필드 개수 확인
        assert len(embeds[0]["fields"]) == 6

    @patch("src.notification.discord_bot.requests.post")
    def test_webhook_empty_url_no_crash(self, mock_post):
        """Webhook URL이 빈 문자열이면 실패해도 에러 안 남."""
        mock_post.side_effect = Exception("Connection refused")

        notifier = DiscordNotifier(webhook_url="")
        # 모든 알림 메서드가 에러 없이 실행되어야 함
        notifier.notify_entry("BTC/USDT", "long", 50000, 49000, 52000, 0.01)
        notifier.notify_exit("BTC/USDT", "long", 52000, 20.0, "TP")
        notifier.notify_error("test error")
        notifier.notify_circuit_breaker("test")
        notifier.notify_daily_report({"total_pnl": 0})

    @patch("src.notification.discord_bot.requests.post")
    def test_send_returns_true_on_success(self, mock_post, notifier):
        """_send가 성공 시 True 반환."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        result = notifier._send({"embeds": [{"title": "test"}]})
        assert result is True

    @patch("src.notification.discord_bot.requests.post")
    def test_send_returns_false_on_failure(self, mock_post, notifier):
        """_send가 실패 시 False 반환."""
        mock_post.side_effect = Exception("network error")

        result = notifier._send({"embeds": [{"title": "test"}]})
        assert result is False

    @patch("src.notification.discord_bot.requests.post")
    def test_notify_entry_rr_calculation(self, mock_post, notifier):
        """R:R 비율이 올바르게 계산되는지 확인."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        notifier.notify_entry(
            symbol="BTC/USDT:USDT",
            direction="long",
            entry=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            qty=0.01,
        )

        payload = mock_post.call_args[1]["json"]
        rr_field = [f for f in payload["embeds"][0]["fields"] if f["name"] == "R:R"][0]
        assert "2.00" in rr_field["value"]  # (52000-50000)/(50000-49000) = 2.0
