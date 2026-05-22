"""터미널 대시보드 — 실시간 잔고, 포지션, 성과 통계 출력."""

import os
import sys
from datetime import datetime, timezone


class Dashboard:

    def __init__(self):
        self._last_price = 0.0
        self._start_time = datetime.now(timezone.utc)

    def update_price(self, price: float):
        self._last_price = price

    def render(self, stats: dict, positions: list, symbol: str):
        elapsed = datetime.now(timezone.utc) - self._start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)

        lines = []
        lines.append("")
        lines.append("\033[36m" + "=" * 60 + "\033[0m")
        lines.append("\033[1m  ICT Paper Trading Dashboard\033[0m")
        lines.append("\033[36m" + "=" * 60 + "\033[0m")
        lines.append(f"  {symbol} | 현재가: {self._last_price:,.2f} | 경과: {hours}h {minutes}m")
        lines.append("")

        pnl = stats["total_pnl"]
        pnl_color = "\033[32m" if pnl >= 0 else "\033[31m"
        ret = stats["return_pct"]
        ret_color = "\033[32m" if ret >= 0 else "\033[31m"

        lines.append(f"  잔고:  {stats['balance']:>12,.2f} USDT")
        lines.append(f"  평가:  {stats['equity']:>12,.2f} USDT")
        lines.append(f"  PnL:   {pnl_color}{pnl:>+12,.2f} USDT\033[0m")
        lines.append(f"  수익률: {ret_color}{ret:>+11.2f}%\033[0m")
        lines.append("")
        lines.append("\033[36m" + "-" * 60 + "\033[0m")
        lines.append(
            f"  거래: {stats['total_trades']} | "
            f"승률: {stats['win_rate']:.1f}% | "
            f"PF: {stats['profit_factor']:.2f} | "
            f"MDD: {stats['max_drawdown']:.1f}%"
        )

        if positions:
            lines.append("")
            lines.append("\033[36m" + "-" * 60 + "\033[0m")
            lines.append("  \033[1m열린 포지션:\033[0m")
            for p in positions:
                pnl_str = f"{p.unrealized_pnl:+.2f}"
                p_color = "\033[32m" if p.unrealized_pnl >= 0 else "\033[31m"
                lines.append(
                    f"    {p.side:<4} {p.qty} @ {p.entry_price:,.2f} | "
                    f"SL: {p.sl:,.2f} TP: {p.tp:,.2f} | "
                    f"{p_color}{pnl_str}\033[0m"
                )

        lines.append("\033[36m" + "=" * 60 + "\033[0m")
        lines.append("")

        output = "\n".join(lines)

        if sys.stdout.isatty():
            os.system("clear" if os.name != "nt" else "cls")
        print(output)
