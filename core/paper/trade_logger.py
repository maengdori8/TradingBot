"""거래 기록 — 모든 페이퍼 트레이드를 CSV로 저장, 세션 리포트 생성."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from loguru import logger


class TradeLogger:

    def __init__(self, log_dir: str = "paper_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"trades_{self.session_id}.csv"
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "side", "entry_price", "exit_price",
                "qty", "pnl", "exit_reason", "reason", "duration_min",
            ])
        logger.info(f"[PAPER] 거래 로그: {self.csv_path}")

    def log_trade(self, trade):
        duration = (trade.close_time - trade.open_time) / 60
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(trade.close_time, tz=timezone.utc).isoformat(),
                trade.symbol,
                trade.side,
                trade.entry_price,
                trade.exit_price,
                trade.qty,
                round(trade.pnl, 2),
                trade.exit_reason,
                trade.reason,
                round(duration, 1),
            ])

    def write_report(self, stats: dict):
        report_path = self.log_dir / f"report_{self.session_id}.txt"
        lines = [
            "=" * 50,
            f"  ICT Paper Trading Report",
            f"  세션: {self.session_id}",
            "=" * 50,
            "",
            f"  총 거래 수:     {stats['total_trades']}",
            f"  승리:           {stats.get('wins', 0)}",
            f"  패배:           {stats.get('losses', 0)}",
            f"  승률:           {stats['win_rate']:.1f}%",
            "",
            f"  총 PnL:         {stats['total_pnl']:+.2f} USDT",
            f"  평균 PnL:       {stats['avg_pnl']:+.2f} USDT",
            f"  최고 수익:      {stats['best_trade']:+.2f} USDT",
            f"  최대 손실:      {stats['worst_trade']:+.2f} USDT",
            "",
            f"  평균 수익:      {stats['avg_winner']:+.2f} USDT",
            f"  평균 손실:      {stats['avg_loser']:+.2f} USDT",
            f"  Profit Factor:  {stats['profit_factor']:.2f}",
            "",
            f"  최대 낙폭:      {stats['max_drawdown']:.2f}%",
            f"  수익률:         {stats['return_pct']:+.2f}%",
            f"  최종 잔고:      {stats['balance']:.2f} USDT",
            "",
            "=" * 50,
        ]
        report = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"[PAPER] 리포트 저장: {report_path}")
        return report
