"""
페이퍼 트레이딩 대시보드 — Flask 기반 웹 UI
실시간 잔고, 성과 지표, 오픈 포지션, 거래 내역, 에쿼티 커브를 제공한다.

사용법:
    python -m src.dashboard.app [--port 5000] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from flask import Flask, jsonify, render_template

from src.scan_store import load_scan_state, to_tradingview

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "logs" / "paper_trades.db"
CB_DB_PATH = ROOT / "logs" / "circuit_breaker.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))


# ------------------------------------------------------------------
# 설정 로드
# ------------------------------------------------------------------

def _load_config() -> dict:
    """config.yaml 로드."""
    try:
        with open(ROOT / "config" / "config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


# ------------------------------------------------------------------
# DB 헬퍼
# ------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """paper_trades.db 연결. 없으면 빈 DB 생성."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL,
            qty REAL, pnl REAL, pnl_pct REAL,
            margin REAL,
            entry_time TEXT, exit_time TEXT, status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS engine_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id TEXT PRIMARY KEY,
            symbol TEXT, direction TEXT,
            entry_price REAL, qty REAL,
            stop_loss REAL, take_profit REAL,
            margin REAL, entry_time TEXT
        )
    """)
    return conn


def _get_cb_conn() -> sqlite3.Connection | None:
    """circuit_breaker.db 연결. 없으면 None."""
    if not CB_DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(CB_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# 데이터 수집 함수
# ------------------------------------------------------------------

def _fetch_balance(conn: sqlite3.Connection) -> float:
    """현재 잔고 조회."""
    row = conn.execute(
        "SELECT value FROM engine_state WHERE key = 'balance'"
    ).fetchone()
    if row:
        return float(row["value"])
    cfg = _load_config()
    cap = cfg.get("capital", {})
    return cap.get("total_capital", 5000) * cap.get("trading_allocation", 0.25)


def _fetch_trades(conn: sqlite3.Connection) -> list[dict]:
    """전체 거래 내역 조회 (최신순)."""
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY exit_time DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_open_positions(conn: sqlite3.Connection) -> list[dict]:
    """오픈 포지션 조회."""
    rows = conn.execute(
        "SELECT * FROM open_positions ORDER BY entry_time DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def _calc_performance(trades: list[dict], initial_balance: float) -> dict:
    """성과 지표 계산."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "long_count": 0,
            "short_count": 0,
        }

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # 에쿼티 커브 (시간순 — trades는 최신순이므로 뒤집기)
    pnls_chrono = list(reversed(pnls))
    equity = np.array(
        [initial_balance]
        + list(np.cumsum(pnls_chrono) + initial_balance)
    )
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
    mdd = float(drawdown.max())

    pnl_arr = np.array(pnl_pcts)
    sharpe = (
        float(pnl_arr.mean() / pnl_arr.std() * math.sqrt(252))
        if len(pnl_arr) > 1 and pnl_arr.std() > 0
        else 0.0
    )

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss

    long_count = sum(1 for t in trades if t["direction"] == "long")
    short_count = sum(1 for t in trades if t["direction"] == "short")

    return {
        "total_trades": len(pnls),
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
        "total_pnl": round(sum(pnls), 4),
        "avg_pnl": round(sum(pnls) / len(pnls), 4),
        "mdd": round(mdd, 6),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "best_trade": round(max(pnls), 4),
        "worst_trade": round(min(pnls), 4),
        "avg_win": round(sum(wins) / len(wins), 4) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
        "long_count": long_count,
        "short_count": short_count,
    }


def _build_equity_curve(trades: list[dict], initial_balance: float) -> dict:
    """에쿼티 커브 데이터 생성."""
    if not trades:
        return {"labels": [], "values": []}

    # 시간순 정렬
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time", ""))
    labels = ["시작"]
    values = [round(initial_balance, 2)]

    running = initial_balance
    for t in sorted_trades:
        running += t["pnl"]
        exit_time = t.get("exit_time", "")
        if exit_time:
            # ISO -> 짧은 형식
            try:
                dt = datetime.fromisoformat(exit_time)
                label = dt.strftime("%m/%d %H:%M")
            except (ValueError, TypeError):
                label = exit_time[:16]
        else:
            label = "?"
        labels.append(label)
        values.append(round(running, 2))

    return {"labels": labels, "values": values}


def _fetch_circuit_breaker_status() -> dict:
    """서킷브레이커 상태 조회."""
    conn = _get_cb_conn()
    if conn is None:
        return {"daily_pnl": 0.0, "consecutive_losses": 0, "is_blocked": False}

    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT SUM(pnl) as total FROM trade_results WHERE date(recorded_at) = ?",
            (today,),
        ).fetchone()
        daily_pnl = float(row["total"]) if row and row["total"] else 0.0

        rows = conn.execute(
            "SELECT pnl FROM trade_results ORDER BY recorded_at DESC LIMIT 10"
        ).fetchall()

        consecutive = 0
        for r in rows:
            if r["pnl"] < 0:
                consecutive += 1
            else:
                break

        cfg = _load_config()
        risk = cfg.get("risk", {})
        cap = cfg.get("capital", {})
        trading_cap = cap.get("total_capital", 5000) * cap.get("trading_allocation", 0.25)
        daily_limit = risk.get("daily_loss_limit", 0.03)
        max_consec = risk.get("max_consecutive_losses", 3)

        is_blocked = (
            daily_pnl < -(trading_cap * daily_limit)
            or consecutive >= max_consec
        )

        conn.close()
        return {
            "daily_pnl": round(daily_pnl, 4),
            "consecutive_losses": consecutive,
            "is_blocked": is_blocked,
        }
    except Exception as e:
        logger.warning("서킷브레이커 상태 조회 실패: %s", e)
        if conn:
            conn.close()
        return {"daily_pnl": 0.0, "consecutive_losses": 0, "is_blocked": False}


def _promote_status(perf: dict) -> dict:
    """실전 전환 기준 충족 상태."""
    cfg = _load_config()
    promote = cfg.get("promote", {})

    criteria = {
        "거래 수": {
            "value": perf.get("total_trades", 0),
            "threshold": promote.get("min_trades", 20),
            "passed": perf.get("total_trades", 0) >= promote.get("min_trades", 20),
            "format": "d",
        },
        "승률": {
            "value": perf.get("win_rate", 0) * 100,
            "threshold": promote.get("min_win_rate", 0.55) * 100,
            "passed": perf.get("win_rate", 0) >= promote.get("min_win_rate", 0.55),
            "format": ".1f",
            "suffix": "%",
        },
        "Profit Factor": {
            "value": perf.get("profit_factor", 0),
            "threshold": promote.get("min_profit_factor", 1.5),
            "passed": perf.get("profit_factor", 0) >= promote.get("min_profit_factor", 1.5),
            "format": ".2f",
        },
        "MDD": {
            "value": perf.get("mdd", 1) * 100,
            "threshold": promote.get("max_mdd", 0.05) * 100,
            "passed": perf.get("mdd", 1) <= promote.get("max_mdd", 0.05),
            "format": ".2f",
            "suffix": "%",
            "lower_better": True,
        },
        "Sharpe": {
            "value": perf.get("sharpe", 0),
            "threshold": promote.get("min_sharpe", 1.0),
            "passed": perf.get("sharpe", 0) >= promote.get("min_sharpe", 1.0),
            "format": ".2f",
        },
        "수익률": {
            "value": (perf.get("total_pnl", 0) / 1250) * 100 if perf.get("total_trades", 0) > 0 else 0,
            "threshold": promote.get("min_return_pct", 0) * 100,
            "passed": True if perf.get("total_pnl", 0) >= 0 else False,
            "format": ".2f",
            "suffix": "%",
        },
    }

    passed_count = sum(1 for c in criteria.values() if c["passed"])
    total_count = len(criteria)

    return {
        "criteria": criteria,
        "passed_count": passed_count,
        "total_count": total_count,
        "eligible": passed_count == total_count,
    }


# ------------------------------------------------------------------
# 라우트
# ------------------------------------------------------------------

@app.route("/")
def index():
    """메인 대시보드."""
    conn = _get_conn()

    cfg = _load_config()
    cap = cfg.get("capital", {})
    initial_balance = cap.get("total_capital", 5000) * cap.get("trading_allocation", 0.25)

    balance = _fetch_balance(conn)
    trades = _fetch_trades(conn)
    positions = _fetch_open_positions(conn)
    perf = _calc_performance(trades, initial_balance)
    equity = _build_equity_curve(trades, initial_balance)
    cb_status = _fetch_circuit_breaker_status()
    promote = _promote_status(perf)

    conn.close()

    # 각 포지션에 TradingView 심볼 추가
    for p in positions:
        p["tradingview"] = to_tradingview(p["symbol"])

    # 관심종목(watchlist) 스캔 상태 로드
    scan = load_scan_state()

    return render_template(
        "index.html",
        balance=balance,
        initial_balance=initial_balance,
        trades=trades,
        positions=positions,
        perf=perf,
        equity_json=json.dumps(equity),
        cb_status=cb_status,
        promote=promote,
        config=cfg,
        scan=scan,
        watchlist=scan.get("watchlist", []),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@app.route("/api/status")
def api_status():
    """JSON API — 대시보드 데이터."""
    conn = _get_conn()

    cfg = _load_config()
    cap = cfg.get("capital", {})
    initial_balance = cap.get("total_capital", 5000) * cap.get("trading_allocation", 0.25)

    balance = _fetch_balance(conn)
    trades = _fetch_trades(conn)
    positions = _fetch_open_positions(conn)
    perf = _calc_performance(trades, initial_balance)
    equity = _build_equity_curve(trades, initial_balance)
    cb_status = _fetch_circuit_breaker_status()

    conn.close()

    scan = load_scan_state()
    for p in positions:
        p["tradingview"] = to_tradingview(p["symbol"])

    return jsonify({
        "balance": balance,
        "initial_balance": initial_balance,
        "performance": perf,
        "open_positions": positions,
        "recent_trades": trades[:20],
        "equity_curve": equity,
        "circuit_breaker": cb_status,
        "scan": scan,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    """대시보드 서버 실행."""
    parser = argparse.ArgumentParser(description="ICT Paper Trading Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="바인딩 호스트 (기본: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="포트 번호 (기본: 5000)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("ICT Paper Trading Dashboard")
    logger.info("=" * 50)
    logger.info("URL: http://%s:%d", args.host, args.port)
    logger.info("DB: %s", DB_PATH)
    logger.info("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
