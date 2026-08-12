from __future__ import annotations

# Flask 기반 페이퍼 트레이딩 대시보드.
# 사용법: python -m src.dashboard.app [--port 5000] [--host 0.0.0.0]


import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from flask import Flask, jsonify, render_template

from src.scan_store import load_scan_state, to_tradingview

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "logs" / "paper_trades.db"
CB_DB_PATH = ROOT / "logs" / "circuit_breaker.db"

# Windows cp949 콘솔 출력 인코딩 방어
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

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
        with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
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
        daily_limit = risk.get("daily_loss_limit", 0.05)
        max_consec = risk.get("max_consecutive_losses", 7)

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


def _promote_status(perf: dict, initial_balance: float = 1250.0) -> dict:
    """실전 전환 기준 충족 상태.

    Args:
        perf: 성과 지표 dict
        initial_balance: 수익률 계산 기준 초기 자본
    """
    cfg = _load_config()
    promote = cfg.get("promote", {})

    criteria = {
        "거래 수": {
            "value": perf.get("total_trades", 0),
            "threshold": promote.get("min_trades", 30),
            "passed": perf.get("total_trades", 0) >= promote.get("min_trades", 30),
            "format": "d",
        },
        "승률": {
            "value": perf.get("win_rate", 0) * 100,
            "threshold": promote.get("min_win_rate", 0.38) * 100,
            "passed": perf.get("win_rate", 0) >= promote.get("min_win_rate", 0.38),
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
            "threshold": promote.get("max_mdd", 0.10) * 100,
            "passed": perf.get("mdd", 1) <= promote.get("max_mdd", 0.10),
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
            "value": (perf.get("total_pnl", 0) / initial_balance) * 100
                     if perf.get("total_trades", 0) > 0 and initial_balance > 0 else 0,
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
        "criteria_met": passed_count == total_count,
        "eligible": False,
        "informational_only": True,
        "frozen": bool(promote.get("frozen", True)),
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
    promote = _promote_status(perf, initial_balance)

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
# 실시간 가격 (보유 포지션 평가손익용, 1초 캐시)
# ------------------------------------------------------------------

_price_cache: dict[str, tuple[float, float]] = {}   # symbol -> (price, fetched_at)
_PRICE_TTL = 1.0
_market_client = None


def _get_market_client():
    """MarketDataClient 싱글톤 (lazy)."""
    global _market_client
    if _market_client is None:
        from src.exchange.bybit_client import MarketDataClient
        _market_client = MarketDataClient()
    return _market_client


def _live_price(symbol: str) -> float | None:
    """현재가 조회 (1초 캐시). 실패 시 None."""
    now = time.time()
    cached = _price_cache.get(symbol)
    if cached and (now - cached[1]) < _PRICE_TTL:
        return cached[0]
    try:
        price = _get_market_client().fetch_current_price(symbol)
        _price_cache[symbol] = (price, now)
        return price
    except Exception as e:
        logger.debug("실시간 가격 조회 실패 %s: %s", symbol, e)
        return cached[0] if cached else None


@app.route("/api/live")
def api_live():
    """초단위 폴링용 — 잔고 + 보유 포지션 실시간 평가손익."""
    conn = _get_conn()
    cfg = _load_config()
    cap = cfg.get("capital", {})
    initial_balance = cap.get("total_capital", 5000) * cap.get("trading_allocation", 0.25)
    balance = _fetch_balance(conn)
    positions = _fetch_open_positions(conn)
    conn.close()

    total_upnl = 0.0
    live_positions = []
    for p in positions:
        price = _live_price(p["symbol"])
        entry = p["entry_price"]
        qty = p["qty"]
        if price is not None:
            upnl = (price - entry) * qty if p["direction"] == "long" else (entry - price) * qty
        else:
            price, upnl = entry, 0.0
        total_upnl += upnl
        live_positions.append({
            "symbol": p["symbol"],
            "direction": p["direction"],
            "entry_price": entry,
            "current_price": round(price, 6),
            "qty": qty,
            "stop_loss": p["stop_loss"],
            "take_profit": p["take_profit"],
            "margin": p.get("margin", 0),
            "unrealized_pnl": round(upnl, 4),
            "unrealized_pct": round(upnl / p["margin"] * 100, 2) if p.get("margin") else 0.0,
            "tradingview": to_tradingview(p["symbol"]),
        })

    return jsonify({
        "balance": round(balance, 2),
        "total_unrealized": round(total_upnl, 4),
        "equity": round(balance + total_upnl, 2),
        "initial_balance": initial_balance,
        "return_pct": round((balance + total_upnl - initial_balance) / initial_balance * 100, 2) if initial_balance else 0.0,
        "open_positions": live_positions,
        "position_count": len(live_positions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _lan_ip() -> str:
    """현재 머신의 LAN IP 추정 (외부 접속 안내용)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    """대시보드 서버 실행."""
    parser = argparse.ArgumentParser(description="ICT Paper Trading Dashboard")
    parser.add_argument("--host", default="127.0.0.1",
                        help="바인딩 호스트 (외부 접속은 0.0.0.0, 기본: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="포트 번호 (기본: 5000)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    args = parser.parse_args()

    logger.info("=" * 56)
    logger.info("ICT Paper Trading Dashboard")
    logger.info("=" * 56)
    logger.info("로컬:   http://127.0.0.1:%d", args.port)
    if args.host == "0.0.0.0":
        logger.info("외부(같은 WiFi): http://%s:%d", _lan_ip(), args.port)
        logger.info("⚠ 외부 노출 — 인증 없는 읽기전용 화면. 신뢰된 네트워크에서만 사용")
    else:
        logger.info("외부 접속하려면: --host 0.0.0.0 옵션으로 재실행")
    logger.info("DB: %s", DB_PATH)
    logger.info("=" * 56)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
