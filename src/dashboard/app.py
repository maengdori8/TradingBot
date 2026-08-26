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



# ------------------------------------------------------------------
# 투 트랙 페이퍼 검증 (Track A 캐리 / Track B 터틀)
# ------------------------------------------------------------------

def _load_track_curves(logs_dir: Path | None = None) -> dict:
    """Track A/B 페이퍼 검증 이력 CSV를 차트 데이터로 변환한다.

    Args:
        logs_dir: 이력 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        {"a": {...}, "b": {...}} — 각 트랙의 labels/pct/현재 상태.
        파일이 없거나 비어 있으면 빈 시리즈를 반환한다 (대시보드는 항상 뜬다).
    """
    import csv as _csv

    logs = logs_dir or (ROOT / "logs")
    spec = {
        "a": ("tracka_history.csv", "Track A — 캐리 (검증 트랙)", "events"),
        "b": ("trackb_history.csv", "Track B — 터틀 (페이퍼 전용)", "fills"),
        "c": ("trackc_history.csv", "Track C — 교차거래소 차익 (ROE는 ÷2)", "day_diff"),
        "d": ("trackd_history.csv", "Track D — 1h 스윙 돌파 (고위험)", "fills"),
    }
    out: dict = {}
    for key, (fname, label, note_col) in spec.items():
        labels: list[str] = []
        pct: list[float] = []
        last: dict = {}
        path = logs / fname
        try:
            with open(path, encoding="utf-8") as f:
                rows = [r for r in _csv.DictReader(f) if r.get("equity")]
            base = float(rows[0]["equity"]) if rows else 1.0
            for r in rows:
                labels.append(r["day"])
                pct.append(round((float(r["equity"]) / base - 1.0) * 100.0, 4))
            if rows:
                last = rows[-1]
        except (FileNotFoundError, ValueError, KeyError, IndexError):
            pass
        out[key] = {
            "label": label,
            "labels": labels,
            "pct": pct,
            "equity": float(last["equity"]) if last.get("equity") else None,
            "n_pos": int(last["n_pos"]) if last.get("n_pos") else 0,
            "note": (last.get(note_col) or "-")[:120],
            "last_day": last.get("day", "-"),
        }
    return out



# ------------------------------------------------------------------
# 트레이더 실력 지속성 연구 (Hyperliquid 코호트)
# ------------------------------------------------------------------

_TRADER_CACHE: dict = {}


def _load_trader_study(logs_dir: Path | None = None) -> dict:
    """지속성 연구 데이터를 차트용으로 변환한다.

    코호트 잠금 시점의 월 ROI로 십분위를 나누고, 잠금 '이후' 일별 손익의
    누적(잠금 시점 계좌 대비 %)을 십분위별 중앙값으로 집계한다.
    상위·하위 십분위 곡선이 벌어지면 지속성 존재의 시각적 신호다.
    판정은 T+30/60/90 사전등록 기준으로만 한다 (이 차트는 참고용).

    Args:
        logs_dir: 이력 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        코호트 메타 + labels/top/bottom/mid 누적 % 시리즈. 데이터 없으면 빈 구조.
    """
    import gzip as _gzip
    import json as _json
    import csv as _csv

    logs = logs_dir or (ROOT / "logs")
    empty = {"available": False, "n": 0, "locked_at": "-", "days": 0,
             "labels": [], "top": [], "bottom": [], "mid": [], "spread": None,
             "verdicts": ["2026-09-24", "2026-10-24", "2026-11-23"]}
    cohort_path = logs / "trader_cohort.json.gz"
    daily_dir = logs / "trader_daily"
    if not cohort_path.exists() or not daily_dir.is_dir():
        return empty

    files = tuple(sorted(daily_dir.glob("*.csv.gz")))
    cache_key = (str(cohort_path), files)
    if _TRADER_CACHE.get("key") == cache_key:
        return _TRADER_CACHE["value"]

    try:
        with _gzip.open(cohort_path, "rt", encoding="utf-8") as f:
            cohort = _json.load(f)
        locked_at = cohort.get("locked_at", "-")
        t0_account: dict[str, float] = {}
        t0_roi: dict[str, float] = {}
        for w in cohort.get("wallets", []):
            acct = float(w.get("t0_account") or 0)
            roi = w.get("t0_month_roi")
            if acct > 0 and roi is not None and roi == roi:
                t0_account[w["address"]] = acct
                t0_roi[w["address"]] = min(max(float(roi), -0.95), 5.0)   # 입출금 왜곡 클리핑
        if len(t0_roi) < 100:
            return empty

        # 십분위 경계 (T0 월 ROI 순위)
        addrs = sorted(t0_roi, key=lambda a: t0_roi[a])
        n = len(addrs)
        bottom_set = set(addrs[: n // 10])
        top_set = set(addrs[-(n // 10):])

        labels: list[str] = []
        top_c: list[float] = []
        bot_c: list[float] = []
        mid_c: list[float] = []
        cum: dict[str, float] = {a: 0.0 for a in t0_roi}
        for fp in files:
            day = fp.name.replace(".csv.gz", "")
            if day <= locked_at:          # 잠금일 스냅샷의 day_pnl은 잠금 이전분 → 제외
                continue
            with _gzip.open(fp, "rt", encoding="utf-8") as f:
                for r in _csv.DictReader(f):
                    a = r.get("address")
                    if a not in cum:
                        continue
                    try:
                        d_ret = float(r["day_pnl"]) / t0_account[a]
                    except (KeyError, ValueError, ZeroDivisionError):
                        continue
                    cum[a] += min(max(d_ret, -1.0), 1.0)   # 일 ±100% 클리핑 (강건성)
            def _median(vals: list[float]) -> float:
                vals = sorted(vals)
                m = len(vals)
                return vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2
            labels.append(day)
            top_c.append(round(_median([cum[a] for a in top_set]) * 100, 4))
            bot_c.append(round(_median([cum[a] for a in bottom_set]) * 100, 4))
            mid_c.append(round(_median(list(cum.values())) * 100, 4))
        out = {"available": True, "n": n, "locked_at": locked_at, "days": len(labels),
               "labels": labels, "top": top_c, "bottom": bot_c, "mid": mid_c,
               "spread": round(top_c[-1] - bot_c[-1], 3) if labels else None,
               "verdicts": ["2026-09-24", "2026-10-24", "2026-11-23"]}
    except (OSError, ValueError, KeyError):
        return empty

    _TRADER_CACHE["key"] = cache_key
    _TRADER_CACHE["value"] = out
    return out



def _read_track_positions(fname: str) -> list[str]:
    """트랙 상태 파일에서 보유 심볼·방향을 읽는다 (없으면 빈 목록)."""
    try:
        d = json.loads((ROOT / "logs" / fname).read_text())
        out = []
        for sym, p in d.get("positions", {}).items():
            dr = p.get("direction", p.get("d", 0))
            out.append(f"{sym} {'롱' if dr > 0 else '숏'}")
        return out
    except (OSError, ValueError, KeyError):
        return []


def _build_summary(balance: float, positions: list[dict], trades: list[dict],
                   tracks: dict, trader_study: dict) -> dict:
    """상단 요약 스트립 데이터 — '지금 한 줄' + 실험 4개 카드.

    모든 값은 이미 로드된 데이터에서 조립한다 (외부 호출 없음 → 페이지 항상 뜬다).
    """
    a, b = tracks.get("a", {}), tracks.get("b", {})
    c, dd_ = tracks.get("c", {}), tracks.get("d", {})
    def _pct(tr):
        return (tr.get("pct") or [0.0])[-1] if tr.get("pct") else 0.0
    a_pct, b_pct, c_pct, d_pct = _pct(a), _pct(b), _pct(c), _pct(dd_)
    a_pos, b_pos = int(a.get("n_pos") or 0), int(b.get("n_pos") or 0)
    b_syms = _read_track_positions("trackb_state.json")
    d_syms = _read_track_positions("trackd_state.json")

    dday = None
    try:
        verdict = trader_study.get("verdicts", [None])[0]
        if verdict:
            dday = (datetime.strptime(verdict, "%Y-%m-%d").date()
                    - datetime.now(timezone.utc).date()).days
    except (ValueError, TypeError):
        pass

    parts = []
    parts.append("터틀 " + "·".join(b_syms) if b_syms else "터틀 대기")
    parts.append("스윙 " + "·".join(d_syms) if d_syms else "스윙 대기")
    parts.append(f"캐리 {a_pos}종목" if a_pos else "캐리 현금 대기")
    parts.append("차익 수취 중" if c.get("labels") else "차익 준비")
    if dday is not None:
        parts.append(f"연구 D-{dday}")
    headline = " · ".join(parts)

    return dict(
        headline=headline,
        dday=dday,
        cards=[
            dict(key="ict", name="ICT 봇 (15분)", color="muted",
                 value=f"{balance:,.0f}", unit="USDT",
                 badge=(f"포지션 {len(positions)}" if positions else "관망"),
                 active=bool(positions),
                 sub=f"거래 {len(trades)}건 · 신호 스캔 중"),
            dict(key="carry", name="Track A · 캐리", color="green",
                 value=f"{a_pct:+.3f}%", unit="",
                 badge=(f"보유 {a_pos}종목" if a_pos else "현금 대기"),
                 active=bool(a_pos),
                 sub=(a.get("note") or "-")[:60]),
            dict(key="turtle", name="Track B · 터틀", color="gold",
                 value=f"{b_pct:+.3f}%", unit="",
                 badge=(f"포지션 {b_pos}" if b_pos else "대기"),
                 active=bool(b_pos),
                 sub=(b.get("note") or "-")[:60]),
            dict(key="xvenue", name="Track C · 차익", color="blue",
                 value=f"{c_pct:+.3f}%", unit="",
                 badge=("수취 중" if c.get("labels") else "준비"),
                 active=bool(c.get("labels")),
                 sub="교차거래소 펀딩 · ROE는 표시값 ÷2"),
            dict(key="swing", name="Track D · 스윙", color="red",
                 value=f"{d_pct:+.3f}%", unit="",
                 badge=("·".join(d_syms) if d_syms else "대기"),
                 active=bool(d_syms),
                 sub="1h 돌파 · 고위험 (MDD 49% 프로파일)"),
            dict(key="study", name="트레이더 지속성", color="blue",
                 value=(f"D-{dday}" if dday is not None else "-"), unit="",
                 badge=f"축적 {trader_study.get('days', 0)}일",
                 active=False,
                 sub=f"코호트 {trader_study.get('n', 0):,} 지갑 · 판정 {trader_study.get('verdicts', ['-'])[0]}"),
        ],
    )


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
    promote = _promote_status(perf, initial_balance)

    conn.close()

    # 각 포지션에 TradingView 심볼 추가
    for p in positions:
        p["tradingview"] = to_tradingview(p["symbol"])

    # 관심종목(watchlist) 스캔 상태 로드
    scan = load_scan_state()
    tracks = _load_track_curves()
    trader_study = _load_trader_study()
    summary = _build_summary(balance, positions, trades, tracks, trader_study)

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
        tracks=tracks,
        tracks_json=json.dumps(tracks),
        trader_study=trader_study,
        trader_study_json=json.dumps(trader_study),
        summary=summary,
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@app.route("/api/status")
def api_status():
    """JSON API — 대시보드 데이터."""
    conn = _get_conn()
    tracks = _load_track_curves()
    trader_study = _load_trader_study()

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
        "tracks": tracks,
        "trader_study": trader_study,
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
