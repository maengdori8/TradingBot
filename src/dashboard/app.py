"""
페이퍼 트레이딩 대시보드 — Flask 기반 웹 UI
실시간 잔고, 성과 지표, 오픈 포지션, 거래 내역, 에쿼티 커브를 제공한다.

사용법:
    python -m src.dashboard.app [--port 5000] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timezone
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

# Track C 표기 규율 (docs/XVENUE_ARBITRAGE_2026-08.md 사전등록: "ROE = 차익/2 병기")
# trackc_history.csv 의 equity 는 **명목(양다리 차익) 기준**이다. 담보가 두
# 거래소에 이중 소요되므로 자기자본 수익률(ROE)은 명목 수익률의 1/2 이다.
# 표시할 때는 둘 다 명시한다 — 한쪽만 그리고 "÷2" 만 적으면 오독을 부른다.
TRACKC_ROE_DIVISOR: float = 2.0
TRACKC_NOTIONAL_LABEL: str = "명목 (양다리 차익 기준)"
TRACKC_ROE_LABEL: str = "ROE (담보 이중 · 명목÷2)"
# 사후 발견 결함 공시 — 카드에 상시 노출 (숨기지 않는다)
TRACKC_CAVEAT: str = "2024-12 인샘플 청산 발견 — 과거 수치는 상한(정정 공시 참조)"


def _load_track_curves(logs_dir: Path | None = None) -> dict:
    """Track A/B 페이퍼 검증 이력 CSV를 차트 데이터로 변환한다.

    Args:
        logs_dir: 이력 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        {"a": {...}, "b": {...}} — 각 트랙의 labels/pct/현재 상태.
        pct 는 각 트랙의 기준(basis)에 따른 수익률이다: Track C 만 명목
        (양다리 차익) 기준이라 roe_pct(=pct÷TRACKC_ROE_DIVISOR)를 함께 싣고
        caveat(정정 공시)을 붙인다 — 나머지 트랙은 자본 기준이라 roe_pct 는
        None 이다. 파일이 없거나 비어 있으면 빈 시리즈를 반환한다
        (대시보드는 항상 뜬다).
    """
    import csv as _csv

    logs = logs_dir or (ROOT / "logs")
    spec = {
        "a": ("tracka_history.csv", "Track A — 캐리 (검증 트랙)", "events"),
        "b": ("trackb_history.csv", "Track B — 터틀 (페이퍼 전용)", "fills"),
        "c": ("trackc_history.csv",
              "Track C — 교차거래소 차익 (명목 기준 · ROE 병기)", "day_diff"),
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
        is_c = key == "c"
        out[key] = {
            "label": label,
            "labels": labels,
            "pct": pct,
            # 이 트랙의 pct 가 "무엇에 대한 수익률"인지 — 라벨이 정확히 말한다
            "basis": TRACKC_NOTIONAL_LABEL if is_c else "자본",
            "roe_pct": ([round(p / TRACKC_ROE_DIVISOR, 4) for p in pct]
                        if is_c else None),
            "roe_label": TRACKC_ROE_LABEL if is_c else None,
            "caveat": TRACKC_CAVEAT if is_c else None,
            "equity": float(last["equity"]) if last.get("equity") else None,
            "n_pos": int(last["n_pos"]) if last.get("n_pos") else 0,
            "note": (last.get(note_col) or "-")[:120],
            "last_day": last.get("day", "-"),
        }
    return out



# ------------------------------------------------------------------
# Track E — 단타 팜 (페이퍼 전용 · 사후선택 시연, 2026-08-27 사전등록)
# ------------------------------------------------------------------
# 표시 전용 로더다. Track E 상태·이력은 승급/실거래 게이트(_promote_status,
# src/risk/promote_checker)나 /api/live 경로에 절대 입력하지 않는다 (방화벽).

# 셀 구성 (명세 동결 — 고정 순서·고정 라벨, 성과순 정렬 금지)
TRACKE_CELLS: tuple[tuple[str, str, str, str], ...] = (
    ("E01", "BRK24", "A", "역사적 탈락"),
    ("E02", "BRK24", "B", "역사적 탈락"),
    ("E03", "BRK48", "A", "역사적 탈락"),
    ("E04", "BRK48", "B", "역사적 탈락"),
    ("E05", "BRK96", "A", "격자 1/8 선택 · Track D 중복 · 선택할인"),
    ("E06", "BRK96", "B", "격자 1/8 선택 · Track D 중복 · 선택할인"),
    ("E07", "MR", "A", "역사적 탈락"),
    ("E08", "MR", "B", "역사적 탈락"),
    ("E09", "RSI-DIV 4h", "A", "미검증 가설 U1"),
    ("E10", "RSI-DIV 4h", "B", "미검증 가설 U1"),
)
TRACKE_CELL_IDS: tuple[str, ...] = tuple(c[0] for c in TRACKE_CELLS)
TRACKE_BASKET_LABELS: dict[str, str] = {
    "A": "BTC·ETH·SOL", "B": "OOD·미검증 코인셋"}
TRACKE_VERDICT_DATES: tuple[str, ...] = (
    "2026-09-26", "2026-11-25", "2027-02-23")
TRACKE_CELL_CAPITAL: float = 10_000.0        # 셀당 가상 자본 (USD)

# 변형 셀 (표시 전용 분리 소구역 — 공식 판정 대상 아님)
# 셀 목록·전략명·바스켓·라벨은 엔진 동결 상수(carrybot.aggressive.scalp_farm 의
# V*CELLS/V*LABELS 그룹 — VCELLS·V2CELLS, ...)에서 읽는다 — 엔진에 셀·그룹이
# 추가돼도 대시보드는 무수정 (데이터 주도). 본 표(E01~E10)와 섞지 않으며
# "사후 최대값" 태그 계산에도 절대 넣지 않는다.
# 아래는 엔진 import 실패 시에만 쓰는 동결 폴백 (2026-08-28 E11·E12 분).
TRACKE_VARIANT_FALLBACK_CELLS: tuple[tuple[str, str, str], ...] = (
    ("E11", "BRK24TP", "A"),
    ("E12", "BRK24TP", "B"),
)
TRACKE_VARIANT_FALLBACK_LABEL: str = "빠른 익절 변형 · 미검증 · 판정 권한 없음"
TRACKE_VARIANT_BASKET_LABELS: dict[str, str] = {
    "A": "BTC·ETH·SOL", "B": "XRP·HYPE·BTR"}


def _tracke_variant_spec() -> tuple[
        tuple[tuple[str, str, str], ...], dict[str, str], dict[str, str]]:
    """변형 셀 명세를 엔진 동결 상수에서 읽는다 (데이터 주도 — 표시 전용).

    carrybot.aggressive.scalp_farm 의 변형 그룹 상수 V<번호>CELLS(셀·전략·
    바스켓 고정 순서)와 짝 라벨 V<번호>LABELS 를 그룹 번호 순으로 합친다 —
    VCELLS(=1그룹, E11·E12)·V2CELLS(E13~E18), 이후 V3CELLS 등 그룹이
    추가돼도 대시보드는 무수정으로 따라간다. 엔진이 바스켓 표기 dict 를
    노출하면 그것을 우선 쓰고, 없으면 폴백 표기(A/B)를 쓴다. import/속성
    오류 시 동결 폴백 E11·E12 를 반환한다 (크래시 금지). 상수만 읽는 순수
    함수이며 엔진 상태·성과에는 일절 접근하지 않는다.

    Returns:
        (cells, labels, basket_labels) — cells 는 (id, strategy, basket)
        그룹 번호 순·그룹 내 정의 순 튜플(성과순 정렬 금지),
        labels 는 {셀 id: 라벨}, basket_labels 는 {바스켓: 구성 표기}.
    """
    try:
        sf = importlib.import_module("carrybot.aggressive.scalp_farm")
        groups: list[tuple[int, tuple, dict]] = []
        for name in dir(sf):
            m = re.fullmatch(r"V(\d*)CELLS", name)
            if not m:
                continue
            raw_labels = getattr(sf, f"V{m.group(1)}LABELS", None)
            groups.append((int(m.group(1) or 1), getattr(sf, name),
                           raw_labels if isinstance(raw_labels, dict) else {}))
        groups.sort(key=lambda g: g[0])
        cells: list[tuple[str, str, str]] = []
        labels: dict[str, str] = {}
        for _, specs, raw in groups:
            for s in specs:
                cid = str(s.cell)
                cells.append((cid, str(s.strategy), str(s.basket)))
                labels[cid] = str(raw.get(cid, TRACKE_VARIANT_FALLBACK_LABEL))
        if not cells:
            raise ValueError("변형 셀 상수(V*CELLS) 비어 있음")
        baskets = dict(TRACKE_VARIANT_BASKET_LABELS)
        for name in ("VBASKET_LABELS", "VBASKETS", "BASKET_LABELS"):
            eng = getattr(sf, name, None)
            if isinstance(eng, dict):
                baskets.update({str(k): "·".join(v) if isinstance(
                    v, (list, tuple)) else str(v) for k, v in eng.items()})
                break
        return tuple(cells), labels, baskets
    except (ImportError, AttributeError, TypeError, ValueError):
        labels = {cid: TRACKE_VARIANT_FALLBACK_LABEL
                  for cid, _, _ in TRACKE_VARIANT_FALLBACK_CELLS}
        return (TRACKE_VARIANT_FALLBACK_CELLS, labels,
                dict(TRACKE_VARIANT_BASKET_LABELS))


def _tracke_variant_blocks(st: dict) -> list[dict]:
    """상태 JSON 의 변형 병렬 블록(variant<번호>_cells)을 그룹 번호 순으로 찾는다.

    variant_cells(=1그룹, E11·E12)·variant2_cells(E13~E18) 등 별도 t0 키를
    가진 병렬 블록 구조 — 그룹이 늘어도 무수정. dict 가 아닌 블록은 손상으로
    보고 건너뛴다 (크래시 금지).

    Args:
        st: tracke_state.json 파싱 결과 dict.

    Returns:
        블록 dict 목록 (그룹 번호 순) — 없으면 빈 목록.
    """
    out: list[tuple[int, dict]] = []
    for k, v in st.items():
        m = re.fullmatch(r"variant(\d*)_cells", str(k))
        if m and isinstance(v, dict):
            out.append((int(m.group(1) or 1), v))
    out.sort(key=lambda t: t[0])
    return [v for _, v in out]


def _tracke_metric(cell_state: dict, names: tuple[str, ...]) -> float | None:
    """셀 상태 dict에서 후보 키 중 첫 유효 수치를 읽는다 (없으면 None)."""
    for n in names:
        v = cell_state.get(n)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            return float(v)
    return None


def _tracke_ts_sort_key(raw: str) -> tuple:
    """원시 시각 키 정렬 키 — 숫자(epoch)는 수치로, 그 외는 문자열로 비교."""
    v = (raw or "").strip()
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


def _tracke_ts_label(raw: str) -> str:
    """원시 시각 키를 표시용 라벨로 변환 (표시 직전에만 호출).

    epoch ms 는 연도 포함 "%y-%m-%d %H:%M" 로 포맷해 연도 경계에서도
    라벨이 충돌하지 않게 한다. 그 외 형식은 원문 그대로 반환한다.
    """
    raw = (raw or "").strip()
    try:
        num = float(raw)
        if num > 1e11:
            return datetime.fromtimestamp(
                num / 1000.0, tz=timezone.utc).strftime("%y-%m-%d %H:%M")
    except ValueError:
        pass
    return raw


def _tracke_history_series(
    path: Path,
    cell_ids: tuple[str, ...] = TRACKE_CELL_IDS,
) -> dict[str, tuple[list[str], list[float]]]:
    """tracke(_variant)_history.csv 를 셀별 (원시 시각 키, 자본) 시리즈로 읽는다.

    두 형식을 허용한다 (엔진 스키마 관용, 손상 행은 개별 스킵):
    - wide: 시각 열(ts|day|bar_close|time) + 셀 열 (셀별 자본, 대소문자 무관)
    - long: 시각 열 + cell + equity

    시각 키는 포맷하지 않은 원시 문자열(epoch ms 그대로)로 반환한다 —
    집계·정렬은 원시 키로 하고, 표시 라벨은 _tracke_ts_label 로
    표시 직전에만 만든다 (연도 경계 정렬·라벨 충돌 방지).

    Args:
        path: 이력 CSV 경로.
        cell_ids: 읽을 셀 id 집합 (기본 본 셀 E01~E10, 변형은 E11~E12).

    Returns:
        {셀 id: (원시 시각 키 리스트, 자본 리스트)} — 파일 없음/손상 시 빈 dict.
    """
    import csv as _csv

    try:
        with open(path, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fields = [fn for fn in (reader.fieldnames or []) if fn]
            rows = [r for r in reader if r]
    except (OSError, _csv.Error):
        return {}
    if not rows or not fields:
        return {}

    lower = {fn.lower().strip(): fn for fn in fields}
    tkey = next((lower[k] for k in ("ts", "day", "bar_close", "time", "date")
                 if k in lower), None)
    if tkey is None:
        return {}

    rows.sort(key=lambda r: _tracke_ts_sort_key(r.get(tkey) or ""))

    out: dict[str, tuple[list[str], list[float]]] = {}
    cell_cols = {fn.upper().strip(): fn for fn in fields
                 if fn.upper().strip() in cell_ids}
    if cell_cols:                               # wide 형식
        for r in rows:
            ts_key = (r.get(tkey) or "").strip()
            for cid, col in cell_cols.items():
                try:
                    eq = float(r[col])
                except (KeyError, TypeError, ValueError):
                    continue
                keys, eqs = out.setdefault(cid, ([], []))
                keys.append(ts_key)
                eqs.append(eq)
    elif "cell" in lower and "equity" in lower:  # long 형식
        ckey, ekey = lower["cell"], lower["equity"]
        for r in rows:
            cid = (r.get(ckey) or "").upper().strip()
            if cid not in cell_ids:
                continue
            try:
                eq = float(r[ekey])
            except (TypeError, ValueError):
                continue
            keys, eqs = out.setdefault(cid, ([], []))
            keys.append((r.get(tkey) or "").strip())
            eqs.append(eq)
    return out


def _load_tracke(logs_dir: Path | None = None) -> dict:
    """Track E 단타 팜 표시 데이터를 조립한다 (표시 전용 — 게이트 입력 금지).

    E01~E10 고정 순서를 항상 유지하고(성과순 정렬 금지), 셀별 고정 라벨·
    동일가중 팜 곡선·중앙값/IQR·지표를 만든다. 현재 최대 셀에는 상시
    "사후 최대값 — 선택 금지" 태그용 is_max 플래그만 단다 (강조 없음).
    데이터가 하나도 없으면 available=False ("T0 대기" 표시, 크래시 금지).

    Args:
        logs_dir: 이력 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        available/t0/cells(고정 순서)/farm(labels·mean·median·q1·q3)/
        max_cell/verdicts 를 담은 dict (JSON 직렬화 가능).
    """
    logs = logs_dir or (ROOT / "logs")
    series = _tracke_history_series(logs / "tracke_history.csv")

    try:
        st = json.loads((logs / "tracke_state.json").read_text(encoding="utf-8"))
        if not isinstance(st, dict):
            st = {}
    except (OSError, ValueError):
        st = {}
    raw_cells = st.get("cells")
    if not isinstance(raw_cells, dict):
        raw_cells = {k: v for k, v in st.items()
                     if isinstance(v, dict) and k.upper() in TRACKE_CELL_IDS}
    cells_state = {k.upper(): v for k, v in raw_cells.items()
                   if isinstance(v, dict)}

    t0 = st.get("t0") or st.get("t0_ts")
    if t0 is not None:
        try:                                    # 엔진은 epoch ms 로 기록
            num = float(t0)
            if num > 1e11:
                t0 = datetime.fromtimestamp(
                    num / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            pass
    if t0 is None:
        for keys, _ in series.values():
            if keys:
                t0 = _tracke_ts_label(keys[0])
                break

    cells: list[dict] = []
    for cid, strat, basket, label in TRACKE_CELLS:
        s = cells_state.get(cid, {})
        _, eqs = series.get(cid, ([], []))
        pct: float | None = None
        mdd: float | None = None
        if eqs and eqs[0] > 0:
            base = eqs[0]
            pct = round((eqs[-1] / base - 1.0) * 100.0, 4)
            peak, worst = float("-inf"), 0.0
            for eq in eqs:
                peak = max(peak, eq)
                if peak > 0:
                    worst = max(worst, (peak - eq) / peak)
            mdd = round(worst * 100.0, 4)
        else:
            eq = _tracke_metric(s, ("equity", "eq"))
            if eq is not None:
                # 1.0 기준 정규화 자본과 USD($10,000 기준) 둘 다 허용
                pct = (round((eq - 1.0) * 100.0, 4) if eq < 100.0
                       else round((eq / TRACKE_CELL_CAPITAL - 1.0) * 100.0, 4))
        halts = _tracke_metric(s, ("halts", "n_halts", "halt_count", "halt_days"))
        cells.append(dict(
            id=cid, strategy=strat, basket=basket,
            basket_label=TRACKE_BASKET_LABELS[basket], label=label,
            pct=pct, mdd=mdd,
            cost=_tracke_metric(s, ("cost_cum", "costs", "cost", "fees")),
            funding=_tracke_metric(s, ("funding_cum", "funding", "fund")),
            turnover=_tracke_metric(s, ("turnover", "turnover_cum")),
            gross=_tracke_metric(s, ("gross", "gross_exposure")),
            halts=int(halts) if halts is not None else None,
            is_max=False,
        ))

    max_cell = None
    with_pct = [c for c in cells if c["pct"] is not None]
    if with_pct:
        best = max(with_pct, key=lambda c: c["pct"])   # 동률 → 고정 순서 앞 셀
        best["is_max"] = True
        max_cell = best["id"]

    # 동일가중 팜 곡선 + 셀 간 중앙값/IQR (누적 %)
    # 집계·정렬은 원시 시각 키(epoch ms 등)로 하고 표시 라벨은 마지막에만
    # 포맷한다 — 연도 경계(2026-12→2027-01) 순서 붕괴·라벨 충돌 방지.
    pct_series: dict[str, dict[str, float]] = {}
    all_keys: list[str] = []
    seen: set[str] = set()
    for cid in TRACKE_CELL_IDS:                        # 고정 순서 순회
        keys, eqs = series.get(cid, ([], []))
        if not eqs or eqs[0] <= 0:
            continue
        base = eqs[0]
        m = pct_series.setdefault(cid, {})
        for k, eq in zip(keys, eqs):
            m[k] = (eq / base - 1.0) * 100.0
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    all_keys.sort(key=_tracke_ts_sort_key)
    farm = {"labels": [], "mean": [], "median": [], "q1": [], "q3": []}
    for k in all_keys:
        vals = [m[k] for m in pct_series.values() if k in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        farm["labels"].append(_tracke_ts_label(k))
        farm["mean"].append(round(float(arr.mean()), 4))
        farm["median"].append(round(float(np.median(arr)), 4))
        farm["q1"].append(round(float(np.percentile(arr, 25)), 4))
        farm["q3"].append(round(float(np.percentile(arr, 75)), 4))

    return {
        "available": bool(series) or bool(cells_state),
        "t0": t0,
        "cells": cells,
        "farm": farm,
        "max_cell": max_cell,
        "verdicts": list(TRACKE_VERDICT_DATES),
    }


def _load_tracke_variant(logs_dir: Path | None = None) -> dict:
    """변형 셀 표시 데이터 — 표시 전용 분리 소구역 (데이터 주도).

    셀 목록·전략명·바스켓·라벨은 _tracke_variant_spec() (엔진 동결 상수,
    폴백 E11·E12)에서 읽는다 — 엔진에 셀이 추가돼도 여기는 무수정.
    공식 판정 대상이 아니다: 본 셀(E01~E10) 표·"사후 최대값 — 선택 금지"
    태그 계산과 완전히 분리되며(_load_tracke 는 이 데이터를 일절 안 본다),
    승급/실거래 게이트에도 입력하지 않는다. 수익률은 통합 변형 이력
    (tracke_variant_history.csv) 우선, 없으면 상태의 병렬 블록
    (variant_cells·variant2_cells, ... — _tracke_variant_blocks)에서 셀
    equity 로 계산한다. 아직 상태·이력이 없는 셀(별도 t0 키 미가동 포함)은
    pct None ("대기" 표시)으로 남긴다. 파일·키 부재/손상 시
    available=False ("대기" 표시, 크래시 금지).

    Args:
        logs_dir: 이력 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        {"available": bool, "cells": [엔진 명세 고정 순서 — id/strategy/
         basket/basket_label/label/pct/positions]} (JSON 직렬화 가능).
    """
    vcells, vlabels, vbaskets = _tracke_variant_spec()
    vids = tuple(c[0] for c in vcells)
    logs = logs_dir or (ROOT / "logs")
    series = _tracke_history_series(
        logs / "tracke_variant_history.csv", cell_ids=vids)

    try:
        st = json.loads((logs / "tracke_state.json").read_text(encoding="utf-8"))
        if not isinstance(st, dict):
            st = {}
    except (OSError, ValueError):
        st = {}
    cells_state: dict[str, dict] = {}
    for blk in _tracke_variant_blocks(st):        # 병렬 블록 전부 병합
        raw_cells = blk.get("cells")
        if not isinstance(raw_cells, dict):
            continue
        for k, v in raw_cells.items():
            if isinstance(v, dict):
                cells_state.setdefault(k.upper(), v)   # 그룹 번호 앞 블록 우선

    cells: list[dict] = []
    for cid, strat, basket in vcells:                 # 고정 순서 (정렬 금지)
        s = cells_state.get(cid, {})
        _, eqs = series.get(cid, ([], []))
        pct: float | None = None
        if eqs and eqs[0] > 0:
            pct = round((eqs[-1] / eqs[0] - 1.0) * 100.0, 4)
        else:
            eq = _tracke_metric(s, ("equity", "eq"))
            if eq is not None:
                # 1.0 기준 정규화 자본과 USD($10,000 기준) 둘 다 허용
                pct = (round((eq - 1.0) * 100.0, 4) if eq < 100.0
                       else round((eq / TRACKE_CELL_CAPITAL - 1.0) * 100.0, 4))
        positions: list[str] = []
        pos = s.get("positions")
        if isinstance(pos, dict):
            for sym, pp in pos.items():
                try:
                    d_ = int(pp["d"])
                except (KeyError, TypeError, ValueError):
                    continue
                positions.append(f"{sym} {'롱' if d_ > 0 else '숏'}")
        cells.append(dict(
            id=cid, strategy=strat, basket=basket,
            basket_label=vbaskets.get(basket, basket),
            label=vlabels.get(cid, TRACKE_VARIANT_FALLBACK_LABEL),
            pct=pct, positions=positions,
        ))
    return {
        "available": bool(series) or bool(cells_state),
        "cells": cells,
    }


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



# ------------------------------------------------------------------
# H2 꾸준함 가설 연구 (하방 일관성 — 2026-08-27 사전등록)
# ------------------------------------------------------------------

# 사전등록 고정 판정 일정 (변경 금지 — 판정일·명칭은 명세 문서 기준)
H2_SCHEDULE: tuple[tuple[str, str], ...] = (
    ("2026-09-24", "H1 T+30"),
    ("2026-09-26", "트랙A T+30"),
    ("2026-10-24", "H1 T+60"),
    ("2026-10-26", "트랙A T+60"),
    ("2026-11-23", "H1 T+90"),
    ("2026-11-25", "트랙A T+90"),
    ("2026-11-26", "트랙B 형성종료"),
    ("2026-12-26", "트랙B T+30"),
    ("2027-01-25", "트랙B T+60 (게이트 확정)"),
    ("2027-02-24", "트랙B T+90"),
)


def _h2_cohort_meta(path: Path) -> dict | None:
    """h2_cohort.json.gz 헤더에서 1차 적격 수·MDE IC를 읽는다.

    Args:
        path: 코호트 파일 경로 (gzip JSON, {"header": {...}, "wallets": [...]}).

    Returns:
        {"n": 1차 적격 지갑 수, "mde_ic": MDE IC} — 파일 없음/손상 시 None.
    """
    import gzip as _gzip
    try:
        with _gzip.open(path, "rt", encoding="utf-8") as f:
            header = json.load(f).get("header", {})
        n = header.get("counts", {}).get("eligible_primary")
        mde = header.get("mde", {}).get("ic")
        if n is None:
            return None
        return {"n": int(n), "mde_ic": float(mde) if mde is not None else None}
    except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _h2_snapshot_status(snap_dir: Path) -> dict | None:
    """h2_snapshots/ 최신 스냅샷 파일의 날짜·행수를 읽는다.

    수집 중 프로세스가 죽으면 마지막 gzip 멤버가 잘릴 수 있으므로
    절단 지점까지 읽힌 유효 행수만 센다 (크래시 금지).

    Args:
        snap_dir: 스냅샷 디렉토리 (<YYYY-MM-DD>.jsonl.gz, 지갑당 1줄).

    Returns:
        {"day": 최신 파일 날짜, "rows": 행수} — 디렉토리/파일 없으면 None.
    """
    import gzip as _gzip
    import zlib as _zlib
    try:
        files = sorted(snap_dir.glob("*.jsonl.gz"))
    except OSError:
        return None
    if not files:
        return None
    latest = files[-1]
    rows = 0
    try:
        with _gzip.open(latest, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows += 1
    except (OSError, EOFError, _zlib.error):
        pass    # 잘린 gzip — 그 지점까지의 유효 행수만 사용
    return {"day": latest.name.replace(".jsonl.gz", ""), "rows": rows}


def _h2_fills_status(path: Path) -> dict | None:
    """h2_fills_state.json 에서 절단·초기절단 지갑 수를 센다.

    Args:
        path: 체결 이력 수집 상태 파일 ({"wallets": {addr: {...}}}).

    Returns:
        {"tracked": 추적 지갑 수, "censored": fill-history-censored 수,
         "truncated": initial_window_truncated 수} — 파일 없음/손상 시 None.
    """
    try:
        wallets = json.loads(path.read_text(encoding="utf-8")).get("wallets", {})
        censored = sum(1 for w in wallets.values() if isinstance(w, dict)
                       and w.get("status") == "fill-history-censored")
        truncated = sum(1 for w in wallets.values() if isinstance(w, dict)
                        and w.get("initial_window_truncated"))
        return {"tracked": len(wallets), "censored": censored,
                "truncated": truncated}
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _h2_gate_status(path: Path) -> dict | None:
    """h2_trackb_gate.json 의 판정 기록·stage2_eligible 을 요약한다.

    Args:
        path: 트랙 B 게이트 상태 파일 (lab/h2_trackb.py gate 가 기록).

    Returns:
        {"n_entries": 기록 건수, "stage2_eligible": bool,
         "verdict": 최근 판정 한 줄 (IC·p·통과 여부, 없으면 None)}
        — 파일 없음/손상 시 None (카드에서 "판정 전" 표시).
    """
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
        entries = gate.get("entries") or []
        verdict = None
        if entries:
            e = entries[-1]
            head = f"{e.get('judgment_date', '-')} T+{e.get('horizon_days', '?')}"
            if e.get("indeterminate") or e.get("ic") is None or e.get("p") is None:
                verdict = f"{head} · 판정불가"
            else:
                verdict = (f"{head} · IC {float(e['ic']):+.3f}"
                           f" · p {float(e['p']):.4f}"
                           f" · {'통과' if e.get('passed') else '미통과'}")
        return {"n_entries": len(entries),
                "stage2_eligible": bool(gate.get("stage2_eligible", False)),
                "verdict": verdict}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _load_h2_study(logs_dir: Path | None = None, today: date | None = None) -> dict:
    """H2(꾸준함 가설) 연구 상태 카드 데이터를 조립한다.

    손익 트랙이 아닌 사전등록 연구라 실시간 수익률이 없다 — 코호트 규모·
    수집 상태·다음 판정 카운트다운·게이트 기록만 요약한다. 입력 파일이
    하나도 없어도 크래시 없이 해당 항목만 None("대기" 표시)인 구조를
    반환한다. /api/live 대상이 아니며 페이지 로드 시 1회 계산된다.

    Args:
        logs_dir: 이력 디렉토리 (기본 ROOT/logs, 테스트 주입용).
        today: 카운트다운 기준일 (기본 오늘 UTC, 테스트 주입용).

    Returns:
        cohort/snapshot/fills/gate 각 요약(dict | None) + upcoming
        (미래 최근접 판정 1~2개, {"day", "label", "dday"}).
    """
    logs = logs_dir or (ROOT / "logs")
    t = today or datetime.now(timezone.utc).date()
    upcoming: list[dict] = []
    for day_s, label in H2_SCHEDULE:
        try:
            d = datetime.strptime(day_s, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d >= t:
            upcoming.append({"day": day_s, "label": label, "dday": (d - t).days})
        if len(upcoming) == 2:
            break
    return {
        "cohort": _h2_cohort_meta(logs / "h2_cohort.json.gz"),
        "snapshot": _h2_snapshot_status(logs / "h2_snapshots"),
        "fills": _h2_fills_status(logs / "h2_fills_state.json"),
        "gate": _h2_gate_status(logs / "h2_trackb_gate.json"),
        "upcoming": upcoming,
    }


# ------------------------------------------------------------------
# 전 코인 스캐너 (carrybot/live/market_scanner.py 산출물 — 표시 전용)
# ------------------------------------------------------------------

MARKET_SCAN_STALE_H = 2.0       # 시간당 갱신 데이터 — 이 이상이면 "n시간 전" 회색

# 코인 행 정규화 스키마 — 템플릿이 참조하는 전 키를 항상 채워 UndefinedError 차단
_MS_NUM_FIELDS = ("price", "chg24h_pct", "turnover24h", "dist24h_pct",
                  "dist96h_pct", "rsi14", "rsi2", "bb_pctb", "sma200_pct",
                  "vol_surge", "close")


def _ms_coin_row(c: dict) -> dict | None:
    """스캐너 코인 행 1개를 표시 스키마로 정규화한다 (기형 필드 = None).

    coin/symbol 둘 다 없으면 행 자체를 버린다 (None 반환).
    """
    coin = c.get("coin") or (str(c.get("symbol") or "").split("/")[0] or None)
    if not coin:
        return None
    row: dict = {"coin": str(coin), "symbol": str(c.get("symbol") or "")}
    for k in _MS_NUM_FIELDS:
        v = c.get(k)
        try:
            # float 승격 뒤 검사 — 거대 int 는 math.isfinite(int) 단계에서
            # OverflowError 를 던진다 (총체적 무크래시 계약)
            fv = float(v) if not isinstance(v, bool) else float("nan")
            ok = math.isfinite(fv)
        except (TypeError, ValueError, OverflowError):
            ok = False
        row[k] = fv if ok else None
    row["gate_long"] = c.get("gate_long") is True
    row["gate_short"] = c.get("gate_short") is True
    return row


def _load_market_scan(logs_dir: Path | None = None,
                      now: datetime | None = None) -> dict:
    """전 코인 스캐너 카드 데이터를 읽는다 (표시 전용 — 게이트/주문 입력 금지).

    logs/market_scan.json (스캐너가 원자 저장) 을 그대로 표시한다. 코인 순서는
    파일 순서(24h 거래대금 내림차순 고정)를 보존한다 — 성과·근접도 재정렬 금지
    (표시 규율). 파일 부재/손상/빈 목록은 available=False ("스캔 대기" 표시,
    크래시 금지). 시간당 갱신 데이터라 /api/live 대상이 아니며 페이지 로드 시
    1회 계산된다.

    Args:
        logs_dir: 산출물 디렉토리 (기본 ROOT/logs, 테스트 주입용).
        now: 신선도 기준 시각 (기본 현재 UTC, 테스트 주입용).

    Returns:
        {"available", "generated_at", "age_label", "stale", "coins",
         "skipped"} (JSON 직렬화 가능).
    """
    logs = logs_dir or (ROOT / "logs")
    empty = {"available": False, "generated_at": None, "age_label": None,
             "stale": False, "coins": [], "skipped": 0}
    try:
        d = json.loads((logs / "market_scan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(d, dict):
        return empty
    raw_coins = d.get("coins")
    if not isinstance(raw_coins, list):
        return empty
    coins = [r for r in (_ms_coin_row(c) for c in raw_coins
                         if isinstance(c, dict)) if r is not None]
    if not coins:
        return empty

    gen_disp: str | None = None
    age_label: str | None = None
    stale = False
    raw = d.get("generated_at_utc")
    if isinstance(raw, str):
        try:
            gen = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if gen.tzinfo is None:
                gen = gen.replace(tzinfo=timezone.utc)
            gen_disp = gen.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            ref = now or datetime.now(timezone.utc)
            age_h = (ref - gen).total_seconds() / 3600.0
            if age_h >= MARKET_SCAN_STALE_H:
                stale = True
                age_label = f"{int(age_h)}시간 전"
        except (ValueError, OverflowError, OSError):
            gen_disp = raw[:19].replace("T", " ")

    try:
        skipped = int(d.get("skipped") or 0)
    except (TypeError, ValueError, OverflowError):
        skipped = 0
    return {"available": True, "generated_at": gen_disp,
            "age_label": age_label, "stale": stale,
            "coins": coins, "skipped": skipped}


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
    Track C 카드만 두 수치를 병기한다: 큰 값은 명목(양다리 차익) 수익률,
    roe 줄은 ROE(=명목÷TRACKC_ROE_DIVISOR). 각 줄의 라벨이 어떤 수치인지
    정확히 말한다 (unit/roe_label). caveat 는 정정 공시 한 줄이다.
    """
    a, b = tracks.get("a", {}), tracks.get("b", {})
    c, dd_ = tracks.get("c", {}), tracks.get("d", {})
    def _pct(tr):
        return (tr.get("pct") or [0.0])[-1] if tr.get("pct") else 0.0
    a_pct, b_pct, c_pct, d_pct = _pct(a), _pct(b), _pct(c), _pct(dd_)
    c_roe = c_pct / TRACKC_ROE_DIVISOR
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
                 value=f"{c_pct:+.3f}%", unit="명목",
                 roe=f"{c_roe:+.3f}%", roe_label=TRACKC_ROE_LABEL,
                 badge=("수취 중" if c.get("labels") else "준비"),
                 active=bool(c.get("labels")),
                 sub=f"교차거래소 펀딩 · 큰 값 = {TRACKC_NOTIONAL_LABEL}",
                 caveat=TRACKC_CAVEAT),
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
    h2_study = _load_h2_study()
    tracke = _load_tracke()
    tracke_variant = _load_tracke_variant()
    # 셀 펼침 상세 초기값 — 표시 전용 스냅샷 (읽기 전용, 이후 pollLive 갱신).
    # 가격은 1초 캐시(_live_price)라 /api/live 폴링과 비용을 공유한다.
    tracke_detail = (_tracke_live() or {}).get("cells_detail", {})
    market_scan = _load_market_scan()
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
        h2=h2_study,
        tracke=tracke,
        tracke_json=json.dumps(tracke),
        tracke_variant=tracke_variant,
        tracke_detail=tracke_detail,
        tracke_detail_json=json.dumps(tracke_detail),
        tracke_cell_capital=TRACKE_CELL_CAPITAL,
        market_scan=market_scan,
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
        "h2_study": _load_h2_study(),
        "market_scan": _load_market_scan(),   # 시간당 갱신 — /api/live 비대상
        "tracke": _load_tracke(),     # 표시 전용 — 게이트/승급 입력 아님
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



def _tracks_live(logs_dir: Path | None = None) -> dict:
    """트랙별 실시간 시가평가 — 초단위 폴링용.

    B/D는 보유 포지션을 실시간 가격으로 평가하고, A/C는 상태 자본(펀딩형이라
    가격 틱 없음)을 그대로 반환한다. 표시값 = (시가평가자본 − 1) × 100 (%).

    Track C 의 pct 는 명목(양다리 차익) 기준이라 담보 이중 소요를 반영한
    roe_pct(=pct÷TRACKC_ROE_DIVISOR)와 basis 라벨을 함께 싣는다 — 프런트가
    두 수치를 각각 정확한 라벨로 갱신하게 한다 (÷2 표기만으로는 오독).
    """
    logs = logs_dir or (ROOT / "logs")
    out: dict = {}
    spec = {
        "a": ("tracka_state.json", None),
        "b": ("trackb_state.json", ("direction", "units", "entry")),
        "c": ("trackc_state.json", None),
        "d": ("trackd_state.json", ("d", "u", "e")),
    }
    for key, (fname, poskeys) in spec.items():
        try:
            st = json.loads((logs / fname).read_text())
        except (OSError, ValueError):
            continue
        mtm = float(st.get("equity", 1.0))
        details = []
        if poskeys:
            kd, ku, ke = poskeys
            for sym, pp in st.get("positions", {}).items():
                try:
                    d_, u_, e_ = int(pp[kd]), float(pp[ku]), float(pp[ke])
                except (KeyError, TypeError, ValueError):
                    continue
                px = _live_price(f"{sym}/USDT:USDT")
                if px is None:
                    px = e_
                upnl = u_ * (px - e_) * d_
                mtm += upnl
                details.append(dict(sym=sym, direction=("long" if d_ > 0 else "short"),
                                    entry=e_, price=round(px, 6),
                                    upnl_pct=round(upnl * 100, 4)))
        pct = round((mtm - 1.0) * 100, 4)
        out[key] = dict(pct=pct, positions=details)
        if key == "c":                       # 명목 기준 → ROE 병기
            out[key].update(
                basis=TRACKC_NOTIONAL_LABEL,
                roe_pct=round(pct / TRACKC_ROE_DIVISOR, 4),
                roe_label=TRACKC_ROE_LABEL,
            )
    return out


# 셀 상세 포지션 레벨 키 — 실키는 엔진 동결 계약(carrybot.aggressive.scalp_farm
# 의 FarmPos: stop·tgt, 0.0 = "레벨 없음" 센티널)을 import 로 확인했다 (수정 금지).
# 추매가는 v5(E24·E25) 상태에 실릴 수 있는 후보 키를 관용 수용한다 (부재 = None).
_TRACKE_POS_LEVEL_KEYS: dict[str, tuple[str, ...]] = {
    "stop": ("stop", "pstop", "sl"),
    "tgt": ("tgt", "ptgt", "tp", "target"),
    "add": ("next_add", "add_px", "padd", "add"),
}


def _tracke_pos_level(pp: dict, names: tuple[str, ...]) -> float | None:
    """포지션 상태 dict 에서 레벨 후보 키 중 첫 유효 양수 가격을 읽는다.

    엔진 계약상 0.0 은 "레벨 없음" 센티널이므로 0 이하·비수치·부재는 전부
    None 을 반환한다 (표시부는 "—", 크래시 금지).

    Args:
        pp: 포지션 상태 dict ({"d", "u", "e", "stop", "tgt", ...}).
        names: 우선순위 순 후보 키 목록.

    Returns:
        첫 유효 레벨 가격 — 없으면 None.
    """
    for n in names:
        v = pp.get(n)
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and v == v and v > 0):
            return float(v)
    return None


def _tracke_level_dist(level: float | None, px: float) -> float | None:
    """현재가 대비 레벨까지의 부호 있는 거리 % — 계산 불가 시 None.

    Args:
        level: 레벨 가격 (없으면 None).
        px: 현재가 (폴백 포함).

    Returns:
        (level/px − 1)×100 (%) — level 부재·px≤0 이면 None.
    """
    if level is None or px <= 0:
        return None
    return round((level / px - 1.0) * 100.0, 2)


def _tracke_level_bar(entry: float, mark: float, stop: float | None,
                      tgt: float | None, add: float | None) -> dict | None:
    """미니 레벨 바 마커 좌표(0~100%)를 계산한다 (표시 전용 — 정렬·강조 없음).

    가격축을 [최소 레벨, 최대 레벨] 구간으로 선형 매핑해 각 마커를 4~96%
    사이에 놓는다 (끝 마커 클리핑 방지). 숏이면 손절이 현재가 오른쪽에
    오는 것이 자연스럽고 정직한 배치다. 손절·목표·추매가 하나도 없거나
    구간이 퇴화(전 가격 동일)하면 None ("—" 표시, 크래시 금지).

    Args:
        entry: 진입가.
        mark: 현재가 (조회 실패 시 폴백 마크).
        stop: 손절가 or None.
        tgt: 목표가 or None.
        add: 다음 추매가 or None.

    Returns:
        {"stop", "entry", "mark", "tgt", "add"} 좌표(%) dict — 불가 시 None.
    """
    if stop is None and tgt is None and add is None:
        return None
    pts = [p for p in (stop, entry, mark, tgt, add) if p is not None]
    lo, hi = min(pts), max(pts)
    if hi <= lo:
        return None

    def _x(v: float | None) -> float | None:
        return (None if v is None
                else round(4.0 + 92.0 * (v - lo) / (hi - lo), 2))

    return dict(stop=_x(stop), entry=_x(entry), mark=_x(mark),
                tgt=_x(tgt), add=_x(add))


def _tracke_mtm_cell(cs: dict, ind: dict,
                     px_cache: dict[str, tuple[float, bool]],
                     fallback: list[str]) -> dict | None:
    """Track E 셀 하나를 현재가로 시가평가한다 (_tracke_live 공용 헬퍼).

    셀 시가평가 = equity + Σ u×(px−e)×d. 현재가는 1초 캐시(_live_price)를
    재사용하고, 실패 시 상태의 마지막 유효 종가(ind[sym].pc) → 진입가
    (평가손익 0) 순으로 폴백한다. 읽기 전용 — 어디에도 쓰지 않는다.

    Args:
        cs: 셀 상태 dict (equity, positions{sym:{e,u,d}}).
        ind: 폴백 종가 소스 ({sym: {"pc": ...}}).
        px_cache: 심볼별 (가격, 폴백 여부) 캐시 (한 평가 패스 내 공유).
        fallback: 현재가 폴백 심볼 목록 (in-place 추가).

    Returns:
        {"mtm": 시가평가 자본, "n_pos": 평가 포지션 수,
         "detail": {equity(실현 기준 현금)/unrealized(미실현 합 $)/
                    cost(누적 수수료)/fund(누적 펀딩)/positions(개별 상세)}}
        — equity 손상 시 None (셀 스킵). detail 은 행 펼침 상세 표시 전용이며
        positions 항목은 {sym, dir(롱/숏), entry, mark(현재가 or 폴백), qty,
        upnl(개별 미실현 $), upnl_pct(진입가 대비 방향 반영 %),
        stop/tgt/add(손절·목표·다음 추매가 — 부재·0.0 센티널은 None),
        stop_pct/tgt_pct/add_pct(현재가 대비 거리 %),
        bar(미니 레벨 바 마커 좌표 — _tracke_level_bar, 불가 시 None)} 이다.
    """
    try:
        equity = float(cs.get("equity", TRACKE_CELL_CAPITAL))
    except (TypeError, ValueError):
        return None
    unrealized = 0.0
    pos_detail: list[dict] = []
    positions = cs.get("positions")
    if isinstance(positions, dict):
        for sym, pp in positions.items():
            try:
                d_, u_, e_ = int(pp["d"]), float(pp["u"]), float(pp["e"])
            except (KeyError, TypeError, ValueError):
                continue
            if sym in px_cache:
                px, fb = px_cache[sym]
            else:
                live = _live_price(f"{sym}/USDT:USDT")
                fb = live is None
                if fb:
                    pc = ind.get(sym)
                    pc = pc.get("pc") if isinstance(pc, dict) else None
                    px = (float(pc) if isinstance(pc, (int, float))
                          and not isinstance(pc, bool) else e_)
                else:
                    px = live
                px_cache[sym] = (px, fb)
            if fb and sym not in fallback:
                fallback.append(sym)
            upnl = u_ * (px - e_) * d_
            unrealized += upnl
            stop_lv = _tracke_pos_level(pp, _TRACKE_POS_LEVEL_KEYS["stop"])
            tgt_lv = _tracke_pos_level(pp, _TRACKE_POS_LEVEL_KEYS["tgt"])
            add_lv = _tracke_pos_level(pp, _TRACKE_POS_LEVEL_KEYS["add"])
            # BBADD(v5): FarmPos 스키마 재사용으로 추매 트리거가가 tgt 키에 실린다
            # (엔진 확정 #a) — 목표가가 아니라 추매가로 재매핑 (감사 MAJOR 2 교정)
            if pp.get("kind") == "BBADD":
                add_lv, tgt_lv = tgt_lv, None
            pos_detail.append(dict(
                sym=sym, dir=("롱" if d_ > 0 else "숏"),
                entry=e_, mark=round(px, 6), qty=u_,
                upnl=round(upnl, 4),
                upnl_pct=(round((px / e_ - 1.0) * 100.0
                                * (1 if d_ > 0 else -1), 4)
                          if e_ > 0 else 0.0),
                stop=stop_lv, stop_pct=_tracke_level_dist(stop_lv, px),
                tgt=tgt_lv, tgt_pct=_tracke_level_dist(tgt_lv, px),
                add=add_lv, add_pct=_tracke_level_dist(add_lv, px),
                bar=_tracke_level_bar(e_, px, stop_lv, tgt_lv, add_lv),
            ))
    return dict(
        mtm=equity + unrealized,
        n_pos=len(pos_detail),
        detail=dict(
            equity=round(equity, 4),
            unrealized=round(unrealized, 4),
            cost=_tracke_metric(cs, ("cost_cum", "costs", "cost", "fees")),
            fund=_tracke_metric(cs, ("funding_cum", "funding", "fund")),
            positions=pos_detail,
        ),
    )


def _tracke_live(logs_dir: Path | None = None) -> dict | None:
    """Track E 단타 팜 실시간 시가평가 — 초단위 폴링용 (표시 전용).

    logs/tracke_state.json 의 cells{...}.positions{sym:{e,u,d}} 를 현재가로
    평가한다. 엔진 계약상 셀 equity 는 실현 기준 현금 자본이므로
    셀 시가평가 = equity + Σ u×(px−e)×d (트랙 B/D 시가평가와 동일 관례).
    현재가는 기존 1초 캐시(_live_price)를 그대로 재사용하며, 심볼 매핑은
    엔진(scalp_farm_runner)과 동일한 Bybit linear "SYM/USDT:USDT" 다 —
    바스켓 B(HYPE·BTR 포함)도 Bybit linear 티커에서 선정되므로 동일 경로.
    조회 실패 심볼은 상태의 마지막 유효 종가(ind[sym].pc), 그것도 없으면
    진입가(평가손익 0)로 폴백하고 fallback 목록에 표기한다.

    읽기 전용 경로다: 판정·원장·상태 파일에 절대 쓰지 않으며, 결과는
    승급/실거래 게이트(_promote_status, promote_checker)에 입력하지 않는다.

    Args:
        logs_dir: 상태 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    변형 셀(_tracke_variant_spec 의 엔진 명세 — 폴백 E11·E12)은 같은
    관례로 별도 평가해 variant 블록에만 싣는다 — 본 팜 합계·cells·n_pos·
    fallback 에는 절대 섞지 않는다 (분리 소구역 · 공식 판정 대상 아님).
    상태는 병렬 블록(variant_cells·variant2_cells, ...)을 모두 읽되 종가
    폴백(ind)은 각 블록 자체 것만 쓴다. 상태에 아직 없는 변형 셀(별도 t0
    미가동)은 variant 블록에서 빠진다 (프런트는 페이지 로드의 "대기" 유지).

    Returns:
        {"farm_equity": 팜 합계(USD), "farm_pct": 팜 수익률(%),
         "cells": {셀 id: 수익률 %}, "n_pos": 평가 포지션 수,
         "fallback": [현재가 폴백 심볼],
         "variant": {가동 중인 변형 셀 id: %, ..., "n_pos": 합} | None,
         "cells_detail": {셀 id: _tracke_mtm_cell 의 detail}}
        — cells_detail 은 본 셀 + 변형 셀 전부를 담는 행 펼침 상세
        (표시 전용 — 팜 합계·max 태그 등 어떤 집계에도 입력하지 않는다).
        상태 없음/손상 시 None (프런트는 페이지 로드 값을 유지한다).
    """
    logs = logs_dir or (ROOT / "logs")
    try:
        st = json.loads((logs / "tracke_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(st, dict):
        return None
    cells_state = st.get("cells")
    if not isinstance(cells_state, dict):
        return None
    ind = st.get("ind") if isinstance(st.get("ind"), dict) else {}

    cells: dict[str, float] = {}
    cells_detail: dict[str, dict] = {}             # 셀별 펼침 상세 (표시 전용)
    farm_equity = 0.0
    base_total = 0.0
    n_pos = 0
    fallback: list[str] = []
    px_cache: dict[str, tuple[float, bool]] = {}   # sym -> (가격, 폴백 여부)
    for cid in TRACKE_CELL_IDS:                    # 고정 순서 (성과순 정렬 금지)
        cs = cells_state.get(cid)
        if not isinstance(cs, dict):
            continue
        res = _tracke_mtm_cell(cs, ind, px_cache, fallback)
        if res is None:
            continue
        mtm, cell_npos = res["mtm"], res["n_pos"]
        cells[cid] = round((mtm / TRACKE_CELL_CAPITAL - 1.0) * 100.0, 4)
        cells_detail[cid] = res["detail"]
        farm_equity += mtm
        base_total += TRACKE_CELL_CAPITAL
        n_pos += cell_npos
    if not cells:
        return None

    # 변형 셀 — 별도 패스, 본 집계와 완전 분리 (셀 목록은 엔진 명세,
    # 상태는 병렬 블록 전부 — 폴백 ind·가격 캐시는 블록별 분리)
    variant: dict | None = None
    v_blocks: list[tuple[dict, dict, dict, list]] = []
    for blk in _tracke_variant_blocks(st):
        b_cells = blk.get("cells")
        if isinstance(b_cells, dict):
            b_ind = blk.get("ind") if isinstance(blk.get("ind"), dict) else {}
            v_blocks.append((b_cells, b_ind, {}, []))   # (+px 캐시, fb 목록)
    v_out: dict[str, float] = {}
    v_npos = 0
    for cid, _, _ in _tracke_variant_spec()[0]:        # 고정 순서
        for b_cells, b_ind, b_px, b_fb in v_blocks:
            cs = b_cells.get(cid)
            if not isinstance(cs, dict):
                continue
            res = _tracke_mtm_cell(cs, b_ind, b_px, b_fb)
            if res is not None:
                mtm, cell_npos = res["mtm"], res["n_pos"]
                v_out[cid] = round((mtm / TRACKE_CELL_CAPITAL - 1.0) * 100.0, 4)
                cells_detail[cid] = res["detail"]      # 상세만 공유 (집계 분리)
                v_npos += cell_npos
            break                                      # 그룹 번호 앞 블록 우선
    if v_out:
        variant = {**v_out, "n_pos": v_npos}

    return {
        "farm_equity": round(farm_equity, 2),
        "farm_pct": round((farm_equity / base_total - 1.0) * 100.0, 4),
        "cells": cells,
        "n_pos": n_pos,
        "fallback": fallback,
        "variant": variant,
        "cells_detail": cells_detail,
    }


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
        "tracks_live": _tracks_live(),
        "tracke": _tracke_live(),     # 표시 전용 — 승급/게이트 입력 아님
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
# 코인 차트 뷰 — /api/chart/<coin> (관측 도구 · 표시 전용)
# ------------------------------------------------------------------
# 읽기 전용 경로다: 어떤 상태·원장·판정 파일에도 쓰지 않으며, 결과는
# 승급/실거래 게이트·주문 경로에 입력하지 않는다 (방화벽). 지표 산식은
# 엔진 교정 관례와 동일하다 — prev-close TR ATR(swing.py 버그 재발 금지),
# Wilder RSI 첫 diff 시드, BB ddof=0 모표준편차, shift(1) 채널.

CHART_TF = "1h"
CHART_CANDLE_LIMIT = 220        # 1h 조회 깊이 — SMA200 + 여유 (스캐너 동수)
CHART_BARS = 200                # 응답 봉 수 상한
CHART_CACHE_TTL = 300.0         # 코인별 서버 캐시 5분 (거래소 남용 방지)
CHART_ATR_N = 24                # 1h ATR — scalp_farm ATR1H_N 동수
CHART_STOP_ATR_MULT = 6.0       # BRK 스탑 배수 — scalp_farm BRK_ATR_MULT 동수
CHART_BB_N = 20                 # BB 길이 (중심 SMA20)
CHART_BB_K = 2.0                # BB 폭 (2σ, ddof=0)
CHART_RSI_N = 14                # Wilder RSI
CHART_SMA_N = 200               # 추세 SMA
CHART_VOL_N = 20                # 거래량 평균 창 (직전 20봉)
CHART_CH_FAST = 24              # 돌파 채널 (24h)
CHART_CH_SLOW = 96              # 돌파 채널 (96h)

# 컴포짓 점수 상시 라벨 — 응답에 항상 동봉, 프런트 상시 표시 (표시 규율)
COMPOSITE_NOTE = ("관측 도구 — 이 지표 조합의 예측력은 검증 결과 우연 수준"
                  "(SWEEP-2026-08-31). 레벨·상태 표시용이며 매매 신호가 아님.")

_chart_cache: dict[str, tuple[float, dict]] = {}   # coin -> (fetched_at, 시장 페이로드)


def _chart_fetch_1h(symbol: str) -> list[tuple] | None:
    """Bybit 1h 캔들을 확정봉만 (ts, o, h, l, c, v) 오름차순으로 수집한다.

    기존 대시보드 시세 관례(_get_market_client)를 재사용하고, 스캐너와 같은
    확정봉 규약을 적용한다 — 현재 1h 구간에 속한 미확정 봉은 제외.

    Args:
        symbol: 통합 심볼 (예: 'BTC/USDT:USDT').

    Returns:
        [(ts_ms, open, high, low, close, volume)] 시간 오름차순 —
        조회 실패/기형 응답이면 None (크래시 금지).
    """
    try:
        df = _get_market_client().fetch_ohlcv(symbol, CHART_TF, CHART_CANDLE_LIMIT)
    except Exception as e:  # noqa: BLE001 — 네트워크/거래소 계열 전반 (무크래시)
        logger.warning("차트 캔들 조회 실패 %s: %s", symbol, e)
        return None
    now_h = int(time.time() * 1000)
    now_h -= now_h % 3_600_000
    bars: list[tuple] = []
    try:
        for ts, row in df.iterrows():
            t = int(ts.timestamp() * 1000)
            if t >= now_h:
                continue                    # 미확정 봉 제외 (확정봉 규약)
            bars.append((t, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"]),
                         float(row["volume"]) if "volume" in row else float("nan")))
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    bars.sort(key=lambda b: b[0])
    return bars or None


def _chart_series(bars: list[tuple]) -> dict:
    """확정봉 시퀀스에서 차트 표시 시계열을 계산한다 (엔진 동일식).

    산식 관례 (scalp_farm 교정판 · market_scanner 동형):
    - ATR(24): TR = max(h−l, |h−pc|, |l−pc|) — **previous close** 기준,
      Wilder 평활 a += (tr−a)/24 (swing.py 버그 재발 금지 관례).
    - RSI14: Wilder 평활, 첫 diff 시드. dn==0 은 published_systems 원전 —
      상승만 100, 무변동 50.
    - BB(20, 2σ): 중심 SMA20 · ddof=0 모표준편차.
    - 채널 24/96: 봉 [i] 를 **제외한** 직전 N봉 고가 최대·저가 최소 (shift 1).
    - 거래량 평균: mean(vol[i−20..i−1]) — 직전 20봉 (shift 1).
    미형성·NaN 구간은 None (JSON 직렬화 안전).

    Args:
        bars: [(ts, o, h, l, c, v), ...] 시간 오름차순, 확정봉만.

    Returns:
        {"ts","open","high","low","close","volume"(봉),
         "bb_up","bb_low","sma20","sma200","rsi14",
         "ch24_up","ch24_dn","ch96_up","ch96_dn","vol_avg20"(시계열),
         "atr24"(마지막 확정봉까지의 ATR 스칼라)}.
    """
    def _v(x: float) -> float | None:
        return round(x, 8) if isinstance(x, float) and math.isfinite(x) else None

    n = len(bars)
    ts = [int(b[0]) for b in bars]
    opens = [float(b[1]) for b in bars]
    highs = [float(b[2]) for b in bars]
    lows = [float(b[3]) for b in bars]
    closes = [float(b[4]) for b in bars]
    vols = [float(b[5]) for b in bars]

    bb_up: list[float | None] = [None] * n
    bb_low: list[float | None] = [None] * n
    sma20: list[float | None] = [None] * n
    sma200: list[float | None] = [None] * n
    rsi14: list[float | None] = [None] * n
    ch24_up: list[float | None] = [None] * n
    ch24_dn: list[float | None] = [None] * n
    ch96_up: list[float | None] = [None] * n
    ch96_dn: list[float | None] = [None] * n
    vol_avg20: list[float | None] = [None] * n

    # RSI14 — Wilder, 첫 diff 시드 (market_scanner.wilder_rsi 동형 점화식)
    u: float | None = None
    dn: float | None = None
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        ux, dx = max(diff, 0.0), max(-diff, 0.0)
        u = ux if u is None else u + (ux - u) / CHART_RSI_N
        dn = dx if dn is None else dn + (dx - dn) / CHART_RSI_N
        if dn > 0:
            r = 100.0 - 100.0 / (1.0 + u / dn)
        else:
            r = 100.0 if u > 0 else 50.0
        rsi14[i] = _v(r)

    # ATR(24) — prev-close TR, Wilder 평활 (scalp_farm _update_1h 동형)
    atr: float | None = None
    pc: float | None = None
    for i in range(n):
        tr = (highs[i] - lows[i] if pc is None
              else max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
        atr = tr if atr is None else atr + (tr - atr) / CHART_ATR_N
        pc = closes[i]

    for i in range(n):
        if i >= CHART_BB_N - 1:
            w = closes[i - CHART_BB_N + 1: i + 1]
            m = sum(w) / CHART_BB_N
            sd = math.sqrt(sum((x - m) ** 2 for x in w) / CHART_BB_N)   # ddof=0
            sma20[i] = _v(m)
            bb_up[i] = _v(m + CHART_BB_K * sd)
            bb_low[i] = _v(m - CHART_BB_K * sd)
        if i >= CHART_SMA_N - 1:
            sma200[i] = _v(sum(closes[i - CHART_SMA_N + 1: i + 1]) / CHART_SMA_N)
        for cn, cu, cd in ((CHART_CH_FAST, ch24_up, ch24_dn),
                           (CHART_CH_SLOW, ch96_up, ch96_dn)):
            if i >= cn:                       # 봉 [i] 제외 직전 n봉 (shift 1)
                cu[i] = _v(max(highs[i - cn: i]))
                cd[i] = _v(min(lows[i - cn: i]))
        if i >= CHART_VOL_N:
            vm = sum(vols[i - CHART_VOL_N: i]) / CHART_VOL_N
            vol_avg20[i] = _v(vm)

    return {
        "ts": ts, "open": [_v(x) for x in opens], "high": [_v(x) for x in highs],
        "low": [_v(x) for x in lows], "close": [_v(x) for x in closes],
        "volume": [_v(x) for x in vols],
        "bb_up": bb_up, "bb_low": bb_low, "sma20": sma20, "sma200": sma200,
        "rsi14": rsi14, "ch24_up": ch24_up, "ch24_dn": ch24_dn,
        "ch96_up": ch96_up, "ch96_dn": ch96_dn, "vol_avg20": vol_avg20,
        "atr24": _v(atr) if atr is not None else None,
    }


def _composite_score(close: float | None, sma20: float | None,
                     sma200: float | None, rsi14: float | None,
                     pctb: float | None, vol_surge: float | None,
                     dist24_pct: float | None) -> dict:
    """컴포짓 관측 점수 −100~+100 — **산식 동결 2026-09-01, 변경 금지**.

    5요소 각 −20~+20 클립, 합산 후 [−100, +100] 클립·정수 라운드.
    + 는 매수측 관측, − 는 매도측 관측. 미형성(None) 요소는 0 (중립).

    동결 산식:
    1. 추세 trend = clip( clip((close/sma200 − 1)×100, −10, +10) × 1.5
       + 정배열 가중, −20, +20 ) — 정배열 가중: sma20 > sma200 이면 +5,
       sma20 < sma200 이면 −5, 그 외 0.
    2. 모멘텀 momentum = clip((rsi14 − 50) × 0.4, −20, +20)
       — RSI 0 → −20, 50 → 0, 100 → +20.
    3. 밴드 위치 band = clip((0.5 − %b) × 40, −20, +20)
       — 낮을수록 매수측 + (%b 0 하단 → +20, 1 상단 → −20).
    4. 거래량 서지 volume = clip(log2(vol_surge) × 10, −20, +20)
       — 평균 1× → 0, 2× → +10, 4× → +20, 0.5× → −10. surge ≤ 0 은 중립 0.
    5. 돌파 근접 breakout: d = 24h 채널 상단까지 거리 %.
       d ≤ 0 (이미 돌파) → +20, d > 0 → 20/(1+d) — 역수·캡 +20,
       하한 0 (원거리 = 중립 수렴, 음수 없음).

    지위: 관측 도구 — 이 지표 조합의 예측력은 검증 결과 우연 수준
    (SWEEP-2026-08-31, logs/sweep_returns.npz). 레벨·상태 표시용이며
    매매 신호가 아니다. 승급/실거래 게이트에 입력하지 않는다.

    Args:
        close: 최신 확정봉 종가.
        sma20: SMA20 (BB 중심).
        sma200: SMA200.
        rsi14: Wilder RSI14.
        pctb: BB %b.
        vol_surge: vol[i] / mean(vol[i−20..i−1]).
        dist24_pct: 24h 채널 상단까지 거리 % (음수 = 이미 돌파).

    Returns:
        {"score": int, "parts": {"trend","momentum","band","volume","breakout"}
         (각 float ±20, 라운드 2), "note": COMPOSITE_NOTE}.
    """
    def _clip(x: float, lo: float = -20.0, hi: float = 20.0) -> float:
        return max(lo, min(hi, x))

    def _num(x) -> bool:
        return (isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x))

    trend = 0.0
    if _num(close) and _num(sma200) and sma200 > 0:
        base = _clip((close / sma200 - 1.0) * 100.0, -10.0, 10.0) * 1.5
        align = 0.0
        if _num(sma20):
            align = 5.0 if sma20 > sma200 else (-5.0 if sma20 < sma200 else 0.0)
        trend = _clip(base + align)

    momentum = _clip((rsi14 - 50.0) * 0.4) if _num(rsi14) else 0.0
    band = _clip((0.5 - pctb) * 40.0) if _num(pctb) else 0.0
    volume = (_clip(math.log2(vol_surge) * 10.0)
              if _num(vol_surge) and vol_surge > 0 else 0.0)
    breakout = 0.0
    if _num(dist24_pct):
        breakout = 20.0 if dist24_pct <= 0 else _clip(20.0 / (1.0 + dist24_pct))

    parts = {"trend": trend, "momentum": momentum, "band": band,
             "volume": volume, "breakout": breakout}
    score = int(round(_clip(sum(parts.values()), -100.0, 100.0)))
    return {"score": score,
            "parts": {k: round(v, 2) for k, v in parts.items()},
            "note": COMPOSITE_NOTE}


def _chart_pos_levels(coin: str, logs_dir: Path | None = None) -> list[dict]:
    """tracke_state.json **전 그룹**에서 해당 코인 오픈 포지션 레벨을 모은다.

    본 셀(cells)과 병렬 변형 블록(variant_cells·variant2_cells, ... —
    _tracke_variant_blocks) 전부를 스캔한다 (표시 전용 — 게이트 입력 금지,
    읽기 전용). BBADD 는 FarmPos 스키마 재사용으로 추매 트리거가가 tgt 키에
    실리므로(엔진 확정 #a) 추매가로 재매핑한다 — _tracke_mtm_cell 동일 교정.
    동일 (방향, kind, 진입, 손절, 목표, 추매) 레벨은 셀 목록으로 병합한다
    (같은 전략 셀들의 중복 수평선 방지 — 셀 순서는 고정 순회 순서 보존).

    Args:
        coin: 코인 심볼 (예: 'BTC' — 대문자 정규화 후 비교).
        logs_dir: 상태 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        [{"cells": [셀 id, ...], "kind", "dir"(롱/숏), "entry",
          "stop", "tgt", "add"}] — 상태 부재/손상/보유 없음이면 빈 목록
        (크래시 금지). 0.0 센티널·부재 레벨은 None.
    """
    logs = logs_dir or (ROOT / "logs")
    try:
        st = json.loads((logs / "tracke_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(st, dict):
        return []
    coin = (coin or "").upper().strip()

    groups: list[tuple[str, dict]] = []
    main = st.get("cells")
    if isinstance(main, dict):
        for cid in TRACKE_CELL_IDS:                      # 고정 순서
            cs = main.get(cid)
            if isinstance(cs, dict):
                groups.append((cid, cs))
    for blk in _tracke_variant_blocks(st):               # 그룹 번호 순
        b_cells = blk.get("cells")
        if not isinstance(b_cells, dict):
            continue
        for cid in sorted(b_cells):                      # 셀 id 순 (고정)
            cs = b_cells.get(cid)
            if isinstance(cs, dict):
                groups.append((str(cid).upper(), cs))

    merged: dict[tuple, dict] = {}
    for cid, cs in groups:
        positions = cs.get("positions")
        if not isinstance(positions, dict):
            continue
        for sym, pp in positions.items():
            if not isinstance(pp, dict) or str(sym).upper().strip() != coin:
                continue
            try:
                d_, e_ = int(pp["d"]), float(pp["e"])
            except (KeyError, TypeError, ValueError):
                continue
            stop_lv = _tracke_pos_level(pp, _TRACKE_POS_LEVEL_KEYS["stop"])
            tgt_lv = _tracke_pos_level(pp, _TRACKE_POS_LEVEL_KEYS["tgt"])
            add_lv = _tracke_pos_level(pp, _TRACKE_POS_LEVEL_KEYS["add"])
            kind = str(pp.get("kind") or "")
            if kind == "BBADD":              # tgt 키 = 추매 트리거가 (재매핑)
                add_lv, tgt_lv = tgt_lv, None
            key = (d_, kind, e_, stop_lv, tgt_lv, add_lv)
            g = merged.get(key)
            if g is None:
                merged[key] = dict(
                    cells=[cid], kind=kind, dir=("롱" if d_ > 0 else "숏"),
                    entry=e_, stop=stop_lv, tgt=tgt_lv, add=add_lv)
            elif cid not in g["cells"]:
                g["cells"].append(cid)
    return list(merged.values())


def _chart_levels(coin: str, market: dict, logs_dir: Path | None = None) -> dict:
    """차트 레벨 블록 — 팜 오픈 포지션 우선, 없으면 전략 기준 레벨.

    전략 기준 레벨 (보유 없음 시 표기용 — 주문 아님):
    - brk_entry: 24h 채널 상단 (돌파 진입 레벨, shift 1)
    - mr_entry: BB 하단 (평균회귀 진입 레벨)
    - stop_dist: 돌파 진입 가정 시 6×ATR(24) 스탑 거리
      (scalp_farm BRK_ATR_MULT × ATR1H_N 동수 — 표기용 값)

    Args:
        coin: 코인 심볼.
        market: _chart_market 시장 페이로드 (series·atr24 참조).
        logs_dir: 상태 디렉토리 (기본 ROOT/logs, 테스트 주입용).

    Returns:
        {"mode": "positions", "positions": [...]} 또는
        {"mode": "strategy", "brk_entry", "mr_entry", "atr24",
         "stop_mult", "stop_dist"} (계산 불가 항목은 None).
    """
    positions = _chart_pos_levels(coin, logs_dir)
    if positions:
        return {"mode": "positions", "positions": positions}
    series = market.get("series") or {}

    def _last(key: str) -> float | None:
        s = series.get(key) or []
        return s[-1] if s and s[-1] is not None else None

    atr = market.get("atr24")
    stop_dist = (round(CHART_STOP_ATR_MULT * atr, 8)
                 if isinstance(atr, (int, float)) and atr > 0 else None)
    return {"mode": "strategy", "brk_entry": _last("ch24_up"),
            "mr_entry": _last("bb_low"), "atr24": atr,
            "stop_mult": CHART_STOP_ATR_MULT, "stop_dist": stop_dist}


def _chart_market(coin: str) -> dict | None:
    """코인별 시장 페이로드 (캔들 + 시계열 + 컴포짓) — 5분 서버 캐시.

    캐시 유효분은 그대로 재사용한다 (거래소 남용 방지). 만료 후 재페치가
    실패하면 스테일 잔존값을 유지하고("무크래시 — 대기보다 낫다"), 잔존값도
    없으면 None (호출부는 "차트 데이터 대기" 응답).

    Args:
        coin: 정규화된 코인 심볼 (대문자).

    Returns:
        {"available", "coin", "symbol", "generated_at", "bars"(응답 상한
         CHART_BARS 로 절단된 시계열 dict), "series", "atr24", "composite",
         "label"} — 실패·잔존값 없음이면 None.
    """
    now = time.time()
    cached = _chart_cache.get(coin)
    if cached and now - cached[0] < CHART_CACHE_TTL:
        return cached[1]
    symbol = f"{coin}/USDT:USDT"
    bars = _chart_fetch_1h(symbol)
    if not bars or len(bars) < 2:
        return cached[1] if cached else None    # 스테일 잔존값 유지 (무크래시)
    s = _chart_series(bars)

    close = s["close"][-1]
    bb_u, bb_l, mid = s["bb_up"][-1], s["bb_low"][-1], s["sma20"][-1]
    pctb = None
    if (close is not None and bb_u is not None and bb_l is not None
            and bb_u > bb_l):
        pctb = (close - bb_l) / (bb_u - bb_l)
    v_avg = s["vol_avg20"][-1]
    v_last = s["volume"][-1]
    vol_surge = (v_last / v_avg
                 if v_last is not None and v_avg is not None and v_avg > 0
                 else None)
    ch24 = s["ch24_up"][-1]
    dist24 = ((ch24 / close - 1.0) * 100.0
              if ch24 is not None and close is not None and close > 0 else None)
    composite = _composite_score(close, mid, s["sma200"][-1], s["rsi14"][-1],
                                 pctb, vol_surge, dist24)

    cut = -CHART_BARS
    bars_keys = ("ts", "open", "high", "low", "close", "volume")
    series_keys = ("bb_up", "bb_low", "sma20", "sma200", "rsi14",
                   "ch24_up", "ch24_dn", "ch96_up", "ch96_dn", "vol_avg20")
    payload = {
        "available": True, "coin": coin, "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
        "bars": {k: s[k][cut:] for k in bars_keys},
        "series": {k: s[k][cut:] for k in series_keys},
        "atr24": s["atr24"],
        "composite": composite,
        "label": COMPOSITE_NOTE,
    }
    _chart_cache[coin] = (now, payload)
    return payload


@app.route("/api/chart/<coin>")
def api_chart(coin: str):
    """코인 차트 데이터 — 캔들 + 엔진 동일식 지표 + 레벨 + 컴포짓 점수.

    관측 도구 (표시 전용): 매매 신호가 아니며(COMPOSITE_NOTE 상시 동봉),
    승급/실거래 게이트·주문 경로에 입력하지 않는다. 캔들·지표·컴포짓은
    5분 서버 캐시(_chart_market), 레벨은 매 요청 상태 파일에서 재계산
    (읽기 전용 — 포지션 변화를 캐시보다 빨리 반영). 코인명 기형·페치 실패
    전부 available=False "차트 데이터 대기" 응답으로 흡수 (크래시 금지).
    """
    coin = (coin or "").strip().upper()
    waiting = {"available": False, "coin": coin, "note": "차트 데이터 대기",
               "label": COMPOSITE_NOTE}
    if not re.fullmatch(r"[A-Z0-9]{1,15}", coin):
        return jsonify(waiting)
    market = _chart_market(coin)
    if market is None:
        return jsonify(waiting)
    payload = dict(market)              # 캐시 엔트리 오염 방지 (얕은 복사)
    payload["levels"] = _chart_levels(coin, market)
    return jsonify(payload)


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
