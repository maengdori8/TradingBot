"""외부 스냅샷 상황판 HTML 생성기 (단타 팜).

logs/ 의 스냅샷 데이터를 읽어 self-contained HTML 상황판을 지정 경로에 쓴다.
셀 메타(전략·바스켓·라벨)는 carrybot.aggressive.scalp_farm 의 동결 상수
(CELLS/VCELLS/V2CELLS/... + LABELS 계열)를 동적으로 발견해 사용한다 —
엔진에 셀·그룹이 추가돼도 이 스크립트는 무수정으로 따라간다.

표시 규율 (기존 상황판 계승):
- PAPER ONLY 배너 (가상 자본 합계 명시), 계정 표는 고정 순서 (성과순 정렬 금지)
- "사후 최대값 · 선택 금지" 태그는 본 팜 E01~E10 에만
- 변형 셀은 전부 판정 비대상 (동결 라벨 문구 그대로 표기)
- 파일 부재·키 부재는 전부 "대기" 처리 (크래시 금지)

사용법:
    python tools/gen_status_page.py --out <path>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

# 판정 일정 고정표 (사전 지정 — 성과 조회와 무관, 변경 금지).
# 매월 (H1, 트랙 A + 단타 팜) 판정 쌍이 반복된다.
VERDICT_SCHEDULE: tuple[tuple[str, str], ...] = (
    ("2026-09-24", "H1"),
    ("2026-09-26", "트랙 A + 단타 팜"),
    ("2026-10-24", "H1"),
    ("2026-10-26", "트랙 A + 단타 팜"),
    ("2026-11-25", "H1"),
    ("2026-11-26", "트랙 A + 단타 팜"),
    ("2026-12-26", "트랙 A + 단타 팜"),
    ("2027-01-25", "트랙 A + 단타 팜"),
    ("2027-02-23", "H1"),
    ("2027-02-24", "트랙 A + 단타 팜 (최종)"),
)

# 최근 체결 표시 대상 action (funding 제외; exit 는 exit* 프리픽스 매칭)
FILL_ACTIONS: tuple[str, ...] = ("enter", "add", "timeout", "target")


# ---------------------------------------------------------------- 안전 로더

def _load_json(path: Path) -> dict | None:
    """JSON 파일 안전 로드 — 부재·손상 시 None ("대기" 처리용)."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError) as e:
        logger.warning("JSON 로드 실패 %s: %s", path, e)
        return None


def _load_json_gz(path: Path) -> dict | None:
    """gzip JSON 파일 안전 로드 — 부재·손상 시 None."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError) as e:
        logger.warning("gz JSON 로드 실패 %s: %s", path, e)
        return None


def _load_csv(path: Path) -> list[dict]:
    """CSV 파일 안전 로드 — 부재·손상 시 빈 리스트."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as e:
        logger.warning("CSV 로드 실패 %s: %s", path, e)
        return []


def _f(v: Any) -> float | None:
    """느슨한 float 변환 — 실패 시 None."""
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 엔진 메타

def _load_engine_meta() -> dict:
    """scalp_farm 동결 상수에서 셀 그룹 메타를 동적으로 발견한다.

    반환: {"groups": [(state_key, 그룹명, [CellSpec...], {cell: label}, t0_key)],
           "capital0": 셀당 가상 자본, "basket_a": 튜플, "vbasket_labels": dict}
    엔진 import 실패 시 groups 는 빈 리스트 (섹션 "대기" 폴백).
    """
    meta: dict = {"groups": [], "capital0": 10_000.0,
                  "basket_a": (), "vbasket_labels": {}}
    try:
        sys.path.insert(0, str(ROOT))
        from carrybot.aggressive import scalp_farm as sf
    except Exception as e:  # noqa: BLE001 — 표시 전용 폴백 (크래시 금지)
        logger.warning("scalp_farm import 실패 — 셀 메타 없이 진행: %s", e)
        return meta
    meta["capital0"] = float(getattr(sf, "CAPITAL0", 10_000.0))
    meta["basket_a"] = tuple(getattr(sf, "BASKET_A", ()))
    meta["vbasket_labels"] = dict(getattr(sf, "VBASKET_LABELS", {}))
    groups: list[tuple] = []
    if hasattr(sf, "CELLS"):
        groups.append(("cells", "본 팜", list(sf.CELLS),
                       dict(getattr(sf, "LABELS", {})), "t0"))
    if hasattr(sf, "VCELLS"):
        groups.append(("variant_cells", "변형", list(sf.VCELLS),
                       dict(getattr(sf, "VLABELS", {})), "t0_variant"))
    n = 2
    while hasattr(sf, f"V{n}CELLS") and n < 100:
        groups.append((f"variant{n}_cells", f"변형{n}",
                       list(getattr(sf, f"V{n}CELLS")),
                       dict(getattr(sf, f"V{n}LABELS", {})), f"t0_variant{n}"))
        n += 1
    meta["groups"] = groups
    return meta


# ---------------------------------------------------------------- 포맷 헬퍼

def esc(s: Any) -> str:
    """HTML 이스케이프 (None 은 '대기')."""
    return html.escape(str(s)) if s is not None else "대기"


def fmt_usd(v: float | None, digits: int = 2) -> str:
    """달러 포맷 — None 은 '-'."""
    return f"${v:,.{digits}f}" if v is not None else "-"


def fmt_price(v: float | None) -> str:
    """가격 포맷 — 크기에 따라 유효자리 조절."""
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.1f}"
    if a >= 1:
        return f"{v:,.2f}"
    return f"{v:.6g}"


def fmt_pct(v: float | None) -> str:
    """부호 있는 % 포맷."""
    return f"{v:+.2f}%" if v is not None else "-"


def pnl_span(v: float | None, text: str | None = None) -> str:
    """손익 색상(녹/적 분리) span."""
    if v is None:
        return '<span class="mono">-</span>'
    cls = "pos" if v > 0 else ("neg" if v < 0 else "flat")
    return f'<span class="mono {cls}">{esc(text if text is not None else f"{v:+,.2f}")}</span>'


def fmt_ts_ms(ms: Any) -> str:
    """epoch ms → 'YYYY-MM-DD HH:MM UTC' (실패 시 '대기')."""
    v = _f(ms)
    if v is None:
        return "대기"
    try:
        return datetime.fromtimestamp(v / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC")
    except (OSError, OverflowError, ValueError):
        return "대기"


def fmt_ts_short(ms: Any) -> str:
    """epoch ms → 'MM-DD HH:MM' (실패 시 '-')."""
    v = _f(ms)
    if v is None:
        return "-"
    try:
        return datetime.fromtimestamp(v / 1000, tz=timezone.utc).strftime(
            "%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return "-"


# ---------------------------------------------------------------- 데이터 조립

def _basket_label(code: str, sub: dict, meta: dict) -> str:
    """바스켓 코드 → 표기 문자열 (구성은 데이터 주도 — 하드코딩 없음)."""
    if code in meta["vbasket_labels"]:
        return str(meta["vbasket_labels"][code])
    if code == "A" and meta["basket_a"]:
        return "·".join(meta["basket_a"])
    if code == "B":
        bb = sub.get("basket_b")
        if isinstance(bb, list) and bb:
            return "·".join(str(x) for x in bb)
    return code


def build_groups(state: dict | None, meta: dict) -> list[dict]:
    """엔진 그룹 메타 × 상태 JSON → 렌더용 그룹 리스트."""
    out: list[dict] = []
    top_ind = (state or {}).get("ind") or {}
    for state_key, gname, specs, labels, t0_key in meta["groups"]:
        if state_key == "cells":
            sub = state or {}
            cells_state = sub.get("cells") or {}
            t0 = sub.get(t0_key)
        else:
            sub = (state or {}).get(state_key) or {}
            cells_state = sub.get("cells") or {}
            t0 = sub.get(t0_key)
        g_ind = sub.get("ind") or {}
        rows: list[dict] = []
        positions: list[dict] = []
        g_total = 0.0
        g_known = 0
        for spec in specs:                       # 고정 순서 — 성과순 정렬 금지
            cid = str(spec.cell)
            cs = cells_state.get(cid) or {}
            eq = _f(cs.get("equity"))
            pos = cs.get("positions") or {}
            if eq is not None:
                g_total += eq
                g_known += 1
            rows.append({
                "id": cid,
                "strategy": str(spec.strategy),
                "basket": str(spec.basket),
                "basket_label": _basket_label(str(spec.basket), sub, meta),
                "label": str(labels.get(cid, "")),
                "equity": eq,
                "pct": (eq / meta["capital0"] - 1.0) * 100.0
                       if eq is not None and meta["capital0"] else None,
                "n_pos": len(pos) if isinstance(pos, dict) else 0,
                "is_max": False,
                "halted": bool(cs.get("halted")),
            })
            if not isinstance(pos, dict):
                continue
            for sym in sorted(pos):
                p = pos[sym] or {}
                e = _f(p.get("e"))
                u = _f(p.get("u"))
                d = _f(p.get("d"))
                kind = str(p.get("kind", ""))
                ind_sym = g_ind.get(sym) or top_ind.get(sym) or {}
                cur = _f(ind_sym.get("pc"))      # 현재가 = ind.pc 폴백
                upnl = upct = None
                if None not in (e, u, d, cur) and e:
                    upnl = (cur - e) * u * d
                    upct = d * (cur / e - 1.0) * 100.0
                stop = _f(p.get("stop"))
                tgt = _f(p.get("tgt"))
                positions.append({
                    "cell": cid, "sym": str(sym), "kind": kind,
                    "dir": d, "entry": e, "cur": cur, "qty": u,
                    "upnl": upnl, "upct": upct,
                    "stop": stop if stop else None,
                    # kind=="BBADD" 면 tgt 는 추매가 (목표가 아님)
                    "tgt": (tgt if tgt else None) if kind != "BBADD" else None,
                    "add": (tgt if tgt else None) if kind == "BBADD" else None,
                })
        # "사후 최대값 · 선택 금지" 태그 — 본 팜(E01~E10)에만
        if state_key == "cells":
            known = [r for r in rows if r["equity"] is not None]
            if known:
                max(known, key=lambda r: r["equity"])["is_max"] = True
        out.append({
            "key": state_key, "name": gname,
            "range": f"{specs[0].cell}~{specs[-1].cell}" if len(specs) > 1
                     else str(specs[0].cell) if specs else "",
            "rows": rows, "positions": positions,
            "t0": t0, "total": g_total if g_known else None,
            "n_cells": len(specs), "n_known": g_known,
            "capital": meta["capital0"] * len(specs),
        })
    return out


def build_recent_fills(limit: int = 8) -> list[dict]:
    """본·변형 원장에서 최근 체결 limit 건 (funding 류 제외)."""
    rows: list[dict] = []
    for path in (LOGS / "tracke_ledger.csv", LOGS / "tracke_variant_ledger.csv"):
        for i, r in enumerate(_load_csv(path)):
            action = str(r.get("action", ""))
            if not (action in FILL_ACTIONS or action.startswith("exit")):
                continue
            rows.append({
                "ts": _f(r.get("bar_close")) or 0.0, "seq": i,
                "cell": r.get("cell", "-"), "sym": r.get("sym", "-"),
                "strategy": r.get("strategy", "-"), "action": action,
                "price": _f(r.get("price")), "pnl": _f(r.get("pnl")),
                "dir": _f(r.get("direction")),
            })
    rows.sort(key=lambda r: (r["ts"], r["seq"]))
    return rows[-limit:][::-1]                   # 최신이 위


def build_sparkline(hist: list[dict], capital: float) -> dict | None:
    """팜 곡선 (equity 열) → SVG 폴리라인 좌표. 데이터 없으면 None."""
    pts = [(_f(r.get("ts")), _f(r.get("equity"))) for r in hist]
    pts = [(t, e) for t, e in pts if t is not None and e is not None]
    if len(pts) < 2:
        return None
    w, h, pad = 660, 90, 6
    ts = [p[0] for p in pts]
    eq = [p[1] for p in pts]
    lo, hi = min(eq + [capital]), max(eq + [capital])
    span = (hi - lo) or 1.0
    tspan = (ts[-1] - ts[0]) or 1.0

    def xy(t: float, e: float) -> tuple[float, float]:
        x = pad + (t - ts[0]) / tspan * (w - 2 * pad)
        y = pad + (hi - e) / span * (h - 2 * pad)
        return round(x, 1), round(y, 1)

    line = " ".join(f"{x},{y}" for x, y in (xy(t, e) for t, e in pts))
    base_y = xy(ts[0], capital)[1]
    return {"w": w, "h": h, "line": line, "base_y": base_y,
            "last": eq[-1], "lo": lo, "hi": hi,
            "pct": (eq[-1] / capital - 1.0) * 100.0 if capital else None,
            "t_from": fmt_ts_short(ts[0]), "t_to": fmt_ts_short(ts[-1])}


def build_h2_card() -> dict | None:
    """h2_cohort.json.gz 헤더 요약. 부재 시 None."""
    d = _load_json_gz(LOGS / "h2_cohort.json.gz")
    if not d:
        return None
    hdr = d.get("header") or {}
    counts = hdr.get("counts") or {}
    mde = hdr.get("mde") or {}
    return {
        "spec": hdr.get("spec"),
        "generated": hdr.get("generated_at_utc"),
        "cohort": counts.get("cohort"),
        "n_primary": counts.get("eligible_primary"),
        "mde_ic": _f(mde.get("ic")),
        "alpha": mde.get("alpha_one_sided"),
        "power": mde.get("power"),
    }


def build_market_scan(top_n: int = 5) -> dict | None:
    """market_scan.json → 돌파 근접(24h 채널 거리 최소) 상위 top_n. 부재 시 None."""
    d = _load_json(LOGS / "market_scan.json")
    if not d:
        return None
    coins = [c for c in (d.get("coins") or [])
             if isinstance(c, dict) and _f(c.get("dist24h_pct")) is not None]
    coins.sort(key=lambda c: _f(c.get("dist24h_pct")) or 0.0)
    return {
        "generated": d.get("generated_at_utc"),
        "n_universe": len(d.get("coins") or []),
        "top": coins[:top_n],
    }


def dday(dstr: str, today: date) -> str:
    """'YYYY-MM-DD' → D-day 문자열."""
    try:
        target = date.fromisoformat(dstr)
    except ValueError:
        return "-"
    n = (target - today).days
    if n > 0:
        return f"D-{n}"
    if n == 0:
        return "D-DAY"
    return f"D+{-n}"


# ---------------------------------------------------------------- HTML 렌더

CSS = """
:root {
  --bg: #f4f1ea; --card: #ffffff; --card2: #faf8f3;
  --ink: #23282e; --muted: #6b7280; --line: #e2ddd2;
  --accent: #8a6420; --accent-soft: #b98a2e;
  --green: #1a7f4b; --red: #c23a3a;
  --banner-bg: #fdf6e6; --banner-line: #d9a441;
  --tag-bg: #f0e6d2; --chip-bg: #eee9de;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #101418; --card: #171c22; --card2: #131820;
    --ink: #dbe2e8; --muted: #7d8a96; --line: #232b34;
    --accent: #d9a441; --accent-soft: #b98a2e;
    --green: #3ecf8e; --red: #f0625d;
    --banner-bg: #1a1710; --banner-line: #d9a441;
    --tag-bg: #2a2115; --chip-bg: #1d242c;
  }
}
:root[data-theme="dark"] {
  --bg: #101418; --card: #171c22; --card2: #131820;
  --ink: #dbe2e8; --muted: #7d8a96; --line: #232b34;
  --accent: #d9a441; --accent-soft: #b98a2e;
  --green: #3ecf8e; --red: #f0625d;
  --banner-bg: #1a1710; --banner-line: #d9a441;
  --tag-bg: #2a2115; --chip-bg: #1d242c;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: "IBM Plex Sans KR", -apple-system, "Apple SD Gothic Neo",
               "Noto Sans KR", sans-serif;
  font-size: 14px; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo,
        monospace; font-size: 0.92em; }
.wrap { max-width: 720px; margin: 0 auto; padding: 16px 12px 48px; }
h1 { font-size: 1.15rem; margin: 4px 0 2px; letter-spacing: 0.02em; }
h1 .amber { color: var(--accent); }
.sub { color: var(--muted); font-size: 0.78rem; margin-bottom: 14px; }
.banner {
  background: var(--banner-bg); border: 1px solid var(--banner-line);
  border-left: 4px solid var(--banner-line);
  border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;
  font-size: 0.85rem;
}
.banner strong { color: var(--accent); letter-spacing: 0.06em; }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 14px 14px 12px; margin-bottom: 16px;
}
.card h2 {
  font-size: 0.92rem; margin: 0 0 10px; color: var(--ink);
  display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
}
.card h2 .meta { color: var(--muted); font-size: 0.72rem; font-weight: 400; }
.tscroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
th {
  text-align: left; color: var(--muted); font-weight: 500;
  font-size: 0.7rem; padding: 4px 8px 4px 0; border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
td { padding: 5px 8px 5px 0; border-bottom: 1px solid var(--line);
     vertical-align: top; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
.num { text-align: right; }
th.num { text-align: right; }
.pos { color: var(--green); }
.neg { color: var(--red); }
.flat { color: var(--muted); }
.lbl { display: block; color: var(--muted); font-size: 0.66rem;
       white-space: normal; max-width: 240px; }
.tag {
  display: inline-block; background: var(--tag-bg); color: var(--accent);
  border: 1px solid var(--accent-soft); border-radius: 4px;
  font-size: 0.62rem; padding: 0 5px; margin-left: 6px; vertical-align: 1px;
}
.chip {
  display: inline-block; background: var(--chip-bg); color: var(--muted);
  border-radius: 4px; font-size: 0.66rem; padding: 1px 6px; margin-left: 6px;
}
.total-row td { border-top: 1px solid var(--line); color: var(--ink);
                font-weight: 600; }
.wait { color: var(--muted); font-size: 0.82rem; padding: 6px 0; }
.kv { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.8rem; }
.kv div { min-width: 120px; }
.kv .k { color: var(--muted); font-size: 0.68rem; display: block; }
.spark-note { color: var(--muted); font-size: 0.7rem; margin-top: 4px;
              display: flex; justify-content: space-between; }
svg.spark { width: 100%; height: auto; display: block; }
.footnote { color: var(--muted); font-size: 0.68rem; margin-top: 20px;
            line-height: 1.7; }
"""


def _render_group(g: dict, capital0: float) -> str:
    """계정 표 한 그룹 렌더."""
    h: list[str] = []
    note = "" if g["key"] == "cells" else \
        '<span class="chip">판정 비대상</span>'
    t0s = fmt_ts_ms(g["t0"]) if g["t0"] is not None else "대기"
    h.append(f'<div class="card"><h2>{esc(g["name"])} '
             f'<span class="mono">{esc(g["range"])}</span>{note}'
             f'<span class="meta">T0 {esc(t0s)} · 셀당 {fmt_usd(capital0, 0)}'
             f'</span></h2>')
    if g["n_known"] == 0:
        h.append('<div class="wait">대기 — 상태 데이터 없음</div></div>')
        return "".join(h)
    h.append('<div class="tscroll"><table><thead><tr>'
             '<th>셀 · 전략</th><th>바스켓</th>'
             '<th class="num">자산($)</th><th class="num">수익률</th>'
             '<th class="num">포지션</th></tr></thead><tbody>')
    for r in g["rows"]:                          # 고정 순서 (정렬 금지)
        tag = ('<span class="tag">사후 최대값 · 선택 금지</span>'
               if r["is_max"] else "")
        halted = '<span class="chip">중단</span>' if r["halted"] else ""
        lbl = f'<span class="lbl">{esc(r["label"])}</span>' if r["label"] else ""
        h.append(
            f'<tr><td><span class="mono">{esc(r["id"])}</span> '
            f'{esc(r["strategy"])}{tag}{halted}{lbl}</td>'
            f'<td><span class="mono">{esc(r["basket"])}</span> '
            f'<span class="lbl">{esc(r["basket_label"])}</span></td>'
            f'<td class="num mono">{fmt_usd(r["equity"]) if r["equity"] is not None else "대기"}</td>'
            f'<td class="num">{pnl_span(r["pct"], fmt_pct(r["pct"]))}</td>'
            f'<td class="num mono">{r["n_pos"]}</td></tr>')
    tot = g["total"]
    tot_pct = ((tot / g["capital"] - 1.0) * 100.0
               if tot is not None and g["capital"] else None)
    h.append(f'<tr class="total-row"><td>소계 ({g["n_known"]}/{g["n_cells"]}셀)'
             f'</td><td></td><td class="num mono">{fmt_usd(tot)}</td>'
             f'<td class="num">{pnl_span(tot_pct, fmt_pct(tot_pct))}</td>'
             f'<td class="num mono">{sum(r["n_pos"] for r in g["rows"])}</td>'
             '</tr>')
    h.append("</tbody></table></div></div>")
    return "".join(h)


def _render_positions(groups: list[dict]) -> str:
    """오픈 포지션 통합 표."""
    rows = [p for g in groups for p in g["positions"]]
    h = ['<div class="card"><h2>오픈 포지션 '
         f'<span class="meta">{len(rows)}건 · 현재가 = 마지막 1h 종가(ind.pc)'
         '</span></h2>']
    if not rows:
        h.append('<div class="wait">대기 — 오픈 포지션 없음</div></div>')
        return "".join(h)
    h.append('<div class="tscroll"><table><thead><tr>'
             '<th>셀</th><th>심볼</th><th>방향</th>'
             '<th class="num">진입가</th><th class="num">현재가</th>'
             '<th class="num">평가손익</th>'
             '<th class="num">손절가</th><th class="num">목표가</th>'
             '<th class="num">추매가</th></tr></thead><tbody>')
    for p in rows:
        if p["dir"] is None:
            dir_html = "-"
        elif p["dir"] > 0:
            dir_html = '<span class="pos">롱</span>'
        else:
            dir_html = '<span class="neg">숏</span>'
        upnl_html = ("-" if p["upnl"] is None else pnl_span(
            p["upnl"], f'{p["upnl"]:+,.2f} ({p["upct"]:+.2f}%)'))
        kind_chip = (f'<span class="chip">{esc(p["kind"])}</span>'
                     if p["kind"] else "")
        h.append(
            f'<tr><td class="mono">{esc(p["cell"])}</td>'
            f'<td class="mono">{esc(p["sym"])}{kind_chip}</td>'
            f'<td>{dir_html}</td>'
            f'<td class="num mono">{fmt_price(p["entry"])}</td>'
            f'<td class="num mono">{fmt_price(p["cur"])}</td>'
            f'<td class="num">{upnl_html}</td>'
            f'<td class="num mono">{fmt_price(p["stop"])}</td>'
            f'<td class="num mono">{fmt_price(p["tgt"])}</td>'
            f'<td class="num mono">{fmt_price(p["add"])}</td></tr>')
    h.append("</tbody></table></div>"
             '<div class="spark-note"><span>BBADD 셀의 tgt 는 추매 트리거가로,'
             ' 목표가 아님 — 추매가 열에 표기</span></div></div>')
    return "".join(h)


def _render_spark(spark: dict | None, capital: float) -> str:
    """팜 곡선 카드."""
    h = ['<div class="card"><h2>본 팜 곡선 '
         '<span class="meta">tracke_history.csv · 실현+평가 합산 스냅샷</span></h2>']
    if spark is None:
        h.append('<div class="wait">대기 — 곡선 데이터 없음</div></div>')
        return "".join(h)
    h.append(
        f'<svg class="spark" viewBox="0 0 {spark["w"]} {spark["h"]}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="팜 자본 곡선">'
        f'<line x1="0" y1="{spark["base_y"]}" x2="{spark["w"]}" '
        f'y2="{spark["base_y"]}" stroke="var(--line)" stroke-width="1" '
        'stroke-dasharray="4 4"/>'
        f'<polyline points="{spark["line"]}" fill="none" '
        'stroke="var(--accent)" stroke-width="1.8" stroke-linejoin="round" '
        'stroke-linecap="round"/></svg>')
    h.append(f'<div class="spark-note"><span>{esc(spark["t_from"])} → '
             f'{esc(spark["t_to"])} UTC</span>'
             f'<span class="mono">{fmt_usd(spark["last"])} '
             f'({fmt_pct(spark["pct"])}) · 저 {fmt_usd(spark["lo"], 0)} / '
             f'고 {fmt_usd(spark["hi"], 0)}</span></div></div>')
    return "".join(h)


def _render_fills(fills: list[dict]) -> str:
    """최근 체결 카드."""
    h = ['<div class="card"><h2>최근 체결 '
         '<span class="meta">본+변형 원장 · funding 제외 · 최신순</span></h2>']
    if not fills:
        h.append('<div class="wait">대기 — 체결 없음</div></div>')
        return "".join(h)
    h.append('<div class="tscroll"><table><thead><tr>'
             '<th>시각(UTC)</th><th>셀</th><th>심볼</th><th>전략</th>'
             '<th>액션</th><th class="num">가격</th><th class="num">손익</th>'
             '</tr></thead><tbody>')
    for r in fills:
        pnl = r["pnl"]
        pnl_html = pnl_span(pnl) if r["action"] != "enter" else \
            '<span class="mono flat">-</span>'
        h.append(
            f'<tr><td class="mono">{esc(fmt_ts_short(r["ts"]))}</td>'
            f'<td class="mono">{esc(r["cell"])}</td>'
            f'<td class="mono">{esc(r["sym"])}</td>'
            f'<td class="mono">{esc(r["strategy"])}</td>'
            f'<td class="mono">{esc(r["action"])}</td>'
            f'<td class="num mono">{fmt_price(r["price"])}</td>'
            f'<td class="num">{pnl_html}</td></tr>')
    h.append("</tbody></table></div></div>")
    return "".join(h)


def _render_h2(h2: dict | None) -> str:
    """H2 카드."""
    h = ['<div class="card"><h2>H2 — 하방 일관성 전향 연구 '
         '<span class="meta">h2_cohort.json.gz 헤더</span></h2>']
    if h2 is None:
        h.append('<div class="wait">대기 — 코호트 파일 없음</div></div>')
        return "".join(h)
    mde = f'{h2["mde_ic"]:.3f}' if h2["mde_ic"] is not None else "대기"
    h.append('<div class="kv">'
             f'<div><span class="k">1차 적격 표본</span>'
             f'<span class="mono">n = {esc(h2["n_primary"])}</span></div>'
             f'<div><span class="k">MDE (IC)</span>'
             f'<span class="mono">{esc(mde)}</span></div>'
             f'<div><span class="k">단측 α / 검정력</span>'
             f'<span class="mono">{esc(h2["alpha"])} / {esc(h2["power"])}'
             '</span></div>'
             f'<div><span class="k">스크린 코호트</span>'
             f'<span class="mono">{esc(h2["cohort"])} 지갑</span></div>'
             '</div>')
    if h2["spec"]:
        h.append(f'<div class="spark-note"><span>{esc(h2["spec"])} · 동결 '
                 f'{esc(h2["generated"])}</span></div>')
    h.append("</div>")
    return "".join(h)


def _render_verdicts(today: date) -> str:
    """판정 일정 D-day 카드 (고정표 — 사전 지정 시점만)."""
    h = ['<div class="card"><h2>판정 일정 '
         '<span class="meta">사전 지정 시점만 · 중간 성과로 판단 금지</span></h2>'
         '<div class="tscroll"><table><thead><tr>'
         '<th>날짜</th><th>대상</th><th class="num">D-day</th>'
         '</tr></thead><tbody>']
    for dstr, label in VERDICT_SCHEDULE:
        dd = dday(dstr, today)
        past = dd.startswith("D+")
        cls = "flat" if past else ("pos" if dd == "D-DAY" else "")
        h.append(f'<tr><td class="mono">{esc(dstr)}</td><td>{esc(label)}</td>'
                 f'<td class="num"><span class="mono {cls}">{esc(dd)}</span>'
                 '</td></tr>')
    h.append("</tbody></table></div></div>")
    return "".join(h)


def _render_scan(scan: dict | None) -> str:
    """시장 스캔 카드 (있을 때만)."""
    if scan is None:
        return ""
    h = [f'<div class="card"><h2>시장 스캔 — 돌파 근접 상위 {len(scan["top"])} '
         f'<span class="meta">유니버스 {scan["n_universe"]}코인 · '
         f'{esc(scan["generated"])}</span></h2>']
    if not scan["top"]:
        h.append('<div class="wait">대기 — 스캔 데이터 없음</div></div>')
        return "".join(h)
    h.append('<div class="tscroll"><table><thead><tr>'
             '<th>코인</th><th class="num">가격</th><th class="num">24h</th>'
             '<th class="num">24h 채널 거리</th><th class="num">RSI14</th>'
             '<th class="num">거래량 서지</th></tr></thead><tbody>')
    for c in scan["top"]:
        chg = _f(c.get("chg24h_pct"))
        dist = _f(c.get("dist24h_pct"))
        rsi = _f(c.get("rsi14"))
        surge = _f(c.get("vol_surge"))
        h.append(
            f'<tr><td class="mono">{esc(c.get("coin", "-"))}</td>'
            f'<td class="num mono">{fmt_price(_f(c.get("price")))}</td>'
            f'<td class="num">{pnl_span(chg, fmt_pct(chg))}</td>'
            f'<td class="num mono">{f"{dist:.2f}%" if dist is not None else "-"}</td>'
            f'<td class="num mono">{f"{rsi:.1f}" if rsi is not None else "-"}</td>'
            f'<td class="num mono">{f"{surge:.2f}" if surge is not None else "-"}</td>'
            '</tr>')
    h.append("</tbody></table></div>"
             '<div class="spark-note"><span>표시 전용 — 진입 신호 아님</span>'
             "</div></div>")
    return "".join(h)


def render_html(state: dict | None, meta: dict) -> str:
    """전체 페이지 HTML 조립."""
    today = datetime.now(timezone.utc).date()
    groups = build_groups(state, meta)
    main = next((g for g in groups if g["key"] == "cells"), None)
    variants = [g for g in groups if g["key"] != "cells"]
    cap0 = meta["capital0"]
    n_main = main["n_cells"] if main else 0
    n_var = sum(g["n_cells"] for g in variants)
    cap_main = cap0 * n_main
    cap_var = cap0 * n_var
    cap_total = cap_main + cap_var
    last_ts = fmt_ts_ms((state or {}).get("last_ts"))
    hist = _load_csv(LOGS / "tracke_history.csv")
    spark = build_sparkline(hist, cap_main) if cap_main else None
    fills = build_recent_fills(8)
    h2 = build_h2_card()
    scan = build_market_scan(5)

    body: list[str] = []
    body.append('<div class="wrap">')
    body.append('<h1>단타 팜 <span class="amber">상황판</span></h1>')
    body.append(f'<div class="sub">외부 스냅샷 · 데이터 기준 {esc(last_ts)} · '
                f'생성 {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}'
                '</div>')
    if cap_total:
        body.append(
            '<div class="banner"><strong>PAPER ONLY</strong> — 가상 자본 합계 '
            f'<span class="mono">{fmt_usd(cap_total, 0)}</span> = '
            f'본 팜 {n_main}셀 <span class="mono">{fmt_usd(cap_main, 0)}</span>'
            f' + 변형 {n_var}셀 <span class="mono">{fmt_usd(cap_var, 0)}</span>'
            f' (셀당 {fmt_usd(cap0, 0)}) · 실거래 아님 · 판정은 사전 지정 '
            '시점만</div>')
    else:
        body.append('<div class="banner"><strong>PAPER ONLY</strong> — '
                    '대기 (엔진 메타 로드 실패)</div>')
    body.append(_render_spark(spark, cap_main))
    if groups:
        for g in groups:
            body.append(_render_group(g, cap0))
    else:
        body.append('<div class="card"><h2>계정</h2>'
                    '<div class="wait">대기 — 셀 메타 없음</div></div>')
    body.append(_render_positions(groups))
    body.append(_render_fills(fills))
    body.append(_render_h2(h2))
    body.append(_render_verdicts(today))
    body.append(_render_scan(scan))
    body.append(
        '<div class="footnote">계정 표는 스펙 고정 순서 (성과순 정렬 금지) · '
        '"사후 최대값 · 선택 금지" 태그는 본 팜 E01~E10 에만 적용 · '
        '변형 셀은 전부 공식 판정 비대상 · 이 페이지는 표시 전용 스냅샷으로 '
        '판정 입력이 아님</div>')
    body.append("</div>")

    return (
        "<title>단타 팜 상황판</title>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500&'
        'family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap" '
        'rel="stylesheet">\n'
        f"<style>{CSS}</style>\n" + "".join(body) + "\n")


# ---------------------------------------------------------------- 엔트리

def main() -> int:
    """CLI 엔트리 — logs 를 읽어 --out 경로에 HTML 을 쓴다."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="단타 팜 외부 스냅샷 상황판 생성")
    ap.add_argument("--out", required=True, help="출력 HTML 경로")
    args = ap.parse_args()

    meta = _load_engine_meta()
    state = _load_json(LOGS / "tracke_state.json")
    page = render_html(state, meta)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    logger.info("상황판 생성 완료: %s (%d bytes)", out, len(page.encode()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
