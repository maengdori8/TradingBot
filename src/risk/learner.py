"""
자동 파라미터 튜너 — 청산된 거래를 분석해 "어떤 상황에서 잃고 버는지"를 스스로 파악하고
진입 기준(scan.min_score, risk_tiers, min_rr)을 보수적으로 자동 조정한다.

ML이 아니라 조건부 빈도 추정 + 보수적 의사결정 규칙이다:
  - 청산된 거래만 사용 (미청산 포지션 제외 → 룩어헤드/누수 차단)
  - 세그먼트(세션/방향/점수대)별 라플라스 평활 승률 + Wilson 95% 신뢰구간 + R-multiple 기대값
  - 통계적으로 신뢰할 만한 신호에 대해서만, 한 run당 1개 파라미터를 작은 스텝으로 조정
  - config.yaml은 절대 안 씀 → logs/learned_params.yaml 오버레이에만 기록 (로더가 병합)
  - 무게중심은 "변경 보류(no-op)" — 표본 부족/애매하면 아무것도 안 건드림

CLI:
    python -m src.risk.learner --dry-run   # 분석/제안만 (기본)
    python -m src.risk.learner --apply      # 실제 오버레이 적용
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config_loader import (
    MIN_RR_MAX,
    MIN_RR_MIN,
    MIN_SCORE_MAX,
    MIN_SCORE_MIN,
    OVERLAY_PATH,
    RISK_PCT_MAX,
    RISK_PCT_MIN,
    load_config,
    load_raw_config,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "logs" / "paper_trades.db"
REPORT_PATH = ROOT / "logs" / "learning_report.json"
HISTORY_PATH = ROOT / "logs" / "tuning_history.jsonl"

MIN_SCORE_DRIFT = 10        # baseline 대비 누적 변화 한도
RISK_PCT_DRIFT = 0.004      # risk_pct baseline 대비 누적 변화 한도
Z_EXPAND = 1.645            # 확대(위험) 판정: 단측 95% — 엄격


# 스텝
RISK_STEP_UP, RISK_STEP_DOWN = 1.15, 0.85
MINSCORE_STEP = 2
MINRR_STEP = 0.25


# ──────────────────────────────────────────────────────────────────────
# 통계 헬퍼
# ──────────────────────────────────────────────────────────────────────

def laplace_winrate(wins: int, n: int) -> float:
    """라플라스 평활 승률 = (wins+1)/(n+2). 극단(0/100%) 추정 방지."""
    return (wins + 1) / (n + 2)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """승률의 Wilson 95% 신뢰구간 (lower, upper)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def expectancy_r(trades: list[dict]) -> float | None:
    """거래 리스트의 평균 R-multiple. r_multiple 있는 거래만. 없으면 None."""
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    if not rs:
        return None
    return sum(rs) / len(rs)


def _wins(trades: list[dict]) -> int:
    """R-multiple > 0 인 거래 수 (r_multiple 있는 것만)."""
    return sum(1 for t in trades if t.get("r_multiple") is not None and t["r_multiple"] > 0)


def _n_with_r(trades: list[dict]) -> int:
    """r_multiple 이 기록된 거래 수."""
    return sum(1 for t in trades if t.get("r_multiple") is not None)


def expectancy_lower_bound(trades: list[dict], z: float) -> float | None:
    """평균 R-multiple의 단측 신뢰 하한 = mean - z*SE. 표본<2면 None."""
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    if len(rs) < 2:
        return None
    mean = sum(rs) / len(rs)
    var = sum((x - mean) ** 2 for x in rs) / (len(rs) - 1)  # 표본분산
    se = math.sqrt(var) / math.sqrt(len(rs))
    return mean - z * se


def _seg_stats(trades: list[dict]) -> dict:
    """세그먼트 통계 묶음 (승률 + R-multiple 기대값/신뢰하한)."""
    n = _n_with_r(trades)
    w = _wins(trades)
    lb, ub = wilson_interval(w, n) if n else (0.0, 1.0)
    e = expectancy_r(trades)
    e_lb = expectancy_lower_bound(trades, Z_EXPAND)
    return {
        "n": n,
        "wins": w,
        "laplace_winrate": round(laplace_winrate(w, n), 4) if n else None,
        "wilson_lb": round(lb, 4),
        "wilson_ub": round(ub, 4),
        "expectancy_r": round(e, 4) if e is not None else None,
        # 확대 판정용: 평균 R의 95% 단측 하한 (이게 0보다 크면 통계적으로 흑자 확신)
        "expectancy_lb": round(e_lb, 4) if e_lb is not None else None,
    }


# ──────────────────────────────────────────────────────────────────────
# 데이터 수집 / 세그먼트화
# ──────────────────────────────────────────────────────────────────────

def _open_position_ids(conn: sqlite3.Connection) -> set[str]:
    """현재 미청산 포지션 id 집합."""
    try:
        return {r[0] for r in conn.execute("SELECT id FROM open_positions")}
    except sqlite3.OperationalError:
        return set()


def fetch_closed_trades(conn: sqlite3.Connection, lookback: int = 0) -> list[dict]:
    """청산된 거래만 조회한다 (미청산 open_positions는 절대 포함 안 함).

    부분청산 행(id="parent-타임스탬프")은 모(母)포지션이 아직 열려 있으면 제외한다
    (잔여가 살아있는 부분결과 = 생존편향/룩어헤드).

    Args:
        conn: paper_trades.db 연결
        lookback: 최근 N건만 (0이면 전체)

    Returns:
        거래 dict 리스트 (시간순)
    """
    cols = ("id, direction, entry_score, entry_session, status, "
            "entry_rr, risk_amount, r_multiple, pnl, exit_time")
    rows = conn.execute(
        f"SELECT {cols} FROM trades WHERE status IN ('SL','TP','manual') "
        "ORDER BY exit_time ASC"
    ).fetchall()
    keys = ["id", "direction", "entry_score", "entry_session", "status",
            "entry_rr", "risk_amount", "r_multiple", "pnl", "exit_time"]
    trades = [dict(zip(keys, r)) for r in rows]

    # 부분청산 + 모포지션 미청산 → 제외 (잔여 포지션이 살아있는 부분결과)
    open_ids = _open_position_ids(conn)
    if open_ids:
        trades = [
            t for t in trades
            if not ("-" in t["id"] and t["id"].split("-", 1)[0] in open_ids)
        ]

    if lookback and len(trades) > lookback:
        trades = trades[-lookback:]
    return trades


def count_closed_with_r(conn: sqlite3.Connection) -> int:
    """lookback 무관 전체 청산거래(r_multiple 기록) 누계 — 쿨다운/게이트 기준."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status IN ('SL','TP','manual') "
            "AND r_multiple IS NOT NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def compute_mdd(conn: sqlite3.Connection, initial_balance: float) -> float:
    """청산거래 누적손익 기준 최대낙폭(MDD) — paper 없이도 킬스위치 평가용."""
    rows = conn.execute(
        "SELECT pnl FROM trades WHERE pnl IS NOT NULL AND status IN ('SL','TP','manual') "
        "ORDER BY exit_time ASC"
    ).fetchall()
    if not rows or initial_balance <= 0:
        return 0.0
    equity = initial_balance
    peak = initial_balance
    mdd = 0.0
    for (pnl,) in rows:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            mdd = max(mdd, (peak - equity) / peak)
    return mdd


def _score_bucket(score: float | None, boundaries: list[float]) -> str | None:
    """점수를 risk_tiers 경계 기준 버킷명으로 변환. boundaries 내림차순(예: [85,75,70])."""
    if score is None:
        return None
    for b in boundaries:
        if score >= b:
            return f">={b:g}"
    return None


def aggregate(trades: list[dict], tier_boundaries: list[float]) -> dict:
    """세그먼트별 통계 집계 (세션/방향/점수버킷)."""
    overall = _seg_stats(trades)
    baseline_e = overall["expectancy_r"]

    by_session: dict[str, dict] = {}
    for sess in ["london", "newyork", None]:
        seg = [t for t in trades if t.get("entry_session") == sess]
        if seg:
            by_session[str(sess)] = _seg_stats(seg)

    by_direction: dict[str, dict] = {}
    for d in ["long", "short"]:
        seg = [t for t in trades if t.get("direction") == d]
        if seg:
            by_direction[d] = _seg_stats(seg)

    by_bucket: dict[str, dict] = {}
    for b in tier_boundaries:
        seg = [t for t in trades if _score_bucket(t.get("entry_score"), tier_boundaries) == f">={b:g}"]
        by_bucket[f">={b:g}"] = _seg_stats(seg)

    return {
        "overall": overall,
        "baseline_expectancy_r": baseline_e,
        "by_session": by_session,
        "by_direction": by_direction,
        "by_score_bucket": by_bucket,
    }


# ──────────────────────────────────────────────────────────────────────
# 의사결정
# ──────────────────────────────────────────────────────────────────────

def decide_adjustment(agg: dict, cfg: dict, baseline: dict, state: dict) -> dict | None:
    """가드레일/게이트를 통과한 단 1개의 조정을 결정한다. 없으면 None.

    의사결정은 R-multiple 기대값 기반(승률 손익분기 비교 아님):
      - 위험 축소(안전): 기대값 점추정이 명백히 음수(< -deadband)면 발동 — 쉽게.
      - 위험 확대(위험): 평균 R의 95% 단측 신뢰하한 > 0 이고 기대값 > deadband 일 때만 — 엄격.
    안전 행동을 위험 행동보다 쉽게 게이팅(자본 보호 우선). 킬스위치 시 확대 동결.
    쿨다운 기준 카운터(cool_n)는 lookback 무관 전체 누계.
    """
    risk = cfg.get("risk", {})
    scan = cfg.get("scan", {})
    learning = cfg.get("learning", {})

    bucket_min_n = learning.get("bucket_min_n", 12)
    global_min_score_n = learning.get("global_min_n_for_min_score", 20)
    cooldown = learning.get("cooldown_trades", 10)
    deadband = learning.get("deadband_R", 0.05)

    min_rr = risk.get("min_rr_ratio", 2.0)
    total_n = agg["overall"]["n"]
    cool_n = state.get("cool_n", total_n)   # 쿨다운 기준(전체 누계)
    tiers = risk.get("risk_tiers", [])
    boundaries = sorted([t["min_score"] for t in tiers], reverse=True)
    last_change = state.get("last_change_at", {})
    risk_expansion_frozen = state.get("kill_switch", False)
    base_tiers = {t.get("min_score"): t.get("risk_pct") for t in baseline.get("risk.risk_tiers", [])}

    def cooled(param: str) -> bool:
        """쿨다운 통과 (직전 변경 이후 cooldown건 이상 신규 청산)."""
        return (cool_n - last_change.get(param, -10**9)) >= cooldown

    def losing(seg: dict) -> bool:
        """명백한 손실 세그먼트 (기대값 점추정 < -deadband). 안전 판정 — 쉽게."""
        e = seg.get("expectancy_r")
        return e is not None and e < -deadband

    def winning(seg: dict) -> bool:
        """통계적으로 흑자 확신 (기대값 > deadband AND 95% 단측 하한 > 0). 위험 판정 — 엄격."""
        e = seg.get("expectancy_r")
        lb = seg.get("expectancy_lb")
        return e is not None and e > deadband and lb is not None and lb > 0

    # ── 1) 위험 축소: 지는 점수버킷 risk_pct 하향 (약한 버킷부터) ──
    for b in sorted(boundaries):
        seg = agg["by_score_bucket"].get(f">={b:g}", {})
        if seg.get("n", 0) < bucket_min_n or not losing(seg):
            continue
        tier = next((t for t in tiers if t["min_score"] == b), None)
        if not tier:
            continue
        param = f"risk_tiers[{b:g}].risk_pct"
        if not cooled(param):
            continue
        cur = tier["risk_pct"]
        new = max(RISK_PCT_MIN, round(cur * RISK_STEP_DOWN, 6))
        if new >= cur:
            continue
        return {
            "kind": "risk_pct", "tier_min_score": b, "param": param,
            "old": cur, "new": new, "segment": seg,
            "reason": f"점수 >={b:g} 버킷 {seg['n']}건 기대값 {seg['expectancy_r']:+.2f}R 적자 → 리스크 축소",
        }

    # ── 2) 위험 축소: 최저 버킷이 지면 min_score 상향 ──
    if boundaries:
        lowest = min(boundaries)
        seg = agg["by_score_bucket"].get(f">={lowest:g}", {})
        cur_ms = scan.get("min_score", 70)
        base_ms = baseline.get("scan.min_score", cur_ms)
        if (seg.get("n", 0) >= bucket_min_n and total_n >= global_min_score_n
                and cooled("scan.min_score") and losing(seg)
                and cur_ms < MIN_SCORE_MAX and (cur_ms - base_ms) < MIN_SCORE_DRIFT):
            new = min(MIN_SCORE_MAX, cur_ms + MINSCORE_STEP)
            if new > cur_ms:
                return {
                    "kind": "min_score", "param": "scan.min_score",
                    "old": cur_ms, "new": new, "segment": seg,
                    "reason": f"최저 점수버킷 {seg['n']}건 기대값 {seg['expectancy_r']:+.2f}R 적자 → 진입 문턱 상향",
                }

    # ── 3) 위험 축소: min_rr 상향 (승률 양호한데 기대값 음수 = 손익비 나쁨) ──
    ov = agg["overall"]
    if (total_n >= 30 and cooled("min_rr_ratio") and min_rr < MIN_RR_MAX
            and losing(ov)
            and ov["laplace_winrate"] is not None and ov["laplace_winrate"] > 0.5):
        new = min(MIN_RR_MAX, round(min_rr + MINRR_STEP, 2))
        if new > min_rr:
            return {
                "kind": "min_rr", "param": "min_rr_ratio",
                "old": min_rr, "new": new, "segment": ov,
                "reason": f"전체 {total_n}건 승률 {ov['laplace_winrate']:.0%} 양호하나 기대값 {ov['expectancy_r']:+.2f}R 음수 → 목표 R:R 상향",
            }

    if risk_expansion_frozen:
        return None  # 킬스위치: 위험 확대 동결

    # ── 4) 위험 확대: 흑자 확신 버킷 risk_pct 상향 (높은 점수부터, 엄격) ──
    for b in sorted(boundaries, reverse=True):
        seg = agg["by_score_bucket"].get(f">={b:g}", {})
        if seg.get("n", 0) < bucket_min_n or not winning(seg):
            continue
        tier = next((t for t in tiers if t["min_score"] == b), None)
        if not tier:
            continue
        param = f"risk_tiers[{b:g}].risk_pct"
        if not cooled(param):
            continue
        cur = tier["risk_pct"]
        new = min(RISK_PCT_MAX, round(cur * RISK_STEP_UP, 6))
        if new <= cur:
            continue
        # baseline 대비 누적 드리프트 캡 (위험 방향 폭주 방지)
        base_p = base_tiers.get(b)
        if base_p is not None and new - base_p > RISK_PCT_DRIFT:
            continue
        if not _monotonic_ok(tiers, b, new):
            continue
        return {
            "kind": "risk_pct", "tier_min_score": b, "param": param,
            "old": cur, "new": new, "segment": seg,
            "reason": f"점수 >={b:g} 버킷 {seg['n']}건 기대값 {seg['expectancy_r']:+.2f}R 흑자확신(하한 {seg['expectancy_lb']:+.2f}R) → 리스크 확대",
        }

    # ── 5) 위험 확대: 전 버킷 흑자확신이면 min_score 하향 (baseline 복귀 방향만) ──
    cur_ms = scan.get("min_score", 70)
    base_ms = baseline.get("scan.min_score", cur_ms)
    if cur_ms > base_ms and total_n >= global_min_score_n and cooled("scan.min_score"):
        qualified = [s for s in agg["by_score_bucket"].values() if s.get("n", 0) >= bucket_min_n]
        if qualified and all(winning(s) for s in qualified):
            new = max(base_ms, cur_ms - MINSCORE_STEP)
            if new < cur_ms:
                return {
                    "kind": "min_score", "param": "scan.min_score",
                    "old": cur_ms, "new": new, "segment": agg["overall"],
                    "reason": f"전 점수버킷 흑자확신 → 진입 문턱 완화(원값 {base_ms}로 복귀)",
                }

    return None


def _monotonic_ok(tiers: list[dict], changed_min_score: float, new_pct: float) -> bool:
    """risk_tiers 단조성: 점수 높을수록 risk_pct가 같거나 커야 한다."""
    pct_by = {t["min_score"]: t["risk_pct"] for t in tiers}
    pct_by[changed_min_score] = new_pct
    ordered = sorted(pct_by.items(), key=lambda x: x[0])  # 점수 오름차순
    pcts = [p for _, p in ordered]
    return all(pcts[i] <= pcts[i + 1] + 1e-12 for i in range(len(pcts) - 1))


# ──────────────────────────────────────────────────────────────────────
# 오버레이 쓰기 (원자적 + 백업 + 검증)
# ──────────────────────────────────────────────────────────────────────

def _current_overlay() -> dict:
    """기존 오버레이를 읽는다 (없으면 빈 구조)."""
    if OVERLAY_PATH.exists():
        try:
            with open(OVERLAY_PATH) as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            pass
    return {}


def _build_baseline(raw_cfg: dict) -> dict:
    """현재 config.yaml 원본에서 baseline 스냅샷 생성 (드리프트 감지용)."""
    risk = raw_cfg.get("risk", {})
    return {
        "scan.min_score": raw_cfg.get("scan", {}).get("min_score"),
        "risk.min_rr_ratio": risk.get("min_rr_ratio"),
        "risk.risk_tiers": [
            {"min_score": t.get("min_score"), "risk_pct": t.get("risk_pct")}
            for t in risk.get("risk_tiers", [])
        ],
    }


def _apply_change_to_values(values: dict, adj: dict) -> dict:
    """조정 결과를 오버레이 values에 누적 반영한다."""
    values = dict(values or {})
    if adj["kind"] == "min_score":
        values.setdefault("scan", {})["min_score"] = adj["new"]
    elif adj["kind"] == "min_rr":
        values.setdefault("risk", {})["min_rr_ratio"] = adj["new"]
    elif adj["kind"] == "risk_pct":
        risk = values.setdefault("risk", {})
        tiers = risk.get("risk_tiers", [])
        found = False
        for t in tiers:
            if t.get("min_score") == adj["tier_min_score"]:
                t["risk_pct"] = adj["new"]
                found = True
        if not found:
            tiers.append({"min_score": adj["tier_min_score"], "risk_pct": adj["new"]})
        risk["risk_tiers"] = tiers
    return values


def write_overlay(adj: dict, raw_cfg: dict, total_n: int) -> bool:
    """오버레이를 원자적으로 갱신한다. 성공 시 True.

    tempfile→os.replace 원자 교체, .bak 백업, 쓰기 후 재파싱+가드레일 검증(실패 시 롤백).
    """
    existing = _current_overlay()
    values = _apply_change_to_values(existing.get("values", {}), adj)

    meta = existing.get("meta", {})
    last_change = meta.get("last_change_at", {})
    last_change[adj["param"]] = total_n
    changes_hist = meta.get("changes", [])[-9:]  # 최근 10개 유지
    changes_hist.append({
        "param": adj["param"], "old": adj["old"], "new": adj["new"],
        "reason": adj["reason"], "trades_analyzed": total_n,
    })

    overlay = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trades_analyzed": total_n,
            "last_change_at": last_change,
            "changes": changes_hist,
            "note": "AUTO-GENERATED by src/risk/learner.py — config.yaml은 수정하지 않음",
        },
        "baseline": _build_baseline(raw_cfg),
        "values": values,
    }

    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OVERLAY_PATH.exists():
        shutil.copy2(OVERLAY_PATH, OVERLAY_PATH.with_suffix(".yaml.bak"))

    fd, tmp = tempfile.mkstemp(dir=str(OVERLAY_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(overlay, f, allow_unicode=True, sort_keys=False)
        # 검증: 재파싱 + 가드레일 범위
        with open(tmp) as f:
            reloaded = yaml.safe_load(f)
        if not _validate_overlay(reloaded):
            raise ValueError("오버레이 가드레일 검증 실패")
        os.replace(tmp, OVERLAY_PATH)
        return True
    except Exception as e:
        logger.error("오버레이 쓰기 실패, 롤백: %s", e)
        if os.path.exists(tmp):
            os.remove(tmp)
        bak = OVERLAY_PATH.with_suffix(".yaml.bak")
        if bak.exists():
            shutil.copy2(bak, OVERLAY_PATH)
        return False


def _validate_overlay(overlay: dict) -> bool:
    """오버레이 값이 하드 가드레일 범위 안인지 검증."""
    if not isinstance(overlay, dict):
        return False
    v = overlay.get("values", {})
    ms = v.get("scan", {}).get("min_score")
    if ms is not None and not (MIN_SCORE_MIN <= ms <= MIN_SCORE_MAX):
        return False
    rr = v.get("risk", {}).get("min_rr_ratio")
    if rr is not None and not (MIN_RR_MIN <= rr <= MIN_RR_MAX):
        return False
    for t in v.get("risk", {}).get("risk_tiers", []):
        p = t.get("risk_pct")
        if p is not None and not (RISK_PCT_MIN <= p <= RISK_PCT_MAX):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────
# 리포트 / 알림
# ──────────────────────────────────────────────────────────────────────

def write_report(agg: dict, decision: dict | None, applied: bool) -> None:
    """세그먼트 리포트를 JSON으로 저장 (대시보드/사람 확인용)."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segments": agg,
        "decision": decision,
        "applied": applied,
    }
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("학습 리포트 저장 실패: %s", e)


def _append_history(adj: dict, total_n: int) -> None:
    """튜닝 이력 jsonl append."""
    try:
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "param": adj["param"], "old": adj["old"], "new": adj["new"],
                "reason": adj["reason"], "trades_analyzed": total_n,
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────────────

def maybe_update(paper=None, notifier=None, dry_run: bool | None = None,
                 conn: sqlite3.Connection | None = None) -> dict:
    """run() 말미 호출 진입점 — 자격검사 통과 시에만 1개 파라미터 조정.

    Args:
        paper: PaperEngine (conn 제공). conn 직접 줄 수도 있음.
        notifier: DiscordNotifier (옵셔널, notify_tuning 있으면 호출)
        dry_run: True면 제안만 (config의 learning.dry_run 기본)
        conn: SQLite 연결 (테스트용)

    Returns:
        결과 dict (status, decision 등)
    """
    cfg = load_config()
    learning = cfg.get("learning", {})
    if not learning.get("enabled", True):
        return {"status": "disabled"}

    if dry_run is None:
        dry_run = learning.get("dry_run", False)

    # 단독 실행 시 _init_db로 열어 마이그레이션 보장 (구스키마 DB 대응)
    if conn is not None:
        db_conn = conn
    elif paper is not None:
        db_conn = paper.conn
    else:
        from src.paper_trading.paper_engine import _init_db
        db_conn = _init_db(DB_PATH)
    lookback = learning.get("lookback_trades", 0)
    trades = fetch_closed_trades(db_conn, lookback)        # 통계용(윈도우 가능)
    total_n = _n_with_r(trades)
    cum_n = count_closed_with_r(db_conn)                    # 게이트/쿨다운 기준(전체 누계)

    min_trades = learning.get("min_trades_to_learn", 20)
    every_n = learning.get("every_n_trades", 5)

    # 자격검사: 전역 표본 부족 또는 직전 평가 이후 변화 미미 → no-op (전체 누계 기준)
    existing = _current_overlay()
    last_analyzed = existing.get("meta", {}).get("trades_analyzed", 0)
    if cum_n < min_trades:
        return {"status": "insufficient", "n": cum_n, "need": min_trades}
    if existing and (cum_n - last_analyzed) < every_n:
        return {"status": "cooldown_global", "n": cum_n, "since_last": cum_n - last_analyzed}

    tiers = cfg.get("risk", {}).get("risk_tiers", [])
    boundaries = sorted([t["min_score"] for t in tiers], reverse=True) or [70]
    agg = aggregate(trades, boundaries)

    # 킬스위치: MDD 초과 시 위험 확대 동결 (paper 유무 무관 — conn으로 계산)
    state = {
        "last_change_at": existing.get("meta", {}).get("last_change_at", {}),
        "cool_n": cum_n,
        "kill_switch": False,
    }
    try:
        cap = cfg.get("capital", {})
        initial = cap.get("total_capital", 5000) * cap.get("trading_allocation", 0.25)
        max_mdd = cfg.get("promote", {}).get("max_mdd", 0.05)
        if compute_mdd(db_conn, initial) > max_mdd:
            state["kill_switch"] = True
    except Exception:
        pass

    baseline = _build_baseline(load_raw_config())
    decision = decide_adjustment(agg, cfg, baseline, state)

    applied = False
    if decision and not dry_run:
        applied = write_overlay(decision, load_raw_config(), cum_n)
        if applied:
            _append_history(decision, cum_n)
            logger.info(
                "🧠 자동 튜닝 적용: %s %s→%s | %s",
                decision["param"], decision["old"], decision["new"], decision["reason"],
            )
            if notifier is not None and hasattr(notifier, "notify_tuning"):
                try:
                    notifier.notify_tuning(decision)
                except Exception as e:
                    logger.warning("튜닝 알림 실패: %s", e)
    elif decision:
        logger.info(
            "🧠 자동 튜닝 제안(dry-run): %s %s→%s | %s",
            decision["param"], decision["old"], decision["new"], decision["reason"],
        )
    else:
        logger.info("🧠 자동 튜닝: 조정 없음 (표본/신뢰 부족 또는 안정) — 거래 %d건 분석", total_n)

    write_report(agg, decision, applied)
    if conn is None and paper is None:
        db_conn.close()
    return {"status": "ok", "n": total_n, "decision": decision, "applied": applied}


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="ICT 봇 자동 파라미터 튜너")
    parser.add_argument("--apply", action="store_true", help="실제 오버레이 적용 (기본: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="제안만 (기본값)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    dry = not args.apply
    result = maybe_update(dry_run=dry)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
