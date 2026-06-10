"""
출구 관리 변형 검증 — 6개월 재현 신호에 BE스톱/부분익절/트레일링/시간손절을 적용해
승률·기대값 변화를 측정한다 (외부 문헌 규칙의 사전 검증용).

베이스라인: 고정 손절 2.0ATR / 목표 RR 2.5 (현행 봇).
보수 규칙: 동시터치=손절 우선, 상태 변경(BE암/트레일)은 다음 캔들부터 적용(인트라캔들 낙관 방지).
비용: 왕복 0.21% (study.py와 동일, 펀딩 제외 — 베이스라인과 비교 가능성 유지).

사용법: python3 research/exit_variants.py
출력: research/out/exit_variants.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "research" / "data"
OUT_DIR = ROOT / "research" / "out"

logger = logging.getLogger("exit_variants")

COST_PCT = 0.0021
SL_MULT = 2.0          # 현행 손절 (ATR 배수)
RR = 2.5               # 현행 목표
MAX_HOLD = 672         # 7일 (15m)


def _san(symbol: str) -> str:
    """심볼 → 파일명."""
    return symbol.replace("/", "_").replace(":", "-")


def simulate_variants(entry: float, direction: str, dist: float,
                      highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                      start: int) -> dict:
    """한 신호에 대해 출구 변형별 순R을 계산한다.

    변형:
      base     — 고정 SL/TP (현행)
      be1r     — +1R 도달 다음 캔들부터 SL=본전
      partial  — +1.25R에 50% 익절 + 잔여 BE, TP 2.5R
      trail1r  — +1R 도달 후 1.0ATR 트레일링 (TP 없음, 트레일/SL로만 청산)
      time48h  — 고정 SL/TP + 48시간 미해결 시 종가 청산
    """
    end = min(start + MAX_HOLD, len(closes))
    if end <= start:
        return {}
    sgn = 1.0 if direction == "long" else -1.0
    cost_r = COST_PCT * entry / dist

    tp = entry + sgn * dist * RR
    arm = entry + sgn * dist * 1.0          # +1R 레벨
    partial_tp = entry + sgn * dist * 1.25  # 부분익절 레벨

    def hit_sl(j: int, sl: float) -> bool:
        return lows[j] <= sl if sgn > 0 else highs[j] >= sl

    def hit_up(j: int, level: float) -> bool:
        """수익 방향 레벨 터치."""
        return highs[j] >= level if sgn > 0 else lows[j] <= level

    def r_of(price: float) -> float:
        return sgn * (price - entry) / dist

    out: dict = {}

    # ── base: 고정 SL/TP ──
    sl = entry - sgn * dist
    r = None
    for j in range(start, end):
        if hit_sl(j, sl):
            r = -1.0
            break
        if hit_up(j, tp):
            r = RR
            break
    if r is None:
        r = r_of(closes[end - 1])
    out["base"] = round(r - cost_r, 4)

    # ── time48h: base + 192캔들 시간손절 ──
    t_end = min(start + 192, end)
    r = None
    for j in range(start, t_end):
        if hit_sl(j, sl):
            r = -1.0
            break
        if hit_up(j, tp):
            r = RR
            break
    if r is None:
        r = r_of(closes[t_end - 1])
    out["time48h"] = round(r - cost_r, 4)

    # ── be1r: +1R 도달 다음 캔들부터 SL=본전 ──
    cur_sl = entry - sgn * dist
    armed = False
    r = None
    for j in range(start, end):
        if hit_sl(j, cur_sl):
            r = r_of(cur_sl)
            break
        if hit_up(j, tp):
            r = RR
            break
        if not armed and hit_up(j, arm):
            armed = True
            cur_sl = entry            # 다음 캔들부터 본전 스톱
    if r is None:
        r = r_of(closes[end - 1])
    out["be1r"] = round(r - cost_r, 4)

    # ── partial: +1.25R에 50% 익절, 잔여 SL=본전 + TP 2.5R ──
    cur_sl = entry - sgn * dist
    took_partial = False
    r = None
    for j in range(start, end):
        if hit_sl(j, cur_sl):
            r = (0.5 * 1.25 + 0.5 * r_of(cur_sl)) if took_partial else r_of(cur_sl)
            break
        if took_partial and hit_up(j, tp):
            r = 0.5 * 1.25 + 0.5 * RR
            break
        if not took_partial and hit_up(j, partial_tp):
            took_partial = True
            cur_sl = entry            # 다음 캔들부터 잔여분 본전 스톱
            if hit_up(j, tp):         # 같은 캔들에 TP까지 — 보수적으로 partial만 인정
                pass
    if r is None:
        last = r_of(closes[end - 1])
        r = (0.5 * 1.25 + 0.5 * last) if took_partial else last
    out["partial"] = round(r - cost_r, 4)

    # ── trail1r: +1R 도달 후 1.0ATR 트레일 (다음 캔들부터 갱신 적용) ──
    cur_sl = entry - sgn * dist
    armed = False
    best = entry
    r = None
    for j in range(start, end):
        if hit_sl(j, cur_sl):
            r = r_of(cur_sl)
            break
        # 극값 갱신
        ext = highs[j] if sgn > 0 else lows[j]
        if sgn * (ext - best) > 0:
            best = ext
        if not armed and hit_up(j, arm):
            armed = True
        if armed:
            new_sl = best - sgn * dist * 0.5   # 트레일 폭 = 1.0ATR = 0.5×dist(2ATR)
            if sgn * (new_sl - cur_sl) > 0:
                cur_sl = new_sl                 # 다음 캔들부터 적용
    if r is None:
        r = r_of(closes[end - 1])
    out["trail1r"] = round(r - cost_r, 4)

    return out


def run() -> dict:
    """전 신호에 변형 시뮬 적용 + 집계."""
    df = pd.read_csv(OUT_DIR / "signals_tagged.csv", parse_dates=["ts"])
    logger.info("신호 %d개 로드", len(df))

    cache: dict = {}
    rows: list[dict] = []
    skipped = 0
    for _, s in df.iterrows():
        sym = s["symbol"]
        if sym not in cache:
            try:
                d15 = pd.read_pickle(DATA_DIR / f"{_san(sym)}_15m.pkl")
                atr = (d15["high"] - d15["low"]).rolling(14).mean().to_numpy()
                cache[sym] = (d15, d15["high"].to_numpy(), d15["low"].to_numpy(),
                              d15["close"].to_numpy(), atr)
            except FileNotFoundError:
                cache[sym] = None
        if cache[sym] is None:
            skipped += 1
            continue
        d15, highs, lows, closes, atr_arr = cache[sym]
        i = d15.index.searchsorted(s["ts"])
        if i >= len(d15) or d15.index[i] != s["ts"]:
            skipped += 1
            continue
        atr = atr_arr[i]
        if not np.isfinite(atr) or atr <= 0:
            skipped += 1
            continue
        res = simulate_variants(
            float(s["entry"]), s["direction"], float(atr) * SL_MULT,
            highs, lows, closes, i + 1,
        )
        if not res:
            skipped += 1
            continue
        res["ts"] = s["ts"]
        res["score_raw"] = s["score_raw"]
        res["base_stored"] = s.get("r_m2_rr2.5")
        rows.append(res)

    out = pd.DataFrame(rows)
    logger.info("시뮬 완료: %d개 (스킵 %d)", len(out), skipped)

    def agg(sub: pd.DataFrame) -> dict:
        st: dict = {}
        for v in ["base", "be1r", "partial", "trail1r", "time48h"]:
            r = sub[v].dropna()
            if len(r) == 0:
                continue
            st[v] = {
                "n": int(len(r)),
                "winrate": round(float((r > 0).mean()), 3),
                "avg_r": round(float(r.mean()), 4),
                "total_r": round(float(r.sum()), 1),
                "worst_streak_r": round(float(r.rolling(10).sum().min()), 1) if len(r) > 10 else None,
            }
        return st

    mid = out["ts"].quantile(0.5)
    report = {
        "sanity_base_vs_stored": {
            "sim_base_avg": round(float(out["base"].mean()), 4),
            "stored_avg": round(float(pd.to_numeric(out["base_stored"], errors="coerce").mean()), 4),
            "note": "두 값이 비슷해야 시뮬 재현 정상",
        },
        "all": agg(out),
        "first_half": agg(out[out["ts"] <= mid]),
        "second_half": agg(out[out["ts"] > mid]),
        "score70": agg(out[out["score_raw"] >= 70]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "exit_variants.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장: %s", OUT_DIR / "exit_variants.json")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
