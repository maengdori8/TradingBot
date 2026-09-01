"""수수료 민감도 재분석 엔진 — AVGDOWN-FEES-2026-09-01 사전등록 구현.

명세: `docs/PREREGISTRATION_AVGDOWN_FEES_2026-09-01.md`. 원 스윕
(AVGDOWN-2026-09-01, 왕복 16bp, 판정 실패 RC p=0.7632·생존 0)의 **동일한 1,248
시행 격자**를 비용 시나리오 3개로 재실행한다. 격자·파라미터·심볼·평가 창·인과성은
원 스윕과 완전 동일 — 바뀌는 것은 수수료율과 (시나리오 c 에 한해) 체결 모델뿐이다.

시나리오 (동결)
--------------
* **a** — 실측 테이커: 편도 5.5bp(왕복 11bp), 체결 모델 = 원 스윕 동일
  (다음 봉 시가 시장가).
* **b** — 메이커 상한: 편도 2bp(왕복 4bp), 체결 모델 동일. **시장가 체결 가정을
  유지한 채 메이커 수수료만 적용하므로 실현 불가능한 상한선**이다 (진단 전용).
* **c** — 보수적 메이커 체결: 편도 2bp + 지정가 체결 모델. 진입·추매는 신호봉
  종가에 지정가 → 다음 봉 **저가 < 지정가** 일 때만 체결(동가 미체결 — 엄격 관통),
  체결가 = min(다음 봉 시가, 지정가). 익절은 대칭: 다음 봉 **고가 > 지정가** 시
  체결, 체결가 = max(시가, 지정가). 미체결 진입·추매 신호는 소멸(이월 금지),
  미체결 익절은 다음 확정봉에서 재평가. 손절(S6/S10)은 원 스윕의 봉내 스탑주문
  모델을 그대로 유지한다.

동결 원본 취급
--------------
`lab/avgdown_sweep.py` 는 **읽기 전용 임포트**만 한다 (열거·데이터 로드·평가 창
상수 재사용). `simulate_sleeve` 의 수수료 파라미터화 + 지정가 체결 분기 복제본은
**이 파일 안에만** 존재한다(`simulate_sleeve_fees`). 편도 8bp·시장가로 놓으면 동결
엔진과 산술 순서까지 동일해 일수익률이 비트 동일해야 하며, selftest (i) 이
`logs/avgdown_returns.npz` 전체 행렬과의 1e-12 동일성을 강제한다.

주의: `--run` 은 결과를 생성한다 — 사전등록 문서 커밋 후 1회만. 개발·검증은
`--selftest` 만 사용한다 (본 격자의 신규 시나리오는 실행하지 않는다).

실행:
  .venv/bin/python lab/avgdown_fees.py --selftest   # 자가검증 (신규 격자 미실행)
  .venv/bin/python lab/avgdown_fees.py --run        # 시나리오 3개 1회 (커밋 후에만)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── 동결 엔진 로드 (읽기 전용 — 수정 없음) ────────────────────────────────
_SPEC = importlib.util.spec_from_file_location(
    "avgdown_sweep_frozen", str(ROOT / "lab" / "avgdown_sweep.py"))
asw = importlib.util.module_from_spec(_SPEC)
sys.modules["avgdown_sweep_frozen"] = asw
_SPEC.loader.exec_module(asw)

# ── 동결 상수 (원 스윕 §10.3 계승 + 본 트랙 시나리오) ─────────────────────
SEED = asw.SEED                                      # 20260901
N_TRIALS = asw.N_TRIALS                              # 1248
N_DAYS = asw.N_DAYS                                  # 2056
WARMUP = asw.WARMUP                                  # 100
SYMS = asw.SYMS
TFS = asw.TFS
COST_SIDE_FROZEN: float = asw.COST_IN                # 0.0008 — selftest (i) 대조용
assert COST_SIDE_FROZEN == asw.COST_OUT == 0.0008

# 시나리오 정의 (동결 — 변경·추가 금지, 사전등록 §3)
SCENARIOS: dict[str, dict] = {
    "a": {"cost_side": 0.00055, "fill_model": "market",
          "label": "실측 테이커 — 편도 5.5bp (왕복 11bp), 다음 봉 시가 시장가"},
    "b": {"cost_side": 0.0002, "fill_model": "market",
          "label": "메이커 상한 — 편도 2bp (왕복 4bp), 시장가 체결 가정 유지 = "
                   "메이커 실현 불가능한 상한선 (진단 전용)"},
    "c": {"cost_side": 0.0002, "fill_model": "limit",
          "label": "보수적 메이커 — 편도 2bp + 지정가 체결 (엄격 관통, 미체결 "
                   "신호 소멸, 손절은 원 스윕 봉내 스탑 모델 유지)"},
}


# ── 수수료 파라미터화 슬리브 (동결 simulate_sleeve 의 복제·수정본 — §3) ────
def simulate_sleeve_fees(
    df: pd.DataFrame, fund_sym: pd.Series, ta: dict[str, np.ndarray],
    cost_in: float, cost_out: float, fill_model: str = "market",
    causal: bool = True, fill_log: list | None = None,
) -> dict[str, np.ndarray | pd.DatetimeIndex]:
    """`avgdown_sweep.simulate_sleeve` 의 수수료 파라미터화 + 지정가 체결 복제본.

    `fill_model="market"` 이고 `cost_in == cost_out == 0.0008` 이면 동결 엔진과
    산술 순서까지 동일하다 — selftest (i) 이 전체 격자 일수익률의 1e-12 동일성을
    강제한다. `fill_model="limit"` 은 시나리오 (c) 전용:

    * 진입·추매: 지정가 = 확정봉 종가 `c1`. 체결 조건 `low[i] < c1` (엄격 —
      동가 미체결), 체결가 = `min(open[i], c1)` (갭 하락 개장 시 유리한 시가).
      미체결 시 신호 소멸 (이월 금지 — 다음 봉에서 새 확정봉 기준 재평가만).
    * 익절: 지정가 = `c1`. 체결 조건 `high[i] > c1` (엄격), 체결가 =
      `max(open[i], c1)`. 미체결 익절은 다음 확정봉에서 재평가. 익절 **신호**가
      뜬 봉에서는 미체결이어도 추매를 차단한다 (같은 봉 우선순위: 익절 > 추매 —
      원 스윕 규약 유지).
    * 손절: 원 스윕과 동일한 봉내 스탑주문 (`low <= 레벨` → `min(시가, 레벨)`).

    Args:
        df: OHLCV (UTC DatetimeIndex). fund_sym: 일별 펀딩률 합.
        ta: `avgdown_sweep.trial_arrays()` 산출물.
        cost_in: 진입 편도 수수료율. cost_out: 청산 편도 수수료율.
        fill_model: "market" | "limit".
        causal: False 면 같은 봉 신호 평가 (룩어헤드 위반 대조군 전용).
        fill_log: 지정 시 (종류, 봉 인덱스, 체결가) 튜플 append — 소형 검증 전용.

    Returns:
        dict — 동결 엔진 동일 키 + `n_fills (R,)` (트랜치 체결 수).
    """
    assert fill_model in ("market", "limit"), fill_model
    limit_fill = fill_model == "limit"
    r = ta["R"]
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    sh = 1 if causal else 0

    cs = pd.Series(c)
    mid = cs.rolling(asw.BB_N).mean().to_numpy()
    sd = cs.rolling(asw.BB_N).std(ddof=0).to_numpy()
    c1 = asw._shift(c, sh)
    m1 = asw._shift(mid, sh)
    lb1 = asw._shift(mid - asw.BB_K * sd, sh)
    a1 = asw._shift(asw.bb.atr(df).to_numpy(), sh)
    rs1 = asw._shift(asw.rsi_wilder(c), sh)
    tr1 = asw._shift(cs.rolling(asw.TREND_N).mean().to_numpy(), sh)

    days_all = df.index.normalize()
    new_day = np.ones(n, dtype=bool)
    new_day[1:] = days_all[1:] != days_all[:-1]
    day_last = np.ones(n, dtype=bool)
    day_last[:-1] = new_day[1:]
    f_map = fund_sym.reindex(days_all).to_numpy(float)

    snap_i = np.flatnonzero(day_last & (np.arange(n) >= WARMUP))
    days = days_all[snap_i]
    day_pos = np.full(n, -1, dtype=np.int64)
    day_pos[snap_i] = np.arange(len(snap_i))

    eq = np.ones(r)
    u = np.zeros(r)
    basis = np.zeros(r)
    fees = np.zeros(r)
    thr = np.full(r, np.nan)
    stoplvl = np.full(r, np.nan)
    tplvl = np.full(r, np.nan)
    k = np.zeros(r, dtype=np.int64)
    entry_i = np.full(r, -1, dtype=np.int64)
    halted = np.zeros(r, dtype=bool)
    day_eq = np.ones(r)
    n_trades = np.zeros(r, dtype=np.int64)
    n_wins = np.zeros(r, dtype=np.int64)
    n_fills = np.zeros(r, dtype=np.int64)
    time_viol = np.zeros(r, dtype=np.int64)
    day_mat = np.ones((r, len(snap_i)))

    tiny = 1e-300

    def close_mask(m: np.ndarray, x: float, i: int) -> None:
        """마스크 시행 전량 청산 — 동결 close_mask 동형 (같은 산술 순서)."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        pnl = u * x - basis - (fees + u * x * cost_out)
        eq = np.where(m, eq + u * x - basis - u * x * cost_out, eq)
        n_trades[m] += 1
        n_wins[m & (pnl > 0)] += 1
        time_viol[m & (entry_i > i)] += 1
        u = np.where(m, 0.0, u)
        basis = np.where(m, 0.0, basis)
        fees = np.where(m, 0.0, fees)
        thr = np.where(m, np.nan, thr)
        stoplvl = np.where(m, np.nan, stoplvl)
        tplvl = np.where(m, np.nan, tplvl)
        k = np.where(m, 0, k)
        entry_i = np.where(m, -1, entry_i)
        if fill_log is not None and m.any():
            fill_log.append(("exit", i, float(x)))

    def try_fill(m: np.ndarray, o_i: float, a1_i: float, pc: float, i: int) -> None:
        """트랜치 1개 체결 시도 — 동결 try_fill 동형. `o_i` = 체결가 (지정가
        모델에서는 min(시가, 지정가) 가 들어온다)."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        m = m & (eq > 0)
        if not (np.isfinite(o_i) and o_i > 0) or not m.any():
            return
        mk = pc if np.isfinite(pc) else np.nan
        gross = np.where(u > 0, u * np.where(np.isfinite(mk), mk,
                                             basis / np.maximum(u, tiny)), 0.0)
        m = m & (gross < asw.GROSS_CAP * eq)
        unew = np.minimum(asw.TRANCHE_FRAC * eq / o_i,
                          np.maximum(0.0, asw.GROSS_CAP * eq - gross) / o_i)
        m = m & (unew > 0)
        heat = basis * asw.HEAT_FRAC
        m = m & (heat + unew * o_i * asw.HEAT_FRAC
                 <= asw.HEAT_CAP * eq * (1 + 1e-9))
        if not m.any():
            return
        eq = np.where(m, eq - unew * o_i * cost_in, eq)
        fees = np.where(m, fees + unew * o_i * cost_in, fees)
        entry_i = np.where(m & (u == 0), i, entry_i)
        basis = np.where(m, basis + unew * o_i, basis)
        u = np.where(m, u + unew, u)
        k = np.where(m, k + 1, k)
        n_fills[m] += 1
        good = np.isfinite(a1_i) and a1_i > 0
        thr = np.where(m, (o_i - ta["spacing"] * a1_i) if good else np.nan, thr)
        if good:
            avg = basis / np.maximum(u, tiny)
            stoplvl = np.where(m & np.isfinite(ta["stop_mult"]),
                               avg - ta["stop_mult"] * a1_i, stoplvl)
            tplvl = np.where(m & ~ta["tp_mid"], avg + ta["tp_mult"] * a1_i, tplvl)
        if fill_log is not None:
            fill_log.append(("fill", i, float(o_i)))

    for i in range(n):
        if i < WARMUP:                               # 워밍업 — 주문 생성 금지
            continue
        pc = c[i - 1] if i > 0 else np.nan
        if new_day[i]:
            has = u > 0
            day_eq = eq + np.where(has & np.isfinite(pc), u * pc - basis, 0.0)
            halted[:] = False
            f = f_map[i]
            if np.isfinite(f) and np.isfinite(pc):
                eq = eq - np.where(has, f * u * pc, 0.0)
        o_i, h_i, l_i, c_i = o[i], hi[i], l[i], c[i]
        if np.isnan(c_i):                            # 데이터 결측/종료 — 강제청산
            m = u > 0
            if m.any() and np.isfinite(pc):
                close_mask(m, pc, i)
            if day_pos[i] >= 0:
                day_mat[:, day_pos[i]] = eq
            continue
        c1_i, m1_i, lb1_i, a1_i, rs1_i, tr1_i = (c1[i], m1[i], lb1[i], a1[i],
                                                 rs1[i], tr1[i])
        has0 = u > 0
        sig_ok = np.isfinite(c1_i)
        # 지정가 체결 조건 (시나리오 c 전용 — 스칼라, 지정가 = 확정봉 종가 c1)
        buy_ok = sell_ok = True                      # market: 시가 무조건 체결
        buy_px = sell_px = o_i
        if limit_fill:
            if sig_ok:
                buy_ok = np.isfinite(o_i) and np.isfinite(l_i) and (l_i < c1_i)
                buy_px = min(o_i, c1_i) if buy_ok else np.nan
                sell_ok = np.isfinite(o_i) and np.isfinite(h_i) and (h_i > c1_i)
                sell_px = max(o_i, c1_i) if sell_ok else np.nan
            else:
                buy_ok = sell_ok = False             # fail-closed
                buy_px = sell_px = np.nan
        # 1) 익절 — market: 시가 체결 / limit: 지정가 체결 (미체결 = 다음 봉 재평가)
        exit_sig = np.zeros(r, dtype=bool)
        if sig_ok and np.isfinite(o_i):
            mid_sig = np.isfinite(m1_i) and c1_i >= m1_i
            exit_sig = has0 & np.where(ta["tp_mid"], mid_sig,
                                       np.isfinite(tplvl) & (c1_i >= tplvl))
            if sell_ok and exit_sig.any():
                close_mask(exit_sig, sell_px, i)
        # 2) 추매 — 익절 신호 봉 차단 (우선순위 유지), limit 미체결 = 신호 소멸
        if sig_ok:
            add_m = (has0 & ~exit_sig & (k < ta["ntr"]) & np.isfinite(thr)
                     & (c1_i <= thr) & ~halted)
            if buy_ok and add_m.any():
                try_fill(add_m, buy_px, a1_i, pc, i)
        # 3) 신규 진입 — 청산 봉 재진입 금지 (has0 기준), limit 미체결 = 신호 소멸
        if sig_ok:
            e1_sig = np.isfinite(lb1_i) and c1_i < lb1_i
            e2_sig = np.isfinite(rs1_i) and rs1_i < asw.RSI_TH
            base = np.where(ta["is_e1"], e1_sig, e2_sig)
            trend_ok = np.isfinite(tr1_i) and c1_i > tr1_i
            ent_m = (~has0) & base & (~ta["filt"] | trend_ok) & ~halted
            if buy_ok and ent_m.any():
                try_fill(ent_m, buy_px, a1_i, pc, i)
        # 4) 재해손절 — 봉내 스탑주문 (원 스윕 모델 유지, 전 시나리오 공통)
        if np.isfinite(l_i) and np.isfinite(o_i):
            stop_m = (u > 0) & np.isfinite(stoplvl) & (l_i <= stoplvl)
            if stop_m.any():
                x = np.minimum(o_i, stoplvl)
                pnl = u * x - basis - (fees + u * x * cost_out)
                eq = np.where(stop_m, eq + u * x - basis - u * x * cost_out, eq)
                n_trades[stop_m] += 1
                n_wins[stop_m & (pnl > 0)] += 1
                time_viol[stop_m & (entry_i > i)] += 1
                u = np.where(stop_m, 0.0, u)
                basis = np.where(stop_m, 0.0, basis)
                fees = np.where(stop_m, 0.0, fees)
                thr = np.where(stop_m, np.nan, thr)
                stoplvl = np.where(stop_m, np.nan, stoplvl)
                tplvl = np.where(stop_m, np.nan, tplvl)
                k = np.where(stop_m, 0, k)
                entry_i = np.where(stop_m, -1, entry_i)
        # 5) MTM · 일손실 정지 · 일말 스냅샷
        mtm = eq + np.where(u > 0, u * c_i - basis, 0.0)
        halted = halted | ((day_eq > 0) & (mtm / day_eq - 1 < asw.DAILY_HALT))
        if day_pos[i] >= 0:
            day_mat[:, day_pos[i]] = mtm
    m = u > 0                                        # 기간 말 잔여 — 강제청산
    if m.any():
        px = c[~np.isnan(c)][-1]
        close_mask(m, px, n - 1)
    if len(snap_i):
        day_mat[:, -1] = eq
    return {"final_eq": eq, "n_trades": n_trades, "n_wins": n_wins,
            "n_fills": n_fills, "day_eq": day_mat, "days": days,
            "time_viol": time_viol}


def run_grid_fees(data: dict, fund: pd.DataFrame, trials: list,
                  cost_side: float, fill_model: str, causal: bool = True,
                  progress: bool = True) -> dict[str, np.ndarray]:
    """전 시행 × 3심볼 실행 — `avgdown_sweep.run_grid` 동형, 슬리브만 파라미터화.

    Args:
        data: `{tf: {sym: df}}`. fund: 일별 펀딩. trials: 시행 리스트.
        cost_side: 편도 수수료율 (진입 = 청산). fill_model: "market" | "limit".
        causal: 룩어헤드 대조군 스위치. progress: 진행 로그.

    Returns:
        dict — 동결 run_grid 동일 키 + `n_fills (N, 3)`.
    """
    n = len(trials)
    per_sym = {s: np.zeros((n, N_DAYS)) for s in SYMS}
    n_trades = np.zeros((n, len(SYMS)), dtype=np.int64)
    n_wins = np.zeros((n, len(SYMS)), dtype=np.int64)
    n_fills = np.zeros((n, len(SYMS)), dtype=np.int64)
    final_eq = np.ones((n, len(SYMS)))
    time_viol = np.zeros(n, dtype=np.int64)
    for tf in TFS:
        rows = [i for i, t in enumerate(trials) if t.tf == tf]
        if not rows:
            continue
        ta = asw.trial_arrays([trials[i] for i in rows])
        for j, s in enumerate(SYMS):
            t0 = time.time()
            res = simulate_sleeve_fees(data[tf][s], fund[s], ta,
                                       cost_in=cost_side, cost_out=cost_side,
                                       fill_model=fill_model, causal=causal)
            per_sym[s][rows] = asw.sleeve_returns(res["day_eq"], res["days"])
            n_trades[rows, j] = res["n_trades"]
            n_wins[rows, j] = res["n_wins"]
            n_fills[rows, j] = res["n_fills"]
            final_eq[rows, j] = res["final_eq"]
            time_viol[rows] += res["time_viol"]
            if progress:
                logger.info("%s %s: %d 시행 %.1fs", tf, s, len(rows),
                            time.time() - t0)
    combined = sum(per_sym[s] for s in SYMS) / len(SYMS)
    out = {"ret_combined": combined, "n_trades": n_trades, "n_wins": n_wins,
           "n_fills": n_fills, "final_eq": final_eq, "time_viol": time_viol}
    for s in SYMS:
        out[f"ret_{s}"] = per_sym[s]
    return out


# ── 자가검증 (사전등록 §9 — 위반 시 AssertionError) ───────────────────────
def _flat_df(n: int, bars: dict[int, tuple[float, float, float, float]]
             ) -> pd.DataFrame:
    """평탄 100 시계열 + 지정 봉 오버라이드 (o, h, l, c) — 합성 검증 전용."""
    o = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    c = np.full(n, 100.0)
    for i, (oo, hh, ll, cc) in bars.items():
        o[i], h[i], l[i], c[i] = oo, hh, ll, cc
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.ones(n)}, index=idx)


def _synthetic_walk(seed: int, n: int = 4000) -> pd.DataFrame:
    """합성 랜덤워크 OHLCV — (iii)/(iv) 구조 검증 전용."""
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1] * (1 + rng.normal(0, 1e-3, n - 1))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.ones(n)}, index=idx)


ZERO_FUND = pd.Series(dtype=float)
REF_TRIAL = asw.REF_TRIAL                            # 1h·E1·f0·sp1·k3·MID·sl-


def _selftest_limit_arith() -> None:
    """(ii) 지정가 체결 모델 수기 산술 3케이스 — 관통 / 동가 미체결 / 갭 유리.

    합성: 100 평탄 130봉, 봉 130 종가 90 (신호봉 — E1: 90 < BB 하단 100).
    지정가 = 90. 봉 131 이 체결 시험 봉이다. kmax=0 시행이라 추매 없음.
    """
    j = 130
    ta = asw.trial_arrays([asw.Trial("E1", 0, None, 0, "MID", None, "1h")])
    sig = {j: (100.0, 100.0, 90.0, 90.0)}

    # 케이스 1 (관통 체결): low 89 < 90 → 체결가 = min(95, 90) = 90.
    #   이후 봉 132 종가 100 → 봉 133 익절 신호 (c1=100 >= SMA20≈99.25),
    #   지정가 100. 봉 133 고가 100 == 100 → 미체결 (엄격 >). 봉 134 종가 100
    #   → 봉 135 재평가, 고가 101 > 100 → 체결가 = max(99.5, 100) = 100.
    log1: list = []
    df1 = _flat_df(140, {**sig,
                         131: (95.0, 96.0, 89.0, 95.0),
                         132: (100.0, 100.0, 96.0, 100.0),
                         133: (99.0, 100.0, 98.0, 100.0),
                         134: (99.5, 100.0, 99.0, 100.0),
                         135: (99.5, 101.0, 99.0, 100.0)})
    r1 = simulate_sleeve_fees(df1, ZERO_FUND, ta, 0.0002, 0.0002,
                              fill_model="limit", fill_log=log1)
    fills = [e for e in log1 if e[0] == "fill"]
    exits = [e for e in log1 if e[0] == "exit"]
    assert fills == [("fill", 131, 90.0)], fills     # 관통 → 지정가 90 체결
    assert exits == [("exit", 135, 100.0)], exits    # 고가==지정가 봉 133 미체결
    u = (1.0 / 12.0) / 90.0                          # 트랜치 = eq/12 ÷ 체결가 90
    expect = (1.0 - u * 90.0 * 0.0002) \
        + (u * 100.0 - u * 90.0 - u * 100.0 * 0.0002)
    assert abs(float(r1["final_eq"][0]) - expect) < 1e-15, \
        (float(r1["final_eq"][0]), expect)
    assert int(r1["n_fills"][0]) == 1 and int(r1["n_trades"][0]) == 1
    print(f"  [OK] (ii-1) 관통 체결 — 진입 90.0 · 동가 익절 미체결 · "
          f"재평가 익절 100.0 · 최종자본 {float(r1['final_eq'][0]):.8f} 수기 일치")

    # 케이스 2 (동가 미체결): low == 90 → 미체결, 신호 소멸. 거래 0, 자본 1.0.
    log2: list = []
    df2 = _flat_df(140, {**sig, 131: (95.0, 96.0, 90.0, 100.0)})
    r2 = simulate_sleeve_fees(df2, ZERO_FUND, ta, 0.0002, 0.0002,
                              fill_model="limit", fill_log=log2)
    assert log2 == [], log2
    assert int(r2["n_fills"][0]) == 0 and int(r2["n_trades"][0]) == 0
    assert float(r2["final_eq"][0]) == 1.0
    print("  [OK] (ii-2) 동가 미체결 — low==지정가 → 체결 0 · 자본 1.0 (엄격 관통)")

    # 케이스 3 (갭 유리): 시가 88 < 지정가 90 → 체결가 = min(88, 90) = 88.
    #   봉 132 종가 100 → 봉 133 익절 재평가: 시가 103 갭 상승, 고가 104 > 100
    #   → 체결가 = max(103, 100) = 103 (갭 유리 대칭).
    log3: list = []
    df3 = _flat_df(140, {**sig,
                         131: (88.0, 89.0, 87.0, 95.0),
                         132: (100.0, 100.0, 95.0, 100.0),
                         133: (103.0, 104.0, 102.0, 103.0)})
    r3 = simulate_sleeve_fees(df3, ZERO_FUND, ta, 0.0002, 0.0002,
                              fill_model="limit", fill_log=log3)
    fills = [e for e in log3 if e[0] == "fill"]
    exits = [e for e in log3 if e[0] == "exit"]
    assert fills == [("fill", 131, 88.0)], fills
    assert exits == [("exit", 133, 103.0)], exits
    u3 = (1.0 / 12.0) / 88.0
    expect3 = (1.0 - u3 * 88.0 * 0.0002) \
        + (u3 * 103.0 - u3 * 88.0 - u3 * 103.0 * 0.0002)
    assert abs(float(r3["final_eq"][0]) - expect3) < 1e-15
    print(f"  [OK] (ii-3) 갭 유리 체결가 — 진입 88.0 (시가<지정가) · 익절 103.0 "
          f"(시가>지정가) · 최종자본 {float(r3['final_eq'][0]):.8f} 수기 일치")


def selftest() -> None:
    """사전등록 §9 셀프테스트 (i)~(v) — 전부 통과해야 한다.

    **신규 시나리오 격자(a/b/c)는 실행하지 않는다.** 실데이터에 닿는 것은
    (i) 편도 8bp 재현(이미 공개된 원 스윕 결과의 재계산 — 신규 정보 0)과
    (iii)/(iv) 의 기준 시행 1개(§2.1 에서 결과가 공개된 시행, 체결 수·상이성만
    확인)뿐이다.

    Raises:
        AssertionError: 어느 하나라도 위반 시.
    """
    print("--- selftest (AVGDOWN-FEES-2026-09-01) — 신규 시나리오 격자 미실행 ---")
    # (v) 데이터 6종 SHA256 = 원 스윕 동결 해시
    for pth, want in {**asw.bb.SHA_EXPECT, **asw.SHA_EXPECT_15M}.items():
        got = asw.sha256_file(pth)
        assert got == want, (pth, got, want)
    print("  [OK] (v) 데이터 6종 SHA256 = 원 스윕 동결 해시")

    # (ii) 지정가 모델 수기 산술 3케이스
    _selftest_limit_arith()

    # (iv-합성) 룩어헤드 대조군 상이 — market·limit 양쪽
    df_syn = _synthetic_walk(11)
    ta1 = asw.trial_arrays([REF_TRIAL])
    for fm in ("market", "limit"):
        a = simulate_sleeve_fees(df_syn, ZERO_FUND, ta1, 0.0002, 0.0002,
                                 fill_model=fm, causal=True)
        b = simulate_sleeve_fees(df_syn, ZERO_FUND, ta1, 0.0002, 0.0002,
                                 fill_model=fm, causal=False)
        assert abs(float(a["final_eq"][0]) - float(b["final_eq"][0])) > 1e-9, fm
    print("  [OK] (iv-합성) 룩어헤드 대조군(같은 봉 신호) 상이 — market·limit")

    # (iii-합성) 지정가 체결 수 <= 시장가 체결 수 — 다양 시행 24개 × 합성 3종.
    # 공시: per-trial "<=" 는 불변식이 아니다 — 미체결로 상태가 분기하면 지정가
    # 쪽이 뒤늦게 다른 진입 기회를 잡아 체결이 더 많아질 수 있다 (합성 seed 11
    # 에서 24개 중 2개 관측: 113>110, 41>39). 따라서 동결 단언은 (1) 시행 집계
    # 합, (2) 실측 기준 시행 per-trial 이다. 위반 시행 수는 보고만 한다.
    sub = [t for t in asw.enumerate_trials() if t.tf == "1h"][::26]  # 24개 표집
    ta_sub = asw.trial_arrays(sub)
    for seed in (11, 23, 47):
        d = _synthetic_walk(seed)
        mk = simulate_sleeve_fees(d, ZERO_FUND, ta_sub, 0.00055, 0.00055,
                                  fill_model="market")
        lm = simulate_sleeve_fees(d, ZERO_FUND, ta_sub, 0.0002, 0.0002,
                                  fill_model="limit")
        s_lm, s_mk = int(lm["n_fills"].sum()), int(mk["n_fills"].sum())
        n_vio = int((lm["n_fills"] > mk["n_fills"]).sum())
        assert s_lm <= s_mk, (seed, s_lm, s_mk)
        print(f"  [OK] (iii-합성 seed {seed}) 지정가 체결 합 {s_lm} <= 시장가 "
              f"{s_mk} ({len(sub)}시행, per-trial 역전 {n_vio}개 — 경로 분기 공시)")

    # 실데이터 로드 (이후 항목 공용)
    data, fund = asw.load_data()

    # (iii-실측)/(iv-실측) 기준 시행 1개 (결과 공개 시행 — 체결 수·상이성만)
    res_a = simulate_sleeve_fees(data["1h"]["BTC"], fund["BTC"], ta1,
                                 0.00055, 0.00055, fill_model="market")
    res_c = simulate_sleeve_fees(data["1h"]["BTC"], fund["BTC"], ta1,
                                 0.0002, 0.0002, fill_model="limit")
    assert int(res_c["n_fills"][0]) <= int(res_a["n_fills"][0]), \
        (int(res_c["n_fills"][0]), int(res_a["n_fills"][0]))
    print(f"  [OK] (iii-실측) 기준 시행 BTC 1h — 지정가 체결 "
          f"{int(res_c['n_fills'][0])}건 <= 시장가 {int(res_a['n_fills'][0])}건")
    vio = simulate_sleeve_fees(data["1h"]["BTC"], fund["BTC"], ta1,
                               COST_SIDE_FROZEN, COST_SIDE_FROZEN,
                               fill_model="market", causal=False)
    base = simulate_sleeve_fees(data["1h"]["BTC"], fund["BTC"], ta1,
                                COST_SIDE_FROZEN, COST_SIDE_FROZEN,
                                fill_model="market", causal=True)
    assert abs(float(vio["final_eq"][0]) - float(base["final_eq"][0])) > 1e-9
    print(f"  [OK] (iv-실측) 위반본 {float(vio['final_eq'][0]):.4f} vs 교정본 "
          f"{float(base['final_eq'][0]):.4f} — 상이")

    # (i) 편도 8bp·시장가 = 동결 스윕 산출물과 전 격자 일수익률 1e-12 동일
    npz_path = ROOT / "logs" / "avgdown_returns.npz"
    assert npz_path.exists(), "logs/avgdown_returns.npz 없음 — 원 스윕 산출물 필요"
    z = np.load(npz_path, allow_pickle=True)
    trials = asw.enumerate_trials()
    tids = np.array([t.tid() for t in trials], dtype=object)
    assert (np.asarray(z["trial_ids"], dtype=object) == tids).all(), \
        "시행 순서 불일치"
    rep = run_grid_fees(data, fund, trials, cost_side=COST_SIDE_FROZEN,
                        fill_model="market", progress=False)
    assert int(rep["time_viol"].sum()) == 0
    errs = {}
    errs["combined"] = float(np.max(np.abs(
        rep["ret_combined"] - np.asarray(z["daily_returns"], dtype=np.float64))))
    for s in SYMS:
        errs[s] = float(np.max(np.abs(
            rep[f"ret_{s}"] - np.asarray(z[f"ret_{s}"], dtype=np.float64))))
    for k_, e in errs.items():
        assert e <= 1e-12, (k_, e)
    print("  [OK] (i) 편도 8bp·시장가 재현 = 동결 avgdown_returns.npz — 최대 오차 "
          + ", ".join(f"{k_}={e:.1e}" for k_, e in errs.items())
          + " (파라미터화가 기존 경로를 바꾸지 않음)")
    print("--- selftest 전부 통과 ---")


# ── 본 실행 (시나리오 3개 — 사전등록 커밋 후 1회만) ───────────────────────
def run_scenarios(outdir: Path) -> None:
    """시나리오 a/b/c 격자 실행 + 산출물 원자적 기록.

    Args:
        outdir: 산출물 디렉터리.

    Raises:
        AssertionError: 시간 역행 탐지 시 (결과 폐기).
    """
    trials = asw.enumerate_trials()
    data, fund = asw.load_data()
    grid = asw.master_days()
    tids = np.array([t.tid() for t in trials], dtype=object)
    outdir.mkdir(parents=True, exist_ok=True)
    for key, sc in SCENARIOS.items():
        t0 = time.time()
        res = run_grid_fees(data, fund, trials, cost_side=sc["cost_side"],
                            fill_model=sc["fill_model"])
        assert int(res["time_viol"].sum()) == 0, f"시간 역행 탐지({key}) — 폐기"
        meta = {
            "spec": "AVGDOWN-FEES-2026-09-01", "scenario": key,
            "label": sc["label"], "cost_side": sc["cost_side"],
            "fill_model": sc["fill_model"], "seed": SEED,
            "n_trials": N_TRIALS, "n_days": N_DAYS,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_engine_sha256": hashlib.sha256(
                (ROOT / "lab" / "avgdown_sweep.py").read_bytes()).hexdigest(),
            "data_sha256": {p: asw.sha256_file(p) for p in
                            list(asw.bb.SHA_EXPECT) + list(asw.SHA_EXPECT_15M)},
            "numpy": np.__version__, "pandas": pd.__version__,
            "python": sys.version.split()[0],
        }
        tmp = outdir / f"avgdown_fees_{key}.npz.tmp"
        np.savez_compressed(
            tmp, daily_returns=res["ret_combined"], trial_ids=tids,
            snap_ts=np.array([str(t) for t in grid]), meta=json.dumps(meta),
            n_fills=res["n_fills"], n_trades=res["n_trades"],
            **{f"ret_{s}": res[f"ret_{s}"] for s in SYMS})
        os.replace(tmp, outdir / f"avgdown_fees_{key}.npz")
        mu = res["ret_combined"].mean(axis=1)
        sdv = res["ret_combined"].std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sdv > 0, mu / sdv, 0.0) * np.sqrt(365.0)
        summ = pd.DataFrame({
            "trial_id": tids, "mean_daily_ret": mu, "sharpe_ann": sr,
            "n_trades": res["n_trades"].sum(axis=1),
            "n_fills": res["n_fills"].sum(axis=1),
            **{f"final_eq_{s}": res["final_eq"][:, j]
               for j, s in enumerate(SYMS)},
        })
        tmpc = outdir / f"avgdown_fees_summary_{key}.csv.tmp"
        summ.to_csv(tmpc, index=False)
        os.replace(tmpc, outdir / f"avgdown_fees_summary_{key}.csv")
        logger.info("시나리오 %s 완료 (%.1fs): %s", key, time.time() - t0,
                    outdir / f"avgdown_fees_{key}.npz")


def main(argv: list[str] | None = None) -> int:
    """CLI — `--selftest` 는 자가검증만, `--run` 은 시나리오 3개 1회 (§8.1)."""
    ap = argparse.ArgumentParser(description="AVGDOWN-FEES-2026-09-01 엔진")
    ap.add_argument("--selftest", action="store_true", help="자가검증만 (격자 미실행)")
    ap.add_argument("--run", action="store_true", help="시나리오 3개 1회 (커밋 후에만)")
    ap.add_argument("--outdir", default="logs", help="산출물 디렉터리")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        selftest()
        return 0
    if not args.run:
        print("아무 것도 하지 않음 — --selftest 또는 --run 지정")
        return 1
    selftest()                                       # 본 실행 전 자가검증 강제
    run_scenarios(ROOT / args.outdir)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
