"""추매 사다리 구조 스윕 엔진 — AVGDOWN-2026-09-01 사전등록 구현.

명세: `docs/PREREGISTRATION_AVGDOWN_2026-09-01.md` (§3 격자 · §4 평가 프로토콜 ·
§5 지표 · §10.3 동결 상수). 본 파일은 **엔진**만 담당한다. 다중검정 보정
(White RC 고정 ω̂ · Romano–Wolf StepM · DSR)은 `lab/avgdown_verdict.py` 의 몫이다.

구조
----
1. `enumerate_trials()` — §3 격자를 파라미터에서 생성. 중복 제거(최대추매 0 이면
   추매간격 차원 붕괴 — §3.7) 후 **N = 1,248** 시행. 총계는 selftest 가 강제한다.
2. `simulate_sleeve()`  — 심볼 1개 × 타임프레임 1개의 **단독 슬리브** 봉 루프.
   시행 축(R)을 numpy 로 동시 처리한다. 자본 모델은 동결 `lab/bbadd_test.py` 의
   단독 슬리브 런(`bb.run(..., syms=(s,))`)과 **연산 순서까지 동형**이다 —
   selftest (a) 가 기준 시행의 수치 동일성을 강제한다.
3. `build_matrices()`   — 슬리브 일별 자본 → 마스터 일그리드 → 일수익률 행렬.
   1차 지표용 3심볼 균등 합산(결측 슬리브 = 현금 0%)과 심볼별 행렬을 기록한다.

실행 인과성 (위반 시 결과 폐기 — §4.4)
------------------------------------
* 신호 = 확정봉 `[i−1]` 종가·지표만 (`shift(1)`). 체결 = 봉 `[i]` 시가.
* 봉 `[i]` 의 동일봉 데이터 사용은 (a) 체결가 `open[i]`, (b) 손절 봉내 트리거
  판정용 `low[i]`, (c) 봉 마감 후 MTM·일손실 정지 판정용 `close[i]` 뿐이다.
  `close[i]` 는 어떤 체결가도 되지 않는다.
* 추매 트리거 = 직전 트랜치 **체결가 − 간격×ATR24[체결봉−1]**, 체결 시 동결·재귀.
* 손절 레벨 = **평단 − n×ATR24[마지막 체결봉−1]**, 체결 시 동결 (ATR NaN 이면
  기존 레벨 유지 — fail-closed). 봉내 스탑주문: `low <= 레벨` → `min(시가, 레벨)`
  체결 (갭 악화). ATR 익절 레벨도 동일 규약으로 동결·재동결된다.
* 같은 봉 우선순위: 익절(시가) > 추매(시가) > 손절(봉내). 청산 봉 재진입 금지.
* 워밍업 100봉 무주문 (bbadd_test 동결 관례). 결측·NaN 신호 = 무행동 (fail-closed).

주의: `--run` 은 **결과를 생성한다**. §8.1 에 따라 코드·문서 커밋 후 1회만 실행한다.
개발·검증은 `--selftest` (기준 시행 1개 + 합성 데이터) 만 사용한다.

실행:
  .venv/bin/python lab/avgdown_sweep.py --selftest     # 자가검증 (본 격자 미실행)
  .venv/bin/python lab/avgdown_sweep.py --run          # 본 격자 1회 (커밋 후에만)
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── 동결 모듈 로드 (읽기 전용 — 수정 없음) ────────────────────────────────
_SPEC = importlib.util.spec_from_file_location(
    "bbadd_test_frozen", str(ROOT / "lab" / "bbadd_test.py"))
bb = importlib.util.module_from_spec(_SPEC)
_cwd = os.getcwd()
os.chdir(ROOT)                       # bbadd_test 의 상대경로 규약 (import 시 무해)
_SPEC.loader.exec_module(bb)
os.chdir(_cwd)

# ── 동결 상수 (§10.3 — 변경 금지) ─────────────────────────────────────────
SEED: int = 20260901
N_PATHS: int = 1000
MEAN_BLOCK_DAYS: float = 5.0
N_TRIALS: int = 1248
WARMUP: int = 100                    # bbadd_test 동결 관례 (봉 수, 타임프레임 불변)
COST_IN, COST_OUT = bb.COST_IN, bb.COST_OUT          # 편도 8bp = 왕복 16bp
TRANCHE_FRAC: float = bb.TRANCHE_FRAC                # equity × 1/12
GROSS_CAP, DAILY_HALT = bb.GROSS_CAP, bb.DAILY_HALT
HEAT_CAP, HEAT_FRAC = bb.HEAT_CAP, bb.HEAT_FRAC
BB_N, BB_K, ATR_N = bb.BB_N, bb.BB_K, bb.ATR_N       # 20 / 2.0 / 24
RSI_N, RSI_TH = 14, 30.0
TREND_N = 200
SYMS = bb.SYMS                                       # ('BTC','ETH','SOL') 순서 고정

ENTRIES = ("E1", "E2")
FILTERS = (0, 1)
SPACINGS = (0.5, 1.0, 1.5, 2.0)
KMAXES = (0, 1, 2, 3)
TPS = ("MID", "A0.5", "A1.0", "A2.0")
TP_MULT = {"MID": float("nan"), "A0.5": 0.5, "A1.0": 1.0, "A2.0": 2.0}
STOPS = (None, 6.0, 10.0)
TFS = ("15m", "1h")

# 평가 창 (§4.2 — 전 시행 동일): 일수익률 r_d, d = 2021-01-06 … 2026-08-23 (UTC)
EVAL_SNAP0 = pd.Timestamp("2021-01-05", tz="utc")    # 첫 스냅샷 일자 (일말 자본)
EVAL_SNAP1 = pd.Timestamp("2026-08-23", tz="utc")    # 마지막 스냅샷 일자
N_DAYS: int = 2056

PATHS_15M = {s: f"lab/data/{s.lower()}_15m.parquet" for s in SYMS}
# §10.2 실측 해시 — 불일치 시 fail-closed (수집 완료 후 기입, 결과 조회 전 동결)
SHA_EXPECT_15M: dict[str, str] = {
    "lab/data/btc_15m.parquet":
        "861eb947ba59244e627538345f3f65f08dafef67f62f52d9b5c2bedb9b3c5164",
    "lab/data/eth_15m.parquet":
        "42a3067a8b5a5f58f2c1a136101255db8c0106d5c2c64c590db82aacae39b9ad",
    "lab/data/sol_15m.parquet":
        "08bfa833b35caddda6d5ada60dd5491ae269bf924368eecd8e271805c8e2c0b8",
}


# ── 시행 열거 (§3) ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Trial:
    """시행 1개 — §3 격자의 한 점 (중복 제거 후)."""

    entry: str            # 'E1' = BB(20,2) 하단 종가 이탈 / 'E2' = RSI(14)<30 종가
    filt: int             # 0 = 없음 / 1 = 확정봉 종가 > SMA200
    spacing: float | None  # 추매 간격 ×ATR24 (kmax=0 이면 None — §3.7 중복 제거)
    kmax: int             # 최대 추매 횟수 0..3 (트랜치 = kmax+1)
    tp: str               # 'MID' | 'A0.5' | 'A1.0' | 'A2.0'
    stop: float | None    # None | 6.0 | 10.0  (평단 − n×ATR)
    tf: str               # '15m' | '1h'

    def tid(self) -> str:
        """시행 ID — `AD|e=..|f=..|sp=..|k=..|tp=..|sl=..|tf` (§3.8)."""
        sp = "-" if self.spacing is None else f"{self.spacing:g}"
        sl = "-" if self.stop is None else f"{self.stop:g}"
        return (f"AD|e={self.entry}|f={self.filt}|sp={sp}|k={self.kmax}"
                f"|tp={self.tp}|sl={sl}|{self.tf}")


def enumerate_trials() -> list[Trial]:
    """§3 격자를 전수 열거한다 (중복 제거 규약 §3.7 적용, 순서 동결).

    Returns:
        길이 1,248 의 `Trial` 리스트 (tf → entry → filter → 사다리 → tp → stop 순).

    Raises:
        AssertionError: 총계·ID 유일성이 동결 상수와 다를 때.
    """
    ladders: list[tuple[float | None, int]] = [(None, 0)]
    ladders += [(sp, k) for sp in SPACINGS for k in KMAXES if k > 0]
    out = [Trial(e, f, sp, k, tp, sl, tf)
           for tf in TFS for e in ENTRIES for f in FILTERS
           for sp, k in ladders for tp in TPS for sl in STOPS]
    assert len(out) == N_TRIALS, f"시행 총계 {len(out)} != {N_TRIALS} (§3.7 위반)"
    tids = [t.tid() for t in out]
    assert len(set(tids)) == N_TRIALS, "시행 ID 유일성 위반"
    return out


# ── 유틸 ──────────────────────────────────────────────────────────────────
def sha256_file(path: str) -> str:
    """ROOT 기준 상대경로 파일의 SHA256."""
    h = hashlib.sha256()
    with open(ROOT / path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """`out[i] = a[i-k]` (앞쪽 NaN). k=0 이면 복사 — 룩어헤드 대조군 전용."""
    if k == 0:
        return a.astype(np.float64, copy=True)
    out = np.full_like(np.asarray(a, dtype=np.float64), np.nan)
    out[k:] = a[:-k]
    return out


def rsi_wilder(c: np.ndarray, n: int = RSI_N) -> np.ndarray:
    """Wilder RSI — `lab/sweep_engine.py` `Feat.rsi` 의 규약 복사 (동결 원본 무수정).

    `dn==0 → 100`, 무변동 → 50. selftest 가 원본과의 수치 동일성을 강제한다.

    Args:
        c: 종가 배열. n: 기간.

    Returns:
        RSI 배열 (초기 미형성 구간 NaN).
    """
    d = np.empty(len(c))
    d[0] = np.nan
    d[1:] = np.diff(c)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + up / dn)
    flat = (dn == 0)
    out = np.where(flat & (up == 0), 50.0, out)
    out = np.where(flat & (up > 0), 100.0, out)
    out[~np.isfinite(up) | ~np.isfinite(dn)] = np.nan
    return out


def load_data() -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """1h(동결) + 15m(본 문서 동결) OHLCV 와 일별 펀딩을 읽는다.

    Returns:
        (`{tf: {sym: df}}`, 일별 펀딩 DataFrame — bbadd_test.load 동형).
    """
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        d1h, fund = bb.load()
    finally:
        os.chdir(cwd)
    d15 = {s: pd.read_parquet(ROOT / PATHS_15M[s]) for s in SYMS}
    return {"1h": d1h, "15m": d15}, fund


# ── 시행 정적 배열 ────────────────────────────────────────────────────────
def trial_arrays(trials: list[Trial]) -> dict[str, np.ndarray]:
    """시행 리스트 → 시뮬레이션용 정적 배열 묶음 (R,)."""
    r = len(trials)
    return {
        "is_e1": np.array([t.entry == "E1" for t in trials]),
        "filt": np.array([t.filt == 1 for t in trials]),
        "spacing": np.array([np.nan if t.spacing is None else t.spacing for t in trials]),
        "ntr": np.array([t.kmax + 1 for t in trials], dtype=np.int64),
        "tp_mid": np.array([t.tp == "MID" for t in trials]),
        "tp_mult": np.array([TP_MULT[t.tp] for t in trials]),
        "stop_mult": np.array([np.nan if t.stop is None else t.stop for t in trials]),
        "R": r,
    }


# ── 슬리브 시뮬레이션 (자본 모델 = bbadd_test 단독 슬리브 동형) ───────────
def simulate_sleeve(df: pd.DataFrame, fund_sym: pd.Series, ta: dict[str, np.ndarray],
                    causal: bool = True) -> dict[str, np.ndarray | pd.DatetimeIndex]:
    """심볼 1개 슬리브를 시행 축(R) 벡터화로 굴린다.

    Args:
        df: OHLCV (UTC DatetimeIndex, 해당 심볼·타임프레임).
        fund_sym: 일별 펀딩률 합 (인덱스 = UTC 자정) — 일 시작 선차감, 롱 지불.
        ta: `trial_arrays()` 산출물.
        causal: False 면 같은 봉 신호 평가 (**룩어헤드 위반 대조군 전용**).

    Returns:
        dict — `final_eq (R,)`, `n_trades (R,)`, `n_wins (R,)`,
        `day_eq (R, D)` 일말 자본, `days (D,)` 일자 인덱스, `time_viol (R,)`.
    """
    r = ta["R"]
    o = df["open"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    sh = 1 if causal else 0

    cs = pd.Series(c)
    mid = cs.rolling(BB_N).mean().to_numpy()
    sd = cs.rolling(BB_N).std(ddof=0).to_numpy()      # 모표준편차 (출판 BB 관례)
    c1 = _shift(c, sh)                                # 확정봉 종가 (신호)
    m1 = _shift(mid, sh)
    lb1 = _shift(mid - BB_K * sd, sh)
    a1 = _shift(bb.atr(df).to_numpy(), sh)            # ATR24[i-1]
    rs1 = _shift(rsi_wilder(c), sh)
    tr1 = _shift(cs.rolling(TREND_N).mean().to_numpy(), sh)

    days_all = df.index.normalize()
    new_day = np.ones(n, dtype=bool)
    new_day[1:] = days_all[1:] != days_all[:-1]
    day_last = np.ones(n, dtype=bool)
    day_last[:-1] = new_day[1:]
    f_map = fund_sym.reindex(days_all).to_numpy(float)

    # 일말 스냅샷 대상 일자 (워밍업 이후 봉만 — bbadd rows 관례)
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
    time_viol = np.zeros(r, dtype=np.int64)
    day_mat = np.ones((r, len(snap_i)))

    tiny = 1e-300

    def close_mask(m: np.ndarray, x: float, i: int) -> None:
        """마스크 시행 전량 청산 — bbadd close_all 동형 (같은 산술 순서)."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        pnl = u * x - basis - (fees + u * x * COST_OUT)
        eq = np.where(m, eq + u * x - basis - u * x * COST_OUT, eq)
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

    def try_fill(m: np.ndarray, o_i: float, a1_i: float, pc: float, i: int) -> None:
        """트랜치 1개 체결 시도 — bbadd try_fill 동형 (단독 슬리브)."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        m = m & (eq > 0)
        if not (np.isfinite(o_i) and o_i > 0) or not m.any():
            return
        mk = pc if np.isfinite(pc) else np.nan
        gross = np.where(u > 0, u * np.where(np.isfinite(mk), mk,
                                             basis / np.maximum(u, tiny)), 0.0)
        m = m & (gross < GROSS_CAP * eq)
        unew = np.minimum(TRANCHE_FRAC * eq / o_i,
                          np.maximum(0.0, GROSS_CAP * eq - gross) / o_i)
        m = m & (unew > 0)
        heat = basis * HEAT_FRAC                     # Σ e·u × 5% = 명목 대리 (동결)
        m = m & (heat + unew * o_i * HEAT_FRAC <= HEAT_CAP * eq * (1 + 1e-9))
        if not m.any():
            return
        eq = np.where(m, eq - unew * o_i * COST_IN, eq)
        fees = np.where(m, fees + unew * o_i * COST_IN, fees)
        entry_i = np.where(m & (u == 0), i, entry_i)
        basis = np.where(m, basis + unew * o_i, basis)
        u = np.where(m, u + unew, u)
        k = np.where(m, k + 1, k)
        good = np.isfinite(a1_i) and a1_i > 0
        # 추매 임계: 체결가 − 간격×ATR[체결봉−1] 동결 (ATR 불량 → NaN, bbadd 동형)
        thr = np.where(m, (o_i - ta["spacing"] * a1_i) if good else np.nan, thr)
        if good:                                     # 손절·ATR익절: 재동결, 불량 시 유지
            avg = basis / np.maximum(u, tiny)
            stoplvl = np.where(m & np.isfinite(ta["stop_mult"]),
                               avg - ta["stop_mult"] * a1_i, stoplvl)
            tplvl = np.where(m & ~ta["tp_mid"], avg + ta["tp_mult"] * a1_i, tplvl)

    for i in range(n):
        if i < WARMUP:                               # 워밍업 — 주문 생성 금지
            continue
        pc = c[i - 1] if i > 0 else np.nan
        if new_day[i]:
            has = u > 0
            day_eq = eq + np.where(has & np.isfinite(pc), u * pc - basis, 0.0)
            halted[:] = False
            f = f_map[i]                             # 펀딩 (일 1회 선차감, 롱 지불)
            if np.isfinite(f) and np.isfinite(pc):
                eq = eq - np.where(has, f * u * pc, 0.0)
        o_i, l_i, c_i = o[i], l[i], c[i]
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
        # 1) 익절 (전량, 시가 체결, 같은 봉 재진입 금지)
        exit_m = np.zeros(r, dtype=bool)
        if sig_ok and np.isfinite(o_i):
            mid_sig = np.isfinite(m1_i) and c1_i >= m1_i
            exit_m = has0 & np.where(ta["tp_mid"], mid_sig,
                                     np.isfinite(tplvl) & (c1_i >= tplvl))
            if exit_m.any():
                close_mask(exit_m, o_i, i)
        # 2) 추매 (임계 = 체결 시 동결) — 일손실 정지 시 차단
        if sig_ok:
            add_m = (has0 & ~exit_m & (k < ta["ntr"]) & np.isfinite(thr)
                     & (c1_i <= thr) & ~halted)
            if add_m.any():
                try_fill(add_m, o_i, a1_i, pc, i)
        # 3) 신규 진입 (청산 봉 재진입 금지: has0 기준) — 추세필터 NaN fail-closed
        if sig_ok:
            e1_sig = np.isfinite(lb1_i) and c1_i < lb1_i
            e2_sig = np.isfinite(rs1_i) and rs1_i < RSI_TH
            base = np.where(ta["is_e1"], e1_sig, e2_sig)
            trend_ok = np.isfinite(tr1_i) and c1_i > tr1_i
            ent_m = (~has0) & base & (~ta["filt"] | trend_ok) & ~halted
            if ent_m.any():
                try_fill(ent_m, o_i, a1_i, pc, i)
        # 4) 재해손절 — 봉내 스탑주문 (진입·추매 직후 상태 반영, 갭 악화 체결)
        if np.isfinite(l_i) and np.isfinite(o_i):
            stop_m = (u > 0) & np.isfinite(stoplvl) & (l_i <= stoplvl)
            if stop_m.any():
                x = np.minimum(o_i, stoplvl)
                pnl = u * x - basis - (fees + u * x * COST_OUT)
                eq = np.where(stop_m, eq + u * x - basis - u * x * COST_OUT, eq)
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
        halted = halted | ((day_eq > 0) & (mtm / day_eq - 1 < DAILY_HALT))
        if day_pos[i] >= 0:
            day_mat[:, day_pos[i]] = mtm
    m = u > 0                                        # 기간 말 잔여 — 강제청산
    if m.any():
        px = c[~np.isnan(c)][-1]
        close_mask(m, px, n - 1)
    if len(snap_i):
        day_mat[:, -1] = eq                          # 최종점 = 청산 완료 자본
    return {"final_eq": eq, "n_trades": n_trades, "n_wins": n_wins,
            "day_eq": day_mat, "days": days, "time_viol": time_viol}


# ── 일수익률 행렬 조립 (§5.1) ─────────────────────────────────────────────
def master_days() -> pd.DatetimeIndex:
    """마스터 스냅샷 일그리드 — 2021-01-05 … 2026-08-23 (양끝 포함, 2,057개)."""
    return pd.date_range(EVAL_SNAP0, EVAL_SNAP1, freq="D")


def sleeve_returns(day_eq: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    """슬리브 일말 자본 → 마스터 그리드 일수익률 (R, 2056).

    슬리브 시작 전/워밍업 전 = 현금 1.0 (수익률 0). 스냅샷 없는 날 = 직전 값
    유지 (결측 봉 보간이 아니라 자본의 항등 지속 — fail-closed 와 무관).
    자본 ≤ 0 인 날의 다음 수익률은 0 (산술 정의 — 퇴화 방어, 사후 필터 아님).

    Args:
        day_eq: (R, D) 슬리브 일말 자본. days: (D,) 해당 일자.

    Returns:
        (R, N_DAYS) 일수익률.
    """
    grid = master_days()
    idx = np.searchsorted(days.values, grid.values, side="right") - 1
    snap = np.where(idx[None, :] >= 0, day_eq[:, np.maximum(idx, 0)], 1.0)
    prev, cur = snap[:, :-1], snap[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(prev > 0, cur / prev - 1.0, 0.0)
    assert ret.shape[1] == N_DAYS
    return ret


def run_grid(data: dict, fund: pd.DataFrame, trials: list[Trial],
             causal: bool = True, progress: bool = True) -> dict[str, np.ndarray]:
    """전 시행 × 3심볼 실행 → 심볼별·합산 일수익률 행렬과 요약 배열.

    Args:
        data: `{tf: {sym: df}}`. fund: 일별 펀딩. trials: 시행 리스트.
        causal: 룩어헤드 대조군 스위치 (본 실행 True).
        progress: 진행 로그.

    Returns:
        dict — `ret_combined (N,T)`, `ret_<SYM> (N,T)`, `n_trades (N,3)`,
        `n_wins (N,3)`, `final_eq (N,3)`, `time_viol (N,)`.
    """
    n = len(trials)
    per_sym = {s: np.zeros((n, N_DAYS)) for s in SYMS}
    n_trades = np.zeros((n, len(SYMS)), dtype=np.int64)
    n_wins = np.zeros((n, len(SYMS)), dtype=np.int64)
    final_eq = np.ones((n, len(SYMS)))
    time_viol = np.zeros(n, dtype=np.int64)
    for tf in TFS:
        rows = [i for i, t in enumerate(trials) if t.tf == tf]
        if not rows:
            continue
        ta = trial_arrays([trials[i] for i in rows])
        for j, s in enumerate(SYMS):
            t0 = time.time()
            res = simulate_sleeve(data[tf][s], fund[s], ta, causal=causal)
            per_sym[s][rows] = sleeve_returns(res["day_eq"], res["days"])
            n_trades[rows, j] = res["n_trades"]
            n_wins[rows, j] = res["n_wins"]
            final_eq[rows, j] = res["final_eq"]
            time_viol[rows] += res["time_viol"]
            if progress:
                logger.info("%s %s: %d 시행 %.1fs", tf, s, len(rows), time.time() - t0)
    combined = sum(per_sym[s] for s in SYMS) / len(SYMS)
    out = {"ret_combined": combined, "n_trades": n_trades, "n_wins": n_wins,
           "final_eq": final_eq, "time_viol": time_viol}
    for s in SYMS:
        out[f"ret_{s}"] = per_sym[s]
    return out


# ── 자가검증 (§9 — 위반 시 AssertionError) ────────────────────────────────
REF_TRIAL = Trial("E1", 0, 1.0, 3, "MID", None, "1h")   # = bbadd_v2_check 설정 A


def selftest() -> None:
    """셀프테스트 (a)~(e) — 전부 통과해야 한다. **본 격자는 실행하지 않는다.**

    (a) 기준 시행(1h·E1·무필터·간격1.0·추매3·BB중심선·무손절)이 동결
        `bbadd_test.run` 단독 슬리브와 거래 수·최종자본 수치 동일.
        (`bbadd_v2_check` 설정 A 는 자체 selftest 로 `bb.run` 과 수치 동일이
        증명돼 있으므로 `bb.run` 대조 = 설정 A 대조다. 자본 모델을 동일하게
        맞추기 위해 3심볼 공유자본이 아닌 **단독 슬리브 런**과 비교한다.)
    (b) 룩어헤드 대조군(같은 봉 신호)과 결과 상이.
    (c) 시행 열거 수 == N_TRIALS == 사전등록 문서의 N.
    (d) 데이터 6종 SHA256 == 동결 해시.
    (e) 시간 역행 없음 + 평가 창 일수 == N_DAYS.

    Raises:
        AssertionError: 어느 하나라도 위반 시.
    """
    print("--- selftest (AVGDOWN-2026-09-01) — 본 격자 미실행 ---")
    # (d) 데이터 동결
    for pth, want in {**bb.SHA_EXPECT, **SHA_EXPECT_15M}.items():
        got = sha256_file(pth)
        assert got == want, (pth, got, want)
    print("  [OK] (d) 데이터 6종 SHA256 = 사전등록 해시")
    # (c) 열거 수
    trials = enumerate_trials()
    dedup_full = len(ENTRIES) * len(FILTERS) * len(SPACINGS) * len(KMAXES) \
        * len(TPS) * len(STOPS) * len(TFS)
    print(f"  [OK] (c) 시행 열거 N = {len(trials)} (전수 {dedup_full} 에서 "
          f"최대추매 0 중복 제거 — §3.7)")
    # RSI 규약 동일성 (sweep_engine.Feat.rsi 복사본 검증)
    _sw_spec = importlib.util.spec_from_file_location(
        "sweep_engine_frozen", str(ROOT / "lab" / "sweep_engine.py"))
    sw = importlib.util.module_from_spec(_sw_spec)
    sys.modules["sweep_engine_frozen"] = sw          # dataclass 처리에 필요
    _sw_spec.loader.exec_module(sw)
    rng = np.random.default_rng(7)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    feat = sw.Feat(px, px, px, px, np.ones_like(px))
    ref = feat.rsi(RSI_N)
    got = rsi_wilder(px)
    both = np.isfinite(ref) & np.isfinite(got)
    assert np.isnan(ref[0]) == np.isnan(got[0])
    assert float(np.max(np.abs(ref[both] - got[both]))) == 0.0
    print("  [OK] RSI 복사본 == 동결 sweep_engine.Feat.rsi (수치 동일)")
    # (a) 기준 시행 == 동결 엔진 단독 슬리브. bbadd_test.run 은 syms 인자가 없어
    #     단독 슬리브를 직접 돌릴 수 없다 — bbadd_v2_check.run(기본값 = 설정 A,
    #     자체 selftest 1 이 bb.run 과 수치 동일성을 증명) 을 읽기 전용 로드해 쓴다.
    _v2_spec = importlib.util.spec_from_file_location(
        "bbadd_v2_check_frozen", str(ROOT / "lab" / "bbadd_v2_check.py"))
    v2 = importlib.util.module_from_spec(_v2_spec)
    sys.modules["bbadd_v2_check_frozen"] = v2
    _saved_cwd = os.getcwd()
    _v2_spec.loader.exec_module(v2)                  # 모듈이 os.chdir(ROOT) 수행
    os.chdir(_saved_cwd)
    data, fund = load_data()
    ta = trial_arrays([REF_TRIAL])
    mism = []
    for s in SYMS:
        ref_dfr, ref_tl, _ = v2.run({s: data["1h"][s]}, fund, syms=(s,))
        mine = simulate_sleeve(data["1h"][s], fund[s], ta)
        n_ref, n_mine = len(ref_tl), int(mine["n_trades"][0])
        eq_ref = float(ref_dfr.equity.iloc[-1])
        eq_mine = float(mine["final_eq"][0])
        assert n_ref == n_mine, (s, n_ref, n_mine)
        assert abs(eq_ref - eq_mine) < 1e-9, (s, eq_ref, eq_mine)
        assert int(mine["time_viol"][0]) == 0
        mism.append((s, n_mine, eq_mine))
        # (e) 일말 자본 vs 동결 엔진 일별 resample — 창 정의 동일성 확인
        ref_daily = ref_dfr.equity.resample("D").last().dropna()
        my_daily = pd.Series(mine["day_eq"][0], index=mine["days"])
        common = ref_daily.index.intersection(my_daily.index)
        assert len(common) > 1000
        assert float(np.max(np.abs(ref_daily[common] - my_daily[common]))) < 1e-9
    print("  [OK] (a) 기준 시행 == 동결 bb.run 단독 슬리브 — " +
          ", ".join(f"{s} {n}건 eq={e:.6f}" for s, n, e in mism) +
          " (거래수·최종자본·일별 자본 수치 동일)")
    # (b) 룩어헤드 대조군 — 같은 봉 신호 평가 시 결과 상이
    vio = simulate_sleeve(data["1h"]["BTC"], fund["BTC"], ta, causal=False)
    base = simulate_sleeve(data["1h"]["BTC"], fund["BTC"], ta, causal=True)
    assert abs(float(vio["final_eq"][0]) - float(base["final_eq"][0])) > 1e-9
    print(f"  [OK] (b) 위반본(같은 봉 신호) 최종자본 {float(vio['final_eq'][0]):.4f} "
          f"vs 교정본 {float(base['final_eq'][0]):.4f} — 상이")
    # (e) 평가 창
    grid = master_days()
    assert len(grid) == N_DAYS + 1, len(grid)
    ret = sleeve_returns(base["day_eq"], base["days"])
    assert ret.shape == (1, N_DAYS) and np.isfinite(ret).all()
    print(f"  [OK] (e) 시간 역행 0건 · 평가 창 스냅샷 {len(grid)}개 → "
          f"일수익률 T = {N_DAYS}")
    print("--- selftest 전부 통과 ---")


# ── 본 실행 ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """CLI — `--selftest` 는 자가검증만, `--run` 은 본 격자 1회 (§8.1)."""
    ap = argparse.ArgumentParser(description="AVGDOWN-2026-09-01 스윕 엔진")
    ap.add_argument("--selftest", action="store_true", help="자가검증만 (격자 미실행)")
    ap.add_argument("--run", action="store_true", help="본 격자 1회 실행 (커밋 후에만)")
    ap.add_argument("--outdir", default="logs", help="산출물 디렉터리")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        selftest()
        return 0
    if not args.run:
        print("아무 것도 하지 않음 — --selftest 또는 --run 지정")
        return 1
    selftest()                                       # 본 실행 전 자가검증 강제
    trials = enumerate_trials()
    data, fund = load_data()
    res = run_grid(data, fund, trials)
    assert int(res["time_viol"].sum()) == 0, "시간 역행 탐지 — 결과 폐기"
    grid = master_days()
    meta = {
        "spec": "AVGDOWN-2026-09-01", "seed": SEED, "n_trials": N_TRIALS,
        "n_days": N_DAYS, "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": {p: sha256_file(p) for p in
                        list(bb.SHA_EXPECT) + list(SHA_EXPECT_15M)},
        "numpy": np.__version__, "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    tids = np.array([t.tid() for t in trials], dtype=object)
    tmp = outdir / "avgdown_returns.npz.tmp"
    np.savez_compressed(
        tmp, daily_returns=res["ret_combined"], trial_ids=tids,
        snap_ts=np.array([str(t) for t in grid]), meta=json.dumps(meta),
        **{f"ret_{s}": res[f"ret_{s}"] for s in SYMS})
    os.replace(tmp, outdir / "avgdown_returns.npz")
    mu = res["ret_combined"].mean(axis=1)
    sdv = res["ret_combined"].std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sdv > 0, mu / sdv, 0.0) * np.sqrt(365.0)
    summ = pd.DataFrame({
        "trial_id": tids,
        "tf": [t.tf for t in trials], "entry": [t.entry for t in trials],
        "filt": [t.filt for t in trials],
        "spacing": [t.spacing if t.spacing is not None else "" for t in trials],
        "kmax": [t.kmax for t in trials], "tp": [t.tp for t in trials],
        "stop": [t.stop if t.stop is not None else "" for t in trials],
        "mean_daily_ret": mu, "sharpe_ann": sr,
        "n_trades": res["n_trades"].sum(axis=1),
        "win": np.where(res["n_trades"].sum(axis=1) > 0,
                        res["n_wins"].sum(axis=1)
                        / np.maximum(res["n_trades"].sum(axis=1), 1), np.nan),
        **{f"final_eq_{s}": res["final_eq"][:, j] for j, s in enumerate(SYMS)},
    })
    tmpc = outdir / "avgdown_summary.csv.tmp"
    summ.to_csv(tmpc, index=False)
    os.replace(tmpc, outdir / "avgdown_summary.csv")
    logger.info("기록 완료: %s (N=%d, T=%d)", outdir / "avgdown_returns.npz",
                N_TRIALS, N_DAYS)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
