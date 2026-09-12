"""리스크 통제 스윕 엔진 — RISK-2026-09-04 사전등록 구현.

명세: `docs/PREREGISTRATION_RISK_2026-09-04.md`. 본 파일은 **엔진**만 담당한다.
다중검정 보정과 shrink 곡선 판정은 `lab/risk_verdict.py` 의 몫이다.

왜 리스크인가
------------
CONF-TUNE-2026-09-04 (`docs/TRADINGVIEW_CONFLUENCE_TUNING.md`) 에서 이 구조의
수익 예측 시도는 실패했다 — 14,976 시행 RC `p=0.7922`, 최고 시행이 반증 대조군,
등급의 정보량은 변동성 통제 후 `+0.004 (p=0.95)`. 수익은 예측되지 않는다.

**낙폭은 예측 없이도 통제된다.** 사이즈를 3배 키우면 낙폭이 3.2배가 됐다는 관측은
반대 방향으로도 정확히 성립한다. 그래서 이번 질문은 하나다:

> 리스크 통제 규칙이 **"그냥 사이즈를 줄인 것"보다 나은가?**

반증 대조 (이것이 본 검정의 전부다)
---------------------------------
`shr*` — 아무 규칙 없이 트랜치 명목만 상수배 축소한 팔. 어떤 리스크 규칙이든
같은 MDD 로 맞췄을 때 이 축소 대조보다 수익이 높아야 "작동한다"고 말할 수 있다.
낙폭만 줄이는 것은 리스크 관리가 아니라 그냥 작게 하는 것이다.

`plcvt` — 변동성 타깃과 **동일한 배수 분포**를 봉 인덱스 해시로 무작위 재배정한
위약. 배수 분포가 같으므로, 위약이 실제 vt 와 비슷하면 개선의 원인은 "변동성에
맞춘 것"이 아니라 "배수를 흔든 것"이다.

기반 구조
--------
`lab/avgdown_sweep.py` (AVGDOWN-2026-09-01) 의 실행 규약을 **연산 순서까지 계승**한다.
원 엔진을 수정하지 않고 읽기 전용으로 임포트하며, 중립 설정(`risk='base'`)에서
원 엔진의 `sl=None` 시행과 **비트 단위 동일** 결과를 내야 한다 — `--selftest` 가
이를 강제한다.

축의 재배치 (AVGDOWN 대비 유일한 구조 변경)
-----------------------------------------
손절 배수는 AVGDOWN 에서 구조 축이었으나, 본 명세에서는 **리스크 축**으로 옮긴다.
구조 축 = 진입 2 × 추세필터 2 × 사다리 13 × 익절 4 = **208**.
리스크 축 = **18**. 타임프레임 2. → **7,488 시행**.

실행 인과성 (위반 시 결과 폐기 — AVGDOWN §4.4 계승)
-------------------------------------------------
- 모든 신호·리스크 상태는 확정봉 `[i-1]` 이하 값만 쓴다. 체결은 봉 `[i]` 시가.
- 봉 `[i]` 의 동일봉 사용은 (a) 체결가 `open[i]`, (b) 손절 봉내 트리거 `low[i]`,
  (c) 봉 마감 후 MTM·일손실 판정 `close[i]` 뿐이다.
- 사이즈 배수는 **첫 트랜치 진입 봉에서 동결**되고 그 거래의 추매 전부에 적용된다.
- 낙폭 스로틀 상태는 **일 시작 시점의 전일 말 자본과 전일까지의 peak** 로만 판정한다.
  당일 장중 자본으로 당일 진입을 막지 않는다 (동일봉 정보 사용 금지).
- 시간 청산은 `i - 첫 트랜치 봉 >= H` 인 봉의 **시가**에 체결한다 (익절과 같은 규약).

공개된 한계 (AVGDOWN 계승 — 본 명세가 고치지 않는다)
--------------------------------------------------
- 데이터 종료 시 잔여 포지션은 **마지막 유효 종가로 강제청산**한다. 이는 동일봉
  종가를 체결가로 쓰는 유일한 지점이며 동결 원 엔진의 규약이다. 표본 끝 1건에만
  영향을 주지만, 리스크 지표(특히 최악 거래)에서는 그대로 계상된다.
- 강제청산·마진콜·부분체결은 모델하지 않는다. 실현 MDD 는 계좌 낙폭 보장이 아니다.
- heat 캡(6%)·gross 캡(10x)·일손실 정지(−5%)는 동결 위험모델 그대로다.

실행:
  python3 -m lab.risk_tune --selftest   # 동치성·인과성 자가검증 (격자 미실행)
  python3 -m lab.risk_tune --run        # 본 격자 1회 (커밋·태그 후에만)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lab import avgdown_sweep as ads
from lab import confluence_tune as ct

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "logs"

# ── 동결 상수 (AVGDOWN 계승 — 변경 금지) ──────────────────────────────────
SYMS = ads.SYMS
COST_IN, COST_OUT = ads.COST_IN, ads.COST_OUT
TRANCHE_FRAC = ads.TRANCHE_FRAC
GROSS_CAP, DAILY_HALT = ads.GROSS_CAP, ads.DAILY_HALT
HEAT_CAP, HEAT_FRAC = ads.HEAT_CAP, ads.HEAT_FRAC
BB_N, BB_K, ATR_N = ads.BB_N, ads.BB_K, ads.ATR_N
RSI_N, RSI_TH, TREND_N = ads.RSI_N, ads.RSI_TH, ads.TREND_N
WARMUP, N_DAYS = ads.WARMUP, ads.N_DAYS

# ── 본 명세 동결 상수 ─────────────────────────────────────────────────────
SPEC = "RISK-2026-09-04"
PLACEBO_SEED = 20260904
VT_LO, VT_HI = 0.25, 4.00        # 변동성 타깃 배수 클립 (기준 대비)
VT_LO_T, VT_HI_T = 0.50, 2.00    # 타이트 클립 변형
NO_CAP = 10 ** 9                 # 시간 청산 없음 (봉 수)

ENTRIES, FILTERS = ads.ENTRIES, ads.FILTERS
SPACINGS, KMAXES = ads.SPACINGS, ads.KMAXES
TPS, TP_MULT = ads.TPS, ads.TP_MULT
TFS = ("15m", "1h")
BARS_PER_HOUR = {"15m": 4, "1h": 1}


@dataclass(frozen=True)
class RiskRule:
    """리스크 통제 규칙 한 개.

    Attributes:
        stop: 평단 − n×ATR 손절 배수 (None = 무손절, 동결 기본).
        size: 트랜치 명목 상수배 (축소 대조용).
        hours: 최대 보유 시간 (None = 무제한).
        dd_half: 자본 낙폭이 이 값을 넘으면 사이즈 절반 (None = 사용 안 함).
        dd_stop: 자본 낙폭이 이 값을 넘으면 신규 진입 정지 (None = 사용 안 함).
        vt: 변동성 타깃 클립 `(lo, hi)` (None = 사용 안 함).
        placebo: True 면 vt 배수를 봉 해시로 무작위 재배정 (위약 대조).
        note: 한 줄 설명.
    """

    stop: float | None = None
    size: float = 1.0
    hours: int | None = None
    dd_half: float | None = None
    dd_stop: float | None = None
    vt: tuple[float, float] | None = None
    placebo: bool = False
    note: str = ""


# 리스크 규칙 18종 (동결 — 결과 조회 전 확정)
RISK_RULES: dict[str, RiskRule] = {
    "base":   RiskRule(note="동결 기준 — 무손절·상수 사이즈"),
    # ── 반증 대조: 그냥 작게 한다 ──────────────────────────────────────
    "shr75":  RiskRule(size=0.75, note="대조: 명목 ×0.75"),
    "shr50":  RiskRule(size=0.50, note="대조: 명목 ×0.50"),
    "shr33":  RiskRule(size=0.33, note="대조: 명목 ×0.33"),
    "shr25":  RiskRule(size=0.25, note="대조: 명목 ×0.25"),
    # ── 손절 ──────────────────────────────────────────────────────────
    "stop2":  RiskRule(stop=2.0,  note="평단 −2×ATR (격자 밖 타이트)"),
    "stop3":  RiskRule(stop=3.0,  note="평단 −3×ATR (격자 밖)"),
    "stop6":  RiskRule(stop=6.0,  note="평단 −6×ATR (AVGDOWN 격자)"),
    "stop10": RiskRule(stop=10.0, note="평단 −10×ATR (AVGDOWN 격자)"),
    # ── 시간 청산 ─────────────────────────────────────────────────────
    "time24": RiskRule(hours=24,  note="보유 24시간 초과 시 전량 청산"),
    "time96": RiskRule(hours=96,  note="보유 96시간 초과 시 전량 청산"),
    # ── 변동성 타깃 사이징 ────────────────────────────────────────────
    "vt":     RiskRule(vt=(VT_LO, VT_HI),     note="명목 ∝ 30일중앙ATR비 / 현재ATR비"),
    "vtT":    RiskRule(vt=(VT_LO_T, VT_HI_T), note="같은 규칙, 타이트 클립"),
    # ── 자본 낙폭 스로틀 ──────────────────────────────────────────────
    "ddt10":  RiskRule(dd_half=0.10, dd_stop=0.20, note="낙폭 10% ×0.5 · 20% 진입정지"),
    "ddt05":  RiskRule(dd_half=0.05, dd_stop=0.10, note="낙폭 5% ×0.5 · 10% 진입정지"),
    # ── 조합 ──────────────────────────────────────────────────────────
    "s3_time": RiskRule(stop=3.0, hours=24, note="손절3 + 시간24"),
    "full":    RiskRule(stop=3.0, hours=24, dd_half=0.10, dd_stop=0.20,
                        vt=(VT_LO, VT_HI), note="손절3 + 시간24 + 변동성타깃 + 스로틀"),
    # ── 위약 대조 ─────────────────────────────────────────────────────
    "plcvt":  RiskRule(vt=(VT_LO, VT_HI), placebo=True,
                       note="위약: vt 와 배수 분포 동일, 봉 해시로 무작위 재배정"),
}
RISK_KEYS = tuple(RISK_RULES)
N_STRUCT = 208


# ── 시행 열거 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Trial:
    """시행 1개 — 구조 격자 한 점 × 리스크 규칙 한 개."""

    entry: str
    filt: int
    spacing: float | None
    kmax: int
    tp: str
    risk: str
    tf: str

    def tid(self) -> str:
        """시행 ID — `RK|e=..|f=..|sp=..|k=..|tp=..|rk=..|tf`."""
        sp = "-" if self.spacing is None else f"{self.spacing:g}"
        return (f"RK|e={self.entry}|f={self.filt}|sp={sp}|k={self.kmax}"
                f"|tp={self.tp}|rk={self.risk}|{self.tf}")

    def base(self) -> ads.Trial:
        """대응하는 AVGDOWN 원 시행 (`risk='base'` 동치성 검증용)."""
        return ads.Trial(self.entry, self.filt, self.spacing, self.kmax,
                         self.tp, None, self.tf)


def enumerate_trials() -> list[Trial]:
    """격자 전수 열거 (AVGDOWN §3.7 중복 제거 규약 계승, 순서 동결).

    Returns:
        `208 × 18 × 2 = 7,488` 개의 `Trial`.

    Raises:
        AssertionError: 총계·ID 유일성 위반 시.
    """
    ladders: list[tuple[float | None, int]] = [(None, 0)]
    ladders += [(sp, k) for sp in SPACINGS for k in KMAXES if k > 0]
    out = [Trial(e, f, sp, k, tp, rk, tf)
           for tf in TFS for rk in RISK_KEYS for e in ENTRIES for f in FILTERS
           for sp, k in ladders for tp in TPS]
    n_expect = N_STRUCT * len(RISK_RULES) * len(TFS)
    assert len(out) == n_expect, f"시행 총계 {len(out)} != {n_expect}"
    assert len({t.tid() for t in out}) == len(out), "시행 ID 유일성 위반"
    return out


# ── 리스크 배수 사전계산 (확정봉 [i-1] 기준) ──────────────────────────────
def vt_multipliers(df: pd.DataFrame, tf: str, sym: str) -> tuple[np.ndarray, np.ndarray]:
    """변동성 타깃 배수와 그 위약 배수 (n,).

    배수 = clip(30일 중앙 ATR비 / 현재 ATR비, lo, hi). 30일 중앙값은
    `lab/confluence_tune.py` 의 `LOWVOL` 조건과 같은 창을 쓴다 (인과적 롤링).
    위약은 **원신호 봉에서의 배수 다중집합을 그대로 보존**한 채 봉을 뒤섞는다.

    Args:
        df: OHLCV. tf: 타임프레임 키. sym: 심볼 (해시 시드 구분).

    Returns:
        `(raw (n,), placebo_raw (n,))` — 클립 전 원시 비율. 클립은 규칙별로 적용한다.
    """
    n = len(df)
    c = df["close"].to_numpy(float)
    atr = ads.bb.atr(df).to_numpy()
    bpd = ct.BARS_PER_DAY[tf]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = atr / c
    med = pd.Series(ratio).rolling(30 * bpd, min_periods=10 * bpd).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = med / ratio
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1.0)
    raw = ct._lag(raw, 1)                       # 확정봉 [i-1]
    raw = np.where(np.isfinite(raw), raw, 1.0)

    sig = ct.raw_signal_mask(df)
    plc = np.ones(n)
    idx = np.flatnonzero(sig)
    if len(idx):
        seed = int(hashlib.sha256(f"{PLACEBO_SEED}:{sym}:{tf}".encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(seed).permutation(len(idx))
        plc[idx[order]] = raw[idx]              # 배수 다중집합 보존, 봉만 뒤섞음
    return raw, plc


def trial_arrays(trials: list[Trial]) -> dict[str, np.ndarray]:
    """시행 리스트 → 시뮬레이션용 정적 배열 (AVGDOWN `trial_arrays` + 리스크 축)."""
    r = len(trials)
    rr = [RISK_RULES[t.risk] for t in trials]
    bph = np.array([BARS_PER_HOUR[t.tf] for t in trials], dtype=np.int64)
    return {
        "is_e1": np.array([t.entry == "E1" for t in trials]),
        "filt": np.array([t.filt == 1 for t in trials]),
        "spacing": np.array([np.nan if t.spacing is None else t.spacing for t in trials]),
        "ntr": np.array([t.kmax + 1 for t in trials], dtype=np.int64),
        "tp_mid": np.array([t.tp == "MID" for t in trials]),
        "tp_mult": np.array([TP_MULT[t.tp] for t in trials]),
        "stop_mult": np.array([np.nan if x.stop is None else x.stop for x in rr]),
        "size": np.array([x.size for x in rr]),
        "hold_bars": np.array([NO_CAP if x.hours is None else x.hours * b
                               for x, b in zip(rr, bph)], dtype=np.int64),
        "dd_half": np.array([np.inf if x.dd_half is None else x.dd_half for x in rr]),
        "dd_stop": np.array([np.inf if x.dd_stop is None else x.dd_stop for x in rr]),
        "vt_lo": np.array([np.nan if x.vt is None else x.vt[0] for x in rr]),
        "vt_hi": np.array([np.nan if x.vt is None else x.vt[1] for x in rr]),
        "use_vt": np.array([x.vt is not None for x in rr]),
        "placebo": np.array([x.placebo for x in rr]),
        "R": r,
    }


# ── 슬리브 시뮬레이션 (AVGDOWN `simulate_sleeve` + 리스크 통제) ────────────
def simulate_sleeve(df: pd.DataFrame, fund_sym: pd.Series, ta: dict,
                    vt_raw: np.ndarray, vt_plc: np.ndarray,
                    causal: bool = True) -> dict:
    """심볼 1개 슬리브를 시행 축(R) 벡터화로 굴린다.

    `lab/avgdown_sweep.simulate_sleeve` 와 **연산 순서까지 동일**하다. 추가된 것은
    (a) 시행별 트랜치 비율 `frac`, (b) 시간 청산, (c) 자본 낙폭 스로틀뿐이며,
    `risk='base'` 에서는 셋 다 무해한 항등원이 된다.

    Args:
        df: OHLCV. fund_sym: 일별 펀딩합. ta: `trial_arrays()` 산출물.
        vt_raw: (n,) 변동성 타깃 원시 배수. vt_plc: (n,) 위약 배수.
        causal: False 면 같은 봉 신호 평가 (**룩어헤드 대조군 전용**).

    Returns:
        dict — `final_eq`, `n_trades`, `n_wins`, `day_eq`, `days`, `time_viol`,
        `bars_in_pos`, `worst_trade`, `n_stop`, `n_time`, `n_dd_block`.
    """
    r = ta["R"]
    o = df["open"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    sh = 1 if causal else 0

    cs = pd.Series(c)
    mid = cs.rolling(BB_N).mean().to_numpy()
    sd = cs.rolling(BB_N).std(ddof=0).to_numpy()
    c1 = ads._shift(c, sh)
    m1 = ads._shift(mid, sh)
    lb1 = ads._shift(mid - BB_K * sd, sh)
    a1 = ads._shift(ads.bb.atr(df).to_numpy(), sh)
    rs1 = ads._shift(ads.rsi_wilder(c), sh)
    tr1 = ads._shift(cs.rolling(TREND_N).mean().to_numpy(), sh)

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

    size = ta["size"]
    use_vt = ta["use_vt"]
    is_plc = ta["placebo"]
    vt_lo, vt_hi = ta["vt_lo"], ta["vt_hi"]
    hold_bars = ta["hold_bars"]
    dd_half, dd_stop = ta["dd_half"], ta["dd_stop"]

    eq = np.ones(r)
    u = np.zeros(r)
    basis = np.zeros(r)
    fees = np.zeros(r)
    thr = np.full(r, np.nan)
    stoplvl = np.full(r, np.nan)
    tplvl = np.full(r, np.nan)
    k = np.zeros(r, dtype=np.int64)
    entry_i = np.full(r, -1, dtype=np.int64)
    frac_open = np.full(r, TRANCHE_FRAC)
    halted = np.zeros(r, dtype=bool)
    day_eq = np.ones(r)
    peak = np.ones(r)                     # 전일까지의 일말 자본 최고치 (인과적)
    dd_scale = np.ones(r)                 # 낙폭 스로틀 배수 (일 시작에 갱신)
    dd_block = np.zeros(r, dtype=bool)    # 낙폭 진입정지 (일 시작에 갱신)
    n_trades = np.zeros(r, dtype=np.int64)
    n_wins = np.zeros(r, dtype=np.int64)
    time_viol = np.zeros(r, dtype=np.int64)
    bars_in_pos = np.zeros(r, dtype=np.int64)
    worst_trade = np.zeros(r)
    n_stop = np.zeros(r, dtype=np.int64)
    n_time = np.zeros(r, dtype=np.int64)
    n_dd_block = np.zeros(r, dtype=np.int64)
    day_mat = np.ones((r, len(snap_i)))

    tiny = 1e-300

    def close_mask(m: np.ndarray, x: float, i: int, kind: int = 0) -> None:
        """마스크 시행 전량 청산 — AVGDOWN `close_mask` 동형."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        pnl = u * x - basis - (fees + u * x * COST_OUT)
        eq = np.where(m, eq + u * x - basis - u * x * COST_OUT, eq)
        n_trades[m] += 1
        n_wins[m & (pnl > 0)] += 1
        time_viol[m & (entry_i > i)] += 1
        worst_trade[:] = np.where(m, np.minimum(worst_trade, pnl), worst_trade)
        if kind == 1:
            n_time[m] += 1
        u = np.where(m, 0.0, u)
        basis = np.where(m, 0.0, basis)
        fees = np.where(m, 0.0, fees)
        thr = np.where(m, np.nan, thr)
        stoplvl = np.where(m, np.nan, stoplvl)
        tplvl = np.where(m, np.nan, tplvl)
        k = np.where(m, 0, k)
        entry_i = np.where(m, -1, entry_i)

    def try_fill(m: np.ndarray, o_i: float, a1_i: float, pc: float, i: int,
                 frac: np.ndarray) -> None:
        """트랜치 1개 체결 시도 — AVGDOWN `try_fill` 동형 + 시행별 트랜치 비율."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        m = m & (eq > 0)
        if not (np.isfinite(o_i) and o_i > 0) or not m.any():
            return
        mk = pc if np.isfinite(pc) else np.nan
        gross = np.where(u > 0, u * np.where(np.isfinite(mk), mk,
                                             basis / np.maximum(u, tiny)), 0.0)
        m = m & (gross < GROSS_CAP * eq)
        unew = np.minimum(frac * eq / o_i,
                          np.maximum(0.0, GROSS_CAP * eq - gross) / o_i)
        m = m & (unew > 0)
        heat = basis * HEAT_FRAC
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
        thr = np.where(m, (o_i - ta["spacing"] * a1_i) if good else np.nan, thr)
        if good:
            avg = basis / np.maximum(u, tiny)
            stoplvl = np.where(m & np.isfinite(ta["stop_mult"]),
                               avg - ta["stop_mult"] * a1_i, stoplvl)
            tplvl = np.where(m & ~ta["tp_mid"], avg + ta["tp_mult"] * a1_i, tplvl)

    for i in range(n):
        if i < WARMUP:
            continue
        pc = c[i - 1] if i > 0 else np.nan
        if new_day[i]:
            has = u > 0
            day_eq = eq + np.where(has & np.isfinite(pc), u * pc - basis, 0.0)
            halted[:] = False
            # 낙폭 스로틀 — 전일 말 자본과 전일까지의 peak 로만 판정 (인과적)
            peak = np.maximum(peak, day_eq)
            dd = np.where(peak > 0, 1.0 - day_eq / peak, 0.0)
            dd_scale = np.where(dd >= dd_half, 0.5, 1.0)
            dd_block = dd >= dd_stop
            f = f_map[i]
            if np.isfinite(f) and np.isfinite(pc):
                eq = eq - np.where(has, f * u * pc, 0.0)
        o_i, l_i, c_i = o[i], lo[i], c[i]
        if np.isnan(c_i):
            m = u > 0
            if m.any() and np.isfinite(pc):
                close_mask(m, pc, i)
            if day_pos[i] >= 0:
                day_mat[:, day_pos[i]] = eq
            continue
        c1_i, m1_i, lb1_i, a1_i, rs1_i, tr1_i = (c1[i], m1[i], lb1[i], a1[i],
                                                 rs1[i], tr1[i])
        has0 = u > 0
        bars_in_pos += has0.astype(np.int64)
        sig_ok = np.isfinite(c1_i)
        # 1) 익절 (전량, 시가 체결, 같은 봉 재진입 금지)
        exit_m = np.zeros(r, dtype=bool)
        if sig_ok and np.isfinite(o_i):
            mid_sig = np.isfinite(m1_i) and c1_i >= m1_i
            exit_m = has0 & np.where(ta["tp_mid"], mid_sig,
                                     np.isfinite(tplvl) & (c1_i >= tplvl))
            if exit_m.any():
                close_mask(exit_m, o_i, i)
        # 1b) 시간 청산 (익절 다음, 시가 체결 — 익절 우선)
        if np.isfinite(o_i):
            time_m = (u > 0) & (entry_i >= 0) & ((i - entry_i) >= hold_bars)
            if time_m.any():
                close_mask(time_m, o_i, i, kind=1)
                exit_m = exit_m | time_m
        # 2) 추매 — 진입 시 동결된 트랜치 비율 사용
        if sig_ok:
            add_m = ((u > 0) & ~exit_m & (k < ta["ntr"]) & np.isfinite(thr)
                     & (c1_i <= thr) & ~halted)
            if add_m.any():
                try_fill(add_m, o_i, a1_i, pc, i, frac_open)
        # 3) 신규 진입 — 사이즈 배수를 이 봉에서 동결
        if sig_ok:
            e1_sig = np.isfinite(lb1_i) and c1_i < lb1_i
            e2_sig = np.isfinite(rs1_i) and rs1_i < RSI_TH
            base = np.where(ta["is_e1"], e1_sig, e2_sig)
            trend_ok = np.isfinite(tr1_i) and c1_i > tr1_i
            ent_m = (~has0) & base & (~ta["filt"] | trend_ok) & ~halted
            n_dd_block += (ent_m & dd_block).astype(np.int64)
            ent_m = ent_m & ~dd_block
            if ent_m.any():
                vraw = np.where(is_plc, vt_plc[i], vt_raw[i])
                vmul = np.where(use_vt, np.clip(vraw, vt_lo, vt_hi), 1.0)
                frac_new = TRANCHE_FRAC * size * vmul * dd_scale
                frac_open = np.where(ent_m, frac_new, frac_open)
                try_fill(ent_m, o_i, a1_i, pc, i, frac_open)
        # 4) 재해손절 — 봉내 스탑주문 (갭 악화 체결)
        if np.isfinite(l_i) and np.isfinite(o_i):
            stop_m = (u > 0) & np.isfinite(stoplvl) & (l_i <= stoplvl)
            if stop_m.any():
                x = np.minimum(o_i, stoplvl)
                pnl = u * x - basis - (fees + u * x * COST_OUT)
                eq = np.where(stop_m, eq + u * x - basis - u * x * COST_OUT, eq)
                n_trades[stop_m] += 1
                n_wins[stop_m & (pnl > 0)] += 1
                time_viol[stop_m & (entry_i > i)] += 1
                worst_trade[:] = np.where(stop_m, np.minimum(worst_trade, pnl),
                                          worst_trade)
                n_stop[stop_m] += 1
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
    m = u > 0
    if m.any():
        px = c[~np.isnan(c)][-1]
        close_mask(m, px, n - 1)
    if len(snap_i):
        day_mat[:, -1] = eq
    return {"final_eq": eq, "n_trades": n_trades, "n_wins": n_wins,
            "day_eq": day_mat, "days": days, "time_viol": time_viol,
            "bars_in_pos": bars_in_pos, "worst_trade": worst_trade,
            "n_stop": n_stop, "n_time": n_time, "n_dd_block": n_dd_block,
            "n_bars": n}


# ── 격자 실행 ─────────────────────────────────────────────────────────────
def build_context(data: dict) -> dict:
    """타임프레임·심볼별 변동성 타깃 배수와 위약 배수를 미리 계산한다."""
    return {tf: {s: vt_multipliers(data[tf][s], tf, s) for s in SYMS} for tf in TFS}


def run_grid(data: dict, fund: pd.DataFrame, ctx: dict, trials: list[Trial],
             causal: bool = True, progress: bool = True) -> dict[str, np.ndarray]:
    """전 시행 × 3심볼 실행 → 일수익률 행렬과 요약 배열."""
    n = len(trials)
    per_sym = {s: np.zeros((n, N_DAYS)) for s in SYMS}
    agg = {key: np.zeros((n, len(SYMS)), dtype=np.int64)
           for key in ("n_trades", "n_wins", "n_stop", "n_time", "n_dd_block",
                       "bars_in_pos")}
    final_eq = np.ones((n, len(SYMS)))
    worst_trade = np.zeros((n, len(SYMS)))
    time_viol = np.zeros(n, dtype=np.int64)
    total_bars = {s: 0 for s in SYMS}
    for tf in TFS:
        rows = [i for i, t in enumerate(trials) if t.tf == tf]
        if not rows:
            continue
        ta = trial_arrays([trials[i] for i in rows])
        for j, s in enumerate(SYMS):
            t0 = time.time()
            vt_raw, vt_plc = ctx[tf][s]
            res = simulate_sleeve(data[tf][s], fund[s], ta, vt_raw, vt_plc,
                                  causal=causal)
            per_sym[s][rows] = ads.sleeve_returns(res["day_eq"], res["days"])
            for key in agg:
                agg[key][rows, j] = res[key]
            final_eq[rows, j] = res["final_eq"]
            worst_trade[rows, j] = res["worst_trade"]
            time_viol[rows] += res["time_viol"]
            total_bars[s] = res["n_bars"]
            if progress:
                logger.info("%s %s: %d 시행 %.1fs", tf, s, len(rows), time.time() - t0)
    out = {"ret_combined": sum(per_sym[s] for s in SYMS) / len(SYMS),
           "final_eq": final_eq, "worst_trade": worst_trade,
           "time_viol": time_viol, "total_bars": total_bars}
    out.update(agg)
    for s in SYMS:
        out[f"ret_{s}"] = per_sym[s]
    return out


# ── 리스크 지표 ───────────────────────────────────────────────────────────
def risk_metrics(ret: np.ndarray) -> dict[str, np.ndarray]:
    """일수익률 행렬 (N, T) → 리스크 중심 지표.

    Returns:
        `cum`, `ann`, `sharpe`, `mdd`, `calmar`, `ulcer`, `worst_day`,
        `dd_days` (낙폭 5% 초과 일수 비율), `ruin` (자본 −50% 도달 여부).
    """
    t = ret.shape[1]
    eq = np.cumprod(1.0 + ret, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / peak - 1.0
        mdd = dd.min(axis=1)
        ulcer = np.sqrt((dd ** 2).mean(axis=1))
        sd = ret.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, ret.mean(axis=1) / sd * np.sqrt(365.0), 0.0)
        cum = eq[:, -1] - 1.0
        ann = np.where(eq[:, -1] > 0, eq[:, -1] ** (365.0 / t) - 1.0, -1.0)
        calmar = np.where(mdd < 0, ann / -mdd, np.nan)
    return {"cum": cum, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "calmar": calmar, "ulcer": ulcer,
            "worst_day": ret.min(axis=1),
            "dd_days": (dd < -0.05).mean(axis=1),
            "ruin": (eq.min(axis=1) < 0.5).astype(float)}


# ── 자가검증 ──────────────────────────────────────────────────────────────
def selftest() -> None:
    """AVGDOWN 원 엔진과의 동치성·인과성·위약 대칭성을 강제한다.

    (a) `risk='base'` 15m 208 시행이 `avgdown_sweep` 의 `sl=None` 시행과 비트 동일.
    (b) `shr50` 이 `base` 와 정확히 같은 거래 수를 내고 (사이즈만 다름),
        `stop*`·`time*` 은 거래 수가 달라진다 (규칙이 실제로 물린다).
    (c) 위약 배수의 다중집합이 실제 배수와 같다.
    (d) 시간 청산이 보유 봉수 상한을 실제로 지킨다.

    Raises:
        AssertionError: 어떤 불변식이라도 깨질 때.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data, fund, _ = ct.load_data()
    ctx = build_context(data)

    ladders: list[tuple[float | None, int]] = [(None, 0)]
    ladders += [(sp, k) for sp in SPACINGS for k in KMAXES if k > 0]
    mine = [Trial(e, f, sp, k, tp, "base", "15m")
            for e in ENTRIES for f in FILTERS for sp, k in ladders for tp in TPS]
    assert len(mine) == N_STRUCT, f"구조 축 {len(mine)} != {N_STRUCT}"
    logger.info("(a) 동치성 검증 — base 15m %d 시행", len(mine))
    got = run_grid(data, fund, ctx, mine, progress=False)
    ref = ads.run_grid({"15m": data["15m"]}, fund, [t.base() for t in mine],
                       progress=False)
    for key in ("ret_combined", "n_trades", "n_wins", "final_eq"):
        assert np.array_equal(got[key], ref[key], equal_nan=True), \
            f"(a) 동치성 위반: {key} 가 avgdown_sweep 과 다르다"
    logger.info("(a) OK — base 설정이 avgdown_sweep(sl=None) 과 비트 단위 동일")

    probe = [Trial("E2", 1, 1.0, 3, "A2.0", rk, "15m")
             for rk in ("base", "shr50", "stop3", "time24", "vt", "plcvt", "ddt05")]
    pr = run_grid(data, fund, ctx, probe, progress=False)
    tr = pr["n_trades"].sum(axis=1)
    named = dict(zip([t.risk for t in probe], tr))
    assert named["shr50"] == named["base"], \
        f"(b) shr50 거래수 {named['shr50']} != base {named['base']} — 축소가 신호를 바꿨다"
    assert named["stop3"] > named["base"], "(b) stop3 이 거래를 늘리지 않았다 — 손절 미작동"
    assert named["time24"] > named["base"], "(b) time24 가 거래를 늘리지 않았다 — 시간청산 미작동"
    assert pr["n_stop"].sum(axis=1)[2] > 0, "(b) stop3 손절 체결 0건"
    assert pr["n_time"].sum(axis=1)[3] > 0, "(b) time24 시간청산 0건"
    logger.info("(b) OK — 거래수 base %d · shr50 %d · stop3 %d · time24 %d · "
                "vt %d · ddt05 %d (손절 %d건 · 시간청산 %d건 · 진입차단 %d건)",
                named["base"], named["shr50"], named["stop3"], named["time24"],
                named["vt"], named["ddt05"], pr["n_stop"].sum(axis=1)[2],
                pr["n_time"].sum(axis=1)[3], pr["n_dd_block"].sum(axis=1)[6])

    for tf in TFS:
        for s in SYMS:
            raw, plc = ctx[tf][s]
            sig = ct.raw_signal_mask(data[tf][s])
            a = np.sort(raw[sig])
            b = np.sort(plc[sig])
            assert np.allclose(a, b), f"(c) {tf}/{s}: 위약 배수 다중집합 불일치"
    logger.info("(c) OK — 위약 배수가 실제 배수와 분포 동일")

    ta = trial_arrays([Trial("E2", 1, 1.0, 3, "A2.0", "time24", "15m")])
    raw, plc = ctx["15m"]["BTC"]
    res = simulate_sleeve(data["15m"]["BTC"], fund["BTC"], ta, raw, plc)
    assert res["n_time"][0] > 0, "(d) 시간 청산 0건 — 규칙 미작동"
    assert int(ta["hold_bars"][0]) == 96, f"(d) 24h 가 {ta['hold_bars'][0]}봉으로 환산됨"
    logger.info("(d) OK — 24h = 96봉(15m), 시간청산 %d건", res["n_time"][0])
    logger.info("자가검증 전부 통과")


# ── 본 실행 ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    ap = argparse.ArgumentParser(description=f"{SPEC} 리스크 통제 스윕")
    ap.add_argument("--selftest", action="store_true", help="자가검증만 수행")
    ap.add_argument("--run", action="store_true", help="본 격자 실행")
    ap.add_argument("--batch", type=int, default=4000, help="시행 배치 크기")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("--selftest 또는 --run 중 하나가 필요하다")

    t_start = time.time()
    data, fund, _ = ct.load_data()
    ctx = build_context(data)
    trials = enumerate_trials()
    logger.info("%s — %d 시행 (구조 %d × 리스크 %d × tf %d)",
                SPEC, len(trials), N_STRUCT, len(RISK_RULES), len(TFS))

    chunks = [trials[i:i + a.batch] for i in range(0, len(trials), a.batch)]
    parts: list[dict] = []
    for ci, ch in enumerate(chunks):
        logger.info("배치 %d/%d — %d 시행", ci + 1, len(chunks), len(ch))
        parts.append(run_grid(data, fund, ctx, ch))
    keys = [k for k in parts[0] if k != "total_bars"]
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    tot_bars = sum(parts[0]["total_bars"].values())

    m = risk_metrics(out["ret_combined"])
    rows = []
    for i, t in enumerate(trials):
        rr = RISK_RULES[t.risk]
        rows.append({
            "trial_id": t.tid(), "tf": t.tf, "entry": t.entry, "filt": t.filt,
            "spacing": t.spacing, "kmax": t.kmax, "tp": t.tp, "risk": t.risk,
            "stop": rr.stop, "size": rr.size, "hours": rr.hours,
            "cum": m["cum"][i], "ann": m["ann"][i], "sharpe": m["sharpe"][i],
            "mdd": m["mdd"][i], "calmar": m["calmar"][i], "ulcer": m["ulcer"][i],
            "worst_day": m["worst_day"][i], "dd_days": m["dd_days"][i],
            "ruin": m["ruin"][i],
            "n_trades": int(out["n_trades"][i].sum()),
            "n_wins": int(out["n_wins"][i].sum()),
            "n_stop": int(out["n_stop"][i].sum()),
            "n_time": int(out["n_time"][i].sum()),
            "n_dd_block": int(out["n_dd_block"][i].sum()),
            "worst_trade": float(out["worst_trade"][i].min()),
            "exposure": float(out["bars_in_pos"][i].sum()) / max(tot_bars, 1),
            "time_viol": int(out["time_viol"][i]),
        })
    df = pd.DataFrame(rows)
    df["win_rate"] = np.where(df.n_trades > 0, df.n_wins / df.n_trades, np.nan)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "risk_tune_summary.csv", index=False)

    npz_path = OUT_DIR / "risk_tune_returns.npz"
    np.savez_compressed(
        npz_path,
        daily_returns=np.ascontiguousarray(out["ret_combined"], dtype=np.float64),
        trial_ids=np.array([t.tid() for t in trials], dtype=object),
        snap_ts=np.array([str(d) for d in ads.master_days()], dtype="<U25"),
        meta=json.dumps({"spec": SPEC, "n_trials": len(trials), "n_days": N_DAYS},
                        ensure_ascii=False))
    h = hashlib.sha256()
    with open(npz_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    meta = {
        "spec": SPEC, "generated_at": datetime.now(timezone.utc).isoformat(),
        "returns_npz_sha256": h.hexdigest(), "n_trials": len(trials),
        "n_struct": N_STRUCT,
        "risk_rules": {k: {"stop": v.stop, "size": v.size, "hours": v.hours,
                           "dd_half": v.dd_half, "dd_stop": v.dd_stop,
                           "vt": list(v.vt) if v.vt else None,
                           "placebo": v.placebo, "note": v.note}
                       for k, v in RISK_RULES.items()},
        "placebo_seed": PLACEBO_SEED, "elapsed_s": time.time() - t_start,
    }
    (OUT_DIR / "risk_tune_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    assert int(df.time_viol.sum()) == 0, "시간 인과성 위반 발생 — 결과 폐기"
    logger.info("완료 %.1fs → logs/risk_tune_summary.csv", time.time() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
