"""컨플루언스 확신도 사이징 튜닝 엔진 — CONF-TUNE-2026-09-04.

목적
----
"모든 조건이 딱 맞아떨어지는 자리에서 비중을 키운다" 는 가설을 AVGDOWN-2026-09-01
구조 위에서 **분모를 고정한 채** 검정한다.

기반 구조는 `lab/avgdown_sweep.py` (사전등록 AVGDOWN-2026-09-01) 를 **연산 순서까지
계승**한다. 본 모듈은 그 엔진을 수정하지 않고 읽기 전용으로 임포트하며, 중립 설정
(`conv='flat'`) 에서 원 엔진과 **비트 단위 동일** 결과를 내야 한다 —
`--selftest` 가 이를 강제한다.

추가한 축은 하나뿐이다
----------------------
**확신도 사이징**: 트랜치 명목 `equity × 1/12` 에 진입 시점 등급별 배수를 곱한다.
게이트(선택적 거래 배제)가 아니라 **사이징**이 1차 가설이다. 이유:

- 기반 신호(`RSI14<30 & close>SMA200`)는 이미 전체 봉의 **0.19~0.24%** 에서만
  발생한다 (BTC 371 / ETH 459 / SOL 301 원신호, 5.6년). 여기에 AND 게이트를 더하면
  표본이 붕괴한다 — `docs/TRADINGVIEW_PRECISION_V6.md` 가 기록한 실패 형태
  (Ω 858 → 선택 116, coverage 13.5%, 요구 표본 300 미달) 와 같은 함정이다.
- 사이징은 **모든 원신호를 분모에 남긴다**. 등급이 정보를 담고 있다면 크기가
  실린 거래의 가중 수익이 평탄 사이징을 이긴다. 담고 있지 않다면 지지 않는다.

게이트 규칙(`gateA`, `gateAp`)도 격자에 포함하되 비교군으로만 둔다.

반증 대조군 (이 두 개가 본 검정을 낚시와 구분한다)
---------------------------------------------
- `inv*`  — 등급을 **뒤집어** 하위 등급에 큰 비중을 준다. 컨플루언스가 정보라면
  평탄보다 나빠야 한다.
- `plc*`  — 등급 라벨을 심볼·봉 인덱스 해시로 **무작위 재배정**하되 등급별 빈도는
  동일하게 유지한다(시드 고정·결정론적). 크기 분포가 같으므로, 이 위약군이 실제
  등급과 비슷하게 좋다면 관측된 개선은 "컨플루언스"가 아니라 **단순 레버리지**다.

실행 인과성 (위반 시 결과 폐기 — AVGDOWN §4.4 계승)
-------------------------------------------------
- 모든 조건은 확정봉 `[i-1]` 이하 값만 쓴다. 체결은 봉 `[i]` 시가.
- 봉 `[i]` 의 동일봉 사용은 (a) 체결가 `open[i]`, (b) 손절 봉내 트리거 `low[i]`,
  (c) 봉 마감 후 MTM·일손실 판정 `close[i]` 뿐이다.
- 확신도 배수는 **첫 트랜치 진입 봉에서 동결**되고 그 거래의 추매 전부에 동일하게
  적용된다. 보유 중 재채점은 하지 않는다 (악화되는 셋업이 비중을 늘리는 것을 금지).
- 등급 임계값은 **IS 구간(≤2024-12-31)의 원신호 봉 점수 분포**에서만 계산하고
  전 구간에 적용한다. 손익을 보지 않고 빈도만 쓰므로 결과 의존이 아니다.

동결 사항 (결과 조회 전 고정)
---------------------------
- 조건 10종과 그 파라미터 (`CONDITIONS`) — 경제적 근거와 기존 저장소 문서에서 유도.
- 등급 빈도 분할: A+ = 상위 10%, A = 상위 30%(A+ 포함), 그 외 B.
- IS/OOS 경계 = `2024-12-31` (`logs/sweep_meta.json` 의 `is_end` 계승).
- 사이징 배수 격자, 위약 시드 `20260904`.

공개된 한계
----------
- `lab/data/sol_1h.parquet` 은 저장소에 없다. 1h 팔은 `sol_15m` 을 1시간으로
  재집계해 만든다. BTC 로 검증한 재집계 오차는 평균 상대차 ~1e-8 (`--selftest`
  가 BTC 재집계 대 동결 1h 의 일치도를 보고한다).
- heat 캡(6%)·gross 캡(10x)은 동결 위험모델 그대로다. 큰 배수에서는 heat 캡이
  실제로 구속하므로, 배수를 올려도 명목이 비례해 늘지 않는다. 구속률을 함께 보고한다.
- 강제청산·부분체결·MTM 마진콜은 모델하지 않는다. 실현 MDD 는 계좌 낙폭 보장이 아니다.

실행:
  python3 -m lab.confluence_tune --selftest   # 동치성·인과성 자가검증 (격자 미실행)
  python3 -m lab.confluence_tune --run        # 본 격자 1회
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
WARMUP = ads.WARMUP
N_DAYS = ads.N_DAYS

# ── 본 명세 동결 상수 ─────────────────────────────────────────────────────
SPEC = "CONF-TUNE-2026-09-04"
PLACEBO_SEED = 20260904
IS_END = pd.Timestamp("2024-12-31 23:59:00", tz="utc")   # sweep_meta.json 계승
TIER_Q_AP = 0.90          # A+ = 원신호 점수 상위 10%
TIER_Q_A = 0.70           # A  = 원신호 점수 상위 30% (A+ 포함)
N_COND = 10

TIER_B, TIER_A, TIER_AP = 0, 1, 2

# 구조 격자 (AVGDOWN §3 그대로)
ENTRIES = ads.ENTRIES
FILTERS = ads.FILTERS
SPACINGS = ads.SPACINGS
KMAXES = ads.KMAXES
TPS = ads.TPS
TP_MULT = ads.TP_MULT
STOPS = ads.STOPS
TFS = ("15m", "1h")

# 확신도 사이징 규칙 — (B배수, A배수, A+배수, 등급소스)
#   등급소스 'real' = 실제 컨플루언스 등급, 'placebo' = 빈도동일 무작위 재배정
CONV_RULES: dict[str, tuple[float, float, float, str]] = {
    "flat":    (1.0, 1.0, 1.0, "real"),      # 기준선 = AVGDOWN 원본
    "gateA":   (0.0, 1.0, 1.0, "real"),      # A 이상만 거래 (비교군)
    "gateAp":  (0.0, 0.0, 1.0, "real"),      # A+ 만 거래 (비교군)
    "x2Ap":    (1.0, 1.0, 2.0, "real"),      # A+ 에서 2배
    "x3Ap":    (1.0, 1.0, 3.0, "real"),      # A+ 에서 3배
    "x4Ap":    (1.0, 1.0, 4.0, "real"),      # A+ 에서 4배
    "x2A":     (1.0, 2.0, 2.0, "real"),
    "x3A":     (1.0, 3.0, 3.0, "real"),
    "ramp3":   (1.0, 2.0, 3.0, "real"),      # 계단식 증량
    "ramp4":   (1.0, 2.0, 4.0, "real"),
    "invR":    (3.0, 2.0, 1.0, "real"),      # 반증 대조 — 등급 역전
    "plcR3":   (1.0, 2.0, 3.0, "placebo"),   # 위약 대조 — ramp3 과 크기분포 동일
}
CONV_KEYS = tuple(CONV_RULES)

# ── 컨플루언스 조건 10종 (동결 — 결과 조회 전 확정) ────────────────────────
# 전부 확정봉 [i-1] 이하만 사용한다. 각 항목: (키, 한 줄 근거)
CONDITIONS: tuple[tuple[str, str], ...] = (
    ("TR_SLOPE",  "SMA200 이 200봉 전보다 높다 — 추세가 실제로 상승 중"),
    ("HTF_TREND", "종가 > SMA800 (상위 시간대 추세 대용)"),
    ("BTC_REG",   "같은 시각 BTC 종가 > BTC SMA200 — 시장 전체 레짐"),
    ("LOWVOL",    "ATR24/종가 <= 자기 30일 중앙값 — 저변동 국면 (PRECISION_V6 최강 게이트)"),
    ("DEEP",      "(종가-SMA20)/σ20 <= -2.5 — 되돌림 깊이"),
    ("VOLCLIMAX", "거래량 >= 직전 20봉 평균 × 1.5 — 투매 소진"),
    ("WICK",      "아래꼬리 >= 봉 레인지 × 0.5 — 하방 거부"),
    ("NEARHIGH",  "종가 >= 최근 30일 고가 × 0.80 — 낙하하는 칼이 아님"),
    ("FUND_OK",   "직전 72시간 펀딩 누계 <= 0.0004 — 롱 캐리 비용 낮음"),
    ("RSI_UP",    "RSI14 가 직전 봉보다 상승 — 모멘텀 반전 시작"),
)
assert len(CONDITIONS) == N_COND

BARS_PER_DAY = {"15m": 96, "1h": 24}


# ── 시행 열거 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Trial:
    """시행 1개 — 구조 격자 한 점 × 확신도 규칙 한 개."""

    entry: str
    filt: int
    spacing: float | None
    kmax: int
    tp: str
    stop: float | None
    conv: str
    tf: str

    def tid(self) -> str:
        """시행 ID — `CT|e=..|f=..|sp=..|k=..|tp=..|sl=..|cv=..|tf`."""
        sp = "-" if self.spacing is None else f"{self.spacing:g}"
        sl = "-" if self.stop is None else f"{self.stop:g}"
        return (f"CT|e={self.entry}|f={self.filt}|sp={sp}|k={self.kmax}"
                f"|tp={self.tp}|sl={sl}|cv={self.conv}|{self.tf}")

    def base(self) -> ads.Trial:
        """대응하는 AVGDOWN 원 시행 (동치성 검증용)."""
        return ads.Trial(self.entry, self.filt, self.spacing, self.kmax,
                         self.tp, self.stop, self.tf)


def enumerate_trials() -> list[Trial]:
    """격자 전수 열거 (AVGDOWN §3.7 중복 제거 규약 계승, 순서 동결).

    Returns:
        `624 × len(CONV_RULES) × len(TFS)` 개의 `Trial`.

    Raises:
        AssertionError: ID 유일성 위반 시.
    """
    ladders: list[tuple[float | None, int]] = [(None, 0)]
    ladders += [(sp, k) for sp in SPACINGS for k in KMAXES if k > 0]
    out = [Trial(e, f, sp, k, tp, sl, cv, tf)
           for tf in TFS for cv in CONV_KEYS for e in ENTRIES for f in FILTERS
           for sp, k in ladders for tp in TPS for sl in STOPS]
    n_expect = 624 * len(CONV_RULES) * len(TFS)
    assert len(out) == n_expect, f"시행 총계 {len(out)} != {n_expect}"
    assert len({t.tid() for t in out}) == len(out), "시행 ID 유일성 위반"
    return out


# ── 데이터 ────────────────────────────────────────────────────────────────
def _resample_1h(df15: pd.DataFrame) -> pd.DataFrame:
    """15분봉 → 1시간봉 재집계 (OHLCV 표준 규약)."""
    out = df15.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                                  low=("low", "min"), close=("close", "last"),
                                  volume=("volume", "sum"))
    return out.dropna(subset=["close"])


def load_data() -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    """15m(동결) + 1h OHLCV, 일별 펀딩, 8시간 펀딩 원본을 읽는다.

    1h 은 BTC/ETH 를 `lab/frozen/perp_1h.parquet` 에서, SOL 은 15m 재집계로 만든다
    (`lab/data/sol_1h.parquet` 미존재 — 모듈 docstring 의 공개된 한계).

    Returns:
        (`{tf: {sym: df}}`, 일별 펀딩합, 8시간 펀딩 원본).
    """
    d15 = {s: pd.read_parquet(ROOT / ads.PATHS_15M[s]) for s in SYMS}
    perp = pd.read_parquet(ROOT / "lab/frozen/perp_1h.parquet")
    cols = ["open", "high", "low", "close", "volume"]
    d1h = {s: perp.xs(s, level="sym")[cols] for s in ("BTC", "ETH")}
    d1h["SOL"] = _resample_1h(d15["SOL"])[cols]
    f8 = pd.read_parquet(ROOT / "lab/frozen/funding.parquet")[list(SYMS)]
    fund = f8.resample("D").sum(min_count=1)
    return {"15m": d15, "1h": d1h}, fund, f8


# ── 컨플루언스 조건 계산 (전부 확정봉 [i-1]) ───────────────────────────────
def condition_matrix(df: pd.DataFrame, sym: str, tf: str,
                     btc_close: pd.Series, f8: pd.Series) -> np.ndarray:
    """조건 10종의 통과 여부 행렬 (n, 10) — 봉 `i` 행은 확정봉 `[i-1]` 값 기준.

    NaN(워밍업 포함) 은 **불통과**로 처리한다 (fail-closed). 따라서 워밍업 구간의
    점수는 0 이고, AVGDOWN 워밍업 100봉 규약과 겹쳐 주문을 만들지 않는다.

    Args:
        df: 해당 심볼·타임프레임 OHLCV (UTC DatetimeIndex).
        sym: 심볼명. tf: 타임프레임 키.
        btc_close: BTC 종가 시리즈 (해당 타임프레임) — 레짐 조건용.
        f8: 해당 심볼의 8시간 펀딩 정산률 시리즈.

    Returns:
        bool 배열 (n, 10). 열 순서는 `CONDITIONS` 와 같다.
    """
    n = len(df)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    cs = pd.Series(c)
    bpd = BARS_PER_DAY[tf]

    sma200 = cs.rolling(TREND_N).mean().to_numpy()
    sma800 = cs.rolling(TREND_N * 4).mean().to_numpy()
    mid = cs.rolling(BB_N).mean().to_numpy()
    sd = cs.rolling(BB_N).std(ddof=0).to_numpy()
    atr = ads.bb.atr(df).to_numpy()
    rsi = ads.rsi_wilder(c)

    with np.errstate(divide="ignore", invalid="ignore"):
        atr_ratio = atr / c
        stretch = (c - mid) / sd
    atr_med30 = pd.Series(atr_ratio).rolling(30 * bpd, min_periods=10 * bpd).median().to_numpy()
    vol_ma20 = pd.Series(v).rolling(20).mean().to_numpy()
    hi30 = pd.Series(h).rolling(30 * bpd, min_periods=5 * bpd).max().to_numpy()
    rng = h - lo

    # BTC 레짐 — 해당 심볼 인덱스에 ffill 정렬 (t 이하 마지막 관측만 사용)
    btc = btc_close.reindex(df.index, method="ffill").to_numpy(float)
    btc_sma = pd.Series(btc).rolling(TREND_N).mean().to_numpy()

    # 펀딩 72시간 누계 — 8시간 정산 9회분. 정산시각 이하 마지막 값만 ffill.
    f72 = f8.rolling(9, min_periods=3).sum()
    fund72 = f72.reindex(df.index, method="ffill").to_numpy(float)

    raw = np.zeros((n, N_COND), dtype=bool)
    with np.errstate(invalid="ignore"):
        raw[:, 0] = _gt(sma200, _lag(sma200, TREND_N))
        raw[:, 1] = _gt(c, sma800)
        raw[:, 2] = _gt(btc, btc_sma)
        raw[:, 3] = _le(atr_ratio, atr_med30)
        raw[:, 4] = _le(stretch, -2.5)
        raw[:, 5] = _ge(v, vol_ma20 * 1.5)
        raw[:, 6] = _ge(c - lo, rng * 0.5)
        raw[:, 7] = _ge(c, hi30 * 0.80)
        raw[:, 8] = _le(fund72, 0.0004)
        raw[:, 9] = _gt(rsi, _lag(rsi, 1))

    out = np.zeros_like(raw)
    out[1:] = raw[:-1]                      # 확정봉 [i-1] 로 한 봉 지연
    return out


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    """`out[i] = a[i-k]` (앞쪽 NaN)."""
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:len(a) - k]
    return out


def _gt(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """NaN 을 불통과로 보는 `a > b`."""
    return np.greater(a, b, out=np.zeros(len(a), dtype=bool),
                      where=np.isfinite(a) & np.isfinite(np.broadcast_to(b, a.shape)))


def _ge(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """NaN 을 불통과로 보는 `a >= b`."""
    return np.greater_equal(a, b, out=np.zeros(len(a), dtype=bool),
                            where=np.isfinite(a) & np.isfinite(np.broadcast_to(b, a.shape)))


def _le(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """NaN 을 불통과로 보는 `a <= b`."""
    return np.less_equal(a, b, out=np.zeros(len(a), dtype=bool),
                         where=np.isfinite(a) & np.isfinite(np.broadcast_to(b, a.shape)))


def raw_signal_mask(df: pd.DataFrame) -> np.ndarray:
    """등급 임계 보정용 원신호 마스크 — E1 또는 E2 가 확정봉에서 성립한 봉.

    임계값은 손익이 아니라 **점수 분포의 빈도**만으로 정하므로 결과 의존이 아니다.
    """
    c = df["close"].to_numpy(float)
    cs = pd.Series(c)
    mid = cs.rolling(BB_N).mean().to_numpy()
    sd = cs.rolling(BB_N).std(ddof=0).to_numpy()
    c1 = ads._shift(c, 1)
    lb1 = ads._shift(mid - BB_K * sd, 1)
    rs1 = ads._shift(ads.rsi_wilder(c), 1)
    return _lt_pair(c1, lb1) | _lt_pair(rs1, np.full(len(c), RSI_TH))


def _lt_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NaN 을 불통과로 보는 `a < b`."""
    return np.less(a, b, out=np.zeros(len(a), dtype=bool),
                   where=np.isfinite(a) & np.isfinite(b))


def tier_array(score: np.ndarray, sig: np.ndarray, index: pd.DatetimeIndex,
               sym: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """점수 → 등급(B/A/A+) 과 위약 등급.

    임계값은 **IS 구간(<= IS_END) 의 원신호 봉 점수**의 70/90 분위로 정한다.
    위약 등급은 같은 원신호 봉 집합에서 등급별 개수를 보존한 채 결정론적 해시
    순서로 재배정한다 (시드 `PLACEBO_SEED`).

    Args:
        score: (n,) 조건 통과 개수. sig: (n,) 원신호 마스크.
        index: 봉 타임스탬프. sym: 심볼명 (해시 시드 구분용).

    Returns:
        (실제 등급 (n,) int8, 위약 등급 (n,) int8, 임계·빈도 진단 dict).
    """
    is_mask = sig & np.asarray(index <= IS_END)
    pool = score[is_mask]
    if len(pool) < 20:                                   # 표본 부족 — 전부 B
        z = np.zeros(len(score), dtype=np.int8)
        return z, z.copy(), {"n_is_signals": float(len(pool))}
    thr_a = float(np.quantile(pool, TIER_Q_A))
    thr_ap = float(np.quantile(pool, TIER_Q_AP))
    tier = np.where(score >= thr_ap, TIER_AP,
                    np.where(score >= thr_a, TIER_A, TIER_B)).astype(np.int8)

    # 위약: 원신호 봉만 대상으로 등급 라벨을 결정론적으로 셔플 (빈도 보존)
    plc = np.zeros(len(score), dtype=np.int8)
    idx = np.flatnonzero(sig)
    if len(idx):
        seed = int(hashlib.sha256(f"{PLACEBO_SEED}:{sym}".encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(seed).permutation(len(idx))
        labels = np.sort(tier[idx])[::-1]                 # 등급별 개수 그대로
        plc[idx[order]] = labels
    diag = {
        "n_is_signals": float(len(pool)),
        "thr_A": thr_a, "thr_Ap": thr_ap,
        "freq_B": float((tier[sig] == TIER_B).mean()),
        "freq_A": float((tier[sig] == TIER_A).mean()),
        "freq_Ap": float((tier[sig] == TIER_AP).mean()),
        "mean_score_signals": float(score[sig].mean()) if sig.any() else float("nan"),
    }
    return tier, plc, diag


# ── 시행 정적 배열 ────────────────────────────────────────────────────────
def trial_arrays(trials: list[Trial]) -> dict[str, np.ndarray]:
    """시행 리스트 → 시뮬레이션용 정적 배열 (AVGDOWN `trial_arrays` + 확신도 축)."""
    r = len(trials)
    mult = np.array([[CONV_RULES[t.conv][0], CONV_RULES[t.conv][1],
                      CONV_RULES[t.conv][2]] for t in trials], dtype=np.float64)
    return {
        "is_e1": np.array([t.entry == "E1" for t in trials]),
        "filt": np.array([t.filt == 1 for t in trials]),
        "spacing": np.array([np.nan if t.spacing is None else t.spacing for t in trials]),
        "ntr": np.array([t.kmax + 1 for t in trials], dtype=np.int64),
        "tp_mid": np.array([t.tp == "MID" for t in trials]),
        "tp_mult": np.array([TP_MULT[t.tp] for t in trials]),
        "stop_mult": np.array([np.nan if t.stop is None else t.stop for t in trials]),
        "mult": mult,                                     # (R, 3) 등급별 배수
        "placebo": np.array([CONV_RULES[t.conv][3] == "placebo" for t in trials]),
        "R": r,
    }


# ── 슬리브 시뮬레이션 (AVGDOWN `simulate_sleeve` + 확신도 사이징) ──────────
def simulate_sleeve(df: pd.DataFrame, fund_sym: pd.Series, ta: dict,
                    tier: np.ndarray, plc: np.ndarray,
                    causal: bool = True, record: bool = False) -> dict:
    """심볼 1개 슬리브를 시행 축(R) 벡터화로 굴린다.

    `lab/avgdown_sweep.simulate_sleeve` 와 **연산 순서까지 동일**하다. 유일한 차이는
    `try_fill` 의 트랜치 비율이 상수 `TRANCHE_FRAC` 대신 시행별 `frac (R,)` 이라는
    점이며, 모든 배수가 1.0 이면 `frac` 은 `TRANCHE_FRAC` 과 비트 단위로 같다.

    Args:
        df: OHLCV. fund_sym: 일별 펀딩합. ta: `trial_arrays()` 산출물.
        tier: (n,) 실제 등급. plc: (n,) 위약 등급.
        causal: False 면 같은 봉 신호 평가 (**룩어헤드 대조군 전용**).
        record: True 면 거래 원장을 함께 반환한다 (`R == 1` 전용, 산술 불변).

    Returns:
        dict — `final_eq`, `n_trades`, `n_wins`, `day_eq`, `days`, `time_viol`,
        `n_by_tier (R,3)`, `heat_blocked (R,)`.
    """
    r = ta["R"]
    o = df["open"].to_numpy(float)
    l = df["low"].to_numpy(float)
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

    mult = ta["mult"]                                    # (R, 3)
    is_plc = ta["placebo"]                               # (R,)

    eq = np.ones(r)
    u = np.zeros(r)
    basis = np.zeros(r)
    fees = np.zeros(r)
    thr = np.full(r, np.nan)
    stoplvl = np.full(r, np.nan)
    tplvl = np.full(r, np.nan)
    k = np.zeros(r, dtype=np.int64)
    entry_i = np.full(r, -1, dtype=np.int64)
    frac_open = np.full(r, TRANCHE_FRAC)                 # 진입 시 동결된 트랜치 비율
    entry_tier = np.zeros(r, dtype=np.int8)              # 진입 시 동결된 등급
    ledger: list[dict] = []
    halted = np.zeros(r, dtype=bool)
    day_eq = np.ones(r)
    n_trades = np.zeros(r, dtype=np.int64)
    n_wins = np.zeros(r, dtype=np.int64)
    time_viol = np.zeros(r, dtype=np.int64)
    n_by_tier = np.zeros((r, 3), dtype=np.int64)
    heat_blocked = np.zeros(r, dtype=np.int64)
    day_mat = np.ones((r, len(snap_i)))

    tiny = 1e-300

    def close_mask(m: np.ndarray, x: float, i: int) -> None:
        """마스크 시행 전량 청산 — AVGDOWN `close_mask` 동형."""
        nonlocal eq, u, basis, fees, thr, stoplvl, tplvl, k, entry_i
        pnl = u * x - basis - (fees + u * x * COST_OUT)
        eq = np.where(m, eq + u * x - basis - u * x * COST_OUT, eq)
        if record and m[0]:
            ledger.append({"entry_i": int(entry_i[0]), "exit_i": int(i),
                           "tier": int(entry_tier[0]), "pnl": float(pnl[0]),
                           "notional": float(basis[0]), "k": int(k[0]),
                           "exit": "signal"})
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
        heat_ok = (heat + unew * o_i * HEAT_FRAC <= HEAT_CAP * eq * (1 + 1e-9))
        heat_blocked[m & ~heat_ok] += 1
        m = m & heat_ok
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
            f = f_map[i]
            if np.isfinite(f) and np.isfinite(pc):
                eq = eq - np.where(has, f * u * pc, 0.0)
        o_i, l_i, c_i = o[i], l[i], c[i]
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
        sig_ok = np.isfinite(c1_i)
        # 1) 익절
        exit_m = np.zeros(r, dtype=bool)
        if sig_ok and np.isfinite(o_i):
            mid_sig = np.isfinite(m1_i) and c1_i >= m1_i
            exit_m = has0 & np.where(ta["tp_mid"], mid_sig,
                                     np.isfinite(tplvl) & (c1_i >= tplvl))
            if exit_m.any():
                close_mask(exit_m, o_i, i)
        # 2) 추매 — 진입 시 동결된 트랜치 비율 사용 (보유 중 재채점 금지)
        if sig_ok:
            add_m = (has0 & ~exit_m & (k < ta["ntr"]) & np.isfinite(thr)
                     & (c1_i <= thr) & ~halted)
            if add_m.any():
                try_fill(add_m, o_i, a1_i, pc, i, frac_open)
        # 3) 신규 진입 — 확신도 배수를 이 봉에서 동결
        if sig_ok:
            e1_sig = np.isfinite(lb1_i) and c1_i < lb1_i
            e2_sig = np.isfinite(rs1_i) and rs1_i < RSI_TH
            base = np.where(ta["is_e1"], e1_sig, e2_sig)
            trend_ok = np.isfinite(tr1_i) and c1_i > tr1_i
            ent_m = (~has0) & base & (~ta["filt"] | trend_ok) & ~halted
            if ent_m.any():
                t_i = np.where(is_plc, plc[i], tier[i])
                frac_new = TRANCHE_FRAC * mult[np.arange(r), t_i]
                frac_open = np.where(ent_m, frac_new, frac_open)
                entry_tier = np.where(ent_m, tier[i], entry_tier).astype(np.int8)
                try_fill(ent_m, o_i, a1_i, pc, i, frac_open)
                filled = ent_m & (u > 0) & (entry_i == i)
                if filled.any():
                    np.add.at(n_by_tier, (np.flatnonzero(filled),
                                          t_i[filled]), 1)
        # 4) 재해손절 — 봉내 스탑주문
        if np.isfinite(l_i) and np.isfinite(o_i):
            stop_m = (u > 0) & np.isfinite(stoplvl) & (l_i <= stoplvl)
            if stop_m.any():
                x = np.minimum(o_i, stoplvl)
                pnl = u * x - basis - (fees + u * x * COST_OUT)
                eq = np.where(stop_m, eq + u * x - basis - u * x * COST_OUT, eq)
                if record and stop_m[0]:
                    ledger.append({"entry_i": int(entry_i[0]), "exit_i": int(i),
                                   "tier": int(entry_tier[0]), "pnl": float(pnl[0]),
                                   "notional": float(basis[0]), "k": int(k[0]),
                                   "exit": "stop"})
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
    m = u > 0
    if m.any():
        px = c[~np.isnan(c)][-1]
        close_mask(m, px, n - 1)
    if len(snap_i):
        day_mat[:, -1] = eq
    return {"final_eq": eq, "n_trades": n_trades, "n_wins": n_wins,
            "day_eq": day_mat, "days": days, "time_viol": time_viol,
            "n_by_tier": n_by_tier, "heat_blocked": heat_blocked,
            "ledger": ledger}


# ── 격자 실행 ─────────────────────────────────────────────────────────────
def build_context(data: dict, f8: pd.DataFrame) -> dict:
    """타임프레임·심볼별 등급 배열과 진단을 미리 계산한다."""
    ctx: dict = {}
    for tf, dd in data.items():
        btc_close = dd["BTC"]["close"]
        ctx[tf] = {}
        for s in SYMS:
            df = dd[s]
            cm = condition_matrix(df, s, tf, btc_close, f8[s])
            score = cm.sum(axis=1).astype(np.int16)
            sig = raw_signal_mask(df)
            tier, plc, diag = tier_array(score, sig, df.index, f"{s}|{tf}")
            ctx[tf][s] = {"tier": tier, "plc": plc, "score": score,
                          "sig": sig, "diag": diag, "cond": cm}
    return ctx


def run_grid(data: dict, fund: pd.DataFrame, ctx: dict, trials: list[Trial],
             causal: bool = True, progress: bool = True) -> dict[str, np.ndarray]:
    """전 시행 × 3심볼 실행 → 일수익률 행렬과 요약 배열."""
    n = len(trials)
    per_sym = {s: np.zeros((n, N_DAYS)) for s in SYMS}
    n_trades = np.zeros((n, len(SYMS)), dtype=np.int64)
    n_wins = np.zeros((n, len(SYMS)), dtype=np.int64)
    final_eq = np.ones((n, len(SYMS)))
    time_viol = np.zeros(n, dtype=np.int64)
    n_by_tier = np.zeros((n, 3), dtype=np.int64)
    heat_blocked = np.zeros(n, dtype=np.int64)
    for tf in TFS:
        rows = [i for i, t in enumerate(trials) if t.tf == tf]
        if not rows:
            continue
        ta = trial_arrays([trials[i] for i in rows])
        for j, s in enumerate(SYMS):
            t0 = time.time()
            cc = ctx[tf][s]
            res = simulate_sleeve(data[tf][s], fund[s], ta, cc["tier"], cc["plc"],
                                  causal=causal)
            per_sym[s][rows] = ads.sleeve_returns(res["day_eq"], res["days"])
            n_trades[rows, j] = res["n_trades"]
            n_wins[rows, j] = res["n_wins"]
            final_eq[rows, j] = res["final_eq"]
            time_viol[rows] += res["time_viol"]
            n_by_tier[rows] += res["n_by_tier"]
            heat_blocked[rows] += res["heat_blocked"]
            if progress:
                logger.info("%s %s: %d 시행 %.1fs", tf, s, len(rows), time.time() - t0)
    out = {"ret_combined": sum(per_sym[s] for s in SYMS) / len(SYMS),
           "n_trades": n_trades, "n_wins": n_wins, "final_eq": final_eq,
           "time_viol": time_viol, "n_by_tier": n_by_tier,
           "heat_blocked": heat_blocked}
    for s in SYMS:
        out[f"ret_{s}"] = per_sym[s]
    return out


# ── 지표 ──────────────────────────────────────────────────────────────────
def metrics(ret: np.ndarray, days: pd.DatetimeIndex,
            lo: pd.Timestamp | None = None,
            hi: pd.Timestamp | None = None) -> dict[str, np.ndarray]:
    """일수익률 행렬 (N, T) → 구간 지표.

    Args:
        ret: 일수익률. days: 대응 일자 (길이 T). lo/hi: 구간 경계 (포함).

    Returns:
        `cum`, `ann`, `sharpe`, `mdd`, `calmar`, `mdd_norm_cum` (MDD 를 기준선
        수준으로 맞췄을 때의 등가 누적수익 — 레버리지 효과를 제거한 비교용).
    """
    m = np.ones(len(days), dtype=bool)
    if lo is not None:
        m &= np.asarray(days >= lo)
    if hi is not None:
        m &= np.asarray(days <= hi)
    r = ret[:, m]
    t = r.shape[1]
    eq = np.cumprod(1.0 + r, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mdd = (eq / peak - 1.0).min(axis=1)
        sd = r.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, r.mean(axis=1) / sd * np.sqrt(365.0), 0.0)
        cum = eq[:, -1] - 1.0
        ann = np.where(eq[:, -1] > 0, eq[:, -1] ** (365.0 / t) - 1.0, -1.0)
        calmar = np.where(mdd < 0, ann / -mdd, np.nan)
    return {"cum": cum, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "calmar": calmar, "n_days": np.full(len(cum), t)}


def risk_equalised_cum(ret: np.ndarray, mdd: np.ndarray,
                       target_mdd: float) -> np.ndarray:
    """MDD 를 `target_mdd` 로 맞추도록 수익률을 선형 스케일한 등가 누적수익.

    비중 확대는 수익과 낙폭을 함께 키우므로, 크기를 뺀 **정보량** 비교가 필요하다.
    `k = target_mdd / mdd` 로 일수익률을 스케일한 뒤 누적한다 (1차 근사 — 복리·
    청산 비선형성은 반영하지 않는다).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(mdd < 0, target_mdd / mdd, 1.0)
    k = np.clip(k, 0.0, 20.0)
    eq = np.cumprod(1.0 + ret * k[:, None], axis=1)
    return eq[:, -1] - 1.0


# ── 자가검증 ──────────────────────────────────────────────────────────────
def selftest() -> None:
    """AVGDOWN 원 엔진과의 동치성·인과성·위약 대칭성을 강제한다.

    (a) `conv='flat'` 15m 전 구조 624 시행이 `avgdown_sweep` 과 비트 단위 동일.
    (b) 조건 행렬이 확정봉 지연을 지킨다 (마지막 봉 조건이 미래를 안 본다).
    (c) 위약 등급의 등급별 빈도가 실제 등급과 같다.
    (d) 1h 재집계 충실도 보고 (BTC 재집계 vs 동결 1h).

    Raises:
        AssertionError: 어떤 불변식이라도 깨질 때.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data, fund, f8 = load_data()
    ctx = build_context(data, f8)

    # (d) 1h 재집계 충실도
    perp = pd.read_parquet(ROOT / "lab/frozen/perp_1h.parquet")
    btc_f = perp.xs("BTC", level="sym")[["open", "high", "low", "close"]]
    btc_r = _resample_1h(data["15m"]["BTC"])[["open", "high", "low", "close"]]
    j = btc_f.join(btc_r, how="inner", lsuffix="_f", rsuffix="_r")
    rel = (j["close_f"] - j["close_r"]).abs() / j["close_f"].abs()
    logger.info("(d) 1h 재집계 충실도 (BTC, n=%d): 평균 상대차 %.3e · 최대 %.3e",
                len(j), rel.mean(), rel.max())

    # (a) 동치성
    ladders: list[tuple[float | None, int]] = [(None, 0)]
    ladders += [(sp, k) for sp in SPACINGS for k in KMAXES if k > 0]
    mine = [Trial(e, f, sp, k, tp, sl, "flat", "15m")
            for e in ENTRIES for f in FILTERS for sp, k in ladders
            for tp in TPS for sl in STOPS]
    theirs = [t.base() for t in mine]
    logger.info("(a) 동치성 검증 — flat 15m %d 시행", len(mine))
    got = run_grid(data, fund, ctx, mine, progress=False)
    ref = ads.run_grid({"15m": data["15m"]}, fund, theirs, progress=False)
    for key in ("ret_combined", "n_trades", "n_wins", "final_eq"):
        assert np.array_equal(got[key], ref[key], equal_nan=True), \
            f"(a) 동치성 위반: {key} 가 avgdown_sweep 과 다르다"
    logger.info("(a) OK — flat 설정이 avgdown_sweep 과 비트 단위 동일")

    # (b) 인과성 — 조건 행렬은 [i-1] 지연이므로 첫 행이 전부 False
    for tf in TFS:
        for s in SYMS:
            cm = ctx[tf][s]["cond"]
            assert not cm[0].any(), f"(b) {tf}/{s}: 첫 봉 조건이 참 — 지연 누락"
    # 마지막 봉 이후 데이터를 잘라도 앞쪽 조건이 변하지 않아야 한다
    df = data["15m"]["BTC"]
    cut = len(df) - 500
    full = condition_matrix(df, "BTC", "15m", df["close"], f8["BTC"])
    part = condition_matrix(df.iloc[:cut], "BTC", "15m",
                            df["close"].iloc[:cut], f8["BTC"])
    tail = 5 * BARS_PER_DAY["15m"]          # 롤링 창 재시작 영향 구간은 제외
    assert np.array_equal(full[:cut - 30 * BARS_PER_DAY["15m"]],
                          part[:cut - 30 * BARS_PER_DAY["15m"]]), \
        "(b) 미래 데이터를 자르자 과거 조건이 변했다 — 룩어헤드"
    logger.info("(b) OK — 조건 행렬 인과성 (tail=%d 봉 롤링 창 제외)", tail)

    # (c) 위약 대칭성
    for tf in TFS:
        for s in SYMS:
            cc = ctx[tf][s]
            sig = cc["sig"]
            a = np.bincount(cc["tier"][sig], minlength=3)
            b = np.bincount(cc["plc"][sig], minlength=3)
            assert np.array_equal(a, b), f"(c) {tf}/{s}: 위약 등급 빈도 불일치 {a} vs {b}"
    logger.info("(c) OK — 위약 등급이 실제 등급과 빈도 동일")

    for tf in TFS:
        for s in SYMS:
            d = ctx[tf][s]["diag"]
            logger.info("    %s/%s  IS원신호 %5d · thrA %.1f · thrA+ %.1f · "
                        "빈도 B/A/A+ %.2f/%.2f/%.2f · 평균점수 %.2f",
                        tf, s, int(d["n_is_signals"]), d.get("thr_A", float("nan")),
                        d.get("thr_Ap", float("nan")), d.get("freq_B", float("nan")),
                        d.get("freq_A", float("nan")), d.get("freq_Ap", float("nan")),
                        d.get("mean_score_signals", float("nan")))
    logger.info("자가검증 전부 통과")


# ── 본 실행 ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    ap = argparse.ArgumentParser(description=f"{SPEC} 컨플루언스 확신도 사이징 튜닝")
    ap.add_argument("--selftest", action="store_true", help="자가검증만 수행")
    ap.add_argument("--run", action="store_true", help="본 격자 실행")
    ap.add_argument("--batch", type=int, default=5000, help="시행 배치 크기")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("--selftest 또는 --run 중 하나가 필요하다")

    t_start = time.time()
    data, fund, f8 = load_data()
    ctx = build_context(data, f8)
    trials = enumerate_trials()
    logger.info("%s — %d 시행 (구조 624 × 확신도 %d × tf %d)",
                SPEC, len(trials), len(CONV_RULES), len(TFS))

    grid = ads.master_days()[1:]
    chunks = [trials[i:i + a.batch] for i in range(0, len(trials), a.batch)]
    parts: list[dict] = []
    for ci, ch in enumerate(chunks):
        logger.info("배치 %d/%d — %d 시행", ci + 1, len(chunks), len(ch))
        parts.append(run_grid(data, fund, ctx, ch))
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}

    base_mdd = float(metrics(out["ret_combined"][[trials.index(
        Trial("E2", 1, 1.0, 3, "A2.0", None, "flat", "15m"))]], grid)["mdd"][0])
    full = metrics(out["ret_combined"], grid)
    ins = metrics(out["ret_combined"], grid, hi=IS_END)
    oos = metrics(out["ret_combined"], grid, lo=IS_END)
    req = risk_equalised_cum(out["ret_combined"], full["mdd"], base_mdd)

    rows = []
    for i, t in enumerate(trials):
        rows.append({
            "trial_id": t.tid(), "tf": t.tf, "entry": t.entry, "filt": t.filt,
            "spacing": t.spacing, "kmax": t.kmax, "tp": t.tp, "stop": t.stop,
            "conv": t.conv,
            "cum": full["cum"][i], "ann": full["ann"][i],
            "sharpe": full["sharpe"][i], "mdd": full["mdd"][i],
            "calmar": full["calmar"][i], "req_cum": req[i],
            "is_cum": ins["cum"][i], "is_sharpe": ins["sharpe"][i],
            "is_mdd": ins["mdd"][i],
            "oos_cum": oos["cum"][i], "oos_sharpe": oos["sharpe"][i],
            "oos_mdd": oos["mdd"][i],
            "n_trades": int(out["n_trades"][i].sum()),
            "n_wins": int(out["n_wins"][i].sum()),
            "n_B": int(out["n_by_tier"][i, 0]), "n_A": int(out["n_by_tier"][i, 1]),
            "n_Ap": int(out["n_by_tier"][i, 2]),
            "heat_blocked": int(out["heat_blocked"][i]),
            "time_viol": int(out["time_viol"][i]),
        })
    df = pd.DataFrame(rows)
    df["win_rate"] = np.where(df.n_trades > 0, df.n_wins / df.n_trades, np.nan)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "conf_tune_summary.csv", index=False)
    npz_path = OUT_DIR / "conf_tune_returns.npz"
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
        "returns_npz_sha256": h.hexdigest(),
        "spec": SPEC, "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": len(trials), "conv_rules": {k: list(v) for k, v in CONV_RULES.items()},
        "conditions": [c[0] for c in CONDITIONS],
        "is_end": str(IS_END), "tier_q": [TIER_Q_A, TIER_Q_AP],
        "placebo_seed": PLACEBO_SEED, "baseline_mdd": base_mdd,
        "tier_diag": {f"{tf}|{s}": ctx[tf][s]["diag"] for tf in TFS for s in SYMS},
        "elapsed_s": time.time() - t_start,
    }
    (OUT_DIR / "conf_tune_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    assert int(df.time_viol.sum()) == 0, "시간 인과성 위반 발생 — 결과 폐기"
    logger.info("완료 %.1fs → logs/conf_tune_summary.csv", time.time() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
