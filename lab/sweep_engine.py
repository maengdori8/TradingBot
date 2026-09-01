"""대규모 지표 스윕 백테스트 엔진 — SWEEP-2026-08-31 사전등록 구현.

명세: `docs/PREREGISTRATION_SWEEP_2026-08-31.md` (§3 규칙 공간 · §4 평가 프로토콜 ·
§5 지표 · §11.3 동결 상수). 본 파일은 **엔진**만 담당한다. 다중검정 보정(White RC ·
Romano–Wolf StepM · DSR · SPA)은 별도 판정 스크립트(`lab/sweep_rc.py`)의 몫이다.

구조
----
1. `enumerate_rules()`  — §3 격자를 **파라미터에서 생성**한다(하드코딩 목록 아님).
   1,695 규칙형 × 2 타임프레임 = 3,390 시행. 총계는 실행 시 출력·검증한다.
2. `Feat`              — 심볼×타임프레임별 지표 1회 계산 캐시.
3. `_build_*`          — 규칙이 참조하는 "변형(variant) 열"을 만든다. 168 진입 변형 ·
   ~96 신호청산 변형 · 동적 스탑/목표 변형을 **전 규칙이 공유**하므로 지표 재계산이 없다.
4. `simulate_timeframe` — 봉 루프 1회, 규칙 축(R=1,695)을 numpy 로 동시 처리.
   상태 배열 `cash(R)`, `pdir(R,3)`, … 로 셀 하나하나를 벡터화해서 굴린다.
5. `main`              — 두 타임프레임을 돌려 (3,390 × 1,737) 일수익률 행렬과 요약 CSV 를
   기록한다. 입력 SHA256·seed·라이브러리 버전을 함께 남겨 재현 가능하게 한다.

실행 인과성 (위반 시 결과 폐기 — §4.4)
------------------------------------
* 모든 지표·조건은 확정봉 `[i−1]` 이하만 사용한다. `ATR` 은 항상 `ATR[i−1]`.
  유일한 동일봉 데이터 사용은 (a) 체결가로서의 `open[i]`, (b) 봉내 트리거 판정용
  `high[i]/low[i]`, (c) V-A 의 `ref = 봉 [i] 시가` (§3.4 가 명시적으로 허용) 뿐이다.
  `close[i]` 는 **봉 마감 후 평가**(일손실 정지 판정·MTM)에만 쓰이며 어떤 체결가도
  되지 않는다.
* TR = previous close 기준. 4h 봉은 닫힌 1h 봉 4개가 전부 있을 때만 확정.
* 비돌파형 체결 = 봉 `[i]` 시가. 돌파형 = 봉내 스탑주문, `fill = max(level, open)` (롱).
* 워밍업 구간(평가 창 이전)에는 주문을 생성하지 않는다.

명세가 침묵해 본 엔진이 결정론적으로 확정한 사항 (전부 결과 조회 전 고정)
--------------------------------------------------------------------
D1. **심볼 처리 순서 = BTC → ETH → SOL** (§4.1 표 순서). 그로스 캡·heat 캡은 같은 봉
    복수 심볼 진입 시 순서 의존적이므로 고정이 필요하다.
D2. **보유 중 신규 진입 신호는 무시한다** (Track E §9 이식 대조 #9 "보유 중 신규 신호
    무시"). §4.4-11 의 "반대 신호 시 청산 후 다음 봉 재진입"은 규칙 자신의 청산 조건
    (대부분 X1 = 반대 신호)이 발동해 청산된 뒤 **같은 봉 재진입을 금지**한다는 뜻으로
    구현한다. 반대 방향 진입신호가 청산을 강제하지는 않는다.
D3. **보유봉수 청산 체결 시점** = `i − 체결봉 == H` 인 봉의 **시가**. 체결봉을 1봉으로
    세어 정확히 H 봉을 보유한 뒤 다음 봉 시가에 나간다(§3.0 "체결봉 = 1" + §4.4-3
    "신호봉 종가 체결 금지"의 보수적 결합).
D4. **봉내 사건 순서** = 시가 체결 청산(신호청산·보유봉수) → 스탑(갭 악화) → 목표(레벨).
    시가가 시간상 먼저이므로 시가 청산이 우선한다. 체결봉에서는 진입 후 그 봉의
    고가/저가 전체로 스탑·목표를 검사한다(경로 미상 → 비관, §4.4-10).
D5. **동적 레벨의 시점** = 봉 `i` 동안 유효한 레벨은 `[i−1]` 까지의 데이터로 만든 값
    (돈치안 `LL_M[i] = min(low[i−M..i−1])`, 밴드는 `band[i−1]`). V-A 의 반대 밴드만
    `ref` 정의상 `open[i]` 를 쓸 수 있다.
D6. **사이징의 "가장 가까운 확정 스탑 레벨"** = 진입 시점에 확정 가능한 청산 레벨 중
    **체결가의 불리한 쪽**(롱이면 아래)에 있는 것만 후보로 하고 그중 가장 가까운 값.
    유리한 쪽 레벨(평균회귀의 반대 밴드, 1R/2R 목표)은 손실을 결정하지 않으므로
    스탑이 아니다 → 그런 규칙은 §4.5 의 명목 1/3 사이징을 쓴다.
D7. **자정 스냅샷과 펀딩의 순서** = 스냅샷(직전 확정봉 종가 MTM) 후 펀딩 정산.
    00:00 UTC 펀딩은 새 날에 귀속된다.
D8. **규칙 ID 파라미터 구분자** = `,` (§3.6 은 "키=값 …" 으로만 규정). 실수 표기는
    `f"{v:g}"`.
D9. **V-C `D2`** = 해제 봉 다음 봉부터 12봉 안에 처음 발생한 `Donchian(12)` 돌파를
    **봉내 스탑주문**으로 체결한다(§4.4-3 이 "V-C 의 D2 방향 판정"을 돌파형으로 열거).
    미발생 시 신호 폐기. 무장은 새 해제가 오면 갱신된다.
D10. **자본 소진 방어** = 어떤 날의 시작 자본이 0 이하면 그날 수익률을 0 으로 둔다
    (퇴화 방어. 사후 필터가 아니라 산술 정의).
D11. **Q-B 피벗 확정 1봉 지연** — 명세 §3.5 문언과의 **의도적 편차**. 원전
    (`lab/rsi_divergence_test.py`)은 신호봉 **종가** 체결이라 `[j−K, j+K]` 창에 봉
    `[i]` 를 포함해도 인과적이었으나, 본 스윕은 §4.4-3 에 따라 **시가** 체결이므로
    같은 창을 쓰면 동일봉 룩어헤드가 된다. `j = i−K−1` 로 확정을 1봉 늦춰 창을
    `[i−1]` 이하로 닫는다 (`_qb_signals` docstring 참조). §4.4 는 위반 시 결과
    폐기를 명령하는 상위 조항이므로 §3.5 문언보다 우선한다. 사전등록 동결 커밋
    전에 §3.5 에 이 교정을 반영하는 것이 바람직하다.

D12. **사이징 자본의 마킹** = `cash + Σ 미실현(직전 확정 종가 `close[i−1]` 기준)`.
    단 **같은 봉에 막 체결된 포지션은 체결가로 마킹**한다 — 과거 종가로 마킹하면
    존재하지 않는 평가손익이 생겨 뒤 심볼의 사이징이 오염된다. 그로스 명목도 동일.
    (`close[i]` 를 쓰면 동일봉 룩어헤드가 되므로 절대 금지.)

구조적 성질 (버그 아님 — 결과 해석 시 유의)
* 심볼 3종 × 심볼당 1포지션이므로 `MAX_POS = 3` 은 사실상 구속력이 없다.
* heat 캡도 스탑 포지션 3개 = 정확히 6% 라 **진입 시점 자본 기준으로는** 구속력이
  없다. 실제 구속은 진입 이후 자본이 하락해 기존 포지션의 고정 리스크 금액 비중이
  커졌을 때뿐이다 (`test_heat_cap_gates_entries` 가 게이트 배선을 검증한다).
* 반면 그로스 캡 10x 는 스탑이 좁은 규칙에서 실제로 자주 구속한다.

주의: 본 모듈은 **실행하면 결과가 생성된다**. §9.1 에 따라 스윕은 코드·문서 커밋·태그
후 1회만 실행한다. 개발 중 검증은 전부 합성 데이터로 한다(`tests/test_sweep_engine.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 동결 상수 (§11.3 — 변경 금지) ──────────────────────────────────────────
N_RULES: int = 1695
N_TRIALS: int = 3390
SEED: int = 20260831
RT_COST: float = 0.0016
COST_SIDE: float = 0.0008
RISK: float = 0.02
STOPLESS_NOTIONAL: float = 1.0 / 3.0
STOPLESS_HEAT: float = 0.05
GROSS_CAP: float = 10.0
MAX_POS: int = 3
HEAT_CAP: float = 0.06
DAILY_HALT: float = -0.05
CELL_CAPITAL: float = 10_000.0
WIN_START: pd.Timestamp = pd.Timestamp("2021-11-21T00:00:00Z")
WIN_END: pd.Timestamp = pd.Timestamp("2026-08-24T00:00:00Z")
IS_END: pd.Timestamp = pd.Timestamp("2024-12-31T23:59:00Z")
N_DAYS: int = 1737

SYMBOLS: tuple[str, ...] = ("BTC", "ETH", "SOL")
TIMEFRAMES: tuple[str, ...] = ("1h", "4h")
FUNDING_HOURS: tuple[int, ...] = (0, 8, 16)
DIRECTIONS: tuple[str, ...] = ("L", "S", "LS")

REPO_ROOT = Path(__file__).resolve().parent.parent
PERP_PATH = REPO_ROOT / "lab" / "frozen" / "perp_1h.parquet"
FUND_PATH = REPO_ROOT / "lab" / "frozen" / "funding.parquet"
SOL_PATH = REPO_ROOT / "lab" / "data" / "sol_1h.parquet"

# §11.2 실측 해시 — 불일치 시 fail-closed
EXPECTED_SHA256: dict[str, str] = {
    "lab/frozen/perp_1h.parquet": "c06a3301457dfec8f68b184e1f8ac8797acc49874fa96c409a6a57e6743ebac0",
    "lab/frozen/funding.parquet": "534642ee677424abb949492a7b8f21e43ea635bd0eb9f980e12f829f89a0a128",
    "lab/data/sol_1h.parquet": "80d7f7574d680505eb280d05bfecc4ce3bc38ff461e651b91eb122e666f04785",
}


# ── 유틸 ──────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    """파일의 SHA256 16진 문자열을 반환한다.

    Args:
        path: 대상 파일 경로.

    Returns:
        소문자 16진 SHA256.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(a: np.ndarray, k: int) -> np.ndarray:
    """배열을 k 봉 뒤로 민다 — `sh(a, k)[i] == a[i-k]` (앞쪽은 NaN/False).

    확정봉 규칙(§4.4-1)을 코드 수준에서 강제하는 유일한 도구다. 지표 원값을
    직접 인덱싱하는 코드는 이 모듈에 존재해서는 안 된다.

    Args:
        a: 원 배열 (float 또는 bool).
        k: 밀 봉 수 (k >= 0).

    Returns:
        같은 길이·같은 dtype 의 시프트 배열.
    """
    if k == 0:
        return a.copy()
    out = np.empty_like(a)
    if a.dtype == np.bool_:
        out[:k] = False
    else:
        out[:k] = np.nan
    out[k:] = a[:-k]
    return out


def _series(a: np.ndarray) -> pd.Series:
    """numpy 배열을 rolling/ewm 용 pandas Series 로 감싼다."""
    return pd.Series(a, copy=False)


# ── 지표 캐시 ─────────────────────────────────────────────────────────────
class Feat:
    """심볼 × 타임프레임 1개의 지표 캐시.

    모든 메서드는 **시프트하지 않은 원값**을 반환한다. 확정봉 규칙은 호출부에서
    `sh(...)` 로 적용한다 (이 분리가 인과성 감사를 가능하게 한다).
    """

    def __init__(self, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                 c: np.ndarray, v: np.ndarray) -> None:
        """OHLCV 배열로 캐시를 만든다.

        Args:
            o: 시가. h: 고가. l: 저가. c: 종가. v: 거래량. 전부 float64 1-D.
        """
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.n = len(c)
        self._cache: dict[Any, Any] = {}

    def _memo(self, key: Any, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # 기본 이동평균/편차
    def sma(self, n: int) -> np.ndarray:
        """단순이동평균 SMA_n."""
        return self._memo(("sma", n), lambda: _series(self.c).rolling(n).mean().to_numpy())

    def ema(self, n: int) -> np.ndarray:
        """지수이동평균 EMA_n (adjust=False, alpha=2/(n+1))."""
        return self._memo(("ema", n),
                          lambda: _series(self.c).ewm(span=n, adjust=False).mean().to_numpy())

    def ma(self, kind: str, n: int) -> np.ndarray:
        """`kind` 가 'SMA'/'EMA' 인 이동평균."""
        return self.sma(n) if kind == "SMA" else self.ema(n)

    def std(self, n: int) -> np.ndarray:
        """종가 표본표준편차 (ddof=0)."""
        return self._memo(("std", n),
                          lambda: _series(self.c).rolling(n).std(ddof=0).to_numpy())

    # 변동성
    def tr(self) -> np.ndarray:
        """True Range — **previous close 기준** (§4.4-2). `fmax` = 저장소 skipna 관례."""
        def _f() -> np.ndarray:
            pc = sh(self.c, 1)
            return np.fmax(self.h - self.l,
                           np.fmax(np.abs(self.h - pc), np.abs(self.l - pc)))
        return self._memo(("tr",), _f)

    def atr(self, n: int) -> np.ndarray:
        """Wilder ATR_n (`ewm(alpha=1/n, adjust=False)`)."""
        def _f() -> np.ndarray:
            tr = self.tr().copy()
            tr[0] = np.nan  # prev close 없음 → 확정 불가
            return _series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
        return self._memo(("atr", n), _f)

    # 모멘텀
    def rsi(self, n: int) -> np.ndarray:
        """Wilder RSI_n. `dn==0 → 100`, 무변동 → 50 (`published_systems_test` 관례)."""
        def _f() -> np.ndarray:
            d = np.empty(self.n)
            d[0] = np.nan
            d[1:] = np.diff(self.c)
            up = _series(np.clip(d, 0, None)).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
            dn = _series(np.clip(-d, 0, None)).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                out = 100.0 - 100.0 / (1.0 + up / dn)
            flat = (dn == 0)
            out = np.where(flat & (up == 0), 50.0, out)
            out = np.where(flat & (up > 0), 100.0, out)
            out[~np.isfinite(up) | ~np.isfinite(dn)] = np.nan
            return out
        return self._memo(("rsi", n), _f)

    def macd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD 12/26/9 → (macd, signal, hist). 표준 고정, 격자 없음 (§3.2 M-B)."""
        def _f() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            m = self.ema(12) - self.ema(26)
            s = _series(m).ewm(span=9, adjust=False).mean().to_numpy()
            return m, s, m - s
        return self._memo(("macd",), _f)

    def stoch(self) -> tuple[np.ndarray, np.ndarray]:
        """스토캐스틱 14/3 → (%K slow, %D). `HH14−LL14 == 0` 은 NaN (fail-closed)."""
        def _f() -> tuple[np.ndarray, np.ndarray]:
            hh = _series(self.h).rolling(14).max().to_numpy()
            ll = _series(self.l).rolling(14).min().to_numpy()
            rng = hh - ll
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(rng > 0, 100.0 * (self.c - ll) / rng, np.nan)
            k = _series(raw).rolling(3).mean().to_numpy()
            d = _series(k).rolling(3).mean().to_numpy()
            return k, d
        return self._memo(("stoch",), _f)

    # 밴드/채널
    def bb(self, n: int, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """볼린저 밴드 → (upper, mid, lower). 중심 SMA_n, 폭 k×std_n(ddof=0)."""
        def _f() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            m, s = self.sma(n), self.std(n)
            return m + k * s, m, m - k * s
        return self._memo(("bb", n, k), _f)

    def kc(self, p: int, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """켈트너 채널 → (upper, mid, lower). 중심 EMA20, 폭 m×ATR_p."""
        def _f() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            e, a = self.ema(20), self.atr(p)
            return e + m * a, e, e - m * a
        return self._memo(("kc", p, m), _f)

    def z(self, n: int) -> np.ndarray:
        """z-score = (close − SMA_n) / std_n."""
        def _f() -> np.ndarray:
            s = self.std(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(s > 0, (self.c - self.sma(n)) / s, np.nan)
        return self._memo(("z", n), _f)

    def hh(self, n: int) -> np.ndarray:
        """돈치안 상단 `HH_N[i] = max(high[i−N..i−1])` (shift 1 포함)."""
        return self._memo(("hh", n),
                          lambda: sh(_series(self.h).rolling(n).max().to_numpy(), 1))

    def ll(self, n: int) -> np.ndarray:
        """돈치안 하단 `LL_N[i] = min(low[i−N..i−1])` (shift 1 포함)."""
        return self._memo(("ll", n),
                          lambda: sh(_series(self.l).rolling(n).min().to_numpy(), 1))

    # 거래량
    def obv(self) -> np.ndarray:
        """Granville OBV 누적 (`close>prev → +vol`, `<prev → −vol`, `== → 0`)."""
        def _f() -> np.ndarray:
            pc = sh(self.c, 1)
            step = np.where(self.c > pc, self.v, np.where(self.c < pc, -self.v, 0.0))
            step = np.where(np.isfinite(step), step, 0.0)
            out = np.cumsum(step)
            out[~np.isfinite(self.c)] = np.nan
            return out
        return self._memo(("obv",), _f)

    def sma_of(self, x: np.ndarray, n: int, tag: str) -> np.ndarray:
        """임의 배열의 SMA_n (`tag` 로 캐시 구분)."""
        return self._memo(("smaof", tag, n),
                          lambda: _series(x).rolling(n).mean().to_numpy())

    def vol_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """거래량 서지용 `mean/std(vol[i−21..i−2])` (ddof=0) — 이미 시프트 완료."""
        def _f() -> tuple[np.ndarray, np.ndarray]:
            s = _series(self.v)
            m = sh(s.rolling(20).mean().to_numpy(), 2)
            d = sh(s.rolling(20).std(ddof=0).to_numpy(), 2)
            return m, d
        return self._memo(("volstats",), _f)

    def surge(self, g: str) -> np.ndarray:
        """서지 게이트 `G1..G6` (§3.5 Q-A). 전부 확정봉 `[i−1]` 기준."""
        def _f() -> np.ndarray:
            v1 = sh(self.v, 1)
            m20, s20 = self.vol_stats()
            with np.errstate(divide="ignore", invalid="ignore"):
                if g in _SURGE_Q:
                    out = v1 > _SURGE_Q[g] * m20
                else:
                    out = np.where(s20 > 0, (v1 - m20) / s20, np.nan) > _SURGE_Z[g]
            return np.asarray(out, dtype=bool) & np.isfinite(m20) & np.isfinite(v1)
        return self._memo(("surge", g), _f)

    def squeeze(self, code: str) -> tuple[np.ndarray, np.ndarray]:
        """스퀴즈 상태 `S1..S7` → (on, valid). `valid=False` 구간은 판정 불가."""
        def _f() -> tuple[np.ndarray, np.ndarray]:
            bu, bm, bl = self.bb(20, 2.0)
            if code in _SQUEEZE_TTM:
                ku, _, kl = self.kc(20, _SQUEEZE_TTM[code])
                valid = np.isfinite(bu) & np.isfinite(ku)
                on = np.asarray((bu < ku) & (bl > kl), dtype=bool) & valid
                return on, valid
            q, lb = _SQUEEZE_PCT[code]
            with np.errstate(divide="ignore", invalid="ignore"):
                bw = np.where(bm > 0, (bu - bl) / bm, np.nan)
            thr = sh(_series(bw).rolling(lb).quantile(q).to_numpy(), 1)
            valid = np.isfinite(bw) & np.isfinite(thr)
            on = np.asarray(bw <= thr, dtype=bool) & valid
            return on, valid
        return self._memo(("sq", code), _f)


# ── §3 격자 상수 ──────────────────────────────────────────────────────────
TA_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 20), (5, 50), (5, 100), (5, 200), (10, 20), (10, 50), (10, 100), (10, 200),
    (20, 50), (20, 100), (20, 200), (50, 100), (50, 200),
)
MA_TYPES: tuple[str, ...] = ("SMA", "EMA")
TB_N: tuple[int, ...] = (12, 24, 48, 96, 192)
TC_N: tuple[int, ...] = (20, 50, 100, 200)
MA_RSI_N: tuple[int, ...] = (2, 7, 14)
MA_U: tuple[int, ...] = (50, 60, 70, 80)
RA_NK: tuple[tuple[int, float], ...] = ((10, 1.9), (20, 2.0), (50, 2.1))
RB_LB: tuple[int, ...] = (20, 50)
RB_TH: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
RC_N: tuple[int, ...] = (2, 7, 14)
RC_T: tuple[int, ...] = (5, 10, 20, 30)
VA_M: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
VA_REF: tuple[str, ...] = ("open", "prevclose")
VB_P: tuple[int, ...] = (10, 20)
VB_M: tuple[float, ...] = (1.5, 2.0, 2.5)
VC_SQ: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
QA_G: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")
QA_A: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5")
QB_K: tuple[int, ...] = (2, 3)
QB_S: tuple[int, ...] = (20, 50)
QC_LEN: tuple[int, ...] = (20, 50, 100)

_SURGE_Q: dict[str, float] = {"G1": 1.0, "G2": 1.5, "G3": 2.0, "G4": 3.0}
_SURGE_Z: dict[str, float] = {"G5": 1.0, "G6": 2.0}
_SQUEEZE_TTM: dict[str, float] = {"S1": 1.0, "S2": 1.5, "S3": 2.0}
_SQUEEZE_PCT: dict[str, tuple[float, int]] = {
    "S4": (0.10, 100), "S5": (0.10, 200), "S6": (0.20, 100), "S7": (0.20, 200),
}
# Q-A 앵커 (§3.5 표 — 동결)
_ANCHOR_ENTRY: dict[str, tuple] = {
    "A1": ("TB", 24),
    "A2": ("TA", "EMA", 20, 50),
    "A3": ("MA", 14, 60),
    "A4": ("RA", 20, 2.0, 1),
    "A5": ("VA", 1.0, "open"),
}


# ── 규칙 명세 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExitSpec:
    """규칙 1개의 청산 구성요소 (모두 OR 결합 — 먼저 닿는 것이 청산)."""

    sigx: tuple | None = None          # 신호청산 변형 키 (확정봉 조건 → 다음 봉 시가)
    dyn_stop: tuple | None = None      # 매 봉 갱신되는 불리한 쪽 레벨 (갭 악화)
    dyn_tgt: tuple | None = None       # 매 봉 갱신되는 유리한 쪽 레벨 (레벨 체결)
    atr_n: int = 0                     # 체결가 기준 고정 ATR 스탑 (0 = 없음)
    atr_mult: float = 0.0
    hold: int = 0                      # 보유봉수 청산 (0 = 없음)
    tgt_r: float = 0.0                 # R 배수 목표 (Q-B 전용, 0 = 없음)
    ent_stop: bool = False             # 진입 시점 확정 고정 스탑 사용 (Q-B 피벗)


@dataclass(frozen=True)
class RuleSpec:
    """규칙형 1개 (타임프레임 제외). `rid(tf)` 로 §3.6 규칙 ID 를 만든다."""

    family: str
    params: tuple[tuple[str, Any], ...]
    exit_code: str
    direction: str
    entry: tuple
    exits: ExitSpec

    def rid(self, tf: str) -> str:
        """`{계열}-{하위}|{키=값,…}|{청산코드}|{방향}|{타임프레임}` 문자열."""
        ps = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in self.params)
        return f"{self.family}|{ps}|{self.exit_code}|{self.direction}|{tf}"


def _atr_exit(sigx: tuple | None, n: int, mult: float) -> ExitSpec:
    return ExitSpec(sigx=sigx, atr_n=n, atr_mult=mult)


def enumerate_rules() -> list[RuleSpec]:
    """§3 의 격자를 그대로 열거해 1,695 규칙형을 생성한다.

    하드코딩된 목록이 아니라 `TA_PAIRS`/`TB_N`/… 격자 상수의 곱집합에서 생성한다.
    열거 순서는 §3.6 표 순서(T→M→R→V→Q)로 결정론적이다.

    Returns:
        길이 1,695 의 `RuleSpec` 리스트.

    Raises:
        AssertionError: 총계가 1,695 가 아닐 때 (동결 상수 위반).
    """
    out: list[RuleSpec] = []

    def emit(family: str, params: tuple, exit_code: str, entry: tuple, ex: ExitSpec) -> None:
        for d in DIRECTIONS:
            out.append(RuleSpec(family, params, exit_code, d, entry, ex))

    # ── T-A 이동평균 교차 (2 × 13 × 3 × 3 = 234)
    for mt in MA_TYPES:
        for f, s in TA_PAIRS:
            entry = ("TA", mt, f, s)
            x1 = ("TA_X1", mt, f, s)
            p = (("ma", mt), ("fast", f), ("slow", s))
            emit("T-A", p, "X1", entry, ExitSpec(sigx=x1))
            emit("T-A", p, "X2", entry, _atr_exit(x1, 14, 2.0))
            emit("T-A", p, "X3", entry, _atr_exit(x1, 14, 3.0))

    # ── T-B 돈치안 돌파 (5 × 7 × 3 = 105)
    for n in TB_N:
        entry = ("TB", n)
        p = (("N", n),)
        half = ("TB_OPP", n // 2)
        full = ("TB_OPP", n)
        emit("T-B", p, "X1", entry, ExitSpec(dyn_stop=half))
        emit("T-B", p, "X2", entry, ExitSpec(dyn_stop=full))
        emit("T-B", p, "X3", entry, ExitSpec(atr_n=24, atr_mult=2.0))
        emit("T-B", p, "X4", entry, ExitSpec(atr_n=24, atr_mult=3.0))
        emit("T-B", p, "X5", entry, ExitSpec(atr_n=24, atr_mult=6.0))
        emit("T-B", p, "X6", entry, ExitSpec(dyn_stop=half, atr_n=24, atr_mult=6.0))
        emit("T-B", p, "X7", entry, ExitSpec(dyn_stop=half, atr_n=24, atr_mult=3.0))

    # ── T-C 가격 대 이동평균 (2 × 4 × 4 × 3 = 96)
    for mt in MA_TYPES:
        for n in TC_N:
            entry = ("TC", mt, n)
            x1 = ("TC_X1", mt, n)
            p = (("ma", mt), ("N", n))
            emit("T-C", p, "X1", entry, ExitSpec(sigx=x1))
            emit("T-C", p, "X2", entry, _atr_exit(x1, 14, 3.0))
            emit("T-C", p, "X3", entry, ExitSpec(sigx=x1, hold=24))
            emit("T-C", p, "X4", entry, ExitSpec(sigx=x1, hold=48))

    # ── M-A RSI 임계 돌파 (3 × 4 × 5 × 3 = 180)
    for n in MA_RSI_N:
        for u in MA_U:
            entry = ("MA", n, u)
            x1 = ("MA_X1", n)
            p = (("n", n), ("U", u))
            emit("M-A", p, "X1", entry, ExitSpec(sigx=x1))
            emit("M-A", p, "X2", entry, ExitSpec(sigx=("MA_X2", n, u)))
            emit("M-A", p, "X3", entry, ExitSpec(hold=12))
            emit("M-A", p, "X4", entry, ExitSpec(hold=24))
            emit("M-A", p, "X5", entry, _atr_exit(x1, 14, 3.0))

    # ── M-B MACD 12/26/9 (3 × 4 × 3 = 36)
    for v in (1, 2, 3):
        entry = ("MB", v)
        x1 = ("MB_X1", v)
        p = (("v", f"V{v}"),)
        emit("M-B", p, "X1", entry, ExitSpec(sigx=x1))
        emit("M-B", p, "X2", entry, _atr_exit(x1, 14, 3.0))
        emit("M-B", p, "X3", entry, ExitSpec(sigx=x1, hold=24))
        emit("M-B", p, "X4", entry, ExitSpec(sigx=x1, hold=48))

    # ── M-C 스토캐스틱 14/3 (3 × 4 × 3 = 36)
    for v in (1, 2, 3):
        entry = ("MC", v)
        x1 = ("MC_X1", v)
        p = (("v", f"V{v}"),)
        emit("M-C", p, "X1", entry, ExitSpec(sigx=x1))
        emit("M-C", p, "X2", entry, ExitSpec(sigx=("MC_X2",)))
        emit("M-C", p, "X3", entry, ExitSpec(sigx=("MC_X3",)))
        emit("M-C", p, "X4", entry, _atr_exit(x1, 14, 3.0))

    # ── R-A 볼린저 페이드 (3 × 2 × 4 × 3 = 72)
    for n, k in RA_NK:
        for trig in (1, 2):
            entry = ("RA", n, k, trig)
            x1 = ("RA_X1", n)
            p = (("n", n), ("k", k), ("e", f"E{trig}"))
            emit("R-A", p, "X1", entry, ExitSpec(sigx=x1))
            emit("R-A", p, "X2", entry, ExitSpec(dyn_tgt=("RA_OPP", n, k)))
            emit("R-A", p, "X3", entry, _atr_exit(x1, 14, 3.0))
            emit("R-A", p, "X4", entry, ExitSpec(sigx=x1, hold=24))

    # ── R-B z-score 페이드 (2 × 4 × 5 × 3 = 120)
    for lb in RB_LB:
        for th in RB_TH:
            entry = ("RB", lb, th)
            x1 = ("RB_X1", lb)
            p = (("lb", lb), ("th", th))
            emit("R-B", p, "X1", entry, ExitSpec(sigx=x1))
            emit("R-B", p, "X2", entry, ExitSpec(sigx=("RB_X2", lb)))
            emit("R-B", p, "X3", entry, _atr_exit(x1, 24, 4.0))
            emit("R-B", p, "X4", entry, ExitSpec(sigx=x1, hold=24))
            emit("R-B", p, "X5", entry, ExitSpec(sigx=x1, hold=48))

    # ── R-C RSI 과매도 반등 (3 × 4 × 2 × 4 × 3 = 288)
    for n in RC_N:
        for t in RC_T:
            for filt in (0, 1):
                entry = ("RC", n, t, filt)
                x1 = ("RC_X1",)
                p = (("n", n), ("t", t), ("filt", filt))
                emit("R-C", p, "X1", entry, ExitSpec(sigx=x1))
                emit("R-C", p, "X2", entry, ExitSpec(sigx=("RC_X2", n)))
                emit("R-C", p, "X3", entry, ExitSpec(sigx=("RC_X3", n, t)))
                emit("R-C", p, "X4", entry, ExitSpec(sigx=x1, hold=24))

    # ── V-A ATR 레인지 돌파 (5 × 2 × 5 × 3 = 150)
    for m in VA_M:
        for ref in VA_REF:
            entry = ("VA", m, ref)
            p = (("m", m), ("ref", ref))
            emit("V-A", p, "X1", entry, ExitSpec(dyn_stop=("VA_OPP", m, ref)))
            emit("V-A", p, "X2", entry, ExitSpec(atr_n=14, atr_mult=2.0))
            emit("V-A", p, "X3", entry, ExitSpec(atr_n=14, atr_mult=3.0))
            emit("V-A", p, "X4", entry, ExitSpec(hold=12))
            emit("V-A", p, "X5", entry, ExitSpec(hold=24))

    # ── V-B 켈트너 (2 × 3 × 2 × 3 × 3 = 108)
    for pp in VB_P:
        for m in VB_M:
            for usage in (1, 2):
                entry = ("VB", pp, m, usage)
                x1 = ("VB_X1", usage)
                p = (("p", pp), ("m", m), ("u", f"U{usage}"))
                emit("V-B", p, "X1", entry, ExitSpec(sigx=x1))
                if usage == 1:      # 돌파 → 반대 밴드는 불리한 쪽 (스탑)
                    emit("V-B", p, "X2", entry, ExitSpec(dyn_stop=("VB_OPP", pp, m)))
                else:               # 페이드 → 반대 밴드는 유리한 쪽 (목표)
                    emit("V-B", p, "X2", entry, ExitSpec(dyn_tgt=("VB_OPP", pp, m)))
                emit("V-B", p, "X3", entry, _atr_exit(x1, pp, 3.0))

    # ── V-C 스퀴즈 돌파 (7 × 2 × 3 × 3 = 126)
    for sq in VC_SQ:
        for dm in (1, 2):
            entry = ("VC", sq, dm)
            p = (("sq", sq), ("d", f"D{dm}"))
            emit("V-C", p, "X1", entry, ExitSpec(atr_n=14, atr_mult=2.0, hold=24))
            emit("V-C", p, "X2", entry, ExitSpec(sigx=("VC_X2",)))
            emit("V-C", p, "X3", entry, ExitSpec(hold=12))

    # ── Q-A 서지 × 앵커 (6 × 5 × 3 = 90)
    anchor_exit: dict[str, tuple[str, ExitSpec]] = {
        "A1": ("X1", ExitSpec(dyn_stop=("TB_OPP", 12))),
        "A2": ("X1", ExitSpec(sigx=("TA_X1", "EMA", 20, 50))),
        "A3": ("X1", ExitSpec(sigx=("MA_X1", 14))),
        "A4": ("X1", ExitSpec(sigx=("RA_X1", 20))),
        "A5": ("X2", ExitSpec(atr_n=14, atr_mult=2.0)),
    }
    for g in QA_G:
        for a in QA_A:
            code, ex = anchor_exit[a]
            emit("Q-A", (("g", g), ("a", a)), code, ("QA", g, a), ex)

    # ── Q-B OBV 다이버전스 (2 × 2 × 3 × 3 = 36)
    for k in QB_K:
        for s in QB_S:
            entry = ("QB", k, s)
            p = (("K", k), ("S", s))
            emit("Q-B", p, "X1", entry, ExitSpec(ent_stop=True, tgt_r=2.0))
            emit("Q-B", p, "X2", entry, ExitSpec(ent_stop=True, tgt_r=1.0))
            emit("Q-B", p, "X3", entry, ExitSpec(ent_stop=True, hold=42))

    # ── Q-C OBV 추세 (3 × 2 × 3 = 18)
    for ln in QC_LEN:
        entry = ("QC", ln)
        x1 = ("QC_X1", ln)
        p = (("len", ln),)
        emit("Q-C", p, "X1", entry, ExitSpec(sigx=x1))
        emit("Q-C", p, "X2", entry, _atr_exit(x1, 14, 3.0))

    assert len(out) == N_RULES, f"규칙형 총계 {len(out)} != {N_RULES} (§3.6 동결 위반)"
    return out


# ── 변형 열 빌더 ──────────────────────────────────────────────────────────
def _entry_has_level(key: tuple) -> bool:
    """진입 변형이 봉내 스탑주문 레벨을 쓰는가 (§4.4-3 돌파형)."""
    fam = key[0]
    if fam in ("TB", "VA"):
        return True
    if fam == "VC":
        return key[2] == 2
    if fam == "QA":
        return _entry_has_level(_ANCHOR_ENTRY[key[2]])
    return False


def _entry_has_stop(key: tuple) -> bool:
    """진입 변형이 진입 시점 확정 고정 스탑을 제공하는가 (Q-B 피벗)."""
    return key[0] == "QB"


def _d2_scan(release: np.ndarray, high: np.ndarray, low: np.ndarray,
             hh12: np.ndarray, ll12: np.ndarray, window: int = 12
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """V-C `D2` 무장 스캔 — 해제 후 12봉 내 최초 Donchian(12) 돌파만 신호화한다.

    무장·발화 판정은 전부 봉 시작 시점에 알 수 있는 값(`hh12/ll12` 은 `[i−1]` 까지의
    극값, 트리거는 그 봉의 고·저가)만 쓴다. 미발생 시 신호 폐기(§3.4).

    Args:
        release: 봉 i 에서 "해제 확정" 여부 (`sq[i−2] ON` → `sq[i−1] OFF`).
        high/low: 봉 고가·저가.
        hh12/ll12: `HH_12`/`LL_12` (이미 shift 1 포함).
        window: 무장 유효 봉 수.

    Returns:
        (sigL, sigS, lvlL, lvlS) — 발화 봉에만 True/레벨, 그 외 False/NaN.
    """
    n = len(high)
    sig_l = np.zeros(n, bool)
    sig_s = np.zeros(n, bool)
    lvl_l = np.full(n, np.nan)
    lvl_s = np.full(n, np.nan)
    armed_until = -1
    for i in range(n):
        if release[i]:
            armed_until = i + window - 1
        if i > armed_until:
            continue
        up, dn = hh12[i], ll12[i]
        if np.isfinite(up) and high[i] >= up:
            sig_l[i] = True
            lvl_l[i] = up
            armed_until = -1
        elif np.isfinite(dn) and low[i] <= dn:
            sig_s[i] = True
            lvl_s[i] = dn
            armed_until = -1
    return sig_l, sig_s, lvl_l, lvl_s


def _pivot_confirm(x: np.ndarray, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """피벗 확정 배열과 "직전 피벗 인덱스" 배열을 만든다.

    `lab/rsi_divergence_test.py` 동결 관례: `j` 가 `[j−k, j+k]` 창의 극값(`==` 포함)이면
    피벗이며 봉 `j+k` 에서 확정된다.

    Args:
        x: 저가(모드 'low') 또는 고가(모드 'high') 배열.
        k: 좌우 봉 수.
        mode: 'low' 또는 'high'.

    Returns:
        (is_piv, prev_piv) — `prev_piv[j]` 는 `j` 미만의 마지막 피벗 인덱스(-1 = 없음).
    """
    n = len(x)
    is_piv = np.zeros(n, bool)
    if n > 2 * k:
        win = np.lib.stride_tricks.sliding_window_view(x, 2 * k + 1)
        ext = win.min(axis=1) if mode == "low" else win.max(axis=1)
        is_piv[k:n - k] = x[k:n - k] == ext
    idx = np.where(is_piv, np.arange(n), -1)
    run = np.maximum.accumulate(idx)
    prev_piv = np.full(n, -1, dtype=np.int64)
    prev_piv[1:] = run[:-1]
    return is_piv, prev_piv


def _qb_signals(F: Feat, k: int, s: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Q-B OBV 다이버전스 진입 신호와 피벗 스탑 레벨.

    **인과성 교정 (D11 — 명세 §3.5 문언과의 의도적 편차)**: §3.5 는 원전
    `lab/rsi_divergence_test.py` 를 그대로 옮겨 "`j = i−K` 가 `[j−K, j+K]` 창의
    극값이면 봉 `[i]` 에서 확정" 이라고 적었다. 그 창은 봉 `[i]` 자신의 고·저가를
    포함한다. 원전은 **신호봉 종가 체결**이었으므로 인과적이었지만, 본 스윕은
    §4.4-3 에 따라 **봉 `[i]` 시가**에 체결한다 — 시가 시점에는 그 봉의 고·저가를
    알 수 없으므로 문언 그대로 구현하면 §4.4-1(확정봉 `[i−1]` 이하만 사용)을
    위반하는 동일봉 룩어헤드가 된다. 따라서 확정을 1봉 늦춰 `j = i−K−1`
    (창 `[i−2K−1, i−1]` ⊆ 확정봉)로 구현한다. 신선도 조건도 `j2 == i−K−1`.
    이 편차는 규칙을 **덜 정보화**하는 보수적 방향이며 결과 조회 전에 확정됐다.

    Args:
        F: 지표 캐시. k: 피벗 좌우 봉. s: 최대 피벗 간격(봉).

    Returns:
        (sigL, sigS, stopL, stopS). 스탑은 `p2 ∓ 0.5×ATR14[i−1]`.
    """
    obv = F.obv()
    a1 = sh(F.atr(14), 1)
    out: list[np.ndarray] = []
    for mode, px in (("low", F.l), ("high", F.h)):
        is_piv, prev = _pivot_confirm(px, k, mode)
        ok = is_piv & (prev >= 0)
        j1 = np.where(ok, prev, 0)
        j = np.arange(len(px))
        gap = (j - j1) <= s
        if mode == "low":
            cond = (px < px[j1]) & (obv > obv[j1])
        else:
            cond = (px > px[j1]) & (obv < obv[j1])
        at_j = ok & gap & np.asarray(cond, dtype=bool) & np.isfinite(obv) & np.isfinite(obv[j1])
        sig = sh(at_j, k + 1)      # D11: 확정 1봉 지연 → 창이 [i−1] 이하로 닫힌다
        lvl = sh(px, k + 1)
        out.append(sig)
        out.append(lvl)
    sig_l, piv_l, sig_s, piv_s = out
    stop_l = piv_l - 0.5 * a1
    stop_s = piv_s + 0.5 * a1
    sig_l = sig_l & np.isfinite(stop_l)
    sig_s = sig_s & np.isfinite(stop_s)
    return sig_l, sig_s, np.where(sig_l, stop_l, np.nan), np.where(sig_s, stop_s, np.nan)


def build_entry(key: tuple, F: Feat) -> dict[str, np.ndarray]:
    """진입 변형 1개의 신호·레벨·고정스탑 배열을 만든다.

    Args:
        key: 진입 변형 키 (`("TA", "EMA", 20, 50)` 등).
        F: 해당 심볼·타임프레임의 지표 캐시.

    Returns:
        `{"sigL","sigS","lvlL","lvlS","stopL","stopS"}` — 레벨/스탑은 NaN 이면 미사용
        (= 다음 봉 시가 MOO 체결).
    """
    fam = key[0]
    n = F.n
    nan = np.full(n, np.nan)
    lvl_l = lvl_s = stop_l = stop_s = nan

    if fam == "TA":
        _, mt, f, s = key
        a, b = F.ma(mt, f), F.ma(mt, s)
        a1, a2, b1, b2 = sh(a, 1), sh(a, 2), sh(b, 1), sh(b, 2)
        sig_l = (a1 > b1) & (a2 <= b2)
        sig_s = (a1 < b1) & (a2 >= b2)
    elif fam == "TB":
        _, nn = key
        lvl_l, lvl_s = F.hh(nn), F.ll(nn)
        sig_l, sig_s = np.isfinite(lvl_l), np.isfinite(lvl_s)
    elif fam == "TC":
        _, mt, nn = key
        m = F.ma(mt, nn)
        c1, c2, m1, m2 = sh(F.c, 1), sh(F.c, 2), sh(m, 1), sh(m, 2)
        sig_l = (c1 > m1) & (c2 <= m2)
        sig_s = (c1 < m1) & (c2 >= m2)
    elif fam == "MA":
        _, nn, u = key
        r = F.rsi(nn)
        r1, r2 = sh(r, 1), sh(r, 2)
        sig_l = (r1 > u) & (r2 <= u)
        sig_s = (r1 < 100 - u) & (r2 >= 100 - u)
    elif fam == "MB":
        _, v = key
        macd, signal, hist = F.macd()
        a, b = (macd, signal) if v == 1 else ((macd, np.zeros(n)) if v == 2 else (hist, np.zeros(n)))
        a1, a2, b1, b2 = sh(a, 1), sh(a, 2), sh(b, 1), sh(b, 2)
        sig_l = (a1 > b1) & (a2 <= b2)
        sig_s = (a1 < b1) & (a2 >= b2)
    elif fam == "MC":
        _, v = key
        kk, dd = F.stoch()
        k1, k2, d1, d2 = sh(kk, 1), sh(kk, 2), sh(dd, 1), sh(dd, 2)
        if v == 1:
            sig_l = (k1 > d1) & (k2 <= d2) & (k1 < 20)
            sig_s = (k1 < d1) & (k2 >= d2) & (k1 > 80)
        elif v == 2:
            sig_l = (k1 > 20) & (k2 <= 20)
            sig_s = (k1 < 80) & (k2 >= 80)
        else:
            sig_l = (k1 > 50) & (k2 <= 50)
            sig_s = (k1 < 50) & (k2 >= 50)
    elif fam == "RA":
        _, nn, kk, trig = key
        up, _, lo = F.bb(nn, kk)
        c1, c2 = sh(F.c, 1), sh(F.c, 2)
        u1, l1, u2, l2 = sh(up, 1), sh(lo, 1), sh(up, 2), sh(lo, 2)
        if trig == 1:
            sig_l, sig_s = c1 < l1, c1 > u1
        else:
            sig_l = (c2 < l2) & (c1 >= l1)
            sig_s = (c2 > u2) & (c1 <= u1)
    elif fam == "RB":
        _, lb, th = key
        z1 = sh(F.z(lb), 1)
        sig_l, sig_s = z1 < -th, z1 > th
    elif fam == "RC":
        _, nn, t, filt = key
        r1 = sh(F.rsi(nn), 1)
        sig_l, sig_s = r1 < t, r1 > 100 - t
        if filt:
            c1, s200 = sh(F.c, 1), sh(F.sma(200), 1)
            sig_l = sig_l & (c1 > s200)
            sig_s = sig_s & (c1 < s200)
    elif fam == "VA":
        _, m, ref = key
        a1 = sh(F.atr(14), 1)
        base = F.o if ref == "open" else sh(F.c, 1)
        lvl_l, lvl_s = base + m * a1, base - m * a1
        sig_l, sig_s = np.isfinite(lvl_l), np.isfinite(lvl_s)
    elif fam == "VB":
        _, p, m, usage = key
        e, a = F.ema(20), F.atr(p)
        c1, u1, l1 = sh(F.c, 1), sh(e + m * a, 1), sh(e - m * a, 1)
        if usage == 1:
            sig_l, sig_s = c1 > u1, c1 < l1
        else:
            sig_l, sig_s = c1 < l1, c1 > u1
    elif fam == "VC":
        _, sq, dm = key
        on, valid = F.squeeze(sq)
        rel = sh(on, 2) & (~sh(on, 1)) & sh(valid, 2) & sh(valid, 1)
        if dm == 1:
            c1, s1 = sh(F.c, 1), sh(F.sma(20), 1)
            sig_l, sig_s = rel & (c1 > s1), rel & (c1 < s1)
        else:
            sig_l, sig_s, lvl_l, lvl_s = _d2_scan(rel, F.h, F.l, F.hh(12), F.ll(12))
    elif fam == "QA":
        _, g, anchor = key
        base = build_entry(_ANCHOR_ENTRY[anchor], F)
        surge = F.surge(g)
        return {"sigL": base["sigL"] & surge, "sigS": base["sigS"] & surge,
                "lvlL": base["lvlL"], "lvlS": base["lvlS"],
                "stopL": base["stopL"], "stopS": base["stopS"]}
    elif fam == "QB":
        _, k, s = key
        sig_l, sig_s, stop_l, stop_s = _qb_signals(F, k, s)
    elif fam == "QC":
        _, ln = key
        ob = F.obv()
        m = F.sma_of(ob, ln, "obv")
        o1, o2, m1, m2 = sh(ob, 1), sh(ob, 2), sh(m, 1), sh(m, 2)
        sig_l = (o1 > m1) & (o2 <= m2)
        sig_s = (o1 < m1) & (o2 >= m2)
    else:
        raise ValueError(f"알 수 없는 진입 변형: {key!r}")

    return {"sigL": np.asarray(sig_l, dtype=bool), "sigS": np.asarray(sig_s, dtype=bool),
            "lvlL": lvl_l, "lvlS": lvl_s, "stopL": stop_l, "stopS": stop_s}


def build_sigx(key: tuple, F: Feat) -> tuple[np.ndarray, np.ndarray]:
    """신호청산 변형 1개 → (롱 청산 조건, 숏 청산 조건). 확정봉 기준, 체결은 시가."""
    fam = key[0]
    if fam == "TA_X1":
        e = build_entry(("TA",) + key[1:], F)
        return e["sigS"], e["sigL"]
    if fam == "TC_X1":
        e = build_entry(("TC",) + key[1:], F)
        return e["sigS"], e["sigL"]
    if fam == "MB_X1":
        e = build_entry(("MB", key[1]), F)
        return e["sigS"], e["sigL"]
    if fam == "MC_X1":
        e = build_entry(("MC", key[1]), F)
        return e["sigS"], e["sigL"]
    if fam == "QC_X1":
        e = build_entry(("QC", key[1]), F)
        return e["sigS"], e["sigL"]
    if fam == "MA_X1":
        r = F.rsi(key[1])
        r1, r2 = sh(r, 1), sh(r, 2)
        return (r1 < 50) & (r2 >= 50), (r1 > 50) & (r2 <= 50)
    if fam == "MA_X2":
        _, n, u = key
        r = F.rsi(n)
        r1, r2 = sh(r, 1), sh(r, 2)
        return (r1 < 100 - u) & (r2 >= 100 - u), (r1 > u) & (r2 <= u)
    if fam == "MC_X2":
        k, _ = F.stoch()
        k1, k2 = sh(k, 1), sh(k, 2)
        return (k1 < 50) & (k2 >= 50), (k1 > 50) & (k2 <= 50)
    if fam == "MC_X3":
        k, _ = F.stoch()
        k1 = sh(k, 1)
        return np.asarray(k1 >= 80), np.asarray(k1 <= 20)
    if fam == "RA_X1":
        _, mid, _ = F.bb(key[1], 2.0)
        c1, m1 = sh(F.c, 1), sh(mid, 1)
        return np.asarray(c1 >= m1), np.asarray(c1 <= m1)
    if fam == "RB_X1":
        z1 = sh(F.z(key[1]), 1)
        return np.asarray(z1 >= 0), np.asarray(z1 <= 0)
    if fam == "RB_X2":
        z1 = sh(F.z(key[1]), 1)
        near = np.asarray(np.abs(z1) <= 0.5)
        return near, near
    if fam == "RC_X1":
        c1, s1 = sh(F.c, 1), sh(F.sma(5), 1)
        return np.asarray(c1 > s1), np.asarray(c1 < s1)
    if fam == "RC_X2":
        r1 = sh(F.rsi(key[1]), 1)
        return np.asarray(r1 >= 50), np.asarray(r1 <= 50)
    if fam == "RC_X3":
        _, n, t = key
        r1 = sh(F.rsi(n), 1)
        return np.asarray(r1 >= 100 - t), np.asarray(r1 <= t)
    if fam == "VB_X1":
        # X1 = "close 가 EMA20 복귀" — 중심선은 고정이므로 ATR 기간 p·승수 m 과 무관.
        # 진입 방향(용법)만 필요하다.
        _, usage = key
        c1, e1 = sh(F.c, 1), sh(F.ema(20), 1)
        if usage == 1:
            return np.asarray(c1 <= e1), np.asarray(c1 >= e1)
        return np.asarray(c1 >= e1), np.asarray(c1 <= e1)
    if fam == "VC_X2":
        c1, s1 = sh(F.c, 1), sh(F.sma(20), 1)
        return np.asarray(c1 < s1), np.asarray(c1 > s1)
    raise ValueError(f"알 수 없는 신호청산 변형: {key!r}")


def build_dyn(key: tuple, F: Feat) -> tuple[np.ndarray, np.ndarray]:
    """동적 청산 레벨 변형 → (롱 포지션용 레벨, 숏 포지션용 레벨).

    스탑으로 쓰이면 불리한 쪽, 목표로 쓰이면 유리한 쪽 레벨이다. 전부 봉 시작
    시점에 확정 가능한 값만 쓴다 (§ D5).
    """
    fam = key[0]
    if fam == "TB_OPP":
        m = key[1]
        return F.ll(m), F.hh(m)
    if fam == "VA_OPP":
        _, m, ref = key
        a1 = sh(F.atr(14), 1)
        base = F.o if ref == "open" else sh(F.c, 1)
        return base - m * a1, base + m * a1
    if fam == "VB_OPP":
        _, p, m = key
        e, a = F.ema(20), F.atr(p)
        return sh(e - m * a, 1), sh(e + m * a, 1)
    if fam == "RA_OPP":
        _, n, k = key
        up, _, lo = F.bb(n, k)
        return sh(up, 1), sh(lo, 1)
    raise ValueError(f"알 수 없는 동적 레벨 변형: {key!r}")


# ── 규칙 → 인덱스 컴파일 ──────────────────────────────────────────────────
class Registry:
    """변형 키를 열 인덱스로 사상한다. 열 0 은 센티널(False / NaN)."""

    def __init__(self) -> None:
        self.keys: list[tuple] = []
        self._idx: dict[tuple, int] = {}

    def get(self, key: tuple | None) -> int:
        """키의 열 인덱스를 반환한다 (`None` → 0 = 센티널)."""
        if key is None:
            return 0
        if key not in self._idx:
            self._idx[key] = len(self.keys) + 1
            self.keys.append(key)
        return self._idx[key]

    def __len__(self) -> int:
        return len(self.keys) + 1


@dataclass
class Compiled:
    """규칙 축(R)으로 정렬된 인덱스·파라미터 배열 묶음."""

    ent: np.ndarray
    lvl: np.ndarray
    estop: np.ndarray
    sigx: np.ndarray
    dstop: np.ndarray
    dtgt: np.ndarray
    atr: np.ndarray
    atr_mult: np.ndarray
    hold: np.ndarray
    tgt_r: np.ndarray
    dir_l: np.ndarray
    dir_s: np.ndarray
    reg_ent: Registry
    reg_lvl: Registry
    reg_estop: Registry
    reg_sigx: Registry
    reg_dstop: Registry
    reg_dtgt: Registry
    reg_atr: Registry


def compile_rules(specs: Sequence[RuleSpec]) -> Compiled:
    """규칙 명세 리스트를 규칙 축 인덱스 배열로 컴파일한다.

    Args:
        specs: `enumerate_rules()` 결과 (또는 그 부분집합).

    Returns:
        `Compiled` — 모든 배열의 길이는 `len(specs)`.
    """
    r_ent, r_lvl, r_estop = Registry(), Registry(), Registry()
    r_sigx, r_dstop, r_dtgt, r_atr = Registry(), Registry(), Registry(), Registry()
    n = len(specs)
    ent = np.zeros(n, np.int64)
    lvl = np.zeros(n, np.int64)
    estop = np.zeros(n, np.int64)
    sigx = np.zeros(n, np.int64)
    dstop = np.zeros(n, np.int64)
    dtgt = np.zeros(n, np.int64)
    atr = np.zeros(n, np.int64)
    amult = np.zeros(n)
    hold = np.zeros(n, np.int64)
    tgt_r = np.zeros(n)
    dir_l = np.zeros(n, bool)
    dir_s = np.zeros(n, bool)
    for i, sp in enumerate(specs):
        ent[i] = r_ent.get(sp.entry)
        if _entry_has_level(sp.entry):
            lvl[i] = r_lvl.get(sp.entry)
        if sp.exits.ent_stop and _entry_has_stop(sp.entry):
            estop[i] = r_estop.get(sp.entry)
        sigx[i] = r_sigx.get(sp.exits.sigx)
        dstop[i] = r_dstop.get(sp.exits.dyn_stop)
        dtgt[i] = r_dtgt.get(sp.exits.dyn_tgt)
        if sp.exits.atr_n:
            atr[i] = r_atr.get(("ATR", sp.exits.atr_n))
            amult[i] = sp.exits.atr_mult
        hold[i] = sp.exits.hold
        tgt_r[i] = sp.exits.tgt_r
        dir_l[i] = sp.direction in ("L", "LS")
        dir_s[i] = sp.direction in ("S", "LS")
    return Compiled(ent, lvl, estop, sigx, dstop, dtgt, atr, amult, hold, tgt_r,
                    dir_l, dir_s, r_ent, r_lvl, r_estop, r_sigx, r_dstop, r_dtgt, r_atr)


def _stack(cols: list[np.ndarray], n: int, dtype: Any, fill: Any) -> np.ndarray:
    """센티널 열 + 변형 열들을 (n, 1+len(cols)) 행렬로 쌓는다."""
    out = np.full((n, len(cols) + 1), fill, dtype=dtype)
    for j, col in enumerate(cols):
        out[:, j + 1] = col
    return out


def materialize(comp: Compiled, F: Feat) -> dict[str, np.ndarray]:
    """컴파일된 변형 키들을 한 심볼에 대해 행렬로 실체화한다.

    열 순서는 `Registry` 등록 순서이므로 **심볼이 달라도 동일**하다 — 규칙 축 인덱스
    배열을 세 심볼에 그대로 재사용할 수 있는 이유다.

    Args:
        comp: `compile_rules` 결과.
        F: 한 심볼·타임프레임의 지표 캐시.

    Returns:
        행렬 딕셔너리 (`entL/entS/lvlL/lvlS/estopL/estopS/sxL/sxS/dsL/dsS/dtL/dtS/atr`).
    """
    n = F.n
    built = {k: build_entry(k, F) for k in comp.reg_ent.keys}
    mats: dict[str, np.ndarray] = {
        "entL": _stack([built[k]["sigL"] for k in comp.reg_ent.keys], n, bool, False),
        "entS": _stack([built[k]["sigS"] for k in comp.reg_ent.keys], n, bool, False),
        "lvlL": _stack([built[k]["lvlL"] for k in comp.reg_lvl.keys], n, np.float64, np.nan),
        "lvlS": _stack([built[k]["lvlS"] for k in comp.reg_lvl.keys], n, np.float64, np.nan),
        "estopL": _stack([built[k]["stopL"] for k in comp.reg_estop.keys], n, np.float64, np.nan),
        "estopS": _stack([built[k]["stopS"] for k in comp.reg_estop.keys], n, np.float64, np.nan),
    }
    sx = [build_sigx(k, F) for k in comp.reg_sigx.keys]
    mats["sxL"] = _stack([a for a, _ in sx], n, bool, False)
    mats["sxS"] = _stack([b for _, b in sx], n, bool, False)
    ds = [build_dyn(k, F) for k in comp.reg_dstop.keys]
    mats["dsL"] = _stack([a for a, _ in ds], n, np.float64, np.nan)
    mats["dsS"] = _stack([b for _, b in ds], n, np.float64, np.nan)
    dt = [build_dyn(k, F) for k in comp.reg_dtgt.keys]
    mats["dtL"] = _stack([a for a, _ in dt], n, np.float64, np.nan)
    mats["dtS"] = _stack([b for _, b in dt], n, np.float64, np.nan)
    # ATR 은 미리 1봉 시프트해 열 [i] 가 ATR[i−1] 을 담는다 (§4.4-1 강제)
    mats["atr"] = _stack([sh(F.atr(k[1]), 1) for k in comp.reg_atr.keys], n, np.float64, np.nan)
    return mats


# ── 시뮬레이션 ────────────────────────────────────────────────────────────
@dataclass
class SimResult:
    """한 타임프레임 실행 결과."""

    rule_ids: list[str]
    snap_ts: pd.DatetimeIndex
    equity: np.ndarray          # (R, D+1) 자정 MTM 자본
    returns: np.ndarray         # (R, D) 일수익률
    trades: np.ndarray          # (R,)
    cost: np.ndarray            # (R,)
    funding: np.ndarray         # (R,)
    halts: np.ndarray           # (R,)
    trace: list[dict] = field(default_factory=list)
    missing_funding: int = 0


def simulate_timeframe(
    tf: str,
    index: pd.DatetimeIndex,
    ohlcv: dict[str, dict[str, np.ndarray]],
    funding: dict[str, np.ndarray],
    specs: Sequence[RuleSpec],
    win_start: pd.Timestamp = WIN_START,
    win_end: pd.Timestamp = WIN_END,
    trace_rules: set[int] | None = None,
) -> SimResult:
    """규칙 축을 벡터화해 한 타임프레임 전체를 1회 봉 루프로 시뮬레이션한다.

    Args:
        tf: '1h' 또는 '4h' (규칙 ID 표기용).
        index: 공통 봉 인덱스 (UTC, tz-aware, 등간격).
        ohlcv: `sym → {'open','high','low','close','volume'}` numpy 배열 (index 정렬).
        funding: `sym → 배열` — 정산 시각 봉에만 요율, 그 외 NaN.
        specs: 규칙 명세.
        win_start/win_end: 평가 창 (§4.2). 창 이전에는 주문을 생성하지 않는다.
        trace_rules: 거래 원장을 수집할 규칙 인덱스 집합 (테스트용, None = 수집 안 함).

    Returns:
        `SimResult`.
    """
    comp = compile_rules(specs)
    syms = tuple(ohlcv.keys())
    ns = len(syms)
    feats = {s: Feat(ohlcv[s]["open"], ohlcv[s]["high"], ohlcv[s]["low"],
                     ohlcv[s]["close"], ohlcv[s]["volume"]) for s in syms}
    mats = {s: materialize(comp, feats[s]) for s in syms}
    O = [ohlcv[s]["open"] for s in syms]
    H = [ohlcv[s]["high"] for s in syms]
    L = [ohlcv[s]["low"] for s in syms]
    C = [ohlcv[s]["close"] for s in syms]
    FD = [funding[s] for s in syms]
    M = [mats[s] for s in syms]

    r = len(specs)
    cash = np.full(r, CELL_CAPITAL)
    day_eq = np.full(r, CELL_CAPITAL)
    halted = np.zeros(r, bool)
    pdir = np.zeros((r, ns), np.int8)
    pqty = np.zeros((r, ns))
    pfill = np.zeros((r, ns))
    pstop = np.full((r, ns), np.nan)
    ptgt = np.full((r, ns), np.nan)
    pent = np.zeros((r, ns), np.int64)
    prisk = np.zeros((r, ns))
    exited = np.zeros((r, ns), bool)
    n_trades = np.zeros(r, np.int64)
    cost_acc = np.zeros(r)
    fund_acc = np.zeros(r)
    halt_acc = np.zeros(r, np.int64)

    ent_i, lvl_i, estop_i = comp.ent, comp.lvl, comp.estop
    sx_i, ds_i, dt_i, atr_i = comp.sigx, comp.dstop, comp.dtgt, comp.atr
    amult, hold_b, tgt_r = comp.atr_mult, comp.hold, comp.tgt_r
    dir_l, dir_s = comp.dir_l, comp.dir_s
    has_hold = hold_b > 0

    pos_all = np.flatnonzero(index == win_start)
    if pos_all.size != 1:
        raise ValueError(f"평가 창 시작 {win_start} 이 봉 인덱스에 정확히 1회 없다")
    i0 = int(pos_all[0])
    end_all = np.flatnonzero(index == win_end)
    if end_all.size != 1:
        raise ValueError(f"평가 창 종료 {win_end} 이 봉 인덱스에 정확히 1회 없다")
    i_end = int(end_all[0])          # 이 봉은 거래하지 않는다 (창 밖)
    if i0 == 0:
        raise ValueError("평가 창 시작 봉 이전 데이터가 없다 (워밍업 불가)")

    snaps: list[np.ndarray] = []
    snap_ts: list[pd.Timestamp] = []
    trace: list[dict] = []
    missing_fund = 0
    hours = index.hour.to_numpy()
    minutes = index.minute.to_numpy()

    def mtm(bar: int) -> np.ndarray:
        """봉 `bar` 의 종가로 평가한 셀 순자본."""
        out = cash.copy()
        for s in range(ns):
            px = C[s][bar]
            if np.isfinite(px):
                out += pqty[:, s] * (px - pfill[:, s]) * pdir[:, s]
        return out

    def sizing_equity_gross(bar: int) -> tuple[np.ndarray, np.ndarray]:
        """사이징용 (자본, 그로스 명목). 같은 봉에 체결된 포지션은 체결가로 평가한다.

        직전 확정 종가 `close[bar]` 로 마킹하되, 이 봉에서 막 체결된 포지션에
        과거 가격을 적용하면 존재하지 않는 평가손익이 생기므로 체결가를 쓴다.
        """
        eq = cash.copy()
        gr = np.zeros(len(cash))
        for s in range(ns):
            px = C[s][bar]
            fresh = pent[:, s] == bar + 1
            mark = np.where(fresh | ~np.isfinite(px), pfill[:, s], px)
            eq += pqty[:, s] * (mark - pfill[:, s]) * pdir[:, s]
            gr += np.abs(pqty[:, s]) * mark
        return eq, gr

    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(i0, i_end):
            midnight = hours[i] == 0 and minutes[i] == 0
            if midnight:
                eq = mtm(i - 1)
                snaps.append(eq)
                snap_ts.append(index[i])
                day_eq = eq.copy()
                halted[:] = False

            if hours[i] in FUNDING_HOURS and minutes[i] == 0:
                for s in range(ns):
                    rate = FD[s][i]
                    px = C[s][i - 1]
                    if not np.isfinite(rate):
                        if np.any(pdir[:, s] != 0):
                            missing_fund += 1
                        continue
                    if not np.isfinite(px):
                        continue
                    pay = pdir[:, s] * rate * pqty[:, s] * px
                    cash -= pay
                    fund_acc -= pay

            exited[:] = False

            # ── 청산 (§4.4-9/10, D4)
            for s in range(ns):
                d = pdir[:, s]
                if not d.any():
                    continue
                o_, h_, l_, c_ = O[s][i], H[s][i], L[s][i], C[s][i]
                if not np.isfinite(c_):
                    continue
                ms = M[s]
                is_long = d > 0
                held = d != 0
                sig = np.where(is_long, ms["sxL"][i][sx_i], ms["sxS"][i][sx_i])
                hold_hit = has_hold & ((i - pent[:, s]) >= hold_b)
                at_open = held & (sig | hold_hit)
                dyn = np.where(is_long, ms["dsL"][i][ds_i], ms["dsS"][i][ds_i])
                adv = np.where(is_long, np.fmax(pstop[:, s], dyn), np.fmin(pstop[:, s], dyn))
                hit_stop = held & np.where(is_long, l_ <= adv, h_ >= adv)
                dtg = np.where(is_long, ms["dtL"][i][dt_i], ms["dtS"][i][dt_i])
                tgt = np.where(is_long, np.fmin(ptgt[:, s], dtg), np.fmax(ptgt[:, s], dtg))
                hit_tgt = held & np.where(is_long, h_ >= tgt, l_ <= tgt)
                go = at_open | hit_stop | hit_tgt
                idx = np.flatnonzero(go)
                if idx.size == 0:
                    continue
                px = np.where(
                    at_open[idx], o_,
                    np.where(hit_stop[idx],
                             np.where(is_long[idx], np.minimum(adv[idx], o_),
                                      np.maximum(adv[idx], o_)),
                             tgt[idx]))
                q = pqty[idx, s]
                fee = q * px * COST_SIDE
                cash[idx] += q * (px - pfill[idx, s]) * pdir[idx, s] - fee
                cost_acc[idx] += fee
                n_trades[idx] += 1
                if trace_rules is not None:
                    for jj, rr in enumerate(idx):
                        if int(rr) in trace_rules:
                            trace.append(dict(rule=int(rr), sym=syms[s], bar=i,
                                              ts=index[i], action="exit",
                                              price=float(px[jj]), qty=float(q[jj]),
                                              direction=int(pdir[rr, s]),
                                              reason=("open" if at_open[rr] else
                                                      "stop" if hit_stop[rr] else "target")))
                pdir[idx, s] = 0
                pqty[idx, s] = 0.0
                pfill[idx, s] = 0.0
                pstop[idx, s] = np.nan
                ptgt[idx, s] = np.nan
                prisk[idx, s] = 0.0
                exited[idx, s] = True

            # ── 진입 (§4.5, D1/D2/D6)
            npos = (pdir != 0).sum(axis=1)
            for s in range(ns):
                o_, h_, l_, c_ = O[s][i], H[s][i], L[s][i], C[s][i]
                if not np.isfinite(c_):
                    continue
                free = (pdir[:, s] == 0) & (~exited[:, s]) & (~halted) & (npos < MAX_POS)
                if not free.any():
                    continue
                ms = M[s]
                sig_l = ms["entL"][i][ent_i] & dir_l & free
                sig_s = ms["entS"][i][ent_i] & dir_s & free
                if not (sig_l.any() or sig_s.any()):
                    continue
                ll_ = ms["lvlL"][i][lvl_i]
                ls_ = ms["lvlS"][i][lvl_i]
                brk_l, brk_s = np.isfinite(ll_), np.isfinite(ls_)
                trig_l = sig_l & (~brk_l | (h_ >= ll_))
                trig_s = sig_s & (~brk_s | (l_ <= ls_)) & (~trig_l)   # 동시 성립 시 롱 우선
                take = trig_l | trig_s
                idx = np.flatnonzero(take)
                if idx.size == 0:
                    continue
                dirn = np.where(trig_l[idx], 1, -1).astype(np.int8)
                fill = np.where(trig_l[idx],
                                np.where(brk_l[idx], np.maximum(ll_[idx], o_), o_),
                                np.where(brk_s[idx], np.minimum(ls_[idx], o_), o_))

                est = np.where(dirn > 0, ms["estopL"][i][estop_i[idx]],
                               ms["estopS"][i][estop_i[idx]])
                av = ms["atr"][i][atr_i[idx]]
                atr_stop = fill - dirn * amult[idx] * av
                fixed = np.where(np.isfinite(est), est, atr_stop)
                dyn0 = np.where(dirn > 0, ms["dsL"][i][ds_i[idx]], ms["dsS"][i][ds_i[idx]])
                dtg0 = np.where(dirn > 0, ms["dtL"][i][dt_i[idx]], ms["dtS"][i][dt_i[idx]])
                # § D6: 후보를 먼저 "불리한 쪽" 으로 거르고 나서 가장 가까운 것을 고른다.
                # (먼저 최근접을 고르면 유리한 쪽 동적 레벨이 실재하는 고정 스탑을
                #  가려 규칙이 잘못 명목 사이징으로 떨어진다.)
                adv_fixed = np.where(np.where(dirn > 0, fixed < fill, fixed > fill),
                                     fixed, np.nan)
                adv_dyn = np.where(np.where(dirn > 0, dyn0 < fill, dyn0 > fill),
                                   dyn0, np.nan)
                cand = np.where(dirn > 0, np.fmax(adv_fixed, adv_dyn),
                                np.fmin(adv_fixed, adv_dyn))
                dist = np.abs(fill - cand)
                use_stop = np.isfinite(cand) & (dist > 0)

                eq_full, gross_full = sizing_equity_gross(i - 1)
                eq_mark = eq_full[idx]
                gross = gross_full[idx]
                heat = prisk[idx].sum(axis=1)

                with np.errstate(divide="ignore", invalid="ignore"):
                    q_stop = np.where(use_stop, RISK * eq_mark / dist, 0.0)
                q_noml = eq_mark * STOPLESS_NOTIONAL / fill
                qty = np.where(use_stop, q_stop, q_noml)
                room = np.maximum(0.0, GROSS_CAP * eq_mark - gross)
                qty = np.minimum(qty, room / fill)
                risk_new = np.where(use_stop, qty * dist, qty * fill * STOPLESS_HEAT)
                ok = (np.isfinite(fill) & (fill > 0) & (qty > 0) & (eq_mark > 0)
                      & ((heat + risk_new) <= HEAT_CAP * eq_mark))
                if not ok.any():
                    continue
                idx = idx[ok]
                dirn, fill, qty = dirn[ok], fill[ok], qty[ok]
                risk_new, use_stop = risk_new[ok], use_stop[ok]
                fixed, dyn0, dtg0, dist = fixed[ok], dyn0[ok], dtg0[ok], dist[ok]

                fee = qty * fill * COST_SIDE
                cash[idx] -= fee
                cost_acc[idx] += fee
                n_trades[idx] += 1
                pdir[idx, s] = dirn
                pqty[idx, s] = qty
                pfill[idx, s] = fill
                pstop[idx, s] = fixed
                pent[idx, s] = i
                prisk[idx, s] = risk_new
                rmul = tgt_r[idx]
                with np.errstate(invalid="ignore"):
                    tval = np.where((rmul > 0) & np.isfinite(fixed),
                                    fill + dirn * rmul * np.abs(fill - fixed), np.nan)
                ptgt[idx, s] = tval
                npos[idx] += 1
                if trace_rules is not None:
                    for jj, rr in enumerate(idx):
                        if int(rr) in trace_rules:
                            trace.append(dict(rule=int(rr), sym=syms[s], bar=i,
                                              ts=index[i], action="entry",
                                              price=float(fill[jj]), qty=float(qty[jj]),
                                              direction=int(dirn[jj]),
                                              stop=float(fixed[jj]),
                                              sized_by=("stop" if use_stop[jj] else "notional")))

                # 체결봉 스탑/목표 검사 (§4.4-10, D4) — 경로 미상이므로 비관
                is_long2 = dirn > 0
                adv2 = np.where(is_long2, np.fmax(fixed, dyn0), np.fmin(fixed, dyn0))
                tgt2 = np.where(is_long2, np.fmin(tval, dtg0), np.fmax(tval, dtg0))
                hs = np.where(is_long2, l_ <= adv2, h_ >= adv2)
                ht = np.where(is_long2, h_ >= tgt2, l_ <= tgt2)
                ge = hs | ht
                sub = np.flatnonzero(ge)
                if sub.size:
                    ridx = idx[sub]
                    epx = np.where(hs[sub],
                                   np.where(is_long2[sub], np.minimum(adv2[sub], o_),
                                            np.maximum(adv2[sub], o_)),
                                   tgt2[sub])
                    q2 = qty[sub]
                    fee2 = q2 * epx * COST_SIDE
                    cash[ridx] += q2 * (epx - fill[sub]) * dirn[sub] - fee2
                    cost_acc[ridx] += fee2
                    n_trades[ridx] += 1
                    if trace_rules is not None:
                        for jj, rr in enumerate(ridx):
                            if int(rr) in trace_rules:
                                trace.append(dict(rule=int(rr), sym=syms[s], bar=i,
                                                  ts=index[i], action="exit",
                                                  price=float(epx[jj]), qty=float(q2[jj]),
                                                  direction=int(dirn[sub][jj]),
                                                  reason="stop" if hs[sub][jj] else "target"))
                    pdir[ridx, s] = 0
                    pqty[ridx, s] = 0.0
                    pfill[ridx, s] = 0.0
                    pstop[ridx, s] = np.nan
                    ptgt[ridx, s] = np.nan
                    prisk[ridx, s] = 0.0
                    exited[ridx, s] = True
                    npos[ridx] -= 1

            # ── 일손실 정지 판정 (봉 마감 후 — 다음 봉부터 적용)
            eq_now = mtm(i)
            with np.errstate(invalid="ignore", divide="ignore"):
                trip = (~halted) & (day_eq > 0) & ((eq_now / day_eq - 1.0) <= DAILY_HALT)
            if trip.any():
                halted |= trip
                halt_acc += trip

        # ── 창 종료: 마지막 확정 봉 종가로 강제 청산 (§4.2 eod)
        last = i_end - 1
        for s in range(ns):
            idx = np.flatnonzero(pdir[:, s] != 0)
            if idx.size == 0:
                continue
            px = C[s][last]
            if not np.isfinite(px):     # fail-closed: 마지막 확정 종가로 후퇴
                valid = np.flatnonzero(np.isfinite(C[s][:last + 1]))
                if valid.size == 0:
                    logger.warning("%s: 강제 청산 가격 없음 — 포지션 %d 개 미평가",
                                   syms[s], idx.size)
                    continue
                px = C[s][valid[-1]]
                logger.warning("%s: 창 종료 봉 종가 결측 → 직전 확정 종가로 강제 청산", syms[s])
            q = pqty[idx, s]
            fee = q * px * COST_SIDE
            cash[idx] += q * (px - pfill[idx, s]) * pdir[idx, s] - fee
            cost_acc[idx] += fee
            n_trades[idx] += 1
            if trace_rules is not None:
                for jj, rr in enumerate(idx):
                    if int(rr) in trace_rules:
                        trace.append(dict(rule=int(rr), sym=syms[s], bar=last,
                                          ts=index[last], action="exit",
                                          price=float(px), qty=float(q[jj]),
                                          reason="eod"))
            pdir[idx, s] = 0
            pqty[idx, s] = 0.0
        snaps.append(cash.copy())
        snap_ts.append(index[i_end])

    eq = np.vstack(snaps).T.copy()
    prev = eq[:, :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(prev > 0, eq[:, 1:] / prev - 1.0, 0.0)
    return SimResult([sp.rid(tf) for sp in specs], pd.DatetimeIndex(snap_ts),
                     eq, ret, n_trades, cost_acc, fund_acc, halt_acc, trace, missing_fund)


# ── 데이터 로딩·정렬 ──────────────────────────────────────────────────────
def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """1h → 4h 리샘플. **닫힌 1h 봉 4개가 전부 있을 때만 확정** (§4.3).

    Args:
        df: UTC DatetimeIndex 의 1h OHLCV.

    Returns:
        UTC [00,04,08,12,16,20) 정렬 4h OHLCV. 부분 봉은 전 열 NaN.
    """
    g = df.resample("4h", origin="epoch", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(min_count=1),
    })
    cnt = g["close"].count()
    out.loc[cnt != 4, :] = np.nan
    return out


def load_inputs(verify_hashes: bool = True) -> tuple[dict[str, pd.DataFrame], pd.DataFrame,
                                                     dict[str, str]]:
    """동결 parquet 3종을 읽고 SHA256 을 검증한다 (§11.2, fail-closed).

    Args:
        verify_hashes: False 면 해시 불일치를 경고로만 남긴다 (합성 데이터 테스트용).

    Returns:
        (심볼별 1h OHLCV, 펀딩 DataFrame, 파일별 SHA256).

    Raises:
        ValueError: 해시 불일치 (검증 활성 시).
    """
    hashes = {rel: sha256_file(REPO_ROOT / rel) for rel in EXPECTED_SHA256}
    for rel, want in EXPECTED_SHA256.items():
        if hashes[rel] != want:
            msg = f"입력 해시 불일치 {rel}: {hashes[rel]} != {want} (§11.2 동결 위반)"
            if verify_hashes:
                raise ValueError(msg)
            logger.warning(msg)
    perp = pd.read_parquet(PERP_PATH)
    cols = ["open", "high", "low", "close", "volume"]
    data = {"BTC": perp.xs("BTC", level="sym")[cols],
            "ETH": perp.xs("ETH", level="sym")[cols],
            "SOL": pd.read_parquet(SOL_PATH)[cols]}
    fund = pd.read_parquet(FUND_PATH)[list(SYMBOLS)]
    return data, fund, hashes


def align(data: dict[str, pd.DataFrame], fund: pd.DataFrame, tf: str
          ) -> tuple[pd.DatetimeIndex, dict[str, dict[str, np.ndarray]],
                     dict[str, np.ndarray], dict[str, int]]:
    """심볼별 원 데이터를 공통 봉 인덱스에 정렬하고 펀딩을 봉에 매핑한다.

    결측 봉은 NaN 으로 남긴다 — 엔진은 그 봉에서 해당 심볼에 대해 **무행동**한다
    (§4.4-5 fail-closed. 보간·추정 없음).

    Args:
        data: 심볼별 1h OHLCV. fund: 8h 그리드 펀딩. tf: '1h' 또는 '4h'.

    Returns:
        (공통 인덱스, 심볼별 OHLCV 배열, 심볼별 펀딩 배열, 심볼별 내부 결측 봉 수).
    """
    step = "1h" if tf == "1h" else "4h"
    frames = {s: (df if tf == "1h" else resample_4h(df)) for s, df in data.items()}
    lo = min(f.index.min() for f in frames.values())
    hi = max(f.index.max() for f in frames.values())
    idx = pd.date_range(lo.ceil(step), hi.floor(step), freq=step)
    out: dict[str, dict[str, np.ndarray]] = {}
    gaps: dict[str, int] = {}
    for s, f in frames.items():
        rf = f.reindex(idx)
        first = rf["close"].first_valid_index()
        last = rf["close"].last_valid_index()
        inner = rf.loc[first:last, "close"] if first is not None else rf["close"]
        gaps[s] = int(inner.isna().sum())
        out[s] = {c: rf[c].to_numpy(dtype=np.float64) for c in
                  ("open", "high", "low", "close", "volume")}
    settle = np.isin(idx.hour.to_numpy(), FUNDING_HOURS) & (idx.minute.to_numpy() == 0)
    fa = {s: np.where(settle, fund[s].reindex(idx).to_numpy(dtype=np.float64), np.nan)
          for s in data}
    return idx, out, fa, gaps


# ── 요약·저장 ─────────────────────────────────────────────────────────────
def summarize(res: SimResult, specs: Sequence[RuleSpec], tf: str) -> pd.DataFrame:
    """§5 1차 지표와 §5.2 보조 지표를 규칙별로 계산한다 (판정 권한 없음).

    Args:
        res: 시뮬레이션 결과. specs: 규칙 명세. tf: 타임프레임.

    Returns:
        규칙 1행짜리 요약 DataFrame.
    """
    ret = res.returns
    mean = ret.mean(axis=1)
    sd = ret.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr_d = np.where(sd > 0, mean / sd, 0.0)
    sr = sr_d * np.sqrt(365.0)
    sr = np.where((res.trades == 0) | ~np.isfinite(sr), 0.0, sr)   # §5.1 퇴화 처리
    eq = res.equity
    dd = 1.0 - eq / np.maximum.accumulate(eq, axis=1)
    # 수익률 j 는 snap_ts[j] 로 시작하는 UTC 하루를 덮는다 → IS 는 snap_ts[:-1] 기준
    split = int(np.searchsorted(res.snap_ts[:-1], IS_END, side="right"))

    def _sr(block: np.ndarray) -> np.ndarray:
        if block.shape[1] < 2:
            return np.zeros(block.shape[0])
        m, d = block.mean(axis=1), block.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(d > 0, m / d, 0.0) * np.sqrt(365.0)
        return np.where(np.isfinite(v), v, 0.0)

    return pd.DataFrame({
        "rule_id": res.rule_ids,
        "family": [sp.family for sp in specs],
        "params": [",".join(f"{k}={v}" for k, v in sp.params) for sp in specs],
        "exit_code": [sp.exit_code for sp in specs],
        "direction": [sp.direction for sp in specs],
        "tf": tf,
        "sharpe_ann": sr,
        "sharpe_daily": sr_d,
        "n_trades": res.trades,
        "cum_return": eq[:, -1] / eq[:, 0] - 1.0,
        "mdd": dd.max(axis=1),
        "cost_usd": res.cost,
        "funding_usd": res.funding,
        "halt_days": res.halts,
        "is_sharpe_ann": _sr(ret[:, :split]),
        "oos_sharpe_ann": _sr(ret[:, split:]),
    })


def _build_meta(hashes: dict[str, str], specs: Sequence[RuleSpec],
                gaps: dict[str, dict[str, int]], missing_fund: dict[str, int]) -> dict:
    """재현용 메타데이터 (입력 해시·상수·환경)."""
    return {
        "spec": "SWEEP-2026-08-31",
        "engine": Path(__file__).name,
        "engine_sha256": sha256_file(Path(__file__)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_rules": len(specs),
        "n_trials": len(specs) * len(TIMEFRAMES),
        "input_sha256": hashes,
        "window": [str(WIN_START), str(WIN_END)],
        "is_end": str(IS_END),
        "constants": {
            "RT_COST": RT_COST, "RISK": RISK, "STOPLESS_NOTIONAL": STOPLESS_NOTIONAL,
            "GROSS_CAP": GROSS_CAP, "MAX_POS": MAX_POS, "HEAT_CAP": HEAT_CAP,
            "DAILY_HALT": DAILY_HALT, "CELL_CAPITAL": CELL_CAPITAL,
            "STOPLESS_HEAT": STOPLESS_HEAT,
        },
        "interior_gap_bars": gaps,
        "missing_funding_events": missing_fund,
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "pandas": pd.__version__, "platform": sys.platform},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 진입점 — 스윕 1회 실행 후 일수익률 행렬과 요약을 기록한다."""
    ap = argparse.ArgumentParser(description="SWEEP-2026-08-31 백테스트 엔진")
    ap.add_argument("--outdir", default="logs", help="산출물 디렉터리")
    ap.add_argument("--tf", choices=list(TIMEFRAMES), action="append",
                    help="실행할 타임프레임 (기본: 둘 다)")
    ap.add_argument("--dry-run", action="store_true", help="격자만 열거하고 종료")
    ap.add_argument("--no-verify-hashes", action="store_true",
                    help="입력 해시 검증 생략 (비공식 리허설 전용)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    specs = enumerate_rules()
    tfs = tuple(args.tf) if args.tf else TIMEFRAMES
    by_family: dict[str, int] = {}
    for sp in specs:
        by_family[sp.family] = by_family.get(sp.family, 0) + 1
    logger.info("규칙형 %d 개 (계열별: %s)", len(specs),
                ", ".join(f"{k} {v}" for k, v in by_family.items()))
    logger.info("시행 수 = %d 규칙형 × %d 타임프레임 = %d", len(specs), len(tfs),
                len(specs) * len(tfs))
    if len(specs) != N_RULES:
        raise SystemExit(f"규칙형 총계 {len(specs)} != {N_RULES}")
    if args.dry_run:
        return 0

    data, fund, hashes = load_inputs(verify_hashes=not args.no_verify_hashes)
    rid_all: list[str] = []
    ret_all: list[np.ndarray] = []
    eq_all: list[np.ndarray] = []
    sums: list[pd.DataFrame] = []
    gaps_all: dict[str, dict[str, int]] = {}
    miss_all: dict[str, int] = {}
    snap_ts: pd.DatetimeIndex | None = None
    for tf in tfs:
        idx, ohlcv, fa, gaps = align(data, fund, tf)
        gaps_all[tf] = gaps
        if any(gaps.values()):
            logger.warning("%s 내부 결측 봉: %s — 해당 봉 무행동 (fail-closed)", tf, gaps)
        logger.info("%s: 봉 %d 개, 규칙 %d 개 시뮬레이션 시작", tf, len(idx), len(specs))
        res = simulate_timeframe(tf, idx, ohlcv, fa, specs)
        miss_all[tf] = res.missing_funding
        logger.info("%s 완료: 스냅샷 %d 개 (일수익률 %d)", tf, res.equity.shape[1],
                    res.returns.shape[1])
        if res.returns.shape[1] != N_DAYS:
            raise SystemExit(f"{tf} 일수익률 {res.returns.shape[1]} != {N_DAYS}")
        rid_all.extend(res.rule_ids)
        ret_all.append(res.returns)
        eq_all.append(res.equity)
        sums.append(summarize(res, specs, tf))
        snap_ts = res.snap_ts

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    returns = np.vstack(ret_all)
    equity = np.vstack(eq_all)
    meta = _build_meta(hashes, specs, gaps_all, miss_all)
    assert snap_ts is not None
    np.savez_compressed(
        outdir / "sweep_returns.npz",
        daily_returns=returns, daily_equity=equity,
        rule_ids=np.array(rid_all, dtype=object),
        snap_ts=np.array([str(t) for t in snap_ts], dtype=object),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    pd.concat(sums, ignore_index=True).to_csv(outdir / "sweep_summary.csv", index=False)
    (outdir / "sweep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info("기록 완료: %s (일수익률 %s)", outdir / "sweep_returns.npz", returns.shape)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
