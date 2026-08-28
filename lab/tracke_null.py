"""Track E 판정 스크립트 — 공동 max-stat null (사전등록 동결, 2026-08-27).

명세: docs/TRACKE_SCALP_FARM_2026-08-27.md / scratchpad TRACKE_SPEC.md
- 입력: logs/tracke_ledger.csv (셀별 실현 이벤트 원장, gross 손익과 비용 분리).
- 절차 (동결):
  1) [T0, 컷오프] 전체 달력일 그리드에서 10셀 동시 일별 gross 손익 벡터 구성
     (비용·펀딩은 분리 보관, 이벤트 없는 날 = 0, 보간 없음)
  2) 셀별 평균 0 중심화 — null = "비용 차감 전 기대 gross 손익이 0" (zero
     pre-cost gross edge). gross 가 비용만 겨우 메꾸는 전략도 상단을 넘을 수
     있다 — 이는 명세 레시피의 의도된 성질이다.
  3) 동기화 stationary block bootstrap (블록 기하분포 평균 5일, 10,000경로,
     seed=20260827) — 10셀에 같은 날짜 인덱스 (교차상관 보존). 인덱스는 전
     경로를 한 번에 생성하므로 평가 배치 크기와 무관하게 결정론적이다.
  4) 경로별 비용 재차감 — 재표집된 그 날짜의 (펀딩−비용)을 그대로 합산
  5) "10셀 최대 누적수익" 분포의 95% 상단(higher 분위)과 관측 최대를 비교
- 허용 판정 문구는 2가지뿐 (VERDICT_NULL / VERDICT_EXCEED). 다른 해석 금지.
- 판정은 사전 지정 시점만: T+30=2026-09-26, T+90=2026-11-25, T+180=2027-02-23.
  해당 일자(UTC)가 아니면 --force 없이 실행 거부. --force = 비공식 리허설
  모드 (입력 검증 일부 완화, 결과는 판정에 사용 불가).

한계 (동결 시점에 명시):
- 95% 상단은 판정일 1회 기준 — 판정 3회(T+30/90/180) 다중성은 미보정 (명세).
- 추정 대상은 원장의 "실현 이벤트" 손익 — 판정 시점 미청산 포지션의 평가
  손익은 제외된다 (미청산 손실은 그 판정일에 보이지 않는다).
- 달력 그리드는 텔레메트리 결측(러너 중단)을 스스로 탐지하지 못한다 —
  결측 감시는 러너의 갭 fail-closed 로깅이 담당.

원장 계약 (엔진 scalp_farm_runner 가 기록, 실현 이벤트만 — MTM 스냅샷 행 금지):
  실제 스키마 — cell, sym, strategy, bar_close(epoch ms), action, price, qty,
             pnl(= gross, 비용 차감 전 USD), cost(USD, 양수 = 지불), direction.
  펀딩 — 다음 둘 중 하나로 기록돼야 한다:
    (a) funding|funding_pnl|fund 열 (USD 부호 있음, 양수 = 수취, 전부 0 허용)
    (b) action == "funding" 행 — 부호 있는 USD 를 gross(pnl) 열에 싣는다
        (양수 = 수취 = 자본 증가분; 해당 행의 gross 는 0 으로 재분류된다).
  공식 판정 필수 — 펀딩 기록 (a) 또는 (b) 존재 ("펀딩 없음"과 "펀딩 미구현"
             구분), 유일키 열 전부 (cell·sym·strategy·시각·action; 정규화 후
             중복·공란 검사).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 사전등록 상수 (명세 동결 — 변경 금지) ─────────────────────────────
JUDGMENT_DATES: tuple[str, ...] = ("2026-09-26", "2026-11-25", "2027-02-23")
N_PATHS: int = 10_000
MEAN_BLOCK_DAYS: float = 5.0
SEED: int = 20260827
UPPER_Q: float = 0.95
CELLS: tuple[str, ...] = tuple(f"E{i:02d}" for i in range(1, 11))
CELL_CAPITAL: float = 10_000.0            # 셀당 가상 자본 (USD)
LEDGER_PATH = Path("logs/tracke_ledger.csv")
STATE_PATH = Path("logs/tracke_state.json")
MIN_DAYS: int = 10    # 재난 방어용 하한 (통계적 최소 아님 — 실제 보호는 전체 창)

# 허용 판정 문구 2가지 (명세 원문 그대로 — 수정 금지)
VERDICT_NULL = "최대값이 zero-edge 공동 null과 구별되지 않음"
VERDICT_EXCEED = "공동 null 상단(95%) 초과 — 엣지 입증 아님, 별도 전방 확인 필요"

# 원장 열 이름 후보 (엔진 스키마 관용 — scalp_farm_runner: sym/pnl/cost)
_TIME_COLS = ("bar_close", "ts", "time", "day")
_GROSS_COLS = ("gross_pnl", "pnl_gross", "gross", "pnl")
_COST_COLS = ("cost", "costs", "fee", "fees")
_FUND_COLS = ("funding", "funding_pnl", "fund")
_SYM_COLS = ("symbol", "sym")
_FUNDING_ACTION = "funding"
# 실현 이벤트 계약 위반 action (MTM 스냅샷 행 — 추정 대상 오염 금지)
_REJECT_ACTIONS = frozenset({"mtm", "mark", "valuation", "snapshot"})

_EPOCH_RE = re.compile(r"^\d{9,16}(\.\d+)?$")


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """DataFrame 열 이름을 소문자 비교로 후보 중에서 찾는다.

    Args:
        df: 원장 DataFrame.
        candidates: 우선순위 순 열 이름 후보.

    Returns:
        실제 열 이름 (없으면 None).
    """
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def parse_ts(value: object) -> pd.Timestamp:
    """시각 값 1개(ISO 문자열 또는 epoch 초/밀리초)를 UTC Timestamp로 파싱한다.

    행별로 타입을 판별하므로 ISO·epoch 혼합 열도 안전하다 (일괄 파싱 시
    epoch 문자열이 1970년으로 오염되는 함정 차단).

    Args:
        value: 원장 시각 값.

    Returns:
        tz-aware UTC Timestamp.

    Raises:
        ValueError: 파싱 불가 또는 연도가 [2020, 2100] 밖 (오염 감지).
    """
    s = str(value).strip()
    if not s:
        raise ValueError("빈 시각 값")
    if _EPOCH_RE.match(s):
        num = float(s)
        ts = pd.Timestamp(num, unit=("ms" if num > 1e11 else "s"), tz="UTC")
    else:
        ts = pd.Timestamp(s)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if pd.isna(ts) or not 2020 <= ts.year <= 2100:
        raise ValueError(f"시각 범위 이상: {value!r} -> {ts}")
    return ts


def _numeric_strict(series: pd.Series, name: str) -> pd.Series:
    """수치 열을 파싱하고 비유한값(NaN/±inf)이 있으면 fail-closed 한다.

    Args:
        series: 원장 수치 열.
        name: 오류 메시지용 열 이름.

    Returns:
        float64 Series.

    Raises:
        ValueError: 파싱 불가·NaN·±inf 존재 시.
    """
    num = pd.to_numeric(series, errors="coerce")
    bad = ~np.isfinite(num.to_numpy(dtype=float))
    if bad.any():
        raise ValueError(
            f"{name} 열에 비유한값 {int(bad.sum())}개 (예: "
            f"{series[bad].iloc[0]!r}) — fail-closed")
    return num.astype(float)


def _blankish(series: pd.Series) -> pd.Series:
    """정규화된 문자열 열에서 공란·결측 표기를 찾는다.

    pandas 3 의 str dtype 에서는 결측이 NA 로 유지되어 .str 연산을 통과하며
    isin 이 False 가 되므로(가드 무력화), NA 를 먼저 공란으로 치환한다 —
    판정 산식 무변경, fail-closed 가드의 환경 호환 버그픽스 (2026-08-28).
    """
    s = series.where(series.notna(), "")
    return s.astype(str).str.strip().str.lower().isin(
        ("", "nan", "none", "null", "<na>"))


def load_ledger(path: Path, official: bool = False) -> pd.DataFrame:
    """원장 CSV를 (cell, ts, day, gross, cost, funding) 표준형으로 정규화한다.

    fail-closed 검증 (정규화 후 수행): 필수 열 누락, 시각 파싱 실패, 비유한
    수치, 알 수 없는 셀 id, MTM 스냅샷 행(실현 이벤트 계약 위반), 정규화
    유일키 중복, (공식 모드) 펀딩 기록 부재·유일키 열 부재·유일키 공란 —
    전부 ValueError.

    Args:
        path: logs/tracke_ledger.csv 경로.
        official: True면 공식 판정 수준 검증 (펀딩 기록·유일키 완전성 필수).
                  False(리허설)면 해당 항목은 경고로 완화.

    Returns:
        columns = [cell, ts, day, gross, cost, funding] (USD, ts는 UTC).
        action=="funding" 행의 손익은 gross 가 아니라 funding 으로 재분류된다.

    Raises:
        ValueError: 위 검증 실패 시.
        FileNotFoundError: 원장 파일이 없을 때.
    """
    df = pd.read_csv(path)
    cell_col = _pick_col(df, ("cell",))
    time_col = _pick_col(df, _TIME_COLS)
    gross_col = _pick_col(df, _GROSS_COLS)
    cost_col = _pick_col(df, _COST_COLS)
    fund_col = _pick_col(df, _FUND_COLS)
    sym_col = _pick_col(df, _SYM_COLS)
    strat_col = _pick_col(df, ("strategy",))
    action_col = _pick_col(df, ("action",))
    missing = [name for name, col in
               (("cell", cell_col), ("시각", time_col),
                ("gross", gross_col), ("cost", cost_col)) if col is None]
    if missing:
        raise ValueError(
            f"원장 필수 열 없음: {missing} — 실제 열: {list(df.columns)}")

    # ── 정규화 (검증은 전부 정규화된 값 기준) ──────────────────────
    try:
        ts = df[time_col].map(parse_ts)
    except ValueError as e:
        raise ValueError(f"시각 파싱 실패 — {e}") from e
    cells = df[cell_col].astype(str).str.upper().str.strip()
    action = (df[action_col].astype(str).str.lower().str.strip()
              if action_col is not None
              else pd.Series("", index=df.index, dtype=str))
    gross = _numeric_strict(df[gross_col], "gross")
    cost = _numeric_strict(df[cost_col], "cost")

    # 실현 이벤트 계약: MTM 스냅샷 행 금지 (이중계상·추정 대상 변경 방지)
    offending = sorted(set(action) & _REJECT_ACTIONS)
    if offending:
        raise ValueError(f"MTM 스냅샷 행 발견 (실현 이벤트 계약 위반): "
                         f"{offending}")

    # 펀딩: 열이 있으면 열을 쓰고, 없으면 action=='funding' 행의 gross 값을
    # 펀딩(양수 = 수취)으로 재분류한다. 어느 쪽이든 해당 행 gross 는 0.
    is_fund_row = action == _FUNDING_ACTION
    if fund_col is not None:
        funding = _numeric_strict(df[fund_col], "funding")
    else:
        funding = pd.Series(0.0, index=df.index)
        funding[is_fund_row] = gross[is_fund_row]
    gross = gross.mask(is_fund_row, 0.0)
    if official and fund_col is None and not bool(is_fund_row.any()):
        raise ValueError(
            "공식 판정에는 펀딩 기록이 필수다 (funding 열 또는 "
            "action=='funding' 행) — '펀딩 없음'과 '펀딩 미구현' 구분 불가")
    if fund_col is None and not bool(is_fund_row.any()) and not official:
        logger.warning("펀딩 기록 없음 — 리허설이므로 0으로 간주")

    unknown = sorted(set(cells) - set(CELLS))
    if unknown:
        raise ValueError(f"알 수 없는 셀 id: {unknown} — fail-closed")

    # 유일키 검사 — 정규화 후 (cell, sym, strategy, 시각, action)
    if sym_col is not None and strat_col is not None and action_col is not None:
        sym = df[sym_col].astype(str).str.upper().str.strip()
        strat = df[strat_col].astype(str).str.strip()
        blank = _blankish(sym) | _blankish(strat) | _blankish(action)
        if blank.any():
            msg = f"유일키 구성값이 비어 있는 행 {int(blank.sum())}건"
            if official:
                raise ValueError(f"{msg} — fail-closed")
            logger.warning("%s — 리허설이라 통과", msg)
        key = pd.DataFrame({
            "c": cells, "s": sym, "g": strat, "a": action,
            "t": ts.map(lambda t: t.isoformat()),
        })
        dup = key.duplicated()
        if dup.any():
            raise ValueError(f"유일키 중복 이벤트 {int(dup.sum())}건 — "
                             "멱등 재실행 오염 의심, fail-closed")
    elif official:
        raise ValueError("공식 판정에는 유일키 열 전부 필요 "
                         "(cell·sym·strategy·시각·action) — 중복 검사 불가")
    else:
        logger.warning("유일키 열 불완전 — 중복 검사 생략 (리허설)")

    return pd.DataFrame({
        "cell": cells,
        "ts": ts,
        "day": ts.map(lambda t: t.strftime("%Y-%m-%d")),
        "gross": gross.astype(float),
        "cost": cost.astype(float),
        "funding": funding.astype(float),
    })


def resolve_window(
    ledger: pd.DataFrame,
    t0_arg: str | None,
    end_arg: str | None,
    state_path: Path,
    official: bool,
    today: date,
) -> tuple[pd.Timestamp, date]:
    """판정 창 [T0, 컷오프]를 확정한다 — 이벤트 유무와 무관한 고정 창.

    공식 모드: 상태 파일의 사전등록 T0 가 필수 (--t0 는 그 값과 timestamp
    단위로 정확히 일치할 때만 허용 — 대체 불가), 컷오프는 어제(UTC)로 고정
    (--end 변경 금지). 리허설 모드: --t0 > 상태 t0 > 최초 이벤트 시각(경고).

    Args:
        ledger: load_ledger() 결과.
        t0_arg: --t0 CLI 값 (ISO, 선택).
        end_arg: --end CLI 값 (YYYY-MM-DD, 선택).
        state_path: logs/tracke_state.json 경로 (t0 키 조회).
        official: 공식 판정 모드 여부.
        today: 기준일 (UTC).

    Returns:
        (T0 timestamp UTC, 컷오프 date).

    Raises:
        ValueError: T0 미확정/불일치(공식), 컷오프 변경(공식),
                    T0 이전 이벤트 존재, 창 역전 시.
    """
    import json as _json

    state_t0: pd.Timestamp | None = None
    try:
        raw = _json.loads(state_path.read_text(encoding="utf-8")).get("t0")
        if raw:
            state_t0 = parse_ts(raw)
    except (OSError, ValueError, AttributeError):
        state_t0 = None

    if official:
        if state_t0 is None:
            raise ValueError(
                "공식 판정은 상태 파일의 사전등록 T0가 필수다 "
                "(--t0 로 대체 불가 — 이벤트/인자 기반 창 금지)")
        if t0_arg and parse_ts(t0_arg) != state_t0:
            raise ValueError(
                f"--t0({parse_ts(t0_arg).isoformat()})가 사전등록 "
                f"T0({state_t0.isoformat()})와 다르다 — 공식 판정에서 "
                "T0 변경 금지")
        t0 = state_t0
    elif t0_arg:
        t0 = parse_ts(t0_arg)
    elif state_t0 is not None:
        t0 = state_t0
    else:
        if ledger.empty:
            raise ValueError("리허설 T0 추정 불가 — 원장이 비어 있고 t0 없음")
        t0 = ledger["ts"].min()
        print(f"[경고] T0 미확정 — 최초 이벤트 시각 {t0.isoformat()} 사용 "
              "(리허설 전용, 이벤트 의존 창)")

    yesterday = today - timedelta(days=1)
    end = (datetime.strptime(end_arg, "%Y-%m-%d").date() if end_arg
           else yesterday)
    if official and end != yesterday:
        raise ValueError(
            f"공식 판정 컷오프는 어제({yesterday})로 고정 — --end({end}) "
            "변경 금지")
    if end < t0.date():
        raise ValueError(f"창 역전: 컷오프 {end} < T0 {t0.date()}")

    pre = ledger[ledger["ts"] < t0]
    if not pre.empty:
        raise ValueError(
            f"T0({t0.isoformat()}) 이전 이벤트 {len(pre)}건 — 명세상 T0 이전 "
            "주문 생성 금지, 원장 오염 의심 (fail-closed)")
    return t0, end


def build_daily_matrix(
    ledger: pd.DataFrame, t0_day: date, end_day: date,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """[T0, 컷오프] 고정 창에서 10셀 동시 일별 (gross, drag) 행렬을 만든다.

    전 셀이 같은 달력일 그리드를 공유한다 (동기화 재표집의 전제). 이벤트
    없는 날/셀은 0, 컷오프 이후 이벤트(진행 중인 오늘 등)는 제외. 이벤트가
    전혀 없어도 창이 정의되면 전부 0인 행렬이 유효한 결과다.
    drag = (펀딩 수취 − 비용) / 셀 자본 — gross 이외의 결정적 손익 항 전부.

    Args:
        ledger: load_ledger() 결과.
        t0_day: 창 시작일 (포함).
        end_day: 창 종료일 (포함, 완전히 닫힌 날).

    Returns:
        (일자 리스트, gross[T,10], drag[T,10]) — 값은 셀 자본 대비 수익률.
    """
    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(t0_day, end_day, freq="D")]
    day_ix = {d: i for i, d in enumerate(days)}
    cell_ix = {c: j for j, c in enumerate(CELLS)}
    gross = np.zeros((len(days), len(CELLS)))
    drag = np.zeros((len(days), len(CELLS)))
    if ledger.empty:
        return days, gross, drag
    inside = ledger[ledger["day"].isin(day_ix)]
    n_after = int((ledger["day"] > days[-1]).sum())
    if n_after:
        logger.info("컷오프 이후 이벤트 %d건 제외 (진행 중인 날)", n_after)
    grouped = inside.groupby(["day", "cell"], sort=False)[
        ["gross", "cost", "funding"]].sum()
    for (d, c), row in grouped.iterrows():
        i, j = day_ix[d], cell_ix[c]
        gross[i, j] = row["gross"] / CELL_CAPITAL
        drag[i, j] = (row["funding"] - row["cost"]) / CELL_CAPITAL
    return days, gross, drag


def center_gross(gross: np.ndarray) -> np.ndarray:
    """셀별(열별) 평균을 빼서 zero-edge(비용 차감 전 gross 기대 0) null을 만든다.

    Args:
        gross: 일별 gross 수익률 행렬 [T, 10].

    Returns:
        열 평균이 0인 행렬 (원본 비파괴).
    """
    return gross - gross.mean(axis=0, keepdims=True)


def stationary_block_indices(
    t_len: int, n_paths: int, mean_block: float, rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary block bootstrap 날짜 인덱스를 생성한다.

    각 시점에서 확률 p=1/mean_block 로 새 블록(균등 시작점), 아니면 직전+1
    (순환 wrap). 10셀 전부에 같은 인덱스를 적용해 교차상관을 보존한다.

    Args:
        t_len: 관측 일수 T.
        n_paths: 경로 수.
        mean_block: 기하분포 평균 블록 길이 (일).
        rng: 난수 생성기 (결정론 재현용).

    Returns:
        int64 인덱스 배열 [n_paths, T], 값 범위 [0, T).
    """
    p = 1.0 / mean_block
    starts = rng.integers(0, t_len, size=(n_paths, t_len))
    restart = rng.random(size=(n_paths, t_len)) < p
    idx = np.empty((n_paths, t_len), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, t_len):
        cont = (idx[:, t - 1] + 1) % t_len
        idx[:, t] = np.where(restart[:, t], starts[:, t], cont)
    return idx


def bootstrap_max_dist(
    centered: np.ndarray,
    drag: np.ndarray,
    n_paths: int = N_PATHS,
    mean_block: float = MEAN_BLOCK_DAYS,
    seed: int = SEED,
    chunk: int = 1_000,
) -> np.ndarray:
    """동기화 부트스트랩으로 '10셀 최대 누적수익' null 분포를 만든다.

    경로마다: 중심화 gross를 재표집하고, 재표집된 같은 날짜의 drag
    (비용−펀딩)를 그대로 재차감해 셀별 누적수익을 구한 뒤 10셀 최댓값을
    취한다. 인덱스는 전 경로를 단일 rng 스트림에서 한 번에 생성하므로
    chunk(평가 메모리 배치)는 결과에 영향을 주지 않는다.

    Args:
        centered: 평균 0 중심화된 gross [T, 10].
        drag: 비용·펀딩 항 [T, 10] (음수 = 순비용).
        n_paths: 경로 수 (명세 10,000).
        mean_block: 평균 블록 길이 (명세 5일).
        seed: 시드 (명세 20260827).
        chunk: 평가 배치 크기 (메모리 절약 전용 — 결과 불변).

    Returns:
        길이 n_paths 의 최대 누적수익 분포 (수익률 단위).
    """
    combined = centered + drag                       # [T, C]
    idx = stationary_block_indices(
        combined.shape[0], n_paths, mean_block, np.random.default_rng(seed))
    out = np.empty(n_paths)
    for s in range(0, n_paths, chunk):
        sl = idx[s:s + chunk]
        out[s:s + chunk] = combined[sl].sum(axis=1).max(axis=1)
    return out


def observed_max(gross: np.ndarray, drag: np.ndarray) -> tuple[float, str]:
    """관측된 '10셀 최대 누적 순수익'과 해당 셀 id를 구한다.

    Args:
        gross: 일별 gross 수익률 [T, 10] (중심화 전 원본).
        drag: 비용·펀딩 항 [T, 10].

    Returns:
        (최대 누적 순수익, 셀 id). 동률이면 고정 순서상 앞 셀.
    """
    cum = (gross + drag).sum(axis=0)
    j = int(np.argmax(cum))
    return float(cum[j]), CELLS[j]


def upper_quantile(dist: np.ndarray, q: float = UPPER_Q) -> float:
    """분포의 상단 분위를 'higher' 방식(보간 없음)으로 구한다.

    sorted[ceil(q*(n-1))] — 보간형 분위가 관측 경로 수 경계에서 기각을
    부풀리는 것을 막는 보수적 규칙 (numpy 버전 비의존 수동 구현).

    Args:
        dist: null 최대 분포.
        q: 분위 (명세 0.95).

    Returns:
        상단 분위값.
    """
    s = np.sort(np.asarray(dist, dtype=float))
    return float(s[int(np.ceil(q * (len(s) - 1)))])


def mc_p_value(obs_max: float, dist: np.ndarray) -> float:
    """몬테카를로 꼬리확률 (1 + #{null >= 관측}) / (n + 1).

    Args:
        obs_max: 관측 최대 누적수익.
        dist: null 최대 분포.

    Returns:
        보정된 MC p-값.
    """
    return float((1 + int((dist >= obs_max).sum())) / (len(dist) + 1))


def choose_verdict(obs_max: float, dist: np.ndarray, q: float = UPPER_Q) -> str:
    """관측 최대 vs null 분포 상단(higher 분위) 비교로 허용 문구를 고른다.

    95%는 이 판정일 1회 기준이다 — 판정 3회 다중성은 미보정 (명세 동결).

    Args:
        obs_max: 관측 최대 누적수익.
        dist: null 최대 분포.
        q: 상단 분위 (명세 0.95).

    Returns:
        VERDICT_EXCEED (관측 > 상단, strict) 또는 VERDICT_NULL. 다른 문구 없음.
    """
    return VERDICT_EXCEED if obs_max > upper_quantile(dist, q) else VERDICT_NULL


def check_judgment_day(today: date, force: bool) -> bool:
    """오늘(UTC)이 사전 지정 판정일인지 검사한다.

    Args:
        today: 기준일 (UTC).
        force: True면 판정일이 아니어도 허용 (리허설 — 결과는 비공식).

    Returns:
        실행 허용 여부.
    """
    return force or today.isoformat() in JUDGMENT_DATES


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 판정일 가드 후 부트스트랩 판정을 1회 수행한다.

    공식 모드(판정일 실행) = 엄격 검증: 상태 T0 필수, funding 열·유일키 필수.
    --force 리허설 = 판정일 가드 우회 + 검증 일부 완화 (결과는 비공식).

    Args:
        argv: 인자 목록 (테스트 주입용, 기본 sys.argv[1:]).

    Returns:
        종료 코드 (0 정상, 2 가드/데이터 거부).
    """
    parser = argparse.ArgumentParser(
        description="Track E 공동 max-stat null 판정 (사전등록 동결)")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH,
                        help="셀별 원장 CSV 경로 (기본 logs/tracke_ledger.csv)")
    parser.add_argument("--state", type=Path, default=STATE_PATH,
                        help="상태 파일 경로 (t0 조회, 기본 logs/tracke_state.json)")
    parser.add_argument("--t0", default=None,
                        help="T0 (ISO). 공식 판정에서는 등록값과 일치해야 함")
    parser.add_argument("--end", default=None,
                        help="컷오프 일자 YYYY-MM-DD (기본 어제 UTC)")
    parser.add_argument("--force", action="store_true",
                        help="판정일이 아니어도 실행 (리허설 — 결과는 비공식)")
    args = parser.parse_args(argv)

    today = datetime.now(timezone.utc).date()
    if not check_judgment_day(today, args.force):
        print(f"거부: 오늘({today.isoformat()} UTC)은 사전 지정 판정일이 아니다 — "
              f"허용 일자 {', '.join(JUDGMENT_DATES)} (리허설은 --force)")
        return 2
    # --force 는 리허설 전용 — 판정일이어도 --force 가 붙으면 비공식 취급
    official = today.isoformat() in JUDGMENT_DATES and not args.force

    try:
        ledger = load_ledger(args.ledger, official=official)
        t0, end = resolve_window(ledger, args.t0, args.end, args.state,
                                 official, today)
        days, gross, drag = build_daily_matrix(ledger, t0.date(), end)
    except (FileNotFoundError, ValueError) as e:
        print(f"거부: 입력 검증 실패 — {e}")
        return 2
    if len(days) < MIN_DAYS:
        print(f"거부: 관측 {len(days)}일 < 최소 {MIN_DAYS}일 — 판정 불가 "
              "(fail-closed, 어느 쪽 문구도 내지 않음)")
        return 2

    centered = center_gross(gross)
    dist = bootstrap_max_dist(centered, drag)
    if len(dist) != N_PATHS or not np.isfinite(dist).all():
        print("거부: null 분포 이상 (경로 수 또는 비유한값) — fail-closed")
        return 2
    obs, obs_cell = observed_max(gross, drag)
    q95 = upper_quantile(dist)
    p_mc = mc_p_value(obs, dist)
    verdict = choose_verdict(obs, dist)

    print("=" * 72)
    print("Track E — 공동 max-stat null 판정 (lab/tracke_null.py, 사전등록 동결)")
    print("=" * 72)
    if not official:
        print("[비공식 리허설 — 공식 판정 아님 (판정일 미해당 또는 --force)]")
    print(f"창 {days[0]} ~ {days[-1]} ({len(days)}일, T0 {t0.isoformat()}) · "
          f"셀 10 · 경로 {N_PATHS:,} · 블록평균 {MEAN_BLOCK_DAYS:.0f}일 · "
          f"seed {SEED}")
    print(f"관측 최대 누적수익 {obs * 100:+.3f}% (셀 {obs_cell}) · "
          f"null 95% 상단(higher) {q95 * 100:+.3f}% · MC p {p_mc:.4f}")
    print("주: 95%는 이 판정일 1회 기준 (판정 3회 다중성 미보정) · "
          "실현 이벤트 기준 (미청산 평가손익 제외)")
    print(f"판정: {verdict}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
