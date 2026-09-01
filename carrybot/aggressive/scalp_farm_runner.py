"""Track E 라이브 러너 — 매시 실행, 마지막 처리 이후의 닫힌 1h 봉을 전부 재생한다 (멱등).

페이퍼 전용 (실주문·실자금 0, 승급 근거 사용 금지 — docs/TRACKE_SCALP_FARM_2026-08-27.md).

규약:
- T0 동결: 최초 실행에서 t0(실행 시각)와 바스켓 B(봇 유니버스 규칙: USDT 무기한
  거래대금 상위, BTC/ETH/SOL 제외 차상위 3종)를 상태에 기록하고 이후 재계산하지 않는다.
- 워밍업: T0 이전 닫힌 봉(WARMUP_1H개)은 지표 초기화 전용 — 엔진이 주문 생성을 금지한다.
- fail-closed: 캔들·펀딩 수집 실패, 신선한 봉 갭이면 이번 실행을 통째로 중단한다
  (재생 기준이 state.last_ts라 다음 실행이 무손실로 따라잡는다). 중단 사유는
  logs/tracke_last_error.txt 마커와 GITHUB_STEP_SUMMARY 에 남긴다 (종료코드 0 유지 —
  Actions 초록 위장 방지, 상태 커밋 스텝 보존).
- 노화 갭: 구멍의 최신 결측 시각이 현재 최신 봉 대비 DATA_STALL_H(48h) 이상
  과거로 굳으면 그 구멍 시간대만 결측(엔진 심볼별 무행동 fail-closed)으로 두고
  재생을 진행한다 — 신선한 심볼의 영구 갭이 팜 전체를 영구 동결시키는 것 방지.
- 원자성: 원장(tracke_ledger.csv, 유일키 중복 제거) → 이력(tracke_history.csv) →
  상태(tracke_state.json, 커밋 지점) 순서로 임시파일→rename 저장.
- 폐지: 심볼이 거래소 마켓에서 사라지면 마지막 유효가로 전 셀 청산 후 영구 공석.
- 변형 셀 E11·E12 (BRK24TP — scalp_farm.py 동결 명세): 본 셀 재생·저장이 전부
  끝난 뒤에만 별도 단계로 재생하며, 변형의 어떤 실패도 본 셀 커밋을 막지 않는다
  (_safe_variant 격리). 원장·이력은 tracke_variant_ledger.csv /
  tracke_variant_history.csv 분리 파일 전용 — 본 원장 기록 금지 (공식 판정
  계약 보호). 상태는 tracke_state.json 의 variant_cells 키 (t0_variant
  write-once). 변형 단계가 실패한 실행이 있어도 자체 last_ts 로 다음 실행에서
  따라잡는다 (봉·펀딩 따라잡기 수집 fail-closed).
- 변형2 셀 E13~E18 (BRK24GATE·BBMR·RSI2 — scalp_farm.py 동결 명세): E11·E12
  단계 '뒤'의 셋째 단계로 같은 격리 규약을 따른다 (각 단계 _safe_variant 분리 —
  변형2 실패는 본 셀·E11/E12 커밋에 무영향, 역도 성립). 원장·이력은 기존 변형
  파일에 통합 (원장 유일키에 cell 포함 — 행 충돌 없음; 이력은 e13~e18 열 추가,
  keep-last(ts) 관례 유지 — 같은 실행에서 그룹1→그룹2 순 저장이면 마지막 행이
  남고, 다른 그룹 열은 그 시점 상태 스냅숏을 읽는다). 상태는 variant2_cells 키
  (t0_variant2 write-once — 확장 지표 워밍업 수집까지 성공한 최초 원자적 기록
  시각, 수집 실패 시 동결 지연). 본 원장 기록 금지 방화벽 동일.
- 변형3 셀 E19~E21 (전 유니버스 BRK24·BRK24GATE·MR — scalp_farm.py 동결 명세):
  E13~E18 단계 '뒤'의 넷째 단계, 같은 격리 규약 (_safe_variant — 변형3 실패는
  본·E11/E12·E13~E18 커밋에 무영향, 역도 성립). 유니버스는 첫 호출에서 봇
  유니버스 관례(거래대금 상위 40)로 선정해 알파벳순 write-once 동결하고,
  40심볼 전부 fresh 워밍업(WARMUP_1H, 레이트리밋은 ccxt enableRateLimit —
  실패 시 t0 동결 지연 fail-closed 재시도) 후 t0_variant3 를 동결한다.
  이후 실행: 바스켓 중복 심볼은 본 실행 수집분 공유(중복 페치 회피), 유니버스
  전용 심볼만 [vsince, replay_end] 페치 (통상 새 봉 1개 = 1페이지). 폐지·단절은
  본 폐지 미러가 아니라 **자체 판정** (마켓 소멸 검사 + 48h 단절 정책 + 신선
  갭/단절 유예 단계 실패 — 본 러너 규칙 미러). 원장·이력은 기존 변형 파일
  통합 (이력 e19~e21 열 추가, 스키마 이월 관례). 본 팜이 종말(전 심볼 폐지)
  하면 변형3 단계도 더 돌지 않으므로 그 시점 보유 포지션은 미실현으로 동결
  된다 (문서화된 한계 — 페이퍼 전용).
- 변형4 셀 E22·E23 (BRK24R15 = BRK24 + 손익비 1.5:1 익절 — scalp_farm.py 동결
  명세): 변형3 단계 '뒤'의 **다섯째 격리 단계**, 같은 규약 (_safe_variant —
  변형4 실패는 본·v1·v2·v3 커밋에 무영향, 역도 성립). 지표는 본 팜 스냅숏
  상속(E11 전례)이라 별도 워밍업 수집이 없고, 본 팜이 봉을 한 개도 처리하지
  않았으면(last_ts=0) t0 를 동결하지 않는다 (워밍업 fail-closed). 폐지는 본 팜
  폐지의 미러. 원장·이력은 기존 변형 파일 통합 (이력 e22·e23 열 추가, 스키마
  이월 관례). 상태는 variant4_cells 키 (t0_variant4 write-once). 본 원장 기록
  금지 방화벽 동일.
- 변형5 셀 E24·E25 (BBADD = 볼린저 추매 — scalp_farm.py 동결 명세): 변형4
  단계 '뒤'의 **여섯째 격리 단계**, 같은 규약 (_safe_variant — 변형5 실패는
  본·v1~v4 커밋에 무영향, 역도 성립). 지표는 본 팜 스냅숏 상속(E11/E22
  전례 — BBADD 는 본 팜이 추적하는 cl·atr1 만 사용)이라 별도 워밍업 수집이
  없고, 본 팜 미가동(last_ts=0)이면 t0 를 동결하지 않는다 (fail-closed).
  폐지는 본 팜 폐지의 미러. 원장·이력은 기존 변형 파일 통합 (원장 action
  enter/add/exit_signal, 이력 e24·e25 열 추가, 스키마 이월 관례). 상태는
  variant5_cells 키 (t0_variant5 write-once). 본 원장 기록 금지 방화벽 동일.
- 변형6 셀 E26·E27 (AD = AVGDOWN 스윕 전향 검증 — scalp_farm.py 동결 명세):
  변형5 단계 '뒤'의 **일곱째 격리 단계**, 같은 규약 (_safe_variant — 변형6
  실패는 본·v1~v5 커밋에 무영향, 역도 성립). **자체 15m 그리드**: 본 실행의
  1h 수집분을 쓰지 않고 BASKET_A 3종의 15m 봉을 fetch_15m_paged 로 수집한다
  (candle 수집기의 timeframe 인자화 — 기존 1h 경로 fetch_1h_paged 는 무수정).
  첫 호출: 본 팜 가동(last_ts>0) 후 15m 봉 V6_WARMUP_15M(1680 = 420h)개를
  전 심볼 완전 깊이·무갭으로 수집해 warmup_v6 지표 워밍업 → 성공 시에만
  t0_variant6 동결 (write-once, 실패 시 동결 지연 fail-closed). 이후:
  본 팜 폐지 미러 → 15m 수집(v6since = last_ts+M15) → 신선 갭은 단계 실패·
  노화 갭(48h)은 결측 재생(check_gaps 공용) → 펀딩 커버(본 수집분 재사용
  또는 범위 페이지네이션) → step_variant6 재생 → 통합 파일 저장. 원장·이력
  통합 (원장 action enter/add/exit_target, 이력 e26·e27 열 추가, 스키마
  이월 관례 — 이력 ts 는 15m 그리드 값이 섞인다: keep-last(ts) 관례 그대로).
  상태는 variant6_cells 키 (t0_variant6 write-once). 본 원장 기록 금지
  방화벽 동일.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from pathlib import Path

import ccxt
import pandas as pd

from carrybot.aggressive.scalp_farm import (
    BASKET_A,
    CELLS,
    H1,
    M15,
    V2CELLS,
    V2LABELS,
    V3CELLS,
    V3LABELS,
    V3_UNIVERSE_N,
    V4CELLS,
    V4LABELS,
    V5CELLS,
    V5LABELS,
    V6CELLS,
    V6LABELS,
    V6_WARMUP_15M,
    VCELLS,
    VLABELS,
    WARMUP_1H,
    BarE,
    FarmState,
    farm_equities,
    mark_delisted,
    new_farm,
    new_variant,
    new_variant2,
    new_variant3,
    new_variant4,
    new_variant5,
    new_variant6,
    step,
    step_variant,
    step_variant2,
    step_variant3,
    step_variant4,
    step_variant5,
    step_variant6,
    variant2_delist,
    variant2_equities,
    variant2_from_dict,
    variant2_to_dict,
    variant3_delist,
    variant3_equities,
    variant3_from_dict,
    variant3_to_dict,
    variant4_delist,
    variant4_equities,
    variant4_from_dict,
    variant4_to_dict,
    variant5_delist,
    variant5_equities,
    variant5_from_dict,
    variant5_to_dict,
    variant6_delist,
    variant6_equities,
    variant6_from_dict,
    variant6_to_dict,
    variant_delist,
    variant_equities,
    variant_from_dict,
    variant_to_dict,
    warmup_full,
    warmup_v6,
    warmup_x2,
)

logger = logging.getLogger(__name__)

STATE = Path("logs/tracke_state.json")
HIST = Path("logs/tracke_history.csv")
LEDGER = Path("logs/tracke_ledger.csv")
ERR_MARK = Path("logs/tracke_last_error.txt")   # 중단 관측성 마커 (성공 시 삭제)
LEDGER_COLS = ["cell", "sym", "strategy", "bar_close", "action",
               "price", "qty", "pnl", "cost", "direction", "funding"]
LEDGER_KEY = ["cell", "sym", "strategy", "bar_close", "action"]
# 변형 셀(E11·E12 + E13~E18) 통합 파일 — 본 원장(tracke_ledger.csv)과 분리해
# lab/tracke_null.py 공식 10셀 계약(미지 셀 거부)을 보호한다 (기록 교차 금지)
VLEDGER = Path("logs/tracke_variant_ledger.csv")
VHIST = Path("logs/tracke_variant_history.csv")
# 이력 스키마 — e13~e18(변형2)·e19~e21(변형3)·e22~e23(변형4)·e24~e25(변형5)·
# e26~e27(변형6) 열 추가 (구 파일의 결여 열은 0.0 이월 = 부재 표기). equity =
# 행 기록 시점의 변형 전 셀(전 그룹) 시가평가 합, keep-last(ts).
VHIST_COLS = ["day", "ts", "equity", "e11", "e12", "e13", "e14", "e15", "e16",
              "e17", "e18", "e19", "e20", "e21", "e22", "e23", "e24", "e25",
              "e26", "e27", "n_pos", "bars", "fills"]
OFFICIAL_IDS = frozenset(s.cell for s in CELLS)     # E01~E10
VARIANT_IDS = frozenset(s.cell for s in VCELLS)     # E11·E12
VARIANT2_IDS = frozenset(s.cell for s in V2CELLS)   # E13~E18
V2_STRATS = frozenset(s.strategy for s in V2CELLS)  # BRK24GATE·BBMR·RSI2
MIN_TURNOVER = 5_000_000.0      # 봇 유니버스 최소 24h 거래대금 규칙과 동일
MAX_PAGES = 50
DATA_STALL_H = 48               # 이 시간 이상 캔들이 끊긴 심볼 = 데이터 단절 → 영구 공석


def _retry(fn, *a, **k):
    """공개 엔드포인트 재시도 (6회, 선형 백오프). 실패 시 None."""
    for i in range(6):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001 — 네트워크 계열 전반 재시도
            if i == 5:
                logger.error("호출 실패: %s %s", type(exc).__name__, str(exc)[:160])
                return None
            time.sleep(1.5 * (i + 1))
    return None


def pick_basket_b(tickers: list) -> list:
    """봇 유니버스 규칙에서 바스켓 B 선정 (T0 1회, 이후 동결).

    USDT 선형 무기한(만기물 '-' 제외)을 24h 거래대금 내림차순 정렬,
    거래대금 $5M 미만 제외, BTC/ETH/SOL 제외 후 차상위 3종.
    """
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or "-" in sym:
            continue
        coin = sym[:-4]
        if coin in BASKET_A:
            continue
        try:
            vol = float(t.get("turnover24h") or 0.0)
        except (TypeError, ValueError):
            continue
        if vol < MIN_TURNOVER:
            continue
        rows.append((coin, vol))
    rows.sort(key=lambda x: -x[1])
    return [c for c, _ in rows[:3]]


def pick_universe40(tickers: list, markets: dict | None = None) -> list:
    """봇 유니버스 규칙(거래대금 상위 40)에서 변형3 전 유니버스 선정 (T0(v3) 1회).

    USDT 선형 무기한(만기물 '-' 제외)을 24h 거래대금 내림차순 — 동률은 심볼
    오름차순(결정론 tie-break) — 정렬, 비유한 값·$5M(MIN_TURNOVER) 미만 제외,
    BTC/ETH/SOL 포함(바스켓 중복 허용) 상위 V3_UNIVERSE_N종. 같은 코인의 중복
    행은 최대 유효 거래대금으로 집계한다 (입력 순서 무관 — 결정론, Codex 검토).
    markets 가 주어지면 활성·무만기·swap/linear 인 `{coin}/USDT:USDT` 마켓이
    있는 심볼만 남긴다 (메타 키 부재는 통과 — ccxt 통일 심볼 형식 자체가 선형
    스왑을 함의). 선정 시점 거래대금 순위는 감사 보조용으로 로그에만 남긴다
    (write-once 계약은 알파벳순 목록 쪽).

    Returns:
        코인명 **알파벳순** 목록 — 동시 신호 6개 초과 시 선착 순서의 사전 고정.
    """
    best: dict = {}
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or "-" in sym:
            continue
        coin = sym[:-4]
        try:
            vol = float(t.get("turnover24h") or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(vol) or vol < MIN_TURNOVER:
            continue
        if markets is not None:
            m = markets.get(f"{coin}/USDT:USDT")
            if m is None or m.get("active") is False or m.get("swap") is False \
                    or m.get("linear") is False or m.get("expiry") is not None:
                continue
        best[coin] = max(best.get(coin, 0.0), vol)
    rows = sorted(best.items(), key=lambda x: (-x[1], x[0]))
    top = rows[:V3_UNIVERSE_N]
    logger.info("변형3 유니버스 선정 (거래대금 순, 감사용): %s",
                ", ".join(f"{c}={v:.0f}" for c, v in top))
    return sorted(c for c, _ in top)


def missing_hours(have: set, grid: list, first_ts: int) -> list:
    """심볼 자체 시작 이후 구간의 결측 봉 ts 목록 (갭 fail-closed 판정)."""
    return [t for t in grid if t >= first_ts and t not in have]


def contiguous_prefix(d: dict, start: int) -> dict:
    """start부터 1h 간격으로 끊김 없이 이어지는 선두 구간만 남긴다.

    단절 판정을 받은 심볼의 잔여 캔들 재생용 — 내부 갭 이후 꼬리는 버려
    죽어가는 심볼이 전체 재생을 영구 차단하지 않게 한다.
    """
    out = {}
    t = start
    while t in d:
        out[t] = d[t]
        t += H1
    return out


def stalled_syms(latest: dict, overall_end: int) -> list:
    """데이터 단절 심볼 — 최신 심볼 대비 DATA_STALL_H 이상 캔들이 끊긴 것들.

    Args:
        latest: sym -> 마지막으로 확인된 닫힌 봉 ts (없으면 state.last_ts).
        overall_end: 전 심볼 중 가장 최신 닫힌 봉 ts.

    Returns:
        단절 판정(경계 포함: 지연 >= DATA_STALL_H) 심볼 목록.
    """
    cut = overall_end - DATA_STALL_H * H1
    return [s for s, t in latest.items() if t <= cut]


def apply_stall_policy(state: FarmState, syms: list, data: dict, latest: dict,
                       since: int, delist_fn=mark_delisted) -> list:
    """데이터 단절 정책 (명세 §4) — 인과 보존 2단계. syms/data/latest 를 변형한다.

    반드시 선두 절단(fail-closed) 검사보다 **먼저** 호출한다 — 죽은 심볼의
    갭 낀 꼬리가 선두 갭 중단을 유발해 폐지를 영구 회피하는 것을 막는 순서 계약.

    (i) 단절 의심 심볼에 미재생 캔들이 남았으면 since부터의 연속 구간만 남겨
        이번 실행 재생을 그 끝까지로 캡 (replay_end = min 자동 캡 — 청산
        손익이 과거 재생 구간의 자본·사이징에 새지 않는다).
    (ii) 잔여 연속 캔들이 없으면 상태의 마지막 처리 종가로 청산 후 영구 공석.

    Args:
        delist_fn: 폐지 함수 (기본 본 셀 mark_delisted — 변형3 은 variant3_delist
            로 자체 상태에 적용한다. 시그니처 (state, sym) -> fills 동일).

    Returns:
        폐지 청산 체결 목록.
    """
    fills: list = []
    if not any(latest.values()):
        return fills
    overall_end = max(latest.values())
    for s in stalled_syms(latest, overall_end):
        if data[s]:
            start = since if state.last_ts else min(data[s])
            data[s] = contiguous_prefix(data[s], start)
        if data[s]:
            latest[s] = max(data[s])
            logger.warning("%s 단절 의심 — 잔여 %d봉 재생 후 다음 실행에서 폐지 판정",
                           s, len(data[s]))
            continue
        logger.warning("%s 캔들 %d시간 이상 단절 — 마지막 처리 종가로 폐지 처리",
                       s, DATA_STALL_H)
        fills += delist_fn(state, s)
        syms.remove(s)
        data.pop(s)
        latest.pop(s)
    return fills


def _atomic_write(path: Path, text: str) -> None:
    """임시파일→rename 원자적 쓰기."""
    path.parent.mkdir(exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _abort(reason: str) -> None:
    """중단 관측성 — 사유를 마커 파일과 Actions 스텝 요약에 남긴다 (감사 #3).

    종료코드는 0을 유지한다 (상태 커밋 스텝 보존) — 대신 마커가 저장소에
    커밋되어 '초록인데 멈춤'을 밖에서 볼 수 있게 한다. 성공 실행이
    _clear_abort() 로 지운다.

    Args:
        reason: 중단 사유 (한 줄).
    """
    logger.error("%s — fail-closed 중단", reason)
    _atomic_write(ERR_MARK,
                  f"{pd.Timestamp.now(tz='utc').isoformat()} {reason}\n")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(f"\n**Track E 중단**: {reason}\n")
        except OSError as exc:
            logger.warning("GITHUB_STEP_SUMMARY 기록 실패: %s", exc)


def _clear_abort() -> None:
    """성공 실행 — 직전 중단 마커를 지운다 (없으면 무시)."""
    try:
        ERR_MARK.unlink()
    except FileNotFoundError:
        pass


def _read_csv_text(path_or_buf) -> pd.DataFrame:
    """CSV를 필드 문자열 그대로 읽는다 — **수치형 재해석 금지**.

    무결성 교정 2026-08-29 — `append_csv_atomic` 은 append 마다 파일 전체를
    재작성한다. 예전에는 기존 행을 `pd.read_csv` 기본 파싱(float_precision
    ='high' = precise_xstrtod)으로 float 복원한 뒤 다시 직렬화했는데, 그
    파서는 correctly-rounded 가 아니라서 왕복마다 값이 ±1 ULP 흔들렸다.
    실제로 커밋된 라이브 원장에서 기존 행 113건이 그렇게 변조됐다
    (tracke_ledger 71 + tracke_variant_ledger 42) — keep="first" 가 약속한
    "기존 이벤트 불변" 계약 위반. 게다가 이동 방향이 플랫폼(Linux x86-64 /
    macOS arm64)마다 달라, 같은 파일에 append 하는 변형 그룹 수(=재작성
    횟수)가 다른 두 재생이 서로 다른 바이트를 내는 그룹 간 결합을 만들었다.

    dtype=str + NA 추론 해제는 그 재해석 단계를 없앤다. 부수적으로 dtype
    승격 사고(정수 열이 float 로 올라가 "100" → "100.0" 이 되는 것)도 사라진다.

    **보장 범위 (Codex 검토 반영)**: 이건 raw-byte copy-on-write 가 아니다.
    보장은 "pandas `to_csv` 가 만든 정규형 CSV(LF 개행, 불필요한 인용 없음,
    결측은 빈 필드)"에 한정된다 — 손으로 넣은 인용부호·CRLF·BOM 은 재작성에서
    정규화된다. Track E 의 네 CSV 는 전부 이 함수 계열이 생성·관리하는
    정규형이고, 라이브 4개 파일 전부에 대해 재작성 바이트 동일이 실측됐다.
    """
    return pd.read_csv(path_or_buf, dtype=str, keep_default_na=False,
                       na_filter=False, engine="c")


def _text_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """신규 행을 to_csv 직렬화 결과 문자열 프레임으로 정규화.

    기존 행(_read_csv_text)과 같은 표현 공간에 두어 유일키 비교가 어긋나지
    않게 한다 (int 1767… vs str "1767…" 불일치 → 중복 제거 실패 방지).

    직렬화기는 예전 경로와 같은 `to_csv` 지만 **단독** 직렬화라, 예전처럼
    기존 열과 concat 되며 받던 dtype 강제가 없다 (예전: float 열에 정수 100 이
    섞이면 "100.0", 지금: "100"). 현재 원장 스키마에서는 price·qty·pnl·cost·
    funding 이 항상 float, bar_close·direction 이 항상 int 라 신규 행 바이트는
    실측상 예전과 동일하다 — "언제나 동일"은 아니다 (Codex 검토 반영).
    """
    return _read_csv_text(io.StringIO(rows.to_csv(index=False)))


def append_csv_atomic(path: Path, rows: pd.DataFrame, key: list | None = None,
                      keep: str = "first") -> int:
    """CSV에 행 추가 — 유일키 중복 처리 후 임시파일→rename.

    기존 행은 문자열 그대로 옮겨진다 (_read_csv_text — 수치 재해석 없음)
    — 재작성이 기존 이벤트의 값을 바꾸지 못한다는 멱등 계약의 이행.
    보장 범위(정규형 CSV 한정)는 _read_csv_text docstring 참조.

    Args:
        path: CSV 경로.
        rows: 추가할 행.
        key: 유일키 열 (None 이면 중복 제거 없음).
        keep: 중복 시 남길 행 — 원장은 "first"(멱등 계약: 재실행이 기존
            이벤트를 덮지 못함, 불변), 이력은 "last"(폐지 강제청산 뒤 같은
            ts 보존 저장이 최신 자본을 반영해야 함 — 감사 #4).

    Returns:
        추가된 행 수 (대체는 0).
    """
    path.parent.mkdir(exist_ok=True)
    new = _text_frame(rows)
    if path.exists():
        old = _read_csv_text(path)
        for c in new.columns:               # 구 스키마 이월 — NaN 오염 방지
            if c not in old.columns:
                old[c] = "0.0"
        merged = pd.concat([old, new], ignore_index=True) if len(new) else old
    else:
        old, merged = None, new.copy()
    if key:
        merged = merged.drop_duplicates(subset=key, keep=keep)
    added = len(merged) - (0 if old is None else len(old))
    tmp = Path(str(path) + ".tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return added


def check_gaps(data: dict, grid: list, overall_end: int,
               continuing: bool) -> str | None:
    """봉 갭 처분 — 중단 사유 문자열 또는 None(재생 진행) (감사 #2).

    계속 실행(continuing=True)은 grid 시작(=since)부터 이어져야 하므로 선두
    결손도 갭으로 본다. 워밍업 실행은 심볼 자체 첫 봉 이후만 본다 (늦은
    상장의 선행 결측은 갭이 아님).

    노화 판정: 구멍의 최신 결측 시각이 현재 최신 봉(overall_end) 대비
    DATA_STALL_H(48h) 이상 과거로 굳었으면 그 시간대만 결측으로 남기고
    재생을 허용한다 (엔진의 심볼별 fail-closed 무행동이 처리) — 최신 봉이
    신선해 stalled_syms 에 안 걸리는 심볼의 영구 구멍이 매 실행 중단을
    일으켜 팜 전체(state.last_ts)를 영구 동결시키는 경로 차단.
    신선한 갭(<48h)은 일시 수집 결손일 수 있어 기존대로 전체 중단·재시도.

    Args:
        data: sym -> {ts: ohlc} (비어 있지 않은 심볼만).
        grid: 이번 실행 재생 대상 ts 격자 (since..replay_end).
        overall_end: 전 심볼 중 가장 최신 닫힌 봉 ts.
        continuing: state.last_ts 가 있는 계속 실행 여부.

    Returns:
        신선 갭 발견 시 중단 사유, 전부 무갭/노화 갭이면 None.
    """
    aged_cut = overall_end - DATA_STALL_H * H1
    for s, d in data.items():
        have = set(d)
        first = grid[0] if continuing else min(have)
        miss = missing_hours(have, grid, first)
        if not miss:
            continue
        if max(miss) <= aged_cut:
            logger.warning("%s 노화 갭 %d봉 (최신 결측 %d) — 해당 시간대만 "
                           "결측 재생 (심볼별 무행동)", s, len(miss), max(miss))
            continue
        return f"{s} 봉 갭 {len(miss)}개 (예: {miss[0]})"
    return None


def fetch_1h_paged(ex, coin: str, since: int, now_h: int) -> dict | None:
    """닫힌 1h 봉을 since부터 페이지네이션 수집. {ts: (o,h,l,c,vol)} 또는 실패 시 None.

    5번째 원소(vol)는 변형2 게이트 전용 — 본 셀·E11/E12 는 읽지 않는다
    (BarE(t, *tuple) 로 채워지지만 엔진의 해당 경로가 사용하지 않음).
    결측 거래량은 NaN — 게이트만 차단 (fail-closed).
    """
    out: dict = {}
    cur = since
    for _ in range(MAX_PAGES):
        rs = _retry(ex.fetch_ohlcv, f"{coin}/USDT:USDT", "1h", since=cur, limit=1000)
        if rs is None:
            return None
        rs = [r for r in rs if cur <= r[0] < now_h]
        if not rs:
            break
        for r in rs:
            out[int(r[0])] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                              float(r[5]) if len(r) > 5 and r[5] is not None
                              else float("nan"))
        nxt = int(rs[-1][0]) + H1
        if nxt <= cur:
            break
        cur = nxt
        if cur >= now_h:
            break
    return out


def fetch_15m_paged(ex, coin: str, since: int, end: int) -> dict | None:
    """닫힌 15m 봉을 since부터 페이지네이션 수집 — 변형6(E26·E27) 전용.

    fetch_1h_paged 와 같은 계약을 timeframe 인자만 15m 로 바꾼 것 (기존 1h
    경로는 무수정 — 다른 셀 경로 영향 0). {ts: (o,h,l,c)} 또는 실패 시 None.
    거래량은 수집하지 않는다 (변형6 은 사용하지 않음 — BarE vol 기본 NaN).

    Args:
        ex: ccxt bybit 인스턴스.
        coin: 심볼 (예: "BTC").
        since: 시작 ts (포함, 15m 정렬 ms).
        end: 끝 ts (미포함 — 형성중 봉 배제 경계).
    """
    out: dict = {}
    cur = since
    for _ in range(MAX_PAGES):
        rs = _retry(ex.fetch_ohlcv, f"{coin}/USDT:USDT", "15m",
                    since=cur, limit=1000)
        if rs is None:
            return None
        rs = [r for r in rs if cur <= r[0] < end]
        if not rs:
            break
        for r in rs:
            out[int(r[0])] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        nxt = int(rs[-1][0]) + M15
        if nxt <= cur:
            break
        cur = nxt
        if cur >= end:
            break
    return out


def fetch_funding_range(ex, coin: str, need_from: int) -> dict | None:
    """need_from(ms)까지 덮는 펀딩 정산 맵 — endTime 후진 페이지네이션.

    변형 따라잡기 전용: 최신 200건 창(fetch_funding)이 못 덮는 과거 구간을
    endTime 을 물려가며 수집한다 (같은 최신 창 재시도는 회복 수단이 아님 —
    Codex 지적). 상장 초기 도달(빈 응답)은 '그 이전 정산 없음'으로 덮은 것.

    Args:
        ex: ccxt bybit 인스턴스.
        coin: 심볼 (예: "BTC").
        need_from: 이 시각(포함)까지의 정산 이력이 필요하다.

    Returns:
        {정산 ts: 펀딩률 합}. 실패·미커버·무진전 시 None (fail-closed).
    """
    ev: dict = {}
    end: int | None = None
    for _ in range(MAX_PAGES):
        params = {"category": "linear", "symbol": f"{coin}USDT", "limit": "200"}
        if end is not None:
            params["endTime"] = str(end)
        r = _retry(ex.publicGetV5MarketFundingHistory, params)
        if not r or str(r.get("retCode", "")) != "0" \
                or "list" not in r.get("result", {}):
            return None
        rows = r["result"]["list"]
        if not rows:
            return ev                       # 상장 초기 도달 — 커버 완료로 간주
        page: dict = {}
        for x in rows:
            t = int(x["fundingRateTimestamp"])
            page[t] = page.get(t, 0.0) + float(x["fundingRate"])
        for t, v_ in page.items():          # 페이지 경계 중복은 첫 값 유지
            ev.setdefault(t, v_)
        oldest = min(page)
        if oldest <= need_from:
            return ev
        if end is not None and oldest > end:
            return None                     # 진전 없음 (API 이상 — fail-closed)
        end = oldest - 1
    return None                             # 페이지 한도 내 미커버 (fail-closed)


def fetch_funding(ex, coin: str) -> dict | None:
    """최근 펀딩 정산 이벤트 {정산 ts: 펀딩률 합}. 실패 시 None.

    Bybit 알트 무기한은 주기가 8h가 아닐 수 있어 정산 타임스탬프 기반으로 합산한다.
    """
    r = _retry(ex.publicGetV5MarketFundingHistory,
               {"category": "linear", "symbol": f"{coin}USDT", "limit": "200"})
    if not r or str(r.get("retCode", "")) != "0" or "list" not in r.get("result", {}):
        return None                    # 기형 응답을 '펀딩 0'으로 오인 금지 (fail-closed)
    ev: dict = {}
    for x in r["result"]["list"]:
        t = int(x["fundingRateTimestamp"])
        ev[t] = ev.get(t, 0.0) + float(x["fundingRate"])
    return ev


def _save_all(state: FarmState, fills: list, replay_end: int, bars_done: int) -> None:
    """원장 → 이력 → 상태 순서의 원자적 체크포인트 (상태 저장이 커밋 지점).

    원장은 체결이 없어도 헤더 파일을 만든다 — 워크플로 git add 가 존재하지 않는
    경로에서 전체 실패해 첫 실행 상태(T0·바스켓 동결)가 유실되는 사고 방지.

    방화벽: 본 원장에는 공식 셀(E01~E10) 행만 허용 — 변형 행이 섞이면 기록
    대신 예외로 죽는다 (공식 판정 계약 오염의 구조적 차단, 정상 경로 불발).
    """
    bad = {f["cell"] for f in fills} - OFFICIAL_IDS
    if bad:
        raise ValueError(f"본 원장(tracke_ledger.csv)에 비공식 셀 기록 금지: "
                         f"{sorted(bad)}")
    led = pd.DataFrame(fills, columns=LEDGER_COLS)
    n = append_csv_atomic(LEDGER, led, LEDGER_KEY)
    if n:
        logger.info("원장 %d행 추가 (중복 제거 후)", n)
    eqs = farm_equities(state)
    total = sum(eqs.values())
    n_pos = sum(len(c.positions) for c in state.cells.values())
    row = {"day": str(pd.Timestamp(replay_end + H1, unit="ms", tz="utc").date()),
           "ts": replay_end, "equity": round(total, 8)}
    row.update({c.lower(): round(v, 8) for c, v in sorted(eqs.items())})
    row.update({"n_pos": n_pos, "bars": bars_done, "fills": len(fills)})
    # 이력은 keep='last' — 폐지 강제청산 뒤 같은 ts 보존 저장이 최신 자본을
    # 남긴다 (감사 #4). 원장은 위에서 keep='first' 멱등 계약 유지.
    append_csv_atomic(HIST, pd.DataFrame([row]), key=["ts"], keep="last")
    _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    logger.info("팜 자본 %.2f (셀 10개 합), 포지션 %d, 처리 봉 %d, 체결 %d",
                total, n_pos, bars_done, len(fills))


# ── 변형 셀 단계 (E11·E12 → E13~E18 → E19~E21 → E22·E23 순) — 본 셀 커밋
# 뒤에만 실행, 각 단계 실패 격리 ──
# 지위·규칙은 scalp_farm.py 모듈 docstring 의 동결 명세를 따른다. 본 셀
# 원장·이력·상태 경로에는 어떤 기록도 하지 않는다 (variant*_cells 키 제외).


def _mirror_delist(state: FarmState, v: FarmState, delist_fn=variant_delist) -> list:
    """본 팜 폐지를 변형에 미러 — 변형 상태의 마지막 처리 종가로 청산.

    Args:
        delist_fn: 그룹별 폐지 함수 (variant_delist | variant2_delist |
            variant4_delist — 변형3 은 미러하지 않고 자체 판정한다).

    Returns:
        변형 force_exit 체결 목록.
    """
    fills: list = []
    for sym in state.delisted:
        if sym not in v.delisted:
            fills += delist_fn(v, sym)
    return fills


def _vhist_row(state: FarmState, v: FarmState, own_cells: tuple,
               ts_end: int, bars_done: int, n_fills: int) -> dict:
    """통합 변형 이력 행 (VHIST_COLS) — 자기 그룹은 라이브 v, 다른 그룹은
    상태 스냅숏(그 그룹 자체 last_ts 기준 시가평가)에서 채운다.

    keep-last(ts) 관례: 같은 실행에서 그룹1 → 그룹2 순으로 저장되면 마지막
    행이 남는다. 미초기화 그룹 열은 0.0 (구 스키마 이월과 동일 표기), 다른
    그룹 상태가 손상이면 경고 후 0.0 — 각 그룹의 정본 수치는 자기 저장 행.
    bars/fills 는 이 행을 쓴 그룹의 재생 통계다.
    """
    row: dict = {"day": str(pd.Timestamp(ts_end + H1, unit="ms", tz="utc").date()),
                 "ts": ts_end}
    total, n_pos = 0.0, 0
    groups = ((VCELLS, state.variant_cells, variant_from_dict, variant_equities),
              (V2CELLS, state.variant2_cells, variant2_from_dict,
               variant2_equities),
              (V3CELLS, state.variant3_cells, variant3_from_dict,
               variant3_equities),
              (V4CELLS, state.variant4_cells, variant4_from_dict,
               variant4_equities),
              (V5CELLS, state.variant5_cells, variant5_from_dict,
               variant5_equities),
              (V6CELLS, state.variant6_cells, variant6_from_dict,
               variant6_equities))
    for cells, vc, from_d, eq_fn in groups:
        if cells is own_cells:
            g = v
        elif vc is None:
            g = None
        else:
            try:
                g = from_d(vc)
            except ValueError as exc:
                logger.warning("이력 행 — 다른 변형 그룹 상태 손상, 0.0 기록: %s",
                               exc)
                g = None
        if g is None:
            row.update({spec.cell.lower(): 0.0 for spec in cells})
            continue
        eqs = eq_fn(g)
        row.update({c.lower(): round(x, 8) for c, x in sorted(eqs.items())})
        total += sum(eqs.values())
        n_pos += sum(len(cc.positions) for cc in g.cells.values())
    row["equity"] = round(total, 8)
    row.update({"n_pos": n_pos, "bars": bars_done, "fills": n_fills})
    return row


def _save_variant(state: FarmState, v: FarmState, vfills: list,
                  ts_end: int, bars_done: int) -> None:
    """변형 원장 → 변형 이력 → 상태(variant_cells 갱신) 순 원자적 체크포인트.

    본 원장(tracke_ledger.csv)에는 절대 기록하지 않는다 — 공식 판정
    (lab/tracke_null.py, 10셀 동결·미지 셀 거부) 계약 보호를 위한 파일 분리.
    방화벽: E11/E12·전략 BRK24TP 외 행이 섞이면 기록 대신 예외.
    t0_variant 는 write-once — 기존 기록과 다른 t0 저장 시도는 어떤 파일도
    쓰기 전에 예외로 죽는다.
    """
    prev = state.variant_cells
    if prev is not None and prev.get("t0_variant") != v.t0:
        raise ValueError(f"t0_variant 변경 금지 (write-once): "
                         f"{prev.get('t0_variant')} != {v.t0}")
    bad = ({f["cell"] for f in vfills} - VARIANT_IDS) \
        | {f["strategy"] for f in vfills if f["strategy"] != "BRK24TP"}
    if bad:
        raise ValueError(f"변형 원장에 비변형 행 기록 금지: {sorted(bad)}")
    n = append_csv_atomic(VLEDGER, pd.DataFrame(vfills, columns=LEDGER_COLS),
                          LEDGER_KEY)
    if n:
        logger.info("변형 원장 %d행 추가 (중복 제거 후)", n)
    row = _vhist_row(state, v, VCELLS, ts_end, bars_done, len(vfills))
    append_csv_atomic(VHIST, pd.DataFrame([row], columns=VHIST_COLS),
                      key=["ts"], keep="last")
    prev_vc = state.variant_cells
    try:
        state.variant_cells = variant_to_dict(v)
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    except BaseException:
        state.variant_cells = prev_vc   # 실패 격리 — 뒤따르는 변형2 상태 저장이
        raise                           # 미커밋 변이를 대신 영속화하지 못하게 롤백
    logger.info("변형 자본 %.2f (E11+E12), 포지션 %d, 처리 봉 %d, 체결 %d",
                row["e11"] + row["e12"],
                sum(len(c.positions) for c in v.cells.values()),
                bars_done, len(vfills))


def _save_variant2(state: FarmState, v: FarmState, vfills: list,
                   ts_end: int, bars_done: int) -> None:
    """변형2(E13~E18) 원장 → 이력 → 상태(variant2_cells) 순 원자적 체크포인트.

    _save_variant 와 같은 계약 — 파일은 통합(VLEDGER/VHIST), 상태 키·t0 만
    분리. 방화벽: E13~E18·전략 {BRK24GATE, BBMR, RSI2} 외 행이 섞이면 기록
    대신 예외. t0_variant2 는 write-once — 기존 기록과 다른 t0 저장 시도는
    어떤 파일도 쓰기 전에 예외로 죽는다.
    """
    prev = state.variant2_cells
    if prev is not None and prev.get("t0_variant2") != v.t0:
        raise ValueError(f"t0_variant2 변경 금지 (write-once): "
                         f"{prev.get('t0_variant2')} != {v.t0}")
    # (cell, strategy) 쌍 단위 방화벽 — 셀·전략 집합 독립 검사는 E13+BBMR 같은
    # 교차 오염을 놓친다 (Codex 검토 반영)
    allowed = {(s.cell, s.strategy) for s in V2CELLS}
    bad = {(f["cell"], f["strategy"]) for f in vfills} - allowed
    if bad:
        raise ValueError(f"변형2 원장에 비변형2 행 기록 금지: {sorted(bad)}")
    n = append_csv_atomic(VLEDGER, pd.DataFrame(vfills, columns=LEDGER_COLS),
                          LEDGER_KEY)
    if n:
        logger.info("변형2 원장 %d행 추가 (중복 제거 후)", n)
    row = _vhist_row(state, v, V2CELLS, ts_end, bars_done, len(vfills))
    append_csv_atomic(VHIST, pd.DataFrame([row], columns=VHIST_COLS),
                      key=["ts"], keep="last")
    try:
        state.variant2_cells = variant2_to_dict(v)
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    except BaseException:
        state.variant2_cells = prev     # 실패 격리 롤백 (_save_variant 와 대칭)
        raise
    logger.info("변형2 자본 %.2f (E13~E18), 포지션 %d, 처리 봉 %d, 체결 %d",
                sum(row[s.cell.lower()] for s in V2CELLS),
                sum(len(c.positions) for c in v.cells.values()),
                bars_done, len(vfills))


def _run_variant(ex, state: FarmState, data: dict, fund_ev: dict,
                 since: int, replay_end: int) -> None:
    """변형 셀(E11·E12) 재생 — 본 셀 커밋 직후 호출 (실패는 _safe_variant 가 격리).

    첫 호출: t0_variant 동결(write-once) + 본 팜 지표 스냅숏 상속 후 종료 —
    다음 실행부터 재생. 이후: 폐지 미러 → (뒤처졌으면 fail-closed 따라잡기
    수집: 봉은 fetch_1h_paged + check_gaps, 펀딩은 범위 페이지네이션) →
    본 셀과 같은 봉 격자 재생 → 분리 파일 저장.

    Args:
        ex: ccxt bybit (따라잡기 수집 전용 — 정상 경로에서는 미사용).
        state: 본 팜 상태 (variant_cells 키만 갱신됨).
        data: 본 실행이 수집한 sym -> {ts: ohlc} (읽기 전용으로 사용).
        fund_ev: 본 실행이 수집한 sym -> {정산 ts: 펀딩률 합}.
        since: 본 실행 재생 시작 ts.
        replay_end: 본 실행 재생 끝 ts (변형도 같은 끝으로 정렬).
    """
    if state.variant_cells is None:
        v = new_variant(state, t0=int(time.time() * 1000))
        try:
            state.variant_cells = variant_to_dict(v)
            # 헤더 파일 선생성 — 워크플로 git add pathspec 함정 방지 (본 원장 동일 근거)
            append_csv_atomic(VLEDGER, pd.DataFrame([], columns=LEDGER_COLS),
                              LEDGER_KEY)
            append_csv_atomic(VHIST, pd.DataFrame([], columns=VHIST_COLS),
                              key=["ts"])
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1,
                                            default=float))
        except BaseException:
            state.variant_cells = None  # 미기록 t0 를 다른 단계가 영속화 금지 (롤백)
            raise
        logger.info("변형 t0_variant=%d 동결 (write-once) — E11·E12: %s",
                    v.t0, VLABELS["E11"])
        return
    v = variant_from_dict(state.variant_cells)
    if v.basket_b != list(state.basket_b):
        raise ValueError(f"변형 바스켓 B 불일치: {v.basket_b} != {state.basket_b}")
    if v.last_ts > state.last_ts:
        raise ValueError(f"변형 last_ts({v.last_ts})가 본 팜({state.last_ts}) 초과")
    vfills = _mirror_delist(state, v)
    vsince, vdata, vfund = _variant_inputs(ex, v, data, fund_ev, since, replay_end)
    fills2, bars_done = _replay_variant(v, vdata, vfund, vsince, replay_end,
                                        step_variant)
    _save_variant(state, v, vfills + fills2, replay_end, bars_done)


def _variant_inputs(ex, v: FarmState, data: dict, fund_ev: dict,
                    since: int, replay_end: int) -> tuple:
    """변형 그룹 재생 입력 준비 — 뒤처짐 따라잡기 수집 + 펀딩 커버 (fail-closed).

    본 팜 바스켓을 그대로 쓰는 변형 그룹(E11·E12 / E13~E18 / E22·E23) 공용 —
    그룹 무관한 순수 수집 로직 (변형3 은 자체 유니버스라 _variant3_inputs 별도).

    Returns:
        (vsince, vdata, vfund).

    Raises:
        RuntimeError: 따라잡기 봉/펀딩 수집 실패 또는 갭 (그룹 단계만 실패).
    """
    vsince = v.last_ts + H1
    vdata = data
    if vsince < since:                  # 변형이 뒤처짐 (직전 변형 단계 실패) — 따라잡기
        vdata = {s: dict(d) for s, d in data.items()}
        for s in vdata:
            extra = fetch_1h_paged(ex, s, vsince, since)
            if extra is None:
                raise RuntimeError(f"{s} 변형 따라잡기 봉 수집 실패")
            vdata[s].update(extra)
        nonempty = {s: d for s, d in vdata.items() if d}
        if nonempty:
            reason = check_gaps(nonempty, list(range(vsince, replay_end + 1, H1)),
                                max(max(d) for d in nonempty.values()),
                                continuing=True)
            if reason:
                raise RuntimeError(f"변형 따라잡기 갭 — {reason}")
    need = max(vsince, v.t0)
    vfund: dict = {}
    for s in vdata:
        ev = fund_ev.get(s)
        if ev is not None and not (len(ev) >= 200 and ev and min(ev) > need):
            vfund[s] = ev               # 본 실행 수집분이 변형 구간을 덮는다
            continue
        rng = fetch_funding_range(ex, s, need)
        if rng is None:
            raise RuntimeError(f"{s} 변형 펀딩 범위 수집 실패")
        vfund[s] = rng
    return vsince, vdata, vfund


def _replay_variant(v: FarmState, vdata: dict, vfund: dict, vsince: int,
                    replay_end: int, step_fn) -> tuple:
    """변형 그룹 봉 격자 재생 공용 — (체결 목록, 처리 봉 수) 반환.

    Args:
        step_fn: 그룹별 step (step_variant | step_variant2 | step_variant3 |
            step_variant4).
    """
    fills: list = []
    bars_done = 0
    for t in range(vsince, replay_end + 1, H1):
        bars = {s: BarE(t, *vdata[s][t]) for s in vdata if t in vdata[s]}
        if not bars:
            continue
        fmap = {s: vfund.get(s, {}).get(t + H1, 0.0) for s in bars}
        fills += step_fn(v, bars, fmap)
        bars_done += 1
    return fills, bars_done


def _migrate_vhist_schema() -> None:
    """VHIST 를 VHIST_COLS 스키마·열 순서로 정렬 (구 파일 e13~e18 = 0.0 이월).

    append_csv_atomic 은 결여 열을 파일 끝에 붙일 뿐 열 '순서'는 못 맞춘다 —
    변형2 초기화 시 1회 정렬해 파일을 선언 스키마와 일치시킨다 (행 값 불변,
    Codex 검토 반영). 파일이 없으면 헤더만 만든다 (git add pathspec 함정 방지).
    """
    old = _read_csv_text(VHIST) if VHIST.exists() \
        else pd.DataFrame(columns=VHIST_COLS, dtype=str)
    for c in VHIST_COLS:
        if c not in old.columns:
            old[c] = "0.0"
    _atomic_write(VHIST, old[VHIST_COLS].to_csv(index=False))


def _run_variant2(ex, state: FarmState, data: dict, fund_ev: dict,
                  since: int, replay_end: int) -> None:
    """변형2 셀(E13~E18) 재생 — E11·E12 단계 뒤 호출 (_safe_variant 격리).

    첫 호출: 확장 지표(x2) 워밍업 수집(WARMUP_1H 봉, state.last_ts 까지) →
    성공 시에만 t0_variant2 동결(write-once) 후 종료 — 다음 실행부터 재생
    (수집 실패는 예외 → 동결 지연, 다음 실행 재시도. 기존 T0 원칙 그대로
    t0 이전 주문은 구조적으로 불가능). 이후: E11·E12 와 동일한
    폐지 미러 → 따라잡기 수집 → 재생 → 통합 파일 저장.

    Args:
        ex: ccxt bybit (워밍업·따라잡기 수집 전용).
        state: 본 팜 상태 (variant2_cells 키만 갱신됨).
        data: 본 실행이 수집한 sym -> {ts: (o,h,l,c,vol)} (읽기 전용).
        fund_ev: 본 실행이 수집한 sym -> {정산 ts: 펀딩률 합}.
        since: 본 실행 재생 시작 ts.
        replay_end: 본 실행 재생 끝 ts (변형2도 같은 끝으로 정렬).
    """
    if state.variant2_cells is None:
        if not state.last_ts:
            return                      # 본 팜 미가동 — 다음 실행에서 초기화
        v = new_variant2(state, t0=int(time.time() * 1000))
        w_since = state.last_ts - (WARMUP_1H - 1) * H1
        for s in [x for x in list(BASKET_A) + list(state.basket_b)
                  if x not in state.delisted]:
            rows = fetch_1h_paged(ex, s, w_since, state.last_ts + H1)
            # 워밍업 완결성 검증 (Codex 검토 반영) — 빈/부분 응답으로 t0 를
            # 식은 지표 채로 동결하는 것 금지: 반드시 state.last_ts 까지 덮고,
            # 내부 갭 앞 구간은 버려(연속 꼬리) 지표 정렬을 보장한다.
            if not rows or max(rows) != state.last_ts:
                raise RuntimeError(
                    f"{s} 변형2 워밍업 수집 불완전 — t0 동결 지연 (fail-closed)")
            tail = []
            t = state.last_ts
            while t in rows:
                tail.append(t)
                t -= H1
            tail.reverse()
            if s in BASKET_A and len(tail) < WARMUP_1H:
                # 상장 이력이 확실한 바스켓 A 는 완전 깊이 필수 (본 러너 워밍업
                # 관례와 동일 — 짧은 응답을 '새 상장'으로 오인 금지). 바스켓 B
                # 는 늦은 상장 가능 — 연속 꼬리만으로 시작 (지표 늦게 형성).
                raise RuntimeError(
                    f"{s} 변형2 워밍업 깊이 부족({len(tail)}/{WARMUP_1H}) — "
                    f"t0 동결 지연 (fail-closed)")
            if len(tail) != len(rows):
                # 꼬리 밖 관측 봉 존재 = 내부 갭 (늦은 상장이 아님) — 일시
                # 수집 결손일 수 있으므로 동결하지 말고 재시도 (Codex 검토 반영:
                # 갭 앞 유효 관측을 조용히 버리는 제3의 정책 금지)
                raise RuntimeError(
                    f"{s} 변형2 워밍업 내부 갭({len(rows) - len(tail)}봉) — "
                    f"t0 동결 지연 (fail-closed)")
            warmup_x2(v, s, [(rows[t][3],
                              rows[t][4] if len(rows[t]) > 4 else float("nan"))
                             for t in tail])
        try:
            state.variant2_cells = variant2_to_dict(v)
            # 헤더 선생성 + 이력 스키마 이월 — git add pathspec 함정 방지 겸
            # 구 파일 열 순서를 선언 스키마(VHIST_COLS)로 정렬 (값 불변)
            append_csv_atomic(VLEDGER, pd.DataFrame([], columns=LEDGER_COLS),
                              LEDGER_KEY)
            _migrate_vhist_schema()
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1,
                                            default=float))
        except BaseException:
            state.variant2_cells = None   # 미기록 t0 영속화 금지 (롤백)
            raise
        logger.info("변형2 t0_variant2=%d 동결 (write-once) — E13~E18: %s / %s / %s",
                    v.t0, V2LABELS["E13"], V2LABELS["E15"], V2LABELS["E17"])
        return
    v = variant2_from_dict(state.variant2_cells)
    if v.basket_b != list(state.basket_b):
        raise ValueError(f"변형2 바스켓 B 불일치: {v.basket_b} != {state.basket_b}")
    if v.last_ts > state.last_ts:
        raise ValueError(f"변형2 last_ts({v.last_ts})가 본 팜({state.last_ts}) 초과")
    vfills = _mirror_delist(state, v, variant2_delist)
    vsince, vdata, vfund = _variant_inputs(ex, v, data, fund_ev, since, replay_end)
    fills2, bars_done = _replay_variant(v, vdata, vfund, vsince, replay_end,
                                        step_variant2)
    _save_variant2(state, v, vfills + fills2, replay_end, bars_done)


def _save_variant3(state: FarmState, v: FarmState, vfills: list,
                   ts_end: int, bars_done: int) -> None:
    """변형3(E19~E21) 원장 → 이력 → 상태(variant3_cells) 순 원자적 체크포인트.

    _save_variant2 와 같은 계약 — 파일 통합(VLEDGER/VHIST), 상태 키·t0 분리.
    방화벽: (cell, strategy) 쌍이 V3CELLS 밖이면 기록 대신 예외 (E19~E21 의
    전략명 BRK24/BRK24GATE/MR 이 본·변형2 와 겹치므로 쌍 단위가 필수).
    t0_variant3 와 유니버스(basket_b)는 write-once — 다른 값 저장 시도는
    어떤 파일도 쓰기 전에 예외로 죽는다.
    """
    prev = state.variant3_cells
    if prev is not None and prev.get("t0_variant3") != v.t0:
        raise ValueError(f"t0_variant3 변경 금지 (write-once): "
                         f"{prev.get('t0_variant3')} != {v.t0}")
    if prev is not None and list(prev.get("basket_b", [])) != list(v.basket_b):
        raise ValueError(f"변형3 유니버스 변경 금지 (write-once): "
                         f"{prev.get('basket_b')} != {v.basket_b}")
    if prev is not None and not set(prev.get("delisted", [])) <= set(v.delisted):
        raise ValueError(f"변형3 영구 공석 축소 금지 (단조 증가): "
                         f"{prev.get('delisted')} -> {v.delisted}")
    allowed = {(s.cell, s.strategy) for s in V3CELLS}
    bad = {(f["cell"], f["strategy"]) for f in vfills} - allowed
    if bad:
        raise ValueError(f"변형3 원장에 비변형3 행 기록 금지: {sorted(bad)}")
    n = append_csv_atomic(VLEDGER, pd.DataFrame(vfills, columns=LEDGER_COLS),
                          LEDGER_KEY)
    if n:
        logger.info("변형3 원장 %d행 추가 (중복 제거 후)", n)
    row = _vhist_row(state, v, V3CELLS, ts_end, bars_done, len(vfills))
    append_csv_atomic(VHIST, pd.DataFrame([row], columns=VHIST_COLS),
                      key=["ts"], keep="last")
    try:
        state.variant3_cells = variant3_to_dict(v)
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    except BaseException:
        state.variant3_cells = prev     # 실패 격리 롤백 (_save_variant 와 대칭)
        raise
    logger.info("변형3 자본 %.2f (E19~E21), 포지션 %d, 처리 봉 %d, 체결 %d",
                sum(row[s.cell.lower()] for s in V3CELLS),
                sum(len(c.positions) for c in v.cells.values()),
                bars_done, len(vfills))


def _variant3_inputs(ex, v: FarmState, data: dict, fund_ev: dict,
                     since: int, replay_end: int) -> tuple:
    """변형3(E19~E21) 재생 입력 — 유니버스 캔들·펀딩 수집 + 자체 단절 정책.

    바스켓 중복 심볼은 본 실행 수집분(data)을 공유해 재페치를 피하고
    (공유 캐시), 유니버스 전용 심볼만 [vsince, replay_end] 구간을 페치한다
    (통상 새 봉 1개 = 1페이지 — 페이지네이션 최소). 단절·재생 끝 규칙은 본
    러너 미러: 48h 단절은 잔여 연속 구간 재생 후 폐지(apply_stall_policy,
    variant3_delist), v3_end = min(비단절 심볼 최신 봉, 본 replay_end),
    신선 갭·새 봉 없음(단절 유예)은 그룹 단계 실패 (다음 실행 따라잡기).
    펀딩은 재생할 봉이 있을 때만 수집한다 (본 수집분이 덮으면 재사용).

    Returns:
        (vsince, v3_end, vdata, vfund, dfills) — v3_end < vsince 면 재생 없음.

    Raises:
        RuntimeError: 봉/펀딩 수집 실패·신선 갭·단절 유예 (변형3 단계만 실패).
    """
    vsince = v.last_ts + H1
    syms = [s for s in v.basket_b if s not in v.delisted]
    vdata: dict = {}
    latest: dict = {}
    for s in syms:
        have = data.get(s)
        if have is not None and vsince >= since:
            d = dict(have)                  # 본 수집분이 [vsince, replay_end] 커버
        else:
            d = dict(have) if have is not None else {}
            end = since if have is not None else replay_end + H1
            extra = fetch_1h_paged(ex, s, vsince, end)
            if extra is None:
                raise RuntimeError(f"{s} 변형3 봉 수집 실패")
            d.update(extra)
        vdata[s] = d
        latest[s] = max(d) if d else v.last_ts
    dfills = apply_stall_policy(v, syms, vdata, latest, vsince,
                                delist_fn=variant3_delist)
    for s in syms:
        if not vdata[s]:
            raise RuntimeError(f"{s} 변형3 새 봉 없음 (단절 유예 중)")
    if not syms:
        return vsince, v.last_ts, {}, {}, dfills
    v3_end = min(replay_end, min(latest[s] for s in syms))
    if v3_end < vsince:
        return vsince, v3_end, {}, {}, dfills
    reason = check_gaps(vdata, list(range(vsince, v3_end + 1, H1)),
                        max(latest.values()), continuing=True)
    if reason:
        raise RuntimeError(f"변형3 갭 — {reason}")
    need = max(vsince, v.t0)
    vfund: dict = {}
    for s in vdata:
        ev = fund_ev.get(s)
        if ev is not None and not (len(ev) >= 200 and ev and min(ev) > need):
            vfund[s] = ev                   # 본 실행 수집분이 변형3 구간을 덮는다
            continue
        rng = fetch_funding_range(ex, s, need)
        if rng is None:
            raise RuntimeError(f"{s} 변형3 펀딩 범위 수집 실패")
        vfund[s] = rng
    return vsince, v3_end, vdata, vfund, dfills


def _run_variant3(ex, state: FarmState, data: dict, fund_ev: dict,
                  since: int, replay_end: int) -> None:
    """변형3 셀(E19~E21) 재생 — E13~E18 단계 뒤 넷째 단계 (_safe_variant 격리).

    첫 호출: 티커 조회 → pick_universe40 (40 미만이면 동결 지연 fail-closed) →
    유니버스 40심볼 전부 fresh 워밍업(WARMUP_1H, warmup_full — 본 팜 지표
    상속 없음) → 성공 시에만 t0_variant3 + 유니버스 동결(write-once) 후 종료.
    이후: 자체 마켓 소멸 검사 → 자체 수집/단절 정책(_variant3_inputs) →
    재생 → 통합 파일 저장. 본 폐지 미러는 하지 않는다 (자체 판정 —
    본 팜의 과거 단절 판정이 회복 심볼을 배제하지 않음, 모듈 docstring).

    Args:
        ex: ccxt bybit (티커·워밍업·유니버스 수집).
        state: 본 팜 상태 (variant3_cells 키만 갱신됨).
        data: 본 실행이 수집한 sym -> {ts: (o,h,l,c,vol)} (읽기 전용 공유 캐시).
        fund_ev: 본 실행이 수집한 sym -> {정산 ts: 펀딩률 합}.
        since: 본 실행 재생 시작 ts.
        replay_end: 본 실행 재생 끝 ts (변형3 재생 끝의 상한).
    """
    if state.variant3_cells is None:
        if not state.last_ts:
            return                      # 본 팜 미가동 — 다음 실행에서 초기화
        r = _retry(ex.publicGetV5MarketTickers, {"category": "linear"})
        if not r or str(r.get("retCode", "")) != "0":
            raise RuntimeError("변형3 티커 조회 실패 — t0 동결 지연 (fail-closed)")
        uni = pick_universe40(r.get("result", {}).get("list", []),
                              getattr(ex, "markets", None))
        if len(uni) < V3_UNIVERSE_N:
            raise RuntimeError(f"변형3 유니버스 후보 부족({len(uni)}/"
                               f"{V3_UNIVERSE_N}) — t0 동결 지연 (fail-closed)")
        v = new_variant3(state, uni, t0=int(time.time() * 1000))
        w_since = state.last_ts - (WARMUP_1H - 1) * H1
        for s in v.basket_b:
            rows = fetch_1h_paged(ex, s, w_since, state.last_ts + H1)
            # 워밍업 완결성 검증 — 변형2 전례 그대로: last_ts 까지 덮어야 하고,
            # 연속 꼬리만 사용, 꼬리 밖 관측 봉 = 내부 갭 재시도 (fail-closed)
            if not rows or max(rows) != state.last_ts:
                raise RuntimeError(
                    f"{s} 변형3 워밍업 수집 불완전 — t0 동결 지연 (fail-closed)")
            tail = []
            t = state.last_ts
            while t in rows:
                tail.append(t)
                t -= H1
            tail.reverse()
            if s in BASKET_A and len(tail) < WARMUP_1H:
                # 상장 이력이 확실한 메이저는 완전 깊이 필수 — 그 외 유니버스
                # 심볼은 늦은 상장 가능, 연속 꼬리만으로 시작 (지표 늦게 형성)
                raise RuntimeError(
                    f"{s} 변형3 워밍업 깊이 부족({len(tail)}/{WARMUP_1H}) — "
                    f"t0 동결 지연 (fail-closed)")
            if len(tail) != len(rows):
                raise RuntimeError(
                    f"{s} 변형3 워밍업 내부 갭({len(rows) - len(tail)}봉) — "
                    f"t0 동결 지연 (fail-closed)")
            warmup_full(v, s, [rows[t] for t in tail])
        try:
            state.variant3_cells = variant3_to_dict(v)
            # 헤더 선생성 + 이력 스키마 이월 (e19~e21 열 정렬 — 변형2 전례)
            append_csv_atomic(VLEDGER, pd.DataFrame([], columns=LEDGER_COLS),
                              LEDGER_KEY)
            _migrate_vhist_schema()
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1,
                                            default=float))
        except BaseException:
            state.variant3_cells = None   # 미기록 t0 영속화 금지 (롤백)
            raise
        logger.info("변형3 t0_variant3=%d 동결 (write-once) — E19~E21: %s / "
                    "유니버스 40종(알파벳순): %s",
                    v.t0, V3LABELS["E19"], ",".join(v.basket_b))
        return
    v = variant3_from_dict(state.variant3_cells)
    if v.last_ts > state.last_ts:
        raise ValueError(f"변형3 last_ts({v.last_ts})가 본 팜({state.last_ts}) 초과")
    # 자체 마켓 소멸 검사 (본 폐지 미러 아님) — 죽은 마켓 페치 실패로 단계가
    # 영구 실패하는 것 방지. 마켓 정보가 없으면(테스트 더블 등) 이번 실행은
    # 생략 — 그 사이 죽은 마켓은 수집 실패로 이 단계만 실패하고, 마켓이 로드된
    # 다음 실행에서 여기서 폐지된다 (살아있는 마켓의 데이터 단절은 48h 정책).
    mk = getattr(ex, "markets", None)
    vfills: list = []
    if isinstance(mk, dict) and mk:
        for s in [x for x in v.basket_b if x not in v.delisted]:
            m = mk.get(f"{s}/USDT:USDT")
            if m is None or m.get("active") is False:
                logger.warning("%s 변형3 마켓 소멸/비활성 — 폐지 처리", s)
                vfills += variant3_delist(v, s)
    vsince, v3_end, vdata, vfund, dfills = _variant3_inputs(
        ex, v, data, fund_ev, since, replay_end)
    fills2, bars_done = _replay_variant(v, vdata, vfund, vsince, v3_end,
                                        step_variant3)
    _save_variant3(state, v, vfills + dfills + fills2,
                   max(v3_end, v.last_ts), bars_done)


def _finalize_variant(state: FarmState) -> None:
    """폐지만 있는 조기 종료 분기용 — 초기화된 변형에 폐지 미러만 수행·저장.

    변형 포지션이 본 팜 종말(전 심볼 폐지 등) 뒤에도 미실현으로 영구 방치되는
    것을 막는다 (Codex 지적). 미초기화(None)면 아무것도 하지 않는다 —
    죽은 팜에 빈 변형을 새로 만들지 않음.
    """
    if state.variant_cells is None:
        return
    v = variant_from_dict(state.variant_cells)
    missing = [s for s in state.delisted if s not in v.delisted]
    if not missing:
        return
    vfills = _mirror_delist(state, v)      # 포지션이 없어도 공석 표시는 저장해야 함
    _save_variant(state, v, vfills, v.last_ts or state.last_ts, 0)


def _finalize_variant2(state: FarmState) -> None:
    """변형2(E13~E18) 종말 폐지 미러 — _finalize_variant 와 동일 계약, 그룹 분리."""
    if state.variant2_cells is None:
        return
    v = variant2_from_dict(state.variant2_cells)
    missing = [s for s in state.delisted if s not in v.delisted]
    if not missing:
        return
    vfills = _mirror_delist(state, v, variant2_delist)
    _save_variant2(state, v, vfills, v.last_ts or state.last_ts, 0)


def _save_variant4(state: FarmState, v: FarmState, vfills: list,
                   ts_end: int, bars_done: int) -> None:
    """변형4(E22·E23) 원장 → 이력 → 상태(variant4_cells) 순 원자적 체크포인트.

    _save_variant2/3 과 같은 계약 — 파일 통합(VLEDGER/VHIST), 상태 키·t0 만
    분리. 방화벽: (cell, strategy) 쌍이 V4CELLS 밖이면 기록 대신 예외
    (셀·전략 집합 독립 검사는 교차 오염을 놓친다 — 변형2 전례).
    t0_variant4 는 write-once — 기존 기록과 다른 t0 저장 시도는 어떤 파일도
    쓰기 전에 예외로 죽는다.
    """
    prev = state.variant4_cells
    if prev is not None and prev.get("t0_variant4") != v.t0:
        raise ValueError(f"t0_variant4 변경 금지 (write-once): "
                         f"{prev.get('t0_variant4')} != {v.t0}")
    allowed = {(s.cell, s.strategy) for s in V4CELLS}
    bad = {(f["cell"], f["strategy"]) for f in vfills} - allowed
    if bad:
        raise ValueError(f"변형4 원장에 비변형4 행 기록 금지: {sorted(bad)}")
    n = append_csv_atomic(VLEDGER, pd.DataFrame(vfills, columns=LEDGER_COLS),
                          LEDGER_KEY)
    if n:
        logger.info("변형4 원장 %d행 추가 (중복 제거 후)", n)
    row = _vhist_row(state, v, V4CELLS, ts_end, bars_done, len(vfills))
    append_csv_atomic(VHIST, pd.DataFrame([row], columns=VHIST_COLS),
                      key=["ts"], keep="last")
    try:
        state.variant4_cells = variant4_to_dict(v)
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    except BaseException:
        state.variant4_cells = prev     # 실패 격리 롤백 (_save_variant 와 대칭)
        raise
    logger.info("변형4 자본 %.2f (E22+E23), 포지션 %d, 처리 봉 %d, 체결 %d",
                sum(row[s.cell.lower()] for s in V4CELLS),
                sum(len(c.positions) for c in v.cells.values()),
                bars_done, len(vfills))


def _run_variant4(ex, state: FarmState, data: dict, fund_ev: dict,
                  since: int, replay_end: int) -> None:
    """변형4 셀(E22·E23) 재생 — 변형3 단계 뒤 다섯째 단계 (_safe_variant 격리).

    첫 호출: 본 팜이 봉을 처리한 적이 있어야(last_ts>0) t0_variant4 동결
    (write-once) — 지표는 본 팜 스냅숏 상속이므로 별도 워밍업 수집이 없고,
    본 팜 미가동 상태의 빈 지표로 동결하지 않는다 (fail-closed, 다음 실행
    재시도). 동결 실행은 재생 없이 종료하고 다음 실행부터 재생한다.
    이후: E11·E12 와 동일한 폐지 미러 → 따라잡기 수집 → 재생 → 통합 파일 저장.

    Args:
        ex: ccxt bybit (따라잡기 수집 전용 — 정상 경로에서는 미사용).
        state: 본 팜 상태 (variant4_cells 키만 갱신됨).
        data: 본 실행이 수집한 sym -> {ts: ohlc} (읽기 전용으로 사용).
        fund_ev: 본 실행이 수집한 sym -> {정산 ts: 펀딩률 합}.
        since: 본 실행 재생 시작 ts.
        replay_end: 본 실행 재생 끝 ts (변형4도 같은 끝으로 정렬).
    """
    if state.variant4_cells is None:
        if not state.last_ts:
            return                      # 본 팜 미가동 — 다음 실행에서 초기화
        v = new_variant4(state, t0=int(time.time() * 1000))
        try:
            state.variant4_cells = variant4_to_dict(v)
            # 헤더 선생성 + 이력 스키마 이월 (e22·e23 열 정렬 — 변형2/3 전례)
            append_csv_atomic(VLEDGER, pd.DataFrame([], columns=LEDGER_COLS),
                              LEDGER_KEY)
            _migrate_vhist_schema()
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1,
                                            default=float))
        except BaseException:
            state.variant4_cells = None   # 미기록 t0 영속화 금지 (롤백)
            raise
        logger.info("변형4 t0_variant4=%d 동결 (write-once) — E22·E23: %s",
                    v.t0, V4LABELS["E22"])
        return
    v = variant4_from_dict(state.variant4_cells)
    if v.basket_b != list(state.basket_b):
        raise ValueError(f"변형4 바스켓 B 불일치: {v.basket_b} != {state.basket_b}")
    if v.last_ts > state.last_ts:
        raise ValueError(f"변형4 last_ts({v.last_ts})가 본 팜({state.last_ts}) 초과")
    vfills = _mirror_delist(state, v, variant4_delist)
    vsince, vdata, vfund = _variant_inputs(ex, v, data, fund_ev, since, replay_end)
    fills2, bars_done = _replay_variant(v, vdata, vfund, vsince, replay_end,
                                        step_variant4)
    _save_variant4(state, v, vfills + fills2, replay_end, bars_done)


def _finalize_variant4(state: FarmState) -> None:
    """변형4(E22·E23) 종말 폐지 미러 — _finalize_variant2 와 동일 계약, 그룹 분리."""
    if state.variant4_cells is None:
        return
    v = variant4_from_dict(state.variant4_cells)
    missing = [s for s in state.delisted if s not in v.delisted]
    if not missing:
        return
    vfills = _mirror_delist(state, v, variant4_delist)
    _save_variant4(state, v, vfills, v.last_ts or state.last_ts, 0)


def _save_variant5(state: FarmState, v: FarmState, vfills: list,
                   ts_end: int, bars_done: int) -> None:
    """변형5(E24·E25) 원장 → 이력 → 상태(variant5_cells) 순 원자적 체크포인트.

    _save_variant4 와 같은 계약 — 파일 통합(VLEDGER/VHIST), 상태 키·t0 만
    분리. 방화벽: (cell, strategy) 쌍이 V5CELLS 밖이면 기록 대신 예외
    (셀·전략 집합 독립 검사는 교차 오염을 놓친다 — 변형2 전례).
    t0_variant5 는 write-once — 기존 기록과 다른 t0 저장 시도는 어떤 파일도
    쓰기 전에 예외로 죽는다.
    """
    prev = state.variant5_cells
    if prev is not None and prev.get("t0_variant5") != v.t0:
        raise ValueError(f"t0_variant5 변경 금지 (write-once): "
                         f"{prev.get('t0_variant5')} != {v.t0}")
    allowed = {(s.cell, s.strategy) for s in V5CELLS}
    bad = {(f["cell"], f["strategy"]) for f in vfills} - allowed
    if bad:
        raise ValueError(f"변형5 원장에 비변형5 행 기록 금지: {sorted(bad)}")
    n = append_csv_atomic(VLEDGER, pd.DataFrame(vfills, columns=LEDGER_COLS),
                          LEDGER_KEY)
    if n:
        logger.info("변형5 원장 %d행 추가 (중복 제거 후)", n)
    row = _vhist_row(state, v, V5CELLS, ts_end, bars_done, len(vfills))
    append_csv_atomic(VHIST, pd.DataFrame([row], columns=VHIST_COLS),
                      key=["ts"], keep="last")
    try:
        state.variant5_cells = variant5_to_dict(v)
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    except BaseException:
        state.variant5_cells = prev     # 실패 격리 롤백 (_save_variant 와 대칭)
        raise
    logger.info("변형5 자본 %.2f (E24+E25), 포지션 %d, 처리 봉 %d, 체결 %d",
                sum(row[s.cell.lower()] for s in V5CELLS),
                sum(len(c.positions) for c in v.cells.values()),
                bars_done, len(vfills))


def _run_variant5(ex, state: FarmState, data: dict, fund_ev: dict,
                  since: int, replay_end: int) -> None:
    """변형5 셀(E24·E25) 재생 — 변형4 단계 뒤 여섯째 단계 (_safe_variant 격리).

    첫 호출: 본 팜이 봉을 처리한 적이 있어야(last_ts>0) t0_variant5 동결
    (write-once) — 지표는 본 팜 스냅숏 상속(BBADD 는 본 팜이 추적하는
    cl·atr1 만 사용)이므로 별도 워밍업 수집이 없고, 본 팜 미가동 상태의 빈
    지표로 동결하지 않는다 (fail-closed, 다음 실행 재시도). 동결 실행은
    재생 없이 종료하고 다음 실행부터 재생한다. 이후: E11·E22 와 동일한
    폐지 미러 → 따라잡기 수집 → 재생 → 통합 파일 저장.

    Args:
        ex: ccxt bybit (따라잡기 수집 전용 — 정상 경로에서는 미사용).
        state: 본 팜 상태 (variant5_cells 키만 갱신됨).
        data: 본 실행이 수집한 sym -> {ts: ohlc} (읽기 전용으로 사용).
        fund_ev: 본 실행이 수집한 sym -> {정산 ts: 펀딩률 합}.
        since: 본 실행 재생 시작 ts.
        replay_end: 본 실행 재생 끝 ts (변형5도 같은 끝으로 정렬).
    """
    if state.variant5_cells is None:
        if not state.last_ts:
            return                      # 본 팜 미가동 — 다음 실행에서 초기화
        v = new_variant5(state, t0=int(time.time() * 1000))
        try:
            state.variant5_cells = variant5_to_dict(v)
            # 헤더 선생성 + 이력 스키마 이월 (e24·e25 열 정렬 — 변형2/3/4 전례)
            append_csv_atomic(VLEDGER, pd.DataFrame([], columns=LEDGER_COLS),
                              LEDGER_KEY)
            _migrate_vhist_schema()
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1,
                                            default=float))
        except BaseException:
            state.variant5_cells = None   # 미기록 t0 영속화 금지 (롤백)
            raise
        logger.info("변형5 t0_variant5=%d 동결 (write-once) — E24·E25: %s",
                    v.t0, V5LABELS["E24"])
        return
    v = variant5_from_dict(state.variant5_cells)
    if v.basket_b != list(state.basket_b):
        raise ValueError(f"변형5 바스켓 B 불일치: {v.basket_b} != {state.basket_b}")
    if v.last_ts > state.last_ts:
        raise ValueError(f"변형5 last_ts({v.last_ts})가 본 팜({state.last_ts}) 초과")
    vfills = _mirror_delist(state, v, variant5_delist)
    vsince, vdata, vfund = _variant_inputs(ex, v, data, fund_ev, since, replay_end)
    fills2, bars_done = _replay_variant(v, vdata, vfund, vsince, replay_end,
                                        step_variant5)
    _save_variant5(state, v, vfills + fills2, replay_end, bars_done)


def _finalize_variant5(state: FarmState) -> None:
    """변형5(E24·E25) 종말 폐지 미러 — _finalize_variant4 와 동일 계약, 그룹 분리."""
    if state.variant5_cells is None:
        return
    v = variant5_from_dict(state.variant5_cells)
    missing = [s for s in state.delisted if s not in v.delisted]
    if not missing:
        return
    vfills = _mirror_delist(state, v, variant5_delist)
    _save_variant5(state, v, vfills, v.last_ts or state.last_ts, 0)


def _save_variant6(state: FarmState, v: FarmState, vfills: list,
                   ts_end: int, bars_done: int) -> None:
    """변형6(E26·E27) 원장 → 이력 → 상태(variant6_cells) 순 원자적 체크포인트.

    _save_variant5 와 같은 계약 — 파일 통합(VLEDGER/VHIST), 상태 키·t0 만
    분리. 방화벽: (cell, strategy) 쌍이 V6CELLS 밖이면 기록 대신 예외.
    t0_variant6 는 write-once — 기존 기록과 다른 t0 저장 시도는 어떤 파일도
    쓰기 전에 예외로 죽는다. 이력 ts 는 변형6 의 15m 그리드 값 — 정시(:00)
    값은 다른 그룹의 1h ts 와 겹칠 수 있으나 keep-last(ts) 관례 그대로다
    (마지막 저장 그룹의 행이 남고 다른 그룹 열은 상태 스냅숏).
    """
    prev = state.variant6_cells
    if prev is not None and prev.get("t0_variant6") != v.t0:
        raise ValueError(f"t0_variant6 변경 금지 (write-once): "
                         f"{prev.get('t0_variant6')} != {v.t0}")
    allowed = {(s.cell, s.strategy) for s in V6CELLS}
    bad = {(f["cell"], f["strategy"]) for f in vfills} - allowed
    if bad:
        raise ValueError(f"변형6 원장에 비변형6 행 기록 금지: {sorted(bad)}")
    n = append_csv_atomic(VLEDGER, pd.DataFrame(vfills, columns=LEDGER_COLS),
                          LEDGER_KEY)
    if n:
        logger.info("변형6 원장 %d행 추가 (중복 제거 후)", n)
    row = _vhist_row(state, v, V6CELLS, ts_end, bars_done, len(vfills))
    append_csv_atomic(VHIST, pd.DataFrame([row], columns=VHIST_COLS),
                      key=["ts"], keep="last")
    try:
        state.variant6_cells = variant6_to_dict(v)
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
    except BaseException:
        state.variant6_cells = prev     # 실패 격리 롤백 (_save_variant 와 대칭)
        raise
    logger.info("변형6 자본 %.2f (E26+E27), 포지션 %d, 처리 봉 %d, 체결 %d",
                sum(row[s.cell.lower()] for s in V6CELLS),
                sum(len(c.positions) for c in v.cells.values()),
                bars_done, len(vfills))


def _variant6_inputs(ex, v: FarmState, fund_ev: dict, now15: int) -> tuple:
    """변형6 재생 입력 — 15m 봉 수집 + 갭 처분 + 펀딩 커버 (fail-closed).

    자체 15m 그리드라 본 실행 1h 수집분을 쓰지 않는다. 신선 갭(<48h)은 단계
    실패(다음 실행 따라잡기), 노화 갭은 결측 재생 허용 (check_gaps 공용 —
    엔진의 심볼별 무행동 fail-closed 가 처리). 심볼 하나라도 새 봉이 없으면
    v6_end < v6since 로 재생이 없다 (보수적 정렬 — 폐지는 본 팜 미러 담당).

    Returns:
        (v6since, v6_end, vdata, vfund) — v6_end < v6since 면 재생 없음.

    Raises:
        RuntimeError: 봉/펀딩 수집 실패 또는 신선 갭 (변형6 단계만 실패).
    """
    v6since = v.last_ts + M15
    syms = [s for s in BASKET_A if s not in v.delisted]
    vdata: dict = {}
    latest: dict = {}
    for s in syms:
        d = fetch_15m_paged(ex, s, v6since, now15)
        if d is None:
            raise RuntimeError(f"{s} 변형6 15m 수집 실패")
        vdata[s] = d
        latest[s] = max(d) if d else v.last_ts
    if not syms:
        return v6since, v.last_ts, {}, {}
    v6_end = min(latest.values())
    if v6_end < v6since:
        return v6since, v6_end, {}, {}
    nonempty = {s: d for s, d in vdata.items() if d}
    reason = check_gaps(nonempty, list(range(v6since, v6_end + 1, M15)),
                        max(latest.values()), continuing=True)
    if reason:
        raise RuntimeError(f"변형6 갭 — {reason}")
    need = max(v6since, v.t0)
    vfund: dict = {}
    for s in syms:
        ev = fund_ev.get(s)
        if ev is not None and not (len(ev) >= 200 and ev and min(ev) > need):
            vfund[s] = ev               # 본 실행 수집분이 변형6 구간을 덮는다
            continue
        rng = fetch_funding_range(ex, s, need)
        if rng is None:
            raise RuntimeError(f"{s} 변형6 펀딩 범위 수집 실패")
        vfund[s] = rng
    return v6since, v6_end, vdata, vfund


def _run_variant6(ex, state: FarmState, fund_ev: dict, now15: int) -> None:
    """변형6 셀(E26·E27) 재생 — 변형5 단계 뒤 일곱째 단계 (_safe_variant 격리).

    첫 호출: 본 팜 가동(last_ts>0) 후 BASKET_A 3종의 15m 봉 V6_WARMUP_15M
    (1680 = 420h)개를 **완전 깊이·무갭**으로 수집(fail-closed — 메이저는 상장
    이력이 확실해 짧은 응답을 '새 상장'으로 오인 금지)해 warmup_v6 로 지표를
    채우고, 성공 시에만 t0_variant6 동결 (write-once) 후 종료 — 다음 실행부터
    재생. 이후: 본 팜 폐지 미러 → 15m 수집·갭 처분·펀딩 커버(_variant6_inputs)
    → step_variant6 재생 → 통합 파일 저장. 워밍업 무주문 (t0 이전 봉은 지표
    전용 — 엔진 live 게이트가 구조적으로 차단).

    Args:
        ex: ccxt bybit (15m 수집·펀딩 범위 수집).
        state: 본 팜 상태 (variant6_cells 키만 갱신됨).
        fund_ev: 본 실행이 수집한 sym -> {정산 ts: 펀딩률 합} (커버 시 재사용).
        now15: 현재 시각의 15m 내림 정렬 ms — 형성중 봉 배제 경계
            (마지막 닫힌 15m 봉 = now15 − M15).
    """
    if state.variant6_cells is None:
        if not state.last_ts:
            return                      # 본 팜 미가동 — 다음 실행에서 초기화
        warm_end = now15 - M15
        w_since = now15 - V6_WARMUP_15M * M15
        v = new_variant6(state, t0=int(time.time() * 1000))
        for s in [x for x in BASKET_A if x not in state.delisted]:
            rows = fetch_15m_paged(ex, s, w_since, now15)
            if not rows or max(rows) != warm_end:
                raise RuntimeError(
                    f"{s} 변형6 워밍업 수집 불완전 — t0 동결 지연 (fail-closed)")
            if len(rows) != V6_WARMUP_15M or min(rows) != w_since \
                    or any((t - w_since) % M15 for t in rows):
                raise RuntimeError(
                    f"{s} 변형6 워밍업 깊이/연속성 부족({len(rows)}/"
                    f"{V6_WARMUP_15M}) — t0 동결 지연 (fail-closed)")
            warmup_v6(v, s, [(t, *rows[t]) for t in sorted(rows)])
        v.last_ts = warm_end            # 워밍업 끝 = 멱등 재생 기준 (15m 그리드)
        try:
            state.variant6_cells = variant6_to_dict(v)
            # 헤더 선생성 + 이력 스키마 이월 (e26·e27 열 정렬 — 변형2~5 전례)
            append_csv_atomic(VLEDGER, pd.DataFrame([], columns=LEDGER_COLS),
                              LEDGER_KEY)
            _migrate_vhist_schema()
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1,
                                            default=float))
        except BaseException:
            state.variant6_cells = None   # 미기록 t0 영속화 금지 (롤백)
            raise
        logger.info("변형6 t0_variant6=%d 동결 (write-once) — E26·E27: %s",
                    v.t0, V6LABELS["E26"])
        return
    v = variant6_from_dict(state.variant6_cells)
    vfills = _mirror_delist(state, v, variant6_delist)
    v6since, v6_end, vdata, vfund = _variant6_inputs(ex, v, fund_ev, now15)
    fills2: list = []
    bars_done = 0
    for t in range(v6since, v6_end + 1, M15):
        bars = {s: BarE(t, *vdata[s][t]) for s in vdata if t in vdata[s]}
        if not bars:
            continue
        fmap = {s: vfund.get(s, {}).get(t + M15, 0.0) for s in bars}
        fills2 += step_variant6(v, bars, fmap)
        bars_done += 1
    _save_variant6(state, v, vfills + fills2, max(v6_end, v.last_ts), bars_done)


def _finalize_variant6(state: FarmState) -> None:
    """변형6(E26·E27) 종말 폐지 미러 — _finalize_variant5 와 동일 계약, 그룹 분리."""
    if state.variant6_cells is None:
        return
    v = variant6_from_dict(state.variant6_cells)
    missing = [s for s in state.delisted if s not in v.delisted]
    if not missing:
        return
    vfills = _mirror_delist(state, v, variant6_delist)
    _save_variant6(state, v, vfills, v.last_ts or state.last_ts, 0)


def _safe_variant(fn, *args) -> None:
    """변형 단계 격리 실행 — 어떤 실패도 본 셀 커밋·중단 마커에 영향 없음.

    실패는 로그 + GITHUB_STEP_SUMMARY 로만 남기고 (ERR_MARK 는 본 팜 전용),
    변형은 자체 last_ts 로 다음 실행에서 따라잡는다.
    """
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 — 변형 실패는 본 팜과 완전 분리 (요건)
        stage = fn.__name__     # 단계 식별 (E11·E12 vs E13~E18 오귀속 방지)
        logger.exception("변형 단계(%s) 처리 실패 — 본 셀 저장은 완료됨: %s",
                         stage, exc)
        summary = os.getenv("GITHUB_STEP_SUMMARY")
        if summary:
            try:
                with open(summary, "a", encoding="utf-8") as f:
                    f.write(f"\n**Track E 변형 단계({stage}) 실패**: {exc}\n")
            except OSError as e2:
                logger.warning("GITHUB_STEP_SUMMARY 기록 실패: %s", e2)


def main() -> None:
    """매시 1회: T0 동결 → 새 닫힌 봉 재생 → 원자적 저장. 전 단계 fail-closed."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Track E 단타 팜 — PAPER ONLY (실주문·실자금 0, 승급 근거 사용 금지)")
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    if _retry(ex.load_markets) is None:
        _abort("마켓 로드 실패")
        return

    if STATE.exists():
        state = FarmState.from_dict(json.loads(STATE.read_text()))
    else:
        r = _retry(ex.publicGetV5MarketTickers, {"category": "linear"})
        if not r or str(r.get("retCode", "")) != "0":
            _abort("티커 조회 실패 — T0 초기화 불가")
            return
        bb = pick_basket_b(r.get("result", {}).get("list", []))
        if len(bb) < 3:
            _abort(f"바스켓 B 후보 부족({bb})")
            return
        state = new_farm(bb, t0=int(time.time() * 1000))
        _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
        logger.info("T0=%d 동결 — 바스켓 B=%s (이후 교체 없음)", state.t0, bb)

    syms = [s for s in list(BASKET_A) + list(state.basket_b)
            if s not in state.delisted]

    # 폐지 감지: 마켓 자체가 사라졌으면 마지막 유효가 청산 + 영구 공석
    fills: list = []
    for s in list(syms):
        m = ex.markets.get(f"{s}/USDT:USDT")
        if m is None or m.get("active") is False:
            logger.warning("%s 마켓 소멸/비활성 — 폐지 처리", s)
            fills += mark_delisted(state, s)
            syms.remove(s)
    if not syms:
        _abort("잔여 심볼 없음 — 폐지 변이 보존")
        if state.last_ts:
            _save_all(state, fills, state.last_ts, 0)
        else:
            _atomic_write(STATE, json.dumps(state.to_dict(), indent=1, default=float))
        _safe_variant(_finalize_variant, state)    # 변형 폐지 미러 (미실현 방치 금지)
        _safe_variant(_finalize_variant2, state)
        _safe_variant(_finalize_variant4, state)
        _safe_variant(_finalize_variant5, state)
        _safe_variant(_finalize_variant6, state)
        return

    now_h = int(pd.Timestamp.now(tz="utc").floor("h").timestamp() * 1000)
    since = state.last_ts + H1 if state.last_ts else now_h - WARMUP_1H * H1

    data = {}
    latest = {}
    for s in syms:
        d = fetch_1h_paged(ex, s, since, now_h)
        if d is None:
            _abort(f"{s} 1h 수집 실패")
            return
        if not d and not state.last_ts:
            _abort(f"{s} 워밍업 수집 공백")
            return
        data[s] = d
        latest[s] = max(d) if d else state.last_ts

    # 데이터 단절 정책 — 선두 절단 검사보다 먼저 (순서 계약: apply_stall_policy 참조)
    fills += apply_stall_policy(state, syms, data, latest, since)

    # 선두 절단 검사 — 워밍업 실행 전용. 계속 실행의 선두 결손은 아래 갭 검사가
    # 노화 판정(check_gaps)과 함께 처리한다 (감사 #2 — 영구 갭 동결 방지).
    # 상장 이력이 확실한 바스켓 A는 워밍업도 처음부터 이어져야 한다
    # (짧은 응답을 '새 상장'으로 오인한 절단 금지).
    for s in syms:
        d = data[s]
        if d and min(d) > since and not state.last_ts:
            if s in BASKET_A:
                _abort(f"{s} 선두 봉 결측 ({min(d)} > {since}) — 워밍업")
                return
            logger.warning("%s 워밍업이 %d부터 시작 (신규 상장 추정 — 지표 늦게 형성)",
                           s, min(d))

    # 단절 유예(48h 미만) 구간의 빈 심볼은 fail-closed 중단 (다음 시간 재시도)
    for s in syms:
        if not data[s]:
            _abort(f"{s} 새 봉 없음 (단절 유예 중)")
            _save_all(state, fills, state.last_ts, 0)      # 폐지 변이 보존
            _safe_variant(_finalize_variant, state)
            _safe_variant(_finalize_variant2, state)
            _safe_variant(_finalize_variant4, state)
            _safe_variant(_finalize_variant5, state)
            _safe_variant(_finalize_variant6, state)
            return
    if not syms:
        _abort("잔여 심볼 없음 — 상태 보존")
        _save_all(state, fills, state.last_ts, 0)
        _safe_variant(_finalize_variant, state)
        _safe_variant(_finalize_variant2, state)
        _safe_variant(_finalize_variant4, state)
        _safe_variant(_finalize_variant5, state)
        _safe_variant(_finalize_variant6, state)
        return

    replay_end = min(max(d) for d in data.values())
    if replay_end < since:
        logger.info("새 봉 없음")
        _save_all(state, fills, state.last_ts or now_h - H1, 0)   # 폐지·T0 변이 보존
        _clear_abort()
        _safe_variant(_finalize_variant, state)    # 변형 따라잡기는 다음 새 봉 실행에서
        _safe_variant(_finalize_variant2, state)
        _safe_variant(_finalize_variant4, state)
        _safe_variant(_finalize_variant5, state)
        _safe_variant(_finalize_variant6, state)
        return
    grid = list(range(since, replay_end + 1, H1))

    # 갭 검사 — 신선 갭은 통째 중단(다음 실행 재시도), 노화 갭은 결측 재생 진행
    reason = check_gaps(data, grid, max(latest.values()), bool(state.last_ts))
    if reason:
        _abort(reason)
        if state.last_ts:
            _save_all(state, fills, state.last_ts, 0)      # 폐지 변이 보존
            _safe_variant(_finalize_variant, state)
            _safe_variant(_finalize_variant2, state)
            _safe_variant(_finalize_variant4, state)
            _safe_variant(_finalize_variant5, state)
            _safe_variant(_finalize_variant6, state)
        return

    # 펀딩 — 라이브 구간이 있을 때만 필요 (워밍업 봉엔 포지션이 없다)
    fund_ev: dict = {}
    if state.t0 and grid[-1] >= state.t0:
        need_from = max(grid[0], state.t0)
        for s in syms:
            ev = fetch_funding(ex, s)
            if ev is None:
                _abort(f"{s} 펀딩 조회 실패")
                return
            if len(ev) >= 200 and ev and min(ev) > need_from:
                _abort(f"{s} 펀딩 이력이 재생 구간을 못 덮음")
                return
            fund_ev[s] = ev

    bars_done = 0
    for t in grid:
        bars = {s: BarE(t, *data[s][t]) for s in syms if t in data[s]}
        if not bars:
            continue
        fmap = {s: fund_ev.get(s, {}).get(t + H1, 0.0) for s in bars}
        for f in step(state, bars, fmap):
            fills.append(f)
            logger.info("  %s %s %s %s @ %.6f pnl %+.4f",
                        pd.Timestamp(t, unit="ms", tz="utc"), f["cell"], f["sym"],
                        f["action"], f["price"], f["pnl"])
        bars_done += 1

    _save_all(state, fills, replay_end, bars_done)
    _clear_abort()          # 본 팜 성공 커밋 확정 — 변형 실패는 마커에 영향 없음
    # 변형 셀 E11·E12 — 본 셀 저장 뒤에만, 실패해도 본 커밋 불변 (분리 요건)
    _safe_variant(_run_variant, ex, state, data, fund_ev, since, replay_end)
    # 변형2 셀 E13~E18 — E11·E12 뒤 셋째 단계, 각 단계 실패 상호 무영향 (분리)
    _safe_variant(_run_variant2, ex, state, data, fund_ev, since, replay_end)
    # 변형3 셀 E19~E21 — 넷째 단계 (전 유니버스, 자체 수집·자체 폐지 판정)
    _safe_variant(_run_variant3, ex, state, data, fund_ev, since, replay_end)
    # 변형4 셀 E22·E23 — 다섯째 단계 (손익비 1.5:1 익절, 본 팜 지표 스냅숏 상속)
    _safe_variant(_run_variant4, ex, state, data, fund_ev, since, replay_end)
    # 변형5 셀 E24·E25 — 여섯째 단계 (볼린저 추매 BBADD, 본 팜 지표 스냅숏 상속)
    _safe_variant(_run_variant5, ex, state, data, fund_ev, since, replay_end)
    # 변형6 셀 E26·E27 — 일곱째 단계 (AVGDOWN 전향 검증 AD, 자체 15m 그리드)
    now15 = int(pd.Timestamp.now(tz="utc").floor("15min").timestamp() * 1000)
    _safe_variant(_run_variant6, ex, state, fund_ev, now15)


if __name__ == "__main__":
    main()
