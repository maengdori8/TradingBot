from __future__ import annotations

"""H2 트랙 B — 코호트 전 지갑 userFills 수집기 (fills 프로토콜).

사전등록: docs/PREREGISTRATION_H2_2026-08-27.md (H2_SPEC v1) 프로토콜 전문:
- userFills 일 1회 폴링. 원본은 append-only 일별 gzip
  (logs/h2_fills/<YYYY-MM-DD>/fills.jsonl.gz), fill ID(tid)로 중복 제거.
- 응답별 oldest/newest fill timestamp 를 요약(summary.jsonl.gz)에 기록.
- 연속성: 이번 응답이 직전 폴링의 newest ts 를 덮어야(겹침) 인정.
- userFills 응답은 최대 2,000건(FILLS_RESP_CAP, 실측). 만석 AND 겹침 실패 시
  userFillsByTime(startTime=커서) 순방향 페이지네이션으로 커서까지의 갭을 소급
  복구한다. 명세의 "10k cap" 은 userFillsByTime 의 가용 윈도(FILLS_WINDOW =
  최근 10,000건까지만 소급 조회 가능)이며, 소급 누계가 윈도에 닿아도 갭을 못
  메울 때만 `fill-history-censored` 확정
  (영구 — 이후 원본 저장 생략, 집계만: 체결수·명목·maker비중·ts범위).
- 겹침 실패 but 만석 미달 → 불완전(incomplete) 표시 (기술 실패는 판정불가, 기각 아님).
- 일시 실패(응답 없음)는 상태를 갱신하지 않음 → 다음 응답이 이전 cursor 를 덮으면 복구.
- 고회전율(t0_month_vlm/t0_account 상위 3분위) 지갑 목록을 상태 파일에 고정,
  --intraday 모드는 그 지갑만 폴링 (6시간 간격 실행용, 절단 완화).
- --positions-snapshot: T0 clearinghouseState 포지션 스냅 1회 (left-censoring 기준선).
- 상태: logs/h2_fills_state.json. 사후 폴링빈도·적격기준 완화 금지.

내구성: 200지갑(CHUNK) 단위로 gzip 멤버를 닫고 상태를 저장 — 크래시 재개 시
마지막 미저장 청크만 재폴링돼 소량 중복 가능. 원본은 tid, 요약은
(address, mode, prev_newest_ts) 별 마지막 행만 취해 분석 단계에서 최종 제거.
(잔여 코너: 절단 지갑이 크래시 재개 사이 cap 윈도 이동을 겪으면 집계 일부가
keep-last 에서 유실될 수 있으나, 절단 지갑은 기술통계 전용이라 수용.)
크래시로 잘린 마지막 gzip 멤버(fills/summary)는 append 재개 전에
recover_truncated 로 유효 행만 원자적 재작성 — 이후 append 멤버가 표준 gzip
리더에 도달 가능 (portfolio/positions 경로와 동일한 recover-and-rewrite).
분석 읽기는 read_fills_dedup(전역 tid keep-first)이 원본 중복을 최종 흡수한다
(명세 §3.2 "분석 단계 유일성 보장").

T0 초기화 (명세 §2.3): --t0-init 원샷이 ① portfolio t0 스냅샷 →
② clearinghouseState 포지션 기준선 → ③ 전 코호트 fills 첫 폴링(커서 기준선)
을 단일 절차로 수행한다 (h2-collectors.yml mode=t0 와 로컬 수동 실행 공용).
fail-closed: 각 단계가 코호트 전체를 완주(n_done == n_cohort)해야 다음 단계로
진행하며, 완주 시 상태에 t0_initialized_at 를 기록한다 — 그 전에는 일별/
intraday 크론이 폴링을 시작하지 않는다 (§2.3 순서 강제).
지갑의 최초 폴링 응답이 FILLS_RESP_CAP(2,000건) 만석이면 상태에
initial_window_truncated=true 를 기록·유지하고 요약(집계)에 노출한다
(T0 이전 이력 미관측 — first 처리는 유지하되 초기 절단을 명시).
빈 최초 응답은 empty_first_at_ms(요청 직전 시각) 기준선만 남긴다(이력 전무
= 완전 관측, 플래그 없음) — 이후 만석 응답은 초기 절단이 아니라 post-T0
갭이므로 그 기준선에서 userFillsByTime 소급 복구/절단으로 처리한다.
"""

import argparse
import calendar
import gzip
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from carrybot.live.portfolio_snapshot import (
    PAUSE_S,
    collect,
    gzip_intact,
    load_cohort_wallets,
    load_done,
    post_info,
    read_rows_tolerant,
    recover_truncated,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

STATE_F = Path("logs/h2_fills_state.json")
FILLS_DIR = Path("logs/h2_fills")
POSITIONS_F = Path("logs/h2_positions_t0.jsonl.gz")
FILLS_RESP_CAP = 2000     # userFills/userFillsByTime 응답당 최대 체결 수 (실측)
FILLS_WINDOW = 10_000     # userFillsByTime 가용 윈도 — 최근 1만 건까지만 소급 조회 가능
CHUNK = 200               # 이 단위로 gzip 멤버 확정 + 상태 저장 (크래시 내구성)
STATUS_OK = "ok"
STATUS_CENSORED = "fill-history-censored"


def _fill_ts(f: dict) -> int | None:
    """체결 dict 에서 epoch ms 타임스탬프를 안전하게 꺼낸다."""
    try:
        return int(f["time"])
    except (KeyError, TypeError, ValueError):
        return None


def _iso_to_ms(iso: str) -> int | None:
    """ISO UTC 문자열(%Y-%m-%dT%H:%M:%SZ)을 epoch ms 로 바꾼다 (실패 시 None)."""
    try:
        return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")) * 1000
    except ValueError:
        return None


def judge_continuity(prev_newest: int | None, resp_oldest: int | None,
                     resp_newest: int | None, n_fills: int,
                     cap: int = FILLS_RESP_CAP) -> str:
    """연속성 판정 (명세: 이번 응답이 직전 폴링의 newest ts 를 덮어야 인정).

    - "first": 직전 폴링 없음 (기준선 수립).
    - "empty": 직전 폴링이 있는데 빈 응답 — 겹침 미증명 (당일 폴링 완료로 치지 않음).
    - "stale": 응답 전체가 직전 newest 이전 — 최신 체결 부재를 증명 못함 (동일 취급).
    - "ok": 응답이 직전 newest 를 덮음 (oldest ≤ 직전 newest ≤ newest).
    - "gap-censored": 겹침 실패 AND 응답 만석(cap=응답당 2,000건 실측 상한).
      poll_wallets 는 이 판정을 가로채 userFillsByTime 소급 페이지네이션으로 복구를
      먼저 시도하고, 가용 윈도(FILLS_WINDOW) 소진 시에만 절단을 확정한다.
      (process_response 를 직접 쓰는 경로에서는 즉시 절단으로 동작.)
    - "gap-incomplete": 겹침 실패, 만석 미달 → 불완전 처리."""
    if prev_newest is None:
        return "first"
    if n_fills == 0 or resp_oldest is None or resp_newest is None:
        return "empty"
    if resp_oldest <= prev_newest:
        return "ok" if resp_newest >= prev_newest else "stale"
    return "gap-censored" if n_fills >= cap else "gap-incomplete"


def dedup_new_fills(fills: list[dict], prev_newest: int | None,
                    boundary_tids: set) -> list[dict]:
    """직전 폴링 이후 신규 체결만 남긴다 (tid 중복 제거).

    경계(ts == prev_newest)의 체결은 직전 폴링에서 이미 저장된 tid 를 제외.
    응답 내부 tid 중복도 제거한다."""
    out: list[dict] = []
    seen: set = set()
    for f in fills:
        t = _fill_ts(f)
        if t is None:
            continue
        tid = f.get("tid")
        if tid is not None and tid in seen:
            continue
        if prev_newest is not None and (
                t < prev_newest or (t == prev_newest and tid in boundary_tids)):
            continue
        out.append(f)
        if tid is not None:
            seen.add(tid)
    return out


def aggregate_fills(fills: list[dict]) -> dict:
    """체결 리스트 집계 — 체결수·명목(USD)·maker 비중·ts 범위."""
    if not fills:
        return dict(n=0, notional=0.0, maker_frac=None, ts_min=None, ts_max=None)
    notional, makers, ts = 0.0, 0, []
    for f in fills:
        try:
            notional += abs(float(f["px"]) * float(f["sz"]))
        except (KeyError, TypeError, ValueError):
            pass
        if f.get("crossed") is False:      # crossed=True → taker
            makers += 1
        t = _fill_ts(f)
        if t is not None:
            ts.append(t)
    return dict(n=len(fills), notional=round(notional, 8),
                maker_frac=round(makers / len(fills), 6),
                ts_min=min(ts) if ts else None, ts_max=max(ts) if ts else None)


def process_response(addr: str, fills: list[dict], wst: dict, polled_at: str,
                     mode: str, cap: int = FILLS_RESP_CAP,
                     cont_override: str | None = None) -> tuple[dict, dict | None, dict]:
    """단일 지갑의 userFills 응답을 처리한다 (순수 — I/O 없음).

    반환: (갱신된 지갑 상태, 원본 기록 dict | None, 요약 기록 dict).
    - cont_override: poll_wallets 가 userFillsByTime 소급 복구에 성공해 갭 없음을
      증명한 경우 "ok" 를 주입한다 (fills 는 소급분+원본 병합본). None 이면
      judge_continuity 로 판정.
    - 절단(censored) 지갑은 원본을 생략하고 집계만 요약에 남긴다 (영구).
    - 커서(newest_ts) 전진 정책:
      * first/ok — 전진 (겹침 확인됨). ok 응답이 갭 시작점(gap_until_ts)까지
        도달하면 미복구 갭 플래그(incomplete)도 해제 (응답 내부는 연속이므로
        oldest~newest 구간이 메워짐).
      * gap-censored — 전진 (이후는 집계 전용이라 일별 이중집계 방지가 우선).
        이미 절단된 지갑은 gap-incomplete 라도 전진 (원본이 없어 복구 무의미,
        커서 동결 시 같은 체결이 일별 집계에 반복 산입됨).
      * gap-incomplete(비절단) — 전진하지 않음: 이후 덮는 응답이 구멍(직전 커서~
        응답 oldest)을 복구할 수 있게 커서를 보존한다. 그 사이 저장분은 원본에
        중복될 수 있으나 tid 로 분석 단계에서 최종 제거된다.
      * empty/stale — 전진할 것이 없음 (당일 폴링 완료로 치지 않음 — poll_wallets).
    - 요약의 prev_newest_ts: 이 폴링이 사용한 커서 — 크래시 재개로 요약이 중복되면
      (address, mode, prev_newest_ts) 별 마지막 행만 취해 이중집계를 제거한다.
    - 최초 폴링(first)이 cap 만석이면 initial_window_truncated=true 를 상태에
      기록한다 (이후 영구 유지, 요약에 노출 — 명세 §2.3: T0 이전 이력 미관측
      절단 명시. first 처리 자체는 유지).
    - 빈 최초 폴링(first, 체결 0건)은 empty_first_at_ms 기준선만 기록한다
      (polled_at = 요청 직전 시각, setdefault 로 최초값 보존): 이력 전무 =
      완전 관측이므로 플래그 없음. 이후 만석 응답은 poll_wallets 가 이 기준선
      에서 소급 복구를 시도하고, 실패 시 override="gap-censored" 를 주입한다
      (post-T0 유실은 초기 절단이 아니라 절단[fill-history-censored])."""
    ts_list = [t for t in (_fill_ts(f) for f in fills) if t is not None]
    resp_oldest = min(ts_list) if ts_list else None
    resp_newest = max(ts_list) if ts_list else None
    prev_newest = wst.get("newest_ts")
    cont = cont_override if cont_override is not None else judge_continuity(
        prev_newest, resp_oldest, resp_newest, len(fills), cap)

    if cont == "first" and len(fills) >= cap:
        # 최초 폴링 응답 만석 = T0 이전 이력 미관측 (초기 윈도 절단 명시 — §2.3)
        wst["initial_window_truncated"] = True
    if cont == "first" and not ts_list:
        # 빈 최초 응답 — 체결 이력 전무(완전 관측). 기준선 시각만 남겨 이후
        # 만석 응답을 post-T0 갭으로 소급 복구 가능하게 한다 (최초값 보존).
        base_ms = _iso_to_ms(polled_at)
        if base_ms is not None:
            wst.setdefault("empty_first_at_ms", base_ms)

    status = wst.get("status", STATUS_OK)
    if cont == "gap-censored" and status != STATUS_CENSORED:
        status = STATUS_CENSORED
        wst["censored_at"] = polled_at
    if cont == "gap-incomplete":
        wst["incomplete"] = True
        g = wst.get("gap_until_ts")
        wst["gap_until_ts"] = resp_oldest if g is None else min(g, resp_oldest)
    elif cont == "ok" and wst.get("incomplete"):
        # 복구 조건: 덮는 응답이 갭 시작점(불완전 응답의 최소 oldest)까지 도달해야 함
        if resp_newest >= wst.get("gap_until_ts", resp_newest):
            wst["incomplete"] = False
            wst.pop("gap_until_ts", None)

    new = dedup_new_fills(fills, prev_newest, set(wst.get("boundary_tids", [])))
    agg = aggregate_fills(new)

    raw = None
    if status != STATUS_CENSORED and new:
        raw = dict(address=addr, polled_at_utc=polled_at, mode=mode, fills=new)

    summary = dict(address=addr, polled_at_utc=polled_at, mode=mode,
                   continuity=cont, status=status, resp_n=len(fills),
                   prev_newest_ts=prev_newest,
                   resp_oldest_ts=resp_oldest, resp_newest_ts=resp_newest,
                   new_n=agg["n"], new_notional=agg["notional"],
                   new_maker_frac=agg["maker_frac"],
                   new_ts_min=agg["ts_min"], new_ts_max=agg["ts_max"])
    if wst.get("initial_window_truncated"):
        summary["initial_window_truncated"] = True      # 집계 노출 (영구 유지)

    advance = cont in ("first", "ok", "gap-censored") or (
        status == STATUS_CENSORED and cont == "gap-incomplete")
    if advance and resp_newest is not None and (
            prev_newest is None or resp_newest >= prev_newest):
        boundary = {f.get("tid") for f in fills
                    if _fill_ts(f) == resp_newest and f.get("tid") is not None}
        if prev_newest == resp_newest:      # 커서 정지 시 기존 경계와 합집합
            boundary |= set(wst.get("boundary_tids", []))
        wst["newest_ts"] = resp_newest
        prev_oldest = wst.get("oldest_ts")
        wst["oldest_ts"] = resp_oldest if prev_oldest is None else min(prev_oldest, resp_oldest)
        wst["boundary_tids"] = sorted(boundary, key=str)
    wst["status"] = status
    return wst, raw, summary


def pick_high_turnover(wallets: list[dict]) -> list[str]:
    """T0 회전율(t0_month_vlm / t0_account) 상위 3분위 지갑 주소를 고른다.

    회전율 내림차순, 동률은 주소 오름차순 (결정적). k = n // 3 (최소 1)."""
    scored: list[tuple[float, str]] = []
    for w in wallets:
        try:
            turn = float(w["t0_month_vlm"]) / float(w["t0_account"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        scored.append((turn, w["address"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    k = max(1, len(scored) // 3)
    return [a for _, a in scored[:k]]


def daily_todo(addrs: list[str], state: dict, day: str) -> list[str]:
    """오늘(UTC) 아직 일별 폴링하지 않은 지갑만 남긴다 (멱등 재실행)."""
    ws = state.get("wallets", {})
    return [a for a in addrs if ws.get(a, {}).get("last_daily") != day]


def load_state(path: Path = STATE_F) -> dict:
    """상태 파일 로드 (없으면 빈 골격)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(high_turnover=[], wallets={})


def save_state(state: dict, path: Path = STATE_F) -> None:
    """상태 파일을 임시파일 경유로 원자적 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def fetch_user_fills(addr: str) -> object | None:
    """지갑 1개의 userFills 응답 (최신 최대 FILLS_RESP_CAP 건, 실패 시 None)."""
    return post_info({"type": "userFills", "user": addr})


def fetch_user_fills_by_time(addr: str, start_time: int) -> object | None:
    """userFillsByTime 순방향 페이지 1개 (startTime 이후 최대 FILLS_RESP_CAP 건).

    startTime 은 epoch ms 인클루시브로 가정. 실패 시 None (post_info 내장 재시도 후)."""
    return post_info({"type": "userFillsByTime", "user": addr,
                      "startTime": int(start_time)})


def backfill_gap(addr: str, prev_newest: int, gap_target: int | None,
                 fetch_by_time: Callable[[str, int], object | None],
                 pause_s: float = 0.0, resp_cap: int = FILLS_RESP_CAP,
                 window: int = FILLS_WINDOW) -> tuple[str, list[dict]]:
    """userFills 만석+겹침 실패 시 userFillsByTime 으로 커서까지의 갭을 소급 채운다.

    순방향 페이지네이션: startTime = prev_newest(커서, 인클루시브 — 경계 체결 재수신은
    tid 로 제거)부터 resp_cap 건 페이지 단위, 각 페이지의 마지막 fill time 을 다음
    startTime 으로 전진. 종료 조건:
    - 페이지가 만석 미달(데이터 끝 = 현재까지 도달) 또는 gap_target(메인 userFills
      응답의 oldest ts) 도달 → 갭이 메워짐.
    - 고유 체결 누계가 가용 윈도(window, 최근 1만 건)에 닿았는데 갭을 못 메움 →
      커서 직후 체결이 윈도 밖으로 밀려나 영구 소실됐을 수 있음 = 연속성 증명 불가
      → "censored" (명세: 윈도 소진 시에만 절단 확정).
    - 페이지 실패(응답 없음·스키마 이상·안전 한도 초과) → "failed" — 상태 미변경,
      다음 폴링에서 재시도 (기술 실패는 절단이 아님).
    한 페이지가 동일 ms 만석이면 +1ms 로 최소 전진한다 (같은 ms 의 resp_cap 초과분은
    API 특성상 소실 가능 — 경고 로깅).

    반환: (판정, 수집 체결) — 판정 ∈ {"recovered", "censored", "failed"}."""
    fills: list[dict] = []
    seen: set = set()
    start = int(prev_newest)
    bridged = False
    max_pages = window // resp_cap + 2          # 안전 한도 (무진행 루프 방지)
    for _ in range(max_pages):
        page = fetch_by_time(addr, start)
        if not valid_fills_resp(page):
            logger.warning("[backfill] %s 페이지 실패(start=%d) — 실패 처리", addr, start)
            return "failed", []
        page_ts = [t for t in (_fill_ts(f) for f in page) if t is not None]
        page_newest = max(page_ts) if page_ts else None
        for f in page:
            tid = f.get("tid")
            if tid is None or tid in seen:
                continue
            seen.add(tid)
            fills.append(f)
        if len(page) < resp_cap or (
                gap_target is not None and page_newest is not None
                and page_newest >= gap_target):
            bridged = True
            break
        if len(seen) >= window:
            break                               # 윈도 소진 — 갭 미복구
        if page_newest is None or page_newest <= start:
            logger.warning("[backfill] %s 동일 ms 만석(start=%d) — +1ms 강제 전진",
                           addr, start)
            start += 1
        else:
            start = page_newest
        if pause_s:
            time.sleep(pause_s)
    else:
        logger.warning("[backfill] %s 안전 한도(%d페이지) 초과 — 실패 처리",
                       addr, max_pages)
        return "failed", []
    if bridged and len(seen) < window:
        return "recovered", fills
    return "censored", fills


def valid_fills_resp(resp: object) -> bool:
    """userFills 응답이 알려진 스키마인지 검사한다.

    비-리스트, 또는 원소의 10% 초과가 time/tid 없는 형태면 스키마 변경으로 간주 →
    실패 처리 (상태 미변경). 전 지갑에서 동시 발생하면 수집이 사실상 정지되고
    원본은 오염 없이 보존된다 (명세의 스키마 변경 중단 규칙). 10% 허용은 드문
    개별 손상 레코드로 지갑-일 전체를 잃지 않기 위한 여유다."""
    if not isinstance(resp, list):
        return False
    if not resp:
        return True
    bad = sum(1 for f in resp
              if not isinstance(f, dict) or _fill_ts(f) is None or f.get("tid") is None)
    return bad * 10 <= len(resp)


def poll_wallets(addrs: list[str], state: dict, mode: str,
                 fetch: Callable[[str], object | None] = fetch_user_fills,
                 fetch_by_time: Callable[[str, int], object | None] = fetch_user_fills_by_time,
                 fills_dir: Path = FILLS_DIR, state_f: Path = STATE_F,
                 pause_s: float = PAUSE_S) -> dict:
    """지갑 목록을 폴링해 원본/요약을 append 하고 상태를 갱신한다.

    - 응답 없음(None)·형식 이상(리스트 아님, 비-dict 원소)은 실패 처리 —
      상태 미변경 (다음 응답이 이전 커서를 덮으면 복구).
    - userFills 만석(FILLS_RESP_CAP)+겹침 실패(비절단 지갑)는 즉시 절단하지 않고
      backfill_gap(userFillsByTime 순방향 페이지네이션)으로 커서까지의 갭 복구를
      먼저 시도한다: recovered → 소급분을 병합해 "ok" 처리(원본 저장·커서 전진),
      censored(가용 윈도 1만 건 소진) → 그때만 절단 확정, failed → 지갑 실패
      처리(상태 미변경, 다음 폴링 재시도). 이미 절단된 지갑은 기존 동작 유지
      (원본 생략·집계만·커서 전진 — 복구 무의미).
      소급 커서는 newest_ts(체결 커서), 없으면 empty_first_at_ms(빈 최초 응답
      기준선) — 빈 기준선 이후의 만석은 초기 절단이 아니라 post-T0 갭이다.
    - 내구성: CHUNK 지갑마다 gzip 멤버를 닫아(확정) 상태를 저장한다.
      append 재개 전 두 출력 파일의 잘린 마지막 멤버(직전 kill)를
      recover_truncated 로 유효 행만 원자적 재작성해, 이후 append 멤버가
      표준 gzip 리더에 도달 가능함을 보장한다 (portfolio 경로와 동일 패턴).
      크래시 재개 시 마지막 미저장 청크만 재폴링돼 원본/요약이 중복될 수 있다 —
      원본은 tid(read_fills_dedup, 전역 keep-first), 요약은 (address, mode,
      prev_newest_ts) 별 마지막 행만 취해 분석 단계에서 최종 제거
      (재폴링은 같은 커서를 재사용하므로 키가 일치)."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    ddir = fills_dir / day
    ddir.mkdir(parents=True, exist_ok=True)
    raw_f = ddir / "fills.jsonl.gz"
    sum_f = ddir / "summary.jsonl.gz"
    for p in (raw_f, sum_f):            # 직전 kill 로 잘린 마지막 gzip 멤버 복구
        if not gzip_intact(p):          # 정상 파일은 행 실체화 없이 통과 (상수 메모리)
            recover_truncated(p)
    wallets_st = state.setdefault("wallets", {})
    failed: list[str] = []
    n_raw = n_censored = n_backfilled = 0
    t_start = time.time()

    for start in range(0, len(addrs), CHUNK):
        chunk = addrs[start:start + CHUNK]
        with gzip.open(raw_f, "at", encoding="utf-8") as fr, \
                gzip.open(sum_f, "at", encoding="utf-8") as fs:
            for addr in chunk:
                polled_at = utc_now_iso()   # 요청 직전 시각 — 빈 최초 응답 기준선
                resp = fetch(addr)
                ok = valid_fills_resp(resp)
                override: str | None = None
                back_n = 0
                if ok:
                    wst = wallets_st.setdefault(addr, {})
                    prev_newest = wst.get("newest_ts")
                    cursor = prev_newest if prev_newest is not None \
                        else wst.get("empty_first_at_ms")
                    ts_list = [t for t in (_fill_ts(f) for f in resp)
                               if t is not None]
                    resp_oldest = min(ts_list) if ts_list else None
                    # 만석+겹침 실패 (커서 = 체결 커서 또는 빈 최초 응답 기준선)
                    if (wst.get("status") != STATUS_CENSORED and cursor is not None
                            and resp_oldest is not None and resp_oldest > cursor
                            and len(resp) >= FILLS_RESP_CAP):
                        outcome, back = backfill_gap(
                            addr, cursor, resp_oldest, fetch_by_time,
                            pause_s=pause_s)
                        if outcome == "failed":
                            ok = False           # 상태 미변경 — 다음 폴링 재시도
                        elif outcome == "recovered":
                            resp = back + resp   # 소급분 병합 — 갭 없음 증명됨
                            override = "ok"
                            back_n = len(back)
                            n_backfilled += 1
                            logger.info("[%s] %s 갭 소급 복구 — %d건 병합",
                                        mode, addr, back_n)
                        else:                    # censored — 윈도 소진 시에만 확정
                            override = "gap-censored"   # 빈 기준선 케이스도 절단으로
                            logger.warning("[%s] %s 가용 윈도(%d) 소진 — 절단 확정",
                                           mode, addr, FILLS_WINDOW)
                if not ok:
                    failed.append(addr)
                else:
                    wst, raw, summary = process_response(
                        addr, resp, wst, polled_at, mode,
                        cont_override=override)
                    if back_n:
                        summary["backfill_n"] = back_n
                    if raw is not None:
                        fr.write(json.dumps(raw) + "\n")
                        n_raw += 1
                    fs.write(json.dumps(summary) + "\n")
                    if mode == "daily" and summary["continuity"] not in ("empty", "stale"):
                        wst["last_daily"] = day      # empty/stale 은 당일 재시도 허용
                    if wst["status"] == STATUS_CENSORED:
                        n_censored += 1
                    wallets_st[addr] = wst
                if pause_s:
                    time.sleep(pause_s)
        save_state(state, state_f)          # gzip 멤버 확정 후에만 커밋
        done_n = min(start + CHUNK, len(addrs))
        if done_n < len(addrs):
            rate = done_n / max(time.time() - t_start, 1e-9)
            logger.info("[%s] %d/%d (%.1f/s, ETA %.0f분)", mode, done_n,
                        len(addrs), rate, (len(addrs) - done_n) / max(rate, 0.1) / 60)
    if failed:
        logger.warning("[%s] 실패 %d지갑 (상태 미변경 — 다음 폴링이 덮으면 복구): %s",
                       mode, len(failed), ", ".join(failed))
    logger.info("[%s] 완료: 폴링 %d, 원본기록 %d, 소급복구 %d, 절단누적 %d, 실패 %d",
                mode, len(addrs) - len(failed), n_raw, n_backfilled,
                n_censored, len(failed))
    return dict(day=day, mode=mode, n_polled=len(addrs) - len(failed),
                n_raw=n_raw, n_backfilled=n_backfilled, n_censored=n_censored,
                n_failed=len(failed), failed=failed)


def read_fills_dedup(fills_dir: Path = FILLS_DIR) -> list[dict]:
    """분석용 fills 원본 로더 — 전역 tid 기준 keep-first 중복 제거 (명세 §3.2).

    일자 디렉토리 오름차순(YYYY-MM-DD 사전순 = 시간순) → 파일 내 기록 순으로
    전 원본을 순회한다. 크래시 복구 후 재폴링(같은 커서 재사용)으로 생긴 원본
    중복은 여기서 흡수돼 분석 단계 유일성이 보장된다 — 수집 단계 중복은 허용.
    잘린 gzip 멤버는 read_rows_tolerant 로 유효 행까지만 읽는다 (읽기 전용 —
    파일 재작성은 수집 경로의 recover_truncated 담당).
    tid 없는 체결은 유일성 판정이 불가하므로 분석 목록에서 제외하고 개수를
    경고 로깅한다 (원본에는 보존 — 데이터 품질 지표). 중복 키는 str(tid) 로
    정규화해 int/str 표기 흔들림이 keep-first 를 우회하지 못하게 한다.

    반환: fill dict 에 address/polled_at_utc/mode 를 부가한 평탄 목록
    (전역 tid 유일, 최초 기록분 유지)."""
    out: list[dict] = []
    seen: set[str] = set()
    n_no_tid = 0
    for raw_f in sorted(fills_dir.glob("*/fills.jsonl.gz")):
        rows, _ = read_rows_tolerant(raw_f)
        for row in rows:
            for f in row.get("fills", []):
                if not isinstance(f, dict):
                    continue
                tid = f.get("tid")
                if tid is None:
                    n_no_tid += 1        # 유일성 보장 불가 — 분석 목록에서 제외
                    continue
                key = str(tid)
                if key in seen:
                    continue
                seen.add(key)
                out.append(dict(f, address=row.get("address"),
                                polled_at_utc=row.get("polled_at_utc"),
                                mode=row.get("mode")))
    if n_no_tid:
        logger.warning("read_fills_dedup: tid 없는 체결 %d건 제외 (원본 보존 — "
                       "데이터 품질 지표)", n_no_tid)
    return out


def fetch_clearinghouse(addr: str) -> object | None:
    """지갑 1개의 clearinghouseState 응답 (실패 시 None)."""
    return post_info({"type": "clearinghouseState", "user": addr})


def snapshot_positions(cohort_path: Path | None = None, out: Path = POSITIONS_F,
                       fetch: Callable[[str], object | None] = fetch_clearinghouse,
                       pause_s: float = PAUSE_S) -> dict:
    """T0 clearinghouseState 포지션 스냅 1회 저장 (left-censoring 기준선).

    이어받기: 이미 기록된 지갑은 건너뛴다. 실패 지갑은 로깅 후 재실행 시 재시도.
    내구성: CHUNK 지갑마다 gzip 멤버를 닫아 확정 — 잘린 멤버는 이어받기의
    load_done(recover-and-rewrite)이 복구한다."""
    wallets = [w["address"] for w in (
        load_cohort_wallets(cohort_path) if cohort_path else load_cohort_wallets())]
    done = load_done(out)                # 잘린 멤버는 여기서 복구 후 이어쓰기
    todo = [w for w in wallets if w not in done]
    logger.info("[positions] 코호트 %d | 완료 %d | 남음 %d → %s",
                len(wallets), len(done), len(todo), out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    n_ok = 0
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        with gzip.open(out, "at", encoding="utf-8") as fo:
            for addr in chunk:
                resp = fetch(addr)
                if not isinstance(resp, dict) or "marginSummary" not in resp:
                    failed.append(addr)      # 에러 형태 dict 를 완료로 오인하지 않음
                else:
                    fo.write(json.dumps(dict(address=addr,
                                             captured_at_utc=utc_now_iso(),
                                             state=resp)) + "\n")
                    n_ok += 1
                if pause_s:
                    time.sleep(pause_s)
        done_n = min(start + CHUNK, len(todo))
        if done_n < len(todo):
            logger.info("[positions] %d/%d", done_n, len(todo))
    if failed:
        logger.warning("[positions] 실패 %d지갑 (재실행 시 재시도): %s",
                       len(failed), ", ".join(failed))
    logger.info("[positions] 완료: 신규 %d, 실패 %d, 누적 %d/%d",
                n_ok, len(failed), len(done) + n_ok, len(wallets))
    return dict(n_ok=n_ok, n_failed=len(failed), failed=failed,
                n_done=len(done) + n_ok, n_cohort=len(wallets))


def ensure_high_turnover(state: dict, wallets: list[dict]) -> None:
    """고회전율(상위 3분위) 지갑 목록이 없으면 선정해 상태 파일에 고정한다."""
    if not state.get("high_turnover"):
        state["high_turnover"] = pick_high_turnover(wallets)
        save_state(state)
        logger.info("고회전율 지갑 %d/%d개 선정(상위 3분위) — 상태 파일에 고정",
                    len(state["high_turnover"]), len(wallets))


def run_daily_poll() -> dict | None:
    """전 코호트 fills 일별 폴링 1회 (t0 첫 폴링과 일별 크론 공용 경로).

    커서 기준선: 지갑별 최초 폴링은 first 처리되며, 응답이 FILLS_RESP_CAP
    만석이면 process_response 가 initial_window_truncated 를 기록한다.
    같은 UTC 일자 재실행은 미폴링 지갑만 폴링 (멱등).
    반환: poll_wallets 결과 dict (오늘 폴링할 지갑이 없으면 None)."""
    state = load_state()
    wallets = load_cohort_wallets()
    ensure_high_turnover(state, wallets)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    addrs = daily_todo([w["address"] for w in wallets], state, day)
    if not addrs:
        logger.info("%s 일별 폴링 이미 완료 — 종료", day)
        return None
    return poll_wallets(addrs, state, "daily")


def verify_t0_fills_baseline(state: dict, wallets: list[dict]) -> list[str]:
    """코호트 전 지갑의 fills 커서 기준선 수립 여부를 검사한다.

    기준선 = newest_ts(체결 커서) 또는 empty_first_at_ms(빈 최초 응답 기준선).
    반환: 기준선이 없는 주소 목록 (빈 목록 = 완주)."""
    ws = state.get("wallets", {})
    return [w["address"] for w in wallets
            if ws.get(w["address"], {}).get("newest_ts") is None
            and ws.get(w["address"], {}).get("empty_first_at_ms") is None]


def t0_init() -> None:
    """T0 단일 초기화 원샷 (명세 §2.3 — yml mode=t0 와 로컬 수동 실행 공용).

    순서 고정: ① portfolio t0 스냅샷(collect "t0") → ② clearinghouseState
    포지션 기준선(snapshot_positions) → ③ 전 코호트 fills 첫 폴링
    (run_daily_poll — 커서 기준선 수립, 만석 지갑은 initial_window_truncated).

    fail-closed: 각 단계가 코호트 전체를 완주(n_done == n_cohort, fills 는
    전 지갑 기준선 수립)해야 다음 단계로 진행하며, 미완주면 exit 1. 각 단계는
    멱등이라 재디스패치가 실패분만 이어받는다 (portfolio 이어받기는 UTC 일자
    파일 단위이므로 T0 당일 내 완주 필수). 완주 시 t0_initialized_at 를 상태에
    기록 — 이 플래그 전에는 일별/intraday 크론이 폴링을 시작하지 않는다."""
    logger.info("[t0-init] ① portfolio t0 스냅샷")
    r1 = collect("t0")
    if r1["n_done"] != r1["n_cohort"]:
        logger.error("[t0-init] portfolio t0 미완주 (%d/%d) — 중단, "
                     "재디스패치로 이어받기", r1["n_done"], r1["n_cohort"])
        raise SystemExit(1)
    logger.info("[t0-init] ② clearinghouseState 포지션 기준선")
    r2 = snapshot_positions()
    if r2["n_done"] != r2["n_cohort"]:
        logger.error("[t0-init] 포지션 기준선 미완주 (%d/%d) — 중단, "
                     "재디스패치로 이어받기", r2["n_done"], r2["n_cohort"])
        raise SystemExit(1)
    logger.info("[t0-init] ③ 전 코호트 fills 첫 폴링 (커서 기준선)")
    run_daily_poll()                     # None = 오늘 이미 폴링 완료 (기준선 검사로 확인)
    state = load_state()
    missing = verify_t0_fills_baseline(state, load_cohort_wallets())
    if missing:
        logger.error("[t0-init] fills 기준선 미수립 %d지갑 — 중단, 재디스패치로 "
                     "이어받기: %s%s", len(missing), ", ".join(missing[:10]),
                     " ..." if len(missing) > 10 else "")
        raise SystemExit(1)
    if not state.get("t0_initialized_at"):
        state["t0_initialized_at"] = utc_now_iso()
        save_state(state)
    logger.info("[t0-init] 완료 — T0 기준선 수립 (t0_initialized_at=%s)",
                state["t0_initialized_at"])


def main(argv: list[str] | None = None) -> None:
    """CLI 진입점.

    기본: 전 코호트 일 1회 폴링 (같은 UTC 일자 재실행은 미폴링 지갑만).
    --intraday: 고회전율 상위 3분위만 폴링 (6시간 간격 실행용).
    --positions-snapshot: T0 clearinghouseState 스냅 1회.
    --t0-init: T0 단일 초기화 원샷 (portfolio t0 → 포지션 → fills 첫 폴링).

    크론 게이트: t0_initialized_at 이 상태에 기록되기 전에는 일별/intraday
    폴링을 시작하지 않는다 (§2.3 — T0 단일 절차보다 먼저 커서가 생기는 것을
    차단. t0-init 완주 후에만 자동 수집 개시)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="H2 fills 수집기")
    ap.add_argument("--intraday", action="store_true",
                    help="고회전율(상위 3분위) 지갑만 폴링")
    ap.add_argument("--positions-snapshot", action="store_true",
                    help="T0 clearinghouseState 포지션 스냅 1회")
    ap.add_argument("--t0-init", action="store_true",
                    help="T0 단일 초기화: portfolio t0 → 포지션 기준선 → fills 첫 폴링")
    args = ap.parse_args(argv)

    if args.positions_snapshot:
        snapshot_positions()
        return
    if args.t0_init:
        t0_init()
        return
    state = load_state()
    if not state.get("t0_initialized_at"):
        logger.warning("T0 미초기화 (t0_initialized_at 없음) — 폴링 대기 "
                       "(§2.3 순서: mode=t0 완주 후 자동 수집 개시)")
        return
    if args.intraday:
        ensure_high_turnover(state, load_cohort_wallets())
        poll_wallets(state["high_turnover"], state, "intraday")
        return
    run_daily_poll()


if __name__ == "__main__":
    main()
