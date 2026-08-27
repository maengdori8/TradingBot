from __future__ import annotations

"""H2 트랙 B — 코호트 전 지갑 perpAllTime 스냅샷 수집기 (T0 / 일별 / 판정일).

사전등록: docs/PREREGISTRATION_H2_2026-08-27.md (H2_SPEC v1).
- 일별 리더보드 alltime_pnl 사용 금지 (perpAllTime 계열과 중앙 $8.8k 괴리 실측)
  → portfolio API 직접 수집.
- 지갑별 최종점만 저장해 용량 절약: perp_alltime_pnl(pnl 최종점),
  account_value(acct 최종점), captured_at_utc. 단 곡선 마지막 3점은 진단용 보존.
- 출력: logs/h2_snapshots/<YYYY-MM-DD>.jsonl.gz (UTC 일자, 지갑당 1줄).
- 멱등: 같은 날 재실행 시 이미 수집된 지갑은 건너뛰고 이어받는다.
- 내구성: CHUNK(200지갑)마다 gzip 멤버를 닫아 확정 — kill 시 피해를 마지막
  청크로 국한. 잘린 멤버(수집 중 kill)는 이어받기 재실행의 load_done* 가
  유효 행만으로 원자적 재작성(recover-and-rewrite)한 뒤 append 를 재개하므로,
  이후 append 멤버가 표준 gzip 리더에 도달 불가해지는 문제가 없다.
- 레이트리밋·백오프: lab/collect_portfolio.py 패턴 재사용 (지갑당 0.18s,
  4회 재시도 선형 백오프 — 단 429는 지수 백오프).
"""

import argparse
import gzip
import json
import logging
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

API = "https://api.hyperliquid.xyz/info"
COHORT = Path("logs/trader_cohort.json.gz")
SNAP_DIR = Path("logs/h2_snapshots")
LABELS = ("t0", "daily", "verdict")
PAUSE_S = 0.18            # ~5.5 req/s — 공개 한도의 절반 이하 (collect_portfolio 동일)
TAIL_N = 3                # 진단용 곡선 꼬리 점수
CHUNK = 200               # 이 단위로 gzip 멤버를 닫아 확정 (크래시 내구성)


def post_info(body: dict, retries: int = 4, timeout: int = 30) -> object | None:
    """HL info API POST 호출 — 429는 지수, 그 외 실패는 선형 백오프 후 None."""
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            time.sleep(2.0 ** (i + 1) if e.code == 429 else 2.0 * (i + 1))
        except Exception:  # noqa: BLE001
            time.sleep(2.0 * (i + 1))
    return None


def fetch_portfolio(addr: str) -> object | None:
    """지갑 1개의 portfolio 곡선 응답을 가져온다 (실패 시 None)."""
    return post_info({"type": "portfolio", "user": addr})


def parse_row(addr: str, payload: object, label: str, captured_at: str) -> dict | None:
    """portfolio 응답에서 perpAllTime 최종점을 뽑아 저장 행을 만든다.

    저장 스키마: address, label, captured_at_utc, perp_alltime_pnl(pnl 최종점 float),
    account_value(acct 최종점 float), pnl_ts/acct_ts(최종점 epoch ms),
    pnl_tail/acct_tail(마지막 3점 원본 [ts, "값"] — 진단용).
    파싱 불가·곡선 결측이면 None (다음 실행에서 재시도)."""
    if not isinstance(payload, list):
        return None
    perp = None
    for item in payload:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "perpAllTime":
            perp = item[1]
            break
    if not isinstance(perp, dict):
        return None
    pnl = perp.get("pnlHistory") or []
    acct = perp.get("accountValueHistory") or []
    if not pnl or not acct:
        return None
    try:
        return dict(address=addr, label=label, captured_at_utc=captured_at,
                    perp_alltime_pnl=float(pnl[-1][1]),
                    account_value=float(acct[-1][1]),
                    pnl_ts=int(pnl[-1][0]), acct_ts=int(acct[-1][0]),
                    pnl_tail=pnl[-TAIL_N:], acct_tail=acct[-TAIL_N:])
    except (IndexError, TypeError, ValueError):
        return None


def load_cohort_wallets(cohort: Path = COHORT) -> list[dict]:
    """잠긴 코호트의 지갑 dict 목록을 로드한다."""
    with gzip.open(cohort, "rt", encoding="utf-8") as f:
        return json.load(f)["wallets"]


def read_rows_tolerant(path: Path) -> tuple[list[dict], bool]:
    """jsonl.gz 를 읽되 gzip 절단(잘린 멤버)을 견딘다.

    수집 중 kill 되면 마지막 gzip 멤버가 트레일러 없이 잘려 표준 리더가
    EOFError/BadGzipFile 로 죽는다 — 예외 전까지 읽힌 유효 행만 반환하고
    절단 여부를 함께 알린다. JSON 파싱 불가 행(절단 잔여물 포함)은 건너뛴다.

    반환: (유효 dict 행 목록, 절단 감지 여부)."""
    rows: list[dict] = []
    truncated = False
    if not path.exists():
        return rows, truncated
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (EOFError, gzip.BadGzipFile, zlib.error, OSError) as e:
        truncated = True
        logger.warning("gzip 절단 감지: %s (%s) — 유효 %d행까지만 사용",
                       path, e, len(rows))
    return rows, truncated


def gzip_intact(path: Path) -> bool:
    """gzip 파일이 끝까지 정상 해제되는지 스트리밍으로 검사한다 (상수 메모리).

    행을 메모리에 올리지 않으므로 대용량 일별 파일의 append 재개 전 사전
    검사에 적합하다 — 정상이면 recover_truncated 의 전체 행 실체화를 건너뛸
    수 있다. 파일이 없으면 True (복구할 것 없음)."""
    if not path.exists():
        return True
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except (EOFError, gzip.BadGzipFile, zlib.error, OSError):
        return False


def recover_truncated(path: Path) -> list[dict]:
    """절단 감지 시 유효 행만으로 원자적 재작성(recover-and-rewrite) 후 행을 반환.

    append 재개 전에 호출 — 잘린 멤버를 제거해 이후 'at' append 멤버가 표준
    gzip 리더에 도달 가능하도록 만든다. 절단이 없으면 파일을 건드리지 않는다."""
    rows, truncated = read_rows_tolerant(path)
    if truncated:
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        tmp.replace(path)
        logger.warning("gzip 절단 복구: %s — 유효 %d행으로 재작성 후 이어쓰기",
                       path, len(rows))
    return rows


def load_done(out: Path) -> set[str]:
    """기존 출력 파일에서 이미 수집된 주소 집합을 읽는다 (이어받기).

    잘린 gzip 멤버는 유효 행만으로 재작성한 뒤 읽는다 (recover_truncated)."""
    return {row["address"] for row in recover_truncated(out) if "address" in row}


def load_done_labeled(out: Path, label: str) -> set[str]:
    """같은 라벨로 이미 수집된 주소만 읽는다.

    같은 UTC 일자에 t0/daily/verdict 가 겹칠 수 있으므로 (판정일은 일별 크론과
    같은 날) 이어받기는 (address, label) 기준 — 다른 라벨 수집을 방해하지 않는다.
    잘린 gzip 멤버는 유효 행만으로 재작성한 뒤 읽는다 (recover_truncated)."""
    return {row["address"] for row in recover_truncated(out)
            if row.get("label") == label and "address" in row}


def utc_now_iso() -> str:
    """현재 UTC 시각 ISO 문자열."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def collect(label: str, cohort: Path = COHORT, out_dir: Path = SNAP_DIR,
            fetch: Callable[[str], object | None] = fetch_portfolio,
            pause_s: float = PAUSE_S) -> dict:
    """코호트 전 지갑의 perpAllTime 최종점을 오늘자(UTC) 파일로 수집한다 (멱등).

    실패 지갑은 기록하지 않고 목록을 로깅 — 같은 날 재실행 시 재시도된다.
    내구성: CHUNK 지갑마다 gzip 멤버를 닫아 확정 — kill 시 잘리는 것은 마지막
    청크뿐이고, 그 절단도 다음 이어받기의 load_done_labeled 가 복구한다."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{day}.jsonl.gz"
    wallets = [w["address"] for w in load_cohort_wallets(cohort)]
    done = load_done_labeled(out, label)      # 잘린 멤버는 여기서 복구 후 이어쓰기
    todo = [w for w in wallets if w not in done]
    logger.info("[%s] 코호트 %d | 완료 %d | 남음 %d → %s",
                label, len(wallets), len(done), len(todo), out)

    failed: list[str] = []
    n_ok = 0
    t_start = time.time()
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        with gzip.open(out, "at", encoding="utf-8") as fo:
            for addr in chunk:
                payload = fetch(addr)
                row = parse_row(addr, payload, label, utc_now_iso())
                if row is None:
                    failed.append(addr)
                else:
                    fo.write(json.dumps(row) + "\n")
                    n_ok += 1
                if pause_s:
                    time.sleep(pause_s)
        done_n = min(start + CHUNK, len(todo))
        if done_n < len(todo):
            rate = done_n / max(time.time() - t_start, 1e-9)
            logger.info("[%s] %d/%d (%.1f/s, ETA %.0f분)", label, done_n,
                        len(todo), rate, (len(todo) - done_n) / max(rate, 0.1) / 60)

    if failed:
        logger.warning("[%s] 실패 %d지갑 (다음 실행 재시도): %s",
                       label, len(failed), ", ".join(failed))
    logger.info("[%s] 완료: 신규 %d, 실패 %d, 누적 %d/%d",
                label, n_ok, len(failed), len(done) + n_ok, len(wallets))
    return dict(day=day, out=str(out), n_ok=n_ok, n_failed=len(failed),
                failed=failed, n_done=len(done) + n_ok, n_cohort=len(wallets))


def main(argv: list[str] | None = None) -> None:
    """CLI 진입점 — --label t0|daily|verdict 필수."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="H2 perpAllTime 스냅샷 수집기")
    ap.add_argument("--label", required=True, choices=LABELS,
                    help="수집 목적 태그 (t0=기준선, daily=일별 추적, verdict=판정일)")
    args = ap.parse_args(argv)
    collect(args.label)


if __name__ == "__main__":
    main()
