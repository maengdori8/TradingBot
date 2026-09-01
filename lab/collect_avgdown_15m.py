"""AVGDOWN-2026-09-01 15분봉 수집 — Bybit USDT 무기한, ccxt 페이징.

수집 범위 (동결 — 심볼별 1h 창과 동일, `docs/PREREGISTRATION_AVGDOWN_2026-09-01.md` §4.1):
  BTC 2021-01-01 00:00 → 2026-08-24 02:45 UTC  (1h 마지막 봉 02:00 + 45m)
  ETH 2021-03-15 00:00 → 2026-08-24 02:45 UTC
  SOL 2021-10-15 00:00 → 2026-08-26 04:45 UTC

규약:
- 멱등: 대상 parquet 이 이미 존재하면 재수집하지 않는다 (재실행 무변화).
- 원자성: 임시파일에 쓰고 os.replace 로 rename.
- fail-closed: 수집 결과에 창 밖 봉·중복 봉이 있으면 제거하되, 결측 봉은
  보간하지 않는다 — 결측 수를 집계해 출력하고 그대로 저장한다.
- 형식: `lab/data/sol_1h.parquet` 동형 — UTC DatetimeIndex(name='ts'),
  columns open/high/low/close/volume float64.

실행: .venv/bin/python lab/collect_avgdown_15m.py
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time

import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 심볼별 [시작, 종료] (양끝 포함, UTC) — 1h 동결 창에서 유도 (§4.1)
WINDOWS: dict[str, tuple[str, str, str]] = {
    "BTC": ("BTC/USDT:USDT", "2021-01-01T00:00:00Z", "2026-08-24T02:45:00Z"),
    "ETH": ("ETH/USDT:USDT", "2021-03-15T00:00:00Z", "2026-08-24T02:45:00Z"),
    "SOL": ("SOL/USDT:USDT", "2021-10-15T00:00:00Z", "2026-08-26T04:45:00Z"),
}
OUT = {s: os.path.join(ROOT, "lab", "data", f"{s.lower()}_15m.parquet") for s in WINDOWS}
TF, TF_MS, LIMIT = "15m", 15 * 60 * 1000, 1000
PD_FREQ = "15min"                                  # pandas 전용 표기 (ccxt 는 '15m')


def sha256(path: str) -> str:
    """파일 SHA256 16진 문자열."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_symbol(ex: ccxt.Exchange, sym: str) -> pd.DataFrame:
    """한 심볼의 15m OHLCV 를 창 전체에 대해 페이징 수집한다.

    Args:
        ex: ccxt bybit 인스턴스 (defaultType='swap').
        sym: 'BTC' 등 내부 심볼 키.

    Returns:
        UTC DatetimeIndex(name='ts') OHLCV DataFrame (창 내부, 중복 제거, 정렬).
    """
    market, t0, t1 = WINDOWS[sym]
    start = int(pd.Timestamp(t0).timestamp() * 1000)
    end = int(pd.Timestamp(t1).timestamp() * 1000)
    rows: list[list[float]] = []
    since = start
    while since <= end:
        for attempt in range(5):
            try:
                batch = ex.fetch_ohlcv(market, TF, since=since, limit=LIMIT)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                logger.warning("%s since=%d 재시도 %d: %s", sym, since, attempt + 1, exc)
                time.sleep(2.0 * (attempt + 1))
        else:
            raise SystemExit(f"{sym} 수집 실패 — fail-closed 중단 (부분 저장 없음)")
        if not batch:
            logger.warning("%s since=%s 빈 응답 — 1000봉 건너뜀", sym,
                           pd.Timestamp(since, unit="ms", tz="utc"))
            since += LIMIT * TF_MS
            continue
        rows.extend(r for r in batch if start <= r[0] <= end)
        last = batch[-1][0]
        if last < since:                       # 진행 없음 → 중단 (무한루프 방지)
            raise SystemExit(f"{sym} 페이징 역행: {last} < {since}")
        since = last + TF_MS
        if len(rows) % 20000 < LIMIT:
            logger.info("%s … %s (%d봉)", sym,
                        pd.Timestamp(last, unit="ms", tz="utc"), len(rows))
    df = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ms").sort_values("ms")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ms"), unit="ms", utc=True),
                                name="ts")
    return df.astype("float64")


def main() -> int:
    """수집 본체 — 심볼별 멱등 수집·검증·원자 저장."""
    ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    for sym in WINDOWS:
        path = OUT[sym]
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logger.info("%s 이미 존재 — 재사용 (%d봉, %s~%s)", sym, len(df),
                        df.index.min(), df.index.max())
        else:
            df = fetch_symbol(ex, sym)
            tmp = path + ".tmp"
            df.to_parquet(tmp)
            os.replace(tmp, path)                     # 원자적 교체
        _, t0, t1 = WINDOWS[sym]
        full = pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq=PD_FREQ)
        missing = len(full) - len(df)
        assert df.index.is_monotonic_increasing and df.index.is_unique
        assert df.index.min() >= pd.Timestamp(t0) and df.index.max() <= pd.Timestamp(t1)
        print(f"{sym}: {df.index.min()} ~ {df.index.max()}  {len(df)}봉  "
              f"기대 {len(full)}  결측 {missing}  sha256={sha256(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
