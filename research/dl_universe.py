"""WFO 유니버스 데이터 다운로더 — research/data/*.pkl 캐시를 채운다.

자율 연구 루프가 검증에 쓸 캔들 데이터를 사전 확보한다. study.fetch_history의
캐시(존재 시 스킵)를 그대로 사용하므로 재실행 안전(idempotent)하다.

사용법:
    python -m research.dl_universe --start 2024-01-01
    python -m research.dl_universe --start 2024-01-01 --symbols BTC/USDT:USDT ETH/USDT:USDT
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402
import research.wfo as wfo  # noqa: E402

logger = logging.getLogger("dl_universe")


def main() -> None:
    """유니버스 전 심볼의 15m/1h/4h 데이터를 캐시에 받는다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="미지정 시 wfo.UNIVERSE 사용")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import ccxt
    import pandas as pd

    symbols = args.symbols or wfo.UNIVERSE
    months = int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(args.start, tz="UTC")).days / 30) + 1
    logger.info("다운로드 시작: %d심볼 × ~%d개월 (start=%s)", len(symbols), months, args.start)

    ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        for tf in ("15m", "1h", "4h"):
            try:
                df = study.fetch_history(ex, sym, tf, months)
                logger.info("  %s %s: %d캔들", sym, tf, len(df))
            except Exception as e:  # noqa: BLE001  (네트워크/거래소 다양한 예외 — 루프 지속)
                logger.warning("  %s %s 실패: %s", sym, tf, e)
        logger.info("진행 %d/%d (%.1f분 경과)", i, len(symbols), (time.time() - t0) / 60)

    logger.info("다운로드 완료 (%.1f분)", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
