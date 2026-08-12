from __future__ import annotations

# legacy ICT 연구 파이프라인 — 다운로드 → 신호 재생 → 레짐 태깅/집계.

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from research import aggregate, study  # noqa: E402

logger = logging.getLogger("run_study")


def main() -> None:
    """파이프라인 실행."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="1심볼 스모크")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    t0 = time.time()
    import ccxt
    ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})

    if args.smoke:
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        months, workers = 2, 2
    else:
        symbols = study.top_symbols(ex, args.top)
        months, workers = args.months, args.workers
    logger.info("대상 %d심볼 × %d개월", len(symbols), months)

    logger.info("─── 1/3 데이터 다운로드 ───")
    study.download_all(symbols, months)

    logger.info("─── 2/3 신호 재생 (%d workers) ───", workers)
    df = study.run_replay(symbols, workers)
    logger.info("신호 총 %d개", len(df))
    if len(df) == 0:
        logger.error("신호 0개 — 중단")
        return

    logger.info("─── 3/3 레짐 태깅 + 집계 ───")
    aggregate.main()
    logger.info("완료 (%.1f분)", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
