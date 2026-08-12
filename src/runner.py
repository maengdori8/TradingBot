from __future__ import annotations

# 봇 주기 실행 러너 — 로컬에서 봇을 일정 간격으로 반복 실행한다.
# 대시보드와 함께 띄워두면 관심종목/포지션이 실시간(주기)으로 갱신된다.
#
# 사용법:
#     python -m src.runner                 # config.runner.interval_sec 사용
#     python -m src.runner --interval 300  # 5분마다
#     python -m src.runner --once          # 1회만 실행
#
# GitHub Actions의 15분 cron을 쓰는 경우 이 러너는 불필요하다.

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.bot import run

logger = logging.getLogger("runner")


def _load_interval(default: int = 900) -> int:
    """config.yaml의 runner.interval_sec 값을 읽는다 (없으면 기본값)."""
    try:
        with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("runner", {}).get("interval_sec", default))
    except (FileNotFoundError, ValueError, TypeError):
        return default


def main() -> None:
    """주기 실행 루프 진입점."""
    parser = argparse.ArgumentParser(description="ICT 봇 주기 실행 러너")
    parser.add_argument(
        "--interval", type=int, default=None,
        help="실행 간격(초). 미지정 시 config.runner.interval_sec 또는 900",
    )
    parser.add_argument(
        "--once", action="store_true", help="1회만 실행하고 종료",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    interval = args.interval if args.interval is not None else _load_interval()

    if args.once:
        logger.info("봇 1회 실행 모드")
        run()
        return

    logger.info("=" * 56)
    logger.info("ICT 봇 주기 실행 시작 — 간격 %d초 (%.1f분)", interval, interval / 60)
    logger.info("Ctrl+C로 종료")
    logger.info("=" * 56)

    cycle = 0
    while True:
        cycle += 1
        start = time.time()
        logger.info("───── 사이클 #%d 시작 ─────", cycle)
        try:
            run()
        except KeyboardInterrupt:
            logger.info("사용자 종료 (Ctrl+C)")
            break
        except Exception as e:  # 한 사이클 실패가 루프를 죽이지 않도록
            logger.error("사이클 #%d 실패: %s", cycle, e, exc_info=True)

        elapsed = time.time() - start
        sleep_for = max(1, interval - int(elapsed))
        logger.info(
            "사이클 #%d 완료 (%.1f초 소요), 다음 실행까지 %d초 대기",
            cycle, elapsed, sleep_for,
        )
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("사용자 종료 (Ctrl+C)")
            break


if __name__ == "__main__":
    main()
