"""Loguru 기반 로거 설정."""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(config: dict):
    logger.remove()

    level = config.get("level", "DEBUG")
    rotation = config.get("rotation", "10 MB")
    retention = config.get("retention", "7 days")

    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "bot_{time:YYYY-MM-DD}.log",
        level=level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {module}:{function}:{line} | {message}",
    )
