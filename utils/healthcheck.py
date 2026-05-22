"""헬스체크 — 봇 상태 감시 및 heartbeat 파일 갱신."""

import time
import threading
from pathlib import Path

from loguru import logger

HEARTBEAT_FILE = Path("healthcheck")


class HealthCheck:

    def __init__(self, interval: int = 30):
        self.interval = interval
        self._running = False
        self._thread = None
        self._last_tick_time = 0.0
        self._tick_count = 0

    def record_tick(self):
        self._last_tick_time = time.time()
        self._tick_count += 1

    def _write_heartbeat(self):
        while self._running:
            try:
                status = {
                    "alive": True,
                    "last_tick": self._last_tick_time,
                    "tick_count": self._tick_count,
                    "uptime": time.time() - self._start_time,
                }
                HEARTBEAT_FILE.write_text(
                    f"ts={time.time():.0f}\n"
                    f"ticks={self._tick_count}\n"
                    f"last_tick={self._last_tick_time:.0f}\n"
                    f"uptime={status['uptime']:.0f}\n"
                )
            except Exception as e:
                logger.warning(f"[HEALTH] heartbeat 쓰기 실패: {e}")
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._write_heartbeat, daemon=True)
        self._thread.start()
        logger.info(f"[HEALTH] 헬스체크 시작 (간격: {self.interval}s)")

    def stop(self):
        self._running = False
        if HEARTBEAT_FILE.exists():
            HEARTBEAT_FILE.unlink()
