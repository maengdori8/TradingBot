"""Kill Zone 필터 — ICT 세션 시간대 (런던/뉴욕/아시아) 활성 여부 판별."""

from datetime import datetime, timezone


class KillZoneFilter:

    def __init__(self, config: dict):
        self.zones = {
            "london": (config["london_open"], config["london_close"]),
            "new_york": (config["ny_open"], config["ny_close"]),
            "asian": (config["asian_open"], config["asian_close"]),
        }

    def _parse_time(self, t: str):
        return datetime.strptime(t, "%H:%M").time()

    def is_active(self, now: datetime = None) -> dict:
        if now is None:
            now = datetime.now(timezone.utc)
        current_time = now.time()

        active = {}
        for name, (start_str, end_str) in self.zones.items():
            start = self._parse_time(start_str)
            end = self._parse_time(end_str)
            active[name] = start <= current_time <= end

        active["any"] = any(v for k, v in active.items() if k != "any")
        return active
