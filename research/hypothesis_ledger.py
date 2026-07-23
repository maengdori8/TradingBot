from __future__ import annotations

"""사전 가설 등록과 결과 보존을 위한 append-only JSONL 원장."""

import fcntl
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

logger = logging.getLogger(__name__)

MAX_CONFIGS_PER_FAMILY = 20


def _canonical_json(value: object) -> str:
    """해시 입력에 사용할 결정적 JSON 문자열을 만든다."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class HypothesisSpec:
    """실험 전에 고정해야 하는 가설 매니페스트."""

    hypothesis_id: str
    family: str
    thesis: str
    features: tuple[str, ...]
    universe: Mapping[str, object]
    parameters: Mapping[str, object]
    costs: Mapping[str, object]
    primary_metric: str
    created_by: str

    def __post_init__(self) -> None:
        """필수 필드와 JSON 직렬화 가능성을 검증한다."""
        required = (
            self.hypothesis_id,
            self.family,
            self.thesis,
            self.primary_metric,
            self.created_by,
        )
        if any(not value.strip() for value in required):
            raise ValueError("가설 원장의 필수 문자열은 비어 있을 수 없습니다")
        if not self.features:
            raise ValueError("features는 하나 이상이어야 합니다")
        _canonical_json(self.manifest())

    def manifest(self) -> dict[str, object]:
        """해시에 포함되는 불변 가설 매니페스트를 반환한다."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "family": self.family,
            "thesis": self.thesis,
            "features": list(self.features),
            "universe": dict(self.universe),
            "parameters": dict(self.parameters),
            "costs": dict(self.costs),
            "primary_metric": self.primary_metric,
            "created_by": self.created_by,
        }

    @property
    def manifest_hash(self) -> str:
        """가설 매니페스트의 SHA-256 해시를 반환한다."""
        payload = _canonical_json(self.manifest()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class HypothesisLedger:
    """등록과 결과 이벤트만 뒤에 추가하는 파일 기반 가설 원장."""

    def __init__(self, path: Path | str) -> None:
        """원장 경로를 설정한다."""
        self.path = Path(path)

    @staticmethod
    def _parse_lines(lines: list[str]) -> list[dict[str, object]]:
        """JSONL 행을 이벤트 목록으로 검증해 변환한다."""
        events: list[dict[str, object]] = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"가설 원장 {line_number}행이 손상됐습니다"
                ) from exc
            if not isinstance(event, dict) or event.get("event") not in {
                "registered",
                "result",
            }:
                raise ValueError(f"가설 원장 {line_number}행 이벤트가 유효하지 않습니다")
            events.append(event)
        return events

    def read_events(self) -> list[dict[str, object]]:
        """현재 원장의 모든 이벤트를 순서대로 읽는다."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as ledger_file:
            fcntl.flock(ledger_file.fileno(), fcntl.LOCK_SH)
            try:
                return self._parse_lines(ledger_file.readlines())
            finally:
                fcntl.flock(ledger_file.fileno(), fcntl.LOCK_UN)

    def register(self, spec: HypothesisSpec) -> str:
        """가설을 실행 전에 등록하고 매니페스트 해시를 반환한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as ledger_file:
            fcntl.flock(ledger_file.fileno(), fcntl.LOCK_EX)
            try:
                ledger_file.seek(0)
                events = self._parse_lines(ledger_file.readlines())
                registrations = [
                    event for event in events if event["event"] == "registered"
                ]
                for event in registrations:
                    if event.get("manifest_hash") == spec.manifest_hash:
                        return spec.manifest_hash
                    manifest = event.get("manifest")
                    if (
                        isinstance(manifest, dict)
                        and manifest.get("hypothesis_id") == spec.hypothesis_id
                    ):
                        raise ValueError(
                            "같은 hypothesis_id를 다른 매니페스트로 재등록할 수 없습니다"
                        )
                family_count = sum(
                    1
                    for event in registrations
                    if isinstance(event.get("manifest"), dict)
                    and event["manifest"].get("family") == spec.family
                )
                if family_count >= MAX_CONFIGS_PER_FAMILY:
                    raise ValueError(
                        f"{spec.family} 설정은 최대 {MAX_CONFIGS_PER_FAMILY}개입니다"
                    )
                event = {
                    "event": "registered",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "manifest_hash": spec.manifest_hash,
                    "manifest": spec.manifest(),
                }
                ledger_file.seek(0, 2)
                ledger_file.write(_canonical_json(event) + "\n")
                ledger_file.flush()
                logger.info(
                    "가설 등록 family=%s hypothesis_id=%s hash=%s",
                    spec.family,
                    spec.hypothesis_id,
                    spec.manifest_hash,
                )
                return spec.manifest_hash
            finally:
                fcntl.flock(ledger_file.fileno(), fcntl.LOCK_UN)

    def record_result(
        self,
        manifest_hash: str,
        status: Literal["succeeded", "failed", "rejected"],
        metrics: Mapping[str, object],
        *,
        note: str = "",
    ) -> None:
        """등록된 가설의 최종 결과를 새 이벤트로 한 번만 추가한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _canonical_json(dict(metrics))
        with self.path.open("a+", encoding="utf-8") as ledger_file:
            fcntl.flock(ledger_file.fileno(), fcntl.LOCK_EX)
            try:
                ledger_file.seek(0)
                events = self._parse_lines(ledger_file.readlines())
                known = any(
                    event["event"] == "registered"
                    and event.get("manifest_hash") == manifest_hash
                    for event in events
                )
                if not known:
                    raise ValueError("등록되지 않은 manifest_hash의 결과입니다")
                completed = any(
                    event["event"] == "result"
                    and event.get("manifest_hash") == manifest_hash
                    for event in events
                )
                if completed:
                    raise ValueError("동일 매니페스트의 최종 결과가 이미 기록됐습니다")
                event = {
                    "event": "result",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "manifest_hash": manifest_hash,
                    "status": status,
                    "metrics": dict(metrics),
                    "note": note,
                }
                ledger_file.seek(0, 2)
                ledger_file.write(_canonical_json(event) + "\n")
                ledger_file.flush()
                logger.info("가설 결과 기록 hash=%s status=%s", manifest_hash, status)
            finally:
                fcntl.flock(ledger_file.fileno(), fcntl.LOCK_UN)
