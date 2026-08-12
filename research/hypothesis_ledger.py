from __future__ import annotations

"""사전 가설 등록과 결과 보존을 위한 append-only JSONL 원장."""

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Literal, Mapping

logger = logging.getLogger(__name__)

MAX_CONFIGS_PER_FAMILY = 20
_DEFAULT_LOCK_TIMEOUT_SECONDS = 10.0
_DEFAULT_STALE_LOCK_SECONDS = 120.0
_LOCK_POLL_SECONDS = 0.05


def _canonical_json(value: object) -> str:
    """해시 입력에 사용할 결정적 JSON 문자열을 만든다."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class _PortableFileLock:
    """원자적 lockfile 생성으로 구현한 운영체제 독립 프로세스 잠금."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        stale_after_seconds: float,
    ) -> None:
        """잠금 경로와 대기·stale 제한을 설정한다."""
        if timeout_seconds <= 0:
            raise ValueError("lock timeout은 양수여야 합니다")
        if stale_after_seconds <= 0:
            raise ValueError("stale lock 제한은 양수여야 합니다")
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_after_seconds = stale_after_seconds
        self.token = uuid.uuid4().hex
        self._acquired = False

    @property
    def _breaker_path(self) -> Path:
        """stale 잠금 정리 경쟁을 막는 보조 잠금 경로를 반환한다."""
        return self.path.with_name(f"{self.path.name}.breaker")

    def _create(self) -> bool:
        """배타적 파일 생성에 성공하면 잠금 소유 정보를 기록한다."""
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            return False
        owner = {
            "token": self.token,
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            os.write(descriptor, _canonical_json(owner).encode("utf-8"))
            os.fsync(descriptor)
        except OSError:
            os.close(descriptor)
            try:
                self.path.unlink()
            except OSError as cleanup_error:
                logger.warning(
                    "초기화 실패 잠금 정리 실패 path=%s error=%s",
                    self.path,
                    cleanup_error,
                )
            raise
        os.close(descriptor)
        self._acquired = True
        return True

    @staticmethod
    def _age_seconds(path: Path) -> float | None:
        """파일이 존재하면 수정 이후 경과 초를 반환한다."""
        try:
            return max(0.0, time.time() - path.stat().st_mtime)
        except FileNotFoundError:
            return None

    def _remove_stale_breaker(self) -> None:
        """중단된 정리 작업이 남긴 오래된 breaker를 제거한다."""
        age = self._age_seconds(self._breaker_path)
        if age is None or age <= self.stale_after_seconds:
            return
        try:
            self._breaker_path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.debug("stale breaker 제거 실패 path=%s error=%s", self.path, exc)

    def _remove_stale_lock(self) -> bool:
        """단일 정리자만 stale lock을 재확인하고 제거한다."""
        age = self._age_seconds(self.path)
        if age is None:
            return True
        if age <= self.stale_after_seconds:
            return False

        self._remove_stale_breaker()
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            breaker = os.open(self._breaker_path, flags, 0o600)
        except FileExistsError:
            return False
        try:
            os.close(breaker)
            current_age = self._age_seconds(self.path)
            if current_age is None:
                return True
            if current_age <= self.stale_after_seconds:
                return False
            try:
                self.path.unlink()
            except FileNotFoundError:
                return True
            logger.warning("stale 가설 원장 잠금 제거 path=%s", self.path)
            return True
        finally:
            try:
                self._breaker_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.debug(
                    "stale breaker 해제 실패 path=%s error=%s",
                    self.path,
                    exc,
                )

    def acquire(self) -> None:
        """타임아웃까지 잠금 생성을 재시도한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if self._create():
                return
            self._remove_stale_lock()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"가설 원장 잠금 획득 시간 초과: {self.path}")
            time.sleep(_LOCK_POLL_SECONDS)

    def _owned_token(self) -> str | None:
        """현재 lockfile에 기록된 소유 토큰을 읽는다."""
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        token = owner.get("token") if isinstance(owner, dict) else None
        return token if isinstance(token, str) else None

    def release(self) -> None:
        """자신이 소유한 lockfile만 제거한다."""
        if not self._acquired:
            return
        try:
            if self._owned_token() == self.token:
                self.path.unlink()
            else:
                logger.warning("가설 원장 잠금 소유권 불일치 path=%s", self.path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("가설 원장 잠금 해제 실패 path=%s error=%s", self.path, exc)
        finally:
            self._acquired = False

    def __enter__(self) -> _PortableFileLock:
        """컨텍스트 진입 시 잠금을 획득한다."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """컨텍스트 종료 시 잠금을 해제한다."""
        self.release()


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

    def __init__(
        self,
        path: Path | str,
        *,
        lock_timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS,
        stale_lock_seconds: float = _DEFAULT_STALE_LOCK_SECONDS,
    ) -> None:
        """원장 경로와 portable lock 제한을 설정한다."""
        self.path = Path(path)
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds는 양수여야 합니다")
        if stale_lock_seconds <= 0:
            raise ValueError("stale_lock_seconds는 양수여야 합니다")

    def _lock(self) -> _PortableFileLock:
        """원장 파일에 대응하는 프로세스 잠금을 만든다."""
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        return _PortableFileLock(
            lock_path,
            timeout_seconds=self.lock_timeout_seconds,
            stale_after_seconds=self.stale_lock_seconds,
        )

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
                "run_result",
            }:
                raise ValueError(f"가설 원장 {line_number}행 이벤트가 유효하지 않습니다")
            events.append(event)
        return events

    def read_events(self) -> list[dict[str, object]]:
        """현재 원장의 모든 이벤트를 순서대로 읽는다."""
        with self._lock():
            if not self.path.exists():
                return []
            with self.path.open("r", encoding="utf-8") as ledger_file:
                return self._parse_lines(ledger_file.readlines())

    def register(self, spec: HypothesisSpec) -> str:
        """가설을 실행 전에 등록하고 매니페스트 해시를 반환한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            with self.path.open("a+", encoding="utf-8") as ledger_file:
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
                os.fsync(ledger_file.fileno())
                logger.info(
                    "가설 등록 family=%s hypothesis_id=%s hash=%s",
                    spec.family,
                    spec.hypothesis_id,
                    spec.manifest_hash,
                )
                return spec.manifest_hash

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
        with self._lock():
            with self.path.open("a+", encoding="utf-8") as ledger_file:
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
                os.fsync(ledger_file.fileno())
                logger.info("가설 결과 기록 hash=%s status=%s", manifest_hash, status)

    def record_run_result(
        self,
        hypothesis_hash: str,
        run_manifest_hash: str,
        status: Literal["succeeded", "insufficient_data", "failed"],
        metrics: Mapping[str, object],
        *,
        note: str = "",
    ) -> None:
        """실행 hash별 결과를 idempotent하게 append-only 원장에 기록한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if len(run_manifest_hash) != 64:
            raise ValueError("run_manifest_hash는 SHA-256이어야 합니다")
        event_payload = {
            "event": "run_result",
            "hypothesis_hash": hypothesis_hash,
            "run_manifest_hash": run_manifest_hash,
            "status": status,
            "metrics": dict(metrics),
            "note": note,
        }
        _canonical_json(event_payload)
        with self._lock():
            with self.path.open("a+", encoding="utf-8") as ledger_file:
                ledger_file.seek(0)
                events = self._parse_lines(ledger_file.readlines())
                known = any(
                    event["event"] == "registered"
                    and event.get("manifest_hash") == hypothesis_hash
                    for event in events
                )
                if not known:
                    raise ValueError("등록되지 않은 hypothesis_hash의 실행 결과입니다")
                previous = next(
                    (
                        event
                        for event in events
                        if event["event"] == "run_result"
                        and event.get("run_manifest_hash") == run_manifest_hash
                    ),
                    None,
                )
                if previous is not None:
                    comparable = dict(previous)
                    comparable.pop("recorded_at", None)
                    if comparable == event_payload:
                        return
                    raise ValueError("동일 run_manifest_hash에 다른 결과를 기록할 수 없습니다")
                event = {
                    **event_payload,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                ledger_file.seek(0, 2)
                ledger_file.write(_canonical_json(event) + "\n")
                ledger_file.flush()
                os.fsync(ledger_file.fileno())
                logger.info(
                    "가설 실행 결과 기록 run_hash=%s status=%s",
                    run_manifest_hash,
                    status,
                )
