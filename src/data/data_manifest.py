from __future__ import annotations

"""연구 증거가 참조하는 데이터 범위와 해시를 고정하는 manifest."""

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data.feature_store import DataQualitySummary, MarketFeatureStore
from src.data.market_snapshot import ensure_utc


def file_sha256(path: Path) -> str:
    """파일을 chunk 단위로 읽어 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_sequence(values: Sequence[str]) -> str:
    """순서가 보존된 SHA-256 목록의 Merkle-like root를 계산한다."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    """문자열이 lowercase 64자리 SHA-256인지 반환한다."""
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """중복 key가 있는 JSON object를 즉시 거부한다."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest JSON에 중복 key가 있습니다: {key}")
        result[key] = value
    return result


def _parse_utc(value: Any, label: str) -> datetime:
    """manifest의 ISO timestamp를 timezone-aware UTC로 변환한다."""
    if not isinstance(value, str):
        raise ValueError(f"{label}은 ISO timestamp 문자열이어야 합니다")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label}이 유효한 ISO timestamp가 아닙니다") from exc
    return ensure_utc(parsed)


def _quality_to_dict(quality: DataQualitySummary) -> dict[str, Any]:
    """품질 요약을 canonical JSON 호환 dict로 변환한다."""
    payload = asdict(quality)
    payload["start"] = ensure_utc(quality.start).isoformat()
    payload["end"] = ensure_utc(quality.end).isoformat()
    return payload


def _quality_from_dict(payload: Any) -> DataQualitySummary:
    """허용된 key만 가진 품질 요약을 복원한다."""
    if not isinstance(payload, dict):
        raise ValueError("manifest quality는 object여야 합니다")
    expected = {
        "dataset",
        "symbol",
        "timestamp_axis",
        "start",
        "end",
        "event_count",
        "expected_count",
        "completeness",
        "largest_gap_seconds",
        "unresolved_gap_count",
    }
    if set(payload) != expected:
        raise ValueError("manifest quality key 집합이 계약과 다릅니다")
    return DataQualitySummary(
        dataset=str(payload["dataset"]),
        symbol=str(payload["symbol"]),
        timestamp_axis=str(payload["timestamp_axis"]),
        start=_parse_utc(payload["start"], "quality.start"),
        end=_parse_utc(payload["end"], "quality.end"),
        event_count=int(payload["event_count"]),
        expected_count=int(payload["expected_count"]),
        completeness=float(payload["completeness"]),
        largest_gap_seconds=float(payload["largest_gap_seconds"]),
        unresolved_gap_count=int(payload["unresolved_gap_count"]),
    )


@dataclass(frozen=True)
class DataQualityBinding:
    """주 데이터셋 승급에 필수인 보조 데이터 품질과 원본 hash 묶음."""

    dataset: str
    raw_payload_root_sha256: str
    raw_payload_count: int
    quality: DataQualitySummary

    def __post_init__(self) -> None:
        """binding의 데이터셋·hash·품질 일치를 검증한다."""
        if not self.dataset.strip() or self.quality.dataset != self.dataset:
            raise ValueError("품질 binding 데이터셋이 비어 있거나 일치하지 않습니다")
        if not _is_sha256(self.raw_payload_root_sha256):
            raise ValueError("품질 binding raw payload root가 SHA-256이 아닙니다")
        if self.raw_payload_count < 0:
            raise ValueError("품질 binding raw payload count는 음수일 수 없습니다")

    @property
    def evidence_eligible(self) -> bool:
        """원본과 데이터 품질이 모두 증거 기준을 통과하는지 반환한다."""
        return self.raw_payload_count > 0 and self.quality.evidence_eligible

    def to_dict(self) -> dict[str, Any]:
        """canonical JSON에 포함할 dict를 반환한다."""
        return {
            "dataset": self.dataset,
            "raw_payload_root_sha256": self.raw_payload_root_sha256,
            "raw_payload_count": self.raw_payload_count,
            "quality": _quality_to_dict(self.quality),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DataQualityBinding:
        """strict JSON object에서 품질 binding을 복원한다."""
        if not isinstance(payload, dict):
            raise ValueError("manifest required binding은 object여야 합니다")
        expected = {
            "dataset",
            "raw_payload_root_sha256",
            "raw_payload_count",
            "quality",
        }
        if set(payload) != expected:
            raise ValueError("manifest required binding key 집합이 다릅니다")
        return cls(
            dataset=str(payload["dataset"]),
            raw_payload_root_sha256=str(payload["raw_payload_root_sha256"]),
            raw_payload_count=int(payload["raw_payload_count"]),
            quality=_quality_from_dict(payload["quality"]),
        )


@dataclass(frozen=True)
class DataManifest:
    """한 데이터 구간을 재현·감사하는 데 필요한 불변 메타데이터."""

    dataset: str
    symbol: str
    start: datetime
    end: datetime
    generated_at: datetime
    code_commit: str
    raw_payload_root_sha256: str
    raw_payload_count: int
    source_file_sha256: dict[str, str]
    quality: DataQualitySummary
    required_bindings: tuple[DataQualityBinding, ...] = ()
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """manifest 범위·hash·품질 연결을 검증하고 증거 hash를 계산한다."""
        ensure_utc(self.start)
        ensure_utc(self.end)
        ensure_utc(self.generated_at)
        if self.end <= self.start:
            raise ValueError("manifest end는 start보다 뒤여야 합니다")
        if re.fullmatch(r"[0-9a-f]{7,64}", self.code_commit) is None:
            raise ValueError("manifest code_commit은 7~64자리 lowercase hex여야 합니다")
        if not _is_sha256(self.raw_payload_root_sha256):
            raise ValueError("manifest raw payload root가 SHA-256이 아닙니다")
        if self.raw_payload_count < 0:
            raise ValueError("manifest raw payload count는 음수일 수 없습니다")
        if self.quality.dataset != self.dataset or self.quality.symbol != self.symbol:
            raise ValueError("manifest 품질 요약의 데이터셋 또는 심볼이 다릅니다")
        for path, digest in self.source_file_sha256.items():
            if not path.strip() or not _is_sha256(digest):
                raise ValueError("manifest source file hash가 올바르지 않습니다")
        binding_names: set[str] = set()
        for binding in self.required_bindings:
            if binding.dataset in binding_names:
                raise ValueError("manifest required binding 데이터셋이 중복됩니다")
            binding_names.add(binding.dataset)
            if (
                binding.quality.symbol != self.symbol
                or binding.quality.start != self.start
                or binding.quality.end != self.end
            ):
                raise ValueError(
                    "manifest required binding 범위가 주 데이터와 다릅니다"
                )
        canonical = json.dumps(
            self.to_dict(include_evidence_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        object.__setattr__(
            self,
            "evidence_hash",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @property
    def evidence_eligible(self) -> bool:
        """원시 레코드와 99%·15분 gap 기준을 모두 충족하는지 반환한다."""
        primary_has_evidence = (
            self.raw_payload_count > 0 or self.dataset == "liquidation"
        )
        bindings_eligible = all(
            binding.evidence_eligible for binding in self.required_bindings
        )
        if self.dataset == "liquidation":
            required_dataset = "heartbeat:public_ws_all_liquidation_connection"
            has_connection_binding = any(
                binding.dataset == required_dataset
                for binding in self.required_bindings
            )
            bindings_eligible = bindings_eligible and has_connection_binding
        return (
            primary_has_evidence
            and self.quality.evidence_eligible
            and bindings_eligible
        )

    def assert_evidence_eligible(self) -> None:
        """연구 증거로 사용할 수 없는 manifest를 fail-closed 처리한다."""
        if not self.evidence_eligible:
            raise RuntimeError(
                "데이터 manifest가 승급 증거 기준을 충족하지 않습니다: "
                f"completeness={self.quality.completeness:.4f}, "
                f"largest_gap={self.quality.largest_gap_seconds:.3f}s, "
                f"unresolved={self.quality.unresolved_gap_count}"
            )

    def to_dict(self, include_evidence_hash: bool = True) -> dict[str, Any]:
        """서명·저장을 위한 JSON 호환 dict를 반환한다."""
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "symbol": self.symbol,
            "start": ensure_utc(self.start).isoformat(),
            "end": ensure_utc(self.end).isoformat(),
            "generated_at": ensure_utc(self.generated_at).isoformat(),
            "code_commit": self.code_commit,
            "raw_payload_root_sha256": self.raw_payload_root_sha256,
            "raw_payload_count": self.raw_payload_count,
            "source_file_sha256": dict(self.source_file_sha256),
            "quality": _quality_to_dict(self.quality),
            "required_bindings": [
                binding.to_dict() for binding in self.required_bindings
            ],
        }
        if include_evidence_hash:
            evidence_hash = getattr(self, "evidence_hash", None)
            if evidence_hash is not None:
                payload["evidence_hash"] = evidence_hash
        return payload

    def to_json(self) -> str:
        """외부 고정 hash와 함께 보관할 canonical JSON을 반환한다."""
        return json.dumps(
            self.to_dict(include_evidence_hash=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(
        cls,
        serialized: str,
        expected_sha256: str,
    ) -> DataManifest:
        """중복 key를 거부하고 고정 evidence hash가 일치할 때만 복원한다."""
        if not _is_sha256(expected_sha256):
            raise ValueError("expected_sha256은 lowercase SHA-256이어야 합니다")
        try:
            payload = json.loads(serialized, object_pairs_hook=_strict_object)
        except json.JSONDecodeError as exc:
            raise ValueError("manifest JSON 형식이 올바르지 않습니다") from exc
        if not isinstance(payload, dict):
            raise ValueError("manifest JSON 최상위 값은 object여야 합니다")
        expected_keys = {
            "dataset",
            "symbol",
            "start",
            "end",
            "generated_at",
            "code_commit",
            "raw_payload_root_sha256",
            "raw_payload_count",
            "source_file_sha256",
            "quality",
            "required_bindings",
            "evidence_hash",
        }
        if set(payload) != expected_keys:
            raise ValueError("manifest JSON key 집합이 계약과 다릅니다")
        declared_hash = str(payload["evidence_hash"])
        if not _is_sha256(declared_hash):
            raise ValueError("manifest evidence_hash가 SHA-256이 아닙니다")
        raw_sources = payload["source_file_sha256"]
        if not isinstance(raw_sources, dict):
            raise ValueError("manifest source_file_sha256은 object여야 합니다")
        raw_bindings = payload["required_bindings"]
        if not isinstance(raw_bindings, list):
            raise ValueError("manifest required_bindings는 배열이어야 합니다")
        manifest = cls(
            dataset=str(payload["dataset"]),
            symbol=str(payload["symbol"]),
            start=_parse_utc(payload["start"], "manifest.start"),
            end=_parse_utc(payload["end"], "manifest.end"),
            generated_at=_parse_utc(payload["generated_at"], "manifest.generated_at"),
            code_commit=str(payload["code_commit"]),
            raw_payload_root_sha256=str(payload["raw_payload_root_sha256"]),
            raw_payload_count=int(payload["raw_payload_count"]),
            source_file_sha256={
                str(path): str(digest) for path, digest in raw_sources.items()
            },
            quality=_quality_from_dict(payload["quality"]),
            required_bindings=tuple(
                DataQualityBinding.from_dict(item) for item in raw_bindings
            ),
        )
        if not hmac.compare_digest(declared_hash, manifest.evidence_hash):
            raise ValueError("manifest 내부 evidence_hash가 내용과 일치하지 않습니다")
        if not hmac.compare_digest(expected_sha256, manifest.evidence_hash):
            raise ValueError("manifest evidence_hash가 외부 고정 hash와 다릅니다")
        return manifest


def build_data_manifest(
    store: MarketFeatureStore,
    dataset: str,
    symbol: str,
    start: datetime,
    end: datetime,
    expected_interval_seconds: float,
    code_commit: str,
    source_files: Sequence[Path] = (),
    timestamp_axis: str = "receive",
    require_evidence_eligible: bool = True,
    required_heartbeat_feed: str | None = None,
    heartbeat_interval_seconds: float = 30.0,
) -> DataManifest:
    """저장소 품질·원본 hash와 선택 파일 hash로 manifest를 생성한다."""
    started = ensure_utc(start)
    ended = ensure_utc(end)
    quality = store.summarize_quality(
        dataset=dataset,
        symbol=symbol,
        start=started,
        end=ended,
        expected_interval_seconds=expected_interval_seconds,
        maximum_allowed_gap_seconds=900.0,
        timestamp_axis=timestamp_axis,
    )
    raw_hashes = store.payload_hashes(
        dataset,
        symbol,
        started,
        ended,
        timestamp_axis=timestamp_axis,
    )
    files = {
        str(path.resolve()): file_sha256(path)
        for path in sorted(source_files, key=lambda item: str(item.resolve()))
    }
    heartbeat_feed = required_heartbeat_feed
    if dataset == "liquidation":
        required_feed = "public_ws_all_liquidation_connection"
        if heartbeat_feed is not None and heartbeat_feed != required_feed:
            raise ValueError("liquidation은 공식 연결 heartbeat에만 결합할 수 있습니다")
        heartbeat_feed = required_feed
    bindings: tuple[DataQualityBinding, ...] = ()
    if heartbeat_feed is not None:
        heartbeat_dataset = f"heartbeat:{heartbeat_feed}"
        heartbeat_quality = store.summarize_quality(
            dataset=heartbeat_dataset,
            symbol=symbol,
            start=started,
            end=ended,
            expected_interval_seconds=heartbeat_interval_seconds,
            maximum_allowed_gap_seconds=900.0,
            timestamp_axis="receive",
        )
        heartbeat_hashes = store.payload_hashes(
            heartbeat_dataset,
            symbol,
            started,
            ended,
            timestamp_axis="receive",
        )
        bindings = (
            DataQualityBinding(
                dataset=heartbeat_dataset,
                raw_payload_root_sha256=_hash_sequence(heartbeat_hashes),
                raw_payload_count=len(heartbeat_hashes),
                quality=heartbeat_quality,
            ),
        )
    manifest = DataManifest(
        dataset=dataset,
        symbol=symbol,
        start=started,
        end=ended,
        generated_at=datetime.now(timezone.utc),
        code_commit=code_commit,
        raw_payload_root_sha256=_hash_sequence(raw_hashes),
        raw_payload_count=len(raw_hashes),
        source_file_sha256=files,
        quality=quality,
        required_bindings=bindings,
    )
    if require_evidence_eligible:
        manifest.assert_evidence_eligible()
    return manifest
