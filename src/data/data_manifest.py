from __future__ import annotations

"""연구 증거가 참조하는 데이터 범위와 해시를 고정하는 manifest."""

import hashlib
import json
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
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """manifest 범위·hash·품질 연결을 검증하고 증거 hash를 계산한다."""
        ensure_utc(self.start)
        ensure_utc(self.end)
        ensure_utc(self.generated_at)
        if self.end <= self.start:
            raise ValueError("manifest end는 start보다 뒤여야 합니다")
        if not self.code_commit.strip():
            raise ValueError("manifest code_commit은 비어 있을 수 없습니다")
        if len(self.raw_payload_root_sha256) != 64:
            raise ValueError("manifest raw payload root가 SHA-256이 아닙니다")
        if self.raw_payload_count < 0:
            raise ValueError("manifest raw payload count는 음수일 수 없습니다")
        if self.quality.dataset != self.dataset or self.quality.symbol != self.symbol:
            raise ValueError("manifest 품질 요약의 데이터셋 또는 심볼이 다릅니다")
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
        return (
            self.raw_payload_count > 0
            and self.quality.evidence_eligible
            and self.quality.unresolved_gap_count == 0
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
        quality = asdict(self.quality)
        quality["start"] = ensure_utc(self.quality.start).isoformat()
        quality["end"] = ensure_utc(self.quality.end).isoformat()
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
            "quality": quality,
        }
        if include_evidence_hash:
            evidence_hash = getattr(self, "evidence_hash", None)
            if evidence_hash is not None:
                payload["evidence_hash"] = evidence_hash
        return payload


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
    )
    if require_evidence_eligible:
        manifest.assert_evidence_eligible()
    return manifest
