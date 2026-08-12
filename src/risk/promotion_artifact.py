from __future__ import annotations

# 검증 증거와 실행 모드를 위변조 감지 해시로 결합하는 순수 계약.

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np

from src.risk.validation_gate import (
    DemoPromotionGate,
    DemoValidationEvidence,
    GateDecision,
    OfflineEvidenceReport,
    OfflinePromotionGate,
)

PromotionStage = Literal["offline", "demo"]
ActivationMode = Literal["demo", "live"]
PROMOTION_ARTIFACT_SCHEMA = "promotion-artifact/v1"
STRATEGY_ACTIVATION_SCHEMA = "strategy-activation/v1"
OFFLINE_EVIDENCE_METHODOLOGY = (
    "two-way-day-symbol-bootstrap+derived-regimes+dSR+"
    "CSCV-PBO+block-bootstrap-SPA/v2"
)
_HASH_NAMES = (
    "strategy_sha256",
    "code_sha256",
    "data_sha256",
    "hypothesis_sha256",
    "evidence_sha256",
)


def _canonical_json(value: object) -> str:
    """해시 계산에 사용할 정렬·공백 제거 JSON을 반환한다."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    """UTF-8 문자열의 SHA-256 헥스 다이제스트를 반환한다."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_hash(name: str, value: object, *, allow_empty: bool = False) -> str:
    """SHA-256 문자열을 정규 소문자 형식으로 검증한다."""
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name}은 64자 SHA-256 문자열이어야 합니다.")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name}은 소문자 SHA-256 헥스여야 합니다.")
    return value


def _validate_text(name: str, value: object) -> str:
    """비어 있지 않은 문자열 필드를 검증한다."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}은 비어 있지 않은 문자열이어야 합니다.")
    return value


def _parse_time(value: object) -> datetime:
    """ISO-8601 시각을 timezone-aware UTC로 파싱한다."""
    if not isinstance(value, str):
        raise ValueError("generated_at은 ISO-8601 문자열이어야 합니다.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at을 ISO-8601 시각으로 파싱할 수 없습니다.") from exc
    if parsed.tzinfo is None:
        raise ValueError("generated_at은 timezone-aware여야 합니다.")
    return parsed.astimezone(timezone.utc)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    """JSON 객체의 중복 키를 거부한다."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"중복 JSON 키가 있습니다: {key}")
        result[key] = value
    return result


def _load_json(payload: str) -> Mapping[str, object]:
    """중복 키와 비객체 최상위 JSON을 거부하여 파싱한다."""
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("유효한 JSON이 아닙니다.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("최상위 JSON은 객체여야 합니다.")
    return parsed


def _primitive(value: object) -> str | int | float | bool:
    """기준 값을 유한한 JSON 기본형으로 검증한다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("기준 값과 임계치는 유한한 JSON 기본형이어야 합니다.")


def _freeze_criteria(
    criteria: Mapping[str, Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
    """승급 기준을 검증하고 2단계 불변 맵으로 변환한다."""
    if not criteria:
        raise ValueError("승급 기준은 비어 있을 수 없습니다.")
    frozen: dict[str, Mapping[str, object]] = {}
    expected_fields = {"name", "passed", "value", "threshold"}
    for key, raw in sorted(criteria.items()):
        criterion_id = _validate_text("criterion_id", key)
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError(f"{criterion_id} 기준 스키마가 올바르지 않습니다.")
        if not isinstance(raw["passed"], bool):
            raise ValueError(f"{criterion_id}.passed는 bool이어야 합니다.")
        frozen[criterion_id] = MappingProxyType(
            {
                "name": _validate_text("criterion.name", raw["name"]),
                "passed": raw["passed"],
                "value": _primitive(raw["value"]),
                "threshold": _primitive(raw["threshold"]),
            }
        )
    return MappingProxyType(frozen)


def _decision_criteria(
    decision: GateDecision,
) -> Mapping[str, Mapping[str, object]]:
    """게이트 판정 기준을 아티팩트 스키마로 변환한다."""
    return _freeze_criteria(
        {
            key: {
                "name": criterion.name,
                "passed": criterion.passed,
                "value": criterion.value,
                "threshold": criterion.threshold,
            }
            for key, criterion in decision.criteria.items()
        }
    )


@dataclass(frozen=True)
class PromotionArtifact:
    """전략·코드·데이터·가설·증거 계보를 묶는 불변 승급 아티팩트."""

    schema_version: str
    stage: PromotionStage
    strategy_id: str
    strategy_version: str
    passed: bool
    criteria: Mapping[str, Mapping[str, object]]
    generated_at: datetime
    strategy_sha256: str
    code_sha256: str
    data_sha256: str
    hypothesis_sha256: str
    evidence_sha256: str
    upstream_artifact_sha256: str = ""
    _verified: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """아티팩트 스키마와 종합 판정을 fail-closed로 검증한다."""
        if self.schema_version != PROMOTION_ARTIFACT_SCHEMA:
            raise ValueError("지원하지 않는 PromotionArtifact 스키마입니다.")
        if self.stage not in ("offline", "demo"):
            raise ValueError("stage는 offline 또는 demo여야 합니다.")
        _validate_text("strategy_id", self.strategy_id)
        _validate_text("strategy_version", self.strategy_version)
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at은 timezone-aware여야 합니다.")
        for name in _HASH_NAMES:
            _validate_hash(name, getattr(self, name))
        _validate_hash(
            "upstream_artifact_sha256",
            self.upstream_artifact_sha256,
            allow_empty=self.stage == "offline",
        )
        if self.stage == "offline" and self.upstream_artifact_sha256:
            raise ValueError("offline 아티팩트는 상위 아티팩트를 가질 수 없습니다.")
        frozen = _freeze_criteria(self.criteria)
        calculated_pass = all(bool(item["passed"]) for item in frozen.values())
        if self.passed is not calculated_pass:
            raise ValueError("passed가 개별 기준 판정과 일치하지 않습니다.")
        object.__setattr__(self, "criteria", frozen)
        object.__setattr__(self, "generated_at", self.generated_at.astimezone(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        """파일 형식과 무관한 정규 아티팩트 딕셔너리를 반환한다."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "passed": self.passed,
            "criteria": {key: dict(value) for key, value in self.criteria.items()},
            "generated_at": self.generated_at.isoformat(),
            "strategy_sha256": self.strategy_sha256,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "hypothesis_sha256": self.hypothesis_sha256,
            "evidence_sha256": self.evidence_sha256,
            "upstream_artifact_sha256": self.upstream_artifact_sha256,
        }

    def to_json(self) -> str:
        """키 순서와 공백을 고정한 canonical JSON을 반환한다."""
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        """canonical JSON의 SHA-256을 반환한다."""
        return _sha256(self.to_json())

    @property
    def verified(self) -> bool:
        """게이트 빌더 또는 외부 고정 해시로 검증됐는지 반환한다."""
        return self._verified

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        expected_sha256: str | None = None,
    ) -> PromotionArtifact:
        """스키마가 엄격한 딕셔너리를 파싱하고 선택적 고정 해시를 확인한다."""
        expected_fields = {
            "schema_version",
            "stage",
            "strategy_id",
            "strategy_version",
            "passed",
            "criteria",
            "generated_at",
            "strategy_sha256",
            "code_sha256",
            "data_sha256",
            "hypothesis_sha256",
            "evidence_sha256",
            "upstream_artifact_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("PromotionArtifact 필드 집합이 스키마와 다릅니다.")
        stage = payload["stage"]
        if stage not in ("offline", "demo"):
            raise ValueError("stage는 offline 또는 demo여야 합니다.")
        passed = payload["passed"]
        if not isinstance(passed, bool):
            raise ValueError("passed는 bool이어야 합니다.")
        criteria = payload["criteria"]
        if not isinstance(criteria, Mapping):
            raise ValueError("criteria는 JSON 객체여야 합니다.")
        artifact = cls(
            schema_version=_validate_text("schema_version", payload["schema_version"]),
            stage=stage,
            strategy_id=_validate_text("strategy_id", payload["strategy_id"]),
            strategy_version=_validate_text(
                "strategy_version", payload["strategy_version"]
            ),
            passed=passed,
            criteria=criteria,
            generated_at=_parse_time(payload["generated_at"]),
            strategy_sha256=_validate_hash(
                "strategy_sha256", payload["strategy_sha256"]
            ),
            code_sha256=_validate_hash("code_sha256", payload["code_sha256"]),
            data_sha256=_validate_hash("data_sha256", payload["data_sha256"]),
            hypothesis_sha256=_validate_hash(
                "hypothesis_sha256", payload["hypothesis_sha256"]
            ),
            evidence_sha256=_validate_hash(
                "evidence_sha256", payload["evidence_sha256"]
            ),
            upstream_artifact_sha256=_validate_hash(
                "upstream_artifact_sha256",
                payload["upstream_artifact_sha256"],
                allow_empty=stage == "offline",
            ),
        )
        if expected_sha256 is not None:
            expected = _validate_hash("expected_sha256", expected_sha256)
            if artifact.sha256 != expected:
                raise ValueError("PromotionArtifact 고정 해시가 일치하지 않습니다.")
            object.__setattr__(artifact, "_verified", True)
        return artifact

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        expected_sha256: str | None = None,
    ) -> PromotionArtifact:
        """중복 키를 거부하는 JSON 파서로 아티팩트를 복원한다."""
        return cls.from_dict(_load_json(payload), expected_sha256=expected_sha256)


@dataclass(frozen=True)
class StrategyActivation:
    """승급 아티팩트와 허용 모드를 해시로 결합한 불변 활성화 계약."""

    schema_version: str
    promotion_stage: PromotionStage
    strategy_id: str
    strategy_version: str
    allowed_modes: tuple[ActivationMode, ...]
    promotion_artifact_sha256: str
    strategy_sha256: str
    code_sha256: str
    data_sha256: str
    hypothesis_sha256: str
    evidence_sha256: str
    generated_at: datetime
    _verified: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """승급 단계와 실행 모드의 유일한 허용 전이를 검증한다."""
        if self.schema_version != STRATEGY_ACTIVATION_SCHEMA:
            raise ValueError("지원하지 않는 StrategyActivation 스키마입니다.")
        _validate_text("strategy_id", self.strategy_id)
        _validate_text("strategy_version", self.strategy_version)
        expected_modes: tuple[ActivationMode, ...] = (
            ("demo",) if self.promotion_stage == "offline" else ("live",)
        )
        if self.promotion_stage not in ("offline", "demo"):
            raise ValueError("promotion_stage는 offline 또는 demo여야 합니다.")
        if self.allowed_modes != expected_modes:
            raise ValueError("오프라인 통과는 demo만, demo 통과는 live만 허용합니다.")
        _validate_hash("promotion_artifact_sha256", self.promotion_artifact_sha256)
        for name in _HASH_NAMES:
            _validate_hash(name, getattr(self, name))
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at은 timezone-aware여야 합니다.")
        object.__setattr__(self, "generated_at", self.generated_at.astimezone(timezone.utc))

    @classmethod
    def from_promotion_artifact(
        cls,
        artifact: PromotionArtifact,
        *,
        generated_at: datetime | None = None,
    ) -> StrategyActivation:
        """검증된 승급 아티팩트에서 다음 단계 활성화 계약을 만든다."""
        if not artifact.verified or not artifact.passed:
            raise PermissionError("검증되지 않았거나 미통과한 아티팩트는 활성화할 수 없습니다.")
        report_time = generated_at or datetime.now(timezone.utc)
        if report_time.tzinfo is None:
            raise ValueError("generated_at은 timezone-aware여야 합니다.")
        modes: tuple[ActivationMode, ...] = (
            ("demo",) if artifact.stage == "offline" else ("live",)
        )
        activation = cls(
            schema_version=STRATEGY_ACTIVATION_SCHEMA,
            promotion_stage=artifact.stage,
            strategy_id=artifact.strategy_id,
            strategy_version=artifact.strategy_version,
            allowed_modes=modes,
            promotion_artifact_sha256=artifact.sha256,
            strategy_sha256=artifact.strategy_sha256,
            code_sha256=artifact.code_sha256,
            data_sha256=artifact.data_sha256,
            hypothesis_sha256=artifact.hypothesis_sha256,
            evidence_sha256=artifact.evidence_sha256,
            generated_at=report_time.astimezone(timezone.utc),
        )
        object.__setattr__(activation, "_verified", True)
        return activation

    def to_dict(self) -> dict[str, object]:
        """활성화 계약의 정규 딕셔너리를 반환한다."""
        return {
            "schema_version": self.schema_version,
            "promotion_stage": self.promotion_stage,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "allowed_modes": list(self.allowed_modes),
            "promotion_artifact_sha256": self.promotion_artifact_sha256,
            "strategy_sha256": self.strategy_sha256,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "hypothesis_sha256": self.hypothesis_sha256,
            "evidence_sha256": self.evidence_sha256,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self) -> str:
        """키 순서와 공백을 고정한 canonical JSON을 반환한다."""
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        """canonical JSON의 SHA-256을 반환한다."""
        return _sha256(self.to_json())

    def assert_mode_allowed(
        self,
        mode: str | object,
        *,
        strategy_version: str,
        code_sha256: str,
        data_sha256: str,
        hypothesis_sha256: str,
        strategy_sha256: str | None = None,
        promotion_artifact_sha256: str | None = None,
    ) -> None:
        """런타임 모드·버전·계보 해시가 고정 계약과 다르면 실행을 거부한다."""
        if not self._verified:
            raise PermissionError("외부 고정 해시와 승급 아티팩트가 검증되지 않았습니다.")
        normalized_mode = getattr(mode, "value", mode)
        mismatches: list[str] = []
        if normalized_mode not in self.allowed_modes:
            mismatches.append("mode")
        if strategy_version != self.strategy_version:
            mismatches.append("strategy_version")
        bindings = {
            "code_sha256": code_sha256,
            "data_sha256": data_sha256,
            "hypothesis_sha256": hypothesis_sha256,
        }
        if strategy_sha256 is not None:
            bindings["strategy_sha256"] = strategy_sha256
        if promotion_artifact_sha256 is not None:
            bindings["promotion_artifact_sha256"] = promotion_artifact_sha256
        for name, value in bindings.items():
            _validate_hash(name, value)
            if value != getattr(self, name):
                mismatches.append(name)
        if mismatches:
            raise PermissionError(
                "활성화 계약 불일치: " + ",".join(sorted(mismatches))
            )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        expected_sha256: str | None = None,
        promotion_artifact: PromotionArtifact | None = None,
    ) -> StrategyActivation:
        """활성화 딕셔너리를 파싱하고 외부 해시와 승급 계보를 검증한다."""
        expected_fields = {
            "schema_version",
            "promotion_stage",
            "strategy_id",
            "strategy_version",
            "allowed_modes",
            "promotion_artifact_sha256",
            "strategy_sha256",
            "code_sha256",
            "data_sha256",
            "hypothesis_sha256",
            "evidence_sha256",
            "generated_at",
        }
        if set(payload) != expected_fields:
            raise ValueError("StrategyActivation 필드 집합이 스키마와 다릅니다.")
        stage = payload["promotion_stage"]
        if stage not in ("offline", "demo"):
            raise ValueError("promotion_stage는 offline 또는 demo여야 합니다.")
        raw_modes = payload["allowed_modes"]
        if not isinstance(raw_modes, list) or not all(
            isinstance(item, str) for item in raw_modes
        ):
            raise ValueError("allowed_modes는 문자열 JSON 배열이어야 합니다.")
        activation = cls(
            schema_version=_validate_text("schema_version", payload["schema_version"]),
            promotion_stage=stage,
            strategy_id=_validate_text("strategy_id", payload["strategy_id"]),
            strategy_version=_validate_text(
                "strategy_version", payload["strategy_version"]
            ),
            allowed_modes=tuple(raw_modes),  # type: ignore[arg-type]
            promotion_artifact_sha256=_validate_hash(
                "promotion_artifact_sha256",
                payload["promotion_artifact_sha256"],
            ),
            strategy_sha256=_validate_hash(
                "strategy_sha256", payload["strategy_sha256"]
            ),
            code_sha256=_validate_hash("code_sha256", payload["code_sha256"]),
            data_sha256=_validate_hash("data_sha256", payload["data_sha256"]),
            hypothesis_sha256=_validate_hash(
                "hypothesis_sha256", payload["hypothesis_sha256"]
            ),
            evidence_sha256=_validate_hash(
                "evidence_sha256", payload["evidence_sha256"]
            ),
            generated_at=_parse_time(payload["generated_at"]),
        )
        hash_verified = False
        if expected_sha256 is not None:
            expected = _validate_hash("expected_sha256", expected_sha256)
            if activation.sha256 != expected:
                raise ValueError("StrategyActivation 고정 해시가 일치하지 않습니다.")
            hash_verified = True
        artifact_verified = False
        if promotion_artifact is not None:
            artifact_verified = _activation_matches_artifact(
                activation,
                promotion_artifact,
            )
        object.__setattr__(
            activation,
            "_verified",
            hash_verified and artifact_verified,
        )
        return activation

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        expected_sha256: str | None = None,
        promotion_artifact: PromotionArtifact | None = None,
    ) -> StrategyActivation:
        """중복 키를 거부하는 JSON 파서로 활성화 계약을 복원한다."""
        return cls.from_dict(
            _load_json(payload),
            expected_sha256=expected_sha256,
            promotion_artifact=promotion_artifact,
        )


def _activation_matches_artifact(
    activation: StrategyActivation,
    artifact: PromotionArtifact,
) -> bool:
    """활성화 계약의 모든 계보 필드가 검증된 아티팩트와 일치하는지 확인한다."""
    if not artifact.verified or not artifact.passed:
        raise PermissionError("승급 아티팩트가 검증되지 않았거나 미통과했습니다.")
    expected = {
        "promotion_stage": artifact.stage,
        "strategy_id": artifact.strategy_id,
        "strategy_version": artifact.strategy_version,
        "promotion_artifact_sha256": artifact.sha256,
        "strategy_sha256": artifact.strategy_sha256,
        "code_sha256": artifact.code_sha256,
        "data_sha256": artifact.data_sha256,
        "hypothesis_sha256": artifact.hypothesis_sha256,
        "evidence_sha256": artifact.evidence_sha256,
    }
    mismatches = [
        name for name, value in expected.items() if getattr(activation, name) != value
    ]
    if mismatches:
        raise ValueError(
            "StrategyActivation과 PromotionArtifact 계보 불일치: "
            + ",".join(sorted(mismatches))
        )
    return True


def build_offline_promotion_artifact(
    report: OfflineEvidenceReport,
    *,
    strategy_sha256: str,
    code_sha256: str,
    data_sha256: str,
    hypothesis_sha256: str,
    gate: OfflinePromotionGate | None = None,
    generated_at: datetime | None = None,
) -> PromotionArtifact:
    """원시 레코드 v2 리포트를 재판정해 demo 후보 아티팩트를 만든다."""
    if report.methodology != OFFLINE_EVIDENCE_METHODOLOGY:
        raise PermissionError("수동 라벨 또는 레거시 리포트는 승급 증거가 아닙니다.")
    if report.evidence.hypothesis_configs < 2:
        raise PermissionError("단일 후보 증거로는 승급할 수 없습니다.")
    decision = (gate or OfflinePromotionGate()).evaluate(report.evidence)
    return _artifact_from_decision(
        decision,
        strategy_sha256=strategy_sha256,
        code_sha256=code_sha256,
        data_sha256=data_sha256,
        hypothesis_sha256=hypothesis_sha256,
        evidence_sha256=report.sha256,
        generated_at=generated_at,
    )


def build_demo_promotion_artifact(
    evidence: DemoValidationEvidence,
    *,
    offline_artifact: PromotionArtifact,
    strategy_sha256: str,
    code_sha256: str,
    data_sha256: str,
    hypothesis_sha256: str,
    raw_event_sha256: str,
    gate: DemoPromotionGate | None = None,
    generated_at: datetime | None = None,
) -> PromotionArtifact:
    """오프라인 통과 계보와 원시 데모 이벤트를 재판정해 live 후보를 만든다."""
    if (
        not offline_artifact.verified
        or not offline_artifact.passed
        or offline_artifact.stage != "offline"
    ):
        raise PermissionError("검증된 offline 통과 아티팩트가 필요합니다.")
    bindings = {
        "strategy_sha256": strategy_sha256,
        "code_sha256": code_sha256,
        "hypothesis_sha256": hypothesis_sha256,
    }
    for name, value in bindings.items():
        _validate_hash(name, value)
        if value != getattr(offline_artifact, name):
            raise PermissionError(f"demo 검증 계보가 offline과 다릅니다: {name}")
    if evidence.strategy_id != offline_artifact.strategy_id or (
        evidence.strategy_version != offline_artifact.strategy_version
    ):
        raise PermissionError("demo 전략 ID/버전이 offline 아티팩트와 다릅니다.")
    _validate_hash("data_sha256", data_sha256)
    event_hash = _validate_hash("raw_event_sha256", raw_event_sha256)
    evidence_payload = {
        "demo_evidence": evidence.__dict__,
        "raw_event_sha256": event_hash,
        "data_sha256": data_sha256,
        "code_sha256": code_sha256,
        "hypothesis_sha256": hypothesis_sha256,
        "offline_artifact_sha256": offline_artifact.sha256,
    }
    decision = (gate or DemoPromotionGate()).evaluate(evidence)
    return _artifact_from_decision(
        decision,
        strategy_sha256=strategy_sha256,
        code_sha256=code_sha256,
        data_sha256=data_sha256,
        hypothesis_sha256=hypothesis_sha256,
        evidence_sha256=_sha256(_canonical_json(evidence_payload)),
        upstream_artifact_sha256=offline_artifact.sha256,
        generated_at=generated_at,
    )


def _artifact_from_decision(
    decision: GateDecision,
    *,
    strategy_sha256: str,
    code_sha256: str,
    data_sha256: str,
    hypothesis_sha256: str,
    evidence_sha256: str,
    upstream_artifact_sha256: str = "",
    generated_at: datetime | None = None,
) -> PromotionArtifact:
    """게이트 결과와 계보 해시를 검증된 아티팩트로 결합한다."""
    if decision.stage not in ("offline", "demo"):
        raise ValueError("승급 아티팩트는 offline/demo 판정만 지원합니다.")
    report_time = generated_at or datetime.now(timezone.utc)
    if report_time.tzinfo is None:
        raise ValueError("generated_at은 timezone-aware여야 합니다.")
    artifact = PromotionArtifact(
        schema_version=PROMOTION_ARTIFACT_SCHEMA,
        stage=decision.stage,
        strategy_id=decision.strategy_id,
        strategy_version=decision.strategy_version,
        passed=decision.passed,
        criteria=_decision_criteria(decision),
        generated_at=report_time.astimezone(timezone.utc),
        strategy_sha256=_validate_hash("strategy_sha256", strategy_sha256),
        code_sha256=_validate_hash("code_sha256", code_sha256),
        data_sha256=_validate_hash("data_sha256", data_sha256),
        hypothesis_sha256=_validate_hash("hypothesis_sha256", hypothesis_sha256),
        evidence_sha256=_validate_hash("evidence_sha256", evidence_sha256),
        upstream_artifact_sha256=upstream_artifact_sha256,
    )
    object.__setattr__(artifact, "_verified", True)
    return artifact
