from __future__ import annotations

"""다중 후보 탐색의 discovery trial을 변경 불가능한 JSONL 원장에 기록한다."""

import fcntl
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DISCOVERY_CLASSIFICATION = "DISCOVERY_ONLY_NOT_PREREGISTERED"
PROMOTION_CAPABILITY = "DISABLED_IN_DISCOVERY_LEDGER"
OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_FAILED = "FAILED"
_ZERO_HASH = "0" * 64


class LedgerIntegrityError(ValueError):
    """원장 구문·해시 체인·중복 식별자가 손상되었을 때 발생한다."""


class TrialConflictError(ValueError):
    """같은 trial ID에 서로 다른 결과를 기록하려 할 때 발생한다."""


@dataclass(frozen=True)
class ParetoMetrics:
    """수익·안전·빈도 세 축을 비교하기 위한 최소 성과 지표다."""

    profit_factor: float
    expectancy_r: float
    net_r: float
    max_drawdown_r: float
    bootstrap_mdd_p95_r: float
    trades_per_month: float

    def __post_init__(self) -> None:
        """모든 Pareto 지표가 유한하고 의미 있는 범위인지 검증한다."""

        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}은(는) 유한한 수여야 합니다.")
        if self.profit_factor < 0.0:
            raise ValueError("profit_factor는 0 이상이어야 합니다.")
        if self.max_drawdown_r < 0.0 or self.bootstrap_mdd_p95_r < 0.0:
            raise ValueError("drawdown 지표는 0 이상이어야 합니다.")
        if self.trades_per_month < 0.0:
            raise ValueError("trades_per_month는 0 이상이어야 합니다.")

    def to_dict(self) -> dict[str, float]:
        """JSON 직렬화에 안정적인 성과 지표 사전을 반환한다."""

        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ParetoMetrics:
        """일반 매핑을 검증된 Pareto 성과 지표로 변환한다."""

        required = {
            "profit_factor",
            "expectancy_r",
            "net_r",
            "max_drawdown_r",
            "bootstrap_mdd_p95_r",
            "trades_per_month",
        }
        missing = required.difference(value)
        extra = set(value).difference(required)
        if missing or extra:
            raise ValueError(
                f"Pareto 지표 필드 오류: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(**{name: float(value[name]) for name in sorted(required)})


@dataclass(frozen=True)
class ParetoComparison:
    """후보와 기준 전략의 세 축 Pareto 비교 결과다."""

    profit_no_worse: bool
    profit_improved: bool
    safety_no_worse: bool
    safety_improved: bool
    frequency_no_worse: bool
    frequency_improved: bool
    dominates: bool
    improves_all_axes: bool
    promotion_allowed: bool = False
    classification: str = DISCOVERY_CLASSIFICATION
    promotion_capability: str = PROMOTION_CAPABILITY

    def to_dict(self) -> dict[str, Any]:
        """Pareto 비교 결과를 JSON 직렬화 가능한 사전으로 반환한다."""

        return asdict(self)


@dataclass(frozen=True)
class AppendResult:
    """원장 append가 새 기록인지 멱등 중복인지 나타낸다."""

    trial_id: str
    appended: bool
    record: dict[str, Any]


def _normalize_json(value: Any) -> Any:
    """허용된 값을 결정론적 JSON 기본형으로 재귀 정규화한다."""

    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(
                "NaN 또는 무한대는 canonical JSON에 기록할 수 없습니다."
            )
        return 0.0 if normalized == 0.0 else normalized
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON 매핑 키는 문자열이어야 합니다.")
            normalized_mapping[key] = _normalize_json(item)
        return normalized_mapping
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(
        f"canonical JSON에서 지원하지 않는 타입입니다: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """동일 입력이 항상 동일 바이트열을 만드는 canonical JSON을 반환한다."""

    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    """한 파일의 내용을 스트리밍해 SHA256 해시를 반환한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Sequence[Path], root: Path | None = None) -> str:
    """상대경로와 파일 내용을 묶어 결정론적 코드 SHA256을 계산한다."""

    if not paths:
        raise ValueError("해시할 코드 파일이 하나 이상 필요합니다.")
    resolved_root = root.resolve() if root is not None else None
    entries: list[tuple[str, Path]] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved_root is not None:
            try:
                name = resolved.relative_to(resolved_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"코드 파일이 root 밖에 있습니다: {resolved}"
                ) from error
        else:
            name = resolved.name
        entries.append((name, resolved))
    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("코드 해시 입력 경로 이름이 중복됩니다.")

    digest = hashlib.sha256()
    for name, resolved in sorted(entries, key=lambda entry: entry[0]):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, field: str) -> str:
    """SHA256 문자열을 소문자 64자리 16진수로 검증한다."""

    normalized = value.lower()
    invalid_character = any(
        character not in "0123456789abcdef" for character in normalized
    )
    if len(normalized) != 64 or invalid_character:
        raise ValueError(f"{field}은(는) 64자리 SHA256 16진수여야 합니다.")
    return normalized


def canonical_data_hashes(data_hashes: Mapping[str, str]) -> dict[str, str]:
    """데이터 소스 이름과 SHA256 매핑을 검증하고 정렬한다."""

    if not data_hashes:
        raise ValueError(
            "data_hashes는 하나 이상의 데이터 소스를 포함해야 합니다."
        )
    canonical: dict[str, str] = {}
    for name, digest in data_hashes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "데이터 소스 이름은 비어 있지 않은 문자열이어야 합니다."
            )
        if not isinstance(digest, str):
            raise TypeError(f"{name} 데이터 해시는 문자열이어야 합니다.")
        canonical[name] = _validate_sha256(digest, f"data_hashes[{name!r}]")
    return dict(sorted(canonical.items()))


def build_trial_id(
    trial_name: str,
    params: Mapping[str, Any],
    data_hashes: Mapping[str, str],
    code_hash: str,
) -> str:
    """이름·canonical 파라미터·데이터·코드 해시로 고정 trial ID를 만든다."""

    if not trial_name.strip():
        raise ValueError("trial_name은 비어 있을 수 없습니다.")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "trial_name": trial_name,
        "params": _normalize_json(params),
        "data_hashes": canonical_data_hashes(data_hashes),
        "code_hash": _validate_sha256(code_hash, "code_hash"),
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def compare_pareto(candidate: ParetoMetrics, incumbent: ParetoMetrics) -> ParetoComparison:
    """수익·낙폭·월 거래빈도의 Pareto 우위를 판정한다."""

    profit_deltas = (
        candidate.profit_factor - incumbent.profit_factor,
        candidate.expectancy_r - incumbent.expectancy_r,
        candidate.net_r - incumbent.net_r,
    )
    safety_reductions = (
        incumbent.max_drawdown_r - candidate.max_drawdown_r,
        incumbent.bootstrap_mdd_p95_r - candidate.bootstrap_mdd_p95_r,
    )
    frequency_delta = candidate.trades_per_month - incumbent.trades_per_month

    profit_no_worse = all(delta >= 0.0 for delta in profit_deltas)
    profit_improved = profit_no_worse and any(delta > 0.0 for delta in profit_deltas)
    safety_no_worse = all(reduction >= 0.0 for reduction in safety_reductions)
    safety_improved = safety_no_worse and any(reduction > 0.0 for reduction in safety_reductions)
    frequency_no_worse = frequency_delta >= 0.0
    frequency_improved = frequency_delta > 0.0
    dominates = (
        profit_no_worse
        and safety_no_worse
        and frequency_no_worse
        and (profit_improved or safety_improved or frequency_improved)
    )
    improves_all_axes = profit_improved and safety_improved and frequency_improved
    return ParetoComparison(
        profit_no_worse=profit_no_worse,
        profit_improved=profit_improved,
        safety_no_worse=safety_no_worse,
        safety_improved=safety_improved,
        frequency_no_worse=frequency_no_worse,
        frequency_improved=frequency_improved,
        dominates=dominates,
        improves_all_axes=improves_all_axes,
    )


class ParetoTrialLedger:
    """파일 잠금과 해시 체인으로 보호되는 append-only discovery 원장이다."""

    def __init__(self, path: Path) -> None:
        """원장 경로만 보관하며 실제 파일 생성은 첫 append 시 수행한다."""

        self.path = path

    def read_records(self) -> list[dict[str, Any]]:
        """공유 잠금 아래 전체 원장을 읽고 해시 체인을 검증한다."""

        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_locked(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append_success(
        self,
        trial_name: str,
        params: Mapping[str, Any],
        data_hashes: Mapping[str, str],
        code_hash: str,
        metrics: ParetoMetrics,
        metadata: Mapping[str, Any] | None = None,
    ) -> AppendResult:
        """완료 trial을 기록하며 동일 요청은 중복 append하지 않는다."""

        return self._append(
            trial_name=trial_name,
            params=params,
            data_hashes=data_hashes,
            code_hash=code_hash,
            outcome=OUTCOME_COMPLETED,
            metrics=metrics.to_dict(),
            error=None,
            metadata=metadata,
        )

    def append_failure(
        self,
        trial_name: str,
        params: Mapping[str, Any],
        data_hashes: Mapping[str, str],
        code_hash: str,
        error: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> AppendResult:
        """실패 trial도 기록해 선택편향 없는 시도 수를 보존한다."""

        if not error.strip():
            raise ValueError("실패 trial에는 비어 있지 않은 error가 필요합니다.")
        return self._append(
            trial_name=trial_name,
            params=params,
            data_hashes=data_hashes,
            code_hash=code_hash,
            outcome=OUTCOME_FAILED,
            metrics=None,
            error=error,
            metadata=metadata,
        )

    def _append(
        self,
        trial_name: str,
        params: Mapping[str, Any],
        data_hashes: Mapping[str, str],
        code_hash: str,
        outcome: str,
        metrics: Mapping[str, Any] | None,
        error: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> AppendResult:
        """잠금 아래 trial을 추가하거나 멱등 중복을 반환한다."""

        normalized_params = _normalize_json(params)
        normalized_hashes = canonical_data_hashes(data_hashes)
        normalized_code_hash = _validate_sha256(code_hash, "code_hash")
        normalized_metrics = _normalize_json(metrics) if metrics is not None else None
        normalized_metadata = _normalize_json(metadata or {})
        trial_id = build_trial_id(
            trial_name,
            normalized_params,
            normalized_hashes,
            normalized_code_hash,
        )
        immutable_payload = {
            "schema_version": SCHEMA_VERSION,
            "trial_id": trial_id,
            "trial_name": trial_name,
            "classification": DISCOVERY_CLASSIFICATION,
            "promotion_capability": PROMOTION_CAPABILITY,
            "params": normalized_params,
            "data_hashes": normalized_hashes,
            "code_hash": normalized_code_hash,
            "outcome": outcome,
            "metrics": normalized_metrics,
            "error": error,
            "metadata": normalized_metadata,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_locked(handle)
                duplicate = next(
                    (record for record in records if record.get("trial_id") == trial_id),
                    None,
                )
                if duplicate is not None:
                    existing_payload = {
                        key: duplicate.get(key) for key in immutable_payload
                    }
                    if canonical_json(existing_payload) != canonical_json(immutable_payload):
                        message = (
                            f"trial_id={trial_id}에 서로 다른 결과가 "
                            "이미 기록되어 있습니다."
                        )
                        raise TrialConflictError(message)
                    logger.info("중복 trial을 멱등 처리했습니다: %s", trial_id)
                    return AppendResult(trial_id=trial_id, appended=False, record=duplicate)

                previous_hash = records[-1]["record_hash"] if records else _ZERO_HASH
                record = {
                    **immutable_payload,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "previous_record_hash": previous_hash,
                }
                record["record_hash"] = hashlib.sha256(
                    canonical_json(record).encode("utf-8")
                ).hexdigest()
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                logger.info("discovery trial을 append했습니다: %s", trial_id)
                return AppendResult(trial_id=trial_id, appended=True, record=record)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_locked(self, handle: TextIO) -> list[dict[str, Any]]:
        """잠긴 파일을 읽어 각 JSONL 레코드와 체인을 검증한다."""

        handle.seek(0)
        records: list[dict[str, Any]] = []
        previous_hash = _ZERO_HASH
        trial_ids: set[str] = set()
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith("\n"):
                raise LedgerIntegrityError(
                    f"{line_number}행이 완전한 JSONL 레코드가 아닙니다."
                )
            if not raw_line.strip():
                raise LedgerIntegrityError(f"{line_number}행이 비어 있습니다.")
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise LedgerIntegrityError(
                    f"{line_number}행 JSON이 손상되었습니다."
                ) from error
            if not isinstance(parsed, dict):
                raise LedgerIntegrityError(f"{line_number}행은 JSON 객체여야 합니다.")
            record = _normalize_json(parsed)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise LedgerIntegrityError(f"{line_number}행 schema_version이 다릅니다.")
            if record.get("classification") != DISCOVERY_CLASSIFICATION:
                raise LedgerIntegrityError(
                    f"{line_number}행 classification이 허용되지 않습니다."
                )
            if record.get("promotion_capability") != PROMOTION_CAPABILITY:
                raise LedgerIntegrityError(
                    f"{line_number}행이 승격 가능 상태를 포함합니다."
                )
            self._validate_record_contract(record, line_number)
            if record.get("previous_record_hash") != previous_hash:
                raise LedgerIntegrityError(
                    f"{line_number}행 이전 레코드 해시가 일치하지 않습니다."
                )
            record_hash = record.get("record_hash")
            if not isinstance(record_hash, str):
                raise LedgerIntegrityError(f"{line_number}행 record_hash가 없습니다.")
            _validate_sha256(record_hash, f"{line_number}행 record_hash")
            hash_payload = dict(record)
            del hash_payload["record_hash"]
            expected_hash = hashlib.sha256(
                canonical_json(hash_payload).encode("utf-8")
            ).hexdigest()
            if record_hash != expected_hash:
                raise LedgerIntegrityError(
                    f"{line_number}행 record_hash가 일치하지 않습니다."
                )
            trial_id = record.get("trial_id")
            if not isinstance(trial_id, str):
                raise LedgerIntegrityError(f"{line_number}행 trial_id가 없습니다.")
            _validate_sha256(trial_id, f"{line_number}행 trial_id")
            if trial_id in trial_ids:
                raise LedgerIntegrityError(f"{line_number}행 trial_id가 중복되었습니다.")
            trial_ids.add(trial_id)
            records.append(record)
            previous_hash = record_hash
        return records

    def _validate_record_contract(
        self,
        record: Mapping[str, Any],
        line_number: int,
    ) -> None:
        """레코드의 trial ID와 성공·실패 payload 계약을 검증한다."""

        trial_name = record.get("trial_name")
        params = record.get("params")
        data_hashes = record.get("data_hashes")
        code_hash = record.get("code_hash")
        if not isinstance(trial_name, str) or not isinstance(params, Mapping):
            raise LedgerIntegrityError(
                f"{line_number}행 trial 이름 또는 params 형식이 잘못되었습니다."
            )
        if not isinstance(data_hashes, Mapping) or not isinstance(code_hash, str):
            raise LedgerIntegrityError(
                f"{line_number}행 데이터 또는 코드 해시 형식이 잘못되었습니다."
            )
        try:
            expected_trial_id = build_trial_id(
                trial_name,
                params,
                data_hashes,
                code_hash,
            )
        except (TypeError, ValueError) as error:
            raise LedgerIntegrityError(
                f"{line_number}행 trial identity가 잘못되었습니다."
            ) from error
        if record.get("trial_id") != expected_trial_id:
            raise LedgerIntegrityError(
                f"{line_number}행 trial_id가 canonical identity와 다릅니다."
            )

        outcome = record.get("outcome")
        metrics = record.get("metrics")
        error_message = record.get("error")
        if outcome == OUTCOME_COMPLETED:
            if not isinstance(metrics, Mapping) or error_message is not None:
                raise LedgerIntegrityError(
                    f"{line_number}행 완료 trial payload가 잘못되었습니다."
                )
            try:
                ParetoMetrics.from_mapping(metrics)
            except (TypeError, ValueError) as error:
                raise LedgerIntegrityError(
                    f"{line_number}행 Pareto metrics가 잘못되었습니다."
                ) from error
        elif outcome == OUTCOME_FAILED:
            if metrics is not None or not isinstance(error_message, str):
                raise LedgerIntegrityError(
                    f"{line_number}행 실패 trial payload가 잘못되었습니다."
                )
            if not error_message.strip():
                raise LedgerIntegrityError(
                    f"{line_number}행 실패 error가 비어 있습니다."
                )
        else:
            raise LedgerIntegrityError(
                f"{line_number}행 outcome이 허용되지 않습니다."
            )
