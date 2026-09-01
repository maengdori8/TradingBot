from __future__ import annotations

"""Pareto discovery 원장의 멱등성·무결성·승격 차단 회귀 테스트."""

import json
from pathlib import Path

import pytest

from lab.pareto_trial_ledger import (
    LedgerIntegrityError,
    ParetoMetrics,
    ParetoTrialLedger,
    TrialConflictError,
    build_trial_id,
    compare_pareto,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _metrics(**changes: float) -> ParetoMetrics:
    """기본 비교 지표에 요청한 값만 덮어쓴다."""

    values = {
        "profit_factor": 1.2,
        "expectancy_r": 0.03,
        "net_r": 30.0,
        "max_drawdown_r": 20.0,
        "bootstrap_mdd_p95_r": 30.0,
        "trades_per_month": 20.0,
    }
    values.update(changes)
    return ParetoMetrics(**values)


def test_trial_id_is_canonical_across_mapping_order() -> None:
    """파라미터와 데이터 해시의 삽입 순서는 trial ID를 바꾸지 않아야 한다."""

    first = build_trial_id(
        "candidate",
        {"b": [2, 3], "a": 1},
        {"prices": HASH_A, "funding": HASH_B},
        HASH_A,
    )
    second = build_trial_id(
        "candidate",
        {"a": 1, "b": (2, 3)},
        {"funding": HASH_B, "prices": HASH_A},
        HASH_A,
    )

    assert first == second


def test_append_is_idempotent_and_conflicting_result_is_rejected(
    tmp_path: Path,
) -> None:
    """동일 trial 재호출은 한 줄만 남기고 다른 결과는 거부해야 한다."""

    ledger = ParetoTrialLedger(tmp_path / "trials.jsonl")
    arguments = {
        "trial_name": "candidate",
        "params": {"entry": 24},
        "data_hashes": {"prices": HASH_A},
        "code_hash": HASH_B,
    }

    first = ledger.append_success(**arguments, metrics=_metrics())
    duplicate = ledger.append_success(**arguments, metrics=_metrics())

    assert first.appended is True
    assert duplicate.appended is False
    assert len(ledger.read_records()) == 1
    with pytest.raises(TrialConflictError):
        ledger.append_success(
            **arguments,
            metrics=_metrics(expectancy_r=0.04),
        )


def test_failure_is_recorded_and_hash_tampering_is_detected(tmp_path: Path) -> None:
    """실패도 시도 수에 남고 기존 행 변경은 다음 읽기에서 탐지돼야 한다."""

    path = tmp_path / "trials.jsonl"
    ledger = ParetoTrialLedger(path)
    ledger.append_failure(
        trial_name="network-failure",
        params={"symbol": "BTC"},
        data_hashes={"prices": HASH_A},
        code_hash=HASH_B,
        error="funding coverage missing",
    )

    record = ledger.read_records()[0]
    assert record["outcome"] == "FAILED"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["error"] = "changed"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="record_hash"):
        ledger.read_records()


def test_pareto_comparison_requires_all_axes_and_never_promotes() -> None:
    """세 축 개선을 모두 보고해도 discovery 원장은 승격을 허용하지 않아야 한다."""

    incumbent = _metrics()
    candidate = _metrics(
        profit_factor=1.3,
        expectancy_r=0.04,
        net_r=40.0,
        max_drawdown_r=18.0,
        bootstrap_mdd_p95_r=28.0,
        trades_per_month=22.0,
    )

    comparison = compare_pareto(candidate, incumbent)

    assert comparison.dominates is True
    assert comparison.improves_all_axes is True
    assert comparison.promotion_allowed is False
    assert comparison.promotion_capability == "DISABLED_IN_DISCOVERY_LEDGER"
