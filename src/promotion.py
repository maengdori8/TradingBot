from __future__ import annotations

"""검증 아티팩트를 다음 실행 단계에 연결하는 fail-closed 오케스트레이터."""

import argparse
import hashlib
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.exchange.contracts import TradingMode
from src.exchange.order_executor import BybitOrderExecutor
from src.risk.promotion_artifact import PromotionArtifact, StrategyActivation
from src.strategy.evidence_decision import StrategyTradeIntent

logger = logging.getLogger(__name__)

ARTIFACT_HASH_ENV = "PROMOTION_ARTIFACT_SHA256"
ACTIVATION_HASH_ENV = "STRATEGY_ACTIVATION_SHA256"
_MAX_CONTRACT_BYTES = 2 * 1024 * 1024


def _sha256_bytes(payload: bytes) -> str:
    """바이트열의 SHA-256을 반환한다."""
    return hashlib.sha256(payload).hexdigest()


def _required_hash(value: str | None, environment_name: str) -> str:
    """명시값 또는 환경변수에서 소문자 SHA-256을 읽는다."""
    selected = value or os.getenv(environment_name)
    if not isinstance(selected, str):
        raise PermissionError(f"고정 해시가 필요합니다: {environment_name}")
    normalized = selected.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{environment_name}은 SHA-256 형식이어야 합니다")
    return normalized


def _read_regular_file(path: Path) -> bytes:
    """심볼릭 링크와 검증 중 교체를 거부하고 정규 파일을 읽는다."""
    try:
        initial = os.lstat(path)
        if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"정규 파일만 허용됩니다: {path}")
        if initial.st_size > _MAX_CONTRACT_BYTES:
            raise ValueError(f"승급 계약 파일이 너무 큽니다: {path}")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (initial.st_dev, initial.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError(f"검증 중 파일이 교체됐습니다: {path}")
            payload = handle.read(_MAX_CONTRACT_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"승급 계약 파일을 읽을 수 없습니다: {path}") from exc
    if len(payload) > _MAX_CONTRACT_BYTES:
        raise ValueError(f"승급 계약 파일이 너무 큽니다: {path}")
    return payload


def _decode_contract(payload: bytes, path: Path) -> str:
    """승급 계약을 엄격한 UTF-8 문자열로 변환한다."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"승급 계약은 UTF-8이어야 합니다: {path}") from exc


@dataclass(frozen=True)
class VerifiedPromotion:
    """외부 고정 해시와 계보 검증을 마친 실행 권한."""

    mode: TradingMode
    artifact: PromotionArtifact
    activation: StrategyActivation
    artifact_path: Path
    activation_path: Path

    def authorize_intent(self, intent: StrategyTradeIntent) -> None:
        """주문 의도가 승인 후보·버전·가족과 다르면 거부한다."""
        mismatches: list[str] = []
        if intent.candidate_id != self.activation.strategy_id:
            mismatches.append("candidate_id")
        if intent.context.strategy_version != self.activation.strategy_version:
            mismatches.append("strategy_version")
        if not intent.context.run_id.strip():
            mismatches.append("run_id")
        if self.mode is TradingMode.LIVE and self.artifact.stage != "demo":
            mismatches.append("demo_gate")
        if mismatches:
            raise PermissionError(
                "승급 계약과 주문 의도가 다릅니다: " + ",".join(mismatches)
            )

    def create_executor(self, *, db_path: Path | None = None) -> BybitOrderExecutor:
        """검증된 모드의 Bybit 실행기를 만들되 주문은 제출하지 않는다."""
        kwargs: dict[str, object] = {
            "mode": self.mode,
            "db_path": db_path,
        }
        if self.mode is TradingMode.LIVE:
            kwargs.update(
                {
                    "live_approval_token": os.getenv(
                        "LIVE_TRADING_APPROVAL_TOKEN"
                    ),
                    "validation_report_hash": self.artifact.sha256,
                    "validation_report_path": self.artifact_path,
                }
            )
        return BybitOrderExecutor(**kwargs)


def load_verified_promotion(
    *,
    mode: TradingMode | str,
    artifact_path: Path | str,
    activation_path: Path | str,
    strategy_version: str,
    code_sha256: str,
    data_sha256: str,
    hypothesis_sha256: str,
    strategy_sha256: str,
    artifact_sha256: str | None = None,
    activation_sha256: str | None = None,
) -> VerifiedPromotion:
    """두 계약과 모든 런타임 계보가 맞을 때만 실행 권한을 반환한다."""
    trading_mode = TradingMode(mode)
    if trading_mode is TradingMode.PAPER:
        raise ValueError("paper는 승급 계약 대신 PaperEngine을 사용합니다")
    artifact_file = Path(os.path.abspath(Path(artifact_path).expanduser()))
    activation_file = Path(os.path.abspath(Path(activation_path).expanduser()))
    expected_artifact = _required_hash(artifact_sha256, ARTIFACT_HASH_ENV)
    expected_activation = _required_hash(activation_sha256, ACTIVATION_HASH_ENV)
    artifact_bytes = _read_regular_file(artifact_file)
    activation_bytes = _read_regular_file(activation_file)
    artifact = PromotionArtifact.from_json(
        _decode_contract(artifact_bytes, artifact_file),
        expected_sha256=expected_artifact,
    )
    activation = StrategyActivation.from_json(
        _decode_contract(activation_bytes, activation_file),
        expected_sha256=expected_activation,
        promotion_artifact=artifact,
    )
    # Live 실행기는 원본 파일의 SHA-256도 검사하므로 canonical 바이트만 허용한다.
    if artifact_bytes != artifact.to_json().encode("utf-8"):
        raise ValueError("PromotionArtifact 파일은 canonical JSON이어야 합니다")
    if activation_bytes != activation.to_json().encode("utf-8"):
        raise ValueError("StrategyActivation 파일은 canonical JSON이어야 합니다")
    if _sha256_bytes(artifact_bytes) != artifact.sha256:
        raise ValueError("PromotionArtifact 원본 해시가 canonical 해시와 다릅니다")
    if _sha256_bytes(activation_bytes) != activation.sha256:
        raise ValueError("StrategyActivation 원본 해시가 canonical 해시와 다릅니다")
    activation.assert_mode_allowed(
        trading_mode,
        strategy_version=strategy_version,
        code_sha256=code_sha256,
        data_sha256=data_sha256,
        hypothesis_sha256=hypothesis_sha256,
        strategy_sha256=strategy_sha256,
        promotion_artifact_sha256=artifact.sha256,
    )
    return VerifiedPromotion(
        mode=trading_mode,
        artifact=artifact,
        activation=activation,
        artifact_path=artifact_file,
        activation_path=activation_file,
    )


def write_next_activation(
    *,
    artifact_path: Path | str,
    output_path: Path | str,
    artifact_sha256: str | None = None,
) -> StrategyActivation:
    """통과한 승급 아티팩트에서 다음 단계 활성화 계약을 원자 저장한다."""
    source = Path(os.path.abspath(Path(artifact_path).expanduser()))
    expected = _required_hash(artifact_sha256, ARTIFACT_HASH_ENV)
    artifact = PromotionArtifact.from_json(
        _decode_contract(_read_regular_file(source), source),
        expected_sha256=expected,
    )
    if not artifact.passed:
        raise PermissionError("미통과 전략은 활성화 파일을 만들 수 없습니다")
    activation = StrategyActivation.from_promotion_artifact(artifact)
    destination = Path(os.path.abspath(Path(output_path).expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(activation.to_json(), encoding="utf-8")
    temporary.replace(destination)
    return activation


def _parser() -> argparse.ArgumentParser:
    """승급 계약 CLI parser를 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    activate = subparsers.add_parser("activate", help="다음 단계 활성화 계약 생성")
    activate.add_argument("--artifact", type=Path, required=True)
    activate.add_argument("--artifact-sha256")
    activate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="실행 전 계보·모드 검증")
    verify.add_argument("--mode", choices=("demo", "live"), required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--activation", type=Path, required=True)
    verify.add_argument("--artifact-sha256")
    verify.add_argument("--activation-sha256")
    verify.add_argument("--strategy-version", required=True)
    verify.add_argument("--strategy-sha256", required=True)
    verify.add_argument("--code-sha256", required=True)
    verify.add_argument("--data-sha256", required=True)
    verify.add_argument("--hypothesis-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """활성화 계약을 만들거나 실행 전 검증만 수행한다."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "activate":
        activation = write_next_activation(
            artifact_path=args.artifact,
            output_path=args.output,
            artifact_sha256=args.artifact_sha256,
        )
        logger.info(
            "활성화 계약 생성: mode=%s strategy=%s sha256=%s",
            activation.allowed_modes[0],
            activation.strategy_version,
            activation.sha256,
        )
        return 0
    verified = load_verified_promotion(
        mode=args.mode,
        artifact_path=args.artifact,
        activation_path=args.activation,
        strategy_version=args.strategy_version,
        code_sha256=args.code_sha256,
        data_sha256=args.data_sha256,
        hypothesis_sha256=args.hypothesis_sha256,
        strategy_sha256=args.strategy_sha256,
        artifact_sha256=args.artifact_sha256,
        activation_sha256=args.activation_sha256,
    )
    logger.info(
        "승급 실행 검증 통과: mode=%s strategy=%s activation=%s",
        verified.mode.value,
        verified.activation.strategy_version,
        verified.activation.sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
