"""
공용 설정 로더 — config.yaml(불변) 위에 학습 오버레이를 화이트리스트 키만 병합한다.

config.yaml은 사람이 소유하며 자동으로 절대 수정하지 않는다. 자동 튜너(learner)는
logs/learned_params.yaml 오버레이에만 기록하고, 이 로더가 로드 시점에 허용된 키만
얕게 덮어쓴다. 사용자가 config.yaml을 손수정하면(baseline 드리프트) 오버레이는 무효화된다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
OVERLAY_PATH = ROOT / "logs" / "learned_params.yaml"

# 하드 가드레일 (학습이 쓰는 값과 로더가 읽는 값 모두 이 범위 — 단일 출처)
RISK_PCT_MIN, RISK_PCT_MAX = 0.002, 0.010
MIN_SCORE_MIN, MIN_SCORE_MAX = 65, 85
MIN_RR_MIN, MIN_RR_MAX = 1.5, 3.0


def _in_range(v: float, lo: float, hi: float) -> bool:
    """값이 [lo, hi] 범위 안인지."""
    return isinstance(v, (int, float)) and lo <= v <= hi


def _load_yaml(path: Path) -> dict:
    """YAML 파일을 dict로 로드한다 (빈 파일이면 {})."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_raw_config() -> dict:
    """오버레이를 병합하지 않은 config.yaml 원본을 로드한다 (baseline 기준값용)."""
    return _load_yaml(CONFIG_PATH)


def load_config(overlay_path: Path | None = None) -> dict:
    """config.yaml 로드 후 학습 오버레이를 화이트리스트 키만 병합해 반환한다.

    Args:
        overlay_path: 오버레이 경로 (테스트용, 기본 logs/learned_params.yaml)

    Returns:
        병합된 설정 dict
    """
    cfg = _load_yaml(CONFIG_PATH)
    overlay = _read_overlay(overlay_path or OVERLAY_PATH)
    if overlay:
        _apply_overlay(cfg, overlay)
    return cfg


def _read_overlay(path: Path) -> dict | None:
    """오버레이 파일을 읽는다. 없거나 손상 시 None."""
    if not path.exists():
        return None
    try:
        data = _load_yaml(path)
    except (OSError, yaml.YAMLError) as e:
        logger.warning("학습 오버레이 로드 실패(무시): %s", e)
        return None
    return data if isinstance(data, dict) else None


def _baseline_ok(cfg: dict, overlay: dict) -> bool:
    """오버레이 baseline이 현재 config.yaml과 일치하는지 확인한다.

    사용자가 config.yaml을 손수정했다면 baseline이 어긋나므로 오버레이를 무효화한다.
    """
    baseline = overlay.get("baseline", {})
    if not baseline:
        return True  # baseline 미기록 오버레이는 일단 허용 (구버전 호환)

    # scan.min_score
    if "scan.min_score" in baseline:
        cur = cfg.get("scan", {}).get("min_score")
        if cur != baseline["scan.min_score"]:
            return False
    # risk.min_rr_ratio
    if "risk.min_rr_ratio" in baseline:
        cur = cfg.get("risk", {}).get("min_rr_ratio")
        if cur != baseline["risk.min_rr_ratio"]:
            return False
    # risk.risk_tiers (min_score별 risk_pct)
    if "risk.risk_tiers" in baseline:
        cur_tiers = {
            t.get("min_score"): t.get("risk_pct")
            for t in cfg.get("risk", {}).get("risk_tiers", [])
        }
        for bt in baseline["risk.risk_tiers"]:
            if cur_tiers.get(bt.get("min_score")) != bt.get("risk_pct"):
                return False
    return True


def _apply_overlay(cfg: dict, overlay: dict) -> None:
    """오버레이의 values를 화이트리스트 키만 cfg에 병합한다 (in-place)."""
    if not _baseline_ok(cfg, overlay):
        logger.warning(
            "config.yaml이 오버레이 생성 이후 손수정됨 — 학습 오버레이 무효화"
        )
        return

    values = overlay.get("values", {})
    if not isinstance(values, dict):
        return

    # scan.min_score (범위 밖이면 손상/손편집으로 간주 → 적용 거부)
    sm = values.get("scan", {}).get("min_score")
    if sm is not None:
        if _in_range(sm, MIN_SCORE_MIN, MIN_SCORE_MAX):
            cfg.setdefault("scan", {})["min_score"] = sm
        else:
            logger.warning("오버레이 min_score=%s 범위 밖 — 적용 거부", sm)

    # risk.min_rr_ratio
    rr = values.get("risk", {}).get("min_rr_ratio")
    if rr is not None:
        if _in_range(rr, MIN_RR_MIN, MIN_RR_MAX):
            cfg.setdefault("risk", {})["min_rr_ratio"] = rr
        else:
            logger.warning("오버레이 min_rr_ratio=%s 범위 밖 — 적용 거부", rr)

    # risk.risk_tiers[*].risk_pct — min_score 매칭으로 risk_pct만 갱신 (임의 키 주입 차단)
    tiers_ov = values.get("risk", {}).get("risk_tiers")
    if tiers_ov and cfg.get("risk", {}).get("risk_tiers"):
        by_score = {}
        for t in tiers_ov:
            if not (isinstance(t, dict) and "min_score" in t and "risk_pct" in t):
                continue
            p = t["risk_pct"]
            if _in_range(p, RISK_PCT_MIN, RISK_PCT_MAX):
                by_score[t["min_score"]] = p
            else:
                logger.warning("오버레이 risk_pct=%s 범위 밖 — 적용 거부", p)
        for tier in cfg["risk"]["risk_tiers"]:
            if tier.get("min_score") in by_score:
                tier["risk_pct"] = by_score[tier["min_score"]]

    logger.info("학습 오버레이 적용됨: %s", overlay.get("meta", {}).get("generated_at", "?"))
