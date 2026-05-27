"""
실전 전환 판별기 -- 페이퍼 트레이딩 성과가 실전 전환 기준을 충족하는지 판별한다.
config.yaml의 promote 섹션에서 기준을 로드하며, 없으면 기본값을 사용한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent


# ------------------------------------------------------------------
# 결과 데이터클래스
# ------------------------------------------------------------------


@dataclass
class CriterionResult:
    """개별 판별 기준의 결과."""

    name: str
    passed: bool
    value: float
    threshold: float
    weight: float


@dataclass
class PromoteResult:
    """실전 전환 판별 종합 결과."""

    eligible: bool
    score: float
    criteria: dict[str, CriterionResult] = field(default_factory=dict)
    summary: str = ""


# ------------------------------------------------------------------
# 기본값
# ------------------------------------------------------------------

_DEFAULTS: dict[str, float] = {
    "min_trades": 20,
    "min_win_rate": 0.55,
    "min_profit_factor": 1.5,
    "max_mdd": 0.05,
    "min_sharpe": 1.0,
    "min_return_pct": 0.0,
}

# 가중치: 기준 키 -> 가중치 (합계 = 100)
_WEIGHTS: dict[str, float] = {
    "win_rate": 25.0,
    "profit_factor": 25.0,
    "mdd": 20.0,
    "sharpe": 15.0,
    "return_pct": 15.0,
}


# ------------------------------------------------------------------
# PromoteChecker
# ------------------------------------------------------------------


class PromoteChecker:
    """페이퍼 트레이딩 성과의 실전 전환 가능 여부를 판별한다."""

    def __init__(self) -> None:
        """config.yaml에서 promote 기준을 로드한다. 없으면 기본값 사용."""
        cfg = self._load_promote_config()

        self.min_trades: int = int(cfg.get("min_trades", _DEFAULTS["min_trades"]))
        self.min_win_rate: float = float(cfg.get("min_win_rate", _DEFAULTS["min_win_rate"]))
        self.min_profit_factor: float = float(cfg.get("min_profit_factor", _DEFAULTS["min_profit_factor"]))
        self.max_mdd: float = float(cfg.get("max_mdd", _DEFAULTS["max_mdd"]))
        self.min_sharpe: float = float(cfg.get("min_sharpe", _DEFAULTS["min_sharpe"]))
        self.min_return_pct: float = float(cfg.get("min_return_pct", _DEFAULTS["min_return_pct"]))

        logger.info(
            "PromoteChecker 초기화: trades>=%d, wr>=%.2f, pf>=%.2f, mdd<=%.2f, sharpe>=%.2f, ret>=%.2f",
            self.min_trades,
            self.min_win_rate,
            self.min_profit_factor,
            self.max_mdd,
            self.min_sharpe,
            self.min_return_pct,
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _load_promote_config() -> dict:
        """config.yaml에서 promote 섹션을 로드한다.

        Returns:
            promote 설정 딕셔너리. 파일이 없거나 섹션이 없으면 빈 딕셔너리.
        """
        config_path = ROOT / "config" / "config.yaml"
        try:
            with open(config_path) as f:
                full = yaml.safe_load(f) or {}
            return full.get("promote", {})
        except FileNotFoundError:
            logger.warning("config.yaml을 찾을 수 없습니다. 기본값을 사용합니다.")
            return {}

    # ------------------------------------------------------------------
    # 판별 실행
    # ------------------------------------------------------------------

    def check(self, performance: dict) -> PromoteResult:
        """
        페이퍼 트레이딩 성과를 기준과 비교하여 실전 전환 가능 여부를 판별한다.

        Args:
            performance: PaperEngine.get_performance()의 반환값과 동일한 형식의 딕셔너리.
                필수 키: total_trades, win_rate, profit_factor, mdd, sharpe, return_pct

        Returns:
            PromoteResult 종합 판별 결과
        """
        criteria: dict[str, CriterionResult] = {}

        # --- 1. 최소 거래 수 (점수 가중치 없음, 필수 조건) ---
        total_trades = performance.get("total_trades", 0)
        trades_passed = total_trades >= self.min_trades
        criteria["min_trades"] = CriterionResult(
            name="최소 거래 수",
            passed=trades_passed,
            value=float(total_trades),
            threshold=float(self.min_trades),
            weight=0.0,  # 가중치 점수에 포함하지 않지만 eligible 판별에는 포함
        )

        # --- 2. 승률 ---
        win_rate = performance.get("win_rate", 0.0)
        wr_passed = win_rate >= self.min_win_rate
        criteria["win_rate"] = CriterionResult(
            name="승률",
            passed=wr_passed,
            value=round(win_rate, 8),
            threshold=self.min_win_rate,
            weight=_WEIGHTS["win_rate"],
        )

        # --- 3. Profit Factor ---
        pf = performance.get("profit_factor", 0.0)
        pf_passed = pf >= self.min_profit_factor
        criteria["profit_factor"] = CriterionResult(
            name="Profit Factor",
            passed=pf_passed,
            value=round(pf, 8),
            threshold=self.min_profit_factor,
            weight=_WEIGHTS["profit_factor"],
        )

        # --- 4. 최대 낙폭 (MDD) ---
        mdd = performance.get("mdd", 1.0)
        mdd_passed = mdd <= self.max_mdd
        criteria["mdd"] = CriterionResult(
            name="최대 낙폭(MDD)",
            passed=mdd_passed,
            value=round(mdd, 8),
            threshold=self.max_mdd,
            weight=_WEIGHTS["mdd"],
        )

        # --- 5. Sharpe Ratio ---
        sharpe = performance.get("sharpe", 0.0)
        sharpe_passed = sharpe >= self.min_sharpe
        criteria["sharpe"] = CriterionResult(
            name="Sharpe Ratio",
            passed=sharpe_passed,
            value=round(sharpe, 8),
            threshold=self.min_sharpe,
            weight=_WEIGHTS["sharpe"],
        )

        # --- 6. 총 수익률 ---
        return_pct = performance.get("return_pct", -1.0)
        ret_passed = return_pct >= self.min_return_pct
        criteria["return_pct"] = CriterionResult(
            name="총 수익률",
            passed=ret_passed,
            value=round(return_pct, 8),
            threshold=self.min_return_pct,
            weight=_WEIGHTS["return_pct"],
        )

        # --- 종합 점수 계산 ---
        score = 0.0
        for cr in criteria.values():
            if cr.passed and cr.weight > 0:
                score += cr.weight

        # 모든 기준 충족 시에만 eligible
        eligible = all(cr.passed for cr in criteria.values())

        # --- 요약 메시지 ---
        passed_count = sum(1 for cr in criteria.values() if cr.passed)
        total_count = len(criteria)

        if eligible:
            summary = f"실전 전환 가능: 모든 기준 충족 (점수 {score:.0f}/100)"
        else:
            failed = [cr.name for cr in criteria.values() if not cr.passed]
            summary = (
                f"실전 전환 불가: {passed_count}/{total_count}개 기준 충족, "
                f"미충족 항목: {', '.join(failed)} (점수 {score:.0f}/100)"
            )

        result = PromoteResult(
            eligible=eligible,
            score=round(score, 2),
            criteria=criteria,
            summary=summary,
        )

        logger.info("실전 전환 판별 결과: %s", summary)
        return result
