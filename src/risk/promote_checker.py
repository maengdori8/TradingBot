from __future__ import annotations

# 레거시 정보성 성과 표시와 엄격한 단계별 승급 게이트 공개 API.

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
    informational_only: bool = True
    live_authorized: bool = False
    frozen: bool = True
    strategy_id: str = "ict-benchmark-v1"


# ------------------------------------------------------------------
# 기본값
# ------------------------------------------------------------------

# 2026-06 전략 개정(RR 2.5, 설계 승률 37~42%)에 맞춘 기본값 — config promote 섹션과 동기화
_DEFAULTS: dict[str, float] = {
    "min_trades": 50,          # 운/실력 구분 표본 (Barber 2014)
    "min_win_rate": 0.38,      # RR2.5 손익분기 28.6% + 마진
    "min_profit_factor": 1.5,
    "max_mdd": 0.10,
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
        # 통계 게이트: 승률의 Wilson 95% 신뢰하한 > 손익분기 승률 (운/실력 구분 —
        # Barber et al. 2014: 지속 수익 데이트레이더 <1%. 점추정만으론 운도 통과함)
        self.require_wilson_gate: bool = bool(cfg.get("require_wilson_gate", True))
        self.breakeven_winrate: float = float(cfg.get("breakeven_winrate", 0.286))
        self.frozen: bool = bool(cfg.get("frozen", True))
        self.strategy_id: str = str(cfg.get("strategy_id", "ict-benchmark-v1"))

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
            with open(config_path, encoding="utf-8") as f:
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

        # --- 7. 통계 게이트: 승률 Wilson 95% 신뢰하한 > 손익분기 (운/실력 구분) ---
        if self.require_wilson_gate:
            from src.risk.learner import wilson_interval
            wins = int(round(win_rate * total_trades))
            wl_lb, _ = wilson_interval(wins, total_trades) if total_trades else (0.0, 1.0)
            criteria["winrate_lb"] = CriterionResult(
                name="승률 신뢰하한",
                passed=wl_lb > self.breakeven_winrate,
                value=round(wl_lb, 4),
                threshold=self.breakeven_winrate,
                weight=0.0,   # 점수 미반영, eligible 판별에만 포함
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
            summary = (
                f"정보성 성과 기준 충족 (점수 {score:.0f}/100) — "
                "offline/demo 게이트와 수동 승인 없이는 실전 전환 불가"
            )
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
            informational_only=True,
            live_authorized=False,
            frozen=self.frozen,
            strategy_id=self.strategy_id,
        )

        logger.info("실전 전환 판별 결과: %s", summary)
        return result

    def can_activate_live(self, performance: dict) -> bool:
        """레거시 표시 판정이 실전 활성화 권한을 갖지 않음을 명시한다.

        Args:
            performance: 레거시 성과 딕셔너리.

        Returns:
            항상 False. 실전 활성화는 데모 게이트와 수동 승인이 담당한다.
        """
        result = self.check(performance)
        logger.warning(
            "레거시 PromoteChecker는 live 권한이 없습니다: strategy=%s score=%.0f",
            result.strategy_id,
            result.score,
        )
        return False
