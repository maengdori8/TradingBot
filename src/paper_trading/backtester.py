from __future__ import annotations

# 백테스팅 프레임워크 — 과거 OHLCV 데이터를 종료 봉 기준으로 재생한다.

import logging
import math
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.paper_trading.paper_engine import PaperEngine
from src.risk.position_sizer import calculate_position_size
from src.strategy.decision import DecisionContext, slice_decision_frames
from src.strategy.signal_engine import generate_signal

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """백테스트 결과 데이터 클래스."""

    total_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class Backtester:
    """ICT 전략 백테스팅 프레임워크."""

    def __init__(
        self,
        initial_balance: float = 1250.0,
        risk_per_trade: float = 0.005,
        leverage: float = 5.0,
        min_rr: float = 2.5,
        ignore_kill_zone: bool = False,
        strategy_version: str | None = None,
    ) -> None:
        """
        백테스터 초기화.

        Args:
            initial_balance: 초기 잔고 (USDT)
            risk_per_trade: 트레이드당 리스크 비율 (예: 0.01 = 1%)
            leverage: 레버리지 배수
            min_rr: 최소 R:R 비율
            ignore_kill_zone: (구버전 호환) 킬존 게이트는 2026-06 개정으로
                폐지되어 신호 생성이 이미 24h — 이 옵션은 사실상 no-op
            strategy_version: 검증할 전략 버전. 미지정 시 runtime 설정 사용.
        """
        self.initial_balance: float = initial_balance
        self.risk_per_trade: float = risk_per_trade
        self.leverage: float = leverage
        self.min_rr: float = min_rr
        self.ignore_kill_zone: bool = ignore_kill_zone
        runtime = load_config().get("runtime", {})
        self.strategy_version = strategy_version or str(
            runtime.get("strategy_version", "unknown")
        )
        self._result: BacktestResult | None = None

        logger.info(
            "백테스터 초기화: balance=%.2f risk=%.2f%% leverage=%.1fx min_rr=%.1f "
            "진입=24h(킬존 게이트 없음)",
            initial_balance,
            risk_per_trade * 100,
            leverage,
            min_rr,
        )

    def run(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        symbol: str = "BTC/USDT:USDT",
    ) -> BacktestResult:
        """
        15분봉 기준 시간 순회로 백테스트 실행.

        1. 현재 시점의 4h/1h/15m 슬라이스 추출
        2. generate_signal() 호출
        3. 신호 있으면 PaperEngine으로 진입
        4. 매 캔들마다 check_stops() 호출
        5. 결과 집계

        Args:
            df_4h: 4시간봉 OHLCV DataFrame
            df_1h: 1시간봉 OHLCV DataFrame
            df_15m: 15분봉 OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            백테스트 결과
        """
        # 임시 DB로 PaperEngine 격리 (기존 DB 충돌 방지)
        tmp_dir = tempfile.mkdtemp()
        tmp_db = Path(tmp_dir) / "backtest.db"
        engine = PaperEngine(
            initial_balance=self.initial_balance,
            db_path=tmp_db,
        )

        lookback_4h = 100
        lookback_1h = 100
        lookback_15m = 100

        start_idx = lookback_15m
        total_candles = len(df_15m)

        if total_candles <= start_idx:
            logger.warning(
                "15분봉 데이터 부족: %d개 (최소 %d개 필요)",
                total_candles,
                start_idx + 1,
            )
            self._result = self._build_empty_result()
            return self._result

        logger.info(
            "백테스트 시작: %s 캔들=%d (순회=%d~%d)",
            symbol,
            total_candles,
            start_idx,
            total_candles - 1,
        )

        equity_curve: list[float] = []
        trade_log: list[dict] = []
        run_id = f"backtest-{uuid.uuid4()}"

        # 킬존 몽키패치 제거(2026-06): 킬존은 더 이상 게이트가 아니므로 패치하면
        # 오히려 모든 시점에 +5 가점이 부여돼 결과가 오염됨. ignore_kill_zone은 no-op.
        if self.ignore_kill_zone:
            logger.warning("ignore_kill_zone은 폐기됨(no-op) — 신호 생성은 이미 24h")

        try:
            for i in range(start_idx, total_candles):
                bar_open_time = df_15m.index[i]
                bar_close_time = bar_open_time.to_pydatetime() + timedelta(minutes=15)
                context = DecisionContext.for_closed_bar(
                    bar_close_time=bar_close_time,
                    strategy_version=self.strategy_version,
                    run_id=f"{run_id}:{bar_open_time.isoformat()}",
                )
                sim_time = context.decision_time
                current_price = float(df_15m.iloc[i]["close"])
                current_high = float(df_15m.iloc[i]["high"])
                current_low = float(df_15m.iloc[i]["low"])

                # SL/TP 체크 (매 캔들) — 캔들 시각 전달 (펀딩비 정확 산정)
                engine.check_stops(symbol, current_high, current_low,
                                   current_time=sim_time)

                # 미실현손익 갱신
                engine.update_unrealized_pnl(
                    symbol,
                    current_price,
                    current_time=sim_time,
                )

                # 현금 + 묶인 증거금 + 미실현손익 기록
                equity_curve.append(engine.equity)

                # 포지션 수 제한 (최대 1개)
                if len(engine.positions) >= 1:
                    continue

                # 각 타임프레임에서 완전히 닫힌 봉만 동일 규칙으로 추출
                frames = slice_decision_frames(
                    df_4h,
                    df_1h,
                    df_15m,
                    context,
                    lookbacks={
                        "4h": lookback_4h,
                        "1h": lookback_1h,
                        "15m": lookback_15m,
                    },
                )
                slice_4h = frames.df_4h
                slice_1h = frames.df_1h
                slice_15m = frames.df_15m

                # 최소 데이터 확인
                if len(slice_4h) < 20 or len(slice_1h) < 20:
                    continue

                # 신호 생성
                try:
                    signal = generate_signal(
                        slice_4h,
                        slice_1h,
                        slice_15m,
                        symbol,
                        current_price,
                        self.min_rr,
                        context=context,
                    )
                except Exception as e:
                    logger.debug("신호 생성 오류 (idx=%d): %s", i, e)
                    continue

                if signal is None:
                    continue

                # 포지션 사이징
                try:
                    qty = calculate_position_size(
                        engine.balance,
                        self.risk_per_trade,
                        signal.entry_price,
                        signal.stop_loss,
                        self.leverage,
                    )
                except ValueError as e:
                    logger.debug("포지션 사이징 오류: %s", e)
                    continue

                if qty <= 0:
                    continue

                pos = engine.open_position(
                    symbol,
                    signal.direction,
                    signal.entry_price,
                    qty,
                    signal.stop_loss,
                    signal.take_profit,
                    entry_time=sim_time,   # 시뮬레이션 시각 (펀딩비 산정 기준)
                )

                if pos is not None:
                    trade_log.append({
                        "entry_time": context.decision_time.isoformat(),
                        "direction": signal.direction,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "qty": qty,
                        "reason": signal.reason,
                        "rr_ratio": signal.rr_ratio,
                    })
                    logger.debug(
                        "[BT] %s %s 진입 @ %.4f (SL=%.4f, TP=%.4f)",
                        symbol,
                        signal.direction.upper(),
                        signal.entry_price,
                        signal.stop_loss,
                        signal.take_profit,
                    )

            # 미청산 포지션 강제 청산 (마지막 캔들 시각 기준)
            last_price = float(df_15m.iloc[-1]["close"])
            last_time = (
                df_15m.index[-1].to_pydatetime() + timedelta(minutes=15)
            )
            for pos in list(engine.positions):
                pnl = engine.close_position(
                    pos, last_price, "backtest_end", exit_time=last_time
                )
                logger.info(
                    "[BT] 미청산 포지션 강제 청산: PnL=%.4f", pnl
                )

        finally:
            pass   # (킬존 패치 복원 로직 제거 — 패치 자체가 폐기됨)

        self._result = self._build_result(engine, trade_log, equity_curve)
        self._log_summary(self._result)
        return self._result

    def get_results(self) -> BacktestResult:
        """
        백테스트 결과 반환.

        Returns:
            마지막 실행의 BacktestResult

        Raises:
            RuntimeError: run()이 아직 호출되지 않은 경우
        """
        if self._result is None:
            raise RuntimeError("run()을 먼저 호출하세요.")
        return self._result

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _build_result(
        self,
        engine: PaperEngine,
        trade_log: list[dict],
        equity_curve: list[float],
    ) -> BacktestResult:
        """
        PaperEngine 성과 데이터로 BacktestResult를 구성한다.

        Args:
            engine: 백테스트에 사용된 PaperEngine 인스턴스
            trade_log: 거래 진입 로그
            equity_curve: 자본 변화 곡선

        Returns:
            BacktestResult 객체
        """
        perf = engine.get_performance()

        if "message" in perf:
            # 거래 없음
            return BacktestResult(
                total_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                trades=trade_log,
                equity_curve=equity_curve,
            )

        # 최대 낙폭 계산 (equity_curve 기반)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # UTC 일별 순자산 수익률 기반 Sharpe
        sharpe_ratio = float(perf.get("daily_sharpe", 0.0))

        return BacktestResult(
            total_trades=perf["total_trades"],
            win_rate=round(perf["win_rate"], 4),
            total_pnl=round(perf["total_pnl"], 8),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            profit_factor=round(perf["profit_factor"], 4),
            avg_trade_pnl=round(perf["avg_pnl"], 8),
            trades=trade_log,
            equity_curve=equity_curve,
        )

    def _build_empty_result(self) -> BacktestResult:
        """데이터 부족 시 빈 결과 반환.

        Returns:
            빈 BacktestResult 객체
        """
        return BacktestResult(
            total_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
        )

    @staticmethod
    def _calculate_max_drawdown(equity_curve: list[float]) -> float:
        """
        Equity curve에서 최대 낙폭(MDD)을 계산한다.

        Args:
            equity_curve: 자본 변화 곡선

        Returns:
            최대 낙폭 비율 (0~1)
        """
        if not equity_curve:
            return 0.0

        arr = np.array(equity_curve)
        peak = np.maximum.accumulate(arr)
        drawdown = (peak - arr) / np.where(peak > 0, peak, 1.0)
        return float(drawdown.max())

    @staticmethod
    def _calculate_sharpe(
        equity_curve: list[float],
        periods_per_year: int = 365,
    ) -> float:
        """
        UTC 일별 Equity curve에서 Sharpe ratio를 계산한다.

        Args:
            equity_curve: 자본 변화 곡선
            periods_per_year: 연간 일수 (기본: 암호화폐 365일)

        Returns:
            Sharpe ratio
        """
        if len(equity_curve) < 2:
            return 0.0

        arr = np.array(equity_curve)
        returns = np.diff(arr) / arr[:-1]

        if len(returns) < 2 or returns.std(ddof=1) == 0:
            return 0.0

        return float(
            returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year)
        )

    @staticmethod
    def _log_summary(result: BacktestResult) -> None:
        """
        백테스트 결과 요약 로그 출력.

        Args:
            result: 백테스트 결과
        """
        logger.info("=" * 60)
        logger.info("백테스트 결과 요약")
        logger.info("=" * 60)
        logger.info("총 거래 수: %d", result.total_trades)
        logger.info("승률: %.2f%%", result.win_rate * 100)
        logger.info("총 손익: %.2f USDT", result.total_pnl)
        logger.info("평균 거래 손익: %.4f USDT", result.avg_trade_pnl)
        logger.info("최대 낙폭(MDD): %.2f%%", result.max_drawdown * 100)
        logger.info("Sharpe Ratio: %.4f", result.sharpe_ratio)
        logger.info("Profit Factor: %.4f", result.profit_factor)
        logger.info("=" * 60)
