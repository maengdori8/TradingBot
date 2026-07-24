from __future__ import annotations

# ICT Paper Trading Bot — 메인 실행 파일
# GitHub Actions에서 15분마다 실행됨

import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# PYTHONPATH 보정 (GitHub Actions 환경 대응)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Windows cp949 콘솔에서 이모지/박스문자 출력 시 UnicodeEncodeError 방지
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")

DECISION_DB_PATH = LOG_DIR / "decision_ledger.db"


class DecisionLedger:
    """심볼·봉·전략 버전별 주문 결정을 한 번만 허용하는 영속 원장."""

    def __init__(
        self,
        db_path: Path = DECISION_DB_PATH,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        """SQLite 결정 원장을 초기화한다.

        Args:
            db_path: 결정 원장 SQLite 경로.
            connection: 기존 거래 DB 연결. 전달하면 연결 수명은 호출자가 관리한다.
        """
        self._owns_connection = connection is None
        self.conn = connection or sqlite3.connect(str(db_path), timeout=30)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entry_decisions (
                decision_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                bar_close_time TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                run_id TEXT NOT NULL,
                claimed_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def claim(
        self,
        symbol: str,
        bar_close_time: datetime,
        strategy_version: str,
        run_id: str,
    ) -> bool:
        """진입 결정을 원자적으로 선점한다.

        Returns:
            처음 선점한 결정이면 True, 이미 처리된 결정이면 False.
        """
        decision_key = (
            f"{strategy_version}|{symbol}|{bar_close_time.astimezone(timezone.utc).isoformat()}"
        )
        try:
            self.conn.execute(
                """
                INSERT INTO entry_decisions
                    (decision_key, symbol, bar_close_time, strategy_version, run_id, claimed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_key,
                    symbol,
                    bar_close_time.astimezone(timezone.utc).isoformat(),
                    strategy_version,
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self) -> None:
        """SQLite 연결을 닫는다."""
        if self._owns_connection:
            self.conn.close()


def _closed_15m_boundary(now: datetime) -> datetime:
    """현재 시각 이전의 마지막 15분 봉 종료 경계를 반환한다."""
    utc_now = now.astimezone(timezone.utc)
    return utc_now.replace(
        minute=(utc_now.minute // 15) * 15,
        second=0,
        microsecond=0,
    )


def _require_paper_runtime(cfg: dict) -> str:
    """현재 오케스트레이터가 안전하게 지원하는 paper 모드만 허용한다."""
    runtime = cfg.get("runtime", {})
    mode = str(runtime.get("mode", "paper")).lower()
    if mode == "live":
        raise RuntimeError(
            "live 모드는 승인 리포트·수동 토큰·전용 실행 대사 경로 없이는 시작할 수 없습니다"
        )
    if mode == "demo":
        raise RuntimeError(
            "demo 주문은 BybitOrderExecutor 전용 검증 러너에서만 허용됩니다"
        )
    if mode != "paper":
        raise ValueError(f"지원하지 않는 runtime.mode입니다: {mode}")
    return mode


def load_config() -> dict:
    # config.yaml + 학습 오버레이(logs/learned_params.yaml) 병합 로더 일원화
    from src.config_loader import load_config as _load
    return _load()


def _resolve_symbols(cfg: dict, client) -> list[str]:
    """스캔 대상 심볼 목록을 결정한다 (dynamic=거래량 상위 / static=고정)."""
    scan = cfg.get("scan", {})
    mode = scan.get("mode", "static")
    if mode == "dynamic":
        symbols = client.fetch_top_symbols(
            limit=scan.get("top_n", 40),
            min_volume_usdt=scan.get("min_volume_usdt", 5_000_000),
        )
        if symbols:
            return symbols
        logger.warning("동적 심볼 조회 실패 — config 고정 목록으로 폴백")
    return list(cfg["exchange"]["symbols"])


def run() -> None:
    from src.exchange.bybit_client import MarketDataClient
    from src.strategy import (
        DecisionContext,
        load_strategy_params,
        slice_closed_bars,
        slice_decision_frames,
    )
    from src.strategy.signal_engine import scan_symbol
    from src.strategy.kill_zone import get_active_session
    from src.strategy.market_structure import detect_htf_trend
    from src.risk.risk_manager import RiskManager
    from src.paper_trading.execution_model import (
        OrderBookExecutionModel,
        OrderBookSnapshot,
    )
    from src.paper_trading.paper_engine import PaperEngine
    from src.notification.discord_bot import DiscordNotifier
    from src.scan_store import save_scan_state, to_tradingview

    cfg      = load_config()
    mode     = _require_paper_runtime(cfg)
    scan_cfg = cfg.get("scan", {})
    runtime_cfg = cfg.get("runtime", {})
    execution_cfg = cfg.get("paper_execution", {})
    client   = MarketDataClient()
    notifier = DiscordNotifier(os.getenv("DISCORD_WEBHOOK_URL", ""))
    risk     = RiskManager()
    paper    = PaperEngine(
        initial_balance=risk.trading_capital,
        execution_model=OrderBookExecutionModel(
            queue_fill_ratio=float(
                execution_cfg.get("queue_fill_ratio", 0.25)
            ),
            adverse_selection_bps=float(
                execution_cfg.get("adverse_selection_bps", 2.0)
            ),
            max_slippage_bps=float(
                execution_cfg.get("max_slippage_bps", 25.0)
            ),
        ),
    )
    ledger   = DecisionLedger(connection=paper.conn)

    decision_time = datetime.now(timezone.utc)
    bar_close_time = _closed_15m_boundary(decision_time)
    run_id = uuid.uuid4().hex
    strategy_version = str(
        runtime_cfg.get("strategy_version", "ict-benchmark-v1")
    )
    context = DecisionContext.for_closed_bar(
        bar_close_time,
        strategy_version=strategy_version,
        run_id=run_id,
        decision_time=decision_time,
        data_cutoff=bar_close_time,
    )

    # 청산 이벤트 → 리스크 기록 + Discord 알림 연동
    def _on_trade(pnl: float, reason: str, pos: object) -> None:
        risk.record_result(pnl, reason)
        notifier.notify_exit(
            symbol=pos.symbol, direction=pos.direction,
            exit_price=pos.entry_price, pnl=pnl, reason=reason,
        )
    paper.register_on_trade(_on_trade)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info(
        "══════ 봇 실행 [%s] %s | bar_close=%s run_id=%s ══════",
        mode,
        now_str,
        context.bar_close_time.isoformat(),
        run_id,
    )

    # 스캔 대상 = 동적 상위 심볼 ∪ 보유 포지션 심볼 (보유분 SL/TP 체크 보장)
    scan_symbols = _resolve_symbols(cfg, client)
    open_symbols = {p.symbol for p in paper.get_positions()}
    all_symbols = list(dict.fromkeys(scan_symbols + list(open_symbols)))
    logger.info("스캔 대상 %d개 심볼 (보유 %d개 포함)", len(all_symbols), len(open_symbols))

    min_score = scan_cfg.get("min_score", 70)
    require_volume = scan_cfg.get("require_volume", False)

    # BTC 상위TF 추세 레짐 (역추세 신호 사이징 강등용).
    # 필수 시장 컨텍스트가 없으면 신규 진입은 fail-closed한다.
    btc_trend = None
    btc_context_valid = False
    try:
        bt_params = load_strategy_params().get("btc_trend", {})
        df_btc_raw = client.fetch_ohlcv("BTC/USDT:USDT", "4h", limit=200)
        df_btc = slice_closed_bars(
            df_btc_raw,
            "4h",
            context,
            lookback=200,
        )
        if len(df_btc) < 60:
            raise ValueError("BTC 완전종료 4H 봉이 부족합니다")
        btc_trend = detect_htf_trend(
            df_btc,
            ema_period=bt_params.get("ema_period", 50),
            slope_lookback=bt_params.get("slope_lookback", 10),
        )
        btc_context_valid = True
        logger.info("BTC 4H 추세 레짐: %s", btc_trend)
    except Exception as e:
        logger.error("BTC 추세 판정 실패(fail-closed, 신규 진입 중단): %s", e)

    results = []   # 모든 ScanResult
    scanned = 0
    for symbol in all_symbols:
        try:
            df_4h_raw  = client.fetch_ohlcv(symbol, "4h",  limit=120)
            df_1h_raw  = client.fetch_ohlcv(symbol, "1h",  limit=120)
            df_15m_raw = client.fetch_ohlcv(symbol, "15m", limit=120)
            frames = slice_decision_frames(
                df_4h_raw,
                df_1h_raw,
                df_15m_raw,
                context,
                lookbacks={"4h": 100, "1h": 100, "15m": 100},
            )
            if (
                len(frames.df_4h) < 20
                or len(frames.df_1h) < 20
                or len(frames.df_15m) < 20
            ):
                raise ValueError("완전종료 OHLCV 봉이 부족합니다")

            last_close = (
                frames.df_15m.index[-1].to_pydatetime()
                + timedelta(minutes=15)
            )
            data_age = (context.bar_close_time - last_close).total_seconds()
            max_age_value = runtime_cfg.get("max_data_age_sec")
            max_age = int(max_age_value) if max_age_value is not None else None
            if data_age < 0 or (max_age is not None and data_age > max_age):
                raise ValueError(
                    f"15m 데이터 stale/future: age={data_age:.0f}s limit={max_age}s"
                )
            signal_price = float(frames.df_15m.iloc[-1]["close"])
            scanned += 1

            # 보유 포지션: 미실현 손익 갱신 + SL/TP 체크
            if symbol in open_symbols:
                stop_orderbook = None
                try:
                    market_snapshot = client.fetch_market_snapshot(
                        symbol,
                        order_book_limit=int(
                            execution_cfg.get("order_book_limit", 25)
                        ),
                        max_age_seconds=float(
                            execution_cfg.get(
                                "max_snapshot_age_seconds",
                                5.0,
                            )
                        ),
                    )
                    mark_price = market_snapshot.last
                    stop_orderbook = OrderBookSnapshot(
                        symbol=symbol,
                        bids=market_snapshot.bids,
                        asks=market_snapshot.asks,
                        timestamp=market_snapshot.exchange_timestamp,
                        source=(
                            f"{market_snapshot.provenance.exchange}:"
                            f"{market_snapshot.provenance.market_type}"
                        ),
                    )
                    execution_time = market_snapshot.exchange_timestamp
                except (RuntimeError, ValueError) as exc:
                    logger.error(
                        "[%s] 주문장 스냅샷 실패 — 보수적 청산 모델 사용: %s",
                        symbol,
                        exc,
                    )
                    mark_price = signal_price
                    execution_time = context.decision_time
                paper.update_unrealized_pnl(
                    symbol,
                    mark_price,
                    current_time=execution_time,
                )
                last = frames.df_15m.iloc[-1]
                paper.check_stops(
                    symbol,
                    float(last["high"]),
                    float(last["low"]),
                    current_time=execution_time,
                    orderbook=stop_orderbook,
                )

            res = scan_symbol(
                df_4h_raw,
                df_1h_raw,
                df_15m_raw,
                symbol,
                signal_price,
                min_rr=risk.min_rr, min_score=min_score,
                require_volume=require_volume,
                context=context,
            )
            results.append(res)

        except Exception as e:
            logger.error("[%s] 오류: %s", symbol, e, exc_info=True)

    # 점수 내림차순 정렬
    results.sort(key=lambda r: r.score, reverse=True)

    # ── 진입 처리: 확정(qualified) 신호를 점수 높은 순으로 ──────────────
    open_now = {p.symbol for p in paper.get_positions()}
    entered = 0
    for res in results:
        if not res.qualified or res.signal is None:
            continue
        if res.symbol in open_now:
            continue
        if not btc_context_valid:
            logger.warning("[%s] BTC 시장 컨텍스트 부재로 신규 진입 차단", res.symbol)
            continue
        if not ledger.claim(
            res.symbol,
            context.bar_close_time,
            context.strategy_version,
            context.run_id,
        ):
            logger.info(
                "[%s] 중복 결정 차단: %s %s",
                res.symbol,
                context.strategy_version,
                context.bar_close_time.isoformat(),
            )
            continue
        allowed, reason = risk.check_trade_allowed(
            current_positions=len(paper.positions),
            positions=paper.get_positions(),
            symbol=res.symbol,
            direction=res.direction,
        )
        if not allowed:
            logger.info("[%s] 거래 차단: %s", res.symbol, reason)
            continue

        # 손절 직후 동일 심볼 재진입 쿨다운 (복수매매 차단 — Coval&Shumway 2005:
        # 손실 직후 거래는 기대값 음수. 봇도 같은 셋업 재시도 패턴 구조적 차단)
        last_sl = paper.last_sl_exit(res.symbol)
        if last_sl is not None:
            elapsed_h = (context.decision_time - last_sl).total_seconds() / 3600
            if elapsed_h < risk.reentry_cooldown_hours:
                logger.info(
                    "[%s] SL 재진입 쿨다운 (%.1f/%.0fh)",
                    res.symbol, elapsed_h, risk.reentry_cooldown_hours,
                )
                continue

        sig = res.signal

        # BTC 역추세 신호 → 리스크 최하단 강등
        alignment = "unknown"
        override = None
        if btc_trend is not None:
            if btc_trend == "flat":
                alignment = "flat"
            elif (res.direction == "long") == (btc_trend == "bull"):
                alignment = "aligned"
            else:
                alignment = "counter"
                override = load_strategy_params().get("btc_trend", {}).get(
                    "counter_risk_pct", 0.003
                )
                logger.info("[%s] BTC 역추세(%s vs %s) — 리스크 %.1f%%로 강등",
                            res.symbol, res.direction, btc_trend, override * 100)

        params = risk.calculate_trade_params(
            sig.entry_price, sig.stop_loss, score=res.score,
            risk_pct_override=override,
        )
        entry_session = get_active_session(context.decision_time)
        checks = dict(res.checks or {})
        checks["btc_aligned"] = alignment       # 학습/사후분석 태깅 (JSON 보존)
        try:
            market_snapshot = client.fetch_market_snapshot(
                res.symbol,
                order_book_limit=int(
                    execution_cfg.get("order_book_limit", 25)
                ),
                max_age_seconds=float(
                    execution_cfg.get("max_snapshot_age_seconds", 5.0)
                ),
            )
            entry_orderbook = OrderBookSnapshot(
                symbol=res.symbol,
                bids=market_snapshot.bids,
                asks=market_snapshot.asks,
                timestamp=market_snapshot.exchange_timestamp,
                source=(
                    f"{market_snapshot.provenance.exchange}:"
                    f"{market_snapshot.provenance.market_type}"
                ),
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "[%s] 신규 진입 차단 — 주문장 출처/최신성 검증 실패: %s",
                res.symbol,
                exc,
            )
            continue
        pos = paper.open_position(
            symbol=res.symbol, direction=sig.direction,
            entry_price=sig.entry_price, qty=params["qty"],
            stop_loss=sig.stop_loss, take_profit=sig.take_profit,
            leverage=params["leverage"],
            score=res.score, checks=checks,
            entry_rr=sig.rr_ratio, entry_session=entry_session,
            entry_time=entry_orderbook.timestamp,
            orderbook=entry_orderbook,
            order_type="market",
            time_in_force="IOC",
        )
        if pos:
            entered += 1
            open_now.add(res.symbol)
            notifier.notify_entry(
                symbol=res.symbol, direction=sig.direction,
                entry=pos.entry_price, stop_loss=pos.stop_loss,
                take_profit=pos.take_profit, qty=pos.qty,
                reason=sig.reason,
            )
    logger.info("스캔 완료: %d개 스캔, %d개 신규 진입", scanned, entered)

    # ── 관심종목(watchlist) 저장 ─────────────────────────────────────
    wl_min = scan_cfg.get("watchlist_min_score", 40)
    wl_size = scan_cfg.get("watchlist_size", 12)
    held = {p.symbol for p in paper.get_positions()}
    watchlist = []
    for res in results:
        if res.symbol in held:
            continue          # 보유 중은 관심종목이 아닌 '포지션'으로 표시
        if res.score < wl_min:
            continue
        d = res.to_dict()
        d["tradingview"] = to_tradingview(res.symbol)
        watchlist.append(d)
        if len(watchlist) >= wl_size:
            break
    qualified_count = sum(1 for r in results if r.qualified)
    save_scan_state(watchlist, scanned, qualified_count)

    # 성과 요약
    # 성과/실전전환 판정은 신체제(epoch) 거래만 — 구체제(다른 출구기하) 거래 혼입 방지
    epoch = cfg.get("learning", {}).get("epoch_start")
    perf = paper.get_performance(since=epoch)
    if "total_trades" in perf:
        logger.info(
            "성과 | 거래:%d 승률:%.1f%% PnL:%.2f USDT 잔고:%.2f USDT",
            perf["total_trades"], perf["win_rate"] * 100,
            perf["total_pnl"], perf["current_balance"],
        )

        # 실전 전환 판별
        try:
            from src.risk.promote_checker import PromoteChecker
            checker = PromoteChecker()
            result = checker.check(perf)
            if result.eligible:
                logger.info(
                    "레거시 정보성 성과 기준 충족(%.0f/100) — "
                    "실전 권한 없음, offline/demo 검증 필요",
                    result.score,
                )
            else:
                logger.info(
                    "실전 전환 미충족 (점수: %.0f/100): %s",
                    result.score, result.summary,
                )
        except ImportError:
            logger.debug("promote_checker 미설치 — 실전 전환 판별 건너뜀")
        except Exception as e:
            logger.warning("실전 전환 판별 오류: %s", e)

        # 매 6시간(00:00, 06:00, 12:00, 18:00 UTC)에 일일 리포트 발송
        now = datetime.now(timezone.utc)
        if now.hour % 6 == 0 and now.minute < 15:
            notifier.notify_daily_report(perf)
            logger.info("일일 리포트 Discord 발송 완료")

    # 자동 파라미터 학습 (청산거래 분석 → 진입기준 자동 조정, 오버레이 기록)
    try:
        from src.risk.learner import maybe_update
        maybe_update(paper, notifier)
    except Exception as e:
        logger.warning("자동 학습 오류(무시): %s", e)

    ledger.close()
    logger.info("══════ 완료 ══════")


if __name__ == "__main__":
    run()
