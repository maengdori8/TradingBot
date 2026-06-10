"""
ICT Paper Trading Bot — 메인 실행 파일
GitHub Actions에서 15분마다 실행됨
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime, timezone
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
    from src.strategy import load_strategy_params
    from src.strategy.signal_engine import scan_symbol
    from src.strategy.kill_zone import get_active_session
    from src.strategy.market_structure import detect_htf_trend
    from src.risk.risk_manager import RiskManager
    from src.paper_trading.paper_engine import PaperEngine
    from src.notification.discord_bot import DiscordNotifier
    from src.scan_store import save_scan_state, to_tradingview

    cfg      = load_config()
    scan_cfg = cfg.get("scan", {})
    client   = MarketDataClient()
    notifier = DiscordNotifier(os.getenv("DISCORD_WEBHOOK_URL", ""))
    risk     = RiskManager()
    paper    = PaperEngine(initial_balance=risk.trading_capital)

    # 청산 이벤트 → 리스크 기록 + Discord 알림 연동
    def _on_trade(pnl: float, reason: str, pos: object) -> None:
        risk.record_result(pnl, reason)
        notifier.notify_exit(
            symbol=pos.symbol, direction=pos.direction,
            exit_price=pos.entry_price, pnl=pnl, reason=reason,
        )
    paper.register_on_trade(_on_trade)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("══════ 봇 실행 [페이퍼] %s ══════", now_str)

    # 스캔 대상 = 동적 상위 심볼 ∪ 보유 포지션 심볼 (보유분 SL/TP 체크 보장)
    scan_symbols = _resolve_symbols(cfg, client)
    open_symbols = {p.symbol for p in paper.get_positions()}
    all_symbols = list(dict.fromkeys(scan_symbols + list(open_symbols)))
    logger.info("스캔 대상 %d개 심볼 (보유 %d개 포함)", len(all_symbols), len(open_symbols))

    min_score = scan_cfg.get("min_score", 70)
    require_volume = scan_cfg.get("require_volume", False)

    # BTC 상위TF 추세 레짐 (역추세 신호 사이징 강등용) — 실패 시 fail-open(강등 없음)
    # 근거: 신호연구에서 BTC 역행 신호는 조건부 부분집합에서 -0.258R. 단 전체표본 +0.145R라
    # 하드 차단 대신 리스크 최하단(0.3%) 강등. counter 100건 누적 후 EV로 격상/해제 재평가.
    btc_trend = None
    try:
        bt_params = load_strategy_params().get("btc_trend", {})
        df_btc = client.fetch_ohlcv("BTC/USDT:USDT", "4h", limit=200)
        btc_trend = detect_htf_trend(
            df_btc,
            ema_period=bt_params.get("ema_period", 50),
            slope_lookback=bt_params.get("slope_lookback", 10),
        )
        logger.info("BTC 4H 추세 레짐: %s", btc_trend)
    except Exception as e:
        logger.warning("BTC 추세 판정 실패(fail-open, 강등 없음): %s", e)

    results = []   # 모든 ScanResult
    scanned = 0
    for symbol in all_symbols:
        try:
            price  = client.fetch_current_price(symbol)
            df_4h  = client.fetch_ohlcv(symbol, "4h",  limit=100)
            df_1h  = client.fetch_ohlcv(symbol, "1h",  limit=100)
            df_15m = client.fetch_ohlcv(symbol, "15m", limit=100)
            scanned += 1

            # 보유 포지션: 미실현 손익 갱신 + SL/TP 체크
            if symbol in open_symbols:
                paper.update_unrealized_pnl(symbol, price)
                last = df_15m.iloc[-1]
                paper.check_stops(symbol, float(last["high"]), float(last["low"]))

            res = scan_symbol(
                df_4h, df_1h, df_15m, symbol, price,
                min_rr=risk.min_rr, min_score=min_score,
                require_volume=require_volume,
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
        allowed, reason = risk.check_trade_allowed(
            current_positions=len(paper.positions),
            positions=paper.get_positions(),
            symbol=res.symbol,
            direction=res.direction,
        )
        if not allowed:
            logger.info("[%s] 거래 차단: %s", res.symbol, reason)
            continue

        sig = res.signal

        # BTC 역추세 신호 → 리스크 최하단 강등 (차단 아님, fail-open: btc_trend=None이면 미적용)
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
        entry_session = get_active_session(datetime.now(timezone.utc))
        checks = dict(res.checks or {})
        checks["btc_aligned"] = alignment       # 학습/사후분석 태깅 (JSON 보존)
        pos = paper.open_position(
            symbol=res.symbol, direction=sig.direction,
            entry_price=sig.entry_price, qty=params["qty"],
            stop_loss=sig.stop_loss, take_profit=sig.take_profit,
            leverage=params["leverage"],
            score=res.score, checks=checks,
            entry_rr=sig.rr_ratio, entry_session=entry_session,
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
                logger.info("🏆 실전 전환 기준 충족! 점수: %.0f/100", result.score)
                notifier.notify_promote(result)
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

    logger.info("══════ 완료 ══════")


if __name__ == "__main__":
    run()
