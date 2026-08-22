"""
ICT Paper Trading Bot — 메인 실행 파일
GitHub Actions에서 15분마다 실행됨
"""
from __future__ import annotations
import logging
import os
import sys
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
    from src.paper_trading import Position
    from src.paper_trading.paper_engine import PaperEngine
    from src.notification.discord_bot import DiscordNotifier
    from src.scan_store import save_scan_state, to_tradingview

    cfg      = load_config()
    scan_cfg = cfg.get("scan", {})
    exec_cfg = cfg.get("execution", {})
    exec_mode = exec_cfg.get("mode", "taker")   # maker(지정가, 출혈↓) | taker(시장가)
    if exec_mode not in ("maker", "taker"):
        # 오타를 조용히 taker로 해석하지 않는다 — 기동 실패(CI 실패 알림)로 드러낸다
        raise ValueError(f"execution.mode 값이 잘못됨: {exec_mode!r} (maker|taker)")
    client   = MarketDataClient()
    notifier = DiscordNotifier(os.getenv("DISCORD_WEBHOOK_URL", ""))
    risk     = RiskManager()
    paper    = PaperEngine(initial_balance=risk.trading_capital,
                           maker_fee=exec_cfg.get("maker_fee", 0.00035))
    if exec_mode == "maker":
        logger.info("실행방식: 메이커 지정가 (offset=%.2fR, 만료=%d봉) — 출혈↓ (수익보장 아님)",
                    exec_cfg.get("limit_offset_r", 0.10), exec_cfg.get("fill_expiry_bars", 8))
    else:
        # maker→taker 전환 시 남아있는 미체결 지정가는 체결/만료될 경로가 없으므로 즉시 정리
        # (같은 심볼에 대기주문+시장가 포지션이 공존하는 중복 노출 차단)
        paper.cancel_all_pending("mode_change")
    # 서킷브레이커(일/주 손실한도·연패 휴식)가 켜져 있으면 신규 포지션 금지 — 미체결 지정가가
    # 체결돼 신규 포지션이 되는 경로도 같이 막는다 (진입 차단과 동일 규칙)
    cb_ok, cb_reason = risk.cb.is_trading_allowed()
    if not cb_ok and paper.get_pending_orders():
        n_cancel = paper.cancel_all_pending(f"risk_blocked:{cb_reason}")
        logger.info("서킷브레이커 활성(%s) — 미체결 지정가 %d건 취소", cb_reason, n_cancel)

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
    pending_symbols = {o["symbol"] for o in paper.get_pending_orders()}  # 미체결 지정가 체결판정 보장
    all_symbols = list(dict.fromkeys(scan_symbols + list(open_symbols) + list(pending_symbols)))
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

    # 실행 판정(지정가 체결 / SL·TP)은 거래 대상 거래소(Bybit 무기한)의 캔들만 쓴다. Bybit이 막혀 있으면
    # (GitHub Actions 등 지역 차단 403) 다른 거래소 현물 폴백으로 기록을 오염시키지 않도록 '관찰 전용' 모드:
    # 신규 주문·진입·체결·청산 판정을 모두 보류하고 알린다. (신호 스캔/관심종목은 폴백 데이터로 계속)
    exec_ok = bool(client.bybit_available())
    if not exec_ok:
        logger.error("Bybit 접근 불가 — 관찰 전용 모드: 신규 진입/주문·체결·SL/TP 판정 모두 보류 (기록 무결성 보호)")
        notifier.notify_error("Bybit 접근 불가(지역 차단?) — 페이퍼 봇 관찰 전용 모드: 진입/체결/청산 판정 보류")

    # 관측 기준 시각(사이클 단위로 고정): 프레임을 받기 '전'에 찍는다. 이 시각 기준으로 닫힌 봉은 프레임 수신
    # 시점에 이미 닫혀 있었으므로 부분 봉을 닫힌 봉으로 오판할 수 없다. 모든 판정(사전판정·체결·SL/TP)이 같은
    # now 를 쓴다 — 같은 스냅샷을 다른 시계로 두 번 해석하지 않기 위함.
    cycle_now = datetime.now(timezone.utc)

    def _valid_exec_frame(df) -> bool:
        """실행 판정에 쓸 수 있는 프레임인지: 비어 있지 않고, UTC 인덱스, 마지막 봉이 최신(지연 ≤ 1봉), 미래 봉 없음."""
        try:
            if df is None or len(df) == 0 or not hasattr(df.index, "tz"):
                return False
            last = df.index[-1].to_pydatetime()
            # 검증 시계는 '조회 시점'의 현재 — 사이클 시작(cycle_now) 뒤 15분 경계를 넘겨 받은 프레임의 새 형성 봉을
            # 미래로 오인하지 않기 위함. 닫힌 봉 판정 자체는 엔진이 cycle_now 기준으로 한다.
            now_f = datetime.now(timezone.utc)
            cur = now_f.replace(second=0, microsecond=0) - timedelta(minutes=now_f.minute % 15)
            # 최신성: 마지막 봉이 '직전 닫힌 봉' 이상. 미래 금지: 마지막 봉 시작 ≤ 현재 형성 중 봉 시작(cur)
            return (last >= cur - timedelta(minutes=15)) and (last <= cur)
        except (AttributeError, TypeError, ValueError):
            return False

    results = []   # 모든 ScanResult
    scanned = 0
    decision_bars: dict[str, datetime] = {}   # 심볼별 결정봉(마지막 15m 봉 시작) — signal_key용
    prices: dict[str, float] = {}
    exec_frames: dict[str, object] = {}       # 심볼별 Bybit 전용 15m 프레임 (실행 판정용, 검증 통과분만)
    exec_prices: dict[str, float] = {}        # 심볼별 Bybit 현재가 (진입가 기준 — 진입에만 필수)
    signal_src_ok: dict[str, bool] = {}       # 신호 프레임이 타 거래소 폴백이 아닌지 (폴백이면 진입 생략)
    for symbol in all_symbols:
        try:
            price  = client.fetch_current_price(symbol)
            df_4h  = client.fetch_ohlcv(symbol, "4h",  limit=100)
            df_1h  = client.fetch_ohlcv(symbol, "1h",  limit=100)
            df_15m = client.fetch_ohlcv(symbol, "15m", limit=100)
            scanned += 1
            prices[symbol] = price
            try:
                decision_bars[symbol] = df_15m.index[-1].to_pydatetime()
            except (AttributeError, IndexError):
                pass
            if exec_ok:
                # 실행용 데이터는 Bybit 전용. 15m 프레임(체결/SL·TP 판정)과 현재가(진입가)는 따로 확보한다 —
                # 티커만 실패해도 기존 포지션/미체결 판정은 계속되고 진입만 막힌다. 프레임이 없거나 비었거나
                # 지연/미래면 이 심볼은 이번 사이클 실행 판정에서 제외된다 — 폴백 데이터로 판정/진입하지 않는다.
                try:
                    frame_b = client.fetch_ohlcv_bybit(symbol, "15m", limit=100)
                    if _valid_exec_frame(frame_b):
                        exec_frames[symbol] = frame_b
                    else:
                        logger.warning("[%s] Bybit 15m 프레임이 비었거나 지연/미래 — 이번 사이클 실행 판정 보류", symbol)
                except Exception as e:  # noqa: BLE001 — 이 심볼만 이번 사이클 판정 보류
                    logger.warning("[%s] Bybit 15m 조회 실패 — 이번 사이클 실행 판정 보류: %s", symbol, e)
                try:
                    bybit_price = client.fetch_current_price_bybit(symbol)
                    exec_prices[symbol] = bybit_price
                    prices[symbol] = bybit_price
                    price = bybit_price              # 신호/진입가도 Bybit 기준으로
                except Exception as e:  # noqa: BLE001 — 진입만 보류
                    logger.warning("[%s] Bybit 현재가 조회 실패 — 이번 사이클 진입 보류: %s", symbol, e)
                # 신호 프레임 출처: 타 거래소 폴백(kraken/coinbase)이면 그 신호로 주문/진입하지 않는다 (출처 미상=허용)
                srcs = {getattr(df, "attrs", {}).get("source") for df in (df_4h, df_1h, df_15m)}
                signal_src_ok[symbol] = srcs <= {None, "bybit"}

            res = scan_symbol(
                df_4h, df_1h, df_15m, symbol, price,
                min_rr=risk.min_rr, min_score=min_score,
                require_volume=require_volume,
            )
            results.append(res)

        except Exception as e:
            logger.error("[%s] 오류: %s", symbol, e, exc_info=True)

    # ── 실행 판정 (Bybit 전용 프레임) ──────────────────────────────────────
    # 1패스: 기존 포지션 SL/TP(전 심볼) → 서킷브레이커 재확인(방금 기록된 손실 반영) → 미체결 취소
    # 2패스: 미체결 체결/만료 + 신규 체결분 SL/TP. 이 순서는 "다른 심볼의 손실로 한도가 먼저 찼다면
    # 그 뒤의 체결은 없어야 한다"를 심볼 순회 순서와 무관하게 보장한다(보수 방향: 체결이 시간상 먼저였어도
    # 같은 사이클 안의 손실이 한도를 넘기면 취소될 수 있다).
    # 닫힌 봉 기준·시간순(형성 중 봉 1회 체크가 아님 — 사이클 사이/누락 실행 구간의 터치도 잡는다, 진입봉은 SL만).
    # 프레임(100봉≈25h)이 판정 시작점까지 못 미치거나 봉이 비어 있으면 엔진이 보류 표시 →
    # 1000봉(~10일) → 필요 시각부터 히스토리 백필 순으로 재조회해 재판정(깊은 프레임에서만 빈 봉=무데이터 인정).
    def _reset_flags(symbol: str) -> None:
        paper.last_history_gaps.discard(symbol)
        paper.last_eval_incomplete.discard(symbol)

    def _eval_complete(symbol: str) -> bool:
        return symbol not in paper.last_eval_incomplete and symbol not in paper.last_history_gaps

    def _deeper_and_anchored(deep, base) -> bool:
        """깊은 프레임이 (1) 비어 있지 않고 (2) 기본 프레임보다 실제로 더 깊으며(시작이 더 이르거나 봉이 더 많음)
        (3) 기본 프레임의 끝(현재)까지 닿을 때만 신뢰한다 — 같은/잘린 응답을 '깊은 관측'으로 오인하지 않기 위함."""
        if deep is None or len(deep) == 0 or len(base) == 0:
            return False
        deeper = deep.index[0] < base.index[0] or len(deep) > len(base)
        anchored_tail = deep.index[-1] >= base.index[-1]
        return bool(deeper and anchored_tail)

    held: set[str] = set()     # 이번 사이클 판정을 끝내지 못해 전체 보류를 유발한 심볼 (경보용)

    def _with_backfill(symbol: str, evaluate) -> tuple[bool, object, bool]:
        """기본(100봉, 엄격) → 1000봉 → 필요 시각부터 히스토리 백필 순으로 판정하고
        (완료 여부, 마지막에 쓴 프레임, 그때의 allow_holes)를 돌려준다.
        깊은 프레임은 기본 프레임보다 실제로 깊고 현재까지 닿을 때만 allow_holes=True로 평가한다."""
        frame = exec_frames.get(symbol)
        if frame is None:
            held.add(symbol)
            return False, None, False
        _reset_flags(symbol)
        evaluate(frame, False)
        if _eval_complete(symbol):
            return True, frame, False
        if symbol not in paper.last_history_gaps:
            return False, frame, False   # 지연 프레임/NaN 등 — 더 깊은 조회로 해결되지 않음 (다음 사이클)
        start = paper.required_history_start(symbol)
        fetchers = [
            ("1000봉", lambda: client.fetch_ohlcv_bybit(symbol, "15m", limit=1000)),
        ]
        if start is not None:
            fetchers.append(("히스토리 백필", lambda: client.fetch_ohlcv_history(
                symbol, "15m", since=start - timedelta(minutes=15))))
        last_frame, last_holes = frame, False
        for label, fetch in fetchers:
            try:
                deep = fetch()
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] %s 조회 실패 — 다음 단계 시도: %s", symbol, label, e)
                continue
            if not _deeper_and_anchored(deep, frame):
                logger.warning("[%s] %s 프레임이 기본 프레임보다 깊지 않거나 현재까지 닿지 않음 — 신뢰하지 않음", symbol, label)
                continue
            if start is not None and deep.index[0] > start:
                logger.warning("[%s] %s 프레임 시작(%s) > 판정 시작점(%s) — 다음 단계",
                               symbol, label, deep.index[0].isoformat(), start.isoformat())
                continue
            _reset_flags(symbol)
            evaluate(deep, True)
            last_frame, last_holes = deep, True
            if _eval_complete(symbol):
                return True, deep, True
            if symbol not in paper.last_history_gaps:
                return False, deep, True
        # 운영자 경보: 자동으로 해소되지 않는 상태(임의 가격으로 강제 정리하지 않는다 — 사람이 확인해야 함)
        logger.error("[%s] 백필로도 판정을 끝내지 못함 — 체결/SL 판정 보류 유지(수동 확인 필요)", symbol)
        notifier.notify_error(f"[{symbol}] 캔들 히스토리 백필로도 판정 불가 — 실행 판정 보류 중 (수동 확인 필요)")
        return False, last_frame, last_holes

    positions_complete = True
    fills_held = False
    if exec_ok:
        for symbol in sorted({p.symbol for p in paper.get_positions()}):
            try:
                if symbol in prices:
                    paper.update_unrealized_pnl(symbol, prices[symbol])
                ok, _f, _h = _with_backfill(
                    symbol, lambda f, h, s=symbol: paper.check_stops_history(s, f, now=cycle_now, allow_holes=h))
            except Exception as e:
                logger.error("[%s] SL/TP 판정 오류: %s", symbol, e, exc_info=True)
                ok = False
            if not ok:
                held.add(symbol)
            positions_complete = positions_complete and ok

        cb_ok, cb_reason = risk.cb.is_trading_allowed()
        if not cb_ok and paper.get_pending_orders():
            n_cancel = paper.cancel_all_pending(f"risk_blocked:{cb_reason}")
            logger.info("서킷브레이커 활성(%s) — 미체결 지정가 %d건 취소 (체결 전 차단)", cb_reason, n_cancel)

        # 기존 포지션 중 하나라도 마지막 닫힌 봉까지 판정을 끝내지 못했으면(프레임 지연/누락/조회 실패) 이번
        # 사이클엔 체결 판정·신규 진입을 하지 않는다 — 먼저 일어난 손실이 한도를 넘겼을 가능성을 배제할 수 없기 때문.
        # 체결은 닫힌 봉 기준이라 다음 사이클에 같은 봉·같은 시각으로 동일하게 판정된다(기록 불변, 알림만 지연).
        fills_held = not positions_complete
        pending_symbols = sorted({o["symbol"] for o in paper.get_pending_orders()})
        if fills_held and pending_symbols:
            logger.warning("기존 포지션 판정 미완 심볼 있음 — 이번 사이클 미체결 체결 판정 보류(다음 사이클에 동일 판정)")

        # 2패스 (a): 각 미체결 심볼의 프레임을 확정하고 '예상 체결봉'만 미리 본다(상태 변경 없음).
        #   한 심볼이라도 판정 미완이면 전체 체결을 보류한다(심볼 순서에 따른 비대칭 방지).
        chosen: dict[str, tuple[object, bool]] = {}
        earliest_fill: dict[str, datetime | None] = {}
        if not fills_held:
            for symbol in pending_symbols:
                def _peek(frame, allow_holes, s=symbol) -> None:
                    res = paper.peek_pending_fills(s, frame, now=cycle_now, allow_holes=allow_holes)
                    earliest_fill[s] = min((ts for _, ts in res if ts is not None), default=None)
                try:
                    ok, used_frame, used_holes = _with_backfill(symbol, _peek)
                except Exception as e:
                    logger.error("[%s] 체결 사전판정 오류: %s", symbol, e, exc_info=True)
                    ok = False
                if not ok:
                    fills_held = True
                    held.add(symbol)
                    logger.warning("[%s] 미체결 판정 미완 — 이번 사이클 전체 체결 판정 보류", symbol)
                    break
                chosen[symbol] = (used_frame, used_holes)

        # 2패스 (b): 예상 체결봉이 이른 심볼부터 시간순으로 실제 체결 → 신규 체결분 SL/TP → 서킷브레이커 재확인.
        #   (같은 사이클 안에서 먼저 체결돼 손실이 난 심볼이 뒤 심볼의 체결을 차단해야 하므로 알파벳 순 금지)
        if not fills_held and pending_symbols:
            far = datetime.max.replace(tzinfo=timezone.utc)
            ordered = sorted(pending_symbols, key=lambda s_: (earliest_fill.get(s_) is None, earliest_fill.get(s_) or far))
            for symbol in ordered:
                try:
                    cb_ok, cb_reason = risk.cb.is_trading_allowed()
                    if not cb_ok:
                        paper.cancel_all_pending(f"risk_blocked:{cb_reason}")
                        break
                    used_frame, used_holes = chosen[symbol]
                    _reset_flags(symbol)
                    for fp in paper.check_pending_fills(symbol, used_frame, now=cycle_now, allow_holes=used_holes):
                        notifier.notify_entry(
                            symbol=fp.symbol, direction=fp.direction, entry=fp.entry_price,
                            stop_loss=fp.stop_loss, take_profit=fp.take_profit, qty=fp.qty,
                            reason="지정가 체결(메이커)",
                        )
                    if any(p.symbol == symbol for p in paper.positions):
                        if symbol in prices:
                            paper.update_unrealized_pnl(symbol, prices[symbol])
                        paper.check_stops_history(symbol, used_frame, now=cycle_now, allow_holes=used_holes)
                    if not _eval_complete(symbol):
                        # 체결 뒤 SL/TP 판정이 끝까지 가지 못함(사전판정이 못 잡은 경우의 안전망) → 이후 전부 보류
                        logger.warning("[%s] 체결 후 판정 미완 — 이후 심볼 체결·신규 진입 보류", symbol)
                        fills_held = True
                        break
                except Exception as e:
                    logger.error("[%s] 체결 판정 오류: %s", symbol, e, exc_info=True)
                    fills_held = True
                    break

    # 전체 보류 상태는 조용히 지나가면 안 된다(한 심볼의 데이터 결손이 전 심볼 체결/진입을 멈춘다) — 원인 심볼을 명시해 경보.
    # 자동으로 풀지 않는다(폴백 가격으로 정리 금지). 사람이 해당 심볼을 확인/정리해야 한다.
    if exec_ok and (not positions_complete or fills_held):
        logger.error("실행 판정 보류(전 심볼 체결/진입 정지) — 미완 심볼: %s", sorted(held))
        notifier.notify_error(f"실행 판정 보류(전 심볼 체결/진입 정지) — 미완 심볼: {sorted(held)} (수동 확인 필요)")

    # 점수 내림차순 정렬
    results.sort(key=lambda r: r.score, reverse=True)

    # ── 진입 처리: 확정(qualified) 신호를 점수 높은 순으로 ──────────────
    # 보유 포지션 + 미체결 지정가(예약분) 모두 '점유'로 본다 — 모드 무관 (심볼당 1개 강제)
    open_now = {p.symbol for p in paper.get_positions()} | \
               {o["symbol"] for o in paper.get_pending_orders()}
    entered = 0
    entries_allowed = exec_ok and positions_complete and not fills_held
    if not exec_ok:
        logger.error("관찰 전용 모드 — 신규 진입/지정가 등록 생략 (Bybit 접근 불가)")
    elif not entries_allowed:
        # 기존 포지션/미체결 판정이 미완이면 손실 한도 상태가 불확실 → 신규 진입·주문도 보류 (다음 사이클)
        logger.warning("실행 판정 미완 — 이번 사이클 신규 진입/지정가 등록 보류")
    for res in (results if entries_allowed else []):
        if not res.qualified or res.signal is None:
            continue
        if res.symbol in open_now:
            continue
        if res.symbol not in exec_frames or res.symbol not in exec_prices:
            # 이 심볼의 Bybit 15m/현재가 조회가 실패한 사이클 — 폴백(타 거래소 현물) 데이터로 주문/진입하지 않는다
            logger.info("[%s] Bybit 실행 프레임/현재가 없음 — 진입/주문 생략 (이번 사이클)", res.symbol)
            continue
        if not signal_src_ok.get(res.symbol, True):
            logger.info("[%s] 신호 프레임이 타 거래소 폴백 — 진입/주문 생략 (이번 사이클)", res.symbol)
            continue
        # 손절 직후 동일 심볼 재진입 쿨다운 (복수매매 차단 — Coval&Shumway 2005:
        # 손실 직후 거래는 기대값 음수. 봇도 같은 셋업 재시도 패턴 구조적 차단)
        last_sl = paper.last_sl_exit(res.symbol)
        if last_sl is not None:
            elapsed_h = (datetime.now(timezone.utc) - last_sl).total_seconds() / 3600
            if elapsed_h < risk.reentry_cooldown_hours:
                logger.info(
                    "[%s] SL 재진입 쿨다운 (%.1f/%.0fh)",
                    res.symbol, elapsed_h, risk.reentry_cooldown_hours,
                )
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
        # 후보 진입가: 메이커면 지정가(유리한 오프셋), 테이커면 신호 진입가
        limit_price = sig.entry_price
        if exec_mode == "maker":
            # 지정가 = 진입가 ∓ offset × |진입가-손절가| (유리한 오프셋). 미체결=거래없음.
            risk_dist = abs(sig.entry_price - sig.stop_loss)
            off = exec_cfg.get("limit_offset_r", 0.10) * risk_dist
            limit_price = (sig.entry_price - off if sig.direction == "long"
                           else sig.entry_price + off)
        cand_lev = params["leverage"] if params["leverage"] and params["leverage"] > 0 else 1.0
        candidate = Position(
            id="candidate", symbol=res.symbol, direction=sig.direction,
            entry_price=limit_price, qty=params["qty"], stop_loss=sig.stop_loss,
            take_profit=sig.take_profit, margin=round(limit_price * params["qty"] / cand_lev, 8),
        )
        # 미체결 지정가를 가상 포지션으로 포함 → 슬롯/심볼/방향쏠림 한도에 예약분 반영,
        # 후보 자신의 증거금·명목도 총노출/명목노출 한도에 더해 판정 (사후 초과 방지)
        risk_positions = paper.get_positions() + paper.pending_as_positions()
        allowed, reason = risk.check_trade_allowed(
            current_positions=len(risk_positions),
            positions=risk_positions,
            symbol=res.symbol,
            direction=res.direction,
            candidate=candidate,
        )
        if not allowed:
            logger.info("[%s] 거래 차단: %s", res.symbol, reason)
            continue

        entry_session = get_active_session(datetime.now(timezone.utc))
        checks = dict(res.checks or {})
        checks["btc_aligned"] = alignment       # 학습/사후분석 태깅 (JSON 보존)
        if exec_mode == "maker":
            # 지정가 기준 실효 RR (신호의 시장가 RR이 아니라 실제 체결가 기하)
            lim_risk = abs(limit_price - sig.stop_loss)
            eff_rr = round(abs(sig.take_profit - limit_price) / lim_risk, 4) if lim_risk > 0 else sig.rr_ratio
            dbar = decision_bars.get(res.symbol)
            if dbar is None:
                _n = datetime.now(timezone.utc)
                dbar = _n.replace(minute=_n.minute - _n.minute % 15, second=0, microsecond=0)
            signal_key = f"{res.symbol}|{sig.direction}|{dbar.isoformat()}"
            oid = paper.place_pending_limit(
                symbol=res.symbol, direction=sig.direction, limit_price=limit_price,
                qty=params["qty"], stop_loss=sig.stop_loss, take_profit=sig.take_profit,
                leverage=params["leverage"], expiry_bars=exec_cfg.get("fill_expiry_bars", 8),
                score=res.score, checks=checks, entry_rr=eff_rr,
                entry_session=entry_session, signal_key=signal_key, decision_bar=dbar,
            )
            if oid is None:
                logger.info("[%s] 지정가 등록 안 됨 (중복/예약잔고 부족) key=%s", res.symbol, signal_key)
                continue
            entered += 1
            open_now.add(res.symbol)   # 같은 사이클 중복 등록 방지
            logger.info("[%s] 지정가 대기 등록 (메이커, limit=%.4f, RR=%.2f, key=%s)",
                        res.symbol, limit_price, eff_rr, signal_key)
        else:
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
