"""
ICT 전략 백테스트 CLI
사용법: python -m src.backtest [--symbol BTC/USDT:USDT] [--days 30] [--balance 1250]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import yaml

from src.exchange.bybit_client import MarketDataClient
from src.paper_trading.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest")


def load_config() -> dict:
    """config.yaml 로드."""
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """백테스트 CLI 진입점."""
    cfg = load_config()
    cap = cfg["capital"]
    risk = cfg["risk"]

    parser = argparse.ArgumentParser(description="ICT 전략 백테스트")
    parser.add_argument(
        "--symbol", default=cfg["exchange"]["symbols"][0],
        help="거래 심볼 (기본: config.yaml의 첫 번째 심볼)",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="백테스트 기간 (일, 기본: 7)",
    )
    parser.add_argument(
        "--balance", type=float,
        default=cap["total_capital"] * cap["trading_allocation"],
        help="초기 잔고 (기본: config 기반 트레이딩 자본)",
    )
    parser.add_argument(
        "--risk", type=float, default=cap["risk_per_trade"],
        help="거래당 리스크 비율 (기본: config 값)",
    )
    parser.add_argument(
        "--leverage", type=float, default=cfg["exchange"]["leverage"],
        help="레버리지 (기본: config 값)",
    )
    parser.add_argument(
        "--min-rr", type=float, default=risk["min_rr_ratio"],
        help="최소 R:R 비율 (기본: config 값)",
    )
    parser.add_argument(
        "--ignore-kz", action="store_true",
        help="(구버전 호환) 킬존 게이트는 2026-06 개정으로 폐지됨 — 이 옵션은 무시됨",
    )
    args = parser.parse_args()

    # 캔들 수 계산 (15분봉 기준)
    candles_15m = args.days * 24 * 4
    candles_1h = args.days * 24
    candles_4h = args.days * 6

    logger.info("=" * 60)
    logger.info("ICT 전략 백테스트 시작")
    logger.info("=" * 60)
    logger.info("심볼: %s", args.symbol)
    logger.info("기간: %d일 (15분봉 %d개)", args.days, candles_15m)
    logger.info("초기 잔고: %.2f USDT", args.balance)
    logger.info("리스크/트레이드: %.1f%%", args.risk * 100)
    logger.info("레버리지: %.0fx", args.leverage)
    logger.info("최소 R:R: %.1f", args.min_rr)
    logger.info("진입 시간: 24시간 (킬존 게이트 폐지, 2026-06 개정)")
    logger.info("=" * 60)

    # 시세 데이터 수집
    logger.info("시세 데이터 수집 중...")
    client = MarketDataClient()

    try:
        df_4h = client.fetch_ohlcv(args.symbol, "4h", limit=min(candles_4h, 200))
        df_1h = client.fetch_ohlcv(args.symbol, "1h", limit=min(candles_1h, 500))
        df_15m = client.fetch_ohlcv(args.symbol, "15m", limit=min(candles_15m, 1000))
    except RuntimeError as e:
        logger.error("시세 데이터 수집 실패: %s", e)
        sys.exit(1)

    logger.info(
        "데이터 수집 완료: 4H=%d, 1H=%d, 15M=%d 캔들",
        len(df_4h), len(df_1h), len(df_15m),
    )

    # 백테스트 실행
    bt = Backtester(
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        min_rr=args.min_rr,
        ignore_kill_zone=args.ignore_kz,
    )

    result = bt.run(df_4h, df_1h, df_15m, args.symbol)

    # 결과 출력
    print()
    print("=" * 60)
    print("  백테스트 결과")
    print("=" * 60)
    print(f"  총 거래 수:    {result.total_trades}")
    print(f"  승률:          {result.win_rate * 100:.1f}%")
    print(f"  총 손익:       {result.total_pnl:+.2f} USDT")
    print(f"  평균 거래:     {result.avg_trade_pnl:+.4f} USDT")
    print(f"  최대 낙폭:     {result.max_drawdown * 100:.2f}%")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:.4f}")
    print(f"  Profit Factor: {result.profit_factor:.4f}")
    if result.equity_curve:
        final = result.equity_curve[-1]
        ret = (final - args.balance) / args.balance * 100
        print(f"  최종 자본:     {final:.2f} USDT ({ret:+.1f}%)")
    print("=" * 60)

    if result.trades:
        print(f"\n  거래 상세 ({len(result.trades)}건):")
        for i, t in enumerate(result.trades, 1):
            print(
                f"    #{i} {t['direction'].upper():5s} "
                f"entry={t['entry_price']:.2f} "
                f"SL={t['stop_loss']:.2f} TP={t['take_profit']:.2f} "
                f"R:R={t['rr_ratio']:.1f}"
            )


if __name__ == "__main__":
    main()
