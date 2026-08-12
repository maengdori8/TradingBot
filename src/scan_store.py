from __future__ import annotations

# 봇의 관심종목·진입 스캔 결과를 기록하는 대시보드용 저장소.


import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SCAN_STATE_PATH = ROOT / "logs" / "scan_state.json"


def to_tradingview(symbol: str) -> str:
    """ccxt 심볼을 TradingView 심볼로 변환한다.

    예: 'BTC/USDT:USDT' -> 'BYBIT:BTCUSDT.P' (무기한 선물)

    Args:
        symbol: ccxt 형식 심볼

    Returns:
        TradingView 위젯용 심볼 문자열
    """
    base = symbol.split(":")[0]          # BTC/USDT
    pair = base.replace("/", "")          # BTCUSDT
    if ":" in symbol:                     # 무기한 선물
        return f"BYBIT:{pair}.P"
    return f"BYBIT:{pair}"


def save_scan_state(
    watchlist: list[dict],
    scanned_count: int,
    qualified_count: int,
    path: Path | None = None,
) -> None:
    """스캔 결과를 JSON 파일로 저장한다.

    Args:
        watchlist: 관심종목 리스트 (ScanResult.to_dict() 결과들, score 내림차순)
        scanned_count: 스캔한 전체 심볼 수
        qualified_count: 진입 확정된 심볼 수
        path: 저장 경로 (기본: logs/scan_state.json)
    """
    target = path or SCAN_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scanned_count": scanned_count,
        "qualified_count": qualified_count,
        "watchlist": watchlist,
    }
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(
            "스캔 상태 저장: 관심종목 %d개 (스캔 %d, 확정 %d)",
            len(watchlist), scanned_count, qualified_count,
        )
    except OSError as exc:
        logger.warning("스캔 상태 저장 실패: %s", exc)


def load_scan_state(path: Path | None = None) -> dict:
    """저장된 스캔 결과를 읽는다.

    Args:
        path: 읽을 경로 (기본: logs/scan_state.json)

    Returns:
        스캔 상태 딕셔너리. 파일이 없으면 빈 기본값.
    """
    target = path or SCAN_STATE_PATH
    if not target.exists():
        return {
            "updated_at": None, "scanned_count": 0,
            "qualified_count": 0, "watchlist": [],
        }
    try:
        with open(target, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("스캔 상태 로드 실패: %s", exc)
        return {
            "updated_at": None, "scanned_count": 0,
            "qualified_count": 0, "watchlist": [],
        }
