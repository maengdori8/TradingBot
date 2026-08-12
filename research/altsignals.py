from __future__ import annotations

# 대체 신호 연구 — ICT가 아닌 다른 가설을 룩어헤드 없이 탐색한다.
# 각 전략은 결정 봉까지의 데이터만으로 방향을 산출하며 study 재생 형식을 공유한다.
# 아래 목록과 출력은 legacy_non_evidence이며 자동 승급에 사용하지 않는다.
# 전략:
# - donchian: 직전 N봉 고점·저점 돌파
# - tsmom: EMA 추세
# - meanrev: 표준화 평균회귀
# - bbreak: ATR 변동성 돌파
# 사용: python3 research/altsignals.py --strategy donchian --param 96 --workers 6

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402  (_simulate, _san, DATA_DIR, MAX_HOLD 재사용)

logger = logging.getLogger("altsignals")
OUT_DIR = ROOT / "research" / "out"

# wfo.py 유니버스와 동일
UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "BNB/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT", "DOT/USDT:USDT", "TRX/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT",
]


# ──────────────────────────────────────────────────────────────────────
# 지표 (전부 결정봉까지만 — 룩어헤드 차단)
# ──────────────────────────────────────────────────────────────────────

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """지수이동평균 (현재봉 포함 — 종가 확정 시점에 계산 가능, 룩어헤드 아님)."""
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """ATR(n). True Range의 이동평균."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


# ──────────────────────────────────────────────────────────────────────
# 전략별 방향 산출 (벡터). dir[i] ∈ {+1, -1, 0}. 전부 i까지의 정보만.
# ──────────────────────────────────────────────────────────────────────

def sig_donchian(df: pd.DataFrame, n: int) -> np.ndarray:
    """직전 n봉(현재봉 제외) 고점 상향돌파=long, 저점 하향돌파=short."""
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prior_hi = pd.Series(high).rolling(n).max().shift(1).to_numpy()  # 현재봉 제외
    prior_lo = pd.Series(low).rolling(n).min().shift(1).to_numpy()
    d = np.zeros(len(df), dtype=np.int8)
    d[close > prior_hi] = 1
    d[close < prior_lo] = -1
    return d


def sig_tsmom(df: pd.DataFrame, slow: int) -> np.ndarray:
    """추세추종: close>EMA_slow & EMA_fast>EMA_slow = long, 역 = short. fast=slow//4."""
    close = df["close"].to_numpy()
    fast = max(2, slow // 4)
    ef, es = _ema(close, fast), _ema(close, slow)
    d = np.zeros(len(df), dtype=np.int8)
    d[(close > es) & (ef > es)] = 1
    d[(close < es) & (ef < es)] = -1
    return d


def sig_meanrev(df: pd.DataFrame, n: int, k: float = 2.0) -> np.ndarray:
    """평균회귀: z=(close-SMA_n)/std_n. z<-k=long, z>+k=short. 추세 약할 때만(EMA200 평탄)."""
    close = pd.Series(df["close"].to_numpy())
    sma = close.rolling(n).mean()
    std = close.rolling(n).std()
    z = ((close - sma) / std).to_numpy()
    d = np.zeros(len(df), dtype=np.int8)
    d[z < -k] = 1
    d[z > k] = -1
    return d


def sig_bbreak(df: pd.DataFrame, n: int, k: float = 1.5) -> np.ndarray:
    """변동성 돌파: 종가가 직전 SMA_n ± k*ATR 밖이면 그 방향으로 추세 진입."""
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    sma = pd.Series(close).rolling(n).mean().shift(1).to_numpy()
    atr = _atr(high, low, close)
    up = sma + k * atr
    dn = sma - k * atr
    d = np.zeros(len(df), dtype=np.int8)
    d[close > up] = 1
    d[close < dn] = -1
    return d


STRATEGIES = {
    "donchian": sig_donchian,
    "tsmom": sig_tsmom,
    "meanrev": sig_meanrev,
    "bbreak": sig_bbreak,
}


# ──────────────────────────────────────────────────────────────────────
# 재생 (심볼 1개) — 방향 벡터 → busy_until 게이트 → _simulate 출구 그리드
# ──────────────────────────────────────────────────────────────────────

def replay_one(symbol: str) -> list[dict]:
    """전략 방향 벡터를 만들어 신호+출구결과 행 리스트 반환 (룩어헤드 없음).

    전략/파라미터는 env(ALT_STRAT/ALT_PARAM)에서 읽는다 — ProcessPool 워커는 모듈을
    새로 import하므로 부모의 전역이 전달 안됨. env는 자식 프로세스에 상속되어 안전.
    """
    import os
    strat = os.environ.get("ALT_STRAT", "donchian")
    param = int(float(os.environ.get("ALT_PARAM", "96")))
    try:
        df15 = pd.read_pickle(study.DATA_DIR / f"{study._san(symbol)}_15m.pkl")
    except FileNotFoundError:
        return []
    if len(df15) < 800:
        return []

    direction = STRATEGIES[strat](df15, param)

    high = df15["high"].to_numpy()
    low = df15["low"].to_numpy()
    close = df15["close"].to_numpy()
    atr = _atr(high, low, close, 14)
    ts15 = df15.index

    start_i = 250  # 지표 워밍업
    rows: list[dict] = []
    busy_until = -1
    for i in range(start_i, len(df15) - 1):
        if i <= busy_until or direction[i] == 0:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        d = "long" if direction[i] == 1 else "short"
        entry = float(close[i])
        sim = study._simulate(entry, d, float(atr[i]), high, low, close, i + 1)
        if not sim:
            continue
        busy_until = sim.pop("resolve_idx")
        rows.append({
            "symbol": symbol,
            "ts": ts15[i].isoformat(),
            "direction": d,
            "score_raw": 100.0,        # 전략 무점수 → 필터 영향 없도록 고정
            "zone_both": False,
            "entry": entry,
            **sim,
        })
    return rows


def run(symbols: list[str], workers: int) -> pd.DataFrame:
    """전 심볼 병렬 재생 → signals 테이블."""
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(replay_one, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                logger.warning("%s 실패: %s", sym, e)
                rows = []
            all_rows.extend(rows)
            done += 1
            logger.info("재생 %d/%d — %s 신호 %d (누적 %d)", done, len(symbols), sym, len(rows), len(all_rows))
    return pd.DataFrame(all_rows)


def main() -> None:
    """단일 전략 재생 → CSV 저장 (이후 wfo/portfolio로 평가)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    parser.add_argument("--param", type=float, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cost", default="0.0021", help="STUDY_COST_PCT (taker 0.0021 / maker 0.0007)")
    args = parser.parse_args()

    import os
    os.environ["STUDY_COST_PCT"] = args.cost   # study._simulate 비용 (워커 상속)
    os.environ["ALT_STRAT"] = args.strategy     # 워커 상속
    os.environ["ALT_PARAM"] = str(args.param)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    df = run(UNIVERSE, args.workers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"alt_{args.strategy}_{int(args.param)}.csv"
    df.to_csv(out, index=False)
    logger.info("저장: %s (%d 신호)", out, len(df))


if __name__ == "__main__":
    main()
