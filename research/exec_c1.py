"""[H-007 / C1] 세션 단기 평균회귀 — 벡터화 재생 (ICT 스캔 불필요).

가설(메커니즘): 암호화폐 단기(분~시간)는 급변 후 과확장→평균회귀 경향이 있다. k봉 누적이동이
ATR 대비 Z 이상이면 **역방향(페이드)** 진입 → 단기 반전 수확. 가격 효율성의 미시구조 한계
(유동성 공백·과민반응)에서 나오는 엣지 가설. ICT 컨플루언스/죽은 meanrev_96(개별종목 96기간
종가회귀)과 **신호메커니즘 축에서 구분**: 세션조건부 + ATR정규화 + k봉 모멘텀 페이드.

신규성(5축): 데이터=가격동일 / **신호메커니즘=신규(모멘텀페이드)** / 비용구조=메이커·테이커 양측 /
유니버스=14메이저 / TF=15m. 죽은 6군과 신호메커니즘 축에서 분리.

사전등록(결과 전 고정): k=4봉(1h) 누적이동, Z=2.0·ATR, 역방향 진입, 진입=신호봉 다음봉 시가근사,
SL×RR 그리드, 메이커0.0007/테이커0.0021 양측, 비중복(기준출구 해소까지 슬롯점유), 24h(세션무관 primary).
출력: r_m*_rr*(메이커) / tkr_m*_rr*(테이커) / score_raw=|move/ATR| / placebo_r_ref.
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402
from research.exec_a1 import _grid_from_entry, MAKER_COST, TAKER_COST  # noqa: E402

logger = logging.getLogger("exec_c1")
OUT_DIR = ROOT / "research" / "out"

# 사전등록 고정
K_BARS = 4          # 누적이동 룩백 (4×15m = 1h)
Z_THRESH = 2.0      # ATR 대비 과확장 문턱
ATR_WIN = 14
PLACEBO_SEED = 20260616
SL_MULTS = study.SL_MULTS
RR_LEVELS = study.RR_LEVELS
MAX_HOLD = study.MAX_HOLD


def _base_resolve(entry: float, direction: str, atr: float,
                  highs: np.ndarray, lows: np.ndarray, start: int) -> int:
    """기준출구(m2.0/rr2.5) 해소 봉 — 슬롯 점유용(룩어헤드 없음, 진입 이후만)."""
    risk = atr * 2.0
    end = min(start + MAX_HOLD, len(highs))
    h, lo = highs[start:end], lows[start:end]
    if len(h) == 0 or risk <= 0:
        return start
    if direction == "long":
        sl_hits = np.nonzero(lo <= entry - risk)[0]
        tp_hits = np.nonzero(h >= entry + risk * 2.5)[0]
    else:
        sl_hits = np.nonzero(h >= entry + risk)[0]
        tp_hits = np.nonzero(lo <= entry - risk * 2.5)[0]
    si = sl_hits[0] if len(sl_hits) else 10**9
    ti = tp_hits[0] if len(tp_hits) else 10**9
    done = min(si, ti)
    return start + (done if done < 10**9 else (end - start - 1))


def replay_symbol_c1(symbol: str) -> list[dict]:
    """심볼 1개: 벡터화 모멘텀-페이드 신호 + 그리드 시뮬."""
    from src.strategy.kill_zone import get_active_session as real_session
    try:
        df15 = pd.read_pickle(study.DATA_DIR / f"{study._san(symbol)}_15m.pkl")
    except FileNotFoundError:
        return []
    if len(df15) < 800:
        return []

    ts15 = df15.index
    highs = df15["high"].to_numpy()
    lows = df15["low"].to_numpy()
    closes = df15["close"].to_numpy()
    atr = (df15["high"] - df15["low"]).rolling(ATR_WIN).mean().to_numpy()
    move = closes - np.concatenate([np.full(K_BARS, np.nan), closes[:-K_BARS]])
    z = np.where(atr > 0, move / atr, np.nan)

    rng = np.random.default_rng(PLACEBO_SEED + hash(symbol) % 10000)
    rows: list[dict] = []
    busy_until = ATR_WIN + K_BARS
    for i in range(ATR_WIN + K_BARS, len(df15) - 1):
        if i <= busy_until or np.isnan(z[i]) or atr[i] <= 0:
            continue
        # 신호 발생(문턱 교차) + 역방향
        if z[i] >= Z_THRESH and z[i - 1] < Z_THRESH:
            direction = "short"
        elif z[i] <= -Z_THRESH and z[i - 1] > -Z_THRESH:
            direction = "long"
        else:
            continue
        entry = float(closes[i])
        a = float(atr[i])
        sim_start = i + 1
        mk = _grid_from_entry(entry, direction, a, MAKER_COST, highs, lows, closes, sim_start, "r")
        tk = _grid_from_entry(entry, direction, a, TAKER_COST, highs, lows, closes, sim_start, "tkr")
        if not mk:
            continue
        # 플라시보: 방향 무작위
        pdir = direction if rng.random() < 0.5 else ("short" if direction == "long" else "long")
        plc = _grid_from_entry(entry, pdir, a, MAKER_COST, highs, lows, closes, sim_start, "plc")
        ts = ts15[i]
        rows.append({
            "symbol": symbol, "ts": ts.isoformat(), "direction": direction,
            "score_raw": round(abs(float(z[i])), 2),
            "session": real_session(ts.to_pydatetime()) or "none",
            "zone_both": False, "filled": 1,
            "placebo_r_ref": plc.get("plc_m2_rr2.5"),
            **mk, **tk,
        })
        busy_until = _base_resolve(entry, direction, a, highs, lows, sim_start)
    return rows


def run(symbols: list[str], workers: int) -> pd.DataFrame:
    """전 심볼 병렬."""
    all_rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(replay_symbol_c1, s): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s 실패: %s", sym, e)
                rows = []
            done += 1
            logger.info("재생 %d/%d — %s 행 %d (누적 %d)", done, len(symbols), sym, len(rows), len(all_rows))
    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "c1_signals.csv", index=False)
    return df


def main() -> None:
    """C1 재생."""
    import argparse
    import research.wfo as wfo
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    symbols = args.symbols or wfo.UNIVERSE
    df = run(symbols, args.workers)
    logger.info("총 %d신호", len(df))


if __name__ == "__main__":
    main()
