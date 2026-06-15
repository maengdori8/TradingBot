"""[H-001 / A1] 메이커 지정가 실행 모델 — 룩어헤드 차단 재생 변형.

가설(메커니즘): 죽은 ICT 신호가 음수였던 1차 원인은 taker 왕복비용(0.21%≈0.16~0.20R 드래그,
docs/WFO_2026-06.md)이다. 그렇다면 maker 지정가 실행(0.07%)이 어떤 출구조합을 순양수로
되살리는가? 단, 단순히 비용만 낮추면 안 된다(GOAL_LOOP §5-4): 지정가는

  ① 가격이 지정가에 닿아야만 체결된다(미체결=거래없음). 유리한 오프셋일수록 체결률↓.
  ② 체결된 거래는 역선택된다 — '내 롱 지정가가 채워졌다'는 곧 '가격이 먼저 내려왔다'는 뜻.

이 두 효과를 명시 모델링한다. 기존 study.replay_symbol을 미러링하되 진입을 시장가→지정가로
바꾸고, 체결창 K봉 내 미체결 신호는 버린다(비용 0). 동일 신호에 taker 시장가 베이스라인과
방향셔플 플라시보(음성통제)를 나란히 계산한다.

출력 컬럼:
  r_m{mult}_rr{rr}   = MAKER 지정가(오프셋 PRIMARY) 순R   ← wfo.py가 그대로 평가
  tkr_m{mult}_rr{rr} = TAKER 시장가 베이스라인 순R
  mk0_m{mult}_rr{rr} = MAKER 오프셋 0 순R (진단)
  filled, fill_delay_bars, offset_atr, placebo_r_ref
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402

logger = logging.getLogger("exec_a1")

OUT_DIR = ROOT / "research" / "out"

# ── 사전등록 고정 파라미터 (검증 전 고정 — 결과 보고 바꾸면 데이터스누핑) ──
MAKER_COST = float(os.getenv("A1_MAKER_COST", "0.0007"))   # 메이커 왕복 (≈0.035%×2)
TAKER_COST = float(os.getenv("A1_TAKER_COST", "0.0021"))   # 테이커 왕복 (베이스라인)
OFFSET_PRIMARY = 0.25     # 사전등록 PRIMARY: 지정가 = 진입 -0.25·ATR (롱 기준 유리)
FILL_WINDOW = 8           # 체결창: 8×15m = 2시간. 내 미체결 시 신호 폐기
PLACEBO_SEED = 20260615   # 방향셔플 음성통제 시드 (고정)

SL_MULTS = study.SL_MULTS
RR_LEVELS = study.RR_LEVELS
MAX_HOLD = study.MAX_HOLD


def _grid_from_entry(entry: float, direction: str, atr: float, cost_pct: float,
                     highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                     start: int, prefix: str) -> dict:
    """진입가에서 (SL배수×RR) 그리드 순R 계산 (study._simulate 로직, 비용·진입가 파라미터화).

    동시터치=손절 우선(보수적). 미해결 시 종가 청산.
    """
    end = min(start + MAX_HOLD, len(closes))
    h = highs[start:end]
    lo = lows[start:end]
    if len(h) == 0:
        return {}
    out: dict = {}
    for mult in SL_MULTS:
        risk = atr * mult
        if risk <= 0:
            continue
        cost_r = cost_pct * entry / risk
        if direction == "long":
            sl = entry - risk
            sl_hits = np.nonzero(lo <= sl)[0]
        else:
            sl = entry + risk
            sl_hits = np.nonzero(h >= sl)[0]
        sl_i = sl_hits[0] if len(sl_hits) else 10**9
        for rr in RR_LEVELS:
            if direction == "long":
                tp = entry + risk * rr
                tp_hits = np.nonzero(h >= tp)[0]
            else:
                tp = entry - risk * rr
                tp_hits = np.nonzero(lo <= tp)[0]
            tp_i = tp_hits[0] if len(tp_hits) else 10**9
            if sl_i <= tp_i and sl_i < 10**9:
                r = -1.0
            elif tp_i < sl_i:
                r = float(rr)
            else:
                last = closes[end - 1]
                r = (last - entry) / risk if direction == "long" else (entry - last) / risk
            out[f"{prefix}_m{mult:g}_rr{rr:g}"] = round(r - cost_r, 4)
    return out


def _limit_fill(direction: str, limit: float, offset_atr: float, atr: float,
                highs: np.ndarray, lows: np.ndarray, start: int) -> int | None:
    """지정가가 체결창 내 체결되는 첫 봉 인덱스. 미체결이면 None.

    롱: 가격이 limit까지 '내려와야' 체결 (low<=limit) → 역선택 내장.
    숏: 가격이 limit까지 '올라와야' 체결 (high>=limit).
    오프셋 0이면 진입봉 다음봉부터 즉시 체결 가능(시장 근처).
    """
    end = min(start + FILL_WINDOW, len(highs))
    if direction == "long":
        hits = np.nonzero(lows[start:end] <= limit)[0]
    else:
        hits = np.nonzero(highs[start:end] >= limit)[0]
    return int(start + hits[0]) if len(hits) else None


def replay_symbol_maker(symbol: str) -> list[dict]:
    """심볼 1개 재생: 메이커 지정가 체결 모델 + 테이커 베이스라인 + 플라시보."""
    import src.strategy.signal_engine as SE
    from src.strategy.kill_zone import is_in_kill_zone as real_kz
    from src.strategy.kill_zone import get_active_session as real_session
    from src.strategy import load_strategy_params

    SE.datetime = study._FakeDT
    SE.is_in_kill_zone = lambda dt: True
    SE.get_active_session = lambda dt: real_session(dt) or "none"

    try:
        df15 = pd.read_pickle(study.DATA_DIR / f"{study._san(symbol)}_15m.pkl")
        df1h = pd.read_pickle(study.DATA_DIR / f"{study._san(symbol)}_1h.pkl")
        df4h = pd.read_pickle(study.DATA_DIR / f"{study._san(symbol)}_4h.pkl")
    except FileNotFoundError:
        return []
    if len(df15) < 800 or len(df4h) < 120:
        return []

    atr_mult = load_strategy_params()["atr"]["multiplier"]
    ts15 = df15.index
    highs = df15["high"].to_numpy()
    lows = df15["low"].to_numpy()
    closes = df15["close"].to_numpy()

    dec = ts15.values + np.timedelta64(15, "m")
    idx1h = np.searchsorted(df1h.index.values, dec - np.timedelta64(1, "h"), side="right") - 1
    idx4h = np.searchsorted(df4h.index.values, dec - np.timedelta64(4, "h"), side="right") - 1
    start_i = max(int(np.searchsorted(idx4h, 100)), 100)

    rng = np.random.default_rng(PLACEBO_SEED + hash(symbol) % 10000)
    rows: list[dict] = []
    busy_until = -1
    for i in range(start_i, len(df15)):
        if i <= busy_until:
            continue
        j1, j4 = idx1h[i], idx4h[i]
        if j1 < 100 or j4 < 100:
            continue
        ts = ts15[i].to_pydatetime()
        study._FakeDT._now = ts
        w15 = df15.iloc[i - 99: i + 1]
        w1h = df1h.iloc[j1 - 99: j1 + 1]
        w4h = df4h.iloc[j4 - 99: j4 + 1]
        price = closes[i]
        try:
            res = SE.scan_symbol(w4h, w1h, w15, symbol, float(price),
                                 min_rr=2.0, min_score=0.0, require_volume=False)
        except Exception:  # noqa: BLE001 (전략 스캔 내부 예외 — 해당 봉 스킵)
            continue
        sig = res.signal
        if sig is None:
            continue

        direction = sig.direction
        entry_mkt = float(sig.entry_price)
        risk_d = abs(sig.entry_price - sig.stop_loss)
        atr = risk_d / atr_mult if atr_mult > 0 else risk_d
        if atr <= 0:
            continue

        # ── 진입봉 i 종가 이후(i+1)부터 체결/시뮬 (룩어헤드 차단) ──
        sim_start = i + 1
        # 테이커 시장가 베이스라인: i+1 시가 근사 = 신호가 즉시 체결
        tkr = _grid_from_entry(entry_mkt, direction, atr, TAKER_COST,
                               highs, lows, closes, sim_start, "tkr")

        row = {
            "symbol": symbol, "ts": ts15[i].isoformat(), "direction": direction,
            "score_raw": round(res.score, 1),
            "kz_true": bool(real_kz(ts)),
            "zone_both": "FVG+OB" in (res.reason or ""),
            "entry_mkt": entry_mkt, "offset_atr": OFFSET_PRIMARY,
            **tkr,
        }

        # ── 메이커 오프셋 0 (진단): 지정가=시장가, 거의 즉시 체결, maker 비용 ──
        f0 = _limit_fill(direction, entry_mkt, 0.0, atr, highs, lows, sim_start)
        if f0 is not None:
            row.update(_grid_from_entry(entry_mkt, direction, atr, MAKER_COST,
                                        highs, lows, closes, f0, "mk0"))

        # ── 메이커 PRIMARY (오프셋 0.25·ATR 유리): 체결창 내 닿아야 체결 ──
        limit = entry_mkt - OFFSET_PRIMARY * atr if direction == "long" \
            else entry_mkt + OFFSET_PRIMARY * atr
        fp = _limit_fill(direction, limit, OFFSET_PRIMARY, atr, highs, lows, sim_start)
        filled = fp is not None
        row["filled"] = int(filled)
        if filled:
            row["fill_delay_bars"] = int(fp - sim_start)
            # PRIMARY 메이커 결과를 r_m*_rr* 로 저장 → wfo.py가 그대로 평가
            grid = _grid_from_entry(limit, direction, atr, MAKER_COST,
                                    highs, lows, closes, fp, "r")
            row.update(grid)
            # 플라시보(음성통제): 방향 셔플 — 같은 지정가/체결로직, 방향만 무작위
            pdir = direction if rng.random() < 0.5 else (
                "short" if direction == "long" else "long")
            plimit = entry_mkt - OFFSET_PRIMARY * atr if pdir == "long" \
                else entry_mkt + OFFSET_PRIMARY * atr
            pf = _limit_fill(pdir, plimit, OFFSET_PRIMARY, atr, highs, lows, sim_start)
            if pf is not None:
                pgrid = _grid_from_entry(plimit, pdir, atr, MAKER_COST,
                                         highs, lows, closes, pf, "plc")
                # 기준 출구(m2.0_rr2.5)의 플라시보 R 참조값
                row["placebo_r_ref"] = pgrid.get("plc_m2_rr2.5")
            # 슬롯 점유: PRIMARY 기준출구 해소까지 — 간단히 진입 후 1봉 점유로 근사
            busy_until = fp
        else:
            busy_until = i  # 미체결: 다음 봉부터 다시 탐색

        rows.append(row)
    return rows


def run(symbols: list[str], workers: int) -> pd.DataFrame:
    """전 심볼 병렬 재생 → research/out/a1_signals.csv."""
    all_rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(replay_symbol_maker, s): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s 재생 실패: %s", sym, e)
                rows = []
            done += 1
            logger.info("재생 %d/%d — %s 행 %d (누적 %d)",
                        done, len(symbols), sym, len(rows), len(all_rows))
    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "a1_signals.csv", index=False)
    return df


def main() -> None:
    """A1 재생 실행."""
    import argparse
    import research.wfo as wfo
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    symbols = args.symbols or wfo.UNIVERSE
    df = run(symbols, args.workers)
    fill_rate = df["filled"].mean() if len(df) else 0.0
    logger.info("총 %d신호, 메이커 체결률 %.1f%%", len(df), 100 * fill_rate)


if __name__ == "__main__":
    main()
