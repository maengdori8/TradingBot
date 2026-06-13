"""
포트폴리오 시뮬레이터 — '신호당 R'이 아니라 실거래 제약 하의 자본 곡선을 낸다.

신호당 기대값(R)은 모든 신호를 다 잡을 수 있다고 가정하지만, 실제 봇은:
- 동시 보유 N슬롯 제한 (config max_positions=4) → 슬롯 차면 이후 신호는 스킵 (선착순)
- 점수 차등 리스크 (risk_tiers) + 연패 단계 축소 (streak_risk_decay)
- 손절 심볼 재진입 쿨다운 (reentry_cooldown_hours)
이 시뮬은 그 제약을 시간순으로 적용해 '봇이 실제로 했을 거래'만 체결한다.

전제: signals.csv에 r_m{mult}_rr{rr}(순R) + hb_m{mult}_rr{rr}(보유캔들수)가 있어야 한다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "research" / "out"
logger = logging.getLogger("portfolio")

BAR_MS = 15 * 60 * 1000  # 15m


def load_cfg() -> dict:
    """config.yaml의 리스크/스캔 설정 로드."""
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _risk_pct(score: float, tiers: list[dict], default: float) -> float:
    """점수 차등 리스크 (config risk_tiers: [{min_score, risk_pct}, ...])."""
    for t in sorted(tiers, key=lambda x: -x["min_score"]):
        if score >= t["min_score"]:
            return t["risk_pct"]
    return default


def simulate(df: pd.DataFrame, exit_col: str, *, min_score: float, zone_both: bool,
             no_counter: bool, max_positions: int, tiers: list[dict],
             cooldown_h: float, streak_decay: bool, max_consec: int,
             notional_cap_mult: float, leverage: float) -> dict:
    """시간순 N슬롯 북 시뮬레이션. 청산을 시각순 정산 후 다음 진입 사이징.

    핵심: 진입 R은 사전계산값(룩어헤드 없음). 진입 시점에 결과를 알지만 자본 반영은
    청산 시각에 한다(연패 카운터/MDD가 실제 체결 순서를 따르도록).
    """
    hb_col = exit_col.replace("r_m", "hb_m", 1)
    d = df.copy()
    m = d["score_raw"] >= min_score
    if zone_both:
        m &= d["zone_both"]
    if no_counter:
        m &= d["btc_aligned"] != "counter"
    d = d[m].dropna(subset=[exit_col, hb_col]).sort_values("ts").reset_index(drop=True)
    if len(d) == 0:
        return {"trades": 0}

    ts_ms = (d["ts"].astype("int64") // 10**6).to_numpy()
    exit_ms = ts_ms + d[hb_col].to_numpy().astype("int64") * BAR_MS
    r_vals = d[exit_col].to_numpy()
    scores = d["score_raw"].to_numpy()
    syms = d["symbol"].to_numpy()

    equity, peak, mdd = 1.0, 1.0, 0.0
    consec_loss = 0
    open_slots: list[dict] = []     # {exit_ms, frac, sym, is_loss}
    cooldown: dict[str, int] = {}   # symbol -> 재진입 가능 ms
    taken = skip_slot = skip_cd = wins = 0
    realized: list[float] = []

    open_syms: set[str] = set()  # 현재 보유 심볼 (1심볼 1포지션 규칙)
    cd_ms = int(cooldown_h * 3600 * 1000)

    def settle_until(now: int) -> None:
        nonlocal equity, peak, mdd, consec_loss, wins
        due = sorted([s for s in open_slots if s["exit_ms"] <= now], key=lambda s: s["exit_ms"])
        for s in due:
            open_slots.remove(s)
            open_syms.discard(s["sym"])
            equity *= (1 + s["frac"])
            peak = max(peak, equity)
            mdd = max(mdd, (peak - equity) / peak if peak > 0 else 0)
            if s["is_loss"]:
                consec_loss += 1
                if cd_ms > 0:   # 손실은 '청산 시점'에 실현 → 그때부터 쿨다운 (룩어헤드 없음)
                    cooldown[s["sym"]] = int(s["exit_ms"] + cd_ms)
            else:
                consec_loss = 0
                wins += 1

    for i in range(len(d)):
        now = int(ts_ms[i])
        settle_until(now)
        sym = syms[i]
        if sym in open_syms:            # 이미 그 심볼 보유 중 → 신규 진입 안함
            skip_slot += 1
            continue
        if sym in cooldown and now < cooldown[sym]:
            skip_cd += 1
            continue
        if len(open_slots) >= max_positions:
            skip_slot += 1
            continue
        rp = _risk_pct(scores[i], tiers, tiers[-1]["risk_pct"] if tiers else 0.003)
        if streak_decay and consec_loss > 0:
            rp *= max(0.25, 0.5 ** min(consec_loss, max_consec))
        r = float(r_vals[i])
        open_slots.append({"exit_ms": int(exit_ms[i]), "frac": rp * r,
                           "sym": sym, "is_loss": r <= 0})
        open_syms.add(sym)
        taken += 1
        realized.append(r)

    settle_until(10**18)  # 잔여 청산

    rr = np.array(realized)
    return {
        "exit_col": exit_col,
        "filters": {"min_score": min_score, "zone_both": zone_both, "no_counter": no_counter},
        "trades": int(taken),
        "skipped_slot": int(skip_slot),
        "skipped_cooldown": int(skip_cd),
        "winrate": round(float((rr > 0).mean()), 4) if len(rr) else 0.0,
        "mean_r": round(float(rr.mean()), 4) if len(rr) else 0.0,
        "equity_mult": round(float(equity), 3),
        "mdd": round(float(mdd), 4),
        "period": [str(d["ts"].min()), str(d["ts"].max())],
    }


def main() -> None:
    """signals.csv에 여러 출구/필터 조합으로 포트폴리오 시뮬을 돌려 비교 출력."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", default=str(OUT_DIR / "wfo_signals.csv"))
    parser.add_argument("--exit", default="r_m2_rr2.5")
    parser.add_argument("--min-score", type=float, default=70)
    parser.add_argument("--since", default=None, help="이 시각 이후 신호만 (홀드아웃 평가용)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    import research.wfo as wfo
    df = wfo._load_and_enrich(Path(args.signals))
    if args.since:
        df = df[df["ts"] >= pd.Timestamp(args.since, tz="UTC")].reset_index(drop=True)

    cfg = load_cfg()
    risk = cfg.get("risk", {})
    tiers = risk.get("risk_tiers", [{"min_score": 70, "risk_pct": 0.005}])
    res = simulate(
        df, args.exit,
        min_score=args.min_score, zone_both=False, no_counter=False,
        max_positions=risk.get("max_positions", 4), tiers=tiers,
        cooldown_h=risk.get("reentry_cooldown_hours", 8),
        streak_decay=risk.get("streak_risk_decay", True),
        max_consec=2, notional_cap_mult=risk.get("max_notional_mult", 3.0),
        leverage=1.0,
    )
    logger.info("포트폴리오 시뮬 결과: %s", res)


if __name__ == "__main__":
    main()
