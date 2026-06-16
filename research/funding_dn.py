"""[H-011] 델타중립 펀딩 캐리 — 저회전 헤지 수확 백테스트.

가설(메커니즘): 펀딩 스프레드는 실재한다(기존 연구: 승률 80%). 방향성 버전은 가격리스크에
죽었고(죽은 6군), 고회전 리밸런싱은 비용에 죽었다(+0.097%/리밸 < 0.28%). 그러나 **델타중립
(롱 현물 + 숏 퍼프)으로 가격리스크를 헤지하고, 한 번 진입해 여러 8h 펀딩을 수확한 뒤 한 번
청산**하면, 진입/청산 비용(~0.3%)이 다수 펀딩 주기에 분산되어 비용 문턱을 넘을 수 있다.

신규성(5축): 데이터=현물+펀딩 추가 / 메커니즘=헤지캐리(방향 아님) / 비용구조=저회전 분산 /
유니버스=14메이저 / TF=8h 펀딩. 죽은 '방향성 펀딩 캐리'와 구조적으로 다름.

손익(숏퍼프+롱현물, 명목 N):
  펀딩수확% = Σ_8h funding_k  (펀딩>0이면 숏퍼프가 수취)
  가격손익% = spot_1/spot_0 − perp_1/perp_0   (헤지 잔차 = 베이시스 수렴/발산)
  순% = 펀딩수확 + 가격손익 − 왕복비용(4레그: 진입 현물+퍼프, 청산 현물+퍼프)

사전등록(결과 전 고정): 진입 funding ≥ 0.0001(/8h), 청산 funding < 0.00002 (캐리 소진).
비용 taker_dn=0.0031(퍼프0.055%+현물0.1%, 진입+청산), maker_dn=0.0020.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import research.study as study  # noqa: E402
import research.wfo as wfo  # noqa: E402
import research.stats_gate as sg  # noqa: E402

logger = logging.getLogger("funding_dn")
DATA = ROOT / "research" / "data"
OUT = ROOT / "research" / "out"

# 사전등록 고정
ENTRY_THR = 0.0001     # 진입: funding ≥ 0.01%/8h (명확한 양의 캐리)
EXIT_THR = 0.00002     # 청산: funding < 0.002%/8h (캐리 소진)
COST_TAKER = 0.0031    # 왕복 4레그 (퍼프 0.055%×2 + 현물 0.1%×2)
COST_MAKER = 0.0020    # 메이커 (퍼프 0.02%×2 + 현물 0.08%×2)


def _san(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "-")


def _price_at(df: pd.DataFrame, ts) -> float | None:
    """ts 이하 가장 최근 1h 종가."""
    i = df.index.searchsorted(ts, side="right") - 1
    return float(df["close"].iloc[i]) if i >= 0 else None


def simulate(symbol: str, entry_thr: float, exit_thr: float, cost_pct: float,
             min_hold: int = 0) -> list[dict]:
    """델타중립 캐리 시뮬 — 펀딩 타임스탬프 순회 상태기계.

    min_hold: 최소 보유 주기 강제(비용 분산용). 도달 전엔 펀딩 하락해도 청산 안 함.
    """
    try:
        perp = pd.read_pickle(DATA / f"{_san(symbol)}_1h.pkl")
        spot = pd.read_pickle(DATA / f"{_san(symbol)}_spot_1h.pkl")
        fund = pd.read_pickle(DATA / f"{_san(symbol)}_funding.pkl")
    except FileNotFoundError:
        return []
    if len(fund) < 10:
        return []

    trades: list[dict] = []
    in_pos = False
    e_ts = e_perp = e_spot = None
    f_acc = 0.0
    periods = 0
    for ts, row in fund.iterrows():
        f = float(row["funding"])
        pp = _price_at(perp, ts)
        sp = _price_at(spot, ts)
        if pp is None or sp is None or pp <= 0 or sp <= 0:
            continue
        if not in_pos:
            if f >= entry_thr:
                in_pos, e_ts, e_perp, e_spot = True, ts, pp, sp
                f_acc, periods = 0.0, 0
        else:
            f_acc += f           # 숏퍼프 = 펀딩>0 시 수취
            periods += 1
            if f < exit_thr and periods >= min_hold:
                price_pnl = (sp / e_spot) - (pp / e_perp)   # 헤지 잔차(베이시스)
                net = f_acc + price_pnl - cost_pct
                trades.append({
                    "symbol": symbol, "entry_ts": e_ts, "exit_ts": ts, "periods": periods,
                    "funding_pct": round(f_acc, 6), "price_pnl_pct": round(price_pnl, 6),
                    "cost_pct": cost_pct, "net_pct": round(net, 6),
                })
                in_pos = False
    return trades


def run_all(entry_thr: float, exit_thr: float, cost_pct: float, min_hold: int = 0) -> pd.DataFrame:
    """전 유니버스 캐리 거래 수집."""
    rows: list[dict] = []
    for sym in wfo.UNIVERSE:
        rows.extend(simulate(sym, entry_thr, exit_thr, cost_pct, min_hold))
    df = pd.DataFrame(rows)
    if len(df):
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
        df = df.sort_values("entry_ts").reset_index(drop=True)
    return df


def _gate_pct(r: np.ndarray, ts, n_tested: int) -> dict:
    """% 수익용 게이트 — stats_gate 구성요소 재사용(R 전용 r_min 문턱은 제외)."""
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return {"passed": False, "reason": "표본<2"}
    eff = sg.effective_sample(ts)
    n_eff = eff["n_days"]
    sh = sg.deflated_sharpe(r, n_tested, n_eff)
    bs = sg.block_bootstrap_ci(r, ts)
    tt = sg.cluster_t_pvalue(r, n_eff)
    bonf = sg.ALPHA / max(int(n_tested), 1)
    reasons = []
    if n_eff < 30:
        reasons.append(f"실효표본 n_eff={n_eff} < 30")
    if sh["dsr_prob"] < sg.DSR_PROB_MIN:
        reasons.append(f"DSR {sh['dsr_prob']} < {sg.DSR_PROB_MIN}")
    if bs["ci_low"] <= 0:
        reasons.append(f"부트스트랩 CI 하한 {bs['ci_low']} ≤ 0")
    if tt["p_one_sided"] >= bonf:
        reasons.append(f"본페로니 p {tt['p_one_sided']} ≥ α/N {bonf:.6f}")
    return {"passed": len(reasons) == 0, "reasons": reasons or ["통과"],
            "mean_pct": round(float(r.mean()), 6), "n": int(len(r)), "eff": eff,
            "sharpe": sh, "bootstrap": bs, "ttest": tt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-until", default="2026-04-16T00:00:00+00:00")
    parser.add_argument("--n-prior", type=int, default=850)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    hs = pd.Timestamp(args.holdout_until, tz="UTC")
    # 사전등록 2 config: base(즉시청산) + minhold21(7일 강제보유로 비용분산)
    CONFIGS = [("base", 0), ("minhold21", 21)]
    report = {"params": {"entry_thr": ENTRY_THR, "exit_thr": EXIT_THR},
              "holdout_frozen_until": str(hs), "configs_tested": [c[0] for c in CONFIGS],
              "results": {}}
    n_tested = args.n_prior + len(CONFIGS) + 1   # 2 config + 홀드아웃1

    any_promote = False
    for cfg_label, min_hold in CONFIGS:
        for cost_label, cost in (("taker", COST_TAKER), ("maker", COST_MAKER)):
            key = f"{cfg_label}/{cost_label}"
            df = run_all(ENTRY_THR, EXIT_THR, cost, min_hold)
            if len(df) == 0:
                report["results"][key] = {"verdict": "거래 0"}
                continue
            search = df[df["entry_ts"] < hs]
            hold = df[df["entry_ts"] >= hs]
            gate = _gate_pct(search["net_pct"].to_numpy(), search["entry_ts"].to_numpy(), n_tested)
            ho_mean = round(float(hold["net_pct"].mean()), 6) if len(hold) else None
            verdict = ("PROMOTE" if (gate["passed"] and ho_mean is not None and ho_mean > 0)
                       else "REJECT")
            if verdict == "PROMOTE" and cost_label == "taker":
                any_promote = True
            report["results"][key] = {
                "n_trades": len(df), "avg_periods_held": round(float(df["periods"].mean()), 1),
                "gross_funding_pct_mean": round(float(df["funding_pct"].mean()), 6),
                "price_pnl_pct_mean": round(float(df["price_pnl_pct"].mean()), 6),
                "net_pct_mean_all": round(float(df["net_pct"].mean()), 6),
                "holdout_net_mean": ho_mean, "stats_gate_passed": gate["passed"],
                "stats_gate_reasons": gate["reasons"], "sharpe": gate.get("sharpe"),
                "bootstrap": gate.get("bootstrap"), "verdict": verdict,
            }
            logger.info("[%s] 거래%d 보유%.1f주기 펀딩%.4f 가격%.4f 순%.4f → %s",
                        key, len(df), df["periods"].mean(), df["funding_pct"].mean(),
                        df["price_pnl_pct"].mean(), df["net_pct"].mean(), verdict)

    report["n_tested"] = n_tested
    report["final_verdict"] = "PROMOTE" if any_promote else "REJECT"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "funding_dn.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:2200])


if __name__ == "__main__":
    main()
