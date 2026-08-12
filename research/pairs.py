from __future__ import annotations

# 공적분 페어 트레이딩 legacy 탐색. 자동 승급 후보군에는 포함하지 않는다.

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import research.study as study  # noqa: E402
from research.altsignals import UNIVERSE as MAJORS  # noqa: E402

logger = logging.getLogger("pairs")
OUT_DIR = ROOT / "research" / "out"

# 유동성 충분·역사 긴 종목 (공적분 안정성)
SYMS = list(MAJORS) + [f"{s}/USDT:USDT" for s in
                       ("LINK", "BCH", "XLM", "ETC", "FIL", "UNI", "AAVE", "ALGO")]


def load_panel(tf: str = "4h") -> pd.DataFrame:
    """로그 종가 패널 (공통 구간)."""
    series = {}
    for sym in SYMS:
        cache = study.DATA_DIR / f"{study._san(sym)}_{tf}.pkl"
        if cache.exists():
            series[sym] = np.log(pd.read_pickle(cache)["close"])
    panel = pd.DataFrame(series).dropna()
    return panel


def _coint_pval(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """엥글-그레인저 근사: OLS 잔차의 ADF p값 + 헤지비율 β. (statsmodels 없으면 단순회귀+잔차 자기상관)."""
    beta = np.polyfit(b, a, 1)[0]
    resid = a - beta * b
    try:
        from statsmodels.tsa.stattools import adfuller
        p = adfuller(resid, maxlag=1, regression="c", autolag=None)[1]
    except Exception:
        # 폴백: 잔차 1차 자기회귀계수 φ가 작을수록 평균회귀 (p 대용)
        dr = np.diff(resid)
        phi = np.polyfit(resid[:-1], dr, 1)[0]
        p = 1.0 if phi >= 0 else max(0.0, 1.0 + phi)  # φ<0이면 회귀 강함
    return float(p), float(beta)


def backtest(panel: pd.DataFrame, *, train_bars: int, test_bars: int, top_pairs: int,
             entry: float, exit_z: float, stop: float, cost: float, p_max: float = 0.05) -> dict:
    """WFO 페어트레이딩: train에서 공적분 상위 페어 선택+파라미터 추정 → test 거래."""
    cols = list(panel.columns)
    X = panel.to_numpy()
    dates = panel.index
    n = len(X)
    all_pairs = list(itertools.combinations(range(len(cols)), 2))

    pnl_series = []
    t = train_bars
    while t + test_bars <= n:
        tr = slice(t - train_bars, t)
        te = slice(t, t + test_bars)
        # train 공적분 상위 페어
        scored = []
        for (i, j) in all_pairs:
            a, b = X[tr, i], X[tr, j]
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                continue
            p, beta = _coint_pval(a, b)
            if beta <= 0:
                continue
            scored.append((p, i, j, beta))
        scored.sort(key=lambda x: x[0])
        chosen = [s for s in scored if s[0] < p_max][:top_pairs]  # 실제 공적분만

        for (p, i, j, beta) in chosen:
            sp_tr = X[tr, i] - beta * X[tr, j]
            mu, sd = sp_tr.mean(), sp_tr.std()
            if sd < 1e-9:
                continue
            sp_te = X[te, i] - beta * X[te, j]
            z = (sp_te - mu) / sd
            # test 구간 z기반 포지션 시뮬 (1페어, 다음봉 수익)
            pos = 0  # +1: 스프레드 롱(A롱B숏), -1: 스프레드 숏
            a_ret = np.diff(X[te, i])
            b_ret = np.diff(X[te, j])
            for s in range(len(z) - 1):
                newpos = pos
                if pos == 0:
                    if z[s] > entry:
                        newpos = -1
                    elif z[s] < -entry:
                        newpos = 1
                else:
                    if abs(z[s]) < exit_z or abs(z[s]) > stop:
                        newpos = 0
                if newpos != pos:
                    pnl_series.append(("__cost__", dates[t + s], 2 * cost))  # 양다리 거래비용
                    pos = newpos
                if pos != 0:
                    # 스프레드 수익 = pos*(a_ret - beta*b_ret), 명목 정규화(1+β)
                    r = pos * (a_ret[s] - beta * b_ret[s]) / (1 + beta)
                    pnl_series.append(("r", dates[t + s + 1], r))
        t += test_bars

    # 시각별 합산 → 포트폴리오 수익(동시 다수 페어 평균)
    rows = [(d, v, kind) for (kind, d, v) in pnl_series]
    if not rows:
        return {"trades": 0}
    df = pd.DataFrame(rows, columns=["date", "val", "kind"])
    rets = df[df["kind"] == "r"].groupby("date")["val"].mean()
    costs = df[df["kind"] == "__cost__"].groupby("date")["val"].sum()
    net = rets.subtract(costs, fill_value=0.0).sort_index()
    return _stats(net, test_bars)


def _stats(net: pd.Series, test_bars: int) -> dict:
    a = net.to_numpy()
    if len(a) == 0:
        return {"trades": 0}
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    per_year = 365.0 * 6  # 4h봉
    sh = float(a.mean() / a.std() * np.sqrt(per_year)) if a.std() > 0 else 0.0
    ho = net[net.index >= net.index.max() - pd.Timedelta(days=60)].to_numpy()
    return {"intervals": int(len(a)), "ann_ret": round(float(a.mean() * per_year), 4),
            "sharpe_ann": round(sh, 3), "equity_mult": round(eq, 3), "mdd": round(mdd, 4),
            "holdout_equity": round(float(np.prod(1 + ho)), 3) if len(ho) else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0007)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    panel = load_panel("4h")
    logger.info("패널: %d종목 × %d개 4h봉 (%s ~ %s)", panel.shape[1], panel.shape[0],
                panel.index.min(), panel.index.max())
    try:
        import statsmodels  # noqa: F401
        logger.info("statsmodels 사용 (ADF 공적분)")
    except ImportError:
        logger.info("statsmodels 없음 → 잔차 자기회귀 폴백")

    # train/test(4h봉수), 페어수, z 임계 — 표준값
    configs = [
        {"train_bars": 540, "test_bars": 180, "top_pairs": 5, "entry": 2.0, "exit_z": 0.5, "stop": 4.0},
        {"train_bars": 540, "test_bars": 180, "top_pairs": 10, "entry": 2.0, "exit_z": 0.5, "stop": 4.0},
        {"train_bars": 360, "test_bars": 120, "top_pairs": 5, "entry": 2.5, "exit_z": 0.75, "stop": 4.0},
        {"train_bars": 720, "test_bars": 180, "top_pairs": 8, "entry": 1.5, "exit_z": 0.5, "stop": 3.5},
    ]
    out = []
    for c in configs:
        for cost in (args.cost, 0.0021):
            r = backtest(panel, cost=cost, **c)
            r["config"] = {**c, "cost": cost}
            out.append(r)
            logger.info("tp%d entry%.1f tr%d cost%.4f → 인터벌%s 연%.1f%% Sharpe%.2f 자본x%.2f MDD%.1f%% 홀드x%s",
                        c["top_pairs"], c["entry"], c["train_bars"], cost, r.get("intervals"),
                        r.get("ann_ret", 0) * 100, r.get("sharpe_ann", 0), r.get("equity_mult", 0),
                        r.get("mdd", 0) * 100, r.get("holdout_equity"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pairs.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장: %s", OUT_DIR / "pairs.json")


if __name__ == "__main__":
    main()
