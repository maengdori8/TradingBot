from __future__ import annotations

"""트레이더 실력 지속성 연구 — Hyperliquid 전 지갑 전향적 추적.

가설: "과거 성적 상위 트레이더는 미래에도 상위인가?" (실력 지속성)
- 지속성이 없으면: 리더보드 상위는 로또 당첨자 → 카피 전략 폐기 (문헌의 다수 결과)
- 지속성이 있으면: 시점보존 카피-팔로우가 검증 가능한 전략 후보가 됨

왜 Hyperliquid인가: 전체 지갑(4.3만 개)의 손익이 공개된다. 승자만 보여주는
거래소 리더보드와 달리 패자를 포함한 전 모집단 → 생존편향 없는 코호트 구성 가능.

설계 (사전등록: docs/TRADER_PERSISTENCE_STUDY.md):
- 코호트 잠금: 활동 필터만 사용 (계좌 $10k+ AND 월 거래대금 $1M+) — 성과 필터
  금지 (성과로 고르면 그 자체가 선택편향).
- 매일 코호트 지갑의 계좌가치·월 손익·거래대금을 기록.
- T+30/60/90일에 순위 지속성(Spearman IC)과 상하위 십분위 스프레드를 1회 판정.
"""

import gzip
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
COHORT = Path("logs/trader_cohort.json.gz")
DAILY_DIR = Path("logs/trader_daily")

MIN_ACCOUNT_USD = 1e4       # 활동 필터 (성과 무관)
MIN_MONTH_VLM = 1e6
ROI_CLIP = (-0.95, 5.0)     # 입출금 왜곡 ROI 차단 (연구 시점에 재적용)


def fetch_leaderboard(timeout: int = 120) -> list[dict]:
    """전체 리더보드를 내려받는다 (~36MB)."""
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["leaderboardRows"]


def _win(row: dict, window: str, key: str) -> float:
    """windowPerformances에서 값을 꺼낸다."""
    for name, v in row.get("windowPerformances", []):
        if name == window:
            try:
                return float(v[key])
            except (KeyError, TypeError, ValueError):
                return float("nan")
    return float("nan")


def build_cohort(rows: list[dict]) -> pd.DataFrame:
    """활동 필터만으로 코호트를 만든다 (성과 필터 금지 — 선택편향 차단)."""
    rec = []
    for r in rows:
        try:
            av = float(r["accountValue"])
        except (KeyError, ValueError):
            continue
        vlm = _win(r, "month", "vlm")
        if av >= MIN_ACCOUNT_USD and vlm == vlm and vlm >= MIN_MONTH_VLM:
            rec.append(dict(address=r["ethAddress"], t0_account=av,
                            t0_month_roi=_win(r, "month", "roi"),
                            t0_month_pnl=_win(r, "month", "pnl"),
                            t0_month_vlm=vlm,
                            is_vault=bool(r.get("displayName"))))
    return pd.DataFrame(rec)


def snapshot_daily(rows: list[dict], cohort_addrs: set[str]) -> pd.DataFrame:
    """코호트 지갑의 당일 상태를 추출한다."""
    rec = []
    for r in rows:
        a = r.get("ethAddress")
        if a not in cohort_addrs:
            continue
        rec.append(dict(address=a, account=float(r.get("accountValue", "nan")),
                        day_pnl=_win(r, "day", "pnl"),
                        month_pnl=_win(r, "month", "pnl"),
                        month_roi=_win(r, "month", "roi"),
                        month_vlm=_win(r, "month", "vlm"),
                        alltime_pnl=_win(r, "allTime", "pnl")))
    return pd.DataFrame(rec)


def main() -> None:
    """일 1회: 코호트가 없으면 잠그고, 있으면 당일 스냅샷을 기록한다 (멱등)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    day = pd.Timestamp.now(tz="utc").date()
    out = DAILY_DIR / f"{day}.csv.gz"
    if out.exists():
        logger.info("%s 이미 기록됨 — 종료", day)
        return

    rows = fetch_leaderboard()
    logger.info("리더보드 %d개 지갑 수신", len(rows))

    if not COHORT.exists():
        c = build_cohort(rows)
        COHORT.parent.mkdir(exist_ok=True)
        with gzip.open(COHORT, "wt", encoding="utf-8") as f:
            json.dump(dict(locked_at=str(day), n=len(c),
                           filters=dict(min_account=MIN_ACCOUNT_USD, min_month_vlm=MIN_MONTH_VLM),
                           wallets=c.to_dict(orient="records")), f)
        logger.info("코호트 잠금: %d개 지갑 (%s)", len(c), day)

    with gzip.open(COHORT, "rt", encoding="utf-8") as f:
        cohort = json.load(f)
    addrs = {w["address"] for w in cohort["wallets"]}
    snap = snapshot_daily(rows, addrs)
    DAILY_DIR.mkdir(exist_ok=True)
    snap.to_csv(out, index=False, compression="gzip")
    logger.info("%s 스냅샷: 코호트 %d 중 %d개 관측 (%.0fKB)",
                day, len(addrs), len(snap), out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
