from __future__ import annotations

"""Track C 페이퍼 러너 — 교차 거래소 펀딩 차익 (HL 숏 / Bybit 롱, 고정 방향).

매일 1회: 유니버스 각 코인의 전일 실현 펀딩 차이(Bybit−HL, 일별 합)를 수취분으로
기록하고, 양쪽 일봉 종가로 마크 베이시스 변화도 병기한다. 사양은
docs/XVENUE_ARBITRAGE_2026-08.md 사전 등록을 따른다. 페이퍼 전용.
"""

import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STATE = Path("logs/trackc_state.json")
HIST = Path("logs/trackc_history.csv")
UNIVERSE_F = Path("logs/trackc_universe.json")
COST_RT = 0.0012
DAY = 86400 * 1000


def _post_hl(body: dict, retries: int = 4):
    """HL info 호출."""
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (i + 1))
    return None


def _get_bybit(url: str, retries: int = 4):
    """Bybit 공개 호출."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,
                    headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (i + 1))
    return None


def build_universe() -> list[str]:
    """월 1회: 양쪽 상장 + 양쪽 일거래대금 ≥ $5M."""
    meta = _post_hl({"type": "metaAndAssetCtxs"})
    hl_vol = {a["name"]: float(c.get("dayNtlVlm", 0))
              for a, c in zip(meta[0]["universe"], meta[1]) if not a.get("isDelisted")}
    r = _get_bybit("https://api.bybit.com/v5/market/tickers?category=linear")
    by_vol = {t["symbol"][:-4]: float(t.get("turnover24h", 0))
              for t in r["result"]["list"]
              if t["symbol"].endswith("USDT") and "-" not in t["symbol"]}
    uni = sorted(c for c in hl_vol
                 if c in by_vol and hl_vol[c] >= 5e6 and by_vol[c] >= 5e6)
    return uni


def day_funding_diff(coin: str, day: pd.Timestamp) -> tuple[float, float] | None:
    """해당 UTC 일자의 (Bybit 일합, HL 일합) 실현 펀딩."""
    t0, t1 = int(day.timestamp() * 1000), int(day.timestamp() * 1000) + DAY
    hl = _post_hl({"type": "fundingHistory", "coin": coin, "startTime": t0 - 1})
    if hl is None:
        return None
    h = sum(float(x["fundingRate"]) for x in hl if t0 <= int(x["time"]) < t1)
    d = _get_bybit(f"https://api.bybit.com/v5/market/funding/history?category=linear"
                   f"&symbol={coin}USDT&limit=60")
    if d is None:
        return None
    b = sum(float(x["fundingRate"]) for x in d["result"]["list"]
            if t0 <= int(x["fundingRateTimestamp"]) < t1)
    return b, h


def main() -> None:
    """직전 닫힌 UTC 일자를 처리한다 (멱등)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    day = pd.Timestamp.now(tz="utc").normalize() - pd.Timedelta(days=1)
    if HIST.exists():
        h = pd.read_csv(HIST)
        if len(h) and str(h["day"].iloc[-1]) == str(day.date()):
            logger.info("%s 이미 처리됨", day.date())
            return

    state = json.loads(STATE.read_text()) if STATE.exists() else dict(equity=1.0, month="")
    month = f"{day.year}-{day.month:02d}"
    if state.get("month") != month or not UNIVERSE_F.exists():
        uni = build_universe()
        UNIVERSE_F.write_text(json.dumps(uni))
        prev = set(json.loads(UNIVERSE_F.read_text())) if state.get("month") else set()
        turn = len(set(uni) ^ prev) if prev else len(uni)
        state["equity"] -= (turn / max(len(uni), 1)) * COST_RT   # 재구성 비용
        state["month"] = month
        logger.info("유니버스 재구성: %d개", len(uni))
    uni = json.loads(UNIVERSE_F.read_text())

    diffs, got = [], 0
    for c in uni:
        r = day_funding_diff(c, day)
        if r is None:
            continue
        b, h = r
        diffs.append(h - b)          # HL 숏 수취 − Bybit 롱 지불
        got += 1
        time.sleep(0.1)
    if got < max(5, len(uni) // 2):
        logger.error("데이터 부족 (%d/%d) — fail-closed, 오늘 스킵", got, len(uni))
        return
    pnl = sum(diffs) / len(diffs)
    state["equity"] *= (1 + pnl)
    STATE.write_text(json.dumps(state))
    row = pd.DataFrame([dict(day=str(day.date()), equity=round(state["equity"], 8),
                             day_diff=round(pnl, 8), n_coins=got)])
    row.to_csv(HIST, mode="a", header=not HIST.exists(), index=False)
    logger.info("%s: 일수취 %+.5f%% (코인 %d) 자본 %.6f (ROE 환산은 ÷2)",
                day.date(), pnl * 100, got, state["equity"])


if __name__ == "__main__":
    main()
