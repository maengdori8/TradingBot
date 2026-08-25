from __future__ import annotations

"""현재 시점의 전략 판단 — 사전등록된 알고리즘을 그대로 적용한다.

Codex 7라운드 지적 반영:
- `resample().sum()`에 min_count=1을 주어 빈 날을 0으로 만들지 않는다.
- 미완료 당일을 제외하고, 등록된 1일 정보 지연을 적용한다.
- 유동성은 24h 회전율이 아니라 '양다리 중 얇은 쪽의 30일 중앙 거래대금'을 쓴다.
- 유니버스는 ADV 상위 N으로 먼저 확정하고, 그 안에서만 허들을 적용한다.
"""

import logging
import time
from dataclasses import dataclass

import ccxt
import numpy as np
import pandas as pd

from carrybot.research.ledger import LedgerConfig

logger = logging.getLogger(__name__)


def _retry(fn, *a, **k):
    """공개 엔드포인트 호출 재시도."""
    for i in range(6):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            if i == 5:
                return None
            time.sleep(1.5 * (i + 1))
    return None


@dataclass(frozen=True)
class SignalRow:
    """한 심볼의 현재 판단."""

    symbol: str
    carry_ann: float
    excess_ann: float
    adv_usd: float
    in_universe: bool
    qualifies: bool
    reason: str


def _daily_funding(ex, mid: str, days: int) -> pd.Series | None:
    """정산 완료된 펀딩을 일별 합산으로 반환한다 (미완료 당일 제외)."""
    r = _retry(ex.publicGetV5MarketFundingHistory,
               {"category": "linear", "symbol": mid, "limit": 200})
    if not r:
        return None
    lst = r.get("result", {}).get("list", [])
    if not lst:
        return None
    s = (pd.DataFrame([(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst],
                      columns=["ts", "f"])
         .assign(ts=lambda d: pd.to_datetime(d.ts, unit="ms", utc=True))
         .drop_duplicates("ts").set_index("ts").sort_index()["f"])
    d = s.resample("D").sum(min_count=1)                 # 빈 날은 NaN 유지
    today = pd.Timestamp.now(tz="utc").normalize()
    d = d[d.index < today]                                # 미완료 당일 제외
    return d.tail(days)


def _adv_thin_leg(ex, base: str, days: int = 30) -> float:
    """양다리 중 얇은 쪽의 30일 중앙 거래대금 (전일까지)."""
    out = []
    for cat in ("linear", "spot"):
        r = _retry(ex.publicGetV5MarketKline,
                   {"category": cat, "symbol": f"{base}USDT", "interval": "D", "limit": days + 2})
        if not r:
            return 0.0
        lst = r.get("result", {}).get("list", [])
        if not lst:
            return 0.0
        # v5 kline: [start, open, high, low, close, volume, turnover]
        rows = sorted(((int(x[0]), float(x[6])) for x in lst), key=lambda z: z[0])
        today_ms = int(pd.Timestamp.now(tz="utc").normalize().timestamp() * 1000)
        rows = [v for ts, v in rows if ts < today_ms][-days:]
        out.append(float(np.median(rows)) if rows else 0.0)
    return min(out) if out else 0.0


def current_signals(bases: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK"),
                    cfg: LedgerConfig | None = None) -> pd.DataFrame:
    """사전등록 알고리즘으로 현재 종목별 판단을 계산한다."""
    cfg = cfg or LedgerConfig()
    ex = ccxt.bybit({"enableRateLimit": True})
    _retry(ex.load_markets)

    raw = []
    for b in bases:
        d = _daily_funding(ex, f"{b}USDT", cfg.lookback_days)
        adv = _adv_thin_leg(ex, b)
        if d is None or len(d) < cfg.lookback_days * cfg.min_valid_ratio:
            raw.append((b, np.nan, adv, "펀딩 관측 부족"))
            continue
        vals = np.sort(d.dropna().to_numpy())
        k = int(len(vals) * 0.10)
        if k > 0 and len(vals) - 2 * k >= 3:
            vals = vals[k:len(vals) - k]
        raw.append((b, float(np.mean(vals)) * 365, adv, ""))

    eligible = [(b, c, a) for b, c, a, _ in raw if a >= cfg.min_adv_usd and not np.isnan(c)]
    universe = {b for b, _, _ in sorted(eligible, key=lambda z: -z[2])[:cfg.universe_top_n]}

    rows = []
    for b, carry, adv, note in raw:
        excess = carry - cfg.cash_rate if not np.isnan(carry) else np.nan
        in_uni = b in universe
        ok = bool(in_uni and not np.isnan(excess) and excess >= cfg.hurdle_ann)
        if note:
            reason = note
        elif adv < cfg.min_adv_usd:
            reason = f"유동성 미달 (${adv/1e6:.0f}M < ${cfg.min_adv_usd/1e6:.0f}M)"
        elif not in_uni:
            reason = f"ADV 상위 {cfg.universe_top_n} 밖"
        elif excess < cfg.hurdle_ann:
            reason = f"초과캐리 {excess*100:.2f}% < 허들 {cfg.hurdle_ann*100:.2f}%"
        else:
            reason = "진입 가능"
        rows.append(SignalRow(b, carry, excess, adv, in_uni, ok, reason).__dict__)
    return pd.DataFrame(rows).sort_values("adv_usd", ascending=False)


def main() -> None:
    """현재 신호를 출력한다."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = LedgerConfig()
    df = current_signals(cfg=cfg)
    print(f"\n[사전등록 설정] 현금 {cfg.cash_rate*100:.1f}%  허들(초과) {cfg.hurdle_ann*100:.2f}%  "
          f"룩백 {cfg.lookback_days}일  최소보유 {cfg.min_hold_days}일  유니버스 ADV상위 {cfg.universe_top_n}")
    print("=" * 100)
    print(f"{'심볼':>6s} {'30일캐리':>10s} {'초과':>9s} {'30일ADV(얇은다리)':>18s} {'유니버스':>8s} {'판정':>6s}  사유")
    for _, r in df.iterrows():
        c = f"{r.carry_ann*100:+9.2f}%" if not np.isnan(r.carry_ann) else "         -"
        e = f"{r.excess_ann*100:+8.2f}%" if not np.isnan(r.excess_ann) else "        -"
        print(f"{r.symbol:>6s} {c} {e} ${r.adv_usd/1e6:16.0f}M {'O' if r.in_universe else 'X':>8s} "
              f"{'진입' if r.qualifies else '대기':>6s}  {r.reason}")
    n = int(df.qualifies.sum())
    print("=" * 100)
    print(f"결론: {'진입 대상 %d개' % n if n else '진입 대상 없음 → 전액 현금 보유'}")


if __name__ == "__main__":
    main()
