from __future__ import annotations

"""전향적 시장·실행 기록기.

목적 (Codex 6라운드 최우선 지시): 역방향 캐리의 핵심 입력인 '시점보존 차입금리와
차입 가능량'은 과거 이력이 존재하지 않는다. 오늘의 값을 과거에 소급 적용하는 것은
룩어헤드이며, 차입난은 음수 펀딩과 동시에 발생하므로 편향의 방향도 나쁘다.
따라서 지금부터 기록을 시작해야만 그 가설을 검증할 수 있다.

기록 항목:
- 펀딩: 직전 정산값, 다음 정산 예측값
- 가격: 현물/perp 호가와 의도한 규모에서의 체결 가능 가격
- 차입: 연율 차입금리, 최대 차입량, 담보인정비율
- 리스크: 리스크티어 유지증거금률
- 벤치마크: 현금 대안 수익률(설정값)

계정 고유값(실제 VIP 등급 금리·한도)은 인증이 필요하다. 미인증 시 공개 'No VIP'
값을 기록하되 `authenticated=False`로 표시해 이후 분석에서 구분한다.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ccxt

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    payload       TEXT NOT NULL,
    authenticated INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ts, symbol, kind)
);
CREATE INDEX IF NOT EXISTS idx_obs_kind_ts ON observations(kind, ts);
"""


@dataclass
class RecorderConfig:
    """기록기 설정."""

    db_path: str = "logs/carry_recorder.db"
    bases: tuple[str, ...] = ("BTC", "ETH")
    probe_notional_usd: float = 10_000.0   # 이 규모에서의 체결 가능 가격을 계산
    depth_limit: int = 50


def _utcnow() -> str:
    """UTC ISO8601 타임스탬프."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _retry(fn, *a, **k):
    """공개 엔드포인트 호출 재시도."""
    for i in range(5):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001 - 네트워크 계열 광범위
            if i == 4:
                logger.warning("호출 실패: %s", str(exc)[:160])
                return None
            time.sleep(1.5 * (i + 1))
    return None


def _walk_book(levels: list, notional: float) -> tuple[float | None, float | None]:
    """호가를 소진하며 목표 명목에서의 평균 체결가와 소진 깊이를 계산한다."""
    filled = cost = 0.0
    for price, qty in levels:
        price, qty = float(price), float(qty)
        take = min(qty * price, notional - cost)
        if take <= 0:
            break
        filled += take / price
        cost += take
        if cost >= notional - 1e-9:
            break
    if filled <= 0 or cost < notional * 0.999:
        return None, cost
    return cost / filled, cost


@dataclass
class Recorder:
    """공개 데이터로 전향적 관측을 축적한다."""

    cfg: RecorderConfig = field(default_factory=RecorderConfig)

    def __post_init__(self) -> None:
        """DB와 거래소 클라이언트를 준비한다."""
        Path(self.cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.cfg.db_path)
        self._con.executescript(SCHEMA)
        self._con.commit()
        self._ex = ccxt.bybit({"enableRateLimit": True})
        _retry(self._ex.load_markets)

    def _write(self, ts: str, symbol: str, kind: str, payload: dict, auth: bool = False) -> None:
        """관측 1건을 저장한다 (동일 키는 무시)."""
        self._con.execute(
            "INSERT OR IGNORE INTO observations(ts,symbol,kind,payload,authenticated) VALUES (?,?,?,?,?)",
            (ts, symbol, kind, json.dumps(payload, sort_keys=True), int(auth)),
        )

    def record_funding(self, ts: str, base: str) -> None:
        """직전 정산 펀딩과 다음 정산 예측값을 기록한다."""
        mid = f"{base}USDT"
        hist = _retry(self._ex.publicGetV5MarketFundingHistory,
                      {"category": "linear", "symbol": mid, "limit": 3})
        tick = _retry(self._ex.publicGetV5MarketTickers, {"category": "linear", "symbol": mid})
        payload: dict = {}
        if hist:
            payload["recent"] = hist.get("result", {}).get("list", [])
        if tick:
            lst = tick.get("result", {}).get("list", [])
            if lst:
                t = lst[0]
                payload["predicted"] = t.get("fundingRate")
                payload["next_funding_time"] = t.get("nextFundingTime")
                payload["mark"] = t.get("markPrice")
                payload["index"] = t.get("indexPrice")
                payload["open_interest"] = t.get("openInterest")
        if payload:
            self._write(ts, base, "funding", payload)

    def record_execution(self, ts: str, base: str) -> None:
        """현물·perp 호가와 의도 규모에서의 체결 가능 가격을 기록한다."""
        n = self.cfg.probe_notional_usd
        for cat, mid, kind in (("linear", f"{base}USDT", "perp_book"),
                               ("spot", f"{base}USDT", "spot_book")):
            r = _retry(self._ex.publicGetV5MarketOrderbook,
                       {"category": cat, "symbol": mid, "limit": self.cfg.depth_limit})
            if not r:
                continue
            res = r.get("result", {})
            bids, asks = res.get("b", []), res.get("a", [])
            if not bids or not asks:
                continue
            buy_px, buy_fill = _walk_book(asks, n)
            sell_px, sell_fill = _walk_book(bids, n)
            best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
            mid_px = (best_bid + best_ask) / 2
            self._write(ts, base, kind, {
                "best_bid": best_bid, "best_ask": best_ask,
                "spread_bp": (best_ask - best_bid) / mid_px * 1e4,
                "buy_vwap": buy_px, "sell_vwap": sell_px,
                "buy_slip_bp": (buy_px / mid_px - 1) * 1e4 if buy_px else None,
                "sell_slip_bp": (1 - sell_px / mid_px) * 1e4 if sell_px else None,
                "probe_notional": n, "depth_ok": buy_px is not None and sell_px is not None,
            })

    def record_borrow(self, ts: str) -> None:
        """차입금리·한도·담보인정비율을 기록한다 (미인증 시 No VIP 공개값)."""
        for cur in set(self.cfg.bases) | {"USDT"}:
            r = _retry(self._ex.publicGetV5SpotMarginTradeData,
                       {"vipLevel": "No VIP", "currency": cur})
            if not r:
                continue
            try:
                item = r["result"]["vipCoinList"][0]["list"][0]
            except (KeyError, IndexError, TypeError):
                continue
            hourly = float(item.get("hourlyBorrowRate", 0.0))
            self._write(ts, cur, "borrow", {
                "hourly_rate": hourly, "annual_rate": hourly * 24 * 365,
                "max_borrow": item.get("maxBorrowingAmount"),
                "collateral_ratio": item.get("collateralRatio"),
                "borrowable": item.get("borrowable"),
                "vip_level": "No VIP",
            })

    def record_risk(self, ts: str, base: str) -> None:
        """리스크티어 유지증거금률을 기록한다."""
        r = _retry(self._ex.publicGetV5MarketRiskLimit,
                   {"category": "linear", "symbol": f"{base}USDT"})
        if not r:
            return
        lst = r.get("result", {}).get("list", [])
        if lst:
            self._write(ts, base, "risk_tier", {"tiers": lst[:4]})

    def snapshot(self) -> str:
        """전 항목을 1회 기록하고 타임스탬프를 반환한다."""
        ts = _utcnow()
        self.record_borrow(ts)
        for b in self.cfg.bases:
            self.record_funding(ts, b)
            self.record_execution(ts, b)
            self.record_risk(ts, b)
        self._con.commit()
        logger.info("스냅샷 기록 완료 ts=%s", ts)
        return ts

    def stats(self) -> dict:
        """축적된 관측 수와 기간을 요약한다."""
        cur = self._con.execute(
            "SELECT kind, COUNT(*), MIN(ts), MAX(ts) FROM observations GROUP BY kind")
        return {k: dict(n=n, first=lo, last=hi) for k, n, lo, hi in cur.fetchall()}

    def close(self) -> None:
        """DB 연결을 닫는다."""
        self._con.close()


def main() -> None:
    """1회 스냅샷을 기록하고 요약을 출력한다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rec = Recorder()
    rec.snapshot()
    for kind, s in rec.stats().items():
        logger.info("%-12s n=%-6d %s ~ %s", kind, s["n"], s["first"], s["last"])
    rec.close()


if __name__ == "__main__":
    main()
