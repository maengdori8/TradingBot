"""전 코인 시장 스캐너 단위 테스트 — 합성 캔들, 네트워크 없음."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import carrybot.live.market_scanner as ms

H1 = ms.H1


def bars_from(closes, highs=None, vols=None, t0=0):
    """합성 확정봉 시퀀스 — (ts, open, high, low, close, vol)."""
    n = len(closes)
    highs = highs if highs is not None else [c + 1.0 for c in closes]
    vols = vols if vols is not None else [10.0] * n
    return [(t0 + i * H1, closes[i], highs[i], closes[i] - 1.0,
             closes[i], vols[i]) for i in range(n)]


class TestUniverse:
    """봇 유니버스 규칙 동형 — USDT 선형 무기한 · 거래대금 상위 · $5M 필터."""

    def test_무기한만_거래대금_내림차순으로_선별한다(self):
        tickers = {
            "BTC/USDT": {"quoteVolume": 9e9},              # 현물 — 제외
            "BTC/USDT:USDT": {"quoteVolume": 5e9},
            "ETH/USDT:USDT": {"quoteVolume": 8e9},
            "XRP/USDT:USDT": {"quoteVolume": 1e6},         # $5M 미만 — 제외
            "SOL/USDT:USDT": {"quoteVolume": None},        # 결측 — 제외
        }
        top = ms.top_usdt_perp(tickers)
        assert [s for s, _ in top] == ["ETH/USDT:USDT", "BTC/USDT:USDT"]

    def test_상한을_지킨다(self):
        tickers = {f"C{i}/USDT:USDT": {"quoteVolume": 1e7 + i}
                   for i in range(50)}
        top = ms.top_usdt_perp(tickers, limit=40)
        assert len(top) == 40
        # 내림차순 — 가장 큰 거래대금이 맨 앞
        assert top[0][0] == "C49/USDT:USDT"


class TestWilderRsi:
    def test_무변동은_50(self):
        assert ms.wilder_rsi([100.0] * 30, 14) == 50.0

    def test_상승만은_100(self):
        assert ms.wilder_rsi([100.0 + i for i in range(30)], 14) == 100.0

    def test_하락만은_0(self):
        assert ms.wilder_rsi([100.0 - i * 0.5 for i in range(30)], 14) == 0.0

    def test_표본_부족은_None(self):
        assert ms.wilder_rsi([100.0], 14) is None
        assert ms.wilder_rsi([], 14) is None

    def test_pandas_ewm과_동치(self):
        """첫 diff 시드 = ewm(alpha=1/n, adjust=False) 선행 NaN 스킵 동치."""
        rng = np.random.default_rng(7)
        closes = list(100.0 + np.cumsum(rng.normal(0, 1, 120)))
        diffs = np.diff(closes)
        up = np.maximum(diffs, 0.0)
        dn = np.maximum(-diffs, 0.0)
        alpha = 1.0 / 14
        u = up[0]
        d = dn[0]
        for i in range(1, len(diffs)):
            u = u + (up[i] - u) * alpha
            d = d + (dn[i] - d) * alpha
        expected = 100.0 - 100.0 / (1.0 + u / d)
        assert ms.wilder_rsi(closes, 14) == pytest.approx(expected)


class TestComputeMetrics:
    def test_채널_근접도_양수(self):
        """상단 아래에 있으면 거리 % 양수 (채널은 확정봉 [i] 제외 직전 N봉)."""
        m = ms.compute_metrics(bars_from([95.0] * 120, highs=[100.0] * 120))
        assert m["dist24h_pct"] == pytest.approx((100.0 / 95.0 - 1) * 100, abs=1e-3)
        assert m["dist96h_pct"] == pytest.approx((100.0 / 95.0 - 1) * 100, abs=1e-3)

    def test_이미_돌파하면_음수(self):
        closes = [95.0] * 119 + [105.0]
        highs = [100.0] * 119 + [106.0]     # [i] 고가는 채널에서 제외돼야 함
        m = ms.compute_metrics(bars_from(closes, highs=highs))
        assert m["dist24h_pct"] == pytest.approx((100.0 / 105.0 - 1) * 100, abs=1e-3)
        assert m["dist24h_pct"] < 0

    def test_봉_부족이면_해당_지표_None(self):
        m = ms.compute_metrics(bars_from([100.0] * 50))
        assert m["dist24h_pct"] is not None
        assert m["dist96h_pct"] is None     # 97봉 필요
        assert m["sma200_pct"] is None      # 200봉 필요
        assert m["gate_long"] is False      # fail-closed
        assert m["gate_short"] is False

    def test_롱_게이트_충족(self):
        """추세(>SMA200)·RSI14>50·거래량 서지 전부 참 → 롱만 켜진다."""
        closes = [100.0 + 0.1 * i for i in range(220)]
        vols = [10.0] * 219 + [50.0]
        m = ms.compute_metrics(bars_from(closes, vols=vols))
        assert m["gate_long"] is True
        assert m["gate_short"] is False
        assert m["rsi14"] == 100.0
        assert m["sma200_pct"] > 0
        assert m["vol_surge"] == pytest.approx(5.0)
        assert m["gate_parts"]["vol"] is True

    def test_숏_게이트_충족(self):
        closes = [200.0 - 0.1 * i for i in range(220)]
        vols = [10.0] * 219 + [50.0]
        m = ms.compute_metrics(bars_from(closes, vols=vols))
        assert m["gate_short"] is True
        assert m["gate_long"] is False
        assert m["rsi14"] == 0.0
        assert m["sma200_pct"] < 0

    def test_거래량_미달이면_양방향_차단(self):
        closes = [100.0 + 0.1 * i for i in range(220)]
        vols = [10.0] * 220                  # 직전 평균과 동일 — 초과 아님
        m = ms.compute_metrics(bars_from(closes, vols=vols))
        assert m["gate_long"] is False
        assert m["gate_short"] is False
        assert m["gate_parts"]["trend_long"] is True   # 다른 게이트는 참

    def test_거래량_NaN은_차단(self):
        closes = [100.0 + 0.1 * i for i in range(220)]
        vols = [10.0] * 219 + [float("nan")]
        m = ms.compute_metrics(bars_from(closes, vols=vols))
        assert m["vol_surge"] is None
        assert m["gate_long"] is False

    def test_BB_pctb는_모표준편차_기준(self):
        closes = [100.0] * 219 + [110.0]
        m = ms.compute_metrics(bars_from(closes))
        w = np.array(closes[-20:])
        sd = w.std(ddof=0)
        expected = (110.0 - (w.mean() - 2 * sd)) / (4 * sd)
        assert m["bb_pctb"] == pytest.approx(expected, abs=1e-3)

    def test_무변동_밴드는_None(self):
        m = ms.compute_metrics(bars_from([100.0] * 220))
        assert m["bb_pctb"] is None          # sd == 0 — 나눗셈 금지

    def test_빈_시퀀스와_기형_종가는_None(self):
        assert ms.compute_metrics([]) is None
        assert ms.compute_metrics(bars_from([float("nan")] * 30)) is None

    def test_채널_창_내_NaN은_거리_None(self):
        """창 중간 NaN 이 max() 위치에 따라 무시되는 함정 — 전량 유효 요구."""
        highs = [100.0] * 120
        highs[-10] = float("nan")           # 24봉 창 안
        m = ms.compute_metrics(bars_from([95.0] * 120, highs=highs))
        assert m["dist24h_pct"] is None
        assert m["dist96h_pct"] is None


class FakeEx:
    """네트워크 목 — fetch_tickers/fetch_ohlcv 합성 응답."""

    def __init__(self, tickers, candles, fail=()):
        self.tickers = tickers
        self.candles = candles
        self.fail = set(fail)

    def fetch_tickers(self):
        return self.tickers

    def fetch_ohlcv(self, sym, timeframe, limit=None):
        if sym in self.fail:
            raise RuntimeError("boom")
        return self.candles[sym]


class TestFetchClosed:
    def test_미확정_봉은_제외한다(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        now_h = 300 * H1
        rows = [[t * H1, 1, 2, 0.5, 1.5, 3.0] for t in range(295, 301)]
        ex = FakeEx({}, {"BTC/USDT:USDT": rows})
        bars = ms.fetch_closed_1h(ex, "BTC/USDT:USDT", now_h)
        assert [b[0] for b in bars] == [t * H1 for t in range(295, 300)]

    def test_수집_실패는_None(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        ex = FakeEx({}, {}, fail={"BTC/USDT:USDT"})
        assert ms.fetch_closed_1h(ex, "BTC/USDT:USDT", 300 * H1) is None

    def test_오래된_응답은_None(self, monkeypatch):
        """최신 확정봉(now−1h)이 없으면 스킵 — 낡은 캔들을 신선한 척 금지."""
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        rows = [[t * H1, 1, 2, 0.5, 1.5, 3.0] for t in range(250, 260)]
        ex = FakeEx({}, {"BTC/USDT:USDT": rows})
        assert ms.fetch_closed_1h(ex, "BTC/USDT:USDT", 300 * H1) is None

    def test_갭이_있으면_연속_꼬리만_남긴다(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        rows = ([[t * H1, 1, 2, 0.5, 1.5, 3.0] for t in range(290, 293)]
                + [[t * H1, 1, 2, 0.5, 1.5, 3.0] for t in range(295, 300)])
        ex = FakeEx({}, {"BTC/USDT:USDT": rows})
        bars = ms.fetch_closed_1h(ex, "BTC/USDT:USDT", 300 * H1)
        assert [b[0] for b in bars] == [t * H1 for t in range(295, 300)]

    def test_기형_행은_행_단위로_건너뛴다(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        rows = [[295 * H1], ["x", 1, 2],            # 기형 행 2개
                *[[t * H1, 1, 2, 0.5, 1.5, 3.0] for t in range(296, 300)]]
        ex = FakeEx({}, {"BTC/USDT:USDT": rows})
        bars = ms.fetch_closed_1h(ex, "BTC/USDT:USDT", 300 * H1)
        assert [b[0] for b in bars] == [t * H1 for t in range(296, 300)]


class TestRunScan:
    def _mk(self, fail=()):
        now_ms = 400 * H1 + 123
        closes = [100.0 + 0.1 * i for i in range(220)]
        rows = [[(400 - 220 + i) * H1, closes[i], closes[i] + 1.0,
                 closes[i] - 1.0, closes[i], 10.0] for i in range(220)]
        tickers = {
            "BTC/USDT:USDT": {"quoteVolume": 5e9, "last": 122.0,
                              "percentage": 1.5},
            "ETH/USDT:USDT": {"quoteVolume": 8e9, "last": 121.0,
                              "percentage": -2.0},
            "BTC/USDT": {"quoteVolume": 9e9},   # 현물 — 유니버스 제외
        }
        candles = {"BTC/USDT:USDT": rows, "ETH/USDT:USDT": rows}
        return FakeEx(tickers, candles, fail=fail), now_ms

    def test_거래대금_순으로_전_코인을_담는다(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        ex, now_ms = self._mk()
        p = ms.run_scan(ex, now_ms=now_ms)
        assert [c["coin"] for c in p["coins"]] == ["ETH", "BTC"]  # 거래대금 순
        assert p["skipped"] == 0
        assert "generated_at_utc" in p
        btc = p["coins"][1]
        assert btc["price"] == 122.0            # 티커 현재가
        assert btc["chg24h_pct"] == 1.5
        assert btc["turnover24h"] == 5e9
        assert btc["bar_ts"] == 399 * H1        # 최신 확정봉
        assert btc["rsi14"] == 100.0

    def test_실패_심볼은_스킵하고_카운트한다(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        ex, now_ms = self._mk(fail={"ETH/USDT:USDT"})
        p = ms.run_scan(ex, now_ms=now_ms)
        assert [c["coin"] for c in p["coins"]] == ["BTC"]
        assert p["skipped"] == 1
        assert p["skipped_symbols"] == ["ETH/USDT:USDT"]

    def test_티커_실패는_예외로_드러난다(self, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)

        class Dead:
            def fetch_tickers(self):
                raise RuntimeError("down")

        with pytest.raises(RuntimeError):
            ms.run_scan(Dead(), now_ms=400 * H1)

    def test_연속_실패는_조기_중단하고_잔여를_스킵_계상한다(self, monkeypatch):
        """거래소 광역 장애 — 심볼당 백오프 낭비 방지, 잔여도 스킵으로 투명 계상."""
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        n = ms.CONSEC_FAIL_MAX + 2
        tickers = {f"C{i}/USDT:USDT": {"quoteVolume": 1e9 - i}
                   for i in range(n)}
        ex = FakeEx(tickers, {}, fail=set(tickers))
        p = ms.run_scan(ex, now_ms=400 * H1)
        assert p["coins"] == []
        assert p["skipped"] == n            # 연속 10 + 잔여 2


class TestSaveAtomic:
    def test_원자_저장_후_임시파일이_남지_않는다(self, tmp_path):
        path = tmp_path / "market_scan.json"
        payload = {"generated_at_utc": "2026-08-28T00:00:00+00:00",
                   "coins": [{"coin": "BTC"}], "skipped": 0}
        ms.save_atomic(payload, path=path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
        assert not list(tmp_path.glob("market_scan.json.tmp*"))

    def test_재실행은_덮어쓴다_멱등(self, tmp_path):
        path = tmp_path / "market_scan.json"
        ms.save_atomic({"coins": [], "skipped": 0}, path=path)
        ms.save_atomic({"coins": [{"coin": "ETH"}], "skipped": 1}, path=path)
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["coins"] == [{"coin": "ETH"}]

    def test_NaN없는_유효_JSON을_만든다(self, tmp_path, monkeypatch):
        """run_scan 산출물이 표준 JSON 으로 강건하게 저장되는지 왕복 검증."""
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        ex, now_ms = TestRunScan()._mk()
        p = ms.run_scan(ex, now_ms=now_ms)
        path = tmp_path / "market_scan.json"
        ms.save_atomic(p, path=path)
        d = json.loads(path.read_text(encoding="utf-8"))
        assert len(d["coins"]) == 2
        assert math.isfinite(d["coins"][0]["price"])

    def test_NaN_잔존은_저장을_거부하고_기존_파일을_보존한다(self, tmp_path):
        """allow_nan=False — 비표준 NaN 직렬화로 소비측 파서를 깨지 않는다."""
        path = tmp_path / "market_scan.json"
        path.write_text('{"coins": []}', encoding="utf-8")   # 직전 정상 산출물
        with pytest.raises(ValueError):
            ms.save_atomic({"coins": [{"price": float("nan")}]}, path=path)
        assert path.read_text(encoding="utf-8") == '{"coins": []}'  # 미갱신
        assert not list(tmp_path.glob("market_scan.json.tmp*"))


class TestMain:
    """main() 종료코드 매핑 — 네트워크 전면 목 (실행 아님)."""

    def test_마켓_로드_실패는_종료코드_1_파일_미생성(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        monkeypatch.setattr(ms, "OUT", tmp_path / "market_scan.json")

        class DeadEx:
            def load_markets(self):
                raise RuntimeError("down")

        monkeypatch.setattr(ms.ccxt, "bybit", lambda cfg: DeadEx())
        assert ms.main() == 1
        assert not (tmp_path / "market_scan.json").exists()

    def test_정상_경로는_종료코드_0_파일_생성(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        monkeypatch.setattr(ms, "OUT", tmp_path / "market_scan.json")
        fake, now_ms = TestRunScan()._mk()
        fake.load_markets = lambda: {}
        monkeypatch.setattr(ms.ccxt, "bybit", lambda cfg: fake)
        monkeypatch.setattr(ms.time, "time", lambda: now_ms / 1000.0)
        assert ms.main() == 0
        d = json.loads((tmp_path / "market_scan.json").read_text("utf-8"))
        assert [c["coin"] for c in d["coins"]] == ["ETH", "BTC"]

    def test_전_심볼_실패는_종료코드_1_파일_미갱신(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms.time, "sleep", lambda *_: None)
        monkeypatch.setattr(ms, "OUT", tmp_path / "market_scan.json")
        fake, now_ms = TestRunScan()._mk(
            fail={"BTC/USDT:USDT", "ETH/USDT:USDT"})
        fake.load_markets = lambda: {}
        monkeypatch.setattr(ms.ccxt, "bybit", lambda cfg: fake)
        monkeypatch.setattr(ms.time, "time", lambda: now_ms / 1000.0)
        assert ms.main() == 1
        assert not (tmp_path / "market_scan.json").exists()
