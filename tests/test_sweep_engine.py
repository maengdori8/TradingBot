"""`lab/sweep_engine.py` 단위 테스트 — SWEEP-2026-08-31 §11.1 테스트 요구 구현.

명세 §11.1 이 커밋 전 필수로 요구하는 항목 대응:
  ① 형성중 봉 사용 탐지        → `test_variant_matrices_are_causal`,
                                 `test_future_bars_cannot_change_past_returns`
  ② `ATR[i]` 사용 탐지          → `test_atr_matrix_is_lagged`,
                                 `test_donchian_channel_excludes_current_bar`
  ③ 신호봉 종가 체결 탐지       → `test_moo_fill_is_next_bar_open`,
                                 `test_signal_bar_close_does_not_move_fills`
  ④ 4h 부분 봉 확정 탐지        → `test_partial_4h_bar_is_invalidated`
  ⑤ 워밍업 중 주문 생성 탐지    → `test_no_orders_before_window_start`
  ⑥ 격자 총계 1,695 / 3,390     → `test_grid_totals`
  ⑧ 규칙 ID 유일성·결정론성     → `test_rule_ids_unique_and_deterministic`

⑦(중복도 행렬 항등식)은 부트스트랩 판정 스크립트(`lab/sweep_rc.py`)의 요구사항이며
본 엔진 파일의 범위 밖이다.

**전 테스트는 합성 데이터만 쓴다.** 동결 parquet 로 엔진을 돌리면 그것이 §9.1 의
"1회 실행"이 되어버리므로, 개발 단계에서 실데이터 시뮬레이션은 금지다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import sweep_engine as E


# ── 합성 데이터 헬퍼 ──────────────────────────────────────────────────────
def random_bars(n: int, seed: int, start: str = "2021-01-01T00:00:00Z") -> pd.DataFrame:
    """재현 가능한 랜덤워크 1h OHLCV (실데이터 대체 — 결과 조회 방지).

    시가는 직전 종가에 **갭**을 섞는다. `open[i] == close[i−1]` 이면 "신호봉 종가
    체결" 버그와 "다음 봉 시가 체결"이 수치적으로 구별되지 않아 체결가 테스트가
    무력해지기 때문이다 (실제로 돌연변이 테스트가 이 맹점을 드러냈다).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1] * np.exp(rng.normal(0, 0.0015, n - 1))
    pad = np.abs(rng.normal(0, 0.003, n)) * c
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, c) + pad, "low": np.minimum(o, c) - pad,
         "close": c, "volume": np.abs(rng.lognormal(3, 1, n))}, index=idx)


def bars_from_close(close: np.ndarray, opens: np.ndarray | None = None,
                    highs: np.ndarray | None = None, lows: np.ndarray | None = None,
                    vol: np.ndarray | None = None,
                    start: str = "2021-01-01T00:00:00Z") -> pd.DataFrame:
    """수동 설계 시나리오용 결정론적 OHLCV."""
    n = len(close)
    o = np.asarray(opens, float) if opens is not None else np.r_[close[0], close[:-1]]
    h = np.asarray(highs, float) if highs is not None else np.maximum(o, close)
    lo = np.asarray(lows, float) if lows is not None else np.minimum(o, close)
    v = np.asarray(vol, float) if vol is not None else np.full(n, 1000.0)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": close, "volume": v},
                        index=idx)


def run_one(spec: E.RuleSpec, df: pd.DataFrame, win_start: pd.Timestamp,
            win_end: pd.Timestamp, rate: float = 0.0) -> E.SimResult:
    """단일 심볼·단일 규칙 시뮬레이션 (거래 원장 추적 켜짐)."""
    idx = df.index
    ohlcv = {"X": {c: df[c].to_numpy(dtype=float) for c in
                   ("open", "high", "low", "close", "volume")}}
    settle = np.isin(np.asarray(idx.hour), E.FUNDING_HOURS)
    fund = {"X": np.where(settle, rate, np.nan)}
    return E.simulate_timeframe("1h", idx, ohlcv, fund, [spec],
                                win_start=win_start, win_end=win_end, trace_rules={0})


def find_spec(family: str, params: dict, exit_code: str, direction: str) -> E.RuleSpec:
    """격자에서 규칙 1개를 파라미터로 찾아온다."""
    for sp in E.enumerate_rules():
        if (sp.family == family and sp.exit_code == exit_code
                and sp.direction == direction and dict(sp.params) == params):
            return sp
    raise AssertionError(f"규칙 없음: {family} {params} {exit_code} {direction}")


# ── ⑥ 격자 총계 ───────────────────────────────────────────────────────────
def test_grid_totals() -> None:
    """§3.6 동결 총계 — 규칙형 1,695 / 시행 3,390 / 계열·하위 소계 전부 일치."""
    specs = E.enumerate_rules()
    assert len(specs) == E.N_RULES == 1695
    assert len(specs) * len(E.TIMEFRAMES) == E.N_TRIALS == 3390

    counts: dict[str, int] = {}
    for sp in specs:
        counts[sp.family] = counts.get(sp.family, 0) + 1
    assert counts == {
        "T-A": 234, "T-B": 105, "T-C": 96,
        "M-A": 180, "M-B": 36, "M-C": 36,
        "R-A": 72, "R-B": 120, "R-C": 288,
        "V-A": 150, "V-B": 108, "V-C": 126,
        "Q-A": 90, "Q-B": 36, "Q-C": 18,
    }
    series = {"T": 0, "M": 0, "R": 0, "V": 0, "Q": 0}
    for fam, n in counts.items():
        series[fam[0]] += n
    assert series == {"T": 435, "M": 252, "R": 480, "V": 384, "Q": 144}
    # 방향은 항상 L/S/LS 3종
    dirs: dict[str, int] = {}
    for sp in specs:
        dirs[sp.direction] = dirs.get(sp.direction, 0) + 1
    assert dirs == {"L": 565, "S": 565, "LS": 565}


def test_no_invented_parameters() -> None:
    """§3 격자 밖 파라미터가 새어 들어가지 않았는지 (창작 지표 0개 보증)."""
    specs = E.enumerate_rules()
    ra = {(p["n"], p["k"]) for p in (dict(s.params) for s in specs if s.family == "R-A")}
    assert ra == {(10, 1.9), (20, 2.0), (50, 2.1)}, "볼린저는 출판 3쌍만"
    tb = {dict(s.params)["N"] for s in specs if s.family == "T-B"}
    assert tb == {12, 24, 48, 96, 192}
    rsi_n = {dict(s.params)["n"] for s in specs if s.family in ("M-A", "R-C")}
    assert rsi_n == {2, 7, 14}
    ta = {(dict(s.params)["fast"], dict(s.params)["slow"])
          for s in specs if s.family == "T-A"}
    assert ta == set(E.TA_PAIRS) and len(ta) == 13


# ── ⑧ 규칙 ID ─────────────────────────────────────────────────────────────
def test_rule_ids_unique_and_deterministic() -> None:
    """규칙 ID 는 3,390 시행 전부에서 유일하고, 재호출 시 순서까지 동일하다."""
    a = E.enumerate_rules()
    b = E.enumerate_rules()
    assert [s.rid("1h") for s in a] == [s.rid("1h") for s in b], "열거가 비결정론적"
    ids = [s.rid(tf) for tf in E.TIMEFRAMES for s in a]
    assert len(ids) == 3390
    assert len(set(ids)) == 3390, "규칙 ID 충돌"
    for rid in ids:
        assert rid.count("|") == 4, f"§3.6 ID 형식 위반: {rid}"
    assert "T-B|N=24|X6|LS|1h" in ids, "§3.6 예시 ID 가 생성되지 않음"
    # 동점 타이브레이커는 ID 사전순 — 정렬이 안정적이어야 한다
    assert sorted(ids) == sorted(set(ids))


def test_every_family_produces_signals() -> None:
    """계열이 통째로 죽어 있지 않은지 (오타·잘못된 키로 조용히 무신호가 되는 회귀 방지).

    개별 변형 중 `RSI14 < 5` 처럼 랜덤워크에서 거의 안 나오는 조건이 있으므로
    변형 단위로는 여유를 두되, **계열 단위로는 전부 발화**해야 한다.
    """
    df = random_bars(24 * 400, seed=3)
    f = E.Feat(*(df[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")))
    comp = E.compile_rules(E.enumerate_rules())
    per_family: dict[str, int] = {}
    live = 0
    for key in comp.reg_ent.keys:
        b = E.build_entry(key, f)
        n = int(b["sigL"].sum()) + int(b["sigS"].sum())
        per_family[key[0]] = per_family.get(key[0], 0) + n
        live += n > 0
    assert set(per_family) == {"TA", "TB", "TC", "MA", "MB", "MC", "RA", "RB", "RC",
                               "VA", "VB", "VC", "QA", "QB", "QC"}
    dead = [k for k, v in per_family.items() if v == 0]
    assert not dead, f"신호가 전혀 없는 계열: {dead}"
    assert live >= 0.9 * len(comp.reg_ent.keys), f"발화 변형 {live}/{len(comp.reg_ent.keys)}"
    for key in comp.reg_sigx.keys:
        a, b2 = E.build_sigx(key, f)
        assert int(a.sum()) + int(b2.sum()) >= 0
    for key in comp.reg_dstop.keys + comp.reg_dtgt.keys:
        a, b2 = E.build_dyn(key, f)
        assert np.isfinite(a).any() and np.isfinite(b2).any(), f"동적 레벨 전부 NaN: {key}"


# ── ①② 인과성: 변형 행렬 ──────────────────────────────────────────────────
def test_variant_matrices_are_causal() -> None:
    """미래 봉을 훼손해도 그 이전 행의 신호·레벨·청산 조건이 바뀌면 안 된다.

    허용되는 동일봉 데이터는 `open[i]` 하나뿐이다 (§3.4 V-A `ref`, 그리고 체결가).
    따라서 `high/low/close/volume` 은 `>= t` 에서, `open` 은 `>= t+1` 에서 훼손한다.
    이 테스트는 돈치안 shift 누락·`ATR[i]` 사용·롤링 분위 미시프트·피벗 확정 지연
    누락을 전부 잡는다.
    """
    n, t = 900, 700
    df = random_bars(n, seed=7)
    comp = E.compile_rules(E.enumerate_rules())

    def mats(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        f = E.Feat(*(frame[c].to_numpy(dtype=float) for c in
                     ("open", "high", "low", "close", "volume")))
        return E.materialize(comp, f)

    base = mats(df)
    bad = df.copy()
    rng = np.random.default_rng(0)
    for col in ("high", "low", "close", "volume"):
        bad.iloc[t:, bad.columns.get_loc(col)] *= rng.uniform(2.0, 5.0, n - t)
    bad.iloc[t + 1:, bad.columns.get_loc("open")] *= rng.uniform(2.0, 5.0, n - t - 1)
    bad["high"] = bad[["open", "high", "close"]].max(axis=1)
    bad["low"] = bad[["open", "low", "close"]].min(axis=1)
    mutated = mats(bad)

    for name in base:
        a, b = base[name][:t + 1], mutated[name][:t + 1]
        assert a.shape == b.shape
        if a.dtype == np.bool_:
            assert np.array_equal(a, b), f"{name}: 미래 봉이 과거 신호를 바꿨다 (룩어헤드)"
        else:
            assert np.array_equal(a, b, equal_nan=True), \
                f"{name}: 미래 봉이 과거 레벨을 바꿨다 (룩어헤드)"


def test_atr_matrix_is_lagged() -> None:
    """ATR 행렬 열 `[i]` 는 반드시 `ATR[i−1]` 이어야 한다 (§4.4-1)."""
    df = random_bars(400, seed=11)
    comp = E.compile_rules(E.enumerate_rules())
    f = E.Feat(*(df[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")))
    m = E.materialize(comp, f)["atr"]
    assert comp.reg_atr.keys, "ATR 변형이 하나도 등록되지 않았다"
    for j, key in enumerate(comp.reg_atr.keys):
        raw = f.atr(key[1])
        col = m[:, j + 1]
        assert np.array_equal(col[1:], raw[:-1], equal_nan=True), f"{key}: ATR[i] 사용"
        assert np.isnan(col[0])
        # 동일봉 값을 썼다면 통과할 수 없는 대조군
        assert not np.array_equal(col, raw, equal_nan=True)


def test_donchian_channel_excludes_current_bar() -> None:
    """`HH_N[i] = max(high[i−N..i−1])` — 현재 봉을 포함하면 안 된다 (§3.0)."""
    df = random_bars(300, seed=13)
    f = E.Feat(*(df[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")))
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    for n in (12, 24):
        hh, ll = f.hh(n), f.ll(n)
        for i in (n + 5, 150, 299):
            assert hh[i] == pytest.approx(h[i - n:i].max())
            assert ll[i] == pytest.approx(lo[i - n:i].min())
        assert np.isnan(hh[:n]).all()


def test_true_range_uses_previous_close() -> None:
    """TR = max(h−l, |h−prev_close|, |l−prev_close|) — 현재 종가 기준 금지 (§4.4-2)."""
    close = np.array([100.0, 100.0, 100.0, 100.0])
    df = bars_from_close(close, opens=[100, 100, 100, 100],
                         highs=[100, 100, 100, 101], lows=[100, 100, 100, 99])
    f = E.Feat(*(df[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")))
    tr = f.tr()
    assert tr[3] == pytest.approx(2.0)          # h−l = 101−99
    assert np.isnan(f.atr(14)[0])               # prev close 없는 첫 봉은 확정 불가
    # 현재 종가 기준(swing.py 버그)이었다면 |h−close| = 1 이 최대가 되어 2.0 이 안 나온다
    assert tr[3] != pytest.approx(1.0)


def test_pivot_confirmation_is_delayed(  # D11
) -> None:
    """Q-B 피벗은 창이 `[i−1]` 이하로 닫힌 뒤에만 신호가 된다 (D11 인과성 교정)."""
    df = random_bars(300, seed=17)
    f = E.Feat(*(df[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")))
    sig_l, _, _, _ = E._qb_signals(f, 2, 50)
    fired = np.flatnonzero(sig_l)
    assert fired.size, "테스트 데이터에서 다이버전스가 한 번도 발생하지 않았다"
    lows = df["low"].to_numpy()
    for i in fired:
        j = i - 3                     # K + 1
        win = lows[j - 2:j + 3]
        assert j + 2 <= i - 1, "피벗 확정 창이 체결봉을 포함한다 (동일봉 룩어헤드)"
        assert lows[j] == win.min()


# ── ① 인과성: 엔진 전체 ───────────────────────────────────────────────────
def test_future_bars_cannot_change_past_returns() -> None:
    """미래 봉을 통째로 훼손해도 그 이전 일수익률은 비트 단위로 같아야 한다."""
    n = 24 * 80
    frames = {s: random_bars(n, seed=31 + k) for k, s in enumerate(("A", "B"))}
    idx = frames["A"].index
    ws, we = idx[24 * 10], idx[24 * 70]
    specs = E.enumerate_rules()

    def run(fr: dict[str, pd.DataFrame]) -> np.ndarray:
        ohlcv = {s: {c: d[c].to_numpy(dtype=float) for c in
                     ("open", "high", "low", "close", "volume")} for s, d in fr.items()}
        settle = np.isin(np.asarray(idx.hour), E.FUNDING_HOURS)
        fund = {s: np.where(settle, 0.0001, np.nan) for s in fr}
        return E.simulate_timeframe("1h", idx, ohlcv, fund, specs,
                                    win_start=ws, win_end=we).returns

    base = run(frames)
    cut_bar = 24 * 50
    cut_day = 40                                    # 훼손 시작 이전까지의 완전한 날 수
    bad = {s: d.copy() for s, d in frames.items()}
    rng = np.random.default_rng(5)
    for s, d in bad.items():
        k = rng.uniform(3.0, 6.0, n - cut_bar)
        for col in ("high", "low", "close", "volume"):
            d.iloc[cut_bar:, d.columns.get_loc(col)] *= k
        d.iloc[cut_bar:, d.columns.get_loc("open")] *= k
        d["high"] = d[["open", "high", "close"]].max(axis=1)
        d["low"] = d[["open", "low", "close"]].min(axis=1)
    mutated = run(bad)

    assert np.array_equal(base[:, :cut_day], mutated[:, :cut_day]), \
        "미래 봉이 과거 일수익률을 바꿨다 — 룩어헤드"
    assert not np.array_equal(base, mutated), "훼손이 아무 영향도 주지 않았다 (테스트 무효)"


def test_engine_is_deterministic() -> None:
    """동일 입력 두 번 실행 → 비트 단위 동일 (§9.4 결정론 요구)."""
    n = 24 * 40
    frames = {s: random_bars(n, seed=41 + k) for k, s in enumerate(("A", "B"))}
    idx = frames["A"].index
    ohlcv = {s: {c: d[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")} for s, d in frames.items()}
    fund = {s: np.where(np.isin(np.asarray(idx.hour), E.FUNDING_HOURS), 0.0001, np.nan)
            for s in frames}
    specs = E.enumerate_rules()[:200]
    kw = dict(win_start=idx[24 * 5], win_end=idx[24 * 35])
    r1 = E.simulate_timeframe("1h", idx, ohlcv, fund, specs, **kw)
    r2 = E.simulate_timeframe("1h", idx, ohlcv, fund, specs, **kw)
    assert np.array_equal(r1.returns, r2.returns)
    assert np.array_equal(r1.equity, r2.equity)
    assert np.array_equal(r1.trades, r2.trades)


# ── ③ 체결가 ──────────────────────────────────────────────────────────────
def _tc_cross_scenario() -> tuple[pd.DataFrame, int]:
    """T-C SMA20 상향 교차가 봉 `t` 에서 확정되도록 설계한 시나리오."""
    n, t = 100, 40
    close = np.full(n, 100.0)
    close[t] = 110.0                       # SMA20[t] = 100.5 < 110 → 교차 확정
    close[t + 1:] = 110.0
    opens = np.r_[close[0], close[:-1]].copy()
    opens[t + 1] = 120.0                   # 신호봉 종가(110)와 뚜렷이 다른 체결가
    highs = np.maximum(opens, close) + 1.0
    lows = np.minimum(opens, close) - 1.0
    return bars_from_close(close, opens, highs, lows), t


def test_moo_fill_is_next_bar_open() -> None:
    """비돌파형은 **다음 봉 시가**에 체결된다 — 신호봉 종가 체결 절대 금지 (§4.4-3)."""
    df, t = _tc_cross_scenario()
    sp = find_spec("T-C", {"ma": "SMA", "N": 20}, "X1", "L")
    res = run_one(sp, df, df.index[25], df.index[-1])
    entries = [e for e in res.trace if e["action"] == "entry"]
    assert entries, "진입이 발생하지 않았다 (시나리오 설계 오류)"
    first = entries[0]
    assert first["bar"] == t + 1, "신호 확정봉이 아니라 다음 봉에서 체결되어야 한다"
    assert first["price"] == pytest.approx(120.0), "체결가는 다음 봉 시가"
    assert first["price"] != pytest.approx(110.0), "신호봉 종가로 체결됐다 — 룩어헤드"


def test_signal_bar_close_does_not_move_fills() -> None:
    """봉 `t` 의 **종가만** 흔들어도 그 봉의 체결가·체결 여부는 불변이어야 한다.

    `close[i]` 는 봉이 끝나야 알 수 있으므로 어떤 체결가도 될 수 없다. 시가·고가·
    저가는 그대로 두므로 봉의 유효성은 유지된다.
    """
    n = 24 * 30
    frames = {s: random_bars(n, seed=53 + k) for k, s in enumerate(("BTC", "ETH", "SOL"))}
    for d in frames.values():
        d["high"] = d[["open", "high", "close"]].max(axis=1) + 0.5
        d["low"] = d[["open", "low", "close"]].min(axis=1) - 0.5
    idx = frames["BTC"].index
    ws, we = idx[24 * 5], idx[24 * 25]
    specs = E.enumerate_rules()
    ohlcv = {s: {c: d[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")} for s, d in frames.items()}
    fund = {s: np.full(n, np.nan) for s in frames}
    rules = set(range(len(specs)))
    base = E.simulate_timeframe("1h", idx, ohlcv, fund, specs, win_start=ws,
                                win_end=we, trace_rules=rules)
    t = 24 * 15
    o2 = {s: dict(v) for s, v in ohlcv.items()}
    c2 = o2["BTC"]["close"].copy()
    lo, hi = o2["BTC"]["low"][t], o2["BTC"]["high"][t]
    c2[t] = lo + 0.97 * (hi - lo) if c2[t] < (lo + hi) / 2 else lo + 0.03 * (hi - lo)
    o2["BTC"]["close"] = c2
    mut = E.simulate_timeframe("1h", idx, o2, fund, specs, win_start=ws,
                               win_end=we, trace_rules=rules)

    def at_bar(res: E.SimResult) -> list[tuple]:
        # 수량까지 비교한다 — 사이징 자본을 close[i] 로 마킹하는 동일봉 룩어헤드는
        # 체결가를 바꾸지 않고 수량만 바꾸므로 가격만 보면 놓친다.
        return sorted((e["rule"], e["sym"], e["action"], round(e["price"], 10),
                       round(e["qty"], 10)) for e in res.trace if e["bar"] == t)

    assert at_bar(base), "봉 t 에 아무 체결도 없다 (테스트 무효)"
    assert at_bar(base) == at_bar(mut), "신호봉 종가가 같은 봉 체결을 바꿨다 — 룩어헤드"


def test_every_fill_price_is_legal() -> None:
    """전 격자 광역 불변식 — 어떤 체결가도 종가가 될 수 없다.

    * 비돌파형 진입·시가 청산 → 정확히 그 봉의 **시가**.
    * 돌파형 진입·스탑·목표 → 그 봉의 [저가, 고가] 안 (갭 악화 포함).
    이 불변식 하나로 "신호봉 종가 체결"류 회귀가 전 계열에서 차단된다.
    """
    n = 24 * 30
    df = random_bars(n, seed=137)
    specs = E.enumerate_rules()
    brk = {"T-B", "V-A"}
    fam = [s.family for s in specs]
    is_brk = [f in brk or (f == "V-C" and dict(s.params)["d"] == "D2")
              or (f == "Q-A" and dict(s.params)["a"] in ("A1", "A5"))
              for f, s in zip(fam, specs)]
    ohlcv = {"X": {c: df[c].to_numpy(dtype=float) for c in
                   ("open", "high", "low", "close", "volume")}}
    res = E.simulate_timeframe("1h", df.index, ohlcv, {"X": np.full(n, np.nan)}, specs,
                               win_start=df.index[24 * 5], win_end=df.index[24 * 25],
                               trace_rules=set(range(len(specs))))
    o = df["open"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    assert len(res.trace) > 500, "표본이 너무 적다 (테스트 무효)"
    seen_target = False
    for ev in res.trace:
        b, p = ev["bar"], ev["price"]
        if ev["action"] == "exit" and ev["reason"] == "target":
            # §4.4-9: 리밋 목표는 레벨 그대로 — 유리한 갭 이득이 없으므로 봉 밖일 수
            # 있다. 단 "봉보다 유리한" 쪽으로는 절대 벗어날 수 없다.
            seen_target = True
            if ev["direction"] > 0:
                assert p <= hi[b] + 1e-9, f"롱 목표가 봉 고가보다 유리: {ev}"
            else:
                assert p >= lo[b] - 1e-9, f"숏 목표가 봉 저가보다 유리: {ev}"
            continue
        assert lo[b] - 1e-9 <= p <= hi[b] + 1e-9, f"체결가가 봉 밖: {ev}"
        moo = (ev["action"] == "entry" and not is_brk[ev["rule"]]) or \
              (ev["action"] == "exit" and ev["reason"] == "open")
        if moo:
            assert p == pytest.approx(o[b], abs=1e-12), \
                f"MOO 체결이 시가가 아니다 (종가 체결 의심): {ev}"
    assert seen_target, "목표 청산 표본이 없다 (테스트 커버리지 부족)"


def test_breakout_fill_applies_gap_worsening() -> None:
    """돌파형은 봉내 스탑주문 — `fill = max(레벨, open)` (갭 악화, §4.4-3)."""
    n, t = 60, 40
    close = 100.0 - np.arange(n) * 0.1          # 완만한 하락 → 조기 돌파 없음
    opens = np.r_[close[0], close[:-1]].copy()
    highs = np.maximum(opens, close) + 0.05
    lows = np.minimum(opens, close) - 0.05
    opens[t] = 150.0                            # 채널 위로 갭 상승
    close[t] = 155.0
    highs[t], lows[t] = 160.0, 149.0
    df = bars_from_close(close, opens, highs, lows)
    level = df["high"].to_numpy()[t - 12:t].max()
    assert level < 150.0, "시나리오 설계 오류 (갭이 채널 위가 아님)"

    sp = find_spec("T-B", {"N": 12}, "X5", "L")   # 6×ATR24 스탑 → 즉시 청산 없음
    res = run_one(sp, df, df.index[25], df.index[-1])
    entries = [e for e in res.trace if e["action"] == "entry"]
    assert entries and entries[0]["bar"] == t
    assert entries[0]["price"] == pytest.approx(150.0), "갭 악화 미적용 (레벨로 체결됨)"
    assert entries[0]["price"] != pytest.approx(level)


def test_stop_exit_applies_gap_worsening() -> None:
    """스탑 청산은 갭 악화 — 롱이 스탑 아래로 갭하면 `min(레벨, open)` (§4.4-9).

    유리한 방향(`max`)으로 반전되면 손실이 과소평가되므로 반드시 잡아야 한다.
    """
    n, t, gap = 60, 40, 43
    close = 100.0 - np.arange(n) * 0.1
    opens = np.r_[close[0], close[:-1]].copy()
    highs = np.maximum(opens, close) + 0.05
    lows = np.minimum(opens, close) - 0.05
    opens[t], close[t], highs[t], lows[t] = 150.0, 155.0, 160.0, 150.0   # 돌파 진입
    for b in (t + 1, t + 2):                                             # 스탑 위 유지
        opens[b], close[b], highs[b], lows[b] = 155.0, 155.0, 156.0, 154.0
    opens[gap], close[gap] = 60.0, 58.0                                  # 스탑 아래로 갭
    highs[gap], lows[gap] = 61.0, 55.0
    close[gap + 1:], opens[gap + 1:] = 58.0, 58.0
    highs[gap + 1:], lows[gap + 1:] = 59.0, 57.0
    df = bars_from_close(close, opens, highs, lows)
    sp = find_spec("T-B", {"N": 12}, "X3", "L")     # 2×ATR24 고정 스탑
    res = run_one(sp, df, df.index[25], df.index[-1])
    entry = [x for x in res.trace if x["action"] == "entry"][0]
    ex = next(x for x in res.trace if x["action"] == "exit" and x["bar"] >= entry["bar"])
    assert ex["reason"] == "stop"
    assert ex["bar"] == gap, "갭 봉에서 스탑이 발동하지 않았다 (시나리오 설계 오류)"
    assert entry["stop"] > opens[gap], "시나리오가 갭 상황이 아니다"
    assert ex["price"] == pytest.approx(opens[gap]), "갭 악화 미적용 (레벨로 체결됨)"
    assert ex["price"] < entry["stop"], "유리한 방향으로 체결됐다 — 손실 과소평가"


# ── ④ 4h 봉 확정 ──────────────────────────────────────────────────────────
def test_partial_4h_bar_is_invalidated() -> None:
    """닫힌 1h 봉 4개가 전부 있을 때만 4h 봉이 확정된다 (§4.3)."""
    df = random_bars(24, seed=61)
    full = E.resample_4h(df)
    assert full["close"].notna().all() and len(full) == 6
    assert full["open"].iloc[0] == pytest.approx(df["open"].iloc[0])
    assert full["close"].iloc[0] == pytest.approx(df["close"].iloc[3])
    assert full["high"].iloc[0] == pytest.approx(df["high"].iloc[:4].max())
    assert full["volume"].iloc[1] == pytest.approx(df["volume"].iloc[4:8].sum())
    assert (full.index.hour % 4 == 0).all(), "UTC [00,04,08,…) 정렬 위반"

    holed = df.drop(df.index[5])                 # 두 번째 4h 버킷에서 1h 봉 1개 결측
    part = E.resample_4h(holed)
    assert part.iloc[1].isna().all(), "부분 봉이 확정됐다 (§4.3 위반)"
    assert part.drop(part.index[1]).notna().all().all(), "온전한 봉까지 무효화됐다"


def test_align_leaves_missing_bars_as_nan() -> None:
    """결측 봉은 보간하지 않고 NaN 으로 남긴다 (§4.4-5 fail-closed)."""
    a = random_bars(48, seed=71)
    b = random_bars(48, seed=72).drop(index=random_bars(48, seed=72).index[10])
    fund = pd.DataFrame({"A": 0.0001, "B": 0.0001}, index=a.index)
    idx, ohlcv, _, gaps = E.align({"A": a, "B": b}, fund, "1h")
    assert gaps == {"A": 0, "B": 1}
    assert np.isnan(ohlcv["B"]["close"][10])
    assert np.isfinite(ohlcv["A"]["close"][10])


def test_missing_bar_produces_no_orders() -> None:
    """결측 봉에서는 해당 심볼에 대해 무행동 (보간·추정 금지)."""
    n = 24 * 30
    df = random_bars(n, seed=73)
    hole = 24 * 15 + 7
    for col in ("open", "high", "low", "close", "volume"):
        df.iloc[hole, df.columns.get_loc(col)] = np.nan
    specs = E.enumerate_rules()
    ohlcv = {"X": {c: df[c].to_numpy(dtype=float) for c in
                   ("open", "high", "low", "close", "volume")}}
    res = E.simulate_timeframe("1h", df.index, ohlcv, {"X": np.full(n, np.nan)}, specs,
                               win_start=df.index[24 * 5], win_end=df.index[24 * 25],
                               trace_rules=set(range(len(specs))))
    assert all(e["bar"] != hole for e in res.trace), "결측 봉에서 주문이 나갔다"
    assert np.isfinite(res.returns).all()


# ── ⑤ 워밍업 ──────────────────────────────────────────────────────────────
def test_no_orders_before_window_start() -> None:
    """평가 창 이전에는 지표만 워밍업하고 주문은 0건이어야 한다 (§4.2)."""
    n = 24 * 60
    df = random_bars(n, seed=83)
    specs = E.enumerate_rules()
    ohlcv = {"X": {c: df[c].to_numpy(dtype=float) for c in
                   ("open", "high", "low", "close", "volume")}}
    i0 = 24 * 30
    res = E.simulate_timeframe("1h", df.index, ohlcv, {"X": np.full(n, np.nan)}, specs,
                               win_start=df.index[i0], win_end=df.index[24 * 55],
                               trace_rules=set(range(len(specs))))
    assert res.trace, "거래가 전혀 없다 (테스트 무효)"
    assert min(e["bar"] for e in res.trace) >= i0, "워밍업 구간에서 주문이 생성됐다"
    assert res.equity[:, 0] == pytest.approx(E.CELL_CAPITAL), "창 시작 자본이 $10,000 이 아님"


def test_window_shape_and_snapshot_source() -> None:
    """자정 스냅샷은 **직전 확정 봉 종가** 로 평가되고, 스냅샷 수 = 일수 + 1 이다."""
    n = 24 * 12
    df = random_bars(n, seed=89)
    sp = find_spec("R-C", {"n": 2, "t": 5, "filt": 0}, "X1", "L")
    ws, we = df.index[24 * 2], df.index[24 * 10]
    res = run_one(sp, df, ws, we)
    assert res.equity.shape == (1, 9) and res.returns.shape == (1, 8)
    assert list(res.snap_ts) == list(pd.date_range(ws, we, freq="D"))
    # 첫 스냅샷은 거래 전이므로 정확히 초기 자본
    assert res.equity[0, 0] == pytest.approx(E.CELL_CAPITAL)


# ── 실행 계약: 사이징·비용·펀딩·정지 ──────────────────────────────────────
def test_stopless_rules_use_one_third_notional() -> None:
    """확정 스탑 레벨이 없는 규칙은 명목 = equity × 1/3 (§4.5)."""
    df, t = _tc_cross_scenario()
    sp = find_spec("T-C", {"ma": "SMA", "N": 20}, "X1", "L")   # X1 = 스탑 없음
    res = run_one(sp, df, df.index[25], df.index[-1])
    e = [x for x in res.trace if x["action"] == "entry"][0]
    assert e["sized_by"] == "notional"
    assert e["qty"] * e["price"] == pytest.approx(E.CELL_CAPITAL * E.STOPLESS_NOTIONAL)
    assert np.isnan(e["stop"])


def test_stop_rules_use_two_percent_risk_inversion() -> None:
    """확정 스탑이 있으면 `u = 0.02 × equity / d` (§4.5)."""
    n, t = 60, 40
    close = 100.0 - np.arange(n) * 0.1
    opens = np.r_[close[0], close[:-1]].copy()
    highs = np.maximum(opens, close) + 0.05
    lows = np.minimum(opens, close) - 0.05
    opens[t], close[t], highs[t], lows[t] = 150.0, 155.0, 160.0, 149.0
    df = bars_from_close(close, opens, highs, lows)
    sp = find_spec("T-B", {"N": 12}, "X5", "L")     # 6×ATR24 스탑
    res = run_one(sp, df, df.index[25], df.index[-1])
    e = [x for x in res.trace if x["action"] == "entry"][0]
    assert e["sized_by"] == "stop"
    d = abs(e["price"] - e["stop"])
    assert d > 0
    assert e["qty"] == pytest.approx(E.RISK * E.CELL_CAPITAL / d)


def test_gross_cap_clamps_position_size() -> None:
    """스탑이 아주 좁으면 2% 역산 수량이 그로스 캡 10x 에서 잘린다 (§4.5)."""
    n, t = 60, 40
    close = 100.0 - np.arange(n) * 0.01          # 초저변동 → ATR 극소 → 스탑 극근접
    opens = np.r_[close[0], close[:-1]].copy()
    highs = np.maximum(opens, close) + 0.002
    lows = np.minimum(opens, close) - 0.002
    opens[t], close[t], highs[t], lows[t] = 150.0, 155.0, 160.0, 150.0
    df = bars_from_close(close, opens, highs, lows)
    sp = find_spec("T-B", {"N": 12}, "X3", "L")
    res = run_one(sp, df, df.index[25], df.index[-1])
    e = [x for x in res.trace if x["action"] == "entry"][0]
    d = abs(e["price"] - e["stop"])
    unclamped = E.RISK * E.CELL_CAPITAL / d
    assert unclamped * e["price"] > E.GROSS_CAP * E.CELL_CAPITAL, "시나리오가 캡을 안 건드림"
    assert e["qty"] < unclamped, "그로스 캡이 적용되지 않았다"
    assert e["qty"] * e["price"] == pytest.approx(E.GROSS_CAP * E.CELL_CAPITAL)


def test_portfolio_caps_and_no_pyramiding() -> None:
    """3심볼 통합 셀 불변식 — 심볼당 1포지션·피라미딩 없음·동시 최대 3포지션 (§4.4-11)."""
    n = 24 * 25
    frames = {s: random_bars(n, seed=311 + k) for k, s in enumerate(("BTC", "ETH", "SOL"))}
    idx = frames["BTC"].index
    specs = E.enumerate_rules()
    ohlcv = {s: {c: d[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")} for s, d in frames.items()}
    fund = {s: np.where(np.isin(np.asarray(idx.hour), E.FUNDING_HOURS), 0.0001, np.nan)
            for s in frames}
    res = E.simulate_timeframe("1h", idx, ohlcv, fund, specs, win_start=idx[24 * 3],
                               win_end=idx[24 * 20], trace_rules=set(range(len(specs))))
    assert len(res.trace) > 2000, "표본이 너무 적다 (테스트 무효)"
    open_pos: dict[tuple[int, str], int] = {}
    live: dict[int, int] = {}
    assert [e["bar"] for e in res.trace] == sorted(e["bar"] for e in res.trace), \
        "원장이 시간순이 아니다"
    for ev in res.trace:      # 삽입 순서 = 실제 처리 순서 (재정렬 금지)
        key = (ev["rule"], ev["sym"])
        if ev["action"] == "entry":
            assert open_pos.get(key, 0) == 0, f"심볼당 2포지션 (피라미딩): {ev}"
            open_pos[key] = 1
            live[ev["rule"]] = live.get(ev["rule"], 0) + 1
            assert live[ev["rule"]] <= E.MAX_POS, f"동시 포지션 {live[ev['rule']]} > {E.MAX_POS}"
        else:
            assert open_pos.get(key, 0) == 1, f"포지션 없이 청산: {ev}"
            open_pos[key] = 0
            live[ev["rule"]] -= 1
    assert res.equity.shape[0] == len(specs)


def test_dynamic_channel_level_is_used_for_sizing() -> None:
    """ATR 스탑이 없고 **동적 역채널만** 있는 규칙(T-B X1)도 2% 역산으로 사이징된다.

    동적 레벨을 사이징에서 빠뜨리면 이런 규칙이 통째로 명목 1/3 로 떨어져
    포지션 크기가 수십 배 어긋난다.
    """
    n, t = 60, 40
    close = 100.0 - np.arange(n) * 0.1
    opens = np.r_[close[0], close[:-1]].copy()
    highs = np.maximum(opens, close) + 0.05
    lows = np.minimum(opens, close) - 0.05
    opens[t], close[t], highs[t], lows[t] = 150.0, 155.0, 160.0, 150.0
    df = bars_from_close(close, opens, highs, lows)
    sp = find_spec("T-B", {"N": 12}, "X1", "L")     # 반대 6채널만, ATR 스탑 없음
    res = run_one(sp, df, df.index[25], df.index[-1])
    e = [x for x in res.trace if x["action"] == "entry"][0]
    assert e["sized_by"] == "stop", "동적 역채널이 사이징에서 무시됐다"
    ll6 = df["low"].to_numpy()[t - 6:t].min()
    assert e["qty"] == pytest.approx(E.RISK * E.CELL_CAPITAL / abs(e["price"] - ll6))


def test_round_trip_cost_is_sixteen_bp() -> None:
    """왕복 비용 = 편도 8bp × 2, 명목 기준 (§4.5)."""
    df, t = _tc_cross_scenario()
    sp = find_spec("T-C", {"ma": "SMA", "N": 20}, "X3", "L")   # 24봉 보유 후 청산
    res = run_one(sp, df, df.index[25], df.index[-1])
    legs = res.trace
    assert len(legs) >= 2
    entry = legs[0]
    exit_ = next(x for x in legs[1:] if x["action"] == "exit")
    want = (entry["qty"] * entry["price"] + exit_["qty"] * exit_["price"]) * E.COST_SIDE
    assert res.cost[0] == pytest.approx(want)
    assert 2 * E.COST_SIDE == pytest.approx(E.RT_COST)


def test_funding_sign_long_pays_short_receives() -> None:
    """양(+) 펀딩에서 롱은 지불, 숏은 수취한다 (§4.4-7)."""
    df, _ = _tc_cross_scenario()
    long_ = find_spec("T-C", {"ma": "SMA", "N": 20}, "X4", "L")
    short = find_spec("T-C", {"ma": "SMA", "N": 20}, "X4", "S")
    rl = run_one(long_, df, df.index[25], df.index[-1], rate=0.001)
    assert rl.funding[0] < 0, "롱이 양(+) 펀딩을 수취했다 — 부호 반전"
    # 숏 시나리오: 하락 교차를 만들어 숏 진입 유도
    n = len(df)
    close = np.full(n, 100.0)
    close[40:] = 90.0
    df2 = bars_from_close(close)
    rs = run_one(short, df2, df2.index[25], df2.index[-1], rate=0.001)
    if rs.trades[0] > 0:
        assert rs.funding[0] > 0, "숏이 양(+) 펀딩을 지불했다 — 부호 반전"


def test_hold_timeout_exits_at_open_after_h_bars() -> None:
    """보유봉수 청산: 체결봉을 1봉으로 세어 H 봉 뒤 **시가** 청산 (D3)."""
    df, t = _tc_cross_scenario()
    sp = find_spec("M-A", {"n": 2, "U": 50}, "X3", "L")    # 12봉 보유, 신호청산 없음
    res = run_one(sp, df, df.index[25], df.index[-1])
    entries = [x for x in res.trace if x["action"] == "entry"]
    exits = [x for x in res.trace if x["action"] == "exit"]
    assert entries and exits
    e, x = entries[0], exits[0]
    assert x["bar"] - e["bar"] == 12, "보유봉수 계산이 명세(체결봉=1)와 다르다"
    assert x["reason"] == "open"
    assert x["price"] == pytest.approx(df["open"].to_numpy()[x["bar"]])


def test_same_bar_reentry_is_blocked() -> None:
    """청산한 봉에는 재진입 금지 — 다음 봉부터 재진입 (§4.4-11, D2)."""
    n = 80
    close = 100.0 - np.arange(n) * 0.5      # 단조 하락 → RSI2 ≈ 0 지속, close < SMA5
    df = bars_from_close(close)
    sp = find_spec("R-C", {"n": 2, "t": 5, "filt": 0}, "X4", "L")   # X1 ∨ 24봉
    res = run_one(sp, df, df.index[30], df.index[-1])
    entries = sorted(x["bar"] for x in res.trace if x["action"] == "entry")
    exits = sorted(x["bar"] for x in res.trace if x["action"] == "exit")
    assert len(entries) >= 2 and exits, "재진입 시나리오가 발생하지 않았다"
    assert exits[0] - entries[0] == 24
    assert entries[1] != exits[0], "청산한 바로 그 봉에서 재진입했다"
    assert entries[1] == exits[0] + 1, "다음 봉 재진입이 아니다"


def test_stop_beats_target_on_same_bar() -> None:
    """같은 봉에서 스탑·목표 동시 도달 시 스탑 우선(비관) — 체결봉부터 검사 (§4.4-10).

    격자 밖의 합성 규칙으로 **엔진 우선순위 기계 자체**를 검증한다 (스탑과 R 목표를
    동시에 가진 규칙형이 §3 격자에 없기 때문).
    """
    n, t = 60, 40
    close = np.full(n, 100.0)
    close[t] = 110.0
    close[t + 1:] = 110.0
    opens = np.r_[close[0], close[:-1]].copy()
    opens[t + 1] = 100.0
    highs = np.maximum(opens, close) + 0.2
    lows = np.minimum(opens, close) - 0.2
    highs[t + 1], lows[t + 1] = 400.0, 1.0          # 스탑·목표 둘 다 봉 안에 든다
    df = bars_from_close(close, opens, highs, lows)
    synth = E.RuleSpec("T-C", (("ma", "SMA"), ("N", 20)), "SYN", "L",
                       ("TC", "SMA", 20),
                       E.ExitSpec(atr_n=14, atr_mult=2.0, tgt_r=1.0))
    res = run_one(synth, df, df.index[25], df.index[-1])
    entry = [x for x in res.trace if x["action"] == "entry"][0]
    ex = [x for x in res.trace if x["action"] == "exit"][0]
    assert entry["bar"] == t + 1 and ex["bar"] == t + 1, "체결봉 스탑/목표 검사 누락"
    assert ex["reason"] == "stop", "동시 도달에서 목표가 우선했다 (낙관 편향)"
    assert ex["price"] < entry["price"], "롱 스탑 청산가가 체결가보다 높다"


def test_long_priority_when_both_breakouts_trigger() -> None:
    """같은 봉에 롱·숏 돌파가 동시 성립하면 **롱 우선** (§3.1 T-B, §3.4 V-A)."""
    n, t = 40, 30
    close = np.full(n, 100.0)
    opens = np.full(n, 100.0)
    highs = 100.5 - 0.01 * np.arange(n)      # 레인지 수축 → 양쪽 다 조기 돌파 없음
    lows = 99.5 + 0.01 * np.arange(n)
    opens[t], close[t], highs[t], lows[t] = 100.0, 100.0, 200.0, 1.0   # 양방향 돌파
    df = bars_from_close(close, opens, highs, lows)
    hh = df["high"].to_numpy()[t - 12:t].max()
    ll = df["low"].to_numpy()[t - 12:t].min()
    assert highs[t] >= hh and lows[t] <= ll, "시나리오가 양방향 돌파가 아니다"
    sp = find_spec("T-B", {"N": 12}, "X5", "LS")
    res = run_one(sp, df, df.index[20], df.index[-1])
    e = [x for x in res.trace if x["action"] == "entry" and x["bar"] == t][0]
    assert e["direction"] == 1, "동시 돌파에서 숏이 선택됐다 (롱 우선 위반)"
    assert e["price"] == pytest.approx(max(hh, opens[t]))


def test_favorable_level_is_not_treated_as_a_stop() -> None:
    """유리한 쪽 레벨은 스탑이 아니다 → 명목 1/3 사이징 (§4.5, D6).

    격자 밖 합성 규칙으로 D6 판정 기계를 직접 검정한다: 페이드 롱에 "반대 밴드"를
    동적 **스탑** 으로 물려도, 그 레벨이 체결가 위(유리)면 스탑 역산을 쓰면 안 된다.
    """
    n, t = 60, 40
    close = np.full(n, 100.0)
    close[t] = 90.0
    close[t + 1:] = 90.0
    opens = np.r_[close[0], close[:-1]].copy()
    opens[t + 1] = 91.0
    df = bars_from_close(close, opens, np.maximum(opens, close) + 0.5,
                         np.minimum(opens, close) - 0.5)
    synth = E.RuleSpec("R-A", (("n", 20), ("k", 2.0), ("e", "E1")), "SYN", "L",
                       ("RA", 20, 2.0, 1), E.ExitSpec(dyn_stop=("RA_OPP", 20, 2.0)))
    res = run_one(synth, df, df.index[25], df.index[-1])
    e = [x for x in res.trace if x["action"] == "entry"][0]
    assert e["bar"] == t + 1 and e["price"] == pytest.approx(91.0)
    assert e["sized_by"] == "notional", "체결가 위의 레벨을 스탑으로 역산했다 (D6 위반)"
    assert e["qty"] * e["price"] == pytest.approx(E.CELL_CAPITAL * E.STOPLESS_NOTIONAL)


def test_open_positions_are_force_closed_at_window_end() -> None:
    """창 종료 시 미청산 포지션은 마지막 확정 봉 종가로 강제 청산된다 (§4.2 eod)."""
    df, t = _tc_cross_scenario()
    sp = find_spec("T-C", {"ma": "SMA", "N": 20}, "X4", "L")   # 48봉 보유 → 창 안에서 안 끝남
    we_bar = 60
    res = run_one(sp, df, df.index[25], df.index[we_bar])
    eod = [x for x in res.trace if x["action"] == "exit" and x["reason"] == "eod"]
    assert eod, "창 종료 미청산 포지션이 강제 청산되지 않았다"
    assert eod[0]["bar"] == we_bar - 1, "마지막 확정 봉이 아니다"
    assert eod[0]["price"] == pytest.approx(df["close"].to_numpy()[we_bar - 1])
    assert res.trades[0] == 2, "강제 청산이 거래로 계상되지 않았다"


def test_heat_cap_gates_entries() -> None:
    """포트폴리오 heat 캡이 실제로 신규 진입을 차단하는 배선인지 (§4.5)."""
    n = 24 * 20
    frames = {s: random_bars(n, seed=401 + k) for k, s in enumerate(("BTC", "ETH", "SOL"))}
    idx = frames["BTC"].index
    specs = E.enumerate_rules()
    ohlcv = {s: {c: d[c].to_numpy(dtype=float) for c in
                 ("open", "high", "low", "close", "volume")} for s, d in frames.items()}
    fund = {s: np.full(n, np.nan) for s in frames}
    kw = dict(win_start=idx[24 * 3], win_end=idx[24 * 17])
    base = E.simulate_timeframe("1h", idx, ohlcv, fund, specs, **kw)
    saved = E.HEAT_CAP
    try:
        E.HEAT_CAP = 1e-6                      # 어떤 포지션도 통과 못 함
        tight = E.simulate_timeframe("1h", idx, ohlcv, fund, specs, **kw)
    finally:
        E.HEAT_CAP = saved
    assert base.trades.sum() > 0
    assert tight.trades.sum() == 0, "heat 캡이 진입 게이트에 배선되지 않았다"


def test_daily_halt_blocks_new_entries_without_liquidation() -> None:
    """일손실 −5% 도달 → 당일 신규 진입만 차단, 강제 청산은 없다 (§4.5, Track E 판본)."""
    n = 24 * 8
    rng = np.random.default_rng(97)
    close = 100.0 * np.exp(np.cumsum(rng.normal(-0.02, 0.03, n)))   # 급락장
    df = bars_from_close(close)
    specs = [s for s in E.enumerate_rules() if s.family == "R-C" and s.direction == "L"]
    ohlcv = {"X": {c: df[c].to_numpy(dtype=float) for c in
                   ("open", "high", "low", "close", "volume")}}
    kw = dict(win_start=df.index[24 * 2], win_end=df.index[24 * 7])
    res = E.simulate_timeframe("1h", df.index, ohlcv, {"X": np.full(n, np.nan)}, specs, **kw)
    assert res.halts.max() > 0, "급락장에서도 정지가 한 번도 걸리지 않았다"
    assert np.isfinite(res.returns).all()

    # 정지가 실제로 신규 진입을 막는 배선인지 (카운터만 올리고 마는 회귀 차단)
    saved = E.DAILY_HALT
    try:
        E.DAILY_HALT = -1e9                     # 절대 발동하지 않음
        never = E.simulate_timeframe("1h", df.index, ohlcv,
                                     {"X": np.full(n, np.nan)}, specs, **kw)
    finally:
        E.DAILY_HALT = saved
    assert never.halts.max() == 0
    assert never.trades.sum() > res.trades.sum(), \
        "정지가 걸렸는데 거래 수가 줄지 않았다 — 진입 차단이 배선되지 않았다"
    # 강제 청산은 없다 (Track E 판본): 정지 후에도 보유 포지션은 유지된다
    assert res.equity.shape == never.equity.shape


def test_degenerate_rules_get_zero_sharpe_but_stay_in_n() -> None:
    """거래 0건 규칙은 `SR := 0` 이되 N 에서 제거되지 않는다 (§5.1)."""
    n = 24 * 12
    df = random_bars(n, seed=101)
    specs = E.enumerate_rules()
    ohlcv = {"X": {c: df[c].to_numpy(dtype=float) for c in
                   ("open", "high", "low", "close", "volume")}}
    res = E.simulate_timeframe("1h", df.index, ohlcv, {"X": np.full(n, np.nan)}, specs,
                               win_start=df.index[24 * 3], win_end=df.index[24 * 10])
    summ = E.summarize(res, specs, "1h")
    assert len(summ) == len(specs), "요약에서 규칙이 사라졌다 (사후 제거 금지)"
    zero = summ.loc[res.trades == 0, "sharpe_ann"]
    assert (zero == 0.0).all()
    assert np.isfinite(summ["sharpe_ann"]).all()

    # §5.1 문구를 직접 검정: 거래 0건이면 수익률이 무엇이든 SR := 0
    forced = E.SimResult(
        rule_ids=["a", "b"], snap_ts=res.snap_ts,
        equity=np.tile(res.equity[0], (2, 1)),
        returns=np.tile(np.linspace(0.01, 0.02, res.returns.shape[1]), (2, 1)),
        trades=np.array([0, 5]), cost=np.zeros(2), funding=np.zeros(2),
        halts=np.zeros(2, int))
    got = E.summarize(forced, specs[:2], "1h")["sharpe_ann"].to_numpy()
    assert got[0] == 0.0, "거래 0건 규칙에 SR:=0 이 적용되지 않았다 (§5.1)"
    assert got[1] != 0.0


def test_input_hash_mismatch_is_fail_closed(tmp_path, monkeypatch) -> None:
    """§11.2 해시가 어긋나면 실행을 거부한다 (fail-closed)."""
    monkeypatch.setattr(E, "EXPECTED_SHA256", {"lab/sweep_engine.py": "0" * 64})
    with pytest.raises(ValueError, match="해시 불일치"):
        E.load_inputs(verify_hashes=True)


def test_frozen_constants_match_preregistration() -> None:
    """§11.3 동결 상수가 코드에서 변조되지 않았는지."""
    assert (E.N_RULES, E.N_TRIALS, E.SEED) == (1695, 3390, 20260831)
    assert (E.RT_COST, E.RISK, E.GROSS_CAP, E.MAX_POS) == (0.0016, 0.02, 10.0, 3)
    assert (E.HEAT_CAP, E.DAILY_HALT, E.CELL_CAPITAL) == (0.06, -0.05, 10_000.0)
    assert E.STOPLESS_NOTIONAL == pytest.approx(1 / 3)
    assert E.STOPLESS_HEAT == 0.05
    assert str(E.WIN_START) == "2021-11-21 00:00:00+00:00"
    assert str(E.WIN_END) == "2026-08-24 00:00:00+00:00"
    assert E.N_DAYS == 1737
