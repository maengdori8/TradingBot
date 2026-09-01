"""손익비 1.5:1 익절 백테스트 — BRK24(교정판) vs BRK24+1.5R 익절. U1 사전 기준선 기록용.

사전 고정 (실행 전 동결 — 튜닝·사후선택 절대 금지):
- **1.5R 은 사용자가 결과를 조회하기 전에 지정한 값이다.** 따라서 1.5R 셀은 결과와
  무관하게 U1 에 편입된다. 아래 익절 배수 격자(1.0/1.5/2.0/3.0/없음)는 **공시 전용**
  이며 셀 선택 근거로 사용되지 않는다 — 독자가 "1.5 를 격자에서 사후 선택하지
  않았음"을 직접 검증하도록 전 셀 수치를 공개하는 것이 유일한 목적이다.
- 기반: lab/scalp_grid.py BRK24 (1h 채널 돌파 N=24, 6xATR(24) 스탑, N/2 역채널 추적,
  리스크 2%, 그로스 캡 10x, 일손실 -5% 정지, 왕복 16bp, 일별 펀딩).
  실행 교정 (scalp_farm 교정 1·2): ATR[i-1] 사용(같은 봉 완성 ATR 금지),
  TR 은 previous close 기준. 기준선 팔(익절 없음)은 lab/confluence_gate_test.py
  의 (a)팔과 **완전 동일 경로**이며 selftest 가 최종자본·거래수 동일성을 강제한다.
- 추가 규칙 (익절, 이것 하나뿐):
  tgt = fill + R×(fill − stop)  (롱/숏 공통식, R = |fill − stop| = 초기 스탑거리)
  · 봉내 **레벨 체결** — 갭이 목표 너머로 유리하게 벌어져도 체결가는 목표 레벨
    (carrybot/aggressive/scalp_farm.py BRK24TP·RSI-DIV #15 관례 그대로).
  · **체결봉부터 검사** — 진입한 그 봉에서도 스탑·목표를 본다 (#17 관례).
  · **같은 봉 동시 도달 시 스탑/역채널 우선 (비관)** — 봉내 도달 순서는 1h OHLC
    로 알 수 없으므로 불리한 쪽으로 확정한다. 청산 우선순위 동결:
    BRK 스탑/역채널(갭 악화) → 익절 목표(레벨).
  · **최대 보유 제한 없음** (scalp_farm BRK24TP 의 12봉 타임아웃은 미적용 —
    본 실험의 사전 지정 규칙에 없다).
- 진입·초기 스탑·사이징(리스크 2%)·기존 청산(반대채널·스탑, 갭 악화)은 BRK24 와
  완전 동일하다. 익절 외 어떤 파라미터도 건드리지 않는다.
- 데이터: lab/frozen/perp_1h.parquet BTC/ETH + lab/data/sol_1h.parquet SOL (전 기간).
  펀딩: lab/frozen/funding.parquet 일합. 비용 왕복 16bp.
- 판정 없음: 이 스크립트는 기준선 기록 전용이다.

계승 한계 (scalp_grid 원본 충실 재현 — 사전등록 범위 밖이라 미수정, 전 팔 동일):
- 펀딩: 당일 '일합'을 일 시작에 차감 (당일 08/16시 정산분 선반영 — 전 팔 동일).
- 그로스 캡 마크: 기존 포지션을 close[i]로 평가 (봉내 체결 시점 미지값 —
  confluence_gate_test Codex 진단: close[i-1] 대체 시 본 데이터셋 결과 불변).
- 이중 돌파봉(상하 동시): 롱 우선 해석.
- 일손실 -5% 도달 시 전 포지션 청산 (scalp_grid 원전 — 라이브 팜의 '진입만 정지'
  와 다르나 전 팔 동일 적용이라 비교는 공정).
측정 교정 (전략 불변, 전 팔 동일 — 기준선 왜곡 방지):
- 심볼 데이터 종료 시 보유 포지션은 마지막 유효 종가로 강제청산(비용 차감).
- 누적수익·CAGR·MDD는 초기자본 1.0 앵커 기준.
- 승률·평균수익/거래는 가격손익+수수료 기준 (펀딩은 자본 차감, 거래 미배분).
- 보유시간 = 청산봉 ts − 진입봉 ts (같은 봉 청산 = 0h).
"""
from __future__ import annotations
import hashlib
import numpy as np, pandas as pd

np.random.seed(0)                       # 무작위성 없음 — 형식상 고정
COST_IN, COST_OUT = 0.0008, 0.0008      # 편도 8bp(taker+슬립) = 왕복 16bp
N, RISK, GROSS_CAP, DAILY_HALT = 24, 0.02, 10.0, -0.05
TP_GRID = (None, 1.0, 1.5, 2.0, 3.0)    # 공시용 격자 — 셀 선택에 사용되지 않음
TP_PRESPEC = 1.5                        # 사용자가 결과 조회 전 지정한 배수 (U1 편입)
PATHS = ('lab/frozen/perp_1h.parquet', 'lab/data/sol_1h.parquet',
         'lab/frozen/funding.parquet')
# 데이터 동결 검증 — frozen 2종은 lab/frozen/MANIFEST.json 과 일치해야 한다.
SHA_EXPECT = {
    'lab/frozen/perp_1h.parquet':
        'c06a3301457dfec8f68b184e1f8ac8797acc49874fa96c409a6a57e6743ebac0',
    'lab/data/sol_1h.parquet':
        '80d7f7574d680505eb280d05bfecc4ce3bc38ff461e651b91eb122e666f04785',
    'lab/frozen/funding.parquet':
        '534642ee677424abb949492a7b8f21e43ea635bd0eb9f980e12f829f89a0a128',
}


def sha256(path: str) -> str:
    """파일 SHA256 (데이터 동결 확인용)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load():
    """BTC/ETH(frozen)+SOL(lab/data) 1h OHLCV, 일별 펀딩 — scalp_grid.load 동형."""
    p = pd.read_parquet(PATHS[0])
    cols = ['open', 'high', 'low', 'close', 'volume']
    d = {s: p.xs(s, level='sym')[cols] for s in ('BTC', 'ETH')}
    d['SOL'] = pd.read_parquet(PATHS[1])[cols]
    fh = pd.read_parquet(PATHS[2])[['BTC', 'ETH', 'SOL']]
    return d, fh.resample('D').sum(min_count=1), fh


def atr(df: pd.DataFrame, n: int = 24) -> pd.Series:
    """ATR — TR은 previous close 기준 (scalp_grid 원형)."""
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def run(data: dict, fund: pd.DataFrame, tp_r: float | None, shift_atr: bool = True,
        fundh: pd.DataFrame | None = None):
    """BRK24 교정판 실행 (+선택적 R배수 익절).

    Args:
        data: sym -> OHLCV DataFrame (UTC DatetimeIndex).
        fund: 일별 펀딩률 합 (열 = 심볼) — 일 시작 선차감(scalp_grid 원전).
        tp_r: 익절 배수 R. None = 익절 없음(기준선, confluence_gate_test (a)팔 동일).
        shift_atr: False 면 ATR[i] (같은 봉 완성값) 사용 — **인과성 위반 대조군
            전용**. 본 실행은 항상 True.
        fundh: 주면 **정확 정산시각 모드** (민감도 전용, scalp_farm 교정 7) —
            일합 선차감 대신 실제 정산 타임스탬프(=봉 종가 시각)에 그 시점
            보유분에만 부과한다. 봉내 청산된 포지션은 부과 대상이 아니다.
            사전등록 본 수치는 fundh=None (원전 계승) 쪽이다.

    Returns:
        (일별 곡선용 DataFrame, 트레이드 로그 DataFrame, 동시도달 봉 수).
    """
    syms = list(data)
    idx = None
    for s in syms:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()
    D = {s: data[s].reindex(idx) for s in syms}
    A = {s: (atr(D[s]).shift(1) if shift_atr else atr(D[s])) for s in syms}  # 교정 1
    HI = {s: D[s].high.rolling(N).max().shift(1) for s in syms}
    LO = {s: D[s].low.rolling(N).min().shift(1) for s in syms}
    XH = {s: D[s].high.rolling(N // 2).max().shift(1) for s in syms}
    XL = {s: D[s].low.rolling(N // 2).min().shift(1) for s in syms}

    eq, pos, rows, tlog = 1.0, {}, [], []
    simul = 0                       # 스탑·목표 같은 봉 동시 도달 (스탑 우선 처리) 수
    day, day_eq, halted = None, 1.0, False

    def log(t_out, s, p, x, reason):
        """청산 1건 기록 — pnl 은 가격손익 + 양방향 수수료."""
        tlog.append(dict(ts=t_out, ts_in=p['t0'], sym=s, d=p['d'], e=p['e'], x=x,
                         stop=p['stop'], u=p['u'], eq0=p['eq0'], reason=reason,
                         hold_h=(t_out - p['t0']).total_seconds() / 3600.0,
                         rden=p['u'] * abs(p['e'] - p['stop']),   # 1R (USD)
                         fee=p['u'] * p['e'] * COST_IN + p['u'] * x * COST_OUT,
                         pnl=p['u'] * (x - p['e']) * p['d']
                         - p['u'] * p['e'] * COST_IN - p['u'] * x * COST_OUT))

    for i, t in enumerate(idx):
        if i < 100:
            continue
        if t.date() != day:
            day, halted = t.date(), False
            day_eq = eq + sum(p['u'] * (D[s].close.iloc[i - 1] - p['e']) * p['d']
                              for s, p in pos.items()
                              if not np.isnan(D[s].close.iloc[i - 1]))
            if fundh is None:                          # 펀딩 (일 1회, 롱 지불)
                for s, p in pos.items():
                    f = fund[s].get(pd.Timestamp(day, tz='utc'), np.nan)
                    px = D[s].close.iloc[i - 1]
                    if not np.isnan(f) and not np.isnan(px):
                        eq -= p['d'] * f * p['u'] * px
        for s in syms:
            o, h, l, c = (D[s].open.iloc[i], D[s].high.iloc[i],
                          D[s].low.iloc[i], D[s].close.iloc[i])
            a = A[s].iloc[i]
            if np.isnan(c):
                p = pos.get(s)
                if p is not None:      # 데이터 종료(내부 갭 없음 확인) — 강제청산
                    px = D[s].close.iloc[i - 1]
                    if not np.isnan(px):
                        eq += p['u'] * (px - p['e']) * p['d'] - p['u'] * px * COST_OUT
                        log(idx[i - 1], s, p, px, 'eod')
                        pos.pop(s)
                continue
            if np.isnan(a) or a <= 0:
                continue
            p = pos.get(s)
            if p:
                # 청산 우선순위 (동결): BRK 스탑/역채널(갭 악화) → 익절 목표(레벨).
                # 같은 봉 동시 도달은 이 순서가 곧 '스탑 우선(비관)'을 강제한다.
                exit_px, reason = None, ''
                if p['d'] > 0:
                    lvl = max(p['stop'], XL[s].iloc[i])
                    if l <= lvl:
                        exit_px, reason = min(lvl, o), 'exit'
                else:
                    lvl = min(p['stop'], XH[s].iloc[i])
                    if h >= lvl:
                        exit_px, reason = max(lvl, o), 'exit'
                tp_hit = tp_r is not None and ((p['d'] > 0 and h >= p['tgt'])
                                               or (p['d'] < 0 and l <= p['tgt']))
                if exit_px is None and tp_hit:
                    exit_px, reason = p['tgt'], 'target'     # 갭 유리해도 레벨 체결
                if exit_px is not None:
                    if reason == 'exit' and tp_hit:
                        simul += 1
                    eq += p['u'] * (exit_px - p['e']) * p['d'] - p['u'] * exit_px * COST_OUT
                    log(t, s, p, exit_px, reason)
                    pos.pop(s)
                    continue
            if halted or s in pos:
                continue
            gross = sum(pp['u'] * D[ss].close.iloc[i] for ss, pp in pos.items()
                        if not np.isnan(D[ss].close.iloc[i]))
            if gross >= GROSS_CAP * eq:
                continue
            d_ = 0
            if h > HI[s].iloc[i]:
                d_, fill = 1, max(o, HI[s].iloc[i])     # 스탑주문 모델: 불리한 쪽
            elif l < LO[s].iloc[i]:
                d_, fill = -1, min(o, LO[s].iloc[i])
            if not d_:
                continue
            stop = fill - d_ * 6 * a                    # a = ATR[i-1] (교정 1)
            u = min(RISK * eq / abs(fill - stop),
                    max(0.0, (GROSS_CAP * eq - gross)) / fill)
            if u <= 0:
                continue
            eq0 = eq
            eq -= u * fill * COST_IN
            tgt = fill + tp_r * (fill - stop) if tp_r is not None else float('nan')
            p = dict(d=d_, u=u, e=fill, stop=stop, tgt=tgt, eq0=eq0, t0=t)
            # 체결봉부터 검사 (#17) — 스탑 우선, 그다음 목표 (비관)
            if (d_ > 0 and l <= stop) or (d_ < 0 and h >= stop):
                if tp_r is not None and ((d_ > 0 and h >= tgt) or (d_ < 0 and l <= tgt)):
                    simul += 1
                eq += u * (stop - fill) * d_ - u * stop * COST_OUT
                log(t, s, p, stop, 'same_bar_stop')
                continue
            if tp_r is not None and ((d_ > 0 and h >= tgt) or (d_ < 0 and l <= tgt)):
                eq += u * (tgt - fill) * d_ - u * tgt * COST_OUT
                log(t, s, p, tgt, 'same_bar_target')
                continue
            pos[s] = p
        if fundh is not None:      # 민감도: 정확 정산시각(=봉 종가) 보유분만 부과
            ts_close = t + pd.Timedelta(hours=1)
            for s, p in pos.items():
                f = fundh[s].get(ts_close, np.nan)
                px = D[s].close.iloc[i]
                if not np.isnan(f) and not np.isnan(px):
                    eq -= p['d'] * f * p['u'] * px
        mtm = eq + sum(p['u'] * (D[s].close.iloc[i] - p['e']) * p['d']
                       for s, p in pos.items() if not np.isnan(D[s].close.iloc[i]))
        if not halted and day_eq > 0 and mtm / day_eq - 1 < DAILY_HALT and pos:
            for s, p in list(pos.items()):
                px = D[s].close.iloc[i]
                if np.isnan(px):
                    continue
                eq += p['u'] * (px - p['e']) * p['d'] - p['u'] * px * COST_OUT
                log(t, s, p, px, 'halt')
                pos.pop(s)
            halted, mtm = True, eq
        rows.append((t, mtm))
    for s, p in list(pos.items()):     # 기간 말 잔여 포지션 — 마지막 유효 종가 청산
        px = D[s].close.dropna().iloc[-1]
        eq += p['u'] * (px - p['e']) * p['d'] - p['u'] * px * COST_OUT
        log(idx[-1], s, p, px, 'eod')
        pos.pop(s)
    if rows:
        rows[-1] = (rows[-1][0], eq)   # 최종점 = 청산 완료 자본
    dfr = pd.DataFrame(rows, columns=['ts', 'equity']).set_index('ts')
    return dfr, pd.DataFrame(tlog), simul


def stats(dfr: pd.DataFrame, tl: pd.DataFrame) -> dict:
    """한 팔의 요약 지표 — 초기자본 1.0 앵커."""
    anchor = pd.Series([1.0], index=[dfr.index[0] - pd.Timedelta(hours=1)])
    d = pd.concat([anchor, dfr.equity.resample('D').last().dropna()])
    eh = pd.concat([anchor, dfr.equity])
    yrs = (dfr.index[-1] - anchor.index[0]).total_seconds() / (365.25 * 86400)
    n = len(tl)
    tp = int(tl.reason.isin(('target', 'same_bar_target')).sum()) if n else 0
    # R배수 = 순손익 / 1R(진입시 스탑거리 명목) — 팔 간 거래품질 비교용 정규화
    # (평균 pnl/eq0 는 자본경로·거래수가 팔마다 달라 비교가능성이 약하다 — Codex #3)
    r = tl.pnl / tl.rden if n else pd.Series(dtype=float)
    w, ls = r[r > 0], r[r <= 0]
    return dict(
        n=n, win=(tl.pnl > 0).mean() if n else float('nan'),
        avg=(tl.pnl / tl.eq0).mean() if n else float('nan'),
        net=d.iloc[-1] - 1.0, cagr=d.iloc[-1] ** (1 / yrs) - 1,
        mdd_d=(1 - d / d.cummax()).max(), mdd_h=(1 - eh / eh.cummax()).max(),
        tp=tp, tp_rate=tp / n if n else float('nan'),
        hold=tl.hold_h.mean() if n else float('nan'),
        hold_med=tl.hold_h.median() if n else float('nan'),
        r=r.mean() if n else float('nan'),
        r_se=r.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan'),
        r_win=w.mean() if len(w) else float('nan'),
        r_loss=ls.mean() if len(ls) else float('nan'),
        r_max=r.max() if n else float('nan'),
        r_p95=r.quantile(0.95) if n else float('nan'),
        fee_r=(tl.fee / tl.rden).mean() if n else float('nan'),
        daily=d)


def report(name: str, dfr: pd.DataFrame, tl: pd.DataFrame, simul: int) -> dict:
    """한 팔(arm) 결과표 출력."""
    st = stats(dfr, tl)
    d, tl_ = st['daily'], tl
    print(f"\n=== {name} ===")
    print(f"거래수 {st['n']}  승률 {st['win']*100:.1f}%  "
          f"평균수익/거래 {st['avg']*100:+.3f}%"
          f"(진입시 자본대비, 가격손익+수수료 — 펀딩 제외)")
    print(f"누적 순수익 {st['net']*100:+.1f}%  CAGR {st['cagr']*100:+.1f}%  "
          f"MDD(일봉) {st['mdd_d']*100:.1f}%  MDD(1h) {st['mdd_h']*100:.1f}%")
    print(f"익절 도달 {st['tp']}건 ({st['tp_rate']*100:.1f}%)  "
          f"평균 보유 {st['hold']:.1f}h (중앙값 {st['hold_med']:.1f}h)  "
          f"스탑·목표 동시도달봉 {simul}건 (전부 스탑/역채널 우선)")
    print(f"R배수: 평균 {st['r']:+.4f}R (SE {st['r_se']:.4f})  "
          f"승 평균 {st['r_win']:+.3f}R  패 평균 {st['r_loss']:+.3f}R  "
          f"손익비 {abs(st['r_win']/st['r_loss']):.2f}  "
          f"최대 {st['r_max']:+.2f}R  p95 {st['r_p95']:+.2f}R  "
          f"수수료 {st['fee_r']*100:.2f}%R")
    rc = tl_.reason.value_counts() if st['n'] else {}
    print("청산사유: " + "  ".join(f"{k} {int(v)}" for k, v in rc.items()))
    yr = (1 + d.pct_change()).groupby(d.index.year).prod() - 1
    ty = tl_.groupby(tl_.ts.dt.year).size() if st['n'] else pd.Series(dtype=int)
    wy = (tl_.groupby(tl_.ts.dt.year).pnl.apply(lambda x: (x > 0).mean())
          if st['n'] else {})
    print(f"{'연도':>6s} {'수익률':>9s} {'거래':>6s} {'승률':>7s}")
    for y in yr.index:
        print(f"{y:>6d} {yr[y]*100:+8.1f}% {ty.get(y, 0):6d} "
              f"{wy.get(y, float('nan'))*100:6.1f}%")
    return st


def selftest(data: dict, fund: pd.DataFrame, base, tp15) -> None:
    """인과성·기준선 동일성 자가검증 — 위반 시 AssertionError 로 실패한다."""
    print("\n--- selftest (인과성·동일성) ---")
    # 0) 데이터 동결 (사전등록 해시와 불일치하면 즉시 실패)
    for pth, want in SHA_EXPECT.items():
        got = sha256(pth)
        assert got == want, (pth, got, want)
    print("  [OK] 데이터 3종 SHA256 = 사전등록 해시 (frozen MANIFEST 일치)")
    # 1) 기준선 팔 == confluence_gate_test (a)팔 (기존 경로 바이트 동일성)
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lab.confluence_gate_test import run as cg_run
    cdfr, ctl, _, _ = cg_run(data, fund, use_gate=False)
    bdfr, btl, _ = base
    assert len(btl) == len(ctl), (len(btl), len(ctl))
    assert abs(bdfr.equity.iloc[-1] - cdfr.equity.iloc[-1]) < 1e-12
    assert float((bdfr.equity - cdfr.equity).abs().max()) < 1e-12
    # 원장 자체 동일성 (곡선·건수만이 아니라 거래별 체결가·손익까지)
    for col in ('e', 'x', 'u', 'pnl'):
        assert float((btl[col].to_numpy() - ctl[col].to_numpy()).__abs__().max()) < 1e-12
    assert bool((btl.ts.to_numpy() == ctl.ts.to_numpy()).all())
    assert bool((btl.sym.to_numpy() == ctl.sym.to_numpy()).all())
    print(f"  [OK] 기준선 = confluence_gate_test (a)팔 원장 동일 "
          f"(거래 {len(btl)}건 전부 ts·심볼·체결가·손익 일치, "
          f"최종자본 {bdfr.equity.iloc[-1]:.6f})")
    # 2) ATR[i-1] 이 실제로 shift 되어 있고, 그 shift 가 결과를 바꾼다 (위반 대조군)
    raw = atr(data['BTC'])
    sh = raw.shift(1)
    assert sh.iloc[500] == raw.iloc[499] and sh.iloc[500] != raw.iloc[500]
    vio = run(data, fund, TP_PRESPEC, shift_atr=False)
    assert abs(vio[0].equity.iloc[-1] - tp15[0].equity.iloc[-1]) > 1e-9
    print(f"  [OK] ATR[i-1] 인과 교정이 결과를 지배 — 위반본(ATR[i]) 최종자본 "
          f"{vio[0].equity.iloc[-1]:.4f} vs 교정본 {tp15[0].equity.iloc[-1]:.4f}")
    # 3) 목표 체결가는 항상 진입 대비 +1.5R 정확히 (레벨 체결, 갭 이득 없음)
    tl = tp15[1]
    tp = tl[tl.reason.isin(('target', 'same_bar_target'))]
    r = ((tp.x - tp.e) * tp.d) / ((tp.e - tp.stop) * tp.d)
    assert len(tp) > 0 and float((r - TP_PRESPEC).abs().max()) < 1e-9
    assert bool((((tp.x - tp.e) * tp.d) > 0).all())
    print(f"  [OK] 익절 체결 {len(tp)}건 전부 정확히 +{TP_PRESPEC}R 레벨 체결 "
          f"(갭 유리분 미반영)")
    # 4) 동시도달 봉이 실제로 존재하고 전부 스탑/역채널로 처리됐다 (비관 강제)
    assert tp15[2] > 0
    print(f"  [OK] 스탑·목표 동시도달 {tp15[2]}건 — 전부 스탑/역채널 우선 청산")
    # 5) 청산은 진입 이후 (시간 역행 없음)
    assert bool((tl.hold_h >= 0).all())
    print("  [OK] 전 거래 청산시각 >= 진입시각")


if __name__ == '__main__':
    print("데이터 SHA256:")
    for pth in PATHS:
        print(f"  {pth}  {sha256(pth)}")
    data, fund, fundh = load()
    for s, df in data.items():
        print(f"  {s}: {df.index.min()} ~ {df.index.max()}  ({len(df)} 봉)")
    arms = {r: run(data, fund, r) for r in TP_GRID}
    selftest(data, fund, arms[None], arms[TP_PRESPEC])

    print("\n" + "=" * 78)
    print("주 비교 — BRK24 원본(익절 없음) vs BRK24 + 1.5R 익절")
    print("1.5 는 사용자가 결과 조회 전 지정한 값이다 (사후선택 아님).")
    print("=" * 78)
    S = {}
    S[None] = report("(a) BRK24 교정판 원본 — 익절 없음", *arms[None])
    S[TP_PRESPEC] = report(f"(b) BRK24 + {TP_PRESPEC}R 익절 (사전 지정)",
                           *arms[TP_PRESPEC])

    print("\n" + "=" * 78)
    print("공시용 격자 (참고 표시 전용 — 셀 선택에 사용되지 않음).")
    print("1.5 는 결과 조회 전 지정됐고, 이 격자는 그 사실의 검증 자료일 뿐이다.")
    print("=" * 78)
    for r in TP_GRID:
        if r not in S:
            S[r] = stats(arms[r][0], arms[r][1])
    print(f"{'익절':>6s} {'거래':>6s} {'승률':>7s} {'평균/거래':>9s} {'누적순익':>9s} "
          f"{'CAGR':>7s} {'MDD일':>7s} {'익절도달':>8s} {'평균보유':>8s}")
    for r in TP_GRID:
        st = S[r]
        nm = '없음' if r is None else f'{r:.1f}R'
        print(f"{nm:>6s} {st['n']:6d} {st['win']*100:6.1f}% {st['avg']*100:+8.3f}% "
              f"{st['net']*100:+8.1f}% {st['cagr']*100:+6.1f}% {st['mdd_d']*100:6.1f}% "
              f"{st['tp_rate']*100:7.1f}% {st['hold']:7.1f}h")
    print("\n연도별 수익률 (전 격자 셀):")
    ys = sorted(S[None]['daily'].index.year.unique())
    print(f"{'익절':>6s} " + " ".join(f"{y:>8d}" for y in ys))
    for r in TP_GRID:
        d = S[r]['daily']
        yr = (1 + d.pct_change()).groupby(d.index.year).prod() - 1
        nm = '없음' if r is None else f'{r:.1f}R'
        print(f"{nm:>6s} " + " ".join(f"{yr.get(y, float('nan'))*100:+7.1f}%"
                                      for y in ys))
