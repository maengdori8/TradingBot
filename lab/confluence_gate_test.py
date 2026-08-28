"""컨플루언스 게이트 백테스트 — BRK24(교정판) vs BRK24+3중 게이트. U1 사전 기준선 기록용.

사전 고정 (실행 전 동결 — 튜닝·조합 탐색 절대 금지, 전부 표준 출판값):
- 기반: lab/scalp_grid.py BRK24 (1h 채널 돌파 N=24, 6xATR(24) 스탑, N/2 역채널 추적,
  리스크 2%, 그로스 캡 10x, 일손실 -5% 정지, 왕복 16bp, 일별 펀딩).
  실행 교정 (scalp_farm 교정 1·2): ATR[i-1] 사용(같은 봉 완성 ATR 금지),
  TR 은 previous close 기준 (scalp_grid.atr 원형이 이미 prev-close — shift(1)만 추가).
- 게이트 (진입 신호 AND, 롱 기준·숏 대칭, 전부 확정봉 i-1 기준 — 같은 봉 금지):
  ① 추세: close[i-1] > SMA200(1h)[i-1]            (숏: <)
  ② 모멘텀: RSI14(Wilder)[i-1] > 50               (숏: <)
  ③ 거래량: vol[i-1] > mean(vol[i-21..i-2])       (롱숏 공통 — 직전 20봉 평균, i-1 제외)
  게이트 값 결측(NaN, 워밍업 포함) = 차단 (fail-closed).
- 데이터: lab/frozen/perp_1h.parquet BTC/ETH + lab/data/sol_1h.parquet SOL (전 기간,
  volume 포함 확인됨 → 3중 게이트 실행). 펀딩: lab/frozen/funding.parquet 일합.
- 비교: (a) BRK24 교정판 원본 vs (b) +게이트 — 동일 실행 모델·동일 비용.
  거래수·승률·평균 수익/거래·누적 순수익(비용 후)·MDD·연도별 분해·게이트 통과율.
- 판정 없음: 결과와 무관하게 U1 편입 예정 — 이 스크립트는 기준선 기록 전용.

계승 한계 (scalp_grid 원본 충실 재현 — 사전등록 범위 밖이라 미수정, Codex 검토 확인):
- 펀딩: 당일 '일합'을 일 시작에 차감 (당일 08/16시 정산분 선반영 — 양팔 동일).
- 그로스 캡 마크: 기존 포지션을 close[i]로 평가 (봉내 체결 시점 미지값 —
  Codex 진단: close[i-1] 대체 시 본 데이터셋 결과 불변).
- 이중 돌파봉(상하 동시): 롱 우선 해석 후 게이트 적용 (게이트 선주문 해석과
  다를 수 있는 봉 = 전 기간 3봉).
- RSI 초기화: SMA 시드 없는 ewm — SMA200 워밍업이 지배하므로 실질 무영향.
측정 교정 (전략·게이트 불변, 양팔 동일 — 기준선 왜곡 방지):
- 심볼 데이터 종료 시 보유 포지션은 마지막 유효 종가로 강제청산(비용 차감) —
  미실현 손익 증발 방지. 누적수익·CAGR·MDD는 초기자본 1.0 앵커 기준.
- 승률·평균수익/거래는 가격손익+수수료 기준 (펀딩은 자본 차감, 거래 미배분).
"""
from __future__ import annotations
import hashlib
import numpy as np, pandas as pd

np.random.seed(0)                       # 무작위성 없음 — 형식상 고정
COST_IN, COST_OUT = 0.0008, 0.0008      # 편도 8bp(taker+슬립) = 왕복 16bp
N, RISK, GROSS_CAP, DAILY_HALT = 24, 0.02, 10.0, -0.05
PATHS = ('lab/frozen/perp_1h.parquet', 'lab/data/sol_1h.parquet',
         'lab/frozen/funding.parquet')


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
    f = pd.read_parquet(PATHS[2])[['BTC', 'ETH', 'SOL']].resample('D').sum(min_count=1)
    return d, f


def atr(df: pd.DataFrame, n: int = 24) -> pd.Series:
    """ATR — TR은 previous close 기준 (scalp_grid 원형)."""
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (출판 표준형 — ewm alpha=1/n)."""
    diff = close.diff()
    ru = diff.clip(lower=0.0).ewm(alpha=1 / n, adjust=False).mean()
    rd = (-diff).clip(lower=0.0).ewm(alpha=1 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd)


def gates(df: pd.DataFrame):
    """롱/숏 게이트 불리언 (인덱스 i에서 전부 i-1 확정값 — NaN은 False=차단)."""
    sma = df.close.rolling(200).mean()
    rsi = rsi_wilder(df.close, 14)
    gv = df.volume.shift(1) > df.volume.rolling(20).mean().shift(2)
    gl = (df.close.shift(1) > sma.shift(1)) & (rsi.shift(1) > 50) & gv
    gs = (df.close.shift(1) < sma.shift(1)) & (rsi.shift(1) < 50) & gv
    return gl.fillna(False), gs.fillna(False)


def run(data: dict, fund: pd.DataFrame, use_gate: bool):
    """BRK24 교정판 실행 — scalp_grid.run(BRK, N=24, risk 2%) 로직 + 교정 + 게이트.

    Returns:
        (일별 곡선용 DataFrame, 트레이드 로그 DataFrame, 신호수, 차단수)
    """
    syms = list(data)
    idx = None
    for s in syms:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()
    D = {s: data[s].reindex(idx) for s in syms}
    A = {s: atr(D[s]).shift(1) for s in syms}          # 교정 1: ATR[i-1]
    HI = {s: D[s].high.rolling(N).max().shift(1) for s in syms}
    LO = {s: D[s].low.rolling(N).min().shift(1) for s in syms}
    XH = {s: D[s].high.rolling(N // 2).max().shift(1) for s in syms}
    XL = {s: D[s].low.rolling(N // 2).min().shift(1) for s in syms}
    GL, GS = {}, {}
    for s in syms:
        GL[s], GS[s] = gates(D[s])

    eq, pos, rows, tlog = 1.0, {}, [], []
    sig_total, sig_blocked = 0, 0
    day, day_eq, halted = None, 1.0, False
    for i, t in enumerate(idx):
        if i < 100:
            continue
        if t.date() != day:
            day, halted = t.date(), False
            day_eq = eq + sum(p['u'] * (D[s].close.iloc[i - 1] - p['e']) * p['d']
                              for s, p in pos.items()
                              if not np.isnan(D[s].close.iloc[i - 1]))
            for s, p in pos.items():                   # 펀딩 (일 1회, 롱 지불)
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
                    # 체결가·기록시각 = 마지막 유효봉 idx[i-1] (귀속 정확).
                    # 청산비용의 자본 반영은 이 봉(i)부터 — 시간상 1봉 차이는
                    # 청산비용뿐. 일경계 펀딩 선차감 엣지(23시 종료 심볼)는
                    # 동결 데이터에 부재 (Codex 확인).
                    px = D[s].close.iloc[i - 1]
                    if not np.isnan(px):
                        eq += p['u'] * (px - p['e']) * p['d'] - p['u'] * px * COST_OUT
                        tlog.append(dict(ts=idx[i - 1], sym=s, d=p['d'], e=p['e'],
                                         x=px, u=p['u'], eq0=p['eq0'],
                                         pnl=p['u'] * (px - p['e']) * p['d']
                                         - p['u'] * p['e'] * COST_IN
                                         - p['u'] * px * COST_OUT))
                        pos.pop(s)
                continue
            if np.isnan(a) or a <= 0:
                continue
            p = pos.get(s)
            if p:
                exit_px = None
                if p['d'] > 0:
                    lvl = max(p['stop'], XL[s].iloc[i])
                    if l <= lvl:
                        exit_px = min(lvl, o)
                else:
                    lvl = min(p['stop'], XH[s].iloc[i])
                    if h >= lvl:
                        exit_px = max(lvl, o)
                if exit_px is not None:
                    eq += p['u'] * (exit_px - p['e']) * p['d'] - p['u'] * exit_px * COST_OUT
                    tlog.append(dict(ts=t, sym=s, d=p['d'], e=p['e'], x=exit_px,
                                     u=p['u'], eq0=p['eq0'],
                                     pnl=p['u'] * (exit_px - p['e']) * p['d']
                                     - p['u'] * p['e'] * COST_IN
                                     - p['u'] * exit_px * COST_OUT))
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
                d_, fill = 1, max(o, HI[s].iloc[i])
            elif l < LO[s].iloc[i]:
                d_, fill = -1, min(o, LO[s].iloc[i])
            if not d_:
                continue
            sig_total += 1
            gpass = bool(GL[s].iloc[i] if d_ > 0 else GS[s].iloc[i])
            if not gpass:
                sig_blocked += 1       # (a)팔에선 섀도 계수 — 행동 무영향
                if use_gate:
                    continue
            stop = fill - d_ * 6 * a
            u = min(RISK * eq / abs(fill - stop),
                    max(0.0, (GROSS_CAP * eq - gross)) / fill)
            if u <= 0:
                continue
            eq0 = eq
            eq -= u * fill * COST_IN
            if (d_ > 0 and l <= stop) or (d_ < 0 and h >= stop):   # 같은 봉 스탑 (비관)
                eq += u * (stop - fill) * d_ - u * stop * COST_OUT
                tlog.append(dict(ts=t, sym=s, d=d_, e=fill, x=stop, u=u, eq0=eq0,
                                 pnl=u * (stop - fill) * d_ - u * fill * COST_IN
                                 - u * stop * COST_OUT))
                continue
            pos[s] = dict(d=d_, u=u, e=fill, stop=stop, eq0=eq0)
        mtm = eq + sum(p['u'] * (D[s].close.iloc[i] - p['e']) * p['d']
                       for s, p in pos.items() if not np.isnan(D[s].close.iloc[i]))
        if not halted and day_eq > 0 and mtm / day_eq - 1 < DAILY_HALT and pos:
            for s, p in list(pos.items()):
                px = D[s].close.iloc[i]
                if np.isnan(px):
                    continue
                eq += p['u'] * (px - p['e']) * p['d'] - p['u'] * px * COST_OUT
                tlog.append(dict(ts=t, sym=s, d=p['d'], e=p['e'], x=px, u=p['u'],
                                 eq0=p['eq0'],
                                 pnl=p['u'] * (px - p['e']) * p['d']
                                 - p['u'] * p['e'] * COST_IN - p['u'] * px * COST_OUT))
                pos.pop(s)
            halted, mtm = True, eq
        rows.append((t, mtm))
    for s, p in list(pos.items()):     # 기간 말 잔여 포지션 — 마지막 유효 종가 청산
        px = D[s].close.dropna().iloc[-1]
        eq += p['u'] * (px - p['e']) * p['d'] - p['u'] * px * COST_OUT
        tlog.append(dict(ts=idx[-1], sym=s, d=p['d'], e=p['e'], x=px, u=p['u'],
                         eq0=p['eq0'],
                         pnl=p['u'] * (px - p['e']) * p['d']
                         - p['u'] * p['e'] * COST_IN - p['u'] * px * COST_OUT))
        pos.pop(s)
    if rows:
        rows[-1] = (rows[-1][0], eq)   # 최종점 = 청산 완료 자본
    dfr = pd.DataFrame(rows, columns=['ts', 'equity']).set_index('ts')
    return dfr, pd.DataFrame(tlog), sig_total, sig_blocked


def gate_pass_static(data: dict) -> tuple:
    """경로 독립 게이트 통과율 — 방향별 돌파 트리거 전수 집계 (지표 유병률)."""
    tot, ok = 0, 0
    for s, df in data.items():
        hi = df.high.rolling(N).max().shift(1)
        lo = df.low.rolling(N).min().shift(1)
        gl, gs = gates(df)
        lt = df.high > hi
        st = df.low < lo
        tot += int(lt.sum() + st.sum())
        ok += int((lt & gl).sum() + (st & gs).sum())
    return tot, ok


def report(name: str, dfr: pd.DataFrame, tl: pd.DataFrame,
           sig: int, blk: int, shadow: bool) -> None:
    """한 팔(arm) 결과표 출력 — 초기자본 1.0 앵커, MDD는 일봉/시봉 병기."""
    anchor = pd.Series([1.0], index=[dfr.index[0] - pd.Timedelta(hours=1)])
    d = pd.concat([anchor, dfr.equity.resample('D').last().dropna()])
    eh = pd.concat([anchor, dfr.equity])
    yrs = (dfr.index[-1] - anchor.index[0]).total_seconds() / (365.25 * 86400)
    net = d.iloc[-1] - 1.0
    cagr = d.iloc[-1] ** (1 / yrs) - 1
    mdd_d = (1 - d / d.cummax()).max()
    mdd_h = (1 - eh / eh.cummax()).max()
    ntr = len(tl)
    win = (tl.pnl > 0).mean() if ntr else float('nan')
    avg = (tl.pnl / tl.eq0).mean() if ntr else float('nan')
    print(f"\n=== {name} ===")
    print(f"거래수 {ntr}  승률 {win*100:.1f}%  평균수익/거래 {avg*100:+.3f}%"
          f"(진입시 자본대비, 가격손익+수수료 — 펀딩 제외)")
    print(f"누적 순수익 {net*100:+.1f}%  CAGR {cagr*100:+.1f}%  "
          f"MDD(일봉) {mdd_d*100:.1f}%  MDD(1h) {mdd_h*100:.1f}%")
    tag = "섀도 차단(행동 무영향)" if shadow else "게이트 차단"
    print(f"돌파신호 {sig}건 중 {tag} {blk}건 ({blk/sig*100 if sig else 0:.1f}%)")
    yr = (1 + d.pct_change()).groupby(d.index.year).prod() - 1
    ty = tl.groupby(tl.ts.dt.year).size() if ntr else pd.Series(dtype=int)
    wy = tl.groupby(tl.ts.dt.year).pnl.apply(lambda x: (x > 0).mean()) if ntr else {}
    print(f"{'연도':>6s} {'수익률':>9s} {'거래':>6s} {'승률':>7s}")
    for y in yr.index:
        print(f"{y:>6d} {yr[y]*100:+8.1f}% {ty.get(y, 0):6d} "
              f"{wy.get(y, float('nan'))*100:6.1f}%")


if __name__ == '__main__':
    print("데이터 SHA256:")
    for pth in PATHS:
        print(f"  {pth}  {sha256(pth)}")
    data, fund = load()
    for s, df in data.items():
        print(f"  {s}: {df.index.min()} ~ {df.index.max()}  ({len(df)} 봉, "
              f"volume {'유' if 'volume' in df else '무'})")
    tot, ok = gate_pass_static(data)
    print(f"\n[경로독립] 전 봉 방향별 돌파트리거 {tot}건 중 게이트 통과 {ok}건 "
          f"({ok/tot*100:.1f}%) — 차단율 {(tot-ok)/tot*100:.1f}% "
          f"(포지션·정지·워밍업 무시한 지표 유병률)")
    a = run(data, fund, use_gate=False)
    b = run(data, fund, use_gate=True)
    report("(a) BRK24 교정판 원본", *a, shadow=True)
    report("(b) BRK24 + 3중 게이트(SMA200·RSI14·거래량)", *b, shadow=False)
