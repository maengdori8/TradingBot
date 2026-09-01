"""BBADD U1 사전 기준선 — 볼린저 하단 추매(트랜치 피라미딩-다운) vs 순수 BBMR.

사전 고정 (결과 조회 전 동결 — 튜닝·사후선택·임의 변경 금지):
- 전략명 **BBADD**. 라이브 셀 E24 = 바스켓 A(BTC/ETH/SOL), E25 = 바스켓 B
  (XRP/HYPE/BTR), 셀당 신규 $10,000. 본 스크립트는 E24 대응 바스켓 A 의
  U1 사전 기준선이다 (바스켓 B 는 동결 데이터 없음 — HYPE/BTR 이력 부재로
  백테스트 불가, 공시).
- 라벨 (동결): "볼린저 추매 변형 · 승률 제조 구조 시연 · 미검증 · 판정 권한 없음".
- 규칙 (전부 확정봉 신호 → 다음 봉 시가 체결, U1 실행 규약):
  · 진입: 확정봉 종가 < BB(20, 2σ, ddof=0 모표준편차, 중심 SMA20) 하단밴드
    → 다음 봉 시가에 트랜치 1 매수. 롱 온리 (출판 BBMR 관례 = E15 계열).
  · 추매: 보유 중 확정봉 종가 <= 직전 트랜치 체결가 − 1.0×ATR(24)[i−1]
    (i = 체결봉, TR 은 previous close 기준) → 다음 봉 시가에 트랜치 1 추가.
    **최대 3회 추매 (총 4트랜치 하드캡)** — 소진 후엔 어떤 조건에도 추가 없음.
  · 익절: 확정봉 종가 >= SMA20 (중심선) → 다음 봉 시가에 전량 청산.
  · **포지션 스탑 없음** (출판 BBMR 관례 유지 — 이 부재가 승률을 '제조'하는
    메커니즘: 추매로 평단을 낮춰 중심선 회귀 시 손실 거래가 승리로 전환되는
    대신, 회귀하지 않는 추세 하락에서 만재 명목의 드로다운이 꼬리에 쌓인다).
- 사이징 (기존 무스탑 셀 관례 = E15~E18): 심볼별 예산 = equity × 1/3 (3슬롯
  균등), 트랜치 명목 = 예산/4 = equity × 1/12 — 각 트랜치는 자기 체결 시점
  equity 기준 (라이브 팜 _try_open 의 '진입 승인 시점 equity' 관례).
- 공통 제약: 비용 왕복 16bp · 일별 펀딩(일합 선차감, scalp_grid 계승) ·
  gross 10x · heat 캡 6% (무스탑 heat 기여 = 명목 × 5% 대리, E15 동결 정의 —
  손실 상한 아님, 진입 차단용 대리변수) · 일손실 −5% **진입정지** (무스탑 셀
  관례 — rr15/scalp_grid 의 '전량청산' 계승 한계와 다름을 명시: BBADD 사전
  규칙이 '진입정지'로 지정했고 라이브 팜 공통 규약과 일치).
- 사전 예측 (결과 조회 전 문서 기록 — 실측과 나란히 공시):
  ① 승률 65~80% 예상. ② 기대값 개선 없음 예상 (순수 BBMR 대비 거래당·누적).
  ③ 추세 하락장에서 4트랜치 만재 상태의 깊은 드로다운 (꼬리) 예상.
  본 그룹의 목적 = 승률과 기대값이 분리되는 구조의 라이브 시연.
- 판정 없음: 이 스크립트는 기준선 기록 전용이다.

실행 세부 (전례 계승 — rr15_test.py 구조, scalp_farm.py 팜 규약):
- 같은 봉 청산 심볼 재진입 금지 (팜 위상 규약 (i)).
- 같은 봉 익절·추매 신호 동시 성립 시 익절 우선 (전량 청산, 추매 없음).
- gross 마크: 기존 포지션 = 직전 확정봉 종가(결측 시 평단), 같은 봉 선행
  체결 심볼 = 체결가 (팜 px 맵 관례 — rr15 의 close[i] 마크 한계 미계승,
  체결 시점에 이번 봉 종가는 미지이므로 인과 무결).
- 심볼 처리 순서 BTC→ETH→SOL 고정 (sweep_engine D1 관례).
- 결측·데이터 종료: fail-closed — 신호 NaN 무행동, 심볼 데이터 종료 시
  마지막 유효 종가 강제청산(eod, 비용 차감).
측정 교정 (rr15 관례): 초기자본 1.0 앵커, 누적·CAGR·MDD 는 자본곡선 기준,
승률·평균수익/거래 = 가격손익+수수료 (펀딩은 자본 차감, 거래 미배분),
보유시간 = 청산봉 ts − 첫 트랜치 진입봉 ts. 만재 꼬리 분석용으로 보유 중
최저 미실현손익(봉 저가 마크 — 봉내 순서 미지라 보수 평가)을 추적한다.

실행: .venv/bin/python lab/bbadd_test.py   (커밋 금지 — 기준선 기록 전용)
"""
from __future__ import annotations
import hashlib
import numpy as np, pandas as pd

np.random.seed(0)                       # 무작위성 없음 — 형식상 고정
COST_IN, COST_OUT = 0.0008, 0.0008      # 편도 8bp(taker+슬립) = 왕복 16bp
BB_N, BB_K, ATR_N, ADD_ATR = 20, 2.0, 24, 1.0
N_TRANCHE = 4                           # 1 진입 + 최대 3 추매 (하드캡)
SLOT_FRAC = 1.0 / 3.0                   # 심볼 예산 = equity × 1/3
TRANCHE_FRAC = SLOT_FRAC / N_TRANCHE    # 트랜치 명목 = equity × 1/12
GROSS_CAP, DAILY_HALT = 10.0, -0.05
HEAT_CAP, HEAT_FRAC = 0.06, 0.05        # 무스탑 heat 기여 = 명목 × 5% (E15 동결)
SYMS = ('BTC', 'ETH', 'SOL')            # 처리 순서 고정 (D1)
PATHS = ('lab/frozen/perp_1h.parquet', 'lab/data/sol_1h.parquet',
         'lab/frozen/funding.parquet')
SHA_EXPECT = {                          # 데이터 동결 검증 (rr15_test 와 동일 3종)
    'lab/frozen/perp_1h.parquet':
        'c06a3301457dfec8f68b184e1f8ac8797acc49874fa96c409a6a57e6743ebac0',
    'lab/data/sol_1h.parquet':
        '80d7f7574d680505eb280d05bfecc4ce3bc38ff461e651b91eb122e666f04785',
    'lab/frozen/funding.parquet':
        '534642ee677424abb949492a7b8f21e43ea635bd0eb9f980e12f829f89a0a128',
}
PREDICTIONS = (
    "① 승률 65~80% 예상 (추매 평단 인하가 손실 거래를 승리로 전환)",
    "② 기대값 개선 없음 예상 (순수 BBMR 대비 거래당·누적 — 승률≠기대값)",
    "③ 추세 하락장 4트랜치 만재 상태의 깊은 드로다운 (꼬리) 예상",
)


def sha256(path: str) -> str:
    """파일 SHA256 (데이터 동결 확인용)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load():
    """BTC/ETH(frozen)+SOL(lab/data) 1h OHLCV, 일별 펀딩 — rr15_test.load 동형."""
    p = pd.read_parquet(PATHS[0])
    cols = ['open', 'high', 'low', 'close', 'volume']
    d = {s: p.xs(s, level='sym')[cols] for s in ('BTC', 'ETH')}
    d['SOL'] = pd.read_parquet(PATHS[1])[cols]
    fh = pd.read_parquet(PATHS[2])[list(SYMS)]
    return d, fh.resample('D').sum(min_count=1)


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    """ATR — TR 은 previous close 기준 (scalp_grid 원형, Wilder ewm)."""
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def run(data: dict, fund: pd.DataFrame, n_tranche: int = N_TRANCHE,
        tranche_frac: float = TRANCHE_FRAC, causal: bool = True):
    """BBADD 실행 (n_tranche=1 이면 순수 BBMR — 추매 경로 완전 비활성).

    Args:
        data: sym -> OHLCV DataFrame (UTC DatetimeIndex).
        fund: 일별 펀딩률 합 (열 = 심볼) — 일 시작 선차감, 롱 지불 (rr15 계승).
        n_tranche: 트랜치 하드캡. 4 = BBADD 본안, 1 = 순수 BBMR 비교군.
        tranche_frac: 트랜치 명목 / equity. 본안 1/12, E15형 참고팔은 1/3.
        causal: False 면 신호를 체결봉 자신의 종가·지표로 평가 (**룩어헤드
            위반 대조군 전용** — selftest 가 결과 상이를 강제). 본 실행은 True.

    Returns:
        (자본곡선 DataFrame, 트레이드 로그 DataFrame, 차단 카운터 dict).
    """
    idx = None
    for s in SYMS:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()
    D = {s: data[s].reindex(idx) for s in SYMS}
    sh = 1 if causal else 0
    C1, MID, LOWB, A = {}, {}, {}, {}
    for s in SYMS:
        c = D[s].close
        m = c.rolling(BB_N).mean()
        sd = c.rolling(BB_N).std(ddof=0)          # 모표준편차 (출판 BB 관례)
        C1[s] = c.shift(sh)                       # 확정봉 종가 (신호)
        MID[s] = m.shift(sh)
        LOWB[s] = (m - BB_K * sd).shift(sh)
        A[s] = atr(D[s]).shift(sh)                # ATR[i-1] (교정 1)

    eq, pos, rows, tlog = 1.0, {}, [], []
    blocked = dict(gross=0, heat=0, halt_entry=0, halt_add=0)
    day, day_eq, halted = None, 1.0, False

    def log(t_out, s, p, x, reason):
        """청산 1건 기록 — pnl 은 가격손익 + 전 트랜치 진입·청산 수수료."""
        fee = p['fees'] + p['u'] * x * COST_OUT
        tlog.append(dict(
            ts=t_out, ts_in=p['t0'], sym=s, k=len(p['tr']), u=p['u'],
            e=p['basis'] / p['u'], x=x, eq0=p['eq0'], reason=reason,
            hold_h=(t_out - p['t0']).total_seconds() / 3600.0,
            full_h=((t_out - p['ts_tr'][-1]).total_seconds() / 3600.0
                    if len(p['tr']) == n_tranche else float('nan')),
            fee=fee, pnl=p['u'] * x - p['basis'] - fee,
            min_unreal=p['min_unreal'],
            tr=tuple(p['tr']), ts_tr=tuple(p['ts_tr']), sig=tuple(p['sig']),
            exit_sig=p.get('exit_sig')))

    def close_all(t_out, s, p, x, reason):
        nonlocal eq
        eq += p['u'] * x - p['basis'] - p['u'] * x * COST_OUT
        log(t_out, s, p, x, reason)
        pos.pop(s)

    def try_fill(t, i, s, o, fillmark, sig):
        """트랜치 1개 체결 시도 — 명목 사이징 + gross/heat 캡 (팜 _try_open 관례)."""
        nonlocal eq
        if eq <= 0 or o <= 0 or np.isnan(o):
            return
        gross = 0.0
        for s2, p2 in pos.items():
            mk = fillmark.get(s2, D[s2].close.iloc[i - 1])
            if np.isnan(mk):
                mk = p2['basis'] / p2['u']        # 결측 마크 폴백 = 평단 (팜 pp.e)
            gross += p2['u'] * mk
        if gross >= GROSS_CAP * eq:
            blocked['gross'] += 1
            return
        u = min(tranche_frac * eq / o, max(0.0, GROSS_CAP * eq - gross) / o)
        heat = sum(e * uu * HEAT_FRAC for p2 in pos.values() for e, uu in p2['tr'])
        if u <= 0:
            return
        if heat + u * o * HEAT_FRAC > HEAT_CAP * eq * (1 + 1e-9):
            blocked['heat'] += 1
            return
        p = pos.get(s)
        if p is None:
            p = dict(tr=[], ts_tr=[], sig=[], basis=0.0, u=0.0, fees=0.0,
                     t0=t, eq0=eq, min_unreal=0.0)
            pos[s] = p
        eq -= u * o * COST_IN
        p['fees'] += u * o * COST_IN
        p['tr'].append((o, u))
        a1f = A[s].iloc[i]                      # 체결봉 i 의 ATR[i-1] (shift 적용됨)
        p['thr'] = (o - ADD_ATR * a1f) if (not np.isnan(a1f) and a1f > 0) else np.nan
        p['ts_tr'].append(t)
        p['sig'].append(sig)
        p['basis'] += u * o
        p['u'] += u
        fillmark[s] = o

    for i, t in enumerate(idx):
        if i < 100:                               # 워밍업 — 주문 생성 금지
            continue
        if t.date() != day:
            day, halted = t.date(), False
            day_eq = eq + sum(p['u'] * D[s].close.iloc[i - 1] - p['basis']
                              for s, p in pos.items()
                              if not np.isnan(D[s].close.iloc[i - 1]))
            for s, p in pos.items():              # 펀딩 (일 1회 선차감, 롱 지불)
                f = fund[s].get(pd.Timestamp(day, tz='utc'), np.nan)
                px = D[s].close.iloc[i - 1]
                if not np.isnan(f) and not np.isnan(px):
                    eq -= f * p['u'] * px
        fillmark = {}                             # 같은 봉 선행 체결 마크 (팜 px 맵)
        for s in SYMS:
            o, l, c = D[s].open.iloc[i], D[s].low.iloc[i], D[s].close.iloc[i]
            p = pos.get(s)
            if np.isnan(c):
                if p is not None:                 # 심볼 데이터 종료 — 강제청산
                    px = D[s].close.iloc[i - 1]
                    if not np.isnan(px):
                        close_all(idx[i - 1], s, p, px, 'eod')
                continue
            c1, m1, lb1, a1 = (C1[s].iloc[i], MID[s].iloc[i],
                               LOWB[s].iloc[i], A[s].iloc[i])
            if p is not None:
                if not np.isnan(c1) and not np.isnan(m1) and c1 >= m1 \
                        and not np.isnan(o):      # 익절 (전량, 같은 봉 재진입 금지)
                    p['exit_sig'] = (c1, m1)
                    close_all(t, s, p, o, 'exit')
                elif (len(p['tr']) < n_tranche and not np.isnan(c1)
                      and not np.isnan(p.get('thr', np.nan))
                      and c1 <= p['thr']):        # 추매 (임계 = 체결 시 동결 — 엔진 #a)
                    if halted:
                        blocked['halt_add'] += 1
                    else:
                        try_fill(t, i, s, o, fillmark, ('add', c1, p['thr']))
            elif not np.isnan(c1) and not np.isnan(lb1) and c1 < lb1:
                if halted:                        # 신규 진입
                    blocked['halt_entry'] += 1
                else:
                    try_fill(t, i, s, o, fillmark, ('entry', c1, lb1))
            p = pos.get(s)
            if p is not None and not np.isnan(l):  # 만재 꼬리 추적 (봉 저가 마크)
                p['min_unreal'] = min(p['min_unreal'], p['u'] * l - p['basis'])
        mtm = eq + sum(p['u'] * D[s].close.iloc[i] - p['basis']
                       for s, p in pos.items() if not np.isnan(D[s].close.iloc[i]))
        if not halted and day_eq > 0 and mtm / day_eq - 1 < DAILY_HALT:
            halted = True                         # 진입정지만 — 청산 없음 (동결 규칙)
        rows.append((t, mtm))
    for s, p in list(pos.items()):                # 기간 말 잔여 — 마지막 유효 종가
        px = D[s].close.dropna().iloc[-1]
        close_all(idx[-1], s, p, px, 'eod')
    if rows:
        rows[-1] = (rows[-1][0], eq)              # 최종점 = 청산 완료 자본
    dfr = pd.DataFrame(rows, columns=['ts', 'equity']).set_index('ts')
    return dfr, pd.DataFrame(tlog), blocked


def stats(dfr: pd.DataFrame, tl: pd.DataFrame) -> dict:
    """한 팔의 요약 지표 — 초기자본 1.0 앵커 (rr15 관례)."""
    anchor = pd.Series([1.0], index=[dfr.index[0] - pd.Timedelta(hours=1)])
    d = pd.concat([anchor, dfr.equity.resample('D').last().dropna()])
    eh = pd.concat([anchor, dfr.equity])
    yrs = (dfr.index[-1] - anchor.index[0]).total_seconds() / (365.25 * 86400)
    n = len(tl)
    ret = tl.pnl / tl.eq0 if n else pd.Series(dtype=float)
    w, ls = ret[tl.pnl > 0] if n else ret, ret[tl.pnl <= 0] if n else ret
    return dict(
        n=n, win=(tl.pnl > 0).mean() if n else float('nan'),
        avg=ret.mean() if n else float('nan'),
        avg_se=ret.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan'),
        w_avg=w.mean() if len(w) else float('nan'),
        l_avg=ls.mean() if len(ls) else float('nan'),
        worst=ret.min() if n else float('nan'),
        net=d.iloc[-1] - 1.0, cagr=d.iloc[-1] ** (1 / yrs) - 1,
        mdd_d=(1 - d / d.cummax()).max(), mdd_h=(1 - eh / eh.cummax()).max(),
        hold=tl.hold_h.mean() if n else float('nan'),
        hold_med=tl.hold_h.median() if n else float('nan'),
        fee_ret=(tl.fee / tl.eq0).mean() if n else float('nan'),
        daily=d)


def report(name: str, dfr: pd.DataFrame, tl: pd.DataFrame, blocked: dict) -> dict:
    """한 팔(arm) 결과표 출력 — rr15 report 형식 계승."""
    st = stats(dfr, tl)
    d = st['daily']
    print(f"\n=== {name} ===")
    print(f"거래수 {st['n']}  승률 {st['win']*100:.1f}%  "
          f"평균수익/거래 {st['avg']*100:+.3f}% (SE {st['avg_se']*100:.3f}%)"
          f"  (진입시 자본대비, 가격손익+수수료 — 펀딩 제외)")
    print(f"누적 순수익 {st['net']*100:+.1f}%  CAGR {st['cagr']*100:+.1f}%  "
          f"MDD(일봉) {st['mdd_d']*100:.1f}%  MDD(1h) {st['mdd_h']*100:.1f}%")
    print(f"승 평균 {st['w_avg']*100:+.3f}%  패 평균 {st['l_avg']*100:+.3f}%  "
          f"손익비 {abs(st['w_avg']/st['l_avg']):.2f}  "
          f"최대 단일손실 {st['worst']*100:+.2f}%  "
          f"평균 보유 {st['hold']:.1f}h (중앙값 {st['hold_med']:.1f}h)  "
          f"수수료 {st['fee_ret']*100:.3f}%/거래")
    rc = tl.reason.value_counts() if st['n'] else {}
    print("청산사유: " + "  ".join(f"{k} {int(v)}" for k, v in rc.items())
          + f"   차단: gross {blocked['gross']} heat {blocked['heat']} "
            f"일손실정지(진입 {blocked['halt_entry']}·추매 {blocked['halt_add']})")
    yr = (1 + d.pct_change()).groupby(d.index.year).prod() - 1
    ty = tl.groupby(tl.ts.dt.year).size() if st['n'] else pd.Series(dtype=int)
    wy = (tl.groupby(tl.ts.dt.year).pnl.apply(lambda x: (x > 0).mean())
          if st['n'] else {})
    print(f"{'연도':>6s} {'수익률':>9s} {'거래':>6s} {'승률':>7s}")
    for y in yr.index:
        print(f"{y:>6d} {yr[y]*100:+8.1f}% {ty.get(y, 0):6d} "
              f"{wy.get(y, float('nan'))*100:6.1f}%")
    return st


def tranche_report(tl: pd.DataFrame) -> None:
    """(b) 트랜치 분해 + (c) 4트랜치 만재 분포 + (d) 최대 단일 손실."""
    ret = tl.pnl / tl.eq0
    mu = tl.min_unreal / tl.eq0
    print("\n(b) 트랜치 분해 — 최종 트랜치 수별 빈도·승률·평균손익")
    print(f"{'트랜치':>6s} {'거래':>6s} {'비중':>7s} {'승률':>7s} {'평균/거래':>9s} "
          f"{'최악/거래':>9s} {'평균보유':>8s} {'최저미실현':>10s}")
    for k in sorted(tl.k.unique()):
        m = tl.k == k
        print(f"{k:>6d} {int(m.sum()):6d} {m.mean()*100:6.1f}% "
              f"{(tl.pnl[m] > 0).mean()*100:6.1f}% {ret[m].mean()*100:+8.3f}% "
              f"{ret[m].min()*100:+8.2f}% {tl.hold_h[m].mean():7.1f}h "
              f"{mu[m].min()*100:+9.2f}%")
    m4 = tl.k == 4
    if m4.any():
        r4 = ret[m4].sort_values()
        print(f"\n(c) 4트랜치 만재 후 결과 분포 (꼬리) — {int(m4.sum())}건, "
              f"승률 {(tl.pnl[m4] > 0).mean()*100:.1f}%")
        q = r4.quantile([0, .05, .25, .5, .75, .95, 1.0])
        print("   pnl/eq0: " + "  ".join(
            f"{lbl} {q[p]*100:+.2f}%" for lbl, p in
            [('min', 0), ('p5', .05), ('p25', .25), ('중앙', .5),
             ('p75', .75), ('p95', .95), ('max', 1.0)]))
        qm = mu[m4].quantile([0, .05, .25, .5])
        print("   보유 중 최저 미실현 (봉 저가 마크): " + "  ".join(
            f"{lbl} {qm[p]*100:+.2f}%" for lbl, p in
            [('min', 0), ('p5', .05), ('p25', .25), ('중앙', .5)]))
        print(f"   만재 상태 평균 체류 {tl.full_h[m4].mean():.1f}h "
              f"(중앙값 {tl.full_h[m4].median():.1f}h, "
              f"최대 {tl.full_h[m4].max():.0f}h)  "
              f"만재 거래 평균 보유 {tl.hold_h[m4].mean():.1f}h")
        print("   최악 5건 (심볼·청산일·pnl/eq0·최저미실현·보유):")
        for _, r in tl.loc[m4].assign(ret=ret, mu=mu).nsmallest(5, 'ret').iterrows():
            print(f"     {r.sym:>4s} {r.ts.date()} {r.ret*100:+8.2f}% "
                  f"{r.mu*100:+8.2f}% {r.hold_h:6.0f}h {r.reason}")
    i = ret.idxmin()
    r = tl.loc[i]
    print(f"\n(d) 최대 단일 거래 손실: {ret[i]*100:+.2f}% (진입시 자본대비) — "
          f"{r.sym} {r.ts_in.date()}→{r.ts.date()} {int(r.k)}트랜치 "
          f"{r.hold_h:.0f}h {r.reason}  |  셀 $10,000 기준 "
          f"${ret[i]*10000:+,.0f} (진입시 자본이 $10,000 일 때)")


def selftest(data: dict, fund: pd.DataFrame, main, pure) -> None:
    """인과성·규칙 배선 자가검증 — 위반 시 AssertionError 로 실패한다."""
    print("\n--- selftest (인과성·규칙 배선) ---")
    for pth, want in SHA_EXPECT.items():                      # 0) 데이터 동결
        got = sha256(pth)
        assert got == want, (pth, got, want)
    print("  [OK] 데이터 3종 SHA256 = 사전등록 해시 (frozen MANIFEST 일치)")
    # 1) BB 는 모표준편차 (ddof=0) — 표본표준편차와 다르고 수동 계산과 일치
    c = data['BTC'].close
    sd0, sd1 = c.rolling(BB_N).std(ddof=0), c.rolling(BB_N).std(ddof=1)
    w = c.iloc[481:501].to_numpy()
    assert abs(sd0.iloc[500] - np.std(w)) < 1e-9 and abs(sd0.iloc[500] - sd1.iloc[500]) > 0
    print("  [OK] BB 표준편차 ddof=0 (모표준편차, 출판 관례)")
    dfr, tl, _ = main
    idx = None
    for s in SYMS:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    D = {s: data[s].reindex(idx.sort_values()) for s in SYMS}
    AS = {s: atr(D[s]).shift(1) for s in SYMS}
    # 2) 전 트랜치 체결가 = 체결봉 시가 (다음 봉 시가 체결 규약)
    nt = 0
    for _, r in tl.iterrows():
        for (e, _u), ts in zip(r.tr, r.ts_tr):
            assert abs(e - D[r.sym].open.at[ts]) < 1e-12, (r.sym, ts)
            nt += 1
        if r.reason == 'exit':
            assert abs(r.x - D[r.sym].open.at[r.ts]) < 1e-12
    print(f"  [OK] 트랜치 {nt}건 전부 체결가 = 다음 봉 시가 (청산 'exit' 동일)")
    # 3) 신호 배선: 진입 = 종가<하단밴드, 추매 = 종가<=직전체결가-1.0×ATR[i-1]
    #    (추매 임계는 직전 트랜치 체결봉 기준으로 재계산해 동결 저장값과 대조)
    na = 0
    for _, r in tl.iterrows():
        for j, ((kind, sc, thr), ts) in enumerate(zip(r.sig, r.ts_tr)):
            if j == 0:
                assert kind == 'entry' and sc < thr
            else:
                assert kind == 'add' and sc <= thr
                # 동결 규칙: 임계 = 직전 트랜치 체결봉의 ATR[i-1] 로 체결 시 고정
                prev_ts = r.ts_tr[j - 1]
                want = r.tr[j - 1][0] - ADD_ATR * AS[r.sym].at[prev_ts]
                assert abs(thr - want) < 1e-9, (r.sym, ts, thr, want)
                na += 1
        if r.reason == 'exit':
            sc, m1 = r.exit_sig
            assert sc >= m1
        assert 1 <= r.k <= N_TRANCHE and len(r.tr) == r.k
    print(f"  [OK] 진입/추매/익절 신호 배선 — 추매 {na}건 임계 = "
          f"직전 체결가 − 1.0×ATR[i−1] (raw 재계산 일치), 하드캡 {N_TRANCHE} 준수")
    # 4) 룩어헤드 위반 대조군 — 같은 봉 신호 평가 시 결과가 달라진다
    vio = run(data, fund, causal=False)
    assert abs(vio[0].equity.iloc[-1] - dfr.equity.iloc[-1]) > 1e-9
    print(f"  [OK] 확정봉 신호 교정이 결과를 지배 — 위반본(같은 봉 신호) 최종자본 "
          f"{vio[0].equity.iloc[-1]:.4f} vs 교정본 {dfr.equity.iloc[-1]:.4f}")
    # 5) 순수 BBMR 팔은 전 거래 1트랜치, 추매 0건
    assert int(pure[1].k.max()) == 1
    print("  [OK] 비교군(순수 BBMR) 전 거래 1트랜치 — 추매 경로 비활성")
    # 6) 시간 역행 없음
    assert bool((tl.hold_h >= 0).all()) and bool((pure[1].hold_h >= 0).all())
    print("  [OK] 전 거래 청산시각 >= 진입시각")


if __name__ == '__main__':
    print("데이터 SHA256:")
    for pth in PATHS:
        print(f"  {pth}  {sha256(pth)}")
    data, fund = load()
    for s in SYMS:
        df = data[s]
        print(f"  {s}: {df.index.min()} ~ {df.index.max()}  ({len(df)} 봉)")

    print("\n" + "=" * 78)
    print("BBADD U1 사전 기준선 — 바스켓 A (BTC/ETH/SOL), 라이브 셀 E24 대응")
    print("라벨: 볼린저 추매 변형 · 승률 제조 구조 시연 · 미검증 · 판정 권한 없음")
    print("사전 예측 (결과 조회 전 동결):")
    for p in PREDICTIONS:
        print(f"  {p}")
    print("=" * 78)

    main = run(data, fund)                                    # (a)~(d) 본안
    pure = run(data, fund, n_tranche=1)                       # (e) 비교군
    e15 = run(data, fund, n_tranche=1, tranche_frac=SLOT_FRAC)  # 참고 (E15형)
    selftest(data, fund, main, pure)

    S = {}
    S['main'] = report("(a) BBADD — 4트랜치 추매 (본안)", *main)
    tranche_report(main[1])
    print("\n" + "=" * 78)
    print("(e) 비교군 — 추매 없는 순수 BBMR, 동일 창")
    print("=" * 78)
    S['pure'] = report("(e1) 순수 BBMR — 1트랜치 (동일 트랜치 명목 eq×1/12)", *pure)
    S['e15'] = report("(e2) 참고: E15형 사이징 — 1트랜치 명목 eq×1/3 "
                      "(라이브 무스탑 셀 관례)", *e15)

    # 메커니즘 분리 공시 — 추매는 진입·청산 시점을 바꾸지 않아야 비교가 순수하다.
    km = set(zip(main[1].ts_in, main[1].sym))
    kp = set(zip(pure[1].ts_in, pure[1].sym))
    em = set(zip(main[1].ts, main[1].sym))
    print(f"\n거래 경계 동일성 (본안 vs 순수 BBMR): 청산 시점 "
          f"{'완전 동일' if em == set(zip(pure[1].ts, pure[1].sym)) else '상이'}, "
          f"진입 시점 불일치 {len(km ^ kp) // 2}건 / {len(km)}건 "
          f"(본안의 큰 노출이 일손실 정지를 먼저 격발해 진입이 지연된 경우)")

    nf = run(data, fund * 0.0)                                # 펀딩 영향 공시
    print(f"\n펀딩 영향 (본안): 펀딩 포함 누적 {S['main']['net']*100:+.1f}% vs "
          f"펀딩 제외 {(stats(*nf[:2])['net'])*100:+.1f}% — 차이가 펀딩 비용")

    print("\n" + "=" * 78)
    print("요약 비교표")
    print("=" * 78)
    print(f"{'팔':>28s} {'거래':>5s} {'승률':>7s} {'평균/거래':>9s} {'누적':>8s} "
          f"{'CAGR':>7s} {'MDD일':>7s} {'최대손실':>9s} {'보유':>7s}")
    for key, nm in (('main', 'BBADD 4트랜치 (본안)'),
                    ('pure', '순수 BBMR 1트랜치'),
                    ('e15', '참고 E15형 (명목 1/3)')):
        st = S[key]
        print(f"{nm:>28s} {st['n']:5d} {st['win']*100:6.1f}% {st['avg']*100:+8.3f}% "
              f"{st['net']*100:+7.1f}% {st['cagr']*100:+6.1f}% {st['mdd_d']*100:6.1f}% "
              f"{st['worst']*100:+8.2f}% {st['hold']:6.1f}h")

    print("\n" + "=" * 78)
    print("사전 예측 vs 실측 (공시)")
    print("=" * 78)
    m, p = S['main'], S['pure']
    print(f"  ① 승률 65~80% 예상        → 실측 {m['win']*100:.1f}% "
          f"(순수 BBMR {p['win']*100:.1f}%)")
    print(f"  ② 기대값 개선 없음 예상   → 평균/거래 본안 {m['avg']*100:+.3f}% vs "
          f"순수 {p['avg']*100:+.3f}%  |  누적 본안 {m['net']*100:+.1f}% vs "
          f"순수 {p['net']*100:+.1f}%  |  CAGR 본안 {m['cagr']*100:+.1f}% vs "
          f"순수 {p['cagr']*100:+.1f}%")
    print(f"  ③ 만재 꼬리 예상          → 최대 단일손실 {m['worst']*100:+.2f}%, "
          f"MDD(1h) {m['mdd_h']*100:.1f}% (위 (c) 분포 참조)")
    print("\n판정 없음 — 기준선 기록 전용. 커밋 금지.")
