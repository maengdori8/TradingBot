"""BBADD v2 장기 표본 체크 — **탐색적 사후 체크 — 사전등록 아님, 승급 근거 사용 금지**.

성격 (동결 규약 준수 공시):
- 본 스크립트는 bbadd_test.py (동결) 결과 조회 **후** 작성된 사후(post-hoc) 변형
  탐색이다. 사전등록 문서 없음 → 어떤 결과도 셀 승급·자본 배정·판정 근거로 사용
  금지. 기록·비교 전용.
- 동결 파일 무수정: lab/bbadd_test.py 를 importlib 로 **읽기 전용** 로드해
  load/atr/sha256/stats/run 을 재사용한다. 기존 경로(설정 A = 필터·손절 전부
  비활성)가 동결 엔진 bb.run 과 **수치 동일**함을 selftest 가 강제한다
  (기존 경로 행동 동일성 증명 — 위반 시 AssertionError).

설정 3개 (그리드 확장 금지 — 이 3개 외 어떤 변형도 실행하지 않는다):
  (A) v1 기준선: BB(20, 2σ, ddof=0) 하단 종가 이탈 진입, 추매 사다리 최대 3회
      (트리거 = 직전 트랜치 체결가 − 1.0×ATR24[체결봉−1], 체결 시 동결, 재귀),
      SMA20 중심선 종가 익절, 필터·손절 없음 — bbadd_test 본안과 동일 경로.
      재현 검증 대상: 기록치 승률 77.4% · 누적 −49.2% 와 대략 일치해야 함.
  (B) v2 기본값: A + 진입 시 확정봉 종가 > SMA200(1h) 필터
      + 재해손절: 레벨 = 평단 − 6.0×ATR24[마지막 체결봉−1] (트랜치 체결 시 동결,
        추매마다 평단·ATR 재동결), 봉내 저가 <= 레벨 터치 시
        **min(시가, 손절가)** 체결 (갭 악화 모델 — max 아님).
  (C) A + SMA200 추세 필터만 (손절 없음).

실행 인과성 (bbadd_test 규약 계승):
- 신호 = 확정봉 종가·지표(shift 1), 체결 = 다음 봉 시가, 워밍업 100봉 무주문,
  형성중 봉 미사용. SMA200 미형성(NaN) 시 진입 무행동 (fail-closed).
- 손절 = 봉내 스탑주문 모델: 레벨은 체결 시점 동결값만 사용 (같은 봉 지표 미사용).
- 같은 봉 우선순위: 익절(시가 체결) > 추매(시가 체결) > 손절(봉내). 시가 체결이
  봉내 터치보다 시간상 선행하므로 인과 무결. 청산 봉 재진입 금지 (팜 규약 (i)).
- 비용 편도 8bp = 왕복 16bp, 일별 펀딩 선차감, 트랜치 명목 = equity×1/12
  (심볼 예산 = equity×1/3, 4트랜치), gross 10x · heat 6% · 일손실 −5% 진입정지
  — 전부 bbadd_test 자본 모델 동일.

산출: 설정×심볼별(단독 슬리브 런 — 해당 심볼만, 동일 상수·펀딩) 거래수·승률·
누적수익률·MDD + 3심볼 균등 합산(= bbadd_test 동일 3심볼 공유자본 포트폴리오,
슬롯 1/3 균등). 단독 런과 포트폴리오는 일손실 정지·gross/heat 상호작용 차이로
합이 정확히 일치하지 않는다 (공시).

실행: .venv/bin/python lab/bbadd_v2_check.py   (탐색 전용 — 판정 권한 없음)
"""
from __future__ import annotations
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                          # bbadd_test 의 상대경로 규약(lab/frozen/…) 계승

_SPEC = importlib.util.spec_from_file_location(
    'bbadd_test_frozen', os.path.join(ROOT, 'lab', 'bbadd_test.py'))
bb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bb)            # 동결 모듈 로드 (읽기 전용 — 수정 없음)

TREND_N = 200                           # 추세 필터: SMA200 (1h 종가)
STOP_MULT = 6.0                         # 재해손절: 평단 − 6×ATR24[마지막 체결봉−1]
CONFIGS = (
    ('A_v1_baseline', dict(trend_filter=False, stop_mult=None)),
    ('B_v2_sma200_stop6atr', dict(trend_filter=True, stop_mult=STOP_MULT)),
    ('C_sma200_only', dict(trend_filter=True, stop_mult=None)),
)


def run(data: dict, fund: pd.DataFrame, syms: tuple = bb.SYMS, *,
        trend_filter: bool = False, stop_mult: float | None = None,
        n_tranche: int = bb.N_TRANCHE, tranche_frac: float = bb.TRANCHE_FRAC,
        causal: bool = True):
    """BBADD v2 체크 실행 — 동결 bb.run 의 복제 + 2개 옵션만 추가.

    trend_filter=False 이고 stop_mult=None 이면 동결 엔진과 연산 순서까지 동일
    (selftest 1 이 수치 동일성을 강제한다).

    Args:
        data: sym -> OHLCV DataFrame (UTC DatetimeIndex).
        fund: 일별 펀딩률 합 (열 = 심볼) — 일 시작 선차감, 롱 지불.
        syms: 처리 심볼 순서 (단독 슬리브 런은 길이 1).
        trend_filter: True 면 신규 진입에 확정봉 종가 > SMA200[확정] 요구.
        stop_mult: None 이면 손절 없음. 값이 있으면 재해손절 활성
            (레벨 = 평단 − stop_mult×ATR24[마지막 체결봉−1], 체결 시 동결).
        n_tranche/tranche_frac: 동결 기본값 유지 (변경 금지 — 그리드 확장 금지).
        causal: False 면 같은 봉 신호 평가 (**룩어헤드 위반 대조군 전용**).

    Returns:
        (자본곡선 DataFrame, 트레이드 로그 DataFrame, 차단 카운터 dict).
    """
    idx = None
    for s in syms:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()
    D = {s: data[s].reindex(idx) for s in syms}
    sh = 1 if causal else 0
    C1, MID, LOWB, A, TREND = {}, {}, {}, {}, {}
    for s in syms:
        c = D[s].close
        m = c.rolling(bb.BB_N).mean()
        sd = c.rolling(bb.BB_N).std(ddof=0)       # 모표준편차 (출판 BB 관례)
        C1[s] = c.shift(sh)                       # 확정봉 종가 (신호)
        MID[s] = m.shift(sh)
        LOWB[s] = (m - bb.BB_K * sd).shift(sh)
        A[s] = bb.atr(D[s]).shift(sh)             # ATR[i-1]
        TREND[s] = c.rolling(TREND_N).mean().shift(sh)

    eq, pos, rows, tlog = 1.0, {}, [], []
    blocked = dict(gross=0, heat=0, halt_entry=0, halt_add=0, trend=0)
    day, day_eq, halted = None, 1.0, False

    def log(t_out, s, p, x, reason):
        """청산 1건 기록 — pnl 은 가격손익 + 전 트랜치 진입·청산 수수료."""
        fee = p['fees'] + p['u'] * x * bb.COST_OUT
        tlog.append(dict(
            ts=t_out, ts_in=p['t0'], sym=s, k=len(p['tr']), u=p['u'],
            e=p['basis'] / p['u'], x=x, eq0=p['eq0'], reason=reason,
            hold_h=(t_out - p['t0']).total_seconds() / 3600.0,
            fee=fee, pnl=p['u'] * x - p['basis'] - fee,
            min_unreal=p['min_unreal'],
            stop_lvl=p.get('stop', float('nan')),
            tr=tuple(p['tr']), ts_tr=tuple(p['ts_tr']), sig=tuple(p['sig']),
            exit_sig=p.get('exit_sig')))

    def close_all(t_out, s, p, x, reason):
        nonlocal eq
        eq += p['u'] * x - p['basis'] - p['u'] * x * bb.COST_OUT
        log(t_out, s, p, x, reason)
        pos.pop(s)

    def try_fill(t, i, s, o, fillmark, sig):
        """트랜치 1개 체결 시도 — 명목 사이징 + gross/heat 캡 (동결 엔진 동형)."""
        nonlocal eq
        if eq <= 0 or o <= 0 or np.isnan(o):
            return
        gross = 0.0
        for s2, p2 in pos.items():
            mk = fillmark.get(s2, D[s2].close.iloc[i - 1])
            if np.isnan(mk):
                mk = p2['basis'] / p2['u']        # 결측 마크 폴백 = 평단
            gross += p2['u'] * mk
        if gross >= bb.GROSS_CAP * eq:
            blocked['gross'] += 1
            return
        u = min(tranche_frac * eq / o, max(0.0, bb.GROSS_CAP * eq - gross) / o)
        heat = sum(e * uu * bb.HEAT_FRAC
                   for p2 in pos.values() for e, uu in p2['tr'])
        if u <= 0:
            return
        if heat + u * o * bb.HEAT_FRAC > bb.HEAT_CAP * eq * (1 + 1e-9):
            blocked['heat'] += 1
            return
        p = pos.get(s)
        if p is None:
            p = dict(tr=[], ts_tr=[], sig=[], basis=0.0, u=0.0, fees=0.0,
                     t0=t, eq0=eq, min_unreal=0.0, stop=float('nan'))
            pos[s] = p
        eq -= u * o * bb.COST_IN
        p['fees'] += u * o * bb.COST_IN
        p['tr'].append((o, u))
        a1f = A[s].iloc[i]                        # 체결봉 i 의 ATR[i-1] (shift 적용됨)
        p['thr'] = (o - bb.ADD_ATR * a1f) if (not np.isnan(a1f) and a1f > 0) \
            else np.nan
        p['ts_tr'].append(t)
        p['sig'].append(sig)
        p['basis'] += u * o
        p['u'] += u
        if stop_mult is not None and not np.isnan(a1f) and a1f > 0:
            # 재해손절 레벨 = 평단 − stop_mult×ATR24[이 체결봉−1] — 체결 시 동결,
            # 다음 체결까지 불변 (ATR NaN 이면 기존 레벨 유지 — fail-closed 보수).
            p['stop'] = p['basis'] / p['u'] - stop_mult * a1f
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
        fillmark = {}                             # 같은 봉 선행 체결 마크
        for s in syms:
            o, l, c = D[s].open.iloc[i], D[s].low.iloc[i], D[s].close.iloc[i]
            p = pos.get(s)
            if np.isnan(c):
                if p is not None:                 # 심볼 데이터 종료 — 강제청산
                    px = D[s].close.iloc[i - 1]
                    if not np.isnan(px):
                        close_all(idx[i - 1], s, p, px, 'eod')
                continue
            c1, m1, lb1 = C1[s].iloc[i], MID[s].iloc[i], LOWB[s].iloc[i]
            if p is not None:
                if not np.isnan(c1) and not np.isnan(m1) and c1 >= m1 \
                        and not np.isnan(o):      # 익절 (전량, 같은 봉 재진입 금지)
                    p['exit_sig'] = (c1, m1)
                    close_all(t, s, p, o, 'exit')
                elif (len(p['tr']) < n_tranche and not np.isnan(c1)
                      and not np.isnan(p.get('thr', np.nan))
                      and c1 <= p['thr']):        # 추매 (임계 = 체결 시 동결)
                    if halted:
                        blocked['halt_add'] += 1
                    else:
                        try_fill(t, i, s, o, fillmark, ('add', c1, p['thr']))
            elif not np.isnan(c1) and not np.isnan(lb1) and c1 < lb1:
                if trend_filter and (np.isnan(TREND[s].iloc[i])
                                     or c1 <= TREND[s].iloc[i]):
                    blocked['trend'] += 1         # 추세 필터 차단 (NaN = fail-closed)
                elif halted:                      # 신규 진입
                    blocked['halt_entry'] += 1
                else:
                    try_fill(t, i, s, o, fillmark, ('entry', c1, lb1))
            if stop_mult is not None:             # 재해손절 — 봉내 스탑주문 모델
                p = pos.get(s)                    # (추매·진입 직후 상태 반영)
                if (p is not None and not np.isnan(p.get('stop', np.nan))
                        and not np.isnan(l) and not np.isnan(o)
                        and l <= p['stop']):
                    # 갭 악화 체결: 시가가 레벨 아래로 갭 시 시가 체결 (min).
                    p['exit_sig'] = ('stop', p['stop'])
                    close_all(t, s, p, min(o, p['stop']), 'stop')
            p = pos.get(s)
            if p is not None and not np.isnan(l):  # 만재 꼬리 추적 (봉 저가 마크)
                p['min_unreal'] = min(p['min_unreal'], p['u'] * l - p['basis'])
        mtm = eq + sum(p['u'] * D[s].close.iloc[i] - p['basis']
                       for s, p in pos.items() if not np.isnan(D[s].close.iloc[i]))
        if not halted and day_eq > 0 and mtm / day_eq - 1 < bb.DAILY_HALT:
            halted = True                         # 진입정지만 — 청산 없음
        rows.append((t, mtm))
    for s, p in list(pos.items()):                # 기간 말 잔여 — 마지막 유효 종가
        px = D[s].close.dropna().iloc[-1]
        close_all(idx[-1], s, p, px, 'eod')
    if rows:
        rows[-1] = (rows[-1][0], eq)              # 최종점 = 청산 완료 자본
    dfr = pd.DataFrame(rows, columns=['ts', 'equity']).set_index('ts')
    return dfr, pd.DataFrame(tlog), blocked


def selftest(data: dict, fund: pd.DataFrame, res: dict) -> None:
    """동결 엔진 동일성 + 신규 로직 인과성 자가검증 — 위반 시 AssertionError."""
    print("\n--- selftest (동결 엔진 동일성 · 신규 로직 인과성) ---")
    for pth, want in bb.SHA_EXPECT.items():                   # 0) 데이터 동결
        got = bb.sha256(pth)
        assert got == want, (pth, got, want)
    print("  [OK] 데이터 3종 SHA256 = bbadd_test 사전등록 해시")
    # 1) 기존 경로 수치 동일성 — 설정 A(포트폴리오) == 동결 bb.run
    ref_dfr, ref_tl, _ = bb.run(data, fund)
    a_dfr, a_tl, _ = res[('A_v1_baseline', 'COMBINED')]
    assert len(ref_tl) == len(a_tl), (len(ref_tl), len(a_tl))
    assert float(np.max(np.abs(ref_tl.pnl.to_numpy()
                               - a_tl.pnl.to_numpy()))) < 1e-12
    assert abs(ref_dfr.equity.iloc[-1] - a_dfr.equity.iloc[-1]) < 1e-12
    print(f"  [OK] 설정 A == 동결 bbadd_test.run — 거래 {len(a_tl)}건, "
          f"최종자본 {a_dfr.equity.iloc[-1]:.6f} 수치 동일 (기존 경로 무변화 증명)")
    idx = None
    for s in bb.SYMS:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    D = {s: data[s].reindex(idx.sort_values()) for s in bb.SYMS}
    AS = {s: bb.atr(D[s]).shift(1) for s in bb.SYMS}
    SM = {s: D[s].close.rolling(TREND_N).mean().shift(1) for s in bb.SYMS}
    C1 = {s: D[s].close.shift(1) for s in bb.SYMS}
    b_tl = res[('B_v2_sma200_stop6atr', 'COMBINED')][1]
    c_tl = res[('C_sma200_only', 'COMBINED')][1]
    # 2) 전 트랜치 체결가 = 체결봉 시가 (다음 봉 시가 체결 규약)
    nt = 0
    for tl in (b_tl, c_tl):
        for _, r in tl.iterrows():
            for (e, _u), ts in zip(r.tr, r.ts_tr):
                assert abs(e - D[r.sym].open.at[ts]) < 1e-9, (r.sym, ts)
                nt += 1
            if r.reason == 'exit':
                assert abs(r.x - D[r.sym].open.at[r.ts]) < 1e-9
    print(f"  [OK] B·C 트랜치 {nt}건 전부 체결가 = 다음 봉 시가 (익절 동일)")
    # 3) 추세 필터 배선 — B·C 전 진입: 확정봉 종가 > SMA200[확정] (raw 재계산)
    for tl in (b_tl, c_tl):
        for _, r in tl.iterrows():
            ts0, sc = r.ts_tr[0], r.sig[0][1]
            assert abs(sc - C1[r.sym].at[ts0]) < 1e-9
            sm = SM[r.sym].at[ts0]
            assert not np.isnan(sm) and sc > sm, (r.sym, ts0, sc, sm)
    print("  [OK] B·C 전 진입 확정봉 종가 > SMA200[확정] — raw 재계산 일치")
    # 4) 재해손절 배선 (B) — 레벨 = 평단 − 6×ATR24[마지막 체결봉−1] 동결,
    #    트리거 = 저가 <= 레벨, 체결 = min(시가, 손절가) (갭 악화)
    ns = 0
    for _, r in b_tl[b_tl.reason == 'stop'].iterrows():
        avg = sum(e * u for e, u in r.tr) / sum(u for _e, u in r.tr)
        want = avg - STOP_MULT * AS[r.sym].at[r.ts_tr[-1]]
        assert abs(r.stop_lvl - want) < 1e-6 * max(1.0, abs(want)), (r.sym, r.ts)
        o_x, l_x = D[r.sym].open.at[r.ts], D[r.sym].low.at[r.ts]
        assert l_x <= r.stop_lvl + 1e-9, (r.sym, r.ts)
        assert abs(r.x - min(o_x, r.stop_lvl)) < 1e-9, (r.sym, r.ts)
        ns += 1
    assert ns > 0, "스탑 청산 0건 — 배선 검증 불능"
    print(f"  [OK] 재해손절 {ns}건 — 레벨 raw 재계산 일치, 체결 = min(시가, 손절가)")
    # 5) 룩어헤드 위반 대조군 — 같은 봉 신호 평가 시 결과가 달라진다 (신규 경로 B)
    vio = run(data, fund, trend_filter=True, stop_mult=STOP_MULT, causal=False)
    b_eq = res[('B_v2_sma200_stop6atr', 'COMBINED')][0].equity.iloc[-1]
    assert abs(vio[0].equity.iloc[-1] - b_eq) > 1e-9
    print(f"  [OK] 위반본(같은 봉 신호) 최종자본 {vio[0].equity.iloc[-1]:.4f} vs "
          f"교정본 {b_eq:.4f} — 상이 (확정봉 교정이 결과 지배)")
    # 6) 시간 역행 없음 (전 런)
    for r_i in res.values():
        tl = r_i[1]
        if len(tl):
            assert bool((tl.hold_h >= 0).all())
    print("  [OK] 전 런 전 거래 청산시각 >= 진입시각")


def summarize(res: dict) -> list[dict]:
    """설정×심볼 요약 행 생성 — bb.stats (초기자본 1.0 앵커) 재사용."""
    out = []
    for name, _kw in CONFIGS:
        for sym in ('COMBINED',) + tuple(bb.SYMS):
            dfr, tl, blocked = res[(name, sym)]
            st = bb.stats(dfr, tl)
            rc = tl.reason.value_counts().to_dict() if len(tl) else {}
            out.append(dict(
                config=name, sym=sym, n=st['n'], win=st['win'], avg=st['avg'],
                net=st['net'], mdd_h=st['mdd_h'], mdd_d=st['mdd_d'],
                n_stop=int(rc.get('stop', 0)), n_eod=int(rc.get('eod', 0)),
                blocked=blocked))
    return out


if __name__ == '__main__':
    print("탐색적 사후 체크 — 사전등록 아님, 승급 근거 사용 금지")
    print("데이터 SHA256 (동결 확인):")
    for pth in bb.PATHS:
        print(f"  {pth}  {bb.sha256(pth)}")
    data, fund = bb.load()
    for s in bb.SYMS:
        df = data[s]
        print(f"  {s}: {df.index.min()} ~ {df.index.max()}  ({len(df)} 봉)")

    res = {}
    for name, kw in CONFIGS:
        res[(name, 'COMBINED')] = run(data, fund, **kw)       # 3심볼 균등 합산
        for s in bb.SYMS:                                     # 단독 슬리브 런
            res[(name, s)] = run({s: data[s]}, fund, syms=(s,), **kw)

    selftest(data, fund, res)

    rows = summarize(res)
    print("\n" + "=" * 96)
    print("설정 × 심볼 요약 — COMBINED = 3심볼 공유자본 포트폴리오 (bbadd_test "
          "동일 자본 모델), 심볼 행 = 단독 슬리브 런")
    print("승률·평균/거래 = 가격손익+수수료 (펀딩은 자본 차감, 거래 미배분) — "
          "bbadd_test 측정 규약 동일")
    print("=" * 96)
    print(f"{'설정':>22s} {'심볼':>9s} {'거래':>5s} {'승률':>7s} {'평균/거래':>9s} "
          f"{'누적':>8s} {'MDD(1h)':>8s} {'MDD(일)':>8s} {'stop':>5s} {'eod':>4s}")
    for r in rows:
        print(f"{r['config']:>22s} {r['sym']:>9s} {r['n']:5d} {r['win']*100:6.1f}% "
              f"{r['avg']*100:+8.3f}% {r['net']*100:+7.1f}% {r['mdd_h']*100:7.1f}% "
              f"{r['mdd_d']*100:7.1f}% {r['n_stop']:5d} {r['n_eod']:4d}")
        if r['sym'] == 'SOL':
            b = r['blocked']
            print(f"{'':>22s} {'(차단':>9s} gross {b['gross']} heat {b['heat']} "
                  f"halt {b['halt_entry']}/{b['halt_add']} trend {b['trend']})")

    a = next(r for r in rows if r['config'] == 'A_v1_baseline'
             and r['sym'] == 'COMBINED')
    print(f"\n재현 검증 (설정 A COMBINED vs bbadd_test 기록치): "
          f"승률 {a['win']*100:.1f}% (기록 77.4%), 누적 {a['net']*100:+.1f}% "
          f"(기록 -49.2%) — selftest 1 이 동결 엔진과 수치 동일성을 증명")
    print("\n판정 없음 — 탐색적 사후 체크. 사전등록 아님. 승급 근거 사용 금지.")
