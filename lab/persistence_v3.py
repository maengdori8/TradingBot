"""트레이더 지속성 v3 — 점-앵커 윈도우 (보간 제로).

v2(달력월+보간)가 유효표본 붕괴로 판정 불가 → 방법 수정. 지위: 탐색적
(사전등록 파이프라인이 데이터 부족을 반환한 뒤의 수정임을 명시).

핵심: 곡선의 '실제 관측점' 사이 차분은 정확한 손익이다(보간 없음).
- 전역 그리드: 28일 간격. 지갑별로 그리드 날짜에서 ±4일 내 실제점에 스냅.
- 윈도우 손익 = 인접 스냅점 손익 차분(정확), 30일 환산 정규화.
- PIT 적격(창 시작 계좌 ≥$10k, 실제점 값), 흐름 제외, 형성 |pnl|≥$100 동일.
- 지표·통과기준: v2 사전등록과 동일 적용.
"""
from __future__ import annotations
import gzip, json, sys
from datetime import datetime, timezone
import numpy as np

PILOT_N = 300
GRID_D = 28
SNAP_D = 4.0
MIN_ACCT = 10000.0
MAX_FLOW_FRAC = 0.50
MIN_ABS_PNL = 100.0
DAY = 86400 * 1000

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else 0.0

def snap(ts, t, tol_ms):
    """t에 가장 가까운 실제점 인덱스 (허용오차 내), 없으면 None."""
    i = int(np.searchsorted(ts, t))
    best, bd = None, tol_ms + 1
    for j in (i - 1, i):
        if 0 <= j < len(ts) and abs(ts[j] - t) < bd:
            best, bd = j, abs(ts[j] - t)
    return best

def load(path, skip_pilot=True):
    # 전역 그리드 (2024-01-01부터 28일 간격)
    g0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    grid = [g0 + k * GRID_D * DAY for k in range(40)]
    tol = SNAP_D * DAY
    windows = {}          # addr -> {k: (pnl_rate30, acct0)}
    n = 0
    with gzip.open(path, 'rt') as f:
        for idx, line in enumerate(f):
            if skip_pilot and idx < PILOT_N: continue
            try: r = json.loads(line)
            except Exception: continue
            cur = r.get('perpAllTime', {})
            pnl_c, acct_c = cur.get('pnl', []), cur.get('acct', [])
            if len(pnl_c) < 4 or len(acct_c) < 4: continue
            n += 1
            pts = np.array([float(q[0]) for q in pnl_c]); pvs = np.array([float(q[1]) for q in pnl_c])
            ats = np.array([float(q[0]) for q in acct_c]); avs = np.array([float(q[1]) for q in acct_c])
            rec = {}
            for k in range(len(grid) - 1):
                i0, i1 = snap(pts, grid[k], tol), snap(pts, grid[k+1], tol)
                if i0 is None or i1 is None or i1 <= i0: continue
                span_d = (pts[i1] - pts[i0]) / DAY
                if span_d < 20 or span_d > 36: continue
                a0i = snap(ats, pts[i0], tol); a1i = snap(ats, pts[i1], tol)
                if a0i is None or a1i is None: continue
                acct0, acct1 = avs[a0i], avs[a1i]
                if acct0 < MIN_ACCT: continue                     # PIT 적격
                pnl = pvs[i1] - pvs[i0]                           # 실제점 차분 = 정확
                if abs((acct1 - acct0) - pnl) > MAX_FLOW_FRAC * acct0: continue
                rec[k] = (pnl * 30.0 / span_d, acct0)
            if rec: windows[r['address']] = rec
    return windows, grid, n

def main(path='logs/trader_portfolio.jsonl.gz'):
    windows, grid, n = load(path)
    lab = lambda k: datetime.fromtimestamp(grid[k]/1000, tz=timezone.utc).strftime('%y-%m-%d')
    ks = sorted({k for rec in windows.values() for k in rec})
    print(f"확인 표본 {n}개 → 유효 지갑 {len(windows)}개 (28일 점-앵커 윈도우, 보간 없음)\n")
    print(f"{'형성창':>9s} {'n':>6s} {'IC':>8s} {'상위→중앙':>10s} {'전체→중앙':>10s} {'초과':>8s} {'상위양수':>7s} {'전체양수':>7s}")
    rows = []
    for k in ks:
        if k + 1 not in ks: continue
        xs, ys = [], []
        for rec in windows.values():
            if k in rec and k + 1 in rec and abs(rec[k][0]) * rec[k][1] >= 0:  # placeholder
                pnl30, acct0 = rec[k]
                if abs(pnl30) < MIN_ABS_PNL: continue
                xs.append(pnl30 / acct0); ys.append(rec[k+1][0] / rec[k+1][1])
        if len(xs) < 100: continue
        x, y = np.array(xs), np.array(ys)
        ic = spearman(x, y)
        kk = max(len(x)//10, 5)
        o = np.argsort(x)
        top_m = float(np.median(y[o[-kk:]])); all_m = float(np.median(y))
        tpos = float((y[o[-kk:]] > 0).mean()); apos = float((y > 0).mean())
        rows.append(dict(ic=ic, top=top_m, allm=all_m, excess=top_m-all_m, tpos=tpos, apos=apos))
        print(f"{lab(k):>9s} {len(x):6d} {ic:+8.3f} {top_m*100:+9.2f}% {all_m*100:+9.2f}% "
              f"{(top_m-all_m)*100:+7.2f}%p {tpos*100:6.0f}% {apos*100:6.0f}%")
    if len(rows) < 6:
        print(f"\n윈도우쌍 {len(rows)}개 — 부족"); return
    ics = np.array([r['ic'] for r in rows]); top = np.array([r['top'] for r in rows])
    ex = np.array([r['excess'] for r in rows])
    tpos = np.array([r['tpos'] for r in rows]); apos = np.array([r['apos'] for r in rows])
    rng = np.random.default_rng(11)
    perm = np.array([(ics * rng.choice([-1, 1], len(ics))).mean() for _ in range(20000)])
    pval = float((perm >= ics.mean()).mean())
    a = (top > 0).mean() >= 0.79
    b = np.median(top) >= 0.02
    c = np.median(tpos) >= 0.60 and np.median(tpos - apos) >= 0.10
    d = ics.mean() > 0 and pval < 0.05
    print(f"\n윈도우쌍 {len(rows)}개 (추론 단위 = 기간):")
    print(f"  IC 평균 {ics.mean():+.4f}  순열 p={pval:.4f}  IC>0 비율 {(ics>0).mean()*100:.0f}%")
    print(f"  상위십분위 전방(30일 환산): 중앙 {np.median(top)*100:+.2f}%  양수기간 {(top>0).mean()*100:.0f}%")
    print(f"  전체 대비 초과 중앙 {np.median(ex)*100:+.2f}%p")
    print(f"  상위 양수비율 {np.median(tpos)*100:.0f}% vs 전체 {np.median(apos)*100:.0f}%")
    print(f"\n기준(v2 사전등록 준용): (a)양수≥79%:{'충족' if a else '미달'} (b)중앙≥+2%:{'충족' if b else '미달'} "
          f"(c)양수율:{'충족' if c else '미달'} (d)p<0.05:{'충족' if d else '미달'}")
    print(f"  종합: {'통과' if all([a,b,c,d]) else '미달'}  [지위: 탐색적 — 방법이 데이터 부족 후 수정됨]")

if __name__ == '__main__':
    main(*sys.argv[1:])
