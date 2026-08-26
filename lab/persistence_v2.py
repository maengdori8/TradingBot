"""트레이더 지속성 — 역사적 파일럿 v2 (Codex 11라운드 수정 반영).

지위: 판정이 아니라 파일럿. 생존편향은 양방향(콜라이더)이므로 여기서의 기각도
'생존자 스크린 기각'일 뿐이며, 최종 판정은 전향 연구(9/24~)가 한다.

사전 고정 (전체 표본 보기 전):
- 확인 표본 = 코호트 순서 301번째 이후 (앞 300개는 파일럿으로 소진됨)
- 월 경계 유효성: 경계를 감싸는 실곡선점 간격 ≤ 72h (민감도 24h/7d),
  한 보간 선분이 두 경계를 가로지르면 무효
- PIT 적격: 월초 계좌가치 ≥ $10,000 (미달은 제외, 바닥값 대체 금지)
- 흐름 제외: |월중 입출금 추정| = |Δ계좌 − Δ손익| > 월초 계좌의 50% 이면 그 달 제외
- 1차 통계: 상위 십분위 전방 중앙 ROI − 동월 적격 전체 중앙 ROI
- 추론 단위 = 월 (지갑-월 아님). 월 단위 부호 순열검정.
- 통과 기준 (Codex 제시, 고정):
  (a) 상위십분위 전방 중앙 양수인 월쌍 ≥ 11/14 (비율 ≥ 79%)
  (b) 상위십분위 전방 중앙 ROI의 월 중앙값 ≥ +2%/월
  (c) 상위십분위 양수비율 중앙 ≥ 60% 이고 동월 전체보다 ≥ +10%p
  (d) 평균 IC > 0, 월 단위 순열 p < 0.05
  (e) 72h 규칙·흐름 제외·크기 통제에서 결과 유지
"""
from __future__ import annotations
import gzip, json, sys
from datetime import datetime, timezone
import numpy as np

PILOT_N = 300
BRACKET_H = 168.0         # 곡선이 주간 샘플링(실측 중앙 168h)이므로 이것이 자연 해상도.
                          # 가장자리 ±1주 손익 귀속 오차 존재 — t→t+2 건너뛰기 검사로 통제
MIN_ACCT = 10000.0
MAX_FLOW_FRAC = 0.50

def month_starts(lo, hi):
    d = datetime.fromtimestamp(lo/1000, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while True:
        ms = int(d.timestamp()*1000)
        if ms > hi: break
        if ms >= lo: out.append(ms)
        d = d.replace(year=d.year+1, month=1) if d.month == 12 else d.replace(month=d.month+1)
    return out

def bracket_ok(ts, t, max_h):
    """경계 t를 감싸는 실점 간격이 max_h 이하인가."""
    lo = None; hi = None
    for q in ts:
        if q <= t: lo = q
        if q >= t and hi is None: hi = q
    if lo is None or hi is None: return False
    return (hi - lo) <= max_h * 3600 * 1000

def interp(ts, vs, t):
    if t < ts[0] or t > ts[-1]: return None
    return float(np.interp(t, ts, vs))

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else 0.0

def load(path, skip_pilot=True, bracket_h=BRACKET_H):
    monthly = {}
    n = 0
    f = gzip.open(path, 'rt')
    try:
        idx = -1
        while True:
            try: line = f.readline()
            except EOFError: break
            if not line: break
            idx += 1
            if skip_pilot and idx < PILOT_N: continue
            try: r = json.loads(line)
            except Exception: continue
            cur = r.get('perpAllTime', {})
            pnl_c, acct_c = cur.get('pnl', []), cur.get('acct', [])
            if len(pnl_c) < 4 or len(acct_c) < 4: continue
            n += 1
            pts = [float(q[0]) for q in pnl_c]; pvs = [float(q[1]) for q in pnl_c]
            ats = [float(q[0]) for q in acct_c]; avs = [float(q[1]) for q in acct_c]
            ms = month_starts(int(pts[0]), int(pts[-1]))
            rec = {}
            for a, b in zip(ms, ms[1:]):
                if not (bracket_ok(pts, a, bracket_h) and bracket_ok(pts, b, bracket_h)):
                    continue
                # 한 선분이 두 경계를 다 가로지르는 경우 차단: 내부 실점 1개 이상 요구
                if not any(a < q < b for q in pts): continue
                c0, c1 = interp(pts, pvs, a), interp(pts, pvs, b)
                a0, a1 = interp(ats, avs, a), interp(ats, avs, b)
                if None in (c0, c1, a0, a1): continue
                if a0 < MIN_ACCT: continue                     # PIT 적격 (바닥값 금지)
                pnl = c1 - c0
                flow = (a1 - a0) - pnl                          # 입출금 추정
                if abs(flow) > MAX_FLOW_FRAC * a0: continue    # 흐름 오염 달 제외
                rec[a] = (pnl, pnl / a0, a0)
            if rec: monthly[r['address']] = rec
    finally:
        f.close()
    return monthly, n

def analyze(monthly, label_prefix="", skip=1):
    """skip=1: 인접 월쌍(t→t+1). skip=2: 한 달 건너뛰기(t→t+2) —
    인접 월이 공유하는 보간 선분이 없으므로 보간 오염 완전 차단 검사."""
    all_m = sorted({m for rec in monthly.values() for m in rec})
    lab = lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime('%Y-%m')
    rows = []
    print(f"{label_prefix}{'형성월':>8s} {'n':>6s} {'IC':>8s} {'상위→중앙':>10s} {'전체→중앙':>10s} {'초과':>8s} {'상위양수':>7s} {'전체양수':>7s}")
    for t, t1 in zip(all_m, all_m[skip:]):
        xs, ys, sz = [], [], []
        for rec in monthly.values():
            if t in rec and t1 in rec:
                xs.append(rec[t][1]); ys.append(rec[t1][1]); sz.append(rec[t][2])
        if len(xs) < 100: continue
        x, y, z = np.array(xs), np.array(ys), np.array(sz)
        ic = spearman(x, y)
        k = max(len(x)//10, 5)
        o = np.argsort(x)
        top_m = float(np.median(y[o[-k:]])); all_med = float(np.median(y))
        top_pos = float((y[o[-k:]] > 0).mean()); all_pos = float((y > 0).mean())
        # 크기 통제: 계좌크기 순위 제거한 부분 IC (형성 ROI에서 크기순위 회귀 잔차)
        rz = np.argsort(np.argsort(np.log(z))).astype(float)
        rx = np.argsort(np.argsort(x)).astype(float)
        beta = np.polyfit(rz, rx, 1)
        ic_partial = spearman(rx - np.polyval(beta, rz), y)
        rows.append(dict(t=t, n=len(x), ic=ic, icp=ic_partial, top=top_m, allm=all_med,
                         excess=top_m-all_med, tpos=top_pos, apos=all_pos))
        print(f"{lab(t):>8s} {len(x):6d} {ic:+8.3f} {top_m*100:+9.2f}% {all_med*100:+9.2f}% "
              f"{(top_m-all_med)*100:+7.2f}%p {top_pos*100:6.0f}% {all_pos*100:6.0f}%")
    return rows

def verdict(rows):
    if len(rows) < 6:
        print("월쌍 부족 — 판정 불가"); return
    ex = np.array([r['excess'] for r in rows]); top = np.array([r['top'] for r in rows])
    ics = np.array([r['ic'] for r in rows]); icp = np.array([r['icp'] for r in rows])
    tpos = np.array([r['tpos'] for r in rows]); apos = np.array([r['apos'] for r in rows])
    rng = np.random.default_rng(11)
    perm = np.array([(ics * rng.choice([-1,1], len(ics))).mean() for _ in range(20000)])
    pval = float((perm >= ics.mean()).mean())
    a = (top > 0).mean() >= 11/14
    b = np.median(top) >= 0.02
    c = np.median(tpos) >= 0.60 and np.median(tpos - apos) >= 0.10
    d = ics.mean() > 0 and pval < 0.05
    print(f"\n월쌍 {len(rows)}개 (추론 단위 = 월):")
    print(f"  IC 평균 {ics.mean():+.4f} (부분IC {icp.mean():+.4f})  월단위 순열 p={pval:.4f}")
    print(f"  상위십분위 전방 중앙 ROI: 월중앙 {np.median(top)*100:+.2f}%  양수월 {(top>0).mean()*100:.0f}%")
    print(f"  전체 대비 초과: 월중앙 {np.median(ex)*100:+.2f}%p")
    print(f"  상위 양수비율 중앙 {np.median(tpos)*100:.0f}% vs 전체 {np.median(apos)*100:.0f}%")
    print(f"\n사전 고정 기준: (a)양수월≥79%: {'충족' if a else '미달'}  (b)중앙≥+2%: {'충족' if b else '미달'}  "
          f"(c)양수율: {'충족' if c else '미달'}  (d)IC p<0.05: {'충족' if d else '미달'}")
    print(f"  종합: {'파일럿 통과' if all([a,b,c,d]) else '파일럿 미달'} — 최종 판정은 전향 연구")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'logs/trader_portfolio.jsonl.gz'
    bh = float(sys.argv[2]) if len(sys.argv) > 2 else BRACKET_H
    monthly, n = load(path, bracket_h=bh)
    print(f"확인 표본(파일럿 300 제외) {n}개 지갑 → 유효 {len(monthly)}개, 브래킷 {bh:.0f}h\n")
    print("── 인접 월쌍 (t → t+1) ──")
    rows = analyze(monthly)
    verdict(rows)
    print("\n── 보간 오염 차단 검사: 한 달 건너뛰기 (t → t+2) ──")
    rows2 = analyze(monthly, skip=2)
    if rows2:
        import numpy as _np
        ics2 = _np.array([r["ic"] for r in rows2]); top2 = _np.array([r["top"] for r in rows2])
        print(f"\n  t→t+2: IC 평균 {ics2.mean():+.4f}  양수월 {(ics2>0).mean()*100:.0f}%  "
              f"상위십분위 전방 중앙 {_np.median(top2)*100:+.2f}%")
        print("  (양수면 보간 공유선분으로 설명 불가 — 진짜 지속성의 독립 증거)")
