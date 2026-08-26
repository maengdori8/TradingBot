"""트레이더 실력 지속성 — 역사적 검증 (수집된 portfolio 곡선 사용).

사전 규정 (결과 보기 전 고정):
- 곡선: perpAllTime (perp 트레이딩 손익만; allTime은 이체/볼트 왜곡 사례 확인됨)
- 월 경계: UTC 달력월. 누적곡선을 경계에서 선형보간, 월손익 = 차분.
  곡선이 그 달 전체를 스팬해야 유효(외삽 금지).
- 형성월 필터: |월손익| ≥ $100 (그 달에 실제로 거래한 지갑만 — 사전 측정 가능)
- ROI = 월손익 / max(월초 계좌가치, $5,000)   (제로분모 폭발 차단)
- 곡선 해상도 통제: 월 내부에 실제 곡선점 ≥ MIN_INTERIOR개 요구. 성긴 곡선은
  인접 월이 같은 보간 선분을 공유해 IC가 기계적으로 부풀 수 있다(핵심 교란).
- 지표: 인접 월쌍별 Spearman IC(형성월 순위 → 다음달 ROI), 십분위 전방 중앙값,
  상위−하위 십분위 스프레드, **상위 십분위 전방 양수 비율**(카피 가능성의 핵심 —
  '패자 지속'만으로도 IC는 높게 나온다). 집계: 월쌍 평균 + 부트스트랩.
- 해석 규칙(단측): 이 표본은 '오늘 생존·활동' 지갑이라 실력 쪽으로 유리한 편향.
  → 여기서도 지속성 없음 = 기각 확정. 있음 = 전향 연구로 무편향 확인 필요.
"""
from __future__ import annotations
import gzip, json, sys
from datetime import datetime, timezone
import numpy as np

MIN_ABS_PNL = 100.0
MIN_INTERIOR = 1     # 월 내부 최소 실곡선점 (민감도: 1/2/3)
ACCT_FLOOR = 5000.0

def month_starts(lo_ms: int, hi_ms: int) -> list[int]:
    d = datetime.fromtimestamp(lo_ms / 1000, tz=timezone.utc)
    d = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while True:
        ms = int(d.timestamp() * 1000)
        if ms > hi_ms: break
        if ms >= lo_ms: out.append(ms)
        d = (d.replace(year=d.year + 1, month=1) if d.month == 12
             else d.replace(month=d.month + 1))
    return out

def interp(curve: list, t: float) -> float | None:
    """누적곡선 선형보간. 범위 밖이면 None (외삽 금지)."""
    if not curve: return None
    ts = [float(p[0]) for p in curve]; vs = [float(p[1]) for p in curve]
    if t < ts[0] or t > ts[-1]: return None
    return float(np.interp(t, ts, vs))

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0

def main(path='logs/trader_portfolio.jsonl.gz'):
    monthly = {}          # addr -> {month_ms: (pnl, roi)}
    n_read = 0
    f = gzip.open(path, 'rt')
    try:
        while True:
            try:
                line = f.readline()
            except EOFError:
                break                      # 수집 진행 중 부분 파일 허용
            if not line:
                break
            try: r = json.loads(line)
            except Exception: continue
            n_read += 1
            cur = r.get('perpAllTime', {})
            pnl_c, acct_c = cur.get('pnl', []), cur.get('acct', [])
            if len(pnl_c) < 3: continue
            lo, hi = float(pnl_c[0][0]), float(pnl_c[-1][0])
            ms = month_starts(int(lo), int(hi))
            ts_pts = [float(q[0]) for q in pnl_c]
            rec = {}
            for a, b in zip(ms, ms[1:]):
                c0, c1 = interp(pnl_c, a), interp(pnl_c, b)
                if c0 is None or c1 is None: continue
                interior = sum(1 for q in ts_pts if a < q < b)
                if interior < MIN_INTERIOR: continue      # 보간 상관 교란 차단
                pnl = c1 - c0
                av = interp(acct_c, a)
                roi = pnl / max(av if av is not None else 0.0, ACCT_FLOOR)
                rec[a] = (pnl, roi)
            if rec: monthly[r['address']] = rec
    finally:
        f.close()

    all_months = sorted({m for rec in monthly.values() for m in rec})
    label = lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime('%Y-%m')
    print(f"지갑 {n_read}개 읽음 → 월손익 산출 가능 {len(monthly)}개, 월 {len(all_months)}개")
    print(f"\n{'형성월':>8s} {'n':>6s} {'IC':>8s} {'상위10%→':>10s} {'하위10%→':>10s} {'스프레드':>9s} {'상위양수':>7s}")
    ics, spreads, rows, top_pos_all = [], [], [], []
    for t, t1 in zip(all_months, all_months[1:]):
        xs, ys = [], []
        for rec in monthly.values():
            if t in rec and t1 in rec and abs(rec[t][0]) >= MIN_ABS_PNL:
                xs.append(rec[t][1]); ys.append(rec[t1][1])
        if len(xs) < 100: continue
        x, y = np.array(xs), np.array(ys)
        ic = spearman(x, y)
        k = max(len(x) // 10, 5)
        order = np.argsort(x)
        top_f = float(np.median(y[order[-k:]])); bot_f = float(np.median(y[order[:k]]))
        top_pos = float((y[order[-k:]] > 0).mean())
        ics.append(ic); spreads.append(top_f - bot_f); top_pos_all.append(top_pos)
        rows.append((label(t), len(x), ic, top_f, bot_f))
        print(f"{label(t):>8s} {len(x):6d} {ic:+8.3f} {top_f*100:+9.2f}% {bot_f*100:+9.2f}% {(top_f-bot_f)*100:+8.2f}%p {top_pos*100:5.0f}%")

    if not ics:
        print("월쌍 표본 부족"); return
    ics_a, sp_a = np.array(ics), np.array(spreads)
    rng = np.random.default_rng(11)
    boot = np.array([rng.choice(ics_a, len(ics_a)).mean() for _ in range(10000)])
    lo95 = float(np.quantile(boot, 0.05))
    print(f"\n{'='*70}")
    print(f"월쌍 {len(ics_a)}개 집계:")
    print(f"  평균 IC {ics_a.mean():+.4f}  (부트스트랩 95% 단측하한 {lo95:+.4f})")
    print(f"  IC>0 비율 {(ics_a>0).mean()*100:.0f}%")
    print(f"  상위−하위 스프레드: 평균 {sp_a.mean()*100:+.2f}%p  중앙 {np.median(sp_a)*100:+.2f}%p  양수 {(sp_a>0).mean()*100:.0f}%")
    tp = np.array(top_pos_all)
    print(f"  [카피 관점] 상위 십분위의 다음달 양수 비율: 평균 {tp.mean()*100:.0f}%  (50% = 동전던지기)")
    print(f"  [카피 관점] 상위 십분위 다음달 중앙 ROI: {np.median([r[3] for r in rows])*100:+.2f}%")
    print(f"\n판정 참고 (사전등록 T+30 기준을 역사적 표본에 적용):")
    print(f"  IC ≥ +0.05: {'충족' if ics_a.mean() >= 0.05 and lo95 > 0 else '미달'}")
    print(f"  스프레드 ≥ +3%p/월: {'충족' if np.median(sp_a) >= 0.03 else '미달'}")
    print(f"  ※ 이 표본은 생존편향이 '실력 존재' 쪽으로 유리 — 미달이면 기각 확정 방향")

if __name__ == '__main__':
    main(*sys.argv[1:])
