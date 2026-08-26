"""교차 거래소 차익 — 최종 무편향 분석 (Codex 13라운드 스펙).

- 유니버스: 사전 규칙(양쪽 상장 + 양쪽 일 $5M+) 37개 전부, 스프레드 선택 없음
- 비용 3단: 5bp(양다리 maker 낙관) / 12bp(1차: maker+taker 헤지) / 25bp(스트레스)
- 허들 = 2 × RT × 365/7 (비용별 스케일)
- 정적 벤치마크: 상시 Bybit롱/HL숏 (전환 없음) — 동적 전환의 부가가치 검증
- 강건성: 상위 5개 기여 제외, 코인 중앙값, ZEC 제외
- ROE 환산: 차익/2 (두 거래소 담보 이중 소요, 각 1x)
"""
from __future__ import annotations
import numpy as np, pandas as pd

def sim(d, rt, hurdle, static=False):
    if static:
        pos = pd.Series(-1.0, index=d.index)      # 상시 B롱/H숏 = (H-B) 수취 = -d 수취
        net = pos * d
        net.iloc[0] -= rt
        return net, pos
    trail = d.rolling(7).mean().shift(1) * 365
    cur, held, vals = 0, 99, []
    for t, tr in trail.items():
        held += 1
        if not np.isnan(tr):
            if cur == 0:
                if tr > hurdle: cur, held = 1, 0
                elif tr < -hurdle: cur, held = -1, 0
            elif held >= 7:
                if (cur == 1 and tr < 0) or (cur == -1 and tr > 0): cur = 0
        vals.append(cur)
    pos = pd.Series(vals, index=d.index, dtype=float)
    return pos * d - pos.diff().abs().fillna(0) * rt, pos

def main():
    scan = pd.read_csv('lab/data/xvenue_scan.csv')
    ALL = list(scan[scan.min_vol >= 5].coin)
    hl = pd.read_parquet('lab/data/xv_hl_deep.parquet')
    by = pd.read_parquet('lab/data/xv_by_deep.parquet')
    diffs = {}
    for c in ALL:
        if c not in hl.columns or c not in by.columns: continue
        h = hl[c].dropna().resample('D').sum(min_count=1)
        b = by[c].dropna().resample('D').sum(min_count=1)
        j = pd.concat({'b': b, 'h': h}, axis=1, sort=True).dropna()
        if len(j) >= 120:
            diffs[c] = j['b'] - j['h']
    print(f"유니버스 {len(ALL)}개 중 이력 충족 {len(diffs)}개 (공통 ≥120일)\n")

    for rt_bp, label in [(5, '낙관 5bp'), (12, '1차 12bp'), (25, '스트레스 25bp')]:
        rt = rt_bp / 1e4
        hurdle = 2 * rt * 365 / 7
        port = {c: sim(d, rt, hurdle)[0] for c, d in diffs.items()}
        P = pd.concat(port, axis=1, sort=True)
        eq = P.mean(axis=1).dropna()
        yr = eq.groupby(eq.index.year).sum() * 100
        r90 = eq.tail(90).sum() * 365 / 90 * 100
        per = pd.Series({c: s.sum() / (s.notna().sum() / 365.25) * 100 for c, s in port.items()})
        contrib = pd.Series({c: s.sum() for c, s in port.items()})
        drop5 = [c for c in contrib.nlargest(5).index]
        eq_d5 = P.drop(columns=drop5).mean(axis=1).dropna()
        eq_nz = P.drop(columns=['ZEC'], errors='ignore').mean(axis=1).dropna()
        print(f"── 왕복 {label} (허들 {hurdle*100:.1f}%/yr) ──")
        print(f"  연도별(차익 기준): " + "  ".join(f"{y}:{v:+.2f}%" for y, v in yr.items()))
        print(f"  최근90일 연환산 {r90:+.2f}%  |  ROE 환산(÷2): {r90/2:+.2f}%")
        print(f"  코인 중앙 순연율 {per.median():+.2f}%  양수 {int((per>0).sum())}/{len(per)}")
        print(f"  상위5 기여 제외 최근90일: {eq_d5.tail(90).sum()*365/90*100:+.2f}%  "
              f"ZEC 제외: {eq_nz.tail(90).sum()*365/90*100:+.2f}%")
        print()

    # 정적 벤치마크 (1차 비용)
    rt = 12 / 1e4
    stat = {c: sim(d, rt, 0, static=True)[0] for c, d in diffs.items()}
    S = pd.concat(stat, axis=1, sort=True).mean(axis=1).dropna()
    yr = S.groupby(S.index.year).sum() * 100
    print(f"── 정적 벤치마크: 상시 Bybit롱/HL숏, 전환 없음 (12bp 1회) ──")
    print(f"  연도별: " + "  ".join(f"{y}:{v:+.2f}%" for y, v in yr.items()))
    print(f"  최근90일 연환산 {S.tail(90).sum()*365/90*100:+.2f}%")

if __name__ == '__main__':
    main()
