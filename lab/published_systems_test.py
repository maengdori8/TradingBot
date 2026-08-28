"""출판 시스템 2종 U1 사전 기준선 — 출판 파라미터 고정, 튜닝·조합 탐색 없음.

정확한 지위: '원전 문자 그대로'가 아니다 — [A]는 원전에 시스템 규칙이 없어 관례적
운용화이고, [B]는 체결 규약(다음 봉 시가)이 원전 문구(신호봉 종가)에 우선하며,
둘 다 원전 시장(미 주식/ETF 일봉)이 아닌 크립토 무기한 선물에 적용된다.

판정 규칙 (본 파일 작성 시점 명문화 — 결과 관찰 후 기록이므로 엄밀한 의미의
사전등록은 아님, 감사 기록용): U1 편입 요건 = BTC·ETH 모두에서 왕복 16bp 차감 후
거래당 순 기대값 양수. 하나라도 음수면 해당 층(L1/L2)에서 탈락. 부호 게이트 통과가
곧 배치 근거는 아님 — 통계적 유의성·표본 수는 별도 평가.

[A] 볼린저밴드 평균회귀 — 정확한 명명: "Bollinger 출판 지표 파라미터를 쓴 교과서 표준
    평균회귀 운용화". John Bollinger("Bollinger on Bollinger Bands", 2001)는 BB(20, 2σ)
    기본값(중심선 SMA20, ±2×모표준편차 ddof=0)을 출판했으나 "밴드 태그 자체는 신호가
    아니다"라고 명시 — 단순 평균회귀 '시스템'은 원전에 없다. 따라서 규칙은 계량 문헌의
    관례적 정의를 채택: 확정봉 종가 < 하단 밴드(종가 기준, 봉내 터치 아님) → 롱,
    청산 = 확정봉 종가 >= 중심선(SMA20). 상단 밴드 청산은 더 공격적 해석이라 배제
    (단, 중심선 청산 = 회전율 증가이므로 비용 후 '보수성'은 방향이 자명하지 않음을 기록).
    SMA200 추세 필터: 현대 계량 검증 관례에는 흔하나 원전에 없음 → 미적용(필터 발명 금지).
    숏 없음(롱 온리, 과제 정의 "하단 밴드 이탈 매수 계열"). 스탑 없음(원전에 없음).

[B] Connors/Alvarez 2-period RSI (원전: "Short Term Trading Strategies That Work", 2008.
    후속 합성 지표 'ConnorsRSI'와 혼동 금지)
    - RSI(2)는 Wilder 평활. 임계값은 원전 수치 5/95 (10/90은 흔한 완화 변형).
    - 롱: 종가 > SMA200 AND RSI(2) < 5 → 진입, 청산 = 종가 > SMA(5).
    - 숏(원전 테이블 수록): 종가 < SMA200 AND RSI(2) > 95 → 진입, 청산 = 종가 < SMA(5).
    - 스탑 없음(원전에 없음), 스케일인(TPS 등) 미사용 — 기본 단일 유닛 규칙만.
    - 원전 체결은 "신호봉 종가(buy on close)" — 여기서는 U1 실행 교정 프로토콜(다음 봉
      시가)로 의도적 대체. 원전 문자 그대로가 아닌 실행 규약 우선임을 명시.

2층 사전등록 설계 (Codex 합의):
    L1 일봉 대조군: 원전 파라미터·일봉, U1 실행 규약 하 — lab/frozen/perp_1d.parquet.
    L2 1h 이식: 구조(수식·봉 개수) 불변, 봉만 1h — 팜 규격. 단 경제적 의미는 변한다
       (SMA200: 200일→8.3일, 세션/오버나이트 소멸, 펀딩 8h 주기 노출). L2 부정 결과는
       "1h 이식 부적격"이지 원전 일봉 시스템의 반증이 아님을 명시.

실행 교정 (U1 관례 = rsi_divergence_test.py 계열):
    - 신호는 확정봉[i-1] 종가·지표만 사용, 체결은 다음 봉[i] 시가(이상화 MOO). 형성중 봉
      미사용 — 동결 파일 끝의 미완성 봉은 MANIFEST _frozen_at 기준으로 로드 시 제거.
    - 워밍업(지표 NaN) 중 주문 없음. 비용 왕복 16bp. 펀딩 미반영(가격-온리 기준선) —
      보유시간이 긴 시스템에 유리/불리 편향 가능성을 한계로 기록.
    - 리스크 사이징 없음: 거래당 %수익 산술 합산(고정 명목, 비복리) — 자본곡선 아님.
    - 데이터 끝에 열린 포지션은 마지막 확정 종가로 강제 청산(eod 표기, 건수 보고).

실행: .venv/bin/python lab/published_systems_test.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

RT_COST = 0.0016          # 왕복 16bp (taker+슬립)
FROZEN = 'lab/frozen'
TF_SEC = {'1h': 3600, '1d': 86400}


def wilder_rsi(c: pd.Series, n: int) -> pd.Series:
    """Wilder 평활 RSI. dn==0 구간은 정의값(상승만: 100, 무변동: 50) 명시 처리."""
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.where(dn != 0))
    rsi = rsi.mask((dn == 0) & (up > 0), 100.0)
    rsi = rsi.mask((dn == 0) & (up == 0), 50.0)
    return rsi


def load(sym: str, tf: str = '1h') -> pd.DataFrame:
    """동결 OHLC 로드. _frozen_at 시점에 미완성이던 마지막 봉은 제거 (형성중 봉 금지)."""
    with open(f'{FROZEN}/MANIFEST.json') as f:
        frozen_at = pd.Timestamp(json.load(f)['_frozen_at'])
    p = pd.read_parquet(f'{FROZEN}/perp_{tf}.parquet').xs(sym, level='sym')
    bar_end = p.index + pd.Timedelta(seconds=TF_SEC[tf])
    return p.loc[bar_end <= frozen_at, ['open', 'high', 'low', 'close']].astype(float)


def run_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """[A] BB(20,2σ) 평균회귀 롱 온리. 진입: 종가 < 하단밴드, 청산: 종가 >= SMA20."""
    c, o = df.close, df.open
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower = mid - 2 * sd
    trades, pos = [], None            # pos = (entry_px, entry_step)
    for i in range(1, len(df)):
        j = i - 1                     # 확정봉
        if np.isnan(lower.iloc[j]):
            continue
        if pos is not None:
            if c.iloc[j] >= mid.iloc[j]:
                e, ei = pos
                x = o.iloc[i]
                trades.append(dict(ts=df.index[i], d=1, ret=(x - e) / e - RT_COST,
                                   bars=i - ei, eod=False))
                pos = None
            continue                  # 청산 스텝 재진입 금지 — 결정적 1행동/봉 규약
        if c.iloc[j] < lower.iloc[j]:
            pos = (o.iloc[i], i)
    if pos is not None:
        e, ei = pos
        trades.append(dict(ts=df.index[-1], d=1, ret=(c.iloc[-1] - e) / e - RT_COST,
                           bars=len(df) - ei, eod=True))
    return pd.DataFrame(trades)


def run_connors(df: pd.DataFrame) -> pd.DataFrame:
    """[B] Connors/Alvarez RSI(2) 5/95, SMA200 레짐 필터, 청산 SMA5 교차. 롱+숏."""
    c, o = df.close, df.open
    r2 = wilder_rsi(c, 2)
    sma200 = c.rolling(200).mean()
    sma5 = c.rolling(5).mean()
    trades, pos = [], None            # pos = (dir, entry_px, entry_step)
    for i in range(1, len(df)):
        j = i - 1
        if np.isnan(sma200.iloc[j]) or np.isnan(r2.iloc[j]):
            continue
        if pos is not None:
            d, e, ei = pos
            if (d > 0 and c.iloc[j] > sma5.iloc[j]) or (d < 0 and c.iloc[j] < sma5.iloc[j]):
                x = o.iloc[i]
                trades.append(dict(ts=df.index[i], d=d, ret=d * (x - e) / e - RT_COST,
                                   bars=i - ei, eod=False))
                pos = None
            continue                  # 청산 스텝 재진입/역전 금지 — 결정적 1행동/봉 규약
        if c.iloc[j] > sma200.iloc[j] and r2.iloc[j] < 5:
            pos = (1, o.iloc[i], i)
        elif c.iloc[j] < sma200.iloc[j] and r2.iloc[j] > 95:
            pos = (-1, o.iloc[i], i)
    if pos is not None:
        d, e, ei = pos
        trades.append(dict(ts=df.index[-1], d=d, ret=d * (c.iloc[-1] - e) / e - RT_COST,
                           bars=len(df) - ei, eod=True))
    return pd.DataFrame(trades)


def report(name: str, t: pd.DataFrame, n_bars: int, span_yrs: float) -> None:
    if not len(t):
        print(f"  {name}: 신호 없음")
        return
    yrs = span_yrs                    # 관측 구간 기준 (첫~끝 거래 아님)
    wr = (t.ret > 0).mean()
    gross = t.ret.mean() + RT_COST
    se = t.ret.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else float('nan')
    expo = t.bars.sum() / n_bars
    print(f"  {name}: 거래 {len(t)}건 ({len(t) / max(yrs, 1e-9):.0f}건/년)  "
          f"승률 {wr * 100:.1f}%  총이익(비용전) {gross * 100:+.3f}%/건  "
          f"순 {t.ret.mean() * 100:+.3f}%/건 (SE {se * 100:.3f}%)")
    print(f"    산술누적(고정명목·비복리) {t.ret.sum() * 100:+.1f}%  "
          f"평균보유 {t.bars.mean():.1f}봉  노출 {expo * 100:.0f}%"
          + (f"  [미청산 강제청산 {int(t.eod.sum())}건]" if t.eod.any() else ""))
    lo, sh = t[t.d > 0], t[t.d < 0]
    if len(sh):
        print(f"    롱 {len(lo)}건 승률 {(lo.ret > 0).mean() * 100:.0f}% 누적 {lo.ret.sum() * 100:+.1f}%  "
              f"숏 {len(sh)}건 승률 {(sh.ret > 0).mean() * 100:.0f}% 누적 {sh.ret.sum() * 100:+.1f}%")
    yr = t.groupby(t.ts.dt.year).ret.sum() * 100
    print("    연도별 누적: " + "  ".join(f"{y}:{v:+.1f}%" for y, v in yr.items()))


def causality_check() -> None:
    """미래 절단 재실행 시 과거 확정 거래가 완전 동일해야 함 (룩어헤드 회귀 가드).

    엄격 비교: 절단 시점 이전 거래 프레임 전체 equals — min-길이 비교 금지(위음성 차단).
    가드일 뿐 증명은 아님 — 인과성의 근거는 j=i-1 신호·open[i] 체결 구조 자체.
    """
    cuts = {'1h': (240, 720, 2000), '1d': (60, 180, 500)}
    for sym in ('BTC', 'ETH'):
      for tf in ('1h', '1d'):
        df = load(sym, tf)
        for runner in (run_bollinger, run_connors):
            full = runner(df)
            for cut in cuts[tf]:
                part = runner(df.iloc[:-cut])
                cut_ts = df.index[-cut - 1]
                f = full[(~full.eod) & (full.ts <= cut_ts)].reset_index(drop=True)
                p = part[~part.eod].reset_index(drop=True)
                # 절단 직전에 진행중이던 포지션은 full 쪽에서 나중에 닫힘 → p가 f의
                # 접두사여야 하고, 길이 차이는 최대 0건(둘 다 같은 확정 거래 집합).
                assert len(p) == len(f), \
                    f"causality FAIL(count): {sym} {tf} {runner.__name__} cut={cut} {len(p)}!={len(f)}"
                assert p.equals(f), \
                    f"causality FAIL(content): {sym} {tf} {runner.__name__} cut={cut}"
    print("causality_check OK (BTC/ETH × 1h/1d × 2시스템 × 절단 3종: 과거 확정 거래 불변)")


if __name__ == '__main__':
    causality_check()
    for tf, label in (('1d', 'L1 일봉 대조군(원전 파라미터·U1 실행 규약)'),
                      ('1h', 'L2 1h 이식(구조 불변·봉만 1h)')):
        print(f"\n===== {label} =====")
        for sym in ('BTC', 'ETH'):
            df = load(sym, tf)
            span = (df.index[-1] - df.index[0]).days / 365.25
            print(f"{sym} ({df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}봉)")
            report('BB(20,2σ) 평균회귀(롱)   ', run_bollinger(df), len(df), span)
            report('Connors/Alvarez RSI(2) 5/95', run_connors(df), len(df), span)
