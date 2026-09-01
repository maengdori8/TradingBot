# ICT Technical Execution Overlay

`ICT + MTF + 추세 + 모멘텀 + 거래량 + 매물대 + 변동성`을 내부에서 분석하고, 차트에는 실행에 필요한 정보만 표시하는 TradingView Pine Script v6 지표다.

> 검증 상태: **FAIL — 연구·페이퍼 전용, 실거래 금지**. 2026-08-31 기준 Bybit BTC·ETH·SOL 180일 검증에서 기본 9개 심볼×시간봉 중 양의 기대값은 3개, 마지막 25% 홀드아웃에서는 2개뿐이었다. 자세한 결과는 [실행형 검증 보고서](./TRADINGVIEW_EXECUTION_OVERLAY_VALIDATION.md)를 참고한다.

## 차트에 표시되는 것

- `롱 진입` 또는 `숏 진입`
- `추매1`~`추매4` 실행 시점
- 현재 평균 진입가
- 아직 실행되지 않은 추매1~추매4 대기 가격
- 추매 후에도 멀어지지 않는 `최후 손절`과 평균가에 따라 다시 계산되는 익절가
- 최대 손실 예산과 현재 수량
- 현재 보유시간과 24시간 제한
- 현재 내부 롱·숏 합의 점수
- 검증 상태

분석 원재료인 EMA, VWAP, Bollinger Band, FVG/OB, 공급·수요대, 매물대는 기본값에서 보이지 않는다. 설정에서 `(디버그)` 항목을 켜야만 표시된다.

## 내부 분석 엔진

진입 합의는 100점 만점이다.

| 범주 | 배점 | 내부 근거 |
|---|---:|---|
| 추세 | 25 | EMA 20/50, 세션 VWAP, Supertrend, MACD, 1·5·15분 MTF 정렬 |
| 모멘텀·강도 | 20 | RSI, Stochastic, STC, DMI/ADX |
| 거래량 | 15 | 상대 거래량, CMF, 캔들 방향 기반 CVD 프록시 |
| ICT | 25 | 유동성 스윕, BOS/CHoCH, FVG, Order Block, OTE, RSI 다이버전스 |
| 위치·변동성 | 15 | Bollinger 복귀, 거래량 가중 가격 중심, ATR 변동성 순위 |

기본 진입은 다음을 모두 요구한다.

- 전체 합의 68점 이상
- 반대 방향보다 12점 이상 우위
- ICT 내부 근거 3점 이상
- 차트 봉 마감 확정
- 1분 차트는 확정 5·15분 방향, 5분 차트는 확정 15분 방향, 15분 차트는 로컬 확정 방향과 정렬

합의 점수는 승률이나 수익 확률이 아니다.

## 분할진입과 위험 관리

- 기본 최대 위험: 기준 자산의 1%
- 위험예산 배분: 최초·추매1·추매2·추매3·추매4 각각 20%
- 추매 위치: 최초 진입에서 최후 손절까지 거리의 20% / 40% / 60% / 80% 불리한 방향
- 추매 조건: 앞 순번부터 해당 가격 터치 + 원래 방향의 재확인 캔들 + 내부 합의 유지
- 손절: 확정 스윙과 ATR을 사용하되 최소 1.2 ATR, 최대 기준 2.5 ATR
- 비용 필터: 손절폭이 가격의 0.25%보다 작지 않도록 제한
- 익절: 현재 평균 진입가 기준 기본 1.8R
- 시간 청산: 최초 진입 뒤 실제 24시간. 1분 1,440봉 / 5분 288봉 / 15분 96봉

추매가 실행돼도 최후 손절가는 더 멀어지지 않는다. 각 진입분은 총 위험예산의 20%씩 사용하며, 손절에 가까운 후속 진입일수록 같은 위험금액으로 더 많은 수량이 계산된다. 네 번 모두 체결된 뒤 최후 손절에 도달해도 이론상 손실이 설정한 최대 위험예산을 넘지 않도록 설계했다. 단, 갭·슬리피지·수수료는 실제 손실을 더 키울 수 있다.

## 사용 방법

1. [Pine 파일](../tradingview/ict_bb_mtf_confluence.pine)의 전체 내용을 TradingView Pine Editor에 넣는다.
2. 1분, 5분 또는 15분 차트에서 사용한다.
3. `기준 자산`, `거래당 최대 위험 %`, 수수료를 고려한 최소 손절폭을 실제 조건에 맞춘다.
4. 알림은 `봉 마감 시 한 번`으로 설정한다.
5. `롱 진입`, `숏 진입`, `추매1`~`추매4`, `익절`, `최후 손절`, `24h 청산` 알림을 필요한 것만 생성한다.
6. 현재 검증 결과가 FAIL이므로 실거래 주문과 연결하지 않는다.

## 매물대 한계

디버그용 고정구간 매물대는 OHLCV 각 봉의 거래량을 봉이 걸친 가격 행에 균등 배분한 근사값이다. POC·VAH·VAL 의미는 표준과 같지만, 하위시간봉을 재구성하는 TradingView 내장 Volume Profile과 동일하지 않다. 내부 진입 점수에는 이 미래에 표시되는 근사 POC를 사용하지 않고, 재도색 없는 롤링 거래량 가중 가격 중심만 사용한다.

## 파일

- 지표: [`tradingview/ict_bb_mtf_confluence.pine`](../tradingview/ict_bb_mtf_confluence.pine)
- 실행형 검증기: [`lab/validate_execution_overlay.py`](../lab/validate_execution_overlay.py)
- 원형 신호 검증기: [`lab/validate_ict_bb_mtf.py`](../lab/validate_ict_bb_mtf.py)
- 실행형 검증 보고서: [`docs/TRADINGVIEW_EXECUTION_OVERLAY_VALIDATION.md`](./TRADINGVIEW_EXECUTION_OVERLAY_VALIDATION.md)

## 참고 자료

- 영상: [클로드에게 맡기는 비트코인 AI자동매매 전략 (실전테스트)](https://www.youtube.com/watch?v=m5aGneOBkSo)
- ICT 자료 인덱스: [상위 10% 트레이더가 되기 위한 가장 현실적인 방법](https://fanding.kr/@easychart/post/176736/)
- TradingView: [Volume Profile 기본 개념](https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/)
- TradingView: [Repainting](https://www.tradingview.com/pine-script-docs/concepts/repainting/)
- TradingView: [Other timeframes and data](https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/)
