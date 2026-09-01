# Precision V6 정확도 우선 WATCH 검증 계약

## 결론

**FAIL — 연구·전향 페이퍼 관찰 전용. 실거래 진입 신호가 아니다.**

V6는 사용자가 요청한 1·5·15분 표시, 1시간 확정 신호, ICT·Bollinger·RSI·DMI·
거래량·매물대 proxy 문맥, 최대 네 번의 추매 계획, 1R 익절과 최후 7ATR 손절선을
구현했다. 그러나 정확도를 가장 먼저 보는 사전 기준을 통과하지 못했다. 따라서 Pine은
`ENTRY`가 아니라 `WATCH`만 표시하고 모든 alert에 `execution_authority=false`,
`actionable=false`, `add_plan_validated=false`를 넣는다.

고정된 원시 기회집합 858건에서 네 A+ 조건을 모두 만족한 것은 116건이었다. 실제
24시간·1R·분할 실행 재생의 승률은 엄격 비용에서 `55.17%`, severe 비용에서 `52.59%`로,
요구한 `60%`와 `55%`에 미달했다. 엄격 비용 Wilson 95% 구간도 `46.10%~63.91%`라서
60% 정확도를 입증하지 못한다. PF와 기대값 점추정이 좋아도 이 실패를 덮을 수 없다.

## 분모를 고정한 이유

정확도는 고유한 `(symbol, entry_time)` 한 건을 한 기회로 센다. 같은 이벤트에 여러
기술지표가 겹쳐도 표본 수를 늘리지 않는다. 승리는 실제 펀딩과 왕복 비용을 뺀 순R이
엄격히 `0`보다 큰 경우뿐이며 0R도 실패다.

필터를 먼저 적용한 뒤 기권이나 조기 청산 때 새 돌파를 다시 열면, 약한 기회가 분모에서
사라지고 정확도가 부풀 수 있다. 이를 막기 위해 V5 CORE를 먼저 독립적으로 재생해
기회집합 Ω를 858건으로 고정한 뒤, V6는 그 안에서 선택 또는 기권만 한다. 기권·용량
거절·조기 청산은 Ω 밖 새 기회를 만들지 못한다. 초기 탐색에서 보였던 178건·52%대
수치는 이 계약과 일치하지 않아 폐기했다.

## 고정 가격·펀딩 게이트

V6는 롱 전용이다. 확정 1시간봉에서 다음 조건을 모두 만족해야 가격 WATCH가 생긴다.

- 이전 24시간 고가 종가 돌파
- 이전 확정봉 `SMA200`, `RSI14 > 50`, 거래량 조건
- 현재 후보를 제외한 `[t-365일, t)` eligible 돌파의 `ATR24/close` 선형 60분위 이하,
  최소 30표본과 full 365일
- 24시간 수익률 `>= 2%`
- 상승 몸통 `>= 이전 확정 ATR24 × 0.75`
- 같은 확정시각의 BTC 168시간 수익률 `>= 0`
- 외부 Bybit 실제 펀딩 정산률의 `(t-72시간, t]` 합 `<= 0.0004`

Ω 858건에서 단일 조건 통과 수는 펀딩 309, 24시간 수익률 522, 몸통 651, BTC 레짐
714였고 네 조건 교집합은 116이었다. 시간대·심볼마다 다른 임계값, 사후 손실 제외,
머신러닝 확률처럼 보이는 이진 점수는 사용하지 않았다. 별도 expanding logistic 후보도
OOS AUC가 약 `0.44`였고 0.60 임계값에서 진입이 없어 폐기했다.

Pine은 실제 거래소 펀딩 정산 원장을 직접 검증하지 못하므로 가격 게이트를 통과해도
`external_funding_required=true`, `external_funding_observed=null`인 WATCH만 낸다.
가격이나 거래량 proxy로 펀딩을 가장하지 않는다. 실제 펀딩 통과 판정은 Bybit API를
읽는 외부 페이퍼 실행기만 할 수 있다.

## 엄격 검증 결과

Bybit BTC·ETH·SOL·XRP·DOGE 무기한 선물 1시간봉과 실제 펀딩을 사용했다. 공통 진입
구간은 `2022-10-23 09:00 UTC`부터 `2026-08-21 02:00 UTC`까지이며 마지막 신호 뒤
73시간 embargo를 적용했다.

| 고정 Ω 기준 | 메타 shadow strict | 메타 shadow severe | 실제 실행 strict | 실제 실행 severe |
|---|---:|---:|---:|---:|
| 선택/체결 | 116 | 116 | 116 | 116 |
| 정확도 | 51.72% | 49.14% | 55.17% | 52.59% |
| PF | 2.320 | 2.007 | 2.113 | 1.824 |
| 위험정규화 기대값 | +0.220R | +0.185R | +0.181R | +0.148R |
| 순손익 | +25.56R | +21.50R | +17.01R | +13.83R |
| 실현 MDD | 3.84R | 4.24R | 2.58R | 2.89R |

Strict는 편도 12bp와 실제 펀딩을, severe는 편도 20bp와 펀딩 차변 2배·대변 0을
적용한다. MTM·마진·강제청산·부분체결은 아직 포함하지 않아 실현 MDD는 계좌 낙폭
보장이 아니다.

선택 coverage는 `116 / 858 = 13.52%`로 요구한 20%보다 낮다. 실행 coverage도
13.52%, 투입위험 coverage는 `93.7R / 858 = 10.92%`로 각각 요구한 15%보다 낮다.
표본 수도 요구한 300건에 미달했다.

실제 실행 정확도의 14·28·56·84일 달력 블록 bootstrap 5% 하한은 strict에서
`47.83% / 48.35% / 50.00% / 50.49%`, severe에서
`44.61% / 45.19% / 46.75% / 47.62%`였다. strict 하한 52%, severe 하한 50%를
모두 만족해야 하는 계약을 통과하지 못했다.

심볼별 severe 실제 정확도와 표본은 BTC `33.33%/21`, ETH `62.50%/24`, SOL
`67.65%/34`, XRP `33.33%/15`, DOGE `50.00%/22`다. 모든 심볼에서 최소 30건과
50% 이상을 요구하므로 교차종목 안정성도 실패다.

## 추매·손절·익절 계획

표시 계획은 최초 80%, 7ATR 최후 손절까지 거리의 20/40/60/80% 하락 지점마다 5%씩
최대 네 번 추가한다. 목표는 최초 기준가에서 +1R, 최대 보유·재검토는 24시간이다.
손절은 추매 뒤에도 더 멀어지지 않는다.

하지만 116건 중 추매 없음 98건, 1차 18건, 2·3·4차는 각각 0건이었다. 각 단계 최소
30건이라는 검증 게이트를 네 단계 모두 실패했다. 그래서 이 선들은
`UNVALIDATED ADD PLAN`이며 주문 지시가 아니다. 특히 2~4차 추매를 안전하거나 수익성
있는 것으로 말할 증거가 전혀 없다.

숏 대칭 규칙은 엄격 비용에서 승률 약 `40.36%`, PF 약 `1.112`, 기대값 약 `+0.022R`로
정확도 우선 기준에 부적합해 실행에서 제외했다. 하락 문맥은 신규 숏 진입이 아니라 기존
롱의 위험 축소·기권 참고 정보로만 사용한다.

## ICT·Bollinger·기술지표의 역할

ICT liquidity sweep·FVG·order-block·OTE proxy, Bollinger 상단 돌파, RSI, DMI/ADX,
거래량 비율, 48시간 거래량가중 가격 proxy를 계산해 alert JSON의 `context`에 넣는다.
차트는 사용자가 요청한 WATCH·추매 계획·TP·최후 SL만 보여 혼잡을 줄인다.

이 문맥 점수는 설명과 관찰용이지 진입 허가가 아니다. 여러 기술지표를 모두 AND로
묶는 방식은 과거 별도 검증에서 거래 수를 줄이고 비용 민감도를 키웠으며, 정확도와
견고성을 동시에 개선하지 못했다. `HIGH WATCH`는 BTC>SMA200과 DMI spread 2봉
조건을 추가한 비승격 관찰 옵션이고 기본값은 꺼져 있다.

## TradingView 파일 계약

- [`precision_v6_60m_watch.pine`](../tradingview/precision_v6_60m_watch.pine): 표준
  60분봉 전용 가격 WATCH. 다음 60분 시가는 계획 기준가일 뿐 체결 증거가 아니다.
- [`precision_v6_ltf_overlay.pine`](../tradingview/precision_v6_ltf_overlay.pine): 표준
  1·5·15분봉 전용. `[1] + lookahead_on`으로 직전 확정 60분 값을 전달한다.

두 파일 모두 비표준 차트와 잘못된 시간봉에서 fail-closed하고 주문 API를 호출하지
않는다. alert event ID는 심볼·60분·신호시각으로 고정해 중복 처리를 막는다.

## TradingView MCP 직접 적용 결과

`2026-08-31`에 로컬 TradingView Desktop 세션을 MCP로 직접 제어해 BYBIT
`BTCUSDT.P` 표준 차트에서 다음을 확인했다.

| 스크립트 | TradingView 저장 이름 | 현재 소스 확인 | 컴파일/표시 확인 |
|---|---|---:|---|
| 1·5·15분 | `DISCOVERY FAIL — Precision V6 Confirmed 1H WATCH [1m/5m/15m]` | 줄바꿈 정규화 SHA-256 `06d33b8...` 일치 | Pine 오류 0, 1·5·15분 전환마다 동일 study 유지 |
| 60분 | `DISCOVERY FAIL — Precision V6 External-Funding WATCH [60m]` | SHA-256 `c8066ac...` 일치 | 표준 60분에서 서버 저장·컴파일 오류 0 |

작업 후 차트는 `BYBIT:BTCUSDT.P` 15분봉으로 복귀했고 1·5·15분 최종 WATCH
study를 그대로 유지했다. TradingView가 저장본을 다시 열 때 LF를 CRLF로 바꾸므로
1·5·15분 소스 비교는 줄바꿈만 정규화한 뒤 수행했다. 코드 내용과 동결 해시는
일치한다.

현재 Basic 플랜의 차트당 지표 한도로 60분 스크립트를 기존 사용자 지표에 추가 탑재하는
런타임 재검사는 막혔다. 기존 지표를 삭제해 우회하지 않았다. 따라서 현재 60분 해시에
대한 **서버 저장·컴파일은 통과했지만 차트 런타임 증적 게이트는 미완료**로 남기며,
이 사실만으로도 전향 승격은 계속 실패다. 1·5·15분 최종본은 각 시간봉에서 실제
탑재 상태와 오류 0을 확인했다.

## 전향 승격 잠금

과거자료에서 임계값을 탐색했으므로 역사 결과는 성과와 무관하게 discovery FAIL이다.
[`config/precision_v6_preregistration.json`](../config/precision_v6_preregistration.json)에
파라미터와 전향 게이트를 고정했다. `2026-09-01 00:00 UTC` 이후 보지 않은 페이퍼
원장만 사용하며, 최소 12개월·선택 300건·체결 300건을 모두 채워야 한다.

그 뒤에도 strict/severe 정확도, coverage, PF·기대값, 네 달력 블록 bootstrap, 5/5
심볼, 추매 단계별 최소 30건, 실제 체결·마진·강제청산 재생을 모두 통과해야 한다.
파라미터·유니버스·신호 또는 PnL을 바꾸는 수정은 새 버전으로 분리하고 전향 시계를
초기화한다. 모든 조건을 통과해도 실거래 자동 전환은 금지하며 사용자 별도 승인이
필요하다.

## 재현

```bash
.venv/bin/python -m lab.validate_precision_candidate
.venv/bin/python -m pytest -q tests/test_validate_precision_candidate.py
```

기계 판정은 `logs/validation/precision_candidate_v6/latest_results.json`, append-only
탐색 원장은 `logs/validation/precision_candidate_trials.jsonl`에 저장된다. 로컬 로그는
Git 추적 대상이 아니다. 과거 수익은 미래 수익을 보장하지 않는다.

## 참고 경계

- 사용자 참고 영상: [YouTube m5aGneOBkSo](https://www.youtube.com/watch?v=m5aGneOBkSo)
- 사용자 ICT 참고 글: [쉽알남 — 상위 10% 트레이더가 되기 위한 가장 현실적인 방법](https://fanding.kr/@easychart/post/176736/)
- 공식 펀딩 설명: [TradingView Funding rate guide](https://www.tradingview.com/support/solutions/43000762390-funding-rate-a-guide-to-market-sentiment/)
- 실제 정산 원장: [Bybit Get Funding Rate History](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)
- 확정 HTF 전달 계약: [TradingView Pine other timeframes and data](https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/)

Fanding 본문은 로그인 경계 때문에 공개 메타정보 외 내용을 자동 검증할 수 없었다. ICT
용어 자체를 수익 근거로 간주하지 않고, 코드로 명확히 정의한 proxy만 문맥으로 사용한
이유다. 영상 구조도 그대로 신뢰하지 않고 별도 비용 후 재생에서 살아남은 규칙만 남겼다.
