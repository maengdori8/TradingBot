# 검증·데모·실전 승급 정책

## 원칙

수익은 구현 기능이 아니라 검증 결과다. 현재 ICT 전략은 비용 차감 후 견고한 OOS 엣지가
없으므로 `ict-benchmark-v1`으로 고정하고 실전 승급을 허용하지 않는다. 과거 데이터를 이미
여러 차례 탐색했기 때문에 최종 증거는 코드 동결 이후 수집되는 미래 데이터여야 한다.

자동 튜너는 보고서만 만들며 실행 파라미터를 변경하지 않는다. 실패한 가설과 실험도 가설
원장에서 삭제하지 않고, 전략군별 사전 정의 설정은 최대 20개로 제한한다.

## 오프라인 게이트

아래 조건을 모두 만족해야 Bybit Demo 단계로 이동한다.

- 유효 OOS 베팅 200건 이상, 기간 12개월 이상, 복수 시장 레짐 포함
- 기본 비용 및 1.5배 비용에서 순기대값 양수
- 일·심볼 클러스터 블록 부트스트랩 순기대값 95% 신뢰하한 양수
- UTC 일별 Sharpe 1.0 이상, Profit Factor 1.2 이상, MDD 10% 이하
- Deflated Sharpe 양의 확률 95% 이상, PBO 10% 미만, SPA p-value 0.05 미만
- 단일 심볼 또는 단일 분기의 이익 기여 25% 이하
- 2배 비용 스트레스 손실 10% 이하

수수료는 계정의 실제 fee-rate 스냅샷을 사용한다. 펀딩, 부분체결, 미체결, 주문 취소,
불리한 선택과 최소 주문수량을 비용 모델에 포함한다.

## 미래 데모 게이트

Bybit Demo에서 최소 90일과 유효 독립 베팅 100건을 모두 충족할 때까지 전략 버전과
파라미터를 고정한다.

- 순기대값 95% 신뢰하한 양수, 일별 Sharpe 1.0 이상, PF 1.2 이상, MDD 7.5% 이하
- 백테스트 대비 체결가 절대오차 중앙값 5bp 이하, 95백분위 25bp 이하
- 체결률 예측오차 10%p 이하
- 주문·체결·포지션·잔고 대사율 100%
- 고아 포지션과 중복 주문 0건

Demo 주문 이력의 거래소 보존기간과 무관하게 모든 private order/execution 이벤트를 로컬
DB에 저장한다.

## 실전 파일럿

실전 실행기는 다음 조건이 동시에 참일 때만 생성할 수 있다.

1. `runtime.mode: live`
2. `runtime.live_enabled: true`
3. 환경변수 승인 토큰 존재
4. 승인 검증 리포트가 존재하고 설정된 SHA-256과 일치
5. 리포트의 전략 버전이 실행 전략 버전과 일치

최초 자본은 100만원과 전체 투자 가능 자산의 5% 중 작은 값이다. 거래당 위험 0.1%,
동시 총 손절 위험 0.5%, 방향성 명목노출 1배, 최대 레버리지 2배·격리마진을 사용한다.
델타중립은 양쪽 합산 2배, 순델타 0.1배를 넘지 않는다.

일 손실 0.5%, 주 손실 1.5%, 고점 대비 3% 하락, stale feed, 대사 불일치 또는 연속 주문
오류가 발생하면 신규 주문을 중단한다. 열린 포지션은 거래소 서버의 reduce-only 보호주문과
비상청산 절차로 관리한다.

## 참고

- Bybit 수수료: <https://www.bybit.com/en/help-center/article/Trading-Fee-Structure>
- 계정 수수료 API: <https://bybit-exchange.github.io/docs/v5/account/fee-rate>
- 주문 동작: <https://bybit-exchange.github.io/docs/v5/order/create-order>
- Demo Trading: <https://bybit-exchange.github.io/docs/v5/demo>
- White Reality Check: <https://doi.org/10.1111/1468-0262.00152>
- Deflated Sharpe Ratio: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
