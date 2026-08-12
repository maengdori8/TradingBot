# 연구 증거 정책

## 현재 판정

- `ict-benchmark-v1`: 오프라인 게이트 미통과. OOS -0.021R, 최종 홀드아웃 -0.368R.
- 기존 funding v2: 메이커 WFO Sharpe -0.307, MDD 34.7%. `legacy_non_evidence`.
- 승급 가능한 전략: 0개. 통과 리포트가 생기기 전에는 Demo와 Live 주문을 허용하지 않는다.

## 사용하는 근거

연구 입력과 거래 규칙은 Bybit 공식 문서를 1순위로 사용한다.

- 과거 캔들과 turnover: <https://bybit-exchange.github.io/docs/v5/market/kline>
- 과거 펀딩: <https://bybit-exchange.github.io/docs/v5/market/history-fund-rate>
- 과거 OI: <https://bybit-exchange.github.io/docs/v5/market/open-interest>
- 상품 상장·tick·qty·funding interval: <https://bybit-exchange.github.io/docs/v5/market/instrument>
- 현재 주문장과 실시간 delta: <https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook>
- 전체 public 청산: <https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation>
- 계정별 수수료: <https://bybit-exchange.github.io/docs/v5/account/fee-rate>
- Demo 보존기간과 endpoint: <https://bybit-exchange.github.io/docs/v5/demo>

통계 검증은 White Reality Check, SPA, Deflated Sharpe와 CSCV PBO를 함께 사용한다.
논문이나 기존 백테스트 하나만으로 전략을 활성화하지 않는다.

## 후보 채택과 제외

델타중립 현물–무기한 캐리는 funding과 basis의 구조적 관계가 있어 연구 후보로 유지한다.
다만 기존 저장 결과는 비용·체결·가설 예산 계약이 다르므로 새 파이프라인에서 다시 검증한다.

OI·펀딩·청산·오더북 강제흐름도 후보로 유지하지만, 공식 과거 API로 확인할 수 없는 청산과
호가 이력은 자체 수집 이후만 사용한다. heartbeat가 없는 구간을 청산 0건으로 보간하지 않는다.
거래소 OI의 지연·오표기 가능성도 보고돼 같은 Bybit 상품의 원시 시각과 수신 시각을 함께
보존한다: <https://arxiv.org/abs/2310.14973>.

15분 주기 효과, 고빈도 시장조성, 강화학습, 딥러닝, 교차거래소 신호는 제외한다. 최근 또는
단일 시장 연구의 결과를 저회전 Bybit 전략으로 직접 이전할 근거가 부족하고 v1 비용·운영
범위를 벗어나기 때문이다.

## 증거가 되기 위한 최소 조건

- 가설 등록이 데이터 조회와 실행보다 먼저 존재해야 한다.
- 데이터, 코드 commit, 파라미터, 비용 스냅샷의 SHA-256이 모두 고정돼야 한다.
- 사전등록된 후보군 전체를 같은 일별 시계열에 맞춰 PBO·SPA·DSR 입력으로 사용한다.
- feed completeness 99% 이상이고 15분 초과 미확인 공백이 없어야 한다.
- 오프라인·Demo 게이트의 모든 항목을 동시에 통과해야 하며 수동으로 성과 값을 입력해
  승인 리포트를 만들 수 없어야 한다.
