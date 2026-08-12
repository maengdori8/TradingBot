# 시스템 아키텍처

## 안전 경계

`src.bot`은 닫힌 봉을 사용하는 ICT 페이퍼 벤치마크 오케스트레이터다. `demo`와 `live`를
이 경로에서 선택하면 시작 전에 실패한다. 거래소 주문은 공통 실행 계약을 사용하는 별도
`BybitOrderExecutor`를 통해서만 가능하며, 실전 실행기는 미래 데모 게이트·수동 승인·고정
리포트 해시가 모두 맞아야 생성된다.

```text
Bybit 시점보존 데이터
        │
        ▼
DecisionContext ── 완전히 닫힌 15m·1h·4h만 선택
        │
        ├── ICT 신호 엔진 (비교 기준)
        ├── 현물–무기한 캐리 후보
        └── OI·청산·펀딩·오더북 강제흐름 후보
        │
        ▼
공통 재생·주문 계약
        ├── paper: 주문장 기반 체결 모델
        ├── demo: Bybit Demo + private event 영구 저장
        └── live: 승인 리포트 + 파일럿 킬스위치
```

## 결정과 데이터

- `DecisionContext`는 `decision_time`, `data_cutoff`, `bar_close_time`,
  `strategy_version`, `run_id`를 모든 신호 호출에 전달한다.
- 데이터 출처에는 거래소·상품·실제 심볼·거래소 시각·수신 시각을 보존한다.
- Bybit 선물 신호에는 다른 거래소 현물 폴백을 섞지 않는다. 출처 불일치, 미래 데이터,
  stale 데이터는 신규 진입을 막는다.
- `strategy_version + symbol + bar_close_time` 결정 키와 주문의 `client_order_id`를
  SQLite에서 원자적으로 선점해 재시작·중복 실행에도 주문을 한 번만 만든다.
- OI·펀딩·오더북은 각각 360초·60초·5초의 별도 최신성 한도를 적용한다. OI의 5분
  완결 버킷 때문에 오더북 최신성까지 느슨하게 만들지 않는다.
- 공식 과거 API가 없는 public liquidation과 주문장 이력은 24시간 collector가 직접
  축적한다. heartbeat가 끊긴 구간은 청산 0건으로 해석하지 않고 데이터 공백으로 남긴다.
- 연구 입력은 원시 payload, 거래소·수신 시각, 코드 commit과 파일 SHA-256을 묶은
  `DataManifest`로 고정한다. completeness 99% 미만 또는 15분 초과 미확인 gap이 있으면
  승급 증거를 생성하지 않는다.

## 실행과 영속 상태

| 모듈 | 역할 |
|------|------|
| `src/exchange/contracts.py` | `TradingMode`, 주문·체결·실행 보고서 공통 계약 |
| `src/exchange/order_executor.py` | Bybit Demo/Live 주문, 계정 수수료, 보호주문, 대사 |
| `src/data/execution_store.py` | 주문·체결·private WebSocket·수수료 스냅샷 영구 저장 |
| `src/data/feature_store.py` | OI·펀딩·주문장 복합 특징과 public 청산 이벤트 시점 보존 |
| `src/data/collector.py` | 동일 Bybit 상품의 24시간 특징 수집·heartbeat·백필 |
| `src/paper_trading/execution_model.py` | 주문장 깊이, 부분체결, IOC/FOK/PostOnly, 불리한 선택 모델 |
| `src/paper_trading/paper_engine.py` | 현금·증거금·미실현손익 자산곡선과 순성과 |
| `src/risk/validation_gate.py` | 오프라인/데모 통계 승급 게이트와 승인 리포트 |
| `src/risk/promotion_artifact.py` | 코드·데이터·가설·성과 계보를 고정한 승급/활성화 계약 |
| `src/risk/live_guard.py` | 실전 파일럿 한도, 영속 킬스위치, 증액 판정 |
| `src/promotion.py` | 고정 외부 해시를 검증하고 offline→demo, demo→live 전이만 허용 |
| `research/hypothesis_ledger.py` | 실행 전 가설 등록과 성공·실패 결과 append-only 보존 |
| `research/pipeline.py` | 사전등록 후보만 재생하고 해시 고정 증거를 생성하는 단일 CLI |

Demo private 주문·체결 이벤트는 거래소 보존기간에 기대지 않고 로컬 DB에 저장한다. 시작과
주기적 실행 때 REST 주문·포지션·체결·잔고를 다시 조회해 대사하며, 불일치·고아 포지션·중복
주문은 실전 신규 주문 중단과 idempotent reduce-only IOC 안전 청산 사유다.

## 승급 흐름

```text
사전등록 가설
  → expanding WFO + purge/embargo + 실제 실행 제약
  → 오프라인 통계 게이트
  → 90일·유효 100베팅 미래 Demo
  → 해시 고정 승인 리포트 + 수동 토큰
  → 제한된 Live 파일럿
  → 90일 이후 30일마다 최대 25% 증액 검토
```

어느 단계든 전략 버전이나 파라미터를 바꾸면 이전 증거는 무효가 되고 처음부터 다시 검증한다.
현재 활성화 가능한 후보 전략은 0개이며, `ict-benchmark-v1`은 paper 비교 기준으로만 실행한다.
승급 파일은 canonical JSON이어야 하고, 파일 자체 SHA-256과 코드·데이터·가설·전략 해시가
모두 일치해야 한다. offline 통과 파일은 Demo만, 90일 Demo 통과 파일은 Live만 열 수 있다.
