# 시스템 아키텍처

## 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│                        bot.py                           │
│                     (메인 봇 루프)                        │
└───────────┬─────────────────────┬───────────────────────┘
            │                     │
    ┌───────▼──────┐      ┌───────▼──────┐
    │  signal_     │      │  risk_       │
    │  engine.py   │      │  manager.py  │
    │  (신호 통합)  │      │  (리스크 통합)│
    └───┬──────────┘      └──────┬───────┘
        │                        │
┌───────▼────────────────────────▼──────────────────────┐
│                  전략 레이어 (strategy/)                 │
│  market_structure → fvg_detector → order_block        │
│  kill_zone → ote → signal_engine                      │
└───────────────────────────────────────────────────────┘
        │                        │
┌───────▼──────┐       ┌─────────▼──────┐
│  exchange/   │       │  paper_trading/ │
│  bybit_      │       │  paper_engine  │
│  client.py   │       │  (가상 실행)    │
└───────┬──────┘       └────────────────┘
        │
┌───────▼──────┐
│  data/       │
│  candle_     │
│  fetcher.py  │
│  data_store  │
└──────────────┘
```

## 데이터 흐름

```
[Bybit API] → candle_fetcher → [OHLCV DataFrame]
                                      │
              ┌───────────────────────┼────────────────┐
              ▼                       ▼                 ▼
        market_structure         fvg_detector     order_block
        (BOS/CHoCH 4H)          (FVG 1H)         (OB 1H)
              │                       │                 │
              └───────────────────────┼─────────────────┘
                                      ▼
                               signal_engine
                               (Kill Zone 체크 + OTE)
                                      │
                              TradeSignal 발생
                                      │
                         ┌────────────┴──────────────┐
                         ▼                           ▼
                   RiskManager                  PaperEngine
                   (포지션 사이징)               (가상 실행)
                         │                           │
                   CircuitBreaker              SQLite 저장
                   (손실 한도 체크)             (성과 추적)
```

## 모듈별 역할

| 모듈 | 역할 |
|------|------|
| `bybit_client.py` | Bybit REST API 래퍼, 재시도 로직 |
| `candle_fetcher.py` | OHLCV 수집 (REST/WebSocket) |
| `market_structure.py` | 스윙 포인트, BOS, CHoCH 탐지 |
| `fvg_detector.py` | Fair Value Gap 탐지 및 추적 |
| `order_block.py` | Order Block 탐지 |
| `kill_zone.py` | London/NY 세션 시간 필터 |
| `ote.py` | 피보나치 OTE 존 계산 |
| `signal_engine.py` | 멀티 타임프레임 신호 통합 |
| `position_sizer.py` | 리스크 기반 수량 계산 |
| `circuit_breaker.py` | 손실 한도 서킷브레이커 |
| `paper_engine.py` | 가상 거래 실행 및 성과 추적 |
