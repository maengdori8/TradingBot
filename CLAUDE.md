# ICT Paper Trading Bot

## 프로젝트 개요
ICT(Inner Circle Trader) 전략 기반 암호화폐 모의 트레이딩 봇.
실전과 100% 동일한 구조로 동작하되, 자산만 가상 자산(페이퍼)으로 운용한다.
GitHub Actions에서 15분마다 실행되며, Bybit 실시간 시세를 사용한다.

## 목표
- 실시간 시세 기반 실전 동일 모의 트레이딩
- 페이퍼 트레이딩 성과가 실전 전환 기준 충족 시 실거래 전환 가능한 구조

## 기술 스택
- Python 3.11+, ccxt >= 4.2, pandas, numpy, ta, SQLite
- CI: GitHub Actions (15분 cron)
- 알림: Discord Webhook

## 디렉토리 구조
```
src/
  bot.py                    # 메인 봇 루프 (Orchestrator 전용)
  strategy/                 # ICT 전략 모듈 (Strategy Agent 전용)
    market_structure.py     #   BOS / CHoCH 탐지
    fvg_detector.py         #   Fair Value Gap
    order_block.py          #   Order Block
    kill_zone.py            #   Kill Zone 시간 필터
    ote.py                  #   Optimal Trade Entry
    signal_engine.py        #   멀티 타임프레임 신호 통합
  exchange/                 # 거래소 연결 (Infra Agent 전용)
    bybit_client.py         #   멀티 거래소 퍼블릭 클라이언트
    order_executor.py       #   주문 실행 (페이퍼/실전 공통 인터페이스)
  data/                     # 데이터 파이프라인 (Infra Agent 전용)
    candle_fetcher.py       #   캔들 데이터 수집
    data_store.py           #   데이터 캐싱/저장
  risk/                     # 리스크 관리 (Risk Agent 전용)
    risk_manager.py         #   리스크 통합 관리
    position_sizer.py       #   포지션 사이징
    circuit_breaker.py      #   서킷브레이커
  paper_trading/            # 페이퍼 트레이딩 (Risk Agent 전용)
    paper_engine.py         #   가상 주문 실행 및 성과 추적
  notification/             # 알림 (Orchestrator 관리)
    discord_bot.py          #   Discord Webhook
config/
  config.yaml               # 봇 설정 (자본, 리스크 한도, 심볼)
  strategy_params.yaml      # 전략 파라미터 (FVG, OB, OTE 등)
tests/                      # 테스트 (QA Agent 전용)
  test_*.py                 #   단위 테스트
  integration/              #   통합 테스트
logs/                       # 런타임 로그 및 DB (gitignore)
docs/                       # 문서
```

## 팀 에이전트 규칙

### 파일 소유권 (절대 준수)
| 에이전트 | 소유 파일 | 수정 금지 |
|---------|----------|----------|
| Orchestrator | bot.py, config/, docs/, CLAUDE.md | - |
| strategy-agent | src/strategy/ 전체 | exchange/, risk/, paper_trading/, data/ |
| infra-agent | src/exchange/, src/data/ | strategy/, risk/, paper_trading/ |
| risk-agent | src/risk/, src/paper_trading/ | strategy/, exchange/, data/ |
| qa-agent | tests/, .github/workflows/ | src/ 전체 (읽기만 가능) |

### 브랜치 규칙
- strategy-agent: `feat/strategy-*`
- infra-agent: `feat/infra-*`
- risk-agent: `feat/risk-*`
- qa-agent: `test/*`
- main 병합은 Orchestrator만 수행

### 인터페이스 계약

#### DataFrame 형식 (모든 OHLCV 데이터)
```python
# columns: open, high, low, close, volume (float64)
# index: timestamp (DatetimeIndex, UTC)
```

#### 거래소 클라이언트 인터페이스
```python
class MarketDataClient:
    def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame
    def fetch_ticker(symbol: str) -> dict
    def fetch_current_price(symbol: str) -> float
```

#### 트레이딩 엔진 인터페이스 (페이퍼/실전 공통)
```python
class TradingEngine(ABC):
    def open_position(symbol, direction, entry_price, qty, stop_loss, take_profit) -> Position | None
    def close_position(position, exit_price, reason) -> float  # PnL
    def check_stops(symbol, current_high, current_low) -> None
    def get_performance() -> dict
```

#### 시그널 형식
```python
@dataclass
class TradeSignal:
    direction: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    reason: str
    rr_ratio: float
```

### ccxt 심볼 규칙
- 선물 (Bybit): `BTC/USDT:USDT` (linear perpetual)
- 현물 fallback: `BTC/USDT`, `BTC/USD`
- defaultType: `"swap"` (절대 `"linear"` 또는 `"future"` 사용 금지)

## 코딩 규칙
- `from __future__ import annotations` 모든 파일 첫줄
- 모든 함수에 type hint + docstring (한국어)
- `logging.getLogger(__name__)` 사용, `print()` 금지
- bare `except:` 금지 → 구체적 예외 타입
- 시크릿은 `os.getenv()`, 절대 하드코딩 금지
- 금액 계산: `round(value, 8)` 부동소수점 주의
- config 값 하드코딩 금지 → config.yaml / strategy_params.yaml 에서 읽기

## 빌드 & 실행
```bash
pip install -r requirements.txt
python -m src.bot                    # 봇 실행
pytest tests/ -v                     # 테스트
pytest tests/ --cov=src --cov-report=term  # 커버리지
```

## 커밋 메시지 형식
```
[영역] 설명

예: [strategy] FVG 파라미터를 strategy_params.yaml에서 로드하도록 변경
예: [infra] binanceus 제거 및 Bybit defaultType swap으로 수정
예: [risk] 백테스팅 프레임워크 초기 구현
예: [test] signal_engine 단위 테스트 추가
```
