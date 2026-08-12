# 검증 우선 암호화폐 트레이딩 봇

Bybit 실시간 데이터로 전략을 연구하고 페이퍼·데모 환경에서 검증하는 트레이딩 봇이다.
기존 ICT 전략은 벤치마크로 보존하지만, 룩어헤드를 제거한 워크포워드 결과 비용 차감 후
메이커 OOS `-0.021R`, 최종 홀드아웃 `-0.368R`로 견고한 엣지가 확인되지 않아
**실전 승급이 동결**되어 있다.

## 설치

```bash
git clone <repo-url>
cd trading-bot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Python 3.11 이상이 필요하다.

## 페이퍼 트레이딩 실행

```bash
python -m src.bot
python -m src.runner --interval 900
```

기본 설정은 `runtime.mode: paper`, `testnet: true`, `runtime.live_enabled: false`다.
Discord 알림이 필요하면 `.env`에 `DISCORD_WEBHOOK_URL`을 설정한다.

## 연구와 검증

```bash
# 기존 ICT 비교 기준(승급 증거 아님)
python -m research.wfo --start 2024-01-01

# 동일 Bybit 상품의 미래 OI·펀딩·호가·청산 수집
python -m src.data.collector \
  --symbols BTC/USDT:USDT ETH/USDT:USDT

# 사전등록된 8개 캐리 + 8개 강제흐름 후보의 단일 증거 파이프라인
python -m research.pipeline \
  --input logs/evidence/input.json \
  --output logs/evidence/run

# 통과한 offline/demo 아티팩트만 다음 실행 단계에 연결
PROMOTION_ARTIFACT_SHA256=<고정된_SHA256> \
  python -m src.promotion activate \
  --artifact logs/evidence/promotion.json \
  --output logs/strategy_activation.json

python -m src.backtest --symbol BTC/USDT:USDT --days 30
```

과거 성과만으로 실전 모드가 열리지 않는다. 후보 전략은 사전등록된 가설, 비용 포함 WFO,
미래 데이터 데모 검증과 수동 승인 리포트를 순서대로 통과해야 한다.

공식 과거 API가 없는 liquidation·orderbook 이력은 collector 실행 이후 데이터만 사용한다.
수집 공백은 0으로 보간하지 않으며, 데이터 완전성 99%와 미확인 gap 한도를 통과하지 못하면
파이프라인 결과는 `insufficient_data`다. 현재 승급 가능한 전략은 **0개**다.

실전 주문 코드는 기본 비활성화되어 있으며 승인 토큰과 승인 리포트 해시가 모두 없으면
실행기가 생성되지 않는다. 승인 리포트도 `stage: demo`, `passed: true`, 실행 주문과 같은
`strategy_version`이어야 한다. 이 보호장치를 우회하는 실행 방법은 제공하지 않는다.

## 테스트

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term --cov-fail-under=80
```

## 프로젝트 구조

```
trading-bot/
├── config/          # 설정 파일 (config.yaml, strategy_params.yaml)
├── src/
│   ├── exchange/    # Bybit API 클라이언트
│   ├── strategy/    # 시간안전 신호·ICT 벤치마크·후보 전략
│   ├── risk/        # 리스크 관리, 서킷브레이커
│   ├── data/        # 캔들 수집, SQLite 저장
│   ├── paper_trading/ # 페이퍼 트레이딩 엔진
│   └── bot.py       # 메인 봇 루프
├── tests/           # pytest 테스트
├── research/        # WFO·가설 원장·후보 연구
├── docs/            # 아키텍처, 전략, 리스크 문서
└── logs/            # 런타임 로그, SQLite DB
```

## 문서

- [아키텍처](docs/architecture.md)
- [전략 로직](docs/strategy_logic.md)
- [리스크 규칙](docs/risk_rules.md)
- [정직한 WFO 결론](docs/WFO_2026-06.md)
- [검증·실전 승급 정책](docs/VALIDATION_AND_LIVE_GATES.md)
- [연구 증거 정책](docs/RESEARCH_EVIDENCE_POLICY.md)
