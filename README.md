# ICT Trading Bot — Bybit Futures

ICT 기법(FVG, Order Block, BOS/CHoCH, Kill Zone, OTE) 기반 자동화 트레이딩 봇.
페이퍼 트레이딩 → 실거래 순차 전환 구조.

## 설치

```bash
git clone <repo-url>
cd trading-bot
pip install -r requirements.txt
```

## .env 설정

`.env.example`을 복사하여 `.env` 생성:

```bash
cp .env.example .env
```

`.env` 파일 편집:

```
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

> Testnet 키는 https://testnet.bybit.com 에서 발급

## 페이퍼 트레이딩 실행

```bash
python src/bot.py --mode paper
```

## 실거래 실행

페이퍼 트레이딩 성과가 실전 전환 기준(config promote 섹션 — 30건+/승률38%+/PF1.5+/MDD10%- 등) 충족 후:

```bash
python src/bot.py --mode live
```

## 백테스트 실행

```bash
python backtest/backtest_runner.py --symbol BTC/USDT:USDT --start 2024-01-01 --end 2024-12-31
```

## 테스트

```bash
pytest tests/ -v
```

## 프로젝트 구조

```
trading-bot/
├── config/          # 설정 파일 (config.yaml, strategy_params.yaml)
├── src/
│   ├── exchange/    # Bybit API 클라이언트
│   ├── strategy/    # ICT 전략 모듈
│   ├── risk/        # 리스크 관리, 서킷브레이커
│   ├── data/        # 캔들 수집, SQLite 저장
│   ├── paper_trading/ # 페이퍼 트레이딩 엔진
│   └── bot.py       # 메인 봇 루프
├── tests/           # pytest 테스트
├── backtest/        # 백테스트 엔진
├── docs/            # 아키텍처, 전략, 리스크 문서
└── logs/            # 런타임 로그, SQLite DB
```

## 문서

- [아키텍처](docs/architecture.md)
- [전략 로직](docs/strategy_logic.md)
- [리스크 규칙](docs/risk_rules.md)
