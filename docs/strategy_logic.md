# ICT 전략 로직

> **2026-06 개정**: 6개월 신호연구(`docs/RESEARCH_2026-06.md`) 결과로 킬존 하드 게이트 폐지(24h 진입),
> 출구 기하 2.0 ATR / RR 2.5, 컨플루언스 점수제 + BTC 추세 정렬이 도입됨.
> 수치의 단일 진실원천은 `config/` — 이 문서는 설명용.

## 진입 프로세스 (컨플루언스 점수제)

신호는 단계별 게이트 + 0~100 점수로 평가된다 (`signal_engine.py`의 `scan_symbol()`):

| 요소 | 게이트? | 점수 |
|------|---------|------|
| 4H 추세 (BOS/CHoCH) | ✅ 필수 | BOS 30 / CHoCH 22 |
| 1H 존 (FVG/OB) | ✅ 필수 | 동시 35 / 단일 20 |
| 15m OTE | ✅ 필수 | 깊이 따라 최대 30 |
| 킬존 (런던 07-10 / 뉴욕 12-15 UTC) | ❌ 가점만 | +5 |
| 거래량 | ❌ 태깅만 | — |

진입 확정 = 필수 3게이트 통과 + **점수 ≥ min_score(70)** + R:R ≥ 2.5.
점수에 따라 리스크 차등(A급 0.7% / B급 0.5% / C급 0.3%).

### 1단계 — 4H 시장 구조 확인 (`market_structure.py`)
4시간봉으로 추세 방향을 결정합니다.

- **BOS (Break of Structure)**: 이전 스윙 고점/저점 돌파 → 추세 방향 확정
- **CHoCH (Change of Character)**: 추세 전환 신호 탐지

```python
from src.strategy.market_structure import detect_bos, detect_choch
trend = detect_bos(df_4h) or detect_choch(df_4h)
# 'bullish' → Long 방향, 'bearish' → Short 방향
```

### 2단계 — 1H OB/FVG 존 탐지 (`order_block.py`, `fvg_detector.py`)
1시간봉에서 고확률 반전 구간을 식별합니다.

- **Order Block**: 강한 이동 직전 마지막 반대색 캔들 구간
- **FVG**: 3봉 패턴에서 발생한 미체워진 갭
- FVG+OB **동시존은 신호연구에서 실측 우위**(+0.093R vs 단일 -0.035R) → 가점 가중

```python
from src.strategy.fvg_detector import detect_fvg, is_price_in_fvg
from src.strategy.order_block import detect_order_blocks, is_price_in_ob

fvg_list = detect_fvg(df_1h)
ob_list  = detect_order_blocks(df_1h)
# 현재 가격이 존 내부에 있어야 진행
```

### 3단계 — 15m OTE 레벨 확인 (`ote.py`)
최근 스윙 기준 피보나치 되돌림 0.618 ~ 0.786 구간에서만 진입합니다.
킬존 여부는 진입을 막지 않으며 가점(+5)과 세션 태깅에만 사용됩니다 (24h 진입).

```python
from src.strategy.ote import calculate_ote_zone, is_price_in_ote
zone = calculate_ote_zone(recent_high, recent_low, direction)
if not is_price_in_ote(current_price, zone):
    return None
```

### 4단계 — BTC 추세 정렬 (`market_structure.detect_htf_trend`)
BTC 4H EMA50 추세와 역행하는 신호는 차단하지 않되 **리스크를 최하단(0.3%)으로 강등**.
(연구: 역행 신호는 조건부 -0.258R — 단 전체표본 +0.145R라 차단 대신 강등)

### 5단계 — 리스크 계산 (`position_sizer.py`, `risk_manager.py`)
손절은 ATR 기반, 목표는 R:R 2.5 (진입 전 확인).

```
손절가      = entry ∓ 2.0 × ATR(14)
risk_amount = trading_capital × 점수별 risk_pct (0.3~0.7%)
qty         = risk_amount / |entry - stop_loss|
take_profit = entry ± |entry - stop_loss| × 2.5
레버리지     = 자동 (손절거리 기반, 최대 10x)
```

```python
from src.risk.position_sizer import calculate_position_size, calculate_take_profit
qty = calculate_position_size(capital, risk_pct, entry, stop_loss, leverage)
tp  = calculate_take_profit(entry, stop_loss, rr_ratio=2.5)
```

## 신호 발생 조건 요약

| 조건 | 타임프레임 | 판별 기준 | 게이트 |
|------|-----------|---------|--------|
| 추세 방향 | 4H | BOS 또는 CHoCH | ✅ |
| 유동성 존 | 1H | OB 또는 FVG | ✅ |
| 진입 타이밍 | 15m | OTE 구간 (24h, 킬존 무관) | ✅ |
| 컨플루언스 | - | 점수 ≥ 70 | ✅ |
| 리스크 확인 | - | R:R ≥ 1:2.5 | ✅ |
| 킬존/거래량/BTC정렬 | - | 가점·태깅·사이징 조정 | ❌ |

**필수 게이트 모두 충족 시에만 진입** — 진입 기준은 자동학습이 실거래 데이터로 계속 조정한다.
