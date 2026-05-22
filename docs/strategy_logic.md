# ICT 전략 로직

## 진입 5단계 프로세스

### 1단계 — Kill Zone 확인 (`kill_zone.py`)
거래는 유동성이 높은 Kill Zone 시간에만 실행합니다.

- **London Kill Zone**: 02:00 ~ 05:00 UTC
- **New York Kill Zone**: 07:00 ~ 10:00 UTC

```python
from src.strategy.kill_zone import is_in_kill_zone
if not is_in_kill_zone(datetime.now(timezone.utc)):
    return None  # Kill Zone 외부 → 신호 없음
```

### 2단계 — 4H 시장 구조 확인 (`market_structure.py`)
4시간봉으로 추세 방향을 결정합니다.

- **BOS (Break of Structure)**: 이전 스윙 고점/저점 돌파 → 추세 방향 확정
- **CHoCH (Change of Character)**: 추세 전환 신호 탐지

```python
from src.strategy.market_structure import detect_bos, detect_choch
trend = detect_bos(df_4h) or detect_choch(df_4h)
# 'bullish' → Long 방향, 'bearish' → Short 방향
```

### 3단계 — 1H OB/FVG 존 탐지 (`order_block.py`, `fvg_detector.py`)
1시간봉에서 고확률 반전 구간을 식별합니다.

- **Order Block**: 강한 이동 직전 마지막 반대색 캔들 구간
- **FVG**: 3봉 패턴에서 발생한 미체워진 갭

```python
from src.strategy.fvg_detector import detect_fvg, is_price_in_fvg
from src.strategy.order_block import detect_order_blocks, is_price_in_ob

fvg_list = detect_fvg(df_1h)
ob_list  = detect_order_blocks(df_1h)
# 현재 가격이 존 내부에 있어야 진행
```

### 4단계 — 15m OTE 레벨 확인 (`ote.py`)
최근 스윙 기준 피보나치 되돌림 0.618 ~ 0.786 구간에서만 진입합니다.

- **Bullish OTE**: 저점 → 고점 이동 후 38.2%~61.8% 되돌림 구간
- **Bearish OTE**: 고점 → 저점 이동 후 되돌림 구간

```python
from src.strategy.ote import calculate_ote_zone, is_price_in_ote
zone = calculate_ote_zone(recent_high, recent_low, direction)
if not is_price_in_ote(current_price, zone):
    return None
```

### 5단계 — 리스크 계산 (`position_sizer.py`, `risk_manager.py`)
진입 전 반드시 R:R ≥ 1:2 확인 후 포지션 수량 결정합니다.

```
risk_amount = trading_capital × risk_per_trade (1%)
qty         = risk_amount / |entry - stop_loss|
take_profit = entry + (entry - stop_loss) × rr_ratio
```

```python
from src.risk.position_sizer import calculate_position_size, calculate_take_profit
qty = calculate_position_size(capital, 0.01, entry, stop_loss, leverage=5)
tp  = calculate_take_profit(entry, stop_loss, rr_ratio=2.0)
```

## 신호 발생 조건 요약

| 조건 | 타임프레임 | 판별 기준 |
|------|-----------|---------|
| 추세 방향 | 4H | BOS 또는 CHoCH |
| 유동성 존 | 1H | OB 또는 FVG |
| 진입 타이밍 | 15m | Kill Zone 내 OTE 구간 |
| 리스크 확인 | - | R:R ≥ 1:2 |

**4개 조건 모두 충족 시에만 신호 발생** (`signal_engine.py` → `generate_signal()`)
