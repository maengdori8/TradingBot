"""Track E — 단타 팜 엔진 (10셀 = 5전략 x 2바스켓, 페이퍼 전용).

사전등록: docs/TRACKE_SCALP_FARM_2026-08-27.md (명세 v1, T0 이후 수치·규칙 변경 금지).
지위: 실주문·실자금 0. 어떤 셀도 실거래·승급 근거로 사용 영구 금지.
목적: 실행 파이프라인 검증·비용 침식 관찰·사후선택 함정의 라이브 시연.

설계 원칙 (터틀 전례):
- 백테스트와 라이브가 같은 step()을 쓴다 — 닫힌 1h봉 1개(전 심볼)가 유일한 입력 단위.
- 실행 교정 (명세 §3, 전 셀 일괄):
  1. BRK 스탑·수량 산정은 ATR[i-1] (이번 봉으로 갱신되기 전 값).
  2. TR은 previous close 기준 (swing.py 버그 재발 금지).
  3. MR·RSI-DIV 신호는 확정봉 종가 계산, 체결은 다음 봉 시가 (대기주문).
     BRK는 봉내 스탑주문 모델 (돌파레벨 vs 시가 중 불리한 쪽 + 갭 악화) 유지.
  4. 4h 봉은 UTC [00,04,08,..) 정렬, 닫힌 1h봉 4개 전부 있어야 확정.
  5. 결측 봉은 해당 셀 무행동 + 경고 (보간 금지, fail-closed).
  6. 워밍업은 T0 이전 봉으로 지표만 — T0 이전 주문 생성 금지.
  7. 펀딩은 실제 정산 타임스탬프(봉 종가 시각과 일치)에 보유 포지션에 적용.
  8. 체결 이벤트 유일키 (cell, sym, strategy, bar_close, action).
- 일손실 -5%는 '당일 신규 진입 정지' 트리거일 뿐 손실 상한이 아니다 (청산하지 않음).

명세 '미결 사항' 확정 (T0 코드 동결 전 결정 — 결과 조회 전이므로 사후선택 아님):
- (#15) RSI-DIV 스탑/목표 청산은 원본대로 **레벨 그대로 체결** (갭 악화 없음).
  갭 악화는 §3이 명시한 BRK(및 MR 스탑, scalp_grid 원전)에만 적용.
- (#17) 다음 봉 시가 체결 직후 **체결봉부터** 스탑/목표를 검사하며, 같은 봉
  동시 도달 시 스탑 우선(비관 — 터틀 same_day_stop 전례). 이후 매 1h봉 검사
  (대기 스탑/목표 주문은 상시 유효하므로 1h 단위가 실행 계약상 자연스럽다).
- (#13) 2R 목표 확정식: tgt = fill + 2·(fill − stop) = 3·fill − 2·stop
  (R = |fill − stop|, 롱은 진입 위·숏은 진입 아래 — 롱/숏 공통식).

봉 내 사건 순서 (인과 규약): 대기청산(시가) → 대기진입(시가, 마크는 직전 종가 —
시가 시점에 이번 봉 종가는 미지) → 봉내 청산/관리 → 신호·봉내 진입(마크는 직전
종가) → 펀딩(봉 종가 정산) → 일손실 판정. 시가 체결이 봉내 사건보다 먼저다.

변형 셀 E11·E12 — BRK24TP (2026-08-28 사전 고정, 성과 조회 전 동결):
- 지위: 공식 판정(lab/tracke_null.py 10셀 공동 null) 대상 아님. 라벨(동결):
  "빠른 익절 변형 · 미검증 · 판정 권한 없음". E11=바스켓 A, E12=동결 바스켓 B 재사용.
- 규칙: BRK24 와 진입·초기 스탑·사이징 완전 동일 (같은 내부 코드 경로 공유).
  추가 규칙 둘뿐:
  (a) 익절 1R: tgt = fill + 1×(fill − stop) = 2·fill − stop (롱/숏 공통식).
      봉내 레벨 체결 (갭이 유리해도 레벨 — RSI-DIV #15 관례), 체결봉부터 검사
      (같은 봉 same_bar_stop/same_bar_target — RSI-DIV #17 관례), 같은 봉에서
      BRK 청산 레벨(스탑·역채널, 갭 악화)과 동시 도달 시 BRK 청산 우선(비관).
  (b) 최대 보유 12×1h봉 (진입봉=1, 존재하는 봉만 카운트 — 결측 봉은 카운트·청산
      모두 정지, 엔진 공통 fail-closed 관례): hold 12 도달 봉 **종가** 청산
      (action "timeout"). 그 외 기존 BRK24 청산 규칙(역채널 추적·갭 악화) 유지.
- 청산 우선순위 동결: BRK 스탑/역채널 → 1R 목표 → 12봉 타임아웃.
- 청산이 발생한 봉의 종가 시각 펀딩 정산은 이미 청산된 포지션에 적용되지 않는다
  (봉 내 사건 순서상 관리가 펀딩보다 먼저 — 전 전략 공통 기존 관례, 변형 동일).
- 상태: 본 상태 JSON(tracke_state.json)의 variant_cells 키. t0_variant = 변형
  서브상태의 최초 원자적 기록에 성공한 러너 실행 시각 (write-once, 이후 불변).
  지표는 초기화 시 본 팜 '시장 전용 지표 상태'의 깊은 스냅숏을 상속하고
  last_ts 를 본 팜과 정렬한다 — t0_variant 이전 봉 재생·주문 생성이 구조적으로
  불가능 (워밍업 무주문, 기존 T0 원칙 그대로).
- 원장·이력 분리: tracke_variant_ledger.csv / tracke_variant_history.csv 전용.
  본 원장 tracke_ledger.csv 기록 금지 — 공식 로더의 미지 셀 거부 계약 보호.
- E01~E10 불변 보증: 본 셀은 step()/CELLS 경로만 사용하며 변형은 step_variant()/
  VCELLS 별도 상태로만 돈다 — 동일 합성 시퀀스에서 본 원장 바이트 동일
  (tests/test_scalp_farm.py 회귀 테스트).

변형 셀 E13~E18 — 2026-08-28 사전 고정, 성과 조회 전 동결 (V2CELLS 그룹):
- 지위: 공식 판정 비대상·판정 권한 없음 (V2LABELS 동결 문구). 원전은
  lab/confluence_gate_test.py(E13/E14)·lab/published_systems_test.py(E15~E18)의
  확정 구현 1:1 이식. 상태는 본 상태 JSON 의 variant2_cells 키 (E11/E12 의
  variant_cells 와 완전 분리), t0_variant2 = 변형2 서브상태의 최초 원자적 기록
  시각 (그룹별 write-once, 기존 t0_variant 불변·E11/E12 무영향).
- E13/E14 = BRK24GATE (바스켓 A/B): BRK24 와 진입·스탑·사이징·청산 완전 동일
  (같은 코드 경로) — 단 진입 허용 = 3중 게이트 AND (전부 확정봉 [i-1] 값,
  NaN/미형성 = 차단 fail-closed, _gate_ok):
  ① 추세: close[i-1] vs SMA200(1h)[i-1] (롱 >, 숏 <)
  ② 모멘텀: Wilder RSI14[i-1] vs 50 (롱 >, 숏 <)
  ③ 거래량: vol[i-1] > mean(vol[i-21..i-2]) (rolling(20).mean().shift(2) 구조)
  이중 돌파봉(상하 동시)은 롱 우선 해석 후 게이트 (confluence_gate_test 확정).
- E15/E16 = BBMR (바스켓 A/B): BB(20, 2σ, ddof=0 모표준편차, 중심 SMA20).
  확정봉 종가 < 하단밴드 → 다음 봉 시가 롱 진입 (U1 실행 규약), 청산 = 확정봉
  종가 >= SMA20 → 다음 봉 시가. 롱 온리, 스탑 없음 (출판 충실).
- E17/E18 = RSI2 (바스켓 A/B): Connors 원전 — 롱: close>SMA200 ∧ RSI(2)<5 →
  다음 봉 시가 진입, 청산 close>SMA5 → 다음 봉 시가. 숏 대칭 (close<SMA200 ∧
  RSI(2)>95, 청산 close<SMA5). Wilder 평활 (published_systems wilder_rsi 의
  dn==0 정의 포함: 상승만 100, 무변동 50). 스탑 없음.
- 사이징 (E15~E18 공통, 스탑 부재): 포지션 명목 = equity × 1/3 (3슬롯 균등,
  레버리지 없음) — 2% 스탑거리 역산 사이징 미적용 (_try_open notional 모드).
  일손실 -5% 진입정지·비용 왕복 16bp·펀딩·MAX_POS 3·그로스 캡은 공통 그대로.
- heat 정의 (스탑 없는 포지션): heat 기여 = 명목 × |일손실 한도|(5%) —
  risk_d = fill × V2_HEAT_FRAC 로 동결. 슬롯당 equity 의 1/3 × 5% ≈ 1.667% →
  3슬롯 만재 시 5% ≤ HEAT_CAP 6% (설계 슬롯은 구조적으로 허용, 드로다운 뒤
  잔존 heat 가 캡을 넘기면 신규 진입 차단). gross 는 통상대로 |u|×마크 산입.
  주의: 이 heat 정의는 손실 상한이 아니다 — 스탑이 없어 포지션 최대 손실은
  명목 전체이며, heat 는 진입 차단용 대리변수일 뿐이다 (실질 손실 제동은
  명목 1/3 고정·MAX_POS 3·일손실 -5% 진입정지가 담당 — Codex 검토 명기).
- 실행 규약과 원전의 명시적 차이 (동결):
  (i) 같은 봉 청산 심볼 재진입 금지 — 팜 공통 위상 규약이 lab 의 '청산 체결봉
      종가 재신호 허용'에 우선한다 (재진입 신호가 lab 대비 최대 1봉 지연).
  (ii) 체결봉 종가는 확정봉이므로 BBMR·RSI2 의 '청산 신호' 평가에 포함한다
      (_step_cells 3단계 예외) — lab 과 1:1, 청산 체결은 다음 봉 시가라 인과 무결.
  (iii) 지표 이력은 관측된 봉만 축적 (결측 봉 보간 없음 — 엔진 공통 관례;
      lab 동결 데이터는 내부 갭이 없어 실질 동일).
- 지표: BRK 핵심(ATR·채널)은 본 팜 스냅숏 상속 (E11 전례). 본 팜이 추적하지
  않는 확장 지표(x2: 200종가·21거래량·RSI14·RSI2 평활)는 러너가 t0_variant2
  동결 전에 warmup_x2 로 과거 봉을 재생해 채운다 (수집 실패 시 동결 지연,
  fail-closed — 워밍업 중·미형성 지표는 전부 차단이며 주문 없음).
- 원장·이력: E11/E12 와 같은 분리 파일(tracke_variant_ledger/history.csv)에
  통합 — 유일키에 cell 이 있어 행 충돌 없음. 이력은 e13~e18 열 추가(keep-last
  관례 유지), equity 열 = 행 기록 시점의 변형 전 셀 시가평가 합. 본 원장
  (tracke_ledger.csv) 기록 금지 방화벽은 그룹별 저장 경계에서 강제.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# --- 명세 동결 상수 (config 아님 — 사전등록 수치, 변경 금지) ---
H1 = 3_600_000                 # 1h (ms)
H4 = 4 * H1
CAPITAL0 = 10_000.0            # 셀당 가상 자본
COST_SIDE = 0.0008             # 편도 8bp (taker 6 + 슬립 2) — 왕복 16bp
RISK = 0.02                    # 포지션당 리스크 (스탑거리 역산 사이징)
MAX_POS = 3                    # 셀당 동시 최대 포지션
GROSS_CAP = 10.0               # 총명목/자본 상한
HEAT_CAP = 0.06                # 포트폴리오 heat 상한 (신규 진입 차단 기준)
DAILY_HALT = -0.05             # 일손실 트리거 (진입 정지, 손실 상한 아님)
ATR1H_N = 24                   # 1h ATR (BRK/MR 공용, scalp_grid 원전)
BRK_ATR_MULT = 6.0
MR_SMA_N = 24
MR_Z = 2.0
MR_ATR_MULT = 4.0
MR_MAX_HOLD = 24               # 최대 보유 24h
RSI_N = 14                     # RSI-DIV 4h (불단왕 규칙, rsi_divergence_test 원전)
ATR4H_N = 14
PIVOT_K = 2                    # 피벗 좌우 봉
DIV_LOOKBACK = 50              # 피벗 간 최대 4h봉 간격
DIV_TIMEOUT = 42               # 타임아웃 42봉(7일)
DIV_STOP_ATR = 0.5
DIV_TGT_R = 2.0
BASKET_A = ("BTC", "ETH", "SOL")
WARMUP_1H = 420                # 워밍업 조회 깊이 (완전한 4h 창 최소 104개 확보)

# --- 변형 셀(E11·E12) 동결 상수 — 본 판정 비대상, 규칙은 모듈 docstring ---
VAR_TP_R = 1.0                 # BRK24TP 익절 배수 (1R)
VAR_MAX_HOLD = 12              # 최대 보유 1h봉 수 (진입봉 = 1)

# --- 변형2 셀(E13~E18) 동결 상수 — 원전: lab/confluence_gate_test.py(게이트),
#     lab/published_systems_test.py(BBMR·RSI2). 규칙은 모듈 docstring ---
GATE_SMA_N = 200               # 게이트 ① 추세 SMA (1h)
GATE_RSI_N = 14                # 게이트 ② Wilder RSI
GATE_VOL_N = 20                # 게이트 ③ 거래량 평균 창 (vol[i-1] vs 직전 20봉)
BB_N = 20                      # BBMR 밴드 길이 (중심 SMA20)
BB_K = 2.0                     # BBMR 밴드 폭 (2σ, ddof=0 모표준편차)
RSI2_N = 2                     # Connors RSI 길이
RSI2_LO = 5.0                  # 롱 진입 임계 (원전 수치 — 10/90 완화 변형 아님)
RSI2_HI = 95.0                 # 숏 진입 임계
RSI2_EXIT_N = 5                # 청산 SMA 길이
RSI2_TREND_N = 200             # 레짐 필터 SMA 길이 (게이트 ①과 동수·별도 출처)
V2_NOTIONAL_FRAC = 1.0 / 3.0   # 스탑 없는 셀(BBMR·RSI2) 명목 = equity × 1/3
V2_HEAT_FRAC = -DAILY_HALT     # 스탑 부재 heat 기여율 = 일손실 한도 크기 (5%)


@dataclass(frozen=True)
class CellSpec:
    """셀 정의 (T0 동결)."""

    cell: str
    strategy: str      # "BRK24"|"BRK48"|"BRK96"|"MR"|"RSIDIV"
    basket: str        # "A"|"B"
    n: int             # BRK 채널 길이 (BRK 외 0)


CELLS: tuple[CellSpec, ...] = (
    CellSpec("E01", "BRK24", "A", 24), CellSpec("E02", "BRK24", "B", 24),
    CellSpec("E03", "BRK48", "A", 48), CellSpec("E04", "BRK48", "B", 48),
    CellSpec("E05", "BRK96", "A", 96), CellSpec("E06", "BRK96", "B", 96),
    CellSpec("E07", "MR", "A", 0), CellSpec("E08", "MR", "B", 0),
    CellSpec("E09", "RSIDIV", "A", 0), CellSpec("E10", "RSIDIV", "B", 0),
)

# 변형 셀 — 본 CELLS 와 분리 (공식 판정·대시보드의 고정 10셀 계약 불변)
VCELLS: tuple[CellSpec, ...] = (
    CellSpec("E11", "BRK24TP", "A", 24), CellSpec("E12", "BRK24TP", "B", 24),
)

# 셀별 고정 라벨 (대시보드 표기용 — 성과와 무관, 변경 금지)
LABELS: dict = {
    "E01": "역사적 탈락", "E02": "역사적 탈락 · OOD",
    "E03": "역사적 탈락", "E04": "역사적 탈락 · OOD",
    "E05": "격자 1/8 선택 · Track D 중복 · 선택할인", "E06": "격자 1/8 · OOD",
    "E07": "역사적 탈락", "E08": "역사적 탈락 · OOD",
    "E09": "미검증 가설 U1", "E10": "미검증 가설 U1 · OOD",
}

# 변형 셀 고정 라벨 (동결 문구 — 성과와 무관, 변경 금지)
VLABELS: dict = {
    "E11": "빠른 익절 변형 · 미검증 · 판정 권한 없음",
    "E12": "빠른 익절 변형 · 미검증 · 판정 권한 없음",
}

# 변형2 셀(E13~E18) — VCELLS 와도 분리된 그룹 (t0_variant2 별도 write-once)
V2CELLS: tuple[CellSpec, ...] = (
    CellSpec("E13", "BRK24GATE", "A", 24), CellSpec("E14", "BRK24GATE", "B", 24),
    CellSpec("E15", "BBMR", "A", 0), CellSpec("E16", "BBMR", "B", 0),
    CellSpec("E17", "RSI2", "A", 0), CellSpec("E18", "RSI2", "B", 0),
)

# 변형2 셀 고정 라벨 (동결 문구 — 성과와 무관, 변경 금지)
V2LABELS: dict = {
    "E13": "컨플루언스 게이트 변형 · 미검증 · 판정 권한 없음",
    "E14": "컨플루언스 게이트 변형 · 미검증 · 판정 권한 없음",
    "E15": "볼린저 평균회귀 (출판) · 미검증 · 판정 권한 없음",
    "E16": "볼린저 평균회귀 (출판) · 미검증 · 판정 권한 없음",
    "E17": "Connors RSI2 (출판) · 미검증 · 판정 권한 없음",
    "E18": "Connors RSI2 (출판) · 미검증 · 판정 권한 없음",
}


@dataclass(frozen=True)
class BarE:
    """닫힌 1h 봉 (ts = 봉 시작 시각 ms, 종가 시각은 ts + H1).

    vol: 거래량 — 변형2 게이트 ③ 전용 (본 셀·E11/E12 는 사용하지 않음).
        OHLC NaN 결측 검사에 불포함, NaN 이면 게이트만 차단 (fail-closed).
    """

    ts: int
    open: float
    high: float
    low: float
    close: float
    vol: float = float("nan")


@dataclass
class FarmPos:
    """한 셀·한 심볼의 포지션."""

    d: int                 # +1 롱 / -1 숏
    u: float               # 수량
    e: float               # 진입가
    stop: float            # 스탑 레벨
    kind: str              # "BRK"|"MR"|"RSIDIV"
    tgt: float = 0.0       # RSIDIV 목표가 (2R)
    hold: int = 0          # MR 보유 1h봉 수 (진입봉 = 1)
    n4_entry: int = 0      # RSIDIV 신호 4h봉 인덱스
    risk_d: float = 0.0    # 진입 시 단위당 스탑거리 (heat 계산)
    pending_exit: str = "" # 다음 봉 시가 청산 사유 (""=없음)


@dataclass
class CellState:
    """가상계정(셀) 상태 — JSON 직렬화 가능."""

    equity: float = CAPITAL0       # 현금 자본 (실현 기준)
    positions: dict = field(default_factory=dict)   # sym -> FarmPos
    pending: dict = field(default_factory=dict)     # sym -> 대기 진입 주문 dict
    day: str = ""
    day_eq: float = CAPITAL0
    halted: bool = False
    halts: int = 0                 # 일손실 트리거 누적 횟수
    cost: float = 0.0              # 수수료+슬립 누적
    fund: float = 0.0              # 펀딩 누적 (양수 = 지불)
    turnover: float = 0.0          # 체결 명목 누적

    def to_dict(self, px: dict | None = None) -> dict:
        """JSON 직렬화.

        직렬화 계약: 'gross' = sum(|u| x 마지막 유효 종가) / equity —
        봉 종가 기준 명목 레버리지 (포지션 없거나 equity<=0 이면 0.0,
        가격 이력 없는 심볼은 진입가 마크). 대시보드 로더가 이 키를 읽는다
        (읽기 전용 파생값 — from_dict 는 무시한다).

        Args:
            px: sym -> 마지막 유효 종가 마크맵 (FarmState.to_dict 가 주입).
        """
        px = px or {}
        gross = 0.0
        if self.positions and self.equity > 0:
            gross = round(sum(abs(p.u) * px.get(s, p.e)
                              for s, p in self.positions.items()) / self.equity, 8)
        return dict(equity=self.equity, gross=gross, day=self.day,
                    day_eq=self.day_eq, halted=self.halted, halts=self.halts,
                    cost=self.cost, fund=self.fund, turnover=self.turnover,
                    pending=dict(self.pending),
                    positions={s: asdict(p) for s, p in self.positions.items()})

    @classmethod
    def from_dict(cls, d: dict) -> "CellState":
        """JSON 역직렬화."""
        st = cls(equity=d["equity"], day=d.get("day", ""),
                 day_eq=d.get("day_eq", d["equity"]), halted=d.get("halted", False),
                 halts=d.get("halts", 0), cost=d.get("cost", 0.0),
                 fund=d.get("fund", 0.0), turnover=d.get("turnover", 0.0))
        st.pending = dict(d.get("pending", {}))
        st.positions = {s: FarmPos(**p) for s, p in d.get("positions", {}).items()}
        return st


def _new_ind() -> dict:
    """심볼별 지표 상태 초기값 (시장 데이터의 순수 함수 — 셀 간 공유)."""
    return {
        "atr1": None,      # 1h ATR(24) EMA — 이번 봉 처리 전 값이 ATR[i-1]
        "pc": None,        # 직전 1h 종가 (TR previous close)
        "hl": [],          # [[high, low], ...] 최근 96봉 (채널은 현재 봉 제외)
        "cl": [],          # [close, ...] 최근 24봉 (MR SMA/SD, 현재 봉 제외)
        "h4": {"w": None, "o": 0.0, "h": 0.0, "l": 0.0, "c": 0.0, "n": 0,
               "pc4": None, "atr4": None, "up": None, "dn": None,
               "b5": [], "n4": 0, "plo": [], "phi": []},
    }


def _new_x2() -> dict:
    """변형2(E13~E18) 확장 지표 초기값 — 본 팜·E11/E12 의 ind 에는 없는 키('x2').

    이 키의 존재 자체가 갱신 스위치다 (_update_1h): 본 상태에는 절대 생기지
    않으므로 본 경로·기존 변형 경로의 상태 바이트가 변하지 않는다.
    """
    return {"c2": [],       # 최근 200 종가 (현재 봉 제외 — 봉 처리 후 갱신)
            "v2": [],       # 최근 21 거래량 (NaN 허용 — 게이트가 차단)
            "u14": None, "d14": None,   # 게이트 RSI14 Wilder 평활
            "u2": None, "d2": None}     # Connors RSI(2) Wilder 평활


def _update_x2(x2: dict, close: float, vol: float) -> None:
    """확장 지표 1봉 반영 (봉 처리 '후' 호출 — 다음 봉에서 [i-1] 값이 된다).

    RSI 평활은 첫 diff 시드 — pandas ewm(alpha=1/n, adjust=False)의
    선행 NaN 스킵과 동치 (confluence_gate_test·published_systems 원전 동형).
    """
    if x2["c2"]:
        diff = close - x2["c2"][-1]
        ux, dx = max(diff, 0.0), max(-diff, 0.0)
        x2["u14"] = ux if x2["u14"] is None else \
            x2["u14"] + (ux - x2["u14"]) / GATE_RSI_N
        x2["d14"] = dx if x2["d14"] is None else \
            x2["d14"] + (dx - x2["d14"]) / GATE_RSI_N
        x2["u2"] = ux if x2["u2"] is None else x2["u2"] + (ux - x2["u2"]) / RSI2_N
        x2["d2"] = dx if x2["d2"] is None else x2["d2"] + (dx - x2["d2"]) / RSI2_N
    x2["c2"].append(close)
    del x2["c2"][:-max(GATE_SMA_N, RSI2_TREND_N)]
    x2["v2"].append(vol)
    del x2["v2"][:-(GATE_VOL_N + 1)]


def warmup_x2(v: FarmState, sym: str, rows: list) -> None:
    """변형2 확장 지표 워밍업 — (close, vol) 시퀀스를 시간순 반영 (주문 없음).

    t0_variant2 동결 전 1회 전용: 본 팜 스냅숏(new_variant2)에 없는 게이트·
    밴드·RSI 이력을 v.last_ts 이전 봉으로만 채운다. NaN 종가 봉은 결측 관례
    (엔진 공통 fail-closed)로 건너뛴다.

    Args:
        v: 변형2 서브상태 (new_variant2 직후).
        sym: 심볼.
        rows: [(close, vol), ...] 시간 오름차순.
    """
    x2 = v.ind.setdefault(sym, _new_ind()).setdefault("x2", _new_x2())
    for close, vol in rows:
        if math.isnan(close):
            continue
        _update_x2(x2, close, vol)


@dataclass
class FarmState:
    """팜 전체 상태 — JSON 직렬화 가능 (라이브 러너가 보존)."""

    t0: int = 0                     # 첫 러너 실행 시각 ms (이전 봉은 워밍업 전용)
    last_ts: int = 0                # 마지막 처리 봉 시작 ts (멱등 기준)
    basket_b: list = field(default_factory=list)    # T0 동결 (교체 없음)
    delisted: list = field(default_factory=list)    # 영구 공석 슬롯
    ind: dict = field(default_factory=dict)         # sym -> 지표 상태
    cells: dict = field(default_factory=dict)       # cell_id -> CellState
    # 변형(E11·E12) 서브상태 — 본 경로는 절대 읽지 않는 불투명 dict.
    # None = 미초기화(초기화 가능), dict = variant_to_dict() 산출물.
    # {} 등 손상값은 variant_from_dict 가 fail-closed (재초기화로 t0 이동 금지).
    variant_cells: dict | None = None
    # 변형2(E13~E18) 서브상태 — 위와 같은 계약, 키·t0(t0_variant2)만 분리.
    variant2_cells: dict | None = None

    def to_dict(self) -> dict:
        """JSON 직렬화 (셀 gross 계산용 마지막 유효 종가 마크맵 주입)."""
        px = {s: i_["pc"] for s, i_ in self.ind.items()
              if i_.get("pc") is not None}
        return dict(t0=self.t0, last_ts=self.last_ts, basket_b=list(self.basket_b),
                    delisted=list(self.delisted), ind=self.ind,
                    cells={c: s.to_dict(px) for c, s in self.cells.items()},
                    variant_cells=self.variant_cells,
                    variant2_cells=self.variant2_cells)

    @classmethod
    def from_dict(cls, d: dict) -> "FarmState":
        """JSON 역직렬화."""
        st = cls(t0=d.get("t0", 0), last_ts=d.get("last_ts", 0),
                 basket_b=list(d.get("basket_b", [])),
                 delisted=list(d.get("delisted", [])), ind=dict(d.get("ind", {})),
                 variant_cells=d.get("variant_cells"),
                 variant2_cells=d.get("variant2_cells"))
        st.cells = {c: CellState.from_dict(s) for c, s in d.get("cells", {}).items()}
        return st


def new_farm(basket_b: list, t0: int) -> FarmState:
    """T0 초기화 — 바스켓 B 동결, 셀 10개 생성."""
    return FarmState(t0=t0, basket_b=list(basket_b),
                     cells={spec.cell: CellState() for spec in CELLS})


def new_variant(state: FarmState, t0: int) -> FarmState:
    """변형 서브팜(E11·E12) 초기화 — t0_variant 동결 시점에 1회 호출 (write-once).

    본 팜의 동결 바스켓 B·폐지 목록을 재사용하고, 지표는 본 팜 '시장 전용 지표
    상태'의 깊은 스냅숏을 상속한다 (_new_ind 주석대로 시장 데이터의 순수 함수 —
    셀 간 공유 가능. 이후 갱신은 완전 분리). last_ts 를 본 팜과 정렬해
    t0_variant 이전 봉 재생(과거 주문 생성)을 구조적으로 차단한다
    (워밍업 무주문 — 기존 T0 원칙 그대로).

    Args:
        state: 본 팜 상태 (변형되지 않음).
        t0: t0_variant (epoch ms) — 기록 후 불변.
    """
    return FarmState(t0=t0, last_ts=state.last_ts,
                     basket_b=list(state.basket_b),
                     delisted=list(state.delisted),
                     ind=copy.deepcopy(state.ind),
                     cells={spec.cell: CellState() for spec in VCELLS})


def new_variant2(state: FarmState, t0: int) -> FarmState:
    """변형2 서브팜(E13~E18) 초기화 — t0_variant2 동결 시점에 1회 (write-once).

    본 팜 지표 스냅숏 상속은 E11 전례(new_variant)와 동일하되, 본 팜이 추적하지
    않는 확장 지표(x2)는 빈 상태로 만든다 — 러너가 t0 동결 전 warmup_x2 로
    과거 봉을 재생해 채운다 (미형성 동안 게이트·밴드·RSI 신호는 전부 차단).

    Args:
        state: 본 팜 상태 (변형되지 않음).
        t0: t0_variant2 (epoch ms) — 기록 후 불변.
    """
    v = FarmState(t0=t0, last_ts=state.last_ts,
                  basket_b=list(state.basket_b),
                  delisted=list(state.delisted),
                  ind=copy.deepcopy(state.ind),
                  cells={spec.cell: CellState() for spec in V2CELLS})
    for i_ in v.ind.values():
        i_.setdefault("x2", _new_x2())
    return v


def _vgroup_to_dict(v: FarmState, key: str) -> dict:
    """변형 그룹 직렬화 공용 — t0 를 그룹 키로 개명, 서브상태 중첩 제거."""
    d = v.to_dict()
    d.pop("variant_cells", None)          # 중첩 없음 (변형 안의 변형 금지)
    d.pop("variant2_cells", None)
    d[key] = d.pop("t0")
    return d


def variant_to_dict(v: FarmState) -> dict:
    """변형 서브상태 직렬화 — 상태 JSON 'variant_cells' 키 계약 (t0_variant 명명)."""
    return _vgroup_to_dict(v, "t0_variant")


def variant2_to_dict(v: FarmState) -> dict:
    """변형2 서브상태 직렬화 — 'variant2_cells' 키 계약 (t0_variant2 명명)."""
    return _vgroup_to_dict(v, "t0_variant2")


def _vgroup_from_dict(d: dict | None, key: str, cells: tuple) -> FarmState:
    """변형 그룹 역직렬화 공용 — 손상 시 fail-closed.

    부재(None)와 손상({}·키 누락·셀 구성 오류)을 구분한다: 부재만 초기화
    대상이고, 손상은 예외다 — 조용한 재초기화로 그룹 t0 가 이동하는 것 금지.

    Raises:
        ValueError: 그룹 t0 부재/0 또는 셀 구성이 그룹 정의와 다를 때.
    """
    t0 = d.get(key) if isinstance(d, dict) else None
    if not isinstance(t0, (int, float)) or isinstance(t0, bool) or t0 <= 0:
        raise ValueError(
            f"변형 서브상태 손상 — {key} 부재/비정상 (재초기화 금지): {t0!r}")
    want = {spec.cell for spec in cells}
    have = set(d.get("cells", {}))
    if have != want:
        raise ValueError(f"변형 셀 구성 불일치: {sorted(have)} != {sorted(want)}")
    d = copy.deepcopy(d)                  # 저장된 dict 와의 중첩 별칭 차단
    d["t0"] = d.pop(key)
    return FarmState.from_dict(d)


def variant_from_dict(d: dict | None) -> FarmState:
    """'variant_cells' 키 역직렬화 — 손상 시 fail-closed (E11·E12)."""
    return _vgroup_from_dict(d, "t0_variant", VCELLS)


def variant2_from_dict(d: dict | None) -> FarmState:
    """'variant2_cells' 키 역직렬화 — 손상 시 fail-closed (E13~E18)."""
    return _vgroup_from_dict(d, "t0_variant2", V2CELLS)


def cell_syms(spec: CellSpec, state: FarmState) -> tuple:
    """셀의 바스켓 심볼."""
    return BASKET_A if spec.basket == "A" else tuple(state.basket_b)


def _cell_mtm(cell: CellState, px: dict) -> float:
    """셀 시가평가 (가격 없는 심볼은 기여 생략 — 보간 금지)."""
    v = cell.equity
    for s, p in cell.positions.items():
        if s in px:
            v += p.u * (px[s] - p.e) * p.d
    return v


def _equities(state: FarmState, cells: tuple) -> dict:
    """셀별 시가평가 (마지막 유효 종가 기준) — 내부 공용 구현."""
    px = {s: i_["pc"] for s, i_ in state.ind.items() if i_.get("pc") is not None}
    return {spec.cell: _cell_mtm(state.cells[spec.cell], px) for spec in cells}


def farm_equities(state: FarmState) -> dict:
    """본 셀(E01~E10) 시가평가 (마지막 유효 종가 기준)."""
    return _equities(state, CELLS)


def variant_equities(v: FarmState) -> dict:
    """변형 셀(E11·E12) 시가평가 — 변형 서브상태 전용."""
    return _equities(v, VCELLS)


def variant2_equities(v: FarmState) -> dict:
    """변형2 셀(E13~E18) 시가평가 — 변형2 서브상태 전용."""
    return _equities(v, V2CELLS)


def _ev(spec: CellSpec, s: str, ts_close: int, action: str, price: float,
        qty: float, pnl: float, cost: float, d: int, fund: float = 0.0) -> dict:
    """원장 이벤트 — 유일키 (cell, sym, strategy, bar_close, action).

    funding 열은 상시 존재한다 (0.0 기본) — 판정(tracke_null) 공식 모드가
    '펀딩 기록 부재'로 원장을 기각하지 않도록 하는 계약. 양수 = 수취.
    """
    return dict(cell=spec.cell, sym=s, strategy=spec.strategy, bar_close=ts_close,
                action=action, price=price, qty=qty, pnl=pnl, cost=cost,
                direction=d, funding=fund)


def _close(cell: CellState, spec: CellSpec, s: str, ts_close: int, px: float,
           action: str, fills: list) -> None:
    """포지션 청산 (편도 비용 차감)."""
    p = cell.positions.pop(s)
    pnl = p.u * (px - p.e) * p.d
    cost = p.u * px * COST_SIDE
    cell.equity += pnl - cost
    cell.cost += cost
    cell.turnover += p.u * px
    fills.append(_ev(spec, s, ts_close, action, px, p.u, pnl, cost, p.d))


def _try_open(cell: CellState, spec: CellSpec, s: str, b: BarE, d: int, fill: float,
              stop: float, kind: str, px: dict, fills: list,
              tgt: float = 0.0, n4_entry: int = 0, notional: bool = False) -> bool:
    """진입 시도 — 리스크 역산 사이징 + 그로스/heat 캡 + 같은 봉 스탑 비관 처리.

    Args:
        px: 그로스 마크용 가격 — 직전 봉 종가에 **이번 봉 확정 체결가를 덮어쓴**
            셀 로컬 마크맵 (체결 시점에 이번 봉 종가는 미지 — 인과 규약).
            체결 성공 시 이 맵의 s 마크를 체결가로 갱신한다. 없는 심볼은 진입가.
        notional: 스탑 없는 셀(BBMR·RSI2)의 명목 사이징 모드 — 수량 =
            equity × 1/3 / fill (2% 스탑 역산 미적용, stop 인자 무시=0.0),
            heat 기여 단가 = fill × V2_HEAT_FRAC (risk_d 로 동결, 명목 기준 정의).

    Returns:
        진입 체결이 발생했으면 True. 같은 봉 스탑/목표 비관 검사는 호출자가
        시가 동시 체결 확정 후 _post_fill_check()로 수행한다.
    """
    ts_close = b.ts + H1
    if cell.halted or len(cell.positions) >= MAX_POS:
        return False
    if notional:
        dist = fill * V2_HEAT_FRAC     # 스탑 부재 heat 기여 = 명목 × 5% (동결 정의)
    else:
        dist = (fill - stop) * d
        if dist <= 0:
            logger.warning("%s %s 진입 스킵 — 체결가가 스탑 반대편 (fail-closed)",
                           spec.cell, s)
            return False
    eq = cell.equity
    if eq <= 0 or fill <= 0:
        return False
    gross = sum(pp.u * px.get(ss, pp.e) for ss, pp in cell.positions.items())
    if gross >= GROSS_CAP * eq:
        return False
    want = V2_NOTIONAL_FRAC * eq / fill if notional else RISK * eq / dist
    u = min(want, max(0.0, GROSS_CAP * eq - gross) / fill)
    if u <= 0:
        return False
    heat = sum(pp.risk_d * pp.u for pp in cell.positions.values())
    if heat + u * dist > HEAT_CAP * eq * (1 + 1e-9):
        logger.info("%s %s 진입 차단 — heat 캡 %.0f%%", spec.cell, s, HEAT_CAP * 100)
        return False
    cost = u * fill * COST_SIDE
    cell.equity -= cost
    cell.cost += cost
    cell.turnover += u * fill
    fills.append(_ev(spec, s, ts_close, "enter", fill, u, 0.0, cost, d))
    cell.positions[s] = FarmPos(d=d, u=u, e=fill, stop=stop, kind=kind, tgt=tgt,
                                hold=1, n4_entry=n4_entry, risk_d=dist)
    px[s] = fill            # 이후 같은 봉 진입 사이징은 실제 체결가로 마크
    return True


def _post_fill_check(cell: CellState, spec: CellSpec, s: str, b: BarE,
                     fills: list) -> None:
    """방금 체결된 포지션의 같은 봉 스탑(우선)·목표 비관 검사 (터틀 전례)."""
    p = cell.positions.get(s)
    if p is None:
        return
    if p.kind in ("BBMR", "RSI2"):
        return          # 스탑·목표 없는 종류 — stop=0.0 센티널 오검(숏 즉시청산) 방지
    ts_close = b.ts + H1
    if (p.d > 0 and b.low <= p.stop) or (p.d < 0 and b.high >= p.stop):
        _close(cell, spec, s, ts_close, p.stop, "same_bar_stop", fills)
        return
    # 목표 레벨 보유 종류(RSIDIV 2R, 변형 BRKTP 1R)만 같은 봉 목표 검사
    if p.kind in ("RSIDIV", "BRKTP") and ((p.d > 0 and b.high >= p.tgt)
                                          or (p.d < 0 and b.low <= p.tgt)):
        _close(cell, spec, s, ts_close, p.tgt, "same_bar_target", fills)


def _update_1h(ind: dict, b: BarE) -> None:
    """1h 지표 갱신 — 봉 처리 '후'에 호출되어 다음 봉에서 ATR[i-1]·shift(1) 채널이 된다.

    확장 지표(x2)는 키가 있을 때만 갱신 — 변형2 서브상태 전용 스위치라
    본 팜·E11/E12 상태는 바이트 단위로 불변이다.
    """
    pc = ind["pc"]
    tr = b.high - b.low if pc is None else max(b.high - b.low,
                                               abs(b.high - pc), abs(b.low - pc))
    a0 = ind["atr1"]
    ind["atr1"] = tr if a0 is None else a0 + (tr - a0) / ATR1H_N
    ind["hl"].append([b.high, b.low])
    del ind["hl"][:-96]
    ind["cl"].append(b.close)
    del ind["cl"][:-MR_SMA_N]
    ind["pc"] = b.close
    x2 = ind.get("x2")
    if x2 is not None:
        _update_x2(x2, b.close, b.vol)


def _confirm_4h(h4: dict, o: float, h: float, low: float, c: float):
    """확정된 4h 봉 반영 — ATR(14)·RSI(14)·피벗·다이버전스 신호.

    Returns:
        (방향, 스탑레벨, 신호 4h봉 인덱스) 또는 None.
    """
    pc4 = h4["pc4"]
    tr = h - low if pc4 is None else max(h - low, abs(h - pc4), abs(low - pc4))
    h4["atr4"] = tr if h4["atr4"] is None else h4["atr4"] + (tr - h4["atr4"]) / ATR4H_N
    rsi = float("nan")
    if pc4 is not None:
        diff = c - pc4
        ux, dx = max(diff, 0.0), max(-diff, 0.0)
        h4["up"] = ux if h4["up"] is None else h4["up"] + (ux - h4["up"]) / RSI_N
        h4["dn"] = dx if h4["dn"] is None else h4["dn"] + (dx - h4["dn"]) / RSI_N
        if h4["dn"] > 0:
            rsi = 100.0 - 100.0 / (1.0 + h4["up"] / h4["dn"])
    h4["pc4"] = c
    idx = h4["n4"]
    h4["n4"] = idx + 1
    b5 = h4["b5"]
    b5.append([low, h, rsi])
    del b5[:-(2 * PIVOT_K + 1)]
    if len(b5) < 2 * PIVOT_K + 1:
        return None
    # 피벗 확정: 좌우 K봉 극값 (j = idx - K 위치가 이번 봉에서 확정)
    j = idx - PIVOT_K
    lo_j, hi_j, r_j = b5[PIVOT_K]
    if lo_j == min(x[0] for x in b5):
        h4["plo"].append([j, lo_j, r_j])
        del h4["plo"][:-2]
    if hi_j == max(x[1] for x in b5):
        h4["phi"].append([j, hi_j, r_j])
        del h4["phi"][:-2]
    a4 = h4["atr4"]
    if a4 is None or a4 <= 0:
        return None
    # 다이버전스: 방금 확정된 피벗(j2 == j)과 직전 피벗 비교 (롱 우선 — 원전 순서)
    plo, phi = h4["plo"], h4["phi"]
    if len(plo) == 2:
        (j1, p1, r1), (j2, p2, r2) = plo
        if j2 == j and j2 - j1 <= DIV_LOOKBACK and p2 < p1 and r2 > r1:
            return (1, p2 - DIV_STOP_ATR * a4, idx)
    if len(phi) == 2:
        (j1, p1, r1), (j2, p2, r2) = phi
        if j2 == j and j2 - j1 <= DIV_LOOKBACK and p2 > p1 and r2 < r1:
            return (-1, p2 + DIV_STOP_ATR * a4, idx)
    return None


def _update_4h(h4: dict, b: BarE):
    """1h 봉을 4h 창(UTC [00,04,..))에 누적 — 4봉 전부 있을 때만 확정 (교정 4·5).

    Returns:
        창 확정 시 _confirm_4h 결과, 아니면 None.
    """
    w = b.ts - (b.ts % H4)
    if h4["w"] != w:
        if h4["w"] is not None and h4["n"] not in (0, 4):
            logger.warning("4h 창 %d 미완성(%d/4) — 폐기 (fail-closed)", h4["w"], h4["n"])
        h4["w"], h4["o"], h4["h"], h4["l"], h4["c"], h4["n"] = \
            w, b.open, b.high, b.low, b.close, 1
    else:
        h4["h"] = max(h4["h"], b.high)
        h4["l"] = min(h4["l"], b.low)
        h4["c"] = b.close
        h4["n"] += 1
    if b.ts % H4 == H4 - H1:            # 창의 마지막 1h 슬롯
        n, o, hh, ll, cc = h4["n"], h4["o"], h4["h"], h4["l"], h4["c"]
        h4["w"], h4["n"] = None, 0
        if n == 4:
            return _confirm_4h(h4, o, hh, ll, cc)
        logger.warning("4h 창 미완성(%d/4) — 신호 없음 (fail-closed)", n)
    return None


def _gate_ok(ind: dict, d: int) -> bool:
    """BRK24GATE 3중 게이트 AND — 전부 확정봉 [i-1] 값, NaN/미형성 = 차단.

    원전 confluence_gate_test.gates 1:1 (이 봉 처리 시점의 x2 는 [i-1]까지 갱신):
    ① 추세: close[i-1] vs SMA200[i-1] (롱 >, 숏 <; SMA 는 close[i-1] 포함 200봉)
    ② 모멘텀: Wilder RSI14[i-1] vs 50 (롱 >, 숏 <). dn==0·up>0 은 pandas
       ru/rd=inf → RSI 100 과 동치로 100.0, dn==up==0 은 0/0=NaN → 차단.
    ③ 거래량: vol[i-1] > mean(vol[i-21..i-2]) — rolling(20).mean().shift(2)
       구조 (롱숏 공통, 직전 20봉 평균에서 i-1 제외). NaN 거래량 창 = 차단.

    Args:
        ind: 심볼 지표 상태 (x2 필수 — 없으면 차단).
        d: 돌파 해석 방향 (+1 롱 / -1 숏; 이중 돌파봉은 호출자가 롱 우선 해석).
    """
    x2 = ind.get("x2")
    if x2 is None:
        return False
    c2, v2 = x2["c2"], x2["v2"]
    if len(c2) < GATE_SMA_N or len(v2) < GATE_VOL_N + 1:
        return False                    # 워밍업 (pandas NaN) — 차단 (fail-closed)
    pc = c2[-1]
    sma = float(np.mean(c2[-GATE_SMA_N:]))
    if not (pc > sma if d > 0 else pc < sma):
        return False
    u, dn = x2["u14"], x2["d14"]
    if u is None or dn is None:
        return False
    if dn > 0:
        rsi = 100.0 - 100.0 / (1.0 + u / dn)
    elif u > 0:
        rsi = 100.0                     # pandas ru/0 = inf → RSI 100 동치
    else:
        return False                    # 0/0 = NaN — 차단
    if not (rsi > 50.0 if d > 0 else rsi < 50.0):
        return False
    vm = float(np.mean(v2[-(GATE_VOL_N + 1):-1]))
    return bool(v2[-1] > vm)            # NaN 비교는 False — 차단 (fail-closed)


def _rsi_next(u, dn, diff: float, n: int) -> float:
    """Wilder RSI 1스텝 선행값 — 확정봉 [i] 종가까지 포함한 RSI[i] (상태 불변).

    published_systems.wilder_rsi 의 dn==0 명시 정의 포함: 상승만 100, 무변동 50.

    Args:
        u: [i-1]까지의 up 평활 (None = 이번 diff 가 첫 시드).
        dn: [i-1]까지의 down 평활.
        diff: close[i] - close[i-1].
        n: 평활 길이.
    """
    ux, dx = max(diff, 0.0), max(-diff, 0.0)
    u = ux if u is None else u + (ux - u) / n
    dn = dx if dn is None else dn + (dx - dn) / n
    if dn > 0:
        return 100.0 - 100.0 / (1.0 + u / dn)
    return 100.0 if u > 0 else 50.0


def _fill_pending(cell: CellState, spec: CellSpec, s: str, b: BarE, ind: dict,
                  live: bool, px: dict, fills: list) -> bool:
    """대기 진입 주문을 이번 봉 시가에 체결 시도한다 (교정 3).

    Returns:
        체결(같은 봉 청산 포함) 발생 여부.
    """
    pend = cell.pending.pop(s, None)
    if pend is None or s in cell.positions or not live:
        return False
    if pend.get("ets") != b.ts:
        logger.warning("%s %s 대기주문 취소 — 다음 봉 연속성 붕괴 (fail-closed)",
                       spec.cell, s)
        return False
    if pend["kind"] in ("BBMR", "RSI2"):
        # 스탑 없는 출판 시스템 — 명목 사이징 (equity × 1/3), stop=0.0 센티널
        return _try_open(cell, spec, s, b, pend["d"], b.open, 0.0, pend["kind"],
                         px, fills, notional=True)
    if pend["kind"] == "MR":
        a = ind["atr1"]            # 신호봉까지 갱신된 ATR = 체결봉 직전 봉 ATR
        if a is None or a <= 0:
            return False
        stop = b.open - pend["d"] * MR_ATR_MULT * a
        return _try_open(cell, spec, s, b, pend["d"], b.open, stop, "MR", px, fills)
    # RSIDIV — 미결 #13 확정식: tgt = fill + 2·(fill−stop) = 3·fill − 2·stop
    stop = pend["stop"]
    tgt = 3.0 * b.open - 2.0 * stop
    return _try_open(cell, spec, s, b, pend["d"], b.open, stop, "RSIDIV", px,
                     fills, tgt=tgt, n4_entry=pend["n4"])


def _manage(cell: CellState, spec: CellSpec, s: str, b: BarE, ind: dict,
            fills: list) -> bool:
    """보유 포지션의 봉내 청산·관리. RSI-DIV 스탑/목표는 레벨 체결 (미결 #15 확정).

    Returns:
        이번 봉에 청산됐으면 True (같은 봉 재진입 금지 판단용).
    """
    p = cell.positions.get(s)
    if p is None:
        return False
    ts_close = b.ts + H1
    if spec.strategy.startswith("BRK"):
        n2 = spec.n // 2
        hl = ind["hl"]
        if p.d > 0:
            xl = min(x[1] for x in hl[-n2:]) if len(hl) >= n2 else -math.inf
            lvl = max(p.stop, xl)
            if b.low <= lvl:
                _close(cell, spec, s, ts_close, min(lvl, b.open), "exit", fills)
                return True
        else:
            xh = max(x[0] for x in hl[-n2:]) if len(hl) >= n2 else math.inf
            lvl = min(p.stop, xh)
            if b.high >= lvl:
                _close(cell, spec, s, ts_close, max(lvl, b.open), "exit", fills)
                return True
        if spec.strategy == "BRK24TP":
            # 변형 (a) 익절 1R — 레벨 체결 (갭 유리해도 레벨), BRK 청산(위)이
            # 먼저 검사되므로 같은 봉 동시 도달 시 스탑/역채널 우선 (비관)
            if (p.d > 0 and b.high >= p.tgt) or (p.d < 0 and b.low <= p.tgt):
                _close(cell, spec, s, ts_close, p.tgt, "target", fills)
                return True
            # 변형 (b) 최대 보유 12×1h봉 (진입봉=1, 결측 봉은 카운트 정지) —
            # 도달 봉 '종가' 청산 (MR 처럼 다음 봉 시가가 아님 — 동결 규칙)
            p.hold += 1
            if p.hold >= VAR_MAX_HOLD:
                _close(cell, spec, s, ts_close, b.close, "timeout", fills)
                return True
    elif spec.strategy == "MR":
        if p.d > 0 and b.low <= p.stop:
            _close(cell, spec, s, ts_close, min(p.stop, b.open), "stop", fills)
            return True
        if p.d < 0 and b.high >= p.stop:
            _close(cell, spec, s, ts_close, max(p.stop, b.open), "stop", fills)
            return True
        p.hold += 1
        cl = ind["cl"]
        if len(cl) == MR_SMA_N:
            sd = float(np.std(cl, ddof=1))
            if sd > 0:
                z = (b.close - float(np.mean(cl))) / sd
                if (p.d > 0 and z >= 0) or (p.d < 0 and z <= 0):
                    p.pending_exit = "signal"
        if not p.pending_exit and p.hold >= MR_MAX_HOLD:
            p.pending_exit = "timeout"
    elif spec.strategy == "BBMR":
        # 스탑 없음 (출판 충실) — 청산 신호만: 확정봉 종가 >= SMA20(현재 종가
        # 포함 20봉) → 다음 봉 시가. 체결봉 종가도 확정봉이라 같은 봉 평가 포함
        # (_step_cells 3단계 예외 — lab run_bollinger 1:1)
        x2 = ind.get("x2")
        if x2 is not None and len(x2["c2"]) >= BB_N - 1:
            mid = float(np.mean(x2["c2"][-(BB_N - 1):] + [b.close]))
            if b.close >= mid:
                p.pending_exit = "signal"
    elif spec.strategy == "RSI2":
        # 스탑 없음 — 청산: 롱 close > SMA5 / 숏 close < SMA5 (현재 종가 포함)
        # → 다음 봉 시가 (lab run_connors 1:1, 체결봉 종가 평가 포함)
        x2 = ind.get("x2")
        if x2 is not None and len(x2["c2"]) >= RSI2_EXIT_N - 1:
            sma5 = float(np.mean(x2["c2"][-(RSI2_EXIT_N - 1):] + [b.close]))
            if (p.d > 0 and b.close > sma5) or (p.d < 0 and b.close < sma5):
                p.pending_exit = "signal"
    elif spec.strategy == "RSIDIV":
        # 원본 유지 (미결 #15 기본값): 스탑/목표 모두 레벨 그대로 체결, 스탑 우선
        if (p.d > 0 and b.low <= p.stop) or (p.d < 0 and b.high >= p.stop):
            _close(cell, spec, s, ts_close, p.stop, "stop", fills)
            return True
        if (p.d > 0 and b.high >= p.tgt) or (p.d < 0 and b.low <= p.tgt):
            _close(cell, spec, s, ts_close, p.tgt, "target", fills)
            return True
        if not p.pending_exit and ind["h4"]["n4"] - 1 - p.n4_entry >= DIV_TIMEOUT:
            p.pending_exit = "timeout"        # 기준점 = 신호봉 인덱스 (미결 #16)
    return False


def _signal(cell: CellState, spec: CellSpec, s: str, b: BarE, ind: dict,
            px: dict, sig, fills: list) -> None:
    """봉 종가 신호 생성 (MR·RSI-DIV·BBMR·RSI2 는 대기주문) 및 BRK 봉내 돌파 진입."""
    if spec.strategy.startswith("BRK"):
        hl, a = ind["hl"], ind["atr1"]      # a = ATR[i-1] (이번 봉 미반영 — 교정 1)
        if cell.halted or len(hl) < spec.n or a is None or a <= 0:
            return
        hi = max(x[0] for x in hl[-spec.n:])
        lo = min(x[1] for x in hl[-spec.n:])
        if b.high > hi:
            d, fill = 1, max(b.open, hi)          # 스탑주문 모델: 불리한 쪽 체결
        elif b.low < lo:
            d, fill = -1, min(b.open, lo)
        else:
            return
        # 변형2 BRK24GATE: 방향 해석(이중 돌파봉 롱 우선) '후' 3중 게이트 —
        # 차단 시 무행동 (반대 방향 재해석 없음, confluence_gate_test 확정 순서).
        # 통과 시 진입·스탑·사이징·청산은 아래 BRK24 경로와 완전 동일.
        if spec.strategy == "BRK24GATE" and not _gate_ok(ind, d):
            return
        stop = fill - d * BRK_ATR_MULT * a
        # 변형 BRK24TP: 진입·스탑·사이징은 위와 완전 동일 경로 — 목표만 추가
        # (1R: tgt = fill + 1×(fill − stop), 롱/숏 공통식. 본 BRK는 tgt=0.0 기본값)
        variant = spec.strategy == "BRK24TP"
        tgt = fill + VAR_TP_R * (fill - stop) if variant else 0.0
        if _try_open(cell, spec, s, b, d, fill, stop,
                     "BRKTP" if variant else "BRK", px, fills, tgt=tgt):
            _post_fill_check(cell, spec, s, b, fills)   # 같은 봉 스탑 비관 (봉내 진입)
    elif spec.strategy == "MR":
        cl = ind["cl"]                       # shift(1) — 현재 봉 제외 24봉
        if len(cl) != MR_SMA_N:
            return
        sd = float(np.std(cl, ddof=1))
        if sd <= 0:
            return
        z = (b.close - float(np.mean(cl))) / sd
        d = -1 if z > MR_Z else (1 if z < -MR_Z else 0)
        if d:
            cell.pending[s] = {"kind": "MR", "d": d, "ets": b.ts + H1}
    elif spec.strategy == "BBMR":
        # 확정봉 종가 < 하단밴드(종가 기준, 봉내 터치 아님) → 다음 봉 시가 롱.
        # 밴드는 현재 종가 포함 20봉 (pandas rolling(20) 동형), 미형성 = 무신호.
        x2 = ind.get("x2")
        if x2 is None or len(x2["c2"]) < BB_N - 1:
            return
        w = x2["c2"][-(BB_N - 1):] + [b.close]
        mid = float(np.mean(w))
        sd = float(np.std(w))               # ddof=0 모표준편차 (출판 기본값)
        if b.close < mid - BB_K * sd:       # 롱 온리 — 상단밴드 숏 없음 (동결)
            cell.pending[s] = {"kind": "BBMR", "d": 1, "ets": b.ts + H1}
    elif spec.strategy == "RSI2":
        # Connors 원전: 롱 close>SMA200 ∧ RSI(2)<5 / 숏 close<SMA200 ∧ RSI(2)>95
        # → 다음 봉 시가 (U1 실행 규약). SMA200·RSI 는 현재 확정 종가 포함.
        x2 = ind.get("x2")
        if x2 is None or len(x2["c2"]) < RSI2_TREND_N - 1:
            return                          # SMA200 미형성 (pandas NaN) — 무신호
        c2 = x2["c2"]
        r2 = _rsi_next(x2["u2"], x2["d2"], b.close - c2[-1], RSI2_N)
        sma200 = float(np.mean(c2[-(RSI2_TREND_N - 1):] + [b.close]))
        if b.close > sma200 and r2 < RSI2_LO:
            cell.pending[s] = {"kind": "RSI2", "d": 1, "ets": b.ts + H1}
        elif b.close < sma200 and r2 > RSI2_HI:
            cell.pending[s] = {"kind": "RSI2", "d": -1, "ets": b.ts + H1}
    elif spec.strategy == "RSIDIV" and sig is not None:
        d, stop, idx = sig
        cell.pending[s] = {"kind": "RSIDIV", "d": d, "stop": stop, "n4": idx,
                           "ets": b.ts + H1}


def step(state: FarmState, bars: dict, funding: dict | None = None) -> list:
    """닫힌 1h 봉 1개(전 심볼)를 본 셀 10개 전부에 처리한다 — 백테스트·라이브 공용.

    본 셀(E01~E10) 전용 진입점 — 변형(E11·E12)은 step_variant() 로만 돈다
    (셀 목록을 공개 인자로 받지 않아 상태·셀 조합 오용을 차단).

    Args:
        state: 본 팜 상태 (변형됨).
        bars: sym -> BarE (동일 ts 필수).
        funding: sym -> 이 봉 종가 시각에 정산되는 펀딩률 합 (없으면 0).

    Returns:
        체결 이벤트 목록 (유일키: cell, sym, strategy, bar_close, action).
    """
    return _step_cells(state, bars, funding, CELLS)


def step_variant(state: FarmState, bars: dict, funding: dict | None = None) -> list:
    """변형 서브팜(E11·E12) 전용 step — 본 셀과 같은 내부 경로, 상태·셀만 분리.

    Args:
        state: 변형 서브상태 (variant_from_dict 결과, 변형됨).
        bars: sym -> BarE (동일 ts 필수).
        funding: sym -> 이 봉 종가 시각에 정산되는 펀딩률 합 (없으면 0).

    Returns:
        체결 이벤트 목록 (cell 은 E11/E12 만).
    """
    return _step_cells(state, bars, funding, VCELLS)


def step_variant2(state: FarmState, bars: dict, funding: dict | None = None) -> list:
    """변형2 서브팜(E13~E18) 전용 step — 같은 내부 경로, 상태·셀만 분리.

    확장 지표(x2)가 없는 심볼(초기화 후 새로 나타난 경우)은 여기서 빈 상태로
    만든다 — 이력이 찰 때까지 게이트·밴드·RSI 신호는 전부 차단 (fail-closed).

    Args:
        state: 변형2 서브상태 (variant2_from_dict 결과, 변형됨).
        bars: sym -> BarE (동일 ts 필수, vol 포함 권장 — 결측 시 게이트만 차단).
        funding: sym -> 이 봉 종가 시각에 정산되는 펀딩률 합 (없으면 0).

    Returns:
        체결 이벤트 목록 (cell 은 E13~E18 만).
    """
    for s, b in bars.items():
        if b.ts > state.last_ts:
            state.ind.setdefault(s, _new_ind()).setdefault("x2", _new_x2())
    return _step_cells(state, bars, funding, V2CELLS)


def _step_cells(state: FarmState, bars: dict, funding: dict | None,
                cells: tuple) -> list:
    """닫힌 1h 봉 1개를 cells 전체에 처리하는 내부 공용 구현 (인과 규약 준수)."""
    if not bars:
        return []
    ts_set = {b.ts for b in bars.values()}
    if len(ts_set) != 1:
        raise ValueError(f"봉 ts 불일치: {sorted(ts_set)}")
    ts = ts_set.pop()
    if ts <= state.last_ts:
        logger.warning("이미 처리한 봉 ts=%d — 무시 (멱등)", ts)
        return []
    funding = funding or {}
    fills: list = []

    ok_bars = {}
    for s, b in bars.items():
        if s in state.delisted:
            continue
        if any(math.isnan(x) for x in (b.open, b.high, b.low, b.close)):
            logger.warning("%s 봉 NaN — 결측 처리 (fail-closed)", s)
            continue
        ok_bars[s] = b

    # 0) 4h 집계·신호 (이 봉 종가에 확정 — 체결은 다음 1h봉 시가)
    sig4 = {}
    for s, b in ok_bars.items():
        ind = state.ind.setdefault(s, _new_ind())
        r = _update_4h(ind["h4"], b)
        if r is not None:
            sig4[s] = r

    live = state.t0 > 0 and ts >= state.t0
    day_key = str(datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date())
    prev_px = {s: i_["pc"] for s, i_ in state.ind.items() if i_.get("pc") is not None}
    cur_px = dict(prev_px)
    cur_px.update({s: b.close for s, b in ok_bars.items()})

    for spec in cells:
        cell = state.cells[spec.cell]
        if cell.day != day_key:                    # UTC 일 경계
            cell.day, cell.halted = day_key, False
            cell.day_eq = _cell_mtm(cell, prev_px)
        act = []
        for s in cell_syms(spec, state):
            if s in state.delisted:
                continue
            if s in ok_bars:
                act.append(s)
                continue
            if s in cell.pending:                  # 다음 봉 시가 체결 불가 — 취소
                cell.pending.pop(s)
                logger.warning("%s %s 봉 결측 — 대기주문 취소 (fail-closed)",
                               spec.cell, s)
            if s in cell.positions:
                logger.warning("%s %s 봉 결측 — 무행동 (fail-closed)", spec.cell, s)
        touched = set()
        marks = dict(prev_px)          # 셀 로컬 마크맵 — 체결 확정가로 갱신됨
        # 1) 대기 청산 — 봉 시가 (봉내 사건보다 먼저)
        for s in act:
            p = cell.positions.get(s)
            if p is not None and p.pending_exit:
                _close(cell, spec, s, ts + H1, ok_bars[s].open,
                       "exit_" + p.pending_exit, fills)
                touched.add(s)
        # 2) 대기 진입 — 봉 시가. 전 심볼 체결을 동시 확정한 뒤에야 같은 봉
        #    스탑/목표를 검사한다 (봉내 사건이 시가 결정을 소급 변경 금지)
        newly = []
        for s in act:
            if s not in touched and _fill_pending(cell, spec, s, ok_bars[s],
                                                  state.ind[s], live, marks, fills):
                touched.add(s)
                newly.append(s)
        for s in newly:
            _post_fill_check(cell, spec, s, ok_bars[s], fills)
        # 3) 봉내 청산·관리 (이번 봉 시가 체결분 제외 — 단 스탑 없는 신호청산형
        #    BBMR·RSI2 는 체결봉 종가도 확정봉 종가이므로 청산 '신호' 평가에
        #    포함한다: 원전 lab 1:1, 실제 청산은 다음 봉 시가라 인과 무결.
        #    두 전략의 _manage 는 봉내 청산이 없어 touched 를 늘리지 않는다)
        sigexit = spec.strategy in ("BBMR", "RSI2")
        for s in act:
            if (s not in touched or sigexit) and _manage(cell, spec, s, ok_bars[s],
                                                         state.ind[s], fills):
                touched.add(s)
        # 4) 신호·봉내 진입 — 같은 봉 청산 심볼 재진입 금지, T0 이전 주문 금지 (교정 6)
        if live:
            for s in act:
                if s in touched or s in cell.positions or s in cell.pending:
                    continue
                _signal(cell, spec, s, ok_bars[s], state.ind[s], marks,
                        sig4.get(s), fills)
        # 5) 펀딩 — 이 봉 종가 시각 정산분 (교정 7) — 일손실 판정보다 먼저
        for s, p in cell.positions.items():
            f = funding.get(s, 0.0)
            b = ok_bars.get(s)
            if f and b is not None:
                amt = p.d * f * p.u * b.close
                cell.equity -= amt
                cell.fund += amt
                # 원장 펀딩 이벤트 — 판정(tracke_null) 계약: 양수 = 수취
                fills.append(_ev(spec, s, ts + H1, "funding", b.close, p.u,
                                 -amt, 0.0, p.d, fund=-amt))
        # 6) 일손실 트리거 (도달=<=) — 신규 진입만 정지, 청산하지 않음
        mtm = _cell_mtm(cell, cur_px)
        if not cell.halted and cell.day_eq > 0 and mtm / cell.day_eq - 1 <= DAILY_HALT:
            cell.halted = True
            cell.halts += 1
            logger.warning("%s 일손실 -5%% 트리거 — 당일 신규 진입 정지 (청산 아님)",
                           spec.cell)

    # 1h 지표 갱신 — 마지막에 하여 다음 봉의 ATR[i-1]·shift(1) 채널을 만든다
    for s, b in ok_bars.items():
        _update_1h(state.ind[s], b)
    state.last_ts = ts
    return fills


def mark_delisted(state: FarmState, sym: str, last_px: float | None = None) -> list:
    """폐지·데이터 단절 — 마지막 유효가로 본 셀 전체 청산 후 슬롯 영구 공석.

    Args:
        last_px: 러너가 아는 더 최신의 마지막 유효 종가 (없으면 상태의 직전 종가).
    """
    return _delist_cells(state, sym, last_px, CELLS)


def variant_delist(v: FarmState, sym: str, last_px: float | None = None) -> list:
    """변형 셀(E11·E12) 폐지 처리 — 본 팜 폐지의 미러 (관례 동일, 상태 분리).

    청산가는 변형 상태의 마지막 처리 종가 — 본 셀 청산 시점과 어긋날 수 있다
    (변형이 뒤처진 채 폐지가 미러되면; fail-closed 관례로 동결).
    """
    return _delist_cells(v, sym, last_px, VCELLS)


def variant2_delist(v: FarmState, sym: str, last_px: float | None = None) -> list:
    """변형2 셀(E13~E18) 폐지 처리 — variant_delist 와 동일 관례, 그룹만 분리."""
    return _delist_cells(v, sym, last_px, V2CELLS)


def _delist_cells(state: FarmState, sym: str, last_px: float | None,
                  cells: tuple) -> list:
    """폐지 내부 공용 구현 — cells 의 포지션을 마지막 유효가로 강제 청산."""
    if sym in state.delisted:
        return []
    state.delisted.append(sym)
    px = last_px if last_px is not None else state.ind.get(sym, {}).get("pc")
    ts_close = state.last_ts + H1
    fills: list = []
    for spec in cells:
        if sym not in cell_syms(spec, state):
            continue
        cell = state.cells[spec.cell]
        cell.pending.pop(sym, None)
        p = cell.positions.pop(sym, None)
        if p is None:
            continue
        use = px if px is not None else p.e        # 가격 이력 없으면 진입가 (fail-closed)
        pnl = p.u * (use - p.e) * p.d
        cost = p.u * use * COST_SIDE
        cell.equity += pnl - cost
        cell.cost += cost
        cell.turnover += p.u * use
        fills.append(_ev(spec, sym, ts_close, "force_exit", use, p.u, pnl, cost, p.d))
        logger.warning("%s %s 폐지 강제청산 @ %.6f — 슬롯 영구 공석", spec.cell, sym, use)
    return fills
