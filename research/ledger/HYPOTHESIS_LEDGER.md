<!-- 자율 연구 루프 가설 원장 — 사이클 간 단일 진실원천 (GOAL_LOOP_PROMPT.md §4) -->
<!-- 기계가독 헤더: 매 사이클 갱신. 사람용 티켓은 아래. -->

```
N_TESTED: 405
HOLDOUT_FROZEN_UNTIL: 2026-04-16T00:00:00+00:00
HOLDOUT_EVALS: 1
BACKLOG_REMAINING: 9
ITER_COUNT: 1
CONSECUTIVE_REJECTS: 1
LAST_VERDICT: REJECT
```
진행: A1=REJECTED. 다음 사이클 가설 = A2(회전율 최소화/보유시간 연장).

마스터 백로그(고정 10): A1 A2 A3 B1 B2 B3 C1 C2 D1 D2 — GOAL_LOOP §6.
죽은 6군(재튜닝 금지): ICT컨플루언스 / 개별종목4(donchian·tsmom·bbreak·meanrev) / 횡단면모멘텀(메이저·광범위) / 펀딩캐리(방향·델타중립) / 공적분페어.

---

## 인프라 (사이클1 부트스트랩)
- `research/stats_gate.py` (+ `test_stats_gate.py` 6/6 통과): 다중검정/데이터스누핑 게이트 단일구현.
  deflated Sharpe(SR0 노이즈천장) + 클러스터 실효표본(거래일) + 블록 부트스트랩 CI + 본페로니 p.
  고정문턱: base_bar=0.05R, k=0.02, min_n_eff=50, DSR≥0.95, p<α/N.
- `research/dl_universe.py`: WFO 유니버스(14심볼) 캔들 다운로더(캐시 idempotent).
- `research/exec_a1.py`: A1 메이커 지정가 실행모델 재생.
- `research/validate_a1.py`: 동결홀드아웃 WFO + stats_gate 판정(wfo 내장 verdict·이동창 미사용).

## 도구 함정 확인 (코드 라인 직접 검증, GOAL_LOOP §2)
- wfo.py `_verdict`(300~312): oos>0 AND holdout>0(n≥20)만 — 노이즈 PASS 가능 → **인용 금지, stats_gate만 판정**. ✅확인
- wfo.py main()(264~282): 3라운드 OOS 최고 사후선택 → N에 +3. ✅확인
- wfo.py 홀드아웃(258): `df.ts.max()-60일` 이동창 → **2026-04-16 고정 동결**. ✅확인
- wfo.py param_grid(122~133): min_score4×zone2×btc2×exit25 = 400조합 → N에 가산. ✅확인
- study.py COST_PCT(37): 기본 0.0021(taker), env STUDY_COST_PCT. ✅확인

---

## [H-001] A1 메이커 전용 지정가 실행 모델  (tier: A, status: REJECTED)

- **가설(메커니즘)**: 죽은 ICT 신호의 그로스는 미세 양수(+0.03~0.07R)였으나 taker 왕복비용
  0.21%(≈0.16~0.20R)가 전부 삼켰다(WFO 39~46행). maker 지정가(0.07%)가 비용 드래그를
  ~1/3로 줄이면 어떤 출구조합이 순양수로 살아나는가? 반대편 손해자=시장가 추격 테이커.
- **신규성(5축 지문)**: 데이터소스=동일(ICT 신호) / 신호메커니즘=동일 / **비용·실행구조=변경(시장가→지정가+체결모델)**
  / 유니버스=14메이저 동일 / 타임프레임=동일. → 죽은 6군과 **실행구조 축**에서 분리(파라미터 재튜닝 아님).
- **실행가능성**: 즉시(기존 캔들·신호 인프라 재사용). 데이터 확보 완료.
- **사전등록(결과 보기 전 고정 — 2026-06-15)**:
  - 진입: 지정가 = 시장진입가 ∓ **0.25·ATR**(롱 -, 숏 +). 유리한 오프셋.
  - 체결모델: 체결창 **8봉(2h)** 내 가격이 지정가에 닿아야 체결(롱: low≤limit). **미체결=거래없음(비용0)**.
    → 역선택 내장: 채워졌다는 건 가격이 먼저 불리하게 왔다는 것.
  - 비용: maker 왕복 **0.0007**. 베이스라인 taker 0.0021 동시 산출.
  - 음성통제: 방향셔플 플라시보(시드 20260615) 동일 체결로직.
  - **성공 바(PROMOTE 조건, 전부 충족)**: stats_gate.gate() passed=True (DSR≥0.95 · CI하한>0 ·
    mean_r≥r_min(N) · n_eff≥50 · 본페로니p<α/N) **AND** 동결홀드아웃 mean_r>0(n≥20)
    **AND** OOS mean_r > 플라시보 mean_r. 하나라도 미달 → REJECT.
  - 평가식: 동결경계(2026-04-16) 이전=탐색(3라운드 WFO), 이후=홀드아웃 1회. N=이전누적+1+그리드400+3+1.
- **룩어헤드 감사(§3)** — 항목별 확인:
  1. 결정시점 정합: HTF 완전닫힌봉 선택은 study.replay_symbol(247~249행, 8에이전트 감사 통과한
     수정본)을 그대로 상속. ✅
  2. rolling: exec_a1에 신규 rolling 없음. ATR=신호 손절거리/atr_mult(study 동일). ✅
  3. 파라미터 격리: WFO가 train에서만 config 선택, 동결홀드아웃 1회만 평가(validate_a1). ✅
  4. 상태누수: 체결 sim_start=i+1 이후 봉만. `fill_delay_bars` min=**0**(음수=과거체결 없음), mean 0.71, max 7(체결창 8 이내). ✅
  5. 비용타이밍: maker 왕복 진입/청산 시 1회. 펀딩 미관여. ✅
  6. 대체데이터: 해당없음(A1은 가격만). ✅
  7. 음성통제: **real vs placebo** — 진짜 OOS +0.0533R > 플라시보(방향셔플) +0.0134R → 방향신호
     ~+0.04R 존재하나 **둘 다 노이즈천장 이하**. leaked-label 별도판은 미구현이나 신호경로가
     기감사 study와 동일 + 신규코드 forward-only(fill_delay≥0)로 입증 → N/A 사유 기록.
  → **룩어헤드 후보 0건.** 음수/노이즈 결과는 진짜다.
- **결과** (14심볼, 2024-01~2026-06, 23,034신호 / 체결 19,614 = **체결률 85.2%**):
  - **테이커 베이스라인 OOS −0.077R**(전 라운드 −0.077~−0.093) — WFO 문서의 taker 사망과 일치.
  - **메이커 PRIMARY OOS +0.0533R**(240d 라운드 최선), **동결홀드아웃 +0.0958R**(n=1163, 승률 27.2%) — 둘 다 양수.
  - robust_param: min_score65 / zone_both=F / btc=no_counter / **r_m2_rr3.5**.
  - **전수(사후선택 없는) r_m2_rr2.5 = −0.0019R** ≈ WFO 문서 "메이커 겨우 손익분기 −0.004R"와 정합.
    → OOS +0.053은 **400조합 사후선택 부풀림**.
  - 비용민감도: taker −0.077 → maker +0.053 (부호반전). **메이커 실행은 비용 문턱을 실제로 제거**한다.
  - 플라시보 +0.0134R(n=18,257): 지정가 체결의 **기계적 평균회귀**(딥매수/랠리매도)가 신호와 무관히 미세 양수 — 외관 엣지의 교란요인.
- **N_TESTED / 문턱 / stats_gate 반환**: N=405(이전0+티켓1+그리드400+라운드3+홀드아웃1).
  r_min(405)=**0.1701R**. stats_gate.**passed=False**, 사유 5건:
  ① mean_r +0.0533 < 문턱 0.1701  ② Sharpe 0.0275 ≤ 노이즈천장 SR0 0.1345(초과 −0.107)
  ③ DSR 확률 0.0079 < 0.95  ④ 부트스트랩 CI 하한 −0.0746 ≤ 0(CI −0.075~+0.190, 실효표본 494거래일)
  ⑤ 본페로니 p 0.2704 ≥ α/N 0.000123.
- **판정 + status 전이**: **REJECT** (TESTING→REJECTED). OOS·홀드아웃 양수로 wfo 내장 _verdict였다면
  "엣지 있음(잠정)" **거짓양성**을 냈을 사례 — stats_gate가 400조합 다중검정·노이즈천장으로 정확히 기각.
  포트폴리오 게이트(§5-7)는 stats_gate에서 이미 REJECT이므로 평가 무의미(moot).
- **남는 함의(다음 사이클 입력)**: 메이커 실행이 비용 부호를 뒤집는 것은 실재하나, 잔존 방향엣지(~+0.04R)는
  다중검정 보정 후 노이즈와 구분 불가. tier-A 비용각도(A2 회전율↓, A3 비용곡선)는 "비용은 넘되 신호가 얇다"는
  본 결과를 출발점으로 삼아야 함. 비용만으론 부족 — 더 두꺼운 그로스 신호(tier B 대체데이터)가 필요.
- **재현**: `python -m research.dl_universe --start 2024-01-01` → `PYTHONIOENCODING=utf-8 python -m research.exec_a1 --workers 6` →
  `python -m research.validate_a1 --holdout-until 2026-04-16T00:00:00+00:00 --n-prior 0`
  (산출: research/out/a1_signals.csv, a1_validation.json)
