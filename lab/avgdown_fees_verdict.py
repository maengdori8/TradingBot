"""AVGDOWN-FEES-2026-09-01 판정 — 시나리오별 독립 RC(고정 ω̂)·StepM·DSR.

명세: `docs/PREREGISTRATION_AVGDOWN_FEES_2026-09-01.md` §6. 판정 기계는 동결
`lab/avgdown_verdict.py` (그 안의 동결 `lab/sweep_verdict.py` 포함)를 **읽기 전용
임포트**해 그대로 쓴다 — 본 파일은 시나리오 3개(a/b/c)에 같은 기계를 독립 적용하고
교차 연구 다중성 경고를 산출물에 강제 각인하는 래퍼일 뿐이다.

시나리오별 구성 (동결 — 원 스윕 §6 동일):
* 고정 ω̂ = 원표본 std(ddof=1) 1회 추정 (경로별 재추정 금지),
* 정상 부트스트랩 평균 블록 5일 · 1,000 경로 · seed 20260901,
* Romano–Wolf StepM (FWER 0.05) · DSR ≥ 0.95, N = 1,248 (시나리오 내부).

**교차 다중성 경고 (판정 권한 제한)**: 시나리오 3개 간 다중성은 보정하지 않으며,
본 트랙 자체가 같은 데이터·같은 1,248 시행에 대한 **두 번째 밀접 재분석**이다.
어느 시나리오의 명목 p 도 액면대로 해석할 수 없다 — 산출물의 `cross_study_warning`
필드가 이 제한을 항구적으로 각인한다.

실행:
  .venv/bin/python lab/avgdown_fees_verdict.py --selftest  # 합성 자가검증만
  .venv/bin/python lab/avgdown_fees_verdict.py --run       # 판정 1회 (산출물 필요)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── 동결 판정 기계 로드 (읽기 전용 — 수정 없음) ───────────────────────────
_AV_SPEC = importlib.util.spec_from_file_location(
    "avgdown_verdict_frozen", str(ROOT / "lab" / "avgdown_verdict.py"))
av = importlib.util.module_from_spec(_AV_SPEC)
sys.modules["avgdown_verdict_frozen"] = av
_AV_SPEC.loader.exec_module(av)

SEED: int = av.SEED                                  # 20260901
N_PATHS: int = av.N_PATHS                            # 1000
N_TRIALS: int = av.N_TRIALS                          # 1248
N_DAYS: int = av.N_DAYS                              # 2056

SCEN_KEYS = ("a", "b", "c")
SCEN_LABELS = {
    "a": "실측 테이커 편도 5.5bp — 시장가 체결 (원 스윕 동일 모델)",
    "b": "메이커 상한 편도 2bp — 시장가 체결 가정 유지 = 실현 불가능한 상한선",
    "c": "보수적 메이커 편도 2bp — 지정가 체결 (엄격 관통, 미체결 신호 소멸)",
}

CROSS_STUDY_WARNING = (
    "교차 연구 다중성 경고: 본 판정은 AVGDOWN-2026-09-01 과 같은 동결 데이터·"
    "같은 1,248 시행에 대한 두 번째 밀접 재분석이며, 원 스윕의 시행별 결과"
    "(왕복 16bp, RC p=0.7632, 생존 0)는 이미 전부 조회된 상태에서 설계됐다. "
    "시나리오 3개 간 다중성도 보정하지 않는다. 따라서 어느 시나리오에서 생존자가 "
    "나와도 명목 p 보다 약한 증거이고, 승급 불가이며, 전향(신규 기간) 검증이 "
    "필수다. 시나리오 (b) 는 체결 모델상 실현 불가능한 상한선으로 진단 전용이다."
)

ORCH_PREDICTIONS = {                                 # 결과 조회 전 동결 (§7)
    "a": "생존 0",
    "b": "생존 0~2 — 진성 불확실",
    "c": "미체결·역선택으로 (b)보다 악화",
}


def run_fees_verdict(indir: Path, n_paths: int = N_PATHS, seed: int = SEED
                     ) -> dict[str, Any]:
    """시나리오 3개에 동결 판정 기계를 독립 적용한다.

    Args:
        indir: `avgdown_fees_{a,b,c}.npz` 가 있는 디렉터리.
        n_paths: 부트스트랩 경로 수. seed: 부트스트랩 seed.

    Returns:
        직렬화 가능한 판정 딕셔너리 (시나리오별 블록 + 교차 다중성 경고).

    Raises:
        SystemExit: 산출물 결측 시 (fail-closed — 부분 판정 금지).
    """
    paths = {k: indir / f"avgdown_fees_{k}.npz" for k in SCEN_KEYS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"산출물 결측 — fail-closed (부분 판정 금지): {missing}")
    scenarios: dict[str, Any] = {}
    for k in SCEN_KEYS:
        v = av.run_verdict(paths[k], n_paths=n_paths, seed=seed)
        v["scenario"] = k
        v["scenario_label"] = SCEN_LABELS[k]
        v["orchestrator_prediction_frozen"] = ORCH_PREDICTIONS[k]
        v["verdict"]["authority"] = (
            "none — 진단 전용. 생존자도 승급 불가, 전향 검증 필수 "
            "(교차 연구 다중성 — cross_study_warning 참조)")
        scenarios[k] = v
    return {
        "spec": "AVGDOWN-FEES-2026-09-01",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cross_study_warning": CROSS_STUDY_WARNING,
        "bootstrap": {"n_paths": n_paths, "seed": seed,
                      "mean_block_days": av.MEAN_BLOCK_DAYS,
                      "note": "시나리오별 독립 RC — 고정 ω̂ 원표본 1회 추정, "
                              "경로별 재추정 금지 (원 스윕 §6.2 동결 계승)"},
        "scenarios": scenarios,
        "question": "원 스윕의 판정 실패가 신호 부재 탓인지 비용 탓인지의 진단 — "
                    "어떤 결과도 시행을 승급시키지 못한다",
    }


def _fmt(v: dict[str, Any]) -> str:
    """판정 딕셔너리 → 사람이 읽는 보고서."""
    out: list[str] = []
    a = out.append
    a("=" * 78)
    a("AVGDOWN-FEES-2026-09-01 판정 (수수료 민감도 진단 — 승급 권한 없음)")
    a("=" * 78)
    a(v["cross_study_warning"])
    a("-" * 78)
    for k in SCEN_KEYS:
        blk = v["scenarios"][k]
        p = blk["primary"]
        rc = p["reality_check"]
        b = p["best"]
        a(f"[시나리오 {k}] {blk['scenario_label']}")
        a(f"  사전 예측(동결): {blk['orchestrator_prediction_frozen']}")
        a(f"  최고 시행: {b.get('trial_id', '?')} — 연환산 SR {b['sharpe_ann']:+.3f}")
        a(f"  RC p = {rc['p']:.4f} · StepM 기각 {p['stepm']['n_rejected']}개 · "
          f"DSR>=0.95 통과 {p['n_passing']}개 · 퇴화 {p['n_degenerate']}개")
        a(f"  판정: {'통과(명목)' if blk['verdict']['pass'] else '실패'} — "
          f"{blk['verdict']['statement']}")
    a("=" * 78)
    return "\n".join(out)


# ── 자가검증 (합성 데이터 전용 — 본 산출물 미접촉) ─────────────────────────
def _make_synth_npz(path: Path, inject_row: int | None) -> None:
    """전체 크기(1248×2056) 합성 npz — 배선 검증 전용."""
    rng = np.random.default_rng(4242)
    ret = rng.normal(0.0, 0.01, (N_TRIALS, N_DAYS))
    if inject_row is not None:
        ret[inject_row] += 0.003                     # 일 +30bp 균일 엣지 주입
    tids = np.array([f"SYN|{i:04d}" for i in range(N_TRIALS)], dtype=object)
    np.savez_compressed(path, daily_returns=ret, trial_ids=tids,
                        meta=json.dumps({"spec": "selftest-synth"}))


def selftest() -> None:
    """판정 래퍼 자가검증 — 위반 시 AssertionError.

    1. 동결 판정 기계 자체 selftest (고정 ω̂ 항등식·오지정 탐지·검정력·결정론).
    2. 래퍼 배선: 합성 전체 크기 npz 3벌(1벌에 신호 주입)에 대해 시나리오 블록·
       경고 문구·동결 예측이 산출물에 각인되고, 주입 신호가 기각된다.
    3. 결정론: 같은 seed → 같은 RC p.
    4. fail-closed: 산출물 결측 시 SystemExit.
    """
    print("--- selftest (avgdown_fees_verdict — 합성 데이터) ---")
    av.selftest()                                    # 1) 동결 기계 자가검증
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        _make_synth_npz(tdp / "avgdown_fees_a.npz", inject_row=None)
        _make_synth_npz(tdp / "avgdown_fees_b.npz", inject_row=7)
        _make_synth_npz(tdp / "avgdown_fees_c.npz", inject_row=None)
        v = run_fees_verdict(tdp, n_paths=200, seed=SEED)
        assert set(v["scenarios"]) == set(SCEN_KEYS)
        assert "두 번째 밀접 재분석" in v["cross_study_warning"]
        for k in SCEN_KEYS:
            blk = v["scenarios"][k]
            assert blk["orchestrator_prediction_frozen"] == ORCH_PREDICTIONS[k]
            assert "승급 불가" in blk["verdict"]["authority"]
        pa = v["scenarios"]["a"]["primary"]["reality_check"]["p"]
        pb = v["scenarios"]["b"]["primary"]["reality_check"]["p"]
        assert pb < 0.05, pb                         # 주입 신호 기각
        assert 7 in v["scenarios"]["b"]["primary"]["stepm"]["rejected_indices"]
        assert pa > 0.05, pa                         # 무신호 시나리오 비기각
        print(f"  [OK] 래퍼 배선 — 주입 시나리오 RC p={pb:.4f} 기각, "
              f"무신호 p={pa:.4f} 비기각, 경고·예측 각인 확인")
        v2 = run_fees_verdict(tdp, n_paths=200, seed=SEED)
        assert v2["scenarios"]["b"]["primary"]["reality_check"]["p"] == pb
        print("  [OK] 같은 seed → 같은 RC p (결정론)")
        rep = _fmt(v)
        assert "실현 불가능한 상한선" in rep and "교차 연구 다중성" in rep
        try:
            run_fees_verdict(tdp / "없는디렉터리")
            raise AssertionError("결측 산출물에서 fail-closed 실패")
        except SystemExit:
            print("  [OK] 산출물 결측 → SystemExit (fail-closed, 부분 판정 금지)")
    print("--- selftest 전부 통과 ---")


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    ap = argparse.ArgumentParser(description="AVGDOWN-FEES-2026-09-01 판정")
    ap.add_argument("--selftest", action="store_true", help="합성 자가검증만")
    ap.add_argument("--run", action="store_true", help="판정 1회 수행")
    ap.add_argument("--indir", default="logs")
    ap.add_argument("--outdir", default="logs")
    ap.add_argument("--paths", type=int, default=N_PATHS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        selftest()
        return 0
    if not args.run:
        print("아무 것도 하지 않음 — --selftest 또는 --run 지정")
        return 1
    selftest()                                       # 판정 전 자가검증 강제
    v = run_fees_verdict(ROOT / args.indir, n_paths=args.paths)
    outd = ROOT / args.outdir
    outd.mkdir(parents=True, exist_ok=True)
    tmp_j = outd / "avgdown_fees_verdict.json.tmp"
    tmp_j.write_text(json.dumps(v, ensure_ascii=False, indent=2))
    os.replace(tmp_j, outd / "avgdown_fees_verdict.json")
    rep = _fmt(v)
    tmp_r = outd / "avgdown_fees_verdict_report.txt.tmp"
    tmp_r.write_text(rep + "\n")
    os.replace(tmp_r, outd / "avgdown_fees_verdict_report.txt")
    print(rep)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
