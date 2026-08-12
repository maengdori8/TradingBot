# Evidence pipeline

`python -m research.pipeline` is the only research entry point whose outputs may be submitted to the promotion gate. `funding.py`, `funding_v2.py`, and `wfo.py` remain exploratory and stamp their JSON with `evidence_status=legacy_non_evidence`.

## Input contract

The CLI accepts one `research-evidence-v1` JSON object. It must contain:

- UTC `data_start`, `data_end`, `data_cutoff`, and `created_at` values;
- lowercase SHA-256 `code_hash` and `data_hash` values;
- feed completeness and longest unresolved gap;
- an account fee snapshot, component freshness limits, replay capital and order constraints;
- UTC daily `benchmark_returns` with `date`, `return`, and `available_at`;
- point-in-time daily liquidity, instrument rules, carry opportunities, and forced-flow raw feature bundles.

At least one external `DataManifest` must be supplied with `--data-manifest ROLE PATH SHA256`. Every file is strict-parsed with `DataManifest.from_json` and must be evidence eligible. Its role, fixed evidence hash, raw payload root, source metadata, and required heartbeat bindings are included in `data_hash`; filesystem paths are not. `data_hash` is the canonical SHA-256 of this binding list plus `data_start`, `data_end`, `data_cutoff`, quality metadata, and `data`. JSON objects with duplicate keys are rejected. Canonical JSON is defined by `research.evidence_contracts.canonical_json`.

Carry needs full-period manifests for spot/perpetual kline, funding settlement, OI, and instrument metadata. Forced-flow needs full-period 12-month manifests for kline/volume, orderbook, OI, funding, and liquidation with connection heartbeat. A complete carry family may run while incomplete forced-flow candidates are emitted as `insufficient_data`; neither family borrows the other family’s provenance.

Carry opportunities provide entry/exit spot and perpetual prices plus every actual funding settlement in `(entry_time, exit_time]`. Forced-flow bundles provide as-of price, OI, completed volume, known funding, normalized liquidation flow, 25-level book, and feed gaps. Raw Bybit liquidation position side must be normalized before it becomes `LiquidationNotional.side`: a long liquidation is aggressive `sell`, and a short liquidation is aggressive `buy`.

## Fixed execution

The runner always registers and evaluates exactly eight `delta_neutral_carry` and eight `forced_flow` candidates from `research.candidates.all_predefined_candidates`. It uses the point-in-time universe and fixed expanding splits: 365-day minimum train, 90-day complete OOS tests, 48-hour purge, and 48-hour embargo. No CLI parameter can add or retune candidates.

Run:

```bash
python -m research.pipeline --input evidence.json --output research/out/evidence \
  --data-manifest carry_spot_kline manifests/btc-spot-kline.json SHA256 \
  --data-manifest carry_perpetual_kline manifests/btc-perp-kline.json SHA256
python -m research.pipeline --list-candidates
```

The authoritative output is `candidate_results.json`; `trades.csv`, `daily.csv`, and `candidate_matrix.csv` are sorted derivatives. `benchmark.csv` has exactly the same OOS dates as the candidate matrix. `load_evidence_outputs` validates externally fixed result/matrix/benchmark hashes, strict schemas, candidate and manifest lineage, and rebuilds Risk-gate trade/daily inputs only from the authoritative results. `evidence_summary.json` reports raw evidence admissibility separately from `eligible_strategy_count`. The latter remains zero with `promotion_evaluation_status=pending_risk_gate` until the independent risk gate evaluates the artifacts.

## Runtime reuse

Paper and demo orchestration must use the same public pure functions exported by `src.strategy`:

- `decide_carry_intent` returns an atomic spot-buy/perpetual-sell two-leg intent or `None`;
- `build_forced_flow_snapshot` builds the 4h/8h as-of features;
- `decide_forced_flow_intent` returns one directional perpetual leg or `None`.

Missing, future, stale, or gap-contaminated feature inputs return no forced-flow intent. These functions do not submit orders or bypass execution and risk controls.
