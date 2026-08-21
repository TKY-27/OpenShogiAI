# Phase 10R-C Progress

Updated: 2026-08-21 (Asia/Tokyo)

## Current state

The campaign is **blocked before the 1m rung**. No bulk acquisition, normalization beyond the
existing Phase 10R-A samples, training, teacher relabeling, Arena, cross-play, self-play, model
promotion, holdout inspection, push, release, or deployment was performed.

The clean execution branch is `codex/phase10r-curriculum`, with the latest execution-infrastructure
commit `414fa0b`. The objective requested
`codex/phase10r-execution`, but the immutable execution prompt requires
`codex/phase10r-curriculum`; the prompt's exact branch lock was treated as the more restrictive
rule.

## Checks completed

- Frozen hash manifest: passed, all 26 entries.
- `make phase10r-validate`: passed.
- `make phase10r-sanity`: passed.
- `make phase10r-micro-overfit`: passed (`12` rows, final loss `8.46417333377758e-06`).
- `make phase10r-memory`: passed.
- `make check`: passed (`157` CLI, `102` core, `9` integration, and `488` Python tests, plus
  build/Wasm checks).
- Phase 10R registry validation: passed; `33` approved training artifacts, `68` total artifacts.
- Phase 10R dry-run: passed; free space observed above the frozen `150 GiB` floor.
- Runner preflight: semantic, hash, source, disk, thermal, and proof gates passed; execution
  remained blocked by the two conditions below.

## Stop reasons

1. The repository has only the historical `OSAVAL01` evaluator. The frozen Phase 10R
   `OSAVAL02` sparse pair/triple model, training backend, and Rust/Wasm incremental feature
   parity are not implemented. Substituting the old MLP would violate the frozen architecture.
2. The full approved-source canonical/history split-leakage scan has not been run. The data
   foundation report explicitly records canonical/transposition overlap as unavailable until the
   canonical join is performed. The frozen preflight therefore cannot authorize training.

## Durable artifacts

- Runner: `training/open_shogi_training/phase10r_run.py`
- Runner tests: `tests/python/test_phase10r_run.py`
- Receipts and append-only events: `local/phase10r-runs/`
- Machine report: `local/phase10r-runs/PHASE10R_EXECUTION_REPORT.json`
- Human report: `local/phase10r-runs/PHASE10R_EXECUTION_REPORT.md`
- Latest preflight receipt: `local/phase10r-runs/20260821T141935.148033Z-preflight.json`

## Exact next command

Implement and hash-bind OSAVAL02, the sparse pair/triple training path, Rust/Wasm feature-key and
incremental/unmake parity, and the approved-source canonical/history leakage scan. Then rerun:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run preflight --root .
```

Do not run `prepare`, `train`, `arena`, `crossplay`, or `selfplay` until that command passes.
