# Phase 10R-C2A Progress

Updated: 2026-08-24 (Asia/Tokyo)

## Current state

The campaign is **blocked before the 1M rung**. The separate frozen OSAVAL02 backend and the
disk-backed approved-population scanner are implemented and bounded-validated, but preflight
does not authorize a training rung because the full split-leakage proof is incomplete.

The actual starting state was branch `codex/phase10r-osaval02-parity` at closure HEAD
`5b8f85f6862eff8ee1216c37b90bf0b0361570bf`. Work continued on
`codex/phase10r-preflight-closure`. The final three local commits are intended to leave that
branch clean.

No 1M-or-larger rung, full teacher labeling, production training, Arena, cross-play, self-play,
final holdout inspection, promotion, push, release, or deployment was performed.

## Checks completed

- Baseline `make check`: passed before Phase 10R-C2A source changes.
- Baseline `make phase10r-osaval02-parity`: passed (`28` tests).
- Baseline `make phase10r-validate`: passed (`33` approved artifacts, `9` configs, `46` frozen
  hashes).
- Backend tests: passed (`7` tests); scanner tests: passed (`2` tests); runner tests: passed
  (`3` tests).
- Ruff checks and Python compilation for changed Python modules: passed.
- Final `make check`: passed (`525` Python tests, Rust workspace tests, and build/Wasm checks).
- Latest exact preflight: `blocked`; frozen hash validation, pipeline sanity, micro-overfit,
  memory, disk, thermal, runtime, and bounded backend checks passed.
- Latest preflight peak RSS: `381,583,360` bytes; observed free disk:
  `331,112,599,552` bytes against the frozen `161,061,273,600`-byte floor.

## Durable artifacts

- Backend implementation: `training/open_shogi_training/phase10r_training.py`
- Scanner implementation: `training/open_shogi_training/data/phase10r_scan.py`
- Backend/scanner/runner tests under `tests/python/`
- Human reports: `PHASE_10R_TRAINING_BACKEND_REPORT.md` and
  `PHASE_10R_SPLIT_LEAKAGE_REPORT.md`
- Machine reports and scan identities under `artifacts/phase10r/`
- Immutable preflight receipts and append-only events under `local/phase10r-runs/`

## Stop reasons

1. Only `15/33` approved artifact streams completed replay; `18` streams remain rejected by
   fail-closed CSA validation.
2. The completed population contains `3,327` canonical cross-split collisions, `310` final
   holdout forbidden-path collisions, and `214` parser/normalization collisions.
3. Exact transposition proof is unavailable because no replay stream supplied the required
   transposition namespace.

These findings are recorded without deduplication, relabeling, or split mutation. They block
training authorization.

## Exact next command

After the rejected streams are resolved with approved, checksum-bound replay inputs and the
canonical/final-holdout/parser/transposition findings are resolved or reviewed under the frozen
policy, rerun exactly:

```bash
PYTHONPATH=training uv run --frozen python \
  -m open_shogi_training.phase10r_run preflight --root .
```

Do not run `prepare`, `train`, `arena`, `crossplay`, or `selfplay` while this command is blocked.
