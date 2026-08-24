# Phase 10R Training Backend Report

Status: **bounded backend validation passed; campaign blocked before the 1M rung**.

Observed in the latest exact preflight run:

- Command: `PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run preflight --root .`
- Receipt: `local/phase10r-runs/20260824T040306.201722Z-preflight.json`
- Receipt SHA-256: `86b3394c342f963688c4887e4bfc835e94651ad201b7e186443695022a0b74fc`
- Frozen hash validation: passed.
- Overall preflight: `blocked` with exit status `2`, solely because the approved-source
  leakage proof is incomplete and source replay has 18 rejected artifact streams.

## Frozen candidates and exact shapes

| Candidate | Parameters | Float32 OSAVAL02 bytes | Int8 OSAVAL02 bytes |
| --- | ---: | ---: | ---: |
| `sparse-pair-policy-wdl` | 2,413,321 | 9,657,380 | 2,417,417 |
| `factorized-pair-triple-policy-score` | 2,676,618 | 10,710,568 | 2,680,714 |

The backend instantiates the frozen OSAVAL02 feature/model contract, preserves source and
record identity in examples, validates legal-move masks and target masks, keeps ranking
teacher identity local to the approved teacher, and rejects unknown schemas, sources,
splits, targets, or unsupported combinations. It implements deterministic seeded training,
CPU fallback for unavailable MPS, bounded heavy-worker/resource checks, atomic resumable
checkpoints containing optimizer/scheduler/sampler and RNG state, early-stop/NaN/gradient
diagnostics, and non-overwriting float32/int8 OSAVAL02 export with parser-bound metadata.

The primary and pair candidates are the only executable training candidates. The legacy
OSAVAL01 path remains available only as a compatibility reader; it is not substituted for
OSAVAL02.

## Bounded validation

- Backend tests: `7 passed`.
- Scanner tests: `2 passed`.
- Runner tests: `3 passed`.
- Deterministic repeated training and exact checkpoint-resume state: passed.
- CPU/MPS fallback receipt: passed.
- Float32/int8 export and parse checks: passed.
- Preflight bounded checkpoint/export validation: passed.

No 1M-or-larger rung, full teacher labeling, production training, Arena, cross-play,
self-play, final holdout inspection, promotion, push, release, or deployment was started.

Machine-readable evidence is in
`artifacts/phase10r/phase10r-backend-validation-report.json`.
