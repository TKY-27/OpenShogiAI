# Phase 10R mixture reconciliation report

Date: 2026-08-27 (Asia/Tokyo)

Status: **reconciled; replacement 1M preparation passed; production training not started**

## Root cause and authoritative interpretation

The 2026-08-21 freeze expressed two different active-source controls in the same rows:

- maximum stage shares: AobaZero/WCSC/Denryu = `35%/55%/20%`;
- relative sampling weights: `1.0/1.0/0.5`.

The allocator and execution implementation normalized only the weights, producing
`40%/40%/20%`. That made the AobaZero realized share five percentage points larger than its
explicit frozen maximum. Git history contains no independent target-share vector: the plan/config
introduced maxima and weights together, the sanity test later froze `4/4/2`, and the preparation
implementation copied those weights.

The authoritative intent was therefore relative weights constrained by maximum shares, but the two
controls were internally inconsistent. The user-authorized reconciliation replaces them for the
active pretraining sources with one canonical definition containing explicit targets and maximum
constraints.

## Decision

The canonical target is:

| source | target | maximum | 1M target/realized count |
| --- | ---: | ---: | ---: |
| AobaZero | 35% | 35% | 350,000 / 350,000 |
| WCSC | 45% | 55% | 450,000 / 450,000 |
| Denryu | 20% | 20% | 200,000 / 200,000 |

The user-provided default `35%/45%/20%` was used. No stronger frozen target vector exists, and this
is the minimum adjustment of the old implied `40%/40%/20%`: clamp AobaZero at 35%, leave Denryu at
its 20% ceiling, and transfer the five-point excess to WCSC. It is not obtained by normalizing the
old weights.

Corpus pressure was evaluated rather than used to invent a new objective. The raw source
populations are AobaZero 15,488, WCSC 618,038, and Denryu 25,993 positions; the effective published
population is 528,570 unique records. A corpus-proportional vector would be materially different
from every frozen curriculum control. A `25%/55%/20%` vector would reduce AobaZero repetition more,
but would be a larger, unsupported semantic revision. The selected minimum correction nevertheless
reduces AobaZero eligible-train repetition from 51.1182 to 44.7284 epochs while increasing the large
WCSC lane only from 1.1229 to 1.2633; Denryu remains 17.4429.

## Canonical control

`configs/phase10r/dataset-mixture.yaml` schema v2 now owns the active pretraining mixture. It
requires:

- target shares summing exactly to decimal `1.0`;
- optional minimums and explicit maximums, with target and realized shares inside them;
- deterministic largest-remainder count allocation, tie order
  `aobazero, wcsc, denryu`, and zero count tolerance;
- deterministic deficit-round-robin ordering with replacement at seed `20260729`;
- equal per-example loss weight after source sampling;
- epoch-equivalent and maximum-record-occurrence reporting against the eligible unique train stream;
- immutable versioned outputs and realized-count validation before publication.

Teacher, cross-play, and self-play stage caps remain separate gated-stage controls. No model,
feature, target, split/holdout, source approval, offline/Arena/self-play/promotion gate, resource
limit, or final 55% objective changed.

## Historical invalid preparation

The original preparation remains byte-for-byte at:

`local/phase10r-data/phase10r-prepared/1m/`

- manifest file SHA-256: `298654d99e04383a8a63ccca837ff9095ecba45e2bb7aadf747262c8fe7d04c5`
- declared manifest-body SHA-256: `5f83994dfc63f4d15a16fa1e17586e107bc06a20e63435b0f40da5872e7cd4c1`
- train SHA-256: `5ab54aaa8fc59c5083f4037e1c9cb4b811164f09c9da225d0eda413d837459c0`
- realized counts: AobaZero 400,000; WCSC 400,000; Denryu 200,000
- realized shares: `40%/40%/20%`

Its adjacent immutable marker is
`local/phase10r-data/phase10r-prepared/1m/REJECTED_MIXTURE_CONTROL_CONFLICT.json`, SHA-256
`d10627da5fe79efc0662d56f2976047b5264201f831c4a7b5d19b6c2ab70b7bf`. Neither the old
manifest nor any old data file was deleted or overwritten.

## Replacement preparation

The replacement is at:

`local/phase10r-data/phase10r-prepared/1m-mixture-v2/`

- manifest file SHA-256: `dcbcfd7584f84bd03d9d948a5633e9b18bb3db06027d1f1a0bec2b2cd7a8e076`
- declared manifest-body SHA-256: `6f7c244e9cba098fed99daef12f11962b2e36c546012f058c961daada28cdd48`
- train SHA-256: `8748f5fdf8af7716a347f1faa94d0f4965afca009f368ce1a66463e0e917b161`
- train bytes/rows: 2,077,794,499 / 1,000,000

| source | raw source positions | effective unique all | effective unique train | eligible unique after game cap | epoch-equivalent | max occurrences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AobaZero | 15,488 | 12,237 | 8,765 | 7,825 | 44.7284 | 45 |
| WCSC | 618,038 | 494,484 | 392,158 | 356,214 | 1.2633 | 2 |
| Denryu | 25,993 | 21,849 | 17,051 | 11,466 | 17.4429 | 18 |

The same seed reproduced the exact train hash and byte/row counts. The bound scan proves zero
effective cross-split/transposition groups and excludes public/internal/final holdouts. All config,
scan input, collision-decision, transposition-proof, SQLite, helper, base-stream, evaluation-stream,
and train hashes are recorded in the manifest. Peak preparation RSS was 238,813,184 bytes against
the 17,179,869,184-byte target, and post-preparation free disk was 283,567,955,968 bytes against the
161,061,273,600-byte floor.

## Changed frozen controls

- `PHASE_10R_FROZEN_PLAN.md`: target/max table and reconciliation addendum.
- `configs/phase10r/dataset-mixture.yaml`: schema v2 canonical mixture.
- `training/open_shogi_training/phase10r.py`: exact-share validation and allocation.
- `training/open_shogi_training/phase10r_execution.py`: versioned preparation, rejection evidence,
  realized-count/repetition/input-hash/resource/determinism validation.
- `training/open_shogi_training/phase10r_campaign.py`: consume only the validated v2 path.
- `training/open_shogi_training/phase10r_run.py`: include mixture/preparation controls in preflight.
- `tests/python/test_phase10r_freeze.py` and `tests/python/test_phase10r_execution.py`: regression tests.
- `prompts/LUNA_PHASE10R_EXECUTION.md`: report canonical targets and realized shares.
- `configs/phase10r/frozen-controls.sha256` and
  `configs/phase10r/phase10r-implementation.sha256`: refreshed bindings.

## Verification

Focused mixture, deterministic-stream, immutable-rejection, repetition, and execution tests passed
(`25` tests). Frozen-hash validation passed (`60` paths). Leakage validation passed with `528,570`
effective records and zero effective cross-split groups. Full `make check` passed (`544` Python
tests, `344` Rust tests, lint, boundary, license, provenance, build, and deterministic Wasm binding
check). Exact Phase 10R preflight passed with no failures; its immutable receipt and SHA-256 are
recorded in `artifacts/phase10r/phase10r-mixture-reconciliation.json`.

No production training, teacher labeling, Arena, cross-play, self-play, final holdout evaluation,
promotion, push, or deployment was run.
