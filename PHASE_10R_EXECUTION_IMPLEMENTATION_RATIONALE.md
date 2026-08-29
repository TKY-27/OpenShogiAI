# Phase 10R execution implementation rationale

Date: 2026-08-27

The frozen Phase 10R execution prompt requires an exact `phase10r_run` command
interface, durable preparation artifacts, resumable training checkpoints, and
cross-runtime evaluation. The repository contained the bounded validation
backend and the frozen control/hash boundary, but did not contain the
large-scale command implementation. This change adds only the missing
execution plumbing over the already-passed v2 replay/leakage population:

- Rust supplies the authoritative legal-root and replay-history facts.
- Python joins those facts to effective SQLite identities and writes immutable,
  source-preserving preparation streams without reading the final holdout.
- Stage 1 and stage 2 train only the two frozen candidate variants with exact
  manifest-bound checkpoints; Apery-dependent stages remain fail-closed.
- Evaluation requires the frozen source-held-out, cross-runtime, and
  incremental-parity flags.

The existing hash-bound `phase10r_run.py` and its test are necessarily updated
because they expose the required command interface. The bounded
`phase10r_training.py` validator remains unchanged: terminal replay roots are
represented as an unavailable policy target rather than by weakening its
non-empty legal-mask invariant. The only control revision is the user-authorized
mixture reconciliation: conflicting maxima and relative weights become one
canonical `35%/45%/20%` target vector with the prior maxima retained as constraints.
The source registry, split assignments, model dimensions, target semantics, gate
thresholds, and public/promotion boundaries are unchanged. The hash manifests are
refreshed to bind the reviewed implementation and narrow control revision.

The first implementation choice is one deterministic streamed epoch per
factual stage, within the frozen per-rung maximum epoch limits. Preparation
uses deterministic largest-remainder allocation over normalized target shares and
records realized quotas and repetition pressure in its immutable manifest. The run stops before any
teacher-dependent hard-example labeling until the rights-gated Apery binary
and evaluation-file identities are present.

## 2026-08-29 ignored teacher-storage boundary repair

Restoring the exact rights-gated Apery install exposed an implementation-only
boundary mismatch: the frozen teacher configuration and setup procedure require
the ignored `local/teacher` root, while the repository boundary audit treated
that root as forbidden. The audit now explicitly skips that local payload root
and independently rejects any tracked path beneath it. The added regression
test proves both behaviors with a temporary repository. This changes neither
the teacher identity or options nor any Phase 10R data, model, split, target,
resource, offline, Arena, self-play, or promotion control; it only makes the
existing ignored installation contract executable under `make check`.

## 2026-08-29 teacher-binding sequence repair

The restored Apery environment exposed a second, separate execution gap. Both completed 1M runs
were intentionally stage-2-only pretraining checkpoints, and their receipts explicitly record
that teacher-dependent stages were not started. The frozen curriculum already requires ranking
and approved-teacher calibration, but the execution CLI had no command between evaluation and
select-hard to create their versioned teacher-bound children. The select-hard identity gate was
therefore correct; the sequence was incomplete.

`configs/phase10r/teacher-binding.yaml` now pins the exact pretraining parents, Apery binary/KKP/
KPP/options/search identity, score semantics, labels-v2 artifacts, same-split calibration rows,
zero-new-label budget, one-epoch stage-3 configuration, monotonic stage-4 calibration, versioned
outputs, and acceptance gates. A new closed lineage schema and live validator reject metadata-only
rebinding, parent/label/teacher drift, forbidden split use, unchanged pretraining weights, missing
parity, or failed gates. The original weights remain untouched. No architecture, model matrix,
source approval, protected split, target, threshold, resource, Arena/self-play, or promotion
control changes.

The frozen-control and implementation hash registries are refreshed because the new config,
schema, validator, tests, report, prompt, and narrow runner/gate integration are now part of the
reviewed boundary. `PHASE_10R_TEACHER_BINDING_REPAIR.md` records the per-file rationale and the
exact next Luna execution; no calibration or select-hard execution occurred in this repair.
