# Phase 10R-B — frozen pure-learned evaluator plan

Frozen: 2026-08-21 (Asia/Tokyo)

Branch: `codex/phase10r-curriculum`

Reference host: MacBook Pro M5, 24 GiB unified memory

## Immutable objective and claim

One pure learned evaluator must score at least 55% against the frozen
`handcrafted-experimental` evaluator in a final 1,600-game paired, reversed-color,
equal-wall-clock Arena. Both the Wilson half-draw and paired-color bootstrap 95% lower bounds
must be strictly above 50%. All 1,600 games must finish with zero illegal moves and zero
unexplained crashes. Search may retain rule, terminal, and mate logic, but no handcrafted
evaluation term, residual, blend, or fallback may contribute to a leaf score.

Project-owned code, independently implemented features/architecture, randomly initialized
weights, approved external records, and approved teacher labels are allowed. Third-party engine
code or pretrained weights, copied KPP/KKP/relation tables, external engines during play, and
handcrafted runtime scores are forbidden.

The precise claim is:

> Independent engine, feature encoder, model architecture and trained weights; approved external records and teacher evaluations were used as supervision.

It must not be described as rules-only self-play. Because every currently approved external
record has `derived_weights_permission: pending_permission`, this freeze permits local
experimental weights only. Publishing weights requires a later explicit rights decision and a
new freeze; it is not delegated to the execution agent.

## Diagnosis of the failed Phase 10 campaign

The preserved campaign stopped correctly before Arena. The 2x128 control failed raw cp MAE
(904.03 > 870), Pearson (0.0967 < 0.40), engine best-move agreement (10.03% < 18%), and mate-sign
accuracy (62.22% < 90%). The 3x256 attack-map model also failed at 931.31 cp, 0.0756, 10.68%, and
64.44%, and missed the throughput/slowdown gates at 2,633 evaluations/second and 2.57x slowdown.

The root-cause hypothesis has four linked parts:

1. The flat absolute 2,287-vector makes king/piece and piece/piece interactions expensive to
   learn; another dense width increase did not create the missing relational inductive bias.
2. Only 6,779 train and 1,854 validation examples from a narrow 100-game lineage entered the
   failed execution. Style MAE was about 1,436 cp in hard middle/endgames and 1,523 cp against
   opponent Furibisha, versus 452 cp in the dominant Ibisya stratum.
3. WDL, cp, mate, ranking, and a binary agreement diagnostic shared one scalar trunk/output
   schedule. Although masked, they did not preserve full policy/value/mate source semantics, and
   the old “policy” head did not predict moves.
4. Offline loss was a weak proxy for search. Phase 9 had already shown pure models losing 0-20
   equal-clock games while the residual diagnostic preserved the handcrafted evaluator's
   structure. The redesign must learn that structure from position relations, not add it at
   runtime.

Pipeline bugs remain a first-class risk. The new plan blocks scale-up unless tiny overfit,
perspective/sign, mate, scale, move codec, history, leaf semantics, split, and sampling tests pass.

## Accepted architecture

`factorized-pair-triple-policy-score` is the primary. It uses independently implemented sparse
king-relative board/hand embeddings, hashed two-piece relations, at most 256 selected tactical
triples, learned tactical/history inputs, a small 2x128 trunk, factorized legal-move policy, WDL,
transformed score, mate, ranking, and uncertainty heads. It has 2,676,618 parameters with frozen
float32/int8 ceilings of 10,710,568/2,680,714 bytes. It must reach at least 3,000 M5 search-context
evaluations/second and fit the 16 MiB browser artifact boundary.

`sparse-pair-policy-wdl` is the only trained ablation. It removes triples and the direct score
head, has 2,413,321 parameters, and calibrates WDL log-odds. `dense-2287-control` is read-only
historical control evidence and cannot be a finalist. Full definitions, incremental-update rules,
history availability, policy cost accounting, and `OSAVAL02` requirements are in
`docs/model/PHASE10R_FEATURE_ARCHITECTURE.md` and `configs/phase10r/model-matrix.yaml`.

## Pipeline sanity gate

Before any rung, all of these must pass:

- intentional 12-row multi-head overfit below 0.001 loss;
- Black/White current-side WDL and score-sign fixtures;
- mate kind/sign/distance separated from cp;
- `FV_SCALE`, legacy output scale, and canonical cp never conflated;
- 13,689-class move codec bijection with project-rule legal masks;
- explicit promotion and drop round trips;
- complete history ends at the represented position and produces matching history facts;
- qsearch evaluates the exact stand-pat leaf; PV/root labels never move to descendants;
- canonical/history identities cannot cross train/validation/final/reserved splits; and
- deterministic sampling counts equal configured source weights.

The bounded commands are `make phase10r-sanity` and `make phase10r-micro-overfit`. The runtime
implementation must add Python/Rust/Wasm feature parity, legal-root uniqueness, and incremental
versus full-recompute/unmake parity before training. Any failure stops all large workers.

## Source-preserving curriculum

No pending, denied, holdout, or local-prior-art source is admitted. The 33 approved artifacts are
the exact AobaZero catalog, WCSC1–29/31–32, and Denryu hardware-3. Family policy applies to every
listed artifact in `dataset-mixture.yaml`.

| Source lane | Role | Max stage fraction / weight | Dedup priority | Public weights |
| --- | --- | ---: | ---: | --- |
| AobaZero exact100 | behavior policy, factual WDL, representation, teacher pool | 35% / 1.0 | 700 | no; derived rights pending |
| WCSC1–29/31–32 | behavior policy, factual WDL, representation | 55% / 1.0 | 600 | no; derived rights pending |
| Denryu hardware-3 | behavior policy, factual WDL, representation | 20% / 0.5 | 500 | no; derived rights pending |
| frozen OpenShogiAI Apery labels | cp, ranking, mate, active learning | 35% / 1.0 | 800 | no; inherits record/teacher review |
| bootstrapped cross-play | WDL and selected teacher pool | 25% / 0.5 | 400 | no under this freeze |
| gated pure self-play | WDL and behavior policy | 40% / 0.5 | 300 | no under this freeze |

WCSC attribution retains program, date, event, and official source; Denryu retains event,
program, date, and source; AobaZero retains project/source-host attribution. Teacher records bind
binary/evaluation hashes, options, nodes, MultiPV, score kind/perspective, and complete legal PVs.

Training is staged, never one concatenated mixture:

1. behavior-policy/representation pretraining on approved legal played moves;
2. source-specific factual WDL/value-probability pretraining;
3. PackedSfenValue/ranking stage—PSV stays disabled because none is approved; Apery ranking is
   active;
4. calibration with the frozen approved OpenShogiAI Apery teacher;
5. train-only hard-example fine-tuning capped at 25% of a rung;
6. equal-wall-clock screening;
7. bootstrapped candidate/handcrafted/older-learned cross-play after the short screen; and
8. pure candidate self-play only after the 400-game candidate score is at least 48%.

Target availability, masks, transforms, move codec, calibration, mate handling, and qsearch/PV
semantics are frozen in `docs/model/PHASE10R_TARGET_SEMANTICS.md` and
`configs/phase10r/target-semantics.yaml`.

## Progressive scale and active learning

The streamed-example rungs are 1M, 10M, 50M, 100M, 500M, and at most 1B. Examples stream from
immutable shards with resumable sampler state; the full population is never materialized locally.
Teacher-label caps progress through 10k, 100k, 500k, 1M, 3M, and 5M unique positions. Later rungs
raise selected hard-position nodes from 25k to 50k and 100k and MultiPV from 3 to at most 8.

Active learning ranks train-only unique positions by model/teacher disagreement,
candidate/handcrafted disagreement (selection only; never a handcrafted target), unstable
MultiPV, tactical/mate errors, source-held-out failure, and calibrated uncertainty. It caps one
game at 32 rows, one history at eight, and one signal at 40%.

Expansion requires validation improvement of at least 0.2% relative, no active-source-held-out
regression over 0.01, no ECE regression over 0.01, zero tactical regression, variant throughput,
no Arena lower-bound regression over 0.02, and resource receipts. Two consecutive sub-threshold
rungs without Arena lower-bound improvement are a plateau: restore the best checkpoint and stop
expanding. Time alone is never a stop condition and lower offline loss never promotes a model.

## Arena, cross-play, and self-play

All strength games use the frozen 800-start manifest, globally unique allocation, reversed colors,
opening disabled, depth cap 8, 32 MiB hash per player, 128 plies, and equal wall clock. Capped or
incomplete games are not silently draws. Every report includes Wilson half-draw and 100,000-
resample paired-color bootstrap intervals.

| Gate | Games / pairs | Frozen requirement |
| --- | ---: | --- |
| short | 40 / 20 | score ≥40%; zero illegal/crash/tactical regression |
| 400 / self-play entry | 400 / 200 | score ≥48%; both lower bounds ≥43%; zero failures |
| 800 | 800 / 400 | score ≥52%; paired lower ≥48%; zero failures |
| practical | 40 / 20 at 20 s/move | score ≥48%; zero failures/deadline violations |
| final | 1,600 / 800 | score ≥55%; both 95% lower bounds >50%; zero illegal/unexplained crash |

Before 48%, bootstrapped games may be candidate versus handcrafted or older learned models and
decisive errors may be teacher-labeled. Handcrafted scores are never targets. Pure self-play begins
only after the 400-game entry pass. Two generations with under 0.5 percentage-point Arena gain
and no calibration gain are a plateau. Any illegal move, unexplained crash, tactical regression,
source-held-out regression over 0.01, or Arena regression over 0.02 quarantines the generation and
restores the last passing checkpoint.

No final holdout is inspected before final Arena selection. One already-defined Phase 3 legacy
final-holdout inspection is allowed only for the completely frozen finalist; its result cannot
select, tune, or retrain. Recent WCSC/Denryu and the public evaluation boundary remain unread.

## Resource and execution freeze

At most two heavy workers may exist, specifically at most one trainer and one teacher or Arena
worker; training and teacher are not concurrent. Aggregate RSS targets at most 16 GiB. Warning is
14 GiB; critical memory pressure checkpoints then stops, with immediate resumable stop when the
OS reports critical pressure. Free disk must remain at least 150 GiB and is checked before every
stage and each streamed GiB. There is no wall-clock deadline.

Jobs use deterministic manifests, periodic 10,000-step checkpoints, best-two plus last retention,
and resumable sampler/optimizer state. Serious thermal pressure checkpoints and stops; sustained
throttling reduces to one heavy worker. Arena runs one game at a time. The execution agent launches
a process and waits on completion/checkpoint events; a polling-heavy agent loop is forbidden.

## Frozen verification and execution boundary

This architecture phase runs only:

```sh
make phase10r-validate
make phase10r-sanity
make phase10r-micro-overfit
make phase10r-memory
make check
```

It performs no bulk acquisition, long training, large teacher labeling, Arena, cross-play,
self-play, model promotion, push, publication, or deployment. The complete future execution prompt
is `prompts/LUNA_PHASE10R_EXECUTION.md`. Luna may implement and run the frozen pipeline but may not
alter architecture, features, source approvals, split/holdout policy, thresholds, scale rungs,
promotion policy, or hashes.
