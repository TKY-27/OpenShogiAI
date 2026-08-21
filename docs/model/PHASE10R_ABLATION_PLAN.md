# Phase 10R ablation plan

The matrix is intentionally small. One change is attributed at a time; there is no width, hash-
size, embedding-dimension, seed, or loss-weight sweep.

## Matrix

| Order | Variant/comparison | Question | Advancement |
| ---: | --- | --- | --- |
| 0 | `dense-2287-control` historical evidence | Does the old flat representation meet the new gate? | never final-eligible |
| 1 | `sparse-pair-policy-wdl` | Do king-relative and pair relations fix generalization at acceptable speed? | all sanity/offline/equal-clock gates |
| 2 | add bounded local triples + direct score head | Do tactical triples improve mate/tactical/source-held-out and Arena evidence? | same gates; offline loss alone is insufficient |
| 3 | primary with policy ordering disabled | Is strength coming from value or policy ordering? | diagnostic 40 games only |
| 4 | primary full-recompute vs incremental accumulator | Is the optimization exact? | scores, logits, keys, and unmake state must be bit-identical |
| 5 | float32 vs int8 | Is browser-size quantization acceptable? | separate 400 paired games; no automatic substitution |

Policy-disabled and full-recompute comparisons reuse the same checkpoint and are not new training
variants. The only trainable candidates are pair and pair/triple, one deterministic seed each.

## Mandatory pre-training gates

`make phase10r-sanity` and `make phase10r-micro-overfit` must pass before every scale starts. They
cover intentional tiny-set memorization, side-to-move and score sign, mate separation, `FV_SCALE`
isolation, move-index bijection, promotion/drop labels, legal-label evidence, represented history,
qsearch leaf semantics, split isolation, and source-weight application. Later implementation must
extend the same gate with Python/Rust/Wasm feature-key parity and incremental/full recomputation.

Any failure stops every large worker. It is not waived because a validation metric improved.

## Offline and scale-boundary metrics

Each active source reports policy top-1/top-5, WDL log loss/Brier, calibration ECE, and source-
held-out results. Apery rows additionally report transformed-score loss, raw cp MAE/Pearson,
candidate ranking, mate direction/distance, and tactical strata. Runtime reports float/int8 size,
full/incremental feature counts, truncation/collision rates, evaluations/second, search slowdown,
and browser peak memory.

A source-held-out model omits the entire AobaZero, WCSC, or Denryu family, not a random position
sample. Improvement must not come from one source, opening prefix, event, or repeated game.

## Promotion and rollback logic

A candidate advances only if its current data-scale gate passes and no source-held-out,
calibration, tactical, latency, memory, or Arena lower-bound regression exceeds the frozen limits.
Two consecutive rungs below 0.2% relative primary validation improvement with no Arena lower-bound
gain are a plateau. Roll back to the best checkpoint and stop expansion. Time spent training is
neither success nor failure.

The Arena ladder is 40, 400, 800, practical 20-second, then 1,600 paired reversed-color games.
The final candidate needs at least 55% score and both frozen 95% lower bounds strictly above 50%,
with zero illegal moves and zero unexplained crashes. Final holdout inspection occurs only after
the candidate and all selection decisions are frozen and cannot trigger retraining.
