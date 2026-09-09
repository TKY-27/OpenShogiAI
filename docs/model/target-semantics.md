> Legacy OSAVAL02 compatibility specification; not a current training plan.

# Phase 10R target semantics

This document is normative together with `configs/phase10r/target-semantics.yaml`.

## Perspective and score sign

Every target is attached to the exact represented position. Factual Black/White results are
converted to loss/draw/win only after legal replay establishes the represented side to move.
A teacher root score is already from that root's current-side perspective. A qsearch or PV leaf
may use a score only if the label identifies that exact leaf and its current side; root scores may
not be copied onto descendants. Negation belongs to negamax recursion, not the feature encoder or
leaf evaluator.

The pipeline gate contains paired Black/White sign fixtures and rejects unknown perspectives.
History-bearing records must contain a replayable path ending at the represented position.

## Separate heads and masks

| Head | Target | Sources | Missing/incompatible behavior |
| --- | --- | --- | --- |
| behavior policy | played legal move | approved AobaZero/WCSC/Denryu; gated self-play | mask; do not call an unobserved move negative |
| WDL | factual loss/draw/win probabilities | approved complete games and generated games | mask unknown/capped outcomes |
| score | signed `log1p(abs(cp))` transform | frozen OpenShogiAI Apery lane | mask mate, unknown scale, and other engines |
| mate kind | mated/no-label/mating | same teacher lane | mask ordinary cp rows for mate direction |
| mate distance | `log1p(plies)` | signed mate rows | mask cp and opposite-sign comparisons |
| ranking | legal teacher candidates | same teacher/config only | do not compare across teachers or missing candidates |
| uncertainty | score variance / WDL calibration | lanes with the matching target | never substitute for a missing target |

Behavioral played moves, teacher best moves, MCTS visits, and candidate rankings are different
targets. No approved MCTS distribution exists in this freeze. The behavioral policy loss is
one-hot within the legal mask and is reported by source so it cannot silently become a claim about
optimal play.

## Move labels

The move space has 13,689 bijective classes. Normal moves use
`((from × 81 + to) × 2 + promotion)` for 13,122 classes. Drops follow at offset 13,122 with piece
order `R,B,G,S,N,L,P`, then target square. Squares scan ranks `a` through `i`, within each rank
files `9` through `1`. Promotion and drop markers are never inferred from the position. Every
training and runtime policy operation applies a legal-move mask produced by the project rule
engine. Python/Rust/Wasm round trips and complete legal-root uniqueness are mandatory gates.

## Centipawns, mates, and `FV_SCALE`

Only the frozen Apery configuration supplies the active centipawn lane. Its exact binary,
evaluation-table hashes, options, nodes, MultiPV, and source position are retained. Ordinary
scores are clipped at 3,000 cp inside that lane and transformed by
`sign(cp) * log1p(abs(cp)) / log1p(3000)`. Calibration back to OpenShogiAI score space is a
monotonic affine fit on transformed validation scores or WDL log-odds and is frozen before Arena.

Mate is never converted to ±3,000 cp. Search owns the mate namespace and terminal distance.
Static heads learn direction and distance only for diagnostics, ordering, and tactical
representation.

`FV_SCALE` is source/model metadata, not a synonym for centipawns, the 1,200 cp legacy output
scale, or the new calibration transform. No PackedSfenValue source is approved. Its stage exists
but is inactive: raw integers, perspective, qsearch/PV-leaf semantics, teacher, and `FV_SCALE`
would remain source-local until a new human-authorized freeze supplies an official conversion or
chooses source-local supervision.

## Calibration and runtime score

The pair model calibrates WDL log-odds to score. The primary model jointly fits WDL and transformed
score, then freezes a monotonic calibration using validation only. Calibration coefficients,
fit-manifest hash, source lane, and metrics are serialized. They may not be refit after an Arena
result. Final output is clamped below the search mate threshold.

Policy inference cost is part of playing-strength evaluation whenever policy ordering is active.
Neither policy, uncertainty, nor handcrafted evaluation is added to the leaf score.

## Public statement

The only permitted precise claim is:

> Independent engine, feature encoder, model architecture and trained weights; approved external records and teacher evaluations were used as supervision.

Do not call the resulting model rules-only self-play. Current source-registry permissions allow
local experimental weights but leave derived-weight publication pending; therefore no Phase 10R
weight may be published under this freeze.
