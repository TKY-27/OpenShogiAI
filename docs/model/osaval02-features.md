> Legacy OSAVAL02 compatibility specification; not a current training plan.

# Phase 10R feature architecture

Status: frozen on 2026-08-21 for implementation and long-running execution.

## Decision

The accepted evaluator is `factorized-pair-triple-policy-score`. The required ablation is
`sparse-pair-policy-wdl`; `dense-2287-control` is historical control evidence only. All tables
and weights are independently implemented and randomly initialized. No Bonanza, KPP, KKP,
relation table, third-party source code, or pretrained weight may be imported.

The failed Phase 10 candidates establish why another dense-width increase is not defensible. The
2,287-input 2x128 control obtained 904.03 cp MAE, 0.0967 Pearson correlation, 10.03% engine
best-move agreement, and 62.22% mate-sign accuracy. Adding attack planes and expanding to 3x256
obtained 931.31 cp, 0.0756, 10.68%, and 64.44%, while slowing search by 2.57x. The common defect is
a flat absolute representation with weak interaction structure trained on a narrow population,
not a shortage of dense parameters.

## Stable identities and orientation

Dataset identity remains canonical SFEN board, hands, and side to move with only the move number
removed. It is never mirrored or color-rotated for deduplication. The feature encoder derives
`us` and `them` relative to the represented side to move while retaining absolute SFEN square
order. Each emitted policy label is transformed only by the frozen move codec and decodes to the
original legal USI move.

The encoder must produce identical Python, Rust, and Wasm feature keys. Hashing uses a project
domain string, fixed seed `20260729`, explicit little-endian fields, and signed feature hashing.
The hash implementation and collision statistics are evidence, not tunable hyperparameters.

## Feature families

### King-relative board and hands

For every board piece, emit tokens relative to both kings. A board token contains piece owner
(`us`/`them`), promoted piece kind, king role (`our`/`opponent`), absolute king square, and piece
square. The bounded table is `2 × 14 × 2 × 81 × 81 = 367,416` rows. Kings participate as anchors
and as ordinary piece identities where a relation requires them.

For each nonzero hand count, emit owner, hand piece, count bucket, king role, and king square. The
count is part of the learned key; material and hand values are not hard-coded scores. The table is
`2 × 7 × 19 × 2 × 81 = 43,092` rows. Impossible count buckets remain unused.

### Two-piece relations

Enumerate the at-most 780 unordered board-piece pairs. The canonical pair key contains both piece
identities and squares, relative owners, promoted states, both king-relative offsets, and flags for
direct attack, direct defence, pin participation, checking participation, common attacked square,
and king-zone membership. Hand/board relations are emitted only for legal hand counts and their
potential drop zones. Keys map into 65,536 signed-hash buckets with eight learned values.

This is a learned relation representation, not a copied pair table. Hash load, collision rate,
feature count, and per-position feature checksum are reported at every data scale.

### Bounded three-piece relations

The primary model adds at most 256 deterministic tactical triples per position. It never
enumerates or allocates a full three-piece table. Triples are selected in this order:

1. king, attacker, defender inside each king's two-square zone;
2. king, pinned piece, pinner for every absolute pin;
3. king, checker, nearest legal defender for checks;
4. occupied combat square, strongest two attackers/defenders by stable piece/square order; and
5. local connected pieces within Manhattan distance four until the cap.

Duplicate keys are combined before lookup. The 32,768 signed-hash buckets have eight learned
values. Truncation at 256 is deterministic and counted; a truncation-rate increase is a pipeline
regression.

### Additional learned inputs

The 64 scalar/categorical inputs include side attack counts by square/zone, pseudo-mobility,
legal-move count, king-zone pressure counts, promoted/unpromoted board and hand counts, opening
phase derived from remaining material and ply, check and pin flags, repetition count, continuous-
check state, and explicit availability bits. These are position facts. No handcrafted term value,
handcrafted total, teacher score, or residual is an input.

History inputs require a complete history ending at the represented position. Native, USI, and
Wasm play must pass root history into search; make/unmake updates a reversible search-path history.
Standalone SFEN analysis encodes `history_available=0`, repetition count one, and no continuous-
check claim. A training row with missing history uses the same unavailable encoding. It must not
invent a history from the move number.

## Network and heads

Sparse embeddings are reduced by signed sum and normalized by emitted-feature count. Their
aggregates and the projected additional inputs enter two 128-unit ReLU layers. The pair model has
48 trunk inputs; the pair/triple model has 56. Runtime heads are WDL, transformed score where
enabled, mate kind/distance, and uncertainty. A factorized 16-dimensional move representation
scores only legal moves for policy/ranking.

Policy may alter move ordering only. It never adds to the leaf score. An Arena candidate that uses
policy ordering must enable it for all candidate games, and measured inference time includes legal
move encoding and policy scoring.

## Incremental runtime

`OSAVAL02` must store feature-domain versions, table dimensions, hash domain/seed, heads,
quantization, calibration, and a trailing SHA-256. Rust owns a reversible accumulator alongside
the search position. A move updates moved/captured/dropped piece tokens, hand buckets, king
anchors when a king moves, affected sliding rays, pins/checks, local pairs/triples, and history.
Unmake restores an exact delta. Tests compare incremental and full recomputation after every ply
of deterministic legal games.

The primary model has 2,676,618 parameters, a float32 artifact ceiling of 10,710,568 bytes, and an
int8 ceiling of 2,680,714 bytes. The browser float artifact stays below the existing 16 MiB load
limit. Advancement requires at least 3,000 search-context evaluations/second on the M5 and exact
integer score/move-index parity across Python, Rust, and Wasm.

## Runtime purity

Search terminal and mate scores remain search-owned. Qsearch uses the learned value only as the
stand-pat score of the represented leaf. The eligible runtime mode is `PureValue`; residual and
composite modes are prohibited. Uncertainty is reported and used for active-learning selection,
not mixed into centipawns. Calibration parameters are learned from validation supervision, frozen
before Arena, serialized in the model, and contain no handcrafted contribution.
