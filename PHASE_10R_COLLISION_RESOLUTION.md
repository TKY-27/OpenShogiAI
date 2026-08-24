# Phase 10R collision resolution

Status: **rules frozen; legacy partial evidence fully enumerated; final v2 scan not run**.

## Diagnosed evidence

The legacy partial database contains:

- 3,327 canonical-position identities present in more than one split;
- 310 of those identities touching the protected Phase 3 final holdout;
- 214 parser/normalization-version collisions, of which 36 occur within one split and 178 also
  participate in a cross-split group; and
- 3,363 identities in the union of canonical cross-split and version-collision groups.

Every one of the 3,363 groups is recorded in
`artifacts/phase10r/phase10r-collision-resolution.json`. The artifact contains the observed splits,
sources, artifacts, parser/normalization pairs, occurrence count, protected-holdout flag, winning
split, deterministic owner, and exclusion count. It contains hashes and identities, not holdout
labels or evaluation results.

Because this is legacy v1 evidence, its owner rows label the old scan identity as
`legacy_game_id`; the final replay must replace that field with the authoritative v2
`source_game_id` before applying the same tie-break.

The legacy split combinations are:

| Splits | Groups |
| --- | ---: |
| train + validation | 1,169 |
| source-held-out + train | 1,163 |
| source-held-out + validation + train | 587 |
| source-held-out + validation | 98 |
| final-holdout + train | 145 |
| final-holdout + validation + train | 101 |
| final-holdout + validation | 15 |
| final-holdout + source-held-out + train + validation | 44 |
| final-holdout + source-held-out + train | 5 |
| same-split parser/normalization only | 36 |

The collisions are mainly ordinary openings reached by different games that were independently
assigned to splits. They are real leakage under the frozen zero-cross-split standard; they are not
false positives and are not waived.

## Frozen resolution rule

Resolution uses the existing frozen precedence exactly:

`public_test > reserved_holdout > final_holdout > source_held_out > validation > train`

For each canonical identity:

1. retain only occurrences in the highest-precedence observed split;
2. within that split, select the owner by lexicographic
   `(artifact_id, source_game_id, record_id, position_index)`;
3. exclude every other occurrence without changing its original split;
4. never copy a label, result, or record from a protected split into training;
5. retain the raw occurrence and the decision in the audit manifest; and
6. publish only the effective owner to the deduplicated scan/training manifest.

This rule protects all 310 final-holdout collisions: the final-holdout occurrence wins and every
train, validation, or source-held-out occurrence of that canonical position is excluded. Zero
final-holdout records move to training.

Same canonical game or protected history across splits remains a hard error rather than an
auto-resolution. A parser or normalization version difference is resolvable only when every input
has an exact replay proof and uses the same v2 canonical identity schema; split precedence is
applied first, then the owner tie-break. Unknown or ambiguous identity blocks the scan.

## Transposition proof

The old scanner reported transposition proof as unavailable because all replay rows supplied
`null`. The v2 proof does not use a process-local or engine-specific Zobrist value.

- `canonical_position_hash` hashes canonical SFEN board, hands, and side-to-move, omitting move
  number, in domain `open-shogiai/phase10r/canonical-position/v2`.
- `transposition_key` hashes the same canonical SFEN in the independent domain
  `open-shogiai/phase10r/transposition/v2`.
- `history_id` hashes canonical initial SFEN plus the complete legal USI move prefix through the
  represented position in domain `open-shogiai/phase10r/history/v2`.

Thus two legal histories reaching the same position always share a transposition key, while
repetition-sensitive histories remain distinguishable. The full prefix is sufficient to derive
repetition occurrence and checking history. A first-24-move prefix or one game-wide history ID is
not accepted.

The final v2 scan passes only when every effective position has all three identities, every raw
cross-split transposition is covered by a precedence decision, and the effective manifest has zero
canonical, transposition, final-holdout, protected-history, game, or parser/normalization leakage.
