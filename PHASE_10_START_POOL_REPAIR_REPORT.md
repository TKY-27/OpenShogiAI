# Phase 10 Arena start-pool repair report

Date: 2026-08-21 (Asia/Tokyo)

Branch: `codex/phase10-execution`

Scope: Arena start-pool definition, population, provenance, leakage evidence, and frozen-control
binding only. No training, teacher call, self-play, Arena game, model export, promotion, registry
mutation, public-test acquisition, or candidate holdout evaluation was performed.

## Decision and root cause

This is **Case A**: `general_opening` was intended as the broad general-opening control population,
but the execution code implemented it as an exclusive residual after Ibisya/opponent-Furibisha
classification.

The approved non-test population contained 2,160 eligible rows below position index 24. The legacy
predicate assigned 1,925 of them to the two style groups first and left only 235 residual rows.
Those residual rows were dominated by repeated common openings and collapsed to 18 canonical
positions within the group (17 under the global first-winner implementation). This was a semantic
grouping error, not source-data insufficiency.

The required alternatives were tested separately:

1. **True source insufficiency:** rejected. The broad predicate has 512 canonical opening states
   before protection and 375 after protected-history and final-holdout exclusion.
2. **Duplicate records:** contributory, not primary. The 13,245 eligible non-test rows contain
   10,419 canonical positions, so canonicalization removes 2,826 repeated rows.
3. **Transpositions:** present but not the root cause. 760 canonical positions are reached through
   more than one move prefix; they are correctly represented once.
4. **Incorrect canonicalization:** rejected. Identity retains board, hands, and side to move and
   omits only the move number. No mirror or color rotation is used.
5. **Accidental predicate exclusion:** confirmed. The exclusive style checks removed 89.1% of the
   opening rows from the purported general-opening control before canonical deduplication.
6. **Umbrella/residual semantic mismatch:** confirmed as the primary cause. No frozen document or
   config required mutual exclusivity, and the broad control interpretation matches the named
   equal-weight Arena stratum.
7. **Leakage-prevention removal:** material but not blocking. History priority and final-holdout
   canonical exclusion remove 2,139 otherwise eligible rows, while leaving every group far above
   the 50-position minimum.

## Repaired design

Eligibility membership is overlapping:

- `general_opening`: `positionIndex < 24` umbrella control;
- `ibisha`: authoritative classifier returns `ibisha` for the retained source continuation;
- `opponent_furibisha`: authoritative classifier returns `ibisha-vs-furibisha`;
- `hard_middlegame_endgame`: the preserved legacy residual, `positionIndex >= 24` with an
  unclassified or Furibisha continuation.

One central manifest row is stored per canonical position. Overlap exists only in
`eligibleGroups`. Deterministic allocation then assigns a reserved position to exactly one Arena
reporting group. Allocation uses seed `20260821`, handles general opening first, caps each source
game at five positions per assigned group, and reserves 200 positions per group. The resulting 800
rows have 800 distinct position IDs and 800 distinct canonical identities.

Every row preserves the approved artifact ID, source ID, raw object SHA-256, game SHA-256,
position index, original split, history-group ID, initial SFEN, complete moves to the position,
current SFEN, side to move, source move, next SFEN, opening classification, eligibility tags, and
assigned group. All 800 current SFENs passed the Rust engine's `perft --depth 0` legality check and
all 800 rows have a complete move-history length equal to the source position index.

## Source and counts

Only `aobazero-no-noise-exact100` was used, through the approved
`aobazero-no-noise-pd-sample100` Phase 3 manifest. The positions artifact SHA-256 is
`d36cfc86757887f05a2d0c5ecc50e606a5836687cff32e757e268f8cf5cb627f`; the dataset manifest
SHA-256 is `db2a286ebfa4edd01c67041fb55d33b0f5f8813d24a2203d7c5e55a36111e243`. No pending or
denied source was used.

| Group | protected unique candidates | distinct reserve |
|---|---:|---:|
| general opening | 375 | 200 |
| Ibisya | 3,852 | 200 |
| opponent-Furibisha | 2,097 | 200 |
| hard middle/endgame | 3,886 | 200 |

## Exact overlap and leakage

Eligibility-group exact canonical intersections are:

| Pair | overlap |
|---|---:|
| general opening / Ibisya | 355 |
| general opening / opponent-Furibisha | 10 |
| general opening / hard middle/endgame | 0 |
| Ibisya / opponent-Furibisha | 0 |
| Ibisya / hard middle/endgame | 0 |
| opponent-Furibisha / hard middle/endgame | 0 |

The protected candidate population overlaps 8,155 Phase 3 train canonical identities and 1,948
validation identities. The 800-row reserve overlaps 674 train identities and 185 validation
identities; 59 reserve identities occur in both non-holdout source splits and retain the
validation-priority representative. Reuse of approved non-holdout source positions is disclosed
and does not expose a holdout result.

Exact canonical overlap with the legacy Phase 3 final test is 0 and protected-history overlap is
0. No candidate was evaluated on that holdout. Taya and dlshogi public-test artifacts remain
unauthorized and unacquired; their exact overlap is therefore **unmeasured**, not zero. They did
not populate the pool and were not inspected.

## Hash transition

The failed legacy builder did not publish a start-pool manifest, so its manifest SHA-256 is
`N/A (construction aborted at 17 general-opening positions)`. The repaired manifest SHA-256 is
`491a3d54d3d67002fc03fc3644c16b405f5b1dbb662e88e4af260bd45757a1f1`.

Supporting hashes:

- overlap/leakage report:
  `aa9b9a356b0bcc86dcd61ce2e932b73378128ae3c7f3d9ee464e60fdac556fb1`;
- legality report:
  `54fd60e557adaef2e1eb9666fae5df49fc1a9508f87c1c23a1621c3dd2eb720f`;
- Rust legality binary:
  `d2014423f2b9197eb1cb0c3d1332547b38bdabe8dc68c8aaff1249fcd9c35309`.

## Frozen controls changed

The only substantive frozen decision changed is the Arena start-pool definition and population:
membership is explicitly overlapping, general opening is explicitly an umbrella control,
holdout/history exclusion is explicit, and the deterministic 200/group reserve is frozen. The
split-policy text, frozen plan, Luna execution prompt, generator/validator, manifests, evidence
reports, and hash registry were revised to bind that decision.

No architecture, training target, active source/license decision, 80/10/10 source assignment,
48% entry gate, 55% objective, confidence requirement, worker/memory/disk budget, Arena time
control, promotion rule, or holdout-access allowance changed.

## Statistical validity

The repair restores the intended broad control while retaining style-stratified reporting.
Overlapping eligibility avoids falsely discarding ordinary openings, and globally distinct
allocation prevents a position from influencing more than one assigned group in a campaign.
Paired colors, equal group weights, deterministic ordering, one-pair maximum reuse, final-holdout
exclusion, and the 50-position integrity floor remain unchanged. The reserve supplies four times
the frozen 50/group final-objective need and enough distinct starts for 1,600 paired-color games,
so the repair does not gain power by duplicating or synthesizing observations.
