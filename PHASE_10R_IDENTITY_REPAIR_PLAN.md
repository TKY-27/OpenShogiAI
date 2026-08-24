# Phase 10R-C2B identity and replay repair plan

Frozen: 2026-08-24 (Asia/Tokyo)

Status: **ready for Luna execution; not a training authorization**.

## Scope and invariants

This repair preserves the completed OSAVAL02 runtime/backend, model architecture, targets, source
rights decisions, statistical gates, 55% objective, protected holdouts, and all prior evidence. It
does not authorize training, Arena, self-play, cross-play, teacher labeling, or holdout evaluation.

The active approved population remains 33 artifacts: AobaZero exact100, WCSC1-29/31-32, and Denryu
hardware-3. No stream is removed. Five exact WCSC illegal-action members are excluded because the
official bytes themselves cannot complete a legal replay; the exclusion does not remove their
containing streams.

## Identity contract

`configs/phase10r/identity.yaml` and
`docs/data/phase10r-replay-proof.schema.json` define the v2 contract:

- exact official artifact hash and archive member path/hash;
- source-game ID bound to those bytes;
- complete legal replay from canonical starting position;
- source-independent canonical game hash;
- canonical position hash and domain-separated transposition key for every position;
- full-prefix, repetition-sensitive history identity for every position; and
- explicit parser and normalization versions.

Terminal/result spelling is bound in the proof but does not split otherwise identical canonical
play into different game identities. Source bytes are immutable. Unknown provenance, ambiguous
orientation, partial replay, internal NUL corruption, or illegal play quarantines the exact member.

## Deterministic execution order

1. Verify the repair commit, clean worktree, frozen hashes, disk floor, and no heavy worker.
2. Verify all 31 WCSC raw archives against their newly pinned sizes and hashes. Download only a
   missing/mismatched object from its exact official URL; reject any different live hash.
3. Produce a complete safe ZIP/LZH member inventory. Use one pinned extractor that supports every
   declared LZH method, including `-lh4-`; a tool error or unaccounted regular member fails the
   artifact. Retain member path, method, size, CRC when present, and SHA-256.
4. Classify every member as game input, documented metadata, or exact quarantined member. Flat
   filenames are forbidden as authoritative identities.
5. Parse CSA V1/V2/V2.1/V2.2/missing-version records under their declared or official compatibility
   profile, then emit an immutable-source-derived canonical V3 view. Apply only the frozen blank,
   trailing-NUL, multi-statement, and unique global KIF-orientation rules.
6. Replay every candidate game from its canonical initial position through every move with the Rust
   rule engine. Emit one v2 replay proof per accepted game. Do not trim illegal moves or invent a
   terminal/result.
7. Apply the five frozen exact-member exclusions. Any additional proposed exclusion must carry the
   same authoritative evidence and complete failed-replay receipt before the manifest and hashes
   are revised; no parser inconvenience is sufficient.
8. Replay/upgrade AobaZero exact100 and Denryu hardware-3 to the same canonical game, position,
   transposition, and full-prefix history identity contract. No source may retain a null
   transposition key or game-wide history substitute.
9. Assign protected game/history groups before position extraction using the frozen split policy.
   Preserve the Phase 3 final holdout and the at-least-10% WCSC/Denryu source-held-out groups.
10. Enumerate raw collisions, apply the frozen precedence and owner tie-break, retain an append-only
    decision ledger, and publish only effective canonical owners. Never reassign a losing record.
11. Run the full population scan exactly once into a fresh output directory. Publish its manifests
    only if all 33 streams are complete, every archive member is accounted for, and every leakage
    class is passed with transposition proof available.
12. Run bounded identity/scanner/runner/freeze validators. Update reports and frozen hashes, commit
    the C2B execution evidence, and stop before any training command.

## Required acceptance evidence

- 33/33 approved artifact streams complete under v2 replay accounting;
- 31/31 WCSC archive hashes equal the frozen registry;
- no unexplained or unaccounted archive member;
- every accepted game has one source-game ID and canonical game hash;
- every accepted position has canonical, transposition, and full-prefix history identities;
- zero same-game or protected-history cross-split collisions;
- every canonical/transposition/version collision has one frozen decision and one effective owner;
- zero protected final-holdout, source-held-out, or public-test occurrence in a forbidden effective
  path;
- zero unresolved parser/normalization collision;
- final WCSC source-held-out game-group fraction at least 10%; and
- frozen validation, focused identity/scanner tests, lint, and compilation pass.

Failure of any gate leaves preflight blocked. Statistical gates, architecture, targets, model
variants, runtime backend, and the 55% objective are unchanged.
