# Phase 10R Split-Leakage Report

Status: **BLOCKED; not a training authorization**.

This is the preserved legacy partial-scan result. The 2026-08-24 repair contract and complete
collision ledger are in `PHASE_10R_IDENTITY_REPAIR_PLAN.md`, `PHASE_10R_WCSC_REPLAY_REPORT.md`,
`PHASE_10R_COLLISION_RESOLUTION.md`, and `artifacts/phase10r/phase10r-collision-resolution.json`.
No v2 full scan has been run yet.

The latest scan used the approved registry population and the exact command below:

```bash
PYTHONPATH=training uv run --frozen python \
  -m open_shogi_training.phase10r_run preflight --root .
```

The preflight receipt is
`local/phase10r-runs/20260824T040306.201722Z-preflight.json` with SHA-256
`86b3394c342f963688c4887e4bfc835e94651ad201b7e186443695022a0b74fc`. The scan input
manifest SHA-256 is
`94d3eec617f81ae68e6012f7f38819fe170423e3ea7587c2042bc30eef04eac8`.

## Coverage

Counts below are from replay-validated position records currently committed to the
disk-backed scan database. Record and game counts are distinct identity counts within
the completed streams.

| Source | Records | Unique games | Positions | Train | Validation | Source-held-out | Final holdout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AobaZero | 100 | 100 | 15,488 | 11,493 | 2,562 | 0 | 1,433 |
| Denryu hardware 3 | 121 | 121 | 22,616 | 17,939 | 2,179 | 2,498 | 0 |
| WCSC | 2,679 | 2,679 | 313,621 | 249,214 | 32,641 | 31,766 | 0 |
| **Total** | **2,900** | **2,900** | **351,725** | **278,646** | **37,382** | **34,264** | **1,433** |

The source-held-out game-group fractions for the external sources are Denryu
`13/121 = 0.107438...` and WCSC `273/2679 = 0.101903...`; both satisfy the frozen 10%
minimum. AobaZero retains its approved Phase 3 split rather than receiving a fabricated
source-held-out bucket.

Only 15 of 33 approved artifact streams completed. The final leakage manifest was
therefore not published. The durable partial manifest has SHA-256
`5d1d4afc9b1f1b696564ca8c54db76c47897d42952b16aaaccca580f8d7302a7`, and the completion
proof records 18 artifact-stream failures.

## Leakage classes

| Class | Identity | Count | Status |
| --- | --- | ---: | --- |
| Same record across splits | `record_id` | 0 | passed |
| Same game across splits | `game_id` | 0 | passed |
| Canonical position across splits | `canonical_position_id` | 3,327 | **failed** |
| Protected history across splits | `history_id` | 0 | passed |
| Different plies of one game in train/eval | `game_id` | 0 | passed |
| Prohibited transposition overlap | `transposition_key` | 0 | **unavailable** |
| AobaZero via GCT | lineage | 0 | not applicable |
| Floodgate via GCT/direct | lineage | 0 | not applicable |
| Distilled vs nodchip | lineage | 0 | not applicable |
| Arena start overlap | `arena_group` | 0 | not applicable |
| Public test entering training/selection | `canonical_position_id` | 0 | passed |
| Final holdout entering a forbidden path | `canonical_position_id` | 310 | **failed** |
| Source-held-out entering training | `source_id+game_id` | 0 | passed |
| Parser/normalization version collision | `canonical_position_id` | 214 | **failed** |

The transposition class is unavailable, not zero: the replay streams supplied no exact
transposition namespace. Canonical collisions were not deduplicated or relabeled; they
remain blocking evidence. The 310 final-holdout collisions likewise prevent any training
authorization.

## Allowed and prohibited overlaps

Same-split duplicate groups are reported separately and are not treated as cross-split
leakage:

| Identity | Same-split duplicate groups |
| --- | ---: |
| `record_id` | 2,899 |
| `game_id` | 2,899 |
| `canonical_position_id` | 4,158 |
| `history_id` | 2,899 |

Prohibited cross-split overlaps are the failed/unavailable classes above. The exact
per-row identity evidence is retained in the ignored partial JSONL manifest and SQLite
database; compact tracked summaries are in
`artifacts/phase10r/phase10r-prohibited-overlaps.json`,
`artifacts/phase10r/phase10r-allowed-overlaps.json`, and
`artifacts/phase10r/phase10r-scan-completion-proof.json`.

## Rejected streams

The following approved WCSC streams are not counted as complete because Rust CSA replay
failed closed: `wcsc02-kifu`, `wcsc04-kifu`, `wcsc05-kifu`, `wcsc06-kifu`, `wcsc07-kifu`,
`wcsc08-kifu`, `wcsc09-kifu`, `wcsc10-kifu`, `wcsc11-kifu`, `wcsc12-kifu`, `wcsc13-kifu`,
`wcsc14-kifu`, `wcsc15-kifu`, `wcsc16-kifu`, `wcsc17-kifu`, `wcsc22-kifu`, `wcsc26-kifu`,
and `wcsc32-kifu`. Reasons include malformed CSA ordering/blank or empty statements,
unsupported CSA versions, wrong-side moves, and illegal pawn drops. The exact file,
archive, input, revision, and observed checksums are preserved in the tracked input
identity manifest and completion proof.

Machine-readable evidence is in
`artifacts/phase10r/phase10r-split-leakage-report.json` and the accompanying scan
manifests/checkpoint files.
