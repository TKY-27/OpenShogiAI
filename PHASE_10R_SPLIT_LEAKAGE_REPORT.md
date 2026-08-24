# Phase 10R Split-Leakage Report

Status: **PASSED for the repaired replay/identity scan; not a training authorization.**

This report records the completed Phase 10R-C2B replay and full external-memory v2
split-leakage scan. All 33 approved streams completed. No stream was excluded. The
candidate-level exclusions are the frozen C2B boundary decisions and exact failed
replay receipts.

## Frozen identities

- Replay scan identity: `6ad465f9c36b96132bd54192470774416b47f14d739f9827a6f3b0145c45bc18`
- Replay completion SHA-256: `77296e5dbf140af7f8a85e151879a61c161fad82d899190486b6135902a86df4`
- Scan input manifest SHA-256: `21b36a06722aed5350add5ea9a1e373add3cca830255ea01b31dc7e87e6620d6`
- Effective scan manifest SHA-256: `42b47eb92951669fad88a3011a4e8671cbcf4793f5e72efede1960dd0212cb92`
- Collision-decision ledger SHA-256: `3baaa4ed33bd6d89d88c3dbd208017d83742adb96c6873348ba89a41b47c5d3b`

## Replay and coverage

| Stream group | Accepted games | Excluded candidate members | Positions |
| --- | ---: | ---: | ---: |
| AobaZero exact 100 | 100 | 0 | 15,488 |
| Denryu hardware 3 | 132 | 0 | 25,993 |
| WCSC C2B candidate scope | 5,226 | 391 | 618,038 |
| **Raw scan total** | **5,458** | **391** | **659,519** |

The WCSC candidate scope contains 5,617 members: 5,273 staged members plus 344
adapter-rejected members. One approved repairable replacement was accepted:
`WCSC2003/yosen1/daemon-shogi/copy/YAMADA.CSA`. The official archive inventory
contains 8,989 members; 3,372 are outside the frozen C2B candidate scope. All 31
approved WCSC archive hashes, the Denryu archive hash, and every accepted member hash
validated. The five frozen exact WCSC illegal-action exclusions remain included in
the 391 exact candidate exclusions.

The scan retained 528,570 effective published records, 4,943 unique games, and
528,570 unique canonical positions.

## Collision classes

Counts are raw observed cross-split/collision groups followed by effective groups
after the frozen deterministic owner/exclusion rules.

| Class | Raw | Effective | Status |
| --- | ---: | ---: | --- |
| Same record across splits | 0 | 0 | passed |
| Same game across splits | 0 | 0 | passed |
| Canonical position across splits | 5,401 | 0 | passed |
| Protected history across splits | 4,155 | 0 | passed |
| Different plies of one game train/eval | 0 | 0 | passed |
| Prohibited transposition overlap | 5,401 | 0 | passed |
| Parser/normalization-version collision | 10,783 | 0 | passed |
| AobaZero via GCT | not applicable | 0 | passed |
| Floodgate via GCT/direct | not applicable | 0 | passed |
| Distilled vs nodchip | not applicable | 0 | passed |
| Arena start overlap | not applicable | 0 | passed |
| Public test entering training/selection | not applicable | 0 | passed |
| Final holdout entering forbidden path | not applicable | 0 | passed |
| Source-held-out entering training | not applicable | 0 | passed |

The transposition proof is complete over the domain-separated transposition identity:
62,746 raw duplicate groups, 5,401 raw cross-split groups, and zero effective
cross-split groups. Every accepted game has replay, canonical-game, canonical-position,
history-prefix, transposition, and terminal proof. Counts reconcile and the frozen
hash validation remains required before the 1M rung.

Machine-readable evidence is in
`artifacts/phase10r/phase10r-replay-completion.json`,
`artifacts/phase10r/phase10r-scan-input-manifest.json`,
`artifacts/phase10r/phase10r-scan-completion-proof.json`, and the accompanying
collision/checkpoint artifacts.
