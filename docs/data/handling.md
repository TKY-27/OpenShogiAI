# Data rights, lineage and evaluation boundaries

Data is not limited to AobaZero. Every new source/version needs its own training, derived-weight,
redistribution and commercial-use decision. Availability or a code repository's license is not
proof of dataset permission. No new rights audit or acquisition was performed in this cleanup.
The AGPL-3.0-only source license does not automatically apply to datasets. A dataset license
requires complete source provenance, explicit ML permission, compatible redistribution terms
and a reviewed manifest. Unapproved sources must not enter training or be called redistributable.

## Source records

- [Exact AobaZero sample audit](../source-audits/aobazero.md) and
  [catalog](../../configs/data_source_objects/aobazero_no_noise.yaml): historical approval
  covers exactly 100 no-noise CSA objects, not weights, live indexes or other revisions.
- [Source registry](../../configs/phase10r/source-registry.yaml): exact WCSC/Denryu and other
  artifact URLs, revisions, formats, rights dimensions and reservations. Historical raw-training
  decisions do not clear derived weights or imply that today's terms have been rechecked.
- [External audit catalog](../../configs/data_sources_external.yaml) and
  [Floodgate audit](../source-audits/floodgate.md): pending/denied material remains excluded.

The historical WCSC decisions cover tournaments 1–29 and 31–32 (30 was cancelled), with the
CSA requirement to identify the program, game date, event or source when publishing use of
records. Denryu approval is scoped to `kifu_dr5hdw3.zip`, not other events. Keep those
attributions. GCT, nodchip/Tanuki/Suisho, Taya/QPD, Lishogi and unpinned revisions remain
pending according to the detailed registry; Bonanza prior-art inspection does not permit
importing third-party evaluation/book/hash tables. No permission request was sent.

## Partitioning and semantics

Assign source-game/history and connected descendant components before extracting positions.
All related original/derived positions retain the strongest protected split. Canonical-position,
transposition and history collisions must not move holdout data into train. Keep unique-position
counts separate from total samples/exposures, and do not claim strength from repetition alone.

[Holdout policy](../../configs/phase10r/holdout-policy.yaml) protects WCSC33–36, recent Denryu
TSEC, designated/public-test material and pending-rights evaluation assets. Final holdout
contents must not be opened, trained on or repartitioned by routine workflows. Local
`local/frozen/metadata/game-partitions.jsonl.gz` contains identifiers/partitions only;
`split-guard.json` retains component/source/position hashes without position contents.
The small mixed-split Phase 3 original remains sealed under `local/frozen/evaluation/`.
Large old derivative evaluation sets were deleted. Acquisition URLs/revisions/hashes and
preparation metadata remain local; those sets are not available until deliberately reconstructed
and independently checked. Do not silently generate a replacement split.

Adapters preserve original bytes/encoding, source score and perspective, played move versus
teacher best move, typed mate, candidate visits and provenance. Unknown fields stay unknown;
no root score is copied to an unlabelled descendant and no absent MultiPV score is fabricated.
Only legally replayed, rights-cleared, partition-bound records become trainable.
See [format contracts](formats.md), [interfaces](../interfaces.md) and
[legacy target semantics](../model/target-semantics.md).

## Teacher and storage

The optional Apery 2.0.0 installation is kept locally to avoid reacquiring its large evaluation
tables. Its source, licenses, exact binary, eval files and install manifest remain together;
configurations under `configs/teacher/` pin identities. GPL engine and MIT evaluation terms
remain separate from project code and generated weights. No teacher runs during play.

Future data, labels and training output belong under `local/runs/`; see [development](../development.md).
No raw data, teacher binary, checkpoint or trained weight belongs in Git.
