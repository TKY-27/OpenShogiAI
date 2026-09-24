# Model distribution and weights license

This document records what is published, under which terms, and which
conditions were actually verified. The machine-readable counterpart is
[configs/models/distribution.json](../../configs/models/distribution.json);
`scripts/build_release_assets.py` assembles the release staging directory from
it after verifying every artifact byte-for-byte.

## What is published

One GitHub release per distribution generation (first: tag `models-v1`)
containing:

- Six representative OSAVAL03 weight files: **OSAI R4** (the R4-C4 run's
  selected G02 candidate) plus five earlier development generations
  (C3, C1, defense best1536, r3 step 6144, initial W256). Not every checkpoint
  is published. The historical R4-C2 selection had bytes and runtime identical
  to defense best1536 and is published as that same configuration, not as a
  separate model.
- The shared browser engine runtime pair (`engine.js` + `engine.wasm`, the
  pure-only Wasm build) and the `pure_learned-v3` runtime profile.
- An Apple Silicon macOS USI command-line binary (unsigned; verified only on
  the build machine — other platforms build from source).
- Per-model rights records (`rights-*.json`), a release manifest and
  SHA-256SUMS.

Generation order in listings is newest-to-earliest development order. It is
not a strength ranking. Measured facts about OSAI R4 that remain on record:
its fixed adoption gate versus C3 was not met (0.5625/0.5625 over 68 games at
both clocks), opening-stage weaknesses were observed, and no formal
human-dan validation exists. Publication is the owner's selection, not a
strength claim.

## Weights license decision

The published weights are offered under **CC BY 4.0** (attribution:
TKY-27 / OpenShogiAI). Redistribution, modification, further training and
commercial use of the weights are permitted with attribution.

This decision is scoped to the weights. Code remains AGPL-3.0-only; training
data, teacher binaries and evaluation tables are not redistributed;
third-party datasets and assets keep their own terms (see
[license scope](../license-scope.md) and [third-party notices](../../THIRD_PARTY.md)).

## Conditions that were verified

Every published model descends from the frozen W256 baseline, so the union
below applies to all of them (each rights record lists its exact set):

| Source | Recorded condition | How it is satisfied |
|---|---|---|
| CSA WCSC records (tournaments 1–29, 31–32) | Use free **with attribution** of program, date, event or source when publishing use of records | The source list in each rights record and this document credit CSA/WCSC |
| Denryu-sen hardware-3 archive `kifu_dr5hdw3.zip` | Game-record use without restrictions (that archive only) | Credited; other Denryu events were not used |
| AobaZero no-noise samples (exact 100-file catalog) | Public domain per the project's own statement | Credited; catalog is file-scoped and hash-pinned in the audit |
| nodchip shogi_hao_depth9 (4 pinned shards; C3/C4 lineage only) | MIT | Publisher and dataset credited; no raw data redistributed |
| Apery 2.0.0 teacher | GPL-3.0 code / MIT evaluation tables — neither is redistributed | Labels were generated offline by running the teacher; the binary and tables stay local |

Audits: [source audits](../source-audits/). Floodgate was reviewed and denied;
dlshogi/GCT and other pending sources were never used (adoption count 0).

### Residual caveats

- The phase10r source registry marks *derived-weights* permission for
  WCSC/AobaZero as "pending" as a conservative internal default. The
  third-party terms actually recorded for those sources are the attribution
  conditions quoted above; this release satisfies them by crediting. This is
  the project's own review conclusion (2026-09-24), not a statement from the
  publishers.
- Hao shard records are positional (packed root + teacher value); the original
  generating engine/weights are unknown and not claimed.
- No permission request was sent to any publisher; the conditions rely on
  their published statements pinned in the audits.

## Reproducing and verifying a release

```sh
make pure-build                      # native CLI + pure Wasm build
python3 scripts/build_release_assets.py   # verify + stage local/release/models-v1/
shasum -a 256 --check local/release/models-v1/SHA256SUMS.txt
```

The staging directory is the local mirror used by
[OpenShogiUI](https://github.com/TKY-27/OpenShogiUI)'s build
(`OPENSHOGI_ASSET_MIRROR`) before the GitHub release is public. The published
release must be created from the same reviewed manifest; asset names and
hashes are fixed there. After publication, OSUI builds fetch the fixed tag
and verify hashes — they never follow `latest`.

Weights are not checked into Git. `configs/models/registry.json` remains the
internal frozen-comparison registry and keeps its own conservative status
fields; it does not govern publication.
