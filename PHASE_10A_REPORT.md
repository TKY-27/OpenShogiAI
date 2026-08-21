# Phase 10A — 外部将棋データ監査・サンプル正規化

## Executive summary

Phase 10A was completed as a rights-and-format audit only. No model training, self-play, Arena campaign, teacher relabeling at scale, promotion decision, architecture change, unknown-license download, push, or release was performed.

Decision counts are **approved 2 / pending 18 / denied 2** at the artifact-record level. The two approved records are the existing exact 100-object AobaZero no-noise scope and the bounded `w4745.csa` sample inside that scope. No new external dataset is approved for training.

The complete ledger is [`configs/data_sources_external.yaml`](configs/data_sources_external.yaml). The two machine-readable handoff files are:

- [`artifacts/phase10a/audit-manifest.json`](artifacts/phase10a/audit-manifest.json)
- [`artifacts/phase10a/sample-normalization-report.json`](artifacts/phase10a/sample-normalization-report.json)

## Approved, pending, and denied

### Approved

- `aobazero-no-noise-exact100`: only the exact versioned 100-object catalog already approved by OpenShogiAI.
- `aobazero-no-noise-w4745-sample`: one bounded CSA sample used for format validation and normalization.

Approval does not expand to live AobaZero indexes, daily records, GCT conversions, weights, or other AobaZero assets. Derived-weight, commercial-use, and attribution conditions not explicitly covered by existing provenance remain pending.

### Pending

- All GCT/dlshogi Drive artifacts, including `hcpe3`, AobaZero-converted HCPE, Taya-derived self-play, mixed Floodgate/local records, and Suisho HCPE.
- nodchip Hao depth9 and tanuki NNUE PackedSfenValue datasets.
- たややん 36SFEN, the Tadao 5247-position derivative, and KIF/PSV teacher collections.
- Qhapaq `QPD_train.7z`.
- AobaZero live sample index and unapproved daily records.

The unresolved license questions are exact permission for: training, derived weights, redistribution, commercial use, and required attribution. A public page, a repository software license, or HF `cardData.license` is not treated as a dataset grant. Qhapaq has an explicit attribution condition for publishing or entering a trained evaluation function, but the remaining rights still require confirmation.

### Denied

- `dlshogi-public-evaluation-test`: immutable final holdout; never use for training.
- `open-shogiai-floodgate`: existing provenance marks it unverified and disallows ML use; Phase 10A did not reauthorize it.

## Storage and size audit

At audit start, the filesystem had 345,509,875,712 bytes (about 321.77 GiB) free. The hard floor was 214,748,364,800 bytes (200 GiB), leaving about 121.78 GiB of headroom. The approved local Phase 10A budget was 1 GiB.

Only 19,762 bytes of external data and 28,677 bytes of official evidence snapshots were retained transiently, for 48,439 bytes total. The sample was removed after hash and parse validation. No full candidate artifact was downloaded or decompressed.

Important candidate sizes:

- GCT Drive UI aggregate: approximately 84 GB; exact file sum and decompressed size unknown.
- nodchip Hao: 320,002,979,440 bytes; fixed 40-byte source records; normalized JSON estimate 2.400–6.400 TB.
- nodchip tanuki: 320,002,292,200 bytes; fixed 40-byte source records; normalized JSON estimate 2.400–6.400 TB.
- `QPD_train.7z`: 11,304,930 compressed bytes; internal decompressed size unknown.
- Taya 36SFEN: 384,535 bytes as observed metadata.
- たややん teacher collection: approximately 150 million positions stated; byte size not resolved.

All estimates and their basis are in the YAML ledger. They are not authorization to acquire the data.

## Sample parse results

`w4745.csa`:

- 19,762 bytes; SHA-256 `6a08c7a9f519ce81505bbdf7c51905781413ddf8955a955b1b5b1d61a2703aad`.
- Strict AobaZero envelope validation passed.
- 109 moves/normalized move-prefix records, 109 source annotations, 33 comments.
- `%TORYO` retained as raw terminal; winner was not inferred.
- `v` retained as raw source score; semantics and perspective remain unknown.
- Position identity is `csa_move_prefix` with `exact: false`, because this audit adapter does not replay a board.

Format fixtures passed for CSA, KIF, HCPE, HCPE3, PackedSfenValue, and the binpack alias. HCPE3 variable candidate visits were preserved as raw visits plus derived probabilities. Truncated binary inputs were rejected.

## Overlap and leakage

Exact overlap was measured only for exact identity namespaces. The external source body was not available for cross-source hashing, so most cross-source counts remain unknown rather than zero.

- GCT’s official article explicitly lists AobaZero-converted HCPE, so AobaZero-in-GCT duplication is lineage-confirmed but not countable without a manifest and crosswalk.
- GCT `taya36` and `selfplay-unique-900xx/...902xx` are explicitly derived from the たややん互角局面 family; treat them as contaminated with that holdout until proven otherwise.
- The official Taya comparison reports 1,895/1,957 same HCP positions and 1,718/1,957 when same 24th-ply paths are considered for the 5,247-position derivative.
- Existing OpenShogiAI Phase 3 uses game-only split assignment and protects the test split. External data must preserve game/history grouping and immutable holdouts.

Full details are in [`docs/data/DATASET_OVERLAP_REPORT.md`](docs/data/DATASET_OVERLAP_REPORT.md).

## Implemented adapters

[`training/open_shogi_training/data/external_audit.py`](training/open_shogi_training/data/external_audit.py) implements bounded readers for:

- CSA and KIF text move prefixes;
- fixed-size HCPE;
- variable-size HCPE3 candidate visits;
- PackedSfenValue / packed SFEN / binpack records.

The `phase10a_normalized_record/v1` schema retains position/history identity, played/best move, policy visits, WDL/result, raw score and score semantics, perspective, mate representation, nodes/playouts/depth, source/artifact/record IDs, raw hashes, and license decision. Incompatible labels are not collapsed.

## Recommended inputs for the later Sol dataset-design decision

1. A rights matrix with an explicit answer for every artifact and every use right; request clarification from the author/publisher where any cell is pending.
2. Exact manifests, per-file sizes, SHA-256 values, archive-internal terms, and source record/game IDs for GCT, HF, Taya, and QPD.
3. A storage plan that can hold at least one 320GB dataset plus decompression and normalized-row headroom without crossing the 200 GiB free-space floor.
4. A source lineage crosswalk for AobaZero → GCT, Taya → GCT, Taya original → Tadao 5247, and any QPD mixtures before split assignment.
5. A canonical board/history identity policy and game/history-grouped split policy, with immutable public test manifests fixed before any training use.
6. A decision on whether source-specific policy/eval/WDL labels are kept as separate views rather than coerced into one target.

The audit does not recommend adopting any pending dataset. It supplies evidence and constraints for the later decision.

## Handoff

- Branch: `codex/phase10a-external-data-audit`
- Push/release: not performed
- Commit SHA: reported in the final task handoff after commit; this report does not embed a self-referential commit hash.
- Worktree: verified clean after the final commit.
