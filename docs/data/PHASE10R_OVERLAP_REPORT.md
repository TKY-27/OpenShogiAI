# Phase 10R overlap and leakage report

Status: bounded sample report; no full external dataset was merged.

## Measured sample inventory

| Sample | Files / archive members | Source records | Exact position identity in source-preserving adapter | Rust replay |
| --- | ---: | ---: | ---: | ---: |
| WCSC32 ZIP | 279 CSA files; 293 ZIP entries including directories and macOS metadata | First bounded CSA sample: 320 move-prefix records | Not yet comparable across source namespaces; CSA prefixes are explicitly non-canonical until replay output is joined. | 10/10 bounded games accepted after CP932/V2.2 derived-view conversion. |
| Denryu hardware-3 ZIP | 132 KIF files; 135 ZIP entries including directories and one metadata text file | First bounded KIF sample: 96 played moves; terminal `千日手` excluded from played moves | Not yet emitted into the normalized canonical-position table; KIF source/history digests remain non-canonical. | 1/1 converted bounded game accepted; 96 moves. |

The current Phase 10R normalized sample therefore has no defensible cross-source canonical-position overlap count. That is reported as “unavailable,” not zero. The overlap tool emits exact raw-record, source position, history, canonical-SFEN, and transposition layers independently so a later rule-engine join can add measured counts without rewriting source rows.

## Known lineage and contamination risks

* GCT material names AobaZero, Floodgate, Denryu, Taya, Yaneura, and Suisho-derived groups. It is not safe to deduplicate only by artifact name; source lineage must be retained and compared after decoding.
* The DL水匠15b release says it transformed nodchip material with Hao qsearch shuffling and rewrote evaluations with DL水匠15b. It is therefore a likely derived overlap with nodchip/Hao, not an independent source.
* GCT Floodgate groups may overlap both Floodgate raw records and other GCT mixed groups.
* KIF and CSA representations of the same game require canonical replay before comparison. Text hashes and archive-member hashes are evidence of byte identity only.
* `provenance.raw_record_sha256` is overlap-report-only; it is not a per-position deduplication key because all positions from one game share the same source-record hash.

## Deterministic policy

The source priority is recorded in [`configs/phase10r/deduplication.yaml`](../../configs/phase10r/deduplication.yaml): existing approved AobaZero first, then approved WCSC/Denryu, followed by any later-approved derived or external families. The priority never overrides a reserved holdout. Deduplication precedence is canonical SFEN, transposition key, exact position, then exact game/history. Missing identities are quarantined rather than dropped or guessed.

No train/validation/test split was created from the external sample in Phase 10R-A. WCSC33–36 and recent Denryu TSEC events remain reserved before normalization, and the denied public evaluation boundary is never read for training.
