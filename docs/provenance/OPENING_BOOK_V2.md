# Opening-book v2 provenance record

Review date: 2026-08-21 (Asia/Tokyo)

The first serious v2 book uses only the approved `aobazero-no-noise-pd-sample100` game slice,
its Phase 3 train-split opening statistics, and a fresh audited MultiPV 32 teacher view of the
same 10,000 unique positions. It imports no third-party engine book, `standard_book.db`,
commercial book, or file whose license was inferred from a source-code repository.

The official pinned AobaZero README identifies the official collected-game page and states that
non-engine project contents are public domain. Immutable evidence snapshots and all 100 source
object hashes remain bound by the Phase 3 dataset manifest. A 2026-08-21 reachability recheck of
the official sample index and `w4745.csa` returned HTTP 200; it was evidence verification, not a
new acquisition.

## Source and teacher bindings

- dataset manifest SHA-256:
  `db2a286ebfa4edd01c67041fb55d33b0f5f8813d24a2203d7c5e55a36111e243`;
- normalized positions SHA-256:
  `d36cfc86757887f05a2d0c5ecc50e606a5836687cff32e757e268f8cf5cb627f`;
- Phase 3 opening statistics SHA-256:
  `41b1b7eb96cf8933697a166abae6696bbbd6b75433fbc2ba695bfde26570b97d`;
- canonical teacher-config binding SHA-256:
  `acab520b5b78055043e558ea59429e420998c9f2ea889757c520e36522f47a32`;
- Apery 2.0.0 binary SHA-256:
  `8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403`;
- 25,000-node/MultiPV 32 benchmark SHA-256:
  `56d89d1a11b37db37aa63f8540dfa9a2f5e623c9dc9ecd2e560b5e4ecfed5718`;
- teacher labels SHA-256:
  `03f5375f09b167a12a4297cd2a404f0fb95a91edd2f6cdeabc955c14c7ce5ea0`;
- teacher-label manifest SHA-256:
  `57f70b734f8c3eab4bd6c0ae2c439eae3373baf4f1b304d4fb32a17bb0e726fb`.

The benchmark completed all nine positions at 25,000 nodes with p95 11 ms and measured peak
process-tree RSS 2,408,628,224 bytes. Labeling used one worker, four teacher threads, 1,024 MiB
hash, and a 16 GiB working-memory ceiling; all 10,000 positions completed and none was
quarantined. Of 10,000 rows, 8,105 returned 32 candidates. Independent legal-root auditing found
25,946 additional legal roots across shorter rows; missing roots have no invented score.

## Build result

The deterministic build used maximum 40 plies, minimum sample count 2, maximum teacher loss
80 cp, standard shogi rule profile, and build version
`openshogiai-book-v2-20260821-multipv32`. From 8,855 input records it rejected 269 without a
teacher candidate and 20 for teacher loss. The result contains 424 positions and 448 candidates:
432 `ibisha`, 4 `ibisha-vs-furibisha`, and 12 other classifications.

The ignored local gzip artifact is 69,423 bytes with SHA-256
`3925fb49cbd9dfaf62f66cb92ebb06116a2cc63d05113ec4ee9490b951f34b85`.
An independent second build from the same inputs was byte-identical. Rust verification rejects
wrong checksums, malformed/corrupt gzip, noncanonical positions, illegal candidates, incompatible
rule profiles, duplicate candidates, and invalid provenance fields.

Generated books, labels, teacher binaries, and teacher evaluation files remain ignored local
artifacts and were not committed or published. The phase report records rebuild, verification,
USI smoke, hit-rate, and profile-comparison commands.
