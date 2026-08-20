# Opening-book v2 provenance record

Review date: 2026-08-21 (Asia/Tokyo)

The first v2 book uses only the already approved `aobazero-no-noise-pd-sample100` dataset and its
existing teacher-label artifact. No third-party engine book, `standard_book.db`, commercial book,
or file licensed only by inference from repository source code was imported.

The official pinned AobaZero README identifies the official collected-game page and states that
non-engine contents are public domain. The immutable evidence snapshots and source-object hashes
remain in the Phase 3 dataset manifest. A fresh 2026-08-21 reachability check of the official sample
index and `w4745.csa` returned HTTP 200; this was a recheck only, not a new acquisition.

Source bindings:

- dataset manifest: `db2a286ebfa4edd01c67041fb55d33b0f5f8813d24a2203d7c5e55a36111e243`
- normalized positions: `d36cfc86757887f05a2d0c5ecc50e606a5836687cff32e757e268f8cf5cb627f`
- Phase 3 opening statistics: `41b1b7eb96cf8933697a166abae6696bbbd6b75433fbc2ba695bfde26570b97d`
- teacher labels (10,000 rows): `195850b3ae7b1dce8a98185f2ba17f794a0200c92eafc672e11674d17d8cd3f4`
- teacher-label manifest: `edeab652bd825410f46bab3010d3e8a017f33cd281cd0f28329d8eaab2a2cb54`

Build parameters were maximum 40 plies, minimum sample count 2, maximum teacher loss 80 cp, and
build version `openshogiai-book-v2-20260821`. The result contains 241 positions and 245 candidates:
234 `ibisha`, 3 `ibisha-vs-furibisha`, and 8 other classifications. Its ignored local artifact is
40,557 bytes with SHA-256
`3fe9da8f9888c9a909066f030b08e6bc18550260a7e2ae0f4eec674ebf9fa3e5`.
An independent second build from the same inputs was byte-identical.

Distribution decision: the generated book is not committed in this change. Its source provenance
is approved and redistributable, but the project continues to keep generated data artifacts out of
Git. Rebuild and verification commands are recorded in the phase completion report.
