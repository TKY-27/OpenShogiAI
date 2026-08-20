# Cross-project tests

- `tests/python/` contains repository-level Python tests.
- Rust unit, integration, and property tests live beside their crates.

Phase 3 Python tests cover strict source/evidence catalogs, URL and filesystem containment,
live robots and pinned evidence, bounded/retry/resume acquisition, immutable manifests,
content and canonical deduplication, CSA adaptation, Rust-backed normalization, stable splits,
deterministic gzip artifacts, and opening aggregation/export. Synthetic CSA fixtures exercise
these boundaries without linking or invoking a third-party shogi engine.
