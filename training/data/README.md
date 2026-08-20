# Training data tools

The implementation lives in `training/open_shogi_training/data/` and covers strict source
registry parsing, exact-catalog acquisition, append-only manifests, the bounded AobaZero CSA
adapter, Rust-backed legal normalization, stable game splits, deterministic gzip JSONL, and a
train-only SQLite opening database/export.

Dataset objects themselves live only under ignored `data/raw/` and `data/processed/` paths.
Only the exact Phase 3 AobaZero 100-object slice is approved; source-specific licensing and
usage decisions are tracked in `configs/data_sources.yaml`, `docs/source-audits/`, and
`data/LICENSE-PENDING.md`.

Run the public entry point as `PYTHONPATH=training uv run python -m
open_shogi_training.data --help`, or use the guarded `make phase3-*` targets. Registry and dry
run commands are network-free. Acquisition requires `--sample-only`, refuses disabled or
ambiguous sources, and is hard-capped at 100 files.
