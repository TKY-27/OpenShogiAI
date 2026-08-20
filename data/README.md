# Dataset storage

`data/raw/` and `data/processed/` are local, ignored storage locations. They must never contain
unapproved or untraceable training data, and their contents must never be added to Git.

Phase 3 permits only the exact 100-object `aobazero-no-noise` source slice declared by the
versioned registry and catalogs. Raw files, evidence snapshots, and the append-only acquisition
manifest live under `data/raw/phase3/aobazero-no-noise/`. Normalized games, positions, dataset
manifests, reports, and opening artifacts live under `data/processed/phase3/`.

Use the guarded `make phase3-*` targets documented in the repository README. Do not place
manually downloaded files into the pipeline, infer permission from public accessibility, or
reuse an object whose manifest identity and local SHA-256 do not verify. The authoritative
rights decision is `docs/source-audits/aobazero.md`; all other dataset and model-weight rights
remain pending.
