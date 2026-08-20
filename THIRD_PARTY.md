# Third-Party Components

OpenShogiAI production engine code is independently implemented. No third-party shogi engine
source, binary, or evaluation file is linked, copied, or distributed by this repository.

Exact resolved dependency versions and integrity metadata are recorded in `Cargo.lock` and
`uv.lock`. Dependencies retain their own licenses; the project's `AGPL-3.0-only` license
does not replace them.

| Ecosystem | Component | Purpose | License family |
| --- | --- | --- | --- |
| Rust | `sha2`, `flate2`, `libc`, `serde`, `serde_json`, `toml`, `web-time`, `wasm-bindgen` | Engine runtime, formats, secure files, and Wasm interface | MIT and/or Apache-2.0 |
| Rust test | `proptest`, `base64` | Property and interoperability tests | MIT and/or Apache-2.0 |
| Python | `PyYAML` 6.0.3 | Closed YAML parsing | MIT |
| Python | `numpy` 2.5.1 | Pinned numerical runtime | BSD-3-Clause |
| Python | `torch` 2.11.0 | Local training, checkpointing, evaluation, and export | BSD-3-Clause |
| Python test/tooling | `pytest`, `ruff` | Tests, formatting, and linting | MIT |

This summary is not a substitute for a distribution-time inventory of the complete locked
transitive dependency closure.

## Generated Wasm interface

`bindings/wasm/` is reproducible output from the project Rust code and `wasm-bindgen`
0.2.127. Linked dependencies and generator components retain their respective licenses,
including MIT and Apache-2.0 terms. The generated files are not relicensed representations of
third-party source.

## External teacher

Apery Rust v2.0.0 is an optional separately installed USI teacher under ignored
`local/teacher/` storage. Its engine is GPL-3.0-only; its referenced evaluation files are MIT.
The configured identities are recorded in `configs/teacher/apery-v2.0.0.yaml` and the local
install manifest. None of those binaries or evaluation files is tracked or distributed here.

## External data and models

The exact approved AobaZero game-record slice is Public Domain under the source-scoped decision
in `docs/source-audits/aobazero.md`; its bytes remain untracked. Floodgate remains denied
pending adequate rights evidence. Generated model weights remain `pending-review` and are not
distributed. See `DATASET_CARD.md`, `MODEL_CARD.md`, `PROVENANCE.md`, and
`LICENSE_SCOPE.md`.
