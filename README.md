# OpenShogiAI

OpenShogiAI is an independently implemented shogi engine, model runtime, training stack, and
reproducible evaluation toolkit. This repository is the AI-side clean-history release candidate;
the browser application lives in the separate OpenShogiUI repository.

## Included

- complete shogi rules, legal move generation, SFEN/USI/CSA handling, and perft tooling;
- handcrafted and checksummed `OSAVAL01` neural evaluation;
- deterministic bounded search, USI, terminal play, Arena, and opening-book tooling;
- audited data acquisition/normalization, external-USI teacher adapters, training/export,
  self-play, replay, promotion, and evaluation workflows;
- a WebAssembly engine and versioned generated bindings under `bindings/wasm/`;
- tests and provenance documentation for the included AI-side functionality.

No browser GUI, teacher binary, teacher evaluation file, raw/processed dataset, checkpoint, or
trained model weight is included.

## Requirements

- Rust 1.89 or newer on stable, with Clippy, Rustfmt, and `wasm32-unknown-unknown`;
- Python 3.12 and `uv`;
- GNU Make;
- exact `wasm-bindgen-cli` 0.2.127 installed under
  `local/tooling/wasm-bindgen-0.2.127/` for deterministic binding checks.

The diagnostic bootstrap does not install software or request administrator privileges:

```sh
./scripts/bootstrap_macos.sh
```

## Verification

```sh
make check
```

The gate syncs the locked Python environment, validates the local toolchain, audits repository
and license boundaries, runs Rust/Python formatting and linting, executes all Rust/Python tests,
builds the Rust workspace, and checks deterministic Wasm regeneration.

Useful focused commands include:

```sh
cargo run --locked -p open-shogi-cli -- usi
cargo run --locked -p open-shogi-cli -- perft --depth 3
make model-validate
make phase3-validate-registry
make phase6-validate-config
make wasm-web-check
```

Networked acquisition, external teacher execution, training, self-play, and evaluation commands
remain bounded workflows with local ignored outputs. Read the relevant documents and configs
before running them.

## Data and models

The exact 100-object AobaZero sample decision is documented in `DATASET_CARD.md` and
`docs/source-audits/`. Other datasets are denied or pending unless separately approved.
Generated model weights remain `pending-review` and are not licensed or distributed by this
source repository. See `MODEL_CARD.md`, `PROVENANCE.md`, and `LICENSE_SCOPE.md`.

## License

Project-owned source code in this clean-history candidate is licensed under
`AGPL-3.0-only`. Dependencies, external teachers, datasets, generated-tool output, and model
weights retain their own terms or pending status; see `THIRD_PARTY.md` and
`LICENSE_SCOPE.md`.
