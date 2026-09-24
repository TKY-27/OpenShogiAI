# OpenShogiAI (OSAI)

[日本語はこちら](README.ja.md)

OpenShogiAI is an independently implemented shogi (Japanese chess) engine and
machine-learning research project, written in Rust with native and
WebAssembly runtimes. It plays book-free shogi with a learned evaluation
network of our own design (OSAVAL03) and ships the training and evaluation
tooling used to produce the published models.

The browser application for playing and analysis is a separate repository:
[OpenShogiUI (OSUI)](https://github.com/TKY-27/OpenShogiUI).
This repository ([OpenShogiAI](https://github.com/TKY-27/OpenShogiAI)) owns
the engine, the model format, training, and model distribution.

Project by [TKY-27](https://github.com/TKY-27).

## Features

- Full shogi rules: legal-move generation with checks, drops, promotions,
  repetition and stalemate handling, SFEN/USI/CSA support.
- Two runtime builds from one code base: a native USI command-line engine and
  a pure WebAssembly module used by [OSUI](https://github.com/TKY-27/OpenShogiUI).
- Pure-learned evaluation: the OSAVAL03 container carries a dual
  king-relative clipped-pair accumulator network (W256/W512) with direct
  cp and win-draw-loss heads. The `pure-only` runtime rejects any handcrafted,
  book or teacher fallback — play strength comes only from the trained weights.
- Book-free play everywhere: no opening books, fixed first moves, position
  tables or online engines, in development matches and in production alike.
- Native/Wasm parity: deterministic Wasm regeneration and cross-runtime
  evaluation checks are part of `make check`.
- Training pipeline: self-play generation, offline fixed-depth teacher
  labeling, dataset preparation with split/lineage control, and bounded
  optimizer runs with resumable state.

## Models

Representative trained weights are distributed from the
[releases page](https://github.com/TKY-27/OpenShogiAI/releases) (tag
`models-v1`) as CC BY 4.0, including the newest generation **OSAI R4** and
earlier development generations. Model generations are listed newest to
earliest; this is development order, not a strength ranking. No human-dan
validation has been performed.

- What is in a release: weight files (OSAVAL03), the shared browser engine
  runtime, the runtime profile, per-model rights and source records, an
  Apple Silicon macOS USI binary, and checksums.
- Weights license and training-data attribution:
  [docs/model/distribution.md](docs/model/distribution.md).
- Format specification: [docs/model/OSAVAL03_FORMAT.md](docs/model/OSAVAL03_FORMAT.md).

Playing in the browser: [OpenShogiUI](https://github.com/TKY-27/OpenShogiUI).

## Build and run

Requirements: stable Rust (>= 1.89), `wasm32-unknown-unknown` target,
wasm-bindgen CLI 0.2.127, Python 3.12 with [`uv`](https://docs.astral.sh/uv/),
GNU Make, and Node.js for the actual-Wasm checks.

```sh
git clone https://github.com/TKY-27/OpenShogiAI.git
cd OpenShogiAI
./scripts/bootstrap_macos.sh        # optional: verifies/installs local tools
uv sync --locked --group dev
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked --root local/tooling/wasm-bindgen-0.2.127
make build
```

Run the default (handcrafted) engine:

```sh
cargo run --locked -p open-shogi-cli -- usi
```

Run a learned model in pure USI mode (weights from a release):

```sh
make pure-build
target/pure/release/open-shogi-cli usi \
  --model osai-r4.osaval03 \
  --model-sha256 9466a7e8cf11b7d165b325edd9a5a421bdbaa4bed550940afcf33c9faf3bfd0f \
  --model-format OSAVAL03 --profile pure_learned
```

All-in-one verification (format, lint, tests, native build, deterministic
Wasm regeneration, license/boundary/provenance checks):

```sh
make check
```

The macOS binary in the release is an unsigned local build; see the release
notes for the Gatekeeper note, or build from source as above.

## Documentation

- [Development guide](docs/development.md) — commands, storage, validation.
- [Architecture](docs/architecture.md) and
  [interface contracts](docs/interfaces.md).
- [Rules profile](docs/rules.md) — the shogi rules implemented.
- [Model handling](docs/model/handling.md),
  [OSAVAL03 format](docs/model/OSAVAL03_FORMAT.md),
  [distribution and weights license](docs/model/distribution.md).
- [Data rights and lineage](docs/data/handling.md),
  [source audits](docs/source-audits/).
- [License scope](docs/license-scope.md) and
  [third-party notices](THIRD_PARTY.md).
- [References](docs/references.md) — rules, algorithms and standards
  consulted, including the prior engines this independent implementation
  learned from (Apery, YaneuraOu lineage materials and others).

## Contributing

Issues and pull requests are welcome; see
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md). This is a solo-maintainer
project: continuous updates, response deadlines and merging of every PR are
not guaranteed, but quality discussions and contributions are actively
considered. AI-assisted contributions are limited to agents with capabilities equivalent to or greater than Astra or Fable.

## License

Project-owned source code is **AGPL-3.0-only** ([LICENSE](LICENSE)).
Dependencies, datasets, third-party assets and model weights keep their own
terms; see [license scope](docs/license-scope.md) and
[third-party notices](THIRD_PARTY.md). Published weights are CC BY 4.0 with
the attribution recorded in [docs/model/distribution.md](docs/model/distribution.md).

Contact: X [@ANAg2bGOD](https://x.com/ANAg2bGOD) (DM).
