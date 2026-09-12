# OpenShogiAI

OpenShogiAI is an independently implemented shogi engine with native and WebAssembly runtimes,
model-format support, and research data/training tools. The browser application is maintained
in the separate OpenShogiUI (OSUI) repository.

It provides legal move generation, SFEN/USI/CSA, bounded search, game-clock allocation,
terminal play, analysis and offline opening-data interfaces. All play is book-free; runtime
book loading is rejected. The default is the built-in
`handcrafted-experimental` evaluator. Experimental learned formats are explicit opt-ins;
no trained weights, datasets or external teacher binaries are distributed here.
Amateur-dan strength and the next design's novelty have not been established.

## Setup

Use stable Rust (minimum 1.89), Rustfmt, Clippy, `wasm32-unknown-unknown`, Python 3.12,
`uv`, GNU Make, and Node.js for actual-Wasm checks. Inspect tools with
`./scripts/bootstrap_macos.sh`. Then:

```sh
uv sync --locked --group dev
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked --root local/tooling/wasm-bindgen-0.2.127
make build
```

`Cargo.lock` and `uv.lock` pin dependencies. Existing local tools/environments may be reused.

## Run and test

```sh
cargo run --locked -p open-shogi-cli -- usi
cargo run --locked -p open-shogi-cli -- perft --depth 3
cargo run --locked -p open-shogi-cli -- play --human black --profile overall-champion --black-time-ms 180000 --white-time-ms 180000
make check
```

Standard builds/tests use synthetic or small checked-in fixtures, not private training data.
For the optional local frozen W256 comparison model, see [model handling](docs/model/handling.md).
The hard4 terminal/runtime-proof failure is fixed. The current defense/opening campaign
continues the local r3 evaluator with corrected offline teacher labels and balanced replay.
The old computation controller stays OFF. Local candidates are not public model promotions. See the
[prototype status](docs/status.md) and [local play instructions](docs/development.md).

## Documentation

- [Current state and next work](docs/status.md): the single handoff document.
- [Development](docs/development.md): commands, storage and validation.
- [Architecture](docs/architecture.md) and [interfaces](docs/interfaces.md).
- [Data handling](docs/data/handling.md), [model handling](docs/model/handling.md).
- [Source provenance](docs/provenance/source.md), [license scope](docs/license-scope.md),
  and [third-party notices](THIRD_PARTY.md).

`engine/`, `training/`, `bindings/`, `configs/`, and `tests/` retain their functional boundaries.
Wasm bindings expose the engine contract; they do not contain a browser GUI. Copying artifacts
into OSUI and verifying actual play is a separate integration task.

## License

Project-owned source is **AGPL-3.0-only**; see [LICENSE](LICENSE). Dependencies, external
teachers, datasets and model weights retain separate terms. Weight distribution remains
pending review and is not authorized by this source license. This cleanup does not publish a release.
