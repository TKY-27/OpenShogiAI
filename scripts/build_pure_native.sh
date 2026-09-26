#!/bin/sh
# Native-only pure runtime build. Requires only a supported Rust toolchain plus a
# normal linker/C toolchain; no Wasm tooling, Node, Python, or model assets.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$project_root/target/pure}"
exec cargo build --locked --release -p open-shogi-cli --no-default-features --features pure-only
