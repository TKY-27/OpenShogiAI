#!/bin/sh
# Reproducible pure-only playing artifacts, separate from development benchmark builds.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$project_root/target/phase10t-pure}"
wasm_bindgen=${WASM_BINDGEN:-"$project_root/local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen"}
if [ "$("$wasm_bindgen" --version)" != "wasm-bindgen 0.2.127" ]; then
    echo "wasm-bindgen 0.2.127 is required" >&2
    exit 1
fi
cargo build --locked --release -p open-shogi-cli --no-default-features --features pure-only
cargo build --locked --release -p open-shogi-wasm --target wasm32-unknown-unknown --no-default-features --features pure-only
"$wasm_bindgen" --target web --typescript --out-name open_shogi_wasm \
    --out-dir "$CARGO_TARGET_DIR/bindings" \
    "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/open_shogi_wasm.wasm"
