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

# Keep source-location strings reproducible and free of host account paths, exactly
# like scripts/build_wasm_web.sh: panic locations must not embed the build host.
python3.12 - "$project_root" <<'PY_BUILD'
import os
import shlex
import subprocess
import sys
from pathlib import Path

environment = os.environ.copy()
separator = chr(31)
encoded = environment.get("CARGO_ENCODED_RUSTFLAGS")
flags = encoded.split(separator) if encoded else shlex.split(environment.get("RUSTFLAGS", ""))
for source, target in (
    (Path.home(), "/user"),
    (Path(environment.get("CARGO_HOME", str(Path.home() / ".cargo"))), "/cargo"),
    (Path(sys.argv[1]), "/open-shogi"),
):
    flags.append(f"--remap-path-prefix={source.resolve()}={target}")
environment["CARGO_ENCODED_RUSTFLAGS"] = separator.join(flags)
subprocess.run(
    ["cargo", "build", "--locked", "--release", "-p", "open-shogi-wasm",
     "--target", "wasm32-unknown-unknown", "--no-default-features",
     "--features", "pure-only"],
    env=environment, check=True,
)
PY_BUILD
"$wasm_bindgen" --target web --typescript --out-name open_shogi_wasm \
    --out-dir "$CARGO_TARGET_DIR/bindings" \
    "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/open_shogi_wasm.wasm"
