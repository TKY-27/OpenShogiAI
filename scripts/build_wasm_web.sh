#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
wasm_bindgen=${WASM_BINDGEN:-"$project_root/local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen"}
wasm_target="$project_root/target/wasm32-unknown-unknown/release/open_shogi_wasm.wasm"
generated_dir="$project_root/bindings/wasm"
mode=${1:-write}

if [ "$mode" != "write" ] && [ "$mode" != "check" ]; then
  echo "usage: scripts/build_wasm_web.sh [write|check]" >&2
  exit 2
fi

if ! rustup target list --installed | grep -qx "wasm32-unknown-unknown"; then
  echo "missing Rust target; run: rustup target add wasm32-unknown-unknown" >&2
  exit 1
fi
if [ ! -x "$wasm_bindgen" ]; then
  echo "missing wasm-bindgen 0.2.127 at $wasm_bindgen" >&2
  echo "install with: cargo install wasm-bindgen-cli --version 0.2.127 --locked --root local/tooling/wasm-bindgen-0.2.127" >&2
  exit 1
fi
if [ "$($wasm_bindgen --version)" != "wasm-bindgen 0.2.127" ]; then
  echo "wasm-bindgen must be exactly 0.2.127" >&2
  exit 1
fi

# Keep source-location strings reproducible and free of host account paths.
# Encoded flags preserve paths containing spaces and existing caller flags.
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
     "--target", "wasm32-unknown-unknown"],
    env=environment, check=True,
)
PY_BUILD

temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/open-shogi-wasm-web.XXXXXX")
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM
"$wasm_bindgen" \
  --target web \
  --typescript \
  --out-name open_shogi_wasm \
  --out-dir "$temporary_dir" \
  "$wasm_target"

generated_files="
open_shogi_wasm.d.ts
open_shogi_wasm.js
open_shogi_wasm_bg.wasm
open_shogi_wasm_bg.wasm.d.ts
"

if [ "$mode" = "check" ]; then
  for generated_file in $generated_files; do
    if [ ! -f "$generated_dir/$generated_file" ]; then
      echo "missing generated Wasm binding: bindings/wasm/$generated_file" >&2
      exit 1
    fi
    if ! cmp -s "$temporary_dir/$generated_file" "$generated_dir/$generated_file"; then
      echo "generated Wasm binding is stale: bindings/wasm/$generated_file" >&2
      exit 1
    fi
  done
  exit 0
fi

mkdir -p "$generated_dir"
for generated_file in $generated_files; do
  cp "$temporary_dir/$generated_file" "$generated_dir/$generated_file"
done
