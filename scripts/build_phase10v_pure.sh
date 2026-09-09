#!/bin/sh
# Build the explicit OSAVAL03 pure runtime.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$project_root/target/pure}"
exec ./scripts/build_phase10t_pure.sh
