#!/bin/sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
expected_license_sha256=0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0
observed_license_sha256=$(shasum -a 256 "$project_root/LICENSE" | awk '{print $1}')

if [ "$observed_license_sha256" != "$expected_license_sha256" ]; then
    printf 'FAIL LICENSE SHA-256: expected %s, found %s\n' \
        "$expected_license_sha256" "$observed_license_sha256" >&2
    exit 1
fi

python3.12 - "$project_root" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


root = Path(sys.argv[1]).resolve()
expected = "AGPL-3.0-only"
failures: list[str] = []

with (root / "pyproject.toml").open("rb") as handle:
    if tomllib.load(handle)["project"].get("license") != expected:
        failures.append("pyproject.toml project license is not AGPL-3.0-only")
with (root / "configs/project.toml").open("rb") as handle:
    if tomllib.load(handle)["project"].get("license") != expected:
        failures.append("configs/project.toml project license is not AGPL-3.0-only")
with (root / "Cargo.toml").open("rb") as handle:
    if tomllib.load(handle)["workspace"]["package"].get("license") != expected:
        failures.append("Cargo.toml workspace license is not AGPL-3.0-only")

metadata = subprocess.run(
    ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
)
for package in json.loads(metadata.stdout)["packages"]:
    manifest = Path(package["manifest_path"])
    if root in manifest.parents and package.get("license") != expected:
        failures.append(f"workspace package license mismatch: {package['name']}")

if failures:
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    raise SystemExit(1)

print("AI license scope check passed")
PY
