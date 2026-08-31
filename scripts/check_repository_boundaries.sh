#!/bin/sh

set -eu

case "$#" in
    0)
        project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
        ;;
    1)
        project_root=$(CDPATH= cd -- "$1" && pwd)
        ;;
    *)
        echo "usage: $0 [project-root]" >&2
        exit 2
        ;;
esac

python3.12 - "$project_root" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
forbidden_local_roots = {
    "data/raw",
    "data/processed",
    "training/data/raw",
    "training/data/processed",
    "local/phase10r-selection",
    "local/phase10r-data",
    "local/phase10r-runs",
    "local/teacher",
}
forbidden_roots = {"node_modules", "package-lock.json", "package.json", "web"}
ui_suffixes = {".css", ".html", ".tsx"}
artifact_suffixes = {".ckpt", ".nnue", ".onnx", ".pt", ".pth", ".safetensors"}
failures: list[str] = []


def tracked_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"FAIL cannot enumerate Git-tracked files: {error}") from error
    try:
        output = result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit("FAIL Git-tracked file inventory is not valid UTF-8") from error
    if output and not output.endswith("\0"):
        raise SystemExit("FAIL Git-tracked file inventory is not NUL terminated")
    return [relative for relative in output.rstrip("\0").split("\0") if relative]


tracked = tracked_files()


def is_under(relative: str, parent: str) -> bool:
    return relative == parent or relative.startswith(f"{parent}/")


# Keep this check separate from the source-text scan: ignored local artifacts must never enter
# the inventory, while a forced/tracked artifact must fail explicitly.
for relative in tracked:
    if is_under(relative, "local/teacher"):
        failures.append(f"tracked teacher artifact under local/teacher: {relative}")
    elif any(is_under(relative, parent) for parent in forbidden_local_roots):
        failures.append(f"tracked forbidden local path: {relative}")

for name in sorted(forbidden_roots):
    if any(is_under(relative, name) for relative in tracked):
        failures.append(f"forbidden AI-root path: {name}")

required_bindings = {
    "bindings/wasm/open_shogi_wasm.d.ts",
    "bindings/wasm/open_shogi_wasm.js",
    "bindings/wasm/open_shogi_wasm_bg.wasm",
    "bindings/wasm/open_shogi_wasm_bg.wasm.d.ts",
}
for relative in sorted(required_bindings):
    if not (root / relative).is_file():
        failures.append(f"missing versioned Wasm interface: {relative}")

for relative_name in tracked:
    relative = Path(relative_name)
    path = root / relative
    if path.suffix.lower() in ui_suffixes:
        failures.append(f"browser-UI source is outside the AI boundary: {relative}")
    if path.suffix.lower() in artifact_suffixes:
        failures.append(f"model/teacher artifact is not allowed in Git: {relative}")
    if relative.as_posix() == "scripts/check_repository_boundaries.sh":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    local_home_marker = "/" + "Users" + "/"
    local_uri_marker = "file" + "://"
    if local_home_marker in text or local_uri_marker in text:
        failures.append(f"machine-local path leaked into versioned text: {relative}")

if failures:
    for failure in sorted(set(failures)):
        print(f"FAIL {failure}", file=sys.stderr)
    raise SystemExit(1)

print("AI repository boundary check passed")
PY
