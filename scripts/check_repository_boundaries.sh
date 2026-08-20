#!/bin/sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

python3.12 - "$project_root" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
ignored_parts = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "target",
}
forbidden_roots = {"node_modules", "package-lock.json", "package.json", "web"}
ui_suffixes = {".css", ".html", ".tsx"}
artifact_suffixes = {".ckpt", ".nnue", ".onnx", ".pt", ".pth", ".safetensors"}
failures: list[str] = []

for name in sorted(forbidden_roots):
    if (root / name).exists():
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

for path in root.rglob("*"):
    relative = path.relative_to(root)
    if any(part in ignored_parts for part in relative.parts):
        continue
    if path.is_dir():
        if relative.parts[:2] in {("data", "raw"), ("data", "processed"), ("local", "teacher")}:
            failures.append(f"forbidden local payload directory: {relative}")
        continue
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
