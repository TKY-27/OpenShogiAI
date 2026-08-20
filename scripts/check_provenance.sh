#!/bin/sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

python3.12 - "$project_root" <<'PY'
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
private_commit = "5451e02d35abc3efc1fcc29260cdb89acad1d416"
required_provenance = {
    private_commit,
    "21bf0c1fbe02250855b66a96f2fe37bd042650d4",
    "aa3eec7afc3bc9dff5a330eb6c3b3a1ee503baa703073281a906bb3753c01731",
    "OpenShogiAI-private-pre-split.bundle",
    "60504e4fe09686f61f3184008c759a2c0ba43182f4538e67bf170b4717715730",
    "2026-08-21",
    "git archive private-pre-split-source",
    "The private source history was not published",
}
expected_bindings = {
    "open_shogi_wasm.d.ts": "be1db38df5cca0a4c55418b1e35bb8107c2a5d2e2a4e834c0ebad270878f7b7a",
    "open_shogi_wasm.js": "165ee2cc0a034216d919baa9f0230ba312f837ce223dd1af71a8739e59dd3d4f",
    "open_shogi_wasm_bg.wasm": "7d3f7f06e0309978eb0f7aa9107721384e14062b724ca00fcee9c626ee8a1caf",
    "open_shogi_wasm_bg.wasm.d.ts": "3c17285e297fbc61c9e3ad306270ecf7361a5f2032582b56a8524b91877d85c8",
}
failures: list[str] = []
text = (root / "PROVENANCE.md").read_text(encoding="utf-8")

for required in sorted(required_provenance):
    if required not in text:
        failures.append(f"PROVENANCE.md is missing required evidence: {required}")

for name, expected in expected_bindings.items():
    observed = hashlib.sha256((root / "bindings/wasm" / name).read_bytes()).hexdigest()
    if observed != expected:
        failures.append(f"binding provenance mismatch for {name}: {observed}")

private_object = subprocess.run(
    ["git", "cat-file", "-e", f"{private_commit}^{{commit}}"],
    cwd=root,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
if private_object.returncode == 0:
    failures.append("private source commit is present in the clean candidate object database")

if failures:
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    raise SystemExit(1)

print("AI provenance check passed")
PY
