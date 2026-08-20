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
    "open_shogi_wasm.d.ts": "19f8a5a9f786666d2a736daff7cedae9fa9c77a55f6e6ed85950882e0e07eee9",
    "open_shogi_wasm.js": "b0862ea56ee808c2feabe2ab67128d317fdbb824856b47fe6fabd5dbfac40d88",
    "open_shogi_wasm_bg.wasm": "67c11f4da7e7e0b7c6c1ace36ab84ab8a5f2ab22a88142218a54f3e0855b1030",
    "open_shogi_wasm_bg.wasm.d.ts": "25de209ae4d389487b6b7db0ae895e935fdd84fa62a9bad0dbadf08ec7c910cd",
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
