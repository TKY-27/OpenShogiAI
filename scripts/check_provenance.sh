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
    "open_shogi_wasm.d.ts": "8b899de7246585f5c7ccafe0b5b2dde012eec0af12a8032d5d4e5741be9cf063",
    "open_shogi_wasm.js": "8d36745850bb91e90f93436a9c40d833686cca6c1232ab2c7c9346a59e109023",
    "open_shogi_wasm_bg.wasm": "13ea44c68bc8a239fbaeb87c80e5fac00ab2ae8faa017ea223f8a3deef1ccd52",
    "open_shogi_wasm_bg.wasm.d.ts": "efb20b72f02808e77a8baed92fc80fa2f3e1c9b8ce0709f2cd49ed7259933068",
}
failures: list[str] = []
text = (root / "docs/provenance/source.md").read_text(encoding="utf-8")

for required in sorted(required_provenance):
    if required not in text:
        failures.append(f"docs/provenance/source.md is missing required evidence: {required}")

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
