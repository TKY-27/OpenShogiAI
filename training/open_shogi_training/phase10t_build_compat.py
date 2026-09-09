"""Explicit Phase 10T runtime successors without rewriting the historical 10R freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

BASE_COMMIT = "93d2ec1303c46000cdebd4c868f18b3ab8b51c9c"
ALLOWED = frozenset(
    {
        "engine/core/src/lib.rs",
        "engine/core/src/position.rs",
        "engine/core/src/phase10r.rs",
        "engine/wasm/src/lib.rs",
        "bindings/wasm/open_shogi_wasm.d.ts",
        "bindings/wasm/open_shogi_wasm.js",
        "bindings/wasm/open_shogi_wasm_bg.wasm",
        "bindings/wasm/open_shogi_wasm_bg.wasm.d.ts",
        "training/open_shogi_training/phase10r.py",
        "training/open_shogi_training/phase10r_run.py",
        "tests/python/test_phase10r_freeze.py",
        "tests/python/test_phase10r_execution.py",
    }
)


def runtime_successor_matches(root: Path, name: str, old: str, actual: str) -> bool:
    """Verify both historical content and the explicitly pinned current runtime digest.

    This does not authorize a historical 10R campaign with a new engine. New runtime
    evidence belongs to Phase 10T; all dataset/model/control freezes remain unchanged.
    """
    if name not in ALLOWED:
        return False
    try:
        manifest = json.loads((root / "configs/phase10t/runtime-successors.json").read_text())
        if (
            manifest.get("schema") != "open_shogiai_phase10t_runtime_successors/v1"
            or manifest.get("base_commit") != BASE_COMMIT
            or not set(manifest["files"]).issubset(ALLOWED)
        ):
            return False
        if manifest["files"].get(name) != {"before": old, "after": actual}:
            from .phase10v_build_compat import runtime_successor_matches as phase10v_matches

            return phase10v_matches(root, name, old, actual, manifest["files"].get(name))
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        original = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{name}"], cwd=root)
        return hashlib.sha256(original).hexdigest() == old
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        return False
