"""Explicit authorized Phase10V source succession; old scientific freezes stay immutable."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

BASE = "9c289734822298bd93f8dbceb56564f71b132895"
ALLOWED = frozenset(
    {
        "engine/core/src/lib.rs",
        "engine/wasm/src/lib.rs",
        "bindings/wasm/open_shogi_wasm_bg.wasm",
    }
)


def runtime_successor_matches(root: Path, name: str, old: str, actual: str, prior: dict) -> bool:
    if name not in ALLOWED or not isinstance(prior, dict) or prior.get("before") != old:
        return False
    try:
        manifest = json.loads((root / "configs/phase10v/runtime-successors.json").read_text())
        if (
            manifest["schema"] != "open_shogiai_phase10v_runtime_successors/v1"
            or manifest["base_commit"] != BASE
            or set(manifest["files"]) != ALLOWED
            or manifest["files"][name] != {"before": prior["after"], "after": actual}
        ):
            return False
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        historical = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=root)
        return hashlib.sha256(historical).hexdigest() == prior["after"]
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        return False
