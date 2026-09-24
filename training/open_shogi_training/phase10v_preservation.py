"""Byte identities for rejected research evidence; never opens final holdout data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .phase10t import safe_path, sha256, tree_identity


def verify(root: Path, manifest: Path) -> dict:
    value = json.loads(manifest.read_text())
    if value.get("schema") != "open_shogiai_phase10v_preservation/v1":
        raise ValueError("unsupported preservation schema")
    if not value.get("files") or not value.get("trees"):
        raise ValueError("empty preservation inventory")
    for row in value["files"]:
        path = safe_path(root, row["path"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"preserved file changed: {row['path']}")
    for row in value["trees"]:
        if tree_identity(root, row["path"]) != row:
            raise ValueError(f"preserved tree changed: {row['path']}")
    return {
        "status": "PASS",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "files": len(value["files"]),
        "trees": len(value["trees"]),
        "final_holdout_opened": False,
    }


def main() -> None:
    raise SystemExit("Closed campaign; use current development commands")


if __name__ == "__main__":
    main()
