#!/usr/bin/env python3.12
"""Repository build-cache maintenance: plan (default) or apply classified cleanup.

The tool only ever touches disposable build/test caches it classified itself.
It never touches tracked files, `local/` evidence, live bindings, the pinned
native CLI, or any path containing a symlink. The default mode prints the plan;
`--apply` performs it. Deleting is leaf-first and idempotent, so an interrupted
run can simply be repeated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = PROJECT_ROOT / "local/maintenance"
ACTIVE_WINDOW_SECONDS = 15 * 60
STALE_ROOTS_KEEP = {"pure", "release"}
CARGO_INTERMEDIATES = ("debug", "tmp", "wasm32-unknown-unknown")
# Subtrees the tool must never classify, even for byte-code caches: everything
# under `local/` is evidence or pinned tooling, and `.venv`/`.uv-cache` are
# reusable Python infrastructure managed by uv itself.
UNCLEANABLE_ROOTS = ("local", ".venv", ".uv-cache", ".git")


def classify(root: Path) -> list[tuple[str, Path, str]]:
    """Return (class, path, reason) for every disposable path that exists.

    Each path is classified at most once; the first matching class wins.
    """
    plan: list[tuple[str, Path, str]] = []
    classified: set[Path] = set()

    def add(class_name: str, path: Path, reason: str) -> None:
        if path not in classified and path.is_dir():
            classified.add(path)
            plan.append((class_name, path, reason))

    target = root / "target"
    if target.is_dir():
        for child in sorted(target.iterdir()):
            if child.name not in STALE_ROOTS_KEEP and (child / "CACHEDIR.TAG").is_file():
                add(
                    "cargo-stale-root",
                    child,
                    "complete cargo build root from an earlier verification round",
                )
        for name in CARGO_INTERMEDIATES:
            add(
                "cargo-pure-intermediate",
                target / "pure" / name,
                "pure-build intermediates; pinned CLI binary and live bindings stay",
            )
        for name in CARGO_INTERMEDIATES:
            path = target / name
            if (target / "CACHEDIR.TAG").is_file():
                add(
                    "cargo-dev-cache",
                    path,
                    "routine development cache; the next build recreates it",
                )
    protected = [root / name for name in UNCLEANABLE_ROOTS]
    for cache in sorted(root.glob("**/__pycache__")):
        if any(cache == guarded or guarded in cache.parents for guarded in protected):
            continue
        add("python-cache", cache, "byte-code cache")
    add("python-cache", root / ".pytest_cache", "pytest cache")
    return plan


def assert_containment(root: Path, path: Path) -> None:
    """Refuse anything outside the repository or reachable through a symlink."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or root_resolved not in resolved.parents:
        raise ValueError(f"path escapes the repository: {path}")
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"refusing a symlinked path: {path}")


def is_recently_active(path: Path, now: float) -> bool:
    """True when anything inside was modified inside the active window."""
    newest = max(
        (entry.stat().st_mtime for entry in path.rglob("*") if entry.is_file()),
        default=0.0,
    )
    return now - newest < ACTIVE_WINDOW_SECONDS


def tracked_prefixes(root: Path) -> set[str]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line for line in output.splitlines() if line}


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the classified paths (default: print the plan only)",
    )
    parser.add_argument(
        "--allow-active",
        action="store_true",
        help="also clean paths modified within the active window (default: skip them)",
    )
    parser.add_argument("--json", action="store_true", help="emit the plan as JSON")
    args = parser.parse_args()

    root = PROJECT_ROOT
    tracked = tracked_prefixes(root)
    now = time.time()
    plan: list[dict[str, object]] = []
    for class_name, path, reason in classify(root):
        assert_containment(root, path)
        relative = path.relative_to(root).as_posix()
        if any(
            tracked_path == relative or tracked_path.startswith(f"{relative}/")
            for tracked_path in tracked
        ):
            raise ValueError(f"refusing to touch a tracked path: {relative}")
        bytes_total = sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
        active = not args.allow_active and is_recently_active(path, now)
        plan.append(
            {
                "class": class_name,
                "path": relative,
                "reason": reason,
                "bytes": bytes_total,
                "skipped_active": active,
            }
        )

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        total = 0
        for entry in plan:
            mark = "SKIP(active)" if entry["skipped_active"] else "ok"
            print(f"[{mark:11}] {entry['path']}: {entry['bytes']} bytes — {entry['reason']}")
            total += int(entry["bytes"])  # type: ignore[arg-type]
        print(f"total: {total} bytes across {len(plan)} paths")

    if not args.apply:
        return 0

    removed: list[dict[str, object]] = []
    for entry in plan:
        if entry["skipped_active"]:
            continue
        path = root / str(entry["path"])  # type: ignore[operator]
        assert_containment(root, path)
        if not path.exists():
            continue
        remove(path)
        removed.append(entry)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    receipt = RECEIPTS / f"cleanup-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.json"
    receipt.write_text(json.dumps(removed, indent=2) + "\n", encoding="utf-8")
    print(f"removed {len(removed)} paths; receipt: {receipt.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
