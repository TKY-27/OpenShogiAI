"""Fixture tests for the build-cache maintenance tool's destructive selector."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/maintenance_cleanup.py"


@pytest.fixture()
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("maintenance_cleanup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_fixture(root: Path) -> None:
    for parent in [
        "target/review/debug",
        "target/pure/bindings",
        "target/pure/release",
        "target/pure/debug",
        "target/debug",
        "src/module/__pycache__",
        "local/maintenance",
        "local/ebadf-evidence/__pycache__",
        ".venv/lib/__pycache__",
        ".pytest_cache/v",
    ]:
        (root / parent).mkdir(parents=True)
    (root / "target/review/CACHEDIR.TAG").write_text("x")
    (root / "target/review/debug/lib.old").write_text("stale")
    (root / "target/pure/CACHEDIR.TAG").write_text("x")
    (root / "target/pure/bindings/open_shogi_wasm.js").write_text("live binding")
    (root / "target/pure/release/open-shogi-cli").write_text("pinned")
    (root / "target/pure/debug/incremental.bin").write_text("intermediate")
    (root / "target/debug/incremental.bin").write_text("dev cache")
    (root / "target/CACHEDIR.TAG").write_text("x")
    (root / "src/module/m.py").write_text("tracked source")
    (root / "src/module/__pycache__/m.pyc").write_text("bytecode")
    (root / "local/maintenance/keep.txt").write_text("evidence")
    (root / "local/ebadf-evidence/__pycache__/x.pyc").write_text("bytecode")
    (root / ".venv/lib/__pycache__/y.pyc").write_text("bytecode")
    (root / ".pytest_cache/v/cache").write_text("pytest")


def test_plan_reports_only_classified_disposable_paths(tool: ModuleType, tmp_path: Path) -> None:
    build_fixture(tmp_path)
    plan = tool.classify(tmp_path)
    paths = [path.relative_to(tmp_path).as_posix() for _, path, _ in plan]
    assert "target/review" in paths
    assert "target/pure/debug" in paths
    assert "target/debug" in paths
    assert "src/module/__pycache__" in paths
    assert ".pytest_cache" in paths
    # Protected assets are never classified.
    assert "target/pure" not in paths
    assert "target/pure/bindings" not in paths
    assert "target/pure/release" not in paths
    # Nothing under local/, .venv or .uv-cache is ever classified, and each
    # path appears exactly once even when several classes could match.
    assert len(paths) == len(set(paths))
    assert not any(path.startswith("local/") for path in paths)
    assert not any(path.startswith(".venv/") for path in paths)


def test_apply_removes_classified_paths_and_keeps_protected_assets(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_fixture(tmp_path)
    monkeypatch.setattr(tool, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tool, "RECEIPTS", tmp_path / "local/maintenance")
    monkeypatch.setattr(tool, "tracked_prefixes", lambda _root: {"src/module/m.py"})
    monkeypatch.setattr(
        tool,
        "is_recently_active",
        lambda _path, _now: False,
    )
    import sys

    monkeypatch.setattr(sys, "argv", ["maintenance_cleanup.py", "--apply"])
    assert tool.main() == 0
    assert not (tmp_path / "target/review").exists()
    assert not (tmp_path / "target/pure/debug").exists()
    assert not (tmp_path / "target/debug").exists()
    assert not (tmp_path / "src/module/__pycache__").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert (tmp_path / "target/pure/bindings/open_shogi_wasm.js").read_text() == "live binding"
    assert (tmp_path / "target/pure/release/open-shogi-cli").read_text() == "pinned"
    assert (tmp_path / "src/module/m.py").exists()
    assert (tmp_path / "local/maintenance/keep.txt").exists()
    assert (tmp_path / "local/ebadf-evidence/__pycache__/x.pyc").exists()
    assert (tmp_path / ".venv/lib/__pycache__/y.pyc").exists()
    receipts = list((tmp_path / "local/maintenance").glob("cleanup-*.json"))
    assert len(receipts) == 1
    removed = {entry["path"] for entry in json.loads(receipts[0].read_text())}
    assert "target/review" in removed


def test_apply_is_idempotent(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_fixture(tmp_path)
    monkeypatch.setattr(tool, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tool, "RECEIPTS", tmp_path / "local/maintenance")
    monkeypatch.setattr(tool, "tracked_prefixes", lambda _root: set())
    monkeypatch.setattr(tool, "is_recently_active", lambda _path, _now: False)
    import sys

    monkeypatch.setattr(sys, "argv", ["maintenance_cleanup.py", "--apply"])
    assert tool.main() == 0
    assert tool.main() == 0


def test_containment_refuses_escapes_and_symlinks(
    tool: ModuleType, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside-root")
    (outside / "payload").write_text("x")
    link = tmp_path / "target" / "sneaky"
    link.mkdir(parents=True)
    (link / "CACHEDIR.TAG").write_text("x")
    (link / "eval").symlink_to(outside / "payload")
    with pytest.raises(ValueError, match="escapes"):
        tool.assert_containment(tmp_path, link / "eval")
    inside_link = link / "local-tag"
    inside_link.symlink_to(tmp_path / "target" / "CACHEDIR.TAG")
    with pytest.raises(ValueError, match="symlink"):
        tool.assert_containment(tmp_path, inside_link)
    with pytest.raises(ValueError, match="escapes"):
        tool.assert_containment(tmp_path, outside)
    with pytest.raises(ValueError, match="escapes"):
        tool.assert_containment(tmp_path, tmp_path)
