from __future__ import annotations

from pathlib import Path

import open_shogi_training.selfplay.runtime_receipt as runtime_module
import pytest
from open_shogi_training.labeling.execution import ExecutableSnapshotError
from open_shogi_training.selfplay.common import ContractError
from open_shogi_training.selfplay.runtime_receipt import (
    PythonRuntimeAuthority,
    RuntimeDependencyGuard,
    validate_python_runtime_receipt,
)


def test_python_runtime_receipt_binds_dependency_trees_and_uv_lock(
    tmp_path: Path,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"version = 1\n")
    source = {"treeSha256": "1" * 64, "files": 1, "bytes": 1}
    authority = PythonRuntimeAuthority.create(
        tmp_path,
        git_commit="a" * 40,
        source_snapshot=source,
    )
    reference = authority.receipt
    try:
        receipt = validate_python_runtime_receipt(
            tmp_path,
            reference,
            expected_git_commit="a" * 40,
        )
        assert receipt["sourceSnapshot"] == source
        assert receipt["uvLock"]["path"] == "uv.lock"
        assert receipt["stdlib"]["files"] > 1
        assert receipt["sitePackages"]["files"] > 1
        authority.assert_unchanged()
    finally:
        authority.close()

    (tmp_path / "uv.lock").write_bytes(b"version = 2\n")
    with pytest.raises(ContractError, match=r"uv\.lock identity changed"):
        validate_python_runtime_receipt(
            tmp_path,
            reference,
            expected_git_commit="a" * 40,
        )


def test_python_runtime_tree_guard_fails_closed_on_dependency_drift(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    dependency = runtime / "dependency.py"
    dependency.write_bytes(b"VALUE = 1\n")
    guard = RuntimeDependencyGuard.create((runtime,))
    try:
        dependency.write_bytes(b"VALUE = 2\n")
        with pytest.raises(ContractError, match="runtime path changed"):
            guard.assert_unchanged()
    finally:
        guard.close()


class _CloseProbe:
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.close_error = close_error

    def close(self) -> None:
        self.calls.append(self.name)
        if self.close_error is not None:
            raise self.close_error


def test_python_runtime_authority_close_attempts_every_guard_after_first_failure() -> None:
    calls: list[str] = []
    authority = object.__new__(PythonRuntimeAuthority)
    authority.interpreter = _CloseProbe(
        "interpreter",
        calls,
        close_error=OSError("interpreter close failed"),
    )
    authority._tree_guard = _CloseProbe("tree", calls)
    authority._library_guard = _CloseProbe("library", calls)

    with pytest.raises(OSError, match="interpreter close failed"):
        authority.close()

    assert calls == ["interpreter", "tree", "library"]


def test_python_runtime_authority_create_closes_all_guards_when_final_assert_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"version = 1\n")
    calls: list[str] = []

    class FakeLibrary(_CloseProbe):
        def __init__(self, name: str, calls: list[str]) -> None:
            super().__init__(name, calls)
            self.identity = {"path": "/fixture/Python", "sha256": "2" * 64, "size": 1}

        def assert_unchanged(self) -> None:
            pass

    class FakeTree(_CloseProbe):
        def __init__(self, name: str, calls: list[str]) -> None:
            super().__init__(name, calls)
            self.identities = (
                {
                    "root": "/fixture/stdlib",
                    "treeSha256": "3" * 64,
                    "entries": 1,
                    "files": 1,
                    "symlinks": 0,
                    "bytes": 1,
                },
                {
                    "root": "/fixture/site-packages",
                    "treeSha256": "4" * 64,
                    "entries": 1,
                    "files": 1,
                    "symlinks": 0,
                    "bytes": 1,
                },
            )

        def assert_unchanged(self) -> None:
            pass

    class FakeInterpreter(_CloseProbe):
        def assert_snapshot_unchanged(self) -> None:
            raise ExecutableSnapshotError("final interpreter barrier failed")

        def assert_source_unchanged(self) -> None:
            pass

    library = FakeLibrary("library", calls)
    tree = FakeTree("tree", calls)
    interpreter = FakeInterpreter("interpreter", calls)
    monkeypatch.setattr(
        runtime_module,
        "_hash_file",
        lambda *_args, **_kwargs: {
            "path": "/fixture/python",
            "sha256": "1" * 64,
            "size": 1,
        },
    )
    monkeypatch.setattr(runtime_module.RuntimeFileGuard, "create", lambda *_: library)
    monkeypatch.setattr(runtime_module.RuntimeDependencyGuard, "create", lambda *_: tree)
    monkeypatch.setattr(
        runtime_module.ExecutableSnapshot,
        "create",
        lambda *_args, **_kwargs: interpreter,
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_python_runtime_receipt",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ContractError, match="Python interpreter identity changed"):
        PythonRuntimeAuthority.create(
            tmp_path,
            git_commit="a" * 40,
            source_snapshot={"treeSha256": "5" * 64, "files": 1, "bytes": 1},
        )

    assert calls == ["interpreter", "tree", "library"]
