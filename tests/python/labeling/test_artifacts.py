from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import open_shogi_training.labeling.artifacts as artifacts_module
import pytest
from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    hash_regular_file,
    iter_jsonl_records,
    jsonl_prefix_digest,
    retire_bound_regular,
    write_json_atomic,
)


def test_hash_rejects_path_replacement_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    replacement.write_bytes(b"b" * (2 * 1024 * 1024))
    real_read = artifacts_module.os.read
    swapped = False

    def replace_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(artifacts_module.os, "read", replace_after_first_read)

    with pytest.raises(ArtifactError, match="changed"):
        hash_regular_file(path, max_bytes=3 * 1024 * 1024)


def test_hash_rejects_ancestor_replacement_even_when_original_file_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    moved = tmp_path / "moved"
    source.mkdir()
    path = source / "artifact.bin"
    path.write_bytes(b"stable source bytes")
    real_read = artifacts_module.os.read
    swapped = False

    def swap_parent_after_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            source.rename(moved)
            source.mkdir()
            (source / path.name).write_bytes(b"replacement bytes")
        return chunk

    monkeypatch.setattr(artifacts_module.os, "read", swap_parent_after_read)

    with pytest.raises(ArtifactError, match="changed"):
        hash_regular_file(path, max_bytes=1024)


def test_jsonl_iterator_rejects_path_replacement_before_completion(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    path.write_bytes(b'{"value":1}\n{"value":2}\n')
    replacement.write_bytes(b'{"value":9}\n')
    records = iter_jsonl_records(
        path,
        max_bytes=1024,
        max_line_bytes=128,
        max_records=10,
    )

    assert next(records) == {"value": 1}
    os.replace(replacement, path)
    with pytest.raises(ArtifactError, match="changed"):
        list(records)


def test_jsonl_prefix_digest_rejects_path_replacement_while_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "records.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    line = b'{"value":"' + b"a" * (1024 * 1024) + b'"}\n'
    path.write_bytes(line + line)
    replacement.write_bytes(b'{"value":9}\n')
    real_read = artifacts_module.os.read
    swapped = False

    def replace_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(artifacts_module.os, "read", replace_after_first_read)

    with pytest.raises(ArtifactError, match="changed"):
        jsonl_prefix_digest(path, size=len(line) * 2, records=2)


def test_atomic_json_cleanup_preserves_a_foreign_temp_relink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "manifest.json"
    owned = tmp_path / "owned-temp"
    real_publish = artifacts_module.publish_regular_at

    def relink_target_after_publication(*args, **kwargs) -> None:
        real_publish(*args, **kwargs)
        target.rename(owned)
        target.write_bytes(b"foreign-do-not-delete")

    monkeypatch.setattr(artifacts_module, "publish_regular_at", relink_target_after_publication)

    with pytest.raises(ArtifactError, match="path changed while it was published"):
        write_json_atomic(target, {"schema": "test/v1"}, replace=False)

    assert owned.read_bytes() == b'{"schema":"test/v1"}\n'
    assert target.read_bytes() == b"foreign-do-not-delete"


def test_retirement_refuses_to_rename_a_foreign_replacement(tmp_path: Path) -> None:
    temporary = tmp_path / "owned.pending"
    temporary.write_bytes(b"owned")
    expected = temporary.lstat()
    retained = tmp_path / "retained-owned"
    temporary.rename(retained)
    temporary.write_bytes(b"foreign-do-not-rename")
    parent = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(ArtifactError, match="changed before retirement"):
            retire_bound_regular(
                parent,
                temporary.name,
                expected,
                display=temporary,
                dispose=True,
            )
    finally:
        os.close(parent)

    assert temporary.read_bytes() == b"foreign-do-not-rename"
    assert retained.read_bytes() == b"owned"


def test_successful_atomic_replacement_does_not_grow_retirement_storage(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.json"
    for value in range(12):
        write_json_atomic(target, {"value": value}, replace=value != 0)

    retired = tmp_path / ".open-shogi-retired"
    assert retired.is_dir()
    assert tuple(retired.iterdir()) == ()


def test_retirement_quota_fails_before_publication_and_preserves_unknown_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"original\n")
    retired = tmp_path / ".open-shogi-retired"
    retired.mkdir()
    for index in range(artifacts_module._MAX_RETIRED_ENTRIES):
        (retired / f"foreign-{index:03d}").write_bytes(b"foreign")

    with pytest.raises(ArtifactError, match="quota is exhausted"):
        write_json_atomic(target, {"value": "replacement"}, replace=True)

    assert target.read_bytes() == b"original\n"
    assert len(tuple(retired.iterdir())) == artifacts_module._MAX_RETIRED_ENTRIES


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork semantics")
def test_directory_authority_lock_is_not_retained_by_a_forked_child(
    tmp_path: Path,
) -> None:
    script = r"""
import os
import sys
from contextlib import suppress
from pathlib import Path

from open_shogi_training.labeling.artifacts import stable_directory_lock

root = Path(sys.argv[1])
first = root / "first"
second = root / "second"
child_result_read, child_result_write = os.pipe()
child_hold_read, child_hold_write = os.pipe()
child_pid = -1
try:
    with stable_directory_lock(first, create=True, exclusive=True, nonblocking=True):
        child_pid = os.fork()
        if child_pid == 0:
            os.close(child_result_read)
            os.close(child_hold_write)
            try:
                try:
                    with stable_directory_lock(
                        second, create=True, exclusive=True, nonblocking=True
                    ):
                        outcome = b"unexpected-lock"
                except BlockingIOError:
                    outcome = b"blocked"
                os.write(child_result_write, outcome)
                os.read(child_hold_read, 1)
            finally:
                os._exit(0)
        os.close(child_result_write)
        child_result_write = -1
        os.close(child_hold_read)
        child_hold_read = -1
        if os.read(child_result_read, 64) != b"blocked":
            raise RuntimeError("forked child acquired the parent directory authority")

    with stable_directory_lock(second, create=True, exclusive=True, nonblocking=True):
        pass
finally:
    if child_hold_write >= 0:
        with suppress(OSError):
            os.write(child_hold_write, b"x")
    if child_pid > 0:
        os.waitpid(child_pid, 0)
    for descriptor in (
        child_result_read,
        child_result_write,
        child_hold_read,
        child_hold_write,
    ):
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
"""
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", script, str(tmp_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=10,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "training"),
            "LC_ALL": "C",
            "LANG": "C",
        },
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
