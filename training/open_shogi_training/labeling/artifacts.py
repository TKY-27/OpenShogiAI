"""Bounded append-only and atomic artifact primitives for labeling."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from ctypes import CDLL, c_char_p, c_int, get_errno
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """Raised when a local artifact is malformed or unsafe to access."""


class DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FileDigest:
    sha256: str
    size: int
    records: int | None = None

    def as_dict(self) -> dict[str, int | str | None]:
        return {"sha256": self.sha256, "size": self.size, "records": self.records}


@contextmanager
def stable_regular_descriptor(path: Path):
    """Yield one ancestor-safe descriptor and reject mutation or entry replacement."""

    parent_descriptor, name = _open_parent_descriptor(path, create=False)
    descriptor = -1
    parent_status = os.fstat(parent_descriptor)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ArtifactError(f"{path} must be a non-symlink regular file")
        yield descriptor
        final = os.fstat(descriptor)
        if _file_state(final) != _file_state(initial):
            raise ArtifactError(f"{path.name} changed while it was consumed")
        _assert_linked_descriptor(
            parent_descriptor,
            name,
            final,
            display=path,
            operation="consumed",
        )
        _assert_parent_path(path, parent_status, operation="consumed")
    except OSError as error:
        raise ArtifactError(f"cannot open regular file {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


@contextmanager
def stable_parent_descriptor(path: Path, *, create: bool):
    """Yield the pinned parent directory descriptor and final component."""

    parent_descriptor, name = _open_parent_descriptor(path, create=create)
    parent_status = os.fstat(parent_descriptor)
    try:
        yield parent_descriptor, name
        _assert_parent_path(path, parent_status, operation="used")
    finally:
        os.close(parent_descriptor)


@contextmanager
def stable_directory_lock(
    directory: Path,
    *,
    create: bool,
    exclusive: bool,
    nonblocking: bool,
):
    """Lock a directory through a filesystem-root-anchored descriptor chain.

    A conventional ``.lock`` entry can be unlinked and recreated while its old
    inode remains locked, allowing a second writer to acquire a different lock.
    Locking only the target and immediate parent has the same problem if the parent
    itself is renamed and recreated. Cooperative writers therefore retain and lock
    every directory descriptor from the filesystem root through the exact target,
    in root-to-leaf order. The common root lock prevents a second writer from
    walking a replacement ancestry while the first authority is held.
    """

    descriptors, components, registrations = _open_directory_descriptor_chain(
        directory, create=create
    )
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if nonblocking:
        operation |= fcntl.LOCK_NB
    locked: list[tuple[int, object]] = []
    try:
        for descriptor, registration in registrations:
            fcntl.flock(descriptor, operation)
            locked.append((descriptor, registration))
        for index, component in enumerate(components, start=1):
            _assert_linked_descriptor(
                descriptors[index - 1],
                component,
                os.fstat(descriptors[index]),
                display=directory,
                operation="locked",
            )
        yield descriptors[-1]
        for index, component in enumerate(components, start=1):
            _assert_linked_descriptor(
                descriptors[index - 1],
                component,
                os.fstat(descriptors[index]),
                display=directory,
                operation="locked",
            )
    finally:
        for descriptor, registration in reversed(locked):
            if _owns_directory_lock_descriptor(descriptor, registration):
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        for descriptor, registration in reversed(registrations):
            if _release_directory_lock_descriptor(descriptor, registration):
                os.close(descriptor)


_ACTIVE_DIRECTORY_LOCK_DESCRIPTORS: dict[int, object] = {}


def _register_directory_lock_descriptor(descriptor: int) -> object:
    registration = object()
    if descriptor in _ACTIVE_DIRECTORY_LOCK_DESCRIPTORS:
        raise ArtifactError("directory lock descriptor registration collided")
    _ACTIVE_DIRECTORY_LOCK_DESCRIPTORS[descriptor] = registration
    return registration


def _owns_directory_lock_descriptor(descriptor: int, registration: object) -> bool:
    return _ACTIVE_DIRECTORY_LOCK_DESCRIPTORS.get(descriptor) is registration


def _release_directory_lock_descriptor(descriptor: int, registration: object) -> bool:
    if not _owns_directory_lock_descriptor(descriptor, registration):
        return False
    del _ACTIVE_DIRECTORY_LOCK_DESCRIPTORS[descriptor]
    return True


def _close_inherited_directory_lock_descriptors() -> None:
    inherited = tuple(_ACTIVE_DIRECTORY_LOCK_DESCRIPTORS)
    _ACTIVE_DIRECTORY_LOCK_DESCRIPTORS.clear()
    for descriptor in inherited:
        with suppress(OSError):
            os.close(descriptor)


os.register_at_fork(after_in_child=_close_inherited_directory_lock_descriptors)


def _open_directory_descriptor_chain(
    directory: Path, *, create: bool
) -> tuple[list[int], tuple[str, ...], list[tuple[int, object]]]:
    """Open and retain each non-symlink directory from ``/`` to ``directory``."""

    absolute = directory.absolute()
    parts = absolute.parts
    if len(parts) < 2:
        raise ArtifactError(f"lock path has no named component: {directory}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    registrations: list[tuple[int, object]] = []
    components = tuple(parts[1:])
    try:
        descriptor = os.open(parts[0], flags)
        descriptors.append(descriptor)
        registrations.append((descriptor, _register_directory_lock_descriptor(descriptor)))
        for component in components:
            if component in {"", ".", ".."}:
                raise ArtifactError("lock path contains an unsafe directory component")
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, 0o700, dir_fd=descriptors[-1])
            descriptor = os.open(component, flags, dir_fd=descriptors[-1])
            descriptors.append(descriptor)
            registrations.append((descriptor, _register_directory_lock_descriptor(descriptor)))
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ArtifactError(f"lock authority is not a directory: {directory}")
    except BaseException:
        for descriptor, registration in reversed(registrations):
            if _release_directory_lock_descriptor(descriptor, registration):
                os.close(descriptor)
        raise
    return descriptors, components, registrations


_RETIRED_DIRECTORY = ".open-shogi-retired"
_MAX_RETIRED_ENTRIES = 64
_MAX_RETIRED_BYTES = 1024 * 1024 * 1024
_MAX_RETIRED_TREE_NODES = 16_384


def assert_retirement_capacity(parent_descriptor: int) -> None:
    """Fail before publication when retained, unproven quarantine exceeds its cap.

    Historical quarantine entries are deliberately never inferred to be ours from
    their names.  They remain untouched and count against a conservative local
    bound.  Entries retired by the current operation may be disposed only while
    their original descriptor remains open and proves the moved inode identity.
    """

    try:
        retired_descriptor = os.open(
            _RETIRED_DIRECTORY,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError:
        return
    except OSError as error:
        raise ArtifactError("retirement quarantine authority is unsafe") from error
    try:
        names = os.listdir(retired_descriptor)
        if len(names) >= _MAX_RETIRED_ENTRIES:
            raise ArtifactError("retirement quarantine entry quota is exhausted")
        nodes = 0
        total = 0
        for name in names:
            nodes, total = _measure_retained_entry(
                retired_descriptor,
                name,
                nodes=nodes,
                total=total,
            )
    finally:
        os.close(retired_descriptor)


def _measure_retained_entry(
    parent_descriptor: int,
    name: str,
    *,
    nodes: int,
    total: int,
) -> tuple[int, int]:
    status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    nodes += 1
    if nodes > _MAX_RETIRED_TREE_NODES:
        raise ArtifactError("retirement quarantine node quota is exhausted")
    if stat.S_ISREG(status.st_mode):
        total += status.st_size
    elif stat.S_ISDIR(status.st_mode):
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            for child in os.listdir(descriptor):
                nodes, total = _measure_retained_entry(
                    descriptor,
                    child,
                    nodes=nodes,
                    total=total,
                )
        finally:
            os.close(descriptor)
    else:
        # Unknown historical bytes stay retained, but cannot evade the node cap.
        total += max(0, status.st_size)
    if total > _MAX_RETIRED_BYTES:
        raise ArtifactError("retirement quarantine byte quota is exhausted")
    return nodes, total


def retire_bound_regular(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    display: Path,
    dispose: bool = False,
) -> bool:
    """Atomically remove an entry from authority without deleting raced-in bytes.

    There is no portable unlink-if-inode primitive. A stat-then-unlink sequence can
    therefore delete a foreign entry swapped into ``name`` between the two calls.
    Move the entry with no-replace semantics into a retained quarantine, validate
    the moved inode, and restore a mismatched entry when possible. Exact owned bytes
    remain recoverable instead of being destructively cleaned up.
    """

    return _retire_bound_entry(
        parent_descriptor,
        name,
        expected,
        display=display,
        directory=False,
        dispose=dispose,
    )


def retire_bound_directory(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    display: Path,
    dispose: bool = False,
) -> bool:
    """Retire one exact directory inode without recursively deleting raced-in data."""

    return _retire_bound_entry(
        parent_descriptor,
        name,
        expected,
        display=display,
        directory=True,
        dispose=dispose,
    )


def _retire_bound_entry(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    display: Path,
    directory: bool,
    dispose: bool,
) -> bool:
    expected_kind = stat.S_ISDIR(expected.st_mode) if directory else stat.S_ISREG(expected.st_mode)
    if not expected_kind:
        kind = "directory" if directory else "regular file"
        raise ArtifactError(f"temporary artifact is not a {kind}: {display}")
    assert_retirement_capacity(parent_descriptor)
    with suppress(FileExistsError):
        os.mkdir(_RETIRED_DIRECTORY, 0o700, dir_fd=parent_descriptor)
    retired_descriptor = os.open(
        _RETIRED_DIRECTORY,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    source_descriptor = -1
    retired_name = f"{name}.{expected.st_dev:x}.{expected.st_ino:x}.{secrets.token_hex(12)}.retired"
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            source_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            return False
        opened = os.fstat(source_descriptor)
        identity_changed = (opened.st_dev, opened.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        )
        if identity_changed or (not directory and _file_state(opened) != _file_state(expected)):
            raise ArtifactError(f"temporary artifact path changed before retirement: {display}")
        try:
            _rename_noreplace(
                parent_descriptor,
                name,
                retired_descriptor,
                retired_name,
            )
        except FileNotFoundError:
            return False
        moved = os.stat(retired_name, dir_fd=retired_descriptor, follow_symlinks=False)
        moved_kind = stat.S_ISDIR(moved.st_mode) if directory else stat.S_ISREG(moved.st_mode)
        if (
            stat.S_ISLNK(moved.st_mode)
            or not moved_kind
            or (
                moved.st_dev,
                moved.st_ino,
            )
            != (expected.st_dev, expected.st_ino)
        ):
            with suppress(OSError):
                _rename_noreplace(
                    retired_descriptor,
                    retired_name,
                    parent_descriptor,
                    name,
                )
            raise ArtifactError(f"temporary artifact path changed before retirement: {display}")
        os.fsync(retired_descriptor)
        os.fsync(parent_descriptor)
        if dispose:
            if directory:
                _dispose_retired_directory(
                    retired_descriptor,
                    retired_name,
                    source_descriptor,
                    opened,
                    display=display,
                )
            else:
                _dispose_retired_regular(
                    retired_descriptor,
                    retired_name,
                    source_descriptor,
                    opened,
                    display=display,
                )
            os.fsync(retired_descriptor)
            os.fsync(parent_descriptor)
        return True
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(retired_descriptor)


def _dispose_retired_regular(
    retired_descriptor: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    display: Path,
) -> None:
    linked = os.stat(name, dir_fd=retired_descriptor, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(linked.st_mode)
        or _retirement_state(linked) != _retirement_state(expected)
        or _retirement_state(opened) != _retirement_state(expected)
        or opened.st_uid != os.getuid()
        or opened.st_nlink != 1
    ):
        raise ArtifactError(f"refusing to dispose unproven retired file: {display}")
    os.unlink(name, dir_fd=retired_descriptor)
    if os.fstat(descriptor).st_nlink != 0:
        raise ArtifactError(f"retired file link survived disposal: {display}")


def _dispose_retired_directory(
    retired_descriptor: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    display: Path,
) -> None:
    linked = os.stat(name, dir_fd=retired_descriptor, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(linked.st_mode)
        or _retirement_state(linked) != _retirement_state(expected)
        or _retirement_state(opened) != _retirement_state(expected)
        or opened.st_uid != os.getuid()
    ):
        raise ArtifactError(f"refusing to dispose unproven retired directory: {display}")
    _measure_disposable_tree(descriptor, nodes=0, total=0)
    _clear_disposable_tree(descriptor, display=display)
    final = os.stat(name, dir_fd=retired_descriptor, follow_symlinks=False)
    if (final.st_dev, final.st_ino, final.st_mode) != (
        expected.st_dev,
        expected.st_ino,
        expected.st_mode,
    ) or os.listdir(descriptor):
        raise ArtifactError(f"retired directory changed during disposal: {display}")
    os.rmdir(name, dir_fd=retired_descriptor)


def _measure_disposable_tree(
    descriptor: int,
    *,
    nodes: int,
    total: int,
) -> tuple[int, int]:
    for name in os.listdir(descriptor):
        status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        nodes += 1
        if nodes > _MAX_RETIRED_TREE_NODES:
            raise ArtifactError("owned retirement tree exceeds its node bound")
        if stat.S_ISREG(status.st_mode):
            if status.st_uid != os.getuid() or status.st_nlink != 1:
                raise ArtifactError("owned retirement tree contains an unproven file")
            total += status.st_size
        elif stat.S_ISDIR(status.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                child_status = os.fstat(child)
                if (
                    _file_state(child_status) != _file_state(status)
                    or child_status.st_uid != os.getuid()
                ):
                    raise ArtifactError("owned retirement directory changed during proof")
                nodes, total = _measure_disposable_tree(
                    child,
                    nodes=nodes,
                    total=total,
                )
            finally:
                os.close(child)
        else:
            raise ArtifactError("owned retirement tree contains a special entry")
        if total > _MAX_RETIRED_BYTES:
            raise ArtifactError("owned retirement tree exceeds its byte bound")
    return nodes, total


def _clear_disposable_tree(descriptor: int, *, display: Path) -> None:
    for name in os.listdir(descriptor):
        linked = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(linked.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                if _file_state(opened) != _file_state(linked):
                    raise ArtifactError(f"retired directory entry changed: {display}")
                _clear_disposable_tree(child, display=display)
            finally:
                os.close(child)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (current.st_dev, current.st_ino, current.st_mode) != (
                linked.st_dev,
                linked.st_ino,
                linked.st_mode,
            ):
                raise ArtifactError(f"retired directory entry changed: {display}")
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(linked.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                if (
                    _file_state(opened) != _file_state(linked)
                    or opened.st_uid != os.getuid()
                    or opened.st_nlink != 1
                ):
                    raise ArtifactError(f"retired file entry changed: {display}")
                os.unlink(name, dir_fd=descriptor)
                if os.fstat(child).st_nlink != 0:
                    raise ArtifactError(f"retired file link survived disposal: {display}")
            finally:
                os.close(child)
        else:
            raise ArtifactError(f"retired directory contains a special entry: {display}")


def publish_regular_at(
    parent_descriptor: int,
    temporary_name: str,
    name: str,
    temporary_status: os.stat_result,
    *,
    display: Path,
    replace: bool,
) -> None:
    """Publish one retained temp without overwriting unpreserved predecessor bytes."""

    assert_retirement_capacity(parent_descriptor)
    if not stat.S_ISREG(temporary_status.st_mode):
        raise ArtifactError(f"temporary artifact is not regular: {display}")
    try:
        existing = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)
    ):
        raise ArtifactError(f"refusing to replace non-regular artifact {display}")
    if not replace or existing is None:
        _rename_noreplace(parent_descriptor, temporary_name, parent_descriptor, name)
        return

    _rename_exchange(parent_descriptor, temporary_name, parent_descriptor, name)
    published = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    predecessor = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (published.st_dev, published.st_ino) != (
        temporary_status.st_dev,
        temporary_status.st_ino,
    ) or (predecessor.st_dev, predecessor.st_ino) != (existing.st_dev, existing.st_ino):
        with suppress(OSError):
            _rename_exchange(parent_descriptor, temporary_name, parent_descriptor, name)
        raise ArtifactError(f"artifact path changed during compare-and-swap publication: {display}")
    retire_bound_regular(
        parent_descriptor,
        temporary_name,
        predecessor,
        display=display,
        dispose=True,
    )


def _rename_noreplace(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    _rename_with_flags(
        source_descriptor,
        source_name,
        destination_descriptor,
        destination_name,
        darwin_flags=0x00000004,
        linux_flags=0x00000001,
    )


def _rename_exchange(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    _rename_with_flags(
        source_descriptor,
        source_name,
        destination_descriptor,
        destination_name,
        darwin_flags=0x00000002,
        linux_flags=0x00000002,
    )


def _rename_with_flags(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
    *,
    darwin_flags: int,
    linux_flags: int,
) -> None:
    """Call the public Darwin/Linux atomic rename-with-flags boundary."""

    library = CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = library.renameatx_np
        flags = darwin_flags
    elif sys.platform.startswith("linux"):
        function = library.renameat2
        flags = linux_flags
    else:
        raise ArtifactError("atomic no-replace/exchange rename is unsupported on this platform")
    function.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]
    function.restype = c_int
    result = function(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error_number = get_errno()
    if error_number == 17:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    if error_number == 2:
        raise FileNotFoundError(error_number, os.strerror(error_number), source_name)
    raise OSError(error_number, os.strerror(error_number), source_name)


def compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def read_regular_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, FileDigest]:
    """Read one non-symlink regular file from a stable descriptor."""

    with stable_regular_descriptor(path) as descriptor:
        status = os.fstat(descriptor)
        if status.st_size > max_bytes:
            raise ArtifactError(f"{path.name} exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise ArtifactError(f"{path.name} exceeds {max_bytes} bytes")
            chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != status.st_size:
            raise ArtifactError(f"{path.name} changed while it was read")
    return raw, FileDigest(hashlib.sha256(raw).hexdigest(), len(raw))


def hash_regular_file(path: Path, *, max_bytes: int) -> FileDigest:
    digest = hashlib.sha256()
    size = 0
    with stable_regular_descriptor(path) as descriptor:
        status = os.fstat(descriptor)
        if status.st_size > max_bytes:
            raise ArtifactError(f"{path.name} exceeds {max_bytes} bytes")
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ArtifactError(f"{path.name} exceeds {max_bytes} bytes")
            digest.update(chunk)
        if size != status.st_size:
            raise ArtifactError(f"{path.name} changed while it was hashed")
    return FileDigest(digest.hexdigest(), size)


def load_json_object(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], FileDigest]:
    raw, digest = read_regular_bytes(path, max_bytes=max_bytes)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (DuplicateJsonKeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid JSON object {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name} must contain a JSON object")
    return value, digest


def iter_jsonl_records(
    path: Path,
    *,
    max_bytes: int,
    max_line_bytes: int,
    max_records: int,
) -> Iterator[dict[str, Any]]:
    """Iterate a bounded, complete, LF-terminated append-only JSONL file."""

    with stable_regular_descriptor(path) as descriptor:
        for value, _ in _iter_jsonl_descriptor(
            descriptor,
            path,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            max_records=max_records,
        ):
            yield value


def iter_jsonl_descriptor_records(
    descriptor: int,
    *,
    display_path: Path,
    max_bytes: int,
    max_line_bytes: int,
    max_records: int,
) -> Iterator[dict[str, Any]]:
    """Consume JSONL from an already identity-verified, rewound descriptor."""

    for value, _ in _iter_jsonl_descriptor(
        descriptor,
        display_path,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
        max_records=max_records,
    ):
        yield value


def append_jsonl_record(path: Path, value: Mapping[str, Any], *, max_line_bytes: int) -> int:
    """Append and fsync exactly one compact JSON line through O_NOFOLLOW."""

    payload = compact_json_bytes(value) + b"\n"
    if len(payload) > max_line_bytes:
        raise ArtifactError(f"JSONL record exceeds {max_line_bytes} bytes")
    parent_descriptor, name = _open_parent_descriptor(path, create=True)
    parent_status = os.fstat(parent_descriptor)
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        except OSError as error:
            raise ArtifactError(f"cannot open append-only artifact {path.name}: {error}") from error
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ArtifactError(f"{path.name} must be a regular file")
        if status.st_size and os.pread(descriptor, 1, status.st_size - 1) != b"\n":
            raise ArtifactError(f"{path.name} has an incomplete final line")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise ArtifactError(f"could not append a complete record to {path.name}")
            written += count
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if final.st_size != status.st_size + len(payload):
            raise ArtifactError(f"{path.name} changed concurrently while it was appended")
        _assert_linked_descriptor(
            parent_descriptor,
            name,
            final,
            display=path,
            operation="appended",
        )
        _assert_parent_path(path, parent_status, operation="appended")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    return len(payload)


def write_json_atomic(path: Path, value: object, *, replace: bool) -> FileDigest:
    """Publish a compact JSON manifest atomically and fsync its directory."""

    payload = compact_json_bytes(value) + b"\n"
    parent_descriptor, name = _open_parent_descriptor(path, create=True)
    parent_status = os.fstat(parent_descriptor)
    temporary_name = f".{name}.{secrets.token_hex(12)}"
    descriptor = -1
    temporary_created = False
    temporary_status: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        temporary_status = os.fstat(descriptor)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            published_status = os.fstat(output.fileno())
            temporary_status = published_status
        assert temporary_status is not None
        try:
            publish_regular_at(
                parent_descriptor,
                temporary_name,
                name,
                temporary_status,
                display=path,
                replace=replace,
            )
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {path}") from None
        temporary_created = False
        _assert_published_descriptor(
            parent_descriptor,
            name,
            path,
            published_status,
            payload,
        )
        os.fsync(parent_descriptor)
        _assert_parent_path(path, parent_status, operation="published")
    except BaseException:
        if temporary_created:
            assert temporary_status is not None
            retire_bound_regular(
                parent_descriptor,
                temporary_name,
                temporary_status,
                display=path,
            )
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    return FileDigest(hashlib.sha256(payload).hexdigest(), len(payload), records=1)


def jsonl_digest(
    path: Path, *, max_bytes: int, max_line_bytes: int, max_records: int
) -> FileDigest:
    sha256 = hashlib.sha256()
    size = 0
    records = 0
    with stable_regular_descriptor(path) as descriptor:
        for _, line in _iter_jsonl_descriptor(
            descriptor,
            path,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            max_records=max_records,
        ):
            sha256.update(line)
            size += len(line)
            records += 1
    return FileDigest(sha256.hexdigest(), size, records)


def jsonl_prefix_digest(path: Path, *, size: int, records: int) -> FileDigest:
    """Hash one previously committed JSONL prefix without trusting later bytes.

    Progress manifests are checkpoints, not truncation authorities.  A process can
    be interrupted after an fsynced append but before the next manifest replace,
    so a valid resume may contain an uncommitted tail.  The complete byte range
    recorded by the manifest must nevertheless still exist verbatim and end on a
    record boundary.
    """

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ArtifactError("recorded JSONL prefix size must be a non-negative integer")
    if isinstance(records, bool) or not isinstance(records, int) or records < 0:
        raise ArtifactError("recorded JSONL prefix count must be a non-negative integer")
    digest = hashlib.sha256()
    observed = 0
    newlines = 0
    last_byte = b""
    with stable_regular_descriptor(path) as descriptor:
        status = os.fstat(descriptor)
        if status.st_size < size:
            raise ArtifactError(f"{path.name} is shorter than its recorded checkpoint")
        while observed < size:
            chunk = os.read(descriptor, min(1024 * 1024, size - observed))
            if not chunk:
                raise ArtifactError(f"{path.name} ended inside its recorded checkpoint")
            observed += len(chunk)
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
            digest.update(chunk)
    if size and last_byte != b"\n":
        raise ArtifactError(f"{path.name} checkpoint does not end on a JSONL boundary")
    if newlines != records:
        raise ArtifactError(
            f"{path.name} checkpoint record count {newlines} disagrees with {records}"
        )
    return FileDigest(digest.hexdigest(), size, records)


def _iter_jsonl_descriptor(
    descriptor: int,
    path: Path,
    *,
    max_bytes: int,
    max_line_bytes: int,
    max_records: int,
) -> Iterator[tuple[dict[str, Any], bytes]]:
    status = os.fstat(descriptor)
    if status.st_size > max_bytes:
        raise ArtifactError(f"{path.name} exceeds {max_bytes} bytes")
    with os.fdopen(descriptor, "rb", closefd=False) as input_file:
        line_number = 0
        observed = 0
        while line := input_file.readline(max_line_bytes + 1):
            line_number += 1
            observed += len(line)
            if observed > max_bytes:
                raise ArtifactError(f"{path.name} exceeds {max_bytes} bytes")
            if line_number > max_records:
                raise ArtifactError(f"{path.name} exceeds {max_records} records")
            if len(line) > max_line_bytes:
                raise ArtifactError(
                    f"{path.name} line {line_number} exceeds {max_line_bytes} bytes"
                )
            if not line.endswith(b"\n"):
                raise ArtifactError(f"{path.name} has an incomplete final line")
            try:
                value = json.loads(line, object_pairs_hook=_unique_object)
            except (DuplicateJsonKeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArtifactError(
                    f"invalid JSON on {path.name} line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ArtifactError(f"{path.name} line {line_number} must be an object")
            yield value, line
    if observed != status.st_size:
        raise ArtifactError(f"{path.name} changed while it was read")
    current = os.fstat(descriptor)
    if _file_state(current) != _file_state(status):
        raise ArtifactError(f"{path.name} changed while it was read")


def _file_state(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _retirement_state(status: os.stat_result) -> tuple[int, int, int, int, int]:
    """Identity fields stable across the intentional quarantine rename."""

    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _assert_linked_descriptor(
    parent_descriptor: int,
    name: str,
    status: os.stat_result,
    *,
    display: Path,
    operation: str,
) -> None:
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise ArtifactError(f"{display.name} disappeared while it was {operation}") from error
    if stat.S_ISLNK(linked.st_mode) or _file_state(linked) != _file_state(status):
        raise ArtifactError(f"{display.name} path changed while it was {operation}")


def _assert_parent_path(path: Path, expected: os.stat_result, *, operation: str) -> None:
    """Reject ancestor replacement after work through a pinned parent descriptor."""

    reopened = -1
    try:
        reopened, _ = _open_parent_descriptor(path, create=False)
        observed = os.fstat(reopened)
    except OSError as error:
        raise ArtifactError(f"{path.parent} changed while {path.name} was {operation}") from error
    finally:
        if reopened >= 0:
            os.close(reopened)
    if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
        raise ArtifactError(f"{path.parent} changed while {path.name} was {operation}")


def _assert_published_descriptor(
    parent_descriptor: int,
    name: str,
    path: Path,
    source_status: os.stat_result,
    expected: bytes,
) -> None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ArtifactError(f"cannot reopen published artifact {path}: {error}") from error
    try:
        initial = os.fstat(descriptor)
        source_identity = (
            source_status.st_dev,
            source_status.st_ino,
            source_status.st_mode,
            source_status.st_size,
        )
        published_identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_mode,
            initial.st_size,
        )
        if published_identity != source_identity:
            raise ArtifactError(f"{path.name} path changed while it was published")
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(1024 * 1024, len(expected) + 1 - observed)):
            observed += len(chunk)
            if observed > len(expected):
                raise ArtifactError(f"{path.name} changed while it was published")
            chunks.append(chunk)
        if b"".join(chunks) != expected:
            raise ArtifactError(f"{path.name} bytes changed while it was published")
        final = os.fstat(descriptor)
        if _file_state(final) != _file_state(initial):
            raise ArtifactError(f"{path.name} changed while it was published")
        _assert_linked_descriptor(
            parent_descriptor,
            name,
            final,
            display=path,
            operation="published",
        )
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _open_parent_descriptor(path: Path, *, create: bool) -> tuple[int, str]:
    absolute = path.absolute()
    parts = absolute.parts
    if len(parts) < 2:
        raise ArtifactError(f"path has no parent: {path}")
    descriptor = -1
    try:
        descriptor = os.open(
            parts[0],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        for part in parts[1:-1]:
            if part in {"", ".", ".."}:
                raise ArtifactError("path contains an unsafe directory component")
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, 0o700, dir_fd=descriptor)
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        name = parts[-1]
        if name in {"", ".", ".."}:
            raise ArtifactError("path contains an unsafe final component")
        return descriptor, name
    except ArtifactError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ArtifactError(f"cannot open parent directory for {path}: {error}") from error
