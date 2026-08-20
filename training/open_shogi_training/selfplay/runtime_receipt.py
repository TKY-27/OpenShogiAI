"""Durable Python interpreter and dependency-tree runtime authority."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import sysconfig
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from open_shogi_training.labeling.artifacts import write_json_atomic
from open_shogi_training.labeling.execution import (
    ExecutableSnapshot,
    ExecutableSnapshotError,
)

from .common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    canonical_sha256,
    contained_path,
    ensure_contained_directory,
    load_json_artifact,
    require_exact_keys,
    require_int,
    require_mapping,
    require_sha256,
    require_string,
)

PYTHON_RUNTIME_RECEIPT_SCHEMA: Final = "phase6_python_runtime_receipt/v1"
PYTHON_RUNTIME_RECEIPT_ROOT: Final = "local/runtime-receipts/python"
_MAX_TREE_ENTRIES: Final = 50_000
_MAX_TREE_BYTES: Final = 8 * 1024**3
_MAX_FILE_BYTES: Final = 2 * 1024**3
_TREE_KEYS: Final = frozenset({"root", "treeSha256", "entries", "files", "symlinks", "bytes"})
_FILE_KEYS: Final = frozenset({"path", "sha256", "size"})
_SOURCE_KEYS: Final = frozenset({"treeSha256", "files", "bytes"})
_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "gitCommit",
        "interpreter",
        "pythonLibrary",
        "stdlib",
        "sitePackages",
        "uvLock",
        "sourceSnapshot",
        "receiptSha256",
    }
)
_INTERPRETER_KEYS: Final = frozenset({"path", "sha256", "size", "version", "basePrefix"})


@dataclass(frozen=True, slots=True)
class _OpenNode:
    path: Path
    descriptor: int
    metadata: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _LinkNode:
    path: Path
    metadata: tuple[int, int, int, int, int, int]
    target: str


class RuntimeDependencyGuard:
    """Retain every runtime file descriptor and fail on any metadata/path drift."""

    def __init__(
        self,
        *,
        identities: tuple[dict[str, object], ...],
        open_nodes: list[_OpenNode],
        links: list[_LinkNode],
    ) -> None:
        self.identities = identities
        self._open_nodes = open_nodes
        self._links = links

    @classmethod
    def create(cls, roots: tuple[Path, ...]) -> RuntimeDependencyGuard:
        identities: list[dict[str, object]] = []
        nodes: list[_OpenNode] = []
        links: list[_LinkNode] = []
        try:
            for root in roots:
                identity, root_nodes, root_links = _open_tree(root)
                identities.append(identity)
                nodes.extend(root_nodes)
                links.extend(root_links)
        except BaseException:
            for node in nodes:
                with suppress(OSError):
                    os.close(node.descriptor)
            raise
        result = cls(identities=tuple(identities), open_nodes=nodes, links=links)
        try:
            result.assert_unchanged()
        except BaseException:
            result.close()
            raise
        return result

    def assert_unchanged(self) -> None:
        for node in self._open_nodes:
            descriptor_status = os.fstat(node.descriptor)
            try:
                linked_status = node.path.lstat()
            except OSError as error:
                raise ContractError(f"Python runtime path disappeared: {node.path}") from error
            if (
                _metadata(descriptor_status) != node.metadata
                or _metadata(linked_status) != node.metadata
                or stat.S_ISLNK(linked_status.st_mode)
            ):
                raise ContractError(f"Python runtime path changed: {node.path}")
        for link in self._links:
            try:
                metadata = link.path.lstat()
                target = os.readlink(link.path)
            except OSError as error:
                raise ContractError(f"Python runtime symlink disappeared: {link.path}") from error
            if (
                not stat.S_ISLNK(metadata.st_mode)
                or _metadata(metadata) != link.metadata
                or target != link.target
            ):
                raise ContractError(f"Python runtime symlink changed: {link.path}")

    def close(self) -> None:
        nodes = self._open_nodes
        self._open_nodes = []
        for node in nodes:
            with suppress(OSError):
                os.close(node.descriptor)


class RuntimeFileGuard:
    """Retain one runtime file descriptor and its exact hashed identity."""

    def __init__(self, *, identity: dict[str, object], node: _OpenNode) -> None:
        self.identity = identity
        self._node: _OpenNode | None = node

    @classmethod
    def create(cls, path: Path) -> RuntimeFileGuard:
        absolute = path.resolve(strict=True)
        node = _open_node(absolute, directory=False)
        try:
            sha256, size = _hash_descriptor(node.descriptor)
            identity = {"path": str(absolute), "sha256": sha256, "size": size}
            _validate_absolute_file(identity, "Python runtime file")
        except BaseException:
            os.close(node.descriptor)
            raise
        result = cls(identity=identity, node=node)
        try:
            result.assert_unchanged()
        except BaseException:
            result.close()
            raise
        return result

    def assert_unchanged(self) -> None:
        node = self._node
        if node is None:
            raise ContractError("Python runtime file guard is closed")
        descriptor_status = os.fstat(node.descriptor)
        try:
            linked_status = node.path.lstat()
        except OSError as error:
            raise ContractError(f"Python runtime file disappeared: {node.path}") from error
        if (
            _metadata(descriptor_status) != node.metadata
            or _metadata(linked_status) != node.metadata
            or stat.S_ISLNK(linked_status.st_mode)
        ):
            raise ContractError(f"Python runtime file changed: {node.path}")

    def close(self) -> None:
        node = self._node
        self._node = None
        if node is not None:
            with suppress(OSError):
                os.close(node.descriptor)


def _close_runtime_authority_parts(
    interpreter: ExecutableSnapshot | None,
    tree_guard: RuntimeDependencyGuard | None,
    library_guard: RuntimeFileGuard | None,
) -> None:
    errors: list[BaseException] = []
    for resource in (interpreter, tree_guard, library_guard):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        first = errors[0]
        for additional in errors[1:]:
            first.add_note(f"additional Python runtime cleanup failure: {additional!r}")
        raise first


class PythonRuntimeAuthority:
    """A sealed interpreter plus same-FD guards for its large dependency trees.

    macOS has no supported exact-FD exec. The executable is therefore copied to a
    private content-addressed immutable snapshot, while stdlib/site-packages and
    the Homebrew framework library remain same-FD/hash bound with pre/post-spawn
    metadata guards. Active hostile same-UID mutation is explicitly out of scope.
    """

    def __init__(
        self,
        *,
        receipt: ArtifactRef,
        interpreter: ExecutableSnapshot,
        tree_guard: RuntimeDependencyGuard,
        library_guard: RuntimeFileGuard,
        base_prefix: str,
        site_paths: tuple[str, ...],
    ) -> None:
        self.receipt = receipt
        self.interpreter = interpreter
        self._tree_guard = tree_guard
        self._library_guard = library_guard
        self.base_prefix = base_prefix
        self.site_paths = site_paths

    @classmethod
    def create(
        cls,
        repository_root: Path,
        *,
        git_commit: str,
        source_snapshot: dict[str, object],
    ) -> PythonRuntimeAuthority:
        root = repository_root.resolve(strict=True)
        interpreter_path = Path(sys.executable).resolve(strict=True)
        base_prefix = Path(sys.base_prefix).resolve(strict=True)
        stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
        site_paths = tuple(
            sorted(
                {
                    str(Path(value).resolve(strict=True))
                    for key, value in sysconfig.get_paths().items()
                    if key in {"purelib", "platlib"} and isinstance(value, str) and value
                }
            )
        )
        if len(site_paths) != 1:
            raise ContractError("Python purelib and platlib must resolve to one pinned tree")
        site_packages = Path(site_paths[0])
        python_library = base_prefix / "Python"
        interpreter_identity = _hash_file(interpreter_path, executable=True)
        library_guard = RuntimeFileGuard.create(python_library)
        tree_guard: RuntimeDependencyGuard | None = None
        snapshot: ExecutableSnapshot | None = None
        try:
            tree_guard = RuntimeDependencyGuard.create((stdlib, site_packages))
            library_identity = library_guard.identity
            uv_lock = artifact_ref(root, "uv.lock", maximum_bytes=16 * 1024 * 1024)
            _validate_source_snapshot(source_snapshot)
            payload: dict[str, object] = {
                "schema": PYTHON_RUNTIME_RECEIPT_SCHEMA,
                "gitCommit": git_commit,
                "interpreter": {
                    **interpreter_identity,
                    "version": sys.version,
                    "basePrefix": str(base_prefix),
                },
                "pythonLibrary": {
                    "path": library_identity["path"],
                    "sha256": library_identity["sha256"],
                    "size": library_identity["size"],
                },
                "stdlib": tree_guard.identities[0],
                "sitePackages": tree_guard.identities[1],
                "uvLock": uv_lock.as_dict(),
                "sourceSnapshot": source_snapshot,
            }
            payload["receiptSha256"] = canonical_sha256(payload)
            relative = f"{PYTHON_RUNTIME_RECEIPT_ROOT}/{payload['receiptSha256']}.json"
            destination = contained_path(root, relative)
            try:
                write_json_atomic(destination, payload, replace=False)
            except FileExistsError:
                existing = artifact_ref(root, relative)
                if load_json_artifact(root, existing) != payload:
                    raise ContractError(
                        "immutable Python runtime receipt contains different bytes"
                    ) from None
            reference = artifact_ref(root, relative)
            validate_python_runtime_receipt(
                root,
                reference,
                expected_git_commit=git_commit,
            )
            snapshot = ExecutableSnapshot.create(
                interpreter_path,
                temporary_directory=ensure_contained_directory(root, "local/runtime-snapshots"),
                max_bytes=128 * 1024 * 1024,
                expected_sha256=str(interpreter_identity["sha256"]),
            )
        except BaseException as error:
            try:
                _close_runtime_authority_parts(snapshot, tree_guard, library_guard)
            except BaseException as cleanup_error:
                error.add_note(f"Python runtime cleanup also failed: {cleanup_error!r}")
            raise
        assert tree_guard is not None and snapshot is not None
        result = cls(
            receipt=reference,
            interpreter=snapshot,
            tree_guard=tree_guard,
            library_guard=library_guard,
            base_prefix=str(base_prefix),
            site_paths=site_paths,
        )
        try:
            result.assert_unchanged()
        except BaseException as error:
            try:
                result.close()
            except BaseException as cleanup_error:
                error.add_note(f"Python runtime cleanup also failed: {cleanup_error!r}")
            raise
        return result

    def assert_unchanged(self) -> None:
        try:
            self.interpreter.assert_snapshot_unchanged()
            self.interpreter.assert_source_unchanged()
        except ExecutableSnapshotError as error:
            raise ContractError(f"Python interpreter identity changed: {error}") from error
        self._library_guard.assert_unchanged()
        self._tree_guard.assert_unchanged()

    def close(self) -> None:
        _close_runtime_authority_parts(
            self.interpreter,
            self._tree_guard,
            self._library_guard,
        )


def validate_python_runtime_receipt(
    repository_root: Path,
    reference: ArtifactRef,
    *,
    expected_git_commit: str,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    receipt = dict(require_mapping(load_json_artifact(root, reference), "Python runtime receipt"))
    require_exact_keys(receipt, _RECEIPT_KEYS, "Python runtime receipt")
    if receipt.get("schema") != PYTHON_RUNTIME_RECEIPT_SCHEMA:
        raise ContractError("unsupported Python runtime receipt schema")
    commit = require_string(receipt, "gitCommit", "Python runtime receipt", maximum_length=64)
    if (
        len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or commit != expected_git_commit
    ):
        raise ContractError("Python runtime receipt belongs to another Git commit")
    interpreter = dict(require_mapping(receipt.get("interpreter"), "runtime interpreter"))
    require_exact_keys(interpreter, _INTERPRETER_KEYS, "runtime interpreter")
    _validate_absolute_file(interpreter, "runtime interpreter")
    version = require_string(interpreter, "version", "runtime interpreter", maximum_length=256)
    if any(character in version for character in "\x00\r\n"):
        raise ContractError("runtime interpreter.version is malformed")
    base_prefix = require_string(
        interpreter, "basePrefix", "runtime interpreter", maximum_length=4_096
    )
    _validate_absolute_path(base_prefix, "runtime interpreter.basePrefix")
    library = dict(require_mapping(receipt.get("pythonLibrary"), "Python library"))
    require_exact_keys(library, _FILE_KEYS, "Python library")
    _validate_absolute_file(library, "Python library")
    for field in ("stdlib", "sitePackages"):
        _validate_tree_identity(receipt.get(field), f"Python runtime {field}")
    uv_lock = ArtifactRef.from_dict(receipt.get("uvLock"), "Python runtime uvLock")
    if uv_lock.path != "uv.lock":
        raise ContractError("Python runtime receipt must bind repository uv.lock")
    verify = artifact_ref(root, "uv.lock", maximum_bytes=16 * 1024 * 1024)
    if verify != uv_lock:
        raise ContractError("Python runtime uv.lock identity changed")
    _validate_source_snapshot(receipt.get("sourceSnapshot"))
    expected_hash = require_sha256(receipt, "receiptSha256", "Python runtime receipt")
    unsigned = dict(receipt)
    del unsigned["receiptSha256"]
    if canonical_sha256(unsigned) != expected_hash:
        raise ContractError("Python runtime receipt self-hash mismatch")
    if reference.path != f"{PYTHON_RUNTIME_RECEIPT_ROOT}/{expected_hash}.json":
        raise ContractError("Python runtime receipt path is not content-addressed")
    return receipt


def _open_tree(
    root: Path,
) -> tuple[dict[str, object], list[_OpenNode], list[_LinkNode]]:
    root = root.absolute()
    if any(character in str(root) for character in "\x00\r\n"):
        raise ContractError("Python runtime root path is malformed")
    nodes: list[_OpenNode] = []
    links: list[_LinkNode] = []
    digest = hashlib.sha256()
    entries = 0
    files = 0
    total = 0
    try:
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(directory)
            relative_directory = current.relative_to(root).as_posix()
            nodes.append(_open_node(current, directory=True))
            _digest_entry(digest, b"D", relative_directory, b"")
            entries += 1
            retained_directories: list[str] = []
            for name in sorted(directory_names):
                path = current / name
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    target = os.readlink(path)
                    _validate_link_target(target)
                    links.append(_LinkNode(path, _metadata(metadata), target))
                    relative = path.relative_to(root).as_posix()
                    _digest_entry(digest, b"L", relative, target.encode("utf-8"))
                    entries += 1
                else:
                    retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in sorted(file_names):
                path = current / name
                metadata = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISLNK(metadata.st_mode):
                    target = os.readlink(path)
                    _validate_link_target(target)
                    links.append(_LinkNode(path, _metadata(metadata), target))
                    _digest_entry(digest, b"L", relative, target.encode("utf-8"))
                    entries += 1
                    continue
                node = _open_node(path, directory=False)
                sha256, size = _hash_descriptor(node.descriptor)
                nodes.append(node)
                payload = size.to_bytes(8, "big") + bytes.fromhex(sha256)
                _digest_entry(digest, b"F", relative, payload)
                entries += 1
                files += 1
                total += size
                if entries > _MAX_TREE_ENTRIES or total > _MAX_TREE_BYTES:
                    raise ContractError("Python runtime tree exceeds its bounded size")
    except BaseException:
        for node in nodes:
            with suppress(OSError):
                os.close(node.descriptor)
        raise
    identity: dict[str, object] = {
        "root": str(root),
        "treeSha256": digest.hexdigest(),
        "entries": entries,
        "files": files,
        "symlinks": len(links),
        "bytes": total,
    }
    _validate_tree_identity(identity, "Python runtime tree")
    return identity, nodes, links


def _open_node(path: Path, *, directory: bool) -> _OpenNode:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if directory and not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ContractError(f"Python runtime directory changed while opened: {path}")
    if not directory and (
        not stat.S_ISREG(metadata.st_mode) or not 0 <= metadata.st_size <= _MAX_FILE_BYTES
    ):
        os.close(descriptor)
        raise ContractError(f"Python runtime file violates its type/size bound: {path}")
    linked = path.lstat()
    if _metadata(linked) != _metadata(metadata) or stat.S_ISLNK(linked.st_mode):
        os.close(descriptor)
        raise ContractError(f"Python runtime path changed while opened: {path}")
    return _OpenNode(path, descriptor, _metadata(metadata))


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def _hash_file(path: Path, *, executable: bool) -> dict[str, object]:
    node = _open_node(path, directory=False)
    try:
        metadata = os.fstat(node.descriptor)
        if executable and not metadata.st_mode & 0o111:
            raise ContractError("Python interpreter is not executable")
        sha256, size = _hash_descriptor(node.descriptor)
    finally:
        os.close(node.descriptor)
    return {"path": str(path), "sha256": sha256, "size": size}


def _digest_entry(digest: Any, kind: bytes, relative: str, payload: bytes) -> None:
    encoded = relative.encode("utf-8")
    digest.update(kind)
    digest.update(len(encoded).to_bytes(4, "big"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _validate_link_target(target: str) -> None:
    if (
        not target
        or len(target.encode("utf-8")) > 4_096
        or any(character in target for character in "\x00\r\n")
    ):
        raise ContractError("Python runtime symlink target is malformed")


def _validate_tree_identity(value: object, context: str) -> dict[str, Any]:
    tree = dict(require_mapping(value, context))
    require_exact_keys(tree, _TREE_KEYS, context)
    _validate_absolute_path(require_string(tree, "root", context, maximum_length=4_096), context)
    require_sha256(tree, "treeSha256", context)
    entries = require_int(tree, "entries", context, minimum=1, maximum=_MAX_TREE_ENTRIES)
    files = require_int(tree, "files", context, minimum=1, maximum=_MAX_TREE_ENTRIES)
    symlinks = require_int(tree, "symlinks", context, minimum=0, maximum=_MAX_TREE_ENTRIES)
    if files + symlinks > entries:
        raise ContractError(f"{context} entry counts disagree")
    require_int(tree, "bytes", context, minimum=1, maximum=_MAX_TREE_BYTES)
    return tree


def _validate_source_snapshot(value: object) -> dict[str, Any]:
    source = dict(require_mapping(value, "Python source snapshot"))
    require_exact_keys(source, _SOURCE_KEYS, "Python source snapshot")
    require_sha256(source, "treeSha256", "Python source snapshot")
    require_int(source, "files", "Python source snapshot", minimum=1, maximum=2_000)
    require_int(
        source,
        "bytes",
        "Python source snapshot",
        minimum=1,
        maximum=128 * 1024 * 1024,
    )
    return source


def _validate_absolute_file(value: dict[str, Any], context: str) -> None:
    path = require_string(value, "path", context, maximum_length=4_096)
    _validate_absolute_path(path, f"{context}.path")
    require_sha256(value, "sha256", context)
    require_int(value, "size", context, minimum=1, maximum=_MAX_FILE_BYTES)


def _validate_absolute_path(path: str, context: str) -> None:
    if not Path(path).is_absolute() or any(character in path for character in "\x00\r\n"):
        raise ContractError(f"{context} must be an absolute path")


def _metadata(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )
