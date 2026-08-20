"""Verified private executable snapshots for local subprocess boundaries."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .artifacts import (
    ArtifactError,
    assert_retirement_capacity,
    retire_bound_directory,
    stable_parent_descriptor,
)


class ExecutableSnapshotError(RuntimeError):
    """Raised when executable bytes cannot be pinned before process creation."""


class RuntimeTreeSnapshotError(RuntimeError):
    """Raised when immutable runtime inputs cannot be copied into a private tree."""


def _raise_first_cleanup_error(errors: list[BaseException]) -> None:
    if not errors:
        return
    first = errors[0]
    for additional in errors[1:]:
        first.add_note(f"additional cleanup failure: {additional!r}")
    raise first


@dataclass(frozen=True, slots=True)
class ExecutableIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeFileIdentity:
    path: str
    sha256: str
    size: int
    metadata: tuple[int, int, int, int, int, int]


class RuntimeTreeSnapshot:
    """Private cwd-shaped copies of exact runtime data files.

    The mirror preserves each project-relative path, so a configured cwd-relative
    option such as ``Eval_Dir`` resolves identically while the subprocess cannot
    observe mutable source evaluation files.
    """

    def __init__(
        self,
        *,
        private_directory: Path,
        private_identity: tuple[int, int],
        cwd: Path,
        executable_storage: Path,
        files: tuple[tuple[Path, int, RuntimeFileIdentity], ...],
        directories: tuple[Path, ...],
    ) -> None:
        self.private_directory = private_directory
        self._private_identity = private_identity
        self.cwd = cwd
        self.executable_storage = executable_storage
        self._files = files
        self._directories = directories
        self._sealed = False

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        working_directory: str,
        files: tuple[tuple[str, str, int], ...],
        storage_directory: Path,
        payloads: Mapping[str, bytes] | None = None,
        executable_files: frozenset[str] = frozenset(),
    ) -> RuntimeTreeSnapshot:
        root = project_root.resolve(strict=True)
        cwd_relative = _validate_project_relative(working_directory)
        _ensure_real_directory(storage_directory)
        storage_descriptor = os.open(
            storage_directory,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            assert_retirement_capacity(storage_descriptor)
        except ArtifactError as error:
            raise RuntimeTreeSnapshotError(
                "runtime snapshot retirement capacity is exhausted"
            ) from error
        finally:
            os.close(storage_descriptor)
        private_directory: Path | None = None
        private_status: os.stat_result | None = None
        opened: list[tuple[Path, int, RuntimeFileIdentity]] = []
        created_directories: set[Path] = set()
        try:
            private_directory = Path(
                tempfile.mkdtemp(prefix=".open-shogi-runtime.", dir=storage_directory)
            )
            private_status = private_directory.lstat()
            if stat.S_ISLNK(private_status.st_mode) or not stat.S_ISDIR(private_status.st_mode):
                raise RuntimeTreeSnapshotError("private runtime directory is unsafe")
            os.chmod(private_directory, 0o700)
            tree = private_directory / "tree"
            executable_storage = private_directory / "executables"
            for directory in (tree, executable_storage):
                directory.mkdir(mode=0o700)
                created_directories.add(directory)
            cwd = tree.joinpath(*cwd_relative.parts)
            cwd.mkdir(mode=0o700, parents=True, exist_ok=True)
            _record_ancestors(cwd, tree, created_directories)

            executable_set = {
                _validate_project_relative(value).as_posix() for value in executable_files
            }
            seen: set[str] = set()
            for relative_value, expected_sha256, maximum_bytes in files:
                relative = _validate_project_relative(relative_value)
                normalized = relative.as_posix()
                if normalized in seen:
                    raise RuntimeTreeSnapshotError("runtime snapshot repeats a file path")
                seen.add(normalized)
                if (
                    len(expected_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in expected_sha256)
                    or isinstance(maximum_bytes, bool)
                    or not isinstance(maximum_bytes, int)
                    or maximum_bytes < 1
                ):
                    raise RuntimeTreeSnapshotError("runtime snapshot file pin is invalid")
                target = tree.joinpath(*relative.parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                _record_ancestors(target.parent, tree, created_directories)
                if payloads is None:
                    identity, descriptor = _copy_runtime_file(
                        root.joinpath(*relative.parts),
                        target,
                        relative=normalized,
                        expected_sha256=expected_sha256,
                        maximum_bytes=maximum_bytes,
                        executable=normalized in executable_set,
                    )
                else:
                    try:
                        payload = payloads[normalized]
                    except KeyError as error:
                        raise RuntimeTreeSnapshotError(
                            f"runtime payload is missing: {normalized}"
                        ) from error
                    identity, descriptor = _write_runtime_payload(
                        payload,
                        target,
                        relative=normalized,
                        expected_sha256=expected_sha256,
                        maximum_bytes=maximum_bytes,
                        executable=normalized in executable_set,
                    )
                opened.append((target, descriptor, identity))
            if not executable_set.issubset(seen):
                raise RuntimeTreeSnapshotError(
                    "runtime executable set contains an untracked file path"
                )
            if payloads is not None and set(payloads) != seen:
                raise RuntimeTreeSnapshotError(
                    "runtime payload set differs from the tracked file set"
                )
            result = cls(
                private_directory=private_directory,
                private_identity=(private_status.st_dev, private_status.st_ino),
                cwd=cwd,
                executable_storage=executable_storage,
                files=tuple(opened),
                directories=tuple(
                    sorted(created_directories, key=lambda path: len(path.parts), reverse=True)
                ),
            )
            result.assert_unchanged()
            private_directory = None
            opened = []
            created_directories = set()
            return result
        except OSError as error:
            raise RuntimeTreeSnapshotError(f"cannot snapshot runtime tree: {error}") from error
        finally:
            for _, descriptor, _ in opened:
                with suppress(OSError):
                    os.close(descriptor)
            if private_directory is not None:
                if private_status is None:
                    raise RuntimeTreeSnapshotError(
                        "private runtime directory identity was not recorded"
                    )
                _retire_private_tree(
                    private_directory,
                    private_status.st_dev,
                    private_status.st_ino,
                    RuntimeTreeSnapshotError,
                )

    def seal(self) -> None:
        self.assert_unchanged()
        self._sealed = True
        try:
            for path, _, _ in self._files:
                _set_user_immutable(path, enabled=True)
            for directory in self._directories:
                os.chmod(directory, 0o500)
                _set_user_immutable(directory, enabled=True)
            os.chmod(self.private_directory, 0o500)
            _set_user_immutable(self.private_directory, enabled=True)
            self._files = tuple(
                (
                    path,
                    descriptor,
                    replace(identity, metadata=_status_identity(os.fstat(descriptor))),
                )
                for path, descriptor, identity in self._files
            )
            self.assert_unchanged()
        except BaseException:
            self.unseal()
            raise

    def unseal(self) -> None:
        if not self._sealed:
            return
        errors: list[BaseException] = []
        try:
            _set_user_immutable(self.private_directory, enabled=False)
            os.chmod(self.private_directory, 0o700)
        except BaseException as error:
            errors.append(error)
        for directory in reversed(self._directories):
            try:
                _set_user_immutable(directory, enabled=False)
                os.chmod(directory, 0o700)
            except BaseException as error:
                errors.append(error)
        for path, _, _ in self._files:
            try:
                _set_user_immutable(path, enabled=False)
            except BaseException as error:
                errors.append(error)
        self._sealed = False
        _raise_first_cleanup_error(errors)

    def assert_unchanged(self) -> None:
        try:
            private_status = self.private_directory.lstat()
        except OSError as error:
            raise RuntimeTreeSnapshotError("private runtime directory disappeared") from error
        if (
            stat.S_ISLNK(private_status.st_mode)
            or not stat.S_ISDIR(private_status.st_mode)
            or (private_status.st_dev, private_status.st_ino) != self._private_identity
        ):
            raise RuntimeTreeSnapshotError("private runtime directory identity changed")
        for path, descriptor, identity in self._files:
            status = os.fstat(descriptor)
            try:
                linked = path.lstat()
            except OSError as error:
                raise RuntimeTreeSnapshotError(
                    f"runtime snapshot file disappeared: {identity.path}"
                ) from error
            if (
                _status_identity(status) != identity.metadata
                or _status_identity(linked) != identity.metadata
                or stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISREG(linked.st_mode)
            ):
                raise RuntimeTreeSnapshotError(f"runtime snapshot file changed: {identity.path}")
            if self._sealed and not _is_user_immutable(linked):
                raise RuntimeTreeSnapshotError(
                    f"runtime snapshot file lost its immutable flag: {identity.path}"
                )
        if self._sealed:
            if not _is_user_immutable(private_status):
                raise RuntimeTreeSnapshotError("private runtime directory lost its immutable flag")
            for directory in self._directories:
                if not _is_user_immutable(directory.lstat()):
                    raise RuntimeTreeSnapshotError(
                        f"runtime directory lost its immutable flag: {directory}"
                    )

    def close(self) -> None:
        errors: list[BaseException] = []
        try:
            self.unseal()
        except BaseException as error:
            errors.append(error)
        files = self._files
        self._files = ()
        for _, descriptor, _ in files:
            try:
                os.close(descriptor)
            except BaseException as error:
                errors.append(error)
        try:
            _retire_private_tree(
                self.private_directory,
                *self._private_identity,
                RuntimeTreeSnapshotError,
            )
        except BaseException as error:
            errors.append(error)
        _raise_first_cleanup_error(errors)


class ExecutableSnapshot:
    """A verified private launch snapshot with retained byte identity.

    macOS has no supported descriptor-based exec primitive. The snapshot therefore
    combines a random private directory, content-addressed leaf, retained descriptors,
    read/execute-only modes, and ``UF_IMMUTABLE`` where the platform provides it.
    Same-UID processes can still signal this process or deliberately clear that flag;
    such active same-user tampering is outside the local pipeline threat model. Any
    mutation observed at a validation boundary fails closed.
    """

    def __init__(
        self,
        *,
        source_path: Path,
        descriptor: int,
        source_identity: ExecutableIdentity,
        snapshot_metadata: tuple[int, int, int, int, int, int],
        snapshot_path: Path,
        private_directory: Path,
        private_directory_device: int,
        private_directory_inode: int,
    ) -> None:
        self.source_path = source_path
        self.descriptor = descriptor
        self.identity = source_identity
        self._snapshot_metadata = snapshot_metadata
        self._snapshot_path = snapshot_path
        self._private_directory = private_directory
        self._private_directory_device = private_directory_device
        self._private_directory_inode = private_directory_inode

    @classmethod
    def create(
        cls,
        source_path: Path,
        *,
        temporary_directory: Path,
        max_bytes: int,
        expected_sha256: str | None,
    ) -> ExecutableSnapshot:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        directory = temporary_directory
        try:
            directory_status = directory.lstat()
        except OSError as error:
            raise ExecutableSnapshotError(
                f"cannot inspect executable snapshot directory: {error}"
            ) from error
        if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
            raise ExecutableSnapshotError("executable snapshot directory must be a real directory")
        source_descriptor, source_parent, source_name = _open_source(source_path)
        writer_descriptor: int | None = None
        snapshot_descriptor: int | None = None
        temporary_path: Path | None = None
        private_directory: Path | None = None
        private_status: os.stat_result | None = None
        try:
            initial = os.fstat(source_descriptor)
            _validate_executable_status(initial, source_path, max_bytes=max_bytes)
            _assert_path_identity(
                source_parent,
                source_name,
                initial,
                "source executable path changed",
            )
            private_directory = Path(
                tempfile.mkdtemp(
                    prefix=".open-shogi-exec.",
                    dir=directory,
                )
            )
            private_status = private_directory.lstat()
            if not stat.S_ISDIR(private_status.st_mode) or stat.S_ISLNK(private_status.st_mode):
                raise ExecutableSnapshotError("private executable directory is unsafe")
            os.chmod(private_directory, 0o700)
            temporary_path = private_directory / "executable.pending"
            writer_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                writer_flags |= os.O_NOFOLLOW
            writer_descriptor = os.open(
                temporary_path,
                writer_flags,
                0o600,
            )
            digest = hashlib.sha256()
            observed = 0
            while chunk := os.read(source_descriptor, 1024 * 1024):
                observed += len(chunk)
                if observed > max_bytes:
                    raise ExecutableSnapshotError(
                        f"executable exceeds the {max_bytes}-byte safety bound"
                    )
                digest.update(chunk)
                _write_all(writer_descriptor, chunk)
            final = os.fstat(source_descriptor)
            if _status_identity(initial) != _status_identity(final) or observed != initial.st_size:
                raise ExecutableSnapshotError("executable changed while its bytes were copied")
            _assert_path_identity(
                source_parent,
                source_name,
                final,
                "source executable path changed",
            )
            sha256 = digest.hexdigest()
            if expected_sha256 is not None and sha256 != expected_sha256:
                raise ExecutableSnapshotError("executable SHA-256 differs from its runtime pin")

            os.fsync(writer_descriptor)
            os.fchmod(writer_descriptor, 0o500)
            os.fsync(writer_descriptor)
            written = os.fstat(writer_descriptor)
            if written.st_size != observed or not stat.S_ISREG(written.st_mode):
                raise ExecutableSnapshotError("executable snapshot bytes were not published fully")
            content_path = private_directory / f"executable.{sha256}"
            os.replace(temporary_path, content_path)
            temporary_path = content_path
            _set_user_immutable(temporary_path, enabled=True)
            snapshot_descriptor = _open_snapshot(temporary_path)
            reopened = os.fstat(snapshot_descriptor)
            if (
                reopened.st_dev != written.st_dev
                or reopened.st_ino != written.st_ino
                or reopened.st_size != written.st_size
            ):
                raise ExecutableSnapshotError("executable snapshot path changed before unlink")
            _set_user_immutable(private_directory, enabled=True)
            os.close(writer_descriptor)
            writer_descriptor = None
            result = cls(
                source_path=source_path,
                descriptor=snapshot_descriptor,
                source_identity=_identity(initial, sha256),
                snapshot_metadata=_status_identity(reopened),
                snapshot_path=temporary_path,
                private_directory=private_directory,
                private_directory_device=private_status.st_dev,
                private_directory_inode=private_status.st_ino,
            )
            result.assert_snapshot_unchanged()
            result.assert_source_unchanged()
            snapshot_descriptor = None
            temporary_path = None
            private_directory = None
            return result
        except OSError as error:
            raise ExecutableSnapshotError(f"cannot snapshot executable: {error}") from error
        finally:
            os.close(source_descriptor)
            os.close(source_parent)
            if writer_descriptor is not None:
                os.close(writer_descriptor)
            if snapshot_descriptor is not None:
                os.close(snapshot_descriptor)
            if temporary_path is not None:
                with suppress(OSError):
                    _set_user_immutable(temporary_path, enabled=False)
            if private_directory is not None:
                with suppress(OSError):
                    _set_user_immutable(private_directory, enabled=False)
                if private_status is None:
                    raise ExecutableSnapshotError(
                        "private executable directory identity was not recorded"
                    )
                _retire_private_tree(
                    private_directory,
                    private_status.st_dev,
                    private_status.st_ino,
                    ExecutableSnapshotError,
                )

    @property
    def executable_path(self) -> str:
        if self.descriptor < 0:
            raise ExecutableSnapshotError("executable snapshot is closed")
        return str(self._snapshot_path)

    def pass_fds(self) -> tuple[int, ...]:
        if self.descriptor < 0:
            raise ExecutableSnapshotError("executable snapshot is closed")
        return ()

    def assert_snapshot_unchanged(self) -> None:
        self.assert_snapshot_metadata_unchanged()
        status = os.fstat(self.descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < status.st_size:
            chunk = os.pread(self.descriptor, min(1024 * 1024, status.st_size - offset), offset)
            if not chunk:
                raise ExecutableSnapshotError("private executable snapshot ended early")
            digest.update(chunk)
            offset += len(chunk)
        if digest.hexdigest() != self.identity.sha256:
            raise ExecutableSnapshotError("private executable snapshot bytes changed")

    def assert_snapshot_metadata_unchanged(self) -> None:
        if self.descriptor < 0:
            raise ExecutableSnapshotError("executable snapshot is closed")
        status = os.fstat(self.descriptor)
        try:
            linked = self._snapshot_path.lstat()
        except OSError as error:
            raise ExecutableSnapshotError("executable snapshot path disappeared") from error
        if (
            not stat.S_ISREG(status.st_mode)
            or _status_identity(status) != self._snapshot_metadata
            or status.st_size != self.identity.size
            or stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or _status_identity(linked) != self._snapshot_metadata
            or linked.st_size != self.identity.size
        ):
            raise ExecutableSnapshotError("private executable snapshot metadata changed")
        if not _is_user_immutable(linked):
            raise ExecutableSnapshotError("private executable snapshot lost its immutable flag")
        if not _is_user_immutable(self._private_directory.lstat()):
            raise ExecutableSnapshotError("private executable directory lost its immutable flag")

    def assert_source_unchanged(self) -> None:
        descriptor, parent_descriptor, name = _open_source(self.source_path)
        try:
            initial = os.fstat(descriptor)
            _validate_executable_status(
                initial,
                self.source_path,
                max_bytes=self.identity.size,
            )
            digest = hashlib.sha256()
            observed = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                observed += len(chunk)
                if observed > self.identity.size:
                    raise ExecutableSnapshotError("source executable grew after snapshotting")
                digest.update(chunk)
            final = os.fstat(descriptor)
            _assert_path_identity(
                parent_descriptor,
                name,
                final,
                "source executable path changed after snapshotting",
            )
        except OSError as error:
            raise ExecutableSnapshotError(f"cannot recheck source executable: {error}") from error
        finally:
            os.close(descriptor)
            os.close(parent_descriptor)
        if (
            _identity(initial, digest.hexdigest()) != self.identity
            or _status_identity(initial) != _status_identity(final)
            or observed != self.identity.size
        ):
            raise ExecutableSnapshotError("source executable changed after snapshotting")

    def assert_source_metadata_unchanged(self) -> None:
        descriptor, parent_descriptor, name = _open_source(self.source_path)
        try:
            status = os.fstat(descriptor)
            _assert_path_identity(
                parent_descriptor,
                name,
                status,
                "source executable path changed after snapshotting",
            )
        finally:
            os.close(descriptor)
            os.close(parent_descriptor)
        if _status_identity(status) != (
            self.identity.device,
            self.identity.inode,
            self.identity.mode,
            self.identity.size,
            self.identity.modified_ns,
            self.identity.changed_ns,
        ):
            raise ExecutableSnapshotError("source executable metadata changed after snapshotting")

    def close(self) -> None:
        errors: list[BaseException] = []
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as error:
                errors.append(error)
        try:
            _set_user_immutable(self._private_directory, enabled=False)
        except BaseException as error:
            errors.append(error)
        try:
            _set_user_immutable(self._snapshot_path, enabled=False)
        except BaseException as error:
            errors.append(error)
        try:
            _retire_private_tree(
                self._private_directory,
                self._private_directory_device,
                self._private_directory_inode,
                ExecutableSnapshotError,
            )
        except BaseException as error:
            errors.append(error)
        _raise_first_cleanup_error(errors)


def _open_source(path: Path) -> tuple[int, int, str]:
    try:
        parent_descriptor, name = _open_path_parent(path)
    except (OSError, RuntimeTreeSnapshotError) as error:
        raise ExecutableSnapshotError(f"cannot open executable {path}: {error}") from error
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ExecutableSnapshotError(f"cannot open executable {path}: {error}") from error
    return descriptor, parent_descriptor, name


def _open_snapshot(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _assert_path_identity(
    parent_descriptor: int,
    name: str,
    status: os.stat_result,
    message: str,
) -> None:
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise ExecutableSnapshotError(message) from error
    if stat.S_ISLNK(linked.st_mode) or _status_identity(linked) != _status_identity(status):
        raise ExecutableSnapshotError(message)


def _validate_executable_status(status: os.stat_result, path: Path, *, max_bytes: int) -> None:
    if not stat.S_ISREG(status.st_mode):
        raise ExecutableSnapshotError(f"executable must be a regular file: {path}")
    if not status.st_mode & 0o111:
        raise ExecutableSnapshotError(f"executable has no execute bit: {path}")
    if not 1 <= status.st_size <= max_bytes:
        raise ExecutableSnapshotError(
            f"executable size is outside the 1..{max_bytes} byte bound: {path}"
        )


def _status_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _identity(status: os.stat_result, sha256: str) -> ExecutableIdentity:
    return ExecutableIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        size=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
        sha256=sha256,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise ExecutableSnapshotError("could not write complete executable snapshot")
        written += count


def _validate_project_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RuntimeTreeSnapshotError("runtime path must be a non-empty portable string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeTreeSnapshotError("runtime path must be normalized and project-relative")
    return path


def _ensure_real_directory(path: Path) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    descriptor = os.open(
        parts[0],
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for part in parts[1:]:
            if part in {"", ".", ".."}:
                raise RuntimeTreeSnapshotError("runtime storage has an unsafe component")
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
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)


def _record_ancestors(path: Path, root: Path, result: set[Path]) -> None:
    current = path
    while current != root:
        if not current.is_relative_to(root):
            raise RuntimeTreeSnapshotError("runtime mirror path escaped its private root")
        result.add(current)
        current = current.parent
    result.add(root)


def _write_runtime_payload(
    payload: bytes,
    target: Path,
    *,
    relative: str,
    expected_sha256: str,
    maximum_bytes: int,
    executable: bool,
) -> tuple[RuntimeFileIdentity, int]:
    if len(payload) > maximum_bytes or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeTreeSnapshotError(f"runtime Git payload identity drifted: {relative}")
    descriptor = -1
    try:
        descriptor = os.open(
            target,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        _write_all_runtime(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o500 if executable else 0o400)
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        linked = target.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_size != len(payload)
            or _status_identity(linked) != _status_identity(status)
        ):
            raise RuntimeTreeSnapshotError(f"runtime Git payload publication failed: {relative}")
        identity = RuntimeFileIdentity(
            path=relative,
            sha256=expected_sha256,
            size=len(payload),
            metadata=_status_identity(status),
        )
        result = descriptor
        descriptor = -1
        return identity, result
    except OSError as error:
        raise RuntimeTreeSnapshotError(
            f"cannot write runtime Git payload {relative}: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _copy_runtime_file(
    source: Path,
    target: Path,
    *,
    relative: str,
    expected_sha256: str,
    maximum_bytes: int,
    executable: bool,
) -> tuple[RuntimeFileIdentity, int]:
    source_descriptor, source_parent, source_name = _open_data_source(source)
    writer_descriptor = -1
    target_descriptor = -1
    pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
    try:
        initial = os.fstat(source_descriptor)
        if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= maximum_bytes:
            raise RuntimeTreeSnapshotError(
                f"runtime source is not a bounded regular file: {relative}"
            )
        if executable and not initial.st_mode & 0o111:
            raise RuntimeTreeSnapshotError(
                f"runtime executable source is not executable: {relative}"
            )
        writer_descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        observed = 0
        while chunk := os.pread(source_descriptor, 1024 * 1024, observed):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise RuntimeTreeSnapshotError(f"runtime source exceeds its byte bound: {relative}")
            digest.update(chunk)
            _write_all_runtime(writer_descriptor, chunk)
        final = os.fstat(source_descriptor)
        if _status_identity(initial) != _status_identity(final) or observed != initial.st_size:
            raise RuntimeTreeSnapshotError(f"runtime source changed while copied: {relative}")
        _assert_data_source_path(source_parent, source_name, final, relative)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeTreeSnapshotError(f"runtime source SHA-256 drifted: {relative}")
        os.fsync(writer_descriptor)
        os.fchmod(writer_descriptor, 0o500 if executable else 0o400)
        os.fsync(writer_descriptor)
        written = os.fstat(writer_descriptor)
        if not stat.S_ISREG(written.st_mode) or written.st_size != observed:
            raise RuntimeTreeSnapshotError(f"runtime snapshot copy is incomplete: {relative}")
        os.close(writer_descriptor)
        writer_descriptor = -1
        os.replace(pending, target)
        target_descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        target_status = os.fstat(target_descriptor)
        target_digest = hashlib.sha256()
        offset = 0
        while offset < target_status.st_size:
            chunk = os.pread(
                target_descriptor,
                min(1024 * 1024, target_status.st_size - offset),
                offset,
            )
            if not chunk:
                raise RuntimeTreeSnapshotError(f"runtime snapshot ended early: {relative}")
            target_digest.update(chunk)
            offset += len(chunk)
        if (
            offset != observed
            or target_digest.hexdigest() != expected_sha256
            or _status_identity(target.lstat()) != _status_identity(target_status)
        ):
            raise RuntimeTreeSnapshotError(f"runtime snapshot verification failed: {relative}")
        identity = RuntimeFileIdentity(
            path=relative,
            sha256=expected_sha256,
            size=observed,
            metadata=_status_identity(target_status),
        )
        result_descriptor = target_descriptor
        target_descriptor = -1
        return identity, result_descriptor
    except OSError as error:
        raise RuntimeTreeSnapshotError(f"cannot copy runtime file {relative}: {error}") from error
    finally:
        os.close(source_descriptor)
        os.close(source_parent)
        if writer_descriptor >= 0:
            os.close(writer_descriptor)
        if target_descriptor >= 0:
            os.close(target_descriptor)


def _retire_private_tree(
    path: Path,
    expected_device: int,
    expected_inode: int,
    error_type: type[RuntimeError],
) -> None:
    """Quarantine one exact private tree; never recursively delete raced-in bytes."""

    try:
        with stable_parent_descriptor(path, create=False) as (parent, name):
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent,
                )
            except FileNotFoundError:
                return
            try:
                observed = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (observed.st_dev, observed.st_ino) != (expected_device, expected_inode):
                raise error_type("private snapshot tree changed before retirement")
            retire_bound_directory(
                parent,
                name,
                observed,
                display=path,
                dispose=True,
            )
    except ArtifactError as error:
        raise error_type("private snapshot tree changed before retirement") from error


def _open_data_source(path: Path) -> tuple[int, int, str]:
    parent_descriptor, name = _open_path_parent(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise RuntimeTreeSnapshotError(f"cannot open runtime source {path}: {error}") from error
    return descriptor, parent_descriptor, name


def _assert_data_source_path(
    parent_descriptor: int,
    name: str,
    status: os.stat_result,
    relative: str,
) -> None:
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise RuntimeTreeSnapshotError(f"runtime source disappeared: {relative}") from error
    if stat.S_ISLNK(linked.st_mode) or _status_identity(linked) != _status_identity(status):
        raise RuntimeTreeSnapshotError(f"runtime source path changed: {relative}")


def _open_path_parent(path: Path) -> tuple[int, str]:
    absolute = path.absolute()
    parts = absolute.parts
    if len(parts) < 2:
        raise RuntimeTreeSnapshotError(f"path has no parent: {path}")
    descriptor = os.open(
        parts[0],
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for part in parts[1:-1]:
            if part in {"", ".", ".."}:
                raise RuntimeTreeSnapshotError("path contains an unsafe directory component")
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
            raise RuntimeTreeSnapshotError("path contains an unsafe final component")
        return descriptor, name
    except BaseException:
        os.close(descriptor)
        raise


def _write_all_runtime(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise RuntimeTreeSnapshotError("could not write complete runtime snapshot")
        written += count


def _set_user_immutable(path: Path, *, enabled: bool) -> None:
    """Seal a private launch entry against replacement on supported macOS hosts."""

    immutable = getattr(stat, "UF_IMMUTABLE", None)
    if immutable is None or not hasattr(os, "chflags"):
        return
    nounlink = getattr(stat, "UF_NOUNLINK", 0)
    seal_flags = immutable | nounlink
    status = path.lstat()
    flags = int(getattr(status, "st_flags", 0))
    requested = flags | seal_flags if enabled else flags & ~seal_flags
    os.chflags(path, requested, follow_symlinks=False)
    observed = path.lstat()
    observed_flags = int(getattr(observed, "st_flags", 0))
    if ((observed_flags & seal_flags) == seal_flags) != enabled:
        raise OSError(f"could not {'set' if enabled else 'clear'} launch seal flags on {path}")


def _is_user_immutable(status: os.stat_result) -> bool:
    immutable = getattr(stat, "UF_IMMUTABLE", None)
    if immutable is None or not hasattr(os, "chflags"):
        return True
    seal_flags = immutable | getattr(stat, "UF_NOUNLINK", 0)
    return (int(getattr(status, "st_flags", 0)) & seal_flags) == seal_flags
