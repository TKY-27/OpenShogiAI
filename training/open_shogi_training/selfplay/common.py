"""Bounded serialization, hashing, and path helpers for Phase 6 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    publish_regular_at,
    retire_bound_regular,
)

MAX_JSON_BYTES: Final = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024 * 1024
MAX_IDENTIFIER_LENGTH: Final = 96
MAX_TEXT_LENGTH: Final = 4_096
MAX_JSON_DEPTH: Final = 32
MAX_JSON_NODES: Final = 250_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")


class ContractError(ValueError):
    """Raised when a Phase 6 artifact violates its closed contract."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Portable reference to a contained, immutable artifact."""

    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, raw: object, context: str) -> ArtifactRef:
        table = require_mapping(raw, context)
        require_exact_keys(table, {"path", "sha256", "size"}, context)
        return cls(
            path=require_relative_path(table, "path", context),
            sha256=require_sha256(table, "sha256", context),
            size=require_int(table, "size", context, minimum=0, maximum=MAX_ARTIFACT_BYTES),
        )


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Serialize a JSON value deterministically and reject non-finite numbers."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error
    return encoded + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def artifact_ref(
    root: Path, relative_path: str, *, maximum_bytes: int = MAX_ARTIFACT_BYTES
) -> ArtifactRef:
    normalized = validate_relative_path(relative_path)
    with stable_contained_descriptor(root, normalized) as descriptor:
        raw, digest, size = _read_descriptor(
            descriptor,
            display=normalized,
            maximum_bytes=maximum_bytes,
            retain_bytes=False,
        )
        assert raw is None
    return ArtifactRef(path=normalized, sha256=digest, size=size)


def verify_artifact_ref(root: Path, reference: ArtifactRef) -> None:
    with stable_contained_descriptor(root, reference.path) as descriptor:
        raw, observed_hash, observed_size = _read_descriptor(
            descriptor,
            display=reference.path,
            maximum_bytes=MAX_ARTIFACT_BYTES,
            retain_bytes=False,
        )
        assert raw is None
    if observed_size != reference.size:
        raise ContractError(f"artifact size mismatch: {reference.path}")
    if observed_hash != reference.sha256:
        raise ContractError(f"artifact SHA-256 mismatch: {reference.path}")
    # Verification is intentionally not a pathname-producing API. Byte consumers
    # must use the descriptor loaders; executable consumers separately create a
    # content-addressed private snapshot pinned to this ArtifactRef.


def load_json_artifact(
    root: Path,
    reference: ArtifactRef,
    *,
    maximum_bytes: int = MAX_JSON_BYTES,
    maximum_nodes: int = MAX_JSON_NODES,
) -> object:
    """Verify, hash, and parse one referenced JSON artifact from the same descriptor."""

    value, observed = load_json_and_ref(
        root,
        reference.path,
        maximum_bytes=maximum_bytes,
        maximum_nodes=maximum_nodes,
    )
    if observed != reference:
        raise ContractError(f"artifact identity mismatch: {reference.path}")
    return value


def load_bytes_artifact(
    root: Path,
    reference: ArtifactRef,
    *,
    maximum_bytes: int,
) -> bytes:
    """Return exactly the bytes verified against an artifact ref from one descriptor."""

    with stable_contained_descriptor(root, reference.path) as descriptor:
        raw, digest, size = _read_descriptor(
            descriptor,
            display=reference.path,
            maximum_bytes=maximum_bytes,
            retain_bytes=True,
        )
    if digest != reference.sha256 or size != reference.size:
        raise ContractError(f"artifact identity mismatch: {reference.path}")
    assert raw is not None
    return raw


@contextmanager
def verified_artifact_descriptor(
    root: Path,
    reference: ArtifactRef,
    *,
    maximum_bytes: int = MAX_ARTIFACT_BYTES,
):
    """Yield a rewound stable descriptor after verifying its exact artifact identity."""

    with stable_contained_descriptor(root, reference.path) as descriptor:
        initial = os.fstat(descriptor)
        _, digest, size = _read_descriptor(
            descriptor,
            display=reference.path,
            maximum_bytes=maximum_bytes,
            retain_bytes=False,
        )
        if digest != reference.sha256 or size != reference.size:
            raise ContractError(f"artifact identity mismatch: {reference.path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        yield descriptor
        if _descriptor_identity(os.fstat(descriptor)) != _descriptor_identity(initial):
            raise ContractError(f"artifact changed while consumed: {reference.path}")


def _descriptor_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(status.st_mode):
        raise ContractError("path parent is not a directory")
    return status.st_dev, status.st_ino


def _assert_parent_binding(path: Path, expected: tuple[int, int]) -> None:
    reopened, name = _open_parent_for_write(path, create=False)
    try:
        if name != path.name or _directory_identity(os.fstat(reopened)) != expected:
            raise ContractError(f"artifact ancestor changed during publication: {path}")
    finally:
        os.close(reopened)


def load_json_and_ref(
    root: Path,
    relative_path: str,
    *,
    maximum_bytes: int = MAX_JSON_BYTES,
    maximum_nodes: int = MAX_JSON_NODES,
) -> tuple[object, ArtifactRef]:
    """Parse JSON and construct its reference from one ancestor-safe stable descriptor."""

    normalized = validate_relative_path(relative_path)
    with stable_contained_descriptor(root, normalized) as descriptor:
        raw, digest, size = _read_descriptor(
            descriptor,
            display=normalized,
            maximum_bytes=maximum_bytes,
            retain_bytes=True,
        )
    assert raw is not None
    value = _parse_json_bytes(raw, display=normalized, maximum_nodes=maximum_nodes)
    return value, ArtifactRef(path=normalized, sha256=digest, size=size)


def load_json(
    path: Path,
    *,
    maximum_bytes: int = MAX_JSON_BYTES,
    maximum_nodes: int = MAX_JSON_NODES,
) -> object:
    if (
        isinstance(maximum_nodes, bool)
        or not isinstance(maximum_nodes, int)
        or not 1 <= maximum_nodes <= 10_000_000
    ):
        raise ContractError("JSON node limit must be in 1..10000000")
    with stable_regular_descriptor(path) as descriptor:
        raw, _, _ = _read_descriptor(
            descriptor,
            display=str(path),
            maximum_bytes=maximum_bytes,
            retain_bytes=True,
        )
    assert raw is not None
    return _parse_json_bytes(raw, display=str(path), maximum_nodes=maximum_nodes)


@contextmanager
def stable_regular_descriptor(path: Path):
    """Yield a regular file while retaining and revalidating its path binding."""

    parent_descriptor, name = _open_parent_for_write(path, create=False)
    parent_identity = _directory_identity(os.fstat(parent_descriptor))
    descriptor = -1
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ContractError(f"path is not a regular file: {path}")
        yield descriptor
        final = os.fstat(descriptor)
        if _descriptor_identity(final) != _descriptor_identity(initial):
            raise ContractError(f"regular file changed while consumed: {path}")
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(linked.st_mode) or _descriptor_identity(linked) != _descriptor_identity(
            final
        ):
            raise ContractError(f"regular-file path changed while consumed: {path}")
        reopened_parent, reopened_name = _open_parent_for_write(path, create=False)
        try:
            if (
                reopened_name != name
                or _directory_identity(os.fstat(reopened_parent)) != parent_identity
            ):
                raise ContractError(f"regular-file ancestor changed while consumed: {path}")
        finally:
            os.close(reopened_parent)
    except OSError as error:
        raise ContractError(f"cannot consume regular file {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


@contextmanager
def stable_contained_descriptor(root: Path, relative_path: str):
    """Yield one root-contained file with retained ancestor and entry bindings."""

    normalized = validate_relative_path(relative_path)
    root_descriptor = _open_directory_chain(root, create=False)
    root_identity = _directory_identity(os.fstat(root_descriptor))
    parent_descriptor = -1
    descriptor = -1
    parts = PurePosixPath(normalized).parts
    try:
        parent_descriptor = _walk_directory_descriptor(
            root_descriptor,
            parts[:-1],
            create=False,
        )
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ContractError(f"artifact is not a regular file: {normalized}")
        yield descriptor
        final = os.fstat(descriptor)
        if _descriptor_identity(final) != _descriptor_identity(initial):
            raise ContractError(f"artifact changed while consumed: {normalized}")
        linked = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(linked.st_mode) or _descriptor_identity(linked) != _descriptor_identity(
            final
        ):
            raise ContractError(f"artifact path changed while consumed: {normalized}")
        reopened_root = _open_directory_chain(root, create=False)
        try:
            if _directory_identity(os.fstat(reopened_root)) != root_identity:
                raise ContractError(f"artifact root changed while consumed: {normalized}")
            reopened_parent = _walk_directory_descriptor(
                reopened_root,
                parts[:-1],
                create=False,
            )
            try:
                if _directory_identity(os.fstat(reopened_parent)) != _directory_identity(
                    os.fstat(parent_descriptor)
                ):
                    raise ContractError(f"artifact ancestor changed while consumed: {normalized}")
            finally:
                os.close(reopened_parent)
        finally:
            os.close(reopened_root)
    except OSError as error:
        raise ContractError(f"cannot consume contained artifact {normalized}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        os.close(root_descriptor)


def _parse_json_bytes(raw: bytes, *, display: str, maximum_nodes: int) -> object:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"invalid JSON in {display}: {error}") from error
    _validate_json_shape(value, maximum_nodes=maximum_nodes)
    return value


def write_json_new(path: Path, value: object) -> None:
    """Publish an immutable JSON artifact without overwriting an existing path."""

    data = canonical_json_bytes(value)
    _write_new(path, data)


def write_bytes_new(path: Path, data: bytes) -> tuple[str, int]:
    """Publish bounded command output without replacing an earlier attempt log."""

    _write_new(path, data)
    return hashlib.sha256(data).hexdigest(), len(data)


def replace_json_state(path: Path, value: object) -> None:
    """Atomically replace the one mutable resume artifact."""

    data = canonical_json_bytes(value)
    parent_descriptor, name = _open_parent_for_write(path, create=True)
    temporary_name = f".{name}.{secrets.token_hex(12)}"
    temporary_descriptor = -1
    temporary_created = False
    temporary_status: os.stat_result | None = None
    parent_identity = _directory_identity(os.fstat(parent_descriptor))
    try:
        try:
            existing = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ContractError(f"state target must be a regular non-symlink file: {path}")
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        temporary_status = os.fstat(temporary_descriptor)
        with os.fdopen(temporary_descriptor, "wb") as stream:
            temporary_descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            published_status = os.fstat(stream.fileno())
        assert temporary_status is not None
        publish_regular_at(
            parent_descriptor,
            temporary_name,
            name,
            temporary_status,
            display=path,
            replace=True,
        )
        temporary_created = False
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino, linked.st_mode, linked.st_size) != (
            published_status.st_dev,
            published_status.st_ino,
            published_status.st_mode,
            published_status.st_size,
        ):
            raise ContractError(f"state path changed during publication: {path}")
        os.fsync(parent_descriptor)
        _assert_parent_binding(path, parent_identity)
    except Exception:
        if temporary_created:
            assert temporary_status is not None
            try:
                retire_bound_regular(
                    parent_descriptor,
                    temporary_name,
                    temporary_status,
                    display=path,
                )
            except ArtifactError as cleanup_error:
                raise ContractError(str(cleanup_error)) from cleanup_error
        raise
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        os.close(parent_descriptor)


def contained_path(root: Path, relative_path: str, *, must_exist: bool = False) -> Path:
    """Resolve a portable relative path below ``root`` without following symlinks."""

    validated = validate_relative_path(relative_path)
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved.joinpath(*PurePosixPath(validated).parts)
    current = root_resolved
    for part in PurePosixPath(validated).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ContractError(f"path crosses a symlink: {relative_path}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root_resolved):
        raise ContractError(f"path escapes root: {relative_path}")
    if must_exist and (not candidate.exists() or candidate.is_symlink() or not candidate.is_file()):
        raise ContractError(f"artifact is not a regular file: {relative_path}")
    return candidate


def ensure_contained_directory(root: Path, relative_path: str) -> Path:
    """Create and revalidate one repository-contained non-symlink directory tree."""

    normalized = validate_relative_path(relative_path)
    try:
        root_descriptor = _open_directory_chain(root, create=False)
        try:
            directory_descriptor = _walk_directory_descriptor(
                root_descriptor,
                PurePosixPath(normalized).parts,
                create=True,
            )
        finally:
            os.close(root_descriptor)
        os.close(directory_descriptor)
    except OSError as error:
        raise ContractError(
            f"cannot create contained directory {relative_path}: {error}"
        ) from error
    return root.absolute().joinpath(*PurePosixPath(normalized).parts)


def validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1_024:
        raise ContractError("path must be a non-empty string of at most 1024 characters")
    if "\\" in value or "\x00" in value:
        raise ContractError("path contains a forbidden character")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ContractError("path must be normalized and relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError("path must not contain empty, dot, or parent components")
    return value


def validate_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ContractError(f"{context} must be a safe identifier")
    return value


def validate_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{context} must be a lowercase SHA-256")
    return value


def validate_utc_timestamp(value: object, context: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ContractError(f"{context} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ContractError(f"{context} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo != UTC:
        raise ContractError(f"{context} must use UTC")
    return value


def require_clean_head(repository_root: Path, supplied_commit: str | None = None) -> str:
    """Return the exact HEAD commit after fail-closed tracked/untracked cleanliness checks."""

    root = repository_root.resolve(strict=True)
    head = _run_git_text(root, ["rev-parse", "--verify", "HEAD^{commit}"], maximum_bytes=128)
    if re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
        raise ContractError("repository HEAD is not a full hexadecimal commit identity")
    if supplied_commit is not None and supplied_commit != head:
        raise ContractError("supplied lowercase git commit is not the exact repository HEAD")
    for arguments, context in (
        (["diff", "--quiet", "--ignore-submodules", "--"], "unstaged"),
        (["diff", "--cached", "--quiet", "--ignore-submodules", "--"], "staged"),
    ):
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
            env=_git_environment(),
        )
        if completed.returncode not in {0, 1}:
            raise ContractError(f"cannot verify {context} repository state")
        if completed.returncode == 1:
            raise ContractError(f"repository has {context} changes")
    untracked = _run_git_text(
        root,
        ["ls-files", "--others", "--exclude-standard"],
        maximum_bytes=4_096,
        allow_empty=True,
    )
    if untracked:
        raise ContractError("repository has untracked non-ignored paths")
    return head


def _run_git_text(
    root: Path,
    arguments: list[str],
    *,
    maximum_bytes: int,
    allow_empty: bool = False,
) -> str:
    process = subprocess.Popen(
        ["/usr/bin/git", *arguments],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=_git_environment(),
    )
    assert process.stdout is not None
    output = bytearray()
    try:
        while chunk := process.stdout.read(min(4_096, maximum_bytes + 1 - len(output))):
            output.extend(chunk)
            if len(output) > maximum_bytes:
                # Git is a trusted single command here; do not signal a numeric
                # process group after its leader may have exited and been reused.
                with suppress(ProcessLookupError):
                    process.kill()
                raise ContractError("git command output exceeds its bound")
        return_code = process.wait(timeout=10)
    except BaseException:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise ContractError(f"git {' '.join(arguments)} failed")
    try:
        value = bytes(output).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ContractError("git emitted non-ASCII identity output") from error
    if not value and not allow_empty:
        raise ContractError("git emitted an empty identity")
    return value


def _git_environment() -> dict[str, str]:
    """Return a locale-stable Git environment without ambient repository/config overrides."""

    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "HOME": "/var/empty",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{context} must be an object")
    return value


def require_exact_keys(
    table: Mapping[str, Any], expected: set[str] | frozenset[str], context: str
) -> None:
    actual = set(table)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        raise ContractError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def require_string(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    maximum_length: int = MAX_TEXT_LENGTH,
    allow_empty: bool = False,
) -> str:
    value = table.get(key)
    if (
        not isinstance(value, str)
        or (not value and not allow_empty)
        or len(value) > maximum_length
        or "\x00" in value
    ):
        raise ContractError(f"{context}.{key} must be a bounded string")
    return value


def require_optional_string(
    table: Mapping[str, Any], key: str, context: str, *, maximum_length: int = MAX_TEXT_LENGTH
) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    return require_string(table, key, context, maximum_length=maximum_length)


def require_bool(table: Mapping[str, Any], key: str, context: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a Boolean")
    return value


def require_int(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{context}.{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{context}.{key} must not exceed {maximum}")
    return value


def require_number(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a finite number")
    converted = float(value)
    if not (-float("inf") < converted < float("inf")):
        raise ContractError(f"{context}.{key} must be finite")
    if minimum is not None and converted < minimum:
        raise ContractError(f"{context}.{key} must be at least {minimum}")
    if maximum is not None and converted > maximum:
        raise ContractError(f"{context}.{key} must not exceed {maximum}")
    return converted


def require_list(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    maximum_items: int,
    minimum_items: int = 0,
) -> list[Any]:
    value = table.get(key)
    if not isinstance(value, list) or not minimum_items <= len(value) <= maximum_items:
        raise ContractError(
            f"{context}.{key} must be an array with {minimum_items}..{maximum_items} items"
        )
    return value


def require_identifier(table: Mapping[str, Any], key: str, context: str) -> str:
    return validate_identifier(table.get(key), f"{context}.{key}")


def require_sha256(table: Mapping[str, Any], key: str, context: str) -> str:
    return validate_sha256(table.get(key), f"{context}.{key}")


def require_relative_path(table: Mapping[str, Any], key: str, context: str) -> str:
    try:
        return validate_relative_path(table.get(key))
    except ContractError as error:
        raise ContractError(f"{context}.{key}: {error}") from error


def require_enum(
    table: Mapping[str, Any], key: str, context: str, allowed: set[str] | frozenset[str]
) -> str:
    value = table.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise ContractError(f"{context}.{key} must be one of {sorted(allowed)}")
    return value


def _write_new(path: Path, data: bytes) -> None:
    parent_descriptor, name = _open_parent_for_write(path, create=True)
    parent_identity = _directory_identity(os.fstat(parent_descriptor))
    temporary_name = f".{name}.{secrets.token_hex(12)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    created = False
    created_status: os.stat_result | None = None
    try:
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        created = True
        created_status = os.fstat(descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ContractError(f"refusing to overwrite artifact {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            final = os.fstat(stream.fileno())
        assert created_status is not None
        publish_regular_at(
            parent_descriptor,
            temporary_name,
            name,
            created_status,
            display=path,
            replace=False,
        )
        _assert_published_bytes(parent_descriptor, name, path, final, data)
        os.fsync(parent_descriptor)
        _assert_parent_binding(path, parent_identity)
        created = False
    except Exception:
        if created:
            assert created_status is not None
            try:
                retire_bound_regular(
                    parent_descriptor,
                    temporary_name,
                    created_status,
                    display=path,
                )
            except ArtifactError as cleanup_error:
                raise ContractError(str(cleanup_error)) from cleanup_error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _assert_published_bytes(
    parent_descriptor: int,
    name: str,
    path: Path,
    source_status: os.stat_result,
    expected: bytes,
) -> None:
    """Reopen, consume, and rebind a newly published entry through one parent fd."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ContractError(f"cannot reopen new artifact {path}: {error}") from error
    try:
        initial = os.fstat(descriptor)
        # rename(2) may legitimately update ctime. The inode, mode, and exact
        # bytes are the publication identity retained from the writer fd.
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_mode,
            initial.st_size,
        ) != (
            source_status.st_dev,
            source_status.st_ino,
            source_status.st_mode,
            source_status.st_size,
        ):
            raise ContractError(f"new artifact path changed during publication: {path}")
        observed = bytearray()
        while len(observed) <= len(expected):
            chunk = os.read(descriptor, min(1024 * 1024, len(expected) + 1 - len(observed)))
            if not chunk:
                break
            observed.extend(chunk)
        if bytes(observed) != expected:
            raise ContractError(f"new artifact bytes changed during publication: {path}")
        final = os.fstat(descriptor)
        if _descriptor_identity(final) != _descriptor_identity(initial):
            raise ContractError(f"new artifact changed during publication: {path}")
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _descriptor_identity(linked) != _descriptor_identity(final):
            raise ContractError(f"new artifact path changed during publication: {path}")
    finally:
        os.close(descriptor)


def _read_descriptor(
    descriptor: int,
    *,
    display: str,
    maximum_bytes: int,
    retain_bytes: bool,
) -> tuple[bytes | None, str, int]:
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
        raise ContractError("artifact byte bound must be a non-negative integer")
    initial = os.fstat(descriptor)
    if not stat.S_ISREG(initial.st_mode) or initial.st_size > maximum_bytes:
        raise ContractError(f"artifact exceeds {maximum_bytes} bytes or is not regular: {display}")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_bytes else None
    total = 0
    while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total)):
        total += len(chunk)
        if total > maximum_bytes:
            raise ContractError(f"artifact exceeds {maximum_bytes} bytes: {display}")
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
    final = os.fstat(descriptor)
    if (
        total != initial.st_size
        or final.st_dev != initial.st_dev
        or final.st_ino != initial.st_ino
        or final.st_size != initial.st_size
        or final.st_mtime_ns != initial.st_mtime_ns
        or final.st_ctime_ns != initial.st_ctime_ns
    ):
        raise ContractError(f"artifact changed while reading: {display}")
    return (b"".join(chunks) if chunks is not None else None), digest.hexdigest(), total


def _open_parent_for_write(path: Path, *, create: bool) -> tuple[int, str]:
    absolute = path.absolute()
    parts = absolute.parts
    if not parts or len(parts) < 2:
        raise ContractError(f"path has no parent: {path}")
    root = Path(parts[0])
    descriptor = _open_directory_chain(root, create=False)
    try:
        parent = _walk_directory_descriptor(descriptor, parts[1:-1], create=create)
    finally:
        os.close(descriptor)
    return parent, parts[-1]


def _open_directory_chain(path: Path, *, create: bool) -> int:
    absolute = path.absolute()
    parts = absolute.parts
    descriptor = os.open(
        parts[0],
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        result = _walk_directory_descriptor(descriptor, parts[1:], create=create)
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)
    return result


def _walk_directory_descriptor(
    descriptor: int,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    current = os.dup(descriptor)
    try:
        for part in parts:
            if part in {"", ".", ".."}:
                raise ContractError("directory path contains an unsafe component")
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, 0o700, dir_fd=current)
            next_descriptor = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            os.close(current)
            current = next_descriptor
        return current
    except BaseException:
        os.close(current)
        raise


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number is forbidden: {value}")


def _validate_json_shape(value: object, *, maximum_nodes: int) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > maximum_nodes:
            raise ContractError(f"JSON exceeds {maximum_nodes} nodes")
        if depth > MAX_JSON_DEPTH:
            raise ContractError(f"JSON exceeds nesting depth {MAX_JSON_DEPTH}")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str) and len(current) > MAX_TEXT_LENGTH * 16:
            raise ContractError("JSON contains an oversized string")
