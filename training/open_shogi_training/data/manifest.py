"""Append-only, fsynced acquisition provenance manifest."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.data.registry import LicenseEvidence

MANIFEST_SCHEMA_VERSION: Final = 1
MAX_MANIFEST_BYTES: Final = 8 * 1024 * 1024
MAX_MANIFEST_LINE_BYTES: Final = 32 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_COMPLETED_KEYS = frozenset(
    {
        "schema_version",
        "event",
        "source_id",
        "object_id",
        "url",
        "retrieved_at",
        "sha256",
        "size",
        "content_type",
        "etag",
        "last_modified",
        "object_path",
        "original_filename",
        "data_format",
        "compression",
        "license",
        "license_evidence",
        "evidence_snapshots",
        "redistributable",
        "machine_learning_allowed",
    }
)
_PARTIAL_KEYS = frozenset(
    {
        "schema_version",
        "event",
        "source_id",
        "object_id",
        "url",
        "recorded_at",
        "partial_path",
        "size",
        "sha256",
        "etag",
        "last_modified",
    }
)
_EVIDENCE_KEYS = frozenset({"url", "local_path", "quote"})
_SNAPSHOT_KEYS = frozenset(
    {
        "evidence_id",
        "url",
        "retrieved_at",
        "sha256",
        "size",
        "content_type",
        "object_path",
    }
)


class ManifestError(ValueError):
    """Raised when manifest state is malformed, conflicting, or unsafe."""


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """Hashed local copy of one exact external evidence page."""

    evidence_id: str
    url: str
    retrieved_at: str
    sha256: str
    size: int
    content_type: str | None
    object_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "size": self.size,
            "content_type": self.content_type,
            "object_path": self.object_path,
        }


@dataclass(frozen=True, slots=True)
class CompletedObject:
    """Stable v1 record consumed by normalization."""

    source_id: str
    object_id: str
    url: str
    retrieved_at: str
    sha256: str
    size: int
    content_type: str | None
    etag: str | None
    last_modified: str | None
    object_path: str
    original_filename: str
    data_format: str
    compression: str
    license: str
    license_evidence: tuple[LicenseEvidence, ...]
    evidence_snapshots: tuple[EvidenceSnapshot, ...]
    redistributable: bool
    machine_learning_allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "event": "completed",
            "source_id": self.source_id,
            "object_id": self.object_id,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "size": self.size,
            "content_type": self.content_type,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "object_path": self.object_path,
            "original_filename": self.original_filename,
            "data_format": self.data_format,
            "compression": self.compression,
            "license": self.license,
            "license_evidence": [item.as_dict() for item in self.license_evidence],
            "evidence_snapshots": [item.as_dict() for item in self.evidence_snapshots],
            "redistributable": self.redistributable,
            "machine_learning_allowed": self.machine_learning_allowed,
        }


@dataclass(frozen=True, slots=True)
class PartialObject:
    """Resumable partial state, valid only when a validator is present."""

    source_id: str
    object_id: str
    url: str
    recorded_at: str
    partial_path: str
    size: int
    sha256: str
    etag: str | None
    last_modified: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "event": "partial",
            "source_id": self.source_id,
            "object_id": self.object_id,
            "url": self.url,
            "recorded_at": self.recorded_at,
            "partial_path": self.partial_path,
            "size": self.size,
            "sha256": self.sha256,
            "etag": self.etag,
            "last_modified": self.last_modified,
        }


@dataclass(frozen=True, slots=True)
class ManifestSnapshot:
    completed: tuple[CompletedObject, ...]
    partials: tuple[PartialObject, ...]

    def completed_by_url(self) -> dict[str, CompletedObject]:
        records: dict[str, CompletedObject] = {}
        for record in self.completed:
            previous = records.get(record.url)
            if previous is not None and previous != record:
                raise ManifestError(f"conflicting completed records for URL: {record.url}")
            records[record.url] = record
        return records

    def completed_by_sha256(self) -> dict[str, CompletedObject]:
        return {record.sha256: record for record in self.completed}

    def latest_partial_by_url(self) -> dict[str, PartialObject]:
        return {record.url: record for record in self.partials}


class ManifestStore:
    """Read and append the bounded JSONL manifest rooted beside object data."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "manifest.jsonl"
        self._append_lock = threading.Lock()

    def snapshot(self) -> ManifestSnapshot:
        try:
            path_status = self.path.lstat()
        except FileNotFoundError:
            return ManifestSnapshot(completed=(), partials=())
        except OSError as error:
            raise ManifestError(f"cannot inspect manifest: {error}") from error
        if not stat.S_ISREG(path_status.st_mode) or stat.S_ISLNK(path_status.st_mode):
            raise ManifestError("manifest must be a non-symlink regular file")

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags)
        except OSError as error:
            raise ManifestError(f"cannot open manifest: {error}") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise ManifestError("manifest must be a regular file")
            if file_status.st_size > MAX_MANIFEST_BYTES:
                raise ManifestError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
            with os.fdopen(descriptor, "rb", closefd=False) as manifest_file:
                raw = manifest_file.read(MAX_MANIFEST_BYTES + 1)
        except OSError as error:
            raise ManifestError(f"cannot read manifest: {error}") from error
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ManifestError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        if raw and not raw.endswith(b"\n"):
            raise ManifestError("manifest has an incomplete final line")

        completed: list[CompletedObject] = []
        partials: list[PartialObject] = []
        for line_number, raw_line in enumerate(raw.splitlines(), start=1):
            if not raw_line:
                raise ManifestError(f"manifest line {line_number} is empty")
            if len(raw_line) > MAX_MANIFEST_LINE_BYTES:
                raise ManifestError(f"manifest line {line_number} is too large")
            payload = _load_unique_json(raw_line, line_number=line_number)
            event = payload.get("event")
            if event == "completed":
                completed.append(_parse_completed(payload, line_number=line_number))
            elif event == "partial":
                partials.append(_parse_partial(payload, line_number=line_number))
            else:
                raise ManifestError(f"manifest line {line_number} has unknown event")

        snapshot = ManifestSnapshot(completed=tuple(completed), partials=tuple(partials))
        snapshot.completed_by_url()
        return snapshot

    def completed_records(self) -> tuple[CompletedObject, ...]:
        """Return strict completed v1 records for the normalization boundary."""

        return self.snapshot().completed

    def append_completed(self, record: CompletedObject) -> None:
        _validate_completed(record)
        self._append(record.as_dict())

    def append_partial(self, record: PartialObject) -> None:
        _validate_partial(record)
        self._append(record.as_dict())

    def resolve_object_path(self, record: CompletedObject) -> Path:
        return _resolve_relative(self.root, record.object_path, "object_path")

    def resolve_partial_path(self, record: PartialObject) -> Path:
        return _resolve_relative(self.root, record.partial_path, "partial_path")

    def _append(self, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_MANIFEST_LINE_BYTES:
            raise ManifestError("manifest event is too large")

        with self._append_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            root_status = self.root.lstat()
            if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
                raise ManifestError("manifest root must be a non-symlink directory")
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except OSError as error:
                raise ManifestError(f"cannot open manifest for append: {error}") from error
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                file_status = os.fstat(descriptor)
                if not stat.S_ISREG(file_status.st_mode):
                    raise ManifestError("manifest must be a regular file")
                if file_status.st_size + len(encoded) > MAX_MANIFEST_BYTES:
                    raise ManifestError(f"manifest append would exceed {MAX_MANIFEST_BYTES} bytes")
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("manifest append made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            except OSError as error:
                raise ManifestError(f"cannot append manifest event: {error}") from error
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            _fsync_directory(self.root)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_unique_json(raw_line: bytes, *, line_number: int) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"manifest line {line_number} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw_line, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ManifestError(f"invalid JSON on manifest line {line_number}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"manifest line {line_number} must be an object")
    return payload


def _parse_completed(payload: dict[str, Any], *, line_number: int) -> CompletedObject:
    _require_keys(payload, _COMPLETED_KEYS, line_number)
    _require_schema(payload, line_number)
    if payload["event"] != "completed":
        raise ManifestError(f"manifest line {line_number} event mismatch")
    raw_evidence = payload["license_evidence"]
    if not isinstance(raw_evidence, list):
        raise ManifestError(f"manifest line {line_number} license_evidence must be a list")
    evidence: list[LicenseEvidence] = []
    for index, raw_item in enumerate(raw_evidence):
        if not isinstance(raw_item, dict):
            raise ManifestError(f"manifest line {line_number} evidence {index} must be an object")
        if frozenset(raw_item) != _EVIDENCE_KEYS:
            raise ManifestError(f"manifest line {line_number} evidence {index} keys mismatch")
        evidence.append(
            LicenseEvidence(
                url=_string(raw_item, "url", line_number),
                local_path=_string(raw_item, "local_path", line_number),
                quote=_string(raw_item, "quote", line_number),
            )
        )
    raw_snapshots = payload["evidence_snapshots"]
    if not isinstance(raw_snapshots, list):
        raise ManifestError(f"manifest line {line_number} evidence_snapshots must be a list")
    snapshots: list[EvidenceSnapshot] = []
    for index, raw_item in enumerate(raw_snapshots):
        if not isinstance(raw_item, dict) or frozenset(raw_item) != _SNAPSHOT_KEYS:
            raise ManifestError(f"manifest line {line_number} snapshot {index} is invalid")
        snapshots.append(
            EvidenceSnapshot(
                evidence_id=_string(raw_item, "evidence_id", line_number),
                url=_string(raw_item, "url", line_number),
                retrieved_at=_string(raw_item, "retrieved_at", line_number),
                sha256=_string(raw_item, "sha256", line_number),
                size=_integer(raw_item, "size", line_number),
                content_type=_optional_string(raw_item, "content_type", line_number),
                object_path=_string(raw_item, "object_path", line_number),
            )
        )
    record = CompletedObject(
        source_id=_string(payload, "source_id", line_number),
        object_id=_string(payload, "object_id", line_number),
        url=_string(payload, "url", line_number),
        retrieved_at=_string(payload, "retrieved_at", line_number),
        sha256=_string(payload, "sha256", line_number),
        size=_integer(payload, "size", line_number),
        content_type=_optional_string(payload, "content_type", line_number),
        etag=_optional_string(payload, "etag", line_number),
        last_modified=_optional_string(payload, "last_modified", line_number),
        object_path=_string(payload, "object_path", line_number),
        original_filename=_string(payload, "original_filename", line_number),
        data_format=_string(payload, "data_format", line_number),
        compression=_string(payload, "compression", line_number),
        license=_string(payload, "license", line_number),
        license_evidence=tuple(evidence),
        evidence_snapshots=tuple(snapshots),
        redistributable=_boolean(payload, "redistributable", line_number),
        machine_learning_allowed=_boolean(payload, "machine_learning_allowed", line_number),
    )
    _validate_completed(record)
    return record


def _parse_partial(payload: dict[str, Any], *, line_number: int) -> PartialObject:
    _require_keys(payload, _PARTIAL_KEYS, line_number)
    _require_schema(payload, line_number)
    if payload["event"] != "partial":
        raise ManifestError(f"manifest line {line_number} event mismatch")
    record = PartialObject(
        source_id=_string(payload, "source_id", line_number),
        object_id=_string(payload, "object_id", line_number),
        url=_string(payload, "url", line_number),
        recorded_at=_string(payload, "recorded_at", line_number),
        partial_path=_string(payload, "partial_path", line_number),
        size=_integer(payload, "size", line_number),
        sha256=_string(payload, "sha256", line_number),
        etag=_optional_string(payload, "etag", line_number),
        last_modified=_optional_string(payload, "last_modified", line_number),
    )
    _validate_partial(record)
    return record


def _validate_completed(record: CompletedObject) -> None:
    _validate_common(record.source_id, record.object_id, record.url)
    _validate_timestamp(record.retrieved_at, "retrieved_at")
    if _SHA256_RE.fullmatch(record.sha256) is None:
        raise ManifestError("sha256 must contain 64 lowercase hexadecimal characters")
    if record.size <= 0:
        raise ManifestError("completed size must be positive")
    _validate_relative(record.object_path, "object_path")
    _validate_basename(record.original_filename, "original_filename")
    if not record.license_evidence:
        raise ManifestError("completed records require license_evidence")
    if not record.evidence_snapshots:
        raise ManifestError("completed records require external evidence snapshots")
    for snapshot in record.evidence_snapshots:
        _validate_common(record.source_id, snapshot.evidence_id, snapshot.url)
        _validate_timestamp(snapshot.retrieved_at, "evidence retrieved_at")
        if _SHA256_RE.fullmatch(snapshot.sha256) is None:
            raise ManifestError("evidence sha256 must be lowercase hexadecimal")
        if snapshot.size <= 0:
            raise ManifestError("evidence snapshot size must be positive")
        _validate_relative(snapshot.object_path, "evidence object_path")
        _validate_optional_header(snapshot.content_type, "evidence content_type")
    if not record.machine_learning_allowed:
        raise ManifestError("completed records must be approved for machine learning")
    for name, value in (
        ("content_type", record.content_type),
        ("etag", record.etag),
        ("last_modified", record.last_modified),
    ):
        _validate_optional_header(value, name)


def _validate_partial(record: PartialObject) -> None:
    _validate_common(record.source_id, record.object_id, record.url)
    _validate_timestamp(record.recorded_at, "recorded_at")
    _validate_relative(record.partial_path, "partial_path")
    if record.size <= 0:
        raise ManifestError("partial size must be positive")
    if _SHA256_RE.fullmatch(record.sha256) is None:
        raise ManifestError("partial sha256 must be lowercase hexadecimal")
    _validate_optional_header(record.etag, "etag")
    _validate_optional_header(record.last_modified, "last_modified")


def _validate_common(source_id: str, object_id: str, url: str) -> None:
    for name, value in (("source_id", source_id), ("object_id", object_id), ("url", url)):
        if not value or len(value) > 2_048:
            raise ManifestError(f"{name} must be a bounded non-empty string")


def _validate_timestamp(value: str, name: str) -> None:
    if not value.endswith("Z"):
        raise ManifestError(f"{name} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ManifestError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo != UTC:
        raise ManifestError(f"{name} must use UTC")


def _validate_optional_header(value: str | None, name: str) -> None:
    if value is not None and (
        not value
        or value != value.strip()
        or len(value) > 2_048
        or "\r" in value
        or "\n" in value
        or "\x00" in value
    ):
        raise ManifestError(f"{name} is invalid")


def _validate_relative(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ManifestError(f"{name} must be a normalized relative POSIX path")


def _validate_basename(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if not value or path.name != value or value in {".", ".."}:
        raise ManifestError(f"{name} must be a safe basename")


def _resolve_relative(root: Path, relative: str, name: str) -> Path:
    _validate_relative(relative, name)
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*PurePosixPath(relative).parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManifestError(f"{name} escapes the manifest root")
    return resolved


def _require_keys(payload: Mapping[str, Any], expected: frozenset[str], line_number: int) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise ManifestError(
            f"manifest line {line_number} keys mismatch; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_schema(payload: Mapping[str, Any], line_number: int) -> None:
    value = payload["schema_version"]
    if not isinstance(value, int) or isinstance(value, bool) or value != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"manifest line {line_number} has unsupported schema_version")


def _string(payload: Mapping[str, Any], key: str, line_number: int) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ManifestError(f"manifest line {line_number} {key} must be a bounded string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str, line_number: int) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"manifest line {line_number} {key} must be string or null")
    return value


def _integer(payload: Mapping[str, Any], key: str, line_number: int) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"manifest line {line_number} {key} must be an integer")
    return value


def _boolean(payload: Mapping[str, Any], key: str, line_number: int) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ManifestError(f"manifest line {line_number} {key} must be a boolean")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
