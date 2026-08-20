"""Deterministic JSON/JSONL writers used by Phase 3 dataset artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    publish_regular_at,
    retire_bound_regular,
    stable_parent_descriptor,
    stable_regular_descriptor,
)


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """Content identity returned after an atomic artifact publication."""

    sha256: str
    size: int
    records: int


def compact_json_bytes(value: object) -> bytes:
    """Encode deterministic UTF-8 JSON with sorted keys and no incidental spaces."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def write_json_atomic(path: Path, value: object) -> ArtifactDigest:
    """Create a compact JSON file atomically without replacing an existing path."""

    payload = compact_json_bytes(value) + b"\n"
    with stable_parent_descriptor(path, create=True) as (parent_descriptor, name):
        descriptor, temporary_name, temporary_status = _temporary_file(parent_descriptor, name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
                temporary_status = os.fstat(output.fileno())
            _publish_without_overwrite(
                parent_descriptor,
                temporary_name,
                path,
                temporary_status,
            )
            temporary_name = ""
        except BaseException:
            if temporary_name:
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
    return ArtifactDigest(
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        records=1,
    )


def write_jsonl_gzip_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    compresslevel: int = 9,
) -> ArtifactDigest:
    """Create reproducible gzip JSONL without replacing an existing path.

    The gzip header has an empty filename and ``mtime=0``.  Every row is compact
    sorted-key JSON followed by exactly one LF.
    """

    if not 0 <= compresslevel <= 9:
        raise ValueError("compresslevel must be between 0 and 9")
    records = 0
    digest = hashlib.sha256()
    size = 0
    with stable_parent_descriptor(path, create=True) as (parent_descriptor, name):
        descriptor, temporary_name, temporary_status = _temporary_file(parent_descriptor, name)
        try:
            with os.fdopen(descriptor, "wb") as raw_output:
                descriptor = -1
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=compresslevel,
                    fileobj=raw_output,
                    mtime=0,
                ) as gzip_output:
                    for row in rows:
                        if not isinstance(row, Mapping):
                            raise TypeError("each JSONL row must be a mapping")
                        gzip_output.write(compact_json_bytes(row))
                        gzip_output.write(b"\n")
                        records += 1
                raw_output.flush()
                os.fsync(raw_output.fileno())
                completed = os.fstat(raw_output.fileno())
                offset = 0
                while offset < completed.st_size:
                    chunk = os.pread(
                        raw_output.fileno(),
                        min(1024 * 1024, completed.st_size - offset),
                        offset,
                    )
                    if not chunk:
                        raise OSError("gzip temporary ended before its recorded size")
                    digest.update(chunk)
                    size += len(chunk)
                    offset += len(chunk)
                temporary_status = completed
            _publish_without_overwrite(
                parent_descriptor,
                temporary_name,
                path,
                temporary_status,
            )
            temporary_name = ""
        except BaseException:
            if temporary_name:
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
    return ArtifactDigest(sha256=digest.hexdigest(), size=size, records=records)


def iter_jsonl_gzip(
    path: Path,
    *,
    max_compressed_bytes: int = 1_073_741_824,
    max_uncompressed_bytes: int = 1_073_741_824,
    max_records: int = 100_000_000,
    max_line_bytes: int = 4_194_304,
) -> Iterator[dict[str, Any]]:
    """Read a bounded gzip JSONL artifact and reject malformed/non-object rows."""

    try:
        with stable_regular_descriptor(path) as descriptor:
            size = os.fstat(descriptor).st_size
            if size > max_compressed_bytes:
                raise ValueError(f"compressed JSONL exceeds {max_compressed_bytes} bytes")
            with os.fdopen(os.dup(descriptor), "rb") as retained:
                input_file = gzip.GzipFile(fileobj=retained, mode="rb")
                try:
                    yield from _iter_gzip_lines(
                        input_file,
                        max_uncompressed_bytes=max_uncompressed_bytes,
                        max_records=max_records,
                        max_line_bytes=max_line_bytes,
                    )
                finally:
                    input_file.close()
    except ArtifactError as error:
        raise ValueError(str(error)) from error


def _iter_gzip_lines(
    input_file: gzip.GzipFile,
    *,
    max_uncompressed_bytes: int,
    max_records: int,
    max_line_bytes: int,
) -> Iterator[dict[str, Any]]:
    uncompressed_bytes = 0
    line_number = 0
    while line := input_file.readline(max_line_bytes + 1):
        line_number += 1
        if line_number > max_records:
            raise ValueError(f"JSONL exceeds {max_records} records")
        if len(line) > max_line_bytes:
            raise ValueError(f"JSONL line {line_number} exceeds {max_line_bytes} bytes")
        uncompressed_bytes += len(line)
        if uncompressed_bytes > max_uncompressed_bytes:
            raise ValueError(f"uncompressed JSONL exceeds {max_uncompressed_bytes} bytes")
        if not line.endswith(b"\n"):
            raise ValueError(f"JSONL line {line_number} is not LF terminated")
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        yield value


def _temporary_file(parent_descriptor: int, name: str) -> tuple[int, str, os.stat_result]:
    temporary_name = f".{name}.{secrets.token_hex(12)}.pending"
    descriptor = os.open(
        temporary_name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_descriptor,
    )
    return descriptor, temporary_name, os.fstat(descriptor)


def _publish_without_overwrite(
    parent_descriptor: int,
    temporary_name: str,
    destination: Path,
    temporary_status: os.stat_result,
) -> None:
    try:
        publish_regular_at(
            parent_descriptor,
            temporary_name,
            destination.name,
            temporary_status,
            display=destination,
            replace=False,
        )
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}") from None
    published = os.stat(
        destination.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        published.st_dev,
        published.st_ino,
        published.st_mode,
        published.st_size,
    ) != (
        temporary_status.st_dev,
        temporary_status.st_ino,
        temporary_status.st_mode,
        temporary_status.st_size,
    ):
        raise ArtifactError(f"artifact path changed while it was published: {destination}")
    os.fsync(parent_descriptor)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result
