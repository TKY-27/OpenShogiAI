"""Small, resumable, rights-gated Phase 10R downloads.

The downloader is deliberately conservative: it accepts only an artifact that
is explicitly approved for training, downloads one object at a time, records
HTTP validators, and never removes an incomplete object.  It is suitable for
the progressive 1M/10M/50M acquisition checkpoints in the Phase 10R plan; it
is not a bulk crawler.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from open_shogi_training.data.phase10r_registry import (
    Phase10RArtifact,
    Phase10RRegistry,
    Phase10RRegistryError,
)

MIN_FREE_BYTES = 150 * 1024**3
MANIFEST_NAME = "phase10r-acquisition-manifest.jsonl"
CHUNK_BYTES = 1024 * 1024


class Phase10RAcquisitionError(RuntimeError):
    """Raised when a rights, storage, integrity, or transport guard fails."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    artifact_id: str
    status: str
    relative_path: str | None
    size_bytes: int | None
    sha256: str | None
    etag: str | None
    last_modified: str | None
    resumed_bytes: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "status": self.status,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "resumed_bytes": self.resumed_bytes,
            "message": self.message,
        }


def data_root_from_environment(
    *,
    environment: dict[str, str] | None = None,
    fallback: Path = Path("local/phase10r-data"),
) -> Path:
    """Resolve ``OPENSHOGI_DATA_ROOT`` without creating a repository payload."""

    values = environment if environment is not None else os.environ
    configured = values.get("OPENSHOGI_DATA_ROOT")
    return Path(configured) if configured else fallback


def check_free_space(data_root: Path, *, minimum_free_bytes: int = MIN_FREE_BYTES) -> int:
    """Return free bytes after checking the Phase 10R hard floor."""

    data_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_root).free
    if free < minimum_free_bytes:
        raise Phase10RAcquisitionError(
            f"free disk {free} is below the Phase 10R floor {minimum_free_bytes}"
        )
    return free


def acquire_artifact(
    registry: Phase10RRegistry,
    artifact_id: str,
    *,
    data_root: Path | None = None,
    purpose: str = "training",
    user_agent: str = "OpenShogiAI-Phase10R/1.0",
    minimum_free_bytes: int = MIN_FREE_BYTES,
    max_single_download_bytes: int | None = None,
    now: float | None = None,
) -> DownloadResult:
    """Resume one explicitly approved artifact and verify its final bytes."""

    artifact = registry.artifact(artifact_id)
    _check_rights(artifact, purpose)
    root = data_root or data_root_from_environment()
    root.mkdir(parents=True, exist_ok=True)
    check_free_space(root, minimum_free_bytes=minimum_free_bytes)
    limit = max_single_download_bytes or int(registry.policy["max_single_download_bytes"])
    if artifact.size_bytes is not None and artifact.size_bytes > limit:
        raise Phase10RAcquisitionError(
            f"artifact {artifact_id} is {artifact.size_bytes} bytes, above the configured limit"
        )
    name = _safe_filename(artifact.source_url)
    relative_path = Path("raw") / artifact.artifact_id / name
    destination = root / relative_path
    partial = destination.with_name(destination.name + ".part")
    metadata_path = partial.with_name(partial.name + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise Phase10RAcquisitionError(
                f"verified object path is not a regular non-symlink file: {destination}"
            )
        observed_size = destination.stat().st_size
        if artifact.size_bytes is not None and observed_size != artifact.size_bytes:
            raise Phase10RAcquisitionError(
                f"existing object size mismatch for {artifact_id}: "
                f"expected {artifact.size_bytes}, observed {observed_size}"
            )
        observed_sha = _sha256_file(destination)
        if artifact.sha256 is not None and observed_sha != artifact.sha256:
            raise Phase10RAcquisitionError(
                f"existing object SHA-256 mismatch for {artifact_id}: expected "
                f"{artifact.sha256}, observed {observed_sha}"
            )
        return DownloadResult(
            artifact_id=artifact_id,
            status="already_verified",
            relative_path=relative_path.as_posix(),
            size_bytes=observed_size,
            sha256=observed_sha,
            etag=None,
            last_modified=None,
            resumed_bytes=0,
            message="existing object is size- and checksum-verified",
        )
    known = _read_json(metadata_path)
    existing_bytes = partial.stat().st_size if partial.exists() else 0
    if existing_bytes and not (known.get("etag") or known.get("last_modified")):
        raise Phase10RAcquisitionError(
            f"partial download for {artifact_id} has no ETag or Last-Modified checkpoint"
        )
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
        if known.get("etag"):
            headers["If-Range"] = str(known["etag"])
        elif known.get("last_modified"):
            headers["If-Range"] = str(known["last_modified"])
    try:
        request = Request(artifact.source_url, headers=headers, method="GET")
        response = urlopen(request, timeout=60)
    except (HTTPError, URLError, TimeoutError) as error:
        raise Phase10RAcquisitionError(f"download failed for {artifact_id}: {error}") from error
    with response:
        response_headers = response.headers
        status = int(getattr(response, "status", 200) or 200)
        etag = response_headers.get("ETag")
        last_modified = response_headers.get("Last-Modified")
        if existing_bytes and status != 206:
            # The server ignored the range or the validator changed.  Preserve
            # the old partial object as evidence and begin a fresh partial.
            stale = partial.with_name(partial.name + f".stale-{int(now or time.time())}")
            partial.rename(stale)
            metadata_path.rename(metadata_path.with_name(metadata_path.name + ".stale"))
            existing_bytes = 0
        content_length = _header_int(response_headers, "Content-Length")
        expected_size = artifact.size_bytes
        if content_length is not None:
            expected_size = (existing_bytes + content_length) if existing_bytes else content_length
        if expected_size is not None and expected_size > limit:
            raise Phase10RAcquisitionError(
                f"response for {artifact_id} exceeds configured download limit: {expected_size}"
            )
        _write_json(
            metadata_path,
            {
                "artifact_id": artifact_id,
                "source_url": artifact.source_url,
                "etag": etag,
                "last_modified": last_modified,
                "size_bytes": existing_bytes,
            },
        )
        mode = "ab" if existing_bytes else "xb"
        resumed_bytes = existing_bytes
        with partial.open(mode) as handle:
            while True:
                check_free_space(root, minimum_free_bytes=minimum_free_bytes)
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                resumed_bytes += len(chunk)
                if resumed_bytes > limit:
                    raise Phase10RAcquisitionError(
                        f"download for {artifact_id} exceeded configured limit while streaming"
                    )
        _write_json(
            metadata_path,
            {
                "artifact_id": artifact_id,
                "source_url": artifact.source_url,
                "etag": etag,
                "last_modified": last_modified,
                "size_bytes": resumed_bytes,
            },
        )
    observed_size = partial.stat().st_size
    if expected_size is not None and observed_size != expected_size:
        raise Phase10RAcquisitionError(
            f"size mismatch for {artifact_id}: expected {expected_size}, observed {observed_size}"
        )
    observed_sha = _sha256_file(partial)
    if artifact.sha256 is not None and observed_sha != artifact.sha256:
        raise Phase10RAcquisitionError(
            f"SHA-256 mismatch for {artifact_id}: expected {artifact.sha256}, "
            f"observed {observed_sha}"
        )
    if destination.exists():
        raise Phase10RAcquisitionError(f"refusing to overwrite verified object: {destination}")
    partial.rename(destination)
    metadata_path.unlink()
    result = DownloadResult(
        artifact_id=artifact_id,
        status="verified",
        relative_path=relative_path.as_posix(),
        size_bytes=observed_size,
        sha256=observed_sha,
        etag=etag,
        last_modified=last_modified,
        resumed_bytes=existing_bytes,
        message="downloaded and checksum-verified",
    )
    _append_manifest(root, result)
    return result


def dry_run(
    registry: Phase10RRegistry,
    artifact_ids: list[str] | None = None,
    *,
    data_root: Path | None = None,
    minimum_free_bytes: int = MIN_FREE_BYTES,
) -> dict[str, Any]:
    """Return a no-write acquisition plan for approved artifacts."""

    root = data_root or data_root_from_environment()
    selected = artifact_ids or [artifact.artifact_id for artifact in registry.approved_artifacts()]
    artifacts = [registry.artifact(artifact_id) for artifact_id in selected]
    for artifact in artifacts:
        _check_rights(artifact, "training")
    free = shutil.disk_usage(root).free if root.exists() else None
    total_known = sum(artifact.size_bytes or 0 for artifact in artifacts)
    return {
        "schema": "phase10r_acquisition_plan/v1",
        "data_root": root.as_posix(),
        "minimum_free_bytes": minimum_free_bytes,
        "free_bytes_observed": free,
        "free_space_check": free is None or free >= minimum_free_bytes,
        "artifact_count": len(artifacts),
        "known_bytes": total_known,
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "source_url": artifact.source_url,
                "exact_artifact_name": artifact.exact_artifact_name,
                "size_bytes": artifact.size_bytes,
                "format": artifact.format,
                "compression": artifact.compression,
                "state": artifact.state,
            }
            for artifact in artifacts
        ],
    }


def _check_rights(artifact: Phase10RArtifact, purpose: str) -> None:
    if purpose == "training":
        allowed = artifact.trainable
    elif purpose == "local-inspection":
        allowed = artifact.trainable or artifact.local_only
    else:
        raise Phase10RAcquisitionError(f"unknown acquisition purpose: {purpose}")
    if not allowed:
        raise Phase10RRegistryError(
            f"artifact {artifact.artifact_id} is {artifact.state}; purpose {purpose} is not allowed"
        )


def _safe_filename(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1] or "artifact.bin"
    if "?" in name or "#" in name:
        name = name.split("?", 1)[0].split("#", 1)[0]
    if not name or name in {".", ".."} or any(char in name for char in "\\\x00\n\r"):
        raise Phase10RAcquisitionError(f"source URL has no safe filename: {url}")
    return name


def _header_int(headers: HTTPMessage, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise Phase10RAcquisitionError(f"invalid HTTP {name}: {value!r}") from error
    if parsed < 0:
        raise Phase10RAcquisitionError(f"invalid HTTP {name}: {value!r}")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase10RAcquisitionError(f"invalid download checkpoint: {path}") from error
    if not isinstance(payload, dict):
        raise Phase10RAcquisitionError(f"download checkpoint must be an object: {path}")
    return payload


def _write_json(path: Path, payload: MappingLike) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise Phase10RAcquisitionError(f"stale temporary checkpoint exists: {temporary}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.rename(path)


def _append_manifest(root: Path, result: DownloadResult) -> None:
    path = root / MANIFEST_NAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")


class MappingLike(dict[str, Any]):
    """Typing helper avoiding a runtime dependency on ``collections.abc``."""


__all__ = [
    "CHUNK_BYTES",
    "MANIFEST_NAME",
    "MIN_FREE_BYTES",
    "DownloadResult",
    "Phase10RAcquisitionError",
    "acquire_artifact",
    "check_free_space",
    "data_root_from_environment",
    "dry_run",
]
