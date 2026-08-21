"""Safe inventory and extraction helpers for Phase 10R release archives."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class Phase10RArchiveError(ValueError):
    """Raised when an archive cannot be safely inspected or extracted."""


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    name: str
    size_bytes: int
    compressed_size_bytes: int
    is_directory: bool
    is_symlink: bool
    crc32: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "is_directory": self.is_directory,
            "is_symlink": self.is_symlink,
            "crc32": self.crc32,
        }


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    path: str
    archive_format: str
    sha256: str
    size_bytes: int
    entries: tuple[ArchiveEntry, ...]
    total_uncompressed_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "archive_format": self.archive_format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "entry_count": len(self.entries),
            "regular_file_count": sum(
                not entry.is_directory and not entry.is_symlink for entry in self.entries
            ),
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def inventory_archive(
    path: Path,
    *,
    max_entries: int = 100_000,
    max_uncompressed_bytes: int = 512 * 1024 * 1024 * 1024,
) -> ArchiveInventory:
    """Inventory a ZIP archive without extracting it.

    LZH is deliberately reported as ``unsupported_without_external_tool``;
    this prevents silently treating a failed extraction as an empty dataset.
    """

    if not path.is_file():
        raise Phase10RArchiveError(f"archive is not a regular file: {path}")
    suffix = path.suffix.lower()
    if suffix in {".lzh", ".lha"}:
        raise Phase10RArchiveError(
            "LZH inventory requires an explicitly selected external extractor; none was run"
        )
    if suffix != ".zip" and not zipfile.is_zipfile(path):
        raise Phase10RArchiveError(f"unsupported archive format: {path.suffix or '<none>'}")
    entries: list[ArchiveEntry] = []
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > max_entries:
            raise Phase10RArchiveError(f"archive exceeds {max_entries} entries")
        for info in infos:
            name = _safe_member_name(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            is_symlink = stat.S_ISLNK(mode)
            is_directory = info.is_dir() or name.endswith("/")
            if is_symlink:
                raise Phase10RArchiveError(f"archive contains a symlink: {name}")
            total += info.file_size
            if total > max_uncompressed_bytes:
                raise Phase10RArchiveError(
                    f"archive exceeds {max_uncompressed_bytes} uncompressed bytes"
                )
            entries.append(
                ArchiveEntry(
                    name=name,
                    size_bytes=info.file_size,
                    compressed_size_bytes=info.compress_size,
                    is_directory=is_directory,
                    is_symlink=is_symlink,
                    crc32=info.CRC,
                )
            )
    return ArchiveInventory(
        path=path.as_posix(),
        archive_format="zip",
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        entries=tuple(entries),
        total_uncompressed_bytes=total,
    )


def extract_zip(
    archive_path: Path,
    output_root: Path,
    *,
    max_entries: int = 100_000,
    max_uncompressed_bytes: int = 512 * 1024 * 1024 * 1024,
) -> ArchiveInventory:
    """Extract a validated ZIP into a new or empty directory.

    Existing files are never overwritten.  The caller can resume by invoking
    this function again after removing only a verified, incomplete extraction;
    the function itself is intentionally conservative and fails on collisions.
    """

    inventory = inventory_archive(
        archive_path,
        max_entries=max_entries,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve(strict=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in inventory.entries:
            destination = (root / Path(*PurePosixPath(entry.name).parts)).resolve()
            if destination != root and root not in destination.parents:
                raise Phase10RArchiveError(f"archive member escapes output root: {entry.name}")
            if entry.is_directory:
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise Phase10RArchiveError(f"refusing to overwrite extracted file: {destination}")
            with archive.open(entry.name, "r") as source, destination.open("xb") as target:
                _copy_bounded(source, target, entry.size_bytes)
            if destination.stat().st_size != entry.size_bytes:
                raise Phase10RArchiveError(f"extracted size mismatch: {entry.name}")
    return inventory


def write_inventory(path: Path, output: Path, inventory: ArchiveInventory) -> None:
    """Write deterministic inventory JSON without machine-local path fields."""

    payload = inventory.as_dict()
    payload["path"] = path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise Phase10RArchiveError(f"refusing to overwrite inventory: {output}")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _safe_member_name(name: str) -> str:
    if not name or "\x00" in name:
        raise Phase10RArchiveError("archive contains an empty or NUL member name")
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Phase10RArchiveError(f"unsafe archive member path: {name!r}")
    if any(ord(char) < 32 for char in normalized):
        raise Phase10RArchiveError(f"archive member contains a control character: {name!r}")
    return "/".join(path.parts)


def _copy_bounded(source: Any, target: Any, expected_size: int) -> None:
    copied = 0
    while copied < expected_size:
        chunk = source.read(min(1024 * 1024, expected_size - copied))
        if not chunk:
            raise Phase10RArchiveError("archive member ended before its declared size")
        target.write(chunk)
        copied += len(chunk)
    if source.read(1):
        raise Phase10RArchiveError("archive member exceeded its declared size")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ArchiveEntry",
    "ArchiveInventory",
    "Phase10RArchiveError",
    "extract_zip",
    "inventory_archive",
    "write_inventory",
]
