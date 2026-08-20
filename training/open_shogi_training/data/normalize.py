"""Bounded, provenance-preserving Phase 3 normalization pipeline."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from open_shogi_training.data.aobazero import (
    AdaptedAobaZeroCsa,
    AobaZeroAdaptationError,
    AobaZeroLimits,
    adapt_aobazero_csa,
    metadata_date,
    metadata_ratings,
)
from open_shogi_training.data.gzip_jsonl import (
    ArtifactDigest,
    write_json_atomic,
    write_jsonl_gzip_atomic,
)
from open_shogi_training.data.registry import DataSource
from open_shogi_training.data.splits import SplitPolicy, assign_game_split
from open_shogi_training.selfplay.common import ArtifactRef, contained_path
from open_shogi_training.selfplay.execution import CommandRunner

_DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_USI_MOVE_RE = re.compile(r"^(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])$")
_EXPORT_SCHEMA = "phase3_csa_export/v1"
_MAX_EXPORT_BYTES = 128 * 1024 * 1024
_MAX_EXPORT_LINE_BYTES = 32 * 1024 * 1024
_MAX_NORMALIZED_POSITIONS = 100 * (2_048 + 1)
_MAX_RAW_BYTES = 16 * 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 0x00000001
_RENAME_EXCL = 0x00000004
_EXPORT_OUTCOMES = frozenset({"black_win", "white_win", "draw", "unknown"})
_RESULT_VALIDATIONS = frozenset({"verified", "external_condition", "missing"})
_OK_EXPORT_KEYS = frozenset(
    {
        "schema",
        "status",
        "inputFile",
        "normalizedCsa",
        "initialSfen",
        "positionSfens",
        "usiMoves",
        "blackName",
        "whiteName",
        "terminalReason",
        "outcome",
        "resultValidation",
    }
)
_REJECTED_EXPORT_KEYS = frozenset({"schema", "status", "inputFile", "reason"})
_DEFAULT_ADAPTER_LIMITS = AobaZeroLimits()


class Exporter(Protocol):
    """Injectable Rust exporter boundary used by tests and the CLI adapter."""

    def __call__(self, stage_dir: Path, output_path: Path, max_games: int) -> None: ...


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Complete bounded configuration for one initial Phase 3 sample."""

    dataset_id: str
    split_policy: SplitPolicy
    max_games: int = 100
    max_positions: int = 100_000
    terminal_tail_positions: int = 8
    max_raw_bytes: int = 16_777_216
    exporter_timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if _DATASET_ID_RE.fullmatch(self.dataset_id) is None:
            raise ValueError("dataset_id must match [a-z0-9][a-z0-9_-]{0,63}")
        if (
            isinstance(self.max_games, bool)
            or not isinstance(self.max_games, int)
            or not 1 <= self.max_games <= 100
        ):
            raise ValueError("initial Phase 3 max_games must be between 1 and 100")
        if (
            isinstance(self.max_positions, bool)
            or not isinstance(self.max_positions, int)
            or not 1 <= self.max_positions <= _MAX_NORMALIZED_POSITIONS
        ):
            raise ValueError(f"max_positions must be between 1 and {_MAX_NORMALIZED_POSITIONS}")
        if (
            isinstance(self.terminal_tail_positions, bool)
            or not isinstance(self.terminal_tail_positions, int)
            or not 1 <= self.terminal_tail_positions <= 2_048
        ):
            raise ValueError("terminal_tail_positions must be between 1 and 2048")
        if (
            isinstance(self.max_raw_bytes, bool)
            or not isinstance(self.max_raw_bytes, int)
            or not 1 <= self.max_raw_bytes <= _MAX_RAW_BYTES
        ):
            raise ValueError(f"max_raw_bytes must be between 1 and {_MAX_RAW_BYTES}")
        if (
            isinstance(self.exporter_timeout_seconds, bool)
            or not isinstance(self.exporter_timeout_seconds, int)
            or not 1 <= self.exporter_timeout_seconds <= 3_600
        ):
            raise ValueError("exporter_timeout_seconds must be between 1 and 3600")

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic public configuration, including the reproducibility salt."""

        return {
            "schema": "phase3_normalization_config/v1",
            "datasetId": self.dataset_id,
            "maxGames": self.max_games,
            "maxPositions": self.max_positions,
            "terminalTailPositions": self.terminal_tail_positions,
            "maxRawBytes": self.max_raw_bytes,
            "exporterTimeoutSeconds": self.exporter_timeout_seconds,
            "split": self.split_policy.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Paths and observed counts from a successfully published dataset."""

    output_dir: Path
    games_path: Path
    positions_path: Path
    manifest_path: Path
    report_path: Path
    included_games: int
    included_positions: int
    excluded_games: int


@dataclass(frozen=True, slots=True)
class _CompletedRecord:
    schema_version: int
    event: str
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
    license_evidence: tuple[dict[str, str], ...]
    evidence_snapshots: tuple[dict[str, Any], ...]
    redistributable: bool
    machine_learning_allowed: bool


@dataclass(frozen=True, slots=True)
class _StagedCandidate:
    record: _CompletedRecord
    adapted: AdaptedAobaZeroCsa
    raw_csa: str
    black_rating: int | None
    white_rating: int | None
    date: str | None
    input_file: str


@dataclass(frozen=True, slots=True)
class _ExportedGame:
    input_file: str
    normalized_csa: str
    initial_sfen: str
    position_sfens: tuple[str, ...]
    usi_moves: tuple[str, ...]
    black_name: str | None
    white_name: str | None
    terminal_reason: str | None
    outcome: str
    result_validation: str


def normalize_aobazero_dataset(
    acquisition_manifest_path: Path,
    *,
    acquisition_root: Path,
    processed_root: Path,
    source: DataSource,
    config: NormalizationConfig,
    repository_root: Path | None = None,
    engine: ArtifactRef | None = None,
    engine_build_receipt: ArtifactRef | None = None,
    exporter: Exporter | None = None,
    adapter_limits: AobaZeroLimits = _DEFAULT_ADAPTER_LIMITS,
) -> NormalizationResult:
    """Normalize approved completed acquisitions into deterministic Phase 3 artifacts.

    Tests may inject ``exporter``. Production supplies an immutable engine and
    its exact build receipt, then invokes:

    ``<engine> export-csa-jsonl --input-dir <stage> --output <jsonl> --max-games <n>``
    """

    _validate_source(source)
    runtime_values = (repository_root, engine, engine_build_receipt)
    if exporter is None:
        if any(value is None for value in runtime_values):
            raise ValueError("production normalization requires immutable engine runtime evidence")
    elif any(value is not None for value in runtime_values):
        raise ValueError("an injected exporter must not carry production engine runtime evidence")
    if source.source_id != config.dataset_id and not config.dataset_id.startswith(
        f"{source.source_id}-"
    ):
        raise ValueError("dataset_id must equal or be namespaced by the approved source_id")

    final_dir = processed_root / config.dataset_id
    if os.path.lexists(final_dir):
        raise FileExistsError(f"refusing to overwrite existing dataset: {final_dir}")
    processed_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{config.dataset_id}.", dir=processed_root))
    stage_dir = temporary_dir / ".stage"
    export_path = temporary_dir / ".export.jsonl"
    stage_dir.mkdir()

    try:
        completed = sorted(
            _load_completed_records(acquisition_manifest_path),
            key=lambda record: (record.object_id, record.sha256),
        )
        _validate_completed_records(
            completed,
            source,
            acquisition_root=acquisition_root,
        )
        exclusions: list[dict[str, Any]] = []
        staged = _stage_candidates(
            completed,
            source=source,
            acquisition_root=acquisition_root,
            stage_dir=stage_dir,
            config=config,
            adapter_limits=adapter_limits,
            exclusions=exclusions,
        )
        if staged:
            if exporter is None:
                assert repository_root is not None
                assert engine is not None
                assert engine_build_receipt is not None
                _run_rust_exporter(
                    repository_root,
                    engine,
                    engine_build_receipt,
                    stage_dir,
                    export_path,
                    config.max_games,
                    config.exporter_timeout_seconds,
                )
            else:
                exporter(stage_dir, export_path, config.max_games)
            exports = _read_export_rows(export_path, {item.input_file for item in staged})
        else:
            exports = {}

        game_rows, position_rows = _assemble_rows(
            staged,
            exports,
            source=source,
            config=config,
            exclusions=exclusions,
        )
        if not game_rows:
            raise ValueError("normalization produced no valid games")
        games_digest = write_jsonl_gzip_atomic(
            temporary_dir / "games-00000.jsonl.gz",
            game_rows,
        )
        positions_digest = write_jsonl_gzip_atomic(
            temporary_dir / "positions-00000.jsonl.gz",
            position_rows,
        )
        report = _build_report(
            completed_count=len(completed),
            staged_count=len(staged),
            games=game_rows,
            positions=position_rows,
            exclusions=exclusions,
        )
        report_digest = write_json_atomic(
            temporary_dir / "normalization-report.json",
            report,
        )
        manifest = _build_dataset_manifest(
            source=source,
            config=config,
            games=game_rows,
            positions=position_rows,
            artifacts={
                "games-00000.jsonl.gz": games_digest,
                "positions-00000.jsonl.gz": positions_digest,
                "normalization-report.json": report_digest,
            },
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        shutil.rmtree(stage_dir)
        export_path.unlink(missing_ok=True)
        _rename_directory_without_overwrite(temporary_dir, final_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return NormalizationResult(
        output_dir=final_dir,
        games_path=final_dir / "games-00000.jsonl.gz",
        positions_path=final_dir / "positions-00000.jsonl.gz",
        manifest_path=final_dir / "manifest.json",
        report_path=final_dir / "normalization-report.json",
        included_games=len(game_rows),
        included_positions=len(position_rows),
        excluded_games=len(exclusions),
    )


def _validate_source(source: DataSource) -> None:
    if not source.enabled or not source.approved:
        raise ValueError(f"source {source.source_id!r} is not enabled and approved")
    if not source.machine_learning_allowed:
        raise ValueError(f"source {source.source_id!r} is not approved for machine learning")
    if source.adapter not in {"aobazero", "aobazero_csa"}:
        raise ValueError(f"source {source.source_id!r} does not select the AobaZero adapter")


def _load_completed_records(path: Path) -> list[_CompletedRecord]:
    # Import here so the storage boundary remains isolated and tests can build
    # synthetic manifests without importing downloader/network code.
    from open_shogi_training.data.manifest import ManifestStore

    if path.name == "manifest.jsonl":
        root = path.parent
        manifest_path = path
    elif path.is_dir():
        root = path
        manifest_path = path / "manifest.jsonl"
    else:
        raise ValueError("acquisition manifest must be manifest.jsonl or its directory")
    try:
        manifest_status = manifest_path.lstat()
    except OSError as error:
        raise ValueError(f"acquisition manifest is unavailable: {error}") from error
    if not stat.S_ISREG(manifest_status.st_mode) or stat.S_ISLNK(manifest_status.st_mode):
        raise ValueError("acquisition manifest must be a non-symlink regular file")
    store = ManifestStore(root)
    records = store.completed_records()
    if not records:
        raise ValueError("acquisition manifest contains no completed records")
    return [_completed_record_view(record) for record in records]


def _completed_record_view(record: object) -> _CompletedRecord:
    if hasattr(record, "__dataclass_fields__"):
        raw = asdict(record)
    elif isinstance(record, dict):
        raw = dict(record)
    else:
        raw = {field: getattr(record, field) for field in _CompletedRecord.__dataclass_fields__}
    raw.setdefault("schema_version", 1)
    raw.setdefault("event", "completed")
    evidence = raw.get("license_evidence")
    if isinstance(evidence, tuple):
        evidence = list(evidence)
    raw["license_evidence"] = tuple(dict(item) for item in evidence)
    snapshots = raw.get("evidence_snapshots")
    if isinstance(snapshots, tuple):
        snapshots = list(snapshots)
    raw["evidence_snapshots"] = tuple(dict(item) for item in snapshots)
    return _CompletedRecord(**raw)


def _validate_completed_records(
    records: list[_CompletedRecord],
    source: DataSource,
    *,
    acquisition_root: Path,
) -> None:
    evidence = tuple(item.as_dict() for item in source.license_evidence)
    verified_snapshots: dict[tuple[str, str, int], bytes] = {}
    catalog_by_url = {item.url: item for item in source.catalog}
    for record in records:
        if record.schema_version != 1 or record.event != "completed":
            raise ValueError("normalization accepts only completed acquisition manifest v1 records")
        if record.source_id != source.source_id:
            raise ValueError(f"manifest record {record.object_id!r} belongs to another source")
        source.validate_url(record.url, require_catalog_entry=True)
        catalog_item = catalog_by_url[record.url]
        if (
            record.object_id,
            record.original_filename,
            record.data_format,
            record.compression,
        ) != (
            catalog_item.object_id,
            catalog_item.filename,
            catalog_item.data_format,
            catalog_item.compression,
        ):
            raise ValueError(
                f"manifest record {record.object_id!r} differs from its exact catalog entry"
            )
        expected_object_path = (
            Path("objects") / "sha256" / record.sha256[:2] / record.sha256
        ).as_posix()
        if record.object_path != expected_object_path:
            raise ValueError(
                f"manifest record {record.object_id!r} is outside the content-addressed layout"
            )
        if record.license != source.license:
            raise ValueError(f"manifest record {record.object_id!r} has a license mismatch")
        if record.license_evidence != evidence:
            raise ValueError(f"manifest record {record.object_id!r} has license-evidence drift")
        if record.redistributable != source.redistributable:
            raise ValueError(
                f"manifest record {record.object_id!r} has redistribution-decision drift"
            )
        if record.machine_learning_allowed != source.machine_learning_allowed:
            raise ValueError(
                f"manifest record {record.object_id!r} has machine-learning-decision drift"
            )
        _validate_evidence_snapshots(
            record,
            source=source,
            acquisition_root=acquisition_root,
            verified=verified_snapshots,
        )
        if record.data_format.casefold() != "csa" or record.compression.casefold() not in {
            "none",
            "identity",
        }:
            raise ValueError(
                f"manifest record {record.object_id!r} is not an individual uncompressed CSA"
            )


def _validate_evidence_snapshots(
    record: _CompletedRecord,
    *,
    source: DataSource,
    acquisition_root: Path,
    verified: dict[tuple[str, str, int], bytes],
) -> None:
    if not record.evidence_snapshots:
        raise ValueError(f"manifest record {record.object_id!r} has no evidence snapshots")
    evidence_ids: set[str] = set()
    object_paths: set[str] = set()
    snapshot_urls: set[str] = set()
    snapshots_by_url: dict[str, bytes] = {}
    expected_keys = {
        "evidence_id",
        "url",
        "retrieved_at",
        "sha256",
        "size",
        "content_type",
        "object_path",
    }
    evidence_catalog = {item.evidence_id: item for item in source.evidence_catalog}
    if not evidence_catalog:
        raise ValueError(f"approved source {source.source_id!r} has no evidence catalog")
    for snapshot in record.evidence_snapshots:
        if set(snapshot) != expected_keys:
            raise ValueError(
                f"manifest record {record.object_id!r} has malformed evidence snapshot keys"
            )
        evidence_id = snapshot["evidence_id"]
        if (
            not isinstance(evidence_id, str)
            or _DATASET_ID_RE.fullmatch(evidence_id) is None
            or evidence_id in evidence_ids
        ):
            raise ValueError(
                f"manifest record {record.object_id!r} has invalid/duplicate evidence_id"
            )
        evidence_ids.add(evidence_id)
        catalog_item = evidence_catalog.get(evidence_id)
        if catalog_item is None:
            raise ValueError(
                f"manifest record {record.object_id!r} references unaudited evidence_id"
            )
        url = snapshot["url"]
        if not isinstance(url, str):
            raise ValueError("evidence snapshot URL must be a string")
        if url != catalog_item.url:
            raise ValueError("evidence snapshot URL has drifted from its audited catalog")
        snapshot_urls.add(url)
        sha256 = snapshot["sha256"]
        size = snapshot["size"]
        object_path = snapshot["object_path"]
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError("evidence snapshot SHA-256 is malformed")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("evidence snapshot size must be a positive integer")
        if size > catalog_item.max_bytes:
            raise ValueError("evidence snapshot exceeds its catalog-specific byte limit")
        if catalog_item.sha256 is not None and sha256 != catalog_item.sha256:
            raise ValueError("evidence snapshot differs from its audited SHA-256")
        if not isinstance(object_path, str) or object_path in object_paths:
            raise ValueError("evidence snapshot object_path is invalid or duplicated")
        object_paths.add(object_path)
        expected_object_path = (Path("evidence") / "sha256" / sha256[:2] / sha256).as_posix()
        if object_path != expected_object_path:
            raise ValueError("evidence snapshot is outside the content-addressed layout")
        identity = (object_path, sha256, size)
        raw = verified.get(identity)
        if raw is None:
            raw = _read_verified_acquisition_file(
                acquisition_root,
                object_path,
                maximum=catalog_item.max_bytes,
                expected_size=size,
                expected_sha256=sha256,
            )
            verified[identity] = raw
        snapshots_by_url[url] = raw
    if evidence_ids != evidence_catalog.keys():
        raise ValueError(
            f"manifest record {record.object_id!r} does not snapshot the complete evidence catalog"
        )
    license_urls = {item["url"] for item in record.license_evidence}
    if not license_urls.issubset(snapshot_urls):
        raise ValueError(
            f"manifest record {record.object_id!r} lacks a snapshot for license evidence"
        )
    for evidence in record.license_evidence:
        try:
            text = snapshots_by_url[evidence["url"]].decode("utf-8")
        except UnicodeError as error:
            raise ValueError("license evidence snapshot is not readable UTF-8") from error
        if evidence["quote"] not in text:
            raise ValueError("pinned license quotation is absent from its evidence snapshot")


def _stage_candidates(
    completed: list[_CompletedRecord],
    *,
    source: DataSource,
    acquisition_root: Path,
    stage_dir: Path,
    config: NormalizationConfig,
    adapter_limits: AobaZeroLimits,
    exclusions: list[dict[str, Any]],
) -> list[_StagedCandidate]:
    candidates: list[_StagedCandidate] = []
    seen_raw_hashes: set[str] = set()
    verified_raw_count = 0
    for record in completed:
        if verified_raw_count >= config.max_games:
            _exclude(exclusions, record, "game_cap_reached", "initial sample game cap reached")
            continue
        try:
            raw = _read_verified_raw(
                record,
                acquisition_root=acquisition_root,
                maximum=min(config.max_raw_bytes, source.max_object_bytes),
            )
            if record.sha256 in seen_raw_hashes:
                _exclude(exclusions, record, "duplicate_raw", "raw SHA-256 was already staged")
                continue
            seen_raw_hashes.add(record.sha256)
            verified_raw_count += 1
            adapted = adapt_aobazero_csa(raw, limits=adapter_limits)
            if adapted.raw_sha256 != record.sha256:
                raise ValueError("adapter and acquisition SHA-256 disagree")
            black_rating, white_rating = metadata_ratings(adapted.metadata)
            date = metadata_date(adapted.metadata)
        except AobaZeroAdaptationError as error:
            _exclude(exclusions, record, error.code, str(error))
            continue
        except (OSError, ValueError) as error:
            _exclude(exclusions, record, "corrupt_raw_object", str(error))
            continue
        input_file = f"{record.sha256}.csa"
        stage_path = stage_dir / input_file
        with stage_path.open("xb") as output:
            output.write(adapted.csa.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        candidates.append(
            _StagedCandidate(
                record=record,
                adapted=adapted,
                raw_csa=raw.decode("utf-8"),
                black_rating=black_rating,
                white_rating=white_rating,
                date=date,
                input_file=input_file,
            )
        )
    return candidates


def _read_verified_raw(
    record: _CompletedRecord,
    *,
    acquisition_root: Path,
    maximum: int,
) -> bytes:
    if _SHA256_RE.fullmatch(record.sha256) is None:
        raise ValueError("manifest SHA-256 is malformed")
    if isinstance(record.size, bool) or not isinstance(record.size, int) or record.size <= 0:
        raise ValueError("manifest size must be a positive integer")
    if record.size > maximum:
        raise ValueError(f"raw object exceeds the {maximum}-byte normalization limit")
    return _read_verified_acquisition_file(
        acquisition_root,
        record.object_path,
        maximum=maximum,
        expected_size=record.size,
        expected_sha256=record.sha256,
    )


def _read_verified_acquisition_file(
    acquisition_root: Path,
    object_path: str,
    *,
    maximum: int,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    relative = PurePosixPath(object_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or str(relative) != object_path
    ):
        raise ValueError("object_path must be a normalized root-relative POSIX path")
    try:
        root_status = acquisition_root.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect acquisition root: {error}") from error
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise ValueError("acquisition root must be a non-symlink directory")

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW

    try:
        current_descriptor = os.open(acquisition_root, directory_flags)
        try:
            for part in relative.parts[:-1]:
                next_descriptor = os.open(part, directory_flags, dir_fd=current_descriptor)
                os.close(current_descriptor)
                current_descriptor = next_descriptor
            file_descriptor = os.open(
                relative.parts[-1],
                file_flags,
                dir_fd=current_descriptor,
            )
        finally:
            os.close(current_descriptor)
    except OSError as error:
        raise ValueError(f"cannot open content-addressed acquisition object: {error}") from error

    try:
        file_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError("acquisition object must be a regular file")
        if file_status.st_size != expected_size:
            raise ValueError("acquisition object size does not match its manifest")
        if expected_size > maximum:
            raise ValueError(f"acquisition object exceeds the {maximum}-byte limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ValueError(f"acquisition object exceeds the {maximum}-byte limit")
        raw = b"".join(chunks)
    finally:
        os.close(file_descriptor)
    if len(raw) != expected_size:
        raise ValueError("acquisition object size changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("acquisition object SHA-256 does not match its manifest")
    return raw


def _run_rust_exporter(
    repository_root: Path,
    engine: ArtifactRef,
    engine_build_receipt: ArtifactRef,
    stage_dir: Path,
    output_path: Path,
    max_games: int,
    timeout_seconds: int,
) -> None:
    root = repository_root.resolve(strict=True)
    try:
        stage_relative = stage_dir.resolve(strict=True).relative_to(root).as_posix()
        output_relative = (
            output_path.parent.resolve(strict=True)
            .relative_to(root)
            .joinpath(output_path.name)
            .as_posix()
        )
    except ValueError as error:
        raise ValueError(
            "Rust exporter staging and output must remain inside the repository"
        ) from error
    stdout_path = f"{output_relative}.stdout.log"
    stderr_path = f"{output_relative}.stderr.log"
    outcome = CommandRunner(
        root,
        require_clean_repository=True,
        require_engine_build_receipt=True,
    ).run(
        {
            "kind": "engine_dataset_export",
            "argv": [
                engine.path,
                "export-csa-jsonl",
                "--input-dir",
                stage_relative,
                "--output",
                output_relative,
                "--max-games",
                str(max_games),
            ],
            "timeoutSeconds": timeout_seconds,
        },
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        expected_executable=engine,
        engine_build_receipt=engine_build_receipt,
        memory_limit_mib=1_024,
    )
    stdout_file = contained_path(root, outcome.stdout.path, must_exist=True)
    stderr_file = contained_path(root, outcome.stderr.path, must_exist=True)
    try:
        if (
            outcome.return_code != 0
            or outcome.timed_out
            or outcome.output_limit_exceeded
            or outcome.memory_limit_exceeded
        ):
            diagnostic = stderr_file.read_bytes()[-2_048:].decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Rust CSA exporter failed with exit {outcome.return_code}: {diagnostic}"
            )
        if stdout_file.stat().st_size != 0:
            raise RuntimeError("Rust CSA exporter wrote unexpected stdout")
    finally:
        stdout_file.unlink()
        stderr_file.unlink()


def _read_export_rows(path: Path, expected_files: set[str]) -> dict[str, _ExportedGame | str]:
    if not path.is_file():
        raise RuntimeError("Rust CSA exporter did not create its JSONL output")
    if path.stat().st_size > _MAX_EXPORT_BYTES:
        raise RuntimeError("Rust CSA exporter output exceeds 128 MiB")
    rows: dict[str, _ExportedGame | str] = {}
    with path.open("rb") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if len(raw_line) > _MAX_EXPORT_LINE_BYTES:
                raise RuntimeError(f"Rust exporter row {line_number} exceeds 32 MiB")
            if not raw_line.endswith(b"\n"):
                raise RuntimeError(f"Rust exporter row {line_number} is not LF terminated")
            try:
                row = json.loads(raw_line, object_pairs_hook=_unique_json_object)
            except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as error:
                raise RuntimeError(f"Rust exporter row {line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"Rust exporter row {line_number} must be an object")
            input_file_name, value = _parse_export_row(row, line_number)
            if input_file_name not in expected_files:
                raise RuntimeError(
                    f"Rust exporter returned unexpected inputFile {input_file_name!r}"
                )
            if input_file_name in rows:
                raise RuntimeError(f"Rust exporter duplicated inputFile {input_file_name!r}")
            rows[input_file_name] = value
    missing = sorted(expected_files - rows.keys())
    if missing:
        raise RuntimeError(f"Rust exporter omitted staged files: {missing}")
    return rows


def _parse_export_row(
    row: dict[str, Any],
    line_number: int,
) -> tuple[str, _ExportedGame | str]:
    if row.get("schema") != _EXPORT_SCHEMA:
        raise RuntimeError(f"Rust exporter row {line_number} has an unsupported schema")
    status = row.get("status")
    expected_keys = _OK_EXPORT_KEYS if status == "ok" else _REJECTED_EXPORT_KEYS
    if status not in {"ok", "rejected"} or frozenset(row) != expected_keys:
        raise RuntimeError(f"Rust exporter row {line_number} violates the closed schema")
    input_file = row["inputFile"]
    if (
        not isinstance(input_file, str)
        or PurePosixPath(input_file).name != input_file
        or not input_file.endswith(".csa")
    ):
        raise RuntimeError(f"Rust exporter row {line_number} has an unsafe inputFile")
    if status == "rejected":
        reason = row["reason"]
        if not isinstance(reason, str) or not reason or len(reason.encode("utf-8")) > 2_048:
            raise RuntimeError(f"Rust exporter row {line_number} has an invalid rejection reason")
        return input_file, reason

    for name in ("normalizedCsa", "initialSfen"):
        if not isinstance(row[name], str) or not row[name]:
            raise RuntimeError(f"Rust exporter row {line_number} has invalid {name}")
    if len(row["normalizedCsa"].encode("utf-8")) > 1_048_576:
        raise RuntimeError(f"Rust exporter row {line_number} normalizedCsa is oversized")
    if not isinstance(row["positionSfens"], list) or not row["positionSfens"]:
        raise RuntimeError(f"Rust exporter row {line_number} has invalid positionSfens")
    if not isinstance(row["usiMoves"], list):
        raise RuntimeError(f"Rust exporter row {line_number} has invalid usiMoves")
    if len(row["positionSfens"]) != len(row["usiMoves"]) + 1:
        raise RuntimeError(f"Rust exporter row {line_number} has inconsistent sequence lengths")
    if (
        any(not _valid_export_sfen(sfen) for sfen in row["positionSfens"])
        or row["positionSfens"][0] != row["initialSfen"]
    ):
        raise RuntimeError(f"Rust exporter row {line_number} has invalid SFEN sequence")
    if any(
        not isinstance(move, str) or _USI_MOVE_RE.fullmatch(move) is None
        for move in row["usiMoves"]
    ):
        raise RuntimeError(f"Rust exporter row {line_number} has invalid USI moves")
    for name in ("blackName", "whiteName", "terminalReason"):
        if row[name] is not None and (
            not isinstance(row[name], str) or not row[name] or "\n" in row[name]
        ):
            raise RuntimeError(f"Rust exporter row {line_number} has invalid {name}")
    if row["outcome"] not in _EXPORT_OUTCOMES:
        raise RuntimeError(f"Rust exporter row {line_number} has invalid outcome")
    if row["resultValidation"] not in _RESULT_VALIDATIONS:
        raise RuntimeError(f"Rust exporter row {line_number} has invalid resultValidation")
    if (row["terminalReason"] is None) != (row["resultValidation"] == "missing"):
        raise RuntimeError(
            f"Rust exporter row {line_number} has inconsistent terminal/result validation"
        )
    return input_file, _ExportedGame(
        input_file=input_file,
        normalized_csa=row["normalizedCsa"],
        initial_sfen=row["initialSfen"],
        position_sfens=tuple(row["positionSfens"]),
        usi_moves=tuple(row["usiMoves"]),
        black_name=row["blackName"],
        white_name=row["whiteName"],
        terminal_reason=row["terminalReason"],
        outcome=row["outcome"],
        result_validation=row["resultValidation"],
    )


def _valid_export_sfen(sfen: object) -> bool:
    if not isinstance(sfen, str) or not sfen or not sfen.isascii() or len(sfen) > 1_024:
        return False
    fields = sfen.split(" ")
    return (
        len(fields) == 4
        and all(fields)
        and fields[1] in {"b", "w"}
        and fields[3].isascii()
        and fields[3].isdecimal()
        and int(fields[3]) > 0
    )


def _assemble_rows(
    staged: list[_StagedCandidate],
    exports: dict[str, _ExportedGame | str],
    *,
    source: DataSource,
    config: NormalizationConfig,
    exclusions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    games: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    position_cap_reached = False
    for candidate in staged:
        exported = exports[candidate.input_file]
        if isinstance(exported, str):
            _exclude(exclusions, candidate.record, "rust_rejected", exported)
            continue
        contract_error = _export_contract_error(candidate, exported)
        if contract_error is not None:
            _exclude(exclusions, candidate.record, "export_contract", contract_error)
            continue
        canonical_sha256 = hashlib.sha256(exported.normalized_csa.encode("utf-8")).hexdigest()
        if canonical_sha256 in seen_canonical:
            _exclude(
                exclusions,
                candidate.record,
                "duplicate_canonical",
                "canonical CSA SHA-256 was already included",
            )
            continue
        seen_canonical.add(canonical_sha256)
        if (
            position_cap_reached
            or len(positions) + len(exported.position_sfens) > config.max_positions
        ):
            position_cap_reached = True
            _exclude(
                exclusions,
                candidate.record,
                "position_cap_reached",
                "game would cross the configured position cap",
            )
            continue

        split = assign_game_split(canonical_sha256, config.split_policy)
        game_id = canonical_sha256
        ply_count = len(exported.usi_moves)
        raw_reference = _raw_reference(candidate.record)
        game = {
            "schema": "phase3_game/v1",
            "gameId": game_id,
            "canonicalSha256": canonical_sha256,
            "rawObject": raw_reference,
            "rawCsa": candidate.raw_csa,
            "normalizedCsa": exported.normalized_csa,
            "initialSfen": exported.initial_sfen,
            "usiMoves": list(exported.usi_moves),
            "plyCount": ply_count,
            "positionCount": len(exported.position_sfens),
            "outcome": exported.outcome,
            "terminalReason": exported.terminal_reason,
            "resultValidation": exported.result_validation,
            "players": {
                "black": {"name": exported.black_name, "rating": candidate.black_rating},
                "white": {"name": exported.white_name, "rating": candidate.white_rating},
            },
            "date": candidate.date,
            "sourceDateTime": candidate.adapted.source_datetime,
            "sourceTimeZone": None,
            "split": split,
            "flags": {"short": ply_count < 20, "long": ply_count > 512},
            "sourceId": source.source_id,
            "url": candidate.record.url,
            "retrievedAt": candidate.record.retrieved_at,
            "licenseDecision": _license_decision(candidate.record),
        }
        games.append(game)

        # The terminal state has no label and is always ineligible.  In addition,
        # the final configured number of move-bearing positions are excluded.
        tail_start = max(0, ply_count - config.terminal_tail_positions)
        for position_index, sfen in enumerate(exported.position_sfens):
            move = (
                exported.usi_moves[position_index]
                if position_index < len(exported.usi_moves)
                else None
            )
            next_sfen = (
                exported.position_sfens[position_index + 1]
                if position_index + 1 < len(exported.position_sfens)
                else None
            )
            terminal_tail = position_index >= tail_start
            side = _sfen_side(sfen)
            positions.append(
                {
                    "schema": "phase3_position/v1",
                    "gameId": game_id,
                    "canonicalSha256": canonical_sha256,
                    "rawSha256": candidate.record.sha256,
                    "sourceId": source.source_id,
                    "split": split,
                    "positionIndex": position_index,
                    "sfen": sfen,
                    "moveUsi": move,
                    "nextSfen": next_sfen,
                    "outcome": exported.outcome,
                    "terminalReason": exported.terminal_reason,
                    "sideToMove": side,
                    "fullPlies": ply_count,
                    "remainingPlies": ply_count - position_index,
                    "eligible": move is not None and not terminal_tail,
                    "terminalTail": terminal_tail,
                }
            )
    return games, positions


def _export_contract_error(
    candidate: _StagedCandidate,
    exported: _ExportedGame,
) -> str | None:
    if exported.black_name != candidate.adapted.black_name:
        return "Rust blackName disagrees with the adapted record"
    if exported.white_name != candidate.adapted.white_name:
        return "Rust whiteName disagrees with the adapted record"
    if exported.terminal_reason != candidate.adapted.terminal.removeprefix("%"):
        return "Rust terminalReason disagrees with the adapted record"
    if len(exported.usi_moves) != candidate.adapted.move_count:
        return "Rust move count disagrees with the adapted record"
    if not exported.normalized_csa.startswith("'CSA encoding=UTF-8\nV3.0\n"):
        return "Rust normalizedCsa is not canonical explicit UTF-8 CSA V3.0"
    return None


def _raw_reference(record: _CompletedRecord) -> dict[str, Any]:
    return {
        "objectId": record.object_id,
        "objectPath": record.object_path,
        "originalFilename": record.original_filename,
        "sha256": record.sha256,
        "size": record.size,
        "response": {
            "contentType": record.content_type,
            "etag": record.etag,
            "lastModified": record.last_modified,
        },
    }


def _license_decision(record: _CompletedRecord) -> dict[str, Any]:
    return {
        "license": record.license,
        "evidence": list(record.license_evidence),
        "evidenceSnapshots": list(record.evidence_snapshots),
        "redistributable": record.redistributable,
        "machineLearningAllowed": record.machine_learning_allowed,
    }


def _sfen_side(sfen: str) -> str:
    fields = sfen.split(" ")
    if len(fields) != 4:
        raise RuntimeError("Rust exporter emitted SFEN without four fields")
    if fields[1] == "b":
        return "black"
    if fields[1] == "w":
        return "white"
    raise RuntimeError("Rust exporter emitted an invalid SFEN side-to-move field")


def _build_report(
    *,
    completed_count: int,
    staged_count: int,
    games: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    per_game_states: dict[str, Counter[str]] = defaultdict(Counter)
    global_states: Counter[str] = Counter()
    state_game_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for position in positions:
        key = _position_state(position["sfen"])
        game_id = position["gameId"]
        per_game_states[game_id][key] += 1
        global_states[key] += 1
        state_game_counts[key][game_id] += 1

    within_states = 0
    within_occurrences = 0
    for counts in per_game_states.values():
        within_states += sum(count > 1 for count in counts.values())
        within_occurrences += sum(count - 1 for count in counts.values() if count > 1)
    cross_states = sum(len(counts) > 1 for counts in state_game_counts.values())
    cross_occurrences = sum(
        sum(counts.values()) - max(counts.values())
        for counts in state_game_counts.values()
        if len(counts) > 1
    )
    split_counts = Counter(game["split"] for game in games)
    exclusion_counts = Counter(exclusion["reason"] for exclusion in exclusions)
    return {
        "schema": "phase3_normalization_report/v1",
        "input": {
            "completedObjects": completed_count,
            "stagedObjects": staged_count,
        },
        "games": {
            "included": len(games),
            "excluded": len(exclusions),
            "uniqueCanonical": len({game["canonicalSha256"] for game in games}),
            "short": sum(game["flags"]["short"] for game in games),
            "long": sum(game["flags"]["long"] for game in games),
            "splits": {
                "train": split_counts["train"],
                "validation": split_counts["validation"],
                "test": split_counts["test"],
            },
        },
        "positions": {
            "identity": "canonical SFEN board, side, and hands; move number omitted",
            "total": len(positions),
            "unique": len(global_states),
            "withinGameDuplicateStates": within_states,
            "withinGameDuplicateOccurrences": within_occurrences,
            "crossGameDuplicateStates": cross_states,
            "crossGameDuplicateOccurrences": cross_occurrences,
            "eligible": sum(position["eligible"] for position in positions),
            "duplicateDefinitions": {
                "withinGameDuplicateOccurrences": (
                    "occurrences after the first matching state in each game"
                ),
                "crossGameDuplicateOccurrences": (
                    "occurrences outside the single game with the most matches for each state"
                ),
            },
        },
        "exclusionCounts": dict(sorted(exclusion_counts.items())),
        "exclusions": exclusions,
    }


def _build_dataset_manifest(
    *,
    source: DataSource,
    config: NormalizationConfig,
    games: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    artifacts: dict[str, ArtifactDigest],
) -> dict[str, Any]:
    raw_hashes = sorted(game["rawObject"]["sha256"] for game in games)
    canonical_hashes = sorted(game["canonicalSha256"] for game in games)
    evidence_snapshots: dict[tuple[object, ...], dict[str, Any]] = {}
    for game in games:
        for snapshot in game["licenseDecision"]["evidenceSnapshots"]:
            identity = (
                snapshot["evidence_id"],
                snapshot["url"],
                snapshot["retrieved_at"],
                snapshot["sha256"],
                snapshot["size"],
                snapshot["content_type"],
                snapshot["object_path"],
            )
            previous = evidence_snapshots.setdefault(identity, snapshot)
            if previous != snapshot:
                raise RuntimeError("conflicting evidence snapshot identity in normalized games")
    return {
        "schema": "phase3_dataset_manifest/v1",
        "datasetId": config.dataset_id,
        "source": {
            "sourceId": source.source_id,
            "name": source.name,
            "officialBase": source.official_base,
            "adapter": source.adapter,
            "license": source.license,
            "licenseEvidence": [item.as_dict() for item in source.license_evidence],
            "redistributable": source.redistributable,
            "machineLearningAllowed": source.machine_learning_allowed,
            "lastReviewed": source.last_reviewed.isoformat(),
        },
        "config": config.as_dict(),
        "counts": {"games": len(games), "positions": len(positions)},
        "rawObjectSha256": raw_hashes,
        "canonicalGameSha256": canonical_hashes,
        "evidenceSnapshots": [evidence_snapshots[key] for key in sorted(evidence_snapshots)],
        "artifacts": {
            name: {
                "sha256": digest.sha256,
                "size": digest.size,
                "records": digest.records,
            }
            for name, digest in sorted(artifacts.items())
        },
    }


def _position_state(sfen: str) -> str:
    fields = sfen.split(" ")
    if len(fields) != 4:
        raise RuntimeError("normalized SFEN does not contain four fields")
    return " ".join(fields[:3])


def _exclude(
    exclusions: list[dict[str, Any]],
    record: _CompletedRecord,
    reason: str,
    detail: str,
) -> None:
    exclusions.append(
        {
            "objectId": record.object_id,
            "rawSha256": record.sha256,
            "reason": reason,
            "detail": detail[:2_048],
        }
    )


def _rename_directory_without_overwrite(source: Path, destination: Path) -> None:
    """Atomically publish a directory with the platform's no-replace primitive."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, _RENAME_EXCL)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    else:
        raise RuntimeError("atomic no-replace directory publication is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to overwrite existing dataset: {destination}") from None
    raise OSError(error_number, os.strerror(error_number), destination)


class _DuplicateJsonKeyError(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result
