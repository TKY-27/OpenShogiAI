"""Resumable append-only teacher labeling with atomic progress manifests."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    append_jsonl_record,
    iter_jsonl_records,
    jsonl_digest,
    jsonl_prefix_digest,
    load_json_object,
    publish_regular_at,
    read_regular_bytes,
    retire_bound_regular,
    stable_directory_lock,
    stable_parent_descriptor,
    stable_regular_descriptor,
    write_json_atomic,
)
from open_shogi_training.labeling.benchmark import (
    BenchmarkError,
    verify_benchmark_report,
)
from open_shogi_training.labeling.config import TeacherConfig
from open_shogi_training.labeling.fingerprint import (
    TeacherFingerprint,
    TeacherFingerprintError,
    fingerprint_teacher,
)
from open_shogi_training.labeling.legality import (
    LegalityCoverage,
    LegalityValidationError,
    LegalityValidatorIdentity,
    RustLegalityValidator,
)
from open_shogi_training.labeling.schema import (
    LABEL_SCHEMA,
    MAX_LABEL_ARTIFACT_BYTES,
    MAX_LABEL_LINE_BYTES,
    PARSER_VERSION,
    QUARANTINE_SCHEMA,
    SCORE_POV,
    iter_teacher_labels,
    validate_label_record,
)
from open_shogi_training.labeling.selection import (
    SelectedPosition,
    SelectionError,
    SelectionResult,
    select_positions,
)
from open_shogi_training.labeling.usi import USIEngine, USIError, USIRetryError

LABEL_MANIFEST_SCHEMA: Final = "phase4_teacher_label_manifest/v2"
LEGACY_LABEL_MANIFEST_SCHEMA: Final = "phase4_teacher_label_manifest/v1"
LABEL_MANIFEST_MIGRATION_SCHEMA: Final = "phase4_teacher_label_manifest_migration/v1"
LEGACY_MANIFEST_EVIDENCE_NAME: Final = "manifest.v1.evidence.json"
MAX_QUARANTINE_BYTES: Final = 512 * 1024 * 1024
MAX_QUARANTINE_LINE_BYTES: Final = 128 * 1024
EngineFactory = Callable[[TeacherConfig, Path], USIEngine]


class LabelingError(RuntimeError):
    """Raised when a labeling run cannot safely continue or resume."""


@dataclass(frozen=True, slots=True)
class LabelingResult:
    output_dir: Path
    labels_path: Path
    quarantine_path: Path
    manifest_path: Path
    selected: int
    completed: int
    quarantined: int
    pending: int
    status: str


@dataclass(frozen=True, slots=True)
class LabelAuditResult:
    output_dir: Path
    manifest_path: Path
    manifest_schema: str
    selected: int
    completed: int
    quarantined: int
    pending: int
    status: str
    selection_sha256: str
    benchmark_sha256: str
    labels_sha256: str
    legality_validator: LegalityValidatorIdentity
    candidate_coverage: dict[str, object]


@dataclass(frozen=True, slots=True)
class LabelMigrationResult:
    audit: LabelAuditResult
    legacy_manifest_sha256: str
    legacy_manifest_path: Path


@dataclass(slots=True)
class _Progress:
    completed: dict[str, dict[str, Any]]
    quarantined: dict[str, dict[str, Any]]
    legality_coverage: dict[str, LegalityCoverage]


class _OutputLock:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.context: Any | None = None

    def __enter__(self) -> _OutputLock:
        try:
            context = stable_directory_lock(
                self.output_dir,
                create=True,
                exclusive=True,
                nonblocking=True,
            )
            context.__enter__()
            self.context = context
        except BlockingIOError as error:
            self._close()
            raise LabelingError("another labeling process holds the output lock") from error
        except (ArtifactError, LabelingError, OSError) as error:
            self._close()
            if isinstance(error, LabelingError):
                raise
            raise LabelingError(f"cannot lock labeling output: {error}") from error
        return self

    def __exit__(self, *_: object) -> None:
        self._close()

    def _close(self) -> None:
        if self.context is not None:
            context = self.context
            self.context = None
            try:
                context.__exit__(None, None, None)
            except ArtifactError as error:
                raise LabelingError("label output lock ancestor changed while held") from error


class _ExistingOutputLock:
    """Share-lock an existing label directory without creating or changing evidence."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.context: Any | None = None

    def __enter__(self) -> _ExistingOutputLock:
        try:
            context = stable_directory_lock(
                self.output_dir,
                create=False,
                exclusive=False,
                nonblocking=True,
            )
            context.__enter__()
            self.context = context
        except BlockingIOError as error:
            self._close()
            raise LabelingError("a labeling process holds the output lock") from error
        except (ArtifactError, LabelingError, OSError) as error:
            self._close()
            if isinstance(error, LabelingError):
                raise
            raise LabelingError(f"cannot share-lock labeling output: {error}") from error
        return self

    def __exit__(self, *_: object) -> None:
        self._close()

    def _close(self) -> None:
        if self.context is not None:
            context = self.context
            self.context = None
            try:
                context.__exit__(None, None, None)
            except ArtifactError as error:
                raise LabelingError("label audit lock ancestor changed while held") from error


def run_labeling(
    *,
    config: TeacherConfig,
    project_root: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    benchmark_report_path: Path,
    output_dir: Path,
    target_completed: int,
    engine_factory: EngineFactory = USIEngine,
) -> LabelingResult:
    """Resume or run one labeling selection; quarantine exhausted engine failures."""

    try:
        selection = select_positions(positions_path, dataset_manifest_path, config.selection)
        fingerprint = fingerprint_teacher(config, project_root)
        benchmark_report, benchmark_digest = verify_benchmark_report(
            benchmark_report_path,
            config=config,
            selection=selection,
            fingerprint=fingerprint,
        )
    except (SelectionError, TeacherFingerprintError, BenchmarkError) as error:
        raise LabelingError(str(error)) from error

    labels_path = output_dir / "labels.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    manifest_path = output_dir / "manifest.json"
    engine: USIEngine | None = None
    if (
        isinstance(target_completed, bool)
        or not isinstance(target_completed, int)
        or not 1 <= target_completed <= len(selection.positions)
    ):
        raise LabelingError(
            f"target_completed must be between 1 and selected count {len(selection.positions)}"
        )
    reported_identity = _benchmark_reported_identity(benchmark_report)
    with _OutputLock(output_dir):
        _ensure_append_file(labels_path)
        _ensure_append_file(quarantine_path)
        validator = RustLegalityValidator(project_root)
        validator_identity: LegalityValidatorIdentity | None = None
        progress = _Progress({}, {}, {})
        changed_since_manifest = 0
        active_error: BaseException | None = None
        can_write_manifest = False
        try:
            try:
                validator_identity = validator.start()
            except LegalityValidationError as error:
                raise LabelingError(str(error)) from error
            existing_manifest = _validate_existing_manifest(
                manifest_path,
                labels_path=labels_path,
                quarantine_path=quarantine_path,
                selection=selection,
                config=config,
                fingerprint=fingerprint,
                benchmark_path=benchmark_report_path,
                benchmark_sha256=benchmark_digest.sha256,
                reported_identity=reported_identity,
                validator_identity=validator_identity,
            )
            if (
                existing_manifest is not None
                and existing_manifest.get("schema") == LEGACY_LABEL_MANIFEST_SCHEMA
            ):
                raise LabelingError(
                    "legacy v1 label manifests require the explicit "
                    "migrate-label-manifest-v2 audit command"
                )
            if existing_manifest is None:
                if _regular_file_size(labels_path) or _regular_file_size(quarantine_path):
                    raise LabelingError(
                        "refusing to adopt label artifacts without a pre-existing "
                        "selection/benchmark manifest checkpoint"
                    )
                _write_manifest(
                    manifest_path,
                    labels_path,
                    quarantine_path,
                    selection=selection,
                    config=config,
                    fingerprint=fingerprint,
                    benchmark_path=benchmark_report_path,
                    benchmark_sha256=benchmark_digest.sha256,
                    progress=progress,
                    reported_identity=reported_identity,
                    validator_identity=validator_identity,
                    target_completed=target_completed,
                )
                can_write_manifest = True
            progress = _load_progress(
                labels_path,
                quarantine_path,
                selection=selection,
                config=config,
                fingerprint=fingerprint,
                reported_identity=reported_identity,
                validator=validator,
            )
            _validate_manifest_progress(
                existing_manifest,
                progress,
                selection,
                config=config,
            )
            can_write_manifest = True
            if len(progress.quarantined) >= config.labeling.max_quarantined and len(
                progress.completed
            ) + len(progress.quarantined) < len(selection.positions):
                raise LabelingError("existing quarantine count reached the configured safety limit")
            for position in selection.positions:
                if len(progress.completed) >= target_completed:
                    break
                if (
                    position.position_id in progress.completed
                    or position.position_id in progress.quarantined
                ):
                    continue
                if engine is None:
                    engine = engine_factory(config, project_root)
                    identity = engine.start()
                    active_identity = {"name": identity.name, "author": identity.author}
                    if active_identity != reported_identity:
                        raise LabelingError(
                            "active teacher USI identity differs from the authorized benchmark"
                        )
                try:
                    search = engine.analyze_with_retry(position.canonical_sfen)
                except USIError as error:
                    quarantine = _quarantine_record(
                        position,
                        error,
                        selection=selection,
                        config=config,
                        fingerprint=fingerprint,
                    )
                    append_jsonl_record(
                        quarantine_path,
                        quarantine,
                        max_line_bytes=MAX_QUARANTINE_LINE_BYTES,
                    )
                    progress.quarantined[position.position_id] = quarantine
                    changed_since_manifest += 1
                    if len(progress.quarantined) >= config.labeling.max_quarantined:
                        raise LabelingError(
                            "quarantine count reached the configured safety limit"
                        ) from error
                else:
                    if engine.identity is None:
                        raise LabelingError("teacher identity disappeared after a completed search")
                    if {
                        "name": engine.identity.name,
                        "author": engine.identity.author,
                    } != reported_identity:
                        raise LabelingError(
                            "teacher USI identity changed after an authorized retry"
                        )
                    label = _label_record(
                        position,
                        search,
                        teacher=fingerprint.teacher_record(engine.identity),
                        selection=selection,
                        config=config,
                    )
                    try:
                        validate_label_record(label)
                    except ValueError as error:
                        raise LabelingError(
                            f"generated label failed its own schema: {error}"
                        ) from error
                    try:
                        coverage = validator.validate(
                            position.canonical_sfen,
                            search.bestmove,
                            [list(candidate.pv) for candidate in search.candidates],
                            configured_multipv=config.multipv,
                        )
                    except LegalityValidationError as error:
                        raise LabelingError(
                            f"teacher moves failed OpenShogiAI legality replay for "
                            f"{position.position_id}: {error}"
                        ) from error
                    append_jsonl_record(
                        labels_path,
                        label,
                        max_line_bytes=MAX_LABEL_LINE_BYTES,
                    )
                    progress.completed[position.position_id] = label
                    progress.legality_coverage[position.position_id] = coverage
                    changed_since_manifest += 1
                if changed_since_manifest >= config.labeling.manifest_interval:
                    _write_manifest(
                        manifest_path,
                        labels_path,
                        quarantine_path,
                        selection=selection,
                        config=config,
                        fingerprint=fingerprint,
                        benchmark_path=benchmark_report_path,
                        benchmark_sha256=benchmark_digest.sha256,
                        progress=progress,
                        reported_identity=reported_identity,
                        validator_identity=validator_identity,
                        target_completed=target_completed,
                    )
                    changed_since_manifest = 0
        except BaseException as error:
            active_error = error
            raise
        finally:
            if engine is not None:
                engine.close()
            try:
                if can_write_manifest and validator_identity is not None:
                    try:
                        validator.assert_binary_unchanged()
                        _write_manifest(
                            manifest_path,
                            labels_path,
                            quarantine_path,
                            selection=selection,
                            config=config,
                            fingerprint=fingerprint,
                            benchmark_path=benchmark_report_path,
                            benchmark_sha256=benchmark_digest.sha256,
                            progress=progress,
                            reported_identity=reported_identity,
                            validator_identity=validator_identity,
                            target_completed=target_completed,
                        )
                    except BaseException:
                        if active_error is None:
                            raise
            finally:
                validator.close()

        completed = len(progress.completed)
        quarantined = len(progress.quarantined)
        pending = len(selection.positions) - completed - quarantined
        status = _status(
            len(selection.positions), completed, quarantined, target_completed=target_completed
        )
        return LabelingResult(
            output_dir=output_dir,
            labels_path=labels_path,
            quarantine_path=quarantine_path,
            manifest_path=manifest_path,
            selected=len(selection.positions),
            completed=completed,
            quarantined=quarantined,
            pending=pending,
            status=status,
        )


def audit_labeling_output(
    *,
    config: TeacherConfig,
    project_root: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    benchmark_report_path: Path,
    output_dir: Path,
) -> LabelAuditResult:
    """Read-only audit of complete label bytes, provenance, and Rust legality replay."""

    try:
        selection = select_positions(positions_path, dataset_manifest_path, config.selection)
        fingerprint = fingerprint_teacher(config, project_root)
        benchmark_report, benchmark_digest = verify_benchmark_report(
            benchmark_report_path,
            config=config,
            selection=selection,
            fingerprint=fingerprint,
        )
    except (SelectionError, TeacherFingerprintError, BenchmarkError) as error:
        raise LabelingError(str(error)) from error

    labels_path = output_dir / "labels.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    manifest_path = output_dir / "manifest.json"
    reported_identity = _benchmark_reported_identity(benchmark_report)
    validator = RustLegalityValidator(project_root)
    with _ExistingOutputLock(output_dir):
        try:
            validator_identity = validator.start()
            manifest = _validate_existing_manifest(
                manifest_path,
                labels_path=labels_path,
                quarantine_path=quarantine_path,
                selection=selection,
                config=config,
                fingerprint=fingerprint,
                benchmark_path=benchmark_report_path,
                benchmark_sha256=benchmark_digest.sha256,
                reported_identity=reported_identity,
                validator_identity=validator_identity,
            )
            if manifest is None:
                raise LabelingError("label audit requires an existing manifest")
            _validate_exact_artifact_snapshot(
                manifest,
                labels_path=labels_path,
                quarantine_path=quarantine_path,
            )
            progress = _load_progress(
                labels_path,
                quarantine_path,
                selection=selection,
                config=config,
                fingerprint=fingerprint,
                reported_identity=reported_identity,
                validator=validator,
            )
            _validate_manifest_progress(
                manifest,
                progress,
                selection,
                config=config,
                exact=True,
            )
            validator.assert_binary_unchanged()
            _validate_exact_artifact_snapshot(
                manifest,
                labels_path=labels_path,
                quarantine_path=quarantine_path,
            )
        except LegalityValidationError as error:
            raise LabelingError(str(error)) from error
        finally:
            validator.close()

    progress_record = manifest["progress"]
    labels_record = manifest["artifacts"][labels_path.name]
    return LabelAuditResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest_schema=manifest["schema"],
        selected=len(selection.positions),
        completed=len(progress.completed),
        quarantined=len(progress.quarantined),
        pending=len(selection.positions) - len(progress.completed) - len(progress.quarantined),
        status=progress_record["status"],
        selection_sha256=(
            selection.legacy_selection_sha256
            if manifest["schema"] == LEGACY_LABEL_MANIFEST_SCHEMA
            else selection.selection_sha256
        ),
        benchmark_sha256=benchmark_digest.sha256,
        labels_sha256=labels_record["sha256"],
        legality_validator=validator_identity,
        candidate_coverage=_candidate_coverage(progress, config),
    )


def migrate_label_manifest_v2(
    *,
    config: TeacherConfig,
    project_root: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    benchmark_report_path: Path,
    output_dir: Path,
    expected_legacy_manifest_sha256: str,
) -> LabelMigrationResult:
    """Audit legacy evidence without a teacher call, then atomically publish manifest v2."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_legacy_manifest_sha256) is None:
        raise LabelingError("expected legacy manifest SHA-256 must be lowercase hexadecimal")
    try:
        selection = select_positions(positions_path, dataset_manifest_path, config.selection)
        fingerprint = fingerprint_teacher(config, project_root)
        benchmark_report, benchmark_digest = verify_benchmark_report(
            benchmark_report_path,
            config=config,
            selection=selection,
            fingerprint=fingerprint,
        )
    except (SelectionError, TeacherFingerprintError, BenchmarkError) as error:
        raise LabelingError(str(error)) from error

    labels_path = output_dir / "labels.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    manifest_path = output_dir / "manifest.json"
    evidence_path = output_dir / LEGACY_MANIFEST_EVIDENCE_NAME
    reported_identity = _benchmark_reported_identity(benchmark_report)
    validator = RustLegalityValidator(project_root)
    with _OutputLock(output_dir):
        try:
            validator_identity = validator.start()
            current, current_digest = load_json_object(manifest_path, max_bytes=4 * 1024 * 1024)
            if current.get("schema") == LABEL_MANIFEST_SCHEMA:
                migration = _validate_migration_record(
                    current.get("migration"), output_dir=output_dir
                )
                if migration["legacy_manifest"]["sha256"] != expected_legacy_manifest_sha256:
                    raise LabelingError("existing v2 manifest migrated from a different v1 digest")
            else:
                if current.get("schema") != LEGACY_LABEL_MANIFEST_SCHEMA:
                    raise LabelingError("migration requires a legacy v1 or migrated v2 manifest")
                if current_digest.sha256 != expected_legacy_manifest_sha256:
                    raise LabelingError("legacy manifest SHA-256 differs from the required digest")
                manifest = _validate_existing_manifest(
                    manifest_path,
                    labels_path=labels_path,
                    quarantine_path=quarantine_path,
                    selection=selection,
                    config=config,
                    fingerprint=fingerprint,
                    benchmark_path=benchmark_report_path,
                    benchmark_sha256=benchmark_digest.sha256,
                    reported_identity=reported_identity,
                    validator_identity=validator_identity,
                )
                if manifest is None:
                    raise LabelingError("migration requires an existing v1 manifest")
                _validate_exact_artifact_snapshot(
                    manifest,
                    labels_path=labels_path,
                    quarantine_path=quarantine_path,
                )
                progress = _load_progress(
                    labels_path,
                    quarantine_path,
                    selection=selection,
                    config=config,
                    fingerprint=fingerprint,
                    reported_identity=reported_identity,
                    validator=validator,
                )
                _validate_manifest_progress(
                    manifest, progress, selection, config=config, exact=True
                )
                if (
                    len(selection.positions) != 10_000
                    or len(progress.completed) != 10_000
                    or progress.quarantined
                    or manifest["progress"]["status"] != "complete"
                ):
                    raise LabelingError(
                        "v2 migration requires the exact complete 10000-label set "
                        "with no quarantine"
                    )
                legacy_bytes, legacy_digest = read_regular_bytes(
                    manifest_path, max_bytes=4 * 1024 * 1024
                )
                if legacy_digest.sha256 != expected_legacy_manifest_sha256:
                    raise LabelingError("legacy manifest changed during migration audit")
                _publish_immutable_evidence(evidence_path, legacy_bytes)
                migration = {
                    "schema": LABEL_MANIFEST_MIGRATION_SCHEMA,
                    "legacy_manifest": {
                        "path": LEGACY_MANIFEST_EVIDENCE_NAME,
                        "sha256": legacy_digest.sha256,
                        "size": legacy_digest.size,
                    },
                    "legacy_schema": LEGACY_LABEL_MANIFEST_SCHEMA,
                    "legacy_updated_at": manifest["updated_at"],
                }
                validator.assert_binary_unchanged()
                _validate_exact_artifact_snapshot(
                    manifest,
                    labels_path=labels_path,
                    quarantine_path=quarantine_path,
                )
                _write_manifest(
                    manifest_path,
                    labels_path,
                    quarantine_path,
                    selection=selection,
                    config=config,
                    fingerprint=fingerprint,
                    benchmark_path=benchmark_report_path,
                    benchmark_sha256=benchmark_digest.sha256,
                    progress=progress,
                    reported_identity=reported_identity,
                    validator_identity=validator_identity,
                    target_completed=10_000,
                    migration=migration,
                )

            migrated = _validate_existing_manifest(
                manifest_path,
                labels_path=labels_path,
                quarantine_path=quarantine_path,
                selection=selection,
                config=config,
                fingerprint=fingerprint,
                benchmark_path=benchmark_report_path,
                benchmark_sha256=benchmark_digest.sha256,
                reported_identity=reported_identity,
                validator_identity=validator_identity,
            )
            if migrated is None or migrated.get("schema") != LABEL_MANIFEST_SCHEMA:
                raise LabelingError("v2 manifest publication could not be verified")
            progress = _load_progress(
                labels_path,
                quarantine_path,
                selection=selection,
                config=config,
                fingerprint=fingerprint,
                reported_identity=reported_identity,
                validator=validator,
            )
            _validate_manifest_progress(migrated, progress, selection, config=config, exact=True)
            validator.assert_binary_unchanged()
            _validate_exact_artifact_snapshot(
                migrated,
                labels_path=labels_path,
                quarantine_path=quarantine_path,
            )
        except (ArtifactError, LegalityValidationError) as error:
            raise LabelingError(str(error)) from error
        finally:
            validator.close()

    labels_record = migrated["artifacts"]["labels.jsonl"]
    audit = LabelAuditResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest_schema=LABEL_MANIFEST_SCHEMA,
        selected=len(selection.positions),
        completed=len(progress.completed),
        quarantined=len(progress.quarantined),
        pending=len(selection.positions) - len(progress.completed) - len(progress.quarantined),
        status=migrated["progress"]["status"],
        selection_sha256=selection.selection_sha256,
        benchmark_sha256=benchmark_digest.sha256,
        labels_sha256=labels_record["sha256"],
        legality_validator=validator_identity,
        candidate_coverage=_candidate_coverage(progress, config),
    )
    return LabelMigrationResult(
        audit=audit,
        legacy_manifest_sha256=expected_legacy_manifest_sha256,
        legacy_manifest_path=evidence_path,
    )


def _label_record(
    position: SelectedPosition,
    search: Any,
    *,
    teacher: dict[str, object],
    selection: SelectionResult,
    config: TeacherConfig,
) -> dict[str, object]:
    primary = search.primary
    return {
        "schema": LABEL_SCHEMA,
        "position_id": position.position_id,
        "canonical_sfen": position.canonical_sfen,
        "canonical_state_sha256": position.canonical_state_sha256,
        "side_to_move": position.side_to_move,
        "score": primary.score.as_dict(),
        "score_pov": SCORE_POV,
        "bestmove": search.bestmove,
        "candidates": [candidate.as_dict() for candidate in search.candidates],
        "depth": primary.depth,
        "seldepth": primary.seldepth,
        "nodes": primary.nodes,
        "elapsed_ms": search.elapsed_ms,
        "teacher": teacher,
        "dataset_manifest_sha256": selection.dataset_manifest_sha256,
        "config_sha256": config.sha256,
        "parser_version": PARSER_VERSION,
        "created_at": _utc_now(),
        "split": position.split,
        "game_id": position.game_id,
        "position_index": position.position_index,
        "stage": position.stage,
        "source_id": position.source_id,
        "outcome": position.outcome,
    }


def _quarantine_record(
    position: SelectedPosition,
    error: USIError,
    *,
    selection: SelectionResult,
    config: TeacherConfig,
    fingerprint: TeacherFingerprint,
) -> dict[str, object]:
    attempts = error.attempts if isinstance(error, USIRetryError) else 1
    category = error.last_category if isinstance(error, USIRetryError) else error.category
    return {
        "schema": QUARANTINE_SCHEMA,
        "position_id": position.position_id,
        "canonical_sfen": position.canonical_sfen,
        "canonical_state_sha256": position.canonical_state_sha256,
        "split": position.split,
        "game_id": position.game_id,
        "position_index": position.position_index,
        "stage": position.stage,
        "source_id": position.source_id,
        "outcome": position.outcome,
        "dataset_manifest_sha256": selection.dataset_manifest_sha256,
        "config_sha256": config.sha256,
        "teacher": fingerprint.identity_record(),
        "error_category": category,
        "error_message": str(error)[:2_048],
        "attempts": attempts,
        "stderr_tail": error.stderr_tail[-config.protocol_limits.max_stderr_bytes :],
        "created_at": _utc_now(),
    }


def _load_progress(
    labels_path: Path,
    quarantine_path: Path,
    *,
    selection: SelectionResult,
    config: TeacherConfig,
    fingerprint: TeacherFingerprint,
    reported_identity: dict[str, str | None],
    validator: RustLegalityValidator,
) -> _Progress:
    selected_by_id = {position.position_id: position for position in selection.positions}
    selected = set(selected_by_id)
    completed: dict[str, dict[str, Any]] = {}
    legality_coverage: dict[str, LegalityCoverage] = {}
    try:
        for label in iter_teacher_labels(
            labels_path,
            expected_dataset_manifest_sha256=selection.dataset_manifest_sha256,
            expected_config_sha256=config.sha256,
        ):
            identity = label["position_id"]
            if identity not in selected:
                raise LabelingError(
                    f"label position_id is outside the current selection: {identity}"
                )
            position = selected_by_id[identity]
            _verify_label_position(label, position)
            _verify_label_teacher(label["teacher"], fingerprint, reported_identity)
            try:
                coverage = validator.validate(
                    position.canonical_sfen,
                    label["bestmove"],
                    [list(candidate["pv"]) for candidate in label["candidates"]],
                    configured_multipv=config.multipv,
                )
            except LegalityValidationError as error:
                raise LabelingError(
                    f"existing label failed OpenShogiAI legality replay for {identity}: {error}"
                ) from error
            completed[identity] = label
            legality_coverage[identity] = coverage
    except (ArtifactError, ValueError) as error:
        raise LabelingError(f"cannot resume labels: {error}") from error

    quarantined: dict[str, dict[str, Any]] = {}
    try:
        for line_number, record in enumerate(
            iter_jsonl_records(
                quarantine_path,
                max_bytes=MAX_QUARANTINE_BYTES,
                max_line_bytes=MAX_QUARANTINE_LINE_BYTES,
                max_records=10_000,
            ),
            start=1,
        ):
            identity = record.get("position_id")
            if not isinstance(identity, str) or identity not in selected_by_id:
                raise LabelingError(
                    f"quarantine position_id is outside the current selection: {identity}"
                )
            _validate_quarantine(
                record,
                selected_by_id[identity],
                selection,
                config,
                fingerprint,
            )
            if identity in quarantined:
                raise LabelingError(f"duplicate quarantine position_id on line {line_number}")
            if identity in completed:
                raise LabelingError(
                    f"position_id appears in both labels and quarantine: {identity}"
                )
            quarantined[identity] = record
    except ArtifactError as error:
        raise LabelingError(f"cannot resume quarantine: {error}") from error
    return _Progress(completed, quarantined, legality_coverage)


def _validate_quarantine(
    record: dict[str, Any],
    position: SelectedPosition,
    selection: SelectionResult,
    config: TeacherConfig,
    fingerprint: TeacherFingerprint,
) -> None:
    keys = {
        "schema",
        "position_id",
        "canonical_sfen",
        "canonical_state_sha256",
        "split",
        "game_id",
        "position_index",
        "stage",
        "source_id",
        "outcome",
        "dataset_manifest_sha256",
        "config_sha256",
        "teacher",
        "error_category",
        "error_message",
        "attempts",
        "stderr_tail",
        "created_at",
    }
    if set(record) != keys or record["schema"] != QUARANTINE_SCHEMA:
        raise LabelingError("quarantine row schema or keys are invalid")
    if record["dataset_manifest_sha256"] != selection.dataset_manifest_sha256:
        raise LabelingError("quarantine dataset manifest hash mismatch")
    if record["config_sha256"] != config.sha256:
        raise LabelingError("quarantine config hash mismatch")
    if record["teacher"] != fingerprint.identity_record():
        raise LabelingError("quarantine teacher fingerprint mismatch")
    expected_position = {
        "position_id": position.position_id,
        "canonical_sfen": position.canonical_sfen,
        "canonical_state_sha256": position.canonical_state_sha256,
        "split": position.split,
        "game_id": position.game_id,
        "position_index": position.position_index,
        "stage": position.stage,
        "source_id": position.source_id,
        "outcome": position.outcome,
    }
    if any(record[key] != value for key, value in expected_position.items()):
        raise LabelingError("quarantine position provenance disagrees with selection")
    attempts = record["attempts"]
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or not 1 <= attempts <= config.labeling.max_retries + 1
    ):
        raise LabelingError("quarantine attempts must be an integer")
    if record["error_category"] not in {"process", "protocol", "timeout"}:
        raise LabelingError("quarantine error category is invalid")
    if not isinstance(record["error_message"], str) or len(record["error_message"]) > 2_048:
        raise LabelingError("quarantine error message is invalid")
    stderr_tail = record["stderr_tail"]
    if (
        not isinstance(stderr_tail, str)
        or len(stderr_tail.encode("utf-8")) > config.protocol_limits.max_stderr_bytes * 3
    ):
        raise LabelingError("quarantine stderr tail is invalid")
    created_at = record["created_at"]
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise LabelingError("quarantine created_at is invalid")
    try:
        datetime.fromisoformat(f"{created_at[:-1]}+00:00")
    except ValueError as error:
        raise LabelingError("quarantine created_at is invalid") from error


def _verify_label_position(value: dict[str, Any], position: SelectedPosition) -> None:
    expected = {
        "position_id": position.position_id,
        "canonical_sfen": position.canonical_sfen,
        "canonical_state_sha256": position.canonical_state_sha256,
        "side_to_move": position.side_to_move,
        "split": position.split,
        "game_id": position.game_id,
        "position_index": position.position_index,
        "stage": position.stage,
        "source_id": position.source_id,
        "outcome": position.outcome,
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise LabelingError("label position/source provenance disagrees with selection")


def _verify_label_teacher(
    value: object,
    fingerprint: TeacherFingerprint,
    reported_identity: dict[str, str | None],
) -> None:
    if not isinstance(value, dict):
        raise LabelingError("label teacher must be an object")
    actual = {
        "name": value.get("name"),
        "version": value.get("version"),
        "binary": {
            "path": fingerprint.binary.path,
            "sha256": value.get("binary_sha256"),
            "size": value.get("binary_size"),
        },
        "eval_files": value.get("eval_files"),
        "options": value.get("options"),
    }
    expected = fingerprint.identity_record()
    if actual != expected:
        raise LabelingError("label teacher fingerprint/options differ from current files")
    if (
        value.get("reported_name") != reported_identity["name"]
        or value.get("reported_author") != reported_identity["author"]
    ):
        raise LabelingError("label teacher USI identity differs from authorized benchmark")


def _write_manifest(
    path: Path,
    labels_path: Path,
    quarantine_path: Path,
    *,
    selection: SelectionResult,
    config: TeacherConfig,
    fingerprint: TeacherFingerprint,
    benchmark_path: Path,
    benchmark_sha256: str,
    progress: _Progress,
    reported_identity: dict[str, str | None],
    validator_identity: LegalityValidatorIdentity,
    target_completed: int,
    migration: dict[str, object] | None = None,
) -> None:
    try:
        labels_digest = jsonl_digest(
            labels_path,
            max_bytes=MAX_LABEL_ARTIFACT_BYTES,
            max_line_bytes=MAX_LABEL_LINE_BYTES,
            max_records=10_000,
        )
        quarantine_digest = jsonl_digest(
            quarantine_path,
            max_bytes=MAX_QUARANTINE_BYTES,
            max_line_bytes=MAX_QUARANTINE_LINE_BYTES,
            max_records=10_000,
        )
    except ArtifactError as error:
        raise LabelingError(f"cannot hash labeling artifacts: {error}") from error
    completed_ids = [
        position.position_id
        for position in selection.positions
        if position.position_id in progress.completed
    ]
    quarantine_ids = [
        position.position_id
        for position in selection.positions
        if position.position_id in progress.quarantined
    ]
    selected = len(selection.positions)
    completed = len(completed_ids)
    quarantined = len(quarantine_ids)
    manifest = {
        "schema": LABEL_MANIFEST_SCHEMA,
        "updated_at": _utc_now(),
        "config": {"schema": config.as_dict()["schema"], "sha256": config.sha256},
        "dataset_manifest": {
            "sha256": selection.dataset_manifest_sha256,
            "size": selection.dataset_manifest_size,
        },
        "positions": {
            "sha256": selection.positions_sha256,
            "size": selection.positions_size,
            "input_rows": selection.input_rows,
        },
        "selection": selection.summary(),
        "teacher": fingerprint.identity_record(),
        "reported_identity": reported_identity,
        "benchmark": {
            "path": benchmark_path.name,
            "sha256": benchmark_sha256,
            "selected_nodes": config.nodes,
        },
        "artifacts": {
            labels_path.name: labels_digest.as_dict(),
            quarantine_path.name: quarantine_digest.as_dict(),
        },
        "progress": {
            "selected": selected,
            "completed": completed,
            "quarantined": quarantined,
            "pending": selected - completed - quarantined,
            "target_completed": target_completed,
            "status": _status(
                selected,
                completed,
                quarantined,
                target_completed=target_completed,
            ),
            "completed_position_ids": completed_ids,
            "quarantined_position_ids": quarantine_ids,
        },
        "binding": {
            "schema": "phase4_teacher_label_binding/v2",
            "compatibility": (
                "phase4_teacher_label/v1 rows preserve the teacher-returned contiguous "
                "MultiPV prefix; this closed manifest binds selection, benchmark, exact "
                "artifacts, Rust replay, and aggregate candidate coverage without rewriting "
                "the append-only label artifact"
            ),
            "selection_sha256": selection.selection_sha256,
            "benchmark_sha256": benchmark_sha256,
            "labels": labels_digest.as_dict(),
            "legality_validator": validator_identity.as_dict(),
            "candidate_coverage": _candidate_coverage(progress, config),
        },
    }
    retained_migration = migration if migration is not None else _retained_migration(path)
    if retained_migration is not None:
        _validate_migration_record(retained_migration, output_dir=path.parent)
        manifest["migration"] = retained_migration
    write_json_atomic(path, manifest, replace=True)


def _validate_existing_manifest(
    path: Path,
    *,
    labels_path: Path,
    quarantine_path: Path,
    selection: SelectionResult,
    config: TeacherConfig,
    fingerprint: TeacherFingerprint,
    benchmark_path: Path,
    benchmark_sha256: str,
    reported_identity: dict[str, str | None],
    validator_identity: LegalityValidatorIdentity,
) -> dict[str, Any] | None:
    try:
        manifest_status = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise LabelingError(f"cannot inspect resume manifest: {error}") from error
    if stat.S_ISLNK(manifest_status.st_mode) or not stat.S_ISREG(manifest_status.st_mode):
        raise LabelingError("existing label manifest must be a non-symlink regular file")
    try:
        manifest, _ = load_json_object(path, max_bytes=4 * 1024 * 1024)
    except ArtifactError as error:
        raise LabelingError(f"cannot resume manifest: {error}") from error
    schema = manifest.get("schema")
    common_keys = {
        "schema",
        "updated_at",
        "config",
        "dataset_manifest",
        "positions",
        "selection",
        "teacher",
        "reported_identity",
        "benchmark",
        "artifacts",
        "progress",
    }
    expected_keys = common_keys
    if schema == LABEL_MANIFEST_SCHEMA:
        expected_keys |= {"binding"}
        if "migration" in manifest:
            expected_keys |= {"migration"}
    if (
        schema not in {LEGACY_LABEL_MANIFEST_SCHEMA, LABEL_MANIFEST_SCHEMA}
        or set(manifest) != expected_keys
    ):
        raise LabelingError("existing label manifest schema is invalid")
    _validate_timestamp(manifest["updated_at"], "existing label manifest updated_at")
    manifest_config = manifest.get("config")
    manifest_dataset = manifest.get("dataset_manifest")
    manifest_positions = manifest.get("positions")
    manifest_selection = manifest.get("selection")
    manifest_benchmark = manifest.get("benchmark")
    manifest_artifacts = manifest.get("artifacts")
    manifest_progress = manifest.get("progress")
    if not all(
        isinstance(value, dict)
        for value in (
            manifest_config,
            manifest_dataset,
            manifest_positions,
            manifest_selection,
            manifest_benchmark,
            manifest_artifacts,
            manifest_progress,
        )
    ):
        raise LabelingError("existing label manifest provenance objects are invalid")
    if not _strict_equal(
        manifest_config,
        {"schema": config.as_dict()["schema"], "sha256": config.sha256},
    ):
        raise LabelingError("existing label manifest config hash differs")
    if not _strict_equal(
        manifest_dataset,
        {
            "sha256": selection.dataset_manifest_sha256,
            "size": selection.dataset_manifest_size,
        },
    ):
        raise LabelingError("existing label manifest dataset hash differs")
    if not _strict_equal(
        manifest_positions,
        {
            "sha256": selection.positions_sha256,
            "size": selection.positions_size,
            "input_rows": selection.input_rows,
        },
    ):
        raise LabelingError("existing label manifest positions identity differs")
    expected_selection = (
        selection.legacy_summary()
        if schema == LEGACY_LABEL_MANIFEST_SCHEMA
        else selection.summary()
    )
    if not _strict_equal(manifest_selection, expected_selection):
        raise LabelingError("existing label manifest selection hash differs")
    if not _strict_equal(manifest.get("teacher"), fingerprint.identity_record()):
        raise LabelingError("existing label manifest teacher identity differs")
    if manifest.get("reported_identity") != reported_identity:
        raise LabelingError("existing label manifest teacher USI identity differs")
    if manifest_benchmark != {
        "path": benchmark_path.name,
        "sha256": benchmark_sha256,
        "selected_nodes": config.nodes,
    }:
        raise LabelingError("existing label manifest benchmark hash differs")
    if set(manifest_artifacts) != {labels_path.name, quarantine_path.name}:
        raise LabelingError("existing label manifest artifact set is invalid")
    for artifact_path in (labels_path, quarantine_path):
        recorded = manifest_artifacts[artifact_path.name]
        if not isinstance(recorded, dict) or set(recorded) != {"sha256", "size", "records"}:
            raise LabelingError("existing label manifest artifact digest is invalid")
        digest = _validate_recorded_digest(recorded, artifact_path)
        try:
            actual = jsonl_prefix_digest(
                artifact_path,
                size=digest["size"],
                records=digest["records"],
            )
        except ArtifactError as error:
            raise LabelingError(f"existing artifact checkpoint is invalid: {error}") from error
        if actual.sha256 != digest["sha256"]:
            raise LabelingError(
                f"existing {artifact_path.name} bytes differ from the recorded full digest"
            )
    if schema == LABEL_MANIFEST_SCHEMA:
        binding = manifest["binding"]
        expected_binding_keys = {
            "schema",
            "compatibility",
            "selection_sha256",
            "benchmark_sha256",
            "labels",
            "legality_validator",
            "candidate_coverage",
        }
        legacy_binding_keys = expected_binding_keys - {"candidate_coverage"}
        if (
            not isinstance(binding, dict)
            or (
                binding.get("schema") == "phase4_teacher_label_binding/v1"
                and set(binding) != legacy_binding_keys
            )
            or (
                binding.get("schema") == "phase4_teacher_label_binding/v2"
                and set(binding) != expected_binding_keys
            )
        ):
            raise LabelingError("existing label manifest binding is invalid")
        if binding.get("schema") not in {
            "phase4_teacher_label_binding/v1",
            "phase4_teacher_label_binding/v2",
        }:
            raise LabelingError("existing label manifest binding schema is invalid")
        compatibility = binding["compatibility"]
        if not isinstance(compatibility, str) or not compatibility:
            raise LabelingError("existing label manifest compatibility note is invalid")
        if (
            binding["selection_sha256"] != selection.selection_sha256
            or binding["benchmark_sha256"] != benchmark_sha256
            or binding["labels"] != manifest_artifacts[labels_path.name]
            or binding["legality_validator"] != validator_identity.as_dict()
            or (
                binding["schema"] == "phase4_teacher_label_binding/v2"
                and not isinstance(binding["candidate_coverage"], dict)
            )
        ):
            raise LabelingError("existing label manifest binding identity differs")
        if "migration" in manifest:
            migration = _validate_migration_record(manifest["migration"], output_dir=path.parent)
            _validate_frozen_legacy_origin(
                manifest,
                migration=migration,
                output_dir=path.parent,
                selection=selection,
            )
    return manifest


def _retained_migration(path: Path) -> dict[str, object] | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise LabelingError(f"cannot inspect label manifest: {error}") from error
    try:
        manifest, _ = load_json_object(path, max_bytes=4 * 1024 * 1024)
    except ArtifactError as error:
        raise LabelingError(f"cannot retain label migration evidence: {error}") from error
    value = manifest.get("migration")
    if value is None:
        return None
    return _validate_migration_record(value, output_dir=path.parent)


def _validate_migration_record(value: object, *, output_dir: Path) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "legacy_manifest",
        "legacy_schema",
        "legacy_updated_at",
    }:
        raise LabelingError("label manifest migration evidence schema is invalid")
    if (
        value.get("schema") != LABEL_MANIFEST_MIGRATION_SCHEMA
        or value.get("legacy_schema") != LEGACY_LABEL_MANIFEST_SCHEMA
    ):
        raise LabelingError("label manifest migration identity is invalid")
    _validate_timestamp(value.get("legacy_updated_at"), "legacy label manifest updated_at")
    reference = value.get("legacy_manifest")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "size"}:
        raise LabelingError("legacy label manifest reference is invalid")
    if reference.get("path") != LEGACY_MANIFEST_EVIDENCE_NAME:
        raise LabelingError("legacy manifest evidence must use the frozen evidence filename")
    sha256 = reference.get("sha256")
    size = reference.get("size")
    if (
        not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= 4 * 1024 * 1024
    ):
        raise LabelingError("legacy manifest evidence identity is invalid")
    try:
        _, observed = read_regular_bytes(
            output_dir / LEGACY_MANIFEST_EVIDENCE_NAME,
            max_bytes=4 * 1024 * 1024,
        )
    except ArtifactError as error:
        raise LabelingError(f"cannot verify legacy manifest evidence: {error}") from error
    if observed.sha256 != sha256 or observed.size != size:
        raise LabelingError("legacy manifest evidence bytes differ from migration record")
    return value


def _validate_frozen_legacy_origin(
    manifest: dict[str, Any],
    *,
    migration: dict[str, object],
    output_dir: Path,
    selection: SelectionResult,
) -> None:
    """Prove on every v2 load that the retained v1 object is the exact origin."""

    reference = migration["legacy_manifest"]
    assert isinstance(reference, dict)
    try:
        legacy, digest = load_json_object(
            output_dir / LEGACY_MANIFEST_EVIDENCE_NAME,
            max_bytes=4 * 1024 * 1024,
        )
    except ArtifactError as error:
        raise LabelingError(f"cannot parse frozen legacy manifest evidence: {error}") from error
    if digest.sha256 != reference["sha256"] or digest.size != reference["size"]:
        raise LabelingError("frozen legacy manifest identity changed during semantic audit")
    common_keys = {
        "schema",
        "updated_at",
        "config",
        "dataset_manifest",
        "positions",
        "selection",
        "teacher",
        "reported_identity",
        "benchmark",
        "artifacts",
        "progress",
    }
    if set(legacy) != common_keys:
        raise LabelingError("frozen legacy manifest violates its closed v1 schema")
    expected = {key: manifest[key] for key in common_keys}
    expected["schema"] = LEGACY_LABEL_MANIFEST_SCHEMA
    expected["updated_at"] = migration["legacy_updated_at"]
    expected["selection"] = selection.legacy_summary()
    if not _strict_equal(legacy, expected):
        raise LabelingError("frozen legacy manifest is not the exact semantic origin of v2")


def _publish_immutable_evidence(path: Path, payload: bytes) -> None:
    expected = hashlib.sha256(payload).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    existed = False
    try:
        with stable_parent_descriptor(path, create=True) as (parent_descriptor, name):
            temporary_name = f".{name}.{secrets.token_hex(12)}.pending"
            temporary_status: os.stat_result | None = None
            temporary_created = False
            descriptor = os.open(
                temporary_name,
                flags | getattr(os, "O_CLOEXEC", 0),
                0o400,
                dir_fd=parent_descriptor,
            )
            temporary_created = True
            temporary_status = os.fstat(descriptor)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                    temporary_status = os.fstat(output.fileno())
                assert temporary_status is not None
                try:
                    publish_regular_at(
                        parent_descriptor,
                        temporary_name,
                        name,
                        temporary_status,
                        display=path,
                        replace=False,
                    )
                except FileExistsError:
                    existed = True
                else:
                    temporary_created = False
                os.fsync(parent_descriptor)
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
            if temporary_created:
                assert temporary_status is not None
                retire_bound_regular(
                    parent_descriptor,
                    temporary_name,
                    temporary_status,
                    display=path,
                )
    except (ArtifactError, OSError) as error:
        raise LabelingError(f"cannot publish legacy manifest evidence: {error}") from error
    if existed:
        try:
            existing, digest = read_regular_bytes(path, max_bytes=4 * 1024 * 1024)
        except ArtifactError as error:
            raise LabelingError(f"cannot adopt legacy manifest evidence: {error}") from error
        if existing != payload or digest.sha256 != expected:
            raise LabelingError("legacy manifest evidence path contains different bytes") from None


def _validate_manifest_progress(
    manifest: dict[str, Any] | None,
    progress: _Progress,
    selection: SelectionResult,
    *,
    config: TeacherConfig,
    exact: bool = False,
) -> None:
    if manifest is None:
        return
    recorded = manifest["progress"]
    expected_keys = {
        "selected",
        "completed",
        "quarantined",
        "pending",
        "target_completed",
        "status",
        "completed_position_ids",
        "quarantined_position_ids",
    }
    if not isinstance(recorded, dict) or set(recorded) != expected_keys:
        raise LabelingError("existing label manifest progress keys are invalid")
    selected = len(selection.positions)
    completed_ids = recorded["completed_position_ids"]
    quarantined_ids = recorded["quarantined_position_ids"]
    if not isinstance(completed_ids, list) or not isinstance(quarantined_ids, list):
        raise LabelingError("existing label manifest progress identities are invalid")
    current_completed = [
        item.position_id for item in selection.positions if item.position_id in progress.completed
    ]
    current_quarantined = [
        item.position_id for item in selection.positions if item.position_id in progress.quarantined
    ]
    expected_completed = current_completed if exact else current_completed[: len(completed_ids)]
    expected_quarantined = (
        current_quarantined if exact else current_quarantined[: len(quarantined_ids)]
    )
    if completed_ids != expected_completed:
        raise LabelingError("existing label manifest completed identities were not append-only")
    if quarantined_ids != expected_quarantined:
        raise LabelingError("existing label manifest quarantine identities were not append-only")
    completed = len(completed_ids)
    quarantined = len(quarantined_ids)
    artifacts = manifest["artifacts"]
    if (
        artifacts["labels.jsonl"]["records"] != completed
        or artifacts["quarantine.jsonl"]["records"] != quarantined
    ):
        raise LabelingError("existing label manifest artifact/progress counts disagree")
    target = recorded["target_completed"]
    for value, name in (
        (recorded["selected"], "selected"),
        (recorded["completed"], "completed"),
        (recorded["quarantined"], "quarantined"),
        (recorded["pending"], "pending"),
        (target, "target_completed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise LabelingError(f"existing label manifest progress {name} must be an integer")
    if (
        recorded["selected"] != selected
        or recorded["completed"] != completed
        or recorded["quarantined"] != quarantined
        or recorded["pending"] != selected - completed - quarantined
        or not 1 <= target <= selected
        or recorded["status"] != _status(selected, completed, quarantined, target_completed=target)
    ):
        raise LabelingError("existing label manifest progress summary is inconsistent")
    binding = manifest.get("binding")
    if (
        manifest["schema"] == LABEL_MANIFEST_SCHEMA
        and isinstance(binding, dict)
        and binding.get("schema") == "phase4_teacher_label_binding/v2"
        and not _strict_equal(
            binding["candidate_coverage"],
            _candidate_coverage_for_ids(progress, config, set(completed_ids)),
        )
    ):
        raise LabelingError("existing label manifest candidate coverage differs")


def _candidate_coverage(
    progress: _Progress,
    config: TeacherConfig,
) -> dict[str, object]:
    return _candidate_coverage_for_ids(progress, config, set(progress.completed))


def _candidate_coverage_for_ids(
    progress: _Progress,
    config: TeacherConfig,
    completed_ids: set[str],
) -> dict[str, object]:
    if not completed_ids <= set(progress.completed):
        raise LabelingError("candidate coverage references an unknown completed label")
    if not completed_ids <= set(progress.legality_coverage):
        raise LabelingError("candidate coverage is missing a completed label")
    returned_counts = {str(count): 0 for count in range(1, config.multipv + 1)}
    short_rows = 0
    exact_short_rows = 0
    short_rows_with_additional_legal_roots = 0
    additional_legal_root_moves = 0
    for identity in completed_ids:
        coverage = progress.legality_coverage[identity]
        if (
            coverage.requested_multipv != config.multipv
            or not 1 <= coverage.returned_candidates <= config.multipv
        ):
            raise LabelingError("candidate coverage disagrees with configured MultiPV")
        returned_counts[str(coverage.returned_candidates)] += 1
        if coverage.returned_candidates == config.multipv:
            if coverage.legal_root_count is not None:
                raise LabelingError("full MultiPV coverage unexpectedly records a legal-root count")
            continue
        short_rows += 1
        if (
            coverage.legal_root_count is None
            or coverage.legal_root_count < coverage.returned_candidates
        ):
            raise LabelingError("short MultiPV coverage has an invalid legal-root count")
        if coverage.has_additional_legal_roots:
            short_rows_with_additional_legal_roots += 1
            additional_legal_root_moves += coverage.legal_root_count - coverage.returned_candidates
        else:
            exact_short_rows += 1
    return {
        "requested_multipv": config.multipv,
        "completed_labels": len(completed_ids),
        "returned_candidate_counts": returned_counts,
        "short_rows": short_rows,
        "short_rows_with_exact_legal_root_coverage": exact_short_rows,
        "short_rows_with_additional_legal_roots": (short_rows_with_additional_legal_roots),
        "additional_legal_root_moves": additional_legal_root_moves,
    }


def _validate_exact_artifact_snapshot(
    manifest: dict[str, Any],
    *,
    labels_path: Path,
    quarantine_path: Path,
) -> None:
    limits = {
        labels_path: (MAX_LABEL_ARTIFACT_BYTES, MAX_LABEL_LINE_BYTES),
        quarantine_path: (MAX_QUARANTINE_BYTES, MAX_QUARANTINE_LINE_BYTES),
    }
    for artifact_path, (max_bytes, max_line_bytes) in limits.items():
        try:
            actual = jsonl_digest(
                artifact_path,
                max_bytes=max_bytes,
                max_line_bytes=max_line_bytes,
                max_records=10_000,
            ).as_dict()
        except ArtifactError as error:
            raise LabelingError(f"cannot audit {artifact_path.name}: {error}") from error
        if not _strict_equal(actual, manifest["artifacts"][artifact_path.name]):
            raise LabelingError(
                f"current {artifact_path.name} is not the exact manifest-bound artifact"
            )


def _validate_recorded_digest(recorded: dict[str, Any], path: Path) -> dict[str, Any]:
    sha256 = recorded["sha256"]
    size = recorded["size"]
    records = recorded["records"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(records, bool)
        or not isinstance(records, int)
        or records < 0
    ):
        raise LabelingError(f"existing {path.name} recorded digest fields are invalid")
    return recorded


def _benchmark_reported_identity(report: dict[str, Any]) -> dict[str, str | None]:
    value = report.get("reported_identity")
    if not isinstance(value, dict) or set(value) != {"name", "author"}:
        raise LabelingError("authorized benchmark has no closed teacher USI identity")
    name = value["name"]
    author = value["author"]
    if not isinstance(name, str) or not name or any(character in name for character in "\r\n\0"):
        raise LabelingError("authorized benchmark teacher name is invalid")
    if author is not None and (
        not isinstance(author, str)
        or not author
        or any(character in author for character in "\r\n\0")
    ):
        raise LabelingError("authorized benchmark teacher author is invalid")
    return {"name": name, "author": author}


def _validate_timestamp(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 64:
        raise LabelingError(f"{name} is invalid")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise LabelingError(f"{name} is invalid") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise LabelingError(f"{name} is not UTC")


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _ensure_append_file(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        with stable_parent_descriptor(path, create=True) as (parent_descriptor, name):
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise LabelingError(f"{path.name} must be a regular file")
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
            finally:
                os.close(descriptor)
    except (ArtifactError, OSError) as error:
        raise LabelingError(f"cannot create append-only artifact {path.name}: {error}") from error


def _regular_file_size(path: Path) -> int:
    try:
        with stable_regular_descriptor(path) as descriptor:
            return os.fstat(descriptor).st_size
    except ArtifactError as error:
        raise LabelingError(f"cannot inspect {path.name}: {error}") from error


def _status(
    selected: int,
    completed: int,
    quarantined: int,
    *,
    target_completed: int,
) -> str:
    if completed + quarantined < selected:
        if completed >= target_completed:
            return "staged"
        return "in_progress"
    if quarantined:
        return "completed_with_quarantine"
    return "complete"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
