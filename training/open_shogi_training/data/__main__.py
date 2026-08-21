"""Command-line entry point for audited sample acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from open_shogi_training.data.downloader import (
    MAX_SAMPLE_FILES,
    AcquisitionError,
    Downloader,
    plan_acquisition,
)
from open_shogi_training.data.manifest import ManifestError
from open_shogi_training.data.normalize import (
    NormalizationConfig,
    normalize_aobazero_dataset,
)
from open_shogi_training.data.opening import (
    OpeningConfig,
    build_opening_database,
    export_opening_jsonl,
)
from open_shogi_training.data.opening_v2 import build_opening_book_v2
from open_shogi_training.data.phase10r_acquisition import (
    acquire_artifact,
    data_root_from_environment,
)
from open_shogi_training.data.phase10r_acquisition import (
    dry_run as phase10r_dry_run,
)
from open_shogi_training.data.phase10r_adapters import (
    normalize_source_file,
    validate_kif_with_engine,
    write_normalized_jsonl,
)
from open_shogi_training.data.phase10r_archive import inventory_archive, write_inventory
from open_shogi_training.data.phase10r_registry import load_phase10r_registry
from open_shogi_training.data.registry import RegistryError, load_source_registry
from open_shogi_training.data.splits import SplitPolicy
from open_shogi_training.selfplay.common import ArtifactRef, artifact_ref
from open_shogi_training.selfplay.engine_receipt import (
    DEFAULT_ENGINE_PATH,
    resolve_active_engine_build,
)

MAX_PROVENANCE_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m open_shogi_training.data",
        description="Rights-gated OpenShogiAI data acquisition",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-registry",
        help="validate the closed source registry and referenced catalogs",
    )
    validate.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/data_sources.yaml"),
    )

    dry_run = subparsers.add_parser(
        "dry-run",
        help="print an exact catalog plan without network or disk writes",
    )
    _add_common_arguments(dry_run)

    acquire = subparsers.add_parser(
        "acquire",
        help="acquire an approved exact catalog into an immutable object store",
    )
    _add_common_arguments(acquire)
    acquire.add_argument(
        "--output",
        type=Path,
        required=True,
        help="local raw-data root (must remain outside Git)",
    )
    acquire.add_argument(
        "--sample-only",
        action="store_true",
        required=True,
        help="acknowledge the Phase 3 hard limit of at most 100 files",
    )

    normalize = subparsers.add_parser(
        "normalize",
        help="normalize an acquired AobaZero sample with the local Rust exporter",
    )
    normalize.add_argument("--registry", type=Path, required=True)
    normalize.add_argument("--source", required=True)
    normalize.add_argument("--manifest", type=Path, required=True)
    normalize.add_argument("--acquisition-root", type=Path, required=True)
    normalize.add_argument("--processed-root", type=Path, required=True)
    normalize.add_argument("--dataset-id", required=True)
    normalize.add_argument("--split-salt", required=True)
    normalize.add_argument("--validation-basis-points", type=int, default=1_000)
    normalize.add_argument("--test-basis-points", type=int, default=1_000)
    normalize.add_argument("--max-games", type=int, default=100)
    normalize.add_argument("--max-positions", type=int, default=100_000)
    normalize.add_argument("--terminal-tail-positions", type=int, default=8)
    normalize.add_argument("--max-raw-bytes", type=int, default=16_777_216)
    normalize.add_argument("--exporter-timeout-seconds", type=int, default=120)
    normalize.add_argument("--cli", type=Path, required=True)

    opening = subparsers.add_parser(
        "build-opening",
        help="build an atomic SQLite opening database from normalized positions",
    )
    opening.add_argument("--positions", type=Path, required=True)
    opening.add_argument("--output", type=Path, required=True)
    opening.add_argument("--provenance", type=Path, required=True)
    opening.add_argument("--max-input-rows", type=int, default=100_000)
    opening.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    opening.add_argument(
        "--max-input-uncompressed-bytes",
        type=int,
        default=256 * 1024 * 1024,
    )

    export = subparsers.add_parser(
        "export-opening",
        help="export deterministic gzip JSONL from a built opening database",
    )
    export.add_argument("--database", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--min-count", type=int, default=1)

    opening_v2 = subparsers.add_parser(
        "build-opening-v2",
        help="build the teacher-safe provenance-bound OpenShogiAI opening book",
    )
    opening_v2.add_argument("--opening-v1", type=Path, required=True)
    opening_v2.add_argument("--labels", type=Path, required=True)
    opening_v2.add_argument("--label-manifest", type=Path, required=True)
    opening_v2.add_argument("--dataset-manifest", type=Path, required=True)
    opening_v2.add_argument("--output", type=Path, required=True)
    opening_v2.add_argument("--build-version", required=True)
    opening_v2.add_argument("--maximum-plies", type=int, default=40)
    opening_v2.add_argument("--minimum-sample-count", type=int, default=2)
    opening_v2.add_argument("--maximum-teacher-loss-cp", type=int, default=80)

    phase10r_validate = subparsers.add_parser(
        "phase10r-validate",
        help="validate the Phase 10R file-level rights registry",
    )
    phase10r_validate.add_argument(
        "--registry", type=Path, default=Path("configs/phase10r/source-registry.yaml")
    )

    phase10r_dry_run = subparsers.add_parser(
        "phase10r-dry-run",
        help="print an approved Phase 10R acquisition plan without writes",
    )
    phase10r_dry_run.add_argument(
        "--registry", type=Path, default=Path("configs/phase10r/source-registry.yaml")
    )
    phase10r_dry_run.add_argument("--artifact", action="append", dest="artifacts")
    phase10r_dry_run.add_argument("--data-root", type=Path)

    phase10r_acquire = subparsers.add_parser(
        "phase10r-acquire",
        help="resume one or more explicitly approved Phase 10R downloads",
    )
    phase10r_acquire.add_argument(
        "--registry", type=Path, default=Path("configs/phase10r/source-registry.yaml")
    )
    phase10r_acquire.add_argument("--artifact", action="append", dest="artifacts", required=True)
    phase10r_acquire.add_argument("--data-root", type=Path)
    phase10r_acquire.add_argument(
        "--purpose", choices=("training", "local-inspection"), default="training"
    )

    phase10r_inventory = subparsers.add_parser(
        "phase10r-inventory",
        help="inventory a ZIP archive without extracting it",
    )
    phase10r_inventory.add_argument("--archive", type=Path, required=True)
    phase10r_inventory.add_argument("--output", type=Path)

    phase10r_normalize = subparsers.add_parser(
        "phase10r-normalize-sample",
        help="normalize one bounded sample into ignored gzip JSONL",
    )
    phase10r_normalize.add_argument(
        "--registry", type=Path, default=Path("configs/phase10r/source-registry.yaml")
    )
    phase10r_normalize.add_argument("--input", type=Path, required=True)
    phase10r_normalize.add_argument("--artifact", required=True)
    phase10r_normalize.add_argument("--output", type=Path, required=True)
    phase10r_normalize.add_argument("--format")
    phase10r_normalize.add_argument("--max-records", type=int, default=100_000)

    phase10r_kif = subparsers.add_parser(
        "phase10r-validate-kif",
        help="convert one bounded KIF game and replay it with the Rust rule engine",
    )
    phase10r_kif.add_argument("--input", type=Path, required=True)
    phase10r_kif.add_argument("--output", type=Path, required=True)
    phase10r_kif.add_argument("--cli", type=Path, required=True)
    phase10r_kif.add_argument("--max-moves", type=int, default=10_000)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/data_sources.yaml"),
        help="strict source registry",
    )
    parser.add_argument("--source", required=True, help="exact source_id")
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_SAMPLE_FILES,
        help=f"catalog prefix size, 1-{MAX_SAMPLE_FILES}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate-registry":
            registry = load_source_registry(arguments.registry)
            _emit_json(
                {
                    "schema_version": 1,
                    "mode": "validate-registry",
                    "sources": [
                        {
                            "source_id": source.source_id,
                            "enabled": source.enabled,
                            "approved": source.approved,
                            "catalog_objects": len(source.catalog),
                            "evidence_objects": len(source.evidence_catalog),
                        }
                        for source in registry.sources
                    ],
                }
            )
            return 0
        if arguments.command == "build-opening":
            provenance = _load_bounded_json_object(arguments.provenance)
            report = build_opening_database(
                arguments.positions,
                arguments.output,
                provenance=provenance,
                config=OpeningConfig(
                    max_input_rows=arguments.max_input_rows,
                    max_input_bytes=arguments.max_input_bytes,
                    max_input_uncompressed_bytes=arguments.max_input_uncompressed_bytes,
                ),
            )
            _emit_json(
                {
                    "schema_version": 1,
                    "mode": "build-opening",
                    "output": str(arguments.output),
                    "input_rows": report.input_rows,
                    "included_rows": report.included_rows,
                    "state_count": report.state_count,
                    "move_count": report.move_count,
                    "source_count": report.source_count,
                    "sha256": report.sha256,
                    "size": report.size,
                }
            )
            return 0
        if arguments.command == "export-opening":
            export_opening_jsonl(
                arguments.database,
                arguments.output,
                min_count=arguments.min_count,
            )
            _emit_json(
                {
                    "schema_version": 1,
                    "mode": "export-opening",
                    "database": str(arguments.database),
                    "output": str(arguments.output),
                    "min_count": arguments.min_count,
                }
            )
            return 0
        if arguments.command == "build-opening-v2":
            report = build_opening_book_v2(
                arguments.opening_v1,
                arguments.labels,
                arguments.label_manifest,
                arguments.dataset_manifest,
                arguments.output,
                build_version=arguments.build_version,
                maximum_plies=arguments.maximum_plies,
                minimum_sample_count=arguments.minimum_sample_count,
                maximum_teacher_loss_cp=arguments.maximum_teacher_loss_cp,
            )
            _emit_json(
                {
                    "schema": "open_shogi_opening_build_report/v1",
                    "output": str(arguments.output),
                    "inputRecords": report.input_records,
                    "retainedPositions": report.retained_positions,
                    "retainedCandidates": report.retained_candidates,
                    "rejectedWithoutTeacher": report.rejected_without_teacher,
                    "rejectedTeacherLoss": report.rejected_teacher_loss,
                    "ibishaCandidates": report.ibisha_candidates,
                    "ibishaVsFuribishaCandidates": report.ibisha_vs_furibisha_candidates,
                    "artifactSha256": report.artifact.sha256,
                    "artifactSize": report.artifact.size,
                }
            )
            return 0
        if arguments.command == "phase10r-validate":
            registry = load_phase10r_registry(arguments.registry)
            _emit_json(
                {
                    "schema": "phase10r_registry_validation/v1",
                    "registry_id": registry.registry_id,
                    "registry_sha256": registry.sha256,
                    "source_count": len(registry.sources),
                    "artifact_count": len(registry.artifacts),
                    "states": registry.by_state(),
                    "approved_training_artifacts": [
                        artifact.artifact_id for artifact in registry.approved_artifacts()
                    ],
                }
            )
            return 0
        if arguments.command == "phase10r-dry-run":
            registry = load_phase10r_registry(arguments.registry)
            _emit_json(
                phase10r_dry_run(
                    registry,
                    arguments.artifacts,
                    data_root=arguments.data_root,
                    minimum_free_bytes=int(registry.policy["minimum_free_bytes"]),
                )
            )
            return 0
        if arguments.command == "phase10r-acquire":
            registry = load_phase10r_registry(arguments.registry)
            results = [
                acquire_artifact(
                    registry,
                    artifact_id,
                    data_root=arguments.data_root or data_root_from_environment(),
                    purpose=arguments.purpose,
                    minimum_free_bytes=int(registry.policy["minimum_free_bytes"]),
                    max_single_download_bytes=int(registry.policy["max_single_download_bytes"]),
                ).as_dict()
                for artifact_id in arguments.artifacts
            ]
            _emit_json(
                {
                    "schema": "phase10r_acquisition_report/v1",
                    "results": results,
                }
            )
            return 0
        if arguments.command == "phase10r-inventory":
            inventory = inventory_archive(arguments.archive)
            if arguments.output:
                write_inventory(arguments.archive, arguments.output, inventory)
            _emit_json(inventory.as_dict())
            return 0
        if arguments.command == "phase10r-normalize-sample":
            registry = load_phase10r_registry(arguments.registry)
            artifact = registry.artifact(arguments.artifact)
            records = normalize_source_file(
                arguments.input,
                source_id=artifact.source_id,
                artifact_id=artifact.artifact_id,
                format_name=arguments.format,
                compression=artifact.compression,
                source_revision=artifact.source_revision,
                license_decision=artifact.as_dict(),
                max_records=arguments.max_records,
            )
            report = write_normalized_jsonl(records, arguments.output)
            _emit_json({**report, "artifact_id": artifact.artifact_id})
            return 0
        if arguments.command == "phase10r-validate-kif":
            _emit_json(
                validate_kif_with_engine(
                    arguments.cli,
                    arguments.input,
                    arguments.output,
                    max_moves=arguments.max_moves,
                )
            )
            return 0
        registry = load_source_registry(arguments.registry)
        source = registry.get(arguments.source)
        if arguments.command == "dry-run":
            plan = plan_acquisition(source, limit=arguments.limit, sample_only=True)
            _emit_json(plan.as_dict())
            return 0
        if arguments.command == "acquire":
            downloader = Downloader(source, arguments.output)
            outcomes = downloader.acquire_catalog(
                limit=arguments.limit,
                sample_only=arguments.sample_only,
            )
            counts: dict[str, int] = {}
            for outcome in outcomes:
                counts[outcome.status] = counts.get(outcome.status, 0) + 1
            _emit_json(
                {
                    "schema_version": 1,
                    "mode": "acquire",
                    "source_id": source.source_id,
                    "sample_only": True,
                    "requested": arguments.limit,
                    "completed": len(outcomes),
                    "statuses": counts,
                    "manifest": str(downloader.manifest.path),
                }
            )
            return 0
        if arguments.command == "normalize":
            repository_root = Path.cwd().resolve(strict=True)
            engine, engine_receipt = _resolve_normalization_engine(
                repository_root,
                arguments.cli,
            )
            split_policy = SplitPolicy(
                salt=arguments.split_salt,
                validation_basis_points=arguments.validation_basis_points,
                test_basis_points=arguments.test_basis_points,
            )
            config = NormalizationConfig(
                dataset_id=arguments.dataset_id,
                split_policy=split_policy,
                max_games=arguments.max_games,
                max_positions=arguments.max_positions,
                terminal_tail_positions=arguments.terminal_tail_positions,
                max_raw_bytes=arguments.max_raw_bytes,
                exporter_timeout_seconds=arguments.exporter_timeout_seconds,
            )
            result = normalize_aobazero_dataset(
                arguments.manifest,
                acquisition_root=arguments.acquisition_root,
                processed_root=arguments.processed_root,
                source=source,
                config=config,
                repository_root=repository_root,
                engine=engine,
                engine_build_receipt=engine_receipt,
            )
            _emit_json(
                {
                    "schema_version": 1,
                    "mode": "normalize",
                    "dataset_id": arguments.dataset_id,
                    "output_dir": str(result.output_dir),
                    "manifest": str(result.manifest_path),
                    "report": str(result.report_path),
                    "included_games": result.included_games,
                    "included_positions": result.included_positions,
                    "excluded_games": result.excluded_games,
                }
            )
            return 0
        parser.error(f"unsupported command: {arguments.command}")
    except (
        AcquisitionError,
        ManifestError,
        RegistryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def _resolve_normalization_engine(
    repository_root: Path,
    configured_alias: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Resolve the mutable operator alias once and return immutable build refs."""

    if configured_alias != Path(DEFAULT_ENGINE_PATH):
        raise ValueError(f"--cli must be the fixed operator alias {DEFAULT_ENGINE_PATH}")
    engine, receipt, _ = resolve_active_engine_build(repository_root)
    alias = artifact_ref(repository_root, DEFAULT_ENGINE_PATH, maximum_bytes=512 * 1024 * 1024)
    if (alias.sha256, alias.size) != (engine.sha256, engine.size):
        raise ValueError("operator engine alias differs from the active immutable build")
    return engine, receipt


def _emit_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _load_bounded_json_object(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > MAX_PROVENANCE_BYTES:
        raise ValueError(f"provenance exceeds {MAX_PROVENANCE_BYTES} bytes")
    raw = path.read_bytes()
    if len(raw) > MAX_PROVENANCE_BYTES:
        raise ValueError(f"provenance exceeds {MAX_PROVENANCE_BYTES} bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"provenance contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid provenance JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("provenance JSON must be an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
