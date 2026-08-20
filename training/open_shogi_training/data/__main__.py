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
