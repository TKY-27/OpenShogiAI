"""CLI for deterministic selection, teacher benchmarking, and labeling."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from open_shogi_training.labeling.artifacts import ArtifactError
from open_shogi_training.labeling.benchmark import BenchmarkError, run_benchmark
from open_shogi_training.labeling.config import TeacherConfigError, load_teacher_config
from open_shogi_training.labeling.fingerprint import TeacherFingerprintError
from open_shogi_training.labeling.pipeline import (
    LabelingError,
    audit_labeling_output,
    migrate_label_manifest_v2,
    run_labeling,
)
from open_shogi_training.labeling.schema import LabelSchemaError, iter_teacher_labels
from open_shogi_training.labeling.selection import SelectionError, select_positions
from open_shogi_training.labeling.usi import USIError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m open_shogi_training.labeling",
        description="Bounded black-box USI teacher labeling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select",
        help="validate and summarize deterministic leakage-safe position selection",
    )
    _add_inputs(select)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="measure conservative node budgets and publish an authorization report",
    )
    _add_inputs(benchmark)
    benchmark.add_argument("--output", type=Path, required=True)

    label = subparsers.add_parser(
        "label",
        help="resume append-only labeling authorized by an exact benchmark report",
    )
    _add_inputs(label)
    label.add_argument("--benchmark-report", type=Path, required=True)
    label.add_argument("--output-dir", type=Path, required=True)
    label.add_argument(
        "--target-completed",
        type=int,
        choices=(10, 100, 1_000, 10_000),
        required=True,
        help="successful-label checkpoint; rerun the same output at the next checkpoint",
    )

    audit = subparsers.add_parser(
        "audit-label-set",
        help="read-only provenance, digest, and Rust-legality audit of a label set",
    )
    _add_inputs(audit)
    audit.add_argument("--benchmark-report", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)

    migrate = subparsers.add_parser(
        "migrate-label-manifest-v2",
        help=(
            "re-audit a complete legacy label set without a teacher call and atomically "
            "publish a legacy-evidence-bound v2 manifest"
        ),
    )
    _add_inputs(migrate)
    migrate.add_argument("--benchmark-report", type=Path, required=True)
    migrate.add_argument("--output-dir", type=Path, required=True)
    migrate.add_argument("--expected-legacy-manifest-sha256", required=True)

    validate = subparsers.add_parser(
        "validate-labels",
        help="stream and validate a completed or partial label JSONL artifact",
    )
    validate.add_argument("--labels", type=Path, required=True)
    validate.add_argument("--dataset-manifest-sha256")
    validate.add_argument("--config-sha256")
    return parser


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate-labels":
            count = sum(
                1
                for _ in iter_teacher_labels(
                    arguments.labels,
                    expected_dataset_manifest_sha256=arguments.dataset_manifest_sha256,
                    expected_config_sha256=arguments.config_sha256,
                )
            )
            _emit({"schema_version": 1, "mode": "validate-labels", "records": count})
            return 0
        config = load_teacher_config(arguments.config)
        if arguments.command == "select":
            selection = select_positions(
                arguments.positions,
                arguments.dataset_manifest,
                config.selection,
            )
            _emit(
                {
                    "schema_version": 1,
                    "mode": "select",
                    "config_sha256": config.sha256,
                    "dataset_manifest_sha256": selection.dataset_manifest_sha256,
                    "positions_sha256": selection.positions_sha256,
                    "selection": selection.summary(),
                }
            )
            return 0
        if arguments.command == "benchmark":
            result = run_benchmark(
                config=config,
                project_root=arguments.project_root,
                positions_path=arguments.positions,
                dataset_manifest_path=arguments.dataset_manifest,
                output_path=arguments.output,
            )
            _emit(
                {
                    "schema_version": 1,
                    "mode": "benchmark",
                    "output": str(result.path),
                    "sha256": result.digest.sha256,
                    "selected_nodes": result.selected_nodes,
                    "configured_nodes": config.nodes,
                    "labeling_authorized": result.selected_nodes == config.nodes,
                }
            )
            return 0 if result.selected_nodes == config.nodes else 3
        if arguments.command == "label":
            result = run_labeling(
                config=config,
                project_root=arguments.project_root,
                positions_path=arguments.positions,
                dataset_manifest_path=arguments.dataset_manifest,
                benchmark_report_path=arguments.benchmark_report,
                output_dir=arguments.output_dir,
                target_completed=arguments.target_completed,
            )
            _emit(
                {
                    "schema_version": 1,
                    "mode": "label",
                    "output_dir": str(result.output_dir),
                    "manifest": str(result.manifest_path),
                    "selected": result.selected,
                    "completed": result.completed,
                    "quarantined": result.quarantined,
                    "pending": result.pending,
                    "status": result.status,
                }
            )
            return 0 if result.status in {"complete", "staged"} else 3
        if arguments.command == "audit-label-set":
            result = audit_labeling_output(
                config=config,
                project_root=arguments.project_root,
                positions_path=arguments.positions,
                dataset_manifest_path=arguments.dataset_manifest,
                benchmark_report_path=arguments.benchmark_report,
                output_dir=arguments.output_dir,
            )
            _emit(
                {
                    "schema_version": 1,
                    "mode": "audit-label-set",
                    "output_dir": str(result.output_dir),
                    "manifest": str(result.manifest_path),
                    "manifest_schema": result.manifest_schema,
                    "selected": result.selected,
                    "completed": result.completed,
                    "quarantined": result.quarantined,
                    "pending": result.pending,
                    "status": result.status,
                    "selection_sha256": result.selection_sha256,
                    "benchmark_sha256": result.benchmark_sha256,
                    "labels_sha256": result.labels_sha256,
                    "legality_validator": result.legality_validator.as_dict(),
                    "candidate_coverage": result.candidate_coverage,
                }
            )
            return 0
        if arguments.command == "migrate-label-manifest-v2":
            result = migrate_label_manifest_v2(
                config=config,
                project_root=arguments.project_root,
                positions_path=arguments.positions,
                dataset_manifest_path=arguments.dataset_manifest,
                benchmark_report_path=arguments.benchmark_report,
                output_dir=arguments.output_dir,
                expected_legacy_manifest_sha256=(arguments.expected_legacy_manifest_sha256),
            )
            _emit(
                {
                    "schema_version": 1,
                    "mode": "migrate-label-manifest-v2",
                    "output_dir": str(result.audit.output_dir),
                    "manifest": str(result.audit.manifest_path),
                    "manifest_schema": result.audit.manifest_schema,
                    "legacy_manifest_sha256": result.legacy_manifest_sha256,
                    "legacy_manifest_evidence": str(result.legacy_manifest_path),
                    "selected": result.audit.selected,
                    "completed": result.audit.completed,
                    "quarantined": result.audit.quarantined,
                    "pending": result.audit.pending,
                    "status": result.audit.status,
                    "selection_sha256": result.audit.selection_sha256,
                    "benchmark_sha256": result.audit.benchmark_sha256,
                    "labels_sha256": result.audit.labels_sha256,
                    "legality_validator": result.audit.legality_validator.as_dict(),
                    "candidate_coverage": result.audit.candidate_coverage,
                }
            )
            return 0
        parser.error(f"unsupported command: {arguments.command}")
    except (
        ArtifactError,
        BenchmarkError,
        LabelSchemaError,
        LabelingError,
        OSError,
        SelectionError,
        TeacherConfigError,
        TeacherFingerprintError,
        USIError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
