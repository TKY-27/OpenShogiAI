"""Model-lab command line for Phase 4 training, analysis, and export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from open_shogi_training.labeling.artifacts import ArtifactError, stable_regular_descriptor
from open_shogi_training.models.checkpoint import (
    load_checkpoint,
    load_checkpoint_with_identity,
    validate_resume_identity,
)
from open_shogi_training.models.config import (
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    combined_config_sha256,
    config_sha256,
    estimated_export_bytes,
    exported_operation_count,
    exported_parameter_count,
    load_feature_config,
    load_model_config,
    load_training_config,
    parse_feature_config,
    parse_model_config,
    parse_training_config,
    validate_config_compatibility,
)
from open_shogi_training.models.dataset import load_training_examples
from open_shogi_training.models.export import MAX_NON_MATE_CP, export_model_artifacts
from open_shogi_training.models.features import (
    feature_flags,
    feature_groups,
    feature_schema,
    input_dimension,
    parse_canonical_sfen,
)
from open_shogi_training.models.network import (
    ValueModel,
    ensure_finite_model,
    parameter_count,
    select_device,
)
from open_shogi_training.models.phase5_arena import (
    PHASE5_GIT_COMMIT,
    run_phase5_arena,
    verify_phase5_arena,
)
from open_shogi_training.models.residual import build_residual_baseline
from open_shogi_training.models.train import (
    compare_checkpoints,
    evaluate_checkpoint,
    feature_ablation,
    model_code_sha256,
    train_model,
)
from open_shogi_training.selfplay.common import require_clean_head

DEFAULT_FEATURES = Path("configs/features/value_v0.toml")
DEFAULT_MODEL = Path("configs/models/value_v0.toml")
DEFAULT_TRAINING = Path("configs/training/value_v0_full_initial.toml")
DEFAULT_SMOKE = Path("configs/training/value_v0_smoke.toml")
DEFAULT_OVERFIT = Path("configs/training/value_v0_overfit.toml")
MAX_PREDICTION_COMPARISON_BYTES = 128 * 1024 * 1024
MAX_PREDICTION_COMPARISON_ROWS = 100_000
_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1
_U128_MAX = 2**128 - 1
_USI_MOVE_RE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")
_PHASE4_PREDICTION_KEYS = frozenset(
    {
        "schema",
        "positionId",
        "canonicalSfen",
        "split",
        "stage",
        "teacherScore",
        "modelCp",
        "candidateGapCp",
        "bestmove",
        "recordedMove",
        "recordedMoveAgrees",
        "alreadyTeacherLabeled",
        "checkpointSha256",
        "configSha256",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m open_shogi_training.models",
        description="OpenShogiAI Phase 4 deterministic value-model laboratory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, training_default, help_text in (
        ("train", DEFAULT_TRAINING, "train or resume the configured value model"),
        ("smoke", DEFAULT_SMOKE, "run a bounded two-batch training smoke check"),
        ("overfit", DEFAULT_OVERFIT, "overfit a deterministic balanced tiny subset"),
    ):
        child = subparsers.add_parser(command, help=help_text, description=help_text)
        _add_config_arguments(child, training_default)
        _add_dataset_arguments(child)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--resume", type=Path)
        child.add_argument(
            "--stop-after-epoch",
            type=int,
            help="execution-only interruption boundary; does not change the config hash",
        )
        if command == "overfit":
            child.add_argument("--examples", type=int, default=32)

    validate = subparsers.add_parser("validate", help="evaluate one checkpoint on validation only")
    _add_evaluation_arguments(validate)
    test = subparsers.add_parser(
        "test", help="evaluate the held-out test split with explicit acknowledgement"
    )
    _add_evaluation_arguments(test)
    test.add_argument(
        "--final-test",
        action="store_true",
        required=True,
        help="explicitly acknowledge one final test-split evaluation",
    )

    compare = subparsers.add_parser(
        "compare", help="compare two checkpoints on the same validation rows"
    )
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    _add_dataset_arguments(compare)

    ablation = subparsers.add_parser(
        "feature-ablation", help="measure validation loss with each feature group zeroed"
    )
    ablation.add_argument("--checkpoint", type=Path, required=True)
    _add_dataset_arguments(ablation)

    export = subparsers.add_parser(
        "export", help="export a validated checkpoint to checksummed OSAVAL01 artifacts"
    )
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--base-name", default="value_v0")

    describe = subparsers.add_parser(
        "describe", help="report feature schema, parameter, operation, and size estimates"
    )
    _add_config_arguments(describe, DEFAULT_TRAINING)

    validate_config = subparsers.add_parser(
        "validate-config", help="validate the closed feature/model/training configuration set"
    )
    _add_config_arguments(validate_config, DEFAULT_TRAINING)

    diff = subparsers.add_parser(
        "config-diff", help="compare two closed configs as JSON or Markdown"
    )
    diff.add_argument("--kind", choices=("features", "model", "training"), required=True)
    diff.add_argument("--left", type=Path, required=True)
    diff.add_argument("--right", type=Path, required=True)
    diff.add_argument("--format", choices=("json", "markdown"), default="json")

    predictions = subparsers.add_parser(
        "compare-predictions",
        help="compare bounded neural, handcrafted, or validation prediction JSONL",
    )
    predictions.add_argument("--left", type=Path, required=True)
    predictions.add_argument("--right", type=Path, required=True)

    residual = subparsers.add_parser(
        "build-residual-baseline",
        help="bind handcrafted-experimental scores to the approved teacher set",
    )
    residual.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    _add_dataset_arguments(residual, target_options=False)
    residual.add_argument("--engine", type=Path, default=Path("target/release/open-shogi-cli"))
    residual.add_argument("--output", type=Path, required=True)

    for command, help_text in (
        ("arena-run", "run or safely resume the frozen Phase 5 comparison matrix"),
        ("arena-verify", "recompute the Phase 5 arena manifest and Rust-replay every CSA"),
    ):
        arena = subparsers.add_parser(command, help=help_text, description=help_text)
        arena.add_argument("--engine", type=Path, default=Path("target/release/open-shogi-cli"))
        arena.add_argument(
            "--starts",
            type=Path,
            required=True,
        )
        arena.add_argument(
            "--starts-validation",
            type=Path,
            required=True,
        )
        arena.add_argument(
            "--output-root",
            type=Path,
            default=Path("local/runs/model-comparison"),
        )
        arena.add_argument(
            "--f32-model",
            type=Path,
            required=True,
        )
        arena.add_argument(
            "--int8-model",
            type=Path,
            required=True,
        )
        arena.add_argument(
            "--opening-book",
            type=Path,
            required=True,
        )
        arena.add_argument("--git-commit", default=PHASE5_GIT_COMMIT)
        if command == "arena-verify":
            arena.add_argument(
                "--manifest",
                type=Path,
                default=Path("local/runs/model-comparison/arena-manifest.json"),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = _run(arguments)
    except (OSError, RuntimeError, ValueError, FloatingPointError, pickle.UnpicklingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if isinstance(result, str):
        print(result)
    else:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return 0


def _run(arguments: argparse.Namespace) -> dict[str, Any] | str:
    if arguments.command == "build-residual-baseline":
        require_clean_head(Path.cwd())
        training = load_training_config(arguments.training)
        loaded = _load_examples(arguments, training)
        return build_residual_baseline(
            loaded,
            engine=arguments.engine,
            output=arguments.output,
        )
    if arguments.command in {"train", "smoke", "overfit"}:
        require_clean_head(Path.cwd())
        feature, model, training = _load_configs(arguments)
        loaded = _load_examples(arguments, training)
        result = train_model(
            loaded,
            feature,
            model,
            training,
            arguments.output_dir,
            resume_path=arguments.resume,
            mode=arguments.command,
            max_train_batches=2 if arguments.command == "smoke" else None,
            max_validation_batches=2 if arguments.command == "smoke" else None,
            overfit_examples=getattr(arguments, "examples", 32),
            stop_after_epoch=arguments.stop_after_epoch,
        )
        return {
            "schema": "phase4_model_command/v1",
            "command": arguments.command,
            "outputDir": str(result.output_dir),
            "bestCheckpoint": str(result.best_checkpoint),
            "lastCheckpoint": str(result.last_checkpoint),
            "completedEpochs": result.completed_epochs,
            "globalStep": result.global_step,
            "bestValidationLoss": result.best_validation_loss,
            "device": result.device,
        }
    if arguments.command in {"validate", "test"}:
        checkpoint = load_checkpoint(arguments.checkpoint)
        training = parse_training_config(checkpoint["training_config"])
        loaded = _load_examples(
            arguments,
            training,
            include_replay_test=arguments.command == "test",
        )
        split = "validation" if arguments.command == "validate" else "test"
        summary = evaluate_checkpoint(
            arguments.checkpoint,
            loaded,
            split=split,
            predictions_output=arguments.predictions_output,
        )
        return {
            "schema": "phase4_model_command/v1",
            "command": arguments.command,
            "metrics": summary.as_dict(),
            "predictionsOutput": (
                str(arguments.predictions_output) if arguments.predictions_output else None
            ),
        }
    if arguments.command == "compare":
        checkpoint = load_checkpoint(arguments.left)
        training = parse_training_config(checkpoint["training_config"])
        loaded = _load_examples(arguments, training)
        return compare_checkpoints(arguments.left, arguments.right, loaded)
    if arguments.command == "feature-ablation":
        checkpoint = load_checkpoint(arguments.checkpoint)
        training = parse_training_config(checkpoint["training_config"])
        loaded = _load_examples(arguments, training)
        return feature_ablation(arguments.checkpoint, loaded)
    if arguments.command == "export":
        checkpoint, checkpoint_sha256, checkpoint_size = load_checkpoint_with_identity(
            arguments.checkpoint
        )
        feature = parse_feature_config(checkpoint["feature_config"])
        model_config = parse_model_config(checkpoint["model_config"])
        training = parse_training_config(checkpoint["training_config"])
        validate_config_compatibility(model_config, training, feature)
        validate_resume_identity(
            checkpoint,
            config_sha256=combined_config_sha256(feature, model_config, training),
            dataset_identity=checkpoint["dataset_identity"],
        )
        model = ValueModel(input_dimension(feature), model_config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        ensure_finite_model(model, include_gradients=False)
        exported = export_model_artifacts(
            model,
            feature,
            model_config,
            training,
            arguments.output_dir,
            base_name=arguments.base_name,
            provenance={
                "checkpointPath": arguments.checkpoint.name,
                "checkpointSchema": checkpoint["schema"],
                "checkpointSha256": checkpoint_sha256,
                "checkpointSize": checkpoint_size,
                "completedEpoch": checkpoint["completed_epoch"],
                "globalStep": checkpoint["global_step"],
                "bestValidationLoss": checkpoint["best_validation_loss"],
                "configSha256": checkpoint["config_sha256"],
                "datasetIdentity": checkpoint["dataset_identity"],
                "runtime": checkpoint["runtime"],
                "exporterModelCodeSha256": model_code_sha256(),
                "auxiliaryHeadExported": False,
            },
        )
        return {
            "schema": "phase4_model_command/v1",
            "command": "export",
            **{key: value for key, value in exported.items() if key != "metadata"},
        }
    if arguments.command == "describe":
        feature, model, training = _load_configs(arguments)
        counts = parameter_count(input_dimension(feature), model)
        selection = select_device(training.device)
        return {
            "schema": "phase4_model_description/v1",
            "inputDimension": input_dimension(feature),
            "featureFlags": feature_flags(feature),
            "featureGroups": [asdict(group) for group in feature_groups(feature)],
            "featureSchema": feature_schema(feature),
            "parameters": counts,
            "exportedParameterLimit": 16_000_000,
            "exportedParameters": exported_parameter_count(feature, model),
            "estimatedOperationsPerPosition": exported_operation_count(feature, model),
            "estimatedArtifactBytes": {
                mode: estimated_export_bytes(feature, model, mode) for mode in ("float32", "int8")
            },
            "configSha256": combined_config_sha256(feature, model, training),
            "individualConfigSha256": {
                "features": config_sha256(feature),
                "model": config_sha256(model),
                "training": config_sha256(training),
            },
            "device": asdict(selection) | {"device": str(selection.device)},
        }
    if arguments.command == "validate-config":
        feature, model, training = _load_configs(arguments)
        return {
            "schema": "phase4_config_validation/v1",
            "valid": True,
            "configSha256": combined_config_sha256(feature, model, training),
            "expectedTeacherLabels": training.expected_teacher_labels,
            "stageBoundariesBasisPoints": list(training.stage_boundaries_basis_points),
            "inputDimension": input_dimension(feature),
            "exportedParameters": exported_parameter_count(feature, model),
            "estimatedOperationsPerPosition": exported_operation_count(feature, model),
            "estimatedArtifactBytes": {
                mode: estimated_export_bytes(feature, model, mode) for mode in ("float32", "int8")
            },
        }
    if arguments.command == "config-diff":
        difference = _config_diff(arguments.kind, arguments.left, arguments.right)
        return _config_diff_markdown(difference) if arguments.format == "markdown" else difference
    if arguments.command == "compare-predictions":
        return _compare_prediction_artifacts(arguments.left, arguments.right)
    if arguments.command in {"arena-run", "arena-verify"}:
        root = Path.cwd().resolve(strict=True)
        git_commit = arguments.git_commit or require_clean_head(root)
        common = {
            "repository_root": root,
            "engine": arguments.engine,
            "starts_path": arguments.starts,
            "starts_validation_path": arguments.starts_validation,
            "output_root": arguments.output_root,
            "f32_model": arguments.f32_model,
            "int8_model": arguments.int8_model,
            "opening_book": arguments.opening_book,
            "git_commit": git_commit,
        }
        if arguments.command == "arena-run":
            run_phase5_arena(**common)
            return {
                "schema": "phase4_model_command/v1",
                "command": "arena-run",
                "outputRoot": str(arguments.output_root),
            }
        manifest = verify_phase5_arena(**common, manifest_path=arguments.manifest)
        return {
            "schema": "phase4_model_command/v1",
            "command": "arena-verify",
            "manifest": str(arguments.manifest),
            "reportsVerified": manifest["reportsVerified"],
            "csaFilesVerified": manifest["csaFilesVerified"],
        }
    raise AssertionError(f"unhandled command: {arguments.command}")


def _add_config_arguments(parser: argparse.ArgumentParser, training_default: Path) -> None:
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--training", type=Path, default=training_default)


def _add_dataset_arguments(parser: argparse.ArgumentParser, *, target_options: bool = True) -> None:
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path)
    if target_options:
        parser.add_argument(
            "--target-semantics",
            choices=("pure-value", "residual"),
            default="pure-value",
        )
        parser.add_argument("--residual-baseline", type=Path)


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    _add_dataset_arguments(parser)
    parser.add_argument("--predictions-output", type=Path)


def _load_configs(
    arguments: argparse.Namespace,
) -> tuple[FeatureConfig, ModelConfig, TrainingConfig]:
    feature = load_feature_config(arguments.features)
    model = load_model_config(arguments.model)
    training = load_training_config(arguments.training)
    validate_config_compatibility(model, training, feature)
    return feature, model, training


def _load_examples(
    arguments: argparse.Namespace,
    training: TrainingConfig,
    *,
    include_replay_test: bool = False,
):
    return load_training_examples(
        arguments.labels,
        arguments.positions,
        arguments.dataset_manifest,
        training,
        label_manifest_path=arguments.label_manifest,
        replay_manifest_path=arguments.replay_manifest,
        include_replay_test=include_replay_test,
        target_semantics=getattr(arguments, "target_semantics", "pure-value"),
        residual_baseline_path=getattr(arguments, "residual_baseline", None),
        repository_root=Path.cwd(),
    )


def _config_diff(kind: str, left: Path, right: Path) -> dict[str, Any]:
    loaders = {
        "features": load_feature_config,
        "model": load_model_config,
        "training": load_training_config,
    }
    left_value = loaders[kind](left).as_dict()
    right_value = loaders[kind](right).as_dict()
    changes = []
    for key in sorted(set(_flatten(left_value)) | set(_flatten(right_value))):
        left_item = _flatten(left_value).get(key)
        right_item = _flatten(right_value).get(key)
        if left_item != right_item:
            changes.append({"field": key, "left": left_item, "right": right_item})
    return {
        "schema": "phase4_config_diff/v1",
        "kind": kind,
        "leftSha256": config_sha256(loaders[kind](left)),
        "rightSha256": config_sha256(loaders[kind](right)),
        "changes": changes,
    }


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        result.update(_flatten(item, name))
    return result


def _config_diff_markdown(difference: dict[str, Any]) -> str:
    lines = [
        f"# {difference['kind'].title()} configuration difference",
        "",
        f"- Left SHA-256: `{difference['leftSha256']}`",
        f"- Right SHA-256: `{difference['rightSha256']}`",
        "",
        "| Field | Left | Right |",
        "| --- | --- | --- |",
    ]
    for change in difference["changes"]:
        field = _markdown_cell(change["field"])
        left = _markdown_cell(_display_scalar(change["left"]))
        right = _markdown_cell(_display_scalar(change["right"]))
        lines.append(f"| `{field}` | `{left}` | `{right}` |")
    if not difference["changes"]:
        lines.append("| _No changes_ |  |  |")
    return "\n".join(lines)


def _display_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


def _compare_prediction_artifacts(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_identity, left = _load_prediction_rows(left_path)
    right_identity, right = _load_prediction_rows(right_path)
    if len(left) != len(right):
        raise ValueError("prediction artifacts have different row counts")
    deltas: list[int] = []
    right_lower = 0
    right_higher = 0
    ties = 0
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if left_row["index"] != index or right_row["index"] != index:
            raise ValueError("prediction artifact indices must be contiguous from zero")
        if left_row["sfen"] != right_row["sfen"]:
            raise ValueError(f"prediction artifacts differ in SFEN identity at row {index}")
        if (
            left_row["positionId"] is not None
            and right_row["positionId"] is not None
            and left_row["positionId"] != right_row["positionId"]
        ):
            raise ValueError(f"prediction artifacts differ in position identity at row {index}")
        delta = right_row["scoreCp"] - left_row["scoreCp"]
        deltas.append(delta)
        right_lower += delta < 0
        right_higher += delta > 0
        ties += delta == 0
    absolute = [abs(delta) for delta in deltas]
    return {
        "schema": "phase5_prediction_comparison/v1",
        "left": left_identity,
        "right": right_identity,
        "rows": len(deltas),
        "rightMinusLeft": {
            "meanCp": sum(deltas) / len(deltas),
            "meanAbsoluteCp": sum(absolute) / len(absolute),
            "maximumAbsoluteCp": max(absolute),
            "rightLower": right_lower,
            "equal": ties,
            "rightHigher": right_higher,
        },
    }


def _load_prediction_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data, sha256 = _read_regular_file(path, MAX_PREDICTION_COMPARISON_BYTES)
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("prediction artifact must be non-empty and LF terminated")
    rows = data.splitlines()
    if len(rows) > MAX_PREDICTION_COMPARISON_ROWS:
        raise ValueError("prediction artifact exceeds the row bound")
    parsed: list[dict[str, Any]] = []
    schema: str | None = None
    identity: dict[str, Any] | None = None
    for line_number, raw in enumerate(rows, start=1):
        if not raw or len(raw) > 64 * 1024:
            raise ValueError(f"prediction row {line_number} violates the line bound")
        try:
            row = json.loads(
                raw,
                object_pairs_hook=_unique_json_object,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"prediction row {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {line_number} must be an object")
        row_identity, normalized = _validate_prediction_row(row, line_number)
        if schema is None:
            schema = row["schema"]
            identity = row_identity
        elif row["schema"] != schema or row_identity != identity:
            raise ValueError("prediction artifact changes schema or evaluator identity")
        parsed.append(normalized)
    assert schema is not None and identity is not None
    return {
        "path": path.name,
        "sha256": sha256,
        "size": len(data),
        "schema": schema,
        "identity": identity,
    }, parsed


def _validate_prediction_row(
    row: dict[str, Any], line_number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = row.get("schema")
    common = {"schema", "index", "sfen", "scoreCp", "elapsedNs"}
    if schema == "phase5_model_inference/v1":
        expected = common | {"modelArtifactSha256", "modelPayloadSha256"}
        identity_keys = ("modelArtifactSha256", "modelPayloadSha256")
    elif schema == "phase5_handcrafted_inference/v1":
        expected = common | {"evaluatorProfile"}
        identity_keys = ("evaluatorProfile",)
    elif schema == "phase4_value_prediction/v1":
        return _validate_phase4_prediction_row(row, line_number)
    else:
        raise ValueError(f"prediction row {line_number} uses an unsupported schema")
    if set(row) != expected:
        raise ValueError(f"prediction row {line_number} violates its closed schema")
    _bounded_prediction_integer(
        row["index"], 0, MAX_PREDICTION_COMPARISON_ROWS - 1, line_number, "index"
    )
    _bounded_prediction_integer(row["scoreCp"], _I32_MIN, _I32_MAX, line_number, "scoreCp")
    _bounded_prediction_integer(row["elapsedNs"], 0, _U128_MAX, line_number, "elapsedNs")
    if not isinstance(row["sfen"], str):
        raise ValueError(f"prediction row {line_number} SFEN is invalid")
    try:
        parse_canonical_sfen(row["sfen"])
    except ValueError as error:
        raise ValueError(f"prediction row {line_number} SFEN is invalid") from error
    identity = {key: row[key] for key in identity_keys}
    for key, value in identity.items():
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError(f"prediction row {line_number} {key} is invalid")
        if key.endswith("Sha256") and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"prediction row {line_number} {key} is invalid")
    if "evaluatorProfile" in identity and identity["evaluatorProfile"] not in {
        "handcrafted-baseline",
        "handcrafted-experimental",
    }:
        raise ValueError(f"prediction row {line_number} evaluatorProfile is invalid")
    return identity, {
        "index": row["index"],
        "sfen": row["sfen"],
        "scoreCp": row["scoreCp"],
        "positionId": None,
    }


def _validate_phase4_prediction_row(
    row: dict[str, Any], line_number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if frozenset(row) != _PHASE4_PREDICTION_KEYS:
        raise ValueError(f"prediction row {line_number} violates its closed schema")
    for key in ("positionId", "checkpointSha256", "configSha256"):
        value = row[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"prediction row {line_number} {key} is invalid")
    if not isinstance(row["canonicalSfen"], str):
        raise ValueError(f"prediction row {line_number} SFEN is invalid")
    try:
        parse_canonical_sfen(row["canonicalSfen"])
    except ValueError as error:
        raise ValueError(f"prediction row {line_number} SFEN is invalid") from error
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError(f"prediction row {line_number} split is invalid")
    if row["stage"] not in {"opening", "middlegame", "endgame"}:
        raise ValueError(f"prediction row {line_number} stage is invalid")
    score = row["teacherScore"]
    if not isinstance(score, dict) or frozenset(score) != {"kind", "value"}:
        raise ValueError(f"prediction row {line_number} teacherScore is invalid")
    if score["kind"] in {"cp", "mate"}:
        _bounded_prediction_integer(score["value"], _I32_MIN, _I32_MAX, line_number, "teacherScore")
        if score["kind"] == "mate" and score["value"] == 0:
            raise ValueError(f"prediction row {line_number} teacherScore is invalid")
    else:
        raise ValueError(f"prediction row {line_number} teacherScore is invalid")
    _bounded_prediction_integer(
        row["modelCp"], -MAX_NON_MATE_CP, MAX_NON_MATE_CP, line_number, "modelCp"
    )
    if row["candidateGapCp"] is not None:
        _bounded_prediction_integer(
            row["candidateGapCp"], 0, 1_000_000, line_number, "candidateGapCp"
        )
    for key in ("bestmove", "recordedMove"):
        value = row[key]
        if not isinstance(value, str) or _USI_MOVE_RE.fullmatch(value) is None:
            raise ValueError(f"prediction row {line_number} {key} is invalid")
    for key in ("recordedMoveAgrees", "alreadyTeacherLabeled"):
        if not isinstance(row[key], bool):
            raise ValueError(f"prediction row {line_number} {key} is invalid")
    if row["recordedMoveAgrees"] != (row["recordedMove"] == row["bestmove"]):
        raise ValueError(f"prediction row {line_number} recordedMoveAgrees is inconsistent")
    if not row["alreadyTeacherLabeled"]:
        raise ValueError(f"prediction row {line_number} is not teacher-labeled")
    return {
        "checkpointSha256": row["checkpointSha256"],
        "configSha256": row["configSha256"],
    }, {
        "index": line_number - 1,
        "sfen": row["canonicalSfen"],
        "scoreCp": row["modelCp"],
        "positionId": row["positionId"],
    }


def _bounded_prediction_integer(
    value: object,
    minimum: int,
    maximum: int,
    line_number: int,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"prediction row {line_number} {field} is invalid")
    return value


def _read_regular_file(path: Path, maximum: int) -> tuple[bytes, str]:
    try:
        with stable_regular_descriptor(path) as descriptor:
            size = os.fstat(descriptor).st_size
            if not 0 < size <= maximum:
                raise ValueError("prediction artifact size or file type is invalid")
            chunks: list[bytes] = []
            observed = 0
            while observed < size:
                chunk = os.pread(descriptor, min(1024 * 1024, size - observed), observed)
                if not chunk:
                    raise ValueError("prediction artifact changed while reading")
                chunks.append(chunk)
                observed += len(chunk)
            data = b"".join(chunks)
    except ArtifactError as error:
        raise ValueError(
            "prediction artifact must be a readable regular non-symlink file"
        ) from error
    if len(data) > maximum:
        raise ValueError("prediction artifact exceeds the byte bound")
    return data, hashlib.sha256(data).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value
