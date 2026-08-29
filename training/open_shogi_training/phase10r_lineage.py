"""Fail-closed lineage validation for Phase 10R teacher-bound candidates."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
import torch

from open_shogi_training.labeling.config import load_teacher_config
from open_shogi_training.labeling.fingerprint import fingerprint_teacher
from open_shogi_training.phase10r import _load_yaml
from open_shogi_training.phase10r_model import parse_osaval02
from open_shogi_training.phase10r_training import CHECKPOINT_SCHEMA

CONTROL_PATH: Final = Path("configs/phase10r/teacher-binding.yaml")
CONTROL_SHA256: Final = "d587e8d06d86af6dc10cd1a844532f29912e4f11da53f7e659e2a8a0f131d9ff"
CONTROL_SCHEMA: Final = "open_shogiai_phase10r_teacher_binding/v1"
LINEAGE_SCHEMA: Final = "open_shogiai_phase10r_candidate_lineage/v1"
CALIBRATION_INPUT_SCHEMA: Final = "open_shogiai_phase10r_teacher_calibration_input/v1"
PARITY_RECEIPT_SCHEMA: Final = "open_shogiai_phase10r_candidate_parity/v1"
CALIBRATION_RECEIPT_SCHEMA: Final = "open_shogiai_phase10r_teacher_calibration/v1"
BINDING_VERSION: Final = "teacher-bound-v1"
VARIANTS: Final = (
    "sparse-pair-policy-wdl",
    "factorized-pair-triple-policy-score",
)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_UTC_RE: Final = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z\Z"
)


class Phase10RLineageError(ValueError):
    """Raised when candidate lineage cannot prove the frozen teacher binding."""


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Phase10RLineageError(f"{context} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], context: str) -> None:
    if set(value) != keys:
        raise Phase10RLineageError(f"{context} keys are invalid")


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Phase10RLineageError(f"{context} must be a lowercase SHA-256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise Phase10RLineageError("lineage value is not canonical JSON") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Phase10RLineageError(f"cannot hash lineage artifact: {path}") from error
    return digest.hexdigest()


def _contained_regular(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise Phase10RLineageError(f"{context}.path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise Phase10RLineageError(f"{context}.path must be repository-relative")
    path = root.joinpath(*relative.parts)
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise Phase10RLineageError(
            f"{context}.path is unavailable or escapes the repository"
        ) from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise Phase10RLineageError(f"{context}.path must be a regular non-symlink file")
    return resolved


def _load_json(path: Path, *, maximum_bytes: int, context: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Phase10RLineageError(f"cannot read {context}") from error
    if not raw or len(raw) > maximum_bytes:
        raise Phase10RLineageError(f"{context} size is invalid")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Phase10RLineageError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Phase10RLineageError(f"{context} is not strict JSON") from error
    return _require_mapping(value, context)


def _verify_control_ref(root: Path, raw: object, expected: Mapping[str, Any], context: str) -> Path:
    reference = _require_mapping(raw, context)
    _require_exact_keys(reference, {"path", "sha256"}, context)
    if reference != expected:
        raise Phase10RLineageError(f"{context} differs from the frozen control")
    path = _contained_regular(root, reference["path"], context)
    if _sha256_file(path) != _require_sha256(reference["sha256"], f"{context}.sha256"):
        raise Phase10RLineageError(f"{context} SHA-256 mismatch")
    return path


def _verify_artifact_ref(root: Path, raw: object, context: str) -> tuple[Path, Mapping[str, Any]]:
    reference = _require_mapping(raw, context)
    _require_exact_keys(reference, {"path", "sha256", "bytes"}, context)
    path = _contained_regular(root, reference["path"], context)
    expected_size = reference["bytes"]
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        raise Phase10RLineageError(f"{context}.bytes must be positive")
    if path.stat().st_size != expected_size:
        raise Phase10RLineageError(f"{context} byte length mismatch")
    if _sha256_file(path) != _require_sha256(reference["sha256"], f"{context}.sha256"):
        raise Phase10RLineageError(f"{context} SHA-256 mismatch")
    return path, reference


def _control_identity(binding: Mapping[str, Any]) -> str:
    identity = dict(binding)
    declared = _require_sha256(identity.pop("identity_sha256", None), "identity_sha256")
    actual = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    if actual != declared:
        raise Phase10RLineageError("teacher binding identity SHA-256 mismatch")
    return actual


def _validate_exact_teacher(root: Path, control: Mapping[str, Any]) -> None:
    binding = _require_mapping(control["teacher_binding_identity"], "teacher_binding_identity")
    _control_identity(binding)
    teacher = _require_mapping(binding["teacher"], "teacher_binding_identity.teacher")
    config_path = _contained_regular(root, teacher["config_path"], "teacher config")
    if _sha256_file(config_path) != teacher["config_file_sha256"]:
        raise Phase10RLineageError("teacher config file SHA-256 mismatch")
    config = load_teacher_config(config_path)
    if config.sha256 != teacher["config_semantic_sha256"]:
        raise Phase10RLineageError("teacher config semantic SHA-256 mismatch")
    fingerprint = fingerprint_teacher(config, root)
    expected_record = {
        "name": teacher["name"],
        "version": teacher["version"],
        "binary": teacher["binary"],
        "eval_files": teacher["eval_files"],
        "options": teacher["options"],
    }
    if fingerprint.identity_record() != expected_record:
        raise Phase10RLineageError("teacher files or options differ from the frozen identity")
    _verify_control_ref(
        root, teacher["install_manifest"], teacher["install_manifest"], "install manifest"
    )
    search = _require_mapping(teacher["search"], "teacher search")
    if search != {
        "nodes": 25_000,
        "multipv": 3,
        "threads": 4,
        "hash_mib": 1024,
        "concurrency": 1,
    }:
        raise Phase10RLineageError("teacher search contract changed")

    semantics = _require_mapping(binding["score_semantics"], "score_semantics")
    target_path = _contained_regular(root, semantics["target_semantics_path"], "target semantics")
    if _sha256_file(target_path) != semantics["target_semantics_sha256"]:
        raise Phase10RLineageError("target semantics SHA-256 mismatch")
    if semantics["perspective"] != "current_side_to_move_at_root":
        raise Phase10RLineageError("teacher score perspective changed")

    labels = _require_mapping(binding["labels"], "labels")
    manifest_path = _verify_control_ref(
        root, labels["manifest"], labels["manifest"], "label manifest"
    )
    labels_path = _verify_control_ref(
        root,
        {"path": labels["rows"]["path"], "sha256": labels["rows"]["sha256"]},
        {"path": labels["rows"]["path"], "sha256": labels["rows"]["sha256"]},
        "label rows",
    )
    _verify_control_ref(root, labels["benchmark"], labels["benchmark"], "teacher benchmark")
    manifest = _load_json(manifest_path, maximum_bytes=4 * 1024 * 1024, context="label manifest")
    if (
        manifest.get("schema") != "phase4_teacher_label_manifest/v2"
        or manifest.get("teacher") != fingerprint.identity_record()
        or manifest.get("artifacts", {}).get("labels.jsonl", {}).get("sha256")
        != labels["rows"]["sha256"]
        or labels_path.stat().st_size
        != manifest.get("artifacts", {}).get("labels.jsonl", {}).get("size")
        or manifest.get("progress", {}).get("status") != "complete"
        or manifest.get("progress", {}).get("completed") != labels["rows"]["records"]
    ):
        raise Phase10RLineageError("label manifest does not prove the frozen complete label set")


def _validate_pretraining(root: Path, control: Mapping[str, Any]) -> None:
    pretraining = _require_mapping(control["pretraining"], "pretraining")
    preparation = _require_mapping(pretraining["preparation_manifest"], "preparation manifest")
    preparation_path = _contained_regular(root, preparation["path"], "preparation manifest")
    if _sha256_file(preparation_path) != preparation["file_sha256"]:
        raise Phase10RLineageError("preparation manifest file SHA-256 mismatch")
    preparation_payload = _load_json(
        preparation_path, maximum_bytes=16 * 1024 * 1024, context="preparation manifest"
    )
    if (
        preparation_payload.get("schema") != "open_shogiai_phase10r_preparation/v2"
        or preparation_payload.get("status") != "passed"
        or preparation_payload.get("scale") != "1m"
        or preparation_payload.get("manifest_sha256") != preparation["manifest_sha256"]
    ):
        raise Phase10RLineageError("preparation manifest identity is invalid")

    variants = pretraining["variants"]
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        raise Phase10RLineageError("pretraining variants must be an array")
    if [item.get("variant_id") for item in variants if isinstance(item, Mapping)] != list(VARIANTS):
        raise Phase10RLineageError("pretraining variant order or identity changed")
    for raw in variants:
        variant = _require_mapping(raw, "pretraining variant")
        variant_id = variant["variant_id"]
        receipt_path = _verify_control_ref(
            root, variant["training_receipt"], variant["training_receipt"], f"{variant_id} receipt"
        )
        checkpoint_path = _verify_control_ref(
            root,
            variant["stage2_checkpoint"],
            variant["stage2_checkpoint"],
            f"{variant_id} stage2 checkpoint",
        )
        artifact_path = _verify_control_ref(
            root,
            variant["stage2_artifact"],
            variant["stage2_artifact"],
            f"{variant_id} stage2 artifact",
        )
        receipt = _load_json(
            receipt_path, maximum_bytes=4 * 1024 * 1024, context="training receipt"
        )
        result = receipt.get("result")
        stages = result.get("stages") if isinstance(result, Mapping) else None
        if (
            receipt.get("status") != "passed"
            or not isinstance(result, Mapping)
            or not isinstance(stages, list)
            or not stages
            or not isinstance(stages[-1], Mapping)
            or result.get("variant") != variant_id
            or result.get("manifest_sha256") != preparation["manifest_sha256"]
            or result.get("teacher_dependent_stages") != "not_started"
            or stages[-1].get("checkpoint_sha256") != variant["stage2_checkpoint"]["sha256"]
            or result.get("artifact", {}).get("sha256") != variant["stage2_artifact"]["sha256"]
        ):
            raise Phase10RLineageError("training receipt does not prove a stage2-only parent")
        if checkpoint_path.stat().st_size <= 0:
            raise Phase10RLineageError("stage2 checkpoint is empty")
        parsed = parse_osaval02(artifact_path.read_bytes())
        if (
            parsed.variant_id != variant_id
            or parsed.dataset_manifest_sha256 != preparation["manifest_sha256"]
        ):
            raise Phase10RLineageError("stage2 OSAVAL02 identity disagrees with its parent")


def load_teacher_binding_control(root: Path) -> Mapping[str, Any]:
    """Load and verify every current file pinned by the teacher-binding freeze."""

    repository = root.resolve(strict=True)
    control_path = repository / CONTROL_PATH
    if _sha256_file(control_path) != CONTROL_SHA256:
        raise Phase10RLineageError("teacher-binding control SHA-256 changed")
    control = _load_yaml(control_path)
    _require_exact_keys(
        control,
        {
            "schema",
            "frozen_as_of",
            "repair_case",
            "binding_version",
            "scale",
            "pretraining",
            "teacher_binding_identity",
            "calibration",
            "acceptance",
        },
        "teacher-binding control",
    )
    if (
        control["schema"] != CONTROL_SCHEMA
        or control["repair_case"] != "execution_sequence_skipped_existing_teacher_binding_stage"
        or control["binding_version"] != BINDING_VERSION
        or control["scale"] != "1m"
    ):
        raise Phase10RLineageError("teacher-binding control identity changed")
    calibration = _require_mapping(control["calibration"], "calibration")
    if (
        calibration.get("expected_rows")
        != {"train": 6570, "validation": 1880, "validation_cp_for_affine_fit": 1831}
        or calibration.get("label_budget")
        != {"rung_cap": 10000, "existing_labels": 10000, "new_teacher_calls": 0}
        or calibration.get("forbidden_splits")
        != ["public_test", "internal_test", "source_held_out", "final_holdout"]
    ):
        raise Phase10RLineageError("teacher calibration population or label budget changed")
    _validate_exact_teacher(repository, control)
    _validate_pretraining(repository, control)
    return control


def _variant_control(control: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
    for raw in control["pretraining"]["variants"]:
        if isinstance(raw, Mapping) and raw.get("variant_id") == variant_id:
            return raw
    raise Phase10RLineageError(f"variant is absent from teacher-binding control: {variant_id}")


def _expected_ref(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = _contained_regular(root, reference["path"], "frozen reference")
    return {
        "path": reference["path"],
        "sha256": reference["sha256"],
        "bytes": path.stat().st_size,
    }


def _validate_calibration_input(
    root: Path,
    value: Mapping[str, Any],
    *,
    control_sha256: str,
    identity_sha256: str,
    preparation_manifest_sha256: str,
    label_manifest_sha256: str,
    labels_sha256: str,
) -> None:
    _require_exact_keys(
        value,
        {
            "schema",
            "status",
            "binding_version",
            "scale",
            "control_sha256",
            "teacher_binding_identity_sha256",
            "preparation_manifest_sha256",
            "label_manifest_sha256",
            "labels_sha256",
            "rows",
            "excluded_rows",
            "new_teacher_calls",
            "files",
        },
        "calibration input manifest",
    )
    if (
        value["schema"] != CALIBRATION_INPUT_SCHEMA
        or value["status"] != "passed"
        or value["binding_version"] != BINDING_VERSION
        or value["scale"] != "1m"
        or value["control_sha256"] != control_sha256
        or value["teacher_binding_identity_sha256"] != identity_sha256
        or value["preparation_manifest_sha256"] != preparation_manifest_sha256
        or value["label_manifest_sha256"] != label_manifest_sha256
        or value["labels_sha256"] != labels_sha256
        or value["rows"] != {"train": 6570, "validation": 1880, "validation_cp": 1831}
        or value["excluded_rows"]
        != {"phase4_test": 1064, "absent_from_current_preparation": 467, "split_mismatch": 19}
        or value["new_teacher_calls"] != 0
    ):
        raise Phase10RLineageError("calibration input manifest violates the frozen population")
    files = _require_mapping(value["files"], "calibration input files")
    _require_exact_keys(files, {"train", "validation"}, "calibration input files")
    for name in ("train", "validation"):
        _verify_artifact_ref(root, files[name], f"calibration input {name}")


def _load_stage3_checkpoint(
    path: Path,
    *,
    variant_id: str,
    input_manifest_sha256: str,
    parent_checkpoint_sha256: str,
    teacher_identity_sha256: str,
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise Phase10RLineageError("stage3 checkpoint cannot be decoded") from error
    checkpoint = _require_mapping(checkpoint, "stage3 checkpoint")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("variant_id") != variant_id
        or checkpoint.get("manifest_sha256") != input_manifest_sha256
        or checkpoint.get("stage_id") != "packed_sfen_value_and_ranking"
        or checkpoint.get("completed") is not True
        or checkpoint.get("parent_checkpoint_sha256") != parent_checkpoint_sha256
        or checkpoint.get("teacher_binding_identity_sha256") != teacher_identity_sha256
        or not isinstance(checkpoint.get("model_state"), Mapping)
    ):
        raise Phase10RLineageError("stage3 checkpoint identity or completion is invalid")
    return checkpoint


def _validate_stage4_calibration(
    value: Mapping[str, Any],
    *,
    variant_id: str,
    input_manifest_sha256: str,
    stage3_checkpoint_sha256: str,
    teacher_identity_sha256: str,
) -> tuple[float, float]:
    _require_exact_keys(
        value,
        {
            "schema",
            "status",
            "variant_id",
            "input_manifest_sha256",
            "stage3_checkpoint_sha256",
            "teacher_binding_identity_sha256",
            "method",
            "fit_rows",
            "calibration_scale",
            "calibration_bias",
            "new_teacher_calls",
        },
        "stage4 calibration",
    )
    scale = value["calibration_scale"]
    bias = value["calibration_bias"]
    if (
        value["schema"] != CALIBRATION_RECEIPT_SCHEMA
        or value["status"] != "passed"
        or value["variant_id"] != variant_id
        or value["input_manifest_sha256"] != input_manifest_sha256
        or value["stage3_checkpoint_sha256"] != stage3_checkpoint_sha256
        or value["teacher_binding_identity_sha256"] != teacher_identity_sha256
        or value["method"] != "positive_monotonic_affine"
        or value["fit_rows"] != 1831
        or value["new_teacher_calls"] != 0
        or not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or not isinstance(bias, (int, float))
        or isinstance(bias, bool)
        or not math.isfinite(float(bias))
    ):
        raise Phase10RLineageError("stage4 calibration identity or fit is invalid")
    return float(scale), float(bias)


def _validate_exported_weights(checkpoint: Mapping[str, Any], parsed: Any) -> None:
    state = _require_mapping(checkpoint["model_state"], "stage3 model state")
    if set(state) != set(parsed.tensors):
        raise Phase10RLineageError("candidate tensors differ from the stage3 checkpoint")
    for name, view in parsed.tensors.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise Phase10RLineageError("stage3 model state contains a non-tensor value")
        encoded = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype(np.dtype("<f4"), copy=False)
            .tobytes()
        )
        if encoded != view.data.tobytes():
            raise Phase10RLineageError(
                f"candidate tensor {name} differs from the stage3 checkpoint"
            )


def _same_exported_weights(first: Any, second: Any) -> bool:
    if set(first.tensors) != set(second.tensors):
        return False
    return all(
        first.tensors[name].data.tobytes() == second.tensors[name].data.tobytes()
        for name in first.tensors
    )


def validate_candidate_lineage(root: Path, path: Path) -> Mapping[str, Any]:
    """Validate one completed lineage and every referenced artifact against current bytes."""

    repository = root.resolve(strict=True)
    control = load_teacher_binding_control(repository)
    lineage_path = path if path.is_absolute() else repository / path
    try:
        lineage_relative = lineage_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise Phase10RLineageError("candidate lineage path escapes the repository") from error
    lineage_path = _contained_regular(
        repository,
        lineage_relative,
        "candidate lineage",
    )
    lineage = _load_json(lineage_path, maximum_bytes=4 * 1024 * 1024, context="candidate lineage")
    _require_exact_keys(
        lineage,
        {
            "schema",
            "status",
            "binding_version",
            "scale",
            "variant_id",
            "created_at_utc",
            "git_commit",
            "parent",
            "teacher_binding",
            "stages",
            "candidate",
            "acceptance",
        },
        "candidate lineage",
    )
    variant_id = lineage["variant_id"]
    if (
        lineage["schema"] != LINEAGE_SCHEMA
        or lineage["status"] != "completed"
        or lineage["binding_version"] != BINDING_VERSION
        or lineage["scale"] != "1m"
        or variant_id not in VARIANTS
        or not isinstance(lineage["created_at_utc"], str)
        or _UTC_RE.fullmatch(lineage["created_at_utc"]) is None
        or not isinstance(lineage["git_commit"], str)
        or _COMMIT_RE.fullmatch(lineage["git_commit"]) is None
    ):
        raise Phase10RLineageError("candidate lineage identity is invalid")

    variant = _variant_control(control, variant_id)
    parent = _require_mapping(lineage["parent"], "lineage parent")
    _require_exact_keys(
        parent,
        {"preparation_manifest", "training_receipt", "stage2_checkpoint", "stage2_artifact"},
        "lineage parent",
    )
    preparation = control["pretraining"]["preparation_manifest"]
    expected_parent = {
        "preparation_manifest": {
            "path": preparation["path"],
            "sha256": preparation["file_sha256"],
            "bytes": _contained_regular(repository, preparation["path"], "preparation")
            .stat()
            .st_size,
        },
        "training_receipt": _expected_ref(repository, variant["training_receipt"]),
        "stage2_checkpoint": _expected_ref(repository, variant["stage2_checkpoint"]),
        "stage2_artifact": _expected_ref(repository, variant["stage2_artifact"]),
    }
    if parent != expected_parent:
        raise Phase10RLineageError(
            "candidate lineage parent differs from the immutable pretraining parent"
        )
    for name, reference in parent.items():
        _verify_artifact_ref(repository, reference, f"lineage parent {name}")

    binding = _require_mapping(lineage["teacher_binding"], "lineage teacher binding")
    _require_exact_keys(
        binding,
        {
            "identity_sha256",
            "control",
            "label_manifest",
            "labels",
            "benchmark",
            "calibration_input_manifest",
        },
        "lineage teacher binding",
    )
    frozen_binding = control["teacher_binding_identity"]
    if binding["identity_sha256"] != frozen_binding["identity_sha256"]:
        raise Phase10RLineageError("candidate teacher identity differs from the frozen identity")
    control_path = repository / CONTROL_PATH
    expected_control_ref = {
        "path": CONTROL_PATH.as_posix(),
        "sha256": _sha256_file(control_path),
        "bytes": control_path.stat().st_size,
    }
    labels = frozen_binding["labels"]
    expected_binding = {
        "identity_sha256": frozen_binding["identity_sha256"],
        "control": expected_control_ref,
        "label_manifest": _expected_ref(repository, labels["manifest"]),
        "labels": {
            "path": labels["rows"]["path"],
            "sha256": labels["rows"]["sha256"],
            "bytes": _contained_regular(repository, labels["rows"]["path"], "labels")
            .stat()
            .st_size,
        },
        "benchmark": _expected_ref(repository, labels["benchmark"]),
    }
    for name in ("control", "label_manifest", "labels", "benchmark"):
        if binding[name] != expected_binding[name]:
            raise Phase10RLineageError(f"lineage teacher binding {name} changed")
        _verify_artifact_ref(repository, binding[name], f"lineage teacher binding {name}")
    calibration_path, calibration_ref = _verify_artifact_ref(
        repository, binding["calibration_input_manifest"], "calibration input manifest"
    )
    calibration = _load_json(
        calibration_path, maximum_bytes=4 * 1024 * 1024, context="calibration input manifest"
    )
    _validate_calibration_input(
        repository,
        calibration,
        control_sha256=expected_control_ref["sha256"],
        identity_sha256=frozen_binding["identity_sha256"],
        preparation_manifest_sha256=preparation["manifest_sha256"],
        label_manifest_sha256=labels["manifest"]["sha256"],
        labels_sha256=labels["rows"]["sha256"],
    )

    stages = lineage["stages"]
    if not isinstance(stages, list) or len(stages) != 2:
        raise Phase10RLineageError("candidate lineage must contain exactly stages 3 and 4")
    expected_stages = ((3, "packed_sfen_value_and_ranking"), (4, "approved_teacher_calibration"))
    output_directory = Path(
        control["calibration"]["output"]["directory_template"].format(
            scale="1m", variant=variant_id
        )
    )
    stage_paths: list[Path] = []
    stage_references: list[Mapping[str, Any]] = []
    for index, (order, stage_id) in enumerate(expected_stages):
        stage = _require_mapping(stages[index], f"lineage stage {order}")
        _require_exact_keys(
            stage,
            {"order", "stage_id", "status", "input_manifest", "output"},
            f"lineage stage {order}",
        )
        if (
            stage["order"] != order
            or stage["stage_id"] != stage_id
            or stage["status"] != "passed"
            or stage["input_manifest"] != calibration_ref
        ):
            raise Phase10RLineageError(f"lineage stage {order} identity is invalid")
        output_path, output_reference = _verify_artifact_ref(
            repository, stage["output"], f"lineage stage {order} output"
        )
        try:
            output_path.relative_to(repository / output_directory)
        except ValueError as error:
            raise Phase10RLineageError(
                f"lineage stage {order} output escapes its versioned directory"
            ) from error
        stage_paths.append(output_path)
        stage_references.append(output_reference)

    stage3_checkpoint = _load_stage3_checkpoint(
        stage_paths[0],
        variant_id=variant_id,
        input_manifest_sha256=calibration_ref["sha256"],
        parent_checkpoint_sha256=variant["stage2_checkpoint"]["sha256"],
        teacher_identity_sha256=frozen_binding["identity_sha256"],
    )
    stage4_value = _load_json(
        stage_paths[1], maximum_bytes=4 * 1024 * 1024, context="stage4 calibration"
    )
    stage4_scale, stage4_bias = _validate_stage4_calibration(
        stage4_value,
        variant_id=variant_id,
        input_manifest_sha256=calibration_ref["sha256"],
        stage3_checkpoint_sha256=stage_references[0]["sha256"],
        teacher_identity_sha256=frozen_binding["identity_sha256"],
    )

    candidate = _require_mapping(lineage["candidate"], "lineage candidate")
    _require_exact_keys(candidate, {"artifact", "osaval02", "parity_receipt"}, "lineage candidate")
    artifact_path, artifact_ref = _verify_artifact_ref(
        repository, candidate["artifact"], "teacher-bound artifact"
    )
    try:
        artifact_path.relative_to(repository / output_directory)
    except ValueError as error:
        raise Phase10RLineageError(
            "teacher-bound artifact escapes its versioned directory"
        ) from error
    if artifact_ref["sha256"] == variant["stage2_artifact"]["sha256"]:
        raise Phase10RLineageError("teacher-bound artifact retained the pretraining-only weights")
    parsed = parse_osaval02(artifact_path.read_bytes())
    parent_artifact_path = _contained_regular(
        repository, variant["stage2_artifact"]["path"], "pretraining artifact"
    )
    parent_parsed = parse_osaval02(parent_artifact_path.read_bytes())
    if _same_exported_weights(parsed, parent_parsed):
        raise Phase10RLineageError("teacher-bound artifact only changed pretraining metadata")
    osaval = _require_mapping(candidate["osaval02"], "candidate OSAVAL02 metadata")
    _require_exact_keys(
        osaval,
        {"variant_id", "dataset_manifest_sha256", "training_run_reference"},
        "candidate OSAVAL02 metadata",
    )
    if (
        parsed.variant_id != variant_id
        or parsed.quantization != "float32"
        or parsed.variant_id != osaval["variant_id"]
        or parsed.dataset_manifest_sha256 != calibration_ref["sha256"]
        or parsed.dataset_manifest_sha256 != osaval["dataset_manifest_sha256"]
        or parsed.training_run_reference != osaval["training_run_reference"]
        or osaval["training_run_reference"] != f"phase10r-1m-{variant_id}-teacher-bound-v1"
    ):
        raise Phase10RLineageError("teacher-bound OSAVAL02 metadata is not lineage-bound")
    _validate_exported_weights(stage3_checkpoint, parsed)

    parity_path, _ = _verify_artifact_ref(
        repository, candidate["parity_receipt"], "candidate parity receipt"
    )
    parity = _load_json(
        parity_path, maximum_bytes=4 * 1024 * 1024, context="candidate parity receipt"
    )
    _require_exact_keys(
        parity,
        {
            "schema",
            "status",
            "variant_id",
            "artifact_sha256",
            "python_native_wasm",
            "incremental_full_recompute_unmake",
        },
        "candidate parity receipt",
    )
    if (
        parity.get("schema") != PARITY_RECEIPT_SCHEMA
        or parity.get("status") != "passed"
        or parity.get("variant_id") != variant_id
        or parity.get("artifact_sha256") != artifact_ref["sha256"]
        or parity.get("python_native_wasm") != "passed"
        or parity.get("incremental_full_recompute_unmake") != "passed"
    ):
        raise Phase10RLineageError("candidate parity receipt is incomplete or mismatched")

    acceptance = _require_mapping(lineage["acceptance"], "lineage acceptance")
    _require_exact_keys(
        acceptance,
        {
            "status",
            "parent_immutable",
            "new_teacher_calls",
            "forbidden_split_rows",
            "calibration_scale",
            "calibration_bias",
            "python_native_wasm_parity",
            "incremental_parity",
            "source_held_out_regression_maximum",
            "calibration_ece_regression",
            "new_tactical_failures",
            "throughput_floor_passed",
        },
        "lineage acceptance",
    )
    scale = acceptance["calibration_scale"]
    bias = acceptance["calibration_bias"]
    regression = acceptance["source_held_out_regression_maximum"]
    ece = acceptance["calibration_ece_regression"]
    if (
        acceptance["status"] != "passed"
        or acceptance["parent_immutable"] is not True
        or acceptance["new_teacher_calls"] != 0
        or acceptance["forbidden_split_rows"] != 0
        or not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or float(scale) != float(parsed.calibration_scale)
        or float(scale) != stage4_scale
        or not isinstance(bias, (int, float))
        or isinstance(bias, bool)
        or not math.isfinite(float(bias))
        or float(bias) != float(parsed.calibration_bias)
        or float(bias) != stage4_bias
        or acceptance["python_native_wasm_parity"] != "passed"
        or acceptance["incremental_parity"] != "passed"
        or not isinstance(regression, (int, float))
        or isinstance(regression, bool)
        or not 0.0 <= float(regression) <= 0.01
        or not isinstance(ece, (int, float))
        or isinstance(ece, bool)
        or not math.isfinite(float(ece))
        or not -1.0 <= float(ece) <= 0.01
        or acceptance["new_tactical_failures"] != 0
        or acceptance["throughput_floor_passed"] is not True
    ):
        raise Phase10RLineageError("candidate acceptance gates did not all pass")
    return lineage


def lineage_path(root: Path, scale: str, variant_id: str) -> Path:
    if scale != "1m" or variant_id not in VARIANTS:
        raise Phase10RLineageError("unsupported teacher-binding lineage target")
    return (
        root
        / "local/phase10r-data/checkpoints/phase10r"
        / scale
        / variant_id
        / BINDING_VERSION
        / "candidate-lineage.json"
    )


def completed_teacher_bound_candidates(root: Path, scale: str) -> list[Mapping[str, Any]]:
    """Return only fully validated candidates; an invalid present lineage stops closed."""

    control = load_teacher_binding_control(root)
    if scale != control["scale"]:
        raise Phase10RLineageError("teacher binding is frozen only for the 1m rung")
    completed: list[Mapping[str, Any]] = []
    for variant_id in VARIANTS:
        path = lineage_path(root, scale, variant_id)
        if not path.exists() and not path.is_symlink():
            continue
        completed.append(validate_candidate_lineage(root, path))
    return completed


__all__ = [
    "BINDING_VERSION",
    "CALIBRATION_INPUT_SCHEMA",
    "CONTROL_PATH",
    "CONTROL_SCHEMA",
    "CONTROL_SHA256",
    "LINEAGE_SCHEMA",
    "PARITY_RECEIPT_SCHEMA",
    "VARIANTS",
    "Phase10RLineageError",
    "completed_teacher_bound_candidates",
    "lineage_path",
    "load_teacher_binding_control",
    "validate_candidate_lineage",
]
