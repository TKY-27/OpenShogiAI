"""Atomic, identity-bound value_v0 checkpoint save and safe resume."""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import secrets
import stat
from pathlib import Path
from typing import Any

import torch
from torch import nn

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    publish_regular_at,
    retire_bound_regular,
    stable_parent_descriptor,
    stable_regular_descriptor,
)
from open_shogi_training.models.config import (
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    combined_config_sha256,
    parse_feature_config,
    parse_model_config,
    parse_training_config,
    validate_config_compatibility,
)
from open_shogi_training.models.features import input_dimension

CHECKPOINT_SCHEMA = "phase4_value_checkpoint/v1"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_CHECKPOINT_TENSOR_BYTES = 512 * 1024 * 1024
MAX_CHECKPOINT_TREE_ITEMS = 1_000_000
MAX_CHECKPOINT_TREE_DEPTH = 64
MPS_RNG_STATE_BYTES = 44
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DATASET_IDENTITY_KEYS_V1 = {
    "dataset_manifest_sha256",
    "positions_sha256",
    "labels_sha256",
    "label_manifest_sha256",
    "replay_manifest_sha256",
}
_DATASET_IDENTITY_KEYS = _DATASET_IDENTITY_KEYS_V1 | {
    "target_semantics",
    "residual_baseline_sha256",
}
_ADAMW_PARAM_GROUP_KEYS = frozenset(
    {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "params",
    }
)
RUNTIME_KEYS = frozenset(
    {
        "python",
        "torch",
        "platform",
        "device",
        "deviceRequested",
        "deviceFallback",
        "mpsBuilt",
        "mpsAvailable",
        "deterministicAlgorithms",
        "gitCommit",
        "gitDirty",
        "modelCodeSha256",
        "torchNumThreads",
        "torchNumInteropThreads",
    }
)


class _BoundedCheckpointWriter:
    def __init__(self, output: Any) -> None:
        self._output = output
        self._bytes_written = 0

    def write(self, data: bytes | bytearray | memoryview) -> int:
        size = len(data)
        if self._bytes_written + size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the 512 MiB bound")
        written = self._output.write(data)
        self._bytes_written += written
        return written

    def flush(self) -> None:
        self._output.flush()

    def fileno(self) -> int:
        return self._output.fileno()

    def tell(self) -> int:
        return self._bytes_written


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create or replace a checkpoint after fully flushing a sibling temp."""

    with stable_parent_descriptor(path, create=True) as (parent_descriptor, name):
        temporary_name = f".{name}.{secrets.token_hex(12)}"
        descriptor = -1
        temporary_created = False
        temporary_status: os.stat_result | None = None
        try:
            try:
                existing = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ValueError("checkpoint target must be a regular non-symlink file")
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            temporary_created = True
            temporary_status = os.fstat(descriptor)
            with os.fdopen(descriptor, "wb", closefd=False) as output:
                bounded = _BoundedCheckpointWriter(output)
                torch.save(payload, bounded)
                bounded.flush()
                os.fsync(bounded.fileno())
            assert temporary_status is not None
            publish_regular_at(
                parent_descriptor,
                temporary_name,
                name,
                temporary_status,
                display=path,
                replace=True,
            )
            temporary_created = False
            published = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if _file_state(linked) != _file_state(published):
                raise ValueError("checkpoint path changed during publication")
            os.fsync(parent_descriptor)
            final = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if _file_state(linked) != _file_state(final):
                raise ValueError("checkpoint changed before publication completed")
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
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _file_state(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    value, _, _ = load_checkpoint_with_identity(path)
    return value


def load_checkpoint_with_identity(path: Path) -> tuple[dict[str, Any], str, int]:
    """Load only torch's weights-safe primitive/tensor graph and validate the envelope."""

    try:
        with stable_regular_descriptor(path) as descriptor:
            before = os.fstat(descriptor)
            if before.st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint exceeds the 512 MiB bound")
            with os.fdopen(descriptor, "rb", closefd=False) as input_file:
                try:
                    value = torch.load(input_file, map_location="cpu", weights_only=True)
                except Exception as error:
                    raise ValueError("checkpoint cannot be decoded safely") from error
                input_file.seek(0)
                digest = hashlib.sha256()
                observed_size = 0
                while chunk := input_file.read(1024 * 1024):
                    observed_size += len(chunk)
                    if observed_size > MAX_CHECKPOINT_BYTES:
                        raise ValueError("checkpoint exceeds the 512 MiB bound")
                    digest.update(chunk)
                after = os.fstat(input_file.fileno())
    except ArtifactError as error:
        raise ValueError("checkpoint must be a stable regular non-symlink file") from error
    if (
        observed_size != before.st_size
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise ValueError("checkpoint changed while loading")
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is invalid")
    required = {
        "schema",
        "completed_epoch",
        "global_step",
        "best_validation_loss",
        "model_state",
        "optimizer_state",
        "feature_config",
        "model_config",
        "training_config",
        "config_sha256",
        "dataset_identity",
        "rng_state",
        "runtime",
    }
    if set(value) != required:
        raise ValueError(
            f"checkpoint keys mismatch: missing={sorted(required - set(value))}, "
            f"unknown={sorted(set(value) - required)}"
        )
    for key in ("completed_epoch", "global_step"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise ValueError(f"checkpoint {key} must be a non-negative integer")
    best = value["best_validation_loss"]
    if not isinstance(best, float) or not torch.isfinite(torch.tensor(best)).item():
        raise ValueError("checkpoint best_validation_loss must be finite")
    if (
        not isinstance(value["config_sha256"], str)
        or _SHA256_RE.fullmatch(value["config_sha256"]) is None
    ):
        raise ValueError("checkpoint config_sha256 is invalid")
    identity = value["dataset_identity"]
    if not isinstance(identity, dict) or frozenset(identity) not in {
        frozenset(_DATASET_IDENTITY_KEYS_V1),
        frozenset(_DATASET_IDENTITY_KEYS),
    }:
        raise ValueError("checkpoint dataset_identity schema is invalid")
    if set(identity) == _DATASET_IDENTITY_KEYS_V1:
        identity["target_semantics"] = "pure-value"
        identity["residual_baseline_sha256"] = None
    for key in _DATASET_IDENTITY_KEYS - {
        "replay_manifest_sha256",
        "residual_baseline_sha256",
        "target_semantics",
    }:
        if not isinstance(identity[key], str) or _SHA256_RE.fullmatch(identity[key]) is None:
            raise ValueError(f"checkpoint dataset_identity.{key} is invalid")
    replay_hash = identity["replay_manifest_sha256"]
    if replay_hash is not None and (
        not isinstance(replay_hash, str) or _SHA256_RE.fullmatch(replay_hash) is None
    ):
        raise ValueError("checkpoint replay_manifest_sha256 is invalid")
    if identity["target_semantics"] not in {"pure-value", "residual"}:
        raise ValueError("checkpoint target semantics are invalid")
    residual_hash = identity["residual_baseline_sha256"]
    if residual_hash is not None and (
        not isinstance(residual_hash, str) or _SHA256_RE.fullmatch(residual_hash) is None
    ):
        raise ValueError("checkpoint residual_baseline_sha256 is invalid")
    if (identity["target_semantics"] == "residual") != (residual_hash is not None):
        raise ValueError("checkpoint target semantics and residual baseline disagree")
    for key in ("feature_config", "model_config", "training_config", "runtime"):
        if not isinstance(value[key], dict):
            raise ValueError(f"checkpoint {key} must be an object")
    feature = parse_feature_config(value["feature_config"])
    model = parse_model_config(value["model_config"])
    training = parse_training_config(value["training_config"])
    validate_config_compatibility(model, training, feature)
    embedded_config_sha256 = combined_config_sha256(feature, model, training)
    if value["config_sha256"] != embedded_config_sha256:
        raise ValueError("checkpoint config hash does not match its embedded configurations")
    if value["completed_epoch"] > training.epochs:
        raise ValueError("checkpoint completed_epoch exceeds configured epochs")
    _validate_runtime(value["runtime"], training=training)
    if not isinstance(value["model_state"], dict) or not value["model_state"]:
        raise ValueError("checkpoint model_state must be a non-empty object")
    if not isinstance(value["optimizer_state"], dict):
        raise ValueError("checkpoint optimizer_state must be an object")
    if len(value["model_state"]) > 1_024 or any(
        not isinstance(key, str) or not isinstance(item, torch.Tensor)
        for key, item in value["model_state"].items()
    ):
        raise ValueError("checkpoint model_state structure is invalid")
    expected_state = _expected_model_state_shapes(feature, model)
    if set(value["model_state"]) != set(expected_state):
        raise ValueError("checkpoint model_state keys disagree with its architecture")
    for name, expected_shape in expected_state.items():
        observed = value["model_state"][name]
        if (
            tuple(observed.shape) != expected_shape
            or observed.dtype != torch.float32
            or not observed.is_contiguous()
        ):
            raise ValueError(
                f"checkpoint model_state.{name} shape or dtype disagrees with its architecture"
            )
    if set(value["optimizer_state"]) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer_state structure is invalid")
    if not isinstance(value["optimizer_state"]["state"], dict) or not isinstance(
        value["optimizer_state"]["param_groups"], list
    ):
        raise ValueError("checkpoint optimizer_state structure is invalid")
    _validate_optimizer_state(
        value["optimizer_state"], expected_state, training, value["global_step"]
    )
    budget = _TensorTreeBudget()
    _ensure_finite_tensor_tree(value["model_state"], "model_state", budget=budget)
    _ensure_finite_tensor_tree(value["optimizer_state"], "optimizer_state", budget=budget)
    _validate_rng_state(value["rng_state"], runtime=value["runtime"])
    return value, digest.hexdigest(), observed_size


def _expected_model_state_shapes(
    feature: FeatureConfig, model: ModelConfig
) -> dict[str, tuple[int, ...]]:
    """Describe the closed ValueModel state without constructing or randomizing a model."""

    shapes: dict[str, tuple[int, ...]] = {}
    current_dimension = input_dimension(feature)
    for layer_index in range(model.hidden_layers):
        shapes[f"trunk.{layer_index}.weight"] = (model.hidden_dim, current_dimension)
        shapes[f"trunk.{layer_index}.bias"] = (model.hidden_dim,)
        current_dimension = model.hidden_dim
    for head in ("value_head", "policy_agreement_head"):
        shapes[f"{head}.weight"] = (1, current_dimension)
        shapes[f"{head}.bias"] = (1,)
    return shapes


def _validate_optimizer_state(
    optimizer_state: dict[str, Any],
    expected_model_state: dict[str, tuple[int, ...]],
    training: TrainingConfig,
    global_step: int,
) -> None:
    groups = optimizer_state["param_groups"]
    if len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError("checkpoint optimizer must contain exactly one AdamW parameter group")
    group = groups[0]
    if frozenset(group) != _ADAMW_PARAM_GROUP_KEYS:
        raise ValueError("checkpoint AdamW parameter-group schema is invalid")
    parameter_ids = group["params"]
    expected_ids = list(range(len(expected_model_state)))
    if parameter_ids != expected_ids:
        raise ValueError("checkpoint AdamW parameter identities disagree with the model")
    expected_options = {
        "lr": training.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": training.weight_decay,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    if any(group[key] != expected for key, expected in expected_options.items()):
        raise ValueError("checkpoint AdamW options disagree with the training contract")
    state = optimizer_state["state"]
    if any(isinstance(key, bool) or not isinstance(key, int) for key in state):
        raise ValueError("checkpoint AdamW state contains an invalid parameter identity")
    if not set(state).issubset(expected_ids):
        raise ValueError("checkpoint AdamW state references an unknown model parameter")
    expected_state_ids = set(expected_ids) if global_step > 0 else set()
    if set(state) != expected_state_ids:
        raise ValueError("checkpoint AdamW state disagrees with global_step")
    expected_shapes = tuple(expected_model_state.values())
    for parameter_id, slots in state.items():
        if not isinstance(slots, dict) or frozenset(slots) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("checkpoint AdamW slot schema is invalid")
        step = slots["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.dtype != torch.float32
            or step.device.type != "cpu"
            or step.layout != torch.strided
            or step.shape != torch.Size([])
            or not step.is_contiguous()
            or not math.isfinite(float(step.item()))
            or step.item() != global_step
        ):
            raise ValueError("checkpoint AdamW step slot disagrees with global_step")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = slots[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tuple(tensor.shape) != expected_shapes[parameter_id]
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"checkpoint AdamW {name} slot is invalid")


def validate_resume_identity(
    checkpoint: dict[str, Any],
    *,
    config_sha256: str,
    dataset_identity: dict[str, str | None],
) -> None:
    if checkpoint["config_sha256"] != config_sha256:
        raise ValueError("resume checkpoint config hash does not match")
    expected_identity = dict(dataset_identity)
    if set(expected_identity) == _DATASET_IDENTITY_KEYS_V1:
        expected_identity["target_semantics"] = "pure-value"
        expected_identity["residual_baseline_sha256"] = None
    if checkpoint["dataset_identity"] != expected_identity:
        raise ValueError("resume checkpoint dataset identities do not match")


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": None,
    }
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: object, device: torch.device) -> None:
    state = _validate_rng_state(state)
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "mps":
        mps_state = state["torch_mps"]
        if mps_state is None:
            raise ValueError("checkpoint lacks MPS RNG state required for MPS resume")
        torch.mps.set_rng_state(mps_state)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer tensor slots to the selected execution device."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def build_checkpoint(
    *,
    completed_epoch: int,
    global_step: int,
    best_validation_loss: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    feature_config: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    config_sha256: str,
    dataset_identity: dict[str, str | None],
    device: torch.device,
    runtime: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "best_validation_loss": float(best_validation_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_config": feature_config,
        "model_config": model_config,
        "training_config": training_config,
        "config_sha256": config_sha256,
        "dataset_identity": dataset_identity,
        "rng_state": capture_rng_state(device),
        "runtime": runtime,
    }


def _validate_rng_state(state: object, *, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != {"python", "torch_cpu", "torch_mps"}:
        raise ValueError("checkpoint RNG state is invalid")
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.device.type != "cpu"
        or cpu_state.layout != torch.strided
        or cpu_state.ndim != 1
        or cpu_state.numel() != torch.get_rng_state().numel()
    ):
        raise ValueError("checkpoint CPU RNG state is invalid")
    mps_state = state["torch_mps"]
    if mps_state is not None and (
        not isinstance(mps_state, torch.Tensor)
        or mps_state.dtype != torch.uint8
        or mps_state.device.type != "cpu"
        or mps_state.layout != torch.strided
        or mps_state.ndim != 1
        or mps_state.numel() != MPS_RNG_STATE_BYTES
    ):
        raise ValueError("checkpoint MPS RNG state is invalid")
    if runtime is not None:
        expects_mps_state = runtime.get("device") == "mps"
        if expects_mps_state != (mps_state is not None):
            raise ValueError("checkpoint MPS RNG state disagrees with its runtime device")
    python_state = state["python"]
    if not isinstance(python_state, tuple) or len(python_state) != 3 or python_state[0] != 3:
        raise ValueError("checkpoint Python RNG state is invalid")
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    return state


class _TensorTreeBudget:
    def __init__(self) -> None:
        self.items = 0
        self.tensor_bytes = 0


def _ensure_finite_tensor_tree(
    value: object,
    context: str,
    *,
    budget: _TensorTreeBudget,
    depth: int = 0,
) -> None:
    if depth > MAX_CHECKPOINT_TREE_DEPTH:
        raise ValueError(f"checkpoint {context} exceeds the nesting bound")
    budget.items += 1
    if budget.items > MAX_CHECKPOINT_TREE_ITEMS:
        raise ValueError("checkpoint tensor tree exceeds the item bound")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"checkpoint {context} tensor layout or device is invalid")
        budget.tensor_bytes += value.numel() * value.element_size()
        if budget.tensor_bytes > MAX_CHECKPOINT_TENSOR_BYTES:
            raise ValueError("checkpoint tensors exceed the 512 MiB bound")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"checkpoint {context} contains a non-finite tensor")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str | int):
                raise ValueError(f"checkpoint {context} contains an invalid mapping key")
            _ensure_finite_tensor_tree(
                item,
                f"{context}.{key}",
                budget=budget,
                depth=depth + 1,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite_tensor_tree(
                item,
                f"{context}[{index}]",
                budget=budget,
                depth=depth + 1,
            )
        return
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)).item():
        raise ValueError(f"checkpoint {context} contains a non-finite number")
    if value is not None and not isinstance(value, bool | int | float | str):
        raise ValueError(f"checkpoint {context} contains an unsupported value")


def _validate_runtime(runtime: dict[str, Any], *, training: TrainingConfig | None = None) -> None:
    if frozenset(runtime) != RUNTIME_KEYS:
        raise ValueError("checkpoint runtime schema is invalid")
    for key, item in runtime.items():
        if not isinstance(item, str) or not item or len(item.encode("utf-8")) > 4_096:
            raise ValueError(f"checkpoint runtime.{key} is invalid")
    if _SHA256_RE.fullmatch(runtime["modelCodeSha256"]) is None:
        raise ValueError("checkpoint runtime.modelCodeSha256 is invalid")
    if runtime["device"] not in {"cpu", "mps"}:
        raise ValueError("checkpoint runtime.device is invalid")
    if runtime["deviceRequested"] not in {"auto", "cpu", "mps"}:
        raise ValueError("checkpoint runtime.deviceRequested is invalid")
    if runtime["mpsBuilt"] not in {"true", "false"} or runtime["mpsAvailable"] not in {
        "true",
        "false",
    }:
        raise ValueError("checkpoint runtime MPS availability is invalid")
    if runtime["deterministicAlgorithms"] not in {"true", "false"}:
        raise ValueError("checkpoint runtime deterministicAlgorithms is invalid")
    if (
        training is not None
        and runtime["deterministicAlgorithms"] != str(training.deterministic).lower()
    ):
        raise ValueError("checkpoint runtime deterministicAlgorithms disagrees with config")
    if runtime["gitDirty"] not in {"true", "false", "unknown"}:
        raise ValueError("checkpoint runtime.gitDirty is invalid")
    git_commit = runtime["gitCommit"]
    if git_commit != "unknown" and (
        len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("checkpoint runtime.gitCommit is invalid")
    for key in ("torchNumThreads", "torchNumInteropThreads"):
        try:
            count = int(runtime[key])
        except ValueError as error:
            raise ValueError(f"checkpoint runtime.{key} is invalid") from error
        if str(count) != runtime[key] or not 1 <= count <= 1_000_000:
            raise ValueError(f"checkpoint runtime.{key} is invalid")
    if runtime["mpsAvailable"] == "true" and runtime["mpsBuilt"] != "true":
        raise ValueError("checkpoint runtime MPS availability is inconsistent")
    if runtime["device"] == "mps" and (
        runtime["mpsBuilt"] != "true"
        or runtime["mpsAvailable"] != "true"
        or runtime["deviceRequested"] == "cpu"
        or runtime["deviceFallback"] != "none"
    ):
        raise ValueError("checkpoint runtime MPS device fields are inconsistent")
    if runtime["deviceRequested"] == "cpu" and runtime["deviceFallback"] != "none":
        raise ValueError("checkpoint runtime CPU request cannot have a fallback")
    if (
        runtime["device"] == "cpu"
        and runtime["deviceRequested"] in {"auto", "mps"}
        and (runtime["deviceFallback"] == "none" or runtime["mpsAvailable"] == "true")
    ):
        raise ValueError("checkpoint runtime CPU fallback fields are inconsistent")
