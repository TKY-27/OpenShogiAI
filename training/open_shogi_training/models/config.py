"""Closed, versioned configuration schemas for the Phase 4 value model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tomllib
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from open_shogi_training.labeling.artifacts import ArtifactError, stable_regular_descriptor

CONFIG_SCHEMA_VERSION = 1
QUANTIZATION_MODES = frozenset({"float32", "int8", "both"})
DEVICE_MODES = frozenset({"auto", "cpu", "mps"})
# OSAVAL architecture v1 is intentionally closed to the activation implemented
# bit-for-bit by both the Python exporter and the independent Rust consumer.
ACTIVATIONS = frozenset({"relu"})
MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Feature-group switches for the deterministic SFEN encoder."""

    name: str
    board_planes: bool
    hand_counts: bool
    side_to_move: bool
    king_coordinates: bool
    pseudo_attacks: bool

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": CONFIG_SCHEMA_VERSION, "features": asdict(self)}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Uniform-width MLP architecture and exported score scale."""

    name: str
    hidden_layers: int
    hidden_dim: int
    activation: str
    dropout: float
    output_scale_cp: float
    auxiliary_policy_head: bool

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": CONFIG_SCHEMA_VERSION, "model": asdict(self)}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization, sampling, target, reproducibility, and export settings."""

    name: str
    teacher_loss_weight: float
    game_result_loss_weight: float
    policy_agreement_loss_weight: float
    ranking_loss_weight: float
    ranking_margin: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    sample_ratio: float
    stage_ratios: tuple[float, float, float]
    stage_boundaries_basis_points: tuple[int, int]
    expected_teacher_labels: int
    teacher_clip_cp: float
    teacher_normalization_cp: float
    seed: int
    device: str
    deterministic: bool
    quantization: str
    gradient_clip_norm: float
    checkpoint_every_epochs: int

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["stage_ratios"] = list(self.stage_ratios)
        values["stage_boundaries_basis_points"] = list(self.stage_boundaries_basis_points)
        return {"schema_version": CONFIG_SCHEMA_VERSION, "training": values}


_FEATURE_KEYS = frozenset(
    {
        "name",
        "board_planes",
        "hand_counts",
        "side_to_move",
        "king_coordinates",
        "pseudo_attacks",
    }
)
_MODEL_KEYS = frozenset(
    {
        "name",
        "hidden_layers",
        "hidden_dim",
        "activation",
        "dropout",
        "output_scale_cp",
        "auxiliary_policy_head",
    }
)
_TRAINING_KEYS = frozenset(
    {
        "name",
        "teacher_loss_weight",
        "game_result_loss_weight",
        "policy_agreement_loss_weight",
        "ranking_loss_weight",
        "ranking_margin",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "sample_ratio",
        "stage_ratios",
        "stage_boundaries_basis_points",
        "expected_teacher_labels",
        "teacher_clip_cp",
        "teacher_normalization_cp",
        "seed",
        "device",
        "deterministic",
        "quantization",
        "gradient_clip_norm",
        "checkpoint_every_epochs",
    }
)


def load_feature_config(path: Path) -> FeatureConfig:
    """Load a feature TOML file and reject every unknown or missing value."""

    return parse_feature_config(_load_toml(path))


def load_model_config(path: Path) -> ModelConfig:
    """Load a model TOML file and reject every unknown or missing value."""

    return parse_model_config(_load_toml(path))


def load_training_config(path: Path) -> TrainingConfig:
    """Load a training TOML file and reject every unknown or missing value."""

    return parse_training_config(_load_toml(path))


def parse_feature_config_bytes(raw: bytes, context: str) -> FeatureConfig:
    return parse_feature_config(_parse_toml_bytes(raw, context))


def parse_model_config_bytes(raw: bytes, context: str) -> ModelConfig:
    return parse_model_config(_parse_toml_bytes(raw, context))


def parse_training_config_bytes(raw: bytes, context: str) -> TrainingConfig:
    return parse_training_config(_parse_toml_bytes(raw, context))


def parse_feature_config(raw: dict[str, Any]) -> FeatureConfig:
    table = _root_table(raw, "features")
    _require_exact_keys(table, _FEATURE_KEYS, "features")
    config = FeatureConfig(
        name=_string(table, "name"),
        board_planes=_boolean(table, "board_planes"),
        hand_counts=_boolean(table, "hand_counts"),
        side_to_move=_boolean(table, "side_to_move"),
        king_coordinates=_boolean(table, "king_coordinates"),
        pseudo_attacks=_boolean(table, "pseudo_attacks"),
    )
    if not any(
        (
            config.board_planes,
            config.hand_counts,
            config.side_to_move,
            config.king_coordinates,
            config.pseudo_attacks,
        )
    ):
        raise ValueError("at least one feature group must be enabled")
    return config


def parse_model_config(raw: dict[str, Any]) -> ModelConfig:
    table = _root_table(raw, "model")
    _require_exact_keys(table, _MODEL_KEYS, "model")
    config = ModelConfig(
        name=_string(table, "name"),
        hidden_layers=_integer(table, "hidden_layers"),
        hidden_dim=_integer(table, "hidden_dim"),
        activation=_string(table, "activation"),
        dropout=_number(table, "dropout"),
        output_scale_cp=_number(table, "output_scale_cp"),
        auxiliary_policy_head=_boolean(table, "auxiliary_policy_head"),
    )
    if not 1 <= config.hidden_layers <= 16:
        raise ValueError("hidden_layers must be between 1 and 16")
    if not 1 <= config.hidden_dim <= 8_192:
        raise ValueError("hidden_dim must be between 1 and 8192")
    if config.activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.output_scale_cp <= 0.0:
        raise ValueError("output_scale_cp must be positive")
    try:
        serialized_scale = struct.unpack("<f", struct.pack("<f", config.output_scale_cp))[0]
    except OverflowError as error:
        raise ValueError("output_scale_cp must be representable as positive float32") from error
    if not math.isfinite(serialized_scale) or serialized_scale <= 0.0:
        raise ValueError("output_scale_cp must be representable as positive float32")
    if not config.auxiliary_policy_head:
        raise ValueError("value_v0 requires the policy-agreement auxiliary head")
    return config


def parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    table = _root_table(raw, "training")
    _require_exact_keys(table, _TRAINING_KEYS, "training")
    stage_ratios = _number_tuple(table, "stage_ratios", length=3)
    stage_boundaries = _integer_tuple(table, "stage_boundaries_basis_points", length=2)
    config = TrainingConfig(
        name=_string(table, "name"),
        teacher_loss_weight=_number(table, "teacher_loss_weight"),
        game_result_loss_weight=_number(table, "game_result_loss_weight"),
        policy_agreement_loss_weight=_number(table, "policy_agreement_loss_weight"),
        ranking_loss_weight=_number(table, "ranking_loss_weight"),
        ranking_margin=_number(table, "ranking_margin"),
        learning_rate=_number(table, "learning_rate"),
        weight_decay=_number(table, "weight_decay"),
        batch_size=_integer(table, "batch_size"),
        epochs=_integer(table, "epochs"),
        sample_ratio=_number(table, "sample_ratio"),
        stage_ratios=stage_ratios,
        stage_boundaries_basis_points=stage_boundaries,
        expected_teacher_labels=_integer(table, "expected_teacher_labels"),
        teacher_clip_cp=_number(table, "teacher_clip_cp"),
        teacher_normalization_cp=_number(table, "teacher_normalization_cp"),
        seed=_integer(table, "seed"),
        device=_string(table, "device"),
        deterministic=_boolean(table, "deterministic"),
        quantization=_string(table, "quantization"),
        gradient_clip_norm=_number(table, "gradient_clip_norm"),
        checkpoint_every_epochs=_integer(table, "checkpoint_every_epochs"),
    )
    weights = (
        config.teacher_loss_weight,
        config.game_result_loss_weight,
        config.policy_agreement_loss_weight,
        config.ranking_loss_weight,
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("loss weights must be non-negative")
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("at least one loss weight must be positive")
    if config.ranking_margin < 0.0:
        raise ValueError("ranking_margin must be non-negative")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if not 1 <= config.batch_size <= 1_048_576:
        raise ValueError("batch_size must be between 1 and 1048576")
    if not 1 <= config.epochs <= 1_000_000:
        raise ValueError("epochs must be between 1 and 1000000")
    if not 0.0 < config.sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    if any(ratio < 0.0 for ratio in config.stage_ratios) or not math.isclose(
        sum(config.stage_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("stage_ratios must be non-negative and sum to 1")
    opening_end, middlegame_end = config.stage_boundaries_basis_points
    if not 0 < opening_end < middlegame_end < 10_000:
        raise ValueError(
            "stage_boundaries_basis_points must satisfy 0 < opening < middlegame < 10000"
        )
    if config.expected_teacher_labels != 10_000:
        raise ValueError("expected_teacher_labels must equal the goal-wide 10000-label set")
    if config.teacher_clip_cp <= 0.0 or config.teacher_normalization_cp <= 0.0:
        raise ValueError("teacher clipping and normalization must be positive")
    if config.teacher_clip_cp < config.teacher_normalization_cp:
        raise ValueError("teacher_clip_cp must be at least teacher_normalization_cp")
    if not 0 <= config.seed <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63-1")
    if config.device not in DEVICE_MODES:
        raise ValueError(f"device must be one of {sorted(DEVICE_MODES)}")
    if config.quantization not in QUANTIZATION_MODES:
        raise ValueError(f"quantization must be one of {sorted(QUANTIZATION_MODES)}")
    if config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if config.checkpoint_every_epochs <= 0:
        raise ValueError("checkpoint_every_epochs must be positive")
    return config


def validate_config_compatibility(
    model: ModelConfig,
    training: TrainingConfig,
    feature: FeatureConfig | None = None,
) -> None:
    """Reject score scales that would change meaning between training and export."""

    if not math.isclose(
        model.output_scale_cp,
        training.teacher_normalization_cp,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("model output_scale_cp must equal training teacher_normalization_cp")
    if feature is not None and exported_parameter_count(feature, model) > 16_000_000:
        raise ValueError("exported model must contain at most 16000000 parameters")


def configured_input_dimension(feature: FeatureConfig) -> int:
    """Calculate v1 dimension here so capacity validation has no import cycle."""

    return (
        (2 * 14 * 81 if feature.board_planes else 0)
        + (2 * 7 if feature.hand_counts else 0)
        + (1 if feature.side_to_move else 0)
        + (4 if feature.king_coordinates else 0)
        + (2 * 81 if feature.pseudo_attacks else 0)
    )


def exported_parameter_count(feature: FeatureConfig, model: ModelConfig) -> int:
    input_dim = configured_input_dimension(feature)
    trunk = input_dim * model.hidden_dim + model.hidden_dim
    trunk += (model.hidden_layers - 1) * (model.hidden_dim * model.hidden_dim + model.hidden_dim)
    return trunk + model.hidden_dim + 1


def exported_operation_count(feature: FeatureConfig, model: ModelConfig) -> int:
    """Return dense multiply-adds plus biases and hidden activations per position."""

    parameters = exported_parameter_count(feature, model)
    hidden_activations = model.hidden_layers * model.hidden_dim
    return parameters + hidden_activations


def estimated_export_bytes(feature: FeatureConfig, model: ModelConfig, quantization: str) -> int:
    """Calculate the exact OSAVAL01 artifact size for the configured representation."""

    if quantization not in {"float32", "int8"}:
        raise ValueError("export byte estimate quantization must be float32 or int8")
    dimensions = (
        [configured_input_dimension(feature)] + [model.hidden_dim] * model.hidden_layers + [1]
    )
    header_bytes = 8 + 10 * 4 + 4
    checksum_bytes = 32
    layer_bytes = 0
    for input_dim, output_dim in pairwise(dimensions):
        weights = input_dim * output_dim
        layer_bytes += 8 + output_dim * 4
        layer_bytes += weights * (4 if quantization == "float32" else 1)
        if quantization == "int8":
            layer_bytes += 4
    return header_bytes + layer_bytes + checksum_bytes


def config_sha256(config: FeatureConfig | ModelConfig | TrainingConfig) -> str:
    """Hash the resolved typed configuration, independent of TOML formatting."""

    payload = json.dumps(
        config.as_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combined_config_sha256(
    feature: FeatureConfig,
    model: ModelConfig,
    training: TrainingConfig,
) -> str:
    payload = {
        "features": feature.as_dict(),
        "model": model.as_dict(),
        "training": training.as_dict(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with stable_regular_descriptor(path) as descriptor:
            size = os.fstat(descriptor).st_size
            if not 0 < size <= MAX_CONFIG_BYTES:
                raise ValueError("configuration file type or size is invalid")
            data = _read_descriptor_bytes(descriptor, size)
    except ArtifactError as error:
        raise ValueError("configuration must be a readable regular non-symlink file") from error
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("configuration exceeds the 1 MiB bound")
    return _parse_toml_bytes(data, str(path))


def _parse_toml_bytes(data: bytes, context: str) -> dict[str, Any]:
    if not 0 < len(data) <= MAX_CONFIG_BYTES:
        raise ValueError(f"configuration type or size is invalid: {context}")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"configuration is not valid UTF-8 TOML: {context}") from error
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a table")
    return raw


def _read_descriptor_bytes(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while observed < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - observed), observed)
        if not chunk:
            raise ValueError("configuration changed while loading")
        chunks.append(chunk)
        observed += len(chunk)
    return b"".join(chunks)


def _root_table(raw: dict[str, Any], table_name: str) -> dict[str, Any]:
    _require_exact_keys(raw, frozenset({"schema_version", table_name}), "root")
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("schema_version must be an integer")
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {version}")
    table = raw[table_name]
    if not isinstance(table, dict):
        raise ValueError(f"{table_name} must be a TOML table")
    return table


def _require_exact_keys(table: dict[str, Any], expected: frozenset[str], location: str) -> None:
    actual = frozenset(table)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{location} keys mismatch: missing={missing}, unknown={unknown}")


def _string(table: dict[str, Any], key: str) -> str:
    value = table[key]
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
        raise ValueError(f"{key} must be a non-empty string of at most 256 bytes")
    return value


def _boolean(table: dict[str, Any], key: str) -> bool:
    value = table[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _integer(table: dict[str, Any], key: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(table: dict[str, Any], key: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _number_tuple(table: dict[str, Any], key: str, *, length: int) -> tuple[float, ...]:
    value = table[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must be an array of length {length}")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{key} must contain only numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"{key} must contain only finite numbers")
        result.append(converted)
    return tuple(result)


def _integer_tuple(table: dict[str, Any], key: str, *, length: int) -> tuple[int, ...]:
    value = table[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must be an array of length {length}")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{key} must contain only integers")
        result.append(item)
    return tuple(result)
