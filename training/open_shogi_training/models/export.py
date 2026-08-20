"""Strict OSAVAL01 float/int8 export and dependency-free reference inference."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
    config_sha256,
    exported_parameter_count,
    validate_config_compatibility,
)
from open_shogi_training.models.features import (
    FEATURE_ATTACKS,
    FEATURE_BOARD,
    FEATURE_HANDS,
    FEATURE_KINGS,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SIDE_TO_MOVE,
    feature_flags,
    feature_schema,
    input_dimension,
)

MAGIC = b"OSAVAL01"
FORMAT_VERSION = 1
ARCH_VERSION = 1
QUANTIZATION_FLOAT32 = 0
QUANTIZATION_INT8 = 1
ACTIVATION_RELU = 0
_HEADER = struct.Struct("<8s10If")
_U32_PAIR = struct.Struct("<II")
_F32 = struct.Struct("<f")
_F32_MAX = 3.4028234663852886e38
_SHA256_BYTES = 32
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_PARAMETERS = 16_000_000
MAX_NON_MATE_CP = 28_999
_KNOWN_FEATURE_FLAGS = (
    FEATURE_BOARD | FEATURE_HANDS | FEATURE_SIDE_TO_MOVE | FEATURE_KINGS | FEATURE_ATTACKS
)
_BASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EXPORT_FAILPOINT_ENV = "OPEN_SHOGI_EXPORT_FAILPOINT"


@dataclass(frozen=True, slots=True)
class ExportLayer:
    """One parsed row-major dense layer."""

    input_dim: int
    output_dim: int
    weights: tuple[float | int, ...]
    biases: tuple[float, ...]
    scale: float | None


@dataclass(frozen=True, slots=True)
class ExportedValueModel:
    """Validated OSAVAL01 header, layers, and content identity."""

    feature_flags: int
    input_dim: int
    hidden_layers: int
    hidden_dim: int
    activation: int
    quantization: int
    output_scale_cp: float
    layers: tuple[ExportLayer, ...]
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    quantization: str
    sha256: str
    size: int
    payload_sha256: str


def serialize_value_model(
    model: Any,
    feature_config: FeatureConfig,
    model_config: ModelConfig,
    *,
    quantization: Literal["float32", "int8"],
) -> bytes:
    """Serialize the inference trunk plus value head, never the auxiliary head."""

    if quantization not in {"float32", "int8"}:
        raise ValueError("quantization must be float32 or int8")
    expected_input = input_dimension(feature_config)
    if exported_parameter_count(feature_config, model_config) > _MAX_PARAMETERS:
        raise ValueError("model exceeds the 16000000-parameter export bound")
    if getattr(model, "input_dim", None) != expected_input:
        raise ValueError("model input dimension disagrees with the feature schema")
    layers = tuple(model.export_layers())
    if len(layers) != model_config.hidden_layers + 1:
        raise ValueError("model layer count disagrees with its architecture config")
    quantization_code = QUANTIZATION_FLOAT32 if quantization == "float32" else QUANTIZATION_INT8
    if model_config.activation != "relu":
        raise ValueError("OSAVAL architecture version 1 supports only relu activation")
    activation_code = ACTIVATION_RELU
    chunks = [
        _HEADER.pack(
            MAGIC,
            FORMAT_VERSION,
            ARCH_VERSION,
            FEATURE_SCHEMA_VERSION,
            feature_flags(feature_config),
            expected_input,
            model_config.hidden_layers,
            model_config.hidden_dim,
            activation_code,
            quantization_code,
            len(layers),
            model_config.output_scale_cp,
        )
    ]
    expected_in = expected_input
    for layer_index, layer in enumerate(layers):
        weights, biases, input_dim, output_dim = _linear_values(layer)
        expected_out = model_config.hidden_dim if layer_index < model_config.hidden_layers else 1
        if input_dim != expected_in or output_dim != expected_out:
            raise ValueError(f"layer {layer_index} dimensions disagree with architecture")
        chunks.append(_U32_PAIR.pack(input_dim, output_dim))
        if quantization_code == QUANTIZATION_FLOAT32:
            chunks.append(struct.pack(f"<{len(weights)}f", *weights))
        else:
            maximum = max((abs(value) for value in weights), default=0.0)
            scale = _as_float32(maximum / 127.0 if maximum > 0.0 else 1.0)
            if scale == 0.0:
                scale = _as_float32(2.0**-149)
            quantized = [max(-127, min(127, round(value / scale))) for value in weights]
            chunks.append(_F32.pack(scale))
            chunks.append(struct.pack(f"<{len(quantized)}b", *quantized))
        chunks.append(struct.pack(f"<{len(biases)}f", *biases))
        expected_in = output_dim
    payload = b"".join(chunks)
    result = payload + hashlib.sha256(payload).digest()
    if len(result) > _MAX_MODEL_BYTES:
        raise ValueError("serialized model exceeds the 64 MiB artifact bound")
    return result


def parse_value_model(data: bytes) -> ExportedValueModel:
    """Strictly validate and parse one OSAVAL01 byte string."""

    if not isinstance(data, bytes):
        raise TypeError("model data must be bytes")
    if len(data) > _MAX_MODEL_BYTES:
        raise ValueError("model exceeds the 64 MiB parser bound")
    if len(data) < _HEADER.size + _SHA256_BYTES:
        raise ValueError("model is truncated")
    payload, stored_digest = data[:-_SHA256_BYTES], data[-_SHA256_BYTES:]
    computed_digest = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(stored_digest, computed_digest):
        raise ValueError("model trailing SHA-256 does not match its payload")
    (
        magic,
        format_version,
        arch_version,
        schema_version,
        flags,
        input_dim,
        hidden_layers,
        hidden_dim,
        activation,
        quantization,
        layer_count,
        output_scale_cp,
    ) = _HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("model magic is not OSAVAL01")
    if format_version != FORMAT_VERSION or arch_version != ARCH_VERSION:
        raise ValueError("unsupported model format or architecture version")
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported feature schema version")
    if flags == 0 or flags & ~_KNOWN_FEATURE_FLAGS:
        raise ValueError("model contains unknown or empty feature flags")
    if not 1 <= input_dim <= 1_000_000:
        raise ValueError("model input dimension is outside the supported bound")
    if input_dim != _input_dimension_from_flags(flags):
        raise ValueError("model input dimension disagrees with feature flags")
    if not 1 <= hidden_layers <= 16 or not 1 <= hidden_dim <= 8_192:
        raise ValueError("model hidden architecture is outside the supported bound")
    if activation != ACTIVATION_RELU:
        raise ValueError("OSAVAL architecture version 1 supports only relu activation")
    if quantization not in {QUANTIZATION_FLOAT32, QUANTIZATION_INT8}:
        raise ValueError("model quantization code is invalid")
    if layer_count != hidden_layers + 1:
        raise ValueError("model layer count is inconsistent")
    if not math.isfinite(output_scale_cp) or output_scale_cp <= 0.0:
        raise ValueError("model output scale must be finite and positive")

    offset = _HEADER.size
    expected_in = input_dim
    layers: list[ExportLayer] = []
    parameter_count = 0
    for layer_index in range(layer_count):
        input_size, output_size = _unpack_required(_U32_PAIR, payload, offset)
        offset += _U32_PAIR.size
        expected_out = hidden_dim if layer_index < hidden_layers else 1
        if input_size != expected_in or output_size != expected_out:
            raise ValueError(f"model layer {layer_index} has inconsistent dimensions")
        weight_count = _checked_element_count(input_size, output_size)
        parameter_count += weight_count + output_size
        if parameter_count > _MAX_PARAMETERS:
            raise ValueError("model exceeds the 16000000-parameter bound")
        scale: float | None = None
        if quantization == QUANTIZATION_FLOAT32:
            weights, offset = _read_float32s(payload, offset, weight_count)
        else:
            (scale,) = _unpack_required(_F32, payload, offset)
            offset += _F32.size
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"model layer {layer_index} has an invalid int8 scale")
            end = offset + weight_count
            if end > len(payload):
                raise ValueError("model is truncated in int8 weights")
            weights = tuple(struct.unpack_from(f"<{weight_count}b", payload, offset))
            offset = end
        biases, offset = _read_float32s(payload, offset, output_size)
        if any(not math.isfinite(float(value)) for value in weights) or any(
            not math.isfinite(value) for value in biases
        ):
            raise ValueError(f"model layer {layer_index} contains non-finite values")
        layers.append(
            ExportLayer(
                input_dim=input_size,
                output_dim=output_size,
                weights=weights,
                biases=biases,
                scale=scale,
            )
        )
        expected_in = output_size
    if offset != len(payload):
        raise ValueError("model contains trailing bytes before its SHA-256")
    return ExportedValueModel(
        feature_flags=flags,
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        activation=activation,
        quantization=quantization,
        output_scale_cp=output_scale_cp,
        layers=tuple(layers),
        payload_sha256=computed_digest.hex(),
    )


def load_value_model(path: Path) -> ExportedValueModel:
    try:
        with stable_regular_descriptor(path) as descriptor:
            before = os.fstat(descriptor)
            if before.st_size > _MAX_MODEL_BYTES:
                raise ValueError("model exceeds the 64 MiB parser bound")
            with os.fdopen(descriptor, "rb", closefd=False) as input_file:
                data = input_file.read(_MAX_MODEL_BYTES + 1)
                after = os.fstat(input_file.fileno())
    except ArtifactError as error:
        raise ValueError("model must be a stable regular non-symlink file") from error
    if len(data) > _MAX_MODEL_BYTES:
        raise ValueError("model exceeds the 64 MiB parser bound")
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("model changed while loading")
    return parse_value_model(data)


def infer_normalized(model: ExportedValueModel, features: list[float]) -> float:
    """Run dependency-free dense inference and return current-side normalized value."""

    if len(features) != model.input_dim:
        raise ValueError("feature vector dimension does not match the exported model")
    if any(not math.isfinite(value) for value in features):
        raise ValueError("feature vector contains a non-finite value")
    activations = [_finite_to_float32(value) for value in features]
    for layer_index, layer in enumerate(model.layers):
        output: list[float] = []
        for output_index in range(layer.output_dim):
            start = output_index * layer.input_dim
            total = layer.biases[output_index]
            if layer.scale is None:
                total += sum(
                    float(layer.weights[start + index]) * value
                    for index, value in enumerate(activations)
                )
            else:
                total += layer.scale * sum(
                    int(layer.weights[start + index]) * value
                    for index, value in enumerate(activations)
                )
            if not math.isfinite(total):
                raise FloatingPointError("exported inference produced a non-finite value")
            if layer_index < model.hidden_layers:
                total = max(0.0, total)
            output.append(_finite_to_float32(total))
        activations = output
    if len(activations) != 1:
        raise RuntimeError("exported value head did not produce one scalar")
    return activations[0]


def infer_centipawns(model: ExportedValueModel, features: list[float]) -> int:
    """Return current-side CP, rounded half away from zero and outside mate space."""

    return normalized_to_centipawns(infer_normalized(model, features), model.output_scale_cp)


def normalized_to_centipawns(normalized: float, output_scale_cp: float) -> int:
    if not math.isfinite(normalized) or not math.isfinite(output_scale_cp) or output_scale_cp <= 0:
        raise ValueError("normalized value and positive output scale must be finite")
    scaled = normalized * output_scale_cp
    rounded = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
    return max(-MAX_NON_MATE_CP, min(MAX_NON_MATE_CP, rounded))


def export_model_artifacts(
    model: Any,
    feature_config: FeatureConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    base_name: str = "value_v0",
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write configured binary variants, feature schema, and binding metadata."""

    if _BASE_NAME.fullmatch(base_name) is None:
        raise ValueError("base_name must be a safe portable filename stem")
    validate_config_compatibility(model_config, training_config, feature_config)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("export provenance must be a non-empty object")
    schema_value = feature_schema(feature_config)
    schema_path = output_dir / f"{base_name}.feature-schema.json"
    schema_payload = _json_payload(schema_value)
    schema_identity = (hashlib.sha256(schema_payload).hexdigest(), len(schema_payload))

    modes: tuple[tuple[Literal["float32", "int8"], str, bool], ...] = (
        ("float32", "f32", training_config.quantization in {"float32", "both"}),
        ("int8", "int8", training_config.quantization in {"int8", "both"}),
    )
    artifacts: list[ArtifactIdentity] = []
    artifact_payloads: list[tuple[Path, bytes]] = []
    for mode, suffix, enabled in modes:
        if not enabled:
            continue
        data = serialize_value_model(
            model,
            feature_config,
            model_config,
            quantization=mode,
        )
        path = output_dir / f"{base_name}.{suffix}.osaval"
        parsed = parse_value_model(data)
        sha256, size = hashlib.sha256(data).hexdigest(), len(data)
        artifacts.append(
            ArtifactIdentity(
                path=path.name,
                quantization=mode,
                sha256=sha256,
                size=size,
                payload_sha256=parsed.payload_sha256,
            )
        )
        artifact_payloads.append((path, data))
    metadata = {
        "schema": "phase4_value_model_metadata/v1",
        "format": {
            "magic": MAGIC.decode("ascii"),
            "formatVersion": FORMAT_VERSION,
            "architectureVersion": ARCH_VERSION,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
            "endianness": "little",
            "weightOrder": "output_major_row_major",
        },
        "model": {
            "name": model_config.name,
            "inputDimension": input_dimension(feature_config),
            "hiddenLayers": model_config.hidden_layers,
            "hiddenDimension": model_config.hidden_dim,
            "activation": model_config.activation,
            "dropoutTrainingOnly": model_config.dropout,
            "outputScaleCp": model_config.output_scale_cp,
            "outputPerspective": "current_side_to_move",
            "centipawnConversion": {
                "rounding": "nearest_half_away_from_zero",
                "clamp": [-MAX_NON_MATE_CP, MAX_NON_MATE_CP],
            },
            "valueHead": {"trained": True, "exported": True},
            "policyAgreementAuxiliaryHead": {
                "trained": model_config.auxiliary_policy_head,
                "target": "recordedMove_equals_teacherBestmove",
                "exported": False,
            },
        },
        "featureSchema": {
            "path": schema_path.name,
            "sha256": schema_identity[0],
            "size": schema_identity[1],
            "configSha256": config_sha256(feature_config),
            "featureFlags": feature_flags(feature_config),
        },
        "configs": {
            "features": feature_config.as_dict(),
            "model": model_config.as_dict(),
            "training": training_config.as_dict(),
        },
        "provenance": provenance,
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    metadata_path = output_dir / f"{base_name}.metadata.json"
    metadata_payload = _json_payload(metadata)
    payloads = [
        (schema_path, schema_payload),
        *artifact_payloads,
        (metadata_path, metadata_payload),
    ]
    transaction_path = output_dir / f"{base_name}.export-transaction.json"
    marker_path = output_dir / f"{base_name}.export-commit.json"
    publication = {
        "schema": "phase4_value_export_transaction/v1",
        "baseName": base_name,
        "artifacts": [
            {
                "path": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
            for path, payload in payloads
        ],
    }
    try:
        with stable_parent_descriptor(marker_path, create=True) as (
            output_descriptor,
            marker_name,
        ):
            if marker_name != marker_path.name:
                raise ValueError("export marker has an invalid final component")
            _publish_export_transaction(
                output_descriptor=output_descriptor,
                transaction_path=transaction_path,
                marker_path=marker_path,
                publication=publication,
                payloads=payloads,
            )
    except ArtifactError as error:
        raise ValueError("export output_dir must remain a non-symlink directory") from error
    metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
    metadata_size = len(metadata_payload)
    return {
        "metadata": metadata,
        "metadataPath": str(metadata_path),
        "metadataSha256": metadata_sha256,
        "metadataSize": metadata_size,
        "featureSchemaPath": str(schema_path),
        "featureSchemaSha256": schema_identity[0],
        "featureSchemaSize": schema_identity[1],
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }


def _publish_export_transaction(
    *,
    output_descriptor: int,
    transaction_path: Path,
    marker_path: Path,
    publication: dict[str, Any],
    payloads: list[tuple[Path, bytes]],
) -> None:
    transaction_payload = _json_payload(publication)
    marker = {**publication, "schema": "phase4_value_export_commit/v1"}
    marker_payload = _json_payload(marker)
    if _entry_exists(output_descriptor, marker_path.name):
        _require_exact_regular_bytes_at(output_descriptor, marker_path, marker_payload)
        for path, payload in payloads:
            _require_exact_regular_bytes_at(output_descriptor, path, payload)
        if _entry_exists(output_descriptor, transaction_path.name):
            _require_exact_regular_bytes_at(
                output_descriptor, transaction_path, transaction_payload
            )
            _unlink_regular_at(output_descriptor, transaction_path)
        return
    if _entry_exists(output_descriptor, transaction_path.name):
        _require_exact_regular_bytes_at(output_descriptor, transaction_path, transaction_payload)
    else:
        if any(_entry_exists(output_descriptor, path.name) for path, _ in payloads):
            raise FileExistsError(
                "refusing to overwrite export artifacts that exist without a recovery transaction"
            )
        _write_new_atomic_at(output_descriptor, transaction_path, transaction_payload)
    _export_failpoint("after_journal")
    for index, (path, payload) in enumerate(payloads):
        if _entry_exists(output_descriptor, path.name):
            _require_exact_regular_bytes_at(output_descriptor, path, payload)
        else:
            _write_new_atomic_at(output_descriptor, path, payload)
        _export_failpoint(f"after_artifact_{index}")
    _write_new_atomic_at(output_descriptor, marker_path, marker_payload)
    _export_failpoint("after_marker")
    _unlink_regular_at(output_descriptor, transaction_path)


def _require_exact_regular_bytes_at(
    output_descriptor: int,
    path: Path,
    expected: bytes,
) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_descriptor,
        )
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size != len(expected):
            raise ValueError(f"export artifact size or type conflicts with recovery: {path}")
        observed = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, len(expected) + 1 - len(observed))):
            observed.extend(chunk)
            if len(observed) > len(expected):
                raise ValueError(f"export artifact grew during recovery: {path}")
        linked = os.stat(path.name, dir_fd=output_descriptor, follow_symlinks=False)
        final = os.fstat(descriptor)
        if (
            stat.S_ISLNK(linked.st_mode)
            or _file_state(initial) != _file_state(final)
            or _file_state(linked) != _file_state(final)
        ):
            raise ValueError(f"export artifact path changed during recovery: {path}")
    except OSError as error:
        raise ValueError(f"export artifact is unsafe during recovery: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if bytes(observed) != expected:
        raise ValueError(f"export artifact bytes conflict with recovery: {path}")


def _export_failpoint(name: str) -> None:
    if os.environ.get(_EXPORT_FAILPOINT_ENV) == name:
        raise RuntimeError(f"export failpoint: {name}")


def _entry_exists(output_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=output_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _unlink_regular_at(output_descriptor: int, path: Path) -> None:
    descriptor = os.open(
        path.name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_descriptor,
    )
    try:
        retire_bound_regular(
            output_descriptor,
            path.name,
            os.fstat(descriptor),
            display=path,
        )
    finally:
        os.close(descriptor)
    os.fsync(output_descriptor)


def _linear_values(layer: Any) -> tuple[list[float], list[float], int, int]:
    import torch

    weight = layer.weight.detach().to(device="cpu", dtype=torch.float32)
    bias = layer.bias.detach().to(device="cpu", dtype=torch.float32)
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
        raise ValueError("export layer is not a dense linear layer with bias")
    weights = [float(value) for value in weight.contiguous().reshape(-1).tolist()]
    biases = [float(value) for value in bias.contiguous().tolist()]
    if any(not math.isfinite(value) for value in weights + biases):
        raise FloatingPointError("cannot export non-finite model parameters")
    return weights, biases, int(weight.shape[1]), int(weight.shape[0])


def _unpack_required(formatter: struct.Struct, data: bytes, offset: int) -> tuple[Any, ...]:
    if offset + formatter.size > len(data):
        raise ValueError("model is truncated")
    return formatter.unpack_from(data, offset)


def _checked_element_count(input_dim: int, output_dim: int) -> int:
    count = input_dim * output_dim
    if count > 64_000_000:
        raise ValueError("model layer exceeds the parser element bound")
    return count


def _input_dimension_from_flags(flags: int) -> int:
    return (
        (2 * 14 * 81 if flags & FEATURE_BOARD else 0)
        + (2 * 7 if flags & FEATURE_HANDS else 0)
        + (1 if flags & FEATURE_SIDE_TO_MOVE else 0)
        + (4 if flags & FEATURE_KINGS else 0)
        + (2 * 81 if flags & FEATURE_ATTACKS else 0)
    )


def _read_float32s(data: bytes, offset: int, count: int) -> tuple[tuple[float, ...], int]:
    size = count * _F32.size
    end = offset + size
    if end > len(data):
        raise ValueError("model is truncated in float32 values")
    return tuple(struct.unpack_from(f"<{count}f", data, offset)), end


def _as_float32(value: float) -> float:
    return _F32.unpack(_F32.pack(value))[0]


def _finite_to_float32(value: float) -> float:
    return _as_float32(max(-_F32_MAX, min(_F32_MAX, value)))


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_atomic_at(output_descriptor: int, path: Path, payload: bytes) -> None:
    name = path.name
    temporary_name = f".{name}.{secrets.token_hex(12)}"
    descriptor = -1
    temporary_created = False
    temporary_status: os.stat_result | None = None
    published_status: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=output_descriptor,
        )
        temporary_created = True
        temporary_status = os.fstat(descriptor)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            published_status = os.fstat(output.fileno())
        try:
            assert temporary_status is not None
            publish_regular_at(
                output_descriptor,
                temporary_name,
                name,
                temporary_status,
                display=path,
                replace=False,
            )
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from None
        assert published_status is not None
        linked = os.stat(name, dir_fd=output_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(linked.st_mode) or (
            linked.st_dev,
            linked.st_ino,
            linked.st_mode,
            linked.st_size,
        ) != (
            published_status.st_dev,
            published_status.st_ino,
            published_status.st_mode,
            published_status.st_size,
        ):
            raise ValueError(f"export artifact path changed during publication: {path}")
        temporary_created = False
        os.fsync(output_descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            assert temporary_status is not None
            retire_bound_regular(
                output_descriptor,
                temporary_name,
                temporary_status,
                display=path,
            )
        raise


def _file_state(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )
