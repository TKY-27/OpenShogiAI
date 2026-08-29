"""Closed OSAVAL02 export, inspection, and parity-only reference inference.

The module intentionally does not implement Phase 10R training.  It serializes the two
already-frozen sparse candidates and provides a dependency-light reference evaluator used by
the native/Wasm parity gate.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

MAGIC: Final = b"OSAVAL02"
FORMAT_VERSION: Final = 2
ENDIAN_MARKER: Final = 0x0102_0304
HEADER_BYTES: Final = 4_064
CHECKSUM_BYTES: Final = 32
CONTAINER_OVERHEAD_BYTES: Final = HEADER_BYTES + CHECKSUM_BYTES
MAX_MODEL_BYTES: Final = 16 * 1024 * 1024
DESCRIPTOR_OFFSET: Final = 512
DESCRIPTOR_BYTES: Final = 112
MAX_TENSORS: Final = (HEADER_BYTES - DESCRIPTOR_OFFSET) // DESCRIPTOR_BYTES
POLICY_CLASSES: Final = 13_689
SCALAR_INPUTS: Final = 64
TRUNK_HIDDEN: Final = 128
MOVE_DIMENSION: Final = 16
HASH_SEED: Final = 20_260_729
MAX_TRIPLES: Final = 256
MAX_NON_MATE_CP: Final = 28_999
EXPORTER_VERSION: Final = "OpenShogiAI-osaval02-py/v1"
FEATURE_HASH_DOMAIN: Final = b"OpenShogiAI/phase10r/features/v1\0"

FEATURE_SCHEMA_SHA256: Final = "fb5d69c96ae45ed308ee18ab7fd16d4fbefe0f5778e0b0ee2144879bcc7881df"
ARCHITECTURE_CONFIG_SHA256: Final = (
    "50a6873b521f389c766010a0ed83fa5d2399d18a05a860fec78ef35f523dfb6b"
)
TARGET_SEMANTICS_SHA256: Final = "bafe7ba97319fa12d0c7a8ef3e3634fe033926fa77fcb781b67cd8ab12dda3bb"
INPUT_NORMALIZATION_SHA256: Final = (
    "b19dea246a17e059010418f2ac04b5a75fb1e21a30ba050159e539a9f5f69185"
)
DEFAULT_DATASET_MANIFEST_SHA256: Final = (
    "4c2b7a3c5cdd9e9db611a30284aa2dca2d2e8a8711b3186072425f773fd9b9d2"
)
MOVE_INDEX_SHA256: Final = "096b227cadae6e585977b688495f261b163ebd7d3b4d2cc7276555a1b297de2a"
FLOAT_CONFIG_SHA256: Final = "4ab7ec4932dd595a80b3ccddcf46f0fe42fe318c62558d79ca91da3218580327"
INT8_CONFIG_SHA256: Final = "315c92526d38149c67d5fb8b97b88a177b7ee87932910ebdd20e42479d3c7041"

VARIANT_PAIR: Final = "sparse-pair-policy-wdl"
VARIANT_PRIMARY: Final = "factorized-pair-triple-policy-score"
VARIANT_CODES: Final = {VARIANT_PAIR: 1, VARIANT_PRIMARY: 2}
VARIANT_NAMES: Final = {value: key for key, value in VARIANT_CODES.items()}
QUANTIZATION_CODES: Final = {"float32": 0, "int8": 1}
QUANTIZATION_NAMES: Final = {value: key for key, value in QUANTIZATION_CODES.items()}
DTYPE_FLOAT32: Final = 1
DTYPE_INT8: Final = 2
HEAD_WDL: Final = 1 << 0
HEAD_SCORE: Final = 1 << 1
HEAD_MATE: Final = 1 << 2
HEAD_UNCERTAINTY: Final = 1 << 3
HEAD_POLICY: Final = 1 << 4
HEADS_PAIR: Final = HEAD_WDL | HEAD_MATE | HEAD_UNCERTAINTY | HEAD_POLICY
HEADS_PRIMARY: Final = HEADS_PAIR | HEAD_SCORE

_DESCRIPTOR = struct.Struct("<48sII4IQQQfiI4x")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_USI_MOVE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")
_F32 = struct.Struct("<f")
_PIECE_INDEX: Final = {
    "P": 0,
    "L": 1,
    "N": 2,
    "S": 3,
    "G": 4,
    "B": 5,
    "R": 6,
    "K": 7,
    "+P": 8,
    "+L": 9,
    "+N": 10,
    "+S": 11,
    "+B": 12,
    "+R": 13,
}
_HAND_INDEX: Final = {piece: index for index, piece in enumerate("PLNSGBR")}
_DROP_INDEX: Final = {piece: index for index, piece in enumerate("RBGSNLP")}
_HAND_MAXIMA: Final = (18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One required row-major tensor in the frozen architecture."""

    name: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True, slots=True)
class TensorView:
    """Validated tensor descriptor and borrowed payload bytes."""

    spec: TensorSpec
    dtype: int
    data: memoryview
    scale: float
    zero_point: int

    def value(self, index: int) -> float:
        if not 0 <= index < self.spec.elements:
            raise IndexError(index)
        if self.dtype == DTYPE_FLOAT32:
            return struct.unpack_from("<f", self.data, index * 4)[0]
        encoded = int(self.data[index])
        signed = encoded if encoded < 128 else encoded - 256
        return (signed - self.zero_point) * self.scale


@dataclass(frozen=True, slots=True)
class Osaval02Model:
    """A fully validated OSAVAL02 model artifact."""

    data: bytes
    variant_id: str
    quantization: str
    head_flags: int
    parameter_count: int
    calibration_scale: float
    calibration_bias: float
    wdl_epsilon: float
    dataset_manifest_sha256: str
    weight_payload_sha256: str
    exporter_version: str
    git_commit: str
    training_run_reference: str
    artifact_sha256: str
    tensors: Mapping[str, TensorView]


@dataclass(frozen=True, slots=True)
class Piece:
    side: int  # 0 Black, 1 White
    kind: int
    square: int


@dataclass(frozen=True, slots=True)
class ParsedPosition:
    board: tuple[Piece | None, ...]
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    side_to_move: int
    move_number: int
    canonical_state: str


@dataclass(frozen=True, slots=True)
class HistoryFacts:
    """Bounded facts available to standalone/runtime inference."""

    available: bool = False
    repetition_count: int = 1
    continuous_check_by_us: bool = False
    continuous_check_by_them: bool = False

    def validate(self) -> None:
        if not 1 <= self.repetition_count <= 4:
            raise ValueError("history repetition count must be in 1..=4")
        if not self.available and (
            self.repetition_count != 1
            or self.continuous_check_by_us
            or self.continuous_check_by_them
        ):
            raise ValueError("unavailable history must use the standalone default facts")
        if self.continuous_check_by_us and self.continuous_check_by_them:
            raise ValueError("both sides cannot own one continuous-check history")


def tensor_specs(variant_id: str) -> tuple[TensorSpec, ...]:
    """Return the exact frozen tensor names and dimensions for one eligible variant."""

    if variant_id not in VARIANT_CODES:
        raise ValueError("OSAVAL02 supports only the two frozen eligible variants")
    trunk_input = 48 if variant_id == VARIANT_PAIR else 56
    value_outputs = 8 if variant_id == VARIANT_PAIR else 9
    specs = [
        TensorSpec("king_piece_embeddings", (367_416, 4)),
        TensorSpec("king_hand_embeddings", (43_092, 4)),
        TensorSpec("pair_hash_embeddings", (65_536, 8)),
    ]
    if variant_id == VARIANT_PRIMARY:
        specs.append(TensorSpec("triple_hash_embeddings", (32_768, 8)))
    specs.extend(
        [
            TensorSpec("scalar_projection.weight", (32, 64)),
            TensorSpec("scalar_projection.bias", (32,)),
            TensorSpec("trunk.0.weight", (128, trunk_input)),
            TensorSpec("trunk.0.bias", (128,)),
            TensorSpec("trunk.1.weight", (128, 128)),
            TensorSpec("trunk.1.bias", (128,)),
            TensorSpec("value_heads.weight", (value_outputs, 128)),
            TensorSpec("value_heads.bias", (value_outputs,)),
            TensorSpec("policy.move_embeddings", (13_689, 16)),
            TensorSpec("policy.context_weight", (16, 128)),
            TensorSpec("policy.context_bias", (16,)),
            TensorSpec("policy.move_offset", (16,)),
            TensorSpec("policy.log_temperature", (1,)),
        ]
    )
    return tuple(specs)


def expected_parameter_count(variant_id: str) -> int:
    return sum(spec.elements for spec in tensor_specs(variant_id))


def serialize_osaval02(
    tensors: Mapping[str, object],
    *,
    variant_id: str,
    quantization: Literal["float32", "int8"],
    dataset_manifest_sha256: str = DEFAULT_DATASET_MANIFEST_SHA256,
    training_run_reference: str,
    git_commit: str,
    calibration_scale: float = 1_000.0,
    calibration_bias: float = 0.0,
    wdl_epsilon: float = 1.0e-6,
) -> bytes:
    """Deterministically serialize one frozen checkpoint tensor mapping."""

    specs = tensor_specs(variant_id)
    if quantization not in QUANTIZATION_CODES:
        raise ValueError("quantization must be float32 or int8")
    _validate_hex_hash(dataset_manifest_sha256, "dataset manifest")
    if not _GIT_COMMIT.fullmatch(git_commit):
        raise ValueError("git commit must be 40 lowercase hexadecimal characters")
    _validate_fixed_text(
        training_run_reference,
        64,
        "training run reference",
        allow_full_width=True,
    )
    if not math.isfinite(calibration_scale) or calibration_scale <= 0.0:
        raise ValueError("calibration scale must be finite and positive")
    if not math.isfinite(calibration_bias):
        raise ValueError("calibration bias must be finite")
    if not math.isfinite(wdl_epsilon) or not 0.0 < wdl_epsilon <= 0.01:
        raise ValueError("WDL epsilon must be finite and in (0, 0.01]")
    expected_names = {spec.name for spec in specs}
    if set(tensors) != expected_names:
        missing = sorted(expected_names - set(tensors))
        extra = sorted(set(tensors) - expected_names)
        raise ValueError(
            f"checkpoint tensors differ from frozen architecture: missing={missing}, extra={extra}"
        )

    payload_chunks: list[bytes] = []
    descriptors: list[bytes] = []
    payload_offset = 0
    dtype_code = DTYPE_FLOAT32 if quantization == "float32" else DTYPE_INT8
    for spec in specs:
        array = np.asarray(tensors[spec.name])
        if tuple(array.shape) != spec.shape:
            raise ValueError(f"tensor {spec.name} shape {array.shape} != {spec.shape}")
        if array.dtype.kind not in "fiu":
            raise ValueError(f"tensor {spec.name} must be numeric")
        values = np.asarray(array, dtype=np.float32, order="C")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"tensor {spec.name} contains a non-finite parameter")
        if quantization == "float32":
            encoded = values.astype("<f4", copy=False).tobytes(order="C")
            scale = 1.0
        else:
            maximum = float(np.max(np.abs(values), initial=np.float32(0.0)))
            scale = _as_float32(maximum / 127.0 if maximum > 0.0 else 1.0)
            if scale == 0.0:
                scale = _as_float32(2.0**-126)
            quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
            encoded = quantized.tobytes(order="C")
        name = spec.name.encode("ascii")
        if len(name) >= 48:
            raise ValueError(f"tensor name is too long: {spec.name}")
        dims = (*spec.shape, *(1 for _ in range(4 - len(spec.shape))))
        descriptors.append(
            _DESCRIPTOR.pack(
                name.ljust(48, b"\0"),
                dtype_code,
                len(spec.shape),
                *dims,
                payload_offset,
                len(encoded),
                spec.elements,
                scale,
                0,
                1,
            )
        )
        payload_chunks.append(encoded)
        payload_offset += len(encoded)

    payload = b"".join(payload_chunks)
    payload_sha256 = hashlib.sha256(payload).digest()
    parameter_count = expected_parameter_count(variant_id)
    expected_payload_bytes = parameter_count * (4 if quantization == "float32" else 1)
    if len(payload) != expected_payload_bytes:
        raise AssertionError("OSAVAL02 payload size invariant failed")
    file_bytes = CONTAINER_OVERHEAD_BYTES + len(payload)
    if file_bytes > MAX_MODEL_BYTES:
        raise ValueError("serialized OSAVAL02 exceeds the 16 MiB browser boundary")

    header = bytearray(HEADER_BYTES)
    header[:8] = MAGIC
    _pack_u32(header, 8, FORMAT_VERSION)
    _pack_u32(header, 12, ENDIAN_MARKER)
    _pack_u32(header, 16, HEADER_BYTES)
    struct.pack_into("<Q", header, 20, file_bytes)
    _pack_u32(header, 28, VARIANT_CODES[variant_id])
    _pack_u32(header, 32, QUANTIZATION_CODES[quantization])
    _pack_u32(header, 36, len(specs))
    _pack_u32(header, 40, HEADS_PAIR if variant_id == VARIANT_PAIR else HEADS_PRIMARY)
    _pack_u32(header, 44, parameter_count)
    _pack_u32(header, 48, POLICY_CLASSES)
    _pack_u32(header, 52, SCALAR_INPUTS)
    _pack_u32(header, 56, 48 if variant_id == VARIANT_PAIR else 56)
    _pack_u32(header, 60, TRUNK_HIDDEN)
    struct.pack_into("<Q", header, 64, HASH_SEED)
    _pack_u32(header, 72, 0 if variant_id == VARIANT_PAIR else MAX_TRIPLES)
    _pack_u32(header, 76, 1)  # signed feature hashing
    struct.pack_into(
        "<fffI", header, 80, calibration_scale, calibration_bias, wdl_epsilon, MAX_NON_MATE_CP
    )
    hashes = (
        FEATURE_SCHEMA_SHA256,
        ARCHITECTURE_CONFIG_SHA256,
        TARGET_SEMANTICS_SHA256,
        INPUT_NORMALIZATION_SHA256,
        dataset_manifest_sha256,
        payload_sha256.hex(),
        FLOAT_CONFIG_SHA256 if quantization == "float32" else INT8_CONFIG_SHA256,
        MOVE_INDEX_SHA256,
    )
    for index, digest in enumerate(hashes):
        header[96 + index * 32 : 128 + index * 32] = bytes.fromhex(digest)
    _write_fixed_text(header, 352, 32, EXPORTER_VERSION)
    header[384:424] = git_commit.encode("ascii")
    _write_fixed_text(header, 424, 64, training_run_reference, allow_full_width=True)
    _pack_u32(header, 488, DESCRIPTOR_OFFSET)
    _pack_u32(header, 492, DESCRIPTOR_BYTES)
    struct.pack_into("<Q", header, 496, HEADER_BYTES)
    struct.pack_into("<Q", header, 504, 0)
    descriptor_blob = b"".join(descriptors)
    header[DESCRIPTOR_OFFSET : DESCRIPTOR_OFFSET + len(descriptor_blob)] = descriptor_blob
    unsigned = bytes(header) + payload
    return unsigned + hashlib.sha256(unsigned).digest()


def parse_osaval02(data: bytes) -> Osaval02Model:
    """Strictly validate and inspect one OSAVAL02 byte string."""

    if not isinstance(data, bytes):
        raise TypeError("model data must be bytes")
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("OSAVAL02 exceeds the 16 MiB parser bound")
    if len(data) < CONTAINER_OVERHEAD_BYTES:
        raise ValueError("OSAVAL02 is truncated")
    unsigned, stored_checksum = data[:-CHECKSUM_BYTES], data[-CHECKSUM_BYTES:]
    if not hmac.compare_digest(hashlib.sha256(unsigned).digest(), stored_checksum):
        raise ValueError("OSAVAL02 trailing SHA-256 does not match")
    header = unsigned[:HEADER_BYTES]
    if header[:8] != MAGIC:
        raise ValueError("model magic is not OSAVAL02")
    if _u32(header, 8) != FORMAT_VERSION:
        raise ValueError("unsupported OSAVAL02 format version")
    if _u32(header, 12) != ENDIAN_MARKER:
        raise ValueError("OSAVAL02 byte-order marker is invalid")
    if _u32(header, 16) != HEADER_BYTES or _u64(header, 496) != HEADER_BYTES:
        raise ValueError("OSAVAL02 header or payload offset is invalid")
    if _u64(header, 20) != len(data):
        raise ValueError("OSAVAL02 declared file length is invalid")
    if _u64(header, 504) != 0:
        raise ValueError("OSAVAL02 reserved header field is nonzero")
    variant_code = _u32(header, 28)
    variant_id = VARIANT_NAMES.get(variant_code)
    if variant_id is None:
        raise ValueError("unsupported OSAVAL02 architecture identifier")
    quantization_code = _u32(header, 32)
    quantization = QUANTIZATION_NAMES.get(quantization_code)
    if quantization is None:
        raise ValueError("unsupported OSAVAL02 quantization")
    specs = tensor_specs(variant_id)
    tensor_count = _u32(header, 36)
    if tensor_count != len(specs) or tensor_count > MAX_TENSORS:
        raise ValueError("OSAVAL02 tensor count is incompatible")
    expected_heads = HEADS_PAIR if variant_id == VARIANT_PAIR else HEADS_PRIMARY
    if _u32(header, 40) != expected_heads:
        raise ValueError("OSAVAL02 output heads are incompatible")
    parameter_count = _u32(header, 44)
    if parameter_count != expected_parameter_count(variant_id):
        raise ValueError("OSAVAL02 parameter count is incompatible")
    expected_scalars = (
        POLICY_CLASSES,
        SCALAR_INPUTS,
        48 if variant_id == VARIANT_PAIR else 56,
        TRUNK_HIDDEN,
    )
    if tuple(_u32(header, offset) for offset in (48, 52, 56, 60)) != expected_scalars:
        raise ValueError("OSAVAL02 frozen dimensions are incompatible")
    if _u64(header, 64) != HASH_SEED:
        raise ValueError("OSAVAL02 feature hash seed is incompatible")
    if _u32(header, 72) != (0 if variant_id == VARIANT_PAIR else MAX_TRIPLES):
        raise ValueError("OSAVAL02 triple cap is incompatible")
    if _u32(header, 76) != 1 or _u32(header, 92) != MAX_NON_MATE_CP:
        raise ValueError("OSAVAL02 runtime flags are incompatible")
    calibration_scale, calibration_bias, wdl_epsilon = struct.unpack_from("<fff", header, 80)
    if not math.isfinite(calibration_scale) or calibration_scale <= 0.0:
        raise ValueError("OSAVAL02 calibration scale is invalid")
    if not math.isfinite(calibration_bias):
        raise ValueError("OSAVAL02 calibration bias is invalid")
    if not math.isfinite(wdl_epsilon) or not 0.0 < wdl_epsilon <= 0.01:
        raise ValueError("OSAVAL02 WDL epsilon is invalid")
    observed_hashes = tuple(header[96 + index * 32 : 128 + index * 32].hex() for index in range(8))
    expected_compatibility = (
        FEATURE_SCHEMA_SHA256,
        ARCHITECTURE_CONFIG_SHA256,
        TARGET_SEMANTICS_SHA256,
        INPUT_NORMALIZATION_SHA256,
    )
    if observed_hashes[:4] != expected_compatibility:
        labels = ("feature schema", "architecture", "target semantics", "input normalization")
        mismatch = next(
            label
            for label, left, right in zip(
                labels, observed_hashes, expected_compatibility, strict=True
            )
            if left != right
        )
        raise ValueError(f"OSAVAL02 {mismatch} hash is incompatible")
    dataset_sha256, payload_sha256, quantization_sha256, move_index_sha256 = observed_hashes[4:]
    if dataset_sha256 == "0" * 64:
        raise ValueError("OSAVAL02 dataset/training identity is missing")
    expected_quantization_hash = (
        FLOAT_CONFIG_SHA256 if quantization == "float32" else INT8_CONFIG_SHA256
    )
    if quantization_sha256 != expected_quantization_hash:
        raise ValueError("OSAVAL02 quantization-config hash is incompatible")
    if move_index_sha256 != MOVE_INDEX_SHA256:
        raise ValueError("OSAVAL02 move-index table hash is incompatible")
    exporter_version = _read_fixed_text(header, 352, 32, "exporter version")
    if exporter_version != EXPORTER_VERSION:
        raise ValueError("OSAVAL02 exporter version is incompatible")
    try:
        git_commit = header[384:424].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("OSAVAL02 Git commit identity is not ASCII") from error
    if not _GIT_COMMIT.fullmatch(git_commit):
        raise ValueError("OSAVAL02 Git commit identity is invalid")
    training_run_reference = _read_fixed_text(
        header,
        424,
        64,
        "training run reference",
        allow_full_width=True,
    )
    if _u32(header, 488) != DESCRIPTOR_OFFSET or _u32(header, 492) != DESCRIPTOR_BYTES:
        raise ValueError("OSAVAL02 tensor-table layout is incompatible")

    payload = memoryview(unsigned)[HEADER_BYTES:]
    if hashlib.sha256(payload).hexdigest() != payload_sha256:
        raise ValueError("OSAVAL02 weight payload SHA-256 does not match")
    expected_dtype = DTYPE_FLOAT32 if quantization == "float32" else DTYPE_INT8
    item_bytes = 4 if quantization == "float32" else 1
    views: dict[str, TensorView] = {}
    next_offset = 0
    for index, spec in enumerate(specs):
        descriptor_offset = DESCRIPTOR_OFFSET + index * DESCRIPTOR_BYTES
        unpacked = _DESCRIPTOR.unpack_from(header, descriptor_offset)
        raw_name, dtype, rank, *remainder = unpacked
        dimensions = tuple(remainder[:4])
        offset, length, elements, scale, zero_point, flags = remainder[4:]
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        if name != spec.name or rank != len(spec.shape):
            raise ValueError("OSAVAL02 tensor name or rank is incompatible")
        if dimensions[:rank] != spec.shape or any(value != 1 for value in dimensions[rank:]):
            raise ValueError(f"OSAVAL02 tensor {name} shape is incompatible")
        if dtype != expected_dtype:
            raise ValueError(f"OSAVAL02 tensor {name} dtype is incompatible")
        if offset != next_offset or elements != spec.elements or length != elements * item_bytes:
            raise ValueError(f"OSAVAL02 tensor {name} size or offset is incompatible")
        if flags != 1 or zero_point != 0 or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"OSAVAL02 tensor {name} quantization metadata is invalid")
        if quantization == "float32" and scale != 1.0:
            raise ValueError(f"OSAVAL02 float tensor {name} scale must be one")
        end = offset + length
        if end > len(payload):
            raise ValueError("OSAVAL02 tensor requests an oversized allocation")
        view = TensorView(spec, dtype, payload[offset:end], scale, zero_point)
        if dtype == DTYPE_FLOAT32:
            for value_index in range(spec.elements):
                if not math.isfinite(view.value(value_index)):
                    raise ValueError(f"OSAVAL02 tensor {name} contains a non-finite parameter")
        views[name] = view
        next_offset = end
    if next_offset != len(payload):
        raise ValueError("OSAVAL02 has a trailing invalid section")
    descriptor_end = DESCRIPTOR_OFFSET + tensor_count * DESCRIPTOR_BYTES
    if any(header[descriptor_end:HEADER_BYTES]):
        raise ValueError("OSAVAL02 unused header bytes must be zero")
    expected_payload_bytes = parameter_count * item_bytes
    if (
        len(payload) != expected_payload_bytes
        or len(data) != CONTAINER_OVERHEAD_BYTES + expected_payload_bytes
    ):
        raise ValueError("OSAVAL02 artifact size differs from the frozen model matrix")

    return Osaval02Model(
        data=data,
        variant_id=variant_id,
        quantization=quantization,
        head_flags=expected_heads,
        parameter_count=parameter_count,
        calibration_scale=calibration_scale,
        calibration_bias=calibration_bias,
        wdl_epsilon=wdl_epsilon,
        dataset_manifest_sha256=dataset_sha256,
        weight_payload_sha256=payload_sha256,
        exporter_version=exporter_version,
        git_commit=git_commit,
        training_run_reference=training_run_reference,
        artifact_sha256=hashlib.sha256(data).hexdigest(),
        tensors=views,
    )


def inspect_osaval02(data: bytes) -> dict[str, object]:
    """Return a bounded machine-readable identity after full validation."""

    model = parse_osaval02(data)
    return {
        "schema": "open_shogiai_osaval02_inspection/v1",
        "formatVersion": FORMAT_VERSION,
        "variantId": model.variant_id,
        "quantization": model.quantization,
        "headFlags": model.head_flags,
        "parameterCount": model.parameter_count,
        "artifactBytes": len(model.data),
        "artifactSha256": model.artifact_sha256,
        "weightPayloadSha256": model.weight_payload_sha256,
        "featureSchemaSha256": FEATURE_SCHEMA_SHA256,
        "architectureConfigSha256": ARCHITECTURE_CONFIG_SHA256,
        "targetSemanticsSha256": TARGET_SEMANTICS_SHA256,
        "inputNormalizationSha256": INPUT_NORMALIZATION_SHA256,
        "datasetManifestSha256": model.dataset_manifest_sha256,
        "moveIndexSha256": MOVE_INDEX_SHA256,
        "exporterVersion": model.exporter_version,
        "gitCommit": model.git_commit,
        "trainingRunReference": model.training_run_reference,
        "tensors": [
            {"name": spec.name, "shape": list(spec.shape), "elements": spec.elements}
            for spec in tensor_specs(model.variant_id)
        ],
    }


def load_osaval02(path: Path) -> Osaval02Model:
    """Read a regular bounded file once and validate the complete artifact."""

    if path.is_symlink():
        raise ValueError("OSAVAL02 path must be a regular non-symlink file")
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError("OSAVAL02 path must be a regular non-symlink file")
    size = path.stat().st_size
    if size > MAX_MODEL_BYTES:
        raise ValueError("OSAVAL02 exceeds the 16 MiB parser bound")
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError("OSAVAL02 changed while it was read")
    return parse_osaval02(data)


def infer_osaval02(
    model: Osaval02Model,
    sfen: str,
    legal_moves: Sequence[str],
    history: HistoryFacts | None = None,
) -> dict[str, object]:
    """Run deterministic reference inference for cross-runtime parity only."""

    history = HistoryFacts() if history is None else history
    history.validate()
    position = parse_sfen(sfen)
    move_rows = _validate_legal_move_list(legal_moves)
    features = _encode_features(position, move_rows, history, model.variant_id)
    king_piece = _embedding_sum(model.tensors["king_piece_embeddings"], features["king_piece"], 4)
    king_hand = _embedding_sum(model.tensors["king_hand_embeddings"], features["king_hand"], 4)
    pair = _embedding_sum(model.tensors["pair_hash_embeddings"], features["pair"], 8)
    projected = _linear(
        model.tensors["scalar_projection.weight"],
        model.tensors["scalar_projection.bias"],
        features["scalars"],
        32,
        relu=False,
    )
    trunk_input = [*king_piece, *king_hand, *pair]
    if model.variant_id == VARIANT_PRIMARY:
        trunk_input.extend(
            _embedding_sum(model.tensors["triple_hash_embeddings"], features["triple"], 8)
        )
    trunk_input.extend(projected)
    hidden = _linear(
        model.tensors["trunk.0.weight"],
        model.tensors["trunk.0.bias"],
        trunk_input,
        128,
        relu=True,
    )
    hidden = _linear(
        model.tensors["trunk.1.weight"],
        model.tensors["trunk.1.bias"],
        hidden,
        128,
        relu=True,
    )
    output_count = 8 if model.variant_id == VARIANT_PAIR else 9
    outputs = _linear(
        model.tensors["value_heads.weight"],
        model.tensors["value_heads.bias"],
        hidden,
        output_count,
        relu=False,
    )
    wdl = _softmax(outputs[:3])
    if model.variant_id == VARIANT_PRIMARY:
        transformed_score = outputs[3]
        mate_offset = 4
        score_source = "direct_transformed_score"
        raw_score = math.copysign(
            math.expm1(min(abs(transformed_score), 1.0) * math.log1p(3_000.0)),
            transformed_score,
        )
    else:
        transformed_score = math.log((wdl[2] + model.wdl_epsilon) / (wdl[0] + model.wdl_epsilon))
        mate_offset = 3
        score_source = "wdl_log_odds"
        raw_score = transformed_score
    calibrated = raw_score * model.calibration_scale + model.calibration_bias
    calibrated_cp = max(-MAX_NON_MATE_CP, min(MAX_NON_MATE_CP, _round_ties_away(calibrated)))
    mate_probabilities = _softmax(outputs[mate_offset : mate_offset + 3])
    mate_class = ("mated", "no_mate_label", "mating")[
        max(range(3), key=lambda index: (mate_probabilities[index], -index))
    ]
    distance_raw = outputs[mate_offset + 3]
    mate_distance = math.copysign(
        math.expm1(min(abs(distance_raw), math.log1p(512.0))), distance_raw
    )
    uncertainty_log_variance = max(-20.0, min(20.0, outputs[mate_offset + 4]))

    context = _linear(
        model.tensors["policy.context_weight"],
        model.tensors["policy.context_bias"],
        hidden,
        MOVE_DIMENSION,
        relu=False,
    )
    move_offset = [model.tensors["policy.move_offset"].value(index) for index in range(16)]
    temperature = math.exp(max(-4.0, min(4.0, model.tensors["policy.log_temperature"].value(0))))
    move_embeddings = model.tensors["policy.move_embeddings"]
    policies: list[dict[str, object]] = []
    for move, move_index in move_rows:
        base = move_index * MOVE_DIMENSION
        logit = temperature * sum(
            context[dimension] * (move_embeddings.value(base + dimension) + move_offset[dimension])
            for dimension in range(MOVE_DIMENSION)
        )
        policies.append({"move": move, "index": move_index, "logit": logit})
    policies.sort(key=lambda row: (-float(row["logit"]), int(row["index"])))

    us = position.side_to_move
    in_check = _square_attacked(position, _king_square(position, us), 1 - us)
    terminal_kind: str | None = None
    search_score_cp: int | None = None
    if not move_rows:
        terminal_kind = "checkmate" if in_check else "no_legal_move"
        search_score_cp = -30_000 if in_check else 0
    result = {
        "schema": "open_shogiai_osaval02_inference/v1",
        "identity": {
            "formatVersion": FORMAT_VERSION,
            "variantId": model.variant_id,
            "quantization": model.quantization,
            "artifactSha256": model.artifact_sha256,
            "weightPayloadSha256": model.weight_payload_sha256,
        },
        "positionSha256": hashlib.sha256(position.canonical_state.encode("ascii")).hexdigest(),
        "featureSha256": features["checksum"],
        "history": {
            "available": history.available,
            "repetitionCount": history.repetition_count,
            "continuousCheckByUs": history.continuous_check_by_us,
            "continuousCheckByThem": history.continuous_check_by_them,
        },
        "legalMoves": policies,
        "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]},
        "score": {
            "sourceHead": score_source,
            "transformed": transformed_score,
            "calibratedCp": calibrated_cp,
            "perspective": "current_side_to_move",
        },
        "mate": {
            "class": mate_class,
            "probabilities": {
                "mated": mate_probabilities[0],
                "noMateLabel": mate_probabilities[1],
                "mating": mate_probabilities[2],
            },
            "distancePlies": mate_distance,
            "scoreConversion": "forbidden",
        },
        "uncertainty": {
            "logVariance": uncertainty_log_variance,
            "variance": math.exp(uncertainty_log_variance),
            "mixedIntoScore": False,
        },
        "terminal": {"kind": terminal_kind, "searchScoreCp": search_score_cp},
    }
    _assert_finite_tree(result)
    return result


def parse_sfen(sfen: str) -> ParsedPosition:
    """Parse the strict canonical SFEN subset needed by parity reference inference."""

    if not isinstance(sfen, str) or len(sfen.encode("utf-8")) > 1_024:
        raise ValueError("SFEN must be a bounded string")
    fields = sfen.split(" ")
    if len(fields) != 4 or "  " in sfen or sfen.strip() != sfen:
        raise ValueError("SFEN must contain four canonical fields")
    board_field, side_field, hand_field, move_field = fields
    ranks = board_field.split("/")
    if len(ranks) != 9:
        raise ValueError("SFEN board must contain nine ranks")
    board: list[Piece | None] = [None] * 81
    kings = [0, 0]
    for rank_index, rank in enumerate(ranks):
        column = 0
        promoted = False
        for character in rank:
            if character == "+":
                if promoted:
                    raise ValueError("SFEN contains a repeated promotion marker")
                promoted = True
                continue
            if character.isdigit():
                if promoted or character == "0":
                    raise ValueError("SFEN board empty count is invalid")
                column += int(character)
                continue
            if column >= 9 or character.upper() not in "PLNSGBRK":
                raise ValueError("SFEN board piece is invalid")
            side = 0 if character.isupper() else 1
            name = ("+" if promoted else "") + character.upper()
            if name not in _PIECE_INDEX:
                raise ValueError("SFEN promotion is invalid")
            piece = Piece(side, _PIECE_INDEX[name], rank_index * 9 + column)
            board[piece.square] = piece
            if piece.kind == 7:
                kings[side] += 1
            promoted = False
            column += 1
        if promoted or column != 9:
            raise ValueError("SFEN board rank width is invalid")
    if kings != [1, 1]:
        raise ValueError("SFEN must contain exactly one king per side")
    side_to_move = {"b": 0, "w": 1}.get(side_field)
    if side_to_move is None:
        raise ValueError("SFEN side to move is invalid")
    hands = [[0] * 7, [0] * 7]
    if hand_field != "-":
        digits = ""
        previous = -1
        order = "RBGSNLPrbgsnlp"
        for character in hand_field:
            if character.isdigit():
                digits += character
                continue
            if character not in order:
                raise ValueError("SFEN hand piece is invalid")
            current = order.index(character)
            if current <= previous:
                raise ValueError("SFEN hand order is not canonical")
            previous = current
            side = 0 if character.isupper() else 1
            count = int(digits) if digits else 1
            digits = ""
            hand_index = _HAND_INDEX[character.upper()]
            if not 1 <= count <= int(_HAND_MAXIMA[hand_index]):
                raise ValueError("SFEN hand count is invalid")
            hands[side][hand_index] = count
        if digits:
            raise ValueError("SFEN hand count lacks a piece")
    if not move_field.isdigit() or move_field.startswith("0"):
        raise ValueError("SFEN move number is invalid")
    move_number = int(move_field)
    if not 1 <= move_number <= 0xFFFF_FFFF:
        raise ValueError("SFEN move number is invalid")
    canonical_state = f"{board_field} {side_field} {hand_field}"
    return ParsedPosition(
        tuple(board), (tuple(hands[0]), tuple(hands[1])), side_to_move, move_number, canonical_state
    )


def encode_move(move: str) -> int:
    if not isinstance(move, str) or _USI_MOVE.fullmatch(move) is None:
        raise ValueError(f"invalid USI move: {move!r}")
    if "*" in move:
        return 13_122 + _DROP_INDEX[move[0]] * 81 + _square_index(move[2:4])
    return (_square_index(move[:2]) * 81 + _square_index(move[2:4])) * 2 + int(move.endswith("+"))


def decode_move(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < POLICY_CLASSES:
        raise ValueError("move index is outside 0..13688")
    if index >= 13_122:
        piece, square = divmod(index - 13_122, 81)
        return f"{'RBGSNLP'[piece]}*{_index_square(square)}"
    pair, promoted = divmod(index, 2)
    origin, target = divmod(pair, 81)
    return f"{_index_square(origin)}{_index_square(target)}{'+' if promoted else ''}"


def deterministic_test_tensors(variant_id: str) -> dict[str, np.ndarray]:
    """Create nontrivial deterministic weights for parity fixtures, never for training."""

    result: dict[str, np.ndarray] = {}
    for tensor_number, spec in enumerate(tensor_specs(variant_id), start=1):
        indexes = np.arange(spec.elements, dtype=np.int64)
        values = (((indexes * (tensor_number * 2 + 1) + tensor_number) % 29) - 14) / 2048.0
        result[spec.name] = values.astype(np.float32).reshape(spec.shape)
    result["policy.log_temperature"][0] = np.float32(0.125)
    return result


def _encode_features(
    position: ParsedPosition,
    move_rows: Sequence[tuple[str, int]],
    history: HistoryFacts,
    variant_id: str,
) -> dict[str, object]:
    pieces = [piece for piece in position.board if piece is not None]
    us = position.side_to_move
    kings = (_king_square(position, us), _king_square(position, 1 - us))
    attack_masks = {
        piece: _piece_attack_mask(position, piece)
        for piece in pieces
    }
    side_attack_masks = (
        _side_attack_mask(pieces, attack_masks, 0),
        _side_attack_mask(pieces, attack_masks, 1),
    )
    king_piece: list[tuple[int, int]] = []
    for piece in pieces:
        owner = int(piece.side != us)
        for king_role, king_square in enumerate(kings):
            index = (
                ((owner * 14 + piece.kind) * 2 + king_role) * 81 + king_square
            ) * 81 + piece.square
            king_piece.append((index, 1))
    king_hand: list[tuple[int, int]] = []
    for absolute_side in (us, 1 - us):
        owner = int(absolute_side != us)
        for hand_piece, count in enumerate(position.hands[absolute_side]):
            if count == 0:
                continue
            for king_role, king_square in enumerate(kings):
                index = (((owner * 7 + hand_piece) * 19 + count) * 2 + king_role) * 81 + king_square
                king_hand.append((index, 1))

    pinned = {piece.square: _is_pinned(position, piece) for piece in pieces}
    pair: list[tuple[int, int]] = []
    for left_index, left in enumerate(pieces):
        for right in pieces[left_index + 1 :]:
            flags = _pair_flags(left, right, kings, pinned, attack_masks, us)
            key = bytearray(
                (
                    1,
                    _relative_owner(left, us),
                    left.kind,
                    left.square,
                    _relative_owner(right, us),
                    right.kind,
                    right.square,
                )
            )
            for piece in (left, right):
                for king in kings:
                    key.extend(_signed_offsets(piece.square, king))
            key.extend(struct.pack("<H", flags))
            pair.append(_signed_bucket(bytes(key), 65_536))
    drop_count = 0
    for move, _ in move_rows:
        if "*" not in move:
            continue
        drop_count += 1
        hand_piece = _HAND_INDEX[move[0]]
        target = _square_index(move[2:4])
        count = position.hands[us][hand_piece]
        key = bytearray((2, 0, hand_piece, count, target))
        for king in kings:
            key.extend(_signed_offsets(target, king))
        flags = int(bool(side_attack_masks[us] & (1 << target)))
        flags |= int(bool(side_attack_masks[1 - us] & (1 << target))) << 1
        flags |= int(any(_chebyshev(target, king) <= 2 for king in kings)) << 2
        key.append(flags)
        pair.append(_signed_bucket(bytes(key), 65_536))
    pair.sort()

    triple: list[tuple[int, int]] = []
    if variant_id == VARIANT_PRIMARY:
        candidates: list[tuple[int, tuple[int, int, int], bytes]] = []
        for first in range(len(pieces)):
            for second in range(first + 1, len(pieces)):
                for third in range(second + 1, len(pieces)):
                    group = (pieces[first], pieces[second], pieces[third])
                    category = _triple_category(group, kings, pinned, attack_masks)
                    if category is None:
                        continue
                    squares = tuple(piece.square for piece in group)
                    key = bytearray((3, category))
                    for piece in group:
                        key.extend((_relative_owner(piece, us), piece.kind, piece.square))
                        for king in kings:
                            key.extend(_signed_offsets(piece.square, king))
                    candidates.append((category, squares, bytes(key)))
        candidates.sort(key=lambda row: (row[0], row[1]))
        seen: set[bytes] = set()
        for _, _, key in candidates:
            if key in seen:
                continue
            seen.add(key)
            triple.append(_signed_bucket(key, 32_768))
            if len(triple) == MAX_TRIPLES:
                break

    scalars = _scalar_features(
        position,
        move_rows,
        history,
        pieces,
        kings,
        pinned,
        side_attack_masks,
        len(pair),
        len(triple),
        drop_count,
    )
    digest = hashlib.sha256()
    for tag, rows in ((1, king_piece), (2, king_hand), (3, pair), (4, triple)):
        for index, sign in rows:
            digest.update(struct.pack("<BIb", tag, index, sign))
    for scalar in scalars:
        digest.update(_F32.pack(_as_float32(scalar)))
    return {
        "king_piece": king_piece,
        "king_hand": king_hand,
        "pair": pair,
        "triple": triple,
        "scalars": scalars,
        "checksum": digest.hexdigest(),
    }


def _scalar_features(
    position: ParsedPosition,
    move_rows: Sequence[tuple[str, int]],
    history: HistoryFacts,
    pieces: Sequence[Piece],
    kings: tuple[int, int],
    pinned: Mapping[int, bool],
    side_attack_masks: tuple[int, int],
    pair_count: int,
    triple_count: int,
    drop_count: int,
) -> list[float]:
    us = position.side_to_move
    values: list[float] = []
    for relative_side in (us, 1 - us):
        counts = [0] * 14
        for piece in pieces:
            if piece.side == relative_side:
                counts[piece.kind] += 1
        values.extend(count / 18.0 for count in counts)
    for relative_side in (us, 1 - us):
        values.extend(
            position.hands[relative_side][index] / _HAND_MAXIMA[index] for index in range(7)
        )
    attacked_us = side_attack_masks[us].bit_count()
    attacked_them = side_attack_masks[1 - us].bit_count()
    values.extend((attacked_us / 81.0, attacked_them / 81.0))
    values.append(
        sum(bool(side_attack_masks[1 - us] & (1 << square)) for square in _king_zone(kings[0]))
        / 25.0
    )
    values.append(
        sum(bool(side_attack_masks[us] & (1 << square)) for square in _king_zone(kings[1])) / 25.0
    )
    values.append(len(move_rows) / 600.0)
    values.append(float(bool(side_attack_masks[1 - us] & (1 << kings[0]))))
    values.append(float(bool(side_attack_masks[us] & (1 << kings[1]))))
    values.append(sum(pinned[piece.square] and piece.side == us for piece in pieces) / 20.0)
    values.append(sum(pinned[piece.square] and piece.side != us for piece in pieces) / 20.0)
    values.append(sum(piece.side == us and piece.kind >= 8 for piece in pieces) / 10.0)
    values.append(sum(piece.side != us and piece.kind >= 8 for piece in pieces) / 10.0)
    values.append(len(pieces) / 40.0)
    values.append(sum(sum(hand) for hand in position.hands) / 38.0)
    values.append(min(position.move_number - 1, 512) / 512.0)
    values.append(float(history.available))
    values.append(history.repetition_count / 4.0)
    values.append(float(history.continuous_check_by_us))
    values.append(float(history.continuous_check_by_them))
    values.append(min(pair_count, 780) / 780.0)
    values.append(triple_count / 256.0)
    values.append(drop_count / 567.0)
    values.append(1.0)
    if len(values) != SCALAR_INPUTS:
        raise AssertionError(f"scalar feature count is {len(values)} instead of 64")
    return [_as_float32(value) for value in values]


def _triple_category(
    pieces: tuple[Piece, Piece, Piece],
    kings: tuple[int, int],
    pinned: Mapping[int, bool],
    attack_masks: Mapping[Piece, int],
) -> int | None:
    for king in kings:
        if any(piece.square == king for piece in pieces) and all(
            piece.square == king or _chebyshev(piece.square, king) <= 2 for piece in pieces
        ):
            return 0
    for king in kings:
        king_piece = next((piece for piece in pieces if piece.square == king), None)
        if king_piece is None:
            continue
        for pinned_piece in pieces:
            if not pinned.get(pinned_piece.square, False) or pinned_piece.side != king_piece.side:
                continue
            if any(
                other.side != king_piece.side
                and bool(attack_masks[other] & (1 << pinned_piece.square))
                for other in pieces
            ):
                return 1
        if any(
            piece.side != king_piece.side and bool(attack_masks[piece] & (1 << king))
            for piece in pieces
        ):
            return 2
    for target in pieces:
        attackers = [
            piece
            for piece in pieces
            if piece != target and bool(attack_masks[piece] & (1 << target.square))
        ]
        if len(attackers) == 2:
            return 3
    edges = sum(
        _manhattan(pieces[left].square, pieces[right].square) <= 4
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    return 4 if edges >= 2 else None


def _pair_flags(
    left: Piece,
    right: Piece,
    kings: tuple[int, int],
    pinned: Mapping[int, bool],
    attack_masks: Mapping[Piece, int],
    us: int,
) -> int:
    left_mask = attack_masks[left]
    right_mask = attack_masks[right]
    left_attacks = bool(left_mask & (1 << right.square))
    right_attacks = bool(right_mask & (1 << left.square))
    flags = int(left_attacks)
    flags |= int(right_attacks) << 1
    flags |= int(left.side == right.side and (left_attacks or right_attacks)) << 2
    flags |= int(pinned[left.square]) << 3
    flags |= int(pinned[right.square]) << 4
    flags |= int(bool(left_mask & (1 << kings[1 if left.side == us else 0]))) << 5
    flags |= int(bool(right_mask & (1 << kings[1 if right.side == us else 0]))) << 6
    common = bool(left_mask & right_mask)
    flags |= int(common) << 7
    flags |= int(any(_chebyshev(left.square, king) <= 2 for king in kings)) << 8
    flags |= int(any(_chebyshev(right.square, king) <= 2 for king in kings)) << 9
    return flags


def _piece_attack_mask(position: ParsedPosition, piece: Piece) -> int:
    mask = 0
    for target in range(81):
        if _piece_attacks(position, piece, target):
            mask |= 1 << target
    return mask


def _side_attack_mask(
    pieces: Sequence[Piece], attack_masks: Mapping[Piece, int], side: int
) -> int:
    mask = 0
    for piece in pieces:
        if piece.side == side:
            mask |= attack_masks[piece]
    return mask


def _piece_attacks(position: ParsedPosition, piece: Piece, target: int) -> bool:
    if piece.square == target:
        return False
    from_file, from_rank = _file_rank(piece.square)
    to_file, to_rank = _file_rank(target)
    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank
    forward = -1 if piece.side == 0 else 1
    kind = piece.kind
    if kind == 0:
        return file_delta == 0 and rank_delta == forward
    if kind == 1:
        return (
            file_delta == 0
            and _sign(rank_delta) == forward
            and _path_clear(position, piece.square, target)
        )
    if kind == 2:
        return abs(file_delta) == 1 and rank_delta == 2 * forward
    if kind == 3:
        return (rank_delta == forward and abs(file_delta) <= 1) or (
            rank_delta == -forward and abs(file_delta) == 1
        )
    if kind in {4, 8, 9, 10, 11}:
        return (
            (rank_delta == forward and abs(file_delta) <= 1)
            or (rank_delta == 0 and abs(file_delta) == 1)
            or (rank_delta == -forward and file_delta == 0)
        )
    if kind == 5:
        return abs(file_delta) == abs(rank_delta) and _path_clear(position, piece.square, target)
    if kind == 6:
        return (file_delta == 0 or rank_delta == 0) and _path_clear(position, piece.square, target)
    if kind == 7:
        return abs(file_delta) <= 1 and abs(rank_delta) <= 1
    if kind == 12:
        return (
            abs(file_delta) == abs(rank_delta) and _path_clear(position, piece.square, target)
        ) or (
            (file_delta == 0 and abs(rank_delta) == 1) or (rank_delta == 0 and abs(file_delta) == 1)
        )
    if kind == 13:
        return (
            (file_delta == 0 or rank_delta == 0) and _path_clear(position, piece.square, target)
        ) or (abs(file_delta) == 1 and abs(rank_delta) == 1)
    raise AssertionError(kind)


def _path_clear(
    position: ParsedPosition, origin: int, target: int, ignored: int | None = None
) -> bool:
    from_file, from_rank = _file_rank(origin)
    to_file, to_rank = _file_rank(target)
    file_step = _sign(to_file - from_file)
    rank_step = _sign(to_rank - from_rank)
    file_value, rank_value = from_file + file_step, from_rank + rank_step
    while (file_value, rank_value) != (to_file, to_rank):
        square = (rank_value - 1) * 9 + (9 - file_value)
        if square != ignored and position.board[square] is not None:
            return False
        file_value += file_step
        rank_value += rank_step
    return True


def _square_attacked(position: ParsedPosition, square: int, attacker: int) -> bool:
    return any(
        piece is not None and piece.side == attacker and _piece_attacks(position, piece, square)
        for piece in position.board
    )


def _is_pinned(position: ParsedPosition, piece: Piece) -> bool:
    if piece.kind == 7:
        return False
    king = _king_square(position, piece.side)
    king_file, king_rank = _file_rank(king)
    piece_file, piece_rank = _file_rank(piece.square)
    file_delta, rank_delta = piece_file - king_file, piece_rank - king_rank
    if not (file_delta == 0 or rank_delta == 0 or abs(file_delta) == abs(rank_delta)):
        return False
    if not _path_clear(position, king, piece.square):
        return False
    file_step, rank_step = _sign(file_delta), _sign(rank_delta)
    file_value, rank_value = piece_file + file_step, piece_rank + rank_step
    while 1 <= file_value <= 9 and 1 <= rank_value <= 9:
        square = (rank_value - 1) * 9 + (9 - file_value)
        candidate = position.board[square]
        if candidate is not None:
            if candidate.side == piece.side:
                return False
            orthogonal = file_step == 0 or rank_step == 0
            if orthogonal and candidate.kind in {6, 13}:
                return True
            if not orthogonal and candidate.kind in {5, 12}:
                return True
            if file_step == 0 and candidate.kind == 1:
                lance_forward = -1 if candidate.side == 0 else 1
                return _sign(king_rank - rank_value) == lance_forward
            return False
        file_value += file_step
        rank_value += rank_step
    return False


def _embedding_sum(tensor: TensorView, rows: object, dimension: int) -> list[float]:
    typed_rows = rows if isinstance(rows, list) else list(rows)  # type: ignore[arg-type]
    result = [0.0] * dimension
    for row, sign in typed_rows:
        base = row * dimension
        for index in range(dimension):
            result[index] += sign * tensor.value(base + index)
    if typed_rows:
        denominator = float(len(typed_rows))
        result = [value / denominator for value in result]
    return result


def _linear(
    weights: TensorView,
    biases: TensorView,
    inputs: Sequence[float],
    outputs: int,
    *,
    relu: bool,
) -> list[float]:
    input_count = len(inputs)
    result = []
    for row in range(outputs):
        total = biases.value(row)
        base = row * input_count
        for column, value in enumerate(inputs):
            total += value * weights.value(base + column)
        result.append(max(0.0, total) if relu else total)
    return result


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _validate_legal_move_list(moves: Sequence[str]) -> list[tuple[str, int]]:
    if len(moves) > 600:
        raise ValueError("legal move list exceeds the defensive bound")
    result = [(move, encode_move(move)) for move in moves]
    if len({index for _, index in result}) != len(result):
        raise ValueError("legal move list contains a duplicate move index")
    return result


def _signed_bucket(key: bytes, buckets: int) -> tuple[int, int]:
    digest = hashlib.sha256(FEATURE_HASH_DOMAIN + struct.pack("<Q", HASH_SEED) + key).digest()
    return int.from_bytes(digest[:4], "little") % buckets, -1 if digest[4] & 1 else 1


def _relative_owner(piece: Piece, us: int) -> int:
    return int(piece.side != us)


def _signed_offsets(square: int, king: int) -> bytes:
    file_value, rank_value = _file_rank(square)
    king_file, king_rank = _file_rank(king)
    return struct.pack("<bb", file_value - king_file, rank_value - king_rank)


def _king_square(position: ParsedPosition, side: int) -> int:
    return next(
        piece.square
        for piece in position.board
        if piece is not None and piece.side == side and piece.kind == 7
    )


def _king_zone(king: int) -> tuple[int, ...]:
    king_file, king_rank = _file_rank(king)
    return tuple(
        (rank - 1) * 9 + (9 - file)
        for rank in range(max(1, king_rank - 2), min(9, king_rank + 2) + 1)
        for file in range(max(1, king_file - 2), min(9, king_file + 2) + 1)
    )


def _file_rank(square: int) -> tuple[int, int]:
    return 9 - square % 9, square // 9 + 1


def _chebyshev(left: int, right: int) -> int:
    left_file, left_rank = _file_rank(left)
    right_file, right_rank = _file_rank(right)
    return max(abs(left_file - right_file), abs(left_rank - right_rank))


def _manhattan(left: int, right: int) -> int:
    left_file, left_rank = _file_rank(left)
    right_file, right_rank = _file_rank(right)
    return abs(left_file - right_file) + abs(left_rank - right_rank)


def _square_index(square: str) -> int:
    return (ord(square[1]) - ord("a")) * 9 + (9 - int(square[0]))


def _index_square(index: int) -> str:
    rank, column = divmod(index, 9)
    return f"{9 - column}{chr(ord('a') + rank)}"


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _round_ties_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _as_float32(value: float) -> float:
    return _F32.unpack(_F32.pack(value))[0]


def _validate_hex_hash(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} SHA-256 must be lowercase hexadecimal")
    if value == "0" * 64:
        raise ValueError(f"{label} SHA-256 must not be zero")


def _validate_fixed_text(
    value: str, width: int, label: str, *, allow_full_width: bool = False
) -> None:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be ASCII") from error
    if (
        not encoded
        or len(encoded) > width
        or (len(encoded) == width and not allow_full_width)
        or any(byte < 0x21 or byte > 0x7E for byte in encoded)
    ):
        bound = "at most" if allow_full_width else "shorter than"
        raise ValueError(f"{label} must be nonempty printable ASCII {bound} {width} bytes")


def _write_fixed_text(
    target: bytearray,
    offset: int,
    width: int,
    value: str,
    *,
    allow_full_width: bool = False,
) -> None:
    _validate_fixed_text(value, width, "fixed text", allow_full_width=allow_full_width)
    encoded = value.encode("ascii")
    target[offset : offset + len(encoded)] = encoded


def _read_fixed_text(
    source: bytes,
    offset: int,
    width: int,
    label: str,
    *,
    allow_full_width: bool = False,
) -> str:
    field = source[offset : offset + width]
    if b"\0" not in field:
        if not allow_full_width or any(byte < 0x21 or byte > 0x7E for byte in field):
            raise ValueError(f"OSAVAL02 {label} lacks NUL padding")
        value = field
    else:
        value, padding = field.split(b"\0", 1)
        if not value or any(padding):
            raise ValueError(f"OSAVAL02 {label} padding is invalid")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"OSAVAL02 {label} is not ASCII") from error
    _validate_fixed_text(decoded, width, label, allow_full_width=allow_full_width)
    return decoded


def _pack_u32(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", target, offset, value)


def _u32(source: bytes, offset: int) -> int:
    return struct.unpack_from("<I", source, offset)[0]


def _u64(source: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", source, offset)[0]


def _assert_finite_tree(value: object) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("OSAVAL02 inference produced NaN or infinity")
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_tree(child)


__all__ = [
    "ARCHITECTURE_CONFIG_SHA256",
    "CONTAINER_OVERHEAD_BYTES",
    "DEFAULT_DATASET_MANIFEST_SHA256",
    "EXPORTER_VERSION",
    "FEATURE_SCHEMA_SHA256",
    "FORMAT_VERSION",
    "INPUT_NORMALIZATION_SHA256",
    "INT8_CONFIG_SHA256",
    "MAGIC",
    "MOVE_INDEX_SHA256",
    "TARGET_SEMANTICS_SHA256",
    "VARIANT_PAIR",
    "VARIANT_PRIMARY",
    "HistoryFacts",
    "Osaval02Model",
    "decode_move",
    "deterministic_test_tensors",
    "encode_move",
    "expected_parameter_count",
    "infer_osaval02",
    "inspect_osaval02",
    "load_osaval02",
    "parse_osaval02",
    "parse_sfen",
    "serialize_osaval02",
    "tensor_specs",
]
