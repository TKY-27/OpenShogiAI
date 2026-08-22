from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest
from open_shogi_training.phase10r_model import (
    CONTAINER_OVERHEAD_BYTES,
    DESCRIPTOR_BYTES,
    DESCRIPTOR_OFFSET,
    HEADER_BYTES,
    MAX_MODEL_BYTES,
    VARIANT_PAIR,
    VARIANT_PRIMARY,
    deterministic_test_tensors,
    expected_parameter_count,
    inspect_osaval02,
    load_osaval02,
    parse_osaval02,
    serialize_osaval02,
    tensor_specs,
)

GIT_COMMIT = "2196edc2310f691b9a7ae9714a382d70e2a7e3f6"


def _artifact(variant: str = VARIANT_PAIR, quantization: str = "int8") -> bytes:
    return serialize_osaval02(
        deterministic_test_tensors(variant),
        variant_id=variant,
        quantization=quantization,  # type: ignore[arg-type]
        training_run_reference="osaval02-unit-test",
        git_commit=GIT_COMMIT,
    )


def _resign(data: bytearray) -> bytes:
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    return bytes(data)


@pytest.mark.parametrize("variant", [VARIANT_PAIR, VARIANT_PRIMARY])
@pytest.mark.parametrize("quantization,item_bytes", [("float32", 4), ("int8", 1)])
def test_export_is_deterministic_exact_size_and_inspectable(
    variant: str, quantization: str, item_bytes: int
) -> None:
    first = _artifact(variant, quantization)
    second = _artifact(variant, quantization)
    assert first == second
    assert len(first) == CONTAINER_OVERHEAD_BYTES + expected_parameter_count(variant) * item_bytes
    inspected = inspect_osaval02(first)
    assert inspected["variantId"] == variant
    assert inspected["quantization"] == quantization
    assert inspected["parameterCount"] == expected_parameter_count(variant)
    assert len(inspected["tensors"]) == len(tensor_specs(variant))


def test_export_rejects_wrong_tensor_set_shape_and_nonfinite_values() -> None:
    tensors = deterministic_test_tensors(VARIANT_PAIR)
    tensors.pop("trunk.0.bias")
    with pytest.raises(ValueError, match="missing"):
        serialize_osaval02(
            tensors,
            variant_id=VARIANT_PAIR,
            quantization="int8",
            training_run_reference="bad",
            git_commit=GIT_COMMIT,
        )

    tensors = deterministic_test_tensors(VARIANT_PAIR)
    tensors["trunk.0.bias"] = np.zeros((127,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        serialize_osaval02(
            tensors,
            variant_id=VARIANT_PAIR,
            quantization="int8",
            training_run_reference="bad",
            git_commit=GIT_COMMIT,
        )

    tensors = deterministic_test_tensors(VARIANT_PAIR)
    tensors["trunk.0.bias"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        serialize_osaval02(
            tensors,
            variant_id=VARIANT_PAIR,
            quantization="int8",
            training_run_reference="bad",
            git_commit=GIT_COMMIT,
        )


@pytest.mark.parametrize(
    ("offset", "encoded", "message"),
    [
        (0, b"BROKEN!!", "magic"),
        (8, struct.pack("<I", 99), "version"),
        (96, b"0" * 32, "feature schema"),
        (128, b"0" * 32, "architecture"),
        (160, b"0" * 32, "target semantics"),
        (320, b"0" * 32, "move-index"),
        (DESCRIPTOR_OFFSET + 48, struct.pack("<I", 99), "dtype"),
        (DESCRIPTOR_OFFSET + 56, struct.pack("<I", 1), "shape"),
        (DESCRIPTOR_OFFSET + 80, struct.pack("<Q", 1), "size or offset"),
        (DESCRIPTOR_OFFSET + 72, struct.pack("<Q", 2**64 - 1), "size or offset"),
        (DESCRIPTOR_OFFSET + 96, struct.pack("<f", float("nan")), "metadata"),
        (DESCRIPTOR_OFFSET + 104, struct.pack("<I", 0), "metadata"),
    ],
)
def test_parser_rejects_resigned_invalid_metadata(
    offset: int, encoded: bytes, message: str
) -> None:
    mutated = bytearray(_artifact())
    mutated[offset : offset + len(encoded)] = encoded
    with pytest.raises(ValueError, match=message):
        parse_osaval02(_resign(mutated))


def test_parser_rejects_truncation_checksum_payload_corruption_and_oversize() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match=r"truncated|SHA-256"):
        parse_osaval02(artifact[:-1])
    corrupted_checksum = bytearray(artifact)
    corrupted_checksum[-1] ^= 1
    with pytest.raises(ValueError, match="trailing SHA-256"):
        parse_osaval02(bytes(corrupted_checksum))
    corrupted_payload = bytearray(artifact)
    corrupted_payload[4096] ^= 1
    with pytest.raises(ValueError, match="payload SHA-256"):
        parse_osaval02(_resign(corrupted_payload))
    with pytest.raises(ValueError, match="16 MiB"):
        parse_osaval02(b"\0" * (MAX_MODEL_BYTES + 1))


def test_parser_rejects_trailing_section_and_nonfinite_float_parameter() -> None:
    artifact = _artifact()
    unsigned = bytearray(artifact[:-32])
    unsigned.append(0)
    struct.pack_into("<Q", unsigned, 20, len(unsigned) + 32)
    unsigned[256:288] = hashlib.sha256(unsigned[HEADER_BYTES:]).digest()
    trailing = bytes(unsigned) + hashlib.sha256(unsigned).digest()
    with pytest.raises(ValueError, match="trailing invalid section"):
        parse_osaval02(trailing)

    nonfinite = bytearray(_artifact(quantization="float32"))
    nonfinite[HEADER_BYTES : HEADER_BYTES + 4] = struct.pack("<f", float("nan"))
    nonfinite[256:288] = hashlib.sha256(nonfinite[HEADER_BYTES:-32]).digest()
    with pytest.raises(ValueError, match="non-finite parameter"):
        parse_osaval02(_resign(nonfinite))


def test_load_rejects_symlink(tmp_path: Path) -> None:
    model_path = tmp_path / "model.osaval"
    model_path.write_bytes(_artifact())
    link_path = tmp_path / "link.osaval"
    link_path.symlink_to(model_path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_osaval02(link_path)


def test_descriptor_table_is_closed_and_contiguous() -> None:
    artifact = _artifact()
    model = parse_osaval02(artifact)
    header = artifact[:HEADER_BYTES]
    descriptor_end = DESCRIPTOR_OFFSET + len(model.tensors) * DESCRIPTOR_BYTES
    assert not any(header[descriptor_end:])
