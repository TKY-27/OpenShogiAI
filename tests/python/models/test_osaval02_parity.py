from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from open_shogi_training.phase10r_model import (
    DESCRIPTOR_OFFSET,
    HEADER_BYTES,
    MAX_MODEL_BYTES,
    VARIANT_PAIR,
    VARIANT_PRIMARY,
    HistoryFacts,
    deterministic_test_tensors,
    infer_osaval02,
    parse_osaval02,
    serialize_osaval02,
)

ROOT = Path(__file__).parents[3]
CORPUS_PATH = ROOT / "tests/fixtures/osaval02/parity-corpus.json"
GIT_COMMIT = "2196edc2310f691b9a7ae9714a382d70e2a7e3f6"
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-10
QUANTIZED_LOGIT_TOLERANCE = 2.0e-6
QUANTIZED_PROBABILITY_TOLERANCE = 1.0e-5
QUANTIZED_AUXILIARY_TOLERANCE = 2.5e-5
QUANTIZED_SCORE_CP_TOLERANCE = 1


def _artifact(variant: str, quantization: str) -> bytes:
    return serialize_osaval02(
        deterministic_test_tensors(variant),
        variant_id=variant,
        quantization=quantization,  # type: ignore[arg-type]
        training_run_reference="osaval02-parity-test",
        git_commit=GIT_COMMIT,
    )


def _run_native(model_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "open-shogi-core",
            "--example",
            "osaval02_infer",
            "--",
            os.fspath(model_path),
            os.fspath(CORPUS_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def _run_wasm(model_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "node",
            "--experimental-default-type=module",
            "scripts/osaval02_wasm_infer.mjs",
            os.fspath(model_path),
            os.fspath(CORPUS_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def _assert_close(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, float) or isinstance(right, float):
        assert isinstance(left, (int, float)) and isinstance(right, (int, float)), path
        assert math.isfinite(float(left)) and math.isfinite(float(right)), path
        assert math.isclose(
            float(left),
            float(right),
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ), f"{path}: {left!r} != {right!r}"
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_close(left_item, right_item, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


@pytest.mark.parametrize("variant", [VARIANT_PAIR, VARIANT_PRIMARY])
@pytest.mark.parametrize("quantization", ["float32", "int8"])
def test_python_native_and_actual_wasm_parity(
    tmp_path: Path, variant: str, quantization: str
) -> None:
    model_path = tmp_path / f"{variant}-{quantization}.osaval"
    model_path.write_bytes(_artifact(variant, quantization))
    model = parse_osaval02(model_path.read_bytes())
    corpus = json.loads(CORPUS_PATH.read_text())
    native = _run_native(model_path)
    wasm = _run_wasm(model_path)

    assert native["schema"] == "open_shogiai_osaval02_native_parity/v1"
    assert wasm["schema"] == "open_shogiai_osaval02_wasm_parity/v1"
    assert native["modelIdentity"] == wasm["modelIdentity"]
    _assert_close(native["fixtures"], wasm["fixtures"], "native-wasm")
    assert len(native["fixtures"]) == len(corpus["fixtures"]) == 12

    fixtures = {fixture["fixtureId"]: fixture for fixture in corpus["fixtures"]}
    for result in native["fixtures"]:
        fixture = fixtures[result["fixtureId"]]
        observed = result["inference"]
        history_data = fixture.get("history")
        history = (
            HistoryFacts(
                available=history_data["available"],
                repetition_count=history_data["repetitionCount"],
                continuous_check_by_us=history_data["continuousCheckByUs"],
                continuous_check_by_them=history_data["continuousCheckByThem"],
            )
            if history_data is not None
            else HistoryFacts()
        )
        legal_moves = [row["move"] for row in observed["legalMoves"]]
        reference = infer_osaval02(model, fixture["sfen"], legal_moves, history)
        _assert_close(observed, reference, result["fixtureId"])


def test_native_and_wasm_loaders_fail_closed_on_corruption(tmp_path: Path) -> None:
    artifact = _artifact(VARIANT_PAIR, "int8")
    corruptions: list[bytes] = []

    def mutated(offset: int, encoded: bytes) -> bytes:
        value = bytearray(artifact)
        value[offset : offset + len(encoded)] = encoded
        value[-32:] = hashlib.sha256(value[:-32]).digest()
        return bytes(value)

    corruptions.extend(
        [
            mutated(0, b"BROKEN!!"),
            mutated(8, struct.pack("<I", 99)),
            mutated(96, b"0" * 32),
            mutated(128, b"0" * 32),
            mutated(160, b"0" * 32),
            mutated(320, b"0" * 32),
            mutated(DESCRIPTOR_OFFSET + 48, struct.pack("<I", 99)),
            mutated(DESCRIPTOR_OFFSET + 80, struct.pack("<Q", 1)),
            mutated(DESCRIPTOR_OFFSET + 96, struct.pack("<f", float("nan"))),
            artifact[:-1],
        ]
    )
    wrong_checksum = bytearray(artifact)
    wrong_checksum[-1] ^= 1
    corruptions.append(bytes(wrong_checksum))

    nonfinite = bytearray(_artifact(VARIANT_PAIR, "float32"))
    nonfinite[HEADER_BYTES : HEADER_BYTES + 4] = struct.pack("<f", float("nan"))
    nonfinite[256:288] = hashlib.sha256(nonfinite[HEADER_BYTES:-32]).digest()
    nonfinite[-32:] = hashlib.sha256(nonfinite[:-32]).digest()
    corruptions.append(bytes(nonfinite))
    corruptions.append(b"\0" * (MAX_MODEL_BYTES + 1))

    for index, corrupted in enumerate(corruptions):
        model_path = tmp_path / f"corrupt-{index}.osaval"
        model_path.write_bytes(corrupted)
        native = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "-p",
                "open-shogi-core",
                "--example",
                "osaval02_infer",
                "--",
                os.fspath(model_path),
                os.fspath(CORPUS_PATH),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        wasm = subprocess.run(
            [
                "node",
                "--experimental-default-type=module",
                "scripts/osaval02_wasm_infer.mjs",
                os.fspath(model_path),
                os.fspath(CORPUS_PATH),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert native.returncode != 0
        assert wasm.returncode != 0


@pytest.mark.parametrize("variant", [VARIANT_PAIR, VARIANT_PRIMARY])
def test_float_and_int8_native_outputs_stay_within_frozen_fixture_thresholds(
    tmp_path: Path, variant: str
) -> None:
    outputs: dict[str, dict[str, Any]] = {}
    for quantization in ("float32", "int8"):
        model_path = tmp_path / f"{variant}-{quantization}.osaval"
        model_path.write_bytes(_artifact(variant, quantization))
        outputs[quantization] = _run_native(model_path)

    float_rows = outputs["float32"]["fixtures"]
    int8_rows = outputs["int8"]["fixtures"]
    for float_row, int8_row in zip(float_rows, int8_rows, strict=True):
        assert float_row["fixtureId"] == int8_row["fixtureId"]
        left = float_row["inference"]
        right = int8_row["inference"]
        assert [row["move"] for row in left["legalMoves"]] == [
            row["move"] for row in right["legalMoves"]
        ]
        assert (
            max(
                (
                    abs(float_move["logit"] - int8_move["logit"])
                    for float_move, int8_move in zip(
                        left["legalMoves"], right["legalMoves"], strict=True
                    )
                ),
                default=0.0,
            )
            <= QUANTIZED_LOGIT_TOLERANCE
        )
        probabilities = [
            *(abs(left["wdl"][key] - right["wdl"][key]) for key in left["wdl"]),
            *(
                abs(left["mate"]["probabilities"][key] - right["mate"]["probabilities"][key])
                for key in left["mate"]["probabilities"]
            ),
        ]
        assert max(probabilities) <= QUANTIZED_PROBABILITY_TOLERANCE
        auxiliary = [
            abs(left["score"]["transformed"] - right["score"]["transformed"]),
            abs(left["mate"]["distancePlies"] - right["mate"]["distancePlies"]),
            abs(left["uncertainty"]["logVariance"] - right["uncertainty"]["logVariance"]),
            abs(left["uncertainty"]["variance"] - right["uncertainty"]["variance"]),
        ]
        assert max(auxiliary) <= QUANTIZED_AUXILIARY_TOLERANCE
        assert (
            abs(left["score"]["calibratedCp"] - right["score"]["calibratedCp"])
            <= QUANTIZED_SCORE_CP_TOLERANCE
        )
