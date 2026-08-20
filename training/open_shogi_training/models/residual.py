"""Build an immutable handcrafted baseline artifact for residual targets."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from open_shogi_training.labeling.artifacts import stable_regular_descriptor, write_json_atomic
from open_shogi_training.models.dataset import LoadedExamples, hash_file

_ENGINE_OUTPUT_KEYS = frozenset(
    {"schema", "evaluatorProfile", "index", "sfen", "scoreCp", "elapsedNs"}
)
_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1


def build_residual_baseline(
    loaded: LoadedExamples,
    *,
    engine: Path,
    output: Path,
) -> dict[str, Any]:
    """Evaluate every teacher row with the exact handcrafted experimental binary."""

    if loaded.identity.target_semantics != "pure-value":
        raise ValueError("residual baseline input must be loaded with pure-value semantics")
    if loaded.identity.replay_manifest_sha256 is not None:
        raise ValueError("residual baseline construction does not accept replay examples")
    examples = tuple(row for row in loaded.examples if row.source_kind == "phase4_teacher")
    if not examples or len(examples) > 250_000 or len(examples) != len(loaded.examples):
        raise ValueError("residual baseline requires only bounded teacher examples")

    engine_sha256, engine_size = hash_file(engine, max_bytes=256 * 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="open-shogi-residual-") as temporary:
        temporary_dir = Path(temporary)
        input_path = temporary_dir / "positions.sfen"
        engine_output = temporary_dir / "handcrafted.jsonl"
        input_path.write_text(
            "".join(f"{example.sfen}\n" for example in examples), encoding="utf-8"
        )
        completed = subprocess.run(
            [
                str(engine.resolve(strict=True)),
                "model",
                "infer-handcrafted",
                "--profile",
                "handcrafted-experimental",
                "--input",
                str(input_path),
                "--output",
                str(engine_output),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=600,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[:4096]
            raise ValueError(f"handcrafted baseline inference failed: {detail}")
        if completed.stdout:
            raise ValueError("handcrafted baseline inference unexpectedly wrote stdout")
        rows = _read_engine_rows(engine_output, expected=len(examples))
    if hash_file(engine, max_bytes=256 * 1024 * 1024) != (engine_sha256, engine_size):
        raise ValueError("engine binary changed during residual baseline construction")

    records: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    for index, (example, row) in enumerate(zip(examples, rows, strict=True)):
        if (
            row["index"] != index
            or row["sfen"] != example.sfen
            or row["evaluatorProfile"] != "handcrafted-experimental"
        ):
            raise ValueError(f"handcrafted inference row {index} differs from its input")
        order_hash.update(f"{example.position_id}\0{example.sfen}\n".encode())
        records.append(
            {
                "positionId": example.position_id,
                "canonicalSfen": example.sfen,
                "scoreCp": row["scoreCp"],
            }
        )
    artifact = {
        "schema": "open_shogi_residual_baseline/v1",
        "buildVersion": 1,
        "evaluatorProfile": "handcrafted-experimental",
        "engine": {"sha256": engine_sha256, "size": engine_size},
        "datasetIdentity": {
            "datasetManifestSha256": loaded.identity.dataset_manifest_sha256,
            "positionsSha256": loaded.identity.positions_sha256,
            "labelsSha256": loaded.identity.labels_sha256,
            "labelManifestSha256": loaded.identity.label_manifest_sha256,
        },
        "positionOrderSha256": order_hash.hexdigest(),
        "records": records,
    }
    write_json_atomic(output, artifact, replace=False)
    artifact_sha256, artifact_size = hash_file(output, max_bytes=128 * 1024 * 1024)
    return {
        "schema": "open_shogi_residual_baseline_build/v1",
        "output": str(output),
        "sha256": artifact_sha256,
        "size": artifact_size,
        "records": len(records),
        "engine": artifact["engine"],
        "datasetIdentity": artifact["datasetIdentity"],
    }


def _read_engine_rows(path: Path, *, expected: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    observed = 0
    with stable_regular_descriptor(path) as descriptor:
        initial = os.fstat(descriptor)
        if not 0 < initial.st_size <= 128 * 1024 * 1024:
            raise ValueError("handcrafted inference artifact size is invalid")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            while line := stream.readline(1024 * 1024 + 1):
                observed += len(line)
                if len(line) > 1024 * 1024 or not line.endswith(b"\n"):
                    raise ValueError("handcrafted inference row is oversized or incomplete")
                try:
                    row = json.loads(line, object_pairs_hook=_unique_object)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError("handcrafted inference output is invalid JSONL") from error
                if not isinstance(row, dict) or set(row) != _ENGINE_OUTPUT_KEYS:
                    raise ValueError("handcrafted inference row violates its closed schema")
                if row.get("schema") != "phase5_handcrafted_inference/v1":
                    raise ValueError("handcrafted inference schema is unsupported")
                for field in ("index", "scoreCp", "elapsedNs"):
                    value = row.get(field)
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError(f"handcrafted inference {field} is invalid")
                if not _I32_MIN <= row["scoreCp"] <= _I32_MAX or row["elapsedNs"] < 0:
                    raise ValueError("handcrafted inference numeric value is out of bounds")
                if not isinstance(row.get("sfen"), str) or not row["sfen"]:
                    raise ValueError("handcrafted inference SFEN is invalid")
                rows.append(row)
                if len(rows) > expected:
                    raise ValueError("handcrafted inference produced too many rows")
        if observed != initial.st_size:
            raise ValueError("handcrafted inference output changed while read")
    if len(rows) != expected:
        raise ValueError("handcrafted inference row count differs from its inputs")
    return tuple(rows)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value
