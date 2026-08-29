"""Bounded, immutable execution of the frozen Phase 10R teacher stages.

The teacher-binding repair deliberately has no teacher client.  It consumes the
already-complete labels-v2 artifact, joins only the approved current train and
validation streams, and publishes a versioned child only after the existing
closed lineage validator and every frozen gate pass.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import select
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from open_shogi_training.phase10r import _load_yaml
from open_shogi_training.phase10r_campaign import (
    _candidate_parity,
    _evaluate_file,
    _incremental_parity,
)
from open_shogi_training.phase10r_execution import (
    _data_root,
    _json_bytes,
    _publish_immutable,
    _sha256_file,
    preparation_manifest_path,
    validate_preparation,
)
from open_shogi_training.phase10r_lineage import (
    BINDING_VERSION,
    CALIBRATION_INPUT_SCHEMA,
    CALIBRATION_RECEIPT_SCHEMA,
    CONTROL_PATH,
    CONTROL_SHA256,
    LINEAGE_SCHEMA,
    PARITY_RECEIPT_SCHEMA,
    _validate_calibration_input,
    lineage_path,
    load_teacher_binding_control,
    validate_candidate_lineage,
)
from open_shogi_training.phase10r_model import (
    MAX_NON_MATE_CP,
    VARIANT_PAIR,
    VARIANT_PRIMARY,
    parse_osaval02,
    parse_sfen,
    serialize_osaval02,
)
from open_shogi_training.phase10r_training import (
    CHECKPOINT_SCHEMA,
    Phase10RExample,
    Phase10RModel,
    Phase10RTrainingError,
    _seed_everything,
    loss_for_examples,
    resource_snapshot,
    select_device,
)

TEACHER_SOURCE: Final = "openshogiai_apery_teacher"
SCALE: Final = "1m"
VARIANTS: Final = (VARIANT_PAIR, VARIANT_PRIMARY)
STAGE3_ID: Final = "packed_sfen_value_and_ranking"
STAGE4_ID: Final = "approved_teacher_calibration"
STAGE3_SEED: Final = 20_260_729
STAGE3_BATCH_SIZE: Final = 128
STAGE3_LEARNING_RATE: Final = 1.0e-4
STAGE3_WEIGHT_DECAY: Final = 1.0e-4
STAGE3_TRAIN_ROWS: Final = 6_570
STAGE4_VALIDATION_ROWS: Final = 1_880
STAGE4_CP_ROWS: Final = 1_831
RSS_TARGET_BYTES: Final = 16 * 1024**3
MINIMUM_FREE_BYTES: Final = 150 * 1024**3
CALIBRATION_ROOT: Final = Path("local/phase10r-data/phase10r-teacher-binding")
CALIBRATION_DIRECTORY: Final = CALIBRATION_ROOT / SCALE
CALIBRATION_TRAIN_NAME: Final = "calibration-train.jsonl"
CALIBRATION_VALIDATION_NAME: Final = "calibration-validation.jsonl"
CALIBRATION_MANIFEST_NAME: Final = "calibration-input-manifest.json"
STAGE3_NAME: Final = "stage3-checkpoint.pt"
STAGE3_PROGRESS_NAME: Final = "stage3-progress.pt"
STAGE4_NAME: Final = "stage4-calibration.json"
ARTIFACT_NAME: Final = "teacher-bound-v1.osaval02"
PARITY_NAME: Final = "candidate-parity.json"
TACTICAL_SUITE_PATH: Final = Path("configs/evaluation/overall_champion_tactical_suite.json")
TACTICAL_SUITE_SHA256: Final = "90648b1fd1c6e2dacadc30204e1161e79815ba0bb7df2afd498450c9c8d4f79a"
TACTICAL_NODES: Final = 10_000
TACTICAL_HASH_MIB: Final = 32
TACTICAL_MAX_DEPTH: Final = 8
TACTICAL_TIMEOUT_SECONDS: Final = 10
THROUGHPUT_NODES: Final = 25_000
RANKING_UNRETURNED_SCORE: Final = -16.0
_PIECE_NAMES: Final = (
    "P",
    "L",
    "N",
    "S",
    "G",
    "B",
    "R",
    "K",
    "+P",
    "+L",
    "+N",
    "+S",
    "+B",
    "+R",
)
_HAND_INDEX: Final = {piece: index for index, piece in enumerate("PLNSGBR")}
_HAND_ORDER: Final = "RBGSNLP"


class Phase10RTeacherBindingError(RuntimeError):
    """Raised when a frozen teacher-binding stage cannot be proven."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _load_json_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RTeacherBindingError(f"{context} is not a regular file: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Phase10RTeacherBindingError(f"{context} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Phase10RTeacherBindingError(f"cannot read {context}: {path}") from error
    if not isinstance(value, dict):
        raise Phase10RTeacherBindingError(f"{context} must be an object")
    return value


def _read_jsonl_strict(path: Path, context: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RTeacherBindingError(f"{context} is not a regular file: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Phase10RTeacherBindingError(f"{context} contains duplicate key {key!r}")
            value[key] = item
        return value

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Phase10RTeacherBindingError(
                        f"{context} contains a blank row at line {line_number}"
                    )
                value = json.loads(
                    line,
                    object_pairs_hook=unique,
                    parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
                )
                if not isinstance(value, dict):
                    raise Phase10RTeacherBindingError(
                        f"{context} row {line_number} is not an object"
                    )
                rows.append(value)
    except Phase10RTeacherBindingError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Phase10RTeacherBindingError(f"cannot read {context} row {len(rows) + 1}") from error
    return rows


def _regular_path(root: Path, relative: str, context: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise Phase10RTeacherBindingError(f"{context} path is invalid")
    path = root / Path(relative)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise Phase10RTeacherBindingError(
            f"{context} path is unavailable or escapes the root"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise Phase10RTeacherBindingError(f"{context} path is not a regular non-symlink file")
    return resolved


def _artifact_ref(
    root: Path,
    path: Path,
    context: str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise Phase10RTeacherBindingError(f"{context} escapes the repository") from error
    if path.is_symlink() or not path.is_file():
        raise Phase10RTeacherBindingError(f"{context} is not a regular non-symlink file: {path}")
    digest = _sha256_file(path)
    size = path.stat().st_size
    if expected_sha256 is not None and digest != expected_sha256:
        raise Phase10RTeacherBindingError(f"{context} SHA-256 differs from the frozen control")
    if expected_bytes is not None and size != expected_bytes:
        raise Phase10RTeacherBindingError(f"{context} byte length differs from the frozen control")
    return {
        "path": resolved.relative_to(root.resolve(strict=True)).as_posix(),
        "sha256": digest,
        "bytes": size,
    }


def _canonical_sfen(sfen: str) -> str:
    """Serialize a parsed SFEN in the exact canonical form used by the rules core."""

    position = parse_sfen(sfen)
    ranks: list[str] = []
    for rank in range(9):
        encoded: list[str] = []
        empty = 0
        for column in range(9):
            piece = position.board[rank * 9 + column]
            if piece is None:
                empty += 1
                continue
            if empty:
                encoded.append(str(empty))
                empty = 0
            name = _PIECE_NAMES[piece.kind]
            character = name[-1]
            if piece.side == 1:
                character = character.lower()
            encoded.append(("+" if name.startswith("+") else "") + character)
        if empty:
            encoded.append(str(empty))
        ranks.append("".join(encoded))

    hand_parts: list[str] = []
    for side, hands in enumerate(position.hands):
        for piece in _HAND_ORDER:
            count = hands[_HAND_INDEX[piece]]
            if not count:
                continue
            character = piece if side == 0 else piece.lower()
            hand_parts.append((str(count) if count > 1 else "") + character)
    hand = "".join(hand_parts) or "-"
    side = "b" if position.side_to_move == 0 else "w"
    return f"{'/'.join(ranks)} {side} {hand} {position.move_number}"


def _validate_canonical_sfen(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise Phase10RTeacherBindingError(f"{context} is not a string")
    try:
        canonical = _canonical_sfen(value)
    except (TypeError, ValueError) as error:
        raise Phase10RTeacherBindingError(f"{context} is invalid") from error
    if canonical != value:
        raise Phase10RTeacherBindingError(f"{context} is not canonical")
    return canonical


def _preparation_file(
    root: Path, preparation_path: Path, preparation: Mapping[str, Any], name: str
) -> Path:
    files = preparation.get("files")
    entry = files.get(name) if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping):
        raise Phase10RTeacherBindingError(f"preparation file is absent: {name}")
    path = _regular_path(root, str(preparation_path.parent.joinpath(name).relative_to(root)), name)
    _artifact_ref(
        root,
        path,
        f"preparation {name}",
        expected_sha256=str(entry.get("sha256")),
        expected_bytes=int(entry.get("bytes")),
    )
    return path


def _validate_label_teacher(label: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    identity = control["teacher_binding_identity"]
    teacher = identity["teacher"]
    record = label.get("teacher")
    if not isinstance(record, Mapping):
        raise Phase10RTeacherBindingError("label teacher identity is missing")
    expected_eval = teacher["eval_files"]
    if (
        record.get("name") != teacher["name"]
        or record.get("version") != teacher["version"]
        or record.get("binary_sha256") != teacher["binary"]["sha256"]
        or record.get("binary_size") != teacher["binary"]["size"]
        or record.get("eval_files") != expected_eval
        or record.get("options") != teacher["options"]
    ):
        raise Phase10RTeacherBindingError("label teacher identity differs from the frozen identity")
    if label.get("config_sha256") != teacher["config_semantic_sha256"]:
        raise Phase10RTeacherBindingError("label teacher config identity differs from the freeze")
    if label.get("score_pov") != "side_to_move":
        raise Phase10RTeacherBindingError("label score perspective differs from the frozen target")


def _score_mapping(score: object, context: str) -> tuple[str, int]:
    if not isinstance(score, Mapping) or set(score) != {"kind", "value"}:
        raise Phase10RTeacherBindingError(f"{context} is malformed")
    kind = score.get("kind")
    value = score.get("value")
    if kind not in {"cp", "mate"} or isinstance(value, bool) or not isinstance(value, int):
        raise Phase10RTeacherBindingError(f"{context} is malformed")
    if kind == "cp" and abs(value) > MAX_NON_MATE_CP:
        raise Phase10RTeacherBindingError(f"{context} leaves the non-mate namespace")
    if kind == "mate" and value == 0:
        raise Phase10RTeacherBindingError(f"{context} has zero mate distance")
    return str(kind), value


def _validate_label_row(
    label: Mapping[str, Any], current: Mapping[str, Any] | None, control: Mapping[str, Any]
) -> None:
    if label.get("schema") != "phase4_teacher_label/v1":
        raise Phase10RTeacherBindingError("label schema differs from labels-v2")
    canonical_sfen = _validate_canonical_sfen(label.get("canonical_sfen"), "label canonical_sfen")
    try:
        position = parse_sfen(canonical_sfen)
    except (TypeError, ValueError) as error:
        raise Phase10RTeacherBindingError("label SFEN cannot be parsed") from error
    state_hash = label.get("canonical_state_sha256")
    if state_hash != hashlib.sha256(position.canonical_state.encode("ascii")).hexdigest():
        raise Phase10RTeacherBindingError("label canonical-state hash differs")
    if label.get("side_to_move") != ("black" if position.side_to_move == 0 else "white"):
        raise Phase10RTeacherBindingError("label side-to-move differs from its SFEN")
    _validate_label_teacher(label, control)
    split = label.get("split")
    if split not in {"train", "validation", "test"}:
        raise Phase10RTeacherBindingError(
            f"label split is not an approved labels-v2 split: {split}"
        )
    position_id = label.get("position_id")
    if not isinstance(position_id, str) or not position_id:
        raise Phase10RTeacherBindingError("label position ID is missing")
    score_kind, _ = _score_mapping(label.get("score"), "label score")
    candidates = label.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise Phase10RTeacherBindingError("label MultiPV prefix is invalid")
    roots: list[str] = []
    multipv: list[int] = []
    for index, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, Mapping):
            raise Phase10RTeacherBindingError("label MultiPV candidate is invalid")
        number = candidate.get("multipv")
        if isinstance(number, bool) or not isinstance(number, int):
            raise Phase10RTeacherBindingError("label MultiPV rank is invalid")
        multipv.append(number)
        pv = candidate.get("pv")
        if not isinstance(pv, list) or not pv or not isinstance(pv[0], str):
            raise Phase10RTeacherBindingError("label MultiPV root is invalid")
        roots.append(pv[0])
        _score_mapping(candidate.get("score"), f"label candidate {index} score")
    if multipv != list(range(1, len(candidates) + 1)) or len(set(roots)) != len(roots):
        raise Phase10RTeacherBindingError("label MultiPV prefix is not contiguous")
    if label.get("bestmove") != roots[0] or label.get("score") != candidates[0].get("score"):
        raise Phase10RTeacherBindingError("label bestmove or primary score is inconsistent")
    if current is not None:
        if current.get("sfen") != canonical_sfen:
            raise Phase10RTeacherBindingError("label/current canonical SFEN join is not exact")
        legal_moves = current.get("legal_moves")
        if not isinstance(legal_moves, list) or any(root not in legal_moves for root in roots):
            raise Phase10RTeacherBindingError(
                "teacher MultiPV root is outside the current legal mask"
            )
        if current.get("source") not in {"aobazero", "wcsc", "denryu"}:
            raise Phase10RTeacherBindingError("teacher label joined a non-frozen current source")
    if score_kind == "mate" and abs(_score_mapping(label.get("score"), "label score")[1]) > 512:
        raise Phase10RTeacherBindingError("label mate distance exceeds the frozen head bound")


def _teacher_raw(label: Mapping[str, Any]) -> dict[str, Any]:
    candidates = label["candidates"]
    return {
        "label_position_id": label["position_id"],
        "label_canonical_state_sha256": label["canonical_state_sha256"],
        "label_split": label["split"],
        "config_sha256": label["config_sha256"],
        "score": dict(label["score"]),
        "candidates": [
            {
                "multipv": candidate["multipv"],
                "root_move": candidate["pv"][0],
                "score": dict(candidate["score"]),
            }
            for candidate in candidates
        ],
    }


def _joined_row(
    current: Mapping[str, Any], label: Mapping[str, Any], identity_sha256: str
) -> dict[str, Any]:
    legal_moves = current.get("legal_moves")
    if not isinstance(legal_moves, list) or not legal_moves:
        raise Phase10RTeacherBindingError("current row lacks a legal move mask")
    candidate_roots = {
        candidate["pv"][0]: candidate["multipv"] for candidate in label["candidates"]
    }
    ranking_scores = [
        float(len(label["candidates"]) - candidate_roots[move] + 1)
        if move in candidate_roots
        else RANKING_UNRETURNED_SCORE
        for move in legal_moves
    ]
    raw_targets = current.get("raw_targets")
    if not isinstance(raw_targets, Mapping):
        raise Phase10RTeacherBindingError("current row raw target metadata is missing")
    raw = dict(raw_targets)
    raw["prepared_source"] = current["source"]
    raw["prepared_split"] = current["split"]
    raw["teacher_binding"] = _teacher_raw(label)
    row = {
        "schema": "phase10r_training_example/v1",
        "sfen": current["sfen"],
        "source": TEACHER_SOURCE,
        "artifact_id": current["artifact_id"],
        "record_id": current["record_id"],
        "split": current["split"],
        "weight": current.get("weight", 1.0),
        "legal_moves": list(legal_moves),
        "played_move": current.get("played_move"),
        "wdl": current.get("wdl"),
        "wdl_mask": current.get("wdl_mask", False),
        "uncertainty_mask": current.get("uncertainty_mask", False),
        "ranking_scores": ranking_scores,
        "ranking_mask": True,
        "teacher_identity": identity_sha256,
        "raw_targets": raw,
    }
    try:
        Phase10RExample.from_mapping(row).validate()
    except Phase10RTrainingError as error:
        raise Phase10RTeacherBindingError(f"joined calibration row is invalid: {error}") from error
    return row


def _ensure_payload(path: Path, payload: bytes, context: str) -> dict[str, Any]:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise Phase10RTeacherBindingError(f"{context} is not a regular immutable file")
        if path.stat().st_size != len(payload) or _sha256_file(path) != digest:
            raise Phase10RTeacherBindingError(f"{context} already exists with different bytes")
    else:
        written_digest, written_bytes = _publish_immutable(path, payload)
        if written_digest != digest or written_bytes != len(payload):
            raise Phase10RTeacherBindingError(f"{context} publish identity changed")
    return {"path": path.as_posix(), "sha256": digest, "bytes": len(payload)}


def _calibration_input_path(root: Path) -> Path:
    return root / CALIBRATION_DIRECTORY / CALIBRATION_MANIFEST_NAME


def prepare_teacher_binding(root: Path, scale: str) -> dict[str, Any]:
    """Join the immutable labels-v2 artifact to only the approved current rows."""

    if scale != SCALE:
        raise Phase10RTeacherBindingError("teacher binding is frozen only for the 1m scale")
    root = root.resolve(strict=True)
    control = load_teacher_binding_control(root)
    preparation_path = preparation_manifest_path(root, scale)
    preparation = validate_preparation(root, scale)
    pretraining = control["pretraining"]["preparation_manifest"]
    if preparation.get("manifest_sha256") != pretraining["manifest_sha256"]:
        raise Phase10RTeacherBindingError("current preparation manifest identity differs")
    if _sha256_file(preparation_path) != pretraining["file_sha256"]:
        raise Phase10RTeacherBindingError("current preparation manifest file hash differs")

    teacher_identity = control["teacher_binding_identity"]
    identity_sha256 = teacher_identity["identity_sha256"]
    labels_identity = teacher_identity["labels"]
    labels_path = _regular_path(root, labels_identity["rows"]["path"], "labels-v2 rows")
    _artifact_ref(
        root,
        labels_path,
        "labels-v2 rows",
        expected_sha256=labels_identity["rows"]["sha256"],
        expected_bytes=labels_path.stat().st_size,
    )
    labels = _read_jsonl_strict(labels_path, "labels-v2 rows")
    if len(labels) != labels_identity["rows"]["records"]:
        raise Phase10RTeacherBindingError("labels-v2 record count differs from the frozen control")

    current_by_sfen: dict[str, dict[str, Any]] = {}
    for name in ("base-train-aobazero.jsonl", "base-validation.jsonl"):
        path = _preparation_file(root, preparation_path, preparation, name)
        expected = preparation["files"][name]["rows"]
        rows = _read_jsonl_strict(path, f"current preparation {name}")
        if len(rows) != expected:
            raise Phase10RTeacherBindingError(f"current preparation count differs for {name}")
        for row in rows:
            try:
                canonical = _validate_canonical_sfen(row.get("sfen"), f"current {name} SFEN")
                example = Phase10RExample.from_mapping(row)
                example.validate()
            except (Phase10RTrainingError, Phase10RTeacherBindingError) as error:
                raise Phase10RTeacherBindingError(
                    f"current preparation row is invalid: {error}"
                ) from error
            if canonical != row.get("sfen") or row.get("source") not in {
                "aobazero",
                "wcsc",
                "denryu",
            }:
                raise Phase10RTeacherBindingError(
                    "current calibration source row is not a frozen factual lane"
                )
            if row.get("split") not in {"train", "validation"}:
                raise Phase10RTeacherBindingError("current calibration row uses a forbidden split")
            if canonical in current_by_sfen:
                raise Phase10RTeacherBindingError("duplicate current preparation SFEN")
            current_by_sfen[canonical] = row

    seen_sfen: set[str] = set()
    seen_position_ids: set[str] = set()
    joined: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    excluded = {"phase4_test": 0, "absent_from_current_preparation": 0, "split_mismatch": 0}
    for label in labels:
        canonical = _validate_canonical_sfen(label.get("canonical_sfen"), "label canonical_sfen")
        position_id = label.get("position_id")
        if canonical in seen_sfen:
            raise Phase10RTeacherBindingError("duplicate labels-v2 canonical SFEN")
        if not isinstance(position_id, str) or position_id in seen_position_ids:
            raise Phase10RTeacherBindingError("duplicate labels-v2 position ID")
        seen_sfen.add(canonical)
        seen_position_ids.add(position_id)
        current = current_by_sfen.get(canonical)
        _validate_label_row(label, current, control)
        split = label["split"]
        if split == "test":
            excluded["phase4_test"] += 1
            continue
        if current is None:
            excluded["absent_from_current_preparation"] += 1
            continue
        if current["split"] != split:
            excluded["split_mismatch"] += 1
            continue
        joined[split].append(_joined_row(current, label, identity_sha256))

    expected_excluded = control["calibration"]["exclusions"]
    if excluded != expected_excluded:
        raise Phase10RTeacherBindingError(
            f"calibration exclusions differ: {excluded} != {expected_excluded}"
        )
    expected_rows = control["calibration"]["expected_rows"]
    counts = {
        "train": len(joined["train"]),
        "validation": len(joined["validation"]),
        "validation_cp": sum(
            row["raw_targets"]["teacher_binding"]["score"]["kind"] == "cp"
            for row in joined["validation"]
        ),
    }
    if counts != expected_rows:
        raise Phase10RTeacherBindingError(
            f"calibration row counts differ: {counts} != {expected_rows}"
        )

    train_payload = b"".join(_json_bytes(row) for row in joined["train"])
    validation_payload = b"".join(_json_bytes(row) for row in joined["validation"])
    train_path = root / CALIBRATION_DIRECTORY / CALIBRATION_TRAIN_NAME
    validation_path = root / CALIBRATION_DIRECTORY / CALIBRATION_VALIDATION_NAME
    train_ref = _ensure_payload(train_path, train_payload, "calibration train rows")
    validation_ref = _ensure_payload(
        validation_path, validation_payload, "calibration validation rows"
    )
    control_path = root / CONTROL_PATH
    control_ref = _artifact_ref(
        root, control_path, "teacher-binding control", expected_sha256=CONTROL_SHA256
    )
    manifest = {
        "schema": CALIBRATION_INPUT_SCHEMA,
        "status": "passed",
        "binding_version": BINDING_VERSION,
        "scale": scale,
        "control_sha256": control_ref["sha256"],
        "teacher_binding_identity_sha256": identity_sha256,
        "preparation_manifest_sha256": pretraining["manifest_sha256"],
        "label_manifest_sha256": labels_identity["manifest"]["sha256"],
        "labels_sha256": labels_identity["rows"]["sha256"],
        "rows": counts,
        "excluded_rows": excluded,
        "new_teacher_calls": 0,
        "files": {
            "train": {
                "path": train_ref["path"],
                "sha256": train_ref["sha256"],
                "bytes": train_ref["bytes"],
            },
            "validation": {
                "path": validation_ref["path"],
                "sha256": validation_ref["sha256"],
                "bytes": validation_ref["bytes"],
            },
        },
    }
    manifest_payload = _json_bytes(manifest)
    manifest_path = _calibration_input_path(root)
    if manifest_path.exists() or manifest_path.is_symlink():
        existing = _load_json_object(manifest_path, "calibration input manifest")
        if existing != manifest or manifest_path.read_bytes() != manifest_payload:
            raise Phase10RTeacherBindingError(
                "calibration input manifest already exists with different bytes"
            )
    else:
        _publish_immutable(manifest_path, manifest_payload)
    loaded = _load_json_object(manifest_path, "calibration input manifest")
    try:
        _validate_calibration_input(
            root,
            loaded,
            control_sha256=control_ref["sha256"],
            identity_sha256=identity_sha256,
            preparation_manifest_sha256=pretraining["manifest_sha256"],
            label_manifest_sha256=labels_identity["manifest"]["sha256"],
            labels_sha256=labels_identity["rows"]["sha256"],
        )
    except ValueError as error:
        raise Phase10RTeacherBindingError(
            f"published calibration input failed validation: {error}"
        ) from error
    return {
        "schema": CALIBRATION_INPUT_SCHEMA,
        "status": "passed",
        "scale": scale,
        "manifest": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "rows": counts,
        "excluded_rows": excluded,
        "forbidden_split_rows": 0,
        "new_teacher_calls": 0,
        "files": {"train": train_ref, "validation": validation_ref},
    }


def _input_manifest(root: Path, control: Mapping[str, Any]) -> tuple[Path, dict[str, Any], str]:
    path = _calibration_input_path(root)
    value = _load_json_object(path, "calibration input manifest")
    identity = control["teacher_binding_identity"]
    pretraining = control["pretraining"]
    labels = identity["labels"]
    control_path = root / CONTROL_PATH
    _validate_calibration_input(
        root,
        value,
        control_sha256=_sha256_file(control_path),
        identity_sha256=identity["identity_sha256"],
        preparation_manifest_sha256=pretraining["preparation_manifest"]["manifest_sha256"],
        label_manifest_sha256=labels["manifest"]["sha256"],
        labels_sha256=labels["rows"]["sha256"],
    )
    if path.read_bytes() != _json_bytes(value):
        raise Phase10RTeacherBindingError("calibration input manifest is not canonical JSON")
    return path, value, _sha256_file(path)


def _teacher_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("raw_targets")
    binding = raw.get("teacher_binding") if isinstance(raw, Mapping) else None
    if not isinstance(binding, Mapping):
        raise Phase10RTeacherBindingError("calibration row lacks teacher-binding metadata")
    score = binding.get("score")
    _score_mapping(score, "calibration teacher score")
    candidates = binding.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise Phase10RTeacherBindingError("calibration row lacks teacher candidates")
    return binding


def _load_calibration_examples(
    root: Path, control: Mapping[str, Any], variant: str
) -> tuple[Path, dict[str, Any], str, list[Phase10RExample], list[Phase10RExample]]:
    if variant not in VARIANTS:
        raise Phase10RTeacherBindingError(f"unsupported teacher-binding variant: {variant}")
    manifest_path, manifest, manifest_sha256 = _input_manifest(root, control)
    result: dict[str, list[Phase10RExample]] = {}
    for name, expected_split, expected_count in (
        ("train", "train", STAGE3_TRAIN_ROWS),
        ("validation", "validation", STAGE4_VALIDATION_ROWS),
    ):
        file_ref = manifest["files"][name]
        path = _regular_path(root, file_ref["path"], f"calibration {name}")
        _artifact_ref(
            root,
            path,
            f"calibration {name}",
            expected_sha256=file_ref["sha256"],
            expected_bytes=file_ref["bytes"],
        )
        rows = _read_jsonl_strict(path, f"calibration {name}")
        if len(rows) != expected_count:
            raise Phase10RTeacherBindingError(f"calibration {name} row count differs")
        examples: list[Phase10RExample] = []
        cp_count = 0
        for row in rows:
            if (
                row.get("split") != expected_split
                or row.get("teacher_identity")
                != control["teacher_binding_identity"]["identity_sha256"]
            ):
                raise Phase10RTeacherBindingError(f"calibration {name} row identity differs")
            try:
                base = Phase10RExample.from_mapping(row)
                base.validate()
            except Phase10RTrainingError as error:
                raise Phase10RTeacherBindingError(
                    f"calibration {name} row is invalid: {error}"
                ) from error
            binding = _teacher_payload(row)
            score_kind, score_value = _score_mapping(binding["score"], "calibration teacher score")
            if expected_split == "validation" and score_kind == "cp":
                cp_count += 1
            enriched = dict(row)
            enriched["played_move"] = None
            enriched["wdl"] = None
            enriched["wdl_mask"] = False
            enriched["uncertainty_mask"] = False
            if score_kind == "cp" and variant == VARIANT_PRIMARY:
                enriched["score_cp"] = score_value
                enriched["score_mask"] = True
            elif score_kind == "mate":
                enriched["mate_kind"] = 2 if score_value > 0 else 0
                enriched["mate_distance"] = abs(float(score_value))
                enriched["mate_mask"] = True
            else:
                enriched["score_mask"] = False
            try:
                example = Phase10RExample.from_mapping(enriched)
                example.validate()
            except Phase10RTrainingError as error:
                raise Phase10RTeacherBindingError(
                    f"variant calibration row is invalid: {error}"
                ) from error
            examples.append(example)
        if expected_split == "validation" and cp_count != STAGE4_CP_ROWS:
            raise Phase10RTeacherBindingError(f"validation cp count differs: {cp_count}")
        result[name] = examples
    return manifest_path, manifest, manifest_sha256, result["train"], result["validation"]


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _cpu_state(model: Phase10RModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, RuntimeError) as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise Phase10RTeacherBindingError(f"cannot save resumable checkpoint: {path}") from error


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _stage3_progress_payload(
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    manifest_sha256: str,
    parent_checkpoint_sha256: str,
    identity_sha256: str,
    step: int,
    cursor: int,
    order: Sequence[int],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "variant_id": model.variant_id,
        "manifest_sha256": manifest_sha256,
        "seed": STAGE3_SEED,
        "stage_id": STAGE3_ID,
        "step": step,
        "stream_index": cursor,
        "completed": cursor == STAGE3_TRAIN_ROWS,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "teacher_binding_identity_sha256": identity_sha256,
        "order": list(order),
        "model_state": _cpu_state(model),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "scheduler_state": _cpu_tree(scheduler.state_dict()),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().to(device="cpu"),
        "metrics": dict(metrics),
    }


def _load_parent(
    root: Path, control: Mapping[str, Any], variant: str
) -> tuple[Path, Path, dict[str, torch.Tensor], str, str]:
    selected = next(
        item for item in control["pretraining"]["variants"] if item["variant_id"] == variant
    )
    checkpoint_ref = selected["stage2_checkpoint"]
    artifact_ref = selected["stage2_artifact"]
    checkpoint = _regular_path(root, checkpoint_ref["path"], f"{variant} parent checkpoint")
    artifact = _regular_path(root, artifact_ref["path"], f"{variant} parent artifact")
    parent_checkpoint_sha256 = _sha256_file(checkpoint)
    if parent_checkpoint_sha256 != checkpoint_ref["sha256"]:
        raise Phase10RTeacherBindingError(f"{variant} parent checkpoint hash changed")
    parent_artifact_sha256 = _sha256_file(artifact)
    if parent_artifact_sha256 != artifact_ref["sha256"]:
        raise Phase10RTeacherBindingError(f"{variant} parent artifact hash changed")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise Phase10RTeacherBindingError(
            f"{variant} parent checkpoint cannot be loaded"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("variant_id") != variant
        or payload.get("manifest_sha256")
        != control["pretraining"]["preparation_manifest"]["manifest_sha256"]
        or payload.get("stage_id") != "source_specific_wdl_value_pretraining"
        or payload.get("completed") is not True
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise Phase10RTeacherBindingError(
            f"{variant} parent checkpoint is not an immutable stage-2 parent"
        )
    model = Phase10RModel(variant, seed=STAGE3_SEED)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise Phase10RTeacherBindingError(f"{variant} parent tensor set is incompatible") from error
    return checkpoint, artifact, _cpu_state(model), parent_checkpoint_sha256, parent_artifact_sha256


def _resource_guard(data_root: Path) -> dict[str, int | bool]:
    snapshot = resource_snapshot(data_root, minimum_free_bytes=MINIMUM_FREE_BYTES)
    if not snapshot["disk_passed"]:
        raise Phase10RTeacherBindingError("free disk crossed the frozen 150 GiB floor")
    if int(snapshot["peak_rss_bytes"]) > RSS_TARGET_BYTES:
        raise Phase10RTeacherBindingError("RSS exceeded the frozen 16 GiB target")
    return snapshot


def _state_changed(parent: Mapping[str, torch.Tensor], child: Mapping[str, torch.Tensor]) -> bool:
    if set(parent) != set(child):
        raise Phase10RTeacherBindingError("parent and child tensor sets differ")
    return any(not torch.equal(parent[name], child[name]) for name in parent)


def _load_stage3_final(
    path: Path,
    *,
    variant: str,
    manifest_sha256: str,
    parent_checkpoint_sha256: str,
    identity_sha256: str,
) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError) as error:
        raise Phase10RTeacherBindingError("stage-3 checkpoint cannot be decoded") from error
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("variant_id") != variant
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("stage_id") != STAGE3_ID
        or value.get("completed") is not True
        or value.get("parent_checkpoint_sha256") != parent_checkpoint_sha256
        or value.get("teacher_binding_identity_sha256") != identity_sha256
        or value.get("stream_index") != STAGE3_TRAIN_ROWS
        or not isinstance(value.get("model_state"), Mapping)
    ):
        raise Phase10RTeacherBindingError("completed stage-3 checkpoint identity is invalid")
    return value


def _run_stage3(
    root: Path,
    control: Mapping[str, Any],
    variant: str,
    train_examples: Sequence[Phase10RExample],
    manifest_sha256: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    child_dir = _data_root(root) / "checkpoints" / "phase10r" / SCALE / variant / BINDING_VERSION
    final_path = child_dir / STAGE3_NAME
    progress_path = child_dir / STAGE3_PROGRESS_NAME
    checkpoint_path, _parent_artifact_path, parent_state, parent_checkpoint_sha256, _ = (
        _load_parent(root, control, variant)
    )
    identity_sha256 = control["teacher_binding_identity"]["identity_sha256"]
    if final_path.exists() or final_path.is_symlink():
        if final_path.is_symlink() or not final_path.is_file():
            raise Phase10RTeacherBindingError("stage-3 final checkpoint is not a regular file")
        final = _load_stage3_final(
            final_path,
            variant=variant,
            manifest_sha256=manifest_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            identity_sha256=identity_sha256,
        )
        child_state = {
            name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, tensor in final["model_state"].items()
            if isinstance(tensor, torch.Tensor)
        }
        if not _state_changed(parent_state, child_state):
            raise Phase10RTeacherBindingError(
                "stage-3 checkpoint retained unchanged parent tensors"
            )
        return {
            "status": "passed",
            "stage_id": STAGE3_ID,
            "resumed": True,
            "checkpoint": final_path,
            "checkpoint_sha256": _sha256_file(final_path),
            "steps": final.get("step"),
            "stream_rows_consumed": final.get("stream_index"),
            "metrics": final.get("metrics", {}),
            "parent_checkpoint": checkpoint_path,
        }
    if progress_path.exists() or progress_path.is_symlink():
        if progress_path.is_symlink() or not progress_path.is_file():
            raise Phase10RTeacherBindingError("stage-3 progress is not a regular file")
        if not resume:
            raise Phase10RTeacherBindingError("stage-3 progress exists; --resume is required")
    elif not resume:
        raise Phase10RTeacherBindingError(
            "teacher calibration requires --resume for resumable execution"
        )

    _resource_guard(_data_root(root))
    torch.set_num_threads(4)
    _seed_everything(STAGE3_SEED)
    device_receipt = select_device("auto", allow_cpu_fallback=True)
    device = torch.device(device_receipt.selected)
    model = Phase10RModel(variant, seed=STAGE3_SEED)
    model.load_state_dict(parent_state, strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=STAGE3_LEARNING_RATE, weight_decay=STAGE3_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    order = list(range(len(train_examples)))
    random.Random(STAGE3_SEED).shuffle(order)
    step = 0
    cursor = 0
    metrics: dict[str, Any] = {
        "loss_sum": 0.0,
        "active_examples": 0,
        "optimizer_steps": 0,
        "stream_rows_consumed": 0,
    }
    if progress_path.exists():
        try:
            progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError) as error:
            raise Phase10RTeacherBindingError("stage-3 progress cannot be decoded") from error
        required = {
            "schema",
            "variant_id",
            "manifest_sha256",
            "seed",
            "stage_id",
            "step",
            "stream_index",
            "completed",
            "parent_checkpoint_sha256",
            "teacher_binding_identity_sha256",
            "order",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "python_rng_state",
            "numpy_rng_state",
            "torch_rng_state",
            "metrics",
        }
        if (
            not isinstance(progress, Mapping)
            or set(progress) != required
            or progress["schema"] != CHECKPOINT_SCHEMA
            or progress["variant_id"] != variant
            or progress["manifest_sha256"] != manifest_sha256
            or progress["seed"] != STAGE3_SEED
            or progress["stage_id"] != STAGE3_ID
            or progress["parent_checkpoint_sha256"] != parent_checkpoint_sha256
            or progress["teacher_binding_identity_sha256"] != identity_sha256
            or not isinstance(progress["order"], list)
            or sorted(progress["order"]) != list(range(STAGE3_TRAIN_ROWS))
            or not isinstance(progress["step"], int)
            or not isinstance(progress["stream_index"], int)
            or not 0 <= progress["stream_index"] <= STAGE3_TRAIN_ROWS
            or not isinstance(progress["metrics"], Mapping)
        ):
            raise Phase10RTeacherBindingError("stage-3 progress identity or sampler is invalid")
        model.load_state_dict(progress["model_state"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(progress["scheduler_state"])
        random.setstate(progress["python_rng_state"])
        np.random.set_state(progress["numpy_rng_state"])
        torch.set_rng_state(progress["torch_rng_state"])
        step = progress["step"]
        cursor = progress["stream_index"]
        order = [int(value) for value in progress["order"]]
        metrics = dict(progress["metrics"])
        if progress["completed"] and cursor != STAGE3_TRAIN_ROWS:
            raise Phase10RTeacherBindingError(
                "completed stage-3 progress does not cover the train rows"
            )

    started = time.monotonic()
    model.train()
    try:
        while cursor < STAGE3_TRAIN_ROWS:
            next_cursor = min(STAGE3_TRAIN_ROWS, cursor + STAGE3_BATCH_SIZE)
            batch = [train_examples[index] for index in order[cursor:next_cursor]]
            optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = loss_for_examples(model, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise Phase10RTeacherBindingError("stage-3 gradient norm is NaN or Inf")
            optimizer.step()
            scheduler.step()
            cursor = next_cursor
            step += 1
            metrics["optimizer_steps"] = step
            metrics["active_examples"] = int(metrics.get("active_examples", 0)) + len(batch)
            metrics["loss_sum"] = float(metrics.get("loss_sum", 0.0)) + float(loss.detach().cpu())
            metrics["stream_rows_consumed"] = cursor
            metrics["last_batch_loss"] = float(loss.detach().cpu())
            metrics["last_gradient_norm"] = float(gradient_norm.detach().cpu())
            for name, value in batch_metrics.items():
                if not name.endswith("_weight"):
                    metrics[f"last_{name}"] = float(value)
            _resource_guard(_data_root(root))
            _atomic_torch_save(
                progress_path,
                _stage3_progress_payload(
                    model,
                    optimizer,
                    scheduler,
                    manifest_sha256=manifest_sha256,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    identity_sha256=identity_sha256,
                    step=step,
                    cursor=cursor,
                    order=order,
                    metrics=metrics,
                ),
            )
    except Exception:
        _atomic_torch_save(
            progress_path,
            _stage3_progress_payload(
                model,
                optimizer,
                scheduler,
                manifest_sha256=manifest_sha256,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                identity_sha256=identity_sha256,
                step=step,
                cursor=cursor,
                order=order,
                metrics=metrics,
            ),
        )
        raise

    child_state = _cpu_state(model)
    if not _state_changed(parent_state, child_state):
        raise Phase10RTeacherBindingError("stage-3 produced unchanged parent tensors")
    final_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "variant_id": variant,
        "manifest_sha256": manifest_sha256,
        "seed": STAGE3_SEED,
        "stage_id": STAGE3_ID,
        "step": step,
        "stream_index": STAGE3_TRAIN_ROWS,
        "completed": True,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "teacher_binding_identity_sha256": identity_sha256,
        "metrics": dict(metrics),
        "model_state": child_state,
    }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = io.BytesIO()
    torch.save(final_payload, encoded)
    _publish_immutable(final_path, encoded.getvalue())
    _load_stage3_final(
        final_path,
        variant=variant,
        manifest_sha256=manifest_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        identity_sha256=identity_sha256,
    )
    return {
        "status": "passed",
        "stage_id": STAGE3_ID,
        "resumed": progress_path.exists(),
        "epochs": 1,
        "steps": step,
        "stream_rows_consumed": STAGE3_TRAIN_ROWS,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": metrics,
        "device": device_receipt.as_dict(),
        "checkpoint": final_path,
        "checkpoint_sha256": _sha256_file(final_path),
        "parent_checkpoint": checkpoint_path,
    }


def _raw_score(model: Phase10RModel, example: Phase10RExample) -> float:
    output = model.forward_example(example)
    values = output["values"].detach().to(device="cpu", dtype=torch.float32)
    if model.variant_id == VARIANT_PAIR:
        probabilities = torch.softmax(values[:3], dim=0)
        raw = math.log((float(probabilities[2]) + 1.0e-6) / (float(probabilities[0]) + 1.0e-6))
    else:
        transformed = float(values[3])
        raw = math.copysign(
            math.expm1(min(abs(transformed), 1.0) * math.log1p(3_000.0)), transformed
        )
    if not math.isfinite(raw):
        raise Phase10RTeacherBindingError("stage-4 raw calibration prediction is non-finite")
    return raw


def _float32(value: float, context: str) -> float:
    try:
        result = float(np.asarray(value, dtype=np.float32))
    except (OverflowError, TypeError, ValueError) as error:
        raise Phase10RTeacherBindingError(f"{context} cannot be represented as float32") from error
    if not math.isfinite(result):
        raise Phase10RTeacherBindingError(f"{context} is non-finite")
    return result


def _fit_positive_affine(
    model: Phase10RModel, validation_examples: Sequence[Phase10RExample], variant: str
) -> tuple[float, float, dict[str, Any]]:
    raw_values: list[float] = []
    target_values: list[float] = []
    for example in validation_examples:
        binding = _teacher_payload(
            {
                "raw_targets": example.raw_targets,
            }
        )
        score_kind, score_value = _score_mapping(binding["score"], "stage-4 teacher score")
        if score_kind != "cp":
            continue
        raw_values.append(_raw_score(model, example))
        target_values.append(float(score_value))
    if len(raw_values) != STAGE4_CP_ROWS:
        raise Phase10RTeacherBindingError(f"stage-4 fit row count differs: {len(raw_values)}")
    x = np.asarray(raw_values, dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    variance = float(np.mean((x - x_mean) ** 2))
    if not math.isfinite(variance) or variance <= 1.0e-18:
        raise Phase10RTeacherBindingError("stage-4 calibration input has no finite variance")
    covariance = float(np.mean((x - x_mean) * (y - y_mean)))
    slope = covariance / variance
    bias = y_mean - slope * x_mean
    scale = _float32(slope, "stage-4 calibration scale")
    offset = _float32(bias, "stage-4 calibration bias")
    if scale <= 0.0:
        raise Phase10RTeacherBindingError(f"stage-4 positive monotonic scale is invalid: {scale}")
    predictions = scale * x + offset
    residual = float(np.mean((predictions - y) ** 2))
    if not math.isfinite(residual):
        raise Phase10RTeacherBindingError("stage-4 calibration residual is non-finite")
    return (
        scale,
        offset,
        {
            "fit_rows": len(raw_values),
            "raw_input": "wdl_log_odds"
            if variant == VARIANT_PAIR
            else "direct_inverse_transformed_score_cp",
            "target": "apery_cp_current_side_to_move",
            "mean_squared_residual": residual,
        },
    )


def _stage4(
    root: Path,
    control: Mapping[str, Any],
    variant: str,
    manifest_sha256: str,
    stage3_path: Path,
    validation_examples: Sequence[Phase10RExample],
) -> dict[str, Any]:
    child_dir = stage3_path.parent
    output = child_dir / STAGE4_NAME
    stage3_sha256 = _sha256_file(stage3_path)
    identity_sha256 = control["teacher_binding_identity"]["identity_sha256"]
    stage3_payload = _load_stage3_final(
        stage3_path,
        variant=variant,
        manifest_sha256=manifest_sha256,
        parent_checkpoint_sha256=next(
            item["stage2_checkpoint"]["sha256"]
            for item in control["pretraining"]["variants"]
            if item["variant_id"] == variant
        ),
        identity_sha256=identity_sha256,
    )
    model = Phase10RModel(variant, seed=STAGE3_SEED)
    model.load_state_dict(stage3_payload["model_state"], strict=True)
    model.eval()
    scale, bias, fit_details = _fit_positive_affine(model, validation_examples, variant)
    receipt = {
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "status": "passed",
        "variant_id": variant,
        "input_manifest_sha256": manifest_sha256,
        "stage3_checkpoint_sha256": stage3_sha256,
        "teacher_binding_identity_sha256": identity_sha256,
        "method": "positive_monotonic_affine",
        "fit_rows": STAGE4_CP_ROWS,
        "calibration_scale": scale,
        "calibration_bias": bias,
        "new_teacher_calls": 0,
    }
    payload = _json_bytes(receipt)
    if output.exists() or output.is_symlink():
        existing = _load_json_object(output, "stage-4 calibration")
        if existing != receipt or output.read_bytes() != payload:
            raise Phase10RTeacherBindingError(
                "stage-4 calibration already exists with different bytes"
            )
    else:
        _publish_immutable(output, payload)
    loaded = _load_json_object(output, "stage-4 calibration")
    if loaded != receipt:
        raise Phase10RTeacherBindingError("stage-4 calibration changed after publish")
    return {
        "status": "passed",
        "stage_id": STAGE4_ID,
        "calibration": output,
        "calibration_sha256": _sha256_file(output),
        "calibration_scale": scale,
        "calibration_bias": bias,
        "fit": fit_details,
    }


def _assert_exported_weights(stage3_payload: Mapping[str, Any], parsed: Any) -> None:
    state = stage3_payload["model_state"]
    if set(state) != set(parsed.tensors):
        raise Phase10RTeacherBindingError("teacher-bound artifact tensor set differs from stage 3")
    for name, view in parsed.tensors.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise Phase10RTeacherBindingError("stage-3 tensor is not a tensor")
        encoded = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype(np.dtype("<f4"), copy=False)
            .tobytes()
        )
        if encoded != view.data.tobytes():
            raise Phase10RTeacherBindingError(f"teacher-bound tensor differs from stage 3: {name}")


def _artifact(
    root: Path,
    control: Mapping[str, Any],
    variant: str,
    manifest_sha256: str,
    stage3_path: Path,
    calibration_scale: float,
    calibration_bias: float,
    git_commit: str,
) -> dict[str, Any]:
    output = stage3_path.parent / ARTIFACT_NAME
    stage3_payload = _load_stage3_final(
        stage3_path,
        variant=variant,
        manifest_sha256=manifest_sha256,
        parent_checkpoint_sha256=next(
            item["stage2_checkpoint"]["sha256"]
            for item in control["pretraining"]["variants"]
            if item["variant_id"] == variant
        ),
        identity_sha256=control["teacher_binding_identity"]["identity_sha256"],
    )
    model = Phase10RModel(variant, seed=STAGE3_SEED)
    model.load_state_dict(stage3_payload["model_state"], strict=True)
    model.eval()
    reference = f"phase10r-1m-{variant}-teacher-bound-v1"
    try:
        encoded = serialize_osaval02(
            model.export_tensors(),
            variant_id=variant,
            quantization="float32",
            dataset_manifest_sha256=manifest_sha256,
            training_run_reference=reference,
            git_commit=git_commit,
            calibration_scale=calibration_scale,
            calibration_bias=calibration_bias,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise Phase10RTeacherBindingError("teacher-bound OSAVAL02 export failed") from error
    _ensure_payload(output, encoded, "teacher-bound OSAVAL02 artifact")
    try:
        inspected = parse_osaval02(output.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        raise Phase10RTeacherBindingError(
            "teacher-bound OSAVAL02 artifact failed strict parsing"
        ) from error
    if (
        inspected.variant_id != variant
        or inspected.quantization != "float32"
        or inspected.dataset_manifest_sha256 != manifest_sha256
        or inspected.training_run_reference != reference
        or inspected.calibration_scale != calibration_scale
        or inspected.calibration_bias != calibration_bias
    ):
        raise Phase10RTeacherBindingError("teacher-bound OSAVAL02 metadata is not exact")
    _assert_exported_weights(stage3_payload, inspected)
    return {
        "status": "passed",
        "path": output,
        "sha256": _sha256_file(output),
        "bytes": output.stat().st_size,
        "variant_id": variant,
        "quantization": "float32",
        "dataset_manifest_sha256": manifest_sha256,
        "training_run_reference": reference,
        "calibration_scale": calibration_scale,
        "calibration_bias": calibration_bias,
    }


def _read_process_line(process: subprocess.Popen[bytes], buffer: bytearray, deadline: float) -> str:
    while b"\n" not in buffer:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise Phase10RTeacherBindingError("USI acceptance query timed out")
        if process.stdout is None:
            raise Phase10RTeacherBindingError("USI acceptance query has no stdout")
        ready, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
        if not ready:
            raise Phase10RTeacherBindingError("USI acceptance query timed out")
        chunk = os.read(process.stdout.fileno(), 65_536)
        if not chunk:
            raise Phase10RTeacherBindingError("USI acceptance engine ended early")
        buffer.extend(chunk)
        if len(buffer) > 2 * 1024 * 1024:
            raise Phase10RTeacherBindingError("USI acceptance output exceeded its bound")
    raw, _, remainder = bytes(buffer).partition(b"\n")
    buffer[:] = remainder
    try:
        return raw.rstrip(b"\r").decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase10RTeacherBindingError("USI acceptance output is not UTF-8") from error


def _usi_query(
    root: Path,
    artifact: Path,
    sfen: str,
    *,
    nodes: int,
    hash_mib: int,
    max_depth: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    engine = root / "target/release/open-shogi-cli"
    if engine.is_symlink() or not engine.is_file():
        raise Phase10RTeacherBindingError("release engine is unavailable for acceptance gates")
    process = subprocess.Popen(
        [str(engine), "usi"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    buffer = bytearray()

    def send(command: str) -> None:
        if process.stdin is None:
            raise Phase10RTeacherBindingError("USI acceptance query has no stdin")
        process.stdin.write((command + "\n").encode("utf-8"))
        process.stdin.flush()

    def until(expected: str) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            line = _read_process_line(process, buffer, deadline)
            if line.startswith("info string error "):
                raise Phase10RTeacherBindingError(line)
            if line == expected:
                return

    try:
        send("usi")
        until("usiok")
        send(f"setoption name USI_Hash value {hash_mib}")
        send(f"setoption name MaxDepth value {max_depth}")
        send("setoption name ModelKind value neural-float")
        send("setoption name ModelSemantics value pure-value")
        send(f"setoption name ModelPath value {artifact}")
        send("isready")
        until("readyok")
        send(f"position sfen {sfen}")
        send(f"go nodes {nodes}")
        deadline = time.monotonic() + timeout_seconds
        last_info: str | None = None
        while True:
            line = _read_process_line(process, buffer, deadline)
            if line.startswith("info "):
                last_info = line
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) < 2 or parts[1] == "resign":
                    raise Phase10RTeacherBindingError("USI acceptance query returned resign")
                nps = None
                observed_nodes = None
                if last_info is not None:
                    tokens = last_info.split()
                    for key, target in (("nps", "nps"), ("nodes", "nodes")):
                        if key in tokens:
                            index = tokens.index(key)
                            if index + 1 < len(tokens):
                                try:
                                    parsed = int(tokens[index + 1])
                                except ValueError:
                                    continue
                                if target == "nps":
                                    nps = parsed
                                else:
                                    observed_nodes = parsed
                return {
                    "bestmove": parts[1],
                    "info": last_info,
                    "nps": nps,
                    "nodes": observed_nodes,
                }
    finally:
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b"quit\n")
                    process.stdin.flush()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=2)


def _tactical_gate(root: Path, parent_artifact: Path, child_artifact: Path) -> dict[str, Any]:
    suite = _load_json_object(root / TACTICAL_SUITE_PATH, "Phase 10R tactical suite")
    if _sha256_file(root / TACTICAL_SUITE_PATH) != TACTICAL_SUITE_SHA256:
        raise Phase10RTeacherBindingError("Phase 10R tactical suite hash changed")
    if set(suite) != {"schema", "cases"} or suite["schema"] != "open_shogi_tactical_suite/v1":
        raise Phase10RTeacherBindingError("Phase 10R tactical suite schema is invalid")
    cases = suite["cases"]
    if not isinstance(cases, list) or not cases:
        raise Phase10RTeacherBindingError("Phase 10R tactical suite is empty")
    rows: list[dict[str, Any]] = []
    regressions = 0
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != {"id", "sfen", "expectedMoves"}:
            raise Phase10RTeacherBindingError("Phase 10R tactical case is invalid")
        sfen = _validate_canonical_sfen(case["sfen"], "Phase 10R tactical SFEN")
        expected = case["expectedMoves"]
        if (
            not isinstance(expected, list)
            or not expected
            or any(not isinstance(move, str) for move in expected)
        ):
            raise Phase10RTeacherBindingError("Phase 10R tactical expected moves are invalid")
        parent = _usi_query(
            root,
            parent_artifact,
            sfen,
            nodes=TACTICAL_NODES,
            hash_mib=TACTICAL_HASH_MIB,
            max_depth=TACTICAL_MAX_DEPTH,
            timeout_seconds=TACTICAL_TIMEOUT_SECONDS,
        )
        child = _usi_query(
            root,
            child_artifact,
            sfen,
            nodes=TACTICAL_NODES,
            hash_mib=TACTICAL_HASH_MIB,
            max_depth=TACTICAL_MAX_DEPTH,
            timeout_seconds=TACTICAL_TIMEOUT_SECONDS,
        )
        parent_passed = parent["bestmove"] in expected
        child_passed = child["bestmove"] in expected
        regressions += int(parent_passed and not child_passed)
        rows.append(
            {
                "id": case["id"],
                "expected_moves": expected,
                "parent": parent,
                "child": child,
                "parent_passed": parent_passed,
                "child_passed": child_passed,
            }
        )
    return {
        "status": "passed" if regressions == 0 else "failed",
        "suite": TACTICAL_SUITE_PATH,
        "suite_sha256": TACTICAL_SUITE_SHA256,
        "nodes_per_case": TACTICAL_NODES,
        "cases": rows,
        "new_tactical_failures": regressions,
    }


def _throughput_floor(root: Path, variant: str, artifact: Path) -> dict[str, Any]:
    matrix = _load_yaml(root / "configs/phase10r/model-matrix.yaml")
    selected = next(
        item
        for item in matrix["variants"]
        if isinstance(item, Mapping) and item.get("variant_id") == variant
    )
    minimum = selected.get("minimum_search_context_evaluations_per_second")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise Phase10RTeacherBindingError(
            "variant throughput floor is not a frozen positive integer"
        )
    measurement = _usi_query(
        root,
        artifact,
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        nodes=THROUGHPUT_NODES,
        hash_mib=1024,
        max_depth=8,
        timeout_seconds=TACTICAL_TIMEOUT_SECONDS,
    )
    measured = measurement.get("nps")
    passed = isinstance(measured, int) and measured >= minimum
    return {
        "status": "passed" if passed else "failed",
        "minimum_search_context_evaluations_per_second": minimum,
        "measured_search_nodes_per_second": measured,
        "measurement": measurement,
        "passed": passed,
    }


def _source_held_out_gates(
    root: Path,
    control: Mapping[str, Any],
    variant: str,
    parent_checkpoint: Path,
    stage3_payload: Mapping[str, Any],
) -> dict[str, Any]:
    preparation_path = preparation_manifest_path(root, SCALE)
    preparation = validate_preparation(root, SCALE)
    held_out = _preparation_file(root, preparation_path, preparation, "base-source_held_out.jsonl")
    expected_rows = preparation["files"]["base-source_held_out.jsonl"]["rows"]
    parent_model = Phase10RModel(variant, seed=STAGE3_SEED)
    child_model = Phase10RModel(variant, seed=STAGE3_SEED)
    try:
        parent_payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        parent_model.load_state_dict(parent_payload["model_state"], strict=True)
        child_model.load_state_dict(stage3_payload["model_state"], strict=True)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        raise Phase10RTeacherBindingError("source-held-out model loading failed") from error
    data_root = _data_root(root)
    parent_metrics = _evaluate_file(
        held_out, expected_rows=expected_rows, model=parent_model, data_root=data_root
    )
    child_metrics = _evaluate_file(
        held_out, expected_rows=expected_rows, model=child_model, data_root=data_root
    )
    differences: dict[str, float] = {}
    higher_is_better = {"policy_top1"}
    for metric in ("policy_top1", "policy_nll", "wdl_brier", "wdl_nll"):
        parent = parent_metrics.get(metric)
        child = child_metrics.get(metric)
        if not isinstance(parent, (int, float)) or not isinstance(child, (int, float)):
            raise Phase10RTeacherBindingError(f"source-held-out metric is unavailable: {metric}")
        differences[metric] = max(
            0.0, float(parent - child) if metric in higher_is_better else float(child - parent)
        )
    source_regression = max(differences.values())
    parent_ece = parent_metrics.get("calibration_ece")
    child_ece = child_metrics.get("calibration_ece")
    if not isinstance(parent_ece, (int, float)) or not isinstance(child_ece, (int, float)):
        raise Phase10RTeacherBindingError("source-held-out ECE is unavailable")
    ece_regression = float(child_ece - parent_ece)
    threshold = float(control["acceptance"]["require_source_held_out_absolute_regression_at_most"])
    ece_threshold = float(
        control["acceptance"]["require_calibration_ece_absolute_regression_at_most"]
    )
    if source_regression > threshold:
        raise Phase10RTeacherBindingError(
            f"source-held-out regression gate failed: {source_regression} > {threshold}"
        )
    if ece_regression > ece_threshold:
        raise Phase10RTeacherBindingError(
            f"ECE regression gate failed: {ece_regression} > {ece_threshold}"
        )
    return {
        "status": "passed",
        "parent": parent_metrics,
        "child": child_metrics,
        "regressions": differences,
        "source_held_out_regression_maximum": source_regression,
        "calibration_ece_regression": ece_regression,
        "threshold": threshold,
        "ece_threshold": ece_threshold,
    }


def _parity(root: Path, variant: str, artifact: Path) -> tuple[dict[str, Any], Path]:
    details = _candidate_parity(root, artifact, artifact)
    incremental = _incremental_parity(root)
    receipt = {
        "schema": PARITY_RECEIPT_SCHEMA,
        "status": "passed",
        "variant_id": variant,
        "artifact_sha256": _sha256_file(artifact),
        "python_native_wasm": "passed",
        "incremental_full_recompute_unmake": "passed",
    }
    output = artifact.parent / PARITY_NAME
    payload = _json_bytes(receipt)
    if output.exists() or output.is_symlink():
        existing = _load_json_object(output, "candidate parity receipt")
        if existing != receipt or output.read_bytes() != payload:
            raise Phase10RTeacherBindingError(
                "candidate parity receipt already exists with different bytes"
            )
    else:
        _publish_immutable(output, payload)
    return {"receipt": output, "details": details, "incremental": incremental}, output


def _lineage(
    root: Path,
    control: Mapping[str, Any],
    variant: str,
    manifest_path: Path,
    manifest_sha256: str,
    stage3: dict[str, Any],
    stage4: dict[str, Any],
    artifact: dict[str, Any],
    parity_path: Path,
    acceptance: Mapping[str, Any],
    git_commit: str,
) -> tuple[Path, Mapping[str, Any]]:
    output = lineage_path(root, SCALE, variant)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise Phase10RTeacherBindingError("candidate lineage is not a regular file")
        try:
            return output, validate_candidate_lineage(root, output)
        except ValueError as error:
            raise Phase10RTeacherBindingError("present candidate lineage is invalid") from error

    pretraining = control["pretraining"]
    selected = next(item for item in pretraining["variants"] if item["variant_id"] == variant)
    prep_path = _regular_path(
        root, pretraining["preparation_manifest"]["path"], "parent preparation"
    )
    parent = {
        "preparation_manifest": {
            "path": prep_path.relative_to(root).as_posix(),
            "sha256": pretraining["preparation_manifest"]["file_sha256"],
            "bytes": prep_path.stat().st_size,
        },
        "training_receipt": _artifact_ref(
            root,
            _regular_path(root, selected["training_receipt"]["path"], "parent training receipt"),
            "parent training receipt",
            expected_sha256=selected["training_receipt"]["sha256"],
        ),
        "stage2_checkpoint": _artifact_ref(
            root,
            _regular_path(root, selected["stage2_checkpoint"]["path"], "parent stage2 checkpoint"),
            "parent stage2 checkpoint",
            expected_sha256=selected["stage2_checkpoint"]["sha256"],
        ),
        "stage2_artifact": _artifact_ref(
            root,
            _regular_path(root, selected["stage2_artifact"]["path"], "parent stage2 artifact"),
            "parent stage2 artifact",
            expected_sha256=selected["stage2_artifact"]["sha256"],
        ),
    }
    binding_identity = control["teacher_binding_identity"]
    control_path = root / CONTROL_PATH
    labels = binding_identity["labels"]
    binding = {
        "identity_sha256": binding_identity["identity_sha256"],
        "control": _artifact_ref(
            root, control_path, "lineage control", expected_sha256=CONTROL_SHA256
        ),
        "label_manifest": _artifact_ref(
            root,
            _regular_path(root, labels["manifest"]["path"], "lineage label manifest"),
            "lineage label manifest",
            expected_sha256=labels["manifest"]["sha256"],
        ),
        "labels": _artifact_ref(
            root,
            _regular_path(root, labels["rows"]["path"], "lineage labels"),
            "lineage labels",
            expected_sha256=labels["rows"]["sha256"],
        ),
        "benchmark": _artifact_ref(
            root,
            _regular_path(root, labels["benchmark"]["path"], "lineage benchmark"),
            "lineage benchmark",
            expected_sha256=labels["benchmark"]["sha256"],
        ),
        "calibration_input_manifest": _artifact_ref(
            root, manifest_path, "lineage calibration input", expected_sha256=manifest_sha256
        ),
    }
    stage3_ref = _artifact_ref(root, stage3["checkpoint"], "lineage stage 3 checkpoint")
    stage4_ref = _artifact_ref(root, stage4["calibration"], "lineage stage 4 calibration")
    artifact_ref = _artifact_ref(root, artifact["path"], "lineage teacher-bound artifact")
    parity_ref = _artifact_ref(root, parity_path, "lineage parity receipt")
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "status": "completed",
        "binding_version": BINDING_VERSION,
        "scale": SCALE,
        "variant_id": variant,
        "created_at_utc": _utc_now(),
        "git_commit": git_commit,
        "parent": parent,
        "teacher_binding": binding,
        "stages": [
            {
                "order": 3,
                "stage_id": STAGE3_ID,
                "status": "passed",
                "input_manifest": binding["calibration_input_manifest"],
                "output": stage3_ref,
            },
            {
                "order": 4,
                "stage_id": STAGE4_ID,
                "status": "passed",
                "input_manifest": binding["calibration_input_manifest"],
                "output": stage4_ref,
            },
        ],
        "candidate": {
            "artifact": artifact_ref,
            "osaval02": {
                "variant_id": variant,
                "dataset_manifest_sha256": manifest_sha256,
                "training_run_reference": artifact["training_run_reference"],
            },
            "parity_receipt": parity_ref,
        },
        "acceptance": dict(acceptance),
    }
    _publish_immutable(output, _json_bytes(lineage))
    try:
        validated = validate_candidate_lineage(root, output)
    except ValueError as error:
        raise Phase10RTeacherBindingError(
            f"published candidate lineage failed validation: {error}"
        ) from error
    return output, validated


def calibrate_teacher(
    root: Path, scale: str, variant: str, *, resume: bool, git_commit: str
) -> dict[str, Any]:
    """Run exactly stages 3 and 4 for one frozen variant."""

    if scale != SCALE or variant not in VARIANTS:
        raise Phase10RTeacherBindingError("teacher calibration target is outside the frozen matrix")
    root = root.resolve(strict=True)
    control = load_teacher_binding_control(root)
    manifest_path, _manifest, manifest_sha256, train_examples, validation_examples = (
        _load_calibration_examples(root, control, variant)
    )
    existing_lineage = lineage_path(root, scale, variant)
    if existing_lineage.exists() or existing_lineage.is_symlink():
        try:
            validated = validate_candidate_lineage(root, existing_lineage)
        except ValueError as error:
            raise Phase10RTeacherBindingError("present candidate lineage is invalid") from error
        return {
            "schema": LINEAGE_SCHEMA,
            "status": "passed",
            "variant": variant,
            "lineage": existing_lineage,
            "lineage_sha256": _sha256_file(existing_lineage),
            "reused_completed_lineage": True,
            "validated": validated,
            "new_teacher_calls": 0,
        }

    stage3 = _run_stage3(
        root,
        control,
        variant,
        train_examples,
        manifest_sha256,
        resume=resume,
    )
    stage4 = _stage4(
        root,
        control,
        variant,
        manifest_sha256,
        stage3["checkpoint"],
        validation_examples,
    )
    artifact = _artifact(
        root,
        control,
        variant,
        manifest_sha256,
        stage3["checkpoint"],
        stage4["calibration_scale"],
        stage4["calibration_bias"],
        git_commit,
    )
    (
        parent_checkpoint,
        parent_artifact,
        _parent_state,
        parent_checkpoint_sha256,
        parent_artifact_sha256,
    ) = _load_parent(root, control, variant)
    stage3_payload = _load_stage3_final(
        stage3["checkpoint"],
        variant=variant,
        manifest_sha256=manifest_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        identity_sha256=control["teacher_binding_identity"]["identity_sha256"],
    )
    parity, parity_path = _parity(root, variant, artifact["path"])
    source_gates = _source_held_out_gates(root, control, variant, parent_checkpoint, stage3_payload)
    tactical = _tactical_gate(root, parent_artifact, artifact["path"])
    if tactical["new_tactical_failures"] != 0:
        raise Phase10RTeacherBindingError("zero-new-tactical-failures gate failed")
    throughput = _throughput_floor(root, variant, artifact["path"])
    if throughput["passed"] is not True:
        raise Phase10RTeacherBindingError("variant throughput floor gate failed")
    parent_checkpoint_after, parent_artifact_after, _state_after, _, _ = _load_parent(
        root, control, variant
    )
    if (
        _sha256_file(parent_checkpoint_after) != parent_checkpoint_sha256
        or _sha256_file(parent_artifact_after) != parent_artifact_sha256
    ):
        raise Phase10RTeacherBindingError("immutable parent changed during teacher binding")
    acceptance = {
        "status": "passed",
        "parent_immutable": True,
        "new_teacher_calls": 0,
        "forbidden_split_rows": 0,
        "calibration_scale": stage4["calibration_scale"],
        "calibration_bias": stage4["calibration_bias"],
        "python_native_wasm_parity": "passed",
        "incremental_parity": "passed",
        "source_held_out_regression_maximum": source_gates["source_held_out_regression_maximum"],
        "calibration_ece_regression": source_gates["calibration_ece_regression"],
        "new_tactical_failures": tactical["new_tactical_failures"],
        "throughput_floor_passed": True,
    }
    lineage, validated = _lineage(
        root,
        control,
        variant,
        manifest_path,
        manifest_sha256,
        stage3,
        stage4,
        artifact,
        parity_path,
        acceptance,
        git_commit,
    )
    return {
        "schema": LINEAGE_SCHEMA,
        "status": "passed",
        "variant": variant,
        "calibration_input_manifest": manifest_path,
        "calibration_input_manifest_sha256": manifest_sha256,
        "parent": {
            "checkpoint": parent_checkpoint,
            "checkpoint_sha256": parent_checkpoint_sha256,
            "artifact": parent_artifact,
            "artifact_sha256": parent_artifact_sha256,
        },
        "stage3": stage3,
        "stage4": stage4,
        "artifact": artifact,
        "teacher_binding_identity_sha256": control["teacher_binding_identity"]["identity_sha256"],
        "parity": parity,
        "acceptance": acceptance,
        "lineage": lineage,
        "lineage_sha256": _sha256_file(lineage),
        "lineage_validation": "passed" if validated.get("status") == "completed" else "failed",
        "source_held_out": source_gates,
        "tactical": tactical,
        "throughput": throughput,
        "new_teacher_calls": 0,
    }


__all__ = [
    "CALIBRATION_INPUT_SCHEMA",
    "Phase10RTeacherBindingError",
    "calibrate_teacher",
    "prepare_teacher_binding",
]
