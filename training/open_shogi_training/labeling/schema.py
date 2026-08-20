"""Strict public schema and loader for Phase 4 teacher labels."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    iter_jsonl_descriptor_records,
    iter_jsonl_records,
)

LABEL_SCHEMA: Final = "phase4_teacher_label/v1"
QUARANTINE_SCHEMA: Final = "phase4_teacher_quarantine/v1"
PARSER_VERSION: Final = "usi-info/v1"
SCORE_POV: Final = "side_to_move"
MAX_LABEL_LINE_BYTES: Final = 256 * 1024
MAX_LABEL_ARTIFACT_BYTES: Final = 4 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_USI_MOVE_RE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")
_SPLITS = frozenset({"train", "validation", "test"})
_STAGES = frozenset({"opening", "middlegame", "endgame"})
_SIDES = frozenset({"black", "white"})
_OUTCOMES = frozenset({"black_win", "white_win", "draw", "unknown"})
_LABEL_KEYS = frozenset(
    {
        "schema",
        "position_id",
        "canonical_sfen",
        "canonical_state_sha256",
        "side_to_move",
        "score",
        "score_pov",
        "bestmove",
        "candidates",
        "depth",
        "seldepth",
        "nodes",
        "elapsed_ms",
        "teacher",
        "dataset_manifest_sha256",
        "config_sha256",
        "parser_version",
        "created_at",
        "split",
        "game_id",
        "position_index",
        "stage",
        "source_id",
        "outcome",
    }
)
_CANDIDATE_KEYS = frozenset({"multipv", "score", "pv", "depth", "seldepth", "nodes"})
_SCORE_KEYS = frozenset({"kind", "value"})
_TEACHER_KEYS = frozenset(
    {
        "name",
        "version",
        "reported_name",
        "reported_author",
        "binary_sha256",
        "binary_size",
        "eval_files",
        "options",
    }
)
_EVAL_KEYS = frozenset({"path", "sha256", "size"})


class LabelSchemaError(ValueError):
    """Raised when a teacher label violates the closed public schema."""


def canonical_state(sfen: str) -> str:
    """Return the canonical leakage identity, omitting only SFEN move number."""

    if not isinstance(sfen, str) or any(character in sfen for character in "\r\n\x00"):
        raise LabelSchemaError("canonical_sfen must be one safe line")
    fields = sfen.split(" ")
    if len(fields) != 4 or any(not field for field in fields):
        raise LabelSchemaError("canonical_sfen must contain exactly four single-spaced fields")
    if fields[1] not in {"b", "w"}:
        raise LabelSchemaError("canonical_sfen has an invalid side-to-move field")
    try:
        move_number = int(fields[3])
    except ValueError as error:
        raise LabelSchemaError("canonical_sfen move number must be an integer") from error
    if move_number < 1:
        raise LabelSchemaError("canonical_sfen move number must be positive")
    return " ".join(fields[:3])


def canonical_state_sha256(sfen: str) -> str:
    return hashlib.sha256(canonical_state(sfen).encode("utf-8")).hexdigest()


def position_id(game_id: str, position_index: int) -> str:
    _require_sha256(game_id, "game_id")
    _require_int(position_index, "position_index", minimum=0)
    digest = hashlib.sha256()
    digest.update(b"phase3_position/v1\x00")
    digest.update(bytes.fromhex(game_id))
    digest.update(position_index.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def validate_label_record(value: object) -> dict[str, Any]:
    """Validate one exact label row and return it without coercion."""

    record = _closed_mapping(value, "label", _LABEL_KEYS)
    if record["schema"] != LABEL_SCHEMA:
        raise LabelSchemaError(f"label.schema must be {LABEL_SCHEMA!r}")
    _require_sha256(record["position_id"], "position_id")
    state = canonical_state(_require_text(record["canonical_sfen"], "canonical_sfen", 2_048))
    state_sha256 = _require_sha256(record["canonical_state_sha256"], "canonical_state_sha256")
    if state_sha256 != hashlib.sha256(state.encode("utf-8")).hexdigest():
        raise LabelSchemaError("canonical_state_sha256 does not match canonical_sfen")
    side = _require_choice(record["side_to_move"], "side_to_move", _SIDES)
    expected_side = "black" if state.split(" ")[1] == "b" else "white"
    if side != expected_side:
        raise LabelSchemaError("side_to_move disagrees with canonical_sfen")
    root_score = _validate_score(record["score"], "score")
    if record["score_pov"] != SCORE_POV:
        raise LabelSchemaError(f"score_pov must be {SCORE_POV!r}")
    bestmove = _require_usi_move(record["bestmove"], "bestmove")

    candidates_raw = _require_sequence(record["candidates"], "candidates", maximum=500)
    if not candidates_raw:
        raise LabelSchemaError("candidates must not be empty")
    candidates: list[dict[str, Any]] = []
    first_moves: set[str] = set()
    for index, candidate_raw in enumerate(candidates_raw, start=1):
        candidate = _closed_mapping(candidate_raw, f"candidates[{index - 1}]", _CANDIDATE_KEYS)
        if _require_int(candidate["multipv"], "candidate.multipv", minimum=1) != index:
            raise LabelSchemaError("candidate multipv ranks must be contiguous from one")
        _validate_score(candidate["score"], f"candidates[{index - 1}].score")
        pv = _require_sequence(candidate["pv"], f"candidates[{index - 1}].pv", maximum=1_024)
        if not pv:
            raise LabelSchemaError("candidate pv must not be empty")
        validated_pv = [
            _require_usi_move(move, f"candidates[{index - 1}].pv[{move_index}]")
            for move_index, move in enumerate(pv)
        ]
        if validated_pv[0] in first_moves:
            raise LabelSchemaError("candidate first moves must be distinct")
        first_moves.add(validated_pv[0])
        _require_int(candidate["depth"], "candidate.depth", minimum=0)
        _require_int(candidate["seldepth"], "candidate.seldepth", minimum=0)
        _require_int(candidate["nodes"], "candidate.nodes", minimum=0)
        candidates.append(candidate)

    first = candidates[0]
    if first["score"] != root_score:
        raise LabelSchemaError("root score must equal the first candidate score")
    if first["pv"][0] != bestmove:
        raise LabelSchemaError("bestmove must equal the first candidate PV move")
    for field in ("depth", "seldepth", "nodes"):
        _require_int(record[field], field, minimum=0)
        if record[field] != first[field]:
            raise LabelSchemaError(f"root {field} must equal the first candidate {field}")
    _require_int(record["elapsed_ms"], "elapsed_ms", minimum=0)
    teacher = _validate_teacher(record["teacher"])
    configured_multipv = teacher["options"].get("MultiPV")
    if (
        isinstance(configured_multipv, bool)
        or not isinstance(configured_multipv, int)
        or not 1 <= len(candidates) <= configured_multipv
    ):
        raise LabelSchemaError("candidate count exceeds the recorded MultiPV option")
    _require_sha256(record["dataset_manifest_sha256"], "dataset_manifest_sha256")
    _require_sha256(record["config_sha256"], "config_sha256")
    if record["parser_version"] != PARSER_VERSION:
        raise LabelSchemaError(f"parser_version must be {PARSER_VERSION!r}")
    _require_utc_timestamp(record["created_at"], "created_at")
    _require_choice(record["split"], "split", _SPLITS)
    game_id = _require_sha256(record["game_id"], "game_id")
    index = _require_int(record["position_index"], "position_index", minimum=0)
    if record["position_id"] != position_id(game_id, index):
        raise LabelSchemaError("position_id disagrees with game_id and position_index")
    _require_choice(record["stage"], "stage", _STAGES)
    _require_text(record["source_id"], "source_id", 128)
    _require_choice(record["outcome"], "outcome", _OUTCOMES)
    return record


def iter_teacher_labels(
    path: Path,
    *,
    expected_dataset_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    max_bytes: int = MAX_LABEL_ARTIFACT_BYTES,
    max_records: int = 10_000,
) -> Iterator[dict[str, Any]]:
    """Stream strict labels for model training without score coercion."""

    if expected_dataset_manifest_sha256 is not None:
        _require_sha256(expected_dataset_manifest_sha256, "expected_dataset_manifest_sha256")
    if expected_config_sha256 is not None:
        _require_sha256(expected_config_sha256, "expected_config_sha256")
    yield from _iter_validated_teacher_labels(
        iter_jsonl_records(
            path,
            max_bytes=max_bytes,
            max_line_bytes=MAX_LABEL_LINE_BYTES,
            max_records=max_records,
        ),
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_config_sha256=expected_config_sha256,
    )


def iter_teacher_labels_descriptor(
    descriptor: int,
    *,
    display_path: Path,
    expected_dataset_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    max_bytes: int = MAX_LABEL_ARTIFACT_BYTES,
    max_records: int = 10_000,
) -> Iterator[dict[str, Any]]:
    """Validate teacher labels from an already identity-verified descriptor."""

    if expected_dataset_manifest_sha256 is not None:
        _require_sha256(expected_dataset_manifest_sha256, "expected_dataset_manifest_sha256")
    if expected_config_sha256 is not None:
        _require_sha256(expected_config_sha256, "expected_config_sha256")
    yield from _iter_validated_teacher_labels(
        iter_jsonl_descriptor_records(
            descriptor,
            display_path=display_path,
            max_bytes=max_bytes,
            max_line_bytes=MAX_LABEL_LINE_BYTES,
            max_records=max_records,
        ),
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_config_sha256=expected_config_sha256,
    )


def _iter_validated_teacher_labels(
    records: Iterator[dict[str, Any]],
    *,
    expected_dataset_manifest_sha256: str | None,
    expected_config_sha256: str | None,
) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for line_number, raw in enumerate(records, start=1):
        try:
            label = validate_label_record(raw)
        except LabelSchemaError as error:
            raise LabelSchemaError(f"invalid label line {line_number}: {error}") from error
        identity = label["position_id"]
        if identity in seen:
            raise LabelSchemaError(f"duplicate position_id on label line {line_number}")
        seen.add(identity)
        if (
            expected_dataset_manifest_sha256 is not None
            and label["dataset_manifest_sha256"] != expected_dataset_manifest_sha256
        ):
            raise LabelSchemaError(f"dataset manifest mismatch on label line {line_number}")
        if expected_config_sha256 is not None and label["config_sha256"] != expected_config_sha256:
            raise LabelSchemaError(f"config mismatch on label line {line_number}")
        yield label


def validate_score(value: object) -> dict[str, Any]:
    """Public validator used by downstream model loaders."""

    return _validate_score(value, "score")


def _validate_score(value: object, name: str) -> dict[str, Any]:
    score = _closed_mapping(value, name, _SCORE_KEYS)
    kind = _require_choice(score["kind"], f"{name}.kind", frozenset({"cp", "mate"}))
    score_value = _require_int(score["value"], f"{name}.value", minimum=-(2**31), maximum=2**31 - 1)
    return {"kind": kind, "value": score_value}


def _validate_teacher(value: object) -> dict[str, Any]:
    teacher = _closed_mapping(value, "teacher", _TEACHER_KEYS)
    _require_text(teacher["name"], "teacher.name", 128)
    _require_text(teacher["version"], "teacher.version", 64)
    _require_text(teacher["reported_name"], "teacher.reported_name", 256)
    reported_author = teacher["reported_author"]
    if reported_author is not None:
        _require_text(reported_author, "teacher.reported_author", 256)
    _require_sha256(teacher["binary_sha256"], "teacher.binary_sha256")
    _require_int(teacher["binary_size"], "teacher.binary_size", minimum=1)
    eval_files = _require_sequence(teacher["eval_files"], "teacher.eval_files", maximum=64)
    if not eval_files:
        raise LabelSchemaError("teacher.eval_files must not be empty")
    seen_paths: set[str] = set()
    for index, raw in enumerate(eval_files):
        item = _closed_mapping(raw, f"teacher.eval_files[{index}]", _EVAL_KEYS)
        path = _require_text(item["path"], f"teacher.eval_files[{index}].path", 1_024)
        if path in seen_paths:
            raise LabelSchemaError("teacher.eval_files paths must be unique")
        seen_paths.add(path)
        _require_sha256(item["sha256"], f"teacher.eval_files[{index}].sha256")
        _require_int(item["size"], f"teacher.eval_files[{index}].size", minimum=1)
    options = teacher["options"]
    if not isinstance(options, dict) or len(options) > 256:
        raise LabelSchemaError("teacher.options must be a bounded mapping")
    for key, option in options.items():
        _require_text(key, "teacher.options key", 128)
        if not isinstance(option, (str, int, bool)):
            raise LabelSchemaError("teacher.options values must be scalar")
    return teacher


def _closed_mapping(value: object, name: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelSchemaError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        raise LabelSchemaError(
            f"{name} keys mismatch: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _require_sequence(value: object, name: str, *, maximum: int) -> Sequence[Any]:
    if not isinstance(value, list):
        raise LabelSchemaError(f"{name} must be an array")
    if len(value) > maximum:
        raise LabelSchemaError(f"{name} exceeds {maximum} entries")
    return value


def _require_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise LabelSchemaError(f"{name} must be a non-empty string up to {maximum} characters")
    if any(character in value for character in "\r\n\x00"):
        raise LabelSchemaError(f"{name} contains a line delimiter")
    return value


def _require_choice(value: object, name: str, choices: frozenset[str]) -> str:
    text = _require_text(value, name, 128)
    if text not in choices:
        raise LabelSchemaError(f"{name} must be one of {sorted(choices)}")
    return text


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int = 2**63 - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelSchemaError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise LabelSchemaError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LabelSchemaError(f"{name} must be a lowercase SHA-256")
    return value


def _require_usi_move(value: object, name: str) -> str:
    if not isinstance(value, str) or _USI_MOVE_RE.fullmatch(value) is None:
        raise LabelSchemaError(f"{name} must be a normal USI move")
    return value


def _require_utc_timestamp(value: object, name: str) -> str:
    text = _require_text(value, name, 64)
    if not text.endswith("Z"):
        raise LabelSchemaError(f"{name} must be UTC and end in Z")
    try:
        parsed = datetime.fromisoformat(f"{text[:-1]}+00:00")
    except ValueError as error:
        raise LabelSchemaError(f"{name} must be ISO-8601") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise LabelSchemaError(f"{name} must be UTC")
    return text
