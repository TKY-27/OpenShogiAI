"""Streaming Phase 10R population and split-leakage scanner.

The scanner is intentionally source-preserving.  It consumes already replay-validated
canonical streams, assigns game/history-level frozen splits before positions are inserted,
and keeps the working identity table in SQLite rather than materialising the population in
Python memory.  Missing replay, history, or transposition evidence is a blocking result.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from open_shogi_training.data.phase10r_registry import load_phase10r_registry
from open_shogi_training.labeling.schema import canonical_state
from open_shogi_training.phase10r_model import parse_sfen

SCAN_SCHEMA: Final = "open_shogiai_phase10r_leakage_scan/v1"
POSITION_SCHEMA: Final = "open_shogiai_phase10r_scanned_position/v1"
MINIMUM_FREE_BYTES: Final = 150 * 1024**3
SPLITS: Final = frozenset(
    {"train", "validation", "source_held_out", "public_test", "internal_test", "final_holdout"}
)
PROTECTED_SPLITS: Final = frozenset(
    {"validation", "source_held_out", "public_test", "internal_test", "final_holdout"}
)
LEAKAGE_CLASSES: Final = (
    "same_record_cross_split",
    "same_game_cross_split",
    "canonical_position_cross_split",
    "protected_history_cross_split",
    "different_plies_one_game_train_eval",
    "prohibited_transposition_overlap",
    "aobazero_via_gct",
    "floodgate_via_gct_or_direct",
    "distilled_vs_nodchip",
    "arena_start_overlap",
    "public_test_entering_training_or_selection",
    "final_holdout_entering_forbidden_path",
    "source_held_out_entering_training",
    "parser_normalization_version_collision",
)
_META_SCHEMA = "schema"
_META_INPUT_SHA = "input_manifest_sha256"
_META_COMPLETED = "completed_artifacts"
_IDENTITY_COLUMNS = (
    "record_id",
    "game_id",
    "canonical_position_id",
    "history_id",
)
_PROTECTED_SQL = "'validation','source_held_out','public_test','internal_test','final_holdout'"


class Phase10RScanError(RuntimeError):
    """Raised when a leakage scan cannot prove its input or identity contract."""


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise Phase10RScanError(f"scan JSON is not canonical: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Phase10RScanError(f"scan input is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(*parts: object) -> str:
    digest = hashlib.sha256()
    digest.update(b"open-shogiai/phase10r-scan/v1\0")
    for part in parts:
        if not isinstance(part, str) or "\x00" in part:
            raise Phase10RScanError("scan identity component is not a safe string")
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise Phase10RScanError(f"{field} is not a safe non-empty string")
    return value


def _sha256_or_none(value: object, field: str) -> str | None:
    if value is None:
        return None
    result = _safe_string(value, field)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise Phase10RScanError(f"{field} is not a lowercase SHA-256")
    return result


def _protected_role(split: str) -> str:
    return {
        "train": "train",
        "validation": "validation",
        "source_held_out": "source_held_out",
        "public_test": "public_test",
        "internal_test": "internal_test",
        "final_holdout": "final_holdout",
    }[split]


def _canonicalize_sfen(value: object) -> tuple[str, str]:
    raw = _safe_string(value, "canonical_sfen")
    parse_input = raw if len(raw.split(" ")) == 4 else f"{raw} 1"
    try:
        canonical = canonical_state(parse_input)
        parsed = parse_sfen(parse_input)
    except (TypeError, ValueError) as error:
        raise Phase10RScanError(f"invalid replayed SFEN: {error}") from error
    side = "black" if parsed.side_to_move == 0 else "white"
    return canonical, side


def _source_held_out_split(source_id: str, game_id: str) -> str:
    """Apply the frozen external source-held-out/validation/train buckets."""

    bucket = int(_stable_digest("split", source_id, game_id)[:8], 16) % 10_000
    if bucket < 1_000:
        return "source_held_out"
    if bucket < 2_000:
        return "validation"
    return "train"


def _export_game_splits(export_path: Path, source_id: str) -> dict[str, str]:
    """Choose exact deterministic game-group fractions before position extraction."""

    games: dict[str, str] = {}
    with export_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase10RScanError(f"invalid CSA export JSON at line {line_number}") from error
            if not isinstance(row, Mapping) or row.get("schema") != "phase3_csa_export/v1":
                raise Phase10RScanError(f"unknown CSA export schema at line {line_number}")
            if row.get("status") != "ok":
                raise Phase10RScanError(
                    f"CSA replay rejected {row.get('inputFile')!r}: {row.get('reason', 'unknown')}"
                )
            input_file = _safe_string(row.get("inputFile"), "inputFile")
            normalized_csa = row.get("normalizedCsa")
            moves = row.get("usiMoves")
            initial_sfen = _safe_string(row.get("initialSfen"), "initialSfen")
            if (
                not isinstance(normalized_csa, str)
                or not normalized_csa
                or "\x00" in normalized_csa
                or not isinstance(moves, list)
            ):
                raise Phase10RScanError(f"CSA export identity is invalid at line {line_number}")
            record_sha256 = _sha256_bytes(normalized_csa.encode("utf-8"))
            game_id = _stable_digest("game", source_id, input_file, record_sha256)
            identity = _stable_digest(
                "history", source_id, input_file, initial_sfen, *[str(move) for move in moves]
            )
            previous = games.setdefault(game_id, identity)
            if previous != identity:
                raise Phase10RScanError(f"CSA game identity changed for {input_file}")
    ordered = sorted(games)
    if not ordered:
        return {}
    source_held_out_count = max(1, math.ceil(len(ordered) * 0.10))
    validation_count = max(1, math.ceil(len(ordered) * 0.10))
    result: dict[str, str] = {}
    for index, game_id in enumerate(
        sorted(ordered, key=lambda value: _stable_digest("split", source_id, value))
    ):
        if index < source_held_out_count:
            result[game_id] = "source_held_out"
        elif index < source_held_out_count + validation_count:
            result[game_id] = "validation"
        else:
            result[game_id] = "train"
    return result


def _normalise_position(
    row: Mapping[str, Any],
    *,
    artifact_id: str,
    source_id: str,
    artifact_sha256: str | None,
    source_revision: str,
    parser_version: str,
    normalization_version: str,
    input_sha256: str | None,
) -> dict[str, Any]:
    if row.get("schema") not in {None, POSITION_SCHEMA}:
        raise Phase10RScanError(f"unknown scanned-position schema for {artifact_id}")
    if _safe_string(row.get("artifact_id", artifact_id), "artifact_id") != artifact_id:
        raise Phase10RScanError("row artifact identity disagrees with its input manifest")
    if _safe_string(row.get("source_id", source_id), "source_id") != source_id:
        raise Phase10RScanError("row source identity disagrees with its input manifest")
    canonical, derived_side = _canonicalize_sfen(row.get("canonical_sfen", row.get("sfen")))
    side = _safe_string(row.get("side_to_move", derived_side), "side_to_move")
    if side != derived_side:
        raise Phase10RScanError("row side-to-move disagrees with canonical SFEN")
    split = _safe_string(row.get("split"), "split")
    if split not in SPLITS:
        raise Phase10RScanError(f"unknown frozen split: {split}")
    record_id = _safe_string(row.get("record_id"), "record_id")
    game_id = _safe_string(row.get("game_id"), "game_id")
    history_id = _safe_string(row.get("history_id", game_id), "history_id")
    position_index = row.get("position_index", row.get("positionIndex", 0))
    if (
        isinstance(position_index, bool)
        or not isinstance(position_index, int)
        or position_index < 0
    ):
        raise Phase10RScanError("position index is invalid")
    record_sha256 = _sha256_or_none(row.get("record_sha256"), "record_sha256")
    if record_sha256 is None:
        record_sha256 = _sha256_or_none(row.get("raw_sha256"), "raw_sha256")
    transposition_key = row.get("transposition_key")
    if transposition_key is not None:
        transposition_key = _safe_string(transposition_key, "transposition_key")
    opening_group = row.get("opening_group")
    arena_group = row.get("arena_group")
    if opening_group is not None:
        opening_group = _safe_string(opening_group, "opening_group")
    if arena_group is not None:
        arena_group = _safe_string(arena_group, "arena_group")
    return {
        "schema": POSITION_SCHEMA,
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha256,
        "source_id": source_id,
        "source_revision": _safe_string(source_revision, "source_revision"),
        "record_id": record_id,
        "record_sha256": record_sha256,
        "game_id": game_id,
        "position_index": position_index,
        "position_id": _stable_digest("position", source_id, game_id, str(position_index)),
        "canonical_position_id": _sha256_bytes(canonical.encode("utf-8")),
        "canonical_sfen": canonical,
        "history_id": history_id,
        "transposition_key": transposition_key,
        "side_to_move": side,
        "split": split,
        "protected_role": _safe_string(
            row.get("protected_role", _protected_role(split)), "protected_role"
        ),
        "parser_version": _safe_string(parser_version, "parser_version"),
        "normalization_version": _safe_string(normalization_version, "normalization_version"),
        "input_sha256": input_sha256,
        "opening_group": opening_group,
        "arena_group": arena_group,
    }


def iter_aobazero_positions(
    positions_path: Path,
    *,
    artifact_id: str,
    artifact_sha256: str | None,
    source_revision: str,
    input_sha256: str | None,
) -> Iterable[dict[str, Any]]:
    """Stream the existing source-preserving AobaZero Phase 3 position artifact."""

    if positions_path.is_symlink() or not positions_path.is_file():
        raise Phase10RScanError(f"AobaZero positions input is not a regular file: {positions_path}")
    seen_game_splits: dict[str, str] = {}
    with gzip.open(positions_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase10RScanError(f"invalid AobaZero JSON at line {line_number}") from error
            if not isinstance(row, Mapping) or row.get("schema") != "phase3_position/v1":
                raise Phase10RScanError(f"unexpected AobaZero row at line {line_number}")
            game_id = _safe_string(row.get("gameId"), "gameId")
            raw_split = _safe_string(row.get("split"), "split")
            split = {"train": "train", "validation": "validation", "test": "final_holdout"}.get(
                raw_split
            )
            if split is None:
                raise Phase10RScanError(f"unknown AobaZero legacy split: {raw_split}")
            previous = seen_game_splits.setdefault(game_id, split)
            if previous != split:
                raise Phase10RScanError(f"AobaZero game crosses frozen splits: {game_id}")
            position_index = row.get("positionIndex")
            if isinstance(position_index, bool) or not isinstance(position_index, int):
                raise Phase10RScanError("AobaZero positionIndex is invalid")
            yield _normalise_position(
                {
                    "artifact_id": artifact_id,
                    "source_id": "aobazero",
                    "record_id": row.get("rawSha256"),
                    "record_sha256": row.get("rawSha256"),
                    "game_id": game_id,
                    "position_index": position_index,
                    "canonical_sfen": row.get("sfen"),
                    "side_to_move": row.get("sideToMove"),
                    "history_id": game_id,
                    "split": split,
                },
                artifact_id=artifact_id,
                source_id="aobazero",
                artifact_sha256=artifact_sha256,
                source_revision=source_revision,
                parser_version="phase3_position/v1",
                normalization_version="phase3_dataset_manifest/v1",
                input_sha256=input_sha256,
            )


def iter_export_positions(
    export_path: Path,
    *,
    artifact_id: str,
    source_id: str,
    artifact_sha256: str | None,
    source_revision: str,
    input_sha256: str | None,
) -> Iterable[dict[str, Any]]:
    """Stream all positions from strict Rust CSA replay JSONL."""

    if export_path.is_symlink() or not export_path.is_file():
        raise Phase10RScanError(f"CSA export input is not a regular file: {export_path}")
    split_by_game = _export_game_splits(export_path, source_id)
    with export_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase10RScanError(f"invalid CSA export JSON at line {line_number}") from error
            if not isinstance(row, Mapping) or row.get("schema") != "phase3_csa_export/v1":
                raise Phase10RScanError(f"unknown CSA export schema at line {line_number}")
            if row.get("status") != "ok":
                raise Phase10RScanError(
                    f"CSA replay rejected {row.get('inputFile')!r}: {row.get('reason', 'unknown')}"
                )
            input_file = _safe_string(row.get("inputFile"), "inputFile")
            sfens = row.get("positionSfens")
            moves = row.get("usiMoves")
            if (
                not isinstance(sfens, list)
                or not sfens
                or not isinstance(moves, list)
                or len(sfens) != len(moves) + 1
            ):
                raise Phase10RScanError(f"CSA export sequence is invalid at line {line_number}")
            normalized_csa = row.get("normalizedCsa")
            if (
                not isinstance(normalized_csa, str)
                or not normalized_csa
                or "\x00" in normalized_csa
                or len(normalized_csa.encode("utf-8")) > 1_048_576
            ):
                raise Phase10RScanError("normalizedCsa is not a bounded non-empty UTF-8 string")
            record_sha256 = _sha256_bytes(normalized_csa.encode("utf-8"))
            game_id = _stable_digest("game", source_id, input_file, record_sha256)
            history_id = _stable_digest(
                "history",
                source_id,
                input_file,
                str(row.get("initialSfen")),
                *[str(m) for m in moves],
            )
            split = split_by_game[game_id]
            for position_index, sfen in enumerate(sfens):
                yield _normalise_position(
                    {
                        "artifact_id": artifact_id,
                        "source_id": source_id,
                        "record_id": record_sha256,
                        "record_sha256": record_sha256,
                        "game_id": game_id,
                        "position_index": position_index,
                        "canonical_sfen": sfen,
                        "history_id": history_id,
                        "split": split,
                    },
                    artifact_id=artifact_id,
                    source_id=source_id,
                    artifact_sha256=artifact_sha256,
                    source_revision=source_revision,
                    parser_version="phase3_csa_export/v1",
                    normalization_version="rust_replay_export/v1",
                    input_sha256=input_sha256,
                )


def _create_database(path: Path, input_manifest_sha256: str) -> sqlite3.Connection:
    if path.is_symlink():
        raise Phase10RScanError(f"scan database cannot be a symlink: {path}")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id TEXT NOT NULL,
            artifact_sha256 TEXT,
            source_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            record_id TEXT NOT NULL,
            record_sha256 TEXT,
            game_id TEXT NOT NULL,
            position_index INTEGER NOT NULL,
            position_id TEXT NOT NULL,
            canonical_position_id TEXT NOT NULL,
            canonical_sfen TEXT NOT NULL,
            history_id TEXT NOT NULL,
            transposition_key TEXT,
            side_to_move TEXT NOT NULL,
            split TEXT NOT NULL,
            protected_role TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            normalization_version TEXT NOT NULL,
            input_sha256 TEXT,
            opening_group TEXT,
            arena_group TEXT
        );
        CREATE INDEX IF NOT EXISTS positions_record_idx ON positions(record_id, split);
        CREATE INDEX IF NOT EXISTS positions_game_idx ON positions(game_id, split);
        CREATE INDEX IF NOT EXISTS positions_position_idx
            ON positions(canonical_position_id, split);
        CREATE INDEX IF NOT EXISTS positions_history_idx ON positions(history_id, split);
        CREATE INDEX IF NOT EXISTS positions_transposition_idx
            ON positions(transposition_key, split);
        """
    )
    existing_schema = connection.execute(
        "SELECT value FROM meta WHERE key = ?", (_META_SCHEMA,)
    ).fetchone()
    if existing_schema is not None and existing_schema[0] != SCAN_SCHEMA:
        raise Phase10RScanError("scan database schema identity mismatch")
    existing_input = connection.execute(
        "SELECT value FROM meta WHERE key = ?", (_META_INPUT_SHA,)
    ).fetchone()
    if existing_input is not None and existing_input[0] != input_manifest_sha256:
        raise Phase10RScanError("scan database input manifest identity mismatch")
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (_META_SCHEMA, SCAN_SCHEMA)
    )
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (_META_INPUT_SHA, input_manifest_sha256),
    )
    if (
        connection.execute("SELECT 1 FROM meta WHERE key = ?", (_META_COMPLETED,)).fetchone()
        is None
    ):
        connection.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (_META_COMPLETED, "[]"))
    connection.commit()
    return connection


def _completed_artifacts(connection: sqlite3.Connection) -> set[str]:
    value = connection.execute(
        "SELECT value FROM meta WHERE key = ?", (_META_COMPLETED,)
    ).fetchone()
    if value is None:
        return set()
    try:
        parsed = json.loads(value[0])
    except json.JSONDecodeError as error:
        raise Phase10RScanError("scan checkpoint has invalid completed-artifact JSON") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise Phase10RScanError("scan checkpoint has invalid completed-artifact list")
    return set(parsed)


def _set_completed(connection: sqlite3.Connection, completed: set[str]) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (_META_COMPLETED, json.dumps(sorted(completed), separators=(",", ":"))),
    )


def _stable_publish(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise Phase10RScanError(f"refusing to overwrite non-regular scan output: {path}")
        if path.stat().st_size != len(payload) or _sha256_file(path) != _sha256_bytes(payload):
            raise Phase10RScanError(f"scan output already exists with different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != _sha256_bytes(payload):
            raise Phase10RScanError(f"concurrent scan output differs: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _replace_json(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise Phase10RScanError(f"refusing to replace symlinked scan output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cross_split_count(connection: sqlite3.Connection, column: str) -> int:
    if column not in _IDENTITY_COLUMNS and column != "transposition_key":
        raise Phase10RScanError(f"unsupported leakage identity column: {column}")
    query = (
        f"SELECT COUNT(*) FROM (SELECT {column} FROM positions "
        f"WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(DISTINCT split) > 1)"
    )
    return int(connection.execute(query).fetchone()[0])


def _protected_cross_split_count(connection: sqlite3.Connection, column: str) -> int:
    query = (
        f"SELECT COUNT(*) FROM (SELECT {column} FROM positions WHERE {column} IS NOT NULL "
        f"AND split IN ({_PROTECTED_SQL}) GROUP BY {column} "
        f"HAVING COUNT(DISTINCT split) > 1)"
    )
    return int(connection.execute(query).fetchone()[0])


def _source_held_out_training_count(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT source_id, game_id FROM positions
                GROUP BY source_id, game_id
                HAVING SUM(split = 'source_held_out') > 0 AND SUM(split = 'train') > 0
            )
            """
        ).fetchone()[0]
    )


def _version_collision_count(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT canonical_position_id FROM positions
                GROUP BY canonical_position_id
                HAVING COUNT(DISTINCT parser_version) > 1
                    OR COUNT(DISTINCT normalization_version) > 1
            )
            """
        ).fetchone()[0]
    )


def _source_split_counts(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        "SELECT source_id, split, COUNT(*) FROM positions "
        "GROUP BY source_id, split ORDER BY source_id, split"
    )
    result: dict[str, dict[str, int]] = {}
    for source_id, split, count in rows:
        result.setdefault(str(source_id), {})[str(split)] = int(count)
    return result


def _source_group_counts(connection: sqlite3.Connection) -> dict[str, dict[str, int | float]]:
    rows = connection.execute(
        """
        SELECT source_id,
               COUNT(DISTINCT game_id),
               COUNT(DISTINCT CASE WHEN split = 'source_held_out' THEN game_id END),
               COUNT(DISTINCT CASE WHEN split = 'validation' THEN game_id END)
        FROM positions GROUP BY source_id ORDER BY source_id
        """
    )
    result: dict[str, dict[str, int | float]] = {}
    for source_id, games, held_out, validation in rows:
        result[str(source_id)] = {
            "game_groups": int(games),
            "source_held_out_game_groups": int(held_out),
            "validation_game_groups": int(validation),
            "source_held_out_fraction": (int(held_out) / int(games)) if games else 0.0,
        }
    return result


def _leakage_report(connection: sqlite3.Connection, sources: set[str]) -> dict[str, Any]:
    classes: dict[str, dict[str, Any]] = {}
    for name, column in (
        ("same_record_cross_split", "record_id"),
        ("same_game_cross_split", "game_id"),
        ("canonical_position_cross_split", "canonical_position_id"),
        ("protected_history_cross_split", "history_id"),
        ("different_plies_one_game_train_eval", "game_id"),
    ):
        count = _cross_split_count(connection, column)
        if name == "different_plies_one_game_train_eval":
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT game_id FROM positions
                        GROUP BY game_id
                        HAVING SUM(split = 'train') > 0
                           AND SUM(split IN (
                               'validation','source_held_out','public_test',
                               'internal_test','final_holdout'
                           )) > 0
                    )
                    """
                ).fetchone()[0]
            )
        classes[name] = {
            "status": "passed" if count == 0 else "failed",
            "count": count,
            "identity": column,
        }
    transposition_count = _cross_split_count(connection, "transposition_key")
    has_transposition = connection.execute(
        "SELECT 1 FROM positions WHERE transposition_key IS NOT NULL LIMIT 1"
    ).fetchone()
    classes["prohibited_transposition_overlap"] = {
        "status": "passed" if has_transposition and transposition_count == 0 else "unavailable",
        "count": transposition_count,
        "identity": "transposition_key",
        "reason": "no exact transposition namespace was supplied by the replay stream"
        if not has_transposition
        else None,
    }
    for name in (
        "aobazero_via_gct",
        "floodgate_via_gct_or_direct",
        "distilled_vs_nodchip",
        "arena_start_overlap",
    ):
        classes[name] = {
            "status": "not_applicable",
            "count": 0,
            "reason": "no admitted artifact lineage for this class",
        }
    public_count = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT canonical_position_id FROM positions
                GROUP BY canonical_position_id
                HAVING SUM(split = 'public_test') > 0
                   AND SUM(split IN ('train','validation','source_held_out')) > 0
            )
            """
        ).fetchone()[0]
    )
    final_count = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT canonical_position_id FROM positions
                GROUP BY canonical_position_id
                HAVING SUM(split = 'final_holdout') > 0
                   AND SUM(split IN ('train','validation','source_held_out')) > 0
            )
            """
        ).fetchone()[0]
    )
    source_count = _source_held_out_training_count(connection)
    version_count = _version_collision_count(connection)
    classes["public_test_entering_training_or_selection"] = {
        "status": "passed" if public_count == 0 else "failed",
        "count": public_count,
        "identity": "canonical_position_id",
    }
    classes["final_holdout_entering_forbidden_path"] = {
        "status": "passed" if final_count == 0 else "failed",
        "count": final_count,
        "identity": "canonical_position_id",
    }
    classes["source_held_out_entering_training"] = {
        "status": "passed" if source_count == 0 else "failed",
        "count": source_count,
        "identity": "source_id+game_id",
    }
    classes["parser_normalization_version_collision"] = {
        "status": "passed" if version_count == 0 else "failed",
        "count": version_count,
        "identity": "canonical_position_id",
    }
    unclassified = sorted(set(classes) - set(LEAKAGE_CLASSES))
    observed_sources = {
        str(row[0]) for row in connection.execute("SELECT DISTINCT source_id FROM positions")
    }
    if unclassified or not observed_sources <= sources:
        raise Phase10RScanError("scanner leakage class or source accounting is incomplete")
    statuses = {entry["status"] for entry in classes.values()}
    passed = statuses <= {"passed", "not_applicable"}
    return {
        "schema": "open_shogiai_phase10r_split_leakage_report/v1",
        "passed": passed,
        "classes": classes,
        "source_split_counts": _source_split_counts(connection),
        "source_group_counts": _source_group_counts(connection),
        "allowed_same_split_duplicate_groups": {
            column: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM (SELECT {column} FROM positions "
                    f"WHERE {column} IS NOT NULL "
                    f"GROUP BY {column} HAVING COUNT(*) > 1 AND COUNT(DISTINCT split) = 1)"
                ).fetchone()[0]
            )
            for column in ("record_id", "game_id", "canonical_position_id", "history_id")
        },
    }


def _write_manifest(connection: sqlite3.Connection, path: Path) -> str:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise Phase10RScanError(f"leakage manifest is not a regular file: {path}")
        return _sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, mode="wb", delete=False
    ) as handle:
        temporary = Path(handle.name)
        rows = connection.execute(
            """
            SELECT row_id, artifact_id, artifact_sha256, source_id, source_revision,
                   record_id, record_sha256, game_id, position_index, position_id,
                   canonical_position_id, canonical_sfen, history_id, transposition_key,
                   side_to_move, split, protected_role, parser_version,
                   normalization_version, input_sha256, opening_group, arena_group
            FROM positions ORDER BY row_id
            """
        )
        for values in rows:
            (
                row_id,
                artifact_id,
                artifact_sha256,
                source_id,
                source_revision,
                record_id,
                record_sha256,
                game_id,
                position_index,
                position_id,
                canonical_position_id,
                canonical_sfen,
                history_id,
                transposition_key,
                side_to_move,
                split,
                protected_role,
                parser_version,
                normalization_version,
                input_sha256,
                opening_group,
                arena_group,
            ) = values
            encoded = _json_bytes(
                {
                    "schema": POSITION_SCHEMA,
                    "row_index": row_id,
                    "artifact_id": artifact_id,
                    "artifact_sha256": artifact_sha256,
                    "source_id": source_id,
                    "source_revision": source_revision,
                    "record_id": record_id,
                    "record_sha256": record_sha256,
                    "game_id": game_id,
                    "position_index": position_index,
                    "position_id": position_id,
                    "canonical_position_id": canonical_position_id,
                    "canonical_sfen": canonical_sfen,
                    "history_id": history_id,
                    "transposition_key": transposition_key,
                    "side_to_move": side_to_move,
                    "split": split,
                    "protected_role": protected_role,
                    "parser_version": parser_version,
                    "normalization_version": normalization_version,
                    "input_sha256": input_sha256,
                    "opening_group": opening_group,
                    "arena_group": arena_group,
                }
            )
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest.hexdigest():
            raise Phase10RScanError(f"concurrent leakage manifest differs: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def scan_records(
    records_by_artifact: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    output_dir: Path,
    approved_artifact_ids: Sequence[str],
    source_metadata: Mapping[str, Mapping[str, Any]],
    input_identities: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_free_bytes: int = 0,
) -> dict[str, Any]:
    """Scan canonical streams with resumable artifact checkpoints and durable receipts."""

    approved = tuple(sorted(set(approved_artifact_ids)))
    if len(approved) != len(tuple(approved_artifact_ids)):
        raise Phase10RScanError("approved artifact list contains duplicates")
    unknown = sorted(set(records_by_artifact) - set(approved))
    if unknown:
        raise Phase10RScanError(f"scan received unapproved artifact streams: {unknown}")
    if set(source_metadata) - set(approved):
        raise Phase10RScanError("source metadata contains an unapproved artifact")
    input_values = input_identities or {}
    input_manifest = {
        "schema": "open_shogiai_phase10r_scan_input_manifest/v1",
        "approved_artifact_ids": list(approved),
        "inputs": {
            artifact_id: dict(input_values.get(artifact_id, {})) for artifact_id in approved
        },
    }
    input_manifest_bytes = _json_bytes(input_manifest)
    input_manifest_sha256 = _sha256_bytes(input_manifest_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise Phase10RScanError("scan output directory cannot be a symlink")
    free = shutil.disk_usage(output_dir).free
    if free < minimum_free_bytes:
        raise Phase10RScanError(f"free disk {free} is below the scan floor {minimum_free_bytes}")
    _stable_publish(output_dir / "phase10r-scan-input-manifest.json", input_manifest_bytes)
    database_path = output_dir / "phase10r-scan.sqlite3"
    connection = _create_database(database_path, input_manifest_sha256)
    completed = _completed_artifacts(connection)
    failures: list[dict[str, Any]] = []
    for artifact_id in approved:
        if artifact_id in completed:
            continue
        identity = input_values.get(artifact_id, {})
        if identity.get("status") not in {None, "ready"}:
            failures.append(
                {"artifact_id": artifact_id, "reason": identity.get("reason", "input unavailable")}
            )
            continue
        stream = records_by_artifact.get(artifact_id)
        if stream is None:
            failures.append(
                {"artifact_id": artifact_id, "reason": "approved input stream is missing"}
            )
            continue
        metadata = source_metadata.get(artifact_id)
        if metadata is None:
            failures.append({"artifact_id": artifact_id, "reason": "source metadata is missing"})
            continue
        count = 0
        try:
            connection.execute("BEGIN")
            for raw_row in stream:
                if not isinstance(raw_row, Mapping):
                    raise Phase10RScanError(f"{artifact_id} emitted a non-object row")
                row = _normalise_position(
                    raw_row,
                    artifact_id=artifact_id,
                    source_id=_safe_string(metadata.get("source_id"), "source_id"),
                    artifact_sha256=_sha256_or_none(
                        metadata.get("artifact_sha256"), "artifact_sha256"
                    ),
                    source_revision=_safe_string(
                        metadata.get("source_revision"), "source_revision"
                    ),
                    parser_version=_safe_string(metadata.get("parser_version"), "parser_version"),
                    normalization_version=_safe_string(
                        metadata.get("normalization_version"), "normalization_version"
                    ),
                    input_sha256=_sha256_or_none(metadata.get("input_sha256"), "input_sha256"),
                )
                connection.execute(
                    """
                    INSERT INTO positions(
                        artifact_id, artifact_sha256, source_id, source_revision,
                        record_id, record_sha256, game_id, position_index, position_id,
                        canonical_position_id, canonical_sfen, history_id, transposition_key,
                        side_to_move, split, protected_role, parser_version,
                        normalization_version, input_sha256, opening_group, arena_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(
                        row[field]
                        for field in (
                            "artifact_id",
                            "artifact_sha256",
                            "source_id",
                            "source_revision",
                            "record_id",
                            "record_sha256",
                            "game_id",
                            "position_index",
                            "position_id",
                            "canonical_position_id",
                            "canonical_sfen",
                            "history_id",
                            "transposition_key",
                            "side_to_move",
                            "split",
                            "protected_role",
                            "parser_version",
                            "normalization_version",
                            "input_sha256",
                            "opening_group",
                            "arena_group",
                        )
                    ),
                )
                count += 1
            if count == 0:
                raise Phase10RScanError(f"approved artifact {artifact_id} emitted no positions")
            completed.add(artifact_id)
            _set_completed(connection, completed)
            connection.commit()
        except Exception as error:
            connection.rollback()
            failures.append({"artifact_id": artifact_id, "reason": str(error)})
            continue
        _write_checkpoint(
            output_dir / "phase10r-scan-checkpoint.json",
            {
                "schema": "open_shogiai_phase10r_scan_checkpoint/v1",
                "input_manifest_sha256": input_manifest_sha256,
                "completed_artifact_ids": sorted(completed),
                "last_artifact_id": artifact_id,
                "last_artifact_position_count": count,
            },
        )
    leakage = _leakage_report(
        connection, {str(metadata.get("source_id")) for metadata in source_metadata.values()}
    )
    source_counts = leakage["source_split_counts"]
    sources_with_rows = set(source_counts)
    for source_id in sorted(sources_with_rows):
        group = leakage["source_group_counts"][source_id]
        if source_id in {"aobazero"}:
            continue
        if group["source_held_out_game_groups"] == 0 or group["source_held_out_fraction"] < 0.10:
            failures.append(
                {"source_id": source_id, "reason": "frozen 10% source-held-out group is incomplete"}
            )
    missing_artifacts = sorted(set(approved) - completed)
    if missing_artifacts:
        failures.extend(
            {"artifact_id": artifact_id, "reason": "artifact was not completely scanned"}
            for artifact_id in missing_artifacts
            if not any(failure.get("artifact_id") == artifact_id for failure in failures)
        )
    manifest_path = output_dir / "phase10r-leakage-manifest.jsonl"
    partial_manifest_sha256 = None
    if missing_artifacts:
        partial_manifest_sha256 = _write_manifest(
            connection, output_dir / "phase10r-partial-leakage-manifest.jsonl"
        )
        manifest_sha256 = None
    else:
        manifest_sha256 = _write_manifest(connection, manifest_path)
    completion = {
        "schema": "open_shogiai_phase10r_scan_completion/v1",
        "status": "passed" if not failures and leakage["passed"] else "blocked",
        "input_manifest_sha256": input_manifest_sha256,
        "approved_artifact_count": len(approved),
        "completed_artifact_count": len(completed),
        "position_count": int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]),
        "failures": failures,
        "transposition_proof": leakage["classes"]["prohibited_transposition_overlap"],
        "manifest_sha256": manifest_sha256,
        "partial_manifest_sha256": partial_manifest_sha256,
        "database": database_path.name,
    }
    _replace_json(output_dir / "phase10r-source-split-counts.json", _json_bytes(source_counts))
    _replace_json(output_dir / "phase10r-prohibited-overlaps.json", _json_bytes(leakage["classes"]))
    _replace_json(
        output_dir / "phase10r-allowed-overlaps.json",
        _json_bytes(leakage["allowed_same_split_duplicate_groups"]),
    )
    _replace_json(output_dir / "phase10r-completion-proof.json", _json_bytes(completion))
    connection.close()
    return {
        "schema": SCAN_SCHEMA,
        "status": completion["status"],
        "input_manifest_sha256": input_manifest_sha256,
        "manifest_sha256": manifest_sha256,
        "position_count": completion["position_count"],
        "completed_artifact_count": len(completed),
        "approved_artifact_count": len(approved),
        "failures": failures,
        "leakage": leakage,
        "output_dir": output_dir.as_posix(),
    }


def build_registry_scan_inputs(
    root: Path, data_root: Path
) -> tuple[
    dict[str, Iterable[Mapping[str, Any]]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    """Discover only approved, replay-validated inputs; never substitute a different source."""

    registry = load_phase10r_registry(root / "configs/phase10r/source-registry.yaml")
    approved = registry.approved_artifacts()
    streams: dict[str, Iterable[Mapping[str, Any]]] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    aoba_positions = root / (
        "local/campaign-inputs/data-processed/phase3/"
        "aobazero-no-noise-pd-sample100/positions-00000.jsonl.gz"
    )
    aoba_manifest = aoba_positions.parent / "manifest.json"
    for artifact in approved:
        if artifact.source_id == "aobazero":
            if not aoba_positions.is_file() or not aoba_manifest.is_file():
                identities[artifact.artifact_id] = {
                    "status": "missing",
                    "reason": "AobaZero Phase 3 manifest/positions missing",
                }
                continue
            input_sha = _sha256_file(aoba_positions)
            manifest_sha = _sha256_file(aoba_manifest)
            streams[artifact.artifact_id] = iter_aobazero_positions(
                aoba_positions,
                artifact_id=artifact.artifact_id,
                artifact_sha256=manifest_sha,
                source_revision=artifact.source_revision,
                input_sha256=input_sha,
            )
            metadata[artifact.artifact_id] = {
                "source_id": "aobazero",
                "artifact_sha256": manifest_sha,
                "source_revision": artifact.source_revision,
                "parser_version": "phase3_position/v1",
                "normalization_version": "phase3_dataset_manifest/v1",
                "input_sha256": input_sha,
            }
            identities[artifact.artifact_id] = {
                "status": "ready",
                "path": aoba_positions.relative_to(root).as_posix(),
                "input_sha256": input_sha,
                "dataset_manifest_sha256": manifest_sha,
                "source_artifact_sha256": manifest_sha,
                "source_revision": artifact.source_revision,
            }
            continue
        candidate_paths = [
            data_root / "validation" / f"{artifact.artifact_id}-rust.jsonl",
            data_root / "validation" / f"{artifact.artifact_id}-rust-final.jsonl",
        ]
        candidate = next(
            (path for path in candidate_paths if path.is_file() and not path.is_symlink()), None
        )
        if candidate is None:
            identities[artifact.artifact_id] = {
                "status": "missing",
                "reason": "approved artifact lacks a complete replay-validated JSONL input",
            }
            continue
        raw_candidates = sorted(
            path
            for path in (data_root / "raw" / artifact.artifact_id).glob("*")
            if path.is_file() and not path.is_symlink()
        )
        if len(raw_candidates) != 1:
            identities[artifact.artifact_id] = {
                "status": "missing",
                "reason": "approved artifact raw identity is missing or ambiguous",
            }
            continue
        raw_path = raw_candidates[0]
        raw_size = raw_path.stat().st_size
        if artifact.size_bytes is not None and raw_size != artifact.size_bytes:
            identities[artifact.artifact_id] = {
                "status": "invalid",
                "reason": f"raw size mismatch: expected {artifact.size_bytes}, observed {raw_size}",
            }
            continue
        raw_sha256 = _sha256_file(raw_path)
        if artifact.sha256 is not None and raw_sha256 != artifact.sha256:
            identities[artifact.artifact_id] = {
                "status": "invalid",
                "reason": "raw SHA-256 does not match the frozen registry",
            }
            continue
        input_sha = _sha256_file(candidate)
        streams[artifact.artifact_id] = iter_export_positions(
            candidate,
            artifact_id=artifact.artifact_id,
            source_id=artifact.source_id,
            artifact_sha256=raw_sha256,
            source_revision=artifact.source_revision,
            input_sha256=input_sha,
        )
        metadata[artifact.artifact_id] = {
            "source_id": artifact.source_id,
            "artifact_sha256": raw_sha256,
            "source_revision": artifact.source_revision,
            "parser_version": "phase3_csa_export/v1",
            "normalization_version": "rust_replay_export/v1",
            "input_sha256": input_sha,
        }
        identities[artifact.artifact_id] = {
            "status": "ready",
            "path": candidate.relative_to(root).as_posix(),
            "input_sha256": input_sha,
            "source_artifact_sha256": raw_sha256,
            "source_revision": artifact.source_revision,
        }
    return streams, metadata, identities


def scan_phase10r_population(
    root: Path,
    *,
    data_root: Path | None = None,
    output_dir: Path | None = None,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, Any]:
    """Run the approved-population scan and write machine-readable completion proof."""

    resolved_data_root = data_root or root / "local/phase10r-data"
    resolved_output = output_dir or resolved_data_root / "leakage-scan"
    if (resolved_data_root / "replay-v2/replay-completion.json").is_file():
        from open_shogi_training.data.phase10r_scan_v2 import scan_phase10r_v2

        if output_dir is None:
            resolved_output = resolved_data_root / "leakage-scan-v2"
        return scan_phase10r_v2(
            root,
            data_root=resolved_data_root,
            output_dir=resolved_output,
            minimum_free_bytes=minimum_free_bytes,
        )
    registry = load_phase10r_registry(root / "configs/phase10r/source-registry.yaml")
    streams, metadata, identities = build_registry_scan_inputs(root, resolved_data_root)
    return scan_records(
        streams,
        output_dir=resolved_output,
        approved_artifact_ids=[artifact.artifact_id for artifact in registry.approved_artifacts()],
        source_metadata=metadata,
        input_identities=identities,
        minimum_free_bytes=minimum_free_bytes,
    )


__all__ = [
    "LEAKAGE_CLASSES",
    "MINIMUM_FREE_BYTES",
    "POSITION_SCHEMA",
    "SCAN_SCHEMA",
    "Phase10RScanError",
    "build_registry_scan_inputs",
    "iter_aobazero_positions",
    "iter_export_positions",
    "scan_phase10r_population",
    "scan_records",
]
