"""SQLite opening statistics built from normalized training positions."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_shogi_training.data.gzip_jsonl import (
    compact_json_bytes,
    iter_jsonl_gzip,
    write_jsonl_gzip_atomic,
)

OPENING_SCHEMA = "phase3_opening_sqlite/v1"
POSITION_SCHEMA = "phase3_position/v1"
_MAX_PROVENANCE_BYTES = 1_048_576
_MAX_PLIES = 2_048
_OUTCOMES = frozenset({"black_win", "white_win", "draw", "unknown"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_USI_MOVE_RE = re.compile(r"^(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])$")
_POSITION_KEYS = frozenset(
    {
        "schema",
        "gameId",
        "canonicalSha256",
        "rawSha256",
        "sourceId",
        "split",
        "positionIndex",
        "sfen",
        "moveUsi",
        "nextSfen",
        "outcome",
        "terminalReason",
        "sideToMove",
        "fullPlies",
        "remainingPlies",
        "eligible",
        "terminalTail",
    }
)


@dataclass(frozen=True, slots=True)
class OpeningConfig:
    """Bounded build configuration; filtering thresholds are query-time only."""

    max_input_rows: int = 100_000_000
    max_input_bytes: int = 1_073_741_824
    max_input_uncompressed_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        for name in (
            "max_input_rows",
            "max_input_bytes",
            "max_input_uncompressed_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "phase3_opening_config/v1",
            "inputSchema": POSITION_SCHEMA,
            "onlySplit": "train",
            "requireEligible": True,
            "maxInputRows": self.max_input_rows,
            "maxInputBytes": self.max_input_bytes,
            "maxInputUncompressedBytes": self.max_input_uncompressed_bytes,
        }


_DEFAULT_OPENING_CONFIG = OpeningConfig()


@dataclass(frozen=True, slots=True)
class OpeningBuildReport:
    """Observed counts from one completed opening database build."""

    input_rows: int
    included_rows: int
    state_count: int
    move_count: int
    source_count: int
    sha256: str
    size: int


def state_identity(sfen: str) -> tuple[str, str, str]:
    """Return SHA-256 key, full state without move number, and side to move."""

    if not isinstance(sfen, str) or not sfen.isascii() or len(sfen) > 1_024:
        raise ValueError("SFEN must be an ASCII string of at most 1,024 bytes")
    fields = sfen.split(" ")
    if len(fields) != 4 or any(not field for field in fields):
        raise ValueError("SFEN must contain exactly four single-space-separated fields")
    if fields[1] not in {"b", "w"}:
        raise ValueError("SFEN side-to-move field must be 'b' or 'w'")
    if not fields[3].isascii() or not fields[3].isdecimal() or int(fields[3]) <= 0:
        raise ValueError("SFEN move number must be a positive decimal integer")
    state_sfen = " ".join(fields[:3])
    return hashlib.sha256(state_sfen.encode("utf-8")).hexdigest(), state_sfen, fields[1]


def wilson_interval_95(wins: int, trials: int) -> tuple[float | None, float | None]:
    """Return the two-sided 95% Wilson score interval."""

    if isinstance(wins, bool) or isinstance(trials, bool):
        raise ValueError("wins and trials must be integers")
    if not 0 <= wins <= trials:
        raise ValueError("wins must be between zero and trials")
    if trials == 0:
        return None, None
    z = 1.959963984540054
    proportion = wins / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return center - margin, center + margin


def build_opening_database(
    positions_path: Path,
    output_path: Path,
    *,
    provenance: Mapping[str, Any],
    config: OpeningConfig = _DEFAULT_OPENING_CONFIG,
) -> OpeningBuildReport:
    """Build an atomic non-overwriting SQLite database from train+eligible rows."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing opening database: {output_path}")
    provenance_bytes = compact_json_bytes(dict(provenance))
    if len(provenance_bytes) > _MAX_PROVENANCE_BYTES:
        raise ValueError(f"provenance exceeds {_MAX_PROVENANCE_BYTES} bytes")
    provenance_json = provenance_bytes.decode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    input_snapshot: Path | None = None

    input_rows = 0
    included_rows = 0
    try:
        input_snapshot, input_sha256 = _snapshot_input(
            positions_path,
            output_path.parent,
            maximum=config.max_input_bytes,
        )
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            metadata = {
                "schema": OPENING_SCHEMA,
                "config": compact_json_bytes(config.as_dict()).decode("utf-8"),
                "provenance": provenance_json,
                "input_sha256": input_sha256,
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            for row in iter_jsonl_gzip(
                input_snapshot,
                max_compressed_bytes=config.max_input_bytes,
                max_uncompressed_bytes=config.max_input_uncompressed_bytes,
                max_records=config.max_input_rows,
            ):
                input_rows += 1
                observation = _parse_position_observation(row)
                if observation is None:
                    continue
                _add_observation(connection, observation)
                included_rows += 1
            connection.commit()
            state_count = _single_count(connection, "states")
            move_count = _single_count(connection, "moves")
            source_count = _single_count(connection, "source_counts")
            connection.execute("PRAGMA optimize")
        finally:
            connection.close()
        with temporary.open("rb") as database_file:
            os.fsync(database_file.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing opening database: {output_path}"
            ) from None
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if input_snapshot is not None:
            input_snapshot.unlink(missing_ok=True)

    return OpeningBuildReport(
        input_rows=input_rows,
        included_rows=included_rows,
        state_count=state_count,
        move_count=move_count,
        source_count=source_count,
        sha256=_sha256_file(output_path),
        size=output_path.stat().st_size,
    )


def query_opening_moves(
    database_path: Path,
    sfen: str,
    *,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Query derived opening statistics without deleting low-count observations."""

    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count <= 0:
        raise ValueError("min_count must be a positive integer")
    state_key, state_sfen, _ = state_identity(sfen)
    connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        _verify_database_schema(connection)
        stored = connection.execute(
            "SELECT state_sfen FROM states WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        if stored is None:
            return []
        if stored["state_sfen"] != state_sfen:
            raise RuntimeError("state-key collision detected while querying opening database")
        rows = connection.execute(
            """
            SELECT
                move_usi, observation_count, actor_wins, actor_losses, draws, unknown,
                black_wins, white_wins, full_plies_sum, remaining_plies_sum
            FROM moves
            WHERE state_key = ? AND observation_count >= ?
            ORDER BY observation_count DESC, move_usi ASC
            """,
            (state_key, min_count),
        )
        return [_derived_stats(connection, state_key, row) for row in rows]
    finally:
        connection.close()


def export_opening_jsonl(
    database_path: Path,
    output_path: Path,
    *,
    min_count: int = 1,
) -> None:
    """Export all qualifying state/move rows to deterministic gzip JSONL."""

    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count <= 0:
        raise ValueError("min_count must be a positive integer")
    connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        _verify_database_schema(connection)
        rows = connection.execute(
            """
            SELECT
                states.state_key, states.state_sfen, moves.move_usi,
                moves.observation_count, moves.actor_wins, moves.actor_losses,
                moves.draws, moves.unknown, moves.black_wins, moves.white_wins,
                moves.full_plies_sum, moves.remaining_plies_sum
            FROM moves
            JOIN states USING (state_key)
            WHERE moves.observation_count >= ?
            ORDER BY states.state_key ASC, moves.observation_count DESC, moves.move_usi ASC
            """,
            (min_count,),
        )

        def exported_rows() -> Any:
            for row in rows:
                result = _derived_stats(connection, row["state_key"], row)
                result.update(
                    {
                        "schema": "phase3_opening_export/v1",
                        "stateKey": row["state_key"],
                        "stateSfen": row["state_sfen"],
                    }
                )
                yield result

        write_jsonl_gzip_atomic(output_path, exported_rows())
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class _Observation:
    state_key: str
    state_sfen: str
    side_to_move: str
    move_usi: str
    source_id: str
    outcome: str
    full_plies: int
    remaining_plies: int


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA application_id = 1330857799;
        PRAGMA user_version = 1;
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = FULL;
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE states (
            state_key TEXT PRIMARY KEY NOT NULL,
            state_sfen TEXT UNIQUE NOT NULL,
            side_to_move TEXT NOT NULL CHECK (side_to_move IN ('b', 'w'))
        ) WITHOUT ROWID;
        CREATE TABLE moves (
            state_key TEXT NOT NULL REFERENCES states(state_key),
            move_usi TEXT NOT NULL,
            observation_count INTEGER NOT NULL CHECK (observation_count > 0),
            actor_wins INTEGER NOT NULL CHECK (actor_wins >= 0),
            actor_losses INTEGER NOT NULL CHECK (actor_losses >= 0),
            draws INTEGER NOT NULL CHECK (draws >= 0),
            unknown INTEGER NOT NULL CHECK (unknown >= 0),
            black_wins INTEGER NOT NULL CHECK (black_wins >= 0),
            white_wins INTEGER NOT NULL CHECK (white_wins >= 0),
            full_plies_sum INTEGER NOT NULL CHECK (full_plies_sum >= 0),
            remaining_plies_sum INTEGER NOT NULL CHECK (remaining_plies_sum >= 0),
            PRIMARY KEY (state_key, move_usi)
        ) WITHOUT ROWID;
        CREATE TABLE source_counts (
            state_key TEXT NOT NULL,
            move_usi TEXT NOT NULL,
            source_id TEXT NOT NULL,
            observation_count INTEGER NOT NULL CHECK (observation_count > 0),
            PRIMARY KEY (state_key, move_usi, source_id),
            FOREIGN KEY (state_key, move_usi) REFERENCES moves(state_key, move_usi)
        ) WITHOUT ROWID;
        CREATE INDEX moves_count_index
            ON moves(state_key, observation_count DESC, move_usi);
        """
    )


def _parse_position_observation(row: dict[str, Any]) -> _Observation | None:
    if frozenset(row) != _POSITION_KEYS:
        missing = sorted(_POSITION_KEYS - frozenset(row))
        unknown = sorted(frozenset(row) - _POSITION_KEYS)
        raise ValueError(f"position row keys mismatch; missing={missing}, unknown={unknown}")
    if row["schema"] != POSITION_SCHEMA:
        raise ValueError(f"unsupported position schema: {row['schema']!r}")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError("position split is invalid")
    if not isinstance(row["eligible"], bool) or not isinstance(row["terminalTail"], bool):
        raise ValueError("eligible and terminalTail must be booleans")
    if row["outcome"] not in _OUTCOMES:
        raise ValueError("position outcome is invalid")
    for key in ("fullPlies", "remainingPlies", "positionIndex"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_PLIES:
            raise ValueError(f"{key} must be an integer between zero and {_MAX_PLIES}")
    if row["remainingPlies"] > row["fullPlies"]:
        raise ValueError("remainingPlies must not exceed fullPlies")
    if row["sideToMove"] not in {"black", "white"}:
        raise ValueError("sideToMove must be black or white")
    for key in ("gameId", "canonicalSha256", "rawSha256", "sourceId", "sfen"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("gameId", "canonicalSha256", "rawSha256"):
        if _SHA256_RE.fullmatch(row[key]) is None:
            raise ValueError(f"{key} must be 64 lowercase hexadecimal characters")
    if row["gameId"] != row["canonicalSha256"]:
        raise ValueError("gameId must equal canonicalSha256")
    if _SOURCE_ID_RE.fullmatch(row["sourceId"]) is None:
        raise ValueError("sourceId has an invalid bounded identifier")
    if len(row["sfen"].encode("utf-8")) > 1_024:
        raise ValueError("sfen exceeds 1,024 UTF-8 bytes")
    if row["moveUsi"] is not None and (not isinstance(row["moveUsi"], str) or not row["moveUsi"]):
        raise ValueError("moveUsi must be a non-empty string or null")
    if row["nextSfen"] is not None and (
        not isinstance(row["nextSfen"], str) or not row["nextSfen"]
    ):
        raise ValueError("nextSfen must be a non-empty string or null")
    if row["moveUsi"] is not None and _USI_MOVE_RE.fullmatch(row["moveUsi"]) is None:
        raise ValueError("moveUsi is not valid bounded USI move notation")
    if row["nextSfen"] is not None:
        state_identity(row["nextSfen"])
    if row["positionIndex"] + row["remainingPlies"] != row["fullPlies"]:
        raise ValueError("positionIndex + remainingPlies must equal fullPlies")
    if (row["moveUsi"] is None) != (row["nextSfen"] is None):
        raise ValueError("moveUsi and nextSfen must both be present or both be null")
    if row["moveUsi"] is None and (
        row["remainingPlies"] != 0 or row["eligible"] or not row["terminalTail"]
    ):
        raise ValueError("terminal position row has inconsistent null/tail fields")
    if row["moveUsi"] is not None and row["remainingPlies"] == 0:
        raise ValueError("non-terminal position must have positive remainingPlies")
    terminal_reason = row["terminalReason"]
    if terminal_reason is not None and (
        not isinstance(terminal_reason, str)
        or not terminal_reason
        or len(terminal_reason.encode("utf-8")) > 64
        or not terminal_reason.isascii()
    ):
        raise ValueError("terminalReason must be bounded ASCII or null")

    if row["split"] != "train" or not row["eligible"]:
        return None
    if row["terminalTail"]:
        raise ValueError("an eligible training position cannot be terminal-tail")
    if row["moveUsi"] is None or row["nextSfen"] is None:
        raise ValueError("an eligible training position must have a move and successor")

    state_key, state_sfen, side = state_identity(row["sfen"])
    expected_side = "black" if side == "b" else "white"
    if row["sideToMove"] != expected_side:
        raise ValueError("sideToMove disagrees with SFEN")
    return _Observation(
        state_key=state_key,
        state_sfen=state_sfen,
        side_to_move=side,
        move_usi=row["moveUsi"],
        source_id=row["sourceId"],
        outcome=row["outcome"],
        full_plies=row["fullPlies"],
        remaining_plies=row["remainingPlies"],
    )


def _add_observation(connection: sqlite3.Connection, observation: _Observation) -> None:
    existing = connection.execute(
        "SELECT state_sfen, side_to_move FROM states WHERE state_key = ?",
        (observation.state_key,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO states(state_key, state_sfen, side_to_move) VALUES (?, ?, ?)",
            (observation.state_key, observation.state_sfen, observation.side_to_move),
        )
    elif existing != (observation.state_sfen, observation.side_to_move):
        raise RuntimeError("SHA-256 state-key collision detected while building opening database")

    black_win = int(observation.outcome == "black_win")
    white_win = int(observation.outcome == "white_win")
    draw = int(observation.outcome == "draw")
    unknown = int(observation.outcome == "unknown")
    actor_win = int(
        (observation.side_to_move == "b" and black_win)
        or (observation.side_to_move == "w" and white_win)
    )
    actor_loss = int(
        (observation.side_to_move == "b" and white_win)
        or (observation.side_to_move == "w" and black_win)
    )
    connection.execute(
        """
        INSERT INTO moves(
            state_key, move_usi, observation_count, actor_wins, actor_losses, draws,
            unknown, black_wins, white_wins, full_plies_sum, remaining_plies_sum
        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(state_key, move_usi) DO UPDATE SET
            observation_count = observation_count + 1,
            actor_wins = actor_wins + excluded.actor_wins,
            actor_losses = actor_losses + excluded.actor_losses,
            draws = draws + excluded.draws,
            unknown = unknown + excluded.unknown,
            black_wins = black_wins + excluded.black_wins,
            white_wins = white_wins + excluded.white_wins,
            full_plies_sum = full_plies_sum + excluded.full_plies_sum,
            remaining_plies_sum = remaining_plies_sum + excluded.remaining_plies_sum
        """,
        (
            observation.state_key,
            observation.move_usi,
            actor_win,
            actor_loss,
            draw,
            unknown,
            black_win,
            white_win,
            observation.full_plies,
            observation.remaining_plies,
        ),
    )
    connection.execute(
        """
        INSERT INTO source_counts(state_key, move_usi, source_id, observation_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(state_key, move_usi, source_id) DO UPDATE SET
            observation_count = observation_count + 1
        """,
        (observation.state_key, observation.move_usi, observation.source_id),
    )


def _derived_stats(
    connection: sqlite3.Connection,
    state_key: str,
    row: sqlite3.Row,
) -> dict[str, Any]:
    count = row["observation_count"]
    wins = row["actor_wins"]
    losses = row["actor_losses"]
    draws = row["draws"]
    unknown = row["unknown"]
    scored = wins + losses + draws
    decisive = wins + losses
    black_wins = row["black_wins"]
    white_wins = row["white_wins"]
    side_decisive = black_wins + white_wins
    lower, upper = wilson_interval_95(wins, decisive)
    source_rows = connection.execute(
        """
        SELECT source_id, observation_count
        FROM source_counts
        WHERE state_key = ? AND move_usi = ?
        ORDER BY source_id
        """,
        (state_key, row["move_usi"]),
    ).fetchall()
    return {
        "moveUsi": row["move_usi"],
        "count": count,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "unknown": unknown,
        "scoreRate": (wins + 0.5 * draws) / scored if scored else None,
        "decisiveN": decisive,
        "decisiveWinRate": wins / decisive if decisive else None,
        "decisiveWinRateWilson95Low": lower,
        "decisiveWinRateWilson95High": upper,
        "blackWins": black_wins,
        "whiteWins": white_wins,
        "sideSpecificDecisiveN": side_decisive,
        "blackDecisiveWinRate": black_wins / side_decisive if side_decisive else None,
        "whiteDecisiveWinRate": white_wins / side_decisive if side_decisive else None,
        "averageFullPlies": row["full_plies_sum"] / count,
        "averageRemainingPlies": row["remaining_plies_sum"] / count,
        "sourceCounts": {
            source_row["source_id"]: source_row["observation_count"] for source_row in source_rows
        },
    }


def _single_count(connection: sqlite3.Connection, table: str) -> int:
    if table not in {"states", "moves", "source_counts"}:
        raise ValueError("invalid table")
    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    if row is None:
        raise RuntimeError(f"failed to count {table}")
    return int(row[0])


def _verify_database_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT value FROM metadata WHERE key = 'schema'").fetchone()
    if row is None or row[0] != OPENING_SCHEMA:
        raise ValueError("unsupported or missing opening database schema")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_only_uri(path: Path) -> str:
    return f"{path.resolve(strict=True).as_uri()}?mode=ro"


def _snapshot_input(path: Path, directory: Path, *, maximum: int) -> tuple[Path, str]:
    descriptor, name = tempfile.mkstemp(prefix=".opening-input.", dir=directory)
    snapshot = Path(name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            with path.open("rb") as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > maximum:
                        raise ValueError(f"compressed JSONL exceeds {maximum} bytes")
                    digest.update(chunk)
                    output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot, digest.hexdigest()
