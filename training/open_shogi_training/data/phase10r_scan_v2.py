"""External-memory Phase 10R v2 identity and split-leakage scan."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from open_shogi_training.data.phase10r_identity import (
    canonical_game_hash,
    canonical_position_hash,
    history_id,
    resolve_canonical_collision,
    transposition_key,
)

SCAN_SCHEMA = "open_shogiai_phase10r_leakage_scan/v2"
POSITION_SCHEMA = "open_shogiai_phase10r_replayed_position/v2"
COMPLETION_SCHEMA = "open_shogiai_phase10r_scan_completion/v2"
SPLITS = frozenset(
    {"train", "validation", "source_held_out", "public_test", "internal_test", "final_holdout"}
)
PROTECTED = frozenset(
    {"validation", "source_held_out", "public_test", "internal_test", "final_holdout"}
)
SPLIT_RANK = {
    "public_test": 0,
    "reserved_holdout": 1,
    "final_holdout": 2,
    "source_held_out": 3,
    "validation": 4,
    "train": 5,
}
LEAKAGE_CLASSES = (
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


class Phase10RScanV2Error(RuntimeError):
    """Raised when the v2 scan cannot prove the frozen population."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Phase10RScanV2Error(f"scan input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            encoded = _json_bytes(dict(row))
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RScanV2Error(f"missing scan input: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase10RScanV2Error(f"non-object JSONL row at {path}:{line_number}")
            yield value


def _proof_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for proof in _read_jsonl(path):
        if proof.get("schema") != "open_shogiai_phase10r_replay_proof/v2":
            raise Phase10RScanV2Error(f"unknown replay proof schema: {path}")
        key = str(proof.get("source_game_id"))
        if key in result:
            raise Phase10RScanV2Error(f"duplicate source game proof: {key}")
        result[key] = proof
    return result


def _database(path: Path, input_manifest_sha256: str) -> sqlite3.Connection:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise Phase10RScanV2Error(f"refusing to reuse scan database: {path}")
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.executescript(
        """
        CREATE TABLE positions (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id TEXT NOT NULL,
            artifact_sha256 TEXT,
            source_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            archive_member_path TEXT NOT NULL,
            archive_member_sha256 TEXT NOT NULL,
            source_game_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            record_sha256 TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            game_id TEXT NOT NULL,
            position_index INTEGER NOT NULL,
            canonical_position_id TEXT NOT NULL,
            canonical_sfen TEXT NOT NULL,
            history_id TEXT NOT NULL,
            transposition_key TEXT NOT NULL,
            side_to_move TEXT NOT NULL,
            split TEXT NOT NULL,
            protected_role TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            normalization_version TEXT NOT NULL
        );
        CREATE TABLE effective_positions (row_id INTEGER PRIMARY KEY NOT NULL);
        CREATE INDEX positions_record_idx ON positions(record_id, split);
        CREATE INDEX positions_game_idx ON positions(game_id, split);
        CREATE INDEX positions_canonical_idx ON positions(canonical_position_id, split);
        CREATE INDEX positions_history_idx ON positions(history_id, split);
        CREATE INDEX positions_transposition_idx ON positions(transposition_key, split);
        CREATE INDEX positions_version_idx ON positions(
            canonical_position_id, parser_version, normalization_version
        );
        """
    )
    connection.commit()
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?)",
        [("schema", SCAN_SCHEMA), ("input_manifest_sha256", input_manifest_sha256)],
    )
    connection.commit()
    return connection


def _validate_position(row: Mapping[str, Any], proof: Mapping[str, Any]) -> None:
    if row.get("schema") != POSITION_SCHEMA:
        raise Phase10RScanV2Error("replayed position schema mismatch")
    index = row.get("position_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise Phase10RScanV2Error("position index is invalid")
    split = row.get("split")
    if not isinstance(split, str) or split not in SPLITS:
        raise Phase10RScanV2Error(f"unknown frozen split: {split!r}")
    if row.get("protected_role") != split:
        raise Phase10RScanV2Error("protected role does not match split")
    if row.get("source_game_id") != proof.get("source_game_id"):
        raise Phase10RScanV2Error("source game proof identity mismatch")
    if row.get("archive_member_path") != proof.get("archive_member_path"):
        raise Phase10RScanV2Error("archive member path disagrees with proof")
    if row.get("archive_member_sha256") != proof.get("archive_member_sha256"):
        raise Phase10RScanV2Error("archive member hash disagrees with proof")
    sfens = proof.get("canonical_position_hashes")
    transpositions = proof.get("transposition_keys")
    histories = proof.get("history_ids")
    moves = proof.get("usi_moves")
    if not all(isinstance(value, list) for value in (sfens, transpositions, histories, moves)):
        raise Phase10RScanV2Error("proof sequence fields are not lists")
    if len(sfens) != len(moves) + 1 or not index < len(sfens):
        raise Phase10RScanV2Error("proof sequence length is invalid")
    canonical = canonical_position_hash(str(row["canonical_sfen"]))
    if row.get("canonical_position_id") != canonical or sfens[index] != canonical:
        raise Phase10RScanV2Error("canonical position identity cannot be recomputed")
    if row.get("transposition_key") != transpositions[index]:
        raise Phase10RScanV2Error("transposition identity cannot be recomputed")
    expected_history = history_id(str(proof["canonical_initial_sfen"]), moves[:index])
    if row.get("history_id") != expected_history or histories[index] != expected_history:
        raise Phase10RScanV2Error("history identity cannot be recomputed")
    if row.get("game_id") != proof.get("canonical_game_hash"):
        raise Phase10RScanV2Error("canonical game identity disagrees with proof")
    if canonical_game_hash(str(proof["canonical_initial_sfen"]), moves) != row.get("game_id"):
        raise Phase10RScanV2Error("canonical game hash cannot be recomputed")
    if transposition_key(str(row["canonical_sfen"])) != row.get("transposition_key"):
        raise Phase10RScanV2Error("transposition key cannot be recomputed")
    if row.get("side_to_move") != (
        "black" if str(row["canonical_sfen"]).split(" ")[1] == "b" else "white"
    ):
        raise Phase10RScanV2Error("side-to-move identity disagrees with SFEN")


def _load_positions(
    root: Path, connection: sqlite3.Connection, completion: Mapping[str, Any]
) -> tuple[int, dict[str, int]]:
    streams = completion.get("streams")
    if not isinstance(streams, Mapping) or len(streams) != 33:
        raise Phase10RScanV2Error("replay completion does not contain all 33 streams")
    total = 0
    source_counts: dict[str, int] = defaultdict(int)
    for artifact_id in sorted(streams):
        summary = streams[artifact_id]
        if not isinstance(summary, Mapping) or summary.get("status") != "complete":
            raise Phase10RScanV2Error(f"stream is incomplete: {artifact_id}")
        position_path = root / "local/phase10r-data/replay-v2/positions" / f"{artifact_id}.jsonl"
        proof_path = root / "local/phase10r-data/replay-v2/proofs" / f"{artifact_id}.jsonl"
        if _sha256_file(position_path) != summary.get("position_sha256"):
            raise Phase10RScanV2Error(f"position stream hash mismatch: {artifact_id}")
        if _sha256_file(proof_path) != summary.get("proof_sha256"):
            raise Phase10RScanV2Error(f"proof stream hash mismatch: {artifact_id}")
        proofs = _proof_map(proof_path)
        counts: dict[str, set[int]] = defaultdict(set)
        batch: list[tuple[Any, ...]] = []
        for row in _read_jsonl(position_path):
            proof = proofs.get(str(row.get("source_game_id")))
            if proof is None:
                raise Phase10RScanV2Error(f"position has no replay proof: {artifact_id}")
            _validate_position(row, proof)
            counts[str(row["source_game_id"])].add(int(row["position_index"]))
            source_counts[str(row["source_id"])] += 1
            batch.append(
                tuple(
                    row[field]
                    for field in (
                        "artifact_id",
                        "artifact_sha256",
                        "source_id",
                        "source_revision",
                        "archive_member_path",
                        "archive_member_sha256",
                        "source_game_id",
                        "record_id",
                        "record_sha256",
                        "input_sha256",
                        "game_id",
                        "position_index",
                        "canonical_position_id",
                        "canonical_sfen",
                        "history_id",
                        "transposition_key",
                        "side_to_move",
                        "split",
                        "protected_role",
                        "parser_version",
                        "normalization_version",
                    )
                )
            )
            if len(batch) >= 2048:
                connection.executemany(
                    "INSERT INTO positions("
                    "artifact_id,artifact_sha256,source_id,source_revision,archive_member_path,"
                    "archive_member_sha256,source_game_id,record_id,record_sha256,input_sha256,"
                    "game_id,position_index,canonical_position_id,canonical_sfen,history_id,"
                    "transposition_key,side_to_move,split,protected_role,parser_version,"
                    "normalization_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                total += len(batch)
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO positions("
                "artifact_id,artifact_sha256,source_id,source_revision,archive_member_path,"
                "archive_member_sha256,source_game_id,record_id,record_sha256,input_sha256,"
                "game_id,position_index,canonical_position_id,canonical_sfen,history_id,"
                "transposition_key,side_to_move,split,protected_role,parser_version,"
                "normalization_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            total += len(batch)
        for source_game, indexes in counts.items():
            if indexes != set(range(max(indexes) + 1)):
                raise Phase10RScanV2Error(f"position prefix is incomplete: {source_game}")
        if total == 0 and not counts:
            raise Phase10RScanV2Error(f"empty replay stream: {artifact_id}")
    connection.commit()
    return total, dict(source_counts)


def _cross_split(connection: sqlite3.Connection, table: str, column: str) -> int:
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {column} FROM {table} GROUP BY {column} "
            "HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
    )


def _group_count(connection: sqlite3.Connection, column: str) -> int:
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {column} FROM positions "
            f"GROUP BY {column} HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    )


def _collision_decisions(
    connection: sqlite3.Connection, output: Path
) -> tuple[list[dict[str, Any]], set[int], dict[str, int]]:
    rows_for_resolution = (
        "row_id,artifact_id,source_game_id,record_id,position_index,canonical_position_id,"
        "split,parser_version,normalization_version"
    )
    decisions: list[dict[str, Any]] = []
    retained: set[int] = set()
    observed_groups: dict[str, int] = {}
    for identity_class, column in (
        ("canonical_position", "canonical_position_id"),
        ("transposition", "transposition_key"),
        ("history", "history_id"),
    ):
        query = (
            f"SELECT {column} FROM positions GROUP BY {column} HAVING COUNT(*) > 1 "
            f"ORDER BY {column}"
        )
        groups = 0
        for (key,) in connection.execute(query):
            occurrences = [
                dict(
                    zip(
                        (
                            "row_id",
                            "artifact_id",
                            "source_game_id",
                            "record_id",
                            "position_index",
                            "canonical_position_id",
                            "split",
                            "parser_version",
                            "normalization_version",
                        ),
                        values,
                        strict=True,
                    )
                )
                for values in connection.execute(
                    f"SELECT {rows_for_resolution} FROM positions "
                    f"WHERE {column} = ? ORDER BY row_id",
                    (key,),
                )
            ]
            if (
                identity_class == "transposition"
                and len({row["canonical_position_id"] for row in occurrences}) != 1
            ):
                raise Phase10RScanV2Error(f"transposition group mixes canonical positions: {key}")
            resolution = resolve_canonical_collision(occurrences)
            owner = resolution["owner"]
            owner_row = next(
                row
                for row in occurrences
                if row["artifact_id"] == owner["artifact_id"]
                and row["source_game_id"]
                == owner.get("source_game_id", owner.get("legacy_game_id"))
                and row["record_id"] == owner["record_id"]
                and row["position_index"] == owner["position_index"]
            )
            if identity_class == "canonical_position":
                retained.add(int(owner_row["row_id"]))
            decisions.append(
                {
                    "schema": "open_shogiai_phase10r_collision_decision/v2",
                    "identity_class": identity_class,
                    "identity": key,
                    "owner_row_id": owner_row["row_id"],
                    **resolution,
                }
            )
            groups += 1
        observed_groups[identity_class] = groups
    _write_jsonl(output / "phase10r-collision-decisions.jsonl", decisions)
    return decisions, retained, observed_groups


def _populate_effective(connection: sqlite3.Connection, retained: set[int]) -> None:
    all_rows = {int(row[0]) for row in connection.execute("SELECT row_id FROM positions")}
    # A duplicate identity group contributes its deterministic owner; singleton
    # rows are retained.  Derive the losing row ids with one SQL query per group
    # so the collision decision itself remains the source of truth.
    losing: set[int] = set()
    column = "canonical_position_id"
    for (key,) in connection.execute(
        f"SELECT {column} FROM positions GROUP BY {column} HAVING COUNT(*) > 1"
    ):
        for (row_id,) in connection.execute(
            f"SELECT row_id FROM positions WHERE {column} = ?", (key,)
        ):
            if int(row_id) not in retained:
                losing.add(int(row_id))
    connection.executemany(
        "INSERT INTO effective_positions(row_id) SELECT ?",
        ((row_id,) for row_id in sorted(all_rows - losing)),
    )
    connection.commit()


def _classes(
    connection: sqlite3.Connection, observed_groups: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    def effective(column: str) -> int:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM (SELECT p.{column} FROM positions p "
                f"JOIN effective_positions e ON e.row_id=p.row_id GROUP BY p.{column} "
                "HAVING COUNT(DISTINCT p.split) > 1)"
            ).fetchone()[0]
        )

    classes: dict[str, dict[str, Any]] = {}
    for name, column in (
        ("same_record_cross_split", "record_id"),
        ("same_game_cross_split", "game_id"),
        ("canonical_position_cross_split", "canonical_position_id"),
        ("protected_history_cross_split", "history_id"),
        ("prohibited_transposition_overlap", "transposition_key"),
    ):
        raw = _cross_split(connection, "positions", column)
        final = effective(column)
        classes[name] = {
            "status": "passed" if final == 0 else "failed",
            "observed_raw_cross_split_groups": raw,
            "effective_cross_split_groups": final,
            "observed_duplicate_groups": observed_groups.get(
                "canonical_position"
                if name == "canonical_position_cross_split"
                else "transposition"
                if name == "prohibited_transposition_overlap"
                else "history"
                if name == "protected_history_cross_split"
                else "none",
                0,
            ),
            "identity": column,
            "action": "frozen_precedence_owner_then_exclude_losing_occurrences",
        }
    train_eval = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT game_id FROM positions GROUP BY game_id "
            "HAVING SUM(split='train') > 0 AND SUM(split IN "
            "('validation','source_held_out','public_test','internal_test','final_holdout')) > 0)"
        ).fetchone()[0]
    )
    effective_train_eval = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT p.game_id FROM positions p "
            "JOIN effective_positions e ON e.row_id=p.row_id "
            "GROUP BY p.game_id HAVING SUM(p.split='train') > 0 AND "
            "SUM(p.split IN "
            "('validation','source_held_out','public_test','internal_test','final_holdout')) > 0)"
        ).fetchone()[0]
    )
    classes["different_plies_one_game_train_eval"] = {
        "status": "passed" if effective_train_eval == 0 else "failed",
        "observed_raw_cross_split_groups": train_eval,
        "effective_cross_split_groups": effective_train_eval,
        "identity": "game_id",
    }
    version_raw = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT canonical_position_id FROM positions "
            "GROUP BY canonical_position_id HAVING COUNT(DISTINCT parser_version) > 1 "
            "OR COUNT(DISTINCT normalization_version) > 1)"
        ).fetchone()[0]
    )
    version_effective = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT p.canonical_position_id FROM positions p "
            "JOIN effective_positions e ON e.row_id=p.row_id "
            "GROUP BY p.canonical_position_id HAVING COUNT(DISTINCT p.parser_version) > 1 "
            "OR COUNT(DISTINCT p.normalization_version) > 1)"
        ).fetchone()[0]
    )
    classes["parser_normalization_version_collision"] = {
        "status": "passed" if version_effective == 0 else "failed",
        "observed_raw_collision_groups": version_raw,
        "effective_collision_groups": version_effective,
        "identity": "canonical_position_id + parser_version + normalization_version",
        "action": "frozen_precedence_owner_then_exclude_losing_versions",
    }
    for name in LEAKAGE_CLASSES:
        classes.setdefault(
            name,
            {
                "status": "passed",
                "observed_raw_cross_split_groups": 0,
                "effective_cross_split_groups": 0,
                "identity": "not_applicable_under_frozen_source_registry",
                "formal_exclusion": "source is not admitted by the frozen Phase 10R registry",
            },
        )
    return classes


def _source_split_counts(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source, split, positions in connection.execute(
        "SELECT source_id, split, COUNT(*) FROM positions "
        "GROUP BY source_id, split ORDER BY source_id, split"
    ):
        result.setdefault(str(source), {})[str(split)] = int(positions)
    for source in sorted(result):
        groups = int(
            connection.execute(
                "SELECT COUNT(DISTINCT game_id) FROM positions WHERE source_id=?", (source,)
            ).fetchone()[0]
        )
        held = int(
            connection.execute(
                "SELECT COUNT(DISTINCT game_id) FROM positions "
                "WHERE source_id=? AND split='source_held_out'",
                (source,),
            ).fetchone()[0]
        )
        result[source]["game_groups"] = groups
        result[source]["source_held_out_game_groups"] = held
        result[source]["source_held_out_fraction"] = held / groups if groups else 0.0
    return result


def _effective_manifest(connection: sqlite3.Connection, output: Path) -> str:
    rows = connection.execute(
        "SELECT p.row_id,p.artifact_id,p.artifact_sha256,p.source_id,p.source_revision,"
        "p.archive_member_path,p.archive_member_sha256,p.source_game_id,p.record_id,p.record_sha256,"
        "p.input_sha256,p.game_id,p.position_index,p.canonical_position_id,p.canonical_sfen,p.history_id,"
        "p.transposition_key,p.side_to_move,p.split,p.protected_role,p.parser_version,"
        "p.normalization_version "
        "FROM positions p JOIN effective_positions e ON e.row_id=p.row_id ORDER BY p.row_id"
    )

    def generate() -> Iterable[Mapping[str, Any]]:
        for values in rows:
            (
                row_id,
                artifact_id,
                artifact_sha,
                source_id,
                revision,
                member_path,
                member_sha,
                source_game,
                record_id,
                record_sha,
                input_sha,
                game_id,
                index,
                canonical_id,
                sfen,
                history,
                transposition,
                side,
                split,
                role,
                parser,
                normalization,
            ) = values
            yield {
                "schema": "open_shogiai_phase10r_scanned_position/v2",
                "row_index": row_id,
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha,
                "source_id": source_id,
                "source_revision": revision,
                "archive_member_path": member_path,
                "archive_member_sha256": member_sha,
                "source_game_id": source_game,
                "record_id": record_id,
                "record_sha256": record_sha,
                "input_sha256": input_sha,
                "game_id": game_id,
                "position_index": index,
                "canonical_position_id": canonical_id,
                "canonical_sfen": sfen,
                "history_id": history,
                "transposition_key": transposition,
                "side_to_move": side,
                "split": split,
                "protected_role": role,
                "parser_version": parser,
                "normalization_version": normalization,
            }

    return _write_jsonl(output / "phase10r-leakage-manifest.jsonl", generate())


def scan_phase10r_v2(
    root: Path, *, data_root: Path, output_dir: Path, minimum_free_bytes: int = 0
) -> dict[str, Any]:
    replay_root = data_root / "replay-v2"
    completion_path = replay_root / "replay-completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("schema") != "open_shogiai_phase10r_replay_completion/v2"
        or completion.get("status") != "complete"
    ):
        raise Phase10RScanV2Error("replay completion proof is not complete")
    free = shutil.disk_usage(output_dir if output_dir.exists() else root).free
    if free < minimum_free_bytes:
        raise Phase10RScanV2Error(f"free disk is below frozen floor: {free}")
    streams = completion["streams"]
    input_manifest = {
        "schema": "open_shogiai_phase10r_scan_input_manifest/v2",
        "approved_artifact_ids": sorted(streams),
        "replay_completion_sha256": _sha256_file(completion_path),
        "replay_scan_identity": completion.get("scan_identity"),
        "inputs": {artifact: dict(streams[artifact]) for artifact in sorted(streams)},
    }
    input_bytes = _json_bytes(input_manifest)
    input_sha = _sha256_bytes(input_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase10r-scan-input-manifest.json").write_bytes(input_bytes)
    connection = _database(output_dir / "phase10r-scan-v2.sqlite3", input_sha)
    position_count, source_position_counts = _load_positions(root, connection, completion)
    _decisions, retained, observed_groups = _collision_decisions(connection, output_dir)
    _populate_effective(connection, retained)
    classes = _classes(connection, observed_groups)
    source_counts = _source_split_counts(connection)
    failures: list[dict[str, Any]] = []
    for source, counts in source_counts.items():
        if source == "aobazero":
            continue
        groups = int(counts["game_groups"])
        held = int(counts["source_held_out_game_groups"])
        if held < max(1, math.ceil(groups * 0.10)):
            failures.append(
                {"source_id": source, "reason": "frozen 10% source-held-out game groups incomplete"}
            )
    failures.extend(
        {
            "collision_class": name,
            "reason": "effective collision remains",
            "count": value.get(
                "effective_cross_split_groups", value.get("effective_collision_groups", 0)
            ),
        }
        for name, value in classes.items()
        if value.get("status") != "passed"
    )
    effective_positions = int(
        connection.execute("SELECT COUNT(*) FROM effective_positions").fetchone()[0]
    )
    unique_games = int(
        connection.execute("SELECT COUNT(DISTINCT game_id) FROM positions").fetchone()[0]
    )
    unique_positions = int(
        connection.execute(
            "SELECT COUNT(DISTINCT canonical_position_id) FROM positions"
        ).fetchone()[0]
    )
    manifest_sha = _effective_manifest(connection, output_dir)
    collision_sha = _sha256_file(output_dir / "phase10r-collision-decisions.jsonl")
    transposition = {
        "schema": "open_shogiai_phase10r_transposition_proof/v2",
        "status": "complete"
        if classes["prohibited_transposition_overlap"]["status"] == "passed"
        else "blocked",
        "identity": "transposition_key",
        "raw_records": position_count,
        "raw_duplicate_groups": _group_count(connection, "transposition_key"),
        "raw_cross_split_groups": classes["prohibited_transposition_overlap"][
            "observed_raw_cross_split_groups"
        ],
        "effective_records": effective_positions,
        "effective_cross_split_groups": classes["prohibited_transposition_overlap"][
            "effective_cross_split_groups"
        ],
        "database": "phase10r-scan-v2.sqlite3",
        "query": "group raw and effective SQLite positions by domain-separated transposition_key",
        "collision_decisions_sha256": collision_sha,
    }
    _write_json(output_dir / "phase10r-transposition-proof.json", transposition)
    _write_json(output_dir / "phase10r-source-split-counts.json", source_counts)
    _write_json(output_dir / "phase10r-prohibited-overlaps.json", classes)
    _write_json(
        output_dir / "phase10r-allowed-overlaps.json",
        {name: value for name, value in observed_groups.items()},
    )
    checkpoint = {
        "schema": "open_shogiai_phase10r_scan_checkpoint/v2",
        "input_manifest_sha256": input_sha,
        "completed_artifact_ids": sorted(streams),
        "position_count": position_count,
        "effective_position_count": effective_positions,
    }
    _write_json(output_dir / "phase10r-scan-checkpoint.json", checkpoint)
    completion_proof = {
        "schema": COMPLETION_SCHEMA,
        "status": "passed" if not failures else "blocked",
        "input_manifest_sha256": input_sha,
        "scan_manifest_sha256": manifest_sha,
        "collision_decisions_sha256": collision_sha,
        "approved_artifact_count": len(streams),
        "completed_artifact_count": len(streams),
        "total_scanned_records": position_count,
        "effective_published_records": effective_positions,
        "unique_games": unique_games,
        "unique_positions": unique_positions,
        "source_position_counts": source_position_counts,
        "source_split_counts": source_counts,
        "collisions": {
            name: {
                key: value[key]
                for key in value
                if key.endswith("groups") or key.endswith("count") or key == "status"
            }
            for name, value in classes.items()
        },
        "transposition_proof": transposition,
        "failures": failures,
        "database": "phase10r-scan-v2.sqlite3",
    }
    _write_json(output_dir / "phase10r-completion-proof.json", completion_proof)
    connection.close()
    return {
        "schema": SCAN_SCHEMA,
        "status": completion_proof["status"],
        "passed": completion_proof["status"] == "passed",
        "input_manifest_sha256": input_sha,
        "manifest_sha256": manifest_sha,
        "position_count": position_count,
        "total_scanned_records": position_count,
        "effective_published_records": effective_positions,
        "unique_games": unique_games,
        "unique_positions": unique_positions,
        "completed_artifact_count": len(streams),
        "approved_artifact_count": len(streams),
        "failures": failures,
        "leakage": {
            "passed": not failures,
            "classes": classes,
            "transposition_proof": transposition,
        },
        "output_dir": output_dir.as_posix(),
    }


__all__ = ["COMPLETION_SCHEMA", "SCAN_SCHEMA", "Phase10RScanV2Error", "scan_phase10r_v2"]
