"""Durable Phase 10R preparation over the frozen replay/identity population.

The existing Phase 10R model and bounded trainer remain the source of truth for the
architecture and target semantics.  This module only supplies the missing disk-backed
population stream: it joins effective SQLite identities to replay proofs, obtains legal
masks/history facts from the repository Rust rules core, and writes immutable preparation
artifacts.  It deliberately never queries a protected final holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Final

from open_shogi_training.data.phase10r_identity import canonical_game_hash
from open_shogi_training.phase10r import allocate_source_counts
from open_shogi_training.phase10r_model import HistoryFacts
from open_shogi_training.phase10r_training import Phase10RExample

PREPARATION_SCHEMA: Final = "open_shogiai_phase10r_preparation/v1"
TRAINING_EXAMPLE_SCHEMA: Final = "phase10r_training_example/v1"
HELPER_SCHEMA: Final = "phase10r_replay_features/v1"
PREPARATION_SEED: Final = 20_260_729
TRAINING_SOURCES: Final = ("aobazero", "wcsc", "denryu")
TRAINING_SPLIT: Final = "train"
EVALUATION_SPLITS: Final = ("validation", "source_held_out")
PREPARATION_SPLITS: Final = (TRAINING_SPLIT, *EVALUATION_SPLITS)
SOURCE_WEIGHTS: Final = {"aobazero": 1.0, "wcsc": 1.0, "denryu": 0.5}
MAX_POSITIONS_PER_GAME_PER_EPOCH: Final = 128
MAX_HELPER_LINE_BYTES: Final = 64 * 1024 * 1024
HELPER_BUILD_TIMEOUT_SECONDS: Final = 300
SCALE_COUNTS: Final = {
    "1m": 1_000_000,
    "10m": 10_000_000,
    "50m": 50_000_000,
    "100m": 100_000_000,
    "500m": 500_000_000,
    "1b": 1_000_000_000,
}
CONFIG_PATHS: Final = (
    "configs/phase10r/dataset-mixture.yaml",
    "configs/phase10r/curriculum.yaml",
    "configs/phase10r/model-matrix.yaml",
    "configs/phase10r/target-semantics.yaml",
    "configs/phase10r/active-learning.yaml",
    "configs/phase10r/arena-gates.yaml",
    "configs/phase10r/selfplay.yaml",
    "configs/phase10r/resource-budget.yaml",
    "configs/phase10r/holdout-policy.yaml",
    "configs/phase10r/identity.yaml",
    "configs/phase10r/source-registry.yaml",
    "configs/phase10r/normalization.yaml",
    "configs/phase10r/deduplication.yaml",
    "configs/phase10r/storage-budget.yaml",
    "configs/teacher/apery-v2.0.0.yaml",
)
_DB_COLUMNS: Final = (
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


class Phase10RExecutionError(RuntimeError):
    """Raised when preparation cannot prove the frozen execution contract."""


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise Phase10RExecutionError(f"non-canonical execution JSON: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Phase10RExecutionError(f"execution input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RExecutionError(f"missing execution JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase10RExecutionError(f"cannot read execution JSON: {path}") from error
    if not isinstance(value, dict):
        raise Phase10RExecutionError(f"execution JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RExecutionError(f"missing execution JSONL: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(line.encode("utf-8")) > MAX_HELPER_LINE_BYTES:
                raise Phase10RExecutionError(
                    f"execution JSONL line is too large: {path}:{line_number}"
                )
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase10RExecutionError(
                    f"invalid execution JSONL: {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise Phase10RExecutionError(
                    f"execution JSONL row is not an object: {path}:{line_number}"
                )
            yield value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> tuple[str, int]:
    return _publish_immutable(path, _json_bytes(dict(value)))


def _publish_immutable(path: Path, payload: bytes) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise Phase10RExecutionError(
            f"refusing to overwrite immutable preparation artifact: {path}"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Phase10RExecutionError(
                f"refusing to overwrite immutable preparation artifact: {path}"
            ) from error
        return _sha256_bytes(payload), len(payload)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_jsonl_immutable(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    resource_check: Callable[[], object] | None = None,
) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise Phase10RExecutionError(
            f"refusing to overwrite immutable preparation artifact: {path}"
        )
    digest = hashlib.sha256()
    count = 0
    size = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                encoded = _json_bytes(dict(row))
                if len(encoded) > MAX_HELPER_LINE_BYTES:
                    raise Phase10RExecutionError(f"preparation row is too large: {path}")
                handle.write(encoded)
                digest.update(encoded)
                count += 1
                size += len(encoded)
                if resource_check is not None and size // (1024**3) != (size - len(encoded)) // (
                    1024**3
                ):
                    resource_check()
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Phase10RExecutionError(
                f"refusing to overwrite immutable preparation artifact: {path}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"sha256": digest.hexdigest(), "bytes": size, "rows": count}


def _data_root(root: Path) -> Path:
    value = Path(os.environ.get("OPENSHOGI_DATA_ROOT", root / "local/phase10r-data"))
    return value if value.is_absolute() else root / value


def _scan_identity(root: Path, data_root: Path) -> dict[str, Any]:
    scan_root = data_root / "leakage-scan-v2"
    completion_path = data_root / "replay-v2/replay-completion.json"
    proof_path = scan_root / "phase10r-completion-proof.json"
    database_path = scan_root / "phase10r-scan-v2.sqlite3"
    completion = _read_json(completion_path)
    proof = _read_json(proof_path)
    for value, name, expected in (
        (completion, "replay completion", "open_shogiai_phase10r_replay_completion/v2"),
        (proof, "scan completion", "open_shogiai_phase10r_scan_completion/v2"),
    ):
        if value.get("schema") != expected or value.get("status") not in {"complete", "passed"}:
            raise Phase10RExecutionError(f"{name} is not complete")
    if proof.get("status") != "passed" or proof.get("failures"):
        raise Phase10RExecutionError("v2 split-leakage proof is not passed")
    transposition = proof.get("transposition_proof")
    if not isinstance(transposition, Mapping) or transposition.get("status") != "complete":
        raise Phase10RExecutionError("v2 transposition proof is not complete")
    if not database_path.is_file() or database_path.is_symlink():
        raise Phase10RExecutionError("v2 scan database is missing")
    completion_sha = _sha256_file(completion_path)
    input_manifest_sha = proof.get("input_manifest_sha256")
    scan_manifest_sha = proof.get("scan_manifest_sha256")
    if not isinstance(input_manifest_sha, str) or not isinstance(scan_manifest_sha, str):
        raise Phase10RExecutionError("v2 scan identity is incomplete")
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        database_input = connection.execute(
            "SELECT value FROM meta WHERE key='input_manifest_sha256'"
        ).fetchone()
    except sqlite3.DatabaseError as error:
        raise Phase10RExecutionError("v2 scan database metadata is unreadable") from error
    finally:
        connection.close()
    if database_input is None or database_input[0] != input_manifest_sha:
        raise Phase10RExecutionError("v2 scan database input identity mismatches its proof")
    return {
        "data_root": data_root.as_posix(),
        "database": database_path,
        "completion_path": completion_path,
        "completion_sha256": completion_sha,
        "input_manifest_sha256": input_manifest_sha,
        "scan_manifest_sha256": scan_manifest_sha,
        "effective_records": proof.get("effective_published_records"),
        "unique_positions": proof.get("unique_positions"),
        "unique_games": proof.get("unique_games"),
    }


def _configuration_hashes(root: Path) -> dict[str, str]:
    paths = (*CONFIG_PATHS, "configs/phase10r/frozen-controls.sha256")
    return {path: _sha256_file(root / path) for path in paths}


def _resource_check(root: Path, data_root: Path) -> dict[str, int | bool]:
    usage = shutil.disk_usage(data_root if data_root.exists() else root)
    minimum = 150 * 1024**3
    if usage.free < minimum:
        raise Phase10RExecutionError(
            f"free disk crossed the frozen 150 GiB floor: {usage.free} < {minimum}"
        )
    return {
        "free_disk_bytes": int(usage.free),
        "minimum_free_bytes": minimum,
        "disk_passed": True,
    }


def _effective_rows(
    connection: sqlite3.Connection, *, splits: Sequence[str]
) -> Iterator[dict[str, Any]]:
    placeholders = ",".join("?" for _ in splits)
    query = (
        "SELECT "
        + ",".join(f"p.{column}" for column in _DB_COLUMNS)
        + " FROM positions p JOIN effective_positions e ON e.row_id=p.row_id "
        f"WHERE p.split IN ({placeholders}) "
        "ORDER BY p.artifact_id,p.source_game_id,p.position_index"
    )
    for values in connection.execute(query, splits):
        yield dict(zip(_DB_COLUMNS, values, strict=True))


def _effective_game_ids(
    connection: sqlite3.Connection, *, splits: Sequence[str]
) -> dict[str, set[str]]:
    placeholders = ",".join("?" for _ in splits)
    rows = connection.execute(
        "SELECT DISTINCT p.artifact_id,p.source_game_id FROM positions p "
        "JOIN effective_positions e ON e.row_id=p.row_id "
        f"WHERE p.split IN ({placeholders}) "
        "ORDER BY p.artifact_id,p.source_game_id",
        splits,
    )
    result: dict[str, set[str]] = {}
    for artifact_id, source_game_id in rows:
        result.setdefault(str(artifact_id), set()).add(str(source_game_id))
    return result


def _proofs_and_payloads(
    data_root: Path, artifact_id: str, relevant_games: set[str]
) -> dict[str, dict[str, Any]]:
    proof_path = data_root / "replay-v2/proofs" / f"{artifact_id}.jsonl"
    replay_path = data_root / "replay-v2/replays" / f"{artifact_id}.jsonl"
    proofs_by_hash: dict[str, dict[str, Any]] = {}
    for proof in _read_jsonl(proof_path):
        if proof.get("schema") != "open_shogiai_phase10r_replay_proof/v2":
            raise Phase10RExecutionError(f"unknown proof schema for {artifact_id}")
        source_game = proof.get("source_game_id")
        if not isinstance(source_game, str):
            raise Phase10RExecutionError(f"proof source-game identity is missing for {artifact_id}")
        if source_game not in relevant_games:
            continue
        if proof.get("complete_legal_replay") is not True:
            raise Phase10RExecutionError(
                f"relevant proof is not a complete legal replay: {source_game}"
            )
        initial = proof.get("canonical_initial_sfen")
        moves = proof.get("usi_moves")
        game_hash = proof.get("canonical_game_hash")
        if (
            not isinstance(initial, str)
            or not isinstance(moves, list)
            or not isinstance(game_hash, str)
        ):
            raise Phase10RExecutionError(f"proof replay identity is incomplete: {source_game}")
        if canonical_game_hash(initial, moves) != game_hash:
            raise Phase10RExecutionError(f"proof game hash cannot be recomputed: {source_game}")
        if game_hash in proofs_by_hash:
            raise Phase10RExecutionError(f"duplicate proof game hash: {game_hash}")
        proofs_by_hash[game_hash] = proof
    if len(proofs_by_hash) != len(relevant_games):
        raise Phase10RExecutionError(
            f"replay proof coverage is incomplete for {artifact_id}: "
            f"{len(proofs_by_hash)} != {len(relevant_games)}"
        )
    candidates: dict[str, list[dict[str, Any]]] = {}
    for replay in _read_jsonl(replay_path):
        if replay.get("schema") != "phase3_csa_export/v1" or replay.get("status") != "ok":
            continue
        initial = replay.get("initialSfen")
        moves = replay.get("usiMoves")
        if not isinstance(initial, str) or not isinstance(moves, list):
            raise Phase10RExecutionError(f"replay payload is incomplete for {artifact_id}")
        game_hash = canonical_game_hash(initial, moves)
        proof = proofs_by_hash.get(game_hash)
        if proof is None:
            continue
        outcome = replay.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise Phase10RExecutionError(f"replay outcome is missing: {proof['source_game_id']}")
        if initial != proof["canonical_initial_sfen"] or moves != proof["usi_moves"]:
            raise Phase10RExecutionError(
                f"replay payload does not exactly match its proof: {proof['source_game_id']}"
            )
        candidates.setdefault(game_hash, []).append(
            {
                "outcome": outcome,
                "terminal": replay.get("terminalReason"),
            }
        )
    payloads: dict[str, dict[str, Any]] = {}
    for game_hash, proof in proofs_by_hash.items():
        source_game = str(proof["source_game_id"])
        options = candidates.get(game_hash, [])
        if not options:
            continue
        known_outcomes = {
            str(option["outcome"]) for option in options if option["outcome"] != "unknown"
        }
        if not known_outcomes <= {"black_win", "white_win", "draw"}:
            raise Phase10RExecutionError(f"replay outcome is unknown: {source_game}")
        if len(known_outcomes) > 1:
            raise Phase10RExecutionError(
                f"replay outcomes conflict for one proven game: {source_game}"
            )
        proof_terminal = str(proof.get("terminal") or "").removeprefix("%")
        selected = sorted(
            options,
            key=lambda option: (
                option["outcome"] == "unknown",
                str(option.get("terminal") or "").removeprefix("%") != proof_terminal,
                option.get("terminal") is None,
                str(option.get("terminal") or ""),
            ),
        )[0]
        payloads[source_game] = {
            "source_game_id": source_game,
            "initial_sfen": str(proof["canonical_initial_sfen"]),
            "usi_moves": list(proof["usi_moves"]),
            "outcome": selected["outcome"],
            "terminal": selected.get("terminal"),
            "canonical_game_hash": game_hash,
            "replay_candidate_count": len(options),
            "replay_candidates": options,
        }
    if set(payloads) != relevant_games:
        raise Phase10RExecutionError(
            f"replay payload coverage is incomplete for {artifact_id}: "
            f"{len(payloads)} != {len(relevant_games)}"
        )
    return payloads


class _ReplayFeatureProcess:
    """Persistent subprocess for the authoritative Rust legal-root/history path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "target/release/examples/phase10r_replay_features"
        self.process: subprocess.Popen[str] | None = None
        self.sha256: str | None = None

    def build_if_missing(self) -> None:
        completed = subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "open-shogi-core",
                "--example",
                "phase10r_replay_features",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=HELPER_BUILD_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0 or not self.path.is_file() or self.path.is_symlink():
            detail = (completed.stderr or completed.stdout)[-4_096:]
            raise Phase10RExecutionError(f"cannot build replay-feature helper: {detail}")
        self.sha256 = _sha256_file(self.path)

    def __enter__(self) -> _ReplayFeatureProcess:
        self.build_if_missing()
        assert self.sha256 is not None
        try:
            self.process = subprocess.Popen(
                [str(self.path)],
                cwd=self.root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            raise Phase10RExecutionError(f"cannot start replay-feature helper: {error}") from error
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self.sha256 != _sha256_file(self.path):
            raise Phase10RExecutionError("replay-feature helper changed during preparation")
        self.process = None

    def game(self, initial_sfen: str, moves: Sequence[str]) -> list[dict[str, Any]]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise Phase10RExecutionError("replay-feature helper is not running")
        request = _json_bytes({"initialSfen": initial_sfen, "usiMoves": list(moves)})
        if len(request) > 4 * 1024 * 1024:
            raise Phase10RExecutionError("replay request exceeds the helper input bound")
        try:
            process.stdin.write(request.decode("utf-8"))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise Phase10RExecutionError("replay-feature helper input failed") from error
        features: list[dict[str, Any]] = []
        while True:
            line = process.stdout.readline()
            if not line:
                error = process.stderr.read() if process.stderr is not None else ""
                raise Phase10RExecutionError(
                    f"replay-feature helper exited before completion: {error[-2_048:]}"
                )
            if len(line.encode("utf-8")) > MAX_HELPER_LINE_BYTES:
                raise Phase10RExecutionError("replay-feature helper emitted an oversized line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase10RExecutionError(
                    "replay-feature helper emitted invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise Phase10RExecutionError("replay-feature helper emitted a non-object")
            event = value.get("event")
            if event == "position":
                if value.get("positionIndex") != len(features):
                    raise Phase10RExecutionError("replay-feature helper position order changed")
                features.append(value)
            elif event == "done":
                if value.get("positionCount") != len(features):
                    raise Phase10RExecutionError("replay-feature helper count is inconsistent")
                return features
            else:
                raise Phase10RExecutionError("replay-feature helper emitted an unknown event")


def _outcome_wdl(outcome: str, side_to_move: str) -> int | None:
    if outcome == "unknown":
        return None
    if outcome == "draw":
        return 1
    if outcome not in {"black_win", "white_win"} or side_to_move not in {"black", "white"}:
        raise Phase10RExecutionError(f"unknown outcome or side-to-move: {outcome}/{side_to_move}")
    return 2 if outcome.removesuffix("_win") == side_to_move else 0


def _example_row(
    row: Mapping[str, Any], payload: Mapping[str, Any], feature: Mapping[str, Any]
) -> dict[str, Any]:
    index = int(row["position_index"])
    moves = payload["usi_moves"]
    legal = feature.get("legalMoves")
    if not isinstance(legal, list) or any(not isinstance(move, str) for move in legal):
        raise Phase10RExecutionError("Rust replay helper returned an invalid legal move list")
    played = moves[index] if index < len(moves) else None
    if feature.get("positionIndex") != index:
        raise Phase10RExecutionError("Rust replay helper position index disagrees with identity")
    if not legal and index != len(moves):
        raise Phase10RExecutionError(
            f"empty legal-root set occurs before the replay end at {row['source_game_id']}:{index}"
        )
    if played is not None and played not in legal:
        raise Phase10RExecutionError("replay played move is outside the Rust legal-root set")
    helper_sfen = feature.get("sfen")
    if helper_sfen != row["canonical_sfen"]:
        raise Phase10RExecutionError(
            f"Rust replay SFEN disagrees with scanned identity at {row['source_game_id']}:{index}"
        )
    history = feature.get("history")
    if not isinstance(history, Mapping):
        raise Phase10RExecutionError("Rust replay helper did not return history facts")
    history_value = HistoryFacts(
        available=bool(history.get("available", False)),
        repetition_count=int(history.get("repetitionCount", 1)),
        continuous_check_by_us=bool(history.get("continuousCheckByUs", False)),
        continuous_check_by_them=bool(history.get("continuousCheckByThem", False)),
    )
    try:
        history_value.validate()
    except ValueError as error:
        raise Phase10RExecutionError("Rust replay helper returned invalid history facts") from error
    outcome = str(payload["outcome"])
    wdl = _outcome_wdl(outcome, str(row["side_to_move"]))
    raw_targets = {
        "outcome": outcome,
        "terminal": payload.get("terminal"),
        "canonical_position_id": row["canonical_position_id"],
        "history_id": row["history_id"],
        "transposition_key": row["transposition_key"],
        "source_game_id": row["source_game_id"],
        "game_id": row["game_id"],
        "position_index": index,
        "record_sha256": row["record_sha256"],
        "archive_member_path": row["archive_member_path"],
        "archive_member_sha256": row["archive_member_sha256"],
        "parser_version": row["parser_version"],
        "normalization_version": row["normalization_version"],
        "terminal_position": not legal,
        "legal_mask_runtime": "open-shogi-core/phase10r-replay-features/v1",
    }
    result: dict[str, Any] = {
        "schema": TRAINING_EXAMPLE_SCHEMA,
        "sfen": row["canonical_sfen"],
        "source": row["source_id"],
        "artifact_id": row["artifact_id"],
        "record_id": f"{row['record_id']}:{index}",
        "split": row["split"],
        "weight": SOURCE_WEIGHTS.get(str(row["source_id"]), 0.0),
        "legal_moves": legal if legal else None,
        "played_move": played,
        "wdl": wdl,
        "wdl_mask": wdl is not None,
        "uncertainty_mask": wdl is not None,
        "history": {
            "available": history_value.available,
            "repetitionCount": history_value.repetition_count,
            "continuousCheckByUs": history_value.continuous_check_by_us,
            "continuousCheckByThem": history_value.continuous_check_by_them,
        },
        "raw_targets": raw_targets,
    }
    Phase10RExample.from_mapping(result).validate()
    return result


def _iter_rows(
    root: Path,
    connection: sqlite3.Connection,
    helper: _ReplayFeatureProcess,
    *,
    splits: Sequence[str],
) -> Iterator[dict[str, Any]]:
    data_root = _data_root(root)
    games_by_artifact = _effective_game_ids(connection, splits=splits)
    current_artifact: str | None = None
    current_game: str | None = None
    payloads: dict[str, dict[str, Any]] = {}
    features: list[dict[str, Any]] = []
    seen_artifacts: set[str] = set()
    for row in _effective_rows(connection, splits=splits):
        artifact_id = str(row["artifact_id"])
        if artifact_id != current_artifact:
            current_artifact = artifact_id
            seen_artifacts.add(artifact_id)
            games = games_by_artifact.get(artifact_id)
            if not games:
                raise Phase10RExecutionError(f"missing effective-game coverage: {artifact_id}")
            payloads = _proofs_and_payloads(data_root, artifact_id, games)
            current_game = None
            features = []
        source_game = str(row["source_game_id"])
        if source_game != current_game:
            current_game = source_game
            payload = payloads.get(source_game)
            if payload is None:
                raise Phase10RExecutionError(f"missing replay payload: {source_game}")
            features = helper.game(str(payload["initial_sfen"]), payload["usi_moves"])
            if len(features) != len(payload["usi_moves"]) + 1:
                raise Phase10RExecutionError(f"replay feature count mismatch: {source_game}")
        index = int(row["position_index"])
        if index >= len(features):
            raise Phase10RExecutionError(
                f"position index is outside replay feature sequence: {source_game}"
            )
        yield _example_row(row, payloads[source_game], features[index])
    if seen_artifacts != set(games_by_artifact):
        raise Phase10RExecutionError("effective row/artifact coverage changed during preparation")


def _write_base_files(
    root: Path,
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    resource_check: Callable[[], object],
) -> dict[str, dict[str, int | str]]:
    outputs: dict[str, dict[str, int | str]] = {}
    paths = {
        ("train", source): output_dir / f"base-train-{source}.jsonl" for source in TRAINING_SOURCES
    }
    paths.update(
        {(split, "all"): output_dir / f"base-{split}.jsonl" for split in EVALUATION_SPLITS}
    )
    temporary_paths: dict[tuple[str, str], Path] = {}
    handles: dict[tuple[str, str], Any] = {}
    digests: dict[tuple[str, str], Any] = {}
    counts: dict[tuple[str, str], int] = {}
    sizes: dict[tuple[str, str], int] = {}
    game_counts: dict[tuple[str, str], int] = {}
    try:
        with ExitStack() as stack:
            for key, path in paths.items():
                if path.exists() or path.is_symlink():
                    raise Phase10RExecutionError(f"preparation output already exists: {path}")
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{path.name}.", suffix=".partial", dir=output_dir
                )
                os.close(descriptor)
                temporary_paths[key] = Path(name)
                handles[key] = stack.enter_context(temporary_paths[key].open("wb"))
                digests[key] = hashlib.sha256()
                counts[key] = 0
                sizes[key] = 0
            with _ReplayFeatureProcess(root) as helper:
                for row in _iter_rows(root, connection, helper, splits=PREPARATION_SPLITS):
                    split = str(row["split"])
                    source = str(row["source"])
                    key = (split, source) if split == TRAINING_SPLIT else (split, "all")
                    if split == TRAINING_SPLIT:
                        game_id = str(row["raw_targets"]["game_id"])
                        game_key = (source, game_id)
                        used = game_counts.get(game_key, 0)
                        if used >= MAX_POSITIONS_PER_GAME_PER_EPOCH:
                            continue
                        game_counts[game_key] = used + 1
                    encoded = _json_bytes(row)
                    handles[key].write(encoded)
                    digests[key].update(encoded)
                    counts[key] += 1
                    sizes[key] += len(encoded)
                    if sizes[key] // (1024**3) != (sizes[key] - len(encoded)) // (1024**3):
                        resource_check()
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        for key, path in paths.items():
            os.link(temporary_paths[key], path)
            outputs[path.name] = {
                "sha256": digests[key].hexdigest(),
                "bytes": sizes[key],
                "rows": counts[key],
            }
    finally:
        for key in paths:
            temporary_paths[key].unlink(missing_ok=True)
    return outputs


def _cycle_rows(path: Path) -> Iterator[dict[str, Any]]:
    while True:
        yielded = False
        for row in _read_jsonl(path):
            yielded = True
            yield row
        if not yielded:
            raise Phase10RExecutionError(f"cannot sample an empty source stream: {path}")


def _schedule_source_counts(total: int) -> dict[str, int]:
    return allocate_source_counts(total, SOURCE_WEIGHTS)


def _stream_rows(
    source_paths: Mapping[str, Path], total: int, *, seed: int = PREPARATION_SEED
) -> Iterator[dict[str, Any]]:
    quotas = _schedule_source_counts(total)
    emitted = {source: 0 for source in TRAINING_SOURCES}
    iterators = {source: _cycle_rows(source_paths[source]) for source in TRAINING_SOURCES}
    for stream_index in range(total):
        candidates = [source for source in TRAINING_SOURCES if emitted[source] < quotas[source]]
        source = max(
            candidates,
            key=lambda value: (
                (stream_index + 1) * quotas[value] / total - emitted[value],
                value,
            ),
        )
        row = dict(next(iterators[source]))
        raw_targets = dict(row.get("raw_targets", {}))
        raw_targets["stream_index"] = stream_index
        raw_targets["sampling_seed"] = seed
        row["raw_targets"] = raw_targets
        emitted[source] += 1
        yield row
    if emitted != quotas:
        raise Phase10RExecutionError(f"source stream allocation drifted: {emitted} != {quotas}")


def prepare_scale(root: Path, scale: str) -> dict[str, Any]:
    """Prepare one exact streamed rung from the already-passed v2 scan."""

    if scale not in SCALE_COUNTS:
        raise Phase10RExecutionError(f"unsupported preparation scale: {scale}")
    root = root.resolve()
    data_root = _data_root(root)
    scan = _scan_identity(root, data_root)
    output_dir = data_root / "phase10r-prepared" / scale
    manifest_path = output_dir / "preparation-manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        manifest = _read_json(manifest_path)
        if manifest.get("schema") != PREPARATION_SCHEMA or manifest.get("scale") != scale:
            raise Phase10RExecutionError("existing preparation manifest has incompatible identity")
        return {"status": "passed", "manifest": manifest_path, "details": manifest}
    output_dir.mkdir(parents=True, exist_ok=True)

    def resource_check() -> object:
        return _resource_check(root, data_root)

    resource_before = resource_check()
    connection = sqlite3.connect(f"file:{scan['database']}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        files = _write_base_files(
            root,
            connection,
            output_dir,
            resource_check=resource_check,
        )
        source_paths = {
            source: output_dir / f"base-train-{source}.jsonl" for source in TRAINING_SOURCES
        }
        stream_path = output_dir / "train.jsonl"
        stream_info = _write_jsonl_immutable(
            stream_path,
            _stream_rows(source_paths, SCALE_COUNTS[scale]),
            resource_check=resource_check,
        )
        files[stream_path.name] = stream_info
    finally:
        connection.close()
    resource_after = resource_check()
    configuration_hashes = _configuration_hashes(root)
    manifest_body: dict[str, Any] = {
        "schema": PREPARATION_SCHEMA,
        "status": "passed",
        "scale": scale,
        "streamed_examples": SCALE_COUNTS[scale],
        "seed": PREPARATION_SEED,
        "source_weights": SOURCE_WEIGHTS,
        "source_stream_counts": _schedule_source_counts(SCALE_COUNTS[scale]),
        "max_positions_per_game_per_epoch": MAX_POSITIONS_PER_GAME_PER_EPOCH,
        "split_policy": {
            "training": [TRAINING_SPLIT],
            "evaluation": list(EVALUATION_SPLITS),
            "excluded": ["public_test", "internal_test", "final_holdout"],
        },
        "scan": {
            key: value for key, value in scan.items() if key not in {"database", "completion_path"}
        },
        "configuration_hashes": configuration_hashes,
        "legal_mask_helper": {
            "schema": HELPER_SCHEMA,
            "path": "target/release/examples/phase10r_replay_features",
            "sha256": _sha256_file(root / "target/release/examples/phase10r_replay_features"),
        },
        "files": files,
        "resources": {"before": resource_before, "after": resource_after},
    }
    manifest_body["manifest_sha256"] = _sha256_bytes(_json_bytes(manifest_body))
    _write_immutable_json(manifest_path, manifest_body)
    return {"status": "passed", "manifest": manifest_path, "details": manifest_body}


__all__ = [
    "HELPER_SCHEMA",
    "PREPARATION_SCHEMA",
    "SCALE_COUNTS",
    "Phase10RExecutionError",
    "prepare_scale",
]
