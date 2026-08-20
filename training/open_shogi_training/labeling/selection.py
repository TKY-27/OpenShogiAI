"""Deterministic, leakage-safe Phase 3 position selection."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    DuplicateJsonKeyError,
    FileDigest,
    compact_json_bytes,
    load_json_object,
    read_regular_bytes,
)
from open_shogi_training.labeling.config import SelectionConfig
from open_shogi_training.labeling.schema import (
    LabelSchemaError,
    canonical_state,
    canonical_state_sha256,
    position_id,
)

_POSITION_SCHEMA: Final = "phase3_position/v1"
_MANIFEST_SCHEMA: Final = "phase3_dataset_manifest/v1"
_SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}
_SPLIT_ORDER = ("test", "validation", "train")
_STAGE_ORDER = ("opening", "middlegame", "endgame")
_OUTCOMES = frozenset({"black_win", "white_win", "draw", "unknown"})
SELECTION_SCHEMA: Final = "phase4_teacher_selection/v2"
LEGACY_SELECTION_SCHEMA: Final = "phase4_teacher_selection/v1"
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


class SelectionError(ValueError):
    """Raised when Phase 3 input or deterministic selection is invalid."""


@dataclass(frozen=True, slots=True)
class SelectedPosition:
    position_id: str
    canonical_sfen: str
    canonical_state_sha256: str
    side_to_move: str
    split: str
    game_id: str
    position_index: int
    stage: str
    source_id: str
    outcome: str
    rank_key: str

    def selection_identity(self) -> dict[str, object]:
        return {
            "position_id": self.position_id,
            "canonical_sfen": self.canonical_sfen,
            "canonical_state_sha256": self.canonical_state_sha256,
            "side_to_move": self.side_to_move,
            "split": self.split,
            "game_id": self.game_id,
            "position_index": self.position_index,
            "stage": self.stage,
            "source_id": self.source_id,
            "outcome": self.outcome,
        }

    def legacy_selection_identity(self) -> dict[str, object]:
        return {
            "position_id": self.position_id,
            "canonical_state_sha256": self.canonical_state_sha256,
            "split": self.split,
            "game_id": self.game_id,
            "position_index": self.position_index,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    positions: tuple[SelectedPosition, ...]
    selection_sha256: str
    legacy_selection_sha256: str
    dataset_manifest_sha256: str
    dataset_manifest_size: int
    positions_sha256: str
    positions_size: int
    input_rows: int
    eligible_rows: int
    unique_eligible_states: int
    cross_split_duplicates_excluded: int
    same_priority_duplicates_excluded: int
    per_split: dict[str, int]
    per_stage: dict[str, int]
    per_game: dict[str, int]

    def summary(self) -> dict[str, object]:
        return {
            "schema": SELECTION_SCHEMA,
            "sha256": self.selection_sha256,
            "selected": len(self.positions),
            "input_rows": self.input_rows,
            "eligible_rows": self.eligible_rows,
            "unique_eligible_states": self.unique_eligible_states,
            "cross_split_duplicates_excluded": self.cross_split_duplicates_excluded,
            "same_priority_duplicates_excluded": self.same_priority_duplicates_excluded,
            "per_split": dict(sorted(self.per_split.items())),
            "per_stage": dict(sorted(self.per_stage.items())),
            "per_game": dict(sorted(self.per_game.items())),
            "identity": (
                "ordered selected label-source fields, including exact SFEN, source, and outcome"
            ),
            "canonical_dedup_identity": (
                "canonical SFEN board, side, and hands; move number omitted"
            ),
            "cross_split_priority": list(_SPLIT_ORDER),
        }

    def legacy_summary(self) -> dict[str, object]:
        """Reconstruct the exact v1 summary used by completed immutable evidence."""

        summary = self.summary()
        summary.pop("schema")
        summary.pop("canonical_dedup_identity")
        summary["sha256"] = self.legacy_selection_sha256
        summary["identity"] = "canonical SFEN board, side, and hands; move number omitted"
        return summary


def select_positions(
    positions_path: Path,
    dataset_manifest_path: Path,
    config: SelectionConfig,
) -> SelectionResult:
    """Select at most 10,000 positions after global canonical-state deduplication."""

    try:
        compressed, positions_digest = read_regular_bytes(
            positions_path,
            max_bytes=config.max_input_compressed_bytes,
        )
        manifest, manifest_digest = load_json_object(
            dataset_manifest_path,
            max_bytes=4 * 1024 * 1024,
        )
    except ArtifactError as error:
        raise SelectionError(str(error)) from error
    expected_rows = _verify_dataset_manifest(manifest, positions_path, positions_digest)

    winners: dict[str, SelectedPosition] = {}
    input_rows = 0
    eligible_rows = 0
    cross_split_duplicates = 0
    same_priority_duplicates = 0
    for line_number, raw in enumerate(
        _iter_phase3_rows(compressed, positions_path.name, config),
        start=1,
    ):
        input_rows += 1
        try:
            candidate = _parse_position(raw, config)
        except (LabelSchemaError, SelectionError) as error:
            raise SelectionError(f"invalid position line {line_number}: {error}") from error
        if candidate is None:
            continue
        eligible_rows += 1
        state_key = candidate.canonical_state_sha256
        previous = winners.get(state_key)
        if previous is None:
            winners[state_key] = candidate
            continue
        previous_priority = _SPLIT_PRIORITY[previous.split]
        candidate_priority = _SPLIT_PRIORITY[candidate.split]
        if previous_priority != candidate_priority:
            cross_split_duplicates += 1
            if candidate_priority < previous_priority:
                winners[state_key] = candidate
        else:
            same_priority_duplicates += 1
            if _candidate_order(candidate) < _candidate_order(previous):
                winners[state_key] = candidate

    if not winners:
        raise SelectionError("Phase 3 positions contain no eligible states")
    if input_rows != expected_rows:
        raise SelectionError(
            f"positions row count {input_rows} disagrees with dataset manifest {expected_rows}"
        )
    selected = _round_robin_select(tuple(winners.values()), config)
    if not selected:
        raise SelectionError("selection constraints excluded every eligible state")
    identity_payload = {
        "schema": SELECTION_SCHEMA,
        "positions": [position.selection_identity() for position in selected],
    }
    selection_sha256 = hashlib.sha256(compact_json_bytes(identity_payload)).hexdigest()
    legacy_identity_payload = [position.legacy_selection_identity() for position in selected]
    legacy_selection_sha256 = hashlib.sha256(
        compact_json_bytes(legacy_identity_payload)
    ).hexdigest()
    split_counts = Counter(position.split for position in selected)
    stage_counts = Counter(position.stage for position in selected)
    game_counts = Counter(position.game_id for position in selected)
    return SelectionResult(
        positions=selected,
        selection_sha256=selection_sha256,
        legacy_selection_sha256=legacy_selection_sha256,
        dataset_manifest_sha256=manifest_digest.sha256,
        dataset_manifest_size=manifest_digest.size,
        positions_sha256=positions_digest.sha256,
        positions_size=positions_digest.size,
        input_rows=input_rows,
        eligible_rows=eligible_rows,
        unique_eligible_states=len(winners),
        cross_split_duplicates_excluded=cross_split_duplicates,
        same_priority_duplicates_excluded=same_priority_duplicates,
        per_split={split: split_counts[split] for split in _SPLIT_ORDER},
        per_stage={stage: stage_counts[stage] for stage in _STAGE_ORDER},
        per_game=dict(sorted(game_counts.items())),
    )


def _verify_dataset_manifest(
    manifest: dict[str, Any],
    positions_path: Path,
    positions_digest: FileDigest,
) -> int:
    if manifest.get("schema") != _MANIFEST_SCHEMA:
        raise SelectionError(f"dataset manifest schema must be {_MANIFEST_SCHEMA!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SelectionError("dataset manifest artifacts must be an object")
    artifact = artifacts.get(positions_path.name)
    if not isinstance(artifact, dict):
        raise SelectionError(f"dataset manifest does not bind {positions_path.name}")
    if set(artifact) != {"sha256", "size", "records"}:
        raise SelectionError("dataset manifest position artifact keys are invalid")
    if artifact["sha256"] != positions_digest.sha256 or artifact["size"] != positions_digest.size:
        raise SelectionError("positions artifact does not match the dataset manifest")
    records = artifact["records"]
    if isinstance(records, bool) or not isinstance(records, int) or records < 1:
        raise SelectionError("dataset manifest position record count is invalid")
    return records


def _iter_phase3_rows(
    compressed: bytes,
    name: str,
    config: SelectionConfig,
) -> Iterator[dict[str, Any]]:
    uncompressed_bytes = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as input_file:
            line_number = 0
            while line := input_file.readline(config.max_input_line_bytes + 1):
                line_number += 1
                if line_number > config.max_input_rows:
                    raise SelectionError(f"{name} exceeds {config.max_input_rows} records")
                if len(line) > config.max_input_line_bytes:
                    raise SelectionError(
                        f"{name} line {line_number} exceeds {config.max_input_line_bytes} bytes"
                    )
                uncompressed_bytes += len(line)
                if uncompressed_bytes > config.max_input_uncompressed_bytes:
                    raise SelectionError(
                        f"{name} exceeds {config.max_input_uncompressed_bytes} uncompressed bytes"
                    )
                if not line.endswith(b"\n"):
                    raise SelectionError(f"{name} line {line_number} is not LF terminated")
                try:
                    value = json.loads(line, object_pairs_hook=_unique_json_object)
                except (DuplicateJsonKeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SelectionError(f"invalid JSON on {name} line {line_number}") from error
                if not isinstance(value, dict):
                    raise SelectionError(f"{name} line {line_number} must be an object")
                yield value
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise SelectionError(f"invalid gzip positions artifact: {error}") from error


def _parse_position(raw: dict[str, Any], config: SelectionConfig) -> SelectedPosition | None:
    if set(raw) != _POSITION_KEYS:
        raise SelectionError(
            f"position keys mismatch: missing={sorted(_POSITION_KEYS - set(raw))}, "
            f"unknown={sorted(set(raw) - _POSITION_KEYS)}"
        )
    if raw["schema"] != _POSITION_SCHEMA:
        raise SelectionError(f"position schema must be {_POSITION_SCHEMA!r}")
    game_id = _sha256(raw["gameId"], "gameId")
    if _sha256(raw["canonicalSha256"], "canonicalSha256") != game_id:
        raise SelectionError("canonicalSha256 must equal gameId")
    _sha256(raw["rawSha256"], "rawSha256")
    source_id = _text(raw["sourceId"], "sourceId", 128)
    split = _choice(raw["split"], "split", frozenset(_SPLIT_PRIORITY))
    index = _integer(raw["positionIndex"], "positionIndex", minimum=0)
    sfen = _text(raw["sfen"], "sfen", 2_048)
    state = canonical_state(sfen)
    expected_side = "black" if state.split(" ")[1] == "b" else "white"
    if raw["sideToMove"] != expected_side:
        raise SelectionError("sideToMove disagrees with sfen")
    full_plies = _integer(raw["fullPlies"], "fullPlies", minimum=0)
    remaining = _integer(raw["remainingPlies"], "remainingPlies", minimum=0)
    if index > full_plies or remaining != full_plies - index:
        raise SelectionError("position ply fields are inconsistent")
    if not isinstance(raw["eligible"], bool) or not isinstance(raw["terminalTail"], bool):
        raise SelectionError("eligible and terminalTail must be booleans")
    outcome = _choice(raw["outcome"], "outcome", _OUTCOMES)
    if not raw["eligible"]:
        return None
    move = raw["moveUsi"]
    next_sfen = raw["nextSfen"]
    if not isinstance(move, str) or not move or not isinstance(next_sfen, str) or not next_sfen:
        raise SelectionError("eligible positions require moveUsi and nextSfen")
    if raw["terminalTail"]:
        raise SelectionError("eligible positions cannot be terminal-tail positions")
    identity = position_id(game_id, index)
    rank_key = hashlib.sha256(
        config.seed.encode("utf-8") + b"\x00" + bytes.fromhex(identity)
    ).hexdigest()
    return SelectedPosition(
        position_id=identity,
        canonical_sfen=sfen,
        canonical_state_sha256=canonical_state_sha256(sfen),
        side_to_move=expected_side,
        split=split,
        game_id=game_id,
        position_index=index,
        stage=_stage(index, full_plies, config),
        source_id=source_id,
        outcome=outcome,
        rank_key=rank_key,
    )


def _stage(index: int, full_plies: int, config: SelectionConfig) -> str:
    if full_plies <= 0:
        raise SelectionError("eligible positions require positive fullPlies")
    progress = index * 10_000 // full_plies
    if progress < config.opening_end_basis_points:
        return "opening"
    if progress < config.middlegame_end_basis_points:
        return "middlegame"
    return "endgame"


def _candidate_order(position: SelectedPosition) -> tuple[str, str, int]:
    return position.rank_key, position.game_id, position.position_index


@dataclass(slots=True)
class _Lane:
    games: deque[str]
    candidates: dict[str, deque[SelectedPosition]]

    def next(
        self,
        selected_per_game: Counter[str],
        max_positions_per_game: int,
    ) -> SelectedPosition | None:
        checks = len(self.games)
        for _ in range(checks):
            game_id = self.games.popleft()
            queue = self.candidates[game_id]
            if selected_per_game[game_id] >= max_positions_per_game:
                del self.candidates[game_id]
                continue
            candidate = queue.popleft()
            if queue:
                self.games.append(game_id)
            else:
                del self.candidates[game_id]
            return candidate
        return None


def _round_robin_select(
    positions: tuple[SelectedPosition, ...],
    config: SelectionConfig,
) -> tuple[SelectedPosition, ...]:
    grouped: dict[tuple[str, str], dict[str, list[SelectedPosition]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for position in positions:
        grouped[(position.split, position.stage)][position.game_id].append(position)

    lanes: list[_Lane] = []
    lane_keys = [
        (split, _STAGE_ORDER[(split_index + offset) % len(_STAGE_ORDER)])
        for offset in range(len(_STAGE_ORDER))
        for split_index, split in enumerate(_SPLIT_ORDER)
    ]
    for split, stage in lane_keys:
        per_game = grouped.get((split, stage), {})
        if not per_game:
            continue
        game_order = sorted(
            per_game,
            key=lambda game_id: hashlib.sha256(
                config.seed.encode("utf-8")
                + b"\x00lane\x00"
                + split.encode("ascii")
                + b"\x00"
                + stage.encode("ascii")
                + b"\x00"
                + bytes.fromhex(game_id)
            ).digest(),
        )
        queues = {
            game_id: deque(sorted(per_game[game_id], key=_candidate_order))
            for game_id in game_order
        }
        lanes.append(_Lane(deque(game_order), queues))

    selected: list[SelectedPosition] = []
    per_game_counts: Counter[str] = Counter()
    while len(selected) < config.max_positions:
        made_progress = False
        for lane in lanes:
            candidate = lane.next(per_game_counts, config.max_positions_per_game)
            if candidate is None:
                continue
            selected.append(candidate)
            per_game_counts[candidate.game_id] += 1
            made_progress = True
            if len(selected) >= config.max_positions:
                break
        if not made_progress:
            break
    return tuple(selected)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SelectionError(f"{name} must be a SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise SelectionError(f"{name} must be lowercase hexadecimal") from error
    if value.lower() != value:
        raise SelectionError(f"{name} must be lowercase hexadecimal")
    return value


def _text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SelectionError(f"{name} must be a non-empty bounded string")
    if any(character in value for character in "\r\n\x00"):
        raise SelectionError(f"{name} contains a line delimiter")
    return value


def _choice(value: object, name: str, choices: frozenset[str]) -> str:
    text = _text(value, name, 128)
    if text not in choices:
        raise SelectionError(f"{name} must be one of {sorted(choices)}")
    return text


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionError(f"{name} must be an integer at least {minimum}")
    return value
