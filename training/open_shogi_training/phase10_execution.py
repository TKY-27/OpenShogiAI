"""Bounded Phase 10 data, target, and experiment execution helpers.

This module deliberately lives beside the frozen Phase 10 control validator.  The
validator and every file covered by ``frozen-controls.sha256`` remain unchanged;
this file only adapts the already-approved Phase 3/4 artifacts into the frozen
Phase 10 execution contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from torch import Tensor

from open_shogi_training.data.gzip_jsonl import iter_jsonl_gzip, write_json_atomic
from open_shogi_training.data.opening_v2 import _classify_opening
from open_shogi_training.labeling.schema import (
    canonical_state,
    canonical_state_sha256,
    iter_teacher_labels,
    position_id,
)
from open_shogi_training.models.config import (
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    combined_config_sha256,
    validate_config_compatibility,
)
from open_shogi_training.models.features import input_dimension
from open_shogi_training.models.network import (
    parameter_count,
)

PHASE10_SCHEMA = "open_shogiai_phase10_execution/v1"
MANIFEST_SCHEMA = "open_shogiai_phase10_manifest/v1"
START_POOL_SCHEMA = "open_shogiai_phase10_start_pool/v2"
START_POOL_OVERLAP_SCHEMA = "open_shogiai_phase10_start_pool_overlap/v1"
TARGET_SCHEMA = "open_shogiai_phase10_targets/v1"
SEED = 20260729
ARENA_SEED = 20260821
TEACHER_CLIP_CP = 3_000.0
OUTPUT_SCALE_CP = 1_200.0
MAX_HISTORY_PLIES = 24
MAX_GAME_FRACTION_BASIS_POINTS = 200
MAX_OPENING_GROUP_FRACTION_BASIS_POINTS = 500
START_POOL_MINIMUM_UNIQUE = 50
START_POOL_RESERVE_PER_GROUP = 200
START_POOL_MAXIMUM_PER_GAME_PER_GROUP = 5
START_GROUPS = (
    "general_opening",
    "ibisha",
    "opponent_furibisha",
    "hard_middlegame_endgame",
)


class Phase10ExecutionError(ValueError):
    """Raised when a Phase 10 frozen integrity or execution contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_position_sfen(sfen: str) -> str:
    """Return board, hands, and side-to-move identity without move number."""

    return canonical_state(sfen + " 1") if len(sfen.split(" ")) == 3 else canonical_state(sfen)


def history_group_id(initial_sfen: str, moves: Sequence[str]) -> str:
    """Hash the canonical initial state and the first 24 legal USI moves."""

    state = canonical_position_sfen(initial_sfen)
    first_moves = tuple(moves[:MAX_HISTORY_PLIES])
    digest = hashlib.sha256()
    digest.update(state.encode("utf-8"))
    digest.update(b"\x00")
    for move in first_moves:
        if not isinstance(move, str) or not move or "\x00" in move or "\n" in move:
            raise Phase10ExecutionError("history group contains an invalid USI move")
        digest.update(move.encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _side_to_move(sfen: str) -> str:
    fields = sfen.split(" ")
    if len(fields) not in {3, 4} or fields[1] not in {"b", "w"}:
        raise Phase10ExecutionError("invalid side-to-move in SFEN")
    return "black" if fields[1] == "b" else "white"


def outcome_target(outcome: str, side_to_move: str) -> tuple[float, float]:
    """Encode factual WDL as a side-to-move signed target and mask."""

    if outcome == "draw":
        return 0.0, 1.0
    if outcome == "unknown":
        return 0.0, 0.0
    if outcome not in {"black_win", "white_win"} or side_to_move not in {"black", "white"}:
        raise Phase10ExecutionError("unsupported factual outcome or side")
    winner = "black" if outcome == "black_win" else "white"
    return (1.0 if winner == side_to_move else -1.0), 1.0


@dataclass(frozen=True, slots=True)
class Phase10Example:
    """One canonical row with independently masked compatible targets."""

    position_id: str
    canonical_sfen: str
    split: str
    stage: str
    game_id: str
    position_index: int
    source_id: str
    history_group_id: str
    style_group: str
    wdl_target: float
    wdl_mask: float
    cp_target: float
    cp_mask: float
    cp_value: float | None
    policy_target: float
    policy_mask: float
    mate_sign: float
    mate_distance: float
    mate_mask: float
    ranking_target: float
    ranking_mask: float
    candidate_gap_cp: float | None
    teacher_kind: str
    teacher_value: int
    raw_source_value_mask: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def adapt_phase10_label(
    label: dict[str, Any],
    *,
    history_id: str,
    style_group: str,
    teacher_clip_cp: float = TEACHER_CLIP_CP,
    output_scale_cp: float = OUTPUT_SCALE_CP,
) -> Phase10Example:
    """Adapt one strict Phase 4 label without fabricating mate centipawns."""

    if teacher_clip_cp != TEACHER_CLIP_CP or output_scale_cp != OUTPUT_SCALE_CP:
        raise Phase10ExecutionError("Phase 10 target scales are frozen")
    state = canonical_position_sfen(label["canonical_sfen"])
    side = label["side_to_move"]
    wdl, wdl_mask = outcome_target(label["outcome"], side)
    score = label["score"]
    kind = score["kind"]
    value = score["value"]
    if kind not in {"cp", "mate"} or not isinstance(value, int) or isinstance(value, bool):
        raise Phase10ExecutionError("label score is not a strict cp or mate value")
    cp_target = 0.0
    cp_mask = 0.0
    cp_value: float | None = None
    mate_sign = 0.0
    mate_distance = 0.0
    mate_mask = 0.0
    if kind == "cp":
        clipped = max(-teacher_clip_cp, min(teacher_clip_cp, float(value)))
        cp_value = clipped
        cp_target = clipped / output_scale_cp
        cp_mask = 1.0
        ranking_target = cp_target
        ranking_mask = 1.0
    else:
        if value == 0:
            raise Phase10ExecutionError("mate score must have a non-zero sign")
        mate_sign = 1.0 if value > 0 else -1.0
        mate_distance = float(abs(value))
        mate_mask = 1.0
        ranking_target = mate_sign
        ranking_mask = 0.0

    candidates = label["candidates"]
    gap: float | None = None
    if len(candidates) >= 2:
        first, second = candidates[0]["score"], candidates[1]["score"]
        if first["kind"] == "cp" and second["kind"] == "cp":
            gap = float(first["value"] - second["value"])

    # The approved raw AobaZero ``v`` field is never present in this teacher
    # label, and its source semantics are frozen as unknown.  Keep the mask
    # explicit so a future join cannot silently turn it into a target.
    return Phase10Example(
        position_id=label["position_id"],
        canonical_sfen=state,
        split=label["split"],
        stage=label["stage"],
        game_id=label["game_id"],
        position_index=label["position_index"],
        source_id=label["source_id"],
        history_group_id=history_id,
        style_group=style_group,
        wdl_target=wdl,
        wdl_mask=wdl_mask,
        cp_target=cp_target,
        cp_mask=cp_mask,
        cp_value=cp_value,
        policy_target=float(label["bestmove"] == label["candidates"][0]["pv"][0]),
        policy_mask=1.0,
        mate_sign=mate_sign,
        mate_distance=mate_distance,
        mate_mask=mate_mask,
        ranking_target=ranking_target,
        ranking_mask=ranking_mask,
        candidate_gap_cp=gap,
        teacher_kind=kind,
        teacher_value=value,
        raw_source_value_mask=0.0,
    )


@dataclass(frozen=True, slots=True)
class Phase10LossWeights:
    """Stage-local weights from the frozen curriculum."""

    cp: float = 0.0
    wdl: float = 0.0
    ranking: float = 0.0
    mate_margin: float = 0.0
    policy: float = 0.0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum()
    return values.sum() * 0.0 if count.item() == 0.0 else (values * mask).sum() / count


def phase10_losses(
    value: Tensor,
    policy_logit: Tensor,
    batch: dict[str, Tensor],
    weights: Phase10LossWeights,
    *,
    ranking_margin: float = 0.05,
    mate_margin: float = 0.05,
) -> dict[str, Tensor]:
    """Compute independently masked WDL/cp/ranking/mate/policy losses."""

    if value.ndim != 1 or policy_logit.shape != value.shape:
        raise Phase10ExecutionError("model outputs must be one-dimensional and aligned")
    cp = _masked_mean(
        torch_functional.smooth_l1_loss(value, batch["cp_target"], reduction="none"),
        batch["cp_mask"],
    )
    wdl = _masked_mean(
        torch_functional.mse_loss(value, batch["wdl_target"], reduction="none"),
        batch["wdl_mask"],
    )
    policy = _masked_mean(
        torch_functional.binary_cross_entropy_with_logits(
            policy_logit, batch["policy_target"], reduction="none"
        ),
        batch["policy_mask"],
    )
    mate_sign = batch["mate_sign"]
    mate_mask = batch["mate_mask"]
    mate = _masked_mean(torch_functional.relu(mate_margin - mate_sign * value), mate_mask)

    # Pair only compatible factual scores.  Mate rows participate in the
    # signed-margin term above and in no cp ranking pair.
    pair_count = value.shape[0] // 2
    if pair_count:
        left = value[: 2 * pair_count : 2]
        right = value[1 : 2 * pair_count : 2]
        left_target = batch["ranking_target"][: 2 * pair_count : 2]
        right_target = batch["ranking_target"][1 : 2 * pair_count : 2]
        mask = (
            batch["ranking_mask"][: 2 * pair_count : 2]
            * batch["ranking_mask"][1 : 2 * pair_count : 2]
            * (left_target != right_target).to(torch.float32)
        )
        direction = torch.sign(left_target - right_target)
        ranking = _masked_mean(
            torch_functional.relu(ranking_margin - direction * (left - right)), mask
        )
        ranking_count = mask.sum()
    else:
        ranking = value.sum() * 0.0
        ranking_count = value.new_zeros(())
    total = (
        weights.cp * cp
        + weights.wdl * wdl
        + weights.ranking * ranking
        + weights.mate_margin * mate
        + weights.policy * policy
    )
    values = (total, cp, wdl, ranking, mate, policy)
    if any(not torch.isfinite(item).item() for item in values):
        raise FloatingPointError("Phase 10 loss contains NaN or infinity")
    return {
        "total": total,
        "cp": cp,
        "wdl": wdl,
        "ranking": ranking,
        "ranking_count": ranking_count,
        "mate_margin": mate,
        "policy": policy,
    }


def _style_group(sfen: str, move: str, position_index: int) -> str:
    """Return the legacy exclusive group used only for repair diagnostics."""

    if not isinstance(move, str) or not move:
        return "general_opening" if position_index < 24 else "hard_middlegame_endgame"
    classification = _classify_opening(canonical_position_sfen(sfen), move)
    if classification == "ibisha":
        return "ibisha"
    if classification == "ibisha-vs-furibisha":
        return "opponent_furibisha"
    if position_index < 24 and classification in {"unclassified", "furibisha"}:
        return "general_opening"
    return "hard_middlegame_endgame"


def start_group_memberships(sfen: str, move: str, position_index: int) -> tuple[str, ...]:
    """Return overlapping Arena-control and descriptive style memberships."""

    if (
        not isinstance(position_index, int)
        or isinstance(position_index, bool)
        or position_index < 0
    ):
        raise Phase10ExecutionError("start-pool position index is invalid")
    groups: list[str] = []
    if position_index < MAX_HISTORY_PLIES:
        groups.append("general_opening")
    classification = "unclassified"
    if isinstance(move, str) and move:
        classification = _classify_opening(canonical_position_sfen(sfen), move)
        if classification == "ibisha":
            groups.append("ibisha")
        elif classification == "ibisha-vs-furibisha":
            groups.append("opponent_furibisha")
    if position_index >= MAX_HISTORY_PLIES and classification in {"unclassified", "furibisha"}:
        groups.append("hard_middlegame_endgame")
    return tuple(group for group in START_GROUPS if group in groups)


def _load_positions(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    rows = list(
        iter_jsonl_gzip(
            path,
            max_compressed_bytes=2 * 1024 * 1024 * 1024,
            max_uncompressed_bytes=2 * 1024 * 1024 * 1024,
            max_records=1_000_000,
            max_line_bytes=1024 * 1024,
        )
    )
    if not rows:
        raise Phase10ExecutionError("positions artifact is empty")
    by_pid: dict[str, dict[str, Any]] = {}
    game_moves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("schema") != "phase3_position/v1":
            raise Phase10ExecutionError("positions artifact contains a non-Phase-3 row")
        game_id = row.get("gameId")
        index = row.get("positionIndex")
        sfen = row.get("sfen")
        if not isinstance(game_id, str) or not isinstance(index, int) or index < 0:
            raise Phase10ExecutionError("position row has invalid game identity")
        if not isinstance(sfen, str):
            raise Phase10ExecutionError("position row lacks SFEN")
        pid = position_id(game_id, index)
        if pid in by_pid:
            raise Phase10ExecutionError(f"duplicate position id: {pid}")
        if row.get("canonicalSha256") != game_id:
            raise Phase10ExecutionError("Phase 3 game identity is not canonicalSha256")
        # This validates the full four-field SFEN and its side-to-move field.
        canonical_state(sfen)
        if row.get("sideToMove") != _side_to_move(sfen):
            raise Phase10ExecutionError(f"side-to-move mismatch at {pid}")
        by_pid[pid] = row
        game_moves[game_id].append(row)
    history_by_game: dict[str, str] = {}
    for game_id, game_rows in game_moves.items():
        game_rows.sort(key=lambda row: row["positionIndex"])
        if [row["positionIndex"] for row in game_rows] != list(range(len(game_rows))):
            raise Phase10ExecutionError(f"game {game_id} has a non-contiguous position history")
        split_values = {row.get("split") for row in game_rows}
        source_values = {row.get("sourceId") for row in game_rows}
        if len(split_values) != 1 or len(source_values) != 1:
            raise Phase10ExecutionError(f"game {game_id} crosses a split or source boundary")
        for current, following in pairwise(game_rows):
            if current.get("nextSfen") != following.get("sfen"):
                raise Phase10ExecutionError(f"game {game_id} has a broken SFEN chain")
        history_by_game[game_id] = history_group_id(
            game_rows[0]["sfen"], [row.get("moveUsi", "") for row in game_rows]
        )
    return rows, by_pid, history_by_game


def _priority(split: str) -> int:
    # Public-test rows are not used here; Phase 3 ``test`` is the frozen final
    # holdout reservation and therefore has the next-highest priority.
    return {"test": 1, "validation": 3, "train": 4}.get(split, 2)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase10ExecutionError(f"cannot read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise Phase10ExecutionError(f"JSON artifact {path} must be an object")
    return value


def _rank_start_candidate(group: str, identity: str, position_identity: str) -> bytes:
    digest = hashlib.sha256()
    for value in ("phase10-arena-start-v2", str(ARENA_SEED), group, identity, position_identity):
        digest.update(value.encode("ascii"))
        digest.update(b"\x00")
    return digest.digest()


def _construct_start_pool(
    positions: list[dict[str, Any]],
    history_by_game: dict[str, str],
    *,
    source_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Protect splits, deduplicate canonically, and reserve distinct paired starts."""

    rows_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    splits_by_history: dict[str, set[str]] = defaultdict(set)
    splits_by_canonical: dict[str, set[str]] = defaultdict(set)
    for row in positions:
        game_id = row["gameId"]
        rows_by_game[game_id].append(row)
        splits_by_history[history_by_game[game_id]].add(row["split"])
        splits_by_canonical[canonical_state_sha256(row["sfen"])].add(row["split"])
    for game_rows in rows_by_game.values():
        game_rows.sort(key=lambda row: row["positionIndex"])

    allowed_splits = {"train", "validation", "test"}
    if any(not splits <= allowed_splits for splits in splits_by_history.values()):
        raise Phase10ExecutionError("start-pool source contains an unknown split")
    effective_history_split = {
        history_id: min(splits, key=_priority) for history_id, splits in splits_by_history.items()
    }
    final_holdout_canonical = {
        identity for identity, splits in splits_by_canonical.items() if "test" in splits
    }

    eligible_rows = [
        row
        for row in positions
        if row.get("sourceId") == source_id
        and row.get("split") != "test"
        and row.get("eligible") is True
        and row.get("terminalTail") is False
        and row.get("remainingPlies", 0) > 0
    ]
    protected_rows: list[dict[str, Any]] = []
    for row in eligible_rows:
        history_id = history_by_game[row["gameId"]]
        identity = canonical_state_sha256(row["sfen"])
        if row["split"] != effective_history_split[history_id]:
            continue
        if identity in final_holdout_canonical:
            continue
        protected_rows.append(row)

    canonical_winners: dict[str, dict[str, Any]] = {}
    for row in protected_rows:
        identity = canonical_state_sha256(row["sfen"])
        previous = canonical_winners.get(identity)
        key = (_priority(row["split"]), row["gameId"], row["positionIndex"])
        if previous is None:
            canonical_winners[identity] = row
            continue
        previous_key = (
            _priority(previous["split"]),
            previous["gameId"],
            previous["positionIndex"],
        )
        if key < previous_key:
            canonical_winners[identity] = row

    candidates: dict[str, dict[str, Any]] = {}
    group_identities: dict[str, set[str]] = {group: set() for group in START_GROUPS}
    for identity, row in canonical_winners.items():
        memberships = start_group_memberships(
            row["sfen"], row.get("moveUsi", ""), row["positionIndex"]
        )
        if not memberships:
            raise Phase10ExecutionError("protected start has no Arena group membership")
        pid = position_id(row["gameId"], row["positionIndex"])
        game_rows = rows_by_game[row["gameId"]]
        moves_to_position = [item["moveUsi"] for item in game_rows[: row["positionIndex"]]]
        candidate = {
            "positionId": pid,
            "canonicalStateSha256": identity,
            "sfen": canonical_position_sfen(row["sfen"]),
            "sideToMove": row["sideToMove"],
            "initialSfen": canonical_position_sfen(game_rows[0]["sfen"]),
            "movesToPosition": moves_to_position,
            "sourceMoveUsi": row["moveUsi"],
            "sourceNextSfen": canonical_position_sfen(row["nextSfen"]),
            "sourceArtifactId": "aobazero-no-noise-exact100",
            "sourceId": row["sourceId"],
            "sourceGameSha256": row["gameId"],
            "sourceRawSha256": row["rawSha256"],
            "sourcePositionIndex": row["positionIndex"],
            "sourceSplit": row["split"],
            "historyGroupId": history_by_game[row["gameId"]],
            "eligibleGroups": list(memberships),
            "openingClassification": (
                _classify_opening(canonical_position_sfen(row["sfen"]), row["moveUsi"])
                if row.get("moveUsi")
                else "unclassified"
            ),
        }
        candidates[identity] = candidate
        for group in memberships:
            group_identities[group].add(identity)

    for group in START_GROUPS:
        if len(group_identities[group]) < START_POOL_MINIMUM_UNIQUE:
            raise Phase10ExecutionError(
                f"style start pool {group} has only {len(group_identities[group])} unique positions"
            )

    used_identities: set[str] = set()
    reserved_by_group: dict[str, list[str]] = {group: [] for group in START_GROUPS}
    selected_candidates: dict[str, dict[str, Any]] = {}
    for group in START_GROUPS:
        ranked = sorted(
            group_identities[group],
            key=lambda identity: (
                _rank_start_candidate(group, identity, candidates[identity]["positionId"]),
                candidates[identity]["positionId"],
            ),
        )
        per_game: Counter[str] = Counter()
        for identity in ranked:
            if identity in used_identities:
                continue
            candidate = candidates[identity]
            game_id = candidate["sourceGameSha256"]
            if per_game[game_id] >= START_POOL_MAXIMUM_PER_GAME_PER_GROUP:
                continue
            reserved_by_group[group].append(candidate["positionId"])
            per_game[game_id] += 1
            used_identities.add(identity)
            selected = dict(candidate)
            selected["assignedGroup"] = group
            selected["selectionRank"] = len(reserved_by_group[group])
            selected_candidates[identity] = selected
            if len(reserved_by_group[group]) == START_POOL_RESERVE_PER_GROUP:
                break
        if len(reserved_by_group[group]) != START_POOL_RESERVE_PER_GROUP:
            raise Phase10ExecutionError(
                f"start pool {group} cannot reserve {START_POOL_RESERVE_PER_GROUP} "
                "globally distinct positions under the per-game cap"
            )

    overlap: dict[str, int] = {}
    for index, left in enumerate(START_GROUPS):
        for right in START_GROUPS[index + 1 :]:
            overlap[f"{left}__{right}"] = len(group_identities[left] & group_identities[right])

    legacy_raw: Counter[str] = Counter()
    legacy_unique: dict[str, set[str]] = {group: set() for group in START_GROUPS}
    canonical_prefixes: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in eligible_rows:
        identity = canonical_state_sha256(row["sfen"])
        group = _style_group(row["sfen"], row.get("moveUsi", ""), row["positionIndex"])
        legacy_raw[group] += 1
        legacy_unique[group].add(identity)
        game_rows = rows_by_game[row["gameId"]]
        canonical_prefixes[identity].add(
            tuple(item["moveUsi"] for item in game_rows[: row["positionIndex"]])
        )

    source_distribution: dict[str, Any] = {}
    for group in START_GROUPS:
        group_rows = [
            candidate
            for candidate in selected_candidates.values()
            if candidate["assignedGroup"] == group
        ]
        source_distribution[group] = {
            "positions": len(group_rows),
            "games": len({row["sourceGameSha256"] for row in group_rows}),
            "historyGroups": len({row["historyGroupId"] for row in group_rows}),
            "splits": dict(sorted(Counter(row["sourceSplit"] for row in group_rows).items())),
            "openingClassifications": dict(
                sorted(Counter(row["openingClassification"] for row in group_rows).items())
            ),
        }

    group_summary = {
        group: {
            "predicate": {
                "general_opening": "positionIndex < 24 (umbrella control)",
                "ibisha": "authoritative classifier == ibisha (descriptive subgroup)",
                "opponent_furibisha": (
                    "authoritative classifier == ibisha-vs-furibisha (descriptive subgroup)"
                ),
                "hard_middlegame_endgame": (
                    "positionIndex >= 24 and classifier is unclassified or furibisha "
                    "(exclusive later-position residual)"
                ),
            }[group],
            "uniqueEligible": len(group_identities[group]),
            "reservedDistinct": len(reserved_by_group[group]),
            "positionIds": reserved_by_group[group],
        }
        for group in START_GROUPS
    }
    start_pool = {
        "selection": {
            "seed": ARENA_SEED,
            "membershipSemantics": "overlapping_control_and_descriptive_tags",
            "allocationSemantics": "globally_canonical_distinct_assigned_group",
            "minimumUniquePerGroup": START_POOL_MINIMUM_UNIQUE,
            "reservePairedStartsPerGroup": START_POOL_RESERVE_PER_GROUP,
            "maximumPerSourceGamePerAssignedGroup": START_POOL_MAXIMUM_PER_GAME_PER_GROUP,
            "ranking": (
                "sha256(phase10-arena-start-v2 NUL seed NUL group NUL "
                "canonicalStateSha256 NUL positionId)"
            ),
            "groupAllocationOrder": list(START_GROUPS),
        },
        "counts": {
            "inputRows": len(positions),
            "eligibleNonTestRows": len(eligible_rows),
            "protectedRowsBeforeCanonicalDedup": len(protected_rows),
            "protectedCanonicalCandidates": len(canonical_winners),
            "reservedCanonicalPositions": len(selected_candidates),
            "finalHoldoutCanonicalIdentitiesExcluded": len(final_holdout_canonical),
            "eligibleRowsExcludedByHistoryPriorityOrHoldout": (
                len(eligible_rows) - len(protected_rows)
            ),
        },
        "groups": group_summary,
        "positions": sorted(selected_candidates.values(), key=lambda row: row["positionId"]),
        "sourceDistribution": source_distribution,
    }
    overlap_report = {
        "schema": START_POOL_OVERLAP_SCHEMA,
        "groupSemantics": "overlapping",
        "groupEligibleCanonicalCounts": {
            group: len(group_identities[group]) for group in START_GROUPS
        },
        "exactCanonicalOverlapBetweenGroups": overlap,
        "splitLeakage": {
            "legacyFinalHoldoutCanonicalOverlap": 0,
            "legacyFinalHoldoutHistoryGroupOverlap": 0,
            "publicTestInspections": 0,
            "legacyFinalHoldoutCandidateEvaluations": 0,
            "eligibleCandidatePopulation": {
                "canonicalPositions": len(canonical_winners),
                "trainingCanonicalOverlap": sum(
                    "train" in splits_by_canonical[identity] for identity in canonical_winners
                ),
                "validationCanonicalOverlap": sum(
                    "validation" in splits_by_canonical[identity] for identity in canonical_winners
                ),
                "legacyFinalHoldoutCanonicalOverlap": 0,
            },
            "reservedPopulation": {
                "canonicalPositions": len(selected_candidates),
                "trainingCanonicalOverlap": sum(
                    "train" in splits_by_canonical[identity] for identity in selected_candidates
                ),
                "validationCanonicalOverlap": sum(
                    "validation" in splits_by_canonical[identity]
                    for identity in selected_candidates
                ),
                "legacyFinalHoldoutCanonicalOverlap": 0,
            },
            "reservedByAssignedGroup": {
                group: {
                    "canonicalPositions": sum(
                        row["assignedGroup"] == group for row in selected_candidates.values()
                    ),
                    "trainingCanonicalOverlap": sum(
                        row["assignedGroup"] == group and "train" in splits_by_canonical[identity]
                        for identity, row in selected_candidates.items()
                    ),
                    "validationCanonicalOverlap": sum(
                        row["assignedGroup"] == group
                        and "validation" in splits_by_canonical[identity]
                        for identity, row in selected_candidates.items()
                    ),
                    "legacyFinalHoldoutCanonicalOverlap": 0,
                }
                for group in START_GROUPS
            },
            "externalHoldouts": {
                "tayayan-public-test-family": {
                    "status": "not_authorized_or_acquired",
                    "exactOverlap": None,
                },
                "dlshogi-public-evaluation-test": {
                    "status": "not_authorized_or_acquired",
                    "exactOverlap": None,
                },
            },
        },
        "duplicateDiagnosis": {
            "eligibleRows": len(eligible_rows),
            "eligibleCanonicalPositions": len(
                {canonical_state_sha256(row["sfen"]) for row in eligible_rows}
            ),
            "duplicateRowsAfterCanonicalization": len(eligible_rows)
            - len({canonical_state_sha256(row["sfen"]) for row in eligible_rows}),
            "canonicalPositionsReachedByMultipleMovePrefixes": sum(
                len(prefixes) > 1 for prefixes in canonical_prefixes.values()
            ),
            "legacyExclusiveRawCounts": dict(legacy_raw),
            "legacyExclusiveUniqueCounts": {
                group: len(legacy_unique[group]) for group in START_GROUPS
            },
        },
    }
    return start_pool, overlap_report


def build_phase10_start_pool_manifest(
    *,
    positions_path: Path,
    dataset_manifest_path: Path,
    audit_manifest_path: Path,
    output_path: Path,
    overlap_output_path: Path,
    source_id: str = "aobazero-no-noise",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen, provenance-complete Arena start-pool artifacts."""

    positions, _, history_by_game = _load_positions(positions_path)
    dataset_manifest = _read_json(dataset_manifest_path)
    audit_manifest = _read_json(audit_manifest_path)
    if dataset_manifest.get("schema") != "phase3_dataset_manifest/v1":
        raise Phase10ExecutionError("start-pool dataset manifest schema is invalid")
    if dataset_manifest.get("datasetId") != "aobazero-no-noise-pd-sample100":
        raise Phase10ExecutionError("start-pool dataset is outside the approved exact-100 scope")
    source = dataset_manifest.get("source")
    if not isinstance(source, dict) or source.get("sourceId") != source_id:
        raise Phase10ExecutionError("start-pool dataset source identity is invalid")
    if source.get("machineLearningAllowed") is not True or source.get("license") != "Public Domain":
        raise Phase10ExecutionError("start-pool dataset source is not approved")
    positions_binding = dataset_manifest.get("artifacts", {}).get(positions_path.name)
    if not isinstance(positions_binding, dict):
        raise Phase10ExecutionError("dataset manifest does not bind the positions artifact")
    if (
        positions_binding.get("sha256") != sha256_file(positions_path)
        or positions_binding.get("size") != positions_path.stat().st_size
        or positions_binding.get("records") != len(positions)
    ):
        raise Phase10ExecutionError("positions artifact differs from its approved manifest")
    audited = {
        item.get("artifact_id"): item
        for item in audit_manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    exact100 = audited.get("aobazero-no-noise-exact100")
    if not isinstance(exact100, dict) or exact100.get("status") != "approved":
        raise Phase10ExecutionError("Phase 10A did not approve the exact-100 artifact")

    start_pool, overlap_report = _construct_start_pool(
        positions, history_by_game, source_id=source_id
    )
    manifest = {
        "schema": START_POOL_SCHEMA,
        "controlSchema": PHASE10_SCHEMA,
        "inputs": {
            "positions": {
                "path": positions_path.name,
                "sha256": sha256_file(positions_path),
                "size": positions_path.stat().st_size,
                "rows": len(positions),
            },
            "datasetManifest": {
                "path": dataset_manifest_path.name,
                "sha256": sha256_file(dataset_manifest_path),
                "size": dataset_manifest_path.stat().st_size,
            },
            "phase10aAuditManifest": {
                "path": "artifacts/phase10a/audit-manifest.json",
                "sha256": sha256_file(audit_manifest_path),
                "size": audit_manifest_path.stat().st_size,
            },
        },
        "source": {
            "sourceId": source_id,
            "artifactId": "aobazero-no-noise-exact100",
            "datasetId": dataset_manifest["datasetId"],
            "auditDecision": "approved",
            "license": source["license"],
            "machineLearningAllowed": source["machineLearningAllowed"],
            "redistributable": source.get("redistributable"),
        },
        "canonicalIdentity": (
            "SFEN board, hands, and side-to-move; move number omitted; no mirror or color rotation"
        ),
        "historyIdentity": (
            "sha256(canonical initial SFEN plus NUL-delimited first up-to-24 legal USI moves)"
        ),
        "holdoutPolicy": {
            "priority": [
                "public_test",
                "final_holdout",
                "source_held_out",
                "validation",
                "train",
            ],
            "legacyFinalHoldoutAllowed": False,
            "externalPublicTestAllowed": False,
            "publicTestInspections": 0,
            "legacyFinalHoldoutCandidateEvaluations": 0,
        },
        **start_pool,
    }
    overlap_report = {
        **overlap_report,
        "startPoolSchema": START_POOL_SCHEMA,
        "inputs": manifest["inputs"],
        "source": manifest["source"],
    }
    write_json_atomic(output_path, manifest)
    write_json_atomic(overlap_output_path, overlap_report)
    return manifest, overlap_report


def build_phase10_manifest(
    *,
    positions_path: Path,
    labels_path: Path,
    label_manifest_path: Path,
    dataset_manifest_path: Path,
    output_path: Path,
    source_id: str = "aobazero-no-noise",
) -> dict[str, Any]:
    """Build an immutable protected-history, canonical-position, and start manifest."""

    positions, positions_by_pid, history_by_game = _load_positions(positions_path)
    label_index = _read_json(label_manifest_path)
    dataset_index = _read_json(dataset_manifest_path)
    expected_dataset_sha = sha256_file(dataset_manifest_path)
    expected_labels_sha = sha256_file(labels_path)
    if label_index.get("labelsSha256") not in {None, expected_labels_sha}:
        raise Phase10ExecutionError("label manifest does not bind the labels artifact")
    if dataset_index.get("schema") not in {None, "phase3_dataset_manifest/v1"}:
        raise Phase10ExecutionError("dataset manifest schema is not the frozen Phase 3 schema")

    labels = list(
        iter_teacher_labels(
            labels_path,
            expected_dataset_manifest_sha256=dataset_index.get(
                "manifestSha256", expected_dataset_sha
            ),
            max_records=10_000,
        )
    )
    if len(labels) != int(label_index.get("selectedRecords", len(labels))):
        raise Phase10ExecutionError("label count disagrees with the frozen label manifest")
    label_by_pid = {label["position_id"]: label for label in labels}
    if len(label_by_pid) != len(labels):
        raise Phase10ExecutionError("duplicate label position identity")
    missing = sorted(set(label_by_pid) - set(positions_by_pid))
    if missing:
        raise Phase10ExecutionError(
            f"labels reference positions outside the approved artifact: {missing[0]}"
        )

    style_by_pid: dict[str, str] = {}
    for pid, row in positions_by_pid.items():
        style_by_pid[pid] = _style_group(row["sfen"], row["moveUsi"], row["positionIndex"])

    # Resolve protected history groups before canonical position identity.  A
    # lower-priority copy is removed, never reassigned across a frozen split.
    labels_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        row = positions_by_pid[label["position_id"]]
        labels_by_history[history_by_game[row["gameId"]]].append(label)
    history_effective: dict[str, str] = {}
    history_dropped = 0
    for history_id, group in labels_by_history.items():
        effective = min(group, key=lambda item: (_priority(item["split"]), item["position_id"]))[
            "split"
        ]
        history_effective[history_id] = effective
        history_dropped += sum(item["split"] != effective for item in group)

    protected_labels = [
        label
        for label in labels
        if label["split"]
        == history_effective[history_by_game[positions_by_pid[label["position_id"]]["gameId"]]]
    ]
    labels_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in protected_labels:
        labels_by_canonical[label["canonical_state_sha256"]].append(label)
    canonical_winners: dict[str, dict[str, Any]] = {}
    canonical_dropped = 0
    for identity, group in labels_by_canonical.items():
        winner = min(group, key=lambda item: (_priority(item["split"]), item["position_id"]))
        canonical_winners[identity] = winner
        canonical_dropped += len(group) - 1
    selected_labels = sorted(canonical_winners.values(), key=lambda item: item["position_id"])

    examples: list[Phase10Example] = []
    for label in selected_labels:
        position = positions_by_pid[label["position_id"]]
        if canonical_position_sfen(position["sfen"]) != canonical_position_sfen(
            label["canonical_sfen"]
        ):
            raise Phase10ExecutionError(
                f"position and teacher SFEN disagree: {label['position_id']}"
            )
        if label["source_id"] != source_id or position["sourceId"] != source_id:
            raise Phase10ExecutionError("a non-approved source entered the active population")
        history_id = history_by_game[position["gameId"]]
        examples.append(
            adapt_phase10_label(
                label,
                history_id=history_id,
                style_group=style_by_pid[label["position_id"]],
            )
        )

    split_counts = Counter(example.split for example in examples)
    # Final holdout is reserved.  It is counted in the manifest but cannot
    # appear in a training or start pool.
    training_examples = [
        example for example in examples if example.split in {"train", "validation"}
    ]
    for example in training_examples:
        if example.split == "test":
            raise Phase10ExecutionError("final holdout leaked into active examples")

    start_pool, start_overlap = _construct_start_pool(
        positions, history_by_game, source_id=source_id
    )
    starts_by_group = {
        group: [row for row in start_pool["positions"] if row["assignedGroup"] == group]
        for group in START_GROUPS
    }

    game_counts = Counter(row["gameId"] for row in positions)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "controlSchema": PHASE10_SCHEMA,
        "inputs": {
            "positions": {
                "path": positions_path.name,
                "sha256": sha256_file(positions_path),
                "rows": len(positions),
            },
            "labels": {
                "path": labels_path.name,
                "sha256": expected_labels_sha,
                "rows": len(labels),
            },
            "labelManifest": {
                "path": label_manifest_path.name,
                "sha256": sha256_file(label_manifest_path),
            },
            "datasetManifest": {"path": dataset_manifest_path.name, "sha256": expected_dataset_sha},
        },
        "source": {
            "activeSourceId": source_id,
            "approvedPopulation": "aobazero-no-noise-exact100",
            "rawAobaZeroVMask": 0.0,
        },
        "counts": {
            "games": len(game_counts),
            "positions": len(positions),
            "labelsInput": len(labels),
            "selectedCanonicalRows": len(examples),
            "selectedTrain": split_counts.get("train", 0),
            "selectedValidation": split_counts.get("validation", 0),
            "selectedFinalHoldout": split_counts.get("test", 0),
            "protectedHistoryDropped": history_dropped,
            "canonicalDuplicateDropped": canonical_dropped,
            "nonTestStartRows": start_pool["counts"]["eligibleNonTestRows"],
        },
        "splits": {
            "priority": ["public_test", "final_holdout", "source_held_out", "validation", "train"],
            "protectedBeforeExtraction": True,
            "historyGroupFormula": (
                "sha256(canonical_initial_sfen + NUL + "
                + "first_up_to_24_legal_usi_moves_with_NUL)"
            ),
            "canonicalIdentity": "canonical_sfen_board_hands_side_to_move_without_move_number",
            "gameGroups": {game: history_by_game[game] for game in sorted(history_by_game)},
            "crossSplitCanonicalRowsAfterPriority": 0,
            "publicTestInspections": 0,
            "legacyFinalHoldoutInspections": 0,
        },
        "styleCounts": {
            group: start_pool["groups"][group]["uniqueEligible"] for group in START_GROUPS
        },
        "startPoolSelection": start_pool["selection"],
        "startPoolOverlap": start_overlap["exactCanonicalOverlapBetweenGroups"],
        "startPools": starts_by_group,
        "targetSemantics": {
            "valuePerspective": "side_to_move",
            "wdl": "factual_wdl_probability_signed_target_with_mask",
            "cp": "approved_apery_only_huber_clipped_3000_normalized_1200",
            "mate": "signed_mate_margin_and_order_only_no_cp_substitution",
            "rawSourceValueMask": 0.0,
        },
    }
    write_json_atomic(output_path, manifest)
    return manifest


def load_phase10_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != MANIFEST_SCHEMA:
        raise Phase10ExecutionError("unsupported Phase 10 manifest schema")
    counts = value.get("counts")
    if not isinstance(counts, dict) or value.get("splits", {}).get("publicTestInspections") != 0:
        raise Phase10ExecutionError("Phase 10 manifest violates holdout controls")
    for group in START_GROUPS:
        starts = value.get("startPools", {}).get(group)
        if not isinstance(starts, list) or len(starts) < 50:
            raise Phase10ExecutionError(f"manifest start pool is short: {group}")
    return value


def load_phase10_start_pool_manifest(path: Path) -> dict[str, Any]:
    """Validate the frozen standalone start-pool manifest without executing Arena."""

    value = _read_json(path)
    if value.get("schema") != START_POOL_SCHEMA:
        raise Phase10ExecutionError("unsupported Phase 10 start-pool schema")
    selection = value.get("selection")
    groups = value.get("groups")
    positions = value.get("positions")
    holdout = value.get("holdoutPolicy")
    if not all(isinstance(item, dict) for item in (selection, groups, holdout)):
        raise Phase10ExecutionError("start-pool control objects are invalid")
    if not isinstance(positions, list):
        raise Phase10ExecutionError("start-pool positions must be a list")
    if selection.get("membershipSemantics") != "overlapping_control_and_descriptive_tags":
        raise Phase10ExecutionError("start-pool membership semantics changed")
    if selection.get("seed") != ARENA_SEED:
        raise Phase10ExecutionError("start-pool selection seed changed")
    if (
        holdout.get("publicTestInspections") != 0
        or holdout.get("legacyFinalHoldoutCandidateEvaluations") != 0
    ):
        raise Phase10ExecutionError("start-pool manifest violates holdout controls")

    position_ids: set[str] = set()
    canonical_ids: set[str] = set()
    assigned_counts: Counter[str] = Counter()
    for row in positions:
        if not isinstance(row, dict):
            raise Phase10ExecutionError("start-pool position is invalid")
        pid = row.get("positionId")
        identity = row.get("canonicalStateSha256")
        assigned = row.get("assignedGroup")
        memberships = row.get("eligibleGroups")
        if not isinstance(pid, str) or not isinstance(identity, str):
            raise Phase10ExecutionError("start-pool position identity is invalid")
        if pid in position_ids or identity in canonical_ids:
            raise Phase10ExecutionError("start-pool contains an exact or canonical duplicate")
        if assigned not in START_GROUPS or not isinstance(memberships, list):
            raise Phase10ExecutionError("start-pool position group is invalid")
        if assigned not in memberships or any(group not in START_GROUPS for group in memberships):
            raise Phase10ExecutionError("assigned start group is absent from memberships")
        moves_to_position = row.get("movesToPosition")
        source_index = row.get("sourcePositionIndex")
        if not isinstance(moves_to_position, list) or len(moves_to_position) != source_index:
            raise Phase10ExecutionError("start-pool position lacks complete move history")
        sfen = row.get("sfen")
        if not isinstance(sfen, str):
            raise Phase10ExecutionError("start-pool position SFEN is invalid")
        canonical_state(sfen + " 1")
        if canonical_state_sha256(sfen + " 1") != identity:
            raise Phase10ExecutionError("start-pool canonical identity does not match SFEN")
        if row.get("sideToMove") != _side_to_move(sfen):
            raise Phase10ExecutionError("start-pool side-to-move does not match SFEN")
        position_ids.add(pid)
        canonical_ids.add(identity)
        assigned_counts[assigned] += 1

    for group in START_GROUPS:
        detail = groups.get(group)
        if not isinstance(detail, dict):
            raise Phase10ExecutionError(f"start-pool group is missing: {group}")
        if detail.get("uniqueEligible", 0) < START_POOL_MINIMUM_UNIQUE:
            raise Phase10ExecutionError(f"start-pool group is short: {group}")
        if detail.get("reservedDistinct") != START_POOL_RESERVE_PER_GROUP:
            raise Phase10ExecutionError(f"start-pool reserve changed: {group}")
        if assigned_counts[group] != START_POOL_RESERVE_PER_GROUP:
            raise Phase10ExecutionError(f"start-pool assigned count changed: {group}")
    return value


def phase10_feature_config(variant: str) -> FeatureConfig:
    if variant == "pure-v1-2x128-control":
        return FeatureConfig(
            name="phase10-base-v1",
            board_planes=True,
            hand_counts=True,
            side_to_move=True,
            king_coordinates=True,
            pseudo_attacks=False,
        )
    if variant == "pure-v1-3x256-attacks":
        return FeatureConfig(
            name="phase10-attacks-v1",
            board_planes=True,
            hand_counts=True,
            side_to_move=True,
            king_coordinates=True,
            pseudo_attacks=True,
        )
    raise Phase10ExecutionError(f"unknown frozen Phase 10 variant: {variant}")


def phase10_model_config(variant: str) -> ModelConfig:
    if variant == "pure-v1-2x128-control":
        return ModelConfig(
            name=variant,
            hidden_layers=2,
            hidden_dim=128,
            activation="relu",
            dropout=0.10,
            output_scale_cp=OUTPUT_SCALE_CP,
            auxiliary_policy_head=True,
        )
    if variant == "pure-v1-3x256-attacks":
        return ModelConfig(
            name=variant,
            hidden_layers=3,
            hidden_dim=256,
            activation="relu",
            dropout=0.10,
            output_scale_cp=OUTPUT_SCALE_CP,
            auxiliary_policy_head=True,
        )
    raise Phase10ExecutionError(f"unknown frozen Phase 10 variant: {variant}")


def phase10_training_config(*, epochs: int = 20, batch_size: int = 512) -> TrainingConfig:
    return TrainingConfig(
        name="phase10-frozen-execution",
        teacher_loss_weight=1.0,
        game_result_loss_weight=0.2,
        policy_agreement_loss_weight=0.05,
        ranking_loss_weight=0.25,
        ranking_margin=0.05,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=min(batch_size, 512),
        epochs=epochs,
        sample_ratio=1.0,
        stage_ratios=(0.34, 0.33, 0.33),
        stage_boundaries_basis_points=(3400, 6700),
        expected_teacher_labels=10_000,
        teacher_clip_cp=TEACHER_CLIP_CP,
        teacher_normalization_cp=OUTPUT_SCALE_CP,
        seed=SEED,
        device="auto",
        deterministic=True,
        quantization="float32",
        gradient_clip_norm=5.0,
        checkpoint_every_epochs=1,
    )


def validate_frozen_variant(variant: str) -> dict[str, Any]:
    feature = phase10_feature_config(variant)
    model = phase10_model_config(variant)
    training = phase10_training_config()
    validate_config_compatibility(model, training, feature)
    counts = parameter_count(input_dimension(feature), model)
    expected = {
        "pure-v1-2x128-control": {"input": 2287, "training": 309634, "exported": 309505},
        "pure-v1-3x256-attacks": {"input": 2449, "training": 759298, "exported": 759041},
    }[variant]
    observed = {
        "input": input_dimension(feature),
        **{key: counts[key] for key in ("trainingTotal", "exportedTotal")},
    }
    normalized = {
        "input": observed["input"],
        "training": observed["trainingTotal"],
        "exported": observed["exportedTotal"],
    }
    if normalized != expected:
        raise Phase10ExecutionError(
            f"variant parameter identity mismatch: {variant}: {normalized} != {expected}"
        )
    return {
        "variant": variant,
        "feature": feature.as_dict(),
        "model": model.as_dict(),
        "parameters": normalized,
        "configSha256": combined_config_sha256(feature, model, training),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build or structurally verify the frozen start pool without executing Phase 10."""

    parser = argparse.ArgumentParser(prog="python -m open_shogi_training.phase10_execution")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-start-pool")
    build.add_argument("--positions", required=True, type=Path)
    build.add_argument("--dataset-manifest", required=True, type=Path)
    build.add_argument("--audit-manifest", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--overlap-output", required=True, type=Path)
    verify = commands.add_parser("verify-start-pool")
    verify.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "build-start-pool":
        manifest, overlap = build_phase10_start_pool_manifest(
            positions_path=arguments.positions,
            dataset_manifest_path=arguments.dataset_manifest,
            audit_manifest_path=arguments.audit_manifest,
            output_path=arguments.output,
            overlap_output_path=arguments.overlap_output,
        )
        result = {
            "schema": START_POOL_SCHEMA,
            "positions": len(manifest["positions"]),
            "groups": {
                group: manifest["groups"][group]["uniqueEligible"] for group in START_GROUPS
            },
            "overlapSchema": overlap["schema"],
        }
    else:
        manifest = load_phase10_start_pool_manifest(arguments.manifest)
        result = {
            "schema": manifest["schema"],
            "positions": len(manifest["positions"]),
            "status": "valid",
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ARENA_SEED",
    "MANIFEST_SCHEMA",
    "START_GROUPS",
    "START_POOL_OVERLAP_SCHEMA",
    "START_POOL_SCHEMA",
    "Phase10Example",
    "Phase10ExecutionError",
    "Phase10LossWeights",
    "adapt_phase10_label",
    "build_phase10_manifest",
    "build_phase10_start_pool_manifest",
    "canonical_position_sfen",
    "history_group_id",
    "load_phase10_manifest",
    "load_phase10_start_pool_manifest",
    "main",
    "outcome_target",
    "phase10_feature_config",
    "phase10_losses",
    "phase10_model_config",
    "phase10_training_config",
    "start_group_memberships",
    "validate_frozen_variant",
]


if __name__ == "__main__":
    raise SystemExit(main())
