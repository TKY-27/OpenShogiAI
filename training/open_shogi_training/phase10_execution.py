"""Bounded Phase 10 data, target, and experiment execution helpers.

This module deliberately lives beside the frozen Phase 10 control validator.  The
validator and every file covered by ``frozen-controls.sha256`` remain unchanged;
this file only adapts the already-approved Phase 3/4 artifacts into the frozen
Phase 10 execution contract.
"""

from __future__ import annotations

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
TARGET_SCHEMA = "open_shogiai_phase10_targets/v1"
SEED = 20260729
ARENA_SEED = 20260821
TEACHER_CLIP_CP = 3_000.0
OUTPUT_SCALE_CP = 1_200.0
MAX_HISTORY_PLIES = 24
MAX_GAME_FRACTION_BASIS_POINTS = 200
MAX_OPENING_GROUP_FRACTION_BASIS_POINTS = 500
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

    eligible_start_rows = [
        row
        for row in positions
        if row.get("sourceId") == source_id
        and row.get("split") != "test"
        and row.get("eligible") is True
        and row.get("terminalTail") is False
        and row.get("remainingPlies", 0) > 0
    ]
    starts_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_start_states: set[str] = set()
    for row in eligible_start_rows:
        identity = canonical_state_sha256(row["sfen"])
        if identity in seen_start_states:
            continue
        seen_start_states.add(identity)
        group = style_by_pid[position_id(row["gameId"], row["positionIndex"])]
        starts_by_group[group].append(
            {
                "positionId": position_id(row["gameId"], row["positionIndex"]),
                "canonicalStateSha256": identity,
                "sfen": canonical_position_sfen(row["sfen"]),
                "gameId": row["gameId"],
                "positionIndex": row["positionIndex"],
                "historyGroupId": history_by_game[row["gameId"]],
                "styleGroup": group,
            }
        )
    for group in START_GROUPS:
        starts_by_group[group].sort(key=lambda item: item["positionId"])
        if len(starts_by_group[group]) < 50:
            raise Phase10ExecutionError(
                f"style start pool {group} has only {len(starts_by_group[group])} unique positions"
            )

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
            "nonTestStartRows": len(eligible_start_rows),
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
            group: sum(
                1
                for row in eligible_start_rows
                if style_by_pid[position_id(row["gameId"], row["positionIndex"])] == group
            )
            for group in START_GROUPS
        },
        "startPools": {group: starts_by_group[group] for group in START_GROUPS},
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


__all__ = [
    "ARENA_SEED",
    "MANIFEST_SCHEMA",
    "START_GROUPS",
    "Phase10Example",
    "Phase10ExecutionError",
    "Phase10LossWeights",
    "adapt_phase10_label",
    "build_phase10_manifest",
    "canonical_position_sfen",
    "history_group_id",
    "load_phase10_manifest",
    "outcome_target",
    "phase10_feature_config",
    "phase10_losses",
    "phase10_model_config",
    "phase10_training_config",
    "validate_frozen_variant",
]
