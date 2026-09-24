"""Bounded Phase 10U a1 execution helpers.

The frozen Phase 10U policy module only attests receipts.  This module owns the small amount of
execution wiring needed by the Sunday run: verified label loading, game-lineage splitting,
observed-only MultiPV ranking autograd, and a1 checkpoint/export receipts.  It never opens a
holdout or changes the Phase 10T frozen controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from open_shogi_training.phase10r_model import HistoryFacts, Piece, parse_sfen
from open_shogi_training.phase10t_model import (
    Phase10TExample,
    Phase10TModel,
    _feature_batch,
    predict_examples,
)
from open_shogi_training.phase10t_pure_build import verify_evidence
from open_shogi_training.phase10t_run import (
    LABEL_SCHEMA,
    RUNG_LIMITS,
    _label_manifest_row,
    _load_labels,
    _load_train_rows,
    _pure_build_audit,
    _safe_path,
    _sha256_file,
    _teacher_identity,
    _write_immutable,
)


class Phase10UExecutionError(RuntimeError):
    """Raised when raw Phase 10U evidence cannot be verified."""


_BASE_PIECE = {8: 0, 9: 1, 10: 2, 11: 3, 12: 5, 13: 6}
_PROMOTED = {0: 8, 1: 9, 2: 10, 3: 11, 5: 12, 6: 13}
_PIECE_LETTERS = ("P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R")
_HAND_LETTERS = "PLNSGBR"
_USI_HAND_ORDER = "RBGSNLP"


@dataclass(frozen=True, slots=True)
class ObservedTrainingRow:
    """One root plus the observed teacher successor positions used for ranking."""

    root: Phase10TExample
    candidate_sfens: tuple[str, ...]
    ranking_pairs: tuple[tuple[int, int], ...]
    source: str
    component_id: str
    index: int


@dataclass(frozen=True, slots=True)
class SplitRows:
    train: tuple[ObservedTrainingRow, ...]
    validation: tuple[ObservedTrainingRow, ...]
    calibration: tuple[ObservedTrainingRow, ...]
    component_counts: dict[str, int]


def _square_index(value: str) -> int:
    if len(value) != 2 or value[0] not in "123456789" or value[1] not in "abcdefghi":
        raise Phase10UExecutionError(f"invalid USI square: {value!r}")
    return (ord(value[1]) - ord("a")) * 9 + (9 - int(value[0]))


def _square_name(index: int) -> str:
    rank, column = divmod(index, 9)
    return f"{9 - column}{chr(ord('a') + rank)}"


def _history(value: object) -> HistoryFacts:
    if not isinstance(value, dict):
        return HistoryFacts()
    try:
        return HistoryFacts(
            available=bool(value.get("available", False)),
            repetition_count=int(value.get("repetitionCount", value.get("repetition_count", 1))),
            continuous_check_by_us=bool(
                value.get("continuousCheckByUs", value.get("continuous_check_by_us", False))
            ),
            continuous_check_by_them=bool(
                value.get("continuousCheckByThem", value.get("continuous_check_by_them", False))
            ),
        )
    except (TypeError, ValueError) as error:
        raise Phase10UExecutionError("approved row has invalid history facts") from error


def successor_sfen(sfen: str, move: str) -> str:
    """Apply one already legality-replayed USI move without inventing a legal move."""

    parsed = parse_sfen(sfen)
    board = list(parsed.board)
    hands = [list(parsed.hands[0]), list(parsed.hands[1])]
    side = parsed.side_to_move
    if "*" in move:
        if len(move) != 4 or move[1] != "*":
            raise Phase10UExecutionError(f"invalid drop move: {move!r}")
        kind = "PLNSGBR".index(move[0])
        target = _square_index(move[2:])
        if board[target] is not None or hands[side][kind] <= 0:
            raise Phase10UExecutionError(f"drop move is inconsistent with root: {move!r}")
        hands[side][kind] -= 1
        board[target] = Piece(side, kind, target)
    else:
        if len(move) not in (4, 5) or (len(move) == 5 and move[-1] != "+"):
            raise Phase10UExecutionError(f"invalid board move: {move!r}")
        origin, target = _square_index(move[:2]), _square_index(move[2:4])
        piece = board[origin]
        if (
            piece is None
            or piece.side != side
            or (board[target] is not None and board[target].side == side)
        ):
            raise Phase10UExecutionError(f"board move is inconsistent with root: {move!r}")
        captured = board[target]
        if captured is not None:
            if captured.kind == 7:
                raise Phase10UExecutionError("teacher successor attempts to capture a king")
            hands[side][_BASE_PIECE.get(captured.kind, captured.kind)] += 1
        kind = piece.kind
        if move.endswith("+"):
            if kind not in _PROMOTED:
                raise Phase10UExecutionError(f"invalid promotion move: {move!r}")
            kind = _PROMOTED[kind]
        board[origin] = None
        board[target] = Piece(side, kind, target)

    ranks: list[str] = []
    for rank in range(9):
        empty = 0
        encoded: list[str] = []
        for column in range(9):
            piece = board[rank * 9 + column]
            if piece is None:
                empty += 1
                continue
            if empty:
                encoded.append(str(empty))
                empty = 0
            name = _PIECE_LETTERS[piece.kind]
            encoded.append(name if piece.side == 0 else name.lower())
        if empty:
            encoded.append(str(empty))
        ranks.append("".join(encoded))
    hand_parts: list[str] = []
    for absolute_side, character_case in ((0, str.upper), (1, str.lower)):
        for letter in _USI_HAND_ORDER:
            kind = _HAND_LETTERS.index(letter)
            count = hands[absolute_side][kind]
            if count:
                hand_parts.append((str(count) if count != 1 else "") + character_case(letter))
    hand = "".join(hand_parts) or "-"
    next_side = "w" if side == 0 else "b"
    return f"{'/'.join(ranks)} {next_side} {hand} {parsed.move_number + 1}"


def _component_id(row: dict[str, Any], index: int) -> str:
    raw = row.get("raw_targets")
    if not isinstance(raw, dict):
        raise Phase10UExecutionError(f"approved row {index} lacks raw lineage metadata")
    game = raw.get("game_id") or raw.get("source_game_id")
    transposition = raw.get("transposition_key")
    if (
        not isinstance(game, str)
        or not game
        or not isinstance(transposition, str)
        or not transposition
    ):
        raise Phase10UExecutionError(f"approved row {index} lacks game/transposition lineage")
    return f"{row['source']}\0{game}\0{transposition}"


def load_observed_rows(
    root: Path, labels_path: Path, *, expected_nodes: int = 100_000
) -> list[ObservedTrainingRow]:
    """Verify the label file against the deterministic approved train prefix."""

    labels = _load_labels(root, labels_path)
    approved = _load_train_rows(root, len(labels))
    if len(labels) not in (RUNG_LIMITS["100k"], RUNG_LIMITS["1m"]):
        raise Phase10UExecutionError("Phase 10U supervised rung has an unsupported label count")
    result: list[ObservedTrainingRow] = []
    seen: set[str] = set()
    for index, (label, row) in enumerate(zip(labels, approved, strict=True)):
        if label.get("schema") != LABEL_SCHEMA or label.get("split") != "train":
            raise Phase10UExecutionError(f"label {index} has an invalid schema or split")
        if label.get("sfen") != row.get("sfen") or label.get("source") != row.get("source"):
            raise Phase10UExecutionError(f"label lineage differs at row {index}")
        if label["sfen"] in seen:
            raise Phase10UExecutionError("label lineage contains a duplicate root position")
        seen.add(label["sfen"])
        _label_manifest_row(index, row, label)
        teacher = label.get("teacher")
        if not isinstance(teacher, dict) or teacher.get("nodes") != expected_nodes:
            raise Phase10UExecutionError(f"label {index} has an unexpected teacher node budget")
        score_kind = teacher.get("primary_score_kind")
        score_value = teacher.get("primary_score_value")
        if score_kind not in {"cp", "mate"} or not isinstance(score_value, int):
            raise Phase10UExecutionError(f"label {index} has an invalid primary score")
        observed = label.get("observed_targets")
        candidates = teacher.get("candidates")
        if not isinstance(observed, dict) or not isinstance(candidates, list):
            raise Phase10UExecutionError(f"label {index} lacks observed targets")
        pairs = observed.get("ranking_pairs")
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            for pair in pairs
        ):
            raise Phase10UExecutionError(f"label {index} has invalid observed ranking pairs")
        candidate_sfens = tuple(
            successor_sfen(label["sfen"], candidate["pv"][0]) for candidate in candidates
        )
        pair_tuple = tuple((pair[0], pair[1]) for pair in pairs)
        for winner, loser in pair_tuple:
            if not 0 <= winner < len(candidate_sfens) or not 0 <= loser < len(candidate_sfens):
                raise Phase10UExecutionError(f"label {index} ranks an unobserved candidate")
            if winner == loser:
                raise Phase10UExecutionError(f"label {index} has a self-ranking pair")
        history = _history(row.get("history"))
        cp = float(np.clip(score_value, -3000, 3000)) if score_kind == "cp" else None
        wdl = int(label["wdl"]) if label.get("wdl_mask") and label.get("wdl") in (0, 1, 2) else None
        result.append(
            ObservedTrainingRow(
                root=Phase10TExample(
                    sfen=str(label["sfen"]),
                    cp=cp,
                    wdl=wdl,
                    history=history,
                    source=str(label["source"]),
                ),
                candidate_sfens=candidate_sfens,
                ranking_pairs=pair_tuple,
                source=str(label["source"]),
                component_id=_component_id(row, index),
                index=index,
            )
        )
    return result


def split_by_component(rows: Sequence[ObservedTrainingRow], *, seed: int) -> SplitRows:
    """Assign whole game/transposition components to train, validation, or calibration."""

    components = sorted({row.component_id for row in rows})
    assignments: dict[str, str] = {}
    for component in components:
        digest = hashlib.sha256(f"phase10u-component-v1\0{seed}\0{component}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10
        assignments[component] = (
            "validation" if bucket == 0 else "calibration" if bucket == 1 else "train"
        )
    partitions = {"train": [], "validation": [], "calibration": []}
    for row in rows:
        partitions[assignments[row.component_id]].append(row)
    if not partitions["train"] or not partitions["validation"] or not partitions["calibration"]:
        raise Phase10UExecutionError("component split produced an empty partition")
    counts = {
        name: sum(assignments[row.component_id] == name for row in rows) for name in partitions
    }
    return SplitRows(
        train=tuple(partitions["train"]),
        validation=tuple(partitions["validation"]),
        calibration=tuple(partitions["calibration"]),
        component_counts=counts,
    )


def _feature_arrays(
    examples: Sequence[Phase10TExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        raise Phase10UExecutionError("feature batch cannot be empty")
    return _feature_batch(examples)


def _rank_loss(root_scores: Sequence[float], targets: dict[str, Any]) -> float:
    """Reference observed-only loss used by the autograd implementation and tests."""

    if len(root_scores) != targets["observed_candidate_count"]:
        raise ValueError("ranking predictions must cover only observed candidates")
    pairs = targets["ranking_pairs"]
    if not pairs:
        return 0.0
    values = []
    for winner, loser in pairs:
        difference = root_scores[loser] - root_scores[winner]
        values.append(max(difference, 0.0) + math.log1p(math.exp(-abs(difference))))
    return sum(values) / len(values)


def _model_from_parameters(parameters: Sequence[object], seed: int) -> Phase10TModel:
    model = Phase10TModel(
        table=np.asarray(parameters[0], dtype="<f4"),
        bias=np.asarray(parameters[1], dtype="<f4"),
        hidden_weight=np.asarray(parameters[2], dtype="<f4"),
        hidden_bias=np.asarray(parameters[3], dtype="<f4"),
        head_weight=np.asarray(parameters[4], dtype="<f4"),
        head_bias=np.asarray(parameters[5], dtype="<f4"),
        seed=seed,
    )
    model.validate()
    return model


def train_observed_a1(
    split: SplitRows,
    *,
    seed: int,
    max_passes: int = 2,
    batch_size: int = 128,
    validation_every_steps: int = 250,
    patience: int = 4,
    ranking_weight: float = 0.1,
) -> tuple[Phase10TModel, Phase10TModel, Phase10TModel, dict[str, Any]]:
    """Train a1 with direct cp/WDL plus differentiable observed-successor ranking."""

    if not 0.0 < ranking_weight <= 1.0:
        raise ValueError("ranking weight must be in (0, 1]")
    if not split.train or not split.validation:
        raise ValueError("train and validation partitions are required")
    import torch
    from torch.nn import functional as torch_f

    all_rows = list(split.train) + list(split.validation)
    root_examples = [row.root for row in all_rows]
    child_examples = [
        Phase10TExample(sfen, None, None, HistoryFacts(), row.source)
        for row in all_rows
        for sfen in row.candidate_sfens
    ]
    root_ids, root_mask, root_sides, root_cps = _feature_arrays(root_examples)
    child_ids, child_mask, child_sides, _ = _feature_arrays(child_examples)
    root_count = len(all_rows)
    child_offsets = np.zeros((root_count, 3), dtype=np.int64)
    cursor = 0
    for index, row in enumerate(all_rows):
        for slot in range(len(row.candidate_sfens)):
            child_offsets[index, slot] = cursor
            cursor += 1
        for slot in range(len(row.candidate_sfens), 3):
            child_offsets[index, slot] = -1
    root_cps_np = root_cps.astype(np.float32, copy=False)
    root_cp_mask_np = np.asarray([row.root.cp is not None for row in all_rows], dtype=bool)
    root_wdl_np = np.asarray(
        [0 if row.root.wdl is None else row.root.wdl for row in all_rows], dtype=np.int64
    )
    root_wdl_mask_np = np.asarray([row.root.wdl in (0, 1, 2) for row in all_rows], dtype=bool)
    sources_np = np.asarray([row.source for row in all_rows], dtype=object)
    train_indices = np.arange(len(split.train), dtype=np.int64)
    validation_indices = np.arange(len(split.train), root_count, dtype=np.int64)
    rng = np.random.default_rng(seed)

    tensor_root_ids = torch.from_numpy(root_ids)
    tensor_root_mask = torch.from_numpy(root_mask)
    tensor_root_sides = torch.from_numpy(root_sides)
    tensor_child_ids = torch.from_numpy(child_ids)
    tensor_child_mask = torch.from_numpy(child_mask)
    tensor_child_sides = torch.from_numpy(child_sides)
    tensor_cps = torch.from_numpy(root_cps_np)
    tensor_cp_mask = torch.from_numpy(root_cp_mask_np)
    tensor_wdl = torch.from_numpy(root_wdl_np)
    tensor_wdl_mask = torch.from_numpy(root_wdl_mask_np)
    model = Phase10TModel.random(seed)
    pre_model = Phase10TModel.from_bytes(model.to_bytes())
    parameters = [
        torch.nn.Parameter(torch.from_numpy(value.copy()))
        for value in (
            model.table,
            model.bias,
            model.hidden_weight,
            model.hidden_bias,
            model.head_weight,
            model.head_bias,
        )
    ]
    table, bias, hidden_weight, hidden_bias, head_weight, head_bias = parameters
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    steps_per_pass = math.ceil(len(train_indices) / batch_size)
    total_steps = max(1, max_passes * steps_per_pass)
    validation_history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_model = pre_model
    checks_without_improvement = 0
    started = time.monotonic()

    def forward(ids: Any, mask: Any, sides: Any) -> Any:
        values = bias + (table[ids] * mask[..., None]).sum(dim=2)
        positions = torch.arange(len(sides), dtype=torch.long)
        packed = torch.cat((values[positions, sides], values[positions, 1 - sides]), dim=1)
        hidden = torch.relu(packed @ hidden_weight + hidden_bias)
        return hidden @ head_weight + head_bias

    def batch_loss(selected: np.ndarray) -> Any:
        root_output = forward(
            tensor_root_ids[selected], tensor_root_mask[selected], tensor_root_sides[selected]
        )
        child_global: list[int] = []
        local_slots = np.full((len(selected), 3), -1, dtype=np.int64)
        for local, row_index in enumerate(selected.tolist()):
            for slot in range(3):
                global_index = int(child_offsets[row_index, slot])
                if global_index >= 0:
                    local_slots[local, slot] = len(child_global)
                    child_global.append(global_index)
        child_output = (
            forward(
                tensor_child_ids[np.asarray(child_global, dtype=np.int64)],
                tensor_child_mask[np.asarray(child_global, dtype=np.int64)],
                tensor_child_sides[np.asarray(child_global, dtype=np.int64)],
            )
            if child_global
            else None
        )
        source_names = sources_np[selected]
        groups: list[Any] = []
        for source in sorted(set(source_names.tolist())):
            source_rows = torch.from_numpy(source_names == source)
            active: list[Any] = []
            local_cp = tensor_cp_mask[selected] & source_rows
            local_wdl = tensor_wdl_mask[selected] & source_rows
            if bool(local_cp.any()):
                active.append(
                    torch_f.smooth_l1_loss(
                        root_output[local_cp, 0] / 600.0, tensor_cps[selected][local_cp] / 600.0
                    )
                )
            if bool(local_wdl.any()):
                active.append(
                    torch_f.cross_entropy(
                        root_output[local_wdl, 1:], tensor_wdl[selected][local_wdl]
                    )
                )
            rank_values: list[Any] = []
            for local, row_index in enumerate(selected.tolist()):
                if sources_np[row_index] != source or child_output is None:
                    continue
                for winner, loser in all_rows[row_index].ranking_pairs:
                    left, right = local_slots[local, winner], local_slots[local, loser]
                    if left < 0 or right < 0:
                        raise Phase10UExecutionError(
                            "ranking pair references a missing observed child"
                        )
                    rank_values.append(
                        torch_f.softplus(child_output[left, 0] - child_output[right, 0])
                    )
            if rank_values:
                active.append(ranking_weight * torch.stack(rank_values).mean())
            if active:
                groups.append(torch.stack(active).mean())
        if not groups:
            raise Phase10UExecutionError("batch has no active target mask")
        return torch.stack(groups).mean()

    def evaluate(indices: np.ndarray) -> tuple[float, dict[str, Any]]:
        with torch.no_grad():
            loss = float(batch_loss(indices).cpu())
        current = _model_from_parameters(
            [parameter.detach().cpu().numpy() for parameter in parameters], seed
        )
        outputs = predict_examples(current, [all_rows[index].root for index in indices.tolist()])
        rows = [all_rows[index] for index in indices.tolist()]
        cp_rows = [row for row in rows if row.root.cp is not None]
        cp_mae = (
            float(
                np.mean(
                    [
                        abs(float(output[0]) - float(row.root.cp))
                        for output, row in zip(outputs, rows, strict=True)
                        if row.root.cp is not None
                    ]
                )
            )
            if cp_rows
            else None
        )
        rank_total = 0
        rank_correct = 0
        for row, _output_row in zip(rows, outputs, strict=True):
            child_outputs = predict_examples(
                current,
                [
                    Phase10TExample(sfen, None, None, HistoryFacts(), row.source)
                    for sfen in row.candidate_sfens
                ],
            )
            root_scores = [-float(value[0]) for value in child_outputs]
            for winner, loser in row.ranking_pairs:
                rank_total += 1
                rank_correct += int(root_scores[winner] > root_scores[loser])
        metrics: dict[str, Any] = {
            "loss": loss,
            "examples": len(rows),
            "cp_examples": len(cp_rows),
            "ranking_pairs": rank_total,
            "ranking_accuracy": (rank_correct / rank_total) if rank_total else None,
        }
        if cp_mae is not None:
            metrics["cp_mae"] = cp_mae
        return loss, metrics

    steps = 0
    for _ in range(max_passes):
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start : start + batch_size]
            progress = steps / max(1, total_steps - 1)
            if progress < 0.1:
                learning_rate = 3e-4 * max(progress / 0.1, 1e-3)
            else:
                learning_rate = 3e-5 + (3e-4 - 3e-5) * 0.5 * (
                    1.0 + math.cos(math.pi * min(1.0, (progress - 0.1) / 0.9))
                )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(selected)
            if not bool(torch.isfinite(loss).item()):
                raise Phase10UExecutionError("a1 observed training loss is not finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            if not bool(torch.isfinite(norm).item()):
                raise Phase10UExecutionError("a1 observed gradient norm is not finite")
            optimizer.step()
            steps += 1
            if steps % validation_every_steps == 0 or steps == total_steps:
                current_loss, metrics = evaluate(validation_indices)
                metrics["step"] = steps
                validation_history.append(metrics)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = _model_from_parameters(
                        [parameter.detach().cpu().numpy() for parameter in parameters], seed
                    )
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    break
        if checks_without_improvement >= patience:
            break
    last_model = _model_from_parameters(
        [parameter.detach().cpu().numpy() for parameter in parameters], seed
    )
    if not validation_history:
        current_loss, metrics = evaluate(validation_indices)
        best_loss = current_loss
        best_model = Phase10TModel.from_bytes(last_model.to_bytes())
        validation_history.append({**metrics, "step": steps})
    return (
        pre_model,
        best_model,
        last_model,
        {
            "schema": "open_shogiai_phase10u_a1_training/v1",
            "seed": seed,
            "steps": steps,
            "passes_requested": max_passes,
            "train_examples": len(split.train),
            "validation_examples": len(split.validation),
            "calibration_examples": len(split.calibration),
            "ranking_weight": ranking_weight,
            "ranking_pairs_train": sum(len(row.ranking_pairs) for row in split.train),
            "ranking_pairs_validation": sum(len(row.ranking_pairs) for row in split.validation),
            "component_counts": split.component_counts,
            "validation_history": validation_history,
            "best_validation_loss": best_loss,
            "elapsed_seconds": time.monotonic() - started,
            "random_initialization": True,
            "architecture": "a1-king-relative-128",
            "target_masks": {
                "value_cp_clipped_to": 3000,
                "factual_wdl_only": True,
                "observed_ranking_only": True,
                "singleton_and_ties_contribute_zero_ranking": True,
                "mate_numeric_cp": False,
            },
        },
    )


def fit_cp_calibration(
    model: Phase10TModel, rows: Sequence[ObservedTrainingRow], seed: int
) -> dict[str, Any]:
    examples = [row.root for row in rows if row.root.cp is not None]
    if not examples:
        raise Phase10UExecutionError("calibration partition has no teacher cp rows")
    predictions = predict_examples(model, examples)[:, 0].astype(np.float64)
    targets = np.asarray([float(example.cp) for example in examples], dtype=np.float64)
    if len(examples) >= 2 and float(np.var(predictions)) > 1e-12:
        scale, bias = np.linalg.lstsq(
            np.column_stack((predictions, np.ones(len(predictions)))), targets, rcond=None
        )[0]
    else:
        scale, bias = 1.0, float(np.mean(targets - predictions))
    if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0:
        raise Phase10UExecutionError("calibration produced a non-positive affine scale")
    model.head_weight[:, 0] *= np.float32(scale)
    model.head_bias[0] = np.float32(float(model.head_bias[0]) * scale + bias)
    model.validate()
    after = predict_examples(model, examples)[:, 0].astype(np.float64)
    return {
        "scale": float(scale),
        "bias": float(bias),
        "examples": len(examples),
        "mae_before": float(np.mean(np.abs(predictions - targets))),
        "mae_after": float(np.mean(np.abs(after - targets))),
        "seed": seed,
        "target": "approved_teacher_cp_clipped_to_3000_component_calibration_only",
    }


def _explicit_pure_build_audit(root: Path, audit_path: Path) -> dict[str, Any]:
    if audit_path.is_absolute() or ".." in audit_path.parts:
        raise Phase10UExecutionError("pure audit path must remain repository-relative")
    path = root / audit_path
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        verify_evidence(root, evidence)
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        raise Phase10UExecutionError(f"current pure-only audit is invalid: {error}") from error
    if evidence.get("git_commit") != current_commit:
        raise Phase10UExecutionError("pure-only audit belongs to a different Git commit")
    return {
        "passed": True,
        "actual_wasm_tested": True,
        "audit_path": path.relative_to(root).as_posix(),
        "audit_sha256": _sha256_file(path),
        "audit_git_commit": current_commit,
        "native": evidence["artifacts"]["native"],
        "wasm_raw": evidence["artifacts"]["wasm_raw"],
        "wasm": evidence["artifacts"]["wasm"],
        "runtime_model_sha256": evidence["runtime"]["model_sha256"],
        "runtime_usi_proofs": evidence["runtime"]["usi_proofs"],
    }


def _label_input_ref(root: Path, labels_path: Path) -> tuple[Path, Path]:
    if labels_path.is_absolute():
        try:
            relative = labels_path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise Phase10UExecutionError("label path must remain repository-relative") from error
    else:
        relative = labels_path
    return relative, _safe_path(root, relative)


def train_and_write(
    root: Path,
    labels_path: Path,
    *,
    rung: str,
    seed: int,
    pure_audit: Path | None = None,
) -> dict[str, Any]:
    required = RUNG_LIMITS.get(rung)
    if required not in (100_000, 1_000_000):
        raise Phase10UExecutionError("only the frozen 100k and 1m rungs are supported")
    labels_relative, labels_file = _label_input_ref(root, labels_path)
    rows = load_observed_rows(root, labels_relative)
    if len(rows) != required:
        raise Phase10UExecutionError(f"{rung} requires exactly {required} verified labels")
    split = split_by_component(rows, seed=seed)
    pre, selected, last, stats = train_observed_a1(split, seed=seed)
    calibration = fit_cp_calibration(selected, split.calibration, seed)
    attempt_root = root / "local" / "phase10u-runs" / f"train-{rung}"
    attempt_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path
        for path in attempt_root.iterdir()
        if path.is_dir() and path.name.startswith("attempt-")
    )
    number = max((int(path.name.removeprefix("attempt-")) for path in existing), default=0) + 1
    attempt = attempt_root / f"attempt-{number:04d}"
    attempt.mkdir()
    paths = {
        "pre_stage": attempt / "pre-stage.osat10",
        "last": attempt / "last.osat10",
        "selected": attempt / "selected-a1-king-relative-128.osat10",
    }
    hashes = {
        name: model.write(path)
        for name, (path, model) in {
            "pre_stage": (paths["pre_stage"], pre),
            "last": (paths["last"], last),
            "selected": (paths["selected"], selected),
        }.items()
    }
    audit = (
        _explicit_pure_build_audit(root, pure_audit)
        if pure_audit is not None
        else _pure_build_audit(root)
    )
    if not audit.get("passed"):
        raise Phase10UExecutionError(
            "training finished but the frozen pure-only audit is not valid"
        )
    receipt = {
        "schema": "open_shogiai_phase10u_a1_training_receipt/v1",
        "status": "complete",
        "stage": "supervised_training",
        "rung": rung,
        "variant": "a1-king-relative-128",
        "input_labels": labels_relative.as_posix(),
        "input_labels_sha256": _sha256_file(labels_file),
        "teacher": _teacher_identity(root),
        "random_seed": seed,
        "checkpoints": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "model_path": paths["selected"].relative_to(root).as_posix(),
        "model_sha256": hashes["selected"],
        "calibration": calibration,
        "stats": stats,
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "calibration": len(split.calibration),
            "component_counts": split.component_counts,
        },
        "pure_only_build_audit": audit,
        "integrity": {
            "approved_train_only": True,
            "unique_positions": True,
            "source_provenance": True,
            "game_lineage_split": True,
            "target_semantics": True,
            "native_a1": False,
            "native_wasm_parity": False,
            "pure_runtime": False,
        },
        "format_boundary": {
            "a1": "OSAT10A1-float32",
            "osaval02": "STOP_CLOSED_incompatible_with_frozen_a1",
            "int8": "STOP_CLOSED_not_supported_by_frozen_a1_loader",
        },
        "campaign_counted": True,
        "holdout_access": False,
    }
    receipt_path = attempt / "receipt.json"
    receipt["receipt_path"] = receipt_path.relative_to(root).as_posix()
    receipt["receipt_sha256"] = _write_immutable(receipt_path, receipt)
    return receipt


def main() -> None:
    raise SystemExit("Closed campaign; use current development commands")


if __name__ == "__main__":
    main()


__all__ = [
    "ObservedTrainingRow",
    "Phase10UExecutionError",
    "SplitRows",
    "fit_cp_calibration",
    "load_observed_rows",
    "split_by_component",
    "successor_sfen",
    "train_and_write",
    "train_observed_a1",
]
