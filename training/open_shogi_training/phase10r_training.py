"""Frozen Phase 10R training backend.

This module is deliberately narrower than the older value_v0 trainer.  It owns only the
two Phase 10R candidates, keeps targets source-local, and refuses to manufacture masks,
scores, legal moves, or dataset identity.  Large-scale execution is intentionally not
exposed here; the bounded runner is the validation surface before the first 1M rung.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import resource
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import nn

from open_shogi_training.checkpoint_safety import (
    CheckpointSafetyError,
    deserialize,
    read_verified_bytes,
    verify_receipt,
    write_receipt,
)
from open_shogi_training.phase10r import encode_move
from open_shogi_training.phase10r_model import (
    MAX_NON_MATE_CP,
    VARIANT_PAIR,
    VARIANT_PRIMARY,
    HistoryFacts,
    _encode_features,
    expected_parameter_count,
    parse_sfen,
    serialize_osaval02,
    tensor_specs,
)

PHASE10R_TRAINING_SCHEMA: Final = "open_shogiai_phase10r_training_backend/v1"
CHECKPOINT_SCHEMA: Final = "open_shogiai_phase10r_training_checkpoint/v1"
APPROVED_SOURCE_LANES: Final = frozenset(
    {"aobazero", "wcsc", "denryu", "openshogiai_apery_teacher"}
)
TRAINABLE_VARIANTS: Final = (VARIANT_PAIR, VARIANT_PRIMARY)
MAX_BOUNDED_EXAMPLES: Final = 1_000_000
DEFAULT_SEED: Final = 20_260_729
DEFAULT_RSS_TARGET_BYTES: Final = 16 * 1024**3
DEFAULT_RSS_WARNING_BYTES: Final = 14 * 1024**3
DEFAULT_MINIMUM_FREE_BYTES: Final = 150 * 1024**3
_HEX = frozenset("0123456789abcdef")


class Phase10RTrainingError(ValueError):
    """Raised when a Phase 10R backend contract is not proven."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _wdl_from_outcome(outcome: str, side_to_move: str) -> int:
    if outcome == "draw":
        return 1
    winner = outcome.removesuffix("_win")
    if winner not in {"black", "white"} or side_to_move not in {"black", "white"}:
        raise Phase10RTrainingError("unknown factual outcome or side-to-move perspective")
    return 2 if winner == side_to_move else 0


def transform_score_cp(score_cp: float) -> float:
    """Apply the frozen Apery score transform without crossing the mate namespace."""

    if not _finite(score_cp) or abs(float(score_cp)) > MAX_NON_MATE_CP:
        raise Phase10RTrainingError("score target is outside the non-mate namespace")
    clipped = max(-3_000.0, min(3_000.0, float(score_cp)))
    return math.copysign(math.log1p(abs(clipped)) / math.log1p(3_000.0), clipped)


def _history_from_mapping(value: object) -> HistoryFacts:
    if value is None:
        return HistoryFacts()
    if not isinstance(value, Mapping):
        raise Phase10RTrainingError("history facts must be an object")
    history = HistoryFacts(
        available=bool(value.get("available", False)),
        repetition_count=int(value.get("repetition_count", value.get("repetitionCount", 1))),
        continuous_check_by_us=bool(
            value.get("continuous_check_by_us", value.get("continuousCheckByUs", False))
        ),
        continuous_check_by_them=bool(
            value.get("continuous_check_by_them", value.get("continuousCheckByThem", False))
        ),
    )
    try:
        history.validate()
    except ValueError as error:
        raise Phase10RTrainingError(str(error)) from error
    return history


@dataclass(frozen=True, slots=True)
class Phase10RExample:
    """One immutable source-preserving training row.

    ``None`` means unavailable.  A mask is never inferred from a numeric placeholder, and
    score/mate/ranking fields retain their source metadata in ``raw_targets``.
    """

    sfen: str
    source: str
    artifact_id: str
    record_id: str
    split: str
    weight: float = 1.0
    legal_moves: tuple[str, ...] | None = None
    played_move: str | None = None
    wdl: int | None = None
    wdl_mask: bool = False
    score_cp: float | None = None
    score_mask: bool = False
    mate_kind: int | None = None
    mate_distance: float | None = None
    mate_mask: bool = False
    ranking_scores: tuple[float, ...] | None = None
    ranking_mask: bool = False
    uncertainty_mask: bool = False
    teacher_identity: str | None = None
    history: HistoryFacts = field(default_factory=HistoryFacts)
    raw_targets: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> Phase10RExample:
        if row.get("schema") not in {None, "phase10r_training_example/v1"}:
            raise Phase10RTrainingError("unknown training-example schema")
        sfen = row.get("sfen", row.get("position", {}).get("canonical_sfen"))
        source = row.get("source", row.get("source_id", row.get("sourceId")))
        artifact_id = row.get("artifact_id", row.get("artifactId", ""))
        record_id = row.get("record_id", row.get("recordId", ""))
        split = row.get("split")
        if not all(
            isinstance(value, str) and value
            for value in (sfen, source, artifact_id, record_id, split)
        ):
            raise Phase10RTrainingError("training row is missing stable identity fields")
        legal = row.get("legal_moves", row.get("legalMoves"))
        legal_moves = None if legal is None else tuple(legal)
        played_move = row.get("played_move", row.get("playedMove"))
        wdl = row.get("wdl")
        if wdl is None and row.get("outcome") is not None:
            position = parse_sfen(sfen)
            wdl = _wdl_from_outcome(
                str(row["outcome"]), "black" if position.side_to_move == 0 else "white"
            )
        score = row.get("score_cp", row.get("scoreCp"))
        mate_kind = row.get("mate_kind", row.get("mateKind"))
        mate_distance = row.get("mate_distance", row.get("mateDistance"))
        ranking = row.get("ranking_scores", row.get("rankingScores"))
        ranking_scores = None if ranking is None else tuple(float(value) for value in ranking)
        raw_targets_value = row.get("raw_targets", row.get("rawTargets", {}))
        if not isinstance(raw_targets_value, Mapping):
            raise Phase10RTrainingError("raw target metadata must be an object")
        teacher_identity = row.get("teacher_identity", row.get("teacherIdentity"))
        if teacher_identity is None:
            teacher_identity = raw_targets_value.get(
                "teacher_id", raw_targets_value.get("teacherId")
            )
        return cls(
            sfen=sfen,
            source=source,
            artifact_id=artifact_id,
            record_id=record_id,
            split=split,
            weight=float(row.get("weight", 1.0)),
            legal_moves=legal_moves,
            played_move=played_move,
            wdl=None if wdl is None else int(wdl),
            wdl_mask=bool(row.get("wdl_mask", row.get("wdlMask", wdl is not None))),
            score_cp=None if score is None else float(score),
            score_mask=bool(row.get("score_mask", row.get("scoreMask", score is not None))),
            mate_kind=None if mate_kind is None else int(mate_kind),
            mate_distance=None if mate_distance is None else float(mate_distance),
            mate_mask=bool(row.get("mate_mask", row.get("mateMask", mate_kind is not None))),
            ranking_scores=ranking_scores,
            ranking_mask=bool(
                row.get("ranking_mask", row.get("rankingMask", ranking_scores is not None))
            ),
            uncertainty_mask=bool(row.get("uncertainty_mask", row.get("uncertaintyMask", False))),
            teacher_identity=None if teacher_identity is None else str(teacher_identity),
            history=_history_from_mapping(row.get("history")),
            raw_targets=dict(raw_targets_value),
        )

    @property
    def policy_mask(self) -> bool:
        return self.legal_moves is not None and self.played_move is not None

    def validate(self) -> None:
        if self.source not in APPROVED_SOURCE_LANES:
            raise Phase10RTrainingError(f"source lane is not approved: {self.source}")
        if self.split not in {
            "train",
            "validation",
            "final_holdout",
            "source_held_out",
            "public_test",
            "internal_test",
        }:
            raise Phase10RTrainingError(f"unknown protected split: {self.split}")
        try:
            position = parse_sfen(self.sfen)
        except (TypeError, ValueError) as error:
            raise Phase10RTrainingError(f"invalid canonical SFEN: {error}") from error
        if not self.artifact_id or not self.record_id:
            raise Phase10RTrainingError("training row identity is incomplete")
        if not _finite(self.weight) or self.weight <= 0.0:
            raise Phase10RTrainingError("training weight must be finite and positive")
        self.history.validate()
        if self.legal_moves is not None:
            if not self.legal_moves or len(set(self.legal_moves)) != len(self.legal_moves):
                raise Phase10RTrainingError("legal move mask is missing or contains duplicates")
            try:
                indices = [encode_move(move) for move in self.legal_moves]
            except (TypeError, ValueError) as error:
                raise Phase10RTrainingError(f"legal move codec failure: {error}") from error
            if len(set(indices)) != len(indices):
                raise Phase10RTrainingError("legal move codec is not unique")
        elif self.played_move is not None:
            raise Phase10RTrainingError("played move requires an explicit legal move mask")
        if self.policy_mask and self.played_move not in self.legal_moves:
            raise Phase10RTrainingError("played move is outside the legal move mask")
        if self.ranking_mask:
            if self.source != "openshogiai_apery_teacher":
                raise Phase10RTrainingError("ranking target is enabled for a non-Apery source")
            if (
                self.ranking_scores is None
                or self.legal_moves is None
                or len(self.ranking_scores) != len(self.legal_moves)
                or not self.ranking_scores
                or any(not _finite(value) for value in self.ranking_scores)
            ):
                raise Phase10RTrainingError(
                    "ranking target is missing or mismatched with legal moves"
                )
            if not self.teacher_identity:
                raise Phase10RTrainingError("ranking target lacks an exact teacher identity")
        elif self.ranking_scores is not None:
            raise Phase10RTrainingError("unmasked ranking target must remain unavailable")
        if self.wdl_mask and self.wdl not in {0, 1, 2}:
            raise Phase10RTrainingError("WDL target must be loss, draw, or win")
        if not self.wdl_mask and self.wdl is not None:
            raise Phase10RTrainingError("unmasked WDL target must remain unavailable")
        if self.score_mask:
            if self.score_cp is None or not _finite(self.score_cp):
                raise Phase10RTrainingError("masked score target is missing or non-finite")
            if self.source != "openshogiai_apery_teacher":
                raise Phase10RTrainingError("score target is enabled for a non-Apery source")
            transform_score_cp(self.score_cp)
        elif self.score_cp is not None:
            raise Phase10RTrainingError("unmasked score target must remain unavailable")
        if self.mate_mask:
            if (
                self.mate_kind not in {0, 1, 2}
                or self.mate_distance is None
                or not _finite(self.mate_distance)
            ):
                raise Phase10RTrainingError("masked mate target is incomplete")
            if self.mate_distance <= 0.0:
                raise Phase10RTrainingError("mate distance must be positive")
        elif self.mate_kind is not None or self.mate_distance is not None:
            raise Phase10RTrainingError("unmasked mate target must remain unavailable")
        if self.uncertainty_mask:
            if not (self.score_mask or self.wdl_mask):
                raise Phase10RTrainingError(
                    "uncertainty target requires a matching score or WDL target"
                )
            if self.source not in APPROVED_SOURCE_LANES:
                raise Phase10RTrainingError("uncertainty target source is not approved")
        if position.side_to_move not in {0, 1}:
            raise Phase10RTrainingError("side-to-move is invalid")


@dataclass(frozen=True, slots=True)
class DeviceReceipt:
    requested: str
    selected: str
    fallback: bool
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def select_device(
    requested: Literal["auto", "cpu", "mps"] = "auto", *, allow_cpu_fallback: bool = True
) -> DeviceReceipt:
    """Select MPS when requested/available and record an explicit CPU fallback."""

    if requested not in {"auto", "cpu", "mps"}:
        raise Phase10RTrainingError(f"unsupported device request: {requested}")
    if requested == "cpu":
        return DeviceReceipt(requested, "cpu", False, None)
    available = bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    if available:
        return DeviceReceipt(requested, "mps", False, None)
    if requested == "mps" and not allow_cpu_fallback:
        raise Phase10RTrainingError("MPS was requested but is unavailable and fallback is disabled")
    return DeviceReceipt(
        requested, "cpu", True, "MPS unavailable; selected deterministic CPU fallback"
    )


def resource_snapshot(
    data_root: Path, *, minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
) -> dict[str, int | bool]:
    """Observe RSS and free disk without hiding a hard resource failure."""

    data_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_root)
    unit = (
        1
        if os.name == "posix" and hasattr(resource, "getrusage") and os.uname().sysname == "Darwin"
        else 1024
    )
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit)
    return {
        "peak_rss_bytes": peak_rss,
        "free_disk_bytes": int(usage.free),
        "minimum_free_bytes": minimum_free_bytes,
        "disk_passed": usage.free >= minimum_free_bytes,
    }


class Phase10RModel(nn.Module):
    """PyTorch implementation whose parameter names and shapes are OSAVAL02 exact."""

    def __init__(self, variant_id: str, *, seed: int = DEFAULT_SEED) -> None:
        super().__init__()
        if variant_id not in TRAINABLE_VARIANTS:
            raise Phase10RTrainingError(
                "only the two frozen trainable Phase10R variants are allowed"
            )
        self.variant_id = variant_id
        trunk_input = 48 if variant_id == VARIANT_PAIR else 56
        value_outputs = 8 if variant_id == VARIANT_PAIR else 9
        self.king_piece_embeddings = nn.Parameter(torch.empty(367_416, 4))
        self.king_hand_embeddings = nn.Parameter(torch.empty(43_092, 4))
        self.pair_hash_embeddings = nn.Parameter(torch.empty(65_536, 8))
        if variant_id == VARIANT_PRIMARY:
            self.triple_hash_embeddings = nn.Parameter(torch.empty(32_768, 8))
        self.scalar_projection = nn.Linear(64, 32)
        self.trunk = nn.ModuleList([nn.Linear(trunk_input, 128), nn.Linear(128, 128)])
        self.value_heads = nn.Linear(128, value_outputs)
        self.policy = _PolicyParameters()
        self._initialize(seed)
        actual_count = sum(parameter.numel() for parameter in self.parameters())
        expected_count = expected_parameter_count(variant_id)
        if actual_count != expected_count:
            raise Phase10RTrainingError(
                f"model parameter count mismatch: {actual_count} != {expected_count}"
            )

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for parameter in self.parameters():
            values = torch.empty_like(parameter, device="cpu")
            values.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.data.copy_(values)
        self.policy.log_temperature.data.fill_(0.0)

    def freeze_policy_embedding(self, frozen: bool = True) -> None:
        self.policy.move_embeddings.requires_grad_(not frozen)

    def _reduce(
        self, table: torch.Tensor, rows: Sequence[tuple[int, int]], device: torch.device
    ) -> torch.Tensor:
        if not rows:
            return torch.zeros(table.shape[1], dtype=table.dtype, device=device)
        indices = torch.tensor([row for row, _ in rows], dtype=torch.long, device=device)
        signs = torch.tensor(
            [sign for _, sign in rows], dtype=table.dtype, device=device
        ).unsqueeze(1)
        return (table.index_select(0, indices) * signs).sum(dim=0) / len(rows)

    def forward_example(self, example: Phase10RExample) -> dict[str, torch.Tensor]:
        example.validate()
        if example.legal_moves is None:
            move_rows: list[tuple[str, int]] = []
        else:
            move_rows = [(move, encode_move(move)) for move in example.legal_moves]
        position = parse_sfen(example.sfen)
        features = _encode_features(position, move_rows, example.history, self.variant_id)
        device = self.king_piece_embeddings.device
        king_piece = self._reduce(self.king_piece_embeddings, features["king_piece"], device)
        king_hand = self._reduce(self.king_hand_embeddings, features["king_hand"], device)
        pair = self._reduce(self.pair_hash_embeddings, features["pair"], device)
        scalars = torch.tensor(features["scalars"], dtype=torch.float32, device=device).unsqueeze(0)
        projected = self.scalar_projection(scalars).squeeze(0)
        trunk_values = [king_piece, king_hand, pair]
        if self.variant_id == VARIANT_PRIMARY:
            trunk_values.append(
                self._reduce(self.triple_hash_embeddings, features["triple"], device)
            )
        hidden = torch.cat([*trunk_values, projected], dim=0)
        hidden = torch.relu(self.trunk[0](hidden))
        hidden = torch.relu(self.trunk[1](hidden))
        values = self.value_heads(hidden)
        context = torch.mv(self.policy.context_weight, hidden) + self.policy.context_bias
        if example.legal_moves:
            indices = torch.tensor(
                [encode_move(move) for move in example.legal_moves], dtype=torch.long, device=device
            )
            embeddings = self.policy.move_embeddings.index_select(0, indices)
            logits = torch.mv(embeddings + self.policy.move_offset, context) * torch.exp(
                torch.clamp(self.policy.log_temperature, -4.0, 4.0)
            )
        else:
            indices = torch.empty(0, dtype=torch.long, device=device)
            logits = torch.empty(0, dtype=torch.float32, device=device)
        return {"values": values, "policy_logits": logits, "policy_indices": indices}

    def export_tensors(self) -> dict[str, np.ndarray]:
        expected = {spec.name for spec in tensor_specs(self.variant_id)}
        state = self.state_dict()
        if set(state) != expected:
            raise Phase10RTrainingError(
                "model state does not match frozen tensor set: "
                f"missing={sorted(expected - set(state))}, "
                f"extra={sorted(set(state) - expected)}"
            )
        return {
            name: tensor.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
            for name, tensor in state.items()
        }


class _PolicyParameters(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.move_embeddings = nn.Parameter(torch.empty(13_689, 16))
        self.context_weight = nn.Parameter(torch.empty(16, 128))
        self.context_bias = nn.Parameter(torch.empty(16))
        self.move_offset = nn.Parameter(torch.empty(16))
        self.log_temperature = nn.Parameter(torch.empty(1))


def _value_outputs(model: Phase10RModel, values: torch.Tensor) -> dict[str, torch.Tensor]:
    if model.variant_id == VARIANT_PRIMARY:
        return {
            "wdl_logits": values[:3],
            "score": values[3],
            "mate_logits": values[4:7],
            "mate_distance": values[7],
            "uncertainty": values[8],
        }
    return {
        "wdl_logits": values[:3],
        "score": torch.tensor(float("nan"), device=values.device),
        "mate_logits": values[3:6],
        "mate_distance": values[6],
        "uncertainty": values[7],
    }


def loss_for_examples(
    model: Phase10RModel, examples: Sequence[Phase10RExample]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute only masked losses and return per-head finite diagnostics."""

    if not examples:
        raise Phase10RTrainingError("cannot compute loss for an empty batch")
    totals: dict[str, torch.Tensor] = {}
    weights: dict[str, float] = {}
    for example in examples:
        output = model.forward_example(example)
        values = _value_outputs(model, output["values"])
        weight = torch.tensor(example.weight, dtype=torch.float32, device=output["values"].device)
        if example.policy_mask:
            target_index = example.legal_moves.index(example.played_move)
            target = torch.tensor([target_index], dtype=torch.long, device=output["values"].device)
            policy_loss = torch.nn.functional.cross_entropy(
                output["policy_logits"].unsqueeze(0), target
            )
            totals["policy"] = (
                totals.get("policy", torch.tensor(0.0, device=weight.device)) + weight * policy_loss
            )
            weights["policy"] = weights.get("policy", 0.0) + example.weight
        if example.ranking_mask:
            teacher_scores = torch.tensor(
                example.ranking_scores, dtype=torch.float32, device=output["values"].device
            )
            teacher_distribution = torch.softmax(teacher_scores, dim=0)
            ranking_loss = -torch.sum(
                teacher_distribution
                * torch.nn.functional.log_softmax(output["policy_logits"], dim=0)
            )
            totals["ranking"] = (
                totals.get("ranking", torch.tensor(0.0, device=weight.device))
                + weight * ranking_loss
            )
            weights["ranking"] = weights.get("ranking", 0.0) + example.weight
        if example.wdl_mask:
            target = torch.tensor([example.wdl], dtype=torch.long, device=output["values"].device)
            wdl_loss = torch.nn.functional.cross_entropy(values["wdl_logits"].unsqueeze(0), target)
            totals["wdl"] = (
                totals.get("wdl", torch.tensor(0.0, device=weight.device)) + weight * wdl_loss
            )
            weights["wdl"] = weights.get("wdl", 0.0) + example.weight
        if example.uncertainty_mask and example.score_mask:
            if model.variant_id == VARIANT_PAIR:
                raise Phase10RTrainingError(
                    "score uncertainty target is incompatible with the pair variant"
                )
            score_target = torch.tensor(
                transform_score_cp(example.score_cp), dtype=torch.float32, device=weight.device
            )
            log_variance = torch.clamp(values["uncertainty"], -20.0, 20.0)
            gaussian_nll = 0.5 * (
                torch.exp(-log_variance) * (values["score"] - score_target) ** 2 + log_variance
            )
            totals["uncertainty_score"] = (
                totals.get("uncertainty_score", torch.tensor(0.0, device=weight.device))
                + weight * gaussian_nll
            )
            weights["uncertainty_score"] = weights.get("uncertainty_score", 0.0) + example.weight
        if example.uncertainty_mask and example.wdl_mask:
            probabilities = torch.softmax(values["wdl_logits"], dim=0)
            target = torch.nn.functional.one_hot(
                torch.tensor(example.wdl, dtype=torch.long, device=weight.device), num_classes=3
            ).to(dtype=probabilities.dtype)
            brier_loss = torch.mean((probabilities - target) ** 2)
            totals["uncertainty_wdl"] = (
                totals.get("uncertainty_wdl", torch.tensor(0.0, device=weight.device))
                + weight * brier_loss
            )
            weights["uncertainty_wdl"] = weights.get("uncertainty_wdl", 0.0) + example.weight
        if example.score_mask:
            if model.variant_id == VARIANT_PAIR:
                raise Phase10RTrainingError("score target is incompatible with the pair variant")
            target = torch.tensor(
                transform_score_cp(example.score_cp), dtype=torch.float32, device=weight.device
            )
            score_loss = torch.nn.functional.huber_loss(values["score"], target)
            totals["score"] = (
                totals.get("score", torch.tensor(0.0, device=weight.device)) + weight * score_loss
            )
            weights["score"] = weights.get("score", 0.0) + example.weight
        if example.mate_mask:
            kind = torch.tensor([example.mate_kind], dtype=torch.long, device=weight.device)
            kind_loss = torch.nn.functional.cross_entropy(values["mate_logits"].unsqueeze(0), kind)
            distance = torch.tensor(
                math.copysign(
                    math.log1p(example.mate_distance), 1.0 if example.mate_kind == 2 else -1.0
                ),
                dtype=torch.float32,
                device=weight.device,
            )
            distance_loss = torch.nn.functional.huber_loss(values["mate_distance"], distance)
            totals["mate"] = totals.get(
                "mate", torch.tensor(0.0, device=weight.device)
            ) + weight * (kind_loss + distance_loss)
            weights["mate"] = weights.get("mate", 0.0) + example.weight
    if not totals:
        raise Phase10RTrainingError("batch contains no active target masks")
    normalized = [totals[name] / weights[name] for name in sorted(totals)]
    loss = torch.stack(normalized).sum()
    if not bool(torch.isfinite(loss).item()):
        raise Phase10RTrainingError("loss is NaN or Inf")
    return loss, {name: float(value.detach().cpu()) for name, value in sorted(totals.items())} | {
        f"{name}_weight": value for name, value in sorted(weights.items())
    }


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    variant_id: str = VARIANT_PRIMARY
    seed: int = DEFAULT_SEED
    requested_device: Literal["auto", "cpu", "mps"] = "auto"
    allow_cpu_fallback: bool = True
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_steps: int = 4
    batch_size: int = 4
    checkpoint_interval_steps: int = 10_000
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
    rss_target_bytes: int = DEFAULT_RSS_TARGET_BYTES
    rss_warning_bytes: int = DEFAULT_RSS_WARNING_BYTES
    enforce_disk: bool = True

    def validate(self) -> None:
        if self.variant_id not in TRAINABLE_VARIANTS:
            raise Phase10RTrainingError("training variant is not an eligible frozen candidate")
        if self.max_steps <= 0 or self.batch_size <= 0 or self.checkpoint_interval_steps <= 0:
            raise Phase10RTrainingError("training step and batch limits must be positive")
        if not _finite(self.learning_rate) or self.learning_rate <= 0.0:
            raise Phase10RTrainingError("learning rate must be finite and positive")
        if not _finite(self.weight_decay) or self.weight_decay < 0.0:
            raise Phase10RTrainingError("weight decay must be finite and non-negative")
        if self.minimum_free_bytes < 0 or self.rss_target_bytes <= 0 or self.rss_warning_bytes <= 0:
            raise Phase10RTrainingError("resource limits are invalid")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    write_receipt(path, error=Phase10RTrainingError)


def _checkpoint_payload(
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    config: TrainingConfig,
    manifest_sha256: str,
    step: int,
    order: Sequence[int],
    cursor: int,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "variant_id": model.variant_id,
        "manifest_sha256": manifest_sha256,
        "seed": config.seed,
        "step": step,
        "sampler": {"order": list(order), "cursor": cursor},
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "metrics": dict(metrics),
    }


def _load_verified_checkpoint(path: Path) -> Any:
    """Verify the save-time receipt, then deserialize exactly those bytes.

    The file is read once and hashed in memory, so the bytes torch.load sees
    are the bytes the receipt vouches for even if the path is re-pointed
    afterwards.
    """

    try:
        declared = verify_receipt(path, error=Phase10RTrainingError)
        payload_bytes = read_verified_bytes(
            path,
            declared,
            error=Phase10RTrainingError,
            mismatch_message="checkpoint digest mismatches its receipt",
        )
        return deserialize(payload_bytes, error=CheckpointSafetyError)
    except CheckpointSafetyError as error:
        raise Phase10RTrainingError(str(error)) from error


def _load_checkpoint(
    path: Path,
    *,
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    manifest_sha256: str,
    config: TrainingConfig,
) -> tuple[int, list[int], int, dict[str, float]]:
    if not path.is_file() or path.is_symlink():
        raise Phase10RTrainingError("resume checkpoint must be a regular non-symlink file")
    # The verified load already refuses with a receipt-specific message; a
    # broad rewrap here would hide a digest mismatch behind a generic one.
    payload = _load_verified_checkpoint(path)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise Phase10RTrainingError("resume checkpoint schema is invalid")
    for key, expected in (
        ("variant_id", model.variant_id),
        ("manifest_sha256", manifest_sha256),
        ("seed", config.seed),
    ):
        if payload.get(key) != expected:
            raise Phase10RTrainingError(f"resume checkpoint identity mismatch: {key}")
    sampler = payload.get("sampler")
    if (
        not isinstance(sampler, dict)
        or not isinstance(sampler.get("order"), list)
        or not isinstance(sampler.get("cursor"), int)
    ):
        raise Phase10RTrainingError("resume sampler state is incomplete")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    step = payload.get("step")
    if not isinstance(step, int) or step < 0:
        raise Phase10RTrainingError("resume step is invalid")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise Phase10RTrainingError("resume metrics are invalid")
    return (
        step,
        [int(value) for value in sampler["order"]],
        int(sampler["cursor"]),
        {str(k): float(v) for k, v in metrics.items()},
    )


def run_bounded_training(
    examples: Sequence[Phase10RExample],
    *,
    output_dir: Path,
    manifest_sha256: str,
    config: TrainingConfig | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Run the pre-1M deterministic proof surface and emit a resumable checkpoint."""

    if config is None:
        config = TrainingConfig()
    config.validate()
    if not _is_sha256(manifest_sha256):
        raise Phase10RTrainingError("dataset manifest SHA-256 is required before training")
    if not examples or len(examples) >= MAX_BOUNDED_EXAMPLES:
        raise Phase10RTrainingError(
            "bounded backend accepts a non-empty population below the 1M rung"
        )
    for example in examples:
        example.validate()
    data_root = output_dir.parent
    before = resource_snapshot(data_root, minimum_free_bytes=config.minimum_free_bytes)
    if config.enforce_disk and not before["disk_passed"]:
        raise Phase10RTrainingError("free disk is below the frozen 150 GiB floor")
    _seed_everything(config.seed)
    device_receipt = select_device(
        config.requested_device, allow_cpu_fallback=config.allow_cpu_fallback
    )
    device = torch.device(device_receipt.selected)
    model = Phase10RModel(config.variant_id, seed=config.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    order = list(range(len(examples)))
    random.Random(config.seed).shuffle(order)
    cursor = 0
    step = 0
    metrics: dict[str, float] = {}
    if resume_from is not None:
        step, order, cursor, metrics = _load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            manifest_sha256=manifest_sha256,
            config=config,
        )
        if len(order) != len(examples) or not 0 <= cursor <= len(order):
            raise Phase10RTrainingError("resume sampler does not match the manifest population")
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    best_loss = float("inf")
    while step < config.max_steps:
        if cursor >= len(order):
            cursor = 0
            random.Random(config.seed + step + 1).shuffle(order)
        batch_indices = order[cursor : cursor + config.batch_size]
        cursor += len(batch_indices)
        batch = [examples[index] for index in batch_indices]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_for_examples(model, batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise Phase10RTrainingError("gradient norm is NaN or Inf")
        optimizer.step()
        scheduler.step()
        step += 1
        metrics = {
            **metrics,
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        snapshot = resource_snapshot(data_root, minimum_free_bytes=config.minimum_free_bytes)
        if config.enforce_disk and not snapshot["disk_passed"]:
            _atomic_torch_save(
                last_checkpoint,
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    config=config,
                    manifest_sha256=manifest_sha256,
                    step=step,
                    order=order,
                    cursor=cursor,
                    metrics=metrics,
                ),
            )
            raise Phase10RTrainingError(
                "free disk crossed the frozen floor; checkpoint was written"
            )
        if int(snapshot["peak_rss_bytes"]) > config.rss_target_bytes:
            _atomic_torch_save(
                last_checkpoint,
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    config=config,
                    manifest_sha256=manifest_sha256,
                    step=step,
                    order=order,
                    cursor=cursor,
                    metrics=metrics,
                ),
            )
            raise Phase10RTrainingError(
                "RSS exceeded the frozen training target; checkpoint was written"
            )
        if step % config.checkpoint_interval_steps == 0 or step == config.max_steps:
            payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                config=config,
                manifest_sha256=manifest_sha256,
                step=step,
                order=order,
                cursor=cursor,
                metrics=metrics,
            )
            _atomic_torch_save(last_checkpoint, payload)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                _atomic_torch_save(best_checkpoint, payload)
    after = resource_snapshot(data_root, minimum_free_bytes=config.minimum_free_bytes)
    return {
        "schema": PHASE10R_TRAINING_SCHEMA,
        "status": "passed",
        "variant_id": config.variant_id,
        "manifest_sha256": manifest_sha256,
        "steps": step,
        "examples": len(examples),
        "device": device_receipt.as_dict(),
        "metrics": metrics,
        "resources": {"before": before, "after": after},
        "checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "large_scale_authorized": False,
    }


def export_osaval02_artifact(
    model: Phase10RModel,
    output: Path,
    *,
    quantization: Literal["float32", "int8"],
    dataset_manifest_sha256: str,
    training_run_reference: str,
    git_commit: str,
    metadata_verified: bool = True,
) -> dict[str, Any]:
    """Export an identity-bound candidate and validate it with the Python parser."""

    if not metadata_verified:
        raise Phase10RTrainingError("refusing OSAVAL02 export with unverified metadata")
    if not _is_sha256(dataset_manifest_sha256) or len(git_commit) != 40 or set(git_commit) - _HEX:
        raise Phase10RTrainingError("OSAVAL02 export metadata is incomplete")
    if not training_run_reference or len(training_run_reference) > 64:
        raise Phase10RTrainingError("training run reference is invalid")
    artifact = serialize_osaval02(
        model.export_tensors(),
        variant_id=model.variant_id,
        quantization=quantization,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_run_reference=training_run_reference,
        git_commit=git_commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise Phase10RTrainingError(f"refusing to overwrite OSAVAL02 artifact: {output}")
    try:
        with output.open("xb") as handle:
            handle.write(artifact)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise Phase10RTrainingError(f"refusing to overwrite OSAVAL02 artifact: {output}") from error
    return {
        "schema": "open_shogiai_phase10r_export_validation/v1",
        "status": "passed",
        "path": str(output),
        "sha256": hashlib.sha256(artifact).hexdigest(),
        "bytes": len(artifact),
        "variant_id": model.variant_id,
        "quantization": quantization,
    }


__all__ = [
    "APPROVED_SOURCE_LANES",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_SEED",
    "MAX_BOUNDED_EXAMPLES",
    "PHASE10R_TRAINING_SCHEMA",
    "DeviceReceipt",
    "Phase10RExample",
    "Phase10RModel",
    "Phase10RTrainingError",
    "TrainingConfig",
    "export_osaval02_artifact",
    "loss_for_examples",
    "resource_snapshot",
    "run_bounded_training",
    "select_device",
    "transform_score_cp",
]
