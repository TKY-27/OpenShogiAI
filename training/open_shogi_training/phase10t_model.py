"""The Phase 10T a1 reference model and its closed artifact format.

This module is intentionally independent from the frozen Phase 10R model code.  It contains
only the random-initialized a1 feature encoder, the two accumulators, and the direct cp/WDL
heads specified by the Phase 10T freeze.  The native evaluator consumes the same little-endian
artifact emitted here.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from open_shogi_training.phase10r_model import HistoryFacts, ParsedPosition, parse_sfen

FEATURE_COUNT: Final = 8_433
ACCUMULATOR_WIDTH: Final = 128
HIDDEN_WIDTH: Final = 32
HEAD_COUNT: Final = 4
MODEL_MAGIC: Final = b"OSAT10A1"
MODEL_VERSION: Final = 1
FEATURE_SCHEMA_VERSION: Final = 1
DEFAULT_SEED: Final = 2_026_0907
MAX_MODEL_BYTES: Final = 64 * 1024 * 1024
MAX_PARAMETER_MAGNITUDE: Final = 1_000_000.0
_HEADER = struct.Struct("<8s6IQf")
_F32 = np.dtype("<f4")

BOARD_FEATURES: Final = 17 * 17 * 28
KING_OFFSET: Final = BOARD_FEATURES
HAND_OFFSET: Final = KING_OFFSET + 81
HISTORY_OFFSET: Final = HAND_OFFSET + 2 * 7 * 18
_DEFAULT_HISTORY = HistoryFacts()


class Phase10TModelError(ValueError):
    """Raised when a Phase 10T artifact is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class Phase10TExample:
    """One approved, non-holdout training example."""

    sfen: str
    cp: float | None
    wdl: int | None
    history: HistoryFacts = _DEFAULT_HISTORY
    source: str = "unknown"


def feature_ids(
    position: ParsedPosition,
    perspective: int,
    history: HistoryFacts = _DEFAULT_HISTORY,
) -> tuple[int, ...]:
    """Return the exact sparse a1 feature IDs in deterministic order."""

    if perspective not in (0, 1):
        raise ValueError("perspective must be Black or White")
    history.validate()
    king = next(
        piece.square
        for piece in position.board
        if piece is not None and piece.kind == 7 and piece.side == perspective
    )
    oriented_king = 80 - king if perspective else king
    king_rank, king_column = divmod(oriented_king, 9)
    result = {KING_OFFSET + oriented_king}
    for piece in position.board:
        if piece is None:
            continue
        oriented = 80 - piece.square if perspective else piece.square
        rank, column = divmod(oriented, 9)
        relative = (rank - king_rank + 8) * 17 + column - king_column + 8
        if not 0 <= relative < 17 * 17:
            raise Phase10TModelError("a1 relative board feature escaped its 17x17 window")
        result.add(relative * 28 + (piece.side ^ perspective) * 14 + piece.kind)
    for side, hand in enumerate(position.hands):
        for kind, count in enumerate(hand):
            if count > 18:
                raise Phase10TModelError("a1 hand count exceeds the frozen feature range")
            for ordinal in range(count):
                result.add(HAND_OFFSET + ((side ^ perspective) * 7 + kind) * 18 + ordinal)
    result.add(HISTORY_OFFSET + int(history.available))
    result.add(HISTORY_OFFSET + 2 + history.repetition_count - 1)
    checks = (history.continuous_check_by_us, history.continuous_check_by_them)
    own_check, other_check = checks if perspective == position.side_to_move else checks[::-1]
    if own_check:
        result.add(HISTORY_OFFSET + 6)
    if other_check:
        result.add(HISTORY_OFFSET + 7)
    if any(index < 0 or index >= FEATURE_COUNT for index in result):
        raise Phase10TModelError("a1 feature ID escaped the frozen 8433-feature schema")
    return tuple(sorted(result))


def sparse_features(
    sfen: str, history: HistoryFacts = _DEFAULT_HISTORY
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    position = parse_sfen(sfen)
    return (
        feature_ids(position, 0, history),
        feature_ids(position, 1, history),
        position.side_to_move,
    )


@dataclass(slots=True)
class Phase10TModel:
    """Random-initialized a1 parameters with direct cp and WDL outputs."""

    table: np.ndarray
    bias: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    head_weight: np.ndarray
    head_bias: np.ndarray
    seed: int

    @classmethod
    def random(cls, seed: int = DEFAULT_SEED) -> Phase10TModel:
        rng = np.random.default_rng(seed)

        def normal(shape: tuple[int, ...]) -> np.ndarray:
            return rng.normal(0.0, 0.01, shape).astype(_F32)

        return cls(
            table=normal((FEATURE_COUNT, ACCUMULATOR_WIDTH)),
            bias=normal((ACCUMULATOR_WIDTH,)),
            hidden_weight=normal((2 * ACCUMULATOR_WIDTH, HIDDEN_WIDTH)),
            hidden_bias=normal((HIDDEN_WIDTH,)),
            head_weight=normal((HIDDEN_WIDTH, HEAD_COUNT)),
            head_bias=normal((HEAD_COUNT,)),
            seed=seed,
        )

    def validate(self) -> None:
        expected = (
            ((FEATURE_COUNT, ACCUMULATOR_WIDTH), self.table),
            ((ACCUMULATOR_WIDTH,), self.bias),
            ((2 * ACCUMULATOR_WIDTH, HIDDEN_WIDTH), self.hidden_weight),
            ((HIDDEN_WIDTH,), self.hidden_bias),
            ((HIDDEN_WIDTH, HEAD_COUNT), self.head_weight),
            ((HEAD_COUNT,), self.head_bias),
        )
        for shape, value in expected:
            if tuple(value.shape) != shape:
                raise Phase10TModelError(f"a1 tensor shape mismatch: {value.shape} != {shape}")
            if value.dtype != _F32:
                raise Phase10TModelError("a1 tensors must be little-endian float32")
            if not np.isfinite(value).all():
                raise Phase10TModelError("a1 tensors contain a non-finite value")
            if np.any(np.abs(value) > MAX_PARAMETER_MAGNITUDE):
                raise Phase10TModelError("a1 tensor magnitude exceeds safe inference bound")

    def _accumulate(self, features: Sequence[int]) -> np.ndarray:
        value = self.bias.copy()
        for index in features:
            value += self.table[index]
        return value

    def evaluate(
        self, sfen: str, history: HistoryFacts = _DEFAULT_HISTORY
    ) -> tuple[float, np.ndarray]:
        first, second, side_to_move = sparse_features(sfen, history)
        accumulators = (self._accumulate(first), self._accumulate(second))
        values = np.maximum(
            np.concatenate((accumulators[side_to_move], accumulators[1 - side_to_move])), 0
        )
        hidden = np.maximum(values @ self.hidden_weight + self.hidden_bias, 0)
        output = hidden @ self.head_weight + self.head_bias
        return float(output[0]), output[1:].astype(_F32, copy=True)

    def payload(self) -> bytes:
        self.validate()
        arrays = (
            self.table,
            self.bias,
            self.hidden_weight,
            self.hidden_bias,
            self.head_weight,
            self.head_bias,
        )
        return b"".join(np.asarray(array, dtype=_F32, order="C").tobytes() for array in arrays)

    def to_bytes(self) -> bytes:
        payload = self.payload()
        header = _HEADER.pack(
            MODEL_MAGIC,
            MODEL_VERSION,
            FEATURE_SCHEMA_VERSION,
            FEATURE_COUNT,
            ACCUMULATOR_WIDTH,
            HIDDEN_WIDTH,
            HEAD_COUNT,
            self.seed,
            1.0,
        )
        return header + payload + hashlib.sha256(payload).digest()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def write(self, path: Path) -> str:
        data = self.to_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> Phase10TModel:
        if len(data) > MAX_MODEL_BYTES or len(data) < _HEADER.size + 32:
            raise Phase10TModelError("a1 artifact size is outside the closed parser bound")
        (
            magic,
            version,
            feature_schema,
            feature_count,
            accumulator_width,
            hidden_width,
            head_count,
            seed,
            scale,
        ) = _HEADER.unpack_from(data)
        if magic != MODEL_MAGIC or version != MODEL_VERSION:
            raise Phase10TModelError("a1 artifact magic or version mismatch")
        if (
            feature_schema != FEATURE_SCHEMA_VERSION
            or feature_count != FEATURE_COUNT
            or accumulator_width != ACCUMULATOR_WIDTH
            or hidden_width != HIDDEN_WIDTH
            or head_count != HEAD_COUNT
            or scale != 1.0
        ):
            raise Phase10TModelError("a1 artifact architecture/schema mismatch")
        payload_end = len(data) - 32
        payload = data[_HEADER.size : payload_end]
        if hashlib.sha256(payload).digest() != data[payload_end:]:
            raise Phase10TModelError("a1 artifact checksum mismatch")
        expected_bytes = (
            FEATURE_COUNT * ACCUMULATOR_WIDTH
            + ACCUMULATOR_WIDTH
            + 2 * ACCUMULATOR_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH * HEAD_COUNT
            + HEAD_COUNT
        ) * 4
        if len(payload) != expected_bytes:
            raise Phase10TModelError("a1 artifact tensor payload has the wrong length")
        cursor = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            size = int(np.prod(shape)) * 4
            value = np.frombuffer(payload[cursor : cursor + size], dtype=_F32).reshape(shape).copy()
            cursor += size
            return value

        model = cls(
            table=take((FEATURE_COUNT, ACCUMULATOR_WIDTH)),
            bias=take((ACCUMULATOR_WIDTH,)),
            hidden_weight=take((2 * ACCUMULATOR_WIDTH, HIDDEN_WIDTH)),
            hidden_bias=take((HIDDEN_WIDTH,)),
            head_weight=take((HIDDEN_WIDTH, HEAD_COUNT)),
            head_bias=take((HEAD_COUNT,)),
            seed=seed,
        )
        model.validate()
        return model

    @classmethod
    def read(cls, path: Path) -> Phase10TModel:
        data = path.read_bytes()
        return cls.from_bytes(data)


def _feature_batch(
    rows: Sequence[Phase10TExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = [sparse_features(row.sfen, row.history) for row in rows]
    width = max(len(part) for item in features for part in item[:2])
    ids = np.zeros((len(rows), 2, width), dtype=np.int64)
    mask = np.zeros((len(rows), 2, width), dtype=np.float32)
    sides = np.empty((len(rows),), dtype=np.int64)
    cps = np.empty((len(rows),), dtype=np.float32)
    for row_index, (row, item) in enumerate(zip(rows, features, strict=True)):
        pair, side = item[:2], item[2]
        sides[row_index] = side
        cps[row_index] = 0.0 if row.cp is None else row.cp if np.isfinite(row.cp) else 0.0
        for perspective in (0, 1):
            values = pair[perspective]
            ids[row_index, perspective, : len(values)] = values
            mask[row_index, perspective, : len(values)] = 1.0
    return ids, mask, sides, cps


def _model_from_parameters(parameters: Sequence[object], seed: int) -> Phase10TModel:
    table, bias, hidden_weight, hidden_bias, head_weight, head_bias = parameters
    model = Phase10TModel(
        table=np.asarray(table, dtype=_F32),
        bias=np.asarray(bias, dtype=_F32),
        hidden_weight=np.asarray(hidden_weight, dtype=_F32),
        hidden_bias=np.asarray(hidden_bias, dtype=_F32),
        head_weight=np.asarray(head_weight, dtype=_F32),
        head_bias=np.asarray(head_bias, dtype=_F32),
        seed=seed,
    )
    model.validate()
    return model


def predict_examples(
    model: Phase10TModel, rows: Sequence[Phase10TExample], *, batch_size: int = 2_048
) -> np.ndarray:
    """Return direct cp and WDL logits for approved rows in deterministic batches."""

    if not rows:
        return np.empty((0, HEAD_COUNT), dtype=_F32)
    if batch_size <= 0:
        raise ValueError("prediction batch size must be positive")
    outputs: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        ids, mask, sides, _ = _feature_batch(batch)
        values = model.bias[None, None, :] + np.sum(model.table[ids] * mask[..., None], axis=2)
        positions = np.arange(len(batch))
        stm = sides
        packed = np.concatenate((values[positions, stm], values[positions, 1 - stm]), axis=1)
        hidden = np.maximum(packed @ model.hidden_weight + model.hidden_bias, 0.0)
        outputs.append(hidden @ model.head_weight + model.head_bias)
    return np.concatenate(outputs, axis=0).astype(_F32, copy=False)


def _supervised_metrics(
    predictions: np.ndarray, rows: Sequence[Phase10TExample]
) -> dict[str, float | int]:
    cp_mask = np.asarray([row.cp is not None for row in rows], dtype=bool)
    wdl_mask = np.asarray([row.wdl in (0, 1, 2) for row in rows], dtype=bool)
    result: dict[str, float | int] = {
        "examples": len(rows),
        "cp_examples": int(cp_mask.sum()),
        "wdl_examples": int(wdl_mask.sum()),
    }
    if bool(cp_mask.any()):
        targets = np.asarray([row.cp or 0.0 for row in rows], dtype=np.float64)
        errors = predictions[cp_mask, 0].astype(np.float64) - targets[cp_mask]
        result.update(
            cp_mae=float(np.mean(np.abs(errors))),
            cp_rmse=float(np.sqrt(np.mean(errors * errors))),
        )
    if bool(wdl_mask.any()):
        targets = np.asarray([row.wdl or 0 for row in rows], dtype=np.int64)
        result["wdl_accuracy"] = float(
            np.mean(np.argmax(predictions[wdl_mask, 1:], axis=1) == targets[wdl_mask])
        )
    return result


def train_supervised_lineage(
    rows: Sequence[Phase10TExample],
    *,
    seed: int,
    max_passes: int = 2,
    batch_size: int = 128,
    validation_every_steps: int = 250,
    patience: int = 4,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    weight_decay: float = 1e-4,
    gradient_clip: float = 5.0,
) -> tuple[Phase10TModel, Phase10TModel, Phase10TModel, dict[str, object]]:
    """Train the frozen a1 campaign lineage on train-only teacher labels.

    The returned models are the pre-stage random checkpoint, the selected validation checkpoint,
    and the final checkpoint respectively.  No holdout or source-held-out payload is read here.
    """

    if not rows or max_passes <= 0 or batch_size <= 0 or validation_every_steps <= 0:
        raise ValueError("supervised training bounds are invalid")
    if not 0.0 < min_learning_rate <= learning_rate:
        raise ValueError("learning-rate bounds are invalid")
    cp_count = sum(row.cp is not None for row in rows)
    wdl_count = sum(row.wdl in (0, 1, 2) for row in rows)
    if not cp_count and not wdl_count:
        raise ValueError("supervised training has no active target masks")
    import torch
    from torch.nn import functional as torch_f

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    validation_count = max(1, len(rows) // 10)
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    if not len(train_indices):
        raise ValueError("supervised training requires a non-empty train partition")
    ids, mask, sides, cps = _feature_batch(rows)
    wdl_values = np.asarray([0 if row.wdl is None else row.wdl for row in rows], dtype=np.int64)
    cp_mask = np.asarray([row.cp is not None for row in rows], dtype=bool)
    wdl_mask = np.asarray([row.wdl in (0, 1, 2) for row in rows], dtype=bool)
    sources = np.asarray([row.source for row in rows], dtype=object)
    tensor_ids = torch.from_numpy(ids)
    tensor_mask = torch.from_numpy(mask)
    tensor_sides = torch.from_numpy(sides)
    tensor_cps = torch.from_numpy(cps)
    tensor_cp_mask = torch.from_numpy(cp_mask)
    tensor_wdl = torch.from_numpy(wdl_values)
    tensor_wdl_mask = torch.from_numpy(wdl_mask)
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
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    steps_per_pass = math.ceil(len(train_indices) / batch_size)
    total_steps = max(1, max_passes * steps_per_pass)
    started = time.monotonic()
    steps = 0
    best_loss = float("inf")
    best_model = pre_model
    validation_history: list[dict[str, object]] = []
    checks_without_improvement = 0

    def forward(selected: np.ndarray) -> torch.Tensor:
        indices = torch.from_numpy(selected.astype(np.int64, copy=False))
        values = bias + table[tensor_ids[indices]] * tensor_mask[indices, ..., None]
        values = values.sum(dim=2)
        row_indices = torch.arange(len(selected), dtype=torch.long)
        stm = tensor_sides[indices]
        packed = torch.cat((values[row_indices, stm], values[row_indices, 1 - stm]), dim=1)
        hidden = torch.relu(packed @ hidden_weight + hidden_bias)
        return hidden @ head_weight + head_bias

    def source_macro_loss(output: torch.Tensor, selected: np.ndarray) -> torch.Tensor:
        source_names = sources[selected]
        groups: list[torch.Tensor] = []
        for source in sorted(set(source_names.tolist())):
            source_mask = torch.from_numpy(source_names == source)
            active: list[torch.Tensor] = []
            local_cp = tensor_cp_mask[torch.from_numpy(selected)] & source_mask
            local_wdl = tensor_wdl_mask[torch.from_numpy(selected)] & source_mask
            if bool(local_cp.any()):
                active.append(
                    torch_f.smooth_l1_loss(
                        output[local_cp, 0] / 600.0,
                        tensor_cps[torch.from_numpy(selected)][local_cp] / 600.0,
                    )
                )
            if bool(local_wdl.any()):
                logits = output[local_wdl, 1:]
                labels = tensor_wdl[torch.from_numpy(selected)][local_wdl]
                probabilities = torch.softmax(logits, dim=1)
                one_hot = torch_f.one_hot(labels, num_classes=3).to(probabilities.dtype)
                active.append(
                    torch_f.cross_entropy(logits, labels)
                    + 0.1 * torch.mean((probabilities - one_hot) ** 2)
                )
            if active:
                groups.append(torch.stack(active).mean())
        if not groups:
            raise ValueError("batch contains no active source-local targets")
        return torch.stack(groups).mean()

    def evaluate_validation() -> tuple[float, dict[str, object]]:
        selected = validation_indices.astype(np.int64, copy=False)
        with torch.no_grad():
            output = forward(selected)
            loss = float(source_macro_loss(output, selected).cpu())
        validation_rows = [rows[index] for index in selected.tolist()]
        predictions = predict_examples(model, validation_rows)
        metrics = _supervised_metrics(predictions, validation_rows)
        return loss, {"loss": loss, **metrics}

    for _ in range(max_passes):
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start : start + batch_size].astype(np.int64, copy=False)
            progress = steps / max(1, total_steps - 1)
            if progress < 0.1:
                current_lr = learning_rate * max(progress / 0.1, 1e-3)
            else:
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, (progress - 0.1) / 0.9)))
                current_lr = min_learning_rate + (learning_rate - min_learning_rate) * cosine
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            loss = source_macro_loss(forward(selected), selected)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("supervised loss is NaN or Inf")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            if not bool(torch.isfinite(norm).item()):
                raise ValueError("supervised gradient norm is NaN or Inf")
            optimizer.step()
            steps += 1
            if steps % validation_every_steps == 0 or steps == total_steps:
                model = _model_from_parameters(
                    [parameter.detach().cpu().numpy() for parameter in parameters], seed
                )
                validation_loss, metrics = evaluate_validation()
                metrics["step"] = steps
                validation_history.append(metrics)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_model = Phase10TModel.from_bytes(model.to_bytes())
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    break
        if checks_without_improvement >= patience:
            break
    model = _model_from_parameters(
        [parameter.detach().cpu().numpy() for parameter in parameters], seed
    )
    if not validation_history:
        model = Phase10TModel.from_bytes(model.to_bytes())
        validation_loss, metrics = evaluate_validation()
        best_loss = validation_loss
        best_model = Phase10TModel.from_bytes(model.to_bytes())
        validation_history.append({"step": steps, **metrics})
    elapsed = time.monotonic() - started
    stats = {
        "seed": seed,
        "passes_requested": max_passes,
        "passes_completed": min(max_passes, math.ceil(steps / max(1, steps_per_pass))),
        "steps": steps,
        "batch_size": batch_size,
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "cp_examples": cp_count,
        "wdl_examples": wdl_count,
        "source_counts": {
            source: int(np.sum(sources == source)) for source in sorted(set(sources.tolist()))
        },
        "validation_history": validation_history,
        "best_validation_loss": best_loss,
        "elapsed_seconds": elapsed,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "minimum_learning_rate": min_learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "source_macro_weighting": True,
        "random_initialization": True,
        "architecture": "a1",
    }
    return pre_model, best_model, model, stats


def train_random_lineage(
    rows: Sequence[Phase10TExample],
    *,
    seed: int,
    max_steps: int,
    time_limit_seconds: float,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> tuple[Phase10TModel, dict[str, object]]:
    """Train a bounded random lineage with the frozen AdamW-style settings.

    The function is deliberately small and deterministic.  It is used for the 32-position
    semantic/micro gate and for the later runner; it never reads a holdout or invents labels.
    """

    if not rows or max_steps <= 0 or time_limit_seconds <= 0:
        raise ValueError("training requires non-empty rows and positive bounds")
    import torch
    from torch.nn import functional as torch_f

    torch.manual_seed(seed)
    model = Phase10TModel.random(seed)
    ids, mask, sides, cps = _feature_batch(rows)
    tensor_ids = torch.from_numpy(ids)
    tensor_mask = torch.from_numpy(mask)
    tensor_sides = torch.from_numpy(sides)
    tensor_cps = torch.from_numpy(cps)
    table = torch.nn.Parameter(torch.from_numpy(model.table.copy()))
    bias = torch.nn.Parameter(torch.from_numpy(model.bias.copy()))
    hidden_weight = torch.nn.Parameter(torch.from_numpy(model.hidden_weight.copy()))
    hidden_bias = torch.nn.Parameter(torch.from_numpy(model.hidden_bias.copy()))
    head_weight = torch.nn.Parameter(torch.from_numpy(model.head_weight.copy()))
    head_bias = torch.nn.Parameter(torch.from_numpy(model.head_bias.copy()))
    parameters = [table, bias, hidden_weight, hidden_bias, head_weight, head_bias]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    started = time.monotonic()
    steps = 0
    final_loss = float("nan")
    while steps < max_steps and time.monotonic() - started <= time_limit_seconds:
        selected = torch.arange(len(rows))
        values = bias + table[tensor_ids[selected]] * tensor_mask[selected, ..., None]
        values = values.sum(dim=2)
        stm = tensor_sides[selected]
        batch = torch.cat(
            (
                values[torch.arange(len(rows)), stm],
                values[torch.arange(len(rows)), 1 - stm],
            ),
            dim=1,
        )
        hidden = torch.relu(batch @ hidden_weight + hidden_bias)
        output = hidden @ head_weight + head_bias
        cp_loss = torch_f.smooth_l1_loss(output[:, 0] / 600.0, tensor_cps / 600.0)
        # WDL labels in the approved rows are 0=loss, 1=draw, 2=win.  Missing rows are masked.
        wdl_values = torch.tensor(
            [0 if row.wdl is None else row.wdl for row in rows], dtype=torch.long
        )
        wdl_mask = torch.tensor([row.wdl is not None for row in rows], dtype=torch.bool)
        wdl_loss = (
            torch_f.cross_entropy(output[wdl_mask, 1:], wdl_values[wdl_mask])
            if bool(wdl_mask.any())
            else output[:, 1:].sum() * 0.0
        )
        loss = cp_loss + 0.1 * wdl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        steps += 1
        final_loss = float(loss.detach().cpu())
    elapsed = time.monotonic() - started
    model = Phase10TModel(
        table=table.detach().cpu().numpy().astype(_F32),
        bias=bias.detach().cpu().numpy().astype(_F32),
        hidden_weight=hidden_weight.detach().cpu().numpy().astype(_F32),
        hidden_bias=hidden_bias.detach().cpu().numpy().astype(_F32),
        head_weight=head_weight.detach().cpu().numpy().astype(_F32),
        head_bias=head_bias.detach().cpu().numpy().astype(_F32),
        seed=seed,
    )
    model.validate()
    return model, {
        "seed": seed,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "final_loss": final_loss,
        "random_initialization": True,
        "architecture": "a1",
    }


def examples_from_jsonl(path: Path, *, limit: int | None = None) -> list[Phase10TExample]:
    """Read only a caller-selected approved train JSONL file."""

    import json

    result: list[Phase10TExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise Phase10TModelError(f"non-train row encountered at {path}:{line_number}")
            wdl_value = row.get("wdl")
            wdl = int(wdl_value) if row.get("wdl_mask") and wdl_value in (0, 1, 2) else None
            result.append(Phase10TExample(str(row["sfen"]), 0.0, wdl))
            if limit is not None and len(result) >= limit:
                break
    if not result:
        raise Phase10TModelError(f"approved train file is empty: {path}")
    return result


__all__ = [
    "ACCUMULATOR_WIDTH",
    "DEFAULT_SEED",
    "FEATURE_COUNT",
    "HEAD_COUNT",
    "HIDDEN_WIDTH",
    "HistoryFacts",
    "Phase10TExample",
    "Phase10TModel",
    "Phase10TModelError",
    "examples_from_jsonl",
    "feature_ids",
    "predict_examples",
    "sparse_features",
    "train_random_lineage",
    "train_supervised_lineage",
]
