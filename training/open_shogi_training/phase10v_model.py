"""Independent Phase 10V sparse pair evaluator and bounded CPU training.

OSAVAL03: 44-byte little-endian header, six float32 tensors, then SHA256 of
header AND payload. The scalar head is centipawns; WDL never synthesizes its score.
The elementwise product of clipped opposing accumulators supplies learned relations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from open_shogi_training.phase10r_model import ParsedPosition, parse_sfen
from open_shogi_training.phase10v_targets import Phase10VExample, TeacherScore, iter_examples

FEATURE_COUNT = 8427
HIDDEN_WIDTH = 16
HEAD_COUNT = 4
MODEL_MAGIC = b"OSAVAL03"
MODEL_VERSION = 3
FEATURE_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260908
MAX_PARAMETER_MAGNITUDE = 1_000_000.0
MAX_MODEL_BYTES = 32 * 1024 * 1024
KING_OFFSET = 17 * 17 * 28
HAND_OFFSET = KING_OFFSET + 81
STM_OFFSET = HAND_OFFSET + 2 * 7 * 18
ACCUMULATOR_GRID = 2**20
CP_PARAMETER_SCALE = 600.0
HARD_FREE_SPACE_FLOOR = 80 * 1024**3
_HEADER = struct.Struct("<8s6IQf")
_F32 = np.dtype("<f4")


class Phase10VModelError(ValueError):
    """A model/feature violation must stop the learned path, never fall back."""


def feature_ids(position: ParsedPosition, perspective: int) -> tuple[int, ...]:
    """King-relative 17x17 board, promoted kinds, unary hands and relative STM."""
    if perspective not in (0, 1):
        raise Phase10VModelError("invalid perspective")
    kings = [p.square for p in position.board if p and p.kind == 7 and p.side == perspective]
    if len(kings) != 1:
        raise Phase10VModelError("exactly one own king required")
    king = 80 - kings[0] if perspective else kings[0]
    king_rank, king_file = divmod(king, 9)
    features = {KING_OFFSET + king, STM_OFFSET + int(position.side_to_move != perspective)}
    for piece in position.board:
        if piece is None:
            continue
        square = 80 - piece.square if perspective else piece.square
        rank, file = divmod(square, 9)
        relative_square = (rank - king_rank + 8) * 17 + file - king_file + 8
        features.add(relative_square * 28 + (piece.side ^ perspective) * 14 + piece.kind)
    for owner, hand in enumerate(position.hands):
        for kind, count in enumerate(hand):
            if not 0 <= count <= 18:
                raise Phase10VModelError("hand exceeds unary feature range")
            for ordinal in range(count):
                features.add(HAND_OFFSET + ((owner ^ perspective) * 7 + kind) * 18 + ordinal)
    if any(feature < 0 or feature >= FEATURE_COUNT for feature in features):
        raise Phase10VModelError("feature index outside OSAVAL03 schema")
    return tuple(sorted(features))


def sparse_features(sfen: str) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    position = parse_sfen(sfen)
    return feature_ids(position, 0), feature_ids(position, 1), position.side_to_move


def _shapes(width: int) -> tuple[tuple[int, ...], ...]:
    return (
        (FEATURE_COUNT, width),
        (width,),
        (3 * width, HIDDEN_WIDTH),
        (HIDDEN_WIDTH,),
        (HIDDEN_WIDTH, HEAD_COUNT),
        (HEAD_COUNT,),
    )


def _on_accumulator_grid(array: np.ndarray) -> np.ndarray:
    return (np.rint(array.astype(np.float64) * ACCUMULATOR_GRID) / ACCUMULATOR_GRID).astype(_F32)


@dataclass(slots=True)
class Phase10VModel:
    table: np.ndarray
    bias: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    head_weight: np.ndarray
    head_bias: np.ndarray
    seed: int

    @property
    def width(self) -> int:
        return self.bias.size

    @property
    def parameters(self) -> tuple[np.ndarray, ...]:
        return (
            self.table,
            self.bias,
            self.hidden_weight,
            self.hidden_bias,
            self.head_weight,
            self.head_bias,
        )

    @classmethod
    def random(cls, seed: int = DEFAULT_SEED, width: int = 256) -> Phase10VModel:
        if width not in (256, 512) or type(seed) is not int or not 0 <= seed < 2**64:
            raise Phase10VModelError("only width 256/512 and uint64 seeds are supported")
        rng = np.random.default_rng(seed)
        values = [rng.normal(0, 0.025, shape).astype(_F32) for shape in _shapes(width)]
        values[1] += 0.2
        values[3] += 0.2
        values[0] = _on_accumulator_grid(values[0])
        values[1] = _on_accumulator_grid(values[1])
        # A scalar head in actual cp units needs a useful initial derivative scale.
        # This is random initialization, not a handcrafted position-dependent score.
        values[4][:, 0] *= 8000.0
        return cls(*values, seed=seed)

    def validate(self) -> None:
        if self.width not in (256, 512) or type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise Phase10VModelError("invalid architecture width or seed")
        for array, shape in zip(self.parameters, _shapes(self.width), strict=True):
            if array.shape != shape or array.dtype != _F32:
                raise Phase10VModelError("tensor shape/dtype mismatch")
            if not np.isfinite(array).all() or np.any(np.abs(array) > MAX_PARAMETER_MAGNITUDE):
                raise Phase10VModelError("nonfinite or unsafe model parameter")

        for array in (self.table, self.bias):
            if not np.array_equal(array, _on_accumulator_grid(array)):
                raise Phase10VModelError("accumulator parameters must use the exact Q20 grid")

    def accumulate(self, features: Sequence[int]) -> np.ndarray:
        result = self.bias.astype(np.float64)
        for index in features:
            result += self.table[index]
        return result

    def evaluate(self, sfen: str) -> tuple[float, np.ndarray]:
        black, white, stm = sparse_features(sfen)
        pair = (self.accumulate(black), self.accumulate(white))
        us = np.clip(pair[stm], 0, 1).astype(_F32)
        them = np.clip(pair[1 - stm], 0, 1).astype(_F32)
        hidden = np.clip(
            np.concatenate((us, them, us * them)) @ self.hidden_weight + self.hidden_bias, 0, 1
        )
        output = hidden @ self.head_weight + self.head_bias
        return float(output[0]), output[1:].copy()

    def search_score(self, sfen: str) -> int:
        cp, _ = self.evaluate(sfen)
        return max(-20000, min(20000, int(math.copysign(math.floor(abs(cp) + 0.5), cp))))

    def to_bytes(self) -> bytes:
        self.validate()
        header = _HEADER.pack(
            MODEL_MAGIC,
            MODEL_VERSION,
            FEATURE_SCHEMA_VERSION,
            FEATURE_COUNT,
            self.width,
            HIDDEN_WIDTH,
            HEAD_COUNT,
            self.seed,
            1.0,
        )
        body = header + b"".join(value.tobytes(order="C") for value in self.parameters)
        return body + hashlib.sha256(body).digest()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def write(self, path: Path) -> str:
        data = self.to_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> Phase10VModel:
        if not _HEADER.size + 32 <= len(data) <= MAX_MODEL_BYTES:
            raise Phase10VModelError("model size exceeds format bounds")
        magic, version, schema, count, width, hidden, heads, seed, scale = _HEADER.unpack_from(data)
        if (magic, version, schema, count, hidden, heads, scale) != (
            MODEL_MAGIC,
            MODEL_VERSION,
            FEATURE_SCHEMA_VERSION,
            FEATURE_COUNT,
            HIDDEN_WIDTH,
            HEAD_COUNT,
            1.0,
        ) or width not in (256, 512):
            raise Phase10VModelError("incompatible OSAVAL03 architecture/header")
        shapes = _shapes(width)
        if len(data) != _HEADER.size + sum(math.prod(shape) * 4 for shape in shapes) + 32:
            raise Phase10VModelError("model tensor payload length mismatch")
        if hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise Phase10VModelError("header/payload checksum mismatch")
        cursor, arrays = _HEADER.size, []
        for shape in shapes:
            count = math.prod(shape)
            arrays.append(
                np.frombuffer(data, dtype=_F32, count=count, offset=cursor).reshape(shape).copy()
            )
            cursor += count * 4
        model = cls(*arrays, seed=seed)
        model.validate()
        return model

    @classmethod
    def read(cls, path: Path) -> Phase10VModel:
        if path.stat().st_size > MAX_MODEL_BYTES:
            raise Phase10VModelError("model file exceeds format bounds")
        return cls.from_bytes(path.read_bytes())


def torch_parameters(model: Phase10VModel) -> list[Any]:
    import torch

    values = [value.copy() for value in model.parameters]
    # Optimize the scalar head in ordinary neural units; export stores actual cp weights.
    values[4][:, 0] /= CP_PARAMETER_SCALE
    values[5][0] /= CP_PARAMETER_SCALE
    return [torch.nn.Parameter(torch.from_numpy(value)) for value in values]


def torch_forward(parameters: Sequence[Any], sfens: Sequence[str]) -> Any:
    """Autograd parity path: accumulator bias is added exactly once per perspective."""
    import torch

    table, bias, hw, hb, ow, ob = parameters
    encoded = [sparse_features(sfen) for sfen in sfens]
    largest = max(len(part) for item in encoded for part in item[:2])
    ids = torch.zeros((len(sfens), 2, largest), dtype=torch.long)
    mask = torch.zeros((len(sfens), 2, largest), dtype=table.dtype)
    for index, (black, white, _) in enumerate(encoded):
        for side, features in enumerate((black, white)):
            ids[index, side, : len(features)] = torch.tensor(features)
            mask[index, side, : len(features)] = 1
    selected = table[ids]
    # Quantization-aware straight-through gradients; exact Q20/f64 sums match export.
    selected = (
        selected + ((selected * ACCUMULATOR_GRID).round() / ACCUMULATOR_GRID - selected).detach()
    )
    quantized_bias = bias + ((bias * ACCUMULATOR_GRID).round() / ACCUMULATOR_GRID - bias).detach()
    values = quantized_bias.to(torch.float64) + (selected.to(torch.float64) * mask[..., None]).sum(
        dim=2
    )
    sides = torch.tensor([item[2] for item in encoded])
    positions = torch.arange(len(sfens))
    us = values[positions, sides].clamp(0, 1).to(table.dtype)
    them = values[positions, 1 - sides].clamp(0, 1).to(table.dtype)
    hidden = (torch.cat((us, them, us * them), dim=1) @ hw + hb).clamp(0, 1)
    output = hidden @ ow + ob
    return torch.cat((output[:, :1] * CP_PARAMETER_SCALE, output[:, 1:]), dim=1)


def training_loss(
    parameters: Sequence[Any],
    rows: Sequence[Phase10VExample],
    *,
    cp_scale: float = 600.0,
    wdl_weight: float = 0.1,
    ranking_weight: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Primary robust direct cp loss plus masked factual WDL and observed child ranking."""
    import torch
    from torch.nn import functional as functional

    if not rows or cp_scale <= 0 or min(wdl_weight, ranking_weight) < 0:
        raise ValueError("invalid loss batch/config")
    output = torch_forward(parameters, [row.sfen for row in rows])
    zero = output.sum() * 0
    source_cp, source_wdl, source_ranking = [], [], []
    for source in sorted({row.source for row in rows}):
        cp_indices = [
            i for i, row in enumerate(rows) if row.source == source and row.cp is not None
        ]
        if cp_indices:
            source_cp.append(
                functional.smooth_l1_loss(
                    output[cp_indices, 0] / cp_scale,
                    torch.tensor([rows[i].cp for i in cp_indices], dtype=output.dtype) / cp_scale,
                )
            )
        wdl_indices = [
            i for i, row in enumerate(rows) if row.source == source and row.wdl is not None
        ]
        if wdl_indices:
            source_wdl.append(
                functional.cross_entropy(
                    output[wdl_indices, 1:],
                    torch.tensor([rows[i].wdl for i in wdl_indices]),
                )
            )
        pairs = []
        for row in rows:
            if row.source == source and row.ranking_pairs:
                # Labels use parent perspective; network input is the actual child.
                root_cp = -torch_forward(parameters, [c.child_sfen for c in row.candidates])[:, 0]
                for winner, loser in row.ranking_pairs:
                    pairs.append(functional.softplus((root_cp[loser] - root_cp[winner]) / cp_scale))
        if pairs:
            source_ranking.append(torch.stack(pairs).mean())
    cp_loss = torch.stack(source_cp).mean() if source_cp else zero
    wdl_loss = torch.stack(source_wdl).mean() if source_wdl else zero
    ranking_loss = torch.stack(source_ranking).mean() if source_ranking else zero
    loss = cp_loss + wdl_weight * wdl_loss + ranking_weight * ranking_loss
    return loss, {
        "loss": float(loss.detach()),
        "cp_loss": float(cp_loss.detach()),
        "wdl_loss": float(wdl_loss.detach()),
        "ranking_loss": float(ranking_loss.detach()),
    }


def _snapshot(parameters: Sequence[Any], seed: int) -> Phase10VModel:
    values = [p.detach().cpu().numpy().copy() for p in parameters]
    values[0] = _on_accumulator_grid(values[0])
    values[1] = _on_accumulator_grid(values[1])
    values[4][:, 0] *= CP_PARAMETER_SCALE
    values[5][0] *= CP_PARAMETER_SCALE
    model = Phase10VModel(*values, seed=seed)
    model.validate()
    return model


def _batches(
    factory: Callable[[], Iterable[Phase10VExample]], size: int
) -> Iterable[list[Phase10VExample]]:
    batch = []
    for row in factory():
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _split_identity(
    factory: Callable[[], Iterable[Phase10VExample]], split: str
) -> tuple[set[tuple[str, str]], set[str], dict[str, int], str]:
    games, positions, distribution = set(), set(), {}
    digest = hashlib.sha256()
    for row in factory():
        row.validate()
        if row.split != split:
            raise ValueError("stream contains unexpected/final-holdout split")
        games.add((row.source, row.source_game_id))
        positions.add(hashlib.sha256(parse_sfen(row.sfen).canonical_state.encode()).hexdigest())
        distribution[row.leaf_kind] = distribution.get(row.leaf_kind, 0) + 1
        digest.update(repr(row).encode())
        digest.update(b"\n")
    if not positions:
        raise ValueError("empty approved split")
    return games, positions, distribution, digest.hexdigest()


def train_supervised(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    *,
    data_receipt_path: Path,
    data_receipt_sha256: str,
    width: int = 256,
    seed: int = DEFAULT_SEED,
    max_passes: int = 2,
    max_steps: int = 100000,
    batch_size: int = 128,
    validation_every_steps: int = 250,
    learning_rate: float = 0.0003,
    initial_model: Phase10VModel | None = None,
    expected_initial_sha256: str | None = None,
    initial_training_receipt_path: Path | None = None,
    initial_training_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Train only files reconstructed from raw evidence and an independently pinned receipt."""
    from open_shogi_training.phase10v_data import verify_training_inputs

    initial_provenance = _verify_initial_lineage(
        initial_model,
        seed,
        width,
        expected_initial_sha256,
        initial_training_receipt_path,
        initial_training_receipt_sha256,
    )
    _ensure_free_space(output_dir)
    approval = verify_training_inputs(
        train_path, validation_path, data_receipt_path, data_receipt_sha256
    )
    metrics = _train_verified_streams(
        lambda: iter_examples(train_path),
        lambda: iter_examples(validation_path, expected_split="validation"),
        output_dir,
        width=width,
        seed=seed,
        max_passes=max_passes,
        max_steps=max_steps,
        batch_size=batch_size,
        validation_every_steps=validation_every_steps,
        learning_rate=learning_rate,
        initial_model=initial_model,
        initial_provenance=initial_provenance,
        data_approval={"receipt_sha256": data_receipt_sha256, "evidence": approval},
    )
    metrics["verified_data_receipt"] = approval
    metrics["verified_data_receipt_sha256"] = data_receipt_sha256
    _atomic_write(
        output_dir / "metrics.json", json.dumps(metrics, indent=2, allow_nan=False).encode()
    )
    return metrics


def _verify_initial_lineage(
    model: Phase10VModel | None,
    seed: int,
    width: int,
    expected_sha: str | None,
    receipt_path: Path | None,
    receipt_sha: str | None,
) -> dict[str, Any] | None:
    if model is None:
        if any(value is not None for value in (expected_sha, receipt_path, receipt_sha)):
            raise ValueError("initial lineage arguments supplied without a model")
        return None
    if not all((expected_sha, receipt_path, receipt_sha)):
        raise ValueError("continuation requires pinned parent model and training receipt")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt_sha or model.sha256 != expected_sha:
        raise ValueError("parent training receipt/model hash mismatch")
    receipt = json.loads(raw)
    random_root = Phase10VModel.random(seed, width).sha256
    schema = receipt.get("schema")
    allowed_models = (
        (receipt.get("best_sha256"), receipt.get("latest_sha256"))
        if schema == "open_shogiai_phase10v_training/v1"
        else (receipt.get("candidate_sha256"),)
    )
    if (
        schema not in {"open_shogiai_phase10v_training/v1", "open_shogiai_phase10v_calibration/v1"}
        or receipt.get("seed") != seed
        or receipt.get("width") != width
        or expected_sha not in allowed_models
        or receipt.get("lineage_root_sha256") != random_root
        or not receipt.get("verified_data_receipt")
    ):
        raise ValueError("parent checkpoint lacks verified random-initialized training lineage")
    return {
        "model_sha256": expected_sha,
        "training_receipt_sha256": receipt_sha,
        "lineage_root_sha256": random_root,
    }


def _train_verified_streams(
    train_factory: Callable[[], Iterable[Phase10VExample]],
    validation_factory: Callable[[], Iterable[Phase10VExample]],
    output_dir: Path,
    *,
    width: int = 256,
    seed: int = DEFAULT_SEED,
    max_passes: int = 2,
    max_steps: int = 100000,
    batch_size: int = 128,
    validation_every_steps: int = 250,
    learning_rate: float = 0.0003,
    initial_model: Phase10VModel | None = None,
    data_approval: dict[str, Any] | None = None,
    initial_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream deterministic approved stages, preserving best/latest and identity metrics.

    Callers supply immutable re-iterable streams. Development validation is explicit;
    no random row split can leak positions from the same source game across partitions.
    Best is an offline checkpoint proposal only; campaign Arena selects playing strength.
    """
    import torch

    if min(max_passes, max_steps, batch_size, validation_every_steps) <= 0 or learning_rate <= 0:
        raise ValueError("invalid bounded training configuration")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "training output must be new/empty; preserved checkpoints cannot be reused"
        )
    train_games, train_positions, train_distribution, train_hash = _split_identity(
        train_factory, "train"
    )
    val_games, val_positions, val_distribution, validation_hash = _split_identity(
        validation_factory, "validation"
    )
    if train_games & val_games or train_positions & val_positions:
        raise ValueError("train/development split leakage detected")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = initial_model or Phase10VModel.random(seed, width)
    model.validate()
    if model.width != width or model.seed != seed:
        raise ValueError("continuation model width/lineage mismatch")
    parameters = torch_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_sha256 = model.write(output_dir / "initial.osaval03")
    lineage_root = (
        initial_provenance["lineage_root_sha256"] if initial_provenance else initial_sha256
    )
    history, best_loss, best_hash, steps = [], math.inf, None, 0

    def checkpoint() -> None:
        nonlocal best_loss, best_hash
        if data_approval is not None:
            _ensure_free_space(output_dir)
        metrics_sum, cp_abs, cp_squared, cp_count, count = {}, 0.0, 0.0, 0, 0
        with torch.no_grad():
            for batch in _batches(validation_factory, batch_size):
                for row in batch:
                    row.validate()
                    if row.split != "validation":
                        raise ValueError("validation split changed during execution")
                _, metrics = training_loss(parameters, batch)
                for name, value in metrics.items():
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + value * len(batch)
                output = torch_forward(parameters, [row.sfen for row in batch])[:, 0].numpy()
                for prediction, row in zip(output, batch, strict=True):
                    if row.cp is not None:
                        error = float(prediction) - row.cp
                        cp_abs += abs(error)
                        cp_squared += error * error
                        cp_count += 1
                count += len(batch)
        metrics = {name: value / count for name, value in metrics_sum.items()}
        metrics.update(
            step=steps,
            cp_count=cp_count,
            cp_mae=cp_abs / cp_count if cp_count else None,
            cp_rmse=math.sqrt(cp_squared / cp_count) if cp_count else None,
        )
        current = _snapshot(parameters, seed)
        encoded = current.to_bytes()
        metrics["model_sha256"] = hashlib.sha256(encoded).hexdigest()
        _atomic_write(output_dir / "latest.osaval03", encoded)
        if metrics["loss"] < best_loss:
            best_loss, best_hash = metrics["loss"], metrics["model_sha256"]
            _atomic_write(output_dir / "best.osaval03", encoded)
        history.append(metrics)
        _atomic_write(
            output_dir / "metrics.json",
            json.dumps(
                {
                    "schema": "open_shogiai_phase10v_training/v1",
                    "verified_data_receipt": data_approval,
                    "initial_model_sha256": initial_sha256,
                    "initial_model_provenance": initial_provenance,
                    "lineage_root_sha256": lineage_root,
                    "source_macro_weighting": True,
                    "cp_parameter_scale": CP_PARAMETER_SCALE,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "seed": seed,
                    "width": width,
                    "steps": steps,
                    "best_sha256": best_hash,
                    "latest_sha256": metrics["model_sha256"],
                    "train_stream_sha256": train_hash,
                    "validation_stream_sha256": validation_hash,
                    "train_distribution": train_distribution,
                    "validation_distribution": val_distribution,
                    "checkpoint_selection": "offline_proposal_requires_equal_node_and_clock_arena",
                    "random_initialization": initial_model is None,
                    "history": history,
                },
                indent=2,
                allow_nan=False,
            ).encode(),
        )

    for _ in range(max_passes):
        for batch in _batches(train_factory, batch_size):
            if data_approval is not None and steps % 100 == 0:
                _ensure_free_space(output_dir)
            for row in batch:
                row.validate()
                if row.split != "train":
                    raise ValueError("training split changed during execution")
            if not any(
                row.cp is not None or row.wdl is not None or row.ranking_pairs for row in batch
            ):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_loss(parameters, batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite training gradient")
            optimizer.step()
            steps += 1
            if steps % validation_every_steps == 0:
                checkpoint()
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    if not history or history[-1]["step"] != steps:
        checkpoint()
    return json.loads((output_dir / "metrics.json").read_text())


def _ensure_free_space(path: Path) -> None:
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < HARD_FREE_SPACE_FLOOR:
        raise ValueError("Phase 10V hard 80 GiB free-space floor reached")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def micro_overfit(
    rows: Sequence[Phase10VExample],
    *,
    width: int = 256,
    seed: int = DEFAULT_SEED,
    steps: int = 120,
    learning_rate: float = 0.001,
) -> tuple[Phase10VModel, dict[str, Any]]:
    """Small semantic/autograd diagnostic; never a strength or production training claim."""
    import torch

    if not rows or len(rows) > 32 or not 1 <= steps <= 2000:
        raise ValueError("micro gate allows 1..32 positions and 1..2000 optimizer steps")
    for row in rows:
        row.validate(production=False)
        if row.split != "train":
            raise ValueError("micro gate cannot read validation/final holdout")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = Phase10VModel.random(seed, width)
    parameters = torch_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    initial, _ = training_loss(parameters, rows)
    initial_loss = float(initial.detach())
    gradient_norms = [0.0] * len(parameters)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_loss(parameters, rows)
        loss.backward()
        for i, parameter in enumerate(parameters):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError("missing/nonfinite all-layer autograd")
            gradient_norms[i] = max(gradient_norms[i], float(parameter.grad.norm()))
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
    final, metrics = training_loss(parameters, rows)
    model = _snapshot(parameters, seed)
    errors = [abs(model.evaluate(row.sfen)[0] - row.cp) for row in rows if row.cp is not None]
    metrics.update(
        initial_loss=initial_loss,
        final_loss=float(final.detach()),
        steps=steps,
        examples=len(rows),
        cp_mae=float(np.mean(errors)) if errors else None,
        gradient_norms=gradient_norms,
        all_layers_have_gradient=all(gradient_norms),
        width=width,
        model_sha256=model.sha256,
        synthetic_diagnostic=all(row.source == "diagnostic" for row in rows),
        evidence_class="bounded_micro_overfit_not_playing_strength",
        source_macro_weighting=True,
        cp_parameter_scale=CP_PARAMETER_SCALE,
        accumulator_grid="Q20",
        accumulator_arithmetic="float64",
    )
    return model, metrics


def fit_affine_calibration(
    predictions: Sequence[float], targets: Sequence[float]
) -> tuple[float, float]:
    """Positive bounded least-squares cp calibration; no sign reversal is authorized."""
    x, y = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
        raise ValueError("calibration needs at least three paired scalar samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration samples must be finite")
    centered = x - x.mean()
    variance = float(centered @ centered)
    if variance <= 1e-8:
        raise ValueError("degenerate calibration prediction variance")
    unconstrained = float(centered @ (y - y.mean()) / variance)
    if not math.isfinite(unconstrained) or unconstrained <= 0:
        raise ValueError("calibration would reverse or erase the score sign")
    scale = max(0.25, min(4.0, unconstrained))
    offset = max(-2000.0, min(2000.0, float(np.mean(y - scale * x))))
    return scale, offset


def affine_candidate(model: Phase10VModel, scale: float, offset: float) -> Phase10VModel:
    """Transform only the direct cp head; retain every WDL/feature parameter exactly."""
    if not math.isfinite(scale) or not math.isfinite(offset) or not 0.25 <= scale <= 4.0:
        raise ValueError("calibration scale outside the frozen positive bounds")
    if abs(offset) > 2000:
        raise ValueError("calibration offset outside the frozen cp bounds")
    candidate = Phase10VModel.from_bytes(model.to_bytes())
    candidate.head_weight[:, 0] *= scale
    candidate.head_bias[0] = candidate.head_bias[0] * scale + offset
    candidate.validate()
    return candidate


def calibrate_candidate(
    model_path: Path,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    *,
    expected_model_sha256: str,
    parent_receipt_path: Path,
    parent_receipt_sha256: str,
    data_receipt_path: Path,
    data_receipt_sha256: str,
    max_examples: int = 20000,
) -> dict[str, Any]:
    """Publish one development-only proposal; subsequent equal-node/clock Arena decides."""
    from open_shogi_training.phase10v_data import verify_training_inputs

    if not 3 <= max_examples <= 20000:
        raise ValueError("calibration is bounded to 3..20000 development samples")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("calibration output must be new/empty")
    model = Phase10VModel.read(model_path)
    parent = _verify_initial_lineage(
        model,
        model.seed,
        model.width,
        expected_model_sha256,
        parent_receipt_path,
        parent_receipt_sha256,
    )
    _ensure_free_space(output_dir)
    approval = verify_training_inputs(
        train_path, validation_path, data_receipt_path, data_receipt_sha256
    )
    predictions, targets, sfens, sources = [], [], [], {}
    sample_hash = hashlib.sha256()
    scanned = 0
    for row in iter_examples(validation_path, expected_split="validation"):
        if scanned >= max_examples:
            break
        scanned += 1
        if row.cp is None:
            continue
        sfens.append(row.sfen)
        predictions.append(model.evaluate(row.sfen)[0])
        targets.append(row.cp)
        sources[row.source] = sources.get(row.source, 0) + 1
        sample_hash.update(repr(row).encode() + b"\n")
        if len(predictions) >= max_examples:
            break
    scale, offset = fit_affine_calibration(predictions, targets)
    candidate = affine_candidate(model, scale, offset)
    before = np.asarray(predictions, dtype=np.float64)
    after = np.clip([candidate.evaluate(sfen)[0] for sfen in sfens], -20000, 20000)
    target_values = np.asarray(targets)
    metrics = {
        "schema": "open_shogiai_phase10v_calibration/v1",
        "seed": model.seed,
        "width": model.width,
        "parent_model_sha256": expected_model_sha256,
        "parent_receipt_sha256": parent_receipt_sha256,
        "lineage_root_sha256": parent["lineage_root_sha256"],
        "verified_data_receipt": approval,
        "verified_data_receipt_sha256": data_receipt_sha256,
        "fit_split": "validation",
        "scanned_rows": scanned,
        "cp_examples": len(predictions),
        "sample_sha256": sample_hash.hexdigest(),
        "source_counts": sources,
        "scale": scale,
        "offset_cp": offset,
        "before_cp_mae": float(np.mean(np.abs(np.clip(before, -20000, 20000) - target_values))),
        "after_cp_mae": float(np.mean(np.abs(after - target_values))),
        "selection": "proposal_only_requires_equal_node_and_equal_clock_arena",
        "candidate_sha256": candidate.sha256,
        "wdl_unchanged": True,
    }
    _ensure_free_space(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate.write(output_dir / "calibrated.osaval03")
    _atomic_write(
        output_dir / "calibration.json", json.dumps(metrics, indent=2, allow_nan=False).encode()
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--micro-overfit", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-max-examples", type=int, default=20000)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--data-receipt", type=Path)
    parser.add_argument("--data-receipt-sha256")
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--initial-model-sha256")
    parser.add_argument("--initial-training-receipt", type=Path)
    parser.add_argument("--initial-training-receipt-sha256")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-every-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(256, 512), default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    if args.calibrate:
        if not all(
            (
                args.initial_model,
                args.initial_model_sha256,
                args.initial_training_receipt,
                args.initial_training_receipt_sha256,
                args.train,
                args.validation,
                args.data_receipt,
                args.data_receipt_sha256,
            )
        ):
            parser.error(
                "calibration requires pinned model/receipt and verified train/validation inputs"
            )
        metrics = calibrate_candidate(
            args.initial_model,
            args.train,
            args.validation,
            args.output,
            expected_model_sha256=args.initial_model_sha256,
            parent_receipt_path=args.initial_training_receipt,
            parent_receipt_sha256=args.initial_training_receipt_sha256,
            data_receipt_path=args.data_receipt,
            data_receipt_sha256=args.data_receipt_sha256,
            max_examples=args.calibration_max_examples,
        )
    elif args.micro_overfit:
        if args.train:
            import itertools

            rows = list(itertools.islice(iter_examples(args.train), 32))
        else:
            start = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
            child = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2"
            rows = [
                Phase10VExample(start, TeacherScore("cp", 120)),
                Phase10VExample(child, TeacherScore("cp", -90)),
            ]
        model, metrics = micro_overfit(rows, width=args.width, seed=args.seed, steps=args.steps)
        args.output.mkdir(parents=True, exist_ok=True)
        model.write(args.output / "micro.osaval03")
        with (args.output / "micro-metrics.json").open("x") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
    else:
        if not all((args.train, args.validation, args.data_receipt, args.data_receipt_sha256)):
            parser.error(
                "training requires --train, --validation, --data-receipt and its pinned SHA256"
            )
        metrics = train_supervised(
            args.train,
            args.validation,
            args.output,
            data_receipt_path=args.data_receipt,
            data_receipt_sha256=args.data_receipt_sha256,
            width=args.width,
            seed=args.seed,
            max_steps=args.steps,
            max_passes=args.passes,
            batch_size=args.batch_size,
            validation_every_steps=args.validation_every_steps,
            learning_rate=args.learning_rate,
            initial_model=Phase10VModel.read(args.initial_model) if args.initial_model else None,
            expected_initial_sha256=args.initial_model_sha256,
            initial_training_receipt_path=args.initial_training_receipt,
            initial_training_receipt_sha256=args.initial_training_receipt_sha256,
        )
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
