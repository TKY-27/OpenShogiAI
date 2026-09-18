"""All-layer OSAVAL03 learning with exact sampler/optimizer resume and bounded retention."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

from .evaluator_data import atomic, digest, encoded
from .phase10v_model import (
    ACCUMULATOR_GRID,
    CP_PARAMETER_SCALE,
    Phase10VModel,
    _snapshot,
    torch_parameters,
)


class TrainingResourceWaitError(RuntimeError):
    """Confirmed allocation failure; restart from the last published checkpoint."""


class CheckpointPublicationError(RuntimeError):
    """A successful update was not durably saved; automatic replay is forbidden."""


def allocation_failure(error: BaseException) -> bool:
    return isinstance(error, MemoryError | torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and any(
            marker in str(error).lower()
            for marker in (
                "defaultcpuallocator: can't allocate memory",
                "mps backend out of memory",
            )
        )
    )


def _accumulate(parameters: list, data: dict, indexes, microbatch_size: int) -> float:
    """Select rows before evaluation; normalize each sum by the effective batch size."""
    total = 0.0
    if not len(indexes):
        raise ValueError("empty optimizer batch")
    for start in range(0, len(indexes), microbatch_size):
        features, lengths, y = batch(data, indexes[start : start + microbatch_size])
        predicted = forward(parameters, features, lengths)
        loss = functional.smooth_l1_loss(predicted / 600, y / 600, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        (loss / len(indexes)).backward()
        total += float(loss.detach())
    return total


def accumulate(parameters: list, data: dict, indexes, microbatch_size: int) -> tuple[float, int]:
    """Each failed microbatch pass is discarded before a finite, smaller retry."""
    while True:
        for parameter in parameters:
            parameter.grad = None
        try:
            return _accumulate(parameters, data, indexes, microbatch_size), microbatch_size
        except (MemoryError, RuntimeError) as error:
            if not allocation_failure(error):
                raise
            # Clear traceback-held tensors before retrying, not merely Python locals.
            error.__traceback__ = None
            exhausted = microbatch_size == 1
        for parameter in parameters:
            parameter.grad = None
        gc.collect()
        if exhausted:
            raise TrainingResourceWaitError("training allocation failed at microbatch 1")
        microbatch_size = max(1, microbatch_size // 2)


GROUPS = ("general", "opening", "defense", "attack_end")


def grouped_arrays(folder: Path, split: str) -> dict:
    data = arrays(folder, split)
    data["groups"] = np.load(folder / f"{split}-groups.npy", allow_pickle=False)
    if data["groups"].shape != data["targets"].shape or not np.isin(data["groups"], range(4)).all():
        raise ValueError("invalid sampling groups")
    if (folder / f"{split}-sources.npy").exists():
        data["sources"] = np.load(folder / f"{split}-sources.npy", allow_pickle=False)
        if (
            data["sources"].shape != data["targets"].shape
            or not np.isin(data["sources"], (0, 1)).all()
        ):
            raise ValueError("invalid source sampling membership")
    return data


def evaluate_groups(
    parameters: list, data: dict, batch_size: int, *, source: int | None = None
) -> dict:
    result = {}
    for i, name in enumerate(GROUPS):
        mask = data["groups"] == i
        if source is not None:
            mask &= data["sources"] == source
        indexes = np.flatnonzero(mask)
        if not len(indexes):
            raise ValueError(f"missing independent validation group: {name}")
        result[name] = evaluate(parameters, data, batch_size, indexes=indexes)
    return result


def stratified_order(
    data: dict, fractions: list[float], generator: torch.Generator
) -> torch.Tensor:
    """One bounded pass, no replacement; reshuffle independently on the next pass."""
    if len(fractions) != 4 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("invalid four-group sampling fractions")
    members = [torch.from_numpy(np.flatnonzero(data["groups"] == i)) for i in range(4)]
    size = int(min(len(m) / f for m, f in zip(members, fractions, strict=True)))
    if size < 4:
        raise ValueError("insufficient independent rows for stratified training")
    chosen = []
    for m, fraction in zip(members, fractions, strict=True):
        count = max(1, int(size * fraction))
        chosen.append(m[torch.randperm(len(m), generator=generator)[:count]])
    order = torch.cat(chosen)
    return order[torch.randperm(len(order), generator=generator)]


def mixed_order(
    data: dict, fractions: list[float], sources: list[float], generator: torch.Generator
) -> torch.Tensor:
    """Fixed new/replay ratio without replacement; every exposure remains countable."""
    if len(sources) != 2 or min(sources) <= 0 or abs(sum(sources) - 1) > 1e-9:
        raise ValueError("invalid new/replay sampling fractions")
    replay = torch.from_numpy(np.flatnonzero(data["sources"] == 0))
    new = torch.from_numpy(np.flatnonzero(data["sources"] == 1))
    replay = replay[
        stratified_order({"groups": data["groups"][replay.numpy()]}, fractions, generator)
    ]
    size = int(min(len(replay) / sources[0], len(new) / sources[1]))
    if size < 8:
        raise ValueError("insufficient mixed source data")
    order = torch.cat(
        (
            replay[: int(size * sources[0])],
            new[torch.randperm(len(new), generator=generator)[: int(size * sources[1])]],
        )
    )
    return order[torch.randperm(len(order), generator=generator)]


def forward(parameters: list, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pre-encoded own/opponent features; exact Q20/f64 accumulation on the measured CPU route."""
    table, bias, hw, hb, ow, ob = parameters
    if bool((features >= len(table)).any()):
        raise ValueError("encoded feature index outside model table")
    selected = table[features]
    selected = (
        selected + ((selected * ACCUMULATOR_GRID).round() / ACCUMULATOR_GRID - selected).detach()
    )
    qb = bias + ((bias * ACCUMULATOR_GRID).round() / ACCUMULATOR_GRID - bias).detach()
    mask = torch.arange(features.shape[2])[None, None, :] < lengths[:, :, None]
    accumulator = qb.double() + (selected.double() * mask[:, :, :, None]).sum(2)
    us, them = accumulator[:, 0].clamp(0, 1).float(), accumulator[:, 1].clamp(0, 1).float()
    hidden = (torch.cat((us, them, us * them), 1) @ hw + hb).clamp(0, 1)
    return (hidden @ ow + ob)[:, 0] * CP_PARAMETER_SCALE


def arrays(folder: Path, split: str) -> dict:
    return {
        name: np.load(folder / f"{split}-{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("features", "lengths", "targets")
    }


def batch(data: dict, indexes) -> tuple:
    if data["features"].dtype.kind not in "ui" or data["lengths"].dtype.kind not in "ui":
        raise ValueError("feature indexes/lengths must be integers")
    features = np.array(data["features"][indexes], dtype=np.int64)
    lengths = np.array(data["lengths"][indexes], dtype=np.int64)
    targets = np.array(data["targets"][indexes], dtype=np.float32)
    if targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("empty/nonfinite training targets")
    if (
        features.ndim != 3
        or features.shape[:2] != (len(targets), 2)
        or lengths.shape != features.shape[:2]
        or (lengths < 1).any()
        or (lengths > features.shape[2]).any()
        or (features < 0).any()
    ):
        raise ValueError("invalid encoded feature shape/length/index")
    return tuple(torch.from_numpy(value) for value in (features, lengths, targets))


def evaluate(parameters: list, data: dict, batch_size: int, *, indexes=None) -> dict:
    size = min(batch_size, 32)
    while True:
        try:
            return _evaluate(parameters, data, size, indexes=indexes)
        except (MemoryError, RuntimeError) as error:
            if not allocation_failure(error):
                raise
            error.__traceback__ = None
            exhausted = size == 1
        gc.collect()
        if exhausted:
            raise TrainingResourceWaitError("validation allocation failed at microbatch 1")
        size = max(1, size // 2)


def _evaluate(parameters: list, data: dict, batch_size: int, *, indexes=None) -> dict:
    count = len(data["targets"]) if indexes is None else len(indexes)
    total, absolute, square = 0.0, 0.0, 0.0
    severe_over = 0
    if count == 0:
        raise ValueError("empty development validation")
    with torch.no_grad():
        for start in range(0, count, batch_size):
            selection = slice(start, start + batch_size)
            features, lengths, y = batch(data, selection if indexes is None else indexes[selection])
            predicted = forward(parameters, features, lengths)
            total += float(functional.smooth_l1_loss(predicted / 600, y / 600, reduction="sum"))
            absolute += float((predicted - y).abs().sum())
            square += float(((predicted - y) ** 2).sum())
            severe_over += int(((predicted - y) > 600).sum())
    return {
        "loss": total / count,
        "cp_mae": absolute / count,
        "cp_rmse": math.sqrt(square / count),
        "positions": count,
        "overestimate_above_600cp_rate": severe_over / count,
    }


def _save_checkpoint(folder: Path, state: dict) -> None:
    stream = io.BytesIO()
    torch.save(state, stream)
    payload = stream.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    path = folder / f"checkpoint-{state['step']:06d}-{checksum}.pt"
    atomic(path, payload)
    atomic(
        folder / "resume.json",
        encoded({"path": path.name, "sha256": digest(path), "step": state["step"]}),
    )
    # Two complete states cover interrupted publication; best inference model is separate.
    previous = sorted(
        (candidate for candidate in folder.glob("checkpoint-*.pt") if candidate != path),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    for old in previous[1:]:
        handles = subprocess.run(
            ["lsof", "--", str(old)], capture_output=True, text=True, check=False
        )
        if old.is_symlink() or handles.returncode != 1 or handles.stdout or handles.stderr:
            continue
        record = {
            "path": old.name,
            "bytes": old.stat().st_size,
            "sha256": digest(old),
            "reason": "older owned resume state; two newer states and best export retained",
        }
        old.unlink()
        with (folder / "cleanup.jsonl").open("ab") as f:
            f.write(encoded(record) + b"\n")


def train(
    dataset: Path, folder: Path, config: dict, identity: dict, *, stop_after: int | None = None
) -> dict:
    """Development test is never loaded here; stop_after only exercises exact E2E resume."""
    if (
        config["device"] != "cpu"
        or min(
            config[k]
            for k in (
                "batch_size",
                "max_steps",
                "max_epochs",
                "validation_every",
                "patience",
                "threads",
            )
        )
        < 1
    ):
        raise ValueError("invalid bounded CPU training configuration")
    microbatch_size = config.get("microbatch_size", min(config["batch_size"], 32))
    if not isinstance(microbatch_size, int) or not 1 <= microbatch_size <= config["batch_size"]:
        raise ValueError("invalid microbatch size")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    fractions = config.get("sampling_fractions")
    source_fractions = config.get("source_fractions")
    loader = grouped_arrays if fractions is not None else arrays
    data, validation = loader(dataset, "train"), loader(dataset, "validation")
    n = len(data["targets"])
    if not n:
        raise ValueError("empty training split")
    model = Phase10VModel.read(Path(config["initial_model"]))
    if model.sha256 != config["initial_sha256"]:
        raise ValueError("initial model identity changed")
    parameters = torch_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config["seed"])
    step, epoch, offset, exposures, stale, best_step = 0, 0, 0, 0, 0, 0
    counts = torch.zeros(n, dtype=torch.int32)

    def next_order():
        return (
            mixed_order(data, fractions, source_fractions, generator)
            if source_fractions
            else stratified_order(data, fractions, generator)
            if fractions
            else torch.randperm(n, generator=generator)
        )

    order = next_order()
    baseline_groups = None
    best_loss = math.inf
    best_bytes = b""
    history = []
    train_loss_sum, train_rows = 0.0, 0
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "resume.json").exists():
        ref = json.loads((folder / "resume.json").read_text())
        path = folder / ref["path"]
        if path.parent != folder or path.is_symlink() or digest(path) != ref["sha256"]:
            raise ValueError("checkpoint hash or containment mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != identity or state["config"] != config:
            raise ValueError("checkpoint code/data/config identity changed; new run required")
        if state["step"] != ref["step"] or len(state["counts"]) != n:
            raise ValueError("checkpoint step/dataset size mismatch")
        with torch.no_grad():
            for parameter, value in zip(parameters, state["parameters"], strict=True):
                parameter.copy_(value)
        train_loss_sum, train_rows = state["train_loss_sum"], state["train_rows"]
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        step, epoch, offset, exposures, stale, best_step = (
            state[k] for k in ("step", "epoch", "offset", "exposures", "stale", "best_step")
        )
        best_loss, history, counts, order = (
            state[k] for k in ("best_loss", "history", "counts", "order")
        )
        microbatch_size = state.get("microbatch_size", microbatch_size)
        if not isinstance(microbatch_size, int) or not 1 <= microbatch_size <= config["batch_size"]:
            raise ValueError("invalid checkpoint microbatch size")
        baseline_groups = state.get("baseline_groups")
        best_bytes = state["best_model_bytes"]
        if hashlib.sha256(best_bytes).hexdigest() != state["best_sha256"]:
            raise ValueError("checkpoint best model checksum mismatch")
        Phase10VModel.from_bytes(best_bytes)
        best_path = folder / "best.osaval03"
        if best_path.exists() and digest(best_path) != state["best_sha256"]:
            # Keep the uncommitted export; restore only the checkpoint-bound best.
            recovery = folder / f"uncommitted-best-{digest(best_path)}.osaval03"
            best_path.rename(recovery)
            atomic(
                folder / "best-recovery.json",
                encoded(
                    {
                        "retained": recovery.name,
                        "restored_sha256": state["best_sha256"],
                        "checkpoint_step": step,
                    }
                ),
            )
        if not best_path.exists():
            atomic(best_path, best_bytes)
        if (
            counts.sum().item() != exposures
            or not 0 <= offset <= len(order)
            or len(order.unique()) != len(order)
            or bool(((order < 0) | (order >= n)).any())
            or (fractions is None and len(order) != n)
            or int(counts.max()) > config["max_epochs"]
        ):
            raise ValueError("checkpoint sampler/exposure mismatch")
        completed = folder / "training.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if result["status"] == "complete":
                if result["identity"] != identity or result["best_sha256"] != state["best_sha256"]:
                    raise ValueError("completed run identity mismatch")
                return result
    if fractions and baseline_groups is None:
        baseline_groups = evaluate_groups(
            torch_parameters(model),
            validation,
            config["batch_size"],
            source=0 if source_fractions else None,
        )
    initial_step = step
    started = time.monotonic()

    def persist() -> None:
        state = {
            "identity": identity,
            "config": config,
            "parameters": [p.detach() for p in parameters],
            "optimizer": optimizer.state_dict(),
            "sampler_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "step": step,
            "epoch": epoch,
            "offset": offset,
            "exposures": exposures,
            "stale": stale,
            "best_step": best_step,
            "best_loss": best_loss,
            "best_model_bytes": best_bytes,
            "best_sha256": hashlib.sha256(best_bytes).hexdigest(),
            "history": history,
            "counts": counts,
            "order": order,
            "train_loss_sum": train_loss_sum,
            "train_rows": train_rows,
            "microbatch_size": microbatch_size,
            "baseline_groups": baseline_groups,
        }
        try:
            _save_checkpoint(folder, state)
        except (MemoryError, RuntimeError) as error:
            if not allocation_failure(error):
                raise
            raise CheckpointPublicationError(
                f"checkpoint publication failed after step {step}; explicit repair required"
            ) from error

    def save(reason: str) -> dict:
        nonlocal best_loss, best_bytes, stale, best_step, train_loss_sum, train_rows
        groups = None
        if fractions:
            groups = (
                baseline_groups
                if not history
                else evaluate_groups(
                    parameters,
                    validation,
                    config["batch_size"],
                    source=0 if source_fractions else None,
                )
            )
            count = sum(group["positions"] for group in groups.values())
            metrics = {
                key: sum(group[key] * group["positions"] for group in groups.values()) / count
                for key in ("loss", "cp_mae", "overestimate_above_600cp_rate")
            }
            metrics["positions"] = count
            metrics["cp_rmse"] = math.sqrt(
                sum(group["cp_rmse"] ** 2 * group["positions"] for group in groups.values()) / count
            )
        else:
            metrics = evaluate(parameters, validation, config["batch_size"])
        eligible = True
        if groups:
            metrics["loss"] = sum(groups[g]["loss"] * fractions[i] for i, g in enumerate(GROUPS))
            eligible = all(
                groups[g]["loss"]
                <= baseline_groups[g]["loss"] * config["maximum_replay_regression_ratio"]
                for g in (GROUPS if source_fractions else ("general", "attack_end"))
            )
        new_metrics = None
        if source_fractions:
            new_metrics = evaluate(
                parameters,
                validation,
                config["batch_size"],
                indexes=np.flatnonzero(validation["sources"] == 1),
            )
            metrics["loss"] = (
                source_fractions[0] * metrics["loss"] + source_fractions[1] * new_metrics["loss"]
            )
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("nonfinite validation")
        improved = eligible and metrics["loss"] < best_loss * (
            1 - config["minimum_relative_improvement"]
        )
        current = _snapshot(parameters, model.seed)
        if improved:
            best_loss, stale, best_step = metrics["loss"], 0, step
            best_bytes = current.to_bytes()
        else:
            stale += 1
        event = {
            **metrics,
            "step": step,
            "epoch": epoch,
            "offset": offset,
            "exposures": exposures,
            "seen_unique_positions": int((counts > 0).sum()),
            "maximum_exposure": int(counts.max()),
            "train_loss": train_loss_sum / train_rows if train_rows else None,
            "best_step": best_step,
            "best_loss": best_loss,
            "stale_intervals": stale,
            "reason": reason,
            "elapsed_this_invocation_s": time.monotonic() - started,
            "model_sha256": current.sha256,
            "groups": groups,
            "replay_guard_eligible": eligible,
            "new_source": new_metrics,
        }
        history.append(event)
        train_loss_sum, train_rows = 0.0, 0
        persist()
        atomic(folder / "best.osaval03", best_bytes)
        atomic(folder / "progress.json", encoded(event))
        return event

    if not history:
        save("initial_validation")
    elif step % config["validation_every"] == 0 and history[-1]["step"] != step:
        # The update is durable even when validation was interrupted after it.
        save("validation")
    reason = "validation_patience" if stale >= config["patience"] else "max_steps"
    while (
        step < config["max_steps"] and epoch < config["max_epochs"] and stale < config["patience"]
    ):
        if (folder.parent / "STOP").exists():
            reason = "requested_stop"
            break
        if offset == len(order):
            epoch += 1
            if epoch == config["max_epochs"]:
                reason = "max_epochs"
                break
            order = next_order()
            offset = 0
        indexes = order[offset : offset + config["batch_size"]]
        warmup = min(1.0, (step + 1) / config["warmup_steps"])
        progress = min(1.0, step / config["max_steps"])
        lr = config["minimum_learning_rate"] + 0.5 * (
            config["learning_rate"] - config["minimum_learning_rate"]
        ) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr * warmup
        try:
            loss_sum, microbatch_size = accumulate(
                parameters, data, indexes.numpy(), microbatch_size
            )
        except TrainingResourceWaitError:
            # No optimizer mutation yet; retain all preceding successful updates.
            microbatch_size = 1
            persist()
            raise
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("missing/nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_norm_limit"], error_if_nonfinite=True
        )
        try:
            optimizer.step()
        except (MemoryError, RuntimeError) as error:
            if not allocation_failure(error):
                raise
            # AdamW may have partially mutated parameters: never publish this state.
            raise TrainingResourceWaitError(
                "optimizer allocation failed; reload checkpoint"
            ) from error
        try:
            counts[indexes] += 1
            offset += len(indexes)
            exposures += len(indexes)
            step += 1
            train_loss_sum += loss_sum
            train_rows += len(indexes)
            # Every committed optimizer update is durable before another can start.
            persist()
        except (MemoryError, RuntimeError) as error:
            if not allocation_failure(error):
                raise
            raise CheckpointPublicationError(
                "successful optimizer update lacks durable progress; explicit repair required"
            ) from error
        atomic(
            folder / "progress.json",
            encoded(
                {
                    "step": step,
                    "epoch": epoch,
                    "offset": offset,
                    "exposures": exposures,
                    "train_loss": loss_sum / len(indexes),
                    "microbatch_size": microbatch_size,
                    "effective_batch_size": len(indexes),
                    "reason": "optimizer_update",
                }
            ),
        )
        validation_due = step % config["validation_every"] == 0
        limited = stop_after is not None and step - initial_step >= stop_after
        if validation_due:
            save("validation")
            if stale >= config["patience"]:
                reason = "validation_patience"
                break
        if limited:
            return {"status": "paused_for_resume_check", "step": step, "exposures": exposures}
    if reason == "requested_stop":
        persist()
    elif history[-1]["step"] != step:
        save(reason)
    result = {
        "status": "stopped" if reason == "requested_stop" else "complete",
        "reason": reason,
        "step": step,
        "epochs_finished": min(config["max_epochs"], epoch + int(offset == len(order))),
        "unique_training_positions": n,
        "seen_unique_positions": int((counts > 0).sum()),
        "example_exposures": exposures,
        "maximum_exposure": int(counts.max()),
        "sampling_fractions": fractions,
        "source_fractions": source_fractions,
        "source_exposures": {
            name: int(counts[torch.from_numpy(data["sources"] == i)].sum())
            for i, name in enumerate(("replay", "new"))
        }
        if source_fractions
        else None,
        "group_exposures": {
            name: int(counts[torch.from_numpy(data["groups"] == i)].sum())
            for i, name in enumerate(GROUPS)
        }
        if fractions
        else None,
        "best_step": best_step,
        "best_sha256": digest(folder / "best.osaval03"),
        "identity": identity,
        "history": history,
        "controller": "off; old W256 labels do not certify the new evaluator",
    }
    atomic(folder / "training.json", encoded(result))
    return result
