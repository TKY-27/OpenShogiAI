"""All-layer OSAVAL03 learning with exact sampler/optimizer resume and bounded retention."""

from __future__ import annotations

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

GROUPS = ("general", "opening", "defense", "attack_end")


def grouped_arrays(folder: Path, split: str) -> dict:
    data = arrays(folder, split)
    data["groups"] = np.load(folder / f"{split}-groups.npy", allow_pickle=False)
    if data["groups"].shape != data["targets"].shape or not np.isin(data["groups"], range(4)).all():
        raise ValueError("invalid sampling groups")
    return data


def evaluate_groups(parameters: list, data: dict, batch_size: int) -> dict:
    result = {}
    for i, name in enumerate(GROUPS):
        indexes = np.flatnonzero(data["groups"] == i)
        if not len(indexes):
            raise ValueError(f"missing independent validation group: {name}")
        subset = {k: v[indexes] for k, v in data.items() if k != "groups"}
        result[name] = evaluate(parameters, subset, batch_size)
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


def forward(parameters: list, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pre-encoded own/opponent features; exact Q20/f64 accumulation on the measured CPU route."""
    table, bias, hw, hb, ow, ob = parameters
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
    return (
        torch.from_numpy(np.array(data["features"][indexes], dtype=np.int64)),
        torch.from_numpy(np.array(data["lengths"][indexes], dtype=np.int64)),
        torch.from_numpy(np.array(data["targets"][indexes], dtype=np.float32)),
    )


def evaluate(parameters: list, data: dict, batch_size: int) -> dict:
    total, absolute, square, count = 0.0, 0.0, 0.0, len(data["targets"])
    severe_over = 0
    if count == 0:
        raise ValueError("empty development validation")
    with torch.no_grad():
        for start in range(0, count, batch_size):
            features, lengths, y = batch(data, slice(start, start + batch_size))
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
    path = folder / f"checkpoint-{state['step']:06d}.pt"
    atomic(path, stream.getvalue())
    atomic(
        folder / "resume.json",
        encoded({"path": path.name, "sha256": digest(path), "step": state["step"]}),
    )
    # Two complete states cover interrupted publication; best inference model is separate.
    obsolete = sorted(folder.glob("checkpoint-*.pt"))[:-2]
    for old in obsolete:
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
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    fractions = config.get("sampling_fractions")
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
            stratified_order(data, fractions, generator)
            if fractions
            else torch.randperm(n, generator=generator)
        )

    order = next_order()
    baseline_groups = (
        evaluate_groups(parameters, validation, config["batch_size"]) if fractions else None
    )
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
        }
        _save_checkpoint(folder, state)

    def save(reason: str) -> dict:
        nonlocal best_loss, best_bytes, stale, best_step, train_loss_sum, train_rows
        metrics = evaluate(parameters, validation, config["batch_size"])
        groups = (
            evaluate_groups(parameters, validation, config["batch_size"]) if fractions else None
        )
        eligible = True
        if groups:
            metrics["loss"] = sum(groups[g]["loss"] * fractions[i] for i, g in enumerate(GROUPS))
            eligible = all(
                groups[g]["loss"]
                <= baseline_groups[g]["loss"] * config["maximum_replay_regression_ratio"]
                for g in ("general", "attack_end")
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
        }
        history.append(event)
        train_loss_sum, train_rows = 0.0, 0
        persist()
        atomic(folder / "best.osaval03", best_bytes)
        atomic(folder / "progress.json", encoded(event))
        return event

    if not history:
        save("initial_validation")
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
        features, lengths, y = batch(data, indexes.numpy())
        warmup = min(1.0, (step + 1) / config["warmup_steps"])
        progress = min(1.0, step / config["max_steps"])
        lr = config["minimum_learning_rate"] + 0.5 * (
            config["learning_rate"] - config["minimum_learning_rate"]
        ) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr * warmup
        optimizer.zero_grad(set_to_none=True)
        predicted = forward(parameters, features, lengths)
        loss = functional.smooth_l1_loss(predicted / 600, y / 600)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("missing/nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_norm_limit"], error_if_nonfinite=True
        )
        optimizer.step()
        counts[indexes] += 1
        offset += len(indexes)
        exposures += len(indexes)
        step += 1
        train_loss_sum += float(loss.detach()) * len(indexes)
        train_rows += len(indexes)
        validation_due = step % config["validation_every"] == 0
        limited = stop_after is not None and step - initial_step >= stop_after
        if validation_due:
            save("validation")
            if stale >= config["patience"]:
                reason = "validation_patience"
                break
        if limited:
            persist()
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
