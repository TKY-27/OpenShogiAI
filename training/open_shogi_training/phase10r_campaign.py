"""Streaming Phase 10R rung execution over immutable preparation artifacts.

This module deliberately implements only the factual-replay stages that can be
proven from the approved v2 population.  Apery-dependent stages fail closed until
the rights-gated teacher installation and exact identity are present.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import random
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from open_shogi_training.checkpoint_safety import (
    CheckpointSafetyError,
    deserialize,
    read_verified_bytes,
    verify_receipt,
    write_receipt,
)
from open_shogi_training.phase10r import _load_yaml
from open_shogi_training.phase10r_execution import (
    Phase10RExecutionError,
    _data_root,
    _json_bytes,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    preparation_manifest_path,
    validate_preparation,
)
from open_shogi_training.phase10r_model import (
    VARIANT_PAIR,
    HistoryFacts,
    infer_osaval02,
    parse_osaval02,
)
from open_shogi_training.phase10r_training import (
    CHECKPOINT_SCHEMA,
    Phase10RExample,
    Phase10RModel,
    Phase10RTrainingError,
    export_osaval02_artifact,
    loss_for_examples,
    resource_snapshot,
    select_device,
)

CAMPAIGN_SCHEMA: Final = "open_shogiai_phase10r_campaign/v1"
EVALUATION_SCHEMA: Final = "open_shogiai_phase10r_evaluation/v1"
SELECTION_SCHEMA: Final = "open_shogiai_phase10r_selection/v1"
PREPARATION_MANIFEST_SCHEMA: Final = "open_shogiai_phase10r_preparation/v2"
TRAINING_SEED: Final = 20_260_729
TRAINING_BATCH_SIZE: Final = 128
STAGE_EPOCHS: Final = 1
CHECKPOINT_INTERVAL_STEPS: Final = 1_000
CHECKPOINT_INTERVAL_EXAMPLES: Final = 128_000
RSS_TARGET_BYTES: Final = 16 * 1024**3
MINIMUM_FREE_BYTES: Final = 100 * 1024**3
TRAIN_VARIANTS: Final = (VARIANT_PAIR,)
STAGE_ONE: Final = "representation_policy_pretraining"
STAGE_TWO: Final = "source_specific_wdl_value_pretraining"
TRAINING_SPLIT: Final = "train"
EVALUATION_SPLITS: Final = ("validation", "source_held_out")
PARITY_CORPUS: Final = Path("tests/fixtures/osaval02/parity-corpus.json")
OSAVAL02_PARITY_SCHEMA: Final = "open_shogiai_osaval02_parity/v1"
OSAVAL02_IDENTITY_FIELDS: Final = frozenset(
    {
        "formatVersion",
        "variantId",
        "quantization",
        "parameterCount",
        "artifactBytes",
        "artifactSha256",
        "weightPayloadSha256",
        "featureSchemaSha256",
        "architectureConfigSha256",
        "targetSemanticsSha256",
        "inputNormalizationSha256",
        "datasetManifestSha256",
        "moveIndexSha256",
        "exporterVersion",
        "gitCommit",
        "trainingRunReference",
    }
)
TEACHER_EXPECTED_SHA256: Final = "8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403"


class Phase10RCampaignError(RuntimeError):
    """Raised when a campaign operation cannot prove its frozen prerequisites."""


def _failure(message: str, error: Exception | None = None) -> Phase10RCampaignError:
    if error is None:
        return Phase10RCampaignError(message)
    return Phase10RCampaignError(f"{message}: {error}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(dict(value))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise Phase10RCampaignError(f"temporary campaign path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise _failure(f"cannot publish campaign JSON: {path}", error) from error


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _manifest_identity(root: Path, scale: str) -> tuple[Path, dict[str, Any], str]:
    manifest_path = preparation_manifest_path(root, scale)
    try:
        manifest = validate_preparation(root, scale)
    except Phase10RExecutionError as error:
        raise _failure("preparation manifest is unavailable", error) from error
    if (
        manifest.get("schema") != PREPARATION_MANIFEST_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("scale") != scale
    ):
        raise Phase10RCampaignError("preparation manifest identity is invalid")
    declared_digest = manifest.get("manifest_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise Phase10RCampaignError("preparation manifest lacks its body digest")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if _sha256_bytes(_json_bytes(body)) != declared_digest:
        raise Phase10RCampaignError("preparation manifest body digest mismatches")
    streamed_examples = manifest.get("streamed_examples")
    if not isinstance(streamed_examples, int) or streamed_examples <= 0:
        raise Phase10RCampaignError("preparation manifest has no positive stream count")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise Phase10RCampaignError("preparation manifest has no file inventory")
    train_info = files.get("train.jsonl")
    if not isinstance(train_info, Mapping):
        raise Phase10RCampaignError("preparation manifest has no train stream")
    train_path = manifest_path.parent / "train.jsonl"
    try:
        train_digest = _sha256_file(train_path)
    except Phase10RExecutionError as error:
        raise _failure("prepared train stream is unavailable", error) from error
    if train_info.get("sha256") != train_digest or train_info.get("rows") != streamed_examples:
        raise Phase10RCampaignError("prepared train stream identity mismatches its manifest")
    return manifest_path, manifest, declared_digest


def _preparation_file(manifest_path: Path, manifest: Mapping[str, Any], name: str) -> Path:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(name), Mapping):
        raise Phase10RCampaignError(f"preparation file is not recorded: {name}")
    path = manifest_path.parent / name
    if path.is_symlink() or not path.is_file():
        raise Phase10RCampaignError(f"preparation file is not a regular file: {path}")
    if _sha256_file(path) != files[name].get("sha256"):
        raise Phase10RCampaignError(f"preparation file hash mismatches its manifest: {name}")
    return path


def _row_from_line(line: str, stream_index: int | None) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise _failure("prepared stream contains invalid JSON", error) from error
    if not isinstance(value, dict):
        raise Phase10RCampaignError("prepared stream row is not an object")
    if stream_index is not None:
        raw_targets = value.get("raw_targets")
        if not isinstance(raw_targets, Mapping) or raw_targets.get("stream_index") != stream_index:
            raise Phase10RCampaignError(f"prepared stream index drifted at row {stream_index}")
    return value


def _example_from_row(
    row: Mapping[str, Any], *, expected_split: str | None = TRAINING_SPLIT
) -> Phase10RExample:
    try:
        example = Phase10RExample.from_mapping(row)
        example.validate()
    except (Phase10RTrainingError, TypeError, ValueError) as error:
        raise _failure("prepared training row failed validation", error) from error
    if expected_split is not None and example.split != expected_split:
        raise Phase10RCampaignError("prepared row contains an unexpected split")
    return example


def _stream_batches(
    path: Path,
    *,
    expected_rows: int,
    start_cursor: int,
    batch_size: int,
) -> Iterator[tuple[list[Phase10RExample], int]]:
    if not 0 <= start_cursor <= expected_rows:
        raise Phase10RCampaignError("checkpoint cursor is outside the preparation stream")
    batch: list[Phase10RExample] = []
    cursor = start_cursor
    observed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            observed = line_number + 1
            if line_number < start_cursor:
                continue
            if not line.strip():
                raise Phase10RCampaignError(f"prepared stream contains a blank row: {line_number}")
            row = _row_from_line(line, line_number)
            batch.append(_example_from_row(row))
            cursor = line_number + 1
            if len(batch) == batch_size:
                yield batch, cursor
                batch = []
    if observed != expected_rows:
        raise Phase10RCampaignError(
            f"prepared stream row count mismatches its manifest: {observed} != {expected_rows}"
        )
    if batch:
        yield batch, cursor


def _masked_stage_batch(
    examples: Sequence[Phase10RExample], stage_id: str
) -> list[Phase10RExample]:
    if stage_id == STAGE_ONE:
        return [
            replace(
                example,
                wdl=None,
                wdl_mask=False,
                uncertainty_mask=False,
            )
            for example in examples
            if example.policy_mask
        ]
    if stage_id == STAGE_TWO:
        return [replace(example, played_move=None) for example in examples if example.wdl_mask]
    raise Phase10RCampaignError(f"unsupported factual campaign stage: {stage_id}")


def _checkpoint_payload(
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    manifest_sha256: str,
    stage_id: str,
    variant_id: str,
    step: int,
    cursor: int,
    metrics: Mapping[str, Any],
    completed: bool,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "variant_id": variant_id,
        "manifest_sha256": manifest_sha256,
        "seed": TRAINING_SEED,
        "stage_id": stage_id,
        "step": step,
        "stream_index": cursor,
        "completed": completed,
        "metrics": dict(metrics),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }


def _load_verified_checkpoint(path: Path) -> Any:
    """Verify the save-time receipt, then deserialize exactly those bytes.

    The file is read once and hashed in memory, so the bytes torch.load sees
    are the bytes the receipt vouches for even if the path is re-pointed
    afterwards.
    """

    try:
        declared = verify_receipt(path, error=Phase10RCampaignError)
        payload_bytes = read_verified_bytes(
            path,
            declared,
            error=Phase10RCampaignError,
            mismatch_message="checkpoint digest mismatches its receipt",
        )
        return deserialize(payload_bytes, error=CheckpointSafetyError)
    except CheckpointSafetyError as error:
        raise Phase10RCampaignError(str(error)) from error


def _save_checkpoint(
    path: Path,
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    manifest_sha256: str,
    stage_id: str,
    variant_id: str,
    step: int,
    cursor: int,
    metrics: Mapping[str, Any],
    completed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        manifest_sha256=manifest_sha256,
        stage_id=stage_id,
        variant_id=variant_id,
        step=step,
        cursor=cursor,
        metrics=metrics,
        completed=completed,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        raise _failure(f"cannot publish checkpoint: {path}", error) from error
    write_receipt(path, error=lambda message: _failure(message))


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_checkpoint(
    path: Path,
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    manifest_sha256: str,
    stage_id: str,
    variant_id: str,
    expected_rows: int,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RCampaignError(f"checkpoint is not a regular file: {path}")
    try:
        payload = _load_verified_checkpoint(path)
        if not isinstance(payload, dict):
            raise Phase10RCampaignError("checkpoint payload is not an object")
        for key, expected in (
            ("schema", CHECKPOINT_SCHEMA),
            ("variant_id", variant_id),
            ("manifest_sha256", manifest_sha256),
            ("seed", TRAINING_SEED),
            ("stage_id", stage_id),
        ):
            if payload.get(key) != expected:
                raise Phase10RCampaignError(f"checkpoint identity mismatch: {key}")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
        scheduler.load_state_dict(payload["scheduler_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
    except Phase10RCampaignError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        raise _failure(f"checkpoint cannot be resumed: {path}", error) from error
    step = payload.get("step")
    cursor = payload.get("stream_index")
    metrics = payload.get("metrics")
    completed = payload.get("completed")
    if (
        not isinstance(step, int)
        or step < 0
        or not isinstance(cursor, int)
        or not 0 <= cursor <= expected_rows
        or not isinstance(metrics, dict)
        or not isinstance(completed, bool)
    ):
        raise Phase10RCampaignError("checkpoint progress is invalid")
    if completed and cursor != expected_rows:
        raise Phase10RCampaignError("completed checkpoint does not cover the preparation stream")
    return {
        "step": step,
        "cursor": cursor,
        "metrics": {str(key): value for key, value in metrics.items()},
        "completed": completed,
    }


def _resource_guard(data_root: Path) -> dict[str, int | bool]:
    snapshot = resource_snapshot(data_root, minimum_free_bytes=MINIMUM_FREE_BYTES)
    if not snapshot["disk_passed"]:
        raise Phase10RCampaignError("free disk crossed the 100 GiB campaign floor")
    if int(snapshot["peak_rss_bytes"]) > RSS_TARGET_BYTES:
        raise Phase10RCampaignError("RSS exceeded the frozen 16 GiB target")
    return snapshot


def _stage_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    active = int(result.get("active_examples", 0))
    steps = int(result.get("optimizer_steps", 0))
    if active:
        result["mean_loss_per_active_example"] = float(result.get("loss_sum", 0.0)) / active
    if steps:
        result["mean_batch_loss"] = float(result.get("loss_sum", 0.0)) / steps
    return result


def _run_stage(
    *,
    model: Phase10RModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    stage_id: str,
    stage_dir: Path,
    train_path: Path,
    expected_rows: int,
    manifest_sha256: str,
    data_root: Path,
    resume: bool,
) -> dict[str, Any]:
    checkpoint = stage_dir / "last.pt"
    best_checkpoint = stage_dir / "best.pt"
    checkpoint_was_present = checkpoint.exists()
    if checkpoint_was_present and not resume:
        raise Phase10RCampaignError(f"checkpoint exists; --resume is required: {checkpoint}")
    state = {
        "step": 0,
        "cursor": 0,
        "metrics": {
            "loss_sum": 0.0,
            "active_examples": 0,
            "optimizer_steps": 0,
            "stream_rows_consumed": 0,
        },
        "completed": False,
    }
    if checkpoint.exists():
        state = _load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            manifest_sha256=manifest_sha256,
            stage_id=stage_id,
            variant_id=model.variant_id,
            expected_rows=expected_rows,
        )
    best_loss = float(state["metrics"].get("best_batch_loss", float("inf")))
    if best_checkpoint.is_symlink() or (best_checkpoint.exists() and not best_checkpoint.is_file()):
        raise Phase10RCampaignError(f"best checkpoint is not a regular file: {best_checkpoint}")
    if state["completed"]:
        if not best_checkpoint.exists():
            _save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                manifest_sha256=manifest_sha256,
                stage_id=stage_id,
                variant_id=model.variant_id,
                step=state["step"],
                cursor=state["cursor"],
                metrics=state["metrics"],
                completed=True,
            )
        return {
            "stage_id": stage_id,
            "status": "passed",
            "resumed": True,
            "steps": state["step"],
            "stream_rows_consumed": state["cursor"],
            "metrics": _stage_metrics(state["metrics"]),
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256_file(checkpoint),
            "best_checkpoint": best_checkpoint,
            "best_checkpoint_sha256": _sha256_file(best_checkpoint),
        }
    stage_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    cursor = int(state["cursor"])
    step = int(state["step"])
    metrics = dict(state["metrics"])
    state_metrics = metrics
    last_checkpoint_step = step
    last_checkpoint_cursor = cursor
    started = time.monotonic()

    def save_progress(*, completed: bool) -> None:
        nonlocal best_loss
        best_updated = False
        latest_loss = state_metrics.get("last_batch_loss")
        if (
            isinstance(latest_loss, (int, float))
            and math.isfinite(float(latest_loss))
            and (not best_checkpoint.exists() or float(latest_loss) < best_loss)
        ):
            best_loss = float(latest_loss)
            state_metrics["best_batch_loss"] = best_loss
            best_updated = True
        _save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            manifest_sha256=manifest_sha256,
            stage_id=stage_id,
            variant_id=model.variant_id,
            step=step,
            cursor=cursor,
            metrics=state_metrics,
            completed=completed,
        )
        if best_updated or not best_checkpoint.exists():
            _save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                manifest_sha256=manifest_sha256,
                stage_id=stage_id,
                variant_id=model.variant_id,
                step=step,
                cursor=cursor,
                metrics=state_metrics,
                completed=completed,
            )

    for epoch in range(STAGE_EPOCHS):
        if epoch > 0:
            cursor = 0
        try:
            batches = _stream_batches(
                train_path,
                expected_rows=expected_rows,
                start_cursor=cursor,
                batch_size=TRAINING_BATCH_SIZE,
            )
            for batch, next_cursor in batches:
                active_batch = _masked_stage_batch(batch, stage_id)
                cursor = next_cursor
                if active_batch:
                    optimizer.zero_grad(set_to_none=True)
                    loss, batch_metrics = loss_for_examples(model, active_batch)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=10.0
                    )
                    if not bool(torch.isfinite(gradient_norm).item()):
                        raise Phase10RCampaignError("gradient norm is NaN or Inf")
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    metrics["optimizer_steps"] = int(metrics.get("optimizer_steps", 0)) + 1
                    metrics["active_examples"] = int(metrics.get("active_examples", 0)) + len(
                        active_batch
                    )
                    metrics["loss_sum"] = float(metrics.get("loss_sum", 0.0)) + float(
                        loss.detach().cpu()
                    )
                    metrics["last_batch_loss"] = float(loss.detach().cpu())
                    metrics["last_gradient_norm"] = float(gradient_norm.detach().cpu())
                    for name, value in batch_metrics.items():
                        if name.endswith("_weight"):
                            continue
                        metrics[f"last_{name}"] = float(value)
                metrics["stream_rows_consumed"] = cursor
                if step - last_checkpoint_step >= CHECKPOINT_INTERVAL_STEPS or (
                    cursor - last_checkpoint_cursor >= CHECKPOINT_INTERVAL_EXAMPLES
                ):
                    save_progress(completed=False)
                    last_checkpoint_step = step
                    last_checkpoint_cursor = cursor
                if step == 0 or step % 64 == 0:
                    _resource_guard(data_root)
        except Exception:
            save_progress(completed=False)
            raise
    save_progress(completed=True)
    resource_after = _resource_guard(data_root)
    return {
        "stage_id": stage_id,
        "status": "passed",
        "resumed": checkpoint_was_present and resume,
        "epochs": STAGE_EPOCHS,
        "steps": step,
        "stream_rows_consumed": expected_rows,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": _stage_metrics(metrics),
        "resources": resource_after,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_sha256": _sha256_file(best_checkpoint),
    }


def _new_optimizer(
    model: Phase10RModel,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise Phase10RCampaignError("stage optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return optimizer, scheduler


def _artifact_for_model(
    model: Phase10RModel,
    *,
    output: Path,
    manifest_sha256: str,
    git_commit: str,
    reference: str,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise Phase10RCampaignError(f"model artifact is not a regular file: {output}")
        parsed = parse_osaval02(output.read_bytes())
        if (
            parsed.variant_id != model.variant_id
            or parsed.dataset_manifest_sha256 != manifest_sha256
        ):
            raise Phase10RCampaignError("existing model artifact identity mismatches the rung")
        return {
            "status": "passed",
            "path": output,
            "sha256": _sha256_file(output),
            "variant_id": parsed.variant_id,
            "quantization": parsed.quantization,
            "resumed": True,
        }
    try:
        result = export_osaval02_artifact(
            model,
            output,
            quantization="float32",
            dataset_manifest_sha256=manifest_sha256,
            training_run_reference=reference,
            git_commit=git_commit,
        )
    except (OSError, RuntimeError, ValueError, Phase10RTrainingError) as error:
        raise _failure("OSAVAL02 candidate export failed", error) from error
    result["path"] = Path(result["path"])
    return result


def train_variant(
    root: Path, scale: str, variant: str, *, resume: bool, git_commit: str
) -> dict[str, Any]:
    if variant not in TRAIN_VARIANTS:
        raise Phase10RCampaignError(f"unsupported trainable variant: {variant}")
    manifest_path, manifest, manifest_sha256 = _manifest_identity(root, scale)
    train_path = _preparation_file(manifest_path, manifest, "train.jsonl")
    expected_rows = int(manifest["streamed_examples"])
    data_root = _data_root(root)
    output_dir = data_root / "checkpoints" / "phase10r" / scale / variant
    stage_one_dir = output_dir / STAGE_ONE
    stage_two_dir = output_dir / STAGE_TWO
    if not resume and any(
        path.exists() for path in (stage_one_dir / "last.pt", stage_two_dir / "last.pt")
    ):
        raise Phase10RCampaignError("campaign checkpoints exist; --resume is required")

    _resource_guard(data_root)
    _seed_everything(TRAINING_SEED)
    device_receipt = select_device("auto", allow_cpu_fallback=True)
    device = torch.device(device_receipt.selected)
    model = Phase10RModel(variant, seed=TRAINING_SEED).to(device)
    optimizer, scheduler = _new_optimizer(model)
    stage_one = _run_stage(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stage_id=STAGE_ONE,
        stage_dir=stage_one_dir,
        train_path=train_path,
        expected_rows=expected_rows,
        manifest_sha256=manifest_sha256,
        data_root=data_root,
        resume=resume,
    )

    model.freeze_policy_embedding(True)
    optimizer, scheduler = _new_optimizer(model)
    stage_two = _run_stage(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stage_id=STAGE_TWO,
        stage_dir=stage_two_dir,
        train_path=train_path,
        expected_rows=expected_rows,
        manifest_sha256=manifest_sha256,
        data_root=data_root,
        resume=resume,
    )
    model.eval()
    artifact = _artifact_for_model(
        model,
        output=output_dir / f"{variant}.osaval02",
        manifest_sha256=manifest_sha256,
        git_commit=git_commit,
        reference=f"phase10r-{scale}-{variant}-stage2",
    )
    result: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "passed",
        "scale": scale,
        "variant": variant,
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "device": device_receipt.as_dict(),
        "stages": [stage_one, stage_two],
        "artifact": artifact,
        "teacher_dependent_stages": "not_started",
        "teacher_stop_reason": "Apery identity is required before stage 3.",
    }
    _atomic_json(output_dir / "training-summary.json", _json_safe(result))
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _history_from_fixture(value: object) -> HistoryFacts:
    if value is None:
        return HistoryFacts()
    if not isinstance(value, Mapping):
        raise Phase10RCampaignError("parity corpus history is invalid")
    return HistoryFacts(
        available=bool(value.get("available", False)),
        repetition_count=int(value.get("repetitionCount", 1)),
        continuous_check_by_us=bool(value.get("continuousCheckByUs", False)),
        continuous_check_by_them=bool(value.get("continuousCheckByThem", False)),
    )


def _compare_json(left: Any, right: Any, path: str = "root") -> tuple[float, list[str]]:
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0, [] if left == right else [path]
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return 0.0, [path]
        delta = abs(float(left) - float(right))
        return delta, [] if math.isclose(
            float(left), float(right), rel_tol=1.0e-8, abs_tol=1.0e-8
        ) else [path]
    if isinstance(left, dict) and isinstance(right, dict):
        mismatches: list[str] = []
        maximum = 0.0
        if left.keys() != right.keys():
            mismatches.append(path)
        for key in left.keys() & right.keys():
            delta, paths = _compare_json(left[key], right[key], f"{path}.{key}")
            maximum = max(maximum, delta)
            mismatches.extend(paths)
        return maximum, mismatches
    if isinstance(left, list) and isinstance(right, list):
        mismatches = []
        maximum = 0.0
        if len(left) != len(right):
            mismatches.append(path)
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            delta, paths = _compare_json(left_item, right_item, f"{path}[{index}]")
            maximum = max(maximum, delta)
            mismatches.extend(paths)
        return maximum, mismatches
    return 0.0, [] if left == right else [path]


def _run_json_command(root: Path, command: Sequence[str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command), cwd=root, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _failure("cross-runtime command failed to start or timed out", error) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_096:]
        raise Phase10RCampaignError(f"cross-runtime command failed: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise _failure("cross-runtime command emitted invalid JSON", error) from error
    if not isinstance(value, dict):
        raise Phase10RCampaignError("cross-runtime command emitted a non-object")
    return value


def _validate_candidate_parity_document(document: Mapping[str, Any], runtime: str) -> None:
    if "schema" not in document:
        raise Phase10RCampaignError(f"{runtime} candidate parity schema is missing")
    schema = document["schema"]
    if not isinstance(schema, str) or schema != OSAVAL02_PARITY_SCHEMA:
        raise Phase10RCampaignError(
            f"{runtime} candidate parity schema is incompatible: {schema!r}"
        )
    if set(document) != {"schema", "modelIdentity", "fixtures"}:
        raise Phase10RCampaignError(f"{runtime} candidate parity envelope is incompatible")
    identity = document["modelIdentity"]
    if not isinstance(identity, dict) or set(identity) != OSAVAL02_IDENTITY_FIELDS:
        raise Phase10RCampaignError(f"{runtime} candidate model identity is incompatible")
    if identity.get("formatVersion") != 2:
        raise Phase10RCampaignError(f"{runtime} candidate format version is unsupported")
    if not isinstance(document["fixtures"], list):
        raise Phase10RCampaignError(f"{runtime} candidate parity fixtures are incompatible")


def _candidate_parity(root: Path, artifact_path: Path, model_path: Path) -> dict[str, Any]:
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise Phase10RCampaignError(f"candidate artifact is not a regular file: {artifact_path}")
    corpus_path = root / PARITY_CORPUS
    corpus = _read_json(corpus_path)
    if corpus.get("schema") != "open_shogiai_osaval02_parity_corpus/v1":
        raise Phase10RCampaignError("candidate parity corpus schema is invalid")
    native = _run_json_command(
        root,
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "-p",
            "open-shogi-core",
            "--example",
            "osaval02_infer",
            "--",
            str(artifact_path),
            str(corpus_path),
        ],
        300,
    )
    wasm = _run_json_command(
        root,
        [
            "node",
            "--experimental-default-type=module",
            "scripts/osaval02_wasm_infer.mjs",
            str(artifact_path),
            str(corpus_path),
        ],
        300,
    )
    _validate_candidate_parity_document(native, "native")
    _validate_candidate_parity_document(wasm, "Wasm")
    native_wasm_delta, native_wasm_mismatches = _compare_json(native, wasm)
    if native_wasm_mismatches:
        raise Phase10RCampaignError(
            f"candidate native/Wasm parity mismatch: {native_wasm_mismatches[:3]}"
        )
    if model_path.is_symlink() or not model_path.is_file():
        raise Phase10RCampaignError(f"candidate model is not a regular file: {model_path}")
    parsed = parse_osaval02(model_path.read_bytes())
    fixtures = corpus.get("fixtures")
    native_fixtures = native.get("fixtures")
    if not isinstance(fixtures, list) or not isinstance(native_fixtures, list):
        raise Phase10RCampaignError("candidate parity fixtures are invalid")
    by_id = {item.get("fixtureId"): item for item in fixtures if isinstance(item, dict)}
    native_ids = {item.get("fixtureId") for item in native_fixtures if isinstance(item, dict)}
    if native_ids != set(by_id):
        raise Phase10RCampaignError("candidate parity fixture coverage changed")
    python_mismatches: list[str] = []
    python_delta = 0.0
    for native_fixture in native_fixtures:
        if not isinstance(native_fixture, dict):
            raise Phase10RCampaignError("native parity fixture is invalid")
        fixture = by_id.get(native_fixture.get("fixtureId"))
        inference = native_fixture.get("inference")
        if not isinstance(fixture, dict) or not isinstance(inference, dict):
            raise Phase10RCampaignError("native parity fixture identity is invalid")
        native_legal_moves = inference.get("legalMoves")
        if not isinstance(native_legal_moves, list):
            raise Phase10RCampaignError("native parity legal-move output is invalid")
        legal_moves = [
            item.get("move")
            for item in native_legal_moves
            if isinstance(item, dict) and isinstance(item.get("move"), str)
        ]
        reference = infer_osaval02(
            parsed,
            str(fixture["sfen"]),
            legal_moves,
            _history_from_fixture(fixture.get("history")),
        )
        delta, mismatches = _compare_json(inference, reference, str(fixture["fixtureId"]))
        python_delta = max(python_delta, delta)
        python_mismatches.extend(mismatches)
    if python_mismatches:
        raise Phase10RCampaignError(
            f"candidate Python/native parity mismatch: {python_mismatches[:3]}"
        )
    return {
        "status": "passed",
        "native_wasm_max_abs_delta": native_wasm_delta,
        "python_native_max_abs_delta": python_delta,
        "fixtures": len(native_fixtures),
        "model_sha256": _sha256_file(model_path),
    }


def _incremental_parity(root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["cargo", "test", "--locked", "-p", "open-shogi-core", "-p", "open-shogi-wasm"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _failure(
            "incremental/runtime parity test failed to start or timed out", error
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_096:]
        raise Phase10RCampaignError(f"incremental/runtime parity tests failed: {detail}")
    return {
        "status": "passed",
        "command": "cargo test --locked -p open-shogi-core -p open-shogi-wasm",
    }


def _evaluate_file(
    path: Path,
    *,
    expected_rows: int,
    model: Phase10RModel,
    data_root: Path,
) -> dict[str, Any]:
    ranges = _evaluation_file_ranges(path, expected_rows)
    global _EVALUATION_WORKER_MODEL
    _EVALUATION_WORKER_MODEL = model
    try:
        with ProcessPoolExecutor(
            max_workers=len(ranges),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_initialize_evaluation_worker,
        ) as pool:
            partials = list(
                pool.map(
                    _evaluate_file_range,
                    ((path, start, end, first_row) for start, end, first_row in ranges),
                )
            )
    except (OSError, RuntimeError) as error:
        raise _failure("parallel source-held-out evaluation failed", error) from error
    finally:
        _EVALUATION_WORKER_MODEL = None
    result = _merge_evaluation_partials(partials, expected_rows)
    _resource_guard(data_root)
    return result


_EVALUATION_WORKER_MODEL: Phase10RModel | None = None


def _initialize_evaluation_worker() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _evaluation_file_ranges(path: Path, expected_rows: int) -> list[tuple[int, int, int]]:
    if expected_rows <= 0:
        raise Phase10RCampaignError("source-held-out evaluation requires positive rows")
    offsets = [0]
    with path.open("rb") as handle:
        while handle.readline():
            offsets.append(handle.tell())
    if len(offsets) != expected_rows + 1:
        raise Phase10RCampaignError(
            f"evaluation row count mismatches its manifest: {len(offsets) - 1} != {expected_rows}"
        )
    worker_count = min(8, expected_rows)
    ranges = []
    for worker in range(worker_count):
        first_row = (expected_rows * worker) // worker_count
        last_row = (expected_rows * (worker + 1)) // worker_count
        ranges.append((offsets[first_row], offsets[last_row], first_row))
    return ranges


def _evaluate_file_range(
    arguments: tuple[Path, int, int, int],
) -> dict[str, Any]:
    model = _EVALUATION_WORKER_MODEL
    if model is None:
        raise Phase10RCampaignError("source-held-out worker model is not initialized")
    path, start, end, first_row = arguments
    with path.open(encoding="utf-8") as handle:
        handle.seek(start)
        lines = []
        while handle.tell() < end:
            line = handle.readline()
            if not line:
                break
            lines.append(line)
    model.eval()
    with torch.no_grad():
        return _evaluate_rows(lines, path=path, model=model, first_row=first_row)


def _evaluate_rows(
    lines: Sequence[str],
    *,
    path: Path,
    model: Phase10RModel,
    first_row: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    policy_correct = 0
    policy_total = 0
    policy_nll = 0.0
    wdl_brier = 0.0
    wdl_nll = 0.0
    wdl_total = 0
    calibration_bins = [{"count": 0, "confidence": 0.0, "correct": 0.0} for _ in range(10)]
    model.eval()
    for offset, line in enumerate(lines):
        line_number = first_row + offset
        if not line.strip():
            raise Phase10RCampaignError(
                f"evaluation file contains a blank row: {path}:{line_number}"
            )
        row = _row_from_line(line, None)
        split = row.get("split")
        if split not in EVALUATION_SPLITS:
            raise Phase10RCampaignError(f"evaluation file contains a forbidden split: {split}")
        example = _example_from_row(row, expected_split=None)
        counts[str(example.source)] += 1
        output = model.forward_example(example)
        if example.policy_mask:
            probabilities = torch.softmax(output["policy_logits"], dim=0)
            target_index = example.legal_moves.index(example.played_move)
            prediction = int(torch.argmax(probabilities).item())
            policy_correct += int(prediction == target_index)
            policy_total += 1
            policy_nll += float(-torch.log(probabilities[target_index].clamp_min(1.0e-12)).cpu())
        if example.wdl_mask:
            probabilities = torch.softmax(output["values"][:3], dim=0)
            target = torch.nn.functional.one_hot(
                torch.tensor(example.wdl, dtype=torch.long), num_classes=3
            ).to(dtype=probabilities.dtype, device=probabilities.device)
            wdl_brier += float(torch.mean((probabilities - target) ** 2).cpu())
            wdl_nll += float(-torch.log(probabilities[example.wdl].clamp_min(1.0e-12)).cpu())
            wdl_total += 1
            confidence, predicted = torch.max(probabilities, dim=0)
            bin_index = min(9, int(float(confidence.cpu()) * 10.0))
            calibration_bins[bin_index]["count"] += 1
            calibration_bins[bin_index]["confidence"] += float(confidence.cpu())
            calibration_bins[bin_index]["correct"] += int(int(predicted) == example.wdl)
    return {
        "rows": len(lines),
        "source_counts": dict(sorted(counts.items())),
        "policy_correct": policy_correct,
        "policy_examples": policy_total,
        "policy_nll_sum": policy_nll,
        "wdl_examples": wdl_total,
        "wdl_brier_sum": wdl_brier,
        "wdl_nll_sum": wdl_nll,
        "calibration_bins": calibration_bins,
    }


def _merge_evaluation_partials(
    partials: Sequence[Mapping[str, Any]], expected_rows: int
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    observed = 0
    policy_correct = 0
    policy_total = 0
    policy_nll = 0.0
    wdl_brier = 0.0
    wdl_nll = 0.0
    wdl_total = 0
    calibration_bins = [{"count": 0, "confidence": 0.0, "correct": 0.0} for _ in range(10)]
    for partial in partials:
        observed += int(partial["rows"])
        counts.update({str(key): int(value) for key, value in partial["source_counts"].items()})
        policy_correct += int(partial["policy_correct"])
        policy_total += int(partial["policy_examples"])
        policy_nll += float(partial["policy_nll_sum"])
        wdl_total += int(partial["wdl_examples"])
        wdl_brier += float(partial["wdl_brier_sum"])
        wdl_nll += float(partial["wdl_nll_sum"])
        for index, bucket in enumerate(partial["calibration_bins"]):
            calibration_bins[index]["count"] += int(bucket["count"])
            calibration_bins[index]["confidence"] += float(bucket["confidence"])
            calibration_bins[index]["correct"] += float(bucket["correct"])
    if observed != expected_rows:
        raise Phase10RCampaignError(
            f"evaluation row count mismatches its manifest: {observed} != {expected_rows}"
        )
    ece = 0.0
    for bucket in calibration_bins:
        if bucket["count"]:
            ece += (
                bucket["count"]
                / max(wdl_total, 1)
                * abs(bucket["confidence"] / bucket["count"] - bucket["correct"] / bucket["count"])
            )
    return {
        "rows": observed,
        "source_counts": dict(sorted(counts.items())),
        "policy_examples": policy_total,
        "policy_top1": policy_correct / policy_total if policy_total else None,
        "policy_nll": policy_nll / policy_total if policy_total else None,
        "wdl_examples": wdl_total,
        "wdl_brier": wdl_brier / wdl_total if wdl_total else None,
        "wdl_nll": wdl_nll / wdl_total if wdl_total else None,
        "calibration_ece": ece,
    }


def evaluate_scale(
    root: Path,
    scale: str,
    *,
    all_source_held_out: bool,
    cross_runtime: bool,
    incremental_parity: bool,
    git_commit: str,
) -> dict[str, Any]:
    if not all_source_held_out or not cross_runtime or not incremental_parity:
        raise Phase10RCampaignError(
            "evaluation requires all-source-held-out, cross-runtime, and incremental-parity flags"
        )
    manifest_path, manifest, manifest_sha256 = _manifest_identity(root, scale)
    validation_path = _preparation_file(manifest_path, manifest, "base-validation.jsonl")
    held_out_path = _preparation_file(manifest_path, manifest, "base-source_held_out.jsonl")
    data_root = _data_root(root)
    results: list[dict[str, Any]] = []
    for variant in TRAIN_VARIANTS:
        model_dir = data_root / "checkpoints" / "phase10r" / scale / variant
        checkpoint = model_dir / STAGE_TWO / "last.pt"
        artifact_path = model_dir / f"{variant}.osaval02"
        if not checkpoint.is_file() or checkpoint.is_symlink() or not artifact_path.is_file():
            raise Phase10RCampaignError(f"completed stage-2 candidate is missing: {variant}")
        device_receipt = select_device("auto", allow_cpu_fallback=True)
        # Source-held-out evaluation uses forked workers.  Keep the inherited
        # model CPU-resident because forking an MPS-backed model is unsupported
        # on macOS and terminates the worker pool before it can produce a
        # receipt.  Training-device selection remains recorded separately.
        model = Phase10RModel(variant, seed=TRAINING_SEED).to("cpu")
        try:
            payload = _load_verified_checkpoint(checkpoint)
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != CHECKPOINT_SCHEMA
                or payload.get("completed") is not True
                or payload.get("stage_id") != STAGE_TWO
                or payload.get("variant_id") != variant
                or payload.get("manifest_sha256") != manifest_sha256
            ):
                raise Phase10RCampaignError(f"stage-2 checkpoint identity is invalid: {variant}")
            model.load_state_dict(payload["model_state"], strict=True)
        except Phase10RCampaignError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            raise _failure(f"cannot load stage-2 checkpoint: {variant}", error) from error
        model.eval()
        files = manifest["files"]
        validation_rows = int(files["base-validation.jsonl"]["rows"])
        held_out_rows = int(files["base-source_held_out.jsonl"]["rows"])
        validation = _evaluate_file(
            validation_path,
            expected_rows=validation_rows,
            model=model,
            data_root=data_root,
        )
        source_held_out = _evaluate_file(
            held_out_path,
            expected_rows=held_out_rows,
            model=model,
            data_root=data_root,
        )
        parity = _candidate_parity(root, artifact_path, artifact_path)
        results.append(
            {
                "variant": variant,
                "device": device_receipt.as_dict(),
                "evaluation_device": "cpu",
                "checkpoint": checkpoint,
                "checkpoint_sha256": _sha256_file(checkpoint),
                "artifact": artifact_path,
                "validation": validation,
                "source_held_out": source_held_out,
                "parity": parity,
            }
        )
    incremental = _incremental_parity(root)
    result: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "status": "passed",
        "scale": scale,
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit,
        "candidates": results,
        "incremental_parity": incremental,
        "teacher_dependent_stages": "not_started",
        "expansion_gate": "blocked_until_teacher_stages_and_arena",
    }
    output = data_root / "evaluations" / scale / "evaluation.json"
    _atomic_json(output, _json_safe(result))
    return result | {"output": output}


def _teacher_identity(root: Path) -> dict[str, Any]:
    try:
        config = _load_yaml(root / "configs/teacher/apery-v2.0.0.yaml")
        target = _load_yaml(root / "configs/phase10r/target-semantics.yaml")
    except (OSError, ValueError, KeyError) as error:
        raise _failure("teacher identity configuration cannot be read", error) from error
    teacher = config.get("teacher")
    target_teacher = target.get("teacher")
    if not isinstance(teacher, Mapping) or not isinstance(target_teacher, Mapping):
        raise Phase10RCampaignError("teacher identity configuration is incomplete")
    if (
        teacher.get("name") != "Apery"
        or teacher.get("version") != "2.0.0"
        or teacher.get("nodes") != 25_000
        or teacher.get("multipv") != 3
        or teacher.get("threads") != 4
        or teacher.get("concurrency") != 1
    ):
        raise Phase10RCampaignError(
            "Apery baseline options do not match the frozen teacher contract"
        )
    executable_value = teacher.get("executable")
    if not isinstance(executable_value, str) or not executable_value:
        raise Phase10RCampaignError("teacher executable path is not configured")
    executable = root / executable_value
    if executable.is_symlink() or not executable.is_file():
        raise Phase10RCampaignError(
            "Apery teacher binary is unavailable; run the existing rights-gated setup first"
        )
    expected = target_teacher.get("binary_sha256", TEACHER_EXPECTED_SHA256)
    if expected != TEACHER_EXPECTED_SHA256 or _sha256_file(executable) != expected:
        raise Phase10RCampaignError("Apery teacher binary identity mismatches the frozen hash")
    eval_files = teacher.get("eval_files")
    if not isinstance(eval_files, list):
        raise Phase10RCampaignError("Apery evaluation-file identity is incomplete")
    for item in eval_files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise Phase10RCampaignError("Apery evaluation-file identity is malformed")
        path = root / str(item["path"])
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != item.get("sha256"):
            raise Phase10RCampaignError(f"Apery evaluation-file identity mismatches: {path}")
    return {
        "name": teacher.get("name"),
        "version": teacher.get("version"),
        "binary": executable,
        "binary_sha256": expected,
        "eval_files": [
            {"path": root / str(item["path"]), "sha256": item.get("sha256")}
            for item in eval_files
            if isinstance(item, Mapping) and isinstance(item.get("path"), str)
        ],
    }


def select_hard(root: Path, scale: str, *, git_commit: str) -> dict[str, Any]:
    _manifest_identity(root, scale)
    teacher = _teacher_identity(root)
    from open_shogi_training.phase10r_lineage import (
        Phase10RLineageError,
        completed_teacher_bound_candidates,
    )

    try:
        candidates = completed_teacher_bound_candidates(root, scale)
    except Phase10RLineageError as error:
        raise _failure("teacher-bound candidate lineage is invalid", error) from error
    if not candidates:
        raise Phase10RCampaignError(
            "hard-example selection is not authorized without a completed teacher-bound candidate"
            f" ({teacher['name']} {teacher['version']})"
        )
    raise Phase10RCampaignError(
        "teacher-bound candidate gate passed, but hard-example selection execution is outside "
        "the bounded teacher-binding repair"
    )


def label_hard(root: Path, scale: str, *, resume: bool, git_commit: str) -> dict[str, Any]:
    _manifest_identity(root, scale)
    _teacher_identity(root)
    raise Phase10RCampaignError(
        "hard-example labeling requires the frozen selection manifest and is not started"
    )


__all__ = [
    "CAMPAIGN_SCHEMA",
    "EVALUATION_SCHEMA",
    "Phase10RCampaignError",
    "evaluate_scale",
    "label_hard",
    "select_hard",
    "train_variant",
]
