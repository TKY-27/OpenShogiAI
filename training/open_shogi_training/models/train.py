"""Deterministic training, validation, final testing, and model comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import secrets
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    publish_regular_at,
    retire_bound_regular,
    stable_directory_lock,
    stable_parent_descriptor,
    stable_regular_descriptor,
    write_json_atomic,
)
from open_shogi_training.models.checkpoint import (
    RUNTIME_KEYS,
    atomic_save_checkpoint,
    build_checkpoint,
    load_checkpoint,
    optimizer_to_device,
    restore_rng_state,
    validate_resume_identity,
)
from open_shogi_training.models.config import (
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    combined_config_sha256,
    parse_feature_config,
    parse_model_config,
    parse_training_config,
    validate_config_compatibility,
)
from open_shogi_training.models.dataset import (
    DeterministicStageSampler,
    LoadedExamples,
    TrainingExample,
    ValueDataset,
    hash_file,
    split_examples,
)
from open_shogi_training.models.export import MAX_NON_MATE_CP, normalized_to_centipawns
from open_shogi_training.models.features import feature_groups, input_dimension
from open_shogi_training.models.network import (
    DeviceSelection,
    ValueModel,
    ensure_finite_model,
    select_device,
)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    split: str
    examples: int
    batches: int
    teacher_examples: int
    outcome_examples: int
    policy_examples: int
    ranking_pairs: int
    total_loss: float
    teacher_loss: float
    game_result_loss: float
    policy_agreement_loss: float
    ranking_loss: float
    teacher_mae_cp: float
    policy_accuracy: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    completed_epochs: int
    global_step: int
    best_validation_loss: float
    device: str


TRAINING_TRANSACTION_SCHEMA = "phase4_training_transaction/v3"
_FAILPOINT_ENV = "OPEN_SHOGI_TRAINING_FAILPOINT"


@dataclass(slots=True)
class _Accumulator:
    examples: int = 0
    batches: int = 0
    teacher: float = 0.0
    outcome: float = 0.0
    policy: float = 0.0
    ranking: float = 0.0
    outcome_count: float = 0.0
    ranking_count: float = 0.0
    teacher_absolute_cp: float = 0.0
    teacher_count: float = 0.0
    policy_correct: float = 0.0
    policy_count: float = 0.0


def train_model(
    loaded: LoadedExamples,
    feature_config: FeatureConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    resume_path: Path | None = None,
    mode: str = "train",
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    overfit_examples: int = 32,
    stop_after_epoch: int | None = None,
) -> TrainResult:
    """Train under an exclusive output-directory lock."""

    with _training_output_lock(output_dir):
        return _train_model(
            loaded,
            feature_config,
            model_config,
            training_config,
            output_dir,
            resume_path=resume_path,
            mode=mode,
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            overfit_examples=overfit_examples,
            stop_after_epoch=stop_after_epoch,
        )


def _train_model(
    loaded: LoadedExamples,
    feature_config: FeatureConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    resume_path: Path | None = None,
    mode: str = "train",
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    overfit_examples: int = 32,
    stop_after_epoch: int | None = None,
) -> TrainResult:
    """Train on train only, select on validation only, and checkpoint each epoch."""

    execution_started_at = _utc_now()
    execution_started = time.monotonic()
    validate_config_compatibility(model_config, training_config, feature_config)
    if mode not in {"train", "smoke", "overfit"}:
        raise ValueError("training mode must be train, smoke, or overfit")
    selection = select_device(training_config.device)
    _configure_reproducibility(training_config)
    runtime = _runtime_record(selection, training_config)
    train_rows = split_examples(loaded, "train")
    validation_rows = split_examples(loaded, "validation")
    if mode == "overfit":
        train_rows = _balanced_prefix(train_rows, overfit_examples)
        validation_rows = train_rows
    train_dataset = ValueDataset(train_rows, feature_config)
    validation_dataset = ValueDataset(validation_rows, feature_config)
    sampler = DeterministicStageSampler(train_dataset, training_config)
    model = ValueModel(input_dimension(feature_config), model_config).to(selection.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    config_hash = combined_config_sha256(feature_config, model_config, training_config)
    identity = asdict(loaded.identity)
    log_path = output_dir / "training-log.jsonl"
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    experiment_path = output_dir / "experiment.json"
    transaction_path = output_dir / "epoch-transaction.json"
    start_epoch = 0
    global_step = 0
    best_validation_loss = 1.0e30
    prior_executions: list[dict[str, Any]] = []
    recovered_experiment = _recover_epoch_transaction(
        transaction_path=transaction_path,
        log_path=log_path,
        last_path=last_path,
        best_path=best_path,
        experiment_path=experiment_path,
    )
    pending_journal = (
        _load_json_file(transaction_path, max_bytes=4 * 1024 * 1024)
        if transaction_path.exists()
        else None
    )
    pending_experiment_recovery = pending_journal is not None
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        validate_resume_identity(
            checkpoint,
            config_sha256=config_hash,
            dataset_identity=identity,
        )
        _validate_resume_runtime(checkpoint["runtime"], runtime)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(optimizer, selection.device)
        restore_rng_state(checkpoint["rng_state"], selection.device)
        start_epoch = checkpoint["completed_epoch"]
        global_step = checkpoint["global_step"]
        best_validation_loss = checkpoint["best_validation_loss"]
        prior_executions = _validate_resume_outputs(
            resume_path=resume_path,
            log_path=log_path,
            last_path=last_path,
            best_path=best_path,
            experiment_path=experiment_path,
            mode=mode,
            config_sha256=config_hash,
            dataset_identity=identity,
            completed_epoch=start_epoch,
            global_step=global_step,
            pending_journal=pending_journal,
        )
    ensure_finite_model(model, include_gradients=False)

    if resume_path is None:
        _require_new_training_outputs(
            (log_path, last_path, best_path, experiment_path, transaction_path)
        )
    configured_epochs = 1 if mode == "smoke" else training_config.epochs
    if (
        recovered_experiment
        and resume_path is not None
        and start_epoch == configured_epochs
        and stop_after_epoch in {None, start_epoch}
    ):
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=best_path,
            last_checkpoint=last_path,
            completed_epochs=start_epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            device=str(selection.device),
        )
    end_epoch = configured_epochs if stop_after_epoch is None else stop_after_epoch
    if not (
        start_epoch < end_epoch <= configured_epochs
        or (
            pending_experiment_recovery
            and start_epoch == end_epoch
            and 1 <= end_epoch <= configured_epochs
        )
    ):
        raise ValueError(
            "stop_after_epoch must be greater than the resumed epoch and no greater than "
            "configured epochs"
        )
    for epoch in range(start_epoch, end_epoch):
        sampler.set_epoch(epoch)
        train_loader = _training_loader(train_dataset, sampler, training_config, epoch)
        train_summary, steps = _run_epoch(
            model,
            train_loader,
            training_config,
            selection.device,
            split="train",
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        global_step += steps
        validation_loader = _evaluation_loader(validation_dataset, training_config)
        validation_summary, _ = _run_epoch(
            model,
            validation_loader,
            training_config,
            selection.device,
            split="validation" if mode != "overfit" else "overfit",
            optimizer=None,
            max_batches=max_validation_batches,
        )
        completed_epoch = epoch + 1
        improved = validation_summary.total_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_summary.total_loss
        checkpoint_payload = build_checkpoint(
            completed_epoch=completed_epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            model=model,
            optimizer=optimizer,
            feature_config=feature_config.as_dict(),
            model_config=model_config.as_dict(),
            training_config=training_config.as_dict(),
            config_sha256=config_hash,
            dataset_identity=identity,
            device=selection.device,
            runtime=runtime,
        )
        retain_generation = completed_epoch % training_config.checkpoint_every_epochs == 0
        log_record = {
            "schema": "phase4_training_log/v1",
            "mode": mode,
            "epoch": completed_epoch,
            "globalStep": global_step,
            "train": train_summary.as_dict(),
            "validation": validation_summary.as_dict(),
            "bestValidationLoss": best_validation_loss,
            "improved": improved,
        }
        _commit_epoch_transaction(
            transaction_path=transaction_path,
            log_path=log_path,
            last_path=last_path,
            best_path=best_path,
            payload=checkpoint_payload,
            log_record=log_record,
            retain_generation=retain_generation,
            publish_best=improved,
            create_log=resume_path is None and completed_epoch == 1,
        )
    if not last_path.is_file() or not best_path.is_file():
        raise RuntimeError("training did not publish both last and best checkpoints")
    _synchronize_device(selection.device)
    completed_at = _utc_now()
    elapsed_seconds = time.monotonic() - execution_started
    process_peak_resident_bytes = _process_peak_resident_bytes()
    mps_memory = _mps_memory_observation(selection.device)
    execution = {
        "startedAt": execution_started_at,
        "completedAt": completed_at,
        "elapsedSeconds": elapsed_seconds,
        "startEpoch": start_epoch,
        "endEpoch": end_epoch,
        "resumeCheckpoint": resume_path.name if resume_path is not None else None,
        "executionControl": {
            "stopAfterEpoch": stop_after_epoch,
            "maxTrainBatches": max_train_batches,
            "maxValidationBatches": max_validation_batches,
        },
        "processPeakResidentBytes": process_peak_resident_bytes,
        "peakResidentMeasurement": "getrusage_process_lifetime_max",
        "mpsMemory": mps_memory,
    }
    experiment = {
        "schema": "phase4_value_experiment/v1",
        "mode": mode,
        "configSha256": config_hash,
        "configs": {
            "features": feature_config.as_dict(),
            "model": model_config.as_dict(),
            "training": training_config.as_dict(),
        },
        "datasetIdentity": identity,
        "countsBySplit": loaded.counts_by_split,
        "countsByStage": loaded.counts_by_stage,
        "trainExamplesUsed": len(train_rows),
        "validationExamplesUsed": len(validation_rows),
        "testExamplesUsed": 0,
        "replayExamplesUsed": sum(row.source_kind == "phase6_replay" for row in train_rows),
        "supervisionCounts": {
            "trainTeacher": sum(row.teacher_mask == 1.0 for row in train_rows),
            "trainOutcome": sum(row.outcome_mask == 1.0 for row in train_rows),
            "trainPolicyAgreement": sum(row.policy_mask == 1.0 for row in train_rows),
            "trainReplayOutcomeOnly": sum(
                row.source_kind == "phase6_replay"
                and row.teacher_mask == 0.0
                and row.policy_mask == 0.0
                and row.outcome_mask == 1.0
                for row in train_rows
            ),
            "validationTeacher": sum(row.teacher_mask == 1.0 for row in validation_rows),
            "validationOutcome": sum(row.outcome_mask == 1.0 for row in validation_rows),
            "validationPolicyAgreement": sum(row.policy_mask == 1.0 for row in validation_rows),
        },
        "configuredEpochs": configured_epochs,
        "completedEpochs": end_epoch,
        "globalStep": global_step,
        "bestValidationLoss": best_validation_loss,
        "device": asdict(selection),
        "runtime": runtime,
        "executions": [*prior_executions, execution],
        "resources": {
            "processPeakResidentBytes": max(
                execution_record["processPeakResidentBytes"]
                for execution_record in [*prior_executions, execution]
            ),
            "peakResidentMeasurement": "getrusage_process_lifetime_max",
            "mps": _aggregate_mps_observations([*prior_executions, execution]),
        },
        "artifacts": {
            "bestCheckpoint": _identity(best_path),
            "lastCheckpoint": _identity(last_path),
            "trainingLog": _identity(log_path, max_bytes=4 * 1024 * 1024 * 1024),
        },
    }
    experiment["device"]["device"] = str(selection.device)
    _prepare_experiment_transaction(transaction_path, experiment_path, experiment)
    _write_json_replace(experiment_path, experiment)
    _failpoint("after_experiment")
    _finish_epoch_transaction(transaction_path, experiment_path)
    return TrainResult(
        output_dir=output_dir,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        completed_epochs=end_epoch,
        global_step=global_step,
        best_validation_loss=best_validation_loss,
        device=str(selection.device),
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    loaded: LoadedExamples,
    *,
    split: str,
    predictions_output: Path | None = None,
    ablated_group: str | None = None,
) -> EvaluationSummary:
    if split not in {"validation", "test"}:
        raise ValueError("explicit evaluation split must be validation or test")
    checkpoint, feature, model_config, training = _load_model_bundle(checkpoint_path)
    validate_resume_identity(
        checkpoint,
        config_sha256=combined_config_sha256(feature, model_config, training),
        dataset_identity=asdict(loaded.identity),
    )
    selection = select_device(training.device)
    model = ValueModel(input_dimension(feature), model_config).to(selection.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    ensure_finite_model(model, include_gradients=False)
    examples = split_examples(loaded, split)
    dataset = ValueDataset(examples, feature, ablated_group=ablated_group)
    summary, predictions = _run_epoch(
        model,
        _evaluation_loader(dataset, training),
        training,
        selection.device,
        split=split,
        optimizer=None,
        collect_predictions=predictions_output is not None,
    )
    if predictions_output is not None:
        checkpoint_sha256, _ = hash_file(checkpoint_path, max_bytes=512 * 1024 * 1024)
        _write_predictions(
            predictions_output,
            examples,
            predictions,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=checkpoint["config_sha256"],
            output_scale_cp=model_config.output_scale_cp,
        )
    return summary


def compare_checkpoints(
    left_path: Path,
    right_path: Path,
    loaded: LoadedExamples,
) -> dict[str, Any]:
    _, _, _, left_training = _load_model_bundle(left_path)
    _, _, _, right_training = _load_model_bundle(right_path)
    left_contract = _evaluation_contract(left_training)
    right_contract = _evaluation_contract(right_training)
    if left_contract != right_contract:
        raise ValueError(
            "checkpoint comparison requires identical target normalization and evaluation losses"
        )
    left = evaluate_checkpoint(left_path, loaded, split="validation")
    right = evaluate_checkpoint(right_path, loaded, split="validation")
    return {
        "schema": "phase4_model_comparison/v1",
        "split": "validation",
        "evaluationContract": left_contract,
        "left": left.as_dict(),
        "right": right.as_dict(),
        "totalLossDeltaRightMinusLeft": right.total_loss - left.total_loss,
        "teacherMaeCpDeltaRightMinusLeft": right.teacher_mae_cp - left.teacher_mae_cp,
    }


def _evaluation_contract(config: TrainingConfig) -> dict[str, float]:
    return {
        "teacherLossWeight": config.teacher_loss_weight,
        "gameResultLossWeight": config.game_result_loss_weight,
        "policyAgreementLossWeight": config.policy_agreement_loss_weight,
        "rankingLossWeight": config.ranking_loss_weight,
        "rankingMargin": config.ranking_margin,
        "teacherClipCp": config.teacher_clip_cp,
        "teacherNormalizationCp": config.teacher_normalization_cp,
    }


def feature_ablation(
    checkpoint_path: Path,
    loaded: LoadedExamples,
) -> dict[str, Any]:
    checkpoint, feature, _, _ = _load_model_bundle(checkpoint_path)
    baseline = evaluate_checkpoint(checkpoint_path, loaded, split="validation")
    ablations = []
    for group in feature_groups(feature):
        summary = evaluate_checkpoint(
            checkpoint_path,
            loaded,
            split="validation",
            ablated_group=group.name,
        )
        ablations.append(
            {
                "group": group.name,
                "metrics": summary.as_dict(),
                "totalLossDelta": summary.total_loss - baseline.total_loss,
                "teacherMaeCpDelta": summary.teacher_mae_cp - baseline.teacher_mae_cp,
            }
        )
    return {
        "schema": "phase4_feature_ablation/v1",
        "checkpointConfigSha256": checkpoint["config_sha256"],
        "split": "validation",
        "baseline": baseline.as_dict(),
        "ablations": ablations,
    }


def _run_epoch(
    model: ValueModel,
    loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    *,
    split: str,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    collect_predictions: bool = False,
) -> tuple[EvaluationSummary, int | list[float]]:
    training = optimizer is not None
    model.train(training)
    accumulator = _Accumulator()
    predictions: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            moved = {key: value.to(device) for key, value in batch.items()}
            if not torch.isfinite(moved["features"]).all().item():
                raise FloatingPointError("batch features contain NaN or infinity")
            if training:
                optimizer.zero_grad(set_to_none=True)
            value, policy_logit = model(moved["features"])
            if (
                not torch.isfinite(value).all().item()
                or not torch.isfinite(policy_logit).all().item()
            ):
                raise FloatingPointError("model output contains NaN or infinity")
            losses = _losses(value, policy_logit, moved, config)
            if not torch.isfinite(losses["total"]).item():
                raise FloatingPointError("weighted training loss is NaN or infinity")
            if training:
                losses["total"].backward()
                ensure_finite_model(model, include_gradients=True)
                norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                if not torch.isfinite(norm).item():
                    raise FloatingPointError("gradient norm is NaN or infinity")
                optimizer.step()
                ensure_finite_model(model, include_gradients=False)
            _accumulate(accumulator, value, policy_logit, moved, losses, config)
            if collect_predictions:
                predictions.extend(float(item) for item in value.detach().cpu().tolist())
    if accumulator.batches == 0:
        raise ValueError(f"{split} loader produced no batches")
    summary = _summary(split, accumulator, config)
    return summary, predictions if collect_predictions else accumulator.batches


def _losses(
    value: Tensor,
    policy_logit: Tensor,
    batch: dict[str, Tensor],
    config: TrainingConfig,
) -> dict[str, Tensor]:
    teacher = _masked_mean(
        functional.smooth_l1_loss(value, batch["teacher_target"], reduction="none"),
        batch["teacher_mask"],
    )
    outcome = _masked_mean(
        functional.mse_loss(value, batch["outcome_target"], reduction="none"),
        batch["outcome_mask"],
    )
    policy = _masked_mean(
        functional.binary_cross_entropy_with_logits(
            policy_logit, batch["policy_target"], reduction="none"
        ),
        batch["policy_mask"],
    )
    pair_count = value.shape[0] // 2
    if pair_count:
        left = value[: 2 * pair_count : 2]
        right = value[1 : 2 * pair_count : 2]
        target_left = batch["teacher_target"][: 2 * pair_count : 2]
        target_right = batch["teacher_target"][1 : 2 * pair_count : 2]
        mask = (
            batch["teacher_mask"][: 2 * pair_count : 2]
            * batch["teacher_mask"][1 : 2 * pair_count : 2]
            * (target_left != target_right).to(torch.float32)
        )
        direction = torch.sign(target_left - target_right)
        ranking = _masked_mean(torch.relu(config.ranking_margin - direction * (left - right)), mask)
        ranking_count = mask.sum()
    else:
        ranking = value.sum() * 0.0
        ranking_count = value.new_zeros(())
    total = (
        config.teacher_loss_weight * teacher
        + config.game_result_loss_weight * outcome
        + config.policy_agreement_loss_weight * policy
        + config.ranking_loss_weight * ranking
    )
    return {
        "teacher": teacher,
        "outcome": outcome,
        "policy": policy,
        "ranking": ranking,
        "ranking_count": ranking_count,
        "total": total,
    }


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum()
    if count.item() == 0.0:
        return values.sum() * 0.0
    return (values * mask).sum() / count


def _accumulate(
    accumulator: _Accumulator,
    value: Tensor,
    policy_logit: Tensor,
    batch: dict[str, Tensor],
    losses: dict[str, Tensor],
    config: TrainingConfig,
) -> None:
    count = value.shape[0]
    accumulator.examples += count
    accumulator.batches += 1
    teacher_mask = batch["teacher_mask"]
    teacher_count = float(teacher_mask.sum().item())
    outcome_count = float(batch["outcome_mask"].sum().item())
    policy_count = float(batch["policy_mask"].sum().item())
    ranking_count = float(losses["ranking_count"].item())
    accumulator.teacher += float(losses["teacher"].item()) * teacher_count
    accumulator.outcome += float(losses["outcome"].item()) * outcome_count
    accumulator.policy += float(losses["policy"].item()) * policy_count
    accumulator.ranking += float(losses["ranking"].item()) * ranking_count
    accumulator.outcome_count += outcome_count
    accumulator.ranking_count += ranking_count
    accumulator.teacher_absolute_cp += float(
        (torch.abs(value * config.teacher_normalization_cp - batch["teacher_cp"]) * teacher_mask)
        .sum()
        .item()
    )
    accumulator.teacher_count += teacher_count
    policy_mask = batch["policy_mask"]
    predicted = (policy_logit >= 0.0).to(torch.float32)
    accumulator.policy_correct += float(
        ((predicted == batch["policy_target"]).to(torch.float32) * policy_mask).sum().item()
    )
    accumulator.policy_count += policy_count


def _summary(split: str, accumulator: _Accumulator, config: TrainingConfig) -> EvaluationSummary:
    teacher = accumulator.teacher / max(1.0, accumulator.teacher_count)
    outcome = accumulator.outcome / max(1.0, accumulator.outcome_count)
    policy = accumulator.policy / max(1.0, accumulator.policy_count)
    ranking = accumulator.ranking / max(1.0, accumulator.ranking_count)
    total = (
        config.teacher_loss_weight * teacher
        + config.game_result_loss_weight * outcome
        + config.policy_agreement_loss_weight * policy
        + config.ranking_loss_weight * ranking
    )
    values = (
        total,
        teacher,
        outcome,
        policy,
        ranking,
        accumulator.teacher_absolute_cp / max(1.0, accumulator.teacher_count),
        accumulator.policy_correct / max(1.0, accumulator.policy_count),
    )
    if any(not math.isfinite(value) for value in values):
        raise FloatingPointError("epoch metrics contain NaN or infinity")
    return EvaluationSummary(
        split=split,
        examples=accumulator.examples,
        batches=accumulator.batches,
        teacher_examples=int(accumulator.teacher_count),
        outcome_examples=int(accumulator.outcome_count),
        policy_examples=int(accumulator.policy_count),
        ranking_pairs=int(accumulator.ranking_count),
        total_loss=values[0],
        teacher_loss=values[1],
        game_result_loss=values[2],
        policy_agreement_loss=values[3],
        ranking_loss=values[4],
        teacher_mae_cp=values[5],
        policy_accuracy=values[6],
    )


def _training_loader(
    dataset: ValueDataset,
    sampler: DeterministicStageSampler,
    config: TrainingConfig,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def _evaluation_loader(dataset: ValueDataset, config: TrainingConfig) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)


def _load_model_bundle(
    path: Path,
) -> tuple[dict[str, Any], FeatureConfig, ModelConfig, TrainingConfig]:
    checkpoint = load_checkpoint(path)
    feature = parse_feature_config(checkpoint["feature_config"])
    model = parse_model_config(checkpoint["model_config"])
    training = parse_training_config(checkpoint["training_config"])
    validate_config_compatibility(model, training, feature)
    return checkpoint, feature, model, training


def _configure_reproducibility(config: TrainingConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)


def _runtime_record(selection: DeviceSelection, config: TrainingConfig) -> dict[str, str]:
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": str(selection.device),
        "deviceRequested": selection.requested,
        "deviceFallback": selection.fallback_reason or "none",
        "mpsBuilt": str(mps_built).lower(),
        "mpsAvailable": str(mps_available).lower(),
        "deterministicAlgorithms": str(config.deterministic).lower(),
        "gitCommit": _git_commit(),
        "gitDirty": _git_dirty(),
        "modelCodeSha256": model_code_sha256(),
        "torchNumThreads": str(torch.get_num_threads()),
        "torchNumInteropThreads": str(torch.get_num_interop_threads()),
    }
    if frozenset(runtime) != RUNTIME_KEYS:
        raise RuntimeError("internal runtime record does not match the checkpoint schema")
    return runtime


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            cwd=_repository_root(),
            text=True,
            timeout=10,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else "unknown"


def _git_dirty() -> str:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            cwd=_repository_root(),
            text=True,
            timeout=10,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return "true" if completed.stdout else "false"


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "HOME": "/var/empty",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _validate_resume_runtime(stored: object, current: dict[str, str]) -> None:
    if not isinstance(stored, dict) or frozenset(stored) != RUNTIME_KEYS:
        raise ValueError("resume checkpoint runtime record is invalid")
    for key in RUNTIME_KEYS:
        if stored.get(key) != current[key]:
            raise ValueError(f"resume runtime differs from checkpoint field: {key}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _process_peak_resident_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return max(0, int(peak * multiplier))


def _synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _mps_memory_observation(device: torch.device) -> dict[str, Any]:
    if device.type != "mps":
        return {
            "measurement": "not_applicable_cpu",
            "currentAllocatedBytes": None,
            "driverAllocatedBytes": None,
            "recommendedMaxBytes": None,
            "measurementError": None,
        }
    try:
        return {
            "measurement": "execution_end_after_mps_synchronize",
            "currentAllocatedBytes": int(torch.mps.current_allocated_memory()),
            "driverAllocatedBytes": int(torch.mps.driver_allocated_memory()),
            "recommendedMaxBytes": int(torch.mps.recommended_max_memory()),
            "measurementError": None,
        }
    except RuntimeError as error:
        return {
            "measurement": "unavailable",
            "currentAllocatedBytes": None,
            "driverAllocatedBytes": None,
            "recommendedMaxBytes": None,
            "measurementError": type(error).__name__,
        }


def _aggregate_mps_observations(executions: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [execution["mpsMemory"] for execution in executions]

    def observed_max(key: str) -> int | None:
        values = [item[key] for item in observations if isinstance(item.get(key), int)]
        return max(values) if values else None

    return {
        "measurement": "maximum_of_execution_end_observations",
        "observedMaximumCurrentAllocatedBytes": observed_max("currentAllocatedBytes"),
        "observedMaximumDriverAllocatedBytes": observed_max("driverAllocatedBytes"),
        "recommendedMaxBytes": observed_max("recommendedMaxBytes"),
        "measurementErrors": [
            item["measurementError"]
            for item in observations
            if item.get("measurementError") is not None
        ],
    }


def _model_code_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    training_package = package.parent
    dependency_paths = (
        training_package / "data" / "gzip_jsonl.py",
        training_package / "labeling" / "artifacts.py",
        training_package / "labeling" / "schema.py",
    )
    paths = [*package.glob("*.py"), *dependency_paths]
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("cannot identify the model-training source files")
    for path in sorted(paths, key=lambda item: item.relative_to(training_package).as_posix()):
        name = path.relative_to(training_package).as_posix().encode("utf-8")
        source_sha256, source_size = hash_file(path, max_bytes=16 * 1024 * 1024)
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(source_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(source_sha256))
    return digest.hexdigest()


def model_code_sha256() -> str:
    """Return the stable identity of training and direct parsing source code."""

    return _model_code_sha256()


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / "training").is_dir():
        raise RuntimeError("cannot identify the OpenShogiAI repository root")
    return root


def _require_new_training_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite an existing training artifact: {path}")


def _commit_epoch_transaction(
    *,
    transaction_path: Path,
    log_path: Path,
    last_path: Path,
    best_path: Path,
    payload: dict[str, Any],
    log_record: dict[str, Any],
    retain_generation: bool,
    publish_best: bool,
    create_log: bool,
) -> None:
    _close_published_epoch_for_next(
        transaction_path=transaction_path,
        log_path=log_path,
        last_path=last_path,
        best_path=best_path,
    )
    epoch = int(payload["completed_epoch"])
    generation_path = transaction_path.parent / f"checkpoint-epoch-{epoch:04d}.pt"
    if generation_path.exists() or generation_path.is_symlink():
        existing = load_checkpoint(generation_path)
        if not _checkpoint_values_equal(existing, payload):
            raise ValueError("epoch checkpoint generation conflicts with recovery evidence")
    else:
        atomic_save_checkpoint(generation_path, payload)
    checkpoint_identity = _identity(generation_path)
    best_before = (
        _identity(best_path) if best_path.is_file() and not best_path.is_symlink() else None
    )
    last_before = (
        _identity(last_path) if last_path.is_file() and not last_path.is_symlink() else None
    )
    log_before = (
        _identity(log_path, max_bytes=4 * 1024 * 1024 * 1024)
        if log_path.is_file() and not log_path.is_symlink()
        else None
    )
    if create_log != (log_before is None):
        raise ValueError("training log creation mode disagrees with the prior identity")
    experiment_path = transaction_path.parent / "experiment.json"
    experiment_before = (
        _identity(experiment_path, max_bytes=64 * 1024 * 1024)
        if experiment_path.is_file() and not experiment_path.is_symlink()
        else None
    )
    log_after = _appended_log_identity(log_path, log_before, _json_payload(log_record))
    best_after = (
        {
            "path": best_path.name,
            "sha256": checkpoint_identity["sha256"],
            "size": checkpoint_identity["size"],
        }
        if publish_best
        else best_before
    )
    journal = {
        "schema": TRAINING_TRANSACTION_SCHEMA,
        "epoch": epoch,
        "globalStep": payload["global_step"],
        "checkpoint": checkpoint_identity,
        "retainCheckpoint": retain_generation,
        "publishBest": publish_best,
        "bestBefore": best_before,
        "bestAfter": best_after,
        "lastBefore": last_before,
        "logBefore": log_before,
        "logAfter": log_after,
        "experimentBefore": experiment_before,
        "experimentAfter": None,
        "experimentRecord": None,
        "logRecord": log_record,
        "status": "prepared",
    }
    _write_json_replace(transaction_path, journal)
    _failpoint("after_journal")
    _recover_epoch_transaction(
        transaction_path=transaction_path,
        log_path=log_path,
        last_path=last_path,
        best_path=best_path,
        experiment_path=transaction_path.parent / "experiment.json",
    )


def _recover_epoch_transaction(
    *,
    transaction_path: Path,
    log_path: Path,
    last_path: Path,
    best_path: Path,
    experiment_path: Path,
) -> bool:
    if not transaction_path.exists() and not transaction_path.is_symlink():
        return False
    journal = _load_json_file(transaction_path, max_bytes=4 * 1024 * 1024)
    if (
        set(journal)
        != {
            "schema",
            "epoch",
            "globalStep",
            "checkpoint",
            "retainCheckpoint",
            "publishBest",
            "bestBefore",
            "bestAfter",
            "lastBefore",
            "logBefore",
            "logAfter",
            "experimentBefore",
            "experimentAfter",
            "experimentRecord",
            "logRecord",
            "status",
        }
        or journal.get("schema") != TRAINING_TRANSACTION_SCHEMA
    ):
        raise ValueError("training transaction journal schema is invalid")
    if journal.get("status") not in {"prepared", "published"}:
        raise ValueError("training transaction status is invalid")
    checkpoint_record = journal["checkpoint"]
    if not isinstance(checkpoint_record, dict) or set(checkpoint_record) != {
        "path",
        "sha256",
        "size",
    }:
        raise ValueError("training transaction checkpoint identity is invalid")
    generation_path = transaction_path.parent / str(checkpoint_record["path"])
    if _identity(generation_path) != checkpoint_record:
        raise ValueError("training transaction checkpoint bytes changed")
    checkpoint = load_checkpoint(generation_path)
    epoch = journal.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or checkpoint["completed_epoch"] != epoch
        or checkpoint["global_step"] != journal.get("globalStep")
    ):
        raise ValueError("training transaction checkpoint metadata disagrees")
    if not isinstance(journal["retainCheckpoint"], bool) or not isinstance(
        journal["publishBest"], bool
    ):
        raise ValueError("training transaction checkpoint flags are invalid")
    best_before = _optional_artifact_identity(journal["bestBefore"], "bestBefore")
    best_after = _optional_artifact_identity(journal["bestAfter"], "bestAfter")
    last_before = _optional_artifact_identity(journal["lastBefore"], "lastBefore")
    log_before = _optional_artifact_identity(journal["logBefore"], "logBefore")
    log_after = _optional_artifact_identity(journal["logAfter"], "logAfter")
    _optional_artifact_identity(journal["experimentBefore"], "experimentBefore")
    experiment_after = _optional_artifact_identity(journal["experimentAfter"], "experimentAfter")
    experiment_record = journal["experimentRecord"]
    if (experiment_after is None) != (experiment_record is None):
        raise ValueError("training transaction experiment publication is incomplete")
    if experiment_record is not None and not isinstance(experiment_record, dict):
        raise ValueError("training transaction experiment record is invalid")
    if journal["publishBest"]:
        if best_after is None or _record_content(best_after) != _record_content(checkpoint_record):
            raise ValueError("training transaction best-after identity is invalid")
    elif best_after != best_before:
        raise ValueError("training transaction changed best identity without improvement")
    _recover_best_reference(
        source=generation_path,
        target=best_path,
        before=best_before,
        after=best_after,
        publish=journal["publishBest"],
    )
    _failpoint("after_best")
    # `last.pt` is the interruption boundary, not a retention artifact.  It is
    # updated for every committed epoch regardless of the numbered-checkpoint
    # retention interval, but only from the exact prior identity bound in the
    # journal.
    last_after = {
        "path": last_path.name,
        "sha256": checkpoint_record["sha256"],
        "size": checkpoint_record["size"],
    }
    _recover_checkpoint_reference(
        source=generation_path,
        target=last_path,
        before=last_before,
        after=last_after,
    )
    _failpoint("after_last")
    log_record = journal["logRecord"]
    if not isinstance(log_record, dict) or log_record.get("epoch") != epoch:
        raise ValueError("training transaction log record is invalid")
    if log_after is None:
        raise ValueError("training transaction log-after identity is missing")
    _recover_log_append(
        log_path,
        before=log_before,
        after=log_after,
        payload=_json_payload(log_record),
    )
    _failpoint("after_log")
    if journal["status"] != "published":
        journal["status"] = "published"
        _write_json_replace(transaction_path, journal)
    if experiment_after is not None:
        _recover_experiment_publication(
            transaction_path=transaction_path,
            experiment_path=experiment_path,
            journal=journal,
        )
        return True
    return False


def _publish_checkpoint_reference(
    source: Path, target: Path, expected_identity: dict[str, Any]
) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            raise ValueError("checkpoint reference must not be a symlink")
        if _identity_content(target) == _record_content(expected_identity):
            return
    _atomic_copy_regular(source, target)
    if _identity_content(target) != _record_content(expected_identity):
        raise ValueError("published checkpoint reference differs from its generation")


def _optional_artifact_identity(value: object, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size"}:
        raise ValueError(f"training transaction {name} identity is invalid")
    _record_content(value)
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(f"training transaction {name} path is invalid")
    return value


def _recover_best_reference(
    *,
    source: Path,
    target: Path,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    publish: bool,
) -> None:
    observed = _identity(target) if target.is_file() and not target.is_symlink() else None
    if target.is_symlink():
        raise ValueError("best checkpoint reference must not be a symlink")
    if observed == after:
        return
    if observed != before:
        raise ValueError("best checkpoint differs from the transaction's bound prior identity")
    if not publish or after is None:
        return
    _publish_checkpoint_reference(source, target, after)


def _recover_checkpoint_reference(
    *,
    source: Path,
    target: Path,
    before: dict[str, Any] | None,
    after: dict[str, Any],
) -> None:
    if before is not None and before["path"] != target.name:
        raise ValueError("last-before identity names a different checkpoint reference")
    if after["path"] != target.name:
        raise ValueError("last-after identity names a different checkpoint reference")
    if target.is_symlink():
        raise ValueError("last checkpoint reference must not be a symlink")
    observed = _identity(target) if target.is_file() else None
    if observed == after:
        return
    if observed != before:
        raise ValueError("last checkpoint differs from the transaction's bound prior identity")
    _publish_checkpoint_reference(source, target, after)


def _appended_log_identity(
    path: Path,
    before: dict[str, Any] | None,
    payload: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    if before is None:
        if path.exists() or path.is_symlink():
            raise ValueError("new training log already exists without a bound prior identity")
    else:
        if (
            before["path"] != path.name
            or _identity(path, max_bytes=4 * 1024 * 1024 * 1024) != before
        ):
            raise ValueError("training log differs from its bound prior identity")
        try:
            with stable_regular_descriptor(path) as descriptor:
                expected_size = before["size"]
                while size < expected_size:
                    chunk = os.pread(
                        descriptor,
                        min(1024 * 1024, expected_size - size),
                        size,
                    )
                    if not chunk:
                        raise ValueError("training log ended while its append was prepared")
                    digest.update(chunk)
                    size += len(chunk)
        except ArtifactError as error:
            raise ValueError("training log changed while its append was prepared") from error
    digest.update(payload)
    return {"path": path.name, "sha256": digest.hexdigest(), "size": size + len(payload)}


def _recover_log_append(
    path: Path,
    *,
    before: dict[str, Any] | None,
    after: dict[str, Any],
    payload: bytes,
) -> None:
    if after["path"] != path.name or after["size"] != (before["size"] if before else 0) + len(
        payload
    ):
        raise ValueError("training transaction log-after identity is invalid")
    if before is not None and before["path"] != path.name:
        raise ValueError("training transaction log-before identity is invalid")
    if path.is_symlink():
        raise ValueError("training log must not be a symlink")
    exists = path.is_file()
    if not exists and before is not None:
        raise ValueError("training log disappeared after its prior identity was journaled")

    current_size = 0
    tail = b""
    if exists:
        prefix_size = before["size"] if before is not None else 0
        prefix_digest = hashlib.sha256()
        try:
            with stable_regular_descriptor(path) as descriptor:
                status = os.fstat(descriptor)
                if status.st_size < prefix_size or status.st_size > after["size"]:
                    raise ValueError("training log size conflicts with transaction recovery")
                offset = 0
                while offset < prefix_size:
                    chunk = os.pread(
                        descriptor,
                        min(1024 * 1024, prefix_size - offset),
                        offset,
                    )
                    if not chunk:
                        raise ValueError("training log prior prefix ended early")
                    prefix_digest.update(chunk)
                    offset += len(chunk)
                if before is not None and prefix_digest.hexdigest() != before["sha256"]:
                    raise ValueError("training log prior prefix changed")
                tail_size = status.st_size - prefix_size
                tail = os.pread(descriptor, tail_size, prefix_size)
                if len(tail) != tail_size:
                    raise ValueError("training log pending append ended early")
                current_size = status.st_size
        except ArtifactError as error:
            raise ValueError("training log changed during transaction recovery") from error
        if tail != payload[: len(tail)]:
            raise ValueError("training log partial append conflicts with transaction evidence")

    missing = payload[len(tail) :]
    if missing:
        _append_bytes(path, missing, create=not exists)
    observed = _identity(path, max_bytes=4 * 1024 * 1024 * 1024)
    if observed != after or current_size + len(missing) != after["size"]:
        raise ValueError("recovered training log differs from the journaled append")


def _close_published_epoch_for_next(
    *,
    transaction_path: Path,
    log_path: Path,
    last_path: Path,
    best_path: Path,
) -> None:
    """Commit the prior epoch marker before preparing its successor journal."""

    if not transaction_path.exists() and not transaction_path.is_symlink():
        return
    _recover_epoch_transaction(
        transaction_path=transaction_path,
        log_path=log_path,
        last_path=last_path,
        best_path=best_path,
        experiment_path=transaction_path.parent / "experiment.json",
    )
    journal = _load_json_file(transaction_path, max_bytes=4 * 1024 * 1024)
    if journal.get("status") != "published":
        raise ValueError("prior epoch transaction is not published")
    checkpoint = _optional_artifact_identity(journal.get("checkpoint"), "checkpoint")
    if checkpoint is None or _identity_content(last_path) != _record_content(checkpoint):
        raise ValueError("last checkpoint does not commit the prior epoch")
    generation_path: Path | None = None
    if not journal.get("retainCheckpoint"):
        generation_path = transaction_path.parent / str(checkpoint["path"])
        if _identity(generation_path) != checkpoint:
            raise ValueError("prior ephemeral checkpoint generation changed")
    # The published journal is the only durable pointer to an ephemeral
    # generation.  Remove that pointer first: a crash can then leave only an
    # unreferenced (and harmless) generation, never a journal whose checkpoint
    # was already deleted and therefore cannot be recovered.
    _unlink_regular_and_fsync(transaction_path)
    if generation_path is not None:
        _unlink_regular_and_fsync(generation_path)


def _record_content(record: dict[str, Any]) -> tuple[str, int]:
    sha256 = record.get("sha256")
    size = record.get("size")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise ValueError("artifact identity record is invalid")
    return sha256, size


def _identity_content(path: Path) -> tuple[str, int]:
    sha256, size = hash_file(path, max_bytes=512 * 1024 * 1024)
    return sha256, size


def _atomic_copy_regular(source: Path, target: Path) -> None:
    from open_shogi_training.labeling.artifacts import (
        stable_parent_descriptor,
        stable_regular_descriptor,
    )

    with (
        stable_regular_descriptor(source) as source_descriptor,
        stable_parent_descriptor(target, create=True) as (parent_descriptor, name),
    ):
        temporary_name = f".{name}.{secrets.token_hex(12)}"
        writer = -1
        temporary_created = False
        temporary_status: os.stat_result | None = None
        try:
            writer = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            temporary_created = True
            temporary_status = os.fstat(writer)
            offset = 0
            size = os.fstat(source_descriptor).st_size
            while offset < size:
                chunk = os.pread(source_descriptor, min(1024 * 1024, size - offset), offset)
                if not chunk:
                    raise ValueError("checkpoint generation ended while copied")
                written = 0
                while written < len(chunk):
                    count = os.write(writer, chunk[written:])
                    if count <= 0:
                        raise ValueError("checkpoint reference copy was incomplete")
                    written += count
                offset += len(chunk)
            os.fsync(writer)
            os.close(writer)
            writer = -1
            assert temporary_status is not None
            publish_regular_at(
                parent_descriptor,
                temporary_name,
                name,
                temporary_status,
                display=target,
                replace=True,
            )
            temporary_created = False
            os.fsync(parent_descriptor)
        except BaseException:
            if writer >= 0:
                os.close(writer)
            if temporary_created:
                assert temporary_status is not None
                retire_bound_regular(
                    parent_descriptor,
                    temporary_name,
                    temporary_status,
                    display=target,
                )
            raise


def _prepare_experiment_transaction(
    transaction_path: Path,
    experiment_path: Path,
    experiment: dict[str, Any],
) -> None:
    journal = _load_json_file(transaction_path, max_bytes=4 * 1024 * 1024)
    if (
        journal.get("schema") != TRAINING_TRANSACTION_SCHEMA
        or journal.get("status") != "published"
        or journal.get("epoch") != experiment.get("completedEpochs")
        or journal.get("globalStep") != experiment.get("globalStep")
    ):
        raise ValueError("experiment does not match the published epoch transaction")
    payload = _json_payload(experiment)
    after = {
        "path": experiment_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    existing_after = _optional_artifact_identity(journal.get("experimentAfter"), "experimentAfter")
    existing_record = journal.get("experimentRecord")
    if existing_after is not None or existing_record is not None:
        if existing_after != after or existing_record != experiment:
            raise ValueError("prepared experiment transaction conflicts with retry bytes")
        return
    before = _optional_artifact_identity(journal.get("experimentBefore"), "experimentBefore")
    observed = (
        _identity(experiment_path, max_bytes=64 * 1024 * 1024)
        if experiment_path.is_file() and not experiment_path.is_symlink()
        else None
    )
    if experiment_path.is_symlink() or observed != before:
        raise ValueError("experiment changed since the epoch transaction was prepared")
    journal["experimentAfter"] = after
    journal["experimentRecord"] = experiment
    _write_json_replace(transaction_path, journal)


def _recover_experiment_publication(
    *,
    transaction_path: Path,
    experiment_path: Path,
    journal: dict[str, Any],
) -> None:
    before = _optional_artifact_identity(journal.get("experimentBefore"), "experimentBefore")
    after = _optional_artifact_identity(journal.get("experimentAfter"), "experimentAfter")
    record = journal.get("experimentRecord")
    if after is None or not isinstance(record, dict):
        raise ValueError("training transaction has no recoverable experiment publication")
    payload = _json_payload(record)
    expected = {
        "path": experiment_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    if after != expected:
        raise ValueError("training transaction experiment bytes disagree with their identity")
    if experiment_path.is_symlink():
        raise ValueError("experiment publication must not be a symlink")
    observed = (
        _identity(experiment_path, max_bytes=64 * 1024 * 1024)
        if experiment_path.is_file()
        else None
    )
    if observed != after:
        if observed != before:
            raise ValueError("experiment differs from both journaled publication identities")
        _write_json_replace(experiment_path, record)
        if _identity(experiment_path, max_bytes=64 * 1024 * 1024) != after:
            raise ValueError("recovered experiment bytes differ from transaction evidence")
    _finish_epoch_transaction(transaction_path, experiment_path)


def _finish_epoch_transaction(transaction_path: Path, experiment_path: Path) -> None:
    journal = _load_json_file(transaction_path, max_bytes=4 * 1024 * 1024)
    experiment = _load_json_file(experiment_path, max_bytes=64 * 1024 * 1024)
    experiment_after = _optional_artifact_identity(
        journal.get("experimentAfter"), "experimentAfter"
    )
    checkpoint = _optional_artifact_identity(journal.get("checkpoint"), "checkpoint")
    best_after = _optional_artifact_identity(journal.get("bestAfter"), "bestAfter")
    log_after = _optional_artifact_identity(journal.get("logAfter"), "logAfter")
    if checkpoint is None or best_after is None or log_after is None or experiment_after is None:
        raise ValueError("training transaction final identities are incomplete")
    last_after = {
        "path": "last.pt",
        "sha256": checkpoint["sha256"],
        "size": checkpoint["size"],
    }
    if (
        journal.get("status") != "published"
        or experiment.get("completedEpochs") != journal.get("epoch")
        or experiment.get("globalStep") != journal.get("globalStep")
        or journal.get("experimentRecord") != experiment
        or _identity(experiment_path, max_bytes=64 * 1024 * 1024) != experiment_after
        or experiment.get("artifacts")
        != {
            "bestCheckpoint": best_after,
            "lastCheckpoint": last_after,
            "trainingLog": log_after,
        }
    ):
        raise ValueError("experiment does not commit the pending epoch transaction")
    generation_path: Path | None = None
    if not journal.get("retainCheckpoint"):
        generation_path = transaction_path.parent / str(checkpoint["path"])
        if _identity(generation_path) != checkpoint:
            raise ValueError("ephemeral checkpoint generation changed before finalization")
    _unlink_regular_and_fsync(transaction_path)
    if generation_path is not None:
        _unlink_regular_and_fsync(generation_path)


def _unlink_regular_and_fsync(path: Path) -> None:
    with stable_parent_descriptor(path, create=False) as (parent_descriptor, name):
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise ValueError("training transaction artifact must remain a regular file")
            retire_bound_regular(
                parent_descriptor,
                name,
                status,
                display=path,
                dispose=True,
            )
            os.fsync(parent_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _failpoint(name: str) -> None:
    if os.environ.get(_FAILPOINT_ENV) == name:
        raise RuntimeError(f"training failpoint: {name}")


def _checkpoint_values_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _checkpoint_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _checkpoint_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _validate_resume_outputs(
    *,
    resume_path: Path,
    log_path: Path,
    last_path: Path,
    best_path: Path,
    experiment_path: Path,
    mode: str,
    config_sha256: str,
    dataset_identity: dict[str, Any],
    completed_epoch: int,
    global_step: int,
    pending_journal: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not last_path.is_file() or last_path.is_symlink():
        raise ValueError("resume requires output-dir/last.pt as a regular non-symlink file")
    if resume_path.resolve(strict=True) != last_path.resolve(strict=True):
        raise ValueError("resume checkpoint must be output-dir/last.pt")
    if not best_path.is_file() or best_path.is_symlink():
        raise ValueError("resume requires the prior output-dir/best.pt")
    last_log = _read_last_training_log(log_path)
    if (
        last_log.get("schema") != "phase4_training_log/v1"
        or last_log.get("mode") != mode
        or last_log.get("epoch") != completed_epoch
        or last_log.get("globalStep") != global_step
    ):
        raise ValueError("training log does not end at the resume checkpoint boundary")
    if not experiment_path.exists() and not experiment_path.is_symlink():
        return []
    experiment = _load_json_file(experiment_path, max_bytes=64 * 1024 * 1024)
    if (
        experiment.get("schema") != "phase4_value_experiment/v1"
        or experiment.get("mode") != mode
        or experiment.get("configSha256") != config_sha256
        or experiment.get("datasetIdentity") != dataset_identity
    ):
        raise ValueError("prior experiment record disagrees with the resume checkpoint")
    artifacts = experiment.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("prior experiment artifact identities are invalid")
    current_boundary = (
        experiment.get("completedEpochs") == completed_epoch
        and experiment.get("globalStep") == global_step
    )
    if current_boundary:
        expected_artifacts = {
            "bestCheckpoint": _identity(best_path),
            "lastCheckpoint": _identity(last_path),
            "trainingLog": _identity(log_path, max_bytes=4 * 1024 * 1024 * 1024),
        }
    else:
        if pending_journal is None:
            raise ValueError("prior experiment record disagrees with the resume checkpoint")
        experiment_before = _optional_artifact_identity(
            pending_journal.get("experimentBefore"), "experimentBefore"
        )
        if (
            experiment_before is None
            or _identity(experiment_path, max_bytes=64 * 1024 * 1024) != experiment_before
            or pending_journal.get("epoch") != completed_epoch
            or experiment.get("completedEpochs") != completed_epoch - 1
            or isinstance(experiment.get("globalStep"), bool)
            or not isinstance(experiment.get("globalStep"), int)
            or not 0 <= experiment["globalStep"] <= global_step
        ):
            raise ValueError("stale experiment is not bound by the pending epoch journal")
        expected_artifacts = {
            "bestCheckpoint": pending_journal.get("bestBefore"),
            "lastCheckpoint": pending_journal.get("lastBefore"),
            "trainingLog": pending_journal.get("logBefore"),
        }
    if artifacts != expected_artifacts:
        raise ValueError("prior experiment artifact identities no longer match disk")
    executions = experiment.get("executions")
    if not isinstance(executions, list) or not executions:
        raise ValueError("prior experiment executions are invalid")
    for index, execution in enumerate(executions):
        if (
            not isinstance(execution, dict)
            or isinstance(execution.get("processPeakResidentBytes"), bool)
            or not isinstance(execution.get("processPeakResidentBytes"), int)
            or execution["processPeakResidentBytes"] < 0
        ):
            raise ValueError(f"prior experiment execution {index} is invalid")
        elapsed = execution.get("elapsedSeconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed < 0.0
        ):
            raise ValueError(f"prior experiment execution {index} duration is invalid")
        _validate_mps_memory_observation(execution.get("mpsMemory"), index)
    return executions


def _validate_mps_memory_observation(value: object, execution_index: int) -> None:
    keys = {
        "measurement",
        "currentAllocatedBytes",
        "driverAllocatedBytes",
        "recommendedMaxBytes",
        "measurementError",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"prior experiment execution {execution_index} MPS record is invalid")
    if value["measurement"] not in {
        "not_applicable_cpu",
        "execution_end_after_mps_synchronize",
        "unavailable",
    }:
        raise ValueError(f"prior experiment execution {execution_index} MPS record is invalid")
    for key in ("currentAllocatedBytes", "driverAllocatedBytes", "recommendedMaxBytes"):
        item = value[key]
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
            raise ValueError(f"prior experiment execution {execution_index} MPS record is invalid")
    error = value["measurementError"]
    if error is not None and (not isinstance(error, str) or not error):
        raise ValueError(f"prior experiment execution {execution_index} MPS record is invalid")


def _read_last_training_log(path: Path) -> dict[str, Any]:
    try:
        with stable_regular_descriptor(path) as descriptor:
            before = os.fstat(descriptor)
            size = before.st_size
            if not 0 < size <= 4 * 1024 * 1024 * 1024:
                raise ValueError("training log type or size is invalid")
            offset = max(0, size - 1024 * 1024)
            tail = _pread_exact(descriptor, offset, size - offset, "training log")
    except ArtifactError as error:
        raise ValueError("training log must be a stable regular non-symlink file") from error
    if not tail.endswith(b"\n"):
        raise ValueError("training log is not LF terminated")
    lines = tail.splitlines()
    if not lines or (size > len(tail) and len(lines) == 1):
        raise ValueError("training log final row exceeds the 1 MiB bound")
    try:
        value = json.loads(
            lines[-1],
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("training log final row is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("training log final row must be an object")
    return value


def _load_json_file(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        with stable_regular_descriptor(path) as descriptor:
            size = os.fstat(descriptor).st_size
            if not 0 < size <= max_bytes:
                raise ValueError(f"JSON artifact type or size is invalid: {path}")
            raw = _pread_exact(descriptor, 0, size, f"JSON artifact {path}")
    except ArtifactError as error:
        raise ValueError(f"cannot open stable regular JSON artifact: {path}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _balanced_prefix(
    examples: tuple[TrainingExample, ...], maximum: int
) -> tuple[TrainingExample, ...]:
    if maximum < 3:
        raise ValueError("overfit_examples must be at least three")
    groups = {
        stage: [example for example in examples if example.stage == stage]
        for stage in ("opening", "middlegame", "endgame")
    }
    selected: list[TrainingExample] = []
    while len(selected) < min(maximum, len(examples)):
        progressed = False
        for stage in groups:
            if groups[stage] and len(selected) < maximum:
                selected.append(groups[stage].pop(0))
                progressed = True
        if not progressed:
            break
    return tuple(selected)


@contextmanager
def _training_output_lock(output_dir: Path):
    try:
        with stable_directory_lock(
            output_dir,
            create=True,
            exclusive=True,
            nonblocking=True,
        ):
            yield
    except BlockingIOError as error:
        raise ValueError("another training execution already owns output_dir") from error
    except (ArtifactError, OSError) as error:
        raise ValueError(
            "cannot lock the ancestor-safe training output directory authority"
        ) from error


def _append_bytes(path: Path, payload: bytes, *, create: bool) -> None:
    try:
        with stable_parent_descriptor(path, create=create) as (parent_descriptor, name):
            flags = (
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
            try:
                initial = os.fstat(descriptor)
                if not stat.S_ISREG(initial.st_mode):
                    raise ValueError("training log must be a regular non-symlink file")
                written = 0
                while written < len(payload):
                    count = os.write(descriptor, payload[written:])
                    if count <= 0:
                        raise ValueError("training log append was incomplete")
                    written += count
                os.fsync(descriptor)
                final = os.fstat(descriptor)
                linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                if final.st_size != initial.st_size + len(payload) or _file_state(
                    linked
                ) != _file_state(final):
                    raise ValueError("training log changed concurrently while appended")
                os.fsync(parent_descriptor)
            finally:
                os.close(descriptor)
    except (ArtifactError, OSError) as error:
        raise ValueError("cannot safely open the ancestor-pinned training log") from error


def _write_json_replace(path: Path, value: object) -> None:
    try:
        write_json_atomic(path, value, replace=True)
    except ArtifactError as error:
        raise ValueError(f"cannot atomically replace JSON artifact: {path}") from error


def _write_predictions(
    path: Path,
    examples: tuple[TrainingExample, ...],
    predictions: list[float],
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    output_scale_cp: float,
) -> None:
    if len(examples) != len(predictions) or len(examples) > 100_000:
        raise ValueError("prediction artifact row count is invalid")
    teacher_predictions = [
        (example, prediction)
        for example, prediction in zip(examples, predictions, strict=True)
        if example.source_kind == "phase4_teacher"
        and example.teacher_mask == 1.0
        and example.policy_mask == 1.0
    ]
    if not teacher_predictions:
        raise ValueError("prediction artifact requires at least one teacher-labeled example")
    payloads = []
    for example, prediction in teacher_predictions:
        row = {
            "schema": "phase4_value_prediction/v1",
            "positionId": example.position_id,
            "canonicalSfen": example.sfen,
            "split": example.split,
            "stage": example.stage,
            "teacherScore": {
                "kind": example.teacher_score_kind,
                "value": example.teacher_score_value,
            },
            "modelCp": max(
                -MAX_NON_MATE_CP,
                min(
                    MAX_NON_MATE_CP,
                    normalized_to_centipawns(prediction, output_scale_cp)
                    + (example.residual_baseline_cp or 0),
                ),
            ),
            "candidateGapCp": example.candidate_gap_cp,
            "bestmove": example.bestmove,
            "recordedMove": example.recorded_move,
            "recordedMoveAgrees": example.policy_agreement == 1.0,
            "alreadyTeacherLabeled": example.already_teacher_labeled,
            "checkpointSha256": checkpoint_sha256,
            "configSha256": config_sha256,
        }
        payloads.append(_json_payload(row))
    try:
        with stable_parent_descriptor(path, create=True) as (parent_descriptor, name):
            temporary_name = f".{name}.{secrets.token_hex(12)}"
            descriptor = -1
            temporary_created = False
            temporary_status: os.stat_result | None = None
            try:
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                temporary_created = True
                temporary_status = os.fstat(descriptor)
                for payload in payloads:
                    written = 0
                    while written < len(payload):
                        count = os.write(descriptor, payload[written:])
                        if count <= 0:
                            raise ValueError("prediction artifact write was incomplete")
                        written += count
                os.fsync(descriptor)
                temporary_status = os.fstat(descriptor)
                try:
                    assert temporary_status is not None
                    publish_regular_at(
                        parent_descriptor,
                        temporary_name,
                        name,
                        temporary_status,
                        display=path,
                        replace=False,
                    )
                except FileExistsError:
                    raise FileExistsError(
                        f"refusing to overwrite prediction artifact: {path}"
                    ) from None
                written_status = os.fstat(descriptor)
                linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                if _file_state(linked) != _file_state(written_status):
                    raise ValueError("prediction artifact changed while published")
                temporary_created = False
                os.close(descriptor)
                descriptor = -1
                os.fsync(parent_descriptor)
            except BaseException:
                if descriptor >= 0:
                    os.close(descriptor)
                if temporary_created:
                    assert temporary_status is not None
                    retire_bound_regular(
                        parent_descriptor,
                        temporary_name,
                        temporary_status,
                        display=path,
                    )
                raise
    except ArtifactError as error:
        raise ValueError(f"cannot publish prediction artifact {path}: {error}") from error


def _pread_exact(descriptor: int, offset: int, size: int, context: str) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while observed < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - observed), offset + observed)
        if not chunk:
            raise ValueError(f"{context} ended during its bounded read")
        chunks.append(chunk)
        observed += len(chunk)
    return b"".join(chunks)


def _file_state(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _identity(path: Path, *, max_bytes: int = 512 * 1024 * 1024) -> dict[str, Any]:
    sha256, size = hash_file(path, max_bytes=max_bytes)
    return {"path": path.name, "sha256": sha256, "size": size}
