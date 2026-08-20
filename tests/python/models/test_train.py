import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import open_shogi_training.labeling.artifacts as artifacts_module
import open_shogi_training.models.train as train_module
import pytest
import torch
from open_shogi_training.models.checkpoint import atomic_save_checkpoint, load_checkpoint
from open_shogi_training.models.config import (
    combined_config_sha256,
    load_feature_config,
    load_model_config,
    load_training_config,
    parse_feature_config,
    parse_model_config,
    parse_training_config,
)
from open_shogi_training.models.dataset import DatasetIdentity, LoadedExamples, TrainingExample
from open_shogi_training.models.train import (
    EvaluationSummary,
    _commit_epoch_transaction,
    _load_json_file,
    _recover_epoch_transaction,
    _training_output_lock,
    _write_json_replace,
    _write_predictions,
    compare_checkpoints,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_cpu_stop_and_resume_is_bit_exact_and_never_uses_test_for_training(
    tmp_path: Path,
) -> None:
    feature = replace(
        load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml"),
        name="resume_side_only",
        board_planes=False,
        hand_counts=False,
        king_coordinates=False,
    )
    model = replace(
        load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml"),
        name="resume_tiny",
        hidden_layers=1,
        hidden_dim=4,
        dropout=0.25,
    )
    training = replace(
        load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml"),
        name="resume_two_epochs",
        batch_size=3,
        epochs=2,
        sample_ratio=1.0,
        stage_ratios=(1 / 3, 1 / 3, 1 / 3),
        device="cpu",
        checkpoint_every_epochs=1,
    )
    loaded = _loaded_examples()

    continuous_dir = tmp_path / "continuous"
    train_model(loaded, feature, model, training, continuous_dir)

    resumed_dir = tmp_path / "resumed"
    first_segment = train_model(
        loaded,
        feature,
        model,
        training,
        resumed_dir,
        stop_after_epoch=1,
    )
    train_model(
        loaded,
        feature,
        model,
        training,
        resumed_dir,
        resume_path=first_segment.last_checkpoint,
    )

    continuous = load_checkpoint(continuous_dir / "last.pt")
    resumed = load_checkpoint(resumed_dir / "last.pt")
    _assert_nested_equal(continuous["model_state"], resumed["model_state"])
    _assert_nested_equal(continuous["optimizer_state"], resumed["optimizer_state"])
    _assert_nested_equal(continuous["rng_state"], resumed["rng_state"])
    assert continuous["completed_epoch"] == resumed["completed_epoch"] == 2
    assert continuous["global_step"] == resumed["global_step"]
    assert continuous["best_validation_loss"] == resumed["best_validation_loss"]
    assert (continuous_dir / "training-log.jsonl").read_bytes() == (
        resumed_dir / "training-log.jsonl"
    ).read_bytes()

    experiment = json.loads((resumed_dir / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["testExamplesUsed"] == 0
    assert [(item["startEpoch"], item["endEpoch"]) for item in experiment["executions"]] == [
        (0, 1),
        (1, 2),
    ]
    assert experiment["resources"]["processPeakResidentBytes"] > 0
    assert all(item["elapsedSeconds"] >= 0.0 for item in experiment["executions"])

    changed_test = replace(
        loaded,
        examples=tuple(
            replace(example, teacher_target=999.0, teacher_cp_clipped=999_999.0)
            if example.split == "test"
            else example
            for example in loaded.examples
        ),
    )
    changed_test_dir = tmp_path / "changed-test"
    train_model(changed_test, feature, model, training, changed_test_dir)
    changed = load_checkpoint(changed_test_dir / "last.pt")
    _assert_nested_equal(continuous["model_state"], changed["model_state"])
    _assert_nested_equal(continuous["optimizer_state"], changed["optimizer_state"])

    incompatible = dict(continuous)
    incompatible_training = json.loads(json.dumps(continuous["training_config"]))
    incompatible_training["training"]["teacher_loss_weight"] = 2.0
    incompatible["training_config"] = incompatible_training
    incompatible["config_sha256"] = combined_config_sha256(
        parse_feature_config(incompatible["feature_config"]),
        parse_model_config(incompatible["model_config"]),
        parse_training_config(incompatible_training),
    )
    incompatible_path = tmp_path / "incompatible.pt"
    atomic_save_checkpoint(incompatible_path, incompatible)
    with pytest.raises(ValueError, match="identical target normalization"):
        compare_checkpoints(continuous_dir / "last.pt", incompatible_path, loaded)


def test_training_output_lock_rejects_concurrent_execution_and_symlink(tmp_path: Path) -> None:
    output_dir = tmp_path / "locked"
    with (
        _training_output_lock(output_dir),
        pytest.raises(ValueError, match="already owns"),
        _training_output_lock(output_dir),
    ):
        raise AssertionError("unreachable")

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor-safe"), _training_output_lock(linked_dir):
        raise AssertionError("unreachable")


@pytest.mark.parametrize("writer", ["json", "predictions"])
def test_training_publication_rejects_an_ancestor_swap_without_writing_the_replacement_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    moved = tmp_path / "moved"
    target_name = "state.json" if writer == "json" else "predictions.jsonl"
    real_open = artifacts_module.os.open
    swapped = False

    def swap_before_temporary_open(path, flags, *args, **kwargs):
        nonlocal swapped
        name = str(path)
        if not swapped and name.startswith(f".{target_name}."):
            output.rename(moved)
            output.mkdir()
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifacts_module.os, "open", swap_before_temporary_open)
    target = output / target_name
    with pytest.raises(ValueError, match=r"publish|replace"):
        if writer == "json":
            _write_json_replace(target, {"schema": "test/v1"})
        else:
            _write_predictions(
                target,
                (_example(1, "validation", "opening"),),
                [0.0],
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                output_scale_cp=600.0,
            )

    assert swapped
    assert not target.exists()
    assert (moved / target_name).is_file()


def test_prediction_publication_rejects_temp_collision_and_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "predictions"
    output.mkdir()
    target = output / "rows.jsonl"
    collision = output / ".rows.jsonl.fixed"
    collision.write_bytes(b"do-not-overwrite")
    monkeypatch.setattr(train_module.secrets, "token_hex", lambda _size: "fixed")

    with pytest.raises(FileExistsError):
        _write_predictions(
            target,
            (_example(1, "validation", "opening"),),
            [0.0],
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            output_scale_cp=600.0,
        )
    assert collision.read_bytes() == b"do-not-overwrite"
    assert not target.exists()

    collision.unlink()
    target.write_bytes(b"immutable-existing-target")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_predictions(
            target,
            (_example(1, "validation", "opening"),),
            [0.0],
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            output_scale_cp=600.0,
        )
    assert target.read_bytes() == b"immutable-existing-target"
    assert not collision.exists()


def test_prediction_cleanup_preserves_a_foreign_temp_relink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "predictions"
    output.mkdir()
    target = output / "rows.jsonl"
    owned = output / "owned-original-temp"
    real_publish = train_module.publish_regular_at

    def relink_target_after_publication(*args, **kwargs) -> None:
        real_publish(*args, **kwargs)
        target.rename(owned)
        target.write_bytes(b"foreign-do-not-delete")

    monkeypatch.setattr(train_module, "publish_regular_at", relink_target_after_publication)
    with pytest.raises(ValueError, match="changed while published"):
        _write_predictions(
            target,
            (_example(1, "validation", "opening"),),
            [0.0],
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            output_scale_cp=600.0,
        )

    assert owned.is_file()
    assert target.read_bytes() == b"foreign-do-not-delete"


def test_training_transaction_cleanup_preserves_a_foreign_entry_relink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "epoch-transaction.json"
    target.write_bytes(b"owned")
    owned = tmp_path / "owned-transaction"
    real_retire = train_module.retire_bound_regular

    def relink_before_unlink(parent_descriptor, name, expected, *, display, dispose=False):
        assert dispose is True
        target.rename(owned)
        target.write_bytes(b"foreign-do-not-delete")
        return real_retire(
            parent_descriptor,
            name,
            expected,
            display=display,
            dispose=dispose,
        )

    monkeypatch.setattr(train_module, "retire_bound_regular", relink_before_unlink)
    with pytest.raises(artifacts_module.ArtifactError, match="changed before retirement"):
        train_module._unlink_regular_and_fsync(target)

    assert owned.read_bytes() == b"owned"
    assert target.read_bytes() == b"foreign-do-not-delete"


def test_training_json_reader_rejects_final_entry_relink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"value":1}\n', encoding="utf-8")
    moved = tmp_path / "state.original.json"
    real_pread = train_module.os.pread
    swapped = False

    def swap_after_read(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal swapped
        result = real_pread(descriptor, size, offset)
        if not swapped:
            path.rename(moved)
            path.write_text('{"value":2}\n', encoding="utf-8")
            swapped = True
        return result

    monkeypatch.setattr(train_module.os, "pread", swap_after_read)
    with pytest.raises(ValueError, match="stable regular"):
        _load_json_file(path, max_bytes=1_024)


@pytest.mark.parametrize(
    "failpoint",
    ["after_journal", "after_best", "after_last", "after_log", "after_experiment"],
)
def test_every_epoch_transaction_window_is_resumable_with_sparse_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failpoint: str
) -> None:
    feature = replace(
        load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml"),
        name="transaction_side_only",
        board_planes=False,
        hand_counts=False,
        king_coordinates=False,
    )
    model = replace(
        load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml"),
        name="transaction_tiny",
        hidden_layers=1,
        hidden_dim=4,
        dropout=0.0,
    )
    training = replace(
        load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml"),
        name="transaction_two_epochs",
        epochs=2,
        batch_size=3,
        sample_ratio=1.0,
        stage_ratios=(1 / 3, 1 / 3, 1 / 3),
        device="cpu",
        checkpoint_every_epochs=20,
    )
    output = tmp_path / failpoint
    monkeypatch.setenv("OPEN_SHOGI_TRAINING_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        train_model(
            _loaded_examples(),
            feature,
            model,
            training,
            output,
            stop_after_epoch=1,
        )
    monkeypatch.delenv("OPEN_SHOGI_TRAINING_FAILPOINT")

    result = train_model(
        _loaded_examples(),
        feature,
        model,
        training,
        output,
        resume_path=output / "last.pt",
    )

    assert load_checkpoint(result.last_checkpoint)["completed_epoch"] == 2
    assert not (output / "checkpoint-epoch-0001.pt").exists()
    assert not (output / "checkpoint-epoch-0002.pt").exists()
    assert not (output / "epoch-transaction.json").exists()
    log = [
        json.loads(line)
        for line in (output / "training-log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["epoch"] for row in log] == [1, 2]
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["completedEpochs"] == 2
    assert (
        experiment["artifacts"]["lastCheckpoint"]["sha256"]
        == hashlib.sha256((output / "last.pt").read_bytes()).hexdigest()
    )
    assert not any(path.name.startswith(".") and ".tmp" in path.name for path in output.iterdir())


@pytest.mark.parametrize("conflict", [False, True])
def test_epoch_recovery_completes_only_the_journaled_partial_first_log_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conflict: bool,
) -> None:
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    output = tmp_path / ("partial-conflict" if conflict else "partial-recovery")
    monkeypatch.setenv("OPEN_SHOGI_TRAINING_FAILPOINT", "after_journal")
    with pytest.raises(RuntimeError, match="after_journal"):
        train_model(_loaded_examples(), feature, model, training, output)
    monkeypatch.delenv("OPEN_SHOGI_TRAINING_FAILPOINT")

    journal = json.loads((output / "epoch-transaction.json").read_text(encoding="utf-8"))
    payload = train_module._json_payload(journal["logRecord"])
    prefix = payload[: len(payload) // 2]
    if conflict:
        prefix = b"x" + prefix[1:]
    (output / "training-log.jsonl").write_bytes(prefix)

    if conflict:
        with pytest.raises(ValueError, match="partial append conflicts"):
            _recover_epoch_transaction(
                transaction_path=output / "epoch-transaction.json",
                log_path=output / "training-log.jsonl",
                last_path=output / "last.pt",
                best_path=output / "best.pt",
                experiment_path=output / "experiment.json",
            )
        assert (output / "training-log.jsonl").read_bytes() == prefix
        return

    _recover_epoch_transaction(
        transaction_path=output / "epoch-transaction.json",
        log_path=output / "training-log.jsonl",
        last_path=output / "last.pt",
        best_path=output / "best.pt",
        experiment_path=output / "experiment.json",
    )
    assert (output / "training-log.jsonl").read_bytes() == payload
    train_model(
        _loaded_examples(),
        feature,
        model,
        training,
        output,
        resume_path=output / "last.pt",
    )
    assert not (output / "epoch-transaction.json").exists()
    assert (
        json.loads((output / "experiment.json").read_text(encoding="utf-8"))["completedEpochs"] == 1
    )


@pytest.mark.parametrize("failpoint", ["after_journal", "after_best", "after_last", "after_log"])
def test_non_improving_epoch_recovery_rejects_an_unbound_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failpoint: str
) -> None:
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    output = tmp_path / f"best-binding-{failpoint}"
    train_model(_loaded_examples(), feature, model, training, output)
    payload = load_checkpoint(output / "last.pt")
    log_record = json.loads(
        (output / "training-log.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    transaction = output / "epoch-transaction.json"
    monkeypatch.setenv("OPEN_SHOGI_TRAINING_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        _commit_epoch_transaction(
            transaction_path=transaction,
            log_path=output / "training-log.jsonl",
            last_path=output / "last.pt",
            best_path=output / "best.pt",
            payload=payload,
            log_record=log_record,
            retain_generation=False,
            publish_best=False,
            create_log=False,
        )
    monkeypatch.delenv("OPEN_SHOGI_TRAINING_FAILPOINT")
    (output / "best.pt").write_bytes(b"unbound checkpoint")

    with pytest.raises(ValueError, match="bound prior identity"):
        _recover_epoch_transaction(
            transaction_path=transaction,
            log_path=output / "training-log.jsonl",
            last_path=output / "last.pt",
            best_path=output / "best.pt",
            experiment_path=output / "experiment.json",
        )


@pytest.mark.parametrize(
    "failpoint",
    ["after_journal", "after_best", "after_last", "after_log", "after_experiment"],
)
def test_non_improving_epoch_recovers_every_durable_window_without_replacing_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failpoint: str
) -> None:
    feature = replace(
        load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml"),
        name="non_improving_side_only",
        board_planes=False,
        hand_counts=False,
        king_coordinates=False,
    )
    model = replace(
        load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml"),
        name="non_improving_tiny",
        hidden_layers=1,
        hidden_dim=4,
        dropout=0.0,
    )
    training = replace(
        load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml"),
        name="non_improving_two_epochs",
        epochs=2,
        batch_size=3,
        sample_ratio=1.0,
        stage_ratios=(1 / 3, 1 / 3, 1 / 3),
        device="cpu",
        checkpoint_every_epochs=20,
    )
    output = tmp_path / f"non-improving-{failpoint}"
    first = train_model(
        _loaded_examples(),
        feature,
        model,
        training,
        output,
        stop_after_epoch=1,
    )
    best_before = (output / "best.pt").read_bytes()

    def deliberately_worse_epoch(*_args, split: str, optimizer=None, **_kwargs):
        summary = EvaluationSummary(
            split=split,
            examples=1,
            batches=1,
            teacher_examples=1,
            outcome_examples=1,
            policy_examples=1,
            ranking_pairs=0,
            total_loss=1.0e20,
            teacher_loss=1.0e20,
            game_result_loss=0.0,
            policy_agreement_loss=0.0,
            ranking_loss=0.0,
            teacher_mae_cp=1.0e20,
            policy_accuracy=0.0,
        )
        return summary, 0

    monkeypatch.setattr(train_module, "_run_epoch", deliberately_worse_epoch)
    monkeypatch.setenv("OPEN_SHOGI_TRAINING_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        train_model(
            _loaded_examples(),
            feature,
            model,
            training,
            output,
            resume_path=first.last_checkpoint,
        )
    monkeypatch.delenv("OPEN_SHOGI_TRAINING_FAILPOINT")

    train_model(
        _loaded_examples(),
        feature,
        model,
        training,
        output,
        resume_path=output / "last.pt",
    )

    assert (output / "best.pt").read_bytes() == best_before
    assert load_checkpoint(output / "last.pt")["completed_epoch"] == 2
    log = [
        json.loads(line)
        for line in (output / "training-log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["epoch"] for row in log] == [1, 2]
    assert log[-1]["improved"] is False
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["completedEpochs"] == 2
    assert (
        experiment["artifacts"]["bestCheckpoint"]["sha256"]
        == hashlib.sha256(best_before).hexdigest()
    )
    assert not (output / "epoch-transaction.json").exists()
    assert not any(path.name.startswith(".") and ".tmp" in path.name for path in output.iterdir())


def test_prediction_artifact_excludes_outcome_only_replay_placeholders(tmp_path: Path) -> None:
    teacher = _example(1, "validation", "opening")
    replay = replace(
        _example(2, "validation", "middlegame"),
        teacher_target=0.0,
        teacher_cp_clipped=0.0,
        teacher_mask=0.0,
        policy_agreement=0.0,
        policy_mask=0.0,
        teacher_score_kind="unlabeled",
        teacher_score_value=None,
        bestmove="unlabeled",
        recorded_move="unlabeled",
        candidate_gap_cp=None,
        already_teacher_labeled=False,
        source_kind="phase6_replay",
        source_manifest_sha256="d" * 64,
        source_generation_id="generation-1",
        source_game_id="game-1",
        source_ply=1,
    )
    output = tmp_path / "predictions.jsonl"

    _write_predictions(
        output,
        (teacher, replay),
        [0.25, -0.75],
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        output_scale_cp=1200.0,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["positionId"] == teacher.position_id
    assert rows[0]["teacherScore"]["kind"] == "cp"


def test_residual_prediction_artifact_reports_the_combined_absolute_score(tmp_path: Path) -> None:
    example = replace(
        _example(1, "validation", "opening"),
        teacher_target=0.0,
        teacher_cp_clipped=0.0,
        residual_baseline_cp=125,
    )
    output = tmp_path / "residual-predictions.jsonl"

    _write_predictions(
        output,
        (example,),
        [0.25],
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        output_scale_cp=1200.0,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["modelCp"] == 425


def _loaded_examples() -> LoadedExamples:
    examples = []
    stages = ("opening", "middlegame", "endgame")
    for index in range(6):
        examples.append(_example(index, "train", stages[index % 3]))
    examples.extend((_example(6, "validation", "opening"), _example(7, "validation", "endgame")))
    examples.append(_example(8, "test", "middlegame"))
    return LoadedExamples(
        examples=tuple(examples),
        identity=DatasetIdentity(
            dataset_manifest_sha256="a" * 64,
            positions_sha256="b" * 64,
            labels_sha256="c" * 64,
            label_manifest_sha256="d" * 64,
            replay_manifest_sha256=None,
        ),
        counts_by_split={"train": 6, "validation": 2, "test": 1},
        counts_by_stage={"opening": 3, "middlegame": 3, "endgame": 3},
    )


def _example(index: int, split: str, stage: str) -> TrainingExample:
    teacher_cp = float((index - 4) * 100)
    return TrainingExample(
        position_id=f"{index:064x}",
        sfen=STARTPOS,
        split=split,
        stage=stage,
        position_index=index,
        teacher_target=teacher_cp / 1200.0,
        teacher_cp_clipped=teacher_cp,
        teacher_mask=1.0,
        policy_agreement=float(index % 2),
        policy_mask=1.0,
        outcome_target=float((index % 3) - 1),
        outcome_mask=1.0,
        teacher_score_kind="cp",
        teacher_score_value=int(teacher_cp),
        bestmove="7g7f",
        recorded_move="7g7f" if index % 2 else "2g2f",
        candidate_gap_cp=20,
        already_teacher_labeled=True,
        source_manifest_sha256="d" * 64,
        source_game_id="e" * 64,
        source_ply=index,
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right
