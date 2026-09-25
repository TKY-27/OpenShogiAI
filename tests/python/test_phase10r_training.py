from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import open_shogi_training.phase10r_campaign as campaign
import pytest
import torch
from open_shogi_training.phase10r_model import (
    VARIANT_PAIR,
    VARIANT_PRIMARY,
    expected_parameter_count,
    parse_osaval02,
)
from open_shogi_training.phase10r_training import (
    Phase10RExample,
    Phase10RModel,
    Phase10RTrainingError,
    TrainingConfig,
    export_osaval02_artifact,
    loss_for_examples,
    run_bounded_training,
    select_device,
)

SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
MANIFEST_SHA = "1" * 64


def _example(index: int = 0, *, source: str = "aobazero") -> Phase10RExample:
    return Phase10RExample(
        sfen=SFEN,
        source=source,
        artifact_id="approved-artifact",
        record_id=f"record-{index}",
        split="train",
        legal_moves=("7g7f", "2g2f"),
        played_move="7g7f" if index % 2 == 0 else "2g2f",
        wdl=2 if index % 2 == 0 else 0,
        wdl_mask=True,
    )


def _load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_nested_tensors_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_tensors_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_tensors_equal(left_value, right_value)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    else:
        assert left == right


@pytest.mark.parametrize(
    ("variant", "parameter_count"),
    [
        (VARIANT_PAIR, 2_413_321),
        (VARIANT_PRIMARY, 2_676_618),
    ],
)
def test_frozen_model_shapes_and_parameter_counts(variant: str, parameter_count: int) -> None:
    model = Phase10RModel(variant, seed=7)

    assert expected_parameter_count(variant) == parameter_count
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert model.forward_example(_example())["values"].shape == (
        8 if variant == VARIANT_PAIR else 9,
    )


def test_masks_and_source_semantics_fail_closed() -> None:
    with pytest.raises(Phase10RTrainingError, match="legal move mask"):
        Phase10RExample(
            sfen=SFEN,
            source="aobazero",
            artifact_id="a",
            record_id="r",
            split="train",
            played_move="7g7f",
        ).validate()

    with pytest.raises(Phase10RTrainingError, match="non-Apery"):
        Phase10RExample(
            sfen=SFEN,
            source="aobazero",
            artifact_id="a",
            record_id="r",
            split="train",
            score_cp=100.0,
            score_mask=True,
        ).validate()

    with pytest.raises(Phase10RTrainingError, match="pair variant"):
        loss_for_examples(
            Phase10RModel(VARIANT_PAIR, seed=7),
            [
                Phase10RExample(
                    sfen=SFEN,
                    source="openshogiai_apery_teacher",
                    artifact_id="a",
                    record_id="r",
                    split="train",
                    score_cp=100.0,
                    score_mask=True,
                )
            ],
        )


def test_device_selection_records_cpu_fallback_or_mps() -> None:
    receipt = select_device("mps", allow_cpu_fallback=True)

    if receipt.selected == "cpu":
        assert receipt.fallback is True
        assert receipt.reason
    else:
        assert receipt.selected == "mps"
        assert receipt.fallback is False


def test_teacher_ranking_and_masked_uncertainty_losses_are_source_local() -> None:
    ranking = Phase10RExample(
        sfen=SFEN,
        source="openshogiai_apery_teacher",
        artifact_id="teacher-artifact",
        record_id="teacher-record",
        split="train",
        legal_moves=("7g7f", "2g2f"),
        ranking_scores=(300.0, -100.0),
        ranking_mask=True,
        teacher_identity="apery-2.0.0-config-a",
    )
    ranking_loss, ranking_metrics = loss_for_examples(
        Phase10RModel(VARIANT_PAIR, seed=3), [ranking]
    )
    assert torch.isfinite(ranking_loss)
    assert ranking_metrics["ranking_weight"] == 1.0

    uncertainty = replace(_example(), uncertainty_mask=True)
    uncertainty_loss, uncertainty_metrics = loss_for_examples(
        Phase10RModel(VARIANT_PAIR, seed=4), [uncertainty]
    )
    assert torch.isfinite(uncertainty_loss)
    assert uncertainty_metrics["uncertainty_wdl_weight"] == 1.0


def test_deterministic_repeat_and_exact_checkpoint_resume(tmp_path: Path) -> None:
    examples = [_example(index) for index in range(4)]
    config_two = TrainingConfig(
        variant_id=VARIANT_PAIR,
        seed=123,
        requested_device="cpu",
        max_steps=2,
        batch_size=2,
        checkpoint_interval_steps=1,
        minimum_free_bytes=0,
        enforce_disk=False,
    )
    first = run_bounded_training(
        examples,
        output_dir=tmp_path / "first",
        manifest_sha256=MANIFEST_SHA,
        config=config_two,
    )
    repeat = run_bounded_training(
        examples,
        output_dir=tmp_path / "repeat",
        manifest_sha256=MANIFEST_SHA,
        config=config_two,
    )
    first_checkpoint = _load_checkpoint(Path(first["checkpoint"]))
    repeat_checkpoint = _load_checkpoint(Path(repeat["checkpoint"]))
    _assert_nested_tensors_equal(first_checkpoint, repeat_checkpoint)

    config_four = TrainingConfig(
        variant_id=VARIANT_PAIR,
        seed=123,
        requested_device="cpu",
        max_steps=4,
        batch_size=2,
        checkpoint_interval_steps=1,
        minimum_free_bytes=0,
        enforce_disk=False,
    )
    resumed = run_bounded_training(
        examples,
        output_dir=tmp_path / "resumed",
        manifest_sha256=MANIFEST_SHA,
        config=config_four,
        resume_from=Path(first["checkpoint"]),
    )
    uninterrupted = run_bounded_training(
        examples,
        output_dir=tmp_path / "uninterrupted",
        manifest_sha256=MANIFEST_SHA,
        config=config_four,
    )
    resumed_checkpoint = _load_checkpoint(Path(resumed["checkpoint"]))
    uninterrupted_checkpoint = _load_checkpoint(Path(uninterrupted["checkpoint"]))
    _assert_nested_tensors_equal(resumed_checkpoint, uninterrupted_checkpoint)


def test_campaign_completed_checkpoint_keeps_stream_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        campaign,
        "_stream_batches",
        lambda path, *, expected_rows, start_cursor, batch_size: iter(
            [([_example()], expected_rows)]
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_resource_guard",
        lambda data_root: {"disk_passed": True, "peak_rss_bytes": 0},
    )

    model = Phase10RModel(VARIANT_PAIR, seed=123)
    optimizer, scheduler = campaign._new_optimizer(model)
    result = campaign._run_stage(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stage_id=campaign.STAGE_ONE,
        stage_dir=tmp_path / "stage",
        train_path=tmp_path / "train.jsonl",
        expected_rows=1,
        manifest_sha256=MANIFEST_SHA,
        data_root=tmp_path,
        resume=False,
    )

    checkpoint = _load_checkpoint(Path(result["checkpoint"]))
    assert checkpoint["completed"] is True
    assert checkpoint["stream_index"] == 1


def _bounded_config(max_steps: int) -> TrainingConfig:
    return TrainingConfig(
        variant_id=VARIANT_PAIR,
        seed=123,
        requested_device="cpu",
        max_steps=max_steps,
        batch_size=2,
        checkpoint_interval_steps=1,
        minimum_free_bytes=0,
        enforce_disk=False,
    )


def test_checkpoint_receipt_matches_saved_digest(tmp_path: Path) -> None:
    run_bounded_training(
        [_example(index) for index in range(4)],
        output_dir=tmp_path / "run",
        manifest_sha256=MANIFEST_SHA,
        config=_bounded_config(1),
    )

    for name in ("last.pt", "best.pt"):
        receipt = tmp_path / "run" / f"{name}.sha256"
        assert receipt.is_file()
        assert not receipt.is_symlink()
        content = receipt.read_text(encoding="ascii")
        assert content.endswith("\n")
        digest = content[: -len("\n")]
        assert len(digest) == 64
        assert digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")
        assert digest == hashlib.sha256((tmp_path / "run" / name).read_bytes()).hexdigest()


def test_resume_refuses_tampered_checkpoint_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = [_example(index) for index in range(4)]
    first = run_bounded_training(
        examples,
        output_dir=tmp_path / "first",
        manifest_sha256=MANIFEST_SHA,
        config=_bounded_config(2),
    )
    checkpoint = Path(first["checkpoint"])

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load must not run before the digest receipt verifies")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(Phase10RTrainingError, match="digest mismatches its receipt"):
        run_bounded_training(
            examples,
            output_dir=tmp_path / "resumed-tampered",
            manifest_sha256=MANIFEST_SHA,
            config=_bounded_config(4),
            resume_from=checkpoint,
        )
    (tmp_path / "first" / "last.pt.sha256").unlink()
    with pytest.raises(Phase10RTrainingError, match="digest receipt is missing"):
        run_bounded_training(
            examples,
            output_dir=tmp_path / "resumed-unreceipted",
            manifest_sha256=MANIFEST_SHA,
            config=_bounded_config(4),
            resume_from=checkpoint,
        )


def test_campaign_checkpoint_receipt_gates_resume_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Phase10RModel(VARIANT_PAIR, seed=123)
    optimizer, scheduler = campaign._new_optimizer(model)
    checkpoint = tmp_path / "stage" / "last.pt"
    campaign._save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        manifest_sha256=MANIFEST_SHA,
        stage_id=campaign.STAGE_ONE,
        variant_id=model.variant_id,
        step=3,
        cursor=7,
        metrics={"loss_sum": 1.5, "active_examples": 4},
        completed=False,
    )

    def resumed_state() -> dict[str, object]:
        fresh = Phase10RModel(VARIANT_PAIR, seed=123)
        fresh_optimizer, fresh_scheduler = campaign._new_optimizer(fresh)
        return campaign._load_checkpoint(
            checkpoint,
            fresh,
            fresh_optimizer,
            fresh_scheduler,
            manifest_sha256=MANIFEST_SHA,
            stage_id=campaign.STAGE_ONE,
            variant_id=model.variant_id,
            expected_rows=100,
        )

    state = resumed_state()
    assert state["step"] == 3
    assert state["cursor"] == 7
    assert state["completed"] is False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load must not run before the digest receipt verifies")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(campaign.Phase10RCampaignError, match="digest mismatches its receipt"):
        resumed_state()
    (tmp_path / "stage" / "last.pt.sha256").unlink()
    with pytest.raises(campaign.Phase10RCampaignError, match="digest receipt is missing"):
        resumed_state()


def test_primary_float_and_int8_exports_parse_without_overwrite(tmp_path: Path) -> None:
    model = Phase10RModel(VARIANT_PRIMARY, seed=9)
    float_result = export_osaval02_artifact(
        model,
        tmp_path / "primary-float.osaval02",
        quantization="float32",
        dataset_manifest_sha256=MANIFEST_SHA,
        training_run_reference="bounded-test",
        git_commit="a" * 40,
    )
    int8_result = export_osaval02_artifact(
        model,
        tmp_path / "primary-int8.osaval02",
        quantization="int8",
        dataset_manifest_sha256=MANIFEST_SHA,
        training_run_reference="bounded-test",
        git_commit="a" * 40,
    )

    assert (
        parse_osaval02((tmp_path / "primary-float.osaval02").read_bytes()).variant_id
        == VARIANT_PRIMARY
    )
    assert parse_osaval02((tmp_path / "primary-int8.osaval02").read_bytes()).quantization == "int8"
    assert float_result["status"] == int8_result["status"] == "passed"
    with pytest.raises(Phase10RTrainingError, match="overwrite"):
        export_osaval02_artifact(
            model,
            tmp_path / "primary-float.osaval02",
            quantization="float32",
            dataset_manifest_sha256=MANIFEST_SHA,
            training_run_reference="bounded-test",
            git_commit="a" * 40,
        )
