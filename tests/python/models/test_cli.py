import json
from pathlib import Path

import pytest
import torch
from open_shogi_training.models import cli as cli_module
from open_shogi_training.models.checkpoint import atomic_save_checkpoint, build_checkpoint
from open_shogi_training.models.cli import _validate_phase4_prediction_row, build_parser, main
from open_shogi_training.models.config import (
    load_feature_config,
    load_model_config,
    load_training_config,
)
from open_shogi_training.models.dataset import _load_training_examples_for_test
from open_shogi_training.models.features import input_dimension
from open_shogi_training.models.network import ValueModel

from .test_dataset import _dataset_fixture

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_test_examples(
    labels,
    positions,
    dataset_manifest,
    training,
    *,
    label_manifest_path,
    replay_manifest_path=None,
    include_replay_test=False,
    target_semantics="pure-value",
    residual_baseline_path=None,
    repository_root=None,
):
    del label_manifest_path, repository_root
    return _load_training_examples_for_test(
        labels,
        positions,
        dataset_manifest,
        training,
        replay_manifest_path=replay_manifest_path,
        include_replay_test=include_replay_test,
        target_semantics=target_semantics,
        residual_baseline_path=residual_baseline_path,
    )


def test_parser_exposes_all_model_lab_commands() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices

    assert set(choices) == {
        "train",
        "smoke",
        "overfit",
        "validate",
        "test",
        "compare",
        "feature-ablation",
        "export",
        "describe",
        "validate-config",
        "config-diff",
        "compare-predictions",
        "build-residual-baseline",
        "arena-run",
        "arena-verify",
    }
    help_text = parser.format_help()
    assert "run a bounded two-batch training smoke check" in help_text
    assert "overfit a deterministic balanced tiny subset" in help_text
    assert "compare bounded neural, handcrafted" in help_text
    assert "recompute the Phase 5 arena manifest" in help_text


def test_describe_and_config_diff_commands_emit_machine_json(capsys) -> None:
    assert main(["describe"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["inputDimension"] == 2287
    assert description["parameters"]["exportedTotal"] <= 16_000_000
    assert description["estimatedOperationsPerPosition"] > description["exportedParameters"]
    assert (
        description["estimatedArtifactBytes"]["float32"]
        > description["estimatedArtifactBytes"]["int8"]
    )

    assert main(["validate-config"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is True
    assert validation["expectedTeacherLabels"] == 10_000
    assert validation["stageBoundariesBasisPoints"] == [3333, 6667]

    assert (
        main(
            [
                "config-diff",
                "--kind",
                "training",
                "--left",
                str(PROJECT_ROOT / "configs/training/value_v0_smoke.toml"),
                "--right",
                str(PROJECT_ROOT / "configs/training/value_v0_full_initial.toml"),
            ]
        )
        == 0
    )
    difference = json.loads(capsys.readouterr().out)
    assert any(change["field"] == "training.epochs" for change in difference["changes"])

    assert (
        main(
            [
                "config-diff",
                "--kind",
                "training",
                "--left",
                str(PROJECT_ROOT / "configs/training/value_v0_smoke.toml"),
                "--right",
                str(PROJECT_ROOT / "configs/training/value_v0_full_initial.toml"),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    markdown = capsys.readouterr().out
    assert markdown.startswith("# Training configuration difference\n")
    assert "| `training.epochs` | `1` | `50` |" in markdown


def test_compare_predictions_validates_identity_and_summarizes_delta(
    tmp_path: Path, capsys
) -> None:
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    left = tmp_path / "neural.jsonl"
    right = tmp_path / "handcrafted.jsonl"
    left_rows = [
        {
            "schema": "phase5_model_inference/v1",
            "modelArtifactSha256": "a" * 64,
            "modelPayloadSha256": "b" * 64,
            "index": index,
            "sfen": sfen,
            "scoreCp": score,
            "elapsedNs": 1,
        }
        for index, score in enumerate((10, -5))
    ]
    right_rows = [
        {
            "schema": "phase5_handcrafted_inference/v1",
            "evaluatorProfile": "handcrafted-baseline",
            "index": index,
            "sfen": sfen,
            "scoreCp": score,
            "elapsedNs": 2,
        }
        for index, score in enumerate((13, -5))
    ]
    for path, rows in ((left, left_rows), (right, right_rows)):
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    assert main(["compare-predictions", "--left", str(left), "--right", str(right)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 2
    assert result["rightMinusLeft"] == {
        "meanCp": 1.5,
        "meanAbsoluteCp": 1.5,
        "maximumAbsoluteCp": 3,
        "rightLower": 0,
        "equal": 1,
        "rightHigher": 1,
    }

    right_rows[1]["sfen"] = sfen.replace(" b - 1", " w - 1")
    right.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in right_rows
        ),
        encoding="utf-8",
    )
    assert main(["compare-predictions", "--left", str(left), "--right", str(right)]) == 2
    assert "SFEN identity" in capsys.readouterr().err

    right_rows[1]["sfen"] = sfen
    right_rows[1]["evaluatorProfile"] = "handcrafted"
    right.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in right_rows
        ),
        encoding="utf-8",
    )
    assert main(["compare-predictions", "--left", str(left), "--right", str(right)]) == 2
    assert "evaluatorProfile" in capsys.readouterr().err

    right_rows[1]["evaluatorProfile"] = "handcrafted-baseline"
    right_rows[1]["scoreCp"] = 2**31
    right.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in right_rows
        ),
        encoding="utf-8",
    )
    assert main(["compare-predictions", "--left", str(left), "--right", str(right)]) == 2
    assert "scoreCp" in capsys.readouterr().err

    right_rows[1]["scoreCp"] = -5
    right.write_bytes(
        b"\r\n".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True).encode() for row in right_rows
        )
        + b"\r\n"
    )
    assert main(["compare-predictions", "--left", str(left), "--right", str(right)]) == 2
    assert "LF terminated" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("teacherScore", {"kind": "unlabeled", "value": None}, "teacherScore"),
        ("modelCp", 29_000, "modelCp"),
        ("candidateGapCp", 1_000_001, "candidateGapCp"),
        ("bestmove", "not-a-move", "bestmove"),
        ("recordedMoveAgrees", False, "inconsistent"),
        ("alreadyTeacherLabeled", False, "not teacher-labeled"),
    ],
)
def test_phase4_prediction_rows_reject_false_or_out_of_contract_fields(
    field: str, value: object, match: str
) -> None:
    row = {
        "schema": "phase4_value_prediction/v1",
        "positionId": "a" * 64,
        "canonicalSfen": ("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"),
        "split": "validation",
        "stage": "opening",
        "teacherScore": {"kind": "cp", "value": 10},
        "modelCp": 5,
        "candidateGapCp": 2,
        "bestmove": "7g7f",
        "recordedMove": "7g7f",
        "recordedMoveAgrees": True,
        "alreadyTeacherLabeled": True,
        "checkpointSha256": "b" * 64,
        "configSha256": "c" * 64,
    }
    row[field] = value

    with pytest.raises(ValueError, match=match):
        _validate_phase4_prediction_row(row, 1)


def test_export_command_rejects_checkpoint_with_false_config_identity(
    tmp_path: Path, capsys
) -> None:
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model_config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    model = ValueModel(input_dimension(feature), model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    checkpoint = build_checkpoint(
        completed_epoch=1,
        global_step=1,
        best_validation_loss=1.0,
        model=model,
        optimizer=optimizer,
        feature_config=feature.as_dict(),
        model_config=model_config.as_dict(),
        training_config=training.as_dict(),
        config_sha256="0" * 64,
        dataset_identity={
            "dataset_manifest_sha256": "a" * 64,
            "positions_sha256": "b" * 64,
            "labels_sha256": "c" * 64,
            "label_manifest_sha256": "d" * 64,
            "replay_manifest_sha256": None,
        },
        device=torch.device("cpu"),
        runtime={"torch": str(torch.__version__)},
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    atomic_save_checkpoint(checkpoint_path, checkpoint)

    assert (
        main(
            [
                "export",
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(tmp_path / "export"),
            ]
        )
        == 2
    )
    assert "config hash does not match" in capsys.readouterr().err


def test_bounded_cli_train_resume_evaluate_compare_ablate_and_export(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli_module, "require_clean_head", lambda _root: "a" * 40)
    labels, positions, manifest = _dataset_fixture(tmp_path)
    output_dir = tmp_path / "training"
    dataset_arguments = [
        "--labels",
        str(labels),
        "--label-manifest",
        str(tmp_path / "label-manifest.json"),
        "--positions",
        str(positions),
        "--dataset-manifest",
        str(manifest),
    ]
    training_arguments = [
        "--training",
        str(PROJECT_ROOT / "configs/training/value_v0_overfit.toml"),
        *dataset_arguments,
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(
        cli_module,
        "load_training_examples",
        _load_test_examples,
    )

    assert main(["train", *training_arguments, "--stop-after-epoch", "1"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["completedEpochs"] == 1
    assert (
        main(
            [
                "train",
                *training_arguments,
                "--resume",
                first["lastCheckpoint"],
                "--stop-after-epoch",
                "2",
            ]
        )
        == 0
    )
    resumed = json.loads(capsys.readouterr().out)
    checkpoint = resumed["lastCheckpoint"]
    assert resumed["completedEpochs"] == 2

    predictions = tmp_path / "validation-predictions.jsonl"
    assert (
        main(
            [
                "validate",
                "--checkpoint",
                checkpoint,
                *dataset_arguments,
                "--predictions-output",
                str(predictions),
            ]
        )
        == 0
    )
    validation = json.loads(capsys.readouterr().out)
    assert validation["metrics"]["split"] == "validation"
    assert predictions.is_file()
    assert (
        main(
            [
                "compare-predictions",
                "--left",
                str(predictions),
                "--right",
                str(predictions),
            ]
        )
        == 0
    )
    prediction_comparison = json.loads(capsys.readouterr().out)
    assert prediction_comparison["left"]["schema"] == "phase4_value_prediction/v1"
    assert prediction_comparison["rightMinusLeft"]["maximumAbsoluteCp"] == 0

    assert (
        main(
            [
                "test",
                "--final-test",
                "--checkpoint",
                checkpoint,
                *dataset_arguments,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["metrics"]["split"] == "test"

    assert (
        main(
            [
                "compare",
                "--left",
                checkpoint,
                "--right",
                checkpoint,
                *dataset_arguments,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["totalLossDeltaRightMinusLeft"] == 0.0

    assert (
        main(
            [
                "feature-ablation",
                "--checkpoint",
                checkpoint,
                *dataset_arguments,
            ]
        )
        == 0
    )
    ablation = json.loads(capsys.readouterr().out)
    assert len(ablation["ablations"]) == 4

    export_dir = tmp_path / "export"
    assert (
        main(
            [
                "export",
                "--checkpoint",
                checkpoint,
                "--output-dir",
                str(export_dir),
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["artifacts"][0]["quantization"] == "float32"
    assert (export_dir / exported["artifacts"][0]["path"]).is_file()


def test_smoke_and_tiny_overfit_commands_execute_their_distinct_modes(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli_module, "require_clean_head", lambda _root: "a" * 40)
    labels, positions, manifest = _dataset_fixture(tmp_path)
    dataset_arguments = [
        "--labels",
        str(labels),
        "--label-manifest",
        str(tmp_path / "label-manifest.json"),
        "--positions",
        str(positions),
        "--dataset-manifest",
        str(manifest),
    ]
    monkeypatch.setattr(
        cli_module,
        "load_training_examples",
        _load_test_examples,
    )

    smoke_dir = tmp_path / "smoke"
    assert main(["smoke", *dataset_arguments, "--output-dir", str(smoke_dir)]) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["command"] == "smoke"
    assert smoke["completedEpochs"] == 1

    overfit_dir = tmp_path / "overfit"
    assert (
        main(
            [
                "overfit",
                *dataset_arguments,
                "--output-dir",
                str(overfit_dir),
                "--examples",
                "3",
            ]
        )
        == 0
    )
    overfit = json.loads(capsys.readouterr().out)
    assert overfit["command"] == "overfit"
    assert overfit["completedEpochs"] == 200
    log = [
        json.loads(line)
        for line in (overfit_dir / "training-log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(log) == 200
    assert log[-1]["validation"]["total_loss"] < log[0]["validation"]["total_loss"]
    assert tuple((overfit_dir / ".open-shogi-retired").iterdir()) == ()
