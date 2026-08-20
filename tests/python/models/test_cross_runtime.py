import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from open_shogi_training.models.checkpoint import atomic_save_checkpoint, build_checkpoint
from open_shogi_training.models.config import (
    combined_config_sha256,
    load_feature_config,
    load_model_config,
    load_training_config,
)
from open_shogi_training.models.export import (
    infer_centipawns,
    infer_normalized,
    load_value_model,
)
from open_shogi_training.models.features import extract_features, input_dimension
from open_shogi_training.models.network import ValueModel
from open_shogi_training.models.train import model_code_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFENS = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
    "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3",
    "4k4/9/9/9/4+R4/9/9/9/4K4 w Pp 1",
)


def test_python_export_features_and_inference_match_real_rust_cli(tmp_path: Path) -> None:
    feature = replace(
        load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml"),
        name="cross_runtime_all_features",
        pseudo_attacks=True,
    )
    model_config = replace(
        load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml"),
        name="cross_runtime_tiny",
        hidden_layers=1,
        hidden_dim=5,
        dropout=0.0,
    )
    training = replace(
        load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml"),
        name="cross_runtime_export",
        quantization="both",
    )
    model = ValueModel(input_dimension(feature), model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    _take_optimizer_step(model, optimizer)
    _set_deterministic_parameters(model)
    model.eval()
    dataset_identity = {
        "dataset_manifest_sha256": "a" * 64,
        "positions_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
        "label_manifest_sha256": "d" * 64,
        "replay_manifest_sha256": None,
    }
    checkpoint = build_checkpoint(
        completed_epoch=1,
        global_step=1,
        best_validation_loss=0.5,
        model=model,
        optimizer=optimizer,
        feature_config=feature.as_dict(),
        model_config=model_config.as_dict(),
        training_config=training.as_dict(),
        config_sha256=combined_config_sha256(feature, model_config, training),
        dataset_identity=dataset_identity,
        device=torch.device("cpu"),
        runtime=_checkpoint_runtime(),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    atomic_save_checkpoint(checkpoint_path, checkpoint)
    export_dir = tmp_path / "export"
    export_result = _run_python_export(checkpoint_path, export_dir)
    metadata = json.loads((export_dir / "value_v0.metadata.json").read_text(encoding="utf-8"))
    assert export_result["schema"] == "phase4_model_command/v1"
    assert (
        metadata["provenance"]["checkpointSha256"]
        == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    )
    assert metadata["provenance"]["configSha256"] == combined_config_sha256(
        feature, model_config, training
    )
    assert metadata["provenance"]["exporterModelCodeSha256"] == model_code_sha256()
    input_path = tmp_path / "positions.sfen"
    input_path.write_text("".join(f"{sfen}\n" for sfen in SFENS), encoding="utf-8")

    command = _built_rust_cli_command()
    for suffix, quantization, torch_tolerance in (
        ("f32", "float32", 3e-5),
        ("int8", "int8", 6e-2),
    ):
        artifact = export_dir / f"value_v0.{suffix}.osaval"
        exported = load_value_model(artifact)
        inspection = _run_rust_json(
            [*command, "model", "inspect", "--model", str(artifact)],
        )[0]
        assert inspection["schema"] == "phase5_model_inspection/v1"
        assert inspection["quantization"] == quantization
        assert inspection["featureFlags"] == 0b11111
        assert inspection["artifactSha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert inspection["artifactSize"] == artifact.stat().st_size

        rust_rows = _run_rust_json(
            [
                *command,
                "model",
                "infer",
                "--model",
                str(artifact),
                "--input",
                str(input_path),
            ],
        )
        assert len(rust_rows) == len(SFENS)
        for index, (sfen, rust_row) in enumerate(zip(SFENS, rust_rows, strict=True)):
            features = extract_features(sfen, feature)
            python_normalized = infer_normalized(exported, features)
            with torch.no_grad():
                torch_normalized = float(
                    model(torch.tensor([features], dtype=torch.float32))[0].item()
                )
            assert python_normalized == pytest.approx(torch_normalized, abs=torch_tolerance)
            assert rust_row["schema"] == "phase5_model_inference/v1"
            assert rust_row["index"] == index
            assert rust_row["sfen"] == sfen
            # Both runtimes use f64 accumulation, f32 layer boundaries, half-away
            # rounding, and the same non-mate clamp, so integer CP must be exact.
            assert rust_row["scoreCp"] == infer_centipawns(exported, features)

    neural_output = tmp_path / "neural.jsonl"
    handcrafted_output = tmp_path / "handcrafted.jsonl"
    _run_rust(
        [
            *command,
            "model",
            "infer",
            "--model",
            str(export_dir / "value_v0.f32.osaval"),
            "--input",
            str(input_path),
            "--output",
            str(neural_output),
        ]
    )
    _run_rust(
        [
            *command,
            "model",
            "infer-handcrafted",
            "--profile",
            "handcrafted-baseline",
            "--input",
            str(input_path),
            "--output",
            str(handcrafted_output),
        ]
    )
    comparison = _run_python_model_command(
        [
            "compare-predictions",
            "--left",
            str(neural_output),
            "--right",
            str(handcrafted_output),
        ]
    )
    assert comparison["rows"] == len(SFENS)
    assert comparison["left"]["schema"] == "phase5_model_inference/v1"
    assert comparison["right"]["schema"] == "phase5_handcrafted_inference/v1"
    assert comparison["right"]["identity"] == {"evaluatorProfile": "handcrafted-baseline"}


def _set_deterministic_parameters(model: ValueModel) -> None:
    with torch.no_grad():
        width = model.input_dim
        indices = torch.arange(width, dtype=torch.float32)
        first = model.trunk[0]
        for row in range(first.out_features):
            pattern = ((indices * (2 * row + 1) + 11 * row).remainder(31) - 15) * 0.002
            first.weight[row].copy_(pattern)
        first.bias.copy_(torch.tensor([5.0, 5.1, 5.2, 5.3, 5.4]))
        model.value_head.weight.copy_(torch.tensor([[0.11, -0.17, 0.23, -0.29, 0.31]]))
        model.value_head.bias.fill_(0.037)
        model.policy_agreement_head.weight.zero_()
        model.policy_agreement_head.bias.zero_()


def _take_optimizer_step(model: ValueModel, optimizer: torch.optim.Optimizer) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _built_rust_cli_command() -> list[str]:
    subprocess.run(
        ["cargo", "build", "--locked", "-p", "open-shogi-cli"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    binary = PROJECT_ROOT / "target/debug/open-shogi-cli"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("cargo build did not publish target/debug/open-shogi-cli")
    return [str(binary)]


def _run_python_export(checkpoint: Path, output_dir: Path) -> dict[str, object]:
    return _run_python_model_command(
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ]
    )


def _run_python_model_command(arguments: list[str]) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "training")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_shogi_training.models",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return json.loads(completed.stdout)


def _checkpoint_runtime() -> dict[str, str]:
    return {
        "python": "3.12.0",
        "torch": str(torch.__version__),
        "platform": "cross-runtime-test",
        "device": "cpu",
        "deviceRequested": "cpu",
        "deviceFallback": "none",
        "mpsBuilt": "false",
        "mpsAvailable": "false",
        "deterministicAlgorithms": "true",
        "gitCommit": "0" * 40,
        "gitDirty": "false",
        "modelCodeSha256": "f" * 64,
        "torchNumThreads": "1",
        "torchNumInteropThreads": "1",
    }


def _run_rust_json(command: list[str]) -> list[dict[str, object]]:
    completed = _run_rust(command)
    return [json.loads(line) for line in completed.stdout.splitlines()]


def _run_rust(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return completed
