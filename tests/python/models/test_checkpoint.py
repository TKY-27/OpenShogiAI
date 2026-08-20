import copy
from pathlib import Path

import pytest
import torch
from open_shogi_training.models import checkpoint as checkpoint_module
from open_shogi_training.models.checkpoint import (
    atomic_save_checkpoint,
    build_checkpoint,
    load_checkpoint,
    validate_resume_identity,
)
from open_shogi_training.models.config import (
    combined_config_sha256,
    load_feature_config,
    load_model_config,
    load_training_config,
)
from open_shogi_training.models.features import input_dimension
from open_shogi_training.models.network import ValueModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_checkpoint_roundtrip_is_weights_safe_atomic_and_identity_bound(tmp_path: Path) -> None:
    config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    model = ValueModel(input_dimension(feature), config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    _take_optimizer_step(model, optimizer)
    identity = {
        "dataset_manifest_sha256": "a" * 64,
        "positions_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
        "label_manifest_sha256": "d" * 64,
        "replay_manifest_sha256": None,
    }
    payload = build_checkpoint(
        completed_epoch=1,
        global_step=1,
        best_validation_loss=0.5,
        model=model,
        optimizer=optimizer,
        feature_config=feature.as_dict(),
        model_config=config.as_dict(),
        training_config=training.as_dict(),
        config_sha256=combined_config_sha256(feature, config, training),
        dataset_identity=identity,
        device=torch.device("cpu"),
        runtime=_runtime(),
    )
    path = tmp_path / "last.pt"

    atomic_save_checkpoint(path, payload)
    loaded = load_checkpoint(path)

    assert loaded["completed_epoch"] == 1
    symlink = tmp_path / "checkpoint-link.pt"
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_checkpoint(symlink)
    validate_resume_identity(
        loaded,
        config_sha256=combined_config_sha256(feature, config, training),
        dataset_identity=identity,
    )
    with pytest.raises(ValueError, match="dataset identities"):
        validate_resume_identity(
            loaded,
            config_sha256=combined_config_sha256(feature, config, training),
            dataset_identity={**identity, "labels_sha256": "e" * 64},
        )

    corrupt = dict(payload)
    corrupt_state = {
        name: tensor.detach().clone() for name, tensor in payload["model_state"].items()
    }
    first_name = next(iter(corrupt_state))
    corrupt_state[first_name].reshape(-1)[0] = float("nan")
    corrupt["model_state"] = corrupt_state
    corrupt_path = tmp_path / "corrupt.pt"
    atomic_save_checkpoint(corrupt_path, corrupt)
    with pytest.raises(ValueError, match="non-finite tensor"):
        load_checkpoint(corrupt_path)

    bad_identity = dict(payload)
    bad_identity["dataset_identity"] = {**identity, "labels_sha256": "not-a-hash"}
    bad_identity_path = tmp_path / "bad-identity.pt"
    atomic_save_checkpoint(bad_identity_path, bad_identity)
    with pytest.raises(ValueError, match="labels_sha256"):
        load_checkpoint(bad_identity_path)

    false_config_identity = dict(payload)
    false_config_identity["config_sha256"] = "d" * 64
    false_config_path = tmp_path / "false-config.pt"
    atomic_save_checkpoint(false_config_path, false_config_identity)
    with pytest.raises(ValueError, match="embedded configurations"):
        load_checkpoint(false_config_path)

    bad_shape = dict(payload)
    bad_shape["model_state"] = dict(payload["model_state"])
    first_name = next(iter(bad_shape["model_state"]))
    bad_shape["model_state"][first_name] = torch.zeros(1)
    bad_shape_path = tmp_path / "bad-shape.pt"
    atomic_save_checkpoint(bad_shape_path, bad_shape)
    with pytest.raises(ValueError, match="shape or dtype"):
        load_checkpoint(bad_shape_path)

    bad_optimizer = dict(payload)
    bad_optimizer["optimizer_state"] = {
        "state": dict(payload["optimizer_state"]["state"]),
        "param_groups": [dict(payload["optimizer_state"]["param_groups"][0])],
    }
    bad_optimizer["optimizer_state"]["param_groups"][0]["lr"] *= 2
    bad_optimizer_path = tmp_path / "bad-optimizer.pt"
    atomic_save_checkpoint(bad_optimizer_path, bad_optimizer)
    with pytest.raises(ValueError, match="AdamW options"):
        load_checkpoint(bad_optimizer_path)

    bad_step = dict(payload)
    bad_step["optimizer_state"] = copy.deepcopy(payload["optimizer_state"])
    first_state = next(iter(bad_step["optimizer_state"]["state"].values()))
    first_state["step"] += 1
    bad_step_path = tmp_path / "bad-optimizer-step.pt"
    atomic_save_checkpoint(bad_step_path, bad_step)
    with pytest.raises(ValueError, match="global_step"):
        load_checkpoint(bad_step_path)

    bad_runtime = dict(payload)
    bad_runtime["runtime"] = {**payload["runtime"], "deterministicAlgorithms": "false"}
    bad_runtime_path = tmp_path / "bad-runtime.pt"
    atomic_save_checkpoint(bad_runtime_path, bad_runtime)
    with pytest.raises(ValueError, match="disagrees with config"):
        load_checkpoint(bad_runtime_path)


def test_atomic_checkpoint_save_enforces_the_post_serialization_bound(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(checkpoint_module, "MAX_CHECKPOINT_BYTES", 1)
    path = tmp_path / "too-large.pt"

    with pytest.raises(ValueError, match="512 MiB"):
        atomic_save_checkpoint(path, {"value": torch.zeros(1)})

    assert not path.exists()
    retired = tuple((tmp_path / ".open-shogi-retired").iterdir())
    assert len(retired) == 1
    assert retired[0].is_file()


def test_atomic_checkpoint_rejects_entry_replacement_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    owned = tmp_path / "owned-checkpoint.pt"
    real_publish = checkpoint_module.publish_regular_at

    def publish_then_relink(*args, **kwargs) -> None:
        real_publish(*args, **kwargs)
        path.rename(owned)
        path.write_bytes(b"foreign-do-not-delete")

    monkeypatch.setattr(checkpoint_module, "publish_regular_at", publish_then_relink)
    with pytest.raises(ValueError, match="changed during publication"):
        atomic_save_checkpoint(path, {"value": torch.zeros(1)})

    assert owned.is_file()
    assert path.read_bytes() == b"foreign-do-not-delete"


def test_checkpoint_validation_does_not_advance_the_global_torch_rng(tmp_path: Path) -> None:
    config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    model = ValueModel(input_dimension(feature), config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    _take_optimizer_step(model, optimizer)
    path = tmp_path / "rng-stable.pt"
    atomic_save_checkpoint(
        path,
        build_checkpoint(
            completed_epoch=1,
            global_step=1,
            best_validation_loss=0.5,
            model=model,
            optimizer=optimizer,
            feature_config=feature.as_dict(),
            model_config=config.as_dict(),
            training_config=training.as_dict(),
            config_sha256=combined_config_sha256(feature, config, training),
            dataset_identity={
                "dataset_manifest_sha256": "a" * 64,
                "positions_sha256": "b" * 64,
                "labels_sha256": "c" * 64,
                "label_manifest_sha256": "d" * 64,
                "replay_manifest_sha256": None,
            },
            device=torch.device("cpu"),
            runtime=_runtime(),
        ),
    )
    torch.manual_seed(987_654_321)
    before = torch.get_rng_state().clone()

    load_checkpoint(path)

    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0.0, atol=0.0)


def _take_optimizer_step(model: ValueModel, optimizer: torch.optim.Optimizer) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _runtime() -> dict[str, str]:
    return {
        "python": "3.12.0",
        "torch": str(torch.__version__),
        "platform": "test-platform",
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
