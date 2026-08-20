from dataclasses import replace
from pathlib import Path

import pytest
import torch
from open_shogi_training.models.config import load_model_config
from open_shogi_training.models.network import ValueModel, parameter_count, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_value_model_has_two_scalar_heads_but_exports_only_value_path() -> None:
    config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    model = ValueModel(7, config)
    values, agreements = model(torch.zeros((3, 7), dtype=torch.float32))

    assert values.shape == (3,)
    assert agreements.shape == (3,)
    assert len(model.export_layers()) == config.hidden_layers + 1
    counts = parameter_count(7, config)
    assert counts["trainingTotal"] == sum(parameter.numel() for parameter in model.parameters())
    assert counts["exportedTotal"] == sum(
        parameter.numel() for layer in model.export_layers() for parameter in layer.parameters()
    )

    with pytest.raises(ValueError, match="version 1 supports only relu"):
        ValueModel(7, replace(config, activation="tanh"))


def test_device_selection_has_explicit_cpu_and_observable_mps_fallback(monkeypatch) -> None:
    cpu = select_device("cpu")
    assert cpu.device == torch.device("cpu")
    assert cpu.fallback_reason is None

    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    fallback = select_device("mps")
    assert fallback.device == torch.device("cpu")
    assert fallback.requested == "mps"
    assert fallback.fallback_reason is not None

    with pytest.raises(ValueError, match="auto, cpu, or mps"):
        select_device("cuda")
