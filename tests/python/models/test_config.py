from pathlib import Path

import pytest
from open_shogi_training.models.config import (
    combined_config_sha256,
    config_sha256,
    configured_input_dimension,
    estimated_export_bytes,
    exported_operation_count,
    load_feature_config,
    load_model_config,
    load_training_config,
    parse_model_config,
    parse_training_config,
    validate_config_compatibility,
)
from open_shogi_training.models.features import input_dimension

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_repository_value_v0_configs_are_closed_and_compatible() -> None:
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml")

    validate_config_compatibility(model, training, feature)
    assert feature.board_planes
    assert not feature.pseudo_attacks
    assert model.hidden_layers == 2
    assert model.hidden_dim == 64
    assert model.output_scale_cp == 1200.0
    assert training.quantization == "both"
    assert training.stage_boundaries_basis_points == (3333, 6667)
    assert training.expected_teacher_labels == 10_000
    assert "stage_boundaries" not in training.as_dict()["training"]
    assert configured_input_dimension(feature) == input_dimension(feature)
    assert exported_operation_count(feature, model) > 0
    assert estimated_export_bytes(feature, model, "float32") > estimated_export_bytes(
        feature, model, "int8"
    )
    assert len(config_sha256(feature)) == 64
    assert len(combined_config_sha256(feature, model, training)) == 64


def test_smoke_config_is_conservative() -> None:
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")

    assert training.epochs == 1
    assert training.batch_size <= 8
    assert training.sample_ratio <= 0.02
    assert training.quantization == "float32"


def test_config_loader_rejects_symlinks_and_oversized_files(tmp_path: Path) -> None:
    target = PROJECT_ROOT / "configs/training/value_v0_smoke.toml"
    symlink = tmp_path / "training.toml"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_training_config(symlink)

    oversized = tmp_path / "oversized.toml"
    oversized.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match=r"size|1 MiB"):
        load_training_config(oversized)


def test_training_config_rejects_unknown_missing_and_invalid_values() -> None:
    valid = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml").as_dict()

    with_unknown = {**valid, "training": {**valid["training"], "mystery": 1}}
    with pytest.raises(ValueError, match=r"unknown=\['mystery'\]"):
        parse_training_config(with_unknown)

    with_missing = {**valid, "training": dict(valid["training"])}
    del with_missing["training"]["learning_rate"]
    with pytest.raises(ValueError, match=r"missing=\['learning_rate'\]"):
        parse_training_config(with_missing)

    bad_ratios = {**valid, "training": {**valid["training"], "stage_ratios": [0.5, 0.5, 0.5]}}
    with pytest.raises(ValueError, match="sum to 1"):
        parse_training_config(bad_ratios)

    bad_weight = {
        **valid,
        "training": {**valid["training"], "teacher_loss_weight": -1.0},
    }
    with pytest.raises(ValueError, match="non-negative"):
        parse_training_config(bad_weight)

    ambiguous_legacy = {**valid, "training": dict(valid["training"])}
    ambiguous_legacy["training"]["stage_boundaries"] = [40, 100]
    with pytest.raises(ValueError, match=r"unknown=\['stage_boundaries'\]"):
        parse_training_config(ambiguous_legacy)

    for boundaries in ([0, 6667], [3333, 3333], [3333, 10_000]):
        invalid = {
            **valid,
            "training": {**valid["training"], "stage_boundaries_basis_points": boundaries},
        }
        with pytest.raises(ValueError, match="0 < opening < middlegame < 10000"):
            parse_training_config(invalid)

    wrong_count = {
        **valid,
        "training": {**valid["training"], "expected_teacher_labels": 9_999},
    }
    with pytest.raises(ValueError, match="goal-wide 10000"):
        parse_training_config(wrong_count)


def test_compatibility_rejects_changed_score_meaning() -> None:
    model = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training_raw = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml").as_dict()
    training_raw["training"]["teacher_normalization_cp"] = 600.0
    training = parse_training_config(training_raw)

    with pytest.raises(ValueError, match="must equal"):
        validate_config_compatibility(model, training)


@pytest.mark.parametrize("output_scale_cp", [1e-100, 1e100])
def test_model_output_scale_must_roundtrip_as_positive_float32(output_scale_cp: float) -> None:
    raw = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml").as_dict()
    raw["model"]["output_scale_cp"] = output_scale_cp

    with pytest.raises(ValueError, match="positive float32"):
        parse_model_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("hidden_layers", 17, "between 1 and 16"),
        ("hidden_dim", 8193, "between 1 and 8192"),
    ],
)
def test_model_config_enforces_shared_rust_architecture_caps(
    field: str, value: int, match: str
) -> None:
    raw = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml").as_dict()
    raw["model"][field] = value

    with pytest.raises(ValueError, match=match):
        parse_model_config(raw)


def test_architecture_v1_rejects_tanh_without_cross_runtime_parity() -> None:
    raw = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml").as_dict()
    raw["model"]["activation"] = "tanh"

    with pytest.raises(ValueError, match="activation must be one of"):
        parse_model_config(raw)


def test_compatibility_enforces_16_million_exported_parameter_cap() -> None:
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model_raw = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml").as_dict()
    model_raw["model"]["hidden_dim"] = 8192
    model = parse_model_config(model_raw)
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml")

    with pytest.raises(ValueError, match="16000000"):
        validate_config_compatibility(model, training, feature)
