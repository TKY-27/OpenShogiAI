import hashlib
import json
import struct
from pathlib import Path

import pytest
from open_shogi_training.models import export as export_module
from open_shogi_training.models.config import (
    estimated_export_bytes,
    load_feature_config,
    load_model_config,
    load_training_config,
)
from open_shogi_training.models.export import (
    infer_centipawns,
    infer_normalized,
    normalized_to_centipawns,
    parse_value_model,
    serialize_value_model,
)
from open_shogi_training.models.features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _tiny_model_bytes(*, quantization: int = 0) -> bytes:
    header = struct.pack(
        "<8s10If",
        b"OSAVAL01",
        1,
        1,
        1,
        1 << 2,
        1,
        1,
        1,
        0,
        quantization,
        2,
        1200.0,
    )
    if quantization == 0:
        first = struct.pack("<IIff", 1, 1, 2.0, 3.0)
        second = struct.pack("<IIff", 1, 1, 4.0, 5.0)
    else:
        first = struct.pack("<IIfbf", 1, 1, 0.5, 4, 3.0)
        second = struct.pack("<IIfbf", 1, 1, 1.0, 4, 5.0)
    payload = header + first + second
    return payload + hashlib.sha256(payload).digest()


@pytest.mark.parametrize("quantization", [0, 1])
def test_pure_python_parser_and_inference_follow_osaval01(quantization: int) -> None:
    model = parse_value_model(_tiny_model_bytes(quantization=quantization))

    assert model.input_dim == 1
    assert model.hidden_layers == 1
    assert model.quantization == quantization
    assert infer_normalized(model, [2.0]) == pytest.approx(33.0)
    assert infer_centipawns(model, [2.0]) == 28_999


def test_parser_rejects_corruption_even_when_shape_bytes_still_parse() -> None:
    data = bytearray(_tiny_model_bytes())
    data[60] ^= 0x01

    with pytest.raises(ValueError, match="SHA-256"):
        parse_value_model(bytes(data))


def test_parser_uses_shared_64_mib_artifact_cap(monkeypatch) -> None:
    assert export_module._MAX_MODEL_BYTES == 64 * 1024 * 1024
    monkeypatch.setattr(export_module, "_MAX_MODEL_BYTES", len(_tiny_model_bytes()) - 1)

    with pytest.raises(ValueError, match="64 MiB parser bound"):
        parse_value_model(_tiny_model_bytes())


def test_parser_rejects_inconsistent_dimensions_with_valid_digest() -> None:
    data = bytearray(_tiny_model_bytes())
    # First layer input dimension begins immediately after the 52-byte header.
    struct.pack_into("<I", data, 52, 2)
    payload = bytes(data[:-32])
    data[-32:] = hashlib.sha256(payload).digest()

    with pytest.raises(ValueError, match="inconsistent dimensions"):
        parse_value_model(bytes(data))


def test_parser_rejects_tanh_for_architecture_v1_even_with_valid_digest() -> None:
    data = bytearray(_tiny_model_bytes())
    # The activation code is the eighth u32 after the 8-byte magic.
    struct.pack_into("<I", data, 8 + 7 * 4, 1)
    payload = bytes(data[:-32])
    data[-32:] = hashlib.sha256(payload).digest()

    with pytest.raises(ValueError, match="version 1 supports only relu"):
        parse_value_model(bytes(data))


def test_parser_rejects_nonfinite_inputs() -> None:
    model = parse_value_model(_tiny_model_bytes())

    with pytest.raises(ValueError, match="non-finite"):
        infer_normalized(model, [float("nan")])


def test_centipawn_conversion_is_half_away_from_zero_and_clamped_below_mate() -> None:
    assert normalized_to_centipawns(0.5, 1.0) == 1
    assert normalized_to_centipawns(-0.5, 1.0) == -1
    assert normalized_to_centipawns(30_000.0, 1.0) == 28_999
    assert normalized_to_centipawns(-30_000.0, 1.0) == -28_999


def test_torch_roundtrip_float_int8_metadata_and_white_to_move_parity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from open_shogi_training.models.export import export_model_artifacts, load_value_model
    from open_shogi_training.models.features import input_dimension
    from open_shogi_training.models.network import ValueModel

    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model_config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml")
    torch.manual_seed(7)
    model = ValueModel(input_dimension(feature), model_config)
    model.eval()
    result = export_model_artifacts(
        model,
        feature,
        model_config,
        training,
        tmp_path,
        provenance={
            "checkpointSha256": "a" * 64,
            "datasetManifestSha256": "b" * 64,
            "labelJsonlSha256": "c" * 64,
        },
    )

    metadata = json.loads(Path(result["metadataPath"]).read_text(encoding="utf-8"))
    assert Path(result["featureSchemaPath"]).is_file()
    assert {artifact["quantization"] for artifact in result["artifacts"]} == {
        "float32",
        "int8",
    }
    assert metadata["model"]["outputPerspective"] == "current_side_to_move"
    assert not metadata["model"]["policyAgreementAuxiliaryHead"]["exported"]
    assert {artifact["quantization"] for artifact in metadata["artifacts"]} == {
        "float32",
        "int8",
    }
    for artifact in result["artifacts"]:
        assert artifact["size"] == estimated_export_bytes(
            feature, model_config, artifact["quantization"]
        )
    model_symlink = tmp_path / "model-link.osaval"
    model_symlink.symlink_to(tmp_path / "value_v0.f32.osaval")
    with pytest.raises(ValueError, match="non-symlink"):
        load_value_model(model_symlink)

    black_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    white_sfen = black_sfen.replace(" b - 1", " w - 1")
    for sfen in (black_sfen, white_sfen):
        features = extract_features(sfen, feature)
        tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            expected = float(model(tensor)[0].item())
        float_export = load_value_model(tmp_path / "value_v0.f32.osaval")
        int8_export = load_value_model(tmp_path / "value_v0.int8.osaval")
        assert infer_normalized(float_export, features) == pytest.approx(expected, abs=2e-5)
        assert infer_normalized(int8_export, features) == pytest.approx(expected, abs=0.05)

    with pytest.raises(ValueError, match="quantization must be"):
        serialize_value_model(
            model,
            feature,
            model_config,
            quantization="unknown",  # type: ignore[arg-type]
        )


def test_export_rejects_nonfinite_parameters_and_preflights_all_collisions(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from open_shogi_training.models.export import export_model_artifacts
    from open_shogi_training.models.features import input_dimension
    from open_shogi_training.models.network import ValueModel

    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model_config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml")
    model = ValueModel(input_dimension(feature), model_config)
    with torch.no_grad():
        model.value_head.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        serialize_value_model(model, feature, model_config, quantization="float32")

    model = ValueModel(input_dimension(feature), model_config)
    output_dir = tmp_path / "collision"
    output_dir.mkdir()
    collision = output_dir / "value_v0.metadata.json"
    collision.write_text("reserved\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_model_artifacts(
            model,
            feature,
            model_config,
            training,
            output_dir,
            provenance={"checkpointSha256": "a" * 64},
        )
    assert tuple(path.name for path in output_dir.iterdir()) == (collision.name,)

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    symlink = tmp_path / "export-link"
    symlink.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        export_model_artifacts(
            model,
            feature,
            model_config,
            training,
            symlink,
            provenance={"checkpointSha256": "a" * 64},
        )
    assert not tuple(real_dir.iterdir())


def test_export_pins_output_directory_across_ancestor_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from open_shogi_training.models.export import export_model_artifacts
    from open_shogi_training.models.features import input_dimension
    from open_shogi_training.models.network import ValueModel

    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model_config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml")
    model = ValueModel(input_dimension(feature), model_config)
    output = tmp_path / "parent" / "output"
    moved_parent = tmp_path / "moved-parent"
    replacement_output = output
    real_publish = export_module._publish_export_transaction

    def replace_ancestor_then_publish(**kwargs) -> None:
        output.parent.rename(moved_parent)
        replacement_output.mkdir(parents=True)
        real_publish(**kwargs)

    monkeypatch.setattr(
        export_module,
        "_publish_export_transaction",
        replace_ancestor_then_publish,
    )
    with pytest.raises(ValueError, match="output_dir must remain"):
        export_model_artifacts(
            model,
            feature,
            model_config,
            training,
            output,
            provenance={"checkpointSha256": "a" * 64},
        )

    assert not tuple(replacement_output.iterdir())
    assert (moved_parent / "output" / "value_v0.export-commit.json").is_file()
    assert not (moved_parent / "output" / "value_v0.export-transaction.json").exists()


@pytest.mark.parametrize(
    "failpoint",
    [
        "after_journal",
        "after_artifact_0",
        "after_artifact_1",
        "after_artifact_2",
        "after_artifact_3",
        "after_marker",
    ],
)
def test_export_transaction_recovers_every_publication_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failpoint: str
) -> None:
    pytest.importorskip("torch")
    from open_shogi_training.models.export import export_model_artifacts
    from open_shogi_training.models.features import input_dimension
    from open_shogi_training.models.network import ValueModel

    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    model_config = load_model_config(PROJECT_ROOT / "configs/models/value_v0.toml")
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0.toml")
    model = ValueModel(input_dimension(feature), model_config)
    output = tmp_path / failpoint
    provenance = {"checkpointSha256": "a" * 64, "labelManifestSha256": "b" * 64}
    monkeypatch.setenv("OPEN_SHOGI_EXPORT_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        export_model_artifacts(
            model, feature, model_config, training, output, provenance=provenance
        )
    monkeypatch.delenv("OPEN_SHOGI_EXPORT_FAILPOINT")

    first = export_model_artifacts(
        model, feature, model_config, training, output, provenance=provenance
    )
    marker = output / "value_v0.export-commit.json"
    marker_bytes = marker.read_bytes()
    artifact_bytes = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file() and "transaction" not in path.name
    }
    second = export_model_artifacts(
        model, feature, model_config, training, output, provenance=provenance
    )

    assert first == second
    assert marker.read_bytes() == marker_bytes
    assert artifact_bytes == {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file() and "transaction" not in path.name
    }
    assert not (output / "value_v0.export-transaction.json").exists()
