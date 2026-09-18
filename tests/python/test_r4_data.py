"""Public root-value direction and immutable snapshot resume."""

import math

import pytest
from open_shogi_training.evaluator_data import atomic, digest, encoded
from open_shogi_training.r4_data import import_dataset, probability_cp, verify_dataset


def test_public_value_scale_and_snapshot_resume(tmp_path):
    assert probability_cp(0.5) == 0
    assert probability_cp(0.75) == -probability_cp(0.25) == int(600 * math.log(3))
    assert probability_cp(1) == -probability_cp(0) == 5000
    for invalid in (float("nan"), -1, 2):
        with pytest.raises(ValueError):
            probability_cp(invalid)
    source = tmp_path / "source"
    atomic(source / "train.npy", b"data")
    manifest = {
        "schema": "open_shogiai_r4_data/v1",
        "inputs": [],
        "artifacts": [{"path": "train.npy", "sha256": digest(source / "train.npy")}],
    }
    atomic(source / "manifest.json", encoded(manifest))
    ref = {
        "prepared_dataset": {"path": "source", "manifest_sha256": digest(source / "manifest.json")}
    }
    output = tmp_path / "run/data"
    atomic(output / "dataset-building/train.npy", b"interrupted")
    assert import_dataset(tmp_path, output, ref) == manifest
    assert import_dataset(tmp_path, output, ref) == manifest
    (output / "dataset/train.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(tmp_path, output / "dataset")
