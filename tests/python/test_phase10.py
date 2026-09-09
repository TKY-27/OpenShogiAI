from __future__ import annotations

import copy
from pathlib import Path

import pytest
from open_shogi_training.phase10 import (
    Phase10ValidationError,
    _load_yaml,
    _parse_hash_lines,
    _validate_audit_catalog,
    _validate_dataset_mixture,
    validate_phase10,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_phase10_controls_and_hashes_validate() -> None:
    with pytest.raises(ValueError, match="Closed campaign"):
        validate_phase10(ROOT)


def test_pending_source_cannot_be_accepted() -> None:
    mixture = copy.deepcopy(_load_yaml(ROOT / "configs/phase10/dataset-mixture.yaml"))
    audit_status = _validate_audit_catalog(_load_yaml(ROOT / "configs/data_sources_external.yaml"))
    pending = next(
        item for item in mixture["sources"] if item["artifact_id"] == "dlshogi-gct-hcpe3-selfplay"
    )
    pending["curriculum_decision"] = "accepted"

    with pytest.raises(Phase10ValidationError, match="decision must remain deferred"):
        _validate_dataset_mixture(mixture, audit_status)


def test_deferred_source_cannot_gain_sampling_weight() -> None:
    mixture = copy.deepcopy(_load_yaml(ROOT / "configs/phase10/dataset-mixture.yaml"))
    audit_status = _validate_audit_catalog(_load_yaml(ROOT / "configs/data_sources_external.yaml"))
    pending = next(item for item in mixture["sources"] if item["artifact_id"] == "qhapaq-qpd-train")
    pending["stage_weights"]["source_value"] = 0.01

    with pytest.raises(Phase10ValidationError, match="nonzero contribution"):
        _validate_dataset_mixture(mixture, audit_status)


def test_hash_manifest_rejects_duplicate_paths() -> None:
    digest = "a" * 64
    text = f"{digest}  file.yaml\n{digest}  file.yaml\n"

    with pytest.raises(Phase10ValidationError, match="duplicate frozen hash path"):
        _parse_hash_lines(text)
