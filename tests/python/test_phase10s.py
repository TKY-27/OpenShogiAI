from __future__ import annotations

import copy
from pathlib import Path

import pytest
from open_shogi_training.phase10s import (
    Phase10SValidationError,
    _load_yaml,
    _validate_model_matrix,
    _validate_redesign,
    validate_phase10s,
)

ROOT = Path(__file__).resolve().parents[2]


def test_phase10s_static_freeze_validates_without_ignored_local_evidence() -> None:
    with pytest.raises(ValueError, match="Closed campaign"):
        validate_phase10s(ROOT, verify_local_evidence=False)


def test_phase10s_repetition_cap_is_fail_closed() -> None:
    control = _load_yaml(ROOT / "configs/phase10s/supervised-redesign.yaml")
    changed = copy.deepcopy(control)
    changed["sources"]["approved"][0]["maximum_exposures_per_supervised_generation"] = 3
    with pytest.raises(Phase10SValidationError, match="repetition cap changed"):
        _validate_redesign(changed)


def test_phase10s_training_batch_loss_cannot_select_a_checkpoint() -> None:
    control = _load_yaml(ROOT / "configs/phase10s/supervised-redesign.yaml")
    changed = copy.deepcopy(control)
    changed["training"]["checkpointing"]["training_batch_loss_may_select"] = True
    with pytest.raises(Phase10SValidationError, match="optimization contract changed"):
        _validate_redesign(changed)


def test_phase10s_model_matrix_rejects_a_third_variant() -> None:
    matrix = _load_yaml(ROOT / "configs/phase10s/model-matrix.yaml")
    changed = copy.deepcopy(matrix)
    changed["variants"].append({"id": "unattributed-third-variant"})
    with pytest.raises(Phase10SValidationError, match="model variants changed"):
        _validate_model_matrix(changed)
