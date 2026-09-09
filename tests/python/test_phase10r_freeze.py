from __future__ import annotations

import copy
from pathlib import Path

import pytest
from open_shogi_training.phase10r import (
    Phase10RValidationError,
    _load_yaml,
    allocate_source_counts,
    assert_disjoint_splits,
    canonical_pretraining_mixture,
    decode_move,
    encode_move,
    memory_estimates,
    run_micro_overfit,
    run_pipeline_sanity,
    validate_phase10r,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_phase10r_controls_with_explicit_phase10t_runtime_successors() -> None:
    with pytest.raises(ValueError, match="Closed campaign"):
        validate_phase10r(ROOT, phase10t_runtime_successors=True)


def test_pipeline_sanity_gate_passes() -> None:
    result = run_pipeline_sanity(ROOT)
    assert result["status"] == "passed"
    assert all(result["checks"].values())


@pytest.mark.parametrize("move", ["2g2f", "2b2a+", "P*5e", "R*1a"])
def test_move_codec_round_trips_promotion_and_drops(move: str) -> None:
    assert decode_move(encode_move(move)) == move


def test_cross_split_identity_is_rejected() -> None:
    with pytest.raises(Phase10RValidationError, match="entered another split"):
        assert_disjoint_splits(
            [{"identity": "same", "split": "train"}, {"identity": "same", "split": "validation"}]
        )


def test_canonical_target_shares_are_applied_exactly() -> None:
    shares = {"aobazero": 0.35, "wcsc": 0.45, "denryu": 0.20}
    assert allocate_source_counts(100, shares, tie_break_order=["aobazero", "wcsc", "denryu"]) == {
        "aobazero": 35,
        "wcsc": 45,
        "denryu": 20,
    }


def test_source_allocation_rejects_unnormalized_legacy_weights() -> None:
    with pytest.raises(Phase10RValidationError, match="sum exactly"):
        allocate_source_counts(10, {"aobazero": 1.0, "wcsc": 1.0, "denryu": 0.5})


def test_source_allocation_uses_frozen_rounding_tie_break() -> None:
    assert allocate_source_counts(
        5,
        {"aobazero": 0.4, "wcsc": 0.3, "denryu": 0.3},
        tie_break_order=["aobazero", "wcsc", "denryu"],
    ) == {"aobazero": 2, "wcsc": 2, "denryu": 1}


def test_canonical_mixture_rejects_target_above_frozen_maximum() -> None:
    mixture = copy.deepcopy(_load_yaml(ROOT / "configs/phase10r/dataset-mixture.yaml"))
    mixture["canonical_pretraining_mixture"]["sources"][0]["target_share"] = 0.36
    mixture["canonical_pretraining_mixture"]["sources"][1]["target_share"] = 0.44

    with pytest.raises(Phase10RValidationError, match="violates its share bounds"):
        canonical_pretraining_mixture(mixture)


def test_canonical_mixture_requires_exact_normalization() -> None:
    mixture = copy.deepcopy(_load_yaml(ROOT / "configs/phase10r/dataset-mixture.yaml"))
    mixture["canonical_pretraining_mixture"]["sources"][1]["target_share"] = 0.44

    with pytest.raises(Phase10RValidationError, match="sum exactly"):
        canonical_pretraining_mixture(mixture)


def test_canonical_mixture_rejects_a_different_normalized_vector() -> None:
    mixture = copy.deepcopy(_load_yaml(ROOT / "configs/phase10r/dataset-mixture.yaml"))
    mixture["canonical_pretraining_mixture"]["sources"][0]["target_share"] = 0.34
    mixture["canonical_pretraining_mixture"]["sources"][1]["target_share"] = 0.46

    with pytest.raises(Phase10RValidationError, match="target or maximum vector changed"):
        canonical_pretraining_mixture(mixture)


def test_tiny_dataset_can_be_intentionally_overfit() -> None:
    result = run_micro_overfit()
    assert result["final_loss"] < 0.001


def test_model_memory_estimates_fit_frozen_host_and_browser() -> None:
    result = memory_estimates(ROOT)
    assert all(row["artifact_fits_browser_limit"] for row in result["variants"])
    assert all(row["training_fits_rss_target"] for row in result["variants"])
