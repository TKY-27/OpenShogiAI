from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.phase10r import (
    Phase10RValidationError,
    allocate_source_counts,
    assert_disjoint_splits,
    decode_move,
    encode_move,
    memory_estimates,
    run_micro_overfit,
    run_pipeline_sanity,
    validate_phase10r,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_phase10r_controls_and_hashes_validate() -> None:
    result = validate_phase10r(ROOT)
    assert result["status"] == "valid"
    assert result["configs"] == 10
    assert result["approved_external_artifacts"] == 33
    assert result["public_weight_sources"] == 0


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


def test_source_weights_are_applied_exactly() -> None:
    assert allocate_source_counts(10, {"aobazero": 1.0, "wcsc": 1.0, "denryu": 0.5}) == {
        "aobazero": 4,
        "wcsc": 4,
        "denryu": 2,
    }


def test_tiny_dataset_can_be_intentionally_overfit() -> None:
    result = run_micro_overfit()
    assert result["final_loss"] < 0.001


def test_model_memory_estimates_fit_frozen_host_and_browser() -> None:
    result = memory_estimates(ROOT)
    assert all(row["artifact_fits_browser_limit"] for row in result["variants"])
    assert all(row["training_fits_rss_target"] for row in result["variants"])
