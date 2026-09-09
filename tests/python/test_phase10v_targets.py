"""Observed-only teacher target and original split provenance boundaries."""

from copy import deepcopy

import pytest
from open_shogi_training.phase10v_targets import (
    TARGET_SCHEMA,
    CandidateTarget,
    Phase10VExample,
    TeacherScore,
    example_from_mapping,
    observed_targets,
    root_score_from_child,
)

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
CHILD = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2"


def approved_row():
    return {
        "schema": TARGET_SCHEMA,
        "split": "train",
        "sfen": START,
        "score": {"kind": "cp", "value": 200},
        "score_perspective": "side_to_move",
        "provenance": {
            "approved": True,
            "original_split": "train",
            "source": "apery",
            "source_game_id": "game-1",
            "leaf_kind": "quiescence_leaf",
            "search_receipt_sha256": "a" * 64,
            "root_position_sha256": "b" * 64,
            "engine_model_sha256": "c" * 64,
        },
    }


def test_typed_mate_order_and_scalar_namespace():
    scores = [
        TeacherScore("mate", -3),
        TeacherScore("mate", -7),
        TeacherScore("cp", -20000),
        TeacherScore("cp", 20000),
        TeacherScore("mate", 7),
        TeacherScore("mate", 3),
    ]
    assert scores == sorted(scores, key=lambda score: score.order)
    for score in scores:
        assert score.reversed().value == -score.value
    for kind, value in [("cp", 29000), ("mate", 0), ("cp", True), ("cp", 0.5), ("bad", 1)]:
        with pytest.raises(ValueError):
            TeacherScore(kind, value)


def test_partial_multipv_and_mate_masks():
    candidate = CandidateTarget("7g7f", CHILD, TeacherScore("mate", 3))
    row = Phase10VExample(START, TeacherScore("mate", 3), candidates=(candidate,))
    targets = observed_targets(row)
    assert targets["candidate_mask"] == [True, False, False]
    assert targets["value_cp"] is None and not targets["value_mask"]
    assert targets["ranking_pairs"] == [] and not targets["policy_mask"]
    assert root_score_from_child(-200) == 200


def test_leaf_provenance_and_train_only_boundary():
    row = approved_row()
    example = example_from_mapping(row, expected_split="train")
    assert example.cp == 200
    assert len(example.provenance_sha256) == 64
    for mutation in ("leaf_receipt", "perspective", "original_split", "holdout", "pseudo_wdl"):
        changed = deepcopy(row)
        if mutation == "leaf_receipt":
            del changed["provenance"]["search_receipt_sha256"]
        elif mutation == "perspective":
            changed["score_perspective"] = "black"
        elif mutation == "original_split":
            changed["provenance"]["original_split"] = "final_holdout"
        elif mutation == "holdout":
            changed["split"] = "final_holdout"
        else:
            changed["wdl"] = 2
            changed["wdl_source"] = "cp_logistic"
        with pytest.raises(ValueError):
            example_from_mapping(changed, expected_split="train")


def test_successor_parent_score_and_ply_are_explicit():
    row = approved_row()
    candidate = {
        "move": "7g7f",
        "child_sfen": CHILD,
        "score": {"kind": "cp", "value": 20},
        "score_perspective": "parent_side_to_move",
        "replay_receipt_sha256": "d" * 64,
    }
    row["candidates"] = [candidate]
    assert len(example_from_mapping(row, expected_split="train").candidates) == 1
    candidate["child_sfen"] = START
    with pytest.raises(ValueError, match="one ply"):
        example_from_mapping(row, expected_split="train")


def test_fabricated_child_with_correct_turn_and_ply_is_rejected():
    row = approved_row()
    row["candidates"] = [
        {
            "move": "2g2f",
            "child_sfen": CHILD,
            "score": {"kind": "cp", "value": 20},
            "score_perspective": "parent_side_to_move",
            "replay_receipt_sha256": "d" * 64,
        }
    ]
    with pytest.raises(ValueError, match="replayed move"):
        example_from_mapping(row, expected_split="train")


@pytest.mark.parametrize("raw", [20001, 28999, -20001, -28999])
def test_extreme_observed_cp_preserved_with_explicit_scalar_saturation(raw):
    row = Phase10VExample(START, TeacherScore("cp", raw))
    targets = observed_targets(row)
    assert row.score.value == raw
    assert row.cp == (20000 if raw > 0 else -20000)
    assert targets["observed_value_cp"] == raw
    assert targets["scalar_target_saturated"]
    assert targets["mate_metadata"] is None
