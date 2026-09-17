"""Supplementary controls never inherit the candidate's pure proof or a cap draw."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "opponent", Path(__file__).resolve().parents[2] / "scripts/compare_evaluator_opponent.py"
)
opponent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(opponent)


def test_opponent_proof_and_native_endings():
    state = {"sfen": "root", "successors": [{"move": "7g7f"}]}
    response = {
        "final_sfen": "root",
        "termination": "TimeLimit",
        "score": 0,
        "best_move": "7g7f",
        "deadline": {"hard_compliant": True},
        "proof": dict.fromkeys((*opponent.PROHIBITED, "learned_eval_calls"), 0),
    }
    response["proof"]["handcrafted_eval_calls"] = 1
    opponent.validate_response(response, state, handcrafted=True)
    with pytest.raises(ValueError):
        opponent.validate_response(response, state, handcrafted=False)
    response["best_move"] = "illegal"
    with pytest.raises(ValueError):
        opponent.validate_response(response, state, handcrafted=True)
    assert opponent.ending("None") == ("ongoing", None)
    assert opponent.ending("Some(Repetition(NoContest))") == ("repetition", None)
    assert opponent.ending("Some(Checkmate { winner: Black })") == ("checkmate", "black")
