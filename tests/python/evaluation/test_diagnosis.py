from __future__ import annotations

import pytest
from open_shogi_training.evaluation.config import DiagnosisConfig
from open_shogi_training.evaluation.diagnosis import DiagnosisError, diagnose_decision

CONFIG = DiagnosisConfig(
    move_regret_cp=150,
    evaluation_disagreement_cp=200,
    hard_example_cp=150,
    max_hard_examples=128,
)


def _event(
    *,
    actor: str = "ai",
    move: str = "2g2f",
    score: int | None = 90,
    opening: bool = False,
) -> dict[str, object]:
    return {
        "actor": actor,
        "moveUsi": move,
        "scoreCp": score,
        "openingBook": opening,
    }


def _search(*candidates: tuple[str, int]) -> dict[str, object]:
    return {
        "bestmove": candidates[0][0],
        "candidates": [
            {
                "score": {"kind": "cp", "value": score},
                "pv": [move],
            }
            for move, score in candidates
        ],
    }


@pytest.mark.parametrize(
    ("event", "expected_kind", "hard"),
    [
        (_event(move="7g7f", score=100), "aligned", False),
        (_event(move="7g7f", score=None), "aligned_move_unscored", False),
        (_event(move="7g7f", score=-150), "evaluation_failure_candidate", True),
        (_event(move="2g2f", score=90), "search_failure_candidate", True),
        (_event(move="2g2f", score=-200), "mixed_failure_candidate", True),
        (_event(move="3g3f", score=None), "uncertain_failure_candidate", True),
        (_event(actor="human", move="2g2f", score=None), "human_move_observation", False),
        (_event(move="2g2f", score=90, opening=True), "opening_move_difference", False),
    ],
)
def test_diagnosis_keeps_observation_and_failure_signals_separate(
    event: dict[str, object],
    expected_kind: str,
    hard: bool,
) -> None:
    result = diagnose_decision(
        event,
        _search(("7g7f", 100), ("2g2f", -100)),
        CONFIG,
    )

    assert result["kind"] == expected_kind
    assert result["hardExample"] is hard


def test_diagnosis_reports_exact_and_lower_bound_regret_without_causal_claim() -> None:
    exact = diagnose_decision(
        _event(move="2g2f", score=90),
        _search(("7g7f", 100), ("2g2f", -100)),
        CONFIG,
    )
    outside = diagnose_decision(
        _event(move="3g3f", score=90),
        _search(("7g7f", 100), ("2g2f", -100)),
        CONFIG,
    )

    assert exact["moveRegretCp"] == 200
    assert exact["moveRegretLowerBoundCp"] is None
    assert exact["confidence"] == "candidate"
    assert outside["moveRegretCp"] is None
    assert outside["moveRegretLowerBoundCp"] == 200
    assert outside["confidence"] == "candidate"


@pytest.mark.parametrize(
    "search",
    [
        {"bestmove": "7g7f", "candidates": []},
        {
            "bestmove": "2g2f",
            "candidates": [{"score": {"kind": "cp", "value": 1}, "pv": ["7g7f"]}],
        },
        {
            "bestmove": "7g7f",
            "candidates": [{"score": {"kind": "unknown", "value": 1}, "pv": ["7g7f"]}],
        },
    ],
)
def test_diagnosis_rejects_incomplete_or_inconsistent_teacher_evidence(
    search: dict[str, object],
) -> None:
    with pytest.raises(DiagnosisError):
        diagnose_decision(_event(move="7g7f"), search, CONFIG)
