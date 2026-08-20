"""Conservative, evidence-backed Phase 7 failure signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import DiagnosisConfig


class DiagnosisError(ValueError):
    """Raised when a diagnostic input is not internally consistent."""


def diagnose_decision(
    event: Mapping[str, Any],
    teacher_search: Mapping[str, Any],
    config: DiagnosisConfig,
) -> dict[str, object]:
    """Classify one recorded move without claiming causal certainty.

    Teacher MultiPV regret is a move-choice signal.  Root-score disagreement is an
    evaluation signal.  They are deliberately reported separately; the category names
    ending in ``_candidate`` are hypotheses for developer triage, not proof of cause.
    """

    actor = event.get("actor")
    engine_move = _text(event.get("moveUsi"), "event.moveUsi")
    teacher_best = _text(teacher_search.get("bestmove"), "teacher.bestmove")
    candidates = teacher_search.get("candidates")
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        raise DiagnosisError("teacher candidates must be a non-empty sequence")
    parsed: list[tuple[str, int | None]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise DiagnosisError(f"teacher candidate {index} must be an object")
        pv = candidate.get("pv")
        if not isinstance(pv, Sequence) or isinstance(pv, (str, bytes)) or not pv:
            raise DiagnosisError(f"teacher candidate {index} must have a non-empty PV")
        parsed.append(
            (_text(pv[0], f"teacher candidate {index} root move"), _cp(candidate.get("score")))
        )
    if parsed[0][0] != teacher_best:
        raise DiagnosisError("teacher bestmove must equal MultiPV rank one's root move")

    chosen_score = next((score for move, score in parsed if move == engine_move), None)
    engine_in_multipv = any(move == engine_move for move, _ in parsed)
    best_score = parsed[0][1]
    exact_regret = (
        max(0, best_score - chosen_score)
        if best_score is not None and chosen_score is not None
        else None
    )
    comparable_scores = [score for _, score in parsed if score is not None]
    lower_bound = None
    if not engine_in_multipv and best_score is not None and len(comparable_scores) == len(parsed):
        lower_bound = max(0, best_score - min(comparable_scores))

    engine_score = event.get("scoreCp")
    if engine_score is not None and (
        isinstance(engine_score, bool) or not isinstance(engine_score, int)
    ):
        raise DiagnosisError("event.scoreCp must be an integer or null")
    evaluation_gap = (
        abs(engine_score - best_score)
        if isinstance(engine_score, int) and best_score is not None
        else None
    )
    severity = max(
        value for value in (exact_regret, lower_bound, evaluation_gap, 0) if value is not None
    )

    same_move = engine_move == teacher_best
    opening = event.get("openingBook") is True
    if actor == "human":
        kind = "human_move_observation"
        confidence = "observed"
    elif actor != "ai":
        raise DiagnosisError("event.actor must be human or ai")
    elif opening:
        kind = "opening_move_aligned" if same_move else "opening_move_difference"
        confidence = "observed"
    elif same_move:
        if evaluation_gap is None:
            kind = "aligned_move_unscored"
            confidence = "limited"
        elif evaluation_gap >= config.evaluation_disagreement_cp:
            kind = "evaluation_failure_candidate"
            confidence = "candidate"
        else:
            kind = "aligned"
            confidence = "observed"
    else:
        regret_signal = exact_regret if exact_regret is not None else lower_bound
        if regret_signal is None:
            kind = "uncertain_move_difference"
            confidence = "limited"
        elif regret_signal < config.move_regret_cp:
            kind = (
                "move_choice_difference"
                if exact_regret is not None
                else "uncertain_move_outside_multipv"
            )
            confidence = "observed" if exact_regret is not None else "limited"
        elif evaluation_gap is None:
            kind = "uncertain_failure_candidate"
            confidence = "limited"
        elif evaluation_gap >= config.evaluation_disagreement_cp:
            kind = "mixed_failure_candidate"
            confidence = "candidate"
        else:
            kind = "search_failure_candidate"
            confidence = "candidate"

    hard = (
        actor == "ai"
        and not opening
        and severity >= config.hard_example_cp
        and kind
        in {
            "evaluation_failure_candidate",
            "search_failure_candidate",
            "mixed_failure_candidate",
            "uncertain_failure_candidate",
        }
    )
    return {
        "kind": kind,
        "confidence": confidence,
        "teacherBestMove": teacher_best,
        "engineMove": engine_move,
        "engineMoveInMultiPv": engine_in_multipv,
        "moveRegretCp": exact_regret,
        "moveRegretLowerBoundCp": lower_bound,
        "evaluationDisagreementCp": evaluation_gap,
        "severityCp": severity,
        "hardExample": hard,
    }


def _cp(value: object) -> int | None:
    if not isinstance(value, Mapping) or set(value) != {"kind", "value"}:
        raise DiagnosisError("teacher score must contain exactly kind and value")
    kind = value.get("kind")
    score = value.get("value")
    if isinstance(score, bool) or not isinstance(score, int) or abs(score) > 1_000_000_000:
        raise DiagnosisError("teacher score value is outside its integer bound")
    if kind == "cp":
        return score
    if kind == "mate":
        return None
    raise DiagnosisError("teacher score kind must be cp or mate")


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4_096
        or any(character in value for character in "\x00\r\n")
    ):
        raise DiagnosisError(f"{context} must be a bounded single-line string")
    return value
