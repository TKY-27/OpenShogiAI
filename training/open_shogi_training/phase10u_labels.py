"""Observed-only Phase 10U teacher targets; legacy labels remain byte-preserved."""

from __future__ import annotations

import math
from typing import Any

TARGET_SCHEMA = "open_shogiai_phase10u_observed_targets/v1"
REQUESTED_MULTIPV = 3


def score_order(score: dict[str, Any]) -> tuple[int, int]:
    """Order exact mate labels symbolically without inventing a centipawn score."""
    kind, value = score.get("kind"), score.get("value")
    if kind not in {"cp", "mate"} or isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid teacher score")
    if not -(2**31) <= value < 2**31 or (kind == "mate" and value == 0):
        raise ValueError("invalid teacher score domain")
    if kind == "cp":
        if abs(value) > 28999:
            raise ValueError("teacher cp enters reserved mate namespace")
        return 1, value
    return (2, -value) if value > 0 else (0, -value)


def observed_targets(
    candidates: list[dict[str, Any]], *, factual_wdl: int | None
) -> dict[str, Any]:
    """Derive masks for a1's existing heads; no optional hard-policy target is enabled."""
    if not 1 <= len(candidates) <= REQUESTED_MULTIPV:
        raise ValueError("invalid observed candidate count")
    scores = [score_order(candidate["score"]) for candidate in candidates]
    pairs = []
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            if scores[left] != scores[right]:
                winner, loser = (left, right) if scores[left] > scores[right] else (right, left)
                pairs.append([winner, loser])
    primary = candidates[0]["score"]
    if factual_wdl is not None and (isinstance(factual_wdl, bool) or factual_wdl not in (0, 1, 2)):
        raise ValueError("invalid factual WDL")
    return {
        "schema": TARGET_SCHEMA,
        "requested_candidate_count": REQUESTED_MULTIPV,
        "observed_candidate_count": len(candidates),
        "value_mask": primary["kind"] == "cp",
        "value_cp": primary["value"] if primary["kind"] == "cp" else None,
        "wdl_mask": factual_wdl is not None,
        "wdl": factual_wdl,
        "wdl_source": "factual_outcome" if factual_wdl is not None else None,
        "mate_metadata": primary if primary["kind"] == "mate" else None,
        "mate_head_mask": False,
        "ranking_mask": bool(pairs),
        "ranking_pairs": pairs,
        "ranking_indices": "zero_based_observed_candidates",
        "policy_mask": False,
        "policy_target": None,
        "policy_confidence": None,
        "policy_reason": "optional_head_disabled_no_recorded_justification",
    }


def observed_ranking_loss(root_scores: list[float], targets: dict[str, Any]) -> float:
    """Mean pairwise logistic loss on observed successor scores negated to the root.

    Ties and absent candidates contribute no pair and no gradient. This reference
    function is not a claim that the current a1 trainer has a ranking head/path.
    """
    if len(root_scores) != targets["observed_candidate_count"] or any(
        not math.isfinite(value) for value in root_scores
    ):
        raise ValueError("ranking predictions must cover only observed candidates")
    pairs = targets["ranking_pairs"]
    if not pairs:
        return 0.0
    losses = []
    for winner, loser in pairs:
        difference = root_scores[loser] - root_scores[winner]
        losses.append(max(difference, 0.0) + math.log1p(math.exp(-abs(difference))))
    return sum(losses) / len(losses)
