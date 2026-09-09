from __future__ import annotations

import math

import pytest
from open_shogi_training.phase10u_labels import observed_ranking_loss, observed_targets


def _candidates(*scores: tuple[str, int]) -> list[dict]:
    return [{"score": {"kind": kind, "value": value}} for kind, value in scores]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_only_observed_candidates_supply_ranking_pairs(count: int) -> None:
    targets = observed_targets(
        _candidates(*[("cp", 100 - i) for i in range(count)]), factual_wdl=None
    )
    assert targets["observed_candidate_count"] == count
    assert len(targets["ranking_pairs"]) == count * (count - 1) // 2
    assert targets["ranking_mask"] is (count > 1)
    assert targets["policy_mask"] is False
    assert targets["wdl_mask"] is False
    assert observed_ranking_loss([0.0] * count, targets) == pytest.approx(
        math.log(2) if count > 1 else 0
    )
    with pytest.raises(ValueError, match="only observed"):
        observed_ranking_loss([0.0] * (count + 1), targets)


def test_symbolic_mate_never_fabricates_cp_or_wdl() -> None:
    targets = observed_targets(_candidates(("mate", 3), ("cp", 500), ("mate", -9)), factual_wdl=2)
    assert targets["value_mask"] is False
    assert targets["value_cp"] is None
    assert targets["mate_metadata"] == {"kind": "mate", "value": 3}
    assert targets["mate_head_mask"] is False
    assert targets["wdl"] == 2
    assert targets["wdl_source"] == "factual_outcome"
    assert targets["ranking_pairs"] == [[0, 1], [0, 2], [1, 2]]
    assert observed_ranking_loss([2, 1, 0], targets) < observed_ranking_loss([0, 1, 2], targets)


def test_ties_and_singletons_have_zero_ranking_loss() -> None:
    targets = observed_targets(_candidates(("cp", 0), ("cp", 0)), factual_wdl=None)
    assert targets["ranking_mask"] is False
    assert observed_ranking_loss([999, -999], targets) == 0


@pytest.mark.parametrize(
    "score", [("cp", True), ("cp", 30000), ("mate", 0), ("cp", 1.5), ("wdl", 1)]
)
def test_invalid_score_is_rejected(score: tuple) -> None:
    with pytest.raises(ValueError):
        observed_targets(_candidates(score), factual_wdl=None)
