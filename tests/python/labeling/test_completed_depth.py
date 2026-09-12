from dataclasses import replace

import pytest
from open_shogi_training.labeling.usi import (
    USICandidate,
    USIEngine,
    USIProtocolError,
    USIScore,
    USISearchResult,
    _validate_completed_depth,
)

from .helpers import make_fake_project


def result(scores=(100, 50, -20), depths=(12, 12, 12), nodes=100):
    return USISearchResult(
        bestmove="7g7f",
        candidates=tuple(
            USICandidate(rank, USIScore("cp", score), (move,), depth, depth, nodes)
            for rank, score, depth, move in zip(
                (1, 2, 3), scores, depths, ("7g7f", "2g2f", "5g5f"), strict=True
            )
        ),
        elapsed_ms=1,
    )


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (result(depths=(12, 12, 11)), "complete the requested"),
        (result(depths=(11, 11, 11)), "complete the requested"),
        # Apery's final reprint can make interrupted ranks share a depth and look exact.
        (result(nodes=1000), "node ceiling"),
        (result(scores=(-287, 41, -1990)), "disagree with their ranks"),
    ],
)
def test_rejects_unfinished_or_inconsistent_teacher_labels(observation, message):
    with pytest.raises(USIProtocolError, match=message):
        _validate_completed_depth(observation, 12, 1000)


def test_completed_labels_allow_ties_and_preserve_mate_namespace():
    _validate_completed_depth(result(scores=(50, 50, -20)), 12, 1000)
    original = result()
    for scores in [
        (USIScore("mate", 3), USIScore("cp", 100), USIScore("mate", -8)),
        (USIScore("mate", 3), USIScore("mate", 5), USIScore("mate", 8)),
        (USIScore("mate", -8), USIScore("mate", -5), USIScore("mate", -3)),
    ]:
        observation = replace(
            original,
            candidates=tuple(
                replace(c, score=s) for c, s in zip(original.candidates, scores, strict=True)
            ),
        )
        _validate_completed_depth(observation, 12, 1000)


def test_fixed_depth_command_and_natural_single_legal_candidate(tmp_path):
    log = tmp_path / "commands.log"
    config, _, _ = make_fake_project(
        tmp_path, mode="one-rank", extra_arguments=["--command-log", str(log)]
    )
    with USIEngine(config, tmp_path) as engine:
        observation = engine.analyze_with_retry("state b - 1", nodes=100, depth=8)
    assert observation.primary.depth == 8
    assert "go nodes 100 depth 8" in log.read_text().splitlines()
    assert "setoption name Clear_Hash" in log.read_text().splitlines()


def test_fixed_depth_rejects_partial_final_reprint_without_retry(tmp_path):
    log = tmp_path / "commands.log"
    config, _, _ = make_fake_project(
        tmp_path, mode="later-lower-exact", extra_arguments=["--command-log", str(log)]
    )
    with (
        USIEngine(config, tmp_path) as engine,
        pytest.raises(USIProtocolError, match="complete the requested"),
    ):
        engine.analyze_with_retry("state b - 1", nodes=100, depth=8)
    assert sum(line.startswith("go ") for line in log.read_text().splitlines()) == 1
