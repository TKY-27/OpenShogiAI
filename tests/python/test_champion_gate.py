from dataclasses import replace

import pytest
from open_shogi_training.champion_gate import (
    GateConfig,
    _validate_arena_report,
    _wilson,
)
from open_shogi_training.selfplay.common import ArtifactRef


def _config() -> GateConfig:
    return GateConfig(
        sha256="a" * 64,
        games=4,
        movetime_ms=10,
        depth=8,
        hash_mib=32,
        max_plies=128,
        seed=20260821,
        candidate_id="candidate",
        candidate_kind="residual",
        model_path="artifacts/candidate.osaval",
        model_sha256="b" * 64,
        model_size=123,
        incumbent_id="handcrafted-experimental",
        confidence_level=0.95,
        minimum_decisive_games=2,
        minimum_score_rate=0.55,
        minimum_wilson_lower=0.5,
        maximum_illegal_moves=0,
        maximum_crashes=0,
        maximum_tactical_regressions=0,
        suite_path="configs/suite.json",
        suite_sha256="c" * 64,
        tactical_nodes=10_000,
        tactical_timeout_seconds=10,
    )


def _report() -> dict[str, object]:
    candidate = "search:residual:candidate"
    incumbent = "search:handcrafted-experimental"
    games = []
    results = ("black_win", "white_win", "white_win", "white_win")
    for index, result in enumerate(results):
        black, white = (candidate, incumbent) if index % 2 == 0 else (incumbent, candidate)
        games.append({"id": index, "black": black, "white": white, "result": result})
    return {
        "schema": "phase2_arena_report/v2",
        "run": {
            "gameLimit": 4,
            "gitCommit": "d" * 40,
            "maxPlies": 128,
            "seed": 20260821,
            "budget": {"kind": "movetime_ms", "value": 10},
            "opening": {"enabled": False},
            "playerA": {
                "label": candidate,
                "evaluatorKind": "residual",
                "modelArtifactSha256": "b" * 64,
                "modelArtifactSize": 123,
            },
            "playerB": {
                "label": incumbent,
                "evaluatorKind": "handcrafted-experimental",
                "modelArtifactSha256": None,
            },
        },
        "metrics": {
            "games": 4,
            "finishedGames": 4,
            "playerAWins": 3,
            "playerBWins": 1,
            "draws": 0,
            "illegalMoves": 0,
            "playerASearchElapsedMs": 100,
            "playerASearches": 10,
            "playerBSearchElapsedMs": 100,
            "playerBSearches": 10,
        },
        "games": games,
    }


def test_overall_gate_recomputes_paired_results_and_rejects_schedule_tampering() -> None:
    model = ArtifactRef("artifacts/candidate.osaval", "b" * 64, 123)
    counts, timing = _validate_arena_report(_report(), _config(), "d" * 40, model)

    assert counts == {"wins": 3, "losses": 1, "draws": 0, "illegalMoves": 0}
    assert timing["movetimeMs"] == 10

    tampered = _report()
    tampered["games"][1]["black"] = tampered["run"]["playerA"]["label"]
    with pytest.raises(ValueError, match="paired color-reversal"):
        _validate_arena_report(tampered, _config(), "d" * 40, model)


def test_overall_gate_confidence_prevents_a_coin_flip_promotion() -> None:
    lower_even, upper_even = _wilson(0.5, 40)
    lower_strong, upper_strong = _wilson(0.8, 40)

    assert lower_even < 0.5 < upper_even
    assert lower_strong > 0.5
    assert upper_strong <= 1.0
    assert replace(_config(), games=40).minimum_wilson_lower == 0.5
