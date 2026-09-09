from datetime import datetime
from pathlib import Path

import pytest
from open_shogi_training import phase10t, phase10u

NOW = datetime.fromisoformat("2026-09-09T12:00:00+09:00")
ROOT = Path(__file__).resolve().parents[2]


def receipt():
    result = {
        "integrity": dict.fromkeys(phase10t.INTEGRITY, True),
        "implementation_verified": dict.fromkeys(phase10u.IMPLEMENTATION, True),
        "quality": dict.fromkeys(phase10t.QUALITY, True),
        "model_sha256": "a" * 64,
        "relabel_rounds": 0,
        "stage": "200_game_diagnostic",
        "opponent_sha256": {"learned_1m": "b" * 64, "handcrafted_experimental": "c" * 64},
    }

    def group(model, opponent, pair):
        return {
            mode: {
                "model_sha256": model,
                "opponent_sha256": opponent,
                "controls_verified": True,
                "start_ids": [str(i) for i in range(100)],
                "pairs": [pair.copy() for _ in range(100)],
            }
            for mode in phase10u.MODES
        }

    result["arenas"] = {
        op: group("a" * 64, digest, [1, 1]) for op, digest in result["opponent_sha256"].items()
    }
    result["baseline_arenas"] = group("b" * 64, "c" * 64, [1, 0])
    return result


def evaluate(value):
    return phase10u.decision(value, now=NOW)


def test_campaign_matches_frozen_config():
    assert phase10t.load_json(ROOT / phase10u.BASE / "campaign.json") == phase10u.campaign()


def test_strength_opens_expansion_but_not_selfplay_or_research():
    result = evaluate(receipt())
    assert result["action"] == "EXPAND_1M"
    assert result["research_objective_completed"] is False
    assert result["unassisted_selfplay_allowed"] is False


@pytest.mark.parametrize("key", phase10t.INTEGRITY)
def test_every_integrity_failure_stops_before_quality_repair(key):
    value = receipt()
    del value["integrity"][key]
    value["quality"]["ranking"] = False
    assert evaluate(value)["action"] == "STOP_CLOSED"


@pytest.mark.parametrize("fault", ["starts", "opponent", "outcome", "controls", "count"])
def test_corrupt_second_opponent_stops(fault):
    value = receipt()
    result = value["arenas"]["handcrafted_experimental"]["equal_wall_clock"]
    if fault == "starts":
        result["start_ids"][0] = "other"
    elif fault == "opponent":
        result["opponent_sha256"] = "d" * 64
    elif fault == "outcome":
        result["pairs"][0][0] = True
    elif fault == "controls":
        result["controls_verified"] = False
    else:
        result["pairs"].pop()
    assert evaluate(value)["action"] == "STOP_CLOSED"


def test_capped_games_do_not_become_draws():
    value = receipt()
    pairs = value["arenas"]["learned_1m"]["equal_wall_clock"]["pairs"]
    pairs[:6] = [[None, None]] * 6
    assert evaluate(value)["action"] == "REPEAT_BOUNDED_DIAGNOSTIC"


def test_handcrafted_regression_prevents_expansion():
    value = receipt()
    value["arenas"]["handcrafted_experimental"]["equal_nodes"]["pairs"] = [[0, 0]] * 100
    assert evaluate(value)["action"] == "HARD_RELABEL_RETRAIN"


def test_quality_repair_is_bounded_and_can_justify_expansion():
    value = receipt()
    value["quality"]["ranking"] = False
    assert evaluate(value)["action"] == "HARD_RELABEL_RETRAIN"
    value["hard_example_repair"] = {
        "train_only": True,
        "split_safe": True,
        "teacher_identity_verified": True,
        "nodes": 400000,
        "positions": 25000,
        "selection_manifest_sha256": "d" * 64,
        "failure_analysis": "Observed hard ranking errors",
    }
    assert evaluate(value)["action"] == "EXPAND_1M_WITH_HARD_RELABEL_RETRAIN"
    value["hard_example_repair"]["positions"] = 25001
    assert evaluate(value)["action"] == "STOP_CLOSED"
    del value["hard_example_repair"]
    value["relabel_rounds"] = 2
    assert evaluate(value)["action"] == "REVIEW_REQUIRED"


def test_deadline_reserves_final_six_hours():
    value = receipt()
    assert (
        phase10u.decision(value, now=datetime.fromisoformat("2026-09-13T06:00:00+09:00"))["action"]
        == "EXPORT_INTEGRATE_HUMAN_EVALUATE"
    )
    assert (
        phase10u.decision(value, now=datetime.fromisoformat("2026-09-13T12:00:00+09:00"))["action"]
        == "DEADLINE_CLOSED"
    )
    assert phase10u.decision(value, now=datetime(2026, 9, 9))["action"] == "STOP_CLOSED"


def trajectory_receipt():
    value = receipt()
    value["stage"] = "teacher_relabelled_trajectories"
    for key in (
        "pure_models_only",
        "train_only",
        "split_safe",
        "teacher_relabelled",
        "teacher_identity_verified",
        "unassisted_outcomes_excluded",
        "selected_positions_only",
    ):
        value[key] = True
    value.update(trajectory_games=200, trajectory_max_plies=256, selected_positions=25000)
    return value


def test_pre48_trajectories_require_relabeling():
    value = trajectory_receipt()
    assert evaluate(value)["action"] == "RELABELLED_TRAJECTORIES_ELIGIBLE"
    value["teacher_relabelled"] = False
    assert evaluate(value)["action"] == "STOP_CLOSED"


@pytest.mark.parametrize("stage", ["selfplay_entry", "final_objective"])
def test_research_gates_remain_in_phase10t(stage):
    value = receipt()
    value["stage"] = stage
    assert evaluate(value)["action"] == "STOP_CLOSED"


@pytest.mark.parametrize("key", phase10u.IMPLEMENTATION[:3])
def test_missing_runner_wiring_stops(key):
    value = receipt()
    value["implementation_verified"][key] = False
    assert evaluate(value)["action"] == "STOP_CLOSED"


def test_400_game_checkpoint_can_offer_candidate_without_research55():
    value = receipt()
    value["stage"] = "400_game_diagnostic"
    for group in [*value["arenas"].values(), value["baseline_arenas"]]:
        for result in group.values():
            result["start_ids"] = [str(i) for i in range(200)]
            result["pairs"] *= 2
    result = evaluate(value)
    assert result["action"] == "SUNDAY_CANDIDATE_ELIGIBLE"
    assert result["research_objective_completed"] is False


def test_freeze_rejects_missing_required_members(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Closed campaign"):
        phase10u.validate(tmp_path)


@pytest.mark.parametrize(
    "field,limit",
    [("trajectory_games", 200), ("trajectory_max_plies", 256), ("selected_positions", 25000)],
)
@pytest.mark.parametrize("invalid", ["over", "missing", "bool", "float", "zero", "negative"])
def test_trajectory_bounds_are_typed_and_fail_closed(field, limit, invalid):
    value = trajectory_receipt()
    mutations = {"over": limit + 1, "bool": True, "float": float(limit), "zero": 0, "negative": -1}
    if invalid == "missing":
        del value[field]
    else:
        value[field] = mutations[invalid]
    assert evaluate(value)["action"] == "STOP_CLOSED"


def test_trajectory_repair_stops_at_two_rounds():
    value = trajectory_receipt()
    value["relabel_rounds"] = 1
    assert evaluate(value)["action"] == "RELABELLED_TRAJECTORIES_ELIGIBLE"
    value["relabel_rounds"] = 2
    assert evaluate(value)["action"] == "STOP_CLOSED"
