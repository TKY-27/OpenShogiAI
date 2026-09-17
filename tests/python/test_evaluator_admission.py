import gzip
import json

import pytest
from open_shogi_training import evaluator_data as data
from open_shogi_training.defense_scenarios import (
    admission_identity,
    focus_gate,
    validate_manifest_admission,
)
from open_shogi_training.evaluator_coverage import coverage_report


def admit(output):
    data.atomic(
        output / "admission.json",
        data.encoded(
            {
                "schema": "open_shogiai_dataset_admission/v1",
                "policy": "itemwise-focus-v1",
                "generation_sha256": data.digest(output / "generation.json"),
            }
        ),
    )


def cohort(output):
    config = {
        "defense_campaign": {
            "maximum_unlabeled_focus": 200,
            "minimum_focus_completion_rate": 0.95,
            "families": [{"id": "defense", "group": "defense", "split": "train"}],
        },
        "recovery_policy": {
            "minimum_completed_per_family": 1,
            "maximum_deferred_per_family": 1,
            "maximum_deferred_roots": 4,
            "minimum_focus_stratum_requests": 20,
            "minimum_focus_stratum_completion_rate": 0.95,
        },
    }
    data.atomic(output / "generation.json", data.encoded(config))
    error = output / "failures/missing.json"
    data.atomic(error, data.encoded({"error_type": "USIIncompleteDepthError", "sfen": data.START}))
    missing = {
        "sfen": data.START,
        "status": "unlabeled_incomplete_depth",
        "failure_receipt": "failures/missing.json",
        "failure_sha256": data.digest(error),
    }
    raw = {
        "game": 0,
        "family": "defense",
        "variant": 0,
        "split": "train",
        "group": "defense",
        "prefix_moves": [],
        "records": [{"ply": 50, "deviation": missing} for _ in range(263)],
    }
    path = output / "games/000000.json.gz"
    data.atomic(path, gzip.compress(data.encoded(raw)))
    data.atomic(
        path.with_suffix(".receipt.json"),
        data.encoded({"game": 0, "sha256": data.digest(path)}),
    )
    data.atomic(
        output / "recovery-queue.json",
        data.encoded(
            {
                "schema": "open_shogiai_recovery_queue/v1",
                "generation_sha256": data.digest(output / "generation.json"),
                "tasks": [{"game": 0, "branch": "deviation", "ply": 50, "status": "deferred"}],
            }
        ),
    )
    return config


def test_itemwise_revision_retains_missingness_without_total_or_stratum_gate(tmp_path):
    config = cohort(tmp_path)
    assert not coverage_report(tmp_path, config)["passed"]
    admit(tmp_path)
    report = coverage_report(tmp_path, config)
    assert report["passed"]
    assert "focus_total" in report["legacy_focus_reasons"]
    assert "focus_group:defense" in report["legacy_focus_reasons"]
    focus = report["focus_quality"]
    assert focus["unlabeled"] == 263
    assert not focus["legacy_passed"]
    assert focus["by"]["family"]["defense"]["unlabeled"] == 263
    assert focus["unique_positions"]["unlabeled"] == 1
    assert focus["duplicate_unlabeled_observations"] == 262
    assert report["deferred_tasks"] == 1
    config["recovery_policy"]["minimum_completed_per_family"] = 2
    assert "family_completed:defense" in coverage_report(tmp_path, config)["reasons"]
    (tmp_path / "failures/missing.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        focus_gate(tmp_path, config)


def test_admission_manifest_binding_and_invalid_revision_fail_closed(tmp_path):
    cohort(tmp_path)
    validate_manifest_admission(tmp_path, {})
    admit(tmp_path)
    with pytest.raises(ValueError, match="admission changed"):
        validate_manifest_admission(tmp_path, {})
    manifest = {"dataset_admission": admission_identity(tmp_path)}
    validate_manifest_admission(tmp_path, manifest)
    path = tmp_path / "admission.json"
    revision = json.loads(path.read_text())
    revision["reason"] = "changed revision"
    path.write_text(json.dumps(revision))
    with pytest.raises(ValueError, match="admission changed"):
        validate_manifest_admission(tmp_path, manifest)
    revision["generation_sha256"] = "wrong"
    path.write_text(json.dumps(revision))
    with pytest.raises(ValueError, match="admission identity"):
        admission_identity(tmp_path)


def observation_row():
    from open_shogi_training.phase10u_execution import successor_sfen

    children = [successor_sfen(data.START, move) for move in ("7g7f", "2g2f")]
    return {
        "sfen": data.START,
        "ply": 50,
        "candidates": [
            {
                "score": {"kind": "cp", "value": value},
                "child_terminal": "None",
                "child_sfen": child,
                "pv": [move],
            }
            for child, move, value in zip(children, ("7g7f", "2g2f"), (12, 5), strict=True)
        ],
        "deviation": {
            "sfen": children[0],
            "status": "unlabeled_deferred",
            "ledger_task": "retained-task",
            "move": "7g7f",
        },
    }


def test_missing_focus_masks_only_dependent_candidate_and_preserves_independent_values():
    from open_shogi_training.defense_scenarios import focus_quality

    row = observation_row()
    game = {"group": "defense", "split": "train", "prefix_moves": [], "records": [row]}
    observations = list(data._observations(game))
    assert [(item[1]["value"], item[2]) for item in observations] == [
        (12, "root"),
        (-5, "candidate"),
    ]
    quality = focus_quality([game])
    assert quality["by"]["split"]["train"]["unlabeled"] == 1
    assert quality["excluded_candidate_observations_by_split"] == {"train": 1}
    # A labeled focus is an independent value even when its recovery is missing.
    row["deviation"] = {
        "sfen": row["candidates"][0]["child_sfen"],
        "score": {"kind": "cp", "value": -40},
        "move": "7g7f",
        "recovery": {
            "sfen": row["candidates"][1]["child_sfen"],
            "status": "unlabeled_deferred",
            "ledger_task": "retained-recovery-task",
        },
    }
    observations = list(data._observations(game))
    assert [(item[1]["value"], item[2]) for item in observations] == [
        (12, "root"),
        (-12, "candidate"),
        (-40, "deviation"),
    ]
    assert focus_quality([game])["excluded_candidate_observations_by_split"] == {"train": 1}


def test_all_focus_missing_keeps_valid_roots_without_fabricating_targets():
    rows = [observation_row(), observation_row()]
    observations = list(data._observations({"records": rows}))
    assert len(observations) == 4
    assert [item[2] for item in observations] == ["root", "candidate", "root", "candidate"]
    assert all(item[1]["value"] != 0 for item in observations)


@pytest.mark.parametrize("candidates", [[], [{"child_terminal": "None"}]])
def test_missing_root_target_cannot_create_root_or_candidate_values(candidates):
    row = observation_row()
    row["candidates"] = candidates
    observations = data._observations({"records": [row]})
    with pytest.raises((KeyError, IndexError)):
        next(observations)
