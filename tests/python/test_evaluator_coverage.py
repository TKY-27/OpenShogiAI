import gzip
import hashlib
import json

import pytest
from open_shogi_training.evaluator_coverage import coverage_report


def queue(output, tasks):
    generation = output / "generation.json"
    if not generation.exists():
        generation.write_text("{}")
    (output / "recovery-queue.json").write_text(
        json.dumps(
            {
                "schema": "open_shogiai_recovery_queue/v1",
                "generation_sha256": hashlib.sha256(generation.read_bytes()).hexdigest(),
                "tasks": tasks,
            }
        )
    )


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    families = [
        {"id": name, "group": name, "split": "train"}
        for name in ("general", "opening", "defense", "attack_end")
    ]
    config = {
        "defense_campaign": {"families": families},
        "recovery_policy": {
            "minimum_completed_per_family": 1,
            "maximum_deferred_per_family": 1,
            "maximum_deferred_roots": 4,
            "minimum_focus_stratum_requests": 20,
            "minimum_focus_stratum_completion_rate": 0.95,
        },
    }
    focus = {"passed": True, "by": {"group": {}, "ply_stage": {}, "branch": {}}}
    monkeypatch.setattr("open_shogi_training.defense_scenarios.focus_gate", lambda *a: focus)
    (tmp_path / "games").mkdir()
    for game, family in enumerate(families):
        raw = {
            "game": game,
            "family": family["id"],
            "group": family["group"],
            "split": "train",
            "variant": 0,
            "end": "None",
        }
        path = tmp_path / "games" / f"{game:06d}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(raw).encode()))
        path.with_suffix(".receipt.json").write_text(
            json.dumps({"game": game, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        )
    queue(tmp_path, [])
    return tmp_path, config, focus


def test_deferred_does_not_prevent_covered_stage(cohort):
    output, config, _ = cohort
    queue(output, [{"game": 6, "ply": 116, "branch": "root", "status": "deferred"}])
    result = coverage_report(output, config)
    assert result["passed"]
    assert result["deferred_roots"] == 1
    assert result["deferred_by"]["group"] == {"defense": 1}


def test_defense_missingness_cannot_hide_in_general_totals(cohort):
    output, config, focus = cohort
    focus["by"]["group"]["defense"] = {"requested": 20, "completed": 18, "unlabeled": 2}
    result = coverage_report(output, config)
    assert not result["passed"]
    assert "focus_group:defense" in result["reasons"]
    queue(
        output,
        [{"game": game, "ply": 110, "branch": "root", "status": "deferred"} for game in (6, 10)],
    )
    assert "family_deferred:defense" in coverage_report(output, config)["reasons"]


def test_partial_output_is_not_completion_and_split_corruption_fails(cohort):
    output, config, _ = cohort
    path = output / "games/000002.json.gz"
    receipt = path.with_suffix(".receipt.json")
    receipt.unlink()
    assert "family_completed:defense" in coverage_report(output, config)["reasons"]
    raw = json.loads(gzip.decompress(path.read_bytes()))
    raw["split"] = "validation"
    path.write_bytes(gzip.compress(json.dumps(raw).encode()))
    receipt.write_text(
        json.dumps({"game": 2, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    with pytest.raises(ValueError, match="lineage/split"):
        coverage_report(output, config)


def test_missing_queue_is_not_zero_backlog(cohort):
    output, config, _ = cohort
    (output / "recovery-queue.json").unlink()
    with pytest.raises(FileNotFoundError):
        coverage_report(output, config)


def test_focus_gate_ignores_uncommitted_output_without_mock(tmp_path):
    from open_shogi_training.defense_scenarios import focus_gate

    (tmp_path / "games").mkdir()
    (tmp_path / "games/000000.json.gz").write_bytes(b"partial")
    config = {
        "recovery_policy": {"enabled": True},
        "defense_campaign": {"maximum_unlabeled_focus": 200, "minimum_focus_completion_rate": 0.95},
    }
    assert focus_gate(tmp_path, config)["requested"] == 0


def test_ledger_missing_focus_requires_exhausted_identity(tmp_path):
    from open_shogi_training.defense_scenarios import _validate_deferred_observation
    from open_shogi_training.evaluator_ledger import Ledger

    config = {
        "teacher_binary_sha256": "teacher",
        "teacher_config_sha256": "config",
        "recovery_policy": {
            "maximum_hard_attempts": 1,
            "maximum_attempt_seconds": 60,
            "maximum_hard_seconds": 60,
        },
    }
    ledger = Ledger(tmp_path)
    key, _ = ledger.task(
        {"sfen": "sfen", "branch": "deviation", "teacher": "teacher", "teacher_config": "config"}
    )
    ledger.begin(key, 2)
    ledger.finish(key, "hard")
    observation = {"sfen": "sfen", "status": "unlabeled_deferred", "ledger_task": key}
    with pytest.raises(ValueError, match="budget mismatch"):
        _validate_deferred_observation(tmp_path, observation, config)
    ledger.begin(key, 32)
    ledger.finish(key, "deferred")
    _validate_deferred_observation(tmp_path, observation, config)
    with pytest.raises(ValueError, match="identity"):
        _validate_deferred_observation(tmp_path, {**observation, "sfen": "other"}, config)
    ledger.close()
