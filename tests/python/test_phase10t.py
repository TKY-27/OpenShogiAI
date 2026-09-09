from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.phase10t import (
    BASE,
    INTEGRITY,
    QUALITY,
    arena_statistics,
    load_json,
    progression,
    safe_path,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]


def receipt() -> dict:
    return {
        "integrity": dict.fromkeys(INTEGRITY, True),
        "quality": dict.fromkeys(QUALITY, True),
        "gate": "recovery",
        "model_sha256": "a" * 64,
        "arenas": {
            mode: {
                "model_sha256": "a" * 64,
                "controls_verified": True,
                "start_ids": [str(i) for i in range(200)],
                "pairs": [[1, 0]] * 200,
            }
            for mode in ("equal_nodes", "equal_wall_clock")
        },
    }


def arena() -> dict:
    return load_json(ROOT / BASE / "arena.json")


def test_static_freeze() -> None:
    with pytest.raises(ValueError, match="Closed campaign"):
        validate(ROOT)


@pytest.mark.parametrize("key", INTEGRITY)
def test_integrity_cannot_be_repaired_by_quality_retry(key: str) -> None:
    value = receipt()
    value["integrity"][key] = False
    value["quality"]["source_metrics"] = False
    assert progression(value, arena())["action"] == "STOP_CLOSED"


def test_quality_failure_has_bounded_relabel_path() -> None:
    value = receipt()
    value["quality"]["source_metrics"] = False
    assert progression(value, arena())["action"] == "HARD_RELABEL_RETRAIN"
    value["relabel_rounds"] = 2
    assert progression(value, arena())["action"] == "REVIEW_REQUIRED"


def test_missing_quality_is_not_model_quality_failure() -> None:
    value = receipt()
    del value["quality"]["teacher_score"]
    assert progression(value, arena())["action"] == "STOP_CLOSED"


def test_both_modes_and_same_model_required() -> None:
    value = receipt()
    del value["arenas"]["equal_nodes"]
    assert progression(value, arena())["action"] == "STOP_CLOSED"
    value = receipt()
    value["arenas"]["equal_nodes"]["model_sha256"] = "b" * 64
    assert progression(value, arena())["action"] == "STOP_CLOSED"


def test_paired_gate_and_no_final_authorization() -> None:
    value = receipt()
    result = progression(value, arena())
    assert result["action"] == "GATE_PASSED"
    assert result["final_holdout_allowed"] is False
    assert result["promotion_allowed"] is False


def test_final_score_does_not_authorize_promotion() -> None:
    value = receipt()
    value["gate"] = "final_objective"
    for result in value["arenas"].values():
        result["pairs"] = [[1, 0.5, 1, 0.5]] * 400
        result["start_ids"] = [str(i) for i in range(400)]
    assert progression(value, arena())["action"] == "REVIEW_BEFORE_FINAL_PROMOTION"


def test_max_plies_exclusions_cannot_inflate_pass() -> None:
    value = receipt()
    value["arenas"]["equal_nodes"]["pairs"] = [[1, None]] * 200
    assert progression(value, arena())["action"] == "HARD_RELABEL_RETRAIN"
    result = arena_statistics([[1, None], [0, 0.5]], replicates=100)
    assert result["score"] == 0.5
    assert result["conservative_score"] == 0.375


@pytest.mark.parametrize("outcome", [True, float("nan"), -1, 0.2])
def test_invalid_outcomes_fail_closed(outcome: float) -> None:
    with pytest.raises(ValueError):
        arena_statistics([[outcome, 0]], replicates=10)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "local/final-holdout/data"])
def test_evidence_paths_deny_holdout_and_escape(name: str) -> None:
    with pytest.raises(ValueError):
        safe_path(ROOT, name)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    (tmp_path / "escape").symlink_to(ROOT)
    with pytest.raises(ValueError):
        safe_path(tmp_path, "escape/Makefile")


def test_duplicate_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"quality": true, "quality": false}')
    with pytest.raises(ValueError):
        load_json(path)


def test_control_mutation_rejected(tmp_path: Path) -> None:
    # Copy only frozen files, never ignored local data.
    with pytest.raises(ValueError, match="Closed campaign"):
        validate(tmp_path)


def test_weak_first_mode_cannot_hide_corrupt_second_mode() -> None:
    value = receipt()
    value["arenas"]["equal_nodes"]["pairs"] = [[0, 0]] * 200
    value["arenas"]["equal_wall_clock"]["model_sha256"] = "b" * 64
    value["quality"]["source_metrics"] = False
    assert progression(value, arena())["action"] == "STOP_CLOSED"


@pytest.mark.parametrize("field", ["integrity", "quality", "arenas", "model_sha256", "gate"])
def test_null_receipt_fields_fail_closed(field: str) -> None:
    value = receipt()
    value[field] = None
    assert progression(value, arena())["action"] == "STOP_CLOSED"


def test_internal_alias_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "reserved-holdout").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "reserved-holdout")
    with pytest.raises(ValueError):
        safe_path(tmp_path, "alias/data")


def test_repeated_rounds_bootstrap_as_start_clusters() -> None:
    # Two opposite starts remain uncertain even with repeated rounds.
    stats = arena_statistics([[1, 1, 1, 1], [0, 0, 0, 0]], replicates=1000)
    assert stats["lower"] == 0.0
    assert stats["score"] == 0.5


def test_offline_quality_can_relabel_before_arena() -> None:
    value = receipt()
    del value["arenas"]
    value["stage"] = "offline"
    value["quality"]["source_metrics"] = False
    assert progression(value, arena())["action"] == "HARD_RELABEL_RETRAIN"
    value["integrity"]["target_semantics"] = False
    assert progression(value, arena())["action"] == "STOP_CLOSED"
