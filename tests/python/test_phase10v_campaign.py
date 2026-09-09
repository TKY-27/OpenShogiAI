from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest
from open_shogi_training.phase10v_campaign import (
    CUTOFF,
    DEADLINE,
    FLOOR,
    ORIGINS,
    ArenaScore,
    CampaignError,
    CandidateRejectedError,
    larger_labeling_allowed,
    load_configs,
    next_action,
    pair_statistics,
    select_candidate,
    selfplay_permission,
    strength,
    validate_partitions,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.fromisoformat("2026-09-12T10:00:00+09:00")


def scores(value=0.40, lower=0.35, model="a" * 64, count=800):
    return [
        ArenaScore(model, mode, value, lower, 1.0, count, "b" * 64)
        for mode in ("equal_nodes", "equal_wall_clock")
    ]


def test_frozen_configs_are_executable():
    assert load_configs(ROOT)["campaign"]["widths"] == [256, 512]


def test_scientific_failure_advances_all_variants_and_stages_then_hard_rounds():
    state = {"completed": {}}
    seen = []
    for _ in range(90):
        action = next_action(state, now=NOW, free_bytes=FLOOR)
        if action["action"] == "ADVANCE_HARD_ROUND":
            state["hard_round"] = action["hard_round"]
            break
        seen.append(action["step_id"])
        # A failed candidate's immutable failure receipt counts as an attempted
        # step, not a campaign stop. No offline threshold is consulted.
        state["completed"][action["step_id"]] = "f" * 64
    assert "100000/w256/base/train" in seen
    assert "100000/w512/hard1/equal_wall_clock" in seen
    assert "500000/labels" in seen and "1000000/labels" in seen
    assert state["hard_round"] == 3
    assert next_action(state, now=NOW, free_bytes=FLOOR)["step_id"].startswith("hard3/")


def test_integrity_disk_and_sunday_boundaries():
    assert (
        next_action({"integrity_errors": ["hash changed"]}, now=NOW, free_bytes=FLOOR)["action"]
        == "STOP_CLOSED"
    )
    assert next_action({}, now=NOW, free_bytes=FLOOR - 1)["action"] == "PRESERVE_AND_REVIEW_CLEANUP"
    assert next_action({}, now=CUTOFF, free_bytes=FLOOR)["action"] == "FINALIZE"
    assert next_action({}, now=DEADLINE, free_bytes=FLOOR)["action"] == "FINALIZE"
    with pytest.raises(CampaignError):
        next_action({"completed": {"preflight": True}}, now=NOW, free_bytes=FLOOR)


def test_pair_bootstrap_uses_scheduled_denominator_and_keeps_colors_together():
    score, lower, completion, scheduled = pair_statistics([[1, None], [0, 0.5]], replicates=100)
    assert score == 0.375 and completion == 0.75 and scheduled == 4
    assert lower <= score
    score, lower, _, _ = pair_statistics([[1, 0], [1, 0]], replicates=100)
    assert score == lower == 0.5  # A game-level bootstrap would have nonzero variance.
    with pytest.raises(CampaignError):
        pair_statistics([[1], [0]], replicates=100)


def test_gates_require_both_modes_same_model_and_relabelled_training():
    assert selfplay_permission(scores(0.35, 0.30), set(ORIGINS)) == "limited"
    assert selfplay_permission(scores(0.48, 0.43), set(ORIGINS)) == "full"
    assert selfplay_permission(scores(0.48, 0.29), set(ORIGINS)) == "teacher_relabelled_only"
    with pytest.raises(CampaignError):
        selfplay_permission(scores(), {"self_generated"})
    with pytest.raises(CampaignError):
        strength(scores()[:1])
    with pytest.raises(CampaignError):
        strength([scores()[0], scores(model="c" * 64)[1]])
    with pytest.raises(CandidateRejectedError):
        strength(scores(count=20))
    assert not larger_labeling_allowed(scores(0.19, 0.15))
    assert larger_labeling_allowed(scores(0.20, 0.15))


def test_selector_uses_worst_mode_playing_strength():
    a, b = "a" * 64, "b" * 64
    assert select_candidate({a: scores(0.40, model=a), b: scores(0.45, model=b)}) == b


def partitions():
    return {
        name: [{"component_id": name, "position_sha256": name, "source_sha256": "a" * 64}]
        for name in ("train", "validation", "calibration", "final_holdout")
    }


def test_partition_component_position_and_trajectory_leakage():
    values = partitions()
    assert validate_partitions(values)["train"] == 1
    values["validation"][0]["component_id"] = "train"
    with pytest.raises(CampaignError):
        validate_partitions(values)
    values = partitions()
    values["calibration"][0]["position_sha256"] = "train"
    with pytest.raises(CampaignError):
        validate_partitions(values)
    values = partitions()
    values["validation"][0].update(
        trajectory_origin="prior_model", teacher_relabelled=True, teacher_sha256="b" * 64
    )
    with pytest.raises(CampaignError):
        validate_partitions(values)


def test_dataset_manifest_delegates_full_verifier_and_rejects_corruption(tmp_path, monkeypatch):
    import open_shogi_training.phase10v_data as data
    from open_shogi_training.phase10u_arena_evidence import EvidenceError, digest
    from open_shogi_training.phase10v_campaign import verify_dataset

    called = []

    def verifier(*args):
        called.append(args)
        return {"rows": 1}

    monkeypatch.setattr(data, "verify_training_inputs", verifier, raising=False)
    refs = {}
    for name in ("train", "validation", "data_receipt"):
        raw = name.encode()
        (tmp_path / name).write_bytes(raw)
        refs[name] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {**refs, "split": "train"}
    assert verify_dataset(tmp_path, manifest, digest(manifest))["rows"] == 1
    assert len(called) == 1 and called[0][0] == tmp_path / "train"
    (tmp_path / "train").write_text("corrupt")
    with pytest.raises(EvidenceError):
        verify_dataset(tmp_path, manifest, digest(manifest))
    manifest["split"] = "final_holdout"
    with pytest.raises(CampaignError):
        verify_dataset(tmp_path, manifest, digest(manifest))


def test_arena_manifest_adapter_generates_identical_pair_schedule(monkeypatch, tmp_path):
    import open_shogi_training.phase10v_campaign as campaign

    # Identity validation has its own raw-replay negative controls. This unit
    # isolates schedule construction without launching any player or oracle.
    monkeypatch.setattr(campaign, "validate_manifest", lambda *args: None)
    raw = json.dumps(
        [{"sfen": f"position-{i}", "split": "development"} for i in range(200)]
    ).encode()
    (tmp_path / "starts.json").write_bytes(raw)
    ref = {"path": "starts.json", "sha256": hashlib.sha256(raw).hexdigest()}
    args = {
        "candidate": {"adapter_format": "OSAVAL03"},
        "handcrafted": {"adapter_format": "HANDCRAFTED"},
        "starts": ref,
        "oracle": {},
    }
    nodes = campaign.prepare_arena_manifest(tmp_path, mode="equal_nodes", **args)
    clock = campaign.prepare_arena_manifest(tmp_path, mode="equal_wall_clock", **args)
    assert nodes["planned_games"] == clock["planned_games"] == 400
    for key, game in nodes["games"].items():
        other = clock["games"][key]
        assert game["initial_sfen"] == other["initial_sfen"]
        assert game["seed"] == other["seed"]
        assert game["sides"] == other["sides"]
        assert game["controls"]["nodes"] == 2000
        assert other["controls"]["movetime_ns"] == 100_000_000
    with pytest.raises(CampaignError):
        campaign.prepare_arena_manifest(tmp_path, mode="equal_nodes", games=20, **args)


def test_objective_ci_is_strict_and_requires_full_sample():
    from open_shogi_training.phase10v_campaign import research_objective_met

    assert research_objective_met(scores(0.55, 0.51, count=1600))
    assert not research_objective_met(scores(0.55, 0.50, count=1600))
    with pytest.raises(CandidateRejectedError):
        research_objective_met(scores(0.55, 0.51, count=800))


def test_raw_arena_verifier_failure_cannot_be_summarized(monkeypatch, tmp_path):
    import open_shogi_training.phase10v_campaign as campaign
    from open_shogi_training.phase10u_arena_evidence import EvidenceError

    monkeypatch.setattr(campaign, "validate_manifest", lambda *args: None)
    monkeypatch.setattr(campaign, "read_receipt", lambda *args: {"game_id": "g0"})

    def reject(*args):
        raise EvidenceError("independent replay rejected")

    monkeypatch.setattr(campaign, "validate", reject)
    with pytest.raises(CampaignError, match="replay rejected"):
        campaign.verified_arena(
            tmp_path,
            {"phase10v_split": "development"},
            "a" * 64,
            [tmp_path / "receipt.json"],
            "b" * 64,
            "equal_nodes",
        )
    with pytest.raises(CampaignError, match="holdout"):
        campaign.verified_arena(
            tmp_path, {"phase10v_split": "final_holdout"}, "a" * 64, [], "b" * 64, "equal_nodes"
        )


def test_journal_cannot_waive_preflight_as_candidate_failure(tmp_path):
    from open_shogi_training.phase10v_campaign import verify_execution_state

    receipt = {"step_id": "preflight", "status": "CANDIDATE_REJECTED", "evidence": []}
    raw = json.dumps(receipt).encode()
    (tmp_path / "receipt.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    state = {
        "completed": {"preflight": sha},
        "completion_receipts": {sha: {"path": "receipt.json", "sha256": sha}},
    }
    with pytest.raises(CampaignError, match="prerequisite"):
        verify_execution_state(tmp_path, state)


def test_train_trajectory_manifests_allow_same_model_but_never_strength(monkeypatch, tmp_path):
    import open_shogi_training.phase10v_campaign as campaign
    from open_shogi_training.phase10u_arena_evidence import digest

    monkeypatch.setattr(campaign, "validate_manifest", lambda *args: None)
    raw = json.dumps([{"sfen": "train-position", "split": "train"}]).encode()
    (tmp_path / "train-starts.json").write_bytes(raw)
    starts = {"path": "train-starts.json", "sha256": hashlib.sha256(raw).hexdigest()}
    candidate = {"adapter_format": "OSAVAL03", "model": {"sha256": "a" * 64}}
    args = {"candidate": candidate, "starts": starts, "oracle": {}, "games": 2}
    for origin, opponent in (
        ("pure_vs_handcrafted", {"adapter_format": "HANDCRAFTED"}),
        ("prior_model", {"adapter_format": "OSAVAL02", "model": {"sha256": "b" * 64}}),
        ("self_generated", candidate),
    ):
        manifest = campaign.prepare_trajectory_manifest(
            tmp_path, origin=origin, opponent=opponent, **args
        )
        assert manifest["phase10v_split"] == "train"
        assert manifest["purpose"] == "teacher_relabelled_trajectory"
        assert manifest["teacher_relabel_required"] is True
        assert manifest["strength_evidence"] is False
        assert len(manifest["games"]) == 2
        with pytest.raises(CampaignError, match="holdout"):
            campaign.verified_arena(
                tmp_path, manifest, digest(manifest), [], "a" * 64, "equal_nodes"
            )
    with pytest.raises(CampaignError, match="opponent"):
        campaign.prepare_trajectory_manifest(
            tmp_path, origin="pure_vs_handcrafted", opponent=candidate, **args
        )
    with pytest.raises(CampaignError, match="different model"):
        campaign.prepare_trajectory_manifest(
            tmp_path, origin="prior_model", opponent=candidate, **args
        )
    for split in ("development", "validation", "final_holdout"):
        raw = json.dumps([{"sfen": "sealed-position", "split": split}]).encode()
        (tmp_path / "train-starts.json").write_bytes(raw)
        starts["sha256"] = hashlib.sha256(raw).hexdigest()
        with pytest.raises(CampaignError, match="split"):
            campaign.prepare_trajectory_manifest(
                tmp_path, origin="self_generated", opponent=candidate, **args
            )
