from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from open_shogi_training.selfplay.common import ArtifactRef, ContractError
from open_shogi_training.selfplay.config import ReplayConfig, SelfPlayConfig
from open_shogi_training.selfplay.evidence import (
    build_replay_buffer_manifest,
    build_replay_candidates,
    extract_hard_positions,
)

from .conftest import make_ref

SFEN_A = "9/9/9/9/4K4/9/9/9/4k4 b - 1"
SFEN_B = "9/9/9/9/3K5/9/9/9/4k4 w - 1"
SFEN_C = "9/9/9/9/2K6/9/9/9/4k4 b - 1"
SELFPLAY_SOURCE = make_ref("artifacts/selfplay-manifest.json")
TEACHER_SOURCE = make_ref("artifacts/teacher-labels.jsonl")
MODEL_PREDICTIONS = make_ref("artifacts/model-predictions.jsonl")
UNAVAILABLE_MEASUREMENTS = [
    "selfplay_teacher_before_after_cp",
    "selfplay_teacher_cp",
    "selfplay_model_cp",
    "selfplay_champion_challenger_moves",
    "selfplay_candidate_gap_cp",
    "selfplay_mate_distance",
    "selfplay_actual_search_nodes",
]


def _evidence_row(
    position_id: str,
    sfen: str,
    split: str = "train",
    *,
    already_labeled: bool = False,
) -> dict[str, object]:
    side = "black" if sfen.split(" ")[1] == "b" else "white"
    source_game = hashlib.sha256(position_id.encode()).hexdigest()
    return {
        "positionId": position_id,
        "sfen": sfen,
        "split": split,
        "sourceGenerationId": "generation-0" if already_labeled else "generation-0001",
        "generationOrdinal": 0 if already_labeled else 1,
        "sourceType": "teacher" if already_labeled else "selfplay",
        "sourceManifest": (
            TEACHER_SOURCE.as_dict() if already_labeled else SELFPLAY_SOURCE.as_dict()
        ),
        "sourceGameId": f"game-{position_id}" if already_labeled else source_game,
        "sourcePly": 10,
        "sideToMove": side,
        "outcomeKind": "black_win",
        "outcomeTarget": 1 if side == "black" else -1,
        "teacherBeforeCp": 300 if already_labeled else None,
        "teacherAfterCp": 50 if already_labeled else None,
        "teacherCp": 400 if already_labeled else None,
        "modelCp": 0 if already_labeled else None,
        "championMove": "7g7f" if already_labeled else None,
        "challengerMove": "2g2f" if already_labeled else None,
        "candidateGapCp": 20 if already_labeled else None,
        "mateDistance": None,
        "phase": "middlegame" if already_labeled else "endgame",
        "terminalBoundary": not already_labeled,
        "searchNodes": 100 if already_labeled else None,
        "suspectedFailure": "both" if already_labeled else "none",
        "alreadyTeacherLabeled": already_labeled,
    }


def _evidence_root(rows: list[dict[str, object]]) -> dict[str, object]:
    selfplay_rows = [row for row in rows if row["sourceType"] == "selfplay"]
    csa_exports: list[dict[str, object]] = []
    for index in range(0, len(selfplay_rows), 2):
        group = selfplay_rows[index : index + 2]
        csa = [
            ArtifactRef(
                path=f"artifacts/games/{row['sourceGameId']}.csa",
                sha256=str(row["sourceGameId"]),
                size=1,
            )
            for row in group
        ]
        if len(csa) == 1:
            filler = hashlib.sha256(f"filler-{index}".encode()).hexdigest()
            csa.append(ArtifactRef(f"artifacts/games/{filler}.csa", filler, 1))
        csa_exports.append(
            {
                "jobId": f"pair-{index // 2:04d}",
                "csa": [reference.as_dict() for reference in csa],
                "output": make_ref(f"artifacts/exports/{index // 2}.jsonl").as_dict(),
                "stdout": make_ref(f"artifacts/logs/{index // 2}.stdout").as_dict(),
                "stderr": make_ref(f"artifacts/logs/{index // 2}.stderr").as_dict(),
            }
        )
    return {
        "schema": "phase6_position_evidence/v1",
        "generationId": "generation-0001",
        "derivation": {
            "selfplayPlan": make_ref("artifacts/selfplay-plan.json").as_dict(),
            "selfplayManifest": SELFPLAY_SOURCE.as_dict(),
            "teacherLabels": TEACHER_SOURCE.as_dict(),
            "modelPredictions": MODEL_PREDICTIONS.as_dict(),
            "datasetManifest": make_ref("data/manifest.json").as_dict(),
            "generationOrdinal": 1,
            "teacherSourceGenerationId": "generation-0",
            "csaExports": csa_exports,
            "counts": {
                "teacherPositions": len(rows) - len(selfplay_rows),
                "selfplayPositions": len(selfplay_rows),
                "modelPredictionsApplied": sum(
                    row["sourceType"] == "teacher" and row["modelCp"] is not None for row in rows
                ),
                "excludedCrossSplitTeacherRows": 0,
                "excludedProtectedSelfplayRows": 0,
                "excludedDuplicateSelfplayRows": 0,
            },
            "unavailableMeasurements": UNAVAILABLE_MEASUREMENTS,
        },
        "sourceManifests": [
            TEACHER_SOURCE.as_dict(),
            MODEL_PREDICTIONS.as_dict(),
            SELFPLAY_SOURCE.as_dict(),
        ],
        "positions": rows,
    }


def test_hard_position_selection_explains_reasons_deduplicates_and_caps_budget(
    selfplay_config: SelfPlayConfig,
) -> None:
    raw = _evidence_root(
        [
            _evidence_row("position-a", SFEN_A, already_labeled=True),
            _evidence_row("position-a-duplicate", SFEN_A, already_labeled=True),
            _evidence_row("position-b", SFEN_B),
        ]
    )

    manifest = extract_hard_positions(
        raw,
        evidence_ref=make_ref("artifacts/position-evidence.json"),
        config=selfplay_config,
        labels_before=9_999,
        requested_max=10,
    )

    assert manifest["budget"]["selected"] == 2
    assert manifest["budget"]["alreadyLabeledSelected"] == 1
    assert manifest["budget"]["newTeacherLabelsSelected"] == 1
    assert manifest["budget"]["labelsAfterMaximum"] == 10_000
    selected = next(row for row in manifest["positions"] if len(row["sourcePositionIds"]) == 2)
    assert "teacher_evaluation_drop" in selected["reasons"]
    assert "suspected_search_error" in selected["reasons"]
    assert len(selected["sourcePositionIds"]) == 2


def test_hard_position_selection_rejects_cross_split_sfen_leakage(
    selfplay_config: SelfPlayConfig,
) -> None:
    raw = _evidence_root(
        [
            _evidence_row("train-position", SFEN_A, "train", already_labeled=True),
            _evidence_row("test-position", SFEN_A, "test", already_labeled=True),
        ]
    )

    with pytest.raises(ContractError, match="more than one data split"):
        extract_hard_positions(
            raw,
            evidence_ref=make_ref("artifacts/position-evidence.json"),
            config=selfplay_config,
            labels_before=0,
        )


def test_existing_teacher_hard_positions_remain_usable_after_global_label_cap(
    selfplay_config: SelfPlayConfig,
) -> None:
    raw = _evidence_root(
        [
            _evidence_row("already-labeled", SFEN_A, already_labeled=True),
            _evidence_row("needs-new-label", SFEN_B),
        ]
    )

    manifest = extract_hard_positions(
        raw,
        evidence_ref=make_ref("artifacts/position-evidence.json"),
        config=selfplay_config,
        labels_before=10_000,
    )

    assert manifest["budget"]["alreadyLabeledSelected"] == 1
    assert manifest["budget"]["newTeacherLabelsSelected"] == 0
    assert manifest["budget"]["labelsAfterMaximum"] == 10_000
    assert manifest["positions"][0]["alreadyTeacherLabeled"] is True


def test_unknown_game_outcomes_are_omitted_from_replay(
    selfplay_config: SelfPlayConfig,
) -> None:
    unknown = _evidence_row("max-plies-position", SFEN_A)
    unknown["outcomeKind"] = "unknown"
    unknown["outcomeTarget"] = None
    evidence = _evidence_root([unknown])
    evidence_ref = make_ref("artifacts/position-evidence.json")
    hard = extract_hard_positions(
        evidence,
        evidence_ref=evidence_ref,
        config=selfplay_config,
        labels_before=10_000,
    )

    candidates = build_replay_candidates(
        evidence,
        evidence_ref=evidence_ref,
        hard_positions=hard,
        hard_positions_ref=make_ref("artifacts/hard-positions.json"),
        config=selfplay_config,
    )

    assert candidates["entries"] == []


def _replay_row(
    position_id: str,
    sfen: str,
    *,
    priority: int,
    is_new: bool,
    split: str = "train",
) -> dict[str, object]:
    return {
        "positionId": position_id,
        "sfen": sfen,
        "split": split,
        "generationId": "generation-0001" if is_new else "generation-0000",
        "generationOrdinal": 1 if is_new else 0,
        "sourceType": "selfplay" if is_new else "retained",
        "sourceManifest": make_ref(f"artifacts/{position_id}.json").as_dict(),
        "sourceGameId": f"game-{position_id}",
        "sourcePly": 3,
        "sideToMove": "black" if sfen.split(" ")[1] == "b" else "white",
        "outcomeKind": "black_win",
        "outcomeTarget": 1 if sfen.split(" ")[1] == "b" else -1,
        "stage": "middlegame",
        "priority": priority,
        "hardPosition": priority >= 90,
        "isNew": is_new,
    }


def test_replay_buffer_preserves_old_useful_data_and_logs_dedup_and_eviction(
    selfplay_config: SelfPlayConfig,
) -> None:
    bounded = replace(
        selfplay_config,
        replay=ReplayConfig(capacity=2, minimum_older_positions=1, dedup_key="canonical_sfen"),
    )
    raw = {
        "schema": "phase6_replay_candidates/v1",
        "generationId": "generation-0001",
        "positionEvidence": make_ref("artifacts/evidence.json").as_dict(),
        "hardPositions": make_ref("artifacts/hard.json").as_dict(),
        "entries": [
            _replay_row("new-high", SFEN_A, priority=100, is_new=True),
            _replay_row("new-duplicate", SFEN_A, priority=90, is_new=True),
            _replay_row("new-medium", SFEN_B, priority=80, is_new=True),
            _replay_row("old-useful", SFEN_C, priority=70, is_new=False),
        ],
    }

    manifest = build_replay_buffer_manifest(
        raw,
        candidates_ref=make_ref("artifacts/replay-candidates.json"),
        config=bounded,
        config_ref=make_ref("configs/selfplay/phase6-smoke.toml"),
    )

    assert manifest["counts"]["retained"] == 2
    assert manifest["counts"]["olderRetained"] == 1
    assert manifest["splitPolicy"]["test"] == "final_evaluation_only"
    reasons = [row["reason"] for row in manifest["deletionLog"]]
    assert "deduplicated_canonical_sfen" in reasons
    assert "capacity_eviction" in reasons
    assert manifest["configSha256"] == bounded.sha256
