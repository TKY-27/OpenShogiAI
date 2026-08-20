from __future__ import annotations

import copy
import hashlib

import pytest
from open_shogi_training.selfplay.common import ContractError, canonical_sha256
from open_shogi_training.selfplay.config import SelfPlayConfig
from open_shogi_training.selfplay.execution import validate_paired_plan
from open_shogi_training.selfplay.planning import (
    ModelSpec,
    build_arena_plan,
    build_selfplay_plan,
    build_start_position_validation_plan,
    build_teacher_labeling_plan,
    build_training_plan,
    parse_start_positions,
    validate_start_position_validation,
    validate_training_plan,
)

from .conftest import make_ref, make_start_rows

SFEN_A = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"
SFEN_B = "lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 1"


def _start_raw() -> dict[str, object]:
    return {
        "schema": "phase6_start_positions/v1",
        "datasetManifest": make_ref("data/manifest.json").as_dict(),
        "sourcePositions": make_ref("data/positions.jsonl.gz").as_dict(),
        "selection": {
            "seed": 20260808,
            "trainCount": 10,
            "validationCount": 10,
            "requireEligible": True,
            "resetMoveNumber": True,
            "excludedCrossSplitStates": 0,
        },
        "positions": [
            *make_start_rows(split="validation", prefix="validation", base_sfen=SFEN_B),
            *make_start_rows(split="train", prefix="train", base_sfen=SFEN_A),
        ],
    }


def test_selfplay_and_arena_plans_are_deterministic_and_color_paired(
    selfplay_config: SelfPlayConfig,
) -> None:
    starts = parse_start_positions(_start_raw())
    champion = ModelSpec("champion-v0", make_ref("weights/champion.osaval"), "neural")
    challenger = ModelSpec("challenger-v1", make_ref("weights/challenger.osaval"), "neural")
    arguments = {
        "generation_id": "generation-0001",
        "engine": make_ref("target/release/open-shogi-cli"),
        "model_registry": make_ref("weights/model-registry.json"),
        "git_commit": "a399407",
        "config": selfplay_config,
        "config_ref": make_ref("configs/selfplay/phase6_smoke.toml"),
        "start_positions": starts,
        "start_positions_ref": make_ref("artifacts/phase6-inputs/start-positions.json"),
        "validation_ref": make_ref("artifacts/phase6-inputs/start-validation.json"),
    }

    first = build_selfplay_plan(champion=champion, **arguments)
    second = build_selfplay_plan(champion=champion, **arguments)
    arena = build_arena_plan(champion=champion, challenger=challenger, **arguments)

    assert first == second
    assert first["gameCount"] == 40
    assert len(first["jobs"]) == 20
    assert sum(job["startGroup"] == "initial" for job in first["jobs"]) == 10
    assert all(job["modelAColorOrder"] == ["black", "white"] for job in first["jobs"])
    assert len({job["seed"] for job in first["jobs"]}) == 20
    assert validate_paired_plan(first) == first
    assert validate_paired_plan(arena) == arena
    command = arena["jobs"][0]["command"]["argv"]
    assert "--a-model-kind" not in command
    assert command[command.index("--player-a") + 1] == "neural"
    assert command[command.index("--a-model") + 1] == "weights/challenger.osaval"

    uppercase_commit = copy.deepcopy(first)
    uppercase_commit["gitCommit"] = "A399407"
    for job in uppercase_commit["jobs"]:
        argv = job["command"]["argv"]
        argv[argv.index("--git-commit") + 1] = "A399407"
    without_hash = dict(uppercase_commit)
    without_hash.pop("planSha256")
    uppercase_commit["planSha256"] = canonical_sha256(without_hash)
    with pytest.raises(ContractError, match="lowercase hexadecimal"):
        validate_paired_plan(uppercase_commit)


def test_start_positions_require_exact_rust_legality_evidence() -> None:
    starts = parse_start_positions(_start_raw())
    start_ref = make_ref("artifacts/start.json")
    engine_ref = make_ref("target/release/open-shogi-cli")
    plan = build_start_position_validation_plan(
        start_positions_ref=start_ref,
        positions=starts,
        engine_ref=engine_ref,
        git_commit="a399407",
        engine_cli="target/release/open-shogi-cli",
    )
    validation = {
        "schema": "phase6_start_position_validation/v1",
        "startPositions": start_ref.as_dict(),
        "engine": engine_ref.as_dict(),
        "engineBuildReceipt": None,
        "gitCommit": "a399407",
        "method": "open-shogi-cli-perft-depth-0",
        "results": [
            {
                "positionId": position.position_id,
                "sfenSha256": hashlib.sha256(position.sfen.encode()).hexdigest(),
                "legal": True,
                "returnCode": 0,
                "timedOut": False,
                "outputLimitExceeded": False,
                "memoryLimitExceeded": False,
                "peakRssBytes": 1,
                "rssMeasurement": "process_tree_ps_rss_sum",
                "stdout": make_ref(f"artifacts/{position.position_id}.stdout").as_dict(),
                "stderr": make_ref(f"artifacts/{position.position_id}.stderr").as_dict(),
                "completedAt": "2026-08-08T00:00:00Z",
            }
            for position in starts.positions
        ],
    }

    assert len(plan["checks"]) == 20
    validate_start_position_validation(
        validation,
        start_positions_ref=start_ref,
        engine_ref=engine_ref,
        positions=starts,
    )
    validation["results"][0]["legal"] = False
    with pytest.raises(ContractError, match="not proven legal"):
        validate_start_position_validation(
            validation,
            start_positions_ref=start_ref,
            engine_ref=engine_ref,
            positions=starts,
        )

    for measurement, peak, message in (
        ("process_tree_ps_rss_sum", None, "RSS evidence"),
        ("process_tree_ps_short_lived_no_sample", 1, "short-lived RSS"),
        ("unavailable", None, "cannot succeed without RSS"),
    ):
        malformed = copy.deepcopy(validation)
        malformed["results"][0]["legal"] = True
        malformed["results"][0]["rssMeasurement"] = measurement
        malformed["results"][0]["peakRssBytes"] = peak
        with pytest.raises(ContractError, match=message):
            validate_start_position_validation(
                malformed,
                start_positions_ref=start_ref,
                engine_ref=engine_ref,
                positions=starts,
            )


def test_training_plan_invokes_only_the_stable_model_cli() -> None:
    plan = build_training_plan(
        generation_id="generation-0001",
        parent_model=ModelSpec("champion-v0", make_ref("weights/champion.osaval"), "neural"),
        replay_manifest=make_ref("artifacts/replay.json"),
        labels=make_ref("artifacts/labels.jsonl"),
        label_manifest=make_ref("artifacts/label-manifest.json"),
        positions=make_ref("data/positions.jsonl.gz"),
        dataset_manifest=make_ref("data/manifest.json"),
        features_config=make_ref("configs/features/value_v0.toml"),
        model_config=make_ref("configs/models/value_v0.toml"),
        training_config=make_ref("configs/training/value_v0.toml"),
        output_dir="artifacts/phase6/generation-0001/training",
        timeout_seconds=3600,
    )

    argv = plan["command"]["argv"]
    assert argv[:4] == ["python", "-m", "open_shogi_training.models", "train"]
    assert argv[argv.index("--dataset-manifest") + 1] == "data/manifest.json"
    assert argv[argv.index("--replay-manifest") + 1] == "artifacts/replay.json"
    assert argv[argv.index("--positions") + 1] == "data/positions.jsonl.gz"
    assert plan["planSha256"]
    mutated = copy.deepcopy(plan)
    mutated["command"]["argv"].extend(["--unplanned", "value"])
    with pytest.raises(ContractError, match="does not exactly match"):
        validate_training_plan(mutated)


def test_supplemental_teacher_plan_preserves_selection_provenance_and_global_cap() -> None:
    plan = build_teacher_labeling_plan(
        generation_id="generation-0001",
        hard_positions=make_ref("artifacts/hard-positions.json"),
        normalized_positions=make_ref("artifacts/hard-positions.jsonl.gz"),
        dataset_manifest=make_ref("artifacts/hard-dataset-manifest.json"),
        labeling_config=make_ref("configs/teacher/apery.yaml"),
        benchmark_report=make_ref("artifacts/teacher-benchmark.json"),
        output_dir="artifacts/phase6/generation-0001/teacher-labels",
        target_completed=10_000,
        labels_before=0,
        teacher_label_limit=10_000,
        timeout_seconds=3600,
    )

    assert plan["command"]["argv"][:4] == [
        "python",
        "-m",
        "open_shogi_training.labeling",
        "label",
    ]
    assert plan["hardPositions"]["path"] == "artifacts/hard-positions.json"
    final_increment = build_teacher_labeling_plan(
        generation_id="generation-0001",
        hard_positions=make_ref("artifacts/final-hard-positions.json"),
        normalized_positions=make_ref("artifacts/final-hard-positions.jsonl.gz"),
        dataset_manifest=make_ref("artifacts/final-hard-dataset-manifest.json"),
        labeling_config=make_ref("configs/teacher/apery.yaml"),
        benchmark_report=make_ref("artifacts/teacher-benchmark.json"),
        output_dir="artifacts/phase6/generation-0001/final-teacher-labels",
        target_completed=10_000,
        labels_before=9_999,
        teacher_label_limit=10_000,
        timeout_seconds=3600,
    )
    assert final_increment["targetCompleted"] == 10_000
    with pytest.raises(ContractError, match="must be one of"):
        build_teacher_labeling_plan(
            generation_id="generation-0001",
            hard_positions=make_ref("artifacts/hard-positions.json"),
            normalized_positions=make_ref("artifacts/hard-positions.jsonl.gz"),
            dataset_manifest=make_ref("artifacts/hard-dataset-manifest.json"),
            labeling_config=make_ref("configs/teacher/apery.yaml"),
            benchmark_report=make_ref("artifacts/teacher-benchmark.json"),
            output_dir="artifacts/teacher",
            target_completed=10_001,
            labels_before=0,
            teacher_label_limit=10_000,
            timeout_seconds=3600,
        )
    with pytest.raises(ContractError, match="must be one of"):
        build_teacher_labeling_plan(
            generation_id="generation-0001",
            hard_positions=make_ref("artifacts/hard-positions.json"),
            normalized_positions=make_ref("artifacts/hard-positions.jsonl.gz"),
            dataset_manifest=make_ref("artifacts/hard-dataset-manifest.json"),
            labeling_config=make_ref("configs/teacher/apery.yaml"),
            benchmark_report=make_ref("artifacts/teacher-benchmark.json"),
            output_dir="artifacts/teacher",
            target_completed=1,
            labels_before=0,
            teacher_label_limit=10_000,
            timeout_seconds=3600,
        )
    with pytest.raises(ContractError, match="cap is exhausted"):
        build_teacher_labeling_plan(
            generation_id="generation-0001",
            hard_positions=make_ref("artifacts/hard-positions.json"),
            normalized_positions=make_ref("artifacts/hard-positions.jsonl.gz"),
            dataset_manifest=make_ref("artifacts/hard-dataset-manifest.json"),
            labeling_config=make_ref("configs/teacher/apery.yaml"),
            benchmark_report=make_ref("artifacts/teacher-benchmark.json"),
            output_dir="artifacts/teacher",
            target_completed=1,
            labels_before=10_000,
            teacher_label_limit=10_000,
            timeout_seconds=3600,
        )
