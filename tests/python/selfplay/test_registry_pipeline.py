from __future__ import annotations

import copy
from pathlib import Path

import pytest
from open_shogi_training.selfplay.arena import analyze_arena_results, decide_promotion
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    replace_json_state,
)
from open_shogi_training.selfplay.config import load_generation_policy, parse_selfplay_config_bytes
from open_shogi_training.selfplay.pipeline import STAGES, GenerationPipeline
from open_shogi_training.selfplay.registry import (
    ModelRegistryStore,
    build_initial_registry,
    record_generation_promotion,
    register_challenger_generation,
    validate_model_registry,
    validate_pending_human_review,
)

from .conftest import (
    held_directory_authority,
    make_ref,
    tiny_model_bytes,
    write_complete_arena_evidence,
    write_ref,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _registry_results(outcome: str) -> dict[str, object]:
    games: list[dict[str, object]] = []
    metrics = {
        "championInferenceCalls": None,
        "championInferenceTimeNs": None,
        "challengerInferenceCalls": None,
        "challengerInferenceTimeNs": None,
        "championSearchNodes": None,
        "championSearchElapsedMs": None,
        "challengerSearchNodes": None,
        "challengerSearchElapsedMs": None,
        "championSearchDepthSum": None,
        "championSearches": None,
        "challengerSearchDepthSum": None,
        "challengerSearches": None,
    }
    for pair_index in range(20):
        start_group = "initial" if pair_index < 10 else "start_set"
        first_result, second_result = (
            ("black_win", "white_win") if outcome == "challenger" else ("black_win", "black_win")
        )
        for offset, (black, white, result) in enumerate(
            (
                ("challenger-v1", "champion-v0", first_result),
                ("champion-v0", "challenger-v1", second_result),
            )
        ):
            games.append(
                {
                    "gameId": f"game-{pair_index * 2 + offset:06d}",
                    "pairId": f"pair-{pair_index:04d}",
                    "startGroup": start_group,
                    "startPositionId": (
                        "standard-initial"
                        if start_group == "initial"
                        else f"start-{pair_index:02d}"
                    ),
                    "blackModelId": black,
                    "whiteModelId": white,
                    "result": result,
                    "plies": 100,
                    "illegalMoves": 0,
                    "crashes": 0,
                    "metrics": dict(metrics),
                }
            )
    return {
        "schema": "phase6_paired_arena_results/v1",
        "generationId": "generation-0001",
        "plan": make_ref("artifacts/arena-plan.json").as_dict(),
        "execution": make_ref("artifacts/arena-execution.json").as_dict(),
        "championModelId": "champion-v0",
        "challengerModelId": "challenger-v1",
        "games": games,
    }


def _registry_analysis(outcome: str) -> dict[str, object]:
    return analyze_arena_results(
        _registry_results(outcome),
        results_ref=make_ref("artifacts/arena-results.json"),
    )


def test_model_registry_round_trips_generation_zero_with_verified_artifacts(
    tmp_path: Path,
) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes(quantization=1))
    training = write_ref(tmp_path, "artifacts/training-run.json", b"training")
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="int8",
        training_run=training,
        registered_at="2026-08-08T00:00:00Z",
    )
    assert registry["models"][0]["licenseStatus"] == "pending-review"
    store = ModelRegistryStore(tmp_path, "weights/model-registry.json")

    store.create(registry)

    loaded = store.load()
    assert (tmp_path / "weights/model-registry.revisions/revision-0000000001.json").is_file()
    assert loaded["championModelId"] == "champion-v0"
    assert loaded["models"][0]["artifact"]["sha256"] == champion.sha256
    assert loaded["generations"][0]["trainingRunManifest"] == training.as_dict()


@pytest.mark.parametrize("failpoint", ["after_journal", "after_registry"])
def test_registry_transaction_recovers_create_and_update_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failpoint: str
) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    initial = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    store = ModelRegistryStore(tmp_path, "weights/model-registry.json")
    monkeypatch.setenv("OPEN_SHOGI_REGISTRY_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        store.create(initial)
    monkeypatch.delenv("OPEN_SHOGI_REGISTRY_FAILPOINT")
    assert store.load()["revision"] == 1

    challenger_ref = write_ref(
        tmp_path, "weights/challenger.osaval", tiny_model_bytes(quantization=1)
    )
    selfplay = write_ref(tmp_path, "artifacts/selfplay.json", b"selfplay")
    labels = write_ref(tmp_path, "artifacts/labels.json", b"labels")
    training = write_ref(tmp_path, "artifacts/training.json", b"training")
    candidate = register_challenger_generation(
        initial,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_ref,
        architecture_version="1",
        quantization="int8",
        selfplay_manifest=selfplay,
        teacher_labeling_manifest=labels,
        training_run_manifest=training,
        registered_at="2026-08-08T01:00:00Z",
    )
    monkeypatch.setenv("OPEN_SHOGI_REGISTRY_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        store.update(candidate, expected_revision=1)
    monkeypatch.delenv("OPEN_SHOGI_REGISTRY_FAILPOINT")

    assert store.load()["revision"] == 2
    assert not (tmp_path / "weights/.model-registry.json.transaction.json").exists()


@pytest.mark.parametrize("mutation", ["candidate", "revision"])
def test_registry_transaction_rejects_candidate_or_revision_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    store = ModelRegistryStore(tmp_path, "weights/model-registry.json")
    monkeypatch.setenv("OPEN_SHOGI_REGISTRY_FAILPOINT", "after_journal")
    with pytest.raises(RuntimeError, match="after_journal"):
        store.create(registry)
    monkeypatch.delenv("OPEN_SHOGI_REGISTRY_FAILPOINT")
    transaction = tmp_path / "weights/.model-registry.json.transaction.json"
    journal = dict(load_json(transaction))
    if mutation == "candidate":
        journal["candidate"] = {**journal["candidate"], "sha256": "f" * 64}
        expected = "identity mismatch"
    else:
        journal["expectedRevision"] = 1
        expected = "revision is invalid"
    replace_json_state(transaction, journal)

    with pytest.raises(ContractError, match=expected):
        store.load()


def test_registry_rejects_hash_mismatch(tmp_path: Path) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", b"model")
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=ArtifactRef(champion.path, "f" * 64, champion.size),
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )

    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        ModelRegistryStore(tmp_path, "weights/model-registry.json").create(registry)


def test_registry_rejects_quantization_claim_that_disagrees_with_model_bytes(
    tmp_path: Path,
) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="int8",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )

    with pytest.raises(ContractError, match="metadata disagrees"):
        ModelRegistryStore(tmp_path, "weights/model-registry.json").create(registry)


def test_registry_records_full_challenger_lineage_and_refuses_rootless_promotion(
    tmp_path: Path,
) -> None:
    champion_ref = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes(quantization=1))
    challenger_ref = write_ref(
        tmp_path, "weights/challenger.osaval", tiny_model_bytes(quantization=1)
    )
    selfplay = write_ref(tmp_path, "artifacts/selfplay.json", b"selfplay")
    labels = write_ref(tmp_path, "artifacts/labels.json", b"labels")
    training = write_ref(tmp_path, "artifacts/training.json", b"training")
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion_ref,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="int8",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    challenger = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_ref,
        architecture_version="1",
        quantization="int8",
        selfplay_manifest=selfplay,
        teacher_labeling_manifest=labels,
        training_run_manifest=training,
        registered_at="2026-08-08T01:00:00Z",
    )

    assert challenger["revision"] == 2
    assert challenger["models"][-1]["licenseStatus"] == "pending-review"
    assert challenger["championModelId"] == "champion-v0"
    assert challenger["generations"][-1]["selfplayManifest"]["sha256"] == selfplay.sha256
    with pytest.raises(ContractError, match="requires repository_root"):
        record_generation_promotion(
            challenger,
            generation_id="generation-0001",
            arena_manifest=ArtifactRef("artifacts/arena.json", "f" * 64, 15),
            promotion_decision=ArtifactRef("artifacts/promotion.json", "1" * 64, 16),
            decision="promoted",
        )


@pytest.mark.parametrize("decision", ["rejected", "inconclusive"])
def test_rootless_nonpromotion_is_not_accepted_as_an_unverified_outcome(
    decision: str, tmp_path: Path
) -> None:
    champion_ref = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    challenger_ref = write_ref(tmp_path, "weights/challenger.osaval", tiny_model_bytes())
    selfplay = write_ref(tmp_path, "artifacts/selfplay.json", b"selfplay")
    labels = write_ref(tmp_path, "artifacts/labels.json", b"labels")
    training = write_ref(tmp_path, "artifacts/training.json", b"training")
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion_ref,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    challenger = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_ref,
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=selfplay,
        teacher_labeling_manifest=labels,
        training_run_manifest=training,
        registered_at="2026-08-08T01:00:00Z",
    )

    with pytest.raises(ContractError, match="requires repository_root"):
        record_generation_promotion(
            challenger,
            generation_id="generation-0001",
            arena_manifest=ArtifactRef("artifacts/arena.json", "f" * 64, 15),
            promotion_decision=ArtifactRef("artifacts/promotion.json", "1" * 64, 16),
            decision=decision,
        )


def test_registry_rejects_rehashed_promotion_decision_and_champion_transition_mismatch(
    tmp_path: Path,
) -> None:
    champion_ref = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion_ref,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    challenger_ref = write_ref(tmp_path, "weights/challenger.osaval", tiny_model_bytes())
    registry = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_ref,
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=write_ref(tmp_path, "artifacts/selfplay.json"),
        teacher_labeling_manifest=write_ref(tmp_path, "artifacts/labels.json"),
        training_run_manifest=write_ref(tmp_path, "artifacts/training.json"),
        registered_at="2026-08-08T01:00:00Z",
    )
    results, results_ref = write_complete_arena_evidence(
        tmp_path,
        registry=registry,
        champion_ref=champion_ref,
        champion_bytes=tiny_model_bytes(),
        challenger_ref=challenger_ref,
        challenger_bytes=tiny_model_bytes(),
    )
    analysis = analyze_arena_results(results, results_ref=results_ref)
    analysis_ref = write_ref(
        tmp_path,
        "artifacts/analysis.json",
        canonical_json_bytes(analysis),
    )
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )
    policy_ref = write_ref(
        tmp_path,
        "configs/generation/phase6_promotion.toml",
        (PROJECT_ROOT / "configs/generation/phase6_promotion.toml").read_bytes(),
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        decided_at="2026-08-08T01:01:00Z",
    )
    forged = copy.deepcopy(decision)
    forged["decision"] = "promoted"
    forged["reasons"] = ["all_promotion_gates_passed"]
    unsigned = dict(forged)
    unsigned.pop("decisionSha256")
    forged["decisionSha256"] = canonical_sha256(unsigned)
    forged_ref = write_ref(
        tmp_path,
        "artifacts/promotion.json",
        canonical_json_bytes(forged),
    )

    with pytest.raises(ContractError, match="argument differs"):
        record_generation_promotion(
            registry,
            generation_id="generation-0001",
            arena_manifest=results_ref,
            promotion_decision=forged_ref,
            decision="inconclusive",
            repository_root=tmp_path,
        )

    valid_ref = write_ref(
        tmp_path,
        "artifacts/promotion-valid.json",
        canonical_json_bytes(decision),
    )
    finalized = record_generation_promotion(
        registry,
        generation_id="generation-0001",
        arena_manifest=results_ref,
        promotion_decision=valid_ref,
        decision="inconclusive",
        repository_root=tmp_path,
    )
    finalized["championModelId"] = "challenger-v1"
    with pytest.raises(ContractError, match="champion transition"):
        validate_model_registry(finalized, repository_root=tmp_path)

    wrong_results = write_ref(tmp_path, "artifacts/wrong-arena-results.json")
    mismatched_arena = copy.deepcopy(finalized)
    mismatched_arena["championModelId"] = "champion-v0"
    mismatched_arena["generations"][-1]["arenaManifest"] = wrong_results.as_dict()
    with pytest.raises(ContractError, match="differs from promotion analysis results"):
        validate_model_registry(mismatched_arena, repository_root=tmp_path)

    mismatched_analysis = copy.deepcopy(analysis)
    mismatched_analysis["championModelId"] = "challenger-v1"
    mismatched_analysis["challengerModelId"] = "champion-v0"
    unsigned_analysis = dict(mismatched_analysis)
    unsigned_analysis.pop("analysisSha256")
    mismatched_analysis["analysisSha256"] = canonical_sha256(unsigned_analysis)
    mismatched_analysis_ref = write_ref(
        tmp_path,
        "artifacts/mismatched-analysis.json",
        canonical_json_bytes(mismatched_analysis),
    )
    mismatched_decision = copy.deepcopy(decision)
    mismatched_decision["arenaAnalysis"] = mismatched_analysis_ref.as_dict()
    unsigned_decision = dict(mismatched_decision)
    unsigned_decision.pop("decisionSha256")
    mismatched_decision["decisionSha256"] = canonical_sha256(unsigned_decision)
    mismatched_decision_ref = write_ref(
        tmp_path,
        "artifacts/mismatched-promotion.json",
        canonical_json_bytes(mismatched_decision),
    )
    mismatched_identity = copy.deepcopy(finalized)
    mismatched_identity["championModelId"] = "champion-v0"
    mismatched_identity["generations"][-1]["promotionDecision"] = mismatched_decision_ref.as_dict()
    with pytest.raises(ContractError, match="identity differs"):
        validate_model_registry(mismatched_identity, repository_root=tmp_path)

    # Rehashing every derived layer must not launder results that were never
    # produced by the closed plan/execution/report/CSA chain.
    fabricated_results = copy.deepcopy(results)
    fabricated_results["games"][0]["result"] = "white_win"
    fabricated_results_ref = write_ref(
        tmp_path,
        "artifacts/fabricated-results.json",
        canonical_json_bytes(fabricated_results),
    )
    fabricated_analysis = analyze_arena_results(
        fabricated_results,
        results_ref=fabricated_results_ref,
    )
    fabricated_analysis_ref = write_ref(
        tmp_path,
        "artifacts/fabricated-analysis.json",
        canonical_json_bytes(fabricated_analysis),
    )
    fabricated_decision = decide_promotion(
        fabricated_analysis,
        analysis_ref=fabricated_analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        decided_at="2026-08-08T01:02:00Z",
    )
    fabricated_decision_ref = write_ref(
        tmp_path,
        "artifacts/fabricated-promotion.json",
        canonical_json_bytes(fabricated_decision),
    )
    fabricated_registry = copy.deepcopy(finalized)
    fabricated_registry["championModelId"] = "champion-v0"
    fabricated_registry["generations"][-1]["arenaManifest"] = fabricated_results_ref.as_dict()
    fabricated_registry["generations"][-1]["promotionDecision"] = fabricated_decision_ref.as_dict()
    with pytest.raises(ContractError, match="deterministic report collection"):
        validate_model_registry(fabricated_registry, repository_root=tmp_path)


def test_registry_rejects_claiming_project_generated_as_a_license() -> None:
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=ArtifactRef("weights/champion.osaval", "a" * 64, 10),
        evaluator_kind="neural",
        architecture_version="1",
        quantization="int8",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry["models"][0]["licenseStatus"] = "project-generated"

    with pytest.raises(ContractError, match="pending-review"):
        validate_model_registry(registry)


@pytest.mark.parametrize(
    ("evaluator", "architecture", "quantization", "message"),
    [
        ("material", "1", "float32", "evaluatorKind"),
        ("neural", "2", "float32", "architectureVersion"),
        ("neural", "1", "unknown", "quantization"),
    ],
)
def test_registry_builder_rejects_every_non_phase6_model_identity(
    evaluator: str, architecture: str, quantization: str, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        build_initial_registry(
            generation_id="generation-0",
            champion_model_id="champion-v0",
            champion_artifact=ArtifactRef("weights/champion.osaval", "a" * 64, 10),
            evaluator_kind=evaluator,
            architecture_version=architecture,
            quantization=quantization,
            training_run=None,
            registered_at="2026-08-08T00:00:00Z",
        )


def test_registry_rejects_an_active_initial_generation_without_a_challenger() -> None:
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=ArtifactRef("weights/champion.osaval", "a" * 64, 10),
        evaluator_kind="neural",
        architecture_version="1",
        quantization="int8",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry["generations"][0]["status"] = "arena"

    with pytest.raises(ContractError, match="initial-generation lifecycle"):
        validate_model_registry(registry)


def test_registry_rejects_a_second_initial_generation_root() -> None:
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=ArtifactRef("weights/champion.osaval", "a" * 64, 10),
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry["models"].append(
        {
            **registry["models"][0],
            "modelId": "second-root-v0",
            "generationId": "generation-second-root",
            "artifact": ArtifactRef("weights/second-root.osaval", "b" * 64, 11).as_dict(),
            "registeredAt": "2026-08-08T01:00:00Z",
        }
    )
    registry["generations"].append(
        {
            **registry["generations"][0],
            "generationId": "generation-second-root",
            "championModelId": "second-root-v0",
            "createdAt": "2026-08-08T01:00:00Z",
        }
    )
    registry["championModelId"] = "second-root-v0"

    with pytest.raises(ContractError, match="exactly one initial generation"):
        validate_model_registry(registry)


def test_registry_rejects_rebinding_the_champion_during_an_active_generation(
    tmp_path: Path,
) -> None:
    champion_ref = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    challenger_ref = write_ref(tmp_path, "weights/challenger.osaval", tiny_model_bytes())
    selfplay = write_ref(tmp_path, "artifacts/selfplay.json", b"selfplay")
    labels = write_ref(tmp_path, "artifacts/labels.json", b"labels")
    training = write_ref(tmp_path, "artifacts/training.json", b"training")
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion_ref,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_ref,
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=selfplay,
        teacher_labeling_manifest=labels,
        training_run_manifest=training,
        registered_at="2026-08-08T01:00:00Z",
    )
    registry["models"].append(
        {
            **registry["models"][0],
            "modelId": "stale-champion-v0",
            "parentModelId": "champion-v0",
            "artifact": ArtifactRef("weights/stale.osaval", "f" * 64, 15).as_dict(),
            "registeredAt": "2026-08-08T01:01:00Z",
        }
    )
    registry["championModelId"] = "stale-champion-v0"

    with pytest.raises(ContractError, match="active generation incumbent"):
        validate_model_registry(registry)


def test_registry_rejects_a_rejected_challenger_laundered_as_the_next_incumbent(
    tmp_path: Path,
) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=write_ref(tmp_path, "weights/challenger-v1.osaval", tiny_model_bytes()),
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=write_ref(tmp_path, "artifacts/selfplay-1.json"),
        teacher_labeling_manifest=write_ref(tmp_path, "artifacts/labels-1.json"),
        training_run_manifest=write_ref(tmp_path, "artifacts/training-1.json"),
        registered_at="2026-08-08T01:00:00Z",
    )
    results, results_ref = write_complete_arena_evidence(
        tmp_path,
        registry=registry,
        champion_ref=champion,
        champion_bytes=tiny_model_bytes(),
        challenger_ref=ArtifactRef.from_dict(
            registry["models"][-1]["artifact"], "challenger fixture artifact"
        ),
        challenger_bytes=tiny_model_bytes(),
    )
    analysis = analyze_arena_results(results, results_ref=results_ref)
    analysis_ref = write_ref(
        tmp_path,
        "artifacts/arena-analysis.json",
        canonical_json_bytes(analysis),
    )
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )
    policy_ref = write_ref(
        tmp_path,
        "configs/generation/phase6_promotion.toml",
        (PROJECT_ROOT / "configs/generation/phase6_promotion.toml").read_bytes(),
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        decided_at="2026-08-08T01:01:00Z",
    )
    assert decision["decision"] == "inconclusive"
    registry = record_generation_promotion(
        registry,
        generation_id="generation-0001",
        arena_manifest=results_ref,
        promotion_decision=write_ref(
            tmp_path,
            "artifacts/promotion.json",
            canonical_json_bytes(decision),
        ),
        decision="inconclusive",
        repository_root=tmp_path,
    )

    challenger_v2 = write_ref(tmp_path, "weights/challenger-v2.osaval", tiny_model_bytes())
    training_v2 = write_ref(tmp_path, "artifacts/training-2.json")
    registry["models"].append(
        {
            "modelId": "challenger-v2",
            "generationId": "generation-0002",
            "parentModelId": "challenger-v1",
            "artifact": challenger_v2.as_dict(),
            "evaluatorKind": "neural",
            "architectureVersion": "1",
            "quantization": "float32",
            "trainingRun": training_v2.as_dict(),
            "registeredAt": "2026-08-08T02:00:00Z",
            "licenseStatus": "pending-review",
        }
    )
    registry["generations"].append(
        {
            "generationId": "generation-0002",
            "parentGenerationId": "generation-0001",
            "championModelId": "challenger-v1",
            "challengerModelId": "challenger-v2",
            "selfplayManifest": write_ref(tmp_path, "artifacts/selfplay-2.json").as_dict(),
            "teacherLabelingManifest": write_ref(tmp_path, "artifacts/labels-2.json").as_dict(),
            "trainingRunManifest": training_v2.as_dict(),
            "arenaManifest": None,
            "promotionDecision": None,
            "status": "arena",
            "createdAt": "2026-08-08T02:00:00Z",
        }
    )
    registry["championModelId"] = "challenger-v1"
    registry["challengerModelId"] = "challenger-v2"

    with pytest.raises(ContractError, match="parent generation outcome"):
        validate_model_registry(registry, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("arena_outcome", "expected_decision", "expected_parent_model"),
    [
        ("challenger", "promoted", "challenger-v1"),
        ("champion", "rejected", "champion-v0"),
        ("balanced", "inconclusive", "champion-v0"),
    ],
)
def test_second_generation_uses_the_verified_immediate_parent_outcome(
    tmp_path: Path,
    arena_outcome: str,
    expected_decision: str,
    expected_parent_model: str,
) -> None:
    champion = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    challenger_v1 = write_ref(tmp_path, "weights/challenger-v1.osaval", tiny_model_bytes())
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_v1,
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=write_ref(tmp_path, "artifacts/selfplay-1.json"),
        teacher_labeling_manifest=write_ref(tmp_path, "artifacts/labels-1.json"),
        training_run_manifest=write_ref(tmp_path, "artifacts/training-1.json"),
        registered_at="2026-08-08T01:00:00Z",
    )
    results, results_ref = write_complete_arena_evidence(
        tmp_path,
        registry=registry,
        champion_ref=champion,
        champion_bytes=tiny_model_bytes(),
        challenger_ref=challenger_v1,
        challenger_bytes=tiny_model_bytes(),
        outcome=arena_outcome,
    )
    analysis = analyze_arena_results(results, results_ref=results_ref)
    analysis_ref = write_ref(
        tmp_path,
        "artifacts/arena-analysis.json",
        canonical_json_bytes(analysis),
    )
    policy = load_generation_policy(PROJECT_ROOT / "configs/generation/phase6_promotion.toml")
    policy_ref = write_ref(
        tmp_path,
        "configs/generation/phase6_promotion.toml",
        (PROJECT_ROOT / "configs/generation/phase6_promotion.toml").read_bytes(),
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        decided_at="2026-08-08T01:01:00Z",
    )
    assert decision["decision"] == expected_decision
    registry = record_generation_promotion(
        registry,
        generation_id="generation-0001",
        arena_manifest=results_ref,
        promotion_decision=write_ref(
            tmp_path,
            "artifacts/promotion.json",
            canonical_json_bytes(decision),
        ),
        decision=expected_decision,
        repository_root=tmp_path,
    )

    second = register_challenger_generation(
        registry,
        repository_root=tmp_path,
        generation_id="generation-0002",
        parent_generation_id="generation-0001",
        challenger_model_id="challenger-v2",
        challenger_artifact=write_ref(tmp_path, "weights/challenger-v2.osaval", tiny_model_bytes()),
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=write_ref(tmp_path, "artifacts/selfplay-2.json"),
        teacher_labeling_manifest=write_ref(tmp_path, "artifacts/labels-2.json"),
        training_run_manifest=write_ref(tmp_path, "artifacts/training-2.json"),
        registered_at="2026-08-08T02:00:00Z",
    )

    assert second["models"][-1]["parentModelId"] == expected_parent_model
    assert second["generations"][-1]["championModelId"] == expected_parent_model
    validate_model_registry(second, repository_root=tmp_path)


def test_generation_pipeline_requires_ordered_hashed_stages_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_bytes = (PROJECT_ROOT / "configs/selfplay/phase6_smoke.toml").read_bytes()
    config = write_ref(tmp_path, "configs/selfplay.toml", config_bytes)
    stage_refs = {
        stage: write_ref(tmp_path, f"artifacts/{stage}.json", stage.encode()) for stage in STAGES
    }
    pipeline = GenerationPipeline(
        repository_root=tmp_path,
        state_path="artifacts/generation/state.json",
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        config=config,
        config_sha256=parse_selfplay_config_bytes(config_bytes, config.path).sha256,
        output_root="artifacts/phase6/generation-0001",
    )
    pipeline.initialize()

    with pytest.raises(ContractError, match="cannot precede"):
        pipeline.record_stage("selfplayPlan", stage_refs["selfplayPlan"])
    # This unit test isolates immutable ordering and finalization. Stage contract
    # validators have dedicated producer/consumer tests with real artifacts.
    monkeypatch.setattr(pipeline, "_validate_stage_contract", lambda _stage, _ref: None)
    for stage in STAGES:
        pipeline.record_stage(stage, stage_refs[stage])

    resumed = pipeline.resume()
    assert all(resumed["stages"].values())
    manifest = pipeline.finalize("artifacts/generation/manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["stages"]["promotionDecision"] == stage_refs["promotionDecision"].as_dict()


def test_generation_pipeline_rejects_a_second_process_holding_its_state_authority(
    tmp_path: Path,
) -> None:
    config_bytes = (PROJECT_ROOT / "configs/selfplay/phase6_smoke.toml").read_bytes()
    config = write_ref(tmp_path, "configs/selfplay.toml", config_bytes)
    pipeline = GenerationPipeline(
        repository_root=tmp_path,
        state_path="artifacts/generation/state.json",
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        config=config,
        config_sha256=parse_selfplay_config_bytes(config_bytes, config.path).sha256,
        output_root="artifacts/phase6/generation-0001",
    )
    with held_directory_authority(tmp_path / "artifacts/generation"):
        with pytest.raises(ContractError, match="another process owns"):
            pipeline.initialize()
        assert not pipeline.state_path.exists()
    assert pipeline.initialize()["revision"] == 1


@pytest.mark.parametrize("failpoint", ["after_state", "after_manifest"])
def test_generation_finalization_is_idempotent_across_crash_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failpoint: str
) -> None:
    config_bytes = (PROJECT_ROOT / "configs/selfplay/phase6_smoke.toml").read_bytes()
    config = write_ref(tmp_path, "configs/selfplay.toml", config_bytes)
    pipeline = GenerationPipeline(
        repository_root=tmp_path,
        state_path="artifacts/generation/state.json",
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        config=config,
        config_sha256=parse_selfplay_config_bytes(config_bytes, config.path).sha256,
        output_root="artifacts/phase6/generation-0001",
    )
    pipeline.initialize()
    monkeypatch.setattr(pipeline, "_validate_stage_contract", lambda _stage, _ref: None)
    for stage in STAGES:
        pipeline.record_stage(
            stage,
            write_ref(tmp_path, f"artifacts/stages/{stage}.json", stage.encode()),
        )
    monkeypatch.setenv("OPEN_SHOGI_PIPELINE_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match=failpoint):
        pipeline.finalize("artifacts/generation/manifest.json")
    monkeypatch.delenv("OPEN_SHOGI_PIPELINE_FAILPOINT")

    first = pipeline.finalize("artifacts/generation/manifest.json")
    second = pipeline.finalize("artifacts/generation/manifest.json")

    assert first == second
    assert first["status"] == "complete"


def test_pending_human_review_is_never_automatically_trainable(tmp_path: Path) -> None:
    csa = write_ref(tmp_path, "artifacts/human/game.csa", b"V3.0\n")
    sfens = write_ref(tmp_path, "artifacts/human/sfens.json", b"[]\n")
    manifest = {
        "schema": "phase6_pending_human_review/v1",
        "runId": "human-run-001",
        "createdAt": "2026-08-08T00:00:00Z",
        "status": "pending_human_review",
        "autoTrainingEligible": False,
        "configSha256": "a" * 64,
        "games": [
            {
                "gameId": "human-game-001",
                "modelId": "champion-v0",
                "modelSha256": "b" * 64,
                "configSha256": "a" * 64,
                "nodes": 500,
                "elapsedMs": 1000,
                "pv": ["7g7f"],
                "evaluationCp": 25,
                "humanMoves": ["7g7f"],
                "aiMoves": ["3c3d"],
                "csa": csa.as_dict(),
                "sfenSequence": sfens.as_dict(),
            }
        ],
    }

    validate_pending_human_review(manifest, repository_root=tmp_path)
    manifest["autoTrainingEligible"] = True
    with pytest.raises(ContractError, match="never be automatically trainable"):
        validate_pending_human_review(manifest, repository_root=tmp_path)
