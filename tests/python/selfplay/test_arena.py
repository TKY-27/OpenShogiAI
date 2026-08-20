from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from open_shogi_training.selfplay.arena import (
    analyze_arena_results,
    arena_config_signature_bytes,
    collect_arena_results,
    decide_promotion,
    validate_phase2_pair_report,
    validate_phase6_pair_report_binding,
    validate_promotion_decision_binding,
)
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    ContractError,
    canonical_json_bytes,
    canonical_sha256,
)
from open_shogi_training.selfplay.config import SelfPlayConfig, load_generation_policy
from open_shogi_training.selfplay.planning import (
    ModelSpec,
    build_arena_plan,
)

from .conftest import (
    make_phase2_pair_report,
    make_ref,
    tiny_model_bytes,
    write_arena_registry_ref,
    write_completed_attempt_receipt,
    write_ref,
    write_selfplay_config_ref,
    write_validated_phase3_starts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_REF = ArtifactRef("configs/generation/phase6_promotion.toml", "f" * 64, 1)
NULL_METRICS = {
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


def test_arena_config_signature_matches_rust_known_vector() -> None:
    run = {
        "gameLimit": 2,
        "seed": 7,
        "initialSfen": "fixture sfen",
        "maxPlies": 3,
        "gitCommit": "abc",
        "budget": {"kind": "nodes", "value": 500},
        "playerA": {
            "label": "A",
            "evaluatorKind": "neural",
            "searchDepth": 4,
            "hashMegabytes": 16,
            "transposition": True,
            "modelArtifactSha256": "00" * 32,
            "modelArtifactSize": 123,
            "modelPayloadSha256": "11" * 32,
            "architectureVersion": 1,
            "quantization": "int8",
            "openingEnabled": True,
        },
        "playerB": {
            "label": "B",
            "evaluatorKind": "random",
            "searchDepth": None,
            "hashMegabytes": None,
            "transposition": None,
            "modelArtifactSha256": None,
            "modelArtifactSize": None,
            "modelPayloadSha256": None,
            "architectureVersion": None,
            "quantization": None,
            "openingEnabled": False,
        },
        "opening": {
            "enabled": True,
            "artifactSha256": "22" * 32,
            "artifactSize": 456,
            "maxPlies": 24,
        },
    }
    expected = (
        '{"schema":"phase2_arena_config_signature/v1","games":2,"seed":7,'
        '"initialSfen":"fixture sfen","maxPlies":3,"gitCommit":"abc",'
        '"budget":{"kind":"nodes","value":500},"playerA":{'
        '"label":"A","evaluatorKind":"neural","searchDepth":4,'
        '"hashMegabytes":16,"transposition":true,"modelArtifactSha256":"'
        + "00" * 32
        + '","modelArtifactSize":123,"modelPayloadSha256":"'
        + "11" * 32
        + '","architectureVersion":1,"quantization":"int8","openingEnabled":true},'
        '"playerB":{"label":"B","evaluatorKind":"random","searchDepth":null,'
        '"hashMegabytes":null,"transposition":null,"modelArtifactSha256":null,'
        '"modelArtifactSize":null,"modelPayloadSha256":null,'
        '"architectureVersion":null,"quantization":null,"openingEnabled":false},'
        '"opening":{"enabled":true,"artifactSha256":"'
        + "22" * 32
        + '","artifactSize":456,"maxPlies":24}}'
    ).encode()

    observed = arena_config_signature_bytes(run)
    assert observed == expected
    assert hashlib.sha256(observed).hexdigest() == (
        "da23e594633deefb8b329a1202045e4f4c559eeb1c048ccec83f6aa7913d9848"
    )


def _results(outcome: str) -> dict[str, object]:
    games: list[dict[str, object]] = []
    for pair_index in range(20):
        start_group = "initial" if pair_index < 10 else "start_set"
        start_id = "standard-initial" if start_group == "initial" else f"start-{pair_index:02d}"
        if outcome == "challenger":
            first_result, second_result = "black_win", "white_win"
        elif outcome == "champion":
            first_result, second_result = "white_win", "black_win"
        else:
            first_result, second_result = "black_win", "black_win"
        games.extend(
            [
                {
                    "gameId": f"game-{pair_index * 2:06d}",
                    "pairId": f"pair-{pair_index:04d}",
                    "startGroup": start_group,
                    "startPositionId": start_id,
                    "blackModelId": "challenger-v1",
                    "whiteModelId": "champion-v0",
                    "result": first_result,
                    "plies": 100,
                    "illegalMoves": 0,
                    "crashes": 0,
                    "metrics": dict(NULL_METRICS),
                },
                {
                    "gameId": f"game-{pair_index * 2 + 1:06d}",
                    "pairId": f"pair-{pair_index:04d}",
                    "startGroup": start_group,
                    "startPositionId": start_id,
                    "blackModelId": "champion-v0",
                    "whiteModelId": "challenger-v1",
                    "result": second_result,
                    "plies": 120,
                    "illegalMoves": 0,
                    "crashes": 0,
                    "metrics": dict(NULL_METRICS),
                },
            ]
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


def test_arena_analysis_computes_groups_sides_elo_wilson_and_optional_metrics() -> None:
    analysis = analyze_arena_results(
        _results("balanced"), results_ref=make_ref("artifacts/arena-results.json")
    )

    assert analysis["overall"]["games"] == 40
    assert analysis["overall"]["wins"] == 20
    assert analysis["overall"]["losses"] == 20
    assert analysis["overall"]["scoreRate"] == 0.5
    assert analysis["overall"]["approximateElo"] == 0.0
    assert analysis["overall"]["averagePlies"] == 110.0
    assert analysis["byStartGroup"]["initial"]["games"] == 20
    assert analysis["byStartGroup"]["start_set"]["games"] == 20
    assert analysis["metrics"]["inferenceSlowdownRatio"] is None
    lower, upper = analysis["overall"]["scoreWilson95"].values()
    assert lower < 0.5 < upper


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [("challenger", "promoted"), ("champion", "rejected"), ("balanced", "inconclusive")],
)
def test_promotion_policy_never_forces_weak_or_ambiguous_evidence(
    outcome: str, expected: str
) -> None:
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )
    analysis = analyze_arena_results(
        _results(outcome), results_ref=make_ref("artifacts/arena-results.json")
    )

    decision = decide_promotion(
        analysis,
        analysis_ref=make_ref("artifacts/arena-analysis.json"),
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:00Z",
    )

    assert decision["decision"] == expected
    if expected == "inconclusive":
        assert "evidence_does_not_cross_a_decision_boundary" in decision["reasons"]


def test_promotion_decision_binding_rejects_a_rehashed_forged_outcome() -> None:
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )
    analysis_ref = make_ref("artifacts/arena-analysis.json")
    analysis = analyze_arena_results(
        _results("balanced"), results_ref=make_ref("artifacts/arena-results.json")
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:00Z",
    )
    forged = copy.deepcopy(decision)
    forged["decision"] = "promoted"
    forged["reasons"] = ["all_promotion_gates_passed"]
    forged_without_hash = dict(forged)
    forged_without_hash.pop("decisionSha256")
    forged["decisionSha256"] = canonical_sha256(forged_without_hash)

    with pytest.raises(ContractError, match="deterministic policy result"):
        validate_promotion_decision_binding(
            forged,
            analysis=analysis,
            analysis_ref=analysis_ref,
            policy=policy,
            policy_ref=POLICY_REF,
        )


def test_arena_rejects_an_unpaired_color_schedule() -> None:
    raw = _results("balanced")
    raw["games"][1]["blackModelId"] = "challenger-v1"
    raw["games"][1]["whiteModelId"] = "champion-v0"

    with pytest.raises(ContractError, match="not color-swapped"):
        analyze_arena_results(raw, results_ref=make_ref("artifacts/arena-results.json"))


def test_promotion_gates_make_every_insufficient_sample_inconclusive() -> None:
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )

    too_few_games = _results("challenger")
    too_few_games["games"] = too_few_games["games"][:38]
    analysis = analyze_arena_results(
        too_few_games, results_ref=make_ref("artifacts/arena-results.json")
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=make_ref("artifacts/arena-analysis.json"),
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:00Z",
    )
    assert decision["decision"] == "inconclusive"
    assert "insufficient_games" in decision["reasons"]

    too_few_decisive = _results("challenger")
    for game in too_few_decisive["games"][10:]:
        game["result"] = "draw"
    analysis = analyze_arena_results(
        too_few_decisive, results_ref=make_ref("artifacts/arena-results.json")
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=make_ref("artifacts/arena-analysis.json"),
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:00Z",
    )
    assert decision["decision"] == "inconclusive"
    assert "insufficient_decisive_games" in decision["reasons"]

    too_few_start_set = _results("challenger")
    for game in too_few_start_set["games"][20:32]:
        game["startGroup"] = "initial"
    analysis = analyze_arena_results(
        too_few_start_set, results_ref=make_ref("artifacts/arena-results.json")
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=make_ref("artifacts/arena-analysis.json"),
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:00Z",
    )
    assert decision["decision"] == "inconclusive"
    assert "insufficient_start_set_games" in decision["reasons"]


@pytest.mark.parametrize(
    ("field", "reason"),
    [("illegalMoves", "illegal_move_limit_exceeded"), ("crashes", "crash_limit_exceeded")],
)
def test_promotion_rejects_any_illegal_move_or_crash(field: str, reason: str) -> None:
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )
    raw = _results("challenger")
    raw["games"][0][field] = 1
    analysis = analyze_arena_results(raw, results_ref=make_ref("artifacts/arena-results.json"))

    decision = decide_promotion(
        analysis,
        analysis_ref=make_ref("artifacts/arena-analysis.json"),
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:00Z",
    )

    assert decision["decision"] == "rejected"
    assert reason in decision["reasons"]


def test_arena_collector_binds_phase2_reports_to_planned_model_hashes(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    champion_bytes = tiny_model_bytes(quantization=0)
    challenger_bytes = tiny_model_bytes(quantization=1)
    champion = ModelSpec(
        "champion-v0",
        write_ref(tmp_path, "weights/champion.osaval", champion_bytes),
        "neural",
    )
    challenger = ModelSpec(
        "challenger-v1",
        write_ref(tmp_path, "weights/challenger.osaval", challenger_bytes),
        "neural",
    )
    engine = write_ref(tmp_path, "target/release/open-shogi-cli", b"engine")
    starts, starts_ref, validation, _, _ = write_validated_phase3_starts(
        tmp_path,
        engine_ref=engine,
        train_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1",
        validation_sfen=("lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 1"),
    )
    registry = write_arena_registry_ref(
        tmp_path,
        champion_ref=champion.artifact,
        champion_quantization="float32",
        challenger_ref=challenger.artifact,
        challenger_quantization="int8",
    )
    config_ref = write_selfplay_config_ref(tmp_path)
    plan = build_arena_plan(
        generation_id="generation-0001",
        champion=champion,
        challenger=challenger,
        engine=engine,
        model_registry=registry,
        git_commit="a399407",
        config=selfplay_config,
        config_ref=config_ref,
        start_positions=starts,
        start_positions_ref=starts_ref,
        validation_ref=validation,
    )
    plan_ref = write_ref(tmp_path, "artifacts/arena-plan.json", canonical_json_bytes(plan))
    attempts: list[dict[str, object]] = []
    for job in plan["jobs"]:
        output_dir = job["outputDir"]
        csa = (
            write_ref(tmp_path, f"{output_dir}/games/game-000001.csa", b"V3.0\n"),
            write_ref(tmp_path, f"{output_dir}/games/game-000002.csa", b"V3.0\n"),
        )
        report = make_phase2_pair_report(
            job=job,
            model_a_ref=challenger.artifact,
            model_a_bytes=challenger_bytes,
            model_b_ref=champion.artifact,
            model_b_bytes=champion_bytes,
            csa_refs=csa,
        )
        report_ref = write_ref(
            tmp_path, f"{output_dir}/arena-report.json", canonical_json_bytes(report)
        )
        stdout = write_ref(
            tmp_path,
            f"artifacts/logs/{job['jobId']}.stdout",
            b"ok\n",
        )
        stderr = write_ref(tmp_path, f"artifacts/logs/{job['jobId']}.stderr", b"")
        command_receipt = write_completed_attempt_receipt(
            tmp_path,
            plan=plan,
            job=job,
            stdout=stdout,
            stderr=stderr,
            report=report_ref,
            csa=csa,
        )
        attempts.append(
            {
                "jobId": job["jobId"],
                "attempt": 1,
                "status": "completed",
                "returnCode": 0,
                "timedOut": False,
                "outputLimitExceeded": False,
                "memoryLimitExceeded": False,
                "peakRssBytes": 1,
                "rssMeasurement": "process_tree_ps_rss_sum",
                "stdout": stdout.as_dict(),
                "stderr": stderr.as_dict(),
                "report": report_ref.as_dict(),
                "csa": [item.as_dict() for item in csa],
                "quarantine": None,
                "failureCategory": None,
                "completedAt": "2026-08-08T00:00:01Z",
                "commandReceipt": command_receipt.as_dict(),
            }
        )
    execution = {
        "schema": "phase6_arena_execution_manifest/v1",
        "generationId": "generation-0001",
        "plan": plan_ref.as_dict(),
        "planSha256": plan["planSha256"],
        "status": "completed",
        "gameCountPlanned": 40,
        "jobsPlanned": 20,
        "jobsCompleted": 20,
        "jobsQuarantined": 0,
        "gamesCompleted": 40,
        "gamesQuarantined": 0,
        "quarantinedAttempts": 0,
        "attempts": attempts,
    }
    execution["manifestSha256"] = canonical_sha256(execution)
    execution_ref = write_ref(
        tmp_path,
        "artifacts/arena-execution.json",
        canonical_json_bytes(execution),
    )

    results = collect_arena_results(
        repository_root=tmp_path,
        plan=plan,
        plan_ref=plan_ref,
        execution=execution,
        execution_ref=execution_ref,
    )

    assert len(results["games"]) == 40
    assert results["games"][0]["blackModelId"] == "challenger-v1"
    assert results["games"][0]["metrics"]["challengerInferenceCalls"] == 2
    assert results["games"][0]["metrics"]["championInferenceCalls"] == 3
    results_ref = write_ref(
        tmp_path,
        "artifacts/arena-results.json",
        canonical_json_bytes(results),
    )
    analysis = analyze_arena_results(
        results,
        results_ref=results_ref,
        repository_root=tmp_path,
    )
    assert analysis["overall"]["wins"] == 40
    assert analysis["metrics"]["inferenceSlowdownRatio"] == 1.0
    assert analysis["metrics"]["searchSlowdownRatio"] == 0.8
    analysis_ref = write_ref(
        tmp_path,
        "artifacts/arena-analysis.json",
        canonical_json_bytes(analysis),
    )
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=POLICY_REF,
        decided_at="2026-08-08T00:00:02Z",
        repository_root=tmp_path,
    )
    validate_promotion_decision_binding(
        decision,
        analysis=analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=POLICY_REF,
        repository_root=tmp_path,
    )


def test_phase2_v2_report_rejects_legacy_counter_and_csa_identity_drift(
    tmp_path: Path,
) -> None:
    model_a_bytes = tiny_model_bytes(quantization=1)
    model_b_bytes = tiny_model_bytes(quantization=0)
    model_a = write_ref(tmp_path, "weights/a.osaval", model_a_bytes)
    model_b = write_ref(tmp_path, "weights/b.osaval", model_b_bytes)
    output_dir = "artifacts/phase6/generation-0001/arena/jobs/pair-0000"
    csa_refs = (
        write_ref(tmp_path, f"{output_dir}/games/game-000001.csa", b"game-a"),
        write_ref(tmp_path, f"{output_dir}/games/game-000002.csa", b"game-b"),
    )
    job = {
        "jobId": "pair-0000",
        "seed": 123,
        "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "outputDir": output_dir,
        "csaPaths": [reference.path for reference in csa_refs],
        "command": {
            "argv": [
                "target/release/open-shogi-cli",
                "arena",
                "--games",
                "2",
                "--player-a",
                "neural",
                "--player-b",
                "neural",
                "--a-depth",
                "6",
                "--b-depth",
                "6",
                "--a-hash-mb",
                "64",
                "--b-hash-mb",
                "64",
                "--nodes",
                "500",
                "--max-plies",
                "256",
                "--seed",
                "123",
                "--sfen",
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                "--git-commit",
                "a399407",
                "--output-dir",
                output_dir,
                "--a-model",
                model_a.path,
                "--b-model",
                model_b.path,
            ]
        },
    }
    report = make_phase2_pair_report(
        job=job,
        model_a_ref=model_a,
        model_a_bytes=model_a_bytes,
        model_b_ref=model_b,
        model_b_bytes=model_b_bytes,
        csa_refs=csa_refs,
    )

    validate_phase2_pair_report(report, context="pair report")
    validate_phase6_pair_report_binding(
        report,
        repository_root=tmp_path,
        job=job,
        git_commit="a399407",
        nodes_per_move=500,
        model_a={
            "modelId": "challenger-v1",
            "artifact": model_a.as_dict(),
            "evaluatorKind": "neural",
        },
        model_b={
            "modelId": "champion-v0",
            "artifact": model_b.as_dict(),
            "evaluatorKind": "neural",
        },
        csa_refs=csa_refs,
        context="pair report",
    )

    legacy = copy.deepcopy(report)
    legacy["schema"] = "phase2_arena_report/v1"
    with pytest.raises(ContractError, match="unsupported schema"):
        validate_phase2_pair_report(legacy, context="legacy report")

    counter_drift = copy.deepcopy(report)
    counter_drift["games"][0]["playerANeuralInferenceCalls"] = 1
    with pytest.raises(ContractError, match="combined inference counters"):
        validate_phase2_pair_report(counter_drift, context="counter report")

    csa_drift = copy.deepcopy(report)
    csa_drift["games"][0]["csaSha256"] = "f" * 64
    with pytest.raises(ContractError, match="CSA identity"):
        validate_phase6_pair_report_binding(
            csa_drift,
            repository_root=tmp_path,
            job=job,
            git_commit="a399407",
            nodes_per_move=500,
            model_a={
                "modelId": "challenger-v1",
                "artifact": model_a.as_dict(),
                "evaluatorKind": "neural",
            },
            model_b={
                "modelId": "champion-v0",
                "artifact": model_b.as_dict(),
                "evaluatorKind": "neural",
            },
            csa_refs=csa_refs,
            context="CSA report",
        )
