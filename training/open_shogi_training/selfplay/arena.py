"""Strict paired-arena analysis and conservative promotion decisions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.models.export import (
    ARCH_VERSION,
    QUANTIZATION_FLOAT32,
    QUANTIZATION_INT8,
    parse_value_model,
)

from .common import (
    ArtifactRef,
    ContractError,
    canonical_sha256,
    load_bytes_artifact,
    load_json_artifact,
    require_bool,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_list,
    require_mapping,
    require_number,
    require_sha256,
    require_string,
    validate_utc_timestamp,
    verify_artifact_ref,
)
from .config import (
    PHASE6_MAX_PLIES,
    PHASE6_NODES_PER_MOVE,
    GenerationPolicy,
)

ARENA_RESULTS_SCHEMA: Final = "phase6_paired_arena_results/v1"
ARENA_ANALYSIS_SCHEMA: Final = "phase6_paired_arena_analysis/v1"
PROMOTION_DECISION_SCHEMA: Final = "phase6_promotion_decision/v1"
MAX_ARENA_GAMES: Final = 10_000
MAX_JSON_SAFE_INTEGER: Final = 9_007_199_254_740_991
MAX_MODEL_BYTES: Final = 64 * 1024 * 1024

_RESULT_KEYS = frozenset(
    {
        "schema",
        "generationId",
        "plan",
        "execution",
        "championModelId",
        "challengerModelId",
        "games",
    }
)
_GAME_KEYS = frozenset(
    {
        "gameId",
        "pairId",
        "startGroup",
        "startPositionId",
        "blackModelId",
        "whiteModelId",
        "result",
        "plies",
        "illegalMoves",
        "crashes",
        "metrics",
    }
)
_METRIC_KEYS = frozenset(
    {
        "championInferenceCalls",
        "championInferenceTimeNs",
        "challengerInferenceCalls",
        "challengerInferenceTimeNs",
        "championSearchNodes",
        "championSearchElapsedMs",
        "challengerSearchNodes",
        "challengerSearchElapsedMs",
        "championSearchDepthSum",
        "championSearches",
        "challengerSearchDepthSum",
        "challengerSearches",
    }
)
_REPORT_RUN_KEYS = frozenset(
    {
        "seed",
        "gameLimit",
        "engine",
        "gitCommit",
        "startedAt",
        "completedAt",
        "initialSfen",
        "maxPlies",
        "configSha256",
        "budget",
        "playerA",
        "playerB",
        "opening",
    }
)
_REPORT_BUDGET_KEYS = frozenset({"kind", "value"})
_REPORT_PLAYER_KEYS = frozenset(
    {
        "label",
        "evaluatorKind",
        "searchDepth",
        "hashMegabytes",
        "transposition",
        "modelArtifactSha256",
        "modelArtifactSize",
        "modelPayloadSha256",
        "architectureVersion",
        "quantization",
        "openingEnabled",
    }
)
_REPORT_OPENING_KEYS = frozenset({"enabled", "artifactSha256", "artifactSize", "maxPlies"})
_REPORT_METRIC_KEYS = frozenset(
    {
        "games",
        "finishedGames",
        "playerAWins",
        "playerBWins",
        "searchWins",
        "draws",
        "nodesPerSecond",
        "averageDepth",
        "ttHitRate",
        "cutoffRate",
        "pruningRate",
        "millisecondsPerMove",
        "neuralInferenceCalls",
        "neuralInferenceTimeNs",
        "playerASearchNodes",
        "playerASearchElapsedMs",
        "playerADepthSum",
        "playerASearches",
        "playerANeuralInferenceCalls",
        "playerANeuralInferenceTimeNs",
        "playerBSearchNodes",
        "playerBSearchElapsedMs",
        "playerBDepthSum",
        "playerBSearches",
        "playerBNeuralInferenceCalls",
        "playerBNeuralInferenceTimeNs",
        "peakMemoryBytes",
        "illegalMoves",
    }
)
_REPORT_GAME_KEYS = frozenset(
    {
        "id",
        "black",
        "white",
        "result",
        "moves",
        "csaPath",
        "csaSha256",
        "csaSize",
        "neuralInferenceCalls",
        "neuralInferenceTimeNs",
        "playerASearchNodes",
        "playerASearchElapsedMs",
        "playerADepthSum",
        "playerASearches",
        "playerANeuralInferenceCalls",
        "playerANeuralInferenceTimeNs",
        "playerBSearchNodes",
        "playerBSearchElapsedMs",
        "playerBDepthSum",
        "playerBSearches",
        "playerBNeuralInferenceCalls",
        "playerBNeuralInferenceTimeNs",
    }
)
_REPORT_COUNTER_KEYS = (
    "neuralInferenceCalls",
    "neuralInferenceTimeNs",
    "playerASearchNodes",
    "playerASearchElapsedMs",
    "playerADepthSum",
    "playerASearches",
    "playerANeuralInferenceCalls",
    "playerANeuralInferenceTimeNs",
    "playerBSearchNodes",
    "playerBSearchElapsedMs",
    "playerBDepthSum",
    "playerBSearches",
    "playerBNeuralInferenceCalls",
    "playerBNeuralInferenceTimeNs",
)


def analyze_arena_results(
    raw: object, *, results_ref: ArtifactRef, repository_root: Path | None = None
) -> dict[str, object]:
    root = require_mapping(raw, "arena results")
    require_exact_keys(root, _RESULT_KEYS, "arena results")
    if root.get("schema") != ARENA_RESULTS_SCHEMA:
        raise ContractError("unsupported paired-arena results schema")
    generation_id = require_identifier(root, "generationId", "arena results")
    plan_ref = ArtifactRef.from_dict(root.get("plan"), "arena results.plan")
    execution_ref = ArtifactRef.from_dict(root.get("execution"), "arena results.execution")
    if repository_root is not None:
        verify_artifact_ref(repository_root, results_ref)
        verify_artifact_ref(repository_root, plan_ref)
        verify_artifact_ref(repository_root, execution_ref)
    champion_id = require_identifier(root, "championModelId", "arena results")
    challenger_id = require_identifier(root, "challengerModelId", "arena results")
    if champion_id == challenger_id:
        raise ContractError("arena champion and challenger IDs must differ")
    if repository_root is not None:
        expected = collect_arena_results(
            repository_root=repository_root,
            plan=load_json_artifact(repository_root, plan_ref),
            plan_ref=plan_ref,
            execution=load_json_artifact(repository_root, execution_ref),
            execution_ref=execution_ref,
        )
        if root != expected:
            raise ContractError("arena results are not the deterministic report collection")
    rows = require_list(
        root,
        "games",
        "arena results",
        minimum_items=2,
        maximum_items=MAX_ARENA_GAMES,
    )
    parsed = [
        _parse_game(row, index, champion_id=champion_id, challenger_id=challenger_id)
        for index, row in enumerate(rows)
    ]
    _validate_pairs(parsed, champion_id=champion_id, challenger_id=challenger_id)

    overall = _summarize(parsed, challenger_id)
    initial = _summarize(
        [game for game in parsed if game["startGroup"] == "initial"], challenger_id
    )
    start_set = _summarize(
        [game for game in parsed if game["startGroup"] == "start_set"], challenger_id
    )
    black = _summarize(
        [game for game in parsed if game["blackModelId"] == challenger_id], challenger_id
    )
    white = _summarize(
        [game for game in parsed if game["whiteModelId"] == challenger_id], challenger_id
    )
    side_gap = (
        abs(float(black["scoreRate"]) - float(white["scoreRate"]))
        if black["games"] and white["games"]
        else None
    )
    metrics = _aggregate_metrics(parsed)
    result: dict[str, object] = {
        "schema": ARENA_ANALYSIS_SCHEMA,
        "generationId": generation_id,
        "results": results_ref.as_dict(),
        "plan": plan_ref.as_dict(),
        "execution": execution_ref.as_dict(),
        "championModelId": champion_id,
        "challengerModelId": challenger_id,
        "method": {
            "pairing": "same-start-color-swapped",
            "score": "win=1,draw-or-max-plies=0.5,loss=0",
            "wilson95": "fractional-score Wilson approximation, z=1.959963984540054",
            "elo": "400*log10(score/(1-score)); null at boundary scores",
        },
        "overall": overall,
        "byStartGroup": {"initial": initial, "start_set": start_set},
        "byChallengerSide": {"black": black, "white": white},
        "sideScoreGap": _rounded(side_gap),
        "metrics": metrics,
    }
    result["analysisSha256"] = canonical_sha256(result)
    return result


def collect_arena_results(
    *,
    repository_root: Path,
    plan: object,
    plan_ref: ArtifactRef,
    execution: object,
    execution_ref: ArtifactRef,
) -> dict[str, object]:
    """Bind successful pair reports to model IDs using plan order and model hashes."""

    from .execution import validate_execution_manifest, validate_paired_plan

    root = validate_paired_plan(plan)
    if root.get("schema") != "phase6_paired_arena_plan/v1":
        raise ContractError("arena result collection requires a paired-arena plan")
    execution_root = validate_execution_manifest(
        execution,
        repository_root=repository_root,
        manifest_ref=execution_ref,
        plan=root,
        plan_ref=plan_ref,
    )

    champion = require_mapping(root.get("champion"), "arena plan.champion")
    challenger = require_mapping(root.get("challenger"), "arena plan.challenger")
    champion_id = require_identifier(champion, "modelId", "arena plan.champion")
    challenger_id = require_identifier(challenger, "modelId", "arena plan.challenger")
    attempts_raw = require_list(execution_root, "attempts", "arena execution", maximum_items=20_000)
    latest: dict[str, Mapping[str, Any]] = {}
    quarantined_by_job: dict[str, int] = {}
    for index, raw_attempt in enumerate(attempts_raw):
        attempt = require_mapping(raw_attempt, f"arena execution.attempts[{index}]")
        job_id = require_identifier(attempt, "jobId", f"arena execution.attempts[{index}]")
        number = require_int(
            attempt,
            "attempt",
            f"arena execution.attempts[{index}]",
            minimum=1,
            maximum=100,
        )
        if attempt.get("status") == "quarantined":
            quarantined_by_job[job_id] = quarantined_by_job.get(job_id, 0) + 1
        if job_id not in latest or number > int(latest[job_id]["attempt"]):
            latest[job_id] = attempt

    games: list[dict[str, object]] = []
    jobs = require_list(root, "jobs", "arena plan", maximum_items=5_000)
    for job_index, raw_job in enumerate(jobs):
        job = require_mapping(raw_job, f"arena plan.jobs[{job_index}]")
        job_id = require_identifier(job, "jobId", f"arena plan.jobs[{job_index}]")
        attempt = latest.get(job_id)
        if attempt is None or attempt.get("status") != "completed":
            raise ContractError(f"arena job lacks a completed attempt: {job_id}")
        report_ref = ArtifactRef.from_dict(attempt.get("report"), f"attempt {job_id}.report")
        csa_values = require_list(
            attempt, "csa", f"attempt {job_id}", minimum_items=2, maximum_items=2
        )
        csa_refs = tuple(
            ArtifactRef.from_dict(value, f"attempt {job_id}.csa[{index}]")
            for index, value in enumerate(csa_values)
        )
        report = validate_phase6_pair_report_binding(
            load_json_artifact(repository_root, report_ref),
            repository_root=repository_root,
            job=job,
            git_commit=str(root["gitCommit"]),
            nodes_per_move=int(root["nodesPerMove"]),
            model_a=challenger,
            model_b=champion,
            csa_refs=csa_refs,
            context=f"report {job_id}",
        )
        illegal_total = int(report["metrics"]["illegalMoves"])
        for local_index, raw_game in enumerate(report["games"]):
            report_game = require_mapping(raw_game, f"report {job_id}.games[{local_index}]")
            if local_index == 0:
                black_id, white_id = challenger_id, champion_id
            else:
                black_id, white_id = champion_id, challenger_id
            games.append(
                {
                    "gameId": job["gameIds"][local_index],
                    "pairId": job_id,
                    "startGroup": job["startGroup"],
                    "startPositionId": job["startPositionId"],
                    "blackModelId": black_id,
                    "whiteModelId": white_id,
                    "result": report_game["result"],
                    "plies": report_game["moves"],
                    "illegalMoves": illegal_total if local_index == 0 else 0,
                    "crashes": quarantined_by_job.get(job_id, 0) if local_index == 0 else 0,
                    "metrics": _normalized_game_metrics(report_game),
                }
            )
    result: dict[str, object] = {
        "schema": ARENA_RESULTS_SCHEMA,
        "generationId": root["generationId"],
        "plan": plan_ref.as_dict(),
        "execution": execution_ref.as_dict(),
        "championModelId": champion_id,
        "challengerModelId": challenger_id,
        "games": games,
    }
    # Reuse the public strict analyzer to prove pair integrity before publication.
    analyze_arena_results(result, results_ref=execution_ref)
    return result


def _normalized_game_metrics(report_game: Mapping[str, Any]) -> dict[str, int]:
    """Map logical Rust player A/B counters to challenger/champion counters.

    Paired-arena plans always assign the challenger to logical player A and the
    champion to logical player B. Those identities remain stable when colors swap.
    """

    return {
        "championInferenceCalls": int(report_game["playerBNeuralInferenceCalls"]),
        "championInferenceTimeNs": int(report_game["playerBNeuralInferenceTimeNs"]),
        "challengerInferenceCalls": int(report_game["playerANeuralInferenceCalls"]),
        "challengerInferenceTimeNs": int(report_game["playerANeuralInferenceTimeNs"]),
        "championSearchNodes": int(report_game["playerBSearchNodes"]),
        "championSearchElapsedMs": int(report_game["playerBSearchElapsedMs"]),
        "challengerSearchNodes": int(report_game["playerASearchNodes"]),
        "challengerSearchElapsedMs": int(report_game["playerASearchElapsedMs"]),
        "championSearchDepthSum": int(report_game["playerBDepthSum"]),
        "championSearches": int(report_game["playerBSearches"]),
        "challengerSearchDepthSum": int(report_game["playerADepthSum"]),
        "challengerSearches": int(report_game["playerASearches"]),
    }


def decide_promotion(
    analysis: object,
    *,
    analysis_ref: ArtifactRef,
    policy: GenerationPolicy,
    policy_ref: ArtifactRef,
    decided_at: str,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Return promoted/rejected/inconclusive without treating weak evidence as a win."""

    root = validate_arena_analysis(analysis)
    if repository_root is not None:
        if load_json_artifact(repository_root, analysis_ref) != root:
            raise ContractError("arena-analysis value differs from its referenced artifact")
        results_ref = ArtifactRef.from_dict(root.get("results"), "arena analysis.results")
        expected = analyze_arena_results(
            load_json_artifact(repository_root, results_ref),
            results_ref=results_ref,
            repository_root=repository_root,
        )
        if root != expected:
            raise ContractError("arena analysis is not the deterministic result analysis")
    validate_utc_timestamp(decided_at, "decided_at")
    overall = require_mapping(root.get("overall"), "arena analysis.overall")
    groups = require_mapping(root.get("byStartGroup"), "arena analysis.byStartGroup")
    metrics = require_mapping(root.get("metrics"), "arena analysis.metrics")
    games = _analysis_int(overall, "games")
    decisive = _analysis_int(overall, "decisiveGames")
    illegal = _analysis_int(overall, "illegalMoves")
    crashes = _analysis_int(overall, "crashes")
    score_rate = _analysis_float(overall, "scoreRate")
    interval = require_mapping(overall.get("scoreWilson95"), "overall.scoreWilson95")
    wilson_lower = _analysis_float(interval, "lower")
    wilson_upper = _analysis_float(interval, "upper")
    side_gap = root.get("sideScoreGap")
    if side_gap is not None and not isinstance(side_gap, (int, float)):
        raise ContractError("arena analysis.sideScoreGap must be numeric or null")

    reasons: list[str] = []
    decision = "inconclusive"
    weak_evidence = False
    if illegal > policy.evidence.max_illegal:
        decision = "rejected"
        reasons.append("illegal_move_limit_exceeded")
    if crashes > policy.evidence.max_crashes:
        decision = "rejected"
        reasons.append("crash_limit_exceeded")
    if decision != "rejected" and games < policy.evidence.minimum_games:
        weak_evidence = True
        reasons.append("insufficient_games")
    if decision != "rejected" and decisive < policy.evidence.minimum_decisive_games:
        weak_evidence = True
        reasons.append("insufficient_decisive_games")

    group_rates: dict[str, float] = {}
    for group_name in ("initial", "start_set"):
        group = require_mapping(groups.get(group_name), f"byStartGroup.{group_name}")
        group_games = _analysis_int(group, "games")
        group_rates[group_name] = _analysis_float(group, "scoreRate")
        if group_games < policy.evidence.minimum_group_games:
            weak_evidence = True
            reasons.append(f"insufficient_{group_name}_games")

    performance_ok, performance_reasons, performance_missing = _check_performance(metrics, policy)
    reasons.extend(performance_reasons)
    weak_evidence = weak_evidence or performance_missing
    if decision != "rejected" and not performance_ok and not performance_missing:
        decision = "rejected"

    if decision != "rejected" and not weak_evidence:
        if (
            score_rate <= policy.thresholds.reject_score_rate
            and wilson_upper <= policy.thresholds.reject_wilson_upper
        ):
            decision = "rejected"
            reasons.append("statistically_supported_regression")
        elif (
            score_rate >= policy.thresholds.promote_score_rate
            and wilson_lower >= policy.thresholds.promote_wilson_lower
            and all(
                rate >= policy.thresholds.minimum_group_score_rate for rate in group_rates.values()
            )
            and side_gap is not None
            and float(side_gap) <= policy.thresholds.maximum_side_score_gap
            and performance_ok
        ):
            decision = "promoted"
            reasons.append("all_promotion_gates_passed")
        else:
            reasons.append("evidence_does_not_cross_a_decision_boundary")
    elif decision == "inconclusive" and not reasons:
        reasons.append("evidence_does_not_cross_a_decision_boundary")

    result: dict[str, object] = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "generationId": require_identifier(root, "generationId", "arena analysis"),
        "championModelId": require_identifier(root, "championModelId", "arena analysis"),
        "challengerModelId": require_identifier(root, "challengerModelId", "arena analysis"),
        "arenaAnalysis": analysis_ref.as_dict(),
        "policy": policy_ref.as_dict(),
        "policySha256": policy.sha256,
        "decision": decision,
        "weakEvidence": weak_evidence,
        "reasons": reasons,
        "evidence": {
            "games": games,
            "decisiveGames": decisive,
            "scoreRate": score_rate,
            "scoreWilson95": {"lower": wilson_lower, "upper": wilson_upper},
            "initialScoreRate": group_rates["initial"],
            "startSetScoreRate": group_rates["start_set"],
            "sideScoreGap": side_gap,
            "illegalMoves": illegal,
            "crashes": crashes,
        },
        "decidedAt": decided_at,
    }
    result["decisionSha256"] = canonical_sha256(result)
    validate_promotion_decision(result)
    return result


def validate_promotion_decision(raw: object) -> Mapping[str, Any]:
    root = require_mapping(raw, "promotion decision")
    expected = {
        "schema",
        "generationId",
        "championModelId",
        "challengerModelId",
        "arenaAnalysis",
        "policy",
        "policySha256",
        "decision",
        "weakEvidence",
        "reasons",
        "evidence",
        "decidedAt",
        "decisionSha256",
    }
    require_exact_keys(root, expected, "promotion decision")
    if root.get("schema") != PROMOTION_DECISION_SCHEMA:
        raise ContractError("unsupported promotion-decision schema")
    require_identifier(root, "generationId", "promotion decision")
    require_identifier(root, "championModelId", "promotion decision")
    require_identifier(root, "challengerModelId", "promotion decision")
    ArtifactRef.from_dict(root.get("arenaAnalysis"), "promotion decision.arenaAnalysis")
    ArtifactRef.from_dict(root.get("policy"), "promotion decision.policy")
    require_sha256(root, "policySha256", "promotion decision")
    require_enum(
        root,
        "decision",
        "promotion decision",
        {"promoted", "rejected", "inconclusive"},
    )
    require_bool(root, "weakEvidence", "promotion decision")
    reasons = require_list(root, "reasons", "promotion decision", minimum_items=1, maximum_items=32)
    if any(not isinstance(reason, str) or not reason or len(reason) > 128 for reason in reasons):
        raise ContractError("promotion decision reasons must be bounded strings")
    evidence = require_mapping(root.get("evidence"), "promotion decision.evidence")
    require_exact_keys(
        evidence,
        {
            "games",
            "decisiveGames",
            "scoreRate",
            "scoreWilson95",
            "initialScoreRate",
            "startSetScoreRate",
            "sideScoreGap",
            "illegalMoves",
            "crashes",
        },
        "promotion decision.evidence",
    )
    require_int(evidence, "games", "promotion decision.evidence", minimum=0, maximum=10_000)
    require_int(
        evidence,
        "decisiveGames",
        "promotion decision.evidence",
        minimum=0,
        maximum=10_000,
    )
    for key in ("scoreRate", "initialScoreRate", "startSetScoreRate"):
        require_number(evidence, key, "promotion decision.evidence", minimum=0.0, maximum=1.0)
    side_gap = evidence.get("sideScoreGap")
    if side_gap is not None:
        require_number(
            evidence,
            "sideScoreGap",
            "promotion decision.evidence",
            minimum=0.0,
            maximum=1.0,
        )
    interval = require_mapping(
        evidence.get("scoreWilson95"), "promotion decision.evidence.scoreWilson95"
    )
    require_exact_keys(interval, {"lower", "upper"}, "promotion decision.evidence.scoreWilson95")
    lower = require_number(interval, "lower", "scoreWilson95", minimum=0.0, maximum=1.0)
    upper = require_number(interval, "upper", "scoreWilson95", minimum=0.0, maximum=1.0)
    if lower > upper:
        raise ContractError("promotion decision Wilson interval is reversed")
    require_int(evidence, "illegalMoves", "promotion decision.evidence", minimum=0, maximum=10_000)
    require_int(evidence, "crashes", "promotion decision.evidence", minimum=0, maximum=10_000)
    validate_utc_timestamp(root.get("decidedAt"), "promotion decision.decidedAt")
    expected_hash = require_sha256(root, "decisionSha256", "promotion decision")
    without_hash = dict(root)
    without_hash.pop("decisionSha256")
    if canonical_sha256(without_hash) != expected_hash:
        raise ContractError("promotion decision self-hash mismatch")
    return root


def validate_promotion_decision_binding(
    raw: object,
    *,
    analysis: object,
    analysis_ref: ArtifactRef,
    policy: GenerationPolicy,
    policy_ref: ArtifactRef,
    repository_root: Path | None = None,
) -> Mapping[str, Any]:
    """Require a decision to be the exact deterministic result of its evidence and policy."""

    root = validate_promotion_decision(raw)
    if (
        ArtifactRef.from_dict(root.get("arenaAnalysis"), "promotion decision.arenaAnalysis")
        != analysis_ref
    ):
        raise ContractError("promotion decision references a different arena analysis")
    if ArtifactRef.from_dict(root.get("policy"), "promotion decision.policy") != policy_ref:
        raise ContractError("promotion decision references a different promotion policy")
    expected = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        decided_at=str(root["decidedAt"]),
        repository_root=repository_root,
    )
    if root != expected:
        raise ContractError("promotion decision is not its deterministic policy result")
    return root


def validate_arena_analysis(raw: object) -> Mapping[str, Any]:
    """Validate the complete closed analysis contract and aggregate invariants."""

    root = require_mapping(raw, "arena analysis")
    _validate_analysis_shape(root)
    require_identifier(root, "generationId", "arena analysis")
    champion = require_identifier(root, "championModelId", "arena analysis")
    challenger = require_identifier(root, "challengerModelId", "arena analysis")
    if champion == challenger:
        raise ContractError("arena analysis champion and challenger must differ")
    for key in ("results", "plan", "execution"):
        ArtifactRef.from_dict(root.get(key), f"arena analysis.{key}")
    method = require_mapping(root.get("method"), "arena analysis.method")
    require_exact_keys(method, {"pairing", "score", "wilson95", "elo"}, "arena analysis.method")
    if method != {
        "pairing": "same-start-color-swapped",
        "score": "win=1,draw-or-max-plies=0.5,loss=0",
        "wilson95": "fractional-score Wilson approximation, z=1.959963984540054",
        "elo": "400*log10(score/(1-score)); null at boundary scores",
    }:
        raise ContractError("arena analysis uses an unsupported pairing method")
    groups = require_mapping(root.get("byStartGroup"), "arena analysis.byStartGroup")
    require_exact_keys(groups, {"initial", "start_set"}, "arena analysis.byStartGroup")
    sides = require_mapping(root.get("byChallengerSide"), "arena analysis.byChallengerSide")
    require_exact_keys(sides, {"black", "white"}, "arena analysis.byChallengerSide")
    summaries = {
        "overall": require_mapping(root.get("overall"), "arena analysis.overall"),
        "initial": require_mapping(groups.get("initial"), "byStartGroup.initial"),
        "start_set": require_mapping(groups.get("start_set"), "byStartGroup.start_set"),
        "black": require_mapping(sides.get("black"), "byChallengerSide.black"),
        "white": require_mapping(sides.get("white"), "byChallengerSide.white"),
    }
    for name, summary in summaries.items():
        _validate_summary(summary, f"arena analysis.{name}")
    overall = summaries["overall"]
    for left, right in (("initial", "start_set"), ("black", "white")):
        for key in (
            "games",
            "wins",
            "losses",
            "draws",
            "maxPlies",
            "decisiveGames",
            "illegalMoves",
            "crashes",
        ):
            if int(summaries[left][key]) + int(summaries[right][key]) != int(overall[key]):
                raise ContractError(f"arena analysis {left}/{right} summaries disagree")
    side_gap = root.get("sideScoreGap")
    if side_gap is not None:
        side_gap_value = require_number(
            root, "sideScoreGap", "arena analysis", minimum=0.0, maximum=1.0
        )
        expected_gap = _rounded(
            abs(float(summaries["black"]["scoreRate"]) - float(summaries["white"]["scoreRate"]))
        )
        if side_gap_value != expected_gap:
            raise ContractError("arena analysis side-score gap disagrees with side summaries")
    metrics = require_mapping(root.get("metrics"), "arena analysis.metrics")
    require_exact_keys(
        metrics,
        {
            "championInferenceCallsPerSecond",
            "challengerInferenceCallsPerSecond",
            "inferenceSlowdownRatio",
            "championSearchNodesPerSecond",
            "challengerSearchNodesPerSecond",
            "searchSlowdownRatio",
            "championAverageSearchDepth",
            "challengerAverageSearchDepth",
            "rawTotals",
        },
        "arena analysis.metrics",
    )
    for key, value in metrics.items():
        if key != "rawTotals" and value is not None:
            require_number(metrics, key, "arena analysis.metrics", minimum=0.0)
    raw_totals = require_mapping(metrics.get("rawTotals"), "arena analysis.metrics.rawTotals")
    require_exact_keys(raw_totals, _METRIC_KEYS, "arena analysis.metrics.rawTotals")
    for key, value in raw_totals.items():
        if value is not None:
            require_int(
                raw_totals,
                key,
                "arena analysis.metrics.rawTotals",
                minimum=0,
                maximum=10**18,
            )
    return root


def _parse_game(raw: object, index: int, *, champion_id: str, challenger_id: str) -> dict[str, Any]:
    context = f"arena results.games[{index}]"
    table = require_mapping(raw, context)
    require_exact_keys(table, _GAME_KEYS, context)
    black = require_identifier(table, "blackModelId", context)
    white = require_identifier(table, "whiteModelId", context)
    if {black, white} != {champion_id, challenger_id}:
        raise ContractError(f"{context} does not contain exactly champion and challenger")
    metrics_raw = require_mapping(table.get("metrics"), f"{context}.metrics")
    require_exact_keys(metrics_raw, _METRIC_KEYS, f"{context}.metrics")
    metrics = {
        key: _optional_metric(metrics_raw, key, f"{context}.metrics") for key in _METRIC_KEYS
    }
    return {
        "gameId": require_identifier(table, "gameId", context),
        "pairId": require_identifier(table, "pairId", context),
        "startGroup": require_enum(table, "startGroup", context, {"initial", "start_set"}),
        "startPositionId": require_identifier(table, "startPositionId", context),
        "blackModelId": black,
        "whiteModelId": white,
        "result": require_enum(
            table, "result", context, {"black_win", "white_win", "draw", "max_plies"}
        ),
        "plies": require_int(table, "plies", context, minimum=0, maximum=10_000),
        "illegalMoves": require_int(table, "illegalMoves", context, minimum=0, maximum=10_000),
        "crashes": require_int(table, "crashes", context, minimum=0, maximum=10_000),
        "metrics": metrics,
    }


def _validate_pairs(games: list[dict[str, Any]], *, champion_id: str, challenger_id: str) -> None:
    game_ids: set[str] = set()
    pairs: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        if game["gameId"] in game_ids:
            raise ContractError(f"duplicate arena game ID: {game['gameId']}")
        game_ids.add(game["gameId"])
        pairs.setdefault(game["pairId"], []).append(game)
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ContractError(f"arena pair {pair_id} does not have exactly two games")
        first, second = pair
        if (
            first["startGroup"] != second["startGroup"]
            or first["startPositionId"] != second["startPositionId"]
        ):
            raise ContractError(f"arena pair {pair_id} does not share one start position")
        if not (
            first["blackModelId"] == second["whiteModelId"]
            and first["whiteModelId"] == second["blackModelId"]
        ):
            raise ContractError(f"arena pair {pair_id} is not color-swapped")
        if {first["blackModelId"], first["whiteModelId"]} != {champion_id, challenger_id}:
            raise ContractError(f"arena pair {pair_id} has unexpected model IDs")


def _summarize(games: list[dict[str, Any]], challenger_id: str) -> dict[str, object]:
    wins = 0
    losses = 0
    draws = 0
    max_plies = 0
    for game in games:
        result = game["result"]
        if result in {"draw", "max_plies"}:
            draws += 1
            max_plies += int(result == "max_plies")
            continue
        winner = game["blackModelId"] if result == "black_win" else game["whiteModelId"]
        if winner == challenger_id:
            wins += 1
        else:
            losses += 1
    count = len(games)
    points = wins + 0.5 * draws
    score_rate = points / count if count else 0.0
    decisive = wins + losses
    decisive_win_rate = wins / decisive if decisive else 0.0
    lower, upper = _wilson(points, count)
    elo = None
    if 0.0 < score_rate < 1.0:
        elo = 400.0 * math.log10(score_rate / (1.0 - score_rate))
    return {
        "games": count,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "maxPlies": max_plies,
        "decisiveGames": decisive,
        "scoreRate": _rounded(score_rate),
        "decisiveWinRate": _rounded(decisive_win_rate),
        "drawRate": _rounded(draws / count if count else 0.0),
        "scoreWilson95": {"lower": _rounded(lower), "upper": _rounded(upper)},
        "approximateElo": _rounded(elo),
        "averagePlies": _rounded(sum(game["plies"] for game in games) / count if count else 0.0),
        "illegalMoves": sum(game["illegalMoves"] for game in games),
        "crashes": sum(game["crashes"] for game in games),
    }


def _aggregate_metrics(games: list[dict[str, Any]]) -> dict[str, object]:
    totals: dict[str, int | None] = {}
    for key in _METRIC_KEYS:
        values = [game["metrics"][key] for game in games]
        totals[key] = None if any(value is None for value in values) else sum(values)
    champion_inference = _throughput_ns(
        totals["championInferenceCalls"], totals["championInferenceTimeNs"]
    )
    challenger_inference = _throughput_ns(
        totals["challengerInferenceCalls"], totals["challengerInferenceTimeNs"]
    )
    champion_search = _throughput_ms(
        totals["championSearchNodes"], totals["championSearchElapsedMs"]
    )
    challenger_search = _throughput_ms(
        totals["challengerSearchNodes"], totals["challengerSearchElapsedMs"]
    )
    return {
        "championInferenceCallsPerSecond": _rounded(champion_inference),
        "challengerInferenceCallsPerSecond": _rounded(challenger_inference),
        "inferenceSlowdownRatio": _rounded(_slowdown(champion_inference, challenger_inference)),
        "championSearchNodesPerSecond": _rounded(champion_search),
        "challengerSearchNodesPerSecond": _rounded(challenger_search),
        "searchSlowdownRatio": _rounded(_slowdown(champion_search, challenger_search)),
        "championAverageSearchDepth": _rounded(
            _ratio(totals["championSearchDepthSum"], totals["championSearches"])
        ),
        "challengerAverageSearchDepth": _rounded(
            _ratio(totals["challengerSearchDepthSum"], totals["challengerSearches"])
        ),
        "rawTotals": totals,
    }


def _check_performance(
    metrics: Mapping[str, Any], policy: GenerationPolicy
) -> tuple[bool, list[str], bool]:
    inference = metrics.get("inferenceSlowdownRatio")
    search = metrics.get("searchSlowdownRatio")
    if inference is None or search is None:
        if policy.performance.require_metrics:
            return False, ["required_performance_metrics_missing"], True
        return True, [], False
    if not isinstance(inference, (int, float)) or not isinstance(search, (int, float)):
        raise ContractError("arena analysis performance ratios must be numeric or null")
    reasons: list[str] = []
    if float(inference) > policy.performance.maximum_inference_slowdown:
        reasons.append("inference_slowdown_limit_exceeded")
    if float(search) > policy.performance.maximum_search_slowdown:
        reasons.append("search_slowdown_limit_exceeded")
    return not reasons, reasons, False


def _wilson(successes: float, count: int) -> tuple[float, float]:
    if count == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _throughput_ms(numerator: int | None, milliseconds: int | None) -> float | None:
    if numerator is None or milliseconds is None or milliseconds <= 0:
        return None
    return numerator * 1000.0 / milliseconds


def _throughput_ns(numerator: int | None, nanoseconds: int | None) -> float | None:
    if numerator is None or nanoseconds is None or nanoseconds <= 0:
        return None
    return numerator * 1_000_000_000.0 / nanoseconds


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _slowdown(baseline: float | None, challenger: float | None) -> float | None:
    if baseline is None or challenger is None or challenger <= 0:
        return None
    return baseline / challenger


def _optional_metric(table: Mapping[str, Any], key: str, context: str) -> int | None:
    if table.get(key) is None:
        return None
    return require_int(table, key, context, minimum=0, maximum=10**18)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 12)


def _validate_analysis_shape(root: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "generationId",
        "results",
        "plan",
        "execution",
        "championModelId",
        "challengerModelId",
        "method",
        "overall",
        "byStartGroup",
        "byChallengerSide",
        "sideScoreGap",
        "metrics",
        "analysisSha256",
    }
    require_exact_keys(root, expected, "arena analysis")
    if root.get("schema") != ARENA_ANALYSIS_SCHEMA:
        raise ContractError("unsupported arena-analysis schema")
    expected_hash = root.get("analysisSha256")
    if not isinstance(expected_hash, str):
        raise ContractError("arena analysis lacks analysisSha256")
    without_hash = dict(root)
    without_hash.pop("analysisSha256")
    if canonical_sha256(without_hash) != expected_hash:
        raise ContractError("arena analysis self-hash mismatch")


def _validate_summary(summary: Mapping[str, Any], context: str) -> None:
    require_exact_keys(
        summary,
        {
            "games",
            "wins",
            "losses",
            "draws",
            "maxPlies",
            "decisiveGames",
            "scoreRate",
            "decisiveWinRate",
            "drawRate",
            "scoreWilson95",
            "approximateElo",
            "averagePlies",
            "illegalMoves",
            "crashes",
        },
        context,
    )
    integers = {
        key: require_int(summary, key, context, minimum=0, maximum=10_000)
        for key in (
            "games",
            "wins",
            "losses",
            "draws",
            "maxPlies",
            "decisiveGames",
            "illegalMoves",
            "crashes",
        )
    }
    if integers["games"] != integers["wins"] + integers["losses"] + integers["draws"]:
        raise ContractError(f"{context} outcome counts disagree")
    if integers["decisiveGames"] != integers["wins"] + integers["losses"]:
        raise ContractError(f"{context} decisive-game count disagrees")
    if integers["maxPlies"] > integers["draws"]:
        raise ContractError(f"{context} max-plies count exceeds draws")
    for key in ("scoreRate", "decisiveWinRate", "drawRate"):
        require_number(summary, key, context, minimum=0.0, maximum=1.0)
    expected_score = _rounded(
        (integers["wins"] + 0.5 * integers["draws"]) / integers["games"]
        if integers["games"]
        else 0.0
    )
    expected_decisive = _rounded(
        integers["wins"] / integers["decisiveGames"] if integers["decisiveGames"] else 0.0
    )
    expected_draw = _rounded(integers["draws"] / integers["games"] if integers["games"] else 0.0)
    if (
        summary["scoreRate"] != expected_score
        or summary["decisiveWinRate"] != expected_decisive
        or summary["drawRate"] != expected_draw
    ):
        raise ContractError(f"{context} rates disagree with outcome counts")
    interval = require_mapping(summary.get("scoreWilson95"), f"{context}.scoreWilson95")
    require_exact_keys(interval, {"lower", "upper"}, f"{context}.scoreWilson95")
    lower = require_number(interval, "lower", context, minimum=0.0, maximum=1.0)
    upper = require_number(interval, "upper", context, minimum=0.0, maximum=1.0)
    if lower > upper:
        raise ContractError(f"{context} Wilson interval is reversed")
    elo = summary.get("approximateElo")
    if elo is not None:
        require_number(summary, "approximateElo", context)
    require_number(summary, "averagePlies", context, minimum=0.0, maximum=10_000.0)


def _analysis_int(table: Mapping[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"arena analysis {key} must be a non-negative integer")
    return value


def _analysis_float(table: Mapping[str, Any], key: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(f"arena analysis {key} must be finite")
    return float(value)


def validate_phase2_pair_report(raw: object, *, context: str) -> Mapping[str, Any]:
    """Validate the complete Rust ``phase2_arena_report/v2`` pair contract."""

    root = require_mapping(raw, context)
    require_exact_keys(root, {"schema", "run", "metrics", "games"}, context)
    if root.get("schema") != "phase2_arena_report/v2":
        raise ContractError(f"{context} has an unsupported schema")
    run = require_mapping(root.get("run"), f"{context}.run")
    require_exact_keys(run, _REPORT_RUN_KEYS, f"{context}.run")
    require_int(run, "seed", f"{context}.run", minimum=0, maximum=MAX_JSON_SAFE_INTEGER)
    if require_int(run, "gameLimit", f"{context}.run", minimum=1, maximum=10_000) != 2:
        raise ContractError(f"{context} must contain exactly one two-game pair")
    engine_identity = require_string(run, "engine", f"{context}.run", maximum_length=4_096)
    git_commit = run.get("gitCommit")
    if git_commit is not None:
        value = require_string(run, "gitCommit", f"{context}.run", maximum_length=64)
        if not 7 <= len(value) <= 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ContractError(f"{context}.run.gitCommit is not a lowercase hexadecimal object ID")
    started_at = validate_utc_timestamp(run.get("startedAt"), f"{context}.run.startedAt")
    if run.get("completedAt") is not None:
        completed_at = validate_utc_timestamp(run.get("completedAt"), f"{context}.run.completedAt")
        if _parse_timestamp(completed_at) < _parse_timestamp(started_at):
            raise ContractError(f"{context}.run completion precedes its start")
    initial_sfen = require_string(run, "initialSfen", f"{context}.run", maximum_length=512)
    _validate_report_sfen(initial_sfen, f"{context}.run.initialSfen")
    max_plies = require_int(run, "maxPlies", f"{context}.run", minimum=1, maximum=10_000)
    config_sha256 = require_sha256(run, "configSha256", f"{context}.run")
    budget = require_mapping(run.get("budget"), f"{context}.run.budget")
    require_exact_keys(budget, _REPORT_BUDGET_KEYS, f"{context}.run.budget")
    budget_kind = require_enum(
        budget,
        "kind",
        f"{context}.run.budget",
        {"nodes", "movetime_ms"},
    )
    require_int(
        budget,
        "value",
        f"{context}.run.budget",
        minimum=1,
        maximum=1_000_000_000 if budget_kind == "nodes" else 3_600_000,
    )
    player_a = _validate_report_player(run.get("playerA"), f"{context}.run.playerA")
    player_b = _validate_report_player(run.get("playerB"), f"{context}.run.playerB")
    opening = _validate_report_opening(run.get("opening"), f"{context}.run.opening")
    opening_enabled = bool(opening["enabled"])
    if opening_enabled != (bool(player_a["openingEnabled"]) or bool(player_b["openingEnabled"])):
        raise ContractError(f"{context} opening identity disagrees with its players")
    if hashlib.sha256(arena_config_signature_bytes(run)).hexdigest() != config_sha256:
        raise ContractError(f"{context}.run.configSha256 disagrees with its configuration")
    budget_name = "Nodes" if budget_kind == "nodes" else "MoveTime"
    expected_engine_suffix = (
        f" a={player_a['label']} b={player_b['label']} budget={budget_name}({budget['value']})"
    )
    if (
        any(character in engine_identity for character in "\r\n\x00")
        or not engine_identity.endswith(expected_engine_suffix)
        or not engine_identity.removesuffix(expected_engine_suffix)
    ):
        raise ContractError(f"{context}.run.engine identity disagrees with its run")

    games_raw = require_list(root, "games", context, minimum_items=2, maximum_items=2)
    games: list[Mapping[str, Any]] = []
    for index, raw_game in enumerate(games_raw):
        game_context = f"{context}.games[{index}]"
        game = require_mapping(raw_game, game_context)
        require_exact_keys(game, _REPORT_GAME_KEYS, game_context)
        if require_int(game, "id", game_context, minimum=0, maximum=1) != index:
            raise ContractError(f"{context} game IDs are not contiguous")
        expected_black = player_a["label"] if index % 2 == 0 else player_b["label"]
        expected_white = player_b["label"] if index % 2 == 0 else player_a["label"]
        if (
            require_string(game, "black", game_context, maximum_length=512) != expected_black
            or require_string(game, "white", game_context, maximum_length=512) != expected_white
        ):
            raise ContractError(f"{game_context} player labels violate the A/B color swap")
        require_enum(
            game,
            "result",
            game_context,
            {"black_win", "white_win", "draw", "max_plies"},
        )
        require_int(game, "moves", game_context, minimum=0, maximum=max_plies)
        expected_csa_path = f"games/game-{index + 1:06d}.csa"
        if require_string(game, "csaPath", game_context, maximum_length=1_024) != expected_csa_path:
            raise ContractError(f"{game_context}.csaPath is not deterministic")
        require_sha256(game, "csaSha256", game_context)
        require_int(game, "csaSize", game_context, minimum=1, maximum=MAX_JSON_SAFE_INTEGER)
        for key in _REPORT_COUNTER_KEYS:
            require_int(game, key, game_context, minimum=0, maximum=MAX_JSON_SAFE_INTEGER)
        if int(game["neuralInferenceCalls"]) != _safe_sum(
            int(game["playerANeuralInferenceCalls"]),
            int(game["playerBNeuralInferenceCalls"]),
        ) or int(game["neuralInferenceTimeNs"]) != _safe_sum(
            int(game["playerANeuralInferenceTimeNs"]),
            int(game["playerBNeuralInferenceTimeNs"]),
        ):
            raise ContractError(f"{game_context} combined inference counters disagree")
        _validate_player_game_counters(game, player_a, prefix="playerA", context=game_context)
        _validate_player_game_counters(game, player_b, prefix="playerB", context=game_context)
        games.append(game)

    metrics = require_mapping(root.get("metrics"), f"{context}.metrics")
    require_exact_keys(metrics, _REPORT_METRIC_KEYS, f"{context}.metrics")
    if require_int(metrics, "games", f"{context}.metrics", minimum=0, maximum=2) != len(games):
        raise ContractError(f"{context} metrics game count disagrees")
    for key in (
        "finishedGames",
        "playerAWins",
        "playerBWins",
        "searchWins",
        "draws",
        "illegalMoves",
        *_REPORT_COUNTER_KEYS,
    ):
        require_int(metrics, key, f"{context}.metrics", minimum=0, maximum=MAX_JSON_SAFE_INTEGER)
    for key in ("nodesPerSecond", "averageDepth", "millisecondsPerMove"):
        require_number(metrics, key, f"{context}.metrics", minimum=0.0)
    for key in ("ttHitRate", "cutoffRate", "pruningRate"):
        require_number(metrics, key, f"{context}.metrics", minimum=0.0, maximum=1.0)
    if metrics.get("peakMemoryBytes") is not None:
        require_int(
            metrics,
            "peakMemoryBytes",
            f"{context}.metrics",
            minimum=0,
            maximum=MAX_JSON_SAFE_INTEGER,
        )
    _validate_report_aggregates(
        metrics,
        games,
        player_a=player_a,
        player_b=player_b,
        context=context,
    )
    return root


def arena_config_signature_bytes(run: Mapping[str, Any]) -> bytes:
    """Mirror the public Rust phase2 config-signature byte contract exactly."""

    def player(value: object) -> dict[str, object]:
        table = require_mapping(value, "arena config signature player")
        return {
            "label": table.get("label"),
            "evaluatorKind": table.get("evaluatorKind"),
            "searchDepth": table.get("searchDepth"),
            "hashMegabytes": table.get("hashMegabytes"),
            "transposition": table.get("transposition"),
            "modelArtifactSha256": table.get("modelArtifactSha256"),
            "modelArtifactSize": table.get("modelArtifactSize"),
            "modelPayloadSha256": table.get("modelPayloadSha256"),
            "architectureVersion": table.get("architectureVersion"),
            "quantization": table.get("quantization"),
            "openingEnabled": table.get("openingEnabled"),
        }

    budget = require_mapping(run.get("budget"), "arena config signature budget")
    opening = require_mapping(run.get("opening"), "arena config signature opening")
    signature = {
        "schema": "phase2_arena_config_signature/v1",
        "games": run.get("gameLimit"),
        "seed": run.get("seed"),
        "initialSfen": run.get("initialSfen"),
        "maxPlies": run.get("maxPlies"),
        "gitCommit": run.get("gitCommit"),
        "budget": {"kind": budget.get("kind"), "value": budget.get("value")},
        "playerA": player(run.get("playerA")),
        "playerB": player(run.get("playerB")),
        "opening": {
            "enabled": opening.get("enabled"),
            "artifactSha256": opening.get("artifactSha256"),
            "artifactSize": opening.get("artifactSize"),
            "maxPlies": opening.get("maxPlies"),
        },
    }
    try:
        return json.dumps(
            signature,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError(f"arena config signature is not canonical JSON: {error}") from error


def validate_phase6_pair_report_binding(
    report: object,
    *,
    repository_root: Path,
    job: Mapping[str, Any],
    git_commit: str,
    nodes_per_move: int,
    model_a: Mapping[str, Any],
    model_b: Mapping[str, Any],
    csa_refs: tuple[ArtifactRef, ArtifactRef],
    context: str,
) -> Mapping[str, Any]:
    """Bind a valid Rust report to one immutable Phase 6 plan job and its files."""

    root = validate_phase2_pair_report(report, context=context)
    run = require_mapping(root.get("run"), f"{context}.run")
    if run.get("completedAt") is None:
        raise ContractError(f"{context} is not a completed Rust report")
    expected_run_values = {
        "seed": job.get("seed"),
        "gameLimit": 2,
        "gitCommit": git_commit,
        "initialSfen": job.get("sfen"),
        "maxPlies": PHASE6_MAX_PLIES,
    }
    for key, expected in expected_run_values.items():
        if run.get(key) != expected:
            raise ContractError(f"{context}.run.{key} differs from its plan job")
    if run.get("budget") != {"kind": "nodes", "value": nodes_per_move}:
        raise ContractError(f"{context}.run.budget differs from its plan")
    if nodes_per_move != PHASE6_NODES_PER_MOVE:
        raise ContractError(f"{context} does not use the Phase 6 node budget")
    if run.get("opening") != {
        "enabled": False,
        "artifactSha256": None,
        "artifactSize": None,
        "maxPlies": None,
    }:
        raise ContractError(f"{context} unexpectedly enables an opening artifact")

    command = require_mapping(job.get("command"), f"{context}.job.command")
    argv_raw = require_list(
        command,
        "argv",
        f"{context}.job.command",
        minimum_items=2,
        maximum_items=256,
    )
    argv = [str(value) for value in argv_raw]
    options = _report_command_options(argv, f"{context}.job.command")
    model_a_artifact = ArtifactRef.from_dict(model_a.get("artifact"), f"{context}.modelA")
    model_b_artifact = ArtifactRef.from_dict(model_b.get("artifact"), f"{context}.modelB")
    expected_command_options = {
        "--games": "2",
        "--player-a": "neural",
        "--player-b": "neural",
        "--a-depth": options["--a-depth"],
        "--b-depth": options["--b-depth"],
        "--a-hash-mb": options["--a-hash-mb"],
        "--b-hash-mb": options["--b-hash-mb"],
        "--nodes": str(nodes_per_move),
        "--max-plies": str(PHASE6_MAX_PLIES),
        "--seed": str(job["seed"]),
        "--sfen": str(job["sfen"]),
        "--git-commit": git_commit,
        "--output-dir": str(job["outputDir"]),
        "--a-model": model_a_artifact.path,
        "--b-model": model_b_artifact.path,
    }
    if options != expected_command_options:
        raise ContractError(f"{context} job argv differs from its report binding")
    expected_players = (
        _expected_report_player(
            repository_root,
            model_a,
            depth=int(options["--a-depth"]),
            hash_megabytes=int(options["--a-hash-mb"]),
            context=f"{context}.playerA",
        ),
        _expected_report_player(
            repository_root,
            model_b,
            depth=int(options["--b-depth"]),
            hash_megabytes=int(options["--b-hash-mb"]),
            context=f"{context}.playerB",
        ),
    )
    if run.get("playerA") != expected_players[0] or run.get("playerB") != expected_players[1]:
        raise ContractError(f"{context} player identity differs from immutable model artifacts")

    expected_paths = job.get("csaPaths")
    if not isinstance(expected_paths, list) or len(expected_paths) != 2:
        raise ContractError(f"{context} plan job lacks two CSA paths")
    output_dir = PurePosixPath(str(job.get("outputDir")))
    games = require_list(root, "games", context, minimum_items=2, maximum_items=2)
    for index, (reference, expected_path, raw_game) in enumerate(
        zip(csa_refs, expected_paths, games, strict=True)
    ):
        if reference.path != expected_path:
            raise ContractError(f"{context} CSA artifact path differs from its plan")
        verify_artifact_ref(repository_root, reference)
        try:
            relative = PurePosixPath(reference.path).relative_to(output_dir).as_posix()
        except ValueError as error:
            raise ContractError(f"{context} CSA path is outside its job output") from error
        game = require_mapping(raw_game, f"{context}.games[{index}]")
        if (
            game.get("csaPath") != relative
            or game.get("csaSha256") != reference.sha256
            or game.get("csaSize") != reference.size
        ):
            raise ContractError(f"{context}.games[{index}] CSA identity differs from its artifact")
    return root


def _validate_report_player(raw: object, context: str) -> Mapping[str, Any]:
    player = require_mapping(raw, context)
    require_exact_keys(player, _REPORT_PLAYER_KEYS, context)
    evaluator = require_enum(
        player,
        "evaluatorKind",
        context,
        {"random", "material", "handcrafted-baseline", "handcrafted-experimental", "neural"},
    )
    label = require_string(player, "label", context, maximum_length=512)
    opening_enabled = require_bool(player, "openingEnabled", context)
    model_keys = (
        "modelArtifactSha256",
        "modelArtifactSize",
        "modelPayloadSha256",
        "architectureVersion",
        "quantization",
    )
    if evaluator == "random":
        if any(
            player.get(key) is not None for key in ("searchDepth", "hashMegabytes", "transposition")
        ):
            raise ContractError(f"{context} random evaluator has search options")
        if label != "random":
            raise ContractError(f"{context} random evaluator label is not canonical")
    else:
        depth = require_int(player, "searchDepth", context, minimum=1, maximum=64)
        hash_megabytes = require_int(player, "hashMegabytes", context, minimum=1, maximum=1_024)
        transposition = require_bool(player, "transposition", context)
        model_suffix = ""
        if evaluator == "neural":
            artifact_sha = require_sha256(player, "modelArtifactSha256", context)
            require_int(
                player,
                "modelArtifactSize",
                context,
                minimum=1,
                maximum=MAX_MODEL_BYTES,
            )
            require_sha256(player, "modelPayloadSha256", context)
            architecture = require_int(player, "architectureVersion", context, minimum=1, maximum=1)
            if architecture != ARCH_VERSION:
                raise ContractError(f"{context} architecture version is unsupported")
            require_enum(player, "quantization", context, {"float32", "int8"})
            model_suffix = f":m-{artifact_sha[:12]}"
        elif any(player.get(key) is not None for key in model_keys):
            raise ContractError(f"{context} non-neural evaluator has model identity fields")
        expected_label = (
            f"search:{evaluator}:d{depth}:h{hash_megabytes}:"
            f"tt-{'on' if transposition else 'off'}:"
            f"book-{'on' if opening_enabled else 'off'}{model_suffix}"
        )
        if label != expected_label:
            raise ContractError(f"{context} evaluator label is not canonical")
    if evaluator != "neural" and any(player.get(key) is not None for key in model_keys):
        raise ContractError(f"{context} non-neural evaluator has model identity fields")
    return player


def _validate_report_opening(raw: object, context: str) -> Mapping[str, Any]:
    opening = require_mapping(raw, context)
    require_exact_keys(opening, _REPORT_OPENING_KEYS, context)
    enabled = require_bool(opening, "enabled", context)
    if enabled:
        require_sha256(opening, "artifactSha256", context)
        require_int(opening, "artifactSize", context, minimum=1, maximum=MAX_JSON_SAFE_INTEGER)
        require_int(opening, "maxPlies", context, minimum=1, maximum=10_000)
    elif any(
        opening.get(key) is not None for key in ("artifactSha256", "artifactSize", "maxPlies")
    ):
        raise ContractError(f"{context} disabled opening has artifact fields")
    return opening


def _validate_player_game_counters(
    game: Mapping[str, Any],
    player: Mapping[str, Any],
    *,
    prefix: str,
    context: str,
) -> None:
    if player["evaluatorKind"] == "random" and any(
        int(game[f"{prefix}{suffix}"]) != 0
        for suffix in ("SearchNodes", "SearchElapsedMs", "DepthSum", "Searches")
    ):
        raise ContractError(f"{context} random evaluator has search counters")
    if player["evaluatorKind"] != "neural" and any(
        int(game[f"{prefix}{suffix}"]) != 0
        for suffix in ("NeuralInferenceCalls", "NeuralInferenceTimeNs")
    ):
        raise ContractError(f"{context} non-neural evaluator has inference counters")


def _validate_report_aggregates(
    metrics: Mapping[str, Any],
    games: list[Mapping[str, Any]],
    *,
    player_a: Mapping[str, Any],
    player_b: Mapping[str, Any],
    context: str,
) -> None:
    for key in _REPORT_COUNTER_KEYS:
        expected = _safe_sum(*(int(game[key]) for game in games))
        if int(metrics[key]) != expected:
            raise ContractError(f"{context}.metrics.{key} disagrees with per-game counters")
    if int(metrics["neuralInferenceCalls"]) != _safe_sum(
        int(metrics["playerANeuralInferenceCalls"]),
        int(metrics["playerBNeuralInferenceCalls"]),
    ) or int(metrics["neuralInferenceTimeNs"]) != _safe_sum(
        int(metrics["playerANeuralInferenceTimeNs"]),
        int(metrics["playerBNeuralInferenceTimeNs"]),
    ):
        raise ContractError(f"{context}.metrics combined inference counters disagree")
    results = [str(game["result"]) for game in games]
    expected_finished = sum(result != "max_plies" for result in results)
    expected_draws = results.count("draw")
    player_a_wins = 0
    player_b_wins = 0
    search_wins = 0
    for index, result in enumerate(results):
        winner: str | None = None
        if result == "black_win":
            winner = "a" if index % 2 == 0 else "b"
        elif result == "white_win":
            winner = "b" if index % 2 == 0 else "a"
        if winner == "a":
            player_a_wins += 1
            search_wins += player_a["evaluatorKind"] != "random"
        elif winner == "b":
            player_b_wins += 1
            search_wins += player_b["evaluatorKind"] != "random"
    expected_counts = {
        "finishedGames": expected_finished,
        "playerAWins": player_a_wins,
        "playerBWins": player_b_wins,
        "searchWins": search_wins,
        "draws": expected_draws,
    }
    for key, expected in expected_counts.items():
        if int(metrics[key]) != expected:
            raise ContractError(f"{context}.metrics.{key} disagrees with game results")
    search_nodes = _safe_sum(int(metrics["playerASearchNodes"]), int(metrics["playerBSearchNodes"]))
    search_elapsed_ms = _safe_sum(
        int(metrics["playerASearchElapsedMs"]), int(metrics["playerBSearchElapsedMs"])
    )
    depth_sum = _safe_sum(int(metrics["playerADepthSum"]), int(metrics["playerBDepthSum"]))
    searches = _safe_sum(int(metrics["playerASearches"]), int(metrics["playerBSearches"]))
    expected_ratios = {
        "nodesPerSecond": _rust_decimal_ratio(search_nodes, search_elapsed_ms, multiplier=1_000),
        "averageDepth": _rust_decimal_ratio(depth_sum, searches),
        "millisecondsPerMove": _rust_decimal_ratio(search_elapsed_ms, searches),
    }
    for key, expected in expected_ratios.items():
        if float(metrics[key]) != expected:
            raise ContractError(f"{context}.metrics.{key} disagrees with raw counters")


def _safe_sum(*values: int) -> int:
    total = sum(values)
    if total > MAX_JSON_SAFE_INTEGER:
        raise ContractError("arena report counter sum exceeds the JSON-safe integer limit")
    return total


def _rust_decimal_ratio(numerator: int, denominator: int, *, multiplier: int = 1) -> float:
    if denominator == 0:
        return 0.0
    scaled = numerator * multiplier * 1_000_000 // denominator
    return float(f"{scaled // 1_000_000}.{scaled % 1_000_000:06d}")


def _validate_report_sfen(value: str, context: str) -> None:
    fields = value.split(" ")
    if (
        not value.isascii()
        or len(fields) != 4
        or any(not field for field in fields)
        or fields[1] not in {"b", "w"}
        or not fields[3].isdecimal()
        or int(fields[3]) < 1
        or any(character in value for character in "\r\n")
    ):
        raise ContractError(f"{context} is not one canonical four-field SFEN")


def _report_command_options(argv: list[str], context: str) -> dict[str, str]:
    if len(argv) < 2 or argv[1] != "arena" or (len(argv) - 2) % 2 != 0:
        raise ContractError(f"{context} is not one closed arena argv")
    required = {
        "--games",
        "--player-a",
        "--player-b",
        "--a-depth",
        "--b-depth",
        "--a-hash-mb",
        "--b-hash-mb",
        "--nodes",
        "--max-plies",
        "--seed",
        "--sfen",
        "--git-commit",
        "--output-dir",
        "--a-model",
        "--b-model",
    }
    values: dict[str, str] = {}
    for index in range(2, len(argv), 2):
        option = argv[index]
        if option not in required or option in values or index + 1 >= len(argv):
            raise ContractError(f"{context} contains an unknown, duplicate, or valueless option")
        values[option] = argv[index + 1]
    if set(values) != required:
        raise ContractError(f"{context} does not contain the exact Phase 6 arena options")
    for option, minimum, maximum in (
        ("--a-depth", 1, 64),
        ("--b-depth", 1, 64),
        ("--a-hash-mb", 1, 1_024),
        ("--b-hash-mb", 1, 1_024),
    ):
        value = values[option]
        if not value.isascii() or not value.isdecimal() or not minimum <= int(value) <= maximum:
            raise ContractError(f"{context} lacks a bounded {option} value")
    return values


def _parse_timestamp(value: str) -> datetime:
    """Parse a timestamp already accepted by ``validate_utc_timestamp``."""

    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _expected_report_player(
    repository_root: Path,
    model: Mapping[str, Any],
    *,
    depth: int,
    hash_megabytes: int,
    context: str,
) -> dict[str, object]:
    evaluator = require_enum(
        model,
        "evaluatorKind",
        context,
        {"material", "handcrafted-baseline", "handcrafted-experimental", "neural"},
    )
    if evaluator != "neural":
        raise ContractError(f"{context} Phase 6 model must use the neural evaluator")
    artifact = ArtifactRef.from_dict(model.get("artifact"), f"{context}.artifact")
    try:
        parsed = parse_value_model(
            load_bytes_artifact(repository_root, artifact, maximum_bytes=64 * 1024 * 1024)
        )
    except (OSError, ValueError) as error:
        raise ContractError(f"{context} model artifact is not valid OSAVAL01: {error}") from error
    quantization = {
        QUANTIZATION_FLOAT32: "float32",
        QUANTIZATION_INT8: "int8",
    }.get(parsed.quantization)
    if quantization is None:
        raise ContractError(f"{context} model artifact uses an unsupported quantization")
    return {
        "label": (
            f"search:neural:d{depth}:h{hash_megabytes}:tt-on:book-off:m-{artifact.sha256[:12]}"
        ),
        "evaluatorKind": "neural",
        "searchDepth": depth,
        "hashMegabytes": hash_megabytes,
        "transposition": True,
        "modelArtifactSha256": artifact.sha256,
        "modelArtifactSize": artifact.size,
        "modelPayloadSha256": parsed.payload_sha256,
        "architectureVersion": ARCH_VERSION,
        "quantization": quantization,
        "openingEnabled": False,
    }
