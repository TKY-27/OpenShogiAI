"""Command line entry point for deterministic Phase 6 orchestration artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .arena import (
    analyze_arena_results,
    collect_arena_results,
    decide_promotion,
    validate_arena_analysis,
    validate_promotion_decision,
    validate_promotion_decision_binding,
)
from .common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    contained_path,
    load_json_and_ref,
    validate_relative_path,
    write_json_new,
)
from .config import load_generation_policy, load_selfplay_config
from .derivation import build_position_evidence
from .engine_receipt import DEFAULT_ENGINE_PATH, resolve_active_engine_build
from .evidence import (
    MAX_EVIDENCE_JSON_NODES,
    build_replay_buffer_manifest,
    build_replay_candidates,
    extract_hard_positions,
)
from .execution import CommandRunner, execute_paired_plan
from .planning import (
    ModelSpec,
    build_arena_plan,
    build_selfplay_plan,
    build_teacher_labeling_plan,
    build_training_plan,
    parse_start_positions,
    validate_start_position_validation,
)
from .registry import (
    ModelRegistryStore,
    build_initial_registry,
    record_generation_promotion,
    register_challenger_generation,
    registry_snapshot_ref,
    validate_model_registry,
)
from .starts import (
    build_start_positions_from_phase3,
    execute_start_position_validation,
    validate_phase3_dataset_binding,
)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    root = Path(options.project_root).resolve(strict=True)
    try:
        payload = _dispatch(options, root)
    except (ContractError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _dispatch(options: argparse.Namespace, root: Path) -> object:
    command = options.command
    if command == "validate-config":
        config = load_selfplay_config(_path(root, options.selfplay_config, must_exist=True))
        policy = load_generation_policy(_path(root, options.promotion_policy, must_exist=True))
        return {
            "schema": "phase6_config_validation/v1",
            "selfplayConfigSha256": config.sha256,
            "promotionPolicySha256": policy.sha256,
            "games": config.run.games,
            "workers": config.resources.workers,
            "memoryLimitMiB": config.resources.memory_limit_mib,
        }
    if command == "build-start-positions":
        result = build_start_positions_from_phase3(
            repository_root=root,
            positions_ref=_ref(root, options.positions),
            dataset_manifest_ref=_ref(root, options.dataset_manifest),
            seed=options.seed,
            train_count=options.train_count,
            validation_count=options.validation_count,
        )
        return _publish(root, options.output, result).as_dict()
    if command == "validate-start-positions":
        config, _ = _config_and_ref(root, options.selfplay_config)
        starts_ref, starts = _json_and_ref(root, options.start_positions)
        if config.paths.engine_cli != DEFAULT_ENGINE_PATH:
            raise ContractError("self-play config engineCli must remain the operator alias")
        engine_ref, receipt_ref, _ = resolve_active_engine_build(
            root, expected_git_commit=options.git_commit
        )
        result = execute_start_position_validation(
            repository_root=root,
            start_positions=starts,
            start_positions_ref=starts_ref,
            engine_ref=engine_ref,
            git_commit=options.git_commit,
            output_root=validate_relative_path(options.log_root),
            timeout_seconds=options.timeout_seconds,
            runner=CommandRunner(root, require_clean_repository=True),
            engine_build_receipt=receipt_ref,
        )
        validate_start_position_validation(
            result,
            start_positions_ref=starts_ref,
            engine_ref=engine_ref,
            positions=parse_start_positions(starts),
            repository_root=root,
            engine_build_receipt=receipt_ref,
            runtime_authorization=True,
        )
        return _publish(root, options.output, result).as_dict()
    if command in {"plan-selfplay", "plan-arena"}:
        return _plan_paired(options, root, arena=command == "plan-arena")
    if command == "extract-hard":
        config, _ = _config_and_ref(root, options.selfplay_config)
        if options.labels_before != config.hard_positions.teacher_label_limit:
            raise ContractError(
                "Phase 4 already consumed the 10000-label cap; --labels-before must be 10000"
            )
        evidence_ref, evidence = _json_and_ref(
            root,
            options.evidence,
            maximum_nodes=MAX_EVIDENCE_JSON_NODES,
        )
        result = extract_hard_positions(
            evidence,
            evidence_ref=evidence_ref,
            config=config,
            labels_before=options.labels_before,
            requested_max=options.requested_max,
            repository_root=root,
        )
        return _publish(root, options.output, result).as_dict()
    if command == "build-evidence":
        plan_ref, plan = _json_and_ref(root, options.selfplay_plan)
        manifest_ref, manifest = _json_and_ref(root, options.selfplay_manifest)
        result = build_position_evidence(
            repository_root=root,
            selfplay_plan=plan,
            selfplay_plan_ref=plan_ref,
            selfplay_manifest=manifest,
            selfplay_manifest_ref=manifest_ref,
            teacher_labels_ref=_ref(root, options.teacher_labels),
            model_predictions_ref=_ref(root, options.model_predictions),
            dataset_manifest_ref=_ref(root, options.dataset_manifest),
            generation_ordinal=options.generation_ordinal,
            teacher_source_generation_id=options.teacher_source_generation_id,
            export_root=validate_relative_path(options.export_root),
            timeout_seconds=options.timeout_seconds,
            runner=CommandRunner(root, require_clean_repository=True),
        )
        return _publish(root, options.output, result).as_dict()
    if command == "build-replay":
        config, config_ref = _config_and_ref(root, options.selfplay_config)
        candidates_ref, candidates = _json_and_ref(root, options.candidates)
        result = build_replay_buffer_manifest(
            candidates,
            candidates_ref=candidates_ref,
            config=config,
            config_ref=config_ref,
            repository_root=root,
        )
        return _publish(root, options.output, result).as_dict()
    if command == "build-replay-candidates":
        config, _ = _config_and_ref(root, options.selfplay_config)
        evidence_ref, evidence = _json_and_ref(
            root,
            options.evidence,
            maximum_nodes=MAX_EVIDENCE_JSON_NODES,
        )
        hard_ref, hard = _json_and_ref(root, options.hard_positions)
        result = build_replay_candidates(
            evidence,
            evidence_ref=evidence_ref,
            hard_positions=hard,
            hard_positions_ref=hard_ref,
            config=config,
            repository_root=root,
        )
        return _publish(root, options.output, result).as_dict()
    if command == "analyze-arena":
        results_ref, results = _json_and_ref(root, options.results)
        analysis = analyze_arena_results(results, results_ref=results_ref, repository_root=root)
        return _publish(root, options.output, analysis).as_dict()
    if command == "collect-arena":
        plan_ref, plan = _json_and_ref(root, options.plan)
        execution_ref, execution = _json_and_ref(root, options.execution)
        results = collect_arena_results(
            repository_root=root,
            plan=plan,
            plan_ref=plan_ref,
            execution=execution,
            execution_ref=execution_ref,
        )
        return _publish(root, options.output, results).as_dict()
    if command == "decide-promotion":
        analysis_ref, analysis = _json_and_ref(root, options.analysis)
        policy_ref = _ref(root, options.policy)
        policy = load_generation_policy(_path(root, policy_ref.path, must_exist=True))
        decision = decide_promotion(
            analysis,
            analysis_ref=analysis_ref,
            policy=policy,
            policy_ref=policy_ref,
            decided_at=options.decided_at,
            repository_root=root,
        )
        return _publish(root, options.output, decision).as_dict()
    if command == "plan-training":
        parent = _registry_model(root, options.registry, options.parent_model)
        positions_ref = _ref(root, options.positions)
        dataset_manifest_ref, dataset_manifest = _json_and_ref(root, options.dataset_manifest)
        validate_phase3_dataset_binding(
            dataset_manifest,
            positions_ref=positions_ref,
            dataset_manifest_ref=dataset_manifest_ref,
        )
        plan = build_training_plan(
            generation_id=options.generation_id,
            parent_model=parent,
            replay_manifest=_ref(root, options.replay_manifest),
            labels=_ref(root, options.labels),
            label_manifest=_ref(root, options.label_manifest),
            positions=positions_ref,
            dataset_manifest=dataset_manifest_ref,
            features_config=_ref(root, options.features_config),
            model_config=_ref(root, options.model_config),
            training_config=_ref(root, options.training_config),
            output_dir=validate_relative_path(options.output_dir),
            timeout_seconds=options.timeout_seconds,
            resume_checkpoint=_ref(root, options.resume_checkpoint)
            if options.resume_checkpoint
            else None,
        )
        return _publish(root, options.output, plan).as_dict()
    if command == "plan-teacher":
        config, _ = _config_and_ref(root, options.selfplay_config)
        if options.labels_before != config.hard_positions.teacher_label_limit:
            raise ContractError(
                "Phase 4 already consumed the 10000-label cap; no new teacher plan is permitted"
            )
        plan = build_teacher_labeling_plan(
            generation_id=options.generation_id,
            hard_positions=_ref(root, options.hard_positions),
            normalized_positions=_ref(root, options.normalized_positions),
            dataset_manifest=_ref(root, options.dataset_manifest),
            labeling_config=_ref(root, options.labeling_config),
            benchmark_report=_ref(root, options.benchmark_report),
            output_dir=validate_relative_path(options.output_dir),
            target_completed=options.target_completed,
            labels_before=options.labels_before,
            teacher_label_limit=config.hard_positions.teacher_label_limit,
            timeout_seconds=options.timeout_seconds,
        )
        return _publish(root, options.output, plan).as_dict()
    if command == "execute-paired":
        plan_ref, plan = _json_and_ref(root, options.plan)
        manifest = execute_paired_plan(
            repository_root=root,
            plan=plan,
            plan_ref=plan_ref,
            state_path=validate_relative_path(options.state),
            manifest_path=validate_relative_path(options.manifest),
            runner=CommandRunner(root, require_clean_repository=True),
            retry_quarantined=options.retry_quarantined,
        )
        return {
            "status": manifest["status"],
            "jobsCompleted": manifest["jobsCompleted"],
            "jobsQuarantined": manifest["jobsQuarantined"],
            "manifest": _ref(root, options.manifest).as_dict()
            if manifest["status"] == "completed"
            else None,
        }
    if command == "init-registry":
        training = _ref(root, options.training_run) if options.training_run else None
        registry = build_initial_registry(
            generation_id=options.generation_id,
            champion_model_id=options.model_id,
            champion_artifact=_ref(root, options.model_artifact),
            evaluator_kind="neural",
            architecture_version=options.architecture_version,
            quantization=options.quantization,
            training_run=training,
            registered_at=options.registered_at,
        )
        relative = validate_relative_path(options.output)
        ModelRegistryStore(root, relative).create(registry)
        return _ref(root, relative).as_dict()
    if command == "register-challenger":
        store = ModelRegistryStore(root, validate_relative_path(options.registry))
        current = store.load()
        if current.get("revision") != options.expected_revision:
            raise ContractError("model registry revision does not match --expected-revision")
        updated = register_challenger_generation(
            current,
            repository_root=root,
            generation_id=options.generation_id,
            parent_generation_id=options.parent_generation_id,
            challenger_model_id=options.model_id,
            challenger_artifact=_ref(root, options.model_artifact),
            architecture_version=options.architecture_version,
            quantization=options.quantization,
            selfplay_manifest=_ref(root, options.selfplay_manifest),
            teacher_labeling_manifest=_ref(root, options.teacher_manifest),
            training_run_manifest=_ref(root, options.training_run),
            registered_at=options.registered_at,
        )
        store.update(updated, expected_revision=options.expected_revision)
        return _ref(root, options.registry).as_dict()
    if command in {"record-promotion", "finalize-generation"}:
        store = ModelRegistryStore(root, validate_relative_path(options.registry))
        current = store.load()
        if current.get("revision") != options.expected_revision:
            raise ContractError("model registry revision does not match --expected-revision")
        promotion_ref, promotion = _json_and_ref(root, options.promotion_decision)
        promotion_root = validate_promotion_decision(promotion)
        if promotion_root.get("generationId") != options.generation_id:
            raise ContractError("promotion decision manifest does not match the generation")
        generations = current.get("generations")
        assert isinstance(generations, list)
        target = next(
            (
                item
                for item in generations
                if isinstance(item, dict) and item.get("generationId") == options.generation_id
            ),
            None,
        )
        if target is None:
            raise ContractError("generation is not present in the model registry")
        if promotion_root.get("championModelId") != target.get(
            "championModelId"
        ) or promotion_root.get("challengerModelId") != target.get("challengerModelId"):
            raise ContractError("promotion decision model IDs differ from the registry generation")
        analysis_reference = ArtifactRef.from_dict(
            promotion_root.get("arenaAnalysis"), "promotion decision.arenaAnalysis"
        )
        observed_analysis_ref, analysis = _json_and_ref(root, analysis_reference.path)
        if observed_analysis_ref != analysis_reference:
            raise ContractError("promotion decision references invalid arena analysis")
        analysis = validate_arena_analysis(analysis)
        policy_ref = _ref(root, options.policy)
        policy = load_generation_policy(_path(root, policy_ref.path, must_exist=True))
        promotion_root = validate_promotion_decision_binding(
            promotion_root,
            analysis=analysis,
            analysis_ref=analysis_reference,
            policy=policy,
            policy_ref=policy_ref,
            repository_root=root,
        )
        arena_reference = _ref(root, options.arena_manifest)
        if (
            analysis.get("generationId") != options.generation_id
            or analysis.get("championModelId") != target.get("championModelId")
            or analysis.get("challengerModelId") != target.get("challengerModelId")
            or ArtifactRef.from_dict(analysis.get("results"), "arena analysis.results")
            != arena_reference
        ):
            raise ContractError("arena manifest is not the result evidence behind the decision")
        updated = record_generation_promotion(
            current,
            generation_id=options.generation_id,
            arena_manifest=arena_reference,
            promotion_decision=promotion_ref,
            decision=str(promotion_root["decision"]),
            repository_root=root,
        )
        store.update(updated, expected_revision=options.expected_revision)
        return _ref(root, options.registry).as_dict()
    raise ContractError(f"unsupported command: {command}")


def _plan_paired(options: argparse.Namespace, root: Path, *, arena: bool) -> dict[str, object]:
    config, config_ref = _config_and_ref(root, options.selfplay_config)
    _, registry_raw = _json_and_ref(root, options.registry)
    registry = validate_model_registry(registry_raw, repository_root=root)
    registry_ref = registry_snapshot_ref(root, options.registry, registry)
    champion_id = options.champion or registry.get("championModelId")
    if not isinstance(champion_id, str):
        raise ContractError("model registry has no champion")
    champion = _model_from_registry(registry, champion_id)
    start_ref, start_raw = _json_and_ref(root, options.start_positions)
    if start_ref.path != config.paths.start_positions_manifest:
        raise ContractError("start-position path differs from the strict self-play configuration")
    starts = parse_start_positions(start_raw)
    validation_ref, validation = _json_and_ref(root, options.start_validation)
    if config.paths.engine_cli != DEFAULT_ENGINE_PATH:
        raise ContractError("self-play config engineCli must remain the operator alias")
    engine_ref, receipt_ref, _ = resolve_active_engine_build(
        root, expected_git_commit=options.git_commit
    )
    validate_start_position_validation(
        validation,
        start_positions_ref=start_ref,
        engine_ref=engine_ref,
        positions=starts,
        repository_root=root,
        engine_build_receipt=receipt_ref,
        runtime_authorization=True,
    )
    common = {
        "generation_id": options.generation_id,
        "champion": champion,
        "engine": engine_ref,
        "engine_build_receipt": receipt_ref,
        "model_registry": registry_ref,
        "git_commit": options.git_commit,
        "config": config,
        "config_ref": config_ref,
        "start_positions": starts,
        "start_positions_ref": start_ref,
        "validation_ref": validation_ref,
    }
    if arena:
        challenger_id = options.challenger or registry.get("challengerModelId")
        if not isinstance(challenger_id, str):
            raise ContractError("model registry has no challenger")
        plan = build_arena_plan(challenger=_model_from_registry(registry, challenger_id), **common)
    else:
        plan = build_selfplay_plan(**common)
    return _publish(root, options.output, plan).as_dict()


def _registry_model(root: Path, registry_path: str, model_id: str) -> ModelSpec:
    _, raw = _json_and_ref(root, registry_path)
    registry = validate_model_registry(raw, repository_root=root)
    return _model_from_registry(registry, model_id)


def _model_from_registry(registry: Mapping[str, Any], model_id: str) -> ModelSpec:
    models = registry.get("models")
    assert isinstance(models, list)
    for raw in models:
        if isinstance(raw, dict) and raw.get("modelId") == model_id:
            return ModelSpec(
                model_id=model_id,
                artifact=ArtifactRef.from_dict(raw.get("artifact"), f"model {model_id}.artifact"),
                evaluator_kind=str(raw.get("evaluatorKind")),
            )
    raise ContractError(f"model is not present in registry: {model_id}")


def _config_and_ref(root: Path, relative: str) -> tuple[Any, ArtifactRef]:
    path = _path(root, relative, must_exist=True)
    return load_selfplay_config(path), artifact_ref(root, validate_relative_path(relative))


def _json_and_ref(
    root: Path,
    relative: str,
    *,
    maximum_nodes: int | None = None,
) -> tuple[ArtifactRef, object]:
    normalized = validate_relative_path(relative)
    load_options = {} if maximum_nodes is None else {"maximum_nodes": maximum_nodes}
    value, reference = load_json_and_ref(root, normalized, **load_options)
    return reference, value


def _ref(root: Path, relative: str) -> ArtifactRef:
    return artifact_ref(root, validate_relative_path(relative))


def _path(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    return contained_path(root, validate_relative_path(relative), must_exist=must_exist)


def _publish(root: Path, relative: str, value: object) -> ArtifactRef:
    normalized = validate_relative_path(relative)
    write_json_new(contained_path(root, normalized), value)
    return artifact_ref(root, normalized)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m open_shogi_training.selfplay")
    parser.add_argument("--project-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--selfplay-config", required=True)
    validate.add_argument("--promotion-policy", required=True)

    starts = subparsers.add_parser("build-start-positions")
    starts.add_argument("--positions", required=True)
    starts.add_argument("--dataset-manifest", required=True)
    starts.add_argument("--seed", type=int, required=True)
    starts.add_argument("--train-count", type=int, required=True)
    starts.add_argument("--validation-count", type=int, required=True)
    starts.add_argument("--output", required=True)

    start_validation = subparsers.add_parser("validate-start-positions")
    start_validation.add_argument("--selfplay-config", required=True)
    start_validation.add_argument("--start-positions", required=True)
    start_validation.add_argument("--git-commit", required=True)
    start_validation.add_argument("--log-root", required=True)
    start_validation.add_argument("--timeout-seconds", type=int, default=30)
    start_validation.add_argument("--output", required=True)

    for name in ("plan-selfplay", "plan-arena"):
        plan = subparsers.add_parser(name)
        plan.add_argument("--selfplay-config", required=True)
        plan.add_argument("--registry", required=True)
        plan.add_argument("--generation-id", required=True)
        plan.add_argument("--champion")
        if name == "plan-arena":
            plan.add_argument("--challenger")
        plan.add_argument("--start-positions", required=True)
        plan.add_argument("--start-validation", required=True)
        plan.add_argument("--git-commit", required=True)
        plan.add_argument("--output", required=True)

    hard = subparsers.add_parser("extract-hard")
    hard.add_argument("--selfplay-config", required=True)
    hard.add_argument("--evidence", required=True)
    hard.add_argument("--labels-before", type=int, required=True)
    hard.add_argument("--requested-max", type=int)
    hard.add_argument("--output", required=True)

    evidence = subparsers.add_parser("build-evidence")
    evidence.add_argument("--selfplay-plan", required=True)
    evidence.add_argument("--selfplay-manifest", required=True)
    evidence.add_argument("--teacher-labels", required=True)
    evidence.add_argument("--model-predictions", required=True)
    evidence.add_argument("--dataset-manifest", required=True)
    evidence.add_argument("--generation-ordinal", type=int, required=True)
    evidence.add_argument("--teacher-source-generation-id", default="generation-0")
    evidence.add_argument("--export-root", required=True)
    evidence.add_argument("--timeout-seconds", type=int, default=120)
    evidence.add_argument("--output", required=True)

    replay = subparsers.add_parser("build-replay")
    replay.add_argument("--selfplay-config", required=True)
    replay.add_argument("--candidates", required=True)
    replay.add_argument("--output", required=True)

    replay_candidates = subparsers.add_parser("build-replay-candidates")
    replay_candidates.add_argument("--selfplay-config", required=True)
    replay_candidates.add_argument("--evidence", required=True)
    replay_candidates.add_argument("--hard-positions", required=True)
    replay_candidates.add_argument("--output", required=True)

    analyze = subparsers.add_parser("analyze-arena")
    analyze.add_argument("--results", required=True)
    analyze.add_argument("--output", required=True)

    collect = subparsers.add_parser("collect-arena")
    collect.add_argument("--plan", required=True)
    collect.add_argument("--execution", required=True)
    collect.add_argument("--output", required=True)

    promote = subparsers.add_parser("decide-promotion")
    promote.add_argument("--analysis", required=True)
    promote.add_argument("--policy", required=True)
    promote.add_argument("--decided-at", required=True)
    promote.add_argument("--output", required=True)

    training = subparsers.add_parser("plan-training")
    training.add_argument("--generation-id", required=True)
    training.add_argument("--registry", required=True)
    training.add_argument("--parent-model", required=True)
    training.add_argument("--replay-manifest", required=True)
    training.add_argument("--labels", required=True)
    training.add_argument("--label-manifest", required=True)
    training.add_argument("--positions", required=True)
    training.add_argument("--dataset-manifest", required=True)
    training.add_argument("--features-config", required=True)
    training.add_argument("--model-config", required=True)
    training.add_argument("--training-config", required=True)
    training.add_argument("--output-dir", required=True)
    training.add_argument("--resume-checkpoint")
    training.add_argument("--timeout-seconds", type=int, default=86_400)
    training.add_argument("--output", required=True)

    teacher = subparsers.add_parser("plan-teacher")
    teacher.add_argument("--selfplay-config", required=True)
    teacher.add_argument("--generation-id", required=True)
    teacher.add_argument("--hard-positions", required=True)
    teacher.add_argument("--normalized-positions", required=True)
    teacher.add_argument("--dataset-manifest", required=True)
    teacher.add_argument("--labeling-config", required=True)
    teacher.add_argument("--benchmark-report", required=True)
    teacher.add_argument("--output-dir", required=True)
    teacher.add_argument("--target-completed", type=int, required=True)
    teacher.add_argument("--labels-before", type=int, required=True)
    teacher.add_argument("--timeout-seconds", type=int, default=86_400)
    teacher.add_argument("--output", required=True)

    execute = subparsers.add_parser("execute-paired")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--state", required=True)
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--retry-quarantined", action="store_true")

    registry = subparsers.add_parser("init-registry")
    registry.add_argument("--generation-id", default="generation-0")
    registry.add_argument("--model-id", required=True)
    registry.add_argument("--model-artifact", required=True)
    registry.add_argument("--architecture-version", required=True)
    registry.add_argument("--quantization", choices=("float32", "int8"), required=True)
    registry.add_argument("--training-run")
    registry.add_argument("--registered-at", required=True)
    registry.add_argument("--output", required=True)

    challenger = subparsers.add_parser("register-challenger")
    challenger.add_argument("--registry", required=True)
    challenger.add_argument("--expected-revision", type=int, required=True)
    challenger.add_argument("--generation-id", required=True)
    challenger.add_argument("--parent-generation-id", required=True)
    challenger.add_argument("--model-id", required=True)
    challenger.add_argument("--model-artifact", required=True)
    challenger.add_argument("--architecture-version", required=True)
    challenger.add_argument("--quantization", choices=("float32", "int8"), required=True)
    challenger.add_argument("--selfplay-manifest", required=True)
    challenger.add_argument("--teacher-manifest", required=True)
    challenger.add_argument("--training-run", required=True)
    challenger.add_argument("--registered-at", required=True)

    for name in ("record-promotion", "finalize-generation"):
        promotion = subparsers.add_parser(name)
        promotion.add_argument("--registry", required=True)
        promotion.add_argument("--expected-revision", type=int, required=True)
        promotion.add_argument("--generation-id", required=True)
        promotion.add_argument("--arena-manifest", required=True)
        promotion.add_argument("--promotion-decision", required=True)
        promotion.add_argument("--policy", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
