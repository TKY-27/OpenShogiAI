"""Command-line entry point for the bounded Phase 7 evaluation workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from open_shogi_training.selfplay.common import (
    ArtifactRef,
    artifact_ref,
    contained_path,
    ensure_contained_directory,
    load_json_and_ref,
    load_json_artifact,
    validate_relative_path,
    write_json_new,
)

from .config import EvaluationConfig, load_evaluation_config
from .pipeline import (
    EvaluationError,
    analyze_official_evaluation,
    build_official_evaluation_plan,
    curate_hard_examples,
    prepare_official_games,
    validate_evaluation_report,
    validate_hard_examples,
    validate_official_evaluation_plan,
)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        root = Path(options.project_root).resolve(strict=True)
        config = load_evaluation_config(
            contained_path(root, validate_relative_path(options.config), must_exist=True)
        )
        payload = _dispatch(options, root, config)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _dispatch(
    options: argparse.Namespace,
    root: Path,
    config: EvaluationConfig,
) -> object:
    if options.command == "validate-config":
        return {
            "schema": "phase7_config_validation/v1",
            "configSha256": config.sha256,
            "runId": config.official.run_id,
            "games": len(config.official.human_sides),
            "teacherNodes": config.teacher.nodes,
            "maxTeacherCalls": config.resources.max_teacher_calls,
            "autoTrainingEligible": False,
        }
    if options.command == "plan":
        output_root = validate_relative_path(options.output_root)
        plan_relative = f"{output_root}/plan.json"
        plan_path = contained_path(root, plan_relative)
        if plan_path.exists() or plan_path.is_symlink():
            value, reference = load_json_and_ref(root, plan_relative)
            plan = validate_official_evaluation_plan(
                value,
                config=config,
                repository_root=root,
            )
            if plan.get("outputRoot") != output_root:
                raise EvaluationError("existing official plan uses another output root")
        else:
            plan = build_official_evaluation_plan(
                config=config,
                repository_root=root,
                registry_path=options.registry,
                output_root=output_root,
                git_commit=options.git_commit,
            )
            ensure_contained_directory(root, output_root)
            write_json_new(plan_path, plan)
            reference = artifact_ref(root, plan_relative)
        return {
            "schema": "phase7_plan_command/v1",
            "plan": reference.as_dict(),
            "games": plan["games"],
        }
    if options.command == "commands":
        plan_ref, plan = _load_plan(root, options.plan, config)
        return {
            "schema": "phase7_official_commands/v1",
            "plan": plan_ref.as_dict(),
            "commands": [game["command"] for game in plan["games"]],
        }
    if options.command == "prepare-games":
        plan_ref, _ = _load_plan(root, options.plan, config)
        manifest, reference = prepare_official_games(
            config=config,
            repository_root=root,
            plan_ref=plan_ref,
        )
        return {
            "schema": "phase7_prepare_games_command/v1",
            "gamesManifest": reference.as_dict(),
            "games": manifest["games"],
        }
    if options.command == "analyze":
        plan_ref, _ = _load_plan(root, options.plan, config)
        report, reference = analyze_official_evaluation(
            config=config,
            repository_root=root,
            plan_ref=plan_ref,
        )
        return {
            "schema": "phase7_analyze_command/v1",
            "evaluationReport": reference.as_dict(),
            "counts": report["counts"],
            "classifications": report["classifications"],
        }
    if options.command == "curate":
        report_relative = validate_relative_path(options.report)
        report_ref = artifact_ref(root, report_relative)
        report = validate_evaluation_report(
            load_json_artifact(root, report_ref),
            config=config,
            repository_root=root,
        )
        plan_ref = ArtifactRef.from_dict(report.get("plan"), "evaluation report.plan")
        plan = validate_official_evaluation_plan(
            load_json_artifact(root, plan_ref),
            config=config,
            repository_root=root,
        )
        relative = f"{plan['outputRoot']}/hard-examples.json"
        path = contained_path(root, relative)
        if path.exists() or path.is_symlink():
            value, reference = load_json_and_ref(root, relative)
            hard = validate_hard_examples(
                value,
                config=config,
                repository_root=root,
                expected_report=report_ref,
            )
        else:
            hard = curate_hard_examples(
                config=config,
                repository_root=root,
                report_ref=report_ref,
            )
            write_json_new(path, hard)
            reference = artifact_ref(root, relative)
        return {
            "schema": "phase7_curate_command/v1",
            "hardExamples": reference.as_dict(),
            "selected": hard["selected"],
            "autoTrainingEligible": hard["autoTrainingEligible"],
        }
    if options.command == "verify":
        report_ref = artifact_ref(root, validate_relative_path(options.report))
        report = validate_evaluation_report(
            load_json_artifact(root, report_ref),
            config=config,
            repository_root=root,
        )
        hard_ref: ArtifactRef | None = None
        if options.hard_examples is not None:
            hard_ref = artifact_ref(root, validate_relative_path(options.hard_examples))
            validate_hard_examples(
                load_json_artifact(root, hard_ref),
                config=config,
                repository_root=root,
                expected_report=report_ref,
            )
        return {
            "schema": "phase7_verification/v1",
            "evaluationReport": report_ref.as_dict(),
            "hardExamples": hard_ref.as_dict() if hard_ref is not None else None,
            "counts": report["counts"],
            "status": "verified",
        }
    raise AssertionError(f"unhandled command {options.command!r}")


def _load_plan(
    root: Path,
    value: str,
    config: EvaluationConfig,
) -> tuple[ArtifactRef, Mapping[str, object]]:
    relative = validate_relative_path(value)
    plan, reference = load_json_and_ref(root, relative)
    return reference, validate_official_evaluation_plan(
        plan,
        config=config,
        repository_root=root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m open_shogi_training.evaluation",
        description="Run the bounded two-game Phase 7 developer evaluation.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/evaluation/phase7_official.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")

    plan = commands.add_parser("plan")
    plan.add_argument("--registry", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--git-commit", required=True)

    for name in ("commands", "prepare-games", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)

    curate = commands.add_parser("curate")
    curate.add_argument("--report", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--report", required=True)
    verify.add_argument("--hard-examples")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
