"""Bounded Phase 7 official evaluation, teacher analysis, and hard-example curation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import stable_directory_lock
from open_shogi_training.labeling.config import TeacherConfig, load_teacher_config
from open_shogi_training.labeling.fingerprint import (
    TeacherFingerprint,
    fingerprint_teacher,
)
from open_shogi_training.labeling.legality import (
    LegalityCoverage,
    LegalityValidatorIdentity,
    RustLegalityValidator,
)
from open_shogi_training.labeling.usi import USIEngine, USIIdentity, USISearchResult
from open_shogi_training.models.export import parse_value_model
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    artifact_ref,
    canonical_sha256,
    contained_path,
    ensure_contained_directory,
    load_bytes_artifact,
    load_json_and_ref,
    load_json_artifact,
    require_bool,
    require_clean_head,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_list,
    require_mapping,
    require_string,
    validate_relative_path,
    validate_utc_timestamp,
    verify_artifact_ref,
    write_json_new,
)
from open_shogi_training.selfplay.engine_receipt import (
    resolve_active_engine_build,
    validate_engine_build_receipt,
    validate_engine_build_receipt_document,
)
from open_shogi_training.selfplay.execution import (
    CommandOutcome,
    CommandRunner,
    validate_command_receipt,
)
from open_shogi_training.selfplay.registry import validate_model_registry

from .config import EvaluationConfig
from .diagnosis import diagnose_decision

PLAN_SCHEMA: Final = "phase7_official_evaluation_plan/v1"
GAMES_SCHEMA: Final = "phase7_official_games/v1"
ANALYSIS_SCHEMA: Final = "phase7_teacher_analysis/v1"
REPORT_SCHEMA: Final = "phase7_evaluation_report/v1"
HARD_EXAMPLES_SCHEMA: Final = "phase7_hard_examples/v1"

MAX_GAMES: Final = 2
MAX_POSITIONS: Final = 512
MAX_DECISION_LOG_BYTES: Final = 64 * 1024 * 1024
MAX_CSA_BYTES: Final = 16 * 1024 * 1024
MAX_EXPORT_BYTES: Final = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES: Final = 4 * 1024 * 1024
MAX_EXPORT_ATTEMPTS: Final = 3

_GIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_USI_MOVE_RE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")
_PLAY_CONFIG_KEYS = frozenset(
    {
        "schema",
        "configSha256",
        "humanSide",
        "budgetKind",
        "budgetValue",
        "depth",
        "initialSfen",
        "maxPlies",
        "profile",
        "modelId",
        "modelArtifactSha256",
        "modelPayloadSha256",
        "architectureVersion",
        "quantization",
        "registrySha256",
        "registryRevision",
        "openingArtifactSha256",
        "openingArtifactSize",
        "openingMaxPlies",
        "transpositionEntries",
        "engineName",
        "engineVersion",
    }
)
_DECISION_KEYS = frozenset(
    {
        "schema",
        "ply",
        "actor",
        "modelId",
        "modelArtifactSha256",
        "modelPayloadSha256",
        "configSha256",
        "sfenBefore",
        "moveUsi",
        "nodes",
        "elapsedMs",
        "depth",
        "pv",
        "scoreCp",
        "openingBook",
    }
)
_EXPORT_OK_KEYS = frozenset(
    {
        "schema",
        "status",
        "inputFile",
        "normalizedCsa",
        "initialSfen",
        "positionSfens",
        "usiMoves",
        "blackName",
        "whiteName",
        "terminalReason",
        "outcome",
        "resultValidation",
    }
)

EngineResolver = Callable[[Path], tuple[ArtifactRef, ArtifactRef, dict[str, Any]]]
EngineFactory = Callable[[TeacherConfig, Path], USIEngine]
ReceiptLoader = Callable[[Path], tuple[ArtifactRef, ArtifactRef]]
ValidatorFactory = Callable[[Path, ReceiptLoader], RustLegalityValidator]


class EvaluationError(ValueError):
    """Raised when Phase 7 evidence cannot be safely produced or verified."""


def build_official_evaluation_plan(
    *,
    config: EvaluationConfig,
    repository_root: Path,
    registry_path: str,
    output_root: str,
    git_commit: str,
    created_at: str | None = None,
    engine_resolver: EngineResolver | None = None,
) -> dict[str, object]:
    """Build the fixed two-game human-vs-champion evaluation plan."""

    root = repository_root.resolve(strict=True)
    commit = require_clean_head(root, git_commit)
    registry_relative = validate_relative_path(registry_path)
    output_relative = validate_relative_path(output_root)
    if output_relative.startswith("local/") or not output_relative.startswith("artifacts/phase7/"):
        raise EvaluationError("official output_root must be below artifacts/phase7")
    resolver = engine_resolver or _resolve_engine
    engine, build_receipt, _ = resolver(root)
    validate_engine_build_receipt(
        root,
        build_receipt,
        expected_engine=engine,
        expected_git_commit=commit,
    )
    registry_value, registry_ref = load_json_and_ref(root, registry_relative)
    registry = validate_model_registry(registry_value, repository_root=root)
    champion = _champion_record(root, registry)
    teacher_path = contained_path(root, config.teacher.config_path, must_exist=True)
    teacher_config = load_teacher_config(teacher_path)
    if teacher_config.nodes != config.teacher.nodes:
        raise EvaluationError("evaluation teacher node budget differs from the teacher config")
    teacher_ref = artifact_ref(root, config.teacher.config_path, maximum_bytes=64 * 1024)
    timestamp = created_at or _utc_now()
    validate_utc_timestamp(timestamp, "official plan.createdAt")

    games = []
    for side in config.official.human_sides:
        game_id = f"official-{side}"
        csa_path = f"{output_relative}/games/{game_id}.csa"
        decision_path = f"{output_relative}/games/{game_id}.decisions.jsonl"
        argv = [
            engine.path,
            "play",
            "--profile",
            config.official.profile,
            "--registry",
            registry_ref.path,
            "--human",
            side,
            "--nodes",
            str(config.official.nodes),
            "--depth",
            str(config.official.depth),
            "--max-plies",
            str(config.official.max_plies),
            "--opening-max-plies",
            str(config.official.opening_max_plies),
            "--output",
            csa_path,
            "--decision-log",
            decision_path,
        ]
        games.append(
            {
                "gameId": game_id,
                "humanSide": side,
                "csaPath": csa_path,
                "decisionLogPath": decision_path,
                "command": {"kind": "human_play", "argv": argv},
            }
        )
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "runId": config.official.run_id,
        "gitCommit": commit,
        "createdAt": timestamp,
        "evaluationConfigSha256": config.sha256,
        "autoTrainingEligible": False,
        "outputRoot": output_relative,
        "engine": engine.as_dict(),
        "engineBuildReceipt": build_receipt.as_dict(),
        "registry": registry_ref.as_dict(),
        "registryRevision": registry["revision"],
        "champion": champion,
        "teacherConfig": teacher_ref.as_dict(),
        "teacherConfigSha256": teacher_config.sha256,
        "teacherNodes": config.teacher.nodes,
        "games": games,
    }
    validate_official_evaluation_plan(plan, config=config, repository_root=root)
    return plan


def validate_official_evaluation_plan(
    raw: object,
    *,
    config: EvaluationConfig,
    repository_root: Path,
) -> Mapping[str, Any]:
    root = repository_root.resolve(strict=True)
    plan = require_mapping(raw, "official evaluation plan")
    require_exact_keys(
        plan,
        {
            "schema",
            "runId",
            "gitCommit",
            "createdAt",
            "evaluationConfigSha256",
            "autoTrainingEligible",
            "outputRoot",
            "engine",
            "engineBuildReceipt",
            "registry",
            "registryRevision",
            "champion",
            "teacherConfig",
            "teacherConfigSha256",
            "teacherNodes",
            "games",
        },
        "official evaluation plan",
    )
    if plan.get("schema") != PLAN_SCHEMA or plan.get("runId") != config.official.run_id:
        raise EvaluationError("official plan schema or run ID is invalid")
    commit = _git_commit(plan.get("gitCommit"), "official plan.gitCommit")
    validate_utc_timestamp(plan.get("createdAt"), "official plan.createdAt")
    if plan.get("evaluationConfigSha256") != config.sha256:
        raise EvaluationError("official plan config digest is stale")
    if require_bool(plan, "autoTrainingEligible", "official plan"):
        raise EvaluationError("official evaluation evidence must never be automatically trainable")
    output_root = require_string(plan, "outputRoot", "official plan", maximum_length=1_024)
    output_root = validate_relative_path(output_root)
    if not output_root.startswith("artifacts/phase7/"):
        raise EvaluationError("official plan output root is outside artifacts/phase7")
    engine = ArtifactRef.from_dict(plan.get("engine"), "official plan.engine")
    build_receipt = ArtifactRef.from_dict(
        plan.get("engineBuildReceipt"), "official plan.engineBuildReceipt"
    )
    _validate_durable_engine_build_receipt(
        root,
        build_receipt,
        expected_engine=engine,
        expected_git_commit=commit,
    )
    registry_ref = ArtifactRef.from_dict(plan.get("registry"), "official plan.registry")
    registry = validate_model_registry(load_json_artifact(root, registry_ref), repository_root=root)
    if plan.get("registryRevision") != registry["revision"]:
        raise EvaluationError("official plan registry revision differs from its bytes")
    champion = _champion_record(root, registry)
    if plan.get("champion") != champion:
        raise EvaluationError("official plan champion differs from its registry")
    teacher_ref = ArtifactRef.from_dict(plan.get("teacherConfig"), "official plan.teacherConfig")
    verify_artifact_ref(root, teacher_ref)
    if teacher_ref.path != config.teacher.config_path:
        raise EvaluationError("official plan teacher config path is not fixed by config")
    teacher = load_teacher_config(contained_path(root, teacher_ref.path, must_exist=True))
    if (
        teacher.sha256 != plan.get("teacherConfigSha256")
        or teacher.nodes != plan.get("teacherNodes")
        or teacher.nodes != config.teacher.nodes
    ):
        raise EvaluationError("official plan teacher identity or node budget differs")
    games = require_list(
        plan, "games", "official plan", minimum_items=MAX_GAMES, maximum_items=MAX_GAMES
    )
    expected_games = _expected_plan_games(
        config=config,
        output_root=output_root,
        engine=engine,
        registry=registry_ref,
    )
    if games != expected_games:
        raise EvaluationError("official plan game commands differ from the fixed mode")
    return plan


def _expected_plan_games(
    *,
    config: EvaluationConfig,
    output_root: str,
    engine: ArtifactRef,
    registry: ArtifactRef,
) -> list[dict[str, object]]:
    games: list[dict[str, object]] = []
    for side in config.official.human_sides:
        game_id = f"official-{side}"
        csa = f"{output_root}/games/{game_id}.csa"
        decisions = f"{output_root}/games/{game_id}.decisions.jsonl"
        games.append(
            {
                "gameId": game_id,
                "humanSide": side,
                "csaPath": csa,
                "decisionLogPath": decisions,
                "command": {
                    "kind": "human_play",
                    "argv": [
                        engine.path,
                        "play",
                        "--profile",
                        config.official.profile,
                        "--registry",
                        registry.path,
                        "--human",
                        side,
                        "--nodes",
                        str(config.official.nodes),
                        "--depth",
                        str(config.official.depth),
                        "--max-plies",
                        str(config.official.max_plies),
                        "--opening-max-plies",
                        str(config.official.opening_max_plies),
                        "--output",
                        csa,
                        "--decision-log",
                        decisions,
                    ],
                },
            }
        )
    return games


def _champion_record(root: Path, registry: Mapping[str, Any]) -> dict[str, object]:
    champion_id = require_identifier(registry, "championModelId", "model registry")
    models = require_list(
        registry, "models", "model registry", minimum_items=1, maximum_items=10_000
    )
    model = next(
        (
            require_mapping(item, "registry model")
            for item in models
            if isinstance(item, dict) and item.get("modelId") == champion_id
        ),
        None,
    )
    if model is None:
        raise EvaluationError("registry champion model is missing")
    artifact = ArtifactRef.from_dict(model.get("artifact"), "registry champion.artifact")
    parsed = parse_value_model(load_bytes_artifact(root, artifact, maximum_bytes=64 * 1024 * 1024))
    return {
        "modelId": champion_id,
        "artifact": artifact.as_dict(),
        "payloadSha256": parsed.payload_sha256,
        "architectureVersion": model.get("architectureVersion"),
        "quantization": model.get("quantization"),
    }


def _resolve_engine(root: Path) -> tuple[ArtifactRef, ArtifactRef, dict[str, Any]]:
    return resolve_active_engine_build(root)


def _validate_durable_engine_build_receipt(
    root: Path,
    receipt: ArtifactRef,
    *,
    expected_engine: ArtifactRef,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Validate immutable build evidence without reauthorizing historical execution."""

    return validate_engine_build_receipt_document(
        load_json_artifact(root, receipt),
        expected_engine=expected_engine,
        expected_git_commit=expected_git_commit,
    )


def prepare_official_games(
    *,
    config: EvaluationConfig,
    repository_root: Path,
    plan_ref: ArtifactRef,
    runner_factory: Callable[[Path], CommandRunner] = CommandRunner,
) -> tuple[Mapping[str, Any], ArtifactRef]:
    """Rust-replay every fixed game and bind it to its human decision log."""

    root = repository_root.resolve(strict=True)
    plan = validate_official_evaluation_plan(
        load_json_artifact(root, plan_ref), config=config, repository_root=root
    )
    output_root = require_string(plan, "outputRoot", "official plan", maximum_length=1_024)
    manifest_relative = f"{output_root}/games-manifest.json"
    manifest_path = contained_path(root, manifest_relative)
    if manifest_path.exists() or manifest_path.is_symlink():
        value, reference = load_json_and_ref(root, manifest_relative)
        return (
            validate_official_games_manifest(
                value,
                config=config,
                repository_root=root,
                expected_plan=plan_ref,
            ),
            reference,
        )

    _assert_exact_game_directory(root, plan)
    export_ref, command_outcome = _run_csa_export(
        root=root,
        plan=plan,
        config=config,
        runner_factory=runner_factory,
    )
    exports = _parse_export_rows(
        load_bytes_artifact(root, export_ref, maximum_bytes=MAX_EXPORT_BYTES)
    )
    games = []
    for planned in require_list(
        plan, "games", "official plan", minimum_items=MAX_GAMES, maximum_items=MAX_GAMES
    ):
        game_plan = require_mapping(planned, "official plan game")
        game_id = require_identifier(game_plan, "gameId", "official plan game")
        csa_path = require_string(game_plan, "csaPath", "official plan game", maximum_length=1_024)
        decision_path = require_string(
            game_plan, "decisionLogPath", "official plan game", maximum_length=1_024
        )
        csa_ref = artifact_ref(root, csa_path, maximum_bytes=MAX_CSA_BYTES)
        decision_ref = artifact_ref(root, decision_path, maximum_bytes=MAX_DECISION_LOG_BYTES)
        export = exports.get(PurePosixPath(csa_path).name)
        if export is None:
            raise EvaluationError(f"Rust exporter omitted official game {game_id}")
        play_config, events = _load_and_validate_decision_log(
            root,
            decision_ref,
            game_plan=game_plan,
            plan=plan,
            export=export,
        )
        csa_bytes = load_bytes_artifact(root, csa_ref, maximum_bytes=MAX_CSA_BYTES)
        if export["normalizedCsa"].encode("utf-8") != csa_bytes:
            raise EvaluationError(f"official game {game_id} CSA is not canonical")
        human_moves = [str(event["moveUsi"]) for event in events if event.get("actor") == "human"]
        ai_moves = [str(event["moveUsi"]) for event in events if event.get("actor") == "ai"]
        games.append(
            {
                "gameId": game_id,
                "humanSide": game_plan["humanSide"],
                "csa": csa_ref.as_dict(),
                "decisionLog": decision_ref.as_dict(),
                "configSha256": play_config["configSha256"],
                "modelId": play_config["modelId"],
                "modelArtifactSha256": play_config["modelArtifactSha256"],
                "modelPayloadSha256": play_config["modelPayloadSha256"],
                "initialSfen": export["initialSfen"],
                "positionSfens": export["positionSfens"],
                "usiMoves": export["usiMoves"],
                "humanMoves": human_moves,
                "aiMoves": ai_moves,
                "terminalReason": export["terminalReason"],
                "outcome": export["outcome"],
                "resultValidation": export["resultValidation"],
                "decisionCount": len(events),
            }
        )
    if command_outcome.command_receipt is None:
        raise EvaluationError("CSA export did not publish its command receipt")
    manifest: dict[str, object] = {
        "schema": GAMES_SCHEMA,
        "runId": plan["runId"],
        "plan": plan_ref.as_dict(),
        "engineValidation": {
            "export": export_ref.as_dict(),
            "commandReceipt": command_outcome.command_receipt.as_dict(),
            "stdout": command_outcome.stdout.as_dict(),
            "stderr": command_outcome.stderr.as_dict(),
        },
        "autoTrainingEligible": False,
        "games": games,
    }
    validate_official_games_manifest(
        manifest,
        config=config,
        repository_root=root,
        expected_plan=plan_ref,
    )
    write_json_new(manifest_path, manifest)
    reference = artifact_ref(root, manifest_relative)
    return manifest, reference


def validate_official_games_manifest(
    raw: object,
    *,
    config: EvaluationConfig,
    repository_root: Path,
    expected_plan: ArtifactRef | None = None,
) -> Mapping[str, Any]:
    root = repository_root.resolve(strict=True)
    manifest = require_mapping(raw, "official games manifest")
    require_exact_keys(
        manifest,
        {"schema", "runId", "plan", "engineValidation", "autoTrainingEligible", "games"},
        "official games manifest",
    )
    if manifest.get("schema") != GAMES_SCHEMA or manifest.get("runId") != config.official.run_id:
        raise EvaluationError("official games manifest schema or run ID is invalid")
    if require_bool(manifest, "autoTrainingEligible", "official games manifest"):
        raise EvaluationError("official games must remain excluded from automatic training")
    plan_ref = ArtifactRef.from_dict(manifest.get("plan"), "official games manifest.plan")
    if expected_plan is not None and plan_ref != expected_plan:
        raise EvaluationError("official games manifest references another plan")
    plan = validate_official_evaluation_plan(
        load_json_artifact(root, plan_ref), config=config, repository_root=root
    )
    validation = require_mapping(manifest.get("engineValidation"), "engine validation")
    require_exact_keys(
        validation, {"export", "commandReceipt", "stdout", "stderr"}, "engine validation"
    )
    refs = {
        key: ArtifactRef.from_dict(validation.get(key), f"engine validation.{key}")
        for key in ("export", "commandReceipt", "stdout", "stderr")
    }
    for reference in refs.values():
        verify_artifact_ref(root, reference)
    export_prefix = _csa_export_attempt_prefix(str(plan["outputRoot"]), refs["export"])
    if (
        refs["stdout"].path != f"{export_prefix}/stdout.log"
        or refs["stderr"].path != f"{export_prefix}/stderr.log"
        or refs["commandReceipt"].path != f"{export_prefix}/command-receipt.json"
    ):
        raise EvaluationError("CSA replay evidence paths differ from their bounded attempt")
    engine = ArtifactRef.from_dict(plan.get("engine"), "official plan.engine")
    build_receipt = ArtifactRef.from_dict(
        plan.get("engineBuildReceipt"), "official plan.engineBuildReceipt"
    )
    command = _csa_export_command(
        output_root=str(plan["outputRoot"]),
        engine=engine,
        output=refs["export"].path,
    )
    command_identity = canonical_sha256(
        {
            "command": command,
            "resume": False,
            "memoryLimitMiB": config.resources.memory_limit_mib,
            "expectedExecutable": engine.as_dict(),
            "engineBuildReceipt": build_receipt.as_dict(),
            "runtimeReceipt": None,
        }
    )
    outcome = validate_command_receipt(
        root,
        refs["commandReceipt"],
        command=command,
        command_identity=command_identity,
        resume=False,
        memory_limit_mib=config.resources.memory_limit_mib,
        expected_executable=engine,
        engine_build_receipt=build_receipt,
        runtime_receipt=None,
        expected_git_commit=str(plan["gitCommit"]),
        stdout_path=refs["stdout"].path,
        stderr_path=refs["stderr"].path,
    )
    if (
        outcome.return_code != 0
        or outcome.timed_out
        or outcome.output_limit_exceeded
        or outcome.memory_limit_exceeded
    ):
        raise EvaluationError("CSA replay receipt records a failed command")
    exports = _parse_export_rows(
        load_bytes_artifact(root, refs["export"], maximum_bytes=MAX_EXPORT_BYTES)
    )
    games = require_list(
        manifest,
        "games",
        "official games manifest",
        minimum_items=MAX_GAMES,
        maximum_items=MAX_GAMES,
    )
    planned_games = require_list(
        plan, "games", "official plan", minimum_items=MAX_GAMES, maximum_items=MAX_GAMES
    )
    if len(games) != len(planned_games):
        raise EvaluationError("official games count differs from its plan")
    for index, (game_raw, planned_raw) in enumerate(zip(games, planned_games, strict=True)):
        context = f"official games[{index}]"
        game = require_mapping(game_raw, context)
        require_exact_keys(
            game,
            {
                "gameId",
                "humanSide",
                "csa",
                "decisionLog",
                "configSha256",
                "modelId",
                "modelArtifactSha256",
                "modelPayloadSha256",
                "initialSfen",
                "positionSfens",
                "usiMoves",
                "humanMoves",
                "aiMoves",
                "terminalReason",
                "outcome",
                "resultValidation",
                "decisionCount",
            },
            context,
        )
        planned = require_mapping(planned_raw, f"official plan games[{index}]")
        if game.get("gameId") != planned.get("gameId") or game.get("humanSide") != planned.get(
            "humanSide"
        ):
            raise EvaluationError(f"{context} identity differs from its plan")
        csa = ArtifactRef.from_dict(game.get("csa"), f"{context}.csa")
        decisions = ArtifactRef.from_dict(game.get("decisionLog"), f"{context}.decisionLog")
        if csa.path != planned.get("csaPath") or decisions.path != planned.get("decisionLogPath"):
            raise EvaluationError(f"{context} artifact paths differ from its plan")
        export = exports.get(PurePosixPath(csa.path).name)
        if export is None:
            raise EvaluationError(f"{context} has no Rust replay row")
        play_config, events = _load_and_validate_decision_log(
            root,
            decisions,
            game_plan=planned,
            plan=plan,
            export=export,
        )
        csa_bytes = load_bytes_artifact(root, csa, maximum_bytes=MAX_CSA_BYTES)
        if export["normalizedCsa"].encode("utf-8") != csa_bytes:
            raise EvaluationError(f"{context} CSA differs from Rust canonical replay")
        expected = {
            "gameId": planned["gameId"],
            "humanSide": planned["humanSide"],
            "csa": csa.as_dict(),
            "decisionLog": decisions.as_dict(),
            "configSha256": play_config["configSha256"],
            "modelId": play_config["modelId"],
            "modelArtifactSha256": play_config["modelArtifactSha256"],
            "modelPayloadSha256": play_config["modelPayloadSha256"],
            "initialSfen": export["initialSfen"],
            "positionSfens": export["positionSfens"],
            "usiMoves": export["usiMoves"],
            "humanMoves": [event["moveUsi"] for event in events if event["actor"] == "human"],
            "aiMoves": [event["moveUsi"] for event in events if event["actor"] == "ai"],
            "terminalReason": export["terminalReason"],
            "outcome": export["outcome"],
            "resultValidation": export["resultValidation"],
            "decisionCount": len(events),
        }
        if game != expected:
            raise EvaluationError(f"{context} is not the deterministic game derivation")
    if set(exports) != {PurePosixPath(str(game["csa"]["path"])).name for game in games}:  # type: ignore[index]
        raise EvaluationError("Rust replay contains an unexpected official game")
    return manifest


def _run_csa_export(
    *,
    root: Path,
    plan: Mapping[str, Any],
    config: EvaluationConfig,
    runner_factory: Callable[[Path], CommandRunner],
) -> tuple[ArtifactRef, CommandOutcome]:
    output_root = str(plan["outputRoot"])
    engine = ArtifactRef.from_dict(plan.get("engine"), "official plan.engine")
    build_receipt = ArtifactRef.from_dict(
        plan.get("engineBuildReceipt"), "official plan.engineBuildReceipt"
    )
    runner = runner_factory(root)
    for attempt in range(1, MAX_EXPORT_ATTEMPTS + 1):
        prefix = f"{output_root}/validation/attempt-{attempt:03d}"
        output = f"{prefix}/games.jsonl"
        stdout = f"{prefix}/stdout.log"
        stderr = f"{prefix}/stderr.log"
        receipt = f"{prefix}/command-receipt.json"
        ensure_contained_directory(root, prefix)
        occupied = [
            contained_path(root, path).exists() or contained_path(root, path).is_symlink()
            for path in (output, stdout, stderr, receipt)
        ]
        if any(occupied) and not contained_path(root, receipt).is_file():
            continue
        command = _csa_export_command(
            output_root=output_root,
            engine=engine,
            output=output,
        )
        outcome = runner.run(
            command,
            stdout_path=stdout,
            stderr_path=stderr,
            expected_executable=engine,
            engine_build_receipt=build_receipt,
            memory_limit_mib=config.resources.memory_limit_mib,
            receipt_path=receipt,
        )
        if (
            outcome.return_code != 0
            or outcome.timed_out
            or outcome.output_limit_exceeded
            or outcome.memory_limit_exceeded
        ):
            raise EvaluationError("Rust CSA replay command failed within its bounded attempt")
        return artifact_ref(root, output, maximum_bytes=MAX_EXPORT_BYTES), outcome
    raise EvaluationError("CSA replay exhausted its bounded recovery attempts")


def _csa_export_command(
    *,
    output_root: str,
    engine: ArtifactRef,
    output: str,
) -> dict[str, object]:
    return {
        "kind": "engine_dataset_export",
        "argv": [
            engine.path,
            "export-csa-jsonl",
            "--input-dir",
            f"{output_root}/games",
            "--output",
            output,
            "--max-games",
            str(MAX_GAMES),
        ],
        "timeoutSeconds": 120,
    }


def _csa_export_attempt_prefix(output_root: str, reference: ArtifactRef) -> str:
    for attempt in range(1, MAX_EXPORT_ATTEMPTS + 1):
        prefix = f"{output_root}/validation/attempt-{attempt:03d}"
        if reference.path == f"{prefix}/games.jsonl":
            return prefix
    raise EvaluationError("CSA replay output path is outside its bounded attempt set")


def _assert_exact_game_directory(root: Path, plan: Mapping[str, Any]) -> None:
    games = require_list(
        plan, "games", "official plan", minimum_items=MAX_GAMES, maximum_items=MAX_GAMES
    )
    expected = {
        PurePosixPath(str(game[key])).name
        for game in games
        if isinstance(game, dict)
        for key in ("csaPath", "decisionLogPath")
    }
    directory = contained_path(root, f"{plan['outputRoot']}/games")
    if directory.is_symlink() or not directory.is_dir():
        raise EvaluationError("official games directory must be a non-symlink directory")
    entries = list(directory.iterdir())
    cleanup = next((entry for entry in entries if entry.name == ".open-shogi-cleanup"), None)
    if cleanup is not None and (
        cleanup.is_symlink() or not cleanup.is_dir() or any(cleanup.iterdir())
    ):
        raise EvaluationError("official games cleanup directory is not empty and canonical")
    observed = {entry.name for entry in entries if entry.name != ".open-shogi-cleanup"}
    if observed != expected:
        raise EvaluationError("official games directory differs from the fixed plan")


def _parse_export_rows(raw: bytes) -> dict[str, dict[str, Any]]:
    rows = _parse_jsonl_bytes(raw, "Rust CSA export", maximum_records=MAX_GAMES)
    if len(rows) != MAX_GAMES:
        raise EvaluationError(f"Rust CSA export must contain exactly {MAX_GAMES} games")
    parsed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        context = f"Rust CSA export[{index}]"
        if (
            set(row) != _EXPORT_OK_KEYS
            or row.get("schema") != "phase3_csa_export/v1"
            or row.get("status") != "ok"
        ):
            raise EvaluationError(f"{context} violates the accepted export schema")
        name = row.get("inputFile")
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or not name.endswith(".csa")
            or name in parsed
        ):
            raise EvaluationError(f"{context}.inputFile is unsafe or duplicated")
        moves = row.get("usiMoves")
        sfens = row.get("positionSfens")
        if (
            not isinstance(moves, list)
            or not isinstance(sfens, list)
            or len(sfens) != len(moves) + 1
            or len(moves) > MAX_POSITIONS
            or any(not _usi_move(move) for move in moves)
            or any(not _sfen(sfen) for sfen in sfens)
            or row.get("initialSfen") != sfens[0]
        ):
            raise EvaluationError(f"{context} has an invalid replay sequence")
        if row.get("blackName") not in {"human", "OpenShogiAI"} or row.get("whiteName") not in {
            "human",
            "OpenShogiAI",
        }:
            raise EvaluationError(f"{context} has unexpected player names")
        if row.get("terminalReason") is None or row.get("resultValidation") not in {
            "verified",
            "external_condition",
        }:
            raise EvaluationError(f"{context} lacks a supported terminal result")
        if row.get("outcome") not in {"black_win", "white_win", "draw", "unknown"}:
            raise EvaluationError(f"{context}.outcome is invalid")
        normalized = row.get("normalizedCsa")
        if (
            not isinstance(normalized, str)
            or not normalized
            or len(normalized.encode()) > MAX_CSA_BYTES
        ):
            raise EvaluationError(f"{context}.normalizedCsa is invalid")
        parsed[name] = dict(row)
    return parsed


def _load_and_validate_decision_log(
    root: Path,
    reference: ArtifactRef,
    *,
    game_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    export: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = load_bytes_artifact(root, reference, maximum_bytes=MAX_DECISION_LOG_BYTES)
    rows = _parse_jsonl_bytes(
        raw, f"decision log {reference.path}", maximum_records=MAX_POSITIONS + 1
    )
    if not rows:
        raise EvaluationError("human-play decision log lacks its configuration row")
    play_config = rows[0]
    events = rows[1:]
    if (
        set(play_config) != _PLAY_CONFIG_KEYS
        or play_config.get("schema") != "phase6_human_play_config/v1"
    ):
        raise EvaluationError("human-play configuration violates its closed schema")
    _validate_play_config(play_config, game_plan=game_plan, plan=plan, export=export)
    moves = export["usiMoves"]
    sfens = export["positionSfens"]
    if len(events) != len(moves):
        raise EvaluationError("human-play decision count differs from Rust CSA replay")
    human_side = str(game_plan["humanSide"])
    champion = require_mapping(plan.get("champion"), "official plan.champion")
    for index, (event, move, sfen) in enumerate(zip(events, moves, sfens, strict=False), start=1):
        context = f"human-play decision[{index}]"
        if set(event) != _DECISION_KEYS or event.get("schema") != "phase6_human_decision/v1":
            raise EvaluationError(f"{context} violates its closed schema")
        side = str(sfen).split(" ")[1]
        expected_actor = "human" if (side == "b") == (human_side == "black") else "ai"
        if (
            event.get("ply") != index
            or event.get("actor") != expected_actor
            or event.get("modelId") != champion["modelId"]
            or event.get("modelArtifactSha256") != champion["artifact"]["sha256"]  # type: ignore[index]
            or event.get("modelPayloadSha256") != champion["payloadSha256"]
            or event.get("configSha256") != play_config["configSha256"]
            or event.get("sfenBefore") != sfen
            or event.get("moveUsi") != move
        ):
            raise EvaluationError(f"{context} differs from config or Rust replay")
        _validate_decision_event(event, context=context, plan=plan)
    return dict(play_config), [dict(event) for event in events]


def _validate_play_config(
    record: Mapping[str, Any],
    *,
    game_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    export: Mapping[str, Any],
) -> None:
    champion = require_mapping(plan.get("champion"), "official plan.champion")
    artifact = require_mapping(champion.get("artifact"), "official plan.champion.artifact")
    expected = {
        "humanSide": game_plan["humanSide"],
        "budgetKind": "nodes",
        "budgetValue": _plan_option(game_plan, "--nodes"),
        "depth": _plan_option(game_plan, "--depth"),
        "initialSfen": export["initialSfen"],
        "maxPlies": _plan_option(game_plan, "--max-plies"),
        "profile": "champion",
        "modelId": champion["modelId"],
        "modelArtifactSha256": artifact["sha256"],
        "modelPayloadSha256": champion["payloadSha256"],
        "architectureVersion": int(str(champion["architectureVersion"])),
        "quantization": champion["quantization"],
        "registrySha256": plan["registry"]["sha256"],  # type: ignore[index]
        "registryRevision": plan["registryRevision"],
        "openingArtifactSha256": None,
        "openingArtifactSize": None,
        "openingMaxPlies": _plan_option(game_plan, "--opening-max-plies"),
        "transpositionEntries": 16_384,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise EvaluationError(f"human-play config {key} differs from the official plan")
    for key in ("engineName", "engineVersion"):
        if not isinstance(record.get(key), str) or not record[key] or len(record[key]) > 128:
            raise EvaluationError(f"human-play config {key} is invalid")
    if record["engineName"] != "OpenShogiAI":
        raise EvaluationError("human-play config engineName is not OpenShogiAI")
    digest = _recorded_play_config_sha256(record)
    if record.get("configSha256") != digest:
        raise EvaluationError("human-play config digest is invalid")
    if game_plan["humanSide"] == "black":
        expected_names = ("human", "OpenShogiAI")
    else:
        expected_names = ("OpenShogiAI", "human")
    if (export["blackName"], export["whiteName"]) != expected_names:
        raise EvaluationError("human-play CSA player assignment differs from the plan")


def _validate_decision_event(
    event: Mapping[str, Any], *, context: str, plan: Mapping[str, Any]
) -> None:
    actor = event.get("actor")
    if (
        actor not in {"human", "ai"}
        or not _usi_move(event.get("moveUsi"))
        or not _sfen(event.get("sfenBefore"))
    ):
        raise EvaluationError(f"{context} has an invalid actor, move, or SFEN")
    for key, maximum in (
        ("nodes", 1_000_000_000),
        ("elapsedMs", 9_007_199_254_740_991),
        ("depth", 64),
    ):
        value = event.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise EvaluationError(f"{context}.{key} is outside its bound")
    pv = event.get("pv")
    if not isinstance(pv, list) or len(pv) > 512 or any(not _usi_move(move) for move in pv):
        raise EvaluationError(f"{context}.pv is invalid")
    score = event.get("scoreCp")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, int) or abs(score) > 1_000_000_000
    ):
        raise EvaluationError(f"{context}.scoreCp is invalid")
    if not isinstance(event.get("openingBook"), bool):
        raise EvaluationError(f"{context}.openingBook must be boolean")
    if actor == "human":
        if (
            any((event["nodes"], event["elapsedMs"], event["depth"]))
            or pv
            or score is not None
            or event["openingBook"]
        ):
            raise EvaluationError(f"{context} human move carries engine evidence")
    elif event["openingBook"]:
        raise EvaluationError(f"{context} official mode must not use an opening book")
    elif (
        not pv
        or pv[0] != event["moveUsi"]
        or score is None
        or event["nodes"] > _plan_option_from_plan(plan, "--nodes")
        or event["depth"] > _plan_option_from_plan(plan, "--depth")
    ):
        raise EvaluationError(f"{context} AI search evidence is inconsistent")


def _recorded_play_config_sha256(record: Mapping[str, Any]) -> str:
    payload = (
        "schema=phase6_human_play_config/v1;"
        f"human={record['humanSide']};budget_kind={record['budgetKind']};"
        f"budget_value={record['budgetValue']};depth={record['depth']};"
        f"initial_sfen={record['initialSfen']};max_plies={record['maxPlies']};"
        f"profile={record['profile']};model_id={record['modelId']};"
        f"artifact={_rust_option(record['modelArtifactSha256'])};"
        f"payload={_rust_option(record['modelPayloadSha256'])};"
        f"architecture={_rust_option(record['architectureVersion'])};"
        f"quantization={_rust_option(record['quantization'])};"
        f"registry_sha256={_rust_option(record['registrySha256'])};"
        f"registry_revision={_rust_option(record['registryRevision'])};"
        f"opening_sha256={_rust_option(record['openingArtifactSha256'])};"
        f"opening_size={_rust_option(record['openingArtifactSize'])};"
        f"opening_max_plies={record['openingMaxPlies']};"
        f"transposition_entries={record['transpositionEntries']};"
        f"engine={record['engineName']}:{record['engineVersion']}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rust_option(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        if not value.isascii() or any(character in value for character in "\x00\r\n"):
            raise EvaluationError("human-play optional strings must be safe ASCII")
        return f"Some({json.dumps(value, ensure_ascii=True)})"
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError("human-play optional value has no Rust debug representation")
    return f"Some({value})"


def _plan_option(game: Mapping[str, Any], option: str) -> int:
    command = require_mapping(game.get("command"), "official game command")
    argv = command.get("argv")
    if not isinstance(argv, list) or option not in argv:
        raise EvaluationError(f"official game command lacks {option}")
    index = argv.index(option)
    try:
        value = int(argv[index + 1])
    except (IndexError, TypeError, ValueError) as error:
        raise EvaluationError(f"official game command has an invalid {option}") from error
    return value


def _plan_option_from_plan(plan: Mapping[str, Any], option: str) -> int:
    games = require_list(plan, "games", "official plan", minimum_items=2, maximum_items=2)
    return _plan_option(require_mapping(games[0], "official plan game"), option)


def _parse_jsonl_bytes(raw: bytes, context: str, *, maximum_records: int) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw or len(raw) > MAX_DECISION_LOG_BYTES:
        raise EvaluationError(f"{context} must be bounded canonical LF-terminated JSONL")
    result: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line or len(line) + 1 > MAX_JSONL_LINE_BYTES:
            raise EvaluationError(f"{context} line {index} is empty or oversized")
        try:
            value = json.loads(
                line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError, EvaluationError) as error:
            raise EvaluationError(f"{context} line {index} is invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise EvaluationError(f"{context} line {index} must be an object")
        result.append(value)
        if len(result) > maximum_records:
            raise EvaluationError(f"{context} exceeds {maximum_records} records")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvaluationError(f"non-finite JSON number {value!r}")


def _usi_move(value: object) -> bool:
    return isinstance(value, str) and _USI_MOVE_RE.fullmatch(value) is not None


def _sfen(value: object) -> bool:
    if not isinstance(value, str) or not value.isascii() or len(value) > 1_024:
        return False
    fields = value.split(" ")
    return (
        len(fields) == 4
        and all(fields)
        and fields[1] in {"b", "w"}
        and fields[3].isdecimal()
        and str(int(fields[3])) == fields[3]
        and int(fields[3]) >= 1
    )


def analyze_official_evaluation(
    *,
    config: EvaluationConfig,
    repository_root: Path,
    plan_ref: ArtifactRef,
    runner_factory: Callable[[Path], CommandRunner] = CommandRunner,
    engine_factory: EngineFactory = USIEngine,
    validator_factory: ValidatorFactory | None = None,
) -> tuple[Mapping[str, Any], ArtifactRef]:
    """Analyze every recorded move position, resuming at immutable row granularity."""

    root = repository_root.resolve(strict=True)
    plan = validate_official_evaluation_plan(
        load_json_artifact(root, plan_ref), config=config, repository_root=root
    )
    output_root = str(plan["outputRoot"])
    report_relative = f"{output_root}/evaluation-report.json"
    report_path = contained_path(root, report_relative)
    lock_directory = contained_path(root, output_root)
    with stable_directory_lock(
        lock_directory,
        create=True,
        exclusive=True,
        nonblocking=False,
    ):
        if report_path.exists() or report_path.is_symlink():
            value, reference = load_json_and_ref(root, report_relative)
            return (
                validate_evaluation_report(
                    value,
                    config=config,
                    repository_root=root,
                    expected_plan=plan_ref,
                ),
                reference,
            )
        games, games_ref = prepare_official_games(
            config=config,
            repository_root=root,
            plan_ref=plan_ref,
            runner_factory=runner_factory,
        )
        teacher_ref = ArtifactRef.from_dict(
            plan.get("teacherConfig"), "official plan.teacherConfig"
        )
        teacher_config = load_teacher_config(
            contained_path(root, teacher_ref.path, must_exist=True)
        )
        if teacher_config.nodes != config.teacher.nodes:
            raise EvaluationError("active teacher config has a different node budget")
        if teacher_config.benchmark.max_peak_rss_mib != config.resources.memory_limit_mib:
            raise EvaluationError(
                "evaluation memory limit must equal the teacher's enforced process-tree ceiling"
            )
        fingerprint = fingerprint_teacher(teacher_config, root)
        positions = _expected_analysis_positions(root, plan=plan, games=games)
        if len(positions) > config.resources.max_teacher_calls:
            raise EvaluationError("official games exceed the configured teacher-call ceiling")

        analysis_rows: list[Mapping[str, Any] | None] = [None] * len(positions)
        analysis_refs: list[ArtifactRef | None] = [None] * len(positions)
        missing: list[int] = []
        for index, position in enumerate(positions):
            relative = _analysis_relative(output_root, position["gameId"], position["ply"])
            path = contained_path(root, relative)
            if path.exists() or path.is_symlink():
                value, reference = load_json_and_ref(root, relative)
                analysis_rows[index] = validate_analysis_record(
                    value,
                    config=config,
                    plan=plan,
                    game=position["game"],
                    event=position["event"],
                    fingerprint=fingerprint,
                    repository_root=root,
                )
                analysis_refs[index] = reference
            else:
                missing.append(index)

        if missing:
            engine = engine_factory(teacher_config, root)
            validator_loader = _plan_receipt_loader(plan)
            validator = (
                validator_factory(root, validator_loader)
                if validator_factory is not None
                else RustLegalityValidator(root, receipt_loader=validator_loader)
            )
            active_error: BaseException | None = None
            try:
                identity = engine.start()
                validator_identity = validator.start()
                for index in missing:
                    position = positions[index]
                    event = position["event"]
                    search = engine.analyze_with_retry(
                        str(event["sfenBefore"]), nodes=config.teacher.nodes
                    )
                    coverage = validator.validate(
                        str(event["sfenBefore"]),
                        search.bestmove,
                        [list(candidate.pv) for candidate in search.candidates],
                        configured_multipv=teacher_config.multipv,
                    )
                    row = _analysis_record(
                        config=config,
                        plan=plan,
                        game=position["game"],
                        event=event,
                        search=search,
                        fingerprint=fingerprint,
                        identity=identity,
                        validator_identity=validator_identity,
                        coverage=coverage,
                    )
                    validated = validate_analysis_record(
                        row,
                        config=config,
                        plan=plan,
                        game=position["game"],
                        event=event,
                        fingerprint=fingerprint,
                        repository_root=root,
                    )
                    relative = _analysis_relative(
                        output_root, str(position["gameId"]), int(position["ply"])
                    )
                    path = contained_path(root, relative)
                    write_json_new(path, row)
                    analysis_rows[index] = validated
                    analysis_refs[index] = artifact_ref(root, relative)
            except BaseException as error:
                active_error = error
                raise
            finally:
                try:
                    engine.close()
                except BaseException as error:
                    if active_error is None:
                        raise
                    active_error.add_note(f"teacher cleanup also failed: {error!r}")
                try:
                    validator.close()
                except BaseException as error:
                    if active_error is None:
                        raise
                    active_error.add_note(f"legality-validator cleanup also failed: {error!r}")

        if any(row is None for row in analysis_rows) or any(
            reference is None for reference in analysis_refs
        ):
            raise AssertionError("analysis completed with unresolved rows")
        rows = [row for row in analysis_rows if row is not None]
        references = [reference for reference in analysis_refs if reference is not None]
        report = _evaluation_report(
            config=config,
            plan=plan,
            plan_ref=plan_ref,
            games=games,
            games_ref=games_ref,
            fingerprint=fingerprint,
            rows=rows,
            references=references,
        )
        validate_evaluation_report(
            report,
            config=config,
            repository_root=root,
            expected_plan=plan_ref,
        )
        write_json_new(report_path, report)
        reference = artifact_ref(root, report_relative)
        return report, reference


def validate_analysis_record(
    raw: object,
    *,
    config: EvaluationConfig,
    plan: Mapping[str, Any],
    game: Mapping[str, Any],
    event: Mapping[str, Any],
    fingerprint: TeacherFingerprint,
    repository_root: Path,
) -> Mapping[str, Any]:
    record = require_mapping(raw, "teacher analysis")
    require_exact_keys(
        record,
        {
            "schema",
            "runId",
            "gameId",
            "ply",
            "actor",
            "source",
            "configSha256",
            "sfen",
            "engineDecision",
            "teacherConfigSha256",
            "teacherNodes",
            "teacher",
            "reportedIdentity",
            "search",
            "legality",
            "diagnosis",
            "createdAt",
        },
        "teacher analysis",
    )
    if (
        record.get("schema") != ANALYSIS_SCHEMA
        or record.get("runId") != plan.get("runId")
        or record.get("gameId") != game.get("gameId")
        or record.get("ply") != event.get("ply")
        or record.get("actor") != event.get("actor")
        or record.get("configSha256") != event.get("configSha256")
        or record.get("sfen") != event.get("sfenBefore")
        or record.get("engineDecision") != event
    ):
        raise EvaluationError("teacher analysis source identity differs from its game decision")
    source = require_mapping(record.get("source"), "teacher analysis.source")
    require_exact_keys(source, {"csa", "decisionLog"}, "teacher analysis.source")
    csa = ArtifactRef.from_dict(source.get("csa"), "teacher analysis.source.csa")
    decisions = ArtifactRef.from_dict(
        source.get("decisionLog"), "teacher analysis.source.decisionLog"
    )
    if csa.as_dict() != game.get("csa") or decisions.as_dict() != game.get("decisionLog"):
        raise EvaluationError("teacher analysis source refs differ from its game")
    verify_artifact_ref(repository_root, csa)
    verify_artifact_ref(repository_root, decisions)
    if (
        record.get("teacherConfigSha256") != plan.get("teacherConfigSha256")
        or record.get("teacherNodes") != config.teacher.nodes
        or record.get("teacher") != fingerprint.identity_record()
    ):
        raise EvaluationError("teacher analysis uses a different teacher identity")
    reported = _validate_reported_identity(record.get("reportedIdentity"))
    search = _validate_search_record(record.get("search"), config=config)
    legality = _validate_legality_record(
        record.get("legality"),
        search=search,
        plan=plan,
        repository_root=repository_root,
    )
    del legality
    expected_diagnosis = diagnose_decision(event, search, config.diagnosis)
    if record.get("diagnosis") != expected_diagnosis:
        raise EvaluationError("teacher analysis diagnosis is not its deterministic derivation")
    validate_utc_timestamp(record.get("createdAt"), "teacher analysis.createdAt")
    if not reported["name"]:
        raise EvaluationError("teacher analysis lacks a reported teacher name")
    return record


def _analysis_record(
    *,
    config: EvaluationConfig,
    plan: Mapping[str, Any],
    game: Mapping[str, Any],
    event: Mapping[str, Any],
    search: USISearchResult,
    fingerprint: TeacherFingerprint,
    identity: USIIdentity,
    validator_identity: LegalityValidatorIdentity,
    coverage: LegalityCoverage,
) -> dict[str, object]:
    search_record = {
        "bestmove": search.bestmove,
        "elapsedMs": search.elapsed_ms,
        "candidates": [candidate.as_dict() for candidate in search.candidates],
    }
    diagnosis = diagnose_decision(event, search_record, config.diagnosis)
    return {
        "schema": ANALYSIS_SCHEMA,
        "runId": plan["runId"],
        "gameId": game["gameId"],
        "ply": event["ply"],
        "actor": event["actor"],
        "source": {"csa": game["csa"], "decisionLog": game["decisionLog"]},
        "configSha256": event["configSha256"],
        "sfen": event["sfenBefore"],
        "engineDecision": dict(event),
        "teacherConfigSha256": plan["teacherConfigSha256"],
        "teacherNodes": config.teacher.nodes,
        "teacher": fingerprint.identity_record(),
        "reportedIdentity": {"name": identity.name, "author": identity.author},
        "search": search_record,
        "legality": {
            "validator": validator_identity.as_dict(),
            "requestedMultiPv": coverage.requested_multipv,
            "returnedCandidates": coverage.returned_candidates,
            "legalRootCount": coverage.legal_root_count,
            "hasAdditionalLegalRoots": coverage.has_additional_legal_roots,
        },
        "diagnosis": diagnosis,
        "createdAt": _utc_now(),
    }


def _validate_search_record(raw: object, *, config: EvaluationConfig) -> Mapping[str, Any]:
    search = require_mapping(raw, "teacher analysis.search")
    require_exact_keys(search, {"bestmove", "elapsedMs", "candidates"}, "teacher analysis.search")
    bestmove = require_string(search, "bestmove", "teacher analysis.search", maximum_length=16)
    if not _usi_move(bestmove):
        raise EvaluationError("teacher bestmove is not canonical USI")
    require_int(
        search,
        "elapsedMs",
        "teacher analysis.search",
        minimum=0,
        maximum=3_600_000,
    )
    candidates = require_list(
        search,
        "candidates",
        "teacher analysis.search",
        minimum_items=1,
        maximum_items=500,
    )
    roots: set[str] = set()
    for index, candidate_raw in enumerate(candidates, start=1):
        context = f"teacher analysis.search.candidates[{index - 1}]"
        candidate = require_mapping(candidate_raw, context)
        require_exact_keys(
            candidate, {"multipv", "score", "pv", "depth", "seldepth", "nodes"}, context
        )
        if candidate.get("multipv") != index:
            raise EvaluationError("teacher MultiPV ranks must be contiguous")
        score = require_mapping(candidate.get("score"), f"{context}.score")
        require_exact_keys(score, {"kind", "value"}, f"{context}.score")
        require_enum(score, "kind", f"{context}.score", {"cp", "mate"})
        require_int(
            score, "value", f"{context}.score", minimum=-1_000_000_000, maximum=1_000_000_000
        )
        pv = require_list(candidate, "pv", context, minimum_items=1, maximum_items=1_024)
        if any(not _usi_move(move) for move in pv):
            raise EvaluationError(f"{context}.pv contains a noncanonical move")
        if pv[0] in roots:
            raise EvaluationError("teacher MultiPV root moves must be distinct")
        roots.add(str(pv[0]))
        for key in ("depth", "seldepth", "nodes"):
            require_int(candidate, key, context, minimum=0, maximum=10_000_000_000)
    if candidates[0]["pv"][0] != bestmove or len(candidates) > 3:  # type: ignore[index]
        raise EvaluationError("teacher bestmove or MultiPV count differs from fixed config")
    return search


def _validate_legality_record(
    raw: object,
    *,
    search: Mapping[str, Any],
    plan: Mapping[str, Any],
    repository_root: Path,
) -> Mapping[str, Any]:
    legality = require_mapping(raw, "teacher analysis.legality")
    require_exact_keys(
        legality,
        {
            "validator",
            "requestedMultiPv",
            "returnedCandidates",
            "legalRootCount",
            "hasAdditionalLegalRoots",
        },
        "teacher analysis.legality",
    )
    validator = require_mapping(legality.get("validator"), "teacher analysis.legality.validator")
    require_exact_keys(
        validator,
        {"path", "sha256", "size", "build_receipt", "reported_name", "reported_author"},
        "teacher analysis.legality.validator",
    )
    engine = ArtifactRef.from_dict(plan.get("engine"), "official plan.engine")
    receipt = ArtifactRef.from_dict(
        plan.get("engineBuildReceipt"), "official plan.engineBuildReceipt"
    )
    if (
        validator.get("path") != engine.path
        or validator.get("sha256") != engine.sha256
        or validator.get("size") != engine.size
        or validator.get("build_receipt") != receipt.as_dict()
        or not isinstance(validator.get("reported_name"), str)
        or not str(validator["reported_name"]).startswith("OpenShogiAI ")
    ):
        raise EvaluationError("teacher legality validator differs from the planned engine")
    verify_artifact_ref(repository_root, engine)
    _validate_durable_engine_build_receipt(
        repository_root,
        receipt,
        expected_engine=engine,
        expected_git_commit=str(plan["gitCommit"]),
    )
    returned = len(search["candidates"])
    if legality.get("requestedMultiPv") != 3 or legality.get("returnedCandidates") != returned:
        raise EvaluationError("teacher legality coverage differs from the search")
    legal_roots = legality.get("legalRootCount")
    if legal_roots is not None and (
        isinstance(legal_roots, bool) or not isinstance(legal_roots, int) or legal_roots < returned
    ):
        raise EvaluationError("teacher legality root count is invalid")
    if legality.get("hasAdditionalLegalRoots") is not (
        legal_roots is not None and legal_roots > returned
    ):
        raise EvaluationError("teacher legality coverage flag is invalid")
    return legality


def _validate_reported_identity(raw: object) -> Mapping[str, Any]:
    identity = require_mapping(raw, "reported teacher identity")
    require_exact_keys(identity, {"name", "author"}, "reported teacher identity")
    name = identity.get("name")
    author = identity.get("author")
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 256
        or any(character in name for character in "\x00\r\n")
        or (
            author is not None
            and (
                not isinstance(author, str)
                or not author
                or len(author) > 256
                or any(character in author for character in "\x00\r\n")
            )
        )
    ):
        raise EvaluationError("reported teacher identity is invalid")
    return identity


def _expected_analysis_positions(
    root: Path,
    *,
    plan: Mapping[str, Any],
    games: Mapping[str, Any],
) -> list[dict[str, Any]]:
    planned = {
        str(game["gameId"]): require_mapping(game, "official plan game")
        for game in require_list(plan, "games", "official plan", minimum_items=2, maximum_items=2)
    }
    positions: list[dict[str, Any]] = []
    for game_raw in require_list(
        games, "games", "official games manifest", minimum_items=2, maximum_items=2
    ):
        game = require_mapping(game_raw, "official game")
        game_id = str(game["gameId"])
        game_plan = planned.get(game_id)
        if game_plan is None:
            raise EvaluationError("official games manifest contains an unplanned game")
        decision_ref = ArtifactRef.from_dict(game.get("decisionLog"), "official game.decisionLog")
        export = _game_export_view(game)
        _, events = _load_and_validate_decision_log(
            root,
            decision_ref,
            game_plan=game_plan,
            plan=plan,
            export=export,
        )
        for event in events:
            positions.append(
                {
                    "gameId": game_id,
                    "ply": event["ply"],
                    "game": game,
                    "event": event,
                }
            )
    return positions


def _game_export_view(game: Mapping[str, Any]) -> dict[str, Any]:
    human_side = str(game["humanSide"])
    return {
        "initialSfen": game["initialSfen"],
        "positionSfens": game["positionSfens"],
        "usiMoves": game["usiMoves"],
        "blackName": "human" if human_side == "black" else "OpenShogiAI",
        "whiteName": "OpenShogiAI" if human_side == "black" else "human",
        "terminalReason": game["terminalReason"],
        "outcome": game["outcome"],
        "resultValidation": game["resultValidation"],
    }


def _analysis_relative(output_root: str, game_id: object, ply: object) -> str:
    if not isinstance(game_id, str) or not isinstance(ply, int):
        raise EvaluationError("analysis identity is invalid")
    return f"{output_root}/analyses/{game_id}/ply-{ply:04d}.json"


def _plan_receipt_loader(
    plan: Mapping[str, Any],
) -> Callable[[Path], tuple[ArtifactRef, ArtifactRef]]:
    engine = ArtifactRef.from_dict(plan.get("engine"), "official plan.engine")
    receipt = ArtifactRef.from_dict(
        plan.get("engineBuildReceipt"), "official plan.engineBuildReceipt"
    )

    def load(_: Path) -> tuple[ArtifactRef, ArtifactRef]:
        return engine, receipt

    return load


def _evaluation_report(
    *,
    config: EvaluationConfig,
    plan: Mapping[str, Any],
    plan_ref: ArtifactRef,
    games: Mapping[str, Any],
    games_ref: ArtifactRef,
    fingerprint: TeacherFingerprint,
    rows: Sequence[Mapping[str, Any]],
    references: Sequence[ArtifactRef],
) -> dict[str, object]:
    classifications = Counter(str(row["diagnosis"]["kind"]) for row in rows)  # type: ignore[index]
    hard = sum(bool(row["diagnosis"]["hardExample"]) for row in rows)  # type: ignore[index]
    ai = sum(row["actor"] == "ai" for row in rows)
    human = sum(row["actor"] == "human" for row in rows)
    created = min(str(row["createdAt"]) for row in rows) if rows else _utc_now()
    first = rows[0] if rows else None
    if first is None:
        raise EvaluationError("official evaluation contains no recorded decision positions")
    return {
        "schema": REPORT_SCHEMA,
        "runId": plan["runId"],
        "plan": plan_ref.as_dict(),
        "games": games_ref.as_dict(),
        "createdAt": created,
        "completedAt": _utc_now(),
        "status": "complete",
        "autoTrainingEligible": False,
        "teacherConfig": plan["teacherConfig"],
        "teacherConfigSha256": plan["teacherConfigSha256"],
        "teacher": fingerprint.identity_record(),
        "reportedIdentity": first["reportedIdentity"],
        "legalityValidator": first["legality"]["validator"],  # type: ignore[index]
        "counts": {
            "games": len(games["games"]),  # type: ignore[arg-type]
            "positions": len(rows),
            "aiDecisions": ai,
            "humanDecisions": human,
            "hardExamples": hard,
        },
        "classifications": dict(sorted(classifications.items())),
        "analysis": [reference.as_dict() for reference in references],
    }


def validate_evaluation_report(
    raw: object,
    *,
    config: EvaluationConfig,
    repository_root: Path,
    expected_plan: ArtifactRef | None = None,
) -> Mapping[str, Any]:
    root = repository_root.resolve(strict=True)
    report = require_mapping(raw, "evaluation report")
    require_exact_keys(
        report,
        {
            "schema",
            "runId",
            "plan",
            "games",
            "createdAt",
            "completedAt",
            "status",
            "autoTrainingEligible",
            "teacherConfig",
            "teacherConfigSha256",
            "teacher",
            "reportedIdentity",
            "legalityValidator",
            "counts",
            "classifications",
            "analysis",
        },
        "evaluation report",
    )
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("runId") != config.official.run_id
        or report.get("status") != "complete"
    ):
        raise EvaluationError("evaluation report schema, run ID, or status is invalid")
    if require_bool(report, "autoTrainingEligible", "evaluation report"):
        raise EvaluationError("evaluation report must remain excluded from training")
    created = validate_utc_timestamp(report.get("createdAt"), "evaluation report.createdAt")
    completed = validate_utc_timestamp(report.get("completedAt"), "evaluation report.completedAt")
    del created, completed
    plan_ref = ArtifactRef.from_dict(report.get("plan"), "evaluation report.plan")
    if expected_plan is not None and plan_ref != expected_plan:
        raise EvaluationError("evaluation report references another plan")
    plan = validate_official_evaluation_plan(
        load_json_artifact(root, plan_ref), config=config, repository_root=root
    )
    games_ref = ArtifactRef.from_dict(report.get("games"), "evaluation report.games")
    games = validate_official_games_manifest(
        load_json_artifact(root, games_ref),
        config=config,
        repository_root=root,
        expected_plan=plan_ref,
    )
    teacher_ref = ArtifactRef.from_dict(
        report.get("teacherConfig"), "evaluation report.teacherConfig"
    )
    if teacher_ref.as_dict() != plan.get("teacherConfig"):
        raise EvaluationError("evaluation report teacher config differs from its plan")
    teacher_config = load_teacher_config(contained_path(root, teacher_ref.path, must_exist=True))
    fingerprint = fingerprint_teacher(teacher_config, root)
    if (
        report.get("teacherConfigSha256") != teacher_config.sha256
        or report.get("teacherConfigSha256") != plan.get("teacherConfigSha256")
        or report.get("teacher") != fingerprint.identity_record()
    ):
        raise EvaluationError("evaluation report teacher identity is invalid")
    reported = _validate_reported_identity(report.get("reportedIdentity"))
    positions = _expected_analysis_positions(root, plan=plan, games=games)
    analysis = require_list(
        report,
        "analysis",
        "evaluation report",
        minimum_items=1,
        maximum_items=config.resources.max_teacher_calls,
    )
    if len(analysis) != len(positions):
        raise EvaluationError("evaluation report does not cover every recorded decision")
    rows: list[Mapping[str, Any]] = []
    expected_paths: set[str] = set()
    for index, (reference_raw, position) in enumerate(zip(analysis, positions, strict=True)):
        reference = ArtifactRef.from_dict(reference_raw, f"evaluation report.analysis[{index}]")
        expected_path = _analysis_relative(
            str(plan["outputRoot"]), position["gameId"], position["ply"]
        )
        if reference.path != expected_path or reference.path in expected_paths:
            raise EvaluationError("evaluation report analysis paths are duplicated or noncanonical")
        expected_paths.add(reference.path)
        row = validate_analysis_record(
            load_json_artifact(root, reference),
            config=config,
            plan=plan,
            game=position["game"],
            event=position["event"],
            fingerprint=fingerprint,
            repository_root=root,
        )
        if row.get("reportedIdentity") != reported:
            raise EvaluationError("teacher reported identity changed across analysis rows")
        if row.get("legality", {}).get("validator") != report.get("legalityValidator"):
            raise EvaluationError("legality validator changed across analysis rows")
        rows.append(row)
    classifications = Counter(str(row["diagnosis"]["kind"]) for row in rows)  # type: ignore[index]
    counts = require_mapping(report.get("counts"), "evaluation report.counts")
    require_exact_keys(
        counts,
        {"games", "positions", "aiDecisions", "humanDecisions", "hardExamples"},
        "evaluation report.counts",
    )
    expected_counts = {
        "games": MAX_GAMES,
        "positions": len(rows),
        "aiDecisions": sum(row["actor"] == "ai" for row in rows),
        "humanDecisions": sum(row["actor"] == "human" for row in rows),
        "hardExamples": sum(bool(row["diagnosis"]["hardExample"]) for row in rows),  # type: ignore[index]
    }
    if counts != expected_counts or report.get("classifications") != dict(
        sorted(classifications.items())
    ):
        raise EvaluationError("evaluation report aggregates are not deterministic")
    if rows and report.get("createdAt") != min(str(row["createdAt"]) for row in rows):
        raise EvaluationError("evaluation report createdAt is not the first analysis timestamp")
    return report


def curate_hard_examples(
    *,
    config: EvaluationConfig,
    repository_root: Path,
    report_ref: ArtifactRef,
) -> dict[str, object]:
    """Select diagnostic examples while preserving a permanent training quarantine."""

    root = repository_root.resolve(strict=True)
    report = validate_evaluation_report(
        load_json_artifact(root, report_ref), config=config, repository_root=root
    )
    artifact = _derive_hard_examples(
        config=config,
        repository_root=root,
        report_ref=report_ref,
        report=report,
    )
    validate_hard_examples(
        artifact,
        config=config,
        repository_root=root,
        expected_report=report_ref,
    )
    return artifact


def _derive_hard_examples(
    *,
    config: EvaluationConfig,
    repository_root: Path,
    report_ref: ArtifactRef,
    report: Mapping[str, Any],
) -> dict[str, object]:
    candidates: list[tuple[int, str, int, ArtifactRef, Mapping[str, Any]]] = []
    for index, raw_ref in enumerate(report["analysis"]):  # type: ignore[index]
        reference = ArtifactRef.from_dict(raw_ref, f"evaluation report.analysis[{index}]")
        row = require_mapping(load_json_artifact(repository_root, reference), "teacher analysis")
        diagnosis = require_mapping(row.get("diagnosis"), "teacher analysis.diagnosis")
        if diagnosis.get("hardExample") is True:
            severity = require_int(
                diagnosis,
                "severityCp",
                "teacher analysis.diagnosis",
                minimum=0,
                maximum=1_000_000_000,
            )
            candidates.append((severity, str(row["gameId"]), int(row["ply"]), reference, row))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3].sha256))
    selected = candidates[: config.diagnosis.max_hard_examples]
    examples = []
    for severity, game_id, ply, reference, row in selected:
        diagnosis = require_mapping(row.get("diagnosis"), "teacher analysis.diagnosis")
        identity = hashlib.sha256(
            b"phase7_hard_example/v1\0"
            + game_id.encode("utf-8")
            + b"\0"
            + ply.to_bytes(8, "big")
            + b"\0"
            + str(row["sfen"]).encode("utf-8")
        ).hexdigest()
        examples.append(
            {
                "exampleId": identity,
                "gameId": game_id,
                "ply": ply,
                "actor": row["actor"],
                "sfen": row["sfen"],
                "classification": diagnosis["kind"],
                "severityCp": severity,
                "analysis": reference.as_dict(),
                "source": row["source"],
            }
        )
    return {
        "schema": HARD_EXAMPLES_SCHEMA,
        "runId": report["runId"],
        "evaluationReport": report_ref.as_dict(),
        "status": "pending_human_review",
        "autoTrainingEligible": False,
        "criteria": config.diagnosis.as_dict(),
        "selected": len(examples),
        "omitted": len(candidates) - len(examples),
        "examples": examples,
    }


def validate_hard_examples(
    raw: object,
    *,
    config: EvaluationConfig,
    repository_root: Path,
    expected_report: ArtifactRef | None = None,
) -> Mapping[str, Any]:
    root = repository_root.resolve(strict=True)
    value = require_mapping(raw, "hard examples")
    require_exact_keys(
        value,
        {
            "schema",
            "runId",
            "evaluationReport",
            "status",
            "autoTrainingEligible",
            "criteria",
            "selected",
            "omitted",
            "examples",
        },
        "hard examples",
    )
    if (
        value.get("schema") != HARD_EXAMPLES_SCHEMA
        or value.get("runId") != config.official.run_id
        or value.get("status") != "pending_human_review"
        or require_bool(value, "autoTrainingEligible", "hard examples")
    ):
        raise EvaluationError("hard examples schema/status/training quarantine is invalid")
    report_ref = ArtifactRef.from_dict(value.get("evaluationReport"), "hard examples.report")
    if expected_report is not None and report_ref != expected_report:
        raise EvaluationError("hard examples reference another evaluation report")
    report = validate_evaluation_report(
        load_json_artifact(root, report_ref), config=config, repository_root=root
    )
    expected = _derive_hard_examples(
        config=config,
        repository_root=root,
        report_ref=report_ref,
        report=report,
    )
    if value != expected:
        raise EvaluationError("hard examples are not the deterministic report derivation")
    return value


def _git_commit(value: object, context: str) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise EvaluationError(f"{context} must be a lowercase full Git object ID")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
