"""Deterministic job plans for self-play, challenger training, and paired arena."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .common import (
    ArtifactRef,
    ContractError,
    canonical_sha256,
    load_json_artifact,
    require_bool,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_list,
    require_mapping,
    require_relative_path,
    require_sha256,
    require_string,
    validate_identifier,
    validate_relative_path,
    validate_utc_timestamp,
    verify_artifact_ref,
)
from .config import (
    MAX_JSON_SAFE_INTEGER,
    PHASE6_SEED,
    SelfPlayConfig,
)

INITIAL_SFEN: Final = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
START_POSITIONS_SCHEMA: Final = "phase6_start_positions/v1"
START_VALIDATION_SCHEMA: Final = "phase6_start_position_validation/v1"
SELFPLAY_PLAN_SCHEMA: Final = "phase6_selfplay_plan/v1"
ARENA_PLAN_SCHEMA: Final = "phase6_paired_arena_plan/v1"
TRAINING_PLAN_SCHEMA: Final = "phase6_challenger_training_plan/v1"
TEACHER_PLAN_SCHEMA: Final = "phase6_supplemental_teacher_plan/v1"
MAX_START_POSITIONS: Final = 10_000
MAX_PHASE3_SOURCE_POSITIONS: Final = 250_000

_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{7,64}\Z")
_START_ROOT_KEYS = frozenset(
    {"schema", "datasetManifest", "sourcePositions", "selection", "positions"}
)
_START_SELECTION_KEYS = frozenset(
    {
        "seed",
        "trainCount",
        "validationCount",
        "requireEligible",
        "resetMoveNumber",
        "excludedCrossSplitStates",
    }
)
_START_POSITION_KEYS = frozenset(
    {"positionId", "sfen", "sourceGameSha256", "positionIndex", "split"}
)
_VALIDATION_ROOT_KEYS = frozenset(
    {
        "schema",
        "startPositions",
        "engine",
        "engineBuildReceipt",
        "gitCommit",
        "method",
        "results",
    }
)
_VALIDATION_RESULT_KEYS = frozenset(
    {
        "positionId",
        "sfenSha256",
        "legal",
        "returnCode",
        "timedOut",
        "outputLimitExceeded",
        "memoryLimitExceeded",
        "peakRssBytes",
        "rssMeasurement",
        "stdout",
        "stderr",
        "completedAt",
    }
)
_MODEL_SPEC_KEYS = frozenset({"modelId", "artifact", "evaluatorKind"})
_TRAINING_PLAN_KEYS = frozenset(
    {"schema", "generationId", "parentModel", "inputs", "outputDir", "command", "planSha256"}
)
_TRAINING_INPUT_KEYS = frozenset(
    {
        "replayManifest",
        "teacherLabels",
        "teacherLabelManifest",
        "positions",
        "datasetManifest",
        "featuresConfig",
        "modelConfig",
        "trainingConfig",
        "resumeCheckpoint",
    }
)
_TEACHER_PLAN_KEYS = frozenset(
    {
        "schema",
        "generationId",
        "hardPositions",
        "normalizedPositions",
        "datasetManifest",
        "labelingConfig",
        "benchmarkReport",
        "outputDir",
        "targetCompleted",
        "labelsBefore",
        "teacherLabelLimit",
        "command",
        "planSha256",
    }
)
_PLAN_COMMAND_KEYS = frozenset({"kind", "argv", "timeoutSeconds"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A registry-resolved model used by an engine command."""

    model_id: str
    artifact: ArtifactRef
    evaluator_kind: str

    def __post_init__(self) -> None:
        validate_identifier(self.model_id, "model_id")
        if self.evaluator_kind not in {
            "material",
            "handcrafted-baseline",
            "handcrafted-experimental",
            "neural",
        }:
            raise ContractError("unsupported evaluator_kind")

    def as_dict(self) -> dict[str, object]:
        return {
            "modelId": self.model_id,
            "artifact": self.artifact.as_dict(),
            "evaluatorKind": self.evaluator_kind,
        }


@dataclass(frozen=True, slots=True)
class StartPosition:
    position_id: str
    sfen: str
    source_game_sha256: str
    position_index: int
    split: str


@dataclass(frozen=True, slots=True)
class StartPositionSet:
    dataset_manifest: ArtifactRef
    source_positions: ArtifactRef
    positions: tuple[StartPosition, ...]


def parse_start_positions(raw: object) -> StartPositionSet:
    root = require_mapping(raw, "start positions")
    require_exact_keys(root, _START_ROOT_KEYS, "start positions")
    if root.get("schema") != START_POSITIONS_SCHEMA:
        raise ContractError("unsupported start-position schema")
    dataset_manifest = ArtifactRef.from_dict(root.get("datasetManifest"), "datasetManifest")
    source_positions = ArtifactRef.from_dict(root.get("sourcePositions"), "sourcePositions")
    selection = require_mapping(root.get("selection"), "start positions.selection")
    require_exact_keys(selection, _START_SELECTION_KEYS, "start positions.selection")
    selection_seed = require_int(
        selection, "seed", "start positions.selection", minimum=0, maximum=MAX_JSON_SAFE_INTEGER
    )
    if selection_seed != PHASE6_SEED:
        raise ContractError(f"start-position selection seed must be {PHASE6_SEED}")
    train_count = require_int(
        selection, "trainCount", "start positions.selection", minimum=1, maximum=5_000
    )
    validation_count = require_int(
        selection, "validationCount", "start positions.selection", minimum=1, maximum=5_000
    )
    if train_count + validation_count > MAX_START_POSITIONS:
        raise ContractError("start-position selection exceeds its total limit")
    if not require_bool(selection, "requireEligible", "start positions.selection"):
        raise ContractError("start positions must require eligible Phase 3 rows")
    if not require_bool(selection, "resetMoveNumber", "start positions.selection"):
        raise ContractError("start positions must reset the SFEN move number")
    require_int(
        selection,
        "excludedCrossSplitStates",
        "start positions.selection",
        minimum=0,
        maximum=MAX_PHASE3_SOURCE_POSITIONS,
    )
    rows = require_list(
        root, "positions", "start positions", minimum_items=1, maximum_items=MAX_START_POSITIONS
    )
    positions: list[StartPosition] = []
    identifiers: set[str] = set()
    sfens: set[str] = set()
    game_splits: dict[str, str] = {}
    for index, raw_position in enumerate(rows):
        context = f"positions[{index}]"
        table = require_mapping(raw_position, context)
        require_exact_keys(table, _START_POSITION_KEYS, context)
        position_id = require_identifier(table, "positionId", context)
        if position_id in identifiers:
            raise ContractError(f"duplicate start position ID: {position_id}")
        identifiers.add(position_id)
        sfen = _require_sfen(table, context)
        if sfen in sfens:
            raise ContractError(f"duplicate canonical start SFEN: {position_id}")
        sfens.add(sfen)
        source_game = require_sha256(table, "sourceGameSha256", context)
        split = require_enum(table, "split", context, {"train", "validation"})
        position_index = require_int(table, "positionIndex", context, minimum=0, maximum=100_000)
        expected_position_id = start_position_identity(source_game, position_index, sfen)
        if position_id != expected_position_id:
            raise ContractError(f"{context}.positionId disagrees with its source position")
        previous_split = game_splits.setdefault(source_game, split)
        if previous_split != split:
            raise ContractError("one source game appears in multiple start-position splits")
        positions.append(
            StartPosition(
                position_id=position_id,
                sfen=sfen,
                source_game_sha256=source_game,
                position_index=position_index,
                split=split,
            )
        )
    observed_counts = {
        "train": sum(position.split == "train" for position in positions),
        "validation": sum(position.split == "validation" for position in positions),
    }
    if observed_counts != {"train": train_count, "validation": validation_count}:
        raise ContractError("start-position selection counts disagree with its positions")
    return StartPositionSet(
        dataset_manifest=dataset_manifest,
        source_positions=source_positions,
        positions=tuple(sorted(positions, key=lambda item: item.position_id)),
    )


def build_start_position_validation_plan(
    *,
    start_positions_ref: ArtifactRef,
    positions: StartPositionSet,
    engine_ref: ArtifactRef,
    git_commit: str,
    engine_cli: str,
    engine_build_receipt: ArtifactRef | None = None,
) -> dict[str, object]:
    """Build bounded Rust legality checks without implementing shogi rules in Python."""

    _validate_git_commit(git_commit)
    engine_cli = validate_relative_path(engine_cli)
    if engine_cli != engine_ref.path:
        raise ContractError("validation command must use the immutable engine artifact path")
    checks = []
    for position in positions.positions:
        checks.append(
            {
                "positionId": position.position_id,
                "sfenSha256": hashlib.sha256(position.sfen.encode("utf-8")).hexdigest(),
                "argv": [engine_cli, "perft", "--depth", "0", "--sfen", position.sfen],
                "timeoutSeconds": 30,
            }
        )
    return {
        "schema": "phase6_start_position_validation_plan/v1",
        "startPositions": start_positions_ref.as_dict(),
        "engine": engine_ref.as_dict(),
        "engineBuildReceipt": (
            engine_build_receipt.as_dict() if engine_build_receipt is not None else None
        ),
        "gitCommit": git_commit,
        "method": "open-shogi-cli-perft-depth-0",
        "checks": checks,
    }


def validate_start_position_validation(
    raw: object,
    *,
    start_positions_ref: ArtifactRef,
    engine_ref: ArtifactRef,
    positions: StartPositionSet,
    repository_root: Path | None = None,
    engine_build_receipt: ArtifactRef | None = None,
    runtime_authorization: bool = False,
) -> None:
    root = require_mapping(raw, "start-position validation")
    require_exact_keys(root, _VALIDATION_ROOT_KEYS, "start-position validation")
    if root.get("schema") != START_VALIDATION_SCHEMA:
        raise ContractError("unsupported start-position validation schema")
    if ArtifactRef.from_dict(root.get("startPositions"), "startPositions") != start_positions_ref:
        raise ContractError("start-position validation references a different input")
    if ArtifactRef.from_dict(root.get("engine"), "engine") != engine_ref:
        raise ContractError("start-position validation references a different engine")
    receipt_raw = root.get("engineBuildReceipt")
    receipt = (
        ArtifactRef.from_dict(receipt_raw, "engineBuildReceipt")
        if receipt_raw is not None
        else None
    )
    if engine_build_receipt is not None and receipt != engine_build_receipt:
        raise ContractError("start-position validation references a different build receipt")
    git_commit = require_string(root, "gitCommit", "start-position validation")
    _validate_git_commit(git_commit)
    if root.get("method") != "open-shogi-cli-perft-depth-0":
        raise ContractError("start-position validation has an unsupported method")
    if repository_root is not None:
        from .starts import validate_start_position_source_artifact

        verified_positions = validate_start_position_source_artifact(
            repository_root=repository_root,
            start_positions=load_json_artifact(repository_root, start_positions_ref),
            start_positions_ref=start_positions_ref,
        )
        if verified_positions != positions:
            raise ContractError("start-position validation received a different parsed start set")
        verify_artifact_ref(repository_root, engine_ref)
        if receipt is not None and runtime_authorization:
            _revalidate_start_sfens(
                repository_root,
                positions=positions,
                engine_ref=engine_ref,
                engine_build_receipt=receipt,
                git_commit=git_commit,
            )
    rows = require_list(
        root,
        "results",
        "start-position validation",
        minimum_items=len(positions.positions),
        maximum_items=len(positions.positions),
    )
    expected = {
        item.position_id: hashlib.sha256(item.sfen.encode("utf-8")).hexdigest()
        for item in positions.positions
    }
    observed: dict[str, str] = {}
    for index, row in enumerate(rows):
        context = f"validation.results[{index}]"
        table = require_mapping(row, context)
        require_exact_keys(table, _VALIDATION_RESULT_KEYS, context)
        position_id = require_identifier(table, "positionId", context)
        if table.get("legal") is not True:
            raise ContractError(f"start position was not proven legal: {position_id}")
        return_code = require_int(table, "returnCode", context, minimum=-255, maximum=255)
        timed_out = require_bool(table, "timedOut", context)
        output_limit = require_bool(table, "outputLimitExceeded", context)
        memory_limit = require_bool(table, "memoryLimitExceeded", context)
        peak_rss = table.get("peakRssBytes")
        if peak_rss is not None:
            require_int(table, "peakRssBytes", context, minimum=0, maximum=2**63 - 1)
        measurement = require_enum(
            table,
            "rssMeasurement",
            context,
            {
                "process_tree_ps_rss_sum",
                "process_tree_ps_short_lived_no_sample",
                "unavailable",
            },
        )
        if return_code != 0 or timed_out or output_limit or memory_limit:
            raise ContractError(f"start-position validation execution failed: {position_id}")
        if measurement == "process_tree_ps_rss_sum":
            if peak_rss is None or peak_rss > 1024 * 1024 * 1024:
                raise ContractError(
                    f"start-position validation RSS evidence is invalid: {position_id}"
                )
        elif measurement == "process_tree_ps_short_lived_no_sample":
            if peak_rss != 0:
                raise ContractError(
                    f"start-position validation short-lived RSS must be zero: {position_id}"
                )
        else:
            raise ContractError(
                f"start-position validation cannot succeed without RSS evidence: {position_id}"
            )
        stdout = ArtifactRef.from_dict(table.get("stdout"), f"{context}.stdout")
        stderr = ArtifactRef.from_dict(table.get("stderr"), f"{context}.stderr")
        validate_utc_timestamp(table.get("completedAt"), f"{context}.completedAt")
        if repository_root is not None:
            verify_artifact_ref(repository_root, stdout)
            verify_artifact_ref(repository_root, stderr)
        digest = require_sha256(table, "sfenSha256", context)
        if position_id in observed:
            raise ContractError(f"duplicate validation result: {position_id}")
        observed[position_id] = digest
    if observed != expected:
        raise ContractError("start-position validation results do not match the input set")


def _revalidate_start_sfens(
    repository_root: Path,
    *,
    positions: StartPositionSet,
    engine_ref: ArtifactRef,
    engine_build_receipt: ArtifactRef,
    git_commit: str,
) -> None:
    """Parse every persisted start again through the exact receipt-bound Rust CLI.

    Validation JSON is durable evidence, not an authorization oracle: an attacker
    could otherwise mark an impossible SFEN legal and rehash the surrounding
    artifacts.  This bounded check performs no arena/self-play work and publishes
    no output; it only runs `perft --depth 0` against a private executable snapshot.
    """

    from open_shogi_training.labeling.execution import (
        ExecutableSnapshot,
        ExecutableSnapshotError,
    )

    from .common import ensure_contained_directory
    from .engine_receipt import validate_engine_build_receipt
    from .execution import _run_bounded_process

    root = repository_root.resolve(strict=True)
    validate_engine_build_receipt(
        root,
        engine_build_receipt,
        expected_engine=engine_ref,
        expected_git_commit=git_commit,
    )
    verify_artifact_ref(root, engine_ref)
    from .common import contained_path

    source = contained_path(root, engine_ref.path, must_exist=True)
    try:
        snapshot = ExecutableSnapshot.create(
            source,
            temporary_directory=ensure_contained_directory(root, "local/runtime-snapshots"),
            max_bytes=512 * 1024 * 1024,
            expected_sha256=engine_ref.sha256,
        )
    except ExecutableSnapshotError as error:
        raise ContractError(f"cannot snapshot start-position legality engine: {error}") from error
    try:
        for position in positions.positions:
            (
                _,
                stderr,
                return_code,
                timed_out,
                output_limit,
                memory_limit,
                _,
                rss_measurement,
            ) = _run_bounded_process(
                [
                    snapshot.executable_path,
                    "perft",
                    "--depth",
                    "0",
                    "--sfen",
                    position.sfen,
                ],
                cwd=root,
                timeout_seconds=30,
                pass_fds=snapshot.pass_fds(),
                memory_limit_bytes=1024 * 1024 * 1024,
                launch_guard=lambda: (
                    snapshot.assert_snapshot_unchanged(),
                    snapshot.assert_source_unchanged(),
                ),
            )
            snapshot.assert_snapshot_unchanged()
            snapshot.assert_source_unchanged()
            if (
                return_code != 0
                or timed_out
                or output_limit
                or memory_limit
                or rss_measurement == "unavailable"
            ):
                detail = stderr.decode("utf-8", errors="replace")[-2_048:]
                raise ContractError(
                    "receipt-bound Rust legality revalidation rejected start position "
                    f"{position.position_id}: {detail}"
                )
    finally:
        snapshot.close()
    from .starts import revalidate_phase3_source_with_engine

    revalidate_phase3_source_with_engine(
        repository_root=root,
        start_positions=positions,
        engine_ref=engine_ref,
        engine_build_receipt=engine_build_receipt,
        git_commit=git_commit,
    )


def build_selfplay_plan(
    *,
    generation_id: str,
    champion: ModelSpec,
    engine: ArtifactRef,
    model_registry: ArtifactRef,
    git_commit: str,
    config: SelfPlayConfig,
    config_ref: ArtifactRef,
    start_positions: StartPositionSet,
    start_positions_ref: ArtifactRef,
    validation_ref: ArtifactRef,
    engine_build_receipt: ArtifactRef | None = None,
) -> dict[str, object]:
    generation_id = validate_identifier(generation_id, "generation_id")
    if champion.evaluator_kind != "neural":
        raise ContractError("Phase 6 self-play requires a neural champion")
    _validate_git_commit(git_commit)
    selected = _select_start_positions(
        tuple(position for position in start_positions.positions if position.split == "train"),
        count=config.run.start_set_pairs,
        seed=config.run.seed,
    )
    jobs = _paired_jobs(
        kind="selfplay",
        generation_id=generation_id,
        model_a=champion,
        model_b=champion,
        config=config,
        engine_path=engine.path,
        git_commit=git_commit,
        selected_positions=selected,
    )
    plan: dict[str, object] = {
        "schema": SELFPLAY_PLAN_SCHEMA,
        "generationId": generation_id,
        "champion": champion.as_dict(),
        "engine": engine.as_dict(),
        "engineBuildReceipt": (
            engine_build_receipt.as_dict() if engine_build_receipt is not None else None
        ),
        "modelRegistry": model_registry.as_dict(),
        "gitCommit": git_commit,
        "config": config_ref.as_dict(),
        "configSha256": config.sha256,
        "startPositions": start_positions_ref.as_dict(),
        "startPositionValidation": validation_ref.as_dict(),
        "datasetManifest": start_positions.dataset_manifest.as_dict(),
        "gameCount": config.run.games,
        "pairCount": config.run.games // 2,
        "normalStartPairs": config.run.normal_start_pairs,
        "startSetPairs": config.run.start_set_pairs,
        "seed": config.run.seed,
        "nodesPerMove": config.run.nodes_per_move,
        "maxWorkers": config.resources.workers,
        "memoryLimitMiB": config.resources.memory_limit_mib,
        "memoryPerWorkerMiB": config.resources.memory_per_worker_mib,
        "jobs": jobs,
    }
    plan["planSha256"] = canonical_sha256(plan)
    return plan


def build_arena_plan(
    *,
    generation_id: str,
    champion: ModelSpec,
    challenger: ModelSpec,
    engine: ArtifactRef,
    model_registry: ArtifactRef,
    git_commit: str,
    config: SelfPlayConfig,
    config_ref: ArtifactRef,
    start_positions: StartPositionSet,
    start_positions_ref: ArtifactRef,
    validation_ref: ArtifactRef,
    engine_build_receipt: ArtifactRef | None = None,
) -> dict[str, object]:
    generation_id = validate_identifier(generation_id, "generation_id")
    if champion.model_id == challenger.model_id:
        raise ContractError("arena champion and challenger must be different models")
    if champion.evaluator_kind != "neural" or challenger.evaluator_kind != "neural":
        raise ContractError("Phase 6 arena requires neural champion and challenger models")
    _validate_git_commit(git_commit)
    validation_positions = tuple(
        position for position in start_positions.positions if position.split == "validation"
    )
    selected = _select_start_positions(
        validation_positions,
        count=config.run.start_set_pairs,
        seed=config.run.seed ^ 0x4152_454E_41,
    )
    jobs = _paired_jobs(
        kind="arena",
        generation_id=generation_id,
        model_a=challenger,
        model_b=champion,
        config=config,
        engine_path=engine.path,
        git_commit=git_commit,
        selected_positions=selected,
    )
    plan: dict[str, object] = {
        "schema": ARENA_PLAN_SCHEMA,
        "generationId": generation_id,
        "champion": champion.as_dict(),
        "challenger": challenger.as_dict(),
        "engine": engine.as_dict(),
        "engineBuildReceipt": (
            engine_build_receipt.as_dict() if engine_build_receipt is not None else None
        ),
        "modelRegistry": model_registry.as_dict(),
        "gitCommit": git_commit,
        "config": config_ref.as_dict(),
        "configSha256": config.sha256,
        "startPositions": start_positions_ref.as_dict(),
        "startPositionValidation": validation_ref.as_dict(),
        "datasetManifest": start_positions.dataset_manifest.as_dict(),
        "gameCount": config.run.games,
        "pairCount": config.run.games // 2,
        "normalStartPairs": config.run.normal_start_pairs,
        "startSetPairs": config.run.start_set_pairs,
        "seed": config.run.seed,
        "nodesPerMove": config.run.nodes_per_move,
        "maxWorkers": config.resources.workers,
        "memoryLimitMiB": config.resources.memory_limit_mib,
        "memoryPerWorkerMiB": config.resources.memory_per_worker_mib,
        "jobs": jobs,
    }
    plan["planSha256"] = canonical_sha256(plan)
    return plan


def validate_training_plan(raw: object) -> Mapping[str, Any]:
    """Validate the complete immutable challenger-training command contract."""

    root = require_mapping(raw, "training plan")
    require_exact_keys(root, _TRAINING_PLAN_KEYS, "training plan")
    if root.get("schema") != TRAINING_PLAN_SCHEMA:
        raise ContractError("unsupported challenger-training plan schema")
    require_identifier(root, "generationId", "training plan")
    parent = _parse_model_spec(root.get("parentModel"), "training plan.parentModel")
    inputs = require_mapping(root.get("inputs"), "training plan.inputs")
    require_exact_keys(inputs, _TRAINING_INPUT_KEYS, "training plan.inputs")
    references = {
        key: ArtifactRef.from_dict(inputs.get(key), f"training plan.inputs.{key}")
        for key in _TRAINING_INPUT_KEYS - {"resumeCheckpoint"}
    }
    resume_raw = inputs.get("resumeCheckpoint")
    resume = (
        ArtifactRef.from_dict(resume_raw, "training plan.inputs.resumeCheckpoint")
        if resume_raw is not None
        else None
    )
    if len(set(references.values())) != len(references):
        raise ContractError("training plan input artifacts must be distinct")
    output_dir = require_relative_path(root, "outputDir", "training plan")
    if resume is not None and resume.path != f"{output_dir}/last.pt":
        raise ContractError("training resume checkpoint must be OUTPUT_DIR/last.pt")
    command = require_mapping(root.get("command"), "training plan.command")
    require_exact_keys(command, _PLAN_COMMAND_KEYS, "training plan.command")
    if command.get("kind") != "model_cli":
        raise ContractError("training plan command kind must be model_cli")
    expected_argv = [
        "python",
        "-m",
        "open_shogi_training.models",
        "train",
        "--features",
        references["featuresConfig"].path,
        "--model",
        references["modelConfig"].path,
        "--training",
        references["trainingConfig"].path,
        "--labels",
        references["teacherLabels"].path,
        "--label-manifest",
        references["teacherLabelManifest"].path,
        "--positions",
        references["positions"].path,
        "--dataset-manifest",
        references["datasetManifest"].path,
        "--replay-manifest",
        references["replayManifest"].path,
        "--output-dir",
        output_dir,
    ]
    if resume is not None:
        expected_argv.extend(["--resume", resume.path])
    if command.get("argv") != expected_argv:
        raise ContractError("training plan argv does not exactly match its artifact inputs")
    require_int(
        command,
        "timeoutSeconds",
        "training plan.command",
        minimum=1,
        maximum=172_800,
    )
    _validate_plan_hash(root, "training plan")
    # Parse the artifact even though only the evaluator kind is consumed below; this
    # keeps the parent model's identity within the same closed validator.
    if parent.evaluator_kind != "neural":
        raise ContractError("Phase 6 challenger training requires a neural parent model")
    return root


def validate_teacher_labeling_plan(raw: object) -> Mapping[str, Any]:
    """Validate a bounded supplemental teacher command for pre-cap runs only."""

    root = require_mapping(raw, "teacher labeling plan")
    require_exact_keys(root, _TEACHER_PLAN_KEYS, "teacher labeling plan")
    if root.get("schema") != TEACHER_PLAN_SCHEMA:
        raise ContractError("unsupported supplemental-teacher plan schema")
    require_identifier(root, "generationId", "teacher labeling plan")
    references = {
        key: ArtifactRef.from_dict(root.get(key), f"teacher labeling plan.{key}")
        for key in (
            "hardPositions",
            "normalizedPositions",
            "datasetManifest",
            "labelingConfig",
            "benchmarkReport",
        )
    }
    if len(set(references.values())) != len(references):
        raise ContractError("teacher labeling plan artifacts must be distinct")
    output_dir = require_relative_path(root, "outputDir", "teacher labeling plan")
    target = require_int(
        root,
        "targetCompleted",
        "teacher labeling plan",
        minimum=1,
        maximum=10_000,
    )
    if target not in {10, 100, 1_000, 10_000}:
        raise ContractError("teacher target must be an approved cumulative milestone")
    limit = require_int(
        root,
        "teacherLabelLimit",
        "teacher labeling plan",
        minimum=1,
        maximum=10_000,
    )
    labels_before = require_int(
        root,
        "labelsBefore",
        "teacher labeling plan",
        minimum=0,
        maximum=limit,
    )
    if labels_before == limit or not labels_before < target <= limit:
        raise ContractError("teacher target must advance within the global label budget")
    command = require_mapping(root.get("command"), "teacher labeling plan.command")
    require_exact_keys(command, _PLAN_COMMAND_KEYS, "teacher labeling plan.command")
    if command.get("kind") != "labeling_cli":
        raise ContractError("teacher labeling plan command kind must be labeling_cli")
    expected_argv = [
        "python",
        "-m",
        "open_shogi_training.labeling",
        "label",
        "--config",
        references["labelingConfig"].path,
        "--project-root",
        ".",
        "--positions",
        references["normalizedPositions"].path,
        "--dataset-manifest",
        references["datasetManifest"].path,
        "--benchmark-report",
        references["benchmarkReport"].path,
        "--output-dir",
        output_dir,
        "--target-completed",
        str(target),
    ]
    if command.get("argv") != expected_argv:
        raise ContractError("teacher labeling argv does not exactly match its artifact inputs")
    require_int(
        command,
        "timeoutSeconds",
        "teacher labeling plan.command",
        minimum=1,
        maximum=172_800,
    )
    _validate_plan_hash(root, "teacher labeling plan")
    return root


def build_training_plan(
    *,
    generation_id: str,
    parent_model: ModelSpec,
    replay_manifest: ArtifactRef,
    labels: ArtifactRef,
    label_manifest: ArtifactRef,
    positions: ArtifactRef,
    dataset_manifest: ArtifactRef,
    features_config: ArtifactRef,
    model_config: ArtifactRef,
    training_config: ArtifactRef,
    output_dir: str,
    timeout_seconds: int,
    resume_checkpoint: ArtifactRef | None = None,
) -> dict[str, object]:
    """Plan the stable model CLI command; this module never imports training internals."""

    generation_id = validate_identifier(generation_id, "generation_id")
    output_dir = validate_relative_path(output_dir)
    if not 1 <= timeout_seconds <= 172_800:
        raise ContractError("training timeout_seconds must be in 1..172800")
    if resume_checkpoint is not None and resume_checkpoint.path != f"{output_dir}/last.pt":
        raise ContractError("training resume checkpoint must be OUTPUT_DIR/last.pt")
    argv = [
        "python",
        "-m",
        "open_shogi_training.models",
        "train",
        "--features",
        features_config.path,
        "--model",
        model_config.path,
        "--training",
        training_config.path,
        "--labels",
        labels.path,
        "--label-manifest",
        label_manifest.path,
        "--positions",
        positions.path,
        "--dataset-manifest",
        dataset_manifest.path,
        "--replay-manifest",
        replay_manifest.path,
        "--output-dir",
        output_dir,
    ]
    if resume_checkpoint is not None:
        argv.extend(["--resume", resume_checkpoint.path])
    plan: dict[str, object] = {
        "schema": TRAINING_PLAN_SCHEMA,
        "generationId": generation_id,
        "parentModel": parent_model.as_dict(),
        "inputs": {
            "replayManifest": replay_manifest.as_dict(),
            "teacherLabels": labels.as_dict(),
            "teacherLabelManifest": label_manifest.as_dict(),
            "positions": positions.as_dict(),
            "datasetManifest": dataset_manifest.as_dict(),
            "featuresConfig": features_config.as_dict(),
            "modelConfig": model_config.as_dict(),
            "trainingConfig": training_config.as_dict(),
            "resumeCheckpoint": resume_checkpoint.as_dict()
            if resume_checkpoint is not None
            else None,
        },
        "outputDir": output_dir,
        "command": {
            "kind": "model_cli",
            "argv": argv,
            "timeoutSeconds": timeout_seconds,
        },
    }
    plan["planSha256"] = canonical_sha256(plan)
    validate_training_plan(plan)
    return plan


def build_teacher_labeling_plan(
    *,
    generation_id: str,
    hard_positions: ArtifactRef,
    normalized_positions: ArtifactRef,
    dataset_manifest: ArtifactRef,
    labeling_config: ArtifactRef,
    benchmark_report: ArtifactRef,
    output_dir: str,
    target_completed: int,
    labels_before: int,
    teacher_label_limit: int,
    timeout_seconds: int,
) -> dict[str, object]:
    """Plan supplemental labels through the existing strict teacher-label CLI.

    ``normalized_positions`` must use the teacher pipeline's accepted
    ``phase3_position/v1`` gzip contract. The hard-position manifest is retained as
    selection provenance and is never injected into an existing append-only label log.
    """

    generation_id = validate_identifier(generation_id, "generation_id")
    output_dir = validate_relative_path(output_dir)
    if not 1 <= teacher_label_limit <= 10_000:
        raise ContractError("teacher_label_limit must be in 1..10000")
    if not 0 <= labels_before <= teacher_label_limit:
        raise ContractError("labels_before is outside the teacher-label budget")
    if labels_before == teacher_label_limit:
        raise ContractError("teacher label cap is exhausted; reuse the existing label manifest")
    if target_completed not in {10, 100, 1_000, 10_000}:
        raise ContractError("target_completed must be one of 10, 100, 1000, or 10000")
    if not labels_before < target_completed <= teacher_label_limit:
        raise ContractError(
            "target_completed must advance the cumulative count within the teacher-label limit"
        )
    if not 1 <= timeout_seconds <= 172_800:
        raise ContractError("teacher timeout_seconds must be in 1..172800")
    argv = [
        "python",
        "-m",
        "open_shogi_training.labeling",
        "label",
        "--config",
        labeling_config.path,
        "--project-root",
        ".",
        "--positions",
        normalized_positions.path,
        "--dataset-manifest",
        dataset_manifest.path,
        "--benchmark-report",
        benchmark_report.path,
        "--output-dir",
        output_dir,
        "--target-completed",
        str(target_completed),
    ]
    plan: dict[str, object] = {
        "schema": TEACHER_PLAN_SCHEMA,
        "generationId": generation_id,
        "hardPositions": hard_positions.as_dict(),
        "normalizedPositions": normalized_positions.as_dict(),
        "datasetManifest": dataset_manifest.as_dict(),
        "labelingConfig": labeling_config.as_dict(),
        "benchmarkReport": benchmark_report.as_dict(),
        "outputDir": output_dir,
        "targetCompleted": target_completed,
        "labelsBefore": labels_before,
        "teacherLabelLimit": teacher_label_limit,
        "command": {
            "kind": "labeling_cli",
            "argv": argv,
            "timeoutSeconds": timeout_seconds,
        },
    }
    plan["planSha256"] = canonical_sha256(plan)
    validate_teacher_labeling_plan(plan)
    return plan


def _paired_jobs(
    *,
    kind: str,
    generation_id: str,
    model_a: ModelSpec,
    model_b: ModelSpec,
    config: SelfPlayConfig,
    engine_path: str,
    git_commit: str,
    selected_positions: Sequence[StartPosition],
) -> list[dict[str, object]]:
    starts: list[tuple[str, str, str]] = [
        ("initial", "standard-initial", INITIAL_SFEN) for _ in range(config.run.normal_start_pairs)
    ]
    starts.extend(("start_set", item.position_id, item.sfen) for item in selected_positions)
    jobs: list[dict[str, object]] = []
    output_base = f"{config.paths.output_root}/{generation_id}/{kind}"
    for pair_index, (start_group, start_id, sfen) in enumerate(starts):
        job_id = f"pair-{pair_index:04d}"
        job_seed = _derive_seed(config.run.seed, kind, pair_index, start_id)
        output_dir = f"{output_base}/jobs/{job_id}"
        command = [
            engine_path,
            "arena",
            "--games",
            "2",
            "--player-a",
            _arena_player(model_a),
            "--player-b",
            _arena_player(model_b),
            "--a-depth",
            str(config.run.search_depth),
            "--b-depth",
            str(config.run.search_depth),
            "--a-hash-mb",
            str(config.run.hash_mib),
            "--b-hash-mb",
            str(config.run.hash_mib),
            "--nodes",
            str(config.run.nodes_per_move),
            "--max-plies",
            str(config.run.max_plies),
            "--seed",
            str(job_seed),
            "--sfen",
            sfen,
            "--git-commit",
            git_commit,
            "--output-dir",
            output_dir,
        ]
        if model_a.evaluator_kind == "neural":
            command.extend(["--a-model", model_a.artifact.path])
        if model_b.evaluator_kind == "neural":
            command.extend(["--b-model", model_b.artifact.path])
        jobs.append(
            {
                "jobId": job_id,
                "pairIndex": pair_index,
                "startGroup": start_group,
                "startPositionId": start_id,
                "sfen": sfen,
                "seed": job_seed,
                "gameIds": [f"game-{pair_index * 2:06d}", f"game-{pair_index * 2 + 1:06d}"],
                "modelAColorOrder": ["black", "white"],
                "outputDir": output_dir,
                "reportPath": f"{output_dir}/arena-report.json",
                "csaPaths": [
                    f"{output_dir}/games/game-000001.csa",
                    f"{output_dir}/games/game-000002.csa",
                ],
                "quarantinePath": f"{output_base}/quarantine/{job_id}.json",
                "command": {
                    "kind": "engine_arena",
                    "argv": command,
                    "timeoutSeconds": config.resources.game_timeout_seconds,
                },
            }
        )
    return jobs


def _select_start_positions(
    positions: Sequence[StartPosition], *, count: int, seed: int
) -> tuple[StartPosition, ...]:
    if len(positions) < count:
        raise ContractError(
            f"provided start-position split has {len(positions)} unique positions; {count} required"
        )
    ordered = sorted(
        positions,
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item.position_id}".encode()).digest(),
            item.position_id,
        ),
    )
    return tuple(ordered[:count])


def start_position_identity(source_game_sha256: str, position_index: int, sfen: str) -> str:
    """Derive the immutable identity used by Phase 6 start-position artifacts."""

    return hashlib.sha256(
        b"phase6_start_position/v1\0"
        + bytes.fromhex(source_game_sha256)
        + position_index.to_bytes(8, "big")
        + b"\0"
        + sfen.encode("utf-8")
    ).hexdigest()


def _derive_seed(base_seed: int, kind: str, pair_index: int, start_id: str) -> int:
    digest = hashlib.sha256(
        f"phase6\0{base_seed}\0{kind}\0{pair_index}\0{start_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (MAX_JSON_SAFE_INTEGER + 1)


def _arena_player(model: ModelSpec) -> str:
    return model.evaluator_kind


def _parse_model_spec(raw: object, context: str) -> ModelSpec:
    table = require_mapping(raw, context)
    require_exact_keys(table, _MODEL_SPEC_KEYS, context)
    return ModelSpec(
        model_id=require_identifier(table, "modelId", context),
        artifact=ArtifactRef.from_dict(table.get("artifact"), f"{context}.artifact"),
        evaluator_kind=require_enum(
            table,
            "evaluatorKind",
            context,
            {"material", "handcrafted-baseline", "handcrafted-experimental", "neural"},
        ),
    )


def _validate_plan_hash(root: Mapping[str, Any], context: str) -> None:
    expected = require_sha256(root, "planSha256", context)
    without_hash = dict(root)
    without_hash.pop("planSha256")
    if canonical_sha256(without_hash) != expected:
        raise ContractError(f"{context} self-hash mismatch")


def _require_sfen(table: Mapping[str, Any], context: str) -> str:
    sfen = require_string(table, "sfen", context, maximum_length=512)
    fields = sfen.split(" ")
    if (
        not sfen.isascii()
        or len(fields) != 4
        or any(not field for field in fields)
        or fields[1] not in {"b", "w"}
        or any(character in sfen for character in "\r\n")
    ):
        raise ContractError(f"{context}.sfen must have four non-empty fields")
    if fields[3] != "1":
        raise ContractError(f"{context}.sfen move number must be 1 for canonical CSA output")
    return sfen


def _validate_git_commit(value: str) -> None:
    if _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ContractError("git_commit must be a 7..64 character lowercase hexadecimal object ID")
