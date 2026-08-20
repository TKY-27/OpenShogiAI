"""Derive factual Phase 6 evidence from completed self-play and teacher labels."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import iter_jsonl_descriptor_records
from open_shogi_training.labeling.schema import iter_teacher_labels_descriptor, position_id
from open_shogi_training.models.dataset import _iter_jsonl_gzip_descriptor

from .arena import validate_phase6_pair_report_binding
from .common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    contained_path,
    ensure_contained_directory,
    load_json_artifact,
    require_bool,
    require_clean_head,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_list,
    require_mapping,
    require_sha256,
    require_string,
    validate_relative_path,
    verified_artifact_descriptor,
    verify_artifact_ref,
)
from .evidence import POSITION_EVIDENCE_SCHEMA
from .execution import (
    SELFPLAY_MANIFEST_SCHEMA,
    CommandRunner,
    validate_execution_manifest,
    validate_paired_plan,
)
from .planning import SELFPLAY_PLAN_SCHEMA, parse_start_positions
from .starts import _parse_phase3_position, validate_phase3_dataset_binding

MAX_EXPORT_BYTES: Final = 128 * 1024 * 1024
MAX_EXPORT_LINE_BYTES: Final = 32 * 1024 * 1024
MAX_SELFPLAY_PLIES: Final = 256
_EXPORT_KEYS = frozenset(
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
_PREDICTION_KEYS = frozenset(
    {
        "schema",
        "positionId",
        "canonicalSfen",
        "split",
        "stage",
        "teacherScore",
        "modelCp",
        "candidateGapCp",
        "bestmove",
        "recordedMove",
        "recordedMoveAgrees",
        "alreadyTeacherLabeled",
        "checkpointSha256",
        "configSha256",
    }
)


def build_position_evidence(
    *,
    repository_root: Path,
    selfplay_plan: object,
    selfplay_plan_ref: ArtifactRef,
    selfplay_manifest: object,
    selfplay_manifest_ref: ArtifactRef,
    teacher_labels_ref: ArtifactRef,
    model_predictions_ref: ArtifactRef,
    dataset_manifest_ref: ArtifactRef,
    generation_ordinal: int,
    teacher_source_generation_id: str,
    export_root: str,
    timeout_seconds: int,
    runner: CommandRunner,
) -> dict[str, object]:
    """Build strict evidence and outcome targets without inventing unavailable scores."""

    plan = validate_paired_plan(selfplay_plan)
    if isinstance(runner, CommandRunner) and runner.require_clean_repository:
        require_clean_head(repository_root, str(plan["gitCommit"]))
    if plan.get("schema") != SELFPLAY_PLAN_SCHEMA:
        raise ContractError("position evidence requires a self-play plan")
    execution = validate_execution_manifest(
        selfplay_manifest,
        repository_root=repository_root,
        manifest_ref=selfplay_manifest_ref,
        plan=plan,
        plan_ref=selfplay_plan_ref,
    )
    if execution.get("schema") != SELFPLAY_MANIFEST_SCHEMA:
        raise ContractError("position evidence requires a self-play execution manifest")
    if (
        ArtifactRef.from_dict(plan.get("datasetManifest"), "self-play plan.datasetManifest")
        != dataset_manifest_ref
    ):
        raise ContractError("teacher-label dataset manifest differs from the self-play dataset")
    if not 1 <= generation_ordinal <= 1_000_000:
        raise ContractError("generation ordinal must be in 1..1000000")
    teacher_source_generation_id = _identifier(
        teacher_source_generation_id, "teacher_source_generation_id"
    )
    export_root = validate_relative_path(export_root)
    if not 1 <= timeout_seconds <= 3_600:
        raise ContractError("CSA export timeout must be in 1..3600 seconds")
    verify_artifact_ref(repository_root, teacher_labels_ref)
    verify_artifact_ref(repository_root, dataset_manifest_ref)
    engine_ref = ArtifactRef.from_dict(plan.get("engine"), "self-play plan.engine")
    receipt_raw = plan.get("engineBuildReceipt")
    engine_receipt_ref = (
        ArtifactRef.from_dict(receipt_raw, "self-play plan.engineBuildReceipt")
        if receipt_raw is not None
        else None
    )
    verify_artifact_ref(repository_root, engine_ref)
    starts_ref = ArtifactRef.from_dict(plan.get("startPositions"), "self-play plan.startPositions")
    starts = parse_start_positions(load_json_artifact(repository_root, starts_ref))
    if starts.dataset_manifest != dataset_manifest_ref:
        raise ContractError("self-play starts and teacher labels use different Phase 3 data")

    teacher_rows, protected_splits, excluded_teacher_rows, predictions_applied = _teacher_evidence(
        repository_root=repository_root,
        labels_ref=teacher_labels_ref,
        predictions_ref=model_predictions_ref,
        dataset_manifest_ref=dataset_manifest_ref,
        source_positions_ref=starts.source_positions,
        source_generation_id=teacher_source_generation_id,
    )
    (
        selfplay_rows,
        export_runs,
        excluded_selfplay_rows,
        duplicate_selfplay_rows,
    ) = _selfplay_evidence(
        repository_root=repository_root,
        plan=plan,
        execution=execution,
        execution_ref=selfplay_manifest_ref,
        engine_ref=engine_ref,
        engine_receipt_ref=engine_receipt_ref,
        generation_ordinal=generation_ordinal,
        export_root=export_root,
        timeout_seconds=timeout_seconds,
        runner=runner,
        protected_splits=protected_splits,
    )
    positions = [*teacher_rows, *selfplay_rows]
    positions.sort(
        key=lambda row: (
            int(row["generationOrdinal"]),
            str(row["sourceType"]),
            str(row["positionId"]),
        )
    )
    sources = sorted(
        {teacher_labels_ref, model_predictions_ref, selfplay_manifest_ref},
        key=lambda reference: reference.path,
    )
    return {
        "schema": POSITION_EVIDENCE_SCHEMA,
        "generationId": plan["generationId"],
        "derivation": {
            "selfplayPlan": selfplay_plan_ref.as_dict(),
            "selfplayManifest": selfplay_manifest_ref.as_dict(),
            "teacherLabels": teacher_labels_ref.as_dict(),
            "modelPredictions": model_predictions_ref.as_dict(),
            "datasetManifest": dataset_manifest_ref.as_dict(),
            "generationOrdinal": generation_ordinal,
            "teacherSourceGenerationId": teacher_source_generation_id,
            "csaExports": export_runs,
            "counts": {
                "teacherPositions": len(teacher_rows),
                "selfplayPositions": len(selfplay_rows),
                "modelPredictionsApplied": predictions_applied,
                "excludedCrossSplitTeacherRows": excluded_teacher_rows,
                "excludedProtectedSelfplayRows": excluded_selfplay_rows,
                "excludedDuplicateSelfplayRows": duplicate_selfplay_rows,
            },
            "unavailableMeasurements": [
                "selfplay_teacher_before_after_cp",
                "selfplay_teacher_cp",
                "selfplay_model_cp",
                "selfplay_champion_challenger_moves",
                "selfplay_candidate_gap_cp",
                "selfplay_mate_distance",
                "selfplay_actual_search_nodes",
            ],
        },
        "sourceManifests": [reference.as_dict() for reference in sources],
        "positions": positions,
    }


def _teacher_evidence(
    *,
    repository_root: Path,
    labels_ref: ArtifactRef,
    predictions_ref: ArtifactRef,
    dataset_manifest_ref: ArtifactRef,
    source_positions_ref: ArtifactRef,
    source_generation_id: str,
) -> tuple[list[dict[str, object]], dict[str, str], int, int]:
    try:
        with verified_artifact_descriptor(
            repository_root,
            labels_ref,
            maximum_bytes=4 * 1024 * 1024 * 1024,
        ) as descriptor:
            labels = list(
                iter_teacher_labels_descriptor(
                    descriptor,
                    display_path=Path(labels_ref.path),
                    expected_dataset_manifest_sha256=dataset_manifest_ref.sha256,
                    max_records=10_000,
                )
            )
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read strict teacher labels: {error}") from error
    with verified_artifact_descriptor(
        repository_root, predictions_ref, maximum_bytes=128 * 1024 * 1024
    ) as descriptor:
        predictions = _load_model_predictions_descriptor(descriptor, predictions_ref.path)
    labels_by_id = {str(label["position_id"]): label for label in labels}
    recorded_moves = _load_phase3_recorded_moves(
        repository_root=repository_root,
        positions_ref=source_positions_ref,
        dataset_manifest_ref=dataset_manifest_ref,
        labels_by_id=labels_by_id,
    )
    unknown_predictions = set(predictions) - set(labels_by_id)
    if unknown_predictions:
        raise ContractError("model predictions contain positions absent from teacher labels")
    for prediction_position_id, prediction in predictions.items():
        _validate_prediction_against_label(
            prediction,
            labels_by_id[prediction_position_id],
            recorded_move=recorded_moves[prediction_position_id],
            context=f"model prediction {prediction_position_id}",
        )
    splits_by_state: dict[str, set[str]] = {}
    for label in labels:
        canonical_sfen = _canonical_sfen(str(label["canonical_sfen"]))
        splits_by_state.setdefault(canonical_sfen, set()).add(str(label["split"]))
    cross_split_states = {state for state, splits in splits_by_state.items() if len(splits) > 1}
    protected = {
        state: next(iter(splits)) for state, splits in splits_by_state.items() if len(splits) == 1
    }
    rows: list[dict[str, object]] = []
    excluded = 0
    predictions_applied = 0
    for label in labels:
        canonical_sfen = _canonical_sfen(str(label["canonical_sfen"]))
        if canonical_sfen in cross_split_states:
            excluded += 1
            continue
        score = require_mapping(label.get("score"), "teacher label.score")
        score_kind = require_enum(score, "kind", "teacher label.score", {"cp", "mate"})
        score_value = require_int(
            score,
            "value",
            "teacher label.score",
            minimum=-(2**31),
            maximum=2**31 - 1,
        )
        outcome_kind = str(label["outcome"])
        side = str(label["side_to_move"])
        prediction = predictions.get(str(label["position_id"]))
        predictions_applied += int(prediction is not None)
        rows.append(
            {
                "positionId": label["position_id"],
                "sfen": canonical_sfen,
                "split": label["split"],
                "sourceGenerationId": source_generation_id,
                "generationOrdinal": 0,
                "sourceType": "teacher",
                "sourceManifest": labels_ref.as_dict(),
                "sourceGameId": label["game_id"],
                "sourcePly": label["position_index"],
                "sideToMove": side,
                "outcomeKind": outcome_kind,
                "outcomeTarget": _outcome_target(outcome_kind, side),
                "teacherBeforeCp": None,
                "teacherAfterCp": None,
                "teacherCp": score_value if score_kind == "cp" else None,
                "modelCp": prediction["modelCp"] if prediction is not None else None,
                "championMove": None,
                "challengerMove": None,
                "candidateGapCp": _candidate_gap(label),
                "mateDistance": score_value if score_kind == "mate" else None,
                "phase": label["stage"],
                "terminalBoundary": False,
                "searchNodes": label["nodes"],
                "suspectedFailure": "none",
                "alreadyTeacherLabeled": True,
            }
        )
    return rows, protected, excluded, predictions_applied


def _selfplay_evidence(
    *,
    repository_root: Path,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_ref: ArtifactRef,
    engine_ref: ArtifactRef,
    engine_receipt_ref: ArtifactRef | None,
    generation_ordinal: int,
    export_root: str,
    timeout_seconds: int,
    runner: CommandRunner,
    protected_splits: Mapping[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    attempts = require_list(execution, "attempts", "self-play execution", maximum_items=20_000)
    latest: dict[str, Mapping[str, Any]] = {}
    for index, raw_attempt in enumerate(attempts):
        attempt = require_mapping(raw_attempt, f"self-play execution.attempts[{index}]")
        job_id = str(attempt["jobId"])
        if job_id not in latest or int(attempt["attempt"]) > int(latest[job_id]["attempt"]):
            latest[job_id] = attempt
    rows: list[dict[str, object]] = []
    rows_by_identity: dict[str, dict[str, object]] = {}
    export_runs: list[dict[str, object]] = []
    excluded = 0
    duplicates = 0
    jobs = require_list(plan, "jobs", "self-play plan", maximum_items=5_000)
    champion = require_mapping(plan.get("champion"), "self-play plan.champion")
    for raw_job in jobs:
        job = require_mapping(raw_job, "self-play job")
        job_id = str(job["jobId"])
        attempt = latest[job_id]
        csa_values = attempt.get("csa")
        if not isinstance(csa_values, list) or len(csa_values) != 2:
            raise ContractError(f"self-play attempt lacks two CSA artifacts: {job_id}")
        csa_refs = [
            ArtifactRef.from_dict(value, f"attempt {job_id}.csa[{index}]")
            for index, value in enumerate(csa_values)
        ]
        report_ref = ArtifactRef.from_dict(attempt.get("report"), f"attempt {job_id}.report")
        report = validate_phase6_pair_report_binding(
            load_json_artifact(repository_root, report_ref),
            repository_root=repository_root,
            job=job,
            git_commit=str(plan["gitCommit"]),
            nodes_per_move=int(plan["nodesPerMove"]),
            model_a=champion,
            model_b=champion,
            csa_refs=(csa_refs[0], csa_refs[1]),
            context=f"self-play report {job_id}",
        )
        if report["metrics"]["illegalMoves"] != 0:
            raise ContractError(f"self-play report contains illegal moves: {job_id}")
        expected_csa = [str(path) for path in job["csaPaths"]]
        if [reference.path for reference in csa_refs] != expected_csa:
            raise ContractError(f"self-play CSA references differ from plan job {job_id}")
        export_path = f"{export_root}/exports/{job_id}.jsonl"
        output_file = contained_path(repository_root, export_path)
        ensure_contained_directory(
            repository_root,
            PurePosixPath(export_path).parent.as_posix(),
        )
        stdout_path, stderr_path, reusable = _export_log_paths(
            repository_root,
            f"{export_root}/logs/{job_id}",
            output_exists=output_file.exists(),
        )
        if output_file.exists():
            if not reusable:
                raise ContractError(f"existing CSA export lacks complete command logs: {job_id}")
            stdout_ref = artifact_ref(repository_root, stdout_path)
            stderr_ref = artifact_ref(repository_root, stderr_path)
        else:
            outcome = runner.run(
                {
                    "kind": "engine_dataset_export",
                    "argv": [
                        engine_ref.path,
                        "export-csa-jsonl",
                        "--input-dir",
                        f"{job['outputDir']}/games",
                        "--output",
                        export_path,
                        "--max-games",
                        "2",
                    ],
                    "timeoutSeconds": timeout_seconds,
                },
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                expected_executable=engine_ref,
                engine_build_receipt=engine_receipt_ref,
                memory_limit_mib=1_024,
            )
            if (
                outcome.return_code != 0
                or outcome.timed_out
                or outcome.output_limit_exceeded
                or outcome.memory_limit_exceeded
            ):
                raise ContractError(f"Rust CSA export failed for self-play job {job_id}")
            stdout_ref = outcome.stdout
            stderr_ref = outcome.stderr
        export_ref = artifact_ref(repository_root, export_path, maximum_bytes=MAX_EXPORT_BYTES)
        with verified_artifact_descriptor(
            repository_root, export_ref, maximum_bytes=MAX_EXPORT_BYTES
        ) as export_descriptor:
            exports = _load_exports_descriptor(export_descriptor, export_ref.path)
        expected_files = {PurePosixPath(path).name for path in expected_csa}
        if set(exports) != expected_files:
            raise ContractError(f"Rust CSA export filenames differ from plan job {job_id}")
        export_runs.append(
            {
                "jobId": job_id,
                "csa": [reference.as_dict() for reference in csa_refs],
                "output": export_ref.as_dict(),
                "stdout": stdout_ref.as_dict(),
                "stderr": stderr_ref.as_dict(),
            }
        )
        report_games = require_list(
            report, "games", f"self-play report {job_id}", minimum_items=2, maximum_items=2
        )
        for local_index, raw_game in enumerate(report_games):
            game = require_mapping(raw_game, f"self-play report {job_id}.games[{local_index}]")
            csa_ref = csa_refs[local_index]
            file_name = PurePosixPath(csa_ref.path).name
            if game.get("csaPath") != f"games/{file_name}":
                raise ContractError(f"CSA path differs from report for {job_id}/{file_name}")
            exported = exports[file_name]
            if exported.get("blackName") != game.get("black") or exported.get(
                "whiteName"
            ) != game.get("white"):
                raise ContractError(f"CSA player names differ from report for {job_id}/{file_name}")
            if _canonical_sfen(str(exported["initialSfen"])) != _canonical_sfen(str(job["sfen"])):
                raise ContractError(f"CSA initial SFEN differs from plan for {job_id}/{file_name}")
            if int(game["moves"]) != len(exported["usiMoves"]):
                raise ContractError(f"CSA move count differs from report for {job_id}/{file_name}")
            _validate_export_outcome(str(game["result"]), str(exported["outcome"]))
            # A max-plies stop is not a factual board result. The Rust replay keeps
            # it as ``unknown`` and it must therefore be omitted from outcome replay.
            outcome_kind = str(exported["outcome"])
            full_plies = len(exported["usiMoves"])
            source_game_id = csa_ref.sha256
            for ply, raw_sfen in enumerate(exported["positionSfens"][:-1]):
                sfen = _canonical_sfen(str(raw_sfen))
                protected_split = protected_splits.get(sfen)
                if protected_split is not None and protected_split != "train":
                    excluded += 1
                    continue
                side = "black" if sfen.split(" ")[1] == "b" else "white"
                identity = hashlib.sha256(
                    b"phase6_selfplay_position/v1\0"
                    + bytes.fromhex(source_game_id)
                    + ply.to_bytes(8, "big")
                ).hexdigest()
                stage = _stage(ply, full_plies)
                row: dict[str, object] = {
                    "positionId": identity,
                    "sfen": sfen,
                    "split": "train",
                    "sourceGenerationId": plan["generationId"],
                    "generationOrdinal": generation_ordinal,
                    "sourceType": "selfplay",
                    "sourceManifest": execution_ref.as_dict(),
                    "sourceGameId": source_game_id,
                    "sourcePly": ply,
                    "sideToMove": side,
                    "outcomeKind": outcome_kind,
                    "outcomeTarget": _outcome_target(outcome_kind, side),
                    "teacherBeforeCp": None,
                    "teacherAfterCp": None,
                    "teacherCp": None,
                    "modelCp": None,
                    "championMove": None,
                    "challengerMove": None,
                    "candidateGapCp": None,
                    "mateDistance": None,
                    "phase": stage,
                    "terminalBoundary": stage == "endgame" and ply >= max(0, full_plies - 8),
                    "searchNodes": None,
                    "suspectedFailure": "none",
                    "alreadyTeacherLabeled": False,
                }
                previous = rows_by_identity.get(identity)
                if previous is None:
                    rows_by_identity[identity] = row
                    rows.append(row)
                elif previous == row:
                    duplicates += 1
                else:
                    raise ContractError(
                        f"self-play position identity has divergent evidence: {identity}"
                    )
    return rows, export_runs, excluded, duplicates


def _load_exports_descriptor(descriptor: int, display_path: str) -> dict[str, Mapping[str, Any]]:
    try:
        raw_rows = list(
            iter_jsonl_descriptor_records(
                descriptor,
                display_path=Path(display_path),
                max_bytes=MAX_EXPORT_BYTES,
                max_line_bytes=MAX_EXPORT_LINE_BYTES,
                max_records=2,
            )
        )
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read Rust CSA export: {error}") from error
    if len(raw_rows) != 2:
        raise ContractError("Rust CSA export must contain exactly two games")
    exports: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        context = f"Rust CSA export row {index}"
        require_exact_keys(row, _EXPORT_KEYS, context)
        if row.get("schema") != "phase3_csa_export/v1" or row.get("status") != "ok":
            raise ContractError(f"{context} was rejected or has an unsupported schema")
        input_file = require_string(row, "inputFile", context, maximum_length=256)
        if PurePosixPath(input_file).name != input_file or not input_file.endswith(".csa"):
            raise ContractError(f"{context}.inputFile is unsafe")
        if input_file in exports:
            raise ContractError("Rust CSA export duplicated an input file")
        normalized = row.get("normalizedCsa")
        if (
            not isinstance(normalized, str)
            or not normalized
            or len(normalized.encode()) > 1_048_576
        ):
            raise ContractError(f"{context}.normalizedCsa is invalid")
        initial = _canonical_sfen(require_string(row, "initialSfen", context, maximum_length=1_024))
        sfens = row.get("positionSfens")
        moves = row.get("usiMoves")
        if (
            not isinstance(sfens, list)
            or not isinstance(moves, list)
            or len(sfens) != len(moves) + 1
            or len(moves) > MAX_SELFPLAY_PLIES
        ):
            raise ContractError(f"{context} has inconsistent position and move sequences")
        if not sfens or _canonical_sfen(str(sfens[0])) != initial:
            raise ContractError(f"{context} initial SFEN disagrees with its sequence")
        for sfen in sfens:
            if not isinstance(sfen, str):
                raise ContractError(f"{context} has a non-string SFEN")
            _canonical_sfen(sfen)
        if any(not isinstance(move, str) or not move or len(move) > 16 for move in moves):
            raise ContractError(f"{context} has an invalid USI move")
        for key in ("blackName", "whiteName", "terminalReason"):
            value = row.get(key)
            if value is not None and (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or any(character in value for character in "\r\n\0")
            ):
                raise ContractError(f"{context}.{key} is invalid")
        require_enum(row, "outcome", context, {"black_win", "white_win", "draw", "unknown"})
        result_validation = require_enum(
            row,
            "resultValidation",
            context,
            {"verified", "external_condition", "missing"},
        )
        if (row.get("terminalReason") is None) != (result_validation == "missing"):
            raise ContractError(f"{context} terminal reason and result validation disagree")
        exports[input_file] = row
    return exports


def _load_model_predictions_descriptor(
    descriptor: int, display_path: str
) -> dict[str, Mapping[str, Any]]:
    try:
        raw_rows = iter_jsonl_descriptor_records(
            descriptor,
            display_path=Path(display_path),
            max_bytes=128 * 1024 * 1024,
            max_line_bytes=64 * 1024,
            max_records=10_000,
        )
        predictions: dict[str, Mapping[str, Any]] = {}
        checkpoint_sha256: str | None = None
        config_sha256: str | None = None
        for index, raw_row in enumerate(raw_rows):
            context = f"model prediction row {index}"
            row = require_mapping(raw_row, context)
            require_exact_keys(row, _PREDICTION_KEYS, context)
            if row.get("schema") != "phase4_value_prediction/v1":
                raise ContractError(f"{context} has an unsupported schema")
            position_id = require_identifier(row, "positionId", context)
            if position_id in predictions:
                raise ContractError(f"duplicate model prediction position ID: {position_id}")
            _canonical_sfen(require_string(row, "canonicalSfen", context, maximum_length=1_024))
            require_enum(row, "split", context, {"train", "validation", "test"})
            require_enum(row, "stage", context, {"opening", "middlegame", "endgame"})
            score = require_mapping(row.get("teacherScore"), f"{context}.teacherScore")
            require_exact_keys(score, {"kind", "value"}, f"{context}.teacherScore")
            require_enum(score, "kind", f"{context}.teacherScore", {"cp", "mate"})
            require_int(
                score,
                "value",
                f"{context}.teacherScore",
                minimum=-(2**31),
                maximum=2**31 - 1,
            )
            require_int(row, "modelCp", context, minimum=-1_000_000, maximum=1_000_000)
            if row.get("candidateGapCp") is not None:
                require_int(
                    row,
                    "candidateGapCp",
                    context,
                    minimum=0,
                    maximum=1_000_000,
                )
            bestmove = require_string(row, "bestmove", context, maximum_length=16)
            recorded_move = require_string(row, "recordedMove", context, maximum_length=16)
            agrees = require_bool(row, "recordedMoveAgrees", context)
            if agrees != (bestmove == recorded_move):
                raise ContractError(f"{context}.recordedMoveAgrees is inconsistent")
            if not require_bool(row, "alreadyTeacherLabeled", context):
                raise ContractError(f"{context} must originate from a teacher-labeled example")
            row_checkpoint = require_sha256(row, "checkpointSha256", context)
            row_config = require_sha256(row, "configSha256", context)
            if checkpoint_sha256 is None:
                checkpoint_sha256 = row_checkpoint
                config_sha256 = row_config
            elif row_checkpoint != checkpoint_sha256 or row_config != config_sha256:
                raise ContractError("model predictions mix checkpoint or configuration identities")
            predictions[position_id] = row
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"cannot read strict model predictions: {error}") from error
    return predictions


def _validate_prediction_against_label(
    prediction: Mapping[str, Any],
    label: Mapping[str, Any],
    *,
    recorded_move: str,
    context: str,
) -> None:
    label_score = require_mapping(label.get("score"), f"{context}.labelScore")
    if (
        _canonical_sfen(str(prediction["canonicalSfen"]))
        != _canonical_sfen(str(label["canonical_sfen"]))
        or prediction["split"] != label["split"]
        or prediction["stage"] != label["stage"]
        or prediction["teacherScore"]
        != {"kind": label_score.get("kind"), "value": label_score.get("value")}
        or prediction["candidateGapCp"] != _candidate_gap(label)
        or prediction["bestmove"] != label["bestmove"]
        or prediction["recordedMove"] != recorded_move
    ):
        raise ContractError(f"{context} disagrees with its teacher label")


def _load_phase3_recorded_moves(
    *,
    repository_root: Path,
    positions_ref: ArtifactRef,
    dataset_manifest_ref: ArtifactRef,
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Join prediction rows back to their immutable move-bearing Phase 3 rows."""

    if not labels_by_id:
        return {}
    expected_records = validate_phase3_dataset_binding(
        load_json_artifact(repository_root, dataset_manifest_ref),
        positions_ref=positions_ref,
        dataset_manifest_ref=dataset_manifest_ref,
    )
    found: dict[str, str] = {}
    observed = 0
    try:
        with verified_artifact_descriptor(
            repository_root, positions_ref, maximum_bytes=1_073_741_824
        ) as descriptor:
            for raw in _iter_jsonl_gzip_descriptor(
                descriptor,
                Path(positions_ref.path),
                max_uncompressed_bytes=4_294_967_296,
                max_records=250_000,
                max_line_bytes=64 * 1024,
            ):
                observed += 1
                parsed = _parse_phase3_position(raw, observed)
                game_id = str(parsed["gameId"])
                position_index = int(parsed["positionIndex"])
                try:
                    identity = position_id(game_id, position_index)
                except ValueError as error:
                    raise ContractError(
                        f"Phase 3 position row {observed} identity is invalid"
                    ) from error
                label = labels_by_id.get(identity)
                if label is None:
                    continue
                if identity in found:
                    raise ContractError(f"Phase 3 positions duplicate teacher label {identity}")
                recorded_move = parsed.get("moveUsi")
                if (
                    not isinstance(recorded_move, str)
                    or not recorded_move
                    or len(recorded_move) > 16
                ):
                    raise ContractError(
                        f"teacher-labeled Phase 3 row has no recorded move: {identity}"
                    )
                if (
                    _canonical_sfen(str(parsed.get("sfen")))
                    != _canonical_sfen(str(label["canonical_sfen"]))
                    or parsed.get("split") != label["split"]
                    or parsed.get("outcome") != label["outcome"]
                    or parsed.get("canonicalSha256") != label["game_id"]
                    or parsed.get("eligible") is not True
                ):
                    raise ContractError(f"teacher label disagrees with Phase 3 row {identity}")
                found[identity] = recorded_move
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"cannot join teacher labels to Phase 3 positions: {error}") from error
    if observed != expected_records:
        raise ContractError("Phase 3 position count changed during prediction join")
    missing = sorted(set(labels_by_id) - set(found))
    if missing:
        raise ContractError(f"Phase 3 positions omit teacher label {missing[0]}")
    return found


def _canonical_sfen(value: str) -> str:
    if not value or not value.isascii() or len(value) > 1_024 or any(c in value for c in "\r\n\0"):
        raise ContractError("SFEN must be one bounded ASCII line")
    fields = value.split(" ")
    if (
        len(fields) != 4
        or any(not field for field in fields)
        or fields[1] not in {"b", "w"}
        or not fields[3].isdecimal()
        or int(fields[3]) < 1
    ):
        raise ContractError("SFEN has invalid fields")
    return " ".join((*fields[:3], "1"))


def _candidate_gap(label: Mapping[str, Any]) -> int | None:
    candidates = label.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    scores: list[int] = []
    for raw in candidates[:2]:
        candidate = require_mapping(raw, "teacher candidate")
        score = require_mapping(candidate.get("score"), "teacher candidate.score")
        if score.get("kind") != "cp":
            return None
        scores.append(
            require_int(
                score, "value", "teacher candidate.score", minimum=-(2**31), maximum=2**31 - 1
            )
        )
    return abs(scores[0] - scores[1])


def _outcome_target(outcome: str, side: str) -> int | None:
    if outcome == "unknown":
        return None
    if outcome == "draw":
        return 0
    winner = "black" if outcome == "black_win" else "white"
    return 1 if side == winner else -1


def _stage(ply: int, full_plies: int) -> str:
    if full_plies <= 0:
        raise ContractError("self-play outcome example requires at least one move")
    progress = ply * 10_000 // full_plies
    if progress < 3_333:
        return "opening"
    if progress < 6_667:
        return "middlegame"
    return "endgame"


def _validate_export_outcome(report: str, exported: str) -> None:
    if report == "max_plies":
        if exported != "unknown":
            raise ContractError("max-plies CSA export must retain an unknown board outcome")
    elif report != exported:
        raise ContractError("CSA outcome differs from its arena report")


def _export_log_paths(
    repository_root: Path, base: str, *, output_exists: bool
) -> tuple[str, str, bool]:
    """Reuse complete export evidence or select fresh immutable recovery logs."""

    complete: tuple[str, str] | None = None
    available: tuple[str, str] | None = None
    bases = [base, *(f"{base}.recovery-{index:03d}" for index in range(1, 101))]
    for candidate in bases:
        stdout_path = f"{candidate}.stdout.log"
        stderr_path = f"{candidate}.stderr.log"
        stdout = contained_path(repository_root, stdout_path)
        stderr = contained_path(repository_root, stderr_path)
        stdout_exists = stdout.exists()
        stderr_exists = stderr.exists()
        if stdout_exists and stderr_exists:
            complete = (stdout_path, stderr_path)
        elif not stdout_exists and not stderr_exists and available is None:
            available = (stdout_path, stderr_path)
    if output_exists:
        if complete is None:
            return "", "", False
        return *complete, True
    if available is None:
        raise ContractError("CSA export exhausted its immutable recovery log paths")
    return *available, False


def _identifier(value: object, context: str) -> str:
    return require_identifier({"value": value}, "value", context)
