"""Hard-position selection and provenance-preserving replay-buffer manifests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

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
    require_optional_string,
    require_sha256,
    require_string,
    verify_artifact_ref,
)
from .config import SelfPlayConfig, parse_selfplay_config_bytes

POSITION_EVIDENCE_SCHEMA: Final = "phase6_position_evidence/v1"
HARD_POSITIONS_SCHEMA: Final = "phase6_hard_positions/v1"
REPLAY_CANDIDATES_SCHEMA: Final = "phase6_replay_candidates/v1"
REPLAY_MANIFEST_SCHEMA: Final = "phase6_replay_buffer_manifest/v2"
MAX_EVIDENCE_POSITIONS: Final = 200_000
MAX_EVIDENCE_JSON_NODES: Final = 8_000_000
MAX_REPLAY_CANDIDATES: Final = 1_000_000

_EVIDENCE_ROOT_KEYS = frozenset(
    {"schema", "generationId", "derivation", "sourceManifests", "positions"}
)
_EVIDENCE_POSITION_KEYS = frozenset(
    {
        "positionId",
        "sfen",
        "split",
        "sourceGenerationId",
        "generationOrdinal",
        "sourceType",
        "sourceManifest",
        "sourceGameId",
        "sourcePly",
        "sideToMove",
        "outcomeKind",
        "outcomeTarget",
        "teacherBeforeCp",
        "teacherAfterCp",
        "teacherCp",
        "modelCp",
        "championMove",
        "challengerMove",
        "candidateGapCp",
        "mateDistance",
        "phase",
        "terminalBoundary",
        "searchNodes",
        "suspectedFailure",
        "alreadyTeacherLabeled",
    }
)
_REPLAY_ROOT_KEYS = frozenset(
    {"schema", "generationId", "positionEvidence", "hardPositions", "entries"}
)
_REPLAY_ENTRY_KEYS = frozenset(
    {
        "positionId",
        "sfen",
        "split",
        "generationId",
        "generationOrdinal",
        "sourceType",
        "sourceManifest",
        "sourceGameId",
        "sourcePly",
        "sideToMove",
        "outcomeKind",
        "outcomeTarget",
        "stage",
        "priority",
        "hardPosition",
        "isNew",
    }
)
_DERIVATION_KEYS = frozenset(
    {
        "selfplayPlan",
        "selfplayManifest",
        "teacherLabels",
        "modelPredictions",
        "datasetManifest",
        "generationOrdinal",
        "teacherSourceGenerationId",
        "csaExports",
        "counts",
        "unavailableMeasurements",
    }
)
_DERIVATION_EXPORT_KEYS = frozenset({"jobId", "csa", "output", "stdout", "stderr"})
_DERIVATION_COUNT_KEYS = frozenset(
    {
        "teacherPositions",
        "selfplayPositions",
        "modelPredictionsApplied",
        "excludedCrossSplitTeacherRows",
        "excludedProtectedSelfplayRows",
        "excludedDuplicateSelfplayRows",
    }
)
_UNAVAILABLE_MEASUREMENTS = (
    "selfplay_teacher_before_after_cp",
    "selfplay_teacher_cp",
    "selfplay_model_cp",
    "selfplay_champion_challenger_moves",
    "selfplay_candidate_gap_cp",
    "selfplay_mate_distance",
    "selfplay_actual_search_nodes",
)

_REASON_WEIGHTS: Final[dict[str, int]] = {
    "teacher_evaluation_drop": 100,
    "teacher_model_disagreement": 90,
    "champion_challenger_move_disagreement": 70,
    "ambiguous_candidates": 45,
    "mate_related": 85,
    "endgame_boundary": 55,
    "insufficient_search": 60,
    "suspected_search_error": 80,
    "suspected_evaluation_error": 80,
}
_REASON_ORDER = tuple(_REASON_WEIGHTS)


@dataclass(frozen=True, slots=True)
class _HardCandidate:
    position_id: str
    sfen: str
    split: str
    generation_ordinal: int
    source_manifest: ArtifactRef
    source_type: str
    reasons: tuple[str, ...]
    priority: int
    already_teacher_labeled: bool


@dataclass(frozen=True, slots=True)
class _ReplayCandidate:
    position_id: str
    sfen: str
    split: str
    generation_id: str
    generation_ordinal: int
    source_type: str
    source_manifest: ArtifactRef
    source_game_id: str
    source_ply: int
    side_to_move: str
    outcome_kind: str
    outcome_target: int
    stage: str
    priority: int
    hard_position: bool
    is_new: bool


def extract_hard_positions(
    raw: object,
    *,
    evidence_ref: ArtifactRef,
    config: SelfPlayConfig,
    labels_before: int,
    requested_max: int | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Rank evidence deterministically without exceeding the global teacher budget."""

    root, declared_sources = _validate_position_evidence(
        raw,
        evidence_ref=evidence_ref,
        config=config,
        repository_root=repository_root,
    )
    generation_id = require_identifier(root, "generationId", "position evidence")
    source_set = set(declared_sources)
    rows = require_list(
        root,
        "positions",
        "position evidence",
        maximum_items=MAX_EVIDENCE_POSITIONS,
    )
    if not 0 <= labels_before <= config.hard_positions.teacher_label_limit:
        raise ContractError("labels_before is outside the configured teacher-label budget")
    maximum = (
        config.hard_positions.max_additional_labels if requested_max is None else requested_max
    )
    if not 0 <= maximum <= config.hard_positions.max_additional_labels:
        raise ContractError("requested_max exceeds max_additional_labels")
    available = config.hard_positions.teacher_label_limit - labels_before

    candidates: list[_HardCandidate] = []
    position_ids: set[str] = set()
    for index, row in enumerate(rows):
        context = f"position evidence.positions[{index}]"
        table = require_mapping(row, context)
        require_exact_keys(table, _EVIDENCE_POSITION_KEYS, context)
        position_id = require_identifier(table, "positionId", context)
        if position_id in position_ids:
            raise ContractError(f"duplicate evidence position ID: {position_id}")
        position_ids.add(position_id)
        source_manifest = ArtifactRef.from_dict(
            table.get("sourceManifest"), f"{context}.sourceManifest"
        )
        if source_manifest not in source_set:
            raise ContractError(f"{context} uses an undeclared source manifest")
        candidate = _parse_hard_candidate(table, context, config)
        if candidate is not None:
            candidates.append(candidate)

    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = hashlib.sha256(candidate.sfen.encode("utf-8")).hexdigest()
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = {
                "dedupKey": key,
                "sfen": candidate.sfen,
                "split": candidate.split,
                "priority": candidate.priority,
                "reasons": set(candidate.reasons),
                "sourcePositionIds": {candidate.position_id},
                "sourceManifests": {candidate.source_manifest},
                "sourceTypes": {candidate.source_type},
                "newestGenerationOrdinal": candidate.generation_ordinal,
                "alreadyTeacherLabeled": candidate.already_teacher_labeled,
            }
            continue
        if existing["split"] != candidate.split:
            raise ContractError("the same SFEN appears in more than one data split")
        existing["priority"] = max(existing["priority"], candidate.priority)
        existing["reasons"].update(candidate.reasons)
        existing["sourcePositionIds"].add(candidate.position_id)
        existing["sourceManifests"].add(candidate.source_manifest)
        existing["sourceTypes"].add(candidate.source_type)
        existing["newestGenerationOrdinal"] = max(
            existing["newestGenerationOrdinal"], candidate.generation_ordinal
        )
        existing["alreadyTeacherLabeled"] = (
            existing["alreadyTeacherLabeled"] or candidate.already_teacher_labeled
        )

    ranked = sorted(
        deduplicated.values(),
        key=lambda item: (-item["priority"], -item["newestGenerationOrdinal"], item["dedupKey"]),
    )
    selected_items: list[dict[str, Any]] = []
    new_teacher_labels = 0
    for item in ranked:
        if len(selected_items) >= maximum:
            break
        if item["alreadyTeacherLabeled"]:
            selected_items.append(item)
        elif new_teacher_labels < available:
            selected_items.append(item)
            new_teacher_labels += 1
    selected = [_serialize_hard_position(item) for item in selected_items]
    already_labeled = len(selected) - new_teacher_labels
    result: dict[str, object] = {
        "schema": HARD_POSITIONS_SCHEMA,
        "generationId": generation_id,
        "inputEvidence": evidence_ref.as_dict(),
        "configSha256": config.sha256,
        "budget": {
            "teacherLabelLimit": config.hard_positions.teacher_label_limit,
            "labelsBefore": labels_before,
            "requestedMaximum": maximum,
            "availableNewTeacherLabelsBeforeSelection": available,
            "selected": len(selected),
            "alreadyLabeledSelected": already_labeled,
            "newTeacherLabelsSelected": new_teacher_labels,
            "labelsAfterMaximum": labels_before + new_teacher_labels,
        },
        "eligibleUniquePositions": len(ranked),
        "positions": selected,
    }
    result["manifestSha256"] = canonical_sha256(result)
    return result


def build_replay_candidates(
    evidence: object,
    *,
    evidence_ref: ArtifactRef,
    hard_positions: object,
    hard_positions_ref: ArtifactRef,
    config: SelfPlayConfig,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Convert factual evidence into outcome-only replay examples.

    Unknown outcomes are intentionally omitted. No teacher or policy target is
    synthesized; consumers must apply only the recorded current-side outcome target.
    """

    evidence_root, source_references = _validate_position_evidence(
        evidence,
        evidence_ref=evidence_ref,
        config=config,
        repository_root=repository_root,
    )
    generation_id = require_identifier(evidence_root, "generationId", "position evidence")
    declared_sources = set(source_references)
    hard_root = _validate_hard_positions_manifest(
        hard_positions,
        evidence_ref=evidence_ref,
        config=config,
    )
    if hard_root.get("generationId") != generation_id:
        raise ContractError("hard-position generation does not match position evidence")
    if (
        repository_root is not None
        and load_json_artifact(repository_root, hard_positions_ref) != hard_positions
    ):
        raise ContractError("hard-position value differs from its referenced artifact")

    hard_by_state: dict[str, int] = {}
    hard_rows = require_list(
        hard_root,
        "positions",
        "hard positions",
        maximum_items=MAX_EVIDENCE_POSITIONS,
    )
    for raw_hard in hard_rows:
        hard = require_mapping(raw_hard, "hard position")
        canonical_sfen = _bounded_sfen(hard, "hard position")
        dedup_key = require_sha256(hard, "dedupKey", "hard position")
        if dedup_key != hashlib.sha256(canonical_sfen.encode("utf-8")).hexdigest():
            raise ContractError("hard-position dedup key disagrees with canonical SFEN")
        hard_by_state[dedup_key] = require_int(
            hard, "priority", "hard position", minimum=0, maximum=1_000_000
        )

    entries: list[dict[str, object]] = []
    rows = require_list(
        evidence_root,
        "positions",
        "position evidence",
        maximum_items=MAX_EVIDENCE_POSITIONS,
    )
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        context = f"position evidence.positions[{index}]"
        row = require_mapping(raw_row, context)
        require_exact_keys(row, _EVIDENCE_POSITION_KEYS, context)
        # The hard-candidate parser validates every evidence field even when the
        # row does not cross a hard-position threshold.
        _parse_hard_candidate(row, context, config)
        position_id = require_identifier(row, "positionId", context)
        if position_id in seen:
            raise ContractError(f"duplicate evidence position ID: {position_id}")
        seen.add(position_id)
        source = ArtifactRef.from_dict(row.get("sourceManifest"), f"{context}.sourceManifest")
        if source not in declared_sources:
            raise ContractError(f"{context} uses an undeclared source manifest")
        side, outcome_kind, outcome_target = _validate_outcome_fields(
            row, context, allow_unknown=True
        )
        if outcome_target is None:
            continue
        canonical_sfen = _bounded_sfen(row, context)
        hard_priority = hard_by_state.get(
            hashlib.sha256(canonical_sfen.encode("utf-8")).hexdigest()
        )
        source_type = require_enum(
            row, "sourceType", context, {"teacher", "model", "arena", "selfplay"}
        )
        source_generation = require_identifier(row, "sourceGenerationId", context)
        is_new = source_generation == generation_id and source_type in {"arena", "selfplay"}
        replay_source_type = source_type if source_type in {"teacher", "selfplay"} else "retained"
        entries.append(
            {
                "positionId": position_id,
                "sfen": canonical_sfen,
                "split": require_enum(row, "split", context, {"train", "validation", "test"}),
                "generationId": source_generation,
                "generationOrdinal": require_int(
                    row, "generationOrdinal", context, minimum=0, maximum=1_000_000
                ),
                "sourceType": replay_source_type,
                "sourceManifest": source.as_dict(),
                "sourceGameId": require_identifier(row, "sourceGameId", context),
                "sourcePly": require_int(row, "sourcePly", context, minimum=0, maximum=10_000),
                "sideToMove": side,
                "outcomeKind": outcome_kind,
                "outcomeTarget": outcome_target,
                "stage": require_enum(row, "phase", context, {"opening", "middlegame", "endgame"}),
                "priority": max(100 if is_new else 30, hard_priority or 0),
                "hardPosition": hard_priority is not None,
                "isNew": is_new,
            }
        )
    entries.sort(key=lambda item: (str(item["split"]), str(item["positionId"])))
    return {
        "schema": REPLAY_CANDIDATES_SCHEMA,
        "generationId": generation_id,
        "positionEvidence": evidence_ref.as_dict(),
        "hardPositions": hard_positions_ref.as_dict(),
        "entries": entries,
    }


def build_replay_buffer_manifest(
    raw: object,
    *,
    candidates_ref: ArtifactRef,
    config: SelfPlayConfig,
    config_ref: ArtifactRef,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Deduplicate, prioritize, and cap replay entries while retaining old useful data."""

    root = require_mapping(raw, "replay candidates")
    require_exact_keys(root, _REPLAY_ROOT_KEYS, "replay candidates")
    if root.get("schema") != REPLAY_CANDIDATES_SCHEMA:
        raise ContractError("unsupported replay-candidates schema")
    generation_id = require_identifier(root, "generationId", "replay candidates")
    evidence_ref = ArtifactRef.from_dict(
        root.get("positionEvidence"), "replay candidates.positionEvidence"
    )
    hard_positions_ref = ArtifactRef.from_dict(
        root.get("hardPositions"), "replay candidates.hardPositions"
    )
    if repository_root is not None:
        if load_json_artifact(repository_root, candidates_ref) != root:
            raise ContractError("replay-candidates value differs from its referenced artifact")
        observed_config = parse_selfplay_config_bytes(
            load_bytes_artifact(repository_root, config_ref, maximum_bytes=64 * 1024),
            config_ref.path,
        )
        if observed_config != config:
            raise ContractError("replay config differs from its referenced artifact")
        expected = build_replay_candidates(
            load_json_artifact(
                repository_root,
                evidence_ref,
                maximum_nodes=MAX_EVIDENCE_JSON_NODES,
            ),
            evidence_ref=evidence_ref,
            hard_positions=load_json_artifact(repository_root, hard_positions_ref),
            hard_positions_ref=hard_positions_ref,
            config=config,
            repository_root=repository_root,
        )
        if root != expected:
            raise ContractError("replay candidates are not the deterministic evidence derivation")
    rows = require_list(
        root,
        "entries",
        "replay candidates",
        maximum_items=MAX_REPLAY_CANDIDATES,
    )
    candidates: list[_ReplayCandidate] = []
    position_ids: set[str] = set()
    for index, row in enumerate(rows):
        context = f"replay candidates.entries[{index}]"
        table = require_mapping(row, context)
        require_exact_keys(table, _REPLAY_ENTRY_KEYS, context)
        candidate = _parse_replay_candidate(table, context)
        if candidate.is_new and (
            candidate.generation_id != generation_id or candidate.source_type != "selfplay"
        ):
            raise ContractError(f"{context} has inconsistent new-generation provenance")
        if candidate.source_type == "selfplay" and candidate.split != "train":
            raise ContractError(f"{context} self-play examples must remain in the train split")
        if repository_root is not None:
            verify_artifact_ref(repository_root, candidate.source_manifest)
        if candidate.position_id in position_ids:
            raise ContractError(f"duplicate replay position ID: {candidate.position_id}")
        position_ids.add(candidate.position_id)
        candidates.append(candidate)

    grouped: dict[str, list[_ReplayCandidate]] = {}
    deletion_log: list[dict[str, object]] = []
    for candidate in candidates:
        key = hashlib.sha256(candidate.sfen.encode("utf-8")).hexdigest()
        existing = grouped.setdefault(key, [])
        if existing and existing[0].split != candidate.split:
            raise ContractError("the same replay SFEN appears in more than one split")
        existing.append(candidate)

    entries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        ordered = sorted(group, key=_replay_candidate_rank)
        representative = ordered[0]
        for duplicate in ordered[1:]:
            reason = (
                "deduplicated_conflicting_outcome_target"
                if duplicate.outcome_target != representative.outcome_target
                else "deduplicated_canonical_sfen"
            )
            deletion_log.append(
                {
                    "positionId": duplicate.position_id,
                    "entryId": key,
                    "reason": reason,
                }
            )
        entries.append(
            {
                "entryId": key,
                "canonicalSfen": representative.sfen,
                "sideToMove": representative.side_to_move,
                "outcomeKind": representative.outcome_kind,
                "outcomeTarget": representative.outcome_target,
                "stage": representative.stage,
                "priority": max(item.priority for item in ordered),
                "hardPosition": any(item.hard_position for item in ordered),
                "split": representative.split,
                "newestGenerationOrdinal": max(item.generation_ordinal for item in ordered),
                "isNew": any(item.is_new for item in ordered),
                "origins": [_replay_origin(item) for item in ordered],
            }
        )
    ranking = sorted(
        entries,
        key=lambda item: (
            -item["priority"],
            -int(item["isNew"]),
            -item["newestGenerationOrdinal"],
            item["entryId"],
        ),
    )
    old_ranking = [item for item in ranking if not item["isNew"]]
    old_reserve = min(
        config.replay.minimum_older_positions,
        config.replay.capacity,
        len(old_ranking),
    )
    selected_ids = {item["entryId"] for item in old_ranking[:old_reserve]}
    for item in ranking:
        if len(selected_ids) >= config.replay.capacity:
            break
        selected_ids.add(item["entryId"])
    selected = [item for item in ranking if item["entryId"] in selected_ids]
    selected.sort(key=lambda item: (item["split"], -item["priority"], item["entryId"]))
    for item in ranking:
        if item["entryId"] not in selected_ids:
            deletion_log.append(
                {
                    "positionId": None,
                    "entryId": item["entryId"],
                    "reason": "capacity_eviction",
                }
            )
    deletion_log.sort(key=lambda item: (item["reason"], item["entryId"], item["positionId"] or ""))
    if len(deletion_log) != len(candidates) - len(selected):
        raise ContractError("replay deletion log does not account for every removed candidate")
    expected_older = min(
        config.replay.minimum_older_positions,
        config.replay.capacity,
        len(old_ranking),
    )
    if sum(not bool(item["isNew"]) for item in selected) < expected_older:
        raise ContractError("replay selection violated its older-position reserve")

    split_counts = {
        split: sum(item["split"] == split for item in selected)
        for split in ("train", "validation", "test")
    }
    result: dict[str, object] = {
        "schema": REPLAY_MANIFEST_SCHEMA,
        "generationId": generation_id,
        "inputCandidates": candidates_ref.as_dict(),
        "config": config_ref.as_dict(),
        "configSha256": config.sha256,
        "dedupKey": config.replay.dedup_key,
        "capacity": config.replay.capacity,
        "minimumOlderPositions": config.replay.minimum_older_positions,
        "counts": {
            "input": len(candidates),
            "unique": len(entries),
            "retained": len(selected),
            "newRetained": sum(bool(item["isNew"]) for item in selected),
            "olderRetained": sum(not bool(item["isNew"]) for item in selected),
            "hardRetained": sum(bool(item["hardPosition"]) for item in selected),
            "outcomeTargets": {
                "loss": sum(item["outcomeTarget"] == -1 for item in selected),
                "draw": sum(item["outcomeTarget"] == 0 for item in selected),
                "win": sum(item["outcomeTarget"] == 1 for item in selected),
            },
            "splits": split_counts,
        },
        "splitPolicy": {
            "train": "training",
            "validation": "selection_metrics_only",
            "test": "final_evaluation_only",
        },
        "entries": selected,
        "deletionLog": deletion_log,
    }
    result["manifestSha256"] = canonical_sha256(result)
    return result


def _validate_position_evidence(
    raw: object,
    *,
    evidence_ref: ArtifactRef,
    config: SelfPlayConfig,
    repository_root: Path | None,
) -> tuple[Mapping[str, Any], tuple[ArtifactRef, ...]]:
    """Validate evidence provenance before any selection or replay transformation."""

    root = require_mapping(raw, "position evidence")
    require_exact_keys(root, _EVIDENCE_ROOT_KEYS, "position evidence")
    if root.get("schema") != POSITION_EVIDENCE_SCHEMA:
        raise ContractError("unsupported position-evidence schema")
    generation_id = require_identifier(root, "generationId", "position evidence")
    derivation = require_mapping(root.get("derivation"), "position evidence.derivation")
    require_exact_keys(derivation, _DERIVATION_KEYS, "position evidence.derivation")
    selfplay_plan = ArtifactRef.from_dict(
        derivation.get("selfplayPlan"), "position evidence.derivation.selfplayPlan"
    )
    selfplay_manifest = ArtifactRef.from_dict(
        derivation.get("selfplayManifest"),
        "position evidence.derivation.selfplayManifest",
    )
    teacher_labels = ArtifactRef.from_dict(
        derivation.get("teacherLabels"), "position evidence.derivation.teacherLabels"
    )
    model_predictions = ArtifactRef.from_dict(
        derivation.get("modelPredictions"),
        "position evidence.derivation.modelPredictions",
    )
    dataset_manifest = ArtifactRef.from_dict(
        derivation.get("datasetManifest"),
        "position evidence.derivation.datasetManifest",
    )
    if len({selfplay_manifest, teacher_labels, model_predictions}) != 3:
        raise ContractError("position evidence source manifests must be distinct")
    generation_ordinal = require_int(
        derivation,
        "generationOrdinal",
        "position evidence.derivation",
        minimum=1,
        maximum=1_000_000,
    )
    teacher_generation_id = require_identifier(
        derivation,
        "teacherSourceGenerationId",
        "position evidence.derivation",
    )
    declared_sources = _artifact_ref_array(
        root, "sourceManifests", "position evidence", maximum_items=10_000
    )
    if set(declared_sources) != {teacher_labels, model_predictions, selfplay_manifest}:
        raise ContractError("position evidence source manifests disagree with its derivation")

    export_rows = require_list(
        derivation,
        "csaExports",
        "position evidence.derivation",
        maximum_items=5_000,
    )
    export_job_ids: set[str] = set()
    export_csa_by_job: dict[str, tuple[ArtifactRef, ArtifactRef]] = {}
    referenced_artifacts = {
        selfplay_plan,
        selfplay_manifest,
        teacher_labels,
        model_predictions,
        dataset_manifest,
    }
    for index, raw_export in enumerate(export_rows):
        context = f"position evidence.derivation.csaExports[{index}]"
        export = require_mapping(raw_export, context)
        require_exact_keys(export, _DERIVATION_EXPORT_KEYS, context)
        job_id = require_identifier(export, "jobId", context)
        if job_id in export_job_ids:
            raise ContractError(f"duplicate evidence CSA export job ID: {job_id}")
        export_job_ids.add(job_id)
        csa_values = require_list(export, "csa", context, minimum_items=2, maximum_items=2)
        csa = tuple(
            ArtifactRef.from_dict(value, f"{context}.csa[{csa_index}]")
            for csa_index, value in enumerate(csa_values)
        )
        if csa[0] == csa[1]:
            raise ContractError(f"{context}.csa contains duplicate artifacts")
        export_csa_by_job[job_id] = (csa[0], csa[1])
        referenced_artifacts.update(csa)
        for key in ("output", "stdout", "stderr"):
            referenced_artifacts.add(ArtifactRef.from_dict(export.get(key), f"{context}.{key}"))

    counts = require_mapping(derivation.get("counts"), "position evidence.derivation.counts")
    require_exact_keys(counts, _DERIVATION_COUNT_KEYS, "position evidence.derivation.counts")
    parsed_counts = {
        key: require_int(
            counts,
            key,
            "position evidence.derivation.counts",
            minimum=0,
            maximum=MAX_EVIDENCE_POSITIONS,
        )
        for key in _DERIVATION_COUNT_KEYS
    }
    unavailable = require_list(
        derivation,
        "unavailableMeasurements",
        "position evidence.derivation",
        minimum_items=len(_UNAVAILABLE_MEASUREMENTS),
        maximum_items=len(_UNAVAILABLE_MEASUREMENTS),
    )
    if tuple(unavailable) != _UNAVAILABLE_MEASUREMENTS:
        raise ContractError("position evidence unavailable-measurement declaration drifted")

    expected_job_ids: set[str] | None = None
    completed_csa_by_job: dict[str, tuple[ArtifactRef, ArtifactRef]] = {}
    expected_teacher_rows: list[dict[str, object]] | None = None
    expected_selfplay_rows: list[dict[str, object]] | None = None
    if repository_root is not None:
        if (
            load_json_artifact(
                repository_root,
                evidence_ref,
                maximum_nodes=MAX_EVIDENCE_JSON_NODES,
            )
            != root
        ):
            raise ContractError("position-evidence value differs from its referenced artifact")
        for reference in referenced_artifacts:
            verify_artifact_ref(repository_root, reference)
        from .derivation import _selfplay_evidence, _teacher_evidence
        from .execution import (
            SELFPLAY_MANIFEST_SCHEMA,
            validate_execution_manifest,
            validate_paired_plan,
        )
        from .planning import SELFPLAY_PLAN_SCHEMA

        plan = validate_paired_plan(load_json_artifact(repository_root, selfplay_plan))
        if plan.get("schema") != SELFPLAY_PLAN_SCHEMA:
            raise ContractError("position evidence derivation requires a self-play plan")
        if plan.get("generationId") != generation_id:
            raise ContractError("position evidence generation disagrees with its self-play plan")
        if (
            ArtifactRef.from_dict(plan.get("datasetManifest"), "self-play plan.datasetManifest")
            != dataset_manifest
        ):
            raise ContractError("position evidence dataset disagrees with its self-play plan")
        from .planning import parse_start_positions

        starts_ref = ArtifactRef.from_dict(
            plan.get("startPositions"), "self-play plan.startPositions"
        )
        starts = parse_start_positions(load_json_artifact(repository_root, starts_ref))
        if starts.dataset_manifest != dataset_manifest:
            raise ContractError("position evidence start set uses a different Phase 3 dataset")
        execution = validate_execution_manifest(
            load_json_artifact(repository_root, selfplay_manifest),
            repository_root=repository_root,
            manifest_ref=selfplay_manifest,
            plan=plan,
            plan_ref=selfplay_plan,
        )
        if execution.get("schema") != SELFPLAY_MANIFEST_SCHEMA:
            raise ContractError("position evidence derivation requires self-play execution")
        (
            expected_teacher_rows,
            protected_splits,
            expected_excluded_teacher,
            expected_predictions_applied,
        ) = _teacher_evidence(
            repository_root=repository_root,
            labels_ref=teacher_labels,
            predictions_ref=model_predictions,
            dataset_manifest_ref=dataset_manifest,
            source_positions_ref=starts.source_positions,
            source_generation_id=teacher_generation_id,
        )
        if (
            parsed_counts["excludedCrossSplitTeacherRows"] != expected_excluded_teacher
            or parsed_counts["modelPredictionsApplied"] != expected_predictions_applied
        ):
            raise ContractError("position evidence teacher derivation counts disagree")
        jobs = require_list(plan, "jobs", "self-play plan", maximum_items=5_000)
        expected_job_ids = {str(job["jobId"]) for job in jobs}
        attempts = require_list(execution, "attempts", "self-play execution", maximum_items=20_000)
        latest: dict[str, Mapping[str, Any]] = {}
        for raw_attempt in attempts:
            attempt = require_mapping(raw_attempt, "self-play attempt")
            job_id = str(attempt["jobId"])
            if job_id not in latest or int(attempt["attempt"]) > int(latest[job_id]["attempt"]):
                latest[job_id] = attempt
        for job_id, attempt in latest.items():
            csa_values = require_list(
                attempt, "csa", f"self-play attempt {job_id}", minimum_items=2, maximum_items=2
            )
            completed_csa_by_job[job_id] = tuple(
                ArtifactRef.from_dict(value, f"self-play attempt {job_id}.csa[{index}]")
                for index, value in enumerate(csa_values)
            )
        export_roots = {
            PurePosixPath(
                ArtifactRef.from_dict(export.get("output"), "evidence export.output").path
            ).parent.parent.as_posix()
            for export in (require_mapping(value, "evidence export") for value in export_rows)
        }
        if len(export_roots) != 1:
            raise ContractError("position evidence CSA exports do not share one output root")
        export_root = next(iter(export_roots))
        receipt_raw = plan.get("engineBuildReceipt")
        engine_receipt_ref = (
            None
            if receipt_raw is None
            else ArtifactRef.from_dict(receipt_raw, "self-play plan.engineBuildReceipt")
        )
        (
            expected_selfplay_rows,
            expected_export_runs,
            expected_excluded_selfplay,
            expected_duplicate_selfplay,
        ) = _selfplay_evidence(
            repository_root=repository_root,
            plan=plan,
            execution=execution,
            execution_ref=selfplay_manifest,
            engine_ref=ArtifactRef.from_dict(plan.get("engine"), "self-play plan.engine"),
            engine_receipt_ref=engine_receipt_ref,
            generation_ordinal=generation_ordinal,
            export_root=export_root,
            timeout_seconds=1,
            runner=_NoCommandRunner(),
            protected_splits=protected_splits,
        )
        if expected_export_runs != export_rows:
            raise ContractError(
                "position evidence export runs are not their deterministic derivation"
            )
        if parsed_counts["excludedProtectedSelfplayRows"] != expected_excluded_selfplay:
            raise ContractError("position evidence self-play exclusion count disagrees")
        if parsed_counts["excludedDuplicateSelfplayRows"] != expected_duplicate_selfplay:
            raise ContractError("position evidence self-play duplicate count disagrees")
    if expected_job_ids is not None and export_job_ids != expected_job_ids:
        raise ContractError("position evidence CSA exports do not cover every self-play job")
    for job_id, csa in export_csa_by_job.items():
        if completed_csa_by_job and completed_csa_by_job.get(job_id) != csa:
            raise ContractError(f"position evidence CSA export differs from job {job_id}")

    rows = require_list(
        root,
        "positions",
        "position evidence",
        maximum_items=MAX_EVIDENCE_POSITIONS,
    )
    if len(rows) != parsed_counts["teacherPositions"] + parsed_counts["selfplayPositions"]:
        raise ContractError("position evidence row count disagrees with derivation counts")
    if parsed_counts["modelPredictionsApplied"] > parsed_counts["teacherPositions"]:
        raise ContractError("position evidence applies more predictions than teacher rows")
    position_ids: set[str] = set()
    split_by_sfen: dict[str, str] = {}
    observed_source_counts = {"teacher": 0, "selfplay": 0}
    observed_predictions = 0
    observed_teacher_rows: list[Mapping[str, Any]] = []
    observed_selfplay_rows: list[Mapping[str, Any]] = []
    csa_hashes = {reference.sha256 for pair in export_csa_by_job.values() for reference in pair}
    for index, raw_row in enumerate(rows):
        context = f"position evidence.positions[{index}]"
        row = require_mapping(raw_row, context)
        require_exact_keys(row, _EVIDENCE_POSITION_KEYS, context)
        _parse_hard_candidate(row, context, config)
        position_id = require_identifier(row, "positionId", context)
        if position_id in position_ids:
            raise ContractError(f"duplicate evidence position ID: {position_id}")
        position_ids.add(position_id)
        canonical_sfen = _bounded_sfen(row, context)
        split = require_enum(row, "split", context, {"train", "validation", "test"})
        previous_split = split_by_sfen.setdefault(canonical_sfen, split)
        if previous_split != split:
            raise ContractError("the same evidence SFEN appears in more than one data split")
        source_type = require_enum(row, "sourceType", context, {"teacher", "selfplay"})
        observed_source_counts[source_type] += 1
        source = ArtifactRef.from_dict(row.get("sourceManifest"), f"{context}.sourceManifest")
        source_generation = require_identifier(row, "sourceGenerationId", context)
        source_game_id = require_identifier(row, "sourceGameId", context)
        ordinal = require_int(row, "generationOrdinal", context, minimum=0, maximum=1_000_000)
        already_labeled = require_bool(row, "alreadyTeacherLabeled", context)
        if source_type == "teacher":
            if (
                source != teacher_labels
                or source_generation != teacher_generation_id
                or ordinal != 0
                or not already_labeled
            ):
                raise ContractError(f"{context} has inconsistent teacher provenance")
            observed_predictions += int(row.get("modelCp") is not None)
            observed_teacher_rows.append(row)
        elif (
            source != selfplay_manifest
            or source_generation != generation_id
            or ordinal != generation_ordinal
            or split != "train"
            or already_labeled
            or source_game_id not in csa_hashes
        ):
            raise ContractError(f"{context} has inconsistent self-play provenance")
        if source_type == "selfplay" and (
            any(
                row.get(key) is not None
                for key in (
                    "teacherBeforeCp",
                    "teacherAfterCp",
                    "teacherCp",
                    "modelCp",
                    "championMove",
                    "challengerMove",
                    "candidateGapCp",
                    "mateDistance",
                    "searchNodes",
                )
            )
            or row.get("suspectedFailure") != "none"
        ):
            raise ContractError(f"{context} fabricates unavailable self-play measurements")
        if source_type == "selfplay":
            observed_selfplay_rows.append(row)
        if source not in declared_sources:
            raise ContractError(f"{context} uses an undeclared source manifest")
    if observed_source_counts != {
        "teacher": parsed_counts["teacherPositions"],
        "selfplay": parsed_counts["selfplayPositions"],
    }:
        raise ContractError("position evidence source counts disagree with its rows")
    if observed_predictions != parsed_counts["modelPredictionsApplied"]:
        raise ContractError("position evidence model-prediction count disagrees with its rows")
    if expected_teacher_rows is not None and sorted(
        observed_teacher_rows, key=lambda row: str(row["positionId"])
    ) != sorted(expected_teacher_rows, key=lambda row: str(row["positionId"])):
        raise ContractError("position evidence teacher rows are not their deterministic derivation")
    if expected_selfplay_rows is not None and sorted(
        observed_selfplay_rows, key=lambda row: str(row["positionId"])
    ) != sorted(expected_selfplay_rows, key=lambda row: str(row["positionId"])):
        raise ContractError(
            "position evidence self-play rows are not their deterministic derivation"
        )
    return root, declared_sources


class _NoCommandRunner:
    def run(self, *_: object, **__: object) -> Any:
        raise ContractError("validated evidence must reuse its existing Rust export artifacts")


def _parse_hard_candidate(
    table: Mapping[str, Any], context: str, config: SelfPlayConfig
) -> _HardCandidate | None:
    position_id = require_identifier(table, "positionId", context)
    sfen = _bounded_sfen(table, context)
    split = require_enum(table, "split", context, {"train", "validation", "test"})
    require_identifier(table, "sourceGenerationId", context)
    generation_ordinal = require_int(
        table, "generationOrdinal", context, minimum=0, maximum=1_000_000
    )
    source_type = require_enum(
        table, "sourceType", context, {"teacher", "model", "arena", "selfplay"}
    )
    source_manifest = ArtifactRef.from_dict(
        table.get("sourceManifest"), f"{context}.sourceManifest"
    )
    require_identifier(table, "sourceGameId", context)
    require_int(table, "sourcePly", context, minimum=0, maximum=10_000)
    _validate_outcome_fields(table, context, allow_unknown=True)
    teacher_before = _optional_int(table, "teacherBeforeCp", context, -1_000_000, 1_000_000)
    teacher_after = _optional_int(table, "teacherAfterCp", context, -1_000_000, 1_000_000)
    teacher_cp = _optional_int(table, "teacherCp", context, -1_000_000, 1_000_000)
    model_cp = _optional_int(table, "modelCp", context, -1_000_000, 1_000_000)
    champion_move = require_optional_string(table, "championMove", context, maximum_length=16)
    challenger_move = require_optional_string(table, "challengerMove", context, maximum_length=16)
    candidate_gap = _optional_int(table, "candidateGapCp", context, 0, 1_000_000)
    mate_distance = _optional_int(table, "mateDistance", context, -100_000, 100_000)
    phase = require_enum(table, "phase", context, {"opening", "middlegame", "endgame"})
    terminal_boundary = require_bool(table, "terminalBoundary", context)
    search_nodes = _optional_int(table, "searchNodes", context, 0, 1_000_000_000_000)
    suspected = require_enum(
        table, "suspectedFailure", context, {"none", "search", "evaluation", "both"}
    )
    already_labeled = require_bool(table, "alreadyTeacherLabeled", context)
    reasons: list[str] = []
    magnitude_bonus = 0
    if (
        teacher_before is not None
        and teacher_after is not None
        and teacher_before - teacher_after >= config.hard_positions.teacher_drop_cp
    ):
        reasons.append("teacher_evaluation_drop")
        magnitude_bonus += min(50, (teacher_before - teacher_after) // 100)
    if (
        teacher_cp is not None
        and model_cp is not None
        and abs(teacher_cp - model_cp) >= config.hard_positions.evaluation_disagreement_cp
    ):
        reasons.append("teacher_model_disagreement")
        magnitude_bonus += min(50, abs(teacher_cp - model_cp) // 100)
    if (
        champion_move is not None
        and challenger_move is not None
        and champion_move != challenger_move
    ):
        reasons.append("champion_challenger_move_disagreement")
    if candidate_gap is not None and candidate_gap <= config.hard_positions.candidate_gap_cp:
        reasons.append("ambiguous_candidates")
    if mate_distance is not None:
        reasons.append("mate_related")
    if phase == "endgame" and terminal_boundary:
        reasons.append("endgame_boundary")
    if search_nodes is not None and search_nodes < config.hard_positions.minimum_search_nodes:
        reasons.append("insufficient_search")
    if suspected in {"search", "both"}:
        reasons.append("suspected_search_error")
    if suspected in {"evaluation", "both"}:
        reasons.append("suspected_evaluation_error")
    if not reasons:
        return None
    ordered_reasons = tuple(reason for reason in _REASON_ORDER if reason in reasons)
    if split == "test":
        # Test evidence remains available to the explicit final-evaluation path but
        # may never seed Phase 6 training or supplemental labeling selection.
        return None
    return _HardCandidate(
        position_id=position_id,
        sfen=sfen,
        split=split,
        generation_ordinal=generation_ordinal,
        source_manifest=source_manifest,
        source_type=source_type,
        reasons=ordered_reasons,
        priority=sum(_REASON_WEIGHTS[reason] for reason in ordered_reasons) + magnitude_bonus,
        already_teacher_labeled=already_labeled,
    )


def _parse_replay_candidate(table: Mapping[str, Any], context: str) -> _ReplayCandidate:
    side_to_move, outcome_kind, outcome_target = _validate_outcome_fields(
        table, context, allow_unknown=False
    )
    if outcome_target is None:
        raise ContractError(f"{context}.outcomeTarget must be present")
    return _ReplayCandidate(
        position_id=require_identifier(table, "positionId", context),
        sfen=_bounded_sfen(table, context),
        split=require_enum(table, "split", context, {"train", "validation", "test"}),
        generation_id=require_identifier(table, "generationId", context),
        generation_ordinal=require_int(
            table, "generationOrdinal", context, minimum=0, maximum=1_000_000
        ),
        source_type=require_enum(
            table,
            "sourceType",
            context,
            {"audited_game", "teacher", "selfplay", "hard_position", "retained"},
        ),
        source_manifest=ArtifactRef.from_dict(
            table.get("sourceManifest"), f"{context}.sourceManifest"
        ),
        source_game_id=require_identifier(table, "sourceGameId", context),
        source_ply=require_int(table, "sourcePly", context, minimum=0, maximum=10_000),
        side_to_move=side_to_move,
        outcome_kind=outcome_kind,
        outcome_target=outcome_target,
        stage=require_enum(table, "stage", context, {"opening", "middlegame", "endgame"}),
        priority=require_int(table, "priority", context, minimum=0, maximum=1_000_000),
        hard_position=require_bool(table, "hardPosition", context),
        is_new=require_bool(table, "isNew", context),
    )


def _serialize_hard_position(item: Mapping[str, Any]) -> dict[str, object]:
    reasons = [reason for reason in _REASON_ORDER if reason in item["reasons"]]
    return {
        "dedupKey": item["dedupKey"],
        "sfen": item["sfen"],
        "split": item["split"],
        "priority": item["priority"],
        "reasons": reasons,
        "sourcePositionIds": sorted(item["sourcePositionIds"]),
        "sourceManifests": [
            ref.as_dict() for ref in sorted(item["sourceManifests"], key=lambda ref: ref.path)
        ],
        "sourceTypes": sorted(item["sourceTypes"]),
        "newestGenerationOrdinal": item["newestGenerationOrdinal"],
        "alreadyTeacherLabeled": item["alreadyTeacherLabeled"],
    }


def _replay_origin(candidate: _ReplayCandidate) -> dict[str, object]:
    return {
        "positionId": candidate.position_id,
        "generationId": candidate.generation_id,
        "generationOrdinal": candidate.generation_ordinal,
        "sourceType": candidate.source_type,
        "sourceManifest": candidate.source_manifest.as_dict(),
        "sourceGameId": candidate.source_game_id,
        "sourcePly": candidate.source_ply,
        "sideToMove": candidate.side_to_move,
        "outcomeKind": candidate.outcome_kind,
        "outcomeTarget": candidate.outcome_target,
        "stage": candidate.stage,
        "priority": candidate.priority,
        "hardPosition": candidate.hard_position,
        "isNew": candidate.is_new,
    }


def _replay_candidate_rank(candidate: _ReplayCandidate) -> tuple[int, int, int, int, str]:
    """Choose one factual target deterministically when canonical states repeat."""

    return (
        -candidate.priority,
        -int(candidate.hard_position),
        -int(candidate.is_new),
        -candidate.generation_ordinal,
        candidate.position_id,
    )


def _artifact_ref_array(
    table: Mapping[str, Any], key: str, context: str, *, maximum_items: int
) -> tuple[ArtifactRef, ...]:
    values = require_list(table, key, context, maximum_items=maximum_items)
    references = tuple(
        ArtifactRef.from_dict(value, f"{context}.{key}[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(references)) != len(references):
        raise ContractError(f"{context}.{key} contains duplicate references")
    return references


def _validate_hard_positions_manifest(
    raw: object, *, evidence_ref: ArtifactRef, config: SelfPlayConfig
) -> Mapping[str, Any]:
    root = require_mapping(raw, "hard positions")
    require_exact_keys(
        root,
        {
            "schema",
            "generationId",
            "inputEvidence",
            "configSha256",
            "budget",
            "eligibleUniquePositions",
            "positions",
            "manifestSha256",
        },
        "hard positions",
    )
    if root.get("schema") != HARD_POSITIONS_SCHEMA:
        raise ContractError("unsupported hard-position schema")
    require_identifier(root, "generationId", "hard positions")
    if (
        ArtifactRef.from_dict(root.get("inputEvidence"), "hard positions.inputEvidence")
        != evidence_ref
    ):
        raise ContractError("hard positions reference different position evidence")
    if root.get("configSha256") != config.sha256:
        raise ContractError("hard positions use a different configuration")
    eligible = require_int(
        root,
        "eligibleUniquePositions",
        "hard positions",
        minimum=0,
        maximum=MAX_EVIDENCE_POSITIONS,
    )
    budget = require_mapping(root.get("budget"), "hard positions.budget")
    require_exact_keys(
        budget,
        {
            "teacherLabelLimit",
            "labelsBefore",
            "requestedMaximum",
            "availableNewTeacherLabelsBeforeSelection",
            "selected",
            "alreadyLabeledSelected",
            "newTeacherLabelsSelected",
            "labelsAfterMaximum",
        },
        "hard positions.budget",
    )
    parsed_budget: dict[str, int] = {}
    for key in budget:
        parsed_budget[key] = require_int(
            budget, key, "hard positions.budget", minimum=0, maximum=10_000
        )
    if parsed_budget["teacherLabelLimit"] != config.hard_positions.teacher_label_limit:
        raise ContractError("hard-position teacher-label limit disagrees with configuration")
    if parsed_budget["requestedMaximum"] > config.hard_positions.max_additional_labels:
        raise ContractError("hard-position requested maximum exceeds configuration")
    expected_available = parsed_budget["teacherLabelLimit"] - parsed_budget["labelsBefore"]
    if expected_available < 0:
        raise ContractError("hard-position labelsBefore exceeds the teacher-label limit")
    if parsed_budget["availableNewTeacherLabelsBeforeSelection"] != expected_available:
        raise ContractError("hard-position available teacher-label budget disagrees")
    if parsed_budget["newTeacherLabelsSelected"] > expected_available:
        raise ContractError("hard-position selection exceeds the remaining teacher-label budget")
    if (
        parsed_budget["labelsAfterMaximum"]
        != parsed_budget["labelsBefore"] + parsed_budget["newTeacherLabelsSelected"]
    ):
        raise ContractError("hard-position labels-after count disagrees")
    if parsed_budget["selected"] > parsed_budget["requestedMaximum"]:
        raise ContractError("hard-position selected count exceeds the requested maximum")
    if (
        parsed_budget["alreadyLabeledSelected"] + parsed_budget["newTeacherLabelsSelected"]
        != parsed_budget["selected"]
    ):
        raise ContractError("hard-position selection counts disagree")
    rows = require_list(
        root,
        "positions",
        "hard positions",
        maximum_items=MAX_EVIDENCE_POSITIONS,
    )
    if len(rows) != parsed_budget["selected"] or len(rows) > eligible:
        raise ContractError("hard-position row count disagrees with its budget")
    dedup_keys: set[str] = set()
    observed_already_labeled = 0
    for index, raw_row in enumerate(rows):
        context = f"hard positions.positions[{index}]"
        row = require_mapping(raw_row, context)
        require_exact_keys(
            row,
            {
                "dedupKey",
                "sfen",
                "split",
                "priority",
                "reasons",
                "sourcePositionIds",
                "sourceManifests",
                "sourceTypes",
                "newestGenerationOrdinal",
                "alreadyTeacherLabeled",
            },
            context,
        )
        dedup_key = require_sha256(row, "dedupKey", context)
        canonical_sfen = _bounded_sfen(row, context)
        if dedup_key != hashlib.sha256(canonical_sfen.encode("utf-8")).hexdigest():
            raise ContractError(f"{context}.dedupKey disagrees with canonical SFEN")
        if dedup_key in dedup_keys:
            raise ContractError("hard-position manifest contains a duplicate canonical SFEN")
        dedup_keys.add(dedup_key)
        require_enum(row, "split", context, {"train", "validation"})
        require_int(row, "priority", context, minimum=0, maximum=1_000_000)
        reasons = require_list(row, "reasons", context, minimum_items=1, maximum_items=32)
        if any(reason not in _REASON_WEIGHTS for reason in reasons):
            raise ContractError(f"{context}.reasons contains an unknown reason")
        if reasons != [reason for reason in _REASON_ORDER if reason in set(reasons)]:
            raise ContractError(f"{context}.reasons must be unique and canonically ordered")
        identifiers = require_list(
            row, "sourcePositionIds", context, minimum_items=1, maximum_items=10_000
        )
        for source_index, value in enumerate(identifiers):
            validate_table = {"value": value}
            require_identifier(
                validate_table, "value", f"{context}.sourcePositionIds[{source_index}]"
            )
        _artifact_ref_array(row, "sourceManifests", context, maximum_items=10_000)
        source_types = require_list(row, "sourceTypes", context, minimum_items=1, maximum_items=4)
        if any(value not in {"teacher", "model", "arena", "selfplay"} for value in source_types):
            raise ContractError(f"{context}.sourceTypes contains an unknown source type")
        require_int(
            row,
            "newestGenerationOrdinal",
            context,
            minimum=0,
            maximum=1_000_000,
        )
        observed_already_labeled += int(require_bool(row, "alreadyTeacherLabeled", context))
    if observed_already_labeled != parsed_budget["alreadyLabeledSelected"]:
        raise ContractError("hard-position already-labeled count disagrees with its rows")
    expected_hash = require_sha256(root, "manifestSha256", "hard positions")
    without_hash = dict(root)
    without_hash.pop("manifestSha256")
    if canonical_sha256(without_hash) != expected_hash:
        raise ContractError("hard-position manifest self-hash mismatch")
    return root


def _bounded_sfen(table: Mapping[str, Any], context: str) -> str:
    value = require_string(table, "sfen", context, maximum_length=512)
    fields = value.split(" ")
    if (
        len(fields) != 4
        or any(not field for field in fields)
        or fields[1] not in {"b", "w"}
        or not fields[3].isascii()
        or not fields[3].isdecimal()
        or fields[3] != "1"
        or "\n" in value
        or "\r" in value
    ):
        raise ContractError(f"{context}.sfen must be a bounded four-field line")
    return value


def _validate_outcome_fields(
    table: Mapping[str, Any], context: str, *, allow_unknown: bool
) -> tuple[str, str, int | None]:
    side = require_enum(table, "sideToMove", context, {"black", "white"})
    sfen = _bounded_sfen(table, context)
    expected_side = "black" if sfen.split(" ")[1] == "b" else "white"
    if side != expected_side:
        raise ContractError(f"{context}.sideToMove disagrees with SFEN")
    outcomes = {"black_win", "white_win", "draw"}
    if allow_unknown:
        outcomes.add("unknown")
    outcome = require_enum(table, "outcomeKind", context, outcomes)
    raw_target = table.get("outcomeTarget")
    if outcome == "unknown":
        if raw_target is not None:
            raise ContractError(f"{context}.outcomeTarget must be null for unknown outcome")
        return side, outcome, None
    target = require_int(table, "outcomeTarget", context, minimum=-1, maximum=1)
    if outcome == "draw":
        expected_target = 0
    else:
        winner = "black" if outcome == "black_win" else "white"
        expected_target = 1 if side == winner else -1
    if target != expected_target:
        raise ContractError(f"{context}.outcomeTarget disagrees with outcome and side")
    return side, outcome, target


def _optional_int(
    table: Mapping[str, Any], key: str, context: str, minimum: int, maximum: int
) -> int | None:
    if table.get(key) is None:
        return None
    return require_int(table, key, context, minimum=minimum, maximum=maximum)
