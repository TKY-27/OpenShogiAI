"""Versioned model/generation registry and quarantined human-game contracts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    retire_bound_regular,
    stable_directory_lock,
    stable_parent_descriptor,
)
from open_shogi_training.models.export import (
    ARCH_VERSION,
    QUANTIZATION_FLOAT32,
    QUANTIZATION_INT8,
    parse_value_model,
)

from .common import (
    ArtifactRef,
    ContractError,
    contained_path,
    load_bytes_artifact,
    load_json,
    load_json_and_ref,
    load_json_artifact,
    replace_json_state,
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
    validate_identifier,
    validate_relative_path,
    validate_utc_timestamp,
    verified_artifact_descriptor,
    verify_artifact_ref,
    write_json_new,
)

MODEL_REGISTRY_SCHEMA: Final = "phase6_model_registry/v1"
REGISTRY_TRANSACTION_SCHEMA: Final = "phase6_model_registry_transaction/v1"
PENDING_HUMAN_REVIEW_SCHEMA: Final = "phase6_pending_human_review/v1"
MAX_MODELS: Final = 10_000
MAX_GENERATIONS: Final = 10_000
MAX_HUMAN_GAMES: Final = 1_000

_REGISTRY_KEYS = frozenset(
    {"schema", "revision", "championModelId", "challengerModelId", "models", "generations"}
)
_MODEL_KEYS = frozenset(
    {
        "modelId",
        "generationId",
        "parentModelId",
        "artifact",
        "evaluatorKind",
        "architectureVersion",
        "quantization",
        "trainingRun",
        "registeredAt",
        "licenseStatus",
    }
)
_GENERATION_KEYS = frozenset(
    {
        "generationId",
        "parentGenerationId",
        "championModelId",
        "challengerModelId",
        "selfplayManifest",
        "teacherLabelingManifest",
        "trainingRunManifest",
        "arenaManifest",
        "promotionDecision",
        "status",
        "createdAt",
    }
)
_HUMAN_ROOT_KEYS = frozenset(
    {
        "schema",
        "runId",
        "createdAt",
        "status",
        "autoTrainingEligible",
        "configSha256",
        "games",
    }
)
_HUMAN_GAME_KEYS = frozenset(
    {
        "gameId",
        "modelId",
        "modelSha256",
        "configSha256",
        "nodes",
        "elapsedMs",
        "pv",
        "evaluationCp",
        "humanMoves",
        "aiMoves",
        "csa",
        "sfenSequence",
    }
)


class ModelRegistryStore:
    """Atomic registry store with optimistic revision checks and a process lock."""

    def __init__(self, repository_root: Path, relative_path: str) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.path = contained_path(repository_root, relative_path)
        self.transaction_path = self.path.with_name(f".{self.path.name}.transaction.json")

    def create(self, registry: object) -> None:
        validate_model_registry(registry, repository_root=self.repository_root)
        root = require_mapping(registry, "model registry")
        if root.get("revision") != 1:
            raise ContractError("a new registry must start at revision 1")
        with self._lock():
            self._recover_transaction()
            if self.path.exists() or self.path.is_symlink():
                raise ContractError("model registry already exists")
            self._publish_transaction(registry, expected_revision=0)

    def load(self) -> Mapping[str, Any]:
        with self._lock():
            self._recover_transaction()
            raw = load_json(self.path)
            return validate_model_registry(raw, repository_root=self.repository_root)

    def update(self, registry: object, *, expected_revision: int) -> None:
        candidate = validate_model_registry(registry, repository_root=self.repository_root)
        revision = candidate.get("revision")
        if revision != expected_revision + 1:
            raise ContractError("registry update must increment revision by exactly one")
        with self._lock():
            self._recover_transaction()
            current = validate_model_registry(
                load_json(self.path), repository_root=self.repository_root
            )
            if current.get("revision") != expected_revision:
                raise ContractError("model registry revision changed concurrently")
            _validate_registry_delta(current, candidate)
            self._publish_transaction(candidate, expected_revision=expected_revision)

    def _publish_transaction(self, candidate: object, *, expected_revision: int) -> None:
        snapshot = _publish_registry_snapshot(
            self.repository_root,
            self._relative_path(),
            candidate,
        )
        journal = {
            "schema": REGISTRY_TRANSACTION_SCHEMA,
            "expectedRevision": expected_revision,
            "candidate": snapshot.as_dict(),
        }
        replace_json_state(self.transaction_path, journal)
        _registry_failpoint("after_journal")
        self._recover_transaction()

    def _recover_transaction(self) -> None:
        if not self.transaction_path.exists() and not self.transaction_path.is_symlink():
            return
        if self.transaction_path.is_symlink():
            raise ContractError("model registry transaction must not be a symlink")
        journal_value, journal_ref = load_json_and_ref(
            self.repository_root,
            self.transaction_path.relative_to(self.repository_root).as_posix(),
        )
        journal = require_mapping(journal_value, "model registry transaction")
        require_exact_keys(
            journal,
            {"schema", "expectedRevision", "candidate"},
            "model registry transaction",
        )
        if journal.get("schema") != REGISTRY_TRANSACTION_SCHEMA:
            raise ContractError("unsupported model registry transaction schema")
        expected = require_int(
            journal,
            "expectedRevision",
            "model registry transaction",
            minimum=0,
            maximum=1_000_000_000,
        )
        reference = ArtifactRef.from_dict(
            journal.get("candidate"), "model registry transaction.candidate"
        )
        candidate = validate_model_registry(
            load_json_artifact(self.repository_root, reference),
            repository_root=self.repository_root,
        )
        if candidate.get("revision") != expected + 1:
            raise ContractError("model registry transaction revision is invalid")
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink():
                raise ContractError("model registry must not be a symlink")
            current = validate_model_registry(
                load_json(self.path), repository_root=self.repository_root
            )
            current_revision = int(current["revision"])
            if current_revision == expected + 1:
                if current != candidate:
                    raise ContractError("published registry differs from transaction candidate")
            elif current_revision == expected:
                _validate_registry_delta(current, candidate)
                replace_json_state(self.path, candidate)
            else:
                raise ContractError("registry transaction cannot follow the current revision")
        elif expected == 0:
            replace_json_state(self.path, candidate)
        else:
            raise ContractError("registry disappeared during an update transaction")
        _registry_failpoint("after_registry")
        with verified_artifact_descriptor(self.repository_root, journal_ref) as journal_descriptor:
            journal_status = os.fstat(journal_descriptor)
        with stable_parent_descriptor(self.transaction_path, create=False) as (
            parent_descriptor,
            name,
        ):
            retire_bound_regular(
                parent_descriptor,
                name,
                journal_status,
                display=self.transaction_path,
            )
            os.fsync(parent_descriptor)

    def _lock(self) -> _RegistryLock:
        return _RegistryLock(self.path.parent)

    def _relative_path(self) -> str:
        return self.path.relative_to(self.repository_root).as_posix()


def _registry_failpoint(name: str) -> None:
    if os.environ.get("OPEN_SHOGI_REGISTRY_FAILPOINT") == name:
        raise RuntimeError(f"model registry failpoint: {name}")


class _RegistryLock:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._context: Any | None = None

    def __enter__(self) -> _RegistryLock:
        try:
            context = stable_directory_lock(
                self.directory,
                create=True,
                exclusive=True,
                nonblocking=True,
            )
            context.__enter__()
        except BlockingIOError as error:
            raise ContractError("another process owns the model registry directory") from error
        except (OSError, ArtifactError, ContractError) as error:
            raise ContractError(f"cannot lock model registry: {error}") from error
        self._context = context
        return self

    def __exit__(self, *_: object) -> None:
        context = self._context
        self._context = None
        if context is not None:
            context.__exit__(None, None, None)


def registry_snapshot_ref(
    repository_root: Path,
    registry_path: str,
    registry: object,
) -> ArtifactRef:
    """Return the immutable snapshot for one validated current registry revision."""

    root = validate_model_registry(registry, repository_root=repository_root)
    return _publish_registry_snapshot(repository_root, registry_path, root)


def _publish_registry_snapshot(
    repository_root: Path,
    registry_path: str,
    registry: object,
) -> ArtifactRef:
    normalized = validate_relative_path(registry_path)
    path = Path(normalized)
    revision = require_int(
        require_mapping(registry, "model registry"),
        "revision",
        "model registry",
        minimum=1,
        maximum=1_000_000_000,
    )
    snapshot_path = (
        path.parent / f"{path.stem}.revisions" / f"revision-{revision:010d}.json"
    ).as_posix()
    destination = contained_path(repository_root, snapshot_path)
    if destination.exists() or destination.is_symlink():
        observed, reference = load_json_and_ref(repository_root, snapshot_path)
        if observed != registry:
            raise ContractError("model registry revision snapshot conflicts with existing evidence")
    else:
        write_json_new(destination, registry)
        observed, reference = load_json_and_ref(repository_root, snapshot_path)
        if observed != registry:
            raise ContractError("published model registry revision snapshot changed")
    return reference


def validate_model_registry(
    raw: object, *, repository_root: Path | None = None
) -> Mapping[str, Any]:
    root = require_mapping(raw, "model registry")
    require_exact_keys(root, _REGISTRY_KEYS, "model registry")
    if root.get("schema") != MODEL_REGISTRY_SCHEMA:
        raise ContractError("unsupported model registry schema")
    require_int(root, "revision", "model registry", minimum=1, maximum=1_000_000_000)
    champion_id = _optional_identifier(root.get("championModelId"), "championModelId")
    challenger_id = _optional_identifier(root.get("challengerModelId"), "challengerModelId")
    models = require_list(root, "models", "model registry", maximum_items=MAX_MODELS)
    model_ids: set[str] = set()
    generation_ids_from_models: set[str] = set()
    model_generations: dict[str, str] = {}
    parent_models: dict[str, str | None] = {}
    models_by_id: dict[str, Mapping[str, Any]] = {}
    for index, model in enumerate(models):
        context = f"models[{index}]"
        table = require_mapping(model, context)
        require_exact_keys(table, _MODEL_KEYS, context)
        model_id = require_identifier(table, "modelId", context)
        if model_id in model_ids:
            raise ContractError(f"duplicate model ID: {model_id}")
        model_ids.add(model_id)
        models_by_id[model_id] = table
        model_generation = require_identifier(table, "generationId", context)
        generation_ids_from_models.add(model_generation)
        model_generations[model_id] = model_generation
        parent_model = _optional_identifier(table.get("parentModelId"), f"{context}.parentModelId")
        parent_models[model_id] = parent_model
        if parent_model == model_id:
            raise ContractError(f"{context} cannot be its own parent")
        if parent_model is not None and parent_model not in model_ids:
            raise ContractError(f"{context}.parentModelId must reference an earlier model")
        artifact = ArtifactRef.from_dict(table.get("artifact"), f"{context}.artifact")
        require_enum(
            table,
            "evaluatorKind",
            context,
            {"neural"},
        )
        architecture = require_string(table, "architectureVersion", context, maximum_length=16)
        if architecture != "1":
            raise ContractError(f"{context}.architectureVersion must match OSAVAL architecture 1")
        quantization = require_enum(table, "quantization", context, {"float32", "int8"})
        if repository_root is not None:
            _validate_registered_model_artifact(
                repository_root,
                artifact,
                architecture=architecture,
                quantization=quantization,
                context=context,
            )
        training_run = _optional_ref(table.get("trainingRun"), f"{context}.trainingRun")
        if repository_root is not None and training_run is not None:
            verify_artifact_ref(repository_root, training_run)
        validate_utc_timestamp(table.get("registeredAt"), f"{context}.registeredAt")
        require_enum(table, "licenseStatus", context, {"pending-review"})
    if champion_id is not None and champion_id not in model_ids:
        raise ContractError("championModelId is not present in models")
    if challenger_id is not None and challenger_id not in model_ids:
        raise ContractError("challengerModelId is not present in models")
    if champion_id is not None and champion_id == challenger_id:
        raise ContractError("champion and challenger must be different models")
    generations = require_list(root, "generations", "model registry", maximum_items=MAX_GENERATIONS)
    generation_ids: set[str] = set()
    parent_generations: dict[str, str | None] = {}
    generation_outcomes: dict[str, str] = {}
    initial_generation_id: str | None = None
    previous_generation_id: str | None = None
    for index, generation in enumerate(generations):
        context = f"generations[{index}]"
        table = require_mapping(generation, context)
        require_exact_keys(table, _GENERATION_KEYS, context)
        generation_id = require_identifier(table, "generationId", context)
        if generation_id in generation_ids:
            raise ContractError(f"duplicate generation ID: {generation_id}")
        generation_ids.add(generation_id)
        parent_generation = _optional_identifier(
            table.get("parentGenerationId"), f"{context}.parentGenerationId"
        )
        parent_generations[generation_id] = parent_generation
        if parent_generation == generation_id:
            raise ContractError(f"{context} cannot be its own parent")
        if parent_generation is not None and parent_generation not in generation_ids:
            raise ContractError(
                f"{context}.parentGenerationId must reference an earlier generation"
            )
        if parent_generation is None:
            if initial_generation_id is not None:
                raise ContractError("model registry must contain exactly one initial generation")
            initial_generation_id = generation_id
        elif parent_generation != previous_generation_id:
            raise ContractError(
                f"{context}.parentGenerationId must be the immediate previous generation"
            )
        generation_champion = require_identifier(table, "championModelId", context)
        if generation_champion not in model_ids:
            raise ContractError(f"{context}.championModelId is unknown")
        generation_challenger = _optional_identifier(
            table.get("challengerModelId"), f"{context}.challengerModelId"
        )
        if generation_challenger is not None and generation_challenger not in model_ids:
            raise ContractError(f"{context}.challengerModelId is unknown")
        if (
            generation_challenger is not None
            and model_generations[generation_challenger] != generation_id
        ):
            raise ContractError(f"{context}.challengerModelId belongs to another generation")
        references: dict[str, ArtifactRef | None] = {}
        for key in (
            "selfplayManifest",
            "teacherLabelingManifest",
            "trainingRunManifest",
            "arenaManifest",
            "promotionDecision",
        ):
            reference = _optional_ref(table.get(key), f"{context}.{key}")
            references[key] = reference
            if repository_root is not None and reference is not None:
                verify_artifact_ref(repository_root, reference)
        status = require_enum(
            table,
            "status",
            context,
            {"arena", "complete"},
        )
        validate_utc_timestamp(table.get("createdAt"), f"{context}.createdAt")
        champion_model = models_by_id[generation_champion]
        if parent_generation is None:
            if (
                generation_challenger is not None
                or status != "complete"
                or champion_model["generationId"] != generation_id
                or any(
                    references[key] is not None
                    for key in (
                        "selfplayManifest",
                        "teacherLabelingManifest",
                        "arenaManifest",
                        "promotionDecision",
                    )
                )
                or references["trainingRunManifest"]
                != _optional_ref(
                    champion_model.get("trainingRun"), f"{context}.champion.trainingRun"
                )
            ):
                raise ContractError(f"{context} violates the initial-generation lifecycle")
            generation_outcomes[generation_id] = generation_champion
        else:
            if generation_challenger is None:
                raise ContractError(f"{context} non-initial generation requires a challenger")
            challenger_model = models_by_id[generation_challenger]
            parent_outcome = generation_outcomes.get(parent_generation)
            if parent_outcome is None:
                raise ContractError(
                    f"{context}.parentGenerationId has no verified promotion outcome"
                )
            if generation_champion != parent_outcome:
                raise ContractError(
                    f"{context}.championModelId differs from its parent generation outcome"
                )
            if (
                challenger_model["parentModelId"] != generation_champion
                or references["selfplayManifest"] is None
                or references["teacherLabelingManifest"] is None
                or references["trainingRunManifest"] is None
                or _optional_ref(
                    challenger_model.get("trainingRun"), f"{context}.challenger.trainingRun"
                )
                != references["trainingRunManifest"]
            ):
                raise ContractError(f"{context} violates challenger lineage or training evidence")
            arena_complete = references["arenaManifest"] is not None
            decision_complete = references["promotionDecision"] is not None
            if arena_complete != decision_complete:
                raise ContractError(f"{context} arena and promotion evidence must appear together")
            if (status == "arena" and arena_complete) or (
                status == "complete" and not arena_complete
            ):
                raise ContractError(f"{context} status disagrees with promotion evidence")
            if repository_root is not None and references["promotionDecision"] is not None:
                assert references["arenaManifest"] is not None
                decision_root = _validate_generation_promotion_evidence(
                    repository_root=repository_root,
                    promotion_decision=references["promotionDecision"],
                    arena_manifest=references["arenaManifest"],
                    generation_id=generation_id,
                    champion_model_id=generation_champion,
                    challenger_model_id=generation_challenger,
                )
                decision = str(decision_root["decision"])
                generation_outcomes[generation_id] = (
                    generation_challenger if decision == "promoted" else generation_champion
                )
            elif status == "arena":
                generation_outcomes[generation_id] = generation_champion
            else:
                raise ContractError(
                    f"{context} completed promotion outcome requires repository_root validation"
                )
        previous_generation_id = generation_id
    if initial_generation_id is None:
        raise ContractError("model registry must contain exactly one initial generation")
    if not generation_ids_from_models.issubset(generation_ids):
        raise ContractError("every model generationId must be present in generations")
    active_generations = [
        require_mapping(item, "active generation")
        for item in generations
        if isinstance(item, dict) and item.get("status") == "arena"
    ]
    if challenger_id is None:
        if active_generations:
            raise ContractError("registry has an arena generation without an active challenger")
    elif (
        len(active_generations) != 1
        or active_generations[0].get("challengerModelId") != challenger_id
    ):
        raise ContractError("active challenger must belong to exactly one arena generation")
    if active_generations and champion_id != active_generations[0].get("championModelId"):
        raise ContractError("registry champion must remain the active generation incumbent")
    if active_generations and active_generations[0] is not generations[-1]:
        raise ContractError("only the latest generation may remain active")
    if champion_id is not None and not any(
        generation.get("status") == "complete"
        and champion_id in {generation.get("championModelId"), generation.get("challengerModelId")}
        for generation in generations
        if isinstance(generation, dict)
    ):
        raise ContractError("registry champion has no completed generation evidence")
    latest = require_mapping(generations[-1], "latest generation")
    expected_champion = generation_outcomes.get(str(latest["generationId"]))
    if expected_champion is None or champion_id != expected_champion:
        raise ContractError(
            "registry champion transition disagrees with its latest generation outcome"
        )
    return root


def _validate_registry_delta(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Allow exactly one append or one latest-generation lifecycle transition."""

    if int(candidate["revision"]) != int(current["revision"]) + 1:
        raise ContractError("registry delta must increment revision exactly once")
    current_models = list(current["models"])
    candidate_models = list(candidate["models"])
    current_generations = list(current["generations"])
    candidate_generations = list(candidate["generations"])
    appended = (
        len(candidate_models) == len(current_models) + 1
        and candidate_models[:-1] == current_models
        and len(candidate_generations) == len(current_generations) + 1
        and candidate_generations[:-1] == current_generations
        and candidate["championModelId"] == current["championModelId"]
        and candidate["challengerModelId"] == candidate_models[-1].get("modelId")
        and candidate_generations[-1].get("status") == "arena"
    )
    if appended:
        return
    if len(candidate_models) != len(current_models) or candidate_models != current_models:
        raise ContractError("registry update mutated immutable model history")
    if len(candidate_generations) != len(current_generations) or not current_generations:
        raise ContractError("registry update is not one allowed lifecycle delta")
    if candidate_generations[:-1] != current_generations[:-1]:
        raise ContractError("registry update mutated immutable generation history")
    before = require_mapping(current_generations[-1], "current latest generation")
    after = require_mapping(candidate_generations[-1], "candidate latest generation")
    changed = {key for key in _GENERATION_KEYS if before.get(key) != after.get(key)}
    if (
        before.get("status") != "arena"
        or after.get("status") != "complete"
        or changed != {"arenaManifest", "promotionDecision", "status"}
        or before.get("arenaManifest") is not None
        or before.get("promotionDecision") is not None
        or after.get("arenaManifest") is None
        or after.get("promotionDecision") is None
        or candidate.get("challengerModelId") is not None
    ):
        raise ContractError("registry update is not one allowed latest-generation finalization")


def _validate_registered_model_artifact(
    repository_root: Path,
    artifact: ArtifactRef,
    *,
    architecture: str,
    quantization: str,
    context: str,
) -> None:
    try:
        model = parse_value_model(
            load_bytes_artifact(repository_root, artifact, maximum_bytes=64 * 1024 * 1024)
        )
    except ContractError as error:
        raise ContractError(f"{context}.artifact SHA-256 mismatch: {error}") from error
    except (OSError, ValueError) as error:
        raise ContractError(f"{context}.artifact is not valid OSAVAL01: {error}") from error
    parsed_quantization = {
        QUANTIZATION_FLOAT32: "float32",
        QUANTIZATION_INT8: "int8",
    }.get(model.quantization)
    if architecture != str(ARCH_VERSION) or parsed_quantization != quantization:
        raise ContractError(f"{context} metadata disagrees with its OSAVAL01 artifact")


def build_initial_registry(
    *,
    generation_id: str,
    champion_model_id: str,
    champion_artifact: ArtifactRef,
    evaluator_kind: str,
    architecture_version: str,
    quantization: str,
    training_run: ArtifactRef | None,
    registered_at: str,
) -> dict[str, object]:
    """Register the best Phase 5 model as provisional generation zero."""

    validate_identifier(generation_id, "generation_id")
    validate_identifier(champion_model_id, "champion_model_id")
    validate_utc_timestamp(registered_at, "registered_at")
    if evaluator_kind != "neural":
        raise ContractError("Phase 6 registry evaluatorKind must be neural")
    if architecture_version != "1":
        raise ContractError("Phase 6 registry architectureVersion must be 1")
    if quantization not in {"float32", "int8"}:
        raise ContractError("Phase 6 registry quantization must be float32 or int8")
    registry: dict[str, object] = {
        "schema": MODEL_REGISTRY_SCHEMA,
        "revision": 1,
        "championModelId": champion_model_id,
        "challengerModelId": None,
        "models": [
            {
                "modelId": champion_model_id,
                "generationId": generation_id,
                "parentModelId": None,
                "artifact": champion_artifact.as_dict(),
                "evaluatorKind": evaluator_kind,
                "architectureVersion": architecture_version,
                "quantization": quantization,
                "trainingRun": training_run.as_dict() if training_run is not None else None,
                "registeredAt": registered_at,
                "licenseStatus": "pending-review",
            }
        ],
        "generations": [
            {
                "generationId": generation_id,
                "parentGenerationId": None,
                "championModelId": champion_model_id,
                "challengerModelId": None,
                "selfplayManifest": None,
                "teacherLabelingManifest": None,
                "trainingRunManifest": training_run.as_dict() if training_run is not None else None,
                "arenaManifest": None,
                "promotionDecision": None,
                "status": "complete",
                "createdAt": registered_at,
            }
        ],
    }
    validate_model_registry(registry)
    return registry


def register_challenger_generation(
    registry: object,
    *,
    repository_root: Path,
    generation_id: str,
    parent_generation_id: str,
    challenger_model_id: str,
    challenger_artifact: ArtifactRef,
    architecture_version: str,
    quantization: str,
    selfplay_manifest: ArtifactRef,
    teacher_labeling_manifest: ArtifactRef,
    training_run_manifest: ArtifactRef,
    registered_at: str,
) -> dict[str, object]:
    """Append one challenger generation without mutating the current champion."""

    repository_root = repository_root.resolve(strict=True)
    root = validate_model_registry(registry, repository_root=repository_root)
    generation_id = validate_identifier(generation_id, "generation_id")
    parent_generation_id = validate_identifier(parent_generation_id, "parent_generation_id")
    challenger_model_id = validate_identifier(challenger_model_id, "challenger_model_id")
    validate_utc_timestamp(registered_at, "registered_at")
    if architecture_version != "1":
        raise ContractError("challenger architectureVersion must be 1")
    if quantization not in {"float32", "int8"}:
        raise ContractError("challenger quantization must be float32 or int8")
    champion_id = root.get("championModelId")
    if not isinstance(champion_id, str):
        raise ContractError("a challenger generation requires an existing champion")
    models = [dict(require_mapping(item, "registry model")) for item in root["models"]]
    generations = [
        dict(require_mapping(item, "registry generation")) for item in root["generations"]
    ]
    if any(item["modelId"] == challenger_model_id for item in models):
        raise ContractError("challenger model ID already exists")
    if any(item["generationId"] == generation_id for item in generations):
        raise ContractError("generation ID already exists")
    if not generations or generations[-1]["generationId"] != parent_generation_id:
        raise ContractError("parent generation must be the completed immediate registry head")
    if generations[-1]["status"] != "complete":
        raise ContractError("parent generation must be complete before registering a challenger")
    for reference in (
        challenger_artifact,
        selfplay_manifest,
        teacher_labeling_manifest,
        training_run_manifest,
    ):
        verify_artifact_ref(repository_root, reference)
    models.append(
        {
            "modelId": challenger_model_id,
            "generationId": generation_id,
            "parentModelId": champion_id,
            "artifact": challenger_artifact.as_dict(),
            "evaluatorKind": "neural",
            "architectureVersion": architecture_version,
            "quantization": quantization,
            "trainingRun": training_run_manifest.as_dict(),
            "registeredAt": registered_at,
            "licenseStatus": "pending-review",
        }
    )
    generations.append(
        {
            "generationId": generation_id,
            "parentGenerationId": parent_generation_id,
            "championModelId": champion_id,
            "challengerModelId": challenger_model_id,
            "selfplayManifest": selfplay_manifest.as_dict(),
            "teacherLabelingManifest": teacher_labeling_manifest.as_dict(),
            "trainingRunManifest": training_run_manifest.as_dict(),
            "arenaManifest": None,
            "promotionDecision": None,
            "status": "arena",
            "createdAt": registered_at,
        }
    )
    result: dict[str, object] = {
        "schema": MODEL_REGISTRY_SCHEMA,
        "revision": int(root["revision"]) + 1,
        "championModelId": champion_id,
        "challengerModelId": challenger_model_id,
        "models": models,
        "generations": generations,
    }
    validate_model_registry(result, repository_root=repository_root)
    return result


def record_generation_promotion(
    registry: object,
    *,
    generation_id: str,
    arena_manifest: ArtifactRef,
    promotion_decision: ArtifactRef,
    decision: str,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Finalize a generation and update the champion only for an explicit promotion."""

    if repository_root is None:
        raise ContractError("promotion finalization requires repository_root evidence validation")
    root = validate_model_registry(registry)
    generation_id = validate_identifier(generation_id, "generation_id")
    if decision not in {"promoted", "rejected", "inconclusive"}:
        raise ContractError("promotion decision is invalid")
    generations = [
        dict(require_mapping(item, "registry generation")) for item in root["generations"]
    ]
    target = next((item for item in generations if item["generationId"] == generation_id), None)
    if target is None:
        raise ContractError("generation is not present in the registry")
    if target["status"] == "complete":
        raise ContractError("generation is already complete")
    if target["status"] != "arena":
        raise ContractError("only a generation in arena status can be finalized")
    challenger_id = target.get("challengerModelId")
    if not isinstance(challenger_id, str):
        raise ContractError("generation has no challenger to evaluate")
    if root.get("challengerModelId") != challenger_id:
        raise ContractError("generation challenger is not the registry's active challenger")
    decision_root = _validate_generation_promotion_evidence(
        repository_root=repository_root,
        promotion_decision=promotion_decision,
        arena_manifest=arena_manifest,
        generation_id=generation_id,
        champion_model_id=str(target["championModelId"]),
        challenger_model_id=challenger_id,
        claimed_decision=decision,
    )
    if decision_root.get("decision") != decision:
        raise ContractError("promotion decision argument differs from artifact bytes")
    target["arenaManifest"] = arena_manifest.as_dict()
    target["promotionDecision"] = promotion_decision.as_dict()
    target["status"] = "complete"
    champion_id = challenger_id if decision == "promoted" else root["championModelId"]
    result: dict[str, object] = {
        "schema": MODEL_REGISTRY_SCHEMA,
        "revision": int(root["revision"]) + 1,
        "championModelId": champion_id,
        "challengerModelId": None,
        "models": [dict(require_mapping(item, "registry model")) for item in root["models"]],
        "generations": generations,
    }
    validate_model_registry(result, repository_root=repository_root)
    return result


def _validate_generation_promotion_evidence(
    *,
    repository_root: Path,
    promotion_decision: ArtifactRef,
    arena_manifest: ArtifactRef,
    generation_id: str,
    champion_model_id: str,
    challenger_model_id: str,
    claimed_decision: str | None = None,
) -> Mapping[str, Any]:
    """Bind registry lifecycle state through decision -> analysis -> arena results."""

    from .arena import (
        analyze_arena_results,
        validate_arena_analysis,
        validate_promotion_decision,
        validate_promotion_decision_binding,
    )
    from .config import parse_generation_policy_bytes

    decision = validate_promotion_decision(load_json_artifact(repository_root, promotion_decision))
    if claimed_decision is not None and decision.get("decision") != claimed_decision:
        raise ContractError("promotion decision argument differs from artifact bytes")
    analysis_ref = ArtifactRef.from_dict(
        decision.get("arenaAnalysis"),
        "promotion decision.arenaAnalysis",
    )
    analysis = validate_arena_analysis(load_json_artifact(repository_root, analysis_ref))
    expected_identity = (generation_id, champion_model_id, challenger_model_id)
    decision_identity = (
        decision.get("generationId"),
        decision.get("championModelId"),
        decision.get("challengerModelId"),
    )
    analysis_identity = (
        analysis.get("generationId"),
        analysis.get("championModelId"),
        analysis.get("challengerModelId"),
    )
    if decision_identity != expected_identity or analysis_identity != expected_identity:
        raise ContractError("promotion evidence identity differs from registry generation")
    results_ref = ArtifactRef.from_dict(analysis.get("results"), "arena analysis.results")
    if results_ref != arena_manifest:
        raise ContractError("generation arenaManifest differs from promotion analysis results")
    results = load_json_artifact(repository_root, results_ref)
    expected_analysis = analyze_arena_results(
        results,
        results_ref=results_ref,
        repository_root=repository_root,
    )
    if analysis != expected_analysis:
        raise ContractError("promotion analysis is not the deterministic arena derivation")
    policy_ref = ArtifactRef.from_dict(decision.get("policy"), "promotion decision.policy")
    policy = parse_generation_policy_bytes(
        load_bytes_artifact(repository_root, policy_ref, maximum_bytes=64 * 1024),
        policy_ref.path,
    )
    if policy.sha256 != decision.get("policySha256"):
        raise ContractError("promotion policy semantic digest differs from the decision")
    return validate_promotion_decision_binding(
        decision,
        analysis=analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        repository_root=None,
    )


def validate_pending_human_review(
    raw: object, *, repository_root: Path | None = None
) -> Mapping[str, Any]:
    root = require_mapping(raw, "pending human review")
    require_exact_keys(root, _HUMAN_ROOT_KEYS, "pending human review")
    if root.get("schema") != PENDING_HUMAN_REVIEW_SCHEMA:
        raise ContractError("unsupported pending-human-review schema")
    require_identifier(root, "runId", "pending human review")
    validate_utc_timestamp(root.get("createdAt"), "pending human review.createdAt")
    if root.get("status") != "pending_human_review":
        raise ContractError("human games must remain pending_human_review")
    if require_bool(root, "autoTrainingEligible", "pending human review"):
        raise ContractError("pending human games must never be automatically trainable")
    require_sha256(root, "configSha256", "pending human review")
    games = require_list(
        root,
        "games",
        "pending human review",
        minimum_items=1,
        maximum_items=MAX_HUMAN_GAMES,
    )
    game_ids: set[str] = set()
    for index, game in enumerate(games):
        context = f"human games[{index}]"
        table = require_mapping(game, context)
        require_exact_keys(table, _HUMAN_GAME_KEYS, context)
        game_id = require_identifier(table, "gameId", context)
        if game_id in game_ids:
            raise ContractError(f"duplicate human game ID: {game_id}")
        game_ids.add(game_id)
        require_identifier(table, "modelId", context)
        require_sha256(table, "modelSha256", context)
        require_sha256(table, "configSha256", context)
        require_int(table, "nodes", context, minimum=0, maximum=1_000_000_000_000)
        require_int(table, "elapsedMs", context, minimum=0, maximum=604_800_000)
        _string_array(table, "pv", context, maximum_items=1_024)
        evaluation = table.get("evaluationCp")
        if evaluation is not None:
            require_number(table, "evaluationCp", context, minimum=-1_000_000, maximum=1_000_000)
        _string_array(table, "humanMoves", context, maximum_items=10_000)
        _string_array(table, "aiMoves", context, maximum_items=10_000)
        csa_ref = ArtifactRef.from_dict(table.get("csa"), f"{context}.csa")
        sfen_ref = ArtifactRef.from_dict(table.get("sfenSequence"), f"{context}.sfenSequence")
        if repository_root is not None:
            verify_artifact_ref(repository_root, csa_ref)
            verify_artifact_ref(repository_root, sfen_ref)
    return root


def write_pending_human_review(
    path: Path, manifest: object, *, repository_root: Path | None = None
) -> None:
    validate_pending_human_review(manifest, repository_root=repository_root)
    write_json_new(path, manifest)


def _optional_identifier(value: object, context: str) -> str | None:
    if value is None:
        return None
    return validate_identifier(value, context)


def _optional_ref(value: object, context: str) -> ArtifactRef | None:
    if value is None:
        return None
    return ArtifactRef.from_dict(value, context)


def _string_array(
    table: Mapping[str, Any], key: str, context: str, *, maximum_items: int
) -> tuple[str, ...]:
    rows = require_list(table, key, context, maximum_items=maximum_items)
    values: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, str) or not row or len(row) > 512 or "\x00" in row:
            raise ContractError(f"{context}.{key}[{index}] must be a bounded string")
        values.append(row)
    return tuple(values)
