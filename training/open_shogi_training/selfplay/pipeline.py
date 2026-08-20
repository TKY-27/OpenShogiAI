"""One-generation Phase 6 state machine built from immutable stage artifacts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import stable_directory_lock

from .common import (
    ArtifactRef,
    ContractError,
    canonical_sha256,
    contained_path,
    load_bytes_artifact,
    load_json,
    load_json_artifact,
    replace_json_state,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_list,
    require_mapping,
    require_relative_path,
    require_sha256,
    validate_identifier,
    validate_relative_path,
    verified_artifact_descriptor,
    verify_artifact_ref,
    write_json_new,
)

PIPELINE_STATE_SCHEMA: Final = "phase6_generation_pipeline_state/v1"
GENERATION_MANIFEST_SCHEMA: Final = "phase6_generation_manifest/v1"
STAGES: Final = (
    "modelRegistry",
    "selfplayPlan",
    "selfplayManifest",
    "positionEvidence",
    "hardPositions",
    "teacherLabelingManifest",
    "replayManifest",
    "trainingPlan",
    "trainingRunManifest",
    "challengerExportMetadata",
    "challengerModel",
    "arenaPlan",
    "arenaResults",
    "arenaAnalysis",
    "promotionDecision",
    "finalModelRegistry",
)

_STATE_KEYS = frozenset(
    {
        "schema",
        "revision",
        "generationId",
        "parentGenerationId",
        "config",
        "configSha256",
        "outputRoot",
        "status",
        "stages",
    }
)


class GenerationPipeline:
    """Resume-safe coordinator; stage producers remain independently testable."""

    def __init__(
        self,
        *,
        repository_root: Path,
        state_path: str,
        generation_id: str,
        parent_generation_id: str,
        config: ArtifactRef,
        config_sha256: str,
        output_root: str,
    ) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.state_path = contained_path(repository_root, validate_relative_path(state_path))
        self.generation_id = validate_identifier(generation_id, "generation_id")
        self.parent_generation_id = validate_identifier(
            parent_generation_id, "parent_generation_id"
        )
        if self.generation_id == self.parent_generation_id:
            raise ContractError("a generation cannot be its own parent")
        self.config = config
        self.config_sha256 = require_sha256({"value": config_sha256}, "value", "pipeline config")
        self.output_root = validate_relative_path(output_root)

    def initialize(self) -> Mapping[str, Any]:
        with self._lock(exclusive=True):
            self._validate_pipeline_config()
            if self.state_path.exists() or self.state_path.is_symlink():
                raise ContractError("generation state already exists; use resume")
            state = self._empty_state()
            write_json_new(self.state_path, state)
            return state

    def resume(self) -> Mapping[str, Any]:
        with self._lock(exclusive=False):
            return self._resume_unlocked()

    def _resume_unlocked(self) -> Mapping[str, Any]:
        self._validate_pipeline_config()
        state = self._load()
        for reference in state["stages"].values():
            if reference is not None:
                verify_artifact_ref(
                    self.repository_root, ArtifactRef.from_dict(reference, "pipeline stage")
                )
        return state

    def _validate_pipeline_config(self) -> None:
        from .config import parse_selfplay_config_bytes

        config = parse_selfplay_config_bytes(
            load_bytes_artifact(
                self.repository_root,
                self.config,
                maximum_bytes=64 * 1024,
            ),
            self.config.path,
        )
        if config.sha256 != self.config_sha256:
            raise ContractError("pipeline semantic config hash differs from its artifact")
        expected_output = f"{config.paths.output_root}/{self.generation_id}"
        if self.output_root != expected_output:
            raise ContractError(
                "pipeline output root must be CONFIG.paths.output_root/GENERATION_ID"
            )

    def record_stage(self, stage: str, reference: ArtifactRef) -> Mapping[str, Any]:
        if stage not in STAGES:
            raise ContractError(f"unknown generation stage: {stage}")
        with self._lock(exclusive=True):
            verify_artifact_ref(self.repository_root, reference)
            state = dict(self._resume_unlocked())
            revision = int(state["revision"])
            stages = dict(require_mapping(state["stages"], "pipeline stages"))
            existing = stages[stage]
            if existing is not None:
                if ArtifactRef.from_dict(existing, f"stages.{stage}") == reference:
                    return state
                raise ContractError(f"generation stage is immutable once recorded: {stage}")
            next_stage = next(name for name in STAGES if stages[name] is None)
            if stage != next_stage:
                raise ContractError(f"stage {stage} cannot precede required stage {next_stage}")
            self._validate_stage_contract(stage, reference)
            stages[stage] = reference.as_dict()
            state["stages"] = stages
            state["status"] = "running"
            state["revision"] = revision + 1
            self._replace_state_cas(state, expected_revision=revision)
            return state

    def interrupt(self) -> Mapping[str, Any]:
        with self._lock(exclusive=True):
            state = dict(self._load())
            if state["status"] == "complete":
                raise ContractError("a completed generation cannot be interrupted")
            revision = int(state["revision"])
            state["status"] = "interrupted"
            state["revision"] = revision + 1
            self._replace_state_cas(state, expected_revision=revision)
            return state

    def finalize(self, manifest_path: str) -> dict[str, object]:
        with self._lock(exclusive=True):
            state = dict(self._resume_unlocked())
            stages = require_mapping(state["stages"], "pipeline stages")
            missing = [stage for stage in STAGES if stages[stage] is None]
            if missing:
                raise ContractError(f"generation cannot complete with missing stages: {missing}")
            # Validate the immutable chain before publishing the terminal state.  A
            # validation failure must leave the generation resumable instead of
            # stranding an invalid state marked complete.
            for stage in STAGES:
                self._validate_stage_contract(
                    stage,
                    ArtifactRef.from_dict(stages[stage], f"pipeline stages.{stage}"),
                )
            if state["status"] != "complete":
                revision = int(state["revision"])
                state["status"] = "complete"
                state["revision"] = revision + 1
                self._replace_state_cas(state, expected_revision=revision)
                _pipeline_failpoint("after_state")
            manifest: dict[str, object] = {
                "schema": GENERATION_MANIFEST_SCHEMA,
                "generationId": self.generation_id,
                "parentGenerationId": self.parent_generation_id,
                "config": self.config.as_dict(),
                "configSha256": self.config_sha256,
                "outputRoot": self.output_root,
                "status": "complete",
                "stages": dict(stages),
            }
            manifest["manifestSha256"] = canonical_sha256(manifest)
            destination = contained_path(
                self.repository_root, validate_relative_path(manifest_path)
            )
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or load_json(destination) != manifest:
                    raise ContractError(
                        "generation manifest conflicts with finalized pipeline state"
                    )
            else:
                write_json_new(destination, manifest)
            _pipeline_failpoint("after_manifest")
            return manifest

    def _validate_stage_contract(self, stage: str, reference: ArtifactRef) -> None:
        """Reject a valid hash whose JSON schema or generation binding is for another stage."""

        if stage == "challengerModel":
            from open_shogi_training.models.export import parse_value_model

            try:
                parsed = parse_value_model(
                    load_bytes_artifact(
                        self.repository_root, reference, maximum_bytes=64 * 1024 * 1024
                    )
                )
            except ValueError as error:
                raise ContractError("challenger model is not a valid OSAVAL01 artifact") from error
            metadata = self._load_stage("challengerExportMetadata")
            artifacts = require_list(
                metadata,
                "artifacts",
                "challenger export metadata",
                minimum_items=1,
                maximum_items=2,
            )
            matches: list[Mapping[str, Any]] = []
            for index, raw_artifact in enumerate(artifacts):
                artifact = require_mapping(
                    raw_artifact, f"challenger export metadata.artifacts[{index}]"
                )
                artifact_ref = self._metadata_sibling_ref(
                    self._recorded_ref("challengerExportMetadata"),
                    artifact,
                    f"challenger export metadata.artifacts[{index}]",
                )
                if artifact_ref == reference:
                    matches.append(artifact)
            if len(matches) != 1:
                raise ContractError(
                    "challenger model must be exactly one recorded export metadata artifact"
                )
            quantization = {0: "float32", 1: "int8"}.get(parsed.quantization)
            if (
                matches[0].get("quantization") != quantization
                or matches[0].get("payload_sha256") != parsed.payload_sha256
            ):
                raise ContractError("challenger model differs from its export metadata")
            return
        value = load_json_artifact(
            self.repository_root,
            reference,
            maximum_bytes=256 * 1024 * 1024,
            maximum_nodes=8_000_000,
        )
        table = require_mapping(value, f"pipeline stage {stage}")
        expected = {
            "modelRegistry": "phase6_model_registry/v1",
            "selfplayPlan": "phase6_selfplay_plan/v1",
            "selfplayManifest": "phase6_selfplay_manifest/v1",
            "positionEvidence": "phase6_position_evidence/v1",
            "hardPositions": "phase6_hard_positions/v1",
            "teacherLabelingManifest": "phase4_teacher_label_manifest/v2",
            "replayManifest": "phase6_replay_buffer_manifest/v2",
            "trainingPlan": "phase6_challenger_training_plan/v1",
            "trainingRunManifest": "phase4_value_experiment/v1",
            "challengerExportMetadata": "phase4_value_model_metadata/v1",
            "arenaPlan": "phase6_paired_arena_plan/v1",
            "arenaResults": "phase6_paired_arena_results/v1",
            "arenaAnalysis": "phase6_paired_arena_analysis/v1",
            "promotionDecision": "phase6_promotion_decision/v1",
            "finalModelRegistry": "phase6_model_registry/v1",
        }[stage]
        if table.get("schema") != expected:
            raise ContractError(f"pipeline stage {stage} has the wrong schema")
        generation = table.get("generationId")
        if (
            stage
            not in {
                "modelRegistry",
                "teacherLabelingManifest",
                "trainingRunManifest",
                "challengerExportMetadata",
                "finalModelRegistry",
            }
            and generation != self.generation_id
        ):
            raise ContractError(f"pipeline stage {stage} belongs to another generation")

        if stage in {"modelRegistry", "finalModelRegistry"}:
            from .registry import _validate_registry_delta, validate_model_registry

            registry = validate_model_registry(table, repository_root=self.repository_root)
            if stage == "modelRegistry":
                generations = require_list(
                    registry,
                    "generations",
                    "initial pipeline registry",
                    minimum_items=1,
                    maximum_items=10_000,
                )
                latest = require_mapping(generations[-1], "initial pipeline registry latest")
                if (
                    latest.get("generationId") != self.parent_generation_id
                    or latest.get("status") != "complete"
                    or registry.get("challengerModelId") is not None
                ):
                    raise ContractError(
                        "initial registry is not the completed parent-generation head"
                    )
            else:
                arena_plan = self._load_stage("arenaPlan")
                arena_registry_ref = ArtifactRef.from_dict(
                    arena_plan.get("modelRegistry"), "arena plan.modelRegistry"
                )
                arena_registry = validate_model_registry(
                    load_json_artifact(self.repository_root, arena_registry_ref),
                    repository_root=self.repository_root,
                )
                _validate_registry_delta(self._load_stage("modelRegistry"), arena_registry)
                _validate_registry_delta(arena_registry, registry)
                generation_row = next(
                    (
                        item
                        for item in registry["generations"]
                        if isinstance(item, dict) and item.get("generationId") == self.generation_id
                    ),
                    None,
                )
                if generation_row is None or generation_row.get("status") != "complete":
                    raise ContractError("final registry has no completed target generation")
                self._require_ref_field(
                    generation_row,
                    "arenaManifest",
                    self._recorded_ref("arenaResults"),
                    "final registry arena evidence",
                )
                self._require_ref_field(
                    generation_row,
                    "promotionDecision",
                    self._recorded_ref("promotionDecision"),
                    "final registry promotion evidence",
                )
                decision = self._load_stage("promotionDecision")
                expected_champion = (
                    generation_row.get("challengerModelId")
                    if decision.get("decision") == "promoted"
                    else generation_row.get("championModelId")
                )
                if (
                    registry.get("championModelId") != expected_champion
                    or registry.get("challengerModelId") is not None
                ):
                    raise ContractError("final registry champion differs from promotion outcome")
            return

        if stage in {"selfplayPlan", "arenaPlan"}:
            from .execution import _validate_paired_plan_artifacts, validate_paired_plan

            validate_paired_plan(table)
            self._require_ref_field(table, "config", self.config, f"{stage} configuration")
            if table.get("configSha256") != self.config_sha256:
                raise ContractError(f"pipeline stage {stage} config digest differs")
            registry_ref = ArtifactRef.from_dict(
                table.get("modelRegistry"), f"{stage} model registry"
            )
            if table.get("engineBuildReceipt") is None:
                raise ContractError(f"pipeline stage {stage} requires an engine build receipt")
            _validate_paired_plan_artifacts(
                table,
                repository_root=self.repository_root,
                runtime_authorization=True,
            )
            if stage == "selfplayPlan":
                if registry_ref != self._recorded_ref("modelRegistry"):
                    raise ContractError("self-play plan registry differs from the recorded stage")
                initial_registry = self._load_stage("modelRegistry")
                champion = require_mapping(table.get("champion"), "self-play plan.champion")
                if table.get("generationId") != self.generation_id or champion.get(
                    "modelId"
                ) != initial_registry.get("championModelId"):
                    raise ContractError("self-play plan is not the target generation champion run")
            else:
                from .registry import _validate_registry_delta, validate_model_registry

                registry = validate_model_registry(
                    load_json_artifact(self.repository_root, registry_ref),
                    repository_root=self.repository_root,
                )
                _validate_registry_delta(self._load_stage("modelRegistry"), registry)
                champion = require_mapping(table.get("champion"), "arena plan.champion")
                challenger = require_mapping(table.get("challenger"), "arena plan.challenger")
                if registry.get("championModelId") != champion.get("modelId") or registry.get(
                    "challengerModelId"
                ) != challenger.get("modelId"):
                    raise ContractError("arena plan model IDs differ from its registry")
                generations = require_list(
                    registry, "generations", "arena registry", maximum_items=10_000
                )
                target = require_mapping(generations[-1], "arena registry target generation")
                if (
                    target.get("generationId") != self.generation_id
                    or target.get("parentGenerationId") != self.parent_generation_id
                    or target.get("status") != "arena"
                    or target.get("selfplayManifest")
                    != self._recorded_ref("selfplayManifest").as_dict()
                    or target.get("teacherLabelingManifest")
                    != self._recorded_ref("teacherLabelingManifest").as_dict()
                    or target.get("trainingRunManifest")
                    != self._recorded_ref("trainingRunManifest").as_dict()
                    or challenger.get("artifact") != self._recorded_ref("challengerModel").as_dict()
                ):
                    raise ContractError("arena registry is not the exact pipeline challenger delta")
            return

        if stage == "selfplayManifest":
            from .execution import validate_execution_manifest

            plan_ref = self._recorded_ref("selfplayPlan")
            plan = self._load_stage("selfplayPlan")
            self._require_ref_field(
                table,
                "plan",
                plan_ref,
                "self-play execution plan",
            )
            validate_execution_manifest(
                table,
                repository_root=self.repository_root,
                manifest_ref=reference,
                plan=plan,
                plan_ref=plan_ref,
            )
            return

        if stage == "positionEvidence":
            from .config import parse_selfplay_config_bytes
            from .evidence import _validate_position_evidence

            derivation = require_mapping(table.get("derivation"), "position evidence.derivation")
            self._require_ref_field(
                derivation,
                "selfplayPlan",
                self._recorded_ref("selfplayPlan"),
                "position-evidence self-play plan",
            )
            self._require_ref_field(
                derivation,
                "selfplayManifest",
                self._recorded_ref("selfplayManifest"),
                "position-evidence self-play manifest",
            )
            config = parse_selfplay_config_bytes(
                load_bytes_artifact(self.repository_root, self.config, maximum_bytes=64 * 1024),
                self.config.path,
            )
            _validate_position_evidence(
                table,
                evidence_ref=reference,
                config=config,
                repository_root=self.repository_root,
            )
            return

        if stage == "hardPositions":
            from .config import parse_selfplay_config_bytes
            from .evidence import extract_hard_positions

            evidence_ref = self._recorded_ref("positionEvidence")
            self._require_ref_field(
                table,
                "inputEvidence",
                evidence_ref,
                "hard-position evidence",
            )
            if table.get("configSha256") != self.config_sha256:
                raise ContractError("hard-position config digest differs")
            config = parse_selfplay_config_bytes(
                load_bytes_artifact(self.repository_root, self.config, maximum_bytes=64 * 1024),
                self.config.path,
            )
            budget = require_mapping(table.get("budget"), "hard positions.budget")
            expected = extract_hard_positions(
                self._load_stage("positionEvidence"),
                evidence_ref=evidence_ref,
                config=config,
                labels_before=require_int(
                    budget,
                    "labelsBefore",
                    "hard positions.budget",
                    minimum=0,
                    maximum=10_000,
                ),
                requested_max=require_int(
                    budget,
                    "requestedMaximum",
                    "hard positions.budget",
                    minimum=0,
                    maximum=10_000,
                ),
                repository_root=self.repository_root,
            )
            if table != expected:
                raise ContractError("hard positions are not the deterministic evidence selection")
            return

        if stage == "teacherLabelingManifest":
            self._validate_teacher_label_manifest_stage(table, reference)
            return

        if stage == "replayManifest":
            from .config import parse_selfplay_config_bytes
            from .evidence import build_replay_buffer_manifest

            config = parse_selfplay_config_bytes(
                load_bytes_artifact(self.repository_root, self.config, maximum_bytes=64 * 1024),
                self.config.path,
            )
            candidates_ref = ArtifactRef.from_dict(
                table.get("inputCandidates"), "replay manifest.inputCandidates"
            )
            expected = build_replay_buffer_manifest(
                load_json_artifact(
                    self.repository_root,
                    candidates_ref,
                    maximum_bytes=64 * 1024 * 1024,
                    maximum_nodes=8_000_000,
                ),
                candidates_ref=candidates_ref,
                config=config,
                config_ref=self.config,
                repository_root=self.repository_root,
            )
            if table != expected:
                raise ContractError("replay manifest is not its deterministic evidence derivation")
            candidates = require_mapping(
                load_json_artifact(self.repository_root, candidates_ref),
                "replay candidates",
            )
            self._require_ref_field(
                candidates,
                "positionEvidence",
                self._recorded_ref("positionEvidence"),
                "replay candidate evidence",
            )
            self._require_ref_field(
                candidates,
                "hardPositions",
                self._recorded_ref("hardPositions"),
                "replay candidate hard positions",
            )
            return

        if stage == "trainingPlan":
            from .planning import parse_start_positions, validate_training_plan

            validate_training_plan(table)
            inputs = require_mapping(table.get("inputs"), "training plan.inputs")
            self._require_ref_field(
                inputs,
                "replayManifest",
                self._recorded_ref("replayManifest"),
                "training-plan replay input",
            )
            self._require_ref_field(
                inputs,
                "teacherLabelManifest",
                self._recorded_ref("teacherLabelingManifest"),
                "training-plan label input",
            )
            references = {
                key: ArtifactRef.from_dict(inputs.get(key), f"training plan.inputs.{key}")
                for key in (
                    "replayManifest",
                    "teacherLabels",
                    "teacherLabelManifest",
                    "positions",
                    "datasetManifest",
                    "featuresConfig",
                    "modelConfig",
                    "trainingConfig",
                )
            }
            label_manifest = self._load_stage("teacherLabelingManifest")
            labels_record = require_mapping(
                require_mapping(
                    label_manifest.get("artifacts"), "teacher label manifest.artifacts"
                ).get("labels.jsonl"),
                "teacher label manifest labels",
            )
            expected_labels = self._manifest_sibling_ref(
                self._recorded_ref("teacherLabelingManifest"),
                "labels.jsonl",
                labels_record,
                "teacher labels",
            )
            if references["teacherLabels"] != expected_labels:
                raise ContractError("training plan labels differ from the recorded label snapshot")
            selfplay_plan = self._load_stage("selfplayPlan")
            if references["datasetManifest"] != ArtifactRef.from_dict(
                selfplay_plan.get("datasetManifest"), "self-play plan.datasetManifest"
            ):
                raise ContractError("training plan dataset differs from the self-play lineage")
            starts_ref = ArtifactRef.from_dict(
                selfplay_plan.get("startPositions"), "self-play plan.startPositions"
            )
            starts = parse_start_positions(load_json_artifact(self.repository_root, starts_ref))
            if references["positions"] != starts.source_positions:
                raise ContractError("training plan positions differ from the Phase 3 start source")
            initial_registry = self._load_stage("modelRegistry")
            parent = require_mapping(table.get("parentModel"), "training plan.parentModel")
            if parent.get("modelId") != initial_registry.get("championModelId"):
                raise ContractError("training plan parent is not the recorded registry champion")
            _, _, _, loaded = self._load_training_inputs(references)
            expected_identity = {
                "dataset_manifest_sha256": references["datasetManifest"].sha256,
                "positions_sha256": references["positions"].sha256,
                "labels_sha256": references["teacherLabels"].sha256,
                "label_manifest_sha256": references["teacherLabelManifest"].sha256,
                "replay_manifest_sha256": references["replayManifest"].sha256,
            }
            if {
                key: getattr(loaded.identity, key) for key in expected_identity
            } != expected_identity:
                raise ContractError("training plan input identity differs after full dataset audit")
            if table.get("outputDir") != f"{self.output_root}/training":
                raise ContractError("training plan output directory differs from its generation")
            return

        if stage == "trainingRunManifest":
            expected_keys = {
                "schema",
                "mode",
                "configSha256",
                "configs",
                "datasetIdentity",
                "countsBySplit",
                "countsByStage",
                "trainExamplesUsed",
                "validationExamplesUsed",
                "testExamplesUsed",
                "replayExamplesUsed",
                "supervisionCounts",
                "configuredEpochs",
                "completedEpochs",
                "globalStep",
                "bestValidationLoss",
                "device",
                "runtime",
                "executions",
                "resources",
                "artifacts",
            }
            require_exact_keys(table, expected_keys, "training run manifest")
            training_plan = self._load_stage("trainingPlan")
            inputs = require_mapping(training_plan.get("inputs"), "training plan.inputs")
            identity = require_mapping(table.get("datasetIdentity"), "training dataset identity")
            require_exact_keys(
                identity,
                {
                    "dataset_manifest_sha256",
                    "positions_sha256",
                    "labels_sha256",
                    "label_manifest_sha256",
                    "replay_manifest_sha256",
                },
                "training dataset identity",
            )
            expected_bindings = {
                "dataset_manifest_sha256": ArtifactRef.from_dict(
                    inputs["datasetManifest"], "training inputs.datasetManifest"
                ).sha256,
                "positions_sha256": ArtifactRef.from_dict(
                    inputs["positions"], "training inputs.positions"
                ).sha256,
                "labels_sha256": ArtifactRef.from_dict(
                    inputs["teacherLabels"], "training inputs.teacherLabels"
                ).sha256,
                "label_manifest_sha256": self._recorded_ref("teacherLabelingManifest").sha256,
                "replay_manifest_sha256": self._recorded_ref("replayManifest").sha256,
            }
            if dict(identity) != expected_bindings:
                raise ContractError("training run dataset identity differs from its plan")
            if reference.path != f"{training_plan['outputDir']}/experiment.json":
                raise ContractError("training run manifest is outside its planned output")
            input_refs = {
                key: ArtifactRef.from_dict(inputs.get(key), f"training plan.inputs.{key}")
                for key in (
                    "replayManifest",
                    "teacherLabels",
                    "teacherLabelManifest",
                    "positions",
                    "datasetManifest",
                    "featuresConfig",
                    "modelConfig",
                    "trainingConfig",
                )
            }
            features, model, training, loaded = self._load_training_inputs(input_refs)
            from open_shogi_training.models.checkpoint import load_checkpoint_with_identity
            from open_shogi_training.models.config import combined_config_sha256

            if (
                table.get("mode") != "train"
                or table.get("configs")
                != {
                    "features": features.as_dict(),
                    "model": model.as_dict(),
                    "training": training.as_dict(),
                }
                or table.get("configSha256") != combined_config_sha256(features, model, training)
                or table.get("countsBySplit") != loaded.counts_by_split
                or table.get("countsByStage") != loaded.counts_by_stage
                or table.get("trainExamplesUsed") != loaded.counts_by_split["train"]
                or table.get("validationExamplesUsed") != loaded.counts_by_split["validation"]
                or table.get("testExamplesUsed") != 0
                or table.get("configuredEpochs") != training.epochs
                or table.get("completedEpochs") != training.epochs
            ):
                raise ContractError("training run manifest differs from its deterministic inputs")
            expected_replay = sum(
                row.source_kind == "phase6_replay" and row.split == "train"
                for row in loaded.examples
            )
            if table.get("replayExamplesUsed") != expected_replay:
                raise ContractError("training run replay count differs from its dataset")
            expected_supervision = {
                "trainTeacher": sum(
                    row.teacher_mask == 1.0 and row.split == "train" for row in loaded.examples
                ),
                "trainOutcome": sum(
                    row.outcome_mask == 1.0 and row.split == "train" for row in loaded.examples
                ),
                "trainPolicyAgreement": sum(
                    row.policy_mask == 1.0 and row.split == "train" for row in loaded.examples
                ),
                "trainReplayOutcomeOnly": sum(
                    row.split == "train"
                    and row.source_kind == "phase6_replay"
                    and row.teacher_mask == 0.0
                    and row.policy_mask == 0.0
                    and row.outcome_mask == 1.0
                    for row in loaded.examples
                ),
                "validationTeacher": sum(
                    row.teacher_mask == 1.0 and row.split == "validation" for row in loaded.examples
                ),
                "validationOutcome": sum(
                    row.outcome_mask == 1.0 and row.split == "validation" for row in loaded.examples
                ),
                "validationPolicyAgreement": sum(
                    row.policy_mask == 1.0 and row.split == "validation" for row in loaded.examples
                ),
            }
            if table.get("supervisionCounts") != expected_supervision:
                raise ContractError("training run supervision counts differ from its dataset")
            runtime = require_mapping(table.get("runtime"), "training run.runtime")
            selfplay_plan = self._load_stage("selfplayPlan")
            if (
                runtime.get("gitDirty") != "false"
                or runtime.get("gitCommit") != selfplay_plan.get("gitCommit")
                or runtime.get("deterministicAlgorithms") != "true"
            ):
                raise ContractError("training runtime is not the immutable deterministic HEAD")
            artifacts = require_mapping(table.get("artifacts"), "training run.artifacts")
            require_exact_keys(
                artifacts,
                {"bestCheckpoint", "lastCheckpoint", "trainingLog"},
                "training run.artifacts",
            )
            best_ref = self._output_artifact_ref(
                reference, artifacts.get("bestCheckpoint"), "best.pt", "best checkpoint"
            )
            last_ref = self._output_artifact_ref(
                reference, artifacts.get("lastCheckpoint"), "last.pt", "last checkpoint"
            )
            self._output_artifact_ref(
                reference,
                artifacts.get("trainingLog"),
                "training-log.jsonl",
                "training log",
            )
            best, best_sha256, best_size = load_checkpoint_with_identity(
                contained_path(self.repository_root, best_ref.path)
            )
            last, last_sha256, last_size = load_checkpoint_with_identity(
                contained_path(self.repository_root, last_ref.path)
            )
            if (
                (best_sha256, best_size) != (best_ref.sha256, best_ref.size)
                or (last_sha256, last_size) != (last_ref.sha256, last_ref.size)
                or best.get("config_sha256") != table.get("configSha256")
                or last.get("config_sha256") != table.get("configSha256")
                or best.get("dataset_identity") != dict(identity)
                or last.get("dataset_identity") != dict(identity)
                or last.get("completed_epoch") != table.get("completedEpochs")
                or last.get("global_step") != table.get("globalStep")
                or best.get("best_validation_loss") != table.get("bestValidationLoss")
                or last.get("best_validation_loss") != table.get("bestValidationLoss")
            ):
                raise ContractError("training checkpoints differ from the experiment evidence")
            best_loss = table.get("bestValidationLoss")
            if (
                isinstance(best_loss, bool)
                or not isinstance(best_loss, (int, float))
                or not math.isfinite(float(best_loss))
            ):
                raise ContractError("training best validation loss is not finite")
            executions = require_list(
                table,
                "executions",
                "training run",
                minimum_items=1,
                maximum_items=training.epochs + 1,
            )
            final_execution = require_mapping(executions[-1], "training run final execution")
            if final_execution.get("endEpoch") != training.epochs:
                raise ContractError("training executions do not reach their configured epoch")
            resources = require_mapping(table.get("resources"), "training run.resources")
            peak = require_int(
                resources,
                "processPeakResidentBytes",
                "training run.resources",
                minimum=1,
                maximum=20 * 1024**3,
            )
            if peak > 20 * 1024**3:
                raise ContractError("training exceeded the 20 GiB working-memory ceiling")
            return

        if stage == "challengerExportMetadata":
            self._validate_export_metadata(table, reference)
            return

        if stage == "arenaResults":
            from .arena import analyze_arena_results

            self._require_ref_field(
                table,
                "plan",
                self._recorded_ref("arenaPlan"),
                "arena-results plan",
            )
            analysis = analyze_arena_results(
                table,
                results_ref=reference,
                repository_root=self.repository_root,
            )
            arena_plan = self._load_stage("arenaPlan")
            champion = require_mapping(arena_plan.get("champion"), "arena plan.champion")
            challenger = require_mapping(arena_plan.get("challenger"), "arena plan.challenger")
            if (
                analysis.get("generationId") != self.generation_id
                or analysis.get("championModelId") != champion.get("modelId")
                or analysis.get("challengerModelId") != challenger.get("modelId")
            ):
                raise ContractError("arena results model identity differs from the arena plan")
            return

        if stage == "arenaAnalysis":
            from .arena import analyze_arena_results

            results_ref = self._recorded_ref("arenaResults")
            self._require_ref_field(
                table,
                "results",
                results_ref,
                "arena-analysis results",
            )
            expected = analyze_arena_results(
                self._load_stage("arenaResults"),
                results_ref=results_ref,
                repository_root=self.repository_root,
            )
            if table != expected:
                raise ContractError("arena analysis is not the deterministic results derivation")
            return

        if stage == "promotionDecision":
            from .arena import validate_promotion_decision_binding
            from .config import parse_generation_policy_bytes

            analysis_ref = self._recorded_ref("arenaAnalysis")
            self._require_ref_field(
                table,
                "arenaAnalysis",
                analysis_ref,
                "promotion analysis",
            )
            policy_ref = ArtifactRef.from_dict(table.get("policy"), "promotion decision.policy")
            policy = parse_generation_policy_bytes(
                load_bytes_artifact(
                    self.repository_root,
                    policy_ref,
                    maximum_bytes=64 * 1024,
                ),
                policy_ref.path,
            )
            if table.get("policySha256") != policy.sha256:
                raise ContractError("promotion policy semantic digest differs")
            analysis = self._load_stage("arenaAnalysis")
            validate_promotion_decision_binding(
                table,
                analysis=analysis,
                analysis_ref=analysis_ref,
                policy=policy,
                policy_ref=policy_ref,
                repository_root=self.repository_root,
            )
            arena_plan = self._load_stage("arenaPlan")
            champion = require_mapping(arena_plan.get("champion"), "arena plan.champion")
            challenger = require_mapping(arena_plan.get("challenger"), "arena plan.challenger")
            if (
                table.get("generationId") != self.generation_id
                or table.get("championModelId") != champion.get("modelId")
                or table.get("challengerModelId") != challenger.get("modelId")
            ):
                raise ContractError("promotion decision identity differs from the arena plan")
            return

        raise AssertionError(f"unhandled pipeline stage contract: {stage}")

    def _validate_export_metadata(
        self, table: Mapping[str, Any], metadata_ref: ArtifactRef
    ) -> None:
        """Bind export metadata and every exported byte to the completed training run."""

        from open_shogi_training.models.checkpoint import load_checkpoint_with_identity
        from open_shogi_training.models.config import (
            config_sha256,
            estimated_export_bytes,
            parse_feature_config,
            parse_model_config,
            parse_training_config,
            validate_config_compatibility,
        )
        from open_shogi_training.models.export import (
            ACTIVATION_RELU,
            ARCH_VERSION,
            FORMAT_VERSION,
            MAGIC,
            MAX_NON_MATE_CP,
            QUANTIZATION_FLOAT32,
            QUANTIZATION_INT8,
            parse_value_model,
        )
        from open_shogi_training.models.features import (
            FEATURE_SCHEMA_VERSION,
            feature_flags,
            feature_schema,
            input_dimension,
        )

        require_exact_keys(
            table,
            {"schema", "format", "model", "featureSchema", "configs", "provenance", "artifacts"},
            "challenger export metadata",
        )
        training_plan = self._load_stage("trainingPlan")
        expected_metadata_path = f"{training_plan['outputDir']}/export/value_v0.metadata.json"
        if metadata_ref.path != expected_metadata_path:
            raise ContractError("challenger export metadata is outside its planned output")

        experiment_ref = self._recorded_ref("trainingRunManifest")
        experiment = self._load_stage("trainingRunManifest")
        configs = require_mapping(table.get("configs"), "challenger export metadata.configs")
        require_exact_keys(configs, {"features", "model", "training"}, "export metadata.configs")
        if configs != experiment.get("configs"):
            raise ContractError("export metadata configs differ from the training run")
        feature_config = parse_feature_config(configs["features"])
        model_config = parse_model_config(configs["model"])
        training_config = parse_training_config(configs["training"])
        validate_config_compatibility(model_config, training_config, feature_config)

        expected_format = {
            "magic": MAGIC.decode("ascii"),
            "formatVersion": FORMAT_VERSION,
            "architectureVersion": ARCH_VERSION,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
            "endianness": "little",
            "weightOrder": "output_major_row_major",
        }
        if table.get("format") != expected_format:
            raise ContractError("export metadata format contract is not OSAVAL architecture v1")
        expected_model = {
            "name": model_config.name,
            "inputDimension": input_dimension(feature_config),
            "hiddenLayers": model_config.hidden_layers,
            "hiddenDimension": model_config.hidden_dim,
            "activation": "relu",
            "dropoutTrainingOnly": model_config.dropout,
            "outputScaleCp": model_config.output_scale_cp,
            "outputPerspective": "current_side_to_move",
            "centipawnConversion": {
                "rounding": "nearest_half_away_from_zero",
                "clamp": [-MAX_NON_MATE_CP, MAX_NON_MATE_CP],
            },
            "valueHead": {"trained": True, "exported": True},
            "policyAgreementAuxiliaryHead": {
                "trained": model_config.auxiliary_policy_head,
                "target": "recordedMove_equals_teacherBestmove",
                "exported": False,
            },
        }
        if table.get("model") != expected_model:
            raise ContractError("export metadata model contract differs from its config")

        feature_record = require_mapping(
            table.get("featureSchema"), "challenger export metadata.featureSchema"
        )
        require_exact_keys(
            feature_record,
            {"path", "sha256", "size", "configSha256", "featureFlags"},
            "challenger export metadata.featureSchema",
        )
        if (
            feature_record.get("path") != "value_v0.feature-schema.json"
            or feature_record.get("configSha256") != config_sha256(feature_config)
            or feature_record.get("featureFlags") != feature_flags(feature_config)
        ):
            raise ContractError("export feature-schema metadata differs from its config")
        feature_ref = self._metadata_sibling_ref(
            metadata_ref,
            feature_record,
            "challenger export metadata.featureSchema",
            allow_payload_digest=False,
        )
        if load_json_artifact(self.repository_root, feature_ref) != feature_schema(feature_config):
            raise ContractError("exported feature schema is not its deterministic config schema")

        experiment_artifacts = require_mapping(
            experiment.get("artifacts"), "training run.artifacts"
        )
        best_ref = self._output_artifact_ref(
            experiment_ref,
            experiment_artifacts.get("bestCheckpoint"),
            "best.pt",
            "best checkpoint",
        )
        checkpoint, checkpoint_sha256, checkpoint_size = load_checkpoint_with_identity(
            contained_path(self.repository_root, best_ref.path)
        )
        if (checkpoint_sha256, checkpoint_size) != (best_ref.sha256, best_ref.size):
            raise ContractError("best checkpoint identity changed before export validation")
        provenance = require_mapping(table.get("provenance"), "export metadata.provenance")
        require_exact_keys(
            provenance,
            {
                "checkpointPath",
                "checkpointSchema",
                "checkpointSha256",
                "checkpointSize",
                "completedEpoch",
                "globalStep",
                "bestValidationLoss",
                "configSha256",
                "datasetIdentity",
                "runtime",
                "exporterModelCodeSha256",
                "auxiliaryHeadExported",
            },
            "export metadata.provenance",
        )
        expected_provenance = {
            "checkpointPath": "best.pt",
            "checkpointSchema": checkpoint.get("schema"),
            "checkpointSha256": best_ref.sha256,
            "checkpointSize": best_ref.size,
            "completedEpoch": checkpoint.get("completed_epoch"),
            "globalStep": checkpoint.get("global_step"),
            "bestValidationLoss": checkpoint.get("best_validation_loss"),
            "configSha256": checkpoint.get("config_sha256"),
            "datasetIdentity": checkpoint.get("dataset_identity"),
            "runtime": checkpoint.get("runtime"),
            "exporterModelCodeSha256": require_mapping(
                checkpoint.get("runtime"), "checkpoint.runtime"
            ).get("modelCodeSha256"),
            "auxiliaryHeadExported": False,
        }
        if dict(provenance) != expected_provenance:
            raise ContractError("export provenance differs from the exact best checkpoint")

        rows = require_list(
            table,
            "artifacts",
            "challenger export metadata",
            minimum_items=1,
            maximum_items=2,
        )
        expected_quantizations = {
            "float32": ("value_v0.f32.osaval", QUANTIZATION_FLOAT32),
            "int8": ("value_v0.int8.osaval", QUANTIZATION_INT8),
        }
        configured = (
            {"float32", "int8"}
            if training_config.quantization == "both"
            else {training_config.quantization}
        )
        observed: set[str] = set()
        for index, raw_row in enumerate(rows):
            context = f"challenger export metadata.artifacts[{index}]"
            row = require_mapping(raw_row, context)
            artifact_ref = self._metadata_sibling_ref(metadata_ref, row, context)
            quantization = require_enum(row, "quantization", context, set(expected_quantizations))
            if quantization in observed:
                raise ContractError("export metadata repeats a quantization")
            observed.add(quantization)
            expected_name, expected_code = expected_quantizations[quantization]
            if PurePosixPath(artifact_ref.path).name != expected_name:
                raise ContractError("export artifact name differs from its quantization")
            if artifact_ref.size != estimated_export_bytes(
                feature_config, model_config, quantization
            ):
                raise ContractError("export artifact size differs from its exact architecture")
            try:
                parsed = parse_value_model(
                    load_bytes_artifact(
                        self.repository_root,
                        artifact_ref,
                        maximum_bytes=64 * 1024 * 1024,
                    )
                )
            except ValueError as error:
                raise ContractError(f"{context} is not valid OSAVAL01") from error
            if (
                parsed.quantization != expected_code
                or parsed.activation != ACTIVATION_RELU
                or parsed.input_dim != input_dimension(feature_config)
                or parsed.hidden_layers != model_config.hidden_layers
                or parsed.hidden_dim != model_config.hidden_dim
                or parsed.feature_flags != feature_flags(feature_config)
                or parsed.output_scale_cp != model_config.output_scale_cp
                or parsed.payload_sha256 != row.get("payload_sha256")
            ):
                raise ContractError("exported OSAVAL bytes differ from their metadata/config")
        if observed != configured:
            raise ContractError("export metadata quantizations differ from training config")

    @staticmethod
    def _output_artifact_ref(
        manifest_ref: ArtifactRef,
        raw: object,
        expected_name: str,
        context: str,
    ) -> ArtifactRef:
        record = require_mapping(raw, context)
        require_exact_keys(record, {"path", "sha256", "size"}, context)
        if record.get("path") != expected_name:
            raise ContractError(f"{context} has an unexpected output filename")
        return ArtifactRef(
            (PurePosixPath(manifest_ref.path).parent / expected_name).as_posix(),
            require_sha256(record, "sha256", context),
            require_int(record, "size", context, minimum=1, maximum=4 * 1024**3),
        )

    @staticmethod
    def _metadata_sibling_ref(
        metadata_ref: ArtifactRef,
        raw: Mapping[str, Any],
        context: str,
        *,
        allow_payload_digest: bool = True,
    ) -> ArtifactRef:
        expected = {"path", "sha256", "size"}
        if allow_payload_digest:
            expected |= {"quantization", "payload_sha256"}
        require_exact_keys(raw, expected, context)
        name = require_relative_path(raw, "path", context)
        if len(PurePosixPath(name).parts) != 1:
            raise ContractError(f"{context}.path must be a sibling filename")
        if allow_payload_digest:
            require_sha256(raw, "payload_sha256", context)
        return ArtifactRef(
            (PurePosixPath(metadata_ref.path).parent / name).as_posix(),
            require_sha256(raw, "sha256", context),
            require_int(raw, "size", context, minimum=1, maximum=64 * 1024 * 1024),
        )

    def _load_stage(self, stage: str) -> Mapping[str, Any]:
        return require_mapping(
            load_json_artifact(self.repository_root, self._recorded_ref(stage)),
            f"pipeline stage {stage}",
        )

    def _load_training_inputs(
        self, references: Mapping[str, ArtifactRef]
    ) -> tuple[Any, Any, Any, Any]:
        """Parse configs and materialize the exact no-test training dataset."""

        from open_shogi_training.models.config import (
            parse_feature_config_bytes,
            parse_model_config_bytes,
            parse_training_config_bytes,
            validate_config_compatibility,
        )
        from open_shogi_training.models.dataset import load_training_examples

        features = parse_feature_config_bytes(
            load_bytes_artifact(
                self.repository_root, references["featuresConfig"], maximum_bytes=1024 * 1024
            ),
            references["featuresConfig"].path,
        )
        model = parse_model_config_bytes(
            load_bytes_artifact(
                self.repository_root, references["modelConfig"], maximum_bytes=1024 * 1024
            ),
            references["modelConfig"].path,
        )
        training = parse_training_config_bytes(
            load_bytes_artifact(
                self.repository_root,
                references["trainingConfig"],
                maximum_bytes=1024 * 1024,
            ),
            references["trainingConfig"].path,
        )
        validate_config_compatibility(model, training, features)
        if training.expected_teacher_labels != 10_000:
            raise ContractError("Phase 6 training must require exactly 10000 teacher labels")
        loaded = load_training_examples(
            contained_path(self.repository_root, references["teacherLabels"].path),
            contained_path(self.repository_root, references["positions"].path),
            contained_path(self.repository_root, references["datasetManifest"].path),
            training,
            label_manifest_path=contained_path(
                self.repository_root, references["teacherLabelManifest"].path
            ),
            replay_manifest_path=contained_path(
                self.repository_root, references["replayManifest"].path
            ),
            include_replay_test=False,
            repository_root=self.repository_root,
        )
        return features, model, training, loaded

    def _validate_teacher_label_manifest_stage(
        self, table: Mapping[str, Any], manifest_ref: ArtifactRef
    ) -> None:
        """Validate the complete bounded label snapshot without invoking a teacher."""

        expected = {
            "schema",
            "updated_at",
            "config",
            "dataset_manifest",
            "positions",
            "selection",
            "teacher",
            "reported_identity",
            "benchmark",
            "artifacts",
            "progress",
            "binding",
        }
        if "migration" in table:
            expected.add("migration")
        require_exact_keys(table, expected, "teacher label manifest")
        selection = require_mapping(table.get("selection"), "teacher label manifest.selection")
        progress = require_mapping(table.get("progress"), "teacher label manifest.progress")
        artifacts = require_mapping(table.get("artifacts"), "teacher label manifest.artifacts")
        require_exact_keys(
            artifacts,
            {"labels.jsonl", "quarantine.jsonl"},
            "teacher label manifest.artifacts",
        )
        if any(
            value != 10_000
            for value in (
                require_int(selection, "selected", "teacher label manifest.selection"),
                require_int(progress, "selected", "teacher label manifest.progress"),
                require_int(progress, "completed", "teacher label manifest.progress"),
                require_int(progress, "target_completed", "teacher label manifest.progress"),
            )
        ):
            raise ContractError("Phase 6 requires the exact complete 10000-label set")
        if (
            require_int(progress, "quarantined", "teacher label manifest.progress") != 0
            or require_int(progress, "pending", "teacher label manifest.progress") != 0
            or progress.get("status") != "complete"
        ):
            raise ContractError("Phase 6 teacher labels must be complete without quarantine")
        labels_record = require_mapping(
            artifacts.get("labels.jsonl"), "teacher label manifest labels"
        )
        quarantine_record = require_mapping(
            artifacts.get("quarantine.jsonl"), "teacher label manifest quarantine"
        )
        labels_ref = self._manifest_sibling_ref(
            manifest_ref,
            "labels.jsonl",
            labels_record,
            "teacher labels",
        )
        quarantine_ref = self._manifest_sibling_ref(
            manifest_ref,
            "quarantine.jsonl",
            quarantine_record,
            "teacher quarantine",
        )
        quarantine_bytes = load_bytes_artifact(
            self.repository_root, quarantine_ref, maximum_bytes=1
        )
        if quarantine_bytes or quarantine_record.get("records") != 0:
            raise ContractError("Phase 6 teacher quarantine artifact must be empty")
        from open_shogi_training.labeling.schema import validate_label_record

        records = 0
        # Hash and then stream from the same retained descriptor.  The bounded
        # 10k audit must not materialize a potentially hundreds-of-MiB JSONL in
        # memory, and duplicate object keys are rejected before schema parsing.
        with verified_artifact_descriptor(
            self.repository_root,
            labels_ref,
            maximum_bytes=512 * 1024 * 1024,
        ) as descriptor:
            duplicate = os.dup(descriptor)
            with os.fdopen(duplicate, "rb") as stream:
                for records, line in enumerate(stream, start=1):
                    if records > 10_000 or len(line) > 4 * 1024 * 1024 or not line.endswith(b"\n"):
                        raise ContractError("teacher label artifact violates its record bounds")
                    try:
                        validate_label_record(_strict_json_object(line, f"teacher label {records}"))
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                        raise ContractError(f"teacher label record {records} is invalid") from error
        if records != 10_000 or labels_record.get("records") != records:
            raise ContractError("teacher label artifact does not contain exactly 10000 records")

    @staticmethod
    def _manifest_sibling_ref(
        manifest_ref: ArtifactRef,
        name: str,
        record: Mapping[str, Any],
        context: str,
    ) -> ArtifactRef:
        require_exact_keys(record, {"sha256", "size", "records"}, context)
        parent = PurePosixPath(manifest_ref.path).parent
        relative = (parent / name).as_posix()
        return ArtifactRef(
            relative,
            require_sha256(record, "sha256", context),
            require_int(record, "size", context, minimum=0, maximum=4 * 1024**3),
        )

    @staticmethod
    def _require_ref_field(
        table: Mapping[str, Any],
        field: str,
        expected: ArtifactRef,
        context: str,
    ) -> None:
        if ArtifactRef.from_dict(table.get(field), context) != expected:
            raise ContractError(f"{context} reference differs from the recorded stage")

    def _recorded_ref(self, stage: str, *, allow_missing: bool = False) -> ArtifactRef | None:
        state = self._load()
        value = require_mapping(state["stages"], "pipeline stages")[stage]
        if value is None:
            if allow_missing:
                return None
            raise ContractError(f"pipeline stage is missing: {stage}")
        return ArtifactRef.from_dict(value, f"pipeline stages.{stage}")

    def _empty_state(self) -> dict[str, object]:
        return {
            "schema": PIPELINE_STATE_SCHEMA,
            "revision": 1,
            "generationId": self.generation_id,
            "parentGenerationId": self.parent_generation_id,
            "config": self.config.as_dict(),
            "configSha256": self.config_sha256,
            "outputRoot": self.output_root,
            "status": "initialized",
            "stages": dict.fromkeys(STAGES),
        }

    def _load(self) -> Mapping[str, Any]:
        raw = load_json(self.state_path)
        root = require_mapping(raw, "generation pipeline state")
        require_exact_keys(root, _STATE_KEYS, "generation pipeline state")
        if root.get("schema") != PIPELINE_STATE_SCHEMA:
            raise ContractError("unsupported generation pipeline state schema")
        require_int(root, "revision", "pipeline state", minimum=1, maximum=1_000_000_000)
        if require_identifier(root, "generationId", "pipeline state") != self.generation_id:
            raise ContractError("pipeline state generation ID mismatch")
        if (
            require_identifier(root, "parentGenerationId", "pipeline state")
            != self.parent_generation_id
        ):
            raise ContractError("pipeline state parent generation mismatch")
        if ArtifactRef.from_dict(root.get("config"), "pipeline state.config") != self.config:
            raise ContractError("pipeline state config reference mismatch")
        if require_sha256(root, "configSha256", "pipeline state") != self.config_sha256:
            raise ContractError("pipeline state config hash mismatch")
        if require_relative_path(root, "outputRoot", "pipeline state") != self.output_root:
            raise ContractError("pipeline state output root mismatch")
        require_enum(
            root,
            "status",
            "pipeline state",
            {"initialized", "running", "interrupted", "complete"},
        )
        stages = require_mapping(root.get("stages"), "pipeline state.stages")
        require_exact_keys(stages, set(STAGES), "pipeline state.stages")
        saw_missing = False
        for stage in STAGES:
            value = stages[stage]
            if value is None:
                saw_missing = True
            else:
                ArtifactRef.from_dict(value, f"pipeline state.stages.{stage}")
                if saw_missing:
                    raise ContractError("pipeline stages are not a contiguous prefix")
        if root["status"] == "complete" and saw_missing:
            raise ContractError("completed pipeline state has missing stages")
        return root

    def _replace_state_cas(self, state: Mapping[str, Any], *, expected_revision: int) -> None:
        current = self._load()
        if int(current["revision"]) != expected_revision:
            raise ContractError("generation pipeline revision changed concurrently")
        replace_json_state(self.state_path, state)
        published = self._load()
        if int(published["revision"]) != expected_revision + 1:
            raise ContractError("generation pipeline state was replaced during publication")

    @contextmanager
    def _lock(self, *, exclusive: bool):
        try:
            with stable_directory_lock(
                self.state_path.parent,
                create=True,
                exclusive=exclusive,
                nonblocking=True,
            ):
                yield
        except BlockingIOError as error:
            raise ContractError("another process owns the generation pipeline state") from error


def _pipeline_failpoint(name: str) -> None:
    if os.environ.get("OPEN_SHOGI_PIPELINE_FAILPOINT") == name:
        raise RuntimeError(f"generation pipeline failpoint: {name}")


def _strict_json_object(raw: bytes, context: str) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ContractError(f"{context} contains invalid number {value}")
        ),
    )
    return require_mapping(value, context)
