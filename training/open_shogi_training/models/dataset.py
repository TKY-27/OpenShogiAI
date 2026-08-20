"""Strict teacher/Phase 3 join and deterministic stage-balanced sampling."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from open_shogi_training.labeling.artifacts import ArtifactError, stable_regular_descriptor
from open_shogi_training.labeling.schema import (
    MAX_LABEL_LINE_BYTES,
    canonical_state_sha256,
    position_id,
    validate_label_record,
)
from open_shogi_training.models.config import FeatureConfig, TrainingConfig
from open_shogi_training.models.features import (
    extract_features,
    input_dimension,
    parse_canonical_sfen,
    zero_feature_group,
)

if TYPE_CHECKING:
    from open_shogi_training.selfplay.common import ArtifactRef

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_USI_MOVE_RE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")
_MAX_TEACHER_LABELS = 10_000
_LEGACY_MANIFEST_EVIDENCE_NAME = "manifest.v1.evidence.json"
_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1
_DATASET_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "datasetId",
        "source",
        "config",
        "counts",
        "rawObjectSha256",
        "canonicalGameSha256",
        "evidenceSnapshots",
        "artifacts",
    }
)
_DATASET_SOURCE_KEYS = frozenset(
    {
        "sourceId",
        "name",
        "officialBase",
        "adapter",
        "license",
        "licenseEvidence",
        "redistributable",
        "machineLearningAllowed",
        "lastReviewed",
    }
)
_POSITION_KEYS = frozenset(
    {
        "schema",
        "gameId",
        "canonicalSha256",
        "rawSha256",
        "sourceId",
        "split",
        "positionIndex",
        "sfen",
        "moveUsi",
        "nextSfen",
        "outcome",
        "terminalReason",
        "sideToMove",
        "fullPlies",
        "remainingPlies",
        "eligible",
        "terminalTail",
    }
)
_STAGE_INDEX = {"opening": 0, "middlegame": 1, "endgame": 2}
_REPLAY_ROOT_KEYS_V1 = frozenset(
    {
        "schema",
        "generationId",
        "inputCandidates",
        "configSha256",
        "dedupKey",
        "capacity",
        "minimumOlderPositions",
        "counts",
        "splitPolicy",
        "entries",
        "deletionLog",
        "manifestSha256",
    }
)
_REPLAY_ROOT_KEYS = _REPLAY_ROOT_KEYS_V1 | {"config"}
_REPLAY_ENTRY_KEYS = frozenset(
    {
        "entryId",
        "canonicalSfen",
        "sideToMove",
        "outcomeKind",
        "outcomeTarget",
        "stage",
        "priority",
        "hardPosition",
        "split",
        "newestGenerationOrdinal",
        "isNew",
        "origins",
    }
)
_REPLAY_ORIGIN_KEYS = frozenset(
    {
        "positionId",
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
_REPLAY_COUNTS_KEYS = frozenset(
    {
        "input",
        "unique",
        "retained",
        "newRetained",
        "olderRetained",
        "hardRetained",
        "outcomeTargets",
        "splits",
    }
)
_REPLAY_OUTCOME_COUNT_KEYS = frozenset({"loss", "draw", "win"})
_REPLAY_SPLIT_KEYS = frozenset({"train", "validation", "test"})
_REPLAY_SPLIT_POLICY = {
    "train": "training",
    "validation": "selection_metrics_only",
    "test": "final_evaluation_only",
}
_REPLAY_DELETION_KEYS = frozenset({"positionId", "entryId", "reason"})
_REPLAY_DELETION_REASONS = frozenset(
    {
        "deduplicated_conflicting_outcome_target",
        "deduplicated_canonical_sfen",
        "capacity_eviction",
    }
)
_REPLAY_SOURCE_TYPES = frozenset(
    {"audited_game", "teacher", "selfplay", "hard_position", "retained"}
)
_LABEL_MANIFEST_ROOT_KEYS = frozenset(
    {
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
)
_LABEL_MANIFEST_MIGRATION_KEYS = frozenset(
    {"schema", "legacy_manifest", "legacy_schema", "legacy_updated_at"}
)
_LABEL_BINDING_KEYS = frozenset(
    {
        "schema",
        "compatibility",
        "selection_sha256",
        "benchmark_sha256",
        "labels",
        "legality_validator",
        "candidate_coverage",
    }
)
_LICENSE_EVIDENCE_KEYS = frozenset({"url", "local_path", "quote"})
_EVIDENCE_SNAPSHOT_KEYS = frozenset(
    {
        "evidence_id",
        "url",
        "retrieved_at",
        "sha256",
        "size",
        "content_type",
        "object_path",
    }
)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    position_id: str
    sfen: str
    split: str
    stage: str
    position_index: int
    teacher_target: float
    teacher_cp_clipped: float
    teacher_mask: float
    policy_agreement: float
    policy_mask: float
    outcome_target: float
    outcome_mask: float
    teacher_score_kind: str
    teacher_score_value: int | None
    bestmove: str
    recorded_move: str
    candidate_gap_cp: int | None
    already_teacher_labeled: bool
    source_kind: str = "phase4_teacher"
    source_manifest_sha256: str | None = None
    source_generation_id: str | None = None
    source_game_id: str | None = None
    source_ply: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_manifest_sha256: str
    positions_sha256: str
    labels_sha256: str
    label_manifest_sha256: str
    replay_manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class LoadedExamples:
    examples: tuple[TrainingExample, ...]
    identity: DatasetIdentity
    counts_by_split: dict[str, int]
    counts_by_stage: dict[str, int]


class ValueDataset:
    """Tensor-backed examples for one explicit split."""

    def __init__(
        self,
        examples: tuple[TrainingExample, ...],
        feature_config: FeatureConfig,
        *,
        ablated_group: str | None = None,
    ) -> None:
        if not examples:
            raise ValueError("dataset split must contain at least one example")
        feature_rows: list[list[float]] = []
        for example in examples:
            _validate_training_example(example)
            features = extract_features(example.sfen, feature_config)
            if ablated_group is not None:
                features = zero_feature_group(features, feature_config, ablated_group)
            feature_rows.append(features)
        self.features = torch.tensor(feature_rows, dtype=torch.float32)
        self.teacher_targets = torch.tensor(
            [example.teacher_target for example in examples], dtype=torch.float32
        )
        self.teacher_cp = torch.tensor(
            [example.teacher_cp_clipped for example in examples], dtype=torch.float32
        )
        self.teacher_masks = torch.tensor(
            [example.teacher_mask for example in examples], dtype=torch.float32
        )
        self.policy_targets = torch.tensor(
            [example.policy_agreement for example in examples], dtype=torch.float32
        )
        self.policy_masks = torch.tensor(
            [example.policy_mask for example in examples], dtype=torch.float32
        )
        self.outcome_targets = torch.tensor(
            [example.outcome_target for example in examples], dtype=torch.float32
        )
        self.outcome_masks = torch.tensor(
            [example.outcome_mask for example in examples], dtype=torch.float32
        )
        self.stages = tuple(example.stage for example in examples)
        self.position_ids = tuple(example.position_id for example in examples)
        if self.features.shape != (len(examples), input_dimension(feature_config)):
            raise RuntimeError("tensorized feature shape violates the schema")
        tensors = (
            self.features,
            self.teacher_targets,
            self.teacher_cp,
            self.teacher_masks,
            self.policy_targets,
            self.policy_masks,
            self.outcome_targets,
            self.outcome_masks,
        )
        if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise FloatingPointError("dataset tensor contains NaN or infinity")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "teacher_target": self.teacher_targets[index],
            "teacher_cp": self.teacher_cp[index],
            "teacher_mask": self.teacher_masks[index],
            "policy_target": self.policy_targets[index],
            "policy_mask": self.policy_masks[index],
            "outcome_target": self.outcome_targets[index],
            "outcome_mask": self.outcome_masks[index],
        }


class DeterministicStageSampler:
    """Epoch-addressable sampler with exact configured stage proportions."""

    def __init__(self, dataset: ValueDataset, config: TrainingConfig) -> None:
        self._groups = tuple(
            tuple(
                index for index, stage in enumerate(dataset.stages) if _STAGE_INDEX[stage] == group
            )
            for group in range(3)
        )
        self._ratios = config.stage_ratios
        self._count = max(1, round(len(dataset) * config.sample_ratio))
        self._seed = config.seed
        self._epoch = 0
        for index, (group, ratio) in enumerate(zip(self._groups, self._ratios, strict=True)):
            if ratio > 0.0 and not group:
                raise ValueError(
                    f"configured stage ratio {index} is positive but the split is empty"
                )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        seed_material = f"value_v0-sampler\0{self._seed}\0{self._epoch}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        randomizer = random.Random(seed)
        counts = _allocate_counts(self._count, self._ratios)
        selected: list[int] = []
        for group, count in zip(self._groups, counts, strict=True):
            pool = list(group)
            while count > 0:
                randomizer.shuffle(pool)
                take = min(count, len(pool))
                selected.extend(pool[:take])
                count -= take
        randomizer.shuffle(selected)
        if len(selected) != self._count:
            raise RuntimeError("stage sampler violated its requested sample count")
        return iter(selected)


def load_training_examples(
    labels_path: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    training_config: TrainingConfig,
    *,
    label_manifest_path: Path,
    replay_manifest_path: Path | None = None,
    include_replay_test: bool = False,
    repository_root: Path | None = None,
) -> LoadedExamples:
    """Load the complete goal-wide teacher set and strictly join it to Phase 3."""

    return _load_training_examples(
        labels_path,
        positions_path,
        dataset_manifest_path,
        training_config,
        label_manifest_path=label_manifest_path,
        replay_manifest_path=replay_manifest_path,
        expected_teacher_labels=training_config.expected_teacher_labels,
        include_replay_test=include_replay_test,
        repository_root=repository_root,
    )


def _load_training_examples_for_test(
    labels_path: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    training_config: TrainingConfig,
    *,
    label_manifest_path: Path | None = None,
    replay_manifest_path: Path | None = None,
    include_replay_test: bool = False,
) -> LoadedExamples:
    """Internal fixture loader; production commands must require the complete 10k set."""

    return _load_training_examples(
        labels_path,
        positions_path,
        dataset_manifest_path,
        training_config,
        label_manifest_path=label_manifest_path,
        replay_manifest_path=replay_manifest_path,
        expected_teacher_labels=None,
        include_replay_test=include_replay_test,
        repository_root=None,
    )


def _load_training_examples(
    labels_path: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    training_config: TrainingConfig,
    *,
    label_manifest_path: Path | None,
    replay_manifest_path: Path | None,
    expected_teacher_labels: int | None,
    include_replay_test: bool,
    repository_root: Path | None,
) -> LoadedExamples:
    """Strictly join labels to move-bearing Phase 3 rows and bind every artifact hash."""

    manifest, dataset_manifest_sha256, dataset_manifest_size = _load_json_object_with_identity(
        dataset_manifest_path, max_bytes=16 * 1024 * 1024
    )
    if manifest.get("schema") != "phase3_dataset_manifest/v1":
        raise ValueError("dataset manifest must use phase3_dataset_manifest/v1")
    try:
        with stable_regular_descriptor(positions_path) as positions_descriptor:
            positions_initial = os.fstat(positions_descriptor)
            positions_sha256, positions_size = _hash_descriptor(
                positions_descriptor,
                positions_path,
                max_bytes=1024 * 1024 * 1024,
            )
            os.lseek(positions_descriptor, 0, os.SEEK_SET)
            position_rows = _iter_jsonl_gzip_descriptor(
                positions_descriptor,
                positions_path,
                max_uncompressed_bytes=4 * 1024 * 1024 * 1024,
                max_records=250_000,
            )
            loaded_rows = list(position_rows)
            _assert_descriptor_stable(positions_descriptor, positions_path, positions_initial)
    except ArtifactError as error:
        raise ValueError(f"artifact path changed while loading: {positions_path}") from error
    expected_position_records = _verify_positions_artifact(
        manifest, positions_path, positions_sha256, positions_size
    )
    labels, labels_sha256, labels_size = _load_teacher_labels_same_fd(
        labels_path,
        expected_dataset_manifest_sha256=dataset_manifest_sha256,
    )
    if label_manifest_path is None:
        if expected_teacher_labels is not None:
            raise ValueError("production training requires a finalized v2 label manifest")
        label_manifest_sha256 = labels_sha256
    else:
        label_manifest, label_manifest_sha256, _ = _load_json_object_with_identity(
            label_manifest_path, max_bytes=16 * 1024 * 1024
        )
        _validate_label_manifest_v2(
            label_manifest,
            label_manifest_path=label_manifest_path,
            labels_path=labels_path,
            positions_path=positions_path,
            dataset_manifest_path=dataset_manifest_path,
            labels=labels,
            labels_sha256=labels_sha256,
            labels_size=labels_size,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_manifest_size=dataset_manifest_size,
            positions_sha256=positions_sha256,
            positions_size=positions_size,
            expected_teacher_labels=expected_teacher_labels,
            repository_root=repository_root,
        )
    replay_sha256 = None
    replay_examples: tuple[TrainingExample, ...] = ()
    if replay_manifest_path is not None:
        replay_manifest, replay_sha256, replay_size = _load_json_object_with_identity(
            replay_manifest_path, max_bytes=64 * 1024 * 1024
        )
        replay_ref = None
        if expected_teacher_labels is not None:
            from open_shogi_training.selfplay.common import ArtifactRef

            if repository_root is None:
                repository_root = Path(__file__).resolve().parents[3]
            repository_root = repository_root.resolve(strict=True)
            try:
                replay_relative = (
                    replay_manifest_path.absolute().relative_to(repository_root).as_posix()
                )
            except ValueError as error:
                raise ValueError(
                    "replay manifest must be a repository-contained artifact"
                ) from error
            replay_ref = ArtifactRef(replay_relative, replay_sha256, replay_size)
        replay_examples = _load_replay_examples(
            replay_manifest_path,
            manifest=replay_manifest,
            manifest_ref=replay_ref,
            include_test=include_replay_test,
            verify_derivation=expected_teacher_labels is not None,
            repository_root=repository_root,
        )

    if not labels:
        raise ValueError("teacher label artifact is empty")
    if expected_teacher_labels is not None and len(labels) != expected_teacher_labels:
        raise ValueError(
            "teacher label artifact must contain exactly "
            f"{expected_teacher_labels} records; observed {len(labels)}"
        )
    manifest_source_id = (
        validate_production_dataset_manifest(
            manifest,
            expected_position_records,
            repository_root=repository_root,
        )
        if expected_teacher_labels is not None
        else None
    )
    wanted = {label["position_id"]: label for label in labels}
    if len(wanted) != len(labels):
        raise ValueError("teacher labels contain duplicate position identities")
    label_config_sha256 = labels[0]["config_sha256"]
    label_teacher = labels[0]["teacher"]
    wanted_by_state: dict[str, dict[str, Any]] = {}
    for label in labels:
        if label["config_sha256"] != label_config_sha256 or label["teacher"] != label_teacher:
            raise ValueError("teacher labels do not share one configuration and teacher identity")
        state_sha256 = label["canonical_state_sha256"]
        if state_sha256 in wanted_by_state:
            raise ValueError("teacher labels contain duplicate canonical states")
        wanted_by_state[state_sha256] = label
    moves: dict[str, str] = {}
    selected_state_splits = {state_sha256: set() for state_sha256 in wanted_by_state}
    position_records = 0
    game_splits: dict[str, str] = {}
    game_raw_hashes: dict[str, str] = {}
    position_identities: set[tuple[str, int]] = set()
    for row in loaded_rows:
        position_records += 1
        if frozenset(row) != _POSITION_KEYS or row.get("schema") != "phase3_position/v1":
            raise ValueError("Phase 3 position row violates its closed schema")
        _validate_position_row(row)
        game_id = row["gameId"]
        prior_game_split = game_splits.setdefault(game_id, row["split"])
        if prior_game_split != row["split"]:
            raise ValueError(f"Phase 3 game {game_id} leaks across data splits")
        prior_raw_hash = game_raw_hashes.setdefault(game_id, row["rawSha256"])
        if prior_raw_hash != row["rawSha256"]:
            raise ValueError(f"Phase 3 game {game_id} maps to multiple raw objects")
        row_identity = (game_id, row["positionIndex"])
        if row_identity in position_identities:
            raise ValueError("Phase 3 positions repeat a game/position identity")
        position_identities.add(row_identity)
        if manifest_source_id is not None and row["sourceId"] != manifest_source_id:
            raise ValueError("Phase 3 position sourceId disagrees with the dataset manifest")
        state_sha256 = canonical_state_sha256(row["sfen"])
        if state_sha256 in selected_state_splits and row["eligible"] and not row["terminalTail"]:
            selected_state_splits[state_sha256].add(row["split"])
        identity = position_id(row["gameId"], row["positionIndex"])
        label = wanted.get(identity)
        if label is None:
            continue
        if identity in moves:
            raise ValueError(f"Phase 3 positions duplicate selected identity {identity}")
        _validate_join(row, label, training_config)
        move = row["moveUsi"]
        if not isinstance(move, str) or not move:
            raise ValueError(f"selected position has no recorded move: {identity}")
        moves[identity] = move
    if position_records != expected_position_records:
        raise ValueError(
            "positions artifact record count disagrees with the dataset manifest: "
            f"expected {expected_position_records}, observed {position_records}"
        )
    if manifest_source_id is not None:
        if set(game_splits) != set(manifest["canonicalGameSha256"]):
            raise ValueError("Phase 3 positions do not exactly cover manifest canonical games")
        if set(game_raw_hashes.values()) != set(manifest["rawObjectSha256"]):
            raise ValueError("Phase 3 positions do not exactly cover manifest raw objects")
    missing = sorted(wanted.keys() - moves.keys())
    if missing:
        raise ValueError(
            f"Phase 3 positions omit {len(missing)} teacher labels; first={missing[0]}"
        )
    split_priority = {"train": 0, "validation": 1, "test": 2}
    for state_sha256, label in wanted_by_state.items():
        observed_splits = selected_state_splits[state_sha256]
        if not observed_splits:
            raise ValueError(
                f"Phase 3 positions omit the selected canonical state for {label['position_id']}"
            )
        expected_split = max(observed_splits, key=split_priority.__getitem__)
        if label["split"] != expected_split:
            raise ValueError(
                "teacher label violates Phase 3 canonical-state split priority for "
                f"{label['position_id']}: expected {expected_split}"
            )

    teacher_examples = tuple(
        _make_example(
            label,
            moves[label["position_id"]],
            training_config,
            label_manifest_sha256=label_manifest_sha256,
        )
        for label in labels
    )
    examples = teacher_examples + replay_examples
    state_splits: dict[str, str] = {}
    source_game_splits: dict[tuple[str, str], str] = {}
    for example in examples:
        state_sha256 = canonical_state_sha256(example.sfen)
        prior_split = state_splits.setdefault(state_sha256, example.split)
        if prior_split != example.split:
            raise ValueError(
                "combined teacher/replay examples expose one canonical state across splits"
            )
        if example.source_game_id is not None:
            source_key = (example.source_kind, example.source_game_id)
            prior_game_split = source_game_splits.setdefault(source_key, example.split)
            if prior_game_split != example.split:
                raise ValueError("combined examples expose one source game across splits")
    split_counts = {split: 0 for split in ("train", "validation", "test")}
    stage_counts = {stage: 0 for stage in _STAGE_INDEX}
    for example in examples:
        split_counts[example.split] += 1
        stage_counts[example.stage] += 1
    return LoadedExamples(
        examples=examples,
        identity=DatasetIdentity(
            dataset_manifest_sha256=dataset_manifest_sha256,
            positions_sha256=positions_sha256,
            labels_sha256=labels_sha256,
            label_manifest_sha256=label_manifest_sha256,
            replay_manifest_sha256=replay_sha256,
        ),
        counts_by_split=split_counts,
        counts_by_stage=stage_counts,
    )


def split_examples(loaded: LoadedExamples, split: str) -> tuple[TrainingExample, ...]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    examples = tuple(example for example in loaded.examples if example.split == split)
    if not examples:
        raise ValueError(f"no examples are available for split {split}")
    return examples


def _validate_label_manifest_v2(
    manifest: dict[str, Any],
    *,
    label_manifest_path: Path,
    labels_path: Path,
    positions_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    labels: tuple[dict[str, Any], ...],
    labels_sha256: str,
    labels_size: int,
    dataset_manifest_sha256: str,
    dataset_manifest_size: int,
    positions_sha256: str,
    positions_size: int,
    expected_teacher_labels: int | None,
    repository_root: Path | None = None,
) -> None:
    expected_root = set(_LABEL_MANIFEST_ROOT_KEYS)
    if "migration" in manifest:
        expected_root.add("migration")
    if (
        set(manifest) != expected_root
        or manifest.get("schema") != "phase4_teacher_label_manifest/v2"
    ):
        raise ValueError("label manifest must be a closed phase4_teacher_label_manifest/v2")
    _bounded_utc_timestamp(manifest["updated_at"], "label manifest updated_at")
    config = manifest["config"]
    if (
        not isinstance(config, dict)
        or set(config) != {"schema", "sha256"}
        or config.get("schema") != "phase4_teacher_config/v1"
        or not isinstance(config.get("sha256"), str)
        or _SHA256_RE.fullmatch(config["sha256"]) is None
    ):
        raise ValueError("label manifest config identity is invalid")
    dataset = manifest["dataset_manifest"]
    positions = manifest["positions"]
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"sha256", "size"}
        or dataset.get("sha256") != dataset_manifest_sha256
        or not isinstance(dataset.get("size"), int)
        or isinstance(dataset.get("size"), bool)
        or dataset["size"] != dataset_manifest_size
    ):
        raise ValueError("label manifest dataset identity differs from training input")
    if (
        not isinstance(positions, dict)
        or set(positions) != {"sha256", "size", "input_rows"}
        or positions.get("sha256") != positions_sha256
        or any(
            isinstance(positions.get(key), bool)
            or not isinstance(positions.get(key), int)
            or positions[key] <= 0
            for key in ("size", "input_rows")
        )
    ):
        raise ValueError("label manifest positions identity differs from training input")
    if positions["size"] != positions_size:
        raise ValueError("label manifest positions size differs from training input")

    selection = manifest["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "schema",
        "input_rows",
        "eligible_rows",
        "unique_eligible_states",
        "cross_split_duplicates_excluded",
        "same_priority_duplicates_excluded",
        "selected",
        "identity",
        "cross_split_priority",
        "per_split",
        "per_stage",
        "per_game",
        "sha256",
        "canonical_dedup_identity",
    }:
        raise ValueError("label manifest selection violates its closed schema")
    if selection["schema"] != "phase4_teacher_selection/v2":
        raise ValueError("label manifest selection schema is invalid")
    selected = _bounded_int(selection["selected"], "label manifest selected", 1, 10_000)
    if selection["cross_split_priority"] != ["test", "validation", "train"]:
        raise ValueError("label manifest split priority is invalid")
    if selection["identity"] != (
        "ordered selected label-source fields, including exact SFEN, source, and outcome"
    ):
        raise ValueError("label manifest selection identity is invalid")
    if selection["canonical_dedup_identity"] != (
        "canonical SFEN board, side, and hands; move number omitted"
    ):
        raise ValueError("label manifest canonical dedup identity is invalid")
    if (
        not isinstance(selection["sha256"], str)
        or _SHA256_RE.fullmatch(selection["sha256"]) is None
    ):
        raise ValueError("label manifest selection SHA-256 is invalid")
    for key in (
        "input_rows",
        "eligible_rows",
        "unique_eligible_states",
        "cross_split_duplicates_excluded",
        "same_priority_duplicates_excluded",
    ):
        _bounded_int(selection[key], f"label manifest selection.{key}", 0, 250_000)
    for key, names in (
        ("per_split", frozenset({"train", "validation", "test"})),
        ("per_stage", frozenset(_STAGE_INDEX)),
    ):
        counts = _bounded_count_table(selection[key], names, f"label manifest {key}")
        if sum(counts.values()) != selected:
            raise ValueError(f"label manifest {key} counts disagree with selected")
    per_game = selection["per_game"]
    if (
        not isinstance(per_game, dict)
        or not per_game
        or len(per_game) > 10_000
        or any(
            not isinstance(game, str)
            or _SHA256_RE.fullmatch(game) is None
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for game, count in per_game.items()
        )
        or sum(per_game.values()) != selected
    ):
        raise ValueError("label manifest per-game selection is invalid")

    observed_split_counts = {
        split: sum(label["split"] == split for label in labels)
        for split in ("train", "validation", "test")
    }
    observed_stage_counts = {
        stage: sum(label["stage"] == stage for label in labels) for stage in _STAGE_INDEX
    }
    observed_game_counts: dict[str, int] = {}
    for label in labels:
        game_id = label["game_id"]
        observed_game_counts[game_id] = observed_game_counts.get(game_id, 0) + 1
    if (
        selection["input_rows"] != positions["input_rows"]
        or selection["per_split"] != observed_split_counts
        or selection["per_stage"] != observed_stage_counts
        or selection["per_game"] != dict(sorted(observed_game_counts.items()))
    ):
        raise ValueError("label manifest selection summary differs from its label rows")
    selection_identity = {
        "schema": "phase4_teacher_selection/v2",
        "positions": [
            {
                "position_id": label["position_id"],
                "canonical_sfen": label["canonical_sfen"],
                "canonical_state_sha256": label["canonical_state_sha256"],
                "side_to_move": label["side_to_move"],
                "split": label["split"],
                "game_id": label["game_id"],
                "position_index": label["position_index"],
                "stage": label["stage"],
                "source_id": label["source_id"],
                "outcome": label["outcome"],
            }
            for label in labels
        ],
    }
    observed_selection_sha256 = hashlib.sha256(
        json.dumps(
            selection_identity,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if observed_selection_sha256 != selection["sha256"]:
        raise ValueError("label manifest selection SHA-256 differs from its label rows")

    benchmark = manifest["benchmark"]
    benchmark_relative = (
        PurePosixPath(benchmark["path"])
        if isinstance(benchmark, dict) and isinstance(benchmark.get("path"), str)
        else None
    )
    if (
        not isinstance(benchmark, dict)
        or set(benchmark) != {"path", "sha256", "selected_nodes"}
        or benchmark_relative is None
        or benchmark_relative.is_absolute()
        or len(benchmark_relative.parts) != 1
        or benchmark_relative.name in {"", ".", ".."}
        or benchmark_relative.as_posix() != benchmark["path"]
        or "\\" in benchmark["path"]
        or "\x00" in benchmark["path"]
        or _SHA256_RE.fullmatch(str(benchmark["sha256"])) is None
        or _bounded_int(benchmark["selected_nodes"], "benchmark selected_nodes", 1, 10**10) <= 0
    ):
        raise ValueError("label manifest benchmark identity is invalid")
    assert benchmark_relative is not None
    benchmark_path = label_manifest_path.parent.parent / benchmark_relative.name
    benchmark_sha256, _ = hash_file(benchmark_path, max_bytes=16 * 1024 * 1024)
    if benchmark_sha256 != benchmark["sha256"]:
        raise ValueError("label manifest benchmark bytes differ from disk")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"labels.jsonl", "quarantine.jsonl"}:
        raise ValueError("label manifest artifact set is invalid")
    expected_labels = {"sha256": labels_sha256, "size": labels_size, "records": len(labels)}
    if artifacts["labels.jsonl"] != expected_labels:
        raise ValueError("label manifest does not bind the exact label bytes")
    if labels_path.name != "labels.jsonl" or not _same_parent_directory(
        labels_path, label_manifest_path
    ):
        raise ValueError("label manifest and labels must share their immutable output directory")
    quarantine = artifacts["quarantine.jsonl"]
    if not isinstance(quarantine, dict) or set(quarantine) != {"sha256", "size", "records"}:
        raise ValueError("label manifest quarantine identity is invalid")
    quarantine_path = label_manifest_path.parent / "quarantine.jsonl"
    quarantine_sha256, quarantine_size = hash_file(quarantine_path, max_bytes=512 * 1024 * 1024)
    if quarantine != {
        "sha256": quarantine_sha256,
        "size": quarantine_size,
        "records": 0 if quarantine_size == 0 else quarantine.get("records"),
    }:
        raise ValueError("label manifest quarantine bytes differ from disk")
    if quarantine.get("records") != 0 or quarantine_size != 0:
        raise ValueError("final training requires a complete label set without quarantine")

    progress = manifest["progress"]
    if not isinstance(progress, dict) or set(progress) != {
        "selected",
        "completed",
        "quarantined",
        "pending",
        "target_completed",
        "status",
        "completed_position_ids",
        "quarantined_position_ids",
    }:
        raise ValueError("label manifest progress violates its closed schema")
    completed_ids = [label["position_id"] for label in labels]
    if (
        progress.get("selected") != selected
        or progress.get("completed") != len(labels)
        or progress.get("quarantined") != 0
        or progress.get("pending") != 0
        or progress.get("target_completed") != selected
        or progress.get("status") != "complete"
        or progress.get("completed_position_ids") != completed_ids
        or progress.get("quarantined_position_ids") != []
    ):
        raise ValueError("label manifest is not the exact finalized complete label set")
    if expected_teacher_labels is not None and selected != expected_teacher_labels:
        raise ValueError(
            f"label manifest must finalize exactly {expected_teacher_labels} teacher labels"
        )

    binding = manifest["binding"]
    if (
        not isinstance(binding, dict)
        or set(binding) != _LABEL_BINDING_KEYS
        or binding.get("schema") != "phase4_teacher_label_binding/v2"
        or not isinstance(binding.get("compatibility"), str)
        or not binding["compatibility"]
        or binding.get("selection_sha256") != selection["sha256"]
        or binding.get("benchmark_sha256") != benchmark["sha256"]
        or binding.get("labels") != expected_labels
    ):
        raise ValueError("label manifest immutable binding is invalid")
    _validate_legality_validator(
        binding["legality_validator"],
        repository_root=repository_root,
        verify_disk=expected_teacher_labels is not None,
    )
    _validate_candidate_coverage(binding["candidate_coverage"], selected)
    _validate_manifest_teacher(manifest, labels)
    if "migration" in manifest:
        _validate_label_manifest_migration(
            manifest["migration"],
            label_manifest_path,
            manifest=manifest,
            labels=labels,
        )
        if expected_teacher_labels is not None:
            if positions_path is None or dataset_manifest_path is None:
                raise ValueError(
                    "migrated production labels require their exact Phase 3 input paths"
                )
            _audit_migrated_label_semantics(
                manifest,
                labels=labels,
                positions_path=positions_path,
                dataset_manifest_path=dataset_manifest_path,
                benchmark_path=benchmark_path,
                repository_root=repository_root,
            )


def _audit_migrated_label_semantics(
    manifest: dict[str, Any],
    *,
    labels: tuple[dict[str, Any], ...],
    positions_path: Path,
    dataset_manifest_path: Path,
    benchmark_path: Path,
    repository_root: Path | None,
) -> None:
    """Re-run the complete bounded, no-teacher migration audit on every production load."""

    from open_shogi_training.labeling import pipeline as labeling_pipeline
    from open_shogi_training.labeling.benchmark import BenchmarkError, verify_benchmark_report
    from open_shogi_training.labeling.config import TeacherConfigError, load_teacher_config
    from open_shogi_training.labeling.fingerprint import (
        TeacherFingerprintError,
        fingerprint_teacher,
    )
    from open_shogi_training.labeling.legality import LegalityValidationError
    from open_shogi_training.labeling.selection import SelectionError, select_positions
    from open_shogi_training.selfplay.common import ArtifactRef

    root = (
        Path(__file__).resolve().parents[3] if repository_root is None else repository_root
    ).resolve(strict=True)
    requested_sha256 = manifest["config"]["sha256"]
    config_candidates = (
        root / "configs/teacher/apery-v2.0.0.yaml",
        root / "teacher.yaml",
    )
    configs = []
    for candidate in config_candidates:
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        try:
            config = load_teacher_config(candidate)
        except TeacherConfigError as error:
            raise ValueError(f"cannot audit the bound teacher config: {error}") from error
        if config.sha256 == requested_sha256:
            configs.append(config)
    if len(configs) != 1:
        raise ValueError(
            "migrated labels must resolve one approved teacher config with the bound semantic hash"
        )
    config = configs[0]
    try:
        selection = select_positions(
            positions_path,
            dataset_manifest_path,
            config.selection,
        )
        fingerprint = fingerprint_teacher(config, root)
        report, benchmark_digest = verify_benchmark_report(
            benchmark_path,
            config=config,
            selection=selection,
            fingerprint=fingerprint,
        )
    except (SelectionError, TeacherFingerprintError, BenchmarkError) as error:
        raise ValueError(f"migrated label semantic audit failed: {error}") from error
    if selection.summary() != manifest["selection"]:
        raise ValueError("migrated label selection differs from deterministic Phase 3 selection")
    if fingerprint.identity_record() != manifest["teacher"]:
        raise ValueError("migrated label teacher fingerprint differs from current approved files")
    if (
        benchmark_digest.sha256 != manifest["benchmark"]["sha256"]
        or report["reported_identity"] != manifest["reported_identity"]
        or report["selected_nodes"] != manifest["benchmark"]["selected_nodes"]
    ):
        raise ValueError("migrated label benchmark evidence differs from its full recomputation")

    selected_by_id = {position.position_id: position for position in selection.positions}
    if [label["position_id"] for label in labels] != [
        position.position_id for position in selection.positions
    ]:
        raise ValueError("migrated label rows differ from deterministic selection order")
    validator_binding = manifest["binding"]["legality_validator"]
    historical_engine = ArtifactRef(
        path=validator_binding["path"],
        sha256=validator_binding["sha256"],
        size=validator_binding["size"],
    )
    historical_receipt = ArtifactRef.from_dict(
        validator_binding["build_receipt"],
        "migrated label legality validator build receipt",
    )

    def load_historical_validator(_root: Path) -> tuple[ArtifactRef, ArtifactRef]:
        return historical_engine, historical_receipt

    validator = labeling_pipeline.RustLegalityValidator(
        root,
        receipt_loader=load_historical_validator,
    )
    coverage = {}
    try:
        identity = validator.start()
        if identity.as_dict() != manifest["binding"]["legality_validator"]:
            raise ValueError("migrated label legality validator identity differs")
        for label in labels:
            position = selected_by_id[label["position_id"]]
            coverage[label["position_id"]] = validator.validate(
                position.canonical_sfen,
                label["bestmove"],
                [list(candidate["pv"]) for candidate in label["candidates"]],
                configured_multipv=config.multipv,
            )
        validator.assert_binary_unchanged()
    except LegalityValidationError as error:
        raise ValueError(f"migrated label legality replay failed: {error}") from error
    finally:
        validator.close()
    progress = labeling_pipeline._Progress(
        completed={label["position_id"]: label for label in labels},
        quarantined={},
        legality_coverage=coverage,
    )
    if (
        labeling_pipeline._candidate_coverage(progress, config)
        != manifest["binding"]["candidate_coverage"]
    ):
        raise ValueError("migrated label legality coverage differs from deterministic replay")


def _validate_manifest_teacher(
    manifest: dict[str, Any], labels: tuple[dict[str, Any], ...]
) -> None:
    teacher = manifest["teacher"]
    reported = manifest["reported_identity"]
    if not isinstance(teacher, dict) or set(teacher) != {
        "name",
        "version",
        "binary",
        "eval_files",
        "options",
    }:
        raise ValueError("label manifest teacher violates its closed schema")
    if not isinstance(reported, dict) or set(reported) != {"name", "author"}:
        raise ValueError("label manifest reported teacher identity is invalid")
    first = labels[0]["teacher"]
    expected = {
        "name": teacher["name"],
        "version": teacher["version"],
        "reported_name": reported["name"],
        "reported_author": reported["author"],
        "binary_sha256": teacher.get("binary", {}).get("sha256")
        if isinstance(teacher.get("binary"), dict)
        else None,
        "binary_size": teacher.get("binary", {}).get("size")
        if isinstance(teacher.get("binary"), dict)
        else None,
        "eval_files": teacher["eval_files"],
        "options": teacher["options"],
    }
    if first != expected or any(label["teacher"] != first for label in labels):
        raise ValueError("label rows disagree with the manifest teacher identity")
    if any(label["config_sha256"] != manifest["config"]["sha256"] for label in labels):
        raise ValueError("label rows disagree with the manifest configuration")


def _validate_legality_validator(
    raw: Any, *, repository_root: Path | None, verify_disk: bool
) -> None:
    if not isinstance(raw, dict) or set(raw) != {
        "path",
        "sha256",
        "size",
        "build_receipt",
        "reported_name",
        "reported_author",
    }:
        raise ValueError("label manifest legality validator violates its closed schema")
    _validate_artifact_ref(
        {"path": raw["path"], "sha256": raw["sha256"], "size": raw["size"]},
        "label manifest legality validator",
    )
    if not isinstance(raw["reported_name"], str) or not raw["reported_name"]:
        raise ValueError("label manifest legality validator identity is invalid")
    if raw["reported_author"] is not None and not isinstance(raw["reported_author"], str):
        raise ValueError("label manifest legality validator author is invalid")
    if not verify_disk:
        return
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[3]
    repository_root = repository_root.resolve(strict=True)
    path = repository_root.joinpath(*PurePosixPath(raw["path"]).parts)
    digest, size = hash_file(path, max_bytes=512 * 1024 * 1024)
    if (digest, size) != (raw["sha256"], raw["size"]):
        raise ValueError("label manifest legality validator bytes differ from disk")
    try:
        from open_shogi_training.selfplay.common import (
            ArtifactRef,
            ContractError,
            load_json_artifact,
        )
        from open_shogi_training.selfplay.engine_receipt import (
            validate_engine_build_receipt_document,
        )

        receipt_ref = ArtifactRef.from_dict(
            raw["build_receipt"],
            "label manifest legality validator build receipt",
        )
        validate_engine_build_receipt_document(
            load_json_artifact(repository_root, receipt_ref),
            expected_engine=ArtifactRef(
                path=raw["path"],
                sha256=raw["sha256"],
                size=raw["size"],
            ),
        )
    except (ContractError, TypeError, ValueError) as error:
        raise ValueError("label manifest legality validator build receipt is invalid") from error


def _validate_candidate_coverage(raw: Any, selected: int) -> None:
    keys = {
        "requested_multipv",
        "completed_labels",
        "returned_candidate_counts",
        "short_rows",
        "short_rows_with_exact_legal_root_coverage",
        "short_rows_with_additional_legal_roots",
        "additional_legal_root_moves",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("label manifest candidate coverage violates its closed schema")
    requested = _bounded_int(raw["requested_multipv"], "requested MultiPV", 1, 256)
    if raw["completed_labels"] != selected:
        raise ValueError("label manifest candidate coverage count is inconsistent")
    expected_keys = frozenset(str(index) for index in range(1, requested + 1))
    returned = _bounded_count_table(
        raw["returned_candidate_counts"], expected_keys, "returned candidate counts"
    )
    if sum(returned.values()) != selected:
        raise ValueError("label manifest candidate coverage histogram is inconsistent")
    short = _bounded_int(raw["short_rows"], "short candidate rows", 0, selected)
    exact = _bounded_int(
        raw["short_rows_with_exact_legal_root_coverage"], "exact short rows", 0, selected
    )
    additional = _bounded_int(
        raw["short_rows_with_additional_legal_roots"], "short rows with roots", 0, selected
    )
    _bounded_int(raw["additional_legal_root_moves"], "additional root moves", 0, 1_000_000)
    if short != selected - returned[str(requested)] or short != exact + additional:
        raise ValueError("label manifest short-candidate coverage is inconsistent")


def _validate_label_manifest_migration(
    raw: Any,
    manifest_path: Path,
    *,
    manifest: dict[str, Any],
    labels: tuple[dict[str, Any], ...],
) -> None:
    if not isinstance(raw, dict) or set(raw) != _LABEL_MANIFEST_MIGRATION_KEYS:
        raise ValueError("label manifest migration evidence violates its closed schema")
    if raw.get("schema") != "phase4_teacher_label_manifest_migration/v1":
        raise ValueError("label manifest migration schema is invalid")
    if raw.get("legacy_schema") != "phase4_teacher_label_manifest/v1":
        raise ValueError("label manifest migration source schema is invalid")
    _bounded_utc_timestamp(raw.get("legacy_updated_at"), "legacy label manifest updated_at")
    reference = raw["legacy_manifest"]
    _validate_artifact_ref(reference, "legacy label manifest evidence")
    if reference["path"] != _LEGACY_MANIFEST_EVIDENCE_NAME:
        raise ValueError(
            "legacy label manifest migration evidence must use the frozen evidence filename"
        )
    evidence_path = manifest_path.parent / reference["path"]
    legacy, observed_sha256, observed_size = _load_json_object_with_identity(
        evidence_path, max_bytes=16 * 1024 * 1024
    )
    if (observed_sha256, observed_size) != (reference["sha256"], reference["size"]):
        raise ValueError("legacy label manifest migration evidence differs from disk")
    common_keys = {
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
    }
    if set(legacy) != common_keys or legacy.get("schema") != "phase4_teacher_label_manifest/v1":
        raise ValueError("legacy label manifest evidence violates its closed v1 schema")
    legacy_selection = dict(manifest["selection"])
    legacy_selection.pop("schema")
    legacy_selection.pop("canonical_dedup_identity")
    legacy_selection["identity"] = "canonical SFEN board, side, and hands; move number omitted"
    legacy_identity = [
        {
            "position_id": label["position_id"],
            "canonical_state_sha256": label["canonical_state_sha256"],
            "split": label["split"],
            "game_id": label["game_id"],
            "position_index": label["position_index"],
            "stage": label["stage"],
        }
        for label in labels
    ]
    legacy_selection["sha256"] = hashlib.sha256(
        json.dumps(
            legacy_identity,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    expected = {key: manifest[key] for key in common_keys}
    expected["schema"] = "phase4_teacher_label_manifest/v1"
    expected["updated_at"] = raw["legacy_updated_at"]
    expected["selection"] = legacy_selection
    if legacy != expected:
        raise ValueError("legacy label manifest is not the exact semantic origin of v2")


def _bounded_utc_timestamp(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 64:
        raise ValueError(f"{context} is invalid")


def hash_file(path: Path, *, max_bytes: int) -> tuple[str, int]:
    if max_bytes < 0:
        raise ValueError("artifact byte bound must be non-negative")
    digest = hashlib.sha256()
    size = 0
    try:
        with stable_regular_descriptor(path) as descriptor:
            before = os.fstat(descriptor)
            if before.st_size > max_bytes:
                raise ValueError(f"artifact exceeds its {max_bytes}-byte bound: {path}")
            with os.fdopen(descriptor, "rb", closefd=False) as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"artifact exceeds its {max_bytes}-byte bound: {path}")
                after = os.fstat(input_file.fileno())
    except ArtifactError as error:
        raise ValueError(f"artifact must remain a regular non-symlink file: {path}") from error
    if size != before.st_size or _file_identity(after) != _file_identity(before):
        raise ValueError(f"artifact changed while hashing: {path}")
    return digest.hexdigest(), size


def _open_parent_descriptor(path: Path) -> int:
    absolute = path.absolute()
    parts = absolute.parts
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(parts[0], flags)
    try:
        for part in parts[1:-1]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"cannot open artifact parent directory: {path}") from error
    return descriptor


def _same_parent_directory(left: Path, right: Path) -> bool:
    left_descriptor = _open_parent_descriptor(left)
    try:
        right_descriptor = _open_parent_descriptor(right)
        try:
            left_status = os.fstat(left_descriptor)
            right_status = os.fstat(right_descriptor)
            return (left_status.st_dev, left_status.st_ino) == (
                right_status.st_dev,
                right_status.st_ino,
            )
        finally:
            os.close(right_descriptor)
    finally:
        os.close(left_descriptor)


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _assert_descriptor_stable(
    descriptor: int,
    path: Path,
    expected: os.stat_result | None = None,
) -> None:
    status = os.fstat(descriptor)
    if expected is not None and _file_identity(status) != _file_identity(expected):
        raise ValueError(f"artifact changed while loading: {path}")


def _hash_descriptor(descriptor: int, path: Path, *, max_bytes: int) -> tuple[str, int]:
    initial = os.fstat(descriptor)
    if initial.st_size > max_bytes:
        raise ValueError(f"artifact exceeds its {max_bytes}-byte bound: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"artifact exceeds its {max_bytes}-byte bound: {path}")
        digest.update(chunk)
    if total != initial.st_size:
        raise ValueError(f"artifact changed while hashing: {path}")
    _assert_descriptor_stable(descriptor, path, initial)
    return digest.hexdigest(), total


def _load_json_object_with_identity(
    path: Path, *, max_bytes: int
) -> tuple[dict[str, Any], str, int]:
    try:
        with stable_regular_descriptor(path) as descriptor:
            initial = os.fstat(descriptor)
            if not 0 < initial.st_size <= max_bytes:
                raise ValueError(f"JSON artifact type or size is invalid: {path}")
            chunks: list[bytes] = []
            observed = 0
            while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed)):
                observed += len(chunk)
                if observed > max_bytes:
                    raise ValueError(f"JSON artifact exceeds its byte bound: {path}")
                chunks.append(chunk)
            raw = b"".join(chunks)
            if len(raw) != initial.st_size:
                raise ValueError(f"JSON artifact changed while reading: {path}")
            _assert_descriptor_stable(descriptor, path, initial)
    except ArtifactError as error:
        raise ValueError(f"JSON artifact path changed while reading: {path}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain one object: {path}")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _load_teacher_labels_same_fd(
    path: Path,
    *,
    expected_dataset_manifest_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], str, int]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    observed = 0
    try:
        with stable_regular_descriptor(path) as descriptor:
            initial = os.fstat(descriptor)
            if initial.st_size > 4 * 1024 * 1024 * 1024:
                raise ValueError("teacher label artifact exceeds its byte bound")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                line_number = 0
                while line := stream.readline(MAX_LABEL_LINE_BYTES + 1):
                    line_number += 1
                    if line_number > _MAX_TEACHER_LABELS:
                        raise ValueError("teacher label artifact exceeds 10000 records")
                    observed += len(line)
                    if observed > 4 * 1024 * 1024 * 1024:
                        raise ValueError("teacher label artifact exceeds its byte bound")
                    if len(line) > MAX_LABEL_LINE_BYTES or not line.endswith(b"\n"):
                        raise ValueError(
                            f"teacher label line {line_number} is oversized or incomplete"
                        )
                    digest.update(line)
                    try:
                        raw = json.loads(
                            line,
                            object_pairs_hook=_unique_json_object,
                            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                        )
                        label = validate_label_record(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                        raise ValueError(
                            f"invalid teacher label line {line_number}: {error}"
                        ) from error
                    identity = label["position_id"]
                    if identity in seen:
                        raise ValueError(f"duplicate position_id on label line {line_number}")
                    seen.add(identity)
                    if label["dataset_manifest_sha256"] != expected_dataset_manifest_sha256:
                        raise ValueError(f"dataset manifest mismatch on label line {line_number}")
                    records.append(label)
            if observed != initial.st_size:
                raise ValueError("teacher label artifact changed while loading")
            _assert_descriptor_stable(descriptor, path, initial)
    except ArtifactError as error:
        raise ValueError("teacher label artifact path changed while loading") from error
    return tuple(records), digest.hexdigest(), observed


def _iter_jsonl_gzip_descriptor(
    descriptor: int,
    path: Path,
    *,
    max_uncompressed_bytes: int,
    max_records: int,
    max_line_bytes: int = 4 * 1024 * 1024,
):
    observed = 0
    try:
        with (
            os.fdopen(os.dup(descriptor), "rb") as raw_stream,
            gzip.GzipFile(fileobj=raw_stream, mode="rb") as stream,
        ):
            line_number = 0
            while line := stream.readline(max_line_bytes + 1):
                line_number += 1
                if line_number > max_records:
                    raise ValueError(f"JSONL exceeds {max_records} records")
                if len(line) > max_line_bytes or not line.endswith(b"\n"):
                    raise ValueError(f"JSONL line {line_number} is oversized or incomplete")
                observed += len(line)
                if observed > max_uncompressed_bytes:
                    raise ValueError(f"uncompressed JSONL exceeds {max_uncompressed_bytes} bytes")
                try:
                    value = json.loads(line, object_pairs_hook=_unique_json_object)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"invalid JSON on line {line_number}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object")
                yield value
    except (gzip.BadGzipFile, EOFError) as error:
        raise ValueError(f"invalid gzip JSONL artifact: {path}") from error


def _make_example(
    label: dict[str, Any],
    recorded_move: str,
    config: TrainingConfig,
    *,
    label_manifest_sha256: str,
) -> TrainingExample:
    score = label["score"]
    if score["kind"] == "cp":
        teacher_cp = max(-config.teacher_clip_cp, min(config.teacher_clip_cp, score["value"]))
    else:
        if score["value"] == 0:
            raise ValueError("a mate teacher score must have a non-zero signed distance")
        teacher_cp = config.teacher_clip_cp if score["value"] > 0 else -config.teacher_clip_cp
    teacher_target = teacher_cp / config.teacher_normalization_cp
    outcome_target, outcome_mask = _outcome_target(label["outcome"], label["side_to_move"])
    values = (teacher_cp, teacher_target, outcome_target, outcome_mask)
    if any(not math.isfinite(value) for value in values):
        raise FloatingPointError("derived training target is non-finite")
    return TrainingExample(
        position_id=label["position_id"],
        sfen=label["canonical_sfen"],
        split=label["split"],
        stage=label["stage"],
        position_index=label["position_index"],
        teacher_target=teacher_target,
        teacher_cp_clipped=teacher_cp,
        teacher_mask=1.0,
        policy_agreement=1.0 if recorded_move == label["bestmove"] else 0.0,
        policy_mask=1.0,
        outcome_target=outcome_target,
        outcome_mask=outcome_mask,
        teacher_score_kind=score["kind"],
        teacher_score_value=score["value"],
        bestmove=label["bestmove"],
        recorded_move=recorded_move,
        candidate_gap_cp=_candidate_gap_cp(label["candidates"]),
        already_teacher_labeled=True,
        source_kind="phase4_teacher",
        source_manifest_sha256=label_manifest_sha256,
        source_game_id=label["game_id"],
        source_ply=label["position_index"],
    )


def _outcome_target(outcome: str, side_to_move: str) -> tuple[float, float]:
    if outcome == "unknown":
        return 0.0, 0.0
    if outcome == "draw":
        return 0.0, 1.0
    winner = "black" if outcome == "black_win" else "white"
    return (1.0 if winner == side_to_move else -1.0), 1.0


def _candidate_gap_cp(candidates: list[dict[str, Any]]) -> int | None:
    if len(candidates) < 2:
        return None
    first = candidates[0]["score"]
    second = candidates[1]["score"]
    if first["kind"] != "cp" or second["kind"] != "cp":
        return None
    # The downstream ambiguity contract is a non-negative magnitude, not a
    # signed ordering assertion: shallow independent MultiPV searches can
    # occasionally score the nominal second line above the first.
    gap = abs(first["value"] - second["value"])
    if gap > 1_000_000:
        raise ValueError("teacher candidate gap exceeds the downstream centipawn bound")
    return gap


def _load_replay_examples(
    path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    manifest_ref: ArtifactRef | None = None,
    include_test: bool = False,
    verify_derivation: bool = False,
    repository_root: Path | None = None,
) -> tuple[TrainingExample, ...]:
    if manifest is None:
        manifest, _, _ = _load_json_object_with_identity(path, max_bytes=64 * 1024 * 1024)
    schema = manifest.get("schema")
    expected_keys = (
        _REPLAY_ROOT_KEYS if schema == "phase6_replay_buffer_manifest/v2" else _REPLAY_ROOT_KEYS_V1
    )
    if frozenset(manifest) != expected_keys:
        raise ValueError("replay manifest violates its closed root schema")
    if schema not in {
        "phase6_replay_buffer_manifest/v1",
        "phase6_replay_buffer_manifest/v2",
    }:
        raise ValueError("unsupported replay manifest schema")
    if verify_derivation:
        if schema != "phase6_replay_buffer_manifest/v2":
            raise ValueError("training requires a deterministically bound v2 replay manifest")
        if manifest_ref is None:
            raise ValueError("replay derivation requires its same-read artifact identity")
        _verify_replay_derivation(manifest, manifest_ref, repository_root=repository_root)
    recorded_hash = manifest["manifestSha256"]
    if not isinstance(recorded_hash, str) or _SHA256_RE.fullmatch(recorded_hash) is None:
        raise ValueError("replay manifestSha256 is invalid")
    without_hash = dict(manifest)
    without_hash.pop("manifestSha256")
    computed_hash = hashlib.sha256(
        json.dumps(
            without_hash,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if recorded_hash != computed_hash:
        raise ValueError("replay manifestSha256 does not match its content")
    _validate_replay_manifest_header(manifest)
    entries = manifest["entries"]
    if not isinstance(entries, list) or len(entries) > 1_000_000:
        raise ValueError("replay entries must be a bounded array")
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or frozenset(raw) != _REPLAY_ENTRY_KEYS:
            raise ValueError(f"replay entry {index} violates its closed schema")
        entry_id = raw["entryId"]
        sfen = raw["canonicalSfen"]
        if not isinstance(entry_id, str) or _SHA256_RE.fullmatch(entry_id) is None:
            raise ValueError(f"replay entry {index} has an invalid entryId")
        if not isinstance(sfen, str):
            raise ValueError(f"replay entry {index} canonicalSfen is invalid")
        if entry_id in seen or entry_id != hashlib.sha256(sfen.encode("utf-8")).hexdigest():
            raise ValueError(f"replay entry {index} identity is duplicate or inconsistent")
        seen.add(entry_id)
        parsed = parse_canonical_sfen(sfen)
        if parsed.move_number != 1:
            raise ValueError(f"replay entry {index} canonicalSfen move number must be one")
        side = raw["sideToMove"]
        expected_side = "black" if parsed.side_to_move == 0 else "white"
        if side != expected_side:
            raise ValueError(f"replay entry {index} side disagrees with SFEN")
        if raw["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"replay entry {index} split is invalid")
        if raw["stage"] not in _STAGE_INDEX:
            raise ValueError(f"replay entry {index} stage is invalid")
        if raw["outcomeKind"] not in {"black_win", "white_win", "draw"}:
            raise ValueError(f"replay entry {index} outcomeKind is invalid")
        _validate_replay_outcome(raw, f"replay entry {index}")
        _bounded_int(raw["priority"], f"replay entry {index} priority", 0, 1_000_000)
        _bounded_int(
            raw["newestGenerationOrdinal"],
            f"replay entry {index} newestGenerationOrdinal",
            0,
            1_000_000,
        )
        if not isinstance(raw["hardPosition"], bool) or not isinstance(raw["isNew"], bool):
            raise ValueError(f"replay entry {index} flags are invalid")
        origins = raw["origins"]
        if not isinstance(origins, list) or not 1 <= len(origins) <= 1_000_000:
            raise ValueError(f"replay entry {index} origins are invalid")
        _validate_replay_origins(raw, origins, index)
        if raw["split"] == "test" and not include_test:
            continue
        examples.append(
            TrainingExample(
                position_id=entry_id,
                sfen=sfen,
                split=raw["split"],
                stage=raw["stage"],
                position_index=0,
                teacher_target=0.0,
                teacher_cp_clipped=0.0,
                teacher_mask=0.0,
                policy_agreement=0.0,
                policy_mask=0.0,
                outcome_target=float(raw["outcomeTarget"]),
                outcome_mask=1.0,
                teacher_score_kind="unlabeled",
                teacher_score_value=None,
                bestmove="unlabeled",
                recorded_move="unlabeled",
                candidate_gap_cp=None,
                already_teacher_labeled=any(
                    origin["sourceType"] == "teacher" for origin in origins
                ),
                source_kind="phase6_replay",
                source_manifest_sha256=origins[0]["sourceManifest"]["sha256"],
                source_generation_id=origins[0]["generationId"],
                source_game_id=origins[0]["sourceGameId"],
                source_ply=origins[0]["sourcePly"],
            )
        )
    _validate_replay_summary(manifest, entries)
    return tuple(examples)


def _verify_replay_derivation(
    manifest: dict[str, Any],
    manifest_ref: ArtifactRef,
    *,
    repository_root: Path | None,
) -> None:
    from open_shogi_training.selfplay.common import (
        ArtifactRef,
        load_bytes_artifact,
        load_json_artifact,
    )
    from open_shogi_training.selfplay.config import parse_selfplay_config_bytes
    from open_shogi_training.selfplay.evidence import build_replay_buffer_manifest

    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[3]
    repository_root = repository_root.resolve(strict=True)
    if not isinstance(manifest_ref, ArtifactRef):
        raise ValueError("replay manifest identity must be an ArtifactRef")
    config_ref = ArtifactRef.from_dict(manifest["config"], "replay config")
    candidates_ref = ArtifactRef.from_dict(manifest["inputCandidates"], "replay candidates")
    config = parse_selfplay_config_bytes(
        load_bytes_artifact(repository_root, config_ref, maximum_bytes=64 * 1024),
        config_ref.path,
    )
    if config.sha256 != manifest["configSha256"]:
        raise ValueError("replay config hash differs from the referenced configuration")
    candidates = load_json_artifact(
        repository_root,
        candidates_ref,
        maximum_bytes=64 * 1024 * 1024,
        maximum_nodes=8_000_000,
    )
    expected = build_replay_buffer_manifest(
        candidates,
        candidates_ref=candidates_ref,
        config=config,
        config_ref=config_ref,
        repository_root=repository_root,
    )
    if manifest != expected:
        raise ValueError("replay manifest is not the deterministic evidence derivation")
    if (
        manifest_ref.sha256
        != hashlib.sha256(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        ).hexdigest()
    ):
        raise ValueError("replay artifact bytes are not canonical JSON")


def _validate_replay_manifest_header(manifest: dict[str, Any]) -> None:
    _safe_identifier(manifest["generationId"], "replay generationId")
    _validate_artifact_ref(manifest["inputCandidates"], "replay inputCandidates")
    if manifest["schema"] == "phase6_replay_buffer_manifest/v2":
        _validate_artifact_ref(manifest["config"], "replay config")
    if (
        not isinstance(manifest["configSha256"], str)
        or _SHA256_RE.fullmatch(manifest["configSha256"]) is None
    ):
        raise ValueError("replay configSha256 is invalid")
    if manifest["dedupKey"] != "canonical_sfen":
        raise ValueError("replay dedupKey must be canonical_sfen")
    capacity = _bounded_int(manifest["capacity"], "replay capacity", 1, 10_000_000)
    minimum_older = _bounded_int(
        manifest["minimumOlderPositions"],
        "replay minimumOlderPositions",
        0,
        10_000_000,
    )
    if minimum_older > capacity:
        raise ValueError("replay minimumOlderPositions exceeds capacity")
    if manifest["splitPolicy"] != _REPLAY_SPLIT_POLICY:
        raise ValueError("replay splitPolicy is invalid")
    deletions = manifest["deletionLog"]
    if not isinstance(deletions, list) or len(deletions) > 1_000_000:
        raise ValueError("replay deletionLog must be a bounded array")
    for index, deletion in enumerate(deletions):
        context = f"replay deletionLog[{index}]"
        if not isinstance(deletion, dict) or frozenset(deletion) != _REPLAY_DELETION_KEYS:
            raise ValueError(f"{context} violates its closed schema")
        position_id_value = deletion["positionId"]
        if position_id_value is not None:
            _safe_identifier(position_id_value, f"{context} positionId")
        if (
            not isinstance(deletion["entryId"], str)
            or _SHA256_RE.fullmatch(deletion["entryId"]) is None
        ):
            raise ValueError(f"{context} entryId is invalid")
        if deletion["reason"] not in _REPLAY_DELETION_REASONS:
            raise ValueError(f"{context} reason is invalid")
        if (deletion["reason"] == "capacity_eviction") != (position_id_value is None):
            raise ValueError(f"{context} positionId disagrees with its reason")
    expected_deletion_order = sorted(
        deletions,
        key=lambda item: (item["reason"], item["entryId"], item["positionId"] or ""),
    )
    if deletions != expected_deletion_order:
        raise ValueError("replay deletionLog is not in canonical order")


def _validate_replay_origins(entry: dict[str, Any], origins: list[Any], entry_index: int) -> None:
    validated: list[dict[str, Any]] = []
    position_ids: set[str] = set()
    for origin_index, origin in enumerate(origins):
        context = f"replay entry {entry_index} origin {origin_index}"
        if not isinstance(origin, dict) or frozenset(origin) != _REPLAY_ORIGIN_KEYS:
            raise ValueError(f"{context} violates its closed schema")
        position_id_value = _safe_identifier(origin["positionId"], f"{context} positionId")
        if position_id_value in position_ids:
            raise ValueError(f"replay entry {entry_index} repeats an origin positionId")
        position_ids.add(position_id_value)
        _safe_identifier(origin["generationId"], f"{context} generationId")
        _bounded_int(origin["generationOrdinal"], f"{context} generationOrdinal", 0, 1_000_000)
        if origin["sourceType"] not in _REPLAY_SOURCE_TYPES:
            raise ValueError(f"{context} sourceType is invalid")
        _validate_artifact_ref(origin["sourceManifest"], f"{context} sourceManifest")
        _safe_identifier(origin["sourceGameId"], f"{context} sourceGameId")
        _bounded_int(origin["sourcePly"], f"{context} sourcePly", 0, 10_000)
        if origin["sideToMove"] not in {"black", "white"}:
            raise ValueError(f"{context} sideToMove is invalid")
        if origin["outcomeKind"] not in {"black_win", "white_win", "draw"}:
            raise ValueError(f"{context} outcomeKind is invalid")
        _validate_replay_outcome(origin, context)
        if origin["stage"] not in _STAGE_INDEX:
            raise ValueError(f"{context} stage is invalid")
        _bounded_int(origin["priority"], f"{context} priority", 0, 1_000_000)
        if not isinstance(origin["hardPosition"], bool) or not isinstance(origin["isNew"], bool):
            raise ValueError(f"{context} flags are invalid")
        validated.append(origin)

    expected_order = sorted(
        validated,
        key=lambda origin: (
            -origin["priority"],
            -int(origin["hardPosition"]),
            -int(origin["isNew"]),
            -origin["generationOrdinal"],
            origin["positionId"],
        ),
    )
    if validated != expected_order:
        raise ValueError(f"replay entry {entry_index} origins are not in canonical order")
    representative = validated[0]
    for entry_key, origin_key in (
        ("sideToMove", "sideToMove"),
        ("outcomeKind", "outcomeKind"),
        ("outcomeTarget", "outcomeTarget"),
        ("stage", "stage"),
    ):
        if entry[entry_key] != representative[origin_key]:
            raise ValueError(
                f"replay entry {entry_index} {entry_key} disagrees with its representative origin"
            )
    aggregates = {
        "priority": max(origin["priority"] for origin in validated),
        "hardPosition": any(origin["hardPosition"] for origin in validated),
        "newestGenerationOrdinal": max(origin["generationOrdinal"] for origin in validated),
        "isNew": any(origin["isNew"] for origin in validated),
    }
    for key, expected in aggregates.items():
        if entry[key] != expected:
            raise ValueError(f"replay entry {entry_index} {key} aggregate is inconsistent")


def _validate_replay_summary(manifest: dict[str, Any], entries: list[Any]) -> None:
    counts = manifest["counts"]
    if not isinstance(counts, dict) or frozenset(counts) != _REPLAY_COUNTS_KEYS:
        raise ValueError("replay counts violate their closed schema")
    scalar_names = ("input", "unique", "retained", "newRetained", "olderRetained", "hardRetained")
    scalar_counts = {
        name: _bounded_int(counts[name], f"replay counts.{name}", 0, 1_000_000)
        for name in scalar_names
    }
    outcome_counts = _bounded_count_table(
        counts["outcomeTargets"], _REPLAY_OUTCOME_COUNT_KEYS, "replay outcomeTargets"
    )
    split_counts = _bounded_count_table(counts["splits"], _REPLAY_SPLIT_KEYS, "replay splits")
    observed_splits = {
        split: sum(entry["split"] == split for entry in entries) for split in _REPLAY_SPLIT_KEYS
    }
    observed_outcomes = {
        "loss": sum(entry["outcomeTarget"] == -1 for entry in entries),
        "draw": sum(entry["outcomeTarget"] == 0 for entry in entries),
        "win": sum(entry["outcomeTarget"] == 1 for entry in entries),
    }
    expected_scalars = {
        "retained": len(entries),
        "newRetained": sum(entry["isNew"] for entry in entries),
        "olderRetained": sum(not entry["isNew"] for entry in entries),
        "hardRetained": sum(entry["hardPosition"] for entry in entries),
    }
    for name, expected in expected_scalars.items():
        if scalar_counts[name] != expected:
            raise ValueError(f"replay counts.{name} disagrees with entries")
    if outcome_counts != observed_outcomes or split_counts != observed_splits:
        raise ValueError("replay outcome/split counts disagree with entries")
    if not scalar_counts["input"] >= scalar_counts["unique"] >= scalar_counts["retained"]:
        raise ValueError("replay input/unique/retained counts are inconsistent")
    observed_unique = len(
        {entry["entryId"] for entry in entries}
        | {deletion["entryId"] for deletion in manifest["deletionLog"]}
    )
    if scalar_counts["unique"] != observed_unique:
        raise ValueError("replay counts.unique disagrees with retained and deleted entries")
    if len(manifest["deletionLog"]) != scalar_counts["input"] - scalar_counts["retained"]:
        raise ValueError("replay deletionLog count disagrees with input and retained counts")
    if scalar_counts["retained"] > manifest["capacity"]:
        raise ValueError("replay retained count exceeds capacity")
    expected_order = sorted(
        entries,
        key=lambda entry: (entry["split"], -entry["priority"], entry["entryId"]),
    )
    if entries != expected_order:
        raise ValueError("replay entries are not in canonical order")


def _validate_replay_outcome(table: dict[str, Any], context: str) -> None:
    target = table["outcomeTarget"]
    if isinstance(target, bool) or not isinstance(target, int) or target not in {-1, 0, 1}:
        raise ValueError(f"{context} outcomeTarget is invalid")
    outcome = table["outcomeKind"]
    side = table["sideToMove"]
    if outcome == "draw":
        expected = 0
    else:
        winner = "black" if outcome == "black_win" else "white"
        expected = 1 if side == winner else -1
    if target != expected:
        raise ValueError(f"{context} outcomeTarget disagrees with outcome and side")


def _validate_training_example(example: TrainingExample) -> None:
    if (
        not isinstance(example.position_id, str)
        or _SHA256_RE.fullmatch(example.position_id) is None
    ):
        raise ValueError("training example position_id is invalid")
    if (
        not isinstance(example.split, str)
        or example.split not in {"train", "validation", "test"}
        or not isinstance(example.stage, str)
        or example.stage not in _STAGE_INDEX
    ):
        raise ValueError(f"training example {example.position_id} has an invalid split or stage")
    if (
        isinstance(example.position_index, bool)
        or not isinstance(example.position_index, int)
        or not 0 <= example.position_index <= 10_000
    ):
        raise ValueError(f"training example {example.position_id} has an invalid position index")
    masks = (example.teacher_mask, example.policy_mask, example.outcome_mask)
    if any(mask not in {0.0, 1.0} for mask in masks):
        raise ValueError(f"training example {example.position_id} has a non-binary mask")
    numeric = (
        example.teacher_target,
        example.teacher_cp_clipped,
        example.policy_agreement,
        example.outcome_target,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise FloatingPointError(f"training example {example.position_id} has a non-finite target")
    if example.policy_mask == 1.0 and example.policy_agreement not in {0.0, 1.0}:
        raise ValueError(f"training example {example.position_id} has an invalid policy target")
    if example.policy_mask == 0.0 and example.policy_agreement != 0.0:
        raise ValueError(f"training example {example.position_id} has a masked policy target")
    if example.outcome_mask == 1.0 and example.outcome_target not in {-1.0, 0.0, 1.0}:
        raise ValueError(f"training example {example.position_id} has an invalid outcome target")
    if example.outcome_mask == 0.0 and example.outcome_target != 0.0:
        raise ValueError(f"training example {example.position_id} has a masked outcome target")
    candidate_gap = example.candidate_gap_cp
    if candidate_gap is not None and (
        isinstance(candidate_gap, bool)
        or not isinstance(candidate_gap, int)
        or not 0 <= candidate_gap <= 1_000_000
    ):
        raise ValueError(f"training example {example.position_id} has an invalid candidate gap")
    if not isinstance(example.already_teacher_labeled, bool):
        raise ValueError(
            f"training example {example.position_id} has an invalid teacher provenance flag"
        )
    if example.source_kind == "phase4_teacher":
        if (
            not example.already_teacher_labeled
            or example.teacher_mask != 1.0
            or example.policy_mask != 1.0
            or example.teacher_score_kind not in {"cp", "mate"}
            or isinstance(example.teacher_score_value, bool)
            or not isinstance(example.teacher_score_value, int)
            or not _I32_MIN <= example.teacher_score_value <= _I32_MAX
            or (example.teacher_score_kind == "mate" and example.teacher_score_value == 0)
            or not isinstance(example.bestmove, str)
            or _USI_MOVE_RE.fullmatch(example.bestmove) is None
            or not isinstance(example.recorded_move, str)
            or _USI_MOVE_RE.fullmatch(example.recorded_move) is None
            or (example.policy_agreement == 1.0) != (example.bestmove == example.recorded_move)
            or not isinstance(example.source_game_id, str)
            or _SHA256_RE.fullmatch(example.source_game_id) is None
            or example.source_generation_id is not None
            or example.source_ply != example.position_index
        ):
            raise ValueError(f"teacher example {example.position_id} has inconsistent supervision")
    elif example.source_kind == "phase6_replay":
        if (
            example.teacher_target != 0.0
            or example.teacher_cp_clipped != 0.0
            or example.teacher_mask != 0.0
            or example.policy_agreement != 0.0
            or example.policy_mask != 0.0
            or example.outcome_mask != 1.0
            or example.teacher_score_kind != "unlabeled"
            or example.teacher_score_value is not None
            or example.bestmove != "unlabeled"
            or example.recorded_move != "unlabeled"
            or example.candidate_gap_cp is not None
            or example.position_index != 0
            or not isinstance(example.source_generation_id, str)
            or _IDENTIFIER_RE.fullmatch(example.source_generation_id) is None
            or not isinstance(example.source_game_id, str)
            or _IDENTIFIER_RE.fullmatch(example.source_game_id) is None
            or isinstance(example.source_ply, bool)
            or not isinstance(example.source_ply, int)
            or not 0 <= example.source_ply <= 10_000
        ):
            raise ValueError(f"replay example {example.position_id} is not outcome-only")
    else:
        raise ValueError(f"training example {example.position_id} has an unknown source kind")
    if (
        not isinstance(example.source_manifest_sha256, str)
        or _SHA256_RE.fullmatch(example.source_manifest_sha256) is None
    ):
        raise ValueError(f"training example {example.position_id} has invalid source provenance")


def _validate_artifact_ref(raw: Any, context: str) -> None:
    if not isinstance(raw, dict) or frozenset(raw) != {"path", "sha256", "size"}:
        raise ValueError(f"{context} violates its closed schema")
    path = raw["path"]
    if not isinstance(path, str) or not path or len(path) > 1_024 or "\\" in path or "\x00" in path:
        raise ValueError(f"{context} path is invalid")
    portable = PurePosixPath(path)
    if (
        portable.is_absolute()
        or portable.as_posix() != path
        or any(part in {"", ".", ".."} for part in portable.parts)
    ):
        raise ValueError(f"{context} path is not normalized and relative")
    if not isinstance(raw["sha256"], str) or _SHA256_RE.fullmatch(raw["sha256"]) is None:
        raise ValueError(f"{context} sha256 is invalid")
    _bounded_int(raw["size"], f"{context} size", 0, 4 * 1024 * 1024 * 1024)


def _bounded_count_table(raw: Any, keys: frozenset[str], context: str) -> dict[str, int]:
    if not isinstance(raw, dict) or frozenset(raw) != keys:
        raise ValueError(f"{context} violates its closed schema")
    return {key: _bounded_int(raw[key], f"{context}.{key}", 0, 1_000_000) for key in keys}


def _bounded_int(value: Any, context: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def _safe_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a safe identifier")
    return value


def _validate_join(row: dict[str, Any], label: dict[str, Any], config: TrainingConfig) -> None:
    comparisons = {
        "gameId": "game_id",
        "positionIndex": "position_index",
        "sfen": "canonical_sfen",
        "split": "split",
        "sourceId": "source_id",
        "outcome": "outcome",
    }
    for position_key, label_key in comparisons.items():
        if row[position_key] != label[label_key]:
            raise ValueError(
                f"teacher/Phase 3 join mismatch for {label['position_id']}: {position_key}"
            )
    if row["sideToMove"] != label["side_to_move"]:
        raise ValueError(f"teacher/Phase 3 side mismatch for {label['position_id']}")
    if not row["eligible"] or row["terminalTail"]:
        raise ValueError(
            f"teacher label selected an ineligible Phase 3 row: {label['position_id']}"
        )
    expected_stage = _stage_from_phase3_position(
        row["positionIndex"],
        row["fullPlies"],
        config.stage_boundaries_basis_points,
    )
    if label["stage"] != expected_stage:
        raise ValueError(
            f"teacher/Phase 3 stage mismatch for {label['position_id']}: expected {expected_stage}"
        )


def _validate_position_row(row: dict[str, Any]) -> None:
    for key in ("gameId", "canonicalSha256", "rawSha256"):
        if not isinstance(row[key], str) or _SHA256_RE.fullmatch(row[key]) is None:
            raise ValueError(f"Phase 3 position {key} is invalid")
    if row["canonicalSha256"] != row["gameId"]:
        raise ValueError("Phase 3 position canonicalSha256 must equal gameId")
    if not isinstance(row["sourceId"], str) or not row["sourceId"]:
        raise ValueError("Phase 3 position sourceId is invalid")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError("Phase 3 position split is invalid")
    index = _bounded_int(row["positionIndex"], "Phase 3 positionIndex", 0, 10_000)
    full_plies = _bounded_int(row["fullPlies"], "Phase 3 fullPlies", 0, 10_000)
    remaining = _bounded_int(row["remainingPlies"], "Phase 3 remainingPlies", 0, 10_000)
    if index > full_plies or remaining != full_plies - index:
        raise ValueError("Phase 3 position ply fields are inconsistent")
    parsed = parse_canonical_sfen(row["sfen"])
    expected_side = "black" if parsed.side_to_move == 0 else "white"
    if row["sideToMove"] != expected_side:
        raise ValueError("Phase 3 position sideToMove disagrees with SFEN")
    if row["outcome"] not in {"black_win", "white_win", "draw", "unknown"}:
        raise ValueError("Phase 3 position outcome is invalid")
    if not isinstance(row["terminalReason"], str) or not row["terminalReason"]:
        raise ValueError("Phase 3 position terminalReason is invalid")
    if not isinstance(row["eligible"], bool) or not isinstance(row["terminalTail"], bool):
        raise ValueError("Phase 3 position eligibility flags are invalid")
    if row["eligible"] and row["terminalTail"]:
        raise ValueError("Phase 3 position cannot be eligible and terminal-tail")
    move = row["moveUsi"]
    next_sfen = row["nextSfen"]
    if (move is None) != (next_sfen is None):
        raise ValueError("Phase 3 position moveUsi and nextSfen presence disagree")
    if move is not None and (not isinstance(move, str) or _USI_MOVE_RE.fullmatch(move) is None):
        raise ValueError("Phase 3 position moveUsi is invalid")
    if next_sfen is not None and not isinstance(next_sfen, str):
        raise ValueError("Phase 3 position nextSfen is invalid")
    if next_sfen is not None:
        parse_canonical_sfen(next_sfen)
    if row["eligible"] and (move is None or next_sfen is None):
        raise ValueError("eligible Phase 3 position requires a move and next SFEN")


def _stage_from_phase3_position(
    position_index: int,
    full_plies: int,
    boundaries_basis_points: tuple[int, int],
) -> str:
    if full_plies <= 0 or not 0 <= position_index < full_plies:
        raise ValueError("eligible Phase 3 position has invalid ply progress")
    progress = position_index * 10_000 // full_plies
    opening_end, middlegame_end = boundaries_basis_points
    if progress < opening_end:
        return "opening"
    if progress < middlegame_end:
        return "middlegame"
    return "endgame"


def _verify_positions_artifact(manifest: dict[str, Any], path: Path, sha256: str, size: int) -> int:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("dataset manifest artifacts must be an object")
    recorded = artifacts.get(path.name)
    if not isinstance(recorded, dict) or frozenset(recorded) != {"sha256", "size", "records"}:
        raise ValueError("positions filename is not recorded by the dataset manifest")
    if not isinstance(recorded["sha256"], str) or _SHA256_RE.fullmatch(recorded["sha256"]) is None:
        raise ValueError("positions artifact identity in the dataset manifest is invalid")
    recorded_size = _bounded_int(
        recorded["size"], "dataset manifest positions size", 0, 1024 * 1024 * 1024
    )
    recorded_records = _bounded_int(
        recorded["records"], "dataset manifest positions records", 1, 100_000_000
    )
    if recorded["sha256"] != sha256 or recorded_size != size:
        raise ValueError("positions artifact hash/size disagrees with the dataset manifest")
    return recorded_records


def validate_production_dataset_manifest(
    manifest: dict[str, Any],
    expected_position_records: int,
    *,
    repository_root: Path | None = None,
) -> str:
    """Validate the canonical closed Phase 3 production manifest contract."""

    if frozenset(manifest) != _DATASET_MANIFEST_KEYS:
        raise ValueError("production dataset manifest violates its closed root schema")
    dataset_id = _safe_identifier(manifest["datasetId"], "dataset manifest datasetId")
    source = manifest["source"]
    if not isinstance(source, dict) or frozenset(source) != _DATASET_SOURCE_KEYS:
        raise ValueError("production dataset manifest source violates its closed schema")
    source_id = _safe_identifier(source["sourceId"], "dataset manifest sourceId")
    if dataset_id != source_id and not dataset_id.startswith(f"{source_id}-"):
        raise ValueError("dataset manifest datasetId is not namespaced by its sourceId")
    for key in ("name", "officialBase", "adapter", "license", "lastReviewed"):
        if not isinstance(source[key], str) or not source[key] or len(source[key]) > 4_096:
            raise ValueError(f"production dataset manifest source.{key} is invalid")
    if not isinstance(source["licenseEvidence"], list) or not source["licenseEvidence"]:
        raise ValueError("production dataset manifest requires license evidence")
    for index, evidence in enumerate(source["licenseEvidence"]):
        if not isinstance(evidence, dict) or frozenset(evidence) != _LICENSE_EVIDENCE_KEYS:
            raise ValueError(f"dataset license evidence {index} violates its closed schema")
        for key in _LICENSE_EVIDENCE_KEYS:
            value = evidence[key]
            if not isinstance(value, str) or not value or len(value) > 4_096 or "\x00" in value:
                raise ValueError(f"dataset license evidence {index}.{key} is invalid")
        if not evidence["url"].startswith(("https://", "http://")):
            raise ValueError(f"dataset license evidence {index}.url is invalid")
        _validate_portable_relative_path(
            evidence["local_path"], f"dataset license evidence {index}.local_path"
        )
    if not isinstance(source["redistributable"], bool):
        raise ValueError("production dataset manifest redistributable is invalid")
    if source["machineLearningAllowed"] is not True:
        raise ValueError("production dataset source is not approved for machine learning")
    if source["adapter"] != "aobazero_csa":
        raise ValueError("production dataset source adapter is not approved")
    if repository_root is not None:
        from open_shogi_training.data.registry import RegistryError, load_source_registry

        registry_path = repository_root / "configs/data_sources.yaml"
        if not registry_path.is_file():
            registry_path = Path(__file__).resolve().parents[3] / "configs/data_sources.yaml"
        try:
            approved = load_source_registry(registry_path).get(source_id)
        except (OSError, RegistryError) as error:
            raise ValueError(
                "production dataset source is absent from the approved registry"
            ) from error
        expected_source = {
            "sourceId": approved.source_id,
            "name": approved.name,
            "officialBase": approved.official_base,
            "adapter": approved.adapter,
            "license": approved.license,
            "licenseEvidence": [item.as_dict() for item in approved.license_evidence],
            "redistributable": approved.redistributable,
            "machineLearningAllowed": approved.machine_learning_allowed,
            "lastReviewed": approved.last_reviewed.isoformat(),
        }
        if not approved.approved or not approved.enabled or source != expected_source:
            raise ValueError("production dataset source differs from its approved registry entry")
    config = manifest["config"]
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "schema",
            "datasetId",
            "exporterTimeoutSeconds",
            "maxGames",
            "maxPositions",
            "maxRawBytes",
            "split",
            "terminalTailPositions",
        }
        or config.get("schema") != "phase3_normalization_config/v1"
        or config.get("datasetId") != dataset_id
    ):
        raise ValueError("production dataset manifest normalization config is invalid")
    _bounded_int(config["exporterTimeoutSeconds"], "exporter timeout", 1, 3_600)
    maximum_games = _bounded_int(config["maxGames"], "maximum games", 1, 100)
    maximum_positions = _bounded_int(
        config["maxPositions"], "maximum positions", 1, 100 * (2_048 + 1)
    )
    _bounded_int(config["maxRawBytes"], "maximum raw bytes", 1, 16 * 1024**2)
    _bounded_int(config["terminalTailPositions"], "terminal tail", 1, 2_048)
    split = config["split"]
    if (
        not isinstance(split, dict)
        or set(split)
        != {
            "schema",
            "salt",
            "saltSha256",
            "testBasisPoints",
            "validationBasisPoints",
        }
        or split.get("schema") != "phase3_game_split/v1"
        or not isinstance(split.get("salt"), str)
        or not split["salt"]
        or hashlib.sha256(split["salt"].encode()).hexdigest() != split.get("saltSha256")
    ):
        raise ValueError("production dataset manifest split policy is invalid")
    test_basis_points = _bounded_int(split["testBasisPoints"], "test split basis points", 0, 10_000)
    validation_basis_points = _bounded_int(
        split["validationBasisPoints"], "validation split basis points", 0, 10_000
    )
    if test_basis_points + validation_basis_points >= 10_000:
        raise ValueError("production dataset split policy leaves no training partition")
    counts = manifest["counts"]
    if not isinstance(counts, dict) or frozenset(counts) != {"games", "positions"}:
        raise ValueError("production dataset manifest counts are invalid")
    games = _bounded_int(counts["games"], "dataset manifest games", 1, 10_000)
    positions = _bounded_int(counts["positions"], "dataset manifest positions", 1, 100_000_000)
    if games > maximum_games or positions > maximum_positions:
        raise ValueError("dataset manifest counts exceed their configured caps")
    if positions != expected_position_records:
        raise ValueError("dataset manifest position count disagrees with its artifact")
    for key in ("rawObjectSha256", "canonicalGameSha256"):
        hashes = manifest[key]
        if (
            not isinstance(hashes, list)
            or len(hashes) != games
            or hashes != sorted(hashes)
            or len(set(hashes)) != games
            or any(
                not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None for item in hashes
            )
        ):
            raise ValueError(f"production dataset manifest {key} is invalid")
    if not isinstance(manifest["evidenceSnapshots"], list) or not manifest["evidenceSnapshots"]:
        raise ValueError("production dataset manifest requires evidence snapshots")
    snapshot_identities: set[tuple[object, ...]] = set()
    snapshot_id_to_url_type: dict[str, tuple[str, str]] = {}
    snapshot_urls_by_id: dict[str, str] = {}
    snapshot_paths_by_id: dict[str, set[str]] = {}
    previous_snapshot_identity: tuple[object, ...] | None = None
    for index, snapshot in enumerate(manifest["evidenceSnapshots"]):
        if not isinstance(snapshot, dict) or frozenset(snapshot) != _EVIDENCE_SNAPSHOT_KEYS:
            raise ValueError(f"dataset evidence snapshot {index} violates its closed schema")
        for key in ("evidence_id", "url", "retrieved_at", "content_type", "object_path"):
            value = snapshot[key]
            if not isinstance(value, str) or not value or len(value) > 4_096 or "\x00" in value:
                raise ValueError(f"dataset evidence snapshot {index}.{key} is invalid")
        if not snapshot["url"].startswith(("https://", "http://")):
            raise ValueError(f"dataset evidence snapshot {index}.url is invalid")
        _bounded_utc_timestamp(
            snapshot["retrieved_at"], f"dataset evidence snapshot {index}.retrieved_at"
        )
        if _SHA256_RE.fullmatch(str(snapshot["sha256"])) is None:
            raise ValueError(f"dataset evidence snapshot {index}.sha256 is invalid")
        _bounded_int(snapshot["size"], f"dataset evidence snapshot {index}.size", 1, 16 * 1024**2)
        _validate_portable_relative_path(
            snapshot["object_path"], f"dataset evidence snapshot {index}.object_path"
        )
        expected_object_path = f"evidence/sha256/{snapshot['sha256'][:2]}/{snapshot['sha256']}"
        if snapshot["object_path"] != expected_object_path:
            raise ValueError(f"dataset evidence snapshot {index} path/hash identity differs")
        identity = (
            snapshot["evidence_id"],
            snapshot["url"],
            snapshot["retrieved_at"],
            snapshot["sha256"],
            snapshot["size"],
            snapshot["content_type"],
            snapshot["object_path"],
        )
        if identity in snapshot_identities or (
            previous_snapshot_identity is not None and previous_snapshot_identity >= identity
        ):
            raise ValueError(
                "dataset evidence snapshots repeat an identity or are not in canonical order"
            )
        snapshot_identities.add(identity)
        previous_snapshot_identity = identity
        evidence_id = snapshot["evidence_id"]
        url_type = (snapshot["url"], snapshot["content_type"])
        prior_url_type = snapshot_id_to_url_type.setdefault(evidence_id, url_type)
        if prior_url_type != url_type:
            raise ValueError("one dataset evidence ID maps to multiple URL/content types")
        snapshot_urls_by_id[evidence_id] = snapshot["url"]
        snapshot_paths_by_id.setdefault(evidence_id, set()).add(snapshot["object_path"])
    if len(set(snapshot_urls_by_id.values())) != len(snapshot_urls_by_id):
        raise ValueError("different dataset evidence IDs reuse one URL")
    path_owners: dict[str, str] = {}
    for evidence_id, paths in snapshot_paths_by_id.items():
        for path in paths:
            previous_owner = path_owners.setdefault(path, evidence_id)
            if previous_owner != evidence_id:
                raise ValueError("different dataset evidence IDs reuse one object path")
    license_urls = {item["url"] for item in source["licenseEvidence"]}
    if not license_urls.issubset(set(snapshot_urls_by_id.values())):
        raise ValueError("dataset license evidence lacks a durable snapshot")
    artifacts = manifest["artifacts"]
    expected_artifacts = {
        "games-00000.jsonl.gz": games,
        "normalization-report.json": 1,
        "positions-00000.jsonl.gz": positions,
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise ValueError("production dataset manifest artifacts are invalid")
    for name, artifact in artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(artifact, dict):
            raise ValueError("production dataset manifest artifact entry is invalid")
        if frozenset(artifact) != {"sha256", "size", "records"}:
            raise ValueError("production dataset manifest artifact entry is invalid")
        if (
            not isinstance(artifact["sha256"], str)
            or _SHA256_RE.fullmatch(artifact["sha256"]) is None
        ):
            raise ValueError("production dataset manifest artifact hash is invalid")
        _bounded_int(artifact["size"], "dataset manifest artifact size", 1, 4 * 1024**3)
        records = _bounded_int(
            artifact["records"], "dataset manifest artifact records", 1, 100_000_000
        )
        if records != expected_artifacts[name]:
            raise ValueError("production dataset manifest artifact record count is invalid")
    return source_id


def _validate_portable_relative_path(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{context} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{context} is not a normalized relative path")


def _load_json_object(path: Path, *, max_bytes: int) -> dict[str, Any]:
    value, _, _ = _load_json_object_with_identity(path, max_bytes=max_bytes)
    return value


def _require_unchanged(path: Path, expected_sha256: str, *, max_bytes: int) -> None:
    observed_sha256, _ = hash_file(path, max_bytes=max_bytes)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"artifact changed while loading: {path}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]
