"""Strict TOML configuration for bounded Phase 6 generation work."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .common import (
    ContractError,
    canonical_sha256,
    require_bool,
    require_enum,
    require_exact_keys,
    require_int,
    require_mapping,
    require_number,
    require_relative_path,
    stable_regular_descriptor,
)

CONFIG_SCHEMA_VERSION: Final = 1
MAX_CONFIG_BYTES: Final = 64 * 1024
MAX_JSON_SAFE_INTEGER: Final = 9_007_199_254_740_991
MAX_WORKING_MEMORY_MIB: Final = 8 * 1024
PHASE6_GAMES: Final = 40
PHASE6_NORMAL_START_PAIRS: Final = 10
PHASE6_START_SET_PAIRS: Final = 10
PHASE6_SEED: Final = 20_260_808
PHASE6_NODES_PER_MOVE: Final = 500
PHASE6_MAX_PLIES: Final = 256
PHASE6_WORKERS: Final = 2

_SELFPLAY_ROOT_KEYS = frozenset(
    {"schema_version", "run", "resources", "paths", "hard_positions", "replay"}
)
_RUN_KEYS = frozenset(
    {
        "games",
        "normal_start_pairs",
        "start_set_pairs",
        "seed",
        "nodes_per_move",
        "max_plies",
        "search_depth",
        "hash_mib",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "workers",
        "memory_limit_mib",
        "memory_per_worker_mib",
        "game_timeout_seconds",
        "command_timeout_seconds",
    }
)
_PATH_KEYS = frozenset({"engine_cli", "output_root", "start_positions_manifest"})
_HARD_KEYS = frozenset(
    {
        "teacher_drop_cp",
        "evaluation_disagreement_cp",
        "candidate_gap_cp",
        "minimum_search_nodes",
        "max_additional_labels",
        "teacher_label_limit",
    }
)
_REPLAY_KEYS = frozenset({"capacity", "minimum_older_positions", "dedup_key"})

_POLICY_ROOT_KEYS = frozenset({"schema_version", "evidence", "thresholds", "performance"})
_EVIDENCE_KEYS = frozenset(
    {"minimum_games", "minimum_decisive_games", "minimum_group_games", "max_illegal", "max_crashes"}
)
_THRESHOLD_KEYS = frozenset(
    {
        "promote_score_rate",
        "promote_wilson_lower",
        "reject_score_rate",
        "reject_wilson_upper",
        "minimum_group_score_rate",
        "maximum_side_score_gap",
    }
)
_PERFORMANCE_KEYS = frozenset(
    {"require_metrics", "maximum_inference_slowdown", "maximum_search_slowdown"}
)


@dataclass(frozen=True, slots=True)
class SelfPlayRunConfig:
    games: int
    normal_start_pairs: int
    start_set_pairs: int
    seed: int
    nodes_per_move: int
    max_plies: int
    search_depth: int
    hash_mib: int


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    workers: int
    memory_limit_mib: int
    memory_per_worker_mib: int
    game_timeout_seconds: int
    command_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class SelfPlayPaths:
    engine_cli: str
    output_root: str
    start_positions_manifest: str


@dataclass(frozen=True, slots=True)
class HardPositionConfig:
    teacher_drop_cp: int
    evaluation_disagreement_cp: int
    candidate_gap_cp: int
    minimum_search_nodes: int
    max_additional_labels: int
    teacher_label_limit: int


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int
    minimum_older_positions: int
    dedup_key: str


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    schema_version: int
    run: SelfPlayRunConfig
    resources: ResourceConfig
    paths: SelfPlayPaths
    hard_positions: HardPositionConfig
    replay: ReplayConfig

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    minimum_games: int
    minimum_decisive_games: int
    minimum_group_games: int
    max_illegal: int
    max_crashes: int


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    promote_score_rate: float
    promote_wilson_lower: float
    reject_score_rate: float
    reject_wilson_upper: float
    minimum_group_score_rate: float
    maximum_side_score_gap: float


@dataclass(frozen=True, slots=True)
class PerformancePolicy:
    require_metrics: bool
    maximum_inference_slowdown: float
    maximum_search_slowdown: float


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    schema_version: int
    evidence: EvidencePolicy
    thresholds: PromotionThresholds
    performance: PerformancePolicy

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def load_selfplay_config(path: Path) -> SelfPlayConfig:
    root = _load_toml(path)
    return _parse_selfplay_config(root)


def parse_selfplay_config_bytes(raw: bytes, context: str) -> SelfPlayConfig:
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ContractError(f"invalid TOML config {context}: {error}") from error
    return _parse_selfplay_config(require_mapping(parsed, f"config {context}"))


def _parse_selfplay_config(root: Mapping[str, Any]) -> SelfPlayConfig:
    require_exact_keys(root, _SELFPLAY_ROOT_KEYS, "selfplay config")
    schema_version = _schema_version(root, "selfplay config")
    run_raw = _table(root, "run", _RUN_KEYS, "selfplay config")
    resources_raw = _table(root, "resources", _RESOURCE_KEYS, "selfplay config")
    paths_raw = _table(root, "paths", _PATH_KEYS, "selfplay config")
    hard_raw = _table(root, "hard_positions", _HARD_KEYS, "selfplay config")
    replay_raw = _table(root, "replay", _REPLAY_KEYS, "selfplay config")

    run = SelfPlayRunConfig(
        games=require_int(run_raw, "games", "run", minimum=40, maximum=100),
        normal_start_pairs=require_int(run_raw, "normal_start_pairs", "run", minimum=1, maximum=49),
        start_set_pairs=require_int(run_raw, "start_set_pairs", "run", minimum=1, maximum=49),
        seed=require_int(run_raw, "seed", "run", minimum=0, maximum=MAX_JSON_SAFE_INTEGER),
        nodes_per_move=require_int(
            run_raw, "nodes_per_move", "run", minimum=1, maximum=1_000_000_000
        ),
        max_plies=require_int(run_raw, "max_plies", "run", minimum=1, maximum=10_000),
        search_depth=require_int(run_raw, "search_depth", "run", minimum=1, maximum=64),
        hash_mib=require_int(run_raw, "hash_mib", "run", minimum=1, maximum=1_024),
    )
    if run.games % 2 != 0:
        raise ContractError("run.games must be even so every start has a color-swapped pair")
    if run.normal_start_pairs + run.start_set_pairs != run.games // 2:
        raise ContractError("normal_start_pairs + start_set_pairs must equal half of run.games")
    required_run = {
        "games": PHASE6_GAMES,
        "normal_start_pairs": PHASE6_NORMAL_START_PAIRS,
        "start_set_pairs": PHASE6_START_SET_PAIRS,
        "seed": PHASE6_SEED,
        "nodes_per_move": PHASE6_NODES_PER_MOVE,
        "max_plies": PHASE6_MAX_PLIES,
    }
    for field, required in required_run.items():
        if getattr(run, field) != required:
            raise ContractError(f"run.{field} must be {required} for the Phase 6 bounded run")

    resources = ResourceConfig(
        workers=require_int(resources_raw, "workers", "resources", minimum=1, maximum=4),
        memory_limit_mib=require_int(
            resources_raw,
            "memory_limit_mib",
            "resources",
            minimum=256,
            maximum=MAX_WORKING_MEMORY_MIB,
        ),
        memory_per_worker_mib=require_int(
            resources_raw,
            "memory_per_worker_mib",
            "resources",
            minimum=128,
            maximum=MAX_WORKING_MEMORY_MIB,
        ),
        game_timeout_seconds=require_int(
            resources_raw,
            "game_timeout_seconds",
            "resources",
            minimum=1,
            maximum=86_400,
        ),
        command_timeout_seconds=require_int(
            resources_raw,
            "command_timeout_seconds",
            "resources",
            minimum=1,
            maximum=172_800,
        ),
    )
    if resources.workers * resources.memory_per_worker_mib > resources.memory_limit_mib:
        raise ContractError("workers * memory_per_worker_mib exceeds memory_limit_mib")
    if resources.workers != PHASE6_WORKERS:
        raise ContractError(f"resources.workers must be {PHASE6_WORKERS} for Phase 6")
    if resources.memory_limit_mib != MAX_WORKING_MEMORY_MIB:
        raise ContractError(
            f"resources.memory_limit_mib must be {MAX_WORKING_MEMORY_MIB} for Phase 6"
        )

    paths = SelfPlayPaths(
        engine_cli=require_relative_path(paths_raw, "engine_cli", "paths"),
        output_root=require_relative_path(paths_raw, "output_root", "paths"),
        start_positions_manifest=require_relative_path(
            paths_raw, "start_positions_manifest", "paths"
        ),
    )

    hard_positions = HardPositionConfig(
        teacher_drop_cp=require_int(
            hard_raw, "teacher_drop_cp", "hard_positions", minimum=1, maximum=100_000
        ),
        evaluation_disagreement_cp=require_int(
            hard_raw,
            "evaluation_disagreement_cp",
            "hard_positions",
            minimum=1,
            maximum=100_000,
        ),
        candidate_gap_cp=require_int(
            hard_raw, "candidate_gap_cp", "hard_positions", minimum=0, maximum=100_000
        ),
        minimum_search_nodes=require_int(
            hard_raw,
            "minimum_search_nodes",
            "hard_positions",
            minimum=1,
            maximum=1_000_000_000,
        ),
        max_additional_labels=require_int(
            hard_raw,
            "max_additional_labels",
            "hard_positions",
            minimum=0,
            maximum=10_000,
        ),
        teacher_label_limit=require_int(
            hard_raw,
            "teacher_label_limit",
            "hard_positions",
            minimum=1,
            maximum=10_000,
        ),
    )
    if hard_positions.teacher_label_limit != 10_000:
        raise ContractError("hard_positions.teacher_label_limit must preserve the 10000-label cap")

    replay = ReplayConfig(
        capacity=require_int(replay_raw, "capacity", "replay", minimum=1, maximum=10_000_000),
        minimum_older_positions=require_int(
            replay_raw,
            "minimum_older_positions",
            "replay",
            minimum=0,
            maximum=10_000_000,
        ),
        dedup_key=require_enum(replay_raw, "dedup_key", "replay", {"canonical_sfen"}),
    )
    if replay.minimum_older_positions > replay.capacity:
        raise ContractError("minimum_older_positions must not exceed replay capacity")

    return SelfPlayConfig(
        schema_version=schema_version,
        run=run,
        resources=resources,
        paths=paths,
        hard_positions=hard_positions,
        replay=replay,
    )


def load_generation_policy(path: Path) -> GenerationPolicy:
    return _parse_generation_policy(_load_toml(path))


def parse_generation_policy_bytes(raw: bytes, context: str) -> GenerationPolicy:
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ContractError(f"invalid TOML generation policy {context}: {error}") from error
    return _parse_generation_policy(require_mapping(parsed, f"generation policy {context}"))


def _parse_generation_policy(root: Mapping[str, Any]) -> GenerationPolicy:
    require_exact_keys(root, _POLICY_ROOT_KEYS, "generation policy")
    schema_version = _schema_version(root, "generation policy")
    evidence_raw = _table(root, "evidence", _EVIDENCE_KEYS, "generation policy")
    thresholds_raw = _table(root, "thresholds", _THRESHOLD_KEYS, "generation policy")
    performance_raw = _table(root, "performance", _PERFORMANCE_KEYS, "generation policy")

    evidence = EvidencePolicy(
        minimum_games=require_int(
            evidence_raw, "minimum_games", "evidence", minimum=40, maximum=10_000
        ),
        minimum_decisive_games=require_int(
            evidence_raw, "minimum_decisive_games", "evidence", minimum=1, maximum=10_000
        ),
        minimum_group_games=require_int(
            evidence_raw, "minimum_group_games", "evidence", minimum=2, maximum=5_000
        ),
        max_illegal=require_int(evidence_raw, "max_illegal", "evidence", minimum=0, maximum=100),
        max_crashes=require_int(evidence_raw, "max_crashes", "evidence", minimum=0, maximum=100),
    )
    if evidence.minimum_games != PHASE6_GAMES:
        raise ContractError(f"evidence.minimum_games must be {PHASE6_GAMES} for Phase 6")
    if evidence.minimum_decisive_games != 12:
        raise ContractError("evidence.minimum_decisive_games must be 12 for Phase 6")
    if evidence.minimum_group_games != 10:
        raise ContractError("evidence.minimum_group_games must be 10 for Phase 6")
    if evidence.max_illegal != 0 or evidence.max_crashes != 0:
        raise ContractError("Phase 6 promotion requires zero illegal moves and zero crashes")
    thresholds = PromotionThresholds(
        promote_score_rate=require_number(
            thresholds_raw, "promote_score_rate", "thresholds", minimum=0.5, maximum=1.0
        ),
        promote_wilson_lower=require_number(
            thresholds_raw,
            "promote_wilson_lower",
            "thresholds",
            minimum=0.5,
            maximum=1.0,
        ),
        reject_score_rate=require_number(
            thresholds_raw, "reject_score_rate", "thresholds", minimum=0.0, maximum=0.5
        ),
        reject_wilson_upper=require_number(
            thresholds_raw,
            "reject_wilson_upper",
            "thresholds",
            minimum=0.0,
            maximum=0.5,
        ),
        minimum_group_score_rate=require_number(
            thresholds_raw,
            "minimum_group_score_rate",
            "thresholds",
            minimum=0.0,
            maximum=1.0,
        ),
        maximum_side_score_gap=require_number(
            thresholds_raw,
            "maximum_side_score_gap",
            "thresholds",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    if thresholds.reject_score_rate >= thresholds.promote_score_rate:
        raise ContractError("reject_score_rate must be below promote_score_rate")

    performance = PerformancePolicy(
        require_metrics=require_bool(performance_raw, "require_metrics", "performance"),
        maximum_inference_slowdown=require_number(
            performance_raw,
            "maximum_inference_slowdown",
            "performance",
            minimum=1.0,
            maximum=100.0,
        ),
        maximum_search_slowdown=require_number(
            performance_raw,
            "maximum_search_slowdown",
            "performance",
            minimum=1.0,
            maximum=100.0,
        ),
    )
    return GenerationPolicy(
        schema_version=schema_version,
        evidence=evidence,
        thresholds=thresholds,
        performance=performance,
    )


def _load_toml(path: Path) -> Mapping[str, Any]:
    try:
        with stable_regular_descriptor(path) as descriptor:
            metadata = os.fstat(descriptor)
            if metadata.st_size > MAX_CONFIG_BYTES:
                raise ContractError(f"config exceeds {MAX_CONFIG_BYTES} bytes: {path}")
            raw = _read_descriptor_bytes(descriptor, metadata.st_size, path)
            parsed = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ContractError(f"invalid UTF-8 TOML config {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"invalid TOML config {path}: {error}") from error
    return require_mapping(parsed, f"config {path}")


def _read_descriptor_bytes(descriptor: int, size: int, path: Path) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while observed < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - observed), observed)
        if not chunk:
            raise ContractError(f"config changed while reading: {path}")
        chunks.append(chunk)
        observed += len(chunk)
    return b"".join(chunks)


def _schema_version(root: Mapping[str, Any], context: str) -> int:
    version = require_int(root, "schema_version", context)
    if version != CONFIG_SCHEMA_VERSION:
        raise ContractError(f"{context} has unsupported schema_version: {version}")
    return version


def _table(
    root: Mapping[str, Any], key: str, expected: frozenset[str], context: str
) -> Mapping[str, Any]:
    table = require_mapping(root.get(key), f"{context}.{key}")
    require_exact_keys(table, expected, f"{context}.{key}")
    return table
