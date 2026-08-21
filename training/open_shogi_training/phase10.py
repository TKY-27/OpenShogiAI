"""Closed validator for the frozen Phase 10 data curriculum."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

import yaml

from open_shogi_training.data.registry import UniqueSafeLoader

MAX_YAML_BYTES: Final = 2 * 1024 * 1024
MAX_HASH_MANIFEST_BYTES: Final = 64 * 1024

CONFIG_PATHS: Final = (
    "configs/phase10/dataset-mixture.yaml",
    "configs/phase10/curriculum.yaml",
    "configs/phase10/split-policy.yaml",
    "configs/phase10/experiment-matrix.yaml",
    "configs/phase10/resource-budget.yaml",
    "configs/phase10/statistical-gates.yaml",
)
FROZEN_HASH_PATHS: Final = frozenset(
    {
        "PHASE_10_FROZEN_PLAN.md",
        *CONFIG_PATHS,
        "prompts/LUNA_PHASE10_EXECUTION.md",
        "training/open_shogi_training/phase10.py",
        "configs/data_sources.yaml",
        "configs/data_sources_external.yaml",
        "configs/data_source_objects/aobazero_no_noise.yaml",
        "artifacts/phase10a/audit-manifest.json",
        "artifacts/phase4/teacher/labels-v2/manifest.json",
        "docs/data/DATASET_OVERLAP_REPORT.md",
        "docs/data/FORMAT_COMPATIBILITY_REPORT.md",
        "configs/evaluation/overall_champion_tactical_suite.json",
        "PHASE_9_REPORT.md",
        "PHASE_10A_REPORT.md",
    }
)

EXPECTED_DECISIONS: Final = {
    "aobazero-no-noise-exact100": "accepted",
    "aobazero-no-noise-w4745-sample": "accepted",
    "aobazero-live-no-noise-sample-index": "deferred",
    "aobazero-daily-public-records": "deferred",
    "dlshogi-gct-hcpe3-selfplay": "deferred",
    "dlshogi-gct-floodgate-hcpe3": "deferred",
    "dlshogi-gct-taya36-hcpe3": "deferred",
    "dlshogi-gct-model-taya36-hcpe3": "deferred",
    "dlshogi-gct-aobazero-hcpe": "deferred",
    "dlshogi-gct-floodgate-play-hcpe": "deferred",
    "dlshogi-gct-taya36-rl-hcpe": "deferred",
    "dlshogi-gct-yaneura36-rl-hcpe": "deferred",
    "dlshogi-gct-suisho-teacher-hcpe": "deferred",
    "nodchip-shogi-hao-depth9": "deferred",
    "nodchip-tanuki-nnue-pytorch-2024-07-30-1": "deferred",
    "tayayan-gokaku-36-sfen": "deferred",
    "tayayan-gokaku-36-generated-5247": "deferred",
    "tayayan-suisho-teacher-kif": "deferred",
    "tayayan-suisho-teacher-psv": "deferred",
    "qhapaq-qpd-train": "deferred",
    "dlshogi-public-evaluation-test": "rejected",
    "open-shogiai-floodgate": "rejected",
}
EXPECTED_AUDIT_STATUS: Final = {
    artifact_id: (
        "approved" if decision == "accepted" else "denied" if decision == "rejected" else "pending"
    )
    for artifact_id, decision in EXPECTED_DECISIONS.items()
}
ROLE_SET: Final = frozenset(
    {
        "policy_pretraining",
        "value_wdl_pretraining",
        "ranking_supervision",
        "opening_start_position_pool",
        "validation",
        "external_holdout",
    }
)
STAGE_WEIGHT_KEYS: Final = frozenset(
    {"representation_policy", "source_value", "canonical_calibration", "hard_position_finetune"}
)


class Phase10ValidationError(ValueError):
    """Raised when a frozen control is missing, altered, or internally inconsistent."""


def validate_phase10(root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate every frozen Phase 10 control and its upstream evidence."""

    root = root.resolve()
    controls = {path: _load_yaml(root / path) for path in CONFIG_PATHS}
    audit = _load_yaml(root / "configs/data_sources_external.yaml")
    audit_status = _validate_audit_catalog(audit)

    _validate_dataset_mixture(controls[CONFIG_PATHS[0]], audit_status)
    _validate_curriculum(controls[CONFIG_PATHS[1]])
    _validate_split_policy(controls[CONFIG_PATHS[2]])
    _validate_experiment_matrix(controls[CONFIG_PATHS[3]])
    _validate_resource_budget(controls[CONFIG_PATHS[4]])
    _validate_statistical_gates(controls[CONFIG_PATHS[5]])
    if verify_hashes:
        _verify_hash_manifest(root)

    return {
        "schema": "open_shogi_phase10_validation/v1",
        "status": "valid",
        "source_decisions": dict(Counter(EXPECTED_DECISIONS.values())),
        "audited_artifacts": len(EXPECTED_DECISIONS),
        "frozen_hashes": len(FROZEN_HASH_PATHS) if verify_hashes else None,
    }


def _validate_audit_catalog(raw: object) -> dict[str, str]:
    root = _mapping(raw, "external audit catalog")
    if root.get("schema_version") != 1:
        raise Phase10ValidationError("external audit catalog must use schema_version 1")
    artifacts = _sequence(root.get("artifacts"), "external audit catalog.artifacts")
    statuses: dict[str, str] = {}
    for index, value in enumerate(artifacts):
        item = _mapping(value, f"external audit catalog.artifacts[{index}]")
        artifact_id = _string(item.get("artifact_id"), f"artifact[{index}].artifact_id")
        status = _string(item.get("status"), f"artifact[{index}].status")
        if artifact_id in statuses:
            raise Phase10ValidationError(f"duplicate audited artifact: {artifact_id}")
        statuses[artifact_id] = status
    if statuses != EXPECTED_AUDIT_STATUS:
        raise Phase10ValidationError(
            "external audit artifact IDs or statuses differ from Phase 10 freeze"
        )
    counts = Counter(statuses.values())
    expected_counts = {"approved": 2, "pending": 18, "denied": 2}
    if dict(counts) != expected_counts:
        raise Phase10ValidationError(f"external audit status counts differ: {dict(counts)!r}")
    declared = _mapping(root.get("decision_counts"), "external audit catalog.decision_counts")
    for status, count in expected_counts.items():
        if declared.get(status) != count:
            raise Phase10ValidationError(f"external audit decision_counts.{status} must be {count}")
    return statuses


def _validate_dataset_mixture(raw: object, audit_status: Mapping[str, str]) -> None:
    root = _mapping(raw, "dataset mixture")
    _schema(root, "open_shogi_phase10_dataset_mixture/v1", "dataset mixture")
    if root.get("deny_by_default") is not True:
        raise Phase10ValidationError("dataset mixture must deny by default")
    roles = _string_set(root.get("roles"), "dataset mixture.roles")
    if roles != ROLE_SET:
        raise Phase10ValidationError("dataset mixture role vocabulary differs from the freeze")
    global_rules = _mapping(root.get("global_rules"), "dataset mixture.global_rules")
    _expect(global_rules, "one_training_row_per_canonical_position", True)
    _expect(global_rules, "average_cross_source_scores", False)
    _expect(global_rules, "convert_unknown_scores_to_centipawns", False)
    _expect_number(global_rules, "pending_license_weight", 0.0)
    _expect_number(global_rules, "rejected_weight", 0.0)

    sources = _sequence(root.get("sources"), "dataset mixture.sources")
    if len(sources) != len(EXPECTED_DECISIONS):
        raise Phase10ValidationError("dataset mixture must contain all 22 audited artifacts")
    seen: set[str] = set()
    for index, value in enumerate(sources):
        item = _mapping(value, f"dataset mixture.sources[{index}]")
        artifact_id = _string(item.get("artifact_id"), f"source[{index}].artifact_id")
        if artifact_id in seen:
            raise Phase10ValidationError(f"duplicate dataset mixture artifact: {artifact_id}")
        seen.add(artifact_id)
        if artifact_id not in EXPECTED_DECISIONS:
            raise Phase10ValidationError(f"unknown frozen artifact: {artifact_id}")
        decision = item.get("curriculum_decision")
        if decision != EXPECTED_DECISIONS[artifact_id]:
            raise Phase10ValidationError(
                f"{artifact_id} decision must remain {EXPECTED_DECISIONS[artifact_id]}"
            )
        if item.get("audit_status") != audit_status[artifact_id]:
            raise Phase10ValidationError(
                f"{artifact_id} audit status differs from the audit catalog"
            )
        if decision == "accepted" and audit_status[artifact_id] != "approved":
            raise Phase10ValidationError(
                f"pending or denied artifact cannot be accepted: {artifact_id}"
            )
        item_roles = _string_set(item.get("roles"), f"{artifact_id}.roles")
        prohibited = _string_set(item.get("prohibited_roles"), f"{artifact_id}.prohibited_roles")
        if not item_roles <= ROLE_SET or not prohibited <= ROLE_SET or item_roles & prohibited:
            raise Phase10ValidationError(f"{artifact_id} has invalid or overlapping roles")
        weights = _mapping(item.get("stage_weights"), f"{artifact_id}.stage_weights")
        if set(weights) != STAGE_WEIGHT_KEYS:
            raise Phase10ValidationError(f"{artifact_id} stage weight keys differ from the freeze")
        numeric_weights = [
            _nonnegative_number(value, f"{artifact_id}.{key}") for key, value in weights.items()
        ]
        contribution = _nonnegative_number(
            item.get("maximum_contribution_fraction"),
            f"{artifact_id}.maximum_contribution_fraction",
        )
        if contribution > 1.0 or any(weight > 1.0 for weight in numeric_weights):
            raise Phase10ValidationError(f"{artifact_id} weight or contribution exceeds one")
        if decision != "accepted" and (contribution != 0.0 or any(numeric_weights)):
            raise Phase10ValidationError(
                f"non-accepted artifact has nonzero contribution: {artifact_id}"
            )
        if artifact_id == "aobazero-no-noise-w4745-sample" and (
            contribution != 0.0 or any(numeric_weights) or item_roles
        ):
            raise Phase10ValidationError("w4745 must remain a zero-weight contained audit fixture")
        _string(item.get("exact_artifact_subset"), f"{artifact_id}.exact_artifact_subset")
        _string(item.get("target_constraints"), f"{artifact_id}.target_constraints")
        _string(
            item.get("attribution_release_constraints"),
            f"{artifact_id}.attribution_release_constraints",
        )
    if seen != set(EXPECTED_DECISIONS):
        raise Phase10ValidationError("dataset mixture is missing a frozen audited artifact")


def _validate_curriculum(raw: object) -> None:
    root = _mapping(raw, "curriculum")
    _schema(root, "open_shogi_phase10_curriculum/v1", "curriculum")
    _expect(root, "final_runtime_evaluator", "pure_learned")
    _expect(root, "handcrafted_runtime_score_contribution", False)
    _expect(root, "third_party_initial_weights", False)
    semantics = _mapping(root.get("target_semantics"), "curriculum.target_semantics")
    _expect(semantics, "canonical_perspective", "side_to_move")
    _expect(
        _mapping(semantics.get("hcpe3_mcts_policy"), "hcpe3 policy"), "active_in_frozen_plan", False
    )
    _expect(_mapping(semantics.get("mate_scores"), "mate scores"), "runtime_mapping", "none")
    unknown = _mapping(semantics.get("unknown_score"), "unknown score")
    _expect_number(unknown, "value_loss_mask", 0.0)

    stages = _sequence(root.get("stages"), "curriculum.stages")
    expected_ids = [
        "representation_policy",
        "source_value",
        "canonical_calibration",
        "hard_position_finetune",
        "equal_wall_clock_screening",
        "bounded_self_play",
    ]
    actual_ids = [_mapping(stage, "curriculum stage").get("stage_id") for stage in stages]
    if actual_ids != expected_ids:
        raise Phase10ValidationError("curriculum stage order differs from the freeze")
    if [_mapping(stage, "curriculum stage").get("order") for stage in stages] != list(range(1, 7)):
        raise Phase10ValidationError("curriculum orders must be 1 through 6")
    selfplay = _mapping(stages[-1], "bounded self-play")
    _expect(selfplay, "active", False)
    _expect(selfplay, "activation_gate", "supervised_48_entry")
    _expect(_mapping(selfplay.get("first_generation_limit"), "self-play limit"), "games", 200)
    _expect(selfplay, "promotion_allowed", False)
    external = _mapping(
        _mapping(root.get("expansion_conditions"), "expansion conditions").get(
            "external_source_activation"
        ),
        "external source activation",
    )
    _expect(external, "allowed", False)


def _validate_split_policy(raw: object) -> None:
    root = _mapping(raw, "split policy")
    _schema(root, "open_shogi_phase10_split_policy/v1", "split policy")
    _expect(root, "assignment_unit", "protected_game_or_history_group_before_position_extraction")
    _expect(root, "addition_stable", True)
    canonical = _mapping(root.get("canonical_position"), "canonical position")
    _expect(canonical, "mirror_equivalence", False)
    _expect(canonical, "color_rotation_equivalence", False)
    assignment = _mapping(root.get("assignment"), "split assignment")
    ratios = _mapping(assignment.get("ratios_basis_points"), "split ratios")
    if ratios != {"train": 8000, "validation": 1000, "final_holdout": 1000}:
        raise Phase10ValidationError("split ratios differ from 80/10/10")
    priority = assignment.get("cross_split_duplicate_priority")
    if priority != [
        "public_test_holdout",
        "final_holdout",
        "source_held_out",
        "validation",
        "train",
    ]:
        raise Phase10ValidationError("cross-split duplicate priority differs from the freeze")
    _expect(assignment, "split_after_position_extraction", False)
    cross = _mapping(root.get("cross_source_duplicates"), "cross-source duplicates")
    _expect(cross, "one_training_row_per_canonical_position", True)
    _expect(cross, "holdout_precedence_over_training", True)
    _expect(cross, "lineage_unknown_means_no_training", True)
    style = _mapping(root.get("style_coverage"), "style coverage")
    arena_pool = _mapping(style.get("arena_start_pool"), "arena start pool")
    _expect(arena_pool, "minimum_unique_per_group", 50)
    _expect(arena_pool, "maximum_pair_reuse", 1)
    holdouts = _sequence(root.get("immutable_holdouts"), "immutable holdouts")
    if [
        item.get("holdout_id") for item in map(lambda value: _mapping(value, "holdout"), holdouts)
    ] != [
        "phase3-legacy-final-test",
        "tayayan-public-test-family",
        "dlshogi-public-evaluation-test",
    ]:
        raise Phase10ValidationError("immutable holdout identities differ from the freeze")
    legacy = _mapping(holdouts[0], "legacy final holdout")
    _expect(legacy, "current_rows", 1064)
    _expect(legacy, "phase10_maximum_additional_inspections", 1)
    for value in holdouts:
        holdout = _mapping(value, "holdout")
        _expect(holdout, "training_allowed", False)
        _expect(holdout, "validation_allowed", False)
        _expect(holdout, "arena_start_allowed", False)


def _validate_experiment_matrix(raw: object) -> None:
    root = _mapping(raw, "experiment matrix")
    _schema(root, "open_shogi_phase10_experiment_matrix/v1", "experiment matrix")
    contract = _mapping(root.get("architecture_contract"), "architecture contract")
    _expect(contract, "runtime_container", "OSAVAL01")
    _expect(contract, "runtime_architecture_version", 1)
    _expect(contract, "runtime_output", "one side-to-move scalar value")
    variants = _sequence(root.get("variants"), "experiment variants")
    expected = {
        "pure-v1-2x128-control": (309634, 309505, 2287, 2, 128),
        "pure-v1-3x256-attacks": (759298, 759041, 2449, 3, 256),
    }
    if len(variants) != 2:
        raise Phase10ValidationError("experiment matrix must contain exactly two variants")
    for value in variants:
        item = _mapping(value, "experiment variant")
        variant_id = item.get("variant_id")
        if variant_id not in expected:
            raise Phase10ValidationError(f"unexpected experiment variant: {variant_id}")
        _expect(item, "eligibility", "pure_learned_objective")
        training, exported, input_dim, layers, width = expected[variant_id]
        parameters = _mapping(item.get("parameters"), f"{variant_id}.parameters")
        if parameters != {"training": training, "exported": exported}:
            raise Phase10ValidationError(f"{variant_id} parameter count differs from the freeze")
        _expect(
            _mapping(item.get("features"), f"{variant_id}.features"), "input_dimension", input_dim
        )
        model = _mapping(item.get("model"), f"{variant_id}.model")
        _expect(model, "hidden_layers", layers)
        _expect(model, "hidden_dimension", width)
    selection = _mapping(root.get("selection"), "experiment selection")
    _expect(selection, "training_replicates_per_variant", 1)
    _expect(selection, "final_holdout_selection_use", False)
    _expect(selection, "maximum_new_supervised_training_runs", 3)
    _expect(selection, "maximum_new_arena_candidates", 2)
    performance = _mapping(root.get("performance_constraints"), "performance constraints")
    _expect(performance, "maximum_exported_parameters", 800000)
    _expect(performance, "quantization_for_strength_gates", "float32")


def _validate_resource_budget(raw: object) -> None:
    root = _mapping(raw, "resource budget")
    _schema(root, "open_shogi_phase10_resource_budget/v1", "resource budget")
    host = _mapping(root.get("reference_host"), "reference host")
    _expect(host, "physical_memory_gib", 24)
    memory = _mapping(root.get("memory"), "memory budget")
    if _number(memory.get("maximum_aggregate_working_memory_gib"), "working memory") > 20:
        raise Phase10ValidationError("aggregate working memory exceeds 20 GiB")
    if _number(memory.get("maximum_process_tree_rss_gib"), "process RSS") > 16:
        raise Phase10ValidationError("process RSS exceeds 16 GiB")
    _expect(memory, "maximum_mps_training_workers", 1)
    disk = _mapping(root.get("disk"), "disk budget")
    _expect(disk, "minimum_free_gib", 200)
    _expect(disk, "maximum_incremental_phase10_gib", 32)
    _expect(disk, "full_external_dataset_download_allowed", False)
    allocations = _mapping(disk.get("allocations_bytes"), "disk allocations")
    if sum(
        _nonnegative_integer(value, f"disk allocation {key}") for key, value in allocations.items()
    ) != disk.get("maximum_incremental_phase10_bytes"):
        raise Phase10ValidationError("disk allocations must sum to the 32 GiB cap")


def _validate_statistical_gates(raw: object) -> None:
    root = _mapping(raw, "statistical gates")
    _schema(root, "open_shogi_phase10_statistical_gates/v1", "statistical gates")
    gates = _sequence(root.get("gates"), "statistical gates.gates")
    expected_ids = [
        "data_integrity",
        "offline_gate",
        "arena_pilot_40",
        "supervised_48_entry",
        "final_pure_learned_55",
    ]
    mapped = [_mapping(value, "statistical gate") for value in gates]
    if [gate.get("gate_id") for gate in mapped] != expected_ids:
        raise Phase10ValidationError("statistical gate order differs from the freeze")
    offline = _mapping(mapped[1].get("requirements"), "offline requirements")
    _expect_number(offline, "approved_teacher_cp_raw_mae_maximum", 870.0)
    _expect_number(offline, "engine_bestmove_agreement_minimum", 0.18)
    pilot = mapped[2]
    _expect(pilot, "games", 40)
    _expect(pilot, "unique_paired_starts_per_group", 5)
    _expect_number(
        _mapping(pilot.get("requirements"), "pilot requirements"), "minimum_score_rate", 0.30
    )
    entry = mapped[3]
    _expect(entry, "games", 160)
    _expect(entry, "unique_paired_starts_per_group", 20)
    entry_requirements = _mapping(entry.get("requirements"), "entry requirements")
    _expect_number(entry_requirements, "minimum_score_rate", 0.48)
    _expect_number(entry_requirements, "minimum_wilson_lower", 0.40)
    final = mapped[4]
    _expect(final, "games", 400)
    _expect(final, "unique_paired_starts_per_group", 50)
    final_requirements = _mapping(final.get("requirements"), "final requirements")
    _expect_number(final_requirements, "minimum_score_rate", 0.55)
    _expect_number(final_requirements, "minimum_wilson_lower", 0.50)
    _expect(final_requirements, "handcrafted_runtime_score_contribution", False)
    _expect(final_requirements, "third_party_weights", False)
    arena = _mapping(root.get("arena_contract"), "arena contract")
    for key, value in {
        "incumbent": "handcrafted-experimental",
        "movetime_ms": 10,
        "depth_cap": 8,
        "hash_mib_per_player": 32,
        "max_plies": 128,
        "paired_colors": True,
        "opening_enabled": False,
        "seed": 20260821,
    }.items():
        _expect(arena, key, value)
    holdout = _mapping(root.get("holdout_gate"), "holdout gate")
    _expect(holdout, "maximum_phase10_final_holdout_inspections", 1)
    _expect(holdout, "public_test_inspections", 0)


def _verify_hash_manifest(root: Path) -> None:
    manifest_path = root / "configs/phase10/frozen-controls.sha256"
    raw = _read_bounded(manifest_path, MAX_HASH_MANIFEST_BYTES)
    entries = _parse_hash_lines(raw.decode("utf-8"))
    if set(entries) != FROZEN_HASH_PATHS:
        missing = sorted(FROZEN_HASH_PATHS - set(entries))
        extra = sorted(set(entries) - FROZEN_HASH_PATHS)
        raise Phase10ValidationError(
            f"frozen hash path set differs; missing={missing}, extra={extra}"
        )
    for relative, expected in entries.items():
        path = root / relative
        actual = hashlib.sha256(_read_bounded(path, _hash_read_limit(relative))).hexdigest()
        if actual != expected:
            raise Phase10ValidationError(
                f"frozen hash mismatch for {relative}: {actual} != {expected}"
            )


def _parse_hash_lines(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not text.endswith("\n"):
        raise Phase10ValidationError("frozen hash manifest must end with LF")
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_./-]*)", line)
        if match is None:
            raise Phase10ValidationError(f"invalid frozen hash line {line_number}")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise Phase10ValidationError(f"unsafe frozen hash path on line {line_number}")
        if relative in entries:
            raise Phase10ValidationError(f"duplicate frozen hash path: {relative}")
        entries[relative] = digest
    if list(entries) != sorted(entries):
        raise Phase10ValidationError("frozen hash manifest paths must be sorted")
    return entries


def _load_yaml(path: Path) -> object:
    raw = _read_bounded(path, MAX_YAML_BYTES)
    try:
        return yaml.load(raw.decode("utf-8"), Loader=UniqueSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Phase10ValidationError(f"invalid YAML in {path}: {error}") from error


def _read_bounded(path: Path, maximum: int) -> bytes:
    try:
        stat = path.stat()
    except OSError as error:
        raise Phase10ValidationError(f"cannot stat required file {path}: {error}") from error
    if not path.is_file() or path.is_symlink() or not 0 < stat.st_size <= maximum:
        raise Phase10ValidationError(
            f"required file is missing, symlinked, empty, or oversized: {path}"
        )
    try:
        data = path.read_bytes()
    except OSError as error:
        raise Phase10ValidationError(f"cannot read required file {path}: {error}") from error
    if len(data) != stat.st_size:
        raise Phase10ValidationError(f"required file changed while reading: {path}")
    return data


def _hash_read_limit(relative: str) -> int:
    if relative == "artifacts/phase4/teacher/labels-v2/manifest.json":
        return 2 * 1024 * 1024
    return 4 * 1024 * 1024


def _schema(root: Mapping[str, Any], expected: str, context: str) -> None:
    if root.get("schema") != expected:
        raise Phase10ValidationError(f"{context} schema must be {expected}")


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Phase10ValidationError(f"{context} must be a string-keyed mapping")
    return value


def _sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise Phase10ValidationError(f"{context} must be a list")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase10ValidationError(f"{context} must be a nonempty string")
    return value


def _string_set(value: object, context: str) -> frozenset[str]:
    values = _sequence(value, context)
    if not all(isinstance(item, str) for item in values) or len(values) != len(set(values)):
        raise Phase10ValidationError(f"{context} must contain unique strings")
    return frozenset(values)


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Phase10ValidationError(f"{context} must be numeric")
    return float(value)


def _nonnegative_number(value: object, context: str) -> float:
    numeric = _number(value, context)
    if numeric < 0.0:
        raise Phase10ValidationError(f"{context} must be nonnegative")
    return numeric


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase10ValidationError(f"{context} must be a nonnegative integer")
    return value


def _expect(table: Mapping[str, Any], key: str, expected: object) -> None:
    if table.get(key) != expected:
        raise Phase10ValidationError(f"{key} must remain {expected!r}")


def _expect_number(table: Mapping[str, Any], key: str, expected: float) -> None:
    actual = _number(table.get(key), key)
    if actual != expected:
        raise Phase10ValidationError(f"{key} must remain {expected!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify frozen Phase 10 controls and hashes")
    verify.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = validate_phase10(arguments.root)
    except Phase10ValidationError as error:
        print(f"phase10 validation failed: {error}", file=sys.stderr)
        return 1
    print(yaml.safe_dump(result, sort_keys=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
