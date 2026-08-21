"""Frozen Phase 10R-B controls and bounded pipeline sanity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from torch import nn

from open_shogi_training.data.registry import UniqueSafeLoader

MAX_YAML_BYTES: Final = 2 * 1024 * 1024
HASH_MANIFEST: Final = "configs/phase10r/frozen-controls.sha256"
CONFIG_SCHEMAS: Final = {
    "configs/phase10r/dataset-mixture.yaml": "open_shogiai_phase10r_dataset_mixture/v1",
    "configs/phase10r/curriculum.yaml": "open_shogiai_phase10r_curriculum/v1",
    "configs/phase10r/model-matrix.yaml": "open_shogiai_phase10r_model_matrix/v1",
    "configs/phase10r/target-semantics.yaml": "open_shogiai_phase10r_target_semantics/v1",
    "configs/phase10r/active-learning.yaml": "open_shogiai_phase10r_active_learning/v1",
    "configs/phase10r/arena-gates.yaml": "open_shogiai_phase10r_arena_gates/v1",
    "configs/phase10r/selfplay.yaml": "open_shogiai_phase10r_selfplay/v1",
    "configs/phase10r/resource-budget.yaml": "open_shogiai_phase10r_resource_budget/v1",
    "configs/phase10r/holdout-policy.yaml": "open_shogiai_phase10r_holdout_policy/v2",
}
FROZEN_PATHS: Final = frozenset(
    {
        "PHASE_10R_FROZEN_PLAN.md",
        "docs/model/PHASE10R_FEATURE_ARCHITECTURE.md",
        "docs/model/PHASE10R_TARGET_SEMANTICS.md",
        "docs/model/PHASE10R_ABLATION_PLAN.md",
        *CONFIG_SCHEMAS,
        "configs/phase10r/source-registry.yaml",
        "configs/phase10r/normalization.yaml",
        "configs/phase10r/deduplication.yaml",
        "configs/phase10r/storage-budget.yaml",
        "configs/teacher/apery-v2.0.0.yaml",
        "prompts/LUNA_PHASE10R_EXECUTION.md",
        "training/open_shogi_training/phase10r.py",
        "tests/python/test_phase10r_freeze.py",
        "PHASE_9_REPORT.md",
        "PHASE_10_START_POOL_REPAIR_REPORT.md",
        "PHASE_10R_DATA_FOUNDATION_REPORT.md",
        "artifacts/phase10/start-pool-manifest.json",
        "artifacts/phase10r/data-foundation-manifest.json",
    }
)
EXPECTED_SCALES: Final = (
    1_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
)
EXPECTED_ARENA_GAMES: Final = {
    "short_screen": 40,
    "arena_400": 400,
    "arena_800": 800,
    "practical_20s": 40,
    "final_1600": 1600,
}
DROP_PIECES: Final = "RBGSNLP"
USI_MOVE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")


class Phase10RValidationError(ValueError):
    """A frozen Phase 10R control or sanity invariant was violated."""


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_YAML_BYTES:
        raise Phase10RValidationError(f"invalid YAML size: {path}")
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=UniqueSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise Phase10RValidationError(f"invalid YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase10RValidationError(f"YAML root must be a mapping: {path}")
    return value


def validate_phase10r(root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate the closed architecture, data, gate, and resource freeze."""

    root = root.resolve()
    configs = {name: _load_yaml(root / name) for name in CONFIG_SCHEMAS}
    for name, schema in CONFIG_SCHEMAS.items():
        if configs[name].get("schema") != schema:
            raise Phase10RValidationError(f"{name} schema must be {schema}")

    registry = _load_yaml(root / "configs/phase10r/source-registry.yaml")
    _validate_dataset_mixture(configs["configs/phase10r/dataset-mixture.yaml"], registry)
    _validate_targets(configs["configs/phase10r/target-semantics.yaml"])
    _validate_models(configs["configs/phase10r/model-matrix.yaml"])
    _validate_curriculum(configs["configs/phase10r/curriculum.yaml"])
    _validate_arena(configs["configs/phase10r/arena-gates.yaml"])
    _validate_resources(configs["configs/phase10r/resource-budget.yaml"])
    _validate_holdout(configs["configs/phase10r/holdout-policy.yaml"], registry)
    if verify_hashes:
        verify_frozen_hashes(root)
    return {
        "schema": "open_shogiai_phase10r_validation/v1",
        "status": "valid",
        "configs": len(CONFIG_SCHEMAS),
        "frozen_hashes": len(FROZEN_PATHS) if verify_hashes else None,
        "approved_external_artifacts": 33,
        "public_weight_sources": 0,
    }


def _validate_dataset_mixture(mixture: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    if mixture.get("deny_by_default") is not True:
        raise Phase10RValidationError("dataset mixture must deny by default")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise Phase10RValidationError("source registry artifacts are invalid")
    approved = {
        item["artifact_id"]
        for item in artifacts
        if isinstance(item, dict) and item.get("state") == "approved"
    }
    if len(approved) != 33:
        raise Phase10RValidationError("approved external artifact count changed")
    sources = mixture.get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        raise Phase10RValidationError("dataset mixture must contain six frozen lanes")
    declared: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise Phase10RValidationError("dataset source must be a mapping")
        artifacts_for_source = source.get("registry_artifacts")
        if not isinstance(artifacts_for_source, list):
            raise Phase10RValidationError("registry_artifacts must be a list")
        declared.update(artifacts_for_source)
        if source.get("may_affect_public_weights") is not False:
            raise Phase10RValidationError("no frozen source may affect public weights")
        fraction = source.get("maximum_stage_fraction")
        weight = source.get("sampling_weight")
        if not _finite_range(fraction, 0.0, 1.0) or not _finite_range(weight, 0.0, 1.0):
            raise Phase10RValidationError("source fraction or sampling weight is invalid")
    if declared != approved:
        raise Phase10RValidationError("dataset mixture differs from approved registry artifacts")
    inactive = mixture.get("inactive_source_policy")
    if not isinstance(inactive, dict) or inactive.get("sampling_weight") != 0.0:
        raise Phase10RValidationError("inactive sources must have zero weight")


def _validate_targets(targets: Mapping[str, Any]) -> None:
    if targets.get("canonical_perspective") != "current_side_to_move":
        raise Phase10RValidationError("target perspective changed")
    runtime = targets.get("runtime_score")
    codec = targets.get("move_codec")
    scale = targets.get("scale_contract")
    lanes = targets.get("source_lanes")
    if not all(isinstance(value, dict) for value in (runtime, codec, scale, lanes)):
        raise Phase10RValidationError("target semantics sections are missing")
    if runtime.get("handcrafted_contribution") is not False:
        raise Phase10RValidationError("pure runtime may not include handcrafted score")
    if runtime.get("mate_namespace_owned_by_search") is not True:
        raise Phase10RValidationError("search must retain the mate namespace")
    if codec.get("total_classes") != 13_689 or codec.get("legality_mask_required") is not True:
        raise Phase10RValidationError("move codec changed")
    if scale.get("fv_scale_is_model_output_scale") is not False:
        raise Phase10RValidationError("FV_SCALE is conflated with model output scale")
    packed = lanes.get("packed_sfen_value")
    if not isinstance(packed, dict) or packed.get("active") is not False:
        raise Phase10RValidationError("unapproved PackedSfenValue lane became active")


def _validate_models(matrix: Mapping[str, Any]) -> None:
    runtime = matrix.get("runtime")
    variants = matrix.get("variants")
    if not isinstance(runtime, dict) or not isinstance(variants, list) or len(variants) != 3:
        raise Phase10RValidationError("model matrix shape changed")
    if (
        runtime.get("pure_value_only") is not True
        or runtime.get("third_party_weights") is not False
    ):
        raise Phase10RValidationError("model runtime is not pure and independently initialized")
    browser_limit = runtime.get("browser_model_limit_bytes")
    for variant in variants:
        if not isinstance(variant, dict):
            raise Phase10RValidationError("model variant must be a mapping")
        parameters = variant.get("parameter_count")
        if not isinstance(parameters, int) or parameters <= 0:
            raise Phase10RValidationError("model parameter count is invalid")
        components = variant.get("components")
        if components is not None:
            if not isinstance(components, dict):
                raise Phase10RValidationError("model components are invalid")
            component_total = sum(
                component.get("parameters", -1)
                for component in components.values()
                if isinstance(component, dict)
            )
            if component_total != parameters:
                raise Phase10RValidationError(
                    f"{variant.get('variant_id')} component parameter total differs"
                )
        if variant.get("float32_size_bytes", browser_limit + 1) > browser_limit:
            raise Phase10RValidationError("model exceeds browser artifact limit")
        if variant.get("m5_training_memory_estimate_bytes", 2**64) >= 16 * 1024**3:
            raise Phase10RValidationError("model training estimate exceeds RSS target")


def _validate_curriculum(curriculum: Mapping[str, Any]) -> None:
    stages = curriculum.get("stages")
    scales = curriculum.get("data_scales")
    if not isinstance(stages, list) or [stage.get("order") for stage in stages] != list(
        range(1, 9)
    ):
        raise Phase10RValidationError("curriculum stages must be ordered one through eight")
    if (
        not isinstance(scales, list)
        or tuple(row.get("streamed_examples") for row in scales) != EXPECTED_SCALES
    ):
        raise Phase10RValidationError("progressive data scales changed")
    expansion = curriculum.get("expansion_gate")
    if (
        not isinstance(expansion, dict)
        or expansion.get("time_alone_is_stopping_condition") is not False
    ):
        raise Phase10RValidationError("time cannot be a stopping condition")


def _validate_arena(arena: Mapping[str, Any]) -> None:
    contract = arena.get("contract")
    gates = arena.get("gates")
    if not isinstance(contract, dict) or not isinstance(gates, list):
        raise Phase10RValidationError("Arena contract is missing")
    if (
        contract.get("opponent") != "handcrafted-experimental"
        or contract.get("equal_wall_clock") is not True
    ):
        raise Phase10RValidationError("Arena opponent or wall-clock contract changed")
    observed = {gate.get("gate_id"): gate.get("games") for gate in gates}
    if observed != EXPECTED_ARENA_GAMES:
        raise Phase10RValidationError("Arena ladder changed")
    final = next(gate for gate in gates if gate.get("gate_id") == "final_1600")
    requirements = final.get("requirements")
    if not isinstance(requirements, dict):
        raise Phase10RValidationError("final Arena requirements are missing")
    if requirements.get("score_minimum") != 0.55:
        raise Phase10RValidationError("final score threshold changed")
    for key in ("wilson_lower_strictly_above", "paired_bootstrap_lower_strictly_above"):
        if requirements.get(key) != 0.50:
            raise Phase10RValidationError("final confidence threshold changed")
    if (
        requirements.get("illegal_moves_maximum") != 0
        or requirements.get("unexplained_crashes_maximum") != 0
    ):
        raise Phase10RValidationError("final safety threshold changed")


def _validate_resources(resources: Mapping[str, Any]) -> None:
    workers = resources.get("workers")
    memory = resources.get("memory")
    disk = resources.get("disk")
    jobs = resources.get("jobs")
    if not all(isinstance(value, dict) for value in (workers, memory, disk, jobs)):
        raise Phase10RValidationError("resource sections are missing")
    if workers.get("maximum_concurrent_heavy") != 2:
        raise Phase10RValidationError("heavy worker limit changed")
    if memory.get("aggregate_rss_target_bytes") != 16 * 1024**3:
        raise Phase10RValidationError("RSS target changed")
    if disk.get("minimum_free_bytes") != 150 * 1024**3:
        raise Phase10RValidationError("disk floor changed")
    if jobs.get("wall_clock_deadline") is not None or jobs.get("polling_agent_loop") != "forbidden":
        raise Phase10RValidationError("long-job scheduling policy changed")


def _validate_holdout(holdout: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    reserved = holdout.get("reserved_artifacts")
    artifacts = registry.get("artifacts")
    expected = {
        item["artifact_id"]
        for item in artifacts
        if isinstance(item, dict) and item.get("state") in {"reserved_holdout", "denied"}
    }
    if not isinstance(reserved, list) or set(reserved) != expected:
        raise Phase10RValidationError("holdout reservation differs from source registry")
    inspection = holdout.get("inspection")
    if (
        not isinstance(inspection, dict)
        or inspection.get("final_result_may_select_or_retrain") is not False
    ):
        raise Phase10RValidationError("holdout result may not select or retrain")


def _finite_range(value: object, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def square_index(square: str) -> int:
    if len(square) != 2 or square[0] not in "123456789" or square[1] not in "abcdefghi":
        raise Phase10RValidationError(f"invalid USI square: {square}")
    return (ord(square[1]) - ord("a")) * 9 + (9 - int(square[0]))


def index_square(index: int) -> str:
    if not 0 <= index < 81:
        raise Phase10RValidationError("square index is outside 0..80")
    rank, column = divmod(index, 9)
    return f"{9 - column}{chr(ord('a') + rank)}"


def encode_move(move: str) -> int:
    """Encode one syntactically valid USI move into the frozen bijective space."""

    if not isinstance(move, str) or USI_MOVE.fullmatch(move) is None:
        raise Phase10RValidationError(f"invalid USI move: {move!r}")
    if "*" in move:
        return 13_122 + DROP_PIECES.index(move[0]) * 81 + square_index(move[2:4])
    promoted = move.endswith("+")
    return (square_index(move[:2]) * 81 + square_index(move[2:4])) * 2 + int(promoted)


def decode_move(index: int) -> str:
    """Decode one frozen policy class without inventing legality."""

    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 13_689:
        raise Phase10RValidationError("move index is outside 0..13688")
    if index >= 13_122:
        drop = index - 13_122
        piece, target = divmod(drop, 81)
        return f"{DROP_PIECES[piece]}*{index_square(target)}"
    pair, promoted = divmod(index, 2)
    origin, target = divmod(pair, 81)
    return f"{index_square(origin)}{index_square(target)}{'+' if promoted else ''}"


def allocate_source_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Apply configured weights by deterministic largest-remainder allocation."""

    if (
        total <= 0
        or not weights
        or any(not _finite_range(value, 0.0, 1.0) for value in weights.values())
    ):
        raise Phase10RValidationError("invalid weighted allocation")
    denominator = sum(weights.values())
    if denominator <= 0.0:
        raise Phase10RValidationError("source weights sum to zero")
    exact = {name: total * weight / denominator for name, weight in weights.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda name: (-(exact[name] - result[name]), name))
    for name in order[:remaining]:
        result[name] += 1
    return result


def assert_disjoint_splits(rows: Sequence[Mapping[str, str]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        identity, split = row.get("identity"), row.get("split")
        if not identity or split not in {"train", "validation", "final_holdout", "reserved"}:
            raise Phase10RValidationError("split row is invalid")
        previous = seen.setdefault(identity, split)
        if previous != split:
            raise Phase10RValidationError("validation/test identity entered another split")


def run_pipeline_sanity(root: Path) -> dict[str, Any]:
    """Run bounded semantic gates that must precede every training rung."""

    root = root.resolve()
    validate_phase10r(root, verify_hashes=False)
    checks: dict[str, bool] = {}
    checks["side_to_move_perspective"] = (
        _wdl("black_win", "black") == 2 and _wdl("black_win", "white") == 0
    )
    checks["score_sign"] = (
        _score_for_side(240, "black", "black") == 240
        and _score_for_side(240, "black", "white") == -240
    )
    checks["mate_encoding"] = _mate(-7) == (0, 7) and _mate(5) == (2, 5)
    targets = _load_yaml(root / "configs/phase10r/target-semantics.yaml")
    checks["fv_scale_isolation"] = (
        targets["scale_contract"]["fv_scale_is_model_output_scale"] is False
    )
    observed: set[int] = set()
    for index in range(13_689):
        decoded = decode_move(index)
        encoded = encode_move(decoded)
        if encoded != index or encoded in observed:
            raise Phase10RValidationError("move codec is not bijective")
        observed.add(encoded)
    checks["move_index_bijection"] = len(observed) == 13_689
    checks["promotion_and_drop_labels"] = (
        decode_move(encode_move("2b2a+")) == "2b2a+" and decode_move(encode_move("P*5e")) == "P*5e"
    )
    manifest = json.loads(
        (root / "artifacts/phase4/teacher/labels-v2/manifest.json").read_text(encoding="utf-8")
    )
    legality = manifest["binding"]["legality_validator"]
    checks["legal_label_evidence"] = (
        manifest["progress"]["completed"] == 10_000
        and legality["sha256"] == "a67f4097cb1f83d5e1fe13f82da3a3a5203366f3e268a558741eba8767b1fd61"
    )
    first_history = _history_signature(["start", "after-2g2f", "represented"], "represented")
    second_history = _history_signature(["start", "after-7g7f", "represented"], "represented")
    checks["history_matches_position"] = first_history != second_history
    checks["qsearch_pv_leaf_semantics"] = (
        targets["runtime_score"]["qsearch_semantics"]
        == "evaluate the represented stand-pat leaf; negate only through search recursion"
    )
    assert_disjoint_splits(
        [
            {"identity": "a", "split": "train"},
            {"identity": "b", "split": "validation"},
            {"identity": "c", "split": "final_holdout"},
        ]
    )
    checks["split_isolation"] = True
    checks["source_weights_applied"] = allocate_source_counts(
        10, {"aobazero": 1.0, "wcsc": 1.0, "denryu": 0.5}
    ) == {"aobazero": 4, "wcsc": 4, "denryu": 2}
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise Phase10RValidationError(f"pipeline sanity failed: {failed}")
    return {
        "schema": "open_shogiai_phase10r_pipeline_sanity/v1",
        "status": "passed",
        "checks": checks,
    }


def _wdl(outcome: str, side: str) -> int:
    if outcome == "draw":
        return 1
    winner = outcome.removesuffix("_win")
    return 2 if winner == side else 0


def _score_for_side(score: int, perspective: str, side: str) -> int:
    return score if perspective == side else -score


def _mate(value: int) -> tuple[int, int]:
    if value == 0:
        raise Phase10RValidationError("mate distance cannot be zero")
    return (2 if value > 0 else 0, abs(value))


def _history_signature(states: Sequence[str], represented: str) -> str:
    if not states or states[-1] != represented:
        raise Phase10RValidationError("history does not end at represented position")
    return hashlib.sha256("\0".join(states).encode()).hexdigest()


def run_micro_overfit() -> dict[str, Any]:
    """Intentionally memorize a 12-row multi-head dataset on CPU."""

    torch.manual_seed(20260729)
    torch.set_num_threads(1)
    features = torch.eye(12)
    score_target = torch.linspace(-1.0, 1.0, 12)
    wdl_target = torch.tensor([0, 1, 2] * 4)
    model = nn.Sequential(nn.Linear(12, 24), nn.ReLU())
    score_head = nn.Linear(24, 1)
    wdl_head = nn.Linear(24, 3)
    parameters = [*model.parameters(), *score_head.parameters(), *wdl_head.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.05)
    initial = None
    loss = torch.tensor(float("inf"))
    for _ in range(600):
        hidden = model(features)
        loss = torch.nn.functional.mse_loss(score_head(hidden).squeeze(1), score_target)
        loss = loss + torch.nn.functional.cross_entropy(wdl_head(hidden), wdl_target)
        if initial is None:
            initial = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = loss.item()
    if initial is None or not final < 0.001 or not final < initial / 1000:
        raise Phase10RValidationError(f"micro-overfit failed: initial={initial} final={final}")
    return {
        "schema": "open_shogiai_phase10r_micro_overfit/v1",
        "status": "passed",
        "rows": 12,
        "steps": 600,
        "initial_loss": initial,
        "final_loss": final,
    }


def memory_estimates(root: Path) -> dict[str, Any]:
    matrix = _load_yaml(root.resolve() / "configs/phase10r/model-matrix.yaml")
    limit = matrix["runtime"]["browser_model_limit_bytes"]
    rows = []
    for variant in matrix["variants"]:
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "parameters": variant["parameter_count"],
                "float32_bytes": variant["float32_size_bytes"],
                "int8_bytes": variant["int8_size_bytes"],
                "training_bytes": variant["m5_training_memory_estimate_bytes"],
                "browser_bytes": variant["browser_memory_estimate_bytes"],
                "artifact_fits_browser_limit": variant["float32_size_bytes"] <= limit,
                "training_fits_rss_target": variant["m5_training_memory_estimate_bytes"]
                < 16 * 1024**3,
            }
        )
    if not all(
        row["artifact_fits_browser_limit"] and row["training_fits_rss_target"] for row in rows
    ):
        raise Phase10RValidationError("one or more model memory estimates exceed the freeze")
    return {
        "schema": "open_shogiai_phase10r_memory_estimates/v1",
        "status": "passed",
        "variants": rows,
    }


def verify_frozen_hashes(root: Path) -> None:
    manifest = root / HASH_MANIFEST
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        digest, separator, name = line.partition("  ")
        if separator != "  " or not re.fullmatch(r"[0-9a-f]{64}", digest) or name in entries:
            raise Phase10RValidationError("frozen hash manifest is malformed")
        entries[name] = digest
    if set(entries) != FROZEN_PATHS:
        raise Phase10RValidationError("frozen hash path set changed")
    for name, expected in entries.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise Phase10RValidationError(f"frozen hash mismatch: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "sanity", "memory"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=Path("."))
    subparsers.add_parser("micro-overfit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = validate_phase10r(args.root)
    elif args.command == "sanity":
        result = run_pipeline_sanity(args.root)
    elif args.command == "memory":
        result = memory_estimates(args.root)
    else:
        result = run_micro_overfit()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
