"""Fail-closed validation for the frozen Phase 10S supervised redesign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml

from open_shogi_training.data.registry import UniqueSafeLoader

CLASSIFICATION: Final = "SUPERVISED_REDESIGN_REQUIRED"
PROHIBITED_COUNTERS: Final = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)


class Phase10SValidationError(ValueError):
    """Raised when a frozen Phase 10S control or evidence item changes."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase10SValidationError(f"cannot load Phase 10S JSON: {path}") from error
    if not isinstance(value, dict):
        raise Phase10SValidationError(f"Phase 10S JSON root is not an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueSafeLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
        raise Phase10SValidationError(f"cannot load Phase 10S YAML: {path}") from error
    if not isinstance(value, dict):
        raise Phase10SValidationError(f"Phase 10S YAML root is not a mapping: {path}")
    return value


def _validate_redesign(value: Mapping[str, Any]) -> None:
    base = value.get("base")
    runtime = value.get("runtime")
    information = value.get("unique_information_contract")
    training = value.get("training")
    if not all(isinstance(item, Mapping) for item in (base, runtime, information, training)):
        raise Phase10SValidationError("Phase 10S redesign sections are incomplete")
    if (
        base.get("required_branch") != "codex/pure-learned-pre-selfplay"
        or base.get("required_ancestor_commit") != "b81efc487ee3911d1c3e66d5ba868c3d4fe08199"
        or base.get("prior_result") != CLASSIFICATION
    ):
        raise Phase10SValidationError("Phase 10S base boundary changed")
    if (
        runtime.get("profile") != "pure_learned"
        or runtime.get("prohibited_call_counters_must_equal") != 0
    ):
        raise Phase10SValidationError("Phase 10S pure-learned boundary changed")
    sources = value.get("sources")
    approved = sources.get("approved") if isinstance(sources, Mapping) else None
    if not isinstance(approved, list) or len(approved) != 4:
        raise Phase10SValidationError("Phase 10S approved source inventory changed")
    factual = [entry for entry in approved if entry.get("source_id") != "openshogiai_apery_teacher"]
    expected_rows = {"aobazero": 7825, "wcsc": 356214, "denryu": 11466}
    if {entry.get("source_id") for entry in factual} != set(expected_rows):
        raise Phase10SValidationError("Phase 10S factual sources changed")
    for entry in factual:
        source = str(entry["source_id"])
        if (
            entry.get("eligible_unique_train_rows") != expected_rows[source]
            or entry.get("maximum_exposures_per_supervised_generation") != 2
        ):
            raise Phase10SValidationError(f"Phase 10S repetition cap changed: {source}")
    if (
        information.get("eligible_unique_factual_rows") != 375505
        or information.get("maximum_factual_exposures_per_supervised_generation") != 751010
        or information.get("no_row_reuse_before_source_exhaustion") is not True
        or information.get("materialized_repeated_row_target") != "forbidden"
    ):
        raise Phase10SValidationError("Phase 10S unique-information contract changed")
    scheduler = training.get("scheduler")
    checkpointing = training.get("checkpointing")
    forgetting = training.get("catastrophic_forgetting_gate")
    if not all(isinstance(item, Mapping) for item in (scheduler, checkpointing, forgetting)):
        raise Phase10SValidationError("Phase 10S training controls are incomplete")
    if (
        training.get("schedule") != "joint_policy_wdl_multitask"
        or training.get("optimizer") != "AdamW"
        or training.get("learning_rate") != 0.0003
        or training.get("maximum_passes_per_supervised_generation") != 2
        or scheduler.get("type") != "cosine_with_linear_warmup"
        or checkpointing.get("training_batch_loss_may_select") is not False
        or forgetting.get("compare_before_and_after_every_value_or_teacher_stage") is not True
    ):
        raise Phase10SValidationError("Phase 10S optimization contract changed")


def _validate_model_matrix(value: Mapping[str, Any]) -> None:
    variants = value.get("variants")
    selection = value.get("selection")
    shared = value.get("shared")
    if (
        not isinstance(variants, list)
        or not isinstance(selection, Mapping)
        or not isinstance(shared, Mapping)
    ):
        raise Phase10SValidationError("Phase 10S model matrix is incomplete")
    if [variant.get("id") for variant in variants] != [
        "phase10s-c0-calibrated-wdl-log-odds",
        "phase10s-v1-calibrated-expected-score-logit",
    ]:
        raise Phase10SValidationError("Phase 10S model variants changed")
    if (
        selection.get("maximum_variants") != 2
        or selection.get("same_training_checkpoint_required") is not True
        or selection.get("offline_loss_alone_may_select") is not False
        or shared.get("handcrafted_score") != "forbidden"
        or shared.get("policy_may_directly_prune") is not False
    ):
        raise Phase10SValidationError("Phase 10S attribution boundary changed")


def _validate_ladder(value: Mapping[str, Any]) -> None:
    common = value.get("common")
    rungs = value.get("rungs")
    if not isinstance(common, Mapping) or not isinstance(rungs, list):
        raise Phase10SValidationError("Phase 10S diagnostic ladder is incomplete")
    if (
        common.get("pairedReversedColors") is not True
        or common.get("equalWallClock") is not True
        or common.get("holdoutInspection") is not False
        or common.get("promotion") is not False
    ):
        raise Phase10SValidationError("Phase 10S Arena boundary changed")
    expected = [(800, 400), (200, 100), (200, 100), (200, 100), (400, 200), (200, 100)]
    observed = [(rung.get("games"), rung.get("pairs")) for rung in rungs[:6]]
    if observed != expected or [rung.get("order") for rung in rungs] != list(range(1, 9)):
        raise Phase10SValidationError("Phase 10S diagnostic rung sizes changed")
    if (
        rungs[6].get("scoreMinimum") != 0.48
        or rungs[6].get("pairedBootstrapLowerMinimum") != 0.43
        or rungs[7].get("scoreMinimum") != 0.55
    ):
        raise Phase10SValidationError("Phase 10S strength gates changed")
    stop_before = value.get("stopBefore")
    if not isinstance(stop_before, list) or not {"final_promotion", "final_holdout_use"} <= set(
        stop_before
    ):
        raise Phase10SValidationError("Phase 10S final boundary changed")


def validate_phase10s(
    root: Path, *, verify_local_evidence: bool = True, require_branch: bool = False
) -> dict[str, Any]:
    raise Phase10SValidationError("Closed campaign; historical freeze is retired")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path("."))
    validate.add_argument("--static-only", action="store_true")
    validate.add_argument("--require-branch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raise SystemExit("Closed campaign; use current development commands")


if __name__ == "__main__":
    raise SystemExit(main())
