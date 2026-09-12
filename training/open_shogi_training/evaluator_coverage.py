"""Finite recovery coverage gates, separate from corrupt-input exceptions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def coverage_report(output: Path, config: dict, manifest: dict | None = None) -> dict:
    from .defense_scenarios import assignment, focus_gate
    from .evaluator_data import atomic, digest, encoded

    policy = config["recovery_policy"]
    families = config["defense_campaign"]["families"]
    by_family = {
        family["id"]: {
            "group": family["group"],
            "split": family["split"],
            "completed": 0,
            "deferred": 0,
        }
        for family in families
    }
    for path in sorted((output / "games").glob("*.json.gz")):
        receipt_path = path.with_suffix(".receipt.json")
        if not receipt_path.exists():
            # An output before its receipt is not committed coverage.
            continue
        receipt = json.loads(receipt_path.read_text())
        if path.is_symlink() or digest(path) != receipt["sha256"]:
            raise ValueError("coverage source identity changed")
        raw = json.loads(gzip.decompress(path.read_bytes()))
        family, variant = assignment(config, raw["game"])
        if (
            raw["family"] != family["id"]
            or raw["split"] != family["split"]
            or raw["group"] != family["group"]
            or raw["variant"] != variant
            or receipt["game"] != raw["game"]
        ):
            raise ValueError("coverage source lineage/split changed")
        if raw.get("end") in ("Deferred", "deferred") or receipt.get("completed") is False:
            continue
        by_family[family["id"]]["completed"] += 1

    # Missing queue evidence cannot mean zero outstanding labels.
    queue_path = output / "recovery-queue.json"
    queue = json.loads(queue_path.read_text())
    if queue.get("schema") != "open_shogiai_recovery_queue/v1" or queue.get(
        "generation_sha256"
    ) != digest(output / "generation.json"):
        raise ValueError("recovery queue generation identity changed")
    deferred = queue["tasks"]
    strata = {"group": {}, "family": {}, "ply_stage": {}, "branch": {}}
    root_tasks = set()
    for task in deferred:
        if task["status"] not in ("hard", "deferred", "pending", "inflight"):
            raise ValueError("failed or unknown task cannot pass coverage")
        family, _ = assignment(config, task["game"])
        branch = task["branch"]
        ply = task["ply"]
        labels = {
            "group": family["group"],
            "family": family["id"],
            "ply_stage": "opening" if ply < 40 else "middle" if ply < 110 else "late",
            "branch": branch,
        }
        for category, value in labels.items():
            strata[category][value] = strata[category].get(value, 0) + 1
        if branch == "root":
            key = (task["game"], ply)
            if key not in root_tasks:
                root_tasks.add(key)
                by_family[family["id"]]["deferred"] += 1
    reasons = []
    for name, counts in by_family.items():
        if counts["completed"] < policy["minimum_completed_per_family"]:
            reasons.append(f"family_completed:{name}")
        if counts["deferred"] > policy["maximum_deferred_per_family"]:
            reasons.append(f"family_deferred:{name}")
    if len(root_tasks) > policy["maximum_deferred_roots"]:
        reasons.append("total_deferred_roots")
    focus = focus_gate(output, config)
    if not focus["passed"]:
        reasons.append("focus_total")
    for category in ("group", "ply_stage", "branch"):
        for name, counts in focus["by"][category].items():
            if counts["requested"] >= policy["minimum_focus_stratum_requests"] and (
                counts["completed"] / counts["requested"]
                < policy["minimum_focus_stratum_completion_rate"]
            ):
                reasons.append(f"focus_{category}:{name}")
    if manifest is not None:
        goal = policy["unique_data_goal"]
        if manifest["unique_positions"]["train"] < goal["minimum_train_positions"]:
            reasons.append("unique_train")
        if manifest["generated_unique_positions"]["train"] < goal["minimum_new_train_positions"]:
            reasons.append("unique_new_train")
        for split in ("train", "validation", "development_test"):
            minimum = (
                goal["minimum_train_rows_per_group"]
                if split == "train"
                else goal["minimum_evaluation_rows_per_group"]
            )
            for group in ("general", "opening", "defense", "attack_end"):
                if manifest["distributions"][split].get(group, 0) < minimum:
                    reasons.append(f"manifest_{split}:{group}")
    result = {
        "schema": "open_shogiai_recovery_coverage/v1",
        "passed": not reasons,
        "reasons": reasons,
        "by_family": by_family,
        "deferred_roots": len(root_tasks),
        "deferred_by": strata,
        "focus_quality": focus,
        "manifest_checked": manifest is not None,
    }
    atomic(output / "recovery-coverage.json", encoded(result))
    return result
